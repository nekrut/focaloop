"""ExpressionLoop 0.4.0 Logan unitig streaming engine (operon-aware equivalence-class EM)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..logan import fetch_ena_run_lengths, is_insdc_run, logan_unitigs_url
from ._medvedev_compile import (
    ensure_user_native_cache_dir,
    prebuilt_native_dirs,
    verify_native_artifact,
)
from .catalog import Catalog, iter_features
from .manifest import StudyManifest
from .regime import available_ram_bytes
from .solver import DE_EXCLUDED_BIOTYPES, de_with_uncertainty
from .state import TARGET_COUNTS_SCHEMA

_DECOY_SENTINEL = "__FOCALOOP_DECOY__"
_COMPILE_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()
_CURL_HTTP_RE = re.compile(r"__CURL_HTTP_STATUS__:(\d{3})")


class LoganZeroMappingError(RuntimeError):
    """Raised when Logan unitig streaming yields 0 unitigs, 0 mapped segments, or cannot fit RAM."""


class _AlignerSignalError(RuntimeError):
    def __init__(self, acc: str, aligner_rc: int, message: str) -> None:
        super().__init__(message)
        self.acc = acc
        self.aligner_rc = aligner_rc


def _bin_cache_dir() -> Path:
    cache_dir, _ = ensure_user_native_cache_dir(
        explicit_build_dir=os.environ.get("FOCALOOP_BUILD_NATIVE_DIR"),
        explicit_cache_root=os.environ.get("FOCALOOP_BIN_CACHE"),
        subdir_stem="focaloop_bins",
    )
    return cache_dir


def _ensure_c_binary(name: str, src_path: Path) -> Path:
    """Locate a prebuilt binary in read-only image dirs / PATH / package, or compile into an 0700 cache (L98)."""
    h = hashlib.sha256(src_path.read_bytes()).hexdigest()[:12]
    for pre_dir in prebuilt_native_dirs():
        for cand_name in (f"{name}_{h}", name):
            cand = pre_dir / cand_name
            if cand.is_file() and os.access(cand, os.X_OK) and verify_native_artifact(cand, require_current_uid=False):
                return cand
    prefix_bin = Path(sys.prefix) / "bin" / name
    if prefix_bin.is_file() and os.access(prefix_bin, os.X_OK) and verify_native_artifact(prefix_bin, require_current_uid=False):
        return prefix_bin
    pre = shutil.which(name)
    if pre:
        pre_p = Path(pre)
        if verify_native_artifact(pre_p, require_current_uid=False):
            return pre_p
    pkg_bin = src_path.parent / name
    if pkg_bin.is_file() and os.access(pkg_bin, os.X_OK) and verify_native_artifact(pkg_bin, require_current_uid=False):
        return pkg_bin

    cache_dir = _bin_cache_dir()
    out_bin = cache_dir / f"{name}_{h}"
    if verify_native_artifact(out_bin, require_current_uid=True) and os.access(out_bin, os.X_OK):
        return out_bin
    with _COMPILE_LOCK:
        if verify_native_artifact(out_bin, require_current_uid=True) and os.access(out_bin, os.X_OK):
            return out_bin
        if out_bin.exists() or out_bin.is_symlink():
            out_bin.unlink(missing_ok=True)
        cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        if not cc:
            raise RuntimeError(f"C compiler required to build {name} from {src_path}")
        tmp_out = Path(f"{out_bin}.{os.getpid()}.{threading.get_ident()}.tmp")
        flags = [cc, "-O3", str(src_path), "-o", str(tmp_out), "-lm"]
        if os.environ.get("FOCALOOP_PORTABLE_ISA", "1") != "0" and os.uname().machine in ("x86_64", "AMD64"):
            flags[2:2] = ["-msse4.2", "-mpopcnt"]
        res = subprocess.run(flags, capture_output=True, text=True)
        if res.returncode != 0:
            tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to compile {name}: {res.stderr.strip()}")
        os.chmod(tmp_out, 0o700)
        os.replace(tmp_out, out_bin)
    return out_bin


def ensure_expr_logan_bins() -> tuple[Path, Path]:
    csrc_dir = Path(__file__).resolve().parent / "csrc"
    prep = _ensure_c_binary("prep_expr_unitigs", csrc_dir / "prep_expr_unitigs.c")
    ec = _ensure_c_binary("expr_sam_ec_v2", csrc_dir / "expr_sam_ec_v2.c")
    return prep, ec


def detect_antisense_genes(gtf_path: Path) -> set[str]:
    """Identify genes with opposite-strand exonic overlap (>=20 bp) in the annotation (#12)."""
    exons_by_chr: dict[str, list[tuple[int, int, str, str]]] = {}
    tx_to_genes: dict[str, list[str]] = {}
    for feat in iter_features(gtf_path):
        if feat.ftype in ("transcript", "mRNA"):
            tid = feat.attrs.get("ID") or feat.attrs.get("transcript_id")
            raw_gid = feat.attrs.get("Parent") or feat.attrs.get("gene_id") or ""
            if tid and raw_gid:
                tx_to_genes[tid] = [g.strip() for g in raw_gid.split(",") if g.strip()]
        elif feat.ftype == "exon":
            if feat.attrs.get("gene_id"):
                gids = [feat.attrs["gene_id"].strip()]
            else:
                parents = [p.strip() for p in (feat.attrs.get("Parent") or "").split(",") if p.strip()]
                gids = []
                for par in parents:
                    mapped = tx_to_genes.get(par)
                    if mapped:
                        for mg in mapped:
                            if mg not in gids:
                                gids.append(mg)
                    elif par not in gids:
                        gids.append(par)
            for gid in gids:
                if gid:
                    exons_by_chr.setdefault(feat.contig, []).append((feat.start, feat.end, feat.strand, gid))

    antisense: set[str] = set()
    for ivs in exons_by_chr.values():
        plus = sorted((s, e, g) for s, e, st, g in ivs if st == "+")
        minus = sorted((s, e, g) for s, e, st, g in ivs if st == "-")
        if not plus or not minus:
            continue
        j = 0
        for ps, pe, pg in plus:
            while j < len(minus) and minus[j][1] < ps:
                j += 1
            k = j
            while k < len(minus) and minus[k][0] <= pe:
                ms, me, mg = minus[k]
                if mg != pg and (min(pe, me) - max(ps, ms) + 1) >= 20:
                    antisense.add(pg)
                    antisense.add(mg)
                k += 1
    return antisense


def _predict_logan_index_ram(aligner_name: str, targets_bp: int) -> int:
    bp = max(0, int(targets_bp))
    if aligner_name == "bwa-mem2":
        return int(25.0 * bp + (256 << 20))
    if aligner_name == "bwa":
        return int(2.5 * bp + (128 << 20))
    return int(4.0 * bp + (128 << 20))


def _predict_logan_worker_ram(aligner_name: str, targets_bp: int) -> int:
    bp = max(0, int(targets_bp))
    if aligner_name == "bwa-mem2":
        return int(6.0 * bp + (128 << 20))
    if aligner_name == "bwa":
        return int(2.5 * bp + (96 << 20))
    return int(3.5 * bp + (96 << 20))


def _select_unitig_aligner(
    targets_bp: int = 0,
    ram_budget_bytes: int | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> tuple[str, str]:
    """Choose bwa-mem2 when RAM allows, falling back cleanly to classic bwa or minimap2 (#17)."""
    budget = int(ram_budget_bytes if ram_budget_bytes is not None else available_ram_bytes())
    installed: list[tuple[str, str]] = []
    for name in ("bwa-mem2", "bwa", "minimap2"):
        p = shutil.which(name)
        if p:
            installed.append((name, p))
    if not installed:
        raise RuntimeError("Logan ExpressionLoop requires bwa-mem2, bwa, or minimap2 on PATH")
    if targets_bp <= 0:
        return installed[0]
    max_idx_ram = int(0.80 * budget)
    skipped: list[str] = []
    for name, p in installed:
        need = _predict_logan_index_ram(name, targets_bp)
        if need <= max_idx_ram:
            if skipped:
                log(
                    f"Selected {name} for {targets_bp:,}-bp catalog within {budget / (1 << 30):.2f} GiB RAM budget "
                    f"(skipped {', '.join(skipped)})"
                )
            return name, p
        skipped.append(f"{name} (~{need / (1 << 30):.2f} GiB)")
    raise LoganZeroMappingError(
        f"Insufficient available RAM ({budget / (1 << 30):.2f} GiB) to index {targets_bp:,} bp Logan catalog "
        f"(predicted index RAM: {', '.join(skipped)})"
    )


def _build_aligner_index(aligner_name: str, aligner_bin: str, tx_fa: Path, log: Callable[[str], None]) -> None:
    if aligner_name == "bwa-mem2":
        marker = Path(str(tx_fa) + ".bwt.2bit.64")
        if marker.is_file() and marker.stat().st_size > 0:
            return
        with _INDEX_LOCK:
            if not (marker.is_file() and marker.stat().st_size > 0):
                t0 = time.time()
                log(f"Building bwa-mem2 transcriptome index ({tx_fa.stat().st_size / 1e6:.1f} MB)...")
                res = subprocess.run(
                    [aligner_bin, "index", str(tx_fa)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode != 0:
                    err_tail = " | ".join((res.stderr or "").strip().splitlines()[-4:])
                    raise RuntimeError(
                        f"bwa-mem2 index failed with exit code {res.returncode}"
                        + (f": {err_tail}" if err_tail else "")
                    )
                log(f"bwa-mem2 index ready in {time.time() - t0:.1f}s")
    elif aligner_name == "bwa":
        marker = Path(str(tx_fa) + ".bwt")
        if marker.is_file() and marker.stat().st_size > 0:
            return
        with _INDEX_LOCK:
            if not (marker.is_file() and marker.stat().st_size > 0):
                t0 = time.time()
                log(f"Building bwa transcriptome index ({tx_fa.stat().st_size / 1e6:.1f} MB)...")
                res = subprocess.run(
                    [aligner_bin, "index", str(tx_fa)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode != 0:
                    err_tail = " | ".join((res.stderr or "").strip().splitlines()[-4:])
                    raise RuntimeError(
                        f"bwa index failed with exit code {res.returncode}"
                        + (f": {err_tail}" if err_tail else "")
                    )
                log(f"bwa index ready in {time.time() - t0:.1f}s")


def _aligner_cmd(aligner_name: str, aligner_bin: str, tx_fa: Path, threads: int) -> list[str]:
    if aligner_name == "bwa-mem2":
        return [aligner_bin, "mem", "-k", "15", "-T", "22", "-a", "-t", str(max(1, threads)), str(tx_fa), "-"]
    if aligner_name == "bwa":
        return [aligner_bin, "mem", "-k", "15", "-T", "22", "-a", "-t", str(max(1, threads)), str(tx_fa), "-"]
    # minimap2 fallback: retain near-equal paralog/ortholog secondary alignments for EM (L16)
    return [
        aligner_bin, "-x", "sr", "-k", "15", "-w", "5",
        "--secondary=yes", "-p", "0.95", "-N", "20", "-a",
        "-t", str(max(1, threads)), str(tx_fa), "-",
    ]


def _parse_curl_status(curl_err_path: Path) -> tuple[int, str]:
    if not curl_err_path.is_file():
        return 0, ""
    txt = curl_err_path.read_text(encoding="utf-8", errors="replace").strip()
    m = _CURL_HTTP_RE.search(txt)
    code = int(m.group(1)) if m else 0
    cleaned = _CURL_HTTP_RE.sub("", txt).strip()
    tail = " | ".join(ln.strip() for ln in cleaned.splitlines() if ln.strip())[-500:]
    return code, tail


def stream_logan_expression_run(
    acc: str,
    *,
    tx_fa: Path,
    tx2gene: Path,
    out_dir: Path,
    prep_bin: Path,
    ec_bin: Path,
    aligner_name: str,
    aligner_bin: str,
    threads: int = 2,
    min_ka: float = 0.0,
    local_unitigs_path: Path | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Stream one run's Logan unitigs through prep_expr_unitigs -> aligner -> expr_sam_ec_v2."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bp = out_dir / f"{acc}.bp.tsv"
    curl_err = out_dir / f"{acc}.curl.err"
    prep_err = out_dir / f"{acc}.prep.err"
    aln_err = out_dir / f"{acc}.aln.err"
    ec_err = out_dir / f"{acc}.ec.err"

    def _run_once(a_name: str, a_bin: str) -> tuple[list[int], int]:
        wbytes = 0
        procs: list[subprocess.Popen] = []
        p_curl: subprocess.Popen | None = None
        p_decomp: subprocess.Popen | None = None
        curl_err.unlink(missing_ok=True)

        def _pipe_and_count(src_stream, dst_stream) -> None:
            nonlocal wbytes
            try:
                while True:
                    chunk = src_stream.read(1 << 18)
                    if not chunk:
                        break
                    wbytes += len(chunk)
                    dst_stream.write(chunk)
            except (BrokenPipeError, OSError):
                pass
            finally:
                for st in (dst_stream, src_stream):
                    try:
                        st.close()
                    except Exception:
                        pass

        with open(curl_err, "wb") as f_cerr, open(prep_err, "wb") as f_perr, open(aln_err, "wb") as f_aerr, open(ec_err, "wb") as f_eerr:
            if local_unitigs_path is not None:
                lp = Path(local_unitigs_path)
                wbytes = lp.stat().st_size
                with open(lp, "rb") as fh:
                    magic = fh.read(4)
                if magic == b"\x28\xb5\x2f\xfd":
                    p_decomp = subprocess.Popen(["zstd", "-dc", str(lp)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    prep_stdin = p_decomp.stdout
                    procs.append(p_decomp)
                elif magic[:2] == b"\x1f\x8b":
                    p_decomp = subprocess.Popen(["gzip", "-dc", str(lp)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    prep_stdin = p_decomp.stdout
                    procs.append(p_decomp)
                else:
                    prep_stdin = open(lp, "rb")
                feeder = None
            else:
                url = logan_unitigs_url(acc)
                p_curl = subprocess.Popen(
                    [
                        "curl", "-4", "-sSL", "--fail",
                        "--connect-timeout", "30",
                        "--speed-limit", "1024",
                        "--speed-time", "120",
                        "-w", "%{stderr}\n__CURL_HTTP_STATUS__:%{http_code}\n",
                        url,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=f_cerr,
                )
                p_decomp = subprocess.Popen(["zstd", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                procs.extend([p_curl, p_decomp])
                feeder = threading.Thread(target=_pipe_and_count, args=(p_curl.stdout, p_decomp.stdin), daemon=True)
                feeder.start()
                prep_stdin = p_decomp.stdout

            p_prep = subprocess.Popen(
                [str(prep_bin), str(min_ka)],
                stdin=prep_stdin,
                stdout=subprocess.PIPE,
                stderr=f_perr,
            )
            p_aln = subprocess.Popen(
                _aligner_cmd(a_name, a_bin, tx_fa, threads),
                stdin=p_prep.stdout,
                stdout=subprocess.PIPE,
                stderr=f_aerr,
            )
            p_ec = subprocess.Popen(
                [str(ec_bin), str(tx2gene), str(out_bp)],
                stdin=p_aln.stdout,
                stdout=subprocess.DEVNULL,
                stderr=f_eerr,
            )
            procs.extend([p_prep, p_aln, p_ec])

            if p_decomp is not None and p_decomp.stdout is not None:
                p_decomp.stdout.close()
            elif local_unitigs_path is not None and hasattr(prep_stdin, "close"):
                prep_stdin.close()
            if p_prep.stdout is not None:
                p_prep.stdout.close()
            if p_aln.stdout is not None:
                p_aln.stdout.close()

            if feeder is not None:
                feeder.join()
            rcs_rev = []
            for proc in reversed(procs):
                if proc is p_curl and proc.poll() is None:
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                rcs_rev.append(proc.wait())
            return list(reversed(rcs_rev)), wbytes

    retry_base_s = float(os.environ.get("FOCALOOP_LOGAN_RETRY_BASE_S", "2.0"))
    rcs, wire_bytes = _run_once(aligner_name, aligner_bin)
    # Full-pipeline retry on transient S3/curl network errors AND HTTP 429/500/502/503/504 (L15)
    for retry_i in range(3):
        if local_unitigs_path is not None or len(rcs) != 5 or rcs[0] == 0:
            break
        curl_rc = rcs[0]
        http_code, _ctail = _parse_curl_status(curl_err)
        transient_net = curl_rc in (6, 7, 18, 28, 35, 52, 56, 92)
        transient_http = (curl_rc == 22 and http_code in (429, 500, 502, 503, 504))
        if transient_net or transient_http:
            wait_s = retry_base_s * (2 ** retry_i)
            log(
                f"Transient Logan download error for {acc} "
                f"(curl rc={curl_rc}, http={http_code or 'n/a'}), retrying in {wait_s:.1f}s ({retry_i + 1}/3)..."
            )
            if wait_s > 0:
                time.sleep(wait_s)
            rcs, wire_bytes = _run_once(aligner_name, aligner_bin)
        else:
            break

    aln_rc_idx = len(rcs) - 2
    if any(rc != 0 for rc in rcs):
        aln_tail = " | ".join(aln_err.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-4:]) if aln_err.is_file() else ""
        http_code, curl_tail = _parse_curl_status(curl_err)
        if aln_rc_idx >= 0 and rcs[aln_rc_idx] < 0:
            raise _AlignerSignalError(
                acc,
                rcs[aln_rc_idx],
                f"Logan aligner {aligner_name} terminated with signal {rcs[aln_rc_idx]} on {acc}"
                + (f" aln_err={aln_tail}" if aln_tail else ""),
            )
        extra_parts = []
        if http_code:
            extra_parts.append(f"http={http_code}")
        if curl_tail:
            extra_parts.append(f"curl_err={curl_tail}")
        if aln_tail:
            extra_parts.append(f"aln_err={aln_tail}")
        raise RuntimeError(
            f"Logan unitig stream failed for {acc} (exit codes={rcs})"
            + (f" ({', '.join(extra_parts)})" if extra_parts else "")
        )
    curl_err.unlink(missing_ok=True)

    wall_s = time.time() - t0
    try:
        ptxt = prep_err.read_text(encoding="utf-8").strip().splitlines()
        prep_m = json.loads(ptxt[-1]) if ptxt else None
    except Exception as exc:
        raise LoganZeroMappingError(f"Invalid prep_expr_unitigs telemetry JSON for {acc}: {exc}") from exc
    try:
        etxt = ec_err.read_text(encoding="utf-8").strip().splitlines()
        ec_m = json.loads(etxt[-1]) if etxt else None
    except Exception as exc:
        raise LoganZeroMappingError(f"Invalid expr_sam_ec_v2 telemetry JSON for {acc}: {exc}") from exc

    if not isinstance(prep_m, dict) or not isinstance(ec_m, dict):
        raise LoganZeroMappingError(f"Missing stage telemetry JSON for Logan run {acc}")

    n_unitigs = int(prep_m.get("n_unitigs", 0))
    kept_unitigs = int(prep_m.get("kept_unitigs", 0))
    mapped_seg = int(ec_m.get("mapped_seg", 0))
    uniq_seg = int(ec_m.get("uniq_seg", 0))
    multi_seg = int(ec_m.get("multi_seg", 0))

    if n_unitigs <= 0 or kept_unitigs <= 0:
        raise LoganZeroMappingError(
            f"Logan unitig archive for {acc} yielded 0 usable unitigs (n_unitigs={n_unitigs}, kept_unitigs={kept_unitigs})"
        )
    if mapped_seg <= 0:
        raise LoganZeroMappingError(
            f"Logan unitigs for {acc} produced 0 mapped segments against the reference catalog"
        )
    if not out_bp.is_file():
        raise LoganZeroMappingError(f"Logan equivalence-class output file missing for {acc}: {out_bp}")
    if "expressed_genes" in ec_m:
        exp_rows = int(ec_m["expressed_genes"])
        with open(out_bp, "rb") as fh_bp:
            n_lines = sum(1 for _ in fh_bp)
        data_rows = max(0, n_lines - 1)
        if data_rows != exp_rows:
            raise LoganZeroMappingError(
                f"Truncated Logan bp.tsv for {acc}: expected {exp_rows} expressed gene rows, saw {data_rows}"
            )

    return {
        "run_id": acc,
        "bp_tsv": str(out_bp),
        "wall_s": round(wall_s, 2),
        "wire_bytes": int(wire_bytes),
        "n_unitigs": n_unitigs,
        "kept_unitigs": kept_unitigs,
        "mapped_seg": mapped_seg,
        "uniq_seg": uniq_seg,
        "multi_seg": multi_seg,
    }


def _write_logan_gene_abundances_tsv(
    out_path: Path,
    counts_by_sample_group: dict[str, dict[str, float]],
    sample_cond: dict[str, str],
    samples: list[str],
    group_ids: list[str],
    rep_target: dict[str, Any],
) -> None:
    """Write honest gene-group TPM table without fabricated isoform_fraction_psi (#10)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "sample", "condition", "group_id", "gene_id", "symbol",
            "biotype", "length_bp", "effective_count", "gene_tpm",
        ])
        for sname in samples:
            s_counts = counts_by_sample_group.get(sname, {})
            rsum = 0.0
            for gid in group_ids:
                c = s_counts.get(gid, 0.0)
                if c > 0.0:
                    rsum += c / max(1.0, float(rep_target[gid].length))
            scale = (1e6 / rsum) if rsum > 0.0 else 0.0
            cond = sample_cond.get(sname, "")
            for gid in group_ids:
                tgt = rep_target[gid]
                c = float(s_counts.get(gid, 0.0))
                rate = (c / max(1.0, float(tgt.length))) if c > 0.0 else 0.0
                tpm = rate * scale
                writer.writerow([
                    sname,
                    cond,
                    gid,
                    tgt.gene_id,
                    tgt.symbol,
                    tgt.biotype,
                    int(tgt.length),
                    f"{c:.6g}",
                    f"{tpm:.6g}",
                ])


def run_logan_expression_cohort(
    *,
    catalog: Catalog,
    manifest: StudyManifest,
    sample_meta_df: pd.DataFrame,
    contrast_list: list[str],
    design: str,
    gtf_path: Path,
    work_dir: Path,
    pan_state_dir: Path,
    threads: int = 4,
    min_ka: float = 0.0,
    local_unitig_map: dict[str, Path] | None = None,
    write_transcript_table: bool = False,
    out_gene_abundances_tsv: Path | None = None,
    ram_budget_bytes: int | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Quantify an entire RNA-seq cohort from Logan unitigs and run PyDESeq2 + state export."""
    t0 = time.time()
    prep_bin, ec_bin = ensure_expr_logan_bins()
    targets_bp = sum(int(t.length) for t in catalog.targets.values())
    budget = int(ram_budget_bytes if ram_budget_bytes is not None else available_ram_bytes())
    aligner_name, aligner_bin = _select_unitig_aligner(targets_bp=targets_bp, ram_budget_bytes=budget, log=log)

    logan_work = work_dir / "logan_expr"
    logan_work.mkdir(parents=True, exist_ok=True)
    tx_fa = logan_work / "targets.fna"
    tx2gene_p = logan_work / "tx2gene.tsv"
    catalog.write_fasta(tx_fa)

    with open(tx2gene_p, "w", encoding="utf-8") as fh:
        fh.write("transcript_id\tgene_id\n")
        for tgt in catalog.targets.values():
            gid = _DECOY_SENTINEL if tgt.is_decoy else tgt.group_id
            fh.write(f"{tgt.target_id}\t{gid}\n")

    try:
        _build_aligner_index(aligner_name, aligner_bin, tx_fa, log)
    except RuntimeError as idx_exc:
        bwa_fallback = shutil.which("bwa")
        if (
            aligner_name == "bwa-mem2"
            and bwa_fallback
            and _predict_logan_index_ram("bwa", targets_bp) <= int(0.80 * budget)
        ):
            log(f"bwa-mem2 index failed ({idx_exc}), switching entire cohort to classic bwa...")
            aligner_name, aligner_bin = "bwa", bwa_fallback
            _build_aligner_index(aligner_name, aligner_bin, tx_fa, log)
        else:
            raise

    antisense_genes = detect_antisense_genes(gtf_path)

    rna_runs = manifest.rna_runs
    # Reuse ENA metadata already fetched by resolve_expression_manifest when present (#14)
    ena_meta: dict[str, dict[str, Any]] = {}
    missing_insdc: list[str] = []
    for r in rna_runs:
        pre_em = (r.extra or {}).get("_ena_meta") if r.extra else None
        if isinstance(pre_em, dict) and pre_em.get("kmers_per_frag"):
            ena_meta[r.run_id.upper()] = pre_em
        elif is_insdc_run(r.run_id) and r.run_id.upper() not in ena_meta:
            missing_insdc.append(r.run_id.upper())
    missing_insdc = list(dict.fromkeys(missing_insdc))
    if missing_insdc:
        fetched_em = fetch_ena_run_lengths(missing_insdc)
        ena_meta.update(fetched_em)
        for acc_m in missing_insdc:
            if acc_m not in ena_meta:
                single_em = fetch_ena_run_lengths([acc_m])
                ena_meta.update(single_em)

    run_kpf: dict[str, float] = {}
    kpf_fallback_runs: list[str] = []
    est_raw_bytes = 0
    for r in rna_runs:
        acc_u = r.run_id.upper()
        em = ena_meta.get(acc_u)
        kpf_val = float(em.get("kmers_per_frag", 0.0)) if em else 0.0
        if em and math.isfinite(kpf_val) and kpf_val >= 2.0:
            run_kpf[r.run_id] = kpf_val
            est_raw_bytes += int(em.get("fastq_bytes", 0)) or int(r.total_bytes)
        else:
            mates = 2 if r.layout == "PE" else 1
            run_kpf[r.run_id] = float(mates * 70.0)
            kpf_fallback_runs.append(r.run_id)
            est_raw_bytes += int(r.total_bytes)
    if kpf_fallback_runs:
        log(
            f"ENA read-length metadata unavailable for {len(kpf_fallback_runs)} run(s) "
            f"({', '.join(kpf_fallback_runs[:5])}): falling back to layout*70 k-mer scaling"
        )

    # Deduplicate by run_id so duplicate entries never race on {run_id}.bp.tsv (L2)
    unique_runs_map: dict[str, Any] = {}
    for r in rna_runs:
        unique_runs_map.setdefault(r.run_id, r)
    unique_runs = list(unique_runs_map.values())

    runs_out_dir = logan_work / "runs"
    runs_out_dir.mkdir(parents=True, exist_ok=True)
    stats_by_run: dict[str, dict[str, Any]] = {}

    try:
        worker_ram = _predict_logan_worker_ram(aligner_name, targets_bp)
        ram_cap_workers = max(1, int(0.75 * budget) // max(1, worker_ram))
        cpu_cap_workers = max(1, min(10, threads // 2 if threads >= 4 else 1))
        n_workers = max(1, min(len(unique_runs), ram_cap_workers, cpu_cap_workers))

        while True:
            pending = [r for r in unique_runs if r.run_id not in stats_by_run]
            if not pending:
                break
            cur_workers = max(1, min(n_workers, len(pending)))
            aln_threads = max(1, threads // cur_workers)
            log(
                f"Streaming {len(pending)} Logan unitig archive(s) with {aligner_name} "
                f"({cur_workers} worker(s) x {aln_threads} threads)..."
            )

            def _one(r_item):
                lpath = (local_unitig_map or {}).get(r_item.run_id)
                return stream_logan_expression_run(
                    r_item.run_id,
                    tx_fa=tx_fa,
                    tx2gene=tx2gene_p,
                    out_dir=runs_out_dir,
                    prep_bin=prep_bin,
                    ec_bin=ec_bin,
                    aligner_name=aligner_name,
                    aligner_bin=aligner_bin,
                    threads=aln_threads,
                    min_ka=min_ka,
                    local_unitigs_path=lpath,
                    log=log,
                )

            wave_signal_err: _AlignerSignalError | None = None
            with ThreadPoolExecutor(max_workers=cur_workers) as ex:
                futs = {ex.submit(_one, r_item): r_item for r_item in pending}
                for fut in futs:
                    try:
                        st = fut.result()
                        stats_by_run[st["run_id"]] = st
                    except _AlignerSignalError as sig_exc:
                        if wave_signal_err is None:
                            wave_signal_err = sig_exc
                    except Exception:
                        for other_f in futs:
                            other_f.cancel()
                        raise

            if wave_signal_err is not None:
                if wave_signal_err.aligner_rc == -9 and cur_workers > 1:
                    n_workers = max(1, cur_workers // 2)
                    log(
                        f"{aligner_name} killed by SIGKILL (-9, likely OOM) on {wave_signal_err.acc}, "
                        f"halving concurrency to {n_workers} worker(s) and retrying remaining runs..."
                    )
                    continue
                bwa_bin = shutil.which("bwa")
                if (
                    aligner_name == "bwa-mem2"
                    and bwa_bin
                    and _predict_logan_index_ram("bwa", targets_bp) <= int(0.80 * budget)
                ):
                    log(
                        f"bwa-mem2 exited with signal {wave_signal_err.aligner_rc} on {wave_signal_err.acc}, "
                        "switching entire cohort to classic bwa for uniform quantification..."
                    )
                    aligner_name, aligner_bin = "bwa", bwa_bin
                    _build_aligner_index(aligner_name, aligner_bin, tx_fa, log)
                    stats_by_run.clear()
                    worker_ram = _predict_logan_worker_ram(aligner_name, targets_bp)
                    ram_cap_workers = max(1, int(0.75 * budget) // max(1, worker_ram))
                    n_workers = max(1, min(len(unique_runs), ram_cap_workers, cpu_cap_workers))
                    continue
                raise RuntimeError(str(wave_signal_err)) from wave_signal_err

        run_stats = [stats_by_run[r.run_id] for r in rna_runs]

        # Build representative target per group_id for TARGET_COUNTS_SCHEMA + Focaloop Query compatibility
        rep_target: dict[str, Any] = {}
        for tgt in catalog.targets.values():
            if tgt.is_decoy:
                continue
            cur = rep_target.get(tgt.group_id)
            if cur is None or (tgt.assembly_id == catalog.primary_assembly and cur.assembly_id != catalog.primary_assembly):
                rep_target[tgt.group_id] = tgt
        group_ids = sorted(rep_target.keys())

        excluded_bt = set(DE_EXCLUDED_BIOTYPES)
        de_group_ids = [
            gid for gid in group_ids
            if not str(rep_target[gid].biotype or "").startswith("decoy")
            and str(rep_target[gid].biotype or "") not in excluded_bt
        ]
        if not de_group_ids:
            de_group_ids = list(group_ids)
        de_group_set = set(de_group_ids)

        meta_idx = sample_meta_df.set_index("sample") if "sample" in sample_meta_df.columns else sample_meta_df.copy()
        meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]
        samples = list(meta_idx.index)
        sample_cond = {
            str(s): str(meta_idx.loc[s, "condition"]) if "condition" in meta_idx.columns else ""
            for s in samples
        }
        counts_mat = pd.DataFrame(0.0, index=de_group_ids, columns=samples)
        avar_mat = pd.DataFrame(0.0, index=de_group_ids, columns=samples)
        need_gene_tsv = out_gene_abundances_tsv is not None
        counts_by_sample_group: dict[str, dict[str, float]] = {s: {} for s in samples} if need_gene_tsv else {}

        # Stream per-run Arrow batches directly into target_counts.parquet via ParquetWriter (#19)
        pan_state_dir.mkdir(parents=True, exist_ok=True)
        tc_final = pan_state_dir / "target_counts.parquet"
        tc_tmp = pan_state_dir / f".target_counts.parquet.{os.getpid()}.tmp"
        parsed_bp_cache: dict[str, list[tuple[str, float, float]]] = {}
        wrote_any_batch = False

        with pq.ParquetWriter(tc_tmp, TARGET_COUNTS_SCHEMA, compression="zstd") as writer:
            for r in rna_runs:
                kpf = run_kpf[r.run_id]
                raw_rows = parsed_bp_cache.get(r.run_id)
                if raw_rows is None:
                    bp_path = Path(stats_by_run[r.run_id]["bp_tsv"])
                    raw_rows = []
                    if bp_path.is_file():
                        with open(bp_path, "r", encoding="utf-8") as fh_bp:
                            reader = csv.DictReader(fh_bp, delimiter="\t")
                            for row in reader:
                                gid = str(row.get("gene") or "")
                                if not gid or gid == _DECOY_SENTINEL or gid not in rep_target:
                                    continue
                                raw_rows.append((gid, float(row["midpoint_bp"]), float(row["uniq_bp"])))
                    raw_rows.sort(key=lambda x: x[0])
                    parsed_bp_cache[r.run_id] = raw_rows

                m_rows = 0
                c_tids: list[str] = []
                c_asms: list[str] = []
                c_gids: list[str] = []
                c_grps: list[str] = []
                c_syms: list[str] = []
                c_bts: list[str] = []
                c_lens: list[int] = []
                c_us: list[float] = []
                c_eff: list[float] = []
                c_var: list[float] = []

                s_grp_acc = counts_by_sample_group.get(r.sample) if need_gene_tsv else None
                in_samples = r.sample in counts_mat.columns

                for gid, mid_bp, uniq_bp in raw_rows:
                    tot_c = mid_bp / kpf
                    if tot_c <= 0.0:
                        continue
                    uniq_c = uniq_bp / kpf
                    ambig_v = 0.25 * max(0.0, tot_c - uniq_c)
                    if gid in de_group_set and in_samples:
                        counts_mat.at[gid, r.sample] += tot_c
                        avar_mat.at[gid, r.sample] += ambig_v
                    if s_grp_acc is not None:
                        s_grp_acc[gid] = s_grp_acc.get(gid, 0.0) + tot_c
                    tgt = rep_target[gid]
                    c_tids.append(tgt.target_id)
                    c_asms.append(tgt.assembly_id)
                    c_gids.append(tgt.gene_id)
                    c_grps.append(gid)
                    c_syms.append(tgt.symbol)
                    c_bts.append(tgt.biotype)
                    c_lens.append(int(tgt.length))
                    c_us.append(float(uniq_c))
                    c_eff.append(float(tot_c))
                    c_var.append(float(ambig_v))
                    m_rows += 1

                if m_rows > 0:
                    wrote_any_batch = True
                    zeros_f = pa.array(np.zeros(m_rows, dtype=np.float64), pa.float64())
                    ones_f = pa.array(np.ones(m_rows, dtype=np.float64), pa.float64())
                    var_arr = pa.array(c_var, pa.float64())
                    batch_tbl = pa.table(
                        {
                            "run_id": pa.repeat(pa.scalar(r.run_id, pa.string()), m_rows),
                            "sample": pa.repeat(pa.scalar(r.sample, pa.string()), m_rows),
                            "condition": pa.repeat(pa.scalar(r.condition, pa.string()), m_rows),
                            "target_id": pa.array(c_tids, pa.string()),
                            "assembly_id": pa.array(c_asms, pa.string()),
                            "gene_id": pa.array(c_gids, pa.string()),
                            "group_id": pa.array(c_grps, pa.string()),
                            "symbol": pa.array(c_syms, pa.string()),
                            "biotype": pa.array(c_bts, pa.string()),
                            "length_bp": pa.array(c_lens, pa.int32()),
                            "unique_sense": pa.array(c_us, pa.float64()),
                            "unique_antisense": zeros_f,
                            "effective_count": pa.array(c_eff, pa.float64()),
                            "assignment_variance": var_arr,
                            "group_assignment_variance": var_arr,
                            "wgs_dna_reads": zeros_f,
                            "dna_present_prob": ones_f,
                        },
                        schema=TARGET_COUNTS_SCHEMA,
                    )
                    writer.write_table(batch_tbl)

            if not wrote_any_batch:
                writer.write_table(
                    pa.Table.from_pydict({f.name: [] for f in TARGET_COUNTS_SCHEMA}, schema=TARGET_COUNTS_SCHEMA)
                )
        os.replace(tc_tmp, tc_final)
        parsed_bp_cache.clear()

        if write_transcript_table or out_gene_abundances_tsv is not None:
            log(
                "Logan k=31 unitig mode quantifies gene groups rather than individual splice isoforms - "
                "omitting isoform_fraction_psi (#10)."
            )
        if out_gene_abundances_tsv is not None:
            _write_logan_gene_abundances_tsv(
                Path(out_gene_abundances_tsv),
                counts_by_sample_group,
                sample_cond,
                samples,
                group_ids,
                rep_target,
            )
        counts_by_sample_group.clear()
    finally:
        for rid_clean in unique_runs_map:
            for ext in (".bp.tsv", ".curl.err", ".prep.err", ".aln.err", ".ec.err"):
                (runs_out_dir / f"{rid_clean}{ext}").unlink(missing_ok=True)

    symbols = {gid: rep_target[gid].symbol for gid in de_group_ids}
    gene_ids = {gid: rep_target[gid].gene_id for gid in de_group_ids}
    dna_prob = pd.Series(1.0, index=de_group_ids)

    de_raw = de_with_uncertainty(
        counts_mat.T,
        avar_mat.T,
        meta_idx,
        contrast_list,
        design=design,
        dna_prob=dna_prob,
        symbols=symbols,
        gene_ids=gene_ids,
        iteration=1,
    )

    zoom_candidates: set[str] = set()
    if not de_raw.empty:
        bm = de_raw["base_mean"].fillna(0.0) if "base_mean" in de_raw.columns else pd.Series(0.0, index=de_raw.index)
        padj = de_raw["padj"] if "padj" in de_raw.columns else pd.Series(np.nan, index=de_raw.index)
        lfc = de_raw["log2fc"].fillna(0.0).abs() if "log2fc" in de_raw.columns else pd.Series(0.0, index=de_raw.index)
        gcol = de_raw["gene_id"].astype(str)
        is_as = gcol.isin(antisense_genes)
        is_border = (padj >= 0.001) & (padj <= 0.30)
        is_low = (bm > 0) & (bm < 50) & (lfc >= 0.5)
        zoom_candidates = set(gcol[is_as | is_border | is_low])

    logan_wire_bytes = sum(int(s["wire_bytes"]) for s in run_stats)
    total_archive_bytes = max(est_raw_bytes, logan_wire_bytes, 1)
    pct_archive = 100.0 * logan_wire_bytes / total_archive_bytes

    per_sample_rows = []
    for r in rna_runs:
        st = stats_by_run[r.run_id]
        em = ena_meta.get(r.run_id.upper(), {})
        raw_b = int(em.get("fastq_bytes", 0)) or int(r.total_bytes) or int(st["wire_bytes"])
        mapped_frags = int(round(float(counts_mat[r.sample].sum()))) if r.sample in counts_mat.columns else 0
        per_sample_rows.append(
            {
                "sample_id": r.sample,
                "condition": r.condition,
                "run_id": r.run_id,
                "layout": em.get("layout", r.layout),
                "archive_mb": raw_b / 1e6,
                "streamed_mb": st["wire_bytes"] / 1e6,
                "pct_streamed": 100.0 * st["wire_bytes"] / max(raw_b, 1),
                "mapped_reads": mapped_frags,
            }
        )

    wall_s = round(time.time() - t0, 2)
    wire_red = round(total_archive_bytes / max(logan_wire_bytes, 1), 2)
    log(
        f"Logan ExpressionLoop finished in {wall_s:.1f}s: "
        f"{logan_wire_bytes / 1e6:.2f} MB streamed ({wire_red:.1f}x wire reduction vs raw FASTQ)"
    )
    return {
        "de_raw": de_raw,
        "per_sample_rows": per_sample_rows,
        "aligner": f"logan+{aligner_name}",
        "logan_wire_bytes": logan_wire_bytes,
        "total_archive_bytes": total_archive_bytes,
        "pct_archive": round(pct_archive, 4),
        "wire_reduction_x": wire_red,
        "wall_seconds": wall_s,
        "run_stats": run_stats,
        "kpf_fallback_runs": kpf_fallback_runs,
        "n_antisense_genes": len(antisense_genes),
        "n_zoom_candidates": len(zoom_candidates),
    }
