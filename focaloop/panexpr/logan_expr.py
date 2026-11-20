"""ExpressionLoop 0.4.0 Logan unitig streaming engine (operon-aware equivalence-class EM)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from ..logan import fetch_ena_run_lengths, is_insdc_run, logan_unitigs_url
from .catalog import Catalog, iter_features
from .manifest import StudyManifest
from .solver import DE_EXCLUDED_BIOTYPES, de_with_uncertainty, transcript_level_table
from .state import StateStore as PanStateStore, TARGET_COUNTS_SCHEMA, TRANSCRIPT_ABUNDANCE_SCHEMA

_DECOY_SENTINEL = "__FOCALOOP_DECOY__"
_COMPILE_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()


def _bin_cache_dir() -> Path:
    env_dir = os.environ.get("FOCALOOP_BIN_CACHE", "").strip()
    if env_dir:
        p = Path(env_dir)
    else:
        p = Path(tempfile.gettempdir()) / f"focaloop_bins_{os.getuid()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_c_binary(name: str, src_path: Path) -> Path:
    """Locate a prebuilt binary on PATH/package or compile it cleanly on demand."""
    pre = shutil.which(name)
    if pre:
        return Path(pre)
    pkg_bin = src_path.parent / name
    if pkg_bin.is_file() and os.access(pkg_bin, os.X_OK):
        return pkg_bin
    h = hashlib.sha256(src_path.read_bytes()).hexdigest()[:12]
    out_bin = _bin_cache_dir() / f"{name}_{h}"
    if out_bin.is_file() and os.access(out_bin, os.X_OK):
        return out_bin
    with _COMPILE_LOCK:
        if out_bin.is_file() and os.access(out_bin, os.X_OK):
            return out_bin
        cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        if not cc:
            raise RuntimeError(f"C compiler required to build {name} from {src_path}")
        tmp_out = out_bin.with_suffix(".tmp")
        flags = [cc, "-O3", str(src_path), "-o", str(tmp_out), "-lm"]
        if os.environ.get("FOCALOOP_PORTABLE_ISA") == "1" and os.uname().machine in ("x86_64", "AMD64"):
            flags[2:2] = ["-msse4.2", "-mpopcnt"]
        res = subprocess.run(flags, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to compile {name}: {res.stderr.strip()}")
        os.chmod(tmp_out, 0o755)
        tmp_out.replace(out_bin)
    return out_bin


def ensure_expr_logan_bins() -> tuple[Path, Path]:
    csrc_dir = Path(__file__).resolve().parent / "csrc"
    prep = _ensure_c_binary("prep_expr_unitigs", csrc_dir / "prep_expr_unitigs.c")
    ec = _ensure_c_binary("expr_sam_ec_v2", csrc_dir / "expr_sam_ec_v2.c")
    return prep, ec


def detect_antisense_genes(gtf_path: Path) -> set[str]:
    """Identify genes with opposite-strand exonic overlap (>=20 bp) in the annotation."""
    exons_by_chr: dict[str, list[tuple[int, int, str, str]]] = {}
    tx_to_gene: dict[str, str] = {}
    for feat in iter_features(gtf_path):
        if feat.ftype in ("transcript", "mRNA"):
            tid = feat.attrs.get("ID") or feat.attrs.get("transcript_id")
            gid = feat.attrs.get("Parent") or feat.attrs.get("gene_id")
            if tid and gid:
                tx_to_gene[tid] = gid.split(",")[0]
        elif feat.ftype == "exon":
            gid = feat.attrs.get("gene_id")
            if not gid:
                par = (feat.attrs.get("Parent") or "").split(",")[0]
                gid = tx_to_gene.get(par, par)
            if not gid:
                continue
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


def _select_unitig_aligner() -> tuple[str, str]:
    """Choose bwa-mem2 (preferred) with clean fallback to bwa or minimap2."""
    for name in ("bwa-mem2", "bwa", "minimap2"):
        p = shutil.which(name)
        if p:
            return name, p
    raise RuntimeError("Logan ExpressionLoop requires bwa-mem2, bwa, or minimap2 on PATH")


def _build_aligner_index(aligner_name: str, aligner_bin: str, tx_fa: Path, log: Callable[[str], None]) -> None:
    if aligner_name == "bwa-mem2":
        marker = Path(str(tx_fa) + ".bwt.2bit.64")
        if marker.is_file() and marker.stat().st_size > 0:
            return
        with _INDEX_LOCK:
            if not (marker.is_file() and marker.stat().st_size > 0):
                t0 = time.time()
                log(f"Building bwa-mem2 transcriptome index ({tx_fa.stat().st_size / 1e6:.1f} MB)...")
                subprocess.run([aligner_bin, "index", str(tx_fa)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log(f"bwa-mem2 index ready in {time.time() - t0:.1f}s")
    elif aligner_name == "bwa":
        marker = Path(str(tx_fa) + ".bwt")
        if marker.is_file() and marker.stat().st_size > 0:
            return
        with _INDEX_LOCK:
            if not (marker.is_file() and marker.stat().st_size > 0):
                t0 = time.time()
                log(f"Building bwa transcriptome index ({tx_fa.stat().st_size / 1e6:.1f} MB)...")
                subprocess.run([aligner_bin, "index", str(tx_fa)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log(f"bwa index ready in {time.time() - t0:.1f}s")


def _aligner_cmd(aligner_name: str, aligner_bin: str, tx_fa: Path, threads: int) -> list[str]:
    if aligner_name == "bwa-mem2":
        return [aligner_bin, "mem", "-k", "15", "-T", "22", "-a", "-t", str(max(1, threads)), str(tx_fa), "-"]
    if aligner_name == "bwa":
        return [aligner_bin, "mem", "-k", "15", "-T", "22", "-a", "-t", str(max(1, threads)), str(tx_fa), "-"]
    # minimap2 fallback (for dev laptops without bwa-mem2)
    return [aligner_bin, "-x", "sr", "-k", "15", "-w", "5", "-N", "20", "-a", "-t", str(max(1, threads)), str(tx_fa), "-"]


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
    prep_err = out_dir / f"{acc}.prep.err"
    aln_err = out_dir / f"{acc}.aln.err"
    ec_err = out_dir / f"{acc}.ec.err"

    def _run_once(a_name: str, a_bin: str) -> tuple[list[int], int]:
        wbytes = 0
        procs: list[subprocess.Popen] = []
        p_curl: subprocess.Popen | None = None
        p_decomp: subprocess.Popen | None = None

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

        with open(prep_err, "wb") as f_perr, open(aln_err, "wb") as f_aerr, open(ec_err, "wb") as f_eerr:
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
                # Use speed-based stall detection instead of hard --max-time or in-pipe --retry
                # so large archives finish cleanly and retries tear down/restart the full pipeline.
                p_curl = subprocess.Popen(
                    ["curl", "-4", "-sSL", "--fail", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120", url],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
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

            # Close parent references to intermediate pipes so downstream EOF propagates cleanly
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

    cur_aln_name, cur_aln_bin = aligner_name, aligner_bin
    rcs, wire_bytes = _run_once(cur_aln_name, cur_aln_bin)
    # Full-pipeline retry (up to 2x) on transient mid-stream S3/curl network failure or stall
    for retry_i in range(2):
        if local_unitigs_path is None and len(rcs) == 5 and rcs[0] in (6, 7, 18, 28, 35, 52, 56, 92):
            log(f"Transient Logan download error for {acc} (curl rc={rcs[0]}), retrying ({retry_i + 1}/2)...")
            time.sleep(2.0 * (retry_i + 1))
            rcs, wire_bytes = _run_once(cur_aln_name, cur_aln_bin)
        else:
            break
    # Fallback to classic bwa if bwa-mem2 exits on a negative signal
    aln_rc_idx = len(rcs) - 2
    if (
        any(rc != 0 for rc in rcs)
        and aln_rc_idx >= 0
        and rcs[aln_rc_idx] < 0
        and cur_aln_name == "bwa-mem2"
        and shutil.which("bwa")
    ):
        bwa_bin = shutil.which("bwa")
        assert bwa_bin is not None
        log(f"bwa-mem2 exited with signal {rcs[aln_rc_idx]} on {acc}, retrying with classic bwa...")
        _build_aligner_index("bwa", bwa_bin, tx_fa, log)
        cur_aln_name, cur_aln_bin = "bwa", bwa_bin
        rcs, wire_bytes = _run_once(cur_aln_name, cur_aln_bin)

    if any(rc != 0 for rc in rcs):
        aln_tail = " | ".join(aln_err.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-4:])
        raise RuntimeError(
            f"Logan unitig stream failed for {acc} (exit codes={rcs})"
            + (f" aln_err={aln_tail}" if aln_tail else "")
        )

    wall_s = time.time() - t0
    prep_m: dict[str, Any] = {}
    ec_m: dict[str, Any] = {}
    try:
        ptxt = prep_err.read_text(encoding="utf-8").strip().splitlines()
        if ptxt:
            prep_m = json.loads(ptxt[-1])
    except Exception:
        pass
    try:
        etxt = ec_err.read_text(encoding="utf-8").strip().splitlines()
        if etxt:
            ec_m = json.loads(etxt[-1])
    except Exception:
        pass
    return {
        "run_id": acc,
        "bp_tsv": str(out_bp),
        "wall_s": round(wall_s, 2),
        "wire_bytes": int(wire_bytes),
        "n_unitigs": int(prep_m.get("n_unitigs", 0)),
        "kept_unitigs": int(prep_m.get("kept_unitigs", 0)),
        "mapped_seg": int(ec_m.get("mapped_seg", 0)),
        "uniq_seg": int(ec_m.get("uniq_seg", 0)),
        "multi_seg": int(ec_m.get("multi_seg", 0)),
    }


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
    write_transcript_table: bool = True,
    log: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Quantify an entire RNA-seq cohort from Logan unitigs and run PyDESeq2 + state export."""
    t0 = time.time()
    prep_bin, ec_bin = ensure_expr_logan_bins()
    aligner_name, aligner_bin = _select_unitig_aligner()

    logan_work = work_dir / "logan_expr"
    logan_work.mkdir(parents=True, exist_ok=True)
    tx_fa = logan_work / "targets.fna"
    tx2gene_p = logan_work / "tx2gene.tsv"
    catalog.write_fasta(tx_fa)

    # Map every non-decoy target to its ortholog group_id; trap decoys under sentinel
    with open(tx2gene_p, "w", encoding="utf-8") as fh:
        fh.write("transcript_id\tgene_id\n")
        for tgt in catalog.targets.values():
            gid = _DECOY_SENTINEL if tgt.is_decoy else tgt.group_id
            fh.write(f"{tgt.target_id}\t{gid}\n")

    _build_aligner_index(aligner_name, aligner_bin, tx_fa, log)
    antisense_genes = detect_antisense_genes(gtf_path)

    rna_runs = manifest.rna_runs
    run_ids = [r.run_id for r in rna_runs]
    ena_meta = fetch_ena_run_lengths([r for r in run_ids if is_insdc_run(r)])

    # Resolve per-run kmers_per_frag from ENA or layout fallback
    run_kpf: dict[str, float] = {}
    est_raw_bytes = 0
    for r in rna_runs:
        acc_u = r.run_id.upper()
        em = ena_meta.get(acc_u)
        if em:
            run_kpf[r.run_id] = float(em["kmers_per_frag"])
            est_raw_bytes += int(em.get("fastq_bytes", 0))
        else:
            mates = 2 if r.layout == "PE" else 1
            run_kpf[r.run_id] = float(mates * 70.0)
            est_raw_bytes += int(r.total_bytes)

    n_workers = min(len(rna_runs), max(1, min(10, threads // 2 if threads >= 4 else 1)))
    aln_threads = max(1, threads // n_workers)
    runs_out_dir = logan_work / "runs"

    log(
        f"Streaming {len(rna_runs)} Logan unitig archives with {aligner_name} "
        f"({n_workers} worker(s) x {aln_threads} threads)..."
    )

    def _one(r):
        lpath = (local_unitig_map or {}).get(r.run_id)
        return stream_logan_expression_run(
            r.run_id,
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

    run_stats: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for st in ex.map(_one, rna_runs):
            run_stats.append(st)
    stats_by_run = {s["run_id"]: s for s in run_stats}

    # Build representative target per group_id for TARGET_COUNTS_SCHEMA + Focaloop Query compatibility
    rep_target: dict[str, Any] = {}
    for tgt in catalog.targets.values():
        if tgt.is_decoy:
            continue
        cur = rep_target.get(tgt.group_id)
        if cur is None or (tgt.assembly_id == catalog.primary_assembly and cur.assembly_id != catalog.primary_assembly):
            rep_target[tgt.group_id] = tgt
    group_ids = sorted(rep_target.keys())

    # Exclude rRNA/tRNA/structural RNAs from primary DE (matching Focaloop Query's gene_level_tables)
    # while still persisting all targets into target_counts.parquet for inspection.
    excluded_bt = set(DE_EXCLUDED_BIOTYPES)
    de_group_ids = [gid for gid in group_ids if str(rep_target[gid].biotype or "") not in excluded_bt]
    if not de_group_ids:
        de_group_ids = list(group_ids)
    de_group_set = set(de_group_ids)

    # Normalize sample metadata to unique sample index (matches run_pydeseq2 samples x genes layout)
    meta_idx = sample_meta_df.set_index("sample") if "sample" in sample_meta_df.columns else sample_meta_df.copy()
    meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]
    samples = list(meta_idx.index)
    counts_mat = pd.DataFrame(0.0, index=de_group_ids, columns=samples)
    avar_mat = pd.DataFrame(0.0, index=de_group_ids, columns=samples)
    tc_rows: list[dict[str, Any]] = []

    for r in rna_runs:
        kpf = run_kpf[r.run_id]
        bp_path = Path(stats_by_run[r.run_id]["bp_tsv"])
        df_bp = pd.read_csv(bp_path, sep="\t") if bp_path.is_file() else pd.DataFrame()
        bp_map: dict[str, tuple[float, float]] = {}
        if not df_bp.empty:
            for row in df_bp.itertuples(index=False):
                gid = str(row.gene)
                if gid == _DECOY_SENTINEL or gid not in rep_target:
                    continue
                tot_c = float(row.midpoint_bp) / kpf
                uniq_c = float(row.uniq_bp) / kpf
                bp_map[gid] = (tot_c, uniq_c)
                if gid in de_group_set and r.sample in counts_mat.columns:
                    counts_mat.at[gid, r.sample] += tot_c
                    ambig = max(0.0, tot_c - uniq_c)
                    avar_mat.at[gid, r.sample] += 0.25 * ambig
        # Clean up per-run temporary files immediately to keep scratch bounded at 1,000+ runs
        for ext in (".bp.tsv", ".prep.err", ".aln.err", ".ec.err"):
            (runs_out_dir / f"{r.run_id}{ext}").unlink(missing_ok=True)

        for gid in group_ids:
            tot_c, uniq_c = bp_map.get(gid, (0.0, 0.0))
            if tot_c <= 0.0:
                continue
            tgt = rep_target[gid]
            ambig = max(0.0, tot_c - uniq_c)
            tc_rows.append(
                {
                    "run_id": r.run_id,
                    "sample": r.sample,
                    "condition": r.condition,
                    "target_id": tgt.target_id,
                    "assembly_id": tgt.assembly_id,
                    "gene_id": tgt.gene_id,
                    "group_id": gid,
                    "symbol": tgt.symbol,
                    "biotype": tgt.biotype,
                    "length_bp": int(tgt.length),
                    "unique_sense": float(uniq_c),
                    "unique_antisense": 0.0,
                    "effective_count": float(tot_c),
                    "assignment_variance": float(0.25 * ambig),
                    "group_assignment_variance": float(0.25 * ambig),
                    "wgs_dna_reads": 0.0,
                    "dna_present_prob": 1.0,
                }
            )

    pan_store = PanStateStore(pan_state_dir)
    if tc_rows:
        tc_table = pa.Table.from_pylist(tc_rows, schema=TARGET_COUNTS_SCHEMA)
    else:
        tc_table = pa.Table.from_pydict({f.name: [] for f in TARGET_COUNTS_SCHEMA}, schema=TARGET_COUNTS_SCHEMA)
    del tc_rows
    pan_store.write("target_counts", tc_table, TARGET_COUNTS_SCHEMA)

    if write_transcript_table:
        tx_tbl = transcript_level_table(tc_table, catalog)
        pan_store.write("transcript_abundances", tx_tbl, TRANSCRIPT_ABUNDANCE_SCHEMA)
    del tc_table

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

    # Identify zero-peeking zoom candidates (antisense exons, borderline padj, low-count shifted genes)
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
        "n_antisense_genes": len(antisense_genes),
        "n_zoom_candidates": len(zoom_candidates),
    }
