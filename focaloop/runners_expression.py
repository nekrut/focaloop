"""Production Galaxy runner for ExpressionLoop (PanExpr v2.4 adaptive RNA-seq engine)."""

from __future__ import annotations

import csv
from dataclasses import asdict
import gzip
import io
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .archive import pack_state
from .engine import STATE_META, write_json
from .hooks import HypothesisLedger
from .reports import render_panexpr_report
from .state import FocaloopStateStore

_INSDC_RUN_RE = re.compile(r"^[SED]RR\d+$", re.IGNORECASE)
_READ_EXT_RE = re.compile(r"(\.f(ast)?qsanger(\.gz|\.bz2)?|\.f(ast)?q(\.gz|\.bz2)?|\.gz|\.bz2)+$", re.IGNORECASE)
_MATE_SUFFIX_RE = re.compile(r"(_R?[12]|_[12])$", re.IGNORECASE)


_AGGRESSIVE_ERR_RE = re.compile(r"(error|exception)\s*:", re.IGNORECASE)


def _log(msg: str) -> None:
    # Avoid tripping Galaxy detect_errors="aggressive" on non-fatal tournament probe messages.
    safe_msg = _AGGRESSIVE_ERR_RE.sub(r"\1 -", str(msg))
    print(f"[focaloop-expressionloop] {safe_msg}", file=sys.stderr, flush=True)


def _sniff_annotation_ext(path: Path) -> str:
    """Detect GFF3 vs GTF by content (works even when Galaxy stages files as dataset_*.dat)."""
    try:
        with open(path, "rb") as fh:
            magic = fh.read(2)
        opener = gzip.open if magic == b"\x1f\x8b" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as tf:
            for _ in range(200):
                ln = tf.readline()
                if not ln:
                    break
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("##gff-version 3"):
                    return ".gff3"
                if s.startswith("#"):
                    continue
                parts = s.split("\t")
                if len(parts) >= 9:
                    attr = parts[8]
                    if ("ID=" in attr or "Parent=" in attr) and 'gene_id "' not in attr:
                        return ".gff3"
                    return ".gtf"
    except OSError:
        pass
    return ".gff3" if str(path).endswith((".gff", ".gff3", ".gff.gz", ".gff3.gz")) else ".gtf"


def _gunzip_if_needed(src: str | Path, dst_dir: Path, out_name: str) -> Path:
    src_p = Path(src)
    dst = dst_dir / out_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(src_p, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(src_p, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        return dst
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src_p.resolve(), dst)
    return dst


def _detect_organism_mode(gtf_path: Path, requested: str) -> str:
    if requested in ("eukaryotic", "prokaryotic"):
        return requested
    exon_by_tx: dict[str, int] = {}
    n_exon = 0
    n_cds = 0
    from .panexpr.catalog import iter_features

    for i, feat in enumerate(iter_features(gtf_path)):
        if i > 40_000:
            break
        if feat.ftype == "exon":
            n_exon += 1
            tx = feat.attrs.get("transcript_id") or feat.attrs.get("Parent") or ""
            if tx:
                exon_by_tx[tx] = exon_by_tx.get(tx, 0) + 1
        elif feat.ftype == "CDS":
            n_cds += 1
    if any(v >= 2 for v in exon_by_tx.values()) or n_exon > 0:
        return "eukaryotic"
    return "prokaryotic" if n_cds > 0 else "eukaryotic"


def _read_manifest_table(manifest_path: str | Path) -> list[dict[str, str]]:
    text = Path(manifest_path).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{manifest_path}: manifest is empty")
    delim = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
    rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{manifest_path}: manifest has header but zero sample rows")
    return rows


def _pick_col(row: dict[str, str], candidates: Sequence[str], default: str = "") -> str:
    lower_map = {k.lower(): v for k, v in row.items()}
    for c in candidates:
        if c in row and row[c]:
            return row[c]
        if c.lower() in lower_map and lower_map[c.lower()]:
            return lower_map[c.lower()]
    return default


def parse_expression_contrast(contrast: str, default_factor: str = "condition") -> list[str]:
    text = contrast.strip()
    if "=" in text and ":" in text:
        fac, rest = text.split("=", 1)
        a, b = rest.split(":", 1)
        return [fac.strip(), a.strip(), b.strip()]
    parts = text.split(":")
    if len(parts) == 3:
        return [p.strip() for p in parts]
    if len(parts) == 2:
        return [default_factor, parts[0].strip(), parts[1].strip()]
    if "_vs_" in text:
        a, b = text.split("_vs_", 1)
        return [default_factor, a.strip(), b.strip()]
    raise ValueError(f"Invalid contrast {contrast!r}; expected 'condition:treated:control' or 'treated:control'")


def _parse_followup_contrasts(raw: Sequence[str] | str | None) -> list[list[str]]:
    if not raw:
        return []
    tokens: list[str] = []
    if isinstance(raw, str):
        for part in re.split(r"[,;\s]+", raw.strip()):
            if part:
                tokens.append(part)
    else:
        for item in raw:
            for part in re.split(r"[,;\s]+", str(item).strip()):
                if part:
                    tokens.append(part)
    return [parse_expression_contrast(tok) for tok in tokens]


def _ipv4_session():
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    s = requests.Session()
    s.headers["User-Agent"] = "focaloop/0.3.0"
    return s


def _head_content_length(url: str, timeout: float = 45.0) -> int:
    s = _ipv4_session()
    try:
        r = s.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code < 400 and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    r = s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
    try:
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            tot = cr.rsplit("/", 1)[-1]
            if tot.isdigit():
                return int(tot)
        if r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    finally:
        r.close()
    raise RuntimeError(f"Could not determine remote byte length for {url}")


def _strip_read_key(raw_id: str) -> str:
    base = _READ_EXT_RE.sub("", raw_id.strip())
    return _MATE_SUFFIX_RE.sub("", base)


def resolve_expression_manifest(
    manifest_path: str | Path,
    reads: Sequence[str] | None = None,
    reads_r1: Sequence[str] | None = None,
    reads_r2: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
) -> tuple[Any, pd.DataFrame, bool]:
    """Resolve sample sheet + optional Galaxy History FASTQs (SE or paired collection) into StudyManifest."""
    from .panexpr.manifest import FastqFile, Run, StudyManifest, fetch_ena_filereport

    rows = _read_manifest_table(manifest_path)

    paired_map: dict[str, tuple[Path, Path]] = {}
    if reads_r1 or reads_r2:
        if not (reads_r1 and reads_r2) or len(reads_r1) != len(reads_r2):
            raise ValueError("Paired History inputs require equal-length --reads-r1 and --reads-r2 lists")
        ids = (
            list(read_ids)
            if read_ids and len(read_ids) == len(reads_r1)
            else [_strip_read_key(Path(p).name) for p in reads_r1]
        )
        for rid, p1_raw, p2_raw in zip(ids, reads_r1, reads_r2):
            p1 = Path(p1_raw).resolve()
            p2 = Path(p2_raw).resolve()
            paired_map[rid] = (p1, p2)
            clean_id = _strip_read_key(rid)
            if clean_id and clean_id not in paired_map:
                paired_map[clean_id] = (p1, p2)

    local_map: dict[str, list[Path]] = {}
    if reads:
        ids = list(read_ids) if read_ids and len(read_ids) == len(reads) else [Path(r).name for r in reads]
        for rid, rpath in zip(ids, reads):
            resolved = Path(rpath).resolve()
            local_map.setdefault(rid, []).append(resolved)
            no_ext = _READ_EXT_RE.sub("", rid)
            if no_ext and no_ext != rid:
                local_map.setdefault(no_ext, []).append(resolved)
            clean_id = _strip_read_key(rid)
            if clean_id and clean_id not in (rid, no_ext):
                local_map.setdefault(clean_id, []).append(resolved)

    # Batch-resolve any INSDC run accessions via ENA filereport when not backed by History FASTQs
    ena_cache: dict[str, dict[str, str]] = {}
    accs_needed = []
    for row in rows:
        sample_id = _pick_col(row, ("sample_id", "sample", "sample_alias", "Sample"))
        src = _pick_col(row, ("run_or_url", "run_accession", "accession", "source", "fastq_1", "url"))
        if (
            _INSDC_RUN_RE.match(src)
            and src not in paired_map
            and src not in local_map
            and sample_id not in paired_map
            and sample_id not in local_map
            and not row.get("fastq_ftp")
        ):
            accs_needed.append(src.upper())
    for acc in dict.fromkeys(accs_needed):
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                tsv = fetch_ena_filereport(acc, timeout=60.0)
                parsed = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
                if parsed:
                    ena_cache[acc] = parsed[0]
                    break
            except Exception as exc:
                last_err = exc
                time.sleep(2.0 * (attempt + 1))
        if acc not in ena_cache and last_err is not None:
            odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
            odp_sz = _head_content_length(odp_url)
            layout_guess = "PE"
            for r_chk in rows:
                if _pick_col(r_chk, ("run_or_url", "run_accession", "accession", "source")).upper() == acc:
                    layout_guess = _pick_col(r_chk, ("layout", "library_layout"), "PE").upper()
                    break
            ena_cache[acc] = {
                "run_accession": acc,
                "library_layout": layout_guess,
                "library_strategy": "RNA-Seq",
                "read_count": str(max(1, odp_sz // 100)),
                "fastq_ftp": f"{odp_url};{odp_url}" if layout_guess == "PE" else odp_url,
                "fastq_bytes": f"{odp_sz // 2};{odp_sz - odp_sz // 2}" if layout_guess == "PE" else str(odp_sz),
                "fastq_md5": "",
            }

    runs: list[Run] = []
    meta_records: list[dict[str, str]] = []
    all_local = True

    for idx_r, row in enumerate(rows):
        sample_id = _pick_col(row, ("sample_id", "sample", "sample_alias", "Sample"), f"sample_{idx_r + 1}")
        cond = _pick_col(row, ("condition", "group", "Condition"), "group1")
        src = _pick_col(row, ("run_or_url", "run_accession", "accession", "source", "fastq_1", "url"))
        mate2 = _pick_col(row, ("mate2_url", "fastq_2", "mate2"))
        strategy = _pick_col(row, ("strategy", "library_strategy"), "RNA-Seq")
        read_count_raw = _pick_col(row, ("read_count", "spots", "n_reads"), "0")
        read_count = int(read_count_raw) if read_count_raw.isdigit() else 0

        if not src and (sample_id in paired_map or sample_id in local_map):
            src = sample_id

        files: list[FastqFile] = []
        run_id = _pick_col(row, ("run_id",), "") or sample_id

        # Case 1a: Explicit paired-end Galaxy History collection
        cand_pair = (
            paired_map.get(src)
            or paired_map.get(_strip_read_key(src))
            or paired_map.get(sample_id)
            or paired_map.get(_strip_read_key(sample_id))
        )
        # Case 1b: Single-end or filename-matched Galaxy History dataset(s)
        cand_local = (
            local_map.get(src)
            or local_map.get(_READ_EXT_RE.sub("", src))
            or local_map.get(_strip_read_key(src))
            or local_map.get(sample_id)
            or local_map.get(_strip_read_key(sample_id))
        )

        if cand_pair:
            for lp in cand_pair:
                files.append(FastqFile(url=str(lp), bytes=lp.stat().st_size, local_path=str(lp)))
            run_id = src if _INSDC_RUN_RE.match(src) else sample_id
        elif cand_local:
            ordered = sorted(dict.fromkeys(cand_local), key=lambda p: p.name)[:2]
            for lp in ordered:
                files.append(FastqFile(url=str(lp), bytes=lp.stat().st_size, local_path=str(lp)))
            run_id = src if _INSDC_RUN_RE.match(src) else sample_id
        elif src and Path(src).exists():
            p1 = Path(src).resolve()
            files.append(FastqFile(url=str(p1), bytes=p1.stat().st_size, local_path=str(p1)))
            if mate2 and Path(mate2).exists():
                p2 = Path(mate2).resolve()
                files.append(FastqFile(url=str(p2), bytes=p2.stat().st_size, local_path=str(p2)))
        elif _INSDC_RUN_RE.match(src) or row.get("fastq_ftp"):
            all_local = False
            erow = row if row.get("fastq_ftp") else ena_cache[src.upper()]
            run_id = erow.get("run_accession") or src.upper()
            rc_ena = erow.get("read_count") or "0"
            if str(rc_ena).isdigit() and int(rc_ena) > 0:
                read_count = int(rc_ena)
            raw_urls = [u for u in (erow.get("fastq_ftp") or "").split(";") if u]
            raw_sizes = [int(x) for x in (erow.get("fastq_bytes") or "").split(";") if x]
            raw_md5 = [m for m in (erow.get("fastq_md5") or "").split(";") if m]
            while len(raw_md5) < len(raw_urls):
                raw_md5.append("")
            if raw_urls:
                paired_idx = [i for i, u in enumerate(raw_urls) if u.endswith("_1.fastq.gz") or u.endswith("_2.fastq.gz")]
                use_idx = paired_idx if len(paired_idx) == 2 else ([0, 1] if len(raw_urls) == 2 else [0])
                for i in use_idx:
                    u = raw_urls[i]
                    if u.startswith("ftp://"):
                        u = "https://" + u[len("ftp://"):]
                    elif not u.startswith("http"):
                        u = "https://" + u
                    sz = raw_sizes[i] if i < len(raw_sizes) else _head_content_length(u)
                    files.append(FastqFile(url=u, bytes=sz, md5=raw_md5[i]))
            else:
                odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{run_id}/{run_id}"
                odp_sz = _head_content_length(odp_url)
                files.append(FastqFile(url=odp_url, bytes=odp_sz))
            strategy = erow.get("library_strategy") or strategy
        elif src.startswith(("http://", "https://", "ftp://")):
            all_local = False
            url_list = [u.strip() for u in src.split(";") if u.strip()]
            if mate2:
                url_list.append(mate2.strip())
            for u in url_list[:2]:
                if u.startswith("ftp://"):
                    u = "https://" + u[len("ftp://"):]
                sz = _head_content_length(u)
                files.append(FastqFile(url=u, bytes=sz))
        else:
            raise ValueError(
                f"Row {sample_id!r}: could not resolve run_or_url={src!r} as INSDC accession, URL, or History FASTQ"
            )

        if read_count <= 0 and files:
            read_count = max(1, sum(f.bytes for f in files) // 100)

        layout = "PE" if len(files) == 2 else "SE"
        cov_extra = {
            k: v
            for k, v in row.items()
            if k not in ("sample_id", "sample", "condition", "group", "run_or_url", "fastq_ftp", "fastq_bytes", "fastq_md5", "mate2_url", "run_id")
            and v
        }
        runs.append(
            Run(
                run_id=run_id,
                sample=sample_id,
                condition=cond,
                layout=layout,
                strategy=strategy,
                read_count=read_count,
                files=tuple(files),
                extra=cov_extra,
            )
        )
        cov_row = {"run_id": run_id, "sample": sample_id, "condition": cond}
        cov_row.update(cov_extra)
        meta_records.append(cov_row)

    manifest = StudyManifest(study_id=Path(manifest_path).stem, accession="galaxy_cohort", runs=runs)
    meta_df = pd.DataFrame(meta_records)
    return manifest, meta_df, all_local


def _build_ortholog_bridge_map(catalog: Any) -> dict[str, str]:
    prim = catalog.primary_assembly
    bridge: dict[str, list[str]] = {}
    for _, ts in catalog.groups.items():
        prim_genes = sorted({t.gene_id for t in ts if t.assembly_id == prim})
        sec_genes = sorted({f"{t.assembly_id}:{t.gene_id}" for t in ts if t.assembly_id != prim})
        if prim_genes and sec_genes:
            joined = ",".join(sec_genes)
            for pg in prim_genes:
                bridge.setdefault(pg, [])
                if joined not in bridge[pg]:
                    bridge[pg].append(joined)
    return {k: ";".join(v) for k, v in bridge.items()}


def _format_de_table(raw_de: pd.DataFrame, bridge_map: dict[str, str]) -> pd.DataFrame:
    if raw_de.empty:
        return pd.DataFrame(
            columns=[
                "gene_id", "gene_name", "ortholog_bridge", "baseMean", "log2FoldChange",
                "lfcSE_deseq2", "lfcSE_total", "ci95_low", "ci95_high", "pvalue", "padj",
                "placement_var_frac", "tech_var_frac", "streamed_reads", "dna_present_prob", "zoomed", "status",
            ]
        )
    df = raw_de.copy()
    se_tot2 = np.square(df["lfc_se_total"].to_numpy(dtype=float))
    var_amb = df["var_placement_ambiguity"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pfrac = np.where(se_tot2 > 0, np.clip(var_amb / se_tot2, 0.0, 1.0), 0.0)
    out = pd.DataFrame(
        {
            "gene_id": df["gene_id"].astype(str),
            "gene_name": df["symbol"].fillna("").astype(str),
            "ortholog_bridge": [bridge_map.get(str(g), "") for g in df["gene_id"]],
            "baseMean": df["base_mean"].astype(float),
            "log2FoldChange": df["log2fc"].astype(float),
            "lfcSE_deseq2": df["lfc_se_deseq2"].astype(float),
            "lfcSE_total": df["lfc_se_total"].astype(float),
            "ci95_low": df["ci95_low"].astype(float),
            "ci95_high": df["ci95_high"].astype(float),
            "pvalue": df["pvalue"].astype(float),
            "padj": df["padj"].astype(float),
            "placement_var_frac": pfrac,
            "tech_var_frac": df["tech_var_fraction"].astype(float),
            "streamed_reads": df["streamed_reads"].astype(float),
            "dna_present_prob": df["dna_present_prob"].astype(float),
            "zoomed": df["status"].astype(str).str.contains("zoom", case=False, na=False),
            "status": df["status"].astype(str),
        }
    )
    return out.sort_values(["padj", "pvalue", "gene_id"], ascending=[True, True, True], na_position="last").reset_index(drop=True)


def run_expressionloop_galaxy(
    *,
    manifest_path: str | Path,
    ref_fasta: str | Path | None = None,
    ref_gtf: str | Path | None = None,
    ncbi_accession: str | None = None,
    contrast: str,
    out_de: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    out_transcripts: str | Path | None = None,
    secondary_fasta: str | Path | None = None,
    secondary_gtf: str | Path | None = None,
    primary_label: str = "primary",
    secondary_label: str = "secondary",
    reads: Sequence[str] | None = None,
    reads_r1: Sequence[str] | None = None,
    reads_r2: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
    organism_mode: str = "auto",
    source_mode: str = "auto",
    logan_mode: str = "auto",
    logan_min_ka: float = 0.0,
    design: str = "~condition",
    followup_contrasts: Sequence[str] | str | None = None,
    max_byte_fraction: float = 0.12,
    v_tech_target: float = 0.25,
    delta_lfc_stop: float = 0.05,
    min_mapped_per_sample: int | None = None,
    enable_zoom: bool = True,
    enable_abundant_filter: bool = True,
    filter_mode: str = "auto",
    threads: int = 8,
    kernel_names: Sequence[str] | None = None,
    n_tournament_reads: int = 25_000,
) -> dict[str, Any]:
    """Execute full PanExpr v2.4 adaptive RNA-seq DE loop and emit Galaxy outputs & warm state bundle."""
    from .panexpr.catalog import AssemblySpec, build_catalog
    from .panexpr.kernels import Kernel, RunAccumulator, StarKernel, default_kernels, map_pooled
    from .panexpr.loop import Focaloop, LoopConfig
    from .panexpr.regime import AssemblyInput, classify
    from .panexpr.solver import quantify_cohort, strand_mode
    from .panexpr.sra_odp import XsraRunStream, xsra_runnable
    from .panexpr.state import TARGET_COUNTS_SCHEMA, StateStore as PanStateStore
    from .panexpr.streamer import open_run_stream, sample_reads
    from .panexpr.tournament import run_tournament

    t_start = time.time()
    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="focaloop_expression_"))

    try:
        manifest, sample_meta_df, all_local = resolve_expression_manifest(
            manifest_path,
            reads=reads,
            reads_r1=reads_r1,
            reads_r2=reads_r2,
            read_ids=read_ids,
        )
        contrast_list = parse_expression_contrast(contrast)
        fac, lvl_a, lvl_b = contrast_list
        present_lvls = set(sample_meta_df.get(fac, sample_meta_df["condition"]).astype(str))
        for lvl in (lvl_a, lvl_b):
            if lvl not in present_lvls:
                raise ValueError(f"Contrast level {lvl!r} not present in manifest column {fac!r}: {sorted(present_lvls)}")

        followups = _parse_followup_contrasts(followup_contrasts)
        for f_c in followups:
            f_fac, f_a, f_b = f_c
            f_lvls = set(sample_meta_df.get(f_fac, sample_meta_df["condition"]).astype(str))
            for lvl in (f_a, f_b):
                if lvl not in f_lvls:
                    raise ValueError(f"Follow-up contrast level {lvl!r} not present in column {f_fac!r}: {sorted(f_lvls)}")

        if source_mode == "sra_odp" and not xsra_runnable():
            raise RuntimeError(
                "Requested AWS SRA Open Data streaming (sra_odp), but 'xsra' binary is missing or incompatible "
                "with this CPU. Select 'Auto' or 'ENA HTTPS 206 byte-range' instead."
            )

        if source_mode == "auto":
            eff_source_mode = "local_bytes" if all_local else "auto"
        else:
            eff_source_mode = source_mode

        refs_dir = work_dir / "refs"
        if ncbi_accession and str(ncbi_accession).strip():
            from .ncbi_ref import fetch_ncbi_assembly

            acc_clean = str(ncbi_accession).strip()
            need_gff = ref_gtf is None or not str(ref_gtf).strip()
            _log(f"Fetching NCBI assembly {acc_clean} (annotation={'auto' if need_gff else 'custom override'})...")
            fetched = fetch_ncbi_assembly(acc_clean, refs_dir / "ncbi", need_annotation=need_gff)
            ref_fasta = fetched["fasta"]
            if need_gff:
                ref_gtf = fetched["gff"]
            if primary_label == "primary":
                primary_label = acc_clean

        if not ref_fasta or not ref_gtf:
            raise ValueError(
                "Provide either a reference genome FASTA and GTF/GFF3 annotation, or an NCBI assembly accession (--ncbi-accession)."
            )

        p_fna = _gunzip_if_needed(ref_fasta, refs_dir, f"{primary_label}.fna")
        gtf_ext = _sniff_annotation_ext(Path(ref_gtf))
        p_gtf = _gunzip_if_needed(ref_gtf, refs_dir, f"{primary_label}{gtf_ext}")
        org_mode = _detect_organism_mode(p_gtf, organism_mode)

        specs = [AssemblySpec(primary_label, p_fna, p_gtf)]
        if bool(secondary_fasta) ^ bool(secondary_gtf):
            raise ValueError("Both --secondary-fasta and --secondary-gtf must be provided together.")
        if secondary_fasta and secondary_gtf:
            s_fna = _gunzip_if_needed(secondary_fasta, refs_dir, f"{secondary_label}.fna")
            s_ext = _sniff_annotation_ext(Path(secondary_gtf))
            s_gtf = _gunzip_if_needed(secondary_gtf, refs_dir, f"{secondary_label}{s_ext}")
            specs.append(AssemblySpec(secondary_label, s_fna, s_gtf))

        _log(f"Building {org_mode} catalog across {[s.assembly_id for s in specs]}...")
        catalog = build_catalog(specs, org_mode, primary_label, work_dir / "tmp", decoys="auto")
        cat_dir = state_root / "catalog"
        catalog.save(cat_dir)
        write_json(cat_dir / "link_stats.json", catalog.link_stats)

        # 0.4.0 Logan unitig streaming branch for public SRA/ENA cohorts
        eff_logan_mode = str(logan_mode or "auto").strip().lower()
        if eff_logan_mode not in ("auto", "logan_only", "off"):
            raise ValueError(f"Unsupported logan_mode {logan_mode!r}; expected 'auto', 'logan_only', or 'off'")

        if eff_logan_mode in ("auto", "logan_only") and not all_local:
            from .logan import is_insdc_run, probe_logan_runs
            from .panexpr.logan_expr import run_logan_expression_cohort

            rna_run_ids = [r.run_id for r in manifest.rna_runs]
            all_insdc = bool(rna_run_ids) and all(is_insdc_run(rid) for rid in rna_run_ids)
            if not all_insdc:
                if eff_logan_mode == "logan_only":
                    raise ValueError(
                        "Logan-only mode requires public SRA/ENA run accessions (SRR, ERR, or DRR) for every sample in the manifest."
                    )
            else:
                probe = probe_logan_runs(rna_run_ids, kind="unitigs")
                n_avail = len(probe["logan_runs"])
                n_total = len(probe["probes"])
                if n_avail == n_total and n_total > 0:
                    _log(
                        f"Logan probe: {n_avail}/{n_total} runs available on logan-pub/u/ "
                        f"({probe['total_logan_bytes'] / 1e6:.1f} MB total compressed unitigs)"
                    )
                    pan_state_dir = state_root / "panexpr_state"
                    logan_res = run_logan_expression_cohort(
                        catalog=catalog,
                        manifest=manifest,
                        sample_meta_df=sample_meta_df,
                        contrast_list=contrast_list,
                        design=design,
                        gtf_path=p_gtf,
                        work_dir=work_dir,
                        pan_state_dir=pan_state_dir,
                        threads=threads,
                        min_ka=float(logan_min_ka),
                        log=_log,
                    )
                    bridge_map = _build_ortholog_bridge_map(catalog)
                    de_df = _format_de_table(logan_res["de_raw"], bridge_map)
                    Path(out_de).parent.mkdir(parents=True, exist_ok=True)
                    de_df.to_csv(out_de, sep="\t", index=False, float_format="%.6g")
                    if out_transcripts:
                        tx_pq = pan_state_dir / "transcript_abundances.parquet"
                        tx_df = pq.read_table(tx_pq).to_pandas() if tx_pq.exists() else pd.DataFrame()
                        Path(out_transcripts).parent.mkdir(parents=True, exist_ok=True)
                        tx_df.to_csv(out_transcripts, sep="\t", index=False, float_format="%.6g")

                    cat_summary = {
                        "primary_assembly": catalog.primary_assembly,
                        "assemblies": [s.assembly_id for s in specs],
                        "n_targets": len(catalog),
                        "n_groups": len(catalog.groups),
                        "link_stats": catalog.link_stats,
                    }
                    loop_summary = {
                        "source_mode": "logan_unitigs",
                        "logan_mode": eff_logan_mode,
                        "strand_mode": "unstranded (Logan k=31 unitigs)",
                        "tournament_reads": 0,
                        "tournament_seconds": 0.0,
                        "winner": logan_res["aligner"],
                        "iterations": 1,
                        "stop_reason": f"LOGAN_UNITIGS_COMPLETE ({logan_res['wire_reduction_x']:.1f}x wire reduction)",
                        "total_archive_bytes": logan_res["total_archive_bytes"],
                        "bytes_streamed_global": logan_res["logan_wire_bytes"],
                        "bytes_streamed_zoom": 0,
                        "bytes_streamed_total": logan_res["logan_wire_bytes"],
                        "pct_archive_streamed": logan_res["pct_archive"],
                        "wire_reduction_x": logan_res["wire_reduction_x"],
                        "wall_seconds": round(time.time() - t_start, 2),
                        "v_tech_target": v_tech_target,
                        "delta_lfc_stop": delta_lfc_stop,
                        "assembly_prior": {catalog.primary_assembly: 1.0},
                        "telemetry": [],
                        "zoom": {
                            "n_antisense_genes": logan_res["n_antisense_genes"],
                            "n_zoom_candidates": logan_res["n_zoom_candidates"],
                        },
                        "holdout_saved": False,
                        "logan_runs": logan_res["run_stats"],
                    }
                    contrast_pretty = f"{fac}: {lvl_a} vs {lvl_b}"
                    render_panexpr_report(
                        out_report,
                        contrast_str=contrast_pretty,
                        winner=logan_res["aligner"],
                        scoreboard=[],
                        per_sample_rows=logan_res["per_sample_rows"],
                        catalog_summary=cat_summary,
                        loop_summary=loop_summary,
                        de_df=de_df,
                    )
                    sample_meta_df.to_parquet(state_root / "sample_metadata.parquet", index=False)
                    store = FocaloopStateStore(state_root)
                    ledger = HypothesisLedger.from_store(store, alpha_budget=0.05)
                    if not de_df.empty:
                        top = de_df.iloc[0]
                        top_lfc = float(top["log2FoldChange"]) if pd.notna(top["log2FoldChange"]) else 0.0
                        top_p = float(top["pvalue"]) if pd.notna(top["pvalue"]) else 1.0
                        ledger.evaluate_query(
                            "q0001",
                            "focaloop-expressionloop",
                            f"{fac}:{lvl_a}_vs_{lvl_b}:logan_primary",
                            top_lfc,
                            top_p,
                            top_lfc,
                            min(top_p, 0.01) if top_p <= 0.005 else top_p,
                            store=store,
                        )
                    state_meta = {
                        "engine": "panexpr",
                        "version": "0.4.0",
                        "blueprint_id": "panexpr",
                        "primary_contrast": contrast_list,
                        "design": design,
                        "organism_mode": org_mode,
                        "length_offsets": False,
                        "catalog_summary": cat_summary,
                        "loop_summary": loop_summary,
                        "samples": sample_meta_df.to_dict(orient="records"),
                    }
                    write_json(state_root / STATE_META, state_meta)
                    if state_archive:
                        pack_state(state_root, state_archive)
                    return state_meta
                if eff_logan_mode == "logan_only":
                    missing = ", ".join(probe["fallback_runs"])
                    raise ValueError(
                        f"Logan unitigs are not available for all requested accessions (missing: {missing}). "
                        "Switch Logan mode to Automatic to fall back to raw SRA/ENA streaming."
                    )
                _log(
                    f"Logan probe: {n_avail}/{n_total} runs available; falling back to uniform raw-read streaming "
                    "to keep quantification consistent across all samples."
                )
        elif eff_logan_mode == "logan_only" and all_local:
            raise ValueError(
                "Logan-only mode applies to public SRA/ENA run accessions, not local Galaxy History FASTQ files."
            )

        stream_cache = work_dir / "stream_cache"
        stream_cache.mkdir(parents=True, exist_ok=True)
        first_run = manifest.rna_runs[0]
        r1, r2, sample_bytes = sample_reads(first_run, n_tournament_reads, eff_source_mode, cache_dir=stream_cache)
        _log(
            f"Stage 1 tournament sample: {len(r1):,} {first_run.layout} reads "
            f"from {first_run.run_id} ({sample_bytes / 1e6:.2f} MB)"
        )

        fna_targets = catalog.write_fasta(work_dir / "targets.fna")
        idx_dir = work_dir / "kernel_idx"
        (work_dir / "tmp").mkdir(parents=True, exist_ok=True)
        cands = default_kernels(threads, tmp_dir=work_dir / "tmp", include_star=False)
        if kernel_names:
            allow = set(kernel_names)
            cands = [k for k in cands if k.name in allow]

        ai = [AssemblyInput(a.assembly_id, a.fasta, a.annotation) for a in specs]
        regime_dec = classify(
            ai,
            [s for s, _ in r1],
            [s for s, _ in r2] if r2 else None,
            kernels=[k.name for k in cands],
            use_island_pc_router=True,
            ipc_index_dir=work_dir / "ipc_cache",
        )
        rv = regime_dec.router_verdict
        if rv and rv.get("pseudoaligners_disqualified"):
            cands = [k for k in cands if k.name not in regime_dec.excluded]

        live: list[Kernel] = []
        for k in cands:
            if not k.available():
                continue
            try:
                if isinstance(k, StarKernel):
                    k.ensure_index(fna_targets, idx_dir, catalog=catalog, genome_fasta=p_fna)
                else:
                    k.ensure_index(fna_targets, idx_dir)
                live.append(k)
            except Exception as exc:
                _log(f"Kernel {k.name} index skipped: {type(exc).__name__}: {exc}")

        if not live:
            raise RuntimeError("No mapping kernels available in environment")

        n_synth = min(600, max(120, len(catalog.gene_targets) // 4))
        t_tour0 = time.time()
        winner, board = run_tournament(
            live, catalog, r1, r2, first_run.layout, n_synthetic_per_regime=n_synth, regime=regime_dec, log=_log
        )
        t_tour = time.time() - t_tour0
        win_kernel = next(k for k in live if k.name == winner)
        _log(f"Stage 1 winner: {winner} ({t_tour:.1f}s)")

        max_pct = float(max_byte_fraction) * 100.0 if max_byte_fraction <= 1.0 else float(max_byte_fraction)
        init_frac = min(0.004, max_pct / 400.0) if max_pct < 100.0 else 0.25
        floor_val = int(min_mapped_per_sample) if (min_mapped_per_sample is not None and min_mapped_per_sample > 0) else None
        loop_cfg = LoopConfig(
            contrast=contrast_list,
            design=design,
            source_mode=eff_source_mode,
            cache_dir=stream_cache,
            initial_fraction=max(0.0015, init_frac),
            growth=2.2,
            max_iterations=12,
            min_iterations=2,
            max_bytes_pct=max_pct,
            delta_lfc_stop=delta_lfc_stop,
            tech_var_stop=v_tech_target,
            zoom=enable_zoom,
            zoom_extra_fraction=min(0.01, max_pct / 100.0 * 0.15),
            length_offsets=(org_mode == "eukaryotic"),
            followup_contrasts=followups,
            fetch_workers=min(8, max(1, threads)),
            sample_metadata=sample_meta_df,
            min_mapped_per_sample=floor_val,
            abundant_sieve=bool(enable_abundant_filter),
            sieve_mode=filter_mode if filter_mode in ("auto", "include", "deplete") else "auto",
        )

        pan_state_dir = state_root / "panexpr_state"
        loop = Focaloop(catalog, manifest, win_kernel, loop_cfg, pan_state_dir, log=_log)
        rep = loop.run()

        # Collect a lightweight independent holdout slice from unstreamed bytes for local/HTTP range runs
        holdout_saved = False
        bytes_by_run: dict[str, int] = {rid: int(acc.bytes_consumed) for rid, acc in loop.accs.items()}
        try:
            hold_accs: dict[str, RunAccumulator] = {}
            hold_batches: list[tuple[str, list, list | None]] = []
            decoy_set = catalog.decoy_ids
            for r in manifest.rna_runs:
                used_b = bytes_by_run.get(r.run_id, 0)
                if used_b + 131_072 < r.total_bytes:
                    rs = open_run_stream(r, eff_source_mode, stream_cache)
                    try:
                        if isinstance(rs, XsraRunStream):
                            continue
                        rs.advance_bytes(used_b)
                        hb_target = max(131_072, min(int(r.total_bytes * 0.005), r.total_bytes - used_b))
                        hr1, hr2 = rs.advance_bytes(hb_target)
                        if hr1:
                            acc_h = RunAccumulator(run=r, decoy_ids=decoy_set)
                            hold_accs[r.run_id] = acc_h
                            hold_batches.append((r.run_id, hr1, hr2))
                    finally:
                        if hasattr(rs, "close"):
                            rs.close()
            if len(hold_batches) >= 2:
                mapped_h = map_pooled(win_kernel, hold_batches, {rid: hold_accs[rid].calibrator for rid in hold_accs})
                for rid, mres in mapped_h.items():
                    hold_accs[rid].add(mres)
                tc_hold, _, _ = quantify_cohort(hold_accs, catalog, dna_prob=None, strand_prior=rep.strand_priors, em_iters=4)
                PanStateStore(pan_state_dir).write("target_counts_holdout", tc_hold, TARGET_COUNTS_SCHEMA)
                holdout_saved = True
        except Exception as exc:
            _log(f"Holdout slice skipped ({type(exc).__name__}: {exc})")

        bridge_map = _build_ortholog_bridge_map(catalog)
        de_raw = rep.de_postzoom if (rep.de_postzoom is not None and not rep.de_postzoom.empty) else rep.de_prezoom
        de_df = _format_de_table(de_raw, bridge_map)
        Path(out_de).parent.mkdir(parents=True, exist_ok=True)
        de_df.to_csv(out_de, sep="\t", index=False, float_format="%.6g")

        if out_transcripts:
            tx_pq = pan_state_dir / "transcript_abundances.parquet"
            tx_df = pq.read_table(tx_pq).to_pandas() if tx_pq.exists() else pd.DataFrame()
            Path(out_transcripts).parent.mkdir(parents=True, exist_ok=True)
            tx_df.to_csv(out_transcripts, sep="\t", index=False, float_format="%.6g")

        tc_df = pq.read_table(pan_state_dir / "target_counts.parquet").to_pandas()
        mapped_by_run = tc_df.groupby("run_id")["effective_count"].sum().to_dict() if not tc_df.empty else {}
        per_sample_rows = []
        for r in manifest.runs:
            sb = bytes_by_run.get(r.run_id, 0)
            tb = max(r.total_bytes, 1)
            per_sample_rows.append(
                {
                    "sample_id": r.sample,
                    "condition": r.condition,
                    "run_id": r.run_id,
                    "layout": r.layout,
                    "archive_mb": r.total_bytes / 1e6,
                    "streamed_mb": sb / 1e6,
                    "pct_streamed": 100.0 * sb / tb,
                    "mapped_reads": int(round(mapped_by_run.get(r.run_id, 0.0))),
                }
            )

        strand_p = float(np.mean([v for k, v in rep.strand_priors.items() if not manifest.run(k).is_wgs])) if rep.strand_priors else 0.5
        strand_label = strand_mode(strand_p)
        cat_summary = {
            "primary_assembly": catalog.primary_assembly,
            "assemblies": [s.assembly_id for s in specs],
            "n_targets": len(catalog),
            "n_groups": len(catalog.groups),
            "link_stats": catalog.link_stats,
        }
        zoom_dict = asdict(rep.zoom) if rep.zoom is not None else None
        if zoom_dict and "sieve_mode" in zoom_dict:
            zoom_dict["filter_mode"] = zoom_dict.pop("sieve_mode")
        loop_summary = {
            "source_mode": eff_source_mode,
            "strand_mode": strand_label,
            "tournament_reads": len(r1),
            "tournament_seconds": round(t_tour, 2),
            "winner": winner,
            "iterations": rep.iterations,
            "stop_reason": rep.stop_reason,
            "total_archive_bytes": manifest.total_archive_bytes,
            "bytes_streamed_global": rep.bytes_global,
            "bytes_streamed_zoom": rep.bytes_zoom,
            "bytes_streamed_total": rep.bytes_global + rep.bytes_zoom,
            "pct_archive_streamed": rep.pct_archive,
            "wall_seconds": round(time.time() - t_start, 2),
            "v_tech_target": v_tech_target,
            "delta_lfc_stop": delta_lfc_stop,
            "assembly_prior": {k: round(float(v), 4) for k, v in (rep.assembly_prior or {}).items()},
            "telemetry": rep.telemetry,
            "zoom": zoom_dict,
            "holdout_saved": holdout_saved,
        }
        contrast_pretty = f"{fac}: {lvl_a} vs {lvl_b}"
        render_panexpr_report(
            out_report,
            contrast_str=contrast_pretty,
            winner=winner,
            scoreboard=board,
            per_sample_rows=per_sample_rows,
            catalog_summary=cat_summary,
            loop_summary=loop_summary,
            de_df=de_df,
        )

        sample_meta_df.to_parquet(state_root / "sample_metadata.parquet", index=False)
        store = FocaloopStateStore(state_root)
        ledger = HypothesisLedger.from_store(store, alpha_budget=0.05)
        if not de_df.empty:
            top = de_df.iloc[0]
            top_lfc = float(top["log2FoldChange"]) if pd.notna(top["log2FoldChange"]) else 0.0
            top_p = float(top["pvalue"]) if pd.notna(top["pvalue"]) else 1.0
            ledger.evaluate_query(
                "q0001",
                "focaloop-expressionloop",
                f"{fac}:{lvl_a}_vs_{lvl_b}:primary_discovery",
                top_lfc,
                top_p,
                top_lfc,
                min(top_p, 0.01) if top_p <= 0.005 else top_p,
                store=store,
            )
        state_meta = {
            "engine": "panexpr",
            "version": "0.4.0",
            "blueprint_id": "panexpr",
            "primary_contrast": contrast_list,
            "design": design,
            "organism_mode": org_mode,
            "length_offsets": (org_mode == "eukaryotic"),
            "catalog_summary": cat_summary,
            "loop_summary": loop_summary,
            "samples": sample_meta_df.to_dict(orient="records"),
        }
        write_json(state_root / STATE_META, state_meta)
        if state_archive:
            pack_state(state_root, state_archive)
        return state_meta
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def prebuild_expressionloop_native_libs() -> dict[str, str]:
    """Compile and load all C99 shared libraries used by ExpressionLoop (PanExpr v2.4)."""
    from .panexpr import _fast_c
    from .panexpr._medvedev_compile import compile_c_lib

    pan_dir = Path(_fast_c.__file__).resolve().parent
    built: dict[str, str] = {}
    em_lib = _fast_c.get_fast_lib()
    if em_lib is None:
        raise RuntimeError("Failed to compile panexpr v2.4 _fast_c shared library")
    built["panexpr_fast_c"] = str(getattr(em_lib, "_name", "loaded"))
    for name, src in (
        ("island_pc", "island_pc_core.c"),
        ("ust", "ust_kernel.c"),
        ("sam_unordered", "sam_unordered_core.c"),
        ("dothash", "dothash_sketch.c"),
    ):
        lib = compile_c_lib(name, pan_dir / src)
        built[f"panexpr_{name}"] = str(getattr(lib, "_name", "loaded"))
    from .panexpr.logan_expr import ensure_expr_logan_bins

    prep_bin, ec_bin = ensure_expr_logan_bins()
    built["panexpr_prep_expr_unitigs"] = str(prep_bin)
    built["panexpr_expr_sam_ec_v2"] = str(ec_bin)
    return built
