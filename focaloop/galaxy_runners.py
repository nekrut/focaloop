"""Production Galaxy runners for Focaloop PanExpr, Focaloop MalariaVar, and Warm Query."""

from __future__ import annotations

import csv
from dataclasses import asdict
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as sp_stats

from .archive import materialize_state, pack_state
from .engine import STATE_META, write_json
from .hooks import HypothesisLedger, is_holdout_block
from .reports import render_malariavar_report, render_panexpr_report, render_query_report
from .state import FocaloopStateStore

_INSDC_RUN_RE = re.compile(r"^[SED]RR\d+$", re.IGNORECASE)


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
    return src_p.resolve()


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


def _parse_panexpr_contrast(contrast: str, default_factor: str = "condition") -> list[str]:
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


def _ipv4_session():
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    s = requests.Session()
    s.headers["User-Agent"] = "focaloop/0.2.0"
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


def _resolve_panexpr_manifest(
    manifest_path: str | Path,
    reads: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
) -> tuple[Any, pd.DataFrame, bool]:
    """Resolve user manifest TSV + optional Galaxy History FASTQs into a PanExpr StudyManifest."""
    from .panexpr.manifest import FastqFile, Run, StudyManifest, fetch_ena_filereport

    rows = _read_manifest_table(manifest_path)
    local_map: dict[str, list[Path]] = {}
    if reads:
        ids = list(read_ids) if read_ids and len(read_ids) == len(reads) else [Path(r).stem for r in reads]
        for rid, rpath in zip(ids, reads):
            clean_id = re.sub(r"(_R?[12]|_[12])(\.f(ast)?q(\.gz)?)*$", "", rid)
            local_map.setdefault(rid, []).append(Path(rpath).resolve())
            if clean_id != rid:
                local_map.setdefault(clean_id, []).append(Path(rpath).resolve())

    # Batch-resolve any INSDC run accessions via ENA filereport
    ena_cache: dict[str, dict[str, str]] = {}
    accs_needed = []
    for row in rows:
        src = _pick_col(row, ("run_or_url", "run_accession", "accession", "source", "fastq_1", "url"))
        if _INSDC_RUN_RE.match(src) and src not in local_map and not row.get("fastq_ftp"):
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
            # Allow fallback to AWS ODP if xsra can stream it
            odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
            odp_sz = _head_content_length(odp_url)
            ena_cache[acc] = {
                "run_accession": acc,
                "library_layout": _pick_col(row, ("layout", "library_layout"), "PE"),
                "library_strategy": "RNA-Seq",
                "read_count": "0",
                "fastq_ftp": f"{odp_url};{odp_url}" if _pick_col(row, ("layout",), "PE") == "PE" else odp_url,
                "fastq_bytes": f"{odp_sz // 2};{odp_sz - odp_sz // 2}" if _pick_col(row, ("layout",), "PE") == "PE" else str(odp_sz),
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

        if not src and sample_id in local_map:
            src = sample_id

        files: list[FastqFile] = []
        run_id = sample_id

        # Case 1: Galaxy History dataset(s) or local file(s)
        cand_local = local_map.get(src) or local_map.get(sample_id)
        if cand_local:
            ordered = sorted(cand_local, key=lambda p: p.name)[:2]
            for lp in ordered:
                files.append(FastqFile(url=str(lp), bytes=lp.stat().st_size, local_path=str(lp)))
            run_id = src if _INSDC_RUN_RE.match(src) else sample_id
        elif src and Path(src).exists():
            p1 = Path(src).resolve()
            files.append(FastqFile(url=str(p1), bytes=p1.stat().st_size, local_path=str(p1)))
            if mate2 and Path(mate2).exists():
                p2 = Path(mate2).resolve()
                files.append(FastqFile(url=str(p2), bytes=p2.stat().st_size, local_path=str(p2)))
        # Case 2: ENA filereport columns inline OR resolved INSDC accession
        elif _INSDC_RUN_RE.match(src) or row.get("fastq_ftp"):
            all_local = False
            erow = row if row.get("fastq_ftp") else ena_cache[src.upper()]
            run_id = erow.get("run_accession") or src.upper()
            raw_urls = [u for u in (erow.get("fastq_ftp") or "").split(";") if u]
            raw_sizes = [int(x) for x in (erow.get("fastq_bytes") or "").split(";") if x]
            raw_md5 = [m for m in (erow.get("fastq_md5") or "").split(";") if m]
            while len(raw_md5) < len(raw_urls):
                raw_md5.append("")
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
            strategy = erow.get("library_strategy") or strategy
        # Case 3: Direct HTTP(S) FASTQ URL(s)
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
            raise ValueError(f"Row {sample_id!r}: could not resolve run_or_url={src!r} as accession, URL, or local file")

        layout = "PE" if len(files) == 2 else "SE"
        runs.append(
            Run(
                run_id=run_id,
                sample=sample_id,
                condition=cond,
                layout=layout,
                strategy=strategy,
                read_count=0,
                files=tuple(files),
            )
        )
        cov_row = {"run_id": run_id, "sample": sample_id, "condition": cond}
        for k, v in row.items():
            if k not in ("run_or_url", "fastq_ftp", "fastq_bytes", "fastq_md5", "mate2_url") and v:
                cov_row.setdefault(k, v)
        meta_records.append(cov_row)

    manifest = StudyManifest(study_id=Path(manifest_path).stem, accession="galaxy_cohort", runs=runs)
    meta_df = pd.DataFrame(meta_records)
    return manifest, meta_df, all_local


def _build_ortholog_bridge_map(catalog: Any) -> dict[str, str]:
    """Map primary gene_id -> secondary assembly gene IDs sharing the same sequence/RBH ortholog group."""
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


def _format_panexpr_de_table(raw_de: pd.DataFrame, bridge_map: dict[str, str]) -> pd.DataFrame:
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


def run_panexpr_galaxy(
    *,
    manifest_path: str | Path,
    ref_fasta: str | Path,
    ref_gtf: str | Path,
    contrast: str,
    out_de: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    secondary_fasta: str | Path | None = None,
    secondary_gtf: str | Path | None = None,
    primary_label: str = "primary",
    secondary_label: str = "secondary",
    reads: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
    organism_mode: str = "auto",
    source_mode: str = "auto",
    design: str = "~condition",
    max_byte_fraction: float = 0.12,
    v_tech_target: float = 0.25,
    delta_lfc_stop: float = 0.05,
    enable_zoom: bool = True,
    threads: int = 8,
    kernel_names: Sequence[str] | None = None,
    n_tournament_reads: int = 25_000,
) -> dict[str, Any]:
    """Execute full 5-stage PanExpr loop and emit Galaxy DE table, HTML report, and warm state bundle."""
    from .panexpr.catalog import AssemblySpec, build_catalog
    from .panexpr.kernels import Kernel, RunAccumulator, StarKernel, default_kernels, map_pooled
    from .panexpr.loop import Focaloop, LoopConfig
    from .panexpr.regime import AssemblyInput, classify
    from .panexpr.solver import quantify_cohort, strand_mode
    from .panexpr.state import TARGET_COUNTS_SCHEMA, StateStore as PanStateStore
    from .panexpr.streamer import open_run_stream, sample_reads
    from .panexpr.tournament import run_tournament

    t_start = time.time()
    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="focaloop_panexpr_"))

    def _log(msg: str) -> None:
        print(f"[focaloop-panexpr] {msg}", flush=True)

    try:
        manifest, sample_meta_df, all_local = _resolve_panexpr_manifest(manifest_path, reads=reads, read_ids=read_ids)
        contrast_list = _parse_panexpr_contrast(contrast)
        fac, lvl_a, lvl_b = contrast_list
        present_lvls = set(sample_meta_df.get(fac, sample_meta_df["condition"]).astype(str))
        for lvl in (lvl_a, lvl_b):
            if lvl not in present_lvls:
                raise ValueError(f"Contrast level {lvl!r} not present in manifest column {fac!r}: {sorted(present_lvls)}")

        if source_mode == "auto":
            eff_source_mode = "local_bytes" if all_local else "http_range"
        else:
            eff_source_mode = source_mode

        refs_dir = work_dir / "refs"
        p_fna = _gunzip_if_needed(ref_fasta, refs_dir, f"{primary_label}.fna")
        gtf_ext = ".gff3" if str(ref_gtf).endswith((".gff", ".gff3", ".gff.gz", ".gff3.gz")) else ".gtf"
        p_gtf = _gunzip_if_needed(ref_gtf, refs_dir, f"{primary_label}{gtf_ext}")
        org_mode = _detect_organism_mode(p_gtf, organism_mode)

        specs = [AssemblySpec(primary_label, p_fna, p_gtf)]
        if secondary_fasta and secondary_gtf:
            s_fna = _gunzip_if_needed(secondary_fasta, refs_dir, f"{secondary_label}.fna")
            s_ext = ".gff3" if str(secondary_gtf).endswith((".gff", ".gff3", ".gff.gz", ".gff3.gz")) else ".gtf"
            s_gtf = _gunzip_if_needed(secondary_gtf, refs_dir, f"{secondary_label}{s_ext}")
            specs.append(AssemblySpec(secondary_label, s_fna, s_gtf))

        _log(f"Building {org_mode} catalog across {[s.assembly_id for s in specs]}...")
        catalog = build_catalog(specs, org_mode, primary_label, work_dir / "tmp", decoys="auto")
        cat_dir = state_root / "catalog"
        catalog.save(cat_dir)
        write_json(cat_dir / "link_stats.json", catalog.link_stats)

        stream_cache = work_dir / "stream_cache"
        stream_cache.mkdir(parents=True, exist_ok=True)
        first_run = manifest.rna_runs[0]
        r1, r2, sample_bytes = sample_reads(first_run, n_tournament_reads, eff_source_mode, cache_dir=stream_cache)
        _log(f"Stage 1 tournament sample: {len(r1):,} {first_run.layout} reads from {first_run.run_id} ({sample_bytes / 1e6:.2f} MB)")

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
        if not kernel_names and regime_dec.excluded:
            pruned = [k for k in cands if k.name not in regime_dec.excluded]
            if pruned:
                cands = pruned

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
        del live, cands
        catalog.release_decoy_sequences()
        import gc
        gc.collect()
        _log(f"Stage 1 winner: {winner} ({t_tour:.1f}s)")

        max_pct = float(max_byte_fraction) * 100.0 if max_byte_fraction <= 1.0 else float(max_byte_fraction)
        init_frac = min(0.004, max_pct / 400.0) if max_pct < 100.0 else 0.25
        loop_cfg = LoopConfig(
            contrast=contrast_list,
            design=design,
            source_mode=eff_source_mode,
            cache_dir=stream_cache,
            initial_fraction=max(0.0015, init_frac),
            growth=2.2,
            max_iterations=8,
            min_iterations=2,
            max_bytes_pct=max_pct,
            delta_lfc_stop=delta_lfc_stop,
            tech_var_stop=v_tech_target,
            zoom=enable_zoom,
            zoom_extra_fraction=min(0.01, max_pct / 100.0 * 0.15),
            length_offsets=(org_mode == "eukaryotic"),
            fetch_workers=min(4, max(1, threads // 2)),
            sample_metadata=sample_meta_df,
        )

        pan_state_dir = state_root / "panexpr_state"
        loop = Focaloop(catalog, manifest, win_kernel, loop_cfg, pan_state_dir, log=_log)
        rep = loop.run()

        # Collect a lightweight independent 1% holdout slice from unstreamed bytes for reusable-holdout queries
        holdout_saved = False
        bytes_by_run: dict[str, int] = {rid: int(acc.bytes_consumed) for rid, acc in loop.accs.items()}
        try:
            hold_accs: dict[str, RunAccumulator] = {}
            hold_batches: list[tuple[str, list, list | None, Any]] = []
            decoy_set = catalog.decoy_ids
            for r in manifest.rna_runs:
                used_b = bytes_by_run.get(r.run_id, 0)
                if used_b + 131_072 < r.total_bytes:
                    rs = open_run_stream(r, eff_source_mode, stream_cache)
                    try:
                        # Advance cursor past discovery bytes, then pull independent holdout slice
                        rs.advance_bytes(used_b)
                        hb_target = max(131_072, min(int(r.total_bytes * 0.005), r.total_bytes - used_b))
                        hr1, hr2 = rs.advance_bytes(hb_target)
                        if hr1:
                            acc_h = RunAccumulator(run=r, decoy_ids=decoy_set)
                            hold_accs[r.run_id] = acc_h
                            hold_batches.append((r.run_id, hr1, hr2, acc_h.calibrator))
                    finally:
                        if hasattr(rs, "close"):
                            rs.close()
            if len(hold_batches) >= 2:
                mapped_h = map_pooled(win_kernel, hold_batches)
                for rid, mres in mapped_h.items():
                    hold_accs[rid].add(mres)
                tc_hold, _, _ = quantify_cohort(hold_accs, catalog, dna_prob=None, strand_prior=rep.strand_priors, em_iters=4)
                PanStateStore(pan_state_dir).write("target_counts_holdout", tc_hold, TARGET_COUNTS_SCHEMA)
                holdout_saved = True
        except Exception as exc:
            _log(f"Holdout slice skipped ({type(exc).__name__}: {exc})")

        bridge_map = _build_ortholog_bridge_map(catalog)
        de_raw = rep.de_postzoom if (rep.de_postzoom is not None and not rep.de_postzoom.empty) else rep.de_prezoom
        de_df = _format_panexpr_de_table(de_raw, bridge_map)
        Path(out_de).parent.mkdir(parents=True, exist_ok=True)
        de_df.to_csv(out_de, sep="\t", index=False, float_format="%.6g")

        # Per-sample summary for HTML report
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
            "zoom": asdict(rep.zoom) if rep.zoom is not None else None,
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
            brand="Focaloop PanExpr",
        )

        # Persist warm metadata + HypothesisLedger inside state_root
        sample_meta_df.to_parquet(state_root / "sample_metadata.parquet", index=False)
        store = FocaloopStateStore(state_root)
        ledger = HypothesisLedger.from_store(store, alpha_budget=0.05)
        if not de_df.empty:
            top = de_df.iloc[0]
            top_lfc = float(top["log2FoldChange"]) if pd.notna(top["log2FoldChange"]) else 0.0
            top_p = float(top["pvalue"]) if pd.notna(top["pvalue"]) else 1.0
            ledger.evaluate_query(
                "q0001",
                "focaloop-panexpr",
                f"{fac}:{lvl_a}_vs_{lvl_b}:primary_discovery",
                top_lfc,
                top_p,
                top_lfc,
                min(top_p, 0.01) if top_p <= 0.005 else top_p,
                store=store,
            )
        state_meta = {
            "engine": "panexpr",
            "version": "0.2.0",
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


def _fetch_crai_file(crai_uri: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(crai_uri).exists():
        shutil.copy2(crai_uri, dst)
        return dst
    s = _ipv4_session()
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = s.get(crai_uri, timeout=60.0)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return dst
        except Exception as exc:
            last_err = exc
            try:
                out = subprocess.run(
                    ["curl", "-4", "-sSL", "--fail", "--max-time", "60", crai_uri],
                    check=True, capture_output=True, timeout=75,
                ).stdout
                dst.write_bytes(out)
                return dst
            except Exception as exc2:
                last_err = exc2
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch CRAI {crai_uri}: {last_err}")


def _resolve_ena_or_odp_for_malariavar(acc: str) -> dict[str, Any]:
    acc = acc.strip().upper()
    odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
    all_odp = False
    odp_bytes = np.nan
    try:
        odp_bytes = float(_head_content_length(odp_url, timeout=20.0))
        all_odp = True
    except Exception:
        pass

    fastq_urls: list[str] = []
    fastq_bytes = np.nan
    order_hint = "flowcell"
    s = _ipv4_session()
    try:
        r = s.get(
            "https://www.ebi.ac.uk/ena/portal/api/filereport",
            params={
                "accession": acc,
                "result": "read_run",
                "fields": "run_accession,fastq_ftp,fastq_bytes,submitted_format",
                "format": "tsv",
            },
            timeout=30.0,
        )
        if r.status_code == 200:
            parsed = list(csv.DictReader(io.StringIO(r.text), delimiter="\t"))
            if parsed:
                row0 = parsed[0]
                fq = row0.get("fastq_ftp") or ""
                if fq:
                    fastq_urls = [fq]
                fb = [int(x) for x in (row0.get("fastq_bytes") or "").split(";") if x.isdigit()]
                if fb:
                    fastq_bytes = float(sum(fb))
                sfmt = (row0.get("submitted_format") or "").upper()
                if sfmt and "SRF" not in sfmt and ("BAM" in sfmt or "CRAM" in sfmt):
                    order_hint = "sorted"
    except Exception:
        pass
    return {
        "ena_runs": [acc],
        "all_odp": all_odp,
        "odp_bytes": odp_bytes,
        "all_paired_fastq": bool(fastq_urls and "_1.fastq.gz" in fastq_urls[0] and "_2.fastq.gz" in fastq_urls[0]),
        "fastq_urls": fastq_urls,
        "fastq_bytes": fastq_bytes,
        "order_hint": order_hint,
    }


def _mapper_runs_cleanly(cand: str) -> bool:
    exe = shutil.which(cand)
    if not exe:
        return False
    try:
        if cand in ("strobealign", "minimap2"):
            with tempfile.TemporaryDirectory(prefix=f"focaloop_{cand}_chk_") as td:
                fa = Path(td) / "r.fa"
                fq = Path(td) / "q.fq"
                seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                fa.write_text(f">chr1\n{seq * 4}\n")
                fq.write_text(f"@r1\n{seq}\n+\n{'I' * len(seq)}\n")
                cmd = [exe, "-t", "1", str(fa), str(fq)] if cand == "strobealign" else [exe, "-t", "1", "-ax", "sr", str(fa), str(fq)]
                rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode
                return rc == 0
        args = ["version"] if cand == "bwa-mem2" else ([] if cand == "bwa" else ["--version"])
        rc = subprocess.run([exe, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode
        return rc >= 0 and rc not in (126, 127, 132, 134, 139)
    except Exception:
        return False


def _select_malariavar_mapper(requested: str) -> str:
    if requested and requested != "auto":
        if _mapper_runs_cleanly(requested):
            return requested
        print(f"[focaloop-malariavar] Requested mapper {requested!r} failed runtime probe; falling back automatically", flush=True)
    for cand in ("strobealign", "minimap2", "bwa-mem2", "bwa"):
        if _mapper_runs_cleanly(cand):
            return cand
    raise RuntimeError("No supported short-read mapper (strobealign/minimap2/bwa-mem2/bwa) found in PATH")


def _format_malariavar_drm_table(
    tel_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    from .malariavar import who_resistance as WR

    cnv_piv = (
        cnv_df.pivot_table(index="sample", columns="locus", values="call", aggfunc="first")
        if not cnv_df.empty
        else pd.DataFrame()
    )
    meta_by_s = meta_df.set_index("Sample").to_dict(orient="index")
    out_rows = []

    def _cnv_str(val: Any, is_del: bool = False) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "undetermined"
        iv = int(float(val))
        if iv < 0:
            return "undetermined"
        if is_del:
            return "deleted" if iv == 1 else "intact"
        return "amplified" if iv == 1 else "single_copy"

    for _, r in tel_df.iterrows():
        s = str(r["sample"])
        m = meta_by_s.get(s, {})
        c_row = cnv_piv.loc[s] if (not cnv_piv.empty and s in cnv_piv.index) else {}
        mdr1_call = c_row.get("mdr1", np.nan) if hasattr(c_row, "get") else np.nan
        pm23_call = c_row.get("pm2_pm3", np.nan) if hasattr(c_row, "get") else np.nan
        gch1_call = c_row.get("gch1", np.nan) if hasattr(c_row, "get") else np.nan
        crt_call = c_row.get("crt", np.nan) if hasattr(c_row, "get") else np.nan
        hrp2_call = c_row.get("hrp2", np.nan) if hasattr(c_row, "get") else np.nan
        hrp3_call = c_row.get("hrp3", np.nan) if hasattr(c_row, "get") else np.nan

        k13_raw = r.get("k13_ns")
        k13_str = "WT" if (k13_raw is None or str(k13_raw).strip() in ("", "none", "None", "nan")) else str(k13_raw)
        who = WR.classify_sample(
            {
                "crt_76": r.get("aa_crt_76"),
                "dhfr_51": r.get("aa_dhfr_51"),
                "dhfr_59": r.get("aa_dhfr_59"),
                "dhfr_108": r.get("aa_dhfr_108"),
                "dhfr_164": r.get("aa_dhfr_164"),
                "dhps_437": r.get("aa_dhps_437"),
                "dhps_540": r.get("aa_dhps_540"),
                "dhps_581": r.get("aa_dhps_581"),
                "dhps_613": r.get("aa_dhps_613"),
                "k13_ns": r.get("k13_ns"),
                "mdr1_amp": mdr1_call,
                "pm2_pm3_amp": pm23_call,
            }
        )
        wire_mb = float(r.get("wire_bytes") or 0.0) / 1e6
        arch_b = r.get("archive_bytes")
        arch_mb = (float(arch_b) / 1e6) if (arch_b is not None and pd.notna(arch_b) and float(arch_b) > 0) else np.nan
        pct_arch = (100.0 * wire_mb / arch_mb) if (pd.notna(arch_mb) and arch_mb > 0) else np.nan
        hfrac = r.get("host_frac")
        host_pct = (100.0 * float(hfrac)) if (hfrac is not None and pd.notna(hfrac)) else 0.0
        fws_val = float(r.get("fws")) if (r.get("fws") is not None and pd.notna(r.get("fws"))) else np.nan
        clonality = "monoclonal" if (np.isfinite(fws_val) and fws_val >= 0.95) else ("polyclonal" if np.isfinite(fws_val) else "unknown")

        dhfr_hap = "".join(str(r.get(f"aa_dhfr_{c}") or "-") for c in (51, 59, 108, 164))
        dhps_hap = "".join(str(r.get(f"aa_dhps_{c}") or "-") for c in (436, 437, 540, 581, 613))
        src_mode = "focused_cram" if str(r.get("source", "")).startswith("sanger_crai") else str(r.get("source", "wgs_stream"))

        out_rows.append(
            {
                "sample_id": s,
                "country_or_group": m.get("Country", ""),
                "population": m.get("Population", m.get("Country", "")),
                "mode": src_mode,
                "stop_reason": str(r.get("stop_reason", "")),
                "wire_mb": round(wire_mb, 3),
                "archive_mb": round(arch_mb, 2) if np.isfinite(arch_mb) else np.nan,
                "pct_archive": round(pct_arch, 3) if np.isfinite(pct_arch) else np.nan,
                "host_sieve_reject_pct": round(host_pct, 2),
                "median_depth": round(float(r.get("median_core_depth") or 0.0), 2),
                "fws": round(fws_val, 4) if np.isfinite(fws_val) else np.nan,
                "clonality": clonality,
                "kelch13_mutations": k13_str,
                "crt_76": str(r.get("aa_crt_76") or "-"),
                "mdr1_86": str(r.get("aa_mdr1_86") or "-"),
                "mdr1_184": str(r.get("aa_mdr1_184") or "-"),
                "dhfr_hap": dhfr_hap,
                "dhps_hap": dhps_hap,
                "hrp2_deletion": _cnv_str(hrp2_call, is_del=True),
                "hrp3_deletion": _cnv_str(hrp3_call, is_del=True),
                "mdr1_cnv": _cnv_str(mdr1_call),
                "pm2_pm3_cnv": _cnv_str(pm23_call),
                "gch1_cnv": _cnv_str(gch1_call),
                "crt_cnv": _cnv_str(crt_call),
                "who_artemisinin": who["Artemisinin"],
                "who_chloroquine": who["Chloroquine"],
                "who_piperaquine": who["Piperaquine"],
                "who_mefloquine": who["Mefloquine"],
                "who_pyrimethamine": who["Pyrimethamine"],
                "who_sulfadoxine": who["Sulfadoxine"],
                "who_sp_uncomplicated": who["SP (uncomplicated)"],
                "who_sp_iptp": who["SP (IPTp)"],
                "who_as_mq": who["AS-MQ"],
                "who_dha_ppq": who["DHA-PPQ"],
            }
        )
    return pd.DataFrame(out_rows)


def run_malariavar_galaxy(
    *,
    manifest_path: str | Path,
    out_drm: str | Path,
    out_variants: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    ref_fasta: str | Path | None = None,
    mode: str = "auto",
    target_depth: float = 24.0,
    mapper: str = "auto",
    threads: int = 8,
    streams: int = 24,
    sample_timeout_s: float = 420.0,
) -> dict[str, Any]:
    """Run MalariaVar surveillance (Mode A: focused .crai CRAM or Mode B: raw SRA + host sieve)."""
    t_start = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix="focaloop_mvar_run_"))
    os.environ["MALARIAVAR_SCRATCH"] = str(work_dir / "scratch")
    os.environ["MALARIAVAR_STAGE_DIR"] = str(work_dir / "stage")
    os.environ["MALARIAVAR_LOGDIR"] = str(work_dir / "logs")

    from .malariavar import engine as E
    from .malariavar import focused_cohort as FC
    from .malariavar import genome as G
    from .malariavar import prepare as P
    from .malariavar import state_store as SS
    from .malariavar import streamer as ST

    # Update module scratch dirs to job-local temp dir
    E.LOGDIR = work_dir / "logs"
    E.STAGE = work_dir / "stage_stream"
    FC.STAGE = work_dir / "stage_focused"
    for d in (E.LOGDIR, E.STAGE, FC.STAGE):
        d.mkdir(parents=True, exist_ok=True)

    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    cohort_dir = state_root / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(f"[focaloop-malariavar] {msg}", flush=True)

    try:
        if ref_fasta:
            user_fa = _gunzip_if_needed(ref_fasta, work_dir / "ref", "Pf3D7_user.fa")
            G.REF_FASTA = user_fa
            if not (user_fa.parent / (user_fa.name + ".fai")).exists() and shutil.which("samtools"):
                subprocess.run(["samtools", "faidx", str(user_fa)], check=False)

        g = G.load()
        rows_raw = _read_manifest_table(manifest_path)
        meta_rows: list[dict[str, Any]] = []
        focused_jobs: list[dict[str, Any]] = []
        stream_jobs: list[dict[str, Any]] = []

        for idx_r, row in enumerate(rows_raw):
            sid = _pick_col(row, ("sample_id", "Sample", "sample"), f"isolate_{idx_r + 1}")
            grp = _pick_col(row, ("country_or_group", "Country", "group", "Population"), "cohort")
            pop = _pick_col(row, ("population", "Population"), grp)
            src = _pick_col(row, ("source_uri_or_srr", "cram_url", "accession", "ena_runs", "source", "url"))
            crai_u = _pick_col(row, ("crai_uri", "crai_uri_optional", "crai_url", "crai"))
            stype = _pick_col(row, ("sample_type", "Sample_type"), "gDNA")
            year_s = _pick_col(row, ("year", "Year"), "2020")
            year = int(year_s) if year_s.isdigit() else 2020

            if not src:
                raise ValueError(f"Isolate {sid!r}: missing source_uri_or_srr")

            row_mode = mode
            if row_mode == "auto":
                row_mode = "focused_cram" if src.split("?", 1)[0].endswith(".cram") else "wgs_stream"

            meta_rows.append(
                {
                    "Sample": sid,
                    "Country": grp,
                    "Population": pop,
                    "Year": year,
                    "Sample_type": stype,
                }
            )
            if row_mode == "focused_cram":
                cram_bytes_s = _pick_col(row, ("cram_bytes", "archive_bytes"))
                cram_bytes = int(float(cram_bytes_s)) if cram_bytes_s else None
                focused_jobs.append(
                    {
                        "Sample": sid,
                        "cram_url": src,
                        "crai_uri": crai_u or (src + ".crai"),
                        "cram_bytes": cram_bytes,
                        "Sample_type": stype,
                    }
                )
            else:
                # Raw SRA/ENA accession(s) or FASTQ URL(s)
                accs = [a.strip() for a in re.split(r"[;,]", src) if a.strip()]
                if all(_INSDC_RUN_RE.match(a) for a in accs):
                    info = _resolve_ena_or_odp_for_malariavar(accs[0])
                    if len(accs) > 1:
                        info["ena_runs"] = [a.upper() for a in accs]
                    stream_jobs.append(
                        {
                            "Sample": sid,
                            "Sample_type": stype,
                            **info,
                        }
                    )
                else:
                    # Direct paired/single FASTQ URLs or local files
                    stream_jobs.append(
                        {
                            "Sample": sid,
                            "Sample_type": stype,
                            "ena_runs": [],
                            "all_odp": False,
                            "all_paired_fastq": ";" in src,
                            "fastq_urls": [src],
                            "fastq_bytes": np.nan,
                            "order_hint": "flowcell",
                        }
                    )

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_parquet(cohort_dir / "manifest.parquet", index=False)

        # --- Execute Mode A: Focused 55-window .crai CRAM HTTP Range Surveillance ---
        if focused_jobs:
            _log(f"Running Mode A (focused_cram) on {len(focused_jobs)} isolates across 55 curated Pf3D7 windows...")
            regs = FC.regions(g)
            FC._init_worker(use_omnitig_zoom=False)
            crai_cache = work_dir / "crai"
            dl_pool = ThreadPoolExecutor(max_workers=streams, thread_name_prefix="crai-range")
            try:
                for idx_j, fj in enumerate(focused_jobs, 1):
                    sid = fj["Sample"]
                    url = fj["cram_url"]
                    crai_local = _fetch_crai_file(fj["crai_uri"], crai_cache / f"{sid}.cram.crai")
                    cram_bytes = fj["cram_bytes"]
                    if not cram_bytes:
                        if Path(url).exists():
                            cram_bytes = Path(url).stat().st_size
                        else:
                            cram_bytes = _head_content_length(url)
                        fj["cram_bytes"] = cram_bytes
                    hdr_end, rr, nc, spans = FC.plan_ranges(crai_local, regs)
                    staged_cram = FC.STAGE / f"{sid}.cram"
                    if Path(url).exists():
                        # Local CRAM slice extraction for offline/unit tests
                        with open(url, "rb") as fi, open(staged_cram, "wb") as fo:
                            fo.write(fi.read(hdr_end))
                            for a, b in rr:
                                fi.seek(a)
                                fo.write(fi.read(b - a))
                            fo.write(ST.CRAM_EOF)
                        fetch_bytes = staged_cram.stat().st_size
                        fetch_s = 0.01
                    else:
                        done_ev = threading.Event()
                        fjob_holder: list[Any] = []

                        def _on_done(j, ev=done_ev, holder=fjob_holder):
                            holder.append(j)
                            ev.set()

                        fjob = FC.FetchJob(sid, url, hdr_end, rr, staged_cram, _on_done)
                        fjob.submit(dl_pool)
                        done_ev.wait(timeout=sample_timeout_s)
                        if not done_ev.is_set():
                            raise TimeoutError(f"Timed out fetching focused CRAM ranges for {sid}")
                        if fjob.failed is not None:
                            raise RuntimeError(f"Focused CRAM range fetch failed for {sid}: {fjob.failed}")
                        fetch_bytes = fjob.total + len(ST.CRAM_EOF)
                        fetch_s = (fjob.t1 or time.time()) - fjob.t0

                    info = {"bytes": fetch_bytes, "n_ranges": len(rr) + 1, "n_containers": nc, "fetch_s": fetch_s}
                    tel = FC.process_one((sid, str(staged_cram), fj, str(cohort_dir), spans.tolist(), info))
                    if tel.get("stop_reason") == "crash" or tel.get("error"):
                        raise RuntimeError(f"Focused CRAM processing failed for {sid}: {tel.get('error')}")
                    _log(
                        f"  [{idx_j}/{len(focused_jobs)}] {sid}: {fetch_bytes / 1e6:.2f} MB / {cram_bytes / 1e6:.0f} MB "
                        f"({100.0 * fetch_bytes / max(cram_bytes, 1):.2f}%), depth={tel.get('median_core_depth', 0):.1f}x, "
                        f"F_WS={tel.get('fws', np.nan):.3f}, k13={tel.get('k13_ns')}"
                    )
            finally:
                dl_pool.shutdown(wait=False, cancel_futures=True)

        # --- Execute Mode B: Raw SRA/FASTQ Streaming + C 31-mer / PLA Host Sieve ---
        if stream_jobs:
            chosen_mapper = _select_malariavar_mapper(mapper)
            use_pla = P.PLA_SIEVE.exists()
            if not use_pla and not P.KSET.exists():
                _log("Building Pf3D7 PLA host-depletion index...")
                P.build_pla_sieve(out_path=work_dir / "pf_pla_sieve.pla", num_threads=min(threads, 8))
                pla_path = str(work_dir / "pf_pla_sieve.pla")
                use_pla = True
            else:
                pla_path = str(P.PLA_SIEVE) if use_pla else None

            if chosen_mapper in ("bwa-mem2", "bwa"):
                ext_check = ".bwt.2bit.64" if chosen_mapper == "bwa-mem2" else ".bwt"
                if not Path(str(G.REF_FASTA) + ext_check).exists():
                    idx_fa = work_dir / "mapper_ref" / G.REF_FASTA.name
                    idx_fa.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(G.REF_FASTA, idx_fa)
                    subprocess.run([chosen_mapper, "index", str(idx_fa)], check=True)
                    G.REF_FASTA = idx_fa

            _log(
                f"Running Mode B (wgs_stream) on {len(stream_jobs)} isolates with mapper={chosen_mapper}, "
                f"pla_sieve={use_pla}, target_depth={target_depth}x..."
            )
            cfg = {
                "map_threads": max(2, min(threads, 6)),
                "xsra_threads": max(2, min(threads // 2, 4)),
                "prefer": "odp_stream",
                "store_candidates": False,
                "mapper": chosen_mapper,
                "min_depth": float(target_depth),
                "min_depth_poly": float(max(target_depth + 8.0, target_depth * 1.25)),
                "use_pla_sieve": use_pla,
                "pla_sieve_path": pla_path,
                "use_omnitig_zoom": True,
                "sample_timeout_s": float(sample_timeout_s),
            }
            for idx_s, srow in enumerate(stream_jobs, 1):
                sid = srow["Sample"]
                tel = E.run_sample(srow, str(cohort_dir), cfg)
                if tel.get("stop_reason") == "error":
                    raise RuntimeError(f"WGS streaming failed for {sid}: {tel.get('error')}")
                _log(
                    f"  [{idx_s}/{len(stream_jobs)}] {sid}: source={tel.get('source')} stop={tel.get('stop_reason')} "
                    f"wire={float(tel.get('wire_bytes', 0)) / 1e6:.1f} MB host_reject={100.0 * float(tel.get('host_frac', 0)):.1f}% "
                    f"depth={float(tel.get('median_core_depth', 0)):.1f}x k13={tel.get('k13_ns')}"
                )

        # Seal shard & materialize warm query matrices (WSAF sketch + DotHash + ESSColor)
        all_samples = [r["Sample"] for r in meta_rows]
        SS.seal_shard(cohort_dir, 0, all_samples)
        warm_info = SS.build_warm(
            cohort_dir,
            meta=meta_df,
            use_dothash_knn=True,
            use_esscolor_panel=True,
            n_threads=min(threads, 4),
        )

        tel_df = SS.read_table(cohort_dir, "sample_telemetry")
        cnv_df = SS.read_table(cohort_dir, "cnv_del_calls")
        drm_df = _format_malariavar_drm_table(tel_df, cnv_df, meta_df)
        Path(out_drm).parent.mkdir(parents=True, exist_ok=True)
        drm_df.to_csv(out_drm, sep="\t", index=False, float_format="%.6g")

        calls_df = SS.read_table(cohort_dir, "variant_calls")
        if not calls_df.empty:
            calls_df = calls_df[calls_df["gt"] > 0].copy()
            contig_names = list(g.names)
            calls_df["chrom"] = [contig_names[int(c)] if 0 <= int(c) < len(contig_names) else str(c) for c in calls_df["chrom"]]
            calls_df = calls_df.rename(columns={"sample": "sample_id"})
            cols_v = ["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "ad_ref", "ad_alt", "dp", "wsaf", "qual", "core"]
            calls_df = calls_df[[c for c in cols_v if c in calls_df.columns]].sort_values(["sample_id", "chrom", "pos"])
        else:
            calls_df = pd.DataFrame(columns=["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "ad_ref", "ad_alt", "dp", "wsaf", "qual", "core"])
        Path(out_variants).parent.mkdir(parents=True, exist_ok=True)
        calls_df.to_csv(out_variants, sep="\t", index=False, float_format="%.6g")

        eff_mode = "focused_cram+wgs_stream" if (focused_jobs and stream_jobs) else ("focused_cram" if focused_jobs else "wgs_stream")
        summary_stats = {
            "mode": eff_mode,
            "n_isolates": len(drm_df),
            "wall_s": round(time.time() - t_start, 2),
            "warm": warm_info,
        }
        render_malariavar_report(out_report, mode=eff_mode, drm_df=drm_df, summary_stats=summary_stats)

        # Initialize HypothesisLedger & STATE_META inside state_root
        store = FocaloopStateStore(state_root)
        HypothesisLedger.from_store(store, alpha_budget=0.05)
        drm_df.to_parquet(state_root / "drm_calls.parquet", index=False)
        state_meta = {
            "engine": "malariavar",
            "version": "0.2.0",
            "blueprint_id": "malariavar",
            "mode": eff_mode,
            "summary": summary_stats,
            "samples": [
                {
                    "sample": r["Sample"],
                    "condition": r["Country"],
                    "country_or_group": r["Country"],
                    "Country": r["Country"],
                    "Population": r["Population"],
                }
                for r in meta_rows
            ],
        }
        write_json(state_root / STATE_META, state_meta)
        if state_archive:
            pack_state(state_root, state_archive)
        return state_meta
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _block_all_network_sockets():
    """Disable socket creation during warm queries (saves and restores original hooks on exit)."""
    orig_sock = socket.socket
    orig_conn = socket.create_connection

    class _BlockedSocket(orig_sock):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Network sockets are disabled during warm Focaloop Query (0 network bytes permitted)")

    def _blocked_conn(*args, **kwargs):
        raise RuntimeError("Network connections are disabled during warm Focaloop Query (0 network bytes permitted)")

    socket.socket = _BlockedSocket  # type: ignore[misc]
    socket.create_connection = _blocked_conn  # type: ignore[assignment]
    return orig_sock, orig_conn


def _restore_network_sockets(saved) -> None:
    orig_sock, orig_conn = saved
    socket.socket = orig_sock  # type: ignore[misc]
    socket.create_connection = orig_conn  # type: ignore[assignment]


def _bh_padj(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(ranked, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    out[ok] = adj[inv]
    return out


def _parse_subset_spec(subset: str | None) -> list[tuple[str, list[str]]]:
    """Parse space- or &-separated sample subset filters like 'plate_id=P1 timepoint=60m,120m'."""
    raw = str(subset or "").strip()
    if not raw:
        return []
    clauses: list[tuple[str, list[str]]] = []
    for tok in re.split(r"[\s&]+", raw):
        t = tok.strip()
        if not t:
            continue
        if "=" not in t:
            raise ValueError(f"Invalid --subset clause {t!r}; expected column=val1,val2 (space-separated for multiple filters)")
        col, vals_s = t.split("=", 1)
        col = col.strip()
        vals = [v.strip() for v in vals_s.split(",") if v.strip()]
        if not col or not vals:
            raise ValueError(f"Invalid --subset clause {t!r}; both column name and value(s) are required")
        clauses.append((col, vals))
    return clauses


def _apply_subset_df(df: pd.DataFrame, clauses: list[tuple[str, list[str]]]) -> pd.DataFrame:
    if not clauses:
        return df
    out = df
    cols_ci = {str(c).lower(): c for c in out.columns}
    for col, vals in clauses:
        real_col = col if col in out.columns else cols_ci.get(col.lower())
        if real_col is None:
            raise ValueError(f"Subset column {col!r} was not found in saved sample metadata columns {list(out.columns)}")
        mask = out[real_col].astype(str).isin(set(vals))
        out = out.loc[mask]
    if out.empty:
        spec = " ".join(f"{c}={','.join(vs)}" for c, vs in clauses)
        raise ValueError(f"Subset filter {spec!r} matched 0 samples in the saved state bundle.")
    return out


def _prune_constant_design_terms(design: str, meta_df: pd.DataFrame, focal_factor: str) -> str:
    """Drop additive covariates that collapsed to a single level after --subset so PyDESeq2 stays full-rank."""
    d = str(design or "~condition").strip()
    if not d.startswith("~") or "*" in d or ":" in d or "(" in d:
        return d
    terms = [t.strip() for t in d[1:].split("+") if t.strip()]
    kept: list[str] = []
    for term in terms:
        if term == focal_factor or term not in meta_df.columns:
            kept.append(term)
        elif int(meta_df[term].astype(str).nunique(dropna=True)) >= 2:
            kept.append(term)
    if not kept:
        kept = [focal_factor]
    return "~" + " + ".join(kept)


def run_warm_engine_query(
    *,
    state_dir: Path,
    meta: dict[str, Any],
    contrast: str,
    actor: str,
    out_table: str | Path,
    out_verdict: str | Path,
    hypothesis: str | None = None,
    design_override: str | None = None,
    subset: str | None = None,
    mode: str = "auto",
    n_perm: int = 199,
    out_report: str | Path | None = None,
    out_state_archive: str | Path | None = None,
    alpha_budget: float = 0.05,
) -> dict[str, Any]:
    """Execute 0-network-byte warm query on an ExpressionLoop or VariationLoop state directory."""
    t0 = time.time()
    engine = meta["engine"]
    store = FocaloopStateStore(state_dir)
    ledger = HypothesisLedger.from_store(store, alpha_budget=alpha_budget)
    subset_clauses = _parse_subset_spec(subset)

    sock_guard = _block_all_network_sockets()
    try:
        if engine in ("panexpr", "expressionloop"):
            if mode == "fst":
                raise ValueError("Mode 'fst' is only valid for VariationLoop states, not ExpressionLoop states.")
            from .panexpr.catalog import Catalog
            from .panexpr.solver import de_with_uncertainty, gene_level_tables

            contrast_list = _parse_panexpr_contrast(contrast)
            fac, lvl_a, lvl_b = contrast_list
            catalog = Catalog.load(state_dir / "catalog")
            pan_state = state_dir / "panexpr_state"
            tc = pq.read_table(pan_state / "target_counts.parquet")
            sample_meta = pd.read_parquet(state_dir / "sample_metadata.parquet") if (state_dir / "sample_metadata.parquet").exists() else None
            design = design_override or meta.get("design", "~condition")
            length_offsets = bool(meta.get("length_offsets", True))

            gl = gene_level_tables(tc, catalog, length_offsets=length_offsets)
            meta_df = gl["meta"].copy()
            if sample_meta is not None and not sample_meta.empty:
                idx_col = "sample" if "sample" in sample_meta.columns else ("sample_id" if "sample_id" in sample_meta.columns else None)
                sm = sample_meta.set_index(idx_col) if idx_col else sample_meta
                sm = sm[~sm.index.duplicated(keep="first")]
                for col in sm.columns:
                    if col not in meta_df.columns:
                        meta_df[col] = sm[col].reindex(meta_df.index)
            counts = gl.get("counts_length_scaled", gl["counts"]) if length_offsets else gl["counts"]
            avar = gl["avar"]
            if subset_clauses:
                meta_df = _apply_subset_df(meta_df, subset_clauses)
                counts = counts.loc[meta_df.index]
                avar = avar.loc[meta_df.index]
                if fac not in meta_df.columns:
                    raise ValueError(f"Contrast factor {fac!r} not found in sample metadata columns {list(meta_df.columns)}")
                avail_lvls = set(meta_df[fac].astype(str))
                missing_lvls = [lv for lv in (lvl_a, lvl_b) if lv not in avail_lvls]
                if missing_lvls:
                    raise ValueError(
                        f"After applying subset {subset!r} ({len(meta_df)} samples), contrast level(s) {missing_lvls} "
                        f"are absent from column {fac!r} (present: {sorted(avail_lvls)})."
                    )
            eff_design = _prune_constant_design_terms(design, meta_df, fac) if subset_clauses else design
            symbols = {g: ts[0].symbol for g, ts in catalog.groups.items()}
            gene_ids = catalog.primary_gene_of_group()

            raw_de = de_with_uncertainty(
                counts,
                avar,
                meta_df,
                contrast_list,
                design=eff_design,
                dna_prob=gl["dna_prob"],
                symbols=symbols,
                gene_ids=gene_ids,
            )
            bridge_map = _build_ortholog_bridge_map(catalog)
            de_df = _format_panexpr_de_table(raw_de, bridge_map)

            # Evaluate independent holdout if target_counts_holdout.parquet exists
            hold_est, hold_p = 0.0, 1.0
            disc_est, disc_p = 0.0, 1.0
            target_gene = None
            if not de_df.empty:
                if hypothesis:
                    key = hypothesis.split("=", 1)[-1].strip()
                    hit = de_df[(de_df["gene_id"] == key) | (de_df["ortholog_bridge"].str.contains(key, na=False))]
                    if hit.empty:
                        raise ValueError(f"Requested hypothesis gene {key!r} was not found in the differential expression table.")
                    row0 = hit.iloc[0]
                else:
                    row0 = de_df.iloc[0]
                target_gene = str(row0["gene_id"])
                disc_est = float(row0["log2FoldChange"]) if pd.notna(row0["log2FoldChange"]) else 0.0
                disc_p = float(row0["pvalue"]) if pd.notna(row0["pvalue"]) else 1.0

                hold_path = pan_state / "target_counts_holdout.parquet"
                if hold_path.exists():
                    tc_h = pq.read_table(hold_path).to_pandas()
                    tc_h_gene = tc_h[tc_h["gene_id"] == target_gene]
                    if not tc_h_gene.empty:
                        a_sum = float(tc_h_gene.loc[tc_h_gene["condition"] == lvl_a, "effective_count"].sum())
                        b_sum = float(tc_h_gene.loc[tc_h_gene["condition"] == lvl_b, "effective_count"].sum())
                        n_a = max(int(tc_h.loc[tc_h["condition"] == lvl_a, "sample"].nunique()), 1)
                        n_b = max(int(tc_h.loc[tc_h["condition"] == lvl_b, "sample"].nunique()), 1)
                        hold_est = math.log2(((a_sum / n_a) + 0.5) / ((b_sum / n_b) + 0.5))
                        tot_n = int(round(a_sum + b_sum))
                        if tot_n >= 4:
                            p_null = n_a / float(n_a + n_b)
                            hold_p = float(sp_stats.binomtest(int(round(a_sum)), tot_n, p=p_null).pvalue)
                        else:
                            hold_p = disc_p
                    else:
                        hold_est, hold_p = disc_est, disc_p
                else:
                    hold_est, hold_p = disc_est, disc_p

            qid = f"q{(ledger._query_index + 1):04d}"
            spec = f"{fac}:{lvl_a}_vs_{lvl_b}:gene={target_gene or 'none'}"
            qv = ledger.evaluate_query(qid, actor, spec, disc_est, disc_p, hold_est, hold_p, store=store)
            Path(out_table).parent.mkdir(parents=True, exist_ok=True)
            de_df.to_csv(out_table, sep="\t", index=False, float_format="%.6g")
            n_deg = int((de_df["padj"].fillna(1.0) < 0.05).sum())
            wall_s = round(time.time() - t0, 3)
            out_doc = {
                "tool": "focaloop",
                "command": "query",
                "engine": "panexpr",
                "mode": "contrast",
                "contrast": {"field": fac, "a": lvl_a, "b": lvl_b},
                "subset": subset or None,
                "effective_design": eff_design,
                "n_samples_queried": int(len(meta_df)),
                "new_bytes": 0,
                "network_sockets_blocked": True,
                "wall_seconds": wall_s,
                "n_genes": len(de_df),
                "n_deg_fdr05": n_deg,
                "top_gene": target_gene,
                "verdict": qv.__dict__,
                "wealth_after": qv.alpha_wealth_after,
                "alpha_spent": qv.alpha_allocated,
                "headline_metric": f"{n_deg:,} DEGs (FDR < 0.05)",
                "headline_detail": f"Top: {target_gene} (log2FC={disc_est:.2f}, p={disc_p:.2e})",
            }
            write_json(out_verdict, out_doc)
            if out_report:
                render_query_report(
                    out_report,
                    engine="ExpressionLoop",
                    contrast_str=f"{fac}: {lvl_a} vs {lvl_b}",
                    verdict=out_doc,
                    table_df=de_df,
                )
        elif engine == "malariavar":
            if mode == "contrast":
                raise ValueError("Mode 'contrast' is only valid for ExpressionLoop states, not VariationLoop states.")
            from .malariavar import query as MQ

            cohort_dir = state_dir / "cohort"
            wm = MQ.Warm(cohort_dir, load_dothash=True, load_esscolor=True)
            drm_saved = pd.read_parquet(state_dir / "drm_calls.parquet") if (state_dir / "drm_calls.parquet").exists() else pd.DataFrame()

            # Parse contrast: e.g. Population:AS-SE-E:AF-W, Country:Cambodia,Vietnam:Ghana,Mali, or AS-SE-E:AF-W
            raw_c = contrast.strip()
            if "=" in raw_c and ":" in raw_c:
                col_hint, rest = raw_c.split("=", 1)
                grp_a_s, grp_b_s = rest.split(":", 1)
            elif raw_c.count(":") >= 2:
                col_hint, grp_a_s, grp_b_s = raw_c.split(":", 2)
            elif ":" in raw_c:
                col_hint = "auto"
                grp_a_s, grp_b_s = raw_c.split(":", 1)
            elif "_vs_" in raw_c:
                col_hint = "auto"
                grp_a_s, grp_b_s = raw_c.split("_vs_", 1)
            else:
                raise ValueError(f"Invalid MalariaVar query contrast {contrast!r}; expected 'GroupA:GroupB' or 'Country:GroupA:GroupB'")

            vals_a = [x.strip() for x in grp_a_s.split(",") if x.strip()]
            vals_b = [x.strip() for x in grp_b_s.split(",") if x.strip()]

            col_map = {
                "country": "Country",
                "group": "Country",
                "country_or_group": "Country",
                "population": "Population",
            }
            s_df = wm.samples
            if col_hint == "auto":
                chosen_col = "Country"
                for cand_col in ("Country", "Population"):
                    if cand_col in s_df.columns:
                        present = set(s_df[cand_col].dropna().astype(str))
                        if set(vals_a).issubset(present) and set(vals_b).issubset(present):
                            chosen_col = cand_col
                            break
            else:
                chosen_col = col_map.get(col_hint.lower(), col_hint)
                if chosen_col not in s_df.columns:
                    raise ValueError(f"Column {chosen_col!r} not in warm sample table: {list(s_df.columns)}")

            ia = np.where(s_df[chosen_col].astype(str).isin(vals_a))[0]
            ib = np.where(s_df[chosen_col].astype(str).isin(vals_b))[0]
            if len(ia) == 0 or len(ib) == 0:
                avail = sorted(s_df[chosen_col].dropna().astype(str).unique())
                raise ValueError(f"Empty sub-cohort for {chosen_col}: {vals_a} (n={len(ia)}) vs {vals_b} (n={len(ib)}); available={avail}")

            # 1. Hudson F_ST across discovery vs holdout panel sites
            min_dp_fst = 3
            min_n_site = max(1, min(len(ia), len(ib)) // 2)
            p1, n1 = wm.freqs(ia, min_dp=min_dp_fst)
            p2, n2 = wm.freqs(ib, min_dp=min_dp_fst)
            ok = (n1 >= min_n_site) & (n2 >= min_n_site) & ((p1 + p2) > 0) & ((p1 + p2) < 2)
            num_all, den_all = MQ.hudson_fst(p1[ok], n1[ok], p2[ok], n2[ok])
            ok_idx = np.nonzero(ok)[0]
            hold_mask = np.array([is_holdout_block("pf8_panel", int(i), holdout_fraction=0.20) for i in ok_idx], dtype=bool)
            disc_mask = ~hold_mask

            genome_fst = float(num_all.sum() / den_all.sum()) if den_all.sum() > 0 else 0.0
            disc_fst = float(num_all[disc_mask].sum() / den_all[disc_mask].sum()) if den_all[disc_mask].sum() > 0 else genome_fst
            hold_fst = float(num_all[hold_mask].sum() / den_all[hold_mask].sum()) if den_all[hold_mask].sum() > 0 else genome_fst

            # Permutation p-value for F_ST across samples on discovery & holdout site partitions
            rng = np.random.default_rng(20261113)
            pooled_idx = np.concatenate([ia, ib])
            na_len = len(ia)
            n_perm = 199 if len(pooled_idx) >= 6 else 49
            perm_disc_ge = 0
            perm_hold_ge = 0
            if ok_idx.size > 0:
                w_sub = np.asarray(wm.wsaf[pooled_idx][:, ok_idx], dtype=np.float32)
                d_sub = np.asarray(wm.dp[pooled_idx][:, ok_idx])
                valid_sub = (w_sub != 255) & (d_sub >= min_dp_fst)
                wf_sub = np.where(valid_sub, w_sub / 200.0, 0.0)
                for _ in range(n_perm):
                    perm = rng.permutation(len(pooled_idx))
                    pa_i, pb_i = perm[:na_len], perm[na_len:]
                    na_cnt = valid_sub[pa_i].sum(0)
                    nb_cnt = valid_sub[pb_i].sum(0)
                    pa_f = wf_sub[pa_i].sum(0) / np.maximum(na_cnt, 1)
                    pb_f = wf_sub[pb_i].sum(0) / np.maximum(nb_cnt, 1)
                    pnum, pden = MQ.hudson_fst(pa_f, na_cnt, pb_f, nb_cnt)
                    pf_d = float(pnum[disc_mask].sum() / max(float(pden[disc_mask].sum()), 1e-12))
                    pf_h = float(pnum[hold_mask].sum() / max(float(pden[hold_mask].sum()), 1e-12))
                    if pf_d >= disc_fst - 1e-12:
                        perm_disc_ge += 1
                    if pf_h >= hold_fst - 1e-12:
                        perm_hold_ge += 1
            disc_p = (perm_disc_ge + 1.0) / (n_perm + 1.0)
            hold_p = (perm_hold_ge + 1.0) / (n_perm + 1.0)

            # Locus-specific F_ST around curated drug-resistance genes
            goff_ok = wm.panel["goff"][ok]
            locus_rows = []
            for lname, gid in MQ.KNOWN.items():
                gg = wm.g.genes[gid]
                x0 = wm.g.goff(gg["contig"], gg["start"]) - 35_000
                x1 = wm.g.goff(gg["contig"], gg["end"]) + 35_000
                near = (goff_ok >= x0) & (goff_ok <= x1)
                if near.any() and den_all[near].sum() > 0:
                    lfst = float(num_all[near].sum() / den_all[near].sum())
                    locus_rows.append(
                        {
                            "category": "locus_fst",
                            "feature": f"F_ST({lname})",
                            "group_a": ",".join(vals_a),
                            "group_b": ",".join(vals_b),
                            "n_a": int(len(ia)),
                            "n_b": int(len(ib)),
                            "value_a": np.nan,
                            "value_b": np.nan,
                            "effect_size": round(lfst, 4),
                            "pvalue": np.nan,
                            "notes": f"{int(near.sum())} panel sites ±35kb of {gid}",
                        }
                    )

            # 2. Marker & WHO drug-resistance prevalence differences (Fisher exact test)
            flags = MQ.mutant_flags(s_df)
            cnv_mat = MQ.cnv_matrix(wm)
            for col_c in ("mdr1", "pm2_pm3", "gch1", "hrp2", "hrp3"):
                if col_c in cnv_mat.columns:
                    flags[f"{col_c}_{'del' if col_c.startswith('hrp') else 'cnv'}"] = cnv_mat[col_c].to_numpy(dtype=float)
            if not drm_saved.empty and "sample_id" in drm_saved.columns:
                drm_idx = drm_saved.set_index("sample_id").reindex(s_df["sample"])
                for who_col in ("who_artemisinin", "who_chloroquine", "who_piperaquine", "who_sp_uncomplicated", "who_dha_ppq"):
                    if who_col in drm_idx.columns:
                        vals = drm_idx[who_col].to_numpy()
                        flags[who_col] = np.where(vals == "Resistant", 1.0, np.where(vals == "Sensitive", 0.0, np.nan))

            prev_rows = []
            pvals_raw = []
            for mcol in flags.columns:
                va = flags[mcol].to_numpy(dtype=float)[ia]
                vb = flags[mcol].to_numpy(dtype=float)[ib]
                va_ok = va[np.isfinite(va)]
                vb_ok = vb[np.isfinite(vb)]
                if len(va_ok) == 0 or len(vb_ok) == 0:
                    continue
                ka, na_m = int(round(va_ok.sum())), len(va_ok)
                kb, nb_m = int(round(vb_ok.sum())), len(vb_ok)
                pa_prev = ka / na_m
                pb_prev = kb / nb_m
                _, fp = sp_stats.fisher_exact([[ka, na_m - ka], [kb, nb_m - kb]])
                pvals_raw.append(float(fp))
                prev_rows.append(
                    {
                        "category": "marker_prevalence",
                        "feature": mcol,
                        "group_a": ",".join(vals_a),
                        "group_b": ",".join(vals_b),
                        "n_a": na_m,
                        "n_b": nb_m,
                        "value_a": round(pa_prev, 4),
                        "value_b": round(pb_prev, 4),
                        "effect_size": round(pa_prev - pb_prev, 4),
                        "pvalue": float(fp),
                        "notes": f"{ka}/{na_m} vs {kb}/{nb_m}",
                    }
                )
            padjs = _bh_padj(np.array(pvals_raw)) if pvals_raw else np.array([])
            for r_p, q_p in zip(prev_rows, padjs):
                r_p["padj"] = float(q_p) if np.isfinite(q_p) else np.nan
            prev_rows.sort(key=lambda x: (x["pvalue"], -abs(x["effect_size"])))

            fst_header_row = {
                "category": "genome_fst",
                "feature": "hudson_fst_genome",
                "group_a": ",".join(vals_a),
                "group_b": ",".join(vals_b),
                "n_a": int(len(ia)),
                "n_b": int(len(ib)),
                "value_a": np.nan,
                "value_b": np.nan,
                "effect_size": round(genome_fst, 4),
                "pvalue": round(disc_p, 5),
                "padj": round(disc_p, 5),
                "notes": f"discovery_fst={disc_fst:.4f}, holdout_fst={hold_fst:.4f} ({int(ok.sum())} informative sites)",
            }
            for lr in locus_rows:
                lr["padj"] = np.nan
            table_df = pd.DataFrame([fst_header_row] + prev_rows + locus_rows)

            qid = f"q{(ledger._query_index + 1):04d}"
            spec = f"{chosen_col}:{','.join(vals_a)}_vs_{','.join(vals_b)}:hudson_fst"
            qv = ledger.evaluate_query(qid, actor, spec, disc_fst, disc_p, hold_fst, hold_p, store=store)

            Path(out_table).parent.mkdir(parents=True, exist_ok=True)
            table_df.to_csv(out_table, sep="\t", index=False, float_format="%.6g")
            wall_s = round(time.time() - t0, 3)
            top_marker = prev_rows[0]["feature"] if prev_rows else "hudson_fst"
            top_diff = prev_rows[0]["effect_size"] if prev_rows else genome_fst
            out_doc = {
                "tool": "focaloop",
                "command": "query",
                "engine": "malariavar",
                "mode": "fst_and_prevalence",
                "contrast": {"field": chosen_col, "a": vals_a, "b": vals_b},
                "new_bytes": 0,
                "network_sockets_blocked": True,
                "wall_seconds": wall_s,
                "genome_fst": round(genome_fst, 4),
                "discovery_fst": round(disc_fst, 4),
                "holdout_fst": round(hold_fst, 4),
                "informative_sites": int(ok.sum()),
                "verdict": qv.__dict__,
                "wealth_after": qv.alpha_wealth_after,
                "alpha_spent": qv.alpha_allocated,
                "headline_metric": f"F_ST = {genome_fst:.3f}",
                "headline_detail": f"Top Δprev: {top_marker} ({top_diff:+.2f})",
            }
            write_json(out_verdict, out_doc)
            if out_report:
                render_query_report(
                    out_report,
                    engine="VariationLoop",
                    contrast_str=f"{chosen_col}: {','.join(vals_a)} vs {','.join(vals_b)}",
                    verdict=out_doc,
                    table_df=table_df,
                )
        elif engine == "variationloop":
            if mode == "contrast":
                raise ValueError("Mode 'contrast' is only valid for ExpressionLoop states, not VariationLoop states.")
            vpath = state_dir / "variants.parquet"
            if not vpath.exists():
                raise FileNotFoundError(f"Saved VariationLoop state is missing variants.parquet: {state_dir}")
            vdf = pd.read_parquet(vpath)
            s_rows = meta.get("samples") or []
            if not s_rows:
                raise ValueError("Saved VariationLoop state contains no sample metadata for group comparisons.")
            s_df = pd.DataFrame(s_rows)
            if subset_clauses:
                s_df = _apply_subset_df(s_df, subset_clauses)

            raw_c = contrast.strip()
            if "=" in raw_c and ":" in raw_c:
                col_hint, rest = raw_c.split("=", 1)
                grp_a_s, grp_b_s = rest.split(":", 1)
            elif raw_c.count(":") >= 2:
                col_hint, grp_a_s, grp_b_s = raw_c.split(":", 2)
            elif ":" in raw_c:
                col_hint = "auto"
                grp_a_s, grp_b_s = raw_c.split(":", 1)
            elif "_vs_" in raw_c:
                col_hint = "auto"
                grp_a_s, grp_b_s = raw_c.split("_vs_", 1)
            else:
                raise ValueError(f"Invalid contrast {contrast!r}; expected field=GroupA:GroupB or GroupA:GroupB")

            vals_a = [x.strip() for x in grp_a_s.split(",") if x.strip()]
            vals_b = [x.strip() for x in grp_b_s.split(",") if x.strip()]
            cand_cols = [c for c in s_df.columns if c != "sample"]
            chosen_col = None
            if col_hint != "auto":
                for c in cand_cols:
                    if c.lower() == col_hint.strip().lower():
                        chosen_col = c
                        break
            if chosen_col is None:
                for c in cand_cols:
                    vals = set(s_df[c].astype(str))
                    if any(x in vals for x in vals_a) and any(x in vals for x in vals_b):
                        chosen_col = c
                        break
            if chosen_col is None:
                raise ValueError(
                    f"Contrast groups {vals_a} vs {vals_b} were not found in VariationLoop sample columns {cand_cols}. "
                    "Multi-group cohort states with at least two labeled groups are required for population comparisons."
                )

            set_a = set(s_df.loc[s_df[chosen_col].astype(str).isin(vals_a), "sample"].astype(str))
            set_b = set(s_df.loc[s_df[chosen_col].astype(str).isin(vals_b), "sample"].astype(str))
            if not set_a or not set_b:
                raise ValueError(f"Both comparison groups must contain at least one sample (got {len(set_a)} vs {len(set_b)}).")

            state_ploidy = str(meta.get("ploidy") or (meta.get("summary") or {}).get("ploidy") or "diploid").strip().lower()

            def _row_allele_freq(gt: Any, wsaf: Any) -> float:
                s = str(gt or "").replace("|", "/")
                if state_ploidy == "haploid" or s == "1":
                    try:
                        w = float(wsaf)
                        if np.isfinite(w):
                            return float(np.clip(w, 0.0, 1.0))
                    except (TypeError, ValueError):
                        pass
                    return 1.0 if s in ("1", "1/1") else (0.5 if s in ("0/1", "1/0", "1/2", "2/1") else 0.0)
                if s in ("1/1", "1"):
                    return 1.0
                if s in ("0/1", "1/0", "1/2", "2/1"):
                    return 0.5
                return 0.0

            n_a, n_b = len(set_a), len(set_b)
            rows_out: list[dict[str, Any]] = []
            num_sum = 0.0
            den_sum = 0.0
            if not vdf.empty:
                vsub = vdf[vdf["sample_id"].astype(str).isin(set_a | set_b)].copy()
                if hypothesis:
                    key = hypothesis.split("=", 1)[-1].strip()
                    if ":" in key and "-" in key.split(":", 1)[1]:
                        ch_h, span_h = key.split(":", 1)
                        s_h, e_h = [int(x) for x in span_h.split("-", 1)]
                        vsub = vsub[(vsub["chrom"].astype(str) == ch_h) & (vsub["pos"] >= s_h) & (vsub["pos"] <= e_h)]
                    else:
                        vsub = vsub[vsub["chrom"].astype(str) == key]
                ws_col = vsub["wsaf"] if "wsaf" in vsub.columns else pd.Series([np.nan] * len(vsub), index=vsub.index)
                vsub["dosage"] = [_row_allele_freq(g, w) for g, w in zip(vsub["gt"], ws_col)]
                for (ch, pos, ref, alt), grp_df in vsub.groupby(["chrom", "pos", "ref", "alt"], sort=True):
                    da = grp_df.loc[grp_df["sample_id"].astype(str).isin(set_a), "dosage"].sum()
                    db = grp_df.loc[grp_df["sample_id"].astype(str).isin(set_b), "dosage"].sum()
                    pa = float(da) / float(n_a)
                    pb = float(db) / float(n_b)
                    c_a = (pa * (1.0 - pa) / (n_a - 1)) if n_a > 1 else 0.0
                    c_b = (pb * (1.0 - pb) / (n_b - 1)) if n_b > 1 else 0.0
                    num = (pa - pb) ** 2 - c_a - c_b
                    den = pa * (1.0 - pb) + pb * (1.0 - pa)
                    fst_val = float(num / den) if den > 1e-12 else 0.0
                    num_sum += num
                    den_sum += den
                    rows_out.append(
                        {
                            "chrom": str(ch),
                            "pos": int(pos),
                            "ref": str(ref),
                            "alt": str(alt),
                            "af_group_a": round(pa, 4),
                            "af_group_b": round(pb, 4),
                            "delta_af": round(pa - pb, 4),
                            "hudson_fst": round(fst_val, 4),
                            "n_group_a": n_a,
                            "n_group_b": n_b,
                        }
                    )

            table_df = pd.DataFrame(rows_out)
            if not table_df.empty:
                table_df = table_df.sort_values(by=["hudson_fst", "delta_af"], ascending=[False, False]).reset_index(drop=True)
            genome_fst = float(num_sum / den_sum) if den_sum > 1e-12 else 0.0

            # Permutation p-value over group labels
            rng = np.random.default_rng(42)
            all_samps = sorted(set_a | set_b)
            n_ge = 0
            n_perm_eff = max(1, int(n_perm))
            if rows_out and len(all_samps) >= 2 and not vdf.empty:
                sites_list = list(vsub.groupby(["chrom", "pos", "ref", "alt"], sort=False))
                samp_to_idx = {s: i for i, s in enumerate(all_samps)}
                mat = np.zeros((len(sites_list), len(all_samps)), dtype=np.float64)
                for si, (_, gdf) in enumerate(sites_list):
                    for sid_v, dos_v in zip(gdf["sample_id"].astype(str), gdf["dosage"].values):
                        j = samp_to_idx.get(sid_v)
                        if j is not None:
                            mat[si, j] = float(dos_v)
                for _ in range(n_perm_eff):
                    perm = rng.permutation(len(all_samps))
                    pa_v = mat[:, perm[:n_a]].mean(axis=1)
                    pb_v = mat[:, perm[n_a:]].mean(axis=1)
                    ca_v = (pa_v * (1.0 - pa_v) / (n_a - 1)) if n_a > 1 else 0.0
                    cb_v = (pb_v * (1.0 - pb_v) / (n_b - 1)) if n_b > 1 else 0.0
                    p_den = float(np.sum(pa_v * (1.0 - pb_v) + pb_v * (1.0 - pa_v)))
                    p_fst = float(np.sum((pa_v - pb_v) ** 2 - ca_v - cb_v) / p_den) if p_den > 1e-12 else 0.0
                    if p_fst >= genome_fst - 1e-12:
                        n_ge += 1
            perm_p = float((n_ge + 1) / (n_perm_eff + 1))

            qid = f"q{(ledger._query_index + 1):04d}"
            spec = f"F_ST({chosen_col}: {','.join(vals_a)} vs {','.join(vals_b)})"
            qv = ledger.evaluate_query(qid, actor, spec, genome_fst, perm_p, genome_fst, perm_p, store=store)
            Path(out_table).parent.mkdir(parents=True, exist_ok=True)
            table_df.to_csv(out_table, sep="\t", index=False, float_format="%.6g")
            wall_s = round(time.time() - t0, 3)
            out_doc = {
                "tool": "focaloop",
                "command": "query",
                "engine": "variationloop",
                "mode": "fst",
                "contrast": {"field": chosen_col, "a": vals_a, "b": vals_b},
                "new_bytes": 0,
                "network_sockets_blocked": True,
                "wall_seconds": wall_s,
                "genome_fst": round(genome_fst, 4),
                "informative_sites": len(rows_out),
                "pvalue": round(perm_p, 6),
                "verdict": qv.__dict__,
                "wealth_after": qv.alpha_wealth_after,
                "alpha_spent": qv.alpha_allocated,
                "headline_metric": f"F_ST = {genome_fst:.3f}",
                "headline_detail": f"{len(rows_out):,} variant sites compared (p = {perm_p:.3g})",
            }
            write_json(out_verdict, out_doc)
            if out_report:
                render_query_report(
                    out_report,
                    engine="VariationLoop",
                    contrast_str=f"{chosen_col}: {','.join(vals_a)} vs {','.join(vals_b)}",
                    verdict=out_doc,
                    table_df=table_df,
                )
        else:
            raise ValueError(f"Unsupported engine {engine!r} in warm query")
    finally:
        _restore_network_sockets(sock_guard)

    if out_state_archive:
        pack_state(state_dir, out_state_archive)
    return out_doc


def prebuild_all_native_libs() -> dict[str, str]:
    """Compile and load all C99/OpenMP binaries and shared libraries across all Focaloop engines."""
    from .malariavar.native._loader import ensure_shared_lib
    from .runners_expression import prebuild_expressionloop_native_libs
    from .runners_geneloop import prebuild_geneloop_native_libs

    built: dict[str, str] = {}
    built.update(prebuild_expressionloop_native_libs())
    for stem, srcs, hdrs, omp in (
        ("libpla_sieve", ["pla_sieve.c"], ["pla_sieve.h"], True),
        ("libdothash", ["dothash_sketch.c"], None, True),
        ("libisland_pc", ["island_pc_core.c"], None, False),
        ("libust", ["ust_kernel.c"], None, False),
        ("varloop_core", ["varloop_core.c"], None, False),
    ):
        p = ensure_shared_lib(stem, srcs, headers=hdrs, use_openmp=omp)
        built[f"malariavar_{stem}"] = str(p)
    from .malariavar.logan_var import ensure_var_logan_bins

    v_prep, v_pile = ensure_var_logan_bins()
    built["malariavar_prep_unitigs_filt_adapt"] = str(v_prep)
    built["malariavar_logan_var_pileup"] = str(v_pile)
    built.update(prebuild_geneloop_native_libs())
    return built
