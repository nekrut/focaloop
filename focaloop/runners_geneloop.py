"""Production Galaxy runner for Focaloop GeneLoop (FocalAnnot) eukaryotic genome annotation."""

from __future__ import annotations

import csv
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Sequence

from .archive import pack_state
from .engine import STATE_META, write_json
from .geneloop import __version__ as GENELOOP_VERSION, prebuild_native_libs as _prebuild_gl_native
from .geneloop.engine import GateRun, LoopParams
from .geneloop.gffio import Transcript, count_models_by_biotype, read_models, write_gff3, write_gtf
from .geneloop.seed import (
    detect_tiberius_runtime,
    discover_tiberius_weights,
    gpu_available,
)
from .geneloop.sequtil import translate
from .geneloop.stream import LocalFastqStream
from .geneloop.synthesis import SynthParams
from .reports import _BASE_CSS

_INSDC_RUN_RE = re.compile(r"^[SED]RR\d+$", re.IGNORECASE)
_LONG_READ_HINTS = ("long", "isoseq", "iso-seq", "pacbio", "hifi", "ont", "nanopore")

CLADE_CONFIGS: dict[str, dict[str, str]] = {
    "vertebrates": {"tiberius_model_cfg": "vertebrates", "compleasm_lineage": "vertebrata_odb10"},
    "mammalia_softmasking_v2": {"tiberius_model_cfg": "mammalia_softmasking_v2", "compleasm_lineage": "mammalia_odb10"},
    "mammalia_nosofttmasking_v2": {"tiberius_model_cfg": "mammalia_nosofttmasking_v2", "compleasm_lineage": "mammalia_odb10"},
    "insecta": {"tiberius_model_cfg": "insecta", "compleasm_lineage": "insecta_odb10"},
    "angiosperms": {"tiberius_model_cfg": "angiosperms", "compleasm_lineage": "embryophyta_odb10"},
    "fungi": {"tiberius_model_cfg": "fungi", "compleasm_lineage": "fungi_odb10"},
    "chlorophyta": {"tiberius_model_cfg": "chlorophyta", "compleasm_lineage": "chlorophyta_odb10"},
    "diatoms": {"tiberius_model_cfg": "diatoms", "compleasm_lineage": "stramenopiles_odb10"},
}


def prebuild_geneloop_native_libs() -> dict[str, str]:
    """Compile and load GeneLoop C/OpenMP native executables and shared libraries."""
    return _prebuild_gl_native()


def _esc(val: Any) -> str:
    return html.escape("" if val is None else str(val))


def _sanitize_run_id(raw: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (raw or "").strip()).strip("._")
    return cleaned or fallback


def _resolve_rna_manifest(manifest_path: str | Path) -> tuple[list[str], list[str], dict[str, str]]:
    """Parse an INSDC SRA/ENA RNA-seq manifest TSV/CSV or plain 1-column accession list."""
    text = Path(manifest_path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return [], [], {}

    first = lines[0]
    has_delim = ("\t" in first) or ("," in first)
    if not has_delim and _INSDC_RUN_RE.match(first):
        short_runs: list[str] = []
        for ln in lines:
            tok = ln.split()[0].strip()
            if _INSDC_RUN_RE.match(tok):
                short_runs.append(tok.upper())
        return list(dict.fromkeys(short_runs)), [], {}

    delim = "\t" if "\t" in first else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
    short_runs = []
    long_runs: list[str] = []
    run_orgs: dict[str, str] = {}

    for row in reader:
        norm = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        acc = ""
        for col in ("run_accession", "accession", "run_id", "run_or_url", "srr", "err", "drr", "run"):
            if norm.get(col):
                acc = norm[col]
                break
        if not acc:
            continue
        if not _INSDC_RUN_RE.match(acc):
            raise ValueError(f"RNA manifest accession {acc!r} is not a valid INSDC run accession (SRR/ERR/DRR...)")
        acc = acc.upper()
        rtype = " ".join(
            norm.get(c, "") for c in ("read_type", "type", "platform", "library_strategy", "instrument", "layout")
        ).lower()
        if any(h in rtype for h in _LONG_READ_HINTS):
            long_runs.append(acc)
        else:
            short_runs.append(acc)
        org = norm.get("organism") or norm.get("species") or norm.get("scientific_name") or ""
        if org:
            run_orgs[acc] = org

    return list(dict.fromkeys(short_runs)), list(dict.fromkeys(long_runs)), run_orgs


def resolve_execution_mode(
    requested_mode: str,
    clade: str,
    tiberius_weights: str | Path | None = None,
    has_proteins: bool = True,
) -> dict[str, Any]:
    """Resolve requested execution mode against available GPU, Tiberius runtime, and weights."""
    if requested_mode not in ("auto", "fast_evidence", "cpu_tiberius", "gpu_tiberius"):
        raise ValueError(f"Unsupported mode {requested_mode!r}; expected auto, fast_evidence, cpu_tiberius, or gpu_tiberius")

    cfg = CLADE_CONFIGS.get(clade, CLADE_CONFIGS["vertebrates"])["tiberius_model_cfg"]
    if requested_mode == "fast_evidence":
        return {
            "effective_mode": "fast_evidence",
            "skip_tiberius": True,
            "cpu_only_tiberius": False,
            "cpu_focaloop_tiberius": False,
            "tiberius_backend": "none",
            "weights_dir": None,
            "gpu_detected": False,
            "reason": "User selected Fast Evidence + Miniprot mode",
        }

    rt = detect_tiberius_runtime()
    wdir = discover_tiberius_weights(cfg, explicit_weights=tiberius_weights)
    has_gpu = gpu_available()
    has_runtime = rt.get("backend", "none") != "none"

    if requested_mode == "gpu_tiberius":
        if not has_gpu:
            raise RuntimeError("GPU Tiberius mode requested, but no usable NVIDIA GPU was detected (nvidia-smi unavailable or driver mismatch).")
        if not has_runtime:
            raise RuntimeError("GPU Tiberius mode requested, but no Tiberius runtime (tiberius.py+tensorflow, Apptainer SIF, or Docker) was found.")
        return {
            "effective_mode": "gpu_tiberius",
            "skip_tiberius": False,
            "cpu_only_tiberius": False,
            "cpu_focaloop_tiberius": False,
            "tiberius_backend": rt["backend"],
            "weights_dir": str(wdir) if wdir else None,
            "gpu_detected": True,
            "reason": "User selected GPU Tiberius mode",
        }

    if requested_mode == "cpu_tiberius":
        if not has_runtime:
            raise RuntimeError("CPU Tiberius mode requested, but no Tiberius runtime (tiberius.py+tensorflow, Apptainer SIF, or Docker) was found.")
        use_inv = bool(has_proteins)
        return {
            "effective_mode": "cpu_tiberius",
            "skip_tiberius": False,
            "cpu_only_tiberius": True,
            "cpu_focaloop_tiberius": use_inv,
            "tiberius_backend": rt["backend"],
            "weights_dir": str(wdir) if wdir else None,
            "gpu_detected": has_gpu,
            "reason": "User selected CPU Tiberius mode" + (" (window-zoomed inverted)" if use_inv else " (full-contig CPU)"),
        }

    # auto mode: GPU Tiberius -> CPU Tiberius (when weights cached locally) -> Fast Evidence+Miniprot
    if has_gpu and has_runtime and wdir is not None:
        return {
            "effective_mode": "gpu_tiberius",
            "skip_tiberius": False,
            "cpu_only_tiberius": False,
            "cpu_focaloop_tiberius": False,
            "tiberius_backend": rt["backend"],
            "weights_dir": str(wdir),
            "gpu_detected": True,
            "reason": "Auto-detected NVIDIA GPU, Tiberius runtime, and cached clade weights",
        }
    if has_runtime and wdir is not None:
        use_inv = bool(has_proteins)
        return {
            "effective_mode": "cpu_tiberius",
            "skip_tiberius": False,
            "cpu_only_tiberius": True,
            "cpu_focaloop_tiberius": use_inv,
            "tiberius_backend": rt["backend"],
            "weights_dir": str(wdir),
            "gpu_detected": has_gpu,
            "reason": "Auto-detected Tiberius runtime and cached weights (CPU mode)",
        }
    missing: list[str] = []
    if not has_runtime:
        missing.append("Tiberius runtime")
    if wdir is None:
        missing.append("Tiberius weights")
    if not has_gpu:
        missing.append("usable GPU")
    reason = (
        f"Falling back to Fast Evidence + Miniprot ({', '.join(missing)} unavailable)"
        if missing
        else "Running Fast Evidence + Miniprot mode cleanly"
    )
    return {
        "effective_mode": "fast_evidence",
        "skip_tiberius": True,
        "cpu_only_tiberius": False,
        "cpu_focaloop_tiberius": False,
        "tiberius_backend": rt.get("backend", "none"),
        "weights_dir": str(wdir) if wdir else None,
        "gpu_detected": has_gpu,
        "reason": reason,
    }


def _write_proteins_fasta(models: Sequence[Transcript], genome: Any, out_faa: Path) -> int:
    """Translate all coding transcript models to protein FASTA."""
    out_faa.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_faa, "w", encoding="utf-8") as fh:
        for t in models:
            if not t.cds:
                continue
            ph = int(getattr(t, "cds_phase", 0) or 0) % 3
            nt = genome.spliced(t.contig, t.strand, t.cds)[ph:]
            aa = translate(nt).rstrip("*")
            if aa:
                fh.write(f">{t.tid} gene={t.gid} loc={t.contig}:{t.start + 1}-{t.end}({t.strand})\n")
                for i in range(0, len(aa), 80):
                    fh.write(aa[i : i + 80] + "\n")
                n_written += 1
    return n_written


def _write_summary_tsv(
    out_summary: Path,
    *,
    clade: str,
    mode_info: dict[str, Any],
    genome_meta: dict[str, Any],
    seed_summary: dict[str, Any],
    arm_summary: dict[str, Any],
    biotypes: dict[str, Any],
    wall_s: float,
) -> None:
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    synth_stats = arm_summary.get("synth_stats") or {}
    filt_stats = arm_summary.get("sieve") or {}
    judge_final = arm_summary.get("judge_final") or {}
    rows = [
        ("engine", "focaloop_geneloop"),
        ("version", GENELOOP_VERSION),
        ("clade", clade),
        ("effective_mode", mode_info["effective_mode"]),
        ("mode_reason", mode_info["reason"]),
        ("tiberius_backend", mode_info.get("tiberius_backend", "none")),
        ("wall_seconds", f"{wall_s:.2f}"),
        ("n_contigs", str(genome_meta.get("n_contigs", 0))),
        ("genome_bp", str(genome_meta.get("genome_bp", 0))),
        ("organelle_contigs", str(len(genome_meta.get("organelle_contigs", [])))),
        ("seed_loci_total", str(seed_summary.get("n_loci", 0))),
        ("seed_loci_locked", str(seed_summary.get("LOCKED_HIGH_CONF", 0))),
        ("seed_loci_unsettled", str(seed_summary.get("UNSETTLED_LOCI", 0))),
        ("total_genes", str(biotypes.get("n_genes", 0))),
        ("total_transcripts", str(biotypes.get("n_transcripts", 0))),
        ("coding_genes", str(biotypes.get("n_coding_genes", 0))),
        ("coding_transcripts", str(biotypes.get("n_coding_transcripts", 0))),
        ("lncrna_genes", str(biotypes.get("n_lnc_genes", 0))),
        ("structural_ncrna_genes", str(synth_stats.get("n_structural_ncrna_genes", 0))),
        ("pseudogenes", str(biotypes.get("n_pseudo_genes", 0))),
        ("microexon_models", str(synth_stats.get("n_microexon_models", 0))),
        ("single_exon_coding_genes", str(synth_stats.get("n_single_exon_genes", 0))),
        ("retrocopies_filtered", str(synth_stats.get("retro_rejected", 0))),
        ("streaming_waves", str(arm_summary.get("waves", 0))),
        ("streaming_stop_reason", str(arm_summary.get("stop_reason", "NONE"))),
        ("spots_streamed", str(arm_summary.get("spots_streamed_all", 0))),
        ("total_spots_available", str(arm_summary.get("total_spots_all", 0))),
        ("pct_archive_streamed", f"{float(arm_summary.get('pct_archive_streamed', 0.0)):.2f}"),
        ("depletion_filter_pairs_in", str(filt_stats.get("pairs_in", 0))),
        ("depletion_filter_pairs_bypassed_locus", str(filt_stats.get("pairs_dropped_locus", 0))),
        ("depletion_filter_pairs_bypassed_bait", str(filt_stats.get("pairs_dropped_bait", 0))),
        ("depletion_filter_bypass_pct", f"{float(filt_stats.get('bypass_pct', 0.0)):.2f}"),
        ("good_turing_unsettled_p0", f"{float(judge_final.get('unsettled_p0', 0.0)):.5f}" if judge_final else "NA"),
        ("saturated_locked_loci", str(arm_summary.get("saturated_loci", 0))),
    ]
    with open(out_summary, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["metric", "value"])
        w.writerows(rows)


def _svg_biotype_bars(biotypes: dict[str, Any], synth_stats: dict[str, Any]) -> str:
    snc = synth_stats.get("n_structural_ncrna_by_biotype") or {}
    bars = [
        ("Protein-coding", int(biotypes.get("n_coding_genes", 0)), "#0f766e"),
        ("lncRNA", int(biotypes.get("n_lnc_genes", 0)), "#1d4ed8"),
        ("tRNA", int(snc.get("tRNA", 0)), "#7c3aed"),
        ("rRNA", int(snc.get("rRNA", 0)), "#0284c7"),
        ("snRNA / snoRNA", int(snc.get("snRNA", 0)) + int(snc.get("snoRNA", 0)), "#059669"),
        ("Pseudogenes", int(biotypes.get("n_pseudo_genes", 0)), "#b45309"),
    ]
    w, h = 500, 210
    max_v = max([v for _, v, _ in bars] + [1])
    rows_svg: list[str] = []
    for i, (label, val, col) in enumerate(bars):
        y = 18 + i * 30
        bw = (val / max_v) * 250.0
        rows_svg.append(
            f'<text x="132" y="{y + 13}" text-anchor="end" font-size="11.5" fill="#334155">{_esc(label)}</text>'
            f'<rect x="142" y="{y}" width="{bw:.1f}" height="18" rx="4" fill="{col}"/>'
            f'<text x="{148 + bw:.1f}" y="{y + 13}" font-size="11.5" font-weight="600" fill="#0f172a">{val:,}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">{"".join(rows_svg)}</svg>'


def _svg_evidence_sources(synth_stats: dict[str, Any]) -> str:
    by_src = synth_stats.get("selected_by_source") or {}
    if not by_src:
        return "<p>No coding locus source breakdown recorded.</p>"
    items = sorted(by_src.items(), key=lambda kv: (-kv[1], kv[0]))[:7]
    w, h = 500, max(120, 24 + len(items) * 28)
    max_v = max([int(v) for _, v in items] + [1])
    rows_svg: list[str] = []
    for i, (src, cnt) in enumerate(items):
        y = 16 + i * 26
        bw = (int(cnt) / max_v) * 220.0
        rows_svg.append(
            f'<text x="190" y="{y + 12}" text-anchor="end" font-size="11" fill="#334155">{_esc(src)}</text>'
            f'<rect x="198" y="{y}" width="{bw:.1f}" height="16" rx="4" fill="#1d4ed8"/>'
            f'<text x="{204 + bw:.1f}" y="{y + 12}" font-size="11" font-weight="600" fill="#0f172a">{int(cnt):,}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">{"".join(rows_svg)}</svg>'


def render_geneloop_report(
    out_report: Path,
    *,
    clade: str,
    mode_info: dict[str, Any],
    genome_meta: dict[str, Any],
    seed_summary: dict[str, Any],
    arm_summary: dict[str, Any],
    biotypes: dict[str, Any],
    wall_s: float,
) -> None:
    """Render self-contained HTML + inline SVG telemetry report for Focaloop GeneLoop."""
    out_report.parent.mkdir(parents=True, exist_ok=True)
    synth_stats = arm_summary.get("synth_stats") or {}
    filt_stats = arm_summary.get("sieve") or {}
    judge_final = arm_summary.get("judge_final") or {}
    lib_state = arm_summary.get("library_state") or {}

    mb_genome = float(genome_meta.get("genome_bp", 0)) / 1e6
    pct_streamed = float(arm_summary.get("pct_archive_streamed", 0.0))
    bypass_pct = float(filt_stats.get("bypass_pct", 0.0))
    p0_val = judge_final.get("unsettled_p0")
    p0_str = f"{float(p0_val):.4f}" if p0_val is not None else "—"

    lib_rows_html = "".join(
        "<tr>"
        f"<td><code>{_esc(rid)}</code></td>"
        f"<td class=\"num\">{int(st.get('waves', 0))}</td>"
        f"<td class=\"num\">{int(st.get('spots', 0)):,}</td>"
        f"<td><span class=\"badge {'b-amber' if st.get('halted') else 'b-green'}\">{'Halted' if st.get('halted') else 'Active'}</span></td>"
        f"<td>{_esc(st.get('reason') or 'Completed')}</td>"
        "</tr>"
        for rid, st in sorted(lib_state.items())
    ) or "<tr><td colspan=\"5\">No adaptive streaming libraries executed (seed / local BAM mode).</td></tr>"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Focaloop GeneLoop Annotation Report</title>
{_BASE_CSS}
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Focaloop GeneLoop — Adaptive Eukaryotic Structural Annotation</h1>
    <p>Clade preset: <b>{_esc(clade)}</b> &middot; Execution mode: <b>{_esc(mode_info['effective_mode'])}</b> ({_esc(mode_info['reason'])}) &middot; Assembly: <b>{mb_genome:.2f} Mb</b> across {int(genome_meta.get('n_contigs', 0)):,} contigs &middot; Wall time: <b>{wall_s:.1f} s</b></p>
  </div>

  <div class="kpis">
    <div class="kpi">
      <div class="label">Protein-Coding Genes</div>
      <div class="val">{int(biotypes.get('n_coding_genes', 0)):,}</div>
      <div class="sub">{int(biotypes.get('n_coding_transcripts', 0)):,} coding transcripts ({int(synth_stats.get('n_with_utr', 0)):,} with UTRs)</div>
    </div>
    <div class="kpi">
      <div class="label">Non-Coding &amp; Pseudogenes</div>
      <div class="val">{int(biotypes.get('n_ncrna_genes', 0)) + int(biotypes.get('n_pseudo_genes', 0)):,}</div>
      <div class="sub">{int(biotypes.get('n_lnc_genes', 0)):,} lncRNA &middot; {int(synth_stats.get('n_structural_ncrna_genes', 0)):,} structural ncRNA &middot; {int(biotypes.get('n_pseudo_genes', 0)):,} pseudogene</div>
    </div>
    <div class="kpi">
      <div class="label">RNA Evidence Streamed</div>
      <div class="val">{pct_streamed:.1f}%</div>
      <div class="sub">{int(arm_summary.get('spots_streamed_all', 0)):,} / {int(arm_summary.get('total_spots_all', 0)):,} spots &middot; Stop: {_esc(arm_summary.get('stop_reason', 'NONE'))}</div>
    </div>
    <div class="kpi">
      <div class="label">Syncmer Depletion Filter</div>
      <div class="val">{bypass_pct:.1f}%</div>
      <div class="sub">{int(filt_stats.get('pairs_dropped_locus', 0)):,} saturated-locus + {int(filt_stats.get('pairs_dropped_bait', 0)):,} rRNA/organelle pairs bypassed</div>
    </div>
    <div class="kpi">
      <div class="label">Good-Turing Unsettled p&#8320;</div>
      <div class="val">{_esc(p0_str)}</div>
      <div class="sub">{int(arm_summary.get('saturated_loci', 0)):,} / {int(arm_summary.get('locked_loci', 0)):,} locked loci saturated</div>
    </div>
  </div>

  <div class="grid2">
    <div class="section">
      <h2>Predicted Gene Biotypes</h2>
      {_svg_biotype_bars(biotypes, synth_stats)}
    </div>
    <div class="section">
      <h2>Coding Locus Evidence Attribution</h2>
      {_svg_evidence_sources(synth_stats)}
    </div>
  </div>

  <div class="section">
    <h2>Stage Partition &amp; Biological Guardrails</h2>
    <table>
      <thead><tr><th>Metric</th><th class="num">Count</th><th>Interpretation</th></tr></thead>
      <tbody>
        <tr><td>Seed loci (high-confidence locked)</td><td class="num">{int(seed_summary.get('LOCKED_HIGH_CONF', 0)):,}</td><td>Multi-exon conserved models protected from paralog depletion</td></tr>
        <tr><td>Seed loci (unsettled / candidate)</td><td class="num">{int(seed_summary.get('UNSETTLED_LOCI', 0)):,}</td><td>Targeted for Good-Turing intron discovery &amp; splice refinement</td></tr>
        <tr><td>Intronless protein-coding genes</td><td class="num">{int(synth_stats.get('n_single_exon_genes', 0)):,}</td><td>Intact single-exon genes retained (e.g., ORs, histones)</td></tr>
        <tr><td>Processed retrocopies filtered</td><td class="num">{int(synth_stats.get('retro_rejected', 0)):,}</td><td>Intronless multi-exon retrocopies separated from true coding genes</td></tr>
        <tr><td>Micro-exon rescued models</td><td class="num">{int(synth_stats.get('n_microexon_models', 0)):,}</td><td>3&ndash;27 nt internal exons recovered via splice-gap scan</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>RNA-Seq Library Streaming &amp; Convergence</h2>
    <table>
      <thead><tr><th>Library / Run</th><th class="num">Waves</th><th class="num">Spots Consumed</th><th>Status</th><th>Stop / Outcome</th></tr></thead>
      <tbody>{lib_rows_html}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    out_report.write_text(html_doc, encoding="utf-8")


def run_geneloop_galaxy(
    *,
    genome_fasta: str | Path,
    out_gff3: str | Path,
    out_proteins: str | Path,
    out_gtf: str | Path,
    out_summary: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    proteins_fasta: str | Path | None = None,
    exclude_genus: str = "",
    clade: str = "vertebrates",
    mode: str = "auto",
    tiberius_weights: str | Path | None = None,
    rna_manifest: str | Path | None = None,
    short_reads_1: Sequence[str | Path] | None = None,
    short_reads_2: Sequence[str | Path] | None = None,
    short_read_ids: Sequence[str] | None = None,
    long_reads: Sequence[str | Path] | None = None,
    long_read_ids: Sequence[str] | None = None,
    short_bams: Sequence[str | Path] | None = None,
    long_bams: Sequence[str | Path] | None = None,
    logan_mode: str = "auto",
    max_frac_spots: float = 0.35,
    p0_stop: float = 0.005,
    wave_spots: int = 1_000_000,
    structural_ncrna: str = "auto",
    rfam_dir: str | Path | None = None,
    enable_zoom: bool = True,
    threads: int = 8,
) -> dict[str, Any]:
    """Execute the 5-stage GeneLoop annotation pipeline and emit Galaxy outputs + warm state."""
    t_start = time.time()
    clade_meta = CLADE_CONFIGS.get(clade)
    if clade_meta is None:
        raise ValueError(f"Unsupported clade {clade!r}; expected one of {sorted(CLADE_CONFIGS)}")

    genome_p = Path(genome_fasta).resolve()
    if not genome_p.is_file():
        raise FileNotFoundError(f"Target genome FASTA not found: {genome_fasta}")

    prot_p: Path | None = None
    if proteins_fasta:
        prot_p = Path(proteins_fasta).resolve()
        if not prot_p.is_file():
            raise FileNotFoundError(f"Homology protein FASTA not found: {proteins_fasta}")

    mode_info = resolve_execution_mode(
        mode, clade, tiberius_weights=tiberius_weights, has_proteins=(prot_p is not None)
    )
    print(f"[focaloop-geneloop] mode={mode_info['effective_mode']}: {mode_info['reason']}", flush=True)

    sra_short: list[str] = []
    sra_long: list[str] = []
    run_orgs: dict[str, str] = {}
    if rna_manifest:
        sra_short, sra_long, run_orgs = _resolve_rna_manifest(rna_manifest)

    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="focaloop_geneloop_"))
    prev_rfam_env = os.environ.get("GENELOOP_RFAM_DIR")
    if rfam_dir:
        os.environ["GENELOOP_RFAM_DIR"] = str(Path(rfam_dir).resolve())

    try:
        gate_dict: dict[str, Any] = {
            "gate_id": "galaxy_run",
            "fasta_path": str(genome_p),
            "proteins_path": str(prot_p) if prot_p else "",
            "exclude_genus": exclude_genus.strip(),
            "species": clade,
            "tiberius_model_cfg": clade_meta["tiberius_model_cfg"],
            "compleasm_lineage": clade_meta["compleasm_lineage"],
            "tiberius_weights": str(mode_info["weights_dir"]) if mode_info.get("weights_dir") else (str(tiberius_weights) if tiberius_weights else ""),
            "sra_short_runs": sra_short,
            "sra_long_runs": sra_long,
            "sra_run_organisms": run_orgs,
        }

        eff_threads = max(1, int(threads))
        lp = LoopParams(
            seed_mode="full" if mode_info["skip_tiberius"] else "targeted",
            wave_spots=max(1_000, int(wave_spots)),
            p0_stop=float(p0_stop),
            max_frac_spots=float(max_frac_spots),
            threads=eff_threads,
            hisat_threads=eff_threads,
            cpu_only_tiberius=bool(mode_info["cpu_only_tiberius"]),
            cpu_focaloop_tiberius=bool(mode_info["cpu_focaloop_tiberius"]),
            enable_structural_ncrna=structural_ncrna,
            logan_mode=logan_mode,
        )
        sp = SynthParams()

        gr = GateRun("galaxy_run", root=work_root, loop=lp, synth=sp, gate=gate_dict)
        gr._skip_tiberius = bool(mode_info["skip_tiberius"])
        gr._enable_zoom = bool(enable_zoom)

        # Register local History short-read FASTQs (paired or single-end)
        sr1_list = [Path(p).resolve() for p in (short_reads_1 or []) if str(p).strip()]
        sr2_list = [Path(p).resolve() for p in (short_reads_2 or []) if str(p).strip()]
        if sr2_list and len(sr2_list) != len(sr1_list):
            raise ValueError(f"--short-reads-2 count ({len(sr2_list)}) must match --short-reads-1 count ({len(sr1_list)})")
        for idx_s, r1p in enumerate(sr1_list):
            raw_id = (
                short_read_ids[idx_s]
                if (short_read_ids and idx_s < len(short_read_ids))
                else r1p.stem
            )
            rid = _sanitize_run_id(raw_id, f"short_{idx_s + 1}")
            while rid in gr._local_streams:
                rid = f"{rid}_{idx_s + 1}"
            r2p = sr2_list[idx_s] if sr2_list else None
            st = LocalFastqStream(rid, r1p, r2p)
            gr._local_streams[rid] = {"stream": st, "long_read": False, "paired": r2p is not None}

        # Register local History long-read FASTQs
        lr_list = [Path(p).resolve() for p in (long_reads or []) if str(p).strip()]
        for idx_l, lrp in enumerate(lr_list):
            raw_id = (
                long_read_ids[idx_l]
                if (long_read_ids and idx_l < len(long_read_ids))
                else lrp.stem
            )
            rid = _sanitize_run_id(raw_id, f"long_{idx_l + 1}")
            while rid in gr._local_streams:
                rid = f"{rid}_{idx_l + 1}"
            st = LocalFastqStream(rid, lrp, None)
            gr._local_streams[rid] = {"stream": st, "long_read": True, "paired": False}

        # Register local pre-aligned History BAMs
        for idx_b, bp in enumerate([Path(p).resolve() for p in (short_bams or []) if str(p).strip()]):
            gr._local_bams.append({"run_id": _sanitize_run_id(bp.stem, f"sbam_{idx_b + 1}"), "bam": bp, "long_read": False})
        for idx_b, bp in enumerate([Path(p).resolve() for p in (long_bams or []) if str(p).strip()]):
            gr._local_bams.append({"run_id": _sanitize_run_id(bp.stem, f"lbam_{idx_b + 1}"), "bam": bp, "long_read": True})

        try:
            seed_summary = gr.seed()
            has_rna = bool(sra_short or sra_long or gr._local_streams or gr._local_bams)
            if has_rna:
                arm_summary = gr.arm_c()
                raw_gff3 = gr.dir / f"arm{gr.c_tag}" / f"arm{gr.c_tag}.gff3"
                ev_dir = gr.dir / f"arm{gr.c_tag}" / "state"
            else:
                arm_a_res = gr.arm_a()
                raw_gff3 = Path(arm_a_res["gff3"])
                a_rec = gr.log.latest("synth_armA") or {}
                arm_summary = {
                    "waves": 0,
                    "stop_reason": "SEED_ONLY_NO_RNA",
                    "spots_streamed_all": 0,
                    "total_spots_all": 0,
                    "pct_archive_streamed": 0.0,
                    "sieve": {"pairs_in": 0, "pairs_dropped_locus": 0, "pairs_dropped_bait": 0, "bypass_pct": 0.0},
                    "saturated_loci": 0,
                    "locked_loci": seed_summary.get("LOCKED_HIGH_CONF", 0),
                    "synth_stats": a_rec.get("extra") or {},
                    "library_state": {},
                }
                ev_dir = None
        finally:
            gr.close()
            for v in gr._local_streams.values():
                v["stream"].close()

        models = read_models(raw_gff3, fmt="gff3") if raw_gff3.exists() else []
        biotypes = count_models_by_biotype(models)

        out_gff3_p = Path(out_gff3)
        out_gff3_p.parent.mkdir(parents=True, exist_ok=True)
        write_gff3(models, out_gff3_p, source="GeneLoop")

        out_gtf_p = Path(out_gtf)
        out_gtf_p.parent.mkdir(parents=True, exist_ok=True)
        write_gtf(models, out_gtf_p, cds_only=False, source="GeneLoop")

        out_prot_p = Path(out_proteins)
        n_proteins_out = _write_proteins_fasta(models, gr.genome, out_prot_p)

        wall_s = round(time.time() - t_start, 2)
        _write_summary_tsv(
            Path(out_summary),
            clade=clade,
            mode_info=mode_info,
            genome_meta=gr.meta,
            seed_summary=seed_summary,
            arm_summary=arm_summary,
            biotypes=biotypes,
            wall_s=wall_s,
        )
        render_geneloop_report(
            Path(out_report),
            clade=clade,
            mode_info=mode_info,
            genome_meta=gr.meta,
            seed_summary=seed_summary,
            arm_summary=arm_summary,
            biotypes=biotypes,
            wall_s=wall_s,
        )

        # Persist portable Zstd-Parquet state bundle into state_dir
        if ev_dir is not None and ev_dir.is_dir():
            for fname in ("loci.parquet", "junctions.parquet", "telemetry.parquet", "bins.npz"):
                src_f = ev_dir / fname
                if src_f.is_file():
                    shutil.copy2(src_f, state_root / fname)
        state_meta = {
            "engine": "geneloop",
            "version": GENELOOP_VERSION,
            "blueprint_id": "geneloop",
            "clade": clade,
            "effective_mode": mode_info["effective_mode"],
            "genome_meta": gr.meta,
            "seed_summary": seed_summary,
            "biotypes": biotypes,
            "proteins_emitted": n_proteins_out,
            "wall_seconds": wall_s,
            "stop_reason": arm_summary.get("stop_reason", ""),
        }
        write_json(state_root / STATE_META, state_meta)
        if state_archive:
            pack_state(state_root, state_archive)
        return state_meta
    finally:
        if prev_rfam_env is None:
            os.environ.pop("GENELOOP_RFAM_DIR", None)
        else:
            os.environ["GENELOOP_RFAM_DIR"] = prev_rfam_env
        shutil.rmtree(work_root, ignore_errors=True)
