"""Self-contained HTML + inline SVG telemetry and surveillance reports for Galaxy."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_BASE_CSS = """
<style>
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --ink: #0f172a;
    --muted: #475569;
    --border: #e2e8f0;
    --accent: #0f766e;
    --accent-soft: #ccfbf1;
    --blue: #1d4ed8;
    --blue-soft: #dbeafe;
    --amber: #b45309;
    --amber-soft: #fef3c7;
    --rose: #be123c;
    --rose-soft: #ffe4e6;
    --green: #15803d;
    --green-soft: #dcfce7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.45; font-size: 14px;
  }
  .wrap { max-width: 1240px; margin: 0 auto; }
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
    color: #fff; border-radius: 12px; padding: 22px 26px; margin-bottom: 18px;
  }
  .hero h1 { margin: 0 0 6px 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
  .hero p { margin: 0; color: #cbd5e1; font-size: 13.5px; }
  .kpis {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 18px;
  }
  .kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  }
  .kpi .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .kpi .val { font-size: 22px; font-weight: 700; margin-top: 4px; color: var(--ink); }
  .kpi .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .section {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 16px;
  }
  .section h2 {
    margin: 0 0 10px 0; font-size: 16px; font-weight: 650;
    border-bottom: 1px solid var(--border); padding-bottom: 8px;
  }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--border); }
  th { background: #f1f5f9; font-weight: 600; color: #334155; }
  tr:hover td { background: #f8fafc; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
  }
  .b-teal { background: var(--accent-soft); color: var(--accent); }
  .b-blue { background: var(--blue-soft); color: var(--blue); }
  .b-amber { background: var(--amber-soft); color: var(--amber); }
  .b-rose { background: var(--rose-soft); color: var(--rose); }
  .b-green { background: var(--green-soft); color: var(--green); }
  .bar-bg { width: 120px; height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; display: inline-block; vertical-align: middle; }
  .bar-fg { height: 100%; background: var(--accent); }
  code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
"""


def _esc(val: Any) -> str:
    return html.escape("" if val is None else str(val))


def _fmt_num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
        if not np.isfinite(fv):
            return "—"
        if abs(fv) > 0 and (abs(fv) < 1e-3 or abs(fv) >= 1e5):
            return f"{fv:.2e}"
        return f"{fv:.{digits}f}"
    except Exception:
        return _esc(v)


def _svg_convergence(telemetry: list[dict[str, Any]], tech_stop: float, dlfc_stop: float) -> str:
    if not telemetry:
        return "<p>No iterative telemetry points recorded.</p>"
    w, h = 500, 220
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 36
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    xs = [float(t.get("pct_archive", i + 1)) for i, t in enumerate(telemetry)]
    vtech = [float(t.get("median_tech_var_fraction", 0.0) or 0.0) for t in telemetry]
    dlfc = [float(t.get("median_abs_delta_lfc", 0.0) or 0.0) for t in telemetry]
    xmax = max(max(xs), 1e-3) * 1.08
    ymax = max(max(vtech + dlfc + [tech_stop, dlfc_stop, 0.4]), 0.1) * 1.1

    def px(x: float) -> float:
        return pad_l + (x / xmax) * pw

    def py(y: float) -> float:
        return pad_t + ph - (min(max(y, 0.0), ymax) / ymax) * ph

    pts_vt = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, vtech))
    pts_dl = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, dlfc))
    circles_vt = "".join(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="#0f766e"/>' for x, y in zip(xs, vtech))
    circles_dl = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="#1d4ed8"/>'
        for x, y in zip(xs, dlfc) if np.isfinite(y) and y > 0
    )
    y_stop = py(tech_stop)
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Focaloop convergence curve">
      <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1"/>
      <line x1="{pad_l}" y1="{y_stop:.1f}" x2="{pad_l + pw}" y2="{y_stop:.1f}" stroke="#be123c" stroke-dasharray="4,3"/>
      <text x="{pad_l + pw - 4}" y="{max(pad_t + 12, y_stop - 4):.1f}" text-anchor="end" font-size="10" fill="#be123c">V_tech stop ({tech_stop:.2f})</text>
      <polyline fill="none" stroke="#0f766e" stroke-width="2.2" points="{pts_vt}"/>
      <polyline fill="none" stroke="#1d4ed8" stroke-width="2.0" stroke-dasharray="3,2" points="{pts_dl}"/>
      {circles_vt}
      {circles_dl}
      <text x="{pad_l + pw / 2:.1f}" y="{h - 6}" text-anchor="middle" font-size="11" fill="#334155">% of Archive Streamed (max {max(xs):.2f}%)</text>
      <text x="14" y="{pad_t + ph / 2:.1f}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 14 {pad_t + ph / 2:.1f})">Metric</text>
      <rect x="{pad_l + 8}" y="{pad_t + 6}" width="10" height="3" fill="#0f766e"/>
      <text x="{pad_l + 22}" y="{pad_t + 10}" font-size="10.5" fill="#0f172a">Median V_tech / (V_tech + V_bio)</text>
      <rect x="{pad_l + 8}" y="{pad_t + 20}" width="10" height="3" fill="#1d4ed8"/>
      <text x="{pad_l + 22}" y="{pad_t + 24}" font-size="10.5" fill="#0f172a">Median |Δlog2FC|</text>
    </svg>
    """


def _svg_volcano(de_df: pd.DataFrame) -> str:
    if de_df.empty or "log2FoldChange" not in de_df.columns or "padj" not in de_df.columns:
        return "<p>No DE points available.</p>"
    d = de_df.dropna(subset=["log2FoldChange", "padj"]).copy()
    if d.empty:
        return "<p>No finite DE points available.</p>"
    w, h = 500, 220
    pad_l, pad_r, pad_t, pad_b = 46, 18, 18, 34
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    lfc = d["log2FoldChange"].to_numpy(dtype=float)
    mlogp = -np.log10(np.clip(d["padj"].to_numpy(dtype=float), 1e-300, 1.0))
    mlogp = np.minimum(mlogp, 80.0)
    xmax = max(float(np.percentile(np.abs(lfc), 99.5)) * 1.15, 2.0)
    ymax = max(float(mlogp.max()) * 1.08, 3.0)
    if len(d) > 1200:
        sig_idx = np.where((d["padj"].to_numpy(dtype=float) < 0.05) & (np.abs(lfc) >= 1.0))[0]
        bg_idx = np.where(~((d["padj"].to_numpy(dtype=float) < 0.05) & (np.abs(lfc) >= 1.0)))[0]
        rng = np.random.default_rng(42)
        if len(sig_idx) > 700:
            sig_idx = rng.choice(sig_idx, size=700, replace=False)
        if len(bg_idx) > 500:
            bg_idx = rng.choice(bg_idx, size=500, replace=False)
        keep_idx = np.concatenate([bg_idx, sig_idx])
    else:
        keep_idx = np.arange(len(d))

    dots = []
    for idx in keep_idx:
        x = np.clip(lfc[idx], -xmax, xmax)
        y = mlogp[idx]
        cx = pad_l + ((x + xmax) / (2 * xmax)) * pw
        cy = pad_t + ph - (y / ymax) * ph
        is_sig = (y >= -np.log10(0.05)) and (abs(x) >= 1.0)
        col = "#be123c" if (is_sig and x < 0) else ("#0f766e" if is_sig else "#94a3b8")
        rad = "2.6" if is_sig else "1.8"
        dots.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad}" fill="{col}" fill-opacity="0.78"/>')
    cx0 = pad_l + 0.5 * pw
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Volcano plot">
      <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1"/>
      <line x1="{cx0:.1f}" y1="{pad_t}" x2="{cx0:.1f}" y2="{pad_t + ph}" stroke="#cbd5e1" stroke-dasharray="3,3"/>
      {''.join(dots)}
      <text x="{pad_l + pw / 2:.1f}" y="{h - 6}" text-anchor="middle" font-size="11" fill="#334155">log2FoldChange (±{xmax:.1f})</text>
      <text x="14" y="{pad_t + ph / 2:.1f}" text-anchor="middle" font-size="11" fill="#334155" transform="rotate(-90 14 {pad_t + ph / 2:.1f})">-log10(padj, cap 80)</text>
    </svg>
    """


def render_panexpr_report(
    out_path: str | Path,
    *,
    contrast_str: str,
    winner: str,
    scoreboard: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    catalog_summary: dict[str, Any],
    loop_summary: dict[str, Any],
    de_df: pd.DataFrame,
    brand: str = "Focaloop ExpressionLoop",
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tot_arch_mb = float(loop_summary.get("total_archive_bytes", 0)) / 1e6
    streamed_mb = float(loop_summary.get("bytes_streamed_total", 0)) / 1e6
    pct_streamed = float(loop_summary.get("pct_archive_streamed", 0.0))
    n_deg = int((de_df["padj"].fillna(1.0) < 0.05).sum()) if "padj" in de_df.columns else 0
    n_bridge = int((de_df["ortholog_bridge"].fillna("") != "").sum()) if "ortholog_bridge" in de_df.columns else 0

    board_rows_html = []
    for b in scoreboard:
        kname = str(b.get("kernel", ""))
        is_win = kname == winner
        badge = '<span class="badge b-teal">WINNER</span>' if is_win else ""
        board_rows_html.append(
            f"<tr><td><code>{_esc(kname)}</code> {badge}</td>"
            f"<td class='num'>{_fmt_num(b.get('composite', b.get('score')), 3)}</td>"
            f"<td class='num'>{_fmt_num(b.get('synth_spearman', b.get('accuracy')), 3)}</td>"
            f"<td class='num'>{_fmt_num(b.get('tie_symmetry', b.get('symmetry')), 3)}</td>"
            f"<td class='num'>{_fmt_num(b.get('mapped_frac', b.get('real_mapped_fraction')), 3)}</td>"
            f"<td class='num'>{_fmt_num(b.get('reads_per_sec', b.get('throughput_rps')), 0)}</td></tr>"
        )

    sample_rows_html = []
    for r in per_sample_rows:
        pct = float(r.get("pct_streamed", 0.0))
        bar_w = min(max(pct, 1.0), 100.0)
        sample_rows_html.append(
            f"<tr><td><strong>{_esc(r.get('sample_id'))}</strong></td>"
            f"<td><span class='badge b-blue'>{_esc(r.get('condition'))}</span></td>"
            f"<td><code>{_esc(r.get('run_id'))}</code> ({_esc(r.get('layout'))})</td>"
            f"<td class='num'>{_fmt_num(r.get('archive_mb'), 1)} MB</td>"
            f"<td class='num'>{_fmt_num(r.get('streamed_mb'), 1)} MB</td>"
            f"<td class='num'><span class='bar-bg'><span class='bar-fg' style='width:{bar_w:.1f}%'></span></span> {pct:.2f}%</td>"
            f"<td class='num'>{100.0 - pct:.2f}%</td>"
            f"<td class='num'>{int(r.get('mapped_reads', 0)):,}</td></tr>"
        )

    top_de = de_df.dropna(subset=["padj"]).sort_values(["padj", "pvalue"], ascending=[True, True]).head(25)
    de_rows_html = []
    for _, row in top_de.iterrows():
        z_badge = '<span class="badge b-amber">zoomed</span>' if bool(row.get("zoomed")) else "—"
        de_rows_html.append(
            f"<tr><td><code>{_esc(row.get('gene_id'))}</code></td>"
            f"<td>{_esc(row.get('gene_name') or '—')}</td>"
            f"<td><code>{_esc(row.get('ortholog_bridge') or '—')}</code></td>"
            f"<td class='num'>{_fmt_num(row.get('baseMean'), 1)}</td>"
            f"<td class='num'><strong>{_fmt_num(row.get('log2FoldChange'), 3)}</strong></td>"
            f"<td class='num'>{_fmt_num(row.get('lfcSE_total'), 3)}</td>"
            f"<td class='num'>{_fmt_num(row.get('padj'), 3)}</td>"
            f"<td class='num'>{_fmt_num(row.get('placement_var_frac'), 3)}</td>"
            f"<td>{z_badge}</td></tr>"
        )

    zoom_info = loop_summary.get("zoom") or {}
    zoom_html = (
        f"<p><strong>Borderline groups baited:</strong> {len(zoom_info.get('groups', []))} &nbsp;|&nbsp; "
        f"<strong>Extra bytes scanned:</strong> {float(zoom_info.get('bytes_scanned', 0)) / 1e6:.2f} MB &nbsp;|&nbsp; "
        f"<strong>Reads scanned:</strong> {int(zoom_info.get('reads_scanned', 0)):,} &nbsp;|&nbsp; "
        f"<strong>Bait-matched reads mapped:</strong> {int(zoom_info.get('reads_mapped', 0)):,}</p>"
        if zoom_info
        else "<p>Borderline 31-mer zoom pass did not need additional reads (or disabled).</p>"
    )

    conv_svg = _svg_convergence(
        loop_summary.get("telemetry", []),
        float(loop_summary.get("v_tech_target", 0.25)),
        float(loop_summary.get("delta_lfc_stop", 0.05)),
    )
    volc_svg = _svg_volcano(de_df)

    is_logan = str(loop_summary.get("source_mode") or "") == "logan_unitigs"
    winner_sub = (
        f"Logan unitig streaming ({float(loop_summary.get('wire_reduction_x', 1.0)):.1f}x wire reduction)"
        if is_logan
        else f"{len(scoreboard)} aligners benchmarked on {int(loop_summary.get('tournament_reads', 0)):,} reads"
    )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{_esc(brand)} Report — {_esc(contrast_str)}</title>{_BASE_CSS}</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>{_esc(brand)} — Adaptive RNA-seq Differential Expression Report</h1>
    <p>Primary Contrast: <strong>{_esc(contrast_str)}</strong> &nbsp;·&nbsp; Source Mode: <code>{_esc(loop_summary.get('source_mode'))}</code> &nbsp;·&nbsp; Strandedness: <code>{_esc(loop_summary.get('strand_mode'))}</code></p>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">Archive Streamed</div><div class="val">{pct_streamed:.2f}%</div><div class="sub">{streamed_mb:.1f} MB of {tot_arch_mb:.1f} MB ({100.0 - pct_streamed:.1f}% skipped)</div></div>
    <div class="kpi"><div class="label">Aligner Engine</div><div class="val">{_esc(winner)}</div><div class="sub">{_esc(winner_sub)}</div></div>
    <div class="kpi"><div class="label">Significant DEGs</div><div class="val">{n_deg:,}</div><div class="sub">FDR &lt; 0.05 across {len(de_df):,} quantified genes</div></div>
    <div class="kpi"><div class="label">Ortholog Bridge</div><div class="val">{n_bridge:,}</div><div class="sub">Primary: {_esc(catalog_summary.get('primary_assembly'))} ({int(catalog_summary.get('n_targets', 0)):,} targets)</div></div>
    <div class="kpi"><div class="label">Loop Stopping</div><div class="val">{_esc(loop_summary.get('stop_reason'))}</div><div class="sub">{int(loop_summary.get('iterations', 0))} iterations · {float(loop_summary.get('wall_seconds', 0.0)):.1f}s</div></div>
  </div>

  <div class="section">
    <h2>Stage 1 — Regime Classifier &amp; Empirical Aligner Tournament</h2>
    <table>
      <thead><tr><th>Aligner Kernel</th><th class="num">Composite Score</th><th class="num">Synth Spearman</th><th class="num">Tie Symmetry</th><th class="num">Real Mapped Frac</th><th class="num">Reads / sec</th></tr></thead>
      <tbody>{''.join(board_rows_html) or '<tr><td colspan="6">Logan unitig engine selected directly.</td></tr>'}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Stage 2 — Per-Sample Streaming Summary</h2>
    <table>
      <thead><tr><th>Sample</th><th>Condition</th><th>Run / Layout</th><th class="num">Full Archive</th><th class="num">Streamed</th><th class="num">% Streamed</th><th class="num">% Skipped</th><th class="num">Mapped Reads</th></tr></thead>
      <tbody>{''.join(sample_rows_html)}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Stage 3 &amp; 4 — EM Conditioning, Convergence &amp; Volcano</h2>
    <p>Catalog conditioned across assemblies <code>{_esc(', '.join(catalog_summary.get('assemblies', [])))}</code> ({int(catalog_summary.get('n_groups', 0)):,} ortholog groups, assembly priors: <code>{_esc(loop_summary.get('assembly_prior'))}</code>).</p>
    <div class="grid2">
      <div>{conv_svg}</div>
      <div>{volc_svg}</div>
    </div>
  </div>

  <div class="section">
    <h2>Stage 5 — Targeted 31-mer Borderline Zoom</h2>
    {zoom_html}
  </div>

  <div class="section">
    <h2>Top 25 Differentially Expressed Genes ({_esc(contrast_str)})</h2>
    <table>
      <thead><tr><th>Gene ID</th><th>Gene Name</th><th>Ortholog Bridge</th><th class="num">baseMean</th><th class="num">log2FC</th><th class="num">lfcSE_total</th><th class="num">padj</th><th class="num">Placement Var Frac</th><th>Zoom</th></tr></thead>
      <tbody>{''.join(de_rows_html)}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out


def render_malariavar_report(
    out_path: str | Path,
    *,
    mode: str,
    drm_df: pd.DataFrame,
    summary_stats: dict[str, Any],
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_iso = len(drm_df)
    wire_mb = float(drm_df["wire_mb"].fillna(0).sum()) if "wire_mb" in drm_df.columns else 0.0
    arch_mb = float(drm_df["archive_mb"].fillna(0).sum()) if "archive_mb" in drm_df.columns else 0.0
    pct_arch = (100.0 * wire_mb / arch_mb) if arch_mb > 0 else 0.0
    n_poly = int((drm_df["clonality"] == "polyclonal").sum()) if "clonality" in drm_df.columns else 0
    n_mono = int((drm_df["clonality"] == "monoclonal").sum()) if "clonality" in drm_df.columns else 0
    med_sieve = float(drm_df["host_sieve_reject_pct"].dropna().median()) if (
        "host_sieve_reject_pct" in drm_df.columns and drm_df["host_sieve_reject_pct"].notna().any()
    ) else 0.0

    # Group prevalence summary
    prev_rows = []
    if not drm_df.empty and "country_or_group" in drm_df.columns:
        for grp, sub in drm_df.groupby("country_or_group", sort=False):
            n_g = len(sub)
            art_r = int((sub.get("who_artemisinin") == "Resistant").sum())
            cq_r = int((sub.get("who_chloroquine") == "Resistant").sum())
            ppq_r = int((sub.get("who_piperaquine") == "Resistant").sum())
            sp_r = int((sub.get("who_sp_uncomplicated") == "Resistant").sum())
            hrp2_d = int((sub.get("hrp2_deletion") == "deleted").sum())
            hrp3_d = int((sub.get("hrp3_deletion") == "deleted").sum())
            poly_g = int((sub.get("clonality") == "polyclonal").sum())
            prev_rows.append(
                f"<tr><td><strong>{_esc(grp)}</strong></td><td class='num'>{n_g}</td>"
                f"<td class='num'>{art_r}/{n_g} ({100*art_r/max(n_g,1):.0f}%)</td>"
                f"<td class='num'>{cq_r}/{n_g} ({100*cq_r/max(n_g,1):.0f}%)</td>"
                f"<td class='num'>{ppq_r}/{n_g} ({100*ppq_r/max(n_g,1):.0f}%)</td>"
                f"<td class='num'>{sp_r}/{n_g} ({100*sp_r/max(n_g,1):.0f}%)</td>"
                f"<td class='num'>{hrp2_d}/{n_g}</td>"
                f"<td class='num'>{hrp3_d}/{n_g}</td>"
                f"<td class='num'>{poly_g}/{n_g}</td></tr>"
            )

    iso_rows = []
    for _, r in drm_df.iterrows():
        pct = float(r.get("pct_archive") or 0.0)
        bar_w = min(max(pct, 1.0), 100.0) if np.isfinite(pct) else 0.0
        art = str(r.get("who_artemisinin", "Undetermined"))
        art_cls = "b-rose" if art == "Resistant" else ("b-green" if art == "Sensitive" else "b-amber")
        clon = str(r.get("clonality", "unknown"))
        clon_cls = "b-amber" if clon == "polyclonal" else "b-teal"
        iso_rows.append(
            f"<tr><td><code>{_esc(r.get('sample_id'))}</code></td>"
            f"<td>{_esc(r.get('country_or_group'))}</td>"
            f"<td><code>{_esc(r.get('mode'))}</code></td>"
            f"<td class='num'>{_fmt_num(r.get('wire_mb'), 1)} / {_fmt_num(r.get('archive_mb'), 0)} MB "
            f"<span class='bar-bg'><span class='bar-fg' style='width:{bar_w:.1f}%'></span></span></td>"
            f"<td class='num'>{_fmt_num(r.get('host_sieve_reject_pct'), 1)}%</td>"
            f"<td class='num'>{_fmt_num(r.get('median_depth'), 1)}x</td>"
            f"<td class='num'>{_fmt_num(r.get('fws'), 3)} <span class='badge {clon_cls}'>{_esc(clon)}</span></td>"
            f"<td><code>{_esc(r.get('kelch13_mutations'))}</code> <span class='badge {art_cls}'>{_esc(art)}</span></td>"
            f"<td><code>{_esc(r.get('crt_76'))}</code></td>"
            f"<td><code>{_esc(r.get('dhfr_hap'))}</code> / <code>{_esc(r.get('dhps_hap'))}</code></td>"
            f"<td>mdr1:{_esc(r.get('mdr1_cnv'))} pm2/3:{_esc(r.get('pm2_pm3_cnv'))} gch1:{_esc(r.get('gch1_cnv'))}</td>"
            f"<td>hrp2:{_esc(r.get('hrp2_deletion'))} hrp3:{_esc(r.get('hrp3_deletion'))}</td></tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Focaloop MalariaVar Surveillance Report</title>{_BASE_CSS}</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Focaloop MalariaVar — Plasmodium Variation &amp; WHO Drug-Resistance Surveillance</h1>
    <p>Streaming Mode: <code>{_esc(mode)}</code> &nbsp;·&nbsp; Reference: <code>Pf3D7 (PlasmoDB-54)</code> &nbsp;·&nbsp; Wall-clock: <strong>{float(summary_stats.get('wall_s', 0.0)):.1f}s</strong></p>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">Isolates Genotyped</div><div class="val">{n_iso}</div><div class="sub">{n_mono} monoclonal · {n_poly} polyclonal (F_WS &lt; 0.95)</div></div>
    <div class="kpi"><div class="label">Bytes Transferred</div><div class="val">{wire_mb:.1f} MB</div><div class="sub">{pct_arch:.2f}% of {arch_mb / 1000.0:.2f} GB full archives</div></div>
    <div class="kpi"><div class="label">Host Read Depletion</div><div class="val">{med_sieve:.1f}%</div><div class="sub">Median non-Plasmodium spots stripped on stream</div></div>
    <div class="kpi"><div class="label">Artemisinin Resistant</div><div class="val">{int((drm_df.get('who_artemisinin') == 'Resistant').sum())}/{n_iso}</div><div class="sub">WHO validated/candidate kelch13 349–726</div></div>
    <div class="kpi"><div class="label">hrp2 / hrp3 Deletions</div><div class="val">{int((drm_df.get('hrp2_deletion') == 'deleted').sum())} / {int((drm_df.get('hrp3_deletion') == 'deleted').sum())}</div><div class="sub">RDT diagnostic-escape surveillance</div></div>
  </div>

  <div class="section">
    <h2>Cohort WHO Resistance &amp; Diagnostic-Escape Prevalence by Group</h2>
    <table>
      <thead><tr><th>Country / Group</th><th class="num">N</th><th class="num">Artemisinin Res</th><th class="num">Chloroquine Res</th><th class="num">Piperaquine Res</th><th class="num">SP (uncomp) Res</th><th class="num">hrp2 Del</th><th class="num">hrp3 Del</th><th class="num">Polyclonal</th></tr></thead>
      <tbody>{''.join(prev_rows)}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Per-Isolate Streaming Telemetry, Clonality &amp; Drug-Resistance Genotypes</h2>
    <table>
      <thead><tr><th>Isolate</th><th>Group</th><th>Mode</th><th class="num">Wire / Archive</th><th class="num">Host Depletion</th><th class="num">Depth</th><th class="num">F_WS</th><th>kelch13 (WHO Art)</th><th>crt 76</th><th>dhfr / dhps</th><th>CNVs</th><th>Deletions</th></tr></thead>
      <tbody>{''.join(iso_rows)}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out


def render_query_report(
    out_path: str | Path,
    *,
    engine: str,
    contrast_str: str,
    verdict: dict[str, Any],
    table_df: pd.DataFrame,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview = table_df.head(30)
    cols = list(preview.columns)
    th_html = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    tr_html = []
    for _, row in preview.iterrows():
        tds = "".join(f"<td>{_fmt_num(row[c], 4) if isinstance(row[c], float) else _esc(row[c])}</td>" for c in cols)
        tr_html.append(f"<tr>{tds}</tr>")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Focaloop Query — {_esc(contrast_str)}</title>{_BASE_CSS}</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Focaloop Query — Instant Follow-Up Comparison</h1>
    <p>Source: <code>{_esc(engine)}</code> &nbsp;·&nbsp; Comparison: <strong>{_esc(contrast_str)}</strong> &nbsp;·&nbsp; Network Access: <span class="badge b-green">OFFLINE (0 bytes downloaded)</span></p>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">Network Bytes</div><div class="val">0 B</div><div class="sub">Computed from saved state bundle</div></div>
    <div class="kpi"><div class="label">Query Runtime</div><div class="val">{float(verdict.get('wall_seconds', 0.0)):.2f}s</div><div class="sub">Zero FASTQ or CRAM re-reading</div></div>
    <div class="kpi"><div class="label">Error Budget Remaining</div><div class="val">{_fmt_num(verdict.get('wealth_after'), 4)}</div><div class="sub">Allocated {_fmt_num(verdict.get('alpha_spent'), 4)} for this test</div></div>
    <div class="kpi"><div class="label">Summary Metric</div><div class="val">{_esc(verdict.get('headline_metric', 'ok'))}</div><div class="sub">{_esc(verdict.get('headline_detail', ''))}</div></div>
  </div>
  <div class="section">
    <h2>Top Query Results ({min(len(table_df), 30)} of {len(table_df):,} rows)</h2>
    <table>
      <thead><tr>{th_html}</tr></thead>
      <tbody>{''.join(tr_html)}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out
