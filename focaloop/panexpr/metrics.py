"""Stratified concordance metrics between DE arms and the standardized study JSON emitter."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DE_COLUMNS = ("gene_id", "base_mean", "log2fc", "lfc_se", "pvalue", "padj")
STRATA = (("lt_10", 0.0, 10.0), ("10_to_50", 10.0, 50.0), ("50_to_200", 50.0, 200.0), ("gt_200", 200.0, float("inf")))


def standardize_de(df: pd.DataFrame, gene_col: str, lfc_col: str, padj_col: str, base_col: str | None = None,
                   se_col: str | None = None, p_col: str | None = None, ci_low: str | None = None, ci_high: str | None = None) -> pd.DataFrame:
    out = pd.DataFrame({
        "gene_id": df[gene_col].astype(str).values,
        "base_mean": df[base_col].astype(float).values if base_col else np.nan,
        "log2fc": df[lfc_col].astype(float).values,
        "lfc_se": df[se_col].astype(float).values if se_col else np.nan,
        "pvalue": df[p_col].astype(float).values if p_col else np.nan,
        "padj": df[padj_col].astype(float).values,
    })
    if ci_low and ci_high:
        out["ci95_low"] = df[ci_low].astype(float).values
        out["ci95_high"] = df[ci_high].astype(float).values
    return out.drop_duplicates("gene_id").set_index("gene_id")


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _de_score(m: pd.DataFrame) -> np.ndarray:
    """Ranking score for DEG classification: -log10 p when informative, else |lfc/se|, else |lfc|."""
    for col in ("p_test", "padj_test"):
        if col in m and m[col].notna().sum() >= 3 and m[col].dropna().nunique() > 2:
            return -np.log10(np.clip(m[col].fillna(1.0).values, 1e-300, 1.0))
    if "se_test" in m and m["se_test"].notna().sum() >= 3 and (m["se_test"].dropna() > 0).all():
        return np.abs(m["lfc_test"].values / m["se_test"].values)
    return np.abs(m["lfc_test"].values)


def _block(m: pd.DataFrame, has_ci: bool) -> dict:
    n = int(len(m))
    if n < 3:
        return {"n_genes": n, "pearson_r": float("nan"), "spearman_rho": float("nan"), "rmse_lfc": float("nan"),
                "sign_agree": float("nan"), "deg_jaccard_fdr05": float("nan"), "deg_auroc": float("nan"),
                "ci95_coverage": float("nan"), "ref_lfc_in_ci95": float("nan")}
    x, y = m["lfc_test"].values, m["lfc_ref"].values
    pr = float(pearsonr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    sr = float(spearmanr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    big = np.abs(y) >= 0.5
    sign = float(np.mean(np.sign(x[big]) == np.sign(y[big]))) if big.any() else float("nan")
    t_sig = (m["padj_test"] < 0.05).fillna(False).values
    r_sig = (m["padj_ref"] < 0.05).fillna(False).values
    union = (t_sig | r_sig).sum()
    jacc = float((t_sig & r_sig).sum() / union) if union else float("nan")
    score = _de_score(m)
    auroc = _auroc(r_sig, score)
    cov = float("nan")
    if has_ci:
        lo, hi = m["ci_low"].values, m["ci_high"].values
        ok = ~np.isnan(lo) & ~np.isnan(hi)
        if ok.any():
            cov = float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))
    return {"n_genes": n, "pearson_r": pr, "spearman_rho": sr, "rmse_lfc": rmse, "sign_agree": sign,
            "deg_jaccard_fdr05": jacc, "deg_auroc": auroc, "ci95_coverage": cov, "ref_lfc_in_ci95": cov}


def compare_de(test: pd.DataFrame, ref: pd.DataFrame, min_base_mean: float = 0.0) -> dict:
    """Compare a test DE table against a reference DE table (both standardized, indexed by gene_id).

    Depth strata use the reference arm's baseMean.  Returns {"overall": {...}, "by_depth_stratum": {...},
    "gene_universe": {...}}.  Note: ``ref_lfc_in_ci95`` (aliased as ``ci95_coverage`` for schema compatibility)
    measures inclusion of the reference arm's point LFC when ``ref`` is a batch pipeline rather than simulation truth.
    """
    if isinstance(test.index, pd.RangeIndex) and "gene_id" in test.columns:
        test = test.set_index("gene_id")
    if isinstance(ref.index, pd.RangeIndex) and "gene_id" in ref.columns:
        ref = ref.set_index("gene_id")
    test_ids = set(test.index[test["log2fc"].notna()]) if "log2fc" in test.columns else set(test.index)
    ref_ids = set(ref.index[ref["log2fc"].notna()]) if "log2fc" in ref.columns else set(ref.index)
    u_union = len(test_ids | ref_ids)
    universe = {
        "n_test": len(test_ids),
        "n_ref": len(ref_ids),
        "n_shared": len(test_ids & ref_ids),
        "n_test_only": len(test_ids - ref_ids),
        "n_ref_only": len(ref_ids - test_ids),
        "jaccard_universe": float(len(test_ids & ref_ids) / u_union) if u_union else float("nan"),
    }
    j = test.merge(ref, how="inner", left_index=True, right_index=True, suffixes=("_test", "_ref"))
    m = pd.DataFrame({
        "lfc_test": j["log2fc_test"], "lfc_ref": j["log2fc_ref"], "padj_test": j["padj_test"], "padj_ref": j["padj_ref"],
        "p_test": j["pvalue_test"] if "pvalue_test" in j else np.nan, "base_ref": j["base_mean_ref"],
        "se_test": j["lfc_se_test"] if "lfc_se_test" in j else np.nan,
    })
    has_ci = "ci95_low" in test.columns
    if has_ci:
        # the reference may carry CIs too, in which case the merge suffixes the test columns
        lo = "ci95_low_test" if "ci95_low_test" in j.columns else "ci95_low"
        hi = "ci95_high_test" if "ci95_high_test" in j.columns else "ci95_high"
        m["ci_low"] = j[lo]
        m["ci_high"] = j[hi]
    m = m.dropna(subset=["lfc_test", "lfc_ref"])
    m = m[m["base_ref"].fillna(0) >= min_base_mean]
    out = {"overall": _block(m, has_ci), "by_depth_stratum": {}, "gene_universe": universe}
    for name, lo, hi in STRATA:
        sub = m[(m["base_ref"] >= lo) & (m["base_ref"] < hi)]
        out["by_depth_stratum"][name] = _block(sub, has_ci)
    return out


def _nan_to_none(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _nan_to_none(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def study_json_skeleton(study_id: str, citation: str, accession: str, organism: str, spliced: bool, layout: str,
                        n_runs: int, total_archive_bytes: int, primary_contrast: str) -> dict:
    arm = {"tool": "", "wall_clock_s": 0.0, "peak_disk_mb": 0.0, "bytes_read": 0, "parquet": ""}
    return {
        "study_id": study_id, "citation": citation, "accession": accession, "organism": organism, "spliced": spliced,
        "layout": layout, "n_runs": n_runs, "total_archive_bytes": int(total_archive_bytes), "primary_contrast": primary_contrast,
        "arms": {
            "arm_a_published": {"source": "", "n_reported_genes": 0, "notes": ""},
            "arm_b_100pct_align": copy.deepcopy(arm),
            "arm_c_100pct_pseudo": copy.deepcopy(arm),
            "arm_d_matched_byte_naive": {**copy.deepcopy(arm), "pct_archive": 0.0},
            "arm_e_panexpr": {
                "winner_kernel": "", "tournament_scores": {}, "iterations": 0, "bytes_streamed_global": 0, "bytes_streamed_zoom": 0,
                "bytes_streamed_total": 0, "pct_archive_streamed": 0.0, "pct_archive_skipped": 0.0, "local_stream_wall_clock_s": 0.0,
                "wan_http206_wall_clock_s": 0.0, "warm_followup_contrast_s": 0.0, "peak_disk_mb": 0.0, "llm_calls_in_loop": 0, "parquet": "",
            },
        },
        "comparisons": {k: {"overall": {}, "by_depth_stratum": {}} for k in ("arm_e_vs_arm_c", "arm_d_vs_arm_c", "arm_e_vs_arm_b", "arm_c_vs_arm_b")} | {"arm_e_vs_arm_a": {"overall": {}}},
        "physical_or_power_truth": {},
        "headline_claims_and_traps": [],
    }


REQUIRED_TOP = ("study_id", "citation", "accession", "organism", "spliced", "layout", "n_runs", "total_archive_bytes", "primary_contrast",
                "arms", "comparisons", "physical_or_power_truth", "headline_claims_and_traps")
REQUIRED_ARMS = ("arm_a_published", "arm_b_100pct_align", "arm_c_100pct_pseudo", "arm_d_matched_byte_naive", "arm_e_panexpr")
REQUIRED_E = ("winner_kernel", "tournament_scores", "iterations", "bytes_streamed_global", "bytes_streamed_zoom", "bytes_streamed_total",
              "pct_archive_streamed", "pct_archive_skipped", "local_stream_wall_clock_s", "wan_http206_wall_clock_s",
              "warm_followup_contrast_s", "peak_disk_mb", "llm_calls_in_loop", "parquet")
REQUIRED_CMP = ("arm_e_vs_arm_c", "arm_d_vs_arm_c", "arm_e_vs_arm_b", "arm_c_vs_arm_b", "arm_e_vs_arm_a")
METRIC_KEYS = ("n_genes", "pearson_r", "spearman_rho", "rmse_lfc", "sign_agree", "deg_jaccard_fdr05", "deg_auroc", "ci95_coverage")


def validate_study_json(d: dict) -> list[str]:
    errs = []
    for k in REQUIRED_TOP:
        if k not in d:
            errs.append(f"missing top-level key {k}")
    for k in REQUIRED_ARMS:
        if k not in d.get("arms", {}):
            errs.append(f"missing arm {k}")
    for k in REQUIRED_E:
        if k not in d.get("arms", {}).get("arm_e_panexpr", {}):
            errs.append(f"missing arm_e key {k}")
    for k in REQUIRED_CMP:
        if k not in d.get("comparisons", {}):
            errs.append(f"missing comparison {k}")
    for k in ("arm_e_vs_arm_c", "arm_d_vs_arm_c", "arm_e_vs_arm_b", "arm_c_vs_arm_b"):
        c = d.get("comparisons", {}).get(k, {})
        if "by_depth_stratum" not in c:
            errs.append(f"{k} missing by_depth_stratum")
        for mk in METRIC_KEYS:
            if c.get("overall") and mk not in c["overall"]:
                errs.append(f"{k}.overall missing {mk}")
    if d.get("layout") not in ("PE", "SE"):
        errs.append("layout must be PE or SE")
    if d.get("arms", {}).get("arm_e_panexpr", {}).get("llm_calls_in_loop", 1) != 0:
        errs.append("llm_calls_in_loop must be 0")
    for c in d.get("headline_claims_and_traps", []):
        for k in ("name", "expected", "arm_b", "arm_c", "arm_d", "arm_e", "passed"):
            if k not in c:
                errs.append(f"claim missing {k}")
    return errs


def write_study_json(d: dict, paths: list[str | Path]) -> list[Path]:
    clean = _nan_to_none(d)
    errs = validate_study_json(clean)
    if errs:
        raise ValueError("study JSON invalid: " + "; ".join(errs))
    out = []
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(clean, indent=2))
        out.append(p)
    return out


def dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    tot = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                tot += f.stat().st_size
        except OSError:
            pass
    return tot / 1e6
