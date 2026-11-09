"""Contrast statistics over saved target accumulators (group level) and Hudson F_ST."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq-style median-of-ratios size factors (columns = samples); library-size fallback."""
    if counts.empty:
        return pd.Series(dtype=float)
    pos = counts[(counts > 0).all(axis=1)]
    if len(pos) >= 3:
        logs = np.log(pos)
        geo = logs.mean(axis=1)
        sf = np.exp(logs.sub(geo, axis=0).median(axis=0))
    else:
        lib = counts.sum(axis=0).replace(0, 1.0)
        sf = lib / lib.mean()
    return sf.replace(0, 1.0)


def group_matrix(acc: pd.DataFrame, partition: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(effective counts, assignment variance) as group x sample matrices."""
    df = acc if partition is None else acc[acc["partition"] == partition]
    eff = df.pivot_table(index="group_id", columns="sample", values="effective_count", aggfunc="sum").fillna(0.0)
    var = df.pivot_table(index="group_id", columns="sample", values="assignment_variance", aggfunc="sum") \
        .reindex_like(eff).fillna(0.0)
    return eff, var


def contrast_table(acc: pd.DataFrame, sample_level: Mapping[str, str], contrast: tuple[str, str],
                   partition: str | None = None, min_mean: float = 1.0) -> pd.DataFrame:
    """Log2 fold change A vs B per group with replicate + placement + Poisson variance (Wald z)."""
    a, b = contrast
    eff, var = group_matrix(acc, partition)
    cols = ["group_id", f"mean_{a}", f"mean_{b}", "log2fc", "se", "z", "pvalue", "padj", "placement_var_frac", "n_a", "n_b"]
    sa = [s for s in eff.columns if sample_level.get(s) == a]
    sb = [s for s in eff.columns if sample_level.get(s) == b]
    if eff.empty or not sa or not sb:
        return pd.DataFrame(columns=cols)
    scale = 1.0 / size_factors(eff)
    normc = eff * scale
    l2 = np.log2(normc + 0.5)
    tech = (var + eff) * scale ** 2 / (normc + 0.5) ** 2 / math.log(2) ** 2

    def side(samples: list[str]) -> tuple[pd.Series, pd.Series]:
        m = l2[samples].mean(axis=1)
        t = tech[samples].mean(axis=1)
        if len(samples) > 1:
            v = np.maximum(l2[samples].var(axis=1, ddof=1), t)
        else:
            v = t
        return m, v / len(samples)

    ma, va = side(sa)
    mb, vb = side(sb)
    lfc = ma - mb
    se = np.sqrt(va + vb).clip(lower=1e-9)
    z = lfc / se
    p = 2.0 * norm.sf(np.abs(z))
    pl = var[sa + sb].sum(axis=1) / (var[sa + sb].sum(axis=1) + eff[sa + sb].sum(axis=1)).clip(lower=1e-12)
    out = pd.DataFrame({
        "group_id": eff.index,
        f"mean_{a}": normc[sa].mean(axis=1).to_numpy(),
        f"mean_{b}": normc[sb].mean(axis=1).to_numpy(),
        "log2fc": lfc.to_numpy(),
        "se": se.to_numpy(),
        "z": z.to_numpy(),
        "pvalue": p,
        "placement_var_frac": pl.to_numpy(),
        "n_a": len(sa),
        "n_b": len(sb),
    })
    keep = (out[f"mean_{a}"] + out[f"mean_{b}"]) / 2.0 >= min_mean
    out = out[keep].copy()
    out["padj"] = bh_adjust(out["pvalue"].to_numpy())
    return out[cols].sort_values(["pvalue", "group_id"]).reset_index(drop=True)


def hudson_fst(ref_a: np.ndarray, alt_a: np.ndarray, ref_b: np.ndarray, alt_b: np.ndarray) -> float:
    """Ratio-of-averages Hudson F_ST across sites (Bhatia et al. 2013) from pooled allele counts."""
    na, nb = ref_a + alt_a, ref_b + alt_b
    ok = (na >= 2) & (nb >= 2)
    if not ok.any():
        return float("nan")
    pa_, pb_ = alt_a[ok] / na[ok], alt_b[ok] / nb[ok]
    num = (pa_ - pb_) ** 2 - pa_ * (1 - pa_) / (na[ok] - 1) - pb_ * (1 - pb_) / (nb[ok] - 1)
    den = pa_ * (1 - pb_) + pb_ * (1 - pa_)
    if den.sum() <= 0:
        return 0.0
    return float(num.sum() / den.sum())


def fst_permutation(alleles: pd.DataFrame, sample_level: Mapping[str, str], contrast: tuple[str, str],
                    n_perm: int = 199, seed: int = 7) -> tuple[float, float]:
    """F_ST between contrast groups (sample-pooled) with a label-permutation p-value."""
    a, b = contrast
    samples = sorted(s for s in alleles["sample"].unique() if sample_level.get(s) in (a, b))
    if not samples:
        return float("nan"), 1.0
    ref = alleles.pivot_table(index="site_id", columns="sample", values="ref_reads", aggfunc="sum").reindex(columns=samples).fillna(0)
    alt = alleles.pivot_table(index="site_id", columns="sample", values="alt_reads", aggfunc="sum").reindex(columns=samples).fillna(0)
    labels = [sample_level[s] for s in samples]

    def stat(lab: list[str]) -> float:
        ia = [i for i, x in enumerate(lab) if x == a]
        ib = [i for i, x in enumerate(lab) if x == b]
        if not ia or not ib:
            return float("nan")
        r, t = ref.to_numpy(), alt.to_numpy()
        return hudson_fst(r[:, ia].sum(1), t[:, ia].sum(1), r[:, ib].sum(1), t[:, ib].sum(1))

    obs = stat(labels)
    if math.isnan(obs):
        return obs, 1.0
    rng = random.Random(seed)
    ge = 1
    for _ in range(n_perm):
        lab = labels[:]
        rng.shuffle(lab)
        v = stat(lab)
        if not math.isnan(v) and v >= obs - 1e-12:
            ge += 1
    return obs, ge / (n_perm + 1)
