"""Backward expectation-maximization over saved equivalence classes.

The conditioner never touches reads: it re-solves the ambiguous equivalence
classes (``ambiguous_eq``) plus unique counts (``target_accumulators``) already
in the state store, optionally conditioned on cross-modal log-priors shaped like
``EXTERNAL_PRIORS_SCHEMA`` (WGS copy number, proteomics, chromatin). Posterior
weights per class come from :func:`focaloop.hooks.apply_external_priors`, so the
same log-space combination is used live and retroactively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from .hooks import apply_external_priors
from .state import AMBIGUOUS_EQ_SCHEMA, TARGET_ACCUMULATOR_SCHEMA, FocaloopStateStore


@dataclass
class EqClass:
    targets: tuple[str, ...]
    loglik: tuple[float, ...]
    orientations: tuple[bool, ...]
    count: float


@dataclass
class EMResult:
    effective: dict[str, float]
    variance: dict[str, float]
    posteriors: list[np.ndarray]
    iterations: int
    converged: bool


def solve_em(
    target_ids: Sequence[str],
    unique_counts: Mapping[str, float],
    eq_classes: Sequence[EqClass],
    log_priors: Mapping[str, float] | None = None,
    dirichlet_alpha: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> EMResult:
    """Abundance EM: w_ct ∝ theta_t · exp(loglik_ct + prior_t); theta from unique + fractional counts."""
    ids = list(target_ids)
    pos = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    uniq = np.zeros(n)
    for t, c in unique_counts.items():
        if t in pos:
            uniq[pos[t]] += c
    prior = log_priors or {}
    theta = uniq + dirichlet_alpha + 1e-9
    theta /= theta.sum() if theta.sum() > 0 else 1.0
    posts: list[np.ndarray] = [np.full(len(e.targets), 1.0 / max(1, len(e.targets))) for e in eq_classes]
    it = 0
    converged = not eq_classes
    for it in range(1, max_iter + 1):
        frac = np.zeros(n)
        for j, e in enumerate(eq_classes):
            lw = {t: math.log(max(theta[pos[t]], 1e-300)) + prior.get(t, 0.0) for t in e.targets if t in pos}
            w = apply_external_priors(e.targets, e.loglik, lw)
            posts[j] = w
            for t, wt in zip(e.targets, w):
                if t in pos:
                    frac[pos[t]] += e.count * wt
        new = uniq + frac + dirichlet_alpha
        new /= new.sum() if new.sum() > 0 else 1.0
        delta = float(np.max(np.abs(new - theta))) if n else 0.0
        theta = new
        if delta < tol:
            converged = True
            break
    eff = uniq.copy()
    var = np.zeros(n)
    for e, w in zip(eq_classes, posts):
        for t, wt in zip(e.targets, w):
            if t in pos:
                eff[pos[t]] += e.count * wt
                var[pos[t]] += e.count * wt * (1.0 - wt)
    return EMResult({t: float(eff[i]) for i, t in enumerate(ids)}, {t: float(var[i]) for i, t in enumerate(ids)},
                    posts, it, converged)


def priors_from_table(tbl: pa.Table | pd.DataFrame | None, sample: str | None = None) -> dict[str, float]:
    """Collapse EXTERNAL_PRIORS rows into per-target log-prior weights (confidence-weighted, summed)."""
    if tbl is None:
        return {}
    df = tbl.to_pandas() if isinstance(tbl, pa.Table) else tbl
    if df.empty:
        return {}
    if sample is not None and "sample" in df:
        df = df[(df["sample"] == sample) | (df["sample"].isin(["", "*"])) | df["sample"].isna()]
    w = df["log_prior_weight"].astype(float) * df["confidence"].astype(float).clip(0.0, 1.0)
    return w.groupby(df["target_id"]).sum().to_dict()


def backward_condition(store: FocaloopStateStore, external_priors: pa.Table | pd.DataFrame | None = None,
                       dirichlet_alpha: float = 0.5, max_iter: int = 200, tol: float = 1e-6) -> dict[str, int]:
    """Re-solve every (run, partition) from saved state; rewrite effective counts and posterior weights."""
    if not (store.exists("target_accumulators") and store.exists("ambiguous_eq")):
        raise FileNotFoundError("state store lacks target_accumulators / ambiguous_eq")
    acc = store.read("target_accumulators").to_pandas()
    eq = store.read("ambiguous_eq").to_pandas()
    if external_priors is None and store.exists("external_priors"):
        external_priors = store.read("external_priors")
    target_ids = list(dict.fromkeys(acc["target_id"]))
    new_eq_weights: dict[int, list[float]] = {}
    n_solved = 0
    acc = acc.copy()
    for (run_id, part), grp in acc.groupby(["run_id", "partition"], sort=True):
        sample = str(grp["sample"].iloc[0])
        uniq = dict(zip(grp["target_id"], grp["unique_primary"] + grp["unique_alt_or_antisense"]))
        sub = eq[(eq["run_id"] == run_id) & (eq["partition"] == part)]
        classes = [EqClass(tuple(r.candidate_targets), tuple(r.aln_loglik), tuple(r.candidate_orientations),
                           float(r.fragment_count)) for r in sub.itertuples()]
        res = solve_em(target_ids, uniq, classes, priors_from_table(external_priors, sample),
                       dirichlet_alpha, max_iter, tol)
        acc.loc[grp.index, "effective_count"] = grp["target_id"].map(res.effective).astype(float)
        acc.loc[grp.index, "assignment_variance"] = grp["target_id"].map(res.variance).astype(float)
        for idx, w in zip(sub.index, res.posteriors):
            new_eq_weights[int(idx)] = [float(x) for x in w]
        n_solved += 1
    if new_eq_weights:
        eq = eq.copy()
        eq["posterior_weights"] = [new_eq_weights.get(int(i), list(w)) for i, w in zip(eq.index, eq["posterior_weights"])]
    store.write("target_accumulators", pa.Table.from_pandas(acc, schema=TARGET_ACCUMULATOR_SCHEMA, preserve_index=False))
    store.write("ambiguous_eq", pa.Table.from_pandas(eq, schema=AMBIGUOUS_EQ_SCHEMA, preserve_index=False))
    return {"partitions_solved": n_solved, "eq_classes": int(len(eq))}
