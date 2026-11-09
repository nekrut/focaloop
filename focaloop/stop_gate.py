"""Replicate-variance / posterior-precision stopping gate and design checks.

At every streaming iteration the gate asks three questions over expressed
targets (median across targets):

1. **Posterior precision** ``1 / Var_tech[log abundance]`` -- is the placement
   + Poisson uncertainty small enough? (``>= min_posterior_precision``)
2. **Technical-variance fraction** ``Var_tech / (Var_tech + Var_bio)`` from
   replicate scatter -- is more sequencing still buying anything?
   (``<= max_tech_var_fraction``; skipped when no condition has replicates)
3. **Stability** -- median absolute change of log estimates since the last
   iteration (``<= delta_tol``).

The gate stops on convergence, on the byte cap (``BYTE_CAP_UNCONVERGED`` when
the criteria are not yet met), or when every stream is exhausted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .blueprint import StopGateConfig
from .stats import size_factors


@dataclass
class GateMetrics:
    n_active_targets: int
    median_posterior_precision: float
    median_tech_var_fraction: float
    median_abs_delta_estimate: float


@dataclass
class GateDecision:
    stop: bool
    reason: str  # continue | converged | byte_cap_unconverged | exhausted | exhausted_unconverged
    metrics: GateMetrics
    converged: bool
    traps: list[str] = field(default_factory=list)


def gate_metrics(per_sample: pd.DataFrame, sample_condition: Mapping[str, str],
                 prev_log: Mapping[str, float] | None = None, min_count: float = 5.0) -> tuple[GateMetrics, dict[str, float]]:
    """``per_sample`` columns: sample, target_id, effective_count, assignment_variance."""
    if per_sample.empty:
        return GateMetrics(0, 0.0, float("nan"), float("inf")), {}
    piv = per_sample.pivot_table(index="target_id", columns="sample", values="effective_count", aggfunc="sum").fillna(0.0)
    var = per_sample.pivot_table(index="target_id", columns="sample", values="assignment_variance", aggfunc="sum") \
        .reindex_like(piv).fillna(0.0)
    scale = 1.0 / size_factors(piv)
    norm = piv * scale
    log_ab = np.log(norm + 0.5)
    tech_log_var = (var + piv) / (piv + 0.5) ** 2  # delta method: Var[log c] ≈ Var[c] / c^2
    active = piv.mean(axis=1) >= min_count
    n_active = int(active.sum())
    pooled_log = np.log(norm.sum(axis=1) + 0.5)
    pooled_tech = (var.sum(axis=1) + piv.sum(axis=1)) / (piv.sum(axis=1) + 0.5) ** 2
    if n_active == 0:
        return GateMetrics(0, 0.0, float("nan"), float("inf")), pooled_log.to_dict()
    precision = float(np.median(1.0 / pooled_tech[active].clip(lower=1e-12)))
    conds: dict[str, list[str]] = {}
    for s in piv.columns:
        conds.setdefault(sample_condition.get(s, "NA"), []).append(s)
    fracs: list[np.ndarray] = []
    for samples in conds.values():
        if len(samples) < 2:
            continue
        rep = log_ab.loc[active, samples].var(axis=1, ddof=1)
        tech = tech_log_var.loc[active, samples].mean(axis=1)
        bio = (rep - tech).clip(lower=0.0)
        fracs.append((tech / (tech + bio).clip(lower=1e-12)).to_numpy())
    tech_frac = float(np.median(np.concatenate(fracs))) if fracs else float("nan")
    if prev_log:
        common = [t for t in pooled_log.index[active] if t in prev_log]
        delta = float(np.median([abs(pooled_log[t] - prev_log[t]) for t in common])) if common else float("inf")
    else:
        delta = float("inf")
    return GateMetrics(n_active, precision, tech_frac, delta), pooled_log.to_dict()


class StopGate:
    def __init__(self, cfg: StopGateConfig) -> None:
        self.cfg = cfg
        self._prev: dict[str, float] | None = None

    def criteria_met(self, m: GateMetrics, iteration: int) -> bool:
        if iteration < self.cfg.min_iterations or m.n_active_targets == 0:
            return False
        if m.median_posterior_precision < self.cfg.min_posterior_precision:
            return False
        if not math.isnan(m.median_tech_var_fraction) and m.median_tech_var_fraction > self.cfg.max_tech_var_fraction:
            return False
        return m.median_abs_delta_estimate <= self.cfg.delta_tol

    def evaluate(self, per_sample: pd.DataFrame, sample_condition: Mapping[str, str], iteration: int,
                 cum_bytes: int, exhausted: bool) -> GateDecision:
        m, cur = gate_metrics(per_sample, sample_condition, self._prev)
        self._prev = cur
        ok = self.criteria_met(m, iteration)
        if ok:
            return GateDecision(True, "converged", m, True)
        if self.cfg.max_bytes and cum_bytes >= self.cfg.max_bytes:
            return GateDecision(True, "byte_cap_unconverged", m, False, ["BYTE_CAP_UNCONVERGED"])
        if exhausted:
            return GateDecision(True, "exhausted_unconverged", m, False)
        return GateDecision(False, "continue", m, False)


def check_design(samples: Sequence[Mapping[str, str]], contrast: tuple[str, str] | None) -> list[str]:
    """Return design problems; any entry means ``CONFOUNDED_DESIGN``."""
    problems: list[str] = []
    conds = {s["condition"] for s in samples}
    if contrast is not None:
        for c in contrast:
            if c not in conds:
                problems.append(f"contrast level {c!r} has no samples")
    batches = {s.get("batch", "") for s in samples}
    if len(batches) > 1 and len(conds) > 1:
        cond_to_batch: dict[str, set[str]] = {}
        for s in samples:
            cond_to_batch.setdefault(s["condition"], set()).add(s.get("batch", ""))
        single = all(len(b) == 1 for b in cond_to_batch.values())
        distinct = len({next(iter(b)) for b in cond_to_batch.values()}) == len(cond_to_batch)
        if single and distinct:
            problems.append("condition is perfectly confounded with batch")
    return problems
