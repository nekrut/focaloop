"""Backward EM over saved eq classes (with external log-priors) and the stop gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from focaloop.blueprint import StopGateConfig
from focaloop.conditioner import EqClass, backward_condition, priors_from_table, solve_em
from focaloop.state import EXTERNAL_PRIORS_SCHEMA, FocaloopStateStore
from focaloop.stats import bh_adjust, hudson_fst
from focaloop.stop_gate import StopGate, check_design, gate_metrics


def test_em_resolves_ambiguity_from_unique_evidence() -> None:
    res = solve_em(["A", "B"], {"A": 90, "B": 10}, [EqClass(("A", "B"), (0.0, 0.0), (True, True), 100)],
                   dirichlet_alpha=0.0)
    assert res.converged
    assert res.effective["A"] == pytest.approx(180, rel=0.02)
    assert res.effective["B"] == pytest.approx(20, rel=0.1)
    assert res.effective["A"] + res.effective["B"] == pytest.approx(200)
    assert res.variance["A"] == pytest.approx(res.variance["B"]) and res.variance["A"] > 0


def test_external_priors_shift_posteriors() -> None:
    eq = [EqClass(("A", "B"), (0.0, 0.0), (True, True), 100)]
    flat = solve_em(["A", "B"], {}, eq, dirichlet_alpha=1.0)
    biased = solve_em(["A", "B"], {}, eq, log_priors={"B": -3.0}, dirichlet_alpha=1.0)
    assert flat.effective["A"] == pytest.approx(50.0)
    assert biased.effective["A"] > 90.0
    tbl = pa.Table.from_pylist([
        {"group_id": "g", "target_id": "B", "sample": "*", "modality": "wgs_copy_number", "log_prior_weight": -2.0,
         "confidence": 0.5, "source_uri": "x"}], schema=EXTERNAL_PRIORS_SCHEMA)
    assert priors_from_table(tbl, "s1") == {"B": -1.0}


def test_backward_condition_rewrites_saved_state(panexpr_run) -> None:
    store = FocaloopStateStore(panexpr_run["dir"] / "state")
    before = store.read("target_accumulators").to_pandas()
    eq = store.read("ambiguous_eq").to_pandas()
    assert len(eq) > 0 and all(abs(sum(w) - 1.0) < 1e-6 for w in eq["posterior_weights"])
    pri = pd.DataFrame([{"group_id": "g1", "target_id": "g1.strainB", "sample": "*", "modality": "wgs_copy_number",
                         "log_prior_weight": -8.0, "confidence": 1.0, "source_uri": "test"}])
    info = backward_condition(store, pri)
    assert info["partitions_solved"] == 12
    after = store.read("target_accumulators").to_pandas()
    b_before = before[before.target_id == "g1.strainB"]["effective_count"].sum()
    b_after = after[after.target_id == "g1.strainB"]["effective_count"].sum()
    assert b_after < b_before
    total = lambda d: d.groupby("sample")["effective_count"].sum()  # noqa: E731
    assert np.allclose(total(before), total(after))  # EM only re-places reads; nothing is created
    backward_condition(store, pri.assign(log_prior_weight=0.0))  # restore for other tests


def _frame(counts: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for s, cs in counts.items():
        for i, c in enumerate(cs):
            rows.append({"sample": s, "target_id": f"t{i}", "effective_count": c, "assignment_variance": 0.0})
    return pd.DataFrame(rows)


def test_stop_gate_converges_caps_and_exhausts() -> None:
    cfg = StopGateConfig(min_posterior_precision=50, max_tech_var_fraction=0.5, delta_tol=0.05, min_iterations=2,
                         max_bytes=10_000)
    cond = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    rng = np.random.default_rng(3)
    base = rng.uniform(200, 3000, 20)
    big = _frame({s: list(base * rng.lognormal(0.0, 0.3, 20)) for s in cond})
    g = StopGate(cfg)
    assert g.evaluate(big, cond, 1, 100, False).reason == "continue"
    d = g.evaluate(big, cond, 2, 200, False)
    assert d.stop and d.converged and d.reason == "converged"
    small = _frame({"a1": [6, 7, 8], "a2": [6, 7, 9], "b1": [6, 8, 7], "b2": [7, 6, 8]})
    g2 = StopGate(cfg)
    g2.evaluate(small, cond, 1, 100, False)
    capped = g2.evaluate(small, cond, 2, 20_000, False)
    assert capped.stop and not capped.converged and capped.traps == ["BYTE_CAP_UNCONVERGED"]
    assert StopGate(cfg).evaluate(small, cond, 1, 100, True).reason == "exhausted_unconverged"
    m, _ = gate_metrics(big, cond)
    assert m.n_active_targets == 20 and m.median_posterior_precision > 50 and m.median_tech_var_fraction < 0.1


def test_design_and_stats_helpers() -> None:
    conf = [{"sample": "a", "condition": "x", "batch": "1"}, {"sample": "b", "condition": "y", "batch": "2"}]
    assert check_design(conf, ("x", "y")) == ["condition is perfectly confounded with batch"]
    assert check_design(conf, ("x", "z")) == ["contrast level 'z' has no samples", "condition is perfectly confounded with batch"]
    ok = conf + [{"sample": "c", "condition": "x", "batch": "2"}]
    assert check_design(ok, ("x", "y")) == []
    assert bh_adjust([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.04, 0.04])
    r = np.array([50.0, 50.0])
    assert hudson_fst(r, r, r, r) < 0.05
    assert hudson_fst(np.array([100.0, 100]), np.array([0.0, 0]), np.array([0.0, 0]), np.array([100.0, 100])) > 0.95
