"""Reusable-holdout isolation: adaptive decisions (stop gate, primary table, steering) read discovery only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from focaloop import engine as eng
from focaloop.archive import pack_state, unpack_state
from focaloop.blueprint import load_blueprint
from focaloop.engine import FocaloopRun, RunSpec
from focaloop.ops import run_steer
from focaloop.streamer import FastqBlockReader
from focaloop.state import TARGET_ACCUMULATOR_SCHEMA, FocaloopStateStore
from focaloop.variants import ALLELE_COUNTS_SCHEMA

from .conftest import BLUEPRINTS


def _run(bid: str, out: Path, monkeypatch=None, frames: list | None = None) -> FocaloopRun:
    if frames is not None:
        orig = eng.StopGate.evaluate

        def spy(self, per_sample, *a, **k):
            frames.append(per_sample.copy())
            return orig(self, per_sample, *a, **k)

        monkeypatch.setattr(eng.StopGate, "evaluate", spy)
    run = FocaloopRun(load_blueprint(bid), RunSpec.load(BLUEPRINTS / bid / "test-data" / "spec.yml"),
                      out / "state", out / "table.tsv", out / "tel.json")
    assert run.execute() == 0
    return run


def _corrupt_holdout(store: FocaloopStateStore) -> None:
    acc = store.read("target_accumulators").to_pandas()
    h = acc["partition"] == "holdout"
    assert h.any()
    acc.loc[h, ["unique_primary", "unique_alt_or_antisense", "effective_count"]] = 0.0
    acc.loc[h & (acc["group_id"] == "g4"), "effective_count"] = 1e6  # also a wild outlier
    store.write("target_accumulators", pa.Table.from_pandas(acc, schema=TARGET_ACCUMULATOR_SCHEMA, preserve_index=False))
    if store.exists("allele_counts"):
        al = store.read("allele_counts").to_pandas()
        al.loc[al["partition"] == "holdout", ["ref_reads", "alt_reads"]] = [0, 999]
        store.write("allele_counts", pa.Table.from_pandas(al, schema=ALLELE_COUNTS_SCHEMA, preserve_index=False),
                    schema=ALLELE_COUNTS_SCHEMA)


@pytest.mark.parametrize("bid", ["panexpr", "malariavar"])
def test_primary_table_ignores_holdout(bid: str, tmp_path: Path) -> None:
    run = _run(bid, tmp_path)
    assert run.bp.stop_gate.holdout_fraction > 0
    assert run.telemetry["table_partitions"] == ["discovery"]
    before = (tmp_path / "table.tsv").read_text()
    store = FocaloopStateStore(tmp_path / "state")
    _corrupt_holdout(store)
    spec = run.spec
    run._write_table(store, {s["sample"]: s["condition"] for s in spec.samples}, run.telemetry.get("contrast") and
                     tuple(run.telemetry["contrast"]), None, None)
    assert (tmp_path / "table.tsv").read_text() == before


def test_stop_gate_sees_only_discovery(tmp_path: Path, monkeypatch) -> None:
    frames: list[pd.DataFrame] = []
    run = _run("panexpr", tmp_path, monkeypatch, frames)
    assert frames and run.telemetry["gate_partitions"] == ["discovery"]
    acc = FocaloopStateStore(tmp_path / "state").read("target_accumulators").to_pandas()
    disc = acc[acc["partition"] == "discovery"].groupby("sample")["effective_count"].sum()
    hold = acc[acc["partition"] == "holdout"].groupby("sample")["effective_count"].sum()
    assert (hold > 0).all()
    seen = frames[-1].groupby("sample")["effective_count"].sum()
    pd.testing.assert_series_equal(seen.sort_index(), disc.sort_index(), check_names=False, rtol=1e-6)


def test_pooling_only_without_holdout(tmp_path: Path) -> None:
    run = FocaloopRun.__new__(FocaloopRun)
    run.bp = load_blueprint("panexpr")
    assert run.inference_partitions() == ("discovery",)
    run.bp = load_blueprint("focaloop_bait")  # holdout_fraction 0.0: every block is discovery
    assert run.inference_partitions() == ("discovery", "holdout")


def test_steer_ignores_holdout(panexpr_run, tmp_path: Path) -> None:
    state = unpack_state(panexpr_run["dir"] / "state.tar.zst", tmp_path / "s")
    bed = BLUEPRINTS / "panexpr" / "test-data" / "targets.bed"
    a = run_steer(state, bed, tmp_path / "a.bed", min_precision=500)
    _corrupt_holdout(FocaloopStateStore(state))
    b = run_steer(state, bed, tmp_path / "b.bed", min_precision=500)
    assert a["partitions"] == ["discovery"]
    assert (tmp_path / "a.bed").read_text() == (tmp_path / "b.bed").read_text()
    assert a["group_precision"] == b["group_precision"]
    assert a["n_enrich_targets"] > 0 and a["n_deplete_targets"] > 0


def test_unpack_state_caps_decompressed_size(tmp_path: Path) -> None:
    d = tmp_path / "st"
    d.mkdir()
    (d / "big.bin").write_bytes(b"\0" * 200_000)  # compresses to a few hundred bytes
    arc = pack_state(d, tmp_path / "bomb.tar.zst")
    assert arc.stat().st_size < 5_000
    with pytest.raises(ValueError, match="inflates past"):
        unpack_state(arc, tmp_path / "out", max_bytes=50_000)
    assert not (tmp_path / "out").exists()
    out = unpack_state(arc, tmp_path / "ok", max_bytes=1_000_000)
    assert (out / "big.bin").stat().st_size == 200_000


def _junk_holdout(monkeypatch, copies: int = 40) -> set[str]:
    """Replace every holdout block with `copies`x as many random (unmappable) reads."""
    rng = np.random.default_rng(7)
    made: set[str] = set()
    orig = FastqBlockReader.next_block

    def junk(self):
        b = orig(self)
        if b is not None and b.partition == "holdout" and b.reads:
            n, ln = len(b.reads) * copies, len(b.reads[0])
            b.reads = ["".join(rng.choice(list("ACGT"), ln)) for _ in range(n)]
            made.update(b.reads)
        return b

    monkeypatch.setattr(FastqBlockReader, "next_block", junk)
    return made


def test_low_map_rate_ignores_unmapped_holdout(tmp_path: Path, monkeypatch) -> None:
    junk = _junk_holdout(monkeypatch)
    seen: list[str] = []
    orig_t = eng.run_tournament
    monkeypatch.setattr(eng, "run_tournament", lambda k, i, real, **kw: (seen.extend(real), orig_t(k, i, real, **kw))[1])
    run = _run("panexpr", tmp_path)
    tel = run.telemetry
    thr = run.bp.trap("LOW_MAP_RATE").threshold
    assert "LOW_MAP_RATE" not in tel["trap_codes"] and tel["status"] != "aborted"
    assert tel["map_rate_partitions"] == ["discovery"] and tel["map_rate"] >= thr
    # pooled (discovery + junk holdout) rate is far below the threshold: a pooled check would have aborted
    for it in tel["iterations"]:
        assert it["mapped_reads"] / it["total_reads"] < thr <= it["map_rate"]
    # the tournament (kernel choice, TIE_SYMMETRY_FAIL) never saw a holdout read
    assert junk and seen and not junk.intersection(seen)
