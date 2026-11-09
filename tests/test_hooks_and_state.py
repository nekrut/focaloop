"""Unit tests verifying all 5 Focaloop capability hooks and Zstd-Parquet round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from focaloop import (
    BaitPrefilter,
    EqClassBackendProtocol,
    EqClassRecord,
    FocaloopStateStore,
    HolobiontLocusRegistry,
    HypothesisLedger,
    SteeringMask,
    apply_external_priors,
    coalesce_residual_spans,
    is_holdout_block,
)

BAIT_SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTTGCATGCATGCATGCATGCATGCATGCATGCA"
ALT_SEQ = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"


def _revcomp(seq: str) -> str:
    tr = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(tr)[::-1]


def test_hook1_stage0_bait_prefilter_canonical_and_parquet(tmp_path: Path) -> None:
    store = FocaloopStateStore(tmp_path / "state")
    prefilter = BaitPrefilter("pfkelch13_C580Y", [BAIT_SEQ], min_kmer_hits_per_read=2, min_hit_reads=2)

    hit_run = prefilter.screen_reads(
        run_id="ERR000001",
        reads=[BAIT_SEQ[:45], _revcomp(BAIT_SEQ[10:55]), ALT_SEQ[:45], ALT_SEQ[10:55]],
        probe_bytes=65536,
        store=store,
    )
    miss_run = prefilter.screen_reads(
        run_id="ERR000002",
        reads=[ALT_SEQ[:45], ALT_SEQ[5:50], ALT_SEQ[10:55]],
        probe_bytes=65536,
        store=store,
    )

    assert hit_run.pass_gate is True
    assert hit_run.hit_reads == 2
    assert hit_run.wilson_ci95_low > 0.05
    assert miss_run.pass_gate is False
    assert miss_run.hit_reads == 0

    tbl = store.read("bait_prefilter")
    assert tbl.num_rows == 2
    assert tbl.column("pass_gate").to_pylist() == [True, False]


def test_hook2_pangenome_graph_backend_and_holobiont_registry(tmp_path: Path) -> None:
    class MockVgBackend:
        backend_kind = "pangenome_graph"

        def map_slice_to_eq_classes(
            self, r1_seqs: list[str], r2_seqs: list[str] | None = None
        ) -> tuple[list[EqClassRecord], list[int]]:
            return ([EqClassRecord(("snarl_101", "snarl_102"), (True, True), (-1.2, -1.2), 3)], [2])

    backend = MockVgBackend()
    assert isinstance(backend, EqClassBackendProtocol)
    eqs, residual_idx = backend.map_slice_to_eq_classes(["A" * 50, "C" * 50, "G" * 50])
    assert len(eqs) == 1 and eqs[0].fragment_count == 3
    assert residual_idx == [2]

    store = FocaloopStateStore(tmp_path / "state")
    reg = HolobiontLocusRegistry()
    reg.register("ENSG000001", "transcript", "grp_host_IFNB1", "GRCh38", "host", 1120)
    reg.register("blaNDM1_p1", "plasmid_contig", "grp_amr_NDM1", "Kp_plasmid", "bacteria", 813, hgt_mobile_flag=True)
    reg.persist(store)

    tbl = store.read("graph_locus_map")
    assert tbl.num_rows == 2
    assert tbl.column("hgt_mobile_flag").to_pylist() == [False, True]


def test_hook3_retroactive_residual_spans_and_cross_modal_priors(tmp_path: Path) -> None:
    store = FocaloopStateStore(tmp_path / "state")
    offsets = [
        (1000, 1450, BAIT_SEQ[:50]),
        (12000, 12400, BAIT_SEQ[10:60]),  # within 64 KiB -> coalesced into span 0
        (500_000, 500_500, ALT_SEQ[:50]),  # > 64 KiB gap -> span 1
    ]
    spans = coalesce_residual_spans("SRR999", "s3_sra_odp", offsets, merge_gap_bytes=65536, store=store)
    assert len(spans) == 2
    assert spans[0].byte_start == 1000 and spans[0].byte_end == 12400 and spans[0].n_reads == 2
    assert len(spans[0].kmer_sketch_u64) > 0
    assert store.read("residual_spans").num_rows == 2

    # Tied RNA alignment (-2.0 vs -2.0) resolved retroactively by WGS + proteomics prior
    post_flat = apply_external_priors(["strainA_tox", "strainB_tox"], [-2.0, -2.0], {})
    assert np.allclose(post_flat, [0.5, 0.5])
    post_fused = apply_external_priors(
        ["strainA_tox", "strainB_tox"],
        [-2.0, -2.0],
        {"strainA_tox": 0.0, "strainB_tox": -8.0},
    )
    assert post_fused[0] > 0.999 and post_fused[1] < 0.001


def test_hook4_realtime_steering_mask_and_bed_export(tmp_path: Path) -> None:
    store = FocaloopStateStore(tmp_path / "state")
    mask = SteeringMask(
        enrich_sequences={"borderline_geneA": BAIT_SEQ},
        deplete_sequences={"abundant_housekeepingB": ALT_SEQ, "shared_paralog": BAIT_SEQ[:40] + ALT_SEQ[:24]},
    )
    assert mask.halt_run is False
    assert mask.decide_read_until(BAIT_SEQ[:40]) == "accept"
    assert mask.decide_read_until(ALT_SEQ[:40]) == "unblock_eject"
    assert mask.decide_read_until("ATATATAT" * 5) == "neutral"

    bed = mask.export_read_until_bed(
        {"borderline_geneA": ("chr1", 100, 900), "abundant_housekeepingB": ("chr2", 500, 2000)},
        tmp_path / "targets.bed",
    )
    bed_txt = bed.read_text(encoding="utf-8")
    assert "chr1\t100\t900\tborderline_geneA\tenrich" in bed_txt
    assert "chr2\t500\t2000\tabundant_housekeepingB\tdeplete" in bed_txt

    # When zero unsettled targets remain, run halts automatically
    done_mask = SteeringMask(enrich_sequences={}, deplete_sequences={"abundant_housekeepingB": ALT_SEQ})
    assert done_mask.halt_run is True
    assert done_mask.decide_read_until(BAIT_SEQ[:40]) == "unblock_eject"
    done_mask.record_telemetry(store, iteration=4, reason="all_hypotheses_locked")
    assert store.read("steering_decisions").column("halt_run").to_pylist() == [True]


def test_hook5_interactive_hypothesis_ledger_and_holdout_split(tmp_path: Path) -> None:
    flags = [is_holdout_block("SRR12345", blk, holdout_fraction=0.10) for blk in range(2000)]
    frac = sum(flags) / len(flags)
    assert 0.07 <= frac <= 0.13
    # Deterministic across calls
    assert flags[:20] == [is_holdout_block("SRR12345", blk, holdout_fraction=0.10) for blk in range(20)]

    store = FocaloopStateStore(tmp_path / "state")
    ledger = HypothesisLedger(alpha_budget=0.05, spend_fraction=0.10, discovery_reward=0.04, holdout_alpha=0.05)

    # True effect confirmed on both discovery and holdout with matching sign -> earns wealth
    v1 = ledger.evaluate_query("q1", "agent:orbit", "DEX vs Veh: FKBP5", 3.4, 1e-8, 3.1, 2e-4, store=store)
    assert v1.confirmed is True
    assert v1.alpha_wealth_after > 0.05

    # Overfit discovery noise failing holdout -> rejected, wealth decays cleanly without going negative
    for i in range(25):
        v = ledger.evaluate_query(f"noise_{i}", "agent:orbit", f"null_{i}", 1.1, 1e-4, -0.2, 0.65, store=store)
        assert v.confirmed is False
        assert v.alpha_wealth_after > 0.0

    assert store.read("hypothesis_ledger").num_rows == 26
