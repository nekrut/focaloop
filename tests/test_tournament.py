"""28,000-read kernel tournament and the >= 0.80 mate-tie symmetry gate."""

from __future__ import annotations

import pytest

from focaloop.engine import load_targets
from focaloop.kernels import KmerBestHitKernel, KmerEqKernel, KmerIndex, build_kernels
from focaloop.streamer import FastqBlockReader, LocalFileSource
from focaloop.tournament import (
    N_REGIMES,
    REAL_READS,
    REGIMES,
    SYNTHETIC_PER_REGIME,
    TIE_SYMMETRY_MIN,
    TOTAL_READS,
    build_synthetic_truth,
    find_tie_pairs,
    run_tournament,
)

from .conftest import BLUEPRINTS

TD = BLUEPRINTS / "panexpr" / "test-data"


@pytest.fixture(scope="module")
def index() -> KmerIndex:
    return KmerIndex(load_targets(TD / "targets.fa"), k=31)


@pytest.fixture(scope="module")
def real_reads() -> list[str]:
    reads: list[str] = []
    for s in ("ctrl1", "ctrl2", "trt1", "trt2"):
        for b in FastqBlockReader(LocalFileSource(TD / f"{s}.fastq.gz"), s, 1 << 20):
            reads += b.reads
    return reads


def test_budget_constants() -> None:
    assert (REAL_READS, SYNTHETIC_PER_REGIME, N_REGIMES, TOTAL_READS) == (25_000, 600, 5, 28_000)
    assert len(REGIMES) == 5 and TIE_SYMMETRY_MIN == 0.80


def test_synthetic_truth_has_five_regimes_of_600(index: KmerIndex) -> None:
    cases = build_synthetic_truth(index, read_len=100)
    by = {r: sum(1 for c in cases if c.regime == r) for r in REGIMES}
    assert by == {r: 600 for r in REGIMES}
    ties = find_tie_pairs(index, 100)
    assert any({"g7.strainA", "g8.strainA"} == {a, b} for a, b, _ in ties)


def test_tie_retaining_kernel_wins_and_besthit_is_disqualified(index: KmerIndex, real_reads: list[str]) -> None:
    res = run_tournament([KmerBestHitKernel(index), KmerEqKernel(index)], index, real_reads)
    board = {s.kernel: s for s in res.board}
    assert res.n_real == len(real_reads) <= REAL_READS and res.n_synthetic == 3000
    assert res.conditioning_active
    assert res.winner == "kmer_eq"
    assert board["kmer_eq"].mate_tie_symmetry >= 0.95 and board["kmer_eq"].gate_passed
    assert board["kmer_besthit"].mate_tie_symmetry < 0.80 and not board["kmer_besthit"].gate_passed
    assert board["kmer_eq"].real_mapped_pct > 95.0
    assert board["kmer_eq"].recall["normal_0.5pct_err"] > 95.0


def test_all_disqualified_and_unavailable_kernels(index: KmerIndex, real_reads: list[str]) -> None:
    res = run_tournament([KmerBestHitKernel(index)], index, real_reads[:500], n_per_regime=100)
    assert res.winner is None and res.all_disqualified
    kernels = build_kernels(["kmer_eq", "bwa-mem2", "salmon"], index)
    assert [k.available() for k in kernels] == [True, False, False]
    res2 = run_tournament(kernels, index, real_reads[:500], n_per_regime=100)
    assert [s.kernel for s in res2.board] == ["kmer_eq"]
