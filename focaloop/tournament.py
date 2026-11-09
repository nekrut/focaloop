"""Startup kernel tournament: 25,000 real + 3,000 synthetic (5 regimes x 600) reads.

Every available kernel maps one combined 28,000-read batch. Scores combine the
real-read mapping rate, per-regime synthetic recall, and throughput. On
``paralog_or_strain_tie`` cases (fragments identical in two targets) a kernel
must report *both* candidates; the fraction of such reads that keep both is the
**mate-tie symmetry**. When conditioning is active (multi-assembly catalogs or
multi-member groups) a kernel with symmetry ``< 0.80`` is disqualified -- it
would leak strain/paralog weights into the backward EM. If no kernel survives,
the run raises the ``TIE_SYMMETRY_FAIL`` trap.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from .kernels import Kernel, KmerEqKernel, KmerIndex, TargetRec, revcomp

REAL_READS = 25_000
SYNTHETIC_PER_REGIME = 600
REGIMES = (
    "normal_0.5pct_err",
    "divergent_3pct_SNP",
    "adapter_overhang",
    "short_edge_or_splice_junction",
    "paralog_or_strain_tie",
)
N_REGIMES = len(REGIMES)
TOTAL_READS = REAL_READS + SYNTHETIC_PER_REGIME * N_REGIMES  # 28,000
TIE_SYMMETRY_MIN = 0.80
ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCACATCACGATCTCGTATGCCGTCTTCTGCTTG" * 3


@dataclass(frozen=True)
class SyntheticCase:
    regime: str
    truth: frozenset[str]
    read: str
    tie_pair: tuple[str, str] | None = None


def _mutate(rng: random.Random, s: str, rate: float) -> str:
    if rate <= 0:
        return s
    b = list(s)
    for i, ch in enumerate(b):
        if rng.random() < rate:
            b[i] = rng.choice([x for x in "ACGT" if x != ch])
    return "".join(b)


def find_tie_pairs(index: KmerIndex, window: int, max_pairs: int = 200) -> list[tuple[str, str, int]]:
    """(target_a, target_b, offset_a): a ``window``-bp block of A occurring verbatim (either strand) in B."""
    genes = [t for t in index.targets if not t.is_decoy and t.length >= window]
    tmap = {t.target_id: t for t in index.targets}
    kern = KmerEqKernel(index, min_hits=1, tie_slack=0)
    out: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for a in genes:
        offs = list(range(0, a.length - window + 1, max(1, window // 2)))
        chunks = [a.seq[o:o + window] for o in offs]
        res = kern.map_reads(chunks)
        for o, chunk, c in zip(offs, chunks, res.candidates):
            if c is None or len(c[0]) < 2:
                continue
            for ti, ll in zip(c[0], c[1]):
                b = index.targets[ti]
                if b.target_id == a.target_id or ll < 0:
                    continue
                if a.group_id == b.group_id and a.assembly_id == b.assembly_id:
                    continue
                key = tuple(sorted((a.target_id, b.target_id)))
                if key in seen:
                    continue
                bseq = tmap[b.target_id].seq
                if chunk in bseq or revcomp(chunk) in bseq:
                    seen.add(key)
                    out.append((a.target_id, b.target_id, o))
                    if len(out) >= max_pairs:
                        return out
    return out


def build_synthetic_truth(index: KmerIndex, read_len: int = 100, n_per_regime: int = SYNTHETIC_PER_REGIME,
                          seed: int = 42) -> list[SyntheticCase]:
    rng = random.Random(seed)
    frag = max(index.k + 10, read_len)
    genes = [t for t in index.targets if not t.is_decoy and t.length >= frag + 20]
    if not genes:
        return []
    groups: dict[str, list[str]] = {}
    for t in index.targets:
        groups.setdefault(t.group_id, []).append(t.target_id)
    truth = {t.target_id: frozenset(groups[t.group_id]) for t in genes}
    cases: list[SyntheticCase] = []

    def orient(s: str) -> str:
        return revcomp(s) if rng.random() < 0.5 else s

    for regime, rate in (("normal_0.5pct_err", 0.005), ("divergent_3pct_SNP", 0.03)):
        for _ in range(n_per_regime):
            t = rng.choice(genes)
            st = rng.randint(0, t.length - frag)
            cases.append(SyntheticCase(regime, truth[t.target_id], orient(_mutate(rng, t.seq[st:st + frag], rate))))
    insert = max(index.k + 12, int(read_len * 0.6))
    for _ in range(n_per_regime):
        t = rng.choice(genes)
        st = rng.randint(0, t.length - insert)
        ins = _mutate(rng, t.seq[st:st + insert], 0.005)
        cases.append(SyntheticCase("adapter_overhang", truth[t.target_id], (orient(ins) + ADAPTER)[:read_len]))
    short = max(index.k + 6, min(55, read_len))
    for _ in range(n_per_regime):
        t = rng.choice(genes)
        seg = t.seq[:short] if rng.random() < 0.5 else t.seq[-short:]
        cases.append(SyntheticCase("short_edge_or_splice_junction", truth[t.target_id], orient(_mutate(rng, seg, 0.01))))
    ties = find_tie_pairs(index, frag)
    tmap = {t.target_id: t for t in index.targets}
    for _ in range(n_per_regime if ties else 0):
        a, b, o = rng.choice(ties)
        s = _mutate(rng, tmap[a].seq[o:o + frag], 0.003)
        cases.append(SyntheticCase("paralog_or_strain_tie", frozenset([a, b]), orient(s), tie_pair=(a, b)))
    return cases


@dataclass
class KernelScore:
    kernel: str
    label: str
    real_reads: int
    synthetic_reads: int
    real_mapped_pct: float
    recall: dict[str, float]
    overall_synthetic_recall_pct: float
    mate_tie_symmetry: float
    n_tie_cases: int
    reads_per_sec: float
    gate_passed: bool
    composite: float
    optimality_score: float
    error: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mate_tie_symmetry"] = None if math.isnan(self.mate_tie_symmetry) else round(self.mate_tie_symmetry, 4)
        for k in ("real_mapped_pct", "overall_synthetic_recall_pct", "reads_per_sec", "composite", "optimality_score"):
            d[k] = round(d[k], 4)
        d["recall"] = {k: round(v, 2) for k, v in self.recall.items()}
        return d


def evaluate_kernel(kernel: Kernel, index: KmerIndex, real: Sequence[str], cases: Sequence[SyntheticCase],
                    conditioning_active: bool, tie_symmetry_min: float = TIE_SYMMETRY_MIN) -> KernelScore:
    batch = list(real) + [c.read for c in cases]
    t0 = time.perf_counter()
    res = kernel.map_reads(batch)
    wall = max(time.perf_counter() - t0, 1e-6)
    n_real = len(real)
    real_mapped = sum(1 for c in res.candidates[:n_real] if c is not None) + int(res.decoy[:n_real].sum())
    map_pct = 100.0 * real_mapped / max(1, n_real)
    hits: dict[str, int] = {}
    tot: dict[str, int] = {}
    sym_n = sym_ok = 0
    for i, case in enumerate(cases):
        cand = res.candidates[n_real + i]
        found = {index.ids[t] for t in cand[0]} if cand else set()
        tot[case.regime] = tot.get(case.regime, 0) + 1
        if found & case.truth:
            hits[case.regime] = hits.get(case.regime, 0) + 1
        if case.tie_pair is not None and found:
            sym_n += 1
            if case.tie_pair[0] in found and case.tie_pair[1] in found:
                sym_ok += 1
    recall = {r: 100.0 * hits.get(r, 0) / n for r, n in tot.items()}
    non_tie = [v for r, v in recall.items() if r != "paralog_or_strain_tie"]
    overall = sum(non_tie) / len(non_tie) if non_tie else 0.0
    symmetry = sym_ok / sym_n if sym_n else float("nan")
    rps = len(batch) / wall
    sym_term = 1.0 if math.isnan(symmetry) else symmetry
    composite = (0.35 * map_pct / 100.0 + 0.40 * overall / 100.0 + 0.15 * sym_term
                 + 0.10 * min(1.0, math.log10(max(10.0, rps)) / 6.0))
    gate = True
    score = composite
    if conditioning_active and not math.isnan(symmetry):
        if symmetry < tie_symmetry_min:
            gate = False
        if symmetry < 0.95:
            score = composite * (0.5 + 0.5 * symmetry)
    return KernelScore(kernel.name, kernel.label, n_real, len(cases), map_pct, recall, overall, symmetry,
                       sym_n, rps, gate, composite, score)


@dataclass
class TournamentResult:
    winner: str | None
    board: list[KernelScore]
    n_real: int
    n_synthetic: int
    conditioning_active: bool
    tie_symmetry_min: float

    @property
    def all_disqualified(self) -> bool:
        return self.winner is None

    def as_dict(self) -> dict:
        return {
            "winner": self.winner,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "n_total": self.n_real + self.n_synthetic,
            "conditioning_active": self.conditioning_active,
            "tie_symmetry_min": self.tie_symmetry_min,
            "board": [s.as_dict() for s in self.board],
        }


def run_tournament(kernels: Sequence[Kernel], index: KmerIndex, real_reads: Sequence[str],
                   n_real: int = REAL_READS, n_per_regime: int = SYNTHETIC_PER_REGIME, seed: int = 42,
                   tie_symmetry_min: float = TIE_SYMMETRY_MIN, conditioning_active: bool | None = None) -> TournamentResult:
    real = list(real_reads[:n_real])
    lens = sorted(len(r) for r in real[:2000])
    read_len = lens[len(lens) // 2] if lens else 100
    cases = build_synthetic_truth(index, read_len, n_per_regime, seed)
    n_tie = sum(1 for c in cases if c.tie_pair is not None)
    if conditioning_active is None:
        groups: dict[str, int] = {}
        for t in index.targets:
            if not t.is_decoy:
                groups[t.group_id] = groups.get(t.group_id, 0) + 1
        conditioning_active = len({t.assembly_id for t in index.targets if not t.is_decoy}) > 1 or \
            any(v > 1 for v in groups.values()) or n_tie > 0
    board: list[KernelScore] = []
    for k in kernels:
        if not k.available():
            continue
        try:
            board.append(evaluate_kernel(k, index, real, cases, conditioning_active, tie_symmetry_min))
        except Exception as exc:  # noqa: BLE001 - a broken kernel must not abort the run
            board.append(KernelScore(k.name, k.label, len(real), len(cases), 0, {}, 0, float("nan"), 0, 0, False,
                                     0, 0, error=f"{type(exc).__name__}: {exc}"))
    board.sort(key=lambda s: s.optimality_score, reverse=True)
    passing = [s for s in board if s.gate_passed and not s.error]
    return TournamentResult(passing[0].kernel if passing else None, board, len(real), len(cases),
                            bool(conditioning_active), tie_symmetry_min)
