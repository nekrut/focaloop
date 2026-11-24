"""First-class architectural hooks for the 5 next-horizon Focaloop capabilities:

  1. BaitPrefilter        — Stage-0 pan-repository 31-mer bait pre-screening
  2. EqClassBackendProtocol & HolobiontLocusRegistry
                          — Pangenome graph / colored de Bruijn / holobiont support
  3. coalesce_residual_spans & apply_external_priors
                          — Retroactive cross-modal fusion + <2% residual re-streaming
  4. SteeringMask         — Real-time instrument ReadUntil / acquisition steering
  5. HypothesisLedger     — Reusable holdout + online alpha-investing for AI/human loops
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import pyarrow as pa

from . import blueprint as _blueprint_mod
from .state import (
    BAIT_PREFILTER_SCHEMA,
    FocaloopStateStore,
    GRAPH_LOCUS_MAP_SCHEMA,
    HYPOTHESIS_LEDGER_SCHEMA,
    RESIDUAL_SPANS_SCHEMA,
    STEERING_DECISION_SCHEMA,
)


def _install_packaged_blueprint_search_hook() -> None:
    """Ensure ``blueprint_search_path`` checks packaged ``focaloop/blueprints`` and ``sys.prefix`` installs."""
    if getattr(_blueprint_mod, "_packaged_search_installed", False):
        return
    orig_search = _blueprint_mod.blueprint_search_path

    def _extended_blueprint_search_path() -> list[Path]:
        paths = list(orig_search())
        pkg_bp = Path(_blueprint_mod.__file__).resolve().parent / "blueprints"
        prefix_bp = Path(sys.prefix) / "share" / "focaloop" / "blueprints"
        for extra in (pkg_bp, prefix_bp):
            if extra not in paths:
                paths.append(extra)
        return paths

    _blueprint_mod.blueprint_search_path = _extended_blueprint_search_path  # type: ignore[assignment]
    _blueprint_mod._packaged_search_installed = True  # type: ignore[attr-defined]


_install_packaged_blueprint_search_hook()

K_DEFAULT = 31
_LUT = np.full(256, 255, dtype=np.uint8)
for _ch, _v in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("a", 0), ("c", 1), ("g", 2), ("t", 3)):
    _LUT[ord(_ch)] = _v


def encode_canonical_kmers(seqs: Sequence[str], k: int = K_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Encode equal-padded sequence batch into canonical 2-bit uint64 k-mers (k <= 31)."""
    if k < 1 or k > 31:
        raise ValueError(f"k must be in [1, 31], got {k}")
    if not seqs:
        return np.zeros((0, 0), dtype=np.uint64), np.zeros((0, 0), dtype=bool)
    max_len = max(len(s) for s in seqs)
    n_win = max_len - k + 1
    if n_win <= 0:
        return np.zeros((len(seqs), 0), dtype=np.uint64), np.zeros((len(seqs), 0), dtype=bool)
    mask = np.uint64((1 << (2 * k)) - 1)
    buf = np.full((len(seqs), max_len), ord("N"), dtype=np.uint8)
    for i, s in enumerate(seqs):
        raw = np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)
        buf[i, : raw.size] = raw
    codes = _LUT[buf]
    bad = codes == 255
    clean = np.where(bad, 0, codes).astype(np.uint64)
    fwd = np.zeros((len(seqs), n_win), dtype=np.uint64)
    rev = np.zeros((len(seqs), n_win), dtype=np.uint64)
    invalid = np.zeros((len(seqs), n_win), dtype=bool)
    for j in range(k):
        col = clean[:, j : j + n_win]
        fwd = (fwd << np.uint64(2)) | col
        rev = rev | ((np.uint64(3) - col) << np.uint64(2 * j))
        invalid |= bad[:, j : j + n_win]
    return np.minimum(fwd & mask, rev & mask), ~invalid


def build_kmer_bait(seqs: Iterable[str], k: int = K_DEFAULT) -> np.ndarray:
    """Sorted deduplicated canonical k-mer array from query or target sequences."""
    chunks: list[np.ndarray] = []
    for s in seqs:
        if len(s) >= k:
            codes, ok = encode_canonical_kmers([s], k=k)
            chunks.append(codes[ok])
    if not chunks:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(chunks))


# ============================================================================
# Hook 1: Stage-0 Live Pan-Repository Discovery Bait Pre-Filter
# ============================================================================


def _wilson_lower_95(hits: int, total: int) -> float:
    if total <= 0 or hits <= 0:
        return 0.0
    z = 1.959963984540054
    p = hits / total
    denom = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    rad = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    return max(0.0, (center - rad) / denom)


@dataclass(frozen=True)
class BaitHitSummary:
    query_name: str
    run_id: str
    stream_tier: str
    probe_bytes: int
    probe_reads: int
    hit_reads: int
    hit_rate: float
    wilson_ci95_low: float
    pass_gate: bool


class BaitPrefilter:
    """Stage-0 k-mer bait scanner for screening 1,000+ public SRA/ENA runs before alignment."""

    def __init__(
        self,
        query_name: str,
        bait_sequences: Sequence[str],
        k: int = K_DEFAULT,
        min_kmer_hits_per_read: int = 2,
        min_hit_reads: int = 2,
    ) -> None:
        self.query_name = query_name
        self.k = k
        self.min_kmer_hits_per_read = min_kmer_hits_per_read
        self.min_hit_reads = min_hit_reads
        self.bait = build_kmer_bait(bait_sequences, k=k)

    def screen_reads(
        self,
        run_id: str,
        reads: Sequence[str],
        probe_bytes: int,
        stream_tier: str = "ena_https_range",
        store: FocaloopStateStore | None = None,
    ) -> BaitHitSummary:
        n_reads = len(reads)
        if n_reads == 0 or self.bait.size == 0:
            summary = BaitHitSummary(self.query_name, run_id, stream_tier, probe_bytes, n_reads, 0, 0.0, 0.0, False)
        else:
            codes, ok = encode_canonical_kmers(reads, k=self.k)
            if codes.shape[1] == 0:
                hit_reads = 0
            else:
                matches = np.isin(codes, self.bait, assume_unique=False) & ok
                hit_reads = int(np.count_nonzero(matches.sum(axis=1) >= self.min_kmer_hits_per_read))
            rate = hit_reads / n_reads
            ci_low = _wilson_lower_95(hit_reads, n_reads)
            summary = BaitHitSummary(
                query_name=self.query_name,
                run_id=run_id,
                stream_tier=stream_tier,
                probe_bytes=probe_bytes,
                probe_reads=n_reads,
                hit_reads=hit_reads,
                hit_rate=rate,
                wilson_ci95_low=ci_low,
                pass_gate=hit_reads >= self.min_hit_reads,
            )
        if store is not None:
            store.append(
                "bait_prefilter",
                {
                    "query_name": [summary.query_name],
                    "run_id": [summary.run_id],
                    "stream_tier": [summary.stream_tier],
                    "probe_bytes": [summary.probe_bytes],
                    "probe_reads": [summary.probe_reads],
                    "hit_reads": [summary.hit_reads],
                    "hit_rate": [summary.hit_rate],
                    "wilson_ci95_low": [summary.wilson_ci95_low],
                    "pass_gate": [summary.pass_gate],
                },
                schema=BAIT_PREFILTER_SCHEMA,
            )
        return summary


# ============================================================================
# Hook 2: Reference-Free Pangenome Graph & Holobiont Equivalence-Class Backend
# ============================================================================


@dataclass(frozen=True)
class EqClassRecord:
    candidate_units: tuple[str, ...]
    candidate_orientations: tuple[bool, ...]
    aln_loglik: tuple[float, ...]
    fragment_count: int


@runtime_checkable
class EqClassBackendProtocol(Protocol):
    """Unified Stage-2 backend protocol across linear SAM/PAF, cDBG, and pangenome graphs."""

    backend_kind: Literal["linear_sam", "transcriptome_cdbg", "colored_dbg", "pangenome_graph"]

    def map_slice_to_eq_classes(
        self,
        r1_seqs: Sequence[str],
        r2_seqs: Sequence[str] | None = None,
    ) -> tuple[list[EqClassRecord], list[int]]:
        """Return `(eq_classes, residual_read_indices)` without collapsing sequence ties."""
        ...


class HolobiontLocusRegistry:
    """Tracks multi-strain, graph-snarl, plasmid, and cross-kingdom holobiont loci."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, str, str, str, str, bool, int]] = []

    def register(
        self,
        unit_id: str,
        unit_kind: Literal["transcript", "gene_body", "cdbg_color", "vg_snarl", "plasmid_contig"],
        group_id: str,
        assembly_id: str,
        kingdom: Literal["host", "bacteria", "fungi", "virus", "protist", "archaea"],
        length_bp: int,
        hgt_mobile_flag: bool = False,
    ) -> None:
        self._rows.append((unit_id, unit_kind, group_id, assembly_id, kingdom, hgt_mobile_flag, length_bp))

    def to_arrow(self) -> pa.Table:
        cols = list(zip(*self._rows)) if self._rows else [(), (), (), (), (), (), ()]
        return pa.Table.from_pydict(
            {
                "unit_id": list(cols[0]),
                "unit_kind": list(cols[1]),
                "group_id": list(cols[2]),
                "assembly_id": list(cols[3]),
                "kingdom": list(cols[4]),
                "hgt_mobile_flag": list(cols[5]),
                "length_bp": list(cols[6]),
            },
            schema=GRAPH_LOCUS_MAP_SCHEMA,
        )

    def persist(self, store: FocaloopStateStore) -> Path:
        return store.write("graph_locus_map", self.to_arrow(), schema=GRAPH_LOCUS_MAP_SCHEMA)


# ============================================================================
# Hook 3: Retroactive Cross-Modal Fusion & Residual Unmapped Byte-Range Index
# ============================================================================


def _splitmix64(x: np.ndarray) -> np.ndarray:
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


def sketch_residual_kmers(seqs: Sequence[str], max_sketch: int = 32, k: int = K_DEFAULT) -> list[int]:
    """Bottom-k canonical 31-mer sketch so retroactive novel-locus queries match residual spans."""
    kmers = build_kmer_bait(seqs, k=k)
    if kmers.size == 0:
        return []
    hashes = _splitmix64(kmers)
    order = np.argsort(hashes)[:max_sketch]
    return [int(v) for v in np.sort(kmers[order])]


@dataclass(frozen=True)
class ResidualSpan:
    run_id: str
    stream_tier: str
    byte_start: int
    byte_end: int
    residual_kind: Literal["unmapped", "soft_clipped", "chimeric_split"]
    n_reads: int
    kmer_sketch_u64: tuple[int, ...]


def coalesce_residual_spans(
    run_id: str,
    stream_tier: str,
    read_offsets: Sequence[tuple[int, int, str]],
    residual_kind: Literal["unmapped", "soft_clipped", "chimeric_split"] = "unmapped",
    merge_gap_bytes: int = 65536,
    max_sketch: int = 32,
    store: FocaloopStateStore | None = None,
) -> list[ResidualSpan]:
    """Coalesce nearby unmapped/clipped read byte intervals (default 64 KiB bgzf block gap)."""
    if not read_offsets:
        return []
    ordered = sorted(read_offsets, key=lambda item: (item[0], item[1]))
    spans: list[ResidualSpan] = []
    cur_start, cur_end, first_seq = ordered[0]
    cur_seqs = [first_seq]
    for start, end, seq in ordered[1:]:
        if start - cur_end <= merge_gap_bytes:
            cur_end = max(cur_end, end)
            cur_seqs.append(seq)
        else:
            spans.append(
                ResidualSpan(
                    run_id=run_id,
                    stream_tier=stream_tier,
                    byte_start=cur_start,
                    byte_end=cur_end,
                    residual_kind=residual_kind,
                    n_reads=len(cur_seqs),
                    kmer_sketch_u64=tuple(sketch_residual_kmers(cur_seqs, max_sketch=max_sketch)),
                )
            )
            cur_start, cur_end, cur_seqs = start, end, [seq]
    spans.append(
        ResidualSpan(
            run_id=run_id,
            stream_tier=stream_tier,
            byte_start=cur_start,
            byte_end=cur_end,
            residual_kind=residual_kind,
            n_reads=len(cur_seqs),
            kmer_sketch_u64=tuple(sketch_residual_kmers(cur_seqs, max_sketch=max_sketch)),
        )
    )
    if store is not None:
        store.append(
            "residual_spans",
            {
                "run_id": [s.run_id for s in spans],
                "stream_tier": [s.stream_tier for s in spans],
                "byte_start": [s.byte_start for s in spans],
                "byte_end": [s.byte_end for s in spans],
                "residual_kind": [s.residual_kind for s in spans],
                "n_reads": [s.n_reads for s in spans],
                "kmer_sketch_u64": [list(s.kmer_sketch_u64) for s in spans],
            },
            schema=RESIDUAL_SPANS_SCHEMA,
        )
    return spans


def apply_external_priors(
    candidate_targets: Sequence[str],
    aln_loglik: Sequence[float],
    prior_log_weights: dict[str, float],
) -> np.ndarray:
    """Condition saved equivalence-class alignment log-likelihoods on cross-modal priors in log space."""
    if len(candidate_targets) != len(aln_loglik):
        raise ValueError("candidate_targets and aln_loglik must have identical length")
    if not candidate_targets:
        return np.zeros(0, dtype=np.float64)
    combined = np.array(
        [ll + prior_log_weights.get(t, 0.0) for t, ll in zip(candidate_targets, aln_loglik)],
        dtype=np.float64,
    )
    shift = np.max(combined)
    weights = np.exp(combined - shift)
    total = float(weights.sum())
    return weights / total if total > 0.0 else np.full(len(candidate_targets), 1.0 / len(candidate_targets))


# ============================================================================
# Hook 4: Real-Time Instrument Steering (ONT ReadUntil / Adaptive Acquisition)
# ============================================================================


class SteeringMask:
    """Compiles Stage-4/5 target sufficiency states into live ReadUntil masks and BED intervals."""

    def __init__(
        self,
        enrich_sequences: dict[str, str],
        deplete_sequences: dict[str, str],
        k: int = K_DEFAULT,
        min_hits: int = 2,
    ) -> None:
        self.k = k
        self.min_hits = min_hits
        self.enrich_targets = tuple(sorted(enrich_sequences))
        self.deplete_targets = tuple(sorted(deplete_sequences))
        self.enrich_kmers = build_kmer_bait(enrich_sequences.values(), k=k)
        raw_deplete = build_kmer_bait(deplete_sequences.values(), k=k)
        # Never eject a read sharing k-mers with an unsettled enrichment target
        self.deplete_kmers = np.setdiff1d(raw_deplete, self.enrich_kmers, assume_unique=True)
        self.halt_run = len(self.enrich_targets) == 0

    def decide_read_until(self, prefix_seq: str) -> Literal["accept", "unblock_eject", "neutral"]:
        if self.halt_run:
            return "unblock_eject"
        if len(prefix_seq) < self.k:
            return "neutral"
        codes, ok = encode_canonical_kmers([prefix_seq], k=self.k)
        valid = codes[ok]
        if valid.size == 0:
            return "neutral"
        if self.enrich_kmers.size > 0 and int(np.count_nonzero(np.isin(valid, self.enrich_kmers))) >= self.min_hits:
            return "accept"
        if self.deplete_kmers.size > 0 and int(np.count_nonzero(np.isin(valid, self.deplete_kmers))) >= self.min_hits:
            return "unblock_eject"
        return "neutral"

    def export_read_until_bed(
        self,
        coords: dict[str, tuple[str, int, int]],
        out_bed: str | Path,
    ) -> Path:
        path = Path(out_bed)
        lines: list[str] = []
        for tid in self.enrich_targets:
            if tid in coords:
                chrom, start, end = coords[tid]
                lines.append(f"{chrom}\t{start}\t{end}\t{tid}\tenrich\n")
        for tid in self.deplete_targets:
            if tid in coords:
                chrom, start, end = coords[tid]
                lines.append(f"{chrom}\t{start}\t{end}\t{tid}\tdeplete\n")
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def record_telemetry(self, store: FocaloopStateStore, iteration: int, reason: str) -> Path:
        return store.append(
            "steering_decisions",
            {
                "iteration": [iteration],
                "n_enrich_targets": [len(self.enrich_targets)],
                "n_deplete_targets": [len(self.deplete_targets)],
                "n_enrich_kmers": [int(self.enrich_kmers.size)],
                "n_deplete_kmers": [int(self.deplete_kmers.size)],
                "halt_run": [self.halt_run],
                "reason": [reason],
            },
            schema=STEERING_DECISION_SCHEMA,
        )


# ============================================================================
# Hook 5: Interactive Hypothesis Search Ledger (Reusable Holdout + Alpha-Investing)
# ============================================================================


def is_holdout_block(run_id: str, block_index: int, holdout_fraction: float = 0.10, salt: str = "focaloop-v1") -> bool:
    """Deterministic block-level split keeping `holdout_fraction` of streamed blocks out of discovery."""
    if holdout_fraction <= 0.0:
        return False
    if holdout_fraction >= 1.0:
        return True
    digest = hashlib.blake2b(f"{salt}:{run_id}:{block_index}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(digest, "little") / float(1 << 64)
    return u < holdout_fraction


@dataclass(frozen=True)
class QueryVerdict:
    query_index: int
    query_id: str
    actor: str
    hypothesis_spec: str
    discovery_estimate: float
    discovery_pvalue: float
    holdout_estimate: float | None
    holdout_pvalue: float | None
    alpha_allocated: float
    alpha_wealth_after: float
    confirmed: bool
    holdout_available: bool = True


class HypothesisLedger:
    """Online Generalized Alpha-Investing + Reusable Holdout guard for human/AI hypothesis loops."""

    def __init__(
        self,
        alpha_budget: float = 0.05,
        spend_fraction: float = 0.10,
        discovery_reward: float = 0.04,
        holdout_alpha: float = 0.05,
    ) -> None:
        if not (0.0 < alpha_budget < 1.0):
            raise ValueError("alpha_budget must be in (0, 1)")
        self.wealth = alpha_budget
        self.spend_fraction = spend_fraction
        self.discovery_reward = discovery_reward
        self.holdout_alpha = holdout_alpha
        self._query_index = 0
        self._seen_specs: set[str] = set()

    @classmethod
    def from_store(cls, store: FocaloopStateStore, **kwargs: float) -> "HypothesisLedger":
        """Resume alpha-wealth, query index, and previously evaluated specs from ``hypothesis_ledger``."""
        ledger = cls(**kwargs)
        if store.exists("hypothesis_ledger"):
            tbl = store.read("hypothesis_ledger")
            if tbl.num_rows:
                ledger.wealth = float(tbl.column("alpha_wealth_after")[-1].as_py())
                ledger._query_index = int(tbl.column("query_index")[-1].as_py())
                for s in tbl.column("hypothesis_spec").to_pylist():
                    if s:
                        ledger._seen_specs.add(str(s).strip())
        return ledger

    def evaluate_query(
        self,
        query_id: str,
        actor: str,
        hypothesis_spec: str,
        discovery_estimate: float,
        discovery_pvalue: float,
        holdout_estimate: float | None = None,
        holdout_pvalue: float | None = None,
        store: FocaloopStateStore | None = None,
        *,
        holdout_available: bool = True,
        allow_reward: bool = True,
    ) -> QueryVerdict:
        self._query_index += 1
        spec_norm = str(hypothesis_spec or "").strip()
        is_repeat_spec = bool(spec_norm and spec_norm in self._seen_specs)
        is_primary_seed = (
            str(query_id) == "q0001"
            and str(actor).startswith("focaloop-")
            and spec_norm.endswith((":primary_discovery", ":logan_primary"))
        )

        d_est = float(discovery_estimate) if discovery_estimate is not None and math.isfinite(float(discovery_estimate)) else 0.0
        d_p = float(discovery_pvalue) if discovery_pvalue is not None and math.isfinite(float(discovery_pvalue)) else 1.0

        eff_holdout_avail = bool(
            holdout_available
            and not is_primary_seed
            and holdout_estimate is not None
            and holdout_pvalue is not None
            and math.isfinite(float(holdout_estimate))
            and math.isfinite(float(holdout_pvalue))
        )

        if is_primary_seed:
            alpha_j = 0.0
            confirmed = False
            reward = 0.0
            h_est_out: float | None = None
            h_p_out: float | None = None
        else:
            alpha_j = self.wealth * self.spend_fraction
            if eff_holdout_avail:
                h_est_f = float(holdout_estimate)  # type: ignore[arg-type]
                h_p_f = float(holdout_pvalue)  # type: ignore[arg-type]
                same_sign = (d_est > 0.0 and h_est_f > 0.0) or (d_est < 0.0 and h_est_f < 0.0)
                confirmed = (d_p <= alpha_j) and (h_p_f <= self.holdout_alpha) and same_sign
                reward = self.discovery_reward if (confirmed and allow_reward and not is_repeat_spec) else 0.0
                h_est_out = h_est_f
                h_p_out = h_p_f
            else:
                confirmed = False
                reward = 0.0
                h_est_out = None
                h_p_out = None

        if spec_norm:
            self._seen_specs.add(spec_norm)
        self.wealth = max(0.0, self.wealth - alpha_j + reward)

        verdict = QueryVerdict(
            query_index=self._query_index,
            query_id=query_id,
            actor=actor,
            hypothesis_spec=hypothesis_spec,
            discovery_estimate=d_est,
            discovery_pvalue=d_p,
            holdout_estimate=h_est_out,
            holdout_pvalue=h_p_out,
            alpha_allocated=alpha_j,
            alpha_wealth_after=self.wealth,
            confirmed=confirmed,
            holdout_available=eff_holdout_avail,
        )
        if store is not None:
            store.append(
                "hypothesis_ledger",
                {
                    "query_index": [verdict.query_index],
                    "query_id": [verdict.query_id],
                    "actor": [verdict.actor],
                    "hypothesis_spec": [verdict.hypothesis_spec],
                    "discovery_estimate": [verdict.discovery_estimate],
                    "discovery_pvalue": [verdict.discovery_pvalue],
                    "holdout_estimate": [verdict.holdout_estimate],
                    "holdout_pvalue": [verdict.holdout_pvalue],
                    "alpha_allocated": [verdict.alpha_allocated],
                    "alpha_wealth_after": [verdict.alpha_wealth_after],
                    "confirmed": [verdict.confirmed],
                },
                schema=HYPOTHESIS_LEDGER_SCHEMA,
            )
        return verdict
