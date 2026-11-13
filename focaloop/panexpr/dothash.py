"""Component 5C: Unsettled-Restricted d = 2048 int16 Rademacher Hypervectors (DotHash).

Python + ctypes interface to `libdothash.dylib` (Apple M4 ARM NEON + OpenMP):
  1. Deterministic 64-bit syncmer -> 2048 {+1, -1} Rademacher signs (32 splitmix64 blocks),
     accumulated into `int16_t hv[2048]` (4 KB / run) with symmetric saturating clamping
     to [-32767, +32767].
  2. Exact overflow-proof int64 NEON inner products <v(A), v(B)>, unbiased intersection,
     containment, and Jaccard estimators, plus analytical SNR / variance diagnostics.
  3. 24,409 x 2048 int16 Pf8 cohort warm-state index (95.35 MiB RAM) with <2 ms 1-vs-24,409
     and L1-cache-tiled 100-vs-24,409 k-NN search.
  4. DotHash-penalized UCB arm scheduler for SRA BioProject replicate deduplication.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
try:
    import zstandard as zstd
except ImportError:
    zstd = None

from ._medvedev_compile import compile_c_lib

DOTHASH_DIM = 2048
DOTHASH_BYTES_PER_SKETCH = DOTHASH_DIM * 2  # 4,096 bytes = 4 KB

_HERE = Path(__file__).resolve().parent


def _load_libdothash() -> ctypes.CDLL:
    lib = compile_c_lib("dothash", _HERE / "dothash_sketch.c")

    lib.dothash_init.argtypes = []
    lib.dothash_init.restype = None

    lib.dothash_update_u64.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.dothash_update_u64.restype = None

    lib.dothash_sketch_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int16),
    ]
    lib.dothash_sketch_u64.restype = None

    lib.dothash_merge_i16.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
    ]
    lib.dothash_merge_i16.restype = None

    lib.dothash_sketch_batch_csr.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
    ]
    lib.dothash_sketch_batch_csr.restype = None

    lib.dothash_dot_i16.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
    ]
    lib.dothash_dot_i16.restype = ctypes.c_int64

    lib.dothash_norms_sq_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    ]
    lib.dothash_norms_sq_batch.restype = None

    lib.dothash_dots_1_vs_n.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    ]
    lib.dothash_dots_1_vs_n.restype = None

    lib.dothash_knn_1_vs_n_jaccard.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.dothash_knn_1_vs_n_jaccard.restype = None

    lib.dothash_knn_q_vs_n_jaccard.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.dothash_knn_q_vs_n_jaccard.restype = None

    lib.dothash_extract_open_syncmers.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.dothash_extract_open_syncmers.restype = ctypes.c_size_t

    lib.dothash_init()
    return lib


_LIB = _load_libdothash()


def sketch_syncmers(keys: np.ndarray | Sequence[int], deduplicate: bool = True) -> np.ndarray:
    """Sketch 64-bit syncmer keys into an int16[2048] Rademacher hypervector (4 KB)."""
    arr = np.asarray(keys, dtype=np.uint64)
    if deduplicate and arr.size > 1:
        arr = np.unique(arr)
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    hv = np.zeros(DOTHASH_DIM, dtype=np.int16)
    if arr.size > 0:
        _LIB.dothash_sketch_u64(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            arr.size,
            hv.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
    return hv


def update_sketch_inplace(hv: np.ndarray, keys: np.ndarray | Sequence[int]) -> np.ndarray:
    """Incrementally add `keys` into existing int16[2048] hypervector with [-32767, +32767] clamping."""
    if hv.dtype != np.int16 or hv.shape != (DOTHASH_DIM,) or not hv.flags["C_CONTIGUOUS"]:
        raise ValueError("hv must be C-contiguous int16[2048]")
    arr = np.ascontiguousarray(keys, dtype=np.uint64)
    if arr.size > 0:
        _LIB.dothash_update_u64(
            hv.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            arr.size,
        )
    return hv


def merge_sketches_saturated(hv_a: np.ndarray, hv_b: np.ndarray) -> np.ndarray:
    """Return clamp(hv_a + hv_b, -32767, +32767) using ARM NEON saturating int16 arithmetic."""
    out = np.array(hv_a, dtype=np.int16, copy=True, order="C")
    b = np.ascontiguousarray(hv_b, dtype=np.int16)
    _LIB.dothash_merge_i16(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
    )
    return out


def dot_product(hv_a: np.ndarray, hv_b: np.ndarray) -> int:
    """Exact overflow-proof int64 NEON inner product <hv_a, hv_b>."""
    a = np.ascontiguousarray(hv_a, dtype=np.int16)
    b = np.ascontiguousarray(hv_b, dtype=np.int16)
    return int(
        _LIB.dothash_dot_i16(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
    )


def estimate_intersection(hv_a: np.ndarray, hv_b: np.ndarray) -> float:
    """Unbiased intersection size estimator <v(A), v(B)> / d (can be negative on disjoint sets; do not clip when testing unbiasedness)."""
    return dot_product(hv_a, hv_b) / float(DOTHASH_DIM)


def estimate_cardinality(hv: np.ndarray) -> float:
    """Unbiased set size estimator ||v(A)||_2^2 / d."""
    return dot_product(hv, hv) / float(DOTHASH_DIM)


def estimate_containment(
    hv_a: np.ndarray,
    hv_b: np.ndarray,
    card_a: float | None = None,
    clip: bool = True,
) -> float:
    """Estimate C(A, B) = |A cap B| / |A| from 4 KB hypervectors."""
    ca = estimate_cardinality(hv_a) if card_a is None else float(card_a)
    if ca <= 0.0:
        return 0.0
    c_hat = estimate_intersection(hv_a, hv_b) / ca
    return float(min(1.0, max(0.0, c_hat))) if clip else float(c_hat)


def estimate_jaccard(
    hv_a: np.ndarray,
    hv_b: np.ndarray,
    card_a: float | None = None,
    card_b: float | None = None,
    clip: bool = True,
) -> float:
    """Estimate J(A, B) = |A cap B| / |A cup B| from 4 KB hypervectors."""
    ca = estimate_cardinality(hv_a) if card_a is None else float(card_a)
    cb = estimate_cardinality(hv_b) if card_b is None else float(card_b)
    inter = estimate_intersection(hv_a, hv_b)
    if clip:
        inter = max(0.0, inter)
    denom = ca + cb - inter
    if denom <= 1e-9:
        return 0.0
    j_hat = inter / denom
    return float(min(1.0, max(0.0, j_hat))) if clip else float(j_hat)


def analytical_intersection_snr(card_a: int, card_b: int, inter_ab: int, dim: int = DOTHASH_DIM) -> dict[str, float]:
    """Exact theoretical SD and SNR of <v(A), v(B)>/d for Rademacher hypervectors."""
    # Exact variance: (|A||B| + |A cap B|^2 - 2|A cap B|) / d
    var_inter = max(0.0, (card_a * card_b + inter_ab * inter_ab - 2 * inter_ab) / float(dim))
    sd_inter = math.sqrt(var_inter)
    snr = float(inter_ab / sd_inter) if sd_inter > 0 else float("inf")
    se_containment = sd_inter / max(1, card_a)
    return {
        "sd_intersection": sd_inter,
        "snr": snr,
        "se_containment_a_in_b": se_containment,
    }


class DotHashCohortIndex:
    """Warm-state cohort matrix of shape (n_samples, 2048) int16 (95.35 MiB for 24,409 Pf8 isolates)."""

    def __init__(self, matrix: np.ndarray, cardinalities: np.ndarray | None = None) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != DOTHASH_DIM:
            raise ValueError(f"Expected (N, {DOTHASH_DIM}) matrix, got {matrix.shape}")
        self.matrix = np.ascontiguousarray(matrix, dtype=np.int16)
        self.n_samples = int(self.matrix.shape[0])
        if cardinalities is None:
            norms_sq = np.empty(self.n_samples, dtype=np.int64)
            _LIB.dothash_norms_sq_batch(
                self.matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                self.n_samples,
                norms_sq.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                0,
            )
            self.cardinalities = (norms_sq.astype(np.float32) / float(DOTHASH_DIM))
        else:
            self.cardinalities = np.ascontiguousarray(cardinalities, dtype=np.float32)

    @property
    def ram_bytes(self) -> int:
        return int(self.matrix.nbytes)

    @property
    def ram_mib(self) -> float:
        return float(self.matrix.nbytes / (1024.0 * 1024.0))

    @classmethod
    def from_csr(
        cls,
        all_keys: np.ndarray,
        offsets: np.ndarray,
        n_threads: int = 0,
    ) -> "DotHashCohortIndex":
        keys_c = np.ascontiguousarray(all_keys, dtype=np.uint64)
        offs_c = np.ascontiguousarray(offsets, dtype=np.int64)
        n_samples = len(offs_c) - 1
        mat = np.empty((n_samples, DOTHASH_DIM), dtype=np.int16)
        _LIB.dothash_sketch_batch_csr(
            keys_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            offs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            n_samples,
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            n_threads,
        )
        # Use pure hypervector self-norms ||v(B_i)||^2 / 2048 (Tanimoto ratio estimator:
        # zero sidecar storage beyond the 4 KB sketch & variance-reduced via Cauchy-Schwarz!)
        return cls(mat, cardinalities=None)

    @classmethod
    def from_syncmer_sets(
        cls,
        syncmer_sets: Sequence[np.ndarray],
        n_threads: int = 0,
    ) -> "DotHashCohortIndex":
        lengths = [len(s) for s in syncmer_sets]
        offsets = np.zeros(len(syncmer_sets) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        all_keys = np.concatenate([np.asarray(s, dtype=np.uint64) for s in syncmer_sets]) if offsets[-1] > 0 else np.empty(0, dtype=np.uint64)
        return cls.from_csr(all_keys, offsets, n_threads=n_threads)

    def knn_1_vs_n(
        self,
        query_hv: np.ndarray,
        k_top: int = 5,
        query_card: float | None = None,
        n_threads: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_hv, dtype=np.int16)
        qc = estimate_cardinality(q) if query_card is None else float(query_card)
        top_idx = np.empty(k_top, dtype=np.int32)
        top_sim = np.empty(k_top, dtype=np.float32)
        _LIB.dothash_knn_1_vs_n_jaccard(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_float(qc),
            self.matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            self.cardinalities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n_samples,
            k_top,
            top_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            top_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_threads,
        )
        return top_idx, top_sim

    def knn_q_vs_n(
        self,
        queries_hv: np.ndarray,
        k_top: int = 5,
        query_cards: np.ndarray | None = None,
        n_threads: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_mat = np.ascontiguousarray(queries_hv, dtype=np.int16)
        n_q = q_mat.shape[0]
        if query_cards is None:
            norms_sq = np.empty(n_q, dtype=np.int64)
            _LIB.dothash_norms_sq_batch(
                q_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                n_q,
                norms_sq.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                n_threads,
            )
            q_cards = (norms_sq.astype(np.float32) / float(DOTHASH_DIM))
        else:
            q_cards = np.ascontiguousarray(query_cards, dtype=np.float32)

        top_idx = np.empty((n_q, k_top), dtype=np.int32)
        top_sim = np.empty((n_q, k_top), dtype=np.float32)
        _LIB.dothash_knn_q_vs_n_jaccard(
            q_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            q_cards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_q,
            self.matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            self.cardinalities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n_samples,
            k_top,
            top_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            top_sim.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_threads,
        )
        return top_idx, top_sim

    def to_zstd_bytes(self, level: int = 3) -> bytes:
        if zstd is None:
            raise ImportError("zstandard is required for to_zstd_bytes")
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(self.matrix.tobytes())


def extract_open_syncmers_from_reads(reads: Sequence[str]) -> np.ndarray:
    """Extract sorted unique canonical (k=31, s=15, t=8) open syncmers from read strings."""
    if not reads:
        return np.empty(0, dtype=np.uint64)
    buf = "\n".join(reads).encode("ascii", "replace")
    n = len(buf)
    if n < 31:
        return np.empty(0, dtype=np.uint64)
    out = np.empty(n, dtype=np.uint64)
    written = int(
        _LIB.dothash_extract_open_syncmers(
            buf,
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
    )
    if written == 0:
        return np.empty(0, dtype=np.uint64)
    return np.unique(out[:written])


def update_sketch_from_reads(
    hv: np.ndarray,
    reads: Sequence[str],
    seen_syncmers: set[int] | None = None,
) -> int:
    """Update a library's 4 KB int16[2048] DotHash sketch with novel open syncmers from ``reads``.

    When ``seen_syncmers`` is passed, only syncmers not yet observed in that library are folded into ``hv``.
    Returns count of new syncmers folded in.
    """
    keys = extract_open_syncmers_from_reads(reads)
    if keys.size == 0:
        return 0
    if seen_syncmers is not None:
        fresh = [int(x) for x in keys.tolist() if int(x) not in seen_syncmers]
        if not fresh:
            return 0
        seen_syncmers.update(fresh)
        keys = np.asarray(fresh, dtype=np.uint64)
    update_sketch_inplace(hv, keys)
    return int(keys.size)


def cosine_similarity(hv_a: np.ndarray, hv_b: np.ndarray, clip: bool = True) -> float:
    """Overflow-safe int64 cosine similarity <v(A), v(B)> / (||v(A)||_2 * ||v(B)||_2)."""
    na = float(dot_product(hv_a, hv_a))
    nb = float(dot_product(hv_b, hv_b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    sim = float(dot_product(hv_a, hv_b)) / math.sqrt(na * nb)
    return float(min(1.0, max(0.0, sim))) if clip else float(sim)


@dataclass
class UCBRunResult:
    scheduler: str
    waves_to_target: int
    http_206_spots_streamed: int
    final_settled_fraction: float
    arm_pull_counts: list[int]
    per_tissue_pulls: dict[int, int]


def simulate_bioproject_ucb_dedup(
    arm_wave0_syncmers: Sequence[np.ndarray],
    arm_spot_batches: Sequence[Sequence[np.ndarray]],
    arm_tissues: Sequence[int],
    universe_unsettled: set[int],
    target_coverage: float = 0.90,
    ucb_c: float = 1.2,
    use_dothash_penalty: bool = True,
    spots_per_wave: int = 250,
) -> UCBRunResult:
    """Simulate Focaloop Core Judge HTTP 206 streaming across SRA BioProject arms.

    Each arm `l` was probed at Wave-0 (1 cheap probe chunk), yielding its unsettled-restricted
    4 KB DotHash hypervector `v(l)`. At each Wave-1+ step, the Judge picks one SRA arm `l` to
    pull `spots_per_wave` spots via HTTP 206 Range request:
      - Standard UCB:       score(l) = mu_hat(l) + c * sqrt(ln(t + 1) / n_l)
      - DotHash-Penalized:  score(l) = (mu_hat(l) + c * sqrt(ln(t + 1) / n_l)) * (1 - C_hat(l, E_t))
        where `E_t` is the 4 KB running DotHash sketch of unsettled syncmers already discovered!
    """
    n_arms = len(arm_wave0_syncmers)
    arm_hvs = [sketch_syncmers(s, deduplicate=True) for s in arm_wave0_syncmers]
    arm_cards = [max(1.0, float(len(np.unique(s)))) for s in arm_wave0_syncmers]

    # Wave-0 initialization: 1 probe batch already observed per arm
    discovered: set[int] = set()
    hv_explored = np.zeros(DOTHASH_DIM, dtype=np.int16)

    pull_counts = [1] * n_arms
    # Initial empirical yield estimate from Wave-0 probe density (normalized [0, 1])
    max_w0 = max(arm_cards)
    emp_reward = [c / max_w0 for c in arm_cards]
    batch_cursors = [0] * n_arms
    max_total_batches = sum(len(b) for b in arm_spot_batches)

    target_count = int(math.ceil(target_coverage * len(universe_unsettled)))
    waves = 0

    while len(discovered) < target_count and waves < max_total_batches:
        # Score all non-exhausted arms
        best_arm = -1
        best_score = -1e30
        log_t = math.log(waves + n_arms + 1.0)
        for l_idx in range(n_arms):
            if batch_cursors[l_idx] >= len(arm_spot_batches[l_idx]):
                continue
            base_ucb = emp_reward[l_idx] + ucb_c * math.sqrt(log_t / pull_counts[l_idx])
            if use_dothash_penalty and len(discovered) > 0:
                c_hat = estimate_containment(arm_hvs[l_idx], hv_explored, card_a=arm_cards[l_idx], clip=True)
                novelty_factor = max(0.02, 1.0 - c_hat)
                score = base_ucb * novelty_factor
            else:
                score = base_ucb
            if score > best_score:
                best_score = score
                best_arm = l_idx

        if best_arm < 0:
            break

        # Stream next HTTP 206 spot batch from `best_arm`
        batch = arm_spot_batches[best_arm][batch_cursors[best_arm]]
        batch_cursors[best_arm] += 1
        waves += 1

        # Novel unsettled syncmers discovered in this wave
        new_keys = [int(x) for x in batch if int(x) in universe_unsettled and int(x) not in discovered]
        if new_keys:
            discovered.update(new_keys)
            if use_dothash_penalty:
                update_sketch_inplace(hv_explored, np.asarray(new_keys, dtype=np.uint64))

        # Update arm's marginal novelty rate (fraction of batch yielding new unsettled syncmers)
        marginal_gain = len(new_keys) / max(1.0, arm_cards[best_arm] * 0.35)
        n_prev = pull_counts[best_arm]
        emp_reward[best_arm] = 0.65 * emp_reward[best_arm] + 0.35 * min(1.0, marginal_gain)
        pull_counts[best_arm] = n_prev + 1

    tissue_pulls: dict[int, int] = {}
    for l_idx, t_id in enumerate(arm_tissues):
        # Count Wave-1+ HTTP 206 pulls (pull_counts - 1)
        tissue_pulls[t_id] = tissue_pulls.get(t_id, 0) + (pull_counts[l_idx] - 1)

    return UCBRunResult(
        scheduler="dothash_ucb" if use_dothash_penalty else "standard_ucb",
        waves_to_target=waves,
        http_206_spots_streamed=waves * spots_per_wave,
        final_settled_fraction=float(len(discovered) / max(1, len(universe_unsettled))),
        arm_pull_counts=[p - 1 for p in pull_counts],
        per_tissue_pulls=tissue_pulls,
    )
