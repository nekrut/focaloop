"""Pre-Alignment SNP/Indel Island Split & Repeat-Robust ``--pc`` Routing Module.

Wraps the compiled C hot-path library ``libisland_pc.dylib`` (``island_pc_core.c``) to provide:
  1. Bijective ``splitmix64`` canonical s-mer middle-offset (``t* = 8``) open syncmers (``k=31, s=15``)
     with exact context-free density ``1 / (k - s + 1) = 1/17`` even on 81%-AT *Plasmodium* genomes.
  2. Wu & Medvedev (2026) repeat-robust ``--pc`` novelty divergence estimation with exact Hamming-1
     collision correction ``d1_bar(s_a)`` and safeguarded 5-step Newton-Raphson inversion.
  3. Exact Blanca, Harris, Koslicki & Medvedev (2022) lag-(k-1) overlap confidence intervals for BOTH
     dense k=31 k-mers and middle-offset (``t* = 8``) open syncmers (accounting for the exact ``d <= 8``
     repulsion zone ``c_2(d) = 0``, shared-window covariance ``2 / (m * (d + m))``, and ratio delta method).
  4. ``< 2 ms`` Wave-0 2,000-read mutated-k-mer island-width (``W``) & unique-anchor coordinate-shift
     (``delta``) SNP / deletion / insertion classifier (Blanca 2022 + Hera et al. 2025) with ``VarGeno``
     Phred gating, ``(1 - r_total)^(2k)`` survival correction, high-divergence ``kr > 0.35`` multi-event
     ``delta`` + ``--pc`` fallback, and concrete ``PanExpr`` / ``MalariaVar`` mapper routing verdicts.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence

import numpy as np

from ._medvedev_compile import compile_c_lib

K_DEFAULT = 31
S_DEFAULT = 15
T_STAR_DEFAULT = (K_DEFAULT - S_DEFAULT) // 2  # 8
INDEL_PSEUDOALIGN_MAX = 0.003                  # 0.3% total indel event rate disqualifies piscem/salmon
KR_HIGH_DIVERGENCE_MAX = 0.35                  # r > 1.129% at k=31 triggers merged-island fallback

PSEUDOALIGN_KERNELS = ("piscem", "salmon", "kallisto", "themisto")

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# ctypes structures matching island_pc_core.c ABI
# ---------------------------------------------------------------------------

class _CIndexInfo(ctypes.Structure):
    _fields_ = [
        ("total_kmer_positions", ctypes.c_uint64),
        ("distinct_kmers", ctypes.c_uint64),
        ("unique_kmer_positions", ctypes.c_uint64),
        ("total_syncmer_positions", ctypes.c_uint64),
        ("distinct_syncmers", ctypes.c_uint64),
        ("D1_dense", ctypes.c_uint64),
        ("d1_dense_bar", ctypes.c_double),
        ("D1_sync", ctypes.c_uint64),
        ("d1_sync_bar", ctypes.c_double),
        ("dense_d1_computed", ctypes.c_int),
    ]


class _CPcCiResult(ctypes.Structure):
    _fields_ = [
        ("r_hat", ctypes.c_double),
        ("y_hat", ctypes.c_double),
        ("g_deriv", ctypes.c_double),
        ("var_y_overlap", ctypes.c_double),
        ("var_y_binom", ctypes.c_double),
        ("variance_inflation_factor", ctypes.c_double),
        ("se_r_overlap", ctypes.c_double),
        ("se_r_binom", ctypes.c_double),
        ("ci_low_overlap", ctypes.c_double),
        ("ci_high_overlap", ctypes.c_double),
        ("ci_low_binom", ctypes.c_double),
        ("ci_high_binom", ctypes.c_double),
    ]


class _CProbeResult(ctypes.Structure):
    _fields_ = [
        ("kernel_ms", ctypes.c_double),
        ("n_reads_processed", ctypes.c_uint64),
        ("total_kmer_windows", ctypes.c_uint64),
        ("total_ocean_kmers", ctypes.c_uint64),
        ("total_novel_kmers", ctypes.c_uint64),
        ("unique_ocean_kmers", ctypes.c_uint64),
        ("ocean_fraction", ctypes.c_double),
        ("syncmer_probe_total", ctypes.c_uint64),
        ("syncmer_probe_novel", ctypes.c_uint64),
        ("r_pc_syncmer_raw", ctypes.c_double),
        ("r_pc_syncmer_bio", ctypes.c_double),
        ("r_pc_syncmer_ci_low", ctypes.c_double),
        ("r_pc_syncmer_ci_high", ctypes.c_double),
        ("r_pc_dense_raw", ctypes.c_double),
        ("r_pc_dense_bio", ctypes.c_double),
        ("c_sub_iso", ctypes.c_uint64),
        ("c_del_iso", ctypes.c_uint64),
        ("c_ins_iso", ctypes.c_uint64),
        ("c_ins_1bp_iso", ctypes.c_uint64),
        ("c_ins_multibp_iso", ctypes.c_uint64),
        ("c_del_microhom_iso", ctypes.c_uint64),
        ("c_merged_uniq", ctypes.c_uint64),
        ("c_merged_sub_only", ctypes.c_uint64),
        ("c_merged_pos_delta", ctypes.c_uint64),
        ("c_merged_neg_delta", ctypes.c_uint64),
        ("sum_del_len_iso", ctypes.c_int64),
        ("sum_ins_len_iso", ctypes.c_int64),
        ("sum_merged_pos_delta", ctypes.c_int64),
        ("sum_merged_neg_delta", ctypes.c_int64),
        ("c_rep_w_le_30", ctypes.c_uint64),
        ("c_rep_w_eq_31", ctypes.c_uint64),
        ("c_rep_w_ge_32", ctypes.c_uint64),
        ("n_end_censored_islands", ctypes.c_uint64),
        ("n_lowq_skipped_islands", ctypes.c_uint64),
        ("mean_base_err_admitted", ctypes.c_double),
        ("p_s_hat", ctypes.c_double),
        ("p_d_hat", ctypes.c_double),
        ("p_i_hat", ctypes.c_double),
        ("r_total_hat", ctypes.c_double),
        ("kr_hat", ctypes.c_double),
        ("high_divergence_merged_islands", ctypes.c_int),
        ("used_high_div_fallback", ctypes.c_int),
    ]


class _CBaselineDiv(ctypes.Structure):
    _fields_ = [
        ("query_total_positions", ctypes.c_uint64),
        ("query_novel_positions", ctypes.c_uint64),
        ("query_distinct_kmers", ctypes.c_uint64),
        ("shared_distinct_kmers", ctypes.c_uint64),
        ("ref_distinct_kmers", ctypes.c_uint64),
        ("query_sync_positions", ctypes.c_uint64),
        ("query_sync_novel_positions", ctypes.c_uint64),
        ("mash_jaccard_exact", ctypes.c_double),
        ("r_mash_exact", ctypes.c_double),
        ("mash_jaccard_sketch2000", ctypes.c_double),
        ("r_mash_sketch2000", ctypes.c_double),
        ("r_containment_query_set", ctypes.c_double),
        ("r_containment_ref_set", ctypes.c_double),
        ("r_containment_pos_naive", ctypes.c_double),
        ("r_pc_dense", ctypes.c_double),
        ("r_pc_syncmer", ctypes.c_double),
    ]


def _load_lib() -> ctypes.CDLL:
    lib = compile_c_lib("island_pc", _HERE / "island_pc_core.c")

    lib.ipc_splitmix64.argtypes = [ctypes.c_uint64]
    lib.ipc_splitmix64.restype = ctypes.c_uint64

    lib.ipc_splitmix64_inv.argtypes = [ctypes.c_uint64]
    lib.ipc_splitmix64_inv.restype = ctypes.c_uint64

    lib.ipc_syncmer_density.argtypes = [
        ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ipc_syncmer_density.restype = None

    lib.ipc_build_index.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int,
    ]
    lib.ipc_build_index.restype = ctypes.c_void_p

    lib.ipc_get_index_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CIndexInfo)]
    lib.ipc_get_index_info.restype = None

    lib.ipc_free_index.argtypes = [ctypes.c_void_p]
    lib.ipc_free_index.restype = None

    lib.ipc_solve_pc_divergence.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
    ]
    lib.ipc_solve_pc_divergence.restype = ctypes.c_double

    lib.ipc_blanca_variance_ci.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
        ctypes.c_double, ctypes.POINTER(_CPcCiResult),
    ]
    lib.ipc_blanca_variance_ci.restype = None

    lib.ipc_classify_reads.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.POINTER(_CProbeResult),
    ]
    lib.ipc_classify_reads.restype = ctypes.c_int

    lib.ipc_compute_baselines.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(_CBaselineDiv),
    ]
    lib.ipc_compute_baselines.restype = ctypes.c_int

    lib.ipc_save_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ipc_save_index.restype = ctypes.c_int

    lib.ipc_load_index.argtypes = [ctypes.c_char_p]
    lib.ipc_load_index.restype = ctypes.c_void_p

    return lib


_LIB = _load_lib()


# ---------------------------------------------------------------------------
# Public Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexInfo:
    total_kmer_positions: int
    distinct_kmers: int
    unique_kmer_positions: int
    total_syncmer_positions: int
    distinct_syncmers: int
    D1_dense: int
    d1_dense_bar: float
    D1_sync: int
    d1_sync_bar: float
    dense_d1_computed: bool


@dataclass(frozen=True)
class PcConfidenceInterval:
    r_hat: float
    y_hat: float
    g_deriv: float
    var_y_overlap: float
    var_y_binom: float
    variance_inflation_factor: float
    se_r_overlap: float
    se_r_binom: float
    ci_low_overlap: float
    ci_high_overlap: float
    ci_low_binom: float
    ci_high_binom: float


@dataclass(frozen=True)
class MapperRoutingVerdict:
    flags: tuple[str, ...]
    pseudoaligners_disqualified: bool
    high_divergence_merged_islands: bool
    panexpr_recommended: str
    panexpr_eligible: tuple[str, ...]
    panexpr_excluded: dict[str, str]
    malariavar_recommended: str
    malariavar_eligible: tuple[str, ...]
    malariavar_excluded: dict[str, str]
    rationale: str


@dataclass(frozen=True)
class IslandProbeSummary:
    kernel_ms: float
    wall_ms: float
    n_reads_processed: int
    total_kmer_windows: int
    total_ocean_kmers: int
    total_novel_kmers: int
    unique_ocean_kmers: int
    ocean_fraction: float

    syncmer_probe_total: int
    syncmer_probe_novel: int
    r_pc_syncmer_raw: float
    r_pc_syncmer_bio: float
    r_pc_syncmer_ci_low: float
    r_pc_syncmer_ci_high: float
    r_pc_dense_raw: float
    r_pc_dense_bio: float

    c_sub_iso: int
    c_del_iso: int
    c_ins_iso: int
    c_ins_1bp_iso: int
    c_ins_multibp_iso: int
    c_del_microhom_iso: int
    c_merged_uniq: int
    c_merged_sub_only: int
    c_merged_pos_delta: int
    c_merged_neg_delta: int
    c_rep_w_le_30: int
    c_rep_w_eq_31: int
    c_rep_w_ge_32: int
    n_end_censored_islands: int
    n_lowq_skipped_islands: int
    mean_base_err_admitted: float

    p_s_hat: float
    p_d_hat: float
    p_i_hat: float
    p_indel_hat: float
    r_total_hat: float
    kr_hat: float
    high_divergence_merged_islands: bool
    used_high_div_fallback: bool
    routing: MapperRoutingVerdict

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PackedReadBatch:
    seq_bytes: bytes
    qual_bytes: bytes | None
    offsets: np.ndarray | None
    lengths: np.ndarray
    n_reads: int


# ---------------------------------------------------------------------------
# Hashing, Syncmer Density & Theoretical Covariance Helpers
# ---------------------------------------------------------------------------

def splitmix64(x: int) -> int:
    return int(_LIB.ipc_splitmix64(ctypes.c_uint64(x & 0xFFFFFFFFFFFFFFFF)))


def splitmix64_inv(z: int) -> int:
    return int(_LIB.ipc_splitmix64_inv(ctypes.c_uint64(z & 0xFFFFFFFFFFFFFFFF)))


def measure_syncmer_density(seq: str | bytes, k: int = K_DEFAULT, s: int = S_DEFAULT, use_splitmix: bool = True) -> dict:
    b = seq.encode("ascii") if isinstance(seq, str) else seq
    n_km = ctypes.c_uint64(0)
    n_sy = ctypes.c_uint64(0)
    _LIB.ipc_syncmer_density(b, len(b), k, s, 1 if use_splitmix else 0, ctypes.byref(n_km), ctypes.byref(n_sy))
    km = int(n_km.value)
    sy = int(n_sy.value)
    theo = 1.0 / (k - s + 1)
    emp = (sy / km) if km > 0 else 0.0
    return {
        "n_kmers": km,
        "n_syncmers": sy,
        "empirical_density": emp,
        "theoretical_density": theo,
        "relative_bias_pct": 100.0 * (emp - theo) / theo if theo > 0 else 0.0,
    }


def compare_syncmer_density_modes(seq: str | bytes, k: int = K_DEFAULT, s: int = S_DEFAULT) -> dict:
    raw = measure_syncmer_density(seq, k=k, s=s, use_splitmix=False)
    smx = measure_syncmer_density(seq, k=k, s=s, use_splitmix=True)
    return {"raw_lexicographic": raw, "splitmix64_bijective": smx}


def theoretical_open_syncmer_c2(d: int, k: int = K_DEFAULT, s: int = S_DEFAULT) -> float:
    """Exact joint selection probability Pr(Z_0 = 1, Z_d = 1) for middle-offset t* = (k - s)//2 open syncmers."""
    m = k - s + 1
    t_star = (k - s) // 2
    if d == 0:
        return 1.0 / m
    d = abs(d)
    if 1 <= d <= t_star:
        return 0.0
    if t_star + 1 <= d < m:
        return 2.0 / (m * (d + m))
    return 1.0 / (m * m)


def solve_pc_divergence(n_novel: int, l_probe: int, k: int = K_DEFAULT, d1_bar: float = 0.0) -> tuple[float, float]:
    """Solve Wu & Medvedev (2026) --pc equation; returns (r_hat, g_prime_at_r_hat)."""
    gp = ctypes.c_double(0.0)
    r = float(_LIB.ipc_solve_pc_divergence(max(0, int(n_novel)), max(0, int(l_probe)), k, float(d1_bar), ctypes.byref(gp)))
    return r, float(gp.value)


def blanca_variance_ci(
    n_novel: int,
    l_probe: int,
    k: int = K_DEFAULT,
    s: int = S_DEFAULT,
    is_syncmer: bool = False,
    d1_bar: float = 0.0,
    lag_pairs: Sequence[int] | np.ndarray | None = None,
    total_windows: int = 0,
    z_crit: float = 1.959963984540054,
) -> PcConfidenceInterval:
    out = _CPcCiResult()
    lp_ptr = None
    lp_arr = None
    if lag_pairs is not None:
        lp_arr = np.ascontiguousarray(lag_pairs, dtype=np.uint64)
        if lp_arr.size < k - 1:
            raise ValueError(f"lag_pairs must have at least {k - 1} elements")
        lp_ptr = lp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    _LIB.ipc_blanca_variance_ci(
        max(0, int(n_novel)),
        max(0, int(l_probe)),
        int(k),
        int(s),
        1 if is_syncmer else 0,
        float(d1_bar),
        lp_ptr,
        max(0, int(total_windows)),
        float(z_crit),
        ctypes.byref(out),
    )
    return PcConfidenceInterval(
        r_hat=float(out.r_hat),
        y_hat=float(out.y_hat),
        g_deriv=float(out.g_deriv),
        var_y_overlap=float(out.var_y_overlap),
        var_y_binom=float(out.var_y_binom),
        variance_inflation_factor=float(out.variance_inflation_factor),
        se_r_overlap=float(out.se_r_overlap),
        se_r_binom=float(out.se_r_binom),
        ci_low_overlap=float(out.ci_low_overlap),
        ci_high_overlap=float(out.ci_high_overlap),
        ci_low_binom=float(out.ci_low_binom),
        ci_high_binom=float(out.ci_high_binom),
    )


# ---------------------------------------------------------------------------
# Concrete PanExpr & MalariaVar Mapper Routing Logic
# ---------------------------------------------------------------------------

def build_mapper_routing_verdict(
    p_s_hat: float,
    p_d_hat: float,
    p_i_hat: float,
    kr_hat: float,
    high_divergence_merged_islands: bool,
    ocean_fraction: float,
    spliced: bool = False,
    strict_tie_gate: bool = False,
    indel_threshold: float = INDEL_PSEUDOALIGN_MAX,
    usable_evidence: bool = True,
) -> MapperRoutingVerdict:
    has_nan = any(math.isnan(x) for x in (p_s_hat, p_d_hat, p_i_hat, kr_hat, ocean_fraction))
    if has_nan or not usable_evidence:
        why_str = "insufficient usable high-Q interior bases or NaN divergence estimate; conservative gap-aware fallback"
        pan_excl = {pk: why_str for pk in PSEUDOALIGN_KERNELS}
        return MapperRoutingVerdict(
            flags=("INSUFFICIENT_USABLE_BASES", "GAP_SENSITIVE_ALIGNMENT_REQUIRED"),
            pseudoaligners_disqualified=True,
            high_divergence_merged_islands=False,
            panexpr_recommended="mappy",
            panexpr_eligible=("mappy", "star_genome_m16") if spliced else ("mappy", "bwa_mem2_spa"),
            panexpr_excluded=pan_excl,
            malariavar_recommended="minimap2 -ax sr",
            malariavar_eligible=("minimap2 -ax sr", "bwa-mem2"),
            malariavar_excluded={"strobealign": why_str},
            rationale=why_str,
        )

    p_indel = p_d_hat + p_i_hat
    flags: list[str] = []
    pan_excl: dict[str, str] = {}
    mal_excl: dict[str, str] = {}

    indel_high = p_indel >= indel_threshold
    if high_divergence_merged_islands or kr_hat > KR_HIGH_DIVERGENCE_MAX:
        flags.append("HIGH_DIVERGENCE_MERGED_ISLANDS")
    if indel_high:
        flags.append("INDEL_PSEUDOALIGN_DISQUALIFIED")
    if p_d_hat >= 0.0015 or p_i_hat >= 0.0015:
        flags.append("GAP_SENSITIVE_ALIGNMENT_REQUIRED")

    disqualify_pseudo = bool(indel_high or high_divergence_merged_islands or kr_hat > KR_HIGH_DIVERGENCE_MAX)
    reasons_parts: list[str] = []

    if disqualify_pseudo:
        why_parts = []
        if indel_high:
            why_parts.append(f"p_d+p_i={p_indel:.4%} >= {indel_threshold:.2%} pseudoalignment cliff")
        if high_divergence_merged_islands or kr_hat > KR_HIGH_DIVERGENCE_MAX:
            why_parts.append(f"kr={kr_hat:.3f} > {KR_HIGH_DIVERGENCE_MAX} / ocean={ocean_fraction:.2%} merged islands")
        why_str = "; ".join(why_parts)
        for pk in PSEUDOALIGN_KERNELS:
            pan_excl[pk] = why_str
        reasons_parts.append(f"Disqualified pseudoaligners ({', '.join(PSEUDOALIGN_KERNELS)}): {why_str}")

        # PanExpr indel/high-divergence routing
        if spliced:
            pan_rec = "star_genome_m16"
            pan_elig = ("star_genome_m16", "mappy", "strobealign_N16")
        elif strict_tie_gate:
            pan_excl["strobealign_N16"] = "paired rescue breaks equal-score strain/paralog ties under strict_tie_gate"
            pan_rec = "mappy"
            pan_elig = ("mappy", "bwa_mem2_spa")
        else:
            pan_rec = "strobealign_N16" if (p_indel < 0.008 and kr_hat <= 0.80) else "mappy"
            pan_elig = ("strobealign_N16", "mappy", "star_genome_m16") if spliced else ("strobealign_N16", "mappy", "bwa_mem2_spa")

        # MalariaVar AT-rich / indel / high-divergence routing
        mal_excl["strobealign"] = why_str
        mal_rec = "minimap2 -ax sr"
        mal_elig = ("minimap2 -ax sr", "bwa-mem2")
        reasons_parts.append(f"Routed PanExpr -> {pan_rec}, MalariaVar -> {mal_rec}")
    else:
        flags.append("LOW_INDEL_PSEUDOALIGN_ELIGIBLE")
        if spliced and strict_tie_gate:
            pan_rec = "piscem"
            pan_elig = ("piscem", "salmon", "star_genome_m16", "mappy")
        else:
            pan_rec = "piscem"
            pan_elig = ("piscem", "salmon", "strobealign_N16", "mappy")
        mal_rec = "strobealign"
        mal_elig = ("strobealign", "minimap2 -ax sr", "bwa-mem2")
        reasons_parts.append(
            f"Clean SNP/low-indel regime (p_s={p_s_hat:.4%}, p_indel={p_indel:.4%}, kr={kr_hat:.3f}): "
            f"PanExpr -> {pan_rec}, MalariaVar -> {mal_rec}"
        )

    return MapperRoutingVerdict(
        flags=tuple(flags),
        pseudoaligners_disqualified=disqualify_pseudo,
        high_divergence_merged_islands=bool(high_divergence_merged_islands or kr_hat > KR_HIGH_DIVERGENCE_MAX),
        panexpr_recommended=pan_rec,
        panexpr_eligible=pan_elig,
        panexpr_excluded=pan_excl,
        malariavar_recommended=mal_rec,
        malariavar_eligible=mal_elig,
        malariavar_excluded=mal_excl,
        rationale=" | ".join(reasons_parts),
    )


# ---------------------------------------------------------------------------
# Fast Read-Batch Packing & Reference Index Wrapper
# ---------------------------------------------------------------------------

def pack_read_batch(
    reads: Sequence[str] | Sequence[tuple[str, str]],
    quals: Sequence[str] | None = None,
) -> PackedReadBatch:
    n = len(reads)
    if n == 0:
        return PackedReadBatch(b"", None, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), 0)

    first = reads[0]
    if isinstance(first, tuple):
        seqs_list = [r[0] for r in reads]  # type: ignore[index]
        quals_list = [r[1] for r in reads] if quals is None else list(quals)  # type: ignore[index]
    else:
        seqs_list = list(reads)  # type: ignore[arg-type]
        quals_list = list(quals) if quals is not None else None

    if quals_list is not None:
        if len(quals_list) != n:
            raise ValueError(f"quals count ({len(quals_list)}) != reads count ({n})")
        for idx_r, (s_str, q_str) in enumerate(zip(seqs_list, quals_list)):
            if len(q_str) != len(s_str):
                raise ValueError(f"Read {idx_r} qual length ({len(q_str)}) != seq length ({len(s_str)})")

    seq_bytes = "".join(seqs_list).encode("latin1")
    qual_bytes = "".join(quals_list).encode("latin1") if quals_list is not None else None

    L0 = len(seqs_list[0])
    if len(seq_bytes) == n * L0 and (qual_bytes is None or len(qual_bytes) == n * L0):
        # Verify fixed stride without slow generator conversion when uniform
        # Check via newline-free join or min/max length only if needed; check all lengths via C-speed map(len)
        lens_set = set(map(len, seqs_list))
        if len(lens_set) == 1:
            return PackedReadBatch(
                seq_bytes=seq_bytes,
                qual_bytes=qual_bytes,
                offsets=None,
                lengths=np.array([L0], dtype=np.int32),
                n_reads=n,
            )

    lengths = np.fromiter(map(len, seqs_list), dtype=np.int32, count=n)
    offsets = np.empty(n, dtype=np.int32)
    offsets[0] = 0
    if n > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])

    return PackedReadBatch(
        seq_bytes=seq_bytes,
        qual_bytes=qual_bytes,
        offsets=np.ascontiguousarray(offsets, dtype=np.int32),
        lengths=np.ascontiguousarray(lengths, dtype=np.int32),
        n_reads=n,
    )


def _pack_super_contigs(contigs: Sequence[str], max_contigs: int = 4096, max_super_bp: int = 8_000_000) -> list[str]:
    """Pack many short contigs/transcripts using N*256 separators so ctg_u16 never saturates."""
    if len(contigs) <= max_contigs and all(len(c) <= 0x7FFFFFFF for c in contigs):
        return list(contigs)
    sep = "N" * 256
    out: list[str] = []
    cur: list[str] = []
    cur_bp = 0
    for c in contigs:
        add = len(c) + (len(sep) if cur else 0)
        if cur and cur_bp + add > max_super_bp:
            out.append(sep.join(cur))
            cur = [c]
            cur_bp = len(c)
        else:
            cur.append(c)
            cur_bp += add
    if cur:
        out.append(sep.join(cur))
    return out


class ReferenceIslandPCIndex:
    """Native C k=31 canonical unique-anchor table + t*=8 open-syncmer --pc index."""

    @staticmethod
    def _read_info(ptr: int) -> IndexInfo:
        cinfo = _CIndexInfo()
        _LIB.ipc_get_index_info(ctypes.c_void_p(ptr), ctypes.byref(cinfo))
        return IndexInfo(
            total_kmer_positions=int(cinfo.total_kmer_positions),
            distinct_kmers=int(cinfo.distinct_kmers),
            unique_kmer_positions=int(cinfo.unique_kmer_positions),
            total_syncmer_positions=int(cinfo.total_syncmer_positions),
            distinct_syncmers=int(cinfo.distinct_syncmers),
            D1_dense=int(cinfo.D1_dense),
            d1_dense_bar=float(cinfo.d1_dense_bar),
            D1_sync=int(cinfo.D1_sync),
            d1_sync_bar=float(cinfo.d1_sync_bar),
            dense_d1_computed=bool(cinfo.dense_d1_computed),
        )

    def __init__(self, contigs: Sequence[str] | str | None, compute_dense_d1: bool = False, _raw_ptr: int | None = None) -> None:
        if _raw_ptr is not None:
            self._ptr: int | None = int(_raw_ptr)
            self.info = self._read_info(self._ptr)
            return
        if contigs is None:
            raise ValueError("contigs required unless loading pre-serialized index")
        if isinstance(contigs, str):
            contigs = [contigs]
        if not contigs:
            raise ValueError("At least one reference contig is required")

        packed = _pack_super_contigs(contigs)
        encoded = [c.encode("ascii", "replace") for c in packed]
        n = len(encoded)
        c_seqs = (ctypes.c_char_p * n)(*encoded)
        c_lens = (ctypes.c_int64 * n)(*(len(b) for b in encoded))

        ptr = _LIB.ipc_build_index(c_seqs, c_lens, n, 1 if compute_dense_d1 else 0)
        if not ptr:
            raise MemoryError("Failed to allocate ReferenceIslandPCIndex in C")
        self._ptr = int(ptr)
        self.info = self._read_info(self._ptr)

    def save(self, path: str | Path) -> Path:
        """Persist compiled binary index to disk for zero-rebuild offline loading."""
        if not getattr(self, "_ptr", None):
            raise RuntimeError("ReferenceIslandPCIndex is closed")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        rc = _LIB.ipc_save_index(ctypes.c_void_p(self._ptr), str(tmp).encode("utf-8"))
        if rc != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ipc_save_index failed with code {rc}")
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceIslandPCIndex":
        """Load pre-serialized binary index from disk."""
        p = Path(path)
        ptr = _LIB.ipc_load_index(str(p).encode("utf-8"))
        if not ptr:
            raise RuntimeError(f"ipc_load_index failed for {p}")
        return cls(None, _raw_ptr=int(ptr))

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            _LIB.ipc_free_index(ctypes.c_void_p(self._ptr))
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "ReferenceIslandPCIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def classify_reads(
        self,
        reads: PackedReadBatch | Sequence[str] | Sequence[tuple[str, str]],
        quals: Sequence[str] | None = None,
        min_base_q: int = 20,
        max_syncmer_err_prob: float = 0.05,
        spliced: bool = False,
        strict_tie_gate: bool = False,
        indel_threshold: float = INDEL_PSEUDOALIGN_MAX,
    ) -> IslandProbeSummary:
        if not self._ptr:
            raise RuntimeError("ReferenceIslandPCIndex is closed")
        t_wall0 = time.perf_counter()
        batch = reads if isinstance(reads, PackedReadBatch) else pack_read_batch(reads, quals=quals)
        cout = _CProbeResult()
        if batch.n_reads > 0:
            off_ptr = (
                batch.offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                if batch.offsets is not None else None
            )
            len_ptr = batch.lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            seq_buf = (ctypes.c_char * len(batch.seq_bytes)).from_buffer_copy(batch.seq_bytes)
            qual_buf = (
                (ctypes.c_char * len(batch.qual_bytes)).from_buffer_copy(batch.qual_bytes)
                if batch.qual_bytes is not None else None
            )
            rc = _LIB.ipc_classify_reads(
                ctypes.c_void_p(self._ptr),
                ctypes.cast(seq_buf, ctypes.c_void_p),
                ctypes.cast(qual_buf, ctypes.c_void_p) if qual_buf is not None else None,
                off_ptr,
                len_ptr,
                int(batch.n_reads),
                int(min_base_q),
                float(max_syncmer_err_prob),
                ctypes.byref(cout),
            )
            if rc != 0:
                raise RuntimeError(f"ipc_classify_reads failed with code {rc}")
        wall_ms = (time.perf_counter() - t_wall0) * 1e3

        usable = (
            int(cout.n_reads_processed) > 0
            and (
                batch.qual_bytes is None
                or int(cout.syncmer_probe_total) > 0
                or float(cout.mean_base_err_admitted) > 0.0
            )
        )
        verdict = build_mapper_routing_verdict(
            p_s_hat=float(cout.p_s_hat),
            p_d_hat=float(cout.p_d_hat),
            p_i_hat=float(cout.p_i_hat),
            kr_hat=float(cout.kr_hat),
            high_divergence_merged_islands=bool(cout.high_divergence_merged_islands),
            ocean_fraction=float(cout.ocean_fraction),
            spliced=spliced,
            strict_tie_gate=strict_tie_gate,
            indel_threshold=indel_threshold,
            usable_evidence=usable,
        )

        return IslandProbeSummary(
            kernel_ms=float(cout.kernel_ms),
            wall_ms=wall_ms,
            n_reads_processed=int(cout.n_reads_processed),
            total_kmer_windows=int(cout.total_kmer_windows),
            total_ocean_kmers=int(cout.total_ocean_kmers),
            total_novel_kmers=int(cout.total_novel_kmers),
            unique_ocean_kmers=int(cout.unique_ocean_kmers),
            ocean_fraction=float(cout.ocean_fraction),
            syncmer_probe_total=int(cout.syncmer_probe_total),
            syncmer_probe_novel=int(cout.syncmer_probe_novel),
            r_pc_syncmer_raw=float(cout.r_pc_syncmer_raw),
            r_pc_syncmer_bio=float(cout.r_pc_syncmer_bio),
            r_pc_syncmer_ci_low=float(cout.r_pc_syncmer_ci_low),
            r_pc_syncmer_ci_high=float(cout.r_pc_syncmer_ci_high),
            r_pc_dense_raw=float(cout.r_pc_dense_raw),
            r_pc_dense_bio=float(cout.r_pc_dense_bio),
            c_sub_iso=int(cout.c_sub_iso),
            c_del_iso=int(cout.c_del_iso),
            c_ins_iso=int(cout.c_ins_iso),
            c_ins_1bp_iso=int(cout.c_ins_1bp_iso),
            c_ins_multibp_iso=int(cout.c_ins_multibp_iso),
            c_del_microhom_iso=int(cout.c_del_microhom_iso),
            c_merged_uniq=int(cout.c_merged_uniq),
            c_merged_sub_only=int(cout.c_merged_sub_only),
            c_merged_pos_delta=int(cout.c_merged_pos_delta),
            c_merged_neg_delta=int(cout.c_merged_neg_delta),
            c_rep_w_le_30=int(cout.c_rep_w_le_30),
            c_rep_w_eq_31=int(cout.c_rep_w_eq_31),
            c_rep_w_ge_32=int(cout.c_rep_w_ge_32),
            n_end_censored_islands=int(cout.n_end_censored_islands),
            n_lowq_skipped_islands=int(cout.n_lowq_skipped_islands),
            mean_base_err_admitted=float(cout.mean_base_err_admitted),
            p_s_hat=float(cout.p_s_hat),
            p_d_hat=float(cout.p_d_hat),
            p_i_hat=float(cout.p_i_hat),
            p_indel_hat=float(cout.p_d_hat + cout.p_i_hat),
            r_total_hat=float(cout.r_total_hat),
            kr_hat=float(cout.kr_hat),
            high_divergence_merged_islands=bool(cout.high_divergence_merged_islands),
            used_high_div_fallback=bool(cout.used_high_div_fallback),
            routing=verdict,
        )

    def compute_baselines(self, query_seqs: Sequence[str] | str, sketch_size: int = 2000) -> dict:
        if not self._ptr:
            raise RuntimeError("ReferenceIslandPCIndex is closed")
        if isinstance(query_seqs, str):
            query_seqs = [query_seqs]
        enc = [s.encode("ascii") for s in query_seqs]
        n = len(enc)
        c_seqs = (ctypes.c_char_p * n)(*enc)
        c_lens = (ctypes.c_int64 * n)(*(len(b) for b in enc))
        out = _CBaselineDiv()
        rc = _LIB.ipc_compute_baselines(ctypes.c_void_p(self._ptr), c_seqs, c_lens, n, int(sketch_size), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"ipc_compute_baselines failed with code {rc}")
        return {f[0]: getattr(out, f[0]) for f in out._fields_}


def route_pangenome_wave0(
    assemblies: dict[str, ReferenceIslandPCIndex],
    reads: PackedReadBatch | Sequence[str] | Sequence[tuple[str, str]],
    use_syncmer_ci: bool = True,
    use_overlap_ci: bool = True,
    spliced: bool = False,
) -> dict:
    """Wave-0 pangenome strain tournament using Wu-Medvedev --pc + Blanca/syncmer overlap 95% CIs."""
    batch = reads if isinstance(reads, PackedReadBatch) else pack_read_batch(reads)
    lag_pairs = np.zeros(K_DEFAULT - 1, dtype=np.uint64)
    if batch.n_reads > 0:
        if batch.offsets is None:
            m_w = max(0, int(batch.lengths[0]) - K_DEFAULT + 1)
            for d in range(1, K_DEFAULT):
                lag_pairs[d - 1] = batch.n_reads * max(0, m_w - d)
        else:
            m_wins = np.maximum(0, batch.lengths.astype(np.int64) - K_DEFAULT + 1)
            for d in range(1, K_DEFAULT):
                lag_pairs[d - 1] = int(np.maximum(0, m_wins - d).sum())

    per_asm: dict[str, dict] = {}
    for asm_id, idx in assemblies.items():
        summary = idx.classify_reads(batch, spliced=spliced, strict_tie_gate=(len(assemblies) > 1))
        if use_syncmer_ci:
            ci = blanca_variance_ci(
                summary.syncmer_probe_novel,
                summary.syncmer_probe_total,
                k=K_DEFAULT,
                s=S_DEFAULT,
                is_syncmer=True,
                d1_bar=idx.info.d1_sync_bar,
                lag_pairs=lag_pairs,
                total_windows=summary.total_kmer_windows,
            )
        else:
            ci = blanca_variance_ci(
                summary.total_novel_kmers,
                summary.total_kmer_windows,
                k=K_DEFAULT,
                s=S_DEFAULT,
                is_syncmer=False,
                d1_bar=idx.info.d1_dense_bar,
                lag_pairs=lag_pairs,
                total_windows=summary.total_kmer_windows,
            )
        lo = ci.ci_low_overlap if use_overlap_ci else ci.ci_low_binom
        hi = ci.ci_high_overlap if use_overlap_ci else ci.ci_high_binom
        per_asm[asm_id] = {
            "r_hat": ci.r_hat,
            "ci_low": lo,
            "ci_high": hi,
            "ci_overlap": (ci.ci_low_overlap, ci.ci_high_overlap),
            "ci_binom": (ci.ci_low_binom, ci.ci_high_binom),
            "summary": summary,
        }

    leader = min(per_asm, key=lambda a: per_asm[a]["r_hat"])
    leader_hi = per_asm[leader]["ci_high"]
    retained = [a for a, st in per_asm.items() if st["ci_low"] <= leader_hi]
    eliminated = [a for a in per_asm if a not in retained]
    strictly_separated = len(retained) == 1

    return {
        "leader": leader,
        "retained_assemblies": retained,
        "eliminated_assemblies": eliminated,
        "strictly_separated": strictly_separated,
        "multi_assembly_zoom_triggered": not strictly_separated,
        "leader_summary": per_asm[leader]["summary"],
        "assemblies": per_asm,
    }
