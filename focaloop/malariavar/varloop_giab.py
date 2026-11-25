"""VarLoop Diploid Eukaryotic/Human Streaming Variant Caller & GIAB Benchmark Runner.

Implements:
  - `PagedContigAccumulator`: Python/ctypes interface to `native/varloop_core.c`'s lock-free
    contig-sharded + 64-kb lazy-paged SAM accumulator (bounded RAM across 64-vCPU workers,
    62-bit exact PCR dedup without 16.77 Mb or 32-contig wrap, O(1) contig lookup).
  - `DiploidParams` & Beta-Binomial overdispersed diploid genotyper (`0/0`, `0/1`, `1/1`, `1/2`,
    exact Phred `QUAL` and `GQ`, spanning-deletion `*` awareness, indel length-bias calibration).
  - Multi-k (`k=31, 55`) Simple Omnitig + C semi-global Phred read-haplotype active-region reassembler
    rescuing soft-clipped medium indels, tandem-repeat indels, and clustered MNPs/CMRG variants.
  - Focused `.crai` container range planner & streamer (`stream_crai_regions`) for targeted panels
    such as the 273 GIAB `CMRG v1.00` challenging medically relevant genes.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from typing import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

from . import caller as C
from . import engine as E
from . import focused_cohort as FC
from . import genome as G
from .native._loader import load_shared_lib
from . import omnitig_zoom as OZ

LN10 = math.log(10.0)
_LIB = None


def _lib() -> ct.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = load_shared_lib("varloop_core", ["varloop_core.c"], use_openmp=False)

    u8p = ct.POINTER(ct.c_uint8)
    u16p = ct.POINTER(ct.c_uint16)
    u64p = ct.POINTER(ct.c_uint64)
    i16p = ct.POINTER(ct.c_int16)
    i32p = ct.POINTER(ct.c_int32)
    i64p = ct.POINTER(ct.c_int64)
    f64p = ct.POINTER(ct.c_double)

    lib.vl_kset_build.argtypes = [u8p, i64p, i64p, ct.c_int64, u64p, ct.c_uint64, ct.c_int32, i64p]
    lib.vl_kset_build.restype = ct.c_int

    lib.vl_fastq_sieve_paired.argtypes = [
        ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        u64p, ct.c_uint64, ct.c_int32, ct.c_int32, ct.c_uint64,
        ct.c_int32, ct.c_int32, ct.c_int64, i64p,
    ]
    lib.vl_fastq_sieve_paired.restype = ct.c_int

    lib.vl_acc_create.argtypes = [
        ct.c_int32, ct.c_char_p, i64p, i64p, i64p, i64p, u8p,
        ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double,
        ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
    ]
    lib.vl_acc_create.restype = ct.c_void_p

    lib.vl_acc_restrict_regions.argtypes = [ct.c_void_p, i32p, i32p, i32p, ct.c_int32]
    lib.vl_acc_restrict_regions.restype = None

    lib.vl_acc_set_zoom_intervals.argtypes = [ct.c_void_p, ct.c_int32, i32p, i32p, ct.c_int32]
    lib.vl_acc_set_zoom_intervals.restype = None

    lib.vl_acc_consume_sam_fd.argtypes = [ct.c_void_p, ct.c_int]
    lib.vl_acc_consume_sam_fd.restype = ct.c_int

    lib.vl_acc_feed_sam_bytes.argtypes = [ct.c_void_p, u8p, ct.c_size_t]
    lib.vl_acc_feed_sam_bytes.restype = ct.c_int

    lib.vl_acc_get_summary.argtypes = [ct.c_void_p, i64p, i64p, i64p, i64p]
    lib.vl_acc_get_summary.restype = None

    lib.vl_acc_count_contig_candidates.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_int32]
    lib.vl_acc_count_contig_candidates.restype = ct.c_int64

    lib.vl_acc_extract_contig_candidates.argtypes = [
        ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_int32,
        i32p, u16p, u16p, u16p, u16p, u8p,
    ]
    lib.vl_acc_extract_contig_candidates.restype = ct.c_int64

    lib.vl_acc_query_positions.argtypes = [ct.c_void_p, ct.c_int32, i32p, ct.c_int64, i32p, i32p, i32p]
    lib.vl_acc_query_positions.restype = None

    lib.vl_acc_count_contig_indels.argtypes = [ct.c_void_p, ct.c_int32, ct.c_int32, i64p]
    lib.vl_acc_count_contig_indels.restype = ct.c_int64

    lib.vl_acc_extract_contig_indels.argtypes = [
        ct.c_void_p, ct.c_int32, ct.c_int32,
        i32p, i16p, i16p, i32p, u64p, ct.c_char_p, i32p,
    ]
    lib.vl_acc_extract_contig_indels.restype = ct.c_int64

    lib.vl_acc_sort_contig_reads.argtypes = [ct.c_void_p, ct.c_int32]
    lib.vl_acc_sort_contig_reads.restype = None

    lib.vl_acc_sort_all_reads.argtypes = [ct.c_void_p]
    lib.vl_acc_sort_all_reads.restype = None

    lib.vl_acc_get_contig_reads_info.argtypes = [ct.c_void_p, ct.c_int32, i64p, i64p]
    lib.vl_acc_get_contig_reads_info.restype = None

    lib.vl_acc_copy_contig_reads.argtypes = [ct.c_void_p, ct.c_int32, i64p, u8p, u8p]
    lib.vl_acc_copy_contig_reads.restype = None

    lib.vl_acc_get_contig_reads_view.argtypes = [ct.c_void_p, ct.c_int32, i64p, u8p, u64p, i64p]
    lib.vl_acc_get_contig_reads_view.restype = None

    lib.vl_acc_test_set_seq_pool_offset.argtypes = [ct.c_void_p, ct.c_int32, ct.c_uint64]
    lib.vl_acc_test_set_seq_pool_offset.restype = None

    lib.vl_acc_free_contig.argtypes = [ct.c_void_p, ct.c_int32]
    lib.vl_acc_free_contig.restype = None

    lib.vl_acc_destroy.argtypes = [ct.c_void_p]
    lib.vl_acc_destroy.restype = None

    lib.vl_score_reads_haplotypes.argtypes = [
        u8p, i64p, i32p, ct.c_int32,
        ct.c_char_p, i32p, i32p, ct.c_int32, f64p,
    ]
    lib.vl_score_reads_haplotypes.restype = None

    lib.vl_acc_extract_callable_runs.argtypes = [
        ct.c_void_p, ct.c_int32, ct.c_int32, i32p, i32p, ct.c_int64,
    ]
    lib.vl_acc_extract_callable_runs.restype = ct.c_int64

    _LIB = lib
    return lib


def _ptr(arr: np.ndarray, ctype):
    return arr.ctypes.data_as(ct.POINTER(ctype))


@dataclass
class DiploidParams:
    min_mapq: int = 20
    min_bq: int = 13
    bin_mapq: int = 1
    zoom_min_mapq: int = 0
    max_mm_frac: float = 0.15
    mask_indel_flank: int = 0
    mask_clip_flank: int = 6
    mask_end_trim: int = 3
    mask_mm_win: int = 0
    mask_mm_n: int = 0
    dedup_bits: int = 20            # 1<<20 slots per shard (8 MB), 62-bit exact key
    n_shards: int = 8
    capture_informative: bool = True

    # Diploid Beta-Binomial SNV model (calibrated dispersion: kappa_err=300 prevents polynomial tail at 50% AB
    # while rejecting 10% paralog bleed at high depth)
    cand_min_alt: int = 2
    snv_min_alt: int = 2
    snv_err: float = 0.0025
    snv_hom_ref_err: float = 0.0035
    snv_het_ab: float = 0.485
    snv_kappa_err: float = 300.0
    snv_kappa_het: float = 60.0
    snv_kappa_hom: float = 250.0
    snv_prior_het: float = 1.0e-3
    snv_prior_hom: float = 5.0e-4
    snv_prior_12: float = 5.0e-6
    snv_min_qual: float = 20.0

    # Diploid Beta-Binomial Indel model (calibrated for PCR-free short reads & repeat-length polymorphism priors)
    indel_min_alt: int = 2
    indel_base_err: float = 0.0015
    indel_hp_err_quad: float = 0.00025
    indel_het_ab_max: float = 0.47
    indel_kappa_err: float = 140.0
    indel_kappa_het: float = 38.0
    indel_kappa_hom: float = 38.0
    indel_prior_het: float = 2.0e-4
    indel_prior_hom: float = 1.0e-4
    indel_prior_12: float = 1.5e-5
    indel_min_qual: float = 15.0

    # Local multi-k Simple Omnitig active-region reassembly
    enable_reassembly: bool = True
    active_merge_bp: int = 35
    active_flank_bp: int = 60
    active_min_clip: int = 4
    active_min_qual: float = 22.0

    # Calling model ploidy ('diploid' for 0/1 & 1/1 genotypes; 'haploid' for continuous WSAF mixtures)
    ploidy: str = "diploid"
    haploid_min_wsaf: float = 0.25


# ============================================================================
# 1. Target 31-mer Bait Table & Fast C Paired FASTQ Sieve
# ============================================================================

def build_target_kset(g: G.Genome, regions: Sequence[tuple[str, int, int]], bits: int = 27, dust_max: int = 64) -> tuple[np.ndarray, dict]:
    """Build canonical non-DUST 31-mer hash table (`1 << bits` uint64 slots) from 0-based `[s, e)` regions."""
    lib = _lib()
    cidx = {n: i for i, n in enumerate(g.names)}
    ivs = []
    for cname, s0, e0 in regions:
        if cname not in cidx:
            continue
        ci = cidx[cname]
        a = max(0, int(s0))
        b = min(int(g.lengths[ci]), int(e0))
        if b - a >= 31:
            ivs.append((int(g.offsets[ci]) + a, int(g.offsets[ci]) + b))
    ivs.sort()
    merged = []
    for a, b in ivs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    iv_arr = np.array(merged, dtype=np.int64) if merged else np.zeros((0, 2), dtype=np.int64)
    starts = np.ascontiguousarray(iv_arr[:, 0], dtype=np.int64)
    ends = np.ascontiguousarray(iv_arr[:, 1], dtype=np.int64)
    seq_u8 = np.ascontiguousarray(g.seq, dtype=np.uint8)
    tab = np.zeros(1 << bits, dtype=np.uint64)
    out_st = np.zeros(2, dtype=np.int64)
    t0 = time.time()
    rc = lib.vl_kset_build(
        _ptr(seq_u8, ct.c_uint8),
        _ptr(starts, ct.c_int64) if len(starts) else ct.POINTER(ct.c_int64)(),
        _ptr(ends, ct.c_int64) if len(ends) else ct.POINTER(ct.c_int64)(),
        len(starts),
        _ptr(tab, ct.c_uint64),
        ct.c_uint64(len(tab)),
        ct.c_int32(dust_max),
        _ptr(out_st, ct.c_int64),
    )
    if rc != 0:
        raise RuntimeError(f"vl_kset_build failed with code {rc}")
    meta = {
        "slots": int(len(tab)),
        "target_bp": int((ends - starts).sum()) if len(starts) else 0,
        "inserted_kmers": int(out_st[0]),
        "dust_masked": int(out_st[1]),
        "load_factor": float(out_st[0] / len(tab)),
        "build_s": round(time.time() - t0, 3),
    }
    return tab, meta


def _fastq_decomp_cmd(path: str | Path, threads: int = 4) -> list[str]:
    """Return decompression command for gzipped FASTQ or plain cat for uncompressed FASTQ."""
    is_gz = False
    try:
        with open(path, "rb") as fh:
            is_gz = fh.read(2) == b"\x1f\x8b"
    except OSError:
        is_gz = str(path).endswith(".gz")
    if not is_gz:
        return ["cat", str(path)]
    bgzip = shutil.which("bgzip") or "gzip"
    return [bgzip, "-dc", "-@", str(threads), str(path)] if "bgzip" in bgzip else ["gzip", "-dc", str(path)]


def extract_sieved_fastq_pair(
    r1_gz: str | Path,
    r2_gz: str | Path,
    out_r1_gz: str | Path,
    out_r2_gz: str | Path,
    kset_tab: np.ndarray | None = None,
    stride: int = 5,
    need: int = 1,
    p_sample: float = 1.0,
    decomp_threads: int = 8,
    comp_threads: int = 8,
) -> dict:
    """Stream paired FASTQ(.gz) through C lockstep verifier and 31-mer target filter into bgzf R1/R2."""
    lib = _lib()
    bgzip = shutil.which("bgzip") or "gzip"
    dec_cmd1 = _fastq_decomp_cmd(r1_gz, decomp_threads)
    dec_cmd2 = _fastq_decomp_cmd(r2_gz, decomp_threads)
    enc_cmd1 = [bgzip, "-c", "-l", "3", "-@", str(comp_threads)] if "bgzip" in bgzip else ["gzip", "-1", "-c"]
    enc_cmd2 = [bgzip, "-c", "-l", "3", "-@", str(comp_threads)] if "bgzip" in bgzip else ["gzip", "-1", "-c"]

    Path(out_r1_gz).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_r1_gz, "wb") as f1, open(out_r2_gz, "wb") as f2:
        p_in1 = subprocess.Popen(dec_cmd1, stdout=subprocess.PIPE)
        p_in2 = subprocess.Popen(dec_cmd2, stdout=subprocess.PIPE)
        p_out1 = subprocess.Popen(enc_cmd1, stdin=subprocess.PIPE, stdout=f1)
        p_out2 = subprocess.Popen(enc_cmd2, stdin=subprocess.PIPE, stdout=f2)
        try:
            stats = np.zeros(8, dtype=np.int64)
            keep_thr = min(int(p_sample * (1 << 32)), 1 << 32)
            tab_ptr = _ptr(kset_tab, ct.c_uint64) if kset_tab is not None else ct.POINTER(ct.c_uint64)()
            tab_len = len(kset_tab) if kset_tab is not None else 0
            rc = lib.vl_fastq_sieve_paired(
                p_in1.stdout.fileno(),
                p_in2.stdout.fileno(),
                -1,
                p_out1.stdin.fileno(),
                p_out2.stdin.fileno(),
                tab_ptr,
                ct.c_uint64(tab_len),
                ct.c_int32(stride),
                ct.c_int32(need if kset_tab is not None else 0),
                ct.c_uint64(keep_thr),
                ct.c_int32(1),
                ct.c_int32(30),
                ct.c_int64(0),
                _ptr(stats, ct.c_int64),
            )
        finally:
            for p_out in (p_out1, p_out2):
                if p_out.stdin:
                    p_out.stdin.close()
            rc_o1 = p_out1.wait()
            rc_o2 = p_out2.wait()
            if p_in1.stdout:
                p_in1.stdout.close()
            if p_in2.stdout:
                p_in2.stdout.close()
            rc_i1 = p_in1.wait()
            rc_i2 = p_in2.wait()
        if rc != 0 or rc_o1 != 0 or rc_o2 != 0 or rc_i1 != 0 or rc_i2 != 0:
            raise RuntimeError(f"FASTQ sieve failed: rc={rc}, in=({rc_i1},{rc_i2}), out=({rc_o1},{rc_o2})")

    return {
        "spots_seen": int(stats[0]),
        "spots_subsampled": int(stats[1]),
        "spots_emitted": int(stats[2]),
        "bases_seen": int(stats[3]),
        "bases_emitted": int(stats[4]),
        "adapter_trimmed": int(stats[7]),
        "out_r1_bytes": Path(out_r1_gz).stat().st_size,
        "out_r2_bytes": Path(out_r2_gz).stat().st_size,
        "wall_s": round(time.time() - t0, 3),
    }


# ============================================================================
# 2. PagedContigAccumulator Wrapper
# ============================================================================

class PagedContigAccumulator:
    """Memory-bounded contig-sharded + 64-kb lazy-paged pileup accumulator."""

    def __init__(
        self,
        g: G.Genome,
        p: DiploidParams | None = None,
        target_regions: Sequence[tuple[str, int, int]] | None = None,
        zoom_regions: Sequence[tuple[str, int, int]] | None = None,
    ):
        self.g = g
        self.p = p or DiploidParams()
        self._lib = _lib()

        names_b = [n.encode("ascii") for n in g.names]
        names_flat = b"".join(names_b)
        name_offs = np.concatenate([[0], np.cumsum([len(x) for x in names_b])]).astype(np.int64)
        lengths = np.ascontiguousarray(g.lengths, dtype=np.int64)
        ref_offs = np.ascontiguousarray(g.offsets, dtype=np.int64)
        bin_offs = np.ascontiguousarray(g.bin_offsets, dtype=np.int64)
        ref_seq = np.ascontiguousarray(g.seq, dtype=np.uint8)
        # Keep underlying arrays alive for borrowed C pointers
        self._keepalive = (names_flat, name_offs, lengths, ref_offs, bin_offs, ref_seq)

        self._handle = self._lib.vl_acc_create(
            ct.c_int32(len(g.names)),
            ct.c_char_p(names_flat),
            _ptr(name_offs, ct.c_int64),
            _ptr(lengths, ct.c_int64),
            _ptr(ref_offs, ct.c_int64),
            _ptr(bin_offs, ct.c_int64),
            _ptr(ref_seq, ct.c_uint8),
            ct.c_int32(max(1, min(self.p.n_shards, len(g.names)))),
            ct.c_int32(self.p.min_mapq),
            ct.c_int32(self.p.min_bq),
            ct.c_int32(self.p.bin_mapq),
            ct.c_int32(self.p.zoom_min_mapq),
            ct.c_double(self.p.max_mm_frac),
            ct.c_int32(self.p.mask_indel_flank),
            ct.c_int32(self.p.mask_clip_flank),
            ct.c_int32(self.p.mask_end_trim),
            ct.c_int32(self.p.mask_mm_win),
            ct.c_int32(self.p.mask_mm_n),
            ct.c_int32(1 if self.p.capture_informative else 0),
            ct.c_int32(self.p.dedup_bits),
        )
        if not self._handle:
            raise MemoryError("vl_acc_create failed")

        cidx = {n: i for i, n in enumerate(g.names)}
        self.eff_lengths = g.lengths.copy().astype(np.int64)
        self._target_bin_mask: np.ndarray | None = None
        if target_regions is not None:
            c_arr, s_arr, e_arr = [], [], []
            by_tc: dict[int, list[tuple[int, int]]] = {}
            for cn, s0, e0 in target_regions:
                if cn in cidx:
                    ci = cidx[cn]
                    a0 = max(0, int(s0))
                    b0 = min(int(g.lengths[ci]), int(e0))
                    if b0 > a0:
                        c_arr.append(ci)
                        s_arr.append(a0)
                        e_arr.append(b0)
                        by_tc.setdefault(ci, []).append((a0, b0))
            eff = np.zeros(len(g.names), dtype=np.int64)
            tmask = np.zeros(int(g.bin_offsets[-1]), dtype=bool)
            for ci, ivs in by_tc.items():
                ivs.sort()
                m_ivs: list[list[int]] = []
                for a0, b0 in ivs:
                    if m_ivs and a0 <= m_ivs[-1][1]:
                        m_ivs[-1][1] = max(m_ivs[-1][1], b0)
                    else:
                        m_ivs.append([a0, b0])
                eff[ci] = sum(b0 - a0 for a0, b0 in m_ivs)
                bo = int(g.bin_offsets[ci])
                nb_c = int(g.bin_offsets[ci + 1] - g.bin_offsets[ci])
                for a0, b0 in m_ivs:
                    b_lo = max(0, a0 // 1000)
                    b_hi = min(nb_c, (b0 + 999) // 1000)
                    tmask[bo + b_lo: bo + b_hi] = True
            if eff.sum() > 0:
                self.eff_lengths = eff
                self._target_bin_mask = tmask
            ca = np.array(c_arr, dtype=np.int32)
            sa = np.array(s_arr, dtype=np.int32)
            ea = np.array(e_arr, dtype=np.int32)
            self._lib.vl_acc_restrict_regions(
                self._handle,
                _ptr(ca, ct.c_int32) if len(ca) else ct.POINTER(ct.c_int32)(),
                _ptr(sa, ct.c_int32) if len(sa) else ct.POINTER(ct.c_int32)(),
                _ptr(ea, ct.c_int32) if len(ea) else ct.POINTER(ct.c_int32)(),
                ct.c_int32(len(ca)),
            )

        self.zoom_by_ctg: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if zoom_regions is not None:
            by_c: dict[int, list[tuple[int, int]]] = {}
            for cn, s0, e0 in zoom_regions:
                if cn in cidx:
                    by_c.setdefault(cidx[cn], []).append((int(s0), int(e0)))
            for cid, ivs in by_c.items():
                ivs.sort()
                merged = []
                for a, b in ivs:
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                sa = np.array([x[0] for x in merged], dtype=np.int32)
                ea = np.array([x[1] for x in merged], dtype=np.int32)
                self.zoom_by_ctg[cid] = (sa, ea)
                self._lib.vl_acc_set_zoom_intervals(
                    self._handle, ct.c_int32(cid), _ptr(sa, ct.c_int32), _ptr(ea, ct.c_int32), ct.c_int32(len(sa))
                )

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.vl_acc_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def consume_sam_fd(self, fd: int) -> None:
        rc = self._lib.vl_acc_consume_sam_fd(self._handle, ct.c_int(fd))
        if rc != 0:
            raise RuntimeError(f"vl_acc_consume_sam_fd failed with code {rc}")

    def feed_bytes(self, data: bytes | np.ndarray) -> int:
        arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else np.ascontiguousarray(data, dtype=np.uint8)
        if len(arr) == 0:
            return 0
        return int(self._lib.vl_acc_feed_sam_bytes(self._handle, _ptr(arr, ct.c_uint8), ct.c_size_t(len(arr))))

    def summary(self) -> dict:
        stats = np.zeros(12, dtype=np.int64)
        ccount = np.zeros(len(self.g.names), dtype=np.int64)
        binbases = np.zeros(int(self.g.bin_offsets[-1]), dtype=np.int64)
        mem = np.zeros(3, dtype=np.int64)
        self._lib.vl_acc_get_summary(
            self._handle,
            _ptr(stats, ct.c_int64),
            _ptr(ccount, ct.c_int64),
            _ptr(binbases, ct.c_int64),
            _ptr(mem, ct.c_int64),
        )
        bb_for_order = binbases[self._target_bin_mask] if self._target_bin_mask is not None else binbases
        return {
            "stats": stats,
            "ccount": ccount,
            "binbases": binbases,
            "order_mode": E.detect_order_mode(ccount, self.eff_lengths, bb_for_order),
            "page_bytes": int(mem[0]),
            "read_buf_bytes": int(mem[1]),
            "indel_tab_bytes": int(mem[2]),
            "total_acc_bytes": int(mem.sum()),
        }

    def contig_candidates(self, ctg_id: int, min_alt: int = 2, min_clip: int = 4) -> dict[str, np.ndarray]:
        n = int(self._lib.vl_acc_count_contig_candidates(
            self._handle, ct.c_int32(ctg_id), ct.c_int32(min_alt), ct.c_int32(min_clip)
        ))
        pos0 = np.empty(n, dtype=np.int32)
        counts = np.empty((n, 4), dtype=np.uint16)
        delc = np.empty(n, dtype=np.uint16)
        clipc = np.empty(n, dtype=np.uint16)
        uncap = np.empty(n, dtype=np.uint16)
        ref = np.empty(n, dtype=np.uint8)
        if n > 0:
            self._lib.vl_acc_extract_contig_candidates(
                self._handle, ct.c_int32(ctg_id), ct.c_int32(min_alt), ct.c_int32(min_clip),
                _ptr(pos0, ct.c_int32), _ptr(counts, ct.c_uint16), _ptr(delc, ct.c_uint16),
                _ptr(clipc, ct.c_uint16), _ptr(uncap, ct.c_uint16), _ptr(ref, ct.c_uint8),
            )
        return dict(pos0=pos0, counts=counts, delc=delc, clipc=clipc, uncap=uncap, ref=ref)

    def query_positions(self, ctg_id: int, pos0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos0 = np.ascontiguousarray(pos0, dtype=np.int32)
        n = len(pos0)
        dp = np.zeros(n, dtype=np.int32)
        delc = np.zeros(n, dtype=np.int32)
        uncap = np.zeros(n, dtype=np.int32)
        if n > 0:
            self._lib.vl_acc_query_positions(
                self._handle, ct.c_int32(ctg_id), _ptr(pos0, ct.c_int32), ct.c_int64(n),
                _ptr(dp, ct.c_int32), _ptr(delc, ct.c_int32), _ptr(uncap, ct.c_int32),
            )
        return dp, delc, uncap

    def contig_indels(self, ctg_id: int, min_cnt: int = 2) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        lb = ct.c_int64(0)
        n = int(self._lib.vl_acc_count_contig_indels(
            self._handle, ct.c_int32(ctg_id), ct.c_int32(min_cnt), ct.byref(lb)
        ))
        if n == 0:
            return np.zeros((0, 4), dtype=np.int64), np.zeros(0, dtype=np.int64), {}
        pos0 = np.empty(n, dtype=np.int32)
        kind = np.empty(n, dtype=np.int16)
        alen = np.empty(n, dtype=np.int16)
        cnt = np.empty(n, dtype=np.int32)
        code = np.empty(n, dtype=np.uint64)
        long_offs = np.empty(n, dtype=np.int32)
        long_buf = ct.create_string_buffer(max(1, int(lb.value)))
        self._lib.vl_acc_extract_contig_indels(
            self._handle, ct.c_int32(ctg_id), ct.c_int32(min_cnt),
            _ptr(pos0, ct.c_int32), _ptr(kind, ct.c_int16), _ptr(alen, ct.c_int16),
            _ptr(cnt, ct.c_int32), _ptr(code, ct.c_uint64), long_buf, _ptr(long_offs, ct.c_int32),
        )
        raw_long = long_buf.raw
        ins_seqs: dict[int, str] = {}
        code_i64 = np.empty(n, dtype=np.int64)
        for i in range(n):
            if int(kind[i]) == 1 or int(alen[i]) <= 31:
                code_i64[i] = int(code[i] & np.uint64(0x7FFFFFFFFFFFFFFF))
            else:
                lo = int(long_offs[i])
                s = raw_long[lo:lo + int(alen[i])].decode("ascii", errors="replace")
                ic = C._fnv63(s)
                ins_seqs[ic] = s
                code_i64[i] = ic
        goff = self.g.offsets[ctg_id] + pos0.astype(np.int64)
        keys = np.stack([goff, kind.astype(np.int64), alen.astype(np.int64), code_i64], axis=1)
        return keys, cnt.astype(np.int64), ins_seqs

    def sort_all_reads(self) -> None:
        self._lib.vl_acc_sort_all_reads(self._handle)

    def test_set_seq_pool_offset(self, ctg_id: int, offset_bytes: int) -> None:
        self._lib.vl_acc_test_set_seq_pool_offset(self._handle, ct.c_int32(ctg_id), ct.c_uint64(offset_bytes))

    def contig_reads(self, ctg_id: int, zero_copy: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._lib.vl_acc_sort_contig_reads(self._handle, ct.c_int32(ctg_id))
        nr = ct.c_int64(0)
        pb = ct.c_int64(0)
        self._lib.vl_acc_get_contig_reads_info(self._handle, ct.c_int32(ctg_id), ct.byref(nr), ct.byref(pb))
        n_reads = int(nr.value)
        pool_bytes = int(pb.value)
        if n_reads == 0:
            return np.zeros((0, 6), dtype=np.int64), np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)
        meta6 = np.empty((n_reads, 6), dtype=np.int64)
        mapq = np.empty(n_reads, dtype=np.uint8)
        if zero_copy and pool_bytes > 0:
            p_addr = ct.c_uint64(0)
            p_len = ct.c_int64(0)
            self._lib.vl_acc_get_contig_reads_view(
                self._handle, ct.c_int32(ctg_id),
                _ptr(meta6, ct.c_int64), _ptr(mapq, ct.c_uint8),
                ct.byref(p_addr), ct.byref(p_len),
            )
            c_buf = (ct.c_uint8 * int(p_len.value)).from_address(int(p_addr.value))
            pool = np.ctypeslib.as_array(c_buf)
            return meta6, mapq, pool
        pool = np.empty(pool_bytes, dtype=np.uint8)
        self._lib.vl_acc_copy_contig_reads(
            self._handle, ct.c_int32(ctg_id),
            _ptr(meta6, ct.c_int64), _ptr(mapq, ct.c_uint8), _ptr(pool, ct.c_uint8),
        )
        return meta6, mapq, pool

    def callable_runs(self, ctg_id: int, min_dp: int = 1) -> tuple[np.ndarray, np.ndarray]:
        n = int(self._lib.vl_acc_extract_callable_runs(
            self._handle, ct.c_int32(ctg_id), ct.c_int32(min_dp),
            ct.POINTER(ct.c_int32)(), ct.POINTER(ct.c_int32)(), ct.c_int64(0),
        ))
        if n <= 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        s0 = np.empty(n, dtype=np.int32)
        e0 = np.empty(n, dtype=np.int32)
        self._lib.vl_acc_extract_callable_runs(
            self._handle, ct.c_int32(ctg_id), ct.c_int32(min_dp),
            _ptr(s0, ct.c_int32), _ptr(e0, ct.c_int32), ct.c_int64(n),
        )
        return s0, e0

    def save_callable_npz(self, out_npz: str | Path, min_dp: int = 1) -> None:
        arrays: dict[str, np.ndarray] = {}
        for cid, cname in enumerate(self.g.names):
            s0, e0 = self.callable_runs(cid, min_dp=min_dp)
            if len(s0) > 0:
                arrays[f"s_{cname}"] = s0
                arrays[f"e_{cname}"] = e0
        np.savez_compressed(Path(out_npz), **arrays)

    def free_contig(self, ctg_id: int) -> None:
        self._lib.vl_acc_free_contig(self._handle, ct.c_int32(ctg_id))


# ============================================================================
# 3. Calibrated Beta-Binomial Diploid Genotype Caller
# ============================================================================

def beta_binom_logpmf(k: np.ndarray, n: np.ndarray, mu: float | np.ndarray, kappa: float | np.ndarray) -> np.ndarray:
    """Vectorized log Beta-Binomial PMF `log P(k | n, alpha=mu*kappa, beta=(1-mu)*kappa)`."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    kappa = np.asarray(kappa, dtype=np.float64)
    a = mu * kappa
    b = (1.0 - mu) * kappa
    return (
        gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(np.maximum(n - k, 0.0) + 1.0)
        + gammaln(k + a) + gammaln(np.maximum(n - k, 0.0) + b) - gammaln(n + a + b)
        + gammaln(a + b) - gammaln(a) - gammaln(b)
    )


def _wsaf_llr_qual(ac: np.ndarray | float, other: np.ndarray | float, err: np.ndarray | float) -> np.ndarray:
    """Phred-scaled binomial likelihood ratio for continuous allele fraction vs sequencing error `err`."""
    ac_a = np.asarray(ac, dtype=np.float64)
    ot_a = np.maximum(np.asarray(other, dtype=np.float64), 0.0)
    tot = ac_a + ot_a
    err_a = np.clip(np.asarray(err, dtype=np.float64), 1e-6, 0.49)
    safe_tot = np.maximum(tot, 1.0)
    p_hat = np.clip(ac_a / safe_tot, err_a, 1.0 - 1e-12)
    ll_alt = ac_a * np.log(p_hat) + ot_a * np.log(np.maximum(1.0 - p_hat, 1e-12))
    ll_err = ac_a * np.log(err_a) + ot_a * np.log(1.0 - err_a)
    llr = np.where((ac_a > 0.0) & (p_hat > err_a), np.maximum(0.0, ll_alt - ll_err), 0.0)
    return np.minimum(999.0, 10.0 * llr / LN10)


def call_diploid_snvs_contig(g: G.Genome, ctg_id: int, cand: dict[str, np.ndarray], p: DiploidParams) -> list[dict]:
    """Call SNVs on contig `ctg_id` under diploid Beta-Binomial (`0/1`, `1/1`, `1/2`) or haploid continuous WSAF (`1`)."""
    pos0 = cand["pos0"]
    if len(pos0) == 0:
        return []
    counts = cand["counts"].astype(np.int32)
    delc = cand["delc"].astype(np.int32)
    ref = cand["ref"].astype(np.int32)

    tot = counts.sum(axis=1)
    idx_n = np.arange(len(pos0))
    rc = counts[idx_n, ref]
    nonref = tot - rc
    keep = nonref >= p.snv_min_alt
    if not keep.any():
        return []

    pos0 = pos0[keep]
    counts = counts[keep]
    delc = delc[keep]
    ref = ref[keep]
    tot = tot[keep]
    rc = rc[keep]
    m = len(pos0)
    idx_m = np.arange(m)

    c_masked = counts.copy()
    c_masked[idx_m, ref] = -1
    order = np.argsort(-c_masked, axis=1)
    a1 = order[:, 0]
    a2 = order[:, 1]
    ac1 = np.maximum(counts[idx_m, a1], 0)
    ac2 = np.maximum(counts[idx_m, a2], 0)

    dp_full = tot + delc
    # Inside a homozygous 1/1 spanning deletion (delc >= 80% of total depth and delc >= 4), suppress residual SNV noise
    hom_del = (delc >= 4) & (delc >= 0.78 * np.maximum(dp_full, 1))

    if str(p.ploidy).strip().lower() == "haploid":
        eff_min_wsaf = max(0.05, min(0.99, float(p.haploid_min_wsaf)))
        out_h: list[dict] = []
        ctg_name = g.names[ctg_id]
        bases_str = "ACGT"
        err_snv_h = max(float(p.snv_err), 0.005)
        for rank_k in range(3):
            ak = order[:, rank_k]
            ack = np.maximum(counts[idx_m, ak], 0)
            wsafk = ack / np.maximum(tot, 1).astype(np.float64)
            qualk = _wsaf_llr_qual(ack, tot - ack, err_snv_h)
            for i in range(m):
                if hom_del[i] or int(ak[i]) == int(ref[i]):
                    continue
                if ack[i] < p.snv_min_alt or wsafk[i] < eff_min_wsaf or qualk[i] < p.snv_min_qual:
                    continue
                q_i = float(qualk[i])
                out_h.append(dict(
                    ctg_id=ctg_id,
                    chrom=ctg_name,
                    pos0=int(pos0[i]),
                    pos=int(pos0[i]) + 1,
                    ref=bases_str[int(ref[i])],
                    alt=bases_str[int(ak[i])],
                    gt="1",
                    qual=round(q_i, 2),
                    gq=min(99, max(0, int(round(q_i)))),
                    dp=int(dp_full[i]),
                    ad=(int(rc[i]), int(ack[i])),
                    wsaf=(round(float(wsafk[i]), 4),),
                    source="pileup_snv_wsaf",
                ))
        return out_h

    # Opposite a heterozygous 0/1 spanning deletion (delc in [22%, 78%] of dp_full and delc >= 3),
    # the remaining homologous chromosome is effectively haploid (0/* vs 1/* -> VCF GT 0/1, never 1/1!)
    het_span_del = (~hom_del) & (delc >= 3) & (delc >= 0.22 * np.maximum(dp_full, 1))

    n01 = rc + ac1
    n02 = rc + ac2
    n12 = ac1 + ac2

    # Log-likelihoods for the 6 diploid genotypes: 0:0/0, 1:0/1, 2:1/1, 3:0/2, 4:2/2, 5:1/2
    ll = np.full((m, 6), -1e9, dtype=np.float64)
    ll[:, 0] = beta_binom_logpmf(tot - rc, tot, p.snv_err, p.snv_kappa_err)

    # 0/1: ac1 vs rc + error model on any third allele ac2
    ll[:, 1] = (
        beta_binom_logpmf(ac1, n01, p.snv_het_ab, p.snv_kappa_het)
        + beta_binom_logpmf(ac2, tot, p.snv_err * 0.5, p.snv_kappa_err)
    )
    # 1/1: non-alt1 (rc + ac2) is sequencing error
    ll[:, 2] = beta_binom_logpmf(tot - ac1, tot, p.snv_hom_ref_err, p.snv_kappa_hom)

    has_a2 = ac2 >= p.snv_min_alt
    if has_a2.any():
        ll[has_a2, 3] = (
            beta_binom_logpmf(ac2[has_a2], n02[has_a2], p.snv_het_ab, p.snv_kappa_het)
            + beta_binom_logpmf(ac1[has_a2], tot[has_a2], p.snv_err * 0.5, p.snv_kappa_err)
        )
        ll[has_a2, 4] = beta_binom_logpmf(tot[has_a2] - ac2[has_a2], tot[has_a2], p.snv_hom_ref_err, p.snv_kappa_hom)
        ll[has_a2, 5] = (
            beta_binom_logpmf(ac1[has_a2], n12[has_a2], 0.50, p.snv_kappa_het)
            + beta_binom_logpmf(rc[has_a2], tot[has_a2], p.snv_hom_ref_err, p.snv_kappa_hom)
        )

    # Under het_span_del (one copy deleted '*'), the single intact copy is either REF (0/*) or ALT1 (1/* -> VCF 0/1)
    if het_span_del.any():
        hs = het_span_del
        ll[hs, 1] = beta_binom_logpmf(tot[hs] - ac1[hs], tot[hs], p.snv_hom_ref_err, p.snv_kappa_hom)
        ll[hs, 2] = -1e9  # 1/1 impossible alongside 0/1 spanning deletion
        ll[hs, 4] = -1e9
        ll[hs, 5] = -1e9

    priors = np.array([
        math.log(1.0 - p.snv_prior_het - p.snv_prior_hom - p.snv_prior_12),
        math.log(p.snv_prior_het),
        math.log(p.snv_prior_hom),
        math.log(p.snv_prior_het * 0.5),
        math.log(p.snv_prior_hom * 0.5),
        math.log(p.snv_prior_12),
    ], dtype=np.float64)

    post = ll + priors[None, :]
    log_z = logsumexp(post, axis=1)
    # QUAL = -10 * log10(P(0/0 | data))
    log_p00 = post[:, 0] - log_z
    qual = np.minimum(999.0, np.maximum(0.0, -10.0 * log_p00 / LN10))

    best_g = np.argmax(post, axis=1)
    part2 = np.partition(post, -2, axis=1)
    gq = np.minimum(99, np.maximum(0, np.rint(10.0 * (part2[:, -1] - part2[:, -2]) / LN10))).astype(np.int32)

    out: list[dict] = []
    ctg_name = g.names[ctg_id]
    bases_str = "ACGT"
    for i in range(m):
        if hom_del[i] or best_g[i] == 0 or qual[i] < p.snv_min_qual:
            continue
        bg = int(best_g[i])
        r_ch = bases_str[int(ref[i])]
        if bg in (1, 2):
            if ac1[i] < p.snv_min_alt:
                continue
            gt_str = "0/1" if bg == 1 else "1/1"
            alt_str = bases_str[int(a1[i])]
            ad_tuple = (int(rc[i]), int(ac1[i]))
        elif bg in (3, 4):
            if ac2[i] < p.snv_min_alt:
                continue
            gt_str = "0/1" if bg == 3 else "1/1"
            alt_str = bases_str[int(a2[i])]
            ad_tuple = (int(rc[i]), int(ac2[i]))
        else:  # bg == 5 -> 1/2 multiallelic
            if ac1[i] < p.snv_min_alt or ac2[i] < p.snv_min_alt:
                continue
            gt_str = "1/2"
            alt_str = f"{bases_str[int(a1[i])]},{bases_str[int(a2[i])]}"
            ad_tuple = (int(rc[i]), int(ac1[i]), int(ac2[i]))
        out.append(dict(
            ctg_id=ctg_id,
            chrom=ctg_name,
            pos0=int(pos0[i]),
            pos=int(pos0[i]) + 1,
            ref=r_ch,
            alt=alt_str,
            gt=gt_str,
            qual=round(float(qual[i]), 2),
            gq=int(gq[i]),
            dp=int(dp_full[i]),
            ad=ad_tuple,
            source="pileup_snv",
        ))
    return out


def call_diploid_indels_contig(
    g: G.Genome,
    acc: PagedContigAccumulator,
    ctg_id: int,
    p: DiploidParams,
) -> tuple[list[dict], list[tuple[int, str, str, int]]]:
    """Call left-normalized diploid indels (`0/1`, `1/1`, `1/2`) and return raw candidates for local assembly."""
    keys, cnt, ins_seqs = acc.contig_indels(ctg_id, min_cnt=p.cand_min_alt)
    if len(keys) == 0:
        return [], []

    norm_keys, norm_cnt = C.normalize_indels(g, keys, cnt, ins_seqs=ins_seqs)
    keep = norm_cnt >= p.cand_min_alt
    if not keep.any():
        return [], []
    norm_keys = norm_keys[keep]
    norm_cnt = norm_cnt[keep]

    cstart = int(g.offsets[ctg_id])
    goff = norm_keys[:, 0]
    kind = norm_keys[:, 1]
    alen = norm_keys[:, 2]
    code = norm_keys[:, 3]

    # Anchor position (0-based on contig) and depth query position
    dpos0 = (goff - cstart).astype(np.int32)
    anchor0 = np.where(kind == 1, np.maximum(dpos0 - 1, 0), dpos0).astype(np.int32)
    dp_q, _, _ = acc.query_positions(ctg_id, dpos0)
    anchor_goff = cstart + anchor0.astype(np.int64)
    hp = C.homopolymer_len(g.seq, anchor_goff)

    # Materialize VCF alleles (anchor0, ref_str, alt_str) and group multiallelic indels at the same anchor0
    raw_cands: list[tuple[int, str, str, int]] = []
    by_anchor: dict[int, list[tuple[int, str, str, int, int, int, int]]] = {}
    for i in range(len(goff)):
        a0 = int(anchor0[i])
        ga = cstart + a0
        L = int(alen[i])
        if int(kind[i]) == 1:
            r_s = g.bases(ga, ga + L + 1)
            a_s = g.bases(ga, ga + 1)
        else:
            ins_s = C._decode_ins(int(code[i]), L, ins_seqs=ins_seqs)
            if ins_s is None or "N" in ins_s:
                continue
            r_s = g.bases(ga, ga + 1)
            a_s = r_s + ins_s
        if "N" in r_s or "N" in a_s:
            continue
        c_i = int(norm_cnt[i])
        dp_i = max(int(dp_q[i]), c_i)
        raw_cands.append((a0, r_s, a_s, c_i))
        by_anchor.setdefault(a0, []).append((c_i, r_s, a_s, int(kind[i]), L, dp_i, int(hp[i])))

    ctg_name = g.names[ctg_id]
    calls: list[dict] = []
    for a0, items in sorted(by_anchor.items()):
        items.sort(key=lambda x: -x[0])
        c1, r1, alt1, k1, L1, dp1, hp1 = items[0]
        min_alt_req = max(p.indel_min_alt, 3 if hp1 >= 8 else (4 if hp1 >= 13 else p.indel_min_alt))
        if hp1 >= 13:
            min_alt_req = max(min_alt_req, 4)
        if c1 < min_alt_req:
            continue
        c2 = items[1][0] if len(items) > 1 else 0
        dp_tot = max(dp1, c1 + c2)
        rc = max(dp_tot - c1 - c2, 0)

        # Length & repeat-calibrated het allele balance (insertions + repeat edges induce soft-clipped ALT reads)
        eff_ins = L1 if k1 == 2 else int(0.35 * L1)
        hp_clip_bias = 0.0045 * max(0, min(hp1, 18) - 5)
        mu_het = float(np.clip(
            (p.indel_het_ab_max - hp_clip_bias) * max(150.0 - 12.0 - eff_ins, 25.0) / max(138.0 - 0.5 * eff_ins, 30.0),
            0.24, p.indel_het_ab_max,
        ))
        # Hom-alt anchor overhang + homopolymer slippage rate: reads ending inside a repeat align as un-gapped M at a0
        err_0 = float(min(0.065, p.indel_base_err + p.indel_hp_err_quad * max(0, hp1 - 4) ** 2))
        if str(p.ploidy).strip().lower() == "haploid":
            eff_indel_wsaf = max(0.15, min(0.99, float(p.haploid_min_wsaf)))
            sum_alt_all = sum(int(x[0]) for x in items)
            dp_all = max(max(int(x[5]) for x in items), sum_alt_all)
            rc_all_ind = max(dp_all - sum_alt_all, 0)
            for c_k, r_k, alt_k, _k_k, _L_k, _dp_k, hp_k in items:
                req_k = max(p.indel_min_alt, 4 if hp_k >= 13 else (3 if hp_k >= 8 else p.indel_min_alt))
                if c_k < req_k:
                    continue
                err_k = float(min(0.065, p.indel_base_err + p.indel_hp_err_quad * max(0, hp_k - 4) ** 2))
                wsaf_k = float(c_k) / float(max(dp_all, 1))
                qual_k = float(_wsaf_llr_qual(c_k, max(dp_all - c_k, 0), max(err_k, 0.015)))
                if wsaf_k >= eff_indel_wsaf and qual_k >= p.indel_min_qual:
                    calls.append(dict(
                        ctg_id=ctg_id, chrom=ctg_name, pos0=a0, pos=a0 + 1,
                        ref=r_k, alt=alt_k, gt="1",
                        qual=round(qual_k, 2), gq=min(99, max(0, int(round(qual_k)))),
                        dp=dp_all, ad=(rc_all_ind, c_k), wsaf=(round(wsaf_k, 4),),
                        source="pileup_indel_wsaf",
                    ))
            continue

        err_hom = float(min(0.24, 0.018 + 0.0040 * min(L1, 40) + 0.0085 * max(0, min(hp1, 20) - 3) + 0.5 * err_0))

        # Germline indel mutation rate increases strongly in homopolymer/STR tracts
        hp_prior_mult = 1.0 + 0.35 * (max(0, min(hp1, 20) - 4) ** 1.5)
        pr_het = min(6.0e-3, p.indel_prior_het * hp_prior_mult)
        pr_hom = min(3.0e-3, p.indel_prior_hom * hp_prior_mult)
        pr_12 = min(5.0e-4, p.indel_prior_12 * hp_prior_mult)

        n01 = max(rc + c1, 1)
        ll0 = float(beta_binom_logpmf(c1 + c2, dp_tot, err_0, p.indel_kappa_err))
        ll1 = float(
            beta_binom_logpmf(c1, n01, mu_het, p.indel_kappa_het)
            + beta_binom_logpmf(c2, dp_tot, err_0 * 0.5, p.indel_kappa_err)
        )
        ll2 = float(beta_binom_logpmf(dp_tot - c1, dp_tot, err_hom, p.indel_kappa_hom))

        ll12 = -1e9
        if len(items) > 1 and c2 >= p.indel_min_alt:
            ll12 = float(
                beta_binom_logpmf(c1, max(c1 + c2, 1), 0.50, p.indel_kappa_het)
                + beta_binom_logpmf(rc, dp_tot, err_hom, p.indel_kappa_hom)
            )

        posts = np.array([
            ll0 + math.log(max(1e-6, 1.0 - pr_het - pr_hom - pr_12)),
            ll1 + math.log(pr_het),
            ll2 + math.log(pr_hom),
            ll12 + math.log(pr_12),
        ], dtype=np.float64)
        lz = float(logsumexp(posts))
        qual = min(999.0, max(0.0, -10.0 * (posts[0] - lz) / LN10))
        bg = int(np.argmax(posts))
        if bg == 0 or qual < p.indel_min_qual:
            continue
        s_posts = np.sort(posts)
        gq = min(99, max(0, int(round(10.0 * (s_posts[-1] - s_posts[-2]) / LN10))))

        if bg in (1, 2):
            calls.append(dict(
                ctg_id=ctg_id, chrom=ctg_name, pos0=a0, pos=a0 + 1,
                ref=r1, alt=alt1, gt="0/1" if bg == 1 else "1/1",
                qual=round(qual, 2), gq=gq, dp=dp_tot, ad=(rc, c1),
                source="pileup_indel",
            ))
        else:
            _, r2, alt2, *_ = items[1]
            # Unify REF length for two overlapping indels at anchor a0
            if len(r1) >= len(r2):
                suf = r1[len(r2):]
                u_ref, u_a1, u_a2 = r1, alt1, alt2 + suf
            else:
                suf = r2[len(r1):]
                u_ref, u_a1, u_a2 = r2, alt1 + suf, alt2
            if u_a1 == u_a2:
                calls.append(dict(
                    ctg_id=ctg_id, chrom=ctg_name, pos0=a0, pos=a0 + 1,
                    ref=r1, alt=alt1, gt="1/1",
                    qual=round(qual, 2), gq=gq, dp=dp_tot, ad=(rc, c1 + c2),
                    source="pileup_indel",
                ))
            else:
                calls.append(dict(
                    ctg_id=ctg_id, chrom=ctg_name, pos0=a0, pos=a0 + 1,
                    ref=u_ref, alt=f"{u_a1},{u_a2}", gt="1/2",
                    qual=round(qual, 2), gq=gq, dp=dp_tot, ad=(rc, c1, c2),
                    source="pileup_indel",
                ))
    return calls, raw_cands


# ============================================================================
# 4. Multi-k Simple Omnitig + C Semi-Global Active-Region Diploid Reassembler
# ============================================================================

def score_reads_against_padded_haps(
    pool: np.ndarray,
    read_offs: np.ndarray,
    read_lens: np.ndarray,
    padded_haps: list[str],
) -> np.ndarray:
    """Call C `vl_score_reads_haplotypes` returning `(n_reads, n_haps)` log-likelihood matrix (64-bit read_offs)."""
    n_r = len(read_offs)
    n_h = len(padded_haps)
    if n_r == 0 or n_h == 0:
        return np.zeros((n_r, n_h), dtype=np.float64)
    lib = _lib()
    ro = np.ascontiguousarray(read_offs, dtype=np.int64)
    rl = np.ascontiguousarray(read_lens, dtype=np.int32)
    if (ro < 0).any() or (rl < 0).any():
        raise ValueError("Negative read offset or length in score_reads_against_padded_haps")
    h_bytes = [h.encode("ascii") for h in padded_haps]
    h_concat = b"".join(h_bytes)
    h_offs = np.array([0] + list(np.cumsum([len(x) for x in h_bytes[:-1]])), dtype=np.int32)
    h_lens = np.array([len(x) for x in h_bytes], dtype=np.int32)
    out_ll = np.empty((n_r, n_h), dtype=np.float64)
    lib.vl_score_reads_haplotypes(
        _ptr(pool, ct.c_uint8),
        _ptr(ro, ct.c_int64),
        _ptr(rl, ct.c_int32),
        ct.c_int32(n_r),
        ct.c_char_p(h_concat),
        _ptr(h_offs, ct.c_int32),
        _ptr(h_lens, ct.c_int32),
        ct.c_int32(n_h),
        _ptr(out_ll, ct.c_double),
    )
    return out_ll


def _format_diploid_pair_calls(
    ctg_id: int,
    ctg_name: str,
    ctg_seq_u8: np.ndarray,
    ref_win: str,
    win_lo: int,
    h1_seq: str,
    h2_seq: str,
    qual: float,
    gq: int,
    dp: int,
    ad_est: tuple[int, ...],
) -> list[dict]:
    """Convert best diploid haplotype pair `(h1_seq, h2_seq)` into clean non-overlapping VCF call dicts.

    Emits unphased `0/1` (never cross-window unanchored `|` without `PS`), uses real measured read support
    in `AD` (never fabricated `(0, dp)`), and tags internal `hap_side` (`1` vs `2`) for same-site `1/2` checks.
    """
    if h1_seq == ref_win and h2_seq == ref_win:
        return []
    sup_0 = int(ad_est[0]) if len(ad_est) >= 1 else 0
    if len(ad_est) >= 3:
        sup_h1, sup_h2 = int(ad_est[1]), int(ad_est[2])
    else:
        sup_h1 = 0
        sup_h2 = int(ad_est[1]) if len(ad_est) >= 2 else 0

    if h1_seq == ref_win or h1_seq == h2_seq:
        alt_seq = h2_seq
        gt_str = "0/1" if h1_seq == ref_win else "1/1"
        ad_pair = (sup_0, sup_h2)
        diffs = OZ.decompose_haplotype_diffs(ref_win, alt_seq, win_lo, ctg_seq_u8)
        return [
            dict(
                ctg_id=ctg_id, chrom=ctg_name, pos0=p0, pos=p0 + 1,
                ref=r_s, alt=a_s, gt=gt_str, qual=round(qual, 2), gq=gq,
                dp=max(dp, sup_0 + sup_h2), ad=ad_pair, hap_side=1 if gt_str == "0/1" else 0,
                source="omnitig_asm",
            )
            for p0, r_s, a_s in diffs
        ]

    # Compound heterozygote (h1 != ref and h2 != ref and h1 != h2)
    d1 = OZ.decompose_haplotype_diffs(ref_win, h1_seq, win_lo, ctg_seq_u8)
    d2 = OZ.decompose_haplotype_diffs(ref_win, h2_seq, win_lo, ctg_seq_u8)
    set1, set2 = set(d1), set(d2)
    shared_hom = set1 & set2
    only1 = [x for x in d1 if x not in shared_hom]
    only2 = [x for x in d2 if x not in shared_hom]

    # Check if any trans alleles overlap on the reference
    overlaps = any(
        max(a0, b0) < min(a0 + len(ra), b0 + len(rb))
        for a0, ra, _ in only1
        for b0, rb, _ in only2
    )
    dp_meas = max(dp, sup_0 + sup_h1 + sup_h2)
    if not overlaps:
        out = []
        for p0, r_s, a_s in shared_hom:
            out.append(dict(
                ctg_id=ctg_id, chrom=ctg_name, pos0=p0, pos=p0 + 1, ref=r_s, alt=a_s,
                gt="1/1", qual=round(qual, 2), gq=gq, dp=dp_meas,
                ad=(sup_0, sup_h1 + sup_h2), hap_side=0, source="omnitig_asm",
            ))
        for p0, r_s, a_s in only1:
            out.append(dict(
                ctg_id=ctg_id, chrom=ctg_name, pos0=p0, pos=p0 + 1, ref=r_s, alt=a_s,
                gt="0/1", qual=round(qual, 2), gq=gq, dp=dp_meas,
                ad=(sup_0 + sup_h2, sup_h1), hap_side=1, source="omnitig_asm",
            ))
        for p0, r_s, a_s in only2:
            out.append(dict(
                ctg_id=ctg_id, chrom=ctg_name, pos0=p0, pos=p0 + 1, ref=r_s, alt=a_s,
                gt="0/1", qual=round(qual, 2), gq=gq, dp=dp_meas,
                ad=(sup_0 + sup_h1, sup_h2), hap_side=2, source="omnitig_asm",
            ))
        out.sort(key=lambda x: x["pos0"])
        return out

    # Overlapping compound-het alleles: emit single unified 1/2 multiallelic block spanning the differing core.
    # Slice directly from `ctg_seq_u8` because left-aligned repeat indels can shift `lo_p < win_lo`!
    all_evs = d1 + d2
    if not all_evs:
        return []
    lo_p = max(0, min(x[0] for x in all_evs))
    hi_p = min(len(ctg_seq_u8), max(lo_p + 1, max(x[0] + len(x[1]) for x in all_evs)))

    def _ctg_slice_str(a0: int, b0: int) -> str:
        if b0 <= a0:
            return ""
        return "".join("ACGTN"[min(int(b), 4)] for b in ctg_seq_u8[a0:b0])

    u_ref = _ctg_slice_str(lo_p, hi_p)
    if not u_ref or "N" in u_ref:
        return []

    def _apply_local(evs):
        cur = lo_p
        parts = []
        for p0, rs, as_ in sorted(evs, key=lambda z: z[0]):
            if p0 < cur:
                continue
            parts.append(_ctg_slice_str(cur, p0))
            parts.append(as_)
            cur = p0 + len(rs)
        parts.append(_ctg_slice_str(cur, hi_p))
        return "".join(parts)

    ua1 = _apply_local(d1)
    ua2 = _apply_local(d2)
    if not ua1 or not ua2 or "N" in ua1 or "N" in ua2:
        return []
    if ua1 == ua2:
        if ua1 == u_ref:
            return []
        return [dict(ctg_id=ctg_id, chrom=ctg_name, pos0=lo_p, pos=lo_p + 1, ref=u_ref, alt=ua1,
                     gt="1/1", qual=round(qual, 2), gq=gq, dp=dp_meas,
                     ad=(sup_0, sup_h1 + sup_h2), hap_side=0, source="omnitig_asm")]
    return [dict(ctg_id=ctg_id, chrom=ctg_name, pos0=lo_p, pos=lo_p + 1, ref=u_ref, alt=f"{ua1},{ua2}",
                 gt="1/2", qual=round(qual, 2), gq=gq, dp=dp_meas,
                 ad=(sup_0, sup_h1, sup_h2), hap_side=0, source="omnitig_asm")]


def reassemble_active_regions_contig(
    g: G.Genome,
    acc: PagedContigAccumulator,
    ctg_id: int,
    cand: dict[str, np.ndarray],
    pileup_calls: list[dict],
    raw_indels: list[tuple[int, str, str, int]],
    p: DiploidParams,
) -> list[dict]:
    """Run multi-k Simple Omnitig + C semi-global diploid pair genotyping over complex/soft-clipped windows."""
    if not p.enable_reassembly:
        return pileup_calls

    L_ctg = int(g.lengths[ctg_id])
    cstart = int(g.offsets[ctg_id])
    ctg_seq_u8 = g.seq[cstart: cstart + L_ctg]
    ctg_name = g.names[ctg_id]

    # Collect active trigger points: variant calls, raw indel candidates, and soft-clip breakpoint clusters
    trig_pts: list[tuple[int, int, str]] = []
    for c in pileup_calls:
        k_tag = "ind" if c["source"] == "pileup_indel" else "snv"
        trig_pts.append((c["pos0"], c["pos0"] + len(c["ref"]), k_tag))
    for a0, r_s, a_s, cnt_i in raw_indels:
        if cnt_i >= p.indel_min_alt:
            trig_pts.append((a0, a0 + max(len(r_s), 2), "ind"))
    if len(cand["pos0"]):
        dp_c = cand["counts"].sum(axis=1, dtype=np.int32) + cand["delc"].astype(np.int32)
        clip_m = (cand["clipc"] >= p.active_min_clip) & (cand["clipc"] >= 0.14 * np.maximum(dp_c, 1))
        for p0_c in cand["pos0"][clip_m]:
            trig_pts.append((int(p0_c), int(p0_c) + 1, "clip"))

    if not trig_pts:
        return pileup_calls

    trig_pts.sort(key=lambda x: x[0])
    clusters: list[list] = []
    for s0, e0, kind_t in trig_pts:
        if clusters and s0 <= clusters[-1][1] + p.active_merge_bp:
            clusters[-1][1] = max(clusters[-1][1], e0)
            clusters[-1][2].append(kind_t)
            clusters[-1][3].append(s0)
        else:
            clusters.append([s0, e0, [kind_t], [s0]])

    z_ivs = acc.zoom_by_ctg.get(ctg_id)

    def _in_zoom(a0: int, b0: int) -> bool:
        if z_ivs is None or len(z_ivs[0]) == 0:
            return False
        zs, ze = z_ivs
        idx = int(np.searchsorted(ze, a0, side="right"))
        return idx < len(zs) and int(zs[idx]) < b0

    # Select windows that genuinely benefit from multi-k Omnitig reassembly:
    #   - Any cluster with an indel candidate ("ind") or soft-clip breakpoint ("clip")
    #   - Adjacent MNP SNVs within <= 8 bp (co-occurring on single reads so nm >= 2 captures 100% of ALT reads)
    #   - Multi-event clusters inside explicit `zoom_regions` (where all reads, including nm=0/1, are captured)
    active_wins: list[tuple[int, int, bool]] = []
    for s0, e0, kinds, starts in clusters:
        if e0 - s0 > 180:
            continue
        iz = _in_zoom(s0, e0)
        has_ind_or_clip = ("ind" in kinds) or ("clip" in kinds)
        has_close_mnp = len(starts) >= 2 and any(starts[i + 1] - starts[i] <= 8 for i in range(len(starts) - 1))
        if has_ind_or_clip or has_close_mnp or (iz and len(kinds) >= 2):
            active_wins.append((s0, e0, iz))

    if not active_wins:
        return pileup_calls

    meta6, _, pool = acc.contig_reads(ctg_id)
    if len(meta6) == 0:
        return pileup_calls
    r_pos0 = meta6[:, 0]

    # Query uncaptured full-span non-indel depth across (s0, mid, e0-1) in one batch,
    # clamping away from the [0, 32) and [L_ctg - 32, L_ctg) contig-end bins where uncap_dp cannot increment (L39)
    q_m = np.array([(s0 + e0) // 2 for s0, e0, _ in active_wins], dtype=np.int32)
    dp_mids, _, _ = acc.query_positions(ctg_id, q_m)
    if L_ctg >= 68:
        lo_clamp = 32
        hi_clamp = L_ctg - 33
        q_s = np.clip(np.array([s0 for s0, _, _ in active_wins], dtype=np.int32), lo_clamp, hi_clamp)
        q_mc = np.clip(q_m, lo_clamp, hi_clamp)
        q_e = np.clip(np.array([max(s0, e0 - 1) for s0, e0, _ in active_wins], dtype=np.int32), lo_clamp, hi_clamp)
    else:
        q_s = np.array([s0 for s0, _, _ in active_wins], dtype=np.int32)
        q_mc = q_m
        q_e = np.array([max(s0, e0 - 1) for s0, e0, _ in active_wins], dtype=np.int32)
    _, _, uncap_m = acc.query_positions(ctg_id, q_mc)
    _, _, uncap_s = acc.query_positions(ctg_id, q_s)
    _, _, uncap_e = acc.query_positions(ctg_id, q_e)
    uncap_min = np.minimum(uncap_m, np.minimum(uncap_s, uncap_e))

    drop_mask = np.zeros(len(pileup_calls), dtype=bool)
    call_pos = np.array([c["pos0"] for c in pileup_calls], dtype=np.int32) if pileup_calls else np.zeros(0, dtype=np.int32)
    call_end = np.array([c["pos0"] + len(c["ref"]) for c in pileup_calls], dtype=np.int32) if pileup_calls else np.zeros(0, dtype=np.int32)
    call_is_ind = np.array(["indel" in str(c.get("source", "")) for c in pileup_calls], dtype=bool) if pileup_calls else np.zeros(0, dtype=bool)
    new_calls: list[dict] = []

    flank = p.active_flank_bp
    for w_idx, (s0, e0, in_zoom_win) in enumerate(active_wins):
        lo = max(0, s0 - flank)
        hi = min(L_ctg, e0 + flank)
        if hi - lo < 65 or hi - lo > 320:
            continue
        ref_win = g.bases(cstart + lo, cstart + hi)
        if "N" in ref_win:
            continue

        # Binary search captured reads starting in [lo - 220, hi]
        i_lo = int(np.searchsorted(r_pos0, lo - 220, side="left"))
        i_hi = int(np.searchsorted(r_pos0, hi + 20, side="right"))
        if i_hi <= i_lo:
            continue

        # Filter to reads whose projected span [pos0 - lead_clip, pos0 + rspan + trail_clip) overlaps [s0 - 8, e0 + 8]
        sel = []
        for ri in range(i_lo, i_hi):
            p_r, rsp, lc, tc, rln, _ = meta6[ri]
            if (p_r - lc) < (e0 + 8) and (p_r + rsp + tc) > (s0 - 8) and rln >= 35:
                sel.append(ri)
        if len(sel) < 2:
            continue
        if len(sel) > 120:
            # Uniform subsample if ultra-deep spike
            step = max(1, len(sel) // 120)
            sel = sel[::step][:120]

        reads_seq = []
        reads_qual = []
        r_offs = np.empty(len(sel), dtype=np.int64)
        r_lens = np.empty(len(sel), dtype=np.int32)
        for k_i, ri in enumerate(sel):
            rln = int(meta6[ri, 4])
            soff = int(meta6[ri, 5])
            r_offs[k_i] = soff
            r_lens[k_i] = rln
            reads_seq.append(pool[soff: soff + rln].tobytes().decode("ascii", errors="replace"))
            reads_qual.append(pool[soff + rln: soff + 2 * rln])

        # Geometric span factor for multi-bp active windows: fraction of 151-bp reads spanning the entire [s0, e0) core
        w_span_bp = max(1, e0 - s0)
        span_frac = max(0.15, min(1.0, (151.0 - float(w_span_bp - 1)) / 151.0))
        raw_uncap = int(uncap_min[w_idx])
        if raw_uncap == 0 and (s0 < 48 or e0 > L_ctg - 48):
            raw_uncap = max(0, int(dp_mids[w_idx]) - len(sel))
        n_uncap = 0 if in_zoom_win else int(round(float(raw_uncap) * span_frac))
        cand_haps = OZ.discover_candidate_haplotypes(
            ref_win, reads_seq, quals=reads_qual, uncap_ref_depth=n_uncap,
            k_list=(31, 55), min_weight=1.8, max_candidates=7,
        )
        # Inject pileup indel candidates overlapping [s0, e0] so k-mer dropouts never lose recall
        for a0, r_s, a_s, cnt_i in raw_indels:
            if len(cand_haps) >= 12:
                break
            if s0 <= a0 and (a0 + len(r_s)) <= hi and a0 >= lo and cnt_i >= p.indel_min_alt:
                off_w = a0 - lo
                if ref_win[off_w: off_w + len(r_s)] == r_s:
                    h_ind = ref_win[:off_w] + a_s + ref_win[off_w + len(r_s):]
                    if h_ind not in cand_haps:
                        cand_haps.append(h_ind)

        if len(cand_haps) <= 1:
            continue

        # Pad every candidate haplotype with 165 bp of flanking reference so all 151 bp reads align end-to-end
        pad_lo = max(0, lo - 165)
        pad_hi = min(L_ctg, hi + 165)
        left_pad = g.bases(cstart + pad_lo, cstart + lo)
        right_pad = g.bases(cstart + hi, cstart + pad_hi)
        padded_haps = [left_pad + h + right_pad for h in cand_haps]

        ll_mat = score_reads_against_padded_haps(pool, r_offs, r_lens, padded_haps)
        # Per-read outlier floor (-18 nats below that read's best haplotype) prevents 1 chimeric read from dominating
        row_best = ll_mat.max(axis=1, keepdims=True)
        ll_mat = np.maximum(ll_mat, row_best - 18.0)

        n_h = len(cand_haps)
        # Every read in `n_uncap` (`16-bp` full-span with >=12 bp clean M anchor on both sides, indel==0, clip<5, nm<=1)
        # is guaranteed to span `[s0, e0)` without an indel or multi-base MNP (-8.5 nats vs indel/MNP haplotypes).
        pure_ll = np.full(n_h, -8.5, dtype=np.float64)
        pure_ll[0] = -0.15
        for h_i in range(1, n_h):
            ch_seq = cand_haps[h_i]
            if len(ch_seq) == len(ref_win):
                n_mm = sum(1 for a_c, b_c in zip(ch_seq, ref_win) if a_c != b_c)
                if n_mm <= 1:
                    pure_ll[h_i] = -0.15

        if str(p.ploidy).strip().lower() == "haploid":
            eff_asm_wsaf = max(0.15, min(0.99, float(p.haploid_min_wsaf)))
            best_per_r = np.argmax(ll_mat, axis=1)
            sup_0 = n_uncap + int((best_per_r == 0).sum())
            total_span_h = max(int(dp_mids[w_idx]), len(sel) + n_uncap, 1)
            ll_ref_only = float(ll_mat[:, 0].sum() + n_uncap * pure_ll[0])
            # Evaluate ALL candidate haplotypes against total spanning depth (#32) so multi-clone mixtures keep every clone
            agg_diffs: dict[tuple[int, str, str], list[float]] = {}
            for h_i in range(1, n_h):
                sup_h = int((best_per_r == h_i).sum())
                if sup_h < 2:
                    continue
                w_hat_single = float(np.clip(sup_h / float(total_span_h), 0.01, 0.999))
                lw_alt = math.log(w_hat_single)
                lw_ref = math.log(max(1.0 - w_hat_single, 1e-9))
                ll_mix = float(
                    np.logaddexp(ll_mat[:, 0] + lw_ref, ll_mat[:, h_i] + lw_alt).sum()
                    + n_uncap * float(np.logaddexp(pure_ll[0] + lw_ref, pure_ll[h_i] + lw_alt))
                )
                q_h = min(999.0, max(0.0, 10.0 * max(0.0, ll_mix - ll_ref_only) / LN10))
                if q_h < p.active_min_qual:
                    continue
                for p0_d, r_s, a_s in OZ.decompose_haplotype_diffs(ref_win, cand_haps[h_i], lo, ctg_seq_u8):
                    cur_agg = agg_diffs.get((p0_d, r_s, a_s))
                    if cur_agg is None:
                        agg_diffs[(p0_d, r_s, a_s)] = [float(sup_h), q_h]
                    else:
                        cur_agg[0] += float(sup_h)
                        cur_agg[1] = max(cur_agg[1], q_h)
            asm_records = []
            for (p0_d, r_s, a_s), (sup_tot, q_best) in agg_diffs.items():
                w_tot = float(np.clip(sup_tot / float(total_span_h), 0.01, 0.999))
                if w_tot < eff_asm_wsaf:
                    continue
                gq_h = min(99, max(0, int(round(q_best))))
                sup_int = int(round(sup_tot))
                asm_records.append(dict(
                    ctg_id=ctg_id, chrom=ctg_name, pos0=p0_d, pos=p0_d + 1,
                    ref=r_s, alt=a_s, gt="1", qual=round(q_best, 2), gq=gq_h,
                    dp=max(total_span_h, sup_0 + sup_int), ad=(sup_0, sup_int),
                    wsaf=(round(w_tot, 4),), hap_side=0, source="omnitig_asm_wsaf",
                ))
            if not asm_records:
                continue
        else:
            pairs: list[tuple[int, int]] = []
            pair_posts: list[float] = []
            ln2 = math.log(2.0)
            for h1 in range(n_h):
                for h2 in range(h1, n_h):
                    if h1 == h2:
                        ll_cap = float(ll_mat[:, h1].sum())
                        ll_uncap = float(n_uncap * pure_ll[h1])
                    else:
                        ll_cap = float(np.logaddexp(ll_mat[:, h1] - ln2, ll_mat[:, h2] - ln2).sum())
                        ll_uncap = float(n_uncap * np.logaddexp(pure_ll[h1] - ln2, pure_ll[h2] - ln2))
                    if h1 == 0 and h2 == 0:
                        prior = 0.0
                    elif h1 == 0:
                        prior = math.log(p.indel_prior_het)
                    elif h1 == h2:
                        prior = math.log(p.indel_prior_hom)
                    else:
                        prior = math.log(p.indel_prior_12)
                    pairs.append((h1, h2))
                    pair_posts.append(ll_cap + ll_uncap + prior)

            posts_arr = np.array(pair_posts, dtype=np.float64)
            lz = float(logsumexp(posts_arr))
            qual = min(999.0, max(0.0, -10.0 * (posts_arr[0] - lz) / LN10))
            best_idx = int(np.argmax(posts_arr))
            h1_b, h2_b = pairs[best_idx]
            s_posts = np.sort(posts_arr)
            gq = min(99, max(0, int(round(10.0 * (s_posts[-1] - s_posts[-2]) / LN10))))

            # Count read support per winning haplotype
            best_per_r = np.argmax(ll_mat, axis=1)
            sup_0 = n_uncap + int((best_per_r == 0).sum())
            sup_1 = int((best_per_r == h2_b).sum())
            sup_a = int((best_per_r == h1_b).sum())
            dp_est = max(int(dp_mids[w_idx]), len(sel) + n_uncap)

            if h1_b == 0 and h2_b == 0:
                if in_zoom_win and gq >= 25 and len(call_pos):
                    ov = (call_pos < e0) & (call_end > s0)
                    for ci_drop in np.where(ov)[0]:
                        if pileup_calls[ci_drop]["qual"] < 60.0:
                            drop_mask[ci_drop] = True
                continue

            if qual < p.active_min_qual or sup_1 < 2:
                continue

            ad_est = (sup_0, sup_a, sup_1)
            asm_records = _format_diploid_pair_calls(
                ctg_id, ctg_name, ctg_seq_u8, ref_win, lo,
                cand_haps[h1_b], cand_haps[h2_b], qual, gq, dp_est, ad_est,
            )

        if not in_zoom_win and asm_records:
            # Outside explicit zoom regions, single-SNV reads (NM==1, indel==0) are uncaptured (#30, #31).
            # Keep indels, MNPs, and close SNV clusters (<=45 bp) from reassembly; leave isolated SNVs to full pileup!
            snv_positions = sorted(
                r["pos0"] for r in asm_records
                if len(r["ref"]) == 1 and all(len(a) == 1 for a in str(r["alt"]).split(","))
            )
            clustered_snv_pos: set[int] = set()
            for idx_sp in range(len(snv_positions)):
                if (idx_sp > 0 and snv_positions[idx_sp] - snv_positions[idx_sp - 1] <= 45) or (
                    idx_sp + 1 < len(snv_positions) and snv_positions[idx_sp + 1] - snv_positions[idx_sp] <= 45
                ):
                    clustered_snv_pos.add(snv_positions[idx_sp])
            asm_records = [
                r for r in asm_records
                if not (len(r["ref"]) == 1 and all(len(a) == 1 for a in str(r["alt"]).split(",")))
                or (r["pos0"] in clustered_snv_pos)
            ]

        if not asm_records:
            continue

        if in_zoom_win:
            rep_lo = min(s0, min(r["pos0"] for r in asm_records))
            rep_hi = max(e0, max(r["pos0"] + len(r["ref"]) for r in asm_records))
            if len(call_pos):
                drop_mask |= (call_pos < rep_hi) & (call_end > rep_lo)
        elif len(call_pos):
            # Outside zoom regions, indels replace nearby pileup indels only; MNP/clustered SNVs replace SNVs at their span (#30, #31)
            is_hap = str(p.ploidy).strip().lower() == "haploid"
            for r in asm_records:
                r_lo = r["pos0"]
                r_hi = r["pos0"] + len(r["ref"])
                r_is_pure_indel = any(len(a) != len(r["ref"]) for a in str(r["alt"]).split(","))
                if r_is_pure_indel:
                    if not is_hap:
                        drop_mask |= call_is_ind & (call_pos < r_hi + 14) & (call_end > r_lo - 14)
                    else:
                        # In haploid mode, only drop identical/shifted duplicate of the same indel length so co-infecting STRs survive (#32)
                        for ci_p, pc in enumerate(pileup_calls):
                            if call_is_ind[ci_p] and pc["pos0"] < r_hi + 14 and (pc["pos0"] + len(pc["ref"])) > r_lo - 14:
                                if (len(pc["alt"]) - len(pc["ref"])) == (len(r["alt"]) - len(r["ref"])):
                                    drop_mask[ci_p] = True
                else:
                    drop_mask |= (~call_is_ind) & (call_pos < r_hi + 1) & (call_end > r_lo - 1)
        new_calls.extend(asm_records)

    kept = [c for i, c in enumerate(pileup_calls) if not drop_mask[i]]
    kept.extend(new_calls)
    return kept


def resolve_diploid_overlaps(calls: list[dict], ploidy: str = "diploid") -> list[dict]:
    """Ensure ploidy consistency across overlapping VCF records on a contig without inventing genotypes."""
    if not calls:
        return []
    # Deduplicate identical (pos0, ref, alt) keeping highest QUAL
    best_by_key: dict[tuple[int, str, str], dict] = {}
    for c in calls:
        k = (c["pos0"], c["ref"], c["alt"])
        if k not in best_by_key or c["qual"] > best_by_key[k]["qual"]:
            best_by_key[k] = c

    if str(ploidy).strip().lower() == "haploid":
        # Haploid / mixed-clone mixtures can carry multiple distinct alleles across co-infecting clones;
        # keep highest-QUAL per unique allele without forcing diploid 1/2 synthesis or 2-copy eviction.
        out_h = list(best_by_key.values())
        out_h.sort(key=lambda x: (x["pos0"], x["ref"], x["alt"]))
        return out_h

    # Group records at identical pos0 (unifying REF if needed) to synthesize trans 1/2 compound heterozygotes (#31)
    by_pos: dict[int, list[dict]] = {}
    for c in best_by_key.values():
        by_pos.setdefault(c["pos0"], []).append(c)

    merged_pos: list[dict] = []
    for p0, grp in by_pos.items():
        if len(grp) == 1:
            merged_pos.append(grp[0])
            continue
        grp.sort(key=lambda x: -x["qual"])
        g0, g1 = grp[0], grp[1]
        hs0 = int(g0.get("hap_side", 0))
        hs1 = int(g1.get("hap_side", 0))
        opp_phase = (hs0, hs1) in ((1, 2), (2, 1))
        ac0 = int(g0["ad"][1])
        ac1 = int(g1["ad"][1])
        ac_sum = ac0 + ac1
        dp_comb = max(int(g0["dp"]), int(g1["dp"]), ac_sum)
        # Each single-allele call's ref AD includes reads supporting the OTHER alt allele at p0 (#31):
        # true reference-only support is dp_comb - ac0 - ac1 when the two alt read sets are disjoint!
        true_rc = max(0, dp_comb - ac_sum)
        disjoint_trans = (
            ac0 >= 2
            and ac1 >= 2
            and ac_sum <= int(1.20 * max(dp_comb, 1))
            and true_rc <= 0.18 * max(1, dp_comb)
        )
        u_ref = g0["ref"] if len(g0["ref"]) >= len(g1["ref"]) else g1["ref"]
        can_unify = u_ref.startswith(g0["ref"]) and u_ref.startswith(g1["ref"])
        u_alt0 = g0["alt"] + u_ref[len(g0["ref"]):] if can_unify else g0["alt"]
        u_alt1 = g1["alt"] + u_ref[len(g1["ref"]):] if can_unify else g1["alt"]
        if (
            can_unify
            and "," not in g0["alt"]
            and "," not in g1["alt"]
            and u_alt0 != u_alt1
            and g0["gt"] == "0/1"
            and g1["gt"] == "0/1"
            and (opp_phase or disjoint_trans)
        ):
            merged_pos.append(dict(
                g0,
                ref=u_ref,
                alt=f"{u_alt0},{u_alt1}",
                gt="1/2",
                qual=max(g0["qual"], g1["qual"]),
                gq=min(g0["gq"], g1["gq"]),
                dp=max(dp_comb, true_rc + ac_sum),
                ad=(true_rc, ac0, ac1),
                hap_side=0,
            ))
        else:
            # Keep non-conflicting records (e.g. SNV at indel anchor base) for the span-conflict sweep below
            for g_item in grp:
                merged_pos.append(g_item)

    merged_pos.sort(key=lambda x: (x["pos0"], -len(x["ref"]), -x["qual"]))

    def _effective_span(rec: dict) -> tuple[int, int]:
        """Modified reference bases: pure 1-bp left-anchored indels do NOT consume their anchor base (#31)."""
        s_r = int(rec["pos0"])
        r_ref = str(rec["ref"])
        alts = str(rec["alt"]).split(",")
        if all(len(a) != len(r_ref) and len(a) >= 1 and len(r_ref) >= 1 and a[0] == r_ref[0] for a in alts):
            # Pure insertion or deletion anchored at s_r: modified span starts at s_r + 1
            if len(r_ref) > 1:
                return s_r + 1, s_r + len(r_ref)
            # Pure insertion between s_r and s_r + 1: zero reference bases deleted (conflicts only with same-pos ins)
            return s_r + 1, s_r + 1
        return s_r, s_r + len(r_ref)

    # Sweep reference intervals to prevent diploid copy-count > 2 across genuinely overlapping modified spans.
    # Never rewrite 1/1 -> 0/1: drop the lower-QUAL conflicting record(s) intact!
    out: list[dict] = []
    active_spans: list[tuple[int, int, int, dict]] = []  # (eff_s0, eff_e0, copies_used, record)
    for rec in merged_pos:
        s0, e0 = _effective_span(rec)
        active_spans = [sp for sp in active_spans if sp[1] > s0 or (sp[0] == sp[1] == s0)]
        ov_recs = [
            sp for sp in active_spans
            if (max(sp[0], s0) < min(sp[1], e0)) or (sp[0] == sp[1] == s0 == e0)
        ]
        used = sum(sp[2] for sp in ov_recs)
        gt = rec["gt"]
        copies = 2 if gt in ("1/1", "1/2") else 1
        if used + copies > 2:
            need_free = used + copies - 2
            freed = 0
            to_evict = []
            for sp in sorted(ov_recs, key=lambda x: x[3]["qual"]):
                to_evict.append(sp)
                freed += sp[2]
                if freed >= need_free:
                    break
            if freed >= need_free and to_evict and rec["qual"] > max(sp[3]["qual"] for sp in to_evict):
                for sp in to_evict:
                    if sp[3] in out:
                        out.remove(sp[3])
                    active_spans.remove(sp)
            else:
                continue
        out.append(rec)
        active_spans.append((s0, e0, copies, rec))

    out.sort(key=lambda x: (x["pos0"], x["ref"], x["alt"]))
    return out


_FORK_CALL_CTX: tuple | None = None


def _fork_call_contig_one(cid: int):
    assert _FORK_CALL_CTX is not None
    g_f, acc_f, p_f, want_both_f = _FORK_CALL_CTX
    res = call_contig_diploid(g_f, acc_f, cid, p_f, free_after=False, return_pileup_also=want_both_f)
    return cid, res


def call_contig_diploid_batch(
    g: G.Genome,
    acc: PagedContigAccumulator,
    contig_ids: Sequence[int],
    p: DiploidParams,
    call_workers: int = 1,
    return_pileup_also: bool = False,
    weights: np.ndarray | None = None,
) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """Call diploid variants across `contig_ids` sequentially (`call_workers=1`) or via <=4 zero-COW fork workers."""
    import multiprocessing as mp

    cids = list(contig_ids)
    eff_w = max(1, min(int(call_workers), 4, max(len(cids), 1)))
    acc.sort_all_reads()

    calls_by_ctg: dict[int, list[dict]] = {}
    pileup_by_ctg: dict[int, list[dict]] = {}
    if not cids:
        return calls_by_ctg, pileup_by_ctg

    use_fork = eff_w > 1 and "fork" in mp.get_all_start_methods()
    if not use_fork:
        for cid in cids:
            if return_pileup_also:
                recs, pil_recs = call_contig_diploid(g, acc, cid, p, free_after=True, return_pileup_also=True)
                pileup_by_ctg[cid] = pil_recs
            else:
                recs = call_contig_diploid(g, acc, cid, p, free_after=True)
            calls_by_ctg[cid] = recs
        return calls_by_ctg, pileup_by_ctg

    # Warm Numba JIT in parent once before fork so workers share compiled machine code via COW
    OZ.decompose_haplotype_diffs("ACGTACGT", "ACGTTACGT", 0)
    ordered = sorted(cids, key=lambda c: -int(weights[c]) if weights is not None else -int(g.lengths[c]))

    global _FORK_CALL_CTX
    _FORK_CALL_CTX = (g, acc, p, return_pileup_also)
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=eff_w, mp_context=ctx) as ex:
            futs = {ex.submit(_fork_call_contig_one, cid): cid for cid in ordered}
            for fut in as_completed(futs):
                cid, res = fut.result()
                if return_pileup_also:
                    recs, pil_recs = res
                    calls_by_ctg[cid] = recs
                    pileup_by_ctg[cid] = pil_recs
                else:
                    calls_by_ctg[cid] = res
    finally:
        _FORK_CALL_CTX = None
    if len(calls_by_ctg) != len(cids):
        raise RuntimeError(f"Incomplete contig batch result: got {len(calls_by_ctg)} of {len(cids)} contigs")
    return calls_by_ctg, pileup_by_ctg


def call_contig_diploid(
    g: G.Genome,
    acc: PagedContigAccumulator,
    ctg_id: int,
    p: DiploidParams,
    free_after: bool = True,
    return_pileup_also: bool = False,
):
    """Complete diploid calling pipeline (SNVs + Indels + Multi-k Omnitig Reassembly) for one contig."""
    cand = acc.contig_candidates(ctg_id, min_alt=p.cand_min_alt, min_clip=p.active_min_clip)
    snvs = call_diploid_snvs_contig(g, ctg_id, cand, p)
    indels, raw_ind = call_diploid_indels_contig(g, acc, ctg_id, p)
    combined = snvs + indels
    pileup_resolved = resolve_diploid_overlaps(combined, ploidy=p.ploidy) if return_pileup_also else None
    if p.enable_reassembly and (len(combined) > 0 or len(raw_ind) > 0 or (len(cand["clipc"]) > 0 and cand["clipc"].max() >= p.active_min_clip)):
        combined = reassemble_active_regions_contig(g, acc, ctg_id, cand, combined, raw_ind, p)
    resolved = resolve_diploid_overlaps(combined, ploidy=p.ploidy)
    if free_after:
        acc.free_contig(ctg_id)
    if return_pileup_also:
        return resolved, pileup_resolved
    return resolved


def write_diploid_vcf(
    g: G.Genome,
    calls_by_contig: dict[int, list[dict]],
    out_vcf_gz: str | Path,
    sample: str = "HG002",
    contig_ids: Sequence[int] | None = None,
) -> int:
    """Write standard bgzipped + tabix-indexed VCF v4.2 with GT:GQ:DP:AD:WSAF."""
    out_vcf_gz = Path(out_vcf_gz)
    out_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp_vcf = out_vcf_gz.with_suffix("") if str(out_vcf_gz).endswith(".gz") else out_vcf_gz.with_name(out_vcf_gz.name + ".tmp")

    cids = list(contig_ids) if contig_ids is not None else list(range(len(g.names)))
    n_written = 0
    with open(tmp_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=FocaloopVariationLoop-0.4.1\n")
        f.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
        for cid in range(len(g.names)):
            f.write(f"##contig=<ID={g.names[cid]},length={int(g.lengths[cid])}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
        f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')
        f.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for ref and alt alleles">\n')
        f.write('##FORMAT=<ID=WSAF,Number=A,Type=Float,Description="Within-sample alternate allele fraction">\n')
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n")
        for cid in cids:
            recs = calls_by_contig.get(cid, [])
            for r in recs:
                ad_vals = list(r["ad"])
                ad_s = ",".join(str(x) for x in ad_vals)
                if r.get("wsaf") is not None:
                    ws_raw = r["wsaf"]
                    ws_vals = list(ws_raw) if isinstance(ws_raw, (list, tuple)) else [float(ws_raw)]
                else:
                    denom = float(max(sum(int(x) for x in ad_vals), int(r.get("dp", 0)), 1))
                    ws_vals = [min(1.0, max(0.0, float(x) / denom)) for x in ad_vals[1:]] or [0.0]
                wsaf_s = ",".join(f"{float(w):.4f}" for w in ws_vals)
                f.write(
                    f"{r['chrom']}\t{r['pos']}\t.\t{r['ref']}\t{r['alt']}\t{r['qual']:.2f}\tPASS\t.\t"
                    f"GT:GQ:DP:AD:WSAF\t{r['gt']}:{r['gq']}:{r['dp']}:{ad_s}:{wsaf_s}\n"
                )
                n_written += 1

    if str(out_vcf_gz).endswith(".gz"):
        bgzip = shutil.which("bgzip")
        tabix = shutil.which("tabix")
        if bgzip:
            subprocess.run([bgzip, "-f", "-@", "4", str(tmp_vcf)], check=True)
            if Path(str(tmp_vcf) + ".gz") != out_vcf_gz:
                os.replace(str(tmp_vcf) + ".gz", out_vcf_gz)
            if tabix:
                subprocess.run([tabix, "-f", "-p", "vcf", str(out_vcf_gz)], check=True)
        else:
            with open(tmp_vcf, "rb") as fin, gzip.open(out_vcf_gz, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            tmp_vcf.unlink(missing_ok=True)
    return n_written


# ============================================================================
# 5. End-to-End Streaming & Focused .crai Pipeline Runners
# ============================================================================

def _rss_bytes() -> int:
    ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1 if os.uname().sysname == "Darwin" else 1024
    return int(max(ru_self, ru_child) * scale)


class ProcTreeRssMonitor:
    """Sample concurrent sum of VmRSS across `root_pid` and all descendant processes via `/proc`."""

    def __init__(self, root_pid: int | None = None, interval: float = 0.5):
        import threading
        self.root_pid = root_pid or os.getpid()
        self.interval = interval
        self.page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        self.peak_tree_rss_bytes = 0
        self.peak_self_rss_bytes = 0
        self.samples = 0
        self._alive = True
        self._has_proc = Path("/proc/self/statm").exists()
        self._thr = threading.Thread(target=self._loop, daemon=True)

    def sample_once(self) -> int:
        if not self._has_proc:
            v = _rss_bytes()
            self.peak_tree_rss_bytes = max(self.peak_tree_rss_bytes, v)
            return v
        children_of: dict[int, list[int]] = {}
        rss_of: dict[int, int] = {}
        try:
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                try:
                    with open(f"/proc/{pid}/stat", "rb") as fs:
                        raw = fs.read()
                    rp = raw.rfind(b")")
                    if rp < 0:
                        continue
                    ppid = int(raw[rp + 2:].split(None, 3)[1])
                    with open(f"/proc/{pid}/statm", "rb") as fm:
                        res_pages = int(fm.read().split(None, 2)[1])
                    children_of.setdefault(ppid, []).append(pid)
                    rss_of[pid] = res_pages * self.page_size
                except OSError:
                    continue
        except OSError:
            return self.peak_tree_rss_bytes

        stack = [self.root_pid]
        seen: set[int] = set()
        tree_rss = 0
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            tree_rss += rss_of.get(cur, 0)
            stack.extend(children_of.get(cur, ()))

        self_rss = rss_of.get(self.root_pid, 0)
        if tree_rss > self.peak_tree_rss_bytes:
            self.peak_tree_rss_bytes = tree_rss
        if self_rss > self.peak_self_rss_bytes:
            self.peak_self_rss_bytes = self_rss
        self.samples += 1
        return tree_rss

    def _loop(self):
        while self._alive:
            self.sample_once()
            time.sleep(self.interval)

    def start(self):
        self.sample_once()
        self._thr.start()
        return self

    def stop(self) -> tuple[int, int]:
        self._alive = False
        self._thr.join(timeout=2.0)
        self.sample_once()
        return self.peak_tree_rss_bytes, self.peak_self_rss_bytes


def _cpu_times() -> tuple[float, float]:
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime, c.ru_utime + c.ru_stime


def run_varloop_fastq(
    g: G.Genome,
    ref_fasta: str | Path,
    r1_gz: str | Path,
    r2_gz: str | Path,
    out_vcf_gz: str | Path,
    params: DiploidParams | None = None,
    map_threads: int = 56,
    target_regions: Sequence[tuple[str, int, int]] | None = None,
    zoom_regions: Sequence[tuple[str, int, int]] | None = None,
    sample: str = "HG002",
    use_sti_index: bool = True,
    p_sample: float = 1.0,
    also_out_pileup_vcf_gz: str | Path | None = None,
    call_workers: int = 1,
    mapper_name: str = "strobealign",
    clip_regions: Sequence[tuple[str, int, int]] | None = None,
    callable_npz_out: str | Path | None = None,
) -> dict:
    """Stream paired FASTQ -> aligner -> lock-free C PagedContigAccumulator -> Diploid Omnitig VCF."""
    import tempfile
    import threading
    p = params or DiploidParams()
    _validate_fastq_input(Path(r1_gz))
    _validate_fastq_input(Path(r2_gz))
    rss_mon = ProcTreeRssMonitor().start()
    t0 = time.time()
    cpu0 = _cpu_times()

    ref_str = str(ref_fasta)
    use_subsample_pipe = p_sample < 1.0
    if mapper_name == "strobealign":
        cmd = ["strobealign", "-t", str(map_threads), "-U", "--no-PG"]
        if use_sti_index and Path(ref_str + ".r150.sti").exists():
            cmd += ["-r", "150", "--use-index"]
        if use_subsample_pipe:
            cmd += ["--interleaved", ref_str, "-"]
        else:
            cmd += [ref_str, str(r1_gz), str(r2_gz)]
    elif mapper_name == "minimap2":
        cmd = ["minimap2", "-t", str(map_threads), "-K", "1M", "-ax", "sr", ref_str]
        if use_subsample_pipe:
            cmd += ["-"]
        else:
            cmd += [str(r1_gz), str(r2_gz)]
    elif mapper_name in ("bwa-mem2", "bwa"):
        cmd = [mapper_name, "mem", "-K", "1000000", "-t", str(map_threads)]
        if use_subsample_pipe:
            cmd += ["-p", ref_str, "-"]
        else:
            cmd += [ref_str, str(r1_gz), str(r2_gz)]
    else:
        raise ValueError(f"Unsupported mapper_name: {mapper_name!r}")

    sieve_stats = np.zeros(8, dtype=np.int64)
    sieve_err: list[str] = []
    t_stream0 = time.time()
    Path(out_vcf_gz).parent.mkdir(parents=True, exist_ok=True)

    with (
        tempfile.TemporaryFile(dir=str(Path(out_vcf_gz).parent)) as mapper_err_f,
        PagedContigAccumulator(g, p, target_regions=target_regions, zoom_regions=zoom_regions) as acc,
    ):
        if use_subsample_pipe:
            r_fd, w_fd = os.pipe()
            mapper = subprocess.Popen(cmd, stdin=r_fd, stdout=subprocess.PIPE, stderr=mapper_err_f, bufsize=0)
            os.close(r_fd)
            dec1 = _fastq_decomp_cmd(r1_gz, 6)
            dec2 = _fastq_decomp_cmd(r2_gz, 6)
            p_in1 = subprocess.Popen(dec1, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p_in2 = subprocess.Popen(dec2, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            def _pump_sieve():
                rc_i1 = 0
                rc_i2 = 0
                try:
                    lib = _lib()
                    keep_thr = min(int(p_sample * (1 << 32)), 1 << 32)
                    rc_s = lib.vl_fastq_sieve_paired(
                        p_in1.stdout.fileno(), p_in2.stdout.fileno(),
                        w_fd, -1, -1,
                        ct.POINTER(ct.c_uint64)(), ct.c_uint64(0),
                        ct.c_int32(5), ct.c_int32(0),
                        ct.c_uint64(keep_thr), ct.c_int32(1), ct.c_int32(30),
                        ct.c_int64(0), _ptr(sieve_stats, ct.c_int64),
                    )
                    if rc_s != 0:
                        sieve_err.append(f"vl_fastq_sieve_paired rc={rc_s}")
                finally:
                    try:
                        os.close(w_fd)
                    except OSError:
                        pass
                    if p_in1.stdout:
                        p_in1.stdout.close()
                    if p_in2.stdout:
                        p_in2.stdout.close()
                    rc_i1 = p_in1.wait()
                    rc_i2 = p_in2.wait()
                    if rc_i1 != 0 or rc_i2 != 0:
                        sieve_err.append(f"FASTQ decompressor failed: rc=({rc_i1},{rc_i2})")

            pump_thr = threading.Thread(target=_pump_sieve, daemon=True)
            pump_thr.start()
        else:
            mapper = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=mapper_err_f, bufsize=0)
            pump_thr = None

        try:
            acc.consume_sam_fd(mapper.stdout.fileno())
        finally:
            if mapper.stdout:
                mapper.stdout.close()
        if pump_thr is not None:
            pump_thr.join(timeout=120)
        rc = mapper.wait()
        mapper_err_f.seek(0)
        stderr_txt = mapper_err_f.read().decode("utf-8", errors="replace")
        if sieve_err:
            raise RuntimeError("; ".join(sieve_err))
        if rc != 0:
            raise RuntimeError(f"{mapper_name} exited with code {rc}: {stderr_txt[-500:]}")

        t_stream = time.time() - t_stream0
        rss_mon.sample_once()
        summ = acc.summary()
        sam_recs = int(summ["stats"][0])
        reads_cnt = int(summ["stats"][2])
        malf_cnt = int(summ["stats"][11])
        if sam_recs <= 0 or reads_cnt <= 0:
            raise RuntimeError(
                f"Sample {sample!r}: zero usable reads aligned/counted "
                f"(sam_records={sam_recs}, reads_counted={reads_cnt}, malformed_sam={malf_cnt})"
            )
        if malf_cnt > 0.5 * max(sam_recs, 1):
            raise RuntimeError(
                f"Sample {sample!r}: majority of SAM records rejected as malformed ({malf_cnt}/{sam_recs})"
            )
        if callable_npz_out is not None:
            acc.save_callable_npz(callable_npz_out, min_dp=1)
        med_dp = _estimate_median_depth(g, summ, acc._target_bin_mask)
        # Pre-sort all captured contig read arrays once before calling
        acc.sort_all_reads()

        t_call0 = time.time()
        cidx = {n: i for i, n in enumerate(g.names)}
        if target_regions is not None:
            want_cids = sorted({cidx[cn] for cn, _, _ in target_regions if cn in cidx})
        else:
            want_cids = [cid for cid in range(len(g.names)) if summ["ccount"][cid] > 0]

        want_both = also_out_pileup_vcf_gz is not None
        eff_call_workers = max(1, min(int(call_workers), 4))
        calls_by_ctg, pileup_by_ctg = call_contig_diploid_batch(
            g, acc, want_cids, p,
            call_workers=eff_call_workers,
            return_pileup_also=want_both,
            weights=summ["ccount"],
        )
        eff_clip = clip_regions if clip_regions is not None else target_regions
        if eff_clip is not None:
            calls_by_ctg = _clip_calls_to_regions(calls_by_ctg, eff_clip, cidx)
            if want_both:
                pileup_by_ctg = _clip_calls_to_regions(pileup_by_ctg, eff_clip, cidx)
        n_asm = sum(1 for recs in calls_by_ctg.values() for r in recs if str(r.get("source", "")).startswith("omnitig_asm"))
        t_call = time.time() - t_call0

    n_vcf = write_diploid_vcf(g, calls_by_ctg, out_vcf_gz, sample=sample, contig_ids=want_cids)
    n_pileup_vcf = None
    if also_out_pileup_vcf_gz is not None:
        n_pileup_vcf = write_diploid_vcf(g, pileup_by_ctg, also_out_pileup_vcf_gz, sample=sample, contig_ids=want_cids)

    cpu1 = _cpu_times()
    peak_tree_rss, peak_self_rss = rss_mon.stop()

    n_snv = sum(1 for recs in calls_by_ctg.values() for r in recs if len(r["ref"]) == 1 and all(len(a) == 1 for a in r["alt"].split(",")))
    n_indel = n_vcf - n_snv
    out_tel = {
        "sample": sample,
        "mapper": mapper_name,
        "map_threads": map_threads,
        "n_shards": p.n_shards,
        "dedup_bits": p.dedup_bits,
        "p_sample": p_sample,
        "zoom_assisted": zoom_regions is not None,
        "order_mode": summ["order_mode"],
        "sam_records": int(summ["stats"][0]),
        "primary_mapped": int(summ["stats"][1]),
        "reads_counted": int(summ["stats"][2]),
        "bases_counted": int(summ["stats"][3]),
        "median_depth": med_dp,
        "pcr_dups_skipped": int(summ["stats"][10]),
        "pcr_dup_fraction": round(float(summ["stats"][10]) / max(int(summ["stats"][1]), 1), 6),
        "malformed_sam_rejected": int(summ["stats"][11]),
        "overlap_bases_skipped": int(summ["stats"][7]),
        "informative_reads_captured": int(summ["stats"][8]),
        "acc_page_bytes": summ["page_bytes"],
        "acc_read_buf_bytes": summ["read_buf_bytes"],
        "acc_indel_tab_bytes": summ["indel_tab_bytes"],
        "acc_total_bytes": summ["total_acc_bytes"],
        "n_variants": n_vcf,
        "n_snv": n_snv,
        "n_indel": n_indel,
        "n_omnitig_assembled": n_asm,
        "stream_wall_s": round(t_stream, 3),
        "call_wall_s": round(t_call, 3),
        "total_wall_s": round(time.time() - t0, 3),
        "cpu_self_s": round(cpu1[0] - cpu0[0], 3),
        "cpu_children_s": round(cpu1[1] - cpu0[1], 3),
        "total_cpu_s": round((cpu1[0] - cpu0[0]) + (cpu1[1] - cpu0[1]), 3),
        "peak_rss_bytes": max(_rss_bytes(), peak_tree_rss),
        "peak_proctree_rss_bytes": peak_tree_rss,
        "peak_self_rss_bytes": peak_self_rss,
        "peak_scratch_bytes": 0,
        "vcf_path": str(out_vcf_gz),
    }
    if use_subsample_pipe:
        out_tel["sieve_spots_seen"] = int(sieve_stats[0])
        out_tel["sieve_spots_subsampled"] = int(sieve_stats[1])
    if also_out_pileup_vcf_gz is not None:
        out_tel["pileup_vcf_path"] = str(also_out_pileup_vcf_gz)
        out_tel["n_pileup_variants"] = n_pileup_vcf
    return out_tel


BGZF_EOF = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00\x42\x43\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"


def _estimate_median_depth(g: G.Genome, summ: dict, target_bin_mask: np.ndarray | None = None) -> float:
    bb = summ["binbases"]
    if target_bin_mask is not None and target_bin_mask.any():
        vals = bb[target_bin_mask].astype(np.float64) / 1000.0
    else:
        vals = bb[bb > 0].astype(np.float64) / 1000.0
    if len(vals) == 0:
        tot_len = max(1, int(summ.get("eff_bp") or int(g.lengths.sum())))
        return round(float(summ["stats"][3]) / float(tot_len), 2)
    return round(float(np.median(vals)), 2)


def _clip_calls_to_regions(
    calls_by_ctg: dict[int, list[dict]],
    clip_regions: Sequence[tuple[str, int, int]],
    cidx: dict[str, int],
) -> dict[int, list[dict]]:
    """Restrict emitted variant records to those overlapping unpadded target BED intervals (L38)."""
    by_cid: dict[int, list[tuple[int, int]]] = {}
    for cn, s0, e0 in clip_regions:
        if cn in cidx and int(e0) > int(s0):
            by_cid.setdefault(cidx[cn], []).append((int(s0), int(e0)))
    for cid in list(by_cid.keys()):
        ivs = sorted(by_cid[cid])
        merged: list[tuple[int, int]] = []
        for a, b in ivs:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        by_cid[cid] = merged

    clipped: dict[int, list[dict]] = {}
    for cid, recs in calls_by_ctg.items():
        ivs = by_cid.get(cid)
        if not ivs:
            clipped[cid] = []
            continue
        ends = [b for _, b in ivs]
        import bisect
        kept_r: list[dict] = []
        for r in recs:
            p0 = int(r["pos0"])
            p1 = p0 + max(1, len(str(r["ref"])))
            idx = bisect.bisect_right(ends, p0)
            if idx < len(ivs) and ivs[idx][0] < p1:
                kept_r.append(r)
        clipped[cid] = kept_r
    return clipped


def _validate_fastq_input(fq_path: Path) -> None:
    """Validate FASTQ(.gz) input magic bytes, non-emptiness, and gzip stream trailer integrity (#23)."""
    if not fq_path.is_file():
        raise FileNotFoundError(f"FASTQ file does not exist: {fq_path.name}")
    sz = fq_path.stat().st_size
    if sz == 0:
        raise ValueError(f"FASTQ input {fq_path.name!r} is empty (0 bytes)")
    with open(fq_path, "rb") as fh:
        head = fh.read(256)
    if head.startswith(b"BZh"):
        raise ValueError(f"Unsupported bzip2 FASTQ {fq_path.name!r} (expected uncompressed FASTQ or .fastq.gz)")
    if head.startswith(b"\xfd7zXZ\x00"):
        raise ValueError(f"Unsupported xz FASTQ {fq_path.name!r} (expected uncompressed FASTQ or .fastq.gz)")
    if head.startswith(b"\x28\xb5\x2f\xfd"):
        raise ValueError(f"Unsupported zstd FASTQ {fq_path.name!r} (expected uncompressed FASTQ or .fastq.gz)")
    if head.startswith((b"BAM\x01", b"CRAM")):
        raise ValueError(f"Alignment file {fq_path.name!r} supplied where FASTQ reads were expected")
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(fq_path, "rb") as gzf:
                first_chunk = gzf.read(256)
        except Exception as exc:
            raise ValueError(f"Corrupt or truncated gzip FASTQ {fq_path.name!r}: {exc}") from exc
        if not first_chunk:
            raise ValueError(f"Gzipped FASTQ {fq_path.name!r} decompresses to 0 bytes")
        if not first_chunk.startswith(b"@"):
            raise ValueError(f"Gzipped input {fq_path.name!r} is not FASTQ (first record does not start with '@')")
        # Verify full gzip stream integrity (catches truncated .gz even when external mappers ignore EOF errors)
        gz_tool = shutil.which("gzip")
        if gz_tool:
            chk = subprocess.run([gz_tool, "-t", str(fq_path)], capture_output=True, text=True)
            if chk.returncode != 0:
                raise ValueError(f"Truncated or corrupt gzip FASTQ {fq_path.name!r}: {chk.stderr.strip()}")
    else:
        low = head[:64].lstrip().lower()
        if low.startswith((b"<!doctype", b"<html", b"<?xml", b"<error")):
            raise ValueError(f"Input {fq_path.name!r} is an HTML/XML error document rather than FASTQ")
        if not head.startswith(b"@"):
            raise ValueError(f"Input {fq_path.name!r} is not valid FASTQ (first byte is not '@')")


def _parse_sam_header_sq(header_text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for ln in header_text.splitlines():
        if ln.startswith("@SQ\t") or ln.startswith("@SQ "):
            fields: dict[str, str] = {}
            for tok in ln.split("\t")[1:]:
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    fields[k] = v.strip()
            if "SN" in fields and "LN" in fields:
                out.append((fields["SN"], int(fields["LN"])))
    return out


def _extract_header_sq_from_cram_bytes(header_bytes: bytes, ref_fasta: Path, tmp_dir: Path) -> list[tuple[str, int]]:
    hdr_cram = tmp_dir / "header_probe.cram"
    hdr_cram.write_bytes(header_bytes + C_CRAM_EOF)
    try:
        if shutil.which("samtools"):
            pr = subprocess.run(
                ["samtools", "view", "-H", "--reference", str(ref_fasta), str(hdr_cram)],
                capture_output=True,
                text=True,
            )
            if pr.returncode != 0:
                raise RuntimeError(f"Failed reading CRAM header: {pr.stderr.strip()[-300:]}")
            return _parse_sam_header_sq(pr.stdout)
        import pysam
        with pysam.AlignmentFile(str(hdr_cram), "rc", reference_filename=str(ref_fasta)) as af:
            return _parse_sam_header_sq(str(af.header))
    finally:
        hdr_cram.unlink(missing_ok=True)


def _validate_sq_against_genome(
    g: G.Genome,
    sq_list: list[tuple[str, int]],
    required_contigs: Sequence[str] | None = None,
    label: str = "alignment",
) -> dict[str, int]:
    """Validate BAM/CRAM @SQ contigs & lengths against reference Genome (#22, #24); return `@SQ` index map."""
    if not sq_list:
        raise ValueError(f"{label}: header contains zero @SQ reference sequence lines")
    cidx = {n: i for i, n in enumerate(g.names)}
    sq_idx: dict[str, int] = {}
    sq_len: dict[str, int] = {}
    matched = 0
    for idx, (sn, ln) in enumerate(sq_list):
        sq_idx[sn] = idx
        sq_len[sn] = ln
        if sn in cidx:
            matched += 1
            ref_ln = int(g.lengths[cidx[sn]])
            if ln != ref_ln:
                raise ValueError(
                    f"{label}: @SQ contig {sn!r} length {ln} does not match reference FASTA length {ref_ln}"
                )
    if matched == 0:
        raise ValueError(
            f"{label}: zero @SQ contigs match reference FASTA "
            f"(header contigs: {[s for s, _ in sq_list[:5]]}, reference contigs: {list(g.names[:5])})"
        )
    if required_contigs:
        missing = [cn for cn in dict.fromkeys(required_contigs) if cn not in sq_idx]
        if missing:
            raise ValueError(
                f"{label}: target contigs {missing[:5]} are absent from alignment @SQ header "
                f"(header contigs: {[s for s, _ in sq_list[:5]]})"
            )
    return sq_idx


def _validate_local_aln_integrity(g: G.Genome, aln_path: Path, ref_fasta: Path, req_contigs: Sequence[str] | None = None) -> None:
    """Validate BAM/CRAM non-emptiness, BGZF/CRAM EOF markers, samtools quickcheck, and @SQ header (#24, #25)."""
    if not aln_path.is_file():
        raise FileNotFoundError(f"Alignment file does not exist: {aln_path.name}")
    sz = aln_path.stat().st_size
    if sz == 0:
        raise ValueError(f"Alignment file {aln_path.name!r} is empty (0 bytes)")
    with open(aln_path, "rb") as fh:
        magic = fh.read(4)
        tail_38 = b""
        if sz >= 28:
            fh.seek(max(0, sz - 38))
            tail_38 = fh.read(38)
    if magic == b"CRAM":
        if sz < len(C_CRAM_EOF) or tail_38[-len(C_CRAM_EOF):] != C_CRAM_EOF:
            raise RuntimeError(f"Truncated CRAM file {aln_path.name!r}: CRAM EOF container is absent")
    elif magic[:2] == b"\x1f\x8b":
        if sz < len(BGZF_EOF) or tail_38[-len(BGZF_EOF):] != BGZF_EOF:
            raise RuntimeError(f"Truncated BAM file {aln_path.name!r}: BGZF EOF marker is absent")
    elif magic != b"BAM\x01":
        raise ValueError(f"Input {aln_path.name!r} is neither a valid BAM nor CRAM file")

    if shutil.which("samtools"):
        qc = subprocess.run(["samtools", "quickcheck", str(aln_path)], capture_output=True, text=True)
        if qc.returncode != 0:
            raise RuntimeError(f"samtools quickcheck failed on {aln_path.name!r}: {qc.stderr.strip()}")
        pr = subprocess.run(
            ["samtools", "view", "-H", "--reference", str(ref_fasta), str(aln_path)],
            capture_output=True,
            text=True,
        )
        if pr.returncode != 0:
            raise RuntimeError(f"Failed reading alignment header of {aln_path.name!r}: {pr.stderr.strip()[-300:]}")
        sq_list = _parse_sam_header_sq(pr.stdout)
    else:
        import pysam
        mode = "rc" if magic == b"CRAM" else "rb"
        with pysam.AlignmentFile(str(aln_path), mode, reference_filename=str(ref_fasta)) as af:
            if getattr(af, "check_truncation", False) and not getattr(af, "eof", True):
                raise RuntimeError(f"Truncated alignment file {aln_path.name!r}: EOF marker is absent")
            sq_list = _parse_sam_header_sq(str(af.header))
    _validate_sq_against_genome(g, sq_list, required_contigs=req_contigs, label=aln_path.name)


def _fetch_remote_range_chunked(url: str, start: int, end: int, write_fn, max_chunk: int = 32 << 20, tries: int = 4) -> int:
    """Fetch remote byte interval `[start, end)` in `<= 32 MiB` HTTP 206 streaming chunks with retries (#26, #42)."""
    from . import streamer as ST
    ss = ST._session()
    total_written = 0
    pos = int(start)
    end_i = int(end)
    while pos < end_i:
        sub_end = min(end_i, pos + max_chunk)
        want = sub_end - pos
        for attempt in range(tries):
            got_sub = 0
            try:
                with ss.get(
                    url,
                    headers={"Range": f"bytes={pos + got_sub}-{sub_end - 1}"},
                    timeout=(30, 180),
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise RuntimeError(
                            f"Remote server ignored HTTP Range request on {url} (status {r.status_code}, expected 206)"
                        )
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            write_fn(chunk)
                            got_sub += len(chunk)
                if got_sub != want:
                    raise RuntimeError(f"Truncated HTTP 206 range response: got {got_sub} bytes, expected {want}")
                total_written += got_sub
                pos = sub_end
                break
            except Exception:
                if attempt + 1 >= tries:
                    raise
                time.sleep(min(15.0, 1.5 * (2 ** attempt)))
    return total_written


def _probe_remote_cram_size(url: str) -> int | None:
    from . import streamer as ST
    ss = ST._session()
    try:
        rh = ss.head(url, allow_redirects=True, timeout=30)
        if 200 <= rh.status_code < 300 and rh.headers.get("Content-Length"):
            return int(rh.headers["Content-Length"])
    except Exception:
        pass
    try:
        with ss.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=30, stream=True) as rg:
            if rg.status_code == 206:
                cr = rg.headers.get("Content-Range", "")
                if "/" in cr:
                    tot_s = cr.rsplit("/", 1)[-1].strip()
                    if tot_s.isdigit():
                        return int(tot_s)
    except Exception:
        pass
    return None


def _stream_aln_via_pysam(
    acc: PagedContigAccumulator,
    aln_path: Path,
    ref_fasta: Path,
    tmp_dir: Path,
    target_regions: Sequence[tuple[str, int, int]] | None = None,
) -> None:
    """Fallback when `samtools` CLI is absent: collate via pysam and stream SAM records into `acc`."""
    import pysam

    coll_bam = tmp_dir / "collated.bam"
    pysam.collate("-u", "-o", str(coll_bam), "--reference", str(ref_fasta), str(aln_path))
    iv_by_ctg: dict[str, list[tuple[int, int]]] = {}
    if target_regions:
        for cn, s0, e0 in target_regions:
            iv_by_ctg.setdefault(cn, []).append((int(s0), int(e0)))
    with pysam.AlignmentFile(str(coll_bam), "rb") as af:
        hdr_bytes = str(af.header).encode("ascii")
        acc.feed_bytes(hdr_bytes)
        buf: list[str] = []
        for seg in af.fetch(until_eof=True):
            if iv_by_ctg and not seg.is_unmapped:
                rn = seg.reference_name
                ivs = iv_by_ctg.get(rn)
                if not ivs:
                    continue
                rs, re_ = seg.reference_start, seg.reference_end or (seg.reference_start + 1)
                if not any(a < re_ and b > rs for a, b in ivs):
                    continue
            buf.append(seg.to_string() + "\n")
            if len(buf) >= 4096:
                acc.feed_bytes("".join(buf).encode("ascii"))
                buf.clear()
        if buf:
            acc.feed_bytes("".join(buf).encode("ascii"))


def run_varloop_crai_focused(
    g: G.Genome,
    ref_fasta: str | Path,
    cram_path_or_url: str | Path,
    crai_path: str | Path,
    regions_bed: Sequence[tuple[str, int, int]],
    out_vcf_gz: str | Path,
    params: DiploidParams | None = None,
    sample: str = "HG002",
    pad_bp: int = 1000,
    zoom_assisted: bool = True,
    clip_regions: Sequence[tuple[str, int, int]] | None = None,
    callable_npz_out: str | Path | None = None,
) -> dict:
    """Slice exact CRAM containers via `.crai` (`plan_ranges`), feed directly into `PagedContigAccumulator`, and call.

    NOTE: Requires a pre-aligned indexed CRAM + `.crai`; upstream read alignment and CRAM encoding are excluded.
    """
    import shlex
    import tempfile
    import threading

    p = params or DiploidParams()
    rss_mon = ProcTreeRssMonitor().start()
    t0 = time.time()
    cpu0 = _cpu_times()
    cidx = {n: i for i, n in enumerate(g.names)}
    if not regions_bed:
        raise ValueError("run_varloop_crai_focused requires at least one target region")
    unknown_bed = [cn for cn, _, _ in regions_bed if cn not in cidx]
    if len(unknown_bed) == len(regions_bed):
        raise ValueError(
            f"Target BED contigs {list(dict.fromkeys(unknown_bed))[:5]} do not match reference FASTA {list(g.names[:5])}"
        )

    crai_arr = FC.parse_crai(crai_path)
    if len(crai_arr) == 0:
        raise ValueError(f"Empty CRAI index: {Path(crai_path).name}")
    hdr_end = int(np.unique(crai_arr[:, 3])[0])
    if hdr_end <= 0:
        raise ValueError(f"Invalid CRAM header offset {hdr_end} in CRAI index")

    is_local_cram = not str(cram_path_or_url).startswith(("http://", "https://", "ftp://")) and Path(str(cram_path_or_url)).exists()
    if is_local_cram:
        local_cp = Path(cram_path_or_url)
        cram_size = local_cp.stat().st_size
        if cram_size < hdr_end:
            raise RuntimeError(f"Truncated CRAM file {local_cp.name!r}: size {cram_size} < header end {hdr_end}")
        with open(local_cp, "rb") as fc_hdr:
            header_bytes = fc_hdr.read(hdr_end)
        if len(header_bytes) != hdr_end:
            raise RuntimeError(f"Short read on CRAM header of {local_cp.name!r}: got {len(header_bytes)} != {hdr_end}")
    else:
        cram_size = _probe_remote_cram_size(str(cram_path_or_url))
        hdr_chunks: list[bytes] = []
        _fetch_remote_range_chunked(str(cram_path_or_url), 0, hdr_end, hdr_chunks.append)
        header_bytes = b"".join(hdr_chunks)

    Path(out_vcf_gz).parent.mkdir(parents=True, exist_ok=True)
    tmp_coll_dir = Path(tempfile.mkdtemp(prefix="vl_crai_", dir=str(Path(out_vcf_gz).parent)))
    try:
        # Map .crai seq_id via CRAM @SQ order rather than FASTA contig order (#22, #24)
        req_ctgs = [cn for cn, _, _ in regions_bed if cn in cidx]
        sq_list = _extract_header_sq_from_cram_bytes(header_bytes, Path(ref_fasta), tmp_coll_dir)
        sq_idx = _validate_sq_against_genome(g, sq_list, required_contigs=req_ctgs, label="CRAM")

        regs_fc = [
            (sq_idx[cn], max(1, int(s0) + 1 - pad_bp), min(int(g.lengths[cidx[cn]]), int(e0) + pad_bp), cn)
            for cn, s0, e0 in regions_bed if cn in cidx and cn in sq_idx
        ]
        hdr_end_plan, byte_ranges, n_cont, _ = FC.plan_ranges(crai_arr, regs_fc, read_pad=200, cram_size=cram_size)
        assert hdr_end_plan == hdr_end
        expected_sliced = hdr_end + sum(b - a for a, b in byte_ranges) + len(C_CRAM_EOF)

        padded_targets = [
            (cn, max(0, int(s0) - pad_bp), min(int(g.lengths[cidx[cn]]), int(e0) + pad_bp))
            for cn, s0, e0 in regions_bed if cn in cidx
        ]
        zoom_arg = padded_targets if zoom_assisted else None
        ref_q = shlex.quote(str(ref_fasta))
        coll_q = shlex.quote(str(tmp_coll_dir / "coll"))
        crai_cmd = f"samtools collate -u -O -r 50000 --reference {ref_q} - {coll_q} | samtools view -h -"

        with PagedContigAccumulator(g, p, target_regions=padded_targets, zoom_regions=zoom_arg) as acc:
            if shutil.which("samtools"):
                with tempfile.TemporaryFile(dir=str(tmp_coll_dir)) as crai_err_f:
                    view_proc = subprocess.Popen(
                        ["bash", "-o", "pipefail", "-c", crai_cmd],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=crai_err_f, bufsize=0,
                    )
                    feeder_exc: list[BaseException] = []
                    bytes_fed = [0]

                    def _write_stdin(chunk: bytes) -> None:
                        assert view_proc.stdin is not None
                        view_proc.stdin.write(chunk)
                        bytes_fed[0] += len(chunk)

                    def _feed_cram():
                        try:
                            _write_stdin(header_bytes)
                            if is_local_cram:
                                with open(cram_path_or_url, "rb") as fc:
                                    for a, b in byte_ranges:
                                        fc.seek(a)
                                        rem = b - a
                                        while rem > 0:
                                            blk = fc.read(min(4 << 20, rem))
                                            if not blk:
                                                raise RuntimeError(
                                                    f"Truncated local CRAM {Path(cram_path_or_url).name!r} at offset {b - rem} (expected {b})"
                                                )
                                            _write_stdin(blk)
                                            rem -= len(blk)
                            else:
                                for a, b in byte_ranges:
                                    _fetch_remote_range_chunked(str(cram_path_or_url), a, b, _write_stdin)
                            _write_stdin(C_CRAM_EOF)
                        except BaseException as exc:
                            feeder_exc.append(exc)
                        finally:
                            if view_proc.stdin:
                                try:
                                    view_proc.stdin.close()
                                except Exception:
                                    pass

                    th = threading.Thread(target=_feed_cram, daemon=True)
                    th.start()
                    try:
                        acc.consume_sam_fd(view_proc.stdout.fileno())
                    finally:
                        if view_proc.stdout:
                            view_proc.stdout.close()
                    th.join(timeout=120)
                    rc = view_proc.wait()
                    crai_err_f.seek(0)
                    err_s = crai_err_f.read().decode("utf-8", errors="replace")
                    if feeder_exc:
                        raise RuntimeError(
                            f"CRAM slice feeder failed for {sample!r}: {type(feeder_exc[0]).__name__}: {feeder_exc[0]}"
                        ) from feeder_exc[0]
                    if bytes_fed[0] != expected_sliced:
                        raise RuntimeError(
                            f"CRAM slice byte count mismatch: fed {bytes_fed[0]} != expected {expected_sliced}"
                        )
                    if rc != 0 or "EOF marker is absent" in err_s or "truncated file" in err_s.lower():
                        raise RuntimeError(f"samtools view on sliced CRAM failed (rc={rc}): {err_s[-400:]}")
            else:
                staged_cram = tmp_coll_dir / "sliced.cram"
                written = 0
                with open(staged_cram, "wb") as fo:
                    fo.write(header_bytes)
                    written += len(header_bytes)
                    if is_local_cram:
                        with open(cram_path_or_url, "rb") as fc:
                            for a, b in byte_ranges:
                                fc.seek(a)
                                rem = b - a
                                while rem > 0:
                                    blk = fc.read(min(4 << 20, rem))
                                    if not blk:
                                        raise RuntimeError(
                                            f"Truncated local CRAM {Path(cram_path_or_url).name!r} at offset {b - rem}"
                                        )
                                    fo.write(blk)
                                    written += len(blk)
                                    rem -= len(blk)
                    else:
                        for a, b in byte_ranges:
                            written += _fetch_remote_range_chunked(str(cram_path_or_url), a, b, fo.write)
                    fo.write(C_CRAM_EOF)
                    written += len(C_CRAM_EOF)
                if written != expected_sliced:
                    raise RuntimeError(f"Staged CRAM slice size mismatch: {written} != {expected_sliced}")
                _stream_aln_via_pysam(acc, staged_cram, Path(ref_fasta), tmp_coll_dir, target_regions=padded_targets)

            summ = acc.summary()
            sam_recs = int(summ["stats"][0])
            reads_cnt = int(summ["stats"][2])
            if n_cont > 0 and sam_recs == 0:
                raise RuntimeError(
                    f"Sample {sample!r}: focused CRAM decode yielded 0 SAM records across {n_cont} planned containers"
                )
            if callable_npz_out is not None:
                acc.save_callable_npz(callable_npz_out, min_dp=1)
            med_dp = _estimate_median_depth(g, summ, acc._target_bin_mask)
            acc.sort_all_reads()
            want_cids = sorted({cidx[cn] for cn, _, _ in padded_targets})
            calls_by_ctg = {cid: call_contig_diploid(g, acc, cid, p, free_after=True) for cid in want_cids}
            eff_clip = clip_regions if clip_regions is not None else regions_bed
            calls_by_ctg = _clip_calls_to_regions(calls_by_ctg, eff_clip, cidx)
    finally:
        shutil.rmtree(tmp_coll_dir, ignore_errors=True)

    n_vcf = write_diploid_vcf(g, calls_by_ctg, out_vcf_gz, sample=sample, contig_ids=want_cids)
    n_snv = sum(
        1
        for calls in calls_by_ctg.values()
        for c in calls
        if len(str(c["ref"])) == 1 and all(len(a) == 1 for a in str(c["alt"]).split(","))
    )
    n_indel = n_vcf - n_snv
    n_omnitig = sum(
        1 for calls in calls_by_ctg.values() for c in calls if str(c.get("source", "")).startswith("omnitig_asm")
    )
    cpu1 = _cpu_times()
    peak_tree_rss, _ = rss_mon.stop()
    prim_mapped = int(summ["stats"][1])
    pcr_dups = int(summ["stats"][10])
    return {
        "sample": sample,
        "mode": "crai_focused (requires pre-aligned CRAM)",
        "requires_prealigned_cram": True,
        "zoom_assisted": zoom_assisted,
        "note": "Requires pre-aligned indexed CRAM + .crai; upstream read alignment and CRAM encoding times are excluded.",
        "cram_total_bytes": cram_size,
        "crai_sliced_bytes": int(expected_sliced),
        "crai_wire_fraction": round(expected_sliced / cram_size, 6) if cram_size else None,
        "n_containers": int(n_cont),
        "n_coalesced_ranges": len(byte_ranges),
        "sam_records": sam_recs,
        "primary_mapped": prim_mapped,
        "reads_counted": reads_cnt,
        "bases_counted": int(summ["stats"][3]),
        "median_depth": med_dp,
        "pcr_dups_skipped": pcr_dups,
        "pcr_dup_fraction": round(float(pcr_dups) / max(prim_mapped, 1), 6),
        "malformed_sam_rejected": int(summ["stats"][11]),
        "acc_total_bytes": summ["total_acc_bytes"],
        "peak_proctree_rss_bytes": peak_tree_rss,
        "n_variants": n_vcf,
        "n_snv": n_snv,
        "n_indel": n_indel,
        "n_omnitig_assembled": n_omnitig,
        "wall_s": round(time.time() - t0, 3),
        "total_wall_s": round(time.time() - t0, 3),
        "total_cpu_s": round((cpu1[0] - cpu0[0]) + (cpu1[1] - cpu0[1]), 3),
        "vcf_path": str(out_vcf_gz),
    }


C_CRAM_EOF = bytes.fromhex("0f000000ffffffff0fe0454f4600000000010005bdd94f0001000606010001000100ee63014b")


def run_varloop_bam_cram(
    g: G.Genome,
    ref_fasta: str | Path,
    bam_or_cram: str | Path,
    out_vcf_gz: str | Path,
    params: DiploidParams | None = None,
    target_regions: Sequence[tuple[str, int, int]] | None = None,
    zoom_regions: Sequence[tuple[str, int, int]] | None = None,
    sample: str = "SAMPLE",
    threads: int = 4,
    call_workers: int = 1,
    targets_bed: str | Path | None = None,
    clip_regions: Sequence[tuple[str, int, int]] | None = None,
    callable_npz_out: str | Path | None = None,
) -> dict:
    """Stream a BAM/CRAM alignment file through `samtools collate` into `PagedContigAccumulator` and call."""
    import shlex
    import tempfile

    p = params or DiploidParams()
    rss_mon = ProcTreeRssMonitor().start()
    t0 = time.time()
    cpu0 = _cpu_times()

    out_vcf_p = Path(out_vcf_gz)
    out_vcf_p.parent.mkdir(parents=True, exist_ok=True)
    cidx = {n: i for i, n in enumerate(g.names)}
    if targets_bed and target_regions is None:
        parsed_regs: list[tuple[str, int, int]] = []
        raw_bed_rows = 0
        with open(targets_bed, encoding="utf-8-sig") as bf:
            for ln in bf:
                if not ln.strip() or ln.startswith(("#", "track", "browser")):
                    continue
                parts = ln.split()
                if len(parts) >= 3:
                    raw_bed_rows += 1
                    if parts[0] in cidx:
                        parsed_regs.append((parts[0], max(0, int(parts[1])), min(int(g.lengths[cidx[parts[0]]]), int(parts[2]))))
        if raw_bed_rows > 0 and not parsed_regs:
            raise ValueError(f"Target BED {Path(targets_bed).name!r} matched 0 contigs in reference FASTA")
        target_regions = parsed_regs or None

    req_ctgs = [cn for cn, _, _ in target_regions] if target_regions else None
    _validate_local_aln_integrity(g, Path(bam_or_cram), Path(ref_fasta), req_contigs=req_ctgs)

    tmp_coll_dir = Path(tempfile.mkdtemp(prefix="vl_bam_", dir=str(out_vcf_p.parent)))
    t_q = max(1, int(threads))
    ref_q = shlex.quote(str(ref_fasta))
    in_q = shlex.quote(str(bam_or_cram))
    coll_q = shlex.quote(str(tmp_coll_dir / "coll"))
    bed_arg = f"-L {shlex.quote(str(targets_bed))} " if targets_bed else ""
    if bed_arg:
        bam_cmd = (
            f"samtools view -@ {t_q} -u --reference {ref_q} {bed_arg}{in_q} | "
            f"samtools collate -u -O -r 50000 --reference {ref_q} - {coll_q} | "
            f"samtools view -h -"
        )
    else:
        bam_cmd = (
            f"samtools collate -@ {t_q} -u -O -r 50000 --reference {ref_q} {in_q} {coll_q} | "
            f"samtools view -h -"
        )

    t_stream0 = time.time()
    try:
        with PagedContigAccumulator(g, p, target_regions=target_regions, zoom_regions=zoom_regions) as acc:
            if shutil.which("samtools"):
                with tempfile.TemporaryFile(dir=str(tmp_coll_dir)) as bam_err_f:
                    view_proc = subprocess.Popen(
                        ["bash", "-o", "pipefail", "-c", bam_cmd],
                        stdout=subprocess.PIPE, stderr=bam_err_f, bufsize=0,
                    )
                    try:
                        acc.consume_sam_fd(view_proc.stdout.fileno())
                    finally:
                        if view_proc.stdout:
                            view_proc.stdout.close()
                    rc = view_proc.wait()
                    bam_err_f.seek(0)
                    err_s = bam_err_f.read().decode("utf-8", errors="replace")
                    if rc != 0 or "EOF marker is absent" in err_s or "truncated file" in err_s.lower():
                        raise RuntimeError(f"samtools collate/view failed (rc={rc}): {err_s[-500:]}")
            else:
                _stream_aln_via_pysam(acc, Path(bam_or_cram), Path(ref_fasta), tmp_coll_dir, target_regions=target_regions)

            t_stream = time.time() - t_stream0
            summ = acc.summary()
            sam_recs = int(summ["stats"][0])
            reads_cnt = int(summ["stats"][2])
            malf_cnt = int(summ["stats"][11])
            if sam_recs <= 0 or reads_cnt <= 0:
                raise RuntimeError(
                    f"Sample {sample!r}: zero usable alignments counted from {Path(bam_or_cram).name!r} "
                    f"(sam_records={sam_recs}, reads_counted={reads_cnt}, malformed_sam={malf_cnt})"
                )
            if malf_cnt > 0.5 * max(sam_recs, 1):
                raise RuntimeError(
                    f"Sample {sample!r}: majority of SAM records rejected as malformed ({malf_cnt}/{sam_recs})"
                )
            if callable_npz_out is not None:
                acc.save_callable_npz(callable_npz_out, min_dp=1)
            med_dp = _estimate_median_depth(g, summ, acc._target_bin_mask)
            acc.sort_all_reads()
            t_call0 = time.time()
            if target_regions is not None:
                want_cids = sorted({cidx[cn] for cn, _, _ in target_regions if cn in cidx})
            else:
                want_cids = [cid for cid in range(len(g.names)) if summ["ccount"][cid] > 0]
            eff_call_workers = max(1, min(int(call_workers), 4))
            calls_by_ctg, _ = call_contig_diploid_batch(
                g, acc, want_cids, p,
                call_workers=eff_call_workers,
                return_pileup_also=False,
                weights=summ["ccount"],
            )
            eff_clip = clip_regions if clip_regions is not None else target_regions
            if eff_clip is not None:
                calls_by_ctg = _clip_calls_to_regions(calls_by_ctg, eff_clip, cidx)
            n_asm = sum(1 for recs in calls_by_ctg.values() for r in recs if str(r.get("source", "")).startswith("omnitig_asm"))
            t_call = time.time() - t_call0
    finally:
        shutil.rmtree(tmp_coll_dir, ignore_errors=True)

    n_vcf = write_diploid_vcf(g, calls_by_ctg, out_vcf_p, sample=sample, contig_ids=want_cids)
    cpu1 = _cpu_times()
    peak_tree_rss, _ = rss_mon.stop()
    n_snv = sum(
        1 for recs in calls_by_ctg.values() for r in recs
        if len(r["ref"]) == 1 and all(len(a) == 1 for a in r["alt"].split(","))
    )
    in_size = Path(bam_or_cram).stat().st_size if Path(str(bam_or_cram)).exists() else 0
    prim_mapped = int(summ["stats"][1])
    pcr_dups = int(summ["stats"][10])
    return {
        "sample": sample,
        "mapper": "prealigned_bam_cram",
        "order_mode": summ["order_mode"],
        "input_bytes": in_size,
        "sam_records": int(summ["stats"][0]),
        "primary_mapped": prim_mapped,
        "reads_counted": int(summ["stats"][2]),
        "bases_counted": int(summ["stats"][3]),
        "median_depth": med_dp,
        "pcr_dups_skipped": pcr_dups,
        "pcr_dup_fraction": round(float(pcr_dups) / max(prim_mapped, 1), 6),
        "malformed_sam_rejected": int(summ["stats"][11]),
        "informative_reads_captured": int(summ["stats"][8]),
        "acc_total_bytes": summ["total_acc_bytes"],
        "n_variants": n_vcf,
        "n_snv": n_snv,
        "n_indel": n_vcf - n_snv,
        "n_omnitig_assembled": n_asm,
        "stream_wall_s": round(t_stream, 3),
        "call_wall_s": round(t_call, 3),
        "total_wall_s": round(time.time() - t0, 3),
        "cpu_self_s": round(cpu1[0] - cpu0[0], 3),
        "cpu_children_s": round(cpu1[1] - cpu0[1], 3),
        "total_cpu_s": round((cpu1[0] - cpu0[0]) + (cpu1[1] - cpu0[1]), 3),
        "peak_proctree_rss_bytes": peak_tree_rss,
        "vcf_path": str(out_vcf_p),
    }


# ============================================================================
# 6. `rtg vcfeval` Haplotype Evaluation Helper (`--decompose` + `--atomize`)
# ============================================================================

def build_phase3_heldout_autosome_bed(
    v421_bed: str | Path,
    cmrg_genes_bed: str | Path,
    cmrg_eval_bed: str | Path,
    out_bed: str | Path,
    pad_bp: int = 5000,
) -> dict:
    """Build strict held-out autosome BED (`chr1..19, chr21..22` minus `chr20` and all 273 CMRG loci +- `pad_bp`)."""
    held_contigs = {f"chr{i}" for i in range(1, 23) if i != 20}
    all_auto_contigs = {f"chr{i}" for i in range(1, 23)}

    excl_by_ctg: dict[str, list[tuple[int, int]]] = {}
    for bed_p in (cmrg_genes_bed, cmrg_eval_bed):
        with open(bed_p) as f:
            for ln in f:
                if not ln.strip() or ln.startswith("#"):
                    continue
                parts = ln.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[0] in held_contigs:
                    s0 = max(0, int(parts[1]) - pad_bp)
                    e0 = int(parts[2]) + pad_bp
                    if e0 > s0:
                        excl_by_ctg.setdefault(parts[0], []).append((s0, e0))

    for cn, ivs in list(excl_by_ctg.items()):
        ivs.sort()
        merged: list[tuple[int, int]] = []
        for s0, e0 in ivs:
            if merged and s0 <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e0))
            else:
                merged.append((s0, e0))
        excl_by_ctg[cn] = merged

    total_auto_bp = 0
    heldout_bp = 0
    n_intervals = 0
    out_p = Path(out_bed)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(v421_bed) as fin, open(out_p, "w") as fout:
        for ln in fin:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            cn, s0, e0 = parts[0], int(parts[1]), int(parts[2])
            if e0 <= s0 or cn not in all_auto_contigs:
                continue
            total_auto_bp += e0 - s0
            if cn not in held_contigs:
                continue
            ex_list = excl_by_ctg.get(cn, [])
            if not ex_list:
                fout.write(f"{cn}\t{s0}\t{e0}\n")
                heldout_bp += e0 - s0
                n_intervals += 1
                continue
            # Subtract sorted merged exclusion intervals from [s0, e0)
            cur = s0
            idx = bisect_left_end(ex_list, s0)
            while idx < len(ex_list) and ex_list[idx][0] < e0:
                ex_s, ex_e = ex_list[idx]
                if ex_s > cur:
                    fout.write(f"{cn}\t{cur}\t{min(e0, ex_s)}\n")
                    heldout_bp += min(e0, ex_s) - cur
                    n_intervals += 1
                cur = max(cur, ex_e)
                if cur >= e0:
                    break
                idx += 1
            if cur < e0:
                fout.write(f"{cn}\t{cur}\t{e0}\n")
                heldout_bp += e0 - cur
                n_intervals += 1

    return {
        "out_bed": str(out_p),
        "total_v421_autosome_bp": total_auto_bp,
        "heldout_autosome_bp": heldout_bp,
        "heldout_fraction": round(heldout_bp / max(total_auto_bp, 1), 6),
        "n_intervals": n_intervals,
        "excluded_contigs": ["chr20"],
        "excluded_cmrg_pad_bp": pad_bp,
    }


def bisect_left_end(ivs: list[tuple[int, int]], pos0: int) -> int:
    lo, hi = 0, len(ivs)
    while lo < hi:
        mid = (lo + hi) // 2
        if ivs[mid][1] <= pos0:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _atomize_vcf_for_eval(in_vcf_gz: Path, ref_fasta: Path, out_vcf_gz: Path) -> Path:
    """Normalize & atomize MNPs/complex alleles (`bcftools norm --atomize -f ref`) for unbiased SNP/Indel strata."""
    import shlex
    out_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    ref_q = shlex.quote(str(ref_fasta))
    in_q = shlex.quote(str(in_vcf_gz))
    out_q = shlex.quote(str(out_vcf_gz))
    cmd = f"bcftools norm --atomize -f {ref_q} -Oz -o {out_q} {in_q} && tabix -f -p vcf {out_q}"
    res = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"bcftools norm --atomize failed ({res.returncode}): {res.stderr[-600:]}")
    return out_vcf_gz


def run_rtg_vcfeval(
    calls_vcf_gz: str | Path,
    truth_vcf_gz: str | Path,
    eval_bed: str | Path,
    sdf_dir: str | Path,
    out_dir: str | Path,
    threads: int = 16,
    squash_ploidy: bool = False,
    decompose: bool = True,
    ref_fasta: str | Path | None = None,
) -> dict:
    """Execute atomized + decomposed `rtg vcfeval` so MNPs/complex alleles never skew SNP vs Indel strata."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    b_vcf = Path(truth_vcf_gz)
    c_vcf = Path(calls_vcf_gz)
    norm_dir = out_dir.parent / "_atomized_cache"
    if decompose and ref_fasta is not None:
        ref_p = Path(ref_fasta)
        b_tag = f"{b_vcf.stem}_{b_vcf.stat().st_size}.atom.vcf.gz"
        b_atom = norm_dir / b_tag
        if not (b_atom.exists() and Path(str(b_atom) + ".tbi").exists()):
            _atomize_vcf_for_eval(b_vcf, ref_p, b_atom)
        c_atom = norm_dir / f"{out_dir.name}_calls.atom.vcf.gz"
        _atomize_vcf_for_eval(c_vcf, ref_p, c_atom)
        b_vcf = b_atom
        c_vcf = c_atom

    cmd = [
        "rtg", "vcfeval",
        "-b", str(b_vcf),
        "-c", str(c_vcf),
        "-e", str(eval_bed),
        "-t", str(sdf_dir),
        "-o", str(out_dir),
        "-T", str(threads),
    ]
    if decompose:
        cmd.append("--decompose")
    if squash_ploidy:
        cmd.append("--squash-ploidy")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"rtg vcfeval failed ({res.returncode}): {res.stderr[-800:]}")

    def _parse_summary(txt: str) -> dict:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("-")]
        best_row = None
        none_row = None
        for ln in lines[1:]:
            cols = ln.split()
            if len(cols) >= 8:
                row = {
                    "threshold": cols[0],
                    "tp_baseline": int(cols[1]),
                    "tp_call": int(cols[2]),
                    "fp": int(cols[3]),
                    "fn": int(cols[4]),
                    "precision": float(cols[5]),
                    "recall": float(cols[6]),
                    "f1": float(cols[7]),
                }
                if cols[0] == "None":
                    none_row = row
                else:
                    if best_row is None or row["f1"] > best_row["f1"]:
                        best_row = row
        return {"all_calls": none_row or best_row, "best_f1_threshold": best_row}

    overall = _parse_summary((out_dir / "summary.txt").read_text())

    strata = {}
    for vtype, expr in (("snp", "TYPE='snp'"), ("indel", "TYPE!='snp'")):
        tp_b = int(subprocess.run(
            ["bcftools", "view", "-H", "-i", expr, str(out_dir / "tp-baseline.vcf.gz")],
            capture_output=True, text=True, check=True,
        ).stdout.count("\n"))
        tp_c = int(subprocess.run(
            ["bcftools", "view", "-H", "-i", expr, str(out_dir / "tp.vcf.gz")],
            capture_output=True, text=True, check=True,
        ).stdout.count("\n"))
        fp_c = int(subprocess.run(
            ["bcftools", "view", "-H", "-i", expr, str(out_dir / "fp.vcf.gz")],
            capture_output=True, text=True, check=True,
        ).stdout.count("\n"))
        fn_b = int(subprocess.run(
            ["bcftools", "view", "-H", "-i", expr, str(out_dir / "fn.vcf.gz")],
            capture_output=True, text=True, check=True,
        ).stdout.count("\n"))
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        strata[vtype] = {
            "tp_baseline": tp_b,
            "tp_call": tp_c,
            "fp": fp_c,
            "fn": fn_b,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
        }

    return {
        "decomposed": bool(decompose),
        "atomized": bool(decompose and ref_fasta is not None),
        "overall": overall,
        "snp": strata["snp"],
        "indel": strata["indel"],
    }
