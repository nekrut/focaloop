"""Python ctypes wrapper for libpla_sieve.dylib (splitmix64-Permuted mmap PLA-index)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from .native._loader import load_shared_lib

HERE = Path(__file__).resolve().parent
PLA_FLAG_PERMUTED = 1
PLA_FLAG_HAS_FLAGS2B = 2
PLA_FLAG_SPLITMIX_SYNCMER = 4


class _CSeg(Structure):
    _fields_ = [
        ("x_start", c_uint64),
        ("intercept", c_uint32),
        ("slope", c_float),
    ]


class _CIndex(Structure):
    _fields_ = [
        ("eps", c_uint32),
        ("k", c_uint32),
        ("s", c_uint32),
        ("flags", c_uint32),
        ("n_keys", c_size_t),
        ("n_segs", c_size_t),
        ("radix16", POINTER(c_uint32)),
        ("segs", POINTER(_CSeg)),
        ("perm_keys", POINTER(c_uint64)),
        ("flags2b", POINTER(c_uint8)),
        ("backing", c_void_p),
        ("backing_bytes", c_size_t),
        ("is_mmap", c_int),
        ("fd", c_int),
    ]


class _CStats(Structure):
    _fields_ = [
        ("n_keys", c_uint64),
        ("n_segs", c_uint64),
        ("eps", c_uint32),
        ("use_perm", c_uint32),
        ("empty_buckets", c_uint64),
        ("max_keys_per_bucket", c_uint64),
        ("max_segs_per_bucket", c_uint64),
        ("mean_keys_per_seg", c_double),
        ("index_overhead_bytes", c_uint64),
        ("keys_bytes", c_uint64),
        ("total_bytes", c_uint64),
        ("overhead_bytes_per_key", c_double),
        ("total_bytes_per_key", c_double),
        ("max_abs_err", c_int64),
    ]


class _CHtab(Structure):
    _fields_ = [
        ("keys", POINTER(c_uint64)),
        ("vals", POINTER(c_uint32)),
        ("cap", c_uint64),
        ("mask", c_uint64),
        ("n", c_uint64),
    ]


class _CSieveStats(Structure):
    _fields_ = [
        ("pairs_in", c_uint64),
        ("pairs_out", c_uint64),
        ("drop_locus", c_uint64),
        ("drop_bait", c_uint64),
    ]


def _load_lib() -> ctypes.CDLL:
    lib = load_shared_lib("libpla_sieve", ["pla_sieve.c"], ["pla_sieve.h"])

    lib.pla_sort_dedup_u64.argtypes = [POINTER(c_uint64), c_size_t, c_int]
    lib.pla_sort_dedup_u64.restype = c_size_t

    lib.pla_build.argtypes = [
        POINTER(_CIndex),
        POINTER(c_uint64),
        POINTER(c_uint8),
        c_size_t,
        c_uint32,
        c_int,
        c_uint32,
        c_uint32,
        c_int,
    ]
    lib.pla_build.restype = c_int

    lib.pla_verify_all.argtypes = [POINTER(_CIndex), POINTER(c_int64), c_int]
    lib.pla_verify_all.restype = c_int

    lib.pla_compute_stats.argtypes = [POINTER(_CIndex), POINTER(_CStats)]
    lib.pla_compute_stats.restype = None

    lib.pla_save_mmap.argtypes = [POINTER(_CIndex), c_char_p]
    lib.pla_save_mmap.restype = c_int

    lib.pla_open_mmap.argtypes = [POINTER(_CIndex), c_char_p, c_int, c_int]
    lib.pla_open_mmap.restype = c_int

    lib.pla_close.argtypes = [POINTER(_CIndex)]
    lib.pla_close.restype = None

    lib.pla_lookup.argtypes = [POINTER(_CIndex), c_uint64]
    lib.pla_lookup.restype = c_int64

    lib.pla_lookup_batch.argtypes = [
        POINTER(_CIndex),
        POINTER(c_uint64),
        POINTER(c_int64),
        c_size_t,
        c_int,
    ]
    lib.pla_lookup_batch.restype = None

    lib.pla_count_hits_batch.argtypes = [POINTER(_CIndex), POINTER(c_uint64), c_size_t, c_int]
    lib.pla_count_hits_batch.restype = c_uint64

    lib.pla_splitmix64_forward_batch.argtypes = [POINTER(c_uint64), POINTER(c_uint64), c_size_t]
    lib.pla_splitmix64_forward_batch.restype = None

    lib.pla_splitmix64_inverse_batch.argtypes = [POINTER(c_uint64), POINTER(c_uint64), c_size_t]
    lib.pla_splitmix64_inverse_batch.restype = None

    lib.pla_bsearch_lookup.argtypes = [POINTER(_CIndex), c_uint64]
    lib.pla_bsearch_lookup.restype = c_int64

    lib.pla_bsearch_count_hits_batch.argtypes = [POINTER(_CIndex), POINTER(c_uint64), c_size_t, c_int]
    lib.pla_bsearch_count_hits_batch.restype = c_uint64

    lib.gl_ht_init.argtypes = [POINTER(_CHtab), c_uint64]
    lib.gl_ht_init.restype = None
    lib.gl_ht_free.argtypes = [POINTER(_CHtab)]
    lib.gl_ht_free.restype = None
    lib.gl_ht_insert.argtypes = [POINTER(_CHtab), c_uint64, c_uint32]
    lib.gl_ht_insert.restype = c_uint64
    lib.gl_ht_find.argtypes = [POINTER(_CHtab), c_uint64]
    lib.gl_ht_find.restype = c_uint64
    lib.gl_ht_save_table.argtypes = [POINTER(_CHtab), c_char_p, c_int, c_int]
    lib.gl_ht_save_table.restype = c_int
    lib.gl_ht_load_table.argtypes = [c_char_p, POINTER(_CHtab), POINTER(c_int), POINTER(c_int)]
    lib.gl_ht_load_table.restype = c_int

    lib.gl_extract_open_syncmers.argtypes = [
        c_char_p,
        c_size_t,
        c_int,
        c_int,
        c_int,
        POINTER(c_uint64),
        c_size_t,
    ]
    lib.gl_extract_open_syncmers.restype = c_size_t

    lib.pla_extract_open_syncmers.argtypes = [
        c_char_p,
        c_size_t,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_uint64),
        c_size_t,
    ]
    lib.pla_extract_open_syncmers.restype = c_size_t

    lib.pla_read_pf_hits.argtypes = [
        POINTER(_CIndex),
        POINTER(c_uint8),
        c_size_t,
        c_size_t,
        c_int,
    ]
    lib.pla_read_pf_hits.restype = c_int

    lib.pla_fastq_block.argtypes = [
        POINTER(c_uint8),
        c_int64,
        POINTER(c_uint8),
        c_int64,
        c_uint64,
        c_int,
        POINTER(_CIndex),
        c_int,
        POINTER(c_int64),
        c_size_t,
        POINTER(c_int64),
        POINTER(c_uint8),
        c_int64,
        c_int64,
        c_int64,
        POINTER(c_int64),
        POINTER(c_int64),
    ]
    lib.pla_fastq_block.restype = c_int64

    lib.gl_sieve_pairs_pla.argtypes = [
        POINTER(_CIndex),
        POINTER(c_char_p),
        POINTER(c_size_t),
        POINTER(c_char_p),
        POINTER(c_size_t),
        c_size_t,
        c_int,
        c_int,
        POINTER(c_uint8),
        POINTER(_CSieveStats),
        c_int,
    ]
    lib.gl_sieve_pairs_pla.restype = None

    lib.gl_sieve_pairs_htab.argtypes = [
        POINTER(_CHtab),
        c_int,
        c_int,
        POINTER(c_char_p),
        POINTER(c_size_t),
        POINTER(c_char_p),
        POINTER(c_size_t),
        c_size_t,
        c_int,
        c_int,
        POINTER(c_uint8),
        POINTER(_CSieveStats),
        c_int,
    ]
    lib.gl_sieve_pairs_htab.restype = None

    lib.pla_generate_distinct_syncmers.argtypes = [
        POINTER(c_uint64),
        c_size_t,
        c_int,
        c_int,
        c_int,
        c_uint64,
        c_int,
    ]
    lib.pla_generate_distinct_syncmers.restype = c_int

    return lib


LIB = _load_lib()


def splitmix64_bijective(keys: np.ndarray | int) -> np.ndarray | int:
    """Apply invertible splitmix64 permutation over uint64 keys via C library."""
    if isinstance(keys, int):
        arr = np.array([keys], dtype=np.uint64)
        out = np.empty(1, dtype=np.uint64)
        LIB.pla_splitmix64_forward_batch(
            arr.ctypes.data_as(POINTER(c_uint64)),
            out.ctypes.data_as(POINTER(c_uint64)),
            1,
        )
        return int(out[0])
    arr = np.ascontiguousarray(keys, dtype=np.uint64)
    out = np.empty_like(arr)
    if arr.size > 0:
        LIB.pla_splitmix64_forward_batch(
            arr.ctypes.data_as(POINTER(c_uint64)),
            out.ctypes.data_as(POINTER(c_uint64)),
            arr.size,
        )
    return out


def splitmix64_inverse(keys: np.ndarray | int) -> np.ndarray | int:
    """Apply exact algebraic inverse of splitmix64 over uint64 keys via C library."""
    if isinstance(keys, int):
        arr = np.array([keys], dtype=np.uint64)
        out = np.empty(1, dtype=np.uint64)
        LIB.pla_splitmix64_inverse_batch(
            arr.ctypes.data_as(POINTER(c_uint64)),
            out.ctypes.data_as(POINTER(c_uint64)),
            1,
        )
        return int(out[0])
    arr = np.ascontiguousarray(keys, dtype=np.uint64)
    out = np.empty_like(arr)
    if arr.size > 0:
        LIB.pla_splitmix64_inverse_batch(
            arr.ctypes.data_as(POINTER(c_uint64)),
            out.ctypes.data_as(POINTER(c_uint64)),
            arr.size,
        )
    return out


@dataclass
class PLAStats:
    n_keys: int
    n_segs: int
    eps: int
    use_perm: bool
    empty_buckets: int
    max_keys_per_bucket: int
    max_segs_per_bucket: int
    mean_keys_per_seg: float
    index_overhead_bytes: int
    keys_bytes: int
    total_bytes: int
    overhead_bytes_per_key: float
    total_bytes_per_key: float
    max_abs_err: int


class PLAIndex:
    """Context-managed wrapper over C pla_index_t (heap or zero-deserialization mmap)."""

    def __init__(self, c_idx: _CIndex) -> None:
        self._c = c_idx
        self._closed = False

    @classmethod
    def build(
        cls,
        keys: np.ndarray,
        eps: int = 8,
        use_perm: bool = True,
        flags: np.ndarray | None = None,
        k: int = 31,
        s: int = 15,
        num_threads: int = 0,
        use_splitmix_syncmer: bool = False,
    ) -> PLAIndex:
        arr = np.ascontiguousarray(keys, dtype=np.uint64)
        k_ptr = arr.ctypes.data_as(POINTER(c_uint64)) if arr.size > 0 else None
        f_arr = None
        f_ptr = None
        if flags is not None:
            f_arr = np.ascontiguousarray(flags, dtype=np.uint8)
            if f_arr.size != arr.size:
                raise ValueError("flags length must match keys length")
            f_ptr = f_arr.ctypes.data_as(POINTER(c_uint8)) if f_arr.size > 0 else None

        c_idx = _CIndex()
        perm_mode = (1 if use_perm else 0) | (2 if use_splitmix_syncmer else 0)
        rc = LIB.pla_build(byref(c_idx), k_ptr, f_ptr, arr.size, eps, perm_mode, k, s, num_threads)
        if rc != 0:
            raise RuntimeError(f"pla_build failed with code {rc}")
        return cls(c_idx)

    @classmethod
    def open_mmap(
        cls, path: str | Path, map_private: bool = False, prefault_spline: bool = True
    ) -> PLAIndex:
        c_idx = _CIndex()
        rc = LIB.pla_open_mmap(
            byref(c_idx), str(path).encode("utf-8"), int(map_private), int(prefault_spline)
        )
        if rc != 0:
            raise RuntimeError(f"pla_open_mmap failed with code {rc}")
        return cls(c_idx)

    def save_mmap(self, path: str | Path) -> None:
        rc = LIB.pla_save_mmap(byref(self._c), str(path).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"pla_save_mmap failed with code {rc}")

    def verify_all(self, num_threads: int = 0) -> int:
        max_err = c_int64(0)
        rc = LIB.pla_verify_all(byref(self._c), byref(max_err), num_threads)
        if rc != 0:
            raise RuntimeError(f"pla_verify_all failed with code {rc}")
        return int(max_err.value)

    @property
    def n_keys(self) -> int:
        return int(self._c.n_keys)

    @property
    def n_segs(self) -> int:
        return int(self._c.n_segs)

    @property
    def eps(self) -> int:
        return int(self._c.eps)

    @property
    def use_perm(self) -> bool:
        return bool(self._c.flags & 1)

    def perm_keys_view(self) -> np.ndarray:
        if self.n_keys == 0:
            return np.empty(0, dtype=np.uint64)
        return np.ctypeslib.as_array(self._c.perm_keys, shape=(self.n_keys,)).copy()

    def seg_starts(self) -> np.ndarray:
        if self.n_segs == 0:
            return np.empty(0, dtype=np.uint64)
        out = np.empty(self.n_segs, dtype=np.uint64)
        for i in range(self.n_segs):
            out[i] = self._c.segs[i].x_start
        return out

    def stats(self) -> PLAStats:
        st = _CStats()
        LIB.pla_compute_stats(byref(self._c), byref(st))
        return PLAStats(
            n_keys=int(st.n_keys),
            n_segs=int(st.n_segs),
            eps=int(st.eps),
            use_perm=bool(st.use_perm),
            empty_buckets=int(st.empty_buckets),
            max_keys_per_bucket=int(st.max_keys_per_bucket),
            max_segs_per_bucket=int(st.max_segs_per_bucket),
            mean_keys_per_seg=float(st.mean_keys_per_seg),
            index_overhead_bytes=int(st.index_overhead_bytes),
            keys_bytes=int(st.keys_bytes),
            total_bytes=int(st.total_bytes),
            overhead_bytes_per_key=float(st.overhead_bytes_per_key),
            total_bytes_per_key=float(st.total_bytes_per_key),
            max_abs_err=int(st.max_abs_err),
        )

    def lookup(self, key: int) -> int:
        return int(LIB.pla_lookup(byref(self._c), c_uint64(key)))

    def lookup_batch(self, queries: np.ndarray, num_threads: int = 0) -> np.ndarray:
        q = np.ascontiguousarray(queries, dtype=np.uint64)
        out = np.empty(q.size, dtype=np.int64)
        if q.size > 0:
            LIB.pla_lookup_batch(
                byref(self._c),
                q.ctypes.data_as(POINTER(c_uint64)),
                out.ctypes.data_as(POINTER(c_int64)),
                q.size,
                num_threads,
            )
        return out

    def count_hits(self, queries: np.ndarray, num_threads: int = 0) -> int:
        q = np.ascontiguousarray(queries, dtype=np.uint64)
        if q.size == 0:
            return 0
        return int(
            LIB.pla_count_hits_batch(
                byref(self._c), q.ctypes.data_as(POINTER(c_uint64)), q.size, num_threads
            )
        )

    @property
    def use_splitmix_syncmer(self) -> bool:
        return bool(self._c.flags & PLA_FLAG_SPLITMIX_SYNCMER)

    def read_pf_hits(self, buf: np.ndarray, s0: int, ln: int, need: int) -> int:
        arr = np.ascontiguousarray(buf, dtype=np.uint8)
        return int(LIB.pla_read_pf_hits(byref(self._c), arr.ctypes.data_as(POINTER(c_uint8)), int(s0), int(ln), int(need)))

    def fastq_block(
        self,
        arr: np.ndarray,
        n: int,
        out: np.ndarray,
        spot0: int,
        keep_thr: int,
        paired: int,
        need: int,
        stats: np.ndarray,
        qhist: np.ndarray,
        ad: np.ndarray,
        min_len: int,
        max_spots: int = 0,
    ) -> tuple[int, int, int]:
        out_o = c_int64(0)
        out_spot = c_int64(spot0)
        ad_ptr = ad.ctypes.data_as(POINTER(c_uint8)) if len(ad) > 0 else None
        used = int(
            LIB.pla_fastq_block(
                arr.ctypes.data_as(POINTER(c_uint8)),
                int(n),
                out.ctypes.data_as(POINTER(c_uint8)),
                int(spot0),
                int(keep_thr),
                int(paired),
                byref(self._c),
                int(need),
                stats.ctypes.data_as(POINTER(c_int64)),
                len(stats),
                qhist.ctypes.data_as(POINTER(c_int64)),
                ad_ptr,
                len(ad),
                int(min_len),
                int(max_spots),
                byref(out_o),
                byref(out_spot),
            )
        )
        return used, int(out_o.value), int(out_spot.value)

    def close(self) -> None:
        if not self._closed:
            LIB.pla_close(byref(self._c))
            self._closed = True

    def __enter__(self) -> PLAIndex:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def decode_kmer_u64(val: int, k: int = 31) -> str:
    alphabet = "ACGT"
    chars = []
    v = int(val)
    for _ in range(k):
        chars.append(alphabet[v & 3])
        v >>= 2
    return "".join(reversed(chars))


def extract_open_syncmers_u64(
    seq: str | bytes,
    k: int = 31,
    s: int = 15,
    skip_lower: bool = False,
    use_splitmix: bool = False,
    dust_max: int = -1,
) -> np.ndarray:
    b = seq.encode("ascii") if isinstance(seq, str) else bytes(seq)
    if len(b) < k:
        return np.empty(0, dtype=np.uint64)
    cap = len(b) - k + 1
    out = np.empty(cap, dtype=np.uint64)
    n = LIB.pla_extract_open_syncmers(
        b, len(b), k, s, int(skip_lower), int(use_splitmix), int(dust_max),
        out.ctypes.data_as(POINTER(c_uint64)), cap,
    )
    return out[:n].copy()


def extract_open_syncmers_str(
    seq: str, k: int = 31, s: int = 15, skip_lower: bool = False
) -> list[str]:
    vals = extract_open_syncmers_u64(seq, k=k, s=s, skip_lower=skip_lower)
    return [decode_kmer_u64(int(v), k=k) for v in vals]


def generate_distinct_syncmers(
    n_target: int,
    mode_at81: bool = True,
    k: int = 31,
    s: int = 15,
    seed: int = 42,
    num_threads: int = 0,
) -> np.ndarray:
    out = np.empty(n_target, dtype=np.uint64)
    if n_target > 0:
        rc = LIB.pla_generate_distinct_syncmers(
            out.ctypes.data_as(POINTER(c_uint64)),
            n_target,
            int(mode_at81),
            k,
            s,
            c_uint64(seed),
            num_threads,
        )
        if rc != 0:
            raise RuntimeError(f"pla_generate_distinct_syncmers failed with code {rc}")
    return out


def sieve_pairs_pla(
    idx: PLAIndex,
    r1_seqs: list[str],
    r2_seqs: list[str],
    min_hits: int = 2,
    max_miss: int = 1,
    num_threads: int = 1,
) -> tuple[np.ndarray, dict[str, int]]:
    n = len(r1_seqs)
    assert len(r2_seqs) == n
    b1 = [x.encode("ascii") for x in r1_seqs]
    b2 = [x.encode("ascii") for x in r2_seqs]
    arr1 = (c_char_p * n)(*b1)
    arr2 = (c_char_p * n)(*b2)
    len1 = (c_size_t * n)(*(len(x) for x in b1))
    len2 = (c_size_t * n)(*(len(x) for x in b2))
    dec = np.zeros(n, dtype=np.uint8)
    st = _CSieveStats()
    LIB.gl_sieve_pairs_pla(
        byref(idx._c),
        arr1,
        len1,
        arr2,
        len2,
        n,
        min_hits,
        max_miss,
        dec.ctypes.data_as(POINTER(c_uint8)),
        byref(st),
        num_threads,
    )
    return dec, {
        "pairs_in": int(st.pairs_in),
        "pairs_out": int(st.pairs_out),
        "drop_locus": int(st.drop_locus),
        "drop_bait": int(st.drop_bait),
    }


def sieve_pairs_htab(
    keys: np.ndarray,
    flags: np.ndarray,
    r1_seqs: list[str],
    r2_seqs: list[str],
    k: int = 31,
    s: int = 15,
    min_hits: int = 2,
    max_miss: int = 1,
    num_threads: int = 1,
) -> tuple[np.ndarray, dict[str, int]]:
    ht = _CHtab()
    k_arr = np.ascontiguousarray(keys, dtype=np.uint64)
    f_arr = np.ascontiguousarray(flags, dtype=np.uint8)
    LIB.gl_ht_init(byref(ht), int(k_arr.size) + 16)
    try:
        for i in range(k_arr.size):
            LIB.gl_ht_insert(byref(ht), int(k_arr[i]), int(f_arr[i]))
        n = len(r1_seqs)
        b1 = [x.encode("ascii") for x in r1_seqs]
        b2 = [x.encode("ascii") for x in r2_seqs]
        arr1 = (c_char_p * n)(*b1)
        arr2 = (c_char_p * n)(*b2)
        len1 = (c_size_t * n)(*(len(x) for x in b1))
        len2 = (c_size_t * n)(*(len(x) for x in b2))
        dec = np.zeros(n, dtype=np.uint8)
        st = _CSieveStats()
        LIB.gl_sieve_pairs_htab(
            byref(ht),
            k,
            s,
            arr1,
            len1,
            arr2,
            len2,
            n,
            min_hits,
            max_miss,
            dec.ctypes.data_as(POINTER(c_uint8)),
            byref(st),
            num_threads,
        )
        return dec, {
            "pairs_in": int(st.pairs_in),
            "pairs_out": int(st.pairs_out),
            "drop_locus": int(st.drop_locus),
            "drop_bait": int(st.drop_bait),
        }
    finally:
        LIB.gl_ht_free(byref(ht))
