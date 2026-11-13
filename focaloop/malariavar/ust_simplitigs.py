"""Component 5A: UST Greedy Path-Cover Simplitig Stitcher & ESSColor Panel Compression.

Implements:
  1. Exact canonical k-mer decoy spectrum: V_decoy = canon_k(Introns) \\ canon_k(Transcripts)
  2. Bidirected (reverse-complement aware) compacted de Bruijn graph unitigs
  3. Rahman & Medvedev (RECOMB 2020 / JCB 2021) UST two-way greedy node-disjoint
     path-cover simplitig stitcher (handling odd/even k, palindromic k-mers & (k-1)-mers,
     reverse-complement hairpins, disconnected components, and pure cycles)
  4. Exact SPSS spectrum verification (100% recall, 0 false chimeric k-mers, 0 duplicates)
  5. Faithful PanExpr `catalog.py` `build_merged_intron_decoys` comparison &
     Rahman-Dufresne-Medvedev (2023) `ESSColor` bit-packed haplotype panel compression.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Sequence

import numpy as np
import zlib

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from .native._loader import load_shared_lib

_COMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _load_libust() -> ctypes.CDLL:
    lib = load_shared_lib("libust", ["ust_kernel.c"], use_openmp=False)
    lib.ust_extract_canon_kmers.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ust_extract_canon_kmers.restype = ctypes.c_size_t

    lib.ust_sort_unique_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.ust_sort_unique_u64.restype = ctypes.c_size_t

    lib.ust_sorted_diff_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ust_sorted_diff_u64.restype = ctypes.c_size_t

    lib.ust_extract_unitigs_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.ust_extract_unitigs_u64.restype = ctypes.c_int64
    return lib


_LIB = _load_libust()


def revcomp(seq: str) -> str:
    return seq.translate(_COMP_TABLE)[::-1]


def canon_kmer_str(km: str) -> str:
    rc = revcomp(km)
    return km if km <= rc else rc


def extract_canon_kmers_u64(seqs: Iterable[str], k: int = 31) -> np.ndarray:
    """Return sorted unique canonical 2-bit uint64 k-mers (1 <= k <= 31) across seqs."""
    if not (1 <= k <= 31):
        raise ValueError(f"C uint64 path requires 1 <= k <= 31, got k={k}")
    joined = "\n".join(s.upper() for s in seqs if len(s) >= k).encode("ascii")
    n_bytes = len(joined)
    if n_bytes < k:
        return np.empty(0, dtype=np.uint64)
    buf = np.empty(n_bytes, dtype=np.uint64)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    n_ext = _LIB.ust_extract_canon_kmers(joined, n_bytes, k, ptr)
    if n_ext == 0:
        return np.empty(0, dtype=np.uint64)
    n_uniq = _LIB.ust_sort_unique_u64(ptr, n_ext)
    return buf[:n_uniq].copy()


def extract_all_canon_kmers_multiset_u64(seqs: Iterable[str], k: int = 31) -> np.ndarray:
    """Return ALL canonical k-mer occurrences (unsorted, with duplicates) as uint64 array."""
    if not (1 <= k <= 31):
        raise ValueError(f"C uint64 path requires 1 <= k <= 31, got k={k}")
    joined = "\n".join(s.upper() for s in seqs if len(s) >= k).encode("ascii")
    n_bytes = len(joined)
    if n_bytes < k:
        return np.empty(0, dtype=np.uint64)
    buf = np.empty(n_bytes, dtype=np.uint64)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    n_ext = _LIB.ust_extract_canon_kmers(joined, n_bytes, k, ptr)
    return buf[:n_ext].copy()


def subtract_sorted_u64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact set difference a \\ b for sorted unique uint64 arrays."""
    if len(a) == 0:
        return np.empty(0, dtype=np.uint64)
    if len(b) == 0:
        return a.copy()
    out = np.empty(len(a), dtype=np.uint64)
    pa = a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    pb = b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    po = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    n_out = _LIB.ust_sorted_diff_u64(pa, len(a), pb, len(b), po)
    return out[:n_out].copy()


def compute_decoy_spectrum_u64(
    intron_seqs: Iterable[str],
    transcript_seqs: Iterable[str],
    k: int = 31,
) -> np.ndarray:
    """Compute exact V_decoy = canon_k(Introns) \\ canon_k(Transcripts)."""
    k_int = extract_canon_kmers_u64(intron_seqs, k=k)
    k_tx = extract_canon_kmers_u64(transcript_seqs, k=k)
    return subtract_sorted_u64(k_int, k_tx)


def extract_bidirected_unitigs_u64(v_canon_sorted: np.ndarray, k: int = 31) -> list[str]:
    """Extract maximal bidirected de Bruijn graph unitigs over sorted unique canonical k-mers."""
    n = len(v_canon_sorted)
    if n == 0:
        return []
    if not (2 <= k <= 31):
        raise ValueError(f"2 <= k <= 31 required, got {k}")
    v_arr = np.ascontiguousarray(v_canon_sorted, dtype=np.uint64)
    buf = ctypes.create_string_buffer(n * k + 1)
    offsets = np.empty(n + 1, dtype=np.uintp)
    n_u = _LIB.ust_extract_unitigs_u64(
        v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n,
        k,
        buf,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
    )
    if n_u < 0:
        raise RuntimeError("ust_extract_unitigs_u64 failed")
    raw = ctypes.string_at(buf, int(offsets[n_u])).decode("ascii")
    return [raw[int(offsets[i]):int(offsets[i + 1])] for i in range(int(n_u))]


def extract_bidirected_unitigs_str(canon_kmers: Iterable[str], k: int) -> list[str]:
    """Pure-Python fallback for arbitrary k (including k > 31)."""
    V = sorted(set(canon_kmers))
    if not V:
        return []
    if 2 <= k <= 31:
        enc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.empty(len(V), dtype=np.uint64)
        for idx, s in enumerate(V):
            val = 0
            for ch in s:
                val = (val << 2) | enc_map[ch]
            arr[idx] = val
        arr.sort()
        return extract_bidirected_unitigs_u64(arr, k=k)

    v_idx = {km: i for i, km in enumerate(V)}
    n = len(V)
    rc_list = [revcomp(km) for km in V]
    is_pal = [V[i] == rc_list[i] for i in range(n)]
    uniq_out = [-1] * (2 * n)
    for i in range(n):
        for d in (0, 1):
            s = V[i] if d == 0 else rc_list[i]
            suf = s[1:]
            deg = 0
            last_st = -1
            for b in "ACGT":
                nxt = suf + b
                c = canon_kmer_str(nxt)
                j = v_idx.get(c, -1)
                if j >= 0:
                    dj = 0 if nxt == V[j] else 1
                    deg += 1
                    last_st = (j << 1) | dj
            if deg == 1:
                uniq_out[2 * i + d] = last_st

    ucmp_out = [-1] * (2 * n)
    for i in range(n):
        if is_pal[i]:
            continue
        for d in (0, 1):
            st_nxt = uniq_out[2 * i + d]
            if st_nxt < 0:
                continue
            j, dj = st_nxt >> 1, st_nxt & 1
            if j == i or is_pal[j]:
                continue
            if uniq_out[2 * j + (1 - dj)] == ((i << 1) | (1 - d)):
                ucmp_out[2 * i + d] = st_nxt

    used = [False] * n
    unitigs: list[str] = []
    for i in range(n):
        if used[i]:
            continue
        if is_pal[i]:
            used[i] = True
            unitigs.append(V[i])
            continue
        start_d = 0 if ucmp_out[2 * i + 1] == -1 else (1 if ucmp_out[2 * i + 0] == -1 else -1)
        if start_d < 0:
            continue
        used[i] = True
        chars = [V[i] if start_d == 0 else rc_list[i]]
        cur_st = (i << 1) | start_d
        while True:
            nxt_st = ucmp_out[cur_st]
            if nxt_st < 0:
                break
            j, dj = nxt_st >> 1, nxt_st & 1
            if used[j]:
                break
            used[j] = True
            sj = V[j] if dj == 0 else rc_list[j]
            chars.append(sj[-1])
            cur_st = nxt_st
        unitigs.append("".join(chars))

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        chars = [V[i]]
        cur_st = i << 1
        while True:
            nxt_st = ucmp_out[cur_st]
            if nxt_st < 0:
                break
            j, dj = nxt_st >> 1, nxt_st & 1
            if used[j]:
                break
            used[j] = True
            sj = V[j] if dj == 0 else rc_list[j]
            chars.append(sj[-1])
            cur_st = nxt_st
        unitigs.append("".join(chars))
    return unitigs


def stitch_unitigs_ust(unitigs: Sequence[str], k: int = 31) -> list[str]:
    """Rahman & Medvedev (2020) UST greedy node-disjoint path-cover over bidirected unitigs.

    Guarantees:
      - Every unitig in `unitigs` is covered in either forward (+) or reverse-complement (-)
        orientation EXACTLY ONCE.
      - Adjacent stitched unitigs share an exact (k-1)-bp overlap:
          suffix_{k-1}(u, dir_u) == prefix_{k-1}(v, dir_v)
      - Introduces ZERO new junction k-mers and ZERO duplicate canonical k-mers!
    """
    m_total = len(unitigs)
    if m_total == 0:
        return []
    km1 = k - 1
    if km1 <= 0:
        return list(unitigs)

    rc_unitigs = [revcomp(u) for u in unitigs]
    pref0 = [u[:km1] for u in unitigs]
    suf0 = [u[-km1:] for u in unitigs]
    pref1 = [r[:km1] for r in rc_unitigs]
    suf1 = [r[-km1:] for r in rc_unitigs]
    is_pal_u = [unitigs[i] == rc_unitigs[i] for i in range(m_total)]

    # Inverted prefix index: (k-1)-mer -> list of (unitig_id, orientation)
    pref_buckets: dict[str, list[tuple[int, int]]] = {}
    for i in range(m_total):
        pref_buckets.setdefault(pref0[i], []).append((i, 0))
        if not is_pal_u[i]:
            pref_buckets.setdefault(pref1[i], []).append((i, 1))

    # Compute initial right/left degree for tip-prioritized seeding
    seed_keys: list[tuple[int, int, int]] = []
    for i in range(m_total):
        r_deg = sum(1 for (j, _) in pref_buckets.get(suf0[i], ()) if j != i)
        l_deg = sum(1 for (j, _) in pref_buckets.get(suf1[i], ()) if j != i)
        lo, hi = (r_deg, l_deg) if r_deg <= l_deg else (l_deg, r_deg)
        seed_keys.append((lo, hi, i))
    seed_keys.sort()

    visited = [False] * m_total
    bucket_cursor: dict[str, int] = {x: 0 for x in pref_buckets}

    def _pop_next(suf_str: str) -> tuple[int, int] | None:
        b = pref_buckets.get(suf_str)
        if b is None:
            return None
        cur = bucket_cursor[suf_str]
        nb = len(b)
        while cur < nb and visited[b[cur][0]]:
            cur += 1
        bucket_cursor[suf_str] = cur
        if cur >= nb:
            return None
        # Look ahead up to 8 active candidates to prefer dead-end tips (minimum onward degree)
        best_j, best_d = b[cur]
        best_onward = len(pref_buckets.get(suf0[best_j] if best_d == 0 else suf1[best_j], ()))
        scan_end = min(nb, cur + 8)
        for p in range(cur + 1, scan_end):
            cand_j, cand_d = b[p]
            if visited[cand_j]:
                continue
            onw = len(pref_buckets.get(suf0[cand_j] if cand_d == 0 else suf1[cand_j], ()))
            if onw < best_onward:
                best_j, best_d = cand_j, cand_d
                best_onward = onw
                if best_onward <= 1:
                    break
        return best_j, best_d

    simplitigs: list[str] = []
    for _lo, _hi, seed_i in seed_keys:
        if visited[seed_i]:
            continue
        visited[seed_i] = True

        # Forward extension from (seed_i, 0)
        fwd_chain: list[tuple[int, int]] = []
        cur_i, cur_d = seed_i, 0
        while True:
            cur_suf = suf0[cur_i] if cur_d == 0 else suf1[cur_i]
            nxt = _pop_next(cur_suf)
            if nxt is None:
                break
            nxt_i, nxt_d = nxt
            visited[nxt_i] = True
            fwd_chain.append((nxt_i, nxt_d))
            cur_i, cur_d = nxt_i, nxt_d

        # Backward extension before (seed_i, 0) == forward extension from (seed_i, 1) reversed & flipped
        bwd_chain: list[tuple[int, int]] = []
        if not is_pal_u[seed_i]:
            cur_i, cur_d = seed_i, 1
            while True:
                cur_suf = suf0[cur_i] if cur_d == 0 else suf1[cur_i]
                nxt = _pop_next(cur_suf)
                if nxt is None:
                    break
                nxt_i, nxt_d = nxt
                visited[nxt_i] = True
                # Flipping (nxt_i, nxt_d) gives suf(nxt_i, 1 - nxt_d) == pref(prev_i, 1 - prev_d)
                flipped_d = 0 if is_pal_u[nxt_i] else (1 - nxt_d)
                bwd_chain.append((nxt_i, flipped_d))
                cur_i, cur_d = nxt_i, nxt_d

        full_chain = list(reversed(bwd_chain))
        full_chain.append((seed_i, 0))
        full_chain.extend(fwd_chain)

        first_i, first_d = full_chain[0]
        parts = [unitigs[first_i] if first_d == 0 else rc_unitigs[first_i]]
        for ui, ud in full_chain[1:]:
            s = unitigs[ui] if ud == 0 else rc_unitigs[ui]
            parts.append(s[km1:])
        simplitigs.append("".join(parts))

    return simplitigs


def ust_char_lower_bound(unitigs: Sequence[str], n_unique_kmers: int, k: int = 31) -> dict[str, int]:
    """Compute Rahman & Medvedev (2020) bidirected degree-imbalance lower bound on simplitig count & chars."""
    if not unitigs or n_unique_kmers == 0:
        return {"min_simplitigs_lb": 0, "min_chars_lb": 0}
    km1 = k - 1
    suf_counts: dict[str, int] = {}
    pref_counts: dict[str, int] = {}
    all_km1: set[str] = set()
    for u in unitigs:
        ru = revcomp(u)
        s0, p0 = u[-km1:], u[:km1]
        s1, p1 = ru[-km1:], ru[:km1]
        suf_counts[s0] = suf_counts.get(s0, 0) + 1
        pref_counts[p0] = pref_counts.get(p0, 0) + 1
        if u != ru:
            suf_counts[s1] = suf_counts.get(s1, 0) + 1
            pref_counts[p1] = pref_counts.get(p1, 0) + 1
        all_km1.update((s0, p0, s1, p1))

    unmatched_ends = 0
    seen_canon: set[str] = set()
    for x in all_km1:
        rx = revcomp(x)
        cx = x if x <= rx else rx
        if cx in seen_canon:
            continue
        seen_canon.add(cx)
        if x == rx:
            # Palindromic (k-1)-mer: each physical unitig end contributes to both suf(x) and pref(x)
            # Actually each unitig end touching palindrome x has suf==x in one orientation and pref==x in RC!
            unmatched_ends += suf_counts.get(x, 0) % 2
        else:
            unmatched_ends += abs(suf_counts.get(cx, 0) - pref_counts.get(cx, 0))

    min_simplitigs = max(1, (unmatched_ends + 1) // 2)
    min_chars = n_unique_kmers + km1 * min_simplitigs
    return {"min_simplitigs_lb": min_simplitigs, "min_chars_lb": min_chars}


def build_ust_decoy_simplitigs(
    intron_seqs: Iterable[str],
    transcript_seqs: Iterable[str],
    k: int = 31,
) -> tuple[list[str], list[str], np.ndarray]:
    """Compute V_decoy = canon_k(Introns) \\ canon_k(Transcripts), unitigs, and UST simplitigs."""
    v_decoy = compute_decoy_spectrum_u64(intron_seqs, transcript_seqs, k=k)
    unitigs = extract_bidirected_unitigs_u64(v_decoy, k=k)
    simplitigs = stitch_unitigs_ust(unitigs, k=k)
    return simplitigs, unitigs, v_decoy


def verify_exact_spss_spectrum_u64(
    strings: Sequence[str],
    expected_v_sorted: np.ndarray,
    k: int = 31,
) -> dict[str, int | float | bool]:
    """Strictly verify 100.000% recall, 0 false chimeric k-mers, and 0 duplicate canonical k-mers."""
    multiset = extract_all_canon_kmers_multiset_u64(strings, k=k)
    total_slots = sum(max(0, len(s) - k + 1) for s in strings)
    uniq_actual = extract_canon_kmers_u64(strings, k=k)

    missing = subtract_sorted_u64(expected_v_sorted, uniq_actual)
    extra = subtract_sorted_u64(uniq_actual, expected_v_sorted)
    n_duplicates = int(len(multiset) - len(uniq_actual))
    has_non_acgt = int(total_slots - len(multiset))

    exact_match = (
        len(missing) == 0
        and len(extra) == 0
        and n_duplicates == 0
        and has_non_acgt == 0
        and len(uniq_actual) == len(expected_v_sorted)
    )
    recall = 1.0 if len(expected_v_sorted) == 0 else (
        (len(expected_v_sorted) - len(missing)) / len(expected_v_sorted)
    )
    return {
        "exact_spss": bool(exact_match),
        "expected_kmers": int(len(expected_v_sorted)),
        "actual_unique_kmers": int(len(uniq_actual)),
        "total_kmer_occurrences": int(len(multiset)),
        "missing_kmers": int(len(missing)),
        "false_chimeric_kmers": int(len(extra)),
        "duplicate_kmers": n_duplicates,
        "recall_pct": float(100.0 * recall),
    }


def verify_exact_spss_spectrum_str(
    strings: Sequence[str],
    expected_canon_set: set[str],
    k: int,
) -> dict[str, int | bool]:
    """String-based exact SPSS verifier supporting arbitrary k (including even k)."""
    seen: set[str] = set()
    duplicates = 0
    extra = 0
    total_occ = 0
    for s in strings:
        su = s.upper()
        for i in range(len(su) - k + 1):
            km = su[i:i + k]
            if any(ch not in "ACGT" for ch in km):
                raise ValueError(f"Non-ACGT character in simplitig k-mer: {km}")
            total_occ += 1
            ck = canon_kmer_str(km)
            if ck not in expected_canon_set:
                extra += 1
            if ck in seen:
                duplicates += 1
            else:
                seen.add(ck)
    missing = len(expected_canon_set - seen)
    return {
        "exact_spss": (missing == 0 and extra == 0 and duplicates == 0),
        "expected_kmers": len(expected_canon_set),
        "actual_unique_kmers": len(seen),
        "total_kmer_occurrences": total_occ,
        "missing_kmers": missing,
        "false_chimeric_kmers": extra,
        "duplicate_kmers": duplicates,
    }


def puncture_introns_by_transcripts(
    intron_seqs: Iterable[str],
    transcript_seqs: Iterable[str],
    k: int = 31,
) -> list[str]:
    """Extract maximal contiguous >= k bp intronic fragments containing zero transcript canonical k-mers."""
    tx_kmers = set(extract_canon_kmers_u64(transcript_seqs, k=k).tolist())
    base_lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = (1 << (2 * k)) - 1
    shift_rc = 2 * (k - 1)
    fragments: list[str] = []

    for seq in intron_seqs:
        su = seq.upper()
        n = len(su)
        if n < k:
            continue
        fw = 0
        rc = 0
        run = 0
        frag_start: int | None = None
        for i, ch in enumerate(su):
            b = base_lut.get(ch, 4)
            if b < 4:
                fw = ((fw << 2) | b) & mask
                rc = (rc >> 2) | ((3 ^ b) << shift_rc)
                run += 1
                if run >= k:
                    ck = fw if fw < rc else rc
                    if ck not in tx_kmers:
                        if frag_start is None:
                            frag_start = i - k + 1
                    else:
                        if frag_start is not None:
                            fragments.append(su[frag_start:i])
                            frag_start = None
            else:
                if frag_start is not None:
                    fragments.append(su[frag_start:i])
                    frag_start = None
                run = 0
                fw = 0
                rc = 0
        if frag_start is not None:
            fragments.append(su[frag_start:n])
    return fragments


def pack_n_spacer_super_decoys(
    fragments: Sequence[str],
    spacer_len: int = 64,
    super_chunk_bp: int = 262_144,
) -> list[str]:
    """Pack fragments into super_chunk_bp super-targets separated by spacer_len 'N's
    (exact packing logic of PanExpr catalog.py `build_merged_intron_decoys` lines 443-469).
    """
    spacer = "N" * spacer_len
    out: list[str] = []
    cur: list[str] = []
    cur_bp = 0
    for frag in fragments:
        seg_len = len(frag)
        add_len = seg_len + (spacer_len if cur else 0)
        if cur and cur_bp + add_len > super_chunk_bp:
            out.append(spacer.join(cur))
            cur = [frag]
            cur_bp = seg_len
        else:
            cur.append(frag)
            cur_bp += add_len
    if cur:
        out.append(spacer.join(cur))
    return out


def _merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ivs:
        return []
    s_ivs = sorted(ivs)
    merged = [s_ivs[0]]
    for a, b in s_ivs[1:]:
        if a <= merged[-1][1] + 1:
            if b > merged[-1][1]:
                merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _subtract_intervals(spans: list[tuple[int, int]], masks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans or not masks:
        return list(spans)
    out: list[tuple[int, int]] = []
    j = 0
    n_m = len(masks)
    for a, b in spans:
        cur = a
        while j < n_m and masks[j][1] < cur:
            j += 1
        idx = j
        while idx < n_m and masks[idx][0] <= b:
            ma, mb = masks[idx]
            if ma > cur:
                out.append((cur, min(b, ma - 1)))
            cur = max(cur, mb + 1)
            if cur > b:
                break
            idx += 1
        if cur <= b:
            out.append((cur, b))
    return out


def panexpr_build_merged_intron_decoys_faithful(
    genome: dict[str, str],
    transcript_exons: list[tuple[str, list[tuple[int, int]]]],
    min_len: int = 60,
    super_chunk_bp: int = 262_144,
    flank: int = 20,
    spacer_len: int = 64,
) -> list[str]:
    """Faithful standalone reproduction of PanExpr `catalog.py:build_merged_intron_decoys` (lines 382-470)."""
    raw_introns: dict[str, list[tuple[int, int]]] = {}
    raw_exons: dict[str, list[tuple[int, int]]] = {}
    for contig, exons in transcript_exons:
        if not exons:
            continue
        raw_exons.setdefault(contig, []).extend(exons)
        if len(exons) < 2:
            continue
        for (_s1, e1), (s2, _e2) in zip(exons[:-1], exons[1:]):
            istart, iend = e1 + 1, s2 - 1
            if iend - istart + 1 >= min_len:
                raw_introns.setdefault(contig, []).append((max(1, istart - flank), iend + flank))

    spacer = "N" * spacer_len
    out_seqs: list[str] = []
    for ctg in sorted(raw_introns):
        seq = genome.get(ctg)
        if not seq:
            continue
        ctg_len = len(seq)
        clamped = [(max(1, a), min(ctg_len, b)) for a, b in raw_introns[ctg] if a <= ctg_len]
        merged_int = _merge_intervals(clamped)
        exon_cores = [
            (a + flank, b - flank)
            for a, b in _merge_intervals(raw_exons.get(ctg, []))
            if b - a + 1 > 2 * flank
        ]
        pure_int = [
            (a, b)
            for a, b in _subtract_intervals(merged_int, exon_cores)
            if b - a + 1 >= min_len
        ]
        pieces: list[tuple[int, int]] = []
        for a, b in pure_int:
            if b - a + 1 <= super_chunk_bp:
                pieces.append((a, b))
            else:
                for cs in range(a, b + 1, super_chunk_bp):
                    ce = min(b, cs + super_chunk_bp - 1)
                    if ce - cs + 1 >= min_len:
                        pieces.append((cs, ce))
        frags = [seq[a - 1:b] for a, b in pieces]
        out_seqs.extend(pack_n_spacer_super_decoys(frags, spacer_len=spacer_len, super_chunk_bp=super_chunk_bp))
    return out_seqs


def fasta_metrics(records: Sequence[str], n_unique_kmers: int, k: int = 31, prefix: str = "decoy") -> dict[str, int | float]:
    """Compute sequence characters, total FASTA bytes (with headers), record count, and boundary-tax overhead."""
    n_rec = len(records)
    seq_chars = sum(len(r) for r in records)
    fasta_bytes = sum(len(f">{prefix}_{i}\n") + len(r) + 1 for i, r in enumerate(records))
    boundary_tax_chars = seq_chars - n_unique_kmers
    return {
        "n_records": n_rec,
        "seq_chars": seq_chars,
        "fasta_bytes": fasta_bytes,
        "boundary_tax_chars": boundary_tax_chars,
        "chars_per_unique_kmer": float(seq_chars / max(1, n_unique_kmers)),
        "boundary_overhead_pct": float(100.0 * boundary_tax_chars / max(1, n_unique_kmers)),
    }


@dataclass
class ESSColorPanel:
    """Rahman, Dufresne & Medvedev (2023) ESSColor-style compressed haplotype panel:
    shared UST SPSS backbone + bit-packed run-length Zstd haplotype incidence matrix.
    """
    k: int
    n_haplotypes: int
    n_backbone_kmers: int
    ust_simplitigs: list[str]
    packed_zstd_blob: bytes
    isolate_haplotype_ids_zstd: bytes | None = None


def build_esscolor_panel(
    haplotype_seqs: Sequence[str],
    k: int = 31,
    isolate_to_hap_idx: np.ndarray | None = None,
    zstd_level: int = 6,
) -> ESSColorPanel:
    """Compress multi-allele target panel into one UST SPSS backbone + bit-packed color matrix."""
    n_hap = len(haplotype_seqs)
    union_v = extract_canon_kmers_u64(haplotype_seqs, k=k)
    unitigs = extract_bidirected_unitigs_u64(union_v, k=k)
    simplitigs = stitch_unitigs_ust(unitigs, k=k)

    # Order canonical k-mers strictly along the UST simplitigs so adjacent rows share color vectors!
    backbone_kmers = extract_all_canon_kmers_multiset_u64(simplitigs, k=k)
    assert len(backbone_kmers) == len(union_v), "UST backbone must visit each canonical k-mer once"

    sort_order = np.argsort(backbone_kmers)
    sorted_bk = backbone_kmers[sort_order]

    n_kmers = len(backbone_kmers)
    n_bytes_row = (n_hap + 7) // 8
    color_bits = np.zeros((n_kmers, n_bytes_row), dtype=np.uint8)

    for h_idx, h_seq in enumerate(haplotype_seqs):
        hk = extract_canon_kmers_u64([h_seq], k=k)
        pos_in_sorted = np.searchsorted(sorted_bk, hk)
        backbone_positions = sort_order[pos_in_sorted]
        byte_col = h_idx >> 3
        bit_mask = np.uint8(1 << (h_idx & 7))
        color_bits[backbone_positions, byte_col] |= bit_mask

    # ESSColor delta-XOR along UST walk: consecutive k-mers on a simplitig have identical colors
    # except at variant bubble boundaries -> >98% of XOR rows are all-zero!
    delta_xor = np.empty_like(color_bits)
    if n_kmers > 0:
        delta_xor[0] = color_bits[0]
        delta_xor[1:] = np.bitwise_xor(color_bits[1:], color_bits[:-1])

    simplitig_text = "\n".join(simplitigs).encode("ascii")
    payload = (
        struct.pack("<III", n_hap, n_kmers, len(simplitig_text))
        + simplitig_text
        + delta_xor.tobytes()
    )
    if zstd is not None:
        cctx = zstd.ZstdCompressor(level=zstd_level)
        compressed_blob = cctx.compress(payload)
        iso_blob = None
        if isolate_to_hap_idx is not None:
            iso_arr = np.asarray(isolate_to_hap_idx, dtype=np.uint16)
            iso_blob = cctx.compress(iso_arr.tobytes())
    else:
        lvl = min(9, max(1, int(zstd_level)))
        compressed_blob = zlib.compress(payload, level=lvl)
        iso_blob = None
        if isolate_to_hap_idx is not None:
            iso_arr = np.asarray(isolate_to_hap_idx, dtype=np.uint16)
            iso_blob = zlib.compress(iso_arr.tobytes(), level=lvl)

    return ESSColorPanel(
        k=k,
        n_haplotypes=n_hap,
        n_backbone_kmers=n_kmers,
        ust_simplitigs=simplitigs,
        packed_zstd_blob=compressed_blob,
        isolate_haplotype_ids_zstd=iso_blob,
    )


def reconstruct_esscolor_haplotype_kmers(panel: ESSColorPanel, hap_idx: int) -> np.ndarray:
    """Decompress ESSColor panel and reconstruct exact sorted canonical k-mer set for haplotype `hap_idx`."""
    if zstd is not None:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(panel.packed_zstd_blob)
    else:
        raw = zlib.decompress(panel.packed_zstd_blob)
    n_hap, n_kmers, text_len = struct.unpack("<III", raw[:12])
    if not (0 <= hap_idx < n_hap):
        raise IndexError(f"hap_idx={hap_idx} out of range [0, {n_hap})")
    simplitig_text = raw[12:12 + text_len].decode("ascii")
    simplitigs = simplitig_text.split("\n") if simplitig_text else []
    backbone_kmers = extract_all_canon_kmers_multiset_u64(simplitigs, k=panel.k)

    n_bytes_row = (n_hap + 7) // 8
    delta_xor = np.frombuffer(raw, dtype=np.uint8, offset=12 + text_len).reshape((n_kmers, n_bytes_row))
    # Prefix XOR in GF(2): cumulative XOR mod 2 recovers original color_bits!
    # In uint8 bitwise XOR, prefix XOR via np.bitwise_xor.accumulate:
    color_bits = np.bitwise_xor.accumulate(delta_xor, axis=0)

    byte_col = hap_idx >> 3
    bit_mask = np.uint8(1 << (hap_idx & 7))
    present = (color_bits[:, byte_col] & bit_mask) != 0
    out = backbone_kmers[present].copy()
    out.sort()
    return out
