"""Optional C99 native accelerators for PanExpr v2.3 mammalian bottlenecks.

Compiles a small self-contained ISO C99 shared library once per machine (cached by source SHA-256
in the system temp directory) and loads it via ``ctypes.CDLL`` so the Python GIL is released on
every call.  When no C compiler is available, every helper returns ``None`` and callers fall back
transparently to the NumPy reference implementation.

Accelerated paths:
  1. ``c_em_sparse``      -- CSR equivalence-class EM + O(L) cross-group ambiguity variance
  2. ``c_rad_aggregate``  -- direct binary ``piscem map-bulk`` ``.rad`` reader + EC deduplicator
  3. ``c_bait_filter``    -- rolling 64-bit canonical k-mer hash-set filter for borderline zoom
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

_C_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* =========================================================================
 * 1. Sparse CSR Equivalence-Class EM + O(L) Cross-Group Ambiguity Variance
 * ========================================================================= */
int panexpr_em_sparse(
    int32_t M,
    int64_t K,
    int64_t L,
    int32_t n_g,
    int32_t two_way,
    int32_t em_iters,
    double dirichlet_alpha,
    const int64_t *starts,
    const int64_t *lens,
    const double *ec_cnt,
    const int32_t *ec_ti,
    const int32_t *t_grp,
    const double *ec_const,
    const uint8_t *sense_mask,
    const double *uniq_arr,
    const double *base_prob,
    const double *prior_arr,
    const double *inv_len,
    double *out_eff,
    double *out_var,
    double *out_ws,
    double *out_gvar
) {
    if (M < 0 || K < 0 || L < 0) return -1;
    memset(out_var, 0, (size_t)M * sizeof(double));
    if (n_g > 0) {
        memset(out_gvar, 0, (size_t)n_g * sizeof(double));
    }

    if (!two_way) {
        if (M > 0) memcpy(out_eff, uniq_arr, (size_t)M * sizeof(double));
        for (int64_t k = 0; k < K; ++k) {
            int64_t s0 = starts[k];
            int64_t s1 = s0 + lens[k];
            double cnt = ec_cnt[k];
            double z = 0.0;
            for (int64_t j = s0; j < s1; ++j) {
                z += ec_const[j];
            }
            double inv_z = (z != 0.0) ? (1.0 / z) : 1.0;
            for (int64_t j = s0; j < s1; ++j) {
                double w = ec_const[j] * inv_z;
                out_ws[j] = w;
                int32_t ti = ec_ti[j];
                if (sense_mask[j] && ti >= 0 && ti < M) {
                    out_eff[ti] += cnt * w;
                    out_var[ti] += cnt * w * (1.0 - w);
                }
            }
        }
    } else {
        if (em_iters <= 0) {
            if (M > 0) memset(out_eff, 0, (size_t)M * sizeof(double));
            return 0;
        }
        double *theta = (double *)malloc((size_t)(M + 1) * sizeof(double));
        if (!theta) return -2;
        for (int32_t i = 0; i < M; ++i) {
            theta[i] = base_prob[i] * (prior_arr[i] + dirichlet_alpha) * inv_len[i];
        }
        theta[M] = 1e-8;
        const double alpha_tail = 0.05 * dirichlet_alpha;

        for (int32_t it = 0; it < em_iters; ++it) {
            if (M > 0) memcpy(out_eff, uniq_arr, (size_t)M * sizeof(double));
            for (int64_t k = 0; k < K; ++k) {
                int64_t s0 = starts[k];
                int64_t lk = lens[k];
                int64_t s1 = s0 + lk;
                double cnt = ec_cnt[k];
                double z = 0.0;
                for (int64_t j = s0; j < s1; ++j) {
                    int32_t ti = ec_ti[j];
                    double th = (ti >= 0 && ti <= M) ? theta[ti] : 1e-8;
                    double r = ec_const[j] * th;
                    out_ws[j] = r;
                    z += r;
                }
                if (z > 0.0) {
                    double inv_z = 1.0 / z;
                    for (int64_t j = s0; j < s1; ++j) {
                        double w = out_ws[j] * inv_z;
                        out_ws[j] = w;
                        int32_t ti = ec_ti[j];
                        if (sense_mask[j] && ti >= 0 && ti < M) {
                            out_eff[ti] += cnt * w;
                        }
                    }
                } else {
                    double inv_l = 1.0 / (double)(lk > 0 ? lk : 1);
                    for (int64_t j = s0; j < s1; ++j) {
                        out_ws[j] = inv_l;
                        int32_t ti = ec_ti[j];
                        if (sense_mask[j] && ti >= 0 && ti < M) {
                            out_eff[ti] += cnt * inv_l;
                        }
                    }
                }
            }
            for (int32_t i = 0; i < M; ++i) {
                theta[i] = base_prob[i] * (out_eff[i] + 0.05 * prior_arr[i] + alpha_tail) * inv_len[i];
            }
        }
        free(theta);

        for (int64_t k = 0; k < K; ++k) {
            int64_t s0 = starts[k];
            int64_t s1 = s0 + lens[k];
            double cnt = ec_cnt[k];
            for (int64_t j = s0; j < s1; ++j) {
                int32_t ti = ec_ti[j];
                if (sense_mask[j] && ti >= 0 && ti < M) {
                    double w = out_ws[j];
                    out_var[ti] += cnt * w * (1.0 - w);
                }
            }
        }
    }

    if (n_g > 0 && K > 0) {
        double *grp_w = (double *)calloc((size_t)n_g, sizeof(double));
        uint8_t *grp_mark = (uint8_t *)calloc((size_t)n_g, 1);
        int32_t *grp_seen = (int32_t *)malloc((size_t)n_g * sizeof(int32_t));
        if (!grp_w || !grp_mark || !grp_seen) {
            free(grp_w);
            free(grp_mark);
            free(grp_seen);
            return -2;
        }
        for (int64_t k = 0; k < K; ++k) {
            int64_t s0 = starts[k];
            int64_t s1 = s0 + lens[k];
            int32_t n_seen = 0;
            for (int64_t j = s0; j < s1; ++j) {
                int32_t ti = ec_ti[j];
                if (sense_mask[j] && ti >= 0 && ti < M) {
                    int32_t g = t_grp[ti];
                    if (g >= 0 && g < n_g) {
                        if (!grp_mark[g]) {
                            grp_mark[g] = 1;
                            grp_seen[n_seen++] = g;
                        }
                        grp_w[g] += out_ws[j];
                    }
                }
            }
            if (n_seen > 1) {
                double cnt = ec_cnt[k];
                for (int32_t u = 0; u < n_seen; ++u) {
                    int32_t g = grp_seen[u];
                    double wg = grp_w[g];
                    out_gvar[g] += cnt * wg * (1.0 - wg);
                }
            }
            for (int32_t u = 0; u < n_seen; ++u) {
                int32_t g = grp_seen[u];
                grp_w[g] = 0.0;
                grp_mark[g] = 0;
            }
        }
        free(grp_w);
        free(grp_mark);
        free(grp_seen);
    }
    return 0;
}

/* =========================================================================
 * 2. Direct Binary piscem .rad Body Reader + Equivalence-Class Aggregator
 * ========================================================================= */
typedef struct {
    int64_t n_mapped;
    int64_t n_eq;
    int64_t total_l;
    int64_t *starts;
    int32_t *lens;
    int32_t *counts;
    uint32_t *flat_packed;
} RadAggOut;

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static inline uint64_t hash_packed_ec(const uint32_t *arr, uint32_t len) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)len * 0xbf58476d1ce4e5b9ULL);
    for (uint32_t i = 0; i < len; ++i) {
        uint64_t x = (uint64_t)arr[i] + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h ^= x + (h << 6) + (h >> 2);
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

RadAggOut *panexpr_rad_aggregate(
    const char *rad_path,
    int64_t body_offset,
    int32_t rw,
    int32_t ori_off,
    int32_t astride,
    int32_t nref,
    int32_t sentinel_idx,
    const int32_t *ref_rank,
    double *out_uniq
) {
    FILE *fp = fopen(rad_path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long fsz = ftell(fp);
    if (fsz < body_offset) { fclose(fp); return NULL; }
    size_t body_len = (size_t)(fsz - body_offset);
    if (fseek(fp, (long)body_offset, SEEK_SET) != 0) { fclose(fp); return NULL; }
    uint8_t *buf = (uint8_t *)malloc(body_len > 0 ? body_len : 1);
    if (!buf) { fclose(fp); return NULL; }
    if (body_len > 0 && fread(buf, 1, body_len, fp) != body_len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    memset(out_uniq, 0, (size_t)nref * 2 * sizeof(double));

    size_t ec_cap = 4096;
    int64_t *ec_starts = (int64_t *)malloc(ec_cap * sizeof(int64_t));
    int32_t *ec_lens = (int32_t *)malloc(ec_cap * sizeof(int32_t));
    int32_t *ec_counts = (int32_t *)malloc(ec_cap * sizeof(int32_t));
    uint64_t *ec_hashes = (uint64_t *)malloc(ec_cap * sizeof(uint64_t));

    size_t flat_cap = 16384;
    uint32_t *flat_packed = (uint32_t *)malloc(flat_cap * sizeof(uint32_t));

    size_t ht_cap = 8192; /* power of 2 */
    int32_t *ht = (int32_t *)malloc(ht_cap * sizeof(int32_t));

    size_t keys_cap = 256;
    uint64_t *keys = (uint64_t *)malloc(keys_cap * sizeof(uint64_t));
    uint32_t *uniq_p = (uint32_t *)malloc(keys_cap * sizeof(uint32_t));

    if (!ec_starts || !ec_lens || !ec_counts || !ec_hashes || !flat_packed || !ht || !keys || !uniq_p) {
        free(buf); free(ec_starts); free(ec_lens); free(ec_counts); free(ec_hashes);
        free(flat_packed); free(ht); free(keys); free(uniq_p);
        return NULL;
    }
    memset(ht, 0xFF, ht_cap * sizeof(int32_t)); /* -1 empty */

    int64_t n_mapped = 0;
    int64_t n_eq = 0;
    int64_t total_l = 0;

    size_t o = 0;
    while (o + 8 <= body_len) {
        uint32_t nbytes = 0, nrec = 0;
        memcpy(&nbytes, buf + o, 4);
        memcpy(&nrec, buf + o + 4, 4);
        if (nbytes < 8 || o + (size_t)nbytes > body_len) break;
        size_t chunk_end = o + (size_t)nbytes;
        o += 8;

        for (uint32_t rec_i = 0; rec_i < nrec; ++rec_i) {
            if (o + 4 + (size_t)rw > chunk_end) break;
            uint32_t na = 0;
            memcpy(&na, buf + o, 4);
            o += 4 + (size_t)rw;
            if (o + (size_t)na * (size_t)astride > chunk_end) break;

            if ((size_t)na > keys_cap) {
                while (keys_cap < (size_t)na) keys_cap *= 2;
                uint64_t *nk = (uint64_t *)realloc(keys, keys_cap * sizeof(uint64_t));
                uint32_t *nu = (uint32_t *)realloc(uniq_p, keys_cap * sizeof(uint32_t));
                if (!nk || !nu) { o = chunk_end; break; }
                keys = nk; uniq_p = nu;
            }

            uint32_t m = 0;
            for (uint32_t a = 0; a < na; ++a) {
                uint32_t v = 0;
                memcpy(&v, buf + o + (size_t)ori_off, 4);
                o += (size_t)astride;
                uint32_t r = v & 0x3FFFFFFFU;
                uint32_t ori = v >> 30;
                if (r >= (uint32_t)nref || (int32_t)r == sentinel_idx) continue;
                uint32_t fl = (ori != 2U) ? 1U : 0U;
                uint64_t rk = (uint64_t)(uint32_t)ref_rank[r];
                keys[m++] = (rk << 33) | ((uint64_t)fl << 32) | ((uint64_t)r << 1) | (uint64_t)fl;
            }
            if (m == 0) continue;

            if (m > 1) {
                if (m <= 24) {
                    for (uint32_t i = 1; i < m; ++i) {
                        uint64_t kv = keys[i];
                        uint32_t j = i;
                        while (j > 0 && keys[j - 1] > kv) {
                            keys[j] = keys[j - 1];
                            --j;
                        }
                        keys[j] = kv;
                    }
                } else {
                    qsort(keys, m, sizeof(uint64_t), cmp_u64);
                }
            }

            uint32_t u = 0;
            uint64_t prev = UINT64_MAX;
            for (uint32_t j = 0; j < m; ++j) {
                if (j == 0 || keys[j] != prev) {
                    prev = keys[j];
                    uniq_p[u++] = (uint32_t)(keys[j] & 0xFFFFFFFFULL);
                }
            }
            n_mapped += 1;

            if (u == 1) {
                uint32_t p0 = uniq_p[0];
                out_uniq[(size_t)(p0 >> 1) * 2 + ((p0 & 1U) ? 0 : 1)] += 1.0;
                continue;
            }

            uint64_t h = hash_packed_ec(uniq_p, u);
            size_t mask = ht_cap - 1;
            size_t slot = (size_t)h & mask;
            int found = 0;
            while (ht[slot] != -1) {
                int32_t idx = ht[slot];
                if (ec_hashes[idx] == h && (uint32_t)ec_lens[idx] == u &&
                    memcmp(flat_packed + ec_starts[idx], uniq_p, (size_t)u * sizeof(uint32_t)) == 0) {
                    ec_counts[idx] += 1;
                    found = 1;
                    break;
                }
                slot = (slot + 1) & mask;
            }
            if (found) continue;

            if ((size_t)n_eq == ec_cap) {
                ec_cap *= 2;
                ec_starts = (int64_t *)realloc(ec_starts, ec_cap * sizeof(int64_t));
                ec_lens = (int32_t *)realloc(ec_lens, ec_cap * sizeof(int32_t));
                ec_counts = (int32_t *)realloc(ec_counts, ec_cap * sizeof(int32_t));
                ec_hashes = (uint64_t *)realloc(ec_hashes, ec_cap * sizeof(uint64_t));
            }
            if ((size_t)(total_l + u) > flat_cap) {
                while (flat_cap < (size_t)(total_l + u)) flat_cap *= 2;
                flat_packed = (uint32_t *)realloc(flat_packed, flat_cap * sizeof(uint32_t));
            }
            int32_t new_id = (int32_t)n_eq++;
            ec_starts[new_id] = total_l;
            ec_lens[new_id] = (int32_t)u;
            ec_counts[new_id] = 1;
            ec_hashes[new_id] = h;
            memcpy(flat_packed + total_l, uniq_p, (size_t)u * sizeof(uint32_t));
            total_l += (int64_t)u;
            ht[slot] = new_id;

            if ((size_t)n_eq * 10 >= ht_cap * 7) {
                ht_cap *= 2;
                int32_t *nht = (int32_t *)malloc(ht_cap * sizeof(int32_t));
                memset(nht, 0xFF, ht_cap * sizeof(int32_t));
                size_t nmask = ht_cap - 1;
                for (int64_t e = 0; e < n_eq; ++e) {
                    size_t s2 = (size_t)ec_hashes[e] & nmask;
                    while (nht[s2] != -1) s2 = (s2 + 1) & nmask;
                    nht[s2] = (int32_t)e;
                }
                free(ht);
                ht = nht;
            }
        }
        o = chunk_end;
    }

    free(buf);
    free(ec_hashes);
    free(ht);
    free(keys);
    free(uniq_p);

    RadAggOut *res = (RadAggOut *)malloc(sizeof(RadAggOut));
    if (!res) {
        free(ec_starts); free(ec_lens); free(ec_counts); free(flat_packed);
        return NULL;
    }
    res->n_mapped = n_mapped;
    res->n_eq = n_eq;
    res->total_l = total_l;
    res->starts = ec_starts;
    res->lens = ec_lens;
    res->counts = ec_counts;
    res->flat_packed = flat_packed;
    return res;
}

void panexpr_rad_free(RadAggOut *res) {
    if (!res) return;
    free(res->starts);
    free(res->lens);
    free(res->counts);
    free(res->flat_packed);
    free(res);
}

/* =========================================================================
 * 3. Rolling 64-bit Canonical k-mer Hash-Set Bait Filter (Borderline Zoom)
 * ========================================================================= */
static inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int64_t panexpr_bait_filter(
    const uint64_t *bait,
    int64_t n_bait,
    const uint8_t *joined_seqs,
    int64_t buf_len,
    int32_t k,
    int32_t min_hits,
    int64_t *out_keep,
    int64_t max_keep
) {
    if (n_bait <= 0 || buf_len <= 0 || k <= 0 || k > 31 || max_keep <= 0) return 0;
    size_t cap = 16;
    while (cap < (size_t)n_bait * 2) cap <<= 1;
    uint64_t *table = (uint64_t *)malloc(cap * sizeof(uint64_t));
    if (!table) return -1;
    memset(table, 0xFF, cap * sizeof(uint64_t)); /* UINT64_MAX = empty */
    size_t hmask = cap - 1;
    for (int64_t i = 0; i < n_bait; ++i) {
        uint64_t v = bait[i];
        size_t s = (size_t)mix64(v) & hmask;
        while (table[s] != UINT64_MAX && table[s] != v) {
            s = (s + 1) & hmask;
        }
        table[s] = v;
    }

    uint8_t lut[256];
    memset(lut, 255, 256);
    lut['A'] = 0; lut['a'] = 0;
    lut['C'] = 1; lut['c'] = 1;
    lut['G'] = 2; lut['g'] = 2;
    lut['T'] = 3; lut['t'] = 3;

    const uint64_t kmask = (k == 32) ? UINT64_MAX : ((1ULL << (2 * k)) - 1ULL);
    const int rev_shift = 2 * (k - 1);

    int64_t n_keep = 0;
    int64_t read_idx = 0;
    int64_t p = 0;

    while (p < buf_len && read_idx < max_keep) {
        uint64_t fwd = 0, rev = 0;
        int32_t run = 0, hits = 0;
        while (p < buf_len) {
            uint8_t ch = joined_seqs[p++];
            if (ch == '\n') break;
            uint8_t c = lut[ch];
            if (c == 255) {
                run = 0;
                fwd = 0;
                rev = 0;
                continue;
            }
            fwd = ((fwd << 2) | (uint64_t)c) & kmask;
            rev = (rev >> 2) | ((uint64_t)(3U - c) << rev_shift);
            if (++run >= k) {
                uint64_t canon = (fwd < rev) ? fwd : rev;
                size_t s = (size_t)mix64(canon) & hmask;
                while (table[s] != UINT64_MAX) {
                    if (table[s] == canon) {
                        if (++hits >= min_hits) {
                            out_keep[n_keep++] = read_idx;
                            /* SIMD-skip remainder of this read */
                            const void *nl = memchr(joined_seqs + p, '\n', (size_t)(buf_len - p));
                            p = nl ? ((const uint8_t *)nl - joined_seqs + 1) : buf_len;
                            goto next_read;
                        }
                        break;
                    }
                    s = (s + 1) & hmask;
                }
            }
        }
    next_read:
        read_idx += 1;
    }

    free(table);
    return n_keep;
}
"""


class _RadAggOutStruct(ctypes.Structure):
    _fields_ = [
        ("n_mapped", ctypes.c_int64),
        ("n_eq", ctypes.c_int64),
        ("total_l", ctypes.c_int64),
        ("starts", ctypes.POINTER(ctypes.c_int64)),
        ("lens", ctypes.POINTER(ctypes.c_int32)),
        ("counts", ctypes.POINTER(ctypes.c_int32)),
        ("flat_packed", ctypes.POINTER(ctypes.c_uint32)),
    ]


_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_TRIED = False


def _find_compiler() -> str | None:
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand:
            exe = shutil.which(cand)
            if exe:
                return exe
    return None


def get_fast_lib() -> ctypes.CDLL | None:
    """Compile (once per machine/source-hash) and load the native C99 library, or return None."""
    global _LIB, _TRIED
    if os.environ.get("PANEXPR_DISABLE_FAST_C") == "1":
        return None
    if _TRIED:
        return _LIB
    with _LOCK:
        if _TRIED:
            return _LIB
        _TRIED = True
        digest = hashlib.sha256(_C_SOURCE.encode("utf-8")).hexdigest()[:16]
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        fname = f"panexpr_fast_{digest}{ext}"
        for pre_dir in (os.environ.get("FOCALOOP_NATIVE_CACHE"), "/opt/focaloop/native_cache"):
            if pre_dir:
                cand = Path(pre_dir) / fname
                if cand.exists():
                    try:
                        lib = ctypes.CDLL(str(cand))
                        _bind_signatures(lib)
                        _LIB = lib
                        return _LIB
                    except Exception:
                        pass
        cc = _find_compiler()
        if not cc:
            return None
        build_dir_env = os.environ.get("FOCALOOP_BUILD_NATIVE_DIR")
        cache_dir = Path(build_dir_env) if build_dir_env else (Path(tempfile.gettempdir()) / f"panexpr_c_v23_{os.getuid()}")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            so_path = cache_dir / fname
            if not so_path.exists():
                pid = os.getpid()
                src_path = cache_dir / f"panexpr_fast_{digest}_{pid}.c"
                tmp_so = cache_dir / f"panexpr_fast_{digest}_{pid}{ext}"
                src_path.write_text(_C_SOURCE)
                isa_flags: list[str] = []
                if os.environ.get("FOCALOOP_PORTABLE_ISA") == "1" and sys.platform != "darwin":
                    isa_flags = ["-msse4.2", "-mpopcnt"]
                try:
                    subprocess.run(
                        [cc, "-O3", "-fPIC", "-shared", *isa_flags, str(src_path), "-lm", "-o", str(tmp_so)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                    os.replace(tmp_so, so_path)
                finally:
                    src_path.unlink(missing_ok=True)
                    tmp_so.unlink(missing_ok=True)
            lib = ctypes.CDLL(str(so_path))
            _bind_signatures(lib)
            _LIB = lib
        except Exception:
            _LIB = None
        return _LIB


def _bind_signatures(lib: ctypes.CDLL) -> None:
    c_i32 = ctypes.c_int32
    c_i64 = ctypes.c_int64
    c_dbl = ctypes.c_double
    p_i32 = ctypes.POINTER(ctypes.c_int32)
    p_i64 = ctypes.POINTER(ctypes.c_int64)
    p_u8 = ctypes.POINTER(ctypes.c_uint8)
    p_u64 = ctypes.POINTER(ctypes.c_uint64)
    p_dbl = ctypes.POINTER(ctypes.c_double)

    lib.panexpr_em_sparse.argtypes = [
        c_i32, c_i64, c_i64, c_i32, c_i32, c_i32, c_dbl,
        p_i64, p_i64, p_dbl, p_i32, p_i32, p_dbl, p_u8,
        p_dbl, p_dbl, p_dbl, p_dbl,
        p_dbl, p_dbl, p_dbl, p_dbl,
    ]
    lib.panexpr_em_sparse.restype = c_i32

    lib.panexpr_rad_aggregate.argtypes = [
        ctypes.c_char_p, c_i64, c_i32, c_i32, c_i32, c_i32, c_i32, p_i32, p_dbl,
    ]
    lib.panexpr_rad_aggregate.restype = ctypes.POINTER(_RadAggOutStruct)

    lib.panexpr_rad_free.argtypes = [ctypes.POINTER(_RadAggOutStruct)]
    lib.panexpr_rad_free.restype = None

    lib.panexpr_bait_filter.argtypes = [
        p_u64, c_i64, p_u8, c_i64, c_i32, c_i32, p_i64, c_i64,
    ]
    lib.panexpr_bait_filter.restype = c_i64


def c_em_sparse(
    M: int,
    K: int,
    L: int,
    n_g: int,
    two_way: bool,
    em_iters: int,
    dirichlet_alpha: float,
    starts: np.ndarray,
    lens: np.ndarray,
    ec_cnt: np.ndarray,
    ec_ti: np.ndarray,
    t_grp: np.ndarray,
    ec_const: np.ndarray,
    sense_mask: np.ndarray,
    uniq_arr: np.ndarray,
    base_prob: np.ndarray,
    prior_arr: np.ndarray,
    inv_len: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Run sparse CSR EM + cross-group variance in C (GIL released); returns (eff, var, ws, gvar) or None."""
    lib = get_fast_lib()
    if lib is None:
        return None
    starts_c = np.ascontiguousarray(starts, dtype=np.int64)
    lens_c = np.ascontiguousarray(lens, dtype=np.int64)
    ec_cnt_c = np.ascontiguousarray(ec_cnt, dtype=np.float64)
    ec_ti_c = np.ascontiguousarray(ec_ti, dtype=np.int32)
    t_grp_c = np.ascontiguousarray(t_grp, dtype=np.int32)
    ec_const_c = np.ascontiguousarray(ec_const, dtype=np.float64)
    sense_c = np.ascontiguousarray(sense_mask, dtype=np.uint8)
    uniq_c = np.ascontiguousarray(uniq_arr, dtype=np.float64)
    base_c = np.ascontiguousarray(base_prob, dtype=np.float64)
    prior_c = np.ascontiguousarray(prior_arr, dtype=np.float64)
    inv_c = np.ascontiguousarray(inv_len, dtype=np.float64)

    out_eff = np.empty(M, dtype=np.float64)
    out_var = np.empty(M, dtype=np.float64)
    out_ws = np.empty(L, dtype=np.float64)
    out_gvar = np.empty(max(1, n_g), dtype=np.float64)

    rc = lib.panexpr_em_sparse(
        M, K, L, n_g, 1 if two_way else 0, em_iters, float(dirichlet_alpha),
        starts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        lens_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ec_cnt_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ec_ti_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        t_grp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ec_const_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sense_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        uniq_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        base_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        prior_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        inv_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_eff.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_var.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_ws.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_gvar.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        return None
    return out_eff, out_var, out_ws, out_gvar[:n_g]


def parse_rad_header(path: str | Path) -> tuple[list[str], int, int, int, int]:
    """Read only the header of a piscem map-bulk .rad file -> (names, body_offset, rw, ori_off, astride)."""
    import struct

    with open(path, "rb") as fh:
        # Header on 250k human targets is ~7 MB; read in chunks if needed
        buf = fh.read(16 << 20)
        o = 0
        if buf[:8] == b"RAD_FILE":
            o = 14
        _is_paired = buf[o]
        o += 1
        (nref,) = struct.unpack_from("<Q", buf, o)
        o += 8
        # Ensure full header is buffered when nref is huge
        if len(buf) < o + nref * 48:
            buf += fh.read()
        names: list[str] = []
        unpack_h = struct.unpack_from
        for _ in range(nref):
            if o + 2 > len(buf):
                buf += fh.read()
            (L,) = unpack_h("<H", buf, o)
            o += 2
            if o + L > len(buf):
                buf += fh.read()
            names.append(buf[o:o + L].decode("ascii"))
            o += L
        if o + 64 > len(buf):
            buf += fh.read()
        (_nchunks,) = unpack_h("<Q", buf, o)
        o += 8
        width = {1: 1, 2: 2, 3: 4, 4: 8, 5: 4, 6: 8}

        def read_descs():
            nonlocal o, buf
            (cnt,) = unpack_h("<H", buf, o)
            o += 2
            out = []
            for _ in range(cnt):
                (L,) = unpack_h("<H", buf, o)
                o += 2
                nm = buf[o:o + L].decode("ascii")
                o += L
                ty = buf[o]
                o += 1
                sub = None
                if ty == 7:
                    sub = (buf[o], buf[o + 1])
                    o += 2
                o += 2
                out.append((nm, ty, sub))
            return out

        file_tags = read_descs()
        read_tags = read_descs()
        aln_tags = read_descs()
        for _nm, ty, sub in file_tags:
            if ty == 8:
                (L,) = unpack_h("<H", buf, o)
                o += 2 + L
            elif ty == 7:
                lw, ew = width[sub[0]], width[sub[1]]
                n = int.from_bytes(buf[o:o + lw], "little")
                o += lw + n * ew
            else:
                o += width[ty]
        rw = sum(width[t] for _, t, _ in read_tags)
        aw = [width[t] for _, t, _ in aln_tags]
        ori_i = next(i for i, (nm, _, _) in enumerate(aln_tags) if nm.startswith("compressed_ori_ref"))
        ori_off = sum(aw[:ori_i])
        astride = sum(aw)
        return names, o, rw, ori_off, astride


def c_rad_aggregate(
    rad_path: str | Path,
    sentinel_id: str,
    cached_meta: dict | None = None,
) -> tuple[int, dict[str, list[float]], dict[tuple[tuple[str, ...], tuple[bool, ...], tuple[float, ...]], int]] | None:
    """Aggregate a piscem .rad file directly into (n_mapped, unique_counts, eq_classes) in C, or return None."""
    lib = get_fast_lib()
    if lib is None:
        return None
    names, body_offset, rw, ori_off, astride = parse_rad_header(rad_path)
    nref = len(names)
    if nref == 0:
        return 0, {}, {}
    if (
        cached_meta is not None
        and cached_meta.get("nref") == nref
        and cached_meta.get("first") == names[0]
        and cached_meta.get("last") == names[-1]
    ):
        ref_rank = cached_meta["ref_rank"]
        sentinel_idx = cached_meta["sentinel_idx"]
    else:
        order = sorted(range(nref), key=names.__getitem__)
        ref_rank = np.empty(nref, dtype=np.int32)
        ref_rank[order] = np.arange(nref, dtype=np.int32)
        try:
            sentinel_idx = names.index(sentinel_id)
        except ValueError:
            sentinel_idx = -1
        if cached_meta is not None:
            cached_meta.update(nref=nref, first=names[0], last=names[-1], ref_rank=ref_rank, sentinel_idx=sentinel_idx)

    out_uniq = np.zeros(nref * 2, dtype=np.float64)
    ptr = lib.panexpr_rad_aggregate(
        str(rad_path).encode("utf-8"),
        int(body_offset),
        int(rw),
        int(ori_off),
        int(astride),
        int(nref),
        int(sentinel_idx),
        ref_rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_uniq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not ptr:
        return None
    try:
        st = ptr.contents
        n_mapped = int(st.n_mapped)
        n_eq = int(st.n_eq)
        total_l = int(st.total_l)
        starts = np.ctypeslib.as_array(st.starts, shape=(n_eq,)).copy() if n_eq > 0 else np.empty(0, dtype=np.int64)
        lens = np.ctypeslib.as_array(st.lens, shape=(n_eq,)).copy() if n_eq > 0 else np.empty(0, dtype=np.int32)
        counts = np.ctypeslib.as_array(st.counts, shape=(n_eq,)).copy() if n_eq > 0 else np.empty(0, dtype=np.int32)
        flat = np.ctypeslib.as_array(st.flat_packed, shape=(total_l,)).copy() if total_l > 0 else np.empty(0, dtype=np.uint32)
    finally:
        lib.panexpr_rad_free(ptr)

    uniq_2d = out_uniq.reshape(nref, 2)
    nz = np.nonzero((uniq_2d[:, 0] != 0.0) | (uniq_2d[:, 1] != 0.0))[0]
    unique_counts: dict[str, list[float]] = {
        names[int(i)]: [float(uniq_2d[i, 0]), float(uniq_2d[i, 1])] for i in nz
    }

    eq_classes: dict[tuple[tuple[str, ...], tuple[bool, ...], tuple[float, ...]], int] = {}
    if n_eq > 0:
        zero_cache: dict[int, tuple[float, ...]] = {}
        flat_r = (flat >> 1).tolist()
        flat_f = [bool(x & 1) for x in flat.tolist()]
        starts_l = starts.tolist()
        lens_l = lens.tolist()
        counts_l = counts.tolist()
        for s0, lk, cnt in zip(starts_l, lens_l, counts_l):
            s1 = s0 + lk
            ts = tuple(names[r] for r in flat_r[s0:s1])
            fls = tuple(flat_f[s0:s1])
            lls = zero_cache.get(lk)
            if lls is None:
                lls = (0.0,) * lk
                zero_cache[lk] = lls
            eq_classes[(ts, fls, lls)] = cnt

    return n_mapped, unique_counts, eq_classes


def c_bait_filter(
    reads: list[tuple[str, str]],
    bait: np.ndarray,
    min_hits: int = 2,
    k: int = 31,
) -> np.ndarray | None:
    """Filter reads sharing >= min_hits canonical k-mers with bait in C, or return None."""
    lib = get_fast_lib()
    if lib is None:
        return None
    n = len(reads)
    if n == 0 or bait.size == 0:
        return np.zeros(0, dtype=np.int64)
    bait_c = np.ascontiguousarray(bait, dtype=np.uint64)
    joined = ("\n".join(s for s, _ in reads) + "\n").encode("ascii", "replace")
    buf_arr = np.frombuffer(joined, dtype=np.uint8)
    out_keep = np.empty(n, dtype=np.int64)
    n_keep = lib.panexpr_bait_filter(
        bait_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        int(bait_c.size),
        buf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(buf_arr.size),
        int(k),
        int(min_hits),
        out_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        int(n),
    )
    if n_keep < 0:
        return None
    return out_keep[:n_keep].copy()
