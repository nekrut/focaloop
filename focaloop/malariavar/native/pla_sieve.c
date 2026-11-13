// pla_sieve.c — Implementation of splitmix64-Permuted mmap PLA-index & glcore.c Baselines

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif
#include "pla_sieve.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------------------------

static inline size_t align_up(size_t x, size_t a) {
    return (x + (a - 1U)) & ~(a - 1U);
}

static int effective_threads(int req) {
#ifdef _OPENMP
    int max_t = omp_get_max_threads();
    int t = (req > 0) ? req : max_t;
    if (t > 64) t = 64;
    if (t < 1) t = 1;
    return t;
#else
    (void)req;
    return 1;
#endif
}

// -----------------------------------------------------------------------------------------------
// 1. Parallel 8-Pass 8-Bit LSD Radix Sort + In-Place Deduplication
// -----------------------------------------------------------------------------------------------

static int radix_sort_u64(uint64_t *a, size_t n, int num_threads) {
    if (n <= 1) return 0;
    uint64_t *tmp = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tmp) return -1;

    int T = (n < 65536U) ? 1 : effective_threads(num_threads);
    size_t *hist = (size_t *)malloc((size_t)T * 256U * sizeof(size_t));
    if (!hist) {
        free(tmp);
        return -1;
    }

    uint64_t *src = a;
    uint64_t *dst = tmp;

    for (int pass = 0; pass < 8; ++pass) {
        const unsigned shift = (unsigned)(pass * 8);

        #pragma omp parallel num_threads(T)
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            size_t s0 = (n * (size_t)tid) / (size_t)T;
            size_t s1 = (n * (size_t)(tid + 1)) / (size_t)T;
            size_t local_h[256];
            memset(local_h, 0, sizeof(local_h));
            for (size_t i = s0; i < s1; ++i) {
                local_h[(src[i] >> shift) & 0xFFU]++;
            }
            memcpy(&hist[(size_t)tid * 256U], local_h, sizeof(local_h));
        }

        // Check whether all n keys share a single byte value in this radix digit (skip scatter if so).
        int skip_pass = 0;
        for (size_t b = 0; b < 256U; ++b) {
            size_t col_sum = 0;
            for (int t = 0; t < T; ++t) col_sum += hist[(size_t)t * 256U + b];
            if (col_sum == n) {
                skip_pass = 1;
                break;
            }
        }
        if (skip_pass) continue;

        size_t running = 0;
        for (size_t b = 0; b < 256U; ++b) {
            for (int t = 0; t < T; ++t) {
                size_t c = hist[(size_t)t * 256U + b];
                hist[(size_t)t * 256U + b] = running;
                running += c;
            }
        }

        #pragma omp parallel num_threads(T)
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            size_t s0 = (n * (size_t)tid) / (size_t)T;
            size_t s1 = (n * (size_t)(tid + 1)) / (size_t)T;
            size_t local_off[256];
            memcpy(local_off, &hist[(size_t)tid * 256U], sizeof(local_off));
            for (size_t i = s0; i < s1; ++i) {
                uint64_t v = src[i];
                unsigned byte = (unsigned)((v >> shift) & 0xFFU);
                dst[local_off[byte]++] = v;
            }
        }

        uint64_t *swp = src;
        src = dst;
        dst = swp;
    }

    if (src != a) {
        memcpy(a, src, n * sizeof(uint64_t));
    }

    free(hist);
    free(tmp);
    return 0;
}

size_t pla_sort_dedup_u64(uint64_t *keys, size_t n, int num_threads) {
    if (n <= 1) return n;
    if (radix_sort_u64(keys, n, num_threads) != 0) return 0;

    int T = (n < 65536U) ? 1 : effective_threads(num_threads);
    size_t n_dups = 0;
    #pragma omp parallel for reduction(+:n_dups) num_threads(T) schedule(static)
    for (size_t i = 1; i < n; ++i) {
        if (keys[i] == keys[i - 1]) n_dups++;
    }
    if (n_dups == 0) return n;

    size_t w = 1;
    for (size_t r = 1; r < n; ++r) {
        if (keys[r] != keys[w - 1]) {
            keys[w++] = keys[r];
        }
    }
    return w;
}

typedef struct {
    uint64_t k;
    uint8_t  f;
} key_flag_t;

static int cmp_key_flag(const void *pa, const void *pb) {
    uint64_t a = ((const key_flag_t *)pa)->k;
    uint64_t b = ((const key_flag_t *)pb)->k;
    return (a > b) - (a < b);
}

// -----------------------------------------------------------------------------------------------
// 2. O'Rourke (1981) Exact O(N) Convex-Hull Shrinking-Cone Optimal eps-Tube PLA Builder
// -----------------------------------------------------------------------------------------------

typedef struct {
    uint64_t dx;
    int64_t  y;
} hull_pt_t;

typedef struct {
    uint64_t dx; // always > 0
    int64_t  dy;
} hull_slope_t;

static inline hull_slope_t pt_slope(hull_pt_t a, hull_pt_t b) {
    hull_slope_t s;
    s.dx = b.dx - a.dx;
    s.dy = b.y - a.y;
    return s;
}

static inline int slope_lt(hull_slope_t a, hull_slope_t b) {
    return ((__int128)a.dy * (__int128)b.dx) < ((__int128)b.dy * (__int128)a.dx);
}

static inline int slope_gt(hull_slope_t a, hull_slope_t b) {
    return ((__int128)a.dy * (__int128)b.dx) > ((__int128)b.dy * (__int128)a.dx);
}

static int verify_seg_on_range(
    const pla_seg_t *seg,
    const uint64_t *keys,
    size_t i0,
    size_t i1,
    size_t n_keys,
    uint32_t eps,
    size_t *first_bad
) {
    int64_t e = (int64_t)eps;
    for (size_t j = i0; j <= i1; ++j) {
        int64_t p_hat = pla_predict_seg(seg, keys[j], n_keys);
        int64_t diff  = p_hat - (int64_t)j;
        if (diff < -e || diff > e) {
            if (first_bad) *first_bad = j;
            return 0;
        }
    }
    return 1;
}

// Fit quantized (uint32_t intercept, float slope) on [i0, i1] and verify bit-for-bit against pla_predict_seg.
static int fit_quantized_seg(
    const uint64_t *keys,
    size_t i0,
    size_t i1,
    size_t n_keys,
    uint32_t eps,
    double a_hint,
    pla_seg_t *out_seg,
    size_t *first_bad
) {
    uint64_t x0 = keys[i0];
    if (i1 <= i0 + 2U * (size_t)eps) {
        size_t mid_idx = (i0 + i1) >> 1;
        if (mid_idx >= n_keys) mid_idx = n_keys - 1;
        pla_seg_t zseg = { .x_start = x0, .intercept = (uint32_t)mid_idx, .slope = 0.0f };
        if (verify_seg_on_range(&zseg, keys, i0, i1, n_keys, eps, first_bad)) {
            *out_seg = zseg;
            return 1;
        }
    }

    if (a_hint < 0.0) a_hint = 0.0;

    // 1. Find valid integer intercept interval [b_lo, b_hi) under floor truncation at slope a_hint.
    double b_lo = 0.0;
    double b_hi = (double)(n_keys - 1);
    for (size_t j = i0; j <= i1; ++j) {
        double off  = a_hint * (double)(keys[j] - x0);
        double lo_j = (double)((int64_t)j - (int64_t)eps) - off;
        double hi_j = (double)((int64_t)j + (int64_t)eps + 1) - 1e-7 - off;
        if (lo_j > b_lo) b_lo = lo_j;
        if (hi_j < b_hi) b_hi = hi_j;
    }

    int64_t ic = (int64_t)ceil(b_lo);
    if ((double)ic > b_hi) {
        ic = (int64_t)llround(0.5 * (b_lo + b_hi));
    }
    if (ic < 0) ic = 0;
    if ((size_t)ic >= n_keys) ic = (int64_t)(n_keys - 1);

    // 2. Re-compute exact feasible slope interval [f_lo, f_hi] for fixed integer intercept ic.
    double f_lo = 0.0;
    double f_hi = 1e300;
    for (size_t j = i0 + 1; j <= i1; ++j) {
        double dx_j = (double)(keys[j] - x0);
        double s0   = ((double)((int64_t)j - (int64_t)eps - ic)) / dx_j;
        double s1   = ((double)((int64_t)j + (int64_t)eps + 1 - ic) - 1e-7) / dx_j;
        if (s0 > f_lo) f_lo = s0;
        if (s1 < f_hi) f_hi = s1;
    }

    float base_s = (f_lo <= f_hi) ? (float)(0.5 * (f_lo + f_hi)) : (float)a_hint;
    if (base_s < 0.0f) base_s = 0.0f;

    // Test base_s and its immediate IEEE-754 float neighbors (+/- 1 ULP) against exact pla_predict_seg.
    float candidates[3] = {
        base_s,
        nextafterf(base_s, INFINITY),
        (base_s > 0.0f) ? nextafterf(base_s, 0.0f) : 0.0f
    };

    size_t best_fail = i0;
    for (int c = 0; c < 3; ++c) {
        pla_seg_t cand = { .x_start = x0, .intercept = (uint32_t)ic, .slope = candidates[c] };
        size_t bad = i0;
        if (verify_seg_on_range(&cand, keys, i0, i1, n_keys, eps, &bad)) {
            *out_seg = cand;
            return 1;
        }
        if (bad > best_fail) best_fail = bad;
    }

    if (first_bad) *first_bad = best_fail;
    return 0;
}

static int build_pla_segments(
    const uint64_t *keys,
    size_t n_keys,
    uint32_t eps,
    pla_seg_t **out_segs,
    size_t *out_n_segs
) {
    if (n_keys == 0) {
        *out_segs = NULL;
        *out_n_segs = 0;
        return 0;
    }
    if (eps < 1U) eps = 1U;

    size_t seg_cap = (n_keys / (2U * (size_t)eps + 1U)) + 64U;
    if (seg_cap < 64U) seg_cap = 64U;
    pla_seg_t *segs = (pla_seg_t *)malloc(seg_cap * sizeof(pla_seg_t));
    if (!segs) return -1;

    if (n_keys == 1) {
        segs[0].x_start   = keys[0];
        segs[0].intercept = 0U;
        segs[0].slope     = 0.0f;
        *out_segs   = segs;
        *out_n_segs = 1U;
        return 0;
    }

    size_t hull_cap = 4096U;
    hull_pt_t *U = (hull_pt_t *)malloc(hull_cap * sizeof(hull_pt_t));
    hull_pt_t *L = (hull_pt_t *)malloc(hull_cap * sizeof(hull_pt_t));
    if (!U || !L) {
        free(U);
        free(L);
        free(segs);
        return -1;
    }

    const int64_t SCALE = 64; // exact power-of-two vertical scale
    const int64_t up_off = (int64_t)eps * SCALE - 1;
    const int64_t lo_off = -(int64_t)eps * SCALE + 1;

    size_t m = 0;
    size_t i0 = 0;

    while (i0 < n_keys) {
        if (m == seg_cap) {
            seg_cap *= 2U;
            pla_seg_t *ns = (pla_seg_t *)realloc(segs, seg_cap * sizeof(pla_seg_t));
            if (!ns) {
                free(U); free(L); free(segs);
                return -1;
            }
            segs = ns;
        }

        if (i0 + 1U >= n_keys) {
            segs[m++] = (pla_seg_t){ .x_start = keys[i0], .intercept = (uint32_t)i0, .slope = 0.0f };
            break;
        }

        uint64_t x0 = keys[i0];
        U[0] = (hull_pt_t){ 0ULL, up_off };
        L[0] = (hull_pt_t){ 0ULL, lo_off };

        uint64_t dx1 = keys[i0 + 1U] - x0;
        hull_pt_t u1 = { dx1, SCALE + up_off };
        hull_pt_t l1 = { dx1, SCALE + lo_off };

        hull_slope_t s_min = pt_slope(U[0], l1);
        if (s_min.dy < 0) {
            s_min.dx = 1ULL;
            s_min.dy = 0;
        }
        hull_slope_t s_max = pt_slope(L[0], u1);

        U[1] = u1;
        L[1] = l1;
        size_t u_len = 2U, l_len = 2U;
        size_t u_start = 0U, l_start = 0U;
        size_t i1 = i0 + 1U;

        for (size_t k = i0 + 2U; k < n_keys; ++k) {
            uint64_t dx = keys[k] - x0;
            int64_t rel = (int64_t)(k - i0) * SCALE;
            hull_pt_t uk = { dx, rel + up_off };
            hull_pt_t lk = { dx, rel + lo_off };

            size_t u_cand = u_start;
            hull_slope_t cand_min = pt_slope(U[u_cand], lk);
            while (u_cand + 1U < u_len) {
                hull_slope_t nxt = pt_slope(U[u_cand + 1U], lk);
                if (slope_gt(nxt, cand_min)) {
                    u_cand++;
                    cand_min = nxt;
                } else {
                    break;
                }
            }

            size_t l_cand = l_start;
            hull_slope_t cand_max = pt_slope(L[l_cand], uk);
            while (l_cand + 1U < l_len) {
                hull_slope_t nxt = pt_slope(L[l_cand + 1U], uk);
                if (slope_lt(nxt, cand_max)) {
                    l_cand++;
                    cand_max = nxt;
                } else {
                    break;
                }
            }

            hull_slope_t new_min = slope_gt(cand_min, s_min) ? cand_min : s_min;
            hull_slope_t new_max = slope_lt(cand_max, s_max) ? cand_max : s_max;
            if (slope_gt(new_min, new_max)) {
                break; // shrinking cone empty: optimal O'Rourke segment ends at k - 1
            }

            if (slope_gt(cand_min, s_min)) {
                s_min = cand_min;
                u_start = u_cand;
            }
            if (slope_lt(cand_max, s_max)) {
                s_max = cand_max;
                l_start = l_cand;
            }

            if (u_len + 1U >= hull_cap || l_len + 1U >= hull_cap) {
                hull_cap *= 2U;
                hull_pt_t *nU = (hull_pt_t *)realloc(U, hull_cap * sizeof(hull_pt_t));
                hull_pt_t *nL = (hull_pt_t *)realloc(L, hull_cap * sizeof(hull_pt_t));
                if (!nU || !nL) {
                    free(nU ? nU : U);
                    free(nL ? nL : L);
                    free(segs);
                    return -1;
                }
                U = nU;
                L = nL;
            }

            while (u_len > u_start + 1U &&
                   !slope_lt(pt_slope(U[u_len - 2U], U[u_len - 1U]), pt_slope(U[u_len - 1U], uk))) {
                u_len--;
            }
            U[u_len++] = uk;

            while (l_len > l_start + 1U &&
                   !slope_gt(pt_slope(L[l_len - 2U], L[l_len - 1U]), pt_slope(L[l_len - 1U], lk))) {
                l_len--;
            }
            L[l_len++] = lk;

            i1 = k;
        }

        double a_lo   = (double)s_min.dy / ((double)SCALE * (double)s_min.dx);
        double a_hi   = (double)s_max.dy / ((double)SCALE * (double)s_max.dx);
        double a_hint = 0.5 * (a_lo + a_hi);

        // Fit quantized pla_seg_t and guarantee exact C runtime tube via fallback split if float drifts.
        size_t cur_end = i1;
        pla_seg_t fitted;
        while (1) {
            size_t bad_at = i0;
            if (fit_quantized_seg(keys, i0, cur_end, n_keys, eps, a_hint, &fitted, &bad_at)) {
                break;
            }
            // If float precision drifted on a huge segment, split cleanly before first bad index.
            size_t safe_min = i0 + 2U * (size_t)eps;
            if (safe_min >= n_keys) safe_min = n_keys - 1U;
            if (bad_at > i0 && bad_at - 1U >= safe_min) {
                cur_end = bad_at - 1U;
            } else {
                cur_end = safe_min;
            }
            if (cur_end > i0) {
                a_hint = (double)(cur_end - i0) / (double)(keys[cur_end] - x0);
            } else {
                a_hint = 0.0;
            }
        }

        segs[m++] = fitted;
        i0 = cur_end + 1U;
    }

    free(U);
    free(L);
    *out_segs   = segs;
    *out_n_segs = m;
    return 0;
}

static void build_radix16(const pla_seg_t *segs, size_t n_segs, uint32_t *radix16) {
    if (n_segs == 0) {
        memset(radix16, 0, PLA_RADIX_LEN * sizeof(uint32_t));
        return;
    }
    size_t s = 0;
    for (uint32_t b = 0; b < PLA_RADIX_BUCKETS; ++b) {
        while (s < n_segs && (uint32_t)(segs[s].x_start >> 48) < b) {
            s++;
        }
        radix16[b] = (uint32_t)s;
    }
    radix16[PLA_RADIX_BUCKETS] = (uint32_t)n_segs;
}

// -----------------------------------------------------------------------------------------------
// 3. Full Index Build, Verification, Stats, and Zero-Deserialization mmap I/O
// -----------------------------------------------------------------------------------------------

int pla_build(
    pla_index_t *out,
    const uint64_t *raw_keys,
    const uint8_t *raw_flags,
    size_t n_raw,
    uint32_t eps,
    int use_perm,
    uint32_t k,
    uint32_t s,
    int num_threads
) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    if (eps < 1U) eps = 1U;
    if (n_raw > (size_t)UINT32_MAX) return -2;

    uint64_t *work_keys = NULL;
    uint8_t *packed_flags = NULL;
    size_t n_unique = 0;
    int T = effective_threads(num_threads);

    if (n_raw > 0) {
        if (!raw_keys) return -1;
        if (!raw_flags) {
            work_keys = (uint64_t *)malloc(n_raw * sizeof(uint64_t));
            if (!work_keys) return -1;
            if (use_perm & 1) {
                #pragma omp parallel for num_threads(T) schedule(static)
                for (size_t i = 0; i < n_raw; ++i) {
                    work_keys[i] = splitmix64_bijective(raw_keys[i]);
                }
            } else {
                memcpy(work_keys, raw_keys, n_raw * sizeof(uint64_t));
            }
            n_unique = pla_sort_dedup_u64(work_keys, n_raw, T);
            if (n_unique == 0) {
                free(work_keys);
                return -1;
            }
        } else {
            key_flag_t *kf = (key_flag_t *)malloc(n_raw * sizeof(key_flag_t));
            if (!kf) return -1;
            int perm_k = (use_perm & 1) ? 1 : 0;
            for (size_t i = 0; i < n_raw; ++i) {
                kf[i].k = perm_k ? splitmix64_bijective(raw_keys[i]) : raw_keys[i];
                kf[i].f = (uint8_t)(raw_flags[i] & 3U);
            }
            qsort(kf, n_raw, sizeof(key_flag_t), cmp_key_flag);
            size_t w = 0;
            for (size_t r = 0; r < n_raw; ++r) {
                if (w > 0 && kf[r].k == kf[w - 1U].k) {
                    kf[w - 1U].f |= kf[r].f;
                } else {
                    kf[w++] = kf[r];
                }
            }
            n_unique = w;
            work_keys = (uint64_t *)malloc(n_unique * sizeof(uint64_t));
            size_t fb_len = (n_unique + 3U) >> 2;
            packed_flags = (uint8_t *)calloc(fb_len ? fb_len : 1U, 1U);
            if (!work_keys || !packed_flags) {
                free(kf); free(work_keys); free(packed_flags);
                return -1;
            }
            for (size_t i = 0; i < n_unique; ++i) {
                work_keys[i] = kf[i].k;
                packed_flags[i >> 2] |= (uint8_t)((kf[i].f & 3U) << ((i & 3U) * 2U));
            }
            free(kf);
        }
    }

    pla_seg_t *tmp_segs = NULL;
    size_t n_segs = 0;
    if (build_pla_segments(work_keys, n_unique, eps, &tmp_segs, &n_segs) != 0) {
        free(work_keys);
        free(packed_flags);
        return -1;
    }

    // Assemble contiguous 128-byte aligned zero-deserialization mmap image in memory.
    size_t off_radix16 = sizeof(pla_mmap_hdr_t); // 128
    size_t off_segs    = align_up(off_radix16 + PLA_RADIX_LEN * sizeof(uint32_t), 128U);
    size_t off_keys    = align_up(off_segs + n_segs * sizeof(pla_seg_t), 128U);
    size_t keys_bytes  = n_unique * sizeof(uint64_t);
    size_t off_flags2b = 0;
    size_t total_bytes = align_up(off_keys + keys_bytes, 128U);

    uint32_t hdr_flags = 0U;
    if (use_perm & 1) hdr_flags |= PLA_FLAG_PERMUTED;
    if (use_perm & 2) hdr_flags |= PLA_FLAG_SPLITMIX_SYNCMER;
    if (raw_flags != NULL && n_unique > 0) {
        hdr_flags |= PLA_FLAG_HAS_FLAGS2B;
        off_flags2b = total_bytes;
        size_t fb_bytes = (n_unique + 3U) >> 2;
        total_bytes = align_up(off_flags2b + fb_bytes, 128U);
    }

    void *buf = NULL;
    if (posix_memalign(&buf, 128U, total_bytes) != 0 || !buf) {
        free(tmp_segs);
        free(work_keys);
        free(packed_flags);
        return -1;
    }
    memset(buf, 0, total_bytes);

    uint8_t *base = (uint8_t *)buf;
    pla_mmap_hdr_t *hdr = (pla_mmap_hdr_t *)base;
    hdr->magic       = PLA_MAGIC;
    hdr->version     = PLA_VERSION;
    hdr->eps         = eps;
    hdr->k           = k;
    hdr->s           = s;
    hdr->flags       = hdr_flags;
    hdr->n_keys      = (uint64_t)n_unique;
    hdr->n_segs      = (uint64_t)n_segs;
    hdr->off_radix16 = (uint64_t)off_radix16;
    hdr->off_segs    = (uint64_t)off_segs;
    hdr->off_keys    = (uint64_t)off_keys;
    hdr->off_flags2b = (uint64_t)off_flags2b;
    hdr->total_bytes = (uint64_t)total_bytes;

    uint32_t *r16   = (uint32_t *)(base + off_radix16);
    pla_seg_t *segs = (pla_seg_t *)(base + off_segs);
    uint64_t *pkeys = (uint64_t *)(base + off_keys);

    if (n_segs > 0) {
        memcpy(segs, tmp_segs, n_segs * sizeof(pla_seg_t));
    }
    build_radix16(segs, n_segs, r16);
    if (n_unique > 0) {
        memcpy(pkeys, work_keys, keys_bytes);
    }
    uint8_t *f2b = NULL;
    if (off_flags2b > 0) {
        f2b = base + off_flags2b;
        memcpy(f2b, packed_flags, (n_unique + 3U) >> 2);
    }

    free(tmp_segs);
    free(work_keys);
    free(packed_flags);

    out->eps           = eps;
    out->k             = k;
    out->s             = s;
    out->flags         = hdr_flags;
    out->n_keys        = n_unique;
    out->n_segs        = n_segs;
    out->radix16       = r16;
    out->segs          = segs;
    out->perm_keys     = pkeys;
    out->flags2b       = f2b;
    out->backing       = buf;
    out->backing_bytes = total_bytes;
    out->is_mmap       = 0;
    out->fd            = -1;

    // Mandatory post-build full verification across ALL N keys!
    if (pla_verify_all(out, NULL, T) != 0) {
        pla_close(out);
        return -3;
    }
    return 0;
}

int pla_verify_all(const pla_index_t *idx, int64_t *out_max_abs_err, int num_threads) {
    if (!idx) return -1;
    if (idx->n_keys == 0) {
        if (out_max_abs_err) *out_max_abs_err = 0;
        return 0;
    }
    if (!idx->radix16 || !idx->segs || !idx->perm_keys || idx->n_segs == 0) return -1;

    const size_t N   = idx->n_keys;
    const int64_t e  = (int64_t)idx->eps;
    const int uperm  = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    int T = (N < 65536U) ? 1 : effective_threads(num_threads);

    int failed = 0;
    int64_t global_max_err = 0;

    #pragma omp parallel num_threads(T)
    {
        int local_fail = 0;
        int64_t local_max = 0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            uint64_t x = idx->perm_keys[i];
            if (i > 0 && x <= idx->perm_keys[i - 1U]) {
                local_fail = 1;
            }
            uint32_t seg_id = pla_find_seg(idx->radix16, idx->segs, x);
            if (seg_id >= idx->n_segs) {
                local_fail = 2;
                continue;
            }
            int64_t p_hat = pla_predict_seg(&idx->segs[seg_id], x, N);
            int64_t err   = p_hat - (int64_t)i;
            int64_t aerr  = (err < 0) ? -err : err;
            if (aerr > local_max) local_max = aerr;
            if (aerr > e) {
                local_fail = 3;
            }
            uint64_t raw_q = uperm ? splitmix64_inverse(x) : x;
            int64_t found  = pla_lookup_inline(
                raw_q, idx->perm_keys, idx->segs, idx->radix16, N, idx->eps, uperm
            );
            if (found != (int64_t)i) {
                local_fail = 4;
            }
        }

        #pragma omp critical
        {
            if (local_fail && !failed) failed = local_fail;
            if (local_max > global_max_err) global_max_err = local_max;
        }
    }

    if (out_max_abs_err) *out_max_abs_err = global_max_err;
    return failed ? -failed : 0;
}

void pla_compute_stats(const pla_index_t *idx, pla_stats_t *st) {
    if (!idx || !st) return;
    memset(st, 0, sizeof(*st));
    st->n_keys   = (uint64_t)idx->n_keys;
    st->n_segs   = (uint64_t)idx->n_segs;
    st->eps      = idx->eps;
    st->use_perm = (idx->flags & PLA_FLAG_PERMUTED) ? 1U : 0U;

    uint64_t max_segs_b = 0;
    for (uint32_t b = 0; b < PLA_RADIX_BUCKETS; ++b) {
        uint64_t sb = (uint64_t)(idx->radix16[b + 1U] - idx->radix16[b]);
        if (sb > max_segs_b) max_segs_b = sb;
    }
    st->max_segs_per_bucket = max_segs_b;

    uint64_t *bcounts = (uint64_t *)calloc(PLA_RADIX_BUCKETS, sizeof(uint64_t));
    if (bcounts) {
        for (size_t i = 0; i < idx->n_keys; ++i) {
            bcounts[idx->perm_keys[i] >> 48]++;
        }
        uint64_t empty_b = 0, max_kb = 0;
        for (uint32_t b = 0; b < PLA_RADIX_BUCKETS; ++b) {
            if (bcounts[b] == 0) empty_b++;
            if (bcounts[b] > max_kb) max_kb = bcounts[b];
        }
        st->empty_buckets       = empty_b;
        st->max_keys_per_bucket = max_kb;
        free(bcounts);
    }

    st->mean_keys_per_seg = (idx->n_segs > 0) ? ((double)idx->n_keys / (double)idx->n_segs) : 0.0;
    st->keys_bytes        = (uint64_t)idx->n_keys * sizeof(uint64_t);
    st->index_overhead_bytes = (uint64_t)sizeof(pla_mmap_hdr_t)
                             + (uint64_t)PLA_RADIX_LEN * sizeof(uint32_t)
                             + (uint64_t)idx->n_segs * sizeof(pla_seg_t);
    st->total_bytes       = (uint64_t)idx->backing_bytes;
    if (idx->n_keys > 0) {
        st->overhead_bytes_per_key = (double)st->index_overhead_bytes / (double)idx->n_keys;
        st->total_bytes_per_key    = (double)st->total_bytes / (double)idx->n_keys;
    }
    int64_t merr = 0;
    pla_verify_all(idx, &merr, 0);
    st->max_abs_err = merr;
}

int pla_save_mmap(const pla_index_t *idx, const char *path) {
    if (!idx || !idx->backing || !path) return -1;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;

    const uint8_t *p = (const uint8_t *)idx->backing;
    size_t rem = idx->backing_bytes;
    while (rem > 0) {
        size_t chunk = (rem > (1U << 26)) ? (1U << 26) : rem;
        ssize_t nw = write(fd, p, chunk);
        if (nw <= 0) {
            close(fd);
            return -1;
        }
        p   += (size_t)nw;
        rem -= (size_t)nw;
    }
    close(fd);
    return 0;
}

int pla_open_mmap(pla_index_t *out, const char *path, int map_private, int prefault_spline) {
    if (!out || !path) return -1;
    memset(out, 0, sizeof(*out));
    out->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(pla_mmap_hdr_t)) {
        close(fd);
        return -1;
    }

    size_t fsize = (size_t)st.st_size;
    int mflags = map_private ? MAP_PRIVATE : MAP_SHARED;
    void *map = mmap(NULL, fsize, PROT_READ, mflags, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -1;
    }

    const uint8_t *base = (const uint8_t *)map;
    const pla_mmap_hdr_t *hdr = (const pla_mmap_hdr_t *)base;
    if (hdr->magic != PLA_MAGIC || hdr->version != PLA_VERSION ||
        hdr->total_bytes < sizeof(pla_mmap_hdr_t) || hdr->total_bytes > fsize) {
        munmap(map, fsize);
        close(fd);
        return -2;
    }

    // Validate section alignments (4B radix16, 8B segs, 64B perm_keys) without UB.
    if ((hdr->off_radix16 & 3ULL) || (hdr->off_segs & 7ULL) || (hdr->off_keys & 63ULL)) {
        munmap(map, fsize);
        close(fd);
        return -3;
    }

    // Overflow-safe bounds checks across ALL sections (including optional flags2b) + structural invariants.
    uint64_t fsz64 = (uint64_t)fsize;
    uint64_t radix_bytes = (uint64_t)PLA_RADIX_LEN * sizeof(uint32_t);
    uint64_t flags_bytes = (hdr->n_keys + 3ULL) >> 2;
    int bad_bounds = 0;
    if (hdr->n_keys > 0 && hdr->n_segs == 0) bad_bounds = 1;
    if (hdr->off_radix16 > fsz64 || radix_bytes > fsz64 - hdr->off_radix16) bad_bounds = 1;
    if (hdr->off_segs > fsz64 || hdr->n_segs > (fsz64 - hdr->off_segs) / sizeof(pla_seg_t)) bad_bounds = 1;
    if (hdr->off_keys > fsz64 || hdr->n_keys > (fsz64 - hdr->off_keys) / sizeof(uint64_t)) bad_bounds = 1;
    if (hdr->off_flags2b > 0) {
        if (hdr->off_flags2b > fsz64 || flags_bytes > fsz64 - hdr->off_flags2b) bad_bounds = 1;
    }
    if (bad_bounds) {
        munmap(map, fsize);
        close(fd);
        return -4;
    }

    const uint32_t *r16 = (const uint32_t *)(base + hdr->off_radix16);
    if (r16[PLA_RADIX_BUCKETS] > hdr->n_segs) {
        munmap(map, fsize);
        close(fd);
        return -4;
    }
    for (uint32_t b = 0; b < PLA_RADIX_BUCKETS; ++b) {
        if (r16[b] > r16[b + 1U]) {
            munmap(map, fsize);
            close(fd);
            return -4;
        }
    }

    out->eps           = hdr->eps;
    out->k             = hdr->k;
    out->s             = hdr->s;
    out->flags         = hdr->flags;
    out->n_keys        = (size_t)hdr->n_keys;
    out->n_segs        = (size_t)hdr->n_segs;
    out->radix16       = r16;
    out->segs          = (const pla_seg_t *)(base + hdr->off_segs);
    out->perm_keys     = (const uint64_t *)(base + hdr->off_keys);
    out->flags2b       = (hdr->off_flags2b > 0) ? (base + hdr->off_flags2b) : NULL;
    out->backing       = map;
    out->backing_bytes = fsize;
    out->is_mmap       = 1;
    out->fd            = fd;

    if (prefault_spline && hdr->off_keys > 0) {
        madvise(map, (size_t)hdr->off_keys, MADV_WILLNEED);
        volatile uint64_t touch = 0;
        for (size_t off = 0; off < (size_t)hdr->off_keys; off += 4096U) {
            touch += base[off];
        }
        (void)touch;
    }
    return 0;
}

void pla_close(pla_index_t *idx) {
    if (!idx) return;
    if (idx->backing) {
        if (idx->is_mmap) {
            munmap(idx->backing, idx->backing_bytes);
            if (idx->fd >= 0) close(idx->fd);
        } else {
            free(idx->backing);
        }
    }
    memset(idx, 0, sizeof(*idx));
    idx->fd = -1;
}

// -----------------------------------------------------------------------------------------------
// 4. Exported Query & Bijection Batch Functions
// -----------------------------------------------------------------------------------------------

int64_t pla_lookup(const pla_index_t *idx, uint64_t query_key) {
    if (!idx || idx->n_keys == 0) return -1;
    int uperm = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    return pla_lookup_inline(
        query_key, idx->perm_keys, idx->segs, idx->radix16, idx->n_keys, idx->eps, uperm
    );
}

void pla_lookup_batch(
    const pla_index_t *idx,
    const uint64_t *queries,
    int64_t *out_ranks,
    size_t n_queries,
    int num_threads
) {
    if (!idx || !queries || !out_ranks || n_queries == 0) return;
    if (idx->n_keys == 0) {
        for (size_t i = 0; i < n_queries; ++i) out_ranks[i] = -1;
        return;
    }
    const uint64_t * restrict pkeys = idx->perm_keys;
    const pla_seg_t * restrict segs = idx->segs;
    const uint32_t * restrict r16   = idx->radix16;
    const size_t N     = idx->n_keys;
    const uint32_t eps = idx->eps;
    const int uperm    = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    int T = (n_queries < 16384U) ? 1 : effective_threads(num_threads);

    #pragma omp parallel for num_threads(T) schedule(static)
    for (size_t i = 0; i < n_queries; ++i) {
        out_ranks[i] = pla_lookup_inline(queries[i], pkeys, segs, r16, N, eps, uperm);
    }
}

uint64_t pla_count_hits_batch(
    const pla_index_t *idx,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
) {
    if (!idx || !queries || n_queries == 0 || idx->n_keys == 0) return 0;
    const uint64_t * restrict pkeys = idx->perm_keys;
    const pla_seg_t * restrict segs = idx->segs;
    const uint32_t * restrict r16   = idx->radix16;
    const size_t N     = idx->n_keys;
    const uint32_t eps = idx->eps;
    const int uperm    = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    int T = effective_threads(num_threads);

    uint64_t hits = 0;
    if (uperm) {
        #pragma omp parallel for reduction(+:hits) num_threads(T) schedule(static)
        for (size_t i = 0; i < n_queries; ++i) {
            if (pla_lookup_inline(queries[i], pkeys, segs, r16, N, eps, 1) >= 0) {
                hits++;
            }
        }
    } else {
        #pragma omp parallel for reduction(+:hits) num_threads(T) schedule(static)
        for (size_t i = 0; i < n_queries; ++i) {
            if (pla_lookup_inline(queries[i], pkeys, segs, r16, N, eps, 0) >= 0) {
                hits++;
            }
        }
    }
    return hits;
}

void pla_splitmix64_forward_batch(const uint64_t *in, uint64_t *out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = splitmix64_bijective(in[i]);
}

void pla_splitmix64_inverse_batch(const uint64_t *in, uint64_t *out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = splitmix64_inverse(in[i]);
}

// -----------------------------------------------------------------------------------------------
// 5. Baselines: Branchless/Standard Binary Search & Verbatim glcore.c htab (lines 55-96)
// -----------------------------------------------------------------------------------------------

static inline int64_t bsearch_u64_inline(const uint64_t * restrict arr, size_t n, uint64_t x) {
    if (__builtin_expect(n == 0, 0)) return -1;
    const uint64_t *base = arr;
    size_t len = n;
    while (len > 1U) {
        size_t half = len >> 1;
        base = (base[half] <= x) ? (base + half) : base;
        len -= half;
    }
    return (*base == x) ? (int64_t)(base - arr) : -1;
}

int64_t pla_bsearch_lookup(const pla_index_t *idx, uint64_t query_key) {
    if (!idx || idx->n_keys == 0) return -1;
    uint64_t x = (idx->flags & PLA_FLAG_PERMUTED) ? splitmix64_bijective(query_key) : query_key;
    return bsearch_u64_inline(idx->perm_keys, idx->n_keys, x);
}

uint64_t pla_bsearch_count_hits_batch(
    const pla_index_t *idx,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
) {
    if (!idx || !queries || n_queries == 0 || idx->n_keys == 0) return 0;
    const uint64_t * restrict pkeys = idx->perm_keys;
    const size_t N  = idx->n_keys;
    const int uperm = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    int T = effective_threads(num_threads);

    uint64_t hits = 0;
    if (uperm) {
        #pragma omp parallel for reduction(+:hits) num_threads(T) schedule(static)
        for (size_t i = 0; i < n_queries; ++i) {
            uint64_t x = splitmix64_bijective(queries[i]);
            if (bsearch_u64_inline(pkeys, N, x) >= 0) hits++;
        }
    } else {
        #pragma omp parallel for reduction(+:hits) num_threads(T) schedule(static)
        for (size_t i = 0; i < n_queries; ++i) {
            if (bsearch_u64_inline(pkeys, N, queries[i]) >= 0) hits++;
        }
    }
    return hits;
}

// Verbatim reproduction of glcore.c lines 55-96 & 275-286:
#define GL_EMPTY_KEY UINT64_MAX

static inline uint64_t gl_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

void gl_ht_init(gl_htab_t *h, uint64_t want) {
    uint64_t c = 1024ULL;
    while (c < want * 2ULL + 16ULL) c <<= 1;
    h->cap  = c;
    h->mask = c - 1ULL;
    h->n    = 0ULL;
    h->keys = (uint64_t *)malloc(c * sizeof(uint64_t));
    h->vals = (uint32_t *)calloc(c, sizeof(uint32_t));
    if (!h->keys || !h->vals) {
        fprintf(stderr, "gl_ht_init: oom\n");
        abort();
    }
    memset(h->keys, 0xff, c * sizeof(uint64_t));
}

void gl_ht_free(gl_htab_t *h) {
    if (!h) return;
    free(h->keys);
    free(h->vals);
    memset(h, 0, sizeof(*h));
}

static void gl_ht_grow(gl_htab_t *h);

static inline uint64_t gl_ht_insert_inline(gl_htab_t *h, uint64_t k) {
    if ((h->n + 1ULL) * 2ULL > h->cap) gl_ht_grow(h);
    uint64_t i = gl_mix64(k) & h->mask;
    while (1) {
        uint64_t kk = h->keys[i];
        if (kk == k) return i;
        if (kk == GL_EMPTY_KEY) {
            h->keys[i] = k;
            h->vals[i] = 0U;
            h->n++;
            return i;
        }
        i = (i + 1ULL) & h->mask;
    }
}

static void gl_ht_grow(gl_htab_t *h) {
    gl_htab_t n2;
    gl_ht_init(&n2, h->cap);
    for (uint64_t i = 0; i < h->cap; ++i) {
        if (h->keys[i] != GL_EMPTY_KEY) {
            uint64_t j = gl_ht_insert_inline(&n2, h->keys[i]);
            n2.vals[j] = h->vals[i];
        }
    }
    gl_ht_free(h);
    *h = n2;
}

uint64_t gl_ht_insert(gl_htab_t *h, uint64_t k, uint32_t val_or) {
    uint64_t j = gl_ht_insert_inline(h, k);
    h->vals[j] |= val_or;
    return j;
}

static inline uint64_t gl_ht_find_inline(const gl_htab_t *h, uint64_t k) {
    uint64_t i = gl_mix64(k) & h->mask;
    while (1) {
        uint64_t kk = h->keys[i];
        if (kk == k) return i;
        if (kk == GL_EMPTY_KEY) return h->cap;
        i = (i + 1ULL) & h->mask;
    }
}

uint64_t gl_ht_find(const gl_htab_t *h, uint64_t k) {
    return gl_ht_find_inline(h, k);
}

int gl_ht_save_table(const gl_htab_t *h, const char *path, int k, int s) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    static char wbuf[1 << 20];
    setvbuf(f, wbuf, _IOFBF, sizeof(wbuf));
    uint64_t hdr[4] = { 0x31544341ULL, (uint64_t)k, (uint64_t)s, h->n };
    if (fwrite(hdr, sizeof(uint64_t), 4, f) != 4) { fclose(f); return -1; }
    for (uint64_t i = 0; i < h->cap; ++i) {
        if (h->keys[i] != GL_EMPTY_KEY) {
            if (fwrite(&h->keys[i], sizeof(uint64_t), 1, f) != 1 ||
                fwrite(&h->vals[i], sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return -1;
            }
        }
    }
    fclose(f);
    return 0;
}

// Verbatim glcore.c load_table (lines 275-286):
int gl_ht_load_table(const char *path, gl_htab_t *h, int *k, int *s) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint64_t hdr[4];
    if (fread(hdr, sizeof(uint64_t), 4, f) != 4) { fclose(f); return -1; }
    if (k) *k = (int)hdr[1];
    if (s) *s = (int)hdr[2];
    gl_ht_init(h, hdr[3] + 16ULL);
    for (uint64_t i = 0; i < hdr[3]; ++i) {
        uint64_t key;
        uint32_t v;
        if (fread(&key, sizeof(uint64_t), 1, f) != 1 ||
            fread(&v, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            return -1;
        }
        uint64_t j = gl_ht_insert_inline(h, key);
        h->vals[j] = v;
    }
    fclose(f);
    return 0;
}

uint64_t gl_ht_count_hits_batch(
    const gl_htab_t *h,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
) {
    if (!h || !h->keys || !queries || n_queries == 0) return 0;
    int T = effective_threads(num_threads);
    uint64_t hits = 0;
    #pragma omp parallel for reduction(+:hits) num_threads(T) schedule(static)
    for (size_t i = 0; i < n_queries; ++i) {
        if (gl_ht_find_inline(h, queries[i]) != h->cap) hits++;
    }
    return hits;
}

// Verbatim malariavar/kernels.py (lines 20-70) keys-only open-addressed uint64_t hash set (EMPTY=0, key+1):
void mv_kset_init(mv_kset_t *ks, uint64_t want) {
    uint64_t c = 1024ULL;
    while (c < want * 2ULL + 16ULL) c <<= 1;
    ks->cap  = c;
    ks->mask = c - 1ULL;
    ks->n    = 0ULL;
    ks->tab  = (uint64_t *)calloc(c, sizeof(uint64_t));
    if (!ks->tab) {
        fprintf(stderr, "mv_kset_init: oom\n");
        abort();
    }
}

void mv_kset_free(mv_kset_t *ks) {
    if (!ks) return;
    free(ks->tab);
    memset(ks, 0, sizeof(*ks));
}

static inline int mv_kset_insert_inline(mv_kset_t *ks, uint64_t key) {
    uint64_t h  = splitmix64_bijective(key) & ks->mask;
    uint64_t k1 = key + 1ULL;
    while (1) {
        uint64_t v = ks->tab[h];
        if (v == 0ULL) {
            ks->tab[h] = k1;
            ks->n++;
            return 1;
        }
        if (v == k1) return 0;
        h = (h + 1ULL) & ks->mask;
    }
}

int mv_kset_insert(mv_kset_t *ks, uint64_t key) {
    return mv_kset_insert_inline(ks, key);
}

static inline int mv_kset_has_inline(const mv_kset_t *ks, uint64_t key) {
    uint64_t h  = splitmix64_bijective(key) & ks->mask;
    uint64_t k1 = key + 1ULL;
    while (1) {
        uint64_t v = ks->tab[h];
        if (v == k1) return 1;
        if (v == 0ULL) return 0;
        h = (h + 1ULL) & ks->mask;
    }
}

int mv_kset_has(const mv_kset_t *ks, uint64_t key) {
    return mv_kset_has_inline(ks, key);
}

uint64_t mv_kset_count_hits_batch(
    const mv_kset_t *ks,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
) {
    if (!ks || !ks->tab || !queries || n_queries == 0) return 0;
    int T = effective_threads(num_threads);
    uint64_t hits = 0;
    #pragma omp parallel for reduction(+:hits) num_threads(T) schedule(static)
    for (size_t i = 0; i < n_queries; ++i) {
        if (mv_kset_has_inline(ks, queries[i])) hits++;
    }
    return hits;
}

// -----------------------------------------------------------------------------------------------
// 6. Verbatim glcore.c Open Syncmer Extraction & Paired-Read Sieve Parity
// -----------------------------------------------------------------------------------------------

static inline void build_nt_table(uint8_t nt[256], int skip_lower) {
    memset(nt, 4, 256);
    nt['A'] = 0; nt['C'] = 1; nt['G'] = 2; nt['T'] = 3;
    if (!skip_lower) {
        nt['a'] = 0; nt['c'] = 1; nt['g'] = 2; nt['t'] = 3;
    }
}

static inline int pla_dust_score_win(const uint8_t *enc_ring, size_t start, int k) {
    int cnt[64] = {0};
    int s = 0;
    for (int j = 0; j < k - 2; ++j) {
        uint8_t a = enc_ring[(start + (size_t)j) & 31U];
        uint8_t b = enc_ring[(start + (size_t)j + 1U) & 31U];
        uint8_t c = enc_ring[(start + (size_t)j + 2U) & 31U];
        if (a > 3U || b > 3U || c > 3U) continue;
        int t = (int)a * 16 + (int)b * 4 + (int)c;
        s += cnt[t];
        cnt[t]++;
    }
    return s;
}

size_t pla_extract_open_syncmers(
    const char *seq,
    size_t len,
    int k,
    int s,
    int skip_lower,
    int use_splitmix,
    int dust_max,
    uint64_t *out_keys,
    size_t max_out
) {
    if (!seq || (int)len < k || s > k || k < 1 || k > 32 || (k - s + 1) > 32) return 0;
    uint8_t nt[256];
    build_nt_table(nt, skip_lower);

    const int m = k - s + 1;
    const int t = (k - s) / 2;
    const uint64_t kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const uint64_t smask = (s == 32) ? ~0ULL : ((1ULL << (2 * s)) - 1ULL);

    uint64_t fs_ring[32], rs_ring[32], csh_ring[32];
    uint8_t enc_ring[32];
    uint64_t fs = 0, rs = 0, fk = 0, rk = 0;
    long run = 0;
    size_t count = 0;

    for (size_t i = 0; i < len; ++i) {
        uint8_t c = nt[(uint8_t)seq[i]];
        if (c > 3U) {
            run = 0;
            fs = rs = fk = rk = 0;
            continue;
        }
        enc_ring[i & 31U] = c;
        run++;
        fs = ((fs << 2) | c) & smask;
        rs = (rs >> 2) | ((uint64_t)(3U - c) << (2 * (s - 1)));
        fk = ((fk << 2) | c) & kmask;
        rk = (rk >> 2) | ((uint64_t)(3U - c) << (2 * (k - 1)));

        if (run >= s) {
            size_t spos = i - (size_t)s + 1U;
            if (use_splitmix) {
                uint64_t cs = (fs < rs) ? fs : rs;
                csh_ring[spos & 31U] = splitmix64_bijective(cs);
            } else {
                fs_ring[spos & 31U] = fs;
                rs_ring[spos & 31U] = rs;
            }
        }
        if (run >= k) {
            size_t p = i - (size_t)k + 1U;
            int best = 0;
            uint64_t bv;
            uint64_t emit_k = (fk < rk) ? fk : rk;
            if (use_splitmix) {
                if (fk < rk) {
                    bv = csh_ring[p & 31U];
                    for (int j = 1; j < m; ++j) {
                        uint64_t v = csh_ring[(p + (size_t)j) & 31U];
                        if (v < bv) { bv = v; best = j; }
                    }
                } else {
                    bv = csh_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) {
                        uint64_t v = csh_ring[(p + (size_t)(m - 1 - j)) & 31U];
                        if (v < bv) { bv = v; best = j; }
                    }
                }
            } else {
                if (fk < rk) {
                    bv = fs_ring[p & 31U];
                    for (int j = 1; j < m; ++j) {
                        uint64_t v = fs_ring[(p + (size_t)j) & 31U];
                        if (v < bv) { bv = v; best = j; }
                    }
                } else {
                    bv = rs_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) {
                        uint64_t v = rs_ring[(p + (size_t)(m - 1 - j)) & 31U];
                        if (v < bv) { bv = v; best = j; }
                    }
                }
            }
            if (best == t) {
                if (dust_max >= 0 && pla_dust_score_win(enc_ring, p, k) > dust_max) {
                    continue;
                }
                if (out_keys && count < max_out) {
                    out_keys[count] = emit_k;
                }
                count++;
            }
        }
    }
    return count;
}

size_t gl_extract_open_syncmers(
    const char *seq,
    size_t len,
    int k,
    int s,
    int skip_lower,
    uint64_t *out_keys,
    size_t max_out
) {
    return pla_extract_open_syncmers(seq, len, k, s, skip_lower, 0, -1, out_keys, max_out);
}

static void eval_one_read_pla(
    const pla_index_t *idx,
    const uint8_t nt[256],
    const char *seq,
    size_t len,
    int *hits_l,
    int *hits_b,
    int *miss
) {
    *hits_l = 0; *hits_b = 0; *miss = 0;
    int k = (int)idx->k, s = (int)idx->s;
    if ((int)len < k) return;
    const int m = k - s + 1, t = (k - s) / 2;
    const uint64_t kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const uint64_t smask = (s == 32) ? ~0ULL : ((1ULL << (2 * s)) - 1ULL);
    const int uperm = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    const int use_sm = (idx->flags & PLA_FLAG_SPLITMIX_SYNCMER) ? 1 : 0;

    uint64_t fs_ring[32], rs_ring[32], csh_ring[32];
    uint64_t fs = 0, rs = 0, fk = 0, rk = 0;
    long run = 0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = nt[(uint8_t)seq[i]];
        if (c > 3U) { run = 0; fs = rs = fk = rk = 0; continue; }
        run++;
        fs = ((fs << 2) | c) & smask;
        rs = (rs >> 2) | ((uint64_t)(3U - c) << (2 * (s - 1)));
        fk = ((fk << 2) | c) & kmask;
        rk = (rk >> 2) | ((uint64_t)(3U - c) << (2 * (k - 1)));
        if (run >= s) {
            size_t sp = i - (size_t)s + 1U;
            if (use_sm) {
                uint64_t cs = (fs < rs) ? fs : rs;
                csh_ring[sp & 31U] = splitmix64_bijective(cs);
            } else {
                fs_ring[sp & 31U] = fs;
                rs_ring[sp & 31U] = rs;
            }
        }
        if (run >= k) {
            size_t p = i - (size_t)k + 1U;
            int best = 0; uint64_t bv;
            uint64_t key = (fk < rk) ? fk : rk;
            if (use_sm) {
                if (fk < rk) {
                    bv = csh_ring[p & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = csh_ring[(p + (size_t)j) & 31U]; if (v < bv) { bv = v; best = j; } }
                } else {
                    bv = csh_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = csh_ring[(p + (size_t)(m - 1 - j)) & 31U]; if (v < bv) { bv = v; best = j; } }
                }
            } else {
                if (fk < rk) {
                    bv = fs_ring[p & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = fs_ring[(p + (size_t)j) & 31U]; if (v < bv) { bv = v; best = j; } }
                } else {
                    bv = rs_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = rs_ring[(p + (size_t)(m - 1 - j)) & 31U]; if (v < bv) { bv = v; best = j; } }
                }
            }
            if (best == t) {
                int64_t rnk = pla_lookup_inline(
                    key, idx->perm_keys, idx->segs, idx->radix16, idx->n_keys, idx->eps, uperm
                );
                if (rnk < 0) {
                    (*miss)++;
                } else {
                    uint8_t fl = idx->flags2b ? pla_get_flag2b(idx, rnk) : 1U;
                    if (fl & 1U) (*hits_l)++;
                    else (*hits_b)++;
                }
            }
        }
    }
}

static const uint8_t PLA_NT_DEFAULT[256] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
};

int pla_read_pf_hits(
    const pla_index_t *idx,
    const uint8_t *buf,
    size_t s0,
    size_t len,
    int need
) {
    if (!idx || !buf || idx->n_keys == 0 || need <= 0) return 0;
    const int k = (int)idx->k, s = (int)idx->s;
    if ((int)len < k || k > 32 || (k - s + 1) > 32) return 0;
    const int m = k - s + 1, t = (k - s) / 2;
    const uint64_t kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const uint64_t smask = (s == 32) ? ~0ULL : ((1ULL << (2 * s)) - 1ULL);
    const int uperm = (idx->flags & PLA_FLAG_PERMUTED) ? 1 : 0;
    const int use_sm = (idx->flags & PLA_FLAG_SPLITMIX_SYNCMER) ? 1 : 0;

    uint64_t fs_ring[32], rs_ring[32], csh_ring[32];
    uint64_t fs = 0, rs = 0, fk = 0, rk = 0;
    long run = 0;
    int hits = 0;
    const uint8_t *seq = buf + s0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = PLA_NT_DEFAULT[seq[i]];
        if (c > 3U) { run = 0; fs = rs = fk = rk = 0; continue; }
        run++;
        fs = ((fs << 2) | c) & smask;
        rs = (rs >> 2) | ((uint64_t)(3U - c) << (2 * (s - 1)));
        fk = ((fk << 2) | c) & kmask;
        rk = (rk >> 2) | ((uint64_t)(3U - c) << (2 * (k - 1)));
        if (run >= s) {
            size_t sp = i - (size_t)s + 1U;
            if (use_sm) {
                uint64_t cs = (fs < rs) ? fs : rs;
                csh_ring[sp & 31U] = splitmix64_bijective(cs);
            } else {
                fs_ring[sp & 31U] = fs;
                rs_ring[sp & 31U] = rs;
            }
        }
        if (run >= k) {
            size_t p = i - (size_t)k + 1U;
            int best = 0; uint64_t bv;
            uint64_t key = (fk < rk) ? fk : rk;
            if (use_sm) {
                if (fk < rk) {
                    bv = csh_ring[p & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = csh_ring[(p + (size_t)j) & 31U]; if (v < bv) { bv = v; best = j; } }
                } else {
                    bv = csh_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = csh_ring[(p + (size_t)(m - 1 - j)) & 31U]; if (v < bv) { bv = v; best = j; } }
                }
            } else {
                if (fk < rk) {
                    bv = fs_ring[p & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = fs_ring[(p + (size_t)j) & 31U]; if (v < bv) { bv = v; best = j; } }
                } else {
                    bv = rs_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) { uint64_t v = rs_ring[(p + (size_t)(m - 1 - j)) & 31U]; if (v < bv) { bv = v; best = j; } }
                }
            }
            if (best == t) {
                if (pla_lookup_inline(key, idx->perm_keys, idx->segs, idx->radix16, idx->n_keys, idx->eps, uperm) >= 0) {
                    hits++;
                    if (hits >= need) return hits;
                }
            }
        }
    }
    return hits;
}

static inline int64_t fq_line_end(const uint8_t *buf, int64_t i, int64_t n) {
    const uint8_t *p = (const uint8_t *)memchr(buf + i, '\n', (size_t)(n - i));
    return p ? (int64_t)(p - buf) : n;
}

static inline int64_t fq_trim_cr(const uint8_t *buf, int64_t s, int64_t e) {
    return (e > s && buf[e - 1] == '\r') ? (e - 1) : e;
}

static inline int64_t fq_stem_len(const uint8_t *buf, int64_t s, int64_t e) {
    int64_t j = e - 1;
    while (j > s && buf[j] != '.' && buf[j] != '/' && buf[j] != ' ') {
        j--;
    }
    if (j > s && (buf[j] == '.' || buf[j] == '/')) return j - s;
    return e - s;
}

static inline int64_t fq_adapter_pos(const uint8_t *buf, int64_t s, int64_t e, const uint8_t *ad, int64_t ad_len) {
    if (ad_len <= 0 || e - s < ad_len) return e - s;
    int64_t lim = e - ad_len;
    for (int64_t i = s; i <= lim; ++i) {
        if (buf[i] == ad[0] && memcmp(buf + i, ad, (size_t)ad_len) == 0) {
            return i - s;
        }
    }
    return e - s;
}

static inline int64_t fq_write_int(uint8_t *out, int64_t o, int64_t v) {
    if (v == 0) { out[o++] = '0'; return o; }
    uint8_t tmp[24];
    int k = 0;
    uint64_t uv = (uint64_t)v;
    while (uv > 0) {
        tmp[k++] = (uint8_t)('0' + (uv % 10ULL));
        uv /= 10ULL;
    }
    for (int j = k - 1; j >= 0; --j) {
        out[o++] = tmp[j];
    }
    return o;
}

static inline int64_t fq_emit(uint8_t *out, int64_t o, int64_t spot, const uint8_t *buf,
                              int64_t ss, int64_t se, int64_t qs, int64_t qe) {
    out[o++] = '@';
    o = fq_write_int(out, o, spot);
    out[o++] = '\n';
    int64_t sl = se - ss;
    if (sl > 0) {
        memcpy(out + o, buf + ss, (size_t)sl);
        o += sl;
    }
    out[o++] = '\n';
    out[o++] = '+';
    out[o++] = '\n';
    int64_t ql = qe - qs;
    if (ql > 0) {
        memcpy(out + o, buf + qs, (size_t)ql);
        o += ql;
    }
    out[o++] = '\n';
    return o;
}

int64_t pla_fastq_block(
    const uint8_t *buf,
    int64_t n,
    uint8_t *out,
    int64_t spot0,
    uint64_t keep_thr,
    int paired,
    const pla_index_t *idx,
    int need,
    int64_t *stats,
    size_t stats_len,
    int64_t *qhist,
    const uint8_t *ad,
    int64_t ad_len,
    int64_t min_len,
    int64_t max_spots,
    int64_t *out_o,
    int64_t *out_spot
) {
    int64_t i = 0;
    int64_t o = 0;
    int64_t spot = spot0;
    int nrec = paired ? 2 : 1;
    int64_t hs[2], he[2], ss[2], se[2], qs[2], qe[2], re[2];

    while (i < n) {
        if (max_spots > 0 && spot >= max_spots) break;
        if (buf[i] != '@') {
            int64_t e = fq_line_end(buf, i, n);
            if (e >= n) break;
            i = e + 1;
            continue;
        }
        int64_t j = i;
        int complete = 1;
        int malformed = 0;
        for (int r = 0; r < nrec; ++r) {
            if (j >= n) { complete = 0; break; }
            if (buf[j] != '@') { malformed = 1; break; }
            int64_t e1 = fq_line_end(buf, j, n);
            if (e1 >= n) { complete = 0; break; }
            int64_t e2 = fq_line_end(buf, e1 + 1, n);
            if (e2 >= n) { complete = 0; break; }
            int64_t e3 = fq_line_end(buf, e2 + 1, n);
            if (e3 >= n) { complete = 0; break; }
            int64_t e4 = fq_line_end(buf, e3 + 1, n);
            if (e4 >= n) { complete = 0; break; }
            int64_t he_r = fq_trim_cr(buf, j + 1, e1);
            int64_t se_r = fq_trim_cr(buf, e1 + 1, e2);
            int64_t qe_r = fq_trim_cr(buf, e3 + 1, e4);
            if (he_r <= j + 1 || e2 + 1 >= e3 || buf[e2 + 1] != '+' ||
                (se_r - (e1 + 1)) != (qe_r - (e3 + 1))) {
                malformed = 1;
                break;
            }
            hs[r] = j + 1; he[r] = he_r;
            ss[r] = e1 + 1; se[r] = se_r;
            qs[r] = e3 + 1; qe[r] = qe_r;
            re[r] = e4;
            j = e4 + 1;
        }
        if (!complete) break;
        if (malformed) {
            stats[5]++;
            int64_t e = fq_line_end(buf, i, n);
            if (e >= n) break;
            i = e + 1;
            continue;
        }
        if (nrec == 2) {
            int64_t l0 = fq_stem_len(buf, hs[0], he[0]);
            int same = (l0 == fq_stem_len(buf, hs[1], he[1]));
            if (same && memcmp(buf + hs[0], buf + hs[1], (size_t)l0) != 0) same = 0;
            if (!same) {
                stats[5]++;
                i = re[0] + 1;
                continue;
            }
            if ((se[0] - ss[0]) >= min_len && (se[1] - ss[1]) < 20 && j < n) {
                if (buf[j] == '@') {
                    int64_t te1 = fq_line_end(buf, j, n);
                    int64_t te2 = (te1 < n) ? fq_line_end(buf, te1 + 1, n) : n;
                    int64_t te3 = (te2 < n) ? fq_line_end(buf, te2 + 1, n) : n;
                    int64_t te4 = (te3 < n) ? fq_line_end(buf, te3 + 1, n) : n;
                    if (te4 >= n) break;
                    int64_t the = fq_trim_cr(buf, j + 1, te1);
                    int64_t tse = fq_trim_cr(buf, te1 + 1, te2);
                    int64_t tqe = fq_trim_cr(buf, te3 + 1, te4);
                    if (the > j + 1 && te2 + 1 < te3 && buf[te2 + 1] == '+' &&
                        (tse - (te1 + 1)) == (tqe - (te3 + 1)) &&
                        fq_stem_len(buf, j + 1, the) == l0 &&
                        memcmp(buf + hs[0], buf + j + 1, (size_t)l0) == 0) {
                        hs[1] = j + 1; he[1] = the;
                        ss[1] = te1 + 1; se[1] = tse;
                        qs[1] = te3 + 1; qe[1] = tqe;
                        re[1] = te4;
                        j = te4 + 1;
                        if (stats_len > 9) stats[9]++;
                    }
                }
            }
        }
        i = j;
        stats[0]++;
        int64_t blen = 0;
        for (int r = 0; r < nrec; ++r) {
            int64_t ln = se[r] - ss[r];
            blen += ln;
            if (ln > stats[6]) stats[6] = ln;
            if (stats[0] <= 2000 && qhist) {
                for (int64_t qk = qs[r]; qk < qe[r]; ++qk) {
                    qhist[buf[qk] & 127]++;
                }
            }
        }
        stats[3] += blen;
        if ((splitmix64_bijective((uint64_t)spot) >> 32) >= keep_thr) {
            spot++;
            continue;
        }
        stats[1]++;
        if (ad && ad_len > 0) {
            int is_short = 0;
            for (int r = 0; r < nrec; ++r) {
                int64_t t = fq_adapter_pos(buf, ss[r], se[r], ad, ad_len);
                if (t < se[r] - ss[r]) {
                    stats[7]++;
                    se[r] = ss[r] + t;
                    qe[r] = qs[r] + t;
                }
                if (t < min_len) is_short = 1;
            }
            if (is_short) {
                spot++;
                continue;
            }
        }
        int hits = 0;
        if (need > 0) {
            for (int r = 0; r < nrec; ++r) {
                hits += pla_read_pf_hits(idx, buf, (size_t)ss[r], (size_t)(se[r] - ss[r]), need - hits);
                if (hits >= need) break;
            }
        }
        if (need == 0 || hits >= need) {
            stats[2]++;
            stats[4] += blen;
            if (stats_len > 8) stats[8] = spot;
            for (int r = 0; r < nrec; ++r) {
                o = fq_emit(out, o, spot, buf, ss[r], se[r], qs[r], qe[r]);
            }
        }
        spot++;
    }
    if (out_o) *out_o = o;
    if (out_spot) *out_spot = spot;
    return i;
}

void gl_sieve_pairs_pla(
    const pla_index_t *idx,
    const char * const *r1_seqs,
    const size_t *r1_lens,
    const char * const *r2_seqs,
    const size_t *r2_lens,
    size_t n_pairs,
    int min_hits,
    int max_miss,
    uint8_t *out_decisions,
    gl_sieve_stats_t *out_stats,
    int num_threads
) {
    uint8_t nt[256];
    build_nt_table(nt, 0);
    int T = effective_threads(num_threads);
    uint64_t dl = 0, db = 0, pout = 0;

    #pragma omp parallel for reduction(+:dl,db,pout) num_threads(T) schedule(static)
    for (size_t p = 0; p < n_pairs; ++p) {
        uint8_t d = 0;
        if (idx && idx->n_keys > 0) {
            int hl0, hb0, ms0, hl1, hb1, ms1;
            eval_one_read_pla(idx, nt, r1_seqs[p], r1_lens[p], &hl0, &hb0, &ms0);
            eval_one_read_pla(idx, nt, r2_seqs[p], r2_lens[p], &hl1, &hb1, &ms1);
            int dep0 = ((hl0 + hb0) >= min_hits) && (ms0 <= max_miss);
            int dep1 = ((hl1 + hb1) >= min_hits) && (ms1 <= max_miss);
            int bait0 = (hb0 > hl0), bait1 = (hb1 > hl1);
            d = (dep0 && dep1) ? (uint8_t)((bait0 || bait1) ? 2U : 1U) : 0U;
        }
        if (out_decisions) out_decisions[p] = d;
        if (d == 1U) dl++;
        else if (d == 2U) db++;
        else pout++;
    }
    if (out_stats) {
        out_stats->pairs_in   = (uint64_t)n_pairs;
        out_stats->pairs_out  = pout;
        out_stats->drop_locus = dl;
        out_stats->drop_bait  = db;
    }
}

static void eval_one_read_htab(
    const gl_htab_t *ht,
    int k,
    int s,
    const uint8_t nt[256],
    const char *seq,
    size_t len,
    int *hits_l,
    int *hits_b,
    int *miss
) {
    *hits_l = 0; *hits_b = 0; *miss = 0;
    if ((int)len < k) return;
    const int m = k - s + 1, t = (k - s) / 2;
    const uint64_t kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const uint64_t smask = (s == 32) ? ~0ULL : ((1ULL << (2 * s)) - 1ULL);

    uint64_t fs_ring[32], rs_ring[32];
    uint64_t fs = 0, rs = 0, fk = 0, rk = 0;
    long run = 0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = nt[(uint8_t)seq[i]];
        if (c > 3U) { run = 0; fs = rs = fk = rk = 0; continue; }
        run++;
        fs = ((fs << 2) | c) & smask;
        rs = (rs >> 2) | ((uint64_t)(3U - c) << (2 * (s - 1)));
        fk = ((fk << 2) | c) & kmask;
        rk = (rk >> 2) | ((uint64_t)(3U - c) << (2 * (k - 1)));
        if (run >= s) {
            size_t sp = i - (size_t)s + 1U;
            fs_ring[sp & 31U] = fs;
            rs_ring[sp & 31U] = rs;
        }
        if (run >= k) {
            size_t p = i - (size_t)k + 1U;
            int best = 0; uint64_t bv, key;
            if (fk < rk) {
                bv = fs_ring[p & 31U];
                for (int j = 1; j < m; ++j) { uint64_t v = fs_ring[(p + (size_t)j) & 31U]; if (v < bv) { bv = v; best = j; } }
                key = fk;
            } else {
                bv = rs_ring[(p + (size_t)(m - 1)) & 31U];
                for (int j = 1; j < m; ++j) { uint64_t v = rs_ring[(p + (size_t)(m - 1 - j)) & 31U]; if (v < bv) { bv = v; best = j; } }
                key = rk;
            }
            if (best == t) {
                uint64_t slot = gl_ht_find_inline(ht, key);
                if (slot == ht->cap) {
                    (*miss)++;
                } else {
                    if (ht->vals[slot] & 1U) (*hits_l)++;
                    else (*hits_b)++;
                }
            }
        }
    }
}

void gl_sieve_pairs_htab(
    const gl_htab_t *ht,
    int k,
    int s,
    const char * const *r1_seqs,
    const size_t *r1_lens,
    const char * const *r2_seqs,
    const size_t *r2_lens,
    size_t n_pairs,
    int min_hits,
    int max_miss,
    uint8_t *out_decisions,
    gl_sieve_stats_t *out_stats,
    int num_threads
) {
    uint8_t nt[256];
    build_nt_table(nt, 0);
    int T = effective_threads(num_threads);
    uint64_t dl = 0, db = 0, pout = 0;

    #pragma omp parallel for reduction(+:dl,db,pout) num_threads(T) schedule(static)
    for (size_t p = 0; p < n_pairs; ++p) {
        uint8_t d = 0;
        if (ht && ht->n > 0) {
            int hl0, hb0, ms0, hl1, hb1, ms1;
            eval_one_read_htab(ht, k, s, nt, r1_seqs[p], r1_lens[p], &hl0, &hb0, &ms0);
            eval_one_read_htab(ht, k, s, nt, r2_seqs[p], r2_lens[p], &hl1, &hb1, &ms1);
            int dep0 = ((hl0 + hb0) >= min_hits) && (ms0 <= max_miss);
            int dep1 = ((hl1 + hb1) >= min_hits) && (ms1 <= max_miss);
            int bait0 = (hb0 > hl0), bait1 = (hb1 > hl1);
            d = (dep0 && dep1) ? (uint8_t)((bait0 || bait1) ? 2U : 1U) : 0U;
        }
        if (out_decisions) out_decisions[p] = d;
        if (d == 1U) dl++;
        else if (d == 2U) db++;
        else pout++;
    }
    if (out_stats) {
        out_stats->pairs_in   = (uint64_t)n_pairs;
        out_stats->pairs_out  = pout;
        out_stats->drop_locus = dl;
        out_stats->drop_bait  = db;
    }
}

// -----------------------------------------------------------------------------------------------
// 7. Fast Parallel Synthetic Genomic Open-Syncmer Generator (50%-GC vs 81%-AT Plasmodium-like)
// -----------------------------------------------------------------------------------------------

typedef struct { uint64_t s[4]; } xoshiro_t;

static inline uint64_t rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro_next(xoshiro_t *rng) {
    const uint64_t res = rotl64(rng->s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t   = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl64(rng->s[3], 45);
    return res;
}

static void xoshiro_seed(xoshiro_t *rng, uint64_t seed) {
    for (int i = 0; i < 4; ++i) {
        seed += 0x9e3779b97f4a7c15ULL;
        rng->s[i] = splitmix64_bijective(seed);
    }
}

static void init_pf_at81_markov(uint8_t tbl[4][256]) {
    // Stationary distribution: AT = 174 / (174 + 41) = 80.93% AT (A=40.47%, T=40.47%, C=9.53%, G=9.53%),
    // with Plasmodium-characteristic poly-dA / poly-dT homopolymer persistence (P(A|A) = P(T|T) = 52.0%).
    const int cnt[4][4] = {
        { 133,  20,  21,  82 }, // from A: A=133, C=20, G=21, T=82 (sum 256)
        {  87,  41,  41,  87 }, // from C: A=87,  C=41, G=41, T=87 (sum 256)
        {  87,  41,  41,  87 }, // from G: A=87,  C=41, G=41, T=87 (sum 256)
        {  82,  21,  20, 133 }  // from T: A=82,  C=21, G=20, T=133 (sum 256)
    };
    for (int prev = 0; prev < 4; ++prev) {
        int idx = 0;
        for (int nxt = 0; nxt < 4; ++nxt) {
            for (int c = 0; c < cnt[prev][nxt]; ++c) {
                tbl[prev][idx++] = (uint8_t)nxt;
            }
        }
    }
}

static void gen_syncmer_chunk(
    uint64_t *dst,
    size_t want,
    int mode_at81,
    int k,
    int s,
    const uint8_t markov[4][256],
    uint64_t stream_seed
) {
    xoshiro_t rng;
    xoshiro_seed(&rng, stream_seed);

    const int m = k - s + 1;
    const int t = (k - s) / 2;
    const uint64_t kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const uint64_t smask = (s == 32) ? ~0ULL : ((1ULL << (2 * s)) - 1ULL);

    uint64_t fs_ring[32], rs_ring[32];
    uint64_t fs = 0, rs = 0, fk = 0, rk = 0;
    size_t pos = 0, produced = 0;
    uint8_t cur_base = (uint8_t)(xoshiro_next(&rng) & 3U);

    while (produced < want) {
        uint64_t rword = xoshiro_next(&rng);
        int steps = mode_at81 ? 8 : 32;
        for (int st = 0; st < steps && produced < want; ++st) {
            uint8_t c;
            if (mode_at81) {
                c = markov[cur_base][rword & 0xFFU];
                rword >>= 8;
                cur_base = c;
            } else {
                c = (uint8_t)(rword & 3U);
                rword >>= 2;
            }
            fs = ((fs << 2) | c) & smask;
            rs = (rs >> 2) | ((uint64_t)(3U - c) << (2 * (s - 1)));
            fk = ((fk << 2) | c) & kmask;
            rk = (rk >> 2) | ((uint64_t)(3U - c) << (2 * (k - 1)));

            if (pos + 1U >= (size_t)s) {
                size_t sp = pos + 1U - (size_t)s;
                fs_ring[sp & 31U] = fs;
                rs_ring[sp & 31U] = rs;
            }
            if (pos + 1U >= (size_t)k) {
                size_t p = pos + 1U - (size_t)k;
                int best = 0;
                uint64_t bv, emit_k;
                if (fk < rk) {
                    bv = fs_ring[p & 31U];
                    for (int j = 1; j < m; ++j) {
                        uint64_t v = fs_ring[(p + (size_t)j) & 31U];
                        if (v < bv) { bv = v; best = j; }
                    }
                    emit_k = fk;
                } else {
                    bv = rs_ring[(p + (size_t)(m - 1)) & 31U];
                    for (int j = 1; j < m; ++j) {
                        uint64_t v = rs_ring[(p + (size_t)(m - 1 - j)) & 31U];
                        if (v < bv) { bv = v; best = j; }
                    }
                    emit_k = rk;
                }
                if (best == t) {
                    dst[produced++] = emit_k;
                }
            }
            pos++;
        }
    }
}

int pla_generate_distinct_syncmers(
    uint64_t *out_keys,
    size_t n_target,
    int mode_at81,
    int k,
    int s,
    uint64_t seed,
    int num_threads
) {
    if (!out_keys && n_target > 0) return -1;
    if (n_target == 0) return 0;

    uint8_t markov[4][256];
    init_pf_at81_markov(markov);
    int T = effective_threads(num_threads);

    // Phase 1: Parallel generation of n_target raw open syncmers across T independent contigs.
    #pragma omp parallel num_threads(T)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        size_t s0 = (n_target * (size_t)tid) / (size_t)T;
        size_t s1 = (n_target * (size_t)(tid + 1)) / (size_t)T;
        if (s1 > s0) {
            gen_syncmer_chunk(
                out_keys + s0, s1 - s0, mode_at81, k, s, markov,
                seed ^ ((uint64_t)(tid + 1) * 0x9e3779b97f4a7c15ULL)
            );
        }
    }

    // Phase 2: Sort + deduplicate; if collisions reduced count below n_target, top up exact stream delta!
    size_t unique = pla_sort_dedup_u64(out_keys, n_target, T);
    uint64_t round = 1ULL;
    while (unique < n_target) {
        size_t missing = n_target - unique;
        size_t batch = missing + (missing >> 2) + 256U;
        uint64_t *extra = (uint64_t *)malloc(batch * sizeof(uint64_t));
        if (!extra) return -1;

        int Tb = (batch < 32768U) ? 1 : T;
        #pragma omp parallel num_threads(Tb)
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            size_t b0 = (batch * (size_t)tid) / (size_t)Tb;
            size_t b1 = (batch * (size_t)(tid + 1)) / (size_t)Tb;
            if (b1 > b0) {
                gen_syncmer_chunk(
                    extra + b0, b1 - b0, mode_at81, k, s, markov,
                    seed ^ (round * 0xd1b54a32d192ed03ULL) ^ ((uint64_t)(tid + 101) * 0x94d049bb133111ebULL)
                );
            }
        }

        // Filter against existing sorted unique keys[0..unique-1] in arrival order so there is zero domain truncation bias!
        size_t base_u = unique;
        for (size_t i = 0; i < batch && unique < n_target; ++i) {
            uint64_t cand = extra[i];
            if (bsearch_u64_inline(out_keys, base_u, cand) < 0) {
                // Also ensure not duplicated within newly appended tail
                int dup_tail = 0;
                for (size_t t = base_u; t < unique; ++t) {
                    if (out_keys[t] == cand) { dup_tail = 1; break; }
                }
                if (!dup_tail) {
                    out_keys[unique++] = cand;
                    // If tail grows > 512, fold into sorted array so tail scan stays O(1)
                    if (unique - base_u >= 512U) {
                        unique = pla_sort_dedup_u64(out_keys, unique, T);
                        base_u = unique;
                    }
                }
            }
        }
        free(extra);
        unique = pla_sort_dedup_u64(out_keys, unique, T);
        round++;
    }

    return 0;
}
