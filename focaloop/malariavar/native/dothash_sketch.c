/*
 * dothash_sketch.c — Unsettled-Restricted d = 2048 int16 Rademacher Hypervectors
 * (DotHash / Hypergen, Faure, Abrar, Chikhi, Wu, Koslicki & Medvedev 2026).
 *
 * Apple Silicon ARM NEON + OpenMP implementation:
 *   - Deterministic 64-bit syncmer key -> 2048 {+1, -1} Rademacher signs
 *     (32 splitmix64 64-bit blocks per distinct key)
 *   - Hardware saturating int16 accumulation clamped symmetrically to [-32767, +32767]
 *   - Overflow-proof int64 NEON dot products (<v(A), v(B)>)
 *   - L1-cache-tiled 1-vs-N and Q-vs-N cohort nearest-neighbor search (24,409 x 2048)
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define DOTHASH_DIM 2048
#define DOTHASH_WORDS 32

/* 256 x 8 int16 lookup table (4 KB, stays hot in L1D):
 * byte b -> 8 Rademacher signs {+1, -1} for bits 0..7.
 */
static int16_t SIGN_LUT[256][8] __attribute__((aligned(64)));
static int SIGN_LUT_READY = 0;

__attribute__((constructor))
void dothash_init(void) {
    if (SIGN_LUT_READY) return;
    for (int b = 0; b < 256; ++b) {
        for (int bit = 0; bit < 8; ++bit) {
            SIGN_LUT[b][bit] = ((b >> bit) & 1) ? (int16_t)1 : (int16_t)-1;
        }
    }
    SIGN_LUT_READY = 1;
}

static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Incrementally accumulate n_keys distinct 64-bit keys into an existing int16_t hv[2048],
 * with symmetric saturating clamping to [-32767, +32767] on every update.
 */
void dothash_update_u64(int16_t *hv, const uint64_t *keys, size_t n_keys) {
    if (!SIGN_LUT_READY) dothash_init();
#if HAVE_NEON
    const int16x8_t vmin = vdupq_n_s16(-32767);
    for (size_t k = 0; k < n_keys; ++k) {
        uint64_t state = keys[k];
        int16_t *dst = hv;
        for (int w = 0; w < DOTHASH_WORDS; ++w) {
            uint64_t bits = splitmix64_next(&state);
            int16x8_t v0 = vld1q_s16(dst + 0);
            int16x8_t v1 = vld1q_s16(dst + 8);
            int16x8_t v2 = vld1q_s16(dst + 16);
            int16x8_t v3 = vld1q_s16(dst + 24);
            int16x8_t v4 = vld1q_s16(dst + 32);
            int16x8_t v5 = vld1q_s16(dst + 40);
            int16x8_t v6 = vld1q_s16(dst + 48);
            int16x8_t v7 = vld1q_s16(dst + 56);

            v0 = vmaxq_s16(vqaddq_s16(v0, vld1q_s16(SIGN_LUT[(bits >>  0) & 0xFF])), vmin);
            v1 = vmaxq_s16(vqaddq_s16(v1, vld1q_s16(SIGN_LUT[(bits >>  8) & 0xFF])), vmin);
            v2 = vmaxq_s16(vqaddq_s16(v2, vld1q_s16(SIGN_LUT[(bits >> 16) & 0xFF])), vmin);
            v3 = vmaxq_s16(vqaddq_s16(v3, vld1q_s16(SIGN_LUT[(bits >> 24) & 0xFF])), vmin);
            v4 = vmaxq_s16(vqaddq_s16(v4, vld1q_s16(SIGN_LUT[(bits >> 32) & 0xFF])), vmin);
            v5 = vmaxq_s16(vqaddq_s16(v5, vld1q_s16(SIGN_LUT[(bits >> 40) & 0xFF])), vmin);
            v6 = vmaxq_s16(vqaddq_s16(v6, vld1q_s16(SIGN_LUT[(bits >> 48) & 0xFF])), vmin);
            v7 = vmaxq_s16(vqaddq_s16(v7, vld1q_s16(SIGN_LUT[(bits >> 56) & 0xFF])), vmin);

            vst1q_s16(dst + 0,  v0);
            vst1q_s16(dst + 8,  v1);
            vst1q_s16(dst + 16, v2);
            vst1q_s16(dst + 24, v3);
            vst1q_s16(dst + 32, v4);
            vst1q_s16(dst + 40, v5);
            vst1q_s16(dst + 48, v6);
            vst1q_s16(dst + 56, v7);
            dst += 64;
        }
    }
#else
    for (size_t k = 0; k < n_keys; ++k) {
        uint64_t state = keys[k];
        for (int w = 0; w < DOTHASH_WORDS; ++w) {
            uint64_t bits = splitmix64_next(&state);
            for (int b = 0; b < 64; ++b) {
                int32_t s = ((bits >> b) & 1ULL) ? 1 : -1;
                int32_t v = (int32_t)hv[64 * w + b] + s;
                if (v > 32767) v = 32767;
                if (v < -32767) v = -32767;
                hv[64 * w + b] = (int16_t)v;
            }
        }
    }
#endif
}

/* Zero-initialize hv_out[2048] and sketch n_keys distinct keys. */
void dothash_sketch_u64(const uint64_t *keys, size_t n_keys, int16_t *hv_out) {
    memset(hv_out, 0, DOTHASH_DIM * sizeof(int16_t));
    dothash_update_u64(hv_out, keys, n_keys);
}

/* Saturating addition of two sketches: hv_dst = clamp(hv_dst + hv_src, -32767, +32767). */
void dothash_merge_i16(int16_t *hv_dst, const int16_t *hv_src) {
#if HAVE_NEON
    const int16x8_t vmin = vdupq_n_s16(-32767);
    for (int j = 0; j < DOTHASH_DIM; j += 8) {
        int16x8_t a = vld1q_s16(hv_dst + j);
        int16x8_t b = vld1q_s16(hv_src + j);
        vst1q_s16(hv_dst + j, vmaxq_s16(vqaddq_s16(a, b), vmin));
    }
#else
    for (int j = 0; j < DOTHASH_DIM; ++j) {
        int32_t v = (int32_t)hv_dst[j] + (int32_t)hv_src[j];
        if (v > 32767) v = 32767;
        if (v < -32767) v = -32767;
        hv_dst[j] = (int16_t)v;
    }
#endif
}

/* OpenMP batch sketching from CSR key array: sample s uses keys[offsets[s] .. offsets[s+1]-1]. */
void dothash_sketch_batch_csr(
    const uint64_t *all_keys,
    const int64_t  *offsets,
    size_t          n_samples,
    int16_t        *matrix_out,
    int             n_threads
) {
    if (!SIGN_LUT_READY) dothash_init();
#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 32) num_threads(n_threads)
#else
    (void)n_threads;
#endif
    for (size_t s = 0; s < n_samples; ++s) {
        int64_t beg = offsets[s];
        int64_t end = offsets[s + 1];
        int16_t *row = matrix_out + s * DOTHASH_DIM;
        memset(row, 0, DOTHASH_DIM * sizeof(int16_t));
        if (end > beg) {
            dothash_update_u64(row, all_keys + beg, (size_t)(end - beg));
        }
    }
}

/* Overflow-proof exact 2048-dim int16 dot product -> int64_t.
 * Widens every 4-lane vmull_s16 / vmull_high_s16 product (max (-32768)^2 = +1,073,741,824 <= INT32_MAX)
 * directly into int64x2_t via vpadalq_s32, preventing signed int32 overflow even on INT16_MIN (-32768).
 */
static inline int64_t dot_i16_2048_fast(const int16_t * restrict a, const int16_t * restrict b) {
#if HAVE_NEON
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);
    int64x2_t sum2 = vdupq_n_s64(0);
    int64x2_t sum3 = vdupq_n_s64(0);

    for (int j = 0; j < DOTHASH_DIM; j += 32) {
        int16x8_t a0 = vld1q_s16(a + j);
        int16x8_t b0 = vld1q_s16(b + j);
        sum0 = vpadalq_s32(sum0, vmull_s16(vget_low_s16(a0), vget_low_s16(b0)));
        sum0 = vpadalq_s32(sum0, vmull_high_s16(a0, b0));

        int16x8_t a1 = vld1q_s16(a + j + 8);
        int16x8_t b1 = vld1q_s16(b + j + 8);
        sum1 = vpadalq_s32(sum1, vmull_s16(vget_low_s16(a1), vget_low_s16(b1)));
        sum1 = vpadalq_s32(sum1, vmull_high_s16(a1, b1));

        int16x8_t a2 = vld1q_s16(a + j + 16);
        int16x8_t b2 = vld1q_s16(b + j + 16);
        sum2 = vpadalq_s32(sum2, vmull_s16(vget_low_s16(a2), vget_low_s16(b2)));
        sum2 = vpadalq_s32(sum2, vmull_high_s16(a2, b2));

        int16x8_t a3 = vld1q_s16(a + j + 24);
        int16x8_t b3 = vld1q_s16(b + j + 24);
        sum3 = vpadalq_s32(sum3, vmull_s16(vget_low_s16(a3), vget_low_s16(b3)));
        sum3 = vpadalq_s32(sum3, vmull_high_s16(a3, b3));
    }
    int64x2_t s01 = vaddq_s64(sum0, sum1);
    int64x2_t s23 = vaddq_s64(sum2, sum3);
    int64x2_t tot = vaddq_s64(s01, s23);
    return vgetq_lane_s64(tot, 0) + vgetq_lane_s64(tot, 1);
#else
    int64_t acc = 0;
    for (int j = 0; j < DOTHASH_DIM; ++j) {
        acc += (int64_t)a[j] * (int64_t)b[j];
    }
    return acc;
#endif
}

int64_t dothash_dot_i16(const int16_t *a, const int16_t *b) {
    return dot_i16_2048_fast(a, b);
}

#if HAVE_NEON
static inline void dot4_i16_2048_fast(
    const int16_t * restrict q0,
    const int16_t * restrict q1,
    const int16_t * restrict q2,
    const int16_t * restrict q3,
    const int16_t * restrict row,
    int64_t *out4
) {
    int64x2_t s0 = vdupq_n_s64(0);
    int64x2_t s1 = vdupq_n_s64(0);
    int64x2_t s2 = vdupq_n_s64(0);
    int64x2_t s3 = vdupq_n_s64(0);

    for (int j = 0; j < DOTHASH_DIM; j += 16) {
        int16x8_t r_lo = vld1q_s16(row + j);
        int16x8_t r_hi = vld1q_s16(row + j + 8);

        int16x8_t a0 = vld1q_s16(q0 + j);
        int16x8_t b0 = vld1q_s16(q0 + j + 8);
        s0 = vpadalq_s32(s0, vmull_s16(vget_low_s16(a0), vget_low_s16(r_lo)));
        s0 = vpadalq_s32(s0, vmull_high_s16(a0, r_lo));
        s0 = vpadalq_s32(s0, vmull_s16(vget_low_s16(b0), vget_low_s16(r_hi)));
        s0 = vpadalq_s32(s0, vmull_high_s16(b0, r_hi));

        int16x8_t a1 = vld1q_s16(q1 + j);
        int16x8_t b1 = vld1q_s16(q1 + j + 8);
        s1 = vpadalq_s32(s1, vmull_s16(vget_low_s16(a1), vget_low_s16(r_lo)));
        s1 = vpadalq_s32(s1, vmull_high_s16(a1, r_lo));
        s1 = vpadalq_s32(s1, vmull_s16(vget_low_s16(b1), vget_low_s16(r_hi)));
        s1 = vpadalq_s32(s1, vmull_high_s16(b1, r_hi));

        int16x8_t a2 = vld1q_s16(q2 + j);
        int16x8_t b2 = vld1q_s16(q2 + j + 8);
        s2 = vpadalq_s32(s2, vmull_s16(vget_low_s16(a2), vget_low_s16(r_lo)));
        s2 = vpadalq_s32(s2, vmull_high_s16(a2, r_lo));
        s2 = vpadalq_s32(s2, vmull_s16(vget_low_s16(b2), vget_low_s16(r_hi)));
        s2 = vpadalq_s32(s2, vmull_high_s16(b2, r_hi));

        int16x8_t a3 = vld1q_s16(q3 + j);
        int16x8_t b3 = vld1q_s16(q3 + j + 8);
        s3 = vpadalq_s32(s3, vmull_s16(vget_low_s16(a3), vget_low_s16(r_lo)));
        s3 = vpadalq_s32(s3, vmull_high_s16(a3, r_lo));
        s3 = vpadalq_s32(s3, vmull_s16(vget_low_s16(b3), vget_low_s16(r_hi)));
        s3 = vpadalq_s32(s3, vmull_high_s16(b3, r_hi));
    }
    out4[0] = vgetq_lane_s64(s0, 0) + vgetq_lane_s64(s0, 1);
    out4[1] = vgetq_lane_s64(s1, 0) + vgetq_lane_s64(s1, 1);
    out4[2] = vgetq_lane_s64(s2, 0) + vgetq_lane_s64(s2, 1);
    out4[3] = vgetq_lane_s64(s3, 0) + vgetq_lane_s64(s3, 1);
}
#endif

/* Compute self squared norms ||mat[i]||_2^2 for i = 0 .. n_rows-1. */
void dothash_norms_sq_batch(
    const int16_t *mat,
    size_t         n_rows,
    int64_t       *norms_sq_out,
    int            n_threads
) {
#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(n_threads) if(n_rows >= 256 && n_threads > 1)
#else
    (void)n_threads;
#endif
    for (size_t i = 0; i < n_rows; ++i) {
        const int16_t *row = mat + i * DOTHASH_DIM;
        norms_sq_out[i] = dot_i16_2048_fast(row, row);
    }
}

/* Compute 1-vs-N exact inner products <query, mat[i]> for i = 0 .. n_rows-1. */
void dothash_dots_1_vs_n(
    const int16_t *query,
    const int16_t *mat,
    size_t         n_rows,
    int64_t       *dots_out,
    int            n_threads
) {
#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(n_threads) if(n_rows >= 256 && n_threads > 1)
#else
    (void)n_threads;
#endif
    for (size_t i = 0; i < n_rows; ++i) {
        dots_out[i] = dot_i16_2048_fast(query, mat + i * DOTHASH_DIM);
    }
}

static inline void topk_insert(
    int k_top,
    float score,
    int32_t idx,
    float *best_scores,
    int32_t *best_idx
) {
    if (score <= best_scores[k_top - 1]) return;
    int pos = k_top - 1;
    while (pos > 0 && score > best_scores[pos - 1]) {
        best_scores[pos] = best_scores[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        pos--;
    }
    best_scores[pos] = score;
    best_idx[pos] = idx;
}

/* Fast 1-vs-N Jaccard top-k search in RAM (e.g. 24,409 x 2048).
 * row_cards[i] = |B_i| (or ||v(B_i)||_2^2 / 2048.0f).
 * query_card   = |A|   (or ||v(A)||_2^2 / 2048.0f).
 */
void dothash_knn_1_vs_n_jaccard(
    const int16_t *query,
    float          query_card,
    const int16_t *mat,
    const float   *row_cards,
    size_t         n_rows,
    int            k_top,
    int32_t       *top_idx_out,
    float         *top_jaccard_out,
    int            n_threads
) {
    if (k_top <= 0) return;
    for (int c = 0; c < k_top; ++c) {
        top_jaccard_out[c] = -1e30f;
        top_idx_out[c] = -1;
    }
    if (n_rows == 0) return;
    const float inv_d = 1.0f / (float)DOTHASH_DIM;

    if (n_threads == 1 || n_rows < 1024) {
        for (size_t i = 0; i < n_rows; ++i) {
            int64_t dot = dot_i16_2048_fast(query, mat + i * DOTHASH_DIM);
            float inter = (float)dot * inv_d;
            if (inter < 0.0f) inter = 0.0f;
            float max_inter = (query_card < row_cards[i]) ? query_card : row_cards[i];
            if (inter > max_inter) inter = max_inter;
            float denom = query_card + row_cards[i] - inter;
            float jacc = (denom > 1e-6f) ? (inter / denom) : 0.0f;
            topk_insert(k_top, jacc, (int32_t)i, top_jaccard_out, top_idx_out);
        }
        return;
    }

#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    float   *local_scores = (float *)malloc((size_t)n_threads * (size_t)k_top * sizeof(float));
    int32_t *local_idx    = (int32_t *)malloc((size_t)n_threads * (size_t)k_top * sizeof(int32_t));
    for (int t = 0; t < n_threads * k_top; ++t) {
        local_scores[t] = -1e30f;
        local_idx[t] = -1;
    }
    int team_T = 1;

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int actual_T = omp_get_num_threads();
        if (tid == 0) team_T = actual_T;
        if (tid < n_threads) {
            float   *my_s = local_scores + tid * k_top;
            int32_t *my_i = local_idx + tid * k_top;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n_rows; ++i) {
                int64_t dot = dot_i16_2048_fast(query, mat + i * DOTHASH_DIM);
                float inter = (float)dot * inv_d;
                if (inter < 0.0f) inter = 0.0f;
                float max_inter = (query_card < row_cards[i]) ? query_card : row_cards[i];
                if (inter > max_inter) inter = max_inter;
                float denom = query_card + row_cards[i] - inter;
                float jacc = (denom > 1e-6f) ? (inter / denom) : 0.0f;
                topk_insert(k_top, jacc, (int32_t)i, my_s, my_i);
            }
        }
    }
    if (team_T > n_threads) team_T = n_threads;

    for (int t = 0; t < team_T; ++t) {
        for (int c = 0; c < k_top; ++c) {
            if (local_idx[t * k_top + c] >= 0) {
                topk_insert(k_top, local_scores[t * k_top + c], local_idx[t * k_top + c],
                            top_jaccard_out, top_idx_out);
            }
        }
    }
    free(local_scores);
    free(local_idx);
#else
    dothash_knn_1_vs_n_jaccard(query, query_card, mat, row_cards, n_rows, k_top,
                               top_idx_out, top_jaccard_out, 1);
#endif
}

/* L1-cache-tiled Q-vs-N Jaccard top-k search (e.g. 100 queries vs 24,409 Pf8 isolates).
 * Streams each cohort row once per 8-query tile, reusing mat[i] in L1/NEON registers!
 */
void dothash_knn_q_vs_n_jaccard(
    const int16_t *queries,
    const float   *query_cards,
    size_t         n_queries,
    const int16_t *mat,
    const float   *row_cards,
    size_t         n_rows,
    int            k_top,
    int32_t       *top_idx_out,
    float         *top_jaccard_out,
    int            n_threads
) {
    if (k_top <= 0 || n_queries == 0) return;
    for (size_t q = 0; q < n_queries * (size_t)k_top; ++q) {
        top_jaccard_out[q] = -1e30f;
        top_idx_out[q] = -1;
    }
    if (n_rows == 0) return;
    const float inv_d = 1.0f / (float)DOTHASH_DIM;

#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
#else
    n_threads = 1;
#endif

    const size_t TILE_Q = 8;
    for (size_t q0 = 0; q0 < n_queries; q0 += TILE_Q) {
        size_t q_end = q0 + TILE_Q;
        if (q_end > n_queries) q_end = n_queries;
        size_t qt = q_end - q0;

        if (n_threads == 1 || n_rows < 1024) {
            for (size_t i = 0; i < n_rows; ++i) {
                const int16_t *row = mat + i * DOTHASH_DIM;
                float rc = row_cards[i];
                size_t dq = 0;
#if HAVE_NEON
                for (; dq + 4 <= qt; dq += 4) {
                    int64_t dots[4];
                    dot4_i16_2048_fast(
                        queries + (q0 + dq + 0) * DOTHASH_DIM,
                        queries + (q0 + dq + 1) * DOTHASH_DIM,
                        queries + (q0 + dq + 2) * DOTHASH_DIM,
                        queries + (q0 + dq + 3) * DOTHASH_DIM,
                        row, dots
                    );
                    for (size_t u = 0; u < 4; ++u) {
                        size_t q = q0 + dq + u;
                        float inter = (float)dots[u] * inv_d;
                        if (inter < 0.0f) inter = 0.0f;
                        float qc = query_cards[q];
                        float max_inter = (qc < rc) ? qc : rc;
                        if (inter > max_inter) inter = max_inter;
                        float denom = qc + rc - inter;
                        float jacc = (denom > 1e-6f) ? (inter / denom) : 0.0f;
                        topk_insert(k_top, jacc, (int32_t)i,
                                    top_jaccard_out + q * (size_t)k_top,
                                    top_idx_out + q * (size_t)k_top);
                    }
                }
#endif
                for (; dq < qt; ++dq) {
                    size_t q = q0 + dq;
                    int64_t dot = dot_i16_2048_fast(queries + q * DOTHASH_DIM, row);
                    float inter = (float)dot * inv_d;
                    if (inter < 0.0f) inter = 0.0f;
                    float qc = query_cards[q];
                    float max_inter = (qc < rc) ? qc : rc;
                    if (inter > max_inter) inter = max_inter;
                    float denom = qc + rc - inter;
                    float jacc = (denom > 1e-6f) ? (inter / denom) : 0.0f;
                    topk_insert(k_top, jacc, (int32_t)i,
                                top_jaccard_out + q * (size_t)k_top,
                                top_idx_out + q * (size_t)k_top);
                }
            }
        } else {
#ifdef _OPENMP
            float   *loc_s = (float *)malloc((size_t)n_threads * qt * (size_t)k_top * sizeof(float));
            int32_t *loc_i = (int32_t *)malloc((size_t)n_threads * qt * (size_t)k_top * sizeof(int32_t));
            for (size_t u = 0; u < (size_t)n_threads * qt * (size_t)k_top; ++u) {
                loc_s[u] = -1e30f;
                loc_i[u] = -1;
            }
            int team_T = 1;

            #pragma omp parallel num_threads(n_threads)
            {
                int tid = omp_get_thread_num();
                int actual_T = omp_get_num_threads();
                if (tid == 0) team_T = actual_T;
                if (tid < n_threads) {
                    float   *my_s = loc_s + (size_t)tid * qt * (size_t)k_top;
                    int32_t *my_i = loc_i + (size_t)tid * qt * (size_t)k_top;
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < n_rows; ++i) {
                        const int16_t *row = mat + i * DOTHASH_DIM;
                        float rc = row_cards[i];
                        size_t dq = 0;
#if HAVE_NEON
                        for (; dq + 4 <= qt; dq += 4) {
                            int64_t dots[4];
                            dot4_i16_2048_fast(
                                queries + (q0 + dq + 0) * DOTHASH_DIM,
                                queries + (q0 + dq + 1) * DOTHASH_DIM,
                                queries + (q0 + dq + 2) * DOTHASH_DIM,
                                queries + (q0 + dq + 3) * DOTHASH_DIM,
                                row, dots
                            );
                            for (size_t u = 0; u < 4; ++u) {
                                size_t q = q0 + dq + u;
                                float inter = (float)dots[u] * inv_d;
                                if (inter < 0.0f) inter = 0.0f;
                                float qc = query_cards[q];
                                float max_inter = (qc < rc) ? qc : rc;
                                if (inter > max_inter) inter = max_inter;
                                float denom = qc + rc - inter;
                                float jacc = (denom > 1e-6f) ? (inter / denom) : 0.0f;
                                topk_insert(k_top, jacc, (int32_t)i,
                                            my_s + (dq + u) * (size_t)k_top,
                                            my_i + (dq + u) * (size_t)k_top);
                            }
                        }
#endif
                        for (; dq < qt; ++dq) {
                            size_t q = q0 + dq;
                            int64_t dot = dot_i16_2048_fast(queries + q * DOTHASH_DIM, row);
                            float inter = (float)dot * inv_d;
                            if (inter < 0.0f) inter = 0.0f;
                            float qc = query_cards[q];
                            float max_inter = (qc < rc) ? qc : rc;
                            if (inter > max_inter) inter = max_inter;
                            float denom = qc + rc - inter;
                            float jacc = (denom > 1e-6f) ? (inter / denom) : 0.0f;
                            topk_insert(k_top, jacc, (int32_t)i,
                                        my_s + dq * (size_t)k_top,
                                        my_i + dq * (size_t)k_top);
                        }
                    }
                }
            }
            if (team_T > n_threads) team_T = n_threads;

            for (int t = 0; t < team_T; ++t) {
                for (size_t dq = 0; dq < qt; ++dq) {
                    size_t q = q0 + dq;
                    const float   *src_s = loc_s + ((size_t)t * qt + dq) * (size_t)k_top;
                    const int32_t *src_i = loc_i + ((size_t)t * qt + dq) * (size_t)k_top;
                    for (int c = 0; c < k_top; ++c) {
                        if (src_i[c] >= 0) {
                            topk_insert(k_top, src_s[c], src_i[c],
                                        top_jaccard_out + q * (size_t)k_top,
                                        top_idx_out + q * (size_t)k_top);
                        }
                    }
                }
            }
            free(loc_s);
            free(loc_i);
#endif
        }
    }
}
