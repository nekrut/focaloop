// pla_sieve.h — Bijective splitmix64-Permuted mmap PLA-index for glcore.c Syncmer Depletion Sieve
//
// Implements Transfer #3 (Paul Medvedev 15-yr Focaloop Synthesis):
//   - Abrar & Medvedev (WABI 2024): PLA-index optimal eps-tube piecewise linear index
//   - Abrar, Medvedev & Vinciguerra (ESA 2025): CaPLa canonical piecewise-linear approximability
//   - Chen, Li, Shi, Shao & Medvedev (Genome Res. 2026): Invertible 64-bit integer permutations
//
// Exact invariants:
//   1. splitmix64_bijective / splitmix64_inverse form a strict 1-to-1 bijection on [0, 2^64).
//   2. Zero false positives (exact 64-bit key comparison) & zero false negatives (verified |p_hat - i| <= eps).
//   3. Zero sentinel reservation: x = 0 and x = UINT64_MAX are supported natively without collision.
//   4. Zero-deserialization mmap binary layout: 128B header + radix16[65537] + pla_seg_t[M] + aligned perm_keys[N].

#ifndef PLA_SIEVE_H
#define PLA_SIEVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PLA_MAGIC            0x3156454953414c50ULL /* "PLASIEV1" */
#define PLA_VERSION          1U
#define PLA_RADIX_BITS       16U
#define PLA_RADIX_BUCKETS    (1U << PLA_RADIX_BITS) /* 65536 */
#define PLA_RADIX_LEN        (PLA_RADIX_BUCKETS + 1U) /* 65537 */

#define PLA_FLAG_PERMUTED    (1U << 0) /* keys stored as splitmix64_bijective(kmer) */
#define PLA_FLAG_HAS_FLAGS2B (1U << 1) /* optional 2-bit locus/bait flags array present */
#define PLA_FLAG_SPLITMIX_SYNCMER (1U << 2) /* canonical s-mers ordered by splitmix64_bijective */

// -----------------------------------------------------------------------------------------------
// 1. Invertible 64-bit Bijective Permutation (Stafford Mix13 / splitmix64 finalizer)
// -----------------------------------------------------------------------------------------------
_Static_assert((uint64_t)(0xbf58476d1ce4e5b9ULL * 0x96de1b173f119089ULL) == 1ULL,
               "splitmix64 multiplicative inverse 1 mismatch");
_Static_assert((uint64_t)(0x94d049bb133111ebULL * 0x319642b2d24d8ec3ULL) == 1ULL,
               "splitmix64 multiplicative inverse 2 mismatch");

static inline uint64_t splitmix64_bijective(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static inline uint64_t splitmix64_inverse(uint64_t x) {
    x ^= (x >> 31) ^ (x >> 62);
    x *= 0x319642b2d24d8ec3ULL;
    x ^= (x >> 27) ^ (x >> 54);
    x *= 0x96de1b173f119089ULL;
    x ^= (x >> 30) ^ (x >> 60);
    return x;
}

// -----------------------------------------------------------------------------------------------
// 2. PLA Segment & Zero-Deserialization mmap Header
// -----------------------------------------------------------------------------------------------
typedef struct {
    uint64_t x_start;   // First permuted (or raw) key covered by this segment
    uint32_t intercept; // Base rank prediction at x == x_start
    float    slope;     // Non-negative quantized slope (d_rank / d_key)
} pla_seg_t;

_Static_assert(sizeof(pla_seg_t) == 16, "pla_seg_t must be exactly 16 bytes");

typedef struct {
    uint64_t magic;        // PLA_MAGIC
    uint32_t version;      // PLA_VERSION
    uint32_t eps;          // Tube half-width epsilon (max_i |p_hat(key[i]) - i| <= eps)
    uint32_t k;            // Syncmer k (default 31)
    uint32_t s;            // Syncmer s (default 15)
    uint32_t flags;        // Bitmask of PLA_FLAG_*
    uint32_t reserved0;    // 0
    uint64_t n_keys;       // N: number of sorted distinct keys
    uint64_t n_segs;       // M: number of pla_seg_t segments
    uint64_t off_radix16;  // Byte offset to uint32_t radix16[65537]
    uint64_t off_segs;     // Byte offset to pla_seg_t segs[M] (128B/64B aligned)
    uint64_t off_keys;     // Byte offset to uint64_t perm_keys[N] (128B/64B aligned)
    uint64_t off_flags2b;  // Byte offset to uint8_t flags2b[(N+3)/4] or 0 if absent
    uint64_t total_bytes;  // Total serialized byte length
    uint64_t reserved[5];  // Pad header to 128 bytes (exact Apple M4 cache line)
} pla_mmap_hdr_t;

_Static_assert(sizeof(pla_mmap_hdr_t) == 128, "pla_mmap_hdr_t must be 128 bytes");

typedef struct {
    uint32_t        eps;
    uint32_t        k;
    uint32_t        s;
    uint32_t        flags;
    size_t          n_keys;
    size_t          n_segs;
    const uint32_t  *radix16;   // [65537]
    const pla_seg_t *segs;      // [n_segs]
    const uint64_t  *perm_keys; // [n_keys], 64/128-byte aligned
    const uint8_t   *flags2b;   // [(n_keys + 3)/4] or NULL
    void            *backing;   // malloc or mmap base pointer
    size_t          backing_bytes;
    int             is_mmap;    // 1 if backed by mmap, 0 if heap
    int             fd;         // file descriptor if mmap'd, else -1
} pla_index_t;

typedef struct {
    uint64_t n_keys;
    uint64_t n_segs;
    uint32_t eps;
    uint32_t use_perm;
    uint64_t empty_buckets;
    uint64_t max_keys_per_bucket;
    uint64_t max_segs_per_bucket;
    double   mean_keys_per_seg;
    uint64_t index_overhead_bytes; // radix16 + segs + header (excludes perm_keys[N])
    uint64_t keys_bytes;           // N * 8
    uint64_t total_bytes;          // exact serialized mmap size
    double   overhead_bytes_per_key;
    double   total_bytes_per_key;
    int64_t  max_abs_err;          // verified max_i |p_hat(perm_keys[i]) - i|
} pla_stats_t;

// -----------------------------------------------------------------------------------------------
// 3. Hot-Path Inline Prediction & SIMD-Accelerated Lookup (Zero FPR, Zero FNR)
// -----------------------------------------------------------------------------------------------

// Safe rank prediction with explicit unsigned-underflow guard and double->int64 overflow clamp.
static inline int64_t pla_predict_seg(const pla_seg_t *seg, uint64_t x, size_t n_keys) {
    if (n_keys == 0) return 0;
    if (x <= seg->x_start) {
        return (seg->intercept < n_keys) ? (int64_t)seg->intercept : (int64_t)(n_keys - 1);
    }
    double dx   = (double)(x - seg->x_start);
    double pred = (double)seg->intercept + dx * (double)seg->slope;
    double maxp = (double)(n_keys - 1);
    if (pred >= maxp) return (int64_t)(n_keys - 1);
    if (pred <= 0.0)  return 0;
    return (int64_t)pred;
}

// Find predecessor segment index in [0, n_segs - 1] via 16-bit radix directory + bounded binary search.
static inline uint32_t pla_find_seg(
    const uint32_t * restrict radix16,
    const pla_seg_t * restrict segs,
    uint64_t x
) {
    uint32_t b  = (uint32_t)(x >> 48);
    uint32_t lo = radix16[b] ? (radix16[b] - 1U) : 0U;
    uint32_t hi = radix16[b + 1U];
    while (lo + 1U < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (segs[mid].x_start <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Core PLA lookup: returns exact rank i in [0, N-1] if key is present, or -1 if absent.
static inline int64_t pla_lookup_inline(
    uint64_t query_key,
    const uint64_t * restrict perm_keys,
    const pla_seg_t * restrict segs,
    const uint32_t * restrict radix16,
    size_t n_keys,
    uint32_t eps,
    int use_perm
) {
    if (__builtin_expect(n_keys == 0, 0)) return -1;
    uint64_t x  = use_perm ? splitmix64_bijective(query_key) : query_key;
    uint32_t lo = pla_find_seg(radix16, segs, x);
    int64_t p_hat = pla_predict_seg(&segs[lo], x, n_keys);

    int64_t e     = (int64_t)eps;
    int64_t start = (p_hat > e) ? (p_hat - e) : 0;
    int64_t end   = (p_hat + e < (int64_t)n_keys) ? (p_hat + e) : ((int64_t)n_keys - 1);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // Branchless ARM64 NEON scan: since perm_keys is strictly sorted, at most one lane equals x.
    // Non-matching lanes yield ~0ULL (-1); matching lane yields its exact index i >= 0.
    // Bitwise AND across all lanes yields i if present, or -1 (~0ULL) if absent, with 0 data branches.
    uint64x2_t vx    = vdupq_n_u64(x);
    uint64x2_t vres0 = vdupq_n_u64(~0ULL);
    uint64x2_t vres1 = vdupq_n_u64(~0ULL);
    uint64_t base_i  = (uint64_t)start;
    uint64x2_t vidx0 = { base_i, base_i + 1ULL };
    uint64x2_t vidx1 = { base_i + 2ULL, base_i + 3ULL };
    const uint64x2_t vstep4 = vdupq_n_u64(4ULL);

    int64_t i = start;
    for (; i + 3 <= end; i += 4) {
        uint64x2_t vk0 = vld1q_u64(&perm_keys[i]);
        uint64x2_t vk1 = vld1q_u64(&perm_keys[i + 2]);
        uint64x2_t c0  = vceqq_u64(vk0, vx);
        uint64x2_t c1  = vceqq_u64(vk1, vx);
        vres0 = vbslq_u64(c0, vidx0, vres0);
        vres1 = vbslq_u64(c1, vidx1, vres1);
        vidx0 = vaddq_u64(vidx0, vstep4);
        vidx1 = vaddq_u64(vidx1, vstep4);
    }
    uint64x2_t vres = vandq_u64(vres0, vres1);
    uint64_t r = vgetq_lane_u64(vres, 0) & vgetq_lane_u64(vres, 1);
    for (; i <= end; ++i) {
        uint64_t mask = (perm_keys[i] == x) ? 0ULL : ~0ULL;
        r &= ((uint64_t)i | mask);
    }
    return (int64_t)r;
#else
    int64_t res = -1;
    for (int64_t i = start; i <= end; ++i) {
        if (perm_keys[i] == x) res = i;
    }
    return res;
#endif
}

static inline uint8_t pla_get_flag2b(const pla_index_t *idx, int64_t rank) {
    if (!idx->flags2b || rank < 0) return 0;
    size_t u = (size_t)rank;
    return (uint8_t)((idx->flags2b[u >> 2] >> ((u & 3U) * 2U)) & 3U);
}

// -----------------------------------------------------------------------------------------------
// 4. Library API (Build, Verify, mmap I/O, Batch Query, Syncmers, glcore Sieve & htab Baselines)
// -----------------------------------------------------------------------------------------------

size_t pla_sort_dedup_u64(uint64_t *keys, size_t n, int num_threads);

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
);

int pla_verify_all(const pla_index_t *idx, int64_t *out_max_abs_err, int num_threads);
void pla_compute_stats(const pla_index_t *idx, pla_stats_t *st);

int pla_save_mmap(const pla_index_t *idx, const char *path);
int pla_open_mmap(pla_index_t *out, const char *path, int map_private, int prefault_spline);
void pla_close(pla_index_t *idx);

int64_t pla_lookup(const pla_index_t *idx, uint64_t query_key);
void pla_lookup_batch(
    const pla_index_t *idx,
    const uint64_t *queries,
    int64_t *out_ranks,
    size_t n_queries,
    int num_threads
);
uint64_t pla_count_hits_batch(
    const pla_index_t *idx,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
);
void pla_splitmix64_forward_batch(const uint64_t *in, uint64_t *out, size_t n);
void pla_splitmix64_inverse_batch(const uint64_t *in, uint64_t *out, size_t n);

int64_t pla_bsearch_lookup(const pla_index_t *idx, uint64_t query_key);
uint64_t pla_bsearch_count_hits_batch(
    const pla_index_t *idx,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
);

typedef struct {
    uint64_t *keys;
    uint32_t *vals;
    uint64_t cap;
    uint64_t mask;
    uint64_t n;
} gl_htab_t;

void gl_ht_init(gl_htab_t *h, uint64_t want);
void gl_ht_free(gl_htab_t *h);
uint64_t gl_ht_insert(gl_htab_t *h, uint64_t k, uint32_t val_or);
uint64_t gl_ht_find(const gl_htab_t *h, uint64_t k);
int gl_ht_save_table(const gl_htab_t *h, const char *path, int k, int s);
int gl_ht_load_table(const char *path, gl_htab_t *h, int *k, int *s);
uint64_t gl_ht_count_hits_batch(
    const gl_htab_t *h,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
);

// Verbatim malariavar/kernels.py (lines 43-70) keys-only open-addressed uint64_t hash set (EMPTY=0, stores key+1).
typedef struct {
    uint64_t *tab;
    uint64_t cap;
    uint64_t mask;
    uint64_t n;
} mv_kset_t;

void mv_kset_init(mv_kset_t *ks, uint64_t want);
void mv_kset_free(mv_kset_t *ks);
int mv_kset_insert(mv_kset_t *ks, uint64_t key);
int mv_kset_has(const mv_kset_t *ks, uint64_t key);
uint64_t mv_kset_count_hits_batch(
    const mv_kset_t *ks,
    const uint64_t *queries,
    size_t n_queries,
    int num_threads
);

size_t gl_extract_open_syncmers(
    const char *seq,
    size_t len,
    int k,
    int s,
    int skip_lower,
    uint64_t *out_keys,
    size_t max_out
);

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
);

int pla_read_pf_hits(
    const pla_index_t *idx,
    const uint8_t *buf,
    size_t s0,
    size_t len,
    int need
);

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
);

typedef struct {
    uint64_t pairs_in;
    uint64_t pairs_out;
    uint64_t drop_locus;
    uint64_t drop_bait;
} gl_sieve_stats_t;

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
);

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
);

int pla_generate_distinct_syncmers(
    uint64_t *out_keys,
    size_t n_target,
    int mode_at81,
    int k,
    int s,
    uint64_t seed,
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif // PLA_SIEVE_H
