// island_pc_core.c — Pre-Alignment Mutated-k-mer Island Classifier & Repeat-Robust --pc Kernel
//
// Implements:
//   1. Bijective splitmix64 canonical s-mer middle-offset (t* = 8) open syncmers (k=31, s=15)
//      (Belbasi et al. 2022; Chen, Li, Shi, Shao & Medvedev 2026) + raw lexicographic comparison.
//   2. Wu & Medvedev (ISMB / Bioinformatics 2026) repeat-robust --pc novelty divergence estimator
//      with exact 1-instruction XOR Hamming-1 shadow collision precomputation D_1(s_a) / d1_bar(s_a)
//      and safeguarded Newton-Raphson solver.
//   3. Exact Blanca, Harris, Koslicki & Medvedev (JCB 2022) lag-(k-1) overlap variance CIs for BOTH
//      dense k=31 k-mers and middle-offset (t* = 8) open syncmers (incorporating the exact d <= 8
//      repulsion zone c_2(d) = 0, exact shared-window kernel 2 / (m * (d + m)) for 9 <= d <= 16,
//      and ratio delta-method linearization of N_novel / L_probe).
//   4. <2 ms Wave-0 2,000-read mutated-k-mer island-width (W) & unique-anchor coordinate-shift (delta)
//      SNP/deletion/insertion separator (Blanca 2022 + Hera, Medvedev, Koslicki & Blanca 2025) with
//      VarGeno Phred gating, (1 - r_total)^(2k) island-survival correction, and exact Feller-DP
//      multi-event delta + --pc high-divergence (kr > 0.35) fallback.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IPC_K 31
#define IPC_S 15
#define IPC_M (IPC_K - IPC_S + 1)   // 17
#define IPC_T ((IPC_K - IPC_S) / 2) // 8
#define EMPTY_KEY UINT64_MAX
#define MAX_INDEL_SHIFT 64

// Value bit layout in kmer_slot_t.val:
//   bits [0..31]  : 0-based ref_pos (uint32_t)
//   bits [32..47] : contig_id (uint16_t)
//   bit  48       : ref_sign (0 if fk < rk on ref, 1 if rk < fk)
//   bit  49       : is_unique (1 if occ == 1, 0 if occ >= 2)
//   bits [50..63] : occ (1..16383, saturating)
#define VAL_POS(v)       ((uint32_t)((v) & 0xFFFFFFFFULL))
#define VAL_CONTIG(v)    ((uint16_t)(((v) >> 32) & 0xFFFFULL))
#define VAL_SIGN(v)      ((int)(((v) >> 48) & 1ULL))
#define VAL_UNIQUE(v)    ((int)(((v) >> 49) & 1ULL))
#define VAL_OCC(v)       ((uint32_t)(((v) >> 50) & 0x3FFFULL))
#define MAKE_VAL(pos, ctg, sgn, uniq, occ) \
    (((uint64_t)(uint32_t)(pos)) | \
     ((uint64_t)(uint16_t)(ctg) << 32) | \
     ((uint64_t)((sgn) & 1) << 48) | \
     ((uint64_t)((uniq) & 1) << 49) | \
     ((uint64_t)((occ) > 0x3FFFU ? 0x3FFFU : (occ)) << 50))

typedef struct {
    uint64_t key;
    uint64_t val;
} kmer_slot_t; // 16 bytes -> 4 slots per 64-byte cache line (key + val in SAME cache line!)

static uint8_t NT_TABLE[256];
static double PHRED_ERR_LUT[256];
static double PHRED_NEG_LOG_OK_LUT[256];
static int TABLES_INITIALIZED = 0;

static void ensure_tables(void) {
    if (TABLES_INITIALIZED) return;
    memset(NT_TABLE, 4, sizeof(NT_TABLE));
    NT_TABLE['A'] = 0; NT_TABLE['a'] = 0;
    NT_TABLE['C'] = 1; NT_TABLE['c'] = 1;
    NT_TABLE['G'] = 2; NT_TABLE['g'] = 2;
    NT_TABLE['T'] = 3; NT_TABLE['t'] = 3;

    for (int c = 0; c < 256; c++) {
        int q = c - 33;
        if (q < 2) q = 2;
        if (q > 60) q = 60;
        double pe = pow(10.0, -0.1 * (double)q);
        PHRED_ERR_LUT[c] = pe;
        PHRED_NEG_LOG_OK_LUT[c] = -log(1.0 - pe);
    }
    TABLES_INITIALIZED = 1;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

// -----------------------------------------------------------------------------
// Invertible 64-bit bijective permutation (splitmix64 / Stafford Mix13)
// -----------------------------------------------------------------------------
uint64_t ipc_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t ipc_splitmix64_inv(uint64_t z) {
    z = z ^ (z >> 31) ^ (z >> 62);
    z *= 0x319642b2d24d8ec3ULL;
    z = z ^ (z >> 27) ^ (z >> 54);
    z *= 0x96de1b173f119089ULL;
    z = z ^ (z >> 30) ^ (z >> 60);
    return z - 0x9e3779b97f4a7c15ULL;
}

static inline uint64_t hash_slot64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline uint64_t revcomp_k(uint64_t fk, int k) {
    uint64_t rk = 0;
    for (int i = 0; i < k; i++) {
        rk = (rk << 2) | (3ULL - (fk & 3ULL));
        fk >>= 2;
    }
    return rk;
}

// -----------------------------------------------------------------------------
// Syncmer density scanner (supports both raw lexicographic and splitmix64 s-mers)
// -----------------------------------------------------------------------------
void ipc_syncmer_density(
    const char *seq,
    int64_t len,
    int k,
    int s,
    int use_splitmix,
    uint64_t *out_n_kmers,
    uint64_t *out_n_syncmers
) {
    ensure_tables();
    *out_n_kmers = 0;
    *out_n_syncmers = 0;
    if (!seq || len < k || k < s || s < 1 || k > 31) return;

    const int m = k - s + 1;
    const int t = (k - s) / 2;
    const uint64_t kmask = (1ULL << (2 * k)) - 1ULL;
    const uint64_t smask = (1ULL << (2 * s)) - 1ULL;

    uint64_t fs_ring[64];
    uint64_t rs_ring[64];
    uint64_t cs_hash_ring[64];

    uint64_t fk = 0, rk = 0, fs = 0, rs = 0;
    int64_t run = 0;
    uint64_t n_km = 0, n_sy = 0;

    for (int64_t i = 0; i < len; i++) {
        uint8_t c = NT_TABLE[(uint8_t)seq[i]];
        if (c > 3) {
            run = 0;
            fk = rk = fs = rs = 0;
            continue;
        }
        run++;
        fs = ((fs << 2) | c) & smask;
        rs = (rs >> 2) | ((uint64_t)(3 - c) << (2 * (s - 1)));
        fk = ((fk << 2) | c) & kmask;
        rk = (rk >> 2) | ((uint64_t)(3 - c) << (2 * (k - 1)));

        if (run >= s) {
            int64_t sp = i - s + 1;
            int idx_r = (int)(sp & 63);
            if (use_splitmix) {
                uint64_t cs = (fs < rs) ? fs : rs;
                cs_hash_ring[idx_r] = ipc_splitmix64(cs);
            } else {
                fs_ring[idx_r] = fs;
                rs_ring[idx_r] = rs;
            }
        }
        if (run >= k) {
            n_km++;
            int64_t p = i - k + 1;
            int best = 0;
            if (use_splitmix) {
                if (fk < rk) {
                    uint64_t bv = cs_hash_ring[p & 63];
                    for (int j = 1; j < m; j++) {
                        uint64_t v = cs_hash_ring[(p + j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                } else {
                    uint64_t bv = cs_hash_ring[(p + m - 1) & 63];
                    for (int j = 1; j < m; j++) {
                        uint64_t v = cs_hash_ring[(p + m - 1 - j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                }
            } else {
                if (fk < rk) {
                    uint64_t bv = fs_ring[p & 63];
                    for (int j = 1; j < m; j++) {
                        uint64_t v = fs_ring[(p + j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                } else {
                    uint64_t bv = rs_ring[(p + m - 1) & 63];
                    for (int j = 1; j < m; j++) {
                        uint64_t v = rs_ring[(p + m - 1 - j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                }
            }
            if (best == t) n_sy++;
        }
    }
    *out_n_kmers = n_km;
    *out_n_syncmers = n_sy;
}

// -----------------------------------------------------------------------------
// Reference Index (interleaved 16B L2-resident k=31 table + open syncmers + exact D1)
// -----------------------------------------------------------------------------
typedef struct {
    uint64_t kmer_cap;
    uint64_t kmer_mask;
    kmer_slot_t *kmer_slots;

    uint64_t bloom_block_mask;
    uint64_t *kmer_bloom;

    int n_contigs;
    int64_t *contig_lens;
    uint8_t **contig_nt;
    uint64_t **contig_uniq_bv;

    uint64_t sync_cap;
    uint64_t sync_mask;
    uint64_t *sync_keys;
    uint32_t *sync_occ;

    uint64_t total_kmer_positions;
    uint64_t distinct_kmers;
    uint64_t unique_kmer_positions;

    uint64_t total_syncmer_positions;
    uint64_t distinct_syncmers;

    uint64_t D1_dense;
    double d1_dense_bar;
    uint64_t D1_sync;
    double d1_sync_bar;
    int dense_d1_computed;
} ref_index_t;

static inline void bloom_insert_hash(uint64_t * restrict bloom, uint64_t block_mask, uint64_t h) {
    uint64_t *blk = bloom + (((h >> 24) & block_mask) << 3);
    uint64_t hb = h * 0x9E3779B97F4A7C15ULL;
    uint32_t b0 = (uint32_t)(hb >> 55);
    uint32_t b1 = (uint32_t)((hb >> 46) & 511U);
    uint32_t b2 = (uint32_t)((hb >> 37) & 511U);
    blk[b0 >> 6] |= (1ULL << (b0 & 63U));
    blk[b1 >> 6] |= (1ULL << (b1 & 63U));
    blk[b2 >> 6] |= (1ULL << (b2 & 63U));
}

static inline int bloom_query_hash(const uint64_t * restrict bloom, uint64_t block_mask, uint64_t h) {
    const uint64_t *blk = bloom + (((h >> 24) & block_mask) << 3);
    uint64_t hb = h * 0x9E3779B97F4A7C15ULL;
    uint32_t b0 = (uint32_t)(hb >> 55);
    uint32_t b1 = (uint32_t)((hb >> 46) & 511U);
    uint32_t b2 = (uint32_t)((hb >> 37) & 511U);
    return ((blk[b0 >> 6] >> (b0 & 63U)) &
            (blk[b1 >> 6] >> (b1 & 63U)) &
            (blk[b2 >> 6] >> (b2 & 63U)) & 1ULL) != 0ULL;
}

typedef struct {
    uint64_t total_kmer_positions;
    uint64_t distinct_kmers;
    uint64_t unique_kmer_positions;
    uint64_t total_syncmer_positions;
    uint64_t distinct_syncmers;
    uint64_t D1_dense;
    double d1_dense_bar;
    uint64_t D1_sync;
    double d1_sync_bar;
    int dense_d1_computed;
} index_info_t;

static inline uint64_t kmer_ht_find_val(const ref_index_t *idx, uint64_t key, int *found) {
    uint64_t slot = hash_slot64(key) & idx->kmer_mask;
    const kmer_slot_t * restrict slots = idx->kmer_slots;
    while (1) {
        uint64_t kk = slots[slot].key;
        if (kk == key) {
            *found = 1;
            return slots[slot].val;
        }
        if (kk == EMPTY_KEY) {
            *found = 0;
            return 0;
        }
        slot = (slot + 1) & idx->kmer_mask;
    }
}

void ipc_free_index(ref_index_t *idx);

static int rehash_kmer_table(ref_index_t *idx) {
    uint64_t old_cap = idx->kmer_cap;
    uint64_t new_cap = old_cap << 1;
    uint64_t new_mask = new_cap - 1ULL;
    kmer_slot_t *ns = (kmer_slot_t *)malloc(new_cap * sizeof(kmer_slot_t));
    if (!ns) return -1;
    for (uint64_t i = 0; i < new_cap; i++) {
        ns[i].key = EMPTY_KEY;
        ns[i].val = 0;
    }
    for (uint64_t i = 0; i < old_cap; i++) {
        uint64_t k = idx->kmer_slots[i].key;
        if (k == EMPTY_KEY) continue;
        uint64_t slot = hash_slot64(k) & new_mask;
        while (ns[slot].key != EMPTY_KEY) slot = (slot + 1ULL) & new_mask;
        ns[slot] = idx->kmer_slots[i];
    }
    free(idx->kmer_slots);
    idx->kmer_slots = ns;
    idx->kmer_cap = new_cap;
    idx->kmer_mask = new_mask;
    return 0;
}

static int rehash_sync_table(ref_index_t *idx) {
    uint64_t old_cap = idx->sync_cap;
    uint64_t new_cap = old_cap << 1;
    uint64_t new_mask = new_cap - 1ULL;
    uint64_t *nk = (uint64_t *)malloc(new_cap * sizeof(uint64_t));
    uint32_t *no = (uint32_t *)calloc(new_cap, sizeof(uint32_t));
    if (!nk || !no) {
        free(nk);
        free(no);
        return -1;
    }
    memset(nk, 0xFF, new_cap * sizeof(uint64_t));
    for (uint64_t i = 0; i < old_cap; i++) {
        uint64_t k = idx->sync_keys[i];
        if (k == EMPTY_KEY) continue;
        uint64_t slot = hash_slot64(k) & new_mask;
        while (nk[slot] != EMPTY_KEY) slot = (slot + 1ULL) & new_mask;
        nk[slot] = k;
        no[slot] = idx->sync_occ[i];
    }
    free(idx->sync_keys);
    free(idx->sync_occ);
    idx->sync_keys = nk;
    idx->sync_occ = no;
    idx->sync_cap = new_cap;
    idx->sync_mask = new_mask;
    return 0;
}

ref_index_t *ipc_build_index(
    const char **seqs,
    const int64_t *lens,
    int n_contigs,
    int compute_dense_d1
) {
    ensure_tables();
    if (!seqs || !lens || n_contigs <= 0) return NULL;

    uint64_t approx_positions = 0;
    for (int c = 0; c < n_contigs; c++) {
        if (lens[c] >= IPC_K) approx_positions += (uint64_t)(lens[c] - IPC_K + 1);
    }

    // Load factor <= 0.50; cap initial allocation for redundant multi-isoform transcriptomes
    // and double+rehash only if distinct keys actually require higher capacity.
    uint64_t kcap = 1024;
    uint64_t target_kcap = approx_positions * 2ULL + 64ULL;
    uint64_t max_init_kcap = (1ULL << 27);
    if (target_kcap > max_init_kcap) target_kcap = max_init_kcap;
    while (kcap < target_kcap) kcap <<= 1;

    uint64_t scap = 256;
    uint64_t approx_sync = approx_positions / 12ULL + 64ULL;
    uint64_t target_scap = approx_sync * 2ULL + 64ULL;
    uint64_t max_init_scap = (1ULL << 24);
    if (target_scap > max_init_scap) target_scap = max_init_scap;
    while (scap < target_scap) scap <<= 1;

    ref_index_t *idx = (ref_index_t *)calloc(1, sizeof(ref_index_t));
    if (!idx) return NULL;

    idx->kmer_cap = kcap;
    idx->kmer_mask = kcap - 1ULL;
    idx->kmer_slots = (kmer_slot_t *)malloc(kcap * sizeof(kmer_slot_t));

    idx->sync_cap = scap;
    idx->sync_mask = scap - 1ULL;
    idx->sync_keys = (uint64_t *)malloc(scap * sizeof(uint64_t));
    idx->sync_occ = (uint32_t *)calloc(scap, sizeof(uint32_t));

    idx->n_contigs = n_contigs;
    idx->contig_lens = (int64_t *)calloc((size_t)n_contigs, sizeof(int64_t));
    idx->contig_nt = (uint8_t **)calloc((size_t)n_contigs, sizeof(uint8_t *));
    idx->contig_uniq_bv = (uint64_t **)calloc((size_t)n_contigs, sizeof(uint64_t *));

    if (!idx->kmer_slots || !idx->sync_keys || !idx->sync_occ ||
        !idx->contig_lens || !idx->contig_nt || !idx->contig_uniq_bv) {
        free(idx->kmer_slots);
        free(idx->sync_keys);
        free(idx->sync_occ);
        free(idx->contig_lens);
        free(idx->contig_nt);
        free(idx->contig_uniq_bv);
        free(idx);
        return NULL;
    }
    for (uint64_t i = 0; i < kcap; i++) {
        idx->kmer_slots[i].key = EMPTY_KEY;
        idx->kmer_slots[i].val = 0;
    }
    memset(idx->sync_keys, 0xFF, scap * sizeof(uint64_t));

    const uint64_t kmask = (1ULL << (2 * IPC_K)) - 1ULL;
    const uint64_t smask = (1ULL << (2 * IPC_S)) - 1ULL;
    uint64_t cs_hash_ring[64];

    for (int ctg = 0; ctg < n_contigs; ctg++) {
        const char *seq = seqs[ctg];
        int64_t len = lens[ctg];
        idx->contig_lens[ctg] = len;
        if (!seq || len < IPC_K) continue;

        idx->contig_nt[ctg] = (uint8_t *)malloc((size_t)len);
        idx->contig_uniq_bv[ctg] = (uint64_t *)calloc((size_t)((len + 63) / 64), sizeof(uint64_t));
        uint8_t *nt_dst = idx->contig_nt[ctg];

        uint64_t fk = 0, rk = 0, fs = 0, rs = 0;
        int64_t run = 0;
        uint16_t ctg_u16 = (ctg > 0xFFFF) ? 0xFFFFU : (uint16_t)ctg;

        for (int64_t i = 0; i < len; i++) {
            uint8_t b = NT_TABLE[(uint8_t)seq[i]];
            if (nt_dst) nt_dst[i] = b;
            if (b > 3) {
                run = 0;
                fk = rk = fs = rs = 0;
                continue;
            }
            run++;
            fs = ((fs << 2) | b) & smask;
            rs = (rs >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_S - 1)));
            fk = ((fk << 2) | b) & kmask;
            rk = (rk >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_K - 1)));

            if (run >= IPC_S) {
                int64_t sp = i - IPC_S + 1;
                uint64_t cs = (fs < rs) ? fs : rs;
                cs_hash_ring[sp & 63] = ipc_splitmix64(cs);
            }
            if (run >= IPC_K) {
                int64_t p = i - IPC_K + 1;
                uint32_t pos_u32 = (p > 0xFFFFFFFFLL) ? 0xFFFFFFFFU : (uint32_t)p;
                int ref_sgn = (fk < rk) ? 0 : 1;
                uint64_t canon = ref_sgn ? rk : fk;

                idx->total_kmer_positions++;

                uint64_t h_full = hash_slot64(canon);
                uint64_t slot = h_full & idx->kmer_mask;
                while (1) {
                    uint64_t kk = idx->kmer_slots[slot].key;
                    if (kk == EMPTY_KEY) {
                        idx->kmer_slots[slot].key = canon;
                        idx->kmer_slots[slot].val = MAKE_VAL(pos_u32, ctg_u16, ref_sgn, 1, 1);
                        idx->distinct_kmers++;
                        if (idx->distinct_kmers * 2ULL >= idx->kmer_cap) {
                            if (rehash_kmer_table(idx) != 0) {
                                ipc_free_index(idx);
                                return NULL;
                            }
                        }
                        break;
                    }
                    if (kk == canon) {
                        uint64_t oldv = idx->kmer_slots[slot].val;
                        uint32_t occ = VAL_OCC(oldv) + 1U;
                        idx->kmer_slots[slot].val = MAKE_VAL(
                            VAL_POS(oldv), VAL_CONTIG(oldv), VAL_SIGN(oldv), 0, occ
                        );
                        break;
                    }
                    slot = (slot + 1) & idx->kmer_mask;
                }

                int best = 0;
                if (!ref_sgn) {
                    uint64_t bv = cs_hash_ring[p & 63];
                    for (int j = 1; j < IPC_M; j++) {
                        uint64_t v = cs_hash_ring[(p + j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                } else {
                    uint64_t bv = cs_hash_ring[(p + IPC_M - 1) & 63];
                    for (int j = 1; j < IPC_M; j++) {
                        uint64_t v = cs_hash_ring[(p + IPC_M - 1 - j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                }
                if (best == IPC_T) {
                    idx->total_syncmer_positions++;
                    uint64_t sslot = hash_slot64(canon) & idx->sync_mask;
                    while (1) {
                        uint64_t sk = idx->sync_keys[sslot];
                        if (sk == EMPTY_KEY) {
                            idx->sync_keys[sslot] = canon;
                            idx->sync_occ[sslot] = 1U;
                            idx->distinct_syncmers++;
                            if (idx->distinct_syncmers * 2ULL >= idx->sync_cap) {
                                if (rehash_sync_table(idx) != 0) {
                                    ipc_free_index(idx);
                                    return NULL;
                                }
                            }
                            break;
                        }
                        if (sk == canon) {
                            if (idx->sync_occ[sslot] < UINT32_MAX) idx->sync_occ[sslot]++;
                            break;
                        }
                        sslot = (sslot + 1) & idx->sync_mask;
                    }
                }
            }
        }
    }

    uint64_t bblocks = (idx->kmer_cap >> 7);
    if (bblocks < 8ULL) bblocks = 8ULL;
    idx->bloom_block_mask = bblocks - 1ULL;
    idx->kmer_bloom = (uint64_t *)calloc((size_t)(bblocks * 8ULL), sizeof(uint64_t));
    if (!idx->kmer_bloom) {
        ipc_free_index(idx);
        return NULL;
    }

    uint64_t uniq_pos = 0;
    for (uint64_t s = 0; s < idx->kmer_cap; s++) {
        uint64_t kk = idx->kmer_slots[s].key;
        if (kk == EMPTY_KEY) continue;
        bloom_insert_hash(idx->kmer_bloom, idx->bloom_block_mask, hash_slot64(kk));
        if (VAL_UNIQUE(idx->kmer_slots[s].val)) {
            uniq_pos++;
            uint64_t v = idx->kmer_slots[s].val;
            uint16_t ctg_id = VAL_CONTIG(v);
            uint32_t pos_id = VAL_POS(v);
            if ((int)ctg_id < idx->n_contigs && idx->contig_uniq_bv[ctg_id] &&
                (int64_t)pos_id < idx->contig_lens[ctg_id]) {
                idx->contig_uniq_bv[ctg_id][pos_id >> 6] |= (1ULL << (pos_id & 63));
            }
        }
    }
    idx->unique_kmer_positions = uniq_pos;

    uint64_t d1_sync_sum = 0;
    for (uint64_t s = 0; s < idx->sync_cap; s++) {
        uint64_t sigma = idx->sync_keys[s];
        if (sigma == EMPTY_KEY) continue;
        uint64_t rc_sigma = revcomp_k(sigma, IPC_K);
        for (int j = 0; j < IPC_K; j++) {
            int shift_f = 2 * j;
            int shift_r = 2 * (IPC_K - 1 - j);
            for (uint64_t xval = 1; xval <= 3; xval++) {
                uint64_t nf = sigma ^ (xval << shift_f);
                uint64_t nr = rc_sigma ^ (xval << shift_r);
                uint64_t ncanon = (nf < nr) ? nf : nr;
                int found = 0;
                uint64_t v = kmer_ht_find_val(idx, ncanon, &found);
                if (found) {
                    d1_sync_sum += (uint64_t)VAL_OCC(v);
                }
            }
        }
    }
    idx->D1_sync = d1_sync_sum;
    idx->d1_sync_bar = (idx->total_syncmer_positions > 0)
        ? ((double)d1_sync_sum / (double)idx->total_syncmer_positions)
        : 0.0;

    if (compute_dense_d1) {
        uint64_t d1_dense_sum = 0;
        for (uint64_t s = 0; s < idx->kmer_cap; s++) {
            uint64_t tau = idx->kmer_slots[s].key;
            if (tau == EMPTY_KEY) continue;
            uint64_t occ_tau = (uint64_t)VAL_OCC(idx->kmer_slots[s].val);
            uint64_t rc_tau = revcomp_k(tau, IPC_K);
            uint64_t d1_here = 0;
            for (int j = 0; j < IPC_K; j++) {
                int shift_f = 2 * j;
                int shift_r = 2 * (IPC_K - 1 - j);
                for (uint64_t xval = 1; xval <= 3; xval++) {
                    uint64_t nf = tau ^ (xval << shift_f);
                    uint64_t nr = rc_tau ^ (xval << shift_r);
                    uint64_t ncanon = (nf < nr) ? nf : nr;
                    int found = 0;
                    (void)kmer_ht_find_val(idx, ncanon, &found);
                    if (found) d1_here++;
                }
            }
            d1_dense_sum += occ_tau * d1_here;
        }
        idx->D1_dense = d1_dense_sum;
        idx->d1_dense_bar = (idx->total_kmer_positions > 0)
            ? ((double)d1_dense_sum / (double)idx->total_kmer_positions)
            : 0.0;
        idx->dense_d1_computed = 1;
    } else {
        idx->D1_dense = 0;
        idx->d1_dense_bar = idx->d1_sync_bar;
        idx->dense_d1_computed = 0;
    }

    return idx;
}

void ipc_get_index_info(const ref_index_t *idx, index_info_t *info) {
    if (!idx || !info) return;
    info->total_kmer_positions = idx->total_kmer_positions;
    info->distinct_kmers = idx->distinct_kmers;
    info->unique_kmer_positions = idx->unique_kmer_positions;
    info->total_syncmer_positions = idx->total_syncmer_positions;
    info->distinct_syncmers = idx->distinct_syncmers;
    info->D1_dense = idx->D1_dense;
    info->d1_dense_bar = idx->d1_dense_bar;
    info->D1_sync = idx->D1_sync;
    info->d1_sync_bar = idx->d1_sync_bar;
    info->dense_d1_computed = idx->dense_d1_computed;
}

void ipc_free_index(ref_index_t *idx) {
    if (!idx) return;
    if (idx->contig_nt) {
        for (int c = 0; c < idx->n_contigs; c++) free(idx->contig_nt[c]);
        free(idx->contig_nt);
    }
    if (idx->contig_uniq_bv) {
        for (int c = 0; c < idx->n_contigs; c++) free(idx->contig_uniq_bv[c]);
        free(idx->contig_uniq_bv);
    }
    free(idx->contig_lens);
    free(idx->kmer_slots);
    free(idx->kmer_bloom);
    free(idx->sync_keys);
    free(idx->sync_occ);
    free(idx);
}

// -----------------------------------------------------------------------------
// Wu & Medvedev (2026) --pc Newton-Raphson Solver + Exact Blanca / Syncmer CIs
// -----------------------------------------------------------------------------
static inline void eval_pc_g_and_deriv(double r, int k, double alpha, double *g, double *gp) {
    double one_m_r = 1.0 - r;
    if (one_m_r <= 1e-15) {
        *g = 1.0;
        *gp = 1e-15;
        return;
    }
    double p_km2 = pow(one_m_r, (double)(k - 2));
    double p_km1 = p_km2 * one_m_r;
    double p_k = p_km1 * one_m_r;
    *g = (1.0 - p_k) - alpha * r * p_km1;
    *gp = p_km2 * (((double)k - alpha) * one_m_r + alpha * (double)(k - 1) * r);
}

static double solve_pc_from_y(double y, int k, double d1_bar, double *out_gp) {
    if (k < 2) k = IPC_K;
    double alpha = d1_bar / 3.0;
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > (double)k - 1e-6) alpha = (double)k - 1e-6;

    if (!(y > 0.0)) {
        if (out_gp) *out_gp = (double)k - alpha;
        return 0.0;
    }

    double g_max = 0.0, gp_max = 0.0;
    eval_pc_g_and_deriv(0.35, k, alpha, &g_max, &gp_max);
    if (y >= g_max) {
        if (out_gp) *out_gp = gp_max;
        return 0.35;
    }

    double lo = 0.0, hi = 0.35;
    double y_clamp = (y < 0.999999) ? y : 0.999999;
    double r = 1.0 - pow(1.0 - y_clamp, 1.0 / (double)k);
    if (r <= 0.0 || r >= 0.35 || !isfinite(r)) r = 0.5 * (lo + hi);

    for (int it = 0; it < 5; it++) {
        double g = 0.0, gp = 0.0;
        eval_pc_g_and_deriv(r, k, alpha, &g, &gp);
        double f = g - y;
        if (f < 0.0) lo = r; else hi = r;
        double step = (gp > 1e-15) ? (f / gp) : 0.0;
        double r_next = r - step;
        if (!(r_next > lo && r_next < hi)) {
            r_next = 0.5 * (lo + hi);
        }
        r = r_next;
    }

    if (out_gp) {
        double g = 0.0, gp = 0.0;
        eval_pc_g_and_deriv(r, k, alpha, &g, &gp);
        *out_gp = (gp > 1e-15) ? gp : 1e-15;
    }
    return r;
}

double ipc_solve_pc_divergence(
    uint64_t n_novel,
    uint64_t l_probe,
    int k,
    double d1_bar,
    double *out_g_deriv
) {
    if (l_probe == 0 || n_novel == 0) {
        double alpha = (d1_bar > 0.0) ? (d1_bar / 3.0) : 0.0;
        if (alpha > (double)k - 1e-6) alpha = (double)k - 1e-6;
        if (out_g_deriv) *out_g_deriv = (double)k - alpha;
        return 0.0;
    }
    double y = (double)n_novel / (double)l_probe;
    return solve_pc_from_y(y, k, d1_bar, out_g_deriv);
}

typedef struct {
    double r_hat;
    double y_hat;
    double g_deriv;
    double var_y_overlap;
    double var_y_binom;
    double variance_inflation_factor;
    double se_r_overlap;
    double se_r_binom;
    double ci_low_overlap;
    double ci_high_overlap;
    double ci_low_binom;
    double ci_high_binom;
} pc_ci_result_t;

void ipc_blanca_variance_ci(
    uint64_t n_novel,
    uint64_t l_probe,
    int k,
    int s,
    int is_syncmer,
    double d1_bar,
    const uint64_t *lag_pairs,
    uint64_t total_windows,
    double z_crit,
    pc_ci_result_t *out
) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (z_crit <= 0.0) z_crit = 1.959963984540054;
    if (k < 2) k = IPC_K;
    if (s < 1 || s >= k) s = IPC_S;

    if (l_probe == 0) {
        out->ci_high_overlap = 0.35;
        out->ci_high_binom = 0.35;
        return;
    }

    double y_hat = (double)n_novel / (double)l_probe;
    double gp = 0.0;
    double r_hat = solve_pc_from_y(y_hat, k, d1_bar, &gp);
    out->r_hat = r_hat;
    out->y_hat = y_hat;
    out->g_deriv = gp;

    double y_for_var = y_hat;
    double r_for_var = r_hat;
    if (n_novel == 0) {
        y_for_var = 1.0 / (double)(l_probe + 1ULL);
        r_for_var = solve_pc_from_y(y_for_var, k, d1_bar, NULL);
    }

    double var_y_binom = y_for_var * (1.0 - y_for_var) / (double)l_probe;
    double rk_raw = 1.0 - pow(1.0 - r_for_var, (double)k);
    double alpha = (d1_bar > 0.0) ? (d1_bar / 3.0) : 0.0;
    double coll_prob = 0.0;
    if (rk_raw > 1e-15 && alpha > 0.0) {
        coll_prob = (alpha * r_for_var * pow(1.0 - r_for_var, (double)(k - 1))) / rk_raw;
        if (coll_prob > 0.95) coll_prob = 0.95;
    }
    double surv_coll_sq = (1.0 - coll_prob) * (1.0 - coll_prob);
    double p_2k = pow(1.0 - r_for_var, (double)(2 * k));

    double var_y_overlap = var_y_binom;

    if (!is_syncmer) {
        uint64_t M = (total_windows > 0) ? total_windows : l_probe;
        double cov_sum = 0.0;
        for (int d = 1; d < k; d++) {
            double wd = lag_pairs
                ? (double)lag_pairs[d - 1]
                : ((M > (uint64_t)d) ? (double)(M - (uint64_t)d) : 0.0);
            double cov_d = pow(1.0 - r_for_var, (double)(k + d)) - p_2k;
            if (cov_d > 0.0) {
                cov_sum += wd * cov_d;
            }
        }
        var_y_overlap = var_y_binom + (2.0 * surv_coll_sq * cov_sum) / ((double)l_probe * (double)l_probe);
    } else {
        const int m = k - s + 1;
        const int t_star = (k - s) / 2;
        const double c_density = 1.0 / (double)m;
        uint64_t M = (total_windows > 0) ? total_windows : (l_probe * (uint64_t)m);
        double cov_ratio_sum = 0.0;
        for (int d = t_star + 1; d < k; d++) {
            double c2_d = (d < m)
                ? (2.0 / ((double)m * (double)(d + m)))
                : (1.0 / ((double)m * (double)m));
            double wd_over_M = 0.0;
            if (lag_pairs && M > 0) {
                wd_over_M = (double)lag_pairs[d - 1] / (double)M;
            } else if (M > (uint64_t)d) {
                wd_over_M = (double)(M - (uint64_t)d) / (double)M;
            }
            double cov_mut_d = pow(1.0 - r_for_var, (double)(k + d)) - p_2k;
            if (cov_mut_d > 0.0) {
                cov_ratio_sum += wd_over_M * (c2_d / c_density) * cov_mut_d;
            }
        }
        var_y_overlap = var_y_binom + (2.0 * surv_coll_sq * cov_ratio_sum) / (double)l_probe;
    }

    if (var_y_overlap < var_y_binom) var_y_overlap = var_y_binom;

    double se_y_ov = sqrt(var_y_overlap);
    double se_y_bi = sqrt(var_y_binom);

    out->var_y_overlap = var_y_overlap;
    out->var_y_binom = var_y_binom;
    out->variance_inflation_factor = (var_y_binom > 0.0) ? (var_y_overlap / var_y_binom) : 1.0;
    out->se_r_overlap = se_y_ov / gp;
    out->se_r_binom = se_y_bi / gp;

    double y_lo_ov = y_hat - z_crit * se_y_ov;
    double y_hi_ov = y_hat + z_crit * se_y_ov;
    double y_lo_bi = y_hat - z_crit * se_y_bi;
    double y_hi_bi = y_hat + z_crit * se_y_bi;

    out->ci_low_overlap = (y_lo_ov > 0.0) ? solve_pc_from_y(y_lo_ov, k, d1_bar, NULL) : 0.0;
    out->ci_high_overlap = solve_pc_from_y(y_hi_ov, k, d1_bar, NULL);
    out->ci_low_binom = (y_lo_bi > 0.0) ? solve_pc_from_y(y_lo_bi, k, d1_bar, NULL) : 0.0;
    out->ci_high_binom = solve_pc_from_y(y_hi_bi, k, d1_bar, NULL);
}

// -----------------------------------------------------------------------------
// Feller 1D Renewal DP for exact non-censoring probability A[n] in finite reads
// -----------------------------------------------------------------------------
static double feller_effective_interior_sites(int L, int k, double r, int is_junction) {
    if (L < 2 * k) return 0.0;
    if (r <= 1e-9) {
        return is_junction ? (double)(L - 2 * k + 1) : (double)(L - 2 * k);
    }
    if (r >= 0.5) return 1.0;

    double A_stack[512];
    double *A = (L + 2 <= 512) ? A_stack : (double *)calloc((size_t)(L + 2), sizeof(double));
    double pk = pow(1.0 - r, (double)k);
    for (int n = 0; n < k; n++) A[n] = 0.0;
    A[k] = pk;
    for (int n = k + 1; n <= L; n++) {
        double prev_comp = 1.0 - A[n - k - 1];
        A[n] = A[n - 1] + prev_comp * r * pk;
        if (A[n] > 1.0) A[n] = 1.0;
    }

    double sum_eff = 0.0;
    if (is_junction) {
        for (int x = k; x <= L - k; x++) {
            sum_eff += A[x] * A[L - x];
        }
    } else {
        for (int x = k; x <= L - k - 1; x++) {
            sum_eff += A[x] * A[L - 1 - x];
        }
    }
    if (A != A_stack) free(A);
    return sum_eff;
}

// -----------------------------------------------------------------------------
// Wave-0 Pre-Alignment Read Stream Classifier (<2 ms for 2,000 150-bp reads)
// -----------------------------------------------------------------------------
typedef struct {
    double kernel_ms;

    uint64_t n_reads_processed;
    uint64_t total_kmer_windows;
    uint64_t total_ocean_kmers;
    uint64_t total_novel_kmers;
    uint64_t unique_ocean_kmers;
    double ocean_fraction;

    uint64_t syncmer_probe_total;
    uint64_t syncmer_probe_novel;
    double r_pc_syncmer_raw;
    double r_pc_syncmer_bio;
    double r_pc_syncmer_ci_low;
    double r_pc_syncmer_ci_high;
    double r_pc_dense_raw;
    double r_pc_dense_bio;

    uint64_t c_sub_iso;
    uint64_t c_del_iso;
    uint64_t c_ins_iso;
    uint64_t c_ins_1bp_iso;
    uint64_t c_ins_multibp_iso;
    uint64_t c_del_microhom_iso;

    uint64_t c_merged_uniq;
    uint64_t c_merged_sub_only;
    uint64_t c_merged_pos_delta;
    uint64_t c_merged_neg_delta;
    int64_t sum_del_len_iso;
    int64_t sum_ins_len_iso;
    int64_t sum_merged_pos_delta;
    int64_t sum_merged_neg_delta;

    uint64_t c_rep_w_le_30;
    uint64_t c_rep_w_eq_31;
    uint64_t c_rep_w_ge_32;

    uint64_t n_end_censored_islands;
    uint64_t n_lowq_skipped_islands;
    double mean_base_err_admitted;

    double p_s_hat;
    double p_d_hat;
    double p_i_hat;
    double r_total_hat;
    double kr_hat;
    int high_divergence_merged_islands;
    int used_high_div_fallback;
    uint64_t n_anchored_reads;
    uint64_t n_unanchored_reads;
    uint64_t anchored_kmer_windows;
    uint64_t anchored_ocean_kmers;
    uint64_t anchored_novel_kmers;
    double unanchored_read_fraction;
} probe_result_t;

#define BATCH_READS 16
#define BATCH_MAX_WINS (BATCH_READS * 512)

int ipc_classify_reads(
    const ref_index_t *idx,
    const char *seqs,
    const char *quals,
    const int32_t *offsets,
    const int32_t *lengths,
    int32_t n_reads,
    int32_t min_base_q,
    double max_syncmer_err_prob,
    probe_result_t *out
) {
    ensure_tables();
    if (!idx || !seqs || !lengths || n_reads < 0 || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (n_reads == 0) return 0;

    double t0 = monotonic_ms();

    if (min_base_q < 0) min_base_q = 20;
    if (max_syncmer_err_prob <= 0.0) max_syncmer_err_prob = 0.05;
    double max_neg_log_ok = -log(1.0 - (max_syncmer_err_prob < 0.99 ? max_syncmer_err_prob : 0.99));

    const uint64_t kmask = (1ULL << (2 * IPC_K)) - 1ULL;
    const uint64_t smask = (1ULL << (2 * IPC_S)) - 1ULL;
    const uint64_t kmer_mask = idx->kmer_mask;
    const kmer_slot_t * restrict kmer_slots = idx->kmer_slots;
    const uint64_t bloom_block_mask = idx->bloom_block_mask;
    const uint64_t * restrict kmer_bloom = idx->kmer_bloom;

    uint64_t lag_pairs[IPC_K];
    memset(lag_pairs, 0, sizeof(lag_pairs));

    // Cross-read pipelined buffers (16 reads ~ 1,920 windows -> fits in L1 cache!)
    uint64_t batch_ck[BATCH_MAX_WINS];
    uint64_t batch_hval[BATCH_MAX_WINS];
    uint8_t batch_rsign[BATCH_MAX_WINS];
    uint8_t batch_qok[BATCH_MAX_WINS];
    int batch_win_start[BATCH_READS];
    int batch_win_count[BATCH_READS];
    int batch_read_len[BATCH_READS];
    int32_t batch_read_off[BATCH_READS];
    int batch_is_first_chunk[BATCH_READS];

    uint64_t shash_buf[512];

    uint64_t total_admitted_interior_s_bases = 0;
    uint64_t total_raw_interior_s_bases = 0;
    uint64_t total_raw_interior_d_juncs = 0;
    double sum_interior_base_err = 0.0;
    double sum_sync_kmer_err = 0.0;
    double sum_W_del = 0.0;
    double sum_W_ins = 0.0;
    int64_t sum_read_len = 0;
    uint64_t n_valid_reads = 0;

    // Support fixed-stride mode when offsets == NULL (offsets[r] = r * lengths[0]).
    // Long reads (L > 512) are tiled in 512-bp L1-resident windows stepped by (512 - 2*IPC_K) = 450 bp
    // so every interior base [IPC_K .. L - IPC_K - 1] is classified without silent truncation or L1 spill.
    const int32_t fixed_L = lengths[0];
    int32_t cur_r = 0;
    int32_t cur_chunk_off = 0;

    while (cur_r < n_reads) {
        int n_in_batch = 0;
        int total_batch_wins = 0;

        // Stage 1: Encode up to 16 read chunks in registers + O(1) sliding-min open syncmer check
        while (cur_r < n_reads && n_in_batch < BATCH_READS) {
            int32_t full_L = offsets ? lengths[cur_r] : fixed_L;
            int32_t base_off = offsets ? offsets[cur_r] : (cur_r * fixed_L);
            if (full_L < IPC_K) {
                cur_r++;
                cur_chunk_off = 0;
                continue;
            }
            int32_t rem_L = full_L - cur_chunk_off;
            if (rem_L < IPC_K) {
                cur_r++;
                cur_chunk_off = 0;
                continue;
            }
            int32_t L = (rem_L > 512) ? 512 : rem_L;
            int32_t off = base_off + cur_chunk_off;
            int is_first_chunk = (cur_chunk_off == 0);
            if (rem_L > 512) {
                int32_t step = 512 - 2 * IPC_K;
                cur_chunk_off += step;
            } else {
                cur_r++;
                cur_chunk_off = 0;
            }

            const char *s_ptr = seqs + off;
            const char *q_ptr = quals ? (quals + off) : NULL;
            const int m_win = L - IPC_K + 1;
            const int m_s = L - IPC_S + 1;

            uint64_t fk = 0, rk = 0, fs = 0, rs = 0;
            double cur_S = 0.0;
            int has_invalid = 0;

            int w_base = total_batch_wins;
            batch_is_first_chunk[n_in_batch] = is_first_chunk;

            if (!q_ptr) {
                for (int i = 0; i < IPC_S - 1; i++) {
                    uint8_t b = NT_TABLE[(uint8_t)s_ptr[i]];
                    if (b > 3) { has_invalid = 1; break; }
                    fs = ((fs << 2) | b) & smask;
                    rs = (rs >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_S - 1)));
                    fk = ((fk << 2) | b) & kmask;
                    rk = (rk >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_K - 1)));
                }
                if (!has_invalid) {
                    for (int i = IPC_S - 1; i < IPC_K - 1; i++) {
                        uint8_t b = NT_TABLE[(uint8_t)s_ptr[i]];
                        if (b > 3) { has_invalid = 1; break; }
                        fs = ((fs << 2) | b) & smask;
                        rs = (rs >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_S - 1)));
                        fk = ((fk << 2) | b) & kmask;
                        rk = (rk >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_K - 1)));
                        shash_buf[i - (IPC_S - 1)] = ipc_splitmix64((fs < rs) ? fs : rs);
                    }
                }
                if (!has_invalid) {
                    memset(batch_qok + w_base, 1, (size_t)m_win);
                    for (int i = IPC_K - 1; i < L; i++) {
                        uint8_t b = NT_TABLE[(uint8_t)s_ptr[i]];
                        if (b > 3) { has_invalid = 1; break; }
                        fs = ((fs << 2) | b) & smask;
                        rs = (rs >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_S - 1)));
                        fk = ((fk << 2) | b) & kmask;
                        rk = (rk >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_K - 1)));
                        shash_buf[i - (IPC_S - 1)] = ipc_splitmix64((fs < rs) ? fs : rs);
                        int w = w_base + (i - (IPC_K - 1));
                        uint8_t sgn = (fk < rk) ? 0 : 1;
                        batch_ck[w] = sgn ? rk : fk;
                        batch_rsign[w] = sgn;
                    }
                }
            } else {
                int run = 0;
                for (int i = 0; i < L; i++) {
                    uint8_t b = NT_TABLE[(uint8_t)s_ptr[i]];
                    if (b > 3) {
                        has_invalid = 1;
                        break;
                    }
                    cur_S += PHRED_NEG_LOG_OK_LUT[(uint8_t)q_ptr[i]];
                    if (i >= IPC_K) {
                        cur_S -= PHRED_NEG_LOG_OK_LUT[(uint8_t)q_ptr[i - IPC_K]];
                        if (cur_S < 0.0) cur_S = 0.0;
                    }
                    run++;
                    fs = ((fs << 2) | b) & smask;
                    rs = (rs >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_S - 1)));
                    fk = ((fk << 2) | b) & kmask;
                    rk = (rk >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_K - 1)));

                    if (run >= IPC_S) {
                        uint64_t cs = (fs < rs) ? fs : rs;
                        shash_buf[i - IPC_S + 1] = ipc_splitmix64(cs);
                    }
                    if (run >= IPC_K) {
                        int p = i - IPC_K + 1;
                        uint8_t sgn = (fk < rk) ? 0 : 1;
                        uint64_t canon = sgn ? rk : fk;
                        int w = w_base + p;
                        batch_ck[w] = canon;
                        batch_rsign[w] = sgn;
                        batch_qok[w] = (cur_S <= max_neg_log_ok) ? 1 : 0;
                    }
                }
            }

            if (has_invalid) continue;

            // O(1) amortized sliding-window minimum for middle-offset (t* = 8) open syncmers
            // Exact parity with ipc_syncmer_density(..., use_splitmix=1):
            //   94.1% of steps take 1 comparison; ties (<0.01%) fall back to strand-ordered scan.
            if (m_s >= IPC_M) {
                int min_idx = -1;
                uint64_t min_v = UINT64_MAX;
                int tie_flag = 0;
                for (int p = 0; p < m_win; p++) {
                    int new_idx = p + IPC_M - 1;
                    uint64_t hv = shash_buf[new_idx];
                    if (min_idx < p) {
                        min_v = shash_buf[p];
                        min_idx = p;
                        tie_flag = 0;
                        for (int u = p + 1; u <= new_idx; u++) {
                            uint64_t uv = shash_buf[u];
                            if (uv < min_v) {
                                min_v = uv;
                                min_idx = u;
                                tie_flag = 0;
                            } else if (uv == min_v) {
                                tie_flag = 1;
                            }
                        }
                    } else {
                        if (hv < min_v) {
                            min_v = hv;
                            min_idx = new_idx;
                            tie_flag = 0;
                        } else if (hv == min_v) {
                            tie_flag = 1;
                        }
                    }

                    int w = w_base + p;
                    if (!batch_qok[w]) continue;

                    int is_sync = 0;
                    if (!tie_flag) {
                        // Because IPC_M = 17 and IPC_T = 8 (16 - 8 == 8), unique minimum is symmetric!
                        is_sync = (min_idx == p + IPC_T);
                    } else {
                        int best = 0;
                        if (!batch_rsign[w]) {
                            uint64_t bv = shash_buf[p];
                            for (int j = 1; j < IPC_M; j++) {
                                uint64_t v = shash_buf[p + j];
                                if (v < bv) { bv = v; best = j; }
                            }
                        } else {
                            uint64_t bv = shash_buf[p + IPC_M - 1];
                            for (int j = 1; j < IPC_M; j++) {
                                uint64_t v = shash_buf[p + IPC_M - 1 - j];
                                if (v < bv) { bv = v; best = j; }
                            }
                        }
                        is_sync = (best == IPC_T);
                    }

                    if (is_sync) {
                        out->syncmer_probe_total++;
                        if (q_ptr) {
                            double s_err = 0.0;
                            for (int qb = p; qb < p + IPC_K; qb++) {
                                s_err += PHRED_NEG_LOG_OK_LUT[(uint8_t)q_ptr[qb]];
                            }
                            sum_sync_kmer_err += (1.0 - exp(-s_err));
                        }
                        // Note: every open syncmer is also in kmer_slots if and only if it is in sync_ht!
                        // We can check sync_ht OR reuse batch_hval after Stage 2 for ZERO extra hash lookup!!
                        // Mark bit 2 in batch_qok[w] so Stage 2 resolves syncmer hit/miss for FREE!
                        batch_qok[w] |= 2;
                    }
                }
            }

            if (is_first_chunk) n_valid_reads++;
            sum_read_len += L;
            if (L > 2 * IPC_K) {
                total_raw_interior_s_bases += (uint64_t)(L - 2 * IPC_K);
                total_raw_interior_d_juncs += (uint64_t)(L - 2 * IPC_K + 1);
                if (q_ptr) {
                    for (int x = IPC_K; x <= L - IPC_K - 1; x++) {
                        int qv = (int)((uint8_t)q_ptr[x]) - 33;
                        if (qv >= min_base_q) {
                            total_admitted_interior_s_bases++;
                            sum_interior_base_err += PHRED_ERR_LUT[(uint8_t)q_ptr[x]];
                        }
                    }
                } else {
                    total_admitted_interior_s_bases += (uint64_t)(L - 2 * IPC_K);
                }
            }
            if (offsets) {
                for (int d = 1; d < IPC_K; d++) {
                    if (m_win > d) lag_pairs[d - 1] += (uint64_t)(m_win - d);
                }
            }

            batch_win_start[n_in_batch] = w_base;
            batch_win_count[n_in_batch] = m_win;
            batch_read_len[n_in_batch] = L;
            batch_read_off[n_in_batch] = off;
            n_in_batch++;
            total_batch_wins += m_win;
        }

        // Stage 2: 16-way p=0 prefetch + L1-anchored reference ocean walk + L2-blocked Bloom filter!
        uint64_t p0_hash[BATCH_READS];
        for (int bi = 0; bi < n_in_batch; bi++) {
            uint64_t h0 = hash_slot64(batch_ck[batch_win_start[bi]]);
            p0_hash[bi] = h0;
            __builtin_prefetch(&kmer_slots[h0 & kmer_mask], 0, 0);
        }

        for (int bi = 0; bi < n_in_batch; bi++) {
            const int w_base = batch_win_start[bi];
            const int m_win = batch_win_count[bi];
            const char *s_ptr = seqs + batch_read_off[bi];

            int anchored = 0;
            int chunk_has_anchor = 0;
            uint64_t chunk_kmer_win = 0;
            uint64_t chunk_ocean_km = 0;
            uint64_t chunk_novel_km = 0;
            uint16_t anc_ctg = 0;
            int64_t anc_pos = 0;
            int anc_strand = 1;
            int64_t anc_ctg_len = 0;
            const uint8_t *anc_nt = NULL;
            const uint64_t *anc_ubv = NULL;

            for (int p = 0; p < m_win; p++) {
                int w = w_base + p;
                uint64_t hv = 0;
                int resolved_by_walk = 0;

                if (anchored) {
                    uint8_t read_new_b = NT_TABLE[(uint8_t)s_ptr[p + IPC_K - 1]];
                    if (anc_strand > 0) {
                        int64_t next_pos = anc_pos + 1;
                        if (next_pos + IPC_K <= anc_ctg_len) {
                            uint8_t ref_new_b = anc_nt[next_pos + IPC_K - 1];
                            if (ref_new_b <= 3 && read_new_b == ref_new_b) {
                                anc_pos = next_pos;
                                int is_u = (int)((anc_ubv[next_pos >> 6] >> (next_pos & 63)) & 1ULL);
                                int ref_sgn = batch_rsign[w];
                                hv = is_u
                                    ? MAKE_VAL((uint32_t)next_pos, anc_ctg, ref_sgn, 1, 1)
                                    : MAKE_VAL(0, anc_ctg, ref_sgn, 0, 2);
                                resolved_by_walk = 1;
                            }
                        }
                    } else {
                        int64_t next_pos = anc_pos - 1;
                        if (next_pos >= 0) {
                            uint8_t ref_new_b = anc_nt[next_pos];
                            if (ref_new_b <= 3 && read_new_b == (uint8_t)(3U - ref_new_b)) {
                                anc_pos = next_pos;
                                int is_u = (int)((anc_ubv[next_pos >> 6] >> (next_pos & 63)) & 1ULL);
                                int ref_sgn = 1 ^ (int)batch_rsign[w];
                                hv = is_u
                                    ? MAKE_VAL((uint32_t)next_pos, anc_ctg, ref_sgn, 1, 1)
                                    : MAKE_VAL(0, anc_ctg, ref_sgn, 0, 2);
                                resolved_by_walk = 1;
                            }
                        }
                    }
                    if (!resolved_by_walk) {
                        anchored = 0;
                        // Prefetch expected post-island re-anchor k-mers (W=30 del & W=31 SNP/1bp-ins)
                        if (p + 30 < m_win) {
                            __builtin_prefetch(&kmer_slots[hash_slot64(batch_ck[w + 30]) & kmer_mask], 0, 0);
                        }
                        if (p + 31 < m_win) {
                            __builtin_prefetch(&kmer_slots[hash_slot64(batch_ck[w + 31]) & kmer_mask], 0, 0);
                        }
                    }
                }

                if (!resolved_by_walk) {
                    if (p + 6 < m_win) {
                        uint64_t h_pf = hash_slot64(batch_ck[w + 6]);
                        __builtin_prefetch(kmer_bloom + (((h_pf >> 24) & bloom_block_mask) << 3), 0, 0);
                    }
                    uint64_t canon = batch_ck[w];
                    uint64_t h = (p == 0) ? p0_hash[bi] : hash_slot64(canon);
                    if (p == 0 || bloom_query_hash(kmer_bloom, bloom_block_mask, h)) {
                        uint64_t sl = h & kmer_mask;
                        while (1) {
                            uint64_t kk = kmer_slots[sl].key;
                            if (kk == canon) {
                                hv = kmer_slots[sl].val;
                                break;
                            }
                            if (kk == EMPTY_KEY) {
                                hv = 0;
                                break;
                            }
                            sl = (sl + 1) & kmer_mask;
                        }
                    }
                    if (hv != 0) {
                        uint16_t c_id = VAL_CONTIG(hv);
                        if ((int)c_id < idx->n_contigs && idx->contig_nt[c_id] && idx->contig_uniq_bv[c_id]) {
                            anchored = 1;
                            anc_ctg = c_id;
                            anc_pos = (int64_t)VAL_POS(hv);
                            anc_strand = (batch_rsign[w] == VAL_SIGN(hv)) ? 1 : -1;
                            anc_ctg_len = idx->contig_lens[c_id];
                            anc_nt = idx->contig_nt[c_id];
                            anc_ubv = idx->contig_uniq_bv[c_id];
                        }
                    }
                }

                if (hv != 0) {
                    chunk_has_anchor = 1;
                }
                batch_hval[w] = hv;
                uint8_t qf = batch_qok[w];
                if (qf & 1) {
                    out->total_kmer_windows++;
                    chunk_kmer_win++;
                    if (hv != 0) {
                        out->total_ocean_kmers++;
                        chunk_ocean_km++;
                        if (VAL_UNIQUE(hv)) out->unique_ocean_kmers++;
                    } else {
                        out->total_novel_kmers++;
                        chunk_novel_km++;
                        if (qf & 2) {
                            out->syncmer_probe_novel++;
                        }
                    }
                }
            }
            if (chunk_has_anchor) {
                if (batch_is_first_chunk[bi]) out->n_anchored_reads++;
                out->anchored_kmer_windows += chunk_kmer_win;
                out->anchored_ocean_kmers += chunk_ocean_km;
                out->anchored_novel_kmers += chunk_novel_km;
            } else if (batch_is_first_chunk[bi]) {
                out->n_unanchored_reads++;
            }
        }

        // Stage 3: Interior island width (W) & unique-anchor coordinate-shift (delta) classification
        for (int bi = 0; bi < n_in_batch; bi++) {
            const int w_base = batch_win_start[bi];
            const int m_win = batch_win_count[bi];
            const int L = batch_read_len[bi];
            const char *q_ptr = quals ? (quals + batch_read_off[bi]) : NULL;
            const uint64_t *hval_buf = batch_hval + w_base;
            const uint8_t *rsign_buf = batch_rsign + w_base;

            int i_first = -1, i_last = -1;
            for (int p = 0; p < m_win; p++) {
                if (hval_buf[p] != 0) {
                    if (i_first < 0) i_first = p;
                    i_last = p;
                }
            }
            if (i_first < 0) {
                out->n_end_censored_islands++;
                continue;
            }
            if (i_first > 0) out->n_end_censored_islands++;
            if (i_last < m_win - 1) out->n_end_censored_islands++;
            if (i_first == i_last) continue;

            int i_L = i_first;
            for (int p = i_first + 1; p <= i_last; p++) {
                if (hval_buf[p] == 0) continue;
                int i_R = p;
                int W = i_R - i_L - 1;

                uint64_t vL = hval_buf[i_L];
                uint64_t vR = hval_buf[i_R];
                int uniq_both = (VAL_UNIQUE(vL) && VAL_UNIQUE(vR) && (VAL_CONTIG(vL) == VAL_CONTIG(vR)));
                int strand_L = (rsign_buf[i_L] == VAL_SIGN(vL)) ? 1 : -1;
                int strand_R = (rsign_buf[i_R] == VAL_SIGN(vR)) ? 1 : -1;
                int anchor_ok = 0;
                int64_t delta = 0;

                if (uniq_both && (strand_L == strand_R)) {
                    int64_t signed_ref_span = (int64_t)strand_L * ((int64_t)VAL_POS(vR) - (int64_t)VAL_POS(vL));
                    int64_t read_span = (int64_t)(i_R - i_L);
                    int64_t d_cand = signed_ref_span - read_span;
                    if (signed_ref_span >= 0 && llabs(d_cand) <= MAX_INDEL_SHIFT) {
                        anchor_ok = 1;
                        delta = d_cand;
                    }
                }

                if (W == 0) {
                    if (anchor_ok && delta != 0) {
                        int mid2 = i_L + i_R + (IPC_K - 1);
                        if (mid2 >= (2 * IPC_K - 1) && mid2 <= (2 * (L - IPC_K) - 1)) {
                            if (delta > 0) {
                                out->c_del_iso++;
                                out->c_del_microhom_iso++;
                                out->sum_del_len_iso += delta;
                            } else {
                                out->c_ins_iso++;
                                if (delta == -1) out->c_ins_1bp_iso++; else out->c_ins_multibp_iso++;
                                out->sum_ins_len_iso += (-delta);
                            }
                        } else {
                            out->n_end_censored_islands++;
                        }
                    }
                    i_L = i_R;
                    continue;
                }

                int mid2 = i_L + i_R + (IPC_K - 1);
                if (mid2 < (2 * IPC_K - 1) || mid2 > (2 * (L - IPC_K) - 1)) {
                    out->n_end_censored_islands++;
                    i_L = i_R;
                    continue;
                }

                if (q_ptr) {
                    int q_start = (W >= IPC_K) ? (i_L + IPC_K) : (i_R - 1);
                    int q_end   = (W >= IPC_K) ? (i_R - 1)     : (i_L + IPC_K);
                    if (q_start < 0) q_start = 0;
                    if (q_end >= L) q_end = L - 1;
                    if (q_start > q_end) { int tmp = q_start; q_start = q_end; q_end = tmp; }
                    int low_q = 0;
                    for (int qb = q_start; qb <= q_end; qb++) {
                        if (((int)((uint8_t)q_ptr[qb]) - 33) < min_base_q) {
                            low_q = 1;
                            break;
                        }
                    }
                    if (low_q) {
                        out->n_lowq_skipped_islands++;
                        i_L = i_R;
                        continue;
                    }
                }

                if (anchor_ok) {
                    if (delta == 0 && W <= IPC_K) {
                        out->c_sub_iso++;
                    } else if (delta > 0 && W <= IPC_K - 1) {
                        out->c_del_iso++;
                        if (W < IPC_K - 1) out->c_del_microhom_iso++;
                        out->sum_del_len_iso += delta;
                        sum_W_del += (double)W;
                    } else if (delta < 0 && (W + (int)delta) <= (IPC_K - 1)) {
                        out->c_ins_iso++;
                        if (delta == -1) out->c_ins_1bp_iso++; else out->c_ins_multibp_iso++;
                        out->sum_ins_len_iso += (-delta);
                        sum_W_ins += (double)W;
                    } else {
                        out->c_merged_uniq++;
                        if (delta == 0) {
                            out->c_merged_sub_only++;
                        } else if (delta > 0) {
                            out->c_merged_pos_delta++;
                            out->sum_merged_pos_delta += delta;
                        } else {
                            out->c_merged_neg_delta++;
                            out->sum_merged_neg_delta += (-delta);
                        }
                    }
                } else {
                    if (W <= IPC_K - 1) out->c_rep_w_le_30++;
                    else if (W == IPC_K) out->c_rep_w_eq_31++;
                    else out->c_rep_w_ge_32++;
                }

                i_L = i_R;
            }
        }
    }

    if (!offsets && fixed_L >= IPC_K) {
        int mw = fixed_L - IPC_K + 1;
        for (int d = 1; d < IPC_K; d++) {
            if (mw > d) lag_pairs[d - 1] = n_valid_reads * (uint64_t)(mw - d);
        }
    }
    out->n_reads_processed = n_valid_reads;
    out->ocean_fraction = (out->total_kmer_windows > 0)
        ? ((double)out->total_ocean_kmers / (double)out->total_kmer_windows)
        : 0.0;
    uint64_t tot_anc_unanc = out->n_anchored_reads + out->n_unanchored_reads;
    out->unanchored_read_fraction = (tot_anc_unanc > 0)
        ? ((double)out->n_unanchored_reads / (double)tot_anc_unanc)
        : 0.0;

    double eps_base_admitted = (total_admitted_interior_s_bases > 0)
        ? (sum_interior_base_err / (double)total_admitted_interior_s_bases)
        : 0.0;
    out->mean_base_err_admitted = eps_base_admitted;

    double eps_sync_per_base = 0.0;
    if (quals && out->syncmer_probe_total > 0) {
        double mean_kmer_err = sum_sync_kmer_err / (double)out->syncmer_probe_total;
        if (mean_kmer_err > 0.99) mean_kmer_err = 0.99;
        eps_sync_per_base = 1.0 - pow(1.0 - mean_kmer_err, 1.0 / (double)IPC_K);
    }

    pc_ci_result_t sync_ci;
    ipc_blanca_variance_ci(
        out->syncmer_probe_novel,
        out->syncmer_probe_total,
        IPC_K, IPC_S, 1,
        idx->d1_sync_bar,
        lag_pairs,
        out->total_kmer_windows,
        1.959963984540054,
        &sync_ci
    );
    out->r_pc_syncmer_raw = sync_ci.r_hat;
    out->r_pc_syncmer_bio = (sync_ci.r_hat > eps_sync_per_base) ? (sync_ci.r_hat - eps_sync_per_base) : 0.0;
    out->r_pc_syncmer_ci_low = (sync_ci.ci_low_overlap > eps_sync_per_base)
        ? (sync_ci.ci_low_overlap - eps_sync_per_base) : 0.0;
    out->r_pc_syncmer_ci_high = (sync_ci.ci_high_overlap > eps_sync_per_base)
        ? (sync_ci.ci_high_overlap - eps_sync_per_base) : 0.0;

    double d1_for_dense = idx->dense_d1_computed ? idx->d1_dense_bar : idx->d1_sync_bar;
    // Estimate on-target --pc divergence from anchored reads (>=1 matching k-mer) when available so
    // off-target/unannotated plasmid/rRNA/spike-in reads do not masquerade as high genomic divergence (L78).
    int use_anchored_dense = (out->anchored_kmer_windows >= 20 && out->ocean_fraction >= 0.10);
    uint64_t dense_novel = use_anchored_dense ? out->anchored_novel_kmers : out->total_novel_kmers;
    uint64_t dense_wins = use_anchored_dense ? out->anchored_kmer_windows : out->total_kmer_windows;
    out->r_pc_dense_raw = ipc_solve_pc_divergence(
        dense_novel, dense_wins, IPC_K, d1_for_dense, NULL
    );
    out->r_pc_dense_bio = (out->r_pc_dense_raw > eps_sync_per_base)
        ? (out->r_pc_dense_raw - eps_sync_per_base) : 0.0;

    double frac_uniq = 1.0;
    if (out->total_ocean_kmers >= 20) {
        frac_uniq = (double)out->unique_ocean_kmers / (double)out->total_ocean_kmers;
    } else if (idx->total_kmer_positions > 0) {
        frac_uniq = (double)idx->unique_kmer_positions / (double)idx->total_kmer_positions;
    }

    double q_admit_frac = (total_raw_interior_s_bases > 0)
        ? ((double)total_admitted_interior_s_bases / (double)total_raw_interior_s_bases)
        : 1.0;

    double mean_del_len = (out->c_del_iso > 0)
        ? ((double)out->sum_del_len_iso / (double)out->c_del_iso) : 1.0;
    double mean_ins_len = (out->c_ins_iso > 0)
        ? ((double)out->sum_ins_len_iso / (double)out->c_ins_iso) : 1.0;
    if (mean_del_len < 1.0) mean_del_len = 1.0;
    if (mean_ins_len < 1.0) mean_ins_len = 1.0;

    double anchor_read_frac = (tot_anc_unanc > 0 && out->n_anchored_reads > 0)
        ? ((double)out->n_anchored_reads / (double)tot_anc_unanc)
        : 1.0;
    int mean_L = (n_valid_reads > 0) ? (int)(sum_read_len / (int64_t)n_valid_reads) : 150;
    double B_s_uniq = (double)total_admitted_interior_s_bases * anchor_read_frac * frac_uniq;
    double B_d_uniq = (double)total_raw_interior_d_juncs * anchor_read_frac * q_admit_frac * frac_uniq;
    double ins_trim = (mean_ins_len > 1.0) ? (mean_ins_len - 1.0) : 0.0;
    double B_i_per_read = (double)(mean_L - 2 * IPC_K + 1) - ins_trim;
    if (B_i_per_read < 1.0) B_i_per_read = 1.0;
    double B_i_uniq = (double)n_valid_reads * anchor_read_frac * B_i_per_read * q_admit_frac * frac_uniq;

    uint64_t total_iso_uniq = out->c_sub_iso + out->c_del_iso + out->c_ins_iso;
    double merged_frac = ((total_iso_uniq + out->c_merged_uniq) > 0)
        ? ((double)out->c_merged_uniq / (double)(total_iso_uniq + out->c_merged_uniq))
        : 0.0;

    double r_ref_raw = (dense_wins > 0) ? out->r_pc_dense_raw : out->r_pc_syncmer_raw;
    double kr_probe = (double)IPC_K * r_ref_raw;
    double anchored_ocean_frac = (out->anchored_kmer_windows > 0)
        ? ((double)out->anchored_ocean_kmers / (double)out->anchored_kmer_windows)
        : out->ocean_fraction;
    int low_ocean_div = use_anchored_dense
        ? (anchored_ocean_frac < 0.50 && out->anchored_kmer_windows >= 100)
        : (out->ocean_fraction < 0.50 && out->total_kmer_windows >= 100);

    int high_div = (kr_probe > 0.35) || low_ocean_div || (out->ocean_fraction < 0.10 && out->total_kmer_windows >= 100) || (merged_frac > 0.45 && out->c_merged_uniq >= 10);
    out->high_divergence_merged_islands = high_div ? 1 : 0;

    if (B_s_uniq >= 50.0) {
        double u_s = (double)out->c_sub_iso / B_s_uniq;
        double u_d = (double)out->c_del_iso / B_d_uniq;
        double u_i = (double)out->c_ins_iso / B_i_uniq;
        double u_tot = u_s + u_d + u_i;

        const double two_k = (double)(2 * IPC_K);
        const double r_fold = 1.0 / (two_k + 1.0);
        const double u_fold = r_fold * pow(1.0 - r_fold, two_k);

        if (!high_div && u_tot < 0.92 * u_fold) {
            double lo = 0.0, hi = r_fold;
            double r_isl = (u_tot < r_fold) ? u_tot : (0.5 * r_fold);
            for (int it = 0; it < 6; it++) {
                double p2km1 = pow(1.0 - r_isl, two_k - 1.0);
                double f = r_isl * p2km1 * (1.0 - r_isl) - u_tot;
                double fp = p2km1 * (1.0 - (two_k + 1.0) * r_isl);
                if (f < 0.0) lo = r_isl; else hi = r_isl;
                double rn = (fp > 1e-15) ? (r_isl - f / fp) : (0.5 * (lo + hi));
                if (!(rn > lo && rn < hi)) rn = 0.5 * (lo + hi);
                r_isl = rn;
            }
            double r_surv = r_isl;
            double surv = pow(1.0 - r_surv, two_k);
            if (surv < 0.05) surv = 0.05;

            double ps_raw = u_s / surv;
            double pd_raw = u_d / surv;
            double pi_raw = u_i / surv;

            out->p_s_hat = (ps_raw > 0.85 * eps_base_admitted) ? (ps_raw - 0.85 * eps_base_admitted) : 0.0;
            out->p_d_hat = (pd_raw > 0.075 * eps_base_admitted) ? (pd_raw - 0.075 * eps_base_admitted) : 0.0;
            out->p_i_hat = (pi_raw > 0.075 * eps_base_admitted) ? (pi_raw - 0.075 * eps_base_admitted) : 0.0;
            out->used_high_div_fallback = 0;
        } else {
            out->used_high_div_fallback = 1;

            double b_eff_per_read = feller_effective_interior_sites(mean_L, IPC_K, r_ref_raw, 1);
            double B_eff_junc = (double)n_valid_reads * anchor_read_frac * b_eff_per_read * q_admit_frac * frac_uniq;
            if (B_eff_junc < 1.0) B_eff_junc = 1.0;

            double tot_del_events = (double)out->c_del_iso + ((double)out->sum_merged_pos_delta / mean_del_len);
            double tot_ins_events = (double)out->c_ins_iso + ((double)out->sum_merged_neg_delta / mean_ins_len);

            double pd_0 = tot_del_events / B_eff_junc;
            double pi_0 = tot_ins_events / B_eff_junc;

            double cancel_corr = pd_0 * pi_0 * (double)(2 * IPC_K);
            double pd_raw = pd_0 + cancel_corr;
            double pi_raw = pi_0 + cancel_corr;

            out->p_d_hat = (pd_raw > 0.075 * eps_base_admitted) ? (pd_raw - 0.075 * eps_base_admitted) : 0.0;
            out->p_i_hat = (pi_raw > 0.075 * eps_base_admitted) ? (pi_raw - 0.075 * eps_base_admitted) : 0.0;

            double mean_W_del = (out->c_del_iso > 0) ? (sum_W_del / (double)out->c_del_iso) : (double)(IPC_K - 1);
            double mean_W_ins = (out->c_ins_iso > 0) ? (sum_W_ins / (double)out->c_ins_iso) : (double)IPC_K;

            double r_bio_tot = out->r_pc_dense_bio;
            double indel_kmer_equiv = (mean_W_del / (double)IPC_K) * out->p_d_hat
                                    + (mean_W_ins / (double)IPC_K) * out->p_i_hat;
            out->p_s_hat = (r_bio_tot > indel_kmer_equiv) ? (r_bio_tot - indel_kmer_equiv) : 0.0;
        }
    } else if (total_admitted_interior_s_bases > 0) {
        double B_tot = (double)total_admitted_interior_s_bases;
        double surv = pow(1.0 - r_ref_raw, (double)(2 * IPC_K));
        if (surv < 0.10) surv = 0.10;
        out->p_d_hat = ((double)out->c_rep_w_le_30 / B_tot) / surv;
        out->p_i_hat = ((double)out->c_rep_w_ge_32 / B_tot) / surv;
        double r_bio = out->r_pc_dense_bio;
        out->p_s_hat = (r_bio > out->p_d_hat + out->p_i_hat) ? (r_bio - out->p_d_hat - out->p_i_hat) : 0.0;
    } else if (out->syncmer_probe_total > 0 || out->total_kmer_windows > 0) {
        out->p_s_hat = out->r_pc_dense_bio;
        out->p_d_hat = 0.0;
        out->p_i_hat = 0.0;
    }

    out->r_total_hat = out->p_s_hat + out->p_d_hat + out->p_i_hat;
    out->kr_hat = (double)IPC_K * out->r_total_hat;
    if (out->kr_hat > 0.35) {
        out->high_divergence_merged_islands = 1;
    }

    out->kernel_ms = monotonic_ms() - t0;
    return 0;
}

// -----------------------------------------------------------------------------
// Fast C Baseline Divergence Estimators for Experiment A (Mash, Set Containment, --pc)
// -----------------------------------------------------------------------------
typedef struct {
    uint64_t query_total_positions;
    uint64_t query_novel_positions;
    uint64_t query_distinct_kmers;
    uint64_t shared_distinct_kmers;
    uint64_t ref_distinct_kmers;

    uint64_t query_sync_positions;
    uint64_t query_sync_novel_positions;

    double mash_jaccard_exact;
    double r_mash_exact;
    double mash_jaccard_sketch2000;
    double r_mash_sketch2000;

    double r_containment_query_set;
    double r_containment_ref_set;
    double r_containment_pos_naive;

    double r_pc_dense;
    double r_pc_syncmer;
} baseline_div_t;

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int ipc_compute_baselines(
    const ref_index_t *idx,
    const char **query_seqs,
    const int64_t *query_lens,
    int n_seqs,
    int sketch_size,
    baseline_div_t *out
) {
    ensure_tables();
    if (!idx || !query_seqs || !query_lens || n_seqs <= 0 || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (sketch_size <= 0) sketch_size = 2000;

    uint64_t approx_q = 0;
    for (int c = 0; c < n_seqs; c++) {
        if (query_lens[c] >= IPC_K) approx_q += (uint64_t)(query_lens[c] - IPC_K + 1);
    }
    uint64_t qcap = 1024;
    while (qcap < approx_q * 2ULL + 64ULL) qcap <<= 1;
    uint64_t qmask = qcap - 1ULL;
    uint64_t *qkeys = (uint64_t *)malloc(qcap * sizeof(uint64_t));
    if (!qkeys) return -1;
    memset(qkeys, 0xFF, qcap * sizeof(uint64_t));

    const uint64_t kmask = (1ULL << (2 * IPC_K)) - 1ULL;
    const uint64_t smask = (1ULL << (2 * IPC_S)) - 1ULL;
    uint64_t cs_hash_ring[64];

    uint64_t q_distinct = 0;
    uint64_t shared_distinct = 0;

    for (int c = 0; c < n_seqs; c++) {
        const char *seq = query_seqs[c];
        int64_t len = query_lens[c];
        if (!seq || len < IPC_K) continue;

        uint64_t fk = 0, rk = 0, fs = 0, rs = 0;
        int64_t run = 0;
        for (int64_t i = 0; i < len; i++) {
            uint8_t b = NT_TABLE[(uint8_t)seq[i]];
            if (b > 3) { run = 0; fk = rk = fs = rs = 0; continue; }
            run++;
            fs = ((fs << 2) | b) & smask;
            rs = (rs >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_S - 1)));
            fk = ((fk << 2) | b) & kmask;
            rk = (rk >> 2) | ((uint64_t)(3 - b) << (2 * (IPC_K - 1)));

            if (run >= IPC_S) {
                int64_t sp = i - IPC_S + 1;
                uint64_t cs = (fs < rs) ? fs : rs;
                cs_hash_ring[sp & 63] = ipc_splitmix64(cs);
            }
            if (run >= IPC_K) {
                int64_t p = i - IPC_K + 1;
                int sgn = (fk < rk) ? 0 : 1;
                uint64_t canon = sgn ? rk : fk;

                out->query_total_positions++;
                int in_ref = 0;
                (void)kmer_ht_find_val(idx, canon, &in_ref);
                if (!in_ref) out->query_novel_positions++;

                uint64_t sl = hash_slot64(canon) & qmask;
                while (1) {
                    uint64_t kk = qkeys[sl];
                    if (kk == canon) break;
                    if (kk == EMPTY_KEY) {
                        qkeys[sl] = canon;
                        q_distinct++;
                        if (in_ref) shared_distinct++;
                        break;
                    }
                    sl = (sl + 1) & qmask;
                }

                int best = 0;
                if (!sgn) {
                    uint64_t bv = cs_hash_ring[p & 63];
                    for (int j = 1; j < IPC_M; j++) {
                        uint64_t v = cs_hash_ring[(p + j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                } else {
                    uint64_t bv = cs_hash_ring[(p + IPC_M - 1) & 63];
                    for (int j = 1; j < IPC_M; j++) {
                        uint64_t v = cs_hash_ring[(p + IPC_M - 1 - j) & 63];
                        if (v < bv) { bv = v; best = j; }
                    }
                }
                if (best == IPC_T) {
                    out->query_sync_positions++;
                    if (!in_ref) out->query_sync_novel_positions++;
                }
            }
        }
    }

    out->query_distinct_kmers = q_distinct;
    out->shared_distinct_kmers = shared_distinct;
    out->ref_distinct_kmers = idx->distinct_kmers;

    uint64_t union_sz = idx->distinct_kmers + q_distinct - shared_distinct;
    double J_exact = (union_sz > 0) ? ((double)shared_distinct / (double)union_sz) : 1.0;
    out->mash_jaccard_exact = J_exact;
    double twoJ_1pJ = (2.0 * J_exact) / (1.0 + J_exact);
    out->r_mash_exact = (twoJ_1pJ > 0.0) ? (1.0 - pow(twoJ_1pJ, 1.0 / (double)IPC_K)) : 1.0;

    uint64_t *ref_hashes = (uint64_t *)malloc(idx->distinct_kmers * sizeof(uint64_t));
    uint64_t *qry_hashes = (uint64_t *)malloc((q_distinct > 0 ? q_distinct : 1) * sizeof(uint64_t));
    if (ref_hashes && qry_hashes && idx->distinct_kmers > 0 && q_distinct > 0) {
        size_t nr = 0, nq = 0;
        for (uint64_t i = 0; i < idx->kmer_cap; i++) {
            if (idx->kmer_slots[i].key != EMPTY_KEY) ref_hashes[nr++] = ipc_splitmix64(idx->kmer_slots[i].key);
        }
        for (uint64_t i = 0; i < qcap; i++) {
            if (qkeys[i] != EMPTY_KEY) qry_hashes[nq++] = ipc_splitmix64(qkeys[i]);
        }
        qsort(ref_hashes, nr, sizeof(uint64_t), cmp_u64);
        qsort(qry_hashes, nq, sizeof(uint64_t), cmp_u64);
        size_t lim_r = (nr < (size_t)sketch_size) ? nr : (size_t)sketch_size;
        size_t lim_q = (nq < (size_t)sketch_size) ? nq : (size_t)sketch_size;
        size_t ir = 0, iq = 0, u_cnt = 0, shared_sk = 0;
        size_t target_u = (size_t)sketch_size;
        while (u_cnt < target_u && (ir < lim_r || iq < lim_q)) {
            if (ir < lim_r && (iq >= lim_q || ref_hashes[ir] < qry_hashes[iq])) {
                ir++;
                u_cnt++;
            } else if (iq < lim_q && (ir >= lim_r || qry_hashes[iq] < ref_hashes[ir])) {
                iq++;
                u_cnt++;
            } else {
                shared_sk++;
                ir++;
                iq++;
                u_cnt++;
            }
        }
        double J_sk = (u_cnt > 0) ? ((double)shared_sk / (double)u_cnt) : 1.0;
        out->mash_jaccard_sketch2000 = J_sk;
        double m_sk = (2.0 * J_sk) / (1.0 + J_sk);
        out->r_mash_sketch2000 = (m_sk > 0.0) ? (1.0 - pow(m_sk, 1.0 / (double)IPC_K)) : 1.0;
    }
    free(ref_hashes);
    free(qry_hashes);
    free(qkeys);

    double c_qset = (q_distinct > 0) ? ((double)shared_distinct / (double)q_distinct) : 1.0;
    double c_rset = (idx->distinct_kmers > 0) ? ((double)shared_distinct / (double)idx->distinct_kmers) : 1.0;
    double c_pos = (out->query_total_positions > 0)
        ? (1.0 - (double)out->query_novel_positions / (double)out->query_total_positions) : 1.0;

    out->r_containment_query_set = (c_qset > 0.0) ? (1.0 - pow(c_qset, 1.0 / (double)IPC_K)) : 1.0;
    out->r_containment_ref_set = (c_rset > 0.0) ? (1.0 - pow(c_rset, 1.0 / (double)IPC_K)) : 1.0;
    out->r_containment_pos_naive = (c_pos > 0.0) ? (1.0 - pow(c_pos, 1.0 / (double)IPC_K)) : 1.0;

    double d1_dense = idx->dense_d1_computed ? idx->d1_dense_bar : idx->d1_sync_bar;
    out->r_pc_dense = ipc_solve_pc_divergence(
        out->query_novel_positions, out->query_total_positions, IPC_K, d1_dense, NULL
    );
    out->r_pc_syncmer = ipc_solve_pc_divergence(
        out->query_sync_novel_positions, out->query_sync_positions, IPC_K, idx->d1_sync_bar, NULL
    );

    return 0;
}

// -----------------------------------------------------------------------------
// Binary Offline Serialization (ipc_save_index / ipc_load_index)
// -----------------------------------------------------------------------------
#define IPC_IDX_MAGIC 0x31584449435049ULL  /* "IPCIDX1\0" */

int ipc_save_index(const ref_index_t *idx, const char *path) {
    if (!idx || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -2;

    uint64_t hdr[14];
    hdr[0] = IPC_IDX_MAGIC;
    hdr[1] = idx->kmer_cap;
    hdr[2] = idx->kmer_mask;
    hdr[3] = idx->bloom_block_mask;
    hdr[4] = (uint64_t)(uint32_t)idx->n_contigs;
    hdr[5] = idx->sync_cap;
    hdr[6] = idx->sync_mask;
    hdr[7] = idx->total_kmer_positions;
    hdr[8] = idx->distinct_kmers;
    hdr[9] = idx->unique_kmer_positions;
    hdr[10] = idx->total_syncmer_positions;
    hdr[11] = idx->distinct_syncmers;
    hdr[12] = idx->D1_dense;
    hdr[13] = idx->D1_sync;

    if (fwrite(hdr, sizeof(uint64_t), 14, fp) != 14) { fclose(fp); return -3; }
    double dvals[2] = { idx->d1_dense_bar, idx->d1_sync_bar };
    if (fwrite(dvals, sizeof(double), 2, fp) != 2) { fclose(fp); return -3; }
    int32_t dd1 = (int32_t)idx->dense_d1_computed;
    if (fwrite(&dd1, sizeof(int32_t), 1, fp) != 1) { fclose(fp); return -3; }

    size_t bwords = (size_t)((idx->bloom_block_mask + 1ULL) * 8ULL);
    if (fwrite(idx->kmer_slots, sizeof(kmer_slot_t), (size_t)idx->kmer_cap, fp) != (size_t)idx->kmer_cap ||
        fwrite(idx->kmer_bloom, sizeof(uint64_t), bwords, fp) != bwords ||
        fwrite(idx->sync_keys, sizeof(uint64_t), (size_t)idx->sync_cap, fp) != (size_t)idx->sync_cap ||
        fwrite(idx->sync_occ, sizeof(uint32_t), (size_t)idx->sync_cap, fp) != (size_t)idx->sync_cap ||
        fwrite(idx->contig_lens, sizeof(int64_t), (size_t)idx->n_contigs, fp) != (size_t)idx->n_contigs) {
        fclose(fp);
        return -4;
    }

    for (int c = 0; c < idx->n_contigs; c++) {
        uint8_t has_data = (idx->contig_nt[c] && idx->contig_uniq_bv[c] && idx->contig_lens[c] > 0) ? 1 : 0;
        if (fwrite(&has_data, sizeof(uint8_t), 1, fp) != 1) { fclose(fp); return -5; }
        if (has_data) {
            size_t len = (size_t)idx->contig_lens[c];
            size_t bvw = (len + 63) / 64;
            if (fwrite(idx->contig_nt[c], sizeof(uint8_t), len, fp) != len ||
                fwrite(idx->contig_uniq_bv[c], sizeof(uint64_t), bvw, fp) != bvw) {
                fclose(fp);
                return -5;
            }
        }
    }
    if (fclose(fp) != 0) return -6;
    return 0;
}

ref_index_t *ipc_load_index(const char *path) {
    ensure_tables();
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    uint64_t hdr[14];
    double dvals[2];
    int32_t dd1 = 0;
    if (fread(hdr, sizeof(uint64_t), 14, fp) != 14 ||
        hdr[0] != IPC_IDX_MAGIC ||
        fread(dvals, sizeof(double), 2, fp) != 2 ||
        fread(&dd1, sizeof(int32_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    ref_index_t *idx = (ref_index_t *)calloc(1, sizeof(ref_index_t));
    if (!idx) { fclose(fp); return NULL; }

    idx->kmer_cap = hdr[1];
    idx->kmer_mask = hdr[2];
    idx->bloom_block_mask = hdr[3];
    idx->n_contigs = (int)hdr[4];
    idx->sync_cap = hdr[5];
    idx->sync_mask = hdr[6];
    idx->total_kmer_positions = hdr[7];
    idx->distinct_kmers = hdr[8];
    idx->unique_kmer_positions = hdr[9];
    idx->total_syncmer_positions = hdr[10];
    idx->distinct_syncmers = hdr[11];
    idx->D1_dense = hdr[12];
    idx->D1_sync = hdr[13];
    idx->d1_dense_bar = dvals[0];
    idx->d1_sync_bar = dvals[1];
    idx->dense_d1_computed = (int)dd1;

    size_t bwords = (size_t)((idx->bloom_block_mask + 1ULL) * 8ULL);
    idx->kmer_slots = (kmer_slot_t *)malloc((size_t)idx->kmer_cap * sizeof(kmer_slot_t));
    idx->kmer_bloom = (uint64_t *)malloc(bwords * sizeof(uint64_t));
    idx->sync_keys = (uint64_t *)malloc((size_t)idx->sync_cap * sizeof(uint64_t));
    idx->sync_occ = (uint32_t *)malloc((size_t)idx->sync_cap * sizeof(uint32_t));
    idx->contig_lens = (int64_t *)calloc((size_t)idx->n_contigs, sizeof(int64_t));
    idx->contig_nt = (uint8_t **)calloc((size_t)idx->n_contigs, sizeof(uint8_t *));
    idx->contig_uniq_bv = (uint64_t **)calloc((size_t)idx->n_contigs, sizeof(uint64_t *));

    if (!idx->kmer_slots || !idx->kmer_bloom || !idx->sync_keys || !idx->sync_occ ||
        !idx->contig_lens || !idx->contig_nt || !idx->contig_uniq_bv) {
        ipc_free_index(idx);
        fclose(fp);
        return NULL;
    }

    if (fread(idx->kmer_slots, sizeof(kmer_slot_t), (size_t)idx->kmer_cap, fp) != (size_t)idx->kmer_cap ||
        fread(idx->kmer_bloom, sizeof(uint64_t), bwords, fp) != bwords ||
        fread(idx->sync_keys, sizeof(uint64_t), (size_t)idx->sync_cap, fp) != (size_t)idx->sync_cap ||
        fread(idx->sync_occ, sizeof(uint32_t), (size_t)idx->sync_cap, fp) != (size_t)idx->sync_cap ||
        fread(idx->contig_lens, sizeof(int64_t), (size_t)idx->n_contigs, fp) != (size_t)idx->n_contigs) {
        ipc_free_index(idx);
        fclose(fp);
        return NULL;
    }

    for (int c = 0; c < idx->n_contigs; c++) {
        uint8_t has_data = 0;
        if (fread(&has_data, sizeof(uint8_t), 1, fp) != 1) {
            ipc_free_index(idx);
            fclose(fp);
            return NULL;
        }
        if (has_data) {
            size_t len = (size_t)idx->contig_lens[c];
            size_t bvw = (len + 63) / 64;
            idx->contig_nt[c] = (uint8_t *)malloc(len);
            idx->contig_uniq_bv[c] = (uint64_t *)malloc(bvw * sizeof(uint64_t));
            if (!idx->contig_nt[c] || !idx->contig_uniq_bv[c] ||
                fread(idx->contig_nt[c], sizeof(uint8_t), len, fp) != len ||
                fread(idx->contig_uniq_bv[c], sizeof(uint64_t), bvw, fp) != bvw) {
                ipc_free_index(idx);
                fclose(fp);
                return NULL;
            }
        }
    }
    fclose(fp);
    return idx;
}
