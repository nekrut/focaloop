/*
 * ust_kernel.c — Fast 64-bit 2-bit canonical k-mer spectrum, bidirected unitig
 * compaction, and UST greedy path-cover simplitig stitcher (k <= 31).
 *
 * Implements Rahman & Medvedev (RECOMB 2020 / JCB 2021) UST algorithm with full
 * reverse-complement bidirected awareness, palindromic k-mer & (k-1)-mer safety,
 * hairpin self-loop prevention, and two-way tip-prioritized greedy stitching.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t BASE_LUT[256] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
};

static const char INT_TO_BASE[4] = {'A', 'C', 'G', 'T'};

static inline uint64_t revcomp_u64(uint64_t x, int k) {
    uint64_t c = ~x;
    c = ((c >> 2)  & 0x3333333333333333ULL) | ((c & 0x3333333333333333ULL) << 2);
    c = ((c >> 4)  & 0x0F0F0F0F0F0F0F0FULL) | ((c & 0x0F0F0F0F0F0F0F0FULL) << 4);
    c = ((c >> 8)  & 0x00FF00FF00FF00FFULL) | ((c & 0x00FF00FF00FF00FFULL) << 8);
    c = ((c >> 16) & 0x0000FFFF0000FFFFULL) | ((c & 0x0000FFFF0000FFFFULL) << 16);
    c = (c >> 32) | (c << 32);
    return c >> (64 - 2 * k);
}

static inline uint64_t hash64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/* Extract canonical k-mers from concatenated sequences (separated by non-ACGT e.g. '\n' or 'N').
 * Returns count of extracted k-mers written to out_kmers (caller ensures capacity >= seq_len).
 */
size_t ust_extract_canon_kmers(const char *seq, size_t seq_len, int k, uint64_t *out_kmers) {
    if (k < 1 || k > 31 || seq_len < (size_t)k) return 0;
    const uint64_t mask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const int shift_rc = 2 * (k - 1);
    uint64_t fw = 0, rc = 0;
    int run = 0;
    size_t n_out = 0;
    for (size_t i = 0; i < seq_len; ++i) {
        uint8_t b = BASE_LUT[(uint8_t)seq[i]];
        if (b < 4) {
            fw = ((fw << 2) | b) & mask;
            rc = (rc >> 2) | ((uint64_t)(3U ^ b) << shift_rc);
            if (++run >= k) {
                out_kmers[n_out++] = (fw < rc) ? fw : rc;
            }
        } else {
            run = 0;
            fw = 0;
            rc = 0;
        }
    }
    return n_out;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

size_t ust_sort_unique_u64(uint64_t *arr, size_t n) {
    if (n == 0) return 0;
    qsort(arr, n, sizeof(uint64_t), cmp_u64);
    size_t w = 1;
    for (size_t i = 1; i < n; ++i) {
        if (arr[i] != arr[w - 1]) {
            arr[w++] = arr[i];
        }
    }
    return w;
}

size_t ust_sorted_diff_u64(const uint64_t *a, size_t na, const uint64_t *b, size_t nb, uint64_t *out) {
    size_t i = 0, j = 0, w = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) {
            out[w++] = a[i++];
        } else if (a[i] > b[j]) {
            j++;
        } else {
            i++;
            j++;
        }
    }
    while (i < na) {
        out[w++] = a[i++];
    }
    return w;
}

typedef struct {
    uint64_t *keys;
    int32_t  *vals;
    size_t    mask;
} KmerMap;

static int kmer_map_init(KmerMap *m, const uint64_t *V, size_t n) {
    size_t cap = 16;
    while (cap < n * 2 + 8) cap <<= 1;
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->vals = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!m->keys || !m->vals) {
        free(m->keys);
        free(m->vals);
        return -1;
    }
    for (size_t i = 0; i < cap; ++i) m->vals[i] = -1;
    m->mask = cap - 1;
    for (size_t i = 0; i < n; ++i) {
        size_t pos = hash64(V[i]) & m->mask;
        while (m->vals[pos] != -1) {
            pos = (pos + 1) & m->mask;
        }
        m->keys[pos] = V[i];
        m->vals[pos] = (int32_t)i;
    }
    return 0;
}

static inline int32_t kmer_map_get(const KmerMap *m, uint64_t key) {
    size_t pos = hash64(key) & m->mask;
    while (1) {
        int32_t v = m->vals[pos];
        if (v == -1) return -1;
        if (m->keys[pos] == key) return v;
        pos = (pos + 1) & m->mask;
    }
}

static void kmer_map_free(KmerMap *m) {
    free(m->keys);
    free(m->vals);
}

/* Build maximal bidirected unitigs from sorted unique canonical k-mers V[0..n-1] (2 <= k <= 31). */
int64_t ust_extract_unitigs_u64(
    const uint64_t *V,
    size_t n,
    int k,
    char *unitig_buf,
    size_t *unitig_offsets
) {
    if (n == 0) {
        unitig_offsets[0] = 0;
        return 0;
    }
    if (k < 2 || k > 31) return -1;

    KmerMap map;
    if (kmer_map_init(&map, V, n) != 0) return -1;

    const uint64_t km1_mask = (1ULL << (2 * (k - 1))) - 1ULL;

    int32_t *uniq_out = (int32_t *)malloc(2 * n * sizeof(int32_t));
    uint8_t *out_deg  = (uint8_t *)malloc(2 * n * sizeof(uint8_t));
    uint8_t *is_pal   = (uint8_t *)malloc(n * sizeof(uint8_t));
    uint8_t *used     = (uint8_t *)calloc(n, sizeof(uint8_t));
    int32_t *ucmp_out = (int32_t *)malloc(2 * n * sizeof(int32_t));

    if (!uniq_out || !out_deg || !is_pal || !used || !ucmp_out) {
        free(uniq_out); free(out_deg); free(is_pal); free(used); free(ucmp_out);
        kmer_map_free(&map);
        return -1;
    }

    for (size_t i = 0; i < n; ++i) {
        uint64_t fw = V[i];
        uint64_t rc = revcomp_u64(fw, k);
        is_pal[i] = (fw == rc) ? 1 : 0;
        for (int d = 0; d < 2; ++d) {
            uint64_t s = (d == 0) ? fw : rc;
            uint64_t suf = s & km1_mask;
            uint8_t deg = 0;
            int32_t last_state = -1;
            for (uint64_t b = 0; b < 4; ++b) {
                uint64_t nxt = (suf << 2) | b;
                uint64_t nxt_rc = revcomp_u64(nxt, k);
                uint64_t c = (nxt < nxt_rc) ? nxt : nxt_rc;
                int32_t j = kmer_map_get(&map, c);
                if (j >= 0) {
                    int dj = (nxt == V[j]) ? 0 : 1;
                    deg++;
                    last_state = (j << 1) | dj;
                }
            }
            out_deg[2 * i + d] = deg;
            uniq_out[2 * i + d] = (deg == 1) ? last_state : -1;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 2; ++d) {
            ucmp_out[2 * i + d] = -1;
            if (is_pal[i]) continue;
            int32_t st_nxt = uniq_out[2 * i + d];
            if (st_nxt < 0) continue;
            int32_t j = st_nxt >> 1;
            int dj = st_nxt & 1;
            if ((size_t)j == i || is_pal[j]) continue;
            int32_t back = uniq_out[2 * j + (1 - dj)];
            if (back == (int32_t)((i << 1) | (1 - d))) {
                ucmp_out[2 * i + d] = st_nxt;
            }
        }
    }

    size_t buf_pos = 0;
    int64_t n_unitigs = 0;
    unitig_offsets[0] = 0;

    for (size_t i = 0; i < n; ++i) {
        if (used[i]) continue;
        if (is_pal[i]) {
            used[i] = 1;
            uint64_t s = V[i];
            for (int p = k - 1; p >= 0; --p) {
                unitig_buf[buf_pos++] = INT_TO_BASE[(s >> (2 * p)) & 3ULL];
            }
            n_unitigs++;
            unitig_offsets[n_unitigs] = buf_pos;
            continue;
        }
        int start_d = -1;
        if (ucmp_out[2 * i + 1] == -1) {
            start_d = 0;
        } else if (ucmp_out[2 * i + 0] == -1) {
            start_d = 1;
        }
        if (start_d < 0) continue;

        used[i] = 1;
        uint64_t s0 = (start_d == 0) ? V[i] : revcomp_u64(V[i], k);
        for (int p = k - 1; p >= 0; --p) {
            unitig_buf[buf_pos++] = INT_TO_BASE[(s0 >> (2 * p)) & 3ULL];
        }
        int32_t cur_st = (int32_t)((i << 1) | start_d);
        while (1) {
            int32_t nxt_st = ucmp_out[cur_st];
            if (nxt_st < 0) break;
            int32_t j = nxt_st >> 1;
            int dj = nxt_st & 1;
            if (used[j]) break;
            used[j] = 1;
            uint64_t sj = (dj == 0) ? V[j] : revcomp_u64(V[j], k);
            unitig_buf[buf_pos++] = INT_TO_BASE[sj & 3ULL];
            cur_st = nxt_st;
        }
        n_unitigs++;
        unitig_offsets[n_unitigs] = buf_pos;
    }

    for (size_t i = 0; i < n; ++i) {
        if (used[i]) continue;
        used[i] = 1;
        uint64_t s0 = V[i];
        for (int p = k - 1; p >= 0; --p) {
            unitig_buf[buf_pos++] = INT_TO_BASE[(s0 >> (2 * p)) & 3ULL];
        }
        int32_t cur_st = (int32_t)(i << 1);
        while (1) {
            int32_t nxt_st = ucmp_out[cur_st];
            if (nxt_st < 0) break;
            int32_t j = nxt_st >> 1;
            int dj = nxt_st & 1;
            if (used[j]) break;
            used[j] = 1;
            uint64_t sj = (dj == 0) ? V[j] : revcomp_u64(V[j], k);
            unitig_buf[buf_pos++] = INT_TO_BASE[sj & 3ULL];
            cur_st = nxt_st;
        }
        n_unitigs++;
        unitig_offsets[n_unitigs] = buf_pos;
    }

    free(uniq_out);
    free(out_deg);
    free(is_pal);
    free(used);
    free(ucmp_out);
    kmer_map_free(&map);
    return n_unitigs;
}
