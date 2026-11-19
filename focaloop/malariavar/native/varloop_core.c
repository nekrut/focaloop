/*
 * varloop_core.c — Memory-bounded, lock-free multi-thread streaming core for Diploid VarLoop.
 *
 * Components:
 *   1. Canonical 31-mer non-DUST target bait table builder (vl_kset_build).
 *   2. Fast C lockstep paired FASTQ framing, QNAME stem check, adapter trim, deterministic
 *      splitmix64 subsampling, and 31-mer target sieve (vl_fastq_sieve_paired).
 *   3. Contig-sharded + 64-kb lazy-paged SAM pileup accumulator (vl_acc_*) with:
 *        - O(1) contig name hash lookup (supports 1 to 65,535 contigs without linear scan)
 *        - Lock-free per-shard pthread consumers (zero cross-thread locks, exact mate-overlap)
 *        - Exact 62-bit streaming coordinate+TLEN PCR/optical dedup (up to 4,096 contigs & 4.29 Gb/contig)
 *        - 64-kb lazy-allocated pages (counts[65536][4], delc[65536], clipc[65536], uncap_dp[1024])
 *        - On-the-fly open-addressing indel event hash table + >31 bp insertion sequence pool
 *        - Bounded informative read capture (indels, soft-clips, multi-mismatch & zoom loci)
 *   4. Exact semi-global affine-gap Phred-weighted read-vs-haplotype log-likelihood scorer
 *      (vl_score_reads_haplotypes) for active-region diploid haplotype genotyping.
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VL_K 31
#define VL_KMASK ((1ULL << (2 * VL_K)) - 1ULL)
#define VL_PAGE_BITS 16
#define VL_PAGE_BP (1 << VL_PAGE_BITS) /* 65536 */
#define VL_PAGE_MASK (VL_PAGE_BP - 1)
#define VL_BIN64_PER_PAGE 1024
#define VL_BIN16_PER_PAGE 4096
#define VL_MAX_CAP_PER_BIN64 96

static const uint8_t ENC_LUT[256] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
};

static inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline uint64_t fnv1a_bytes(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int write_all(int fd, const uint8_t *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, buf + off, len - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}

/* Checked memory allocators: fail loudly with abort() on OOM or size overflow (zero silent drops) */
static void *xmalloc(size_t sz) {
    void *p = malloc(sz ? sz : 1);
    if (!p) {
        fprintf(stderr, "[VarLoop FATAL OOM] malloc(%zu) failed\n", sz);
        abort();
    }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    if (n > 0 && sz > SIZE_MAX / n) {
        fprintf(stderr, "[VarLoop FATAL] calloc size overflow (%zu * %zu)\n", n, sz);
        abort();
    }
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) {
        fprintf(stderr, "[VarLoop FATAL OOM] calloc(%zu, %zu) failed\n", n, sz);
        abort();
    }
    return p;
}

static void *xrealloc(void *ptr, size_t sz) {
    void *p = realloc(ptr, sz ? sz : 1);
    if (!p) {
        fprintf(stderr, "[VarLoop FATAL OOM] realloc(%zu) failed\n", sz);
        abort();
    }
    return p;
}

/* ============================================================================
 * 1. Canonical 31-mer Set Construction & Lookup
 * ============================================================================ */

static inline int32_t dust_score_31(const uint8_t *seq, int64_t pos) {
    int32_t cnt[64];
    memset(cnt, 0, sizeof(cnt));
    int32_t s = 0;
    for (int j = 0; j < VL_K - 2; j++) {
        uint8_t a = seq[pos + j], b = seq[pos + j + 1], c = seq[pos + j + 2];
        if (a > 3 || b > 3 || c > 3) continue;
        int t = (a << 4) | (b << 2) | c;
        s += cnt[t];
        cnt[t]++;
    }
    return s;
}

static inline int kset_insert(uint64_t *tab, uint64_t mask, uint64_t key) {
    uint64_t h = mix64(key) & mask;
    uint64_t k1 = key + 1ULL;
    for (uint64_t step = 0; step <= mask; step++) {
        uint64_t v = tab[h];
        if (v == 0ULL) {
            tab[h] = k1;
            return 1;
        }
        if (v == k1) return 0;
        h = (h + 1ULL) & mask;
    }
    return -1;
}

static inline int kset_has(const uint64_t *tab, uint64_t mask, uint64_t key) {
    uint64_t h = mix64(key) & mask;
    uint64_t k1 = key + 1ULL;
    for (uint64_t step = 0; step <= mask; step++) {
        uint64_t v = tab[h];
        if (v == k1) return 1;
        if (v == 0ULL) return 0;
        h = (h + 1ULL) & mask;
    }
    return 0;
}

int vl_kset_build(
    const uint8_t *seq,
    const int64_t *iv_starts,
    const int64_t *iv_ends,
    int64_t n_iv,
    uint64_t *tab,
    uint64_t tab_len,
    int32_t dust_max,
    int64_t *out_stats
) {
    if ((tab_len & (tab_len - 1ULL)) != 0ULL) return -1;
    uint64_t mask = tab_len - 1ULL;
    int64_t max_ins = (int64_t)((tab_len * 3ULL) / 4ULL);
    int64_t n_ins = 0, n_dust = 0;
    for (int64_t iv = 0; iv < n_iv; iv++) {
        int64_t s = iv_starts[iv], e = iv_ends[iv];
        uint64_t fw = 0ULL, rv = 0ULL;
        int run = 0;
        for (int64_t i = s; i < e; i++) {
            uint8_t c = seq[i];
            if (c > 3) {
                run = 0;
                continue;
            }
            fw = ((fw << 2) | (uint64_t)c) & VL_KMASK;
            rv = (rv >> 2) | ((uint64_t)(3 - c) << (2 * (VL_K - 1)));
            run++;
            if (run >= VL_K) {
                if (dust_score_31(seq, i - VL_K + 1) > dust_max) {
                    n_dust++;
                    continue;
                }
                uint64_t canon = fw < rv ? fw : rv;
                int ins = kset_insert(tab, mask, canon);
                if (ins < 0) return -2;
                n_ins += ins;
                if (n_ins >= max_ins) return -2;
            }
        }
    }
    if (out_stats) {
        out_stats[0] = n_ins;
        out_stats[1] = n_dust;
    }
    return 0;
}

static inline int read_kset_hits(
    const uint8_t *buf,
    int64_t ln,
    const uint64_t *tab,
    uint64_t mask,
    int32_t stride,
    int32_t need
) {
    if (ln < VL_K) return 0;
    int hits = 0;
    for (int64_t p = 0; p + VL_K <= ln; p += stride) {
        uint64_t fw = 0ULL, rv = 0ULL;
        int ok = 1;
        for (int j = 0; j < VL_K; j++) {
            uint8_t c = ENC_LUT[buf[p + j]];
            if (c > 3) {
                ok = 0;
                break;
            }
            fw = (fw << 2) | (uint64_t)c;
            rv |= (uint64_t)(3 - c) << (2 * j);
        }
        if (ok && kset_has(tab, mask, fw < rv ? fw : rv)) {
            hits++;
            if (hits >= need) return hits;
        }
    }
    return hits;
}

/* ============================================================================
 * 2. Fast Lockstep Paired FASTQ Streamer & Target Sieve
 * ============================================================================ */

typedef struct {
    int fd;
    uint8_t *buf;
    size_t cap;
    size_t pos;
    size_t len;
    int eof;
} fd_reader_t;

static int fdr_init(fd_reader_t *r, int fd, size_t cap) {
    r->fd = fd;
    r->cap = cap;
    r->pos = 0;
    r->len = 0;
    r->eof = 0;
    r->buf = (uint8_t *)xmalloc(cap);
    return 0;
}

static void fdr_free(fd_reader_t *r) {
    free(r->buf);
    r->buf = NULL;
}

static int fdr_fill(fd_reader_t *r) {
    if (r->pos > 0) {
        size_t rem = r->len - r->pos;
        if (rem > 0) memmove(r->buf, r->buf + r->pos, rem);
        r->len = rem;
        r->pos = 0;
    }
    while (!r->eof && r->len < r->cap) {
        ssize_t n = read(r->fd, r->buf + r->len, r->cap - r->len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            r->eof = 1;
            break;
        }
        r->len += (size_t)n;
        if (r->len >= (r->cap >> 1)) break;
    }
    return 0;
}

typedef struct {
    const uint8_t *hs; int64_t hlen;
    const uint8_t *ss; int64_t slen;
    const uint8_t *qs; int64_t qlen;
} fq_rec_t;

/* Returns: 1 = got record, 0 = clean EOF, -1 = framing error */
static int fdr_next_fq(fd_reader_t *r, fq_rec_t *out) {
    for (int attempt = 0; attempt < 2; attempt++) {
        size_t i = r->pos, n = r->len;
        if (i == n && r->eof) return 0;
        const uint8_t *b = r->buf;
        size_t e1 = i;
        while (e1 < n && b[e1] != '\n') e1++;
        size_t e2 = e1 + 1;
        while (e2 < n && b[e2] != '\n') e2++;
        size_t e3 = e2 + 1;
        while (e3 < n && b[e3] != '\n') e3++;
        size_t e4 = e3 + 1;
        while (e4 < n && b[e4] != '\n') e4++;
        if (e4 >= n) {
            if (r->eof) {
                /* Handle trailing file without final newline */
                if (e3 < n && e4 == n && e4 > e3 + 1) {
                    /* e4 sits at end of buffer */
                } else {
                    return -1;
                }
            } else {
                if (attempt == 0) {
                    if (fdr_fill(r) < 0) return -1;
                    continue;
                }
                return -1;
            }
        }
        size_t h1 = (e1 > i && b[e1 - 1] == '\r') ? e1 - 1 : e1;
        size_t s1 = (e2 > e1 + 1 && b[e2 - 1] == '\r') ? e2 - 1 : e2;
        size_t p1 = (e3 > e2 + 1 && b[e3 - 1] == '\r') ? e3 - 1 : e3;
        size_t q1 = (e4 > e3 + 1 && b[e4 - 1] == '\r') ? e4 - 1 : e4;
        if (b[i] != '@' || h1 <= i + 1 || p1 <= e2 + 1 || b[e2 + 1] != '+') return -1;
        int64_t slen = (int64_t)(s1 - (e1 + 1));
        int64_t qlen = (int64_t)(q1 - (e3 + 1));
        if (slen != qlen || slen <= 0) return -1;
        out->hs = b + i + 1;
        out->hlen = (int64_t)(h1 - (i + 1));
        out->ss = b + e1 + 1;
        out->slen = slen;
        out->qs = b + e3 + 1;
        out->qlen = qlen;
        r->pos = (e4 < n) ? (e4 + 1) : n;
        return 1;
    }
    return -1;
}

static inline int64_t qname_stem_len(const uint8_t *h, int64_t n) {
    int64_t e = 0;
    while (e < n && h[e] != ' ' && h[e] != '\t') e++;
    if (e >= 2 && (h[e - 2] == '/' || h[e - 2] == '.') && (h[e - 1] == '1' || h[e - 1] == '2' || h[e - 1] == '0')) {
        return e - 2;
    }
    return e;
}

static inline int64_t find_adapter(const uint8_t *s, int64_t ln) {
    static const uint8_t AD[13] = "AGATCGGAAGAGC";
    if (ln < 13) return ln;
    for (int64_t i = 0; i <= ln - 13; i++) {
        if (s[i] == 'A' && memcmp(s + i, AD, 13) == 0) return i;
    }
    return ln;
}

static inline size_t write_u64_ascii(uint8_t *dst, uint64_t v) {
    if (v == 0) {
        dst[0] = '0';
        return 1;
    }
    uint8_t tmp[24];
    size_t k = 0;
    while (v > 0) {
        tmp[k++] = (uint8_t)('0' + (v % 10ULL));
        v /= 10ULL;
    }
    for (size_t j = 0; j < k; j++) dst[j] = tmp[k - 1 - j];
    return k;
}

static size_t append_renamed_fq(uint8_t *dst, uint64_t spot, int mate_suffix, const uint8_t *s, const uint8_t *q, int64_t ln) {
    size_t o = 0;
    dst[o++] = '@';
    o += write_u64_ascii(dst + o, spot);
    if (mate_suffix > 0) {
        dst[o++] = '/';
        dst[o++] = (uint8_t)('0' + mate_suffix);
    }
    dst[o++] = '\n';
    memcpy(dst + o, s, (size_t)ln);
    o += (size_t)ln;
    dst[o++] = '\n';
    dst[o++] = '+';
    dst[o++] = '\n';
    memcpy(dst + o, q, (size_t)ln);
    o += (size_t)ln;
    dst[o++] = '\n';
    return o;
}

int vl_fastq_sieve_paired(
    int fd1,
    int fd2,
    int fd_out_interleaved,
    int fd_out1,
    int fd_out2,
    const uint64_t *tab,
    uint64_t tab_len,
    int32_t stride,
    int32_t need,
    uint64_t keep_thr,
    int32_t trim_adapter,
    int32_t min_len,
    int64_t max_spots,
    int64_t *stats
) {
    /* stats: [0]=spots_seen, [1]=spots_subsampled, [2]=spots_emitted, [3]=bases_seen,
     *        [4]=bases_emitted, [5]=errors, [6]=max_rlen, [7]=adapter_trimmed */
    const size_t BUF_CAP = 8U << 20;
    const size_t OUT_CAP = 4U << 20;
    fd_reader_t r1, r2;
    if (fdr_init(&r1, fd1, BUF_CAP) < 0) return -2;
    if (fdr_init(&r2, fd2, BUF_CAP) < 0) { fdr_free(&r1); return -2; }

    uint8_t *ob_int = fd_out_interleaved >= 0 ? (uint8_t *)xmalloc(OUT_CAP + 65536) : NULL;
    uint8_t *ob_1   = fd_out1 >= 0 ? (uint8_t *)xmalloc(OUT_CAP + 65536) : NULL;
    uint8_t *ob_2   = fd_out2 >= 0 ? (uint8_t *)xmalloc(OUT_CAP + 65536) : NULL;
    size_t oi = 0, o1 = 0, o2 = 0;

    uint64_t kmask = (tab && tab_len > 0) ? (tab_len - 1ULL) : 0ULL;
    uint64_t spot = 0;
    int rc = 0;

    for (;;) {
        if (max_spots > 0 && (int64_t)spot >= max_spots) break;
        fq_rec_t a, b;
        int s1 = fdr_next_fq(&r1, &a);
        int s2 = fdr_next_fq(&r2, &b);
        if (s1 == 0 && s2 == 0) break;
        if (s1 <= 0 || s2 <= 0) {
            stats[5]++;
            rc = -3; /* premature EOF or malformed FASTQ */
            break;
        }
        int64_t l1 = qname_stem_len(a.hs, a.hlen);
        int64_t l2 = qname_stem_len(b.hs, b.hlen);
        if (l1 != l2 || memcmp(a.hs, b.hs, (size_t)l1) != 0) {
            stats[5]++;
            rc = -4; /* QNAME mismatch */
            break;
        }

        stats[0]++;
        stats[3] += a.slen + b.slen;
        if (a.slen > stats[6]) stats[6] = a.slen;
        if (b.slen > stats[6]) stats[6] = b.slen;

        if ((mix64(spot) >> 32) >= keep_thr) {
            spot++;
            continue;
        }
        stats[1]++;

        int64_t ln1 = a.slen, ln2 = b.slen;
        if (trim_adapter) {
            int64_t t1 = find_adapter(a.ss, ln1);
            int64_t t2 = find_adapter(b.ss, ln2);
            if (t1 < ln1) { stats[7]++; ln1 = t1; }
            if (t2 < ln2) { stats[7]++; ln2 = t2; }
            if (ln1 < min_len || ln2 < min_len) {
                spot++;
                continue;
            }
        }

        int pass = 1;
        if (tab != NULL && need > 0) {
            int hits = read_kset_hits(a.ss, ln1, tab, kmask, stride, need);
            if (hits < need) {
                hits += read_kset_hits(b.ss, ln2, tab, kmask, stride, need - hits);
            }
            pass = (hits >= need);
        }

        if (pass) {
            stats[2]++;
            stats[4] += ln1 + ln2;
            if (ob_int) {
                oi += append_renamed_fq(ob_int + oi, spot, 0, a.ss, a.qs, ln1);
                oi += append_renamed_fq(ob_int + oi, spot, 0, b.ss, b.qs, ln2);
                if (oi >= OUT_CAP) {
                    if (write_all(fd_out_interleaved, ob_int, oi) < 0) { rc = -5; break; }
                    oi = 0;
                }
            }
            if (ob_1 && ob_2) {
                o1 += append_renamed_fq(ob_1 + o1, spot, 1, a.ss, a.qs, ln1);
                o2 += append_renamed_fq(ob_2 + o2, spot, 2, b.ss, b.qs, ln2);
                if (o1 >= OUT_CAP) {
                    if (write_all(fd_out1, ob_1, o1) < 0) { rc = -5; break; }
                    o1 = 0;
                }
                if (o2 >= OUT_CAP) {
                    if (write_all(fd_out2, ob_2, o2) < 0) { rc = -5; break; }
                    o2 = 0;
                }
            }
        }
        spot++;
    }

    if (rc == 0) {
        if (ob_int && oi > 0 && write_all(fd_out_interleaved, ob_int, oi) < 0) rc = -5;
        if (ob_1 && o1 > 0 && write_all(fd_out1, ob_1, o1) < 0) rc = -5;
        if (ob_2 && o2 > 0 && write_all(fd_out2, ob_2, o2) < 0) rc = -5;
    }

    free(ob_int);
    free(ob_1);
    free(ob_2);
    fdr_free(&r1);
    fdr_free(&r2);
    return rc;
}

/* ============================================================================
 * 3. Contig-Sharded + 64-kb Lazy-Paged SAM Pileup Accumulator
 * ============================================================================ */

typedef struct {
    uint16_t counts[VL_PAGE_BP][4];
    uint16_t delc[VL_PAGE_BP];
    uint16_t clipc[VL_PAGE_BP];
    uint16_t uncap_dp[VL_BIN16_PER_PAGE];
    uint16_t cap_cnt[VL_BIN64_PER_PAGE];
} vl_page_t;

typedef struct {
    int32_t ctg;       /* -1 = empty slot */
    int32_t pos0;      /* 0-based within contig */
    int16_t kind;      /* 1 = DEL, 2 = INS */
    int16_t len;       /* indel length */
    uint32_t count;
    uint64_t code;     /* <=31 bp: 2-bit packed; >31 bp: offset+1 into ins_pool (high 32) | hash32 */
} vl_indel_slot_t;

typedef struct {
    int32_t pos0;      /* 0-based reference alignment start on contig */
    int32_t rspan;     /* reference span */
    int16_t lead_clip;
    int16_t trail_clip;
    uint16_t len;      /* read sequence length */
    uint8_t mapq;
    uint8_t flag_low;
    uint64_t seq_off;  /* 64-bit offset in contig seq_pool (safe well past 4 GiB per contig) */
} vl_cap_read_t;

typedef struct {
    int32_t id;
    int64_t length;
    int64_t ref_off;
    int64_t bin_off;
    int32_t n_pages;
    int32_t owner_shard;
    uint8_t *page_allowed;     /* NULL = all pages allowed, else bool[n_pages] */
    vl_page_t **pages;         /* lazy: NULL until touched */
    int64_t n_pages_alloc;

    /* Optional zoom intervals [start0, end0) sorted on this contig */
    int32_t *zoom_s;
    int32_t *zoom_e;
    int32_t n_zoom;

    /* Captured informative reads on this contig (owned exclusively by owner_shard) */
    vl_cap_read_t *reads;
    size_t n_reads;
    size_t cap_reads;
    uint8_t *seq_pool;
    size_t seq_pool_used;
    size_t seq_pool_cap;
    int reads_sorted;
} vl_contig_t;

#define VL_QUEUE_CAP 24

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} sam_batch_t;

typedef struct vl_acc_s vl_acc_t;

typedef struct {
    int shard_id;
    vl_acc_t *acc;
    pthread_t thread;

    /* Bounded SPSC/MPSC mutex-cond batch queue */
    pthread_mutex_t mu;
    pthread_cond_t cv_nonempty;
    pthread_cond_t cv_nonfull;
    sam_batch_t *q[VL_QUEUE_CAP];
    int q_head;
    int q_tail;
    int q_count;
    int done;

    /* Writer-side staging batch before enqueueing */
    sam_batch_t *staging;

    /* Per-shard private state (zero contention) */
    uint64_t *dedup_tab;       /* power-of-2 size (e.g. 1<<20) */
    uint64_t dedup_mask;
    int64_t prev_spot;
    int32_t prev_ctg;
    int64_t prev_s;
    int64_t prev_e;
    int64_t stats[12];
    int64_t *ccount;

    /* Per-shard open-addressing indel table */
    vl_indel_slot_t *ind_tab;
    size_t ind_cap;            /* power of 2 */
    size_t ind_used;
    char *ins_pool;            /* null-terminated >31 bp insertion strings */
    size_t ins_pool_used;
    size_t ins_pool_cap;
} vl_shard_t;

typedef struct {
    uint64_t hash;
    int32_t ctg_id;
    int32_t name_len;
    const char *name_ptr;
} ctg_ht_slot_t;

struct vl_acc_s {
    int32_t n_contigs;
    vl_contig_t *contigs;
    char *names_blob;
    ctg_ht_slot_t *ctg_ht;
    uint64_t ctg_ht_mask;

    const uint8_t *ref_seq;    /* borrowed pointer to Genome.seq uint8 array */
    int64_t *binbases;         /* length = total_bins; each contig's range owned by owner_shard */
    int64_t total_bins;

    int32_t n_shards;
    vl_shard_t *shards;

    /* Calling/filtering parameters */
    int32_t min_mapq;
    int32_t min_bq;
    int32_t bin_mapq;
    int32_t zoom_min_mapq;
    double max_mm_frac;
    int32_t flt_indel_flank;
    int32_t flt_clip_flank;
    int32_t flt_end_trim;
    int32_t flt_mm_win;
    int32_t flt_mm_n;
    int32_t capture_informative;
    int32_t dedup_bits;
};

static void ind_tab_init(vl_shard_t *sh, size_t init_cap) {
    sh->ind_cap = init_cap;
    sh->ind_used = 0;
    sh->ind_tab = (vl_indel_slot_t *)xmalloc(init_cap * sizeof(vl_indel_slot_t));
    for (size_t i = 0; i < init_cap; i++) sh->ind_tab[i].ctg = -1;
    sh->ins_pool_cap = 1U << 16;
    sh->ins_pool_used = 0;
    sh->ins_pool = (char *)xmalloc(sh->ins_pool_cap);
}

static uint32_t ins_pool_intern(vl_shard_t *sh, const uint8_t *s, int len) {
    if (sh->ins_pool_used + (size_t)len + 2 >= (size_t)UINT32_MAX) {
        fprintf(stderr, "[VarLoop FATAL] ins_pool offset overflow\n");
        abort();
    }
    if (sh->ins_pool_used + (size_t)len + 2 > sh->ins_pool_cap) {
        while (sh->ins_pool_used + (size_t)len + 2 > sh->ins_pool_cap) sh->ins_pool_cap *= 2;
        sh->ins_pool = (char *)xrealloc(sh->ins_pool, sh->ins_pool_cap);
    }
    uint32_t off = (uint32_t)sh->ins_pool_used;
    memcpy(sh->ins_pool + off, s, (size_t)len);
    sh->ins_pool[off + len] = '\0';
    sh->ins_pool_used += (size_t)len + 1;
    return off;
}

static void ind_tab_rehash(vl_shard_t *sh) {
    size_t old_cap = sh->ind_cap;
    vl_indel_slot_t *old = sh->ind_tab;
    size_t new_cap = old_cap * 2;
    vl_indel_slot_t *nw = (vl_indel_slot_t *)xmalloc(new_cap * sizeof(vl_indel_slot_t));
    for (size_t i = 0; i < new_cap; i++) nw[i].ctg = -1;
    size_t mask = new_cap - 1;
    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].ctg < 0) continue;
        uint64_t key_low = (old[i].kind == 1 || old[i].len <= 31) ? old[i].code : (old[i].code & 0xFFFFFFFFULL);
        uint64_t hk = mix64(((uint64_t)old[i].ctg << 40) ^ ((uint64_t)old[i].pos0 << 8) ^
                            ((uint64_t)old[i].kind << 6) ^ (uint64_t)old[i].len ^ mix64(key_low));
        size_t s = (size_t)hk & mask;
        while (nw[s].ctg >= 0) s = (s + 1) & mask;
        nw[s] = old[i];
    }
    free(old);
    sh->ind_tab = nw;
    sh->ind_cap = new_cap;
}

static inline void ind_tab_add(vl_shard_t *sh, int32_t ctg, int32_t pos0, int16_t kind, int16_t len,
                               uint64_t code, const uint8_t *long_seq) {
    if (sh->ind_used * 10 >= sh->ind_cap * 7) ind_tab_rehash(sh);
    size_t mask = sh->ind_cap - 1;
    int is_long_ins = (kind == 2 && len > 31);
    uint64_t key_low = (!is_long_ins) ? code : (code & 0xFFFFFFFFULL);
    uint64_t hk = mix64(((uint64_t)ctg << 40) ^ ((uint64_t)pos0 << 8) ^
                        ((uint64_t)kind << 6) ^ (uint64_t)len ^ mix64(key_low));
    size_t s = (size_t)hk & mask;
    for (;;) {
        vl_indel_slot_t *sl = &sh->ind_tab[s];
        if (sl->ctg < 0) {
            sl->ctg = ctg;
            sl->pos0 = pos0;
            sl->kind = kind;
            sl->len = len;
            sl->count = 1;
            if (!is_long_ins) {
                sl->code = code;
            } else {
                uint32_t pool_off = ins_pool_intern(sh, long_seq, len);
                sl->code = ((uint64_t)(pool_off + 1U) << 32) | (code & 0xFFFFFFFFULL);
            }
            sh->ind_used++;
            return;
        }
        if (sl->ctg == ctg && sl->pos0 == pos0 && sl->kind == kind && sl->len == len) {
            if (!is_long_ins) {
                if (sl->code == code) {
                    if (sl->count < 0xFFFFFFFFU) sl->count++;
                    return;
                }
            } else if ((sl->code & 0xFFFFFFFFULL) == (code & 0xFFFFFFFFULL)) {
                uint32_t off = (uint32_t)(sl->code >> 32) - 1U;
                if (memcmp(sh->ins_pool + off, long_seq, (size_t)len) == 0) {
                    if (sl->count < 0xFFFFFFFFU) sl->count++;
                    return;
                }
            }
        }
        s = (s + 1) & mask;
    }
}

static inline vl_page_t *get_or_alloc_page(vl_contig_t *ctg, int32_t pidx) {
    if (pidx < 0 || pidx >= ctg->n_pages) return NULL;
    vl_page_t *pg = ctg->pages[pidx];
    if (pg != NULL) return pg;
    if (ctg->page_allowed != NULL && !ctg->page_allowed[pidx]) return NULL;
    pg = (vl_page_t *)xcalloc(1, sizeof(vl_page_t));
    ctg->pages[pidx] = pg;
    ctg->n_pages_alloc++;
    return pg;
}

static inline int overlaps_zoom_ctg(const vl_contig_t *ctg, int32_t s0, int32_t e0) {
    if (ctg->n_zoom == 0) return 0;
    int lo = 0, hi = ctg->n_zoom - 1;
    /* Find first interval with zoom_e > s0 */
    int ans = ctg->n_zoom;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (ctg->zoom_e[mid] > s0) {
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return (ans < ctg->n_zoom && ctg->zoom_s[ans] < e0);
}

static void capture_read_ctg(vl_contig_t *ctg, int32_t pos0, int32_t rspan, int16_t lead_c, int16_t trail_c,
                             uint8_t mapq, int32_t flag, const uint8_t *seq, const uint8_t *qual, int32_t qlen) {
    if (qlen <= 0 || qlen > 65535) return;
    int32_t mid = pos0 + (rspan > 0 ? (rspan >> 1) : 0);
    if (mid < 0) mid = 0;
    if (mid >= ctg->length) mid = (int32_t)(ctg->length - 1);
    int32_t pidx = mid >> VL_PAGE_BITS;
    vl_page_t *pg = get_or_alloc_page(ctg, pidx);
    if (!pg) return;
    int b64 = (mid & VL_PAGE_MASK) >> 6;
    if (pg->cap_cnt[b64] >= VL_MAX_CAP_PER_BIN64) return;
    pg->cap_cnt[b64]++;

    if (ctg->n_reads == ctg->cap_reads) {
        size_t ncap = ctg->cap_reads ? ctg->cap_reads * 2 : 4096;
        if (ncap > SIZE_MAX / sizeof(vl_cap_read_t)) {
            fprintf(stderr, "[VarLoop FATAL] cap_reads overflow\n");
            abort();
        }
        ctg->reads = (vl_cap_read_t *)xrealloc(ctg->reads, ncap * sizeof(vl_cap_read_t));
        ctg->cap_reads = ncap;
    }
    size_t need_bytes = (size_t)qlen * 2U;
    if ((uint64_t)ctg->seq_pool_used > (uint64_t)INT64_MAX - (uint64_t)need_bytes) {
        fprintf(stderr, "[VarLoop FATAL] seq_pool_used exceeded INT64_MAX\n");
        abort();
    }
    if (ctg->seq_pool_used + need_bytes > ctg->seq_pool_cap) {
        size_t pcap = ctg->seq_pool_cap ? ctg->seq_pool_cap * 2 : (1U << 20);
        while (ctg->seq_pool_used + need_bytes > pcap) {
            if (pcap > SIZE_MAX / 2) {
                fprintf(stderr, "[VarLoop FATAL] seq_pool_cap overflow\n");
                abort();
            }
            pcap *= 2;
        }
        ctg->seq_pool = (uint8_t *)xrealloc(ctg->seq_pool, pcap);
        ctg->seq_pool_cap = pcap;
    }
    uint64_t off = (uint64_t)ctg->seq_pool_used;
    memcpy(ctg->seq_pool + off, seq, (size_t)qlen);
    memcpy(ctg->seq_pool + off + (size_t)qlen, qual, (size_t)qlen);
    ctg->seq_pool_used += need_bytes;
    ctg->reads_sorted = 0;

    vl_cap_read_t *r = &ctg->reads[ctg->n_reads++];
    r->pos0 = pos0;
    r->rspan = rspan;
    r->lead_clip = lead_c;
    r->trail_clip = trail_c;
    r->len = (uint16_t)qlen;
    r->mapq = mapq;
    r->flag_low = (uint8_t)(flag & 0xFF);
    r->seq_off = off;
}

static inline int32_t lookup_contig(const vl_acc_t *acc, const uint8_t *s, int32_t len) {
    uint64_t h = fnv1a_bytes(s, (size_t)len);
    uint64_t idx = mix64(h) & acc->ctg_ht_mask;
    for (;;) {
        const ctg_ht_slot_t *sl = &acc->ctg_ht[idx];
        if (sl->ctg_id < 0) return -1;
        if (sl->hash == h && sl->name_len == len && memcmp(sl->name_ptr, s, (size_t)len) == 0) {
            return sl->ctg_id;
        }
        idx = (idx + 1ULL) & acc->ctg_ht_mask;
    }
}

static inline int64_t parse_i64(const uint8_t *p, const uint8_t *e) {
    int64_t v = 0;
    int neg = 0;
    if (p < e && *p == '-') { neg = 1; p++; }
    while (p < e && *p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        p++;
    }
    return neg ? -v : v;
}

static inline int64_t parse_spot_id(const uint8_t *p, const uint8_t *e) {
    int all_num = (p < e);
    int64_t v = 0;
    for (const uint8_t *q = p; q < e; q++) {
        if (*q < '0' || *q > '9') { all_num = 0; break; }
        v = v * 10 + (*q - '0');
    }
    if (all_num) return v;
    int64_t slen = qname_stem_len(p, (int64_t)(e - p));
    return (int64_t)(mix64(fnv1a_bytes(p, (size_t)slen)) & 0x7FFFFFFFFFFFFFFFULL);
}

static int compute_base_mask(
    const uint8_t *cig_s, const uint8_t *cig_e,
    const uint8_t *seq_s, int32_t qlen,
    int64_t g0, const uint8_t *ref_seq, int64_t ref_limit,
    int32_t fi, int32_t fc, int32_t fe, int32_t mw, int32_t mn,
    uint8_t *mask, int32_t *mm
) {
    memset(mask, 0, (size_t)qlen);
    int32_t num = 0, qi = 0, nmm = 0, lead = 0, trail = 0;
    int64_t g = g0;
    for (const uint8_t *p = cig_s; p < cig_e; p++) {
        uint8_t ch = *p;
        if (ch >= '0' && ch <= '9') {
            num = num * 10 + (ch - '0');
            continue;
        }
        if (ch == 'M' || ch == '=' || ch == 'X') {
            if (mn > 0) {
                for (int32_t t = 0; t < num; t++) {
                    if (g + t < ref_limit && qi + t < qlen) {
                        uint8_t b = ENC_LUT[seq_s[qi + t]];
                        uint8_t r = ref_seq[g + t];
                        if (b < 4 && r < 4 && b != r && nmm < 512) mm[nmm++] = qi + t;
                    }
                }
            }
            qi += num;
            g += num;
        } else if (ch == 'I') {
            if (fi > 0) {
                int32_t a = qi - fi < 0 ? 0 : qi - fi;
                int32_t b = qi + num + fi > qlen ? qlen : qi + num + fi;
                for (int32_t q = a; q < b; q++) mask[q] = 1;
            }
            qi += num;
        } else if (ch == 'D') {
            if (fi > 0) {
                int32_t a = qi - fi < 0 ? 0 : qi - fi;
                int32_t b = qi + fi > qlen ? qlen : qi + fi;
                for (int32_t q = a; q < b; q++) mask[q] = 1;
            }
            g += num;
        } else if (ch == 'N') {
            g += num;
        } else if (ch == 'S') {
            if (qi == 0) lead = num;
            else trail = num;
            qi += num;
        }
        num = 0;
    }
    if (fc > 0) {
        if (lead > 0) {
            int32_t b = lead + fc < qlen ? lead + fc : qlen;
            for (int32_t q = lead; q < b; q++) mask[q] = 1;
        }
        if (trail > 0) {
            int32_t a = qlen - trail - fc > 0 ? qlen - trail - fc : 0;
            int32_t b = qlen - trail;
            for (int32_t q = a; q < b; q++) mask[q] = 1;
        }
    }
    if (fe > 0) {
        int32_t b1 = fe < qlen ? fe : qlen;
        for (int32_t q = 0; q < b1; q++) mask[q] = 1;
        int32_t a2 = qlen - fe > 0 ? qlen - fe : 0;
        for (int32_t q = a2; q < qlen; q++) mask[q] = 1;
    }
    if (mn > 1 && nmm >= mn) {
        int a = 0;
        for (int b = 0; b < nmm; b++) {
            while (mm[b] - mm[a] > mw) a++;
            if (b - a + 1 >= mn) {
                for (int32_t q = mm[a]; q <= mm[b]; q++) mask[q] = 1;
            }
        }
    }
    int k = 0;
    for (int32_t q = 0; q < qlen; q++) k += mask[q];
    return k;
}

static void shard_process_line(vl_shard_t *sh, const uint8_t *line, size_t len,
                               uint8_t **bmask_p, size_t *bmask_cap_p, int32_t *mmbuf) {
    vl_acc_t *acc = sh->acc;
    if (len == 0 || line[0] == '@') return;

    const uint8_t *f_s[12], *f_e[12];
    int nf = 0;
    const uint8_t *cur = line;
    const uint8_t *end = line + len;
    if (end > cur && *(end - 1) == '\r') end--;

    for (const uint8_t *p = cur; p <= end; p++) {
        if (p == end || *p == '\t') {
            if (nf < 12) {
                f_s[nf] = cur;
                f_e[nf] = p;
            }
            nf++;
            cur = p + 1;
            if (nf >= 12) break;
        }
    }
    if (nf < 11) return;
    const uint8_t *tag_s = (nf >= 12) ? f_s[11] : end;

    sh->stats[0]++;
    int64_t spot = parse_spot_id(f_s[0], f_e[0]);
    if (spot > sh->stats[4]) sh->stats[4] = spot;

    int32_t flag = (int32_t)parse_i64(f_s[1], f_e[1]);
    if (flag & 0xF04) return;

    int32_t c = lookup_contig(acc, f_s[2], (int32_t)(f_e[2] - f_s[2]));
    if (c < 0) return;
    vl_contig_t *ctg = &acc->contigs[c];

    int64_t pos1_raw = parse_i64(f_s[3], f_e[3]);
    if (pos1_raw < 1 || pos1_raw > ctg->length) return;
    int32_t pos1 = (int32_t)pos1_raw;
    int32_t pos0 = pos1 - 1;
    int32_t mapq = (int32_t)parse_i64(f_s[4], f_e[4]);

    /* Strict SAM SEQ/QUAL/CIGAR validation: reject '*', unequal SEQ/QUAL lengths, or out-of-bounds CIGAR */
    int64_t raw_qlen = (int64_t)(f_e[9] - f_s[9]);
    int64_t raw_quallen = (int64_t)(f_e[10] - f_s[10]);
    if (raw_qlen <= 0 || raw_qlen > 65535 || raw_quallen != raw_qlen ||
        (raw_qlen == 1 && (f_s[9][0] == '*' || f_s[10][0] == '*')) ||
        (f_e[5] <= f_s[5]) || (f_e[5] - f_s[5] == 1 && f_s[5][0] == '*')) {
        sh->stats[11]++;
        return;
    }
    int32_t qlen = (int32_t)raw_qlen;
    const uint8_t *seq_s = f_s[9];
    const uint8_t *qual_s = f_s[10];

    /* Walk CIGAR once with overflow guard (< 2^26 per op) and verify query sum == len(SEQ) */
    int32_t rspan = 0, indel = 0, num = 0, q_cur = 0, lead_clip = 0, trail_clip = 0;
    int cig_bad = 0;
    for (const uint8_t *p = f_s[5]; p < f_e[5]; p++) {
        uint8_t ch = *p;
        if (ch >= '0' && ch <= '9') {
            if (num > (1 << 26)) { cig_bad = 1; break; }
            num = num * 10 + (ch - '0');
            continue;
        }
        if (num <= 0) { cig_bad = 1; break; }
        if (ch == 'M' || ch == '=' || ch == 'X') {
            rspan += num;
            q_cur += num;
        } else if (ch == 'I') {
            indel += num;
            q_cur += num;
        } else if (ch == 'D') {
            rspan += num;
            indel += num;
        } else if (ch == 'N') {
            rspan += num;
        } else if (ch == 'S') {
            if (q_cur == 0) lead_clip = num;
            else trail_clip = num;
            q_cur += num;
        } else if (ch == 'H' || ch == 'P') {
            /* valid non-SEQ-consuming SAM CIGAR operations */
        } else {
            cig_bad = 1;
            break;
        }
        if (rspan > (1 << 28) || indel > (1 << 28) || q_cur > qlen) { cig_bad = 1; break; }
        num = 0;
    }
    if (cig_bad || num != 0 || q_cur != qlen || rspan <= 0) {
        sh->stats[11]++;
        return;
    }

    sh->stats[1]++;

    int32_t end0 = pos0 + rspan;
    if (end0 > ctg->length) end0 = (int32_t)ctg->length;

    /* Quick target-page check: skip off-target reads immediately with zero page allocation */
    if (ctg->page_allowed != NULL) {
        int32_t p_lo = pos0 >> VL_PAGE_BITS;
        int32_t p_hi = (end0 > pos0 ? (end0 - 1) : pos0) >> VL_PAGE_BITS;
        int any_ok = 0;
        for (int32_t px = p_lo; px <= p_hi && px < ctg->n_pages; px++) {
            if (ctg->page_allowed[px]) { any_ok = 1; break; }
        }
        if (!any_ok) return;
    }

    /* 1 kb depth bins (MAPQ >= bin_mapq) */
    if (mapq >= acc->bin_mapq && end0 > pos0) {
        int32_t a = pos0;
        while (a < end0) {
            int64_t bi = ctg->bin_off + (a / 1000);
            int32_t nxt = a + (1000 - (a % 1000));
            if (nxt > end0) nxt = end0;
            acc->binbases[bi] += (nxt - a);
            a = nxt;
        }
    }

    int paired = (flag & 1) != 0;
    int same_spot = (sh->prev_spot == spot);
    int same_chr = ((f_e[6] - f_s[6] == 1) && f_s[6][0] == '=');
    int32_t pn1 = same_chr ? (int32_t)parse_i64(f_s[7], f_e[7]) : 0;
    int32_t tlen = (int32_t)parse_i64(f_s[8], f_e[8]);

    /* Exact 62-bit streaming template PCR/optical duplicate check (run before zoom capture too) */
    if (sh->dedup_tab != NULL && paired && same_chr && tlen != 0 && mapq >= acc->min_mapq && (flag & 2) != 0) {
        uint32_t left_p = (uint32_t)(tlen > 0 ? pos1 : pn1);
        uint32_t span_u = (uint32_t)(tlen < 0 ? -tlen : tlen);
        if (span_u > 0x1FFFFU) span_u = 0x1FFFFU;
        uint64_t side_u = (tlen > 0) ? 0ULL : 1ULL;
        uint64_t dk = (((uint64_t)(c & 0xFFF) << 50) |
                       ((uint64_t)left_p << 18) |
                       ((uint64_t)span_u << 1) |
                       side_u) + 1ULL;
        uint64_t ds = mix64(dk) & sh->dedup_mask;
        if (sh->dedup_tab[ds] == dk) {
            sh->stats[10]++;
            sh->prev_spot = spot; sh->prev_ctg = c; sh->prev_s = pos0; sh->prev_e = end0;
            return;
        }
        sh->dedup_tab[ds] = dk;
    }

    int captured = 0;

    /* Capture reads overlapping explicit zoom intervals even when MAPQ < min_mapq or improper pair */
    if (rspan > 0 && mapq >= acc->zoom_min_mapq && overlaps_zoom_ctg(ctg, pos0, end0)) {
        capture_read_ctg(ctg, pos0, rspan, (int16_t)lead_clip, (int16_t)trail_clip,
                         (uint8_t)(mapq > 255 ? 255 : mapq), flag, seq_s, qual_s, qlen);
        captured = 1;
        sh->stats[8]++;
    }

    if (mapq < acc->min_mapq || (paired && (flag & 2) == 0)) {
        sh->prev_spot = spot; sh->prev_ctg = c; sh->prev_s = pos0; sh->prev_e = end0;
        return;
    }

    sh->ccount[c]++;

    /* Parse NM:i: tag */
    int32_t nm = -1;
    const uint8_t *tp = tag_s;
    while (tp + 5 < end) {
        if (tp[0] == 'N' && tp[1] == 'M' && tp[2] == ':' && tp[3] == 'i' && tp[4] == ':') {
            const uint8_t *qe = tp + 5;
            while (qe < end && *qe != '\t') qe++;
            nm = (int32_t)parse_i64(tp + 5, qe);
            break;
        }
        while (tp < end && *tp != '\t') tp++;
        tp++;
    }
    if (nm >= 0 && rspan > 0 && (double)(nm - indel) > acc->max_mm_frac * (double)rspan) {
        sh->stats[5]++;
        sh->prev_spot = spot; sh->prev_ctg = c; sh->prev_s = pos0; sh->prev_e = end0;
        return;
    }

    /* Mate overlap window on contig c */
    int32_t ov_a = 0, ov_b = 0;
    if (paired && same_spot && sh->prev_ctg == c) {
        ov_a = (int32_t)(sh->prev_s > pos0 ? sh->prev_s : pos0);
        ov_b = (int32_t)(sh->prev_e < end0 ? sh->prev_e : end0);
    }
    sh->prev_spot = spot; sh->prev_ctg = c; sh->prev_s = pos0; sh->prev_e = end0;

    /* Informative read capture for active-region local assembly:
     *   - CIGAR indels (indel > 0)
     *   - High-quality soft-clips (>= 5 bp with boundary Q >= 10)
     *   - Clustered Q >= 10 mismatches (<= 45 bp apart or >= 3 Q >= 10 mismatches; skips Q2 error tails) */
    if (!captured && acc->capture_informative) {
        int want_cap = 0;
        if (indel > 0) {
            want_cap = 1;
        } else if (lead_clip >= 5 && lead_clip < qlen && (qual_s[lead_clip] - 33) >= 10) {
            want_cap = 1;
        } else if (trail_clip >= 5 && (qlen - trail_clip - 1) >= 0 && (qual_s[qlen - trail_clip - 1] - 33) >= 10) {
            want_cap = 1;
        } else if (nm >= 2 || nm < 0) {
            const uint8_t *rptr = acc->ref_seq + ctg->ref_off + pos0;
            int32_t prev_mm_t = -1000;
            int hq_mm = 0;
            int32_t m_lim = qlen - lead_clip - trail_clip;
            for (int32_t t = 0; t < m_lim && (pos0 + t) < ctg->length; t++) {
                int32_t qt = lead_clip + t;
                if ((qual_s[qt] - 33) < 10) continue;
                uint8_t b = ENC_LUT[seq_s[qt]];
                if (b < 4 && rptr[t] < 4 && b != rptr[t]) {
                    hq_mm++;
                    if (t - prev_mm_t <= 45 || hq_mm >= 3) {
                        want_cap = 1;
                        break;
                    }
                    prev_mm_t = t;
                }
            }
        }
        if (want_cap) {
            capture_read_ctg(ctg, pos0, rspan, (int16_t)lead_clip, (int16_t)trail_clip,
                             (uint8_t)(mapq > 255 ? 255 : mapq), flag, seq_s, qual_s, qlen);
            captured = 1;
            sh->stats[8]++;
        }
    }

    /* Track uncaptured non-indel/non-clipped spanning depth per 16-bp bin (requires >=12 bp clean M anchor
     * on BOTH sides of [b_lo, b_lo+16) so flanking reads stopping near a 1/1 indel NEVER bleed into uncap_dp) */
    if (!captured && indel == 0 && lead_clip < 5 && trail_clip < 5 && end0 - pos0 >= 40) {
        int32_t eff_nm = nm;
        if (eff_nm < 0) {
            eff_nm = 0;
            int32_t lim_m = rspan < qlen ? rspan : qlen;
            const uint8_t *rptr = acc->ref_seq + ctg->ref_off + pos0;
            for (int32_t t = 0; t < lim_m && (pos0 + t) < ctg->length; t++) {
                uint8_t b = ENC_LUT[seq_s[t]];
                if (b < 4 && rptr[t] < 4 && b != rptr[t]) {
                    if (++eff_nm > 1) break;
                }
            }
        }
        if (eff_nm <= 1) {
            int32_t span_lo = pos0 + 12;
            int32_t span_hi = end0 - 12;
            int32_t b16_lo = (span_lo + 15) >> 4;
            int32_t b16_hi = (span_hi >> 4) - 1;
            for (int32_t b16 = b16_lo; b16 <= b16_hi; b16++) {
                int32_t mid16 = (b16 << 4) + 8;
                if (mid16 >= ov_a && mid16 < ov_b) continue;
                vl_page_t *pg = get_or_alloc_page(ctg, b16 >> 12);
                if (pg && pg->uncap_dp[b16 & 4095] < 65535) pg->uncap_dp[b16 & 4095]++;
            }
        }
    }

    /* Record high-BQ soft-clip breakpoints in clipc (rescues strobealign soft-clipped indels) */
    if (lead_clip >= 5 && lead_clip < qlen && qual_s[lead_clip] - 33 >= acc->min_bq) {
        if (!(pos0 >= ov_a && pos0 < ov_b)) {
            vl_page_t *pg = get_or_alloc_page(ctg, pos0 >> VL_PAGE_BITS);
            if (pg && pg->clipc[pos0 & VL_PAGE_MASK] < 65535) pg->clipc[pos0 & VL_PAGE_MASK]++;
        }
    }
    if (trail_clip >= 5 && qlen - trail_clip - 1 >= 0 && qual_s[qlen - trail_clip - 1] - 33 >= acc->min_bq && end0 > pos0) {
        int32_t bp = end0 - 1;
        if (!(bp >= ov_a && bp < ov_b)) {
            vl_page_t *pg = get_or_alloc_page(ctg, bp >> VL_PAGE_BITS);
            if (pg && pg->clipc[bp & VL_PAGE_MASK] < 65535) pg->clipc[bp & VL_PAGE_MASK]++;
        }
    }

    /* BAQ-style base mask (grow bmask buffer dynamically if qlen exceeds current capacity) */
    int use_mask = (acc->flt_indel_flank > 0 || acc->flt_clip_flank > 0 ||
                    acc->flt_end_trim > 0 || acc->flt_mm_n > 0);
    if (use_mask) {
        if ((size_t)qlen > *bmask_cap_p) {
            size_t ncap = *bmask_cap_p * 2;
            while ((size_t)qlen > ncap) ncap *= 2;
            *bmask_p = (uint8_t *)xrealloc(*bmask_p, ncap);
            *bmask_cap_p = ncap;
        }
        sh->stats[9] += compute_base_mask(
            f_s[5], f_e[5], seq_s, qlen, ctg->ref_off + pos0, acc->ref_seq,
            ctg->ref_off + ctg->length,
            acc->flt_indel_flank, acc->flt_clip_flank, acc->flt_end_trim,
            acc->flt_mm_win, acc->flt_mm_n, *bmask_p, mmbuf
        );
    }

    int32_t ind_lo = acc->flt_end_trim;
    int32_t lc_lim = lead_clip + (lead_clip > 0 ? acc->flt_clip_flank : 0);
    if (lc_lim > ind_lo) ind_lo = lc_lim;

    int32_t ind_hi = qlen - acc->flt_end_trim;
    int32_t tc_lim = qlen - trail_clip - (trail_clip > 0 ? acc->flt_clip_flank : 0);
    if (tc_lim < ind_hi) ind_hi = tc_lim;

    /* Walk CIGAR and accumulate bases & indels */
    int32_t qi = 0;
    int32_t cp = pos0;
    int32_t counted = 0;
    num = 0;
    vl_page_t *cur_pg = NULL;
    int32_t cur_pidx = -1;

    for (const uint8_t *p = f_s[5]; p < f_e[5]; p++) {
        uint8_t ch = *p;
        if (ch >= '0' && ch <= '9') { num = num * 10 + (ch - '0'); continue; }
        if (ch == 'M' || ch == '=' || ch == 'X') {
            for (int32_t t = 0; t < num; t++) {
                int32_t pos = cp + t;
                if (pos >= ctg->length) break;
                if (pos >= ov_a && pos < ov_b) { sh->stats[7]++; continue; }
                if (use_mask && (*bmask_p)[qi + t]) continue;
                uint8_t b = ENC_LUT[seq_s[qi + t]];
                if (b < 4 && (qual_s[qi + t] - 33) >= acc->min_bq) {
                    int32_t pidx = pos >> VL_PAGE_BITS;
                    if (pidx != cur_pidx) { cur_pidx = pidx; cur_pg = get_or_alloc_page(ctg, pidx); }
                    if (cur_pg) {
                        uint16_t *cell = &cur_pg->counts[pos & VL_PAGE_MASK][b];
                        if (*cell < 65535) (*cell)++;
                        counted++;
                    }
                }
            }
            qi += num;
            cp += num;
        } else if (ch == 'I') {
            int32_t ind_min_bq = acc->min_bq < 10 ? acc->min_bq : 10;
            int bq_ok = (qi >= ind_lo && (qi + num) <= ind_hi && num > 0 && num < 16384);
            if (bq_ok) {
                for (int32_t t = 0; t < num; t++) {
                    if (ENC_LUT[seq_s[qi + t]] > 3 || (qual_s[qi + t] - 33) < ind_min_bq) {
                        bq_ok = 0;
                        break;
                    }
                }
            }
            if (bq_ok && cp > 0 && !((cp - 1) >= ov_a && (cp - 1) < ov_b)) {
                int32_t anchor = cp - 1;
                if (ctg->page_allowed == NULL || ctg->page_allowed[anchor >> VL_PAGE_BITS]) {
                    uint64_t code = 0ULL;
                    if (num <= 31) {
                        for (int32_t t = 0; t < num; t++) code = (code << 2) | (uint64_t)(ENC_LUT[seq_s[qi + t]] & 3);
                    } else {
                        code = fnv1a_bytes(seq_s + qi, (size_t)num);
                    }
                    ind_tab_add(sh, c, anchor, 2, (int16_t)num, code, seq_s + qi);
                }
            }
            qi += num;
        } else if (ch == 'D') {
            int32_t ind_min_bq = acc->min_bq < 10 ? acc->min_bq : 10;
            int bq_ok = (qi >= ind_lo && qi <= ind_hi && num > 0 && num < 16384);
            if (bq_ok) {
                if (qi > 0 && (qual_s[qi - 1] - 33) < ind_min_bq) bq_ok = 0;
                if (qi < qlen && (qual_s[qi] - 33) < ind_min_bq) bq_ok = 0;
            }
            if (bq_ok && !(cp >= ov_a && cp < ov_b) && cp < ctg->length) {
                for (int32_t t = 0; t < num && (cp + t) < ctg->length; t++) {
                    int32_t dpos = cp + t;
                    int32_t pidx = dpos >> VL_PAGE_BITS;
                    if (pidx != cur_pidx) { cur_pidx = pidx; cur_pg = get_or_alloc_page(ctg, pidx); }
                    if (cur_pg && cur_pg->delc[dpos & VL_PAGE_MASK] < 65535) {
                        cur_pg->delc[dpos & VL_PAGE_MASK]++;
                    }
                }
                if (ctg->page_allowed == NULL || ctg->page_allowed[cp >> VL_PAGE_BITS]) {
                    ind_tab_add(sh, c, cp, 1, (int16_t)num, 0ULL, NULL);
                }
            }
            cp += num;
        } else if (ch == 'N') {
            cp += num;
        } else if (ch == 'S') {
            qi += num;
        } else if (ch == 'H' || ch == 'P') {
            /* non-SEQ-consuming */
        }
        num = 0;
    }

    sh->stats[2]++;
    sh->stats[3] += counted;
}

static sam_batch_t *batch_new(size_t cap) {
    sam_batch_t *b = (sam_batch_t *)xmalloc(sizeof(sam_batch_t));
    b->data = (uint8_t *)xmalloc(cap);
    b->len = 0;
    b->cap = cap;
    return b;
}

static void batch_free(sam_batch_t *b) {
    if (!b) return;
    free(b->data);
    free(b);
}

static void shard_enqueue(vl_shard_t *sh, sam_batch_t *b) {
    pthread_mutex_lock(&sh->mu);
    while (sh->q_count == VL_QUEUE_CAP) {
        pthread_cond_wait(&sh->cv_nonfull, &sh->mu);
    }
    sh->q[sh->q_tail] = b;
    sh->q_tail = (sh->q_tail + 1) % VL_QUEUE_CAP;
    sh->q_count++;
    pthread_cond_signal(&sh->cv_nonempty);
    pthread_mutex_unlock(&sh->mu);
}

static void *shard_worker_main(void *arg) {
    vl_shard_t *sh = (vl_shard_t *)arg;
    size_t bmask_cap = 4096;
    uint8_t *bmask = (uint8_t *)xmalloc(bmask_cap);
    int32_t *mmbuf = (int32_t *)xmalloc(512 * sizeof(int32_t));
    for (;;) {
        pthread_mutex_lock(&sh->mu);
        while (sh->q_count == 0 && !sh->done) {
            pthread_cond_wait(&sh->cv_nonempty, &sh->mu);
        }
        if (sh->q_count == 0 && sh->done) {
            pthread_mutex_unlock(&sh->mu);
            break;
        }
        sam_batch_t *b = sh->q[sh->q_head];
        sh->q_head = (sh->q_head + 1) % VL_QUEUE_CAP;
        sh->q_count--;
        pthread_cond_signal(&sh->cv_nonfull);
        pthread_mutex_unlock(&sh->mu);

        size_t i = 0, n = b->len;
        while (i < n) {
            size_t e = i;
            while (e < n && b->data[e] != '\n') e++;
            shard_process_line(sh, b->data + i, e - i, &bmask, &bmask_cap, mmbuf);
            i = e + 1;
        }
        batch_free(b);
    }
    free(bmask);
    free(mmbuf);
    return NULL;
}

vl_acc_t *vl_acc_create(
    int32_t n_contigs,
    const char *names_flat,
    const int64_t *name_offs,
    const int64_t *lengths,
    const int64_t *ref_offs,
    const int64_t *bin_offs,
    const uint8_t *ref_seq,
    int32_t n_shards,
    int32_t min_mapq,
    int32_t min_bq,
    int32_t bin_mapq,
    int32_t zoom_min_mapq,
    double max_mm_frac,
    int32_t flt_indel_flank,
    int32_t flt_clip_flank,
    int32_t flt_end_trim,
    int32_t flt_mm_win,
    int32_t flt_mm_n,
    int32_t capture_informative,
    int32_t dedup_bits
) {
    if (n_contigs <= 0 || n_shards <= 0) return NULL;
    vl_acc_t *acc = (vl_acc_t *)xcalloc(1, sizeof(vl_acc_t));
    acc->n_contigs = n_contigs;
    acc->n_shards = n_shards;
    acc->ref_seq = ref_seq;
    acc->total_bins = bin_offs[n_contigs];
    acc->binbases = (int64_t *)xcalloc((size_t)acc->total_bins, sizeof(int64_t));

    acc->min_mapq = min_mapq;
    acc->min_bq = min_bq;
    acc->bin_mapq = bin_mapq;
    acc->zoom_min_mapq = zoom_min_mapq;
    acc->max_mm_frac = max_mm_frac;
    acc->flt_indel_flank = flt_indel_flank;
    acc->flt_clip_flank = flt_clip_flank;
    acc->flt_end_trim = flt_end_trim;
    acc->flt_mm_win = flt_mm_win;
    acc->flt_mm_n = flt_mm_n;
    acc->capture_informative = capture_informative;
    acc->dedup_bits = dedup_bits;

    int64_t tot_name_bytes = name_offs[n_contigs];
    acc->names_blob = (char *)xmalloc((size_t)tot_name_bytes + 1);
    memcpy(acc->names_blob, names_flat, (size_t)tot_name_bytes);

    /* Build O(1) contig name hash table */
    uint64_t ht_sz = 16;
    while (ht_sz < (uint64_t)n_contigs * 4ULL) ht_sz <<= 1;
    acc->ctg_ht_mask = ht_sz - 1ULL;
    acc->ctg_ht = (ctg_ht_slot_t *)xmalloc((size_t)ht_sz * sizeof(ctg_ht_slot_t));
    for (uint64_t i = 0; i < ht_sz; i++) acc->ctg_ht[i].ctg_id = -1;

    acc->contigs = (vl_contig_t *)xcalloc((size_t)n_contigs, sizeof(vl_contig_t));

    /* LPT load-balancing of contigs across n_shards */
    int64_t *shard_load = (int64_t *)xcalloc((size_t)n_shards, sizeof(int64_t));
    int32_t *order = (int32_t *)xmalloc((size_t)n_contigs * sizeof(int32_t));
    for (int32_t c = 0; c < n_contigs; c++) order[c] = c;
    /* Simple insertion sort by descending contig length (n_contigs ~ 195) */
    for (int32_t i = 1; i < n_contigs; i++) {
        int32_t key = order[i];
        int32_t j = i - 1;
        while (j >= 0 && lengths[order[j]] < lengths[key]) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    for (int32_t idx = 0; idx < n_contigs; idx++) {
        int32_t c = order[idx];
        int best_s = 0;
        for (int s = 1; s < n_shards; s++) {
            if (shard_load[s] < shard_load[best_s]) best_s = s;
        }
        shard_load[best_s] += lengths[c];

        vl_contig_t *ctg = &acc->contigs[c];
        ctg->id = c;
        ctg->length = lengths[c];
        ctg->ref_off = ref_offs[c];
        ctg->bin_off = bin_offs[c];
        ctg->n_pages = (int32_t)((lengths[c] + VL_PAGE_BP - 1) >> VL_PAGE_BITS);
        ctg->owner_shard = best_s;
        ctg->pages = (vl_page_t **)xcalloc((size_t)ctg->n_pages, sizeof(vl_page_t *));

        const char *np = acc->names_blob + name_offs[c];
        int32_t nlen = (int32_t)(name_offs[c + 1] - name_offs[c]);
        uint64_t h = fnv1a_bytes((const uint8_t *)np, (size_t)nlen);
        uint64_t sl = mix64(h) & acc->ctg_ht_mask;
        while (acc->ctg_ht[sl].ctg_id >= 0) sl = (sl + 1ULL) & acc->ctg_ht_mask;
        acc->ctg_ht[sl].hash = h;
        acc->ctg_ht[sl].ctg_id = c;
        acc->ctg_ht[sl].name_len = nlen;
        acc->ctg_ht[sl].name_ptr = np;
    }
    free(order);
    free(shard_load);

    acc->shards = (vl_shard_t *)xcalloc((size_t)n_shards, sizeof(vl_shard_t));
    for (int s = 0; s < n_shards; s++) {
        vl_shard_t *sh = &acc->shards[s];
        sh->shard_id = s;
        sh->acc = acc;
        pthread_mutex_init(&sh->mu, NULL);
        pthread_cond_init(&sh->cv_nonempty, NULL);
        pthread_cond_init(&sh->cv_nonfull, NULL);
        sh->prev_spot = -1;
        sh->prev_ctg = -1;
        sh->ccount = (int64_t *)xcalloc((size_t)n_contigs, sizeof(int64_t));
        if (dedup_bits > 0) {
            size_t dsz = 1ULL << dedup_bits;
            sh->dedup_tab = (uint64_t *)xcalloc(dsz, sizeof(uint64_t));
            sh->dedup_mask = dsz - 1ULL;
        }
        ind_tab_init(sh, 1U << 16);
    }
    return acc;
}

void vl_acc_restrict_regions(vl_acc_t *acc, const int32_t *ctg_ids, const int32_t *starts0,
                             const int32_t *ends0, int32_t n_regs) {
    /* Restrict page allocation strictly to pages overlapping the given regions */
    for (int32_t c = 0; c < acc->n_contigs; c++) {
        vl_contig_t *ctg = &acc->contigs[c];
        if (!ctg->page_allowed) {
            ctg->page_allowed = (uint8_t *)xcalloc((size_t)ctg->n_pages, 1);
        }
    }
    for (int32_t i = 0; i < n_regs; i++) {
        int32_t c = ctg_ids[i];
        if (c < 0 || c >= acc->n_contigs) continue;
        vl_contig_t *ctg = &acc->contigs[c];
        int32_t s = starts0[i] < 0 ? 0 : starts0[i];
        int32_t e = ends0[i] > ctg->length ? (int32_t)ctg->length : ends0[i];
        if (e <= s) continue;
        int32_t p0 = s >> VL_PAGE_BITS;
        int32_t p1 = (e - 1) >> VL_PAGE_BITS;
        for (int32_t p = p0; p <= p1 && p < ctg->n_pages; p++) ctg->page_allowed[p] = 1;
    }
}

void vl_acc_set_zoom_intervals(vl_acc_t *acc, int32_t ctg_id, const int32_t *starts0,
                               const int32_t *ends0, int32_t n_iv) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return;
    vl_contig_t *ctg = &acc->contigs[ctg_id];
    free(ctg->zoom_s);
    free(ctg->zoom_e);
    ctg->zoom_s = NULL;
    ctg->zoom_e = NULL;
    ctg->n_zoom = 0;
    if (n_iv <= 0) return;
    ctg->zoom_s = (int32_t *)xmalloc((size_t)n_iv * sizeof(int32_t));
    ctg->zoom_e = (int32_t *)xmalloc((size_t)n_iv * sizeof(int32_t));
    memcpy(ctg->zoom_s, starts0, (size_t)n_iv * sizeof(int32_t));
    memcpy(ctg->zoom_e, ends0, (size_t)n_iv * sizeof(int32_t));
    ctg->n_zoom = n_iv;
}

static inline void route_line_to_shard(vl_acc_t *acc, const uint8_t *line, size_t len, size_t flush_cap) {
    if (len == 0 || line[0] == '@') return;
    /* Find 3rd tab-separated field (RNAME) */
    size_t i = 0;
    int tabs = 0;
    size_t r_s = 0, r_e = 0;
    for (; i < len; i++) {
        if (line[i] == '\t') {
            tabs++;
            if (tabs == 2) r_s = i + 1;
            else if (tabs == 3) { r_e = i; break; }
        }
    }
    if (tabs < 3 || r_e <= r_s || (r_e - r_s == 1 && line[r_s] == '*')) return;
    int32_t c = lookup_contig(acc, line + r_s, (int32_t)(r_e - r_s));
    if (c < 0) return;
    int s = acc->contigs[c].owner_shard;
    vl_shard_t *sh = &acc->shards[s];
    if (!sh->staging) sh->staging = batch_new(flush_cap + 65536);
    if (sh->staging->len + len + 1 > sh->staging->cap) {
        size_t ncap = (sh->staging->len + len + 1) * 2;
        sh->staging->data = (uint8_t *)xrealloc(sh->staging->data, ncap);
        sh->staging->cap = ncap;
    }
    memcpy(sh->staging->data + sh->staging->len, line, len);
    sh->staging->data[sh->staging->len + len] = '\n';
    sh->staging->len += len + 1;
    if (sh->staging->len >= flush_cap) {
        shard_enqueue(sh, sh->staging);
        sh->staging = batch_new(flush_cap + 65536);
    }
}

int vl_acc_consume_sam_fd(vl_acc_t *acc, int fd_sam) {
    const size_t READ_CAP = 8U << 20;
    const size_t BATCH_FLUSH = 1U << 20;
    uint8_t *buf = (uint8_t *)xmalloc(READ_CAP);

    for (int s = 0; s < acc->n_shards; s++) {
        vl_shard_t *sh = &acc->shards[s];
        sh->done = 0;
        sh->q_head = sh->q_tail = sh->q_count = 0;
        sh->staging = batch_new(BATCH_FLUSH + 65536);
        if (pthread_create(&sh->thread, NULL, shard_worker_main, sh) != 0) {
            fprintf(stderr, "[VarLoop FATAL] pthread_create failed\n");
            abort();
        }
    }

    size_t carry = 0;
    int rc = 0;
    for (;;) {
        ssize_t n = read(fd_sam, buf + carry, READ_CAP - carry);
        if (n < 0) {
            if (errno == EINTR) continue;
            rc = -2;
            break;
        }
        if (n == 0) break;
        size_t tot = carry + (size_t)n;
        size_t i = 0;
        for (;;) {
            uint8_t *nl = (uint8_t *)memchr(buf + i, '\n', tot - i);
            if (!nl) break;
            size_t l_len = (size_t)(nl - (buf + i));
            route_line_to_shard(acc, buf + i, l_len, BATCH_FLUSH);
            i = (size_t)(nl - buf) + 1;
        }
        carry = tot - i;
        if (carry > 0) {
            if (carry == READ_CAP) {
                /* Pathological >8 MB single line: drop safely */
                carry = 0;
            } else {
                memmove(buf, buf + i, carry);
            }
        }
    }
    if (carry > 0 && rc == 0) {
        route_line_to_shard(acc, buf, carry, BATCH_FLUSH);
    }

    for (int s = 0; s < acc->n_shards; s++) {
        vl_shard_t *sh = &acc->shards[s];
        if (sh->staging && sh->staging->len > 0) {
            shard_enqueue(sh, sh->staging);
            sh->staging = NULL;
        } else {
            batch_free(sh->staging);
            sh->staging = NULL;
        }
        pthread_mutex_lock(&sh->mu);
        sh->done = 1;
        pthread_cond_signal(&sh->cv_nonempty);
        pthread_mutex_unlock(&sh->mu);
    }
    for (int s = 0; s < acc->n_shards; s++) {
        pthread_join(acc->shards[s].thread, NULL);
    }
    free(buf);
    return rc;
}

int vl_acc_feed_sam_bytes(vl_acc_t *acc, const uint8_t *buf, size_t n) {
    /* Synchronous helper for unit tests / in-memory buffers (returns bytes consumed up to last '\n') */
    size_t bmask_cap = 4096;
    uint8_t *bmask = (uint8_t *)xmalloc(bmask_cap);
    int32_t *mmbuf = (int32_t *)xmalloc(512 * sizeof(int32_t));
    size_t i = 0;
    for (;;) {
        const uint8_t *nl = (const uint8_t *)memchr(buf + i, '\n', n - i);
        if (!nl) break;
        size_t l_len = (size_t)(nl - (buf + i));
        if (l_len > 0 && buf[i] != '@') {
            /* Determine contig owner shard */
            int tabs = 0;
            size_t r_s = 0, r_e = 0;
            for (size_t k = 0; k < l_len; k++) {
                if (buf[i + k] == '\t') {
                    tabs++;
                    if (tabs == 2) r_s = k + 1;
                    else if (tabs == 3) { r_e = k; break; }
                }
            }
            if (tabs >= 3 && r_e > r_s) {
                int32_t c = lookup_contig(acc, buf + i + r_s, (int32_t)(r_e - r_s));
                if (c >= 0) {
                    int s = acc->contigs[c].owner_shard;
                    shard_process_line(&acc->shards[s], buf + i, l_len, &bmask, &bmask_cap, mmbuf);
                }
            }
        }
        i = (size_t)(nl - buf) + 1;
    }
    free(bmask);
    free(mmbuf);
    return (int)i;
}

void vl_acc_get_summary(const vl_acc_t *acc, int64_t *out_stats, int64_t *out_ccount,
                        int64_t *out_binbases, int64_t *out_mem_bytes) {
    memset(out_stats, 0, 12 * sizeof(int64_t));
    memset(out_ccount, 0, (size_t)acc->n_contigs * sizeof(int64_t));
    for (int s = 0; s < acc->n_shards; s++) {
        const vl_shard_t *sh = &acc->shards[s];
        for (int k = 0; k < 12; k++) {
            if (k == 4) {
                if (sh->stats[4] > out_stats[4]) out_stats[4] = sh->stats[4];
            } else {
                out_stats[k] += sh->stats[k];
            }
        }
        for (int32_t c = 0; c < acc->n_contigs; c++) out_ccount[c] += sh->ccount[c];
    }
    if (out_binbases) memcpy(out_binbases, acc->binbases, (size_t)acc->total_bins * sizeof(int64_t));

    int64_t page_bytes = 0, read_bytes = 0, ind_bytes = 0;
    for (int32_t c = 0; c < acc->n_contigs; c++) {
        page_bytes += acc->contigs[c].n_pages_alloc * (int64_t)sizeof(vl_page_t);
        read_bytes += (int64_t)(acc->contigs[c].n_reads * sizeof(vl_cap_read_t) + acc->contigs[c].seq_pool_used);
    }
    for (int s = 0; s < acc->n_shards; s++) {
        ind_bytes += (int64_t)(acc->shards[s].ind_cap * sizeof(vl_indel_slot_t) + acc->shards[s].ins_pool_used);
    }
    if (out_mem_bytes) {
        out_mem_bytes[0] = page_bytes;
        out_mem_bytes[1] = read_bytes;
        out_mem_bytes[2] = ind_bytes;
    }
}

/* Count candidates on contig c so Python can allocate exact output buffers */
int64_t vl_acc_count_contig_candidates(const vl_acc_t *acc, int32_t ctg_id, int32_t min_alt, int32_t min_clip) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return 0;
    const vl_contig_t *ctg = &acc->contigs[ctg_id];
    const uint8_t *rseq = acc->ref_seq + ctg->ref_off;
    int64_t n_cand = 0;
    for (int32_t p = 0; p < ctg->n_pages; p++) {
        const vl_page_t *pg = ctg->pages[p];
        if (!pg) continue;
        int32_t base_pos = p << VL_PAGE_BITS;
        int32_t lim = (int32_t)(ctg->length - base_pos);
        if (lim > VL_PAGE_BP) lim = VL_PAGE_BP;
        for (int32_t off = 0; off < lim; off++) {
            uint8_t rb = rseq[base_pos + off];
            if (rb > 3) continue;
            const uint16_t *cnt = pg->counts[off];
            int32_t tot = (int32_t)cnt[0] + (int32_t)cnt[1] + (int32_t)cnt[2] + (int32_t)cnt[3];
            int32_t nonref = tot - (int32_t)cnt[rb];
            if (nonref >= min_alt || (min_clip > 0 && pg->clipc[off] >= min_clip)) {
                n_cand++;
            }
        }
    }
    return n_cand;
}

int64_t vl_acc_extract_contig_candidates(
    const vl_acc_t *acc,
    int32_t ctg_id,
    int32_t min_alt,
    int32_t min_clip,
    int32_t *out_pos0,
    uint16_t *out_counts4,
    uint16_t *out_delc,
    uint16_t *out_clipc,
    uint16_t *out_uncap,
    uint8_t *out_ref
) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return 0;
    const vl_contig_t *ctg = &acc->contigs[ctg_id];
    const uint8_t *rseq = acc->ref_seq + ctg->ref_off;
    int64_t k = 0;
    for (int32_t p = 0; p < ctg->n_pages; p++) {
        const vl_page_t *pg = ctg->pages[p];
        if (!pg) continue;
        int32_t base_pos = p << VL_PAGE_BITS;
        int32_t lim = (int32_t)(ctg->length - base_pos);
        if (lim > VL_PAGE_BP) lim = VL_PAGE_BP;
        for (int32_t off = 0; off < lim; off++) {
            uint8_t rb = rseq[base_pos + off];
            if (rb > 3) continue;
            const uint16_t *cnt = pg->counts[off];
            int32_t tot = (int32_t)cnt[0] + (int32_t)cnt[1] + (int32_t)cnt[2] + (int32_t)cnt[3];
            int32_t nonref = tot - (int32_t)cnt[rb];
            if (nonref >= min_alt || (min_clip > 0 && pg->clipc[off] >= min_clip)) {
                out_pos0[k] = base_pos + off;
                out_counts4[4 * k + 0] = cnt[0];
                out_counts4[4 * k + 1] = cnt[1];
                out_counts4[4 * k + 2] = cnt[2];
                out_counts4[4 * k + 3] = cnt[3];
                out_delc[k] = pg->delc[off];
                out_clipc[k] = pg->clipc[off];
                {
                    int32_t ru = (int32_t)pg->uncap_dp[off >> 4];
                    int32_t su = ru > 0 ? (ru * 151 + 56) / 112 : 0;
                    out_uncap[k] = su > 65535 ? 65535 : (uint16_t)su;
                }
                out_ref[k] = rb;
                k++;
            }
        }
    }
    return k;
}

void vl_acc_query_positions(
    const vl_acc_t *acc,
    int32_t ctg_id,
    const int32_t *pos0,
    int64_t n,
    int32_t *out_dp,
    int32_t *out_delc,
    int32_t *out_uncap
) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return;
    const vl_contig_t *ctg = &acc->contigs[ctg_id];
    for (int64_t i = 0; i < n; i++) {
        int32_t p0 = pos0[i];
        if (p0 < 0 || p0 >= ctg->length) {
            out_dp[i] = 0; out_delc[i] = 0; out_uncap[i] = 0;
            continue;
        }
        const vl_page_t *pg = ctg->pages[p0 >> VL_PAGE_BITS];
        if (!pg) {
            out_dp[i] = 0; out_delc[i] = 0; out_uncap[i] = 0;
            continue;
        }
        int off = p0 & VL_PAGE_MASK;
        const uint16_t *cnt = pg->counts[off];
        int32_t dc = (int32_t)pg->delc[off];
        out_dp[i] = (int32_t)cnt[0] + (int32_t)cnt[1] + (int32_t)cnt[2] + (int32_t)cnt[3] + dc;
        out_delc[i] = dc;
        {
            int32_t ru = (int32_t)pg->uncap_dp[off >> 4];
            out_uncap[i] = ru > 0 ? (ru * 151 + 56) / 112 : 0;
        }
    }
}

int64_t vl_acc_count_contig_indels(const vl_acc_t *acc, int32_t ctg_id, int32_t min_cnt, int64_t *out_long_bytes) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return 0;
    const vl_shard_t *sh = &acc->shards[acc->contigs[ctg_id].owner_shard];
    int64_t n = 0, lb = 0;
    for (size_t i = 0; i < sh->ind_cap; i++) {
        const vl_indel_slot_t *sl = &sh->ind_tab[i];
        if (sl->ctg == ctg_id && (int32_t)sl->count >= min_cnt) {
            n++;
            if (sl->kind == 2 && sl->len > 31) lb += sl->len + 1;
        }
    }
    if (out_long_bytes) *out_long_bytes = lb;
    return n;
}

int64_t vl_acc_extract_contig_indels(
    const vl_acc_t *acc,
    int32_t ctg_id,
    int32_t min_cnt,
    int32_t *out_pos0,
    int16_t *out_kind,
    int16_t *out_len,
    int32_t *out_cnt,
    uint64_t *out_code,
    char *out_long_seqs,
    int32_t *out_long_offs
) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return 0;
    const vl_shard_t *sh = &acc->shards[acc->contigs[ctg_id].owner_shard];
    int64_t k = 0;
    int32_t loff = 0;
    for (size_t i = 0; i < sh->ind_cap; i++) {
        const vl_indel_slot_t *sl = &sh->ind_tab[i];
        if (sl->ctg == ctg_id && (int32_t)sl->count >= min_cnt) {
            out_pos0[k] = sl->pos0;
            out_kind[k] = sl->kind;
            out_len[k] = sl->len;
            out_cnt[k] = (int32_t)sl->count;
            if (sl->kind == 1 || sl->len <= 31) {
                out_code[k] = sl->code;
                out_long_offs[k] = -1;
            } else {
                out_code[k] = sl->code & 0xFFFFFFFFULL;
                uint32_t poff = (uint32_t)(sl->code >> 32) - 1U;
                memcpy(out_long_seqs + loff, sh->ins_pool + poff, (size_t)sl->len + 1U);
                out_long_offs[k] = loff;
                loff += sl->len + 1;
            }
            k++;
        }
    }
    return k;
}

static int cmp_cap_read(const void *a, const void *b) {
    const vl_cap_read_t *ra = (const vl_cap_read_t *)a;
    const vl_cap_read_t *rb = (const vl_cap_read_t *)b;
    if (ra->pos0 != rb->pos0) return ra->pos0 < rb->pos0 ? -1 : 1;
    return 0;
}

void vl_acc_sort_contig_reads(vl_acc_t *acc, int32_t ctg_id) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return;
    vl_contig_t *ctg = &acc->contigs[ctg_id];
    if (!ctg->reads_sorted) {
        if (ctg->n_reads > 1) {
            qsort(ctg->reads, ctg->n_reads, sizeof(vl_cap_read_t), cmp_cap_read);
        }
        ctg->reads_sorted = 1;
    }
}

void vl_acc_sort_all_reads(vl_acc_t *acc) {
    if (!acc) return;
    for (int32_t c = 0; c < acc->n_contigs; c++) {
        vl_acc_sort_contig_reads(acc, c);
    }
}

void vl_acc_get_contig_reads_info(const vl_acc_t *acc, int32_t ctg_id, int64_t *out_n_reads, int64_t *out_pool_bytes) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) {
        *out_n_reads = 0; *out_pool_bytes = 0;
        return;
    }
    const vl_contig_t *ctg = &acc->contigs[ctg_id];
    if ((uint64_t)ctg->seq_pool_used > (uint64_t)INT64_MAX) {
        fprintf(stderr, "[VarLoop FATAL] seq_pool_used overflow in get_contig_reads_info\n");
        abort();
    }
    *out_n_reads = (int64_t)ctg->n_reads;
    *out_pool_bytes = (int64_t)ctg->seq_pool_used;
}

void vl_acc_copy_contig_reads(
    const vl_acc_t *acc,
    int32_t ctg_id,
    int64_t *out_meta6,   /* (n_reads, 6) int64: pos0, rspan, lead_clip, trail_clip, len, seq_off */
    uint8_t *out_mapq,
    uint8_t *out_pool
) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) return;
    const vl_contig_t *ctg = &acc->contigs[ctg_id];
    for (size_t i = 0; i < ctg->n_reads; i++) {
        const vl_cap_read_t *r = &ctg->reads[i];
        if (r->seq_off > (uint64_t)INT64_MAX) {
            fprintf(stderr, "[VarLoop FATAL] seq_off exceeded INT64_MAX\n");
            abort();
        }
        out_meta6[6 * i + 0] = (int64_t)r->pos0;
        out_meta6[6 * i + 1] = (int64_t)r->rspan;
        out_meta6[6 * i + 2] = (int64_t)r->lead_clip;
        out_meta6[6 * i + 3] = (int64_t)r->trail_clip;
        out_meta6[6 * i + 4] = (int64_t)r->len;
        out_meta6[6 * i + 5] = (int64_t)r->seq_off;
        out_mapq[i] = r->mapq;
    }
    if (ctg->seq_pool_used > 0 && out_pool != NULL) {
        memcpy(out_pool, ctg->seq_pool, ctg->seq_pool_used);
    }
}

/* Zero-copy read pool accessor: populates int64 meta6 + mapq and returns direct pointer to ctg->seq_pool */
void vl_acc_get_contig_reads_view(
    vl_acc_t *acc,
    int32_t ctg_id,
    int64_t *out_meta6,
    uint8_t *out_mapq,
    uint64_t *out_pool_addr,
    int64_t *out_pool_bytes
) {
    if (ctg_id < 0 || ctg_id >= acc->n_contigs) {
        if (out_pool_addr) *out_pool_addr = 0ULL;
        if (out_pool_bytes) *out_pool_bytes = 0;
        return;
    }
    vl_acc_sort_contig_reads(acc, ctg_id);
    const vl_contig_t *ctg = &acc->contigs[ctg_id];
    vl_acc_copy_contig_reads(acc, ctg_id, out_meta6, out_mapq, NULL);
    if (out_pool_addr) *out_pool_addr = (uint64_t)(uintptr_t)ctg->seq_pool;
    if (out_pool_bytes) *out_pool_bytes = (int64_t)ctg->seq_pool_used;
}

/* Unit-test hook for B1: advance seq_pool_used past 2^31 / 2^32 using sparse virtual memory */
void vl_acc_test_set_seq_pool_offset(vl_acc_t *acc, int32_t ctg_id, uint64_t offset_bytes) {
    if (!acc || ctg_id < 0 || ctg_id >= acc->n_contigs) return;
    vl_contig_t *ctg = &acc->contigs[ctg_id];
    size_t need = (size_t)offset_bytes + (1U << 20);
    if (need > ctg->seq_pool_cap) {
        ctg->seq_pool = (uint8_t *)xrealloc(ctg->seq_pool, need);
        ctg->seq_pool_cap = need;
    }
    ctg->seq_pool_used = (size_t)offset_bytes;
}

void vl_acc_free_contig(vl_acc_t *acc, int32_t ctg_id) {
    /* Release a completed contig's 64-kb pages and read buffers immediately after calling */
    if (!acc || ctg_id < 0 || ctg_id >= acc->n_contigs) return;
    vl_contig_t *ctg = &acc->contigs[ctg_id];
    if (ctg->pages) {
        for (int32_t p = 0; p < ctg->n_pages; p++) {
            free(ctg->pages[p]);
            ctg->pages[p] = NULL;
        }
        ctg->n_pages_alloc = 0;
    }
    free(ctg->reads); ctg->reads = NULL; ctg->n_reads = ctg->cap_reads = 0;
    free(ctg->seq_pool); ctg->seq_pool = NULL; ctg->seq_pool_used = ctg->seq_pool_cap = 0;
}

void vl_acc_destroy(vl_acc_t *acc) {
    if (!acc) return;
    for (int32_t c = 0; c < acc->n_contigs; c++) {
        vl_acc_free_contig(acc, c);
        free(acc->contigs[c].pages);
        free(acc->contigs[c].page_allowed);
        free(acc->contigs[c].zoom_s);
        free(acc->contigs[c].zoom_e);
    }
    free(acc->contigs);
    for (int s = 0; s < acc->n_shards; s++) {
        vl_shard_t *sh = &acc->shards[s];
        pthread_mutex_destroy(&sh->mu);
        pthread_cond_destroy(&sh->cv_nonempty);
        pthread_cond_destroy(&sh->cv_nonfull);
        free(sh->dedup_tab);
        free(sh->ccount);
        free(sh->ind_tab);
        free(sh->ins_pool);
    }
    free(acc->shards);
    free(acc->binbases);
    free(acc->names_blob);
    free(acc->ctg_ht);
    free(acc);
}

/* ============================================================================
 * 4. Semi-Global Affine-Gap Phred Read-vs-Haplotype Log-Likelihood Scorer
 * ============================================================================ */

static double align_read_hap_semiglobal(
    const uint8_t *rseq,
    const uint8_t *rqual,
    int32_t rlen,
    const uint8_t *hseq,
    int32_t hlen,
    double *M_row,
    double *I_row,
    double *D_row
) {
    /* Log-likelihood (natural log) of read given haplotype under semi-global alignment:
     *   - Free leading and trailing haplotype positions (read aligns anywhere inside padded hap)
     *   - Match: log(1 - eps_q) ~ 0; Mismatch: log(eps_q / 3)
     *   - Gap open: -4.60 nats (~Q20); Gap extend: -1.15 nats (~Q5)
     */
    const double NEG_INF = -1e18;
    const double GAP_OPEN = -4.605170186;   /* ln(0.01) */
    const double GAP_EXT  = -1.151292546;   /* ln(0.316) */

    for (int32_t j = 0; j <= hlen; j++) {
        M_row[j] = 0.0;      /* free start on haplotype */
        I_row[j] = NEG_INF;
        D_row[j] = NEG_INF;
    }

    for (int32_t i = 1; i <= rlen; i++) {
        uint8_t rb = ENC_LUT[rseq[i - 1]];
        int q = (int)rqual[i - 1] - 33;
        if (q < 2) q = 2;
        if (q > 42) q = 42;
        double mm_pen = -0.2302585093 * (double)q - 1.098612289; /* -q/10*ln(10) - ln(3) */

        double prev_M_diag = M_row[0];
        double prev_I_diag = I_row[0];
        double prev_D_diag = D_row[0];

        /* Leading read bases before haplotype start cost an insertion */
        M_row[0] = NEG_INF;
        I_row[0] = (i == 1) ? GAP_OPEN : (I_row[0] + GAP_EXT);
        D_row[0] = NEG_INF;

        for (int32_t j = 1; j <= hlen; j++) {
            double old_M = M_row[j];
            double old_I = I_row[j];
            double old_D = D_row[j];

            /* Best state at (i-1, j-1) */
            double best_diag = prev_M_diag;
            if (prev_I_diag > best_diag) best_diag = prev_I_diag;
            if (prev_D_diag > best_diag) best_diag = prev_D_diag;

            uint8_t hb = ENC_LUT[hseq[j - 1]];
            double sub = (rb == hb || rb > 3 || hb > 3) ? -0.001 : mm_pen;
            double m_val = best_diag + sub;

            /* Insertion in read: from (i-1, j) */
            double i_from_m = old_M + GAP_OPEN;
            double i_from_i = old_I + GAP_EXT;
            double i_val = i_from_m > i_from_i ? i_from_m : i_from_i;

            /* Deletion from hap: from (i, j-1) */
            double d_from_m = M_row[j - 1] + GAP_OPEN;
            double d_from_d = D_row[j - 1] + GAP_EXT;
            double d_val = d_from_m > d_from_d ? d_from_m : d_from_d;

            M_row[j] = m_val;
            I_row[j] = i_val;
            D_row[j] = d_val;

            prev_M_diag = old_M;
            prev_I_diag = old_I;
            prev_D_diag = old_D;
        }
    }

    double best = NEG_INF;
    for (int32_t j = 0; j <= hlen; j++) {
        if (M_row[j] > best) best = M_row[j];
        if (I_row[j] > best) best = I_row[j];
        if (D_row[j] > best) best = D_row[j];
    }
    return best;
}

void vl_score_reads_haplotypes(
    const uint8_t *pool,
    const int64_t *read_offs,
    const int32_t *read_lens,
    int32_t n_reads,
    const char *haps_concat,
    const int32_t *hap_offs,
    const int32_t *hap_lens,
    int32_t n_haps,
    double *out_ll            /* shape (n_reads, n_haps) */
) {
    int32_t max_hlen = 0;
    for (int32_t h = 0; h < n_haps; h++) {
        if (hap_lens[h] > max_hlen) max_hlen = hap_lens[h];
    }
    double *M_row = (double *)xmalloc((size_t)(max_hlen + 2) * sizeof(double));
    double *I_row = (double *)xmalloc((size_t)(max_hlen + 2) * sizeof(double));
    double *D_row = (double *)xmalloc((size_t)(max_hlen + 2) * sizeof(double));

    for (int32_t r = 0; r < n_reads; r++) {
        if (read_offs[r] < 0 || read_lens[r] < 0) {
            fprintf(stderr, "[VarLoop FATAL] negative read_off (%lld) or read_len (%d)\n",
                    (long long)read_offs[r], (int)read_lens[r]);
            abort();
        }
        const uint8_t *rseq = pool + (size_t)read_offs[r];
        const uint8_t *rqual = rseq + (size_t)read_lens[r];
        int32_t rlen = read_lens[r];
        for (int32_t h = 0; h < n_haps; h++) {
            const uint8_t *hseq = (const uint8_t *)(haps_concat + hap_offs[h]);
            out_ll[r * n_haps + h] = align_read_hap_semiglobal(
                rseq, rqual, rlen, hseq, hap_lens[h], M_row, I_row, D_row
            );
        }
    }
    free(M_row);
    free(I_row);
    free(D_row);
}
