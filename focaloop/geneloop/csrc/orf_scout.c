/* orf_scout: Fast OpenMP C scanner for unique/low-copy unannotated canonical ORFs.
 *
 * Scans non-excluded regions of nuclear contigs on both strands for uninterrupted
 * canonical ORFs (ATG ... TAA|TAG|TGA) >= min_orf_bp that pass:
 *   1. softmask + 'N' fraction < max_soft_n_frac (default 0.20)
 *   2. GC content in [min_gc, max_gc] (default [0.25, 0.75])
 *   3. Shannon dinucleotide entropy >= min_entropy bits (default 3.30)
 *   4. Genome-wide canonical open-syncmer multiplicity <= max_sync_mult (default 3)
 *      across at least min_lowcopy_frac (default 0.70) of the ORF's open syncmers.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

typedef struct {
    int contig_idx;
    int start;
    int end;
    int strand;
    float gc;
    float dinuc_entropy;
    float soft_n_frac;
    float frac_lowcopy_syncmers;
    int n_syncmers;
} OrfCandidate;

static u8 BASE_CODE[256];
static u8 IS_SOFT_OR_N[256];
static int TABLES_INIT = 0;

static void init_tables(void) {
    if (TABLES_INIT) return;
    memset(BASE_CODE, 4, sizeof(BASE_CODE));
    BASE_CODE['A'] = BASE_CODE['a'] = 0;
    BASE_CODE['C'] = BASE_CODE['c'] = 1;
    BASE_CODE['G'] = BASE_CODE['g'] = 2;
    BASE_CODE['T'] = BASE_CODE['t'] = 3;

    memset(IS_SOFT_OR_N, 0, sizeof(IS_SOFT_OR_N));
    for (int c = 'a'; c <= 'z'; c++) IS_SOFT_OR_N[c] = 1;
    IS_SOFT_OR_N['N'] = 1;
    IS_SOFT_OR_N['n'] = 1;
    TABLES_INIT = 1;
}

/* --------------------------------------------------------------------------
 * Open-addressing hash table: u64 canonical syncmer -> u32 genomic count
 * -------------------------------------------------------------------------- */
#define EMPTY_KEY UINT64_MAX

typedef struct {
    u64 *keys;
    u32 *vals;
    u64 cap;
    u64 mask;
    u64 n;
} SyncTable;

static inline u64 mix64(u64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static int st_init(SyncTable *t, u64 want) {
    u64 c = 1024;
    while (c < want * 2 + 16) c <<= 1;
    t->cap = c;
    t->mask = c - 1;
    t->n = 0;
    t->keys = (u64 *)malloc(c * sizeof(u64));
    t->vals = (u32 *)calloc(c, sizeof(u32));
    if (!t->keys || !t->vals) {
        free(t->keys);
        free(t->vals);
        t->keys = NULL;
        t->vals = NULL;
        return -1;
    }
    memset(t->keys, 0xff, c * sizeof(u64));
    return 0;
}

static void st_free(SyncTable *t) {
    free(t->keys);
    free(t->vals);
    t->keys = NULL;
    t->vals = NULL;
}

static inline u64 st_find(const SyncTable *t, u64 k) {
    u64 i = mix64(k) & t->mask;
    while (1) {
        u64 kk = t->keys[i];
        if (kk == k) return i;
        if (kk == EMPTY_KEY) return t->cap;
        i = (i + 1) & t->mask;
    }
}

static int st_grow(SyncTable *t);

static int st_insert(SyncTable *t, u64 k) {
    if ((t->n + 1) * 2 > t->cap) {
        if (st_grow(t) != 0) return -1;
    }
    u64 i = mix64(k) & t->mask;
    while (1) {
        u64 kk = t->keys[i];
        if (kk == k) return 0;
        if (kk == EMPTY_KEY) {
            t->keys[i] = k;
            t->vals[i] = 0;
            t->n++;
            return 0;
        }
        i = (i + 1) & t->mask;
    }
}

static int st_grow(SyncTable *t) {
    SyncTable n2;
    if (st_init(&n2, t->cap) != 0) return -1;
    for (u64 i = 0; i < t->cap; i++) {
        if (t->keys[i] != EMPTY_KEY) {
            u64 j = mix64(t->keys[i]) & n2.mask;
            while (n2.keys[j] != EMPTY_KEY) j = (j + 1) & n2.mask;
            n2.keys[j] = t->keys[i];
            n2.vals[j] = t->vals[i];
            n2.n++;
        }
    }
    st_free(t);
    *t = n2;
    return 0;
}

/* --------------------------------------------------------------------------
 * Canonical open syncmer scanner (identical semantics to glcore.c)
 * -------------------------------------------------------------------------- */
static void scan_seq_syncmers_insert(const char *seq, int len, int k, int s, SyncTable *ht) {
    if (len < k || k <= 0 || s <= 0 || s > k || k > 32) return;
    const int n_s = k - s + 1;
    const int target_off = (k - s) / 2;
    const u64 kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const u64 smask = (1ULL << (2 * s)) - 1ULL;
    const int rc_shift = 2 * (k - 1);

    u64 f = 0, r = 0;
    int run = 0;
    for (int i = 0; i < len; i++) {
        u8 b = BASE_CODE[(unsigned char)seq[i]];
        if (b > 3) {
            run = 0;
            f = 0;
            r = 0;
            continue;
        }
        f = ((f << 2) | b) & kmask;
        r = (r >> 2) | ((u64)(3 - b) << rc_shift);
        if (++run < k) continue;

        u64 ck = (f <= r) ? f : r;
        int best_pos = 0;
        u64 best_sm = (ck >> (2 * (k - s))) & smask;
        for (int j = 1; j < n_s; j++) {
            u64 sm = (ck >> (2 * (k - s - j))) & smask;
            if (sm < best_sm) {
                best_sm = sm;
                best_pos = j;
            }
        }
        if (best_pos == target_off) {
            st_insert(ht, ck);
        }
    }
}

static void scan_seq_syncmers_count(const char *seq, int len, int k, int s, SyncTable *ht, u32 cap_mult) {
    if (len < k || k <= 0 || s <= 0 || s > k || k > 32) return;
    const int n_s = k - s + 1;
    const int target_off = (k - s) / 2;
    const u64 kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const u64 smask = (1ULL << (2 * s)) - 1ULL;
    const int rc_shift = 2 * (k - 1);

    u64 f = 0, r = 0;
    int run = 0;
    for (int i = 0; i < len; i++) {
        u8 b = BASE_CODE[(unsigned char)seq[i]];
        if (b > 3) {
            run = 0;
            f = 0;
            r = 0;
            continue;
        }
        f = ((f << 2) | b) & kmask;
        r = (r >> 2) | ((u64)(3 - b) << rc_shift);
        if (++run < k) continue;

        u64 ck = (f <= r) ? f : r;
        int best_pos = 0;
        u64 best_sm = (ck >> (2 * (k - s))) & smask;
        for (int j = 1; j < n_s; j++) {
            u64 sm = (ck >> (2 * (k - s - j))) & smask;
            if (sm < best_sm) {
                best_sm = sm;
                best_pos = j;
            }
        }
        if (best_pos == target_off) {
            u64 slot = st_find(ht, ck);
            if (slot < ht->cap) {
                u32 cur = __atomic_load_n(&ht->vals[slot], __ATOMIC_RELAXED);
                if (cur <= cap_mult) {
                    __atomic_fetch_add(&ht->vals[slot], 1U, __ATOMIC_RELAXED);
                }
            }
        }
    }
}

static void eval_orf_syncmers(
    const char *seq, int len, int k, int s,
    const SyncTable *ht, int max_sync_mult,
    int *out_n_sync, float *out_lowcopy_frac
) {
    if (len < k || k <= 0 || s <= 0 || s > k || k > 32) {
        *out_n_sync = 0;
        *out_lowcopy_frac = 0.0f;
        return;
    }
    const int n_s = k - s + 1;
    const int target_off = (k - s) / 2;
    const u64 kmask = (k == 32) ? ~0ULL : ((1ULL << (2 * k)) - 1ULL);
    const u64 smask = (1ULL << (2 * s)) - 1ULL;
    const int rc_shift = 2 * (k - 1);

    u64 f = 0, r = 0;
    int run = 0;
    int n_sync = 0, n_low = 0;
    for (int i = 0; i < len; i++) {
        u8 b = BASE_CODE[(unsigned char)seq[i]];
        if (b > 3) {
            run = 0;
            f = 0;
            r = 0;
            continue;
        }
        f = ((f << 2) | b) & kmask;
        r = (r >> 2) | ((u64)(3 - b) << rc_shift);
        if (++run < k) continue;

        u64 ck = (f <= r) ? f : r;
        int best_pos = 0;
        u64 best_sm = (ck >> (2 * (k - s))) & smask;
        for (int j = 1; j < n_s; j++) {
            u64 sm = (ck >> (2 * (k - s - j))) & smask;
            if (sm < best_sm) {
                best_sm = sm;
                best_pos = j;
            }
        }
        if (best_pos == target_off) {
            n_sync++;
            u64 slot = st_find(ht, ck);
            u32 mult = (slot < ht->cap) ? ht->vals[slot] : 1U;
            if ((int)mult <= max_sync_mult) n_low++;
        }
    }
    *out_n_sync = n_sync;
    *out_lowcopy_frac = (n_sync > 0) ? ((float)n_low / (float)n_sync) : 0.0f;
}

/* --------------------------------------------------------------------------
 * Sequence composition & Shannon dinucleotide entropy check
 * -------------------------------------------------------------------------- */
static int check_orf_seq(
    const char *seq, int len,
    float max_soft_n_frac, float min_gc, float max_gc, float min_entropy,
    float *out_gc, float *out_ent, float *out_soft_n
) {
    if (len <= 0) return 0;
    int soft_n = 0, gc_cnt = 0, acgt_cnt = 0;
    int dinuc[16];
    memset(dinuc, 0, sizeof(dinuc));
    int dinuc_tot = 0;
    u8 prev = 4;

    for (int i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)seq[i];
        if (IS_SOFT_OR_N[ch]) soft_n++;
        u8 b = BASE_CODE[ch];
        if (b <= 3) {
            acgt_cnt++;
            if (b == 1 || b == 2) gc_cnt++;
            if (prev <= 3) {
                dinuc[(prev << 2) | b]++;
                dinuc_tot++;
            }
        }
        prev = b;
    }

    float sn_frac = (float)soft_n / (float)len;
    if (sn_frac >= max_soft_n_frac || acgt_cnt == 0) return 0;

    float gc = (float)gc_cnt / (float)acgt_cnt;
    if (gc < min_gc || gc > max_gc) return 0;

    double H = 0.0;
    if (dinuc_tot > 0) {
        double inv = 1.0 / (double)dinuc_tot;
        for (int d = 0; d < 16; d++) {
            if (dinuc[d] > 0) {
                double p = dinuc[d] * inv;
                H -= p * log2(p);
            }
        }
    }
    if ((float)H < min_entropy) return 0;

    *out_gc = gc;
    *out_ent = (float)H;
    *out_soft_n = sn_frac;
    return 1;
}

typedef struct {
    OrfCandidate *items;
    int n;
    int cap;
} CandVec;

static int cv_push(CandVec *v, const OrfCandidate *c) {
    if (v->n >= v->cap) {
        int nc = v->cap ? v->cap * 2 : 256;
        OrfCandidate *ni = (OrfCandidate *)realloc(v->items, (size_t)nc * sizeof(OrfCandidate));
        if (!ni) return -1;
        v->items = ni;
        v->cap = nc;
    }
    v->items[v->n++] = *c;
    return 0;
}

typedef struct {
    int contig_idx;
    int start;
    int end;
} IntervalTask;

static void scan_interval(
    int ctg_idx, const char *seq, int a, int b,
    int min_orf_bp, float max_soft_n_frac, float min_gc, float max_gc, float min_entropy,
    CandVec *out
) {
    /* Forward (+) strand: frames 0, 1, 2 */
    for (int frame = 0; frame < 3; frame++) {
        int orf_start = -1;
        for (int pos = a + frame; pos + 3 <= b; pos += 3) {
            u8 b0 = BASE_CODE[(unsigned char)seq[pos]];
            u8 b1 = BASE_CODE[(unsigned char)seq[pos + 1]];
            u8 b2 = BASE_CODE[(unsigned char)seq[pos + 2]];
            if (b0 > 3 || b1 > 3 || b2 > 3) {
                if (b0 > 3 && b1 > 3 && b2 > 3) orf_start = -1;
                continue;
            }
            int cod = (b0 << 4) | (b1 << 2) | b2;
            if (orf_start < 0) {
                if (cod == 14) orf_start = pos; /* ATG */
            } else if (cod == 48 || cod == 50 || cod == 56) { /* TAA, TAG, TGA */
                int orf_end = pos + 3;
                int len = orf_end - orf_start;
                if (len >= min_orf_bp) {
                    float gc = 0.0f, ent = 0.0f, sn = 0.0f;
                    if (check_orf_seq(seq + orf_start, len, max_soft_n_frac, min_gc, max_gc, min_entropy, &gc, &ent, &sn)) {
                        OrfCandidate cand = {ctg_idx, orf_start, orf_end, 1, gc, ent, sn, 0.0f, 0};
                        cv_push(out, &cand);
                    }
                }
                orf_start = -1;
            }
        }
    }

    /* Reverse (-) strand: frames 0, 1, 2 */
    for (int frame = 0; frame < 3; frame++) {
        int orf_end = -1;
        for (int pos = b - frame - 3; pos >= a; pos -= 3) {
            u8 r0 = BASE_CODE[(unsigned char)seq[pos + 2]];
            u8 r1 = BASE_CODE[(unsigned char)seq[pos + 1]];
            u8 r2 = BASE_CODE[(unsigned char)seq[pos]];
            if (r0 > 3 || r1 > 3 || r2 > 3) {
                if (r0 > 3 && r1 > 3 && r2 > 3) orf_end = -1;
                continue;
            }
            u8 b0 = 3 - r0;
            u8 b1 = 3 - r1;
            u8 b2 = 3 - r2;
            int cod = (b0 << 4) | (b1 << 2) | b2;
            if (orf_end < 0) {
                if (cod == 14) orf_end = pos + 3; /* ATG on - strand */
            } else if (cod == 48 || cod == 50 || cod == 56) { /* TAA, TAG, TGA on - strand */
                int orf_start = pos;
                int len = orf_end - orf_start;
                if (len >= min_orf_bp) {
                    float gc = 0.0f, ent = 0.0f, sn = 0.0f;
                    if (check_orf_seq(seq + orf_start, len, max_soft_n_frac, min_gc, max_gc, min_entropy, &gc, &ent, &sn)) {
                        OrfCandidate cand = {ctg_idx, orf_start, orf_end, -1, gc, ent, sn, 0.0f, 0};
                        cv_push(out, &cand);
                    }
                }
                orf_end = -1;
            }
        }
    }
}

static int cmp_cand(const void *pa, const void *pb) {
    const OrfCandidate *a = (const OrfCandidate *)pa;
    const OrfCandidate *b = (const OrfCandidate *)pb;
    if (a->contig_idx != b->contig_idx) return (a->contig_idx < b->contig_idx) ? -1 : 1;
    if (a->start != b->start) return (a->start < b->start) ? -1 : 1;
    if (a->end != b->end) return (a->end < b->end) ? -1 : 1;
    return (a->strand > b->strand) ? -1 : (a->strand < b->strand ? 1 : 0);
}

int scan_orphan_orfs(
    int n_contigs,
    const char **contig_seqs,
    const int *contig_lens,
    const int *n_excl_per_contig,
    const int **excl_starts,
    const int **excl_ends,
    int min_orf_bp,
    float max_soft_n_frac,
    float min_gc,
    float max_gc,
    float min_entropy,
    int sync_k,
    int sync_s,
    int max_sync_mult,
    float min_lowcopy_frac,
    int threads,
    OrfCandidate **out_cands,
    int *out_count
) {
    init_tables();
    *out_cands = NULL;
    *out_count = 0;
    if (n_contigs <= 0) return 0;

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    /* Build allowed intervals (complement of sorted/merged excluded spans), chunked to <= 2 Mb for parallelism */
    int task_cap = 256, n_tasks = 0;
    IntervalTask *tasks = (IntervalTask *)malloc((size_t)task_cap * sizeof(IntervalTask));
    if (!tasks) return -1;

    const int CHUNK_BP = 2000000;
    const int OVERLAP_BP = 20000;
    for (int c = 0; c < n_contigs; c++) {
        int L = contig_lens[c];
        int ne = n_excl_per_contig ? n_excl_per_contig[c] : 0;
        int cur = 0;
        for (int j = 0; j <= ne; j++) {
            int es = (j < ne) ? excl_starts[c][j] : L;
            int ee = (j < ne) ? excl_ends[c][j] : L;
            if (es < 0) es = 0;
            if (es > L) es = L;
            if (ee > L) ee = L;
            if (es > cur) {
                int seg_a = cur, seg_b = es;
                if (seg_b - seg_a >= min_orf_bp) {
                    for (int off = seg_a; off < seg_b; off += (CHUNK_BP - OVERLAP_BP)) {
                        int hi = off + CHUNK_BP;
                        if (hi > seg_b) hi = seg_b;
                        if (hi - off >= min_orf_bp) {
                            if (n_tasks >= task_cap) {
                                task_cap *= 2;
                                IntervalTask *nt = (IntervalTask *)realloc(tasks, (size_t)task_cap * sizeof(IntervalTask));
                                if (!nt) { free(tasks); return -1; }
                                tasks = nt;
                            }
                            tasks[n_tasks++] = (IntervalTask){c, off, hi};
                        }
                        if (hi >= seg_b) break;
                    }
                }
            }
            if (ee > cur) cur = ee;
        }
    }

    if (n_tasks == 0) {
        free(tasks);
        return 0;
    }

    /* Parallel Step 1: discover candidate ORFs passing composition & entropy filters */
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
#endif
    CandVec *vecs = (CandVec *)calloc((size_t)n_threads, sizeof(CandVec));
    if (!vecs) { free(tasks); return -1; }

#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (int ti = 0; ti < n_tasks; ti++) {
            IntervalTask tk = tasks[ti];
            scan_interval(
                tk.contig_idx, contig_seqs[tk.contig_idx], tk.start, tk.end,
                min_orf_bp, max_soft_n_frac, min_gc, max_gc, min_entropy,
                &vecs[tid]
            );
        }
    }
#else
    for (int ti = 0; ti < n_tasks; ti++) {
        IntervalTask tk = tasks[ti];
        scan_interval(
            tk.contig_idx, contig_seqs[tk.contig_idx], tk.start, tk.end,
            min_orf_bp, max_soft_n_frac, min_gc, max_gc, min_entropy,
            &vecs[0]
        );
    }
#endif
    free(tasks);

    int total_raw = 0;
    for (int t = 0; t < n_threads; t++) total_raw += vecs[t].n;
    if (total_raw == 0) {
        for (int t = 0; t < n_threads; t++) free(vecs[t].items);
        free(vecs);
        return 0;
    }

    OrfCandidate *raw = (OrfCandidate *)malloc((size_t)total_raw * sizeof(OrfCandidate));
    if (!raw) {
        for (int t = 0; t < n_threads; t++) free(vecs[t].items);
        free(vecs);
        return -1;
    }
    int pos = 0;
    for (int t = 0; t < n_threads; t++) {
        if (vecs[t].n > 0) {
            memcpy(raw + pos, vecs[t].items, (size_t)vecs[t].n * sizeof(OrfCandidate));
            pos += vecs[t].n;
        }
        free(vecs[t].items);
    }
    free(vecs);

    /* Deduplicate candidates from chunk overlaps */
    qsort(raw, (size_t)total_raw, sizeof(OrfCandidate), cmp_cand);
    int n_uniq = 0;
    for (int i = 0; i < total_raw; i++) {
        if (n_uniq > 0 &&
            raw[i].contig_idx == raw[n_uniq - 1].contig_idx &&
            raw[i].start == raw[n_uniq - 1].start &&
            raw[i].end == raw[n_uniq - 1].end &&
            raw[i].strand == raw[n_uniq - 1].strand) {
            continue;
        }
        raw[n_uniq++] = raw[i];
    }

    /* Step 2: Build syncmer hash table from candidate ORFs and count across ALL contigs */
    SyncTable ht;
    u64 est_syncs = (u64)n_uniq * 64ULL + 1024ULL;
    if (st_init(&ht, est_syncs) != 0) {
        free(raw);
        return -1;
    }
    for (int i = 0; i < n_uniq; i++) {
        const OrfCandidate *c = &raw[i];
        scan_seq_syncmers_insert(contig_seqs[c->contig_idx] + c->start, c->end - c->start, sync_k, sync_s, &ht);
    }

    if (ht.n > 0) {
        /* Build 4 Mb genome windows for parallel multiplicity counting */
        int gcap = 128, ng = 0;
        IntervalTask *gtasks = (IntervalTask *)malloc((size_t)gcap * sizeof(IntervalTask));
        if (!gtasks) { st_free(&ht); free(raw); return -1; }
        const int GCHUNK = 4000000;
        const int GOVL = sync_k - 1;
        for (int c = 0; c < n_contigs; c++) {
            int L = contig_lens[c];
            if (L < sync_k) continue;
            for (int off = 0; off < L; off += GCHUNK) {
                int hi = off + GCHUNK + GOVL;
                if (hi > L) hi = L;
                if (hi - off >= sync_k) {
                    if (ng >= gcap) {
                        gcap *= 2;
                        IntervalTask *nt = (IntervalTask *)realloc(gtasks, (size_t)gcap * sizeof(IntervalTask));
                        if (!nt) { free(gtasks); st_free(&ht); free(raw); return -1; }
                        gtasks = nt;
                    }
                    gtasks[ng++] = (IntervalTask){c, off, hi};
                }
                if (off + GCHUNK >= L) break;
            }
        }

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int gi = 0; gi < ng; gi++) {
            IntervalTask tk = gtasks[gi];
            scan_seq_syncmers_count(
                contig_seqs[tk.contig_idx] + tk.start, tk.end - tk.start,
                sync_k, sync_s, &ht, (u32)max_sync_mult
            );
        }
        free(gtasks);
    }

    /* Step 3: Filter candidate ORFs by low-copy syncmer fraction */
    int kept = 0;
    for (int i = 0; i < n_uniq; i++) {
        OrfCandidate c = raw[i];
        int ns = 0;
        float frac = 0.0f;
        eval_orf_syncmers(
            contig_seqs[c.contig_idx] + c.start, c.end - c.start,
            sync_k, sync_s, &ht, max_sync_mult, &ns, &frac
        );
        if (ns > 0 && frac >= min_lowcopy_frac) {
            c.n_syncmers = ns;
            c.frac_lowcopy_syncmers = frac;
            raw[kept++] = c;
        }
    }
    st_free(&ht);

    if (kept == 0) {
        free(raw);
        *out_cands = NULL;
        *out_count = 0;
        return 0;
    }

    OrfCandidate *shrunk = (OrfCandidate *)realloc(raw, (size_t)kept * sizeof(OrfCandidate));
    *out_cands = shrunk ? shrunk : raw;
    *out_count = kept;
    return 0;
}

void free_orf_candidates(OrfCandidate *ptr) {
    free(ptr);
}
