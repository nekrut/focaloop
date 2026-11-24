// glcore: GeneLoop native core (syncmer depletion sieve + SAM junction/capping filter).
//
//   glcore syncmers [-k 31 -s 15]                     stdin seqs (one per line) -> canonical open syncmers
//   glcore mult  -g genome.fa -c cand.fa -o cand.mult [-k -s -t]
//   glcore activate -m cand.mult -c cand.fa -o active.bin [-l loci.txt] [-b bait.fa]
//   glcore sieve -1 r1.fq -2 r2.fq [-a active.bin] [--stats s.json] [--min-hits 2] [--max-miss 1] [-t N]
//   glcore junc  --junc j.tsv --bins b.tsv [--stats s.json] [--bin 1000] [--cap-state f --cap-u N --cap-s N --cap-c N]
//                [--mapq-uniq 10] [--min-anchor 8] [--no-out]  (SAM stdin -> SAM stdout)
//   glcore logan --junc-unw ju.tsv --junc-ka jk.tsv --bins-unw bu.tsv --bins-ka bk.tsv [--stats s.json]
//                [--bin 1000] [--mapq-uniq 10] [--min-anchor 8] [--no-out]  (SAM stdin -> mapped SAM stdout)
//
// Open syncmer definition (identical to geneloop.sieve_syncmer.extract_open_syncmers):
//   canonical k-mer C = min(fwd, revcomp) in A<C<G<T lexicographic (== 2-bit numeric) order;
//   C is an open syncmer iff the FIRST minimal s-mer of C sits at offset t = (k - s) / 2.
//   k-mers containing any non-ACGT byte (incl. soft-masked lowercase when --skip-lower) are skipped.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

static int K = 31, S = 15;
static u8 NT[256];

static void init_nt(int skip_lower) {
    memset(NT, 4, 256);
    NT['A'] = 0; NT['C'] = 1; NT['G'] = 2; NT['T'] = 3;
    if (!skip_lower) { NT['a'] = 0; NT['c'] = 1; NT['g'] = 2; NT['t'] = 3; }
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void die(const char *m) { fprintf(stderr, "glcore: %s\n", m); exit(2); }

static FILE *fopen_checked(const char *path, const char *mode, const char *ctx) {
    FILE *f = fopen(path, mode);
    if (!f) {
        fprintf(stderr, "glcore: %s: cannot open '%s': %s\n", ctx, path ? path : "<null>", strerror(errno));
        exit(2);
    }
    return f;
}

static void close_checked(FILE *f, const char *path, const char *ctx) {
    int ferr = ferror(f);
    int fl = fflush(f);
    int fc = (f == stdout || f == stderr) ? 0 : fclose(f);
    if (ferr || fl != 0 || fc != 0) {
        fprintf(stderr, "glcore: %s: write/close error on '%s': %s\n", ctx, path ? path : "<stream>", strerror(errno));
        exit(2);
    }
}

// ------------------------------------------------------------------------------------------
// Hash table: u64 key -> u32 value (open addressing, linear probe). EMPTY = all ones.
// ------------------------------------------------------------------------------------------
#define EMPTY_KEY UINT64_MAX
typedef struct { u64 *keys; u32 *vals; u64 cap, mask, n; } htab;

static inline u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}
static void ht_init(htab *h, u64 want) {
    u64 c = 1024; while (c < want * 2 + 16) c <<= 1;
    h->cap = c; h->mask = c - 1; h->n = 0;
    h->keys = malloc(c * sizeof(u64)); h->vals = calloc(c, sizeof(u32));
    if (!h->keys || !h->vals) die("oom");
    memset(h->keys, 0xff, c * sizeof(u64));
}
static void ht_free(htab *h) { free(h->keys); free(h->vals); }
static inline u64 ht_find(const htab *h, u64 k) {  // returns slot or cap if missing
    u64 i = mix64(k) & h->mask;
    while (1) {
        u64 kk = h->keys[i];
        if (kk == k) return i;
        if (kk == EMPTY_KEY) return h->cap;
        i = (i + 1) & h->mask;
    }
}
static void ht_grow(htab *h);
static inline u64 ht_insert(htab *h, u64 k) {
    if ((h->n + 1) * 2 > h->cap) ht_grow(h);
    u64 i = mix64(k) & h->mask;
    while (1) {
        u64 kk = h->keys[i];
        if (kk == k) return i;
        if (kk == EMPTY_KEY) { h->keys[i] = k; h->vals[i] = 0; h->n++; return i; }
        i = (i + 1) & h->mask;
    }
}
static void ht_grow(htab *h) {
    htab n2; ht_init(&n2, h->cap);
    for (u64 i = 0; i < h->cap; i++) if (h->keys[i] != EMPTY_KEY) {
        u64 j = ht_insert(&n2, h->keys[i]); n2.vals[j] = h->vals[i];
    }
    ht_free(h); *h = n2;
}

// ------------------------------------------------------------------------------------------
// Syncmer scanning
// ------------------------------------------------------------------------------------------
typedef struct { u64 *fs, *rs; int cap; } scratch_t;

static void scratch_need(scratch_t *sc, int n) {
    if (n <= sc->cap) return;
    int c = sc->cap ? sc->cap : 512; while (c < n) c *= 2;
    sc->fs = realloc(sc->fs, c * sizeof(u64)); sc->rs = realloc(sc->rs, c * sizeof(u64));
    if (!sc->fs || !sc->rs) die("oom");
    sc->cap = c;
}

// Calls cb(canonical_kmer, pos, ctx) for every open syncmer in seq[0..n)
typedef void (*sync_cb)(u64 key, long pos, void *ctx);

static void scan_syncmers(const char *seq, long n, scratch_t *sc, sync_cb cb, void *ctx) {
    if (n < K) return;
    const int m = K - S + 1, t = (K - S) / 2;
    const u64 kmask = (K == 32) ? ~0ULL : ((1ULL << (2 * K)) - 1);
    const u64 smask = (1ULL << (2 * S)) - 1;
    // process in windows to bound scratch for long sequences (genomes)
    const long W = 1L << 20;
    for (long w0 = 0; w0 + K <= n; w0 += W) {
        long w1 = w0 + W + K - 1; if (w1 > n) w1 = n;
        long L = w1 - w0;
        scratch_need(sc, (int)L + 1);
        // s-mer forward / revcomp values at each position (valid flag via run length)
        u64 fs = 0, rs = 0, fk = 0, rk = 0;
        long run = 0;  // consecutive valid bases ending at current
        for (long i = 0; i < L; i++) {
            u8 c = NT[(u8)seq[w0 + i]];
            if (c > 3) { run = 0; fs = rs = fk = rk = 0; continue; }
            run++;
            fs = ((fs << 2) | c) & smask;
            rs = (rs >> 2) | ((u64)(3 - c) << (2 * (S - 1)));
            fk = ((fk << 2) | c) & kmask;
            rk = (rk >> 2) | ((u64)(3 - c) << (2 * (K - 1)));
            if (run >= S) { sc->fs[i - S + 1] = fs; sc->rs[i - S + 1] = rs; }
            if (run >= K) {
                long p = i - K + 1;  // k-mer start (window coords)
                int best = 0; u64 bv;
                if (fk < rk) {
                    bv = sc->fs[p];
                    for (int j = 1; j < m; j++) { u64 v = sc->fs[p + j]; if (v < bv) { bv = v; best = j; } }
                    if (best == t) cb(fk, w0 + p, ctx);
                } else {
                    // canonical = revcomp; its s-mer j == rc s-mer at window pos p + m - 1 - j
                    bv = sc->rs[p + m - 1];
                    for (int j = 1; j < m; j++) { u64 v = sc->rs[p + m - 1 - j]; if (v < bv) { bv = v; best = j; } }
                    if (best == t) cb(rk, w0 + p, ctx);
                }
            }
        }
        if (w1 == n) break;
    }
}

// ------------------------------------------------------------------------------------------
// FASTA reader (whole records)
// ------------------------------------------------------------------------------------------
typedef struct { char *name; char *seq; long len; } rec_t;
typedef struct { rec_t *r; long n, cap; } recs_t;

static void read_fasta(const char *path, recs_t *out) {
    FILE *f = fopen(path, "r"); if (!f) { perror(path); exit(2); }
    char *line = NULL; size_t lc = 0; ssize_t ln;
    out->r = NULL; out->n = out->cap = 0;
    rec_t *cur = NULL; long scap = 0;
    while ((ln = getline(&line, &lc, f)) > 0) {
        while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = 0;
        if (line[0] == '>') {
            if (out->n == out->cap) { out->cap = out->cap ? out->cap * 2 : 1024; out->r = realloc(out->r, out->cap * sizeof(rec_t)); }
            cur = &out->r[out->n++];
            char *sp = line + 1; while (*sp && *sp != ' ' && *sp != '\t') sp++; *sp = 0;
            cur->name = strdup(line + 1); cur->len = 0; scap = 1 << 16; cur->seq = malloc(scap);
        } else if (cur) {
            if (cur->len + ln + 1 > scap) { while (cur->len + ln + 1 > scap) scap *= 2; cur->seq = realloc(cur->seq, scap); }
            memcpy(cur->seq + cur->len, line, ln); cur->len += ln; cur->seq[cur->len] = 0;
        }
    }
    free(line); fclose(f);
}

static void free_fasta(recs_t *out) {
    for (long i = 0; i < out->n; i++) {
        free(out->r[i].name);
        free(out->r[i].seq);
    }
    free(out->r);
    out->r = NULL;
    out->n = out->cap = 0;
}

// ------------------------------------------------------------------------------------------
// syncmers: debug / test parity with the Python reference
// ------------------------------------------------------------------------------------------
static void decode_kmer(u64 v, char *buf) {
    static const char A[4] = {'A', 'C', 'G', 'T'};
    for (int i = K - 1; i >= 0; i--) { buf[i] = A[v & 3]; v >>= 2; }
    buf[K] = 0;
}
static void cb_print(u64 key, long pos, void *ctx) { (void)ctx; (void)pos; char b[40]; decode_kmer(key, b); puts(b); }

static int cmd_syncmers(int argc, char **argv) {
    (void)argc; (void)argv;
    init_nt(0);
    char *line = NULL; size_t lc = 0; ssize_t ln; scratch_t sc = {0};
    while ((ln = getline(&line, &lc, stdin)) > 0) {
        while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = 0;
        scan_syncmers(line, ln, &sc, cb_print, NULL);
        puts("//");
    }
    free(line);
    free(sc.fs);
    free(sc.rs);
    close_checked(stdout, "stdout", "syncmers");
    return 0;
}

// ------------------------------------------------------------------------------------------
// mult: genome-wide occurrence counts of candidate syncmers
// ------------------------------------------------------------------------------------------
static void cb_insert(u64 key, long pos, void *ctx) { (void)pos; ht_insert((htab *)ctx, key); }
typedef struct { htab *h; } mult_ctx;
static void cb_count(u64 key, long pos, void *ctx) {
    (void)pos; htab *h = ((mult_ctx *)ctx)->h;
    u64 i = ht_find(h, key);
    if (i != h->cap) {
        u32 *v = &h->vals[i];
        #pragma omp atomic
        (*v)++;
    }
}

static int cmd_mult(int argc, char **argv) {
    const char *g = NULL, *c = NULL, *o = NULL; int threads = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-g")) { if (i + 1 >= argc) die("mult: missing -g arg"); g = argv[++i]; }
        else if (!strcmp(argv[i], "-c")) { if (i + 1 >= argc) die("mult: missing -c arg"); c = argv[++i]; }
        else if (!strcmp(argv[i], "-o")) { if (i + 1 >= argc) die("mult: missing -o arg"); o = argv[++i]; }
        else if (!strcmp(argv[i], "-k")) { if (i + 1 >= argc) die("mult: missing -k arg"); K = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-s")) { if (i + 1 >= argc) die("mult: missing -s arg"); S = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-t")) { if (i + 1 >= argc) die("mult: missing -t arg"); threads = atoi(argv[++i]); }
    }
    if (!g || !c || !o) die("mult: -g -c -o required");
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    double t0 = now_s();
    // candidate syncmers (soft-masked lowercase skipped)
    init_nt(1);
    recs_t cand; read_fasta(c, &cand);
    htab h; ht_init(&h, 1 << 20); scratch_t sc = {0};
    for (long i = 0; i < cand.n; i++) scan_syncmers(cand.r[i].seq, cand.r[i].len, &sc, cb_insert, &h);
    long ncand = h.n;
    // genome scan (case-insensitive) in parallel chunks
    init_nt(0);
    recs_t gen; read_fasta(g, &gen);
    const long CH = 8L << 20;
    long nchunks = 0;
    for (long i = 0; i < gen.n; i++) nchunks += (gen.r[i].len + CH - 1) / CH;
    long *ci = malloc((size_t)(nchunks ? nchunks : 1) * sizeof(long));
    long *cs = malloc((size_t)(nchunks ? nchunks : 1) * sizeof(long));
    if (!ci || !cs) die("mult: chunks oom");
    long q = 0;
    for (long i = 0; i < gen.n; i++) for (long s0 = 0; s0 < gen.r[i].len; s0 += CH) { ci[q] = i; cs[q] = s0; q++; }
    #pragma omp parallel
    {
        scratch_t lsc = {0}; mult_ctx mc = {&h};
        #pragma omp for schedule(dynamic, 1)
        for (long j = 0; j < nchunks; j++) {
            rec_t *r = &gen.r[ci[j]];
            long a = cs[j], b = a + CH + K - 1; if (b > r->len) b = r->len;
            scan_syncmers(r->seq + a, b - a, &lsc, cb_count, &mc);
        }
        free(lsc.fs); free(lsc.rs);
    }
    FILE *f = fopen_checked(o, "wb", "mult");
    u64 hdr[4] = {0x31544c554d4c47ULL, (u64)K, (u64)S, (u64)h.n};
    if (fwrite(hdr, sizeof(u64), 4, f) != 4) die("mult: write header failed");
    long n1 = 0, n0 = 0, nm = 0;
    for (u64 i = 0; i < h.cap; i++) if (h.keys[i] != EMPTY_KEY) {
        if (fwrite(&h.keys[i], sizeof(u64), 1, f) != 1 || fwrite(&h.vals[i], sizeof(u32), 1, f) != 1)
            die("mult: write entry failed");
        if (h.vals[i] == 1) n1++; else if (h.vals[i] == 0) n0++; else nm++;
    }
    close_checked(f, o, "mult");
    long gbp = 0; for (long i = 0; i < gen.n; i++) gbp += gen.r[i].len;
    fprintf(stderr, "{\"cand_records\":%ld,\"cand_syncmers\":%ld,\"unique\":%ld,\"multi\":%ld,\"absent\":%ld,\"genome_bp\":%ld,\"seconds\":%.2f}\n",
            cand.n, ncand, n1, nm, n0, gbp, now_s() - t0);
    free(ci); free(cs);
    free(sc.fs); free(sc.rs);
    ht_free(&h);
    free_fasta(&cand);
    free_fasta(&gen);
    return 0;
}

static void load_table(const char *path, htab *h, int *k, int *s) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(2); }
    u64 hdr[4]; if (fread(hdr, sizeof(u64), 4, f) != 4) die("bad table");
    *k = (int)hdr[1]; *s = (int)hdr[2];
    ht_init(h, hdr[3] + 16);
    for (u64 i = 0; i < hdr[3]; i++) {
        u64 key; u32 v;
        if (fread(&key, sizeof(u64), 1, f) != 1 || fread(&v, sizeof(u32), 1, f) != 1) die("trunc table");
        u64 j = ht_insert(h, key); h->vals[j] = v;
    }
    fclose(f);
}

// ------------------------------------------------------------------------------------------
// activate: unique syncmers of selected candidate records + bait syncmers
// ------------------------------------------------------------------------------------------
typedef struct { htab *mult, *act; long added, shared, absent; u32 flag; } act_ctx;
static void cb_activate(u64 key, long pos, void *ctx) {
    (void)pos; act_ctx *a = ctx;
    if (a->flag == 2) { u64 j = ht_insert(a->act, key); a->act->vals[j] |= 2; a->added++; return; }
    u64 i = ht_find(a->mult, key);
    if (i == a->mult->cap) { a->absent++; return; }
    if (a->mult->vals[i] == 1) { u64 j = ht_insert(a->act, key); a->act->vals[j] |= 1; a->added++; }
    else a->shared++;
}

static int cmd_activate(int argc, char **argv) {
    const char *m = NULL, *c = NULL, *o = NULL, *l = NULL, *b = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m")) { if (i + 1 >= argc) die("activate: missing -m arg"); m = argv[++i]; }
        else if (!strcmp(argv[i], "-c")) { if (i + 1 >= argc) die("activate: missing -c arg"); c = argv[++i]; }
        else if (!strcmp(argv[i], "-o")) { if (i + 1 >= argc) die("activate: missing -o arg"); o = argv[++i]; }
        else if (!strcmp(argv[i], "-l")) { if (i + 1 >= argc) die("activate: missing -l arg"); l = argv[++i]; }
        else if (!strcmp(argv[i], "-b")) { if (i + 1 >= argc) die("activate: missing -b arg"); b = argv[++i]; }
    }
    if (!o) die("activate: -o required");
    htab mult = {0}, act; ht_init(&act, 1 << 16);
    act_ctx ac = {&mult, &act, 0, 0, 0, 1};
    scratch_t sc = {0};
    if (m && c && l) {
        load_table(m, &mult, &K, &S);
        // selected loci
        htab sel; ht_init(&sel, 1 << 12);
        FILE *lf = fopen_checked(l, "r", "activate");
        char *line = NULL; size_t lc = 0; ssize_t ln;
        while ((ln = getline(&line, &lc, lf)) > 0) {
            while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = 0;
            if (ln) ht_insert(&sel, mix64(strtoull(line + (line[0] == 'L' ? 1 : 0), NULL, 10) + 1));
        }
        fclose(lf); free(line);
        init_nt(1);
        recs_t cand; read_fasta(c, &cand);
        for (long i = 0; i < cand.n; i++) {
            const char *nm = cand.r[i].name;
            u64 id = strtoull(nm + (nm[0] == 'L' ? 1 : 0), NULL, 10);
            if (ht_find(&sel, mix64(id + 1)) == sel.cap) continue;
            scan_syncmers(cand.r[i].seq, cand.r[i].len, &sc, cb_activate, &ac);
        }
        ht_free(&sel);
        free_fasta(&cand);
    }
    long locus_added = ac.added;
    if (b) {
        init_nt(0);
        recs_t bait; read_fasta(b, &bait);
        ac.flag = 2; ac.added = 0;
        for (long i = 0; i < bait.n; i++) scan_syncmers(bait.r[i].seq, bait.r[i].len, &sc, cb_activate, &ac);
        free_fasta(&bait);
    }
    FILE *f = fopen_checked(o, "wb", "activate");
    u64 hdr[4] = {0x31544341ULL, (u64)K, (u64)S, (u64)act.n};
    if (fwrite(hdr, sizeof(u64), 4, f) != 4) die("activate: write header failed");
    for (u64 i = 0; i < act.cap; i++) if (act.keys[i] != EMPTY_KEY) {
        if (fwrite(&act.keys[i], 8, 1, f) != 1 || fwrite(&act.vals[i], 4, 1, f) != 1)
            die("activate: write entry failed");
    }
    close_checked(f, o, "activate");
    fprintf(stderr, "{\"active\":%lu,\"locus_unique_hits\":%ld,\"locus_shared_skipped\":%ld,\"locus_absent\":%ld,\"bait_syncmers\":%ld}\n",
            (unsigned long)act.n, locus_added, ac.shared, ac.absent, ac.flag == 2 ? ac.added : 0L);
    ht_free(&mult);
    ht_free(&act);
    free(sc.fs);
    free(sc.rs);
    return 0;
}

// ------------------------------------------------------------------------------------------
// sieve: interleaved FASTQ stdin -> R1/R2 (FIFOs); drop pairs whose BOTH mates are saturated
// ------------------------------------------------------------------------------------------
typedef struct { long name, seq, qual; int nlen, slen; } fq_t;
typedef struct { const htab *act; int hits_l, hits_b, miss; } read_ctx;
static void cb_read(u64 key, long pos, void *ctx) {
    (void)pos; read_ctx *r = ctx;
    u64 i = ht_find(r->act, key);
    if (i == r->act->cap) { r->miss++; return; }
    if (r->act->vals[i] & 1) r->hits_l++; else r->hits_b++;
}

static int spot_key_len(const char *nm, int n) {
    // key spots on the first whitespace-delimited token, then strip trailing ".D" or "/D"
    int tok = 0;
    while (tok < n && nm[tok] != ' ' && nm[tok] != '\t') tok++;
    if (tok >= 4 && (nm[tok - 2] == '.' || nm[tok - 2] == '/') && nm[tok - 1] >= '0' && nm[tok - 1] <= '9')
        return tok - 2;
    return tok;
}

static void set_pipe(FILE *f) {
#ifdef F_SETPIPE_SZ
    fcntl(fileno(f), F_SETPIPE_SZ, 1 << 20);
#endif
}

static int cmd_sieve(int argc, char **argv) {
    const char *a = NULL, *o1 = NULL, *o2 = NULL, *st = NULL;
    int min_hits = 2, max_miss = 1, threads = 0, allow_empty = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a")) { if (i + 1 >= argc) die("sieve: missing -a arg"); a = argv[++i]; }
        else if (!strcmp(argv[i], "-1")) { if (i + 1 >= argc) die("sieve: missing -1 arg"); o1 = argv[++i]; }
        else if (!strcmp(argv[i], "-2")) { if (i + 1 >= argc) die("sieve: missing -2 arg"); o2 = argv[++i]; }
        else if (!strcmp(argv[i], "--stats")) { if (i + 1 >= argc) die("sieve: missing --stats arg"); st = argv[++i]; }
        else if (!strcmp(argv[i], "--min-hits")) { if (i + 1 >= argc) die("sieve: missing --min-hits arg"); min_hits = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--max-miss")) { if (i + 1 >= argc) die("sieve: missing --max-miss arg"); max_miss = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-t")) { if (i + 1 >= argc) die("sieve: missing -t arg"); threads = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--allow-empty")) allow_empty = 1;
        else if (!strcmp(argv[i], "-k") || !strcmp(argv[i], "-s")) { i++; }
        else die("sieve: unknown flag");
    }
    if (!o1 || !o2) die("sieve: -1 -2 required");
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    init_nt(0);
    htab act = {0}; int have_act = 0;
    if (a) { int k, s; load_table(a, &act, &k, &s); K = k; S = s; have_act = act.n > 0; }
    double t0 = now_s();
    FILE *f1 = fopen_checked(o1, "w", "sieve");
    FILE *f2 = fopen_checked(o2, "w", "sieve");
    set_pipe(f1); set_pipe(f2);
    static char ob1[1 << 17], ob2[1 << 17];
    setvbuf(f1, ob1, _IOFBF, sizeof ob1); setvbuf(f2, ob2, _IOFBF, sizeof ob2);
    setvbuf(stdin, NULL, _IOFBF, 1 << 22);

    const long BATCH = 1 << 16;  // records per batch
    long arena_cap = 64L << 20; char *arena = malloc(arena_cap);
    fq_t *recs = malloc(BATCH * sizeof(fq_t));
    long *pa = malloc(BATCH * sizeof(long));  // pair first-record index
    u8 *dec = malloc(BATCH);
    if (!arena || !recs || !pa || !dec) die("sieve: oom");
    char *line = NULL; size_t lc = 0; ssize_t ln;
    long pairs_in = 0, pairs_out = 0, drop_l = 0, drop_b = 0, orphans = 0, reads_in = 0, bases_in = 0;
    int eof = 0;
    fq_t carry; int have_carry = 0; char *carry_buf = NULL; long carry_len = 0;
    const char *bad_msg = NULL;
    while (!eof && !bad_msg) {
        long nrec = 0, used = 0;
        if (have_carry) {
            memcpy(arena, carry_buf, carry_len);
            recs[0] = carry; nrec = 1; used = carry_len; have_carry = 0;
        }
        while (nrec < BATCH) {
            long start = used; fq_t r = {0, 0, 0, 0, 0};
            int ok = 1;
            for (int L = 0; L < 4; L++) {
                ln = getline(&line, &lc, stdin);
                if (ln <= 0) {
                    if (L > 0) bad_msg = "sieve: truncated FASTQ record at EOF";
                    ok = 0; eof = 1; break;
                }
                while (ln > 0 && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = 0;
                if (L == 0) {
                    if (ln < 2 || line[0] != '@' || line[1] == ' ' || line[1] == '\t') {
                        bad_msg = "sieve: malformed FASTQ header (expected '@')";
                        ok = 0; break;
                    }
                    r.name = used; r.nlen = (int)ln;
                } else if (L == 1) {
                    r.seq = used; r.slen = (int)ln;
                } else if (L == 2) {
                    if (ln < 1 || line[0] != '+') {
                        bad_msg = "sieve: malformed FASTQ separator (expected '+')";
                        ok = 0; break;
                    }
                    continue;
                } else {
                    if ((int)ln != r.slen) {
                        bad_msg = "sieve: FASTQ sequence and quality lengths differ";
                        ok = 0; break;
                    }
                    r.qual = used;
                }
                if (used + ln + 1 > arena_cap) {
                    while (used + ln + 1 > arena_cap) arena_cap *= 2;
                    arena = realloc(arena, arena_cap);
                    if (!arena) die("sieve: arena oom");
                }
                memcpy(arena + used, line, ln + 1); used += ln + 1;
            }
            if (!ok) { used = start; break; }
            recs[nrec++] = r; reads_in++; bases_in += r.slen;
        }
        if (bad_msg) break;
        // pair consecutive records sharing a spot key
        long np = 0, i = 0;
        while (i < nrec) {
            if (i + 1 < nrec) {
                fq_t *x = &recs[i], *y = &recs[i + 1];
                int kx = spot_key_len(arena + x->name, x->nlen), ky = spot_key_len(arena + y->name, y->nlen);
                if (kx == ky && !memcmp(arena + x->name, arena + y->name, kx)) { pa[np++] = i; i += 2; continue; }
                i++; orphans++;
                continue;
            }
            // last record of batch: carry to next batch if not eof
            if (!eof) {
                carry = recs[i];
                long beg = carry.name, end = carry.qual + carry.slen + 1;
                carry_len = end - beg; carry_buf = realloc(carry_buf, carry_len);
                if (!carry_buf) die("sieve: carry oom");
                memcpy(carry_buf, arena + beg, carry_len);
                carry.name -= beg; carry.seq -= beg; carry.qual -= beg;
                have_carry = 1;
            } else orphans++;
            i++;
        }
        pairs_in += np;
        if (have_act) {
            #pragma omp parallel
            {
                scratch_t sc = {0};
                #pragma omp for schedule(static)
                for (long p = 0; p < np; p++) {
                    int dep[2], bait[2];
                    for (int m = 0; m < 2; m++) {
                        fq_t *r = &recs[pa[p] + m];
                        read_ctx rc = {&act, 0, 0, 0};
                        scan_syncmers(arena + r->seq, r->slen, &sc, cb_read, &rc);
                        int hits = rc.hits_l + rc.hits_b;
                        dep[m] = hits >= min_hits && rc.miss <= max_miss;
                        bait[m] = rc.hits_b > rc.hits_l;
                    }
                    dec[p] = (dep[0] && dep[1]) ? ((bait[0] || bait[1]) ? 2 : 1) : 0;
                }
                free(sc.fs); free(sc.rs);
            }
        } else memset(dec, 0, np);
        // write in small synchronized chunks (both FIFOs flushed together -> no lockstep deadlock)
        long since = 0;
        for (long p = 0; p < np; p++) {
            if (dec[p] == 1) { drop_l++; continue; }
            if (dec[p] == 2) { drop_b++; continue; }
            for (int m = 0; m < 2; m++) {
                fq_t *r = &recs[pa[p] + m]; FILE *f = m ? f2 : f1;
                int kl = spot_key_len(arena + r->name, r->nlen);
                fwrite(arena + r->name, 1, kl, f); fputs(m ? "/2\n" : "/1\n", f);
                fwrite(arena + r->seq, 1, r->slen, f); fputs("\n+\n", f);
                fwrite(arena + r->qual, 1, r->slen, f); fputc('\n', f);
            }
            pairs_out++;
            if (++since == 256) { fflush(f1); fflush(f2); since = 0; }
        }
        fflush(f1); fflush(f2);
    }
    close_checked(f1, o1, "sieve");
    close_checked(f2, o2, "sieve");
    free(arena); free(recs); free(pa); free(dec); free(line); free(carry_buf);
    if (bad_msg) {
        ht_free(&act);
        die(bad_msg);
    }
    if (reads_in > 0 && pairs_in <= 0) {
        ht_free(&act);
        die("sieve: zero paired reads found in input");
    }
    if (!allow_empty && pairs_in <= 0) {
        ht_free(&act);
        die("sieve: empty input (zero paired reads)");
    }
    double el = now_s() - t0;
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    double cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    FILE *sf = st ? fopen_checked(st, "w", "sieve") : stderr;
    fprintf(sf, "{\"pairs_in\":%ld,\"pairs_out\":%ld,\"pairs_dropped_locus\":%ld,\"pairs_dropped_bait\":%ld,"
                "\"orphans\":%ld,\"reads_in\":%ld,\"bases_in\":%ld,\"active_syncmers\":%lu,\"wall_s\":%.3f,\"cpu_s\":%.3f}\n",
            pairs_in, pairs_out, drop_l, drop_b, orphans, reads_in, bases_in, (unsigned long)act.n, el, cpu);
    if (st) close_checked(sf, st, "sieve");
    ht_free(&act);
    return 0;
}

// ------------------------------------------------------------------------------------------
// junc: SAM stdin -> SAM stdout; junction counts, coverage bins, optional per-bin capping
// ------------------------------------------------------------------------------------------
typedef struct { u32 tid, a, b; u32 n_total, n_ua, max_anchor; } jrec;
typedef struct { jrec *v; u64 cap, mask, n; } jtab;
static inline u64 jhash(u32 t, u32 a, u32 b) { return mix64(((u64)t << 40) ^ ((u64)a << 20) ^ (u64)b ^ ((u64)a << 44)); }
static void jt_init(jtab *j, u64 c0) { u64 c = 1024; while (c < c0) c <<= 1; j->cap = c; j->mask = c - 1; j->n = 0; j->v = calloc(c, sizeof(jrec)); for (u64 i = 0; i < c; i++) j->v[i].tid = UINT32_MAX; }
static jrec *jt_get(jtab *j, u32 t, u32 a, u32 b);
static void jt_grow(jtab *j) {
    jtab n; jt_init(&n, j->cap * 2);
    for (u64 i = 0; i < j->cap; i++) if (j->v[i].tid != UINT32_MAX) { jrec *r = jt_get(&n, j->v[i].tid, j->v[i].a, j->v[i].b); *r = j->v[i]; }
    free(j->v); *j = n;
}
static jrec *jt_get(jtab *j, u32 t, u32 a, u32 b) {
    if ((j->n + 1) * 2 > j->cap) jt_grow(j);
    u64 i = jhash(t, a, b) & j->mask;
    while (1) {
        jrec *r = &j->v[i];
        if (r->tid == UINT32_MAX) { memset(r, 0, sizeof *r); r->tid = t; r->a = a; r->b = b; j->n++; return r; }
        if (r->tid == t && r->a == a && r->b == b) return r;
        i = (i + 1) & j->mask;
    }
}

typedef struct { char *name; u64 len, off; } ctg_t;

static int cmd_junc(int argc, char **argv) {
    const char *jo = NULL, *bo = NULL, *st = NULL, *capf = NULL;
    int binsz = 1000, mapq_u = 10, min_anchor = 8, cap_u = -1, cap_s = -1, cap_c = -1, no_out = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--junc")) { if (i + 1 >= argc) die("junc: missing --junc arg"); jo = argv[++i]; }
        else if (!strcmp(argv[i], "--bins")) { if (i + 1 >= argc) die("junc: missing --bins arg"); bo = argv[++i]; }
        else if (!strcmp(argv[i], "--stats")) { if (i + 1 >= argc) die("junc: missing --stats arg"); st = argv[++i]; }
        else if (!strcmp(argv[i], "--bin")) { if (i + 1 >= argc) die("junc: missing --bin arg"); binsz = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--mapq-uniq")) { if (i + 1 >= argc) die("junc: missing --mapq-uniq arg"); mapq_u = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--min-anchor")) { if (i + 1 >= argc) die("junc: missing --min-anchor arg"); min_anchor = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cap-state")) { if (i + 1 >= argc) die("junc: missing --cap-state arg"); capf = argv[++i]; }
        else if (!strcmp(argv[i], "--cap-u")) { if (i + 1 >= argc) die("junc: missing --cap-u arg"); cap_u = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cap-s")) { if (i + 1 >= argc) die("junc: missing --cap-s arg"); cap_s = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--cap-c")) { if (i + 1 >= argc) die("junc: missing --cap-c arg"); cap_c = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--no-out")) no_out = 1;
        else die("junc: unknown flag");
    }
    if (binsz < 1) die("junc: --bin must be >= 1");
    int capping = capf != NULL;
    setvbuf(stdin, NULL, _IOFBF, 1 << 22);
    static char obuf[1 << 20]; setvbuf(stdout, obuf, _IOFBF, sizeof obuf);
    ctg_t *ctg = NULL; long nctg = 0, cctg = 0; u64 nbins = 0;
    u32 *bins = NULL; u16 *capv = NULL;
    jtab jt; jt_init(&jt, 1 << 16);
    int jcap = 256;
    long *jA = malloc((size_t)jcap * sizeof(long));
    long *jB = malloc((size_t)jcap * sizeof(long));
    long *blks = malloc((size_t)(jcap + 2) * sizeof(long));
    if (!jA || !jB || !blks) die("junc: cigar oom");
    long nrec = 0, nmapped = 0, n_unk = 0, nprim = 0, nspl = 0, nout = 0, ncapped = 0, nuniq = 0;
    char *line = NULL; size_t lc = 0; ssize_t ln;
    int hdr_done = 0; long last_tid = -1; char *last_name = NULL; long *cidx = NULL; u64 cmask = 0;
    while ((ln = getline(&line, &lc, stdin)) > 0) {
        if (line[0] == '@') {
            if (!strncmp(line, "@SQ", 3)) {
                char *sn = strstr(line, "SN:"), *lnp = strstr(line, "LN:");
                if (sn && lnp) {
                    if (nctg == cctg) {
                        cctg = cctg ? cctg * 2 : 256;
                        ctg = realloc(ctg, (size_t)cctg * sizeof(ctg_t));
                        if (!ctg) die("junc: ctg oom");
                    }
                    char *e = sn + 3; while (*e && *e != '\t' && *e != '\n' && *e != '\r') e++;
                    ctg[nctg].name = strndup(sn + 3, (size_t)(e - sn - 3));
                    if (!ctg[nctg].name) die("junc: ctg name oom");
                    ctg[nctg].len = strtoull(lnp + 3, NULL, 10);
                    ctg[nctg].off = nbins; nbins += ctg[nctg].len / (u64)binsz + 1; nctg++;
                }
            }
            if (!no_out) fwrite(line, 1, (size_t)ln, stdout);
            continue;
        }
        if (!hdr_done) {
            hdr_done = 1;
            u64 cc = 64; while (cc < (u64)nctg * 2 + 2) cc <<= 1; cmask = cc - 1;
            cidx = malloc((size_t)cc * sizeof(long));
            if (!cidx) die("junc: cidx oom");
            for (u64 i = 0; i < cc; i++) cidx[i] = -1;
            for (long t = 0; t < nctg; t++) {
                u64 h = 1469598103934665603ULL; for (const char *q = ctg[t].name; *q; q++) { h ^= (u8)*q; h *= 1099511628211ULL; }
                u64 i = mix64(h) & cmask; while (cidx[i] >= 0) i = (i + 1) & cmask; cidx[i] = t;
            }
            bins = calloc(nbins + 1, sizeof(u32));
            if (!bins) die("junc: bins oom");
            if (capping) {
                capv = calloc(3 * (nbins + 1), sizeof(u16));
                if (!capv) die("junc: capv oom");
                FILE *cf = fopen(capf, "rb");
                if (cf) { size_t r = fread(capv, sizeof(u16), 3 * (nbins + 1), cf); (void)r; fclose(cf); }
            }
        }
        nrec++;
        // parse fields
        char *f[12]; int nf = 0; char *p = line;
        f[nf++] = p;
        while (*p && nf < 12) { if (*p == '\t') { *p = 0; f[nf++] = p + 1; } p++; }
        if (nf < 11) continue;
        int flag = atoi(f[1]);
        if (flag & 4) continue;
        const char *rname = f[2];
        long tid = -1;
        if (last_tid >= 0 && !strcmp(last_name, rname)) tid = last_tid;
        else {
            u64 h = 1469598103934665603ULL; for (const char *q = rname; *q; q++) { h ^= (u8)*q; h *= 1099511628211ULL; }
            u64 i = mix64(h) & cmask;
            while (cidx[i] >= 0) { if (!strcmp(ctg[cidx[i]].name, rname)) { tid = cidx[i]; break; } i = (i + 1) & cmask; }
            if (tid >= 0) { last_tid = tid; last_name = ctg[tid].name; }
        }
        if (tid < 0) { n_unk++; continue; }
        nmapped++;
        long pos = atol(f[3]) - 1;
        if (pos < 0) pos = 0;
        int mapq = atoi(f[4]);
        int primary = !(flag & 0x100) && !(flag & 0x800);
        // CIGAR walk: aligned block lengths between N operations, junction coordinates, soft clips
        const char *c = f[5]; long rp = pos; int clip5 = 0, clip3 = 0, spliced = 0, seen_m = 0;
        int nb = 0; long blk = 0; int nj = 0;
        while (*c && *c != '\t') {
            long n = 0; while (*c >= '0' && *c <= '9') { n = n * 10 + (*c - '0'); c++; }
            if (!*c || *c == '\t') break;
            char op = *c++;
            switch (op) {
                case 'M': case '=': case 'X': blk += n; rp += n; seen_m = 1; break;
                case 'D': rp += n; break;
                case 'N':
                    if (nj >= jcap) {
                        jcap *= 2;
                        jA = realloc(jA, (size_t)jcap * sizeof(long));
                        jB = realloc(jB, (size_t)jcap * sizeof(long));
                        blks = realloc(blks, (size_t)(jcap + 2) * sizeof(long));
                        if (!jA || !jB || !blks) die("junc: cigar realloc oom");
                    }
                    jA[nj] = rp; jB[nj] = rp + n; blks[nb++] = blk; nj++;
                    blk = 0; rp += n; spliced = 1; break;
                case 'S': if (!seen_m) clip5 = (int)n; else clip3 = (int)n; break;
                default: break;
            }
        }
        blks[nb++] = blk;
        u64 ctg_nb = ctg[tid].len / (u64)binsz + 1;
        u64 b_off = (u64)pos / (u64)binsz;
        if (b_off >= ctg_nb) b_off = ctg_nb - 1;
        u64 bin = ctg[tid].off + b_off;
        if (primary) {
            nprim++;
            if (mapq >= mapq_u) { bins[bin]++; nuniq++; }
            if (spliced) nspl++;
            for (int q = 0; q < nj; q++) {
                jrec *r = jt_get(&jt, (u32)tid, (u32)jA[q], (u32)jB[q]);
                int an = (int)(blks[q] < blks[q + 1] ? blks[q] : blks[q + 1]);
                r->n_total++;
                if (mapq >= mapq_u && an >= min_anchor) r->n_ua++;
                if ((u32)an > r->max_anchor) r->max_anchor = (u32)an;
            }
        }
        int emit = 1;
        if (capping) {
            if (!primary) emit = 0;
            else {
                int cls = spliced ? 1 : ((clip5 >= 3 || clip3 >= 3) ? 2 : 0);
                int lim = cls == 1 ? cap_s : (cls == 2 ? cap_c : cap_u);
                u16 *cv = &capv[3 * bin + cls];
                if (lim >= 0 && *cv >= lim) { emit = 0; ncapped++; }
                else if (*cv < 65535) (*cv)++;
            }
        }
        if (emit && !no_out) {
            for (int q = 0; q < nf - 1; q++) f[q + 1][-1] = '\t';
            fwrite(line, 1, (size_t)ln, stdout); nout++;
        }
    }
    if (!no_out) close_checked(stdout, "stdout", "junc");
    if (jo) {
        FILE *jf = fopen_checked(jo, "w", "junc");
        for (u64 i = 0; i < jt.cap; i++) if (jt.v[i].tid != UINT32_MAX) {
            jrec *r = &jt.v[i];
            fprintf(jf, "%s\t%u\t%u\t%u\t%u\t%u\n", ctg[r->tid].name, r->a, r->b, r->n_total, r->n_ua, r->max_anchor);
        }
        close_checked(jf, jo, "junc");
    }
    if (bo) {
        FILE *bf = fopen_checked(bo, "w", "junc");
        if (bins) {
            for (long i = 0; i < nctg; i++) {
                u64 nb = ctg[i].len / (u64)binsz + 1;
                for (u64 q = 0; q < nb; q++) { u32 v = bins[ctg[i].off + q]; if (v) fprintf(bf, "%s\t%lu\t%u\n", ctg[i].name, (unsigned long)q, v); }
            }
        }
        close_checked(bf, bo, "junc");
    }
    if (capping && capv) {
        FILE *cf = fopen_checked(capf, "wb", "junc");
        size_t nw = 3 * (size_t)(nbins + 1);
        if (fwrite(capv, sizeof(u16), nw, cf) != nw) die("junc: cap-state write failed");
        close_checked(cf, capf, "junc");
    }
    FILE *sf = st ? fopen_checked(st, "w", "junc") : stderr;
    fprintf(sf, "{\"records\":%ld,\"mapped_records\":%ld,\"unknown_rname\":%ld,\"n_sq\":%ld,\"primary\":%ld,\"primary_unique\":%ld,\"spliced_primary\":%ld,\"junctions\":%lu,\"out_records\":%ld,\"capped_records\":%ld}\n",
            nrec, nmapped, n_unk, nctg, nprim, nuniq, nspl, (unsigned long)jt.n, nout, ncapped);
    if (st) close_checked(sf, st, "junc");

    free(line);
    free(jA);
    free(jB);
    free(blks);
    free(cidx);
    free(bins);
    free(capv);
    free(jt.v);
    for (long i = 0; i < nctg; i++) free(ctg[i].name);
    free(ctg);
    return 0;
}

// ------------------------------------------------------------------------------------------
// logan: SAM stdin -> mapped SAM stdout; per-run unweighted + ka-weighted junctions & 1 kb bins
// ------------------------------------------------------------------------------------------
typedef struct {
    u32 tid, rid, a, b;
    u32 n_total, n_ua, max_anchor;
    float ka_total, ka_ua;
} lj_rec_t;

typedef struct {
    lj_rec_t *v;
    u64 cap, mask, n;
} lj_tab_t;

static void lj_init(lj_tab_t *j, u64 c0) {
    u64 c = 1024;
    while (c < c0) c <<= 1;
    j->cap = c;
    j->mask = c - 1;
    j->n = 0;
    j->v = calloc(c, sizeof(lj_rec_t));
    if (!j->v) die("logan: oom");
    for (u64 i = 0; i < c; i++) j->v[i].tid = UINT32_MAX;
}

static inline u64 lj_hash(u32 tid, u32 rid, u32 a, u32 b) {
    u64 x = ((u64)tid << 40) ^ ((u64)rid << 32) ^ ((u64)a << 16) ^ (u64)b ^ ((u64)a << 45);
    return mix64(x);
}

static lj_rec_t *lj_get(lj_tab_t *j, u32 tid, u32 rid, u32 a, u32 b);

static void lj_grow(lj_tab_t *j) {
    lj_tab_t n;
    lj_init(&n, j->cap * 2);
    for (u64 i = 0; i < j->cap; i++) {
        if (j->v[i].tid != UINT32_MAX) {
            lj_rec_t *r = lj_get(&n, j->v[i].tid, j->v[i].rid, j->v[i].a, j->v[i].b);
            *r = j->v[i];
        }
    }
    free(j->v);
    *j = n;
}

static lj_rec_t *lj_get(lj_tab_t *j, u32 tid, u32 rid, u32 a, u32 b) {
    if ((j->n + 1) * 4 > j->cap * 3) lj_grow(j);
    u64 i = lj_hash(tid, rid, a, b) & j->mask;
    while (1) {
        lj_rec_t *r = &j->v[i];
        if (r->tid == UINT32_MAX) {
            memset(r, 0, sizeof(*r));
            r->tid = tid;
            r->rid = rid;
            r->a = a;
            r->b = b;
            j->n++;
            return r;
        }
        if (r->tid == tid && r->rid == rid && r->a == a && r->b == b) return r;
        i = (i + 1) & j->mask;
    }
}

typedef struct {
    char **names;
    size_t *lens;
    u64 *n_recs;
    u64 *n_mapped;
    int n, cap;
    int *ht;
    u64 hcap, hmask;
    int last_id;
} lj_runs_t;

static void lj_runs_init(lj_runs_t *rs) {
    memset(rs, 0, sizeof(*rs));
    rs->cap = 64;
    rs->names = malloc((size_t)rs->cap * sizeof(char *));
    rs->lens = malloc((size_t)rs->cap * sizeof(size_t));
    rs->n_recs = calloc((size_t)rs->cap, sizeof(u64));
    rs->n_mapped = calloc((size_t)rs->cap, sizeof(u64));
    rs->hcap = 256;
    rs->hmask = rs->hcap - 1;
    rs->ht = malloc((size_t)rs->hcap * sizeof(int));
    if (!rs->names || !rs->lens || !rs->n_recs || !rs->n_mapped || !rs->ht) die("logan: runs oom");
    for (u64 i = 0; i < rs->hcap; i++) rs->ht[i] = -1;
    rs->last_id = -1;
}

static inline u64 lj_str_hash(const char *s, size_t len) {
    u64 h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (u8)s[i];
        h *= 1099511628211ULL;
    }
    return mix64(h);
}

static void lj_runs_rehash(lj_runs_t *rs) {
    u64 nc = rs->hcap * 2;
    u64 nm = nc - 1;
    int *nh = malloc((size_t)nc * sizeof(int));
    if (!nh) die("logan: runs rehash oom");
    for (u64 i = 0; i < nc; i++) nh[i] = -1;
    for (int r = 0; r < rs->n; r++) {
        u64 slot = lj_str_hash(rs->names[r], rs->lens[r]) & nm;
        while (nh[slot] >= 0) slot = (slot + 1) & nm;
        nh[slot] = r;
    }
    free(rs->ht);
    rs->ht = nh;
    rs->hcap = nc;
    rs->hmask = nm;
}

static int lj_get_run_id(lj_runs_t *rs, const char *qname) {
    size_t qlen = strlen(qname);
    size_t len = qlen;
    const char *u_last = strrchr(qname, '_');
    if (u_last && u_last > qname && u_last[1] >= '0' && u_last[1] <= '9') {
        const char *d = u_last + 1;
        while (*d >= '0' && *d <= '9') d++;
        if (*d == '\0' || *d == '/') {
            len = (size_t)(u_last - qname);
        }
    } else {
        const char *u_first = strchr(qname, '_');
        if (u_first && u_first > qname) {
            len = (size_t)(u_first - qname);
        }
    }
    if (rs->last_id >= 0 && rs->lens[rs->last_id] == len && !memcmp(rs->names[rs->last_id], qname, len)) {
        return rs->last_id;
    }
    u64 h = lj_str_hash(qname, len);
    u64 slot = h & rs->hmask;
    while (rs->ht[slot] >= 0) {
        int id = rs->ht[slot];
        if (rs->lens[id] == len && !memcmp(rs->names[id], qname, len)) {
            rs->last_id = id;
            return id;
        }
        slot = (slot + 1) & rs->hmask;
    }
    if (rs->n == rs->cap) {
        int old_cap = rs->cap;
        rs->cap *= 2;
        rs->names = realloc(rs->names, (size_t)rs->cap * sizeof(char *));
        rs->lens = realloc(rs->lens, (size_t)rs->cap * sizeof(size_t));
        rs->n_recs = realloc(rs->n_recs, (size_t)rs->cap * sizeof(u64));
        rs->n_mapped = realloc(rs->n_mapped, (size_t)rs->cap * sizeof(u64));
        if (!rs->names || !rs->lens || !rs->n_recs || !rs->n_mapped) die("logan: runs realloc oom");
        memset(rs->n_recs + old_cap, 0, (size_t)(rs->cap - old_cap) * sizeof(u64));
        memset(rs->n_mapped + old_cap, 0, (size_t)(rs->cap - old_cap) * sizeof(u64));
    }
    int id = rs->n++;
    rs->names[id] = strndup(qname, len);
    if (!rs->names[id]) die("logan: strndup oom");
    rs->lens[id] = len;
    rs->ht[slot] = id;
    if ((u64)(rs->n + 1) * 2 > rs->hcap) lj_runs_rehash(rs);
    rs->last_id = id;
    return id;
}

static void lj_runs_free(lj_runs_t *rs) {
    for (int i = 0; i < rs->n; i++) free(rs->names[i]);
    free(rs->names);
    free(rs->lens);
    free(rs->n_recs);
    free(rs->n_mapped);
    free(rs->ht);
}

static u64 lj_flush_table(lj_tab_t *jt, int keep_rid, FILE *jf_unw, FILE *jf_ka, const lj_runs_t *runs, const ctg_t *ctg) {
    u64 emitted = 0;
    lj_tab_t next;
    u64 keep_est = (keep_rid >= 0) ? 1024 : 1024;
    if (keep_rid >= 0) {
        u64 nk = 0;
        for (u64 i = 0; i < jt->cap; i++) {
            if (jt->v[i].tid != UINT32_MAX && (int)jt->v[i].rid == keep_rid) nk++;
        }
        while (keep_est < nk * 2 + 16) keep_est <<= 1;
    }
    lj_init(&next, keep_est);
    for (u64 i = 0; i < jt->cap; i++) {
        if (jt->v[i].tid == UINT32_MAX) continue;
        lj_rec_t *r = &jt->v[i];
        if (keep_rid >= 0 && (int)r->rid == keep_rid) {
            lj_rec_t *dst = lj_get(&next, r->tid, r->rid, r->a, r->b);
            *dst = *r;
            continue;
        }
        if (jf_unw) {
            fprintf(jf_unw, "%s\t%s\t%u\t%u\t%u\t%u\t%u\n",
                    runs->names[r->rid], ctg[r->tid].name, r->a, r->b,
                    r->n_total, r->n_ua, r->max_anchor);
        }
        if (jf_ka) {
            u64 kt = r->n_total ? (u64)floor((double)r->ka_total + 1e-6) : 0;
            if (r->n_total && kt < 1) kt = 1;
            u64 ku = r->n_ua ? (u64)floor((double)r->ka_ua + 1e-6) : 0;
            if (r->n_ua && ku < 1) ku = 1;
            fprintf(jf_ka, "%s\t%s\t%u\t%u\t%lu\t%lu\t%u\n",
                    runs->names[r->rid], ctg[r->tid].name, r->a, r->b,
                    (unsigned long)kt, (unsigned long)ku, r->max_anchor);
        }
        emitted++;
    }
    free(jt->v);
    *jt = next;
    return emitted;
}

static int cmd_logan(int argc, char **argv) {
    const char *j_unw = NULL, *j_ka = NULL, *b_unw = NULL, *b_ka = NULL, *st_p = NULL;
    int binsz = 1000, mapq_u = 10, min_anchor = 8, no_out = 0;
    u64 max_junc_mem = 262144;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--junc-unw")) { if (i + 1 >= argc) die("logan: missing --junc-unw arg"); j_unw = argv[++i]; }
        else if (!strcmp(argv[i], "--junc-ka")) { if (i + 1 >= argc) die("logan: missing --junc-ka arg"); j_ka = argv[++i]; }
        else if (!strcmp(argv[i], "--bins-unw")) { if (i + 1 >= argc) die("logan: missing --bins-unw arg"); b_unw = argv[++i]; }
        else if (!strcmp(argv[i], "--bins-ka")) { if (i + 1 >= argc) die("logan: missing --bins-ka arg"); b_ka = argv[++i]; }
        else if (!strcmp(argv[i], "--stats")) { if (i + 1 >= argc) die("logan: missing --stats arg"); st_p = argv[++i]; }
        else if (!strcmp(argv[i], "--bin")) { if (i + 1 >= argc) die("logan: missing --bin arg"); binsz = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--mapq-uniq")) { if (i + 1 >= argc) die("logan: missing --mapq-uniq arg"); mapq_u = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--min-anchor")) { if (i + 1 >= argc) die("logan: missing --min-anchor arg"); min_anchor = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--flush-threshold")) { if (i + 1 >= argc) die("logan: missing --flush-threshold arg"); max_junc_mem = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--no-out")) no_out = 1;
        else die("logan: unknown flag");
    }
    if (binsz < 1) die("logan: --bin must be >= 1");
    if (max_junc_mem < 64) max_junc_mem = 64;

    double t0 = now_s();
    setvbuf(stdin, NULL, _IOFBF, 1 << 22);
    static char obuf[1 << 22];
    setvbuf(stdout, obuf, _IOFBF, sizeof(obuf));

    FILE *jf_unw = j_unw ? fopen_checked(j_unw, "w", "logan") : NULL;
    FILE *jf_ka = j_ka ? fopen_checked(j_ka, "w", "logan") : NULL;

    ctg_t *ctg = NULL;
    long nctg = 0, cctg = 0;
    u64 nbins = 0;
    u64 *bins_u = NULL, *bins_k = NULL;
    lj_tab_t jt;
    lj_init(&jt, 1 << 14);
    lj_runs_t runs;
    lj_runs_init(&runs);
    u64 total_junc_emitted = 0;
    int prev_rid = -1;

    int jcap = 512;
    long *jA = malloc((size_t)jcap * sizeof(long));
    long *jB = malloc((size_t)jcap * sizeof(long));
    long *blks = malloc((size_t)(jcap + 2) * sizeof(long));
    if (!jA || !jB || !blks) die("logan: cigar oom");

    long nrec = 0, nmapped = 0, n_unk = 0, nprim = 0, nuniq = 0, nspl = 0;
    char *line = NULL;
    size_t lc = 0;
    ssize_t ln;
    int hdr_done = 0;
    long last_tid = -1;
    char *last_name = NULL;
    long *cidx = NULL;
    u64 cmask = 0;

    while ((ln = getline(&line, &lc, stdin)) > 0) {
        if (line[0] == '@') {
            if (!strncmp(line, "@SQ", 3)) {
                char *sn = strstr(line, "SN:"), *lnp = strstr(line, "LN:");
                if (sn && lnp) {
                    if (nctg == cctg) {
                        cctg = cctg ? cctg * 2 : 256;
                        ctg = realloc(ctg, (size_t)cctg * sizeof(ctg_t));
                        if (!ctg) die("logan: ctg oom");
                    }
                    char *e = sn + 3;
                    while (*e && *e != '\t' && *e != '\n' && *e != '\r') e++;
                    ctg[nctg].name = strndup(sn + 3, (size_t)(e - sn - 3));
                    if (!ctg[nctg].name) die("logan: ctg name oom");
                    ctg[nctg].len = strtoull(lnp + 3, NULL, 10);
                    ctg[nctg].off = nbins;
                    nbins += ctg[nctg].len / (u64)binsz + 1;
                    nctg++;
                }
            }
            if (!no_out) fwrite(line, 1, (size_t)ln, stdout);
            continue;
        }
        if (!hdr_done) {
            hdr_done = 1;
            u64 cc = 64;
            while (cc < (u64)nctg * 2 + 2) cc <<= 1;
            cmask = cc - 1;
            cidx = malloc((size_t)cc * sizeof(long));
            if (!cidx) die("logan: cidx oom");
            for (u64 i = 0; i < cc; i++) cidx[i] = -1;
            for (long t = 0; t < nctg; t++) {
                u64 h = 1469598103934665603ULL;
                for (const char *q = ctg[t].name; *q; q++) { h ^= (u8)*q; h *= 1099511628211ULL; }
                u64 i = mix64(h) & cmask;
                while (cidx[i] >= 0) i = (i + 1) & cmask;
                cidx[i] = t;
            }
            bins_u = calloc(nbins + 1, sizeof(u64));
            bins_k = calloc(nbins + 1, sizeof(u64));
            if (!bins_u || !bins_k) die("logan: bins oom");
        }
        nrec++;
        char *f[12];
        int nf = 0;
        char *p = line;
        f[nf++] = p;
        while (*p && nf < 12) {
            if (*p == '\t') {
                *p = 0;
                f[nf++] = p + 1;
            }
            p++;
        }
        if (nf < 11) continue;
        int flag = atoi(f[1]);
        int is_secondary_or_supp = (flag & 0x100) || (flag & 0x800);
        int rid = lj_get_run_id(&runs, f[0]);
        if (!is_secondary_or_supp) runs.n_recs[rid]++;
        if (flag & 4) continue;  // drop unmapped BEFORE samtools sort

        const char *rname = f[2];
        long tid = -1;
        if (last_tid >= 0 && !strcmp(last_name, rname)) {
            tid = last_tid;
        } else {
            u64 h = 1469598103934665603ULL;
            for (const char *q = rname; *q; q++) { h ^= (u8)*q; h *= 1099511628211ULL; }
            u64 i = mix64(h) & cmask;
            while (cidx[i] >= 0) {
                if (!strcmp(ctg[cidx[i]].name, rname)) { tid = cidx[i]; break; }
                i = (i + 1) & cmask;
            }
            if (tid >= 0) { last_tid = tid; last_name = ctg[tid].name; }
        }
        if (tid < 0) { n_unk++; continue; }

        nmapped++;
        long pos = atol(f[3]) - 1;
        if (pos < 0) pos = 0;
        int mapq = atoi(f[4]);
        int primary = !is_secondary_or_supp;

        double ka = 1.0;
        if (nf == 12) {
            const char *opt = f[11];
            if (!strncmp(opt, "ka:f:", 5) || !strncmp(opt, "ka:i:", 5)) {
                ka = strtod(opt + 5, NULL);
            } else {
                const char *kp = strstr(opt, "\tka:f:");
                if (!kp) kp = strstr(opt, "\tka:i:");
                if (kp) ka = strtod(kp + 6, NULL);
            }
            if (!isfinite(ka) || !(ka > 0.0)) ka = 1.0;
        }

        if (primary) {
            runs.n_mapped[rid]++;
            if (prev_rid >= 0 && rid != prev_rid && jt.n >= max_junc_mem) {
                total_junc_emitted += lj_flush_table(&jt, rid, jf_unw, jf_ka, &runs, ctg);
            } else if (jt.n >= max_junc_mem * 4) {
                total_junc_emitted += lj_flush_table(&jt, rid, jf_unw, jf_ka, &runs, ctg);
            }
            prev_rid = rid;

            const char *c = f[5];
            long rp = pos;
            int spliced = 0;
            int nb = 0;
            long blk = 0;
            int nj = 0;
            while (*c && *c != '\t') {
                long n = 0;
                while (*c >= '0' && *c <= '9') { n = n * 10 + (*c - '0'); c++; }
                if (!*c || *c == '\t') break;
                char op = *c++;
                switch (op) {
                    case 'M': case '=': case 'X': blk += n; rp += n; break;
                    case 'D': rp += n; break;
                    case 'N':
                        if (nj >= jcap) {
                            jcap *= 2;
                            jA = realloc(jA, (size_t)jcap * sizeof(long));
                            jB = realloc(jB, (size_t)jcap * sizeof(long));
                            blks = realloc(blks, (size_t)(jcap + 2) * sizeof(long));
                            if (!jA || !jB || !blks) die("logan: cigar realloc oom");
                        }
                        jA[nj] = rp;
                        jB[nj] = rp + n;
                        blks[nb++] = blk;
                        nj++;
                        blk = 0;
                        rp += n;
                        spliced = 1;
                        break;
                    default: break;
                }
            }
            blks[nb++] = blk;
            u64 ctg_nb = ctg[tid].len / (u64)binsz + 1;
            u64 b_off = (u64)pos / (u64)binsz;
            if (b_off >= ctg_nb) b_off = ctg_nb - 1;
            u64 bin = ctg[tid].off + b_off;
            nprim++;
            if (mapq >= mapq_u) {
                bins_u[bin] += 1;
                u64 kw = (u64)floor(ka + 1e-6);
                if (kw < 1) kw = 1;
                bins_k[bin] += kw;
                nuniq++;
            }
            if (spliced) nspl++;
            for (int q = 0; q < nj; q++) {
                lj_rec_t *r = lj_get(&jt, (u32)tid, (u32)rid, (u32)jA[q], (u32)jB[q]);
                int an = (int)(blks[q] < blks[q + 1] ? blks[q] : blks[q + 1]);
                r->n_total++;
                r->ka_total += (float)ka;
                if (mapq >= mapq_u && an >= min_anchor) {
                    r->n_ua++;
                    r->ka_ua += (float)ka;
                }
                if ((u32)an > r->max_anchor) r->max_anchor = (u32)an;
            }
        }

        if (!no_out) {
            for (int q = 0; q < nf - 1; q++) f[q + 1][-1] = '\t';
            fwrite(line, 1, (size_t)ln, stdout);
        }
    }
    if (!no_out) close_checked(stdout, "stdout", "logan");

    total_junc_emitted += lj_flush_table(&jt, -1, jf_unw, jf_ka, &runs, ctg);
    if (jf_unw) close_checked(jf_unw, j_unw, "logan");
    if (jf_ka) close_checked(jf_ka, j_ka, "logan");

    if (b_unw) {
        FILE *bf = fopen_checked(b_unw, "w", "logan");
        if (bins_u) {
            for (long i = 0; i < nctg; i++) {
                u64 nb = ctg[i].len / (u64)binsz + 1;
                for (u64 q = 0; q < nb; q++) {
                    u64 v = bins_u[ctg[i].off + q];
                    if (v) fprintf(bf, "%s\t%lu\t%lu\n", ctg[i].name, (unsigned long)q, (unsigned long)v);
                }
            }
        }
        close_checked(bf, b_unw, "logan");
    }

    if (b_ka) {
        FILE *bf = fopen_checked(b_ka, "w", "logan");
        if (bins_k) {
            for (long i = 0; i < nctg; i++) {
                u64 nb = ctg[i].len / (u64)binsz + 1;
                for (u64 q = 0; q < nb; q++) {
                    u64 v = bins_k[ctg[i].off + q];
                    if (v) fprintf(bf, "%s\t%lu\t%lu\n", ctg[i].name, (unsigned long)q, (unsigned long)v);
                }
            }
        }
        close_checked(bf, b_ka, "logan");
    }

    double elapsed = now_s() - t0;
    FILE *sf = st_p ? fopen_checked(st_p, "w", "logan") : stderr;
    fprintf(sf, "{\"records\":%ld,\"mapped_records\":%ld,\"unknown_rname\":%ld,\"n_sq\":%ld,\"primary\":%ld,\"primary_unique\":%ld,\"spliced_primary\":%ld,\"per_run_junc_entries\":%lu,\"n_runs\":%d,\"run_records\":{",
            nrec, nmapped, n_unk, nctg, nprim, nuniq, nspl, (unsigned long)total_junc_emitted, runs.n);
    for (int ri = 0; ri < runs.n; ri++) {
        if (ri > 0) fputc(',', sf);
        fprintf(sf, "\"%s\":%lu", runs.names[ri], (unsigned long)runs.n_recs[ri]);
    }
    fprintf(sf, "},\"run_mapped\":{");
    for (int ri = 0; ri < runs.n; ri++) {
        if (ri > 0) fputc(',', sf);
        fprintf(sf, "\"%s\":%lu", runs.names[ri], (unsigned long)runs.n_mapped[ri]);
    }
    fprintf(sf, "},\"elapsed_s\":%.3f}\n", elapsed);
    if (st_p) close_checked(sf, st_p, "logan");

    free(line);
    free(jA);
    free(jB);
    free(blks);
    free(cidx);
    free(bins_u);
    free(bins_k);
    free(jt.v);
    for (long i = 0; i < nctg; i++) free(ctg[i].name);
    free(ctg);
    lj_runs_free(&runs);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: glcore {syncmers|mult|activate|sieve|junc|logan} ...\n"); return 1; }
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-k") && i + 1 < argc) K = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "-s") && i + 1 < argc) S = atoi(argv[i + 1]);
    }
    if (K > 31 || S >= K || S < 1) die("bad k/s (need s < k <= 31)");
    const char *c = argv[1];
    if (!strcmp(c, "syncmers")) return cmd_syncmers(argc - 1, argv + 1);
    if (!strcmp(c, "mult")) return cmd_mult(argc - 1, argv + 1);
    if (!strcmp(c, "activate")) return cmd_activate(argc - 1, argv + 1);
    if (!strcmp(c, "sieve")) return cmd_sieve(argc - 1, argv + 1);
    if (!strcmp(c, "junc")) return cmd_junc(argc - 1, argv + 1);
    if (!strcmp(c, "logan")) return cmd_logan(argc - 1, argv + 1);
    fprintf(stderr, "unknown subcommand %s\n", c);
    return 1;
}
