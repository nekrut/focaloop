#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *key; int32_t val; } StrEntry;
typedef struct { StrEntry *t; size_t cap, len; } StrMap;

static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) { h ^= (uint8_t)*s; h *= 1099511628211ULL; }
    return h;
}
static void sm_init(StrMap *m, size_t cap) {
    m->cap = cap; m->len = 0;
    m->t = calloc(cap, sizeof(StrEntry));
}
static void sm_grow(StrMap *m) {
    StrMap nm; sm_init(&nm, m->cap * 2);
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->t[i].key) continue;
        size_t idx = (size_t)(fnv1a(m->t[i].key) & (nm.cap - 1));
        while (nm.t[idx].key) idx = (idx + 1) & (nm.cap - 1);
        nm.t[idx] = m->t[i]; nm.len++;
    }
    free(m->t); *m = nm;
}
static int32_t sm_get(const StrMap *m, const char *k) {
    size_t idx = (size_t)(fnv1a(k) & (m->cap - 1));
    while (m->t[idx].key) {
        if (strcmp(m->t[idx].key, k) == 0) return m->t[idx].val;
        idx = (idx + 1) & (m->cap - 1);
    }
    return -1;
}
static int32_t sm_put_if_absent(StrMap *m, const char *k, int32_t v) {
    if (m->len * 2 >= m->cap) sm_grow(m);
    size_t idx = (size_t)(fnv1a(k) & (m->cap - 1));
    while (m->t[idx].key) {
        if (strcmp(m->t[idx].key, k) == 0) return m->t[idx].val;
        idx = (idx + 1) & (m->cap - 1);
    }
    m->t[idx].key = strdup(k);
    m->t[idx].val = v;
    m->len++;
    return v;
}

typedef struct { int32_t off, len; double weight; } EcRec;

typedef struct {
    int32_t gid;
    int as_val;
    int aln;
    int q0, q1;
} Hit;

static int cmp_i32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_hit_as_desc(const void *a, const void *b) {
    const Hit *x = (const Hit *)a, *y = (const Hit *)b;
    if (y->as_val != x->as_val) return y->as_val - x->as_val;
    return y->aln - x->aln;
}

/* Parse CIGAR to extract aligned query span [q0, q1) in forward unitig coordinates */
static void parse_cigar_qspan(const char *cigar, int rev, int *out_aln, int *out_q0, int *out_q1) {
    int num = 0, aln = 0, lead = 0, trail = 0, seen_aln = 0;
    for (const char *p = cigar; *p; p++) {
        if (*p >= '0' && *p <= '9') {
            num = num * 10 + (*p - '0');
        } else {
            char op = *p;
            if (op == 'S' || op == 'H') {
                if (!seen_aln) lead += num;
                else trail += num;
            } else if (op == 'M' || op == '=' || op == 'X' || op == 'I') {
                seen_aln = 1;
                aln += num;
            } else if (op == 'D' || op == 'N') {
                seen_aln = 1;
            }
            num = 0;
        }
    }
    int qlen = lead + aln + trail;
    int q0 = rev ? trail : lead;
    int q1 = q0 + aln;
    if (q1 > qlen) q1 = qlen;
    *out_aln = aln;
    *out_q0 = q0;
    *out_q1 = q1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: expr_sam_ec_v2 <tx2gene.tsv> <out_gene_bp.tsv>\n");
        return 1;
    }
    FILE *tf = fopen(argv[1], "r");
    if (!tf) { perror("tx2gene"); return 1; }

    StrMap tx2g, g2id;
    sm_init(&tx2g, 1 << 18);
    sm_init(&g2id, 1 << 17);
    size_t g_cap = 1 << 16, n_genes = 0;
    char **g_names = malloc(g_cap * sizeof(char *));

    static char line[1 << 20];
    int first = 1;
    while (fgets(line, sizeof(line), tf)) {
        char *nl = strpbrk(line, "\r\n"); if (nl) *nl = '\0';
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        const char *tx = line, *gn = tab + 1;
        if (first && (strcmp(tx, "transcript_id") == 0 || strcmp(tx, "tx") == 0)) { first = 0; continue; }
        first = 0;
        int32_t gid = sm_get(&g2id, gn);
        if (gid < 0) {
            if (n_genes >= g_cap) { g_cap *= 2; g_names = realloc(g_names, g_cap * sizeof(char *)); }
            gid = (int32_t)n_genes;
            g_names[n_genes++] = strdup(gn);
            sm_put_if_absent(&g2id, gn, gid);
        }
        sm_put_if_absent(&tx2g, tx, gid);
    }
    fclose(tf);

    double *uniq_bp = calloc(n_genes, sizeof(double));
    double *tot_bp = calloc(n_genes, sizeof(double));

    size_t ec_cap = 1 << 18, n_ec = 0;
    EcRec *ecs = malloc(ec_cap * sizeof(EcRec));
    size_t pool_cap = 1 << 20, pool_len = 0;
    int32_t *pool = malloc(pool_cap * sizeof(int32_t));

    char cur_q[256] = "";
    Hit hits[2048];
    int n_hits = 0;
    double cur_ka = 0.0;
    long long sam_lines = 0, mapped_seg = 0, uniq_seg = 0, multi_seg = 0, multi_seg_unitigs = 0;

    #define FLUSH_Q() do { \
        if (n_hits > 0) { \
            qsort(hits, n_hits, sizeof(Hit), cmp_hit_as_desc); \
            int used[2048]; memset(used, 0, (size_t)n_hits * sizeof(int)); \
            int n_clusters = 0; \
            for (int _s = 0; _s < n_hits; _s++) { \
                if (used[_s]) continue; \
                int sq0 = hits[_s].q0, sq1 = hits[_s].q1; \
                int best_as = hits[_s].as_val, best_aln = hits[_s].aln; \
                int32_t kept_g[256]; int nk = 0; \
                for (int _i = _s; _i < n_hits; _i++) { \
                    if (used[_i]) continue; \
                    int ov = (hits[_i].q1 < sq1 ? hits[_i].q1 : sq1) - (hits[_i].q0 > sq0 ? hits[_i].q0 : sq0); \
                    int minl = (hits[_i].q1 - hits[_i].q0) < (sq1 - sq0) ? (hits[_i].q1 - hits[_i].q0) : (sq1 - sq0); \
                    if (ov > 0 && ov * 2 >= minl) { \
                        used[_i] = 1; \
                        if (hits[_i].as_val >= best_as - 2 && nk < 256) kept_g[nk++] = hits[_i].gid; \
                    } \
                } \
                if (nk == 0) continue; \
                qsort(kept_g, nk, sizeof(int32_t), cmp_i32); \
                int nu = 0; \
                for (int _i = 0; _i < nk; _i++) { \
                    if (nu == 0 || kept_g[_i] != kept_g[nu - 1]) kept_g[nu++] = kept_g[_i]; \
                } \
                double mid_len = (best_aln > 30) ? (double)(best_aln - 30) : 1.0; \
                double w = cur_ka * mid_len; \
                mapped_seg++; \
                n_clusters++; \
                if (nu == 1) { \
                    uniq_bp[kept_g[0]] += w; \
                    uniq_seg++; \
                } else if (nu > 1) { \
                    if (n_ec >= ec_cap) { ec_cap *= 2; ecs = realloc(ecs, ec_cap * sizeof(EcRec)); } \
                    if (pool_len + (size_t)nu > pool_cap) { while (pool_len + (size_t)nu > pool_cap) pool_cap *= 2; pool = realloc(pool, pool_cap * sizeof(int32_t)); } \
                    ecs[n_ec].off = (int32_t)pool_len; \
                    ecs[n_ec].len = nu; \
                    ecs[n_ec].weight = w; \
                    memcpy(pool + pool_len, kept_g, (size_t)nu * sizeof(int32_t)); \
                    pool_len += (size_t)nu; \
                    n_ec++; \
                    multi_seg++; \
                } \
            } \
            if (n_clusters > 1) multi_seg_unitigs++; \
            n_hits = 0; \
        } \
    } while (0)

    setvbuf(stdin, NULL, _IOFBF, 4 << 20);
    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '@') continue;
        sam_lines++;
        char *f[12];
        int nf = 0;
        char *p = line;
        while (nf < 11) {
            f[nf++] = p;
            char *t = strchr(p, '\t');
            if (!t) break;
            *t = '\0';
            p = t + 1;
        }
        if (nf < 11) continue;
        int flag = atoi(f[1]);
        if (flag & 4) continue; /* skip unmapped ONLY; keep secondary (256) and supplementary (2048) for operons! */
        int32_t gid = sm_get(&tx2g, f[2]);
        if (gid < 0) continue;

        if (strcmp(f[0], cur_q) != 0) {
            FLUSH_Q();
            strncpy(cur_q, f[0], sizeof(cur_q) - 1);
            cur_q[sizeof(cur_q) - 1] = '\0';
            char *us = strrchr(cur_q, '_');
            cur_ka = us ? (atof(us + 1) / 100.0) : 1.0;
        }
        int aln = 0, q0 = 0, q1 = 0;
        parse_cigar_qspan(f[5], (flag & 16) != 0, &aln, &q0, &q1);
        if (aln < 25) continue;
        int as_val = aln;
        char *as_p = strstr(p, "AS:i:");
        if (as_p) as_val = atoi(as_p + 5);
        if (n_hits < 2048) {
            hits[n_hits].gid = gid;
            hits[n_hits].as_val = as_val;
            hits[n_hits].aln = aln;
            hits[n_hits].q0 = q0;
            hits[n_hits].q1 = q1;
            n_hits++;
        }
    }
    FLUSH_Q();

    /* 20-iteration sparse EM over multi-gene equivalence classes */
    for (size_t g = 0; g < n_genes; g++) tot_bp[g] = uniq_bp[g] + 1e-6;
    for (size_t e = 0; e < n_ec; e++) {
        double share = ecs[e].weight / (double)ecs[e].len;
        for (int j = 0; j < ecs[e].len; j++) tot_bp[pool[ecs[e].off + j]] += share;
    }
    double *nxt = malloc(n_genes * sizeof(double));
    for (int it = 0; it < 20; it++) {
        for (size_t g = 0; g < n_genes; g++) nxt[g] = uniq_bp[g];
        for (size_t e = 0; e < n_ec; e++) {
            double denom = 0.0;
            for (int j = 0; j < ecs[e].len; j++) denom += tot_bp[pool[ecs[e].off + j]];
            if (denom <= 0.0) {
                double share = ecs[e].weight / (double)ecs[e].len;
                for (int j = 0; j < ecs[e].len; j++) nxt[pool[ecs[e].off + j]] += share;
            } else {
                double inv = ecs[e].weight / denom;
                for (int j = 0; j < ecs[e].len; j++) {
                    int32_t gi = pool[ecs[e].off + j];
                    nxt[gi] += tot_bp[gi] * inv;
                }
            }
        }
        for (size_t g = 0; g < n_genes; g++) tot_bp[g] = nxt[g] + 1e-9;
    }

    FILE *of = fopen(argv[2], "w");
    if (!of) { perror("out"); return 1; }
    fprintf(of, "gene\tmidpoint_bp\tuniq_bp\n");
    for (size_t g = 0; g < n_genes; g++) {
        if (nxt[g] > 0.0) fprintf(of, "%s\t%.3f\t%.3f\n", g_names[g], nxt[g], uniq_bp[g]);
    }
    fclose(of);
    fprintf(stderr, "{\"sam_lines\": %lld, \"mapped_seg\": %lld, \"uniq_seg\": %lld, \"multi_seg\": %lld, \"multi_seg_unitigs\": %lld}\n",
            sam_lines, mapped_seg, uniq_seg, multi_seg, multi_seg_unitigs);
    return 0;
}
