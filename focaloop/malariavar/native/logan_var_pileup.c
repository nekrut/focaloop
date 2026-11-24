#define _GNU_SOURCE
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Organism-agnostic, interval-backed ka-weighted SAM pileup + indel extractor
 * for Focaloop VariationLoop 0.4.1 Logan unitigs.
 *
 * Usage:
 *   logan_var_pileup <ref.fa> <targets.bed|-> <out_cand.tsv> [min_mapq=20] [max_mm_frac=0.12] [midpoint_only=1] [out_cov.tsv]
 */

static const uint8_t ENC[256] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4, 4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
};
static const char DEC[6] = "ACGTN";

typedef struct {
    char *name;
    char *seq;
    int64_t len;
    int64_t reg_start;
    int64_t reg_count;
} Contig;

typedef struct {
    int32_t cid;
    int64_t start0;
    int64_t end0;
    int64_t pile_off;
} Region;

typedef struct {
    int32_t cid;
    int64_t pos0;
    int64_t r_end0;
    uint8_t kind; /* 1=DEL, 2=INS */
    int32_t alen;
    char *seq;    /* insertion seq (kind==2) or NULL (kind==1) */
    double weight;
    uint64_t hash;
    uint64_t last_uid_hash;
    int used;
} IndelEntry;

typedef struct {
    IndelEntry *t;
    size_t cap;
    size_t len;
} IndelMap;

static uint64_t fnv1a_bytes(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t unitig_parent_hash(const char *qname) {
    /* Collapse tiled headers `<uid>c<tile>_<ka>` to `<uid>` so overlapping 30-bp tiles never double-count an indel */
    const char *us = strrchr(qname, '_');
    size_t end = us ? (size_t)(us - qname) : strlen(qname);
    if (end > 2) {
        size_t j = end;
        while (j > 0 && isdigit((unsigned char)qname[j - 1])) j--;
        if (j < end && j > 1 && qname[j - 1] == 'c' && isdigit((unsigned char)qname[j - 2])) {
            end = j - 1;
        }
    }
    uint64_t h = fnv1a_bytes(qname, end);
    return h ? h : 1ULL;
}

static uint64_t hash_indel(int32_t cid, int64_t pos0, uint8_t kind, int32_t alen, const char *seq) {
    uint64_t h = 1469598103934665603ULL;
    h ^= (uint64_t)(uint32_t)cid; h *= 1099511628211ULL;
    h ^= (uint64_t)pos0; h *= 1099598103934665603ULL;
    h ^= (uint64_t)kind; h *= 1099511628211ULL;
    h ^= (uint64_t)(uint32_t)alen; h *= 1099511628211ULL;
    if (kind == 2 && seq && alen > 0) {
        h ^= fnv1a_bytes(seq, (size_t)alen);
        h *= 1099511628211ULL;
    }
    return h;
}

static void im_init(IndelMap *m, size_t cap) {
    m->cap = cap;
    m->len = 0;
    m->t = calloc(cap, sizeof(IndelEntry));
}

static void im_grow(IndelMap *m) {
    IndelMap nm;
    im_init(&nm, m->cap * 2);
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->t[i].used) continue;
        size_t idx = (size_t)(m->t[i].hash & (nm.cap - 1));
        while (nm.t[idx].used) idx = (idx + 1) & (nm.cap - 1);
        nm.t[idx] = m->t[i];
        nm.len++;
    }
    free(m->t);
    *m = nm;
}

static int im_add(IndelMap *m, int32_t cid, int64_t pos0, int64_t r_end0, uint8_t kind, int32_t alen,
                  const char *seq, double w, uint64_t uid_hash) {
    if (m->len * 2 >= m->cap) im_grow(m);
    if (r_end0 < pos0) r_end0 = pos0;
    uint64_t h = hash_indel(cid, pos0, kind, alen, seq);
    size_t idx = (size_t)(h & (m->cap - 1));
    while (m->t[idx].used) {
        IndelEntry *e = &m->t[idx];
        if (e->hash == h && e->cid == cid && e->pos0 == pos0 && e->kind == kind && e->alen == alen) {
            if (kind == 1 || (e->seq && seq && memcmp(e->seq, seq, (size_t)alen) == 0)) {
                if (r_end0 > e->r_end0) e->r_end0 = r_end0;
                if (uid_hash != 0ULL && e->last_uid_hash == uid_hash) {
                    return 0;
                }
                e->weight += w;
                e->last_uid_hash = uid_hash;
                return 1;
            }
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    IndelEntry *e = &m->t[idx];
    e->cid = cid;
    e->pos0 = pos0;
    e->r_end0 = r_end0;
    e->kind = kind;
    e->alen = alen;
    if (kind == 2 && seq && alen > 0) {
        e->seq = malloc((size_t)alen + 1);
        memcpy(e->seq, seq, (size_t)alen);
        e->seq[alen] = '\0';
    } else {
        e->seq = NULL;
    }
    e->weight = w;
    e->hash = h;
    e->last_uid_hash = uid_hash;
    e->used = 1;
    m->len++;
    return 1;
}

typedef struct { char *key; int32_t val; } CtgEntry;
typedef struct { CtgEntry *t; size_t cap, len; } CtgMap;

static void cm_init(CtgMap *m, size_t cap) {
    m->cap = cap; m->len = 0;
    m->t = calloc(cap, sizeof(CtgEntry));
}
static void cm_free(CtgMap *m) {
    for (size_t i = 0; i < m->cap; i++) free(m->t[i].key);
    free(m->t);
    m->t = NULL; m->cap = 0; m->len = 0;
}
static void cm_grow(CtgMap *m) {
    CtgMap nm; cm_init(&nm, m->cap * 2);
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->t[i].key) continue;
        size_t idx = (size_t)(fnv1a_bytes(m->t[i].key, strlen(m->t[i].key)) & (nm.cap - 1));
        while (nm.t[idx].key) idx = (idx + 1) & (nm.cap - 1);
        nm.t[idx] = m->t[i]; nm.len++;
    }
    free(m->t); *m = nm;
}
static void cm_put(CtgMap *m, const char *k, int32_t v) {
    if (m->len * 2 >= m->cap) cm_grow(m);
    size_t idx = (size_t)(fnv1a_bytes(k, strlen(k)) & (m->cap - 1));
    while (m->t[idx].key) {
        if (strcmp(m->t[idx].key, k) == 0) { m->t[idx].val = v; return; }
        idx = (idx + 1) & (m->cap - 1);
    }
    m->t[idx].key = strdup(k);
    m->t[idx].val = v;
    m->len++;
}
static int32_t cm_get(const CtgMap *m, const char *k) {
    size_t idx = (size_t)(fnv1a_bytes(k, strlen(k)) & (m->cap - 1));
    while (m->t[idx].key) {
        if (strcmp(m->t[idx].key, k) == 0) return m->t[idx].val;
        idx = (idx + 1) & (m->cap - 1);
    }
    return -1;
}

static int cmp_region(const void *a, const void *b) {
    const Region *x = (const Region *)a, *y = (const Region *)b;
    if (x->cid != y->cid) return (x->cid > y->cid) - (x->cid < y->cid);
    if (x->start0 != y->start0) return (x->start0 > y->start0) - (x->start0 < y->start0);
    return (x->end0 > y->end0) - (x->end0 < y->end0);
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static inline int64_t map_pos_to_pile(const Contig *ctg, const Region *regs, int64_t pos0) {
    int64_t lo = ctg->reg_start;
    int64_t hi = lo + ctg->reg_count - 1;
    while (lo <= hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        if (pos0 < regs[mid].start0) {
            hi = mid - 1;
        } else if (pos0 >= regs[mid].end0) {
            lo = mid + 1;
        } else {
            return regs[mid].pile_off + (pos0 - regs[mid].start0);
        }
    }
    return -1;
}

static double parse_ka_from_qname(const char *qname) {
    const char *us = strrchr(qname, '_');
    if (!us || !us[1]) return 1.0;
    double v = atof(us + 1) / 100.0;
    return (v > 0.01) ? v : 0.01;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: logan_var_pileup <ref.fa> <targets.bed|-> <out_cand.tsv> [min_mapq=20] [max_mm_frac=0.12] [midpoint_only=1] [out_cov.tsv]\n");
        return 1;
    }
    const char *ref_path = argv[1];
    const char *bed_path = (argv[2][0] && strcmp(argv[2], "-") != 0) ? argv[2] : NULL;
    const char *out_path = argv[3];
    int min_mapq = (argc > 4) ? atoi(argv[4]) : 20;
    double max_mm_frac = (argc > 5) ? atof(argv[5]) : 0.12;
    int midpoint_only = (argc > 6) ? atoi(argv[6]) : 1;
    const char *cov_path = (argc > 7 && argv[7][0] && strcmp(argv[7], "-") != 0) ? argv[7] : NULL;

    static char line[1 << 20];

    /* 0. Pre-scan BED contigs (L80) so non-target chromosomes are never allocated into RAM */
    CtgMap want_cmap;
    int filter_by_bed = 0;
    if (bed_path) {
        cm_init(&want_cmap, 256);
        filter_by_bed = 1;
        FILE *bf0 = fopen(bed_path, "r");
        if (!bf0) { perror("fopen bed"); return 1; }
        while (fgets(line, sizeof(line), bf0)) {
            if (line[0] == '#' || line[0] == '\n' || strncmp(line, "track", 5) == 0 || strncmp(line, "browser", 7) == 0)
                continue;
            char cname[512];
            long long s0 = 0, e0 = 0;
            if (sscanf(line, "%511s %lld %lld", cname, &s0, &e0) < 3) continue;
            if (e0 > s0) cm_put(&want_cmap, cname, 1);
        }
        fclose(bf0);
    }

    /* 1. Load reference FASTA contigs (allocating sequence only for wanted contigs) */
    FILE *rf = fopen(ref_path, "r");
    if (!rf) { perror("fopen ref"); return 1; }
    size_t c_cap = 64, n_ctgs = 0;
    Contig *ctgs = calloc(c_cap, sizeof(Contig));
    CtgMap cmap;
    cm_init(&cmap, 1024);

    size_t seq_cap = 0, seq_len = 0;
    int32_t cur_cid = -1;
    int cur_keep_seq = 0;
    while (fgets(line, sizeof(line), rf)) {
        if (line[0] == '>') {
            if (cur_cid >= 0) ctgs[cur_cid].len = (int64_t)seq_len;
            char *p = line + 1;
            while (*p && isspace((unsigned char)*p)) p++;
            char *e = p;
            while (*e && !isspace((unsigned char)*e)) e++;
            *e = '\0';
            if (n_ctgs >= c_cap) {
                c_cap *= 2;
                ctgs = realloc(ctgs, c_cap * sizeof(Contig));
            }
            cur_cid = (int32_t)n_ctgs++;
            ctgs[cur_cid].name = strdup(p);
            cur_keep_seq = (!filter_by_bed) || (cm_get(&want_cmap, p) >= 0);
            seq_len = 0;
            if (cur_keep_seq) {
                seq_cap = 65536;
                ctgs[cur_cid].seq = malloc(seq_cap);
                ctgs[cur_cid].seq[0] = '\0';
            } else {
                seq_cap = 0;
                ctgs[cur_cid].seq = NULL;
            }
            ctgs[cur_cid].len = 0;
            ctgs[cur_cid].reg_start = 0;
            ctgs[cur_cid].reg_count = 0;
            cm_put(&cmap, p, cur_cid);
        } else if (cur_cid >= 0) {
            size_t l = strlen(line);
            while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) l--;
            if (cur_keep_seq) {
                if (seq_len + l + 1 > seq_cap) {
                    while (seq_len + l + 1 > seq_cap) seq_cap *= 2;
                    ctgs[cur_cid].seq = realloc(ctgs[cur_cid].seq, seq_cap);
                }
                for (size_t i = 0; i < l; i++) {
                    ctgs[cur_cid].seq[seq_len++] = (char)toupper((unsigned char)line[i]);
                }
                ctgs[cur_cid].seq[seq_len] = '\0';
            } else {
                seq_len += l;
            }
        }
    }
    if (cur_cid >= 0) ctgs[cur_cid].len = (int64_t)seq_len;
    fclose(rf);
    if (filter_by_bed) cm_free(&want_cmap);

    /* 2. Build merged pileup regions (from BED +-500 bp, or all contigs) */
    size_t r_cap = n_ctgs + 64, n_raw = 0;
    Region *raw_regs = malloc(r_cap * sizeof(Region));
    if (bed_path) {
        FILE *bf = fopen(bed_path, "r");
        if (!bf) { perror("fopen bed"); return 1; }
        while (fgets(line, sizeof(line), bf)) {
            if (line[0] == '#' || line[0] == '\n' || strncmp(line, "track", 5) == 0 || strncmp(line, "browser", 7) == 0)
                continue;
            char cname[512];
            long long s0 = 0, e0 = 0;
            if (sscanf(line, "%511s %lld %lld", cname, &s0, &e0) < 3) continue;
            int32_t cid = cm_get(&cmap, cname);
            if (cid < 0 || !ctgs[cid].seq) continue;
            int64_t ps = (s0 > 500) ? (s0 - 500) : 0;
            int64_t pe = (e0 + 500 < ctgs[cid].len) ? (e0 + 500) : ctgs[cid].len;
            if (pe <= ps) continue;
            if (n_raw >= r_cap) { r_cap *= 2; raw_regs = realloc(raw_regs, r_cap * sizeof(Region)); }
            raw_regs[n_raw++] = (Region){cid, ps, pe, 0};
        }
        fclose(bf);
    } else {
        for (size_t c = 0; c < n_ctgs; c++) {
            if (ctgs[c].len <= 0 || !ctgs[c].seq) continue;
            if (n_raw >= r_cap) { r_cap *= 2; raw_regs = realloc(raw_regs, r_cap * sizeof(Region)); }
            raw_regs[n_raw++] = (Region){(int32_t)c, 0, ctgs[c].len, 0};
        }
    }
    if (n_raw == 0) {
        fprintf(stderr, "No callable reference regions matched.\n");
        free(raw_regs);
        return 1;
    }
    qsort(raw_regs, n_raw, sizeof(Region), cmp_region);

    Region *regs = malloc(n_raw * sizeof(Region));
    size_t n_regs = 0;
    int64_t total_pile_bp = 0;
    for (size_t i = 0; i < n_raw; i++) {
        if (n_regs > 0 && regs[n_regs - 1].cid == raw_regs[i].cid && raw_regs[i].start0 <= regs[n_regs - 1].end0) {
            if (raw_regs[i].end0 > regs[n_regs - 1].end0) regs[n_regs - 1].end0 = raw_regs[i].end0;
        } else {
            regs[n_regs++] = raw_regs[i];
        }
    }
    free(raw_regs);

    for (size_t i = 0; i < n_regs; i++) {
        regs[i].pile_off = total_pile_bp;
        total_pile_bp += (regs[i].end0 - regs[i].start0);
        Contig *c = &ctgs[regs[i].cid];
        if (c->reg_count == 0) c->reg_start = (int64_t)i;
        c->reg_count++;
    }

    if (total_pile_bp > 400000000LL) {
        fprintf(stderr, "Pileup span %lld bp exceeds 400 Mb without target BED. Supply --targets-bed for large genomes.\n",
                (long long)total_pile_bp);
        return 2;
    }

    float *counts = calloc((size_t)total_pile_bp * 4ULL, sizeof(float));
    float *delc = calloc((size_t)total_pile_bp, sizeof(float));
    if (!counts || !delc) {
        fprintf(stderr, "OOM allocating pileup arrays for %lld bp\n", (long long)total_pile_bp);
        return 1;
    }
    IndelMap imap;
    im_init(&imap, 1 << 16);

    /* 3. Stream SAM from stdin */
    setvbuf(stdin, NULL, _IOFBF, 4 << 20);
    long long sam_records = 0, primary_mapped = 0, counted_unitigs = 0, counted_bases = 0, mm_filtered = 0;
    char ins_buf[256];
    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '@') continue;
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
        sam_records++;
        int flag = atoi(f[1]);
        if (flag & 0xF04) continue; /* unmapped, secondary, qcfail, dup, supplementary */
        int32_t cid = cm_get(&cmap, f[2]);
        if (cid < 0 || ctgs[cid].reg_count == 0 || !ctgs[cid].seq) continue;
        int64_t pos0 = atoll(f[3]) - 1;
        int mapq = atoi(f[4]);
        primary_mapped++;
        if (mapq < min_mapq) continue;

        /* First CIGAR pass for rspan & indel */
        int rspan = 0, indel = 0, num = 0;
        for (const char *cp = f[5]; *cp; cp++) {
            if (*cp >= '0' && *cp <= '9') {
                num = num * 10 + (*cp - '0');
            } else {
                char op = *cp;
                if (op == 'M' || op == '=' || op == 'X' || op == 'D' || op == 'N') rspan += num;
                if (op == 'I' || op == 'D') indel += num;
                num = 0;
            }
        }
        if (rspan <= 0) continue;

        /* NM tag filter */
        char *nm_p = strstr(p, "NM:i:");
        if (nm_p) {
            int nm = atoi(nm_p + 5);
            if ((nm - indel) > max_mm_frac * (double)rspan) {
                mm_filtered++;
                continue;
            }
        }

        const char *qseq = f[9];
        int qlen = (int)strlen(qseq);
        if (qlen <= 0 || qseq[0] == '*') continue;
        int q_lo = (midpoint_only && qlen > 30) ? 15 : 0;
        int q_hi = (midpoint_only && qlen > 30) ? (qlen - 15) : qlen;
        double w = parse_ka_from_qname(f[0]);
        float wf = (float)w;
        uint64_t uid_hash = unitig_parent_hash(f[0]);

        /* Walk CIGAR */
        int qp = 0;
        int64_t rpos = pos0;
        num = 0;
        int u_counted = 0;
        const Contig *ctg = &ctgs[cid];
        for (const char *cp = f[5]; *cp; cp++) {
            if (*cp >= '0' && *cp <= '9') {
                num = num * 10 + (*cp - '0');
                continue;
            }
            char op = *cp;
            if (op == 'M' || op == '=' || op == 'X') {
                for (int t = 0; t < num; t++) {
                    int qi = qp + t;
                    if (qi < q_lo || qi >= q_hi) continue;
                    uint8_t b = ENC[(uint8_t)qseq[qi]];
                    if (b < 4) {
                        int64_t idx = map_pos_to_pile(ctg, regs, rpos + t);
                        if (idx >= 0) {
                            counts[(size_t)idx * 4ULL + b] += wf;
                            u_counted++;
                        }
                    }
                }
                qp += num;
                rpos += num;
            } else if (op == 'I') {
                if (rpos > 0 && num > 0 && num <= 200 && qp + num <= qlen) {
                    /* Compute full repeat-equivalent placement interval [qp_left, qp_right] (#70) */
                    int qp_left = qp;
                    int64_t r_left = rpos;
                    while (qp_left > 0 && r_left > 1 &&
                           (char)toupper((unsigned char)qseq[qp_left + num - 1]) == ctg->seq[r_left - 1] &&
                           (char)toupper((unsigned char)qseq[qp_left - 1]) == ctg->seq[r_left - 1]) {
                        /* Cyclic left-shift of inserted sequence matches left-flanking ref base */
                        qp_left--;
                        r_left--;
                    }
                    int qp_right = qp;
                    int64_t r_right = rpos;
                    while (qp_right + num < qlen && r_right < ctg->len &&
                           (char)toupper((unsigned char)qseq[qp_right]) == ctg->seq[r_right] &&
                           (char)toupper((unsigned char)qseq[qp_right + num]) == ctg->seq[r_right]) {
                        qp_right++;
                        r_right++;
                    }
                    int ok_pos = (!midpoint_only) || (qlen <= 30) || (qp_right >= q_lo && qp_left < q_hi);
                    if (ok_pos && r_left > 0) {
                        int64_t idx = map_pos_to_pile(ctg, regs, r_left - 1);
                        if (idx >= 0) {
                            int ok_b = 1;
                            for (int t = 0; t < num; t++) {
                                uint8_t enc_b = ENC[(uint8_t)qseq[qp_left + t]];
                                if (enc_b > 3) { ok_b = 0; break; }
                                ins_buf[t] = DEC[enc_b];
                            }
                            ins_buf[num] = '\0';
                            if (ok_b) {
                                im_add(&imap, cid, r_left - 1, r_right, 2, num, ins_buf, w, uid_hash);
                            }
                        }
                    }
                }
                qp += num;
            } else if (op == 'D') {
                if (rpos > 0 && num > 0 && num <= 200 && rpos + num <= ctg->len) {
                    /* Compute full repeat-equivalent placement interval [qp_left, qp_right] (#70) */
                    int qp_left = qp;
                    int64_t r_left = rpos;
                    while (qp_left > 0 && r_left > 1 &&
                           ctg->seq[r_left - 1] == ctg->seq[r_left + num - 1] &&
                           (char)toupper((unsigned char)qseq[qp_left - 1]) == ctg->seq[r_left - 1]) {
                        qp_left--;
                        r_left--;
                    }
                    int qp_right = qp;
                    int64_t r_right = rpos;
                    while (qp_right < qlen && r_right + num < ctg->len &&
                           ctg->seq[r_right] == ctg->seq[r_right + num] &&
                           (char)toupper((unsigned char)qseq[qp_right]) == ctg->seq[r_right]) {
                        qp_right++;
                        r_right++;
                    }
                    int ok_pos = (!midpoint_only) || (qlen <= 30) || (qp_right >= q_lo && qp_left < q_hi);
                    if (ok_pos && r_left > 0) {
                        int64_t anchor_idx = map_pos_to_pile(ctg, regs, r_left - 1);
                        if (anchor_idx >= 0) {
                            if (im_add(&imap, cid, r_left, r_right, 1, num, NULL, w, uid_hash)) {
                                for (int t = 0; t < num; t++) {
                                    int64_t didx = map_pos_to_pile(ctg, regs, r_left + t);
                                    if (didx >= 0) delc[didx] += wf;
                                }
                            }
                        }
                    }
                }
                rpos += num;
            } else if (op == 'N') {
                rpos += num;
            } else if (op == 'S') {
                qp += num;
            }
            num = 0;
        }
        if (u_counted > 0) {
            counted_unitigs++;
            counted_bases += u_counted;
        }
    }

    /* 4. Emit candidate SNVs and indels + optional covered-intervals TSV (#62) + sampled median depth */
    FILE *of = fopen(out_path, "w");
    if (!of) { perror("fopen out"); return 1; }
    fprintf(of, "chrom\tpos\tref\talt\tkind\tdp\tad_ref\tad_alt\twsaf\n");

    FILE *cf = cov_path ? fopen(cov_path, "w") : NULL;

    size_t sample_cap = 200000, sample_n = 0;
    float *depth_samp = malloc(sample_cap * sizeof(float));
    int64_t stride = (total_pile_bp > (int64_t)sample_cap) ? (total_pile_bp / (int64_t)sample_cap) : 1;
    if (stride < 1) stride = 1;

    for (size_t ri = 0; ri < n_regs; ri++) {
        const Region *rg = &regs[ri];
        const Contig *ctg = &ctgs[rg->cid];
        int in_cov = 0;
        int64_t cov_s0 = 0;
        for (int64_t p0 = rg->start0; p0 < rg->end0; p0++) {
            int64_t idx = rg->pile_off + (p0 - rg->start0);
            const float *c = &counts[(size_t)idx * 4ULL];
            float tot = c[0] + c[1] + c[2] + c[3];
            float dp = tot + delc[idx];
            if (cf) {
                if (dp >= 1.0f) {
                    if (!in_cov) { cov_s0 = p0; in_cov = 1; }
                } else if (in_cov) {
                    fprintf(cf, "%s\t%lld\t%lld\n", ctg->name, (long long)cov_s0, (long long)p0);
                    in_cov = 0;
                }
            }
            if ((idx % stride) == 0 && dp > 0.0f && sample_n < sample_cap) {
                depth_samp[sample_n++] = dp;
            }
            if (tot < 2.0f) continue;
            uint8_t rb = ENC[(uint8_t)ctg->seq[p0]];
            if (rb > 3) continue;
            float rc = c[rb];
            if (tot - rc < 1.5f) continue;
            for (int b = 0; b < 4; b++) {
                if (b == (int)rb) continue;
                float ac = c[b];
                if (ac < 1.5f) continue;
                float wsaf = ac / tot;
                if (wsaf < 0.05f) continue;
                fprintf(of, "%s\t%lld\t%c\t%c\tSNV\t%.2f\t%.2f\t%.2f\t%.5f\n",
                        ctg->name, (long long)(p0 + 1), DEC[rb], DEC[b], dp, rc, ac, wsaf);
            }
        }
        if (cf && in_cov) {
            fprintf(cf, "%s\t%lld\t%lld\n", ctg->name, (long long)cov_s0, (long long)rg->end0);
        }
    }
    if (cf) fclose(cf);

    for (size_t i = 0; i < imap.cap; i++) {
        const IndelEntry *e = &imap.t[i];
        if (!e->used || e->weight < 1.5) continue;
        const Contig *ctg = &ctgs[e->cid];
        int64_t span_hi = e->r_end0;
        if (span_hi > e->pos0 + 256) span_hi = e->pos0 + 256;
        if (span_hi >= ctg->len) span_hi = ctg->len - 1;
        if (e->kind == 1) {
            if (e->pos0 <= 0 || e->pos0 + e->alen > ctg->len) continue;
            int64_t idx0 = map_pos_to_pile(ctg, regs, e->pos0);
            if (idx0 < 0) continue;
            double dp = 0.0;
            for (int64_t rp = e->pos0; rp <= span_hi; rp++) {
                int64_t ridx = map_pos_to_pile(ctg, regs, rp);
                if (ridx < 0) continue;
                const float *c = &counts[(size_t)ridx * 4ULL];
                double d_rp = (double)(c[0] + c[1] + c[2] + c[3] + delc[ridx]);
                if (d_rp > dp) dp = d_rp;
            }
            if (dp < e->weight) dp = e->weight;
            double wsaf = e->weight / fmax(dp, 1e-9);
            if (wsaf < 0.05) continue;
            double rc = fmax(dp - e->weight, 0.0);
            char anchor = ctg->seq[e->pos0 - 1];
            fprintf(of, "%s\t%lld\t%c%.*s\t%c\tINDEL\t%.2f\t%.2f\t%.2f\t%.5f\n",
                    ctg->name, (long long)e->pos0, anchor, e->alen, ctg->seq + e->pos0, anchor,
                    dp, rc, e->weight, wsaf);
        } else if (e->kind == 2 && e->seq) {
            if (e->pos0 < 0 || e->pos0 >= ctg->len) continue;
            int64_t idx0 = map_pos_to_pile(ctg, regs, e->pos0);
            if (idx0 < 0) continue;
            double dp = 0.0;
            for (int64_t rp = e->pos0; rp <= span_hi; rp++) {
                int64_t ridx = map_pos_to_pile(ctg, regs, rp);
                if (ridx < 0) continue;
                const float *c = &counts[(size_t)ridx * 4ULL];
                double d_rp = (double)(c[0] + c[1] + c[2] + c[3] + delc[ridx]);
                if (d_rp > dp) dp = d_rp;
            }
            if (dp < e->weight) dp = e->weight;
            double wsaf = e->weight / fmax(dp, 1e-9);
            if (wsaf < 0.05) continue;
            double rc = fmax(dp - e->weight, 0.0);
            char anchor = ctg->seq[e->pos0];
            fprintf(of, "%s\t%lld\t%c\t%c%s\tINDEL\t%.2f\t%.2f\t%.2f\t%.5f\n",
                    ctg->name, (long long)(e->pos0 + 1), anchor, anchor, e->seq,
                    dp, rc, e->weight, wsaf);
        }
    }
    fclose(of);

    float med_dp = 0.0f;
    if (sample_n > 0) {
        qsort(depth_samp, sample_n, sizeof(float), cmp_float);
        med_dp = depth_samp[sample_n / 2];
    }
    fprintf(stderr,
            "{\"sam_records\": %lld, \"primary_mapped\": %lld, \"counted_unitigs\": %lld, "
            "\"counted_bases\": %lld, \"mm_filtered\": %lld, \"median_depth\": %.2f, \"pileup_bp\": %lld}\n",
            sam_records, primary_mapped, counted_unitigs, counted_bases, mm_filtered, med_dp, (long long)total_pile_bp);

    for (size_t i = 0; i < imap.cap; i++) free(imap.t[i].seq);
    free(imap.t);
    for (size_t c = 0; c < n_ctgs; c++) {
        free(ctgs[c].name);
        free(ctgs[c].seq);
    }
    free(ctgs);
    cm_free(&cmap);
    free(depth_samp);
    free(counts);
    free(delc);
    free(regs);
    return 0;
}
