/*
 * Fast unordered SAM line parser and per-(read, mate, ref, strand) deduplicator.
 * Replaces pure-Python per-line SAM splitting/decoding in parse_sam_unordered()
 * while preserving exact insertion order, flag semantics, and float64 scores.
 */
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint64_t n_reads;
    uint32_t n_refs;
    uint32_t ref_names_len;
    char *ref_names_buf;
    uint64_t n_mapped_queries;
    uint64_t records_len;
    uint8_t *records_buf;
} SamUnorderedResult;

typedef struct {
    uint32_t off;
    uint32_t len;
    int64_t ridx;
} StrEntry;

typedef struct {
    uint32_t rid;
    uint8_t key_rev;
    uint32_t next;
    double score;
} CandNode;

typedef struct {
    uint32_t qid;
    uint32_t rid;
    uint8_t mate;
    uint8_t key_rev;
    uint32_t cand_idx; /* UINT32_MAX = empty slot */
} CandHashSlot;

typedef struct {
    uint32_t hash;
    uint32_t idx; /* UINT32_MAX = empty slot */
} StrHashSlot;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} Arena;

static inline uint32_t fnv1a_32(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h ? h : 1u;
}

static inline uint64_t mix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static int arena_ensure(Arena *a, size_t extra) {
    if (a->len + extra <= a->cap) return 0;
    size_t ncap = a->cap ? a->cap * 2 : (1u << 20);
    while (ncap < a->len + extra) ncap *= 2;
    char *nd = (char *)realloc(a->data, ncap);
    if (!nd) return -1;
    a->data = nd;
    a->cap = ncap;
    return 0;
}

static int64_t parse_qidx(const char *s, size_t n) {
    size_t st = 0;
    for (size_t i = n; i > 0; i--) {
        if (s[i - 1] == '|') {
            st = i;
            break;
        }
    }
    if (n - st < 2 || s[st] != 'r') return -1;
    int64_t v = 0;
    for (size_t i = st + 1; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c < '0' || c > '9') return -1;
        v = v * 10 + (c - '0');
    }
    return v;
}

typedef struct {
    Arena arena;
    StrEntry *entries;
    uint32_t count;
    uint32_t cap;
    StrHashSlot *table;
    uint32_t tcap;
} StrTable;

static int strtab_init(StrTable *st, uint32_t init_tcap) {
    memset(st, 0, sizeof(*st));
    st->tcap = init_tcap;
    st->table = (StrHashSlot *)malloc((size_t)init_tcap * sizeof(StrHashSlot));
    if (!st->table) return -1;
    for (uint32_t i = 0; i < init_tcap; i++) st->table[i].idx = UINT32_MAX;
    return 0;
}

static void strtab_free(StrTable *st, int keep_arena_data) {
    if (!keep_arena_data) free(st->arena.data);
    free(st->entries);
    free(st->table);
}

static int strtab_rehash(StrTable *st) {
    uint32_t ntcap = st->tcap * 2;
    StrHashSlot *nt = (StrHashSlot *)malloc((size_t)ntcap * sizeof(StrHashSlot));
    if (!nt) return -1;
    for (uint32_t i = 0; i < ntcap; i++) nt[i].idx = UINT32_MAX;
    uint32_t mask = ntcap - 1;
    for (uint32_t i = 0; i < st->tcap; i++) {
        if (st->table[i].idx == UINT32_MAX) continue;
        uint32_t pos = st->table[i].hash & mask;
        while (nt[pos].idx != UINT32_MAX) pos = (pos + 1) & mask;
        nt[pos] = st->table[i];
    }
    free(st->table);
    st->table = nt;
    st->tcap = ntcap;
    return 0;
}

static uint32_t strtab_intern(StrTable *st, const char *s, size_t n, int null_term, int parse_ridx) {
    if ((uint64_t)(st->count + 1) * 10 >= (uint64_t)st->tcap * 7) {
        if (strtab_rehash(st) != 0) return UINT32_MAX;
    }
    uint32_t h = fnv1a_32(s, n);
    uint32_t mask = st->tcap - 1;
    uint32_t pos = h & mask;
    while (st->table[pos].idx != UINT32_MAX) {
        uint32_t id = st->table[pos].idx;
        if (st->table[pos].hash == h && st->entries[id].len == n &&
            memcmp(st->arena.data + st->entries[id].off, s, n) == 0) {
            return id;
        }
        pos = (pos + 1) & mask;
    }
    if (st->count >= st->cap) {
        uint32_t ncap = st->cap ? st->cap * 2 : 4096u;
        StrEntry *ne = (StrEntry *)realloc(st->entries, (size_t)ncap * sizeof(StrEntry));
        if (!ne) return UINT32_MAX;
        st->entries = ne;
        st->cap = ncap;
    }
    size_t need = n + (null_term ? 1u : 0u);
    if (arena_ensure(&st->arena, need) != 0) return UINT32_MAX;
    uint32_t off = (uint32_t)st->arena.len;
    if (n > 0) memcpy(st->arena.data + off, s, n);
    if (null_term) st->arena.data[off + n] = '\0';
    st->arena.len += need;

    uint32_t id = st->count++;
    st->entries[id].off = off;
    st->entries[id].len = (uint32_t)n;
    st->entries[id].ridx = parse_ridx ? parse_qidx(s, n) : -1;
    st->table[pos].hash = h;
    st->table[pos].idx = id;
    return id;
}

typedef struct {
    uint32_t *head1;
    uint32_t *tail1;
    uint32_t *cnt1;
    uint32_t *head2;
    uint32_t *tail2;
    uint32_t *cnt2;
    uint32_t cap;
} QLists;

static int qlists_ensure(QLists *ql, uint32_t qid) {
    if (qid < ql->cap) return 0;
    uint32_t ncap = ql->cap ? ql->cap * 2 : 4096u;
    while (ncap <= qid) ncap *= 2;
    uint32_t *h1 = (uint32_t *)realloc(ql->head1, (size_t)ncap * sizeof(uint32_t));
    uint32_t *t1 = (uint32_t *)realloc(ql->tail1, (size_t)ncap * sizeof(uint32_t));
    uint32_t *c1 = (uint32_t *)realloc(ql->cnt1, (size_t)ncap * sizeof(uint32_t));
    uint32_t *h2 = (uint32_t *)realloc(ql->head2, (size_t)ncap * sizeof(uint32_t));
    uint32_t *t2 = (uint32_t *)realloc(ql->tail2, (size_t)ncap * sizeof(uint32_t));
    uint32_t *c2 = (uint32_t *)realloc(ql->cnt2, (size_t)ncap * sizeof(uint32_t));
    if (!h1 || !t1 || !c1 || !h2 || !t2 || !c2) {
        free(h1 ? h1 : ql->head1);
        free(t1 ? t1 : ql->tail1);
        free(c1 ? c1 : ql->cnt1);
        free(h2 ? h2 : ql->head2);
        free(t2 ? t2 : ql->tail2);
        free(c2 ? c2 : ql->cnt2);
        memset(ql, 0, sizeof(*ql));
        return -1;
    }
    ql->head1 = h1; ql->tail1 = t1; ql->cnt1 = c1;
    ql->head2 = h2; ql->tail2 = t2; ql->cnt2 = c2;
    for (uint32_t i = ql->cap; i < ncap; i++) {
        ql->head1[i] = UINT32_MAX; ql->tail1[i] = UINT32_MAX; ql->cnt1[i] = 0;
        ql->head2[i] = UINT32_MAX; ql->tail2[i] = UINT32_MAX; ql->cnt2[i] = 0;
    }
    ql->cap = ncap;
    return 0;
}

static void qlists_free(QLists *ql) {
    free(ql->head1); free(ql->tail1); free(ql->cnt1);
    free(ql->head2); free(ql->tail2); free(ql->cnt2);
}

typedef struct {
    CandNode *nodes;
    uint32_t count;
    uint32_t cap;
    CandHashSlot *table;
    uint32_t tcap;
} CandStore;

static int cands_init(CandStore *cs, uint32_t init_tcap) {
    memset(cs, 0, sizeof(*cs));
    cs->tcap = init_tcap;
    cs->table = (CandHashSlot *)malloc((size_t)init_tcap * sizeof(CandHashSlot));
    if (!cs->table) return -1;
    for (uint32_t i = 0; i < init_tcap; i++) cs->table[i].cand_idx = UINT32_MAX;
    return 0;
}

static void cands_free(CandStore *cs) {
    free(cs->nodes);
    free(cs->table);
}

static inline uint32_t cand_hash(uint32_t qid, uint32_t rid, uint8_t mate, uint8_t key_rev) {
    uint64_t k = ((uint64_t)qid << 32) ^ ((uint64_t)rid << 2) ^ ((uint64_t)mate << 1) ^ (uint64_t)key_rev;
    return (uint32_t)mix64(k);
}

static int cands_rehash(CandStore *cs) {
    uint32_t ntcap = cs->tcap * 2;
    CandHashSlot *nt = (CandHashSlot *)malloc((size_t)ntcap * sizeof(CandHashSlot));
    if (!nt) return -1;
    for (uint32_t i = 0; i < ntcap; i++) nt[i].cand_idx = UINT32_MAX;
    uint32_t mask = ntcap - 1;
    for (uint32_t i = 0; i < cs->tcap; i++) {
        CandHashSlot sl = cs->table[i];
        if (sl.cand_idx == UINT32_MAX) continue;
        uint32_t pos = cand_hash(sl.qid, sl.rid, sl.mate, sl.key_rev) & mask;
        while (nt[pos].cand_idx != UINT32_MAX) pos = (pos + 1) & mask;
        nt[pos] = sl;
    }
    free(cs->table);
    cs->table = nt;
    cs->tcap = ntcap;
    return 0;
}

static int cands_upsert(CandStore *cs, QLists *ql, uint32_t qid, uint32_t rid,
                        uint8_t mate, uint8_t key_rev, double score) {
    if ((uint64_t)(cs->count + 1) * 10 >= (uint64_t)cs->tcap * 7) {
        if (cands_rehash(cs) != 0) return -1;
    }
    uint32_t mask = cs->tcap - 1;
    uint32_t pos = cand_hash(qid, rid, mate, key_rev) & mask;
    while (cs->table[pos].cand_idx != UINT32_MAX) {
        CandHashSlot *sl = &cs->table[pos];
        if (sl->qid == qid && sl->rid == rid && sl->mate == mate && sl->key_rev == key_rev) {
            if (score > cs->nodes[sl->cand_idx].score) {
                cs->nodes[sl->cand_idx].score = score;
            }
            return 0;
        }
        pos = (pos + 1) & mask;
    }
    if (cs->count >= cs->cap) {
        uint32_t ncap = cs->cap ? cs->cap * 2 : 8192u;
        CandNode *nn = (CandNode *)realloc(cs->nodes, (size_t)ncap * sizeof(CandNode));
        if (!nn) return -1;
        cs->nodes = nn;
        cs->cap = ncap;
    }
    uint32_t cidx = cs->count++;
    cs->nodes[cidx].rid = rid;
    cs->nodes[cidx].key_rev = key_rev;
    cs->nodes[cidx].next = UINT32_MAX;
    cs->nodes[cidx].score = score;
    cs->table[pos].qid = qid;
    cs->table[pos].rid = rid;
    cs->table[pos].mate = mate;
    cs->table[pos].key_rev = key_rev;
    cs->table[pos].cand_idx = cidx;

    if (mate == 0) {
        if (ql->head1[qid] == UINT32_MAX) ql->head1[qid] = cidx;
        else cs->nodes[ql->tail1[qid]].next = cidx;
        ql->tail1[qid] = cidx;
        ql->cnt1[qid]++;
    } else {
        if (ql->head2[qid] == UINT32_MAX) ql->head2[qid] = cidx;
        else cs->nodes[ql->tail2[qid]].next = cidx;
        ql->tail2[qid] = cidx;
        ql->cnt2[qid]++;
    }
    return 0;
}

static int process_line(const char *line, size_t len, int paired,
                        const char *score_tag, int score_tag_len, double score_scale,
                        StrTable *qtab, StrTable *rtab, QLists *ql, CandStore *cs) {
    if (len == 0 || line[0] == '@') return 0;

    const char *f_ptr[12];
    size_t f_len[12];
    int nf = 0;
    const char *p = line;
    const char *end = line + len;
    while (nf < 11) {
        const char *tab = (const char *)memchr(p, '\t', (size_t)(end - p));
        if (!tab) {
            f_ptr[nf] = p;
            f_len[nf] = (size_t)(end - p);
            nf++;
            p = end;
            break;
        }
        f_ptr[nf] = p;
        f_len[nf] = (size_t)(tab - p);
        nf++;
        p = tab + 1;
    }
    if (nf < 11) return 0;

    const char *qn = f_ptr[0];
    size_t qlen = f_len[0];
    if (qlen >= 2 && qn[qlen - 2] == '/' && (qn[qlen - 1] == '1' || qn[qlen - 1] == '2')) {
        qlen -= 2;
    }
    uint32_t qid = strtab_intern(qtab, qn, qlen, 0, 1);
    if (qid == UINT32_MAX || qlists_ensure(ql, qid) != 0) return -1;

    uint64_t flag = 0;
    for (size_t i = 0; i < f_len[1]; i++) {
        unsigned char c = (unsigned char)f_ptr[1][i];
        if (c >= '0' && c <= '9') flag = flag * 10 + (c - '0');
    }
    if ((flag & 4u) || (flag & 2048u)) return 0;

    const char *rn = f_ptr[2];
    size_t rlen = f_len[2];
    if (rlen == 1 && rn[0] == '*') return 0;

    double score = 0.0;
    if (score_scale != 0.0 && p <= end) {
        const char *tp = p;
        while (tp < end) {
            const char *tab = (const char *)memchr(tp, '\t', (size_t)(end - tp));
            size_t tlen = tab ? (size_t)(tab - tp) : (size_t)(end - tp);
            if ((int)tlen >= score_tag_len && memcmp(tp, score_tag, (size_t)score_tag_len) == 0) {
                char numbuf[64];
                size_t nlen = tlen - (size_t)score_tag_len;
                if (nlen >= sizeof(numbuf)) nlen = sizeof(numbuf) - 1;
                memcpy(numbuf, tp + score_tag_len, nlen);
                numbuf[nlen] = '\0';
                score = strtod(numbuf, NULL) * score_scale;
                break;
            }
            if (!tab) break;
            tp = tab + 1;
        }
    }

    uint32_t rid = strtab_intern(rtab, rn, rlen, 1, 0);
    if (rid == UINT32_MAX) return -1;

    uint8_t is_rev = (flag & 16u) ? 1u : 0u;
    uint8_t mate = (paired && (flag & 128u)) ? 1u : 0u;
    uint8_t key_rev = mate ? (uint8_t)(!is_rev) : is_rev;
    return cands_upsert(cs, ql, qid, rid, mate, key_rev, score);
}

SamUnorderedResult *sam_parse_fd(int fd, int paired, const char *score_tag,
                                 int score_tag_len, double score_scale) {
    StrTable qtab, rtab;
    QLists ql;
    CandStore cs;
    memset(&ql, 0, sizeof(ql));
    if (strtab_init(&qtab, 1u << 16) != 0) return NULL;
    if (strtab_init(&rtab, 4096u) != 0) { strtab_free(&qtab, 0); return NULL; }
    if (cands_init(&cs, 1u << 17) != 0) { strtab_free(&qtab, 0); strtab_free(&rtab, 0); return NULL; }

    size_t bcap = 4u << 20;
    size_t blen = 0;
    char *buf = (char *)malloc(bcap);
    if (!buf) goto fail;

    for (;;) {
        if (blen == bcap) {
            bcap *= 2;
            char *nb = (char *)realloc(buf, bcap);
            if (!nb) goto fail;
            buf = nb;
        }
        ssize_t nread = read(fd, buf + blen, bcap - blen);
        if (nread < 0) {
            if (errno == EINTR) continue;
            goto fail;
        }
        if (nread == 0) break;
        blen += (size_t)nread;

        size_t pos = 0;
        for (;;) {
            const char *nl = (const char *)memchr(buf + pos, '\n', blen - pos);
            if (!nl) break;
            size_t lstart = pos;
            size_t llen = (size_t)(nl - (buf + lstart));
            pos = lstart + llen + 1;
            if (process_line(buf + lstart, llen, paired, score_tag, score_tag_len, score_scale,
                             &qtab, &rtab, &ql, &cs) != 0) {
                goto fail;
            }
        }
        if (pos > 0) {
            size_t rem = blen - pos;
            if (rem > 0) memmove(buf, buf + pos, rem);
            blen = rem;
        }
    }
    if (blen > 0) {
        if (process_line(buf, blen, paired, score_tag, score_tag_len, score_scale,
                         &qtab, &rtab, &ql, &cs) != 0) {
            goto fail;
        }
    }
    free(buf);
    buf = NULL;

    uint64_t n_mapped = 0;
    size_t total_rec_bytes = 0;
    for (uint32_t qid = 0; qid < qtab.count; qid++) {
        uint32_t n1 = ql.cnt1[qid];
        uint32_t n2 = ql.cnt2[qid];
        if (n1 == 0 && n2 == 0) continue;
        n_mapped++;
        total_rec_bytes += 16u + (size_t)(n1 + n2) * 16u;
    }

    uint8_t *rbuf = total_rec_bytes ? (uint8_t *)malloc(total_rec_bytes) : NULL;
    if (total_rec_bytes && !rbuf) goto fail;

    size_t off = 0;
    for (uint32_t qid = 0; qid < qtab.count; qid++) {
        uint32_t n1 = ql.cnt1[qid];
        uint32_t n2 = ql.cnt2[qid];
        if (n1 == 0 && n2 == 0) continue;
        int64_t ridx = qtab.entries[qid].ridx;
        memcpy(rbuf + off, &ridx, 8); off += 8;
        memcpy(rbuf + off, &n1, 4); off += 4;
        memcpy(rbuf + off, &n2, 4); off += 4;
        for (uint32_t e = ql.head1[qid]; e != UINT32_MAX; e = cs.nodes[e].next) {
            uint32_t rid = cs.nodes[e].rid;
            uint32_t rev = (uint32_t)cs.nodes[e].key_rev;
            double sc = cs.nodes[e].score;
            memcpy(rbuf + off, &rid, 4); off += 4;
            memcpy(rbuf + off, &rev, 4); off += 4;
            memcpy(rbuf + off, &sc, 8); off += 8;
        }
        for (uint32_t e = ql.head2[qid]; e != UINT32_MAX; e = cs.nodes[e].next) {
            uint32_t rid = cs.nodes[e].rid;
            uint32_t rev = (uint32_t)cs.nodes[e].key_rev;
            double sc = cs.nodes[e].score;
            memcpy(rbuf + off, &rid, 4); off += 4;
            memcpy(rbuf + off, &rev, 4); off += 4;
            memcpy(rbuf + off, &sc, 8); off += 8;
        }
    }

    SamUnorderedResult *res = (SamUnorderedResult *)calloc(1, sizeof(SamUnorderedResult));
    if (!res) { free(rbuf); goto fail; }
    res->n_reads = qtab.count;
    res->n_refs = rtab.count;
    res->ref_names_len = (uint32_t)rtab.arena.len;
    res->ref_names_buf = rtab.arena.data;
    res->n_mapped_queries = n_mapped;
    res->records_len = (uint64_t)off;
    res->records_buf = rbuf;

    strtab_free(&qtab, 0);
    strtab_free(&rtab, 1); /* keep ref_names_buf */
    qlists_free(&ql);
    cands_free(&cs);
    return res;

fail:
    free(buf);
    strtab_free(&qtab, 0);
    strtab_free(&rtab, 0);
    qlists_free(&ql);
    cands_free(&cs);
    return NULL;
}

void sam_free_result(SamUnorderedResult *res) {
    if (!res) return;
    free(res->ref_names_buf);
    free(res->records_buf);
    free(res);
}
