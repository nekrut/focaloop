#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Prepare Logan unitigs for ExpressionLoop aligner -> expr_sam_ec_v2.
 * Tiles unitigs > MAX_UNITIG_TILE bp into TILE_LEN bp chunks with TILE_OVERLAP=30 bp
 * overlap so bwa-mem2 2.2.1 never hits its signed-16-bit (>=32,768 bp) SIGSEGV while
 * preserving the exact total (L - 30) midpoint-bp weight in expr_sam_ec_v2.c.
 */
#define MAX_UNITIG_TILE 2500
#define TILE_LEN 2000
#define TILE_OVERLAP 30
#define MIN_TILE_TAIL 500

static void emit_unitig_tiled(
    long long uid,
    long long ka100,
    const char *seq,
    size_t slen,
    long long *kept_u,
    long long *kept_bp
) {
    if (slen <= MAX_UNITIG_TILE) {
        fprintf(stdout, ">%lld_%lld\n", uid, ka100);
        fwrite(seq, 1, slen, stdout);
        fputc('\n', stdout);
        (*kept_u)++;
        (*kept_bp) += (long long)slen;
        return;
    }
    size_t pos = 0;
    size_t tile_idx = 0;
    while (pos < slen) {
        size_t rem = slen - pos;
        size_t chunk = (rem <= MAX_UNITIG_TILE) ? rem : TILE_LEN;
        if (rem > chunk && (rem - (chunk - TILE_OVERLAP)) < MIN_TILE_TAIL) {
            chunk = rem;
        }
        fprintf(stdout, ">%lldc%zu_%lld\n", uid, tile_idx, ka100);
        fwrite(seq + pos, 1, chunk, stdout);
        fputc('\n', stdout);
        (*kept_u)++;
        (*kept_bp) += (long long)chunk;
        if (pos + chunk >= slen) break;
        pos += (chunk - TILE_OVERLAP);
        tile_idx++;
    }
}

int main(int argc, char **argv) {
    double min_ka = (argc > 1) ? atof(argv[1]) : 1.5;
    static char line[1 << 20];
    size_t seq_cap = 1 << 20;
    char *seq = (char *)malloc(seq_cap);
    if (!seq) {
        fprintf(stderr, "OOM allocating unitig sequence buffer\n");
        return 1;
    }
    setvbuf(stdin, NULL, _IOFBF, 4 << 20);
    setvbuf(stdout, NULL, _IOFBF, 4 << 20);

    long long n_u = 0, kept_u = 0, tot_bp = 0, kept_bp = 0;
    int have = 0;
    double ka = 1.0;
    size_t slen = 0;

    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '>') {
            if (have) {
                n_u++;
                tot_bp += (long long)slen;
                if (ka >= min_ka && slen >= 31) {
                    long long ka100 = llround(ka * 100.0);
                    if (ka100 < 1) ka100 = 1;
                    emit_unitig_tiled(n_u - 1, ka100, seq, slen, &kept_u, &kept_bp);
                }
            }
            have = 1;
            slen = 0;
            ka = 1.0;
            char *p = strstr(line, "ka:f:");
            if (p) ka = strtod(p + 5, NULL);
        } else {
            size_t l = strlen(line);
            while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) l--;
            if (slen + l + 1 > seq_cap) {
                while (slen + l + 1 > seq_cap) seq_cap *= 2;
                char *nseq = (char *)realloc(seq, seq_cap);
                if (!nseq) {
                    free(seq);
                    fprintf(stderr, "OOM growing unitig sequence buffer\n");
                    return 1;
                }
                seq = nseq;
            }
            memcpy(seq + slen, line, l);
            slen += l;
            seq[slen] = '\0';
        }
    }
    if (have) {
        n_u++;
        tot_bp += (long long)slen;
        if (ka >= min_ka && slen >= 31) {
            long long ka100 = llround(ka * 100.0);
            if (ka100 < 1) ka100 = 1;
            emit_unitig_tiled(n_u - 1, ka100, seq, slen, &kept_u, &kept_bp);
        }
    }
    free(seq);
    fprintf(stderr, "{\"n_unitigs\": %lld, \"kept_unitigs\": %lld, \"total_bp\": %lld, \"kept_bp\": %lld}\n",
            n_u, kept_u, tot_bp, kept_bp);
    return 0;
}
