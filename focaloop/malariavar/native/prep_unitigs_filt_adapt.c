#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define K 15
#define KMASK ((1U << (2 * K)) - 1U)
#define BITSET_BYTES (1ULL << (2 * K - 3)) /* 128 MiB: 2^30 bits */
#define CAL_LIMIT 150000
#define MAX_TILE_LEN 2000
#define TILE_OVERLAP 30
#define MIN_TILE_TAIL 500

/*
 * Emit a unitig, tiling sequences longer than MAX_TILE_LEN + MIN_TILE_TAIL into
 * overlapping chunks (TILE_OVERLAP = 30 bp) so bwa-mem2 never hits its >32 kb
 * signed-16-bit SIGSEGV or quadratic DP slowdown, while midpoint_only=1 (15 bp
 * trim each side) still counts every interior base of the unitig exactly once.
 */
static void emit_unitig_tiled(FILE *out, long long uid, long long ka100, const char *s, size_t slen) {
    if (slen <= (size_t)(MAX_TILE_LEN + MIN_TILE_TAIL)) {
        fprintf(out, ">%lld_%lld\n", uid, ka100);
        fwrite(s, 1, slen, out);
        fputc('\n', out);
        return;
    }
    size_t pos = 0;
    int ci = 0;
    while (pos < slen) {
        size_t end = pos + (size_t)MAX_TILE_LEN;
        if (end >= slen || (slen - end) <= (size_t)MIN_TILE_TAIL) {
            end = slen;
        }
        fprintf(out, ">%lldc%d_%lld\n", uid, ci, ka100);
        fwrite(s + pos, 1, end - pos, out);
        fputc('\n', out);
        if (end >= slen) break;
        pos = end - (size_t)TILE_OVERLAP;
        ci++;
    }
}

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

static int build_index(const char *fasta_path, const char *out_path) {
    FILE *fp = fopen(fasta_path, "r");
    if (!fp) { perror("fopen fasta"); return 1; }
    uint8_t *bits = calloc(1, BITSET_BYTES);
    if (!bits) { fprintf(stderr, "OOM bitset\n"); fclose(fp); return 1; }
    static char line[1 << 16];
    uint32_t fwd = 0, rc = 0;
    int valid = 0;
    long long n_kmers = 0, n_unique = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') { valid = 0; fwd = 0; rc = 0; continue; }
        for (size_t i = 0; line[i]; i++) {
            uint8_t b = ENC[(uint8_t)line[i]];
            if (b > 3) {
                if (line[i] != '\n' && line[i] != '\r' && line[i] != ' ') {
                    valid = 0; fwd = 0; rc = 0;
                }
                continue;
            }
            fwd = ((fwd << 2) | b) & KMASK;
            rc = (rc >> 2) | ((uint32_t)(3U - b) << (2 * (K - 1)));
            if (++valid >= K) {
                uint32_t c = (fwd < rc) ? fwd : rc;
                size_t byte_idx = (size_t)(c >> 3);
                uint8_t mask = (uint8_t)(1U << (c & 7U));
                if (!(bits[byte_idx] & mask)) {
                    bits[byte_idx] |= mask;
                    n_unique++;
                }
                n_kmers++;
            }
        }
    }
    fclose(fp);
    int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open out"); free(bits); return 1; }
    size_t off = 0;
    while (off < BITSET_BYTES) {
        ssize_t w = write(fd, bits + off, BITSET_BYTES - off);
        if (w <= 0) { perror("write bitset"); close(fd); free(bits); return 1; }
        off += (size_t)w;
    }
    close(fd);
    free(bits);
    fprintf(stderr, "Built k=%d bitset: %lld scanned, %lld unique canonical kmers (%.2f%% occupancy) -> %s\n",
            K, n_kmers, n_unique, 100.0 * (double)n_unique / (double)(1ULL << (2 * K - 1)), out_path);
    return 0;
}

static inline int k15_pass(const uint8_t *bits, const char *seq, size_t slen, int need_hits) {
    if (need_hits <= 0) return 1;
    uint32_t fwd = 0, rc = 0;
    int valid = 0, hits = 0;
    for (size_t i = 0; i < slen; i++) {
        uint8_t b = ENC[(uint8_t)seq[i]];
        if (b > 3) { valid = 0; fwd = 0; rc = 0; continue; }
        fwd = ((fwd << 2) | b) & KMASK;
        rc = (rc >> 2) | ((uint32_t)(3U - b) << (2 * (K - 1)));
        if (++valid >= K) {
            uint32_t c = (fwd < rc) ? fwd : rc;
            if (bits[c >> 3] & (uint8_t)(1U << (c & 7U))) {
                if (++hits >= need_hits) return 1;
            }
        }
    }
    return 0;
}

typedef struct {
    long long uid;
    double ka;
    uint32_t slen;
    size_t seq_off;
} CalRec;

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc >= 4 && strcmp(argv[1], "--build") == 0) {
        return build_index(argv[2], argv[3]);
    }
    /* Usage: prep_unitigs_filt_adapt <min_ka|adapt> <min_len> <bit_path|-> <min_hits> <relax_ka> [max_ka_adapt=3.0] */
    int adaptive = (argc > 1 && strcmp(argv[1], "adapt") == 0);
    double min_ka = adaptive ? 1.5 : ((argc > 1) ? atof(argv[1]) : 1.5);
    double max_ka = (argc > 6) ? atof(argv[6]) : 3.0;
    size_t min_len = (argc > 2) ? (size_t)atol(argv[2]) : 0;
    const char *bit_path = (argc > 3 && argv[3][0] && strcmp(argv[3], "-") != 0) ? argv[3] : NULL;
    int min_hits = (argc > 4) ? atoi(argv[4]) : 2;
    double relax_ka = (argc > 5) ? atof(argv[5]) : 5.0;

    const uint8_t *bits = NULL;
    int fd = -1;
    if (bit_path) {
        fd = open(bit_path, O_RDONLY);
        if (fd < 0) { perror("open bitset"); return 1; }
        void *p = mmap(NULL, BITSET_BYTES, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { perror("mmap bitset"); close(fd); return 1; }
        bits = (const uint8_t *)p;
    }

    static char line[1 << 20];
    static char seq[1 << 20];
    setvbuf(stdin, NULL, _IOFBF, 4 << 20);
    setvbuf(stdout, NULL, _IOFBF, 4 << 20);

    /* Calibration buffer for adaptive mode (stores host-passing unitigs with ka >= 1.5) */
    CalRec *cal_recs = adaptive ? malloc(sizeof(CalRec) * CAL_LIMIT) : NULL;
    size_t cal_seq_cap = adaptive ? (16ULL << 20) : 0;
    char *cal_seqs = adaptive ? malloc(cal_seq_cap) : NULL;
    size_t cal_seq_len = 0;
    size_t cal_n = 0;
    double *long_kas = adaptive ? malloc(sizeof(double) * CAL_LIMIT) : NULL;
    size_t long_n = 0;
    int calibrated = !adaptive;
    double med_len80 = -1.0;
    double eff_min_ka = min_ka;

    long long n_u = 0, kept_u = 0, drop_ka = 0, drop_host = 0;
    long long tot_bp = 0, kept_bp = 0;
    double w_bp = 0.0;

    int have = 0;
    double ka = 1.0;
    size_t slen = 0;

    #define FLUSH_CAL() do { \
        if (!calibrated) { \
            if (long_n >= 200) { \
                qsort(long_kas, long_n, sizeof(double), cmp_dbl); \
                med_len80 = long_kas[long_n / 2]; \
                if (med_len80 < 16.0) eff_min_ka = min_ka; \
                else if (med_len80 >= 24.0) eff_min_ka = max_ka; \
                else eff_min_ka = min_ka + (max_ka - min_ka) * ((med_len80 - 16.0) / 8.0); \
            } else { \
                eff_min_ka = min_ka; \
            } \
            for (size_t _k = 0; _k < cal_n; _k++) { \
                CalRec *_r = &cal_recs[_k]; \
                if (_r->ka < eff_min_ka) { \
                    drop_ka++; \
                } else { \
                    long long ka100 = llround(_r->ka * 100.0); \
                    if (ka100 < 1) ka100 = 1; \
                    emit_unitig_tiled(stdout, _r->uid, ka100, cal_seqs + _r->seq_off, _r->slen); \
                    kept_u++; kept_bp += (long long)_r->slen; w_bp += _r->ka * (double)_r->slen; \
                } \
            } \
            free(cal_recs); free(cal_seqs); free(long_kas); \
            cal_recs = NULL; cal_seqs = NULL; long_kas = NULL; \
            calibrated = 1; \
        } \
    } while (0)

    #define HANDLE_REC() do { \
        n_u++; \
        tot_bp += (long long)slen; \
        double cur_floor = calibrated ? eff_min_ka : min_ka; \
        if (ka < cur_floor || slen < min_len) { \
            drop_ka++; \
        } else { \
            int need = min_hits; \
            if (relax_ka > 0.0 && ka >= relax_ka && need > 1) need = 1; \
            if (!bits || k15_pass(bits, seq, slen, need)) { \
                if (!calibrated) { \
                    if (cal_seq_len + slen > cal_seq_cap) { \
                        while (cal_seq_len + slen > cal_seq_cap) cal_seq_cap *= 2; \
                        cal_seqs = realloc(cal_seqs, cal_seq_cap); \
                    } \
                    memcpy(cal_seqs + cal_seq_len, seq, slen); \
                    cal_recs[cal_n].uid = n_u - 1; \
                    cal_recs[cal_n].ka = ka; \
                    cal_recs[cal_n].slen = (uint32_t)slen; \
                    cal_recs[cal_n].seq_off = cal_seq_len; \
                    cal_seq_len += slen; \
                    cal_n++; \
                    if (slen >= 80 && long_n < CAL_LIMIT) long_kas[long_n++] = ka; \
                    if (cal_n >= CAL_LIMIT) FLUSH_CAL(); \
                } else { \
                    long long ka100 = llround(ka * 100.0); \
                    if (ka100 < 1) ka100 = 1; \
                    emit_unitig_tiled(stdout, n_u - 1, ka100, seq, slen); \
                    kept_u++; kept_bp += (long long)slen; w_bp += ka * (double)slen; \
                } \
            } else { \
                drop_host++; \
            } \
        } \
    } while (0)

    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '>') {
            if (have) HANDLE_REC();
            have = 1; slen = 0; ka = 1.0;
            char *p = strstr(line, "ka:f:");
            if (p) ka = strtod(p + 5, NULL);
        } else {
            size_t l = strlen(line);
            while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) l--;
            if (slen + l < sizeof(seq) - 1) {
                memcpy(seq + slen, line, l);
                slen += l;
                seq[slen] = '\0';
            }
        }
    }
    if (have) HANDLE_REC();
    FLUSH_CAL();

    fprintf(stderr,
            "{\"n_unitigs\": %lld, \"total_bp\": %lld, \"kept_unitigs\": %lld, \"kept_bp\": %lld, "
            "\"drop_ka\": %lld, \"drop_host\": %lld, \"weighted_bp\": %.2f, "
            "\"adaptive\": %d, \"med_len80_ka\": %.2f, \"eff_min_ka\": %.2f}\n",
            n_u, tot_bp, kept_u, kept_bp, drop_ka, drop_host, w_bp, adaptive, med_len80, eff_min_ka);

    if (bits) munmap((void *)bits, BITSET_BYTES);
    if (fd >= 0) close(fd);
    return 0;
}
