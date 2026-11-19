#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    double min_ka = (argc > 1) ? atof(argv[1]) : 1.5;
    static char line[1 << 20];
    static char seq[1 << 20];
    setvbuf(stdin, NULL, _IOFBF, 4 << 20);
    setvbuf(stdout, NULL, _IOFBF, 4 << 20);

    long long n_u = 0, kept_u = 0, tot_bp = 0, kept_bp = 0;
    int have = 0;
    double ka = 1.0;
    size_t slen = 0;

    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '>') {
            if (have) {
                n_u++; tot_bp += (long long)slen;
                if (ka >= min_ka && slen >= 31) {
                    long long ka100 = llround(ka * 100.0);
                    if (ka100 < 1) ka100 = 1;
                    fprintf(stdout, ">%lld_%lld\n%s\n", n_u - 1, ka100, seq);
                    kept_u++; kept_bp += (long long)slen;
                }
            }
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
    if (have) {
        n_u++; tot_bp += (long long)slen;
        if (ka >= min_ka && slen >= 31) {
            long long ka100 = llround(ka * 100.0);
            if (ka100 < 1) ka100 = 1;
            fprintf(stdout, ">%lld_%lld\n%s\n", n_u - 1, ka100, seq);
            kept_u++; kept_bp += (long long)slen;
        }
    }
    fprintf(stderr, "{\"n_unitigs\": %lld, \"kept_unitigs\": %lld, \"total_bp\": %lld, \"kept_bp\": %lld}\n",
            n_u, kept_u, tot_bp, kept_bp);
    return 0;
}
