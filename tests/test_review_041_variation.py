"""Comprehensive VariationLoop & malariavar 0.4.1 code-review regression suite.

Covers Important findings (#20-#44, #62 contract, #70, #71, #72, #74) and
Lesser findings (L30-L46, L74, L79-L86, L94), including AddressSanitizer +
UndefinedBehaviorSanitizer compilation and execution across all 5 modified C files.
"""
from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from focaloop import runners_variation as RV
from focaloop.malariavar import caller as C
from focaloop.malariavar import dothash as DH
from focaloop.malariavar import engine as E
from focaloop.malariavar import focused_cohort as FC
from focaloop.malariavar import genome as G
from focaloop.malariavar import kernels as Kn
from focaloop.malariavar import logan_var as LV
from focaloop.malariavar.native import _loader as NL
from focaloop.malariavar import query as Q
from focaloop.malariavar import state_store as SS
from focaloop.malariavar import streamer as ST
from focaloop.malariavar import varloop_giab as VG
from focaloop.malariavar import who_resistance as WR

ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = ROOT / "focaloop" / "malariavar" / "native"
TOOL_XML = ROOT / "tools" / "focaloop" / "variationloop.xml"
TEST_DATA = ROOT / "tools" / "focaloop" / "test-data" / "variationloop"


def _make_random_dna(length: int, seed: int = 12345) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _write_paired_fastq_gz(
    r1_path: Path,
    r2_path: Path,
    template: str,
    pairs: list[tuple[int, int, str | None, str | None]],
    read_len: int = 150,
) -> None:
    rc_map = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

    def _revcomp(s: str) -> str:
        return s.translate(rc_map)[::-1]

    with gzip.open(r1_path, "wt", encoding="ascii") as f1, gzip.open(r2_path, "wt", encoding="ascii") as f2:
        for idx, (s0, frag_len, r1_override, r2_override) in enumerate(pairs):
            frag = template[s0 : s0 + frag_len]
            seq1 = r1_override if r1_override is not None else frag[:read_len]
            seq2 = r2_override if r2_override is not None else _revcomp(frag[-read_len:])
            f1.write(f"@r{idx}/1\n{seq1}\n+\n{'I' * len(seq1)}\n")
            f2.write(f"@r{idx}/2\n{seq2}\n+\n{'I' * len(seq2)}\n")


# ============================================================================
# 1. Native C ASan + UBSan Compilation & Runtime Suite
#    (varloop_core.c, logan_var_pileup.c, prep_unitigs_filt_adapt.c,
#     pla_sieve.c, dothash_sketch.c -> L74, L79-L86, #24, #62, #70)
# ============================================================================


def test_native_c_asan_ubsan_all_five_sources(tmp_path: Path) -> None:
    cc = shutil.which("gcc") or shutil.which("clang")
    assert cc is not None, "gcc or clang required for ASan/UBSan verification"

    san_flags = [
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-Wno-unused-function",
        "-msse4.2",
        "-mpopcnt",
        "-I",
        str(NATIVE_DIR),
    ]
    san_env = {
        **os.environ,
        "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1:halt_on_error=1",
        "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1",
    }

    # --- A. prep_unitigs_filt_adapt.c (L79: >1.2 Mb unitig dynamic realloc) ---
    prep_bin = tmp_path / "prep_asan"
    subprocess.run(
        [cc, *san_flags, str(NATIVE_DIR / "prep_unitigs_filt_adapt.c"), "-lm", "-o", str(prep_bin)],
        check=True,
    )
    long_seq = ("ACGTACGTACGTACGTTGCATGCATGCATGCA" * 39063)[:1_250_000]
    in_fa = f">0 ka:f:12.5\n{long_seq}\n".encode("ascii")
    res_prep = subprocess.run(
        [str(prep_bin), "2.0", "31", "-", "0", "0"],
        input=in_fa,
        capture_output=True,
        env=san_env,
        check=True,
    )
    prep_meta = json.loads(res_prep.stderr.decode("utf-8").strip().splitlines()[-1])
    assert prep_meta["n_unitigs"] == 1
    assert prep_meta["kept_unitigs"] == 1
    assert prep_meta["total_bp"] == 1_250_000
    assert prep_meta["kept_bp"] == 1_250_000
    tiles = [
        ln.strip()
        for ln in res_prep.stdout.decode("ascii").splitlines()
        if ln.strip() and not ln.startswith(">")
    ]
    # Reconstruct full 1.25 Mb sequence from 2,000-bp / 30-bp-overlap tiles
    recon = tiles[0] + "".join(t[30:] for t in tiles[1:])
    assert len(recon) == 1_250_000
    assert recon == long_seq

    # --- B. logan_var_pileup.c (#70 repeat indels + L80 BED-only contig load + #62 cov TSV) ---
    pile_bin = tmp_path / "pile_asan"
    subprocess.run(
        [cc, *san_flags, str(NATIVE_DIR / "logan_var_pileup.c"), "-lm", "-o", str(pile_bin)],
        check=True,
    )
    flank_l = "GCTAGCTAGCTGATCGATCGATCGATCGAT"  # 30 bp non-repeat
    flank_r = "CTAGCTAGCTAGTCGATCGATCGATCGATC"  # 30 bp non-repeat
    # Build chrT with 4 loci: A17 del1, A20 del1, (AT)x12 del2, A18 ins1 + a decoy contig chrUnused
    spacer = "CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG"  # 50 bp
    seg_a17 = flank_l + ("A" * 17) + flank_r
    seg_a20 = flank_l + ("A" * 20) + flank_r
    seg_at12 = flank_l + ("AT" * 12) + flank_r
    seg_a18 = flank_l + ("A" * 18) + flank_r
    chr_t_seq = spacer + seg_a17 + spacer + seg_a20 + spacer + seg_at12 + spacer + seg_a18 + spacer
    ref_p = tmp_path / "logan_ref.fa"
    ref_p.write_text(f">chrUnused\n{'ACGT' * 5000}\n>chrT\n{chr_t_seq}\n", encoding="ascii")
    bed_p = tmp_path / "logan_targets.bed"
    bed_p.write_text(f"chrT\t0\t{len(chr_t_seq)}\n", encoding="ascii")

    # Compute 1-based SAM positions for short k=31 bubble unitigs (14 bp / 11 bp / 12 bp / 13 bp left flank -> qp < 15!)
    off_a17 = len(spacer)
    off_a20 = off_a17 + len(seg_a17) + len(spacer)
    off_at12 = off_a20 + len(seg_a20) + len(spacer)
    off_a18 = off_at12 + len(seg_at12) + len(spacer)

    sam_lines = [
        f"@HD\tVN:1.6\tSO:unsorted",
        f"@SQ\tSN:chrUnused\tLN:20000",
        f"@SQ\tSN:chrT\tLN:{len(chr_t_seq)}",
        # Ref unitigs covering allchrT at ka=15.0 (1500)
        f"1_1500\t0\tchrT\t1\t60\t{len(chr_t_seq)}M\t*\t0\t0\t{chr_t_seq}\t*\tNM:i:0",
        # Alt unitig 1: A17 del1 -> 14M1D46M (qp=14 < 15!)
        f"2_1500\t0\tchrT\t{off_a17 + 16 + 1}\t60\t14M1D46M\t*\t0\t0\t{flank_l[16:] + ('A' * 16) + flank_r}\t*\tNM:i:1",
        # Alt unitig 2: A20 del1 -> 11M1D49M (qp=11 < 15!)
        f"3_1500\t0\tchrT\t{off_a20 + 19 + 1}\t60\t11M1D49M\t*\t0\t0\t{flank_l[19:] + ('A' * 19) + flank_r}\t*\tNM:i:1",
        # Alt unitig 3: (AT)x12 del2 -> 12M2D52M (qp=12 < 15!)
        f"4_1500\t0\tchrT\t{off_at12 + 18 + 1}\t60\t12M2D52M\t*\t0\t0\t{flank_l[18:] + ('AT' * 11) + flank_r}\t*\tNM:i:2",
        # Alt unitig 4: A18 ins1 split across two 30-bp overlapping tiles (`5c0_1500` + `5c1_1500`)!
        # Tile `5c1_1500` has 1M1I48M (qp=1 < 15, qp_right=19 >= 15 -> #70!), and 30-bp tile overlap must NOT double-count!
        f"5c0_1500\t0\tchrT\t{off_a18 + 1}\t60\t30M1I28M\t*\t0\t0\t{flank_l + ('A' * 19) + flank_r[:10]}\t*\tNM:i:1",
        f"5c1_1500\t0\tchrT\t{off_a18 + 29 + 1}\t60\t1M1I48M\t*\t0\t0\t{flank_l[29:] + ('A' * 19) + flank_r}\t*\tNM:i:1",
    ]
    cand_out = tmp_path / "cand.tsv"
    cov_out = tmp_path / "cov.tsv"
    subprocess.run(
        [str(pile_bin), str(ref_p), str(bed_p), str(cand_out), "20", "0.12", "1", str(cov_out)],
        input=("\n".join(sam_lines) + "\n").encode("ascii"),
        capture_output=True,
        env=san_env,
        check=True,
    )
    cand_df = pd.read_csv(cand_out, sep="\t")
    indels_df = cand_df[cand_df["kind"] == "INDEL"]
    assert len(indels_df) == 4, f"Expected all 4 repeat indels, got:\n{indels_df}"
    for w_v in indels_df["wsaf"]:
        assert 0.45 <= float(w_v) <= 0.55, f"Unexpected WSAF (tiled unitig double-count?): {indels_df}"
    assert cov_out.is_file() and "chrT\t" in cov_out.read_text(encoding="utf-8")

    # --- C. pla_sieve.c (L83 OMP_THREAD_LIMIT=2 + L86 Casava 1.8 whitespace headers) ---
    pla_drv_c = tmp_path / "test_pla_asan.c"
    pla_drv_c.write_text(
        r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pla_sieve.h"

int main(void) {
    /* 1. L83: radix_sort_u64 + pla_generate_distinct_syncmers under OMP_THREAD_LIMIT=2 with num_threads=8 */
    size_t n = 100000;
    uint64_t *arr = (uint64_t *)malloc(n * sizeof(uint64_t));
    assert(arr != NULL);
    for (size_t i = 0; i < n; i++) {
        arr[i] = ((uint64_t)(n - i) * 0x9e3779b97f4a7c15ULL) ^ (uint64_t)(i & 7U);
    }
    size_t u = pla_sort_dedup_u64(arr, n, 8);
    assert(u == n);
    for (size_t i = 1; i < u; i++) {
        assert(arr[i - 1] < arr[i]);
    }
    int rc_g = pla_generate_distinct_syncmers(arr, 80000, 1, 31, 15, 42ULL, 8);
    assert(rc_g == 0);
    for (size_t i = 1; i < 80000; i++) {
        assert(arr[i - 1] < arr[i]);
    }

    /* 2. L86: Casava 1.8 paired headers with space + read member 1:N:0 / 2:N:0 */
    pla_index_t idx;
    memset(&idx, 0, sizeof(idx));
    int rc_b = pla_build(&idx, arr, NULL, 1024, 16, 1, 31, 15, 8);
    assert(rc_b == 0);
    const char *fq =
        "@INST:1:FC:1:1101:100:200 1:N:0:ATCG\n"
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@INST:1:FC:1:1101:100:200 2:N:0:ATCG\n"
        "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n";
    uint8_t out[1024];
    int64_t stats[10] = {0};
    int64_t qhist[128] = {0};
    int64_t out_o = 0, out_spot = 0;
    int64_t used = pla_fastq_block(
        (const uint8_t *)fq, (int64_t)strlen(fq), out,
        0, (uint64_t)1ULL << 32, 1, &idx, 0,
        stats, 10, qhist, NULL, 0, 30, 0, &out_o, &out_spot
    );
    assert(used == (int64_t)strlen(fq));
    assert(stats[0] == 1); /* 1 paired spot recognized */
    assert(stats[5] == 0); /* 0 QNAME mismatch errors */
    pla_close(&idx);
    free(arr);
    return 0;
}
""",
        encoding="utf-8",
    )
    pla_bin = tmp_path / "pla_asan"
    subprocess.run(
        [cc, *san_flags, "-fopenmp", str(NATIVE_DIR / "pla_sieve.c"), str(pla_drv_c), "-lm", "-o", str(pla_bin)],
        check=True,
    )
    subprocess.run(
        [str(pla_bin)],
        env={**san_env, "OMP_THREAD_LIMIT": "2", "OMP_NUM_THREADS": "8"},
        check=True,
    )

    # --- D. dothash_sketch.c (L84 non-OpenMP + OpenMP OMP_THREAD_LIMIT + L74 bounds) ---
    dh_drv_c = tmp_path / "test_dh_asan.c"
    dh_drv_c.write_text(
        r"""
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void dothash_knn_1_vs_n_jaccard(
    const int16_t *query, float query_card, const int16_t *mat, const float *row_cards,
    size_t n_rows, int k_top, int32_t *top_idx_out, float *top_jaccard_out, int n_threads
);
void dothash_knn_q_vs_n_jaccard(
    const int16_t *queries, const float *query_cards, size_t n_queries,
    const int16_t *mat, const float *row_cards, size_t n_rows,
    int k_top, int32_t *top_idx_out, float *top_jaccard_out, int n_threads
);

int main(void) {
    const size_t D = 2048;
    const size_t N = 1300; /* >= 1024 so n_threads > 1 hits the parallel/else branch! */
    int16_t *mat = (int16_t *)calloc(N * D, sizeof(int16_t));
    float *cards = (float *)calloc(N, sizeof(float));
    for (size_t i = 0; i < N; i++) {
        mat[i * D + (i % D)] = 10;
        cards[i] = 100.0f / (float)D;
    }
    int16_t q[4 * 2048];
    memset(q, 0, sizeof(q));
    q[0 * D + 7] = 10;
    q[1 * D + 1299] = 10;
    q[2 * D + 500] = 10;
    q[3 * D + 42] = 10;
    float qcards[4] = {100.0f / 2048.0f, 100.0f / 2048.0f, 100.0f / 2048.0f, 100.0f / 2048.0f};
    int32_t idx[4 * 3];
    float sim[4 * 3];
    dothash_knn_q_vs_n_jaccard(q, qcards, 4, mat, cards, N, 3, idx, sim, 4);
    assert(idx[0 * 3] == 7 && sim[0 * 3] > 0.99f);
    assert(idx[1 * 3] == 1299 && sim[1 * 3] > 0.99f);
    assert(idx[2 * 3] == 500 && sim[2 * 3] > 0.99f);
    assert(idx[3 * 3] == 42 && sim[3 * 3] > 0.99f);

    /* Empty cohort n_rows == 0 must still initialize outputs to -1 */
    int32_t idx0[3] = {99, 99, 99};
    float sim0[3] = {1.0f, 1.0f, 1.0f};
    dothash_knn_1_vs_n_jaccard(q, qcards[0], mat, cards, 0, 3, idx0, sim0, 4);
    assert(idx0[0] == -1 && idx0[1] == -1 && idx0[2] == -1);

    free(mat);
    free(cards);
    return 0;
}
""",
        encoding="utf-8",
    )
    dh_noomp = tmp_path / "dh_noomp_asan"
    subprocess.run(
        [cc, *san_flags, str(NATIVE_DIR / "dothash_sketch.c"), str(dh_drv_c), "-lm", "-o", str(dh_noomp)],
        check=True,
    )
    subprocess.run([str(dh_noomp)], env=san_env, check=True)

    dh_omp = tmp_path / "dh_omp_asan"
    subprocess.run(
        [cc, *san_flags, "-fopenmp", str(NATIVE_DIR / "dothash_sketch.c"), str(dh_drv_c), "-lm", "-o", str(dh_omp)],
        check=True,
    )
    subprocess.run([str(dh_omp)], env={**san_env, "OMP_THREAD_LIMIT": "2"}, check=True)

    # --- E. varloop_core.c (L82 >40 kb mate sieve overflow + L85 >4096 contigs + L81 mate overlap + #24 QUAL='*' + #62 callable runs) ---
    vc_drv_c = tmp_path / "test_vc_asan.c"
    vc_drv_c.write_text(
        r"""
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int vl_fastq_sieve_paired(
    int fd1, int fd2, int fd_out_interleaved, int fd_out1, int fd_out2,
    const uint64_t *tab, uint64_t tab_len, int32_t stride, int32_t need,
    uint64_t keep_thr, int32_t trim_adapter, int32_t min_len, int64_t max_spots, int64_t *stats
);
void *vl_acc_create(
    int32_t n_contigs, const char *names_flat, const int64_t *name_offs,
    const int64_t *lengths, const int64_t *ref_offs, const int64_t *bin_offs,
    const uint8_t *ref_seq, int32_t n_shards, int32_t min_mapq, int32_t min_bq,
    int32_t bin_mapq, int32_t zoom_min_mapq, double max_mm_frac,
    int32_t flt_indel_flank, int32_t flt_clip_flank, int32_t flt_end_trim,
    int32_t flt_mm_win, int32_t flt_mm_n, int32_t capture_informative, int32_t dedup_bits
);
int vl_acc_feed_sam_bytes(void *acc, const uint8_t *data, size_t len);
void vl_acc_get_summary(void *acc, int64_t *out_stats, int64_t *out_ccount, int64_t *out_binbases, int64_t *out_mem);
void vl_acc_query_positions(void *acc, int32_t ctg_id, const int32_t *pos0, int64_t n, int32_t *out_dp, int32_t *out_delc, int32_t *out_uncap);
int64_t vl_acc_extract_callable_runs(const void *acc, int32_t ctg_id, int32_t min_dp, int32_t *out_s0, int32_t *out_e0, int64_t max_runs);
void vl_acc_destroy(void *acc);

static void write_fq_file(const char *path, int mate, size_t n_fill, size_t fill_len, size_t big_len) {
    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    char *s_fill = (char *)malloc(fill_len + 1);
    char *q_fill = (char *)malloc(fill_len + 1);
    memset(s_fill, 'A', fill_len); s_fill[fill_len] = '\0';
    memset(q_fill, 'I', fill_len); q_fill[fill_len] = '\0';
    for (size_t i = 0; i < n_fill; i++) {
        fprintf(f, "@r%zu/%d\n%s\n+\n%s\n", i, mate, s_fill, q_fill);
    }
    char *s_big = (char *)malloc(big_len + 1);
    char *q_big = (char *)malloc(big_len + 1);
    memset(s_big, 'C', big_len); s_big[big_len] = '\0';
    memset(q_big, 'I', big_len); q_big[big_len] = '\0';
    fprintf(f, "@rbig/%d\n%s\n+\n%s\n", mate, s_big, q_big);
    free(s_fill); free(q_fill); free(s_big); free(q_big);
    fclose(f);
}

int main(int argc, char **argv) {
    assert(argc >= 2);
    const char *dir = argv[1];
    char p1[512], p2[512], pi[512], po1[512], po2[512];
    snprintf(p1, sizeof(p1), "%s/in1.fq", dir);
    snprintf(p2, sizeof(p2), "%s/in2.fq", dir);
    snprintf(pi, sizeof(pi), "%s/out_int.fq", dir);
    snprintf(po1, sizeof(po1), "%s/out1.fq", dir);
    snprintf(po2, sizeof(po2), "%s/out2.fq", dir);

    /* 1. L82: Fill ob_int to ~4.19 MB (just below OUT_CAP=4 MiB) then feed 50 kb paired reads! */
    write_fq_file(p1, 1, 209, 5000, 50000);
    write_fq_file(p2, 2, 209, 5000, 50000);
    int fd1 = open(p1, O_RDONLY);
    int fd2 = open(p2, O_RDONLY);
    int fdi = open(pi, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int fdo1 = open(po1, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int fdo2 = open(po2, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int64_t fq_stats[8] = {0};
    int rc = vl_fastq_sieve_paired(fd1, fd2, fdi, fdo1, fdo2, NULL, 0, 5, 0, 1ULL << 32, 0, 30, 0, fq_stats);
    assert(rc == 0);
    assert(fq_stats[0] == 210 && fq_stats[2] == 210);
    close(fd1); close(fd2); close(fdi); close(fdo1); close(fdo2);

    /* 2. L85 (>4096 contigs dedup), L81 (rejected mate1 overlap), #24 (QUAL='*'), #62 (callable runs) */
    const int32_t NC = 4200;
    char *names = (char *)malloc((size_t)NC * 12U);
    int64_t *noffs = (int64_t *)malloc((size_t)(NC + 1) * sizeof(int64_t));
    int64_t *lens  = (int64_t *)malloc((size_t)NC * sizeof(int64_t));
    int64_t *roffs = (int64_t *)malloc((size_t)(NC + 1) * sizeof(int64_t));
    int64_t *boffs = (int64_t *)malloc((size_t)(NC + 1) * sizeof(int64_t));
    noffs[0] = 0; roffs[0] = 0; boffs[0] = 0;
    for (int32_t c = 0; c < NC; c++) {
        int n = snprintf(names + noffs[c], 12, "ctg%d", c);
        noffs[c + 1] = noffs[c] + n;
        lens[c] = 1000;
        roffs[c + 1] = roffs[c] + 1000;
        boffs[c + 1] = boffs[c] + 1;
    }
    uint8_t *rseq = (uint8_t *)calloc((size_t)roffs[NC], 1); /* all A (0) */
    void *acc = vl_acc_create(NC, names, noffs, lens, roffs, boffs, rseq,
                              1, 20, 13, 1, 0, 0.20, 0, 0, 0, 0, 0, 0, 16);
    assert(acc != NULL);

    const char *sam =
        /* L85: identical (pos=101, PNEXT=201, TLEN=150) on ctg5 vs ctg4101 (4101 == 5 + 4096!) -> MUST NOT collide */
        "1\t99\tctg5\t101\t60\t50M\t=\t201\t150\tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t*\tNM:i:0\n"
        "2\t99\tctg4101\t101\t60\t50M\t=\t201\t150\tCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\t*\tNM:i:50\n"
        /* L81: spot 3 mate 1 has MAPQ=0 (rejected), mate 2 overlaps at pos 301 with MAPQ=60 -> mate 2 MUST be counted! */
        "3\t99\tctg5\t301\t0\t50M\t=\t301\t50\tGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tNM:i:0\n"
        "3\t147\tctg5\t301\t60\t50M\t=\t301\t-50\tGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\tIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\tNM:i:0\n";
    vl_acc_feed_sam_bytes(acc, (const uint8_t *)sam, strlen(sam));

    int64_t st[12] = {0};
    int64_t *cc = (int64_t *)calloc((size_t)NC, sizeof(int64_t));
    int64_t *bb = (int64_t *)calloc((size_t)NC, sizeof(int64_t));
    int64_t mem[3] = {0};
    vl_acc_get_summary(acc, st, cc, bb, mem);
    assert(st[10] == 0); /* L85: zero false PCR duplicates across >4096 contigs! */
    assert(st[11] == 0); /* #24: QUAL='*' accepted cleanly with zero malformed rejects! */

    int32_t qpos[2] = {110, 310};
    int32_t qdp[2] = {0}, qdel[2] = {0}, qunc[2] = {0};
    vl_acc_query_positions(acc, 5, qpos, 2, qdp, qdel, qunc);
    assert(qdp[0] == 1); /* #24: QUAL='*' read counted */
    assert(qdp[1] == 1); /* L81: mate 2 counted despite rejected overlapping mate 1 */

    int32_t qpos_4101[1] = {110}, qdp_4101[1] = {0}, qdel_4101[1] = {0}, qunc_4101[1] = {0};
    vl_acc_query_positions(acc, 4101, qpos_4101, 1, qdp_4101, qdel_4101, qunc_4101);
    /* Note read 2 had NM=50/50 > 0.20, test a clean read on ctg4101 too: */
    const char *sam2 = "4\t99\tctg4101\t101\t60\t50M\t=\t201\t150\tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t*\tNM:i:0\n";
    /* Wait: read 2 WAS dedup-inserted before NM check; check different pos 102 on ctg5 vs ctg4101 with NM=0! */
    const char *sam3 =
        "5\t99\tctg5\t102\t60\t50M\t=\t202\t150\tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t*\tNM:i:0\n"
        "6\t99\tctg4101\t102\t60\t50M\t=\t202\t150\tAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\t*\tNM:i:0\n";
    vl_acc_feed_sam_bytes(acc, (const uint8_t *)sam3, strlen(sam3));
    vl_acc_get_summary(acc, st, cc, bb, mem);
    assert(st[10] == 0); /* Still zero false PCR duplicates between ctg5 and ctg4101! */

    int32_t rs0[4] = {0}, re0[4] = {0};
    int64_t nruns = vl_acc_extract_callable_runs(acc, 5, 1, rs0, re0, 4);
    assert(nruns == 2);
    assert(rs0[0] == 100 && re0[0] == 151);
    assert(rs0[1] == 300 && re0[1] == 350);

    vl_acc_destroy(acc);
    free(cc); free(bb); free(rseq); free(names); free(noffs); free(lens); free(roffs); free(boffs);
    return 0;
}
""",
        encoding="utf-8",
    )
    vc_bin = tmp_path / "vc_asan"
    subprocess.run(
        [cc, *san_flags, str(NATIVE_DIR / "varloop_core.c"), str(vc_drv_c), "-lpthread", "-lm", "-o", str(vc_bin)],
        check=True,
    )
    vc_work = tmp_path / "vc_work"
    vc_work.mkdir()
    subprocess.run([str(vc_bin), str(vc_work)], env=san_env, check=True)


# ============================================================================
# 2. Portable Native Loader & Cache Security (#74)
# ============================================================================


def test_native_loader_portable_isa_default_and_private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOCALOOP_PORTABLE_ISA", raising=False)
    assert NL._is_portable_isa() is True
    assert LV._is_portable_isa() is True

    monkeypatch.setenv("FOCALOOP_PORTABLE_ISA", "0")
    assert NL._is_portable_isa() is False

    monkeypatch.delenv("FOCALOOP_PORTABLE_ISA", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "custom_tmp"))
    monkeypatch.delenv("FOCALOOP_NATIVE_CACHE", raising=False)
    monkeypatch.delenv("FOCALOOP_BUILD_NATIVE_DIR", raising=False)
    # Force compilation into private per-user cache by hiding sys.prefix cache
    monkeypatch.setattr(NL, "_prebuilt_search_dirs", lambda: [])

    so_path = NL.ensure_shared_lib("libust", ["ust_kernel.c"], use_openmp=False)
    assert so_path.is_file()
    assert so_path.parent != NATIVE_DIR
    mode = stat.S_IMODE(os.lstat(so_path.parent).st_mode)
    assert mode == 0o700
    assert re.search(r"_[0-9a-f]{16}\.(?:so|dylib)$", so_path.name)


# ============================================================================
# 3. Sample ID Collisions (#20), Leading '-' (L94), ENA 3-File (#21),
#    Manifest Path Blocking (#72, L41), BOM/'#' Group Sheets (L32, L36)
# ============================================================================


def test_sample_id_dedup_leading_dash_and_no_local_map_overwrite(tmp_path: Path) -> None:
    assert RV._clean_sample_name("-bad_sample", fallback="s1") == "bad_sample"
    assert RV._clean_sample_name("--genome=/etc/passwd", fallback="s1") == "genome_etc_passwd"
    assert RV._clean_sample_name("sample_1", strip_mate_suffix=False) == "sample_1"
    assert RV._clean_sample_name("sample_2", strip_mate_suffix=False) == "sample_2"

    # Two distinct samples whose raw names collide after punctuation cleaning (`A+1` vs `A-1`)
    # must NOT overwrite each other in local_map or call the same sample twice (#20)!
    ref_fa = TEST_DATA / "ref_chr20_mini.fa"
    r1_var = TEST_DATA / "sample_R1.fastq.gz"
    r2_var = TEST_DATA / "sample_R2.fastq.gz"

    # Create a pure reference (0-variant) sample pair
    _, ref_seqs = G.read_fasta(ref_fa)
    ref_s = ref_seqs[0].decode("ascii")
    r1_wt = tmp_path / "wt_R1.fastq.gz"
    r2_wt = tmp_path / "wt_R2.fastq.gz"
    wt_pairs = [(100 + (i * 37) % (len(ref_s) - 450), 320, None, None) for i in range(180)]
    _write_paired_fastq_gz(r1_wt, r2_wt, ref_s, wt_pairs)

    out_var = tmp_path / "coll.tsv"
    out_vcf = tmp_path / "coll.vcf"
    out_rep = tmp_path / "coll.html"
    state_dir = tmp_path / "coll_state"
    meta = RV.run_variationloop_galaxy(
        reads_r1=[r1_var, r1_wt],
        reads_r2=[r2_var, r2_wt],
        read_ids=["-A+1", "-A_1"],
        ref_fasta=ref_fa,
        threads=2,
        state_dir=state_dir,
        out_variants=out_var,
        out_vcf=out_vcf,
        out_report=out_rep,
    )
    s_names = [s["sample"] for s in meta["samples"]]
    assert s_names == ["A_1", "A_1_2"]
    assert all(not n.startswith("-") for n in s_names)
    df = pd.read_csv(out_var, sep="\t")
    assert set(df["sample_id"].unique()) == {"A_1"}, f"WT sample reused A_1's reads: {df}"


def test_ena_three_file_pairing_and_head_404_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    # #21: ERR330002 has 3 ENA files: ERR330002.fastq.gz;ERR330002_1.fastq.gz;ERR330002_2.fastq.gz
    fake_ena = {
        "ena_runs": ["ERR330002"],
        "all_odp": False,
        "odp_bytes": np.nan,
        "all_paired_fastq": True,
        "fastq_urls": [
            "ftp.sra.ebi.ac.uk/vol1/fastq/ERR330/ERR330002/ERR330002.fastq.gz;"
            "ftp.sra.ebi.ac.uk/vol1/fastq/ERR330/ERR330002/ERR330002_1.fastq.gz;"
            "ftp.sra.ebi.ac.uk/vol1/fastq/ERR330/ERR330002/ERR330002_2.fastq.gz"
        ],
        "fastq_bytes": 3000.0,
        "order_hint": "flowcell",
    }
    monkeypatch.setattr(RV, "_resolve_ena_or_odp", lambda acc, retries=4: fake_ena)
    r1_u, r2_u, kind = RV._resolve_fastq_pair_for_row({"source": "ERR330002"}, {}, Path("/tmp"), "ERR330002")
    assert kind == "remote_fastq"
    assert r1_u.endswith("ERR330002_1.fastq.gz")
    assert r2_u is not None and r2_u.endswith("ERR330002_2.fastq.gz")

    # Single-end run must NOT pair with itself or return fake mate 2
    u1_se, u2_se = RV._match_ena_paired_urls(["https://example.org/SRR1.fastq.gz"])
    assert u1_se == "https://example.org/SRR1.fastq.gz" and u2_se is None

    # L11: _head_content_length must raise on HTTP 404 even when Content-Length header is present
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            if self.path == "/missing":
                self.send_response(404)
                self.send_header("Content-Length", "196")
                self.end_headers()
            else:
                self.send_response(405)
                self.end_headers()

        def do_GET(self):
            if self.path == "/missing":
                self.send_response(404)
                self.send_header("Content-Length", "196")
                self.end_headers()
                self.wfile.write(b"x" * 196)
            else:
                self.send_response(206)
                self.send_header("Content-Range", "bytes 0-0/987654")
                self.send_header("Content-Length", "1")
                self.end_headers()
                self.wfile.write(b"A")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(Exception):
            RV._head_content_length(f"http://127.0.0.1:{port}/missing", timeout=5.0, retries=1)
        assert RV._head_content_length(f"http://127.0.0.1:{port}/range_only", timeout=5.0, retries=1) == 987654
    finally:
        srv.shutdown()


def test_manifest_local_path_blocked_by_default_and_state_sanitized(tmp_path: Path) -> None:
    ref_fa = TEST_DATA / "ref_chr20_mini.fa"
    r1_p = TEST_DATA / "sample_R1.fastq.gz"
    r2_p = TEST_DATA / "sample_R2.fastq.gz"

    man = tmp_path / "local_man.tsv"
    man.write_text(
        f"\ufeff#sample_id\tgroup\tsource\tmate2\n"
        f"s1\tgrpA\t{r1_p}\t{r2_p}\n",
        encoding="utf-8",
    )
    # #72: Must reject arbitrary local filesystem paths in manifest when allow_local_manifest_paths is not set
    with pytest.raises(ValueError, match="local filesystem path"):
        RV.run_variationloop_galaxy(
            manifest_path=man,
            ref_fasta=ref_fa,
            threads=2,
            state_dir=tmp_path / "st_blocked",
            out_variants=tmp_path / "b.tsv",
            out_report=tmp_path / "b.html",
        )

    # When explicitly enabled via allow_local_manifest_paths=True (CLI), succeeds AND strips all local paths from state (L41)
    st_ok = tmp_path / "st_allowed"
    meta = RV.run_variationloop_galaxy(
        manifest_path=man,
        allow_local_manifest_paths=True,
        ref_fasta=ref_fa,
        threads=2,
        state_dir=st_ok,
        out_variants=tmp_path / "ok.tsv",
        out_vcf=tmp_path / "ok.vcf",
        out_report=tmp_path / "ok.html",
    )
    state_txt = (st_ok / "focaloop_state.json").read_text(encoding="utf-8")
    assert str(r1_p) not in state_txt
    assert str(tmp_path) not in state_txt
    assert meta["samples"][0]["reads_counted"] > 0
    assert meta["samples"][0]["median_depth"] > 0
    tel_pq = pd.read_parquet(st_ok / "sample_telemetry.parquet")
    assert "vcf_path" not in tel_pq.columns

    # L32: Duplicate sample_id in manifest raises ValueError
    dup_man = tmp_path / "dup_man.tsv"
    dup_man.write_text("sample_id\tgroup\tsource\ns1\tg1\tSRR123456\ns1\tg2\tSRR123457\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        RV.run_variationloop_galaxy(
            manifest_path=dup_man,
            logan_mode="off",
            ref_fasta=ref_fa,
            state_dir=tmp_path / "st_dup",
            out_variants=tmp_path / "d.tsv",
            out_report=tmp_path / "d.html",
        )

    # L36: Unmatched group sheet raises ValueError
    bad_gs = tmp_path / "bad_gs.tsv"
    bad_gs.write_text("#sample_id\tgroup\nother_sample\tgrpX\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matched 0"):
        RV.run_variationloop_galaxy(
            reads_r1=[r1_p],
            reads_r2=[r2_p],
            read_ids=["actual_sample"],
            manifest_path=bad_gs,
            ref_fasta=ref_fa,
            state_dir=tmp_path / "st_gs",
            out_variants=tmp_path / "gs.tsv",
            out_report=tmp_path / "gs.html",
        )


# ============================================================================
# 4. Input Integrity (#23, #24, #25, #36), BED Padding/Clipping (#28, #29, L38),
#    0-Variant Cohort Merge (#37), and Cross-Worktree State Contract (#62)
# ============================================================================


def test_corrupt_empty_html_and_truncated_inputs_rejected(tmp_path: Path) -> None:
    ref_fa = tmp_path / "ref_chr20_mini.fa"
    shutil.copyfile(TEST_DATA / "ref_chr20_mini.fa", ref_fa)
    g = G.from_fasta(ref_fa, need_core=False)
    good_r1 = TEST_DATA / "sample_R1.fastq.gz"
    good_r2 = TEST_DATA / "sample_R2.fastq.gz"

    # Empty FASTQ
    empty_fq = tmp_path / "empty.fastq.gz"
    empty_fq.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        VG.run_varloop_fastq(g, ref_fa, empty_fq, good_r2, tmp_path / "o1.vcf.gz", map_threads=1, mapper_name="minimap2")

    # HTML error body saved as .fastq.gz
    html_fq = tmp_path / "err.fastq.gz"
    html_fq.write_text("<!DOCTYPE html><html><body>503 Unavailable</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="HTML"):
        VG.run_varloop_fastq(g, ref_fa, html_fq, good_r2, tmp_path / "o2.vcf.gz", map_threads=1, mapper_name="minimap2")

    # Truncated gzip FASTQ (chopped mid-stream)
    trunc_fq = tmp_path / "trunc.fastq.gz"
    raw_gz = good_r1.read_bytes()
    trunc_fq.write_bytes(raw_gz[: len(raw_gz) // 2])
    with pytest.raises(ValueError, match="(?i)truncat|corrupt"):
        VG.run_varloop_fastq(g, ref_fa, trunc_fq, good_r2, tmp_path / "o3.vcf.gz", map_threads=1, mapper_name="minimap2")

    # #25: Block-truncated BAM missing 28-byte BGZF EOF marker
    good_bam = tmp_path / "good_sample.bam"
    subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            f"minimap2 -ax sr {ref_fa} {good_r1} {good_r2} | samtools sort -O bam -o {good_bam} -",
        ],
        check=True,
    )
    trunc_bam = tmp_path / "no_eof.bam"
    bam_bytes = good_bam.read_bytes()
    assert bam_bytes.endswith(VG.BGZF_EOF)
    trunc_bam.write_bytes(bam_bytes[: -len(VG.BGZF_EOF)])
    with pytest.raises(RuntimeError, match="EOF"):
        VG.run_varloop_bam_cram(g, ref_fa, trunc_bam, tmp_path / "o4.vcf.gz", threads=1)

    # #24: BAM aligned to mismatched @SQ contig name ('1' instead of 'chr20_mini')
    other_fa = tmp_path / "other_ref.fa"
    _, seqs = G.read_fasta(ref_fa)
    other_fa.write_bytes(b">1\n" + seqs[0] + b"\n")
    G_other = G.from_fasta(other_fa, need_core=False)
    with pytest.raises(ValueError, match="zero @SQ contigs match"):
        VG.run_varloop_bam_cram(G_other, other_fa, good_bam, tmp_path / "o5.vcf.gz", threads=1)

    # #36: Logan unitigs rejects FASTQ, bz2, empty, and unmapped unitigs
    ctg_map = {str(g.names[0]): int(g.lengths[0])}
    fq_as_u = tmp_path / "reads.fq"
    fq_as_u.write_text("@r1\nACGTACGTACGTACGTACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n", encoding="ascii")
    with pytest.raises(ValueError, match="FASTQ"):
        LV.run_varloop_logan_unitigs(
            acc="LOCAL", sample="s1", ref_fa=ref_fa, ctg_lens=ctg_map,
            out_vcf_gz=tmp_path / "lu1.vcf.gz", work_dir=tmp_path / "lw1", local_unitigs_path=fq_as_u, threads=1,
        )
    unmapped_u = tmp_path / "unmapped.fa"
    unmapped_u.write_text(">0 ka:f:10.0\n" + ("ATATGCGC" * 8) + "\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="zero usable unitigs"):
        LV.run_varloop_logan_unitigs(
            acc="LOCAL", sample="s1", ref_fa=ref_fa, ctg_lens=ctg_map,
            out_vcf_gz=tmp_path / "lu2.vcf.gz", work_dir=tmp_path / "lw2", local_unitigs_path=unmapped_u, threads=1,
        )


def test_narrow_bed_padding_clipping_and_mismatched_bed_error(tmp_path: Path) -> None:
    ref_fa = TEST_DATA / "ref_chr20_mini.fa"
    r1_p = TEST_DATA / "sample_R1.fastq.gz"
    r2_p = TEST_DATA / "sample_R2.fastq.gz"

    # #29: BED with wrong contig names raises ValueError immediately
    bad_bed = tmp_path / "wrong_chr.bed"
    bad_bed.write_text("chr99\t100\t500\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matched 0"):
        RV.run_variationloop_galaxy(
            reads_r1=[r1_p],
            reads_r2=[r2_p],
            read_ids=["s1"],
            ref_fasta=ref_fa,
            targets_bed=bad_bed,
            state_dir=tmp_path / "st_bad_bed",
            out_variants=tmp_path / "bb.tsv",
            out_report=tmp_path / "bb.html",
        )

    # First discover exact variant positions on sample_R1/R2 without BED
    full_tsv = tmp_path / "full.tsv"
    RV.run_variationloop_galaxy(
        reads_r1=[r1_p],
        reads_r2=[r2_p],
        read_ids=["s1"],
        ref_fasta=ref_fa,
        threads=2,
        state_dir=tmp_path / "st_full",
        out_variants=full_tsv,
        out_report=tmp_path / "full.html",
    )
    full_df = pd.read_csv(full_tsv, sep="\t")
    assert len(full_df) == 2
    first_chrom = str(full_df.iloc[0]["chrom"])
    first_pos1 = int(full_df.iloc[0]["pos"])
    second_pos1 = int(full_df.iloc[1]["pos"])

    # #28 + L38: Create a 1-bp hotspot BED around ONLY first_pos1 with host_depletion_mode="on"!
    # Without #28 (+-150 bp padding on 31-mer kset), 0 31-mers were inserted and 0 reads passed;
    # without L38 (clipping to unpadded BED), second_pos1 leaked if within padding!
    hotspot_bed = tmp_path / "hotspot_1bp.bed"
    hotspot_bed.write_text(f"{first_chrom}\t{first_pos1 - 1}\t{first_pos1}\n", encoding="utf-8")
    hot_tsv = tmp_path / "hot.tsv"
    RV.run_variationloop_galaxy(
        reads_r1=[r1_p],
        reads_r2=[r2_p],
        read_ids=["s1"],
        ref_fasta=ref_fa,
        targets_bed=hotspot_bed,
        host_depletion_mode="on",
        threads=2,
        state_dir=tmp_path / "st_hot",
        out_variants=hot_tsv,
        out_report=tmp_path / "hot.html",
    )
    hot_df = pd.read_csv(hot_tsv, sep="\t")
    assert list(hot_df["pos"].astype(int)) == [first_pos1]
    assert second_pos1 not in set(hot_df["pos"].astype(int))


def test_zero_variant_cohort_merge_and_site_nocalls_contract(tmp_path: Path) -> None:
    ref_fa = TEST_DATA / "ref_chr20_mini.fa"
    _, ref_seqs = G.read_fasta(ref_fa)
    ref_s = ref_seqs[0].decode("ascii")

    # #37: Two wild-type samples -> 0 cohort variants -> must NOT SIGSEGV in bcftools merge!
    wt1_r1, wt1_r2 = tmp_path / "wt1_R1.fq.gz", tmp_path / "wt1_R2.fq.gz"
    wt2_r1, wt2_r2 = tmp_path / "wt2_R1.fq.gz", tmp_path / "wt2_R2.fq.gz"
    wt_pairs_full = [(80 + (i * 97) % (len(ref_s) - 420), 320, None, None) for i in range(260)]
    _write_paired_fastq_gz(wt1_r1, wt1_r2, ref_s, wt_pairs_full)
    _write_paired_fastq_gz(wt2_r1, wt2_r2, ref_s, wt_pairs_full)

    wt_vcf = tmp_path / "wt_merged.vcf"
    wt_meta = RV.run_variationloop_galaxy(
        reads_r1=[wt1_r1, wt2_r1],
        reads_r2=[wt1_r2, wt2_r2],
        read_ids=["wt_A", "wt_B"],
        ref_fasta=ref_fa,
        threads=2,
        state_dir=tmp_path / "st_wt",
        out_variants=tmp_path / "wt.tsv",
        out_vcf=wt_vcf,
        out_report=tmp_path / "wt.html",
    )
    assert wt_meta["summary"]["n_variants"] == 0
    vcf_txt = wt_vcf.read_text(encoding="utf-8")
    assert "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\twt_A\twt_B" in vcf_txt

    # #62 State Contract:
    #   - var_sample: calls variants on chr20
    #   - homref_sample: 30x WT coverage across all of chr20 (DP > 0 -> 0/0, NOT in site_nocalls.parquet!)
    #   - dropout_sample: reads ONLY in [50, 420) on chr20 (DP == 0 at variant sites > 500 -> IN site_nocalls.parquet!)
    drop_r1, drop_r2 = tmp_path / "drop_R1.fq.gz", tmp_path / "drop_R2.fq.gz"
    drop_pairs = [(60, 320, None, None) for _ in range(40)]
    _write_paired_fastq_gz(drop_r1, drop_r2, ref_s, drop_pairs)

    st_nc = tmp_path / "st_nocalls"
    nc_vcf = tmp_path / "nc_merged.vcf"
    RV.run_variationloop_galaxy(
        reads_r1=[TEST_DATA / "sample_R1.fastq.gz", wt1_r1, drop_r1],
        reads_r2=[TEST_DATA / "sample_R2.fastq.gz", wt1_r2, drop_r2],
        read_ids=["var_sample", "homref_sample", "dropout_sample"],
        ref_fasta=ref_fa,
        dedup="off",
        threads=2,
        state_dir=st_nc,
        out_variants=tmp_path / "nc.tsv",
        out_vcf=nc_vcf,
        out_report=tmp_path / "nc.html",
    )
    # L41: Merged VCF header must not leak temp .fofn paths (--no-version)
    assert ".fofn" not in nc_vcf.read_text(encoding="utf-8")

    nc_pq = pq.read_table(st_nc / "site_nocalls.parquet")
    assert nc_pq.column_names == ["chrom", "pos", "sample_id"]
    nc_df = nc_pq.to_pandas()
    called_df = pd.read_csv(tmp_path / "nc.tsv", sep="\t")
    far_sites = [int(p) for p in called_df["pos"].unique() if int(p) > 500]
    assert far_sites, f"Expected variant sites > 500 bp, got {called_df['pos'].tolist()}"
    for p_site in far_sites:
        sub_nc = set(nc_df[nc_df["pos"] == p_site]["sample_id"])
        assert "dropout_sample" in sub_nc, f"Uncovered dropout_sample missing from site_nocalls at {p_site}"
        assert "homref_sample" not in sub_nc, f"Covered 0/0 homref_sample falsely marked nocall at {p_site}"


# ============================================================================
# 5. CRAM @SQ Mapping (#22), Last Container (#27), Range Chunks (#42),
#    Feeder Error Propagation (#26), and Focused Telemetry (L37)
# ============================================================================


def test_crai_focused_reordered_sq_last_container_and_http_errors(tmp_path: Path) -> None:
    assert shutil.which("samtools") is not None
    seq_a = _make_random_dna(6000, seed=101)
    seq_b = _make_random_dna(6000, seed=202)

    # CRAM is built with @SQ order: chrA (id=0), chrB (id=1), with reads ONLY on chrB (final container!)
    cram_ref = tmp_path / "cram_order.fa"
    cram_ref.write_text(f">chrA\n{seq_a}\n>chrB\n{seq_b}\n", encoding="ascii")
    subprocess.run(["samtools", "faidx", str(cram_ref)], check=True)

    # Mutate chrB at 0-based 3000 (1-based 3001)
    orig_b = seq_b[3000]
    alt_b = "T" if orig_b != "T" else "C"
    mut_b = seq_b[:3000] + alt_b + seq_b[3001:]
    r1_b, r2_b = tmp_path / "b_R1.fq.gz", tmp_path / "b_R2.fq.gz"
    b_pairs = [(2700 + (i * 7) % 280, 320 + (i % 15), None, None) for i in range(90)]
    _write_paired_fastq_gz(r1_b, r2_b, mut_b, b_pairs)

    cram_p = tmp_path / "focused.cram"
    mapper = shutil.which("minimap2") or shutil.which("bwa")
    assert mapper is not None
    map_cmd = (
        f"minimap2 -ax sr {cram_ref} {r1_b} {r2_b}"
        if "minimap2" in mapper
        else f"bwa index {cram_ref} && bwa mem {cram_ref} {r1_b} {r2_b}"
    )
    subprocess.run(
        ["bash", "-o", "pipefail", "-c", f"{map_cmd} | samtools sort -O cram --reference {cram_ref} -o {cram_p} - && samtools index {cram_p}"],
        check=True,
    )
    crai_p = Path(str(cram_p) + ".crai")

    # Now build a Genome where FASTA contig order is REVERSED (`chrB` first, `chrA` second) -> #22 + #27 + L37!
    rev_ref = tmp_path / "rev_order.fa"
    rev_ref.write_text(f">chrB\n{seq_b}\n>chrA\n{seq_a}\n", encoding="ascii")
    subprocess.run(["samtools", "faidx", str(rev_ref)], check=True)
    g_rev = G.from_fasta(rev_ref, need_core=False)

    out_vcf_local = tmp_path / "crai_local.vcf.gz"
    tel_loc = VG.run_varloop_crai_focused(
        g_rev, rev_ref, cram_p, crai_p, [("chrB", 2950, 3050)], out_vcf_local, sample="S_CRAI", pad_bp=200
    )
    assert tel_loc["sam_records"] > 0
    assert tel_loc["reads_counted"] > 0
    assert tel_loc["n_variants"] == 1
    assert tel_loc["n_snv"] == 1
    assert tel_loc["median_depth"] > 10.0

    # #26 + #42: Serve the CRAM over HTTP Range (testing both clean 206 streaming AND HTTP 500 propagation!)
    cram_raw = cram_p.read_bytes()
    fail_mode = {"mode": None}

    class _CramHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            # Force fallback to Range: bytes=0-0 probe (#27)
            self.send_response(405)
            self.end_headers()

        def do_GET(self):
            rng = self.headers.get("Range", "")
            if fail_mode["mode"] == "500" and rng != "bytes=0-0":
                self.send_response(500)
                self.end_headers()
                return
            if fail_mode["mode"] == "500_body" and rng != "bytes=0-0" and not rng.startswith("bytes=0-"):
                self.send_response(500)
                self.end_headers()
                return
            if fail_mode["mode"] == "200_ignore_range":
                self.send_response(200)
                self.send_header("Content-Length", str(len(cram_raw)))
                self.end_headers()
                self.wfile.write(cram_raw)
                return
            m = re.match(r"bytes=(\d+)-(\d+)", rng)
            assert m is not None
            a, b = int(m.group(1)), int(m.group(2))
            b = min(b, len(cram_raw) - 1)
            chunk = cram_raw[a : b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(cram_raw)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _CramHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/focused.cram"
        tel_rem = VG.run_varloop_crai_focused(
            g_rev, rev_ref, url, crai_p, [("chrB", 2950, 3050)], tmp_path / "crai_rem.vcf.gz", sample="S_REM", pad_bp=200
        )
        assert tel_rem["n_variants"] == 1

        fail_mode["mode"] = "200_ignore_range"
        with pytest.raises(RuntimeError, match="206"):
            VG.run_varloop_crai_focused(
                g_rev, rev_ref, url, crai_p, [("chrB", 2950, 3050)], tmp_path / "crai_200.vcf.gz", sample="S_200", pad_bp=200
            )

        fail_mode["mode"] = "500"
        with pytest.raises(Exception, match="500"):
            VG.run_varloop_crai_focused(
                g_rev, rev_ref, url, crai_p, [("chrB", 2950, 3050)], tmp_path / "crai_500.vcf.gz", sample="S_500", pad_bp=200
            )

        # #26: HTTP 500 inside _feed_cram daemon thread on body range (after header succeeds) propagates cleanly!
        fail_mode["mode"] = "500_body"
        with pytest.raises(Exception, match="500"):
            VG.run_varloop_crai_focused(
                g_rev, rev_ref, url, crai_p, [("chrB", 2950, 3050)], tmp_path / "crai_500b.vcf.gz", sample="S_500B", pad_bp=200
            )
    finally:
        srv.shutdown()


# ============================================================================
# 6. Variant Calling & Local Reassembly Accuracy (#30, #31, #32, L33, L34, L39)
# ============================================================================


def test_reassembly_hom_snv_near_het_indel_compound_het_12_and_haploid_multiclone(tmp_path: Path) -> None:
    # Build a clean 8 kb synthetic contig
    ref_seq = _make_random_dna(8000, seed=777)
    # Ensure exact known bases at test loci:
    # Locus 1 (L39 contig edge, 0-based 24): het 3-bp insertion
    # Locus 2 (#30, 0-based 1500): het 3-bp deletion at 1500 ('CACT'->'C') + HOM 1/1 SNV at 1510 ('G'->'A')
    # Locus 3 (#31, 0-based 3500): trans 1/2 SNV ('G'->'A' on hap1) + 3-bp ins ('G'->'GGAT' on hap2) at same POS 3501!
    # Locus 4 (#32 haploid 2-clone, 0-based 5500): clone 1 has 'C'->'T' at 5500 & 3-bp del at 5700;
    #                                              clone 2 has 'C'->'G' at 5500 & 6-bp del at 5700!
    ref_list = list(ref_seq)
    ref_list[24] = "G"
    ref_list[1500:1504] = list("CACT")
    ref_list[1510] = "G"
    ref_list[3500] = "G"
    ref_list[5500] = "C"
    ref_list[5700:5710] = list("GCATGATCAT")
    ref_s = "".join(ref_list)

    fa_p = tmp_path / "synth.fa"
    fa_p.write_text(f">chrS\n{ref_s}\n", encoding="ascii")
    g = G.from_fasta(fa_p, need_core=False)

    # Construct diploid haplotypes H1 and H2 directly in SAM (coordinate-paired, exact CIGARs)
    sam_recs = [f"@HD\tVN:1.6\n@SQ\tSN:chrS\tLN:{len(ref_s)}"]
    spot = 1
    # L39: Contig-edge het insertion at pos0=24 (1-based 25): 20 ref reads + 20 3I reads
    for k in range(20):
        s_ref = ref_s[0:100]
        sam_recs.append(f"{spot}\t99\tchrS\t1\t60\t100M\t=\t120\t220\t{s_ref}\t{'I'*100}\tNM:i:0")
        spot += 1
        s_ins = ref_s[0:25] + "TTA" + ref_s[25:97]
        sam_recs.append(f"{spot}\t99\tchrS\t1\t60\t25M3I72M\t=\t120\t220\t{s_ins}\t{'I'*100}\tNM:i:3")
        spot += 1

    # #30: Het deletion 1501 CACT->C (hap1) + Hom 1/1 SNV 1511 G->A (BOTH hap1 and hap2!)
    for k in range(25):
        off = 1430 + (k % 25)
        # Hap 1: 3-bp deletion at 1501..1503 AND G->A at 1510
        left_len = 1501 - off
        s_h1 = ref_s[off:1501] + ref_s[1504:1510] + "A" + ref_s[1511 : off + 153]
        cig1 = f"{left_len}M3D{150 - left_len}M"
        sam_recs.append(f"{spot}\t99\tchrS\t{off + 1}\t60\t{cig1}\t=\t{off + 200}\t350\t{s_h1}\t{'I'*150}\tNM:i:4")
        spot += 1
        # Hap 2: no indel, ONLY G->A at 1510 (NM:i:1 -> uncaptured in non-zoom window!)
        s_h2 = ref_s[off:1510] + "A" + ref_s[1511 : off + 150]
        sam_recs.append(f"{spot}\t99\tchrS\t{off + 1}\t60\t150M\t=\t{off + 200}\t350\t{s_h2}\t{'I'*150}\tNM:i:1")
        spot += 1

    # #31: Compound het at 0-based 3500 (1-based 3501): hap1 has G->A (NM:i:1), hap2 has G->GGAT (3I)
    for k in range(28):
        off = 3430 + (k % 25)
        s_snv = ref_s[off:3500] + "A" + ref_s[3501 : off + 150]
        sam_recs.append(f"{spot}\t99\tchrS\t{off + 1}\t60\t150M\t=\t{off + 200}\t350\t{s_snv}\t{'I'*150}\tNM:i:1")
        spot += 1
        left_l = 3501 - off
        s_ins3 = ref_s[off:3501] + "GAT" + ref_s[3501 : off + 147]
        sam_recs.append(f"{spot}\t99\tchrS\t{off + 1}\t60\t{left_l}M3I{147 - left_l}M\t=\t{off + 200}\t350\t{s_ins3}\t{'I'*150}\tNM:i:3")
        spot += 1

    sam_bytes = ("\n".join(sam_recs) + "\n").encode("ascii")
    for reasm in (True, False):
        dp = VG.DiploidParams(enable_reassembly=reasm, dedup_bits=0, ploidy="diploid")
        with VG.PagedContigAccumulator(g, dp) as acc:
            acc.feed_bytes(sam_bytes)
            calls = VG.call_contig_diploid(g, acc, 0, dp)
        by_pos = {c["pos"]: c for c in calls}
        # L39: contig-edge het insertion at pos 25 stays 0/1!
        assert 25 in by_pos and by_pos[25]["gt"] == "0/1", f"L39 failed (reasm={reasm}): {by_pos.get(25)}"
        # #30: 1501 CACT->C is 0/1 AND 1511 G->A is 1/1!
        assert 1501 in by_pos and by_pos[1501]["gt"] == "0/1", f"#30 indel failed (reasm={reasm}): {calls}"
        assert 1511 in by_pos and by_pos[1511]["gt"] == "1/1", f"#30 hom SNV downgraded (reasm={reasm}): {by_pos.get(1511)}"
        # #31: 3501 has GT == 1/2 combining A and GGAT!
        assert 3501 in by_pos and by_pos[3501]["gt"] == "1/2", f"#31 compound het failed (reasm={reasm}): {by_pos.get(3501)}"
        assert set(by_pos[3501]["alt"].split(",")) == {"A", "GGAT"}

    # #32: Haploid 50/50 two-clone mixture (two SNV alts at 5501 + two deletion alts at 5701)
    hap_sam = [f"@HD\tVN:1.6\n@SQ\tSN:chrS\tLN:{len(ref_s)}"]
    sp_h = 1
    for k in range(24):
        off1 = 5440 + (k % 20)
        hap_sam.append(f"{sp_h}\t99\tchrS\t{off1 + 1}\t60\t150M\t=\t{off1 + 200}\t350\t{ref_s[off1:5500] + 'T' + ref_s[5501:off1+150]}\t{'I'*150}\tNM:i:1")
        sp_h += 1
        hap_sam.append(f"{sp_h}\t99\tchrS\t{off1 + 1}\t60\t150M\t=\t{off1 + 200}\t350\t{ref_s[off1:5500] + 'G' + ref_s[5501:off1+150]}\t{'I'*150}\tNM:i:1")
        sp_h += 1
        off2 = 5640 + (k % 20)
        ll = 5701 - off2
        s_d3 = ref_s[off2:5701] + ref_s[5704 : off2 + 153]
        hap_sam.append(f"{sp_h}\t99\tchrS\t{off2 + 1}\t60\t{ll}M3D{150 - ll}M\t=\t{off2 + 200}\t350\t{s_d3}\t{'I'*150}\tNM:i:3")
        sp_h += 1
        s_d6 = ref_s[off2:5701] + ref_s[5707 : off2 + 156]
        hap_sam.append(f"{sp_h}\t99\tchrS\t{off2 + 1}\t60\t{ll}M6D{150 - ll}M\t=\t{off2 + 200}\t350\t{s_d6}\t{'I'*150}\tNM:i:6")
        sp_h += 1
    hp_params = VG.DiploidParams(enable_reassembly=True, dedup_bits=0, ploidy="haploid", haploid_min_wsaf=0.20)
    with VG.PagedContigAccumulator(g, hp_params) as acc_h:
        acc_h.feed_bytes(("\n".join(hap_sam) + "\n").encode("ascii"))
        h_calls = VG.call_contig_diploid(g, acc_h, 0, hp_params)
    snvs_5501 = [c for c in h_calls if c["pos"] == 5501]
    dels_5701 = [c for c in h_calls if c["pos"] == 5701]
    assert {c["alt"] for c in snvs_5501} == {"T", "G"}, f"#32 missing second haploid SNV clone: {snvs_5501}"
    assert len(dels_5701) == 2, f"#32 missing second haploid indel clone: {dels_5701}"
    for c_item in snvs_5501 + dels_5701:
        assert 0.35 <= c_item["wsaf"][0] <= 0.65, f"#32 inflated haploid WSAF: {c_item}"

    # L33 & L34: Logan diploid 1/2 multiallelic genotyping + "01" / "NA" string preservation
    gt_12, gq_12 = LV._multiallelic_diploid_gt(ad_ref=0.0, ad_alt1=22.0, ad_alt2=20.0, err=0.005)
    assert gt_12 == "1/2" and gq_12 > 20


# ============================================================================
# 7. WHO kelch13 Missingness (#33), CNV NaN/Sample_type (#34),
#    Sorted xsra Cap (#35), and --dedup Toggle (#71, L43)
# ============================================================================


def test_kelch13_partial_missingness_undetermined_and_cnv_sample_types() -> None:
    # #33: WT codons on covered sites + 12 uncovered propeller codons -> Undetermined (never false Sensitive!)
    res_gap = WR.classify_sample(
        {
            "crt_76": "K",
            "dhfr_51": "N",
            "dhfr_59": "C",
            "dhfr_108": "S",
            "dhfr_164": "I",
            "dhps_437": "A",
            "dhps_540": "K",
            "dhps_581": "A",
            "dhps_613": "A",
            "k13_ns": "None",
            "k13_missing_codons": 12,
            "mdr1_amp": 0,
            "pm2_pm3_amp": 0,
        }
    )
    assert res_gap["Artemisinin"] == "Undetermined"
    assert WR.classify_artemisinin("A578S", k13_has_missing=True) == "Undetermined"
    assert WR.classify_artemisinin("None", k13_has_missing=False) == "Sensitive"

    tel_df = pd.DataFrame([{"sample": "iso1", "k13_ns": "None", "aa_k13_missing_codons": 8, "median_core_depth": 20.0}])
    meta_df = pd.DataFrame([{"Sample": "iso1", "Country": "GH", "Population": "AF-W"}])
    drm_df = RV._format_malariavar_drm_table(tel_df, pd.DataFrame(), meta_df)
    assert drm_df.iloc[0]["who_artemisinin"] == "Undetermined"
    assert "8 codons uncovered" in drm_df.iloc[0]["kelch13_mutations"]

    # #34: Zero-coverage / NaN bin_median returns -1 across gDNA, gdna, sWGA, MDA; invalid sample_type raises ValueError
    assert C._normalize_sample_type("gdna") == "gdna"
    assert C._normalize_sample_type("sWGA") == "swga"
    assert C._normalize_sample_type("MDA") == "mda"
    with pytest.raises(ValueError, match="Unsupported sample_type"):
        C._normalize_sample_type("RNAseq")

    # Build a minimal mock Genome with curated Pf gene coordinates to verify cnv_calls_v2 on NaN bin_median
    names = [f"Pf3D7_{i:02d}_v3" for i in range(1, 15)]
    lengths = np.full(14, 100_000, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    seq = np.zeros(int(offsets[-1]), dtype=np.uint8)
    core = np.ones(int(offsets[-1]), dtype=bool)
    boff = np.concatenate([[0], np.cumsum(lengths // 1000)])
    bgc = np.full(int(boff[-1]), 0.20, dtype=np.float32)
    bcore = np.ones(int(boff[-1]), dtype=np.float32)
    genes = {}
    for locus_keys in [*G.CNV_LOCI.values(), *[[v] for v in G.DEL_LOCI.values()]]:
        for gk in locus_keys:
            genes[G.GENES.get(gk, gk)] = {
                "contig": "Pf3D7_05_v3",
                "start": 40_000,
                "end": 44_000,
                "strand": "+",
                "cds": [(40_000, 44_000)],
                "exons": [(40_000, 44_000)],
            }
    g_mock = G.Genome(names=names, lengths=lengths, offsets=offsets, seq=seq, core=core, bin_offsets=boff, bin_gc=bgc, bin_core=bcore, genes=genes)
    cp = C.CallParams()
    acc_zero = C.Accumulator(g_mock, cp)  # 0 reads -> bin_median is NaN!
    for stype in ("gDNA", "gdna", "sWGA", "MDA"):
        cnv_out, _, _, bmed = C.cnv_calls_v2(g_mock, acc_zero, cp, sample_type=stype)
        assert np.isnan(bmed)
        assert all(row["call"] == -1 for row in cnv_out), f"#34 emitted false 0 CNV call on empty {stype}: {cnv_out}"


def test_pcr_amplicon_dedup_toggle_and_warning_across_both_kernels(tmp_path: Path) -> None:
    ref_s = _make_random_dna(2000, seed=909)
    alt_base = "G" if ref_s[600] != "G" else "T"
    mut_s = ref_s[:600] + alt_base + ref_s[601:]
    ref_fa = tmp_path / "amp_ref.fa"
    ref_fa.write_text(f">chrAmp\n{ref_s}\n", encoding="ascii")

    # 120 amplicon read pairs ALL starting at exact position 520 with exact fragment length 220!
    r1_amp = tmp_path / "amp_R1.fq.gz"
    r2_amp = tmp_path / "amp_R2.fq.gz"
    amp_pairs = [(520, 220, None, None) for _ in range(120)]
    _write_paired_fastq_gz(r1_amp, r2_amp, mut_s, amp_pairs, read_len=120)

    # 1. With dedup="auto", >95% duplicates triggers high_dup_samples warning banner
    rep_auto = tmp_path / "amp_auto.html"
    meta_auto = RV.run_variationloop_galaxy(
        reads_r1=[r1_amp],
        reads_r2=[r2_amp],
        read_ids=["amp_sample"],
        ref_fasta=ref_fa,
        dedup="auto",
        threads=2,
        state_dir=tmp_path / "st_amp_auto",
        out_variants=tmp_path / "amp_auto.tsv",
        out_report=rep_auto,
    )
    assert meta_auto["summary"].get("high_dup_samples") == ["amp_sample"]
    assert "High PCR Duplicate Rate Notice" in rep_auto.read_text(encoding="utf-8")

    # 2. With dedup="off", all 120 amplicon pairs are retained and the 1/1 SNV at pos 601 is called with DP >= 100!
    tsv_off = tmp_path / "amp_off.tsv"
    RV.run_variationloop_galaxy(
        reads_r1=[r1_amp],
        reads_r2=[r2_amp],
        read_ids=["amp_sample"],
        ref_fasta=ref_fa,
        dedup="off",
        threads=2,
        state_dir=tmp_path / "st_amp_off",
        out_variants=tsv_off,
        out_report=tmp_path / "amp_off.html",
    )
    df_off = pd.read_csv(tsv_off, sep="\t")
    assert len(df_off) == 1
    assert int(df_off.iloc[0]["pos"]) == 601
    assert float(df_off.iloc[0]["dp"]) >= 100

    # 3. Also verify malariavar Numba kernel CallParams(dedup=False) vs CallParams(dedup=True) (L43)
    g = G.from_fasta(ref_fa, need_core=False)
    sam_lines = [f"@HD\tVN:1.6\n@SQ\tSN:chrAmp\tLN:{len(ref_s)}"]
    for i in range(40):
        sam_lines.append(f"{i+1}\t99\tchrAmp\t521\t60\t120M\t=\t621\t220\t{mut_s[520:640]}\t{'I'*120}\tNM:i:1")
    sam_arr = np.frombuffer(("\n".join(sam_lines) + "\n").encode("ascii"), dtype=np.uint8)

    acc_on = C.Accumulator(g, C.CallParams(dedup=True))
    acc_on.feed(sam_arr, len(sam_arr))
    assert int(acc_on.stats[10]) == 39

    acc_off = C.Accumulator(g, C.CallParams(dedup=False))
    acc_off.feed(sam_arr, len(sam_arr))
    assert int(acc_off.stats[10]) == 0
    assert int(acc_off.counts[600].sum()) == 40


# ============================================================================
# 8. Deadlock Regressions (#38, #39, #40) & Resource Guards (#41, #43, #44)
# ============================================================================


def _top_level_suicide_worker(cid: int):
    os.kill(os.getpid(), signal.SIGKILL)


def test_deadlock_regressions_stderr_pipe_broken_pool_and_batch_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # #38: Fake mapper writing 256 KiB to stderr (> 64 KiB Linux pipe buffer!) before closing stdout must NOT deadlock
    fake_bin_dir = tmp_path / "bin"
    fake_bin_dir.mkdir()
    fake_mm2 = fake_bin_dir / "minimap2"
    ref_fa = TEST_DATA / "ref_chr20_mini.fa"
    rnames, rseqs = G.read_fasta(ref_fa)
    rname0 = rnames[0]
    rseq_s = rseqs[0].decode("ascii")[:100]
    fake_mm2.write_text(
        f"#!/usr/bin/env python3\n"
        f"import sys\n"
        f"sys.stderr.write('W' * 262144 + '\\n')\n"
        f"sys.stderr.flush()\n"
        f"sys.stdout.write('@HD\\tVN:1.6\\n@SQ\\tSN:{rname0}\\tLN:{len(rseqs[0])}\\n')\n"
        f"sys.stdout.write('1\\t99\\t{rname0}\\t1\\t60\\t100M\\t=\\t150\\t250\\t{rseq_s}\\t' + 'I'*100 + '\\tNM:i:0\\n')\n"
        f"sys.stdout.flush()\n",
        encoding="utf-8",
    )
    fake_mm2.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin_dir}:{os.environ.get('PATH', '')}")

    g = G.from_fasta(ref_fa, need_core=False)
    t0 = time.monotonic()
    tel = VG.run_varloop_fastq(
        g,
        ref_fa,
        TEST_DATA / "sample_R1.fastq.gz",
        TEST_DATA / "sample_R2.fastq.gz",
        tmp_path / "noisy.vcf.gz",
        map_threads=1,
        mapper_name="minimap2",
    )
    assert time.monotonic() - t0 < 10.0
    assert tel["reads_counted"] == 1

    # #39: Verify reduced batch flags (-K 1M / -K 1000000) in streamer.mapper_cmd
    mm2_cmd = ST.mapper_cmd(4, True, ref="ref.fa", mapper="minimap2")
    assert "-K" in mm2_cmd and mm2_cmd[mm2_cmd.index("-K") + 1] == "1M"
    bwa_cmd = ST.mapper_cmd(4, True, ref="ref.fa", mapper="bwa-mem2")
    assert "-K" in bwa_cmd and bwa_cmd[bwa_cmd.index("-K") + 1] == "1000000"

    # #40: SIGKILL inside forked contig caller raises BrokenProcessPool promptly instead of hanging forever
    two_fa = tmp_path / "two_ctgs.fa"
    two_fa.write_text(f">c1\n{_make_random_dna(1000, 1)}\n>c2\n{_make_random_dna(1000, 2)}\n", encoding="ascii")
    g_two = G.from_fasta(two_fa, need_core=False)

    monkeypatch.setattr(VG, "_fork_call_contig_one", _top_level_suicide_worker)
    t_kill = time.monotonic()
    with VG.PagedContigAccumulator(g_two, VG.DiploidParams()) as acc2:
        with pytest.raises(BrokenProcessPool):
            VG.call_contig_diploid_batch(g_two, acc2, [0, 1], VG.DiploidParams(), call_workers=2)
    assert time.monotonic() - t_kill < 10.0


def test_logan_span_gate_ram_aligner_selection_and_per_sample_temp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #44: RAM-aware aligner selection picks classic bwa when 28 * ref_bp > 0.70 * avail_ram
    if shutil.which("bwa"):
        aln_name, _ = LV._select_var_aligner("auto", ref_bp=3_100_000_000, avail_ram=16 << 30)
        assert aln_name == "bwa"

    # #41: > 400 Mb reference span without --targets-bed must raise before index build in logan_only mode,
    # and fall back without calling ensure_var_aligner_index in auto mode!
    index_called = []
    monkeypatch.setattr(LV, "ensure_var_aligner_index", lambda *a, **k: index_called.append(True))
    fake_g = G.Genome(
        names=["chr1"],
        lengths=np.array([450_000_000], dtype=np.int64),
        offsets=np.array([0, 450_000_000], dtype=np.int64),
        seq=np.zeros(100, dtype=np.uint8),
        core=np.empty(0, dtype=bool),
        bin_offsets=np.array([0, 1], dtype=np.int64),
        bin_gc=np.zeros(1, dtype=np.float32),
        bin_core=np.zeros(1, dtype=np.float32),
        genes={},
    )
    small_fa = TEST_DATA / "ref_chr20_mini.fa"
    monkeypatch.setattr(G, "from_fasta", lambda *a, **k: fake_g)
    with pytest.raises(ValueError, match="400 Mb"):
        RV.run_variationloop_galaxy(
            sra_accessions="SRR999999",
            logan_mode="logan_only",
            ref_fasta=small_fa,
            state_dir=tmp_path / "st_400",
            out_variants=tmp_path / "v400.tsv",
            out_report=tmp_path / "r400.html",
        )
    assert index_called == []


def test_per_sample_remote_fastq_cleanup_and_minor_helpers(tmp_path: Path) -> None:
    # L44: kernels.runs_ge must never truncate high-density alternating runs (> len(depth) // 50)
    depth = np.tile(np.array([10, 0], dtype=np.int32), 500)  # len=1000 -> 500 runs (old cap was 1000//50 = 20!)
    runs = Kn.runs_ge(depth, 5, np.array([0, 1000], dtype=np.int64))
    assert len(runs) == 500

    # L74: DotHashCohortIndex clamps k_top > n_samples and handles n_samples == 0 cleanly
    dh_idx = DH.DotHashCohortIndex.from_syncmer_sets([np.array([1, 2, 3], dtype=np.uint64), np.array([2, 3, 4], dtype=np.uint64)])
    ti, ts = dh_idx.knn_1_vs_n(dh_idx.matrix[0], k_top=10)
    assert len(ti) == 2 and all(int(x) >= 0 for x in ti)
    dh_empty = DH.DotHashCohortIndex.from_syncmer_sets([])
    ti0, ts0 = dh_empty.knn_1_vs_n(np.zeros(DH.DOTHASH_DIM, dtype=np.int16), k_top=5)
    assert len(ti0) == 0 and len(ts0) == 0

    # L30: streamer.make_source supports local paired FASTQ and raises clear ValueError when no source exists
    r1_p = TEST_DATA / "sample_R1.fastq.gz"
    r2_p = TEST_DATA / "sample_R2.fastq.gz"
    src_local = ST.make_source(
        {
            "Sample": "loc1",
            "all_odp": False,
            "all_paired_fastq": True,
            "fastq_urls": [f"{r1_p};{r2_p}"],
        },
        prefer="ena",
    )
    chunk = src_local.read(4096)
    src_local.close()
    assert chunk.startswith(b"@")
    with pytest.raises(ValueError, match="no usable"):
        ST.make_source({"Sample": "empty_src", "all_odp": False, "all_paired_fastq": False})

    # #43: Serve 2 samples' FASTQ files over HTTP and verify materialized fq files are unlinked per sample
    r1_bytes = r1_p.read_bytes()
    r2_bytes = r2_p.read_bytes()

    class _FqHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            payload = r1_bytes if "_1.fastq.gz" in self.path else r2_bytes
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FqHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    max_fq_files_seen = [0]
    orig_run_fq = VG.run_varloop_fastq

    def _spy_run_fq(g_arg, ref_arg, r1_u, r2_u, *args, **kwargs):
        fq_dir = Path(r1_u).parent
        if fq_dir.name == "fq":
            max_fq_files_seen[0] = max(max_fq_files_seen[0], len(list(fq_dir.glob("*.fq.gz"))))
        return orig_run_fq(g_arg, ref_arg, r1_u, r2_u, *args, **kwargs)

    try:
        man = tmp_path / "rem_fq.tsv"
        man.write_text(
            f"sample_id\tgroup\tsource\tmate2\n"
            f"remA\tg1\thttp://127.0.0.1:{port}/remA_1.fastq.gz\thttp://127.0.0.1:{port}/remA_2.fastq.gz\n"
            f"remB\tg2\thttp://127.0.0.1:{port}/remB_1.fastq.gz\thttp://127.0.0.1:{port}/remB_2.fastq.gz\n",
            encoding="utf-8",
        )
        VG.run_varloop_fastq = _spy_run_fq
        try:
            RV.run_variationloop_galaxy(
                manifest_path=man,
                logan_mode="off",
                ref_fasta=TEST_DATA / "ref_chr20_mini.fa",
                threads=2,
                state_dir=tmp_path / "st_rem_fq",
                out_variants=tmp_path / "rem.tsv",
                out_report=tmp_path / "rem.html",
            )
        finally:
            VG.run_varloop_fastq = orig_run_fq
        # At most 1 sample's R1+R2 (2 files) may exist in work_dir/fq at any time (#43)!
        assert max_fq_files_seen[0] == 2
    finally:
        srv.shutdown()


# ============================================================================
# 9. State Store Deduplication & Portable Warm Bundle (L42, L46) + XML Hygiene
# ============================================================================


def test_state_store_dedup_portable_warm_and_xml_dedup_param(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # L42 & L46: Write duplicate sample rows across shard + sample dir, build_warm, and load Warm without host REFDIR!
    cohort = tmp_path / "cohort_l42"
    w_arr = np.array([10, 200, 0, 255], dtype=np.uint8)
    d_arr = np.array([20, 25, 18, 0], dtype=np.uint8)
    r_arr = np.array([1.0, 1.1], dtype=np.float16)

    for s_id, fws_v in [("isoA", 0.99), ("isoB", 0.65)]:
        SS.write_sample(
            cohort,
            s_id,
            {
                "wsaf_sketch": pd.DataFrame({"sample": [s_id], "n_sites": [4], "wsaf_u8": [w_arr.tobytes()], "dp_u8": [d_arr.tobytes()]}),
                "depth_1kb_bins": pd.DataFrame({"sample": [s_id], "ratio_f16": [r_arr.tobytes()], "bin_median": [20.0]}),
                "sample_telemetry": pd.DataFrame([{"sample": s_id, "stop_reason": "done", "error": None, "fws": fws_v}]),
                "variant_calls": pd.DataFrame({"sample": [s_id], "chrom": [0], "pos": [100], "ref": ["A"], "alt": ["G"], "kind": [0], "gt": [1], "wsaf": [0.9]}),
                "cnv_del_calls": pd.DataFrame({"sample": [s_id], "locus": ["mdr1"], "call": [0]}),
            },
        )
    SS.seal_shard(cohort, 0, ["isoA", "isoA", "isoB"])
    # Also write an unsealed rerun of isoA to verify build_warm deduplicates across shard + samples/!
    SS.write_sample(
        cohort,
        "isoA",
        {
            "wsaf_sketch": pd.DataFrame({"sample": ["isoA"], "n_sites": [4], "wsaf_u8": [w_arr.tobytes()], "dp_u8": [d_arr.tobytes()]}),
            "depth_1kb_bins": pd.DataFrame({"sample": ["isoA"], "ratio_f16": [r_arr.tobytes()], "bin_median": [22.0]}),
            "sample_telemetry": pd.DataFrame([{"sample": "isoA", "stop_reason": "done", "error": None, "fws": 0.98}]),
            "variant_calls": pd.DataFrame({"sample": ["isoA"], "chrom": [0], "pos": [100], "ref": ["A"], "alt": ["G"], "kind": [0], "gt": [1], "wsaf": [0.9]}),
            "cnv_del_calls": pd.DataFrame({"sample": ["isoA"], "locus": ["mdr1"], "call": [0]}),
        },
    )
    meta = pd.DataFrame({"Sample": ["isoA", "isoA", "isoB"], "Country": ["GH", "GH", "KH"], "Population": ["AF-W", "AF-W", "AS-SE-E"]})
    warm_res = SS.build_warm(cohort, meta=meta, use_dothash_knn=True)
    assert warm_res["samples"] == 2

    # Inject self-contained panel/genome/genes into warm/ and verify Q.Warm loads with host P.load_panel disabled (L46)!
    w_dir = cohort / "warm"
    np.savez_compressed(w_dir / "panel.npz", goff=np.array([10, 20, 30, 40], dtype=np.int64), af=np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    np.savez_compressed(
        w_dir / "genome_meta.npz",
        names=np.array(["chr1"]),
        lengths=np.array([2000], dtype=np.int64),
        offsets=np.array([0, 2000], dtype=np.int64),
        bin_offsets=np.array([0, 2], dtype=np.int64),
        bin_gc=np.array([0.2, 0.2], dtype=np.float32),
        bin_core=np.array([1.0, 1.0], dtype=np.float32),
    )
    (w_dir / "genes_curated.json").write_text("{}")
    pd.DataFrame({"contig": ["chr1"], "start": [1], "end": [1000], "gene_id": ["G1"], "name": ["gene1"]}).to_parquet(w_dir / "genes_all.parquet", index=False)
    (w_dir / "bundle_manifest.json").write_text(json.dumps({"n_samples": 2, "n_sites": 4}))

    monkeypatch.setattr(Q.P, "load_panel", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("host panel must not be used")))
    monkeypatch.setattr(Q.G, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("host genome must not be used")))
    W = Q.Warm(cohort, load_dothash=True)
    assert len(W.samples) == 2
    knn_out = Q.q_dothash_knn(W, sample_or_idx="isoA", k_top=10)
    assert len(knn_out["neighbors"]) == 1 and knn_out["neighbors"][0]["sample"] == "isoB"
    err_oob = Q.q_dothash_knn(W, sample_or_idx=99, k_top=3)
    assert "error" in err_oob

    # XML verification (#71 & L94): dedup param exposed cleanly with auto/on/off and leading '-' stripped in IDs
    root = ET.parse(TOOL_XML).getroot()
    dedup_el = root.find(".//section[@name='calling_options']/param[@name='dedup']")
    assert dedup_el is not None
    opts = [o.attrib["value"] for o in dedup_el.findall("option")]
    assert opts == ["auto", "on", "off"]
    cmd_txt = root.find("command").text or ""
    assert "lstrip(\"-\")" in cmd_txt
