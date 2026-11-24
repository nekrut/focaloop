"""Comprehensive 0.4.1 code-review regression suite for Focaloop ExpressionLoop."""

from __future__ import annotations

import ast
import gzip
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from focaloop.hooks import HypothesisLedger
from focaloop.panexpr import _fast_c
from focaloop.panexpr._medvedev_compile import (
    ensure_user_native_cache_dir,
    verify_native_artifact,
)
from focaloop.panexpr.catalog import (
    AssemblySpec,
    build_catalog,
    build_prokaryotic_targets,
    build_spliced_transcripts,
    load_fasta,
    restore_exons_from_annotation,
)
from focaloop.panexpr.kernels import (
    KallistoKernel,
    MapResult,
    MappyKernel,
    PiscemKernel,
    RunAccumulator,
    SalmonKernel,
    SamKernel,
    SENTINEL_ID,
    _sentinel_seq,
    read_rad_bulk,
    revcomp_str,
)
from focaloop.panexpr.logan_expr import (
    LoganZeroMappingError,
    _aligner_cmd,
    _select_unitig_aligner,
    detect_antisense_genes,
    run_logan_expression_cohort,
    stream_logan_expression_run,
)
from focaloop.panexpr.loop import Focaloop, LoopConfig
from focaloop.panexpr.manifest import FastqFile, Run, StudyManifest, parse_ena_filereport
from focaloop.panexpr.regime import (
    AssemblyInput,
    annotation_features,
    available_ram_bytes,
    classify,
)
from focaloop.panexpr.solver import (
    gene_level_tables,
    quantify_cohort,
    refit_genes_with_depth_offsets,
    run_pydeseq2,
    validate_additive_design,
)
from focaloop.panexpr.sra_odp import SpotBlockPlan, XsraRunStream
from focaloop.panexpr.streamer import (
    FastqPrefixStream,
    GzipStreamDecoder,
    HttpRangeSource,
    LocalFileSource,
    RunStream,
    open_run_stream,
)
from focaloop.panexpr.tournament import KernelScore, run_tournament
from focaloop.panexpr.zoom import ZoomResult
from focaloop.runners_expression import (
    _detect_organism_mode,
    _head_content_length,
    resolve_expression_manifest,
    run_expressionloop_galaxy,
)
from focaloop.state import FocaloopStateStore

REPO = Path(__file__).resolve().parents[1]
PANEXPR_DIR = REPO / "focaloop" / "panexpr"
EXPR_XML = REPO / "tools" / "focaloop" / "expressionloop.xml"
TEST_DATA = REPO / "tools" / "focaloop" / "test-data" / "expressionloop"


def _build_synthetic_rad(
    ref_names: list[str],
    chunks: list[list[tuple[int, list[tuple[int, int]]]]],
    *,
    expected_chunks_override: int | None = None,
    fmt_in_aln_tag: bool = False,
) -> bytes:
    """Build a valid RAD binary payload with frag_map_type (1=both, 2=first_orphan, 3=second_orphan)."""
    buf = bytearray()
    buf.extend(b"RAD_FILE")
    buf.extend(b"\x00" * 6)
    buf.append(1)  # is_paired
    buf.extend(struct.pack("<Q", len(ref_names)))
    for nm in ref_names:
        b = nm.encode("utf-8")
        buf.extend(struct.pack("<H", len(b)))
        buf.extend(b)
    exp_c = len(chunks) if expected_chunks_override is None else expected_chunks_override
    buf.extend(struct.pack("<Q", exp_c))

    def _write_tag_descs(tags: list[tuple[str, int]]) -> None:
        buf.extend(struct.pack("<H", len(tags)))
        for tid_i, (tnm, tty) in enumerate(tags):
            tb = tnm.encode("utf-8")
            buf.extend(struct.pack("<H", len(tb)))
            buf.extend(tb)
            buf.append(tty)
            buf.extend(struct.pack("<H", tid_i))

    _write_tag_descs([])  # file_tags
    if fmt_in_aln_tag:
        _write_tag_descs([])  # read_tags
        _write_tag_descs([("frag_map_type", 1), ("compressed_ori_ref", 3)])  # aln_tags
    else:
        _write_tag_descs([("frag_map_type", 1)])  # read_tags (uint8)
        _write_tag_descs([("compressed_ori_ref", 3)])  # aln_tags (uint32)

    for chunk_recs in chunks:
        cbuf = bytearray()
        for fmt_val, alns in chunk_recs:
            cbuf.extend(struct.pack("<I", len(alns)))
            if not fmt_in_aln_tag:
                cbuf.append(fmt_val & 0xFF)
            for ref_idx, raw_ori in alns:
                if fmt_in_aln_tag:
                    cbuf.append(fmt_val & 0xFF)
                comp = ((raw_ori & 0x3) << 30) | (ref_idx & 0x3FFFFFFF)
                cbuf.extend(struct.pack("<I", comp))
        nbytes = 8 + len(cbuf)
        buf.extend(struct.pack("<II", nbytes, len(chunk_recs)))
        buf.extend(cbuf)
    return bytes(buf)


def test_c99_asan_ubsan_compilation_and_fixes(tmp_path: Path) -> None:
    """Compile all modified C99 sources under -fsanitize=address,undefined and exercise L75-L78 & #11."""
    cc = shutil.which("gcc") or shutil.which("clang")
    assert cc is not None, "C compiler required for ASan/UBSan verification"
    san_flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    asan_env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1", "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}

    # 1. expr_sam_ec_v2.c (L75): atomic temp write + ferror/fclose checks
    ec_src = PANEXPR_DIR / "csrc" / "expr_sam_ec_v2.c"
    ec_bin = tmp_path / "expr_sam_ec_v2_asan"
    subprocess.run([cc, *san_flags, str(ec_src), "-o", str(ec_bin), "-lm"], check=True)

    t2g = tmp_path / "tx2gene.tsv"
    t2g.write_text("transcript_id\tgene_id\ntxA\tgeneA\ntxB\tgeneB\n", encoding="utf-8")
    sam_in = (
        "@HD\tVN:1.6\n"
        "@SQ\tSN:txA\tLN:200\n"
        "@SQ\tSN:txB\tLN:200\n"
        "0_1200\t0\ttxA\t10\t60\t100M\t*\t0\t0\t" + ("A" * 100) + "\t*\tNM:i:0\n"
    )
    out_bp = tmp_path / "out.bp.tsv"
    r_ok = subprocess.run([str(ec_bin), str(t2g), str(out_bp)], input=sam_in, text=True, capture_output=True, env=asan_env)
    assert r_ok.returncode == 0, r_ok.stderr
    assert out_bp.is_file()
    meta_ec = json.loads(r_ok.stderr.strip().splitlines()[-1])
    assert meta_ec["mapped_seg"] == 1
    assert meta_ec["expressed_genes"] == 1

    # Write failure path (nonexistent parent directory / /dev/full) must exit non-zero and leave no partial file
    bad_target = tmp_path / "missing_dir" / "out.bp.tsv"
    r_fail = subprocess.run([str(ec_bin), str(t2g), str(bad_target)], input=sam_in, text=True, capture_output=True, env=asan_env)
    assert r_fail.returncode != 0
    assert not bad_target.exists()
    if Path("/dev/full").exists():
        # Create a small C harness or test symlinked /dev/full rename/write error handling
        dev_full_err = subprocess.run([str(ec_bin), str(t2g), "/dev/full"], input=sam_in, text=True, capture_output=True, env=asan_env)
        assert dev_full_err.returncode != 0

    # 2. _fast_c.py embedded C source (#11, L76, L77) compiled with ASan+UBSan
    fc_tree = ast.parse((PANEXPR_DIR / "_fast_c.py").read_text(encoding="utf-8"))
    fc_c_src = None
    for node in fc_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_C_SOURCE" for t in node.targets):
            fc_c_src = ast.literal_eval(node.value)
            break
    assert isinstance(fc_c_src, str)
    fc_c_file = tmp_path / "panexpr_fast_asan.c"
    fc_c_file.write_text(fc_c_src, encoding="utf-8")

    # Standalone C driver linking _fast_c.c under ASan+UBSan so no LD_PRELOAD tricks are needed
    rad_valid = tmp_path / "orphan.rad"
    rad_valid.write_bytes(
        _build_synthetic_rad(
            ["tx1", SENTINEL_ID],
            [[
                (3, [(0, 2)]),  # MAPPED_SECOND_ORPHAN + mate2 forward (ori=2) -> fragment RC (fl=1, dUTP sense!)
                (1, [(0, 2)]),  # MAPPED_PAIR + R1 forward (ori=2) -> fragment forward (fl=0, dUTP antisense!)
            ]],
        )
    )
    rad_trunc = tmp_path / "trunc.rad"
    rad_trunc.write_bytes(rad_valid.read_bytes()[:-3])

    fc_driver = tmp_path / "fc_driver.c"
    fc_driver.write_text(
        r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t n_mapped;
    int64_t n_eq;
    int64_t total_l;
    int64_t *starts;
    int32_t *lens;
    int32_t *counts;
    uint32_t *flat_packed;
} RadAggOut;

RadAggOut *panexpr_rad_aggregate(
    const char *rad_path,
    int64_t body_offset,
    int64_t expected_chunks,
    int32_t rw,
    int32_t ori_off,
    int32_t astride,
    int32_t fmt_read_off,
    int32_t fmt_read_width,
    int32_t fmt_aln_off,
    int32_t fmt_aln_width,
    int32_t nref,
    int32_t sentinel_idx,
    const int32_t *ref_rank,
    double *out_uniq,
    int32_t *out_err
);
void panexpr_rad_free(RadAggOut *out);

int64_t panexpr_clip_adapters(
    const uint8_t *seq1,
    const int64_t *off1,
    const uint8_t *seq2,
    const int64_t *off2,
    int64_t n_reads,
    int32_t min_len,
    int32_t n_threads,
    int32_t *out_cut1,
    int32_t *out_cut2
);

int main(int argc, char **argv) {
    assert(argc == 12);
    const char *rad_ok = argv[1];
    const char *rad_bad = argv[2];
    int64_t body_offset = atoll(argv[3]);
    int64_t nchunks = atoll(argv[4]);
    int32_t rw = atoi(argv[5]);
    int32_t ori_off = atoi(argv[6]);
    int32_t astride = atoi(argv[7]);
    int32_t fmt_read_off = atoi(argv[8]);
    int32_t fmt_read_width = atoi(argv[9]);
    int32_t fmt_aln_off = atoi(argv[10]);
    int32_t fmt_aln_width = atoi(argv[11]);

    int32_t ref_rank[2] = {0, 1};
    double uniq[4] = {0};
    int32_t err = 0;

    RadAggOut *res = panexpr_rad_aggregate(
        rad_ok, body_offset, nchunks, rw, ori_off, astride,
        fmt_read_off, fmt_read_width, fmt_aln_off, fmt_aln_width,
        2, 1, ref_rank, uniq, &err
    );
    assert(res != NULL && err == 0);
    assert(res->n_mapped == 2);
    assert(uniq[0] == 1.0 && uniq[1] == 1.0);
    panexpr_rad_free(res);

    // Truncated RAD chunk must return NULL and set err == 2 (L76)
    memset(uniq, 0, sizeof(uniq));
    err = 0;
    RadAggOut *res_bad = panexpr_rad_aggregate(
        rad_bad, body_offset, nchunks, rw, ori_off, astride,
        fmt_read_off, fmt_read_width, fmt_aln_off, fmt_aln_width,
        2, 1, ref_rank, uniq, &err
    );
    assert(res_bad == NULL && err == 2);

    // Multi-threaded adapter clip (>= 8192 reads so pthread_create runs) under ASan/UBSan (L77)
    const char *line = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTAGATCGGAAGAGCACACGTCTGAAC\n";
    int64_t L_line = (int64_t)strlen(line);
    int64_t N = 8200;
    uint8_t *seq_buf = (uint8_t *)malloc((size_t)(N * L_line));
    int64_t *offs = (int64_t *)malloc((size_t)(N + 1) * sizeof(int64_t));
    int32_t *c1 = (int32_t *)calloc((size_t)N, sizeof(int32_t));
    assert(seq_buf && offs && c1);
    for (int64_t i = 0; i < N; ++i) {
        memcpy(seq_buf + i * L_line, line, (size_t)L_line);
        offs[i] = i * L_line;
    }
    offs[N] = N * L_line;
    int64_t n_clipped = panexpr_clip_adapters(seq_buf, offs, NULL, NULL, N, 31, 2, c1, NULL);
    assert(n_clipped == N);
    assert(c1[0] == 36 && c1[N - 1] == 36);

    free(seq_buf);
    free(offs);
    free(c1);
    return 0;
}
""",
        encoding="utf-8",
    )
    (
        _names_r,
        body_offset_r,
        nchunks_r,
        rw_r,
        ori_off_r,
        astride_r,
        fmt_ro_r,
        fmt_rw_r,
        fmt_ao_r,
        fmt_aw_r,
    ) = _fast_c.parse_rad_header(rad_valid)
    fc_bin = tmp_path / "fc_driver_asan"
    subprocess.run(
        [cc, *san_flags, str(fc_c_file), str(fc_driver), "-o", str(fc_bin), "-lm", "-lpthread"],
        check=True,
    )
    subprocess.run(
        [
            str(fc_bin),
            str(rad_valid),
            str(rad_trunc),
            str(body_offset_r),
            str(nchunks_r),
            str(rw_r),
            str(ori_off_r),
            str(astride_r),
            str(fmt_ro_r),
            str(fmt_rw_r),
            str(fmt_ao_r),
            str(fmt_aw_r),
        ],
        check=True,
        env=asan_env,
    )

    # 3. island_pc_core.c (L78): compile under ASan+UBSan and verify 30% unanchored reads do NOT trigger high_div
    ipc_src = PANEXPR_DIR / "island_pc_core.c"
    ipc_driver = tmp_path / "ipc_driver.c"
    ipc_driver.write_text(
        r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ref_index_s ref_index_t;
typedef struct {
    double kernel_ms;
    uint64_t n_reads_processed;
    uint64_t total_kmer_windows;
    uint64_t total_ocean_kmers;
    uint64_t total_novel_kmers;
    uint64_t unique_ocean_kmers;
    double ocean_fraction;
    uint64_t syncmer_probe_total;
    uint64_t syncmer_probe_novel;
    double r_pc_syncmer_raw;
    double r_pc_syncmer_bio;
    double r_pc_syncmer_ci_low;
    double r_pc_syncmer_ci_high;
    double r_pc_dense_raw;
    double r_pc_dense_bio;
    uint64_t c_sub_iso;
    uint64_t c_del_iso;
    uint64_t c_ins_iso;
    uint64_t c_ins_1bp_iso;
    uint64_t c_ins_multibp_iso;
    uint64_t c_del_microhom_iso;
    uint64_t c_merged_uniq;
    uint64_t c_merged_sub_only;
    uint64_t c_merged_pos_delta;
    uint64_t c_merged_neg_delta;
    int64_t sum_del_len_iso;
    int64_t sum_ins_len_iso;
    int64_t sum_merged_pos_delta;
    int64_t sum_merged_neg_delta;
    uint64_t c_rep_w_le_30;
    uint64_t c_rep_w_eq_31;
    uint64_t c_rep_w_ge_32;
    uint64_t n_end_censored_islands;
    uint64_t n_lowq_skipped_islands;
    double mean_base_err_admitted;
    double p_s_hat;
    double p_d_hat;
    double p_i_hat;
    double r_total_hat;
    double kr_hat;
    int high_divergence_merged_islands;
    int used_high_div_fallback;
    uint64_t n_anchored_reads;
    uint64_t n_unanchored_reads;
    uint64_t anchored_kmer_windows;
    uint64_t anchored_ocean_kmers;
    uint64_t anchored_novel_kmers;
    double unanchored_read_fraction;
} probe_result_t;

ref_index_t *ipc_build_index(const char **seqs, const int64_t *lens, int n_contigs, int compute_dense_d1);
void ipc_free_index(ref_index_t *idx);
int ipc_classify_reads(
    const ref_index_t *idx, const char *seqs, const char *quals,
    const int32_t *offsets, const int32_t *lengths, int32_t n_reads,
    int32_t min_base_q, double max_syncmer_err_prob, probe_result_t *out
);

static uint64_t xorshift(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

int main(void) {
    const char bases[4] = {'A', 'C', 'G', 'T'};
    int64_t ref_len = 24000;
    char *ref = (char *)malloc((size_t)ref_len + 1);
    uint64_t st = 0x123456789ABCDEF1ULL;
    for (int64_t i = 0; i < ref_len; i++) ref[i] = bases[xorshift(&st) & 3ULL];
    ref[ref_len] = '\0';

    const char *c_arr[1] = {ref};
    int64_t l_arr[1] = {ref_len};
    ref_index_t *idx = ipc_build_index(c_arr, l_arr, 1, 0);
    assert(idx != NULL);

    // 700 clean/low-SNP on-target 150-bp reads + 300 unanchored plasmid reads (30% off-target!)
    const int32_t n_reads = 1000, L = 150;
    char *reads = (char *)malloc((size_t)(n_reads * L));
    char *quals = (char *)malloc((size_t)(n_reads * L));
    memset(quals, 'I', (size_t)(n_reads * L));
    for (int r = 0; r < 700; r++) {
        int64_t pos = (int64_t)((xorshift(&st) % (uint64_t)(ref_len - L - 10)));
        memcpy(reads + r * L, ref + pos, (size_t)L);
        if ((r % 10) == 0) {
            // 1 interior SNP on every 10th read (~0.067% divergence)
            char orig = reads[r * L + 75];
            reads[r * L + 75] = (orig == 'A') ? 'C' : 'A';
        }
    }
    for (int r = 700; r < 1000; r++) {
        for (int j = 0; j < L; j++) reads[r * L + j] = bases[xorshift(&st) & 3ULL];
    }

    int32_t fixed_len[1] = {L};
    probe_result_t out;
    int rc = ipc_classify_reads(idx, reads, quals, NULL, fixed_len, n_reads, 20, 0.05, &out);
    assert(rc == 0);
    assert(out.n_anchored_reads == 700);
    assert(out.n_unanchored_reads == 300);
    assert(out.unanchored_read_fraction > 0.29 && out.unanchored_read_fraction < 0.31);
    assert(out.high_divergence_merged_islands == 0);
    assert(out.kr_hat < 0.15);

    ipc_free_index(idx);
    free(reads);
    free(quals);
    free(ref);
    return 0;
}
""",
        encoding="utf-8",
    )
    ipc_bin = tmp_path / "ipc_driver_asan"
    subprocess.run([cc, *san_flags, str(ipc_src), str(ipc_driver), "-o", str(ipc_bin), "-lm"], check=True)
    subprocess.run([str(ipc_bin)], check=True, env=asan_env)


def test_manifest_run_ids_history_matching_and_dash_sanitization(tmp_path: Path) -> None:
    """Verify #1, #3, and L94 across StudyManifest, resolve_expression_manifest, and expressionloop.xml."""
    f1 = tmp_path / "ctrl_1.fastq.gz"
    f2 = tmp_path / "trt_1.fastq.gz"
    for fp in (f1, f2):
        with gzip.open(fp, "wb") as fh:
            fh.write(b"@r1\nACGTACGTACGTACGTACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n")

    # #3: Missing ctrl_2 must raise ValueError instead of aliasing ctrl_1 via mate-suffix stripping
    sheet_missing = tmp_path / "missing_ctrl2.tsv"
    sheet_missing.write_text("sample_id\tcondition\nctrl_1\tcontrol\nctrl_2\tcontrol\ntrt_1\ttreated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ctrl_2"):
        resolve_expression_manifest(sheet_missing, reads=[str(f1), str(f2)], read_ids=["ctrl_1", "trt_1"])

    # #3: Two manifest rows claiming the exact same History dataset must raise ValueError
    sheet_dup_claim = tmp_path / "dup_claim.tsv"
    sheet_dup_claim.write_text(
        "sample_id\tcondition\trun_or_url\ns_a\tcontrol\tctrl_1\ns_b\tcontrol\tctrl_1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple manifest rows"):
        resolve_expression_manifest(sheet_dup_claim, reads=[str(f1), str(f2)], read_ids=["ctrl_1", "trt_1"])

    # #1: Explicit run_id preserved in History mode + auto-disambiguated multi-lane sample_id
    lane1 = tmp_path / "lane1.fastq.gz"
    lane2 = tmp_path / "lane2.fastq.gz"
    shutil.copyfile(f1, lane1)
    shutil.copyfile(f2, lane2)
    sheet_lanes = tmp_path / "lanes.tsv"
    sheet_lanes.write_text(
        "sample_id\tcondition\trun_id\trun_or_url\n"
        "bio_s1\tcontrol\t\tlane1\n"
        "bio_s1\tcontrol\t\tlane2\n"
        "bio_s2\ttreated\texplicit_run_99\ttrt_1\n",
        encoding="utf-8",
    )
    study, _mdf, _ = resolve_expression_manifest(
        sheet_lanes,
        reads=[str(lane1), str(lane2), str(f2)],
        read_ids=["lane1", "lane2", "trt_1"],
    )
    assert [r.run_id for r in study.runs] == ["bio_s1_L1", "bio_s1_L2", "explicit_run_99"]
    assert [r.sample for r in study.runs] == ["bio_s1", "bio_s1", "bio_s2"]

    # #1: Duplicate run_id rejected in StudyManifest
    r_dup = Run("dup1", "s1", "control", "SE", "RNA-Seq", 10, (FastqFile(str(f1), f1.stat().st_size, local_path=str(f1)),))
    with pytest.raises(ValueError, match="Duplicate run_id"):
        StudyManifest(study_id="x", accession="y", runs=[r_dup, r_dup])

    # L94: Leading '-' in read_ids and sample_id sanitized and matched; length mismatch raises
    sheet_dash = tmp_path / "dash.tsv"
    sheet_dash.write_text("sample_id\tcondition\n-ctrl1\tcontrol\n-trt1\ttreated\n", encoding="utf-8")
    study_dash, _, _ = resolve_expression_manifest(
        sheet_dash,
        reads=[str(f1), str(f2)],
        read_ids=["-ctrl1", "_trt1"],
    )
    assert [r.sample for r in study_dash.runs] == ["_ctrl1", "_trt1"]
    assert all(not r.run_id.startswith("-") for r in study_dash.runs)
    with pytest.raises(ValueError, match="Length mismatch"):
        resolve_expression_manifest(sheet_dash, reads=[str(f1), str(f2)], read_ids=["only_one"])

    # L94 + L1 XML validator checks
    root = ET.parse(EXPR_XML).getroot()
    design_param = root.find(".//param[@argument='--design']")
    assert design_param is not None
    design_re = re.compile((design_param.find("validator[@type='regex']").text or "").strip())
    assert design_re.match("~condition")
    assert design_re.match("~ batch + C(plate_id) + age + condition")
    assert not design_re.match("~ batch * condition")
    assert not design_re.match("~ batch:condition")

    pl_param = root.find(".//param[@argument='--primary-label']")
    assert pl_param is not None
    pl_re = re.compile((pl_param.find("validator[@type='regex']").text or "").strip())
    assert pl_re.match("primary_1")
    assert not pl_re.match("-bad_flag")


def test_streamer_cache_keys_gzip_corruption_mates_magic_and_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify #2, #4, #16, L8, L9, and L28 in streamer.py."""
    # #2: Distinct cache filenames when two runs share identical URL basenames ('reads.fastq.gz')
    run_a = Run(
        "SRR_A", "sA", "control", "SE", "RNA-Seq", 100,
        (FastqFile("https://mirror1.example.org/a/reads.fastq.gz", 50000),),
    )
    run_b = Run(
        "SRR_B", "sB", "treated", "SE", "RNA-Seq", 100,
        (FastqFile("https://mirror2.example.org/b/reads.fastq.gz", 50000),),
    )
    st_a = open_run_stream(run_a, "http_range", tmp_path / "cache")
    st_b = open_run_stream(run_b, "http_range", tmp_path / "cache")
    cp_a = st_a.streams[0].source.cache_file
    cp_b = st_b.streams[0].source.cache_file
    assert cp_a is not None and cp_b is not None
    assert cp_a != cp_b
    assert "SRR_A_0_" in cp_a.name and "SRR_B_0_" in cp_b.name

    # #4: GzipStreamDecoder raises on corrupt deflate stream and on truncated final member at EOF
    valid_gz = gzip.compress(b"@r1\nACGTACGT\n+\nIIIIIIII\n")
    corrupt_gz = valid_gz[:12] + b"\xff\x00\xff\x00\xff" + valid_gz[-8:]
    with pytest.raises(RuntimeError, match="Corrupt gzip"):
        GzipStreamDecoder(expect_gzip=True).feed_named(corrupt_gz, eof=True)
    with pytest.raises(RuntimeError, match="Truncated gzip"):
        GzipStreamDecoder(expect_gzip=True).feed_named(valid_gz[:-4], eof=True)

    # #4: Malformed skipped counter exposed on RunStream
    mixed_fq = tmp_path / "mixed.fastq"
    mixed_fq.write_bytes(b"stray_header_noise\n@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nACGT\n+\nII\n")
    r_mixed = Run("rmix", "smix", "control", "SE", "RNA-Seq", 2, (FastqFile(str(mixed_fq), mixed_fq.stat().st_size, local_path=str(mixed_fq)),))
    rs_mix = open_run_stream(r_mixed, "local_bytes", tmp_path)
    got_r1, _ = rs_mix.advance_bytes(4096)
    assert len(got_r1) == 1
    assert rs_mix.malformed_skipped >= 2

    # #16: RunStream.skip_bytes advances in bounded chunks without retaining records
    big_fq = tmp_path / "big.fastq"
    big_fq.write_bytes(b"@r\n" + b"ACGT" * 20 + b"\n+\n" + b"IIII" * 20 + b"\n" * 1)
    payload = b"".join(f"@read{i}\n{'ACGT' * 20}\n+\n{'IIII' * 20}\n".encode() for i in range(500))
    big_fq.write_bytes(payload)
    r_big = Run("rbig", "sbig", "control", "SE", "RNA-Seq", 500, (FastqFile(str(big_fq), len(payload), local_path=str(big_fq)),))
    rs_big = open_run_stream(r_big, "local_bytes", tmp_path)
    skipped = rs_big.skip_bytes(len(payload) // 2, chunk_bytes=4096)
    assert skipped >= len(payload) // 2
    tail_r1, _ = rs_big.advance_bytes(len(payload))
    assert 0 < len(tail_r1) < 500

    # L8: Immediate RuntimeError on mismatched PE mate headers; _1/_2 suffixes pair cleanly
    m1_bad = tmp_path / "m1_bad.fq"
    m2_bad = tmp_path / "m2_bad.fq"
    m1_bad.write_bytes(b"@alpha_read_1\nACGTACGT\n+\nIIIIIIII\n")
    m2_bad.write_bytes(b"@omega_read_9\nTGCATGCA\n+\nIIIIIIII\n")
    rs_bad_pe = RunStream(
        Run("pe_bad", "s", "c", "PE", "RNA-Seq", 1, (FastqFile(str(m1_bad), m1_bad.stat().st_size, local_path=str(m1_bad)), FastqFile(str(m2_bad), m2_bad.stat().st_size, local_path=str(m2_bad)))),
        [FastqPrefixStream(LocalFileSource(m1_bad)), FastqPrefixStream(LocalFileSource(m2_bad))],
    )
    with pytest.raises(RuntimeError, match="mate headers do not match"):
        rs_bad_pe.advance_bytes(1024)

    m1_ok = tmp_path / "m1_ok.fq"
    m2_ok = tmp_path / "m2_ok.fq"
    m1_ok.write_bytes(b"@SRR1.42_1\nACGTACGT\n+\nIIIIIIII\n")
    m2_ok.write_bytes(b"@SRR1.42_2\nTGCATGCA\n+\nIIIIIIII\n")
    rs_ok_pe = RunStream(
        Run("pe_ok", "s", "c", "PE", "RNA-Seq", 1, (FastqFile(str(m1_ok), m1_ok.stat().st_size, local_path=str(m1_ok)), FastqFile(str(m2_ok), m2_ok.stat().st_size, local_path=str(m2_ok)))),
        [FastqPrefixStream(LocalFileSource(m1_ok)), FastqPrefixStream(LocalFileSource(m2_ok))],
    )
    p1_ok, p2_ok = rs_ok_pe.advance_bytes(1024)
    assert len(p1_ok) == 1 and p2_ok is not None and len(p2_ok) == 1

    # L9: Unsupported compression / binary .sra magic bytes rejected immediately
    for label, magic in (("bzip2", b"BZh91AY&SY"), ("zstd", b"\x28\xb5\x2f\xfd\x00\x00"), ("xz", b"\xfd7zXZ\x00\x00"), ("NCBI .sra", b"NCBI.sra\x00\x00")):
        bad_f = tmp_path / f"bad_{label.split()[0]}.dat"
        bad_f.write_bytes(magic + b"padding_bytes")
        with pytest.raises(RuntimeError, match=label.split()[0]):
            FastqPrefixStream(LocalFileSource(bad_f)).advance_by_named(64)

    odp_run = Run("SRR123", "s", "c", "SE", "RNA-Seq", 100, (FastqFile("https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR123/SRR123", 99999),))
    with pytest.raises(RuntimeError, match="binary NCBI .sra"):
        open_run_stream(odp_run, "http_range", tmp_path)

    # L28: HTTP 200 ignoring Range on partial request fails fast without retrying
    calls = 0

    class _Fake200Resp:
        status_code = 200
        headers = {"Content-Length": "1000000"}

        def iter_content(self, chunk_size: int = 65536):
            yield b"A" * 1024

        def close(self) -> None:
            pass

    class _Fake200Sess:
        def get(self, url, headers=None, timeout=None, stream=False):
            nonlocal calls
            calls += 1
            return _Fake200Resp()

    src_200 = HttpRangeSource("https://example.org/test.fastq.gz", 1_000_000, tmp_path / "c200.part")
    monkeypatch.setattr(HttpRangeSource, "_get_session", classmethod(lambda cls, host: _Fake200Sess()))
    with pytest.raises(RuntimeError, match="HTTP 200"):
        src_200.read_range(0, 4096)
    assert calls == 1  # No wasteful retries on non-Range server!


def test_multilane_zoom_depth_scale_strandedness_and_cap(tmp_path: Path) -> None:
    """Verify #5, #6, #66, L4, and L5 in loop.py, zoom.py, and solver.py."""
    cat = build_catalog(
        [AssemblySpec("primary", TEST_DATA / "ref_primary.fa", TEST_DATA / "ref_primary.gtf")],
        "eukaryotic",
        "primary",
        tmp_path / "cat",
    )
    f_l1 = TEST_DATA / "ctrl1.fastq.gz"
    f_l2 = TEST_DATA / "ctrl2.fastq.gz"
    f_t1 = TEST_DATA / "trt1.fastq.gz"
    f_t2 = TEST_DATA / "trt2.fastq.gz"
    sz = f_l1.stat().st_size

    runs = [
        Run("S1_L1", "S1", "control", "SE", "RNA-Seq", 500, (FastqFile(str(f_l1), sz, local_path=str(f_l1)),)),
        Run("S1_L2", "S1", "control", "SE", "RNA-Seq", 500, (FastqFile(str(f_l2), sz, local_path=str(f_l2)),)),
        Run("S2_L1", "S2", "control", "SE", "RNA-Seq", 500, (FastqFile(str(f_l2), sz, local_path=str(f_l2)),)),
        Run("T1_L1", "T1", "treated", "SE", "RNA-Seq", 500, (FastqFile(str(f_t1), f_t1.stat().st_size, local_path=str(f_t1)),)),
        Run("T2_L1", "T2", "treated", "SE", "RNA-Seq", 500, (FastqFile(str(f_t2), f_t2.stat().st_size, local_path=str(f_t2)),)),
    ]
    manifest = StudyManifest("m", "a", runs)
    kern = MappyKernel(threads=1)
    fna = cat.write_fasta(tmp_path / "targets.fna")
    kern.ensure_index(fna, tmp_path / "idx")

    cfg = LoopConfig(
        contrast=["condition", "treated", "control"],
        max_bytes_pct=55.0,
        zoom=True,
        zoom_extra_fraction=0.10,
        abundant_sieve=True,
        abundant_sieve_force=True,
        min_bytes_per_file=512,
    )
    loop = Focaloop(cat, manifest, kern, cfg, tmp_path / "pan_state", log=lambda m: None)
    try:
        # #5: Exact depth_scale when S1_L1 had 2000 B global + 1000 B zoom and S1_L2 had 0 B global + 1000 B zoom
        loop.accs["S1_L1"].bytes_consumed = 2000
        loop.accs["S1_L2"].bytes_consumed = 0
        zres = ZoomResult(
            groups=[],
            bytes_scanned=2000,
            reads_scanned=20,
            reads_kept=5,
            reads_mapped=5,
            per_run_scale={"S1_L1": 2000 / 3000, "S1_L2": 0.0},
            per_run_zoom_bytes={"S1_L1": 1000, "S1_L2": 1000},
        )
        ds = loop._sample_depth_scale(["S1", "S2"], zres)
        assert ds["S1"] == pytest.approx(2000.0 / 4000.0)  # Exact 0.50, not 0.667!

        # #6: Canonical lane strandedness propagates across all lanes of (sample, condition, layout)
        first_gene_tid = cat.gene_targets[0].target_id
        loop.accs["S1_L1"].unique_counts[first_gene_tid] = [50.0, 0.0]  # strongly reverse-stranded (dUTP)
        loop.accs["S1_L2"].unique_counts.clear()  # secondary lane unread in global pass
        loop.accs["S2_L1"].unique_counts[first_gene_tid] = [40.0, 0.0]
        loop.accs["T1_L1"].unique_counts[first_gene_tid] = [45.0, 0.0]
        loop.accs["T2_L1"].unique_counts[first_gene_tid] = [45.0, 0.0]
        _de, _tbl, extra = loop.solve(1, two_way=True)
        assert extra["strand"]["S1_L2"] == pytest.approx(extra["strand"]["S1_L1"])
        assert extra["strand"]["S1_L2"] > 0.90
    finally:
        loop.close_streams()

    # L5 + #66: Fresh loop run respects strict max_bytes_pct across global + zoom and writes zoom_context.json
    loop2 = Focaloop(cat, manifest, kern, cfg, tmp_path / "pan_state2", log=lambda m: None)
    rep2 = loop2.run()
    assert rep2.pct_archive <= cfg.max_bytes_pct + 1e-6
    if rep2.zoom_ran:
        zctx_p = tmp_path / "pan_state2" / "zoom_context.json"
        assert zctx_p.is_file()
        zctx = json.loads(zctx_p.read_text(encoding="utf-8"))
        assert "zoomed_groups" in zctx and "sample_depth_scale" in zctx

    # L4: refit_genes_with_depth_offsets updates non-zoomed status when BH crosses 0.05 and excludes low_depth from BH M
    samples = ["s1", "s2", "s3", "t1", "t2", "t3"]
    meta_df = pd.DataFrame({"condition": ["control"] * 3 + ["treated"] * 3}, index=samples)
    g_cols = ["G_ZOOM", "G_FLIP", "G_NULL", "G_LOW"]
    cnt_glob = pd.DataFrame(
        [[30, 30, 30, 1], [32, 28, 31, 1], [29, 31, 29, 0], [35, 45, 30, 1], [34, 46, 31, 0], [36, 44, 29, 1]],
        index=samples, columns=g_cols, dtype=float,
    )
    cnt_zoom = cnt_glob.copy()
    cnt_zoom["G_ZOOM"] = [20.0, 22.0, 19.0, 140.0, 150.0, 145.0]
    de_in = pd.DataFrame({
        "iteration": [1, 1, 1, 1],
        "group_id": g_cols,
        "gene_id": g_cols,
        "symbol": g_cols,
        "dna_present_prob": [1.0, 1.0, 1.0, 1.0],
        "streamed_reads": [196.0, 224.0, 180.0, 4.0],
        "base_mean": [32.0, 37.0, 30.0, 0.6],
        "log2fc": [0.2, 0.6, 0.0, 0.1],
        "lfc_se_deseq2": [0.2, 0.2, 0.2, 0.9],
        "var_placement_ambiguity": [0.0, 0.0, 0.0, 0.0],
        "lfc_se_total": [0.2, 0.2, 0.2, 0.9],
        "ci95_low": [-0.2, 0.2, -0.4, -1.7],
        "ci95_high": [0.6, 1.0, 0.4, 1.9],
        "pvalue": [0.30, 0.035, 0.80, 0.90],
        "padj": [0.45, 0.048, 0.80, 0.90],  # G_FLIP starts < 0.05 ("DE") -> with M=3 BH, rank 2 gets 0.035 * 3 / 2 = 0.0525 >= 0.05 -> "non_DE"!
        "tech_var_fraction": [0.1, 0.1, 0.1, 0.9],
        "demand_score": [0.2, 0.2, 0.0, 0.0],
        "status": ["non_DE", "DE", "non_DE", "low_depth"],
    })
    refit_df = refit_genes_with_depth_offsets(
        de_in, cnt_glob, meta_df, ["condition", "treated", "control"], ["G_ZOOM"],
        cnt_zoom, pd.Series(0.5, index=samples),
        disp=pd.Series(0.05, index=g_cols), sf=pd.Series(1.0, index=samples),
    ).set_index("group_id")
    assert refit_df.loc["G_ZOOM", "status"] == "DE_after_zoom"
    assert refit_df.loc["G_FLIP", "padj"] == pytest.approx(0.0525)  # M=3 (G_LOW excluded from BH!)
    assert refit_df.loc["G_FLIP", "status"] == "non_DE"  # Relabeled from DE -> non_DE!
    assert refit_df.loc["G_LOW", "status"] == "low_depth"


def test_kernels_stderr_zero_mapped_rad_orphan_and_sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify #7, #11, #15, L18, L19, L20, L23, and L76 in kernels.py."""
    # L23: MappyKernel uses ProcessPoolExecutor, not multiprocessing.Pool
    assert "ProcessPoolExecutor" in inspect.getsource(MappyKernel.map_batch)
    assert ".Pool(" not in inspect.getsource(MappyKernel.map_batch)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    def _make_exe(name: str, body: str) -> Path:
        p = fake_bin / name
        p.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        p.chmod(0o755)
        return p

    # #7: SamKernel checks non-zero exit code and surfaces stderr
    _make_exe("fake_aligner", 'import sys\nif "--version" in sys.argv:\n    sys.exit(0)\nsys.stderr.write("BOOM_ALIGNER_OOM\\n")\nsys.exit(137)\n')
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ.get('PATH', '')}")
    import focaloop.panexpr.kernels as kmod
    monkeypatch.setattr(kmod, "_TOOL_PROBE_CACHE", {})

    sk = SamKernel("fake", "fake", "fake_aligner", 1.0, ["{idx}"], ["{idx}"], "fasta", threads=1)
    sk.index_paths["idx"] = tmp_path / "dummy.fa"
    with pytest.raises(RuntimeError, match="BOOM_ALIGNER_OOM"):
        sk.map_batch([("ACGTACGTACGTACGTACGTACGTACGTACGT", "I" * 32)], None)

    # L18: SalmonKernel passes --minAssignedFrags 1 and returns 0-mapped MapResult cleanly on 0 assigned fragments
    salmon_args_log = tmp_path / "salmon_args.json"
    _make_exe(
        "salmon",
        f"""import json, pathlib, sys
args = sys.argv[1:]
if "--version" in args:
    sys.exit(0)
pathlib.Path({str(salmon_args_log)!r}).write_text(json.dumps(args))
if args[0] == "index":
    idx = pathlib.Path(args[args.index("-i") + 1])
    idx.mkdir(parents=True, exist_ok=True)
    (idx / "versionInfo.json").write_text("{{}}")
    sys.exit(0)
out = pathlib.Path(args[args.index("-o") + 1]) / "aux_info"
out.mkdir(parents=True, exist_ok=True)
(out / "meta_info.json").write_text('{{"num_mapped": 0, "num_assigned_fragments": 0}}')
sys.stderr.write("insufficient assigned fragments (0 fragments were mapped)\\n")
sys.exit(1)
""",
    )
    kmod._TOOL_PROBE_CACHE.clear()
    sal_k = SalmonKernel(threads=1, tmp_dir=tmp_path)
    fna_dummy = tmp_path / "t.fa"
    fna_dummy.write_text(">t1\n" + "ACGT" * 30 + "\n", encoding="utf-8")
    sal_k.ensure_index(fna_dummy, tmp_path / "sal_idx")
    sal_res = sal_k.map_batch([("ACGT" * 15, "I" * 60)], None)
    assert sal_res.n_reads == 1 and sal_res.n_mapped == 0
    logged_args = json.loads(salmon_args_log.read_text())
    assert "--minAssignedFrags" in logged_args

    # #15 & L19 & #7: KallistoKernel creates missing idx_dir, handles n_pseudoaligned==0, and checks samtools view rc
    _make_exe(
        "kallisto",
        """import json, pathlib, sys
args = sys.argv[1:]
if "--version" in args:
    sys.exit(0)
if args[0] == "index":
    idx = pathlib.Path(args[args.index("-i") + 1])
    idx.write_bytes(b"KIDX")
    sys.exit(0)
out = pathlib.Path(args[args.index("-o") + 1])
out.mkdir(parents=True, exist_ok=True)
mode = pathlib.Path(args[-1]).read_text()
if "TTTTTTTT" in mode:
    (out / "run_info.json").write_text('{"n_pseudoaligned": 0}')
    sys.stderr.write("Error: zero reads pseudoaligned\\n")
    sys.exit(1)
(out / "run_info.json").write_text('{"n_pseudoaligned": 1}')
(out / "pseudoalignments.bam").write_bytes(b"CORRUPT_BAM")
sys.exit(0)
""",
    )
    _make_exe("samtools", 'import sys\nif "--version" in sys.argv:\n    sys.exit(0)\nsys.stderr.write("CORRUPT_BAM_EOF\\n")\nsys.exit(1)\n')
    kmod._TOOL_PROBE_CACHE.clear()
    kal_k = KallistoKernel(threads=1, tmp_dir=tmp_path)
    nested_missing_idx = tmp_path / "deep" / "missing" / "kal_idx_dir"
    kal_k.ensure_index(fna_dummy, nested_missing_idx)
    assert (nested_missing_idx / "kallisto.idx").is_file()
    kal_zero = kal_k.map_batch([("T" * 40, "I" * 40)], None)
    assert kal_zero.n_reads == 1 and kal_zero.n_mapped == 0
    with pytest.raises(RuntimeError, match="CORRUPT_BAM_EOF"):
        kal_k.map_batch([("ACGT" * 10, "I" * 40)], None)

    # #11 & L76: read_rad_bulk matches c_rad_aggregate on frag_map_type == 3 and raises on truncated RAD
    rad_p = tmp_path / "test_fmt3.rad"
    rad_p.write_bytes(_build_synthetic_rad(["tx_sense", SENTINEL_ID], [[(3, [(0, 2)]), (1, [(0, 2)])]]))
    names, recs = read_rad_bulk(rad_p)
    res_py = MapResult()
    from focaloop.panexpr.kernels import resolve_pair
    for alns in recs:
        resolve_pair(PiscemKernel._cands(alns, names), None, res_py)
    c_agg = _fast_c.c_rad_aggregate(rad_p, SENTINEL_ID, {})
    assert c_agg is not None
    assert c_agg[1]["tx_sense"] == res_py.unique_counts["tx_sense"] == [1.0, 1.0]

    rad_bad = tmp_path / "bad_chunks.rad"
    rad_bad.write_bytes(_build_synthetic_rad(["tx_sense", SENTINEL_ID], [[(1, [(0, 1)])]], expected_chunks_override=3))
    with pytest.raises(RuntimeError, match="Truncated RAD"):
        read_rad_bulk(rad_bad)
    with pytest.raises(RuntimeError, match="(?i)corrupt or truncated.*RAD"):
        _fast_c.c_rad_aggregate(rad_bad, SENTINEL_ID, {})

    # L20: PiscemKernel trace uses >=100-bp sentinel even when first read is 25 bp (< k=31), and checks sentinel count
    trace_fq_lens: list[int] = []

    def _fake_piscem_run_cmd(self, td: Path, r1, r2, threads: int, tag: str) -> Path:
        if tag == "trace":
            trace_fq_lens.extend(len(s) for s, _ in r1)
            # Emit 1 sentinel record for a 2-read batch if _BAD_SENTINEL is set, else all sentinels
            n_sent = (len(r1) // 2 - 1) if getattr(self, "_bad_sentinel_test", False) else (len(r1) // 2)
            recs_c = [(1, [(1, 1)]) for _ in range(n_sent)]
        else:
            recs_c = [(1, [(0, 1)])]
        rp = td / f"{tag}.rad"
        rp.write_bytes(_build_synthetic_rad(["tx1", SENTINEL_ID], [recs_c]))
        return rp

    monkeypatch.setattr(PiscemKernel, "_run_cmd", _fake_piscem_run_cmd)
    pk = PiscemKernel(threads=1, tmp_dir=tmp_path, k=31, clip_adapters=False)
    pk.index_paths["idx"] = tmp_path / "pidx"
    short_r1 = [("ACGT" * 6 + "A", "I" * 25), ("TGCA" * 10, "I" * 40)]  # First read is 25 bp (< k=31)!
    tr_ok = pk.map_batch(short_r1, None, trace=True)
    assert tr_ok.per_read is not None
    assert trace_fq_lens[1] >= 100  # Sentinel length stayed 100 bp, NOT min(100, 25)!

    pk._bad_sentinel_test = True  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="sentinel count mismatch"):
        pk.map_batch(short_r1, None, trace=True)


def test_xsra_odp_progress_retry_reset_and_premature_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify #8 in XsraRunStream: progress resets retries, and failure/premature EOF raises instead of exhausted=True."""
    import focaloop.panexpr.sra_odp as odp_mod

    fake_xsra = tmp_path / "xsra"
    state_f = tmp_path / "xsra_calls.txt"
    fake_xsra.write_text(
        f"""#!/usr/bin/env python3
import pathlib, sys
sf = pathlib.Path({str(state_f)!r})
n = int(sf.read_text()) if sf.exists() else 0
sf.write_text(str(n + 1))
if n < 3:
    # Emit cumulative spots 1..(n+1)*10 then fail transiently -> retry counter MUST reset on progress!
    for sid in range(1, (n + 1) * 10 + 1):
        sys.stdout.write(f"@SRR777.{{sid}}.1\\nACGTACGTACGTACGT\\n+\\nIIIIIIIIIIIIIIII\\n")
    sys.stdout.flush()
    sys.stderr.write(f"transient s3 reset #{{n}}\\n")
    sys.exit(1)
# Call 4+: stall with 0 new spots and exit 1
sys.stderr.write("fatal s3 403 block\\n")
sys.exit(1)
""",
        encoding="utf-8",
    )
    fake_xsra.chmod(0o755)
    monkeypatch.setattr(odp_mod, "xsra_path", lambda: str(fake_xsra))

    run = Run("SRR777", "s1", "control", "SE", "RNA-Seq", 100, (FastqFile("https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR777/SRR777", 10000),))
    plan = SpotBlockPlan("SRR777", 100, block_spots=50_000, holdout_fraction=0.0)
    st = XsraRunStream(run, plan, max_retries=2, retry_backoff_s=0.0)

    # First 3 invocations advance to spot 30 (surviving > max_retries=2 transient drops because progress resets retries!)
    r1_a, _ = st.advance_bytes(2500)
    assert len(r1_a) >= 25
    assert st.total_retries >= 2
    # Subsequent stall without progress exhausts retries and MUST raise RuntimeError with exhausted == False!
    with pytest.raises(RuntimeError, match="fatal s3 403 block"):
        st.advance_bytes(5000)
    assert st.exhausted is False
    assert st.failed is True


def test_logan_zero_mapping_fallback_ram_gating_and_no_psi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify #9, #10, #14, #17, #19, L15, L16, and L17 in logan_expr.py & runners_expression.py."""
    # L16: minimap2 fallback retains secondary alignments for paralog/ortholog EM
    mm_cmd = _aligner_cmd("minimap2", "/usr/bin/minimap2", Path("t.fna"), 4)
    assert "--secondary=yes" in mm_cmd and "-p" in mm_cmd and "0.95" in mm_cmd

    # #17: RAM-aware aligner selection skips bwa-mem2 (~25 B/bp + 256 MiB) when cgroup RAM is 450 MiB on 15 Mb catalog
    if shutil.which("bwa") and shutil.which("bwa-mem2"):
        chosen_name, _ = _select_unitig_aligner(targets_bp=15_000_000, ram_budget_bytes=450 << 20)
        assert chosen_name == "bwa"
    with pytest.raises(LoganZeroMappingError, match="Insufficient available RAM"):
        _select_unitig_aligner(targets_bp=100_000_000, ram_budget_bytes=64 << 20)

    # #14: parse_ena_filereport populates _ena_meta from base_count
    ena_tsv = (
        "run_accession\tsample_alias\tsample_title\tlibrary_layout\tlibrary_strategy\t"
        "read_count\tbase_count\tfastq_bytes\tfastq_ftp\tfastq_md5\n"
        "SRR111\ts1\ts1\tPAIRED\tRNA-Seq\t1000\t300000\t50000;50000\t"
        "ftp.sra.ebi.ac.uk/a_1.fastq.gz;ftp.sra.ebi.ac.uk/a_2.fastq.gz\tm1;m2\n"
    )
    m_ena = parse_ena_filereport(ena_tsv, "st", "acc")
    assert m_ena.runs[0].extra["_ena_meta"]["read_len"] == pytest.approx(150.0)
    assert m_ena.runs[0].extra["_ena_meta"]["kmers_per_frag"] == pytest.approx(240.0)

    # #9 & #10 & #19: Zero-mapping Logan unitigs raise LoganZeroMappingError; valid run writes gene-group TSV without PSI
    cat = build_catalog(
        [AssemblySpec("primary", TEST_DATA / "ref_primary.fa", TEST_DATA / "ref_primary.gtf")],
        "eukaryotic", "primary", tmp_path / "cat",
    )
    unmapped_u = tmp_path / "unmapped.unitigs.fa"
    unmapped_u.write_text(">0 ka:f:10.0\n" + ("AAACCCGGGTTT" * 6) + "\n", encoding="utf-8")
    runs_bad = [
        Run("SRR5001", "s1", "control", "SE", "RNA-Seq", 1000, (FastqFile(str(unmapped_u), 100, local_path=str(unmapped_u)),), extra={"_ena_meta": {"kmers_per_frag": 70.0, "fastq_bytes": 10000}}),
        Run("SRR5002", "s2", "treated", "SE", "RNA-Seq", 1000, (FastqFile(str(unmapped_u), 100, local_path=str(unmapped_u)),), extra={"_ena_meta": {"kmers_per_frag": 70.0, "fastq_bytes": 10000}}),
    ]
    meta_bad = pd.DataFrame([{"run_id": "SRR5001", "sample": "s1", "condition": "control"}, {"run_id": "SRR5002", "sample": "s2", "condition": "treated"}])
    with pytest.raises(LoganZeroMappingError, match="0 mapped segments"):
        run_logan_expression_cohort(
            catalog=cat,
            manifest=StudyManifest("s", "a", runs_bad),
            sample_meta_df=meta_bad,
            contrast_list=["condition", "treated", "control"],
            design="~condition",
            gtf_path=TEST_DATA / "ref_primary.gtf",
            work_dir=tmp_path / "w_bad",
            pan_state_dir=tmp_path / "ps_bad",
            threads=1,
            local_unitig_map={"SRR5001": unmapped_u, "SRR5002": unmapped_u},
        )

    # Valid cohort: #10 no fabricated PSI + #14 zero ENA network calls when _ena_meta is present
    import focaloop.panexpr.logan_expr as le_mod
    monkeypatch.setattr(le_mod, "fetch_ena_run_lengths", lambda accs: (_ for _ in ()).throw(AssertionError("Should reuse _ena_meta!")))
    gene_tgts = cat.gene_targets
    runs_ok = []
    meta_ok_rows = []
    local_u_ok = {}
    for cond, boost in (("control", 1.0), ("treated", 8.0)):
        for rep_i in (1, 2, 3):
            rid = f"SRR60{cond[0].upper()}{rep_i}"
            uf = tmp_path / f"{rid}.fa"
            uf.write_text(
                "".join(
                    f">{gi} ka:f:{(50.0 * boost if gi == 0 else 20.0 + rep_i):.1f}\n{t.seq[:180]}\n"
                    for gi, t in enumerate(gene_tgts) if len(t.seq) >= 60
                ),
                encoding="utf-8",
            )
            local_u_ok[rid] = uf
            runs_ok.append(
                Run(
                    rid, rid, cond, "SE", "RNA-Seq", 5000,
                    (FastqFile(str(uf), uf.stat().st_size, local_path=str(uf)),),
                    extra={"_ena_meta": {"kmers_per_frag": 70.0, "fastq_bytes": 250000, "layout": "SE"}},
                )
            )
            meta_ok_rows.append({"run_id": rid, "sample": rid, "condition": cond})

    gene_ab_tsv = tmp_path / "logan_gene_ab.tsv"
    ps_ok = tmp_path / "ps_ok"
    res_ok = run_logan_expression_cohort(
        catalog=cat,
        manifest=StudyManifest("s_ok", "a_ok", runs_ok),
        sample_meta_df=pd.DataFrame(meta_ok_rows),
        contrast_list=["condition", "treated", "control"],
        design="~condition",
        gtf_path=TEST_DATA / "ref_primary.gtf",
        work_dir=tmp_path / "w_ok",
        pan_state_dir=ps_ok,
        threads=2,
        local_unitig_map=local_u_ok,
        out_gene_abundances_tsv=gene_ab_tsv,
    )
    assert res_ok["kpf_fallback_runs"] == []
    assert not (ps_ok / "transcript_abundances.parquet").exists()
    ab_df = pd.read_csv(gene_ab_tsv, sep="\t")
    assert "gene_tpm" in ab_df.columns
    assert "isoform_fraction_psi" not in ab_df.columns
    assert pq.read_table(ps_ok / "target_counts.parquet").num_rows > 0


def test_catalog_gff3_multiparent_prokka_ensembl_and_biotypes(tmp_path: Path) -> None:
    """Verify #12, #13, L6, L7, and L27 across catalog.py, regime.py, logan_expr.py, and runners_expression.py."""
    rng = np.random.default_rng(19)
    ctg_seq = "".join(rng.choice(list("ACGT"), size=4000))
    fa = tmp_path / "ref.fa"
    fa.write_text(f">chr1 desc\n{ctg_seq[:2000]}\n{ctg_seq[2000:]}\n", encoding="utf-8")
    loaded = load_fasta(fa)
    assert loaded["chr1"] == ctg_seq

    # #12 + L6: GFF3 with comma-separated Parent=tx1,tx2 + GENCODE-style gene_type/transcript_type + opposite-strand antisense
    gff_multi = tmp_path / "multi.gff3"
    gff_multi.write_text(
        "##gff-version 3\n"
        "chr1\tsrc\tgene\t100\t900\t.\t+\t.\tID=g1;gene_type=protein_coding\n"
        "chr1\tsrc\tmRNA\t100\t900\t.\t+\t.\tID=tx1;Parent=g1\n"
        "chr1\tsrc\tmRNA\t100\t900\t.\t+\t.\tID=tx2;Parent=g1\n"
        "chr1\tsrc\texon\t100\t250\t.\t+\t.\tID=ex1;Parent=tx1,tx2\n"
        "chr1\tsrc\texon\t350\t500\t.\t+\t.\tID=ex2;Parent=tx1\n"
        "chr1\tsrc\texon\t600\t800\t.\t+\t.\tID=ex3;Parent=tx1,tx2\n"
        "chr1\tsrc\tgene\t120\t300\t.\t-\t.\tID=g_anti;gene_type=protein_coding\n"
        "chr1\tsrc\tmRNA\t120\t300\t.\t-\t.\tID=tx_anti;Parent=g_anti\n"
        "chr1\tsrc\texon\t120\t300\t.\t-\t.\tID=ex_anti;Parent=tx_anti\n"
        "chr1\tsrc\tgene\t1200\t1600\t.\t+\t.\tID=g_rrna;gene_type=Mt_rRNA\n"
        "chr1\tsrc\trRNA\t1200\t1600\t.\t+\t.\tID=tx_rrna;Parent=g_rrna;transcript_type=Mt_rRNA\n"
        "chr1\tsrc\texon\t1200\t1600\t.\t+\t.\tID=ex_rrna;Parent=tx_rrna\n",
        encoding="utf-8",
    )
    txs = {t.transcript_id: t for t in build_spliced_transcripts(AssemblySpec("p", fa, gff_multi))}
    assert len(txs["tx1"].exons) == 3
    assert len(txs["tx2"].exons) == 2  # Shared exons ex1 and ex3 attached to tx2!
    assert txs["tx_rrna"].biotype == "Mt_rRNA"

    af = annotation_features(gff_multi)
    assert af.n_transcripts == 4
    assert _detect_organism_mode(gff_multi, "auto") == "eukaryotic"
    as_genes = detect_antisense_genes(gff_multi)
    assert {"g1", "g_anti"}.issubset(as_genes)

    cat_euk = build_catalog([AssemblySpec("p", fa, gff_multi)], "eukaryotic", "p", tmp_path / "c_euk")
    for t in cat_euk.gene_targets:
        t.exons = []
    assert restore_exons_from_annotation(cat_euk, gff_multi) >= 3
    # Mt_rRNA must be excluded from DE gene_level_tables (L6)
    acc_dummy = {"r1": RunAccumulator(Run("r1", "s1", "c", "SE", "RNA-Seq", 10, (FastqFile(str(fa), 100, local_path=str(fa)),)))}
    acc_dummy["r1"].unique_counts["p|tx1"] = [10.0, 0.0]
    acc_dummy["r1"].unique_counts["p|tx_rrna"] = [500.0, 0.0]
    tc_tbl, _ = quantify_cohort(acc_dummy, cat_euk, {}, {}, {"r1": 0.5})
    gl = gene_level_tables(tc_tbl, cat_euk)
    assert "OG_g_rrna" not in gl["counts"].columns

    # #13: Prokka CDS-only + tRNA GFF3 (with product= on every feature and ZERO gene features!)
    gff_prokka = tmp_path / "prokka.gff3"
    gff_prokka.write_text(
        "##gff-version 3\n"
        "chr1\tProkka\tCDS\t50\t450\t.\t+\t0\tID=PROKKA_00001;locus_tag=PROKKA_00001;gene=dnaA;product=Replication initiator DnaA\n"
        "chr1\tProkka\ttRNA\t600\t680\t.\t+\t.\tID=PROKKA_00002;locus_tag=PROKKA_00002;product=tRNA-Ala(tgc)\n",
        encoding="utf-8",
    )
    prok_tgts = {t.gene_id: t for t in build_prokaryotic_targets(AssemblySpec("p", fa, gff_prokka))}
    assert set(prok_tgts) == {"PROKKA_00001", "PROKKA_00002"}
    assert prok_tgts["PROKKA_00001"].product == "Replication initiator DnaA"
    assert prok_tgts["PROKKA_00002"].biotype == "tRNA"

    # L7: Ensembl Bacteria gene -> mRNA -> CDS hierarchy must NOT create a duplicate CDS:<prot> target
    gff_ens = tmp_path / "ensembl_bact.gff3"
    gff_ens.write_text(
        "##gff-version 3\n"
        "chr1\tEnsembl\tgene\t100\t700\t.\t+\t.\tID=gene:b0001;Name=thrL;biotype=protein_coding\n"
        "chr1\tEnsembl\tmRNA\t100\t700\t.\t+\t.\tID=transcript:AAC73112;Parent=gene:b0001\n"
        "chr1\tEnsembl\tCDS\t100\t700\t.\t+\t0\tID=CDS:AAC73112;Parent=transcript:AAC73112;product=thr leader\n",
        encoding="utf-8",
    )
    ens_tgts = build_prokaryotic_targets(AssemblySpec("p", fa, gff_ens))
    assert [t.gene_id for t in ens_tgts] == ["b0001"]
    assert ens_tgts[0].product == "thr leader"

    # #13: Contig mismatch yielding 0 primary targets raises ValueError clearly
    gff_wrong_ctg = tmp_path / "wrong.gff3"
    gff_wrong_ctg.write_text("##gff-version 3\nchr_other\tProkka\tCDS\t50\t450\t.\t+\t0\tID=P1;product=X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0 non-decoy targets"):
        build_catalog([AssemblySpec("p", fa, gff_wrong_ctg)], "prokaryotic", "p", tmp_path / "c_empty")


def test_regime_cgroup_ancestors_spliced_probe_and_tournament_guards(tmp_path: Path) -> None:
    """Verify L21, L22, L24, L25, and L26 in regime.py and tournament.py."""
    # L26: Ancestor cgroup v2 memory.max & memory.high + GALAXY_MEMORY_MB + SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK
    cg_root = tmp_path / "cgroup"
    leaf = cg_root / "system.slice" / "slurmstepd.scope" / "job_42" / "task_0"
    leaf.mkdir(parents=True)
    (cg_root / "system.slice" / "slurmstepd.scope" / "job_42" / "memory.max").write_text(str(6 << 30))
    (cg_root / "system.slice" / "slurmstepd.scope" / "job_42" / "memory.high").write_text(str(4 << 30))
    (leaf / "memory.max").write_text("max")
    proc_cg = tmp_path / "proc_cgroup"
    proc_cg.write_text("0::/system.slice/slurmstepd.scope/job_42/task_0\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {(64 << 20)} kB\n")  # 64 GiB host RAM

    assert available_ram_bytes(_cgroup_root=cg_root, _proc_cgroup=proc_cg, _meminfo=meminfo, _env={}) == (4 << 30)
    assert available_ram_bytes(
        _cgroup_root=cg_root, _proc_cgroup=proc_cg, _meminfo=meminfo,
        _env={"GALAXY_MEMORY_MB": "2048"},
    ) == (2048 << 20)
    assert available_ram_bytes(
        _cgroup_root=cg_root, _proc_cgroup=proc_cg, _meminfo=meminfo,
        _env={"SLURM_MEM_PER_CPU": "512", "SLURM_CPUS_PER_TASK": "3"},
    ) == (1536 << 20)

    # L24 & L25: Spliced genome with 50-bp introns (<=64 bp!) probed against spliced cDNA -> 0 false deletions!
    rng = np.random.default_rng(101)
    genome_seq = "".join(rng.choice(list("ACGT"), size=12000))
    fa = tmp_path / "compact_euk.fa"
    fa.write_text(f">chr1\n{genome_seq}\n", encoding="utf-8")
    gtf_lines = []
    spliced_reads = []
    for gi in range(6):
        base = 500 + gi * 1500
        # Two 200-bp exons separated by a 50-bp intron (<= 64 bp!)
        ex1_s, ex1_e = base, base + 199
        ex2_s, ex2_e = base + 250, base + 449
        gtf_lines.append(f'chr1\tsrc\texon\t{ex1_s}\t{ex1_e}\t.\t+\t.\tgene_id "G{gi}"; transcript_id "T{gi}";\n')
        gtf_lines.append(f'chr1\tsrc\texon\t{ex2_s}\t{ex2_e}\t.\t+\t.\tgene_id "G{gi}"; transcript_id "T{gi}";\n')
        cdna = genome_seq[ex1_s - 1 : ex1_e] + genome_seq[ex2_s - 1 : ex2_e]
        # Junction-spanning 150-bp reads right across the 50-bp intron!
        for off in range(110, 140, 2):
            spliced_reads.append(cdna[off : off + 150])
    gtf = tmp_path / "compact_euk.gtf"
    gtf.write_text("".join(gtf_lines), encoding="utf-8")

    ipc_dir = tmp_path / "ipc_tmp"
    dec = classify(
        [AssemblyInput("primary", fa, gtf)],
        spliced_reads * 10,
        kernels=["piscem", "salmon", "mappy"],
        use_island_pc_router=True,
        ipc_index_dir=ipc_dir,
        persist_ipc_cache=False,
    )
    assert dec.spliced_regime is True
    assert dec.router_verdict is not None
    assert dec.router_verdict["status"] == "ok"
    assert dec.router_verdict["index_source"] == "built_spliced_transcripts"
    assert dec.router_verdict["c_del_iso"] == 0
    assert dec.router_verdict["pseudoaligners_disqualified"] is False
    assert "piscem" in dec.candidates and "salmon" in dec.candidates
    assert not ipc_dir.exists()  # L25: zero unreferenced .ipc.idx left in temp dir!

    # L25: RAM budget guard skips in-process index build cleanly when budget is tight
    dec_low_ram = classify(
        [AssemblyInput("primary", fa, gtf)],
        spliced_reads[:50],
        kernels=["mappy"],
        ram_budget_bytes=64 << 20,  # 64 MiB -> 0.35 * 64 = 22.4 MiB < 32 MiB floor
        use_island_pc_router=True,
    )
    assert dec_low_ram.router_verdict is not None
    # Note: max_probe_ram has a 32 MiB floor; verify large cDNA triggers skipped_ram_budget
    from focaloop.panexpr.regime import _run_island_pc_router, AnnotationFeatures, GenomeFeatures
    skip_v = _run_island_pc_router(
        [AssemblyInput("primary", fa, gtf)],
        [GenomeFeatures(str(fa), 50_000_000, 1, "scan")],
        spliced_reads[:20],
        None,
        annots=[AnnotationFeatures(str(gtf), 100, False, 100, 1000, 2000, 0.8, 2.0, 1.0, 2.0, 3.0, 20_000_000, 5_000_000, 25_000_000)],
        spliced=True,
        strict=False,
        ram_budget_bytes=512 << 20,
    )
    assert skip_v["status"] == "skipped_ram_budget"

    # L22: Deterministic speed prior independent of wall-clock jitter
    from focaloop.panexpr.tournament import evaluate_kernel
    cat = build_catalog([AssemblySpec("p", fa, gtf)], "eukaryotic", "p", tmp_path / "cat_t")

    class _MockJitterKernel:
        name = "bwa_spa"
        label = "bwa"
        trace_costs_extra_pass = True

        def __init__(self, wall_s: float) -> None:
            self._wall_s = wall_s

        def map_batch(self, r1_b, r2_b, calibrator=None, trace=False):
            return MapResult(
                n_reads=len(r1_b),
                n_mapped=len(r1_b),
                wall_seconds=self._wall_s,
                per_read={i: ((cat.gene_targets[0].target_id,), (True,)) for i in range(len(r1_b))},
            )

    probe_r1 = [(s, "I" * len(s)) for s in spliced_reads[:20]]
    ks_fast = evaluate_kernel(_MockJitterKernel(0.001), cat, probe_r1, None, [], False)  # type: ignore[arg-type]
    ks_slow = evaluate_kernel(_MockJitterKernel(2.500), cat, probe_r1, None, [], False)  # type: ignore[arg-type]
    assert ks_fast.reads_per_sec > ks_slow.reads_per_sec * 100
    assert ks_fast.composite_raw == pytest.approx(ks_slow.composite_raw)
    assert ks_fast.optimality_score == pytest.approx(ks_slow.optimality_score)

    # L21: run_tournament raises ValueError listing exclusion reasons when all kernels are excluded
    dec_excl = classify([AssemblyInput("p", fa, gtf)], spliced_reads[:20], kernels=["mappy"])
    dec_excl.excluded["mappy"] = "synthetic test exclusion"
    k_mappy = MappyKernel(threads=1)
    with pytest.raises(ValueError, match="synthetic test exclusion"):
        run_tournament([k_mappy], cat, probe_r1, None, "SE", regime=dec_excl, log=lambda m: None)


def test_design_validation_numeric_covariates_holdout_zoom_contract_and_cache_perms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify #61, #16, #66, #18, L1, L3, L9, L10, L11, L12, L13, L14, and L98."""
    # L1: Interaction formulas rejected cleanly
    for bad_d in ("~ batch * condition", "~ batch:condition", "~ a + b:c"):
        with pytest.raises(ValueError, match="Interaction terms"):
            validate_additive_design(bad_d)

    # L98: Native cache 0700 permissions & owner/mode verification
    cdir, _ = ensure_user_native_cache_dir(explicit_cache_root=tmp_path / "ncache", subdir_stem="test_native")
    assert (os.lstat(cdir).st_mode & 0o777) == 0o700
    f_so = cdir / "dummy.so"
    f_so.write_bytes(b"\x7fELF")
    os.chmod(f_so, 0o700)
    assert verify_native_artifact(f_so, require_current_uid=True) is True
    os.chmod(f_so, 0o777)
    assert verify_native_artifact(f_so, require_current_uid=True) is False
    sym_so = cdir / "sym.so"
    os.chmod(f_so, 0o700)
    sym_so.symlink_to(f_so)
    assert verify_native_artifact(sym_so, require_current_uid=True) is False

    # L11: _head_content_length checks HTTP status and never returns Content-Length: 1 on HTTP 206 without Content-Range
    monkeypatch.setenv("FOCALOOP_HEAD_RETRY_BASE_S", "0")

    class _Fake404:
        status_code = 404
        headers: dict[str, str] = {}

        def raise_for_status(self):
            raise RuntimeError("HTTP 404 Not Found")

        def close(self):
            pass

    class _Fake206NoCR:
        status_code = 206
        headers = {"Content-Length": "1"}

        def raise_for_status(self):
            pass

        def close(self):
            pass

    import focaloop.runners_expression as re_mod
    monkeypatch.setattr(re_mod, "_ipv4_session", lambda: type("S", (), {"head": lambda *a, **k: _Fake404(), "get": lambda *a, **k: _Fake404()})())
    with pytest.raises(RuntimeError, match="404"):
        _head_content_length("https://example.org/missing.fq.gz", retries=1)

    monkeypatch.setattr(re_mod, "_ipv4_session", lambda: type("S", (), {"head": lambda *a, **k: type("H", (), {"status_code": 405, "headers": {}})(), "get": lambda *a, **k: _Fake206NoCR()})())
    with pytest.raises(RuntimeError, match="Content-Range"):
        _head_content_length("https://example.org/broken206.fq.gz", retries=1)

    # L14: Custom URL named SRR1234567.fastq.gz must NOT be treated as an INSDC accession for Logan!
    monkeypatch.setattr(re_mod, "_head_content_length", lambda u, timeout=45.0: 500_000)
    url_sheet = tmp_path / "url_srr.tsv"
    url_sheet.write_text(
        "sample_id\tcondition\trun_or_url\n"
        "SRR1234567\tcontrol\thttps://custom-bucket.example.org/SRR1234567.fastq.gz\n"
        "SRR1234568\tcontrol\thttps://custom-bucket.example.org/SRR1234568.fastq.gz\n"
        "SRR1234569\ttreated\thttps://custom-bucket.example.org/SRR1234569.fastq.gz\n"
        "SRR1234570\ttreated\thttps://custom-bucket.example.org/SRR1234570.fastq.gz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Logan-only mode requires public SRA/ENA run accessions"):
        run_expressionloop_galaxy(
            manifest_path=url_sheet,
            ref_fasta=TEST_DATA / "ref_primary.fa",
            ref_gtf=TEST_DATA / "ref_primary.gtf",
            contrast="condition:treated:control",
            logan_mode="logan_only",
            out_de=tmp_path / "de_u.tsv",
            out_report=tmp_path / "rep_u.html",
            state_dir=tmp_path / "st_u",
        )

    # L12 + L3: Blank covariate cell and unmodeled followup factor raise ValueError up front
    reads_6 = [str(TEST_DATA / f"{s}.fastq.gz") for s in ("ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3")]
    rids_6 = ["ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3"]
    blank_sheet = tmp_path / "blank_cov.tsv"
    blank_sheet.write_text(
        "\ufeffsample_id\tcondition\tbatch\n"
        "ctrl1\tcontrol\tB1\nctrl2\tcontrol\t\nctrl3\tcontrol\tB2\n"
        "trt1\ttreated\tB1\ntrt2\ttreated\tB2\ntrt3\ttreated\tB2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty/missing values"):
        run_expressionloop_galaxy(
            manifest_path=blank_sheet, ref_fasta=TEST_DATA / "ref_primary.fa", ref_gtf=TEST_DATA / "ref_primary.gtf",
            reads=reads_6, read_ids=rids_6, contrast="condition:treated:control", design="~batch + condition",
            out_de=tmp_path / "d1.tsv", out_report=tmp_path / "r1.html", state_dir=tmp_path / "s1",
        )
    ok_sheet = tmp_path / "ok_cov.tsv"
    ok_sheet.write_text(
        "\ufeffsample_id\tgroup\tage\tbatch\n"
        "ctrl1\tcontrol\t22.5\tB1\nctrl2\tcontrol\t35.0\tB1\nctrl3\tcontrol\t49.2\tB2\n"
        "trt1\ttreated\t24.0\tB1\ntrt2\ttreated\t38.5\tB2\ntrt3\ttreated\t52.1\tB2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Follow-up contrast factor 'batch' must appear in design"):
        run_expressionloop_galaxy(
            manifest_path=ok_sheet, ref_fasta=TEST_DATA / "ref_primary.fa", ref_gtf=TEST_DATA / "ref_primary.gtf",
            reads=reads_6, read_ids=rids_6, contrast="group:treated:control", design="~group",
            followup_contrasts="batch:B2:B1",
            out_de=tmp_path / "d2.tsv", out_report=tmp_path / "r2.html", state_dir=tmp_path / "s2",
        )

    # #61 + #16 + #66 + L10 + L12 (UTF-8 BOM) + L13 (continuous numeric age covariate + group column):
    # Create >360 KB uncompressed FASTQs so max_byte_fraction=0.45 leaves >131 KB unstreamed for genuine holdout!
    big_reads = []
    for sid in rids_6:
        raw_bytes = gzip.decompress((TEST_DATA / f"{sid}.fastq.gz").read_bytes())
        reps = (380_000 // max(1, len(raw_bytes))) + 2
        lines = raw_bytes.splitlines()
        out_chunks = []
        idx_read = 0
        for _ in range(reps):
            for j in range(0, len(lines), 4):
                if j + 3 < len(lines):
                    idx_read += 1
                    out_chunks.append(
                        f"@{sid}_r{idx_read}\n".encode()
                        + lines[j + 1] + b"\n+\n" + lines[j + 3] + b"\n"
                    )
        p_big = tmp_path / f"{sid}.fastq"
        p_big.write_bytes(b"".join(out_chunks))
        big_reads.append(str(p_big))

    st_hold = tmp_path / "state_holdout"
    meta_hold = run_expressionloop_galaxy(
        manifest_path=ok_sheet,
        ref_fasta=TEST_DATA / "ref_primary.fa",
        ref_gtf=TEST_DATA / "ref_primary.gtf",
        reads=big_reads,
        read_ids=rids_6,
        contrast="group:treated:control",
        design="~ age + group",
        organism_mode="eukaryotic",
        source_mode="local_bytes",
        max_byte_fraction=0.45,
        enable_zoom=True,
        enable_abundant_filter=True,
        filter_mode="include",
        threads=2,
        kernel_names=["mappy"],
        n_tournament_reads=500,
        state_dir=st_hold,
        out_de=tmp_path / "de_hold.tsv",
        out_report=tmp_path / "rep_hold.html",
    )
    # #61: Genuine holdout saved & recorded in focaloop_state.json, AND ledger NOT pre-seeded with q0001!
    assert meta_hold["holdout_saved"] is True
    assert meta_hold["loop_summary"]["holdout_saved"] is True
    assert (st_hold / "panexpr_state" / "target_counts_holdout.parquet").is_file()

    store = FocaloopStateStore(st_hold)
    ledger_tbl = store.read("hypothesis_ledger")
    assert ledger_tbl.num_rows == 0
    ledger = HypothesisLedger.from_store(store, alpha_budget=0.05)
    assert ledger.wealth == pytest.approx(0.05)

    # L13: 'age' persisted as float64 in sample_metadata.parquet
    saved_meta = pd.read_parquet(st_hold / "sample_metadata.parquet")
    assert pd.api.types.is_float_dtype(saved_meta["age"])
    assert "group" in saved_meta.columns

    # L9: Zero mapped reads in ANY sample (even outside primary contrast levels!) raises RuntimeError
    unmapped_fq = tmp_path / "all_polyA.fastq"
    unmapped_fq.write_bytes(b"@u1\n" + b"A" * 80 + b"\n+\n" + b"I" * 80 + b"\n" * 1)
    three_grp_sheet = tmp_path / "three_grp.tsv"
    three_grp_sheet.write_text(
        "sample_id\tcondition\n"
        "ctrl1\tcontrol\nctrl2\tcontrol\nctrl3\tcontrol\n"
        "trt1\ttreated\ntrt2\ttreated\ntrt3\ttreated\n"
        "other1\tthird_group\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Zero mapped reads.*other1"):
        run_expressionloop_galaxy(
            manifest_path=three_grp_sheet,
            ref_fasta=TEST_DATA / "ref_primary.fa",
            ref_gtf=TEST_DATA / "ref_primary.gtf",
            reads=reads_6 + [str(unmapped_fq)],
            read_ids=rids_6 + ["other1"],
            contrast="condition:treated:control",
            design="~condition",
            source_mode="local_bytes",
            max_byte_fraction=1.0,
            kernel_names=["mappy"],
            n_tournament_reads=300,
            state_dir=tmp_path / "st_zero",
            out_de=tmp_path / "de_zero.tsv",
            out_report=tmp_path / "rep_zero.html",
        )
