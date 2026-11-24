"""Comprehensive regression suite for all GeneLoop 0.4.1 review findings (#45-#60, #73, #74, L47-L64, L87-L94, L96, L97)."""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import textwrap
import xml.etree.ElementTree as ET

import pytest

from focaloop.geneloop import engine as eng_mod
from focaloop.geneloop import homology_fetch as hf_mod
from focaloop.geneloop import locus_zoom as lz_mod
from focaloop.geneloop import ncrna as ncrna_mod
from focaloop.geneloop import orf_scout as orf_mod
from focaloop.geneloop import seed as seed_mod
from focaloop.geneloop import sieve_syncmer as ss_mod
from focaloop.geneloop import stream as stream_mod
from focaloop.geneloop import synthesis as synth_mod
from focaloop.geneloop import telemetry as tele_mod
from focaloop.geneloop.cli import build_parser as build_gl_cli_parser
from focaloop.geneloop.evidence import BinStore, JunctionStore
from focaloop.geneloop.gffio import Transcript, read_models, write_gff3, write_gtf
from focaloop.geneloop.intron_judge import GoodTuringUCBJudge, WaveJunction
from focaloop.geneloop.sequtil import Genome, revcomp, translate
from focaloop.geneloop.telemetry import StageLog
import focaloop.runners_geneloop as rg_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
GLCORE_C = REPO_ROOT / "focaloop" / "geneloop" / "glcore" / "glcore.c"
ORF_SCOUT_C = REPO_ROOT / "focaloop" / "geneloop" / "csrc" / "orf_scout.c"


def _san_env() -> dict[str, str]:
    env = dict(os.environ)
    env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1:abort_on_error=0"
    env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"
    return env


def test_asan_ubsan_glcore_and_orf_scout_c99_findings(tmp_path: Path) -> None:
    """Compile glcore.c and orf_scout.c with -fsanitize=address,undefined and test L87-L93 + #46 + L52."""
    cc = shutil.which("gcc") or shutil.which("cc")
    assert cc is not None, "C compiler required"

    glcore_san = tmp_path / "glcore_san"
    subprocess.run(
        [
            cc,
            "-O1",
            "-g",
            "-std=c99",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fopenmp",
            "-pthread",
            "-Wall",
            "-Wextra",
            str(GLCORE_C),
            "-o",
            str(glcore_san),
            "-lm",
        ],
        check=True,
    )

    env = _san_env()

    # Build small genome + syncmer active table under ASan+UBSan
    exon_a = "ATGCGTTCGATCAAGCTAGGCTACCTGACCGTGAACGTAATCGATCGTAGCATCGATCGTAGCTAGCATGCGACTGATGCTAGCTAGC"
    genome_fa = tmp_path / "g.fa"
    genome_fa.write_text(f">chr1\n{'A' * 100}{exon_a}{'T' * 4000}\n>chr2\n{'C' * 2000}\n")
    cand_fa = tmp_path / "cand.fa"
    cand_fa.write_text(f">L000001\n{exon_a}\n")
    mult_bin = tmp_path / "cand.mult"
    sat_txt = tmp_path / "sat.txt"
    sat_txt.write_text("L000001\n")
    act_bin = tmp_path / "active.bin"

    subprocess.run(
        [str(glcore_san), "syncmers"],
        input=f"{exon_a}\n".encode(),
        stdout=subprocess.DEVNULL,
        check=True,
        env=env,
    )
    subprocess.run(
        [str(glcore_san), "mult", "-g", str(genome_fa), "-c", str(cand_fa), "-o", str(mult_bin), "-t", "2"],
        check=True,
        env=env,
    )
    subprocess.run(
        [str(glcore_san), "activate", "-m", str(mult_bin), "-c", str(cand_fa), "-l", str(sat_txt), "-o", str(act_bin)],
        check=True,
        env=env,
    )

    # L90: glcore sieve multi-token headers, empty stdin handling, and malformed FASTQ rejection
    seq80 = "ACGTAGCTGACTGATCGATGCTAGTCGATCGATGCTAGCTAGCTGATCGATGCTAGCTAGCTGACTGATCGATGCTAGCT"
    qual80 = "I" * len(seq80)
    good_interleaved = (
        f"@SRR999.1/1 length=80 comment=ok\n{seq80}\n+\n{qual80}\n"
        f"@SRR999.1/2 length=80 comment=ok\n{seq80}\n+\n{qual80}\n"
    ).encode()
    o1 = tmp_path / "s1.fq"
    o2 = tmp_path / "s2.fq"
    st_json = tmp_path / "sieve.json"
    subprocess.run(
        [str(glcore_san), "sieve", "-1", str(o1), "-2", str(o2), "-a", str(act_bin), "--stats", str(st_json), "-t", "2"],
        input=good_interleaved,
        check=True,
        env=env,
    )
    sieve_stats = json.loads(st_json.read_text())
    assert sieve_stats["pairs_in"] == 1
    assert sieve_stats["orphans"] == 0

    # Empty stdin without --allow-empty fails; with --allow-empty succeeds
    res_empty_fail = subprocess.run(
        [str(glcore_san), "sieve", "-1", str(o1), "-2", str(o2), "-a", str(act_bin)],
        input=b"",
        env=env,
        capture_output=True,
    )
    assert res_empty_fail.returncode != 0
    subprocess.run(
        [str(glcore_san), "sieve", "-1", str(o1), "-2", str(o2), "-a", str(act_bin), "--allow-empty"],
        input=b"",
        check=True,
        env=env,
    )

    # Malformed FASTQ cases: missing '@', missing '+', qual len != seq len, truncated 3-line record, orphan-only stream
    bad_fastqs = [
        b"NOT_AT_HEADER\nACGT\n+\nIIII\n",
        b"@r1/1\nACGT\nBAD_PLUS\nIIII\n",
        b"@r1/1\nACGTACGT\n+\nIII\n",
        b"@r1/1\nACGT\n+\n",
        f"@SRR1.1/1\n{seq80}\n+\n{qual80}\n@SRR1.2/1\n{seq80}\n+\n{qual80}\n".encode(),
    ]
    for bad_fq in bad_fastqs:
        r_bad = subprocess.run(
            [str(glcore_san), "sieve", "-1", str(o1), "-2", str(o2), "-a", str(act_bin)],
            input=bad_fq,
            env=env,
            capture_output=True,
        )
        assert r_bad.returncode != 0

    # L87 + L88 + L52: glcore junc --bin <= 0 rejected, POS=0 / POS>len clamped, >128 splice junctions preserved
    for bad_bin in ("0", "-10"):
        r_bin = subprocess.run(
            [
                str(glcore_san),
                "junc",
                "--bin",
                bad_bin,
                "--junc",
                str(tmp_path / "j.tsv"),
                "--bins",
                str(tmp_path / "b.tsv"),
                "--no-out",
            ],
            input=b"@SQ\tSN:chr1\tLN:1000\n",
            env=env,
            capture_output=True,
        )
        assert r_bin.returncode != 0

    # 150 exons -> 149 introns (> 128 old static array limit!)
    cigar_149_introns = "10M20N" * 149 + "10M"
    seq_len_150 = 150 * 10
    sam_junc = (
        "@HD\tVN:1.6\tSO:unsorted\n"
        "@SQ\tSN:chr1\tLN:6000\n"
        f"r_pos0\t0\tchr1\t0\t60\t10M\t*\t0\t0\t{'A' * 10}\t{'I' * 10}\tNH:i:1\n"
        f"r_past\t0\tchr1\t999999\t60\t10M\t*\t0\t0\t{'A' * 10}\t{'I' * 10}\tNH:i:1\n"
        f"r_manyj\t0\tchr1\t10\t60\t{cigar_149_introns}\t*\t0\t0\t{'A' * seq_len_150}\t{'I' * seq_len_150}\tNH:i:1\n"
    ).encode()
    j_tsv = tmp_path / "junc.tsv"
    b_tsv = tmp_path / "bins.tsv"
    j_stats_p = tmp_path / "junc_stats.json"
    subprocess.run(
        [
            str(glcore_san),
            "junc",
            "--bin",
            "1000",
            "--junc",
            str(j_tsv),
            "--bins",
            str(b_tsv),
            "--stats",
            str(j_stats_p),
            "--no-out",
        ],
        input=sam_junc,
        check=True,
        env=env,
    )
    j_lines = [ln for ln in j_tsv.read_text().splitlines() if ln.strip()]
    assert len(j_lines) == 149
    j_stats = json.loads(j_stats_p.read_text())
    assert j_stats["mapped_records"] == 3
    assert j_stats["n_sq"] == 1

    # L91 + #46: glcore logan dynamic >128 junctions + per-run flush + run_records / run_mapped stats
    sam_logan = (
        "@HD\tVN:1.6\tSO:unsorted\n"
        "@SQ\tSN:chr1\tLN:6000\n"
        f"SRR0_0\t0\tchr1\t10\t60\t{cigar_149_introns}\t*\t0\t0\t{'A' * seq_len_150}\t*\tNH:i:1\tka:f:8.5\n"
        "SRR1_0\t4\t*\t0\t0\t*\t*\t0\t0\tACGTACGTAC\t*\tka:f:5.0\n"
        "SRR2_0\t0\tchr1\t100\t60\t25M100N25M\t*\t0\t0\t" + ("A" * 50) + "\t*\tNH:i:1\tka:f:12.0\n"
    ).encode()
    l_ka = tmp_path / "logan.ka.tsv"
    l_unw = tmp_path / "logan.unw.tsv"
    l_bins = tmp_path / "logan.bins.tsv"
    l_stats_p = tmp_path / "logan.stats.json"
    subprocess.run(
        [
            str(glcore_san),
            "logan",
            "--flush-threshold",
            "64",
            "--junc-ka",
            str(l_ka),
            "--junc-unw",
            str(l_unw),
            "--bins-ka",
            str(l_bins),
            "--stats",
            str(l_stats_p),
            "--no-out",
        ],
        input=sam_logan,
        check=True,
        env=env,
    )
    l_stats = json.loads(l_stats_p.read_text())
    assert l_stats["run_records"] == {"SRR0": 1, "SRR1": 1, "SRR2": 1}
    assert l_stats["run_mapped"] == {"SRR0": 1, "SRR1": 0, "SRR2": 1}
    assert len(l_ka.read_text().splitlines()) == 150

    # L89: fopen_checked failure exits non-zero (rc == 2)
    r_io = subprocess.run(
        [str(glcore_san), "mult", "-g", str(genome_fa), "-c", str(cand_fa), "-o", str(tmp_path / "no_such_dir" / "out.mult")],
        env=env,
        capture_output=True,
    )
    assert r_io.returncode == 2

    # L92 + L93: Compile orf_scout.c with ASan+UBSan and verify 64-bit lengths + 2 Mb chunk-boundary ORF handling
    driver_c = tmp_path / "orf_driver.c"
    driver_c.write_text(
        textwrap.dedent(
            """
            #include <assert.h>
            #include <stdint.h>
            #include <stdio.h>
            #include <stdlib.h>
            #include <string.h>

            typedef struct {
                int contig_idx;
                int64_t start;
                int64_t end;
                int strand;
                float gc;
                float dinuc_entropy;
                float soft_n_frac;
                float frac_lowcopy_syncmers;
                int n_syncmers;
            } OrfCandidate;

            int scan_orphan_orfs(
                int n_contigs, const char **contig_seqs, const int64_t *contig_lens,
                const int *n_excl_per_contig, const int64_t **excl_starts, const int64_t **excl_ends,
                int min_orf_bp, float max_soft_n_frac, float min_gc, float max_gc, float min_entropy,
                int sync_k, int sync_s, int max_sync_mult, float min_lowcopy_frac,
                int threads, OrfCandidate **out_cands, int *out_count
            );
            void free_orf_candidates(OrfCandidate *ptr);

            int main(void) {
                OrfCandidate *out = NULL;
                int cnt = 0;
                const char *tiny = "ATGAAACCCGGGTTTAAATAG";
                int64_t bad_len = -5;
                assert(scan_orphan_orfs(1, &tiny, &bad_len, NULL, NULL, NULL, 450, 0.2f, 0.2f, 0.8f, 1.0f, 31, 15, 3, 0.5f, 1, &out, &cnt) == -2);
                int64_t ok_len = (int64_t)strlen(tiny);
                assert(scan_orphan_orfs(1, &tiny, &ok_len, NULL, NULL, NULL, 0, 0.2f, 0.2f, 0.8f, 1.0f, 31, 15, 3, 0.5f, 1, &out, &cnt) == -2);

                /* Construct 4.3 Mb sequence (filled with 'N' except our spanning ORF across CHUNK_BP = 2,000,000) */
                const int64_t L = 4300000;
                char *seq = (char *)malloc((size_t)L + 1);
                assert(seq != NULL);
                memset(seq, 'N', (size_t)L);
                seq[L] = '\\0';

                /* Place a unique 600 bp (200-codon) + strand ORF from 1999700 to 2000300,
                 * with an internal in-frame ATG at 2000060 (inside the second 2,000,000 chunk!) */
                const char *codons[16] = {
                    "GCT", "CGT", "GAC", "TTC", "GGT", "CAC", "ATC", "CTG",
                    "AAG", "AAC", "CCA", "CAG", "TCC", "ACC", "GTT", "TAC"
                };
                int64_t start_pos = 1999700;
                memcpy(seq + start_pos, "ATG", 3);
                uint32_t lcg = 1234567U;
                for (int i = 1; i < 199; ++i) {
                    int64_t p = start_pos + (int64_t)i * 3;
                    lcg = lcg * 1664525U + 1013904223U;
                    if (i == 120) {
                        /* p = 2000060 > 2000000 (inside second 2 Mb chunk!) */
                        memcpy(seq + p, "ATG", 3);
                    } else {
                        memcpy(seq + p, codons[(lcg >> 16) & 15U], 3);
                    }
                }
                memcpy(seq + start_pos + 199 * 3, "TAA", 3);
                int64_t expected_end = start_pos + 200 * 3;

                int rc = scan_orphan_orfs(
                    1, (const char **)&seq, &L, NULL, NULL, NULL,
                    450, 0.25f, 0.20f, 0.80f, 2.5f, 31, 15, 3, 0.70f, 2, &out, &cnt
                );
                assert(rc == 0);
                int plus_hits = 0;
                for (int i = 0; i < cnt; ++i) {
                    if (out[i].strand == 1) {
                        plus_hits++;
                        assert(out[i].start == start_pos);
                        assert(out[i].end == expected_end);
                    }
                }
                assert(plus_hits == 1);
                free_orf_candidates(out);
                free(seq);
                return 0;
            }
            """
        )
    )
    orf_driver_bin = tmp_path / "orf_driver"
    subprocess.run(
        [
            cc,
            "-O1",
            "-g",
            "-std=c99",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fopenmp",
            str(ORF_SCOUT_C),
            str(driver_c),
            "-o",
            str(orf_driver_bin),
            "-lm",
        ],
        check=True,
    )
    subprocess.run([str(orf_driver_bin)], check=True, env=env)


def test_45_runstream_failure_detection_backoff_and_cohort_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#45: RunStream checks xsra rc + stall, backs off, marks STREAM_FAILED (never EXHAUSTED), and fails cohort if all RNA streams fail."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    xsra_script = fake_bin / "xsra"
    xsra_script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1] == 'describe':\n"
        "    print('{\"total_spots\": 1000, \"n_segments\": 2}')\n"
        "    sys.exit(0)\n"
        "# Emit only 2 spots out of 1000 and exit 1\n"
        "seq = 'A' * 60\n"
        "qual = 'I' * 60\n"
        "for i in (1, 2):\n"
        "    sys.stdout.write(f'@r{i}/1\\n{seq}\\n+\\n{qual}\\n@r{i}/2\\n{seq}\\n+\\n{qual}\\n')\n"
        "sys.stdout.flush()\n"
        "sys.exit(1)\n"
    )
    xsra_script.chmod(0o755)
    monkeypatch.setenv("XSRA_BIN", str(xsra_script))
    monkeypatch.setenv("GENELOOP_XSRA_RETRY_BASE_S", "0.002")

    st = stream_mod.RunStream("SRR111111", 2, 1000, tmp_path / "logs", threads=1)
    buf = io.BytesIO()
    with pytest.raises(RuntimeError, match="xsra stream failed"):
        st.pump(500, buf)
    assert st.stream_failed is True
    assert not st.eof
    st.close()

    # Verify GateRun.arm_c marks STREAM_FAILED and raises when all RNA libraries fail
    gfa = tmp_path / "g.fa"
    gfa.write_text(">chr1\n" + "ACGT" * 500 + "\n")
    gate = {
        "gate_id": "t45",
        "fasta_path": str(gfa),
        "proteins_path": "",
        "exclude_genus": "",
        "tiberius_model_cfg": "vertebrates",
        "sra_short_runs": ["SRR111111"],
        "sra_long_runs": [],
    }
    gr = eng_mod.GateRun(
        "t45",
        root=tmp_path / "runs",
        loop=eng_mod.LoopParams(logan_mode="off", enable_structural_ncrna="off", threads=1, hisat_threads=1),
        gate=gate,
    )
    gr._skip_tiberius = True
    gr.seed()
    monkeypatch.setattr(gr, "hisat_index", lambda *a, **k: tmp_path / "dummy_idx")

    def _fail_slice(stream, *args, **kwargs):
        stream.stream_failed = True
        raise RuntimeError("xsra stream failed on SRR111111")

    monkeypatch.setattr(eng_mod, "align_slice_short", _fail_slice)
    monkeypatch.setattr(eng_mod, "align_slice_short_se", _fail_slice)
    with pytest.raises(RuntimeError, match="All RNA streaming libraries failed"):
        gr.arm_c()


def test_46_and_L49_logan_stub_rejection_cache_schema_and_zero_record_reroute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#46 + L49: Reject <=100B Logan zstd stubs, invalidate v1 probe cache, report skipped runs, reroute 0-record runs."""
    assert stream_mod.LOGAN_PROBE_SCHEMA_VERSION >= 2
    assert stream_mod.MIN_LOGAN_ARCHIVE_BYTES == 100

    # Stale v1 probe cache saying SRR100001 is valid must be ignored and re-probed
    cache_p = tmp_path / "logan_probe.json"
    cache_p.write_text(
        json.dumps(
            {
                "logan_runs": ["SRR100001"],
                "fallback_short_runs": [],
                "probes": {"SRR100001": {"available": True, "compressed_bytes": 13}},
            }
        )
    )

    def _fake_probe_url(acc: str, timeout: float = 8.0, retries: int = 2) -> dict:
        if acc == "SRR100001":
            return {"run_id": acc, "available": False, "http_status": 200, "content_length": 13, "is_stub": True, "url": "u1"}
        return {"run_id": acc, "available": True, "http_status": 200, "content_length": 4096, "is_stub": False, "url": "u2"}

    monkeypatch.setattr(stream_mod, "probe_logan_contigs_url", _fake_probe_url)
    res = stream_mod.probe_logan_runs(["SRR100001", "SRR100002"], cache_path=cache_p)
    assert res["logan_runs"] == ["SRR100002"]
    assert res["fallback_short_runs"] == ["SRR100001"]
    assert res["stub_runs"] == ["SRR100001"]

    # fetch_logan_contig_file rejects <=100-byte downloaded files
    class _StubResp:
        def __enter__(self):
            return io.BytesIO(b"\x28\xb5\x2f\xfd\x20\x00\x01\x00\x00")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(stream_mod.urllib.request, "urlopen", lambda *a, **k: _StubResp())
    with pytest.raises(RuntimeError, match="(?i)stub"):
        stream_mod.fetch_logan_contig_file("SRR100001", tmp_path / "dl", retries=1)

    # L49: partition_short_runs in logan_only mode records skipped_short_runs and raises on transient probe errors
    gfa = tmp_path / "g.fa"
    gfa.write_text(">chr1\n" + "ACGT" * 200 + "\n")
    monkeypatch.setattr(eng_mod, "probe_logan_runs", lambda runs, **kw: res)
    gr = eng_mod.GateRun(
        "t46",
        root=tmp_path / "runs",
        loop=eng_mod.LoopParams(logan_mode="logan_only"),
        gate={"gate_id": "t46", "fasta_path": str(gfa), "sra_short_runs": ["SRR100001", "SRR100002"]},
    )
    lr, fb, pinfo = gr.partition_short_runs()
    assert lr == ["SRR100002"]
    assert fb == []
    assert pinfo["skipped_short_runs"] == ["SRR100001"]

    monkeypatch.setattr(
        eng_mod,
        "probe_logan_runs",
        lambda *a, **k: {
            "logan_runs": ["SRR100002"],
            "fallback_short_runs": ["SRR100001"],
            "transient_failed_runs": ["SRR100001"],
        },
    )
    gr2 = eng_mod.GateRun(
        "t46b",
        root=tmp_path / "runs",
        loop=eng_mod.LoopParams(logan_mode="logan_only"),
        gate={"gate_id": "t46b", "fasta_path": str(gfa), "sra_short_runs": ["SRR100001", "SRR100002"]},
    )
    with pytest.raises(RuntimeError, match="failed transiently"):
        gr2.partition_short_runs()


def test_47_and_L57_manifest_bom_headerless_unknown_cols_zero_runs_and_whole_tokens(tmp_path: Path) -> None:
    """#47 + L57: BOM handling, headerless manifests, unknown column errors, 0-run errors, whole-token ONT/long matching."""
    m_bom = tmp_path / "bom.tsv"
    m_bom.write_bytes(
        "\ufeffrun_accession\tread_type\tinstrument_platform\tsample_title\tprotocol\torganism\n"
        "SRR200001\tcontrol_front_contigs_belong\tILLUMINA\tcontrol_sample\tdUTP\tDanio rerio\n"
        "SRR200002\tcdna\tOXFORD_NANOPORE\tpromethion_lib\tcDNA\tDanio rerio\n"
        "SRR200003\tdirect-rna\tOXFORD_NANOPORE\trna004\tdirect_rna\tDanio rerio\n"
        "ERR300004\tccs\tPACBIO_SMRT\tiso_lib\tIso-Seq\tDanio rerio\n".encode("utf-8")
    )
    short_r, long_r, orgs, stranded_lr = rg_mod._resolve_rna_manifest_detailed(m_bom)
    assert short_r == ["SRR200001"]
    assert long_r == ["SRR200002", "SRR200003", "ERR300004"]
    assert stranded_lr == ["SRR200003"]
    assert orgs["SRR200001"] == "Danio rerio"

    # Headerless TSV with line 1 matching _INSDC_RUN_RE
    m_hdrless = tmp_path / "headerless.tsv"
    m_hdrless.write_text("SRR400001\tshort\nDRR400002\tnanopore\n")
    sr_h, lr_h, _o = rg_mod._resolve_rna_manifest(m_hdrless)
    assert sr_h == ["SRR400001"]
    assert lr_h == ["DRR400002"]

    # Unrecognized column raises ValueError
    m_bad_col = tmp_path / "bad_col.tsv"
    m_bad_col.write_text("run_accession\treadtype_typo\nSRR500001\tshort\n")
    with pytest.raises(ValueError, match="Unrecognized column"):
        rg_mod._resolve_rna_manifest(m_bad_col)

    # Empty / 0-run manifest raises ValueError
    m_empty = tmp_path / "empty.tsv"
    m_empty.write_text("run_accession\tread_type\n")
    with pytest.raises(ValueError, match="0"):
        rg_mod._resolve_rna_manifest(m_empty)

    # --sra-accessions resolving to 0 runs raises ValueError
    gfa = tmp_path / "g.fa"
    gfa.write_text(">chr1\n" + "ACGT" * 100 + "\n")
    pfa = tmp_path / "p.faa"
    pfa.write_text(">p1\nMKTLLLTLVVVTIVCLDLGYT\n")
    with pytest.raises(ValueError, match="0 valid INSDC"):
        rg_mod.run_geneloop_galaxy(
            genome_fasta=gfa,
            proteins_fasta=pfa,
            sra_accessions="   , ;  ",
            mode="fast_evidence",
            out_gff3=tmp_path / "a.gff3",
            out_proteins=tmp_path / "a.faa",
            out_gtf=tmp_path / "a.gtf",
            out_summary=tmp_path / "a.tsv",
            out_report=tmp_path / "a.html",
            state_dir=tmp_path / "st",
        )


def test_48_skip_tiberius_raises_when_no_proteins_and_no_rna(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#48: ValueError when skip_tiberius is true (explicit fast_evidence or auto fallback) and no proteins/RNA exist."""
    monkeypatch.setattr(rg_mod, "discover_tiberius_weights", lambda *a, **k: None)
    monkeypatch.setattr(rg_mod, "detect_tiberius_runtime", lambda: {"backend": "none"})
    monkeypatch.setattr(rg_mod, "gpu_available", lambda: False)

    for mode in ("fast_evidence", "auto"):
        with pytest.raises(ValueError, match="proteins|RNA"):
            rg_mod.resolve_execution_mode(mode, "vertebrates", has_proteins=False, has_rna=False)

    gfa = tmp_path / "g.fa"
    gfa.write_text(">chr1\n" + "ACGT" * 100 + "\n")
    for mode in ("fast_evidence", "auto"):
        with pytest.raises(ValueError, match="proteins|RNA"):
            rg_mod.run_geneloop_galaxy(
                genome_fasta=gfa,
                protein_source="none",
                mode=mode,
                out_gff3=tmp_path / "a.gff3",
                out_proteins=tmp_path / "a.faa",
                out_gtf=tmp_path / "a.gtf",
                out_summary=tmp_path / "a.tsv",
                out_report=tmp_path / "a.html",
                state_dir=tmp_path / "st",
            )


def test_49_history_bam_header_validation_and_mismatched_sq_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#49: Validate History BAM @SQ names/lengths against genome.fna and merge discordant @SQ orders via samtools merge."""
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    genome_lens = {"chr1": 1000, "chr2": 2000}

    def _sam_to_bam(sam_txt: str, out_bam: Path) -> Path:
        subprocess.run(["samtools", "view", "-b", "-o", str(out_bam), "-"], input=sam_txt.encode(), check=True)
        return out_bam

    # Unknown contig rejected with ValueError
    bad_ctg_bam = _sam_to_bam(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chrUnk\tLN:1000\nr1\t0\tchrUnk\t10\t60\t20M\t*\t0\t0\t"
        + ("A" * 20)
        + "\t"
        + ("I" * 20)
        + "\n",
        tmp_path / "bad_ctg.bam",
    )
    with pytest.raises(ValueError, match="not present in target genome"):
        stream_mod.validate_bam_header_against_genome(bad_ctg_bam, genome_lens)

    # Length mismatch rejected with ValueError
    bad_len_bam = _sam_to_bam(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:999\nr1\t0\tchr1\t10\t60\t20M\t*\t0\t0\t"
        + ("A" * 20)
        + "\t"
        + ("I" * 20)
        + "\n",
        tmp_path / "bad_len.bam",
    )
    with pytest.raises(ValueError, match="does not match target genome length"):
        stream_mod.validate_bam_header_against_genome(bad_len_bam, genome_lens)

    # Two BAMs with opposite @SQ orders (chr1,chr2 vs chr2,chr1): _cat_sort must preserve contig identity!
    bam_a = _sam_to_bam(
        "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:chr2\tLN:2000\n"
        f"read_chr1\t0\tchr1\t50\t60\t30M\t*\t0\t0\t{'A' * 30}\t{'I' * 30}\tNH:i:1\n",
        tmp_path / "a.bam",
    )
    bam_b = _sam_to_bam(
        "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr2\tLN:2000\n@SQ\tSN:chr1\tLN:1000\n"
        f"read_chr2\t0\tchr2\t150\t60\t30M\t*\t0\t0\t{'C' * 30}\t{'I' * 30}\tNH:i:1\n",
        tmp_path / "b.bam",
    )
    cap_state = tmp_path / "cap.u16"
    r_a = stream_mod.ingest_bam_to_evidence(bam_a, "bA", tmp_path / "w", "ba", cap_state, (40, 120, 40), genome_lengths=genome_lens)
    r_b = stream_mod.ingest_bam_to_evidence(bam_b, "bB", tmp_path / "w", "bb", cap_state, (40, 120, 40), genome_lengths=genome_lens)
    merged_bam = tmp_path / "merged.bam"
    eng_mod._cat_sort([r_a.bam, r_b.bam], merged_bam, threads=2)
    sam_out = subprocess.run(["samtools", "view", str(merged_bam)], capture_output=True, text=True, check=True).stdout
    by_qname = {ln.split("\t")[0]: (ln.split("\t")[2], int(ln.split("\t")[3])) for ln in sam_out.splitlines() if ln.strip()}
    assert by_qname["read_chr1"] == ("chr1", 50)
    assert by_qname["read_chr2"] == ("chr2", 150)


def test_50_L50_L51_local_fastq_stream_reopen_offset_spot_ids_and_validation(tmp_path: Path) -> None:
    """#50 + L50 + L51: Reopening LocalFastqStream skips scout records; spot IDs stay unique; malformed FASTQs fail."""
    r1 = tmp_path / "r1.fq.gz"
    r2 = tmp_path / "r2.fq.gz"
    with gzip.open(r1, "wt") as f1, gzip.open(r2, "wt") as f2:
        for i in range(1, 6):
            seq = ("ACGT" * 15) + ("A" * i)
            qual = "I" * len(seq)
            f1.write(f"@orig_{i}/1\n{seq}\n+\n{qual}\n")
            f2.write(f"@orig_{i}/2\n{seq}\n+\n{qual}\n")

    st = stream_mod.LocalFastqStream("libA", r1, r2)
    assert st.total_spots == 5

    # Wave-0 scout consumes 2 pairs, then GateRun.close() closes handles before CPU Tiberius
    b1 = io.BytesIO()
    assert st.pump(2, b1) == 2
    txt1 = b1.getvalue().decode()
    assert "@libA.1/1\n" in txt1 and "@libA.2/1\n" in txt1
    assert st.records_consumed == 4
    st.close()

    # Subsequent wave in arm_c reopens and must emit spots 3..5 (never re-reading spots 1..2!)
    b2 = io.BytesIO()
    assert st.pump(5, b2) == 3
    txt2 = b2.getvalue().decode()
    assert "@libA.1/1\n" not in txt2 and "@libA.2/1\n" not in txt2
    assert "@libA.3/1\n" in txt2 and "@libA.4/1\n" in txt2 and "@libA.5/1\n" in txt2
    assert st.eof is True
    st.close()

    # L51: Empty FASTQ, unequal R1/R2 counts, truncated record
    empty_fq = tmp_path / "empty.fq"
    empty_fq.write_text("")
    with pytest.raises(ValueError, match="(?i)empty"):
        stream_mod.LocalFastqStream("e", empty_fq)

    r2_short = tmp_path / "r2_short.fq.gz"
    with gzip.open(r2_short, "wt") as f2s:
        f2s.write("@orig_1/2\n" + ("ACGT" * 15) + "A\n+\n" + ("I" * 61) + "\n")
    with pytest.raises(ValueError, match="mismatch"):
        stream_mod.LocalFastqStream("mismatch", r1, r2_short)

    trunc_fq = tmp_path / "trunc.fq"
    trunc_fq.write_text("@r1\nACGTACGT\n+\n")
    with pytest.raises(ValueError, match="(?i)truncated"):
        stream_mod.LocalFastqStream("tr", trunc_fq)


def test_51_and_52_short_max_budget_continues_long_reads_with_separate_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#51 + #52: Short-read MAX_BUDGET halts only short libraries while long reads keep streaming with separate cap_state_long."""
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    gfa = tmp_path / "g.fa"
    gfa.write_text(">chr1\n" + ("ACGT" * 500) + "\n")
    sr1 = tmp_path / "s.fq"
    lr1 = tmp_path / "l.fq"
    seq = "ACGT" * 20
    qual = "I" * 80
    sr1.write_text("".join(f"@s{i}\n{seq}\n+\n{qual}\n" for i in range(10)))
    lr1.write_text("".join(f"@l{i}\n{seq}\n+\n{qual}\n" for i in range(6)))

    gr = eng_mod.GateRun(
        "t51",
        root=tmp_path / "runs",
        loop=eng_mod.LoopParams(
            wave_spots=4,
            long_wave_spots=3,
            max_frac_spots=0.4,  # budget = 4 short spots (1 short wave), must NOT stop long waves!
            p0_stop=0.0,
            logan_mode="off",
            enable_structural_ncrna="off",
            threads=1,
            hisat_threads=1,
        ),
        gate={"gate_id": "t51", "fasta_path": str(gfa), "proteins_path": "", "sra_short_runs": [], "sra_long_runs": []},
    )
    gr._skip_tiberius = True
    gr._local_streams["short1"] = {"stream": stream_mod.LocalFastqStream("short1", sr1), "long_read": False, "paired": False}
    gr._local_streams["long1"] = {"stream": stream_mod.LocalFastqStream("long1", lr1), "long_read": True, "paired": False}
    gr.seed()
    monkeypatch.setattr(gr, "hisat_index", lambda *a, **k: tmp_path / "hidx")
    monkeypatch.setattr(gr, "mm2_index", lambda *a, **k: tmp_path / "m.mmi")
    monkeypatch.setattr(eng_mod, "_cat_sort", lambda bams, out, threads: 0.0)
    monkeypatch.setattr(
        gr,
        "_post_synth_c",
        lambda *a, **k: {
            "zoom_wall_s": 0.0,
            "synth_wall_s": 0.0,
            "state_wall_s": 0.0,
            "py_cpu_s": 0.0,
            "synth_stats": {},
            "synth_stats_nozoom": None,
            "microexon_rescues": {},
        },
    )
    def _stub_run_cmd_51(cmd, **kw):
        if cmd and cmd[0] == "stringtie" and "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text("")
        return 0, "", 0.0, 0.0

    monkeypatch.setattr(eng_mod, "run_cmd", _stub_run_cmd_51)

    seen_caps: dict[str, Path] = {}

    def _stub_short_se(stream, n_spots, idx, out_dir, tag, cap_state, caps, **kw):
        seen_caps["short"] = Path(cap_state)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = stream.pump(n_spots, io.BytesIO())
        jt = out_dir / f"{tag}.junc.tsv"
        bt = out_dir / f"{tag}.bins.tsv"
        bm = out_dir / f"{tag}.capped.bam"
        jt.write_text(f"chr1\t100\t200\t{max(1, n)}\t{max(1, n)}\t20\n" if n > 0 else "")
        bt.write_text("chr1\t0\t10\n" if n > 0 else "")
        bm.write_bytes(b"")
        return stream_mod.SliceResult(
            run_id=stream.run_id, spots=n, wall=0.01, bam=str(bm), junc_tsv=str(jt), bins_tsv=str(bt)
        )

    def _stub_long(stream, n_spots, mmi, out_dir, tag, cap_state, caps, **kw):
        seen_caps["long"] = Path(cap_state)
        assert caps == gr.lp.long_caps
        out_dir.mkdir(parents=True, exist_ok=True)
        n = stream.pump(n_spots, io.BytesIO())
        jt = out_dir / f"{tag}.junc.tsv"
        bt = out_dir / f"{tag}.bins.tsv"
        bm = out_dir / f"{tag}.capped.bam"
        jt.write_text(f"chr1\t300\t400\t{max(1, n)}\t{max(1, n)}\t20\n" if n > 0 else "")
        bt.write_text("chr1\t0\t10\n" if n > 0 else "")
        bm.write_bytes(b"")
        return stream_mod.SliceResult(
            run_id=stream.run_id, spots=n, wall=0.01, bam=str(bm), junc_tsv=str(jt), bins_tsv=str(bt)
        )

    monkeypatch.setattr(eng_mod, "align_slice_short_se", _stub_short_se)
    monkeypatch.setattr(eng_mod, "align_slice_long", _stub_long)

    summ = gr.arm_c()
    assert seen_caps["short"].name == "cap_state.u16"
    assert seen_caps["long"].name == "cap_state_long.u16"
    assert summ["library_state"]["short1"]["spots"] == 4
    assert "MAX_BUDGET" in summ["library_state"]["short1"]["reason"]
    assert summ["library_state"]["long1"]["spots"] == 6
    assert summ["library_state"]["long1"]["waves"] == 2


def test_53_and_L63_seed_protein_path_and_readthrough_persistence(tmp_path: Path) -> None:
    """#53 + L63: Blank exclude_genus uses proteins_no_target.faa everywhere; readthrough IDs survive header stripping."""
    gfa = tmp_path / "g.fa"
    gfa.write_text(">chr1\n" + ("ACGT" * 200) + "\n")
    pfa = tmp_path / "in_prot.faa"
    pfa.write_text(
        ">XP_001.1 P2RY11-EIF3G readthrough [Homo sapiens]\nMKTLLLTLVVVTIVCLDLGYT\n"
        ">XP_002.1 normal kinase [Homo sapiens]\nMGSSHHHHHHSSGLVPRGSHM\n"
    )
    gate = {"gate_id": "t53", "fasta_path": str(gfa), "proteins_path": str(pfa), "exclude_genus": ""}
    gr = eng_mod.GateRun("t53", root=tmp_path / "runs", gate=gate)
    gr.prepare()

    expected_prot = gr.dir / "seed" / "proteins_no_target.faa"
    assert gr._seed_protein_path() == expected_prot
    assert expected_prot.is_file()
    ctx = gr._synth_context({"hits": [], "tiberius": []}, cache_dir=gr.dir / "armC", include_raw_alignments=False)
    assert "XP_001.1" in ctx["readthrough_targets"]
    assert "XP_002.1" not in ctx["readthrough_targets"]


def test_54_and_L52_long_read_unoriented_ub_default_and_stranded_uf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#54 + L52: Long-read minimap2 defaults to -ub, switches to -uf when stranded=True, scales -I, and fails on 0 mapped."""
    assert stream_mod.minimap2_index_size_args(0) == []
    assert stream_mod.minimap2_index_size_args(100_000_000) == ["-I", "8G"]
    assert stream_mod.minimap2_index_size_args(6_000_000_000) == ["-I", "10G"]

    parser = build_gl_cli_parser()
    ns_default = parser.parse_args(
        ["--state-dir", "s", "--out-gff3", "a", "--out-proteins", "b", "--out-gtf", "c", "--out-summary", "d", "--out-report", "e"]
    )
    ns_stranded = parser.parse_args(
        ["--long-stranded", "--state-dir", "s", "--out-gff3", "a", "--out-proteins", "b", "--out-gtf", "c", "--out-summary", "d", "--out-report", "e"]
    )
    assert ns_default.long_stranded is False
    assert ns_stranded.long_stranded is True

    # Real minimap2 check: reverse-complemented cDNA read aligns on '+' gene under -ub (default)
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    exon1 = "ATGCGTTCGATCAAGCTAGGCTACCTGACCGTGAACGTAATCGATCGTAGCATCGATCGTAGCTAGCATGCGACTGATGCTAGCTAGC"
    exon2 = "ATGCCCAAACTGTTTGGTCAATCTGCTACTGGTAACCTGGATCGTAAGGCCATGACTGAACTGGAGCGTTACCCTGGATCTGCAATTGA"
    intron = "GT" + ("A" * 120) + "AG"
    chr_seq = ("T" * 150) + exon1 + intron + exon2 + ("G" * 150)
    gfa = tmp_path / "g54.fa"
    gfa.write_text(f">chr1\n{chr_seq}\n")
    mmi = tmp_path / "g54.mmi"
    subprocess.run(["minimap2", "-x", "splice:hq", "-d", str(mmi), str(gfa)], check=True, stderr=subprocess.DEVNULL)

    cdna_rc = revcomp(exon1 + exon2)
    lr_fq = tmp_path / "cdna_rc.fq"
    lr_fq.write_text(f"@read_rc\n{cdna_rc}\n+\n{'I' * len(cdna_rc)}\n")

    st_ub = stream_mod.LocalFastqStream("lr_ub", lr_fq)
    res_ub = stream_mod.align_slice_long(
        st_ub, 10, mmi, tmp_path / "w_ub", "ub", tmp_path / "cap_ub.u16", (80, 240, 80), threads=1, stranded=False
    )
    g54 = Genome(gfa)
    js_ub = JunctionStore()
    js_ub.ingest_tsv("lr_ub", res_ub.junc_tsv, g54)
    j_key = ("chr1", 150 + len(exon1), 150 + len(exon1) + len(intron))
    assert js_ub.support(j_key) == 1
    assert js_ub.strand(j_key) == "+"

    # L52: Multi-part minimap2 index (-I 1K on 25 kb 2-contig genome) omits @SQ headers -> align_slice_long raises!
    multi_fa = tmp_path / "multi54.fa"
    multi_fa.write_text(f">c1\n{chr_seq * 20}\n>c2\n{revcomp(chr_seq) * 20}\n")
    multi_mmi = tmp_path / "multi54.mmi"
    subprocess.run(
        ["minimap2", "-x", "splice:hq", "-I", "1K", "-d", str(multi_mmi), str(multi_fa)],
        check=True,
        stderr=subprocess.DEVNULL,
    )
    st_multipart = stream_mod.LocalFastqStream("lr_mp", lr_fq)
    with pytest.raises(RuntimeError, match="without @SQ headers|missing @SQ"):
        stream_mod.align_slice_long(
            st_multipart, 10, multi_mmi, tmp_path / "w_bad", "bad", tmp_path / "cap_bad.u16", (80, 240, 80), threads=1
        )


def test_55_tiberius_cds_repair_phase_propagation_and_internal_stop_drop(tmp_path: Path) -> None:
    """#55: Propagate non-zero Tiberius cds_phase, trim scaffold-edge partial CDS, and drop internal-stop Tiberius models."""
    sense = "GCTCGTGACTTCGGTCACATCCTG" * 16  # 384 nt (128 aa, divisible by 3, zero stops)
    chr1_phased = ("A" * 40) + "CC" + sense + "TAA" + ("T" * 100)
    broken_cds = "ATG" + ("GCT" * 60) + "TGA" + ("CGT" * 60) + "TAA"
    chr2_broken = ("A" * 50) + broken_cds + ("T" * 100)

    gfa = tmp_path / "g55.fa"
    gfa.write_text(f">chr1\n{chr1_phased}\n>chr2\n{chr2_broken}\n")
    g = Genome(gfa)

    # 5'-partial Tiberius model carrying explicit cds_phase=2 (starts at 'CC' + sense + 'TAA')
    p_start = 40
    p_end = 40 + 2 + len(sense) + 3
    tib_phased = Transcript(
        tid="tib_phased",
        contig="chr1",
        strand="+",
        exons=[(p_start, p_end)],
        cds=[(p_start, p_end)],
        gid="g_phased",
        source="Tiberius",
        cds_phase=2,
    )
    tib_broken = Transcript(
        tid="tib_stop",
        contig="chr2",
        strand="+",
        exons=[(50, 50 + len(broken_cds))],
        cds=[(50, 50 + len(broken_cds))],
        gid="g_stop",
        source="Tiberius",
    )

    loci = seed_mod.partition_loci([tib_phased, tib_broken], [], g)
    res = synth_mod.synthesize(
        g, [tib_phased, tib_broken], [], loci, params=synth_mod.SynthParams(strict_retrocopy_scan=False)
    )
    coding = [m for m in res.models if m.cds]
    assert len(coding) == 1
    assert coding[0].contig == "chr1"
    assert coding[0].cds_phase == 2

    out_gff = tmp_path / "m55.gff3"
    out_faa = tmp_path / "m55.faa"
    write_gff3(res.models, out_gff)
    rg_mod._write_proteins_fasta(res.models, g, out_faa)
    faa_txt = out_faa.read_text()
    assert "*" not in faa_txt
    assert ">GL000001" in faa_txt
    reloaded = read_models(out_gff, fmt="gff3")
    assert reloaded[0].cds_phase == 2
    nt = g.spliced(reloaded[0].contig, reloaded[0].strand, reloaded[0].cds)[reloaded[0].cds_phase :]
    assert "*" not in translate(nt).rstrip("*")


def test_56_and_L53_locus_zoom_flank_guard_1codon_corroboration_and_phase_microexon(tmp_path: Path) -> None:
    """#56 + L53: Reject intronic/exonic flank-matching soft clips, gate 1-codon clips, rescue phase!=0 TGA-prefix microexon."""
    # L53: 6 bp micro-exon 'TGAAAC' starts with 'TGA' in frame 0, but with left exon len=4 ('ATGA' + 'TGAAAC' + 'GGCGTAA' = 'ATG'|'ATG'|'AAA'|'CGG'|'CGT'? Wait: 4+6+8=18 -> 'ATGA'+'TGAAAC'+'GGCGTTAA' = ATG|ATG|AAA|CGG|CGT|TAA = MMKRR*!)
    intron_seq = ("T" * 20) + "AG" + "TGAAAC" + "GT" + ("A" * 20)
    hit = lz_mod.scan_junction_gap_micro_exon(intron_seq, "TGAAAC", allow_mismatches=0)
    assert hit is not None and hit.exon_seq == "TGAAAC"

    chr_seq = ("A" * 50) + "ATGA" + intron_seq + "GGCGTTAA" + ("T" * 50)
    gfa = tmp_path / "g56.fa"
    gfa.write_text(f">chr1\n{chr_seq}\n")
    g = Genome(gfa)
    ga = 54
    gb = 54 + len(intron_seq)
    ms = ga + 22
    me = ms + 6
    cand_in = synth_mod.Cand("chr1", "+", ((50, ga), (gb, gb + 8)), {"tiberius"})
    rescued_cand = synth_mod.insert_microexons(cand_in, g, {("chr1", ga, gb): [(ms, me, 5)]})
    assert (ms, me) in rescued_cand.cds
    assert translate(g.spliced("chr1", "+", list(rescued_cand.cds))) == "MMKRR*"

    # #56: Soft clip matching plain intronic donor flank must be rejected in _vote_right
    plain_fa = tmp_path / "plain56.fa"
    plain_seq = ("C" * 100) + "GTACGTACGTACGTACGTAC" + ("T" * 80) + "AGGCT" + "GTAAGT" + ("A" * 200)
    plain_fa.write_text(f">chr1\n{plain_seq}\n")
    g_plain = Genome(plain_fa)
    a_iv, b_iv = 100, 311
    intron_oriented = plain_seq[a_iv:b_iv]
    right_flank = plain_seq[b_iv : b_iv + 30]
    votes_r: dict = {}
    lz_mod._vote_right(
        votes_r,
        "chr1",
        a_iv,
        b_iv,
        intron_oriented,
        plain_seq[a_iv : a_iv + 15],
        right_flank,
        g_plain,
        "+",
        raw_intron=intron_oriented,
    )
    assert votes_r == {}

    # 3-bp exon with short 5-bp remainder (< 8 bp required for m==3) must also be rejected
    votes_r2: dict = {}
    lz_mod._vote_right(
        votes_r2,
        "chr1",
        a_iv,
        b_iv,
        intron_oriented,
        "GCT" + right_flank[:5],
        right_flank,
        g_plain,
        "+",
        raw_intron=intron_oriented,
    )
    assert votes_r2 == {}


def test_57_L47_L48_orthodb_taxid_filter_subsample_scratch_and_leakage_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#57 + L47 + L48: Filter TAXID_N: headers in OrthoDB cached/full_partition, subsample cached, validate downloads, audit TAXID."""
    cache_dir = tmp_path / "orthodb_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("FOCALOOP_ORTHODB_CACHE_DIR", str(cache_dir))

    cached_part = cache_dir / "orthodb_v12_arthropoda.faa"
    cached_part.write_text(
        ">7227_0:000001\n" + ("MKTLLLTLVVVTIVCLDLGYT" * 4) + "\n"
        ">7220_0:000002\n" + ("MGSSHHHHHHSSGLVPRGSHM" * 4) + "\n"
        ">7165_0:000003\n" + ("MGDVEKGKKIFIMKCSQCHTV" * 4) + "\n"
        ">7070_0:000004\n" + ("MVLSPADKTNVKAAWGKVGAH" * 4) + "\n"
    )

    class _FakeResp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._p

    class _FakeSess:
        def get(self, url, params=None, timeout=15.0):
            return _FakeResp(
                {
                    "data": [
                        {"taxon_id": "7227_0", "sciname": "Drosophila melanogaster"},
                        {"taxon_id": "7220_0", "sciname": "Drosophila simulans"},
                    ]
                }
            )

    monkeypatch.setattr(hf_mod, "_ipv4_session", lambda: _FakeSess())
    monkeypatch.setattr(hf_mod, "_fetch_json", lambda url, params=None, **kw: {})

    # Cached representatives mode filters target (7227) + excluded genus (7220) AND subsamples to max_species=1!
    dest_rep = tmp_path / "job_rep"
    meta_rep = hf_mod.fetch_orthodb_clade_proteins(
        clade="insecta",
        dest_dir=dest_rep,
        max_species=1,
        orthodb_mode="representatives",
        target_ctx={"tax_id": "7227", "genus": "Drosophila", "genus_taxid": "7215"},
        exclude_genus="Drosophila",
    )
    rep_txt = Path(meta_rep["proteins_fasta"]).read_text()
    assert "7227_0:" not in rep_txt
    assert "7220_0:" not in rep_txt
    assert rep_txt.count(">") == 1

    # L47: full_partition rejects unknown non-partition taxon override
    with pytest.raises(ValueError, match="partition"):
        hf_mod.fetch_orthodb_clade_proteins(
            clade="insecta",
            dest_dir=tmp_path / "job_bad",
            protein_query="CompletelyUnknownTaxonXYZ",
            orthodb_mode="full_partition",
        )

    # Read-only cache directory falls back cleanly to job scratch
    ro_cache = tmp_path / "ro_cache"
    ro_cache.mkdir()
    ro_cache.chmod(0o555)
    try:
        fb_dir = hf_mod._writable_dir_or_fallback(ro_cache / "sub", tmp_path / "job_rw")
        assert fb_dir == tmp_path / "job_rw"
        assert os.access(fb_dir, os.W_OK)
    finally:
        ro_cache.chmod(0o755)

    # L48: Non-gzip HTML error page disguised as .faa.gz is rejected
    monkeypatch.setattr(hf_mod, "resolve_ncbi_assembly_dir", lambda acc: (f"{acc}_ASM", f"mock://{acc}"))
    monkeypatch.setattr(hf_mod, "_download_file", lambda url, dst, **kw: dst.write_bytes(b"<html>503 error</html>"))
    with pytest.raises(RuntimeError, match="not a valid gzip"):
        hf_mod.fetch_ncbi_assembly_proteins("GCF_000002035.6", tmp_path / "ncbi_bad")

    # #57: leakage_audit strips TAXID_N: prefixes matching target taxid and marks unverifiable when taxid is absent
    src_odb = tmp_path / "src_odb.faa"
    src_odb.write_text(
        ">7227_0:000999\nMKTLLLTLVVVTIVCLDLGYT\n"
        ">7165_0:000003\nMGDVEKGKKIFIMKCSQCHTV\n"
    )
    audit_dir = tmp_path / "audit_run"
    audit_res = seed_mod.leakage_audit(
        {"exclude_genus": "Drosophila", "taxid": 7227, "proteins_path": str(src_odb)},
        audit_dir,
        StageLog(audit_dir / "stages.jsonl"),
    )
    assert audit_res["proteins_removed"] == 1
    assert audit_res["leakage_audit_status"] == "verified_clean"
    assert audit_res["self_protein_leakage"] is False
    assert "7227_0:" not in (audit_dir / "seed" / "proteins_no_Drosophila.faa").read_text()

    audit_unv_dir = tmp_path / "audit_unverifiable"
    audit_unv = seed_mod.leakage_audit(
        {"exclude_genus": "Drosophila", "taxid": 0, "proteins_path": str(src_odb)},
        audit_unv_dir,
        StageLog(audit_unv_dir / "stages.jsonl"),
    )
    assert audit_unv["leakage_audit_status"] == "unverifiable"


def test_58_logan_bam_never_overwrites_or_unlinks_capped_short_bam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#58: Logan contigs + History short BAMs merge cleanly into state/capped_short.bam without self-overwrite or deletion."""
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    gfa = tmp_path / "g58.fa"
    gfa.write_text(">chr1\n" + ("ACGT" * 500) + "\n")

    hist_bam = tmp_path / "hist.bam"
    sam_hist = (
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:2000\n"
        f"hist_read\t0\tchr1\t120\t60\t40M\t*\t0\t0\t{'A' * 40}\t{'I' * 40}\tNH:i:1\n"
    )
    subprocess.run(["samtools", "view", "-b", "-o", str(hist_bam), "-"], input=sam_hist.encode(), check=True)

    gr = eng_mod.GateRun(
        "t58",
        root=tmp_path / "runs",
        loop=eng_mod.LoopParams(logan_mode="auto", enable_structural_ncrna="off", threads=1, hisat_threads=1),
        gate={"gate_id": "t58", "fasta_path": str(gfa), "proteins_path": "", "sra_short_runs": ["SRR888888"], "sra_long_runs": []},
    )
    gr._skip_tiberius = True
    gr._local_bams.append({"run_id": "hb1", "bam": hist_bam, "long_read": False})
    gr.seed()

    monkeypatch.setattr(gr, "partition_short_runs", lambda: (["SRR888888"], [], {"logan_mode": "auto", "skipped_short_runs": []}))
    monkeypatch.setattr(eng_mod, "detect_cross_species_rna", lambda *a, **k: (False, {}))

    def _fake_align_logan(*, logan_runs, genome_fa, mm2_flags, glcore_bin, out_dir, tmp_dir, bam_out, threads, nic, genome_bp=0):
        assert Path(bam_out).name == "logan_contigs.bam"
        out_dir.mkdir(parents=True, exist_ok=True)
        sam_logan = (
            "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:2000\n"
            f"SRR888888_0\t0\tchr1\t300\t60\t40M\t*\t0\t0\t{'C' * 40}\t*\tNH:i:1\tka:f:10.0\n"
        )
        subprocess.run(["samtools", "view", "-b", "-o", str(bam_out), "-"], input=sam_logan.encode(), check=True)
        subprocess.run(["samtools", "index", str(bam_out)], check=True)
        ka = out_dir / "logan.per_run.ka.tsv"
        unw = out_dir / "logan.per_run.unw.tsv"
        bns = out_dir / "logan.bins.ka.tsv"
        ka.write_text("SRR888888\tchr1\t340\t440\t10\t10\t20\n")
        unw.write_text("SRR888888\tchr1\t340\t440\t1\t1\t20\n")
        bns.write_text("chr1\t0\t10\n")
        return {
            "junc_ka_path": str(ka),
            "junc_unw_path": str(unw),
            "bins_ka_path": str(bns),
            "bam_path": str(bam_out),
            "wire_bytes": 1000,
            "wall_s": 0.01,
            "total_cpu_s": 0.01,
            "cpu_by_proc_s": {},
            "stream_stats": {},
            "zero_record_runs": [],
            "zero_mapped_runs": [],
        }

    monkeypatch.setattr(eng_mod, "align_logan_contigs", _fake_align_logan)

    def _fake_run_cmd(cmd, **kw):
        if cmd[0] == "stringtie":
            out_i = cmd.index("-o") + 1
            Path(cmd[out_i]).write_text("")
            return 0, "", 0.0, 0.0
        return tele_mod.run_cmd(cmd, **kw)

    monkeypatch.setattr(eng_mod, "run_cmd", _fake_run_cmd)
    summ = gr.arm_c()
    merged_p = gr.dir / f"arm{gr.c_tag}" / "state" / "capped_short.bam"
    assert merged_p.is_file() and merged_p.stat().st_size > 0
    reads = subprocess.run(["samtools", "view", str(merged_p)], capture_output=True, text=True, check=True).stdout
    assert "hist_read" in reads and "SRR888888_0" in reads
    assert summ["logan"]["n_logan_runs"] == 1


def test_59_good_turing_judge_deterministic_from_spots_not_nic_bytes() -> None:
    """#59: GoodTuringUCBJudge decisions depend on spots, never host-wide NIC rx() wire_bytes."""
    def _run_judge(wire_a: int, wire_b: int) -> tuple[dict, str | None, list[float]]:
        j = GoodTuringUCBJudge(p0_stop=0.01, patience_waves=1, replicate_slope_floor_per_10mb=5.0)
        j.register_library("L1", "L1")
        j.register_library("L2", "L2")
        w1 = [WaveJunction(("chr1", 100 + i * 20, 150 + i * 20, "+"), 2, "GT-AG", "unsettled") for i in range(2)]
        w2 = [WaveJunction(("chr1", 500 + i * 20, 550 + i * 20, "+"), 2, "GT-AG", "unsettled") for i in range(25)]
        r1 = j.ingest_wave_counts("L1", wire_a, w1, spots=125_000)
        r2 = j.ingest_wave_counts("L2", wire_b, w2, spots=125_000)
        return {"r1": r1, "r2": r2}, j.select_next_library_gt_ucb(), list(j.recent_ncrna_slopes)

    rep_zero, pick_zero, nc_zero = _run_judge(0, 0)
    rep_noisy, pick_noisy, nc_noisy = _run_judge(999_999_999, 1)
    assert rep_zero["r1"]["slope_per_10mb"] == rep_noisy["r1"]["slope_per_10mb"]
    assert rep_zero["r2"]["slope_per_10mb"] == rep_noisy["r2"]["slope_per_10mb"]
    assert rep_zero["r1"]["lib_halted"] == rep_noisy["r1"]["lib_halted"] is True
    assert nc_zero == nc_noisy
    assert pick_zero == pick_noisy == "L2"


def test_60_cgroup_hierarchy_memory_gating_bs1_stepdown_and_auto_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#60: Walk cgroup parent+leaf limits + GALAXY_MEMORY_MB, step down to bs=1, and fall back to fast_evidence in auto mode."""
    cg_root = tmp_path / "cgroup"
    parent_cg = cg_root / "galaxy.slice"
    leaf_cg = parent_cg / "job_123.scope"
    leaf_cg.mkdir(parents=True)
    (parent_cg / "memory.max").write_text(str(16 * 1024 ** 3))
    (parent_cg / "memory.current").write_text(str(2 * 1024 ** 3))
    (leaf_cg / "memory.max").write_text("max")
    proc_cg = tmp_path / "proc_cgroup"
    proc_cg.write_text("0::/galaxy.slice/job_123.scope\n")

    lim, rem = tele_mod.cgroup_and_scheduler_limits_bytes(proc_cgroup=proc_cg, cgroup_root=cg_root, env={})
    assert lim == 16 * 1024 ** 3
    assert rem == 14 * 1024 ** 3

    lim_g, _ = tele_mod.cgroup_and_scheduler_limits_bytes(
        proc_cgroup=tmp_path / "missing_cg", cgroup_root=cg_root, env={"GALAXY_MEMORY_MB": "8192"}
    )
    assert lim_g == 8192 * 1024 * 1024

    # Stepdown: 22 GiB available with reserve=8 GiB -> 14 GiB free -> steps down bs=4 -> bs=2 -> bs=1 (12G)!
    monkeypatch.setattr(eng_mod, "_read_mem_available_gib", lambda: 22.0)
    k_eff, bs_ok, stepped, _ = eng_mod._approve_wave_workers_and_bs(2, "vertebrates", reserve_gib=8.0)
    assert (k_eff, bs_ok, stepped) == (1, 1, True)

    # Below bs=1 -> raises TiberiusMemoryError
    monkeypatch.setattr(eng_mod, "_read_mem_available_gib", lambda: 14.0)
    monkeypatch.setattr(eng_mod, "_read_mem_total_gib", lambda: 16.0)
    with pytest.raises(eng_mod.TiberiusMemoryError):
        eng_mod._approve_wave_workers_and_bs(1, "vertebrates", reserve_gib=8.0)

    # In auto mode with cached weights + runtime on CPU but tight cgroup RAM, resolve_execution_mode falls back to fast_evidence!
    monkeypatch.setattr(rg_mod, "gpu_available", lambda: False)
    monkeypatch.setattr(rg_mod, "detect_tiberius_runtime", lambda: {"backend": "python"})
    monkeypatch.setattr(rg_mod, "discover_tiberius_weights", lambda *a, **k: tmp_path / "w")
    mode_info = rg_mod.resolve_execution_mode("auto", "vertebrates", has_proteins=True, has_rna=False)
    assert mode_info["effective_mode"] == "fast_evidence"
    assert mode_info["skip_tiberius"] is True
    assert "CPU RAM" in mode_info["reason"]


def test_73_tar_slip_hardening_in_discover_tiberius_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#73: Reject ../ path traversal, absolute paths, symlinks, and hardlinks in Tiberius weight tarballs."""
    monkeypatch.setenv("TIBERIUS_EXTRACT_CACHE", str(tmp_path / "extract_cache"))

    def _make_tar(path: Path, mutate_fn) -> Path:
        with tarfile.open(path, "w:gz") as tf:
            mutate_fn(tf)
        return path

    t_trav = tmp_path / "trav.tar.gz"
    def _add_trav(tf: tarfile.TarFile) -> None:
        ti = tarfile.TarInfo(name="../escaped_file.txt")
        data = b"pwned"
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    _make_tar(t_trav, _add_trav)
    with pytest.raises(ValueError, match="Unsafe|Disallowed"):
        seed_mod.discover_tiberius_weights("vertebrates", explicit_weights=t_trav)
    assert not (tmp_path / "escaped_file.txt").exists()

    t_sym = tmp_path / "sym.tar.gz"
    def _add_sym(tf: tarfile.TarFile) -> None:
        ti = tarfile.TarInfo(name="weights/link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    _make_tar(t_sym, _add_sym)
    with pytest.raises(ValueError, match="Disallowed|Unsafe"):
        seed_mod.discover_tiberius_weights("vertebrates", explicit_weights=t_sym)

    t_ok = tmp_path / "clean.tar.gz"
    def _add_ok(tf: tarfile.TarFile) -> None:
        data = b'{"model": "vertebrates"}'
        ti = tarfile.TarInfo(name="vertebrates_w/model_config.json")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    _make_tar(t_ok, _add_ok)
    w_res = seed_mod.discover_tiberius_weights("vertebrates", explicit_weights=t_ok)
    assert w_res is not None and (w_res / "model_config.json").is_file()


def test_74_portable_isa_default_sys_prefix_lookup_and_0700_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#74: Default to portable ISA (-msse4.2 -mpopcnt), check sys.prefix/share/focaloop/native_cache first, 0700 user cache."""
    monkeypatch.delenv("FOCALOOP_PORTABLE_ISA", raising=False)
    monkeypatch.delenv("FOCALOOP_BUILD_NATIVE_DIR", raising=False)
    monkeypatch.delenv("GLCORE_BIN", raising=False)
    monkeypatch.delenv("ORF_SCOUT_LIB", raising=False)

    assert ss_mod._portable_isa_enabled() is True
    assert orf_mod._portable_isa_enabled() is True
    monkeypatch.setenv("FOCALOOP_PORTABLE_ISA", "0")
    assert ss_mod._portable_isa_enabled() is False
    monkeypatch.delenv("FOCALOOP_PORTABLE_ISA", raising=False)

    # Build into per-user 0700 cache and verify permissions + hashed naming
    user_cache = tmp_path / "user_native_cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(user_cache))
    glc_p = ss_mod.build_glcore(force=True)
    orf_p = orf_mod.build_orf_scout(force=True)
    assert glc_p.is_file() and glc_p.name.startswith("glcore_")
    assert orf_p.is_file() and orf_p.name.startswith("liborf_scout_")
    cache_dir = glc_p.parent
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700

    # Stage prebuilt binaries in fake sys.prefix/share/focaloop/native_cache and verify CC is NOT needed!
    fake_prefix = tmp_path / "fake_prefix"
    sys_cache = fake_prefix / "share" / "focaloop" / "native_cache"
    sys_cache.mkdir(parents=True)
    shutil.copy2(glc_p, sys_cache / "glcore")
    shutil.copy2(orf_p, sys_cache / "liborf_scout.so")
    monkeypatch.setattr(ss_mod.sys, "prefix", str(fake_prefix))
    monkeypatch.setattr(orf_mod.sys, "prefix", str(fake_prefix))
    monkeypatch.setenv("CC", "/nonexistent_compiler_74")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty_xdg"))
    assert ss_mod.build_glcore() == sys_cache / "glcore"
    assert orf_mod.build_orf_scout() == sys_cache / "liborf_scout.so"

    # Symlinked per-user cache directory is rejected
    sym_target = tmp_path / "real_dir"
    sym_target.mkdir(mode=0o700)
    sym_link = tmp_path / "sym_cache"
    sym_link.symlink_to(sym_target)
    with pytest.raises(RuntimeError, match="symlink"):
        ss_mod._ensure_private_cache_dir(sym_link)


def test_lesser_L54_L55_L56_L58_L59_L60_L62_L64_L89_L94_L96(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify remaining Lesser findings: L54-L56, L58-L60, L62, L64, L89, L94, L96."""
    # L54: parent_intron_counts raises RuntimeError when single+multi exist and diamond is missing/fails
    gfa = tmp_path / "g.fa"
    sense_seq = "ATG" + ("GCTCGTGACTTC" * 30) + "TAA"
    gfa.write_text(f">chr1\n{'A' * 100}{sense_seq}{'T' * 200}{sense_seq[:180]}GT{'A' * 90}AG{sense_seq[180:]}{'C' * 100}\n")
    g = Genome(gfa)
    c_single = synth_mod.Cand("chr1", "+", ((100, 100 + len(sense_seq)),), {"tiberius"})
    m_s1 = 100 + len(sense_seq) + 200
    m_e1 = m_s1 + 180
    m_s2 = m_e1 + 94
    m_e2 = m_s2 + (len(sense_seq) - 180)
    c_multi = synth_mod.Cand("chr1", "+", ((m_s1, m_e1), (m_s2, m_e2)), {"miniprot"})

    def _raise_missing_diamond(*a, **k):
        raise FileNotFoundError("diamond")

    monkeypatch.setattr(synth_mod, "run_cmd", _raise_missing_diamond)
    with pytest.raises(RuntimeError, match="diamond"):
        synth_mod.parent_intron_counts([c_single], [c_multi], g, tmp_path / "retro_strict", strict=True)
    _, _, scan_failed = synth_mod.parent_intron_counts([c_single], [c_multi], g, tmp_path / "retro_soft", strict=False)
    assert scan_failed is True

    # L55: samtools_sort_mem_arg scales down under tight memory limit
    assert tele_mod.samtools_sort_mem_arg(8, avail_bytes=1024 * 1024 * 1024) == "64M"
    assert tele_mod.samtools_sort_mem_arg(4, avail_bytes=32 * 1024 ** 3) == "1024M"

    # L56: GTF biotype & pseudo attributes round-trip
    tx_pseudo = Transcript(
        tid="p1.t1", contig="chr1", strand="+", exons=[(10, 100)], cds=[], gid="p1", biotype="pseudogene"
    )
    tx_lnc = Transcript(
        tid="l1.t1", contig="chr1", strand="-", exons=[(200, 300), (400, 500)], cds=[], gid="l1", biotype="lnc_RNA"
    )
    gtf_p = tmp_path / "biotype.gtf"
    write_gtf([tx_pseudo, tx_lnc], gtf_p)
    gtf_txt = gtf_p.read_text()
    assert 'gene_biotype "pseudogene";' in gtf_txt
    assert 'transcript_biotype "pseudogene";' in gtf_txt
    assert 'pseudo "true";' in gtf_txt
    rt_models = {m.tid: m for m in read_models(gtf_p, fmt="gtf")}
    assert rt_models["p1.t1"].biotype == "pseudogene"
    assert rt_models["l1.t1"].biotype in ("lncRNA", "lnc_RNA")

    # L58 + L59: Partial ncRNA asset detection + log tail preservation before work_dir cleanup
    monkeypatch.setattr(
        ncrna_mod.shutil,
        "which",
        lambda b, *a, **kw: f"/usr/bin/{b}" if b in ("tRNAscan-SE", "barrnap") else None,
    )
    nc_assets = ncrna_mod.check_ncrna_assets(rfam_dir=tmp_path / "no_rfam", mode="auto")
    assert nc_assets["available"] is True
    assert nc_assets["status"] == "partial"
    assert nc_assets["stages"] == {"trnascan": True, "barrnap": True, "rfam": False}
    with pytest.raises(FileNotFoundError):
        ncrna_mod.check_ncrna_assets(rfam_dir=tmp_path / "no_rfam", mode="on")

    fail_log = tmp_path / "ncrna_fail.log"
    with pytest.raises(RuntimeError, match="custom_stderr_diagnostic_line"):
        ncrna_mod._run_ncrna_cmd(
            [sys.executable, "-c", "import sys; sys.stderr.write('custom_stderr_diagnostic_line\\n'); sys.exit(3)"],
            log_path=fail_log,
            stage_label="tRNAscan-SE",
        )

    # L60: Nested intronic gene inside host intron is retained by CDS-block MWIS scheduler!
    host_cds1 = "ATG" + ("GCT" * 40)
    nested_cds = "ATG" + ("CGT" * 50) + "TAA"
    host_cds2 = ("GAC" * 40) + "TAA"
    nested_genome_seq_rc = (
        ("A" * 100)
        + host_cds1
        + "GT"
        + ("T" * 150)
        + revcomp(nested_cds)
        + ("T" * 150)
        + "AG"
        + host_cds2
        + ("C" * 100)
    )
    g_nest_fa = tmp_path / "g_nest.fa"
    g_nest_fa.write_text(f">chr1\n{nested_genome_seq_rc}\n")
    g_nest = Genome(g_nest_fa)
    h1_a, h1_b = 100, 100 + len(host_cds1)
    nest_a = h1_b + 2 + 150
    nest_b = nest_a + len(nested_cds)
    h2_a = nest_b + 150 + 2
    h2_b = h2_a + len(host_cds2)
    t_host = Transcript("host.t1", "chr1", "+", [(h1_a, h1_b), (h2_a, h2_b)], [(h1_a, h1_b), (h2_a, h2_b)], gid="host")
    t_inner = Transcript("inner.t1", "chr1", "-", [(nest_a, nest_b)], [(nest_a, nest_b)], gid="inner")
    loci_nest = seed_mod.partition_loci([t_host, t_inner], [], g_nest)
    assert len(loci_nest) == 2
    res_nest = synth_mod.synthesize(
        g_nest, [t_host, t_inner], [], loci_nest, params=synth_mod.SynthParams(strict_retrocopy_scan=False)
    )
    assert len([m for m in res_nest.models if m.cds]) == 2

    # L62: Two-pass lift_miniprot_gff3 lifts window coords and deduplicates overlap-zone duplicates
    shard_gff = tmp_path / "shard.gff3"
    shard_gff.write_text(
        "##gff-version 3\n"
        "##PAF\tp1\t100\t0\t100\t+\tchr1__w1000_5000\t4000\t200\t500\t300\t300\t60\n"
        "chr1__w1000_5000\tminiprot\tmRNA\t201\t500\t300\t+\t.\tID=MP0001;Target=p1 1 100\n"
        "chr1__w1000_5000\tminiprot\tCDS\t201\t500\t300\t+\t0\tParent=MP0001;Target=p1 1 100\n"
        "chr1__w1100_5100\tminiprot\tmRNA\t101\t400\t290\t+\t.\tID=MP0002;Target=p1 1 100\n"
        "chr1__w1100_5100\tminiprot\tCDS\t101\t400\t290\t+\t0\tParent=MP0002;Target=p1 1 100\n"
    )
    lifted_gff = tmp_path / "lifted.gff3"
    wmap = {
        "chr1__w1000_5000": ("chr1", 1000, 5000),
        "chr1__w1100_5100": ("chr1", 1100, 5100),
    }
    n_kept = seed_mod.lift_miniprot_gff3([(shard_gff, wmap)], lifted_gff)
    assert n_kept == 1
    lifted_txt = lifted_gff.read_text()
    assert "chr1\tminiprot\tmRNA\t1201\t1500\t300" in lifted_txt
    assert "ID=MP000001;" in lifted_txt

    # L64: run_tiberius raises RuntimeError if command exits 0 without writing GTF
    w_dir = tmp_path / "fake_weights"
    w_dir.mkdir()
    (w_dir / "model_config.json").write_text("{}")
    monkeypatch.setattr(seed_mod, "discover_tiberius_weights", lambda *a, **k: w_dir)
    monkeypatch.setattr(
        seed_mod,
        "detect_tiberius_runtime",
        lambda: {"backend": "python", "python": sys.executable, "script": "-c"},
    )
    monkeypatch.setattr(seed_mod, "run_cmd", lambda *a, **k: (0, "", 0.0, 0.0))
    with pytest.raises(RuntimeError, match="without writing output GTF"):
        seed_mod.run_tiberius(
            {"tiberius_model_cfg": "vertebrates"},
            gfa,
            tmp_path / "tib_out",
            StageLog(tmp_path / "st.jsonl"),
            cpu_only=True,
        )

    # L89: Truncated TSV without trailing newline rejected by JunctionStore & BinStore
    trunc_j = tmp_path / "trunc_j.tsv"
    trunc_j.write_text("chr1\t100\t200\t5\t5\t20")  # missing '\n'!
    with pytest.raises(ValueError, match="Truncated"):
        JunctionStore().ingest_tsv("r1", trunc_j, g)
    trunc_b = tmp_path / "trunc_b.tsv"
    trunc_b.write_text("chr1\t0\t1")  # missing '\n'!
    with pytest.raises(ValueError, match="Truncated"):
        BinStore(g.lengths).ingest_tsv(trunc_b)

    # L94 + L96: Leading '-' stripped, ID length mismatch raises, XML empty-BAM guard precedes focaloop geneloop
    assert rg_mod._sanitize_run_id("-weird-sample-", "fb") == "weird-sample"
    assert rg_mod._sanitize_run_id("---", "fb") == "fb"

    with pytest.raises(ValueError, match="--short-read-ids count"):
        rg_mod.run_geneloop_galaxy(
            genome_fasta=gfa,
            proteins_fasta=gfa,
            short_reads_1=[gfa],
            short_read_ids=["a", "b"],
            mode="fast_evidence",
            out_gff3=tmp_path / "x.gff3",
            out_proteins=tmp_path / "x.faa",
            out_gtf=tmp_path / "x.gtf",
            out_summary=tmp_path / "x.tsv",
            out_report=tmp_path / "x.html",
            state_dir=tmp_path / "xst",
        )

    xml_path = REPO_ROOT / "tools" / "focaloop" / "geneloop.xml"
    cmd_txt = ET.parse(xml_path).getroot().find("command").text or ""
    assert cmd_txt.index("Select at least one pre-aligned short-read or long-read BAM dataset.") < cmd_txt.index("focaloop geneloop")
    assert 'lstrip("-")' in cmd_txt
