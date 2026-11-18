"""Tests for Focaloop GeneLoop packaging, C/OpenMP native cores, biological invariants, Galaxy XML/UDT, and runner."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

from focaloop.galaxy_lint import lint_tool_xml, load_and_lint_user_tool
from focaloop.geneloop import (
    OpenSyncmerDepletionFilter,
    build_glcore,
    discover_tiberius_weights,
    gpu_available,
    prebuild_native_libs,
)
from focaloop.geneloop.intron_judge import GoodTuringUCBJudge, WaveJunction
from focaloop.geneloop.locus_zoom import (
    SingleExonCandidate,
    classify_single_exon_candidate,
    scan_junction_gap_micro_exon,
)
from focaloop.geneloop.orf_scout import build_orf_scout, orphan_orf_delta_windows, scan_orphan_orfs
from focaloop.geneloop.sequtil import Genome
from focaloop.geneloop.sieve_syncmer import LocusSaturationState, extract_open_syncmers
from focaloop.geneloop.stream import LocalFastqStream
from focaloop.runners_geneloop import (
    CLADE_CONFIGS,
    _resolve_rna_manifest,
    _write_summary_tsv,
    prebuild_geneloop_native_libs,
    render_geneloop_report,
    resolve_execution_mode,
    run_geneloop_galaxy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "tools" / "focaloop" / "test-data" / "geneloop"


def test_native_build_glcore_and_orf_scout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native_dir = tmp_path / "native"
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(native_dir))
    monkeypatch.setenv("FOCALOOP_PORTABLE_ISA", "1")
    monkeypatch.delenv("GLCORE_BIN", raising=False)
    monkeypatch.delenv("ORF_SCOUT_LIB", raising=False)

    built = prebuild_geneloop_native_libs()
    glcore = Path(built["geneloop_glcore"])
    orf_lib = Path(built["geneloop_orf_scout"])
    assert glcore.is_file() and os.access(glcore, os.X_OK)
    assert orf_lib.is_file()
    assert build_glcore() == glcore
    assert build_orf_scout() == orf_lib


def test_glcore_pipeline_with_local_fastq_stream_and_paralog_protection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native_dir = tmp_path / "native"
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(native_dir))
    monkeypatch.delenv("GLCORE_BIN", raising=False)
    glcore = build_glcore(force=True)

    # Construct two identical paralog exons on chr1 + chr2 and one unique exon on chr1, plus an organellar contig.
    paralog_exon = (
        "ATGCCCAAACTGTTTGGTCAATCTGCTACTGGTAACCTGGATCGTAAGGCCATGACTGAACTGGAGCGTTAC"
        "CCTGGATCTGCAATTGGTACCCTGATGGTCATGACTGGCTAACTGACCAGTTC"
    )
    unique_exon = (
        "ATGCGTTCGATCAAGCTAGGCTACCTGACCGTGAACGTAATCGATCGTAGCATCGATCGTAGCTAGCATGCGA"
        "CTGATGCTAGCTAGCATCGATCGTAGCTAGCATCGATCAACTGATGCTAGCTA"
    )
    chr1_seq = "A" * 120 + paralog_exon + "T" * 120 + unique_exon + "G" * 120
    chr2_seq = "C" * 120 + paralog_exon + "T" * 120
    chrM_seq = "GATCGTAGCTAGTCAGCTGATCGTAGCTAGCTGACTGATCGTAGCTGATCGATGCTAGCTAGCTGACTGATCGTAGCTGA"

    # Python OpenSyncmerDepletionFilter paralog protection invariant
    py_filt = OpenSyncmerDepletionFilter(k=31, s=15, min_exon_depth=20.0, min_junction_depth=20, max_singleton_ratio=0.05)
    py_filt.index_genome_contigs({"chr1": chr1_seq, "chr2": chr2_seq, "chrM": chrM_seq}, organellar_ids={"chrM"})
    paralog_syncs = extract_open_syncmers(paralog_exon, 31, 15)
    unique_syncs = extract_open_syncmers(unique_exon, 31, 15)
    assert paralog_syncs and unique_syncs
    assert all(py_filt.genomic_multiplicity[sm] >= 2 for sm in paralog_syncs)
    assert all(py_filt.genomic_multiplicity[sm] == 1 for sm in unique_syncs)

    sat_state = LocusSaturationState(
        locus_id="L1",
        min_exon_depth=60.0,
        min_junction_depth=60,
        total_spliced_obs=120,
        singleton_junctions=1,
    )
    py_filt.try_saturate_locus(sat_state, paralog_exon + "NNNNN" + unique_exon)
    for sm in paralog_syncs:
        assert sm not in py_filt.active_depletion
    for sm in unique_syncs:
        assert sm in py_filt.active_depletion

    # Now exercise native C glcore: mult -> activate -> LocalFastqStream.pump -> sieve
    genome_fa = tmp_path / "genome.fa"
    genome_fa.write_text(f">chr1\n{chr1_seq}\n>chr2\n{chr2_seq}\n>chrM\n{chrM_seq}\n")
    org_fa = tmp_path / "org.fa"
    org_fa.write_text(f">chrM\n{chrM_seq}\n")

    cand_fa = tmp_path / "cand.fa"
    cand_fa.write_text(f">L_unique|chr1\n{unique_exon}\n>L_paralog|chr1\n{paralog_exon}\n")

    mult_bin = tmp_path / "cand.mult"
    act_bin = tmp_path / "active.bin"
    subprocess.run([str(glcore), "mult", "-g", str(genome_fa), "-c", str(cand_fa), "-o", str(mult_bin), "-t", "2"], check=True)
    sat_file = tmp_path / "sat.txt"
    sat_file.write_text("L_unique\nL_paralog\n")
    subprocess.run(
        [str(glcore), "activate", "-m", str(mult_bin), "-c", str(cand_fa), "-l", str(sat_file), "-b", str(org_fa), "-o", str(act_bin)],
        check=True,
    )

    # Create paired reads: 1 unique saturated read-pair (dropped), 1 paralog read-pair (kept!), 1 novel read-pair (kept!)
    novel_seq = "ACGTAGCTGACTGATCGATGCTAGTCGATCGATGCTAGCTAGCTGATCGATGCTAGCTAGCTGACTGATCGATGCTAGCTAG"
    fq1 = tmp_path / "r1.fastq.gz"
    fq2 = tmp_path / "r2.fastq.gz"
    q = "I" * 80
    with gzip.open(fq1, "wt") as f1, gzip.open(fq2, "wt") as f2:
        for nm, seq in [("sat_pair", unique_exon[:80]), ("paralog_pair", paralog_exon[:80]), ("novel_pair", novel_seq[:80])]:
            f1.write(f"@{nm}/1\n{seq}\n+\n{q[:len(seq)]}\n")
            f2.write(f"@{nm}/2\n{seq}\n+\n{q[:len(seq)]}\n")

    stream = LocalFastqStream("test_run", fq1, fq2)
    interleaved = tmp_path / "interleaved.fq"
    with interleaved.open("wb") as out_fh:
        spots = stream.pump(10, out_fh)
    assert spots == 3
    inter_text = interleaved.read_text()
    assert "@test_run.1/1\n" in inter_text and "@test_run.1/2\n" in inter_text

    out1 = tmp_path / "kept_1.fq"
    out2 = tmp_path / "kept_2.fq"
    stats_json = tmp_path / "sieve_stats.json"
    with interleaved.open("rb") as in_fh:
        subprocess.run(
            [str(glcore), "sieve", "-1", str(out1), "-2", str(out2), "-a", str(act_bin), "--stats", str(stats_json), "--min-hits", "2", "-t", "2"],
            stdin=in_fh,
            check=True,
        )
    st = json.loads(stats_json.read_text())
    assert st["pairs_in"] == 3
    assert st["pairs_dropped_locus"] == 1  # unique saturated pair dropped
    assert st["pairs_out"] == 2            # paralog + novel pairs preserved!
    assert st.get("orphans", 0) == 0       # /1 and /2 headers matched properly


def test_orf_scout_native_c_and_locus_zoom_and_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    monkeypatch.delenv("ORF_SCOUT_LIB", raising=False)
    build_orf_scout(force=True)

    # Build a synthetic contig with an unannotated, high-entropy >= 450 nt intronless ORF
    import random

    rng = random.Random(42)
    sense_codons = [
        a + b + c
        for a in "ACGT"
        for b in "ACGT"
        for c in "ACGT"
        if a + b + c not in ("TAA", "TAG", "TGA")
    ]
    orf_seq = "ATG" + "".join(rng.choice(sense_codons) for _ in range(180)) + "TAA"
    flank = "".join(rng.choice("ACGT") for _ in range(250))
    contig_seq = flank + orf_seq + flank
    fa = tmp_path / "orf_test.fa"
    fa.write_text(f">chrTest\n{contig_seq}\n")
    genome = Genome(fa)
    cands = scan_orphan_orfs(genome, excluded_by_contig={}, min_orf_bp=450)
    assert len(cands) >= 1
    assert any(c.contig == "chrTest" and (c.end - c.start) >= 450 for c in cands)
    wins = orphan_orf_delta_windows(cands, genome.lengths, pad_bp=500, merge_gap=500)
    assert len(wins) >= 1 and wins[0][0] == "chrTest"

    # Micro-exon scan inside an intron (AG ... GT with polypyrimidine tract)
    micro = "GATGGCAAT"  # 9 nt, in-frame without stops
    intron = "T" * 18 + "AG" + micro + "GT" + "A" * 20
    hit = scan_junction_gap_micro_exon(intron, micro, allow_mismatches=0)
    assert hit is not None and hit.exon_seq == micro

    # Single-exon paralog/retrocopy vs true functional single-exon receptor
    retro = SingleExonCandidate("ret1", 800, False, 4, 0.1, True, 0.7, 0.85)
    or_gene = SingleExonCandidate("or1", 930, False, 0, 0.0, False, 0.88, 0.72)
    assert classify_single_exon_candidate(retro) == "processed_pseudogene"
    assert classify_single_exon_candidate(or_gene) == "functional_single_exon_gene"

    # Good-Turing UCB Judge saturation
    judge = GoodTuringUCBJudge(p0_stop=0.05)
    judge.register_library("RUN1", "B1")
    j_list = [WaveJunction(("chr1", 100 + i * 10, 200 + i * 10, "+"), 5, "GT-AG", "unsettled") for i in range(10)]
    rep = judge.ingest_wave_counts("RUN1", 10000, j_list, spots=5000)
    assert rep["unsettled_p0"] == 0.0  # all junctions observed count >= 2


def test_runner_manifest_mode_and_discover_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "rna.tsv"
    manifest.write_text(
        "run_accession\tread_type\torganism\n"
        "SRR123456\tshort_paired\tGallus gallus\n"
        "ERR999001\tpacbio_isoseq\tGallus gallus\n"
    )
    short_r, long_r, orgs = _resolve_rna_manifest(manifest)
    assert short_r == ["SRR123456"]
    assert long_r == ["ERR999001"]
    assert orgs["SRR123456"] == "Gallus gallus"

    # Auto mode gracefully falls back to fast_evidence when weights/GPU absent
    monkeypatch.setenv("TIBERIUS_WEIGHTS_DIR", str(tmp_path / "nonexistent_weights"))
    monkeypatch.setattr("focaloop.runners_geneloop.gpu_available", lambda: False)
    monkeypatch.setattr("focaloop.runners_geneloop.discover_tiberius_weights", lambda *a, **k: None)
    mode_info = resolve_execution_mode("auto", "vertebrates")
    assert mode_info["effective_mode"] == "fast_evidence"
    assert mode_info["skip_tiberius"] is True

    # Explicit gpu_tiberius raises clear error when no GPU
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        resolve_execution_mode("gpu_tiberius", "vertebrates")

    # Weights discovery picks up directory with model_config.json / weights.h5
    wdir = tmp_path / "weights" / "vertebrates_weights"
    wdir.mkdir(parents=True)
    (wdir / "model_config.json").write_text('{"model": "vertebrates"}')
    assert discover_tiberius_weights("vertebrates", explicit_weights=wdir) == wdir.resolve()


def test_run_geneloop_galaxy_fixture_outputs_and_no_sieve_wording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure portable native compilation works in tmp_path
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    monkeypatch.delenv("GLCORE_BIN", raising=False)
    monkeypatch.delenv("ORF_SCOUT_LIB", raising=False)

    # Lightweight GateRun stub when bioinformatics binaries (miniprot/hisat2/samtools) aren't installed locally
    have_bins = all(shutil_which(b) for b in ("miniprot", "hisat2", "samtools"))
    if not have_bins:
        import focaloop.runners_geneloop as rg
        from focaloop.geneloop.gffio import Transcript, write_gff3, write_gtf

        class _StubGateRun:
            def __init__(self, gate_id, *, root=None, loop=None, synth=None, gate=None):
                self.c_tag = "C1"
                self.dir = (Path(root) if root else tmp_path) / gate_id
                self.dir.mkdir(parents=True, exist_ok=True)
                self.genome = Genome(gate["fasta_path"])
                self.meta = {"n_contigs": len(self.genome.lengths), "genome_bp": sum(self.genome.lengths.values()), "organelle_contigs": []}
                self._local_streams = {}
                self._local_bams = []
                self._skip_tiberius = False
                self._enable_zoom = True

            def seed(self, run_tiberius=False):
                return {"n_loci": 2, "LOCKED_HIGH_CONF": 1, "UNSETTLED_LOCI": 1}

            def close(self):
                return None

            def arm_c(self, seed_summary=None):
                ex1 = [(100, 300), (600, 900), (1100, 1300)]
                txs = [
                    Transcript(
                        tid="g1.t1",
                        contig="chr1",
                        strand="+",
                        exons=ex1,
                        cds=ex1,
                        gid="g1",
                        source="GeneLoop",
                        biotype="protein_coding",
                    ),
                    Transcript(
                        tid="g2.t1",
                        contig="chr2",
                        strand="-",
                        exons=[(120, 320), (520, 720)],
                        cds=[],
                        gid="g2",
                        source="GeneLoop",
                        biotype="lncRNA",
                    ),
                ]
                arm_dir = self.dir / f"arm{self.c_tag}"
                arm_dir.mkdir(parents=True, exist_ok=True)
                gff = arm_dir / f"arm{self.c_tag}.gff3"
                write_gff3(txs, gff, source="GeneLoop")
                return {
                    "final_gff": str(gff),
                    "waves": 1,
                    "stop_reason": "P0_CONVERGED",
                    "spots_streamed_all": 12,
                    "total_spots_all": 12,
                    "pct_archive_streamed": 100.0,
                    "sieve": {"pairs_in": 12, "pairs_dropped_locus": 4, "pairs_dropped_bait": 0, "bypass_pct": 33.3},
                    "judge_final": {"global_unsettled_p0": 0.0},
                    "synth_stats": {
                        "n_structural_ncrna_genes": 0,
                        "n_microexon_models": 0,
                        "n_single_exon_genes": 0,
                        "retro_rejected": 0,
                    },
                }

        monkeypatch.setattr(rg, "GateRun", _StubGateRun)

    out_gff3 = tmp_path / "annotation.gff3"
    out_prot = tmp_path / "proteins.faa"
    out_gtf = tmp_path / "transcripts.gtf"
    out_sum = tmp_path / "locus_summary.tsv"
    out_rep = tmp_path / "focaloop_report.html"
    state_dir = tmp_path / "state_dir"
    state_arc = tmp_path / "state.focaloop.tar.zst"

    res = run_geneloop_galaxy(
        genome_fasta=TEST_DATA / "genome.fa",
        out_gff3=out_gff3,
        out_proteins=out_prot,
        out_gtf=out_gtf,
        out_summary=out_sum,
        out_report=out_rep,
        state_dir=state_dir,
        state_archive=state_arc,
        proteins_fasta=TEST_DATA / "proteins.faa",
        clade="vertebrates",
        mode="fast_evidence",
        short_reads_1=[TEST_DATA / "reads_1.fastq.gz"],
        short_reads_2=[TEST_DATA / "reads_2.fastq.gz"],
        short_read_ids=["test_pe"],
        structural_ncrna="off",
        threads=2,
    )

    assert res["effective_mode"] == "fast_evidence"
    for p in (out_gff3, out_prot, out_gtf, out_sum, out_rep, state_arc):
        assert p.is_file() and p.stat().st_size > 0
    assert "##gff-version 3" in out_gff3.read_text()
    assert "coding_genes" in out_sum.read_text()
    html_text = out_rep.read_text()
    assert "GeneLoop" in html_text
    assert "sieve" not in html_text.lower()
    assert "sieve" not in out_sum.read_text().lower()


def shutil_which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)


def test_galaxy_xml_udt_test_data_and_strict_no_sieve() -> None:
    # 1. Blueprint UDT and simple XML lint cleanly
    udt_path = REPO_ROOT / "blueprints" / "geneloop" / "galaxy_udt.yml"
    bp_xml = REPO_ROOT / "blueprints" / "geneloop" / "galaxy_tool.xml"
    assert load_and_lint_user_tool(udt_path) == []
    assert lint_tool_xml(bp_xml) == []

    # 2. Main tools/focaloop/geneloop.xml structural & IUC checks
    tool_xml = REPO_ROOT / "tools" / "focaloop" / "geneloop.xml"
    root = ET.parse(tool_xml).getroot()
    assert root.get("profile") == "@PROFILE@"
    cmd = root.find("command")
    assert cmd is not None and cmd.get("detect_errors") == "aggressive"
    assert "@THREADS@" in (cmd.text or "")
    tests_el = root.find("tests")
    assert tests_el is not None
    tests = tests_el.findall("test")
    assert len(tests) >= 2
    for t in tests:
        assert t.get("expect_num_outputs") == "6"

    # 3. Test data files exist and stay well under 1 MB total
    total_bytes = sum(p.stat().st_size for p in TEST_DATA.iterdir() if p.is_file())
    assert total_bytes < 1_000_000
    for fname in ("genome.fa", "proteins.faa", "reads_1.fastq.gz", "reads_2.fastq.gz"):
        assert (TEST_DATA / fname).is_file()

    # 4. Strict wording rule: ZERO occurrences of "sieve" in any user-facing Galaxy XML, UDT, blueprint, or report helper
    user_facing_files = [
        tool_xml,
        bp_xml,
        udt_path,
        REPO_ROOT / "blueprints" / "geneloop" / "blueprint.focaloop.yml",
    ]
    for uf in user_facing_files:
        filtered = "\n".join(
            ln for ln in uf.read_text(encoding="utf-8").splitlines()
            if '<not_has_text text="sieve"/>' not in ln
        ).lower()
        assert "sieve" not in filtered, f"Forbidden word 'sieve' found in {uf}"
