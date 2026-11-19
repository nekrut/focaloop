"""Comprehensive tests for ExpressionLoop (PanExpr v2.4), Galaxy runner, XML wrapper, and strict wording."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from focaloop.ops import run_query
from focaloop.panexpr._fast_c import FastCSieve, _py_clip_adapters_batch, clip_adapters_batch
from focaloop.panexpr.sra_odp import XsraRunStream
from focaloop.panexpr.streamer import open_run_stream
from focaloop.runners_expression import (
    prebuild_expressionloop_native_libs,
    resolve_expression_manifest,
    run_expressionloop_galaxy,
)

REPO = Path(__file__).resolve().parents[1]
EXPR_XML = REPO / "tools" / "focaloop" / "expressionloop.xml"
TEST_DATA = REPO / "tools" / "focaloop" / "test-data" / "expressionloop"


def test_panexpr_v24_c_accelerators_portable_isa_and_adapter_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOCALOOP_PORTABLE_ISA", "1")
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    built = prebuild_expressionloop_native_libs()
    assert len(built) == 7
    assert "panexpr_fast_c" in built
    assert "panexpr_expr_sam_ec_v2" in built

    # Illumina TruSeq adapter read-through clipping: C99 fast path matches pure-Python reference
    adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    insert = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 44 bp >= 31
    tr = str.maketrans("ACGT", "TGCA")
    rc_ins = insert.translate(tr)[::-1]
    r1 = [(insert + adapter[:22], "I" * (len(insert) + 22)), ("ACGT" * 20, "I" * 80)]
    r2 = [(rc_ins + adapter[:22], "I" * (len(rc_ins) + 22)), ("TGCA" * 20, "I" * 80)]

    c1, c2, cn = clip_adapters_batch(r1, r2, min_len=31, threads=2)
    p1, p2, pn = _py_clip_adapters_batch(r1, r2, min_len=31)
    assert cn == pn == 1
    assert c1 == p1
    assert c2 == p2
    assert c1[0][0] == insert

    # FastCSieve in both include and deplete modes
    target = ["AACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTT"]
    housekeeping = ["TTTTTTTTGGGGGGGGCCCCCCCCAAAAAAAATTTTTTTT"]
    reads = [(target[0], "I" * 40), (housekeeping[0], "I" * 40)]
    inc = FastCSieve.from_sequences(target, k=31, mode="include")
    dep = FastCSieve.from_sequences(housekeeping, k=31, mode="deplete", exclude_seqs=target)
    assert inc.filter_reads(reads, min_hits=2, threads=2).tolist() == [0]
    assert dep.filter_reads(reads, min_hits=2, threads=2).tolist() == [0]


def test_expressionloop_se_dual_assembly_and_warm_query(tmp_path: Path) -> None:
    out_de = tmp_path / "de_results.tsv"
    out_report = tmp_path / "report.html"
    state_dir = tmp_path / "state_dir"
    state_archive = tmp_path / "expression_state.focaloop.tar.zst"

    reads = [str(TEST_DATA / f"{s}.fastq.gz") for s in ("ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3")]
    read_ids = ["ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3"]

    meta = run_expressionloop_galaxy(
        manifest_path=TEST_DATA / "manifest_se.tsv",
        ref_fasta=TEST_DATA / "ref_primary.fa",
        ref_gtf=TEST_DATA / "ref_primary.gtf",
        secondary_fasta=TEST_DATA / "ref_secondary.fa",
        secondary_gtf=TEST_DATA / "ref_secondary.gtf",
        primary_label="v3",
        secondary_label="v2",
        reads=reads,
        read_ids=read_ids,
        contrast="condition:treated:control",
        organism_mode="eukaryotic",
        source_mode="local_bytes",
        max_byte_fraction=1.0,
        enable_zoom=True,
        enable_abundant_filter=True,
        filter_mode="include",
        threads=2,
        kernel_names=["mappy"],
        n_tournament_reads=600,
        state_dir=state_dir,
        state_archive=state_archive,
        out_de=out_de,
        out_report=out_report,
    )
    assert meta["engine"] == "panexpr"
    assert out_de.is_file()
    assert out_report.is_file()
    assert state_archive.is_file()

    de_df = pd.read_csv(out_de, sep="\t")
    g0 = de_df[de_df["gene_id"] == "GENE_00"].iloc[0]
    assert g0["log2FoldChange"] > 2.5
    assert g0["padj"] < 1e-3
    assert "v2:LEGACY_001458" in str(g0["ortholog_bridge"])

    html = out_report.read_text(encoding="utf-8")
    assert "Focaloop ExpressionLoop" in html
    assert "sieve" not in html.lower()

    # Warm query compatibility with Focaloop Query (zero new bytes, sockets blocked)
    q_table = tmp_path / "q_table.tsv"
    q_verdict = tmp_path / "q_verdict.json"
    q_report = tmp_path / "q_report.html"
    q_out = run_query(
        state_archive,
        contrast="condition:treated:control",
        actor="human",
        out_table=q_table,
        out_verdict=q_verdict,
        out_report=q_report,
    )
    assert q_out["new_bytes"] == 0
    assert q_out["network_sockets_blocked"] is True
    assert q_table.is_file() and q_report.is_file()


def test_expressionloop_paired_collection_batch_and_transcript_tpm(tmp_path: Path) -> None:
    samples = ["pe_ctrl1", "pe_ctrl2", "pe_ctrl3", "pe_trt1", "pe_trt2", "pe_trt3"]
    r1s = [str(TEST_DATA / f"{s}_R1.fastq.gz") for s in samples]
    r2s = [str(TEST_DATA / f"{s}_R2.fastq.gz") for s in samples]

    out_de = tmp_path / "de_pe.tsv"
    out_tx = tmp_path / "tx_pe.tsv"
    out_report = tmp_path / "report_pe.html"
    state_dir = tmp_path / "state_pe"
    state_archive = tmp_path / "state_pe.focaloop.tar.zst"

    run_expressionloop_galaxy(
        manifest_path=TEST_DATA / "manifest_pe.tsv",
        ref_fasta=TEST_DATA / "ref_primary.fa",
        ref_gtf=TEST_DATA / "ref_primary.gtf",
        reads_r1=r1s,
        reads_r2=r2s,
        read_ids=samples,
        contrast="condition:treated:control",
        design="~batch + condition",
        followup_contrasts="condition:control:treated",
        organism_mode="auto",
        source_mode="auto",
        max_byte_fraction=1.0,
        enable_zoom=False,
        threads=2,
        kernel_names=["mappy"],
        n_tournament_reads=500,
        state_dir=state_dir,
        state_archive=state_archive,
        out_de=out_de,
        out_report=out_report,
        out_transcripts=out_tx,
    )

    de_df = pd.read_csv(out_de, sep="\t")
    g0 = de_df[de_df["gene_id"] == "GENE_00"].iloc[0]
    assert g0["log2FoldChange"] < -2.5
    assert g0["padj"] < 1e-3

    tx_df = pd.read_csv(out_tx, sep="\t")
    assert {"transcript_tpm", "isoform_fraction_psi", "effective_length"}.issubset(tx_df.columns)
    assert "primary|GENE_00.t1" in set(tx_df["target_id"])
    assert "primary|GENE_00.t2" in set(tx_df["target_id"])
    assert (state_dir / "panexpr_state" / "gene_de_followup_control_vs_treated.parquet").is_file()


def test_expressionloop_uncompressed_fastq_and_prokaryotic_zoom(tmp_path: Path) -> None:
    # Decompress SE fastq.gz files to plain .fastq to verify plain-FASTQ streaming support
    plain_paths = []
    read_ids = ["ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3"]
    for sid in read_ids:
        dst = tmp_path / f"{sid}.fastq"
        with gzip.open(TEST_DATA / f"{sid}.fastq.gz", "rb") as fi:
            dst.write_bytes(fi.read())
        plain_paths.append(str(dst))

    out_de = tmp_path / "de_prok.tsv"
    out_report = tmp_path / "report_prok.html"
    run_expressionloop_galaxy(
        manifest_path=TEST_DATA / "manifest_se.tsv",
        ref_fasta=TEST_DATA / "ref_primary.fa",
        ref_gtf=TEST_DATA / "ref_prok.gtf",
        reads=plain_paths,
        read_ids=read_ids,
        contrast="condition:treated:control",
        organism_mode="prokaryotic",
        source_mode="local_bytes",
        max_byte_fraction=0.90,
        v_tech_target=0.35,
        enable_zoom=True,
        enable_abundant_filter=True,
        filter_mode="include",
        threads=2,
        kernel_names=["mappy"],
        n_tournament_reads=600,
        state_dir=tmp_path / "state_prok",
        out_de=out_de,
        out_report=out_report,
    )
    de_df = pd.read_csv(out_de, sep="\t")
    assert not de_df.empty
    assert "GENE_00" in set(de_df["gene_id"])


def test_expressionloop_insdc_ena_and_xsra_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import focaloop.panexpr.manifest as man_mod
    import focaloop.panexpr.sra_odp as odp_mod

    fake_tsv = (
        "run_accession\tsample_alias\tsample_title\tlibrary_layout\tlibrary_strategy\t"
        "read_count\tfastq_bytes\tfastq_ftp\tfastq_md5\n"
        "SRR9990001\ts1\ts1\tPAIRED\tRNA-Seq\t12000\t1200000;1200000\t"
        "ftp.sra.ebi.ac.uk/vol1/fastq/SRR999/001/SRR9990001/SRR9990001_1.fastq.gz;"
        "ftp.sra.ebi.ac.uk/vol1/fastq/SRR999/001/SRR9990001/SRR9990001_2.fastq.gz\t"
        "aaa;bbb\n"
    )
    monkeypatch.setattr(man_mod, "fetch_ena_filereport", lambda acc, timeout=60.0: fake_tsv)

    manifest_tsv = tmp_path / "insdc.tsv"
    manifest_tsv.write_text("sample_id\tcondition\trun_or_url\ns1\tcontrol\tSRR9990001\n", encoding="utf-8")

    study, meta_df, all_local = resolve_expression_manifest(manifest_tsv)
    assert all_local is False
    assert len(study.runs) == 1
    run = study.runs[0]
    assert run.run_id == "SRR9990001"
    assert run.layout == "PE"
    assert run.read_count == 12000
    assert run.files[0].url.startswith("https://ftp.sra.ebi.ac.uk/")
    assert len(meta_df) == 1

    # Tier routing: auto selects XsraRunStream when xsra + ODP available, else ENA HTTP 206 RunStream
    monkeypatch.setattr(odp_mod, "xsra_runnable", lambda: True)
    monkeypatch.setattr(odp_mod, "odp_available", lambda acc, timeout=10.0: True)
    st_xsra = open_run_stream(run, "auto", tmp_path / "cache")
    assert isinstance(st_xsra, XsraRunStream)

    monkeypatch.setattr(odp_mod, "xsra_runnable", lambda: False)
    st_http = open_run_stream(run, "auto", tmp_path / "cache")
    assert st_http.__class__.__name__ == "RunStream"


def test_expressionloop_xml_structure_help_testdata_and_no_sieve_wording() -> None:
    assert EXPR_XML.is_file()
    raw_xml = EXPR_XML.read_text(encoding="utf-8")
    assert "siev" not in raw_xml.lower(), "Forbidden wording found in expressionloop.xml"

    root = ET.parse(EXPR_XML).getroot()
    assert root.tag == "tool"
    assert root.get("id") == "focaloop_expressionloop"
    assert root.get("version") == "@TOOL_VERSION@+galaxy@VERSION_SUFFIX@"
    assert root.get("profile") == "@PROFILE@"

    cmd_el = root.find("command")
    assert cmd_el is not None and cmd_el.get("detect_errors") == "aggressive"
    cmd_text = cmd_el.text or ""
    assert "@THREADS@" in cmd_text

    # Strict XML element ordering (before macro expansion, macros/expand at expected spots)
    tags = [c.tag for c in root]
    expected_prefix = ["description", "macros"]
    assert tags[:2] == expected_prefix
    assert tags.index("command") < tags.index("inputs") < tags.index("outputs") < tags.index("tests") < tags.index("help")

    # Every input parameter name/argument appears in command (parameter parity)
    inputs_el = root.find("inputs")
    assert inputs_el is not None
    for p in inputs_el.iter("param"):
        pname = p.get("name") or (p.get("argument") or "").lstrip("-").replace("-", "_")
        assert pname, f"Param missing name/argument: {ET.tostring(p, encoding='unicode')}"
        assert re.search(r"\b" + re.escape(pname) + r"\b", cmd_text), f"Orphaned param {pname!r}"

    # Every test declares expect_num_outputs and references < 1 MB test-data
    tests_el = root.find("tests")
    assert tests_el is not None and len(list(tests_el)) >= 3
    for t in tests_el:
        assert t.get("expect_num_outputs") is not None

    total_td_bytes = sum(f.stat().st_size for f in TEST_DATA.rglob("*") if f.is_file())
    assert 0 < total_td_bytes < 1_000_000, f"Test-data too large: {total_td_bytes} bytes"

    # Help section markdown quality checks
    help_el = root.find("help")
    assert help_el is not None and help_el.get("format") == "markdown"
    help_txt = help_el.text or ""
    for required_heading in (
        "What this tool does",
        "How it works",
        "Choosing an input mode",
        "Inputs and manifest format",
        "Key parameters",
        "Outputs",
        "Tips and troubleshooting",
    ):
        assert required_heading in help_txt

    # Runner code/docstrings must also avoid forbidden wording in user-facing strings
    runner_txt = (REPO / "focaloop" / "runners_expression.py").read_text(encoding="utf-8")
    for line in runner_txt.splitlines():
        if "abundant_sieve" in line or "sieve_mode" in line:
            # Internal PanExpr LoopConfig kwargs only
            continue
        assert "siev" not in line.lower(), f"Forbidden wording in runners_expression.py: {line}"
