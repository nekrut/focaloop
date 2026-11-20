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

    # Every input parameter name/argument appears in command or configfiles (parameter parity)
    cf_text = "\n".join(cf.text or "" for cf in root.findall("configfiles/configfile"))
    template_text = cmd_text + "\n" + cf_text
    inputs_el = root.find("inputs")
    assert inputs_el is not None
    for p in inputs_el.iter("param"):
        pname = p.get("name") or (p.get("argument") or "").lstrip("-").replace("-", "_")
        assert pname, f"Param missing name/argument: {ET.tostring(p, encoding='unicode')}"
        assert re.search(r"\b" + re.escape(pname) + r"\b", template_text), f"Orphaned param {pname!r}"

    # Inline condition-group + accession builder present under manifest_insdc
    cf_names = {cf.get("name") for cf in root.findall("configfiles/configfile")}
    assert "inline_manifest_tsv" in cf_names
    assert "$inline_manifest_tsv" in cmd_text
    assert "$input_source.insdc_input.condition_groups" in cf_text

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


def test_panexpr_html_report_tournament_keys_contrast_and_logan_layout(tmp_path: Path) -> None:
    from focaloop.reports import render_panexpr_report

    de_df = pd.DataFrame(
        [
            {
                "gene_id": "G1",
                "gene_name": "entFM",
                "ortholog_bridge": "",
                "baseMean": 410.2,
                "log2FoldChange": 1.37,
                "lfcSE_total": 0.24,
                "pvalue": 0.0002,
                "padj": 0.004,
                "placement_var_frac": 0.01,
                "zoomed": False,
                "status": "sig",
            }
        ]
    )
    sample_rows = [
        {"sample_id": "s1", "condition": "glucose", "layout": "PE", "bytes_read_mb": 45.2, "bytes_pct": 3.5, "mapped_reads": 900000}
    ]
    cat_summary = {"primary_assembly": "primary", "assemblies": ["primary"], "n_groups": 5000, "n_targets": 5000}

    # Case 1: Tournament mode using exact KernelScore.as_dict() keys
    html_tourn = tmp_path / "tourn.html"
    render_panexpr_report(
        html_tourn,
        contrast_str="condition:glucose:control",
        winner="bwa_spa",
        scoreboard=[
            {
                "kernel": "bwa_spa",
                "optimality_score": 0.912,
                "overall_synthetic_recall_pct": 98.4,
                "mate_tie_symmetry": 0.996,
                "real_mapped_pct": 94.3,
                "reads_per_sec": 31200.0,
            },
            {
                "kernel": "strobealign",
                "optimality_score": 0.101,
                "overall_synthetic_recall_pct": 0.0,
                "mate_tie_symmetry": 1.0,
                "real_mapped_pct": 0.0,
                "reads_per_sec": 85000.0,
            },
        ],
        per_sample_rows=sample_rows,
        catalog_summary=cat_summary,
        loop_summary={
            "source_mode": "sra_odp",
            "tournament_reads": 25000,
            "stop_reason": "converged_tech_var_below_bio_var",
            "iterations": 3,
            "wall_seconds": 120.5,
            "assembly_prior": {"primary": 1.0},
            "telemetry": [
                {"iteration": 1, "bytes_pct": 2.0, "max_delta_lfc": 0.4, "median_v_tech_ratio": 0.6, "n_sig": 12},
                {"iteration": 2, "bytes_pct": 5.0, "max_delta_lfc": 0.04, "median_v_tech_ratio": 0.18, "n_sig": 20},
            ],
        },
        de_df=de_df,
    )
    txt_t = html_tourn.read_text(encoding="utf-8")
    assert ".hero code {" in txt_t
    assert "0.912" in txt_t and "98.4%" in txt_t and "0.996" in txt_t and "94.3%" in txt_t and "31,200" in txt_t
    assert "Converged (low technical noise)" in txt_t
    assert "primary: 100.0%" in txt_t
    assert "{'primary': 1.0}" not in txt_t

    # Case 2: Logan unitigs mode omits empty tournament table & iterative telemetry placeholder
    html_logan = tmp_path / "logan.html"
    render_panexpr_report(
        html_logan,
        contrast_str="condition:glucose:control",
        winner="logan_bwa_spa",
        scoreboard=[],
        per_sample_rows=sample_rows,
        catalog_summary=cat_summary,
        loop_summary={
            "source_mode": "logan_unitigs",
            "wire_reduction_x": 28.9,
            "stop_reason": "LOGAN_UNITIGS_COMPLETE",
            "iterations": 1,
            "wall_seconds": 62.5,
            "assembly_prior": {"primary": 1.0},
            "telemetry": [],
        },
        de_df=de_df,
    )
    txt_l = html_logan.read_text(encoding="utf-8")
    assert "Logan unitigs complete" in txt_l
    assert "28.9x smaller" in txt_l
    assert "No iterative telemetry points recorded." not in txt_l
    assert "Stage 1 — Streaming Engine Selection" in txt_l
    assert "primary: 100.0%" in txt_l
    assert "{'primary': 1.0}" not in txt_l
    assert "sieve" not in txt_l.lower()


def test_expressionloop_large_cohort_preflight_hardening(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify all 6 pre-flight hardening fixes for 500-1,000-sample bacterial cohorts."""
    import subprocess
    import time
    import numpy as np
    from focaloop import cvmfs_annot, logan
    from focaloop.galaxy_runners import _parse_subset_spec, _prune_constant_design_terms, run_warm_engine_query
    from focaloop.panexpr.logan_expr import ensure_expr_logan_bins
    from focaloop.panexpr.solver import run_pydeseq2

    # 1. prep_expr_unitigs tiles >2.5 kb unitigs with exact (L - 30) midpoint-bp conservation
    prep_bin, _ = ensure_expr_logan_bins()
    short_seq = "ACGT" * 25
    long_seq = "".join("ACGT"[(i * 7 + (i // 13)) & 3] for i in range(34000))
    res = subprocess.run(
        [str(prep_bin), "1.5"],
        input=f">0 ka:f:8.25\n{short_seq}\n>1 ka:f:14.5\n{long_seq}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    recs: list[tuple[str, str]] = []
    cur_h = None
    for ln in res.stdout.splitlines():
        if ln.startswith(">"):
            cur_h = ln[1:]
        elif cur_h is not None:
            recs.append((cur_h, ln.strip()))
            cur_h = None
    assert recs[0] == ("0_825", short_seq)
    tiles = recs[1:]
    assert len(tiles) >= 16
    assert all(h.endswith("_1450") and "c" in h for h, _ in tiles)
    assert all(500 < len(s) <= 2500 for _, s in tiles)
    # Exact sum of expr_sam_ec_v2 midpoint weights: sum(len(t) - 30) == len(long_seq) - 30
    assert sum(len(s) - 30 for _, s in tiles) == len(long_seq) - 30

    # 2. Logan probe rejects 55-byte empty zstd stubs (<= 100 B)
    class _FakeResp:
        def __init__(self, code: int, cl: int):
            self.status_code = code
            self.headers = {"Content-Length": str(cl)}

    class _FakeSess:
        def head(self, url: str, timeout: float = 8.0, allow_redirects: bool = True):
            if "SRR100001" in url:
                return _FakeResp(200, 55)
            if "SRR100002" in url:
                return _FakeResp(200, 1250000)
            return _FakeResp(404, 0)

    monkeypatch.setattr(logan, "_ipv4_session", lambda: _FakeSess())
    probe = logan.probe_logan_runs(["SRR100001", "SRR100002", "SRR100003"])
    assert probe["logan_runs"] == ["SRR100002"]
    assert probe["stub_runs"] == ["SRR100001"]
    assert set(probe["fallback_runs"]) == {"SRR100001", "SRR100003"}
    assert probe["probes"]["SRR100001"]["available"] is False
    assert probe["probes"]["SRR100001"]["is_stub"] is True

    # 3. PyDESeq2 poscounts auto-switch when every gene has >=1 zero + drops 0-count sample cleanly
    rng = np.random.default_rng(7)
    n_s, n_g = 24, 30
    mat = rng.poisson(lam=45.0, size=(n_s, n_g)).astype(float)
    # Force at least one zero in EVERY gene (which triggers slow Powell fallback under fit_type='ratio')
    for gi in range(n_g):
        mat[gi % n_s, gi] = 0.0
    # Make first 12 samples treated with 4x boost on first 5 genes; inject 1 completely empty sample
    mat[:12, :5] *= 4.0
    mat[11, :] = 0.0
    snames = [f"s{i}" for i in range(n_s)]
    gnames = [f"g{j}" for j in range(n_g)]
    counts_df = pd.DataFrame(mat, index=snames, columns=gnames)
    meta_df = pd.DataFrame({"condition": ["treated"] * 12 + ["control"] * 12}, index=snames)
    t0 = time.time()
    de_res, disp, sf = run_pydeseq2(counts_df, meta_df, ["condition", "treated", "control"])
    assert (time.time() - t0) < 15.0
    assert not de_res.empty
    assert float(de_res.loc["g0", "log2FoldChange"]) > 1.0
    assert sf.reindex(snames).notna().all()

    # 4. CVMFS species typo fix must not double the trailing 'a' on valid Pseudomonas aeruginosa names
    assert cvmfs_annot._clean_species_name("Pseudomonas aeruginos PAO1") == "Pseudomonas aeruginosa"
    assert cvmfs_annot._clean_species_name("Pseudomonas aeruginosa UCBPP-PA14") == "Pseudomonas aeruginosa"
    assert cvmfs_annot._clean_species_name("Pseudomonas aeruginosa") == "Pseudomonas aeruginosa"

    # 5. Focaloop Query --subset parsing & constant-covariate pruning on a multi-plate state
    clauses = _parse_subset_spec("plate_id=P1 timepoint=60m,120m")
    assert clauses == [("plate_id", ["P1"]), ("timepoint", ["60m", "120m"])]
    sub_meta = pd.DataFrame({"plate_id": ["P1"] * 4, "condition": ["cip", "cip", "dmso", "dmso"]})
    assert _prune_constant_design_terms("~plate_id + condition", sub_meta, "condition") == "~condition"

    # End-to-end ExpressionLoop state -> Warm Query with --subset
    state_dir = tmp_path / "st_multi"
    manifest_p = tmp_path / "multi_plate.tsv"
    c1 = (TEST_DATA / "ctrl1.fastq.gz").resolve()
    c2 = (TEST_DATA / "ctrl2.fastq.gz").resolve()
    t1 = (TEST_DATA / "trt1.fastq.gz").resolve()
    t2 = (TEST_DATA / "trt2.fastq.gz").resolve()
    manifest_p.write_text(
        "sample_id\tcondition\tplate_id\trun_or_url\n"
        f"p1_c1\tcontrol\tP1\t{c1}\n"
        f"p1_c2\tcontrol\tP1\t{c2}\n"
        f"p1_t1\ttreated\tP1\t{t1}\n"
        f"p1_t2\ttreated\tP1\t{t2}\n"
        f"p2_c1\tcontrol\tP2\t{c1}\n"
        f"p2_c2\tcontrol\tP2\t{c2}\n"
        f"p2_t1\ttreated\tP2\t{t1}\n"
        f"p2_t2\ttreated\tP2\t{t2}\n",
        encoding="utf-8",
    )
    meta_out = run_expressionloop_galaxy(
        ref_fasta=TEST_DATA / "ref_primary.fa",
        ref_gtf=TEST_DATA / "ref_primary.gtf",
        manifest_path=manifest_p,
        contrast="condition:treated:control",
        design="~plate_id + condition",
        source_mode="local_bytes",
        max_byte_fraction=1.0,
        kernel_names=["mappy"],
        n_tournament_reads=600,
        out_de=tmp_path / "de.tsv",
        out_report=tmp_path / "rep.html",
        state_dir=state_dir,
        threads=2,
        logan_mode="off",
    )
    q_verdict = tmp_path / "q_sub.json"
    q_res = run_warm_engine_query(
        state_dir=state_dir,
        meta=meta_out,
        contrast="condition:treated:control",
        actor="tester",
        out_table=tmp_path / "q_sub.tsv",
        out_verdict=q_verdict,
        subset="plate_id=P1",
        mode="contrast",
    )
    assert q_res["n_samples_queried"] == 4
    assert q_res["effective_design"] == "~condition"
    assert q_res["subset"] == "plate_id=P1"
    with pytest.raises(ValueError, match="matched 0 samples"):
        run_warm_engine_query(
            state_dir=state_dir,
            meta=meta_out,
            contrast="condition:treated:control",
            actor="tester",
            out_table=tmp_path / "q_empty.tsv",
            out_verdict=tmp_path / "q_empty.json",
            subset="plate_id=NOPE",
            mode="contrast",
        )
