"""Tests for VariationLoop (MalariaVar + Diploid Human VarLoop), C core, Galaxy XML, and runner."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from focaloop.archive import is_state_archive, unpack_state
from focaloop.malariavar import caller as C
from focaloop.malariavar import engine as E
from focaloop.malariavar import genome as G
from focaloop.malariavar import omnitig_zoom as OZ
from focaloop.malariavar import varloop_giab as VG
from focaloop.malariavar.native._loader import ensure_shared_lib
from focaloop.runners_variation import run_variationloop_galaxy

ROOT = Path(__file__).resolve().parents[1]
TOOL_XML = ROOT / "tools" / "focaloop" / "variationloop.xml"
TEST_DATA_DIR = ROOT / "tools" / "focaloop" / "test-data" / "variationloop"


def _sam_rec(
    qname: str | int,
    flag: int,
    rname: str,
    pos1: int,
    mapq: int,
    cigar: str,
    seq: str,
    qual: str | None = None,
    nm: int = 0,
    pnext: int | None = None,
    tlen: int = 0,
) -> str:
    qual = qual or ("F" * len(seq))
    pnext = pos1 if pnext is None else pnext
    return f"{qname}\t{flag}\t{rname}\t{pos1}\t{mapq}\t{cigar}\t=\t{pnext}\t{tlen}\t{seq}\t{qual}\tNM:i:{nm}\n"


def _make_synth_genome(tmp_path: Path, n_contigs: int = 40, ctg0_len: int = 21_000_000, other_len: int = 3000) -> tuple[G.Genome, Path]:
    rng = np.random.default_rng(42)
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    fa = tmp_path / "synth.fa"
    with open(fa, "w", encoding="utf-8") as fh:
        for c in range(n_contigs):
            ln = ctg0_len if c == 0 else other_len
            seq = bases[rng.integers(0, 4, size=ln)].tobytes().decode("ascii")
            fh.write(f">chr_s{c}\n{seq}\n")
    return G.from_fasta(fa), fa


def test_varloop_core_portable_isa_compile_and_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOCALOOP_PORTABLE_ISA", "1")
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(tmp_path / "native"))
    monkeypatch.delenv("FOCALOOP_NATIVE_CACHE", raising=False)
    lib_path = ensure_shared_lib("varloop_core", ["varloop_core.c"], use_openmp=False)
    assert lib_path.is_file()
    lib = VG._lib()
    for sym in ("vl_kset_build", "vl_fastq_sieve_paired", "vl_acc_create", "vl_score_reads_haplotypes"):
        assert hasattr(lib, sym)


def test_62bit_dedup_no_wrap_and_order_mode(tmp_path: Path) -> None:
    g, _ = _make_synth_genome(tmp_path, n_contigs=40, ctg0_len=21_000_000, other_len=3000)
    pos_low = 2_000_001
    pos_high = pos_low + (1 << 24)
    s_low = g.bases(g.goff("chr_s0", pos_low), g.goff("chr_s0", pos_low) + 120)
    s_high = g.bases(g.goff("chr_s0", pos_high), g.goff("chr_s0", pos_high) + 120)
    s_c5 = g.bases(g.goff("chr_s5", 501), g.goff("chr_s5", 501) + 120)
    s_c37 = g.bases(g.goff("chr_s37", 501), g.goff("chr_s37", 501) + 120)

    sam_lines = "".join(
        [
            _sam_rec(101, 99, "chr_s0", pos_low, 60, "120M", s_low, pnext=pos_low + 230, tlen=350),
            _sam_rec(102, 99, "chr_s0", pos_high, 60, "120M", s_high, pnext=pos_high + 230, tlen=350),
            _sam_rec(103, 99, "chr_s0", pos_high, 60, "120M", s_high, pnext=pos_high + 230, tlen=350),
            _sam_rec(201, 99, "chr_s5", 501, 60, "120M", s_c5, pnext=701, tlen=320),
            _sam_rec(202, 99, "chr_s37", 501, 60, "120M", s_c37, pnext=701, tlen=320),
        ]
    ).encode("ascii")

    with VG.PagedContigAccumulator(g, VG.DiploidParams(mask_end_trim=0, mask_clip_flank=0)) as pacc:
        pacc.feed_bytes(sam_lines)
        summ = pacc.summary()
        assert int(summ["stats"][10]) == 1
        dp_0, _, _ = pacc.query_positions(0, np.array([pos_low + 50, pos_high + 50], dtype=np.int32))
        dp_5, _, _ = pacc.query_positions(5, np.array([550], dtype=np.int32))
        dp_37, _, _ = pacc.query_positions(37, np.array([550], dtype=np.int32))
        assert list(dp_0) == [1, 1]
        assert int(dp_5[0]) == 1 and int(dp_37[0]) == 1

    acc_py = C.Accumulator(g, C.CallParams(mask_end_trim=0, mask_clip_flank=0, dedup=True), ev_cap=1000)
    acc_py.feed(np.frombuffer(sam_lines, dtype=np.uint8), len(sam_lines))
    assert int(acc_py.stats[10]) == 1

    lens = np.array([248_956_422] * 24 + [20_000] * 171, dtype=np.int64)
    cc_chr20 = np.zeros(195, dtype=np.int64)
    cc_chr20[19] = 250_000
    assert E.detect_order_mode(cc_chr20, lens) == "sorted"


def test_diploid_snv_multiallelic_and_omnitig_softclip_rescue(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    left = bases[rng.integers(0, 4, size=3000)].tobytes().decode("ascii")
    mid = bases[rng.integers(0, 4, size=3000)].tobytes().decode("ascii")
    unit19 = "ACGTACGTACGTACGTTGA"
    ctg_seq = left + (unit19 * 2) + mid
    fa = tmp_path / "omni.fa"
    fa.write_text(f">chr_test\n{ctg_seq}\n", encoding="utf-8")
    g = G.from_fasta(fa)

    # SNV genotyper checks (0/1, 1/1, 1/2, spanning del, paralog bleed)
    pos0 = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)
    ref = np.zeros(len(pos0), dtype=np.uint8)
    counts = np.array(
        [
            [18, 17, 0, 0],
            [0, 34, 0, 0],
            [54, 6, 0, 0],
            [0, 16, 18, 0],
            [0, 17, 0, 0],
            [0, 3, 0, 0],
        ],
        dtype=np.uint16,
    )
    delc = np.array([0, 0, 0, 0, 18, 32], dtype=np.uint16)
    cand = dict(pos0=pos0, counts=counts, delc=delc, clipc=np.zeros_like(delc), uncap=np.zeros_like(delc), ref=ref)
    by_pos = {c["pos0"]: c for c in VG.call_diploid_snvs_contig(g, 0, cand, VG.DiploidParams())}
    assert by_pos[100]["gt"] == "0/1"
    assert by_pos[200]["gt"] == "1/1"
    assert 300 not in by_pos
    assert by_pos[400]["gt"] == "1/2"
    assert by_pos[500]["gt"] == "0/1"
    assert 600 not in by_pos

    # Multi-k (k=31, k=55) repeat bridging + soft-clip 24 bp insertion rescue
    ins_pos0 = 1500
    ins24 = "TGCATGCATGCAGTACTGCATGCA"
    alt_chrom = ctg_seq[:ins_pos0] + ins24 + ctg_seq[ins_pos0:]
    sam_lines = []
    for k in range(22):
        st0 = ins_pos0 - 110 + (k * 3)
        sam_lines.append(_sam_rec(500 + k, 99, "chr_test", st0 + 1, 60, "150M", ctg_seq[st0:st0 + 150], pnext=st0 + 301, tlen=450 + k))
    for k in range(8):
        st0 = ins_pos0 - (95 + k * 3)
        m_len = ins_pos0 - st0
        s_len = 150 - m_len
        sam_lines.append(_sam_rec(700 + k, 99, "chr_test", st0 + 1, 60, f"{m_len}M{s_len}S", alt_chrom[st0:st0 + 150], pnext=st0 + 301, tlen=480 + k))
    for k in range(7):
        lead_s = 32 + k * 3
        alt_st = (ins_pos0 + len(ins24)) - lead_s
        m_len = 150 - lead_s
        sam_lines.append(_sam_rec(800 + k, 99, "chr_test", ins_pos0 + 1, 60, f"{lead_s}S{m_len}M", alt_chrom[alt_st:alt_st + 150], pnext=ins_pos0 + 301, tlen=510 + k))

    with VG.PagedContigAccumulator(g, VG.DiploidParams(enable_reassembly=True)) as acc:
        acc.feed_bytes("".join(sam_lines).encode("ascii"))
        calls = VG.call_contig_diploid(g, acc, 0, VG.DiploidParams(enable_reassembly=True))
    assert len(calls) == 1
    assert calls[0]["source"] == "omnitig_asm"
    assert calls[0]["gt"] == "0/1"
    assert len(calls[0]["alt"]) - len(calls[0]["ref"]) == 24

    # Also verify MalariaVar WSAF Omnitig assembler remains intact in omnitig_zoom.py
    assert callable(OZ.assemble_omnitig_wsaf)
    assert callable(OZ.assemble)


def test_c_target_depletion_filter_lockstep(tmp_path: Path) -> None:
    g, _ = _make_synth_genome(tmp_path, n_contigs=2, ctg0_len=20_000, other_len=20_000)
    tab, meta = VG.build_target_kset(g, [("chr_s0", 0, 20_000)], bits=16)
    assert meta["inserted_kmers"] > 15_000

    r1_in = tmp_path / "in_R1.fq.gz"
    r2_in = tmp_path / "in_R2.fq.gz"
    s0_a, s0_b = g.bases(1000, 1150), g.bases(1300, 1450)
    s1_a, s1_b = g.bases(25_000, 25_150), g.bases(25_300, 25_450)
    r1_in.write_bytes(gzip.compress(f"@READ_A 1\n{s0_a}\n+\n{'F'*150}\n@READ_B 1\n{s1_a}\n+\n{'F'*150}\n".encode()))
    r2_in.write_bytes(gzip.compress(f"@READ_A 2\n{s0_b}\n+\n{'F'*150}\n@READ_B 2\n{s1_b}\n+\n{'F'*150}\n".encode()))

    out1, out2 = tmp_path / "out_R1.fq.gz", tmp_path / "out_R2.fq.gz"
    res = VG.extract_sieved_fastq_pair(r1_in, r2_in, out1, out2, kset_tab=tab, stride=5, need=1, decomp_threads=1, comp_threads=1)
    assert res["spots_seen"] == 2 and res["spots_emitted"] == 1
    out_txt = gzip.decompress(out1.read_bytes()).decode()
    assert s0_a in out_txt and s1_a not in out_txt


def test_runner_all_three_modes_e2e(tmp_path: Path) -> None:
    # 1. Mode C: diploid_wgs (paired FASTQ + BED + subsampling)
    out_vcf_c = tmp_path / "c.vcf"
    out_var_c = tmp_path / "c.tsv"
    out_rep_c = tmp_path / "c.html"
    out_arc_c = tmp_path / "c.tar.zst"
    meta_c = run_variationloop_galaxy(
        analysis_mode="diploid_wgs",
        reads_r1=TEST_DATA_DIR / "sample_R1.fastq.gz",
        reads_r2=TEST_DATA_DIR / "sample_R2.fastq.gz",
        sample_name="HG_MINI",
        ref_fasta=TEST_DATA_DIR / "ref_chr20_mini.fa",
        targets_bed=TEST_DATA_DIR / "targets.bed",
        subsample_fraction=0.95,
        threads=2,
        state_dir=tmp_path / "state_c",
        state_archive=out_arc_c,
        out_variants=out_var_c,
        out_vcf=out_vcf_c,
        out_report=out_rep_c,
    )
    assert meta_c["summary"]["n_variants"] == 2
    df_c = pd.read_csv(out_var_c, sep="\t")
    assert set(df_c["kind"]) == {"SNV", "INDEL"}
    assert set(df_c["gt"]) == {"0/1", "1/1"}
    vcf_txt = out_vcf_c.read_text(encoding="utf-8")
    assert "#CHROM\tPOS\tID" in vcf_txt and "HG_MINI" in vcf_txt
    rep_c_txt = out_rep_c.read_text(encoding="utf-8")
    assert "Focaloop VariationLoop" in rep_c_txt
    assert "sieve" not in rep_c_txt.lower()
    assert is_state_archive(out_arc_c)
    un_c = unpack_state(out_arc_c, tmp_path / "un_c")
    assert (un_c / "focaloop_state.json").is_file()

    # 2. Mode A: focused_cram (History CRAM + CRAI + BED)
    out_vcf_a = tmp_path / "a.vcf"
    out_var_a = tmp_path / "a.tsv"
    out_rep_a = tmp_path / "a.html"
    meta_a = run_variationloop_galaxy(
        analysis_mode="focused_cram",
        bam_path=TEST_DATA_DIR / "sample.cram",
        crai_path=TEST_DATA_DIR / "sample.cram.crai",
        sample_name="HG_CRAM",
        ref_fasta=TEST_DATA_DIR / "ref_chr20_mini.fa",
        targets_bed=TEST_DATA_DIR / "targets.bed",
        cram_pad_bp=500,
        threads=2,
        state_dir=tmp_path / "state_a",
        state_archive=tmp_path / "a.tar.zst",
        out_variants=out_var_a,
        out_vcf=out_vcf_a,
        out_report=out_rep_a,
    )
    assert meta_a["summary"]["n_variants"] == 2
    assert "sieve" not in out_rep_a.read_text(encoding="utf-8").lower()

    # 3. Mode B: wgs_stream (manifest + --reads + C 31-mer depletion filter)
    out_vcf_b = tmp_path / "b.vcf"
    out_var_b = tmp_path / "b.tsv"
    out_drm_b = tmp_path / "b_drm.tsv"
    out_rep_b = tmp_path / "b.html"
    meta_b = run_variationloop_galaxy(
        analysis_mode="wgs_stream",
        manifest_path=TEST_DATA_DIR / "manifest_wgs.tsv",
        reads=[str(TEST_DATA_DIR / "sample_R1.fastq.gz"), str(TEST_DATA_DIR / "sample_R2.fastq.gz")],
        read_ids=["sample_R1.fastq.gz", "sample_R2.fastq.gz"],
        ref_fasta=TEST_DATA_DIR / "ref_chr20_mini.fa",
        targets_bed=TEST_DATA_DIR / "targets.bed",
        use_host_depletion=True,
        threads=2,
        state_dir=tmp_path / "state_b",
        state_archive=tmp_path / "b.tar.zst",
        out_variants=out_var_b,
        out_vcf=out_vcf_b,
        out_drm=out_drm_b,
        out_report=out_rep_b,
    )
    assert meta_b["summary"]["n_variants"] == 2
    assert out_drm_b.is_file()
    assert "sieve" not in out_rep_b.read_text(encoding="utf-8").lower()


def test_variationloop_xml_iuc_standards_and_no_sieve() -> None:
    xml_text = TOOL_XML.read_text(encoding="utf-8")
    assert "sieve" not in xml_text.lower()

    for rel_path in (
        ROOT / "blueprints" / "malariavar" / "galaxy_tool.xml",
        ROOT / "blueprints" / "malariavar" / "galaxy_udt.yml",
    ):
        assert "sieve" not in rel_path.read_text(encoding="utf-8").lower()

    root = ET.parse(TOOL_XML).getroot()
    assert root.attrib["id"] == "focaloop_variationloop"
    assert root.attrib["profile"] == "@PROFILE@"
    assert root.find("command").attrib["detect_errors"] == "aggressive"

    # Element ordering per tools-iuc schema
    order = [el.tag for el in root]
    expected_prefix = ["description", "macros", "edam_operations", "version_command", "command", "inputs", "outputs", "tests", "help"]
    positions = [order.index(t) for t in expected_prefix]
    assert positions == sorted(positions), f"Out-of-order XML tags: {order}"

    cmd_text = root.find("command").text or ""
    assert "@THREADS@" in cmd_text
    cf_text = "\n".join(cf.text or "" for cf in root.findall("configfiles/configfile"))
    template_text = cmd_text + "\n" + cf_text

    # Parameter parity: every param in <inputs> appears in <command> or <configfiles>, and no manual sample_name param
    param_names = set()
    for p in root.findall(".//inputs//param"):
        pname = p.attrib.get("name")
        if not pname and "argument" in p.attrib:
            pname = p.attrib["argument"].lstrip("-").replace("-", "_")
        assert pname, f"Param missing name/argument: {ET.tostring(p)}"
        param_names.add(pname)
        assert pname in template_text, f"Orphaned param {pname!r} not referenced in <command> or <configfiles>"
    assert "sample_name" not in param_names, "Manual sample_name box should not appear in unified variationloop.xml"
    src_opts = [o.attrib["value"] for o in root.findall(".//conditional[@name='input_source']/param[@name='source_type']/option")]
    assert src_opts == ["history_paired_list", "history_bam", "public_sra"], f"Unexpected input_source options: {src_opts}"

    # Zero semicolons in user-facing XML text
    for el in root.iter():
        for attr in ("label", "help", "title", "message"):
            assert ";" not in el.attrib.get(attr, ""), f"Semicolon in {el.tag}@{attr}"

    # Test-data files exist and < 1 MB; every <test> has expect_num_outputs and no pipe syntax
    tests_el = root.find("tests")
    test_els = tests_el.findall("test")
    assert len(test_els) >= 3
    total_td_bytes = 0
    for t in test_els:
        assert "expect_num_outputs" in t.attrib
        for p in t.findall(".//param"):
            assert "|" not in p.attrib["name"], f"Pipe syntax in test param: {p.attrib['name']}"
            val = p.attrib.get("value", "")
            if val.startswith("variationloop/"):
                for item in val.split(","):
                    f = ROOT / "tools" / "focaloop" / "test-data" / item.strip()
                    assert f.is_file(), f"Missing test-data file: {f}"
                    assert f.stat().st_size < 1_000_000
                    total_td_bytes += f.stat().st_size
    assert total_td_bytes < 1_000_000

    # Help markdown sections present and zero semicolons
    help_el = root.find("help")
    assert help_el.attrib.get("format") == "markdown"
    help_txt = help_el.text or ""
    assert ";" not in help_txt, "Semicolon found in variationloop.xml help"
    for heading in (
        "**What this tool does**",
        "**How it works**",
        "**Choosing an input source**",
        "**Inputs and manifest format**",
        "**Key parameters**",
        "**Outputs**",
        "**Tips and troubleshooting**",
    ):
        assert heading in help_txt, f"Missing help section {heading}"
    assert re.search(r"-----\n", help_txt)


def test_variationloop_multisample_paired_collection_auto_names_and_unified_mode(tmp_path: Path) -> None:
    """Verify multi-sample list:paired cohort calling, merged VCF, optional group sheet, and unified auto mode."""
    group_sheet = tmp_path / "groups.tsv"
    group_sheet.write_text("sample_id\tgroup\nisolate_A\tresistant\nisolate_B\tsensitive\n", encoding="utf-8")

    out_vcf = tmp_path / "cohort.vcf"
    out_var = tmp_path / "cohort.tsv"
    out_rep = tmp_path / "cohort.html"
    out_arc = tmp_path / "cohort.tar.zst"

    meta = run_variationloop_galaxy(
        analysis_mode="auto",
        reads_r1=[TEST_DATA_DIR / "sample_R1.fastq.gz", TEST_DATA_DIR / "sample_R1.fastq.gz"],
        reads_r2=[TEST_DATA_DIR / "sample_R2.fastq.gz", TEST_DATA_DIR / "sample_R2.fastq.gz"],
        read_ids=["isolate_A", "isolate_B"],
        manifest_path=group_sheet,
        ref_fasta=TEST_DATA_DIR / "ref_chr20_mini.fa",
        targets_bed=TEST_DATA_DIR / "targets.bed",
        host_depletion_mode="auto",
        threads=2,
        state_dir=tmp_path / "state_cohort",
        state_archive=out_arc,
        out_variants=out_var,
        out_vcf=out_vcf,
        out_report=out_rep,
    )
    assert meta["summary"]["n_samples"] == 2
    assert meta["summary"]["mode"] == "wgs_stream"
    assert {s["sample"]: s["group"] for s in meta["samples"]} == {
        "isolate_A": "resistant",
        "isolate_B": "sensitive",
    }
    vcf_txt = out_vcf.read_text(encoding="utf-8")
    chrom_line = [ln for ln in vcf_txt.splitlines() if ln.startswith("#CHROM")][0]
    assert "isolate_A" in chrom_line and "isolate_B" in chrom_line
    df = pd.read_csv(out_var, sep="\t")
    assert set(df["sample_id"]) == {"isolate_A", "isolate_B"}
    rep_txt = out_rep.read_text(encoding="utf-8")
    assert ".hero code {" in rep_txt
    assert "isolate_A" in rep_txt and "isolate_B" in rep_txt

    # Auto mode on CRAM + CRAI automatically selects focused_cram and derives clean sample name from bam_ids
    out_vcf_cram = tmp_path / "auto_cram.vcf"
    out_var_cram = tmp_path / "auto_cram.tsv"
    out_rep_cram = tmp_path / "auto_cram.html"
    meta_cram = run_variationloop_galaxy(
        analysis_mode="auto",
        bam_path=[TEST_DATA_DIR / "sample.cram"],
        bam_ids=["patient_01.cram"],
        crai_path=[TEST_DATA_DIR / "sample.cram.crai"],
        ref_fasta=TEST_DATA_DIR / "ref_chr20_mini.fa",
        targets_bed=TEST_DATA_DIR / "targets.bed",
        threads=2,
        state_dir=tmp_path / "state_cram",
        out_variants=out_var_cram,
        out_vcf=out_vcf_cram,
        out_report=out_rep_cram,
    )
    assert meta_cram["summary"]["mode"] == "focused_cram"
    assert meta_cram["samples"][0]["sample"] == "patient_01"
    assert "patient_01" in out_vcf_cram.read_text(encoding="utf-8")

    # Clean sample-name preservation for numbered list:paired element identifiers
    from focaloop.runners_variation import _clean_sample_name
    assert _clean_sample_name("sample_1") == "sample_1"
    assert _clean_sample_name("sample_2") == "sample_2"
    assert _clean_sample_name("patient_1.bam") == "patient_1"
    assert _clean_sample_name("sample_R1.fastq.gz") == "sample"
    assert _clean_sample_name("sample_1.fastq.gz") == "sample"


def test_variationloop_parallel_logan_cohort_workers_and_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify multi-sample Logan cohorts split across threads//4 concurrent workers and preserve order."""
    import threading
    from focaloop.malariavar import logan_var as LV
    from focaloop import logan as LG

    manifest = tmp_path / "cohort.tsv"
    manifest.write_text(
        "sample_id\tgroup\tsource\n"
        "pv_01\tAF\tERR100001\n"
        "pv_02\tAF\tERR100002\n"
        "pv_03\tESEA\tERR100003\n"
        "pv_04\tESEA\tERR100004\n"
        "pv_05\tLAM\tERR100005\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        LG,
        "probe_logan_runs",
        lambda accs, kind="unitigs", **kw: {
            "logan_runs": list(accs),
            "fallback_runs": [],
            "probes": {a: {"available": True, "content_length": 1024} for a in accs},
            "total_logan_bytes": 1024 * len(accs),
        },
    )
    monkeypatch.setattr(LV, "ensure_var_logan_bins", lambda: (Path("/bin/true"), Path("/bin/true")))
    monkeypatch.setattr(LV, "_select_var_aligner", lambda mapper: ("bwa-mem2", "/bin/true"))
    monkeypatch.setattr(LV, "ensure_var_aligner_index", lambda *a, **k: TEST_DATA_DIR / "ref_chr20_mini.fa")
    monkeypatch.setattr(LV, "build_k15_reference_bitset", lambda *a, **k: tmp_path / "k15.bit")

    seen_threads: list[int] = []
    seen_tids: set[int] = set()
    lock = threading.Lock()

    def _fake_run_logan(*, acc, sample, ctg_lens, out_vcf_gz, threads, min_wsaf=0.25, **kw):
        import time as _t
        assert abs(min_wsaf - 0.25) < 1e-9
        _t.sleep(0.05)
        with lock:
            seen_threads.append(threads)
            seen_tids.add(threading.get_ident())
        out_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
        vcf_txt = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=chr20_mini,length=8000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n'
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n'
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
            f"chr20_mini\t2201\t.\tC\tT\t99.0\tPASS\t.\tGT:GQ:DP:AD\t1/1:99:40:0,40\n"
        )
        out_vcf_gz.write_bytes(gzip.compress(vcf_txt.encode("utf-8")))
        return {"sample": sample, "accession": acc, "source": "logan_unitigs", "n_variants": 1, "n_snv": 1, "n_indel": 0, "wire_bytes": 1024, "wall_s": 0.05}

    monkeypatch.setattr(LV, "run_varloop_logan_unitigs", _fake_run_logan)

    out_vcf = tmp_path / "pv.vcf"
    out_var = tmp_path / "pv.tsv"
    out_rep = tmp_path / "pv.html"
    meta = run_variationloop_galaxy(
        analysis_mode="auto",
        manifest_path=manifest,
        ref_fasta=TEST_DATA_DIR / "ref_chr20_mini.fa",
        logan_mode="auto",
        threads=20,
        state_dir=tmp_path / "st_pv",
        out_variants=out_var,
        out_vcf=out_vcf,
        out_report=out_rep,
    )
    assert meta["summary"]["n_samples"] == 5
    assert meta["summary"]["mode"] == "logan_unitigs"
    assert seen_threads == [4, 4, 4, 4, 4]
    assert len(seen_tids) >= 2
    chrom_line = [ln for ln in out_vcf.read_text(encoding="utf-8").splitlines() if ln.startswith("#CHROM")][0]
    assert chrom_line.split("\t")[9:] == ["pv_01", "pv_02", "pv_03", "pv_04", "pv_05"]


def test_prep_unitigs_tiles_long_sequences_and_streaming_archive(tmp_path: Path) -> None:
    """Verify prep_unitigs_filt_adapt tiles >2.5 kb unitigs with 30 bp overlap and archive streams cleanly."""
    import subprocess
    from focaloop.malariavar.logan_var import ensure_var_logan_bins
    from focaloop.archive import pack_state, unpack_state

    prep_bin, _ = ensure_var_logan_bins()
    short_seq = "ACGT" * 125  # 500 bp
    # Build deterministic non-repeating 34,000 bp sequence so overlap match is unique
    bases = "ACGT"
    long_seq = "".join(bases[(i * 7 + (i // 13)) & 3] for i in range(34000))
    inp = f">0 ka:f:12.5\n{short_seq}\n>1 ka:f:11.7\n{long_seq}\n"
    res = subprocess.run(
        [str(prep_bin), "1.5", "0", "-", "0", "0", "3.0"],
        input=inp,
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

    assert recs[0] == ("0_1250", short_seq)
    tiles = [(h, s) for h, s in recs[1:]]
    assert len(tiles) >= 16
    assert all(h.endswith("_1170") and "c" in h for h, _ in tiles)
    assert all(500 < len(s) <= 2500 for _, s in tiles)
    # Reconstruct interior [15, L-15) from midpoint-trimmed tiles: must match original 100% bit-for-bit
    reconstructed = "".join(s[15:-15] for _, s in tiles)
    assert reconstructed == long_seq[15:-15]

    # Streaming pack/unpack round-trip + zstd-bomb guard
    st = tmp_path / "st_in"
    st.mkdir()
    (st / "meta.json").write_text('{"ok": true}\n', encoding="utf-8")
    (st / "payload.bin").write_bytes(b"ABCD" * 20000)
    arc = tmp_path / "st.tar.zst"
    pack_state(st, arc)
    out = unpack_state(arc, tmp_path / "st_out")
    assert (out / "meta.json").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert (out / "payload.bin").read_bytes() == b"ABCD" * 20000
    with pytest.raises(ValueError, match="inflates past"):
        unpack_state(arc, tmp_path / "st_bomb", max_bytes=1024)


def test_variationloop_haploid_wsaf_fastq_logan_and_query_fst(tmp_path: Path) -> None:
    """Haploid WSAF mode emits GT=1 + FORMAT/WSAF across FASTQ and Logan, and drives continuous WSAF Fst in Query."""
    from focaloop.galaxy_runners import run_warm_engine_query
    from focaloop.malariavar.logan_var import run_varloop_logan_unitigs
    from focaloop.engine import STATE_META, write_json
    from focaloop.hooks import HypothesisLedger
    from focaloop.state import FocaloopStateStore

    ref_fa = TEST_DATA_DIR / "ref_chr20_mini.fa"
    r1 = TEST_DATA_DIR / "sample_R1.fastq.gz"
    r2 = TEST_DATA_DIR / "sample_R2.fastq.gz"
    bed = TEST_DATA_DIR / "targets.bed"

    # 1. Raw paired FASTQ under haploid WSAF mode
    vcf_h = tmp_path / "haploid_fq.vcf"
    tsv_h = tmp_path / "haploid_fq.tsv"
    rep_h = tmp_path / "haploid_fq.html"
    st_h = tmp_path / "state_h"
    arc_h = tmp_path / "state_h.tar.zst"
    meta_h = run_variationloop_galaxy(
        analysis_mode="auto",
        ploidy="haploid",
        min_wsaf=0.25,
        ref_fasta=ref_fa,
        reads_r1=[r1],
        reads_r2=[r2],
        read_ids=["clone_mix_1"],
        targets_bed=bed,
        out_variants=tsv_h,
        out_vcf=vcf_h,
        out_report=rep_h,
        state_dir=st_h,
        state_archive=arc_h,
        threads=2,
        logan_mode="off",
    )
    assert meta_h["ploidy"] == "haploid"
    assert abs(meta_h["min_wsaf"] - 0.25) < 1e-9
    vcf_txt = vcf_h.read_text(encoding="utf-8")
    assert "##FORMAT=<ID=WSAF,Number=A,Type=Float," in vcf_txt
    data_lines = [ln for ln in vcf_txt.splitlines() if ln and not ln.startswith("#")]
    assert len(data_lines) >= 2
    for ln in data_lines:
        parts = ln.split("\t")
        assert parts[8] == "GT:GQ:DP:AD:WSAF"
        gt = parts[9].split(":")[0]
        ws = float(parts[9].split(":")[4])
        assert gt == "1", f"Expected haploid GT=1, got {gt!r} in {ln}"
        assert 0.25 <= ws <= 1.0
    assert "0/1" not in "\n".join(data_lines)
    assert "1/1" not in "\n".join(data_lines)
    df_h = pd.read_csv(tsv_h, sep="\t")
    assert "wsaf" in df_h.columns
    assert set(df_h["gt"].astype(str)) == {"1"}
    assert (df_h["wsaf"] >= 0.25).all()
    assert "haploid" in rep_h.read_text(encoding="utf-8")

    # 2. Logan unitigs under haploid mode: verify continuous WSAF filtering on a 30% minor clone + 85% major clone
    ref_lines = [ln.strip() for ln in ref_fa.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ctg_name = ref_lines[0][1:].split()[0]
    ref_seq = "".join(ref_lines[1:]).upper()
    assert len(ref_seq) >= 2000
    sub_ref = ref_seq[:2000]
    def _mut(b: str) -> str:
        return "T" if b != "T" else "A"
    # Position 500 (0-based): 30% minor clone (ka=9 alt vs ka=21 ref -> WSAF=0.30)
    # Position 1200 (0-based): 85% major clone (ka=25.5 alt vs ka=4.5 ref -> WSAF=0.85)
    u_ref_1 = sub_ref[300:700]
    u_alt_1 = sub_ref[300:500] + _mut(sub_ref[500]) + sub_ref[501:700]
    u_ref_2 = sub_ref[1000:1400]
    u_alt_2 = sub_ref[1000:1200] + _mut(sub_ref[1200]) + sub_ref[1201:1400]
    unitigs_fa = tmp_path / "mix_unitigs.fa"
    unitigs_fa.write_text(
        f">0 ka:f:21.0\n{u_ref_1}\n"
        f">1 ka:f:9.0\n{u_alt_1}\n"
        f">2 ka:f:4.5\n{u_ref_2}\n"
        f">3 ka:f:25.5\n{u_alt_2}\n",
        encoding="utf-8",
    )
    vcf_l25 = tmp_path / "logan_h25.vcf.gz"
    tel25 = run_varloop_logan_unitigs(
        acc="SYNTH1",
        sample="SYNTH1",
        ref_fa=ref_fa,
        ctg_lens={ctg_name: len(ref_seq)},
        out_vcf_gz=vcf_l25,
        work_dir=tmp_path / "lw25",
        min_wsaf=0.25,
        ploidy="haploid",
        local_unitigs_path=unitigs_fa,
        threads=2,
    )
    assert tel25["ploidy"] == "haploid"
    assert tel25["n_snv"] == 2
    with gzip.open(vcf_l25, "rt", encoding="utf-8") as fh:
        l25_lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    assert len(l25_lines) == 2
    wsafs_25 = {}
    for ln in l25_lines:
        p = ln.split("\t")
        assert p[8] == "GT:GQ:DP:AD:WSAF"
        fvals = p[9].split(":")
        assert fvals[0] == "1"
        wsafs_25[int(p[1])] = float(fvals[4])
    assert 0.25 <= wsafs_25[501] <= 0.35
    assert 0.80 <= wsafs_25[1201] <= 0.90

    vcf_l50 = tmp_path / "logan_h50.vcf.gz"
    tel50 = run_varloop_logan_unitigs(
        acc="SYNTH1",
        sample="SYNTH1",
        ref_fa=ref_fa,
        ctg_lens={ctg_name: len(ref_seq)},
        out_vcf_gz=vcf_l50,
        work_dir=tmp_path / "lw50",
        min_wsaf=0.50,
        ploidy="haploid",
        local_unitigs_path=unitigs_fa,
        threads=2,
    )
    # 30% minor clone filtered out cleanly at min_wsaf=0.50; 85% clonal SNV retained
    assert tel50["n_snv"] == 1

    # 3. Focaloop Query on a haploid state uses continuous WSAF (not quantized GT=1 -> 1.0)
    q_state = tmp_path / "q_haploid_state"
    q_state.mkdir()
    q_vdf = pd.DataFrame(
        [
            {"sample_id": "A1", "chrom": "chr1", "pos": 100, "ref": "A", "alt": "G", "kind": "SNV", "gt": "1", "gq": 60.0, "dp": 40.0, "ad_ref": 8.0, "ad_alt": "32", "wsaf": 0.80, "qual": 60.0, "source": "variationloop"},
            {"sample_id": "A2", "chrom": "chr1", "pos": 100, "ref": "A", "alt": "G", "kind": "SNV", "gt": "1", "gq": 60.0, "dp": 40.0, "ad_ref": 8.0, "ad_alt": "32", "wsaf": 0.80, "qual": 60.0, "source": "variationloop"},
            {"sample_id": "B1", "chrom": "chr1", "pos": 100, "ref": "A", "alt": "G", "kind": "SNV", "gt": "1", "gq": 40.0, "dp": 40.0, "ad_ref": 32.0, "ad_alt": "8", "wsaf": 0.20, "qual": 40.0, "source": "variationloop"},
            {"sample_id": "B2", "chrom": "chr1", "pos": 100, "ref": "A", "alt": "G", "kind": "SNV", "gt": "1", "gq": 40.0, "dp": 40.0, "ad_ref": 32.0, "ad_alt": "8", "wsaf": 0.20, "qual": 40.0, "source": "variationloop"},
        ]
    )
    q_vdf.to_parquet(q_state / "variants.parquet", index=False)
    store = FocaloopStateStore(q_state)
    HypothesisLedger.from_store(store, alpha_budget=0.05)
    q_meta = {
        "engine": "variationloop",
        "version": "0.4.0",
        "blueprint_id": "variationloop",
        "mode": "logan_unitigs",
        "ploidy": "haploid",
        "min_wsaf": 0.20,
        "summary": {"mode": "logan_unitigs", "ploidy": "haploid", "n_samples": 4, "n_variants": 4},
        "samples": [
            {"sample": "A1", "group": "PopA"},
            {"sample": "A2", "group": "PopA"},
            {"sample": "B1", "group": "PopB"},
            {"sample": "B2", "group": "PopB"},
        ],
    }
    write_json(q_state / STATE_META, q_meta)
    q_tsv = tmp_path / "q_fst.tsv"
    q_res = run_warm_engine_query(
        state_dir=q_state,
        meta=q_meta,
        contrast="PopA:PopB",
        actor="tester",
        out_table=q_tsv,
        out_verdict=tmp_path / "q_verdict.json",
        out_report=tmp_path / "q_report.html",
        mode="region",
        n_perm=20,
    )
    q_tbl = pd.read_csv(q_tsv, sep="\t")
    assert len(q_tbl) == 1
    assert abs(float(q_tbl.iloc[0]["af_group_a"]) - 0.80) < 1e-4
    assert abs(float(q_tbl.iloc[0]["af_group_b"]) - 0.20) < 1e-4
    assert abs(float(q_tbl.iloc[0]["delta_af"]) - 0.60) < 1e-4
    # Exact finite-sample Hudson Fst for pa=0.80, pb=0.20, n_a=n_b=2: (0.36 - 0.16 - 0.16) / 0.68 = 0.0588
    assert abs(float(q_tbl.iloc[0]["hudson_fst"]) - (0.04 / 0.68)) < 1e-3
    assert q_res["engine"] == "variationloop"


def test_logan_streaming_curl_flags_and_broken_pipe_no_deadlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify Logan curl pipes use speed timeouts (never --max-time/--retry) and never deadlock on broken pipes."""
    import subprocess
    import sys
    import time
    from focaloop.malariavar import logan_var as LV
    from focaloop.panexpr import logan_expr as LE

    captured_curl_cmds: list[list[str]] = []
    real_popen = subprocess.Popen

    def _mock_popen(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "curl":
            captured_curl_cmds.append(list(cmd))
            # Continuous stdout producer that would block forever in pipe_write if parent leaks p_curl.stdout
            producer = [
                sys.executable,
                "-c",
                "import sys\nchunk = b'X' * 65536\nfor _ in range(512):\n sys.stdout.buffer.write(chunk)\n sys.stdout.buffer.flush()\n",
            ]
            return real_popen(producer, *args, **kwargs)
        if isinstance(cmd, list) and cmd and cmd[0] == "zstd":
            # Downstream dies immediately after reading 4 bytes, triggering BrokenPipeError in _pipe_and_count
            consumer = [sys.executable, "-c", "import sys; sys.stdin.buffer.read(4); sys.exit(1)"]
            return real_popen(consumer, *args, **kwargs)
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(LV.subprocess, "Popen", _mock_popen)
    monkeypatch.setattr(LE.subprocess, "Popen", _mock_popen)

    ref_fa = TEST_DATA_DIR / "ref_chr20_mini.fa"
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="Logan VariationLoop stream failed"):
        LV.run_varloop_logan_unitigs(
            acc="SYNTH_PIPE_VAR",
            sample="SYNTH_PIPE_VAR",
            ref_fa=ref_fa,
            ctg_lens={"chr20_mini": 8000},
            out_vcf_gz=tmp_path / "var_pipe.vcf.gz",
            work_dir=tmp_path / "w_var",
            threads=1,
        )
    assert time.monotonic() - t0 < 10.0, "VariationLoop Logan pipe deadlocked on early downstream exit"

    tx_fa = tmp_path / "tx.fa"
    tx_fa.write_text(">tx1\n" + ("ACGTACGTACGT" * 20) + "\n", encoding="utf-8")
    tx2g = tmp_path / "tx2g.tsv"
    tx2g.write_text("tx1\tg1\n", encoding="utf-8")
    prep_bin, ec_bin = LE.ensure_expr_logan_bins()
    aln_name, aln_bin = LE._select_unitig_aligner()
    LE._build_aligner_index(aln_name, aln_bin, tx_fa, lambda _: None)
    t1 = time.monotonic()
    with pytest.raises(RuntimeError, match="Logan unitig stream failed"):
        LE.stream_logan_expression_run(
            acc="SYNTH_PIPE_EXPR",
            tx_fa=tx_fa,
            tx2gene=tx2g,
            out_dir=tmp_path / "w_expr",
            prep_bin=prep_bin,
            ec_bin=ec_bin,
            aligner_name=aln_name,
            aligner_bin=aln_bin,
            threads=1,
        )
    assert time.monotonic() - t1 < 10.0, "ExpressionLoop Logan pipe deadlocked on early downstream exit"

    assert len(captured_curl_cmds) >= 2
    for cmd in captured_curl_cmds:
        assert "--max-time" not in cmd
        assert "--retry" not in cmd
        assert "--connect-timeout" in cmd
        assert "--speed-limit" in cmd
        assert "--speed-time" in cmd
