"""Portability and asset-fallback tests for tools-iuc 0.3.0 readiness."""
from __future__ import annotations

import pathlib
import re

import pytest

from focaloop import runners_geneloop, runners_variation


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN_PERSONAL_PATHS = re.compile(r"/media/anton|/home/anton|/Users/anton|/dev/shm/vgp")


def test_no_hardcoded_personal_paths():
    offenders: list[str] = []
    for sub in ("focaloop", "tools", "container"):
        base = REPO_ROOT / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix in (".pyc", ".so", ".gz", ".bam", ".cram", ".zst", ".parquet"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                if FORBIDDEN_PERSONAL_PATHS.search(line):
                    offenders.append(f"{p.relative_to(REPO_ROOT)}:{idx}: {line.strip()}")
    assert not offenders, "Hardcoded personal paths found:\n" + "\n".join(offenders)


def test_variationloop_requires_ref_fasta_without_server_pf3d7(tmp_path, monkeypatch):
    monkeypatch.setenv("S", str(tmp_path / "empty_s"))
    monkeypatch.setenv("FOCALOOP_PF3D7_DIR", str(tmp_path / "empty_pf3d7"))
    monkeypatch.setattr(runners_variation, "_pf3d7_assets_ready", lambda ref: False)
    with pytest.raises(ValueError, match="--ref-fasta"):
        runners_variation.run_variationloop_galaxy(
            analysis_mode="wgs_stream",
            ref_fasta=None,
            out_variants=tmp_path / "var.tsv",
            out_vcf=tmp_path / "var.vcf",
            out_report=tmp_path / "report.html",
            state_dir=tmp_path / "state",
        )


def test_geneloop_auto_falls_back_cleanly_without_weights(monkeypatch):
    monkeypatch.setattr(runners_geneloop, "discover_tiberius_weights", lambda *a, **kw: None)
    monkeypatch.setattr(runners_geneloop, "detect_tiberius_runtime", lambda: {"backend": "none"})
    res = runners_geneloop.resolve_execution_mode("auto", "vertebrates", None)
    assert res["effective_mode"] == "fast_evidence"
    assert res["weights_dir"] is None


def test_focaloop_query_writes_html_report_for_blueprint_state(tmp_path):
    from focaloop.ops import run_query

    for fname, contrast, mode in [
        ("panexpr_state.tar.zst", "treated:control", "auto"),
        ("malariavar_state.tar.zst", "site=Ghana:Mali", "fst"),
    ]:
        state_arc = REPO_ROOT / "tools" / "focaloop" / "test-data" / fname
        out_tbl = tmp_path / f"{fname}.tsv"
        out_ver = tmp_path / f"{fname}.json"
        out_rep = tmp_path / f"{fname}.html"
        run_query(state_arc, contrast, "agent:test", out_tbl, out_ver, mode=mode, out_report=out_rep)
        html = out_rep.read_text(encoding="utf-8")
        assert "0 B" in html
        assert "Focaloop Query" in html
        assert "—" not in html.split("Error Budget Remaining", 1)[1][:120]


def test_galaxy_wrappers_biologist_hygiene_and_iuc_standards():
    import xml.etree.ElementTree as ET

    expected_names = {
        "expressionloop.xml": "Focaloop ExpressionLoop",
        "variationloop.xml": "Focaloop VariationLoop",
        "geneloop.xml": "Focaloop GeneLoop",
        "focaloop_query.xml": "Focaloop Query",
    }
    banned_patterns = [
        r"\bsiev",
        r"Pf3D7",
        r"Plasmodium",
        r"MalariaGEN",
        r"FOCALOOP_PF3D7",
        r"PanExpr",
        r"FocalAnnot",
        r"HypothesisLedger",
        r"PagedContig",
        r"glcore",
        r"Wilkinson",
        r"RTX A5000",
        r"CI mode",
        r"\$F_",
        r"nosofttmasking",
    ]
    for fname, want_name in expected_names.items():
        p = REPO_ROOT / "tools" / "focaloop" / fname
        root = ET.parse(p).getroot()
        assert root.attrib["name"] == want_name, f"{fname}: expected name {want_name!r}, got {root.attrib['name']!r}"
        assert root.find("edam_operations") is not None, f"{fname}: missing <edam_operations>"
        prose_chunks = []
        for el in root.iter():
            for attr in ("label", "help", "title"):
                if attr in el.attrib:
                    prose_chunks.append((f"{el.tag}@{attr}", el.attrib[attr]))
            if el.tag in ("description", "option", "help") and el.text:
                prose_chunks.append((el.tag, el.text))
        for where, text in prose_chunks:
            assert ";" not in text, f"{fname} {where}: semicolon in user-facing text"
            # Detect RST double-backticks without triggering on Markdown triple-backtick fences
            cleaned = re.sub(r"```+", "", text)
            assert "``" not in cleaned, f"{fname} {where}: RST double-backtick in Markdown"
            for pat in banned_patterns:
                assert not re.search(pat, text, re.IGNORECASE), f"{fname} {where}: banned pattern {pat!r}"
        for val in root.iter("validator"):
            if val.attrib.get("type") == "regex":
                rx = (val.text or "").strip()
                assert rx.startswith("^") and rx.endswith("$"), f"{fname}: unanchored regex {rx!r}"


def test_variationloop_warm_query_and_uncompressed_fastq(tmp_path):
    import gzip
    import shutil
    from focaloop.ops import run_query

    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "variationloop"
    # Decompress test FASTQs to verify uncompressed FASTQ support with host filter + subsampling
    r1_plain = tmp_path / "s1_R1.fastq"
    r2_plain = tmp_path / "s1_R2.fastq"
    with gzip.open(td / "sample_R1.fastq.gz", "rb") as fi, open(r1_plain, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    with gzip.open(td / "sample_R2.fastq.gz", "rb") as fi, open(r2_plain, "wb") as fo:
        shutil.copyfileobj(fi, fo)

    # Create a 2-group cohort manifest and run VariationLoop -> Focaloop Query end-to-end
    manifest = tmp_path / "cohort.tsv"
    manifest.write_text(
        "sample_id\tgroup\tsource_uri_or_srr\n"
        "s1\tgroup_a\ts1\n"
        "s2\tgroup_b\ts2\n",
        encoding="utf-8",
    )
    r2a = tmp_path / "s2_R1.fastq"
    r2b = tmp_path / "s2_R2.fastq"
    shutil.copyfile(r1_plain, r2a)
    shutil.copyfile(r2_plain, r2b)

    state_arc = tmp_path / "vl_state.focaloop.tar.zst"
    meta = runners_variation.run_variationloop_galaxy(
        analysis_mode="wgs_stream",
        manifest_path=manifest,
        reads=[str(r1_plain), str(r2_plain), str(r2a), str(r2b)],
        read_ids=["s1_R1.fastq", "s1_R2.fastq", "s2_R1.fastq", "s2_R2.fastq"],
        ref_fasta=td / "ref_chr20_mini.fa",
        targets_bed=td / "targets.bed",
        use_host_depletion=True,
        subsample_fraction=0.95,
        threads=2,
        state_dir=tmp_path / "vl_state_dir",
        state_archive=state_arc,
        out_variants=tmp_path / "var.tsv",
        out_vcf=tmp_path / "var.vcf",
        out_report=tmp_path / "var.html",
    )
    assert meta["blueprint_id"] == "variationloop"
    assert state_arc.is_file()

    q_tbl = tmp_path / "q.tsv"
    q_ver = tmp_path / "q.json"
    q_rep = tmp_path / "q.html"
    q_res = run_query(state_arc, "group=group_a:group_b", "human", q_tbl, q_ver, mode="fst", out_report=q_rep)
    assert q_res["new_bytes"] == 0
    assert q_res["engine"] == "variationloop"
    assert "hudson_fst" in q_tbl.read_text(encoding="utf-8")
    assert "VariationLoop" in q_rep.read_text(encoding="utf-8")


def test_geneloop_unmasked_alias_and_dat_weights_and_gff3_sniff(tmp_path):
    import tarfile
    from focaloop.geneloop.seed import discover_tiberius_weights
    from focaloop.runners_expression import _sniff_annotation_ext

    # 1. Clean mammalia_unmasked clade maps cleanly
    assert "mammalia_unmasked" in runners_geneloop.CLADE_CONFIGS
    assert runners_geneloop.CLADE_CONFIGS["mammalia_unmasked"]["tiberius_model_cfg"] == "mammalia_nosofttmasking_v2"

    # 2. Galaxy .dat tarball weights archive is detected by magic bytes (not .tar.gz suffix)
    w_src = tmp_path / "w_src"
    w_src.mkdir()
    (w_src / "weights.h5").write_bytes(b"fake")
    dat_tar = tmp_path / "dataset_99999.dat"
    with tarfile.open(dat_tar, "w:gz") as tf:
        tf.add(w_src / "weights.h5", arcname="weights.h5")
    found = discover_tiberius_weights("vertebrates", explicit_weights=dat_tar)
    assert found is not None and (found / "weights.h5").is_file()

    # 3. Galaxy .dat GFF3 file is sniffed properly
    gff_dat = tmp_path / "dataset_12345.dat"
    gff_dat.write_text('##gff-version 3\nchr1\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1;Name="alpha"\n', encoding="utf-8")
    assert _sniff_annotation_ext(gff_dat) == ".gff3"
