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

    # 4. Uncompressed .dat GFF3 symlink preserves .gff3 suffix so quoted attrs parse cleanly
    from focaloop.panexpr.catalog import iter_features
    from focaloop.runners_expression import _gunzip_if_needed

    staged = _gunzip_if_needed(gff_dat, tmp_path / "refs", "primary.gff3")
    assert staged.suffix == ".gff3"
    feats = list(iter_features(staged))
    assert len(feats) == 1 and feats[0].attrs["ID"] == "g1"

    # 5. NCBI RefSeq bacterial GFF3 (many CDSs + few single-exon tRNA/rRNA exons) auto-detects as prokaryotic
    from focaloop.runners_expression import _detect_organism_mode

    prok_gff = tmp_path / "refseq_bact.gff3"
    lines = ["##gff-version 3"]
    for idx in range(200):
        lines.append(f"chr1\tRefSeq\tgene\t{idx*100+1}\t{idx*100+90}\t.\t+\t.\tID=g{idx};locus_tag=LT_{idx}")
        lines.append(f"chr1\tRefSeq\tCDS\t{idx*100+1}\t{idx*100+90}\t.\t+\t0\tID=cds{idx};Parent=g{idx}")
    for idx in range(10):
        lines.append(f"chr1\tRefSeq\texon\t{30000+idx*100}\t{30080+idx*100}\t.\t+\t.\tID=ex{idx};Parent=trna{idx}")
    prok_gff.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _detect_organism_mode(prok_gff, "auto") == "prokaryotic"


def test_ncbi_ref_resolution_fetch_and_3part_contrast(tmp_path, monkeypatch):
    import gzip
    from focaloop import ncbi_ref
    from focaloop.ops import parse_contrast

    # 3-part contrast parsing used by ExpressionLoop & Focaloop Query
    assert parse_contrast("condition:treated:control") == ("condition", ("treated", "control"))
    assert parse_contrast("site=Ghana:Mali") == ("site", ("Ghana", "Mali"))
    assert parse_contrast("treated:control") == ("condition", ("treated", "control"))

    # Mock NCBI directory listing with numeric version ordering (.10 > .2)
    listing_html = (
        '<a href="GCF_000005845.2_ASM584v2/">GCF_000005845.2_ASM584v2/</a>\n'
        '<a href="GCF_000005845.10_ASM584v10/">GCF_000005845.10_ASM584v10/</a>\n'
    )
    monkeypatch.setattr(ncbi_ref, "_fetch_text", lambda url, **kw: listing_html)

    asm_dir, base_url = ncbi_ref.resolve_ncbi_assembly_dir("GCF_000005845.2")
    assert asm_dir == "GCF_000005845.2_ASM584v2"
    assert base_url.endswith("/GCF/000/005/845/GCF_000005845.2_ASM584v2")

    latest_dir, _ = ncbi_ref.resolve_ncbi_assembly_dir("GCF_000005845")
    assert latest_dir == "GCF_000005845.10_ASM584v10"

    with pytest.raises(ValueError, match="Invalid NCBI assembly accession"):
        ncbi_ref.resolve_ncbi_assembly_dir("NOT_AN_ACCESSION")

    downloaded: list[str] = []

    def _fake_download(url: str, dest: pathlib.Path, **kw):
        downloaded.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(">chr1\nACGT\n" if url.endswith(".fna.gz") else "##gff-version 3\n")
        return dest

    monkeypatch.setattr(ncbi_ref, "_download_file", _fake_download)
    monkeypatch.delenv("FOCALOOP_NCBI_CACHE_DIR", raising=False)

    res = ncbi_ref.fetch_ncbi_assembly("GCF_000005845.2", tmp_path / "ncbi", need_annotation=True)
    assert res["fasta"].is_file() and res["gff"] is not None and res["gff"].is_file()
    assert len(downloaded) == 2

    downloaded.clear()
    res_no_gff = ncbi_ref.fetch_ncbi_assembly("GCF_000005845.2", tmp_path / "ncbi2", need_annotation=False)
    assert res_no_gff["fasta"].is_file() and res_no_gff["gff"] is None
    assert len(downloaded) == 1


def test_variationloop_logan_unitigs_end_to_end_and_warm_query(tmp_path):
    """End-to-end C compilation + k=15 filter + weighted pileup + VCF + Focaloop Query on synthetic Logan unitigs."""
    from focaloop.ops import run_query

    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "variationloop"
    ref_fa = td / "ref_chr20_mini.fa"
    # Extract reference sequence to synthesize ka-tagged Logan unitigs carrying a known SNV and deletion
    seq_lines = [ln.strip() for ln in ref_fa.read_text().splitlines() if not ln.startswith(">")]
    seq = "".join(seq_lines).upper()
    assert len(seq) > 4000

    def _mutate_unitig_fa(out_fa: pathlib.Path, snv_pos0: int, ka: float, add_del: bool = False) -> pathlib.Path:
        alt_base = "C" if seq[snv_pos0] != "C" else "G"
        # Build tiled 120 bp unitigs spanning the locus with ka:f: tag
        recs = []
        uid = 0
        for start in range(100, min(len(seq) - 200, 3500), 60):
            seg = list(seq[start : start + 120])
            if start <= snv_pos0 < start + 120:
                seg[snv_pos0 - start] = alt_base
            s_out = "".join(seg)
            if add_del and start <= 1500 < start + 100:
                rel = 1500 - start
                s_out = s_out[:rel] + s_out[rel + 3 :]
            recs.append(f">{uid} ka:f:{ka:.2f}\n{s_out}\n")
            uid += 1
        # Add off-target random host-like unitigs that the k=15 bitset should drop
        recs.append(">9999 ka:f:8.00\n" + ("ATATGCGC" * 12) + "\n")
        out_fa.write_text("".join(recs), encoding="utf-8")
        return out_fa

    u_a = _mutate_unitig_fa(tmp_path / "sample_a.unitigs.fa", snv_pos0=1000, ka=18.0, add_del=True)
    u_b = _mutate_unitig_fa(tmp_path / "sample_b.unitigs.fa", snv_pos0=2000, ka=18.0, add_del=False)

    manifest = tmp_path / "logan_manifest.tsv"
    manifest.write_text(
        f"sample_id\tgroup\tsource\n"
        f"s_a\tgroup_a\t{u_a}\n"
        f"s_b\tgroup_b\t{u_b}\n",
        encoding="utf-8",
    )
    state_arc = tmp_path / "logan_var.focaloop.tar.zst"
    vcf_out = tmp_path / "logan_var.vcf"
    var_out = tmp_path / "logan_var.tsv"
    rep_out = tmp_path / "logan_var.html"

    meta = runners_variation.run_variationloop_galaxy(
        analysis_mode="logan_unitigs",
        manifest_path=manifest,
        ref_fasta=ref_fa,
        use_host_depletion=True,
        threads=2,
        state_dir=tmp_path / "logan_var_state",
        state_archive=state_arc,
        out_variants=var_out,
        out_vcf=vcf_out,
        out_report=rep_out,
    )
    assert meta["version"] == "0.4.0"
    assert meta["mode"] == "logan_unitigs"
    tsv = var_out.read_text(encoding="utf-8")
    assert "SNV" in tsv
    assert "INDEL" in tsv
    assert "1001" in tsv and "2001" in tsv

    # Verify zero-byte warm query works directly on Logan VariationLoop state
    q_tbl = tmp_path / "q_logan.tsv"
    q_ver = tmp_path / "q_logan.json"
    q_rep = tmp_path / "q_logan.html"
    q_res = run_query(state_arc, "group=group_a:group_b", "human", q_tbl, q_ver, mode="fst", out_report=q_rep)
    assert q_res["new_bytes"] == 0
    assert "hudson_fst" in q_tbl.read_text(encoding="utf-8")


def test_expressionloop_logan_unitigs_end_to_end(tmp_path):
    """End-to-end C compilation + prep_expr_unitigs + expr_sam_ec_v2 + PyDESeq2 + state export on synthetic Logan unitigs."""
    import pandas as pd
    from focaloop.panexpr.catalog import AssemblySpec, build_catalog
    from focaloop.panexpr.logan_expr import run_logan_expression_cohort
    from focaloop.panexpr.manifest import FastqFile, Run, StudyManifest

    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "expressionloop"
    ref_fa = td / "ref_primary.fa"
    ref_gtf = td / "ref_primary.gtf"
    cat = build_catalog([AssemblySpec("primary", ref_fa, ref_gtf)], "eukaryotic", "primary", tmp_path / "cat_tmp")
    gene_tgts = [t for t in cat.targets.values() if not t.is_decoy]
    assert len(gene_tgts) >= 4

    # Synthesize 6 sample unitig files (3 control, 3 treated) with strong differential abundance on first gene
    runs = []
    meta_rows = []
    local_u: dict[str, pathlib.Path] = {}
    for cond, boost in (("control", 1.0), ("treated", 12.0)):
        for rep_i in (1, 2, 3):
            sid = f"{cond}_{rep_i}"
            uf = tmp_path / f"{sid}.unitigs.fa"
            recs = []
            for gi, tgt in enumerate(gene_tgts):
                ka = (40.0 * boost) if gi == 0 else (25.0 + rep_i * 2.0 + (gi % 5))
                seq_u = tgt.seq[: min(len(tgt.seq), 250)]
                if len(seq_u) >= 45:
                    recs.append(f">{gi} ka:f:{ka:.2f}\n{seq_u}\n")
            uf.write_text("".join(recs), encoding="utf-8")
            local_u[sid] = uf
            runs.append(
                Run(
                    run_id=sid,
                    sample=sid,
                    condition=cond,
                    layout="SE",
                    strategy="RNA-Seq",
                    read_count=10000,
                    files=(FastqFile(url=str(uf), bytes=uf.stat().st_size, local_path=str(uf)),),
                )
            )
            meta_rows.append({"run_id": sid, "sample": sid, "condition": cond})

    manifest = StudyManifest(study_id="logan_test", accession="test", runs=runs)
    meta_df = pd.DataFrame(meta_rows).set_index("sample")
    res = run_logan_expression_cohort(
        catalog=cat,
        manifest=manifest,
        sample_meta_df=meta_df,
        contrast_list=["condition", "treated", "control"],
        design="~condition",
        gtf_path=ref_gtf,
        work_dir=tmp_path / "work",
        pan_state_dir=tmp_path / "pan_state",
        threads=2,
        local_unitig_map=local_u,
    )
    de = res["de_raw"]
    assert not de.empty
    top_gene = gene_tgts[0].gene_id
    row0 = de[de["gene_id"] == top_gene]
    assert not row0.empty
    assert float(row0["log2fc"].iloc[0]) > 2.0
    assert (tmp_path / "pan_state" / "target_counts.parquet").is_file()


def test_geneloop_logan_only_mode_partition(tmp_path, monkeypatch):
    from focaloop.geneloop import engine as gl_eng

    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "geneloop"
    gate = {
        "gate_id": "test_logan_only",
        "fasta_path": str(td / "genome.fa"),
        "proteins_path": "",
        "exclude_genus": "",
        "species": "",
        "tiberius_model_cfg": "vertebrates",
        "compleasm_lineage": "vertebrata_odb12",
        "tiberius_weights": "",
        "sra_short_runs": ["SRR111111", "SRR222222"],
        "sra_long_runs": [],
        "sra_run_organisms": {},
    }
    monkeypatch.setattr(
        gl_eng,
        "probe_logan_runs",
        lambda accs, **kw: {
            "logan_runs": ["SRR111111"],
            "fallback_short_runs": ["SRR222222"],
            "total_compressed_bytes": 123456,
        },
    )
    lp = gl_eng.LoopParams(logan_mode="logan_only")
    gr = gl_eng.GateRun("test_logan_only", root=tmp_path, loop=lp, gate=gate)
    lruns, fbruns, info = gr.partition_short_runs()
    assert lruns == ["SRR111111"]
    assert fbruns == []
    assert info["logan_mode"] == "logan_only"


def test_ncbi_related_strain_dedup_and_multi_ref_expressionloop(tmp_path, monkeypatch):
    from focaloop import ncbi_ref
    from focaloop.runners_expression import run_expressionloop_galaxy

    # 1. Verify strain deduplication drops duplicate ATCC 19606 assemblies & skips primary strain
    fake_reports = {
        "reports": [
            {
                "accession": "GCF_000000001.1",
                "assembly_info": {"assembly_name": "ASM1", "assembly_level": "Complete Genome", "refseq_category": "reference genome"},
                "organism": {"tax_id": 470, "organism_name": "Acinetobacter baumannii", "infraspecific_names": {"strain": "ATCC 19606"}},
            },
            {
                "accession": "GCF_000000002.1",
                "assembly_info": {"assembly_name": "ASM2", "assembly_level": "Complete Genome"},
                "organism": {"tax_id": 470, "organism_name": "Acinetobacter baumannii", "infraspecific_names": {"strain": "ATCC 19606"}},
            },
            {
                "accession": "GCF_000000003.1",
                "assembly_info": {"assembly_name": "ASM3", "assembly_level": "Complete Genome"},
                "organism": {"tax_id": 470, "organism_name": "Acinetobacter baumannii", "infraspecific_names": {"strain": "ARC6851"}},
            },
            {
                "accession": "GCF_000000004.1",
                "assembly_info": {"assembly_name": "ASM4", "assembly_level": "Complete Genome"},
                "organism": {"tax_id": 470, "organism_name": "Acinetobacter baumannii", "infraspecific_names": {"strain": "AB5075"}},
            },
        ]
    }
    monkeypatch.setattr(ncbi_ref, "_fetch_json", lambda url, **kw: fake_reports)
    cands = ncbi_ref.discover_related_ncbi_assemblies(
        "Acinetobacter baumannii",
        max_strains=2,
        exclude_accessions=["GCF_025677625.1"],
        exclude_strains=["ARC6851"],
    )
    accs = [c["accession"] for c in cands]
    assert accs == ["GCF_000000001.1", "GCF_000000004.1"]

    # 2. Verify explicit extra_references + auto_expand_strains end-to-end in run_expressionloop_galaxy
    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "expressionloop"
    monkeypatch.setattr(
        ncbi_ref,
        "fetch_ncbi_assembly",
        lambda acc, dest_dir, need_annotation=True: {
            "accession": acc,
            "assembly_dir": acc,
            "base_url": "mock://",
            "fasta": td / "ref_secondary.fa",
            "gff": td / "ref_secondary.gtf",
        },
    )
    reads = [str(td / f) for f in ("ctrl1.fastq.gz", "ctrl2.fastq.gz", "ctrl3.fastq.gz", "trt1.fastq.gz", "trt2.fastq.gz", "trt3.fastq.gz")]
    meta = run_expressionloop_galaxy(
        manifest_path=td / "manifest_se.tsv",
        ref_fasta=td / "ref_primary.fa",
        ref_gtf=td / "ref_primary.gtf",
        primary_label="v3",
        extra_references=[{"label": "v2", "fasta": str(td / "ref_secondary.fa"), "gtf": str(td / "ref_secondary.gtf")}],
        auto_expand_strains=1,
        auto_expand_taxon="Acinetobacter baumannii",
        reads=reads,
        contrast="condition:treated:control",
        out_de=tmp_path / "de.tsv",
        out_report=tmp_path / "rep.html",
        state_dir=tmp_path / "state",
        kernel_names=["mappy"],
        n_tournament_reads=400,
        threads=2,
    )
    asms = meta["catalog_summary"]["assemblies"]
    assert asms == ["v3", "v2", "GCF_000000001.1"]
    de_txt = (tmp_path / "de.tsv").read_text(encoding="utf-8")
    assert "v2:LEGACY_001458" in de_txt
    assert "GCF_000000001.1:LEGACY_001458" in de_txt


def test_makefile_build_native_and_trimmed_geneloop_query_ui(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    import xml.etree.ElementTree as ET

    # 1. Verify Makefile install-native builds all 16 C/OpenMP artifacts and 5 CLI helpers
    prefix = tmp_path / "conda_prefix"
    subprocess.run(
        ["make", "install-native", f"PREFIX={prefix}", f"PYTHON={sys.executable}"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    cache_dir = prefix / "share" / "focaloop" / "native_cache"
    bin_dir = prefix / "bin"
    assert len(list(cache_dir.iterdir())) == 16
    for exe_name in ("glcore", "prep_unitigs_filt_adapt", "logan_var_pileup", "prep_expr_unitigs", "expr_sam_ec_v2"):
        p = bin_dir / exe_name
        assert p.is_file() and os.access(p, os.X_OK)

    # 2. Verify sys.prefix auto-discovery works without FOCALOOP_NATIVE_CACHE or CC (BioContainer mode)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    for env_var in ("FOCALOOP_NATIVE_CACHE", "FOCALOOP_NATIVE_DIR", "FOCALOOP_BIN_CACHE", "FOCALOOP_BUILD_NATIVE_DIR", "GLCORE_BIN", "ORF_SCOUT_LIB"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("CC", "/nonexistent_cc_binary")

    from focaloop.galaxy_runners import prebuild_all_native_libs
    import focaloop.panexpr._fast_c as fc
    import focaloop.panexpr._medvedev_compile as mc

    monkeypatch.setattr(fc, "_TRIED", False)
    monkeypatch.setattr(fc, "_LIB", None)
    monkeypatch.setattr(mc, "_LOADED", {})
    built = prebuild_all_native_libs()
    assert len(built) == 16

    # 3. Verify trimmed orthogonal UI choices in GeneLoop and Focaloop Query XMLs
    gl_root = ET.parse(REPO_ROOT / "tools" / "focaloop" / "geneloop.xml").getroot()
    rna_opts = [
        o.attrib["value"]
        for c in gl_root.findall(".//conditional[@name='rna_source']/param[@name='source_type']/option")
        for o in [c]
    ]
    assert rna_opts == ["history_paired", "history_long", "history_bams", "public_sra", "none"]

    q_root = ET.parse(REPO_ROOT / "tools" / "focaloop" / "focaloop_query.xml").getroot()
    q_opts = [
        o.attrib["value"]
        for o in q_root.findall(".//conditional[@name='query_type']/param[@argument='--mode']/option")
    ]
    assert q_opts == ["contrast", "fst"]


def test_geneloop_ncbi_and_orthodb_homology_auto_fetch(tmp_path, monkeypatch):
    import gzip
    import xml.etree.ElementTree as ET
    from focaloop.geneloop import homology_fetch as hf

    # 1. Verify XML exposes all 4 orthogonal protein_source choices (ncbi, orthodb, history, none)
    gl_root = ET.parse(REPO_ROOT / "tools" / "focaloop" / "geneloop.xml").getroot()
    prot_opts = [
        o.attrib["value"]
        for o in gl_root.findall(".//conditional[@name='protein_source']/param[@name='source_type']/option")
    ]
    assert prot_opts == ["ncbi", "orthodb", "history", "none"]

    # 2. Verify NCBI related-species RefSeq proteome discovery (genus-diverse + target exclusion + unique ID prefixing)
    fake_reports = {
        "reports": [
            {
                "accession": "GCF_000001215.4",
                "assembly_info": {"assembly_level": "Chromosome", "refseq_category": "reference genome"},
                "organism": {"organism_name": "Drosophila melanogaster"},
            },
            {
                "accession": "GCF_009650485.2",
                "assembly_info": {"assembly_level": "Chromosome", "refseq_category": "representative genome"},
                "organism": {"organism_name": "Drosophila albomicans"},
            },
            {
                "accession": "GCF_013141755.1",
                "assembly_info": {"assembly_level": "Chromosome", "refseq_category": "representative genome"},
                "organism": {"organism_name": "Scaptodrosophila lebanonensis"},
            },
        ]
    }
    monkeypatch.setattr(hf, "_fetch_json", lambda url, params=None, **kw: fake_reports)
    monkeypatch.setattr(hf, "resolve_ncbi_assembly_dir", lambda acc: (f"{acc}_ASM", f"mock://{acc}"))

    def _fake_dl(url: str, dest: pathlib.Path, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if str(dest).endswith(".faa.gz"):
            with gzip.open(dest, "wt", encoding="utf-8") as fh:
                fh.write(">XP_001.1 hypothetical protein [Species]\nMKTLLLTLVV\n\n>XP_002.1 kinase\nMAAAKKK\n")
        else:
            dest.write_text('>7291_1:0001 {"organism":"Drosophila albomicans"}\nMKTLLLTLVV\n\n>7291_1:0002 {"organism":"Drosophila albomicans"}\nMAAAKKK\n', encoding="utf-8")
        return dest

    monkeypatch.setattr(hf, "_download_file", _fake_dl)
    monkeypatch.delenv("FOCALOOP_NCBI_CACHE_DIR", raising=False)

    res_ncbi = hf.fetch_ncbi_related_proteomes(
        clade="insecta",
        dest_dir=tmp_path / "ncbi_prot",
        max_species=2,
        target_ctx={"accession": "GCF_000001215.4", "species_name": "Drosophila melanogaster", "family_taxid": "7214"},
    )
    assert res_ncbi["n_species"] == 2
    picked_sp = [s["species_name"] for s in res_ncbi["species"]]
    assert "Drosophila melanogaster" not in picked_sp
    assert "Scaptodrosophila lebanonensis" in picked_sp
    faa_txt = res_ncbi["proteins_fasta"].read_text(encoding="utf-8")
    assert "\n\n" not in faa_txt
    assert ">GCF_013141755.1|XP_001.1" in faa_txt

    # 3. Verify OrthoDB v12 representative fetch + blank-line stripping + cached partition fast-path
    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"sciname": "Drosophila melanogaster", "taxon_id": "7227_0", "organism_id": "GCF_000001215.4"},
                    {"sciname": "Scaptodrosophila lebanonensis", "taxon_id": "7260_0", "organism_id": "GCF_013141755.1"},
                    {"sciname": "Chymomyza costata", "taxon_id": "7244_0", "organism_id": "GCA_018152265.1"},
                ]
            }

    class _FakeSess:
        def get(self, url, params=None, timeout=45.0):
            return _FakeResp()

    monkeypatch.setattr(hf, "_ipv4_session", lambda: _FakeSess())
    monkeypatch.delenv("FOCALOOP_ORTHODB_CACHE_DIR", raising=False)
    res_odb = hf.fetch_orthodb_clade_proteins(
        clade="insecta",
        dest_dir=tmp_path / "odb_prot",
        max_species=2,
        target_ctx={"accession": "GCF_000001215.4", "species_name": "Drosophila melanogaster", "family_taxid": "7214"},
    )
    assert res_odb["source"] == "orthodb_v12_representatives"
    assert res_odb["n_species"] == 2
    assert "\n\n" not in res_odb["proteins_fasta"].read_text(encoding="utf-8")

    cache_d = tmp_path / "odb_cache"
    cache_d.mkdir()
    (cache_d / "Arthropoda.fa.gz").write_bytes(b"nonempty")
    monkeypatch.setenv("FOCALOOP_ORTHODB_CACHE_DIR", str(cache_d))
    res_cached = hf.fetch_orthodb_clade_proteins(clade="insecta", dest_dir=tmp_path / "odb_cached")
    assert res_cached["source"] == "orthodb_cached:Arthropoda"


def test_marius_expressionloop_memory_and_stream_guards(tmp_path, monkeypatch):
    import gzip
    import inspect
    from focaloop import galaxy_runners, runners_expression
    from focaloop.panexpr import catalog as cat_mod
    from focaloop.panexpr import kernels as kern_mod
    from focaloop.panexpr import regime as reg_mod
    from focaloop.panexpr.streamer import FastqPrefixStream, GzipStreamDecoder, LocalFileSource

    # 1. ENA HTML redirect / non-gzip payload raises instead of silently yielding 0 reads
    html_payload = b"<html>\n<head><title>301 Moved Permanently</title></head>\n<body>Moved</body>\n</html>\n"
    with pytest.raises(RuntimeError, match="HTML"):
        GzipStreamDecoder(expect_gzip=True).feed_named(html_payload, eof=True)
    with pytest.raises(RuntimeError, match="gzip"):
        GzipStreamDecoder(expect_gzip=True).feed_named(b"plain non-gzip noise\n", eof=True)

    bad_gz = tmp_path / "redirected.fastq.gz"
    bad_gz.write_bytes(html_payload)
    with pytest.raises(RuntimeError, match="HTML"):
        FastqPrefixStream(LocalFileSource(bad_gz)).advance_by_named(1024)

    ok_gz = tmp_path / "valid.fastq.gz"
    with gzip.open(ok_gz, "wb") as fh:
        fh.write(b"@r1/1\nACGTACGT\n+\nIIIIIIII\n")
    recs = FastqPrefixStream(LocalFileSource(ok_gz)).advance_by_named(1024)
    assert len(recs) == 1 and recs[0][0] == "r1"

    # 2. Catalog.release_decoy_sequences() drops resident decoy strings while keeping Target.length & gene seqs
    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "expressionloop"
    cat = cat_mod.build_catalog(
        [cat_mod.AssemblySpec("primary", td / "ref_primary.fa", td / "ref_primary.gtf")],
        "eukaryotic",
        "primary",
        tmp_path / "cat_decoy",
    )
    decoys = [t for t in cat.targets.values() if t.is_decoy]
    genes = [t for t in cat.targets.values() if not t.is_decoy]
    assert decoys and genes
    orig_decoy_lens = [t.length for t in decoys]
    freed = cat.release_decoy_sequences()
    assert freed > 0
    assert all(t.seq == "" for t in decoys)
    assert [t.length for t in decoys] == orig_decoy_lens
    assert all(len(t.seq) == t.length and t.length > 0 for t in genes)

    # 3. map_pooled scales clone_with_threads concurrency by cgroup RAM & accepts 4-tuple holdout batches
    cloned_threads: list[int] = []

    class _FakeCloneKernel:
        supports_pooling = False

        def __init__(self, threads: int = 8) -> None:
            self.threads = threads

        def clone_with_threads(self, threads: int):
            cloned_threads.append(threads)
            return _FakeCloneKernel(threads)

        def map_batch(self, r1, r2, cal=None, trace=False):
            return kern_mod.MapResult(n_reads=len(r1))

    batches4 = [(f"s{i}", [("ACGT", "IIII")], [("TGCA", "IIII")], None) for i in range(4)]

    cloned_threads.clear()
    monkeypatch.setattr(reg_mod, "available_ram_bytes", lambda: 12 << 30)  # 12 GiB -> sequential (1 worker)
    out_12g = kern_mod.map_pooled(_FakeCloneKernel(threads=8), batches4)
    assert len(out_12g) == 4 and cloned_threads == []

    cloned_threads.clear()
    monkeypatch.setattr(reg_mod, "available_ram_bytes", lambda: 20 << 30)  # 20 GiB -> 2 workers x 4 threads
    out_20g = kern_mod.map_pooled(_FakeCloneKernel(threads=8), batches4)
    assert len(out_20g) == 4 and cloned_threads == [4, 4, 4, 4]

    cloned_threads.clear()
    monkeypatch.setattr(reg_mod, "available_ram_bytes", lambda: 64 << 30)  # 64 GiB -> 4 workers x 2 threads
    out_64g = kern_mod.map_pooled(_FakeCloneKernel(threads=8), batches4)
    assert len(out_64g) == 4 and cloned_threads == [2, 2, 2, 2]

    # 4. Regime exclusions (fast_euk + RAM_SAFETY) apply unconditionally when kernel_names is unset
    for mod in (runners_expression, galaxy_runners):
        src = inspect.getsource(mod)
        assert "if not kernel_names and regime_dec.excluded:" in src
        assert "and regime_dec.pseudoaligners_disqualified" not in src
        assert "map_pooled(win_kernel, hold_batches, calibrators)" not in src
