"""Unit and integration tests for CVMFS built-in genome auto-annotation and species inference."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from focaloop import cvmfs_annot as ca
from focaloop.runners_expression import run_expressionloop_galaxy

REPO = Path(__file__).resolve().parents[1]
TEST_DATA = REPO / "tools" / "focaloop" / "test-data" / "expressionloop"


def test_read_fasta_contig_lengths_fai_sam_index_and_gz(tmp_path: Path) -> None:
    # Direct .fai sidecar
    fa1 = tmp_path / "g1.fa"
    fa1.write_text(">dummy\nACGT\n", encoding="utf-8")
    (tmp_path / "g1.fa.fai").write_text("chr1\t248956422\t6\t50\t51\nchr2\t242193529\t0\t50\t51\n", encoding="utf-8")
    assert ca.read_fasta_contig_lengths(fa1) == {"chr1": 248956422, "chr2": 242193529}

    # Galaxy byhand/<db>/sam_index/<name>.fa.fai layout
    seq_dir = tmp_path / "hg38" / "seq"
    sam_dir = tmp_path / "hg38" / "sam_index"
    seq_dir.mkdir(parents=True)
    sam_dir.mkdir(parents=True)
    fa2 = seq_dir / "hg38canon.fa"
    fa2.write_text(">placeholder\nA\n", encoding="utf-8")
    (sam_dir / "hg38canon.fa.fai").write_text("chr1\t248956422\t6\t50\t51\nchrM\t16569\t0\t50\t51\n", encoding="utf-8")
    assert ca.read_fasta_contig_lengths(fa2) == {"chr1": 248956422, "chrM": 16569}

    # Gzipped FASTA fallback
    fa3 = tmp_path / "tiny.fa.gz"
    with gzip.open(fa3, "wt", encoding="utf-8") as fh:
        fh.write(">chrA desc\nACGTACGT\nAA\n>chrB\nGGG\n")
    assert ca.read_fasta_contig_lengths(fa3) == {"chrA": 10, "chrB": 3}


def test_infer_species_from_cached_meta_ucsc_plants_and_microbes() -> None:
    fake_ucsc = {
        "hg38": {"scientificName": "Homo sapiens", "sourceName": "GRCh38 (GCA_000001405.15)"},
        "dm6": {"scientificName": "Drosophila melanogaster", "sourceName": "BDGP Release 6 (GCA_000001215.4)"},
        "sacCer3": {"scientificName": "Saccharomyces cerevisiae", "sourceName": "S288c (GCA_000146045.2)"},
    }

    m_hg = ca.infer_species_from_cached_meta(
        dbkey="hg38", display_name="Human (Homo sapiens): hg38 Canonical", value="hg38canon", ucsc_genomes=fake_ucsc
    )
    assert m_hg["ucsc_dbkey"] == "hg38"
    assert m_hg["species_name"] == "Homo sapiens"
    assert m_hg["ncbi_accession_hint"] == "GCA_000001405.15"

    m_at = ca.infer_species_from_cached_meta(
        dbkey="Arabidopsis_thaliana_TAIR10",
        display_name="Arabidopsis thaliana: Arabidopsis_thaliana_TAIR10",
        value="Arabidopsis_thaliana_TAIR10",
        ucsc_genomes=fake_ucsc,
    )
    assert m_at["ucsc_dbkey"] == ""
    assert m_at["species_name"] == "Arabidopsis thaliana"
    assert m_at["assembly_hint"] == "TAIR10"

    m_ec = ca.infer_species_from_cached_meta(
        dbkey="eschColi_K12",
        display_name="Escherichia coli (str. K-12 substr. MG1655): eschColi_K12",
        value="eschColi_K12",
        ucsc_genomes=fake_ucsc,
    )
    assert m_ec["species_name"] == "Escherichia coli"
    assert m_ec["strain"] == "MG1655"

    # Real Galaxy CVMFS typo: "Pseudomonas aeruginos (str. PAO1): pseuAeru"
    m_pa = ca.infer_species_from_cached_meta(
        dbkey="pseuAeru",
        display_name="Pseudomonas aeruginos (str. PAO1): pseuAeru",
        value="pseuAeru_PAO1",
        ucsc_genomes=fake_ucsc,
    )
    assert m_pa["species_name"] == "Pseudomonas aeruginosa"
    assert m_pa["strain"] == "PAO1"

    m_sp = ca.infer_species_from_cached_meta(
        dbkey="Schizosaccharomyces_pombe_1.1",
        display_name="Fission Yeast (Schizosaccharomyces pombe): Schizosaccharomyces_pombe_1.1",
        value="Schizosaccharomyces_pombe_1.1",
        ucsc_genomes=fake_ucsc,
    )
    assert m_sp["species_name"] == "Schizosaccharomyces pombe"


def test_exact_contig_map_and_gff3_remap_tair10_and_microbe(tmp_path: Path) -> None:
    report_txt = "\n".join(
        [
            "# Sequence-Name\tSequence-Role\tAssigned-Molecule\tType\tGenBank-Accn\tRel\tRefSeq-Accn\tUnit\tSequence-Length\tUCSC-style-name",
            "1\tassembled-molecule\t1\tChromosome\tCP002684.1\t=\tNC_003070.9\tPrimary\t30427671\tna",
            "MT\tassembled-molecule\tMT\tMitochondrion\tna\t<>\tNC_001284.2\tnon-nuclear\t366924\tna",
            "Pltd\tassembled-molecule\tPltd\tChloroplast\tna\t<>\tNC_000932.1\tnon-nuclear\t154478\tna",
        ]
    )
    rows = ca._parse_assembly_report(report_txt)
    fa_lens = {"chr1": 30427671, "chrM": 366924, "chrC": 154478}
    cmap, bp_frac, ctg_frac, mm = ca._build_exact_contig_map(rows, fa_lens)
    assert bp_frac == 1.0 and ctg_frac == 1.0 and not mm
    assert cmap["NC_003070.9"] == "chr1"
    assert cmap["NC_001284.2"] == "chrM"
    assert cmap["NC_000932.1"] == "chrC"

    # Single-chromosome microbe with exact length matches CVMFS 'chr'
    mic_rows = ca._parse_assembly_report(
        "ANONYMOUS\tassembled-molecule\tna\tChromosome\tAE004091.2\t=\tNC_002516.2\tPrimary\t6264404\tna\n"
    )
    cmap_ok, bp_ok, ctg_ok, _ = ca._build_exact_contig_map(mic_rows, {"chr": 6264404})
    assert bp_ok == 1.0 and ctg_ok == 1.0
    assert cmap_ok["NC_002516.2"] == "chr"

    # Shifted length (e.g. old E. coli revision 4639675 vs RefSeq 4641652) MUST be rejected
    ec_rows = ca._parse_assembly_report(
        "U00096.3\tassembled-molecule\tna\tChromosome\tU00096.3\t=\tNC_000913.3\tPrimary\t4641652\tna\n"
    )
    cmap_bad, bp_bad, _, mm_bad = ca._build_exact_contig_map(ec_rows, {"chr": 4639675})
    assert bp_bad == 0.0 and not cmap_bad
    assert mm_bad and "4,639,675" in mm_bad[0] and "4,641,652" in mm_bad[0]

    # GFF3 remap preserves coordinates and passes verification
    raw_gff = tmp_path / "in.gff.gz"
    with gzip.open(raw_gff, "wt", encoding="utf-8") as fh:
        fh.write("##gff-version 3\n##sequence-region NC_002516.2 1 6264404\n")
        for i in range(12):
            s = 100 + i * 200
            fh.write(f"NC_002516.2\tRefSeq\tgene\t{s}\t{s+100}\t.\t+\t.\tID=gene-{i};locus_tag=PA{i:04d}\n")
            fh.write(f"NC_002516.2\tRefSeq\tCDS\t{s}\t{s+100}\t.\t+\t0\tID=cds-{i};Parent=gene-{i};locus_tag=PA{i:04d}\n")
    out_gff = tmp_path / "remapped.gff3"
    rstats = ca._remap_gff3_contigs(raw_gff, out_gff, cmap_ok)
    assert rstats["raw_total_genes"] == 12 and rstats["kept_genes"] == 12
    chk = ca.verify_annotation_against_fasta(out_gff, {"chr": 6264404}, raw_total_genes=rstats["raw_total_genes"])
    assert chk["ok"] is True and chk["n_oob"] == 0 and chk["matched_gene_frac"] == 1.0


def test_resolve_cvmfs_annotation_ucsc_happy_and_fail_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fa = tmp_path / "sacCer3.fa"
    fa.write_text(">chrI\n" + "A" * 5000 + "\n>chrII\n" + "C" * 8000 + "\n", encoding="utf-8")

    gtf_payload = "\n".join(
        f'chrI\tncbiRefSeq\texon\t{100 + i * 200}\t{180 + i * 200}\t.\t+\t.\tgene_id "G{i}"; transcript_id "T{i}"; gene_name "G{i}";'
        for i in range(12)
    ) + "\n"

    monkeypatch.setattr(ca, "fetch_ucsc_genomes", lambda timeout=20.0: {"sacCer3": {"scientificName": "Saccharomyces cerevisiae"}})
    monkeypatch.setattr(ca, "fetch_ucsc_chrom_lengths", lambda db, timeout=20.0: {"chrI": 5000, "chrII": 8000})
    monkeypatch.setattr(ca, "_url_exists", lambda url, timeout=15.0: url.endswith("sacCer3.ncbiRefSeq.gtf.gz"))

    def _fake_dl(url: str, dest: Path, **kw) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(gtf_payload)
        return dest

    monkeypatch.setattr(ca, "_download_file", _fake_dl)

    res = ca.resolve_cvmfs_annotation(fa, tmp_path / "out_ok", dbkey="sacCer3", display_name="Yeast: sacCer3", value="sacCer3")
    assert res["source"] == "ucsc:sacCer3:ncbiRefSeq"
    assert res["species_name"] == "Saccharomyces cerevisiae"
    assert Path(res["gtf_path"]).is_file()

    # Now introduce a chromosome length mismatch -> must fail fast (never silently shift coordinates)
    monkeypatch.setattr(ca, "fetch_ucsc_chrom_lengths", lambda db, timeout=20.0: {"chrI": 5100, "chrII": 8000})
    monkeypatch.setattr(ca, "_discover_candidate_gcf_bases", lambda meta, max_candidates=5: [])
    with pytest.raises(ValueError, match="Chromosome length mismatch"):
        ca.resolve_cvmfs_annotation(fa, tmp_path / "out_bad", dbkey="sacCer3", display_name="Yeast: sacCer3", value="sacCer3")


def test_resolve_cvmfs_annotation_ncbi_picks_exact_length_version_and_fails_on_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fa = tmp_path / "tair10.fa"
    fa.write_text(">chr1\n" + "A" * 5000 + "\n>chrM\n" + "G" * 1000 + "\n", encoding="utf-8")

    monkeypatch.setattr(ca, "fetch_ucsc_genomes", lambda timeout=20.0: {})
    monkeypatch.setattr(ca, "_discover_candidate_gcf_bases", lambda meta, max_candidates=5: [("GCF_000001735", "GCF_000001735.4")])
    monkeypatch.setattr(
        ca,
        "_list_ncbi_assembly_versions",
        lambda base: [
            (4, "GCF_000001735.4_TAIR10.1", "https://example.org/v4"),
            (3, "GCF_000001735.3_TAIR10", "https://example.org/v3"),
        ],
    )

    rep_v4 = (
        "1\tassembled-molecule\t1\tChr\tCP1\t=\tNC_1\tPrimary\t5000\tna\n"
        "MT\tassembled-molecule\tMT\tMito\tBK1\t=\tNC_MT_V4\tNon\t1200\tna\n"
    )
    rep_v3 = (
        "1\tassembled-molecule\t1\tChr\tCP1\t=\tNC_1\tPrimary\t5000\tna\n"
        "MT\tassembled-molecule\tMT\tMito\tna\t<>\tNC_MT_V3\tNon\t1000\tna\n"
    )
    monkeypatch.setattr(ca, "_fetch_text", lambda url, **kw: rep_v4 if "/v4/" in url else rep_v3)
    monkeypatch.setattr(ca, "_url_exists", lambda url, timeout=15.0: True)

    def _fake_dl(url: str, dest: Path, **kw) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            fh.write("##gff-version 3\n")
            for i in range(10):
                s = 50 + i * 200
                fh.write(f"NC_1\tRefSeq\tgene\t{s}\t{s+90}\t.\t+\t.\tID=g{i};locus_tag=AT{i}\n")
            fh.write("NC_MT_V3\tRefSeq\tgene\t50\t200\t.\t+\t.\tID=gmt;locus_tag=ATM1\n")
        return dest

    monkeypatch.setattr(ca, "_download_file", _fake_dl)

    res = ca.resolve_cvmfs_annotation(
        fa,
        tmp_path / "out_v3",
        dbkey="Arabidopsis_thaliana_TAIR10",
        display_name="Arabidopsis thaliana: Arabidopsis_thaliana_TAIR10",
        value="Arabidopsis_thaliana_TAIR10",
    )
    assert res["source"] == "ncbi:GCF_000001735.3_TAIR10"
    assert res["ncbi_accession"] == "GCF_000001735.3"
    assert res["matched_bp_frac"] == 1.0
    remapped_txt = Path(res["gtf_path"]).read_text(encoding="utf-8")
    assert "chr1\tRefSeq\tgene" in remapped_txt
    assert "chrM\tRefSeq\tgene" in remapped_txt

    # When matching older revision has 404 GFF3 and newer revision has shifted length -> fail fast with recommendation
    monkeypatch.setattr(ca, "_url_exists", lambda url, timeout=15.0: "/v4/" in url)
    rep_shifted = "1\tassembled-molecule\t1\tChr\tCP1\t=\tNC_1\tPrimary\t5500\tna\n"
    monkeypatch.setattr(ca, "_fetch_text", lambda url, **kw: rep_shifted if "/v4/" in url else rep_v3)
    with pytest.raises(ValueError, match="GCF_000001735.4"):
        ca.resolve_cvmfs_annotation(
            fa,
            tmp_path / "out_404",
            dbkey="Arabidopsis_thaliana_TAIR10",
            display_name="Arabidopsis thaliana: Arabidopsis_thaliana_TAIR10",
            value="Arabidopsis_thaliana_TAIR10",
        )


def test_runner_cvmfs_auto_gtf_extra_cvmfs_and_species_auto_expand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import focaloop.ncbi_ref as ncbi_mod

    # Mock CVMFS auto-annotation for both primary and extra cached reference
    calls: list[str] = []

    def _fake_resolve(fasta_path, dest_dir, *, dbkey="", display_name="", value="", log=None):
        calls.append(f"{dbkey}:{value}")
        if dbkey == "test_primary":
            return {
                "gtf_path": TEST_DATA / "ref_primary.gtf",
                "source": "ucsc:test_primary:ncbiRefSeq",
                "species_name": "Escherichia coli",
                "strain": "MG1655",
                "ncbi_accession": "GCF_000005845.2",
            }
        return {
            "gtf_path": TEST_DATA / "ref_secondary.gtf",
            "source": "ncbi:test_extra",
            "species_name": "Escherichia coli",
            "strain": "W3110",
            "ncbi_accession": "GCF_000010245.2",
        }

    monkeypatch.setattr(ca, "resolve_cvmfs_annotation", _fake_resolve)

    captured_expand: dict[str, object] = {}

    def _fake_discover(taxon_or_species, *, max_strains=3, exclude_accessions=(), exclude_strains=(), candidate_pool=12):
        captured_expand["taxon"] = taxon_or_species
        captured_expand["exclude_accessions"] = list(exclude_accessions)
        captured_expand["exclude_strains"] = list(exclude_strains)
        return []  # no network fetch needed, verifies species inference & exclusion wiring

    monkeypatch.setattr(ncbi_mod, "discover_related_ncbi_assemblies", _fake_discover)

    reads = [str(TEST_DATA / f"{s}.fastq.gz") for s in ("ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3")]
    read_ids = ["ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3"]
    out_de = tmp_path / "de.tsv"
    out_report = tmp_path / "rep.html"

    meta = run_expressionloop_galaxy(
        manifest_path=TEST_DATA / "manifest_se.tsv",
        ref_fasta=TEST_DATA / "ref_primary.fa",
        ref_dbkey="test_primary",
        ref_name="Escherichia coli (str. K-12 substr. MG1655): test_primary",
        ref_value="test_primary",
        auto_ref_gtf=True,
        extra_references=[
            {
                "label": "extra_cached",
                "fasta": str(TEST_DATA / "ref_secondary.fa"),
                "gtf": "",
                "auto_gtf": True,
                "dbkey": "test_extra",
                "ref_name": "Escherichia coli (str. K-12 substr. W3110): test_extra",
                "value": "test_extra",
            }
        ],
        auto_expand_strains=2,
        auto_expand_taxon="",  # MUST infer automatically from CVMFS primary metadata!
        reads=reads,
        read_ids=read_ids,
        contrast="condition:treated:control",
        organism_mode="eukaryotic",
        source_mode="local_bytes",
        max_byte_fraction=1.0,
        enable_zoom=False,
        threads=2,
        kernel_names=["mappy"],
        n_tournament_reads=600,
        state_dir=tmp_path / "state",
        out_de=out_de,
        out_report=out_report,
    )
    assert calls == ["test_primary:test_primary", "test_extra:test_extra"]
    assert captured_expand["taxon"] == "Escherichia coli"
    assert "GCF_000005845.2" in captured_expand["exclude_accessions"]
    assert "GCF_000010245.2" in captured_expand["exclude_accessions"]
    assert "MG1655" in captured_expand["exclude_strains"]
    assert out_de.is_file() and meta["engine"] == "panexpr"
    de_df = pd.read_csv(out_de, sep="\t")
    assert "extra_cached:LEGACY_001458" in str(de_df[de_df["gene_id"] == "GENE_00"].iloc[0]["ortholog_bridge"])


def test_discover_related_excludes_paired_gca_gcf(monkeypatch: pytest.MonkeyPatch) -> None:
    import focaloop.ncbi_ref as nr

    fake_payload = {
        "reports": [
            {
                "accession": "GCF_000146045.2",
                "organism": {"infraspecific_names": {"strain": "S288C"}},
                "assembly_info": {"assembly_name": "R64", "assembly_level": "Complete Genome", "refseq_category": "reference genome"},
            },
            {
                "accession": "GCF_000149999.1",
                "organism": {"infraspecific_names": {"strain": "RM11-1a"}},
                "assembly_info": {"assembly_name": "RM11", "assembly_level": "Complete Genome", "refseq_category": ""},
            },
        ]
    }
    monkeypatch.setattr(nr, "_fetch_json", lambda url, params=None, **kw: fake_payload)
    res = nr.discover_related_ncbi_assemblies("Saccharomyces cerevisiae", max_strains=2, exclude_accessions=["GCA_000146045.2"])
    assert [r["accession"] for r in res] == ["GCF_000149999.1"]

