"""Cross-worktree integration tests for Focaloop 0.4.1 (ExpressionLoop/VariationLoop -> Focaloop Query + packaging)."""
from __future__ import annotations

import gzip
import json
import pathlib
import subprocess
import sys

import pytest

from focaloop.ops import run_query
from focaloop.runners_expression import run_expressionloop_galaxy
from focaloop.runners_variation import run_variationloop_galaxy

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_cross_worktree_expressionloop_holdout_and_zoom_query_roundtrip(tmp_path: pathlib.Path) -> None:
    """Verify ExpressionLoop (wt2) holdout + zoom_context round-trip cleanly into Focaloop Query (wt1)."""
    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "expressionloop"
    # Create padded >200 KB compressed FASTQs with unmapped noise reads so Stage-5 zoom & >128 KB holdout both fire
    manifest_rows = ["sample_id\tcondition"]
    reads: list[str] = []
    read_ids: list[str] = []
    src_map = [
        ("ctrl1", "control", td / "ctrl1.fastq.gz"),
        ("ctrl2", "control", td / "ctrl2.fastq.gz"),
        ("ctrl3", "control", td / "ctrl3.fastq.gz"),
        ("trt1", "treated", td / "trt1.fastq.gz"),
        ("trt2", "treated", td / "trt2.fastq.gz"),
        ("trt3", "treated", td / "trt3.fastq.gz"),
    ]
    for idx, (sid, cond, src_gz) in enumerate(src_map):
        raw_fq = gzip.decompress(src_gz.read_bytes())
        # Repeat reads with distinct headers so compressed file exceeds ~250 KB (leaving >128 KiB unread at max_bytes_pct=0.35)
        chunks = []
        for rep in range(30):
            for ln in raw_fq.splitlines(keepends=True):
                if ln.startswith(b"@"):
                    chunks.append(f"@r_{idx}_{rep}_".encode("ascii") + ln[1:])
                else:
                    chunks.append(ln)
        out_gz = tmp_path / f"{sid}.fastq.gz"
        with gzip.open(out_gz, "wb", compresslevel=1) as fh:
            fh.write(b"".join(chunks))
        assert out_gz.stat().st_size > 200_000
        reads.append(str(out_gz))
        read_ids.append(sid)
        manifest_rows.append(f"{sid}\t{cond}")

    manifest_p = tmp_path / "manifest.tsv"
    manifest_p.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    state_arc = tmp_path / "expr_041.focaloop.tar.zst"
    state_dir = tmp_path / "expr_state"

    meta = run_expressionloop_galaxy(
        manifest_path=manifest_p,
        ref_fasta=td / "ref_primary.fa",
        ref_gtf=td / "ref_primary.gtf",
        reads=reads,
        read_ids=read_ids,
        contrast="condition:treated:control",
        max_bytes_pct=0.35,
        out_de=tmp_path / "de.tsv",
        out_report=tmp_path / "rep.html",
        state_dir=state_dir,
        state_archive=state_arc,
        kernel_names=["mappy"],
        n_tournament_reads=400,
        threads=2,
    )
    assert meta.get("holdout_saved") is True
    assert (state_dir / "panexpr_state" / "target_counts_holdout.parquet").is_file() or (
        state_dir / "target_counts_holdout.parquet"
    ).is_file()

    q_tbl = tmp_path / "q_de.tsv"
    q_ver = tmp_path / "q_verdict.json"
    q_rep = tmp_path / "q_rep.html"
    run_query(state_arc, "condition:treated:control", "human", q_tbl, q_ver, mode="contrast", out_report=q_rep)
    verdict = json.loads(q_ver.read_text(encoding="utf-8"))
    assert verdict.get("holdout_available") is True
    assert verdict["holdout_estimate"] != verdict["discovery_estimate"]


def test_cross_worktree_variationloop_nocall_and_diploid_fst_roundtrip(tmp_path: pathlib.Path) -> None:
    """Verify VariationLoop (wt3) site_nocalls.parquet + diploid 2N-1 Fst round-trip cleanly into Query (wt1)."""
    td = REPO_ROOT / "tools" / "focaloop" / "test-data" / "variationloop"
    # Build a 2-contig reference where contig_b has zero reads in group_b2 so site_nocalls excludes uncovered sites
    ref_txt = (td / "ref_chr20_mini.fa").read_text(encoding="utf-8")
    seq_lines = [ln.strip() for ln in ref_txt.splitlines() if not ln.startswith(">")]
    seq_a = "".join(seq_lines)
    ref2 = tmp_path / "ref_two_ctg.fa"
    ref2.write_text(f">chr20_mini\n{seq_a}\n>chr_extra\n{seq_a[:3000]}\n", encoding="utf-8")

    # Mutate a read set on chr_extra for group A only while group B has reads ONLY on chr20_mini
    mut_seq = list(seq_a[500:650])
    mut_seq[70] = "T" if mut_seq[70] != "T" else "G"
    mut_r1 = "".join(mut_seq)
    rc_map = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    mut_r2 = seq_a[750:900].translate(rc_map)[::-1]

    def _make_pair(prefix: str, include_extra: bool) -> tuple[pathlib.Path, pathlib.Path]:
        r1_raw = gzip.decompress((td / "sample_R1.fastq.gz").read_bytes())
        r2_raw = gzip.decompress((td / "sample_R2.fastq.gz").read_bytes())
        extra1 = []
        extra2 = []
        if include_extra:
            for i in range(40):
                extra1.append(f"@extra_{i}/1\n{mut_r1}\n+\n{'I'*len(mut_r1)}\n".encode("ascii"))
                extra2.append(f"@extra_{i}/2\n{mut_r2}\n+\n{'I'*len(mut_r2)}\n".encode("ascii"))
        p1 = tmp_path / f"{prefix}_R1.fastq.gz"
        p2 = tmp_path / f"{prefix}_R2.fastq.gz"
        with gzip.open(p1, "wb") as f1, gzip.open(p2, "wb") as f2:
            f1.write(r1_raw + b"".join(extra1))
            f2.write(r2_raw + b"".join(extra2))
        return p1, p2

    a1_r1, a1_r2 = _make_pair("a1", True)
    a2_r1, a2_r2 = _make_pair("a2", True)
    b1_r1, b1_r2 = _make_pair("b1", False)
    b2_r1, b2_r2 = _make_pair("b2", False)

    groups_tsv = tmp_path / "groups.tsv"
    groups_tsv.write_text(
        "sample_id\tgroup\na1\tpopA\na2\tpopA\nb1\tpopB\nb2\tpopB\n",
        encoding="utf-8",
    )
    state_arc = tmp_path / "var_041.focaloop.tar.zst"
    state_dir = tmp_path / "var_state"
    run_variationloop_galaxy(
        analysis_mode="diploid_wgs",
        reads_r1=[a1_r1, a2_r1, b1_r1, b2_r1],
        reads_r2=[a1_r2, a2_r2, b1_r2, b2_r2],
        read_ids=["a1", "a2", "b1", "b2"],
        manifest_path=groups_tsv,
        ref_fasta=ref2,
        threads=2,
        state_dir=state_dir,
        state_archive=state_arc,
        out_variants=tmp_path / "var.tsv",
        out_vcf=tmp_path / "var.vcf",
        out_report=tmp_path / "var.html",
    )
    assert (state_dir / "site_nocalls.parquet").is_file()

    q_tbl = tmp_path / "fst.tsv"
    q_ver = tmp_path / "fst_verdict.json"
    run_query(state_arc, "group=popA:popB", "human", q_tbl, q_ver, mode="fst", out_report=tmp_path / "fst.html")
    fst_txt = q_tbl.read_text(encoding="utf-8")
    # chr_extra site was uncovered (DP=0) in popB, so it must NOT appear as a bogus F_ST = 1.0 site!
    assert "chr_extra" not in fst_txt
    verdict = json.loads(q_ver.read_text(encoding="utf-8"))
    assert verdict.get("holdout_available") is False
    assert verdict.get("confirmed") is False


def test_clean_git_archive_wheel_and_validate_blueprint(tmp_path: pathlib.Path) -> None:
    """Verify git archive includes Makefile, build_native.py, and builtin blueprints (L102)."""
    tar_p = tmp_path / "src.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar_p), "HEAD"], cwd=str(REPO_ROOT), check=True)
    src_dir = tmp_path / "extracted"
    src_dir.mkdir()
    subprocess.run(["tar", "-xf", str(tar_p), "-C", str(src_dir)], check=True)
    assert (src_dir / "Makefile").is_file()
    assert (src_dir / "scripts" / "build_native.py").is_file()

    res = subprocess.run(
        [sys.executable, "-m", "focaloop.cli", "validate", "panexpr"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
