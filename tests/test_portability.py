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
