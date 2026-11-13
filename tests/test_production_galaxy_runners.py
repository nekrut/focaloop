"""End-to-end tests for production PanExpr, MalariaVar, reports, C accelerators, and warm Query."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from focaloop.cli import main as cli_main
from focaloop.galaxy_runners import _format_malariavar_drm_table, prebuild_all_native_libs
from focaloop.malariavar import pla_sieve, who_resistance
from focaloop.reports import render_malariavar_report


def test_native_c_accelerators_compile_and_run() -> None:
    built = prebuild_all_native_libs()
    assert len(built) == 9

    rng = np.random.default_rng(7)
    seq = "".join(rng.choice(list("ACGT"), size=512)).encode("ascii")
    syncs = pla_sieve.extract_open_syncmers_u64(seq, k=31, s=15)
    assert isinstance(syncs, np.ndarray)
    assert syncs.dtype == np.uint64
    assert syncs.size > 0


def test_panexpr_dual_assembly_and_warm_query_e2e(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    bases = np.array(list("ACGT"))

    # Generate 12 distinct 1200 bp genes so PyDESeq2 dispersion estimation is well-conditioned
    genes = {f"GENE_{i:02d}": "".join(rng.choice(bases, size=1200)) for i in range(12)}

    # Primary assembly v3 has GENE_00..GENE_11; secondary assembly v2 renumbers GENE_00 -> LEGACY_001458
    v3_fa = tmp_path / "v3.fa"
    v3_gtf = tmp_path / "v3.gtf"
    v2_fa = tmp_path / "v2.fa"
    v2_gtf = tmp_path / "v2.gtf"

    v3_fa.write_text("".join(f">{gid}_chr\n{seq}\n" for gid, seq in genes.items()), encoding="utf-8")
    v3_gtf.write_text(
        "".join(
            f'{gid}_chr\ttest\texon\t1\t1200\t.\t+\t.\tgene_id "{gid}"; transcript_id "{gid}.t1"; gene_name "{gid}";\n'
            for gid in genes
        ),
        encoding="utf-8",
    )

    v2_records = [("LEGACY_001458", genes["GENE_00"])] + [(f"V2_{gid}", seq) for gid, seq in list(genes.items())[1:6]]
    v2_fa.write_text("".join(f">{ gid }_chr\n{seq}\n" for gid, seq in v2_records), encoding="utf-8")
    v2_gtf.write_text(
        "".join(
            f'{gid}_chr\ttest\texon\t1\t1200\t.\t+\t.\tgene_id "{gid}"; transcript_id "{gid}.t1"; gene_name "{gid}";\n'
            for gid, _ in v2_records
        ),
        encoding="utf-8",
    )

    # Synthesize 6 single-end FASTQ.gz samples across 3 conditions: WT, KO1, KO2 (2 replicates each)
    # GENE_00 is strongly downregulated in both KO1 and KO2 relative to WT!
    cond_samples = [
        ("WT_1", "WT", 160),
        ("WT_2", "WT", 150),
        ("KO1_1", "KO1", 8),
        ("KO1_2", "KO1", 10),
        ("KO2_1", "KO2", 6),
        ("KO2_2", "KO2", 7),
    ]
    manifest_rows = ["sample_id\tcondition\trun_or_url"]
    read_paths: list[str] = []
    read_ids: list[str] = []

    for sid, cond, g0_n in cond_samples:
        fq_path = tmp_path / f"{sid}.fastq.gz"
        rec_idx = 0
        lines: list[str] = []
        for gid, seq in genes.items():
            n_reads = g0_n if gid == "GENE_00" else int(rng.poisson(75))
            for _ in range(n_reads):
                pos = int(rng.integers(0, len(seq) - 100))
                frag = seq[pos : pos + 100]
                lines.append(f"@{sid}_{rec_idx}\n{frag}\n+\n{'I' * 100}\n")
                rec_idx += 1
        with gzip.open(fq_path, "wt", encoding="utf-8") as gz:
            gz.writelines(lines)
        manifest_rows.append(f"{sid}\t{cond}\t{sid}")
        read_paths.append(str(fq_path))
        read_ids.append(sid)

    manifest_tsv = tmp_path / "manifest.tsv"
    manifest_tsv.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    out_de = tmp_path / "de_results.tsv"
    out_report = tmp_path / "focaloop_report.html"
    state_dir = tmp_path / "state_dir"
    state_archive = tmp_path / "panexpr_state.focaloop.tar.zst"

    rc = cli_main(
        [
            "panexpr",
            "--manifest", str(manifest_tsv),
            "--ref-fasta", str(v3_fa),
            "--ref-gtf", str(v3_gtf),
            "--secondary-fasta", str(v2_fa),
            "--secondary-gtf", str(v2_gtf),
            "--primary-label", "v3",
            "--secondary-label", "v2",
            "--reads", *read_paths,
            "--read-ids", *read_ids,
            "--contrast", "condition:KO1:WT",
            "--organism-mode", "eukaryotic",
            "--source-mode", "local_bytes",
            "--max-byte-fraction", "1.0",
            "--no-zoom",
            "--kernels", "mappy", "bwa_spa",
            "--tournament-reads", "1500",
            "--threads", "2",
            "--state-dir", str(state_dir),
            "--state-archive", str(state_archive),
            "--out-de", str(out_de),
            "--out-report", str(out_report),
        ]
    )
    assert rc == 0
    assert out_de.is_file()
    assert out_report.is_file()
    assert state_archive.is_file()

    de_df = pd.read_csv(out_de, sep="\t")
    g0 = de_df[de_df["gene_id"] == "GENE_00"].iloc[0]
    assert g0["log2FoldChange"] < -2.5
    assert g0["padj"] < 1e-3
    assert "v2:LEGACY_001458" in str(g0["ortholog_bridge"])
    html = out_report.read_text(encoding="utf-8")
    assert "Focaloop PanExpr" in html and ("mappy" in html or "bwa_spa" in html)

    # Follow-up Warm Query for condition:KO2:WT via --args-json with blocked network sockets
    args_json = tmp_path / "query_args.json"
    args_json.write_text(
        json.dumps({"contrast": "condition:KO2:WT", "actor": "human", "hypothesis": "", "design": ""}),
        encoding="utf-8",
    )
    q_table = tmp_path / "query_table.tsv"
    q_verdict = tmp_path / "verdict.json"
    q_report = tmp_path / "query_report.html"
    q_state = tmp_path / "updated_state.focaloop.tar.zst"

    qrc = cli_main(
        [
            "query",
            "--state-dir", str(state_archive),
            "--args-json", str(args_json),
            "--mode", "auto",
            "--out-table", str(q_table),
            "--out-verdict", str(q_verdict),
            "--out-report", str(q_report),
            "--out-state-archive", str(q_state),
        ]
    )
    assert qrc == 0
    verdict = json.loads(q_verdict.read_text(encoding="utf-8"))
    assert verdict["new_bytes"] == 0
    assert verdict["network_sockets_blocked"] is True
    q_df = pd.read_csv(q_table, sep="\t")
    qg0 = q_df[q_df["gene_id"] == "GENE_00"].iloc[0]
    assert qg0["log2FoldChange"] < -2.5
    assert "0 B" in q_report.read_text(encoding="utf-8")


def test_who_resistance_classifier_and_malariavar_report(tmp_path: Path) -> None:
    calls_sea = who_resistance.classify_sample(
        {
            "k13_ns": "C580Y",
            "crt_76": "T",
            "dhfr_51": "I",
            "dhfr_59": "R",
            "dhfr_108": "N",
            "dhfr_164": "I",
            "dhps_437": "G",
            "dhps_540": "E",
            "dhps_581": "A",
            "dhps_613": "A",
            "mdr1_amp": 1,
            "pm2_pm3_amp": 1,
        }
    )
    assert calls_sea["Artemisinin"] == "Resistant"
    assert calls_sea["Chloroquine"] == "Resistant"
    assert calls_sea["Piperaquine"] == "Resistant"
    assert calls_sea["DHA-PPQ"] == "Resistant"
    assert calls_sea["Mefloquine"] == "Resistant"
    assert calls_sea["SP (uncomplicated)"] == "Resistant"

    tel_df = pd.DataFrame(
        [
            {
                "sample": "PH1387-C",
                "source": "sanger_crai_focused",
                "stop_reason": "crai_windows_complete",
                "wire_bytes": 21_500_000,
                "archive_bytes": 1_650_000_000,
                "host_frac": 0.0,
                "median_core_depth": 42.0,
                "fws": 0.992,
                "k13_ns": "C580Y",
                "aa_crt_76": "T",
                "aa_mdr1_86": "N",
                "aa_mdr1_184": "F",
                "aa_dhfr_51": "I",
                "aa_dhfr_59": "R",
                "aa_dhfr_108": "N",
                "aa_dhfr_164": "I",
                "aa_dhps_436": "S",
                "aa_dhps_437": "G",
                "aa_dhps_540": "E",
                "aa_dhps_581": "A",
                "aa_dhps_613": "A",
            },
            {
                "sample": "PF1002-C",
                "source": "sanger_crai_focused",
                "stop_reason": "crai_windows_complete",
                "wire_bytes": 19_200_000,
                "archive_bytes": 1_420_000_000,
                "host_frac": 0.0,
                "median_core_depth": 38.0,
                "fws": 0.742,
                "k13_ns": "",
                "aa_crt_76": "K",
                "aa_mdr1_86": "N",
                "aa_mdr1_184": "Y",
                "aa_dhfr_51": "N",
                "aa_dhfr_59": "C",
                "aa_dhfr_108": "S",
                "aa_dhfr_164": "I",
                "aa_dhps_436": "S",
                "aa_dhps_437": "A",
                "aa_dhps_540": "K",
                "aa_dhps_581": "A",
                "aa_dhps_613": "A",
            },
        ]
    )
    cnv_df = pd.DataFrame(
        [
            {"sample": "PH1387-C", "locus": "mdr1", "call": 1},
            {"sample": "PH1387-C", "locus": "pm2_pm3", "call": 1},
            {"sample": "PH1387-C", "locus": "hrp2", "call": 0},
            {"sample": "PF1002-C", "locus": "mdr1", "call": 0},
            {"sample": "PF1002-C", "locus": "pm2_pm3", "call": 0},
            {"sample": "PF1002-C", "locus": "hrp2", "call": 0},
        ]
    )
    meta_df = pd.DataFrame(
        [
            {"Sample": "PH1387-C", "Country": "Cambodia", "Population": "AS-SE-E"},
            {"Sample": "PF1002-C", "Country": "Ghana", "Population": "AF-W"},
        ]
    )
    drm_df = _format_malariavar_drm_table(tel_df, cnv_df, meta_df)
    assert drm_df.loc[0, "who_artemisinin"] == "Resistant"
    assert drm_df.loc[0, "clonality"] == "monoclonal"
    assert drm_df.loc[1, "who_artemisinin"] == "Sensitive"
    assert drm_df.loc[1, "clonality"] == "polyclonal"

    rep = tmp_path / "mv_report.html"
    render_malariavar_report(rep, mode="focused_cram", drm_df=drm_df, summary_stats={"wall_s": 4.2, "n_variants": 120})
    txt = rep.read_text(encoding="utf-8")
    assert "Focaloop MalariaVar" in txt and "PH1387-C" in txt and "C580Y" in txt


def test_malariavar_mapper_falls_back_on_non_avx2(monkeypatch: pytest.MonkeyPatch) -> None:
    import focaloop.galaxy_runners as gr

    monkeypatch.setattr(gr, "_cpu_has_avx2", lambda: False)
    monkeypatch.setattr(gr.shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("strobealign", "minimap2") else None)
    assert gr._select_malariavar_mapper("strobealign") == "minimap2"
    monkeypatch.setattr(
        gr.subprocess,
        "run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )
    assert gr._select_malariavar_mapper("auto") == "minimap2"
