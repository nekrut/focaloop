"""Comprehensive regression test suite for Focaloop 0.4.1 core, Warm Query, and packaging fixes.

Covers all Worktree 1 (fix/core-query-packaging) review items:
  Important: #14, #61, #62, #63, #64, #65, #66, #67, #68, #69, #74
  Lesser:    #15, #18, L11, L29, L65, L66, L67, L68, L69, L70, L71, L72, L73,
             L95, L98, L99, L100, L101, L102
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from focaloop import _native_cache as nc
from focaloop import archive, cli, cvmfs_annot, galaxy_runners, logan, ncbi_ref
from focaloop.engine import STATE_META, write_json
from focaloop.hooks import HypothesisLedger
from focaloop.ops import run_query
from focaloop.panexpr.catalog import AssemblySpec, build_catalog
from focaloop.panexpr.state import StateStore as PanStateStore, TARGET_COUNTS_SCHEMA
from focaloop.state import FocaloopStateStore
from focaloop.streamer import FastqBlockReader, LocalFileSource

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "tools" / "focaloop" / "test-data"


# ============================================================================
# Helpers for synthetic 0.4.x state bundles
# ============================================================================

VAR_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("chrom", pa.string()),
        ("pos", pa.int64()),
        ("ref", pa.string()),
        ("alt", pa.string()),
        ("kind", pa.string()),
        ("gt", pa.string()),
        ("gq", pa.float64()),
        ("dp", pa.float64()),
        ("ad_ref", pa.float64()),
        ("ad_alt", pa.string()),
        ("wsaf", pa.float64()),
        ("qual", pa.float64()),
        ("source", pa.string()),
    ]
)


def _make_variation_state(
    state_dir: Path,
    samples: list[dict],
    variant_rows: list[dict],
    *,
    ploidy: str = "diploid",
    nocall_rows: list[dict] | None = None,
) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(str(state_dir / "variants.parquet"), VAR_SCHEMA, compression="zstd") as writer:
        if variant_rows:
            by_samp: dict[str, list[dict]] = {}
            for r in variant_rows:
                row_full = {
                    "sample_id": str(r["sample_id"]),
                    "chrom": str(r.get("chrom", "chr1")),
                    "pos": int(r["pos"]),
                    "ref": str(r.get("ref", "A")),
                    "alt": str(r.get("alt", "G")),
                    "kind": str(r.get("kind", "SNV")),
                    "gt": str(r.get("gt", "1/1")),
                    "gq": float(r.get("gq", 99.0)),
                    "dp": float(r.get("dp", 30.0)),
                    "ad_ref": float(r.get("ad_ref", 0.0)),
                    "ad_alt": str(r.get("ad_alt", "30")),
                    "wsaf": float(r.get("wsaf", np.nan)),
                    "qual": float(r.get("qual", 99.0)),
                    "source": str(r.get("source", "diploid_wgs")),
                }
                by_samp.setdefault(row_full["sample_id"], []).append(row_full)
            for s_rows in by_samp.values():
                writer.write_table(pa.Table.from_pylist(s_rows, schema=VAR_SCHEMA))
        else:
            writer.write_table(pa.Table.from_pydict({f.name: [] for f in VAR_SCHEMA}, schema=VAR_SCHEMA))

    if nocall_rows is not None:
        nc_tbl = pa.Table.from_pylist(nocall_rows)
        pq.write_table(nc_tbl, state_dir / "site_nocalls.parquet", compression="zstd")

    HypothesisLedger.from_store(FocaloopStateStore(state_dir), alpha_budget=0.05)
    write_json(
        state_dir / STATE_META,
        {
            "engine": "variationloop",
            "version": "0.4.0",
            "blueprint_id": "variationloop",
            "mode": "diploid_wgs",
            "ploidy": ploidy,
            "min_wsaf": 0.25,
            "summary": {"mode": "diploid_wgs", "ploidy": ploidy, "n_samples": len(samples)},
            "samples": samples,
        },
    )
    return state_dir


def _make_expression_state(
    state_dir: Path,
    tmp_work: Path,
    *,
    with_holdout: bool = False,
    holdout_depth_ratio_a_over_b: float = 1.0,
    holdout_true_lfc_g0: float = 3.0,
    holdout_reads_g0: int = 120,
    with_zoom: bool = False,
    with_zoom_context: bool = True,
    with_secondary: bool = False,
) -> tuple[Path, list[str]]:
    state_dir.mkdir(parents=True, exist_ok=True)
    ref_td = TEST_DATA / "expressionloop"
    specs = [AssemblySpec("primary", ref_td / "ref_primary.fa", ref_td / "ref_primary.gtf")]
    if with_secondary:
        specs.append(AssemblySpec("v2", ref_td / "ref_secondary.fa", ref_td / "ref_secondary.gtf"))
    cat = build_catalog(specs, "eukaryotic", "primary", tmp_work / "cat_tmp")
    cat.save(state_dir / "catalog")
    write_json(state_dir / "catalog" / "link_stats.json", cat.link_stats)

    prim_tgts = sorted(
        [t for t in cat.targets.values() if not t.is_decoy and t.assembly_id == "primary"],
        key=lambda x: (x.gene_id, x.target_id),
    )
    gids = sorted({t.gene_id for t in prim_tgts})
    gid_rank = {gid: i for i, gid in enumerate(gids)}
    g1_grp = next(t.group_id for t in prim_tgts if t.gene_id == gids[1])

    samples = [
        {"run_id": "s1", "sample": "s1", "condition": "control", "drug": "none", "plate": "p1"},
        {"run_id": "s2", "sample": "s2", "condition": "control", "drug": "none", "plate": "p1"},
        {"run_id": "s3", "sample": "s3", "condition": "control", "drug": "none", "plate": "p2"},
        {"run_id": "s4", "sample": "s4", "condition": "control", "drug": "none", "plate": "p2"},
        {"run_id": "s5", "sample": "s5", "condition": "treated", "drug": "cip", "plate": "p1"},
        {"run_id": "s6", "sample": "s6", "condition": "treated", "drug": "cip", "plate": "p1"},
        {"run_id": "s7", "sample": "s7", "condition": "treated", "drug": "cip", "plate": "p2"},
        {"run_id": "s8", "sample": "s8", "condition": "treated", "drug": "cip", "plate": "p2"},
    ]
    pd.DataFrame(samples).to_parquet(state_dir / "sample_metadata.parquet", index=False)

    tc_rows = []
    zoom_rows = []
    hold_rows = []
    for si, sm in enumerate(samples):
        is_a = sm["condition"] == "treated"
        in_p1 = sm["plate"] == "p1"
        for ti, tgt in enumerate(prim_tgts):
            gi = gid_rank[tgt.gene_id]
            n_iso = sum(1 for x in prim_tgts if x.gene_id == tgt.gene_id)
            if gi == 0:
                eff = ((280.0 + si * 4.0) if is_a else (24.0 + (si % 3) * 2.0)) / n_iso
            elif gi == 1:
                # Borderline in global pass, crisp after Stage-5 zoom
                eff = ((26.0 + (si % 2) * 3.0) if is_a else (14.0 + (si % 2) * 2.0)) / n_iso
            elif gi == 2:
                # Unchanged gene (true log2FC == 0)
                eff = (100.0 + (si % 2) * 2.0) / n_iso
            else:
                eff = (75.0 + ((gi * 5 + si * 3) % 12)) / n_iso
            row = {
                "run_id": sm["run_id"],
                "sample": sm["sample"],
                "condition": sm["condition"],
                "target_id": tgt.target_id,
                "assembly_id": tgt.assembly_id,
                "gene_id": tgt.gene_id,
                "group_id": tgt.group_id,
                "symbol": tgt.symbol or tgt.gene_id,
                "biotype": tgt.biotype or "protein_coding",
                "length_bp": int(tgt.length),
                "unique_sense": float(eff),
                "unique_antisense": 0.0,
                "effective_count": float(eff),
                "assignment_variance": 0.05,
                "group_assignment_variance": 0.05,
                "wgs_dna_reads": 0.0,
                "dna_present_prob": 1.0,
            }
            tc_rows.append(row)

            if with_zoom and gi == 1:
                z_eff = ((90.0 + (si % 2) * 4.0) if is_a else (15.0 + (si % 2))) / n_iso
                zoom_rows.append({**row, "unique_sense": float(z_eff), "effective_count": float(z_eff)})

            if with_holdout:
                lib_mult = holdout_depth_ratio_a_over_b if is_a else 1.0
                if gi == 0:
                    if holdout_reads_g0 < 4:
                        h_eff = ((float(holdout_reads_g0) / 4.0) if is_a else 0.0) / n_iso
                    else:
                        # Partition holdout_reads_g0 across p1 vs p2 so --subset plate=p1 is testable
                        base_b = max(1.0, holdout_reads_g0 / 8.0) / n_iso
                        fc_lin = 2.0 ** holdout_true_lfc_g0
                        if in_p1:
                            h_eff = base_b * fc_lin * lib_mult if is_a else base_b
                        else:
                            # Opposite sign in p2 so subsetting to p1 distinctly changes holdout result!
                            h_eff = base_b * lib_mult if is_a else base_b * 4.0
                elif gi == 2:
                    # Unchanged gene: raw holdout counts scale purely with lib_mult
                    h_eff = (60.0 * lib_mult) / n_iso
                else:
                    h_eff = (200.0 * lib_mult) / n_iso
                hold_rows.append({**row, "unique_sense": float(h_eff), "effective_count": float(h_eff)})

    pstore = PanStateStore(state_dir / "panexpr_state")
    pstore.write("target_counts", pa.Table.from_pylist(tc_rows, schema=TARGET_COUNTS_SCHEMA), TARGET_COUNTS_SCHEMA)
    if with_zoom:
        pstore.write(
            "target_counts_zoom",
            pa.Table.from_pylist(zoom_rows, schema=TARGET_COUNTS_SCHEMA),
            TARGET_COUNTS_SCHEMA,
        )
        if with_zoom_context:
            write_json(
                state_dir / "panexpr_state" / "zoom_context.json",
                {
                    "zoomed_groups": [g1_grp],
                    "sample_depth_scale": {sm["sample"]: 0.25 for sm in samples},
                },
            )
    if with_holdout:
        pstore.write(
            "target_counts_holdout",
            pa.Table.from_pylist(hold_rows, schema=TARGET_COUNTS_SCHEMA),
            TARGET_COUNTS_SCHEMA,
        )

    HypothesisLedger.from_store(FocaloopStateStore(state_dir), alpha_budget=0.05)
    write_json(
        state_dir / STATE_META,
        {
            "engine": "panexpr",
            "version": "0.4.0",
            "blueprint_id": "panexpr",
            "primary_contrast": ["condition", "treated", "control"],
            "design": "~condition",
            "organism_mode": "eukaryotic",
            "length_offsets": True,
            "catalog_summary": {
                "primary_assembly": "primary",
                "assemblies": [s.assembly_id for s in specs],
                "n_targets": len(cat),
                "n_groups": len(cat.groups),
            },
            "loop_summary": {
                "source_mode": "local_bytes",
                "bytes_streamed_global": 65536,
                "bytes_streamed_zoom": 16384 if with_zoom else 0,
                "holdout_saved": with_holdout,
            },
            "samples": samples,
        },
    )
    return state_dir, gids


# ============================================================================
# 1. Important Finding #14: ENA run lengths single-accession + retries + row safety
# ============================================================================


def test_finding_14_ena_run_lengths_single_accession_retries_and_safe_rows(monkeypatch):
    monkeypatch.setattr(logan.time, "sleep", lambda _: None)
    requested_urls: list[str] = []
    attempts_by_acc: dict[str, int] = {}

    class _FakeResp:
        def __init__(self, status_code: int, text: str) -> None:
            self.status_code = status_code
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class _FakeSess:
        def get(self, url: str, timeout: float = 30.0) -> _FakeResp:
            requested_urls.append(url)
            m = re.search(r"accession=([^&]+)", url)
            assert m is not None
            acc_param = m.group(1)
            # Comma-joined accession= on ENA returns HTTP 200 header-only!
            if "," in acc_param:
                return _FakeResp(200, "run_accession\tread_count\tbase_count\tlibrary_layout\tfastq_bytes\n")
            attempts_by_acc[acc_param] = attempts_by_acc.get(acc_param, 0) + 1
            if acc_param == "SRR1553607" and attempts_by_acc[acc_param] == 1:
                # First attempt returns header-only (must be treated as failure and retried!)
                return _FakeResp(200, "run_accession\tread_count\tbase_count\tlibrary_layout\tfastq_bytes\n")
            if acc_param == "SRR1553606":
                # Contains a corrupt row alongside the valid row
                return _FakeResp(
                    200,
                    "run_accession\tread_count\tbase_count\tlibrary_layout\tfastq_bytes\n"
                    "SRR1553606\tBAD_INT\tBAD_BASE\tPAIRED\t123\n"
                    "SRR1553606\t1000\t300000\tPAIRED\t50000;51000\n",
                )
            return _FakeResp(
                200,
                "run_accession\tread_count\tbase_count\tlibrary_layout\tfastq_bytes\n"
                "SRR1553607\t2000\t202000\tSINGLE\t90000\n",
            )

    monkeypatch.setattr(logan, "_ipv4_session", lambda: _FakeSess())
    res = logan.fetch_ena_run_lengths(["SRR1553607", "SRR1553606"], retries=3, max_workers=2)
    acc_params = [re.search(r"accession=([^&]+)", u).group(1) for u in requested_urls]
    assert all("," not in a for a in acc_params)
    assert attempts_by_acc["SRR1553607"] == 2
    assert set(res) == {"SRR1553607", "SRR1553606"}
    assert res["SRR1553607"]["layout"] == "SE"
    assert res["SRR1553607"]["read_len"] == pytest.approx(101.0)
    assert res["SRR1553607"]["kmers_per_frag"] == pytest.approx(71.0)
    assert res["SRR1553606"]["layout"] == "PE"
    assert res["SRR1553606"]["read_len"] == pytest.approx(150.0)
    assert res["SRR1553606"]["kmers_per_frag"] == pytest.approx(240.0)
    assert res["SRR1553606"]["fastq_bytes"] == 101000


# ============================================================================
# 2. Important Findings #61 & #67: Ledger integrity, holdout subset/factor/depth
# ============================================================================


def test_finding_61_no_fake_holdout_no_repeat_or_zero_rewards(tmp_path):
    # 1. HypothesisLedger unit checks: primary seed, 0 vs 0, negative vs 0, missing holdout, repeated spec
    store = FocaloopStateStore(tmp_path / "ledger_unit")
    ledger = HypothesisLedger.from_store(store, alpha_budget=0.05)

    # Seed record from primary ExpressionLoop discovery must not inflate wealth to 0.085
    v_seed = ledger.evaluate_query(
        "q0001", "focaloop-expressionloop", "condition:treated_vs_control:primary_discovery",
        4.0, 1e-9, 4.0, 1e-9, store=store,
    )
    assert v_seed.confirmed is False
    assert v_seed.alpha_wealth_after == pytest.approx(0.05)

    # 0.0 vs 0.0 must never confirm
    v_zero = ledger.evaluate_query("q0002", "human", "spec_zero", 0.0, 1e-6, 0.0, 1e-6, store=store)
    assert v_zero.confirmed is False

    # Negative discovery vs 0.0 holdout must never confirm (old `(d > 0) == (h > 0)` False==False bug!)
    v_neg_zero = ledger.evaluate_query("q0003", "human", "spec_neg_zero", -2.5, 1e-6, 0.0, 1e-6, store=store)
    assert v_neg_zero.confirmed is False

    # Genuine holdout confirmation earns reward once, but repeating identical spec earns 0.0 reward
    w_before = ledger.wealth
    v_ok1 = ledger.evaluate_query("q0004", "human", "spec_real_hit", -2.5, 1e-8, -2.1, 1e-5, store=store)
    assert v_ok1.confirmed is True
    assert v_ok1.alpha_wealth_after > w_before

    w_after_1 = ledger.wealth
    v_ok_dup = ledger.evaluate_query("q0005", "human", "spec_real_hit", -2.5, 1e-8, -2.1, 1e-5, store=store)
    assert v_ok_dup.alpha_wealth_after < w_after_1

    # Persisted _seen_specs survives reload via from_store
    reloaded = HypothesisLedger.from_store(store, alpha_budget=0.05)
    w_reload = reloaded.wealth
    v_ok_dup2 = reloaded.evaluate_query("q0006", "human", "spec_real_hit", -2.5, 1e-8, -2.1, 1e-5, store=store)
    assert v_ok_dup2.alpha_wealth_after < w_reload

    # 2. End-to-end ExpressionLoop Query without holdout parquet: holdout_available=False, wealth strictly decreases
    el_no_hold, gids = _make_expression_state(tmp_path / "el_no_hold", tmp_path / "w1", with_holdout=False)
    arc0 = archive.pack_state(el_no_hold, tmp_path / "s0.focaloop.tar.zst")
    arc1 = tmp_path / "s1.focaloop.tar.zst"
    arc2 = tmp_path / "s2.focaloop.tar.zst"

    r1 = run_query(arc0, "condition:treated:control", "human", tmp_path / "q1.tsv", tmp_path / "v1.json", out_state_archive=arc1)
    assert r1["holdout_available"] is False
    assert r1["verdict"]["confirmed"] is False
    assert r1["verdict"]["holdout_estimate"] is None
    assert r1["wealth_after"] == pytest.approx(0.045)

    r2 = run_query(arc1, "condition:treated:control", "human", tmp_path / "q2.tsv", tmp_path / "v2.json", out_state_archive=arc2)
    assert r2["holdout_available"] is False
    assert r2["verdict"]["confirmed"] is False
    assert r2["wealth_after"] == pytest.approx(0.0405)

    # 3. Holdout with <4 reads on target gene: also treated as holdout_available=False (never substitutes disc_p)
    el_low_reads, gids_low = _make_expression_state(
        tmp_path / "el_low_hold", tmp_path / "w2", with_holdout=True, holdout_reads_g0=2
    )
    r_low = run_query(
        el_low_reads, "condition:treated:control", "human",
        tmp_path / "q_low.tsv", tmp_path / "v_low.json", hypothesis=gids_low[0],
    )
    assert r_low["holdout_available"] is False
    assert r_low["verdict"]["confirmed"] is False
    assert r_low["verdict"]["holdout_pvalue"] is None


def test_finding_67_holdout_non_condition_factor_subset_and_depth_normalization(tmp_path):
    # Treated has 2x deeper holdout library than control, and g0 has true negative log2FC (-1.2) in plate=p1!
    el_dir, gids = _make_expression_state(
        tmp_path / "el_depth",
        tmp_path / "w_depth",
        with_holdout=True,
        holdout_depth_ratio_a_over_b=2.0,
        holdout_true_lfc_g0=-1.2,
        holdout_reads_g0=160,
    )
    # Also make global counts of g0 match negative log2FC (-1.2) so discovery and holdout agree on p1
    tc_p = el_dir / "panexpr_state" / "target_counts.parquet"
    df_tc = pq.read_table(tc_p).to_pandas()
    mask_g0_trt = (df_tc["gene_id"] == gids[0]) & (df_tc["condition"] == "treated")
    mask_g0_ctl = (df_tc["gene_id"] == gids[0]) & (df_tc["condition"] == "control")
    df_tc.loc[mask_g0_ctl, ["unique_sense", "effective_count"]] = 260.0
    df_tc.loc[mask_g0_trt, ["unique_sense", "effective_count"]] = 110.0
    pq.write_table(pa.Table.from_pandas(df_tc, schema=TARGET_COUNTS_SCHEMA, preserve_index=False), tc_p, compression="zstd")

    # (a) Non-'condition' factor (`drug:cip:none` with `--design ~drug`) + `--subset plate=p1`
    res_p1 = run_query(
        el_dir,
        "drug:cip:none",
        "human",
        tmp_path / "q_drug_p1.tsv",
        tmp_path / "v_drug_p1.json",
        hypothesis=gids[0],
        design="~drug",
        subset="plate=p1",
    )
    assert res_p1["holdout_available"] is True
    # Despite 2x deeper 'cip' holdout library, library-size normalization recovers negative holdout log2FC!
    assert res_p1["verdict"]["discovery_estimate"] < -0.5
    assert res_p1["verdict"]["holdout_estimate"] < -0.5
    assert res_p1["verdict"]["confirmed"] is True

    # (b) Same factor on `--subset plate=p2` (where holdout was planted in opposite direction) -> holdout differs from p1!
    el_dir2, _ = _make_expression_state(
        tmp_path / "el_depth2",
        tmp_path / "w_depth2",
        with_holdout=True,
        holdout_depth_ratio_a_over_b=2.0,
        holdout_true_lfc_g0=-1.2,
        holdout_reads_g0=160,
    )
    pq.write_table(pa.Table.from_pandas(df_tc, schema=TARGET_COUNTS_SCHEMA, preserve_index=False), el_dir2 / "panexpr_state" / "target_counts.parquet", compression="zstd")
    res_p2 = run_query(
        el_dir2,
        "drug:cip:none",
        "human",
        tmp_path / "q_drug_p2.tsv",
        tmp_path / "v_drug_p2.json",
        hypothesis=gids[0],
        design="~drug",
        subset="plate=p2",
    )
    assert res_p2["verdict"]["holdout_estimate"] != pytest.approx(res_p1["verdict"]["holdout_estimate"])

    # (c) Unchanged gene (gids[2]) under 2x holdout depth imbalance: holdout_estimate ~ 0.0 and pvalue > 0.5 (not +1.0 / 1e-90!)
    res_null = run_query(
        el_dir2,
        "drug:cip:none",
        "human",
        tmp_path / "q_null.tsv",
        tmp_path / "v_null.json",
        hypothesis=gids[2],
        design="~drug",
    )
    assert abs(res_null["verdict"]["holdout_estimate"]) < 0.1
    assert res_null["verdict"]["holdout_pvalue"] > 0.5
    assert res_null["verdict"]["confirmed"] is False


# ============================================================================
# 3. Important Findings #61, #62, #63, #64, #65, #69 + L67: VariationLoop Fst
# ============================================================================


def test_finding_62_variationloop_fst_zero_read_guard_and_nocall_handling(tmp_path):
    # 1. Group containing a zero-read sample is refused
    samples_zero = [
        {"sample": "a1", "group": "A", "reads_counted": 1000},
        {"sample": "a2", "group": "A", "reads_counted": 1000},
        {"sample": "b1", "group": "B", "reads_counted": 0},
        {"sample": "b2", "group": "B", "reads_counted": 1000},
    ]
    v_rows = [{"sample_id": "a1", "pos": 100, "gt": "1/1"}, {"sample_id": "a2", "pos": 100, "gt": "1/1"}]
    st_zero = _make_variation_state(tmp_path / "vl_zero", samples_zero, v_rows)
    with pytest.raises(ValueError, match="zero-read sample"):
        run_query(st_zero, "group=A:B", "human", tmp_path / "z.tsv", tmp_path / "z.json", mode="fst")

    # 2. Per-site no-calls (`./.`, `dp == 0`, and `site_nocalls.parquet`) excluded from N_called; fixed sites excluded
    samples_ok = [
        {"sample": "a1", "group": "A", "reads_counted": 1000},
        {"sample": "a2", "group": "A", "reads_counted": 1000},
        {"sample": "a3", "group": "A", "reads_counted": 1000},
        {"sample": "b1", "group": "B", "reads_counted": 1000},
        {"sample": "b2", "group": "B", "reads_counted": 1000},
        {"sample": "b3", "group": "B", "reads_counted": 1000},
    ]
    # Pos 100: A has a1=1/1, a2=1/1, a3=./. (n_a=2, af_a=1.0);
    #          B has b1=1/1, b2 in site_nocalls.parquet, b3 dp=0 (n_b=1, af_b=1.0 -> fixed in called samples, den=0 -> skipped!)
    # Pos 200: A has a1=1/1, a2=0/1, a3=./. (n_a=2, af_a=0.75);
    #          B has b1 in site_nocalls.parquet, b2=0/0 (implicit), b3=0/0 (implicit) -> n_b=2, af_b=0.0, den > 0!
    v_rows_nc = [
        {"sample_id": "a1", "chrom": "chr1", "pos": 100, "gt": "1/1", "dp": 30.0},
        {"sample_id": "a2", "chrom": "chr1", "pos": 100, "gt": "1/1", "dp": 30.0},
        {"sample_id": "a3", "chrom": "chr1", "pos": 100, "gt": "./.", "dp": 0.0},
        {"sample_id": "b1", "chrom": "chr1", "pos": 100, "gt": "1/1", "dp": 25.0},
        {"sample_id": "b3", "chrom": "chr1", "pos": 100, "gt": "0/0", "dp": 0.0},
        {"sample_id": "a1", "chrom": "chr1", "pos": 200, "gt": "1/1", "dp": 30.0},
        {"sample_id": "a2", "chrom": "chr1", "pos": 200, "gt": "0/1", "dp": 30.0},
        {"sample_id": "a3", "chrom": "chr1", "pos": 200, "gt": ".|.", "dp": 15.0},
    ]
    nc_rows = [
        {"chrom": "chr1", "pos": 100, "sample_id": "b2"},
        {"chrom": "chr1", "pos": 200, "sample_id": "b1"},
    ]
    st_nc = _make_variation_state(tmp_path / "vl_nc", samples_ok, v_rows_nc, nocall_rows=nc_rows)
    res = run_query(st_nc, "group=A:B", "human", tmp_path / "nc.tsv", tmp_path / "nc.json", mode="fst")
    # Pos 100 became fixed (af_a=1.0, af_b=1.0 -> den=0) once b2 & b3 were excluded, so only pos 200 is informative!
    assert res["informative_sites"] == 1
    assert res["holdout_available"] is False
    assert res["verdict"]["confirmed"] is False
    df = pd.read_csv(tmp_path / "nc.tsv", sep="\t")
    assert list(df["pos"]) == [200]
    assert int(df.loc[0, "n_group_a"]) == 2
    assert int(df.loc[0, "n_group_b"]) == 2
    assert float(df.loc[0, "af_group_a"]) == pytest.approx(0.75)
    assert float(df.loc[0, "af_group_b"]) == pytest.approx(0.0)


def test_finding_63_diploid_chromosome_count_vs_haploid_hudson_fst(tmp_path):
    # Reproduce code_review.md #63: N=5 diploid null cohort should have Hudson Fst ~ 0.0 (not -0.125!)
    rng = np.random.default_rng(12345)
    samples = (
        [{"sample": f"a{i}", "group": "A", "reads_counted": 500} for i in range(5)]
        + [{"sample": f"b{i}", "group": "B", "reads_counted": 500} for i in range(5)]
    )
    rows_null = []
    for site_i in range(1, 201):
        for s in samples:
            alleles = int(rng.binomial(2, 0.4))
            if alleles > 0:
                gt = "1/1" if alleles == 2 else "0/1"
                rows_null.append({"sample_id": s["sample"], "chrom": "chr1", "pos": site_i * 10, "gt": gt, "dp": 30.0})

    st_dip = _make_variation_state(tmp_path / "vl_dip_null", samples, rows_null, ploidy="diploid")
    res_dip = run_query(st_dip, "group=A:B", "human", tmp_path / "dip.tsv", tmp_path / "dip.json", mode="fst")
    assert abs(res_dip["genome_fst"]) < 0.04  # Unbiased around 0.0, whereas N-1 individuals gave -0.125!

    # Deterministic single-site algebraic check for diploid (2*N - 1 = 9) vs haploid (N - 1 = 4)
    # Suppose 2 of 5 samples in A and 2 of 5 samples in B have dosage 0.5 (pa = pb = 0.2):
    # den = 2 * 0.2 * 0.8 = 0.32
    # Diploid: c_a = c_b = 0.16 / 9 -> num = -0.32 / 9 -> Fst = -1/9 = -0.1111
    # Haploid: c_a = c_b = 0.16 / 4 -> num = -0.08 -> Fst = -0.08 / 0.32 = -0.25
    rows_exact = [
        {"sample_id": "a0", "chrom": "chr1", "pos": 500, "gt": "0/1", "wsaf": 0.5, "dp": 40.0},
        {"sample_id": "a1", "chrom": "chr1", "pos": 500, "gt": "0/1", "wsaf": 0.5, "dp": 40.0},
        {"sample_id": "b0", "chrom": "chr1", "pos": 500, "gt": "0/1", "wsaf": 0.5, "dp": 40.0},
        {"sample_id": "b1", "chrom": "chr1", "pos": 500, "gt": "0/1", "wsaf": 0.5, "dp": 40.0},
    ]
    st_ex_dip = _make_variation_state(tmp_path / "ex_dip", samples, rows_exact, ploidy="diploid")
    r_ex_dip = run_query(st_ex_dip, "group=A:B", "human", tmp_path / "ed.tsv", tmp_path / "ed.json", mode="fst")
    assert r_ex_dip["genome_fst"] == pytest.approx(-1.0 / 9.0, abs=1e-3)

    st_ex_hap = _make_variation_state(tmp_path / "ex_hap", samples, rows_exact, ploidy="haploid")
    r_ex_hap = run_query(st_ex_hap, "group=A:B", "human", tmp_path / "eh.tsv", tmp_path / "eh.json", mode="fst")
    assert r_ex_hap["genome_fst"] == pytest.approx(-0.25, abs=1e-3)


def test_finding_64_65_69_and_l67_fst_validation_streaming_and_errors(tmp_path, monkeypatch):
    samples = [
        {"sample": "n1", "site": "north", "group": "g1", "reads_counted": 800},
        {"sample": "n2", "site": "north", "group": "g1", "reads_counted": 800},
        {"sample": "s1", "site": "south", "group": "g2", "reads_counted": 800},
        {"sample": "s2", "site": "south", "group": "g2", "reads_counted": 800},
    ]
    v_rows = [
        {"sample_id": "n1", "chrom": "chr20_mini", "pos": 1000, "gt": "1/1"},
        {"sample_id": "n2", "chrom": "chr20_mini", "pos": 1000, "gt": "1/1"},
    ]
    st_dir = _make_variation_state(tmp_path / "vl_guards", samples, v_rows)
    store = FocaloopStateStore(st_dir)

    # #64: Wrong chromosome in --hypothesis raises ValueError naming hypothesis + available contigs before touching ledger
    with pytest.raises(ValueError, match="chr20_mini"):
        run_query(st_dir, "site=north:south", "human", tmp_path / "o.tsv", tmp_path / "o.json", hypothesis="chr20", mode="fst")
    assert not store.exists("hypothesis_ledger")

    # #65: Identical or overlapping groups raise ValueError
    with pytest.raises(ValueError, match="overlap"):
        run_query(st_dir, "site=north:north", "human", tmp_path / "o.tsv", tmp_path / "o.json", mode="fst")
    with pytest.raises(ValueError, match="overlap"):
        run_query(st_dir, "site=north,south:south", "human", tmp_path / "o.tsv", tmp_path / "o.json", mode="fst")

    # L67: Mistyped explicit column raises ValueError (no silent auto-detect fallback!)
    with pytest.raises(ValueError, match="regoin"):
        run_query(st_dir, "regoin=north:south", "human", tmp_path / "o.tsv", tmp_path / "o.json", mode="fst")
    # L67: Unknown label in comma list raises ValueError
    with pytest.raises(ValueError, match="atlantis"):
        run_query(st_dir, "site=north,atlantis:south", "human", tmp_path / "o.tsv", tmp_path / "o.json", mode="fst")
    # L67: Auto-detect requires ALL labels in a column
    with pytest.raises(ValueError, match="not found"):
        run_query(st_dir, "north,atlantis:south", "human", tmp_path / "o.tsv", tmp_path / "o.json", mode="fst")

    # #69: Verify contrast validation runs BEFORE reading variants.parquet AND pd.read_parquet is never called on variants.parquet
    orig_read_pq = pd.read_parquet

    def _forbid_variants_pandas(path, *a, **kw):
        if "variants.parquet" in str(path):
            raise AssertionError("pd.read_parquet must never be called on variants.parquet!")
        return orig_read_pq(path, *a, **kw)

    monkeypatch.setattr(pd, "read_parquet", _forbid_variants_pandas)
    ok_res = run_query(
        st_dir, "site=north:south", "human", tmp_path / "ok.tsv", tmp_path / "ok.json",
        hypothesis="chr20_mini:500-1500", mode="fst",
    )
    assert ok_res["genome_fst"] == pytest.approx(1.0)
    assert ok_res["informative_sites"] == 1

    # #69: cmd_query catches MemoryError cleanly and returns EXIT_ERROR (2)
    import focaloop.ops as ops_mod

    def _raise_mem(*a, **kw):
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(ops_mod, "run_query", _raise_mem)
    rc = cli.main(
        [
            "query",
            "--state",
            str(st_dir),
            "--contrast",
            "site=north:south",
            "--out-table",
            str(tmp_path / "m.tsv"),
            "--out-verdict",
            str(tmp_path / "m.json"),
        ]
    )
    assert rc == 2


# ============================================================================
# 4. Important Finding #66 + L66: ExpressionLoop Zoom preservation & exact hypothesis
# ============================================================================


def test_finding_66_and_l66_expressionloop_zoom_context_and_exact_hypothesis(tmp_path):
    # 1. With target_counts_zoom.parquet + zoom_context.json -> zoom_evidence_used=True & DE_after_zoom
    st_zoom, gids = _make_expression_state(
        tmp_path / "el_zoom", tmp_path / "wz1", with_zoom=True, with_zoom_context=True, with_secondary=True
    )
    z_gene = gids[1]
    res_z = run_query(
        st_zoom, "condition:treated:control", "human",
        tmp_path / "qz.tsv", tmp_path / "vz.json", subset="plate=p1",
    )
    assert res_z["zoom_evidence_used"] is True
    df_z = pd.read_csv(tmp_path / "qz.tsv", sep="\t")
    row_zg = df_z[df_z["gene_id"] == z_gene].iloc[0]
    assert bool(row_zg["zoomed"]) is True
    assert "after_zoom" in str(row_zg["status"])

    # 2. Legacy zoom bundle missing zoom_context.json -> zoom_evidence_used=False + zoom_warning
    st_no_ctx, _ = _make_expression_state(
        tmp_path / "el_no_zctx", tmp_path / "wz2", with_zoom=True, with_zoom_context=False
    )
    res_nz = run_query(
        st_no_ctx, "condition:treated:control", "human",
        tmp_path / "qnz.tsv", tmp_path / "vnz.json",
    )
    assert res_nz["zoom_evidence_used"] is False
    assert "zoom_warning" in res_nz

    # 3. L66: --hypothesis rejects unanchored regex `.` and partial prefix `GENE_0` while matching exact ortholog token
    with pytest.raises(ValueError, match="not found"):
        run_query(st_zoom, "condition:treated:control", "human", tmp_path / "bad1.tsv", tmp_path / "bad1.json", hypothesis=".")
    with pytest.raises(ValueError, match="not found"):
        run_query(st_zoom, "condition:treated:control", "human", tmp_path / "bad2.tsv", tmp_path / "bad2.json", hypothesis="GENE_0")

    # Find an ortholog_bridge token in df_z and verify exact token matching works
    bridged = df_z[df_z["ortholog_bridge"].fillna("").astype(str) != ""]
    assert not bridged.empty
    first_bridge = str(bridged.iloc[0]["ortholog_bridge"]).split(";")[0].split(",")[0].strip()
    expected_prim_gid = str(bridged.iloc[0]["gene_id"])
    res_br = run_query(
        st_zoom, "condition:treated:control", "human",
        tmp_path / "qbr.tsv", tmp_path / "vbr.json", hypothesis=first_bridge,
    )
    assert res_br["top_gene"] == expected_prim_gid


# ============================================================================
# 5. Important Finding #68: Candidatus & Genus sp. NCBI species resolution
# ============================================================================


def test_finding_68_ncbi_candidatus_and_unclassified_species_handling(monkeypatch):
    def _mock_fetch_json(url: str, params=None, **kw):
        if "/accession/GCF_000023765.2/dataset_report" in url:
            return {
                "reports": [
                    {
                        "accession": "GCF_000023765.2",
                        "organism": {
                            "tax_id": 537021,
                            "organism_name": "Candidatus Liberibacter asiaticus psy62",
                            "infraspecific_names": {"strain": "psy62"},
                        },
                    }
                ]
            }
        if "/taxonomy/taxon/537021" in url:
            return {
                "taxonomy_nodes": [
                    {
                        "taxonomy": {
                            "tax_id": 537021,
                            "organism_name": "Candidatus Liberibacter asiaticus psy62",
                            "rank": "STRAIN",
                            "lineage": [1, 131567, 2, 1224, 28211, 356, 34019, 34021],
                        }
                    }
                ]
            }
        if "/taxonomy/taxon/" in url and "34021" in url and "," in url:
            return {
                "taxonomy_nodes": [
                    {"taxonomy": {"tax_id": 34019, "organism_name": "Candidatus Liberibacter", "rank": "GENUS"}},
                    {"taxonomy": {"tax_id": 34021, "organism_name": "Candidatus Liberibacter asiaticus", "rank": "SPECIES"}},
                ]
            }
        if "/taxonomy/taxon/Candidatus%20Liberibacter%20asiaticus" in url or url.endswith("/taxonomy/taxon/34021"):
            return {
                "taxonomy_nodes": [
                    {
                        "taxonomy": {
                            "tax_id": 34021,
                            "organism_name": "Candidatus Liberibacter asiaticus",
                            "rank": "SPECIES",
                            "lineage": [1, 2, 34019],
                        }
                    }
                ]
            }
        if "/taxonomy/taxon/" in url and ("536065" in url or "1273132" in url):
            return {
                "taxonomy_nodes": [
                    {"taxonomy": {"tax_id": 536065, "organism_name": "Candidatus Liberibacter solanacearum", "rank": "SPECIES"}},
                    {"taxonomy": {"tax_id": 1273132, "organism_name": "Candidatus Liberibacter americanus", "rank": "SPECIES"}},
                ]
            }
        if "/taxonomy/taxon/Candidatus%20Liberibacter" in url:
            return {
                "taxonomy_nodes": [
                    {
                        "taxonomy": {
                            "tax_id": 34019,
                            "organism_name": "Candidatus Liberibacter",
                            "rank": "GENUS",
                            "lineage": [1, 2, 1224],
                        }
                    }
                ]
            }
        if any(tok in url for tok in ("/taxonomy/taxon/1%2C2%2C1224", "/taxonomy/taxon/2%2C1224", "/taxonomy/taxon/1,2,1224", "/taxonomy/taxon/2,1224")):
            return {"taxonomy_nodes": [{"taxonomy": {"tax_id": 1224, "organism_name": "Pseudomonadota", "rank": "PHYLUM"}}]}
        if "/genome/taxon/34021/dataset_report" in url:
            return {
                "reports": [
                    {
                        "accession": "GCF_000183665.1",
                        "assembly_info": {"assembly_name": "CLso", "assembly_level": "Complete Genome", "refseq_category": "reference genome"},
                        "organism": {"tax_id": 536065, "organism_name": "Candidatus Liberibacter solanacearum", "infraspecific_names": {"strain": "ZC1"}},
                    },
                    {
                        "accession": "GCF_000496595.1",
                        "assembly_info": {"assembly_name": "CLam", "assembly_level": "Complete Genome", "refseq_category": "reference genome"},
                        "organism": {"tax_id": 1273132, "organism_name": "Candidatus Liberibacter americanus", "infraspecific_names": {"strain": "Sao Paulo"}},
                    },
                    {
                        "accession": "GCF_000590865.3",
                        "assembly_info": {"assembly_name": "CLas_gxpsy", "assembly_level": "Complete Genome"},
                        "organism": {"tax_id": 34021, "organism_name": "Candidatus Liberibacter asiaticus", "infraspecific_names": {"strain": "gxpsy"}},
                    },
                ]
            }
        raise AssertionError(f"Unexpected URL in mock: {url}")

    monkeypatch.setattr(ncbi_ref, "_fetch_json", _mock_fetch_json)

    meta = ncbi_ref.lookup_ncbi_assembly_taxon("GCF_000023765.2")
    assert meta["species_name"] == "Candidatus Liberibacter asiaticus"
    assert meta["species_tax_id"] == 34021

    cands = ncbi_ref.discover_related_ncbi_assemblies(
        meta["species_name"],
        max_strains=3,
        exclude_accessions=["GCF_000023765.2"],
        exclude_strains=["psy62"],
    )
    # Congeneric Ca. L. solanacearum and Ca. L. americanus MUST be filtered out!
    assert [c["accession"] for c in cands] == ["GCF_000590865.3"]

    # Unclassified 'Genus sp.' and genus-only queries raise clear ValueError
    with pytest.raises(ValueError, match="unclassified|species"):
        ncbi_ref.discover_related_ncbi_assemblies("Pseudomonas sp. URMO17")
    with pytest.raises(ValueError, match="species"):
        ncbi_ref.discover_related_ncbi_assemblies("Candidatus Liberibacter")


# ============================================================================
# 6. Important Finding #74 + L98, L100, L102: Native cache & packaging
# ============================================================================


def test_finding_74_l98_l100_native_cache_security_and_isa(tmp_path, monkeypatch):
    # #74: Portable ISA is the default unless FOCALOOP_PORTABLE_ISA == "0"
    assert nc.use_portable_isa({}) is True
    assert nc.use_portable_isa({"FOCALOOP_PORTABLE_ISA": ""}) is True
    assert nc.use_portable_isa({"FOCALOOP_PORTABLE_ISA": "1"}) is True
    assert nc.use_portable_isa({"FOCALOOP_PORTABLE_ISA": "0"}) is False
    assert "-march=native" not in nc.portable_isa_flags("x86_64", False, env={})
    assert "-march=native" in nc.portable_isa_flags("x86_64", False, env={"FOCALOOP_PORTABLE_ISA": "0"})

    # #74: SHA-256 cache key changes when either source OR flags change
    k1 = nc.compute_artifact_key(inline_sources=["int f(void){return 1;}"], flags=["-O3", "-msse4.2"])
    k2 = nc.compute_artifact_key(inline_sources=["int f(void){return 1;}"], flags=["-O3", "-march=native"])
    k3 = nc.compute_artifact_key(inline_sources=["int f(void){return 2;}"], flags=["-O3", "-msse4.2"])
    assert len({k1, k2, k3}) == 3

    # L98: Per-user cache directory has strict 0700 mode and rejects symlinks
    cache_target = tmp_path / "rt_cache"
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(cache_target))
    resolved = nc.resolve_user_cache_dir()
    assert resolved == cache_target
    assert (resolved.lstat().st_mode & 0o777) == 0o700

    sym_dir = tmp_path / "sym_cache"
    sym_dir.symlink_to(cache_target)
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(sym_dir))
    with pytest.raises(PermissionError, match="symlink"):
        nc.resolve_user_cache_dir()

    # L98: Atomic compile + verify_safe_artifact rejects world-writable / group-writable / symlinked .so
    monkeypatch.setenv("FOCALOOP_BUILD_NATIVE_DIR", str(cache_target))
    so_path = nc.ensure_compiled_artifact(
        "tiny_add",
        inline_source="int focaloop_test_add(int a, int b) { return a + b; }\n",
        shared=True,
    )
    lib = nc.load_verified_cdll(so_path, require_per_user_mode=True)
    assert lib.focaloop_test_add(19, 23) == 42

    # Making runtime artifact group/world-writable causes verify_safe_artifact to reject before dlopen!
    os.chmod(so_path, 0o777)
    with pytest.raises(PermissionError):
        nc.load_verified_cdll(so_path, require_per_user_mode=True)
    os.chmod(so_path, 0o700)

    # L100: Makefile defaults PREFIX to $(PYTHON)'s sys.prefix (not /usr/local)
    mk_txt = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "sys.prefix" in mk_txt
    assert "PREFIX ?= /usr/local" not in mk_txt
    assert (Path(sys.prefix) / "share" / "focaloop" / "native_cache") in nc.prebuilt_search_dirs()


def _find_setuptools_python() -> str:
    cands = [sys.executable]
    for pdir in os.environ.get("PATH", "").split(os.pathsep):
        if pdir:
            for nm in ("python3", "python"):
                cp = Path(pdir) / nm
                if cp.is_file() and str(cp) not in cands:
                    cands.append(str(cp))
    for py in cands:
        res = subprocess.run([py, "-c", "import setuptools.build_meta"], capture_output=True)
        if res.returncode == 0:
            return py
    pytest.skip("No Python with setuptools.build_meta on PATH")


def test_l102_sdist_and_wheel_include_makefile_scripts_and_packaged_blueprints(tmp_path):
    build_py = _find_setuptools_python()
    src_copy = tmp_path / "repo_src"
    shutil.copytree(
        REPO_ROOT,
        src_copy,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "build", "dist", "*.egg-info"),
    )
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    # 1. Build sdist via setuptools PEP 517 build_sdist inside isolated copy
    subprocess.run(
        [
            build_py,
            "-c",
            f"import setuptools.build_meta as bm; bm.build_sdist({str(dist_dir)!r})",
        ],
        cwd=str(src_copy),
        check=True,
        capture_output=True,
        text=True,
    )
    sdists = list(dist_dir.glob("focaloop-*.tar.gz"))
    assert len(sdists) == 1
    with tarfile.open(sdists[0], "r:gz") as tf:
        sdist_names = tf.getnames()
    assert any(n.endswith("/Makefile") for n in sdist_names)
    assert any(n.endswith("/scripts/build_native.py") for n in sdist_names)
    for bp_id in ("panexpr", "malariavar", "geneloop", "focaloop_query", "focaloop_bait"):
        assert any(n.endswith(f"/blueprints/{bp_id}/blueprint.focaloop.yml") for n in sdist_names), bp_id

    # 2. Extract sdist to an isolated directory and build a wheel FROM the sdist!
    unpacked_root = tmp_path / "unpacked_sdist"
    with tarfile.open(sdists[0], "r:gz") as tf:
        tf.extractall(unpacked_root, filter="data")
    sdist_src = next(unpacked_root.glob("focaloop-*"))
    whl_dir = tmp_path / "wheel_out"
    whl_dir.mkdir()
    subprocess.run(
        [
            build_py,
            "-c",
            f"import setuptools.build_meta as bm; bm.build_wheel({str(whl_dir)!r})",
        ],
        cwd=str(sdist_src),
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(whl_dir.glob("focaloop-*.whl"))
    assert len(wheels) == 1
    site_pkg = tmp_path / "isolated_site"
    with zipfile.ZipFile(wheels[0]) as zf:
        whl_names = zf.namelist()
        for bp_id in ("panexpr", "malariavar", "geneloop", "focaloop_query", "focaloop_bait"):
            assert f"focaloop/blueprints/{bp_id}/blueprint.focaloop.yml" in whl_names
        zf.extractall(site_pkg)

    # 3. Verify `focaloop validate` succeeds from the unpacked wheel outside the repo with FOCALOOP_BLUEPRINT_PATH unset!
    env = {k: v for k, v in os.environ.items() if k != "FOCALOOP_BLUEPRINT_PATH"}
    env["PYTHONPATH"] = str(site_pkg)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "focaloop.cli",
            "validate",
            "panexpr",
            "malariavar",
            "geneloop",
            "focaloop_query",
            "focaloop_bait",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.count("OK ") == 5


# ============================================================================
# 7. Lesser Findings: #15, #18, L11, L29, L65, L68, L69, L70, L71, L72, L73, L95, L99, L101
# ============================================================================


def test_lesser_l11_l71_l72_and_18_galaxy_runners_helpers(tmp_path, monkeypatch):
    # L11: _head_content_length rejects HTTP 404 with Content-Length: 196 and parses 206 Content-Range total
    class _MockResp:
        def __init__(self, status_code: int, headers: dict[str, str]):
            self.status_code = status_code
            self.headers = headers

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def close(self):
            pass

    class _Sess404:
        def head(self, url, **kw):
            return _MockResp(405, {"Content-Length": "196"})

        def get(self, url, **kw):
            return _MockResp(404, {"Content-Length": "196"})

    monkeypatch.setattr(galaxy_runners, "_ipv4_session", lambda: _Sess404())
    with pytest.raises(RuntimeError, match="404"):
        galaxy_runners._head_content_length("https://example.org/missing.fq.gz")

    class _Sess206:
        def head(self, url, **kw):
            return _MockResp(405, {})

        def get(self, url, **kw):
            return _MockResp(206, {"Content-Length": "1", "Content-Range": "bytes 0-0/987654"})

    monkeypatch.setattr(galaxy_runners, "_ipv4_session", lambda: _Sess206())
    assert galaxy_runners._head_content_length("https://example.org/range.fq.gz") == 987654

    # L71: _resolve_panexpr_manifest orders Galaxy dataset_*.dat paths by mate token (not filename!) and rejects >2 files
    d20 = tmp_path / "dataset_20.dat"
    d100 = tmp_path / "dataset_100.dat"
    d300 = tmp_path / "dataset_300.dat"
    d20.write_text("@r1/1\nACGT\n+\nIIII\n", encoding="utf-8")
    d100.write_text("@r1/2\nTGCA\n+\nIIII\n", encoding="utf-8")
    d300.write_text("@r1\nAAAA\n+\nIIII\n", encoding="utf-8")
    mf = tmp_path / "mf.tsv"
    mf.write_text("sample_id\tcondition\nsampA\tcontrol\n", encoding="utf-8")

    # Note ASCII sort on filenames would put dataset_100.dat (R2!) BEFORE dataset_20.dat (R1!)
    resolved_mf, _, _ = galaxy_runners._resolve_panexpr_manifest(
        mf,
        reads=[str(d100), str(d20)],
        read_ids=["sampA_R2.fastq.gz", "sampA_R1.fastq.gz"],
    )
    assert [Path(f.local_path).name for f in resolved_mf.runs[0].files] == ["dataset_20.dat", "dataset_100.dat"]

    with pytest.raises(ValueError, match="at most 2"):
        galaxy_runners._resolve_panexpr_manifest(
            mf,
            reads=[str(d20), str(d100), str(d300)],
            read_ids=["sampA_R1.fq.gz", "sampA_R2.fq.gz", "sampA.fq.gz"],
        )
    with pytest.raises(ValueError, match="at most 2"):
        galaxy_runners._resolve_panexpr_manifest(
            mf,
            reads=[str(d20), str(d100), str(d300)],
            read_ids=["sampA_R1.fq.gz", "sampA_R2.fq.gz", "sampA_extra.fq.gz"],
        )

    # L72: Per-accession layout map on ENA header-only / failure fallback (row 1 SE, row 2 PE!)
    import focaloop.panexpr.manifest as pm_mod

    monkeypatch.setattr(galaxy_runners.time, "sleep", lambda _: None)
    monkeypatch.setattr(pm_mod, "fetch_ena_filereport", lambda acc, **kw: "run_accession\tfastq_ftp\n")
    monkeypatch.setattr(galaxy_runners, "_head_content_length", lambda url, **kw: 400000)

    mf_ena = tmp_path / "mf_ena.tsv"
    mf_ena.write_text(
        "sample_id\tcondition\trun_or_url\tlayout\n"
        "s_se\tcontrol\tSRR100001\tSE\n"
        "s_pe\ttreated\tSRR100002\tPE\n",
        encoding="utf-8",
    )
    res_ena, _, _ = galaxy_runners._resolve_panexpr_manifest(mf_ena)
    assert res_ena.runs[0].layout == "SE" and len(res_ena.runs[0].files) == 1
    assert res_ena.runs[1].layout == "PE" and len(res_ena.runs[1].files) == 2

    # #15 & #18: Explicit --kernels raises when requested kernel fails ensure_index (and idx_dir exists before ensure_index)
    import focaloop.panexpr.kernels as pk_mod

    class _FailingIndexKernel:
        name = "mock_fail_kernel"

        def available(self):
            return True

        def ensure_index(self, fna_targets, idx_dir):
            assert Path(idx_dir).is_dir(), "idx_dir must be created before ensure_index!"
            raise RuntimeError("mock index failure")

    monkeypatch.setattr(pk_mod, "default_kernels", lambda *a, **kw: [_FailingIndexKernel()])
    td_el = TEST_DATA / "expressionloop"
    reads_el = [
        str(td_el / f)
        for f in ("ctrl1.fastq.gz", "ctrl2.fastq.gz", "ctrl3.fastq.gz", "trt1.fastq.gz", "trt2.fastq.gz", "trt3.fastq.gz")
    ]
    with pytest.raises(RuntimeError, match="mock_fail_kernel"):
        galaxy_runners.run_panexpr_galaxy(
            manifest_path=td_el / "manifest_se.tsv",
            ref_fasta=td_el / "ref_primary.fa",
            ref_gtf=td_el / "ref_primary.gtf",
            reads=reads_el,
            contrast="condition:treated:control",
            out_de=tmp_path / "de_fail.tsv",
            out_report=tmp_path / "rep_fail.html",
            state_dir=tmp_path / "st_fail",
            kernel_names=["mock_fail_kernel"],
            n_tournament_reads=100,
            threads=1,
        )


def test_lesser_l65_and_l68_malariavar_warm_query_subset_nperm_hypothesis_and_chunking(tmp_path, monkeypatch):
    from focaloop.malariavar import query as MQ

    cohort_dir = tmp_path / "mv_state" / "cohort"
    warm_dir = cohort_dir / "warm"
    warm_dir.mkdir(parents=True)

    # Build a synthetic MalariaVar Warm state with 8 samples (2 subsets x 2 countries) and 600 panel sites
    samples_df = pd.DataFrame(
        [
            {"sample": "gh1", "Country": "Ghana", "Population": "AF-W", "Year": 2020},
            {"sample": "gh2", "Country": "Ghana", "Population": "AF-W", "Year": 2020},
            {"sample": "gh3", "Country": "Ghana", "Population": "AF-W", "Year": 2021},
            {"sample": "gh4", "Country": "Ghana", "Population": "AF-W", "Year": 2021},
            {"sample": "kh1", "Country": "Cambodia", "Population": "AS-SE-E", "Year": 2020},
            {"sample": "kh2", "Country": "Cambodia", "Population": "AS-SE-E", "Year": 2020},
            {"sample": "kh3", "Country": "Cambodia", "Population": "AS-SE-E", "Year": 2021},
            {"sample": "kh4", "Country": "Cambodia", "Population": "AS-SE-E", "Year": 2021},
        ]
    )
    n_sites = 600
    wsaf_u8 = np.zeros((8, n_sites), dtype=np.uint8)
    dp_u8 = np.full((8, n_sites), 25, dtype=np.uint8)
    # Sites 0..299 on contig1 differentiated in 2020; sites 300..599 on contig2 differentiated in 2021
    wsaf_u8[4:6, :300] = 200
    wsaf_u8[6:8, 300:] = 200

    class _FakeGenome:
        names = ["Pf3D7_07_v3", "Pf3D7_13_v3"]
        lengths = np.array([1_500_000, 3_000_000], dtype=np.int64)
        genes = pd.DataFrame(
            [{"contig": "Pf3D7_13_v3", "start": 1_720_000, "end": 1_726_000}],
            index=["PF3D7_1343700"],
        )

        def goff(self, contig: str, pos: int) -> int:
            base = 0 if contig == "Pf3D7_07_v3" else 1_500_000
            return base + int(pos)

    class _FakeWarm:
        def __init__(self, cdir, load_dothash=False, load_esscolor=False):
            self.samples = samples_df
            self.wsaf = wsaf_u8
            self.dp = dp_u8
            self.g = _FakeGenome()
            goffs = np.concatenate(
                [
                    np.linspace(100_000, 400_000, 300, dtype=np.int64),
                    np.linspace(1_500_000 + 1_710_000, 1_500_000 + 1_740_000, 300, dtype=np.int64),
                ]
            )
            self.panel = {"goff": goffs}

    monkeypatch.setattr(MQ, "Warm", _FakeWarm)
    monkeypatch.setattr(MQ, "KNOWN", {"kelch13": "PF3D7_1343700"})
    monkeypatch.setattr(MQ, "mutant_flags", lambda df: pd.DataFrame(index=df.index))
    monkeypatch.setattr(MQ, "cnv_matrix", lambda wm: pd.DataFrame(index=wm.samples.index))

    mv_root = tmp_path / "mv_state"
    HypothesisLedger.from_store(FocaloopStateStore(mv_root), alpha_budget=0.05)
    write_json(
        mv_root / STATE_META,
        {"engine": "malariavar", "version": "0.4.0", "blueprint_id": "malariavar", "samples": samples_df.to_dict("records")},
    )

    # Verify --subset Year=2021, --n-perm 499 (NOT overwritten to 49/199!), and --hypothesis kelch13
    res = run_query(
        mv_root,
        "Country=Cambodia:Ghana",
        "human",
        tmp_path / "mv.tsv",
        tmp_path / "mv.json",
        hypothesis="kelch13",
        subset="Year=2021",
        n_perm=499,
        mode="fst",
    )
    assert res["n_perm"] == 499
    assert res["informative_sites"] == 300
    assert res["genome_fst"] > 0.9

    with pytest.raises(ValueError, match="overlap"):
        run_query(mv_root, "Country=Ghana,Cambodia:Ghana", "human", tmp_path / "b.tsv", tmp_path / "b.json", mode="fst")


def test_lesser_l29_fastq_block_reader_validation_and_truncated_gzip(tmp_path):
    # 1. HTML error payload rejected
    html_f = tmp_path / "err.fq"
    html_f.write_text("<html><body>502 Bad Gateway</body></html>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="HTML"):
        list(FastqBlockReader(LocalFileSource(html_f), "r_html"))

    # 2. Missing '+' separator / mismatched qual length rejected
    bad_fq = tmp_path / "bad.fq"
    bad_fq.write_text("@r1\nACGTACGT\nNOT_PLUS\nIIIIIIII\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separator"):
        list(FastqBlockReader(LocalFileSource(bad_fq), "r_plus"))

    bad_len = tmp_path / "bad_len.fq"
    bad_len.write_text("@r1\nACGTACGT\n+\nIII\n", encoding="utf-8")
    with pytest.raises(ValueError, match="length mismatch"):
        list(FastqBlockReader(LocalFileSource(bad_len), "r_len"))

    # 3. Truncated trailing FASTQ record at EOF rejected
    trunc_plain = tmp_path / "trunc.fq"
    trunc_plain.write_text("@r1\nACGT\n+\nIIII\n@r2\nACGT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Truncated FASTQ"):
        list(FastqBlockReader(LocalFileSource(trunc_plain), "r_trunc"))

    # 4. Truncated gzip member rejected (never silently emitted as complete!)
    gz_f = tmp_path / "trunc.fq.gz"
    full_gz = gzip.compress(b"@r1\nACGTACGT\n+\nIIIIIIII\n" * 50)
    gz_f.write_bytes(full_gz[:-10])
    with pytest.raises(ValueError, match="Truncated|Corrupt"):
        list(FastqBlockReader(LocalFileSource(gz_f), "r_gz_trunc", block_bytes=128))

    # 5. Valid multi-member gzip FASTQ streams cleanly
    ok_gz = tmp_path / "ok.fq.gz"
    ok_gz.write_bytes(
        gzip.compress(b"@r1/1 comment\nACGT\n+\nIIII\n") + gzip.compress(b"@r2/1 comment\nTGCA\n+\nJJJJ\n")
    )
    blks = list(FastqBlockReader(LocalFileSource(ok_gz), "r_ok", block_bytes=32))
    all_reads = [r for b in blks for r in b.reads]
    assert all_reads == ["ACGT", "TGCA"]


def test_lesser_l69_and_l70_cvmfs_circular_origin_and_url_exists_retries(tmp_path, monkeypatch):
    # L69: Origin-spanning feature on Is_circular=true replicon (phiX174 = 5386 bp) passes verification
    circ_gff = tmp_path / "phix174.gff3"
    circ_gff.write_text(
        "##gff-version 3\n"
        "NC_001422.1\tRefSeq\tregion\t1\t5386\t.\t+\t.\tID=NC_001422.1:1..5386;Is_circular=true\n"
        "NC_001422.1\tRefSeq\tgene\t5190\t5420\t.\t+\t.\tID=gene-A;locus_tag=PHIX_01\n"
        "NC_001422.1\tRefSeq\tCDS\t5190\t5420\t.\t+\t0\tID=cds-A;Parent=gene-A;locus_tag=PHIX_01\n"
        "NC_001422.1\tRefSeq\tgene\t100\t800\t.\t+\t.\tID=gene-B;locus_tag=PHIX_02\n",
        encoding="utf-8",
    )
    res_circ = cvmfs_annot.verify_annotation_against_fasta(circ_gff, {"NC_001422.1": 5386}, min_genes=1)
    assert res_circ["ok"] is True
    assert res_circ["n_oob"] == 0

    # Linear contig with same coordinates MUST fail with n_oob == 2
    lin_gff = tmp_path / "linear.gff3"
    lin_gff.write_text(
        circ_gff.read_text(encoding="utf-8").replace("Is_circular=true", "Is_circular=false"),
        encoding="utf-8",
    )
    res_lin = cvmfs_annot.verify_annotation_against_fasta(lin_gff, {"NC_001422.1": 5386}, min_genes=1)
    assert res_lin["ok"] is False
    assert res_lin["n_oob"] == 2

    # L70: _url_exists retries transient 503 -> 200, returns False on 404, and raises on persistent 503/network failure
    monkeypatch.setattr(cvmfs_annot.time, "sleep", lambda _: None)
    seq_codes = [503, 200]

    class _Resp:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def close(self):
            pass

    class _SessSeq:
        def head(self, url, **kw):
            return _Resp(seq_codes.pop(0))

    monkeypatch.setattr(cvmfs_annot, "_ipv4_session", lambda: _SessSeq())
    assert cvmfs_annot._url_exists("https://hgdownload.soe.ucsc.edu/ncbiRefSeq.gtf.gz", retries=3) is True

    seq_codes[:] = [404]
    assert cvmfs_annot._url_exists("https://hgdownload.soe.ucsc.edu/missing.gtf.gz", retries=3) is False
    assert seq_codes == []  # 404 returns False immediately without wasting retries

    seq_codes[:] = [503, 503, 503]
    with pytest.raises(RuntimeError, match="Failed probing"):
        cvmfs_annot._url_exists("https://hgdownload.soe.ucsc.edu/ncbiRefSeq.gtf.gz", retries=3)


def test_lesser_l73_archive_atomic_pack_and_manifest_integrity(tmp_path, monkeypatch):
    st_dir = _make_variation_state(
        tmp_path / "st_ok",
        [{"sample": "s1", "group": "A", "reads_counted": 100}, {"sample": "s2", "group": "B", "reads_counted": 100}],
        [{"sample_id": "s1", "pos": 100, "gt": "1/1"}],
    )
    out_arc = tmp_path / "bundle.focaloop.tar.zst"
    archive.pack_state(st_dir, out_arc)
    assert out_arc.is_file()

    # Round-trip unpack verifies manifest and leaves no internal _archive_manifest.json file in dest
    unpacked = archive.unpack_state(out_arc, tmp_path / "unpacked_ok")
    assert (unpacked / "focaloop_state.json").is_file()
    assert (unpacked / "variants.parquet").is_file()
    assert not (unpacked / archive.ARCHIVE_MANIFEST_NAME).exists()

    # Failed pack_state leaves no partial archive at out_path (atomic temp file)
    bad_out = tmp_path / "should_not_exist.focaloop.tar.zst"
    orig_hash = archive._hash_file
    monkeypatch.setattr(archive, "_hash_file", orig_hash)
    bad_file = st_dir / "unreadable.bin"
    bad_file.write_bytes(b"hello")
    real_open = open

    def _fail_mid_pack(file, mode="r", *a, **kw):
        if Path(file).name == "unreadable.bin" and "rb" in mode:
            # First call hashes, second call inside tar loop fails mid-stream!
            _fail_mid_pack.calls += 1
            if _fail_mid_pack.calls >= 2:
                raise OSError("simulated disk read failure mid-tar")
        return real_open(file, mode, *a, **kw)

    _fail_mid_pack.calls = 0
    monkeypatch.setattr("builtins.open", _fail_mid_pack)
    with pytest.raises(OSError, match="simulated disk read failure"):
        archive.pack_state(st_dir, bad_out)
    monkeypatch.undo()
    bad_file.unlink(missing_ok=True)
    assert not bad_out.exists()
    assert not list(tmp_path.glob("*.part"))

    # Complete zstd frame wrapping an incomplete tar (missing variants.parquet) is rejected on unpack!
    incomplete_arc = tmp_path / "incomplete.focaloop.tar.zst"
    with pa.CompressedOutputStream(str(incomplete_arc), "zstd") as fh:
        with tarfile.open(fileobj=fh, mode="w|", format=tarfile.PAX_FORMAT) as tf:
            meta_bytes = (st_dir / "focaloop_state.json").read_bytes()
            ti = tarfile.TarInfo(name="focaloop_state/focaloop_state.json")
            ti.size = len(meta_bytes)
            tf.addfile(ti, io.BytesIO(meta_bytes))
    with pytest.raises(ValueError, match="incomplete variationloop state archive"):
        archive.unpack_state(incomplete_arc, tmp_path / "unpacked_bad")


def test_lesser_l95_l99_l101_wrapper_tests_containers_and_galaxy_setup(tmp_path):
    # L95: Parse focaloop_query.xml and run every <test> block end-to-end via CLI
    xml_path = REPO_ROOT / "tools" / "focaloop" / "focaloop_query.xml"
    root = ET.parse(xml_path).getroot()
    test_els = root.findall("./tests/test")
    assert len(test_els) >= 4

    for idx, t_el in enumerate(test_els, 1):
        state_val = t_el.find("./param[@name='state']").attrib["value"]
        cond_el = t_el.find("./conditional[@name='query_type']")
        q_params = {p.attrib["name"]: p.attrib["value"] for p in cond_el.findall("./param")}
        mode = q_params["mode"]
        if mode == "contrast":
            contrast = f"{q_params['factor_name']}:{q_params['group_a']}:{q_params['group_b']}"
        else:
            contrast = f"{q_params['group_column']}={q_params['group_a']}:{q_params['group_b']}"

        out_tbl = tmp_path / f"t{idx}_table.tsv"
        out_ver = tmp_path / f"t{idx}_verdict.json"
        out_rep = tmp_path / f"t{idx}_report.html"
        out_arc = tmp_path / f"t{idx}_state.focaloop.tar.zst"
        argv = [
            "query",
            "--state",
            str(TEST_DATA / state_val),
            "--mode",
            mode,
            "--contrast",
            contrast,
            "--out-table",
            str(out_tbl),
            "--out-verdict",
            str(out_ver),
            "--out-report",
            str(out_rep),
            "--out-state-archive",
            str(out_arc),
        ]
        if q_params.get("subset"):
            argv.extend(["--subset", q_params["subset"]])
        if q_params.get("hypothesis"):
            argv.extend(["--hypothesis", q_params["hypothesis"]])
        if q_params.get("design"):
            argv.extend(["--design", q_params["design"]])

        assert cli.main(argv) == 0
        out_map = {
            "query_table": out_tbl,
            "verdict": out_ver,
            "query_report": out_rep,
            "state_out": out_arc,
        }
        for out_el in t_el.findall("./output"):
            oname = out_el.attrib["name"]
            opath = out_map[oname]
            ac = out_el.find("./assert_contents")
            for ht in ac.findall("./has_text"):
                txt = opath.read_text(encoding="utf-8")
                assert ht.attrib["text"] in txt, f"Test {idx} {oname} missing {ht.attrib['text']!r}"
            for hs in ac.findall("./has_size"):
                assert opath.stat().st_size >= int(hs.attrib["min"])

    # L99: Dockerfile and Apptainer.def bundle tiberius_weights_v2 and smoke-check mammalia_softmasking_v2
    for cfile in (REPO_ROOT / "container" / "Dockerfile", REPO_ROOT / "container" / "Apptainer.def"):
        ctxt = cfile.read_text(encoding="utf-8")
        assert "tiberius_weights_v2.tar.gz" in ctxt
        assert "mammalia_softmasking_v2" in ctxt

    # L101: galaxy/setup_galaxy25.sh derives version from macros.xml, pulls ghcr.io/nekrut/focaloop, validates geneloop, and has zero personal paths
    setup_sh = (REPO_ROOT / "galaxy" / "setup_galaxy25.sh").read_text(encoding="utf-8")
    assert "macros.xml" in setup_sh
    assert "ghcr.io/nekrut/focaloop:${FOCALOOP_VERSION}" in setup_sh
    assert "ghcr.io/galaxyproject/focaloop" not in setup_sh
    assert "geneloop" in setup_sh
    assert "/media/" not in setup_sh and "/Users/" not in setup_sh
