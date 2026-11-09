"""End-to-end synthetic `focaloop run -> query -> bait -> steer`, plus trap firing."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pandas as pd
import yaml

from focaloop.archive import is_state_archive, unpack_state
from focaloop.state import SCHEMA_REGISTRY, FocaloopStateStore

from .conftest import BLUEPRINTS, run_cli


def test_run_emits_table_state_archive_and_telemetry(panexpr_run) -> None:
    assert panexpr_run["rc"] == 0
    d = panexpr_run["dir"]
    tel = panexpr_run["telemetry"]
    assert tel["status"] == "ok" and tel["trap_codes"] == [] and tel["kernel"] == "kmer_eq"
    assert tel["tournament"]["n_synthetic"] == 3000 and tel["tournament"]["winner"] == "kmer_eq"
    assert tel["total_consumed_bytes"] == sum(tel["consumed_bytes"].values()) > 0
    assert tel["iterations"][-1]["cum_bytes"] == tel["total_consumed_bytes"]
    de = pd.read_csv(d / "de.tsv", sep="\t").set_index("group_id")
    assert de.loc["g2", "log2fc"] > 1.5 and de.loc["g2", "padj"] < 1e-6      # planted 5x up
    assert de.loc["g4", "log2fc"] < -1.5 and de.loc["g4", "padj"] < 1e-6     # planted 5x down
    assert abs(de.loc["g1", "log2fc"]) < 0.6
    store = FocaloopStateStore(d / "state")
    for name in ("target_accumulators", "ambiguous_eq", "iteration_telemetry", "graph_locus_map", "residual_spans"):
        assert store.exists(name)
        assert store.read(name).schema.equals(SCHEMA_REGISTRY[name])
    import pyarrow.parquet as pq
    assert pq.ParquetFile(store.path("target_accumulators")).metadata.row_group(0).column(0).compression == "ZSTD"
    assert is_state_archive(d / "state.tar.zst")
    un = unpack_state(d / "state.tar.zst", d / "unpacked")
    assert (un / "focaloop_state.json").is_file() and (un / "target_accumulators.parquet").is_file()


def test_query_is_zero_byte_and_ledger_accumulates(panexpr_run, tmp_path: Path) -> None:
    d = panexpr_run["dir"]
    out1 = tmp_path / "s1.tar.zst"
    assert run_cli("query", "--state-dir", d / "state.tar.zst", "--contrast", "treated:control", "--actor", "agent:ci",
                   "--out-table", tmp_path / "q1.tsv", "--out-verdict", tmp_path / "v1.json",
                   "--out-state-archive", out1) == 0
    v1 = json.loads((tmp_path / "v1.json").read_text())
    assert v1["new_bytes"] == 0 and v1["mode"] == "contrast"
    assert v1["verdict"]["confirmed"] is True and v1["verdict"]["hypothesis_spec"].endswith("group=g2:log2fc!=0")
    q1 = pd.read_csv(tmp_path / "q1.tsv", sep="\t")
    assert {"log2fc_discovery", "log2fc_holdout", "pvalue_holdout"} <= set(q1.columns)
    # second query on the chained state: a null group is not confirmed and costs alpha-wealth
    assert run_cli("query", "--state-dir", out1, "--contrast", "condition=treated:control", "--actor", "human",
                   "--hypothesis", "group=g6", "--out-table", tmp_path / "q2.tsv", "--out-verdict", tmp_path / "v2.json",
                   "--out-state-archive", tmp_path / "s2.tar.zst") == 0
    v2 = json.loads((tmp_path / "v2.json").read_text())["verdict"]
    assert v2["query_index"] == 2 and v2["confirmed"] is False
    assert v2["alpha_wealth_after"] < v1["verdict"]["alpha_wealth_after"]
    ledger = FocaloopStateStore(unpack_state(tmp_path / "s2.tar.zst", tmp_path / "s2")).read("hypothesis_ledger")
    assert ledger.column("actor").to_pylist() == ["agent:ci", "human"]
    assert run_cli("query", "--state-dir", out1, "--contrast", "a:b", "--actor", "human",
                   "--out-table", tmp_path / "x.tsv", "--out-verdict", tmp_path / "x.json") == 2


def test_bait_screens_runs(tmp_path: Path) -> None:
    td = BLUEPRINTS / "focaloop_bait" / "test-data"
    assert run_cli("bait", "--query-fasta", td / "query.fa", "--inputs", td / "run_hit.fastq.gz", td / "run_miss.fastq.gz",
                   td / "run_weak.fastq", "--state-dir", tmp_path / "st", "--out-tsv", tmp_path / "b.tsv") == 0
    df = pd.read_csv(tmp_path / "b.tsv", sep="\t").set_index("run_id")
    assert bool(df.loc["run_hit", "pass_gate"]) and not bool(df.loc["run_miss", "pass_gate"])
    assert df.loc["run_hit", "hit_reads"] == 40 and df.loc["run_miss", "hit_reads"] == 0
    assert 0 < df.loc["run_weak", "wilson_ci95_low"] < df.loc["run_hit", "wilson_ci95_low"]
    assert FocaloopStateStore(tmp_path / "st").read("bait_prefilter").num_rows == 3


def test_steer_exports_bed_and_halt(panexpr_run, tmp_path: Path) -> None:
    d = panexpr_run["dir"]
    bed = BLUEPRINTS / "panexpr" / "test-data" / "targets.bed"
    assert run_cli("steer", "--state-dir", d / "state.tar.zst", "--coords-bed", bed, "--out-bed", tmp_path / "a.bed",
                   "--out-telemetry", tmp_path / "a.json", "--min-precision", "1000") == 0
    tel = json.loads((tmp_path / "a.json").read_text())
    rows = [ln.split("\t") for ln in (tmp_path / "a.bed").read_text().splitlines()]
    labels = {r[3]: r[4] for r in rows}
    assert tel["halt_run"] is False and tel["n_enrich_targets"] > 0 and tel["n_deplete_targets"] > 0
    assert labels["g2.strainA"] == "deplete" and labels["phage_v1"] == "enrich"
    assert run_cli("steer", "--state-dir", d / "state.tar.zst", "--coords-bed", bed, "--out-bed", tmp_path / "b.bed",
                   "--out-telemetry", tmp_path / "b.json") == 0
    tel2 = json.loads((tmp_path / "b.json").read_text())
    assert tel2["halt_run"] is True and tel2["n_enrich_targets"] == 0


def test_variants_run_and_fst_query(tmp_path: Path) -> None:
    td = BLUEPRINTS / "malariavar" / "test-data"
    # sheet + --reads in shuffled order matched by --read-ids (the Galaxy XML path)
    order = ["GH05", "ML02", "GH01", "ML03"]
    rc = run_cli("run", "--blueprint", "malariavar", "--targets", td / "targets.fa", "--sample-sheet", td / "samples.tsv",
                 "--reads", *[td / f"{s}.fastq.gz" for s in order], "--read-ids", *[f"{s}.fastq.gz" for s in order],
                 "--variant-sites", td / "sites.tsv", "--state-dir", tmp_path / "st", "--state-archive", tmp_path / "st.tar.zst",
                 "--out-table", tmp_path / "v.tsv", "--out-telemetry", tmp_path / "t.json")
    assert rc == 0
    v = pd.read_csv(tmp_path / "v.tsv", sep="\t", keep_default_na=False)
    call = {(r.sample, r.feature_id): r.call for r in v.itertuples()}
    assert call[("GH01", "pfmdr1")] == "amplified" and call[("ML02", "pfmdr1")] == "single_copy"
    assert call[("GH05", "hrp2")] == "deleted" and call[("GH01", "hrp2")] == "present"
    assert call[("GH01", "pfcrt_K76T")] == "alt" and call[("ML02", "pfcrt_K76T")] == "ref"
    assert call[("GH01", "ctrl3_del")] == "alt"
    assert call[("ML03", "F_WS")] == "polyclonal" and call[("ML02", "F_WS")] == "monoclonal"
    tel = json.loads((tmp_path / "t.json").read_text())
    assert tel["trap_codes"] == [] and tel["map_rate"] > 0.95
    assert run_cli("query", "--state-dir", tmp_path / "st.tar.zst", "--contrast", "site=Ghana:Mali", "--actor", "human",
                   "--out-table", tmp_path / "f.tsv", "--out-verdict", tmp_path / "f.json") == 0
    out = json.loads((tmp_path / "f.json").read_text())
    assert out["mode"] == "fst" and out["new_bytes"] == 0
    f = pd.read_csv(tmp_path / "f.tsv", sep="\t").set_index("partition")
    assert f.loc["discovery", "estimate"] > 0.2


def _random_fastq(path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    with gzip.open(path, "wt") as fh:
        for i in range(n):
            s = "".join(rng.choice("ACGT") for _ in range(100))
            fh.write(f"@r{i}\n{s}\n+\n{'I' * 100}\n")


def test_trap_low_map_rate(panexpr_data: Path, tmp_path: Path) -> None:
    for s in ("ctrl1", "ctrl2", "ctrl3", "trt1", "trt2", "trt3"):
        _random_fastq(panexpr_data / f"{s}.fastq.gz", 600, hash(s) % 1000)
    rc = run_cli("run", "--blueprint", "panexpr", "--spec", panexpr_data / "spec.yml", "--state-dir", tmp_path / "st",
                 "--out-table", tmp_path / "t.tsv", "--out-telemetry", tmp_path / "tel.json")
    tel = json.loads((tmp_path / "tel.json").read_text())
    assert rc == 3 and tel["status"] == "trap"
    assert tel["trap_codes"] == ["LOW_MAP_RATE"] and tel["abort_trap"] == "LOW_MAP_RATE"
    assert (tmp_path / "st" / "residual_spans.parquet").is_file()  # unmapped bytes kept for retroactive re-query


def test_trap_tie_symmetry_fail(panexpr_data: Path, tmp_path: Path) -> None:
    doc = yaml.safe_load((BLUEPRINTS / "panexpr" / "blueprint.focaloop.yml").read_text())
    doc["kernel_pool"]["kernels"] = ["kmer_besthit"]
    bp = tmp_path / "tiebreak.focaloop.yml"
    bp.write_text(yaml.safe_dump(doc))
    rc = run_cli("run", "--blueprint", bp, "--spec", panexpr_data / "spec.yml", "--state-dir", tmp_path / "st",
                 "--out-table", tmp_path / "t.tsv", "--out-telemetry", tmp_path / "tel.json")
    tel = json.loads((tmp_path / "tel.json").read_text())
    assert rc == 3 and tel["trap_codes"] == ["TIE_SYMMETRY_FAIL"]
    assert tel["tournament"]["winner"] is None
    assert tel["tournament"]["board"][0]["mate_tie_symmetry"] < 0.80


def test_trap_confounded_design_and_byte_cap(panexpr_data: Path, tmp_path: Path) -> None:
    spec = yaml.safe_load((panexpr_data / "spec.yml").read_text())
    for s in spec["samples"]:
        s["batch"] = "B" if s["condition"] == "treated" else "A"
    (panexpr_data / "conf.yml").write_text(yaml.safe_dump(spec))
    rc = run_cli("run", "--blueprint", "panexpr", "--spec", panexpr_data / "conf.yml", "--state-dir", tmp_path / "st",
                 "--out-table", tmp_path / "t.tsv", "--out-telemetry", tmp_path / "tel.json")
    assert rc == 3 and json.loads((tmp_path / "tel.json").read_text())["trap_codes"] == ["CONFOUNDED_DESIGN"]
    rc = run_cli("run", "--blueprint", "panexpr", "--spec", panexpr_data / "spec.yml", "--state-dir", tmp_path / "st2",
                 "--out-table", tmp_path / "t2.tsv", "--out-telemetry", tmp_path / "tel2.json", "--max-bytes", "60000")
    tel = json.loads((tmp_path / "tel2.json").read_text())
    assert rc == 0 and tel["status"] == "ok_with_warnings"
    assert tel["trap_codes"] == ["BYTE_CAP_UNCONVERGED"] and tel["total_consumed_bytes"] <= 60000 + 6 * 8192
