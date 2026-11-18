"""Zero-new-byte warm-state query, Stage-0 bait screen, and ReadUntil steering export."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .archive import materialize_state, pack_state
from .engine import STATE_META, write_json
from .hooks import BaitPrefilter, HypothesisLedger, SteeringMask
from .state import FocaloopStateStore
from .stats import contrast_table, fst_permutation
from .streamer import FastqBlockReader, open_source, read_fasta


def parse_contrast(text: str) -> tuple[str, tuple[str, str]]:
    """``treated:control`` | ``treated,control`` | ``treated_vs_control`` | ``site=Mali:Ghana``."""
    field = "condition"
    if "=" in text:
        field, text = text.split("=", 1)
    for sep in (":", ",", "_vs_"):
        if sep in text:
            a, b = text.split(sep, 1)
            if a and b:
                return field.strip(), (a.strip(), b.strip())
    raise ValueError(f"cannot parse contrast {text!r}; use A:B or field=A:B")


def _meta(state_dir: Path) -> dict[str, Any]:
    p = state_dir / STATE_META
    if not p.exists():
        raise FileNotFoundError(f"{state_dir}: missing {STATE_META}; not a focaloop state")
    return json.loads(p.read_text(encoding="utf-8"))


def run_query(state_ref: str | Path, contrast: str, actor: str, out_table: str | Path, out_verdict: str | Path,
              hypothesis: str | None = None, mode: str = "auto", out_state_archive: str | Path | None = None,
              alpha_budget: float = 0.05, n_perm: int = 199, out_report: str | Path | None = None,
              design: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    state_dir, tmp = materialize_state(state_ref)
    try:
        meta = _meta(state_dir)
        if meta.get("engine") in ("panexpr", "malariavar"):
            from .galaxy_runners import run_warm_engine_query

            return run_warm_engine_query(
                state_dir=state_dir,
                meta=meta,
                contrast=contrast,
                actor=actor,
                out_table=out_table,
                out_verdict=out_verdict,
                hypothesis=hypothesis,
                design_override=design,
                out_report=out_report,
                out_state_archive=out_state_archive,
                alpha_budget=alpha_budget,
            )
        store = FocaloopStateStore(state_dir)
        field, (a, b) = parse_contrast(contrast)
        level = {s["sample"]: str(s.get(field, "")) for s in meta["samples"]}
        missing = [x for x in (a, b) if x not in level.values()]
        if missing:
            raise ValueError(f"contrast levels {missing} not present in sample field {field!r}")
        if mode == "auto":
            mode = "fst" if store.exists("allele_counts") else "contrast"
        ledger = HypothesisLedger.from_store(store, alpha_budget=alpha_budget)
        if mode == "fst":
            alleles = store.read("allele_counts").to_pandas()
            rows = []
            res: dict[str, tuple[float, float]] = {}
            for part in ("discovery", "holdout"):
                sub = alleles[alleles["partition"] == part]
                res[part] = fst_permutation(sub, level, (a, b), n_perm=n_perm)
                rows.append({"partition": part, "statistic": "hudson_fst", "estimate": res[part][0],
                             "pvalue": res[part][1]})
            table = pd.DataFrame(rows)
            spec = hypothesis or f"F_ST({field}: {a} vs {b}) > 0"
            d_est, d_p = res["discovery"]
            h_est, h_p = res["holdout"]
        else:
            acc = store.read("target_accumulators").to_pandas()
            disc = contrast_table(acc, level, (a, b), partition="discovery")
            hold = contrast_table(acc, level, (a, b), partition="holdout")
            table = disc.merge(hold[["group_id", "log2fc", "se", "pvalue"]], on="group_id", how="left",
                               suffixes=("_discovery", "_holdout"))
            table = table.rename(columns={"padj": "padj_discovery", "z": "z_discovery"})
            if table.empty:
                raise ValueError("no testable groups on the discovery partition")
            if hypothesis:
                gid = hypothesis.split("=", 1)[1] if "=" in hypothesis else hypothesis
                hit = table[table["group_id"] == gid]
                if hit.empty:
                    raise ValueError(f"hypothesis group {gid!r} not found in state")
                row = hit.iloc[0]
            else:
                row = table.iloc[0]
            gid = row["group_id"]
            spec = f"{field}:{a}_vs_{b}:group={gid}:log2fc!=0"
            d_est, d_p = float(row["log2fc_discovery"]), float(row["pvalue_discovery"])
            h_est = float(row["log2fc_holdout"]) if not pd.isna(row["log2fc_holdout"]) else 0.0
            h_p = float(row["pvalue_holdout"]) if not pd.isna(row["pvalue_holdout"]) else 1.0
        clean = lambda v: 0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)  # noqa: E731
        qid = f"q{(ledger._query_index + 1):04d}"
        verdict = ledger.evaluate_query(qid, actor, spec, clean(d_est), clean(d_p) if d_p == d_p else 1.0,
                                        clean(h_est), clean(h_p) if h_p == h_p else 1.0, store=store)
        table.to_csv(out_table, sep="\t", index=False, float_format="%.6g")
        out = {
            "tool": "focaloop", "command": "query", "mode": mode, "contrast": {"field": field, "a": a, "b": b},
            "new_bytes": 0, "wall_seconds": round(time.time() - t0, 3), "verdict": verdict.__dict__,
            "blueprint_id": meta.get("blueprint_id"),
        }
        write_json(out_verdict, out)
        if out_report:
            from .reports import render_query_report

            vdict = {
                **verdict.__dict__,
                "wall_seconds": out["wall_seconds"],
                "headline_metric": f"{mode}: {clean(d_est):.4g}",
                "headline_detail": spec,
            }
            render_query_report(
                out_report,
                engine=str(meta.get("blueprint_id") or meta.get("engine") or mode),
                contrast_str=contrast,
                verdict=vdict,
                table_df=table,
            )
        if out_state_archive:
            pack_state(state_dir, out_state_archive)
        return out
    finally:
        if tmp is not None:
            tmp.cleanup()


def run_bait(query_fasta: str | Path, inputs: Sequence[str], state_dir: str | Path, out_tsv: str | Path,
             probe_bytes: int = 1 << 20, min_kmer_hits: int = 2, min_hit_reads: int = 2, k: int = 31,
             run_ids: Sequence[str] | None = None) -> pd.DataFrame:
    recs = read_fasta(query_fasta)
    if not recs:
        raise ValueError(f"{query_fasta}: no sequences")
    qname = recs[0][0] if len(recs) == 1 else Path(str(query_fasta)).stem
    bait = BaitPrefilter(qname, [r[1] for r in recs], k=k, min_kmer_hits_per_read=min_kmer_hits,
                         min_hit_reads=min_hit_reads)
    store = FocaloopStateStore(state_dir)
    rows = []
    for i, ref in enumerate(inputs):
        rid = run_ids[i] if run_ids else _run_id(ref)
        src = open_source(ref, layout="SE")[0]
        rd = FastqBlockReader(src, rid, block_bytes=probe_bytes, holdout_fraction=0.0)
        blk = rd.next_block()
        reads = blk.reads if blk else []
        s = bait.screen_reads(rid, reads, rd.consumed_bytes, stream_tier=src.tier, store=store)
        rows.append(s.__dict__)
    df = pd.DataFrame(rows, columns=["query_name", "run_id", "stream_tier", "probe_bytes", "probe_reads", "hit_reads",
                                     "hit_rate", "wilson_ci95_low", "pass_gate"])
    df.sort_values(["pass_gate", "wilson_ci95_low", "run_id"], ascending=[False, False, True]).to_csv(
        out_tsv, sep="\t", index=False, float_format="%.6g")
    return df


def _run_id(ref: str) -> str:
    name = Path(str(ref)).name
    for ext in (".fastq.gz", ".fq.gz", ".fastq", ".fq", ".gz"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def run_steer(state_ref: str | Path, coords_bed: str | Path, out_bed: str | Path,
              out_telemetry: str | Path | None = None, min_precision: float | None = None,
              iteration: int | None = None, out_state_archive: str | Path | None = None) -> dict[str, Any]:
    state_dir, tmp = materialize_state(state_ref)
    try:
        meta = _meta(state_dir)
        store = FocaloopStateStore(state_dir)
        acc_all = store.read("target_accumulators").to_pandas()
        disc = acc_all[acc_all["partition"] == "discovery"]
        holdout_fraction = float(meta.get("stop_gate", {}).get("holdout_fraction", 0.0))
        # Steering is an adaptive decision: never let holdout counts influence it.
        use_disc = holdout_fraction > 0 and not disc.empty and disc["effective_count"].sum() > 0
        acc = disc if use_disc else acc_all
        thr = float(min_precision if min_precision is not None else meta["stop_gate"]["min_posterior_precision"])
        g = acc.groupby("group_id")[["effective_count", "assignment_variance"]].sum()
        tech = (g["assignment_variance"] + g["effective_count"]) / (g["effective_count"] + 0.5) ** 2
        precision = 1.0 / tech.clip(lower=1e-12)
        settled = set(precision[precision >= thr].index)
        seqs = {tid: seq for tid, seq, _ in read_fasta(state_dir / meta["targets_file"])}
        group_targets: dict[str, list[str]] = acc_all.groupby("group_id")["target_id"].unique().apply(list).to_dict()
        enrich = {t: seqs[t] for gid, ts in group_targets.items() if gid not in settled for t in ts if t in seqs}
        deplete = {t: seqs[t] for gid, ts in group_targets.items() if gid in settled for t in ts if t in seqs}
        mask = SteeringMask(enrich, deplete)
        coords: dict[str, tuple[str, int, int]] = {}
        for line in Path(coords_bed).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            name = f[3]
            coords[name] = (f[0], int(f[1]), int(f[2]))
            for t in group_targets.get(name, []):
                coords.setdefault(t, (f[0], int(f[1]), int(f[2])))
        mask.export_read_until_bed(coords, out_bed)
        it = iteration
        if it is None:
            it = (store.read("steering_decisions").num_rows + 1) if store.exists("steering_decisions") else 1
        reason = "all targets settled: halt" if mask.halt_run else f"{len(enrich)} unsettled targets below precision {thr}"
        mask.record_telemetry(store, it, reason)
        out = {
            "tool": "focaloop", "command": "steer", "halt_run": bool(mask.halt_run), "iteration": it,
            "n_enrich_targets": len(mask.enrich_targets), "n_deplete_targets": len(mask.deplete_targets),
            "n_enrich_kmers": int(mask.enrich_kmers.size), "n_deplete_kmers": int(mask.deplete_kmers.size),
            "min_posterior_precision": thr, "reason": reason, "new_bytes": 0,
            "partitions": ["discovery"] if use_disc else ["discovery", "holdout"],
            "group_precision": {k: float(v) for k, v in precision.items()},
        }
        if out_telemetry:
            write_json(out_telemetry, out)
        if out_state_archive:
            pack_state(state_dir, out_state_archive)
        return out
    finally:
        if tmp is not None:
            tmp.cleanup()
