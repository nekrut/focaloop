"""Compact Zstd Parquet state store: loci.parquet, junctions.parquet, telemetry.parquet (+ models).

Follow-up locus / isoform queries read only these files (no re-download, no BAM) and run in <1 s.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .evidence import JunctionStore
from .gffio import Transcript

ZSTD = {"compression": "zstd", "compression_level": 9}


def write_loci(path: Path, loci, models: list[Transcript], locus_wave: dict | None = None) -> int:
    """One row per partition locus and one per emitted transcript (kind='locus'|'model')."""
    rows = {k: [] for k in ("kind", "id", "gene_id", "contig", "strand", "start", "end", "status", "category",
                            "n_exons", "n_cds", "exons", "cds", "evidence", "saturated_wave")}
    lw = locus_wave or {}
    for l in loci:
        rows["kind"].append("locus")
        rows["id"].append(l.locus_id)
        rows["gene_id"].append("")
        rows["contig"].append(l.contig)
        rows["strand"].append(l.strand)
        rows["start"].append(l.start)
        rows["end"].append(l.end)
        rows["status"].append(l.status)
        rows["category"].append(l.category)
        lm = l.locked_model
        rows["n_exons"].append(len(lm.exons) if lm else 0)
        rows["n_cds"].append(len(lm.cds) if lm else 0)
        rows["exons"].append(json.dumps(lm.exons) if lm else "")
        rows["cds"].append(json.dumps(lm.cds) if lm else "")
        rows["evidence"].append(f"tib={len(l.tiberius)};hits={len(l.hits)}")
        rows["saturated_wave"].append(int(lw.get(l.locus_id, -1)))
    for t in models:
        rows["kind"].append("model")
        rows["id"].append(t.tid)
        rows["gene_id"].append(t.gid)
        rows["contig"].append(t.contig)
        rows["strand"].append(t.strand)
        rows["start"].append(t.start)
        rows["end"].append(t.end)
        rows["status"].append(t.attrs.get("locus_status", ""))
        rows["category"].append("")
        rows["n_exons"].append(len(t.exons))
        rows["n_cds"].append(len(t.cds))
        rows["exons"].append(json.dumps(t.exons))
        rows["cds"].append(json.dumps(t.cds))
        rows["evidence"].append(t.attrs.get("evidence", ""))
        rows["saturated_wave"].append(-1)
    tab = pa.table(rows).sort_by([("contig", "ascending"), ("start", "ascending")])
    pq.write_table(tab, path, row_group_size=50000, **ZSTD)
    return tab.num_rows


def write_junctions(path: Path, js: JunctionStore, regions: dict | None = None) -> int:
    runs = js.run_ids
    cols: dict[str, list] = {"contig": [], "start": [], "end": [], "strand": [], "motif": [], "support": [],
                             "total_reads": [], "max_anchor": [], "n_runs": [], "region": []}
    per_run = {r: [] for r in runs}
    reg = regions or {}
    for k, d in js.ua.items():
        m = js.motif.get(k, ("", "."))
        cols["contig"].append(k[0])
        cols["start"].append(k[1])
        cols["end"].append(k[2])
        cols["strand"].append(m[1])
        cols["motif"].append(m[0])
        cols["support"].append(sum(d.values()))
        cols["total_reads"].append(js.total.get(k, 0))
        cols["max_anchor"].append(js.max_anchor.get(k, 0))
        cols["n_runs"].append(sum(1 for v in d.values() if v))
        cols["region"].append(reg.get(k, ""))
        for r in runs:
            per_run[r].append(d.get(r, 0))
    for r in runs:
        cols[f"run_{r}"] = per_run[r]
    tab = pa.table(cols).sort_by([("contig", "ascending"), ("start", "ascending")])
    pq.write_table(tab, path, row_group_size=100000, **ZSTD)
    return tab.num_rows


def read_junctions(path: Path) -> tuple[JunctionStore, dict]:
    """Reconstruct JunctionStore and per-junction region labels losslessly from junctions.parquet."""
    tab = pq.read_table(path)
    run_cols = [c for c in tab.schema.names if c.startswith("run_")]
    run_ids = [c[4:] for c in run_cols]
    js = JunctionStore(run_ids=list(run_ids))
    regions: dict = {}
    data = tab.to_pydict()
    n = tab.num_rows
    ctgs = data["contig"]
    starts = data["start"]
    ends = data["end"]
    strands = data["strand"]
    motifs = data["motif"]
    totals = data["total_reads"]
    anchors = data["max_anchor"]
    regs = data["region"]
    per_run_cols = [(rid, data[col]) for rid, col in zip(run_ids, run_cols)]
    for i in range(n):
        k = (ctgs[i], int(starts[i]), int(ends[i]))
        js.ua[k] = {rid: int(vals[i]) for rid, vals in per_run_cols if vals[i]}
        js.total[k] = int(totals[i])
        js.max_anchor[k] = int(anchors[i])
        js.motif[k] = (motifs[i], strands[i])
        if regs[i]:
            regions[k] = regs[i]
    return js, regions


def write_telemetry(path: Path, rows: list[dict]) -> int:
    if not rows:
        rows = [{"stage": "none"}]
    keys = sorted({k for r in rows for k in r})
    cols = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            cols[k].append(json.dumps(v) if isinstance(v, (dict, list)) else (None if v is None else str(v) if not isinstance(v, (int, float, bool)) else v))
    # normalise mixed-type columns to string
    arrays = {}
    for k, v in cols.items():
        try:
            arrays[k] = pa.array(v)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            arrays[k] = pa.array([None if x is None else str(x) for x in v])
    tab = pa.table(arrays)
    pq.write_table(tab, path, **ZSTD)
    return tab.num_rows


def query(state_dir: Path, contig: str, start: int, end: int) -> dict:
    """Zero-download follow-up: everything known about [start, end) on contig. Returns rows + latency."""
    t0 = time.perf_counter()
    filt = [("contig", "=", contig), ("start", "<", end), ("end", ">", start)]
    loci = pq.read_table(state_dir / "loci.parquet", filters=filt)
    j = pq.read_table(state_dir / "junctions.parquet", filters=filt,
                      columns=["contig", "start", "end", "strand", "motif", "support", "n_runs", "region"])
    dt = time.perf_counter() - t0
    return {"loci": loci.to_pylist(), "junctions": j.to_pylist(), "seconds": dt}


def query_gene(state_dir: Path, gene_id: str) -> dict:
    t0 = time.perf_counter()
    loci = pq.read_table(state_dir / "loci.parquet")
    m = loci.filter(pc.equal(loci["gene_id"], gene_id))
    rows = m.to_pylist()
    out = {"models": rows}
    if rows:
        ctg, a, b = rows[0]["contig"], min(r["start"] for r in rows), max(r["end"] for r in rows)
        j = pq.read_table(state_dir / "junctions.parquet", filters=[("contig", "=", ctg), ("start", ">=", a), ("end", "<=", b)],
                          columns=["start", "end", "strand", "motif", "support", "n_runs"])
        out["junctions"] = j.to_pylist()
    out["seconds"] = time.perf_counter() - t0
    return out
