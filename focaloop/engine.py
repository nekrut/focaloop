"""`focaloop run`: stream blocks, map with the tournament winner, condition, gate, persist."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from .archive import pack_state
from .blueprint import Blueprint, regime_settings
from .conditioner import EqClass, backward_condition, solve_em
from .hooks import HolobiontLocusRegistry, coalesce_residual_spans
from .kernels import KmerIndex, TargetRec, build_kernels
from .state import EXTERNAL_PRIORS_SCHEMA, FocaloopStateStore
from .stats import contrast_table
from .stop_gate import StopGate, check_design
from .streamer import FastqBlock, FastqBlockReader, open_source, read_fasta
from .tournament import run_tournament
from .variants import ALLELE_COUNTS_SCHEMA, AlleleProbes, load_sites, variant_table

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_TRAP = 3
STATE_META = "focaloop_state.json"


class TrapAbort(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class RunSpec:
    targets: Path
    samples: list[dict[str, Any]]
    contrast: tuple[str, str] | None
    priors: Path | None = None
    sites: Path | None = None
    base: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> "RunSpec":
        p = Path(path)
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        base = p.parent

        def rel(x: str | None) -> Path | None:
            if not x:
                return None
            q = Path(x)
            return q if q.is_absolute() else base / q

        samples = []
        for s in doc["samples"]:
            s = dict(s)
            s.setdefault("run_id", s["sample"])
            s.setdefault("batch", "")
            src = str(s["reads"])
            if not src.startswith(("http://", "https://")) and not src[:3] in ("SRR", "ERR", "DRR"):
                src = str(rel(src))
            s["reads"] = src
            samples.append(s)
        c = doc.get("contrast")
        return cls(rel(doc["targets"]), samples, tuple(c) if c else None, rel(doc.get("priors")),
                   rel(doc.get("variant_sites")), base)

    @classmethod
    def from_sheet(cls, targets: str | Path, sheet: str | Path, reads: list[str], contrast: tuple[str, str] | None,
                   priors: str | Path | None = None, sites: str | Path | None = None,
                   read_ids: list[str] | None = None) -> "RunSpec":
        df = pd.read_csv(sheet, sep="\t", dtype=str).fillna("")
        if len(df) != len(reads):
            raise ValueError(f"sample sheet has {len(df)} rows but {len(reads)} read files were given")
        if read_ids:
            if len(read_ids) != len(reads):
                raise ValueError("--read-ids must have one id per --reads entry")
            by_id = {_strip_ext(i): r for i, r in zip(read_ids, reads)}
            missing = [s_ for s_ in df["sample"] if s_ not in by_id]
            if missing:
                raise ValueError(f"no reads for sample(s) {missing}; read ids were {sorted(by_id)}")
            reads = [by_id[s_] for s_ in df["sample"]]
        samples = []
        for row, r in zip(df.to_dict("records"), reads):
            row = {k: v for k, v in row.items() if k != "reads"}
            row.setdefault("run_id", row["sample"])
            if not row["run_id"]:
                row["run_id"] = row["sample"]
            row.setdefault("batch", "")
            row["reads"] = r
            samples.append(row)
        return cls(Path(targets), samples, contrast, Path(priors) if priors else None, Path(sites) if sites else None)


def _strip_ext(name: str) -> str:
    for ext in (".fastq.gz", ".fq.gz", ".fastqsanger.gz", ".fastq", ".fq", ".fastqsanger", ".gz"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


@dataclass
class SampleAcc:
    unique: dict[str, dict[int, list[float]]] = field(default_factory=lambda: {"discovery": {}, "holdout": {}})
    eq: dict[str, dict[tuple, float]] = field(default_factory=lambda: {"discovery": {}, "holdout": {}})
    alleles: dict[str, np.ndarray] = field(default_factory=dict)
    residual: list[tuple[int, int, str]] = field(default_factory=list)
    total: int = 0
    mapped: int = 0
    decoy: int = 0
    # discovery-only counters: the only read counts any runtime branch or trap may look at
    disc_total: int = 0
    disc_mapped: int = 0
    disc_decoy: int = 0


def load_targets(path: Path) -> list[TargetRec]:
    out = []
    for tid, seq, attrs in read_fasta(path):
        out.append(TargetRec(tid, seq, attrs.get("group", tid), attrs.get("assembly", "primary"),
                             attrs.get("kind", "transcript"), attrs.get("kingdom", "host")))
    return out


def load_priors(path: Path | None) -> pa.Table | None:
    if path is None:
        return None
    df = pd.read_csv(path, sep="\t")
    for col, default in (("group_id", ""), ("sample", "*"), ("modality", "wgs_copy_number"), ("confidence", 1.0),
                         ("source_uri", str(path.name))):
        if col not in df:
            df[col] = default
    return pa.Table.from_pandas(df[EXTERNAL_PRIORS_SCHEMA.names], schema=EXTERNAL_PRIORS_SCHEMA, preserve_index=False)


class FocaloopRun:
    def __init__(self, bp: Blueprint, spec: RunSpec, state_dir: str | Path, out_table: str | Path,
                 out_telemetry: str | Path, state_archive: str | Path | None = None,
                 max_bytes: int | None = None, log=None) -> None:
        self.bp = bp
        self.spec = spec
        self.state_dir = Path(state_dir)
        self.out_table = Path(out_table)
        self.out_telemetry = Path(out_telemetry)
        self.state_archive = Path(state_archive) if state_archive else None
        self.max_bytes = max_bytes
        self.log = log or (lambda *_: None)
        self.telemetry: dict[str, Any] = {
            "tool": "focaloop", "command": "run", "blueprint_id": bp.id, "blueprint_version": bp.version,
            "modality": bp.modality, "container": bp.container, "status": "running", "trap_codes": [],
            "traps": [], "warnings": [], "iterations": [],
        }

    # -- traps ---------------------------------------------------------------
    def _trap(self, code: str, detail: str) -> None:
        trap = self.bp.trap(code)
        if code not in self.telemetry["trap_codes"]:
            self.telemetry["trap_codes"].append(code)
        self.telemetry["traps"].append({"code": code, "action": trap.action, "detail": detail})
        if trap.action == "abort":
            raise TrapAbort(code, detail)

    # -- main ----------------------------------------------------------------
    def execute(self) -> int:
        t0 = time.time()
        try:
            self._execute()
            self.telemetry["status"] = "ok" if not self.telemetry["trap_codes"] else "ok_with_warnings"
            code = EXIT_OK
        except TrapAbort as exc:
            self.telemetry["status"] = "trap"
            self.telemetry["abort_trap"] = exc.code
            if not self.out_table.exists():
                self.out_table.write_text("# focaloop aborted: " + exc.code + "\n", encoding="utf-8")
            code = EXIT_TRAP
        self.telemetry["wall_seconds"] = 0.0 if os.environ.get("FOCALOOP_FIXED_CLOCK") else round(time.time() - t0, 3)
        write_json(self.out_telemetry, self.telemetry)
        return code

    def _execute(self) -> None:
        bp, spec = self.bp, self.spec
        cfg = bp.stop_gate
        max_bytes = self.max_bytes if self.max_bytes is not None else cfg.max_bytes
        sample_cond = {s["sample"]: s["condition"] for s in spec.samples}
        contrast = spec.contrast
        if contrast is None:
            levels = sorted(set(sample_cond.values()))
            contrast = (levels[-1], levels[0]) if len(levels) >= 2 else None
        self.telemetry["contrast"] = list(contrast) if contrast else None

        problems = check_design(spec.samples, contrast if bp.modality == "expression" else None)
        if problems:
            self._trap("CONFOUNDED_DESIGN", "; ".join(problems))

        targets = load_targets(spec.targets)
        facts = {
            "n_targets": len(targets),
            "n_assemblies": len({t.assembly_id for t in targets if not t.is_decoy}),
            "n_samples": len(spec.samples),
            "has_decoy": any(t.is_decoy for t in targets),
            "modality": bp.modality,
        }
        settings = regime_settings(bp, facts)
        self.telemetry["regime_facts"] = facts
        self.telemetry["regime_settings"] = settings
        k = int(settings.get("k", bp.kernel_pool.tournament.k))
        index = KmerIndex(targets, k=k)
        target_ids = [t.target_id for t in targets if not t.is_decoy]
        lengths = {t.target_id: t.length for t in targets}
        group_of = {t.target_id: t.group_id for t in targets}

        probes = None
        if bp.modality == "variants" and spec.sites is not None:
            probes = AlleleProbes(load_sites(spec.sites), {t.target_id: t.seq for t in targets}, k=k)

        store = FocaloopStateStore(self.state_dir)
        for name in list(store.dir.glob("*.parquet")):
            name.unlink()

        readers: dict[str, FastqBlockReader] = {}
        for s in spec.samples:
            src = open_source(s["reads"], layout="SE")[0]
            readers[s["sample"]] = FastqBlockReader(src, s["run_id"], cfg.block_bytes, cfg.holdout_fraction)
        self.telemetry["stream_tiers"] = {k_: r.tier for k_, r in readers.items()}

        # ---- tournament on the first real reads (blocks are kept, not re-read) ----
        tcfg = bp.kernel_pool.tournament
        pending: dict[str, list[FastqBlock]] = {k_: [] for k_ in readers}
        real: list[str] = []
        run_parts = self.inference_partitions()
        while len(real) < tcfg.real_reads:
            progressed = False
            for name, rd in readers.items():
                if rd.exhausted or sum(r.consumed_bytes for r in readers.values()) >= max_bytes:
                    continue
                b = rd.next_block()
                if b is not None:
                    pending[name].append(b)
                    if b.partition in run_parts:  # holdout reads never pick the kernel
                        real.extend(b.reads)
                    progressed = True
            if not progressed:
                break
        kernels = build_kernels(bp.kernel_pool.kernels, index)
        result = run_tournament(kernels, index, real, n_real=tcfg.real_reads, n_per_regime=tcfg.synthetic_per_regime,
                                seed=tcfg.seed, tie_symmetry_min=bp.trap("TIE_SYMMETRY_FAIL").threshold or tcfg.tie_symmetry_min)
        self.telemetry["tournament"] = result.as_dict()
        skipped = [k_.name for k_ in kernels if not k_.available()]
        if skipped:
            self.telemetry["warnings"].append(f"kernels unavailable in this environment: {skipped}")
        if not result.board:
            raise RuntimeError("no available kernel in kernel_pool")
        if result.winner is None:
            worst = min((s.mate_tie_symmetry for s in result.board), default=float("nan"))
            self._trap("TIE_SYMMETRY_FAIL",
                       f"every kernel failed the mate-tie symmetry gate (min symmetry {worst:.3f} < {result.tie_symmetry_min})")
            winner_name = result.board[0].kernel
        else:
            winner_name = result.winner
        kernel = next(k_ for k_ in kernels if k_.name == winner_name)
        self.telemetry["kernel"] = winner_name

        # ---- streaming iterations ----
        accs = {s["sample"]: SampleAcc() for s in spec.samples}
        gate = StopGate(cfg.__class__(**{**cfg.__dict__, "max_bytes": max_bytes}))
        tel_rows: list[dict] = []
        iteration = 0
        map_trap = bp.trap("LOW_MAP_RATE")
        map_thr = map_trap.threshold if map_trap.threshold is not None else 0.3
        map_checked = False
        map_rate = float("nan")
        decision = None
        while True:
            iteration += 1
            t_it = time.time()
            for name, rd in readers.items():
                block = pending[name].pop(0) if pending[name] else None
                if block is None and not rd.exhausted and sum(r.consumed_bytes for r in readers.values()) < max_bytes:
                    block = rd.next_block()
                if block is not None:
                    self._ingest(accs[name], block, kernel, index, probes)
            cum = sum(r.consumed_bytes for r in readers.values())
            exhausted = all(r.exhausted for r in readers.values()) or cum >= max_bytes
            exhausted = exhausted and not any(pending.values())
            per_sample = self._per_sample_frame(accs, target_ids, group_of, index)
            decision = gate.evaluate(per_sample, sample_cond, iteration, cum, exhausted)
            tot = sum(a.total for a in accs.values())
            mapped = sum(a.mapped for a in accs.values())  # pooled counts: telemetry only
            # LOW_MAP_RATE reads discovery counters only (holdout is write-only during `run`)
            d_tot = sum(a.disc_total for a in accs.values())
            d_den = d_tot - sum(a.disc_decoy for a in accs.values())
            if d_den > 0:
                map_rate = sum(a.disc_mapped for a in accs.values()) / d_den
            m = decision.metrics
            tel_rows.append({
                "iteration": iteration, "modality": bp.modality, "stage": f"stream:{winner_name}",
                "cum_bytes": cum, "pct_archive": 100.0 * cum / max(1, sum(r.total_bytes for r in readers.values())),
                "total_reads": tot, "mapped_reads": mapped, "n_active_targets": m.n_active_targets,
                "median_abs_delta_estimate": _finite(m.median_abs_delta_estimate),
                "median_tech_var_fraction": _finite(m.median_tech_var_fraction),
                "wall_seconds": 0.0 if os.environ.get("FOCALOOP_FIXED_CLOCK") else time.time() - t_it, "stop_reason": decision.reason,
            })
            self.telemetry["iterations"].append({**tel_rows[-1], "map_rate": map_rate,
                                                 "median_posterior_precision": m.median_posterior_precision})
            if d_den > 0 and (not map_checked or decision.stop):
                map_checked = True
                if map_rate < map_thr:
                    self._persist(store, accs, tel_rows, targets, sample_cond, contrast, probes, index, readers)
                    self._trap("LOW_MAP_RATE", f"discovery map rate {map_rate:.3f} < {map_thr} after {d_tot} "
                                               f"discovery reads")
            if decision.stop:
                break
        for code in decision.traps:
            self._trap(code, f"stopped at {decision.metrics} after {tel_rows[-1]['cum_bytes']} bytes")
        self.telemetry["stop_reason"] = decision.reason
        self.telemetry["converged"] = decision.converged
        self.telemetry["consumed_bytes"] = {k_: r.consumed_bytes for k_, r in readers.items()}
        self.telemetry["total_consumed_bytes"] = sum(r.consumed_bytes for r in readers.values())
        self.telemetry["map_rate"] = map_rate
        self.telemetry["map_rate_partitions"] = ["discovery"]
        self._persist(store, accs, tel_rows, targets, sample_cond, contrast, probes, index, readers)
        self._write_table(store, sample_cond, contrast, lengths, probes)
        if self.state_archive:
            pack_state(self.state_dir, self.state_archive)
            self.telemetry["state_archive"] = str(self.state_archive)

    # -- per-block work --------------------------------------------------------
    def _ingest(self, acc: SampleAcc, block: FastqBlock, kernel, index: KmerIndex, probes) -> None:
        res = kernel.map_reads(block.reads)
        part = block.partition
        disc = part == "discovery"
        acc.total += len(block.reads)
        acc.disc_total += len(block.reads) if disc else 0
        for seq, cand, dec in zip(block.reads, res.candidates, res.decoy):
            if dec:
                acc.decoy += 1
                acc.disc_decoy += disc
                continue
            if cand is None:
                acc.residual.append((block.byte_start, block.byte_end, seq))
                continue
            acc.mapped += 1
            acc.disc_mapped += disc
            ts, lls, fs = cand
            if len(ts) == 1:
                u = acc.unique[part].setdefault(ts[0], [0.0, 0.0])
                u[0 if fs[0] else 1] += 1
            else:
                key = (tuple(index.ids[t] for t in ts), tuple(round(x, 3) for x in lls), fs)
                acc.eq[part][key] = acc.eq[part].get(key, 0.0) + 1
        if probes is not None:
            cnt = probes.count([r for r, d in zip(block.reads, res.decoy) if not d])
            acc.alleles[part] = acc.alleles.get(part, np.zeros_like(cnt)) + cnt

    def inference_partitions(self) -> tuple[str, ...]:
        """Partitions that runtime decisions (tournament, LOW_MAP_RATE, stop gate, primary table) may read.

        With a reusable holdout (``holdout_fraction > 0``) only ``discovery`` is used, so the holdout is
        write-only during `focaloop run` and conditionally independent of every adaptive choice until
        `focaloop query` (Dwork et al. 2015). Without a holdout every block is ``discovery`` anyway.
        """
        if self.bp.stop_gate.holdout_fraction > 0:
            return ("discovery",)
        return ("discovery", "holdout")

    def _per_sample_frame(self, accs: dict[str, SampleAcc], target_ids: list[str], group_of: dict[str, str],
                          index: KmerIndex) -> pd.DataFrame:
        rows = []
        parts = self.inference_partitions()
        self.telemetry["gate_partitions"] = list(parts)
        for s, acc in accs.items():
            eff = {t: 0.0 for t in target_ids}
            var = {t: 0.0 for t in target_ids}
            for part in parts:
                uniq = {index.ids[t]: c[0] + c[1] for t, c in acc.unique[part].items()}
                classes = [EqClass(k[0], k[1], k[2], v) for k, v in acc.eq[part].items()]
                r = solve_em(target_ids, uniq, classes, None, self.bp.conditioner.dirichlet_alpha,
                             min(50, self.bp.conditioner.max_iter), max(1e-5, self.bp.conditioner.tol))
                for t in target_ids:
                    eff[t] += r.effective[t]
                    var[t] += r.variance[t]
            g_eff: dict[str, float] = {}
            g_var: dict[str, float] = {}
            for t in target_ids:
                g = group_of[t]
                g_eff[g] = g_eff.get(g, 0.0) + eff[t]
                g_var[g] = g_var.get(g, 0.0) + var[t]
            for g in g_eff:
                rows.append({"sample": s, "target_id": g, "effective_count": g_eff[g], "assignment_variance": g_var[g]})
        return pd.DataFrame(rows)

    # -- persistence -----------------------------------------------------------
    def _persist(self, store, accs, tel_rows, targets, sample_cond, contrast, probes, index, readers) -> None:
        run_of = {s["sample"]: s["run_id"] for s in self.spec.samples}
        target_ids = [t.target_id for t in targets if not t.is_decoy]
        tmap = {t.target_id: t for t in targets}
        acc_rows: dict[str, list] = {n: [] for n in pa.schema(_ACC_FIELDS).names}
        eq_rows: dict[str, list] = {n: [] for n in _EQ_NAMES}
        eq_id = 0
        for s, acc in accs.items():
            for part in ("discovery", "holdout"):
                for t in target_ids:
                    u = acc.unique[part].get(index.pos[t], [0.0, 0.0])
                    rec = tmap[t]
                    for col, val in (("run_id", run_of[s]), ("sample", s), ("condition", sample_cond[s]),
                                     ("partition", part), ("target_id", t), ("assembly_id", rec.assembly_id),
                                     ("group_id", rec.group_id), ("length_bp", rec.length),
                                     ("unique_primary", u[0]), ("unique_alt_or_antisense", u[1]),
                                     ("effective_count", u[0] + u[1]), ("assignment_variance", 0.0)):
                        acc_rows[col].append(val)
                for (ts, lls, fs), cnt in sorted(acc.eq[part].items()):
                    eq_id += 1
                    for col, val in (("run_id", run_of[s]), ("partition", part), ("eq_class_id", eq_id),
                                     ("fragment_count", int(cnt)), ("candidate_targets", list(ts)),
                                     ("candidate_assemblies", [tmap[t].assembly_id for t in ts]),
                                     ("candidate_orientations", list(fs)), ("aln_loglik", list(lls)),
                                     ("posterior_weights", [1.0 / len(ts)] * len(ts))):
                        eq_rows[col].append(val)
        store.write("target_accumulators", pa.Table.from_pydict(acc_rows))
        store.write("ambiguous_eq", pa.Table.from_pydict(eq_rows, schema=_EQ_SCHEMA()))
        store.write("iteration_telemetry", pa.Table.from_pylist(tel_rows) if tel_rows else
                    pa.Table.from_pylist([], schema=_TEL_SCHEMA()))
        reg = HolobiontLocusRegistry()
        for t in targets:
            kind = t.kind if t.kind in ("transcript", "gene_body", "cdbg_color", "vg_snarl", "plasmid_contig") else "gene_body"
            kingdom = t.kingdom if t.kingdom in ("host", "bacteria", "fungi", "virus", "protist", "archaea") else "host"
            reg.register(t.target_id, kind, t.group_id, t.assembly_id, kingdom, t.length,
                         hgt_mobile_flag=t.kind == "plasmid_contig")
        reg.persist(store)
        if store.exists("residual_spans"):
            store.path("residual_spans").unlink()
        for s, acc in accs.items():
            if acc.residual:
                coalesce_residual_spans(run_of[s], readers[s].tier, acc.residual, store=store)
        priors = load_priors(self.spec.priors)
        if priors is not None:
            store.write("external_priors", priors)
        if probes is not None:
            rows = []
            for s, acc in accs.items():
                for part, arr in acc.alleles.items():
                    for site, (r, a) in zip(probes.sites, arr):
                        rows.append({"sample": s, "partition": part, "site_id": site.site_id, "target_id": site.target_id,
                                     "kind": site.kind, "ref_reads": int(r), "alt_reads": int(a)})
            store.write("allele_counts", pa.Table.from_pylist(rows, schema=ALLELE_COUNTS_SCHEMA), schema=ALLELE_COUNTS_SCHEMA)
        backward_condition(store, priors, self.bp.conditioner.dirichlet_alpha, self.bp.conditioner.max_iter,
                           self.bp.conditioner.tol)
        shutil.copyfile(self.spec.targets, store.dir / "targets.fa") if Path(self.spec.targets).suffix != ".gz" \
            else shutil.copyfile(self.spec.targets, store.dir / "targets.fa.gz")
        meta = {
            "blueprint_id": self.bp.id, "blueprint_version": self.bp.version, "modality": self.bp.modality,
            "samples": [{k: v for k, v in s.items() if k != "reads"} | {"reads": Path(str(s["reads"])).name}
                        for s in self.spec.samples],
            "contrast": list(contrast) if contrast else None,
            "stop_gate": self.bp.stop_gate.__dict__,
            "variants": self.bp.variants,
            "targets_file": "targets.fa" if Path(self.spec.targets).suffix != ".gz" else "targets.fa.gz",
            "consumed_bytes": {k_: r.consumed_bytes for k_, r in readers.items()},
        }
        write_json(store.dir / STATE_META, meta)

    def _write_table(self, store, sample_cond, contrast, lengths, probes) -> None:
        acc_all = store.read("target_accumulators").to_pandas()
        parts = self.inference_partitions()
        acc = acc_all[acc_all["partition"].isin(parts)]
        self.telemetry["table_partitions"] = list(parts)
        if self.bp.modality == "variants":
            eff = acc.groupby(["sample", "group_id"], as_index=False)["effective_count"].sum()
            glen = acc_all.groupby("group_id")["length_bp"].max().to_dict()
            alleles = store.read("allele_counts").to_pandas() if store.exists("allele_counts") else \
                pd.DataFrame(columns=ALLELE_COUNTS_SCHEMA.names)
            alleles = alleles[alleles["partition"].isin(parts)]
            tbl = variant_table(alleles, eff, glen, self.bp.variants)
        else:
            tbl = contrast_table(acc, sample_cond, contrast) if contrast else pd.DataFrame()
        tbl.to_csv(self.out_table, sep="\t", index=False, float_format="%.6g")
        self.telemetry["table_rows"] = int(len(tbl))


_ACC_FIELDS = [
    ("run_id", pa.string()), ("sample", pa.string()), ("condition", pa.string()), ("partition", pa.string()),
    ("target_id", pa.string()), ("assembly_id", pa.string()), ("group_id", pa.string()), ("length_bp", pa.int32()),
    ("unique_primary", pa.float64()), ("unique_alt_or_antisense", pa.float64()), ("effective_count", pa.float64()),
    ("assignment_variance", pa.float64()),
]
_EQ_NAMES = ["run_id", "partition", "eq_class_id", "fragment_count", "candidate_targets", "candidate_assemblies",
             "candidate_orientations", "aln_loglik", "posterior_weights"]


def _EQ_SCHEMA():
    from .state import AMBIGUOUS_EQ_SCHEMA
    return AMBIGUOUS_EQ_SCHEMA


def _TEL_SCHEMA():
    from .state import ITERATION_TELEMETRY_SCHEMA
    return ITERATION_TELEMETRY_SCHEMA


def _finite(x: float) -> float:
    return float(x) if np.isfinite(x) else -1.0


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.write_text(json.dumps(_clean(obj), indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
                 encoding="utf-8")
    return p
