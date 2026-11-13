"""Autonomous focused loop (Focaloop): variance-guided streaming with a mathematical stopping rule.

Each iteration:
  1. allocate the iteration byte budget across runs inversely to their effective
     mapped depth (shallow libraries catch up instead of blind uniform subsampling);
  2. stream those bytes (local_bytes or http_range), map with the tournament winner;
  3. Pass-2 backward conditioning (WGS P(DNA), strandedness, cohort EM priors);
  4. gene-level PyDESeq2 with placement variance -> lfc_se_total, 95% CI;
  5. stop when median(|delta log2FC|) < 0.05 over expressed genes (streamed_reads >= 20)
     and technical Poisson variance is small relative to biological dispersion,
     or when the max_bytes_pct cap is hit.
Then: borderline 31-mer zoom, post-zoom DE, warm (0 new byte) follow-up contrasts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa

from .catalog import Catalog
from .kernels import Kernel, RunAccumulator, map_pooled
from .manifest import Run, StudyManifest
from .solver import (assembly_assignment_summary, combine_zoom_gene_tables, de_with_uncertainty, fast_nb_mom_de,
                     gene_level_tables, infer_dna_presence, infer_strandedness, quantify_cohort,
                     refit_genes_with_depth_offsets)
from .state import (DOTHASH_SKETCH_SCHEMA, GENE_DE_SCHEMA, ITERATION_TELEMETRY_SCHEMA,
                    TARGET_COUNTS_SCHEMA, StateStore)
from .streamer import RunStream, open_run_stream
from .zoom import ZoomResult, select_borderline, zoom_pass


@dataclass
class LoopConfig:
    contrast: list[str]
    design: str = "~condition"
    source_mode: str = "local_bytes"
    cache_dir: str | Path | None = None
    focal_conditions: set[str] | None = None
    initial_fraction: float = 0.0015
    growth: float = 2.2
    bg_scale: float = 0.3
    max_iterations: int = 8
    min_iterations: int = 2
    max_bytes_pct: float = 10.0
    delta_lfc_stop: float = 0.05
    tech_var_stop: float = 0.25
    expressed_reads: float = 20.0
    wgs_bytes_per_iter: int = 4 * 1024 * 1024
    em_iters: int = 6
    length_offsets: bool = False
    zoom: bool = True
    zoom_extra_fraction: float = 0.01
    zoom_padj: tuple[float, float] = (0.005, 0.25)
    zoom_reads: tuple[float, float] = (5.0, 35.0)
    followup_contrasts: list[list[str]] = field(default_factory=list)
    fetch_workers: int = 4
    min_bytes_per_file: int = 262_144
    pool_samples: bool = True          # one kernel invocation (one index load) per layout per iteration
    sample_metadata: pd.DataFrame | dict[str, dict] | None = None
    euk_initial_fraction: float = 0.025
    fast_intermediate_de: bool = True
    fast_de_min_groups: int = 5_000
    use_ust_decoys: bool = False
    use_island_pc_router: bool = False
    use_dothash_dedup: bool = False


@dataclass
class LoopReport:
    iterations: int
    stop_reason: str
    bytes_global: int
    bytes_zoom: int
    pct_archive: float
    wall_seconds: float
    io_seconds: float
    map_seconds: float
    solve_seconds: float
    telemetry: list[dict]
    schedule: list[dict]
    pass1_assembly_counts: dict[str, float]
    pass2_assembly_counts: dict[str, float]
    assembly_prior: dict[str, float]
    strand_priors: dict[str, float]
    de_prezoom: pd.DataFrame
    de_postzoom: pd.DataFrame
    zoom: ZoomResult | None
    followups: dict[str, dict]
    paths: dict[str, str]


class Focaloop:
    def __init__(self, catalog: Catalog, manifest: StudyManifest, kernel: Kernel, config: LoopConfig,
                 state_dir: str | Path, log=print) -> None:
        self.catalog = catalog
        self.manifest = manifest
        self.kernel = kernel
        self.cfg = config
        self.state = StateStore(state_dir)
        self.log = log
        dset = catalog.decoy_ids
        self.accs: dict[str, RunAccumulator] = {r.run_id: RunAccumulator(run=r, decoy_ids=dset) for r in manifest.runs}
        self.streams: dict[str, RunStream] = {r.run_id: open_run_stream(r, config.source_mode, config.cache_dir) for r in manifest.runs}
        self.symbols = {g: ts[0].symbol for g, ts in catalog.groups.items()}
        self.gene_ids = catalog.primary_gene_of_group()
        self.llm_calls = 0  # by construction: nothing in this loop talks to a model
        self.pooled_calls = 0
        self.low_map_rate_runs: set[str] = set()
        self._last_alloc_cap_clipped = False
        self._dothash_hvs: dict[str, np.ndarray] = {}
        self._dothash_seen: dict[str, set[int]] = {}
        self._dothash_reads: dict[str, int] = {}
        self._dothash_metrics: dict[str, dict[str, float]] = {}
        if self.cfg.use_dothash_dedup:
            from .dothash import DOTHASH_DIM
            for r in self.manifest.rna_runs:
                self._dothash_hvs[r.run_id] = np.zeros(DOTHASH_DIM, dtype=np.int16)
                self._dothash_seen[r.run_id] = set()
                self._dothash_reads[r.run_id] = 0
                self._dothash_metrics[r.run_id] = {
                    "max_within_cond_cosine": 0.0,
                    "max_within_cond_containment": 0.0,
                    "dothash_weight_scale": 1.0,
                }

    # ---------------- byte allocation ----------------
    def _focal(self, r: Run) -> bool:
        return self.cfg.focal_conditions is None or r.condition in self.cfg.focal_conditions

    def allocate(self, iteration: int, budget_fraction: float) -> dict[str, int]:
        """Bytes per run for this iteration (variance-guided: inverse to effective mapped depth)."""
        alloc: dict[str, int] = {}
        focal = [r for r in self.manifest.rna_runs if self._focal(r)]
        bg = [r for r in self.manifest.rna_runs if not self._focal(r)]
        for r in self.manifest.wgs_runs:
            alloc[r.run_id] = self.cfg.wgs_bytes_per_iter * len(r.files)
        if iteration == 1 or not focal:
            for r in focal:
                alloc[r.run_id] = max(self.cfg.min_bytes_per_file * len(r.files), int(r.total_bytes * budget_fraction))
        else:
            budget = sum(int(r.total_bytes * budget_fraction) for r in focal)
            depth = {r.run_id: float(self.accs[r.run_id].n_mapped) for r in focal}
            bpr = {}
            for r in focal:
                a = self.accs[r.run_id]
                bpr[r.run_id] = (a.bytes_consumed / a.n_mapped) if a.n_mapped > 0 else (r.total_bytes / max(1, r.read_count))
            mean_depth = max(1.0, float(np.mean(list(depth.values()))))
            est_new_reads = budget / max(1.0, float(np.mean(list(bpr.values()))))
            target = (sum(depth.values()) + est_new_reads) / len(focal)
            want = {k: bpr[k] * max(target - depth[k], 0.25 * target) for k in depth}
            if self.cfg.use_dothash_dedup and len(focal) >= 2:
                self._apply_dothash_dedup_weights(focal, depth, want)
            # clip to [0.25x, 4x] of the uniform share so no library is starved or hogs the budget
            share = budget / len(focal)
            tot_w = sum(want.values()) or 1.0
            for r in focal:
                raw = budget * want[r.run_id] / tot_w
                alloc[r.run_id] = int(min(4.0 * share, max(0.25 * share, raw)))
        for r in bg:
            alloc[r.run_id] = max(self.cfg.min_bytes_per_file * len(r.files), int(r.total_bytes * budget_fraction * self.cfg.bg_scale))
        for r in self.manifest.runs:
            rs = self.streams[r.run_id]
            alloc[r.run_id] = int(min(alloc[r.run_id], max(0, rs.total_bytes - rs.consumed_bytes)))
        # never stream past the max_bytes_pct cap: scale this iteration down to the remaining allowance,
        # and skip near-zero tail slivers after min_iterations so we do not run a full wave + DE solve for <0.1% bytes.
        self._last_alloc_cap_clipped = False
        cum = sum(a.bytes_consumed for a in self.accs.values())
        remaining = self.cfg.max_bytes_pct / 100.0 * self.manifest.total_archive_bytes - cum
        planned = sum(alloc.values())
        min_sliver = max(int(0.001 * self.manifest.total_archive_bytes), int(0.10 * planned))
        if remaining <= 0 or (iteration > self.cfg.min_iterations and remaining < min_sliver):
            alloc = {k: 0 for k in alloc}
            self._last_alloc_cap_clipped = True
        elif planned > remaining > 0:
            f = remaining / planned
            alloc = {k: int(v * f) for k, v in alloc.items()}
            self._last_alloc_cap_clipped = True
        return alloc

    def _apply_dothash_dedup_weights(
        self,
        focal: list[Run],
        depth: dict[str, float],
        want: dict[str, float],
    ) -> None:
        """Down-weight redundant within-condition biological replicates via 4 KB DotHash similarity."""
        from .dothash import cosine_similarity, estimate_containment

        by_cond: dict[str, list[Run]] = {}
        for r in focal:
            by_cond.setdefault(r.condition, []).append(r)
        for _cond, runs in by_cond.items():
            if len(runs) < 2:
                continue
            # Rank replicates so deeper (or earlier-indexed on ties) peers stay unweighted representatives
            ordered = sorted(runs, key=lambda r: (-depth.get(r.run_id, 0.0), r.run_id))
            rep_hvs: list[np.ndarray] = []
            for r in ordered:
                hv = self._dothash_hvs.get(r.run_id)
                if hv is None or int(np.count_nonzero(hv)) == 0:
                    continue
                max_cos = 0.0
                max_cont = 0.0
                for peer_hv in rep_hvs:
                    c_sim = cosine_similarity(hv, peer_hv, clip=True)
                    cont = estimate_containment(hv, peer_hv, clip=True)
                    if c_sim > max_cos:
                        max_cos = c_sim
                    if cont > max_cont:
                        max_cont = cont
                # Redundancy above mild background triggers progressive down-weighting; primary rep has scale 1.0
                redundancy = 0.5 * (max_cos + max_cont)
                if rep_hvs and redundancy > 0.30:
                    scale = max(0.35, 1.0 - 0.65 * ((redundancy - 0.30) / 0.70))
                else:
                    scale = 1.0
                want[r.run_id] *= scale
                self._dothash_metrics[r.run_id] = {
                    "max_within_cond_cosine": round(float(max_cos), 6),
                    "max_within_cond_containment": round(float(max_cont), 6),
                    "dothash_weight_scale": round(float(scale), 6),
                }
                rep_hvs.append(hv)

    # ---------------- streaming + mapping ----------------
    def stream_and_map(self, alloc: dict[str, int]) -> tuple[float, float]:
        t0 = time.time()

        def _fetch(run_id: str):
            rs = self.streams[run_id]
            before = rs.consumed_bytes
            r1, r2 = rs.advance_bytes(alloc[run_id])
            return run_id, r1, r2, rs.consumed_bytes - before

        with ThreadPoolExecutor(max_workers=self.cfg.fetch_workers) as ex:
            fetched = list(ex.map(_fetch, [k for k, v in alloc.items() if v > 0]))
        io_s = time.time() - t0
        t1 = time.time()
        for run_id, _r1, _r2, delta in fetched:
            self.accs[run_id].bytes_consumed += delta
        todo = [(run_id, r1, r2) for run_id, r1, r2, _d in fetched if r1]
        if self.cfg.use_dothash_dedup and todo:
            from .dothash import update_sketch_from_reads
            for run_id, r1, r2 in todo:
                if run_id not in self._dothash_hvs:
                    continue
                n_take = min(len(r1), 25_000)
                seqs = [s for s, _q in r1[:n_take]]
                if r2:
                    seqs.extend(s for s, _q in r2[: min(len(r2), 25_000)])
                update_sketch_from_reads(
                    self._dothash_hvs[run_id],
                    seqs,
                    self._dothash_seen[run_id],
                )
                self._dothash_reads[run_id] += n_take
        if self.cfg.pool_samples and (getattr(self.kernel, "supports_pooling", False) or hasattr(self.kernel, "clone_with_threads")):
            for paired in (True, False):
                grp = [(rid, r1, r2, self.accs[rid].calibrator) for rid, r1, r2 in todo if (r2 is not None) == paired]
                if grp:
                    self.pooled_calls += 1
                    for rid, res in map_pooled(self.kernel, grp).items():
                        self.accs[rid].add(res)
        else:
            for run_id, r1, r2 in todo:
                acc = self.accs[run_id]
                acc.add(self.kernel.map_batch(r1, r2, acc.calibrator))
        # LOW_MAP_RATE: sources that partition reads (sra_odp) keep serving discovery blocks only to flagged runs
        for run_id, _r1, _r2 in todo:
            st = self.streams[run_id]
            if hasattr(st, "flag_low_map_rate") and st.flag_low_map_rate(self.accs[run_id].mapped_fraction):
                self.low_map_rate_runs.add(run_id)
        return io_s, time.time() - t1

    # ---------------- solving ----------------
    def _enrich_meta(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Attach sample covariates from ``Run.extra`` (and ``Run.extra['covariates']``) plus ``cfg.sample_metadata``."""
        out = meta.copy()
        per_sample: dict[str, dict] = {}
        for r in self.manifest.rna_runs:
            if not r.extra:
                continue
            d = per_sample.setdefault(r.sample, {})
            for k, v in r.extra.items():
                if k == "covariates" and isinstance(v, dict):
                    d.update(v)
                elif not isinstance(v, (dict, list, tuple, set)):
                    d[k] = v
        if self.cfg.sample_metadata is not None:
            sm = (
                self.cfg.sample_metadata.to_dict(orient="index")
                if isinstance(self.cfg.sample_metadata, pd.DataFrame)
                else self.cfg.sample_metadata
            )
            for s, row in sm.items():
                if isinstance(row, dict):
                    per_sample.setdefault(str(s), {}).update(row)
        if not per_sample:
            return out
        extra_df = pd.DataFrame.from_dict(per_sample, orient="index")
        for col in extra_df.columns:
            if col not in out.columns:
                out[col] = extra_df[col].reindex(out.index)
        return out

    def solve(self, iteration: int, two_way: bool = True, fast_de: bool = False) -> tuple[pd.DataFrame, pa.Table, dict]:
        wgs = [a for a in self.accs.values() if a.run.is_wgs]
        if wgs and any(a.n_mapped > 0 for a in wgs):
            dna_prob, dna_counts, asm_prior = infer_dna_presence(wgs, self.catalog)
        else:
            dna_prob, dna_counts = {}, {}
            asm_prior = {a: 1.0 / len(self.catalog.assemblies) for a in self.catalog.assemblies}
        strand = {a.run.run_id: (0.5 if a.run.is_wgs else infer_strandedness(a, self.catalog)) for a in self.accs.values()}
        table, post = quantify_cohort(
            self.accs, self.catalog, dna_prob, dna_counts, strand, two_way=two_way,
            em_iters=self.cfg.em_iters, include_posteriors=False, include_decoys=False,
        )
        gl = gene_level_tables(table, self.catalog, length_offsets=self.cfg.length_offsets)
        gl["meta"] = self._enrich_meta(gl["meta"])
        counts = gl.get("counts_length_scaled", gl["counts"]) if self.cfg.length_offsets else gl["counts"]
        use_fast = fast_de and len(counts.columns) >= self.cfg.fast_de_min_groups
        if use_fast:
            self._last_de_disp_sf = (None, None)
            de = fast_nb_mom_de(counts, gl["avar"], gl["meta"], self.cfg.contrast, self.cfg.design, gl["dna_prob"],
                                self.symbols, self.gene_ids, iteration, self.cfg.expressed_reads)
        else:
            de = self._de_from_tables(gl, iteration)
        extra = {
            "dna_prob": dna_prob, "assembly_prior": asm_prior, "strand": strand,
            "gene_tables": gl, "posteriors": post, "used_fast_de": use_fast,
        }
        return de, table, extra

    # ---------------- main ----------------
    def run(self) -> LoopReport:
        cfg = self.cfg
        t_start = time.time()
        total_archive = self.manifest.total_archive_bytes
        telemetry: list[dict] = []
        schedule: list[dict] = []
        prev_de: pd.DataFrame | None = None
        self._last_zoom_table: pa.Table | None = None
        self._last_zoom_ctx: dict | None = None
        io_tot = map_tot = solve_tot = 0.0
        stop_reason = "max_iterations"
        if self.catalog.mode == "eukaryotic" and len(self.catalog.gene_targets) >= 25_000 and cfg.initial_fraction == 0.0015:
            frac = min(cfg.euk_initial_fraction, cfg.max_bytes_pct / 100.0)
        else:
            frac = cfg.initial_fraction
        de = table = extra = None
        it = 0
        for it in range(1, cfg.max_iterations + 1):
            t_it = time.time()
            alloc = self.allocate(it, frac)
            if sum(alloc.values()) <= 0:
                stop_reason = "max_bytes_pct_cap"
                it -= 1
                break
            schedule.append({"iteration": it, "alloc": dict(alloc)})
            io_s, map_s = self.stream_and_map(alloc)
            io_tot += io_s
            map_tot += map_s
            t_s = time.time()
            de, table, extra = self.solve(it, fast_de=(cfg.fast_intermediate_de and it < cfg.max_iterations))
            solve_tot += time.time() - t_s
            cum = sum(a.bytes_consumed for a in self.accs.values())
            pct = 100.0 * cum / max(1, total_archive)
            expressed = de[(de["streamed_reads"] >= cfg.expressed_reads) & de["log2fc"].notna()]
            med_tech = float(expressed["tech_var_fraction"].median()) if len(expressed) else 1.0
            if prev_de is not None:
                j = expressed.set_index("group_id")[["log2fc"]].merge(prev_de.set_index("group_id")[["log2fc", "streamed_reads"]], left_index=True, right_index=True, how="inner", suffixes=("", "_prev"))
                j = j[j["streamed_reads"] >= cfg.expressed_reads]
                if len(j):
                    abs_d = (j["log2fc"] - j["log2fc_prev"]).abs()
                    med_delta = float(abs_d.median())
                    p90_delta = float(abs_d.quantile(0.90))
                    p95_delta = float(abs_d.quantile(0.95))
                else:
                    med_delta = p90_delta = p95_delta = float("nan")
            else:
                med_delta = p90_delta = p95_delta = float("nan")
            n_deg = int((de["padj"] < 0.05).sum())
            total_reads = sum(a.n_reads for a in self.accs.values() if not a.run.is_wgs)
            mapped = sum(a.n_mapped for a in self.accs.values() if not a.run.is_wgs)
            reason = ""
            converged = (it >= cfg.min_iterations and not np.isnan(med_delta) and med_delta < cfg.delta_lfc_stop and med_tech <= cfg.tech_var_stop)
            exhausted = all(self.streams[r.run_id].exhausted for r in self.manifest.rna_runs)
            if converged:
                reason = "converged_tech_var_below_bio_var"
            elif pct >= cfg.max_bytes_pct - 1e-6 or (self._last_alloc_cap_clipped and it >= cfg.min_iterations and pct >= cfg.max_bytes_pct * 0.98):
                reason = "max_bytes_pct_cap"
            elif exhausted:
                reason = "archive_exhausted"
            if reason and extra.get("used_fast_de"):
                t_exact = time.time()
                de = self._de_from_tables(extra["gene_tables"], it)
                solve_tot += time.time() - t_exact
                extra["used_fast_de"] = False
                expressed = de[(de["streamed_reads"] >= cfg.expressed_reads) & de["log2fc"].notna()]
                med_tech = float(expressed["tech_var_fraction"].median()) if len(expressed) else 1.0
                n_deg = int((de["padj"] < 0.05).sum())
                if reason == "converged_tech_var_below_bio_var" and med_tech > cfg.tech_var_stop and it < cfg.max_iterations:
                    reason = ""
            telemetry.append({
                "iteration": it, "stage": "global", "cum_bytes": int(cum), "pct_archive": pct, "total_reads": int(total_reads),
                "mapped_reads": int(mapped), "n_expressed_genes": int(len(expressed)),
                "median_abs_delta_lfc": med_delta, "p90_abs_delta_lfc": p90_delta, "p95_abs_delta_lfc": p95_delta,
                "median_tech_var_fraction": med_tech, "n_deg_fdr05": n_deg, "wall_seconds": time.time() - t_it, "stop_reason": reason,
            })
            self.log(f"  [iter {it}] {cum / 1e6:9.1f} MB ({pct:5.2f}%) reads={total_reads:,} mapped={mapped:,} expressed={len(expressed):,} "
                     f"DEG={n_deg} median|dLFC|={med_delta:.4f} p90|dLFC|={p90_delta:.4f} median_tech_var={med_tech:.3f} [{time.time() - t_it:.1f}s] {reason}")
            prev_de = de
            if reason:
                stop_reason = reason
                break
            frac *= cfg.growth
        assert de is not None and table is not None and extra is not None
        if extra.get("used_fast_de"):
            t_exact = time.time()
            de = self._de_from_tables(extra["gene_tables"], it)
            solve_tot += time.time() - t_exact
            extra["used_fast_de"] = False
        bytes_global = sum(a.bytes_consumed for a in self.accs.values())
        de_pre = de.copy()
        # Pass-1 snapshot for leak accounting
        p1_table, _ = quantify_cohort(
            self.accs, self.catalog, extra["dna_prob"], {}, extra["strand"],
            two_way=False, em_iters=1, include_posteriors=False, include_decoys=False,
        )
        p1_asm = assembly_assignment_summary(p1_table)
        p2_asm = assembly_assignment_summary(table)
        # Zoom
        zoom_res: ZoomResult | None = None
        de_post = de_pre
        if cfg.zoom:
            groups = select_borderline(de_pre, cfg.zoom_padj[0], cfg.zoom_padj[1], cfg.zoom_reads[0], cfg.zoom_reads[1])
            if groups:
                t_z = time.time()
                rna_streams = {k: v for k, v in self.streams.items() if not v.run.is_wgs}
                zoom_res = zoom_pass(
                    self.kernel, self.catalog, self.accs, rna_streams, groups,
                    cfg.zoom_extra_fraction, log=self.log, fetch_workers=cfg.fetch_workers, pool_samples=cfg.pool_samples,
                )
                de_post, table = self._post_zoom_de(it, groups, zoom_res, extra, de_global=de_pre, global_table=table)
                telemetry.append({
                    "iteration": it, "stage": "zoom", "cum_bytes": int(bytes_global + zoom_res.bytes_scanned),
                    "pct_archive": 100.0 * (bytes_global + zoom_res.bytes_scanned) / max(1, total_archive),
                    "total_reads": int(zoom_res.reads_scanned), "mapped_reads": int(zoom_res.reads_mapped), "n_expressed_genes": int(len(groups)),
                    "median_abs_delta_lfc": float("nan"), "p90_abs_delta_lfc": float("nan"), "p95_abs_delta_lfc": float("nan"),
                    "median_tech_var_fraction": float("nan"), "n_deg_fdr05": int((de_post["padj"] < 0.05).sum()),
                    "wall_seconds": time.time() - t_z, "stop_reason": "zoom_complete",
                })
        bytes_zoom = zoom_res.bytes_scanned if zoom_res else 0
        # Warm follow-ups (0 new bytes): fitted on pure global counts, then refit zoomed genes with depth_scale offsets
        followups: dict[str, dict] = {}
        gl = extra["gene_tables"] if "gene_tables" in extra else gene_level_tables(table, self.catalog, length_offsets=cfg.length_offsets)
        gl["meta"] = self._enrich_meta(gl["meta"])
        counts = gl.get("counts_length_scaled", gl["counts"]) if cfg.length_offsets else gl["counts"]
        zctx = self._last_zoom_ctx
        for fc in cfg.followup_contrasts:
            t_f = time.time()
            try:
                fde, f_disp, f_sf = de_with_uncertainty(
                    counts, gl["avar"], gl["meta"], fc, cfg.design, gl["dna_prob"],
                    self.symbols, self.gene_ids, it, cfg.expressed_reads, return_fit=True,
                )
                if zctx and zctx.get("groups"):
                    fde = refit_genes_with_depth_offsets(
                        fde, zctx["counts_global"], gl["meta"], fc, zctx["groups"],
                        zctx["combined"], zctx["depth_scale"], zctx["avar_comb"], design=cfg.design,
                        disp=f_disp, sf=f_sf,
                    )
                name = f"{fc[1]}_vs_{fc[2]}"
                p = self.state.write(f"gene_de_followup_{name}", pa.Table.from_pandas(fde, preserve_index=False), GENE_DE_SCHEMA)
                followups[name] = {"wall_seconds": time.time() - t_f, "new_fastq_bytes": 0, "n_deg_fdr05": int((fde["padj"] < 0.05).sum()), "parquet": str(p)}
            except Exception as exc:
                followups[f"{fc[1]}_vs_{fc[2]}"] = {"error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.time() - t_f, "new_fastq_bytes": 0}
        for st in self.streams.values():
            if hasattr(st, "close"):
                try:
                    st.close()
                except Exception:
                    pass
        paths = {
            "target_counts": str(self.state.write("target_counts", table, TARGET_COUNTS_SCHEMA)),
            "gene_de_prezoom": str(self.state.write("gene_de_prezoom", pa.Table.from_pandas(de_pre, preserve_index=False), GENE_DE_SCHEMA)),
            "gene_de_postzoom": str(self.state.write("gene_de_postzoom", pa.Table.from_pandas(de_post, preserve_index=False), GENE_DE_SCHEMA)),
            "telemetry": str(self.state.write("telemetry", pa.Table.from_pylist(telemetry), ITERATION_TELEMETRY_SCHEMA)),
        }
        if self._last_zoom_table is not None:
            paths["target_counts_zoom"] = str(self.state.write("target_counts_zoom", self._last_zoom_table, TARGET_COUNTS_SCHEMA))
        (self.state.dir / "schedule.json").write_text(json.dumps(schedule))
        paths["schedule"] = str(self.state.dir / "schedule.json")
        bp_rows = [{"run_id": rid, "condition": a.run.condition, "target_a": k[0], "target_b": k[1], "pairs": int(c)}
                   for rid, a in self.accs.items() for k, c in a.bridge_pairs.items()]
        if bp_rows:
            paths["bridge_pairs"] = str(self.state.write("bridge_pairs", pa.Table.from_pylist(bp_rows)))
        if cfg.use_dothash_dedup and self._dothash_hvs:
            dh_rows = []
            for r in self.manifest.rna_runs:
                hv = self._dothash_hvs.get(r.run_id)
                if hv is None:
                    continue
                m = self._dothash_metrics.get(r.run_id, {})
                dh_rows.append({
                    "run_id": r.run_id,
                    "sample": r.sample,
                    "condition": r.condition,
                    "dim": int(hv.size),
                    "n_reads_sketched": int(self._dothash_reads.get(r.run_id, 0)),
                    "n_distinct_syncmers_seen": int(len(self._dothash_seen.get(r.run_id, ()))),
                    "max_within_cond_cosine": float(m.get("max_within_cond_cosine", 0.0)),
                    "max_within_cond_containment": float(m.get("max_within_cond_containment", 0.0)),
                    "dothash_weight_scale": float(m.get("dothash_weight_scale", 1.0)),
                    "sketch_int16": hv.astype("<i2", copy=False).tobytes(),
                })
            if dh_rows:
                paths["dothash_sketches"] = str(
                    self.state.write("dothash_sketches", pa.Table.from_pylist(dh_rows), DOTHASH_SKETCH_SCHEMA)
                )
        wall = time.time() - t_start
        return LoopReport(it, stop_reason, bytes_global, bytes_zoom, 100.0 * (bytes_global + bytes_zoom) / max(1, total_archive),
                          wall, io_tot, map_tot, solve_tot, telemetry, schedule, p1_asm, p2_asm, extra["assembly_prior"], extra["strand"],
                          de_pre, de_post, zoom_res, followups, paths)

    def _sample_depth_scale(self, samples: list[str], zoom_res: ZoomResult) -> pd.Series:
        """Per-sample global/(global+zoom) byte fraction, aggregating multi-run samples accurately."""
        glob_b: dict[str, float] = {}
        tot_b: dict[str, float] = {}
        for rid, sc in zoom_res.per_run_scale.items():
            if rid not in self.accs:
                continue
            acc = self.accs[rid]
            s = acc.run.sample
            gb = float(acc.bytes_consumed)
            if gb > 0 and sc > 0:
                tb = gb / float(sc)
            else:
                z_acc = getattr(acc, "zoom", None)
                tb = gb + float(z_acc.bytes_consumed if z_acc is not None else 0.0)
            glob_b[s] = glob_b.get(s, 0.0) + gb
            tot_b[s] = tot_b.get(s, 0.0) + tb
        out: dict[str, float] = {}
        for s in samples:
            if s in tot_b and tot_b[s] > 0:
                out[s] = glob_b[s] / tot_b[s]
            else:
                out[s] = 1.0
        return pd.Series(out, index=samples, dtype=float)

    def _post_zoom_de(
        self,
        iteration: int,
        groups: list[str],
        zoom_res: ZoomResult,
        extra: dict,
        de_global: pd.DataFrame | None = None,
        global_table: pa.Table | None = None,
    ) -> tuple[pd.DataFrame, pa.Table]:
        """Refit borderline genes on (global + zoom) counts with per-sample depth offsets.

        Pure global uniform-depth counts in ``table`` (written to ``target_counts.parquet``) are
        never mutated; zoom-only counts are stored separately in ``self._last_zoom_table`` and
        written to ``target_counts_zoom.parquet``.
        """
        zoom_accs = {k: getattr(a, "zoom", None) for k, a in self.accs.items()}
        zoom_accs = {k: v for k, v in zoom_accs.items() if v is not None}
        wgs = [a for a in self.accs.values() if a.run.is_wgs]
        if wgs and any(a.n_mapped > 0 for a in wgs):
            dna_prob, dna_counts, _ = infer_dna_presence(wgs, self.catalog)
        else:
            dna_prob, dna_counts = {}, {}
        if global_table is not None and de_global is not None and "gene_tables" in extra:
            table = global_table
            gl = extra["gene_tables"]
            gl["meta"] = self._enrich_meta(gl["meta"])
        else:
            table, _ = quantify_cohort(
                self.accs, self.catalog, dna_prob, dna_counts, extra["strand"],
                two_way=True, em_iters=self.cfg.em_iters, include_posteriors=False, include_decoys=False,
            )
            gl = gene_level_tables(table, self.catalog, length_offsets=self.cfg.length_offsets)
            gl["meta"] = self._enrich_meta(gl["meta"])
            de_global = self._de_from_tables(gl, iteration)
        if not zoom_accs:
            return de_global, table
        ztable, _ = quantify_cohort(
            zoom_accs, self.catalog, dna_prob, dna_counts, extra["strand"],
            two_way=True, em_iters=self.cfg.em_iters, include_posteriors=False, include_decoys=False,
        )
        samples = list(gl["counts"].index)
        depth_scale = self._sample_depth_scale(samples, zoom_res)
        counts_g, combined, avar_comb = combine_zoom_gene_tables(
            gl, table, ztable, groups, depth_scale, self.catalog, length_offsets=self.cfg.length_offsets,
        )
        gset = [g for g in groups if g in counts_g.columns]
        p_disp, p_sf = getattr(self, "_last_de_disp_sf", (None, None))
        de_post = refit_genes_with_depth_offsets(
            de_global, counts_g, gl["meta"], self.cfg.contrast, gset, combined, depth_scale, avar_comb,
            design=self.cfg.design, disp=p_disp, sf=p_sf,
        )
        self._last_zoom_table = ztable
        self._last_zoom_ctx = {
            "groups": gset,
            "counts_global": counts_g,
            "combined": combined,
            "depth_scale": depth_scale,
            "avar_comb": avar_comb,
        }
        return de_post, table

    def _de_from_tables(self, gl: dict, iteration: int) -> pd.DataFrame:
        counts = gl.get("counts_length_scaled", gl["counts"]) if self.cfg.length_offsets else gl["counts"]
        df, disp, sf = de_with_uncertainty(
            counts, gl["avar"], gl["meta"], self.cfg.contrast, self.cfg.design, gl["dna_prob"],
            self.symbols, self.gene_ids, iteration, self.cfg.expressed_reads, return_fit=True,
        )
        self._last_de_disp_sf = (disp, sf)
        return df


def replay_http_range(manifest: StudyManifest, schedule: list[dict], cache_dir: str | Path, workers: int = 6, log=print) -> dict:
    """Fetch exactly the byte ranges of a recorded local schedule from ENA via HTTP 206 and time it.

    Returns wall-clock, bytes, request counts.  Used to report WAN cold-start cost for the same
    byte schedule the local run consumed (mapping/solving cost is identical and added by the caller).
    """
    streams = {r.run_id: open_run_stream(r, "http_range", cache_dir) for r in manifest.runs}
    t0 = time.time()
    per_iter = []
    for step in schedule:
        t_i = time.time()
        alloc = step["alloc"]

        def _f(run_id: str) -> int:
            rs = streams[run_id]
            before = rs.consumed_bytes
            rs.advance_bytes(alloc[run_id])
            return rs.consumed_bytes - before

        with ThreadPoolExecutor(max_workers=workers) as ex:
            got = sum(ex.map(_f, [k for k, v in alloc.items() if v > 0]))
        per_iter.append({"iteration": step["iteration"], "bytes": int(got), "seconds": time.time() - t_i})
        log(f"  [http206 iter {step['iteration']}] {got / 1e6:.1f} MB in {time.time() - t_i:.1f}s")
    net = sum(getattr(s.source, "network_bytes", 0) for rs in streams.values() for s in rs.streams)
    reqs = sum(getattr(s.source, "requests_made", 0) for rs in streams.values() for s in rs.streams)
    return {"wall_seconds": time.time() - t0, "bytes": int(sum(p["bytes"] for p in per_iter)), "network_bytes": int(net),
            "requests": int(reqs), "per_iteration": per_iter}
