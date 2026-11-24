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
                     refit_genes_with_depth_offsets, transcript_level_table, validate_additive_design)
from .state import (DOTHASH_SKETCH_SCHEMA, GENE_DE_SCHEMA, ITERATION_TELEMETRY_SCHEMA,
                    TARGET_COUNTS_SCHEMA, TRANSCRIPT_ABUNDANCE_SCHEMA, StateStore)
from .streamer import RunStream, open_run_stream
from .zoom import ZoomResult, select_borderline, select_zoom_targets, zoom_pass


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
    max_iterations: int = 12
    min_iterations: int = 2
    max_bytes_pct: float = 100.0
    delta_lfc_stop: float = 0.05
    tech_var_stop: float = 0.25
    expressed_reads: float = 20.0
    wgs_bytes_per_iter: int = 4 * 1024 * 1024
    em_iters: int = 6
    length_offsets: bool = False
    zoom: bool = True
    zoom_extra_fraction: float = 0.01
    zoom_padj: tuple[float, float] = (0.002, 0.25)
    zoom_reads: tuple[float, float] = (5.0, 35.0)
    followup_contrasts: list[list[str]] = field(default_factory=list)
    fetch_workers: int = 24
    min_bytes_per_file: int = 262_144
    pool_samples: bool = True          # one kernel invocation (one index load) per layout per iteration
    sample_metadata: pd.DataFrame | dict[str, dict] | None = None
    euk_initial_fraction: float = 0.025
    fast_intermediate_de: bool = True
    fast_de_min_groups: int = 5_000
    use_ust_decoys: bool = False
    use_island_pc_router: bool = False
    use_dothash_dedup: bool = False
    min_mapped_per_sample: int | None = None
    abundant_sieve: bool = True
    abundant_sieve_force: bool = False
    sieve_mode: str = "auto"
    sieve_low_reads_hi: float = 50.0
    sieve_max_rem_fraction: float = 0.30


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
    malformed_skipped: dict[str, int] = field(default_factory=dict)
    skipped_unpaired_mates: dict[str, int] = field(default_factory=dict)

    @property
    def zoom_ran(self) -> bool:
        return "target_counts_zoom" in self.paths


class Focaloop:
    def __init__(self, catalog: Catalog, manifest: StudyManifest, kernel: Kernel, config: LoopConfig,
                 state_dir: str | Path, log=print) -> None:
        seen_rids: set[str] = set()
        for r in manifest.runs:
            if r.run_id in seen_rids:
                raise ValueError(f"Duplicate run_id {r.run_id!r} in StudyManifest")
            seen_rids.add(r.run_id)
        design_cols = set(validate_additive_design(config.design))
        if len(config.contrast) != 3:
            raise ValueError(f"Invalid primary contrast {config.contrast!r}: expected [factor, test, reference]")
        for fc in config.followup_contrasts:
            if len(fc) != 3:
                raise ValueError(f"Invalid follow-up contrast {fc!r}: expected [factor, test, reference]")
            if fc[0] not in design_cols:
                raise ValueError(
                    f"Follow-up contrast factor {fc[0]!r} is not included in design {config.design!r} "
                    f"(design terms: {sorted(design_cols)})"
                )
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
        self._lane_cursor: dict[str, int] = {}
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
    def _active_conditions(self) -> set[str] | None:
        """All conditions participating in primary or follow-up contrasts (None = all RNA conditions)."""
        if self.cfg.focal_conditions is None:
            return None
        conds = set(self.cfg.focal_conditions)
        if len(self.cfg.contrast) >= 3:
            conds.add(str(self.cfg.contrast[1]))
            conds.add(str(self.cfg.contrast[2]))
        for fc in self.cfg.followup_contrasts:
            if len(fc) >= 3:
                conds.add(str(fc[1]))
                conds.add(str(fc[2]))
        return conds

    def _global_bytes_cap_pct(self) -> float:
        """Reserve zoom headroom inside max_bytes_pct so global + zoom never exceeds max_bytes_pct (L5)."""
        if self.cfg.max_bytes_pct >= 100.0:
            return float(self.cfg.max_bytes_pct)
        if not self.cfg.zoom or self.cfg.zoom_extra_fraction <= 0:
            return float(self.cfg.max_bytes_pct)
        reserve = min(self.cfg.max_bytes_pct * 0.20, self.cfg.zoom_extra_fraction * 100.0)
        return max(0.0, float(self.cfg.max_bytes_pct) - reserve)
    def _focal(self, r: Run) -> bool:
        ac = self._active_conditions()
        return ac is None or r.condition in ac

    def _effective_min_mapped_per_sample(self) -> int:
        if self.cfg.min_mapped_per_sample is not None:
            return max(0, int(self.cfg.min_mapped_per_sample))
        n_groups = len(self.catalog.groups)
        if n_groups < 500:
            return 0
        return min(15_000_000, max(1_500_000, 750 * n_groups))

    def _floor_aware_next_fraction(self, cur_frac: float, cum_pct: float) -> float:
        """Compute next iteration increment fraction using empirical per-sample mapped-reads-per-byte."""
        floor = self._effective_min_mapped_per_sample()
        if floor < 1_000_000:
            return cur_frac * self.cfg.growth
        focal = [r for r in self.manifest.rna_runs if self._focal(r)]
        if not focal:
            return cur_frac * self.cfg.growth
        by_sample: dict[str, list[Run]] = {}
        for r in focal:
            by_sample.setdefault(r.sample, []).append(r)
        est_full: list[float] = []
        cur_mapped: list[float] = []
        need_cum_fracs: list[float] = []
        for runs_s in by_sample.values():
            m_s = float(sum(self.accs[r.run_id].n_mapped for r in runs_s))
            c_s = float(sum(self.accs[r.run_id].bytes_consumed for r in runs_s))
            t_s = float(sum(max(1, self.streams[r.run_id].total_bytes) for r in runs_s))
            cur_mapped.append(m_s)
            if m_s > 0 and c_s > 0:
                ef = m_s * (t_s / c_s)
                est_full.append(ef)
                need_cum_fracs.append(min(1.0, (1.06 * floor) / max(1.0, ef)))
            else:
                est_full.append(float(floor))
                need_cum_fracs.append(1.0)
        min_est_full = min(est_full) if est_full else float(floor)
        min_cur = min(cur_mapped) if cur_mapped else 0.0
        cur_cum = cum_pct / 100.0
        # Shallow archive (entire archive has <= 1.20x floor mapped reads per sample): finish cleanly in 3 steps (~2.5% -> ~50% -> 100%)
        if min_est_full <= 1.20 * floor:
            if cur_cum < 0.35:
                return max(cur_frac * self.cfg.growth, 0.50 - cur_cum)
            return max(0.60, 1.02 - cur_cum)
        # Deep archive (e.g. 175M-350M pairs/sample): jump iter 2 to ~55% of floor, iter 3 to ~106% of floor, then 1.55x if needed
        f_floor = max(need_cum_fracs) if need_cum_fracs else 1.0
        if min_cur < 0.45 * floor:
            return max(0.005, min(0.50, max(cur_frac * 1.5, 0.55 * f_floor - cur_cum)))
        if min_cur < floor:
            return max(0.008, min(0.65, 1.06 * f_floor - cur_cum))
        return max(0.015, min(0.35, 0.55 * max(cur_cum, 0.02)))

    def _sample_depth_floors_met(self) -> tuple[bool, float]:
        """Check whether every active biological sample has met its mapped-fragment floor (or exhausted its archive)."""
        floor = self._effective_min_mapped_per_sample()
        focal = [r for r in self.manifest.rna_runs if self._focal(r)]
        if not focal:
            return True, 0.0
        by_sample: dict[str, list[Run]] = {}
        for r in focal:
            by_sample.setdefault(r.sample, []).append(r)
        mapped_vals: list[float] = []
        all_met = True
        for s, runs_s in by_sample.items():
            m_s = float(sum(self.accs[r.run_id].n_mapped for r in runs_s))
            mapped_vals.append(m_s)
            if floor <= 0:
                continue
            exh_s = all(self.streams[r.run_id].exhausted for r in runs_s)
            tot_b = sum(self.streams[r.run_id].total_bytes for r in runs_s)
            con_b = sum(self.streams[r.run_id].consumed_bytes for r in runs_s)
            near_end = tot_b > 0 and con_b >= 0.96 * tot_b
            if m_s < floor and not exh_s and not near_end:
                all_met = False
        return all_met, (float(np.mean(mapped_vals)) if mapped_vals else 0.0)

    def _active_contrasts(self, meta: pd.DataFrame) -> list[list[str]]:
        """Ordered active contrasts (primary + followups + remaining multi-group levels when unfiltered)."""
        out: list[list[str]] = [list(self.cfg.contrast)]
        seen = {tuple(self.cfg.contrast)}
        for fc in self.cfg.followup_contrasts:
            k = tuple(fc)
            if len(fc) >= 3 and k not in seen:
                seen.add(k)
                out.append(list(fc))
        if self.cfg.focal_conditions is None and len(self.cfg.contrast) >= 3:
            factor, _a, ref_lvl = self.cfg.contrast[0], str(self.cfg.contrast[1]), str(self.cfg.contrast[2])
            if factor in meta.columns:
                levels = sorted(meta[factor].dropna().astype(str).unique())
                if ref_lvl in levels and len(levels) > 2:
                    for lv in levels:
                        if lv == ref_lvl:
                            continue
                        cand = [factor, lv, ref_lvl]
                        if tuple(cand) not in seen and (ref_lvl, lv) not in {(c[1], c[2]) for c in out}:
                            seen.add(tuple(cand))
                            out.append(cand)
        return out

    def _conv_expr_subset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """CPM-aware expressed gene subset (>= max(expressed_reads, 1.0 CPM)) for ultra-deep convergence checks."""
        min_r = max(float(self.cfg.expressed_reads), 1.0 * float(df["streamed_reads"].sum()) / 1_000_000.0)
        sub = df[(df["streamed_reads"] >= min_r) & df["log2fc"].notna()]
        if sub.empty:
            sub = df[(df["streamed_reads"] >= self.cfg.expressed_reads) & df["log2fc"].notna()]
            min_r = float(self.cfg.expressed_reads)
        return sub, min_r

    def _distribute_sample_budget(self, sample: str, runs_s: list[Run], sample_budget: int, iteration: int) -> dict[str, int]:
        """Distribute a biological sample's byte budget across its sequencing lanes (round-robin / proportional)."""
        if len(runs_s) == 1:
            r = runs_s[0]
            if iteration == 1:
                return {r.run_id: max(self.cfg.min_bytes_per_file * len(r.files), int(sample_budget))}
            return {r.run_id: int(sample_budget)}
        lane_alloc = {r.run_id: 0 for r in runs_s}
        avail = [
            r for r in runs_s
            if not self.streams[r.run_id].exhausted
            and (self.streams[r.run_id].total_bytes - self.streams[r.run_id].consumed_bytes) > 0
        ]
        if not avail or sample_budget <= 0:
            return lane_alloc
        min_lane = max(1, self.cfg.min_bytes_per_file * max(len(r.files) for r in avail))
        eff_budget = max(min_lane, int(sample_budget)) if iteration == 1 else int(sample_budget)
        if eff_budget >= min_lane * len(avail):
            rem_weights = {
                r.run_id: max(1, self.streams[r.run_id].total_bytes - self.streams[r.run_id].consumed_bytes)
                for r in avail
            }
            tot_rem = sum(rem_weights.values()) or 1
            for r in avail:
                lane_alloc[r.run_id] = int(eff_budget * rem_weights[r.run_id] / tot_rem)
            return lane_alloc
        # Budget smaller than min_lane * n_lanes: stripe round-robin across k_lanes and advance cursor next wave
        k_lanes = max(1, min(len(avail), eff_budget // min_lane))
        cur = self._lane_cursor.get(sample, 0) % len(avail)
        chosen = [avail[(cur + j) % len(avail)] for j in range(k_lanes)]
        self._lane_cursor[sample] = (cur + k_lanes) % len(avail)
        per_chosen = max(min_lane if iteration == 1 else 1, eff_budget // k_lanes)
        for r in chosen:
            lane_alloc[r.run_id] = int(per_chosen)
        return lane_alloc

    def allocate(self, iteration: int, budget_fraction: float) -> dict[str, int]:
        """Bytes per run for this iteration (variance-guided per biological sample, striped across lanes)."""
        alloc: dict[str, int] = {}
        focal = [r for r in self.manifest.rna_runs if self._focal(r)]
        bg = [r for r in self.manifest.rna_runs if not self._focal(r)]
        for r in self.manifest.wgs_runs:
            alloc[r.run_id] = self.cfg.wgs_bytes_per_iter * len(r.files)
        by_sample: dict[str, list[Run]] = {}
        for r in focal:
            by_sample.setdefault(r.sample, []).append(r)
        if iteration == 1 or not by_sample:
            for s, runs_s in by_sample.items():
                s_tot = sum(r.total_bytes for r in runs_s)
                alloc.update(self._distribute_sample_budget(s, runs_s, int(s_tot * budget_fraction), iteration))
        else:
            budget = sum(int(r.total_bytes * budget_fraction) for r in focal)
            s_depth = {s: float(sum(self.accs[r.run_id].n_mapped for r in rs)) for s, rs in by_sample.items()}
            s_bpr: dict[str, float] = {}
            for s, rs in by_sample.items():
                cons = sum(self.accs[r.run_id].bytes_consumed for r in rs)
                mapped = sum(self.accs[r.run_id].n_mapped for r in rs)
                tot_b = sum(r.total_bytes for r in rs)
                tot_rc = sum(r.read_count for r in rs)
                s_bpr[s] = (cons / mapped) if mapped > 0 else (tot_b / max(1, tot_rc))
            est_new_reads = budget / max(1.0, float(np.mean(list(s_bpr.values()))))
            target = (sum(s_depth.values()) + est_new_reads) / len(by_sample)
            s_want = {s: s_bpr[s] * max(target - s_depth[s], 0.25 * target) for s in s_depth}
            if self.cfg.use_dothash_dedup and len(focal) >= 2:
                r_depth = {r.run_id: float(self.accs[r.run_id].n_mapped) for r in focal}
                r_want = {r.run_id: 1.0 for r in focal}
                self._apply_dothash_dedup_weights(focal, r_depth, r_want)
                for s, rs in by_sample.items():
                    s_want[s] *= float(np.mean([r_want[r.run_id] for r in rs]))
            # clip to [0.25x, 4x] of the uniform per-sample share so no biological library is starved or hogs the budget
            share = budget / len(by_sample)
            tot_w = sum(s_want.values()) or 1.0
            for s, runs_s in by_sample.items():
                raw = budget * s_want[s] / tot_w
                s_bud = int(min(4.0 * share, max(0.25 * share, raw)))
                alloc.update(self._distribute_sample_budget(s, runs_s, s_bud, iteration))
        for r in bg:
            alloc[r.run_id] = max(self.cfg.min_bytes_per_file * len(r.files), int(r.total_bytes * budget_fraction * self.cfg.bg_scale))
        for r in self.manifest.runs:
            rs = self.streams[r.run_id]
            rem_r = max(0, rs.total_bytes - rs.consumed_bytes)
            want_r = int(min(alloc.get(r.run_id, 0), rem_r))
            # Snap tiny residual tail slivers (<= 2% of run, up to 64 KiB) on iter > 1 so we never run an extra wave on gzip trailers
            if iteration > 1 and want_r > 0 and 0 < (rem_r - want_r) <= min(65_536, max(64, int(0.02 * rs.total_bytes))):
                want_r = rem_r
            alloc[r.run_id] = want_r
        # never stream past an explicit max_bytes_pct ceiling: scale down to remaining allowance,
        # and skip near-zero tail slivers after min_iterations so we do not run a full wave + DE solve for <0.1% bytes.
        self._last_alloc_cap_clipped = False
        if self.cfg.max_bytes_pct < 100.0:
            cum = sum(a.bytes_consumed for a in self.accs.values())
            eff_cap_pct = self._global_bytes_cap_pct()
            remaining = eff_cap_pct / 100.0 * self.manifest.total_archive_bytes - cum
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
    def _stream_and_map_wave(self, alloc: dict[str, int]) -> tuple[float, float]:
        t0 = time.time()

        def _fetch(run_id: str):
            rs = self.streams[run_id]
            before = rs.consumed_bytes
            r1, r2 = rs.advance_bytes(alloc[run_id])
            return run_id, r1, r2, rs.consumed_bytes - before

        active_ids = [k for k, v in alloc.items() if v > 0]
        if not active_ids:
            return 0.0, 0.0
        with ThreadPoolExecutor(max_workers=min(self.cfg.fetch_workers, len(active_ids))) as ex:
            fetched = list(ex.map(_fetch, active_ids))
        io_s = time.time() - t0
        t1 = time.time()
        for run_id, _r1, _r2, delta in fetched:
            self.accs[run_id].bytes_consumed += delta
        todo = [(run_id, r1, r2) for run_id, r1, r2, _d in fetched if r1]
        del fetched
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
        # Pool multi-lane batches belonging to the same biological (sample, condition, layout) before kernel invocation
        # so multi-lane cohorts (e.g. 256 runs = 16 biological samples x 16 lanes) load the mapper index 16x instead of 256x.
        sample_lanes: dict[tuple[str, str, bool], list[str]] = {}
        grouped_todo: dict[tuple[str, str, bool], list[tuple[str, list, list | None]]] = {}
        wgs_todo: list[tuple[str, list, list | None]] = []
        for run_id, r1, r2 in todo:
            r_obj = self.accs[run_id].run
            if r_obj.is_wgs:
                wgs_todo.append((run_id, r1, r2))
            else:
                sk = (r_obj.sample, r_obj.condition, r2 is not None)
                grouped_todo.setdefault(sk, []).append((run_id, r1, r2))
                sample_lanes.setdefault(sk, []).append(run_id)
        del todo

        map_items: list[tuple[str, list, list | None]] = list(wgs_todo)
        rep_to_lanes: dict[str, list[str]] = {}
        for sk, items in grouped_todo.items():
            # Canonical biological-sample accumulator is always the first lane of that sample in manifest order
            canon_rid = next(
                (
                    r.run_id
                    for r in self.manifest.rna_runs
                    if r.sample == sk[0] and r.condition == sk[1] and (r.layout == "PE") == sk[2]
                ),
                items[0][0],
            )
            rep_to_lanes[canon_rid] = sample_lanes[sk]
            if len(items) == 1 and items[0][0] == canon_rid:
                map_items.append(items[0])
            else:
                cat_r1: list = []
                cat_r2: list | None = [] if sk[2] else None
                for _rid, x1, x2 in items:
                    cat_r1.extend(x1)
                    if cat_r2 is not None and x2 is not None:
                        cat_r2.extend(x2)
                map_items.append((canon_rid, cat_r1, cat_r2))
        del grouped_todo

        if self.cfg.pool_samples and (getattr(self.kernel, "supports_pooling", False) or hasattr(self.kernel, "clone_with_threads")):
            for paired in (True, False):
                grp = [(rid, r1, r2, self.accs[rid].calibrator) for rid, r1, r2 in map_items if (r2 is not None) == paired]
                if grp:
                    self.pooled_calls += 1
                    for rid, res in map_pooled(self.kernel, grp).items():
                        self.accs[rid].add(res)
        else:
            for run_id, r1, r2 in map_items:
                acc = self.accs[run_id]
                acc.add(self.kernel.map_batch(r1, r2, acc.calibrator))
        del map_items
        # LOW_MAP_RATE: sources that partition reads (sra_odp) keep serving discovery blocks only to flagged runs
        for canon_rid, lane_rids in rep_to_lanes.items():
            mf = self.accs[canon_rid].mapped_fraction
            for rid in lane_rids:
                st = self.streams[rid]
                if hasattr(st, "flag_low_map_rate") and st.flag_low_map_rate(mf):
                    self.low_map_rate_runs.add(rid)
        return io_s, time.time() - t1

    def stream_and_map(self, alloc: dict[str, int]) -> tuple[float, float]:
        """Bounded-RAM streaming + mapping: caps raw reads in Python heap per micro-wave against cgroup RAM."""
        import gc
        import math
        import os
        from .regime import available_ram_bytes

        raw_env = os.environ.get("PANEXPR_MAX_WAVE_BYTES")
        if raw_env:
            max_wave_bytes = int(raw_env)
        else:
            max_wave_bytes = min(1_200_000_000, max(67_108_864, available_ram_bytes() // 64))
        tot_planned = sum(v for v in alloc.values() if v > 0)
        if tot_planned <= max_wave_bytes:
            return self._stream_and_map_wave(alloc)

        # Group run allocations by biological sample so multi-lane pooling stays intact within each wave
        units: list[dict[str, int]] = []
        by_smp: dict[tuple[str, str, str], dict[str, int]] = {}
        for r in self.manifest.runs:
            v = int(alloc.get(r.run_id, 0))
            if v <= 0:
                continue
            key = ("wgs", r.run_id, "") if r.is_wgs else (r.sample, r.condition, r.layout)
            by_smp.setdefault(key, {})[r.run_id] = v

        for smp_alloc in by_smp.values():
            s_bytes = sum(smp_alloc.values())
            if s_bytes <= max_wave_bytes:
                units.append(smp_alloc)
            else:
                n_sub = max(2, int(math.ceil(s_bytes / max_wave_bytes)))
                rem = dict(smp_alloc)
                for step_i in range(n_sub):
                    left_steps = n_sub - step_i
                    chunk = {}
                    for rid, rem_b in rem.items():
                        take = rem_b if left_steps == 1 else int(math.ceil(rem_b / left_steps))
                        if take > 0:
                            chunk[rid] = take
                            rem[rid] -= take
                    if chunk:
                        units.append(chunk)

        waves: list[dict[str, int]] = []
        cur_wave: dict[str, int] = {}
        cur_bytes = 0
        for u in units:
            u_bytes = sum(u.values())
            if cur_wave and (cur_bytes + u_bytes > max_wave_bytes):
                waves.append(cur_wave)
                cur_wave = {}
                cur_bytes = 0
            for rid, b in u.items():
                cur_wave[rid] = cur_wave.get(rid, 0) + b
            cur_bytes += u_bytes
        if cur_wave:
            waves.append(cur_wave)

        io_tot = map_tot = 0.0
        for w in waves:
            wi, wm = self._stream_and_map_wave(w)
            io_tot += wi
            map_tot += wm
            gc.collect()
        return io_tot, map_tot

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
                if str(k).startswith("_"):
                    continue
                if k == "covariates" and isinstance(v, dict):
                    d.update({ck: cv for ck, cv in v.items() if not str(ck).startswith("_")})
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
        strand: dict[str, float] = {}
        rna_by_scl: dict[tuple[str, str, str], list[RunAccumulator]] = {}
        for a in self.accs.values():
            if a.run.is_wgs:
                strand[a.run.run_id] = 0.5
            else:
                rna_by_scl.setdefault((a.run.sample, a.run.condition, a.run.layout), []).append(a)
        targets_map = self.catalog.targets
        for grp_accs in rna_by_scl.values():
            a_sum, b_sum = 1.0, 1.0
            for ga in grp_accs:
                for tid, (c0, c1) in ga.unique_counts.items():
                    t_obj = targets_map.get(tid)
                    if t_obj is not None and not t_obj.is_decoy:
                        a_sum += c0
                        b_sum += c1
            p_grp = a_sum / (a_sum + b_sum)
            for ga in grp_accs:
                strand[ga.run.run_id] = p_grp
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

    def close_streams(self) -> None:
        for st in self.streams.values():
            if hasattr(st, "close"):
                try:
                    st.close()
                except Exception:
                    pass

    # ---------------- main ----------------
    def run(self, close_streams: bool = True) -> LoopReport:
        cfg = self.cfg
        t_start = time.time()
        total_archive = self.manifest.total_archive_bytes
        telemetry: list[dict] = []
        schedule: list[dict] = []
        prev_de: pd.DataFrame | None = None
        prev_sec_de: dict[tuple[str, ...], pd.DataFrame] = {}
        last_sec_de: dict[tuple[str, ...], pd.DataFrame] = {}
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
                stop_reason = (
                    "archive_exhausted"
                    if all(
                        self.streams[r.run_id].exhausted
                        or (self.streams[r.run_id].consumed_bytes >= self.streams[r.run_id].total_bytes > 0)
                        for r in self.manifest.rna_runs
                        if self._focal(r)
                    )
                    else "max_bytes_pct_cap"
                )
                it -= 1
                break
            schedule.append({"iteration": it, "alloc": dict(alloc)})
            io_s, map_s = self.stream_and_map(alloc)
            io_tot += io_s
            map_tot += map_s
            t_s = time.time()
            de, table, extra = self.solve(it, fast_de=(cfg.fast_intermediate_de and it < cfg.max_iterations))
            gl_it = extra["gene_tables"]
            counts_it = gl_it.get("counts_length_scaled", gl_it["counts"]) if cfg.length_offsets else gl_it["counts"]
            sec_contrasts = self._active_contrasts(gl_it["meta"])[1:]
            cur_sec_de: dict[tuple[str, ...], pd.DataFrame] = {}
            for sc in sec_contrasts:
                try:
                    cur_sec_de[tuple(sc)] = fast_nb_mom_de(
                        counts_it, gl_it["avar"], gl_it["meta"], sc, cfg.design, gl_it["dna_prob"],
                        self.symbols, self.gene_ids, it, cfg.expressed_reads,
                    )
                except Exception:
                    pass
            solve_tot += time.time() - t_s
            cum = sum(a.bytes_consumed for a in self.accs.values())
            pct = 100.0 * cum / max(1, total_archive)
            expressed = de[(de["streamed_reads"] >= cfg.expressed_reads) & de["log2fc"].notna()]
            conv_expr, conv_min_r = self._conv_expr_subset(de)
            med_tech = float(conv_expr["tech_var_fraction"].median()) if len(conv_expr) else 1.0
            if prev_de is not None:
                j = conv_expr.set_index("group_id")[["log2fc"]].merge(prev_de.set_index("group_id")[["log2fc", "streamed_reads"]], left_index=True, right_index=True, how="inner", suffixes=("", "_prev"))
                prev_min_r = max(float(cfg.expressed_reads), 1.0 * float(prev_de["streamed_reads"].sum()) / 1_000_000.0)
                j = j[j["streamed_reads"] >= prev_min_r]
                if len(j):
                    abs_d = (j["log2fc"] - j["log2fc_prev"]).abs()
                    med_delta = float(abs_d.median())
                    p90_delta = float(abs_d.quantile(0.90))
                    p95_delta = float(abs_d.quantile(0.95))
                else:
                    med_delta = p90_delta = p95_delta = float("nan")
            else:
                med_delta = p90_delta = p95_delta = float("nan")
            worst_delta = med_delta
            worst_tech = med_tech
            for sc_key, sde in cur_sec_de.items():
                sexpr, _ = self._conv_expr_subset(sde)
                st_f = float(sexpr["tech_var_fraction"].median()) if len(sexpr) else 1.0
                if st_f > worst_tech:
                    worst_tech = st_f
                if sc_key in prev_sec_de:
                    p_sde = prev_sec_de[sc_key]
                    p_smin = max(float(cfg.expressed_reads), 1.0 * float(p_sde["streamed_reads"].sum()) / 1_000_000.0)
                    sj = sexpr.set_index("group_id")[["log2fc"]].merge(
                        p_sde.set_index("group_id")[["log2fc", "streamed_reads"]],
                        left_index=True, right_index=True, how="inner", suffixes=("", "_prev"),
                    )
                    sj = sj[sj["streamed_reads"] >= p_smin]
                    sd_f = float((sj["log2fc"] - sj["log2fc_prev"]).abs().median()) if len(sj) else float("nan")
                else:
                    sd_f = float("nan")
                if np.isnan(sd_f) or np.isnan(worst_delta):
                    worst_delta = float("nan")
                elif sd_f > worst_delta:
                    worst_delta = sd_f
            floors_met, _mean_sample_mapped = self._sample_depth_floors_met()
            n_deg = int((de["padj"] < 0.05).sum())
            total_reads = sum(a.n_reads for a in self.accs.values() if not a.run.is_wgs)
            mapped = sum(a.n_mapped for a in self.accs.values() if not a.run.is_wgs)
            reason = ""
            converged = (
                it >= cfg.min_iterations
                and floors_met
                and not np.isnan(worst_delta)
                and worst_delta < cfg.delta_lfc_stop
                and worst_tech <= cfg.tech_var_stop
            )
            exhausted = all(
                self.streams[r.run_id].exhausted
                or (self.streams[r.run_id].consumed_bytes >= self.streams[r.run_id].total_bytes > 0)
                for r in self.manifest.rna_runs
                if self._focal(r)
            )
            if converged:
                reason = "converged_tech_var_below_bio_var"
            elif cfg.max_bytes_pct < 100.0 and (
                pct >= self._global_bytes_cap_pct() - 1e-6
                or (self._last_alloc_cap_clipped and it >= cfg.min_iterations and pct >= self._global_bytes_cap_pct() * 0.98)
            ):
                reason = "max_bytes_pct_cap"
            elif exhausted:
                reason = "archive_exhausted"
            if reason and extra.get("used_fast_de"):
                t_exact = time.time()
                de = self._de_from_tables(extra["gene_tables"], it)
                solve_tot += time.time() - t_exact
                extra["used_fast_de"] = False
                expressed = de[(de["streamed_reads"] >= cfg.expressed_reads) & de["log2fc"].notna()]
                conv_expr_ex, _ = self._conv_expr_subset(de)
                med_tech = float(conv_expr_ex["tech_var_fraction"].median()) if len(conv_expr_ex) else 1.0
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
                     f"DEG={n_deg} median|dLFC|={med_delta:.4f} p90|dLFC|={p90_delta:.4f} median_tech_var={med_tech:.3f} "
                     f"floors_met={floors_met} [{time.time() - t_it:.1f}s] {reason}")
            prev_de = de
            prev_sec_de = cur_sec_de
            last_sec_de = cur_sec_de
            if reason:
                stop_reason = reason
                break
            frac = self._floor_aware_next_fraction(frac, pct)
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
        any_stream_left = any(
            not self.streams[r.run_id].exhausted
            and (self.streams[r.run_id].total_bytes - self.streams[r.run_id].consumed_bytes) > 0
            for r in self.manifest.rna_runs
        )
        max_total_allowed_bytes = (
            int(cfg.max_bytes_pct / 100.0 * total_archive)
            if cfg.max_bytes_pct < 100.0
            else None
        )
        rem_cap_bytes = (
            max(0, max_total_allowed_bytes - bytes_global)
            if max_total_allowed_bytes is not None
            else None
        )
        if cfg.zoom and any_stream_left and (rem_cap_bytes is None or rem_cap_bytes > 0):
            use_v24_sieve = cfg.abundant_sieve and (
                (len(self.catalog.groups) >= 500 and cfg.zoom_extra_fraction <= 0.02)
                or cfg.abundant_sieve_force
            )
            abundant_groups: list[str] | None = None
            if use_v24_sieve:
                groups, abundant_groups = select_zoom_targets(
                    de_pre, padj_lo=cfg.zoom_padj[0], padj_hi=cfg.zoom_padj[1], low_reads_hi=cfg.sieve_low_reads_hi,
                )
                if last_sec_de:
                    gset_u = set(groups)
                    aset_u = set(abundant_groups)
                    for sde in last_sec_de.values():
                        sg, sa = select_zoom_targets(
                            sde, padj_lo=cfg.zoom_padj[0], padj_hi=cfg.zoom_padj[1], low_reads_hi=cfg.sieve_low_reads_hi,
                        )
                        gset_u.update(sg)
                        aset_u &= set(sa)
                    groups = [g for g in de_pre["group_id"] if g in gset_u]
                    abundant_groups = [g for g in abundant_groups if g in aset_u and g not in gset_u]
                _fl_met, mean_sample_mapped = self._sample_depth_floors_met()
                global_frac = bytes_global / max(1, total_archive)
                rem_frac = max(0.0, 1.0 - global_frac)
                rem_slice = cfg.sieve_max_rem_fraction * rem_frac
                target_zoom_mapped = (
                    28_000_000.0
                    if len(self.catalog.groups) >= 20_000
                    else max(3_000_000.0, 2.0 * float(self._effective_min_mapped_per_sample()))
                )
                if mean_sample_mapped > 0 and global_frac > 0:
                    needed_frac = global_frac * max(
                        0.35, (target_zoom_mapped - mean_sample_mapped) / max(1.0, mean_sample_mapped)
                    )
                    zoom_frac = max(cfg.zoom_extra_fraction, min(rem_slice, needed_frac))
                else:
                    zoom_frac = max(cfg.zoom_extra_fraction, rem_slice)
            else:
                groups = select_borderline(de_pre, cfg.zoom_padj[0], cfg.zoom_padj[1], cfg.zoom_reads[0], cfg.zoom_reads[1])
                zoom_frac = cfg.zoom_extra_fraction
            if rem_cap_bytes is not None:
                zoom_frac = min(zoom_frac, rem_cap_bytes / max(1, total_archive))
            if groups and zoom_frac > 0:
                t_z = time.time()
                rna_streams = {k: v for k, v in self.streams.items() if not v.run.is_wgs}
                z_lane_alloc: dict[str, int] | None = None
                by_smp_z: dict[str, list[Run]] = {}
                for r in self.manifest.rna_runs:
                    by_smp_z.setdefault(r.sample, []).append(r)
                if rem_cap_bytes is not None or any(len(rs_z) > 1 for rs_z in by_smp_z.values()):
                    z_lane_alloc = {}
                    for s_z, rs_z in by_smp_z.items():
                        s_tot_z = sum(self.streams[r.run_id].total_bytes for r in rs_z)
                        s_rem_z = sum(max(0, self.streams[r.run_id].total_bytes - self.streams[r.run_id].consumed_bytes) for r in rs_z)
                        base_want = int(round(s_tot_z * zoom_frac))
                        if len(rs_z) > 1 and rem_cap_bytes is None:
                            base_want = max(cfg.min_bytes_per_file * 2, base_want)
                        want_z = min(s_rem_z, base_want)
                        if len(rs_z) == 1:
                            z_lane_alloc[rs_z[0].run_id] = max(0, int(want_z))
                        else:
                            z_lane_alloc.update(self._distribute_sample_budget(s_z, rs_z, want_z, it + 1))
                            for r_chk in rs_z:
                                rs_chk = self.streams[r_chk.run_id]
                                rem_chk = max(0, rs_chk.total_bytes - rs_chk.consumed_bytes)
                                z_lane_alloc[r_chk.run_id] = min(z_lane_alloc.get(r_chk.run_id, 0), rem_chk)
                    if rem_cap_bytes is not None:
                        tot_z_planned = sum(z_lane_alloc.values())
                        if tot_z_planned > rem_cap_bytes > 0:
                            scale_z = rem_cap_bytes / tot_z_planned
                            z_lane_alloc = {k: int(v * scale_z) for k, v in z_lane_alloc.items()}
                        elif rem_cap_bytes <= 0:
                            z_lane_alloc = {k: 0 for k in z_lane_alloc}
                if z_lane_alloc is None or sum(z_lane_alloc.values()) > 0:
                    zoom_res = zoom_pass(
                        self.kernel, self.catalog, self.accs, rna_streams, groups,
                        zoom_frac, log=self.log, fetch_workers=cfg.fetch_workers, pool_samples=cfg.pool_samples,
                        abundant_groups=abundant_groups,
                        sieve_mode=(cfg.sieve_mode if use_v24_sieve else "include"),
                        sieve_threads=getattr(self.kernel, "threads", 4),
                        sample_lane_alloc=z_lane_alloc,
                        close_streams=False,
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
            name = f"{fc[1]}_vs_{fc[2]}"
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
                p = self.state.write(f"gene_de_followup_{name}", pa.Table.from_pandas(fde, preserve_index=False), GENE_DE_SCHEMA)
                followups[name] = {"wall_seconds": time.time() - t_f, "new_fastq_bytes": 0, "n_deg_fdr05": int((fde["padj"] < 0.05).sum()), "parquet": str(p)}
            except Exception as exc:
                self.log(f"  [followup {name}] failed ({type(exc).__name__} - {exc})")
                followups[name] = {"error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.time() - t_f, "new_fastq_bytes": 0}
                if close_streams:
                    self.close_streams()
                raise ValueError(f"Follow-up contrast {fc!r} failed: {exc}") from exc
        malformed_by_run = {
            rid: int(getattr(st, "malformed_skipped", 0))
            for rid, st in self.streams.items()
            if int(getattr(st, "malformed_skipped", 0)) > 0
        }
        unpaired_by_run = {
            rid: int(getattr(st, "skipped_unpaired_mates", 0))
            for rid, st in self.streams.items()
            if int(getattr(st, "skipped_unpaired_mates", 0)) > 0
        }
        if malformed_by_run:
            self.log(f"  [stream-qc] skipped malformed FASTQ lines in runs: {malformed_by_run}")
        if unpaired_by_run:
            self.log(f"  [stream-qc] skipped unpaired PE mates in runs: {unpaired_by_run}")
        if close_streams:
            self.close_streams()
        tx_table = transcript_level_table(
            table,
            self.catalog,
            zoom_table=self._last_zoom_table,
            zoom_groups=(self._last_zoom_ctx["groups"] if self._last_zoom_ctx else None),
            depth_scale=(self._last_zoom_ctx["depth_scale"] if self._last_zoom_ctx else None),
        )
        paths = {
            "target_counts": str(self.state.write("target_counts", table, TARGET_COUNTS_SCHEMA)),
            "transcript_abundances": str(self.state.write("transcript_abundances", tx_table, TRANSCRIPT_ABUNDANCE_SCHEMA)),
            "gene_de_prezoom": str(self.state.write("gene_de_prezoom", pa.Table.from_pandas(de_pre, preserve_index=False), GENE_DE_SCHEMA)),
            "gene_de_postzoom": str(self.state.write("gene_de_postzoom", pa.Table.from_pandas(de_post, preserve_index=False), GENE_DE_SCHEMA)),
            "telemetry": str(self.state.write("telemetry", pa.Table.from_pylist(telemetry), ITERATION_TELEMETRY_SCHEMA)),
        }
        if self._last_zoom_table is not None:
            paths["target_counts_zoom"] = str(self.state.write("target_counts_zoom", self._last_zoom_table, TARGET_COUNTS_SCHEMA))
            if self._last_zoom_ctx is not None:
                zoom_ctx_path = self.state.dir / "zoom_context.json"
                zoom_ctx_payload = {
                    "zoomed_groups": [str(g) for g in self._last_zoom_ctx.get("groups", [])],
                    "sample_depth_scale": {
                        str(k): float(v) for k, v in self._last_zoom_ctx["depth_scale"].items()
                    },
                }
                zoom_ctx_path.write_text(json.dumps(zoom_ctx_payload, indent=2))
                paths["zoom_context"] = str(zoom_ctx_path)
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
        return LoopReport(
            it, stop_reason, bytes_global, bytes_zoom, 100.0 * (bytes_global + bytes_zoom) / max(1, total_archive),
            wall, io_tot, map_tot, solve_tot, telemetry, schedule, p1_asm, p2_asm, extra["assembly_prior"], extra["strand"],
            de_pre, de_post, zoom_res, followups, paths,
            malformed_skipped=malformed_by_run,
            skipped_unpaired_mates=unpaired_by_run,
        )

    def _sample_depth_scale(self, samples: list[str], zoom_res: ZoomResult) -> pd.Series:
        """Per-sample global/(global+zoom) byte fraction, aggregating multi-run samples accurately."""
        glob_b: dict[str, float] = {}
        tot_b: dict[str, float] = {}
        has_explicit_zb = bool(zoom_res.per_run_zoom_bytes)
        rids = (
            [r.run_id for r in self.manifest.rna_runs]
            if has_explicit_zb
            else list(dict.fromkeys(list(zoom_res.per_run_scale.keys()) + [r.run_id for r in self.manifest.rna_runs]))
        )
        for rid in rids:
            if rid not in self.accs:
                continue
            acc = self.accs[rid]
            if acc.run.is_wgs:
                continue
            s = acc.run.sample
            gb = float(acc.bytes_consumed)
            if has_explicit_zb:
                zb = float(zoom_res.per_run_zoom_bytes.get(rid, 0))
            elif rid in self.streams and self.streams[rid].consumed_bytes > acc.bytes_consumed:
                zb = float(self.streams[rid].consumed_bytes - acc.bytes_consumed)
            elif rid in zoom_res.per_run_scale and gb > 0 and zoom_res.per_run_scale[rid] > 0:
                zb = max(0.0, gb / float(zoom_res.per_run_scale[rid]) - gb)
            else:
                z_acc = getattr(acc, "zoom", None)
                zb = float(z_acc.bytes_consumed if z_acc is not None else 0.0)
            glob_b[s] = glob_b.get(s, 0.0) + gb
            tot_b[s] = tot_b.get(s, 0.0) + gb + zb
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
