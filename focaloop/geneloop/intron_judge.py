"""Good-Turing unseen-intron Judge and multi-library BioSample bandit scheduler.

Replaces egapx's 100%-archive download + 4x BAM sort/bin/merge barriers with:
1. Per-library Good-Turing unseen canonical-intron probability (p0_hat = f1 / N).
2. Marginal global novel-intron discovery slope (nu_r = delta_novel_introns / delta_mb)
   halting redundant replicate libraries at 3-10% while reallocating stream budget to
   high-novelty tissues (testis, brain, embryo, Iso-Seq/ONT).
3. Separate tracking of protein-anchored coding junctions vs intergenic ncRNA junctions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CANONICAL_MOTIFS = frozenset({"GT-AG", "GC-AG", "AT-AC"})
IntronKey = tuple[str, int, int, str]  # (contig, donor_0b, acceptor_0b, strand)


@dataclass(frozen=True)
class IntronObservation:
    contig: str
    donor: int
    acceptor: int
    strand: str
    motif: str
    overhang: int
    is_coding_anchored: bool = True


@dataclass
class LibraryIntronTracker:
    run_id: str
    biosample_id: str
    is_long_read: bool = False
    min_overhang: int = 8
    bytes_streamed: int = 0
    waves_completed: int = 0
    total_valid_spliced_obs: int = 0
    counts: dict[IntronKey, int] = field(default_factory=dict)
    recent_novel_slopes_per_10mb: list[float] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""

    def good_turing_unseen_rate(self) -> float:
        if self.total_valid_spliced_obs <= 0:
            return 1.0
        f1 = sum(1 for c in self.counts.values() if c == 1)
        return f1 / float(self.total_valid_spliced_obs)


class MultiRunIntronBanditJudge:
    """Tracks global splice-graph saturation across 10-100 SRA libraries."""

    def __init__(
        self,
        replicate_slope_floor_per_10mb: float = 2.0,
        global_good_turing_floor: float = 0.002,
        ncrna_slope_floor_per_10mb: float = 1.0,
        patience_waves: int = 2,
        min_anchor_overhang: int = 8,
        strict_ncrna_mode: bool = False,
    ) -> None:
        self.replicate_slope_floor = replicate_slope_floor_per_10mb
        self.global_good_turing_floor = global_good_turing_floor
        self.ncrna_slope_floor = ncrna_slope_floor_per_10mb
        self.patience_waves = patience_waves
        self.min_anchor_overhang = min_anchor_overhang
        self.strict_ncrna_mode = strict_ncrna_mode

        self.libraries: dict[str, LibraryIntronTracker] = {}
        self.global_coding_counts: dict[IntronKey, int] = {}
        self.global_ncrna_counts: dict[IntronKey, int] = {}
        self.global_total_coding_obs: int = 0
        self.recent_ncrna_slopes: list[float] = []
        self.consecutive_global_converged: int = 0

    def register_library(self, run_id: str, biosample_id: str, is_long_read: bool = False) -> None:
        self.libraries[run_id] = LibraryIntronTracker(
            run_id=run_id,
            biosample_id=biosample_id,
            is_long_read=is_long_read,
            min_overhang=self.min_anchor_overhang,
        )

    def ingest_wave(
        self,
        run_id: str,
        wave_bytes: int,
        observations: list[IntronObservation],
        compleasm_intact_delta: int = 0,
    ) -> dict[str, float | bool | str]:
        lib = self.libraries[run_id]
        lib.bytes_streamed += wave_bytes
        lib.waves_completed += 1

        novel_global_coding = 0
        novel_global_ncrna = 0

        for obs in observations:
            if obs.motif.upper() not in CANONICAL_MOTIFS:
                continue
            if not lib.is_long_read and obs.overhang < self.min_anchor_overhang:
                continue
            key: IntronKey = (obs.contig, obs.donor, obs.acceptor, obs.strand)
            lib.counts[key] = lib.counts.get(key, 0) + 1
            lib.total_valid_spliced_obs += 1

            if obs.is_coding_anchored:
                self.global_total_coding_obs += 1
                prev = self.global_coding_counts.get(key, 0)
                if prev == 1:
                    # Just crossed support >= 2 globally (noise-filtered discovery)
                    novel_global_coding += 1
                self.global_coding_counts[key] = prev + 1
            else:
                prev_nc = self.global_ncrna_counts.get(key, 0)
                if prev_nc == 1:
                    novel_global_ncrna += 1
                self.global_ncrna_counts[key] = prev_nc + 1

        mb = max(wave_bytes / 1e6, 0.1)
        slope_10mb = ((novel_global_coding + novel_global_ncrna) / mb) * 10.0
        ncrna_slope_10mb = (novel_global_ncrna / mb) * 10.0
        lib.recent_novel_slopes_per_10mb.append(slope_10mb)
        self.recent_ncrna_slopes.append(ncrna_slope_10mb)

        # Long-read libraries (Iso-Seq / ONT) are never halted early unless slope hits 0 repeatedly
        if not lib.is_long_read and len(lib.recent_novel_slopes_per_10mb) >= self.patience_waves:
            tail = lib.recent_novel_slopes_per_10mb[-self.patience_waves :]
            if all(s < self.replicate_slope_floor for s in tail):
                lib.halted = True
                lib.halt_reason = "CONVERGED_REPLICATE_OR_SATURATED_TISSUE"

        gt_coding = self.global_coding_good_turing()
        ncrna_ok = True
        if self.strict_ncrna_mode and len(self.recent_ncrna_slopes) >= self.patience_waves:
            ncrna_ok = all(s < self.ncrna_slope_floor for s in self.recent_ncrna_slopes[-self.patience_waves :])
        elif self.strict_ncrna_mode:
            ncrna_ok = False

        if gt_coding < self.global_good_turing_floor and compleasm_intact_delta == 0 and ncrna_ok:
            self.consecutive_global_converged += 1
        else:
            self.consecutive_global_converged = 0

        return {
            "run_id": run_id,
            "slope_per_10mb": slope_10mb,
            "lib_halted": lib.halted,
            "global_coding_gt": gt_coding,
            "global_converged": self.is_globally_converged(),
        }

    def global_coding_good_turing(self) -> float:
        if self.global_total_coding_obs <= 0:
            return 1.0
        f1 = sum(1 for c in self.global_coding_counts.values() if c == 1)
        return f1 / float(self.global_total_coding_obs)

    def is_globally_converged(self) -> bool:
        return self.consecutive_global_converged >= 3

    def select_next_library(self, exploration_c: float = 2.0) -> str | None:
        """UCB1-like bandit prioritizing unprobed BioSamples, long reads, and high-novelty tissues."""
        active = [lib for lib in self.libraries.values() if not lib.halted]
        if not active:
            return None
        total_waves = sum(max(lib.waves_completed, 1) for lib in self.libraries.values())
        best_id = None
        best_score = -1.0
        for lib in active:
            if lib.waves_completed == 0:
                # Always probe every candidate run in Wave 0 first (long reads first)
                score = 1e9 + (1e6 if lib.is_long_read else 0.0)
            else:
                recent = lib.recent_novel_slopes_per_10mb[-1]
                bonus = exploration_c * math.sqrt(math.log(total_waves + 1.0) / lib.waves_completed)
                lr_mult = 2.5 if lib.is_long_read else 1.0
                score = lr_mult * (recent + bonus)
            if score > best_score:
                best_score = score
                best_id = lib.run_id
        return best_id


# ----------------------------------------------------------------------------------------------
# Aggregated-count ingestion + Good-Turing UCB (GT-UCB) allocation for the streaming loop
# ----------------------------------------------------------------------------------------------

REGION_LOCKED_KNOWN = "locked_known"
REGION_LOCKED_NOVEL = "locked_novel"
REGION_UNSETTLED = "unsettled"
REGION_INTERGENIC = "intergenic"


@dataclass
class WaveJunction:
    """One junction key aggregated over a wave (counts = uniquely-mapped, anchor-filtered reads)."""

    key: IntronKey
    count: int
    motif: str
    region: str


class GoodTuringUCBJudge(MultiRunIntronBanditJudge):
    """Extends the multi-run judge with aggregated ingestion and the GT-UCB index

        U_b = f1_b / N_b + c * sqrt(ln T / n_b)

    where, restricted to UNSETTLED-region canonical introns, f1_b is the number of introns library b
    observed exactly once that no other library has observed (b's Good-Turing estimate of the
    probability its next spliced read reveals a globally novel intron), N_b is b's unsettled-region
    spliced observations, n_b the waves pulled from b and T the total waves.

    Global stop: ``p0_hat = f1 / N < p0_stop`` over unsettled-region observations AND, when
    ``n_unsettled_loci > 0``, trailing per-locus reproducible (support >= 2) unsettled intron discovery
    rate ``<= max_novel_per_unsettled_1m`` introns / unsettled locus / 1M spots.
    """

    def __init__(
        self,
        p0_stop: float = 0.005,
        ucb_c: float = 0.02,
        n_unsettled_loci: int = 0,
        max_novel_per_unsettled_1m: float = 0.12,
        stop_window_waves: int = 3,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.p0_stop = p0_stop
        self.ucb_c = ucb_c
        self.n_unsettled_loci = max(0, int(n_unsettled_loci))
        self.max_novel_per_unsettled_1m = max_novel_per_unsettled_1m
        self.stop_window_waves = max(1, int(stop_window_waves))
        self.unsettled_counts: dict[IntronKey, int] = {}
        self.unsettled_total: int = 0
        self.lib_unsettled: dict[str, dict[IntronKey, int]] = {}
        self.lib_unsettled_total: dict[str, int] = {}
        self.region_novel: dict[str, int] = {}
        self.recent_unsettled_rates_1m: list[float] = []
        self.history: list[dict] = []

    def register_library(self, run_id: str, biosample_id: str, is_long_read: bool = False) -> None:
        super().register_library(run_id, biosample_id, is_long_read)
        self.lib_unsettled[run_id] = {}
        self.lib_unsettled_total[run_id] = 0

    def ingest_wave_counts(
        self,
        run_id: str,
        wave_bytes: int,
        juncs: list[WaveJunction],
        spots: int | None = None,
    ) -> dict:
        lib = self.libraries[run_id]
        lib.bytes_streamed += wave_bytes
        lib.waves_completed += 1
        novel_coding = novel_ncrna = novel_unsettled = 0
        lu = self.lib_unsettled[run_id]
        for j in juncs:
            if j.motif.upper() not in CANONICAL_MOTIFS or j.count <= 0:
                continue
            k = j.key
            lib.counts[k] = lib.counts.get(k, 0) + j.count
            lib.total_valid_spliced_obs += j.count
            if j.region == REGION_INTERGENIC:
                prev = self.global_ncrna_counts.get(k, 0)
                if prev < 2 <= prev + j.count:
                    novel_ncrna += 1
                self.global_ncrna_counts[k] = prev + j.count
            else:
                self.global_total_coding_obs += j.count
                prev = self.global_coding_counts.get(k, 0)
                if prev < 2 <= prev + j.count:
                    novel_coding += 1
                    self.region_novel[j.region] = self.region_novel.get(j.region, 0) + 1
                self.global_coding_counts[k] = prev + j.count
            if j.region == REGION_UNSETTLED:
                prev_u = self.unsettled_counts.get(k, 0)
                if prev_u < 2 <= prev_u + j.count:
                    novel_unsettled += 1
                self.unsettled_counts[k] = prev_u + j.count
                self.unsettled_total += j.count
                lu[k] = lu.get(k, 0) + j.count
                self.lib_unsettled_total[run_id] += j.count
        # Derive saturation slope deterministically from spots when available (NIC wave_bytes is telemetry only)
        if spots is not None:
            mb = max(float(spots) / 12_500.0, 0.1)
        else:
            mb = max(wave_bytes / 1e6, 0.1)
        slope = (novel_coding + novel_ncrna) / mb * 10.0
        lib.recent_novel_slopes_per_10mb.append(slope)
        self.recent_ncrna_slopes.append(novel_ncrna / mb * 10.0)
        if not lib.is_long_read and len(lib.recent_novel_slopes_per_10mb) >= self.patience_waves:
            tail = lib.recent_novel_slopes_per_10mb[-self.patience_waves :]
            if all(s < self.replicate_slope_floor for s in tail):
                lib.halted = True
                lib.halt_reason = "CONVERGED_REPLICATE_OR_SATURATED_TISSUE"
        has_short = any(not l.is_long_read for l in self.libraries.values())
        un_denom = max(1, self.n_unsettled_loci)
        if not lib.is_long_read:
            eff_units = max((spots / 1e6) if spots is not None else (wave_bytes / 8e7), 0.05)
            un_rate_1m = (novel_unsettled / un_denom) / eff_units
            self.recent_unsettled_rates_1m.append(un_rate_1m)
        elif not has_short:
            eff_units = max((spots / 5e4) if spots is not None else (wave_bytes / 8e7), 0.05)
            un_rate_1m = (novel_unsettled / un_denom) / eff_units
            self.recent_unsettled_rates_1m.append(un_rate_1m)
        else:
            un_rate_1m = self.trailing_novel_per_unsettled_1m()
        p0 = self.unsettled_p0()
        f1 = self.unsettled_f1()
        f2 = sum(1 for c in self.unsettled_counts.values() if c == 2)
        f3 = sum(1 for c in self.unsettled_counts.values() if c == 3)
        sup2 = len(self.unsettled_counts) - f1
        chao1_sup2 = (f2 * max(f2 - 1, 0)) / (2.0 * (f3 + 1))
        rec = {
            "run_id": run_id,
            "wave_bytes": wave_bytes,
            "novel_coding": novel_coding,
            "novel_unsettled": novel_unsettled,
            "novel_ncrna": novel_ncrna,
            "slope_per_10mb": slope,
            "lib_halted": lib.halted,
            "unsettled_p0": p0,
            "unsettled_f1": f1,
            "unsettled_f2": f2,
            "unsettled_f3": f3,
            "unsettled_sup2": sup2,
            "unsettled_chao1_sup2_unseen": round(chao1_sup2, 1),
            "novel_unsettled_per_locus_1m": round(un_rate_1m, 5) if math.isfinite(un_rate_1m) else None,
            "trailing_novel_per_unsettled_1m": (
                round(self.trailing_novel_per_unsettled_1m(), 5)
                if math.isfinite(self.trailing_novel_per_unsettled_1m())
                else None
            ),
            "unsettled_obs": self.unsettled_total,
            "unsettled_distinct": len(self.unsettled_counts),
            "global_coding_gt": self.global_coding_good_turing(),
        }
        self.history.append(rec)
        return rec

    def trailing_novel_per_unsettled_1m(self) -> float:
        if not self.recent_unsettled_rates_1m:
            return float("inf")
        tail = self.recent_unsettled_rates_1m[-self.stop_window_waves :]
        return sum(tail) / float(len(tail))

    def unsettled_f1(self) -> int:
        return sum(1 for c in self.unsettled_counts.values() if c == 1)

    def unsettled_p0(self) -> float:
        if self.unsettled_total <= 0:
            return 1.0
        return self.unsettled_f1() / float(self.unsettled_total)

    def lib_gt_rate(self, run_id: str) -> float:
        lu = self.lib_unsettled[run_id]
        n = self.lib_unsettled_total[run_id]
        if n <= 0:
            return 1.0
        g = self.unsettled_counts
        f1 = sum(1 for k, c in lu.items() if c == 1 and g.get(k, 0) == 1)
        return f1 / float(n)

    def gt_ucb_scores(self) -> dict[str, float]:
        T = max(1, sum(l.waves_completed for l in self.libraries.values()))
        out = {}
        for rid, lib in self.libraries.items():
            if lib.halted:
                continue
            if lib.waves_completed == 0:
                out[rid] = float("inf")
                continue
            out[rid] = self.lib_gt_rate(rid) + self.ucb_c * math.sqrt(math.log(T + 1.0) / lib.waves_completed)
        return out

    def select_next_library_gt_ucb(self) -> str | None:
        active = [l for l in self.libraries.values() if not l.halted]
        if not active:
            return None
        unprobed = [l for l in active if l.waves_completed == 0]
        if unprobed:
            unprobed.sort(key=lambda l: (not l.is_long_read, l.run_id))
            return unprobed[0].run_id
        scores = self.gt_ucb_scores()
        return max(scores, key=lambda r: scores[r]) if scores else None

    def saturated(self) -> bool:
        """All libraries probed, unsettled Good-Turing p0 < p0_stop, and per-locus novel intron rate converged."""
        if any(l.waves_completed == 0 and not l.halted for l in self.libraries.values()):
            return False
        if not (self.unsettled_total > 0 and self.unsettled_p0() < self.p0_stop):
            return False
        if self.n_unsettled_loci > 0:
            n_short = sum(1 for l in self.libraries.values() if not l.is_long_read)
            min_w = min(self.stop_window_waves, max(1, n_short or len(self.libraries)))
            if len(self.recent_unsettled_rates_1m) < min_w:
                return False
            if self.trailing_novel_per_unsettled_1m() > self.max_novel_per_unsettled_1m:
                return False
        return True
