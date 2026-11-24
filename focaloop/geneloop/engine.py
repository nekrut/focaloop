"""GeneLoop end-to-end engine: seed, three benchmark arms and the closed RNA loop.

Arm A  Tiberius + genus-stripped miniprot seed (0 RNA bytes); Tiberius standalone reported as A0.
Arm B  Full-archive batch workflow: 100 % of every panel run -> HISAT2 --dta full sorted BAMs on disk
       -> StringTie2 + miniprot + Tiberius consensus synthesis (same model rules as C, no Stage-5 zoom).
Arm C  GeneLoop: syncmer depletion sieve + Good-Turing UCB adaptive slice streaming + micro-exon zoom +
       retrocopy discriminator + Zstd Parquet state.

Anti-leakage: nothing in this module reads RefSeq annotation; eval.py is invoked only by ``evaluate``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ncrna as ncrna_mod
from . import orf_scout as orfscout_mod
from . import seed as seedmod
from .evidence import CANONICAL, BinStore, JunctionStore, LocusIndex, junction_motif
from .gffio import Transcript, introns_of, merge_intervals, read_models, write_gff3
from .intron_judge import (
    REGION_INTERGENIC,
    REGION_LOCKED_KNOWN,
    REGION_LOCKED_NOVEL,
    REGION_UNSETTLED,
    GoodTuringUCBJudge,
    WaveJunction,
)
from .locus_zoom import (
    RESCUE_CODE_VERSION,
    candidate_exonic_bridge_gaps,
    count_gap_interior_nonspliced_reads,
    merge_microexon_maps,
    microexons_from_junction_pairs,
    microexons_from_models,
    rescue_microexons_from_bam,
)
from .seed import LOCKED, UNSETTLED
from .sequtil import Genome, max_open_orf_aa
from .sieve_syncmer import LocusSaturationState, OpenSyncmerDepletionSieve, build_glcore
from .stream import (
    LocalFastqStream,
    RunStream,
    align_logan_contigs,
    align_run_full,
    align_slice_long,
    align_slice_short,
    align_slice_short_se,
    describe_run,
    detect_cross_species_rna,
    ingest_bam_to_evidence,
    ingest_per_run_junc_tsv,
    load_unw_junc_stats,
    minimap2_index_size_args,
    probe_logan_runs,
    validate_bam_header_against_genome,
)
from .synthesis import (
    LOGAN_WIDE_COHORT_MIN_RUNS,
    SynthParams,
    _softmask_frac,
    build_safe_locked_miniprot_ref_models,
    compute_eligible_lnc_introns,
    compute_logan_lnc_thresholds,
    split_safe_ortholog_chimeras,
    synthesize,
)
from .telemetry import (
    NicCounter,
    ScratchMonitor,
    StageLog,
    StageRecord,
    available_ram_bytes,
    dir_bytes,
    effective_total_ram_bytes,
    run_cmd,
    safe_env,
    samtools_sort_mem_arg,
)

RUN_ROOT = Path(os.environ.get("GENELOOP_RUN_ROOT", str(Path(tempfile.gettempdir()) / "geneloop_runs")))
SEED_LOGIC_VERSION = "v2_b0_b1_b2"
BACKFILL_LOGIC_VERSION = "v2"
_GENOME_SHA256_CACHE: dict[tuple[str, int, int, int], str] = {}


def _read_long_read_models(gtf_path: Path) -> list[Transcript]:
    """Load StringTie long-read GTF models and tag ``long_read='1'`` for downstream rescue/lncRNA gates."""
    txs = read_models(gtf_path, fmt="gtf")
    for t in txs:
        t.attrs["long_read"] = "1"
    return txs


def partition_windows_by_bp(
    windows: list[tuple[str, int, int]],
    max_chunk_bp: int = 250_000_000,
) -> list[list[tuple[str, int, int]]]:
    """Partition genomic windows into size-bounded batches (preserving order)."""
    if not windows:
        return []
    if max_chunk_bp <= 0:
        return [list(windows)]
    chunks: list[list[tuple[str, int, int]]] = []
    cur: list[tuple[str, int, int]] = []
    cur_bp = 0
    for ctg, a, b in windows:
        wlen = max(0, int(b) - int(a))
        if cur and cur_bp + wlen > max_chunk_bp:
            chunks.append(cur)
            cur = []
            cur_bp = 0
        cur.append((ctg, a, b))
        cur_bp += wlen
    if cur:
        chunks.append(cur)
    return chunks


def pack_windows_lpt(
    windows: list[tuple[str, int, int]],
    k_cfg: int = 1,
    max_chunk_bp: int = 250_000_000,
) -> tuple[list[list[tuple[str, int, int]]], float]:
    """Deterministic LPT bucket packing fixed at pack time (G-A4); returns (buckets, bp_spread)."""
    if not windows:
        return [], 1.0
    k_pack = max(1, int(k_cfg))
    total_win_bp = sum(max(0, int(b) - int(a)) for _, a, b in windows)
    if max_chunk_bp <= 0 or total_win_bp <= 0:
        return [sorted([(c, int(a), int(b)) for c, a, b in windows], key=lambda w: (w[0], w[1]))], 1.0
    n_buckets = k_pack * max(1, math.ceil(total_win_bp / (k_pack * max_chunk_bp)))
    n_buckets = min(n_buckets, len(windows))
    buckets: list[list[tuple[str, int, int]]] = [[] for _ in range(n_buckets)]
    bucket_bp = [0] * n_buckets
    ordered = sorted(windows, key=lambda w: (-(int(w[2]) - int(w[1])), str(w[0]), int(w[1])))
    for ctg, a, b in ordered:
        idx = min(range(n_buckets), key=lambda i: (bucket_bp[i], i))
        buckets[idx].append((str(ctg), int(a), int(b)))
        bucket_bp[idx] += max(0, int(b) - int(a))
    non_empty: list[list[tuple[str, int, int]]] = []
    ne_bp: list[int] = []
    for b_wins, bp in zip(buckets, bucket_bp):
        if b_wins:
            b_wins.sort(key=lambda w: (w[0], w[1]))
            non_empty.append(b_wins)
            ne_bp.append(bp)
    spread = (max(ne_bp) / max(1, min(ne_bp))) if ne_bp else 1.0
    return non_empty, round(spread, 4)


def _available_cpus() -> list[int]:
    """Return sorted CPU core IDs allowed by OS affinity (fallback to 0..cpu_count-1)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            if cpus:
                return cpus
        except OSError:
            pass
    return list(range(max(1, os.cpu_count() or 1)))


class TiberiusMemoryError(MemoryError):
    """Raised when available cgroup/scheduler/host RAM cannot fit even a single bs=1 Tiberius worker."""


def _read_mem_available_gib() -> float | None:
    """Read effective available RAM in GiB across cgroup v1/v2 hierarchy, scheduler limits, and /proc/meminfo."""
    avail_b = available_ram_bytes()
    if avail_b is None:
        return None
    return avail_b / float(1024 ** 3)


def _read_mem_total_gib() -> float | None:
    """Read effective total RAM cap in GiB across cgroup v1/v2 hierarchy, scheduler limits, and /proc/meminfo."""
    tot_b = effective_total_ram_bytes()
    if tot_b is None:
        return None
    return tot_b / float(1024 ** 3)


def _effective_reserve_gib(reserve_gib: float) -> float:
    """Use full 40 GiB reserve on >=128 GiB VMs (f1..f4); scale reserve proportionally on smaller workstations."""
    tot = _read_mem_total_gib()
    return min(reserve_gib, round(0.15 * tot, 2)) if (tot is not None and tot < 128.0) else reserve_gib


def _cpu_bs_rss_ladder(model_cfg: str) -> list[tuple[int, float]]:
    """Ordered (batch_size, estimated_rss_gib) step-down ladder down to bs=1 for Tiberius CPU inference."""
    if "mammalia" in model_cfg:
        return [(2, 62.0), (1, 34.0)]
    if "angiosperms" in model_cfg:
        return [(2, 19.0), (1, 12.0)]
    return [(4, 28.0), (2, 19.0), (1, 12.0)]


def _default_cpu_bs_and_rss(model_cfg: str) -> tuple[int, float]:
    """Default approved CPU batch size and estimated per-worker RSS (GiB) by Tiberius model config."""
    return _cpu_bs_rss_ladder(model_cfg)[0]


def _approve_wave_workers_and_bs(
    k_cfg: int,
    model_cfg: str,
    reserve_gib: float = 40.0,
) -> tuple[int, int, bool, float | None]:
    """Reserve-honoring cgroup-aware RAM gate: steps down to bs=1 before raising TiberiusMemoryError."""
    ladder = _cpu_bs_rss_ladder(model_cfg)
    approved_bs, rss_est = ladder[0]
    avail = _read_mem_available_gib()
    k_req = max(1, int(k_cfg))
    if avail is None:
        return k_req, approved_bs, False, None
    k_eff = min(k_req, math.floor((avail - reserve_gib) / rss_est))
    bs_stepdown = False
    for cand_bs, cand_rss in ladder[1:]:
        if k_eff > 0:
            break
        approved_bs = cand_bs
        rss_est = cand_rss
        bs_stepdown = True
        k_eff = min(k_req, math.floor((avail - reserve_gib) / rss_est))
    if k_eff <= 0:
        raise TiberiusMemoryError(
            f"Insufficient available RAM ({avail:.2f} GiB) for Tiberius CPU worker "
            f"(reserve={reserve_gib:.1f} GiB, rss_est={rss_est:.1f} GiB, bs={approved_bs})"
        )
    return k_eff, approved_bs, bs_stepdown, avail


class _MemAvailableSampler:
    """Background 30s /proc/meminfo MemAvailable sampler recording min_gib across Tiberius execution."""

    def __init__(self, interval_s: float = 30.0):
        self.interval_s = interval_s
        self.min_gib: float | None = None
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    def _sample_once(self) -> None:
        v = _read_mem_available_gib()
        if v is not None and (self.min_gib is None or v < self.min_gib):
            self.min_gib = v

    def __enter__(self) -> "_MemAvailableSampler":
        self._sample_once()
        if self.min_gib is not None:
            def _loop() -> None:
                while not self._stop.wait(self.interval_s):
                    self._sample_once()

            self._thr = threading.Thread(target=_loop, daemon=True)
            self._thr.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)
        self._sample_once()


@dataclass
class LoopParams:
    seed_mode: str = "targeted"  # "targeted" (DIAMOND + windowed miniprot) or "full" (whole-genome miniprot)
    wave_spots: int = 1_000_000
    long_wave_spots: int = 50_000
    p0_stop: float = 0.005
    ucb_c: float = 0.005
    max_novel_per_unsettled_1m: float = 0.12
    stop_window_waves: int = 3
    max_frac_spots: float = 0.35
    replicate_slope_floor_per_10mb: float = 2.0
    patience_waves: int = 2
    min_junction_depth: int = 20
    min_exon_depth: float = 20.0
    max_singleton_ratio: float = 0.01
    paralog_multimap_frac: float = 0.5
    caps: tuple = (40, 120, 40)  # per 1-kb bin: unspliced, spliced, soft-clipped primary short reads retained
    long_caps: tuple = (80, 240, 80)  # separate per 1-kb bin caps for long-read libraries (never starved by short reads)
    hotspot_frac: float = 0.002
    sieve_min_hits: int = 2
    sieve_max_miss: int = 2
    extra_bait_fasta: str = ""
    threads: int = 24
    hisat_threads: int = 20
    cpu_only_tiberius: bool = False
    cpu_focaloop_tiberius: bool = False
    orphan_orf_scout: bool = False
    auto_orf_scout_thresh: float | None = None
    tiberius_chunk_bp: int = 250_000_000
    tiberius_workers: int = 1
    tiberius_cpuset_list: tuple[str, ...] = ()
    tiberius_spacer_bp: int = 4_500
    zoom_preset: str = "default"
    tiberius_extra_env: dict[str, str] = field(default_factory=dict)
    miniprot_max_shard_bp: int = 700_000_000
    miniprot_outs: float | None = None
    miniprot_outn: int | None = None
    overlap_hisat_index_in_seed: bool = False
    emit_nozoom_ablation: bool = False
    auto_orf_scout_max_proteins: int = 80_000
    scout_spots: int = 2_000_000
    scout_max_runs: int = 4
    trim_spacer_bridges: bool = True
    enable_orphan_backfill: bool = True
    max_backfill_bp: int = 15_000_000
    tiberius_reserve_gib: float = 40.0
    enable_structural_ncrna: str = "auto"
    logan_mode: str = "auto"
    long_stranded: bool = False


class GateRun:
    def __init__(self, gate_id: str, root: Path = RUN_ROOT, loop: LoopParams | None = None,
                 synth: SynthParams | None = None, manifest_path: str | None = None,
                 gate: dict | None = None):
        self.gate_id = gate_id
        self.gate = dict(gate) if gate is not None else seedmod.gate_config(gate_id, seedmod.load_manifest(manifest_path))
        self.dir = root / gate_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = StageLog(self.dir / "stages.jsonl")
        self.lp = loop or LoopParams()
        self.sp = synth or SynthParams()
        self.nic = NicCounter()
        self._genome: Genome | None = None
        self._scout_streams: dict[str, RunStream] = {}
        self._local_streams: dict[str, dict] = {}
        self._local_bams: list[dict] = []
        self._skip_tiberius: bool = False
        self._enable_zoom: bool = True
        self._logan_partition_cache: dict[tuple[tuple[str, ...], str], tuple[list[str], list[str], dict]] = {}
        self.meta: dict = {}
        self.c_tag = "C"  # Arm C variant tag -> working dir arm<tag>, models arm<tag>.gff3

    def close(self) -> None:
        for st in self._scout_streams.values():
            try:
                st.close()
            except Exception:
                pass
        self._scout_streams.clear()

    # ------------------------------------------------------------------ shared
    @property
    def genome_fa(self) -> Path:
        return self.dir / "genome" / "genome.fna"

    @property
    def genome(self) -> Genome:
        if self._genome is None:
            self._genome = Genome(self.genome_fa)
        return self._genome

    def _done(self, name: str) -> Path:
        return self.dir / f".done_{name}"

    def stage_rec(self, name: str) -> dict | None:
        return self.log.latest(name)

    def _seed_protein_path(self) -> Path:
        return seedmod.seed_protein_path(self.dir / "seed", self.gate)

    def _is_long_run_stranded(self, run_id: str) -> bool:
        if bool(getattr(self.lp, "long_stranded", False)):
            return True
        loc = self._local_streams.get(run_id)
        if loc and bool(loc.get("stranded", False)):
            return True
        stranded_runs = {str(r) for r in (self.gate.get("sra_long_stranded_runs") or ())}
        return str(run_id) in stranded_runs

    def prepare(self) -> dict:
        self.meta = seedmod.prepare_genome(self.gate_id, self.dir, self.log, gate=self.gate)
        self.audit = seedmod.leakage_audit(self.gate, self.dir, self.log)
        return self.meta

    def _run_tiberius_with_fallback(
        self,
        genome_fa: Path,
        out_dir: Path,
        cpu_only: bool,
        tag_prefix: str = "",
        cpu_threads: int | None = None,
        cpuset_cpus: str | None = None,
        extra_env: dict[str, str] | None = None,
        approved_bs: int | None = None,
        bs_stepdown: bool = False,
        effective_reserve_gib: float | None = None,
    ) -> tuple[str, StageRecord]:
        cfg = self.gate.get("tiberius_model_cfg", "")
        c_threads = cpu_threads if cpu_threads is not None else self.lp.threads
        eff_env = self.lp.tiberius_extra_env or None
        if extra_env:
            eff_env = {**(eff_env or {}), **extra_env}
        if cpu_only:
            default_bs, _ = _default_cpu_bs_and_rss(cfg)
            cap_bs = approved_bs if approved_bs is not None else default_bs
            eff_res = (
                effective_reserve_gib
                if effective_reserve_gib is not None
                else _effective_reserve_gib(self.lp.tiberius_reserve_gib)
            )
            full_order = [2, 1] if ("angiosperms" in cfg or "mammalia" in cfg) else [4, 2, 1]
            bs_order = [b for b in full_order if b <= cap_bs] or [1]
            for bs in bs_order:
                tag = f"{tag_prefix}cpu_bs{bs}"
                step_flag = bool(bs_stepdown or bs < cap_bs)
                meta = {
                    "approved_bs": cap_bs,
                    "bs_stepdown": step_flag,
                    "effective_reserve_gib": eff_res,
                }
                try:
                    rec = seedmod.run_tiberius(
                        self.gate, genome_fa, out_dir, self.log,
                        batch_size=bs, tag=tag, cpu_only=True, cpu_threads=c_threads,
                        cpuset_cpus=cpuset_cpus, extra_env=eff_env, extra_meta=meta,
                    )
                    rec.extra = {**(rec.extra or {}), **meta}
                    return tag, rec
                except RuntimeError:
                    if bs == bs_order[-1]:
                        raise
        for bs in (6, 4):
            tag = f"{tag_prefix}bs{bs}"
            if (out_dir / f"tiberius_{tag}.done.json").exists() and (out_dir / f"tiberius_{tag}.gtf").exists():
                rec = seedmod.run_tiberius(self.gate, genome_fa, out_dir, self.log, batch_size=bs, tag=tag)
                return tag, rec
        for bs in (6, 4):
            tag = f"{tag_prefix}bs{bs}"
            try:
                rec = seedmod.run_tiberius(self.gate, genome_fa, out_dir, self.log, batch_size=bs, tag=tag)
                return tag, rec
            except RuntimeError:
                if bs == 4:
                    raise
        raise RuntimeError("unreachable")

    def _worker_cpu_layout(self, n_workers: int, total_cores: int) -> tuple[int, list[str | None]]:
        if n_workers <= 1:
            return total_cores, [None]
        if self.lp.tiberius_cpuset_list:
            cpusets = [self.lp.tiberius_cpuset_list[i % len(self.lp.tiberius_cpuset_list)] for i in range(n_workers)]
            first = cpusets[0]
            try:
                if "-" in first and "," not in first:
                    lo, hi = (int(x) for x in first.split("-", 1))
                    per_threads = max(1, hi - lo + 1)
                else:
                    per_threads = max(1, len(first.split(",")))
            except Exception:
                per_threads = max(1, total_cores // n_workers)
            return per_threads, cpusets
        avail_pool = _available_cpus()
        usable_cpus = avail_pool[: max(n_workers, min(len(avail_pool), total_cores))]
        if len(usable_cpus) < n_workers:
            per_threads = 1
            return per_threads, [str(usable_cpus[i % max(1, len(usable_cpus))]) for i in range(n_workers)]
        per_threads = max(1, len(usable_cpus) // n_workers)
        cpusets: list[str | None] = []
        for wi in range(n_workers):
            sl = usable_cpus[wi * per_threads : (wi + 1) * per_threads]
            if len(sl) > 1 and sl == list(range(sl[0], sl[-1] + 1)):
                cpusets.append(f"{sl[0]}-{sl[-1]}")
            else:
                cpusets.append(",".join(str(c) for c in sl))
        return per_threads, cpusets

    def _run_tiberius_windows_chunked(
        self,
        windows: list[tuple[str, int, int]],
        work_dir: Path,
        tib_dir: Path,
        prefix: str,
        out_gtf: Path,
        out_faa: Path,
        cpu_only: bool = True,
        workers_override: int | None = None,
    ) -> tuple[StageRecord, list[dict]]:
        """Run Tiberius across LPT-packed window buckets with per-wave RAM gating and sequential B1/B2 lifting."""
        if not windows:
            out_gtf.write_text("")
            out_faa.write_text("")
            return StageRecord(name=f"tiberius_{prefix}empty"), []
        spacer_bp = int(getattr(self.lp, "tiberius_spacer_bp", 4_500))
        n_workers_cfg = max(1, int(workers_override if workers_override is not None else getattr(self.lp, "tiberius_workers", 1)))
        chunks, bp_spread = pack_windows_lpt(windows, n_workers_cfg, self.lp.tiberius_chunk_bp)
        cfg = self.gate.get("tiberius_model_cfg", "")
        g = self.genome
        fb_fn = self._run_tiberius_with_fallback
        supports_approved_bs = getattr(fb_fn, "__func__", None) is GateRun._run_tiberius_with_fallback
        eff_reserve = _effective_reserve_gib(self.lp.tiberius_reserve_gib)
        if len(chunks) == 1:
            man_p = work_dir / f"{prefix}manifest.json"
            super_fa = work_dir / f"{prefix}supercontigs.fa"
            manifest = seedmod.pack_windows_to_supercontigs(g, chunks[0], super_fa, man_p, spacer_bp=spacer_bp)
            approved_bs = None
            bs_stepdown = False
            if cpu_only and supports_approved_bs:
                _keff, approved_bs, bs_stepdown, _avail = _approve_wave_workers_and_bs(1, cfg, eff_reserve)
            with _MemAvailableSampler() as mem_mon:
                fb_kw = (
                    {"approved_bs": approved_bs, "bs_stepdown": bs_stepdown, "effective_reserve_gib": eff_reserve}
                    if supports_approved_bs
                    else {}
                )
                tag, rec = fb_fn(super_fa, tib_dir, cpu_only=cpu_only, tag_prefix=prefix, **fb_kw)
            lift_info = seedmod.lift_tiberius_gtf(
                tib_dir / f"tiberius_{tag}.gtf", manifest, out_gtf,
                genome=g, out_faa=out_faa, trim_spacer_bridges=self.lp.trim_spacer_bridges,
            )
            super_fa.unlink(missing_ok=True)
            min_mem_gib = round(mem_mon.min_gib, 2) if mem_mon.min_gib is not None else None
            chunk_stats = [{
                "chunk": 0,
                "tag": tag,
                "n_windows": len(chunks[0]),
                "bp": sum(b - a for _, a, b in chunks[0]),
                "wall_s": rec.wall_seconds,
                "n_trimmed_spacer_tx": lift_info.get("n_trimmed_spacer_tx", 0),
                "n_discarded_spacer_tx": lift_info.get("n_discarded_spacer_tx", 0),
            }]
            rec.extra = {
                **(rec.extra or {}),
                "bp_spread": bp_spread,
                "effective_reserve_gib": eff_reserve,
                "min_mem_available_gib": min_mem_gib,
                "n_trimmed_spacer_tx": lift_info.get("n_trimmed_spacer_tx", 0),
                "n_discarded_spacer_tx": lift_info.get("n_discarded_spacer_tx", 0),
                "chunks": chunk_stats,
            }
            self.log.append(
                StageRecord(
                    name=f"tiberius_{prefix}chunked",
                    wall_seconds=rec.wall_seconds,
                    extra={
                        **(rec.extra or {}),
                        "n_chunks": 1,
                        "workers": 1,
                        "sum_chunk_wall_s": round(rec.wall_seconds, 3),
                        "sum_chunk_cpu_s": round(rec.cpu_seconds, 3),
                    },
                )
            )
            return rec, chunk_stats

        # Pre-pack fixed LPT buckets sequentially (never repacked mid-run).
        packed_chunks: list[tuple[int, list[tuple[str, int, int]], Path, Path, Path, dict]] = []
        for ci, cwins in enumerate(chunks):
            c_pfx = f"{prefix}c{ci}_"
            c_dir = tib_dir / f"chunk_{ci}"
            c_fa = work_dir / f"{c_pfx}supercontigs.fa"
            c_man_p = work_dir / f"{c_pfx}manifest.json"
            c_gtf = work_dir / f"{c_pfx}lifted.gtf"
            manifest = seedmod.pack_windows_to_supercontigs(g, cwins, c_fa, c_man_p, spacer_bp=spacer_bp)
            packed_chunks.append((ci, cwins, c_dir, c_fa, c_gtf, manifest))

        t_start = time.time()
        raw_by_ci: dict[int, tuple[str, StageRecord, str | None]] = {}
        any_stepdown = False
        max_k_used = 1
        with _MemAvailableSampler() as mem_mon:
            cursor = 0
            while cursor < len(packed_chunks):
                rem = len(packed_chunks) - cursor
                w_req = min(n_workers_cfg, rem)
                if cpu_only and supports_approved_bs:
                    k_eff, approved_bs, stepdown_w, _avail = _approve_wave_workers_and_bs(w_req, cfg, eff_reserve)
                    any_stepdown = any_stepdown or stepdown_w
                else:
                    k_eff, approved_bs, stepdown_w = w_req, None, False
                max_k_used = max(max_k_used, k_eff)
                batch = packed_chunks[cursor : cursor + k_eff]
                cursor += len(batch)
                if k_eff <= 1 and not self.lp.tiberius_cpuset_list and not self.lp.tiberius_extra_env:
                    ci, _cwins, c_dir, c_fa, _c_gtf, _manifest = batch[0]
                    c_pfx = f"{prefix}c{ci}_"
                    fb_kw = (
                        {"approved_bs": approved_bs, "bs_stepdown": stepdown_w, "effective_reserve_gib": eff_reserve}
                        if supports_approved_bs
                        else {}
                    )
                    tag, rec = fb_fn(c_fa, c_dir, cpu_only=cpu_only, tag_prefix=c_pfx, **fb_kw)
                    c_fa.unlink(missing_ok=True)
                    raw_by_ci[ci] = (tag, rec, None)
                else:
                    per_threads, cpusets = self._worker_cpu_layout(k_eff, self.lp.threads)

                    def _run_one(
                        slot: int,
                        item: tuple[int, list[tuple[str, int, int]], Path, Path, Path, dict],
                        _pt: int = per_threads,
                        _cs: list[str | None] = cpusets,
                        _bs: int | None = approved_bs,
                        _sd: bool = stepdown_w,
                    ) -> None:
                        ci, _cwins, c_dir, c_fa, _c_gtf, _manifest = item
                        c_pfx = f"{prefix}c{ci}_"
                        fb_kw = (
                            {"approved_bs": _bs, "bs_stepdown": _sd, "effective_reserve_gib": eff_reserve}
                            if supports_approved_bs
                            else {}
                        )
                        tag, rec = fb_fn(
                            c_fa, c_dir, cpu_only=cpu_only, tag_prefix=c_pfx,
                            cpu_threads=_pt, cpuset_cpus=_cs[slot], **fb_kw,
                        )
                        c_fa.unlink(missing_ok=True)
                        raw_by_ci[ci] = (tag, rec, _cs[slot])

                    import queue

                    slot_q: queue.Queue[int] = queue.Queue()
                    for wi in range(k_eff):
                        slot_q.put(wi)

                    def _pooled(item, _sq=slot_q, _fn=_run_one):
                        slot = _sq.get()
                        try:
                            _fn(slot, item)
                        finally:
                            _sq.put(slot)

                    with ThreadPoolExecutor(max_workers=k_eff) as ex:
                        list(ex.map(_pooled, batch))

        # R3: Lift sequentially on main thread after workers join, always passing genome=g!
        results_by_ci: dict[int, tuple[StageRecord, dict, dict]] = {}
        for ci, cwins, c_dir, _c_fa, c_gtf, manifest in packed_chunks:
            tag, rec, cset = raw_by_ci[ci]
            lift_info = seedmod.lift_tiberius_gtf(
                c_dir / f"tiberius_{tag}.gtf", manifest, c_gtf,
                genome=g, trim_spacer_bridges=self.lp.trim_spacer_bridges,
            )
            any_stepdown = any_stepdown or bool((rec.extra or {}).get("bs_stepdown", False))
            cstat = {
                "chunk": ci,
                "tag": tag,
                "n_windows": len(cwins),
                "bp": sum(b - a for _, a, b in cwins),
                "wall_s": round(rec.wall_seconds, 3),
                "n_trimmed_spacer_tx": lift_info.get("n_trimmed_spacer_tx", 0),
                "n_discarded_spacer_tx": lift_info.get("n_discarded_spacer_tx", 0),
            }
            if cset is not None:
                cstat["cpuset"] = cset
            results_by_ci[ci] = (rec, lift_info, cstat)

        chunk_gtfs = [packed_chunks[ci][4] for ci in range(len(packed_chunks))]
        chunk_stats = [results_by_ci[ci][2] for ci in range(len(packed_chunks))]
        sum_chunk_wall = sum(results_by_ci[ci][0].wall_seconds for ci in range(len(packed_chunks)))
        tot_user = sum(results_by_ci[ci][0].cpu_user for ci in range(len(packed_chunks)))
        tot_sys = sum(results_by_ci[ci][0].cpu_sys for ci in range(len(packed_chunks)))
        n_trimmed_spacer = sum(results_by_ci[ci][1].get("n_trimmed_spacer_tx", 0) for ci in range(len(packed_chunks)))
        n_discarded_spacer = sum(results_by_ci[ci][1].get("n_discarded_spacer_tx", 0) for ci in range(len(packed_chunks)))
        tot_wall = time.time() - t_start
        seedmod.merge_tiberius_gtfs(chunk_gtfs, out_gtf, genome=g, out_faa=out_faa)
        agg = StageRecord(
            name=f"tiberius_{prefix}chunked",
            wall_seconds=tot_wall,
            cpu_user=tot_user,
            cpu_sys=tot_sys,
            extra={
                "n_chunks": len(chunks),
                "workers": max_k_used,
                "bp_spread": bp_spread,
                "bs_stepdown": any_stepdown,
                "effective_reserve_gib": eff_reserve,
                "min_mem_available_gib": round(mem_mon.min_gib, 2) if mem_mon.min_gib is not None else None,
                "sum_chunk_wall_s": round(sum_chunk_wall, 3),
                "sum_chunk_cpu_s": round(tot_user + tot_sys, 3),
                "n_trimmed_spacer_tx": n_trimmed_spacer,
                "n_discarded_spacer_tx": n_discarded_spacer,
                "chunks": chunk_stats,
            },
        )
        # Y2: Persist chunk-aggregate telemetry in stages.jsonl without double-counting per-chunk CPU seconds.
        self.log.append(StageRecord(name=agg.name, wall_seconds=agg.wall_seconds, extra=agg.extra))
        return agg, chunk_stats

    def _bait_spans_and_fa(self, sd: Path, hits: list, tib: list = (), out_fa: Path | None = None) -> tuple[list[tuple[str, int, int]], set[str]]:
        meta = json.loads((self.dir / "genome" / "genome_meta.json").read_text())
        organelle_set = set(meta.get("organelle_contigs", []))
        spans_p = sd / "bait_spans.json"
        if spans_p.exists():
            rrna_spans = [tuple(x) for x in json.loads(spans_p.read_text())]
        else:
            coding_by_ctg: dict[str, list[tuple[int, int]]] = {}
            for t in tib:
                for a, b in t.cds:
                    coding_by_ctg.setdefault(t.contig, []).append((max(0, a - 200), b + 200))
            for h in hits:
                for a, b in h.cds:
                    coding_by_ctg.setdefault(h.contig, []).append((max(0, a - 200), b + 200))
            rrna_spans = _discover_rrna_bait_intervals(
                self.genome, self.genome_fa, coding_by_ctg, organelle_set, threads=min(8, self.lp.threads)
            )
            spans_p.write_text(json.dumps([list(x) for x in rrna_spans]))
        if out_fa is not None and not out_fa.exists():
            out_fa.parent.mkdir(parents=True, exist_ok=True)
            rrna_core = Path(__file__).parent / "data" / "euk_rrna_core.fa"
            extra_bait = Path(self.lp.extra_bait_fasta) if self.lp.extra_bait_fasta else (
                Path(self.gate["bait_fasta"]) if self.gate.get("bait_fasta") else None
            )
            with open(out_fa, "w") as fh:
                for c in meta.get("organelle_contigs", []):
                    fh.write(f">{c}\n{self.genome.fetch(c, 0, self.genome.lengths[c])}\n")
                for idx_r, (ctg, a, b) in enumerate(rrna_spans):
                    fh.write(f">rrna_asm_{idx_r}_{ctg}_{a}_{b}\n{self.genome.fetch(ctg, a, b)}\n")
                if rrna_core.exists():
                    fh.write(rrna_core.read_text().rstrip("\n") + "\n")
                if extra_bait and extra_bait.exists():
                    fh.write(extra_bait.read_text().rstrip("\n") + "\n")
        return rrna_spans, organelle_set

    def partition_short_runs(self) -> tuple[list[str], list[str], dict]:
        """Partition short runs into HTTP-200 ``logan_runs`` and ``fallback_short_runs`` (cached in ``seed/logan_probe.json``)."""
        mode = self.lp.logan_mode
        if mode not in ("auto", "logan_only", "off"):
            raise ValueError(f"Invalid logan_mode={mode!r}; expected 'auto', 'logan_only', or 'off'")
        sra_short = [str(r) for r in self.gate.get("sra_short_runs", [])]
        local_short = [rid for rid, info in self._local_streams.items() if not info.get("long_read", False)]
        ck = (tuple(sra_short), tuple(local_short), mode)
        if ck in self._logan_partition_cache:
            lr, fr, pr = self._logan_partition_cache[ck]
            return list(lr), list(fr), dict(pr)
        if mode == "off" or not sra_short:
            fb_all = list(sra_short) + list(local_short)
            info = {
                "logan_mode": mode,
                "requested_short_runs": list(sra_short),
                "logan_runs": [],
                "fallback_short_runs": fb_all,
                "skipped_short_runs": [],
                "local_short_runs": list(local_short),
                "total_compressed_bytes": 0,
            }
            self._logan_partition_cache[ck] = ([], fb_all, info)
            return [], fb_all, dict(info)
        probe_p = self.dir / "seed" / "logan_probe.json"
        probe_res = probe_logan_runs(sra_short, cache_path=probe_p, timeout=8.0, retries=2)
        logan_runs = list(probe_res.get("logan_runs", []))
        logan_set = set(logan_runs)
        skipped_short_runs: list[str] = []
        if mode == "logan_only":
            transient_failed = [str(r) for r in (probe_res.get("transient_failed_runs") or [])]
            if transient_failed:
                raise RuntimeError(
                    f"Logan-only probe failed transiently on {len(transient_failed)} run(s) "
                    f"({', '.join(transient_failed[:5])}). Retry or switch --logan-mode to auto."
                )
            if not logan_runs:
                raise ValueError(
                    "Logan-only mode found no pre-assembled Logan contig archives for the requested SRA accessions. "
                    "Switch Logan mode to Automatic to fall back to raw SRA/ENA streaming."
                )
            skipped_short_runs = [r for r in sra_short if r not in logan_set]
            fb_runs = list(local_short)
        else:
            fb_runs = list(probe_res.get("fallback_short_runs", [])) + list(local_short)
        info = {
            "logan_mode": mode,
            **probe_res,
            "fallback_short_runs": fb_runs,
            "skipped_short_runs": skipped_short_runs,
            "local_short_runs": list(local_short),
        }
        self._logan_partition_cache[ck] = (logan_runs, fb_runs, info)
        return list(logan_runs), list(fb_runs), dict(info)

    def _run_scout_rna(self, sd: Path, hits: list) -> dict:
        scout_json = sd / "scout_waves.json"
        logan_runs, fb_short_runs, _probe = self.partition_short_runs()
        eligible_short = fb_short_runs if self.lp.logan_mode in ("auto", "logan_only") else [r for r, lr in self.runs() if not lr]
        short_runs = eligible_short[:self.lp.scout_max_runs]
        if scout_json.exists():
            try:
                cached_sw = json.loads(scout_json.read_text())
                if isinstance(cached_sw, dict):
                    if "scout_short_runs" in cached_sw:
                        if list(cached_sw["scout_short_runs"]) == short_runs:
                            return cached_sw
                    elif not logan_runs:
                        return cached_sw
            except (OSError, ValueError):
                pass
        bait_fa = sd / "scout_bait.fa"
        rrna_spans, organelle_set = self._bait_spans_and_fa(sd, hits, out_fa=bait_fa)
        if not short_runs or self.lp.scout_spots <= 0:
            data = {
                "waves": [], "spots_total": 0, "wire_bytes": 0, "wall_s": 0.0,
                "rrna_spans": [list(x) for x in rrna_spans],
                "organelle_contigs": sorted(organelle_set),
                "bait_fa": str(bait_fa), "cap_state": "",
                "scout_short_runs": list(short_runs),
                "logan_mode": self.lp.logan_mode,
                "logan_runs_excluded": len(logan_runs) if self.lp.logan_mode in ("auto", "logan_only") else 0,
            }
            _write_json_atomic(scout_json, data)
            return data
        glc = build_glcore()
        idx = self.hisat_index(force=True)
        info = self.run_info(runs=[(r, False) for r in short_runs])
        scout_dir = sd / "scout_rna"
        scout_dir.mkdir(parents=True, exist_ok=True)
        scout_active = scout_dir / "scout_active.bin"
        _activate(glc, scout_active, bait=bait_fa)
        cap_state = scout_dir / "scout_cap_state.u16"
        t0 = time.time()
        rx0 = self.nic.rx()
        n_sel = len(short_runs)
        base_per_run = self.lp.scout_spots // n_sel
        rem = self.lp.scout_spots % n_sel
        waves: list[dict] = []
        spots_total = 0
        has_remote_scout = any(r not in self._local_streams for r in short_runs)
        for i_r, rid in enumerate(short_runs):
            target_spots = base_per_run + (1 if i_r < rem else 0)
            if target_spots <= 0:
                continue
            is_local_r = rid in self._local_streams
            st = self._scout_streams.get(rid)
            if st is None:
                if is_local_r:
                    st = self._local_streams[rid]["stream"]
                else:
                    st = RunStream(rid, info[rid]["n_segments"], info[rid]["total_spots"], scout_dir / "logs", threads=4)
                self._scout_streams[rid] = st
            rem_r = target_spots
            w_idx = 0
            while rem_r > 0:
                slice_n = min(self.lp.wave_spots, rem_r)
                tag = f"scout_{i_r}_{w_idx}_{rid}"
                use_nic = None if is_local_r else self.nic
                try:
                    if getattr(st, "segs", 2) == 1:
                        res = align_slice_short_se(
                            st, slice_n, idx, scout_dir / "waves", tag, cap_state, self.lp.caps,
                            threads=self.lp.hisat_threads, nic=use_nic,
                        )
                    else:
                        res = align_slice_short(
                            st, slice_n, idx, scout_dir / "waves", tag, scout_active, cap_state, self.lp.caps,
                            threads=self.lp.hisat_threads, nic=use_nic,
                            min_hits=self.lp.sieve_min_hits, max_miss=self.lp.sieve_max_miss,
                        )
                except Exception:
                    if getattr(st, "stream_failed", False):
                        break
                    raise
                if res.spots == 0:
                    break
                waves.append({
                    "run_id": rid,
                    "long_read": False,
                    "spots": res.spots,
                    "wire_bytes": res.wire_bytes,
                    "wall_s": res.wall,
                    "cpu": res.cpu,
                    "sieve": res.sieve,
                    "junc": res.junc,
                    "bam": res.bam,
                    "junc_tsv": res.junc_tsv,
                    "bins_tsv": res.bins_tsv,
                    "records_consumed": st.records_consumed,
                })
                spots_total += res.spots
                rem_r -= res.spots
                w_idx += 1
        wall_s = time.time() - t0
        wire_bytes = max(0, self.nic.rx() - rx0) if has_remote_scout else 0
        self.log.append(StageRecord(
            name="scout_rna", wall_seconds=wall_s,
            cpu_user=sum(sum(w["cpu"].values()) for w in waves),
            wire_bytes=wire_bytes, extra={"spots": spots_total, "waves": len(waves), "runs": short_runs},
        ))
        data = {
            "waves": waves,
            "spots_total": spots_total,
            "wire_bytes": wire_bytes,
            "wall_s": wall_s,
            "rrna_spans": [list(x) for x in rrna_spans],
            "organelle_contigs": sorted(organelle_set),
            "bait_fa": str(bait_fa),
            "cap_state": str(cap_state) if cap_state.exists() else "",
            "scout_short_runs": list(short_runs),
            "logan_mode": self.lp.logan_mode,
            "logan_runs_excluded": len(logan_runs) if self.lp.logan_mode in ("auto", "logan_only") else 0,
        }
        _write_json_atomic(scout_json, data)
        return data

    def _raw_super_gtf_digests(self, sd: Path) -> list[list[str]]:
        out: list[list[str]] = []
        for sub in ("tib_zoom", "tib_orf_delta"):
            d = sd / sub
            if not d.is_dir():
                continue
            for gp in sorted(d.rglob("tiberius_*.gtf")):
                out.append([str(gp.relative_to(sd)), _sha256_file(gp)])
        if not out:
            for fn in ("tiberius_zoomed.gtf", "tiberius_inverted.gtf"):
                fp = sd / fn
                if fp.exists():
                    out.append([fn, _sha256_file(fp)])
        return out

    def _genome_identity_digest(self) -> dict:
        """Exact full-file genome SHA256 identity (X4/Z1), cached by (path, mtime_ns, size) for >64 MiB assemblies."""
        fna = self.genome_fa
        fai = Path(f"{fna}.fai")
        meta = self.dir / "genome" / "genome_meta.json"
        meta_obj: dict = {}
        if meta.exists():
            try:
                loaded_m = json.loads(meta.read_text())
                if isinstance(loaded_m, dict):
                    meta_obj = loaded_m
            except (OSError, ValueError):
                meta_obj = {}
        org_list = sorted(str(c) for c in (meta_obj.get("organelle_contigs") or ()))
        if not fna.exists():
            return {
                "st_size": 0,
                "fna_sha256": "",
                "fai_sha256": _sha256_file(fai) if fai.exists() else "",
                "organelle_contigs": org_list,
            }
        st = fna.stat()
        sz = int(st.st_size)
        res_path = str(fna.resolve())
        cache_tuple = (res_path, int(st.st_mtime_ns), sz, int(st.st_ino))
        fna_sha = ""
        if sz > 64 * 1024 * 1024:
            fna_sha = _GENOME_SHA256_CACHE.get(cache_tuple, "")
            if not fna_sha:
                fc = meta_obj.get("fna_cache")
                if (
                    isinstance(fc, dict)
                    and fc.get("path") == res_path
                    and fc.get("mtime_ns") == int(st.st_mtime_ns)
                    and fc.get("size") == sz
                    and isinstance(fc.get("sha256"), str)
                    and len(fc["sha256"]) == 64
                ):
                    fna_sha = fc["sha256"]
                    _GENOME_SHA256_CACHE[cache_tuple] = fna_sha
        if not fna_sha:
            fna_sha = _sha256_file(fna)
            _GENOME_SHA256_CACHE[cache_tuple] = fna_sha
            if sz > 64 * 1024 * 1024 and meta.exists():
                try:
                    updated_meta = {
                        **meta_obj,
                        "fna_cache": {
                            "path": res_path,
                            "mtime_ns": int(st.st_mtime_ns),
                            "size": sz,
                            "sha256": fna_sha,
                        },
                    }
                    _write_json_atomic(meta, updated_meta)
                except OSError:
                    pass
        return {
            "st_size": sz,
            "fna_sha256": fna_sha,
            "fai_sha256": _sha256_file(fai) if fai.exists() else "",
            "organelle_contigs": org_list,
        }

    def _compute_seed_cache_key(self, sd: Path, b1_b2_applied: bool = True) -> str:
        eff_spacer = int(getattr(self.lp, "tiberius_spacer_bp", 4_500))
        eff_merge_gap = 5_000 if self.lp.zoom_preset in ("tight_v1", "tight_v2") else 10_000
        mp_p = sd / "miniprot.gff3"
        sw_p = sd / "scout_waves.json"
        payload = {
            "seed_logic_version": SEED_LOGIC_VERSION,
            "b1_b2_applied": bool(b1_b2_applied),
            "genome_identity": self._genome_identity_digest(),
            "tiberius_model_cfg": str(self.gate.get("tiberius_model_cfg", "")),
            "tiberius_image": seedmod.TIBERIUS_IMAGE,
            "tiberius_image_id": seedmod._tiberius_image_id(),
            "tiberius_chunk_bp": int(self.lp.tiberius_chunk_bp),
            "tiberius_workers": max(1, int(self.lp.tiberius_workers)),
            "zoom_preset": self.lp.zoom_preset,
            "spacer_bp": eff_spacer,
            "merge_gap": eff_merge_gap,
            "trim_spacer_bridges": bool(self.lp.trim_spacer_bridges),
            "orphan_orf_scout": bool(self.lp.orphan_orf_scout),
            "auto_orf_scout_thresh": self.lp.auto_orf_scout_thresh,
            "miniprot_gff3_sha256": _sha256_file(mp_p) if mp_p.exists() else "",
            "scout_waves_sha256": _sha256_file(sw_p) if sw_p.exists() else "",
            "super_gtf_sha256s": self._raw_super_gtf_digests(sd),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _select_verified_raw_tiberius_gtf(
        self,
        tib_dir: Path,
        tag_prefix: str,
        manifest: dict,
        preferred_tag: str | None = None,
    ) -> tuple[Path, str, dict] | None:
        """X1: Select only a raw Tiberius GTF whose sibling done.json succeeded (rc==0) and matches supercontig input_sha256."""
        if not tib_dir.is_dir():
            return None
        raw_gtfs = sorted(tib_dir.glob(f"tiberius_{tag_prefix}*.gtf"))
        if not raw_gtfs:
            return None
        pref = (preferred_tag or "").removeprefix("tiberius_")
        model_cfg = str(self.gate.get("tiberius_model_cfg", ""))
        g = self.genome

        def _rank_gtf(gp: Path) -> tuple[int, int, str]:
            t = gp.name.removeprefix("tiberius_").removesuffix(".gtf")
            is_pref = 0 if (pref and t == pref) else 1
            m_bs = re.search(r"bs(\d+)$", t)
            bs_num = int(m_bs.group(1)) if m_bs else 0
            return (is_pref, -bs_num, gp.name)

        for gp in sorted(raw_gtfs, key=_rank_gtf):
            tag = gp.name.removeprefix("tiberius_").removesuffix(".gtf")
            done_p = tib_dir / f"tiberius_{tag}.done.json"
            if not done_p.exists():
                continue
            try:
                done_data = json.loads(done_p.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(done_data, dict):
                continue
            extra = done_data.get("extra") or {}
            if extra.get("rc", 0) != 0:
                continue
            cached_sha = done_data.get("input_sha256") or extra.get("input_sha256") or ""
            if not cached_sha:
                continue
            bs_val = extra.get("batch_size")
            if bs_val is None:
                m_bs = re.search(r"bs(\d+)$", tag)
                if not m_bs:
                    continue
                bs_val = int(m_bs.group(1))
            seq_len = extra.get("seq_len")
            cpu_only = bool(extra.get("cpu_only", True))
            expected_sha = seedmod._compute_manifest_tiberius_input_sha256(
                g, manifest, model_cfg, int(bs_val), seq_len, cpu_only=cpu_only
            )
            if cached_sha == expected_sha:
                return gp, tag, done_data
        return None

    def _try_relift_zoom_without_tiberius(
        self,
        sd: Path,
        tib_zoom_dir: Path,
        windows: list[tuple[str, int, int]],
        zoom_gtf: Path,
        zoom_faa: Path,
        old_inv: dict,
        prev_zoom_windows: list[tuple[str, int, int]] | None = None,
    ) -> tuple[str | None, int, int, list[dict]]:
        """Re-lift verified raw chunk GTFs (B1+B2) or reuse lifted zoom_gtf (B0-only) when inputs match."""
        g = self.genome
        spacer_bp = int(getattr(self.lp, "tiberius_spacer_bp", 4_500))

        def _norm_merge(ws_in: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
            norm = sorted(
                (str(c), max(0, int(a)), min(g.lengths.get(str(c), 0), int(b)))
                for c, a, b in ws_in
                if int(b) > int(a)
            )
            out_m: list[tuple[str, int, int]] = []
            for c, lo, hi in norm:
                if out_m and out_m[-1][0] == c and lo <= out_m[-1][2]:
                    out_m[-1] = (c, out_m[-1][1], max(out_m[-1][2], hi))
                else:
                    out_m.append((c, lo, hi))
            return out_m

        merged_expected = _norm_merge(windows)

        def _manifest_windows(man: dict) -> list[tuple[str, int, int]]:
            ws: list[tuple[str, int, int]] = []
            for info in (man.get("super_contigs") or {}).values():
                for seg in info.get("segments") or ():
                    ws.append((str(seg["orig_contig"]), int(seg["orig_lo"]), int(seg["orig_hi"])))
            ws.sort()
            return ws

        old_chunks_by_idx = {
            int(cs["chunk"]): cs
            for cs in (old_inv.get("zoom_tiberius_chunks") or ())
            if isinstance(cs, dict) and "chunk" in cs
        }

        # Case 1: Single-chunk verified raw GTF + zoom_manifest.json
        single_man_p = sd / "zoom_manifest.json"
        if single_man_p.exists() and tib_zoom_dir.is_dir():
            try:
                man = json.loads(single_man_p.read_text())
            except (OSError, ValueError):
                man = {}
            pref_single = (old_chunks_by_idx.get(0) or {}).get("tag") or old_inv.get("zoom_tiberius_tag")
            if man.get("spacer_bp") == spacer_bp and _manifest_windows(man) == merged_expected:
                sel_single = self._select_verified_raw_tiberius_gtf(
                    tib_zoom_dir, "zoom_", man, preferred_tag=str(pref_single or "")
                )
                if sel_single is not None:
                    raw_g0, tag0, _d0 = sel_single
                    li = seedmod.lift_tiberius_gtf(
                        raw_g0, man, zoom_gtf, genome=g, out_faa=zoom_faa,
                        trim_spacer_bridges=self.lp.trim_spacer_bridges,
                    )
                    nt = int(li.get("n_trimmed_spacer_tx", 0))
                    nd = int(li.get("n_discarded_spacer_tx", 0))
                    return "relift_repartition_no_tiberius", nt, nd, [{
                        "chunk": 0,
                        "tag": tag0,
                        "n_windows": len(merged_expected),
                        "bp": sum(b - a for _, a, b in merged_expected),
                        "wall_s": 0.0,
                        "n_trimmed_spacer_tx": nt,
                        "n_discarded_spacer_tx": nd,
                    }]

        # Case 2: Multi-chunk verified raw GTFs + zoom_c*_manifest.json
        multi_mans = sorted(sd.glob("zoom_c*_manifest.json"), key=lambda p: int(p.name.split("_")[1][1:]))
        if multi_mans and tib_zoom_dir.is_dir():
            all_ws: list[tuple[str, int, int]] = []
            chunk_pairs: list[tuple[int, dict, Path, str, Path]] = []
            ok = True
            for mp in multi_mans:
                ci = int(mp.name.split("_")[1][1:])
                try:
                    man = json.loads(mp.read_text())
                except (OSError, ValueError):
                    ok = False
                    break
                if man.get("spacer_bp") != spacer_bp:
                    ok = False
                    break
                c_dir = tib_zoom_dir / f"chunk_{ci}"
                pref_ci = (old_chunks_by_idx.get(ci) or {}).get("tag")
                sel_ci = self._select_verified_raw_tiberius_gtf(
                    c_dir, f"zoom_c{ci}_", man, preferred_tag=str(pref_ci or "")
                )
                if sel_ci is None:
                    ok = False
                    break
                raw_g_ci, tag_ci, _d_ci = sel_ci
                all_ws.extend(_manifest_windows(man))
                chunk_pairs.append((ci, man, raw_g_ci, tag_ci, sd / f"zoom_c{ci}_lifted.gtf"))
            if ok and sorted(all_ws) == merged_expected:
                c_gtfs: list[Path] = []
                c_stats: list[dict] = []
                tot_trim = 0
                tot_disc = 0
                for ci, man, raw_g, tag_ci, c_gtf in chunk_pairs:
                    li = seedmod.lift_tiberius_gtf(
                        raw_g, man, c_gtf, genome=g, trim_spacer_bridges=self.lp.trim_spacer_bridges
                    )
                    nt = int(li.get("n_trimmed_spacer_tx", 0))
                    nd = int(li.get("n_discarded_spacer_tx", 0))
                    tot_trim += nt
                    tot_disc += nd
                    c_gtfs.append(c_gtf)
                    ws_c = _manifest_windows(man)
                    c_stats.append({
                        "chunk": ci,
                        "tag": tag_ci,
                        "n_windows": len(ws_c),
                        "bp": sum(b - a for _, a, b in ws_c),
                        "wall_s": 0.0,
                        "n_trimmed_spacer_tx": nt,
                        "n_discarded_spacer_tx": nd,
                    })
                seedmod.merge_tiberius_gtfs(c_gtfs, zoom_gtf, genome=g, out_faa=zoom_faa)
                return "relift_repartition_no_tiberius", tot_trim, tot_disc, c_stats

        # Case 3 (X2/C3/Z2): Pruned tib_zoom/ scratch where lifted tiberius_zoomed.gtf / tiberius_inverted.gtf exists.
        # Requires full inference-identity match (model, image, genome, spacer) + exact raw sorted window list.
        old_preset = (old_inv.get("partition") or {}).get("zoom_preset", old_inv.get("zoom_preset"))
        same_family = (
            old_preset == self.lp.zoom_preset
            or ({str(old_preset), str(self.lp.zoom_preset)} <= {"tight_v1", "tight_v2"})
        )
        raw_expected_sorted = sorted((str(c), int(a), int(b)) for c, a, b in windows)
        raw_prev_sorted = (
            sorted((str(c), int(a), int(b)) for c, a, b in prev_zoom_windows)
            if prev_zoom_windows is not None
            else None
        )
        if (
            same_family
            and old_inv.get("gate_id") == self.gate_id
            and old_inv.get("tiberius_model_cfg") == str(self.gate.get("tiberius_model_cfg", ""))
            and old_inv.get("tiberius_image", seedmod.TIBERIUS_IMAGE) == seedmod.TIBERIUS_IMAGE
            and old_inv.get("tiberius_image_id") == seedmod._tiberius_image_id()
            and old_inv.get("genome_identity") == self._genome_identity_digest()
            and old_inv.get("spacer_bp") == spacer_bp
            and raw_prev_sorted is not None
            and raw_prev_sorted == raw_expected_sorted
            and (zoom_gtf.exists() or (sd / "tiberius_inverted.gtf").exists())
        ):
            if not zoom_gtf.exists():
                shutil.copyfile(sd / "tiberius_inverted.gtf", zoom_gtf)
            return (
                "reuse_lifted_b0_only",
                0,
                int(old_inv.get("n_discarded_spacer_tx", 0)),
                list(old_inv.get("zoom_tiberius_chunks", [])),
            )
        return None, 0, 0, []

    def _seed_inverted(self) -> dict:
        self.prepare()
        sd = self.dir / "seed"
        sd.mkdir(parents=True, exist_ok=True)
        part_p = sd / "partition.pkl"
        part_json = sd / "partition.json"
        inv_p = sd / "inverted_summary.json"
        zw_json = sd / "zoom_windows.json"
        old_inv: dict = {}
        if inv_p.exists():
            try:
                old_inv = json.loads(inv_p.read_text())
            except (OSError, ValueError):
                old_inv = {}

        # X2: Capture previous zoom_windows.json BEFORE overwriting it so Case 3 requires exact window-list equality.
        prev_zoom_windows: list[tuple[str, int, int]] | None = None
        if zw_json.exists():
            try:
                raw_zw = json.loads(zw_json.read_text())
                ws_raw = raw_zw.get("windows", raw_zw) if isinstance(raw_zw, dict) else raw_zw
                prev_zoom_windows = [(str(c), int(a), int(b)) for c, a, b in ws_raw]
            except (OSError, ValueError, TypeError):
                prev_zoom_windows = None

        # Probe Logan before overlapping HISAT2 index build so 100%-Logan cohorts skip hisat2-build completely!
        _seed_logan_runs, seed_fb_short, _seed_probe = self.partition_short_runs()
        need_hisat_in_seed = bool(self.lp.overlap_hisat_index_in_seed and len(seed_fb_short) > 0)

        if (
            part_p.exists()
            and part_json.exists()
            and inv_p.exists()
            and zw_json.exists()
            and (sd / "miniprot.gff3").exists()
            and (sd / "scout_waves.json").exists()
            and (not self.lp.orphan_orf_scout or (sd / "orf_scout.done.json").exists())
            and old_inv.get("b1_b2_applied") is True
            and old_inv.get("seed_cache_key") == self._compute_seed_cache_key(sd, b1_b2_applied=True)
        ):
            sw_ok = True
            try:
                sw_cached = json.loads((sd / "scout_waves.json").read_text())
                exp_sr = (
                    seed_fb_short[:self.lp.scout_max_runs]
                    if self.lp.logan_mode in ("auto", "logan_only")
                    else [r for r, lr in self.runs() if not lr][:self.lp.scout_max_runs]
                )
                if "scout_short_runs" in sw_cached and list(sw_cached["scout_short_runs"]) != exp_sr:
                    sw_ok = False
            except (OSError, ValueError, TypeError):
                sw_ok = False
            if sw_ok:
                cached_summ = json.loads(part_json.read_text())
                cached_summ["seed_exec_path"] = "cached_partition"
                cached_summ["b1_b2_applied"] = True
                if old_inv.get("seed_exec_path") != "cached_partition":
                    old_inv["seed_exec_path"] = "cached_partition"
                    _write_json_atomic(inv_p, old_inv)
                    _write_json_atomic(part_json, cached_summ)
                return cached_summ

        prot = self._seed_protein_path()
        if need_hisat_in_seed:
            avail_cpus = _available_cpus()
            pool = avail_cpus[: max(1, self.lp.threads)]
            hb_n = max(1, min(max(4, min(16, len(pool) // 3)), max(1, len(pool) - 1))) if len(pool) >= 2 else 1
            hb_cpus = pool[:hb_n]
            mp_cpus = pool[hb_n:] or pool
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_hb = ex.submit(self.hisat_index, len(hb_cpus), hb_cpus, True)
                fut_mp = ex.submit(
                    seedmod.run_miniprot,
                    self.genome_fa, prot, sd, self.log,
                    threads=len(mp_cpus),
                    max_shard_bp=self.lp.miniprot_max_shard_bp,
                    outs=self.lp.miniprot_outs,
                    outn=self.lp.miniprot_outn,
                    cpuset=mp_cpus,
                )
                fut_mp.result()
                fut_hb.result()
        else:
            seedmod.run_miniprot(
                self.genome_fa, prot, sd, self.log,
                threads=self.lp.threads,
                max_shard_bp=self.lp.miniprot_max_shard_bp,
                outs=self.lp.miniprot_outs,
                outn=self.lp.miniprot_outn,
            )
        plen = seedmod.protein_lengths(prot)
        hits = seedmod.read_miniprot(sd / "miniprot.gff3", plen)

        scout = self._run_scout_rna(sd, hits)
        # Close idle xsra streams before long CPU Tiberius passes so sockets/RAM do not hang or bloat;
        # arm_c() cleanly restarts from scout_consumed offsets if needed.
        self.close()
        g = self.genome
        scout_js = JunctionStore()
        scout_bins = BinStore(g.lengths)
        for w in scout.get("waves", []):
            scout_js.ingest_tsv(w["run_id"], w["junc_tsv"], g)
            scout_bins.ingest_tsv(w["bins_tsv"])

        t_sl = time.time()
        safe_loci, _safe_hits, unsettled_hits = seedmod.safe_lock_miniprot_loci(
            hits, g, scout_juncs=scout_js, zoom_preset=self.lp.zoom_preset,
        )
        safe_lock_wall = time.time() - t_sl

        rrna_spans = [tuple(x) for x in scout.get("rrna_spans", [])]
        bait_map: dict[str, list[tuple[int, int]]] = {}
        for c, a, b in rrna_spans:
            bait_map.setdefault(c, []).append((a, b))
        organelle_set = set(scout.get("organelle_contigs", []))
        zw = seedmod.build_tiberius_zoom_windows(
            g, unsettled_hits, safe_loci,
            scout_juncs=scout_js, scout_bins=scout_bins,
            bait_intervals=bait_map, organelle_contigs=organelle_set,
            zoom_preset=self.lp.zoom_preset,
        )
        _write_json_atomic(zw_json, {"windows": [list(w) for w in zw["windows"]]})

        zoom_gtf = sd / "tiberius_zoomed.gtf"
        zoom_faa = sd / "tiberius_zoomed.faa"
        tib_zoom_dir = sd / "tib_zoom"
        relift_mode, relift_trim, relift_disc, relift_chunks = self._try_relift_zoom_without_tiberius(
            sd, tib_zoom_dir, zw["windows"], zoom_gtf, zoom_faa, old_inv, prev_zoom_windows=prev_zoom_windows
        )
        if relift_mode is not None:
            seed_exec_path = relift_mode
            b1_b2_applied = relift_mode == "relift_repartition_no_tiberius"
            z_rec = StageRecord(
                name=str(old_inv.get("zoom_tiberius_tag", "tiberius_zoom_relift")),
                wall_seconds=float(old_inv.get("zoom_tiberius_wall_s", 0.0)),
                cpu_user=float(old_inv.get("zoom_tiberius_cpu_s", 0.0)),
                extra={
                    "n_trimmed_spacer_tx": relift_trim,
                    "n_discarded_spacer_tx": relift_disc,
                    "relifted": b1_b2_applied,
                    "b1_b2_applied": b1_b2_applied,
                },
            )
            zoom_chunk_stats = relift_chunks
        else:
            seed_exec_path = "full_tiberius"
            b1_b2_applied = True
            z_rec, zoom_chunk_stats = self._run_tiberius_windows_chunked(
                zw["windows"], sd, tib_zoom_dir, "zoom_", zoom_gtf, zoom_faa, cpu_only=True,
            )

        def _filter_non_safelock(gtf_path: Path) -> list:
            # B0: Drop redundant same-strand Tiberius models overlapping safe_loci, while preserving intact chains extending beyond partials.
            all_t = seedmod.load_tiberius(gtf_path)
            return seedmod.filter_non_safelock_tiberius(all_t, safe_loci, genome=g)

        tib_filtered = None
        other_loci = None
        safe_frac: float | None = None
        use_orf_scout = bool(self.lp.orphan_orf_scout)
        if not use_orf_scout and self.lp.auto_orf_scout_thresh is not None and len(plen) < self.lp.auto_orf_scout_max_proteins:
            tib_pre = _filter_non_safelock(zoom_gtf)
            prelim_other = seedmod.partition_loci(tib_pre, unsettled_hits, g)
            safe_frac = len(safe_loci) / max(1, len(safe_loci) + len(prelim_other))
            use_orf_scout = safe_frac <= self.lp.auto_orf_scout_thresh
            if not use_orf_scout:
                tib_filtered = tib_pre
                other_loci = prelim_other

        orf_gtf = sd / "tiberius_orf_delta.gtf"
        orf_faa = sd / "tiberius_orf_delta.faa"
        orf_stats = {"n_orfs": 0, "n_delta_windows": 0, "delta_bp": 0, "scan_wall_s": 0.0, "tiberius_wall_s": 0.0, "tiberius_cpu_s": 0.0}
        gtf_inputs = [zoom_gtf]
        if use_orf_scout:
            orf_done = sd / "orf_scout.done.json"
            if orf_done.exists() and orf_gtf.exists():
                orf_stats = json.loads(orf_done.read_text())
            else:
                seed_exec_path = "full_tiberius"
                excluded: dict[str, list[tuple[int, int]]] = {}
                for l in safe_loci:
                    excluded.setdefault(l.contig, []).append((l.start, l.end))
                for c, a, b in zw["windows"]:
                    excluded.setdefault(c, []).append((a, b))
                for c, a, b in rrna_spans:
                    excluded.setdefault(c, []).append((a, b))
                t_scan = time.time()
                orfs = orfscout_mod.scan_orphan_orfs(
                    g, excluded_by_contig=excluded, organelle_contigs=organelle_set, threads=self.lp.threads
                )
                delta_wins = orfscout_mod.orphan_orf_delta_windows(orfs, g.lengths, excluded_by_contig=excluded)
                scan_wall = time.time() - t_scan
                delta_bp = sum(b - a for _, a, b in delta_wins)
                d_rec, _ = self._run_tiberius_windows_chunked(
                    delta_wins, sd, sd / "tib_orf_delta", "orf_", orf_gtf, orf_faa, cpu_only=True,
                )
                orf_stats = {
                    "n_orfs": len(orfs),
                    "n_delta_windows": len(delta_wins),
                    "delta_bp": delta_bp,
                    "scan_wall_s": round(scan_wall, 3),
                    "tiberius_wall_s": round(d_rec.wall_seconds, 3),
                    "tiberius_cpu_s": round(d_rec.cpu_seconds, 3),
                }
                orf_done.write_text(json.dumps(orf_stats, indent=1))
            if orf_gtf.exists() and orf_gtf.stat().st_size > 0:
                gtf_inputs.append(orf_gtf)

        inv_gtf = sd / "tiberius_inverted.gtf"
        inv_faa = sd / "tiberius_inverted.faa"
        if len(gtf_inputs) == 1:
            if inv_gtf.resolve() != zoom_gtf.resolve():
                shutil.copyfile(zoom_gtf, inv_gtf)
            if zoom_faa.exists():
                if inv_faa.resolve() != zoom_faa.resolve():
                    shutil.copyfile(zoom_faa, inv_faa)
            else:
                inv_faa.write_text("")
        else:
            seedmod.merge_tiberius_gtfs(gtf_inputs, inv_gtf, genome=g, out_faa=inv_faa)

        t0 = time.time()
        if tib_filtered is None or other_loci is None:
            tib_filtered = _filter_non_safelock(inv_gtf)
            other_loci = seedmod.partition_loci(tib_filtered, unsettled_hits, g)
        if safe_frac is None:
            safe_frac = len(safe_loci) / max(1, len(safe_loci) + len(other_loci))
        loci = sorted(safe_loci + other_loci, key=lambda l: (l.contig, l.start, l.end))
        for i_l, loc in enumerate(loci, 1):
            loc.locus_id = f"L{i_l:06d}"
        summ = seedmod.partition_summary(loci)
        tib_wall = z_rec.wall_seconds + orf_stats["tiberius_wall_s"]
        tib_cpu = z_rec.cpu_seconds + orf_stats["tiberius_cpu_s"]
        n_trimmed_sp = int((z_rec.extra or {}).get("n_trimmed_spacer_tx", 0))
        n_disc_sp = int((z_rec.extra or {}).get("n_discarded_spacer_tx", 0))
        min_mem_gib = (z_rec.extra or {}).get("min_mem_available_gib", old_inv.get("min_mem_available_gib"))
        bs_stepdown_flag = bool((z_rec.extra or {}).get("bs_stepdown", old_inv.get("bs_stepdown", False)))
        eff_res_gib = (z_rec.extra or {}).get("effective_reserve_gib", old_inv.get("effective_reserve_gib"))
        eff_spacer = int(getattr(self.lp, "tiberius_spacer_bp", 4_500))
        cur_genome_id = self._genome_identity_digest()
        cur_model_cfg = str(self.gate.get("tiberius_model_cfg", ""))
        cur_image_id = seedmod._tiberius_image_id()
        seed_cache_key = self._compute_seed_cache_key(sd, b1_b2_applied=b1_b2_applied)
        summ.update(
            n_tiberius=len(tib_filtered),
            n_hit_chains=len(hits),
            tiberius_run="cpu_focaloop_orf" if use_orf_scout else "cpu_focaloop",
            seed_mode="inverted_cpu",
            seed_logic_version=SEED_LOGIC_VERSION,
            seed_exec_path=seed_exec_path,
            b1_b2_applied=b1_b2_applied,
            bs_stepdown=bs_stepdown_flag,
            min_mem_available_gib=min_mem_gib,
            effective_reserve_gib=eff_res_gib,
            spacer_bp=eff_spacer,
            tiberius_model_cfg=cur_model_cfg,
            tiberius_image_id=cur_image_id,
            safe_locked_loci=len(safe_loci),
            safe_locked_frac=round(safe_frac, 4),
            orf_scout_enabled=use_orf_scout,
            zoom_preset=self.lp.zoom_preset,
            zoom_chunks=len(zoom_chunk_stats),
            tib_window_bp=zw["tib_window_bp"],
            tib_window_frac=round(zw["tib_window_frac"], 6),
            n_trimmed_spacer_tx=n_trimmed_sp,
            n_discarded_spacer_tx=n_disc_sp,
            orf_delta_bp=orf_stats["delta_bp"],
            tiberius_wall_s=round(tib_wall, 3),
            tiberius_cpu_s=round(tib_cpu, 3),
        )
        part_tmp = part_p.with_name(f".{part_p.name}.tmp.{os.getpid()}")
        with open(part_tmp, "wb") as fh:
            pickle.dump({"tiberius": tib_filtered, "hits": hits, "loci": loci, "summary": summ}, fh)
        os.replace(part_tmp, part_p)
        _write_json_atomic(part_json, summ)
        inv_summary = {
            "gate_id": self.gate_id,
            "seed_logic_version": SEED_LOGIC_VERSION,
            "seed_cache_key": seed_cache_key,
            "seed_exec_path": seed_exec_path,
            "b1_b2_applied": b1_b2_applied,
            "bs_stepdown": bs_stepdown_flag,
            "min_mem_available_gib": min_mem_gib,
            "effective_reserve_gib": eff_res_gib,
            "tiberius_model_cfg": cur_model_cfg,
            "tiberius_image": seedmod.TIBERIUS_IMAGE,
            "tiberius_image_id": cur_image_id,
            "genome_identity": cur_genome_id,
            "spacer_bp": eff_spacer,
            "mode": summ["tiberius_run"],
            "zoom_preset": self.lp.zoom_preset,
            "genome_bp": zw["nuclear_genome_bp"],
            "tib_window_bp": zw["tib_window_bp"],
            "tib_window_frac": round(zw["tib_window_frac"], 6),
            "n_zoom_windows": len(zw["windows"]),
            "safe_locked_loci": len(safe_loci),
            "safe_lock_wall_s": round(safe_lock_wall, 3),
            "scout_spots": scout.get("spots_total", 0),
            "scout_wire_bytes": scout.get("wire_bytes", 0),
            "scout_wall_s": round(scout.get("wall_s", 0.0), 3),
            "zoom_tiberius_tag": z_rec.name,
            "zoom_tiberius_chunks": zoom_chunk_stats,
            "n_trimmed_spacer_tx": n_trimmed_sp,
            "n_discarded_spacer_tx": n_disc_sp,
            "zoom_tiberius_wall_s": round(z_rec.wall_seconds, 3),
            "zoom_tiberius_cpu_s": round(z_rec.cpu_seconds, 3),
            "orf_scout": orf_stats,
            "tiberius_wall_s": round(tib_wall, 3),
            "tiberius_cpu_s": round(tib_cpu, 3),
            "partition": summ,
        }
        _write_json_atomic(inv_p, inv_summary)
        self.log.append(StageRecord(name="partition_inverted", wall_seconds=time.time() - t0, extra=summ))
        return summ

    def ensure_structural_ncrna(self) -> dict:
        """Run (or reuse cached) post-Seed genome-only structural ncRNA pipeline in seed/ncrna_raw/."""
        meta_p = self.dir / "genome" / "genome_meta.json"
        org_set: set[str] = set()
        if meta_p.exists():
            try:
                org_set = set(json.loads(meta_p.read_text()).get("organelle_contigs", []))
            except (OSError, ValueError, TypeError):
                org_set = set()
        return ncrna_mod.ensure_structural_ncrna_raw(
            genome=self.genome,
            genome_fa=self.genome_fa,
            genome_identity=self._genome_identity_digest(),
            organelle_contigs=org_set,
            gate=self.gate,
            out_dir=self.dir / "seed" / "ncrna_raw",
            threads=self.lp.threads,
            mode=self.lp.enable_structural_ncrna,
            log=self.log,
        )

    def _get_ncrna_raw_candidates(self, *, run_if_missing: bool = False) -> tuple[list[dict], str, dict]:
        """Load verified raw structural ncRNA candidates and provenance digest token."""
        mode = self.lp.enable_structural_ncrna
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"Invalid enable_structural_ncrna={mode!r}; expected 'auto', 'on', or 'off'")
        if mode == "off":
            return [], "disabled", {"status": "disabled", "n_raw_candidates": 0}

        ncrna_dir = self.dir / "seed" / "ncrna_raw"
        raw_p = ncrna_dir / "raw_ncrna_candidates.json"
        done_p = ncrna_dir / "done.json"
        taxon_group = ncrna_mod.resolve_rfam_taxon_group(self.gate)

        if done_p.exists() and raw_p.exists():
            try:
                done_data = json.loads(done_p.read_text())
            except (OSError, ValueError):
                done_data = {}
            done_stages = done_data.get("stages") if isinstance(done_data, dict) else None
            stages_arg = done_stages if isinstance(done_stages, dict) else None
            expected_ck = ncrna_mod.compute_ncrna_raw_cache_key(
                self._genome_identity_digest(), taxon_group, stages=stages_arg
            )
            full_stages = stages_arg is None or all(
                bool(stages_arg.get(k, False)) for k in ("trnascan", "barrnap", "rfam")
            )
            rfam_used = stages_arg is None or bool(stages_arg.get("rfam", False))
            exp_asset_sha = ncrna_mod.PINNED_RFAM_ASSETS if rfam_used else {}
            if (
                isinstance(done_data, dict)
                and (mode != "on" or full_stages)
                and done_data.get("cache_key") == expected_ck
                and done_data.get("asset_sha256") == exp_asset_sha
            ):
                actual_sha = _sha256_file(raw_p)
                if done_data.get("raw_candidates_sha256") == actual_sha:
                    if rfam_used:
                        rdir_cand = ncrna_mod.default_rfam_dir()
                        for fname, exp_sha in ncrna_mod.PINNED_RFAM_ASSETS.items():
                            fp = rdir_cand / fname
                            if fp.is_file() and _sha256_file(fp) != exp_sha:
                                raise RuntimeError(f"Pinned Rfam 15.1 SHA256 mismatch for {fp}")
                    return json.loads(raw_p.read_text()), actual_sha, {**done_data, "cached": True}

        if run_if_missing:
            info = self.ensure_structural_ncrna()
            if info.get("status") in ("ok", "partial") and info.get("raw_candidates_path"):
                rp = Path(info["raw_candidates_path"])
                return json.loads(rp.read_text()), str(info["raw_candidates_sha256"]), info
            st_tok = str(info.get("status") or "skipped_missing_assets")
            return [], st_tok, info

        if mode == "on":
            ncrna_mod.check_ncrna_assets(mode="on")
            raise FileNotFoundError(
                f"Missing structural ncRNA raw cache at {raw_p} while enable_structural_ncrna='on'"
            )
        return [], "none", {"status": "none", "n_raw_candidates": 0}

    def seed(self, bench_batch: bool = False) -> dict:
        if self._skip_tiberius:
            self.prepare()
            sd = self.dir / "seed"
            prot = self._seed_protein_path()
            seedmod.run_miniprot(
                self.genome_fa, prot, sd, self.log,
                threads=self.lp.threads,
                max_shard_bp=self.lp.miniprot_max_shard_bp,
                outs=self.lp.miniprot_outs,
                outn=self.lp.miniprot_outn,
            )
            part_p = sd / "partition.pkl"
            if not part_p.exists():
                t0 = time.time()
                plen = seedmod.protein_lengths(prot) if Path(str(prot) + ".fai").exists() and prot.stat().st_size > 0 else {}
                hits = seedmod.read_miniprot(sd / "miniprot.gff3", plen) if (sd / "miniprot.gff3").exists() else []
                safe_loci, _sh, unsettled_hits = seedmod.safe_lock_miniprot_loci(hits, self.genome)
                other_loci = seedmod.partition_loci([], unsettled_hits, self.genome)
                loci = sorted(safe_loci + other_loci, key=lambda l: (l.contig, l.start, l.end))
                for i_l, loc in enumerate(loci, 1):
                    loc.locus_id = f"L{i_l:06d}"
                summ = seedmod.partition_summary(loci)
                summ.update(
                    n_tiberius=0, n_hit_chains=len(hits),
                    tiberius_run="evidence_only", seed_mode="fast_evidence",
                    cpu_only_tiberius=False, safe_locked_loci=len(safe_loci),
                )
                with open(part_p, "wb") as fh:
                    pickle.dump({"tiberius": [], "hits": hits, "loci": loci, "summary": summ}, fh)
                (sd / "partition.json").write_text(json.dumps(summ, indent=1))
                self.log.append(StageRecord(name="partition", wall_seconds=time.time() - t0, extra=summ))
            part_summ = json.loads((sd / "partition.json").read_text())
            nc_info = self.ensure_structural_ncrna()
            return {**part_summ, "structural_ncrna": nc_info}
        if self.lp.cpu_focaloop_tiberius:
            summ = self._seed_inverted()
            nc_info = self.ensure_structural_ncrna()
            return {**summ, "structural_ncrna": nc_info}
        self.prepare()
        sd = self.dir / "seed"
        bench = {}
        cpu_only = bool(self.lp.cpu_only_tiberius)
        if bench_batch and not cpu_only:
            r4 = seedmod.run_tiberius(self.gate, self.genome_fa, sd, self.log, batch_size=4)
            bench["bs4"] = _tib_summary(r4)
            if r4.peak_vram_mb < 13000:
                r6 = seedmod.run_tiberius(self.gate, self.genome_fa, sd, self.log, batch_size=6)
                bench["bs6"] = _tib_summary(r6)
            prod_tag = "bs6" if (sd / "tiberius_bs6.done.json").exists() else "bs4"
        else:
            fb_kw: dict = {}
            if cpu_only and getattr(self._run_tiberius_with_fallback, "__func__", None) is GateRun._run_tiberius_with_fallback:
                eff_res = _effective_reserve_gib(self.lp.tiberius_reserve_gib)
                _keff, approved_bs, bs_stepdown, _avail = _approve_wave_workers_and_bs(
                    1, str(self.gate.get("tiberius_model_cfg", "")), eff_res
                )
                fb_kw = {
                    "approved_bs": approved_bs,
                    "bs_stepdown": bs_stepdown,
                    "effective_reserve_gib": eff_res,
                }
            prod_tag, rec_t = self._run_tiberius_with_fallback(self.genome_fa, sd, cpu_only=cpu_only, **fb_kw)
            bench[prod_tag] = _tib_summary(rec_t)
        tib_gtf = sd / f"tiberius_{prod_tag}.gtf"
        prot = self._seed_protein_path()
        if self.lp.seed_mode == "full":
            seedmod.run_miniprot(self.genome_fa, prot, sd, self.log, threads=self.lp.threads)
        else:
            seedmod.run_targeted_miniprot(self.genome_fa, prot, tib_gtf, sd, self.log, threads=self.lp.threads)
        part_p = sd / "partition.pkl"
        if not part_p.exists():
            t0 = time.time()
            tib = seedmod.load_tiberius(tib_gtf)
            plen = seedmod.protein_lengths(prot) if Path(str(prot) + ".fai").exists() and prot.stat().st_size > 0 else {}
            hits = seedmod.read_miniprot(sd / "miniprot.gff3", plen)
            loci = seedmod.partition_loci(tib, hits, self.genome)
            summ = seedmod.partition_summary(loci)
            summ.update(
                n_tiberius=len(tib), n_hit_chains=len(hits),
                tiberius_run=prod_tag, seed_mode=self.lp.seed_mode,
                cpu_only_tiberius=cpu_only,
            )
            with open(part_p, "wb") as fh:
                pickle.dump({"tiberius": tib, "hits": hits, "loci": loci, "summary": summ}, fh)
            (sd / "partition.json").write_text(json.dumps(summ, indent=1))
            self.log.append(StageRecord(name="partition", wall_seconds=time.time() - t0, extra=summ))
        bp = sd / "tiberius_batch_bench.json"
        old = json.loads(bp.read_text()) if bp.exists() else {}
        old.update(bench)
        bp.write_text(json.dumps(old, indent=1))
        part_summ = json.loads((sd / "partition.json").read_text())
        nc_info = self.ensure_structural_ncrna()
        return {**part_summ, "structural_ncrna": nc_info}

    def seed_bench(self, out_subdir: str = "seed_v02_targeted") -> dict:
        """Offline benchmark of v0.2 targeted seed vs existing v0.1 full-genome seed (zero network, non-destructive)."""
        sd = self.dir / "seed"
        bdir = self.dir / out_subdir
        bdir.mkdir(parents=True, exist_ok=True)
        prod_tag = "bs6" if (sd / "tiberius_bs6.done.json").exists() else "bs4"
        tib_gtf = sd / f"tiberius_{prod_tag}.gtf"
        prot = self._seed_protein_path()
        rec_v02 = seedmod.run_targeted_miniprot(self.genome_fa, prot, tib_gtf, bdir, self.log, threads=self.lp.threads)
        plen = seedmod.protein_lengths(prot)
        tib = seedmod.load_tiberius(tib_gtf)
        hits_v02 = seedmod.read_miniprot(bdir / "miniprot.gff3", plen)
        loci_v02 = seedmod.partition_loci(tib, hits_v02, self.genome)
        summ_v02 = seedmod.partition_summary(loci_v02)
        out: dict = {
            "gate_id": self.gate_id,
            "v02_targeted": {
                "wall_s": rec_v02.wall_seconds,
                "cpu_s": rec_v02.cpu_seconds,
                "extra": rec_v02.extra,
                "n_hit_chains": len(hits_v02),
                "partition": summ_v02,
            },
        }
        v01_done = sd / "miniprot.done.json"
        if v01_done.exists() and (sd / "partition.pkl").exists():
            d01 = json.loads(v01_done.read_text())
            s01 = self.load_seed()
            cpu01 = float(d01.get("cpu_user", 0.0)) + float(d01.get("cpu_sys", 0.0))
            wall01 = float(d01.get("wall_seconds", 0.0))
            locked01 = {
                (l.contig, l.strand, tuple(l.locked_model.cds_introns))
                for l in s01["loci"]
                if l.status == LOCKED and l.locked_model is not None
            }
            locked02 = {
                (l.contig, l.strand, tuple(l.locked_model.cds_introns))
                for l in loci_v02
                if l.status == LOCKED and l.locked_model is not None
            }
            union_l = len(locked01 | locked02)
            out["v01_full"] = {
                "wall_s": wall01,
                "cpu_s": cpu01,
                "n_hit_chains": len(s01["hits"]),
                "partition": s01["summary"],
            }
            out["comparison"] = {
                "cpu_speedup_x": round(cpu01 / max(0.001, rec_v02.cpu_seconds), 2),
                "wall_speedup_x": round(wall01 / max(0.001, rec_v02.wall_seconds), 2),
                "locked_v01": len(locked01),
                "locked_v02": len(locked02),
                "locked_recall_vs_v01": round(len(locked01 & locked02) / max(1, len(locked01)), 4),
                "locked_jaccard": round(len(locked01 & locked02) / max(1, union_l), 4),
            }
        (bdir / "seed_bench.json").write_text(json.dumps(out, indent=1))
        return out

    def load_seed(self) -> dict:
        with open(self.dir / "seed" / "partition.pkl", "rb") as fh:
            return pickle.load(fh)

    def hisat_index(self, threads: int | None = None, cpuset: list[int] | None = None, force: bool = False) -> Path:
        idx = self.dir / "genome" / "hisat2" / "genome"
        if self._done("hisat2_index").exists():
            return idx
        if not force and self.lp.logan_mode in ("auto", "logan_only"):
            _logan_runs, fb_short_runs, _probe = self.partition_short_runs()
            if len(fb_short_runs) == 0:
                return idx
        idx.parent.mkdir(parents=True, exist_ok=True)
        rec = StageRecord(name="hisat2_index")
        t0 = time.time()
        n_threads = len(cpuset) if cpuset else (threads if threads is not None else self.lp.threads)
        cmd, taskset_unavail = seedmod.wrap_taskset_cmd(
            ["hisat2-build", "-p", str(n_threads), str(self.genome_fa), str(idx)], cpuset
        )
        with open(idx.parent / "build.log", "w") as lf:
            _, _, u, s = run_cmd(cmd, stdout=lf, stderr=lf)
        rec.wall_seconds, rec.cpu_user, rec.cpu_sys = time.time() - t0, u, s
        rec.extra = {"index_bytes": dir_bytes(idx.parent), "threads": n_threads}
        if taskset_unavail:
            rec.extra["taskset_unavailable"] = True
        self.log.append(rec)
        self._done("hisat2_index").touch()
        return idx

    def mm2_index(self) -> Path | None:
        has_long = bool(self.gate.get("sra_long_runs")) or any(
            v.get("long_read", False) for v in self._local_streams.values()
        )
        if not has_long:
            return None
        mmi = self.dir / "genome" / "genome.splice.mmi"
        if self._done("mm2_index").exists():
            return mmi
        rec = StageRecord(name="minimap2_index")
        t0 = time.time()
        mm2_i_args = minimap2_index_size_args(sum(self.genome.lengths.values()))
        with open(self.dir / "genome" / "mm2_index.log", "w") as lf:
            _, _, u, s = run_cmd(
                ["minimap2", "-x", "splice:hq", *mm2_i_args, "-t", str(self.lp.threads), "-d", str(mmi), str(self.genome_fa)],
                stderr=lf,
            )
        rec.wall_seconds, rec.cpu_user, rec.cpu_sys = time.time() - t0, u, s
        rec.extra = {"index_bytes": mmi.stat().st_size}
        self.log.append(rec)
        self._done("mm2_index").touch()
        return mmi

    def runs(self) -> list[tuple[str, bool]]:
        local_short = [(rid, False) for rid, v in self._local_streams.items() if not v.get("long_read", False)]
        local_long = [(rid, True) for rid, v in self._local_streams.items() if v.get("long_read", False)]
        return (
            [(r, False) for r in self.gate.get("sra_short_runs", [])]
            + local_short
            + [(r, True) for r in self.gate.get("sra_long_runs", [])]
            + local_long
        )

    def run_info(self, runs: list[tuple[str, bool]] | None = None) -> dict:
        info = {}
        target_runs = runs if runs is not None else self.runs()
        for r, lr in target_runs:
            if r in self._local_streams:
                st = self._local_streams[r]["stream"]
                info[r] = {
                    "run_id": r,
                    "total_spots": int(st.total_spots),
                    "n_segments": int(st.segs),
                    "long_read": lr,
                    "local": True,
                }
            else:
                d = describe_run(r, self.dir / "sra_meta")
                d["long_read"] = lr
                info[r] = d
        return info

    # ------------------------------------------------------------------ Arm A
    def arm_a(self) -> dict:
        ad = self.dir / "armA"
        ad.mkdir(exist_ok=True)
        out = ad / "armA.gff3"
        s = self.load_seed()
        if not out.exists():
            raw_ncrna, _nc_sha, _nc_info = self._get_ncrna_raw_candidates(run_if_missing=True)
            meta_p = self.dir / "genome" / "genome_meta.json"
            org_set: set[str] = set()
            if meta_p.exists():
                try:
                    org_set = set(json.loads(meta_p.read_text()).get("organelle_contigs", []))
                except (OSError, ValueError, TypeError):
                    org_set = set()
            t0, c0 = time.time(), self_cpu()
            res = synthesize(
                self.genome, s["tiberius"], s["hits"], s["loci"], params=self.sp, prefix="GLA",
                organelle_contigs=org_set, raw_ncrna_candidates=raw_ncrna, zoom=self._enable_zoom,
                threads=self.lp.threads,
            )
            write_gff3(res.models, out)
            write_gff3(_tiberius_models(s["tiberius"]), ad / "tiberius_standalone.gff3")
            rec = StageRecord(name="synth_armA", wall_seconds=time.time() - t0, cpu_user=self_cpu() - c0, extra=res.stats)
            self.log.append(rec)
        return {"gff3": str(out)}

    # ------------------------------------------------------------------ Arm B
    def arm_b(self, keep_bams: bool = False) -> dict:
        bd = self.dir / "armB"
        bd.mkdir(exist_ok=True)
        out = bd / "armB.gff3"
        if out.exists() and (bd / "armB_summary.json").exists():
            return json.loads((bd / "armB_summary.json").read_text())
        idx = self.hisat_index(force=True)
        mmi = self.mm2_index()
        info = self.run_info()
        aln = bd / "align"
        summary: dict = {"runs": {}}
        with ScratchMonitor(bd, interval=10.0) as mon:
            rx0 = self.nic.rx()
            t0 = time.time()
            for r, lr in self.runs():
                done = aln / f"{r}.done.json"
                if done.exists():
                    summary["runs"][r] = json.loads(done.read_text())
                    continue
                res = align_run_full(
                    r, lr, idx, mmi, aln,
                    threads=self.lp.hisat_threads,
                    nic=None if r in self._local_streams else self.nic,
                    stranded=self._is_long_run_stranded(r),
                    genome_bp=sum(self.genome.lengths.values()),
                )
                d = asdict(res)
                d["long_read"] = lr
                d["total_spots"] = info[r]["total_spots"]
                done.write_text(json.dumps(d))
                summary["runs"][r] = d
                self.log.append(StageRecord(name=f"armB_align_{r}", wall_seconds=res.wall, cpu_user=sum(res.cpu.values()),
                                            wire_bytes=res.wire_bytes, extra={"cpu": res.cpu, "spots": res.spots}))
            align_wall = time.time() - t0
            wire = self.nic.rx() - rx0
            # StringTie on all full BAMs
            short_bams = [summary["runs"][r]["bam"] for r, lr in self.runs() if not lr]
            long_bams = [summary["runs"][r]["bam"] for r, lr in self.runs() if lr]
            gtf = bd / "stringtie.gtf"
            st_cpu = 0.0
            t1 = time.time()
            merged = bd / "merged_short.bam"
            if not gtf.exists():
                # StringTie 2.1.7 segfaults on multi-BAM input: merge sorted BAMs first (counted in peak scratch)
                if len(short_bams) > 1:
                    _, _, u, s_ = run_cmd(["samtools", "merge", "-f", "-@", str(self.lp.threads), "-o", str(merged)] + short_bams)
                    st_cpu += u + s_
                    mon.sample()
                    st_in = str(merged)
                else:
                    st_in = short_bams[0]
                _, _, u, s_ = run_cmd(["stringtie", "-p", str(self.lp.threads), "-o", str(gtf), st_in])
                st_cpu += u + s_
                mon.sample()
                merged.unlink(missing_ok=True)
                if long_bams:
                    for b in long_bams:
                        run_cmd(["samtools", "index", b])
                    _, _, u, s_ = run_cmd(["stringtie", "-L", "-p", str(self.lp.threads), "-o", str(bd / "stringtie_long.gtf")] + long_bams)
                    st_cpu += u + s_
            st_wall = time.time() - t1
            mon.sample()
            bam_bytes = sum(Path(summary["runs"][r]["bam"]).stat().st_size for r, _ in self.runs() if Path(summary["runs"][r]["bam"]).exists())
            # evidence + synthesis
            t2, c2 = time.time(), self_cpu()
            js = JunctionStore()
            bins = BinStore(self.genome.lengths)
            for r, _ in self.runs():
                js.ingest_tsv(r, summary["runs"][r]["junc_tsv"], self.genome)
                bins.ingest_tsv(summary["runs"][r]["bins_tsv"])
            rna_tx = read_models(gtf, fmt="gtf")
            if (bd / "stringtie_long.gtf").exists():
                rna_tx += _read_long_read_models(bd / "stringtie_long.gtf")
            s = self.load_seed()
            ctx = self._synth_context(s, cache_dir=bd, include_raw_alignments=False)
            raw_ncrna, _nc_sha, _nc_info = self._get_ncrna_raw_candidates(run_if_missing=True)
            scanned_wins = self._load_seed_zoom_windows(ctx["organelle_contigs"])
            res = synthesize(
                self.genome, s["tiberius"], s["hits"], s["loci"], junctions=js, rna_tx=rna_tx, bins=bins,
                params=self.sp, prefix="GLB",
                bait_intervals=ctx["bait_intervals"], organelle_contigs=ctx["organelle_contigs"],
                readthrough_targets=ctx["readthrough_targets"], raw_ncrna_candidates=raw_ncrna,
                scanned_tib_windows=scanned_wins, threads=self.lp.threads,
            )
            write_gff3(res.models, out)
            synth_wall = time.time() - t2
            synth_cpu = (self_cpu() - c2) + float(res.stats.get("diamond_cpu_s", 0.0))
            with open(bd / "evidence.pkl", "wb") as fh:
                pickle.dump({"junctions": js, "bins_total": bins.total}, fh)
        peak = mon.peak
        # derive from per-run checkpoints so resumed runs report the real alignment wall / wire bytes
        align_wall = sum(x.get("wall", 0.0) for x in summary["runs"].values())
        wire = sum(x.get("wire_bytes", 0) for x in summary["runs"].values())
        summary.update({
            "align_wall_s": align_wall,
            "wire_bytes": wire,
            "stringtie_cpu_s": st_cpu,
            "stringtie_wall_s": st_wall,
            "synth_wall_s": synth_wall,
            "synth_stats": res.stats,
            "peak_scratch_bytes": peak,
            "bam_bytes": bam_bytes,
            "spots_streamed": sum(summary["runs"][r].get("spots", 0) for r, _ in self.runs()),
            "total_spots": sum(info[r]["total_spots"] for r, _ in self.runs()),
        })
        # retained state of a batch workflow = full sorted BAMs + StringTie GTF + models
        summary["retained_state_bytes"] = bam_bytes + gtf.stat().st_size + out.stat().st_size
        self.log.append(StageRecord(name="armB_stringtie", wall_seconds=st_wall, cpu_user=st_cpu))
        summary["synth_cpu_s"] = synth_cpu
        self.log.append(StageRecord(name="synth_armB", wall_seconds=synth_wall, cpu_user=synth_cpu, extra=res.stats))
        if not keep_bams:
            for r, _ in self.runs():
                for suf in (".bam", ".bam.bai"):
                    Path(str(aln / r) + suf).unlink(missing_ok=True)
            summary["bams_deleted_after_measurement"] = True
        (bd / "armB_summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary

    # ------------------------------------------------------------------ Arm C (GeneLoop)
    def _sieve_inputs(self, s: dict, cd: Path) -> tuple[Path, Path, list]:
        """Candidate FASTA (all CDS exons of LOCKED loci, tantan-masked) + bait FASTA (organelles + 45S rRNA)."""
        cand = cd / "sieve" / "locked_cds_exons.fa"
        cand_m = cd / "sieve" / "locked_cds_exons.tantan.fa"
        bait = cd / "sieve" / "bait.fa"
        cand.parent.mkdir(parents=True, exist_ok=True)
        locked = [l for l in s["loci"] if l.status == LOCKED and l.locked_model is not None]
        if not cand_m.exists():
            with open(cand, "w") as fh:
                for l in locked:
                    assert re.fullmatch(r"L\d+", l.locus_id), f"glcore requires L<digits> locus_id, got {l.locus_id!r}"
                    cds = sorted(l.locked_model.cds)
                    if not cds:
                        continue
                    seqs = [self.genome.fetch(l.contig, a, b) for a, b in cds]
                    fh.write(f">{l.locus_id}\n{'N'.join(seqs)}\n")
            if cand.stat().st_size > 0 and shutil.which("tantan"):
                with open(cand_m, "w") as fo:
                    run_cmd(["tantan", str(cand)], stdout=fo)
            else:
                shutil.copyfile(cand, cand_m)
            self._bait_spans_and_fa(self.dir / "seed", s.get("hits", ()), tib=s.get("tiberius", ()), out_fa=bait)
        return cand_m, bait, locked

    def arm_c(self) -> dict:
        cd = self.dir / f"arm{self.c_tag}"
        cd.mkdir(exist_ok=True)
        if (cd / f"arm{self.c_tag}_summary.json").exists():
            return json.loads((cd / f"arm{self.c_tag}_summary.json").read_text())
        logan_runs, fallback_short_runs, logan_probe = self.partition_short_runs()
        long_runs = [str(r) for r in self.gate.get("sra_long_runs", [])] + [
            rid for rid, v in self._local_streams.items() if v.get("long_read", False)
        ]
        sra_stream_runs = [(r, False) for r in fallback_short_runs] + [(r, True) for r in long_runs]
        glc = build_glcore()
        idx = self.hisat_index(force=True) if fallback_short_runs else (self.dir / "genome" / "hisat2" / "genome")
        mmi = self.mm2_index() if long_runs else None
        info = self.run_info(runs=sra_stream_runs) if sra_stream_runs else {}
        s = self.load_seed()
        lp = self.lp
        g = self.genome
        tele_rows: list[dict] = []
        with ScratchMonitor(cd, interval=5.0) as mon:
            locked = [l for l in s["loci"] if l.status == LOCKED and l.locked_model is not None]
            cand_m: Path | None = None
            bait: Path | None = None
            mult: Path | None = None
            active: Path | None = None
            mult_stats: dict = {}
            act_stats: dict = {}
            sieve_index_wall = 0.0

            def _ensure_short_sieve_ready() -> None:
                nonlocal idx, cand_m, bait, locked, mult, active, mult_stats, act_stats, sieve_index_wall
                if mult is not None and active is not None:
                    return
                idx = self.hisat_index(force=True)
                t0_s = time.time()
                cand_m, bait, locked = self._sieve_inputs(s, cd)
                mult = cd / "sieve" / "cand.mult"
                mres = subprocess.run(
                    [str(glc), "mult", "-g", str(self.genome_fa), "-c", str(cand_m), "-o", str(mult), "-t", str(lp.threads)],
                    capture_output=True, text=True, check=True, env=safe_env(),
                )
                mult_stats = json.loads(mres.stderr.strip().splitlines()[-1])
                sieve_index_wall = time.time() - t0_s
                self.log.append(
                    StageRecord(
                        name=f"arm{self.c_tag}_sieve_index",
                        wall_seconds=sieve_index_wall,
                        cpu_user=mult_stats.get("seconds", 0) * lp.threads,
                        extra=mult_stats,
                    )
                )
                active = cd / "sieve" / "active_0.bin"
                act_stats = _activate(glc, active, bait=bait)

            if fallback_short_runs:
                # ---- Stage 1 index: genome-wide multiplicity of candidate syncmers (needed only for short-read SRA fallback)
                _ensure_short_sieve_ready()

            # ---- loop state
            locked_introns = set()
            locus_by_id = {l.locus_id: l for l in locked}
            locus_introns: dict[str, list] = {}
            for l in locked:
                ii = l.locked_model.cds_introns
                locus_introns[l.locus_id] = ii
                for a, b in ii:
                    locked_introns.add((l.contig, a, b))
            lidx_locked = LocusIndex([(l.contig, l.start, l.end, l) for l in locked])
            lidx_uns = LocusIndex([(l.contig, l.start, l.end, l) for l in s["loci"] if l.status == UNSETTLED], pad=2000)
            lidx_all = LocusIndex([(l.contig, l.start, l.end, l) for l in s["loci"]], pad=5000)
            gate_sieve = OpenSyncmerDepletionSieve(
                min_exon_depth=lp.min_exon_depth,
                min_junction_depth=lp.min_junction_depth,
                max_singleton_ratio=lp.max_singleton_ratio,
            )
            locus_obs: dict[str, dict] = {}
            saturated: dict[str, int] = {}
            js = JunctionStore()
            fallback_js = JunctionStore()
            bins = BinStore(g.lengths)
            region_of: dict = {}
            ev_dir = cd / "state"
            ev_dir.mkdir(exist_ok=True)
            waves_dir = cd / "waves"
            waves_dir.mkdir(parents=True, exist_ok=True)
            cpu_by_tool: dict[str, float] = {}
            sieve_tot = {"pairs_in": 0, "pairs_out": 0, "pairs_dropped_locus": 0, "pairs_dropped_bait": 0}
            wave_bams: dict[str, list[str]] = {"short": [], "long": []}
            spots_used = 0
            wave = 0
            stop_reason = ""
            loop_t0, loop_c0 = time.time(), self_cpu()
            rx0 = self.nic.rx()
            genome_bp_total = sum(g.lengths.values())
            rerouted_zero_logan_runs: list[str] = []

            # ---- Step A: Logan HTTP-200 Contigs + ka + glcore logan (A1/A2)
            logan_meta: dict | None = None
            unw_runs: dict[tuple[str, int, int], int] = {}
            unw_ua: dict[tuple[str, int, int], int] = {}
            logan_wire_bytes = 0
            logan_bam_target: Path | None = None
            if logan_runs:
                is_cross, cross_meta = detect_cross_species_rna(
                    self.gate, logan_runs, cache_dir=self.dir / "sra_meta"
                )
                mm2_flags = ["-ax", "splice", "-k14"] if is_cross else ["-ax", "splice"]
                logan_dir = cd / "logan"
                logan_tmp = cd / "logan_tmp"
                # #58: ALWAYS write Logan alignments to logan/logan_contigs.bam; never overwrite state/capped_short.bam!
                logan_bam_target = logan_dir / "logan_contigs.bam"
                try:
                    logan_meta = align_logan_contigs(
                        logan_runs=logan_runs,
                        genome_fa=self.genome_fa,
                        mm2_flags=mm2_flags,
                        glcore_bin=glc,
                        out_dir=logan_dir,
                        tmp_dir=logan_tmp,
                        bam_out=logan_bam_target,
                        threads=lp.threads,
                        nic=self.nic,
                        genome_bp=genome_bp_total,
                    )
                except RuntimeError as exc_logan:
                    if lp.logan_mode == "logan_only":
                        raise
                    if "stub" in str(exc_logan).lower() or "empty" in str(exc_logan).lower() or "0-record" in str(exc_logan).lower():
                        rerouted_zero_logan_runs = list(logan_runs)
                        for zr in rerouted_zero_logan_runs:
                            if zr not in fallback_short_runs:
                                fallback_short_runs.append(zr)
                        logan_runs = []
                        logan_bam_target = None
                        logan_meta = None
                    else:
                        raise
                if logan_meta is not None:
                    zero_rec_runs = [str(r) for r in (logan_meta.get("zero_record_runs") or [])]
                    if zero_rec_runs:
                        if lp.logan_mode == "logan_only":
                            raise RuntimeError(
                                f"Logan-only mode encountered zero-record Logan contig archive(s): {', '.join(zero_rec_runs)}"
                            )
                        rerouted_zero_logan_runs = list(zero_rec_runs)
                        zero_set = set(zero_rec_runs)
                        logan_runs = [r for r in logan_runs if r not in zero_set]
                        for zr in zero_rec_runs:
                            if zr not in fallback_short_runs:
                                fallback_short_runs.append(zr)
                        if not logan_runs:
                            for p_rm in (logan_bam_target, Path(f"{logan_bam_target}.bai")):
                                if p_rm is not None:
                                    p_rm.unlink(missing_ok=True)
                            logan_bam_target = None
                if rerouted_zero_logan_runs:
                    sra_stream_runs = [(r, False) for r in fallback_short_runs] + [(r, True) for r in long_runs]
                    new_runs = [(r, False) for r in fallback_short_runs if r not in info]
                    if new_runs:
                        info.update(self.run_info(runs=new_runs))
                    if fallback_short_runs:
                        _ensure_short_sieve_ready()
                if logan_meta is not None and logan_runs:
                    logan_meta["cross_species"] = cross_meta
                    logan_wire_bytes = int(logan_meta.get("wire_bytes", 0))
                    for k_proc, v_cpu in (logan_meta.get("cpu_by_proc_s") or {}).items():
                        cpu_by_tool[k_proc] = cpu_by_tool.get(k_proc, 0.0) + float(v_cpu)
                    self.log.append(
                        StageRecord(
                            name=f"arm{self.c_tag}_logan_align",
                            wall_seconds=float(logan_meta.get("wall_s", 0.0)),
                            cpu_user=float(logan_meta.get("total_cpu_s", 0.0)),
                            wire_bytes=logan_wire_bytes,
                            extra={
                                "n_logan_runs": len(logan_runs),
                                "mm2_flags": mm2_flags,
                                "cross_species": cross_meta,
                                "stream_stats": logan_meta.get("stream_stats", {}),
                            },
                        )
                    )
                    # v3(b): Keep ALL ka junctions intact (never drop single-contig unw_ua==1 bridges)
                    logan_added = ingest_per_run_junc_tsv(js, Path(logan_meta["junc_ka_path"]), g)
                    bins.ingest_tsv(Path(logan_meta["bins_ka_path"]))
                    unw_runs, unw_ua = load_unw_junc_stats(Path(logan_meta["junc_unw_path"]))
                    touched_logan: set[str] = set()
                    for k_j, nua_j, nt_j in logan_added:
                        reg = region_of.get(k_j)
                        if reg is None:
                            reg = _region(k_j, locked_introns, lidx_locked, lidx_uns)
                            region_of[k_j] = reg
                        if reg in (REGION_LOCKED_KNOWN, REGION_LOCKED_NOVEL):
                            for l in lidx_locked.containing(k_j[0], k_j[1], k_j[2]):
                                lo = locus_obs.setdefault(l.locus_id, {"keys": {}, "ua": 0, "total": 0})
                                lo["keys"][k_j] = lo["keys"].get(k_j, 0) + nua_j
                                lo["ua"] += nua_j
                                lo["total"] += nt_j
                                touched_logan.add(l.locus_id)
                    for lid in touched_logan:
                        if lid in saturated:
                            continue
                        lo = locus_obs[lid]
                        intr = locus_introns.get(lid, ())
                        ctg = locus_by_id[lid].contig
                        sup = [js.support((ctg, a, b)) for a, b in intr]
                        if not sup:
                            continue
                        st_l = LocusSaturationState(
                            locus_id=lid,
                            min_exon_depth=float(min(sup)),
                            min_junction_depth=int(min(sup)),
                            total_spliced_obs=lo["ua"],
                            singleton_junctions=sum(1 for c in lo["keys"].values() if c == 1),
                            has_unresolved_paralog_eq=(
                                lo["total"] > 0 and (lo["total"] - lo["ua"]) / lo["total"] > lp.paralog_multimap_frac
                            ),
                        )
                        if gate_sieve.locus_ready(st_l):
                            saturated[lid] = 0
                    if saturated and fallback_short_runs and mult is not None and cand_m is not None and bait is not None:
                        lst_0 = cd / "sieve" / "saturated_loci.txt"
                        lst_0.write_text("\n".join(sorted(saturated)) + "\n")
                        active = cd / "sieve" / "active_logan.bin"
                        act_stats = {**act_stats, **_activate(glc, active, mult=mult, cand=cand_m, loci=lst_0, bait=bait)}
                    mon.sample()

            # ---- Step B: SRA Streaming Loop on Fallback (HTTP-404) Short Runs + Long-Read Runs
            n_unsettled = sum(1 for l in s["loci"] if l.status == UNSETTLED)
            judge = GoodTuringUCBJudge(
                p0_stop=lp.p0_stop,
                ucb_c=lp.ucb_c,
                n_unsettled_loci=n_unsettled,
                max_novel_per_unsettled_1m=lp.max_novel_per_unsettled_1m,
                stop_window_waves=lp.stop_window_waves,
                replicate_slope_floor_per_10mb=lp.replicate_slope_floor_per_10mb,
                patience_waves=lp.patience_waves,
            )
            scout_p = self.dir / "seed" / "scout_waves.json"
            scout_data = json.loads(scout_p.read_text()) if scout_p.exists() else {}
            fb_short_set = set(fallback_short_runs)
            scout_waves = [
                w for w in scout_data.get("waves", [])
                if (w.get("run_id") in fb_short_set if lp.logan_mode in ("auto", "logan_only") else True)
            ]
            streams: dict[str, RunStream] = {}
            stream_failures: list[tuple[str, str]] = []
            short_budget_hit = False
            if sra_stream_runs:
                # Seed judge with unsettled/locked junctions already observed in Logan so GT-UCB focuses on uncovered introns
                if logan_runs and js.ua:
                    logan_wj = []
                    for k_j in js.ua:
                        sup_j = js.support(k_j)
                        if sup_j <= 0:
                            continue
                        motif_j, strand_j = js.motif.get(k_j, ("", "."))
                        reg_j = region_of.get(k_j) or _region(k_j, locked_introns, lidx_locked, lidx_uns)
                        region_of[k_j] = reg_j
                        logan_wj.append(WaveJunction((k_j[0], k_j[1], k_j[2], strand_j), sup_j, motif_j, reg_j))
                    if logan_wj:
                        judge.register_library("__logan__", "__logan__", is_long_read=False)
                        judge.ingest_wave_counts("__logan__", logan_wire_bytes, logan_wj, spots=0)
                        judge.libraries["__logan__"].halted = True
                        judge.libraries["__logan__"].halt_reason = "LOGAN_CONTIGS_COMPLETE"

                scout_consumed = {}
                for w in scout_waves:
                    scout_consumed[w["run_id"]] = max(scout_consumed.get(w["run_id"], 0), int(w.get("records_consumed", 0)))
                for r, lr in sra_stream_runs:
                    judge.register_library(r, r, is_long_read=lr)
                    if r in self._scout_streams and not self._scout_streams[r].eof:
                        streams[r] = self._scout_streams.pop(r)
                    elif r in self._local_streams:
                        st_loc = self._local_streams[r]["stream"]
                        if r in scout_consumed and st_loc.records_consumed < scout_consumed[r]:
                            st_loc.records_consumed = scout_consumed[r]
                        streams[r] = st_loc
                    else:
                        st_r = RunStream(r, info[r]["n_segments"], info[r]["total_spots"], cd / "logs", threads=4)
                        if r in scout_consumed:
                            st_r.records_consumed = scout_consumed[r]
                        streams[r] = st_r
                total_short_spots = sum(info[r]["total_spots"] for r, lr in sra_stream_runs if not lr)
                budget = int(lp.max_frac_spots * total_short_spots)
                cap_state = cd / "cap_state.u16"
                cap_state_long = cd / "cap_state_long.u16"
                if scout_data.get("cap_state") and Path(scout_data["cap_state"]).exists() and not cap_state.exists():
                    shutil.copy2(scout_data["cap_state"], cap_state)
                bait_extra_done = False

                def _record_wave(
                    rid_w: str, lr_w: bool, w_spots: int, w_wire: int, w_wall: float,
                    w_cpu: dict, w_sieve: dict, w_junc: dict, w_bam: str,
                    w_jtsv: str, w_btsv: str, is_scout: bool = False,
                ) -> None:
                    nonlocal spots_used, wave, active, act_stats, bait_extra_done
                    if not lr_w:
                        for k in sieve_tot:
                            sieve_tot[k] += w_sieve.get(k, 0)
                        spots_used += w_spots
                    for k, v in w_cpu.items():
                        cpu_by_tool[k] = cpu_by_tool.get(k, 0.0) + v
                    (wave_bams["long" if lr_w else "short"]).append(w_bam)
                    wj = js.ingest_tsv(rid_w, w_jtsv, g)
                    fallback_js.ingest_tsv(rid_w, w_jtsv, g)
                    bins.ingest_tsv(w_btsv)
                    wave_juncs = []
                    touched = set()
                    for k, nua, nt in wj:
                        motif, strand = js.motif.get(k, ("", "."))
                        reg = region_of.get(k)
                        if reg is None:
                            reg = _region(k, locked_introns, lidx_locked, lidx_uns)
                            region_of[k] = reg
                        if reg in (REGION_LOCKED_KNOWN, REGION_LOCKED_NOVEL):
                            for l in lidx_locked.containing(k[0], k[1], k[2]):
                                lo = locus_obs.setdefault(l.locus_id, {"keys": {}, "ua": 0, "total": 0})
                                lo["keys"][k] = lo["keys"].get(k, 0) + nua
                                lo["ua"] += nua
                                lo["total"] += nt
                                touched.add(l.locus_id)
                        wave_juncs.append(WaveJunction((k[0], k[1], k[2], strand), nua, motif, reg))
                    jrec = judge.ingest_wave_counts(rid_w, w_wire, wave_juncs, spots=w_spots)
                    newly = []
                    for lid in touched:
                        if lid in saturated:
                            continue
                        lo = locus_obs[lid]
                        intr = locus_introns.get(lid, ())
                        ctg = locus_by_id[lid].contig
                        sup = [js.support((ctg, a, b)) for a, b in intr]
                        if not sup:
                            continue
                        st = LocusSaturationState(
                            locus_id=lid,
                            min_exon_depth=float(min(sup)),
                            min_junction_depth=int(min(sup)),
                            total_spliced_obs=lo["ua"],
                            singleton_junctions=sum(1 for c in lo["keys"].values() if c == 1),
                            has_unresolved_paralog_eq=(
                                lo["total"] > 0 and (lo["total"] - lo["ua"]) / lo["total"] > lp.paralog_multimap_frac
                            ),
                        )
                        if gate_sieve.locus_ready(st):
                            saturated[lid] = wave
                            newly.append(lid)
                    if (
                        fallback_short_runs
                        and bait is not None
                        and not bait_extra_done
                        and all(l.waves_completed > 0 or l.halted for l in judge.libraries.values())
                    ):
                        bait_extra_done = True
                        hs = [h for h in bins.hotspots(frac=lp.hotspot_frac) if not lidx_all.overlapping(h[0], h[1], h[2])]
                        if hs:
                            with open(bait, "a") as fh:
                                for c, a, b, n in hs:
                                    fh.write(f">hotspot_{c}_{a}\n{g.fetch(c, max(0, a - 500), b + 500)}\n")
                        act_stats["hotspots"] = [list(h) for h in hs]
                        newly.append("__bait__")
                    if newly and fallback_short_runs and mult is not None and cand_m is not None and bait is not None:
                        lst = cd / "sieve" / "saturated_loci.txt"
                        lst.write_text("\n".join(sorted(saturated)) + "\n")
                        active = cd / "sieve" / f"active_{wave + 1}.bin"
                        act_stats = {**act_stats, **_activate(glc, active, mult=mult, cand=cand_m, loci=lst, bait=bait)}
                    row = {
                        "stage": "scout_wave" if is_scout else "wave",
                        "wave": wave,
                        "run_id": rid_w,
                        "long_read": lr_w,
                        "spots": w_spots,
                        "wire_bytes": w_wire,
                        "wall_s": w_wall,
                        "cpu": w_cpu,
                        "sieve": w_sieve,
                        "junc": w_junc,
                        "saturated_loci": len(saturated),
                        "newly_saturated": len([x for x in newly if x != "__bait__"]),
                        "active_syncmers": act_stats.get("active", 0),
                        **{f"judge_{k}": v for k, v in jrec.items() if k != "run_id"},
                        "spots_used_short": spots_used,
                    }
                    tele_rows.append(row)
                    with open(cd / "waves.jsonl", "a") as fh:
                        fh.write(json.dumps(row, default=str) + "\n")
                    wave += 1

                for sw in scout_waves:
                    _record_wave(
                        sw["run_id"], bool(sw.get("long_read", False)), int(sw["spots"]), int(sw.get("wire_bytes", 0)),
                        float(sw.get("wall_s", 0.0)), sw.get("cpu", {}), sw.get("sieve", {}), sw.get("junc", {}),
                        sw["bam"], sw["junc_tsv"], sw["bins_tsv"], is_scout=True,
                    )

                while True:
                    rid = judge.select_next_library_gt_ucb()
                    if rid is None:
                        if short_budget_hit and not long_runs:
                            stop_reason = f"MAX_BUDGET_{lp.max_frac_spots}"
                        elif short_budget_hit:
                            stop_reason = f"SHORT_MAX_BUDGET_{lp.max_frac_spots}_LONG_COMPLETE"
                        else:
                            stop_reason = "ALL_LIBRARIES_HALTED_OR_EXHAUSTED"
                        break
                    if judge.saturated():
                        stop_reason = (
                            f"SATURATED_P0_{lp.p0_stop}_RATE_{lp.max_novel_per_unsettled_1m}"
                            if n_unsettled > 0
                            else f"UNSETTLED_P0_BELOW_{lp.p0_stop}"
                        )
                        break
                    lr = info[rid]["long_read"]
                    if not lr and spots_used >= budget:
                        short_budget_hit = True
                        for r_s, lr_s in sra_stream_runs:
                            if not lr_s and r_s in judge.libraries and not judge.libraries[r_s].halted:
                                judge.libraries[r_s].halted = True
                                judge.libraries[r_s].halt_reason = f"MAX_BUDGET_{lp.max_frac_spots}"
                        continue
                    tag = f"w{wave:03d}_{rid}"
                    use_nic_w = None if rid in self._local_streams else self.nic
                    try:
                        if lr:
                            res = align_slice_long(
                                streams[rid], lp.long_wave_spots, mmi, waves_dir, tag,
                                cap_state_long, lp.long_caps,
                                threads=lp.hisat_threads, nic=use_nic_w,
                                stranded=self._is_long_run_stranded(rid),
                                genome_bp=genome_bp_total,
                            )
                        elif getattr(streams[rid], "segs", 2) == 1:
                            res = align_slice_short_se(
                                streams[rid], lp.wave_spots, idx, waves_dir, tag, cap_state, lp.caps,
                                threads=lp.hisat_threads, nic=use_nic_w,
                            )
                        else:
                            res = align_slice_short(
                                streams[rid], lp.wave_spots, idx, waves_dir, tag, active, cap_state, lp.caps,
                                threads=lp.hisat_threads, nic=use_nic_w,
                                min_hits=lp.sieve_min_hits, max_miss=lp.sieve_max_miss,
                            )
                    except Exception as exc_w:
                        if getattr(streams[rid], "stream_failed", False):
                            judge.libraries[rid].halted = True
                            judge.libraries[rid].halt_reason = "STREAM_FAILED"
                            stream_failures.append((rid, str(exc_w)))
                            continue
                        raise
                    if res.spots == 0:
                        for k, v in res.cpu.items():
                            cpu_by_tool[k] = cpu_by_tool.get(k, 0.0) + v
                        judge.libraries[rid].halted = True
                        if getattr(streams[rid], "stream_failed", False):
                            judge.libraries[rid].halt_reason = "STREAM_FAILED"
                            stream_failures.append((rid, "stream terminated prematurely before expected total_spots"))
                        else:
                            judge.libraries[rid].halt_reason = "EXHAUSTED"
                        continue
                    _record_wave(
                        rid, lr, res.spots, res.wire_bytes, res.wall, res.cpu, res.sieve, res.junc,
                        res.bam, res.junc_tsv, res.bins_tsv, is_scout=False,
                    )
                for st_ in streams.values():
                    st_.close()
                    cpu_by_tool["xsra"] = cpu_by_tool.get("xsra", 0.0) + st_.cpu_user + st_.cpu_sys
            else:
                stop_reason = f"LOGAN_HTTP200_ALL_{len(logan_runs)}_RUNS" if logan_runs else "NO_RNA_RUNS"

            if self._local_bams:
                bam_cap_state = cd / "bam_cap_state.u16"
                bam_cap_state_long = cd / "bam_cap_state_long.u16"
                for bi, binfo in enumerate(self._local_bams):
                    is_lr_b = bool(binfo.get("long_read", False))
                    bres = ingest_bam_to_evidence(
                        Path(binfo["bam"]),
                        binfo["run_id"],
                        waves_dir,
                        f"bam_{bi:03d}_{binfo['run_id']}",
                        bam_cap_state_long if is_lr_b else bam_cap_state,
                        lp.long_caps if is_lr_b else lp.caps,
                        is_long_read=is_lr_b,
                        genome_lengths=g.lengths,
                    )
                    wave_bams["long" if is_lr_b else "short"].append(bres.bam)
                    wj = js.ingest_tsv(binfo["run_id"], bres.junc_tsv, g)
                    bins.ingest_tsv(bres.bins_tsv)
                    for k, _nua, _nt in wj:
                        if k not in region_of:
                            region_of[k] = _region(k, locked_introns, lidx_locked, lidx_uns)
                    for k, v in bres.cpu.items():
                        cpu_by_tool[k] = cpu_by_tool.get(k, 0.0) + v
                if stop_reason == "NO_RNA_RUNS":
                    stop_reason = f"HISTORY_BAMS_{len(self._local_bams)}"

            if stream_failures:
                ok_stream_libs = [
                    r
                    for r, _ in sra_stream_runs
                    if r in judge.libraries
                    and judge.libraries[r].halt_reason != "STREAM_FAILED"
                    and (judge.libraries[r].waves_completed > 0 or (r in streams and streams[r].spots_consumed > 0))
                ]
                if not ok_stream_libs and not logan_runs and not self._local_bams:
                    fail_msg = ", ".join(f"{r}: {m}" for r, m in stream_failures[:3])
                    raise RuntimeError(
                        f"All RNA streaming libraries failed ({len(stream_failures)} failed: {fail_msg})"
                    )

            loop_wall = time.time() - loop_t0
            cpu_by_tool["geneloop_python"] = self_cpu() - loop_c0
            scout_wire = sum(int(w.get("wire_bytes", 0)) for w in scout_waves)
            has_remote_streams = any(r not in self._local_streams for r, _ in sra_stream_runs)
            nic_loop_wire = max(0, self.nic.rx() - rx0) if (has_remote_streams or logan_wire_bytes > 0) else 0
            wire = scout_wire + max(logan_wire_bytes, nic_loop_wire)
            mon.sample()

            # ---- Step C: Compact retained alignments + StringTie default mode + v3 chimera splitter
            t1 = time.time()
            merged = ev_dir / "capped_short.bam"
            merged_bai = Path(f"{merged}.bai")
            post_cpu = 0.0
            merged_long = ev_dir / "capped_long.bam" if wave_bams["long"] else None
            if merged_long:
                post_cpu += _cat_sort(wave_bams["long"], merged_long, lp.threads)

            gtf = cd / "stringtie.gtf"
            v3_tx_split_stats: dict | None = None
            if not logan_runs:
                if wave_bams["short"]:
                    post_cpu += _cat_sort(wave_bams["short"], merged, lp.threads)
                mon.sample()
                if merged.exists():
                    _, _, u, s_ = run_cmd(["stringtie", "-p", str(lp.threads), "-o", str(gtf), str(merged)])
                    post_cpu += u + s_
                    rna_tx_short = read_models(gtf, fmt="gtf")
                else:
                    gtf.write_text("")
                    rna_tx_short = []
            elif not wave_bams["short"]:
                assert logan_bam_target is not None
                if logan_bam_target.resolve() != merged.resolve():
                    os.replace(logan_bam_target, merged)
                    bai_src = Path(f"{logan_bam_target}.bai")
                    if bai_src.exists():
                        os.replace(bai_src, merged_bai)
                    else:
                        _, _, ui, si = run_cmd(["samtools", "index", str(merged)])
                        post_cpu += ui + si
                logan_raw_gtf = cd / "stringtie_logan_raw.gtf"
                # A2: StringTie default mode (NO -L!)
                _, _, u, s_ = run_cmd(["stringtie", "-l", "STLOG", "-p", str(lp.threads), "-o", str(logan_raw_gtf), str(merged)])
                post_cpu += u + s_
                tx_logan = read_models(logan_raw_gtf, fmt="gtf")
                if len(logan_runs) >= LOGAN_WIDE_COHORT_MIN_RUNS:
                    safe_mp_ref = build_safe_locked_miniprot_ref_models(s)
                    tx_logan, v3_tx_split_stats = split_safe_ortholog_chimeras(tx_logan, safe_mp_ref)
                    _write_gtf_transcripts(tx_logan, gtf)
                    logan_raw_gtf.unlink(missing_ok=True)
                else:
                    os.replace(logan_raw_gtf, gtf)
                rna_tx_short = tx_logan
            else:
                assert logan_bam_target is not None and logan_bam_target.resolve() != merged.resolve()
                fb_short_bam = cd / "logan" / "fallback_short.bam"
                post_cpu += _cat_sort(wave_bams["short"], fb_short_bam, lp.threads)
                mon.sample()
                logan_raw_gtf = cd / "stringtie_logan_raw.gtf"
                fb_raw_gtf = cd / "stringtie_fb_raw.gtf"
                _, _, u1, s1 = run_cmd(["stringtie", "-l", "STLOG", "-p", str(lp.threads), "-o", str(logan_raw_gtf), str(logan_bam_target)])
                _, _, u2, s2 = run_cmd(["stringtie", "-l", "STFB", "-p", str(lp.threads), "-o", str(fb_raw_gtf), str(fb_short_bam)])
                post_cpu += u1 + s1 + u2 + s2
                tx_logan = read_models(logan_raw_gtf, fmt="gtf")
                tx_fb = read_models(fb_raw_gtf, fmt="gtf")
                if len(logan_runs) >= LOGAN_WIDE_COHORT_MIN_RUNS:
                    safe_mp_ref = build_safe_locked_miniprot_ref_models(s)
                    tx_logan, v3_tx_split_stats = split_safe_ortholog_chimeras(tx_logan, safe_mp_ref)
                rna_tx_short = list(tx_logan) + list(tx_fb)
                _write_gtf_transcripts(rna_tx_short, gtf)
                logan_raw_gtf.unlink(missing_ok=True)
                fb_raw_gtf.unlink(missing_ok=True)
                # Merge sorted Logan + fallback short BAMs into canonical capped_short.bam and drop temporary copies
                _, _, um, sm = run_cmd(
                    ["samtools", "merge", "-f", "-@", str(min(16, lp.threads)), "-o", str(merged), str(logan_bam_target), str(fb_short_bam)]
                )
                _, _, ui, si = run_cmd(["samtools", "index", "-@", str(min(16, lp.threads)), str(merged)])
                post_cpu += um + sm + ui + si
                protected_paths = {merged.resolve(), merged_bai.resolve()}
                for p_tmp in (logan_bam_target, Path(f"{logan_bam_target}.bai"), fb_short_bam, Path(f"{fb_short_bam}.bai")):
                    if p_tmp.resolve() not in protected_paths:
                        p_tmp.unlink(missing_ok=True)

            waves_root = waves_dir.resolve()
            protected_final = {merged.resolve(), merged_bai.resolve()}
            if merged_long is not None:
                protected_final.add(merged_long.resolve())
                protected_final.add(Path(f"{merged_long}.bai").resolve())
            for b in wave_bams["short"] + wave_bams["long"]:
                bp = Path(b).resolve()
                if bp not in protected_final and (bp == waves_root or waves_root in bp.parents):
                    bp.unlink(missing_ok=True)

            rna_tx = list(rna_tx_short)
            if merged_long:
                _, _, u, s_ = run_cmd(["stringtie", "-L", "-p", str(lp.threads), "-o", str(cd / "stringtie_long.gtf"), str(merged_long)])
                post_cpu += u + s_
                rna_tx += _read_long_read_models(cd / "stringtie_long.gtf")
            st_wall = time.time() - t1

            # v3(c): Wide-cohort (>=32 Logan runs) unweighted lncRNA gate + fallback >=4 rescue
            if len(logan_runs) >= LOGAN_WIDE_COHORT_MIN_RUNS:
                min_lnc_runs, min_lnc_ua = compute_logan_lnc_thresholds(len(logan_runs))
                lnc_eligible_introns: set[tuple[str, int, int]] | None = compute_eligible_lnc_introns(
                    unw_runs, unw_ua, min_lnc_runs=min_lnc_runs, min_lnc_ua=min_lnc_ua, fb_js=fallback_js, fb_min_ua=4
                )
                logan_lnc_gate_meta = {
                    "enabled": True,
                    "n_logan_runs": len(logan_runs),
                    "min_lnc_runs": min_lnc_runs,
                    "min_lnc_ua": min_lnc_ua,
                }
            else:
                lnc_eligible_introns = None
                logan_lnc_gate_meta = {
                    "enabled": False,
                    "n_logan_runs": len(logan_runs),
                    "min_lnc_runs": 0,
                    "min_lnc_ua": 0,
                }

            bf_info = self._ensure_orphan_backfill(cd, ev_dir, rna_tx, js, s)
            post_cpu += float(bf_info.get("backfill_cpu_s", 0.0))
            nc_info = self.ensure_structural_ncrna()
            post = self._post_synth_c(
                js, bins, rna_tx, merged, region_of, saturated, s,
                backfill_info=bf_info,
                lnc_eligible_introns=lnc_eligible_introns,
                logan_lnc_gate_meta=logan_lnc_gate_meta,
            )
            mon.sample()
        sieve_bypass = 100.0 * (sieve_tot["pairs_dropped_locus"] + sieve_tot["pairs_dropped_bait"]) / max(1, sieve_tot["pairs_in"])
        total_spots_all = sum(info[r]["total_spots"] for r, _ in sra_stream_runs)
        spots_all = sum(streams[r].spots_consumed for r, _ in sra_stream_runs if r in streams)
        summary = {
            "waves": wave,
            "stop_reason": stop_reason,
            "spots_streamed_short": spots_used,
            "spots_streamed_all": spots_all,
            "total_spots_all": total_spots_all,
            "pct_archive_streamed": (100.0 * spots_all / max(1, total_spots_all)) if total_spots_all > 0 else 0.0,
            "wire_bytes": wire,
            "restream_restarts": sum(st_.restarts for st_ in streams.values()),
            "logan": {
                "logan_mode": lp.logan_mode,
                "n_logan_runs": len(logan_runs),
                "logan_runs": logan_runs,
                "fallback_short_runs": fallback_short_runs,
                "skipped_short_runs": list(logan_probe.get("skipped_short_runs") or []),
                "zero_record_rerouted_runs": rerouted_zero_logan_runs,
                "long_runs": long_runs,
                "align": logan_meta,
                "v3_wide_cohort_active": len(logan_runs) >= LOGAN_WIDE_COHORT_MIN_RUNS,
                "v3_tx_split": v3_tx_split_stats,
                "v3_lnc_gate": {**logan_lnc_gate_meta, **(post["synth_stats"].get("logan_lnc_gate") or {})},
            },
            "sieve": {**sieve_tot, "bypass_pct": sieve_bypass, "index": mult_stats, "activation": {k: v for k, v in act_stats.items() if k != "hotspots"},
                      "hotspots": act_stats.get("hotspots", [])},
            "saturated_loci": len(saturated),
            "locked_loci": len(locked),
            "judge_final": judge.history[-1] if judge.history else {},
            "library_state": {
                r: {
                    "waves": l.waves_completed,
                    "halted": l.halted,
                    "reason": l.halt_reason,
                    "spots": streams[r].spots_consumed if r in streams else 0,
                }
                for r, l in judge.libraries.items()
                if r != "__logan__"
            },
            "loop_wall_s": loop_wall,
            "cpu_by_tool_s": cpu_by_tool,
            "sieve_index_wall_s": sieve_index_wall,
            "post_stringtie_wall_s": st_wall,
            "post_cpu_s": post_cpu,
            "py_post_cpu_s": post["py_cpu_s"],
            "zoom_wall_s": post["zoom_wall_s"],
            "synth_wall_s": synth_wall if "synth_wall" in locals() else post["synth_wall_s"],
            "state_wall_s": post["state_wall_s"],
            "synth_stats": post["synth_stats"],
            "synth_stats_nozoom": post["synth_stats_nozoom"],
            "structural_ncrna": nc_info,
            "microexon_rescues": post["microexon_rescues"],
            "peak_scratch_bytes": mon.peak,
            "retained_state_bytes": dir_bytes(ev_dir),
            "retained_state_files": {p.name: p.stat().st_size for p in ev_dir.iterdir() if p.is_file()},
        }
        summary["synth_wall_s"] = post["synth_wall_s"]
        from .state import write_telemetry

        tele_rows.append({"stage": "summary", **{k: v for k, v in summary.items() if not isinstance(v, (dict, list))}})
        write_telemetry(ev_dir / "telemetry.parquet", tele_rows)
        summary["retained_state_bytes"] = dir_bytes(ev_dir)
        self.log.append(StageRecord(name=f"arm{self.c_tag}_loop", wall_seconds=loop_wall, cpu_user=sum(cpu_by_tool.values()), wire_bytes=wire,
                                    extra={"waves": wave, "stop_reason": stop_reason}))
        self.log.append(StageRecord(name=f"arm{self.c_tag}_post", wall_seconds=st_wall + post["zoom_wall_s"] + post["synth_wall_s"] + post["state_wall_s"],
                                    cpu_user=post_cpu + post["py_cpu_s"]))
        summary["loop_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.lp).items()}
        (cd / f"arm{self.c_tag}_summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary


    def _load_seed_zoom_windows(self, org_set: set[str]) -> list[tuple[str, int, int]]:
        sd = self.dir / "seed"
        zw_p = sd / "zoom_windows.json"
        if zw_p.exists():
            try:
                raw = json.loads(zw_p.read_text())
                ws = raw.get("windows", raw) if isinstance(raw, dict) else raw
                return [(str(c), int(a), int(b)) for c, a, b in ws]
            except (OSError, ValueError, TypeError):
                pass
        if not self.lp.cpu_focaloop_tiberius:
            return [(c, 0, int(L)) for c, L in self.genome.lengths.items() if c not in org_set]
        return []

    def _compute_backfill_input_key(self, cd: Path, seed_wins: list[tuple[str, int, int]]) -> str:
        """X3: Deterministic cache key over StringTie GTFs, seed zoom windows, genome, model/image, and backfill params."""
        st_gtf = cd / "stringtie.gtf"
        st_long_gtf = cd / "stringtie_long.gtf"
        zw_p = self.dir / "seed" / "zoom_windows.json"
        payload = {
            "backfill_logic_version": BACKFILL_LOGIC_VERSION,
            "genome_identity": self._genome_identity_digest(),
            "tiberius_model_cfg": str(self.gate.get("tiberius_model_cfg", "")),
            "tiberius_image": seedmod.TIBERIUS_IMAGE,
            "tiberius_image_id": seedmod._tiberius_image_id(),
            "stringtie_gtf_sha256": _sha256_file(st_gtf) if st_gtf.exists() else "",
            "stringtie_long_gtf_sha256": _sha256_file(st_long_gtf) if st_long_gtf.exists() else "",
            "seed_zoom_windows_file_sha256": _sha256_file(zw_p) if zw_p.exists() else "",
            "seed_zoom_windows_sha256": hashlib.sha256(
                json.dumps([list(w) for w in seed_wins]).encode("utf-8")
            ).hexdigest(),
            "max_backfill_bp": max(0, int(self.lp.max_backfill_bp)),
            "trim_spacer_bridges": bool(self.lp.trim_spacer_bridges),
            "tiberius_spacer_bp": int(getattr(self.lp, "tiberius_spacer_bp", 4_500)),
            "thresholds": {
                "min_exons": 4,
                "max_seed_ov_frac": 0.50,
                "min_mp_pos": 0.50,
                "min_mp_cov": 0.80,
                "min_junc_sup": 3,
                "min_good_introns": 3,
                "max_softmask": 0.45,
                "min_orf_aa": 120,
                "pad_bp": 5_000,
                "merge_gap_bp": 5_000,
                "erode_bp": 2_000,
                "min_sub_len": 2_000,
                "raw_junc_cluster_gap_bp": 5_000,
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_backfill_windows_payload(raw_obj: object) -> tuple[str, bool, list[tuple[str, int, int]]]:
        """Parse backfill_windows.json supporting both v1 dict {'input_key', 'enabled', 'windows'} and legacy list."""
        if isinstance(raw_obj, dict):
            ws = [(str(c), int(a), int(b)) for c, a, b in (raw_obj.get("windows") or ())]
            return str(raw_obj.get("input_key", "")), bool(raw_obj.get("enabled", True)), ws
        if isinstance(raw_obj, list):
            return "", True, [(str(c), int(a), int(b)) for c, a, b in raw_obj]
        raise ValueError("Invalid backfill_windows.json structure")

    @staticmethod
    def _span_blocked_by_index(
        idx: LocusIndex,
        ctg: str,
        strand: str,
        intervals: list[tuple[int, int]],
        min_frac: float = 0.50,
    ) -> bool:
        """True if same-strand spans in ``idx`` cover >= ``min_frac`` of ``intervals``."""
        tot_bp = sum(b - a for a, b in intervals)
        if tot_bp <= 0:
            return False
        lo, hi = intervals[0][0], intervals[-1][1]
        cov_ivs = [
            (ia, ib)
            for ia, ib, st in idx.overlapping(ctg, lo, hi)
            if st == strand
        ]
        if not cov_ivs:
            return False
        merged_cov = merge_intervals(cov_ivs)
        ov = sum(
            max(0, min(eb, cb) - max(ea, ca))
            for ea, eb in intervals
            for ca, cb in merged_cov
        )
        return ov >= min_frac * tot_bp

    @staticmethod
    def _collect_raw_junction_orphan_windows(
        js: JunctionStore,
        g: Genome,
        org_set: set[str],
        bait_by_ctg: dict[str, list[tuple[int, int]]],
        seed_idx: LocusIndex,
        safe_idx: LocusIndex,
        intact_mp_idx: LocusIndex,
        cand_ivs_by_ctg: dict[str, list[tuple[int, int, int]]],
        rej_introns_by_cs: dict[tuple[str, str], set[tuple[int, int]]] | None = None,
        good_rna_juncs_by_cs: dict[tuple[str, str], set[tuple[int, int]]] | None = None,
    ) -> None:
        """Cluster raw >=3-intron canonical coding-plausible junctions outside seed/safe/intact-protein loci."""
        eff_runs = set(js.run_ids) or {r for d in js.ua.values() for r, v in d.items() if v > 0}
        n_libs = len(eff_runs)
        by_cs: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
        for ctg, jlist in js.by_contig(min_support=3, canonical_only=True).items():
            if ctg in org_set or ctg not in g.lengths:
                continue
            for ja, jb, jsup, jst in jlist:
                if jst in ("+", "-"):
                    by_cs.setdefault((ctg, jst), []).append((ja, jb, jsup))
        for (ctg, strand), juncs in sorted(by_cs.items()):
            juncs.sort()
            clusters: list[list[tuple[int, int, int]]] = []
            cur: list[tuple[int, int, int]] = []
            cur_end = -1
            for ja, jb, jsup in juncs:
                if cur and ja - cur_end > 5_000:
                    clusters.append(cur)
                    cur = []
                    cur_end = -1
                cur.append((ja, jb, jsup))
                cur_end = max(cur_end, jb)
            if cur:
                clusters.append(cur)
            L = g.lengths[ctg]
            rej_set = (rej_introns_by_cs or {}).get((ctg, strand), set())
            good_set = (good_rna_juncs_by_cs or {}).get((ctg, strand), set())
            rej_only = rej_set - good_set
            for cl in clusters:
                if len(cl) < 3:
                    continue
                cl_jset = {(ja, jb) for ja, jb, _s in cl}
                if len(cl_jset - rej_only) < 3:
                    continue
                if n_libs >= 2 and not any(js.n_runs_supporting((ctg, ja, jb)) >= 2 for ja, jb, _s in cl) and max(x[2] for x in cl) < 8:
                    continue
                c_lo, c_hi = cl[0][0], max(x[1] for x in cl)
                span = max(1, c_hi - c_lo)
                if any(ba < c_hi and c_lo < bb for ba, bb in bait_by_ctg.get(ctg, ())):
                    continue
                ov_seed = sum(max(0, min(c_hi, wb) - max(c_lo, wa)) for wa, wb in seed_idx.overlapping(ctg, c_lo, c_hi))
                if ov_seed >= 0.50 * span:
                    continue
                chain: list[tuple[int, int, int]] = []
                for ja, jb, jsup in sorted(cl, key=lambda x: (-x[2], x[0], x[1])):
                    if all(jb <= ea or eb <= ja for ea, eb, _s in chain):
                        chain.append((ja, jb, jsup))
                chain.sort()
                internal_exons = [
                    (chain[i][1], chain[i + 1][0])
                    for i in range(len(chain) - 1)
                    if chain[i + 1][0] > chain[i][1]
                ]
                cl_exons = merge_intervals(
                    [(max(0, chain[0][0] - 100), chain[0][0])]
                    + internal_exons
                    + [(chain[-1][1], min(L, chain[-1][1] + 100))]
                ) if chain else [(c_lo, c_hi)]
                if GateRun._span_blocked_by_index(safe_idx, ctg, strand, cl_exons):
                    continue
                if GateRun._span_blocked_by_index(intact_mp_idx, ctg, strand, cl_exons):
                    continue
                if (
                    len(chain) >= 3
                    and len(internal_exons) >= 2
                    and sum(eb - ea for ea, eb in internal_exons) >= 360
                    and max_open_orf_aa(g.spliced(ctg, strand, internal_exons)) < 60
                ):
                    continue
                anchor_ivs = merge_intervals(
                    [(max(0, a - 100), a) for a, _b, _s in cl if a > 0]
                    + [(b, min(L, b + 100)) for _a, b, _s in cl if b < L]
                )
                if anchor_ivs and _softmask_frac(g, ctg, tuple(anchor_ivs)) > 0.45:
                    continue
                cl_n_runs = max((js.n_runs_supporting((ctg, ja, jb)) for ja, jb, _s in cl), default=0)
                cl_max_sup = max(x[2] for x in cl)
                raw_prio = min(900, cl_n_runs * 50 + min(200, cl_max_sup))
                cand_ivs_by_ctg.setdefault(ctg, []).append(
                    (max(0, c_lo - 5_000), min(L, c_hi + 5_000), raw_prio)
                )

    def _ensure_orphan_backfill(
        self,
        cd: Path,
        ev_dir: Path,
        rna_tx: list[Transcript],
        js: JunctionStore,
        s: dict,
    ) -> dict:
        """B3/B4/X3 post-loop orphan RNA + long-read + raw-junction Tiberius backfill honoring max_backfill_bp."""
        ev_dir.mkdir(parents=True, exist_ok=True)
        # X3(a): Check enable switch FIRST; never merge or clobber valid cached backfill when disabled.
        if not self.lp.cpu_focaloop_tiberius or not self.lp.enable_orphan_backfill:
            return {
                "enabled": False,
                "backfill_bp": 0,
                "backfill_n_windows": 0,
                "backfill_workers": 0,
                "backfill_wall_s": 0.0,
                "backfill_cpu_s": 0.0,
            }

        bf_gtf = ev_dir / "tiberius_backfill.gtf"
        bf_wins_p = ev_dir / "backfill_windows.json"
        g = self.genome
        sd = self.dir / "seed"
        rrna_spans: list[tuple[str, int, int]] = []
        org_set: set[str] = set()
        if (self.dir / "genome" / "genome_meta.json").exists():
            rrna_spans, org_set = self._bait_spans_and_fa(sd, s.get("hits", ()), tib=s.get("tiberius", ()), out_fa=None)

        seed_wins = self._load_seed_zoom_windows(org_set)
        expected_key = self._compute_backfill_input_key(cd, seed_wins)
        if bf_gtf.exists() and bf_wins_p.exists():
            try:
                cached_key, cached_en, ws_cached = self._parse_backfill_windows_payload(json.loads(bf_wins_p.read_text()))
            except (OSError, ValueError, TypeError):
                cached_key, cached_en, ws_cached = "", False, []
            if cached_en and cached_key and cached_key == expected_key:
                return {
                    "enabled": True,
                    "input_key": expected_key,
                    "backfill_bp": sum(int(b) - int(a) for _c, a, b in ws_cached),
                    "backfill_n_windows": len(ws_cached),
                    "backfill_workers": 0,
                    "backfill_wall_s": 0.0,
                    "backfill_cpu_s": 0.0,
                    "cached": True,
                }

        if not rna_tx and not js.ua:
            bf_gtf.write_text("")
            _write_json_atomic(bf_wins_p, {"input_key": expected_key, "enabled": True, "windows": []})
            return {
                "enabled": True,
                "input_key": expected_key,
                "backfill_bp": 0,
                "backfill_n_windows": 0,
                "backfill_workers": 0,
                "backfill_wall_s": 0.0,
                "backfill_cpu_s": 0.0,
            }
        seed_by_ctg: dict[str, list[tuple[int, int]]] = {}
        for c, a, b in seed_wins:
            seed_by_ctg.setdefault(c, []).append((a, b))
        for c in list(seed_by_ctg):
            seed_by_ctg[c] = merge_intervals(seed_by_ctg[c])
        seed_idx = LocusIndex([(c, a, b, (a, b)) for c, ivs in seed_by_ctg.items() for a, b in ivs])

        safe_loci = [l for l in s.get("loci", ()) if getattr(l, "category", "") == "safe_locked_miniprot"]
        safe_exon_items: list[tuple[str, int, int, tuple[int, int, str]]] = []
        for l in safe_loci:
            lm = getattr(l, "locked_model", None)
            if lm is not None and lm.cds:
                s_blocks = merge_intervals(lm.cds)
            elif getattr(l, "hits", None):
                best_h = max(l.hits, key=lambda h: (h.quality, h.n_rank1, h.n_proteins))
                s_blocks = merge_intervals(best_h.cds) if best_h.cds else [(l.start, l.end)]
            else:
                s_blocks = [(l.start, l.end)]
            for ca, cb in s_blocks:
                safe_exon_items.append((l.contig, ca, cb, (ca, cb, l.strand)))
        safe_idx = LocusIndex(safe_exon_items)
        intact_mp_idx = LocusIndex(
            [
                (h.contig, ca, cb, (ca, cb, h.strand))
                for h in s.get("hits", ())
                if h.quality >= 0.50 and h.positive >= 0.50 and h.coverage >= 0.80 and h.frameshift == 0 and h.stop_codon == 0
                for ca, cb in merge_intervals(h.cds)
            ]
        )
        bait_by_ctg: dict[str, list[tuple[int, int]]] = {}
        for c, a, b in rrna_spans:
            bait_by_ctg.setdefault(c, []).append((a, b))

        cand_ivs_by_ctg: dict[str, list[tuple[int, int, int]]] = {}
        rej_introns_by_cs: dict[tuple[str, str], set[tuple[int, int]]] = {}
        good_rna_juncs_by_cs: dict[tuple[str, str], set[tuple[int, int]]] = {}
        for t in rna_tx:
            if t.strand not in ("+", "-") or t.contig in org_set or t.contig not in g.lengths:
                continue
            exons = merge_intervals(t.exons)
            if len(exons) < 2:
                continue
            if _softmask_frac(g, t.contig, tuple(exons)) > 0.45 or max_open_orf_aa(g.spliced(t.contig, t.strand, exons)) < 120:
                rej_introns_by_cs.setdefault((t.contig, t.strand), set()).update(introns_of(exons))
                continue
            if len(exons) < 4:
                continue
            t_lo, t_hi = exons[0][0], exons[-1][1]
            span = max(1, t_hi - t_lo)
            if any(ba < t_hi and t_lo < bb for ba, bb in bait_by_ctg.get(t.contig, ())):
                continue
            ov_seed = sum(max(0, min(t_hi, wb) - max(t_lo, wa)) for wa, wb in seed_idx.overlapping(t.contig, t_lo, t_hi))
            if ov_seed >= 0.50 * span:
                continue
            if self._span_blocked_by_index(safe_idx, t.contig, t.strand, exons):
                continue
            if self._span_blocked_by_index(intact_mp_idx, t.contig, t.strand, exons):
                continue
            good_intr_sups = [
                js.support((t.contig, ex_a[1], ex_b[0]))
                for ex_a, ex_b in zip(exons, exons[1:])
                if js.canonical((t.contig, ex_a[1], ex_b[0]))
                and js.strand((t.contig, ex_a[1], ex_b[0])) == t.strand
                and js.support((t.contig, ex_a[1], ex_b[0])) >= 3
            ]
            is_long = t.attrs.get("long_read") == "1" or "long" in str(t.source or "").lower()
            cov = float(t.attrs.get("cov", 0) or 0)
            long_canon_n = (
                sum(
                    1
                    for ex_a, ex_b in zip(exons, exons[1:])
                    if junction_motif(g, t.contig, ex_a[1], ex_b[0]) in {(m, t.strand) for m in CANONICAL}
                )
                if (is_long and cov >= 2.0 and len(good_intr_sups) < 3)
                else 0
            )
            if len(good_intr_sups) < 3 and long_canon_n < 3:
                continue
            good_rna_juncs_by_cs.setdefault((t.contig, t.strand), set()).update(introns_of(exons))
            L = g.lengths[t.contig]
            n_runs_max = max((js.n_runs_supporting((t.contig, ex_a[1], ex_b[0])) for ex_a, ex_b in zip(exons, exons[1:])), default=0)
            raw_sup = max(good_intr_sups) if good_intr_sups else max(3, int(round(cov)))
            prio_sup = 1_000 + min(400, n_runs_max * 50 + min(200, raw_sup)) + (100 if is_long else 0)
            cand_ivs_by_ctg.setdefault(t.contig, []).append((max(0, t_lo - 5_000), min(L, t_hi + 5_000), prio_sup))

        self._collect_raw_junction_orphan_windows(
            js,
            g,
            org_set,
            bait_by_ctg,
            seed_idx,
            safe_idx,
            intact_mp_idx,
            cand_ivs_by_ctg,
            rej_introns_by_cs=rej_introns_by_cs,
            good_rna_juncs_by_cs=good_rna_juncs_by_cs,
        )

        scored_wins: list[tuple[int, str, int, int]] = []
        for ctg, raw_ivs in sorted(cand_ivs_by_ctg.items()):
            raw_ivs.sort(key=lambda x: (x[0], x[1]))
            merged_w: list[list[int]] = []
            for a, b, sup in raw_ivs:
                if merged_w and a <= merged_w[-1][1] + 5_000:
                    merged_w[-1][1] = max(merged_w[-1][1], b)
                    merged_w[-1][2] = max(merged_w[-1][2], sup)
                else:
                    merged_w.append([a, b, sup])
            eroded_seed = [(za + 2_000, zb - 2_000) for za, zb in seed_by_ctg.get(ctg, ()) if zb - za > 4_000]
            for a, b, sup in merged_w:
                for sa, sb in _subtract_intervals([(a, b)], eroded_seed, min_len=2_000):
                    scored_wins.append((sup, ctg, sa, sb))

        scored_wins.sort(key=lambda x: (-x[0], x[1], x[2]))
        bf_windows: list[tuple[str, int, int]] = []
        acc_bp = 0
        cap_bp = max(0, int(self.lp.max_backfill_bp))
        for _sup, ctg, a, b in scored_wins:
            wlen = b - a
            if acc_bp + wlen > cap_bp:
                continue
            bf_windows.append((ctg, a, b))
            acc_bp += wlen
        bf_windows.sort(key=lambda w: (w[0], w[1]))

        if not bf_windows:
            bf_gtf.write_text("")
            _write_json_atomic(bf_wins_p, {"input_key": expected_key, "enabled": True, "windows": []})
            return {
                "enabled": True,
                "input_key": expected_key,
                "backfill_bp": 0,
                "backfill_n_windows": 0,
                "backfill_workers": 0,
                "backfill_wall_s": 0.0,
                "backfill_cpu_s": 0.0,
            }

        cfg = self.gate.get("tiberius_model_cfg", "")
        k_cfg = max(1, int(self.lp.tiberius_workers))
        k_eff, _bs, _sd, _av = _approve_wave_workers_and_bs(
            k_cfg, cfg, _effective_reserve_gib(self.lp.tiberius_reserve_gib)
        )
        k_bf = min(k_eff, max(1, round(acc_bp / 15_000_000)))
        bf_work = cd / "backfill"
        bf_tib_dir = cd / "tib_backfill"
        bf_work.mkdir(parents=True, exist_ok=True)
        bf_gtf_tmp = bf_work / "tiberius_backfill.gtf"
        bf_faa_tmp = bf_work / "tiberius_backfill.faa"
        rec_bf, _bf_cstats = self._run_tiberius_windows_chunked(
            bf_windows, bf_work, bf_tib_dir, "bf_", bf_gtf_tmp, bf_faa_tmp,
            cpu_only=True, workers_override=k_bf,
        )
        os.replace(bf_gtf_tmp, bf_gtf)
        _write_json_atomic(
            bf_wins_p,
            {"input_key": expected_key, "enabled": True, "windows": [list(w) for w in bf_windows]},
        )
        startup_est_s = 7.0 * k_bf
        info_bf = {
            "enabled": True,
            "input_key": expected_key,
            "backfill_bp": acc_bp,
            "backfill_n_windows": len(bf_windows),
            "backfill_workers": k_bf,
            "backfill_wall_s": round(rec_bf.wall_seconds, 3),
            "backfill_cpu_s": round(rec_bf.cpu_seconds, 3),
            "est_container_startup_share": round(min(1.0, startup_est_s / max(0.001, rec_bf.wall_seconds)), 3),
        }
        self.log.append(
            StageRecord(
                name=f"arm{self.c_tag}_orphan_backfill",
                wall_seconds=rec_bf.wall_seconds,
                extra=info_bf,
            )
        )
        return info_bf

    def _synth_context(self, s: dict, cache_dir: Path | None = None, include_raw_alignments: bool = True) -> dict:
        """Load auxiliary synthesis inputs in memory without mutating seed/ (N6 persisted readthrough fallback)."""
        sd = self.dir / "seed"
        prot = self._seed_protein_path()
        rt_ids: set[str] = set()
        rt_seed = sd / "readthrough_targets.json"
        rt_state = (cache_dir / "state" / "readthrough_targets.json") if cache_dir is not None else None
        rt_cache = (cache_dir / "readthrough_targets.json") if cache_dir is not None else None
        if self.gate.get("egapx_protein_bin"):
            src_prot = seedmod.BUILD_ROOT / "proteins" / self.gate["egapx_protein_bin"]
            if src_prot.exists():
                rt_ids |= seedmod.scan_readthrough_protein_ids(src_prot, cache_json=rt_cache)
        if not rt_ids:
            for cand_rt in (rt_seed, rt_state, rt_cache):
                if cand_rt is not None and cand_rt.exists():
                    try:
                        loaded_rt = json.loads(cand_rt.read_text())
                        if isinstance(loaded_rt, dict):
                            rt_ids |= {str(x) for x in (loaded_rt.get("ids") or ())}
                        elif isinstance(loaded_rt, list):
                            rt_ids |= {str(x) for x in loaded_rt}
                        if rt_ids:
                            break
                    except (OSError, ValueError, TypeError):
                        pass
        if not rt_ids and prot.exists():
            rt_ids |= seedmod.scan_readthrough_protein_ids(prot)

        raw_alns = None
        if include_raw_alignments:
            mp_gff = sd / "miniprot.gff3"
            prot_fai = Path(str(prot) + ".fai")
            if mp_gff.exists() and prot_fai.exists():
                raw_alns = seedmod.read_miniprot_raw_alignments(
                    mp_gff, seedmod.protein_lengths(prot), min_positive=0.40, min_cov=0.0
                )

        bait_map: dict[str, list[tuple[int, int]]] = {}
        org_set: set[str] = set()
        if (self.dir / "genome" / "genome_meta.json").exists():
            rrna_spans, org_set = self._bait_spans_and_fa(sd, s.get("hits", ()), tib=s.get("tiberius", ()), out_fa=None)
            for c, a, b in rrna_spans:
                bait_map.setdefault(c, []).append((a, b))
        return {
            "raw_alignments": raw_alns,
            "readthrough_targets": rt_ids,
            "bait_intervals": bait_map,
            "organelle_contigs": org_set,
        }

    def _verify_and_load_state_cache(
        self, ev_dir: Path
    ) -> tuple[
        JunctionStore,
        BinStore,
        dict,
        dict,
        dict,
        dict,
        list[Transcript],
        list[tuple[str, int, int]],
        set[str],
        list[dict],
        dict,
        set[tuple[str, int, int]] | None,
    ]:
        """Verify M6/P1/B3/C3/N6/Iter2'/Logan SHA256 provenance manifest and load cached BAM-free state; raise RuntimeError on mismatch."""
        prov_p = ev_dir / "provenance.json"
        mbam_p = ev_dir / "microexons_bam.json"
        gap_p = ev_dir / "gap_continuity.json"
        bf_gtf_p = ev_dir / "tiberius_backfill.gtf"
        bf_win_p = ev_dir / "backfill_windows.json"
        rt_p = ev_dir / "readthrough_targets.json"
        lnc_gate_p = ev_dir / "logan_lnc_gate.json"
        junc_p = ev_dir / "junctions.parquet"
        bins_p = ev_dir / "bins.npz"
        for req in (prov_p, mbam_p, gap_p, bf_gtf_p, bf_win_p, rt_p, junc_p, bins_p, self.genome_fa):
            if not req.exists():
                raise RuntimeError(f"State cache incomplete for BAM-free resynth_c; missing {req}")
        try:
            prov = json.loads(prov_p.read_text())
            mbam_raw = json.loads(mbam_p.read_text())
            gap_raw = json.loads(gap_p.read_text())
            bf_win_raw = json.loads(bf_win_p.read_text())
            rt_raw = json.loads(rt_p.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Corrupt state cache JSON in {ev_dir}") from exc
        if not isinstance(prov, dict) or not isinstance(mbam_raw, dict) or not isinstance(gap_raw, dict):
            raise RuntimeError(f"Malformed provenance or state cache payload in {ev_dir}")
        bam_sha = prov.get("capped_short_bam_sha256", "")
        if not (isinstance(bam_sha, str) and len(bam_sha) == 64 and all(ch in "0123456789abcdef" for ch in bam_sha)):
            raise RuntimeError("Invalid or empty capped_short_bam_sha256 in state/provenance.json")
        if prov.get("rescue_code_version") != RESCUE_CODE_VERSION:
            raise RuntimeError(
                f"rescue_code_version mismatch: {prov.get('rescue_code_version')!r} != {RESCUE_CODE_VERSION!r}"
            )
        if prov.get("rescue_params") != _RESCUE_PARAMS:
            raise RuntimeError("rescue_params mismatch in state/provenance.json")
        expected_hashes = {
            "genome_fna_sha256": _sha256_file(self.genome_fa),
            "junctions_parquet_sha256": _sha256_file(junc_p),
            "bins_npz_sha256": _sha256_file(bins_p),
            "microexons_bam_json_sha256": _sha256_file(mbam_p),
            "gap_continuity_json_sha256": _sha256_file(gap_p),
            "tiberius_backfill_gtf_sha256": _sha256_file(bf_gtf_p),
            "backfill_windows_json_sha256": _sha256_file(bf_win_p),
            "readthrough_targets_json_sha256": _sha256_file(rt_p),
        }
        for hk, actual in expected_hashes.items():
            if prov.get(hk) != actual:
                raise RuntimeError(f"Provenance SHA256 mismatch on {hk}: expected {prov.get(hk)!r}, got {actual!r}")

        lnc_eligible_c: set[tuple[str, int, int]] | None = None
        if "logan_lnc_gate_json_sha256" in prov or lnc_gate_p.exists():
            if not lnc_gate_p.exists():
                raise RuntimeError(f"Missing {lnc_gate_p} required by provenance logan_lnc_gate_json_sha256")
            if "logan_lnc_gate_json_sha256" not in prov:
                raise RuntimeError(f"Missing logan_lnc_gate_json_sha256 in {prov_p} for existing {lnc_gate_p}")
            actual_lg_sha = _sha256_file(lnc_gate_p)
            if prov.get("logan_lnc_gate_json_sha256") != actual_lg_sha:
                raise RuntimeError(
                    f"Provenance SHA256 mismatch on logan_lnc_gate_json_sha256: "
                    f"expected {prov.get('logan_lnc_gate_json_sha256')!r}, got {actual_lg_sha!r}"
                )
            try:
                lg_raw = json.loads(lnc_gate_p.read_text())
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Corrupt {lnc_gate_p}") from exc
            if not isinstance(lg_raw, dict):
                raise RuntimeError(f"Malformed {lnc_gate_p}")
            if bool(lg_raw.get("enabled", False)):
                lnc_eligible_c = {
                    (str(c), int(a), int(b))
                    for c, a, b in (lg_raw.get("eligible_introns") or ())
                }

        nc_sha_prov = prov.get("ncrna_raw_sha256")
        if not isinstance(nc_sha_prov, str) or not nc_sha_prov:
            raise RuntimeError("Missing or invalid ncrna_raw_sha256 in state/provenance.json")
        nc_mode = self.lp.enable_structural_ncrna
        if nc_mode not in ("auto", "on", "off"):
            raise ValueError(f"Invalid enable_structural_ncrna={nc_mode!r}; expected 'auto', 'on', or 'off'")
        ncrna_dir = self.dir / "seed" / "ncrna_raw"
        raw_nc_p = ncrna_dir / "raw_ncrna_candidates.json"
        done_nc_p = ncrna_dir / "done.json"
        is_hex64 = len(nc_sha_prov) == 64 and all(ch in "0123456789abcdef" for ch in nc_sha_prov)
        if is_hex64:
            if not raw_nc_p.exists() or not done_nc_p.exists():
                raise RuntimeError(f"Missing seed/ncrna_raw cache required by provenance ncrna_raw_sha256 in {ncrna_dir}")
            actual_nc_sha = _sha256_file(raw_nc_p)
            if actual_nc_sha != nc_sha_prov:
                raise RuntimeError(
                    f"Provenance SHA256 mismatch on ncrna_raw_sha256: expected {nc_sha_prov!r}, got {actual_nc_sha!r}"
                )
            try:
                done_nc = json.loads(done_nc_p.read_text())
                ncrna_raw_loaded = json.loads(raw_nc_p.read_text())
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Corrupt structural ncRNA cache JSON in {ncrna_dir}") from exc
            taxon_grp = ncrna_mod.resolve_rfam_taxon_group(self.gate)
            done_stages_c = done_nc.get("stages") if isinstance(done_nc, dict) else None
            stages_arg_c = done_stages_c if isinstance(done_stages_c, dict) else None
            exp_nc_ck = ncrna_mod.compute_ncrna_raw_cache_key(
                self._genome_identity_digest(), taxon_grp, stages=stages_arg_c
            )
            full_stages_c = stages_arg_c is None or all(
                bool(stages_arg_c.get(k, False)) for k in ("trnascan", "barrnap", "rfam")
            )
            rfam_used_c = stages_arg_c is None or bool(stages_arg_c.get("rfam", False))
            exp_asset_sha_c = ncrna_mod.PINNED_RFAM_ASSETS if rfam_used_c else {}
            if nc_mode == "on" and not full_stages_c:
                raise RuntimeError(
                    f"Partial structural ncRNA cache in {ncrna_dir} cannot satisfy enable_structural_ncrna='on'"
                )
            if (
                not isinstance(done_nc, dict)
                or done_nc.get("raw_candidates_sha256") != actual_nc_sha
                or done_nc.get("cache_key") != exp_nc_ck
                or done_nc.get("asset_sha256") != exp_asset_sha_c
            ):
                raise RuntimeError(f"Stale or tampered structural ncRNA done.json in {ncrna_dir}")
            if rfam_used_c:
                rdir_cand = ncrna_mod.default_rfam_dir()
                for fname, exp_sha in ncrna_mod.PINNED_RFAM_ASSETS.items():
                    fp = rdir_cand / fname
                    if fp.is_file() and _sha256_file(fp) != exp_sha:
                        raise RuntimeError(f"Pinned Rfam 15.1 SHA256 mismatch for {fp}")
            ncrna_raw_c = [] if nc_mode == "off" else list(ncrna_raw_loaded)
            ncrna_info_c = (
                {"status": "disabled", "n_raw_candidates": 0}
                if nc_mode == "off"
                else {**done_nc, "cached": True}
            )
        elif nc_sha_prov in ("skipped_missing_assets", "disabled", "none"):
            if nc_mode == "on":
                raise RuntimeError(
                    f"State cache recorded ncrna_raw_sha256={nc_sha_prov!r}; cannot satisfy enable_structural_ncrna='on'"
                )
            if nc_mode != "off" and raw_nc_p.exists():
                raise RuntimeError(
                    f"State cache ncrna_raw_sha256={nc_sha_prov!r} out of sync with existing {raw_nc_p}; "
                    "re-run non-cached resynth_c to refresh provenance"
                )
            ncrna_raw_c = []
            ncrna_info_c = {"status": "disabled" if nc_mode == "off" else nc_sha_prov, "n_raw_candidates": 0}
        else:
            raise RuntimeError(f"Unrecognized ncrna_raw_sha256 token {nc_sha_prov!r} in state/provenance.json")

        try:
            from .state import read_junctions

            js, region_of = read_junctions(junc_p)
            bins = BinStore.load_npz(bins_p)
            rna_direct = {
                (str(ctg), int(a), int(b)): [(int(x), int(y), int(sup)) for x, y, sup in vals]
                for ctg, a, b, vals in mbam_raw["entries"]
            }
            counts = {
                "junction_pairs": int(mbam_raw["counts"]["junction_pairs"]),
                "softclip": int(mbam_raw["counts"]["softclip"]),
            }
            gap_cont = {
                (str(ctg), int(ga), int(gb)): int(cnt)
                for ctg, ga, gb, cnt in gap_raw.get("entries", [])
            }
            bf_key_c, bf_enabled_c, bf_windows_raw = self._parse_backfill_windows_payload(bf_win_raw)
            use_bf_cache = bool(self.lp.cpu_focaloop_tiberius and self.lp.enable_orphan_backfill)
            if use_bf_cache and not bf_enabled_c:
                raise RuntimeError(
                    "State cache recorded with enable_orphan_backfill=False; retained BAM/waves required to run backfill"
                )
            if use_bf_cache:
                # Z4: Verify stored backfill input_key matches current parameters/inputs on BAM-free replay.
                meta_p = self.dir / "genome" / "genome_meta.json"
                org_set_c: set[str] = set()
                if meta_p.exists():
                    try:
                        org_set_c = set(json.loads(meta_p.read_text()).get("organelle_contigs", []))
                    except (OSError, ValueError, TypeError):
                        org_set_c = set()
                expected_bf_key = self._compute_backfill_input_key(
                    ev_dir.parent, self._load_seed_zoom_windows(org_set_c)
                )
                if not bf_key_c or bf_key_c != expected_bf_key:
                    raise RuntimeError(
                        f"Backfill input_key mismatch in BAM-free state cache ({bf_key_c!r} != {expected_bf_key!r}); "
                        "retained BAM/waves required to recompute"
                    )
                bf_models = seedmod.load_tiberius(bf_gtf_p) if bf_gtf_p.stat().st_size > 0 else []
                for bm in bf_models:
                    bm.source = "tiberius_backfill"
                    bm.attrs["backfill"] = "1"
                bf_windows = bf_windows_raw
            else:
                bf_models = []
                bf_windows = []
            rt_ids = {str(x) for x in rt_raw}
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialize verified state cache in {ev_dir}") from exc
        return (
            js,
            bins,
            region_of,
            rna_direct,
            counts,
            gap_cont,
            bf_models,
            bf_windows,
            rt_ids,
            ncrna_raw_c,
            ncrna_info_c,
            lnc_eligible_c,
        )

    def _post_synth_c(
        self,
        js: JunctionStore,
        bins: BinStore,
        rna_tx: list,
        merged: Path,
        region_of: dict,
        saturated: dict,
        s: dict,
        cached_bam_zoom: tuple | None = None,
        backfill_info: dict | None = None,
        lnc_eligible_introns: set[tuple[str, int, int]] | None = None,
        logan_lnc_gate_meta: dict | None = None,
    ) -> dict:
        """Stage 5 (micro-exon zoom + gap continuity + retrocopy discriminator) + synthesis + Parquet/NPZ state."""
        cd = self.dir / f"arm{self.c_tag}"
        ev_dir = cd / "state"
        ev_dir.mkdir(parents=True, exist_ok=True)
        g = self.genome
        t2, c2 = time.time(), self_cpu()
        bf_gtf_p = ev_dir / "tiberius_backfill.gtf"
        bf_win_p = ev_dir / "backfill_windows.json"
        use_bf = bool(self.lp.cpu_focaloop_tiberius and self.lp.enable_orphan_backfill)
        eff_lnc_eligible = lnc_eligible_introns
        m_mp = microexons_from_models(
            [h for h in s["hits"] if h.quality >= self.sp.hit_candidate_min_quality],
            max_len=_RESCUE_PARAMS["pair_max_len"],
        )
        if cached_bam_zoom is None:
            if not bf_gtf_p.exists():
                bf_gtf_p.write_text("")
            if not bf_win_p.exists():
                _write_json_atomic(bf_win_p, {"input_key": "", "enabled": use_bf, "windows": []})
            if use_bf:
                bf_models = seedmod.load_tiberius(bf_gtf_p) if bf_gtf_p.stat().st_size > 0 else []
                for bm in bf_models:
                    bm.source = "tiberius_backfill"
                    bm.attrs["backfill"] = "1"
                _, _, bf_windows = self._parse_backfill_windows_payload(json.loads(bf_win_p.read_text()))
            else:
                bf_models = []
                bf_windows = []
            tib_all = list(s["tiberius"]) + list(bf_models)
            m_tib = microexons_from_models(tib_all, max_len=_RESCUE_PARAMS["pair_max_len"])
            jl = js.by_contig(min_support=_RESCUE_PARAMS["pair_min_support"])
            micro_pairs = microexons_from_junction_pairs(
                jl, min_support=_RESCUE_PARAMS["pair_min_support"], max_len=_RESCUE_PARAMS["pair_max_len"]
            )
            clip_sides: dict[tuple[str, int, int], dict[tuple[int, int], tuple[int, int]]] = {}
            micro_clip_raw = (
                rescue_microexons_from_bam(
                    [str(merged)],
                    jl,
                    g,
                    min_support=_RESCUE_PARAMS["min_support"],
                    min_intron=_RESCUE_PARAMS["min_intron"],
                    min_clip=_RESCUE_PARAMS["min_clip"],
                    max_reads_per_site=_RESCUE_PARAMS["max_reads_per_site"],
                    sided_votes_out=clip_sides,
                )
                if merged.exists()
                else {}
            )
            # #56: Require independent corroboration (paired junctions, Tiberius/miniprot model, or bidirectional
            # >=2 donor + >=2 acceptor soft-clip votes) before inserting 1-codon (3 bp) soft-clip micro-exons.
            micro_clip: dict[tuple[str, int, int], list[tuple[int, int, int]]] = {}
            for gk, cvals in micro_clip_raw.items():
                pair_coords = {(x, y) for x, y, _ in micro_pairs.get(gk, [])}
                model_coords = {(x, y) for x, y, _ in m_tib.get(gk, [])} | {(x, y) for x, y, _ in m_mp.get(gk, [])}
                sides_gk = clip_sides.get(gk, {})
                kept_cvals: list[tuple[int, int, int]] = []
                for ms, me, sup in cvals:
                    if me - ms <= 3:
                        r_cnt, l_cnt = sides_gk.get((ms, me), (0, 0))
                        if (ms, me) not in pair_coords and (ms, me) not in model_coords and not (r_cnt >= 2 and l_cnt >= 2):
                            continue
                    kept_cvals.append((ms, me, sup))
                if kept_cvals:
                    micro_clip[gk] = kept_cvals
            rna_direct = merge_microexon_maps(micro_pairs, micro_clip)
            n_pairs = sum(len(v) for v in micro_pairs.values())
            n_clip = sum(len(v) for v in micro_clip.values())
            gap_cont: dict[tuple[str, int, int], int] | None = (
                count_gap_interior_nonspliced_reads([str(merged)], candidate_exonic_bridge_gaps(rna_tx, max_gap=1_000))
                if merged.exists()
                else None
            )
            verified_rt_ids: set[str] | None = None
            raw_ncrna, nc_sha_tok, ncrna_info = self._get_ncrna_raw_candidates(run_if_missing=False)
        else:
            if len(cached_bam_zoom) >= 9:
                (
                    rna_direct,
                    cached_counts,
                    gap_cont,
                    bf_models_c,
                    bf_windows_c,
                    verified_rt_ids,
                    raw_ncrna,
                    ncrna_info,
                    eff_lnc_eligible,
                ) = cached_bam_zoom[:9]
            else:
                (
                    rna_direct,
                    cached_counts,
                    gap_cont,
                    bf_models_c,
                    bf_windows_c,
                    verified_rt_ids,
                    raw_ncrna,
                    ncrna_info,
                ) = cached_bam_zoom
            nc_sha_tok = ""
            bf_models = bf_models_c if use_bf else []
            bf_windows = bf_windows_c if use_bf else []
            n_pairs = cached_counts["junction_pairs"]
            n_clip = cached_counts["softclip"]
            tib_all = list(s["tiberius"]) + list(bf_models)
            m_tib = microexons_from_models(tib_all, max_len=_RESCUE_PARAMS["pair_max_len"])

        # model-called micro-exons need independent corroboration: Tiberius AND miniprot agree, or direct RNA evidence
        micro_models = {}
        for k, v in m_tib.items():
            corrob = {(x, y) for x, y, _ in m_mp.get(k, [])} | {(x, y) for x, y, _ in rna_direct.get(k, [])}
            keep = [(x, y, sup) for x, y, sup in v if (x, y) in corrob]
            if keep:
                micro_models[k] = keep
        micro = merge_microexon_maps(rna_direct, micro_models)
        zoom_wall = time.time() - t2
        t3 = time.time()
        ctx = self._synth_context(s, cache_dir=cd, include_raw_alignments=True)
        eff_rt_ids = verified_rt_ids if verified_rt_ids is not None else ctx["readthrough_targets"]
        scanned_wins = self._load_seed_zoom_windows(ctx["organelle_contigs"]) + list(bf_windows)
        res = synthesize(
            g, tib_all, s["hits"], s["loci"], junctions=js, rna_tx=rna_tx, bins=bins,
            microexons=micro, zoom=self._enable_zoom, work=cd, params=self.sp, prefix="GLC",
            raw_alignments=ctx["raw_alignments"], bait_intervals=ctx["bait_intervals"],
            organelle_contigs=ctx["organelle_contigs"], readthrough_targets=eff_rt_ids,
            raw_ncrna_candidates=raw_ncrna, scanned_tib_windows=scanned_wins, gap_continuity=gap_cont,
            lnc_eligible_introns=eff_lnc_eligible, threads=self.lp.threads,
        )
        write_gff3(res.models, cd / f"arm{self.c_tag}.gff3")
        synth_wall = time.time() - t3
        nz_stats: dict | None = None
        if self.lp.emit_nozoom_ablation:
            res_nz = synthesize(
                g, tib_all, s["hits"], s["loci"], junctions=js, rna_tx=rna_tx, bins=bins,
                zoom=False, params=self.sp, prefix="GLCnz",
                bait_intervals=ctx["bait_intervals"], organelle_contigs=ctx["organelle_contigs"],
                readthrough_targets=eff_rt_ids, raw_ncrna_candidates=raw_ncrna,
                scanned_tib_windows=scanned_wins, lnc_eligible_introns=eff_lnc_eligible,
                threads=self.lp.threads,
            )
            write_gff3(res_nz.models, cd / f"arm{self.c_tag}_nozoom.gff3")
            nz_stats = res_nz.stats
        from .state import write_junctions, write_loci

        t4 = time.time()
        write_loci(ev_dir / "loci.parquet", s["loci"], res.models, saturated)
        if cached_bam_zoom is None:
            junc_p = ev_dir / "junctions.parquet"
            bins_p = ev_dir / "bins.npz"
            mbam_p = ev_dir / "microexons_bam.json"
            gap_p = ev_dir / "gap_continuity.json"
            rt_p = ev_dir / "readthrough_targets.json"
            lnc_gate_p = ev_dir / "logan_lnc_gate.json"
            prov_p = ev_dir / "provenance.json"
            write_junctions(junc_p, js, {(k[0], k[1], k[2]): v for k, v in region_of.items()})
            bins.save_npz(bins_p)
            mbam_payload = {
                "entries": [[k[0], k[1], k[2], [list(x) for x in v]] for k, v in sorted(rna_direct.items())],
                "counts": {"junction_pairs": n_pairs, "softclip": n_clip},
            }
            gap_payload = {
                "entries": [[k[0], k[1], k[2], int(cnt)] for k, cnt in sorted((gap_cont or {}).items())]
            }
            lg_meta = logan_lnc_gate_meta or {
                "enabled": eff_lnc_eligible is not None,
                "n_logan_runs": 0,
                "min_lnc_runs": 0,
                "min_lnc_ua": 0,
            }
            lnc_gate_payload = {
                "enabled": bool(lg_meta.get("enabled", False)),
                "n_logan_runs": int(lg_meta.get("n_logan_runs", 0)),
                "min_lnc_runs": int(lg_meta.get("min_lnc_runs", 0)),
                "min_lnc_ua": int(lg_meta.get("min_lnc_ua", 0)),
                "eligible_introns": (
                    [[c, a, b] for c, a, b in sorted(eff_lnc_eligible)]
                    if eff_lnc_eligible is not None
                    else []
                ),
            }
            _write_json_atomic(mbam_p, mbam_payload)
            _write_json_atomic(gap_p, gap_payload)
            _write_json_atomic(rt_p, sorted(eff_rt_ids))
            _write_json_atomic(lnc_gate_p, lnc_gate_payload)
            if merged.exists() and self.genome_fa.exists():
                prov_payload = {
                    "genome_fna_sha256": _sha256_file(self.genome_fa),
                    "capped_short_bam_sha256": _sha256_file(merged),
                    "junctions_parquet_sha256": _sha256_file(junc_p),
                    "bins_npz_sha256": _sha256_file(bins_p),
                    "microexons_bam_json_sha256": _sha256_file(mbam_p),
                    "gap_continuity_json_sha256": _sha256_file(gap_p),
                    "tiberius_backfill_gtf_sha256": _sha256_file(bf_gtf_p),
                    "backfill_windows_json_sha256": _sha256_file(bf_win_p),
                    "readthrough_targets_json_sha256": _sha256_file(rt_p),
                    "logan_lnc_gate_json_sha256": _sha256_file(lnc_gate_p),
                    "ncrna_raw_sha256": nc_sha_tok,
                    "rescue_code_version": RESCUE_CODE_VERSION,
                    "rescue_params": dict(_RESCUE_PARAMS),
                }
                _write_json_atomic(prov_p, prov_payload)
        shutil.copy(cd / f"arm{self.c_tag}.gff3", ev_dir / "models.gff3")
        (ev_dir / "retrocopy_calls.json").write_text(json.dumps(res.retro_calls, default=str))
        (ev_dir / "microexons.json").write_text(json.dumps({f"{k[0]}:{k[1]}-{k[2]}": v for k, v in micro.items()}))
        state_wall = time.time() - t4
        diamond_cpu_s = float(res.stats.get("diamond_cpu_s", 0.0)) + (
            float(nz_stats.get("diamond_cpu_s", 0.0)) if nz_stats else 0.0
        )
        out_post = {
            "zoom_wall_s": zoom_wall,
            "synth_wall_s": synth_wall,
            "state_wall_s": state_wall,
            "py_cpu_s": (self_cpu() - c2) + diamond_cpu_s,
            "diamond_cpu_s": diamond_cpu_s,
            "synth_stats": res.stats,
            "synth_stats_nozoom": nz_stats,
            "structural_ncrna": ncrna_info,
            "microexon_rescues": {
                "junction_pairs": n_pairs,
                "softclip": n_clip,
                "model_called": sum(len(v) for v in micro_models.values()),
            },
        }
        if backfill_info is not None:
            out_post["orphan_backfill"] = backfill_info
        return out_post

    def resynth_b(self) -> dict:
        """Re-run synthesis for Arm B from its retained evidence (junction store, bin tables, StringTie GTF)."""
        bd = self.dir / "armB"
        summ_p = bd / "armB_summary.json"
        summ = json.loads(summ_p.read_text())
        s = self.load_seed()
        ctx = self._synth_context(s, cache_dir=bd, include_raw_alignments=False)
        raw_ncrna, _nc_sha, _nc_info = self._get_ncrna_raw_candidates(run_if_missing=True)
        t0, c0 = time.time(), self_cpu()
        with open(bd / "evidence.pkl", "rb") as fh:
            js = pickle.load(fh)["junctions"]
        bins = BinStore(self.genome.lengths)
        for r, _ in self.runs():
            bins.ingest_tsv(bd / "align" / f"{r}.bins.tsv")
        rna_tx = read_models(bd / "stringtie.gtf", fmt="gtf")
        if (bd / "stringtie_long.gtf").exists():
            rna_tx += _read_long_read_models(bd / "stringtie_long.gtf")
        scanned_wins = self._load_seed_zoom_windows(ctx["organelle_contigs"])
        res = synthesize(
            self.genome, s["tiberius"], s["hits"], s["loci"], junctions=js, rna_tx=rna_tx, bins=bins,
            params=self.sp, prefix="GLB",
            bait_intervals=ctx["bait_intervals"], organelle_contigs=ctx["organelle_contigs"],
            readthrough_targets=ctx["readthrough_targets"], raw_ncrna_candidates=raw_ncrna,
            scanned_tib_windows=scanned_wins, threads=self.lp.threads,
        )
        write_gff3(res.models, bd / "armB.gff3")
        tot_cpu = (self_cpu() - c0) + float(res.stats.get("diamond_cpu_s", 0.0))
        summ["synth_stats"] = res.stats
        summ["resynth"] = {"wall_s": time.time() - t0, "cpu_s": tot_cpu}
        summ_p.write_text(json.dumps(summ, indent=1, default=str))
        (self.dir / "eval" / "armB.eval.json").unlink(missing_ok=True)
        return summ

    def resynth_c(self, use_state_cache: bool = False) -> dict:
        """Re-run Stage 5 + synthesis from retained Arm C evidence (full BAM+waves or M6-verified state cache)."""
        cd = self.dir / f"arm{self.c_tag}"
        ev_dir = cd / "state"
        summ_p = cd / f"arm{self.c_tag}_summary.json"
        summ = json.loads(summ_p.read_text())
        s = self.load_seed()
        g = self.genome
        t0, c0 = time.time(), self_cpu()

        logan_summ = summ.get("logan") if isinstance(summ.get("logan"), dict) else {}
        n_logan_runs_summ = int(logan_summ.get("n_logan_runs", 0))
        used_logan = n_logan_runs_summ > 0
        fb_short_set = set(logan_summ.get("fallback_short_runs") or ()) if used_logan else None
        logan_ka_junc_p = cd / "logan" / "logan.per_run.ka.tsv"
        logan_ka_bins_p = cd / "logan" / "logan.bins.ka.tsv"
        logan_unw_junc_p = cd / "logan" / "logan.per_run.unw.tsv"
        logan_raw_ok = (not used_logan) or (
            logan_ka_junc_p.exists() and logan_ka_bins_p.exists() and logan_unw_junc_p.exists()
        )

        merged_bam = ev_dir / "capped_short.bam"
        waves_dir = cd / "waves"
        scout_p = self.dir / "seed" / "scout_waves.json"
        raw_scout_waves = json.loads(scout_p.read_text()).get("waves", []) if scout_p.exists() else []
        scout_waves = [
            sw for sw in raw_scout_waves
            if (sw.get("run_id") in fb_short_set if fb_short_set is not None else True)
        ]
        scout_ok = all(Path(sw["junc_tsv"]).exists() and Path(sw["bins_tsv"]).exists() for sw in scout_waves)
        jtsvs = (
            [jt for jt in sorted(waves_dir.glob("w*.junc.tsv")) if jt.stat().st_size > 0]
            if waves_dir.is_dir()
            else []
        )
        expected_loop_waves = (
            max(0, int(summ["waves"]) - len(scout_waves)) if "waves" in summ else -1
        )
        waves_ok = (
            logan_raw_ok
            and (waves_dir.is_dir() or (used_logan and expected_loop_waves == 0))
            and (len(jtsvs) == expected_loop_waves if expected_loop_waves >= 0 else (len(jtsvs) > 0 or used_logan))
            and all(Path(str(jt).replace(".junc.tsv", ".bins.tsv")).exists() for jt in jtsvs)
        )
        use_cache = bool(use_state_cache or not waves_ok or not merged_bam.exists() or not scout_ok)

        rna_tx_short = read_models(cd / "stringtie.gtf", fmt="gtf")
        rna_tx = list(rna_tx_short)
        if (cd / "stringtie_long.gtf").exists():
            rna_tx += _read_long_read_models(cd / "stringtie_long.gtf")
        import pyarrow.parquet as pq

        lt = pq.read_table(ev_dir / "loci.parquet", columns=["kind", "id", "saturated_wave"]).to_pylist()
        saturated = {r["id"]: r["saturated_wave"] for r in lt if r["kind"] == "locus" and r["saturated_wave"] >= 0}

        if use_cache:
            (
                js,
                bins,
                region_of,
                rna_direct,
                cached_counts,
                gap_cont,
                bf_models,
                bf_windows,
                rt_ids,
                ncrna_raw_c,
                ncrna_info_c,
                lnc_eligible_c,
            ) = self._verify_and_load_state_cache(ev_dir)
            post = self._post_synth_c(
                js, bins, rna_tx, merged_bam, region_of, saturated, s,
                cached_bam_zoom=(
                    rna_direct,
                    cached_counts,
                    gap_cont,
                    bf_models,
                    bf_windows,
                    rt_ids,
                    ncrna_raw_c,
                    ncrna_info_c,
                    lnc_eligible_c,
                ),
            )
            note = "Stage-5 + synthesis re-run from M6-verified state cache (no BAM, no re-streaming)"
            bf_cpu = 0.0
        else:
            js = JunctionStore()
            fallback_js = JunctionStore()
            bins = BinStore(g.lengths)
            if used_logan:
                ingest_per_run_junc_tsv(js, logan_ka_junc_p, g)
                bins.ingest_tsv(logan_ka_bins_p)
            for sw in scout_waves:
                js.ingest_tsv(sw["run_id"], sw["junc_tsv"], g)
                fallback_js.ingest_tsv(sw["run_id"], sw["junc_tsv"], g)
                bins.ingest_tsv(sw["bins_tsv"])
            for jt in jtsvs:
                rid = jt.name.split("_", 1)[1].rsplit(".junc.tsv", 1)[0]
                js.ingest_tsv(rid, jt, g)
                fallback_js.ingest_tsv(rid, jt, g)
                bins.ingest_tsv(Path(str(jt).replace(".junc.tsv", ".bins.tsv")))
            if n_logan_runs_summ >= LOGAN_WIDE_COHORT_MIN_RUNS:
                unw_r, unw_u = load_unw_junc_stats(logan_unw_junc_p)
                min_lr, min_lu = compute_logan_lnc_thresholds(n_logan_runs_summ)
                lnc_elig_live: set[tuple[str, int, int]] | None = compute_eligible_lnc_introns(
                    unw_r, unw_u, min_lnc_runs=min_lr, min_lnc_ua=min_lu, fb_js=fallback_js, fb_min_ua=4
                )
                lg_meta_live = {
                    "enabled": True,
                    "n_logan_runs": n_logan_runs_summ,
                    "min_lnc_runs": min_lr,
                    "min_lnc_ua": min_lu,
                }
            else:
                lnc_elig_live = None
                lg_meta_live = {
                    "enabled": False,
                    "n_logan_runs": n_logan_runs_summ,
                    "min_lnc_runs": 0,
                    "min_lnc_ua": 0,
                }
            locked = [l for l in s["loci"] if l.status == LOCKED and l.locked_model is not None]
            locked_introns = {(l.contig, a, b) for l in locked for a, b in l.locked_model.cds_introns}
            lidx_locked = LocusIndex([(l.contig, l.start, l.end, l) for l in locked])
            lidx_uns = LocusIndex([(l.contig, l.start, l.end, l) for l in s["loci"] if l.status == UNSETTLED], pad=2000)
            region_of = {k: _region(k, locked_introns, lidx_locked, lidx_uns) for k in js.ua}
            # X3(c) + Z5: Fail closed when cached backfill/gate artifacts exist without matching provenance SHA256s.
            prov_p = ev_dir / "provenance.json"
            bf_pairs = [
                ("tiberius_backfill.gtf", "tiberius_backfill_gtf_sha256"),
                ("backfill_windows.json", "backfill_windows_json_sha256"),
                ("logan_lnc_gate.json", "logan_lnc_gate_json_sha256"),
            ]
            existing_bf = [(ev_dir / fn_b, hk_b) for fn_b, hk_b in bf_pairs if (ev_dir / fn_b).exists()]
            if existing_bf:
                if not prov_p.exists():
                    raise RuntimeError(
                        f"Missing {prov_p} required to verify cached backfill artifacts in {ev_dir}"
                    )
                try:
                    prov_old = json.loads(prov_p.read_text())
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"Corrupt state/provenance.json in {ev_dir}") from exc
                if not isinstance(prov_old, dict):
                    raise RuntimeError(f"Malformed state/provenance.json in {ev_dir}")
                for fp_b, hk_b in existing_bf:
                    if hk_b not in prov_old:
                        raise RuntimeError(f"Missing {hk_b} in {prov_p} for existing {fp_b}")
                    actual_b = _sha256_file(fp_b)
                    if prov_old.get(hk_b) != actual_b:
                        raise RuntimeError(
                            f"Provenance SHA256 mismatch on {hk_b}: "
                            f"expected {prov_old.get(hk_b)!r}, got {actual_b!r}"
                        )
            bf_info = self._ensure_orphan_backfill(cd, ev_dir, rna_tx, js, s)
            bf_cpu = float(bf_info.get("backfill_cpu_s", 0.0))
            self.ensure_structural_ncrna()
            post = self._post_synth_c(
                js, bins, rna_tx, merged_bam, region_of, saturated, s,
                backfill_info=bf_info,
                lnc_eligible_introns=lnc_elig_live,
                logan_lnc_gate_meta=lg_meta_live,
            )
            note = "Stage-5 + synthesis re-run from retained state (no re-streaming)"

        tot_cpu = (self_cpu() - c0) + float(post.get("diamond_cpu_s", 0.0)) + bf_cpu
        for k in ("synth_stats", "synth_stats_nozoom", "microexon_rescues", "structural_ncrna"):
            if k in post:
                summ[k] = post[k]
        if "orphan_backfill" in post:
            summ["orphan_backfill"] = post["orphan_backfill"]
        summ["resynth"] = {"wall_s": time.time() - t0, "cpu_s": tot_cpu, "used_state_cache": use_cache, "note": note}
        summ["retained_state_bytes"] = dir_bytes(ev_dir)
        summ["retained_state_files"] = {p.name: p.stat().st_size for p in ev_dir.iterdir() if p.is_file()}
        summ_p.write_text(json.dumps(summ, indent=1, default=str))
        self.log.append(StageRecord(name=f"arm{self.c_tag}_resynth", wall_seconds=time.time() - t0, cpu_user=tot_cpu))
        for a in (self.c_tag, f"{self.c_tag}nz"):
            (self.dir / "eval" / f"arm{a}.eval.json").unlink(missing_ok=True)
        return summ



# ---------------------------------------------------------------------------------------------- helpers

_RESCUE_PARAMS = {
    "min_support": 3,
    "min_intron": 80,
    "min_clip": 5,
    "max_reads_per_site": 200,
    "pair_min_support": 2,
    "pair_max_len": 27,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _write_gtf_transcripts(txs: list[Transcript], out_path: Path) -> None:
    """Serialize transcript exon models to GTF losslessly (preserving StringTie cov/FPKM/TPM attrs)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    skip_keys = {"gene_id", "transcript_id", "exon_number", "_type", "_gene_type"}
    with open(tmp, "w") as fh:
        for t in txs:
            exons = merge_intervals(t.exons)
            if not exons:
                continue
            src = t.source or "StringTie"
            gid = t.gid or t.tid
            parts = [f'gene_id "{gid}";', f'transcript_id "{t.tid}";']
            for k in sorted(t.attrs):
                if k in skip_keys:
                    continue
                val = str(t.attrs[k]).replace('"', "")
                parts.append(f'{k} "{val}";')
            if t.biotype and "gene_biotype" not in t.attrs:
                parts.append(f'gene_biotype "{t.biotype}";')
            attr_str = " ".join(parts)
            fh.write(f"{t.contig}\t{src}\ttranscript\t{exons[0][0] + 1}\t{exons[-1][1]}\t1000\t{t.strand}\t.\t{attr_str}\n")
            for ei, (ea, eb) in enumerate(exons, 1):
                fh.write(
                    f'{t.contig}\t{src}\texon\t{ea + 1}\t{eb}\t1000\t{t.strand}\t.\t'
                    f'gene_id "{gid}"; transcript_id "{t.tid}"; exon_number "{ei}";\n'
                )
    os.replace(tmp, out_path)


def self_cpu() -> float:
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _region(k, locked_introns, lidx_locked: LocusIndex, lidx_uns: LocusIndex) -> str:
    if k in locked_introns:
        return REGION_LOCKED_KNOWN
    if lidx_locked.containing(k[0], k[1], k[2]):
        return REGION_LOCKED_NOVEL
    if lidx_uns.overlapping(k[0], k[1], k[2]):
        return REGION_UNSETTLED
    return REGION_INTERGENIC


def _activate(glc: Path, out: Path, mult: Path | None = None, cand: Path | None = None, loci: Path | None = None,
              bait: Path | None = None) -> dict:
    cmd = [str(glc), "activate", "-o", str(out)]
    if mult and cand and loci:
        cmd += ["-m", str(mult), "-c", str(cand), "-l", str(loci)]
    if bait and bait.exists():
        cmd += ["-b", str(bait)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, env=safe_env())
    return json.loads(r.stderr.strip().splitlines()[-1])


def _cat_sort(bams: list[str], out: Path, threads: int) -> float:
    """Coordinate-sort and name-aware merge BAMs (never using samtools cat across mismatched @SQ headers)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    sort_threads = max(1, min(8, int(threads)))
    sort_mem = samtools_sort_mem_arg(sort_threads)
    cpu_tot = 0.0
    if len(bams) == 1:
        _, _, u1, s1 = run_cmd(
            [
                "samtools", "sort",
                "-@", str(sort_threads),
                "-m", sort_mem,
                "-T", f"{out}.tmp",
                "-o", str(out),
                str(bams[0]),
            ]
        )
        cpu_tot += u1 + s1
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out.name}.merge.", dir=str(out.parent)))
        try:
            sorted_parts: list[str] = []
            for idx_b, bam_p in enumerate(bams):
                part_bam = tmp_dir / f"part_{idx_b:04d}.sorted.bam"
                _, _, us, ss = run_cmd(
                    [
                        "samtools", "sort",
                        "-@", str(sort_threads),
                        "-m", sort_mem,
                        "-T", str(tmp_dir / f"sort_{idx_b:04d}"),
                        "-o", str(part_bam),
                        str(bam_p),
                    ]
                )
                cpu_tot += us + ss
                sorted_parts.append(str(part_bam))
            _, _, um, sm = run_cmd(
                [
                    "samtools", "merge", "-f",
                    "-@", str(max(1, min(16, int(threads)))),
                    "-o", str(out),
                    *sorted_parts,
                ]
            )
            cpu_tot += um + sm
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    _, _, ui, si = run_cmd(["samtools", "index", str(out)])
    return cpu_tot + ui + si


def _tib_summary(r: StageRecord) -> dict:
    return {"wall_s": r.wall_seconds, "cpu_s": r.cpu_seconds, "gpu_s": r.gpu_seconds, "gpu_util_s": r.gpu_util_seconds,
            "peak_vram_mb": r.peak_vram_mb, "batch_size": r.extra.get("batch_size") if r.extra else None}


def _tiberius_models(tib: list[Transcript]) -> list[Transcript]:
    out = []
    for t in tib:
        nt = Transcript(tid=t.tid, contig=t.contig, strand=t.strand, exons=list(t.cds), cds=list(t.cds), gid=t.gid, source="Tiberius")
        out.append(nt)
    return out


def _subtract_intervals(
    spans: list[tuple[int, int]],
    forbidden: list[tuple[int, int]],
    min_len: int = 80,
) -> list[tuple[int, int]]:
    """Subtract ``forbidden`` [a, b) intervals from ``spans`` so zero coding bases enter bait FASTA."""
    if not spans:
        return []
    spans_s = sorted(spans)
    merged: list[tuple[int, int]] = []
    for a, b in spans_s:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    if not forbidden:
        return [(a, b) for a, b in merged if b - a >= min_len]
    forb_s = sorted(forbidden)
    f_merged: list[tuple[int, int]] = []
    for a, b in forb_s:
        if f_merged and a <= f_merged[-1][1]:
            f_merged[-1] = (f_merged[-1][0], max(f_merged[-1][1], b))
        else:
            f_merged.append((a, b))
    out: list[tuple[int, int]] = []
    for a, b in merged:
        cur = a
        for fa, fb in f_merged:
            if fb <= cur:
                continue
            if fa >= b:
                break
            if fa > cur and fa - cur >= min_len:
                out.append((cur, fa))
            cur = max(cur, fb)
            if cur >= b:
                break
        if b - cur >= min_len:
            out.append((cur, b))
    return out


def _discover_rrna_bait_intervals(
    genome: Genome,
    genome_fa: Path,
    coding_by_ctg: dict[str, list[tuple[int, int]]],
    organelle_set: set[str],
    pad: int = 1500,
    threads: int = 4,
) -> list[tuple[str, int, int]]:
    """Locate multi-copy nuclear 45S rDNA repeats via conserved rRNA core alignment, strictly excluding CDS."""
    rrna_core = Path(__file__).parent / "data" / "euk_rrna_core.fa"
    if not rrna_core.exists() or not shutil.which("minimap2") or not genome_fa.exists():
        return []
    r = subprocess.run(
        ["minimap2", "-c", "--secondary=yes", "-N", "200", "-p", "0.5", "-t", str(threads), str(genome_fa), str(rrna_core)],
        capture_output=True,
        text=True,
        check=False,
        env=safe_env(),
    )
    if r.returncode != 0 or not r.stdout:
        return []
    raw_by_ctg: dict[str, list[tuple[int, int]]] = {}
    for line in r.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 11:
            continue
        ctg = f[5]
        if ctg in organelle_set or ctg not in genome.lengths:
            continue
        aln_len = int(f[10])
        matches = int(f[9])
        if aln_len < 100 or matches / max(1, aln_len) < 0.70:
            continue
        L = genome.lengths[ctg]
        ts, te = int(f[7]), int(f[8])
        raw_by_ctg.setdefault(ctg, []).append((max(0, ts - pad), min(L, te + pad)))
    out: list[tuple[str, int, int]] = []
    for ctg, ivs in sorted(raw_by_ctg.items()):
        for a, b in _subtract_intervals(ivs, coding_by_ctg.get(ctg, [])):
            out.append((ctg, a, b))
    return out
