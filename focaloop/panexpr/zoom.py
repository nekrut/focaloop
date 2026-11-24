"""Borderline-gene targeted 31-mer zoom.

Genes near the DE decision boundary (0.005 <= padj <= 0.25 with
5 <= streamed_reads < 35) get a second, targeted pass: an exact canonical
31-mer set is built from their catalog targets, additional FASTQ byte chunks are
scanned, and only reads sharing >= 2 31-mers with the bait are mapped.  The
extra evidence is folded back into the per-run accumulators and the zoomed
genes' counts are rescaled to the global sampling depth so size factors stay
consistent across the cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from .catalog import Catalog
from .kernels import Kernel, MapResult, Read, RunAccumulator, map_pooled

K = 31
_LUT = np.full(256, 255, dtype=np.uint8)
for _ch, _v in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("a", 0), ("c", 1), ("g", 2), ("t", 3)):
    _LUT[ord(_ch)] = _v
_MASK = np.uint64((1 << (2 * K)) - 1)


def _encode_kmers(seqs: list[str], k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Canonical k-mer codes for a batch of equal-length (padded) sequences -> (codes[n, L-k+1], valid mask)."""
    if not seqs:
        return np.zeros((0, 0), dtype=np.uint64), np.zeros((0, 0), dtype=bool)
    L = max(len(s) for s in seqs)
    buf = np.full((len(seqs), L), ord("N"), dtype=np.uint8)
    for i, s in enumerate(seqs):
        b = np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)
        buf[i, :len(b)] = b
    codes = _LUT[buf]
    n_win = L - k + 1
    if n_win <= 0:
        return np.zeros((len(seqs), 0), dtype=np.uint64), np.zeros((len(seqs), 0), dtype=bool)
    bad = codes == 255
    c = np.where(bad, 0, codes).astype(np.uint64)
    fwd = np.zeros((len(seqs), n_win), dtype=np.uint64)
    rev = np.zeros((len(seqs), n_win), dtype=np.uint64)
    invalid = np.zeros((len(seqs), n_win), dtype=bool)
    for j in range(k):
        col = c[:, j:j + n_win]
        fwd = (fwd << np.uint64(2)) | col
        rev = rev | ((np.uint64(3) - col) << np.uint64(2 * j))
        invalid |= bad[:, j:j + n_win]
    return np.minimum(fwd & _MASK, rev & _MASK), ~invalid


def build_bait(seqs: list[str], k: int = K) -> np.ndarray:
    """Sorted unique canonical k-mer codes of the bait sequences."""
    codes: list[np.ndarray] = []
    for s in seqs:
        if len(s) < k:
            continue
        c, ok = _encode_kmers([s], k)
        codes.append(c[ok])
    if not codes:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(codes))


def bait_filter(reads: list[Read], bait: np.ndarray, min_hits: int = 2, k: int = K, chunk: int = 20000) -> np.ndarray:
    """Indices of reads sharing >= min_hits canonical k-mers with the bait."""
    if bait.size == 0 or not reads:
        return np.zeros(0, dtype=np.int64)
    if k <= 31:
        from ._fast_c import c_bait_filter
        c_res = c_bait_filter(reads, bait, min_hits=min_hits, k=k)
        if c_res is not None:
            return c_res
    keep: list[np.ndarray] = []
    for st in range(0, len(reads), chunk):
        seqs = [s for s, _ in reads[st:st + chunk]]
        codes, ok = _encode_kmers(seqs, k)
        if codes.shape[1] == 0:
            continue
        hit = np.isin(codes, bait, assume_unique=False) & ok
        n_hit = hit.sum(axis=1)
        keep.append(np.nonzero(n_hit >= min_hits)[0] + st)
    return np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)


def select_borderline(de: pd.DataFrame, padj_lo: float = 0.005, padj_hi: float = 0.25, reads_lo: float = 5.0, reads_hi: float = 35.0) -> list[str]:
    m = (de["padj"] >= padj_lo) & (de["padj"] <= padj_hi) & (de["streamed_reads"] >= reads_lo) & (de["streamed_reads"] < reads_hi)
    return list(de.loc[m.fillna(False), "group_id"])


def select_zoom_targets(
    de: pd.DataFrame,
    padj_lo: float = 0.002,
    padj_hi: float = 0.25,
    low_reads_hi: float = 50.0,
    abundant_reads_min: float = 200.0,
    deplete_reads_min: float = 500.0,
) -> tuple[list[str], list[str]]:
    """Select underpowered/borderline target groups and saturated abundant groups for v2.4 C-sieve zoom.

    Target groups cover:
      * undetected / rare transcripts and dilute spike-ins (``0 <= streamed_reads < 5``),
      * low-count expressed genes (``5 <= streamed_reads < low_reads_hi``), and
      * borderline DE candidates (``padj_lo <= padj <= padj_hi``).
    Abundant groups are non-target genes with ``streamed_reads >= max(abundant_reads_min, deplete_reads_min)``.
    """
    if len(de) == 0:
        return [], []
    not_absent = (de["status"] != "genetically_absent") if "status" in de.columns else pd.Series(True, index=de.index)
    sr = de["streamed_reads"].fillna(0.0)
    low_or_rare = (sr >= 0.0) & (sr < low_reads_hi)
    borderline = (de["padj"] >= padj_lo) & (de["padj"] <= padj_hi)
    t_mask = (not_absent & (low_or_rare | borderline.fillna(False))).fillna(False)
    targets = list(de.loc[t_mask, "group_id"])
    tset = set(targets)
    abund_thresh = max(abundant_reads_min, deplete_reads_min)
    a_mask = (not_absent & (sr >= abund_thresh)).fillna(False)
    abundant = [g for g in de.loc[a_mask, "group_id"] if g not in tset]
    return targets, abundant


@dataclass
class ZoomResult:
    groups: list[str]
    bytes_scanned: int
    reads_scanned: int
    reads_kept: int
    reads_mapped: int
    per_run_scale: dict[str, float]
    sieve_mode: str = "include"
    per_run_zoom_bytes: dict[str, int] = field(default_factory=dict)


def zoom_pass(
    kernel: Kernel,
    catalog: Catalog,
    accs: dict[str, RunAccumulator],
    streams: dict,
    borderline_groups: list[str],
    extra_fraction: float,
    min_hits: int = 2,
    log=print,
    fetch_workers: int = 4,
    pool_samples: bool = True,
    abundant_groups: list[str] | None = None,
    sieve_mode: str = "auto",
    sieve_threads: int | None = None,
    sample_lane_alloc: dict[str, int] | None = None,
    close_streams: bool = False,
) -> ZoomResult:
    """Scan additional bytes of RNA runs through a compiled C 31-mer sieve, map survivors, and fold them in.

    Supports two mathematically equivalent wire-speed sieve modes via ``FastCSieve``:
      * ``include``: keep reads sharing ``>= min_hits`` canonical 31-mers with target (rare/low-count/borderline) genes;
      * ``deplete``: drop reads matching saturated abundant transcripts (``>= 500`` reads) whose 31-mers do not overlap
        any target gene, passing informative non-abundant reads to the mapper.
    Per-sample exposure is rescaled by exact bytes scanned (``before / (before + delta)``).
    """
    from ._fast_c import FastCSieve

    bset = set(borderline_groups)
    targets = [t for t in catalog.gene_targets if t.group_id in bset and t.seq]
    target_seqs = [t.seq for t in targets]

    eff_mode = "include"
    sieve = None
    if abundant_groups and sieve_mode in ("deplete", "auto"):
        aset = set(abundant_groups) - bset
        abund_seqs = [t.seq for t in catalog.gene_targets if t.group_id in aset and t.seq]
        if abund_seqs and (
            sieve_mode == "deplete"
            or (len(abundant_groups) >= 20 and sum(len(s) for s in abund_seqs) < 0.75 * max(1, sum(len(s) for s in target_seqs)))
        ):
            eff_mode = "deplete"
            sieve = FastCSieve.from_sequences(abund_seqs, k=K, mode="deplete", exclude_seqs=target_seqs)

    if sieve is None and target_seqs:
        eff_mode = "include"
        sieve = FastCSieve.from_sequences(target_seqs, k=K, mode="include")

    bait = np.zeros(0, dtype=np.uint64)
    if sieve is None:
        eff_mode = "include"
        bait = build_bait(target_seqs)
    n_kmers = sieve.n_kmers if sieve is not None else int(bait.size)

    bytes_scanned = reads_scanned = reads_kept = reads_mapped = 0
    scale: dict[str, float] = {}
    zoom_bytes_by_run: dict[str, int] = {}
    rna_items = [(run_id, rs) for run_id, rs in streams.items() if not accs[run_id].run.is_wgs]
    tot_sieve_t = max(1, sieve_threads if sieve_threads is not None else getattr(kernel, "threads", 4))
    dset = catalog.decoy_ids

    def _filter_chunk(r1: list[Read], r2: list[Read] | None, n_threads_chunk: int) -> tuple[list[Read], list[Read] | None]:
        if not r1:
            return [], ([] if r2 is not None else None)
        if sieve is not None:
            keep = sieve.filter_reads(r1, r2, min_hits=min_hits, threads=n_threads_chunk)
        else:
            keep = bait_filter(r1, bait, min_hits)
            if r2 is not None:
                keep2 = bait_filter(r2, bait, min_hits)
                keep = np.union1d(keep, keep2)
        if keep.size == 0:
            return [], ([] if r2 is not None else None)
        return [r1[i] for i in keep], ([r2[i] for i in keep] if r2 is not None else None)

    def _make_fetcher(n_threads_chunk: int):
        def _fetch_and_filter(item):
            run_id, rs = item
            before = rs.consumed_bytes
            if sample_lane_alloc is not None:
                want_b = int(sample_lane_alloc.get(run_id, 0))
            else:
                want_b = max(0, int(round(rs.total_bytes * extra_fraction)))
                rem_b = max(0, rs.total_bytes - rs.consumed_bytes)
                want_b = min(want_b, rem_b)
            if want_b <= 0:
                if close_streams and hasattr(rs, "close"):
                    rs.close()
                return run_id, before, 0, 0, [], None

            # Stream & sieve in <=128 MB slices per worker so 24 workers never accumulate raw reads in RAM
            max_slice = 128_000_000
            tot_n = 0
            kept_r1: list[Read] = []
            kept_r2: list[Read] | None = None
            rem = want_b
            while rem > 0 and not rs.exhausted:
                step = min(rem, max_slice)
                prev_c = rs.consumed_bytes
                c1, c2 = rs.advance_bytes(step)
                actual = rs.consumed_bytes - prev_c
                if not c1 and actual <= 0:
                    break
                rem -= max(step, actual)
                tot_n += len(c1)
                if c2 is not None and kept_r2 is None:
                    kept_r2 = []
                k1, k2 = _filter_chunk(c1, c2, n_threads_chunk)
                if k1:
                    kept_r1.extend(k1)
                    if kept_r2 is not None and k2 is not None:
                        kept_r2.extend(k2)
            delta = rs.consumed_bytes - before
            if close_streams and hasattr(rs, "close"):
                rs.close()
            if tot_n == 0 or not kept_r1:
                return run_id, before, delta, tot_n, [], None
            return run_id, before, delta, tot_n, kept_r1, kept_r2
        return _fetch_and_filter

    # Partition runs by biological sample into bounded waves (<=4 samples / <=64 lanes per wave)
    # so 256-lane ultra-deep cohorts never hold all samples' survivor reads in RAM at once.
    by_sample: dict[tuple[str, str], list[tuple[str, object]]] = {}
    for run_id, rs in rna_items:
        r_obj = accs[run_id].run
        by_sample.setdefault((r_obj.sample, r_obj.condition), []).append((run_id, rs))
    sample_groups = list(by_sample.values())
    max_samples_per_wave = 4
    waves: list[list[tuple[str, object]]] = []
    cur_w: list[tuple[str, object]] = []
    cur_s = 0
    for sg in sample_groups:
        if cur_w and (cur_s >= max_samples_per_wave or len(cur_w) + len(sg) > 64):
            waves.append(cur_w)
            cur_w = []
            cur_s = 0
        cur_w.extend(sg)
        cur_s += 1
    if cur_w:
        waves.append(cur_w)

    for wave_items in waves:
        n_conc = max(1, min(fetch_workers, len(wave_items))) if (fetch_workers > 1 and len(wave_items) > 1) else 1
        per_item_threads = max(1, tot_sieve_t // n_conc)
        fn_fetch = _make_fetcher(per_item_threads)
        if fetch_workers > 1 and len(wave_items) > 1:
            with ThreadPoolExecutor(max_workers=n_conc) as ex:
                fetched = list(ex.map(fn_fetch, wave_items))
        else:
            fetched = [fn_fetch(it) for it in wave_items]

        to_map: list[tuple[str, list[Read], list[Read] | None, int]] = []
        for run_id, before, delta, n, k1, k2 in fetched:
            bytes_scanned += delta
            reads_scanned += n
            zoom_bytes_by_run[run_id] = int(delta)
            if n == 0:
                scale[run_id] = before / max(1, before + delta) if (before + delta) > 0 else 1.0
                continue
            reads_kept += len(k1)
            scale[run_id] = before / max(1, before + delta)
            if k1:
                to_map.append((run_id, k1, k2, delta))
        del fetched

        mapped_by_run: dict[str, MapResult] = {}
        pooled_deltas: dict[str, int] = {}
        if to_map:
            grouped_map: dict[tuple[str, str, bool], list[tuple[str, list[Read], list[Read] | None, int]]] = {}
            for rid, k1, k2, delta in to_map:
                r_obj = accs[rid].run
                grouped_map.setdefault((r_obj.sample, r_obj.condition, k2 is not None), []).append((rid, k1, k2, delta))
            del to_map
            merged_to_map: list[tuple[str, list[Read], list[Read] | None]] = []
            for (_smp, _cnd, paired), items in grouped_map.items():
                rep_rid = next(
                    (
                        cand_rid
                        for cand_rid, cand_acc in accs.items()
                        if not cand_acc.run.is_wgs
                        and cand_acc.run.sample == _smp
                        and cand_acc.run.condition == _cnd
                        and (cand_acc.run.layout == "PE") == paired
                    ),
                    items[0][0],
                )
                if len(items) == 1:
                    merged_to_map.append((rep_rid, items[0][1], items[0][2]))
                    pooled_deltas[rep_rid] = items[0][3]
                else:
                    cat_k1: list[Read] = []
                    cat_k2: list[Read] | None = [] if paired else None
                    tot_d = 0
                    for _r, x1, x2, d in items:
                        cat_k1.extend(x1)
                        if cat_k2 is not None and x2 is not None:
                            cat_k2.extend(x2)
                        tot_d += d
                    merged_to_map.append((rep_rid, cat_k1, cat_k2))
                    pooled_deltas[rep_rid] = tot_d
            del grouped_map

            if pool_samples:
                for paired in (True, False):
                    grp = [(rid, k1, k2, accs[rid].calibrator) for rid, k1, k2 in merged_to_map if (k2 is not None) == paired]
                    if grp:
                        mapped_by_run.update(map_pooled(kernel, grp))
            else:
                for rid, k1, k2 in merged_to_map:
                    mapped_by_run[rid] = kernel.map_batch(k1, k2, accs[rid].calibrator)
            del merged_to_map

        for rid, res in mapped_by_run.items():
            reads_mapped += res.n_mapped
            zres = MapResult(n_reads=0, n_mapped=0)
            for t, c in res.unique_counts.items():
                zres.unique_counts[t] = list(c)
            zres.eq_classes = dict(res.eq_classes)
            acc = accs[rid]
            zoom_acc = getattr(acc, "zoom", None)
            if zoom_acc is None:
                zoom_acc = RunAccumulator(run=acc.run, decoy_ids=dset)
                acc.zoom = zoom_acc  # type: ignore[attr-defined]
            zoom_acc.add(zres)
            zoom_acc.bytes_consumed += pooled_deltas.get(rid, 0)

    log(f"  [zoom:{eff_mode}] {len(borderline_groups)} target genes, filter={n_kmers:,} 31-mers, scanned {reads_scanned:,} reads "
        f"({bytes_scanned / 1e6:.1f} MB), kept {reads_kept:,}, mapped {reads_mapped:,}")
    return ZoomResult(
        borderline_groups, bytes_scanned, reads_scanned, reads_kept, reads_mapped, scale,
        sieve_mode=eff_mode, per_run_zoom_bytes=zoom_bytes_by_run,
    )
