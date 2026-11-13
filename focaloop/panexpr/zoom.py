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

from dataclasses import dataclass

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


@dataclass
class ZoomResult:
    groups: list[str]
    bytes_scanned: int
    reads_scanned: int
    reads_kept: int
    reads_mapped: int
    per_run_scale: dict[str, float]


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
) -> ZoomResult:
    """Scan ``extra_fraction`` more bytes of every RNA run, keep bait-sharing reads, map and fold them in.

    Fetches stream chunks concurrently and maps kept reads via ``map_pooled`` so multi-sample studies
    pay a single index load instead of reloading the kernel index once per run.
    """
    bset = set(borderline_groups)
    targets = [t for t in catalog.gene_targets if t.group_id in bset]
    bait = build_bait([t.seq for t in targets])
    bytes_scanned = reads_scanned = reads_kept = reads_mapped = 0
    scale: dict[str, float] = {}
    rna_items = [(run_id, rs) for run_id, rs in streams.items() if not accs[run_id].run.is_wgs]

    def _fetch_and_filter(item):
        run_id, rs = item
        before = rs.consumed_bytes
        r1, r2 = rs.advance_fraction(extra_fraction)
        delta = rs.consumed_bytes - before
        n = len(r1)
        if n == 0:
            return run_id, before, delta, 0, [], None
        keep = bait_filter(r1, bait, min_hits)
        if r2 is not None:
            keep2 = bait_filter(r2, bait, min_hits)
            keep = np.union1d(keep, keep2)
        if keep.size == 0:
            return run_id, before, delta, n, [], None
        k1 = [r1[i] for i in keep]
        k2 = [r2[i] for i in keep] if r2 is not None else None
        return run_id, before, delta, n, k1, k2

    if fetch_workers > 1 and len(rna_items) > 1:
        with ThreadPoolExecutor(max_workers=fetch_workers) as ex:
            fetched = list(ex.map(_fetch_and_filter, rna_items))
    else:
        fetched = [_fetch_and_filter(it) for it in rna_items]

    to_map: list[tuple[str, list[Read], list[Read] | None, int]] = []
    for run_id, before, delta, n, k1, k2 in fetched:
        bytes_scanned += delta
        reads_scanned += n
        if n == 0:
            scale[run_id] = 1.0
            continue
        reads_kept += len(k1)
        scale[run_id] = before / max(1, before + delta)
        if k1:
            to_map.append((run_id, k1, k2, delta))

    mapped_by_run: dict[str, MapResult] = {}
    if to_map:
        if pool_samples:
            for paired in (True, False):
                grp = [(rid, k1, k2, accs[rid].calibrator) for rid, k1, k2, _d in to_map if (k2 is not None) == paired]
                if grp:
                    mapped_by_run.update(map_pooled(kernel, grp))
        else:
            for rid, k1, k2, _d in to_map:
                mapped_by_run[rid] = kernel.map_batch(k1, k2, accs[rid].calibrator)

    dset = catalog.decoy_ids
    for rid, _k1, _k2, delta in to_map:
        res = mapped_by_run.get(rid)
        if res is None:
            continue
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
        zoom_acc.bytes_consumed += delta

    log(f"  [zoom] {len(borderline_groups)} borderline genes, bait={bait.size:,} 31-mers, scanned {reads_scanned:,} reads "
        f"({bytes_scanned / 1e6:.1f} MB), kept {reads_kept:,}, mapped {reads_mapped:,}")
    return ZoomResult(borderline_groups, bytes_scanned, reads_scanned, reads_kept, reads_mapped, scale)
