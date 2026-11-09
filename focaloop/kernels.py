"""Mapping kernels that turn read batches into equivalence-class candidates.

Builtin kernels are pure numpy and always available:

* ``kmer_eq``      canonical k-mer voting that *retains* near-ties (an eq-class kernel)
* ``kmer_besthit`` the same index but reports one best target (ties broken by index),
                   the behaviour of pair-rescue aligners that silently collapse ties

External aligners (``bwa_spa`` = ``bwa mem -S -P -a``, ``bwa-mem2``, ``strobealign``, ``minimap2``, ``bowtie2``,
``salmon``, ``piscem``) are registered by name; they report ``available() is False``
unless an adapter and binary are present, and the tournament skips them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

K_MAP = 31
_LUT = np.full(256, 255, dtype=np.uint8)
for _c, _v in zip("ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3)):
    _LUT[ord(_c)] = _v
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def encode_kmers_stranded(seqs: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(canonical codes, valid mask, is_forward_canonical) for an equal-padded batch."""
    if not seqs:
        z = np.zeros((0, 0), dtype=np.uint64)
        return z, z.astype(bool), z.astype(bool)
    max_len = max(len(s) for s in seqs)
    n_win = max(0, max_len - k + 1)
    buf = np.full((len(seqs), max_len), ord("N"), dtype=np.uint8)
    for i, s in enumerate(seqs):
        raw = np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)
        buf[i, : raw.size] = raw
    codes = _LUT[buf]
    bad = codes == 255
    clean = np.where(bad, 0, codes).astype(np.uint64)
    fwd = np.zeros((len(seqs), n_win), dtype=np.uint64)
    rev = np.zeros((len(seqs), n_win), dtype=np.uint64)
    invalid = np.zeros((len(seqs), n_win), dtype=bool)
    mask = np.uint64((1 << (2 * k)) - 1)
    for j in range(k):
        col = clean[:, j : j + n_win]
        fwd = (fwd << np.uint64(2)) | col
        rev = rev | ((np.uint64(3) - col) << np.uint64(2 * j))
        invalid |= bad[:, j : j + n_win]
    fwd &= mask
    rev &= mask
    return np.minimum(fwd, rev), ~invalid, fwd <= rev


@dataclass(frozen=True)
class TargetRec:
    target_id: str
    seq: str
    group_id: str
    assembly_id: str = "primary"
    kind: str = "transcript"  # transcript | gene_body | host_decoy | plasmid_contig ...
    kingdom: str = "host"

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def is_decoy(self) -> bool:
        return self.kind == "host_decoy"


@dataclass
class MapResult:
    """Per-read candidate lists; ``candidates[i] is None`` means unmapped."""

    candidates: list[tuple[tuple[int, ...], tuple[float, ...], tuple[bool, ...]] | None]
    decoy: np.ndarray  # bool per read: best hit was a host decoy (sieved out)

    @property
    def n_reads(self) -> int:
        return len(self.candidates)

    @property
    def mapped(self) -> np.ndarray:
        return np.array([c is not None for c in self.candidates], dtype=bool)


class KmerIndex:
    """Sorted canonical k-mer -> (target index, strand) index over a target catalog."""

    def __init__(self, targets: Sequence[TargetRec], k: int = K_MAP) -> None:
        self.k = k
        self.targets = list(targets)
        self.ids = [t.target_id for t in self.targets]
        self.pos = {t: i for i, t in enumerate(self.ids)}
        self.is_decoy = np.array([t.is_decoy for t in self.targets], dtype=bool)
        kmers, tix, tstr = [], [], []
        for i, t in enumerate(self.targets):
            if len(t.seq) < k:
                continue
            c, ok, f = encode_kmers_stranded([t.seq], k)
            c, f = c[ok], f[ok]
            u, first = np.unique(c, return_index=True)
            kmers.append(u)
            tix.append(np.full(u.size, i, dtype=np.int32))
            tstr.append(f[first])
        if kmers:
            km = np.concatenate(kmers)
            ti = np.concatenate(tix)
            ts = np.concatenate(tstr)
            order = np.lexsort((ti, km))
            self.kmers, self.tix, self.tstrand = km[order], ti[order], ts[order]
        else:
            self.kmers = np.zeros(0, dtype=np.uint64)
            self.tix = np.zeros(0, dtype=np.int32)
            self.tstrand = np.zeros(0, dtype=bool)

    def vote(self, reads: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (read_idx, target_idx, n_shared_kmers, n_forward_votes) per (read, target) pair."""
        T = max(1, len(self.targets))
        empty = (np.zeros(0, np.int64),) * 4
        if not reads or self.kmers.size == 0:
            return empty
        codes, ok, fw = encode_kmers_stranded(reads, self.k)
        if codes.shape[1] == 0:
            return empty
        r_idx = np.broadcast_to(np.arange(len(reads))[:, None], codes.shape)[ok]
        km = codes[ok]
        rf = fw[ok]
        order = np.lexsort((km, r_idx))
        r_idx, km, rf = r_idx[order], km[order], rf[order]
        keep = np.ones(km.size, dtype=bool)
        keep[1:] = (km[1:] != km[:-1]) | (r_idx[1:] != r_idx[:-1])
        r_idx, km, rf = r_idx[keep], km[keep], rf[keep]
        lo = np.searchsorted(self.kmers, km, "left")
        hi = np.searchsorted(self.kmers, km, "right")
        cnt = hi - lo
        nz = cnt > 0
        r_idx, rf, lo, cnt = r_idx[nz], rf[nz], lo[nz], cnt[nz]
        if cnt.size == 0:
            return empty
        total = int(cnt.sum())
        offs = np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.repeat(lo, cnt) + (np.arange(total) - offs)
        rr = np.repeat(r_idx, cnt).astype(np.int64)
        tt = self.tix[pos].astype(np.int64)
        same = np.repeat(rf, cnt) == self.tstrand[pos]
        key = rr * T + tt
        ukey, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        fwd_votes = np.bincount(inv, weights=same.astype(np.float64), minlength=ukey.size)
        return ukey // T, ukey % T, counts, fwd_votes


class Kernel:
    name = "abstract"
    label = "abstract kernel"

    def available(self) -> bool:
        return False

    def map_reads(self, reads: Sequence[str]) -> MapResult:  # pragma: no cover - interface
        raise NotImplementedError


class KmerEqKernel(Kernel):
    """Tie-retaining k-mer kernel: every target within ``tie_slack`` k-mers of the best is kept."""

    name = "kmer_eq"
    label = "builtin canonical k-mer eq-class kernel (tie-retaining)"

    def __init__(self, index: KmerIndex, min_hits: int = 3, tie_slack: int = 2, ll_per_kmer: float = 0.5) -> None:
        self.index = index
        self.min_hits = min_hits
        self.tie_slack = tie_slack
        self.ll_per_kmer = ll_per_kmer

    def available(self) -> bool:
        return True

    def _collect(self, reads: Sequence[str]) -> MapResult:
        n = len(reads)
        cands: list = [None] * n
        decoy = np.zeros(n, dtype=bool)
        rr, tt, cc, fv = self.index.vote(reads)
        if rr.size == 0:
            return MapResult(cands, decoy)
        best = np.zeros(n, dtype=np.int64)
        np.maximum.at(best, rr, cc)
        keep = (cc >= self.min_hits) & (cc >= best[rr] - self.tie_slack)
        rr, tt, cc, fv = rr[keep], tt[keep], cc[keep], fv[keep]
        bounds = np.flatnonzero(np.diff(rr)) + 1
        for grp in np.split(np.arange(rr.size), bounds):
            if grp.size == 0:
                continue
            r = int(rr[grp[0]])
            g_t, g_c, g_f = tt[grp], cc[grp], fv[grp]
            top = g_c.max()
            if self.index.is_decoy[g_t[g_c == top]].any():
                decoy[r] = True
                continue
            nd = ~self.index.is_decoy[g_t]
            g_t, g_c, g_f = g_t[nd], g_c[nd], g_f[nd]
            cands[r] = (
                tuple(int(x) for x in g_t),
                tuple(float((c - top) * self.ll_per_kmer) for c in g_c),
                tuple(bool(f * 2 >= c) for f, c in zip(g_f, g_c)),
            )
        return MapResult(cands, decoy)

    def map_reads(self, reads: Sequence[str]) -> MapResult:
        return self._collect(reads)


class KmerBestHitKernel(KmerEqKernel):
    """Reports a single best target per read (lowest index on ties) -- collapses mate/strain ties."""

    name = "kmer_besthit"
    label = "builtin best-hit k-mer kernel (tie-collapsing baseline)"

    def map_reads(self, reads: Sequence[str]) -> MapResult:
        res = self._collect(reads)
        out = []
        for c in res.candidates:
            if c is None:
                out.append(None)
                continue
            ts, lls, fs = c
            j = max(range(len(ts)), key=lambda i: (lls[i], -ts[i]))
            out.append(((ts[j],), (0.0,), (fs[j],)))
        return MapResult(out, res.decoy)


@dataclass
class ExternalKernel(Kernel):
    """Placeholder adapter for an external aligner binary (bwa-mem2, minimap2, ...)."""

    name: str = "external"
    binary: str = ""
    label: str = field(default="")

    def available(self) -> bool:
        # A binary alone is not enough: SAM/PAF adapters are only shipped in the container image.
        return False

    def binary_present(self) -> bool:
        return bool(self.binary) and shutil.which(self.binary) is not None


EXTERNAL_BINARIES = {
    "bwa_spa": "bwa",
    "bwa-mem2": "bwa-mem2",
    "strobealign": "strobealign",
    "minimap2": "minimap2",
    "bowtie2": "bowtie2",
    "salmon": "salmon",
    "piscem": "piscem",
}


# Fixed argv prefixes for adapters whose name is not just the binary.
EXTERNAL_ARGV = {
    "bwa_spa": ("bwa", "mem", "-S", "-P", "-a"),  # skip mate rescue + pairing, report all alignments
}


def build_kernels(names: Sequence[str], index: KmerIndex) -> list[Kernel]:
    out: list[Kernel] = []
    for n in names:
        if n == "kmer_eq":
            out.append(KmerEqKernel(index))
        elif n == "kmer_besthit":
            out.append(KmerBestHitKernel(index))
        elif n in EXTERNAL_BINARIES:
            out.append(ExternalKernel(name=n, binary=EXTERNAL_BINARIES[n], label=f"{n} (external adapter)"))
    return out
