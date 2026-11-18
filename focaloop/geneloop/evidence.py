"""Compact RNA evidence stores: junction multiplicities, coverage bins, locus interval index."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import numpy as np

from .sequtil import Genome, revcomp

JKey = tuple[str, int, int]  # (contig, intron_start_0b, intron_end_0b_exclusive)

CANONICAL = {"GT-AG", "GC-AG", "AT-AC"}


def junction_motif(genome: Genome, contig: str, a: int, b: int) -> tuple[str, str]:
    """(motif, strand) for intron [a, b). Strand inferred from the canonical motif; '.' if non-canonical."""
    d = genome.fetch(contig, a, a + 2).upper()
    ac = genome.fetch(contig, b - 2, b).upper()
    plus = f"{d}-{ac}"
    if plus in CANONICAL:
        return plus, "+"
    minus = f"{revcomp(ac)}-{revcomp(d)}"
    if minus in CANONICAL:
        return minus, "-"
    return plus, "."


@dataclass
class JunctionStore:
    """Global junction table. ``ua`` = uniquely-mapped reads with min anchor >= 8 (the support unit)."""

    run_ids: list[str] = field(default_factory=list)
    ua: dict[JKey, dict[str, int]] = field(default_factory=dict)
    total: dict[JKey, int] = field(default_factory=dict)
    max_anchor: dict[JKey, int] = field(default_factory=dict)
    motif: dict[JKey, tuple[str, str]] = field(default_factory=dict)

    def ingest_tsv(self, run_id: str, path: str | Path, genome: Genome | None = None) -> list[tuple[JKey, int, int]]:
        """Merge one glcore junction TSV; returns the wave's [(key, n_ua, n_total)]."""
        if run_id not in self.run_ids:
            self.run_ids.append(run_id)
        out = []
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return out
        with open(p) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < 6:
                    continue
                k = (f[0], int(f[1]), int(f[2]))
                nt, nua, ma = int(f[3]), int(f[4]), int(f[5])
                d = self.ua.get(k)
                if d is None:
                    d = {}
                    self.ua[k] = d
                    if genome is not None:
                        self.motif[k] = junction_motif(genome, *k)
                d[run_id] = d.get(run_id, 0) + nua
                self.total[k] = self.total.get(k, 0) + nt
                if ma > self.max_anchor.get(k, 0):
                    self.max_anchor[k] = ma
                out.append((k, nua, nt))
        return out

    def support(self, k: JKey) -> int:
        d = self.ua.get(k)
        return sum(d.values()) if d else 0

    def canonical(self, k: JKey) -> bool:
        m = self.motif.get(k)
        return bool(m) and m[0] in CANONICAL

    def strand(self, k: JKey) -> str:
        m = self.motif.get(k)
        return m[1] if m else "."

    def by_contig(self, min_support: int = 2, canonical_only: bool = True) -> dict[str, list[tuple[int, int, int, str]]]:
        """contig -> sorted [(a, b, support, strand)] for supported junctions."""
        out: dict[str, list] = {}
        for k, d in self.ua.items():
            s = sum(d.values())
            if s < min_support:
                continue
            if canonical_only and not self.canonical(k):
                continue
            out.setdefault(k[0], []).append((k[1], k[2], s, self.strand(k)))
        for v in out.values():
            v.sort()
        return out

    def n_runs_supporting(self, k: JKey) -> int:
        d = self.ua.get(k)
        return sum(1 for v in d.values() if v > 0) if d else 0


class BinStore:
    """Uniquely-mapped read-start counts in fixed bins (default 1 kb)."""

    def __init__(self, lengths: dict[str, int], bin_size: int = 1000):
        self.bin = bin_size
        self.arr = {c: np.zeros(L // bin_size + 1, dtype=np.int64) for c, L in lengths.items()}
        self.total = 0

    def ingest_tsv(self, path: str | Path) -> int:
        p = Path(path)
        n = 0
        if not p.exists():
            return 0
        with open(p) as fh:
            for line in fh:
                c, b, v = line.rstrip("\n").split("\t")
                a = self.arr.get(c)
                if a is not None:
                    a[int(b)] += int(v)
                    n += int(v)
        self.total += n
        return n

    def reads_in(self, contig: str, a: int, b: int) -> int:
        arr = self.arr.get(contig)
        if arr is None:
            return 0
        return int(arr[a // self.bin : b // self.bin + 1].sum())

    def rpkm(self, contig: str, a: int, b: int) -> float:
        if self.total <= 0:
            return 0.0
        n = self.reads_in(contig, a, b)
        kb = max(0.001, ((b // self.bin + 1) - a // self.bin) * self.bin / 1000.0)
        return n / kb / (self.total / 1e6)

    def hotspots(self, frac: float = 0.001, min_reads: int = 1000) -> list[tuple[str, int, int, int]]:
        if self.total <= 0:
            return []
        thr = max(min_reads, frac * self.total)
        out = []
        for c, a in self.arr.items():
            for i in np.flatnonzero(a >= thr):
                out.append((c, int(i) * self.bin, int(i + 1) * self.bin, int(a[i])))
        return out

    def save_npz(self, path: str | Path) -> None:
        """Persist bin size, total read count, and all contig int64 arrays losslessly (pickle-free)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        contigs = list(self.arr.keys())
        meta_bytes = json.dumps({"bin": int(self.bin), "total": int(self.total), "contigs": contigs}).encode("utf-8")
        arrays = {"meta": np.frombuffer(meta_bytes, dtype=np.uint8)}
        for idx, c in enumerate(contigs):
            arrays[f"c{idx}"] = np.asarray(self.arr[c], dtype=np.int64)
        tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, p)

    @classmethod
    def load_npz(cls, path: str | Path) -> "BinStore":
        """Reload a lossless BinStore written by :meth:`save_npz`."""
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            contigs: list[str] = meta["contigs"]
            inst = cls({}, bin_size=int(meta["bin"]))
            inst.total = int(meta["total"])
            inst.arr = {c: np.array(z[f"c{idx}"], dtype=np.int64, copy=True) for idx, c in enumerate(contigs)}
        return inst


class LocusIndex:
    """Per-contig sorted span index (start, end, payload) for containment / overlap queries."""

    def __init__(self, spans: list[tuple[str, int, int, object]], pad: int = 0):
        self.by: dict[str, tuple[list[int], list[tuple[int, int, object]]]] = {}
        tmp: dict[str, list] = {}
        for c, a, b, p in spans:
            tmp.setdefault(c, []).append((max(0, a - pad), b + pad, p))
        self.maxlen = 0
        for c, v in tmp.items():
            v.sort(key=lambda x: x[0])
            self.by[c] = ([x[0] for x in v], v)
            for x in v:
                self.maxlen = max(self.maxlen, x[1] - x[0])

    def overlapping(self, contig: str, a: int, b: int) -> list[object]:
        e = self.by.get(contig)
        if not e:
            return []
        starts, v = e
        i = bisect.bisect_left(starts, b)
        out = []
        j = i - 1
        while j >= 0 and starts[j] >= a - self.maxlen:
            s, t, p = v[j]
            if s < b and a < t:
                out.append(p)
            j -= 1
        return out

    def containing(self, contig: str, a: int, b: int) -> list[object]:
        e = self.by.get(contig)
        if not e:
            return []
        starts, v = e
        i = bisect.bisect_right(starts, a)
        out = []
        j = i - 1
        while j >= 0 and starts[j] >= a - self.maxlen:
            s, t, p = v[j]
            if s <= a and b <= t:
                out.append(p)
            j -= 1
        return out
