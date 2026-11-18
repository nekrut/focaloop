"""Mapping kernels that turn read batches into soft counts + equivalence classes.

Every kernel implements ``map_batch(r1, r2) -> MapResult`` where ``r2`` is
``None`` for single-end data.  A read (pair) that is compatible with exactly one
target becomes a unique count; otherwise its candidate set (targets, orientation
flags, quantised log-likelihood deltas) becomes an equivalence class that the
solver re-weights backward with cohort / WGS priors.

Kernels:
  mappy             in-process libminimap2 ``-x sr`` with ``best_n=16`` (no index files)
  bwa_mem2_spa      ``bwa-mem2 mem -S -P -a`` (independent mates, all hits)
  bwa_mem2_naive_pa ``bwa-mem2 mem -p -a`` (proper-pair collapse; a known tie breaker)
  bowtie2_local_k16 ``bowtie2 --very-sensitive-local -k 16``
  bowtie2_global_k16``bowtie2 --very-sensitive -k 16``
  strobealign_N16   ``strobealign -N 16`` (paired rescue is asymmetric on ties)
  salmon            selective alignment with ``--dumpEq``
  kallisto          pseudoalignment via ``--pseudobam``
  star              ``STAR --quantMode TranscriptomeSAM`` when a spliced genome index exists

Kernels added by the open-source mapper survey (14_open_source_mapper_survey.md):
  bwa_spa           ``bwa mem -S -P -a`` (BWA 0.7.x, independent mates, all hits)
  hisat2_k16        ``hisat2 -k 16 --no-mixed --no-discordant`` on the catalog (spliced alignment left on)
  chromap           ``chromap --SAM -q 0 [--trim-adapters]`` (one best hit per read: no multi-hit option)
  subread           ``subread-align -t 0 --multiMapping -B 16``
  subjunc           ``subjunc --multiMapping -B 16`` (spliced, shares the subread index)
  snap              ``snap-aligner -om 1 -omax 16`` (edit-distance scored, NM-based likelihood)
  dragmap           ``dragen-os`` (DRAGEN-OS open-source mapper; secondary alignments where the build allows)
  piscem            ``piscem map-bulk`` RAD records (SSHash cDBG, Salmon/pufferfish successor), parsed natively
  bwa_meme_spa      ``bwa-meme mem -7 -S -P -a`` (learned-index BWA-MEM2; output identical to bwa-mem2 by design)
  gsnap_k16         ``gsnap -A sam --npaths 16 -N 1`` (spliced, novel splicing on)
  segemehl          ``segemehl.x -S`` (enhanced suffix array, split reads on)
  vg_giraffe        ``vg giraffe -M 16`` on a linear ``vg autoindex`` graph of the catalog
  themisto          ``themisto pseudoalign --threshold 0.7`` per mate, colour sets intersected (one colour per target)
  hisat2_genome_k16 HISAT2 on the primary GENOME (-k 16, annotated splice sites supplied, novel junctions allowed),
                    genomic alignments projected onto catalog targets by exon / decoy overlap
  star_genome_m16   STAR genomic SAM (--outSAMmultNmax 16) on the same STAR index, same projection

The orientation flag stored per candidate is True when the alignment is in the
dUTP (ISR / SR) orientation, so per-run strandedness can be inferred afterwards.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import ctypes
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Iterable

import numpy as np

from ._medvedev_compile import compile_c_lib

from .calibrator import OnlineErrorCalibrator
from .catalog import Catalog, Target, write_fasta
from .manifest import Run

from contextvars import ContextVar
import re as _re

Read = tuple[str, str]
# Pooled (multi-sample) batching: while a pooled call is active, resolve_pair routes each read into the MapResult of
# the sample that owns its global read index, and FASTQ writers name reads ``@<sample_id>|r<i>``.
_ROUTE: ContextVar = ContextVar("panexpr_route", default=None)
_NAME_PREFIX: ContextVar = ContextVar("panexpr_name_prefix", default=None)
EqKey = tuple[tuple[str, ...], tuple[bool, ...], tuple[float, ...]]
VIABLE_WINDOW_NATS = 18.0
MISS_PENALTY_NATS = 60.0


_TOOL_PROBE_CACHE: dict[str, str | None] = {}


def _bin_dir() -> Path | None:
    p = os.environ.get("PANEXPR_BIN")
    return Path(p) if p else None


def _probe_executable(exe: str, name: str) -> bool:
    """Verify binary actually executes on this host CPU without SIGILL/missing shared libs."""
    probe_args = {
        "bwa": [],
        "bwa-mem2": ["version"],
        "bwa-meme": ["version"],
    }.get(name, ["--version"])
    try:
        r = subprocess.run([exe, *probe_args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if r.returncode < 0 or r.returncode in (126, 127, 132, 134, 139):
            return False
        return True
    except Exception:
        return False


def tool_path(name: str) -> str | None:
    b = _bin_dir()
    cand = str(b / name) if (b and (b / name).exists()) else shutil.which(name)
    if not cand:
        return None
    if cand not in _TOOL_PROBE_CACHE:
        _TOOL_PROBE_CACHE[cand] = cand if _probe_executable(cand, name) else None
    return _TOOL_PROBE_CACHE[cand]


@dataclass
class MapResult:
    n_reads: int = 0
    n_mapped: int = 0
    unique_counts: dict[str, list[float]] = field(default_factory=dict)
    eq_classes: dict[EqKey, int] = field(default_factory=dict)
    bridge_pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    cal_counts: np.ndarray | None = None
    cal_n: int = 0
    wall_seconds: float = 0.0
    per_read: dict[int, tuple[tuple[str, ...], tuple[bool, ...]]] | None = None

    def merge(self, other: "MapResult") -> None:
        if other.per_read:
            if self.per_read is None:
                self.per_read = {}
            self.per_read.update(other.per_read)
        self.n_reads += other.n_reads
        self.n_mapped += other.n_mapped
        for t, (a, b) in other.unique_counts.items():
            slot = self.unique_counts.setdefault(t, [0.0, 0.0])
            slot[0] += a
            slot[1] += b
        for k, c in other.eq_classes.items():
            self.eq_classes[k] = self.eq_classes.get(k, 0) + c
        for k, c in other.bridge_pairs.items():
            self.bridge_pairs[k] = self.bridge_pairs.get(k, 0) + c
        if other.cal_counts is not None:
            self.cal_counts = other.cal_counts
            self.cal_n = other.cal_n
        self.wall_seconds += other.wall_seconds


def _collapse_decoy_eq_key(k: EqKey, decoy_ids: frozenset[str] | set[str]) -> EqKey | None:
    """Return None when an eq-class is 100 % decoy; collapse multi-decoy ties to one logsumexp decoy."""
    ts, flags, lls = k
    n_dec = 0
    for t in ts:
        if t in decoy_ids:
            n_dec += 1
    if n_dec == 0:
        return k
    if n_dec == len(ts):
        return None
    if n_dec == 1:
        return k
    kept: list[tuple[str, bool, float]] = []
    dec_cands: list[tuple[str, bool, float]] = []
    for t, fl, ll in zip(ts, flags, lls):
        if t in decoy_ids:
            dec_cands.append((t, fl, ll))
        else:
            kept.append((t, fl, ll))
    best_t, best_fl, best_ll = max(dec_cands, key=lambda x: x[2])
    lse = best_ll + math.log(sum(math.exp(x[2] - best_ll) for x in dec_cands))
    kept.append((best_t, best_fl, round(lse * 4.0) / 4.0))
    kept.sort(key=lambda x: (x[0], x[1]))
    return (
        tuple(x[0] for x in kept),
        tuple(x[1] for x in kept),
        tuple(x[2] for x in kept),
    )


@dataclass
class RunAccumulator:
    """Cumulative sufficient statistics for one run across loop iterations."""

    run: Run
    calibrator: OnlineErrorCalibrator = field(default_factory=OnlineErrorCalibrator)
    n_reads: int = 0
    n_mapped: int = 0
    unique_counts: dict[str, list[float]] = field(default_factory=dict)
    eq_classes: dict[EqKey, int] = field(default_factory=dict)
    bridge_pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    bytes_consumed: int = 0
    decoy_ids: frozenset[str] | set[str] | None = None
    decoy_reads: float = 0.0

    def add(self, res: MapResult) -> None:
        self.n_reads += res.n_reads
        self.n_mapped += res.n_mapped
        dset = self.decoy_ids
        for t, (a, b) in res.unique_counts.items():
            if dset and t in dset:
                self.decoy_reads += a + b
                continue
            slot = self.unique_counts.setdefault(t, [0.0, 0.0])
            slot[0] += a
            slot[1] += b
        if dset:
            for k, c in res.eq_classes.items():
                ck = _collapse_decoy_eq_key(k, dset)
                if ck is None:
                    self.decoy_reads += c
                else:
                    self.eq_classes[ck] = self.eq_classes.get(ck, 0) + c
        else:
            for k, c in res.eq_classes.items():
                self.eq_classes[k] = self.eq_classes.get(k, 0) + c
        for k, c in res.bridge_pairs.items():
            if dset and k[0] in dset and k[1] in dset:
                continue
            self.bridge_pairs[k] = self.bridge_pairs.get(k, 0) + c
        if res.cal_counts is not None:
            self.calibrator.counts = res.cal_counts
            self.calibrator.n_calibrated_reads = res.cal_n
            self.calibrator._dirty = True

    @property
    def mapped_fraction(self) -> float:
        return self.n_mapped / self.n_reads if self.n_reads else 0.0

    def total_soft_count(self, target_ids: set[str] | None = None) -> float:
        tot = sum(a + b for t, (a, b) in self.unique_counts.items() if target_ids is None or t in target_ids)
        if target_ids is None:
            tot += self.decoy_reads
        for (ts, _s, _l), c in self.eq_classes.items():
            if target_ids is None or any(t in target_ids for t in ts):
                tot += c
        return tot


# ---------------------------------------------------------------------------
# Candidate resolution shared by all alignment-style kernels
# ---------------------------------------------------------------------------

def resolve_pair(
    cands1: dict[tuple[str, bool], float],
    cands2: dict[tuple[str, bool], float] | None,
    res: MapResult,
    miss_penalty: float = MISS_PENALTY_NATS,
    read_idx: int | None = None,
) -> None:
    """Combine mate candidate dictionaries into one unique count or one equivalence class."""
    route = _ROUTE.get()
    if route is not None and read_idx is not None:
        res = route(read_idx)        # pooled batch: count into the owning sample, keep no per-read trace
        read_idx = None
    if cands2 is None:
        paired = dict(cands1)
    else:
        if not cands1 and not cands2:
            return
        shared = set(cands1) & set(cands2)
        paired = {}
        if shared:
            for k in shared:
                paired[k] = cands1[k] + cands2[k]
        else:
            for k, v in cands1.items():
                paired[k] = v - miss_penalty
            for k, v in cands2.items():
                if k not in paired or (v - miss_penalty) > paired[k]:
                    paired[k] = v - miss_penalty
            if cands1 and cands2:
                t1 = max(cands1.items(), key=lambda kv: kv[1])[0][0]
                t2 = max(cands2.items(), key=lambda kv: kv[1])[0][0]
                if t1 != t2:
                    key = (t1, t2) if t1 < t2 else (t2, t1)
                    res.bridge_pairs[key] = res.bridge_pairs.get(key, 0) + 1
    if not paired:
        return
    max_ll = max(paired.values())
    viable = [(k[0], k[1], v - max_ll) for k, v in paired.items() if (v - max_ll) >= -VIABLE_WINDOW_NATS]
    viable.sort(key=lambda x: (x[0], x[1]))
    res.n_mapped += 1
    if read_idx is not None:
        if res.per_read is None:
            res.per_read = {}
        res.per_read[read_idx] = (tuple(x[0] for x in viable), tuple(x[1] for x in viable))
    if len(viable) == 1:
        tid, flag, _ = viable[0]
        slot = res.unique_counts.setdefault(tid, [0.0, 0.0])
        slot[0 if flag else 1] += 1.0
    else:
        key: EqKey = (
            tuple(x[0] for x in viable),
            tuple(x[1] for x in viable),
            tuple(round(x[2] * 4.0) / 4.0 for x in viable),
        )
        res.eq_classes[key] = res.eq_classes.get(key, 0) + 1


def parse_sam_stream(lines: Iterable[bytes], as_scale: float, paired: bool, trace: bool = False) -> MapResult:
    """Stream SAM records grouped by query name into a MapResult (no BAM on disk)."""
    res = MapResult()
    cur: str | None = None
    c1: dict[tuple[str, bool], float] = {}
    c2: dict[tuple[str, bool], float] = {}
    seen = 0
    for raw in lines:
        if not raw or raw[0] == 64:  # '@' header
            continue
        parts = raw.rstrip(b"\n").split(b"\t")
        if len(parts) < 11:
            continue
        qn = parts[0]
        if qn.endswith(b"/1") or qn.endswith(b"/2"):
            qn = qn[:-2]
        qname = qn.decode("ascii", "replace")
        if qname != cur:
            if cur is not None:
                resolve_pair(c1, c2 if paired else None, res, read_idx=_qidx(cur) if trace else None)
                c1, c2 = {}, {}
            cur = qname
            seen += 1
        flag = int(parts[1])
        if flag & 4 or flag & 2048:
            continue
        rname = parts[2].decode("ascii", "replace")
        if rname == "*":
            continue
        score = 0.0
        if as_scale:
            for opt in parts[11:]:
                if opt.startswith(b"AS:i:"):
                    score = float(opt[5:]) * as_scale
                    break
        is_rev = bool(flag & 16)
        is_r2 = bool(flag & 128) and paired
        if not is_r2:
            key = (rname, is_rev)        # mate 1 reverse == dUTP orientation
            if key not in c1 or score > c1[key]:
                c1[key] = score
        else:
            key = (rname, not is_rev)    # mate 2 forward == dUTP orientation
            if key not in c2 or score > c2[key]:
                c2[key] = score
    if cur is not None:
        resolve_pair(c1, c2 if paired else None, res, read_idx=_qidx(cur) if trace else None)
    res.n_reads = seen
    return res


class _SamUnorderedRes(ctypes.Structure):
    _fields_ = [
        ("n_reads", ctypes.c_uint64),
        ("n_refs", ctypes.c_uint32),
        ("ref_names_len", ctypes.c_uint32),
        ("ref_names_buf", ctypes.POINTER(ctypes.c_char)),
        ("n_mapped_queries", ctypes.c_uint64),
        ("records_len", ctypes.c_uint64),
        ("records_buf", ctypes.POINTER(ctypes.c_uint8)),
    ]


_SAM_LIB_CACHE: ctypes.CDLL | None = None
_SAM_LIB_TRIED = False


def _load_sam_unordered_lib() -> ctypes.CDLL | None:
    global _SAM_LIB_CACHE, _SAM_LIB_TRIED
    if _SAM_LIB_TRIED:
        return _SAM_LIB_CACHE
    _SAM_LIB_TRIED = True
    try:
        c_path = Path(__file__).resolve().parent / "sam_unordered_core.c"
        if not c_path.exists():
            return None
        lib = compile_c_lib("sam_unordered", c_path)
        lib.sam_parse_fd.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
        lib.sam_parse_fd.restype = ctypes.POINTER(_SamUnorderedRes)
        lib.sam_free_result.argtypes = [ctypes.POINTER(_SamUnorderedRes)]
        lib.sam_free_result.restype = None
        _SAM_LIB_CACHE = lib
    except Exception:
        _SAM_LIB_CACHE = None
    return _SAM_LIB_CACHE


_SAM_Q_HDR = struct.Struct("<qII")
_SAM_CAND = struct.Struct("<IId")


def parse_sam_unordered(lines: Iterable[bytes], paired: bool, trace: bool = False, score_tag: bytes = b"AS:i:",
                        score_scale: float = 1.0, ref_names: dict[str, str] | None = None) -> MapResult:
    """Like parse_sam_stream, but records of one read need not be adjacent (coordinate-sorted SAM, e.g. chromap).

    ``score_tag`` / ``score_scale`` turn an integer SAM tag into a log-likelihood: ``AS:i:`` with a positive scale,
    or ``NM:i:`` with a negative scale (edit distance)."""
    if hasattr(lines, "fileno"):
        lib = _load_sam_unordered_lib()
        if lib is not None:
            can_use_fd = True
            try:
                if hasattr(lines, "seekable") and lines.seekable():
                    can_use_fd = lines.tell() == 0
            except Exception:
                can_use_fd = False
            if can_use_fd:
                res_ptr = lib.sam_parse_fd(
                    lines.fileno(), 1 if paired else 0, score_tag, len(score_tag), float(score_scale)
                )
                if not res_ptr:
                    raise RuntimeError("sam_parse_fd failed")
                try:
                    c_res = res_ptr.contents
                    res = MapResult()
                    raw_refs = (
                        ctypes.string_at(c_res.ref_names_buf, c_res.ref_names_len).split(b"\x00")[:-1]
                        if c_res.ref_names_len else []
                    )
                    refs = [
                        (ref_names.get(s, s) if ref_names is not None else s)
                        for s in (r.decode("ascii", "replace") for r in raw_refs)
                    ]
                    if c_res.records_len:
                        rec_bytes = (ctypes.c_uint8 * c_res.records_len).from_address(
                            ctypes.addressof(c_res.records_buf.contents)
                        )
                        mv = memoryview(rec_bytes)
                        off = 0
                        end = c_res.records_len
                        q_unpack = _SAM_Q_HDR.unpack_from
                        c_unpack = _SAM_CAND.unpack_from
                        while off < end:
                            ridx, n1, n2 = q_unpack(mv, off)
                            off += 16
                            c1: dict[tuple[str, bool], float] = {}
                            for _ in range(n1):
                                rid, rev, sc = c_unpack(mv, off)
                                off += 16
                                k = (refs[rid], bool(rev))
                                if k not in c1 or sc > c1[k]:
                                    c1[k] = sc
                            if paired:
                                c2: dict[tuple[str, bool], float] = {}
                                for _ in range(n2):
                                    rid, rev, sc = c_unpack(mv, off)
                                    off += 16
                                    k = (refs[rid], bool(rev))
                                    if k not in c2 or sc > c2[k]:
                                        c2[k] = sc
                            else:
                                c2 = None
                            resolve_pair(c1, c2, res, read_idx=(ridx if (trace and ridx >= 0) else None))
                    res.n_reads = int(c_res.n_reads)
                    return res
                finally:
                    lib.sam_free_result(res_ptr)
    res = MapResult()
    per: dict[str, tuple[dict, dict]] = {}
    order: list[str] = []
    for raw in lines:
        if not raw or raw[0] == 64:
            continue
        parts = raw.rstrip(b"\n").split(b"\t")
        if len(parts) < 11:
            continue
        qn = parts[0]
        if qn.endswith(b"/1") or qn.endswith(b"/2"):
            qn = qn[:-2]
        qname = qn.decode("ascii", "replace")
        slot = per.get(qname)
        if slot is None:
            slot = per[qname] = ({}, {})
            order.append(qname)
        flag = int(parts[1])
        if flag & 4 or flag & 2048:
            continue
        rname = parts[2].decode("ascii", "replace")
        if rname == "*":
            continue
        if ref_names is not None:
            rname = ref_names.get(rname, rname)
        score = 0.0
        if score_scale:
            for opt in parts[11:]:
                if opt.startswith(score_tag):
                    score = float(opt[len(score_tag):]) * score_scale
                    break
        is_rev = bool(flag & 16)
        if paired and flag & 128:
            key, d = (rname, not is_rev), slot[1]
        else:
            key, d = (rname, is_rev), slot[0]
        if key not in d or score > d[key]:
            d[key] = score
    for q in order:
        c1, c2 = per[q]
        resolve_pair(c1, c2 if paired else None, res, read_idx=_qidx(q) if trace else None)
    res.n_reads = len(order)
    return res


def _qidx(qname: str) -> int | None:
    if "|" in qname:                  # pooled read names: <sample_id>|r<i>
        qname = qname.rsplit("|", 1)[1]
    if qname.startswith("r") and qname[1:].isdigit():
        return int(qname[1:])
    return None


def _rname(i: int) -> str:
    pf = _NAME_PREFIX.get()
    return f"{pf(i)}|r{i}" if pf is not None else f"r{i}"


def _iter_interleaved_fastq_chunks(r1: list[Read], r2: list[Read] | None, chunk: int = 16_384, pf=None):
    """Yield ASCII-encoded interleaved FASTQ chunks to keep peak memory bounded."""
    if pf is None:
        pf = _NAME_PREFIX.get()
    if r2 is None:
        n = len(r1)
        for st in range(0, n, chunk):
            ed = min(n, st + chunk)
            if pf is None:
                yield "".join(f"@r{i}\n{r1[i][0]}\n+\n{r1[i][1]}\n" for i in range(st, ed)).encode("ascii", "replace")
            else:
                yield "".join(f"@{pf(i)}|r{i}\n{r1[i][0]}\n+\n{r1[i][1]}\n" for i in range(st, ed)).encode("ascii", "replace")
    else:
        n = min(len(r1), len(r2))
        for st in range(0, n, chunk):
            ed = min(n, st + chunk)
            if pf is None:
                yield "".join(
                    f"@r{i}/1\n{r1[i][0]}\n+\n{r1[i][1]}\n@r{i}/2\n{r2[i][0]}\n+\n{r2[i][1]}\n"
                    for i in range(st, ed)
                ).encode("ascii", "replace")
            else:
                yield "".join(
                    f"@{pf(i)}|r{i}/1\n{r1[i][0]}\n+\n{r1[i][1]}\n@{pf(i)}|r{i}/2\n{r2[i][0]}\n+\n{r2[i][1]}\n"
                    for i in range(st, ed)
                ).encode("ascii", "replace")


def interleave_fastq(r1: list[Read], r2: list[Read] | None) -> bytes:
    return b"".join(_iter_interleaved_fastq_chunks(r1, r2))


def write_interleaved_fastq(path: Path, r1: list[Read], r2: list[Read] | None) -> None:
    """Stream interleaved FASTQ to disk without materializing a single multi-GiB buffer."""
    with open(path, "wb", buffering=4 * 1024 * 1024) as fh:
        for blk in _iter_interleaved_fastq_chunks(r1, r2):
            fh.write(blk)


def write_fastq(path: Path, reads: list[Read], suffix: str = "") -> None:
    pf = _NAME_PREFIX.get()
    with open(path, "w", buffering=4 * 1024 * 1024) as fh:
        if pf is None:
            chunk = 32_768
            for st in range(0, len(reads), chunk):
                fh.write("".join(
                    f"@r{i}{suffix}\n{s}\n+\n{q}\n"
                    for i, (s, q) in enumerate(reads[st:st + chunk], st)
                ))
        else:
            for i, (s, q) in enumerate(reads):
                fh.write(f"@{pf(i)}|r{i}{suffix}\n{s}\n+\n{q}\n")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _fast_file_fingerprint(path: Path) -> str:
    """Deterministic O(1) content fingerprint (file byte size + 3x 1 MiB SHA-256 windows)."""
    import hashlib
    p = Path(path)
    sz = p.stat().st_size
    h = hashlib.sha256()
    h.update(f"size={sz}\n".encode())
    win = 1 << 20
    with open(p, "rb") as fh:
        h.update(fh.read(win))
        if sz > 2 * win:
            fh.seek(sz // 2)
            h.update(fh.read(win))
        if sz > win:
            fh.seek(max(0, sz - win))
            h.update(fh.read(win))
    return h.hexdigest()[:16]


def resolve_index_cache_dir(idx_dir: Path, kernel_name: str, key_files: list[Path], extra_tag: str = "",
                            index_cache_dir: str | Path | None = None) -> Path:
    """Return persistent organism-cache directory when index_cache_dir or PANEXPR_INDEX_CACHE is configured."""
    import hashlib
    root = index_cache_dir or os.environ.get("PANEXPR_INDEX_CACHE")
    if not root:
        return Path(idx_dir)
    h = hashlib.sha256()
    h.update(f"{kernel_name}|{extra_tag}|".encode())
    for fp in key_files:
        h.update(_fast_file_fingerprint(Path(fp)).encode())
        h.update(b"|")
    out = Path(root) / f"{kernel_name}_{h.hexdigest()[:16]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


class Kernel:
    name: str = "base"
    label: str = "base"
    kind: str = "base"
    symmetric_by_construction: bool = False
    needs_spliced_genome: bool = False
    # True when every read reaches resolve_pair with its batch index under trace=True and no per-read side pass is needed,
    # so several samples can share one tool invocation (one index load) via map_pooled().
    supports_pooling: bool = False

    def __init__(self, threads: int = 4) -> None:
        self.threads = threads
        self.index_paths: dict[str, Path] = {}

    def available(self) -> bool:  # pragma: no cover - trivial
        return True

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        self.index_paths["fasta"] = Path(fasta)

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "kind": self.kind}


_FORK_CTX: tuple | None = None


def _mappy_slice(aligner, cal: OnlineErrorCalibrator, r1: list[Read], r2: list[Read] | None, start: int, end: int, calibrate: bool, trace: bool = False) -> MapResult:
    import mappy

    res = MapResult()
    tbuf = mappy.ThreadBuffer()
    cal = cal.clone() if calibrate else cal
    for i in range(start, end):
        s1, q1 = r1[i]
        mq1 = OnlineErrorCalibrator.mean_q(q1)
        hits1 = list(aligner.map(s1, buf=tbuf))
        if r2 is not None:
            s2, q2 = r2[i]
            mq2 = OnlineErrorCalibrator.mean_q(q2)
            hits2 = list(aligner.map(s2, buf=tbuf))
        else:
            hits2 = []
        res.n_reads += 1
        if not hits1 and not hits2:
            continue
        if calibrate and cal.n_calibrated_reads < 5000:
            if hits1 and hits1[0].mapq >= 20:
                cal.update_from_hit(mq1, 0, hits1[0].mlen, hits1[0].blen, hits1[0].NM)
            if hits2 and hits2[0].mapq >= 20:
                cal.update_from_hit(mq2, 1, hits2[0].mlen, hits2[0].blen, hits2[0].NM)
        c1: dict[tuple[str, bool], float] = {}
        for h in hits1:
            key = (h.ctg, h.strand == -1)
            ll = cal.score_alignment(len(s1), mq1, 0, h.mlen, h.blen, h.NM)
            if key not in c1 or ll > c1[key]:
                c1[key] = ll
        if r2 is None:
            resolve_pair(c1, None, res, read_idx=i if trace else None)
            continue
        c2: dict[tuple[str, bool], float] = {}
        for h in hits2:
            key = (h.ctg, h.strand == 1)
            ll = cal.score_alignment(len(s2), mq2, 1, h.mlen, h.blen, h.NM)
            if key not in c2 or ll > c2[key]:
                c2[key] = ll
        resolve_pair(c1, c2, res, miss_penalty=len(s1) * -math.log(0.02), read_idx=i if trace else None)
    if calibrate:
        res.cal_counts = cal.counts.copy()
        res.cal_n = cal.n_calibrated_reads
    return res


def _fork_worker(task: tuple[int, int]) -> MapResult:
    assert _FORK_CTX is not None
    aligner, cal, r1, r2, trace = _FORK_CTX
    return _mappy_slice(aligner, cal, r1, r2, task[0], task[1], False, trace)


class MappyKernel(Kernel):
    name = "mappy"
    label = "minimap2 -x sr (in-process libminimap2, best_n=16)"
    kind = "mappy"
    symmetric_by_construction = True

    def __init__(self, threads: int = 4, preset: str = "sr", best_n: int = 16) -> None:
        super().__init__(threads)
        self.preset = preset
        self.best_n = best_n
        self.aligner = None

    def available(self) -> bool:
        try:
            import mappy  # noqa: F401
            return True
        except Exception:
            return False

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        import mappy

        self.index_paths["fasta"] = Path(fasta)
        self.aligner = mappy.Aligner(str(fasta), preset=self.preset, best_n=self.best_n)
        if not self.aligner:
            raise RuntimeError(f"mappy failed to index {fasta}")

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        global _FORK_CTX
        assert self.aligner is not None, "call ensure_index first"
        t0 = time.time()
        cal = calibrator or OnlineErrorCalibrator()
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        out = MapResult()
        cur = 0
        if cal.n_calibrated_reads < 5000:
            w_end = min(n, 4000)
            out.merge(_mappy_slice(self.aligner, cal, r1, r2, 0, w_end, True, trace))
            if out.cal_counts is not None:
                cal.counts = out.cal_counts
                cal.n_calibrated_reads = out.cal_n
                cal._dirty = True
            cur = w_end
        if cur < n:
            remaining = n - cur
            use_fork = sys.platform == "linux" and remaining >= 4000 and self.threads > 1
            if use_fork:
                nproc = min(self.threads, 16)
                step = max(1, (remaining + nproc - 1) // nproc)
                spans = [(s, min(n, s + step)) for s in range(cur, n, step)]
                _FORK_CTX = (self.aligner, cal, r1, r2, trace)
                try:
                    with mp.get_context("fork").Pool(processes=min(nproc, len(spans))) as pool:
                        for res in pool.map(_fork_worker, spans):
                            out.merge(res)
                finally:
                    _FORK_CTX = None
            else:
                out.merge(_mappy_slice(self.aligner, cal, r1, r2, cur, n, False, trace))
        out.n_reads = n
        out.wall_seconds = time.time() - t0
        return out


class SamKernel(Kernel):
    """External aligner fed interleaved/single FASTQ on stdin, SAM parsed from stdout."""

    supports_pooling = True

    kind = "sam"

    def __init__(self, name: str, label: str, exe: str, as_scale: float, pe_args: list[str], se_args: list[str],
                 index_kind: str, threads: int = 4, symmetric: bool = False) -> None:
        super().__init__(threads)
        self.name = name
        self.label = label
        self.exe = exe
        self.as_scale = as_scale
        self.pe_args = pe_args
        self.se_args = se_args
        self.index_kind = index_kind
        self.symmetric_by_construction = symmetric

    def available(self) -> bool:
        return tool_path(self.exe) is not None

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        idx_dir = Path(idx_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)
        self.index_paths["fasta"] = Path(fasta)
        if self.index_kind == "bwa":
            prefix = idx_dir / "bwa_mem2_idx"
            if not Path(f"{prefix}.bwt.2bit.64").exists():
                subprocess.run([tool_path("bwa-mem2"), "index", "-p", str(prefix), str(fasta)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.index_paths["idx"] = prefix
        elif self.index_kind == "bt2":
            prefix = idx_dir / "bowtie2_idx"
            if not Path(f"{prefix}.1.bt2").exists() and not Path(f"{prefix}.1.bt2l").exists():
                subprocess.run([tool_path("bowtie2-build"), "--quiet", "--threads", str(self.threads), str(fasta), str(prefix)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.index_paths["idx"] = prefix
        else:
            self.index_paths["idx"] = Path(fasta)

    def command(self, paired: bool) -> list[str]:
        idx = str(self.index_paths["idx"])
        args = self.pe_args if paired else self.se_args
        return [tool_path(self.exe)] + [a.format(t=self.threads, idx=idx) for a in args]

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        paired = r2 is not None
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        if n == 0:
            return MapResult()
        proc = subprocess.Popen(self.command(paired), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=4 * 1024 * 1024)
        assert proc.stdin is not None and proc.stdout is not None
        pf = _NAME_PREFIX.get()

        def _feed() -> None:
            try:
                for blk in _iter_interleaved_fastq_chunks(r1[:n], r2[:n] if paired else None, pf=pf):
                    proc.stdin.write(blk)
            except BrokenPipeError:
                pass
            finally:
                proc.stdin.close()

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_feed)
            res = parse_sam_stream(proc.stdout, self.as_scale, paired, trace)
            fut.result()
        proc.wait()
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


def _salmon_lib_orientation(q_dir: Path) -> tuple[str, float]:
    """Infer ('reverse' | 'forward' | 'unstranded', p_rev) from Salmon's lib_format_counts.json."""
    lf = q_dir / "lib_format_counts.json"
    if not lf.exists():
        return "reverse", 1.0
    try:
        d = json.loads(lf.read_text())
    except Exception:
        return "reverse", 1.0
    exp = str(d.get("expected_format", "")).upper()
    n_rev = float(sum(d.get(k, 0) for k in ("ISR", "OSR", "MSR", "SR")))
    n_fwd = float(sum(d.get(k, 0) for k in ("ISF", "OSF", "MSF", "SF")))
    p_rev = n_rev / (n_rev + n_fwd) if (n_rev + n_fwd) > 0 else 0.5
    if exp in {"ISR", "OSR", "MSR", "SR"}:
        return "reverse", max(p_rev, 0.85)
    if exp in {"ISF", "OSF", "MSF", "SF"}:
        return "forward", min(p_rev, 0.15)
    if exp in {"IU", "OU", "MU", "U"}:
        return "unstranded", min(0.75, max(0.25, p_rev))
    if p_rev >= 0.8:
        return "reverse", p_rev
    if p_rev <= 0.2:
        return "forward", p_rev
    return "unstranded", p_rev


def _salmon_eff_lengths(q_dir: Path) -> dict[str, float]:
    """Read per-transcript EffectiveLength from Salmon's quant.sf so --dumpEqWeights cancels 1/effLen."""
    qsf = q_dir / "quant.sf"
    out: dict[str, float] = {}
    if not qsf.exists():
        return out
    with open(qsf, "rt") as fh:
        _hdr = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    out[parts[0]] = max(1.0, float(parts[2]))
                except ValueError:
                    pass
    return out


class SalmonKernel(Kernel):
    name = "salmon"
    label = "salmon selective alignment (--dumpEqWeights)"
    kind = "salmon"
    symmetric_by_construction = True

    def __init__(self, threads: int = 4, extra_args: list[str] | None = None, tmp_dir: str | Path | None = None,
                 clip_adapters: bool = True) -> None:
        super().__init__(threads)
        self.extra_args = extra_args if extra_args is not None else ["--validateMappings"]
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.clip_adapters = clip_adapters

    def available(self) -> bool:
        return tool_path("salmon") is not None

    def clone_with_threads(self, threads: int) -> "SalmonKernel":
        c = SalmonKernel(threads=threads, extra_args=list(self.extra_args), tmp_dir=self.tmp_dir,
                         clip_adapters=self.clip_adapters)
        c.index_paths = dict(self.index_paths)
        return c

    def ensure_index(self, fasta: Path, idx_dir: Path, decoys_txt: Path | None = None, **kw) -> None:
        eff_dir = resolve_index_cache_dir(
            Path(idx_dir), self.name, [Path(fasta)] + ([Path(decoys_txt)] if decoys_txt else []),
            extra_tag="k31", index_cache_dir=kw.get("index_cache_dir"),
        )
        idx = eff_dir / "salmon_idx"
        self.index_paths["fasta"] = Path(fasta)
        if not (idx / "versionInfo.json").exists() and not (idx / "info.json").exists():
            cmd = [tool_path("salmon"), "index", "-t", str(fasta), "-i", str(idx), "-k", "31", "-p", str(self.threads)]
            if decoys_txt is not None:
                cmd += ["-d", str(decoys_txt)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.index_paths["idx"] = idx

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        res = MapResult()
        if n == 0:
            return res
        r1_in = r1[:n]
        r2_in = r2[:n] if r2 is not None else None
        if self.clip_adapters:
            from ._fast_c import clip_adapters_batch
            r1_in, r2_in, _ = clip_adapters_batch(r1_in, r2_in, min_len=31, threads=self.threads)
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            f1 = td / "r1.fq"
            write_fastq(f1, r1_in, "/1" if r2_in is not None else "")
            cmd = [tool_path("salmon"), "quant", "-i", str(self.index_paths["idx"]), "-l", "A", "-p", str(self.threads),
                   "-o", str(td / "q"), "--dumpEqWeights", "-q"] + self.extra_args
            if trace:
                cmd += ["--writeMappings", str(td / "map.sam")]
            if r2_in is not None:
                f2 = td / "r2.fq"
                write_fastq(f2, r2_in, "/2")
                cmd += ["-1", str(f1), "-2", str(f2)]
            else:
                cmd += ["-r", str(f1)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if trace and (td / "map.sam").exists():
                with open(td / "map.sam", "rb") as fh:
                    traced = parse_sam_stream(fh, 0.0, r2 is not None, trace=True)
                res.per_read = traced.per_read or {}
            mode, p_rev = _salmon_lib_orientation(td / "q")
            eff_lens = _salmon_eff_lengths(td / "q")
            flag_val = False if mode == "forward" else True
            eq = td / "q" / "aux_info" / "eq_classes.txt.gz"
            if not eq.exists():
                eq = td / "q" / "aux_info" / "eq_classes.txt"
            opener = gzip.open if str(eq).endswith(".gz") else open
            with opener(eq, "rt") as fh:
                n_tx = int(fh.readline())
                n_eq = int(fh.readline())
                names = [fh.readline().strip() for _ in range(n_tx)]
                for _ in range(n_eq):
                    parts = fh.readline().split()
                    if not parts:
                        break
                    k = int(parts[0])
                    tids = [names[int(x)] for x in parts[1:1 + k]]
                    cnt = int(float(parts[-1]))
                    if k == 1:
                        slot = res.unique_counts.setdefault(tids[0], [0.0, 0.0])
                        if mode == "reverse":
                            slot[0] += cnt
                        elif mode == "forward":
                            slot[1] += cnt
                        else:
                            slot[0] += cnt * p_rev
                            slot[1] += cnt * (1.0 - p_rev)
                    else:
                        order = sorted(range(k), key=lambda i: tids[i])
                        if len(parts) >= 2 * k + 2:
                            a_probs = [
                                max(1e-12, float(parts[1 + k + i]) * eff_lens.get(tids[i], 1.0))
                                for i in order
                            ]
                            a_max = max(a_probs)
                            lls = tuple(
                                0.0 if abs(v := max(-VIABLE_WINDOW_NATS, round(math.log(p / a_max) * 4.0) / 4.0)) == 0.0 else v
                                for p in a_probs
                            )
                        else:
                            lls = tuple(0.0 for _ in order)
                        key: EqKey = (tuple(tids[i] for i in order), tuple(flag_val for _ in order), lls)
                        res.eq_classes[key] = res.eq_classes.get(key, 0) + cnt
                    res.n_mapped += cnt
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


class KallistoKernel(Kernel):
    supports_pooling = True
    name = "kallisto"
    label = "kallisto pseudoalignment (--pseudobam)"
    kind = "kallisto"
    symmetric_by_construction = True

    def __init__(self, threads: int = 4, frag_len: float = 200.0, frag_sd: float = 30.0, tmp_dir: str | Path | None = None) -> None:
        super().__init__(threads)
        self.frag_len = frag_len
        self.frag_sd = frag_sd
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None

    def available(self) -> bool:
        return tool_path("kallisto") is not None and tool_path("samtools") is not None

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        idx = Path(idx_dir) / "kallisto.idx"
        self.index_paths["fasta"] = Path(fasta)
        if not idx.exists():
            ktmp = Path(idx_dir) / "kallisto_tmp"      # kallisto defaults to ./tmp, which collides between concurrent jobs
            subprocess.run([tool_path("kallisto"), "index", "-i", str(idx), "-t", str(self.threads), "-T", str(ktmp), str(fasta)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(ktmp, ignore_errors=True)
        self.index_paths["idx"] = idx

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        if n == 0:
            return MapResult()
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            f1 = td / "r1.fq"
            write_fastq(f1, r1[:n], "/1" if r2 is not None else "")
            cmd = [tool_path("kallisto"), "quant", "-i", str(self.index_paths["idx"]), "-o", str(td / "q"),
                   "-t", str(self.threads), "--pseudobam", "--plaintext"]
            if r2 is not None:
                f2 = td / "r2.fq"
                write_fastq(f2, r2[:n], "/2")
                cmd += [str(f1), str(f2)]
            else:
                cmd += ["--single", "-l", str(self.frag_len), "-s", str(self.frag_sd), str(f1)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            bam = td / "q" / "pseudoalignments.bam"
            proc = subprocess.Popen([tool_path("samtools"), "view", str(bam)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            res = parse_sam_stream(proc.stdout, 0.0, r2 is not None, trace)
            proc.wait()
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


SECONDARY_GENOME_CONTIG = "panexpr_secondary_chr"
_SECONDARY_PAD_BP = 200


def _fasta_contig_names(path: Path) -> set[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    names: set[str] = set()
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                names.add(line[1:].strip().split()[0])
    return names


def secondary_target_coords(
    catalog: Catalog,
    primary_contigs: set[str] | None = None,
) -> dict[str, tuple[str, int, int]]:
    """Deterministic 1-based layout of secondary-assembly targets onto one synthetic contig."""
    if all(t.assembly_id == catalog.primary_assembly for t in catalog.targets.values()):
        return {}
    ordered = [t for t in catalog.targets.values() if not t.is_decoy] + [t for t in catalog.targets.values() if t.is_decoy]
    out: dict[str, tuple[str, int, int]] = {}
    pos = 1
    for t in ordered:
        if t.assembly_id == catalog.primary_assembly:
            continue
        if primary_contigs is not None and t.contig in primary_contigs and t.start_1b > 0:
            continue
        if not t.seq:
            continue
        end = pos + len(t.seq) - 1
        out[t.target_id] = (SECONDARY_GENOME_CONTIG, pos, end)
        pos = end + 1 + _SECONDARY_PAD_BP
    return out


def prepare_augmented_genome(catalog: Catalog, genome_fasta: Path, out_path: Path) -> tuple[Path, set[str] | None]:
    """Append secondary-assembly targets as a padded synthetic chromosome when absent from genome_fasta."""
    genome_fasta = Path(genome_fasta)
    if all(t.assembly_id == catalog.primary_assembly for t in catalog.targets.values()):
        return genome_fasta, None
    prim_ctgs = _fasta_contig_names(genome_fasta)
    sec = secondary_target_coords(catalog, prim_ctgs)
    if not sec:
        return genome_fasta, prim_ctgs
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if str(genome_fasta).endswith(".gz") else open
        with open(out_path, "w") as out_fh:
            last_nl = True
            with opener(genome_fasta, "rt") as in_fh:
                for chunk in iter(lambda: in_fh.read(1 << 20), ""):
                    out_fh.write(chunk)
                    last_nl = chunk.endswith("\n")
            if not last_nl:
                out_fh.write("\n")
            out_fh.write(f">{SECONDARY_GENOME_CONTIG}\n")
            pad = "N" * _SECONDARY_PAD_BP
            full_seq = pad.join(catalog.targets[tid].seq for tid in sec)
            for i in range(0, len(full_seq), 80):
                out_fh.write(full_seq[i:i + 80] + "\n")
    return out_path, prim_ctgs


class StarKernel(Kernel):
    supports_pooling = True
    name = "star"
    label = "STAR --quantMode TranscriptomeSAM (spliced genome index)"
    kind = "star"
    needs_spliced_genome = True
    symmetric_by_construction = True

    def __init__(self, threads: int = 4, tmp_dir: str | Path | None = None) -> None:
        super().__init__(threads)
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None

    def available(self) -> bool:
        return tool_path("STAR") is not None and tool_path("samtools") is not None

    def clone_with_threads(self, threads: int) -> "StarKernel":
        c = StarKernel(threads=threads, tmp_dir=self.tmp_dir)
        c.index_paths = dict(self.index_paths)
        return c

    def ensure_index(self, fasta: Path, idx_dir: Path, catalog: Catalog | None = None, genome_fasta: Path | None = None,
                     sjdb_overhang: int = 100, **kw) -> None:
        if catalog is None or genome_fasta is None:
            raise RuntimeError("STAR kernel needs catalog + genome_fasta")
        has_sec = any(t.assembly_id != catalog.primary_assembly for t in catalog.targets.values())
        hash_inputs = [Path(genome_fasta), Path(fasta)] if (has_sec and Path(fasta).exists()) else [Path(genome_fasta)]
        eff_dir = resolve_index_cache_dir(
            Path(idx_dir), "star_genome", hash_inputs,
            extra_tag=f"sjdb={sjdb_overhang}|ntx={len(catalog.gene_targets)}|nasm={len(catalog.assemblies)}" if has_sec else f"sjdb={sjdb_overhang}|ntx={len(catalog.gene_targets)}",
            index_cache_dir=kw.get("index_cache_dir"),
        )
        idx = eff_dir / "star_idx"
        self.index_paths["fasta"] = Path(fasta)
        if not (idx / "SA").exists():
            idx.mkdir(parents=True, exist_ok=True)
            eff_genome, prim_ctgs = prepare_augmented_genome(catalog, Path(genome_fasta), idx / "augmented_genome.fa")
            gtf = idx / "catalog_transcripts.gtf"
            write_catalog_gtf(catalog, gtf, primary_contigs=prim_ctgs)
            glen = sum(len(s) for s in _fasta_lengths(eff_genome))
            sa_n = max(4, min(14, int(math.log2(max(glen, 2)) / 2 - 1)))
            cmd = [tool_path("STAR"), "--runMode", "genomeGenerate", "--runThreadN", str(self.threads), "--genomeDir", str(idx),
                   "--genomeFastaFiles", str(eff_genome), "--sjdbGTFfile", str(gtf), "--sjdbOverhang", str(sjdb_overhang),
                   "--genomeSAindexNbases", str(sa_n), "--outFileNamePrefix", str(idx) + "/"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.index_paths["idx"] = idx

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        if n == 0:
            return MapResult()
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            f1 = td / "r1.fq"
            write_fastq(f1, r1[:n])
            files = [str(f1)]
            if r2 is not None:
                f2 = td / "r2.fq"
                write_fastq(f2, r2[:n])
                files.append(str(f2))
            cmd = [tool_path("STAR"), "--runThreadN", str(self.threads), "--genomeDir", str(self.index_paths["idx"]),
                   "--readFilesIn", *files, "--quantMode", "TranscriptomeSAM", "--outSAMtype", "None",
                   "--outFilterMultimapNmax", "20",
                   "--quantTranscriptomeBan", "Singleend", "--outFileNamePrefix", str(td / "s_"),
                   "--outSAMattributes", "NH", "HI", "AS", "nM"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            bam = td / "s_Aligned.toTranscriptome.out.bam"
            proc = subprocess.Popen([tool_path("samtools"), "view", str(bam)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            res = parse_sam_stream(proc.stdout, 1.0, r2 is not None, trace)
            proc.wait()
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


def _fasta_lengths(path: Path) -> list[str]:
    from .catalog import load_fasta
    return list(load_fasta(path).values())


def write_catalog_gtf(catalog: Catalog, path: Path, primary_contigs: set[str] | None = None) -> Path:
    """GTF whose transcript_id equals the catalog target_id (including secondary-assembly targets on synthetic contig)."""
    sec = secondary_target_coords(catalog, primary_contigs)
    with open(path, "w") as fh:
        for t in catalog.gene_targets:
            if t.assembly_id == catalog.primary_assembly or (primary_contigs is not None and t.contig in primary_contigs and t.start_1b > 0):
                exons = t.exons or [(t.start_1b, t.end_1b)]
                ctg, strand = t.contig, t.strand
            elif t.target_id in sec:
                ctg, s, e = sec[t.target_id]
                exons, strand = [(s, e)], "+"
            else:
                continue
            for s, e in exons:
                fh.write(f'{ctg}\tpanexpr\texon\t{s}\t{e}\t.\t{strand}\t.\tgene_id "{t.group_id}"; transcript_id "{t.target_id}";\n')
    return path


# ---------------------------------------------------------------------------
# File-based SAM kernels (tools that need read files and/or write SAM to a file or in coordinate order)
# ---------------------------------------------------------------------------


def tool_version(exe: str, args: tuple[str, ...] = ("--version",)) -> str:
    """First non-empty line a tool prints for its version flag ('' when the tool is missing)."""
    p = tool_path(exe)
    if not p:
        return ""
    try:
        out = subprocess.run([p, *args], capture_output=True, text=True, timeout=60)
    except Exception:
        return ""
    for line in (out.stdout + "\n" + out.stderr).splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


@dataclass
class FileSamSpec:
    exe: str
    label: str
    index_name: str                       # directory entry under idx_dir ({idx})
    marker: tuple[str, ...]               # index is complete when any of these exists ({idx} substituted)
    build: list[list[str]]                # index build commands ({fasta} {idx} {t} {idx_dir})
    pe: list[str]                         # mapping args ({idx} {t} {r1} {r2} {out} {fasta} {ilv})
    se: list[str]
    output: str = "stdout"                # stdout | file
    input: str = "files"                  # files | interleaved
    score_tag: str = "AS:i:"
    score_scale: float = 0.0
    mkdir_idx: bool = False               # create {idx} as a directory before building
    version_args: tuple[str, ...] = ("--version",)
    symmetric: bool = False
    extra_tools: tuple[str, ...] = ()
    rename_refs: bool = False             # index a copy with names r<i> and 60-bp lines ('|' and long lines break the tool)


class FileSamKernel(Kernel):
    """External aligner run on temporary FASTQ files; SAM parsed without assuming read-grouped output."""

    supports_pooling = True

    kind = "sam_file"

    def __init__(self, name: str, spec: FileSamSpec, threads: int = 4, tmp_dir: str | Path | None = None) -> None:
        super().__init__(threads)
        self.name = name
        self.label = spec.label
        self.spec = spec
        self.ref_names: dict[str, str] | None = None
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.symmetric_by_construction = spec.symmetric

    def available(self) -> bool:
        return tool_path(self.spec.exe) is not None and all(tool_path(t) for t in self.spec.extra_tools)

    def _fmt(self, args: list[str], **kw) -> list[str]:
        return [a.format(**kw) for a in args]

    def index_complete(self, idx: Path) -> bool:
        return any(Path(m.format(idx=idx)).exists() for m in self.spec.marker)

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        idx_dir = Path(idx_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx = idx_dir / self.spec.index_name
        self.index_paths["fasta"] = Path(fasta)
        names_tsv = Path(f"{idx}.names.tsv")
        if not self.index_complete(idx):
            if self.spec.mkdir_idx:
                idx.mkdir(parents=True, exist_ok=True)
            src = Path(fasta)
            if self.spec.rename_refs:
                src = Path(f"{idx}.renamed.fa")
                _write_renamed_fasta(Path(fasta), src, names_tsv)
            for cmd in self.spec.build:
                c = self._fmt(cmd, fasta=src, idx=idx, t=self.threads, idx_dir=idx_dir)
                if self.name == "snap" and src.stat().st_size > (2 << 30):
                    c += ["-locationSize", "5"]     # padded multi-contig catalogs above ~4 G locations need 5-byte locations
                c[0] = tool_path(c[0]) or c[0]
                subprocess.run(c, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(idx_dir))
            if not self.index_complete(idx):
                raise RuntimeError(f"{self.name}: index markers {self.spec.marker} missing after build")
            if self.spec.rename_refs:
                src.unlink(missing_ok=True)
        self.index_paths["idx"] = idx
        self.ref_names = None
        if self.spec.rename_refs:
            self.ref_names = dict(line.rstrip("\n").split("\t", 1) for line in open(names_tsv))

    def version(self) -> str:
        return tool_version(self.spec.exe, self.spec.version_args)

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        paired = r2 is not None
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        if n == 0:
            return MapResult()
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            f1, f2, ilv, out = td / "r1.fq", td / "r2.fq", td / "ilv.fq", td / "out.sam"
            if self.spec.input == "interleaved":
                write_interleaved_fastq(ilv, r1[:n], r2[:n] if paired else None)
            else:
                write_fastq(f1, r1[:n], "/1" if paired else "")
                if paired:
                    write_fastq(f2, r2[:n], "/2")
            args = self.spec.pe if paired else self.spec.se
            cmd = [tool_path(self.spec.exe)] + self._fmt(args, idx=self.index_paths["idx"], t=self.threads, r1=f1, r2=f2, reads=f1,
                                                        out=out, fasta=self.index_paths["fasta"], ilv=ilv)
            sc = float(self.spec.score_scale)
            tag = self.spec.score_tag.encode()
            if self.spec.output == "stdout":
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(td), bufsize=4 * 1024 * 1024)
                assert proc.stdout is not None
                res = parse_sam_unordered(proc.stdout, paired, trace, tag, sc, self.ref_names)
                if proc.wait() != 0:
                    raise RuntimeError(f"{self.name} exited with {proc.returncode}")
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(td))
                with open(out, "rb") as fh:
                    res = parse_sam_unordered(fh, paired, trace, tag, sc, self.ref_names)
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


def _write_renamed_fasta(src: Path, dst: Path, names_tsv: Path, width: int = 60) -> None:
    from .catalog import load_fasta
    with open(dst, "w") as out, open(names_tsv, "w") as nm:
        for i, (name, seq) in enumerate(load_fasta(src).items()):
            out.write(f">r{i}\n")
            for j in range(0, len(seq), width):
                out.write(seq[j:j + width] + "\n")
            nm.write(f"r{i}\t{name}\n")


FILE_SAM_SPECS: dict[str, FileSamSpec] = {
    "bwa_spa": FileSamSpec(
        "bwa", "BWA 0.7.x (-S -P -a, independent mates)", "bwa_idx", ("{idx}.bwt",),
        [["bwa", "index", "-p", "{idx}", "{fasta}"]],
        ["mem", "-t", "{t}", "-p", "-S", "-P", "-a", "-T", "20", "{idx}", "{ilv}"],
        ["mem", "-t", "{t}", "-a", "-T", "20", "{idx}", "{ilv}"],
        input="interleaved", score_scale=1.15, version_args=(), symmetric=True),
    "hisat2_k16": FileSamSpec(
        "hisat2", "HISAT2 (-k 16, spliced alignment on, no mixed/discordant)", "hisat2_idx", ("{idx}.1.ht2", "{idx}.1.ht2l"),
        [["hisat2-build", "-p", "{t}", "{fasta}", "{idx}"]],
        ["-p", "{t}", "-k", "16", "--no-mixed", "--no-discordant", "--no-unal", "-x", "{idx}", "-1", "{r1}", "-2", "{r2}"],
        ["-p", "{t}", "-k", "16", "--no-unal", "-x", "{idx}", "-U", "{reads}"],
        score_scale=0.72, extra_tools=("hisat2-build",)),
    "chromap": FileSamSpec(
        "chromap", "chromap (--SAM -q 0, --trim-adapters on PE; one best hit per read)", "chromap.idx", ("{idx}",),
        [["chromap", "-i", "-r", "{fasta}", "-o", "{idx}"]],
        ["-x", "{idx}", "-r", "{fasta}", "-1", "{r1}", "-2", "{r2}", "--SAM", "-o", "{out}", "-q", "0", "-t", "{t}", "--trim-adapters"],
        ["-x", "{idx}", "-r", "{fasta}", "-1", "{reads}", "--SAM", "-o", "{out}", "-q", "0", "-t", "{t}"],
        output="file", score_tag="NM:i:", score_scale=-4.0),
    "subread": FileSamSpec(
        "subread-align", "Subread (seed-and-vote, -t 0 --multiMapping -B 16)", "subread_idx", ("{idx}.00.b.array",),
        [["subread-buildindex", "-o", "{idx}", "{fasta}"]],
        ["-t", "0", "-T", "{t}", "-i", "{idx}", "-r", "{r1}", "-R", "{r2}", "-o", "{out}", "--SAMoutput", "--multiMapping", "-B", "16"],
        ["-t", "0", "-T", "{t}", "-i", "{idx}", "-r", "{reads}", "-o", "{out}", "--SAMoutput", "--multiMapping", "-B", "16"],
        output="file", score_tag="NM:i:", score_scale=-4.0, version_args=("-v",), extra_tools=("subread-buildindex",), rename_refs=True),
    "subjunc": FileSamSpec(
        "subjunc", "Subjunc (spliced seed-and-vote, --multiMapping -B 16)", "subread_idx", ("{idx}.00.b.array",),
        [["subread-buildindex", "-o", "{idx}", "{fasta}"]],
        ["-T", "{t}", "-i", "{idx}", "-r", "{r1}", "-R", "{r2}", "-o", "{out}", "--SAMoutput", "--multiMapping", "-B", "16"],
        ["-T", "{t}", "-i", "{idx}", "-r", "{reads}", "-o", "{out}", "--SAMoutput", "--multiMapping", "-B", "16"],
        output="file", score_tag="NM:i:", score_scale=-4.0, version_args=("-v",), extra_tools=("subread-buildindex",), rename_refs=True),
    "snap": FileSamSpec(
        "snap-aligner", "SNAP 2.x (-om 1 -omax 16, edit-distance hash aligner)", "snap_idx", ("{idx}/GenomeIndex",),
        [["snap-aligner", "index", "{fasta}", "{idx}", "-t{t}"]],
        ["paired", "{idx}", "{r1}", "{r2}", "-o", "{out}", "-t", "{t}", "-om", "1", "-omax", "16", "-F", "a"],
        ["single", "{idx}", "{reads}", "-o", "{out}", "-t", "{t}", "-om", "1", "-omax", "16", "-F", "a"],
        output="file", score_tag="NM:i:", score_scale=-4.0, version_args=()),
    "dragmap": FileSamSpec(
        "dragen-os", "DRAGMAP (DRAGEN-OS, hash-table seed + Smith-Waterman)", "dragmap_idx", ("{idx}/hash_table.cfg",),
        [["dragen-os", "--build-hash-table", "true", "--ht-reference", "{fasta}", "--output-directory", "{idx}", "--ht-num-threads", "{t}"]],
        ["-r", "{idx}", "-1", "{r1}", "-2", "{r2}", "--num-threads", "{t}", "--Aligner.sec-aligns", "15", "--Aligner.sec-score-delta", "10",
         "--Aligner.sec-phred-delta", "60"],
        ["-r", "{idx}", "-1", "{reads}", "--num-threads", "{t}", "--Aligner.sec-aligns", "15", "--Aligner.sec-score-delta", "10",
         "--Aligner.sec-phred-delta", "60"],
        score_scale=0.5, mkdir_idx=True),
    "bwa_meme_spa": FileSamSpec(
        "bwa-meme", "BWA-MEME (-7 learned index, -S -P -a)", "bwa_meme_idx", ("{idx}.suffixarray_uint64_L2_PARAMETERS",),
        [["bwa-meme", "index", "-a", "meme", "-t", "{t}", "-p", "{idx}", "{fasta}"], ["build_rmis_dna.sh", "{idx}"]],
        ["mem", "-7", "-t", "{t}", "-p", "-S", "-P", "-a", "-T", "20", "{idx}", "{ilv}"],
        ["mem", "-7", "-t", "{t}", "-a", "-T", "20", "{idx}", "{ilv}"],
        input="interleaved", score_scale=1.15, version_args=("version",), symmetric=True, extra_tools=("build_rmis_dna.sh",)),
    "gsnap_k16": FileSamSpec(
        "gsnap", "GSNAP (--npaths 16, novel splicing on)", "gsnap_db", ("{idx}/gsnap_db.genomecomp",),
        [["gmap_build", "-D", "{idx_dir}", "-d", "gsnap_db", "{fasta}"]],
        ["-D", "{idx}/..", "-d", "gsnap_db", "-t", "{t}", "-A", "sam", "--npaths", "16", "-N", "1", "{r1}", "{r2}"],
        ["-D", "{idx}/..", "-d", "gsnap_db", "-t", "{t}", "-A", "sam", "--npaths", "16", "-N", "1", "{reads}"],
        score_tag="NM:i:", score_scale=-4.0, extra_tools=("gmap_build",)),
    "segemehl": FileSamSpec(
        "segemehl.x", "segemehl 0.3 (enhanced suffix array, -S split reads)", "segemehl.idx", ("{idx}",),
        [["segemehl.x", "-x", "{idx}", "-d", "{fasta}", "-t", "{t}"]],
        ["-i", "{idx}", "-d", "{fasta}", "-q", "{r1}", "-p", "{r2}", "-t", "{t}", "-S", "splits"],
        ["-i", "{idx}", "-d", "{fasta}", "-q", "{reads}", "-t", "{t}", "-S", "splits"],
        score_tag="NM:i:", score_scale=-4.0, version_args=("--version",)),
    "vg_giraffe": FileSamSpec(
        "vg", "vg giraffe (-M 16, linear autoindex graph of the catalog)", "vg_idx", ("{idx}.giraffe.gbz",),
        [["vg", "autoindex", "--workflow", "giraffe", "-r", "{fasta}", "-p", "{idx}", "-t", "{t}"]],
        ["giraffe", "-Z", "{idx}.giraffe.gbz", "-f", "{r1}", "-f", "{r2}", "-t", "{t}", "-o", "SAM", "-M", "16"],
        ["giraffe", "-Z", "{idx}.giraffe.gbz", "-f", "{reads}", "-t", "{t}", "-o", "SAM", "-M", "16"],
        score_scale=0.5, version_args=("version",)),
}


class ThemistoKernel(Kernel):
    """Colored de Bruijn graph pseudoalignment; one colour per catalog target (``--sequence-colors``).

    Mates are pseudoaligned separately and their colour sets intersected (the union is used when the intersection is
    empty).  Themisto reports no orientation, so every candidate carries the dUTP flag and strand purity is
    uninformative for this kernel."""

    supports_pooling = True

    name = "themisto"
    label = "Themisto 3 (colored SBWT cDBG, --threshold 0.7, mates intersected)"
    kind = "themisto"
    symmetric_by_construction = True

    def __init__(self, threads: int = 4, tmp_dir: str | Path | None = None, threshold: float = 0.7, mem_gb: int = 16) -> None:
        super().__init__(threads)
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.threshold = threshold
        self.mem_gb = mem_gb
        self.names: list[str] = []

    def available(self) -> bool:
        return tool_path("themisto") is not None

    def version(self) -> str:
        return tool_version("themisto", ("--version",))

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        idx_dir = Path(idx_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx = idx_dir / "themisto_idx"
        self.index_paths["fasta"] = Path(fasta)
        names_txt = idx_dir / "themisto_idx.names.txt"
        if not Path(f"{idx}.tcolors").exists():
            tmp = idx_dir / "themisto_tmp"
            tmp.mkdir(exist_ok=True)
            subprocess.run([tool_path("themisto"), "build", "-k", "31", "-i", str(fasta), "-o", str(idx), "--temp-dir", str(tmp),
                            "--sequence-colors", "-t", str(self.threads), "--mem-gigas", str(self.mem_gb)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(tmp, ignore_errors=True)
            with open(fasta) as fh, open(names_txt, "w") as out:
                for line in fh:
                    if line.startswith(">"):
                        out.write(line[1:].split()[0] + "\n")
        self.names = names_txt.read_text().split("\n")[:-1]
        self.index_paths["idx"] = idx

    def _pseudo(self, td: Path, mates: list[list[Read]]) -> list[dict[int, list[int]]]:
        """One themisto call for all mate files (--query-file-list), so the index is loaded once per batch."""
        qs, outs = [], []
        for i, reads in enumerate(mates):
            fq, out = td / f"m{i + 1}.fq", td / f"m{i + 1}.txt"
            write_fastq(fq, reads)
            qs.append(str(fq)); outs.append(str(out))
        (td / "queries.txt").write_text("\n".join(qs) + "\n")
        (td / "outs.txt").write_text("\n".join(outs) + "\n")
        r = subprocess.run([tool_path("themisto"), "pseudoalign", "--query-file-list", str(td / "queries.txt"), "-i", str(self.index_paths["idx"]),
                            "--out-file-list", str(td / "outs.txt"), "--temp-dir", str(td), "-t", str(self.threads), "--threshold", str(self.threshold)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"themisto pseudoalign exited {r.returncode}: {r.stderr.strip()[-600:]}")
        res = []
        for out in outs:
            hits: dict[int, list[int]] = {}
            with open(out) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) > 1:
                        hits[int(p[0])] = [int(x) for x in p[1:]]
            res.append(hits)
        return res

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        res = MapResult()
        if n == 0:
            return res
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            hs = self._pseudo(td, [r1[:n]] + ([r2[:n]] if r2 is not None else []))
            h1, h2 = hs[0], (hs[1] if r2 is not None else {})
        for i in range(n):
            a, b = set(h1.get(i, ())), set(h2.get(i, ()))
            cs = (a & b) or (a | b)
            if cs:
                resolve_pair({(self.names[c], True): 0.0 for c in cs}, None, res, read_idx=i if trace else None)
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


# ---------------------------------------------------------------------------
# Spliced GENOME aligners with exon-overlap projection onto catalog targets
# ---------------------------------------------------------------------------

_CIGAR = __import__("re").compile(rb"(\d+)([MIDNSHP=X])")


def cigar_blocks(pos_1b: int, cigar: bytes) -> list[tuple[int, int]]:
    """Reference blocks (1-based inclusive) covered by aligned bases; N splits blocks, D extends them."""
    blocks: list[tuple[int, int]] = []
    cur = pos_1b
    bs = cur
    for n, op in _CIGAR.findall(cigar):
        n = int(n)
        if op in b"M=XD":
            cur += n
        elif op == b"N":
            if cur > bs:
                blocks.append((bs, cur - 1))
            cur += n
            bs = cur
    if cur > bs:
        blocks.append((bs, cur - 1))
    return blocks


class ExonProjector:
    """Genomic interval -> catalog targets.  Gene targets contribute their exons, decoys their single interval."""

    def __init__(self, rows: list[tuple[str, str, str, list[tuple[int, int]]]]) -> None:
        import bisect  # noqa: F401
        by_ctg: dict[str, list[tuple[int, int, int]]] = {}
        self.tids: list[str] = []
        self.strand: list[str] = []
        for i, (tid, ctg, strand, ivs) in enumerate(rows):
            self.tids.append(tid)
            self.strand.append(strand)
            for s, e in ivs:
                by_ctg.setdefault(ctg, []).append((s, e, i))
        self.idx: dict[str, tuple[list[int], list[int], list[int], list[int]]] = {}
        for ctg, ivs in by_ctg.items():
            ivs.sort()
            starts = [a for a, _, _ in ivs]
            ends = [b for _, b, _ in ivs]
            owners = [c for _, _, c in ivs]
            pm, m = [], 0
            for b in ends:
                m = max(m, b)
                pm.append(m)
            self.idx[ctg] = (starts, ends, owners, pm)

    @classmethod
    def from_tsv(cls, path: Path) -> "ExonProjector":
        rows = []
        with open(path) as fh:
            for line in fh:
                tid, ctg, strand, ex = line.rstrip("\n").split("\t")
                rows.append((tid, ctg, strand, [tuple(int(x) for x in iv.split("-")) for iv in ex.split(",")]))
        return cls(rows)

    def coverage(self, ctg: str, blocks: list[tuple[int, int]]) -> dict[int, int]:
        import bisect
        ix = self.idx.get(ctg)
        cov: dict[int, int] = {}
        if ix is None:
            return cov
        starts, ends, owners, pm = ix
        for s, e in blocks:
            j = bisect.bisect_right(starts, e) - 1
            while j >= 0 and pm[j] >= s:
                if ends[j] >= s:
                    ov = min(e, ends[j]) - max(s, starts[j]) + 1
                    if ov > 0:
                        cov[owners[j]] = cov.get(owners[j], 0) + ov
                j -= 1
        return cov


def write_projection_tsv(catalog: Catalog, path: Path, primary_contigs: set[str] | None = None) -> int:
    """Catalog targets with genomic coordinates (including secondary-assembly targets on synthetic contig)."""
    sec = secondary_target_coords(catalog, primary_contigs)
    n = 0
    with open(path, "w") as fh:
        for t in catalog.targets.values():
            if (t.assembly_id == catalog.primary_assembly or (primary_contigs is not None and t.contig in primary_contigs)) and t.contig and t.start_1b > 0:
                ivs = t.exons if t.exons else [(t.start_1b, t.end_1b)]
                fh.write(f"{t.target_id}\t{t.contig}\t{t.strand}\t{','.join(f'{a}-{b}' for a, b in ivs)}\n")
                n += 1
            elif t.target_id in sec:
                ctg, s, e = sec[t.target_id]
                fh.write(f"{t.target_id}\t{ctg}\t+\t{s}-{e}\n")
                n += 1
    return n


def write_hisat2_splice_sites(catalog: Catalog, path: Path, primary_contigs: set[str] | None = None) -> int:
    """hisat2_extract_splice_sites.py format: contig, 0-based last base of upstream exon, 0-based first base of downstream exon, strand."""
    seen = set()
    with open(path, "w") as fh:
        for t in catalog.gene_targets:
            if not (t.assembly_id == catalog.primary_assembly or (primary_contigs is not None and t.contig in primary_contigs)):
                continue
            if len(t.exons) < 2:
                continue
            ivs = sorted(t.exons)
            for (_s1, e1), (s2, _e2) in zip(ivs, ivs[1:]):
                key = (t.contig, e1 - 1, s2 - 1, t.strand)
                if key not in seen:
                    seen.add(key)
                    fh.write("%s\t%d\t%d\t%s\n" % key)
    return len(seen)


class GenomeProjectKernel(Kernel):
    """Spliced genome alignment (HISAT2 or STAR) + projection of every alignment onto overlapping catalog targets.

    A target is a candidate for an alignment when its exons (or decoy span) cover at least half of the aligned bases
    and at least the best target's coverage minus 10 bp.  Secondary-assembly targets are appended onto a padded
    synthetic cDNA chromosome so cross-assembly ties are retained alongside primary genomic exons."""

    supports_pooling = True

    kind = "genome_project"
    needs_spliced_genome = True

    def __init__(self, tool: str, threads: int = 4, tmp_dir: str | Path | None = None) -> None:
        super().__init__(threads)
        if tool not in ("hisat2", "star"):
            raise ValueError(tool)
        self.tool = tool
        self.name = "hisat2_genome_k16" if tool == "hisat2" else "star_genome_m16"
        self.label = ("HISAT2 genome (-k 16, known + novel splice sites) -> exon projection" if tool == "hisat2"
                      else "STAR genome (--outSAMmultNmax 16) -> exon projection")
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.proj: ExonProjector | None = None
        self.as_scale = 0.72 if tool == "hisat2" else 1.0

    def clone_with_threads(self, threads: int) -> "GenomeProjectKernel":
        c = GenomeProjectKernel(self.tool, threads=threads, tmp_dir=self.tmp_dir)
        c.index_paths = dict(self.index_paths)
        c.proj = self.proj
        return c

    def available(self) -> bool:
        exe = ("hisat2", "hisat2-build") if self.tool == "hisat2" else ("STAR",)
        return all(tool_path(x) for x in exe)

    def version(self) -> str:
        return tool_version("hisat2" if self.tool == "hisat2" else "STAR")

    def ensure_index(self, fasta: Path, idx_dir: Path, catalog: Catalog | None = None, genome_fasta: Path | None = None,
                     sjdb_overhang: int = 100, **kw) -> None:
        idx_dir = Path(idx_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)
        self.index_paths["fasta"] = Path(fasta)
        prim_ctgs: set[str] | None = None
        eff_genome: Path | None = Path(genome_fasta) if genome_fasta is not None else None
        if catalog is not None and eff_genome is not None:
            eff_genome, prim_ctgs = prepare_augmented_genome(catalog, eff_genome, idx_dir / "augmented_genome.fa")
        proj = idx_dir / "genome_projection.tsv"
        if not proj.exists():
            if catalog is None:
                raise RuntimeError(f"{self.name}: projection table missing and no catalog given")
            write_projection_tsv(catalog, proj, primary_contigs=prim_ctgs)
        if self.tool == "hisat2":
            idx = idx_dir / "hisat2_genome"
            ss = idx_dir / "hisat2_genome.splice_sites.txt"
            if not Path(f"{idx}.1.ht2").exists() and not Path(f"{idx}.1.ht2l").exists():
                if eff_genome is None:
                    raise RuntimeError("hisat2_genome_k16 needs genome_fasta to build")
                subprocess.run([tool_path("hisat2-build"), "-p", str(self.threads), str(eff_genome), str(idx)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not ss.exists():
                if catalog is None:
                    raise RuntimeError("hisat2_genome_k16: splice-site file missing and no catalog given")
                write_hisat2_splice_sites(catalog, ss, primary_contigs=prim_ctgs)
            self.index_paths.update(idx=idx, ss=ss)
        else:
            sk = StarKernel(self.threads, tmp_dir=self.tmp_dir)
            has_sec = catalog is not None and any(t.assembly_id != catalog.primary_assembly for t in catalog.targets.values())
            hash_inputs = ([Path(genome_fasta), Path(fasta)] if (has_sec and genome_fasta and Path(fasta).exists())
                           else ([Path(genome_fasta)] if genome_fasta else [Path(fasta)]))
            extra_tag = (
                f"sjdb={sjdb_overhang}|ntx={len(catalog.gene_targets)}|nasm={len(catalog.assemblies)}"
                if (has_sec and catalog is not None)
                else f"sjdb={sjdb_overhang}|ntx={len(catalog.gene_targets) if catalog else 0}"
            )
            eff_star_dir = resolve_index_cache_dir(idx_dir, "star_genome", hash_inputs, extra_tag=extra_tag, index_cache_dir=kw.get("index_cache_dir"))
            idx = eff_star_dir / "star_idx"
            if not (idx / "SA").exists():
                sk.ensure_index(fasta, idx_dir, catalog=catalog, genome_fasta=genome_fasta, sjdb_overhang=sjdb_overhang, **kw)
                idx = sk.index_paths["idx"]
            self.index_paths["idx"] = idx
        self.proj = ExonProjector.from_tsv(proj)

    def _cands(self, recs: list[tuple[str, int, bytes, bool, float]], mate2: bool) -> dict[tuple[str, bool], float]:
        out: dict[tuple[str, bool], float] = {}
        assert self.proj is not None
        for ctg, pos, cigar, is_rev, score in recs:
            blocks = cigar_blocks(pos, cigar)
            aligned = sum(e - s + 1 for s, e in blocks)
            cov = self.proj.coverage(ctg, blocks)
            if not cov:
                continue
            best = max(cov.values())
            if best < 0.5 * aligned:
                continue
            for ti, c in cov.items():
                if c >= best - 10:
                    minus = self.proj.strand[ti] == "-"
                    rev_vs_tx = is_rev != minus
                    flag = (not rev_vs_tx) if mate2 else rev_vs_tx
                    key = (self.proj.tids[ti], flag)
                    ll = score - 0.05 * (aligned - c)
                    if key not in out or ll > out[key]:
                        out[key] = ll
        return out

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        t0 = time.time()
        paired = r2 is not None
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        res = MapResult()
        if n == 0:
            return res
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            f1, f2 = td / "r1.fq", td / "r2.fq"
            write_fastq(f1, r1[:n], "/1" if paired else "")
            if paired:
                write_fastq(f2, r2[:n], "/2")
            if self.tool == "hisat2":
                cmd = [tool_path("hisat2"), "-p", str(self.threads), "-k", "16", "--no-unal", "-x", str(self.index_paths["idx"]),
                       "--known-splicesite-infile", str(self.index_paths["ss"])]
                cmd += (["--no-mixed", "--no-discordant", "-1", str(f1), "-2", str(f2)] if paired else ["-U", str(f1)])
            else:
                cmd = [tool_path("STAR"), "--runThreadN", str(self.threads), "--genomeDir", str(self.index_paths["idx"]),
                       "--readFilesIn", str(f1)] + ([str(f2)] if paired else []) + [
                       "--outSAMtype", "SAM", "--outStd", "SAM", "--outSAMmultNmax", "16", "--outFilterMultimapNmax", "20",
                       "--outSAMattributes", "NH", "HI", "AS", "nM", "--outFileNamePrefix", str(td / "s_")]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=str(td), bufsize=4 * 1024 * 1024)
            assert proc.stdout is not None
            per: dict[str, tuple[list, list]] = {}
            order: list[str] = []
            for raw in proc.stdout:
                if not raw or raw[0] == 64:
                    continue
                parts = raw.rstrip(b"\n").split(b"\t")
                if len(parts) < 11:
                    continue
                flag = int(parts[1])
                qn = parts[0]
                if qn.endswith(b"/1") or qn.endswith(b"/2"):
                    qn = qn[:-2]
                q = qn.decode("ascii", "replace")
                slot = per.get(q)
                if slot is None:
                    slot = per[q] = ([], [])
                    order.append(q)
                if flag & 4 or flag & 2048 or parts[2] == b"*":
                    continue
                sc = 0.0
                for opt in parts[11:]:
                    if opt.startswith(b"AS:i:"):
                        sc = float(opt[5:]) * self.as_scale
                        break
                rec = (parts[2].decode(), int(parts[3]), parts[5], bool(flag & 16), sc)
                (slot[1] if (paired and flag & 128) else slot[0]).append(rec)
            if proc.wait() != 0:
                raise RuntimeError(f"{self.name} exited with {proc.returncode}")
        # STAR with --outStd SAM may still write Aligned.out.sam names without /1 /2; handled by the stripping above
        for q in order:
            a, b = per[q]
            c1 = self._cands(a, False)
            c2 = self._cands(b, True) if paired else None
            resolve_pair(c1, c2, res, read_idx=_qidx(q) if trace else None)
        res.n_reads = n
        res.wall_seconds = time.time() - t0
        return res


# ---------------------------------------------------------------------------
# piscem (RAD output, parsed natively)
# ---------------------------------------------------------------------------

SENTINEL_ID = "PANEXPR_RAD_SENTINEL"


def _sentinel_seq() -> str:
    import random as _r
    rng = _r.Random(20261109)
    return "".join(rng.choice("ACGT") for _ in range(1200))


def read_rad_bulk(path: str | Path) -> tuple[list[str], list[list[tuple[int, int]]]]:
    """Parse a piscem ``map-bulk`` RAD file (bulk / bulk_with_pos).

    Returns (reference names, records) where every record is the list of (ref index, orientation) alignments of
    one mapped fragment, in file order.  ``compressed_ori_ref`` keeps the reference id in the low 30 bits and the
    orientation in the top two: 0b10 = read 1 forward, 0b01 = read 1 reverse complement (0b00 / 0b11 unknown).  Unmapped fragments are absent from RAD.  Tag descriptions are read from the
    header, so extra read- or alignment-level tags are skipped by their declared widths."""
    import struct

    data = Path(path).read_bytes()
    o = 0
    if data[:8] == b"RAD_FILE":
        o = 8 + 6                                    # magic + format version / flags
    is_paired = data[o]; o += 1
    (nref,) = struct.unpack_from("<Q", data, o); o += 8
    names = []
    for _ in range(nref):
        (L,) = struct.unpack_from("<H", data, o); o += 2
        names.append(data[o:o + L].decode()); o += L
    (_nchunks,) = struct.unpack_from("<Q", data, o); o += 8
    width = {1: 1, 2: 2, 3: 4, 4: 8, 5: 4, 6: 8}

    def read_descs():
        nonlocal o
        (cnt,) = struct.unpack_from("<H", data, o); o += 2
        out = []
        for _ in range(cnt):
            (L,) = struct.unpack_from("<H", data, o); o += 2
            nm = data[o:o + L].decode(); o += L
            ty = data[o]; o += 1
            sub = None
            if ty == 7:
                sub = (data[o], data[o + 1]); o += 2
            o += 2                                    # tag id
            out.append((nm, ty, sub))
        return out

    file_tags = read_descs()
    read_tags = read_descs()
    aln_tags = read_descs()
    for _nm, ty, sub in file_tags:                    # skip file-level tag values
        if ty == 8:
            (L,) = struct.unpack_from("<H", data, o); o += 2 + L
        elif ty == 7:
            lw, ew = width[sub[0]], width[sub[1]]
            n = int.from_bytes(data[o:o + lw], "little"); o += lw + n * ew
        else:
            o += width[ty]
    rw = sum(width[t] for _, t, _ in read_tags)
    aw = [width[t] for _, t, _ in aln_tags]
    ori_i = next(i for i, (nm, _, _) in enumerate(aln_tags) if nm.startswith("compressed_ori_ref"))
    ori_off = sum(aw[:ori_i])
    astride = sum(aw)
    recs: list[list[tuple[int, int]]] = []
    while o + 8 <= len(data):
        nbytes, nrec = struct.unpack_from("<II", data, o)
        end = o + nbytes
        o += 8
        for _ in range(nrec):
            (na,) = struct.unpack_from("<I", data, o); o += 4 + rw
            alns = []
            for _a in range(na):
                (v,) = struct.unpack_from("<I", data, o + ori_off)
                alns.append((v & 0x3FFFFFFF, v >> 30))
                o += astride
            recs.append(alns)
        o = end
    _ = is_paired
    return names, recs


class PiscemKernel(Kernel):
    name = "piscem"
    label = "piscem map-bulk (SSHash cDBG + RAD, Salmon/pufferfish successor)"
    kind = "piscem"
    symmetric_by_construction = True
    trace_costs_extra_pass = True       # trace mode adds a single-thread sentinel pass; wall_seconds excludes it

    def __init__(self, threads: int = 4, tmp_dir: str | Path | None = None, k: int = 31, m: int = 19,
                 clip_adapters: bool = True) -> None:
        super().__init__(threads)
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.k, self.m = k, m
        self.clip_adapters = clip_adapters
        self._rad_meta: dict = {}

    def available(self) -> bool:
        return tool_path("piscem") is not None

    def version(self) -> str:
        return tool_version("piscem")

    def clone_with_threads(self, threads: int) -> "PiscemKernel":
        c = PiscemKernel(threads=threads, tmp_dir=self.tmp_dir, k=self.k, m=self.m,
                         clip_adapters=self.clip_adapters)
        c.index_paths = dict(self.index_paths)
        c._rad_meta = self._rad_meta
        return c

    def ensure_index(self, fasta: Path, idx_dir: Path, **kw) -> None:
        idx_dir = resolve_index_cache_dir(Path(idx_dir), self.name, [Path(fasta)], extra_tag=f"k{self.k}_m{self.m}", index_cache_dir=kw.get("index_cache_dir"))
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx = idx_dir / "piscem_idx"
        self.index_paths["fasta"] = Path(fasta)
        if not Path(f"{idx}.ssi").exists() and not Path(f"{idx}_ver.json").exists():
            ref = idx_dir / "piscem_ref_with_sentinel.fa"
            with open(ref, "w") as out, open(fasta) as fin:
                shutil.copyfileobj(fin, out)
                out.write(f"\n>{SENTINEL_ID}\n{_sentinel_seq()}\n")
            subprocess.run([tool_path("piscem"), "build", "-s", str(ref), "-k", str(self.k), "-m", str(self.m), "-t", str(self.threads),
                            "-o", str(idx)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(idx_dir))
            ref.unlink(missing_ok=True)
        self.index_paths["idx"] = idx

    def _run_cmd(self, td: Path, r1: list[Read], r2: list[Read] | None, threads: int, tag: str) -> Path:
        f1, f2 = td / f"{tag}_1.fq", td / f"{tag}_2.fq"
        write_fastq(f1, r1)
        cmd = [tool_path("piscem"), "map-bulk", "-i", str(self.index_paths["idx"]), "-t", str(threads), "-o", str(td / tag)]
        if r2 is not None:
            write_fastq(f2, r2)
            cmd += ["-1", str(f1), "-2", str(f2)]
        else:
            cmd += ["-r", str(f1)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return td / f"{tag}.rad"

    def _run(self, td: Path, r1: list[Read], r2: list[Read] | None, threads: int, tag: str):
        rad_path = self._run_cmd(td, r1, r2, threads, tag)
        return read_rad_bulk(rad_path)

    @staticmethod
    def _cands(alns: list[tuple[int, int]], names: list[str]) -> dict[tuple[str, bool], float]:
        # dUTP orientation == read 1 reverse complement (ori 0b01); forward (0b10) is the other strand
        return {(names[r], ori != 2): 0.0 for r, ori in alns}

    def map_batch(self, r1: list[Read], r2: list[Read] | None, calibrator: OnlineErrorCalibrator | None = None,
                  trace: bool = False) -> MapResult:
        from ._fast_c import c_rad_aggregate, clip_adapters_batch
        n = len(r1) if r2 is None else min(len(r1), len(r2))
        res = MapResult()
        if n == 0:
            return res
        r1_in = r1[:n]
        r2_in = r2[:n] if r2 is not None else None
        if self.tmp_dir is not None:
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as td:
            td = Path(td)
            t0 = time.time()
            if self.clip_adapters:
                r1_in, r2_in, _ = clip_adapters_batch(r1_in, r2_in, min_len=max(25, self.k), threads=self.threads)
            rad_path = self._run_cmd(td, r1_in, r2_in, self.threads, "main")
            c_agg = c_rad_aggregate(rad_path, SENTINEL_ID, self._rad_meta)
            if c_agg is not None:
                res.n_mapped, res.unique_counts, res.eq_classes = c_agg
                wall = time.time() - t0
            else:
                names, recs = read_rad_bulk(rad_path)
                wall = time.time() - t0
                for alns in recs:
                    c = self._cands([a for a in alns if names[a[0]] != SENTINEL_ID], names)
                    if c:
                        resolve_pair(c, None, res)
            if trace:
                # single-thread pass with a sentinel fragment after every read: RAD keeps input order with one
                # thread and drops unmapped fragments, so sentinel records delimit read indices.
                s = _sentinel_seq()
                sr1 = (revcomp_str(s[400:400 + min(100, len(r1[0][0]))]), "I" * min(100, len(r1[0][0])))
                sr2 = (s[100:100 + min(100, len(r1[0][0]))], "I" * min(100, len(r1[0][0])))
                t1, t2 = [], ([] if r2_in is not None else None)
                for i in range(n):
                    t1 += [r1_in[i], sr1]
                    if t2 is not None:
                        t2 += [r2_in[i], sr2]
                tnames, trecs = self._run(td, t1, t2, 1, "trace")
                sid = tnames.index(SENTINEL_ID)
                per: dict[int, tuple[tuple[str, ...], tuple[bool, ...]]] = {}
                i = 0
                for alns in trecs:
                    if alns and all(r == sid for r, _ in alns):
                        i += 1
                        continue
                    tmp = MapResult()
                    resolve_pair(self._cands(alns, tnames), None, tmp, read_idx=i)
                    if tmp.per_read:
                        per.update(tmp.per_read)
                res.per_read = per
        res.n_reads = n
        res.wall_seconds = wall
        return res


def revcomp_str(s: str) -> str:
    from .catalog import revcomp
    return revcomp(s)


# ---------------------------------------------------------------------------
# Pooled multi-sample batching (one index load per streaming iteration)
# ---------------------------------------------------------------------------

_SAFE_ID = _re.compile(r"[^A-Za-z0-9_.-]")


def map_pooled(kernel: "Kernel", batches: list[tuple[str, list[Read], list[Read] | None, OnlineErrorCalibrator | None]]) -> dict[str, MapResult]:
    """Map several samples' reads in ONE kernel invocation and split the result back per sample.

    ``batches`` = [(sample_id, r1, r2_or_None, calibrator), ...]; all must share a layout.  Reads are concatenated, named
    ``@<sample_id>|r<i>`` (i = position in the pooled batch) and every resolved read is counted into its own sample's
    MapResult, so per-sample unique counts and equivalence classes equal those of separate calls for kernels whose
    per-read placement does not depend on the other reads in the batch.  Kernels without ``supports_pooling`` (mappy:
    in-process, no reload cost; salmon / piscem: no per-read identity in their multi-threaded output) are mapped per
    sample, exactly as before."""
    out: dict[str, MapResult] = {}
    batches = [b for b in batches if b[1]]
    if not batches:
        return out
    paired = batches[0][2] is not None
    if any((b[2] is not None) != paired for b in batches):
        raise ValueError("map_pooled needs batches of one layout")
    if not kernel.supports_pooling or len(batches) == 1:
        if len(batches) >= 2 and hasattr(kernel, "clone_with_threads") and kernel.threads >= 4:
            n_w = min(len(batches), max(2, min(4, kernel.threads // 2)))
            sub_t = max(1, kernel.threads // n_w)

            def _map_one(item):
                sid, r1, r2, cal = item
                return sid, kernel.clone_with_threads(sub_t).map_batch(r1, r2, cal)

            with ThreadPoolExecutor(max_workers=n_w) as ex:
                for sid, res in ex.map(_map_one, batches):
                    out[sid] = res
            return out
        for sid, r1, r2, cal in batches:
            out[sid] = kernel.map_batch(r1, r2, cal)
        return out
    lengths = [len(r1) if r2 is None else min(len(r1), len(r2)) for _sid, r1, r2, _cal in batches]
    tot = sum(lengths)
    safe = {sid: _SAFE_ID.sub("_", sid) for sid, _r1, _r2, _cal in batches}
    for (sid, _r1, _r2, _cal), n in zip(batches, lengths):
        out[sid] = MapResult(n_reads=n)

    import bisect
    import gc

    chunk_reads = max(1, int(os.environ.get("PANEXPR_POOLED_CHUNK_READS", "1000000")))

    def _run_chunk(sub_a1: list[Read], sub_a2: list[Read] | None, sub_bounds: list[tuple[int, int, str]]) -> None:
        if not sub_a1:
            return
        sub_starts = [b[0] for b in sub_bounds]

        def owner(i: int) -> str:
            return sub_bounds[bisect.bisect_right(sub_starts, i) - 1][2]

        tok_r = _ROUTE.set(lambda i: out[owner(i)])
        tok_n = _NAME_PREFIX.set(lambda i: safe[owner(i)])
        try:
            kernel.map_batch(sub_a1, sub_a2, None, trace=True)
        finally:
            _ROUTE.reset(tok_r)
            _NAME_PREFIX.reset(tok_n)

    t0 = time.time()
    sub_a1: list[Read] = []
    sub_a2: list[Read] | None = [] if paired else None
    sub_bounds: list[tuple[int, int, str]] = []
    for (sid, r1, r2, _cal), n in zip(batches, lengths):
        pos = 0
        while pos < n:
            take = min(n - pos, chunk_reads - len(sub_a1))
            s0 = len(sub_a1)
            sub_a1.extend(r1[pos:pos + take])
            if sub_a2 is not None and r2 is not None:
                sub_a2.extend(r2[pos:pos + take])
            sub_bounds.append((s0, s0 + take, sid))
            pos += take
            if len(sub_a1) >= chunk_reads:
                _run_chunk(sub_a1, sub_a2, sub_bounds)
                sub_a1 = []
                sub_a2 = [] if paired else None
                sub_bounds = []
                gc.collect()
    if sub_a1:
        _run_chunk(sub_a1, sub_a2, sub_bounds)
        del sub_a1, sub_a2, sub_bounds
    wall = time.time() - t0
    denom = max(1, tot)
    for (sid, _r1, _r2, _cal), n in zip(batches, lengths):
        out[sid].wall_seconds = wall * n / denom
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def default_kernels(threads: int = 4, tmp_dir: str | Path | None = None, include_star: bool = False) -> list[Kernel]:
    ks: list[Kernel] = [
        MappyKernel(threads),
        SamKernel("bwa_mem2_spa", "BWA-MEM2 (-S -P -a, independent mates)", "bwa-mem2", 1.15,
                  ["mem", "-t", "{t}", "-p", "-S", "-P", "-a", "-T", "20", "{idx}", "-"],
                  ["mem", "-t", "{t}", "-a", "-T", "20", "{idx}", "-"], "bwa", threads, symmetric=True),
        SamKernel("bowtie2_local_k16", "Bowtie2 (--very-sensitive-local -k 16)", "bowtie2", 0.72,
                  ["-p", "{t}", "--very-sensitive-local", "--no-unal", "-k", "16", "--no-mixed", "--no-discordant", "-x", "{idx}", "--interleaved", "-"],
                  ["-p", "{t}", "--very-sensitive-local", "--no-unal", "-k", "16", "-x", "{idx}", "-U", "-"], "bt2", threads),
        SamKernel("bowtie2_global_k16", "Bowtie2 (--very-sensitive end-to-end -k 16)", "bowtie2", 0.72,
                  ["-p", "{t}", "--very-sensitive", "--no-unal", "-k", "16", "--no-mixed", "--no-discordant", "-x", "{idx}", "--interleaved", "-"],
                  ["-p", "{t}", "--very-sensitive", "--no-unal", "-k", "16", "-x", "{idx}", "-U", "-"], "bt2", threads),
        SamKernel("strobealign_N16", "strobealign (-N 16, paired rescue)", "strobealign", 0.58,
                  ["-t", "{t}", "--interleaved", "-N", "16", "--st", "0.80", "{idx}", "-"],
                  ["-t", "{t}", "-N", "16", "--st", "0.80", "{idx}", "-"], "fasta", threads),
        SamKernel("bwa_mem2_naive_pa", "BWA-MEM2 (-p -a, proper-pair collapse)", "bwa-mem2", 1.15,
                  ["mem", "-t", "{t}", "-p", "-a", "-T", "20", "{idx}", "-"],
                  ["mem", "-t", "{t}", "-a", "-T", "20", "{idx}", "-"], "bwa", threads),
        SalmonKernel(threads, tmp_dir=tmp_dir),
        KallistoKernel(threads, tmp_dir=tmp_dir),
    ]
    # promoted by the open-source mapper survey (14_open_source_mapper_survey.md): gate-passing contenders
    ks += [FileSamKernel("bwa_spa", FILE_SAM_SPECS["bwa_spa"], threads, tmp_dir), PiscemKernel(threads, tmp_dir=tmp_dir),
           ThemistoKernel(threads, tmp_dir=tmp_dir)]
    if include_star:
        ks.append(StarKernel(threads, tmp_dir=tmp_dir))
        ks.append(GenomeProjectKernel("star", threads, tmp_dir=tmp_dir))
    return ks


PRODUCTION_KERNELS = ("mappy", "bwa_mem2_spa", "bowtie2_local_k16", "bowtie2_global_k16", "strobealign_N16", "bwa_mem2_naive_pa", "salmon",
                      "kallisto", "bwa_spa", "piscem", "themisto", "star", "star_genome_m16")


def survey_kernels(threads: int = 4, tmp_dir: str | Path | None = None) -> list[Kernel]:
    """Kernels added by the open-source mapper survey (opt-in; default_kernels keeps the original 8/9)."""
    promoted = {"bwa_spa"}
    ks: list[Kernel] = [FileSamKernel(n, spec, threads, tmp_dir) for n, spec in FILE_SAM_SPECS.items() if n not in promoted]
    ks.append(GenomeProjectKernel("hisat2", threads, tmp_dir=tmp_dir))
    return ks


def all_kernels(threads: int = 4, tmp_dir: str | Path | None = None, include_star: bool = True) -> list[Kernel]:
    """Production pool (default_kernels) + every other survey kernel, each name once."""
    return default_kernels(threads, tmp_dir, include_star) + survey_kernels(threads, tmp_dir)


def kernel_by_name(name: str, threads: int = 4, tmp_dir: str | Path | None = None) -> Kernel:
    for k in all_kernels(threads, tmp_dir, include_star=True):
        if k.name == name:
            return k
    raise KeyError(name)
