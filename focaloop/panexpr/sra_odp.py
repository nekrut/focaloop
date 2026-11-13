"""Tier-1 raw-read source: NCBI SRA on AWS Open Data (s3://sra-pub-run-odp) through ``xsra``, ahead of ENA HTTPS 206.

Tier resolution per run (``resolve_tier``):
  1. ``sra_odp``  - the run object answers an HTTPS HEAD at https://sra-pub-run-odp.s3.amazonaws.com/sra/<ACC>/<ACC>
                    and ``xsra`` is installed.  Reads are pulled as spot blocks with ``xsra dump -Q -P https``.
  2. ``http_range`` - ENA ``.fastq.gz`` byte prefixes with HTTP 206 Range (the existing streamer path).

Spot-block plan and holdout (``SpotBlockPlan``):
  * the run's spots are cut into fixed-size blocks; a keyed hash of (run, block) puts 10 % of the blocks in a HOLDOUT
    partition and the rest in the DISCOVERY partition (``SpotBlockPlan.discovery`` also lists a stratified visiting order
    for sources with random access);
  * xsra 0.2.29 exposes a spot limit (-l) but no start offset, so ``XsraRunStream`` reads one persistent
    ``xsra dump -Q -P https`` pipe in file order and assigns every spot to its block on the fly;
  * the holdout partition is write-only: holdout spots are discarded inside ``XsraRunStream`` (only counted), and
    ``HoldoutLedger`` records holdout block coordinates for a later, separate validation step without any method that
    returns reads.  A run flagged LOW_MAP_RATE that asks for more data still receives discovery spots only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request

from .manifest import Run

ODP_URL = "https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
HOLDOUT_FRACTION = 0.10
DEFAULT_BLOCK_SPOTS = 50_000
LOW_MAP_RATE = 0.30          # a run whose mapped fraction falls below this is flagged LOW_MAP_RATE


def xsra_path() -> str | None:
    b = os.environ.get("PANEXPR_BIN")
    if b and (Path(b) / "xsra").exists():
        return str(Path(b) / "xsra")
    return shutil.which("xsra")


_RUNNABLE: dict[str, bool] = {}


def xsra_runnable() -> bool:
    """xsra exists AND executes on this node (the conda-forge Rust toolchain emits x86-64-v3 binaries: AVX2 nodes only)."""
    exe = xsra_path()
    if not exe:
        return False
    if exe not in _RUNNABLE:
        try:
            _RUNNABLE[exe] = subprocess.run([exe, "--version"], capture_output=True, timeout=30).returncode == 0
        except Exception:
            _RUNNABLE[exe] = False
    return _RUNNABLE[exe]


def odp_available(accession: str, timeout: float = 10.0) -> bool:
    """HTTPS HEAD on the public bucket object (IPv4-agnostic, no credentials)."""
    req = urllib.request.Request(ODP_URL.format(acc=accession), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_tier(run: Run, check=odp_available) -> str:
    """'sra_odp' (Tier 1) when xsra is installed and the run is mirrored on AWS Open Data, else 'http_range' (ENA 206)."""
    if run.run_id[:3] in ("SRR", "ERR", "DRR") and xsra_runnable() and check(run.run_id):
        return "sra_odp"
    return "http_range"


# ---------------------------------------------------------------------------
# Spot-block plan with a write-only 10 % holdout
# ---------------------------------------------------------------------------


def _h(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


@dataclass
class SpotBlockPlan:
    run_id: str
    n_spots: int
    block_spots: int = DEFAULT_BLOCK_SPOTS
    holdout_fraction: float = HOLDOUT_FRACTION
    salt: str = "panexpr-holdout-v1"
    holdout: list[int] = field(init=False)
    discovery: list[int] = field(init=False)

    def __post_init__(self) -> None:
        n_blocks = max(1, -(-self.n_spots // self.block_spots))
        blocks = list(range(n_blocks))
        if n_blocks == 1:
            self.holdout, disc = [], blocks
        else:
            # exactly round(10 %) of the blocks, chosen by keyed hash (deterministic, unpredictable by position)
            k = max(1, round(self.holdout_fraction * n_blocks))
            ranked = sorted(blocks, key=lambda b: _h(f"{self.salt}|{self.run_id}|{b}"))
            self.holdout = sorted(ranked[:k])
            hs = set(self.holdout)
            disc = [b for b in blocks if b not in hs]
        self.discovery = self._stratified(disc)

    @staticmethod
    def _stratified(blocks: list[int]) -> list[int]:
        """Van der Corput order over the block list: 0, 1/2, 1/4, 3/4, ... so every prefix spans the whole run."""
        n = len(blocks)
        if n <= 2:
            return list(blocks)
        bits = max(1, (n - 1).bit_length())
        order, seen = [], set()
        for i in range(1 << bits):
            r = int(f"{i:0{bits}b}"[::-1], 2)
            j = r * n >> bits
            if j < n and j not in seen:
                seen.add(j)
                order.append(blocks[j])
        order += [b for b in blocks if b not in set(order)]
        return order

    def spot_range(self, block: int) -> tuple[int, int]:
        """1-based inclusive spot ids of a block."""
        a = block * self.block_spots + 1
        return a, min(self.n_spots, a + self.block_spots - 1)

    def is_holdout(self, block: int) -> bool:
        return block in set(self.holdout)


class HoldoutLedger:
    """Write-only record of holdout blocks.  It can append and list coordinates; it has no API that returns reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, plan: SpotBlockPlan) -> None:
        rows = [{"run_id": plan.run_id, "block": b, "spots": list(plan.spot_range(b)), "salt": plan.salt} for b in plan.holdout]
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# xsra-backed run stream (same interface the Focaloop uses for RunStream)
# ---------------------------------------------------------------------------


def xsra_dump_command(accession: str, n_spots: int | None = None, threads: int = 4) -> list[str]:
    """``xsra dump`` from AWS Open Data over HTTPS: -Q (full quality, mandatory), -P https, -t (skip technical reads),
    interleaved FASTQ on stdout.  xsra 0.2.29 has a spot LIMIT (-l) but no start offset, so a run is read in file order
    by one persistent process and spot blocks are assigned on the fly."""
    exe = xsra_path() or "xsra"
    cmd = [exe, "dump", accession, "-Q", "-P", "https", "-t", "-T", str(threads), "-f", "q"]
    if n_spots:
        cmd += ["-l", str(n_spots)]
    return cmd


def _spot_and_segment(name: str) -> tuple[int, int]:
    """xsra read names are <accession>.<spot>.<segment>."""
    parts = name.split()[0].rsplit(".", 2)
    return int(parts[1]), int(parts[2])


def split_mates(records: list[tuple[str, str, str]], paired: bool):
    """Pair consecutive records of one spot (segments 0 and 1); SE keeps every record."""
    if not paired:
        return [(s, q) for _, s, q in records], None
    r1, r2 = [], []
    i = 0
    while i + 1 < len(records):
        a, b = records[i], records[i + 1]
        if _spot_and_segment(a[0])[0] == _spot_and_segment(b[0])[0]:
            r1.append((a[1], a[2])); r2.append((b[1], b[2]))
            i += 2
        else:
            i += 1                       # orphan segment: skip
    return r1, r2


class XsraRunStream:
    """Discovery-only spot stream over one SRA run on AWS Open Data.

    One ``xsra dump -Q -P https`` process per run is read incrementally.  Every record's spot id decides its block; spots in
    holdout blocks are dropped inside this class (only their count is kept), so no holdout read ever reaches a kernel - also
    not for LOW_MAP_RATE runs, which simply draw more discovery spots.  ``advance_bytes`` converts a byte budget into
    discovery spots with the run's archive bytes per spot, so the loop's byte caps keep their meaning."""

    _PROC_COND = threading.Condition()
    _ACTIVE_STREAMS: list["XsraRunStream"] = []

    def __init__(self, run: Run, plan: SpotBlockPlan | None = None, ledger: HoldoutLedger | None = None, threads: int | None = None,
                 line_source=None, cache_dir: str | Path | None = None, max_retries: int = 3,
                 retry_backoff_s: float = 0.2, fallback_factory=None) -> None:
        self.run = run
        self.plan = plan or SpotBlockPlan(run.run_id, max(1, run.read_count))
        self._holdout = set(self.plan.holdout)
        self.bytes_per_spot = run.total_bytes / max(1, run.read_count)
        self.threads = int(threads) if threads is not None else int(os.environ.get("PANEXPR_XSRA_THREADS", "4"))
        self.max_concurrent_procs = max(1, int(os.environ.get("PANEXPR_MAX_XSRA_PROCS", "8")))
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.retries = 0
        self.used_fallback = False
        self.consumed_spots = 0          # discovery spots served to the loop
        self.dropped_holdout_spots = 0   # spots read from the pipe and discarded (holdout partition)
        self.network_bytes = 0
        self.low_map_rate = False
        self.served_spot_ids: list[int] = []
        self._custom_line_source = line_source is not None
        self._lines = line_source        # iterator of FASTQ lines (tests); default: xsra subprocess stdout
        self._proc = None
        self._in_advance = False
        self._done = False
        self._last_dropped = -1
        self._last_raw_spot = 0
        self._seen_bm = bytearray((max(1024, self.plan.n_spots) >> 3) + 64)
        self._seen_overflow: set[int] = set()
        self._carry: list = []
        self._fallback_stream = None
        self._fallback_factory = fallback_factory
        if ledger is not None:
            ledger.record(self.plan)

    # -- RunStream interface
    @property
    def layout(self) -> str:
        return self.run.layout

    @property
    def total_bytes(self) -> int:
        if self._fallback_stream is not None:
            return self._fallback_stream.total_bytes
        n_hold = sum(self.plan.spot_range(b)[1] - self.plan.spot_range(b)[0] + 1 for b in self.plan.holdout)
        return int(self.bytes_per_spot * (self.plan.n_spots - n_hold))

    @property
    def consumed_bytes(self) -> int:
        if self._fallback_stream is not None:
            return self._fallback_stream.consumed_bytes
        return int(self.consumed_spots * self.bytes_per_spot)

    @property
    def exhausted(self) -> bool:
        if self._fallback_stream is not None:
            return self._fallback_stream.exhausted
        return self._done and not self._carry

    def _spawn_proc(self):
        return subprocess.Popen(
            xsra_dump_command(self.run.run_id, None, self.threads),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1 << 20,
        )

    def _iter_lines(self):
        if self._lines is None:
            cond = XsraRunStream._PROC_COND
            active = XsraRunStream._ACTIVE_STREAMS
            with cond:
                while len(active) >= self.max_concurrent_procs:
                    vic = next((s for s in active if s is not self and not s._in_advance), None)
                    if vic is not None:
                        active.remove(vic)
                        vic._lines = None
                        # Kill outside lock? Wait briefly while holding/killing so OS PIDs stay <= max_concurrent_procs.
                        vic._terminate_proc_only()
                        break
                    cond.wait(timeout=0.5)
                self._proc = self._spawn_proc()
                active.append(self)
                self._lines = iter(self._proc.stdout)
        return self._lines

    def _terminate_proc_only(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def _reap_proc(self) -> int | None:
        with XsraRunStream._PROC_COND:
            if self in XsraRunStream._ACTIVE_STREAMS:
                XsraRunStream._ACTIVE_STREAMS.remove(self)
                XsraRunStream._PROC_COND.notify_all()
        if self._proc is None:
            return None
        proc = self._proc
        self._proc = None
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            return proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                return proc.wait(timeout=2)
            except Exception:
                return -1

    def _activate_fallback_if_available(self) -> bool:
        if self._last_raw_spot > 0:
            return False
        if self._fallback_factory is not None:
            self._fallback_stream = self._fallback_factory(self.run)
            self.used_fallback = self._fallback_stream is not None
            return self.used_fallback
        if not self.run.files:
            return False
        from .streamer import open_run_stream
        if all(f.local_path for f in self.run.files):
            self._fallback_stream = open_run_stream(self.run, "local_bytes", self.cache_dir)
            self.used_fallback = True
            return True
        if all(f.url for f in self.run.files):
            self._fallback_stream = open_run_stream(self.run, "http_range", self.cache_dir)
            self.used_fallback = True
            return True
        return False

    def _next_record(self):
        while True:
            it = self._iter_lines()
            try:
                h = next(it)
                while not h.startswith("@"):
                    h = next(it)
                sq = next(it)
                if sq.startswith("+"):
                    # Guard against xsra 3-line zero-length segment quirk (@hdr / + / qual)
                    next(it)
                    continue
                plus = next(it)
                if not plus.startswith("+"):
                    continue
                q = next(it)
                sq_s = sq.rstrip("\n")
                q_s = q.rstrip("\n")
                if not sq_s or len(sq_s) != len(q_s):
                    continue
                self.network_bytes += len(h) + len(sq_s) + len(q_s) + 4
                return h[1:].rstrip("\n"), sq_s, q_s
            except StopIteration:
                if self._custom_line_source:
                    self._done = True
                    return None
                rc = self._reap_proc()
                self._lines = None
                premature = (rc not in (0, None)) or (self._last_raw_spot == 0 and self.plan.n_spots > 0)
                if premature and self.retries < self.max_retries:
                    self.retries += 1
                    if self.retry_backoff_s > 0:
                        time.sleep(min(2.0, self.retry_backoff_s * (2 ** (self.retries - 1))))
                    continue
                if premature and self._activate_fallback_if_available():
                    return None
                self._done = True
                return None

    def advance_bytes(self, n_bytes: int):
        if self._fallback_stream is not None:
            return self._fallback_stream.advance_bytes(n_bytes)
        self._in_advance = True
        try:
            return self._advance_bytes_inner(n_bytes)
        finally:
            self._in_advance = False
            with XsraRunStream._PROC_COND:
                # Move to MRU tail so active pools evict least-recently-used idle streams first.
                if self in XsraRunStream._ACTIVE_STREAMS:
                    XsraRunStream._ACTIVE_STREAMS.remove(self)
                    XsraRunStream._ACTIVE_STREAMS.append(self)
                XsraRunStream._PROC_COND.notify_all()

    def _advance_bytes_inner(self, n_bytes: int):
        want = max(1, int(n_bytes / max(1e-9, self.bytes_per_spot)))
        paired = self.run.layout == "PE"
        r1, r2 = [], ([] if paired else None)
        got = 0
        pending: list = self._carry       # first records of a spot read at the end of the previous call
        self._carry = []
        while got < want and not self._done:
            rec = self._next_record()
            if rec is None:
                if self._fallback_stream is not None:
                    return self._fallback_stream.advance_bytes(n_bytes)
                break
            spot, seg = _spot_and_segment(rec[0])
            if self._is_spot_done(spot):
                continue                                    # already processed prior to an xsra pipe retry
            block = (spot - 1) // self.plan.block_spots
            if block in self._holdout:
                if spot != self._last_dropped:
                    self.dropped_holdout_spots += 1
                    self._last_dropped = spot
                self._last_raw_spot = max(self._last_raw_spot, spot)
                self._mark_spot_done(spot)
                continue                                    # holdout: never leaves this class
            if pending:
                p_spot = _spot_and_segment(pending[0][0])[0]
                if p_spot != spot:
                    self._emit(pending, r1, r2, paired)
                    got += 1
                    pending = []
                elif any(_spot_and_segment(x[0])[1] == seg for x in pending):
                    continue                                # duplicate mate after mid-spot pipe retry
            pending.append(rec)
        if pending:
            if self._done and got < want:
                self._emit(pending, r1, r2, paired)       # last spot of the run
                got += 1
            else:
                self._carry = pending                     # budget met: keep the started spot for the next call
        self.consumed_spots += got
        return r1, r2

    def _is_spot_done(self, spot: int) -> bool:
        idx = spot >> 3
        if idx < len(self._seen_bm):
            return bool(self._seen_bm[idx] & (1 << (spot & 7)))
        return spot in self._seen_overflow

    def _mark_spot_done(self, spot: int) -> None:
        idx = spot >> 3
        if idx < len(self._seen_bm):
            self._seen_bm[idx] |= 1 << (spot & 7)
        else:
            self._seen_overflow.add(spot)

    def _emit(self, recs, r1, r2, paired):
        spot = _spot_and_segment(recs[0][0])[0]
        self.served_spot_ids.append(spot)
        self._last_raw_spot = max(self._last_raw_spot, spot)
        self._mark_spot_done(spot)
        if paired:
            if len(recs) >= 2:
                r1.append((recs[0][1], recs[0][2])); r2.append((recs[1][1], recs[1][2]))
        else:
            for _n, sq, q in recs:
                r1.append((sq, q))

    def advance_fraction(self, frac: float, min_bytes: int = 262_144):
        return self.advance_bytes(max(min_bytes, int(self.total_bytes * frac)))

    def flag_low_map_rate(self, mapped_fraction: float) -> bool:
        """Mark the run LOW_MAP_RATE; any extra data it is given still comes from discovery spots only."""
        self.low_map_rate = mapped_fraction < LOW_MAP_RATE
        return self.low_map_rate

    def close(self) -> None:
        with XsraRunStream._PROC_COND:
            if self in XsraRunStream._ACTIVE_STREAMS:
                XsraRunStream._ACTIVE_STREAMS.remove(self)
                XsraRunStream._PROC_COND.notify_all()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
