"""Byte-budget FASTQ streaming: local gzip prefixes/chunks or live HTTP 206 ranges.

Two source modes share one interface:

* ``local_bytes``  - reads exact compressed byte prefixes from staged ``.fastq.gz``
  files on node scratch (deterministic, used for matched-byte ablations).
* ``http_range``   - fetches the same byte prefixes live from ENA with
  ``Range: bytes=a-b`` requests (HTTP 206), caching the prefix on disk so no
  byte is fetched twice.  Wall-clock here measures WAN cold start.

The decoder is a stateful zlib stream that also handles multi-member (BGZF)
files, so it always yields complete FASTQ records for the bytes consumed so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import socket
import subprocess
import time
import zlib

from .manifest import Run


class ByteSource:
    total_bytes: int

    def read_range(self, start: int, end_exclusive: int) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class LocalFileSource(ByteSource):
    def __init__(self, path: str | Path, total_bytes: int | None = None) -> None:
        self.path = Path(path)
        self.total_bytes = total_bytes if total_bytes is not None else self.path.stat().st_size
        self.bytes_read = 0

    def read_range(self, start: int, end_exclusive: int) -> bytes:
        end_exclusive = min(end_exclusive, self.total_bytes)
        if end_exclusive <= start:
            return b""
        with open(self.path, "rb") as fh:
            fh.seek(start)
            data = fh.read(end_exclusive - start)
        self.bytes_read += len(data)
        return data


class HttpRangeSource(ByteSource):
    """HTTP 206 byte-range source with an append-only on-disk prefix cache."""

    def __init__(self, url: str, total_bytes: int, cache_file: str | Path | None = None, timeout: float = 90.0) -> None:
        self.url = url
        self.total_bytes = total_bytes
        self.cache_file = Path(cache_file) if cache_file else None
        self.timeout = timeout
        self.network_bytes = 0
        self.requests_made = 0
        self.network_seconds = 0.0
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            from urllib3.util import connection as _conn

            _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
            self._session = requests.Session()
            self._session.headers["User-Agent"] = "panexpr/0.1 (HTTP 206 range streamer)"
        return self._session

    def _fetch(self, start: int, end_inclusive: int) -> bytes:
        t0 = time.time()
        last: Exception | None = None
        for attempt in range(8):
            try:
                r = self._get_session().get(self.url, headers={"Range": f"bytes={start}-{end_inclusive}"}, timeout=self.timeout)
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for range {start}-{end_inclusive}")
                data = r.content
                if r.status_code == 200:  # server ignored the range
                    data = data[start:end_inclusive + 1]
                self.requests_made += 1
                self.network_bytes += len(data)
                self.network_seconds += time.time() - t0
                return data
            except Exception as exc:  # fall back to curl -4 which proved robust on this cluster
                last = exc
                try:
                    out = subprocess.run(
                        ["curl", "-4", "-sS", "--fail", "--max-time", str(int(self.timeout)), "-r", f"{start}-{end_inclusive}", self.url],
                        check=True, capture_output=True, timeout=self.timeout + 30,
                    ).stdout
                    self.requests_made += 1
                    self.network_bytes += len(out)
                    self.network_seconds += time.time() - t0
                    return out
                except Exception as exc2:
                    last = exc2
                    time.sleep(min(60.0, 5.0 * (attempt + 1)))
        raise RuntimeError(f"range fetch failed for {self.url}: {last}")

    def read_range(self, start: int, end_exclusive: int) -> bytes:
        end_exclusive = min(end_exclusive, self.total_bytes)
        if end_exclusive <= start:
            return b""
        if self.cache_file is not None:
            have = self.cache_file.stat().st_size if self.cache_file.exists() else 0
            if end_exclusive > have:
                if start > have:
                    # Fill the gap so the cache stays a contiguous prefix
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_file, "ab") as fh:
                        fh.write(self._fetch(have, start - 1))
                    have = start
                chunk = self._fetch(have, end_exclusive - 1)
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "ab") as fh:
                    fh.write(chunk)
            with open(self.cache_file, "rb") as fh:
                fh.seek(start)
                return fh.read(end_exclusive - start)
        return self._fetch(start, end_exclusive - 1)


def _norm_qname(header: bytes) -> str:
    """Normalize a FASTQ ``@header`` line to its mate-stripped query identifier."""
    h = header[1:].strip() if header.startswith(b"@") else header.strip()
    if not h:
        return ""
    tok = h.split(None, 1)[0]
    if len(tok) > 2 and tok[-2:] in (b"/1", b"/2", b".1", b".2") and not tok[:-2].isdigit():
        # Note: SRA headers like @SRR123.45.1 vs @SRR123.45/1: strip /1,/2 unconditionally;
        # strip .1/.2 only when there are >=2 dots (@ACC.SPOT.READ) so @ACC.SPOT stays intact.
        if tok[-2:-1] == b"/" or tok.count(b".") >= 2:
            tok = tok[:-2]
    return tok.decode("ascii", "replace")


class GzipStreamDecoder:
    """Stateful gzip decoder (multi-member aware) that yields complete FASTQ records."""

    def __init__(self) -> None:
        self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._rem = b""
        self.records_emitted = 0
        self.malformed_skipped = 0

    def feed_named(self, raw: bytes) -> list[tuple[str, str, str]]:
        out_txt = bytearray()
        data = raw
        while data:
            try:
                out_txt += self._d.decompress(data)
            except zlib.error:
                break
            if self._d.eof:
                data = self._d.unused_data
                self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                if not data:
                    break
            else:
                data = b""
        buf = self._rem + bytes(out_txt)
        lines = buf.split(b"\n")
        # Hold the trailing incomplete line (and up to 3 complete lines not yet forming a 4-line record)
        limit = len(lines) - 1
        recs: list[tuple[str, str, str]] = []
        i = 0
        while i + 3 < limit:
            hdr = lines[i]
            plus = lines[i + 2]
            if not hdr.startswith(b"@") or not plus.startswith(b"+"):
                self.malformed_skipped += 1
                i += 1
                continue
            seq = lines[i + 1].strip()
            qual = lines[i + 3].strip()
            i += 4
            if seq and len(seq) == len(qual):
                recs.append((_norm_qname(hdr), seq.decode("ascii", "replace"), qual.decode("ascii", "replace")))
            else:
                self.malformed_skipped += 1
        self._rem = b"\n".join(lines[i:])
        self.records_emitted += len(recs)
        return recs

    def feed(self, raw: bytes) -> list[tuple[str, str]]:
        return [(s, q) for _, s, q in self.feed_named(raw)]


class FastqPrefixStream:
    """Sequential compressed-byte cursor over one FASTQ file (any ByteSource)."""

    def __init__(self, source: ByteSource) -> None:
        self.source = source
        self.decoder = GzipStreamDecoder()
        self.consumed_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self.source.total_bytes

    @property
    def exhausted(self) -> bool:
        return self.consumed_bytes >= self.source.total_bytes

    def advance_to_named(self, target_bytes: int) -> list[tuple[str, str, str]]:
        target_bytes = min(int(target_bytes), self.source.total_bytes)
        if target_bytes <= self.consumed_bytes:
            return []
        raw = self.source.read_range(self.consumed_bytes, target_bytes)
        self.consumed_bytes += len(raw)
        return self.decoder.feed_named(raw)

    def advance_by_named(self, n_bytes: int) -> list[tuple[str, str, str]]:
        return self.advance_to_named(self.consumed_bytes + int(n_bytes))

    def advance_to(self, target_bytes: int) -> list[tuple[str, str]]:
        return [(s, q) for _, s, q in self.advance_to_named(target_bytes)]

    def advance_by(self, n_bytes: int) -> list[tuple[str, str]]:
        return self.advance_to(self.consumed_bytes + int(n_bytes))


@dataclass
class RunStream:
    """PE (two cursors) or SE (one cursor) stream for one run."""

    run: Run
    streams: list[FastqPrefixStream]
    reads_emitted: int = 0
    skipped_unpaired_mates: int = 0

    @property
    def layout(self) -> str:
        return self.run.layout

    @property
    def consumed_bytes(self) -> int:
        return sum(s.consumed_bytes for s in self.streams)

    @property
    def total_bytes(self) -> int:
        return sum(s.total_bytes for s in self.streams)

    @property
    def exhausted(self) -> bool:
        return all(s.exhausted for s in self.streams)

    @property
    def network_bytes(self) -> int:
        return sum(getattr(s.source, "network_bytes", 0) for s in self.streams)

    def advance_fraction(self, frac: float, min_bytes: int = 262_144) -> tuple[list[tuple[str, str]], list[tuple[str, str]] | None]:
        """Advance every mate cursor by ``frac`` of its own file size (at least min_bytes)."""
        if len(self.streams) == 1:
            b = self.streams[0].advance_by(max(min_bytes, int(self.streams[0].total_bytes * frac)))
            self.reads_emitted += len(b)
            return b, None
        batches = [s.advance_by_named(max(min_bytes, int(s.total_bytes * frac))) for s in self.streams]
        return self._pair_named(batches)

    def advance_bytes(self, n_bytes_total: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]] | None]:
        """Advance so that this run consumes ``n_bytes_total`` more compressed bytes (split across mates by size)."""
        if len(self.streams) == 1:
            b = self.streams[0].advance_by(max(1, int(n_bytes_total)))
            self.reads_emitted += len(b)
            return b, None
        tot = max(1, self.total_bytes)
        batches = [s.advance_by_named(max(1, int(n_bytes_total * s.total_bytes / tot))) for s in self.streams]
        return self._pair_named(batches)

    _carry: list[list[tuple[str, str]]] = field(default_factory=lambda: [[], []])
    _carry_named: list[list[tuple[str, str, str]]] = field(default_factory=lambda: [[], []])

    def _pair(self, batches: list[list[tuple[str, str]]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]] | None]:
        if len(batches) == 1:
            self.reads_emitted += len(batches[0])
            return batches[0], None
        r1 = self._carry[0] + batches[0]
        r2 = self._carry[1] + batches[1]
        n = min(len(r1), len(r2))
        self._carry = [r1[n:], r2[n:]]
        self.reads_emitted += n
        return r1[:n], r2[:n]

    def _pair_named(self, batches: list[list[tuple[str, str, str]]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]] | None]:
        r1 = self._carry_named[0] + batches[0]
        r2 = self._carry_named[1] + batches[1]
        n = min(len(r1), len(r2))
        if n == 0:
            if self.exhausted:
                self.skipped_unpaired_mates += len(r1) + len(r2)
                self._carry_named = [[], []]
            else:
                self._carry_named = [r1, r2]
            return [], []
        # Fast path: QNAMEs already aligned at every index (standard uncorrupted stream)
        if all(r1[i][0] == r2[i][0] for i in range(n)):
            out1 = [(s, q) for _, s, q in r1[:n]]
            out2 = [(s, q) for _, s, q in r2[:n]]
            if self.exhausted:
                self.skipped_unpaired_mates += (len(r1) - n) + (len(r2) - n)
                self._carry_named = [[], []]
            else:
                self._carry_named = [r1[n:], r2[n:]]
            self.reads_emitted += n
            return out1, out2
        # Resync path: one mate stream skipped a malformed record; pair strictly by matching QNAME in order
        from collections import deque
        pos2: dict[str, deque[int]] = {}
        for j, (qn, _, _) in enumerate(r2):
            pos2.setdefault(qn, deque()).append(j)
        out1, out2 = [], []
        last_i1 = -1
        last_i2 = -1
        matched = 0
        for i1, (qn, s1, q1) in enumerate(r1):
            q_js = pos2.get(qn)
            if not q_js:
                continue
            while q_js and q_js[0] <= last_i2:
                q_js.popleft()
            if not q_js:
                continue
            i2 = q_js.popleft()
            self.skipped_unpaired_mates += (i1 - last_i1 - 1) + (i2 - last_i2 - 1)
            last_i1, last_i2 = i1, i2
            _, s2, q2 = r2[i2]
            out1.append((s1, q1))
            out2.append((s2, q2))
            matched += 1
        rem1 = r1[last_i1 + 1:]
        rem2 = r2[last_i2 + 1:]
        if self.exhausted:
            self.skipped_unpaired_mates += len(rem1) + len(rem2)
            self._carry_named = [[], []]
        else:
            self._carry_named = [rem1, rem2]
        self.reads_emitted += matched
        return out1, out2


def open_run_stream(run: Run, source_mode: str, cache_dir: str | Path | None = None):
    """Create a run stream.

    ``local_bytes`` / ``http_range``: compressed .fastq.gz byte prefixes (staged files / ENA HTTPS 206).
    ``sra_odp``: Tier-1 AWS Open Data spot blocks via ``xsra -Q -P https`` (discovery blocks only, 10 % holdout).
    ``auto``: Tier-1 when the run is on sra-pub-run-odp and xsra is installed, else ENA HTTPS 206."""
    if source_mode in ("sra_odp", "auto"):
        from .sra_odp import HoldoutLedger, XsraRunStream, resolve_tier
        if source_mode == "sra_odp" or resolve_tier(run) == "sra_odp":
            ledger = HoldoutLedger(Path(cache_dir) / "holdout_blocks.jsonl") if cache_dir else None
            return XsraRunStream(run, ledger=ledger, cache_dir=cache_dir)
        source_mode = "http_range"
    streams: list[FastqPrefixStream] = []
    for f in run.files:
        if source_mode == "local_bytes":
            if not f.local_path:
                raise ValueError(f"run {run.run_id}: local_bytes mode needs local_path for {f.basename}")
            src: ByteSource = LocalFileSource(f.local_path, f.bytes if f.bytes else None)
        elif source_mode == "http_range":
            cache = (Path(cache_dir) / f"{f.basename}.part") if cache_dir else None
            src = HttpRangeSource(f.url, f.bytes, cache)
        else:
            raise ValueError(f"unknown source_mode {source_mode!r}")
        streams.append(FastqPrefixStream(src))
    return RunStream(run=run, streams=streams)


def sample_reads(run: Run, n_reads: int, source_mode: str = "local_bytes", cache_dir: str | Path | None = None,
                 bytes_per_read_guess: int = 100) -> tuple[list[tuple[str, str]], list[tuple[str, str]] | None, int]:
    """Read roughly ``n_reads`` records from the start of a run; returns (r1, r2|None, bytes_consumed)."""
    rs = open_run_stream(run, source_mode, cache_dir)
    r1: list[tuple[str, str]] = []
    r2: list[tuple[str, str]] | None = [] if run.layout == "PE" else None
    step = max(262_144, n_reads * bytes_per_read_guess // 4)
    n_streams = len(getattr(rs, "streams", None) or (rs,))
    try:
        while len(r1) < n_reads and not rs.exhausted:
            a, b = rs.advance_bytes(step * n_streams)
            r1.extend(a)
            if r2 is not None and b is not None:
                r2.extend(b)
        n = min(len(r1), len(r2)) if r2 is not None else len(r1)
        n = min(n, n_reads)
        return r1[:n], (r2[:n] if r2 is not None else None), rs.consumed_bytes
    finally:
        if hasattr(rs, "close"):
            rs.close()
