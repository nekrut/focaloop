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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import socket
import subprocess
import threading
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
    """HTTP 206 byte-range source with shared Keep-Alive connection pooling and per-host concurrency guard."""

    _CHUNK_BYTES = 32 * 1024 * 1024  # 32 MiB per HTTP 206 range request
    _HOST_SESSIONS: dict[str, object] = {}
    _HOST_SEMS: dict[str, threading.BoundedSemaphore] = {}
    _HOST_LOCK = threading.Lock()

    def __init__(self, url: str, total_bytes: int, cache_file: str | Path | None = None, timeout: float = 90.0) -> None:
        from urllib.parse import urlparse

        self.url = url
        self.host = (urlparse(url).hostname or "default").lower()
        self.total_bytes = total_bytes
        self.cache_file = Path(cache_file) if cache_file else None
        self.timeout = timeout
        self.network_bytes = 0
        self.requests_made = 0
        self.network_seconds = 0.0
        self._stats_lock = threading.Lock()

    @classmethod
    def _host_sem(cls, host: str) -> threading.BoundedSemaphore:
        with cls._HOST_LOCK:
            sem = cls._HOST_SEMS.get(host)
            if sem is None:
                if "ncbi.nlm.nih.gov" in host:
                    cap = 8
                elif "ebi.ac.uk" in host:
                    cap = 8
                else:
                    cap = 24
                sem = threading.BoundedSemaphore(cap)
                cls._HOST_SEMS[host] = sem
            return sem

    @classmethod
    def _get_session(cls, host: str):
        with cls._HOST_LOCK:
            sess = cls._HOST_SESSIONS.get(host)
            if sess is None:
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util import connection as _conn

                _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
                sess = requests.Session()
                adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32)
                sess.mount("https://", adapter)
                sess.mount("http://", adapter)
                sess.headers["User-Agent"] = "panexpr/2.4 (HTTP 206 range streamer)"
                cls._HOST_SESSIONS[host] = sess
            return sess

    def _fetch(self, start: int, end_inclusive: int) -> bytes:
        sem = self._host_sem(self.host)
        last: Exception | None = None
        gz_url = self.url.split("?", 1)[0].lower().endswith((".gz", ".bgz"))

        def _validate_payload(payload: bytes, ctype: str = "") -> None:
            if "text/html" in ctype.lower():
                raise RuntimeError(f"server returned HTML error page ({ctype}) instead of FASTQ bytes")
            if start == 0 and payload:
                head = payload.lstrip()[:16].lower()
                if head.startswith((b"<!doctype", b"<html", b"<?xml")):
                    raise RuntimeError(f"server returned HTML/XML body ({head!r}) instead of FASTQ bytes")
                if gz_url and len(payload) >= 2 and payload[:2] != b"\x1f\x8b":
                    raise RuntimeError(f"expected gzip magic 1f8b at offset 0 of {self.url}, got {payload[:4].hex()}")

        for attempt in range(10):
            with sem:
                t0 = time.time()
                try:
                    r = self._get_session(self.host).get(self.url, headers={"Range": f"bytes={start}-{end_inclusive}"}, timeout=self.timeout)
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code} for range {start}-{end_inclusive}")
                    data = r.content
                    if r.status_code == 200:  # server ignored the range
                        data = data[start:end_inclusive + 1]
                    _validate_payload(data, r.headers.get("Content-Type", ""))
                    dt = time.time() - t0
                    with self._stats_lock:
                        self.requests_made += 1
                        self.network_bytes += len(data)
                        self.network_seconds += dt
                    return data
                except Exception as exc:  # fall back to curl -4 which proved robust on this cluster
                    last = exc
                    try:
                        out = subprocess.run(
                            ["curl", "-4", "-sS", "-L", "--fail", "--max-time", str(int(self.timeout)), "-r", f"{start}-{end_inclusive}", self.url],
                            check=True, capture_output=True, timeout=self.timeout + 30,
                        ).stdout
                        _validate_payload(out)
                        dt = time.time() - t0
                        with self._stats_lock:
                            self.requests_made += 1
                            self.network_bytes += len(out)
                            self.network_seconds += dt
                        return out
                    except Exception as exc2:
                        last = exc2
            # Sleep outside the host semaphore so other threads are not blocked during backoff
            time.sleep(min(20.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"range fetch failed for {self.url}: {last}")

    def _fetch_chunked(self, start: int, end_exclusive: int) -> bytes:
        span = end_exclusive - start
        if span <= self._CHUNK_BYTES:
            return self._fetch(start, end_exclusive - 1)
        slices: list[tuple[int, int]] = []
        pos = start
        while pos < end_exclusive:
            nxt = min(pos + self._CHUNK_BYTES, end_exclusive)
            slices.append((pos, nxt - 1))
            pos = nxt
        if len(slices) == 1 or "ncbi.nlm.nih.gov" in self.host or "ebi.ac.uk" in self.host:
            # Reuse the warm Keep-Alive TLS session sequentially per worker when outer worker pool already saturates host cap
            return b"".join(self._fetch(a, b) for a, b in slices)
        with ThreadPoolExecutor(max_workers=min(4, len(slices))) as ex:
            parts = list(ex.map(lambda ab: self._fetch(ab[0], ab[1]), slices))
        return b"".join(parts)

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
                        fh.write(self._fetch_chunked(have, start))
                    have = start
                chunk = self._fetch_chunked(have, end_exclusive)
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "ab") as fh:
                    fh.write(chunk)
            with open(self.cache_file, "rb") as fh:
                fh.seek(start)
                return fh.read(end_exclusive - start)
        return self._fetch_chunked(start, end_exclusive)


def _norm_qname(header: bytes) -> str:
    """Normalize a FASTQ ``@header`` line to its mate-stripped query identifier."""
    h = header[1:].strip() if header.startswith(b"@") else header.strip()
    if not h:
        return ""
    sp = h.find(b" ")
    tb = h.find(b"\t")
    cut = sp if (sp >= 0 and (tb < 0 or sp < tb)) else tb
    tok = h[:cut] if cut >= 0 else h
    if len(tok) > 2 and tok[-2:] in (b"/1", b"/2", b".1", b".2") and not tok[:-2].isdigit():
        # Note: SRA headers like @SRR123.45.1 vs @SRR123.45/1: strip /1,/2 unconditionally;
        # strip .1/.2 only when there are >=2 dots (@ACC.SPOT.READ) so @ACC.SPOT stays intact.
        if tok[-2:-1] == b"/" or tok.count(b".") >= 2:
            tok = tok[:-2]
    return tok.decode("latin-1")


class GzipStreamDecoder:
    """Stateful gzip/plain FASTQ decoder (multi-member aware) that yields complete FASTQ records."""

    def __init__(self, expect_gzip: bool = False) -> None:
        self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._rem = b""
        self._plain: bool | None = None
        self.expect_gzip = expect_gzip
        self.records_emitted = 0
        self.malformed_skipped = 0

    def _decompress_lines(self, raw: bytes, eof: bool = False) -> list[bytes]:
        chunks: list[bytes] = [self._rem] if self._rem else []
        if self._plain is None and raw:
            head = raw.lstrip()[:16].lower()
            if head.startswith((b"<!doctype", b"<html", b"<?xml")):
                raise RuntimeError(f"received HTML/XML payload ({head!r}) instead of FASTQ stream")
            if self.expect_gzip and len(raw) >= 2 and raw[:2] != b"\x1f\x8b":
                raise RuntimeError(f"expected gzip-compressed FASTQ stream (magic 1f8b), got {raw[:4].hex()}")
            self._plain = raw[:2] != b"\x1f\x8b"
        if self._plain:
            if raw:
                chunks.append(raw)
        else:
            mv = memoryview(raw)
            n = len(mv)
            step = 262_144  # 256 KiB slice avoids O(N^2) unused_data copies on multi-member BGZF files
            pos = 0
            broken = False
            while pos < n and not broken:
                end = pos + step if pos + step < n else n
                data: bytes | memoryview = mv[pos:end]
                pos = end
                while data:
                    try:
                        part = self._d.decompress(data)
                        if part:
                            chunks.append(part)
                    except zlib.error:
                        broken = True
                        break
                    if self._d.eof:
                        data = self._d.unused_data
                        self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                        if not data:
                            break
                    else:
                        break
        if not chunks:
            return [b""]
        buf = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        if eof and buf and not buf.endswith(b"\n"):
            buf += b"\n"
        return buf.split(b"\n")

    def feed_named(self, raw: bytes, eof: bool = False) -> list[tuple[str, str, str]]:
        lines = self._decompress_lines(raw, eof=eof)
        # Hold the trailing incomplete line (and up to 3 complete lines not yet forming a 4-line record)
        limit = len(lines) - 1
        recs: list[tuple[str, str, str]] = []
        append = recs.append
        norm = _norm_qname
        i = 0
        while i + 3 < limit:
            hdr = lines[i]
            plus = lines[i + 2]
            if not hdr or hdr[0] != 64 or not plus or plus[0] != 43:
                self.malformed_skipped += 1
                i += 1
                continue
            seq = lines[i + 1].rstrip(b"\r ")
            qual = lines[i + 3].rstrip(b"\r ")
            i += 4
            if seq and len(seq) == len(qual):
                append((norm(hdr), seq.decode("latin-1"), qual.decode("latin-1")))
            else:
                self.malformed_skipped += 1
        self._rem = b"\n".join(lines[i:])
        self.records_emitted += len(recs)
        return recs

    def feed(self, raw: bytes, eof: bool = False) -> list[tuple[str, str]]:
        lines = self._decompress_lines(raw, eof=eof)
        limit = len(lines) - 1
        recs: list[tuple[str, str]] = []
        append = recs.append
        i = 0
        while i + 3 < limit:
            hdr = lines[i]
            plus = lines[i + 2]
            if not hdr or hdr[0] != 64 or not plus or plus[0] != 43:
                self.malformed_skipped += 1
                i += 1
                continue
            seq = lines[i + 1].rstrip(b"\r ")
            qual = lines[i + 3].rstrip(b"\r ")
            i += 4
            if seq and len(seq) == len(qual):
                append((seq.decode("latin-1"), qual.decode("latin-1")))
            else:
                self.malformed_skipped += 1
        self._rem = b"\n".join(lines[i:])
        self.records_emitted += len(recs)
        return recs


class FastqPrefixStream:
    """Sequential compressed-byte cursor over one FASTQ file (any ByteSource)."""

    def __init__(self, source: ByteSource, expect_gzip: bool | None = None) -> None:
        self.source = source
        if expect_gzip is None:
            loc = str(getattr(source, "path", "") or getattr(source, "url", "")).split("?", 1)[0].lower()
            expect_gzip = loc.endswith((".gz", ".bgz"))
        self.decoder = GzipStreamDecoder(expect_gzip=bool(expect_gzip))
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
        return self.decoder.feed_named(raw, eof=self.exhausted)

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
        if isinstance(self.streams[0].source, HttpRangeSource):
            with ThreadPoolExecutor(max_workers=len(self.streams)) as ex:
                batches = list(ex.map(lambda s: s.advance_by_named(max(min_bytes, int(s.total_bytes * frac))), self.streams))
        else:
            batches = [s.advance_by_named(max(min_bytes, int(s.total_bytes * frac))) for s in self.streams]
        return self._pair_named(batches)

    def advance_bytes(self, n_bytes_total: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]] | None]:
        """Advance so that this run consumes ``n_bytes_total`` more compressed bytes (split across mates by size)."""
        if len(self.streams) == 1:
            b = self.streams[0].advance_by(max(1, int(n_bytes_total)))
            self.reads_emitted += len(b)
            return b, None
        tot = max(1, self.total_bytes)
        if isinstance(self.streams[0].source, HttpRangeSource):
            with ThreadPoolExecutor(max_workers=len(self.streams)) as ex:
                batches = list(ex.map(lambda s: s.advance_by_named(max(1, int(n_bytes_total * s.total_bytes / tot))), self.streams))
        else:
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
