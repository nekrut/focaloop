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
import hashlib
from pathlib import Path
import socket
import subprocess
import threading
import time
import zlib

from .manifest import Run


class _NonRangeServerError(RuntimeError):
    """Raised when a remote HTTP server ignores Range headers on partial slice requests."""


def _check_unsupported_magic(payload: bytes, source_label: str = "") -> None:
    """Reject binary or unsupported compression formats at stream offset 0 with a clear error."""
    if not payload:
        return
    src_note = f" ({source_label})" if source_label else ""
    if payload.startswith(b"BZh"):
        raise RuntimeError(
            f"Unsupported bzip2-compressed stream{src_note} (magic 425a68) - provide uncompressed or gzip-compressed FASTQ"
        )
    if payload.startswith(b"\x28\xb5\x2f\xfd"):
        raise RuntimeError(
            f"Unsupported zstd-compressed stream{src_note} (magic 28b52ffd) - provide uncompressed or gzip-compressed FASTQ"
        )
    if payload.startswith(b"\xfd7zXZ\x00"):
        raise RuntimeError(
            f"Unsupported xz-compressed stream{src_note} (magic fd377a585a00) - provide uncompressed or gzip-compressed FASTQ"
        )
    if payload.startswith(b"NCBI.sra"):
        raise RuntimeError(
            f"Received binary NCBI .sra archive{src_note} instead of FASTQ stream - enable xsra/sra_odp or provide FASTQ URLs"
        )


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

    def _read_bounded_stream(self, raw_iter, max_bytes: int) -> tuple[bytes, int, bool]:
        """Read up to ``max_bytes + 1`` bytes from an iterator/stream, returning (data, wire_read, exceeded)."""
        buf = bytearray()
        wire = 0
        exceeded = False
        for chunk in raw_iter:
            if not chunk:
                continue
            wire += len(chunk)
            take = min(len(chunk), max_bytes - len(buf))
            if take > 0:
                buf.extend(chunk[:take])
            if wire > max_bytes:
                exceeded = True
                break
        return bytes(buf), wire, exceeded

    def _curl_range_bounded(self, start: int, end_inclusive: int, expected_len: int) -> tuple[bytes, int]:
        """Bounded curl byte-range fallback that aborts immediately if a server ignores Range."""
        proc = subprocess.Popen(
            [
                "curl",
                "-4",
                "-sS",
                "-L",
                "--fail",
                "--max-time",
                str(int(self.timeout)),
                "-r",
                f"{start}-{end_inclusive}",
                self.url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None and proc.stderr is not None
        chunks: list[bytes] = []
        wire = 0
        exceeded = False
        try:
            while True:
                blk = proc.stdout.read(min(262_144, max(4096, expected_len + 1 - wire)))
                if not blk:
                    break
                wire += len(blk)
                if wire > expected_len:
                    exceeded = True
                    proc.kill()
                    break
                chunks.append(blk)
            _, err_b = proc.communicate(timeout=15)
        except Exception:
            proc.kill()
            proc.wait()
            raise
        if exceeded:
            with self._stats_lock:
                self.requests_made += 1
                self.network_bytes += wire
            raise _NonRangeServerError(
                f"Server at {self.url} ignored HTTP Range bytes={start}-{end_inclusive} (returned >{expected_len} bytes)"
            )
        if proc.returncode != 0:
            err_txt = (err_b or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"curl range {start}-{end_inclusive} failed (rc={proc.returncode}): {err_txt}")
        out = b"".join(chunks)
        min_expected = expected_len if (self.total_bytes <= 0 or end_inclusive < self.total_bytes - 1) else (self.total_bytes - start)
        if len(out) < min_expected:
            raise RuntimeError(
                f"Short HTTP range body from {self.url} for {start}-{end_inclusive}: got {len(out)} bytes, expected {min_expected}"
            )
        return out, wire

    def _fetch(self, start: int, end_inclusive: int) -> bytes:
        sem = self._host_sem(self.host)
        last: Exception | None = None
        gz_url = self.url.split("?", 1)[0].lower().endswith((".gz", ".bgz"))
        expected_len = max(0, end_inclusive - start + 1)
        full_file_req = start == 0 and (self.total_bytes > 0 and end_inclusive + 1 >= self.total_bytes)

        def _validate_payload(payload: bytes, ctype: str = "") -> None:
            if "text/html" in ctype.lower():
                raise RuntimeError(f"server returned HTML error page ({ctype}) instead of FASTQ bytes")
            if start == 0 and payload:
                head = payload.lstrip()[:16].lower()
                if head.startswith((b"<!doctype", b"<html", b"<?xml")):
                    raise RuntimeError(f"server returned HTML/XML body ({head!r}) instead of FASTQ bytes")
                _check_unsupported_magic(payload, self.url)
                if gz_url and len(payload) >= 2 and payload[:2] != b"\x1f\x8b":
                    raise RuntimeError(f"expected gzip magic 1f8b at offset 0 of {self.url}, got {payload[:4].hex()}")

        for attempt in range(10):
            with sem:
                t0 = time.time()
                try:
                    r = self._get_session(self.host).get(
                        self.url,
                        headers={"Range": f"bytes={start}-{end_inclusive}"},
                        stream=True,
                        timeout=self.timeout,
                    )
                    try:
                        if r.status_code == 200 and not full_file_req:
                            cl_hdr = r.headers.get("Content-Length", "")
                            cl_int = int(cl_hdr) if cl_hdr.isdigit() else -1
                            if start > 0 or (cl_int >= 0 and cl_int > expected_len):
                                with self._stats_lock:
                                    self.requests_made += 1
                                raise _NonRangeServerError(
                                    f"Server at {self.url} ignored HTTP Range header (HTTP 200 for bytes={start}-{end_inclusive})"
                                )
                            data, wire, exceeded = self._read_bounded_stream(r.iter_content(chunk_size=262_144), expected_len)
                            if exceeded:
                                with self._stats_lock:
                                    self.requests_made += 1
                                    self.network_bytes += wire
                                raise _NonRangeServerError(
                                    f"Server at {self.url} ignored HTTP Range header (HTTP 200 streamed >{expected_len} bytes)"
                                )
                        elif r.status_code in (200, 206):
                            data, wire, exceeded = self._read_bounded_stream(r.iter_content(chunk_size=262_144), expected_len)
                            if exceeded:
                                with self._stats_lock:
                                    self.requests_made += 1
                                    self.network_bytes += wire
                                raise _NonRangeServerError(
                                    f"Server at {self.url} returned oversized body (>{expected_len} bytes) for Range {start}-{end_inclusive}"
                                )
                        else:
                            raise RuntimeError(f"HTTP {r.status_code} for range {start}-{end_inclusive}")
                        ctype = r.headers.get("Content-Type", "")
                    finally:
                        r.close()

                    min_expected = (
                        expected_len
                        if (self.total_bytes <= 0 or end_inclusive < self.total_bytes - 1)
                        else (self.total_bytes - start)
                    )
                    if len(data) < min_expected:
                        with self._stats_lock:
                            self.requests_made += 1
                            self.network_bytes += wire
                        raise RuntimeError(
                            f"Short HTTP {r.status_code} body for range {start}-{end_inclusive}: got {len(data)} of {min_expected} bytes"
                        )
                    _validate_payload(data, ctype)
                    dt = time.time() - t0
                    with self._stats_lock:
                        self.requests_made += 1
                        self.network_bytes += wire
                        self.network_seconds += dt
                    return data
                except _NonRangeServerError:
                    raise
                except Exception as exc:  # fall back to bounded curl -4
                    last = exc
                    try:
                        out, c_wire = self._curl_range_bounded(start, end_inclusive, expected_len)
                        _validate_payload(out)
                        dt = time.time() - t0
                        with self._stats_lock:
                            self.requests_made += 1
                            self.network_bytes += c_wire
                            self.network_seconds += dt
                        return out
                    except _NonRangeServerError:
                        raise
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


def _complementary_mate_stem(q1: str, q2: str) -> str | None:
    """If q1 and q2 end in complementary mate tags (_1/_2, .1/.2, :1/:2) with identical stem, return stem."""
    if len(q1) > 2 and len(q1) == len(q2):
        sep1, m1 = q1[-2], q1[-1]
        sep2, m2 = q2[-2], q2[-1]
        if sep1 == sep2 and sep1 in ("_", ".", ":") and m1 == "1" and m2 == "2" and q1[:-2] == q2[:-2]:
            return q1[:-2]
    return None


class GzipStreamDecoder:
    """Stateful gzip/plain FASTQ decoder (multi-member aware) that yields complete FASTQ records."""

    def __init__(self, expect_gzip: bool = False, source_label: str = "") -> None:
        self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._rem = b""
        self._plain: bool | None = None
        self._member_started: bool = False
        self.expect_gzip = expect_gzip
        self.source_label = source_label
        self.bytes_fed = 0
        self.records_emitted = 0
        self.malformed_skipped = 0

    def _src_note(self) -> str:
        return f" ({self.source_label})" if self.source_label else ""

    def _decompress_lines(self, raw: bytes, eof: bool = False) -> list[bytes]:
        chunks: list[bytes] = [self._rem] if self._rem else []
        base_offset = self.bytes_fed
        self.bytes_fed += len(raw)
        if self._plain is None and raw:
            head = raw.lstrip()[:16].lower()
            if head.startswith((b"<!doctype", b"<html", b"<?xml")):
                raise RuntimeError(f"received HTML/XML payload ({head!r}) instead of FASTQ stream{self._src_note()}")
            _check_unsupported_magic(raw, self.source_label)
            if self.expect_gzip and len(raw) >= 2 and raw[:2] != b"\x1f\x8b":
                raise RuntimeError(
                    f"expected gzip-compressed FASTQ stream (magic 1f8b){self._src_note()}, got {raw[:4].hex()}"
                )
            self._plain = raw[:2] != b"\x1f\x8b"
        if self._plain:
            if raw:
                chunks.append(raw)
        elif self._plain is False:
            mv = memoryview(raw)
            n = len(mv)
            step = 262_144  # 256 KiB slice avoids O(N^2) unused_data copies on multi-member BGZF files
            pos = 0
            while pos < n:
                end = pos + step if pos + step < n else n
                data: bytes | memoryview = mv[pos:end]
                chunk_start = base_offset + pos
                pos = end
                while data:
                    self._member_started = True
                    try:
                        part = self._d.decompress(data)
                        if part:
                            chunks.append(part)
                    except zlib.error as exc:
                        raise RuntimeError(
                            f"Corrupt gzip FASTQ stream{self._src_note()} near compressed byte {chunk_start}: {exc}"
                        ) from exc
                    if self._d.eof:
                        unused = self._d.unused_data
                        # Ignore trailing NUL padding after last gzip member if present
                        if unused and not unused.strip(b"\x00"):
                            unused = b""
                        self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                        self._member_started = False
                        data = unused
                        if not data:
                            break
                    else:
                        break
            if eof:
                try:
                    tail = self._d.flush()
                    if tail:
                        chunks.append(tail)
                except zlib.error as exc:
                    raise RuntimeError(
                        f"Corrupt gzip FASTQ trailer{self._src_note()} at EOF ({self.bytes_fed} bytes): {exc}"
                    ) from exc
                if self._member_started and not self._d.eof:
                    raise RuntimeError(
                        f"Truncated gzip FASTQ stream{self._src_note()} at EOF ({self.bytes_fed} bytes): incomplete gzip member"
                    )
        if not chunks:
            return [b""]
        buf = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        if eof and buf and not buf.endswith(b"\n"):
            buf += b"\n"
        return buf.split(b"\n")

    def _finalize_parse_check(self, eof: bool) -> None:
        if eof and self._rem.strip():
            self.malformed_skipped += 1
        if self.records_emitted == 0 and (
            (eof and self.malformed_skipped > 0) or self.malformed_skipped >= 64
        ):
            raise RuntimeError(
                f"Zero valid FASTQ records decoded from stream{self._src_note()} (malformed lines={self.malformed_skipped})"
            )

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
                if hdr:
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
        self._finalize_parse_check(eof)
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
                if hdr:
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
        self._finalize_parse_check(eof)
        return recs


class FastqPrefixStream:
    """Sequential compressed-byte cursor over one FASTQ file (any ByteSource)."""

    def __init__(self, source: ByteSource, expect_gzip: bool | None = None) -> None:
        self.source = source
        loc_raw = str(getattr(source, "path", "") or getattr(source, "url", ""))
        if expect_gzip is None:
            loc = loc_raw.split("?", 1)[0].lower()
            expect_gzip = loc.endswith((".gz", ".bgz"))
        self.decoder = GzipStreamDecoder(expect_gzip=bool(expect_gzip), source_label=loc_raw)
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

    def skip_bytes(self, n_bytes: int, chunk_bytes: int = 16 * 1024 * 1024) -> int:
        """Advance ``n_bytes`` in bounded chunks while discarding decoded records so RSS stays flat."""
        target = min(self.source.total_bytes, self.consumed_bytes + max(0, int(n_bytes)))
        step = max(65_536, int(chunk_bytes))
        before = self.consumed_bytes
        while self.consumed_bytes < target:
            nxt = min(target, self.consumed_bytes + step)
            raw = self.source.read_range(self.consumed_bytes, nxt)
            if not raw:
                break
            self.consumed_bytes += len(raw)
            _ = self.decoder.feed(raw, eof=self.exhausted)
        return self.consumed_bytes - before


@dataclass
class RunStream:
    """PE (two cursors) or SE (one cursor) stream for one run."""

    _MAX_UNPAIRED_CARRY = 16_384

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

    @property
    def malformed_skipped(self) -> int:
        return sum(s.decoder.malformed_skipped for s in self.streams)

    def skip_bytes(self, n_bytes_total: int, chunk_bytes: int = 16 * 1024 * 1024) -> int:
        """Skip ``n_bytes_total`` compressed bytes in bounded chunks while keeping PE mate cursors aligned."""
        rem = max(0, int(n_bytes_total))
        step = max(65_536, int(chunk_bytes))
        before = self.consumed_bytes
        while rem > 0 and not self.exhausted:
            take = min(rem, step)
            prev = self.consumed_bytes
            _a, _b = self.advance_bytes(take)
            delta = self.consumed_bytes - prev
            if delta <= 0:
                break
            rem -= max(take, delta)
        return self.consumed_bytes - before

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
                if len(r1) > self._MAX_UNPAIRED_CARRY or len(r2) > self._MAX_UNPAIRED_CARRY:
                    raise RuntimeError(
                        f"Paired-end mate streams severely out of balance in run {self.run.run_id!r} "
                        f"(buffered R1={len(r1)}, R2={len(r2)})"
                    )
                self._carry_named = [r1, r2]
            return [], []

        # Detect complementary mate suffixes (_1/_2, .1/.2, :1/:2) when raw tokens differ only by mate digit
        if r1[0][0] != r2[0][0] and _complementary_mate_stem(r1[0][0], r2[0][0]) is not None:
            sep = r1[0][0][-2]
            suf1 = f"{sep}1"
            suf2 = f"{sep}2"
            r1 = [(q[:-2] if len(q) > 2 and q.endswith(suf1) else q, s, ql) for q, s, ql in r1]
            r2 = [(q[:-2] if len(q) > 2 and q.endswith(suf2) else q, s, ql) for q, s, ql in r2]

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
        if matched == 0 and n >= 1:
            raise RuntimeError(
                f"Paired-end mate headers do not match in run {self.run.run_id!r} "
                f"(0 of {n} records paired, first R1={r1[0][0]!r}, first R2={r2[0][0]!r})"
            )
        rem1 = r1[last_i1 + 1:]
        rem2 = r2[last_i2 + 1:]
        if self.exhausted:
            self.skipped_unpaired_mates += len(rem1) + len(rem2)
            self._carry_named = [[], []]
        else:
            if len(rem1) > self._MAX_UNPAIRED_CARRY or len(rem2) > self._MAX_UNPAIRED_CARRY:
                raise RuntimeError(
                    f"Unpaired mate backlog exceeded {self._MAX_UNPAIRED_CARRY} records in run {self.run.run_id!r} "
                    f"(first unpaired R1={rem1[0][0] if rem1 else ''!r}, R2={rem2[0][0] if rem2 else ''!r})"
                )
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
    safe_rid = run.run_id.replace("/", "_").replace("\\", "_")
    for mate_idx, f in enumerate(run.files):
        if source_mode == "local_bytes":
            if not f.local_path:
                raise ValueError(f"run {run.run_id}: local_bytes mode needs local_path for {f.basename}")
            src: ByteSource = LocalFileSource(f.local_path, f.bytes if f.bytes else None)
        elif source_mode == "http_range":
            if "sra-pub-run-odp.s3.amazonaws.com/sra/" in f.url or run.extra.get("_odp_sra_only"):
                raise RuntimeError(
                    f"Run {run.run_id} has no ENA FASTQ mirror (only binary NCBI .sra at {f.url}) and requires 'xsra' on PATH"
                )
            url_hash = hashlib.sha1(f.url.encode("utf-8")).hexdigest()[:12]
            cache = (Path(cache_dir) / f"{safe_rid}_{mate_idx}_{url_hash}.part") if cache_dir else None
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
