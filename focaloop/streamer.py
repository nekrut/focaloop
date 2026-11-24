"""3-tier INSDC byte-range resolver and block reader.

Resolution order for a public run accession:

  Tier 1 ``cram_range``   remote CRAM/BAM over HTTP ``Range`` (when an aligned URL is known)
  Tier 2 ``s3_odp_spot``  AWS Open Data ``sra-pub-run-odp`` SRA object
                          (``https://sra-pub-run-odp.s3.amazonaws.com/sra/<RUN>/<RUN>``), decoded by
                          ``xsra dump -Q -P https <RUN>`` when ``xsra`` is on ``PATH``; ``-Q`` is mandatory
                          so full Illumina qualities are kept instead of SRA-lite ``?`` placeholders
  Tier 3 ``ena_https_range``  ENA FASTQ (gzip) over HTTPS ``206 Partial Content``

plus ``local_file`` for staged FASTQ / FASTQ.gz (the deterministic offline path
used by tests and Galaxy datasets). Every reader counts ``consumed_bytes`` so the
stop gate can enforce byte budgets, assigns each fixed-size block to the
``discovery`` or ``holdout`` partition with :func:`is_holdout_block`, and keeps
compressed byte offsets so unmapped reads can be coalesced into residual spans.
"""

from __future__ import annotations

import gzip
import re
import shutil
import subprocess
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .hooks import coalesce_residual_spans, is_holdout_block

__all__ = [
    "TIERS",
    "TierPlan",
    "ByteSource",
    "LocalFileSource",
    "HttpRangeSource",
    "FastqBlock",
    "FastqBlockReader",
    "resolve_tiers",
    "ena_fastq_urls",
    "s3_odp_url",
    "xsra_argv",
    "XsraSource",
    "open_source",
    "coalesce_residual_spans",
    "is_holdout_block",
]

TIERS = ("cram_range", "s3_odp_spot", "ena_https_range", "local_file")
ACC_RE = re.compile(r"^(SRR|ERR|DRR)\d{6,}$")
USER_AGENT = "focaloop/0.1 (HTTP 206 range streamer)"


@dataclass(frozen=True)
class TierPlan:
    tier: str
    run_id: str
    urls: tuple[str, ...]
    supports_range: bool = True


def ena_fastq_urls(acc: str, layout: str = "PE") -> list[str]:
    """ENA FASTQ paths: vol1/fastq/<6-char prefix>/[<0-padded suffix>/]<acc>/<acc>[_1|_2].fastq.gz."""
    if not ACC_RE.match(acc):
        raise ValueError(f"not an INSDC run accession: {acc!r}")
    digits = len(acc) - 3
    base = f"https://ftp.sra.ebi.ac.uk/vol1/fastq/{acc[:6]}"
    if digits > 6:
        base += "/" + acc[9:].rjust(3, "0")
    base += f"/{acc}"
    if layout == "SE":
        return [f"{base}/{acc}.fastq.gz"]
    return [f"{base}/{acc}_1.fastq.gz", f"{base}/{acc}_2.fastq.gz"]


def s3_odp_url(acc: str) -> str:
    """AWS Open Data SRA object for a run: ``https://sra-pub-run-odp.s3.amazonaws.com/sra/<RUN>/<RUN>``."""
    if not ACC_RE.match(acc):
        raise ValueError(f"not an INSDC run accession: {acc!r}")
    return f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"


def xsra_argv(acc: str, binary: str = "xsra") -> list[str]:
    """``xsra dump -Q -P https <RUN>``: full-quality SRA only (never SRA-lite), HTTPS provider, FASTQ to stdout."""
    return [binary, "dump", "-Q", "-P", "https", acc]


def resolve_tiers(acc: str, layout: str = "PE", cram_url: str | None = None) -> list[TierPlan]:
    """Ordered tier plans for one run accession (first available tier wins at stream time)."""
    plans: list[TierPlan] = []
    if cram_url:
        plans.append(TierPlan("cram_range", acc, (cram_url,)))
    if ACC_RE.match(acc):
        plans.append(TierPlan("s3_odp_spot", acc, (s3_odp_url(acc),)))
        plans.append(TierPlan("ena_https_range", acc, tuple(ena_fastq_urls(acc, layout))))
    else:
        raise ValueError(f"cannot resolve {acc!r}: not an SRR/ERR/DRR accession")
    return plans


class ByteSource:
    tier: str = "local_file"
    total_bytes: int
    bytes_read: int = 0

    def read_range(self, start: int, end_exclusive: int) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class LocalFileSource(ByteSource):
    tier = "local_file"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.total_bytes = self.path.stat().st_size
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
    """HTTP(S) ``Range: bytes=a-b`` source; total size from HEAD / Content-Range."""

    def __init__(self, url: str, tier: str = "ena_https_range", total_bytes: int | None = None,
                 timeout: float = 60.0, retries: int = 3) -> None:
        self.url = url
        self.tier = tier
        self.timeout = timeout
        self.retries = retries
        self.bytes_read = 0
        self.requests_made = 0
        self.total_bytes = total_bytes if total_bytes is not None else self._probe_size()

    def _probe_size(self) -> int:
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                return int(cr.rsplit("/", 1)[1])
            return int(r.headers.get("Content-Length", 0))

    def read_range(self, start: int, end_exclusive: int) -> bytes:
        end_exclusive = min(end_exclusive, self.total_bytes)
        if end_exclusive <= start:
            return b""
        last: Exception | None = None
        for _ in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end_exclusive - 1}", "User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                    if r.status == 200:  # server ignored Range: slice locally
                        data = data[start:end_exclusive]
                self.requests_made += 1
                self.bytes_read += len(data)
                return data
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last = exc
        raise RuntimeError(f"range fetch failed for {self.url}: {last}")


class XsraSource(ByteSource):
    """Tier 2: sequential FASTQ stream from ``xsra dump -Q -P https <RUN>`` (AWS ODP / NCBI full-quality SRA).

    Byte accounting counts decoded FASTQ bytes. Reads must be sequential; the total size is unknown until
    ``xsra`` exits, so ``total_bytes`` is a sentinel until EOF and then equals the bytes streamed. The
    constructor primes the first chunk and raises ``RuntimeError`` if ``xsra`` produces no FASTQ, so
    :func:`open_source` can fall back to ENA HTTPS 206 cleanly.
    """

    tier = "s3_odp_spot"
    UNKNOWN = 1 << 62

    def __init__(self, acc: str, binary: str = "xsra", prime_bytes: int = 1 << 16) -> None:
        self.acc = acc
        self.url = s3_odp_url(acc)
        self.argv = xsra_argv(acc, binary)
        self.bytes_read = 0
        self.total_bytes = self.UNKNOWN
        self._pos = 0
        self._buf = b""
        self._proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._buf = self._pull(prime_bytes)
        if not self._buf.startswith(b"@"):
            err = self._finish()
            raise RuntimeError(f"xsra produced no FASTQ for {acc} (rc={self._proc.returncode}): {err[-300:]!r}")

    def _pull(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n and self._proc.stdout is not None:
            chunk = self._proc.stdout.read(n - len(out))
            if not chunk:
                break
            out += chunk
        return bytes(out)

    def _finish(self) -> bytes:
        if self._proc.poll() is None:
            self._proc.kill()
        _, err = self._proc.communicate()
        return err or b""

    def read_range(self, start: int, end_exclusive: int) -> bytes:
        if start != self._pos:
            raise ValueError(f"XsraSource is sequential: asked for {start}, stream is at {self._pos}")
        want = end_exclusive - start
        data = self._buf[:want]
        self._buf = self._buf[want:]
        if len(data) < want:
            data += self._pull(want - len(data))
        if len(data) < want:  # EOF: the stream size is now known
            err = self._finish()
            if self._proc.returncode != 0:  # never pass a truncated run off as complete
                raise RuntimeError(f"xsra dump exited rc={self._proc.returncode} for {self.acc} after "
                                   f"{self._pos + len(data)} bytes: {err[-300:]!r}")
            self.total_bytes = self._pos + len(data)
        self._pos += len(data)
        self.bytes_read += len(data)
        return data

    def close(self) -> None:
        if self._proc.poll() is None:
            self._finish()

    def __del__(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()


def open_source(ref: str | Path, layout: str = "SE", xsra: str | None = "xsra") -> list[ByteSource]:
    """Local path -> LocalFileSource; http(s) URL -> HttpRangeSource; accession -> Tier 2 ``xsra`` when on
    ``PATH`` (full qualities via ``-Q``), else / on failure Tier 3 ENA HTTPS 206 range sources."""
    s = str(ref)
    if s.startswith(("http://", "https://")):
        return [HttpRangeSource(s)]
    if ACC_RE.match(s):
        binary = shutil.which(xsra) if xsra else None
        if binary:
            try:
                return [XsraSource(s, binary)]
            except (OSError, RuntimeError):
                pass  # fall through to ENA
        plan = [p for p in resolve_tiers(s, layout) if p.tier == "ena_https_range"][0]
        return [HttpRangeSource(u, tier=plan.tier) for u in plan.urls]
    return [LocalFileSource(s)]


@dataclass
class FastqBlock:
    run_id: str
    block_index: int
    partition: str  # "discovery" | "holdout"
    byte_start: int
    byte_end: int
    reads: list[str]
    names: list[str] = field(default_factory=list)


class FastqBlockReader:
    """Sequential fixed-size block reader over one FASTQ (plain or gzip/BGZF) byte source."""

    def __init__(self, source: ByteSource, run_id: str, block_bytes: int = 65_536,
                 holdout_fraction: float = 0.10, salt: str = "focaloop-v1") -> None:
        self.source = source
        self.run_id = run_id
        self.block_bytes = int(block_bytes)
        self.holdout_fraction = holdout_fraction
        self.salt = salt
        self.consumed_bytes = 0
        self.block_index = 0
        self._gz: bool | None = None
        self._d = None
        self._in_member = False
        self._rem = b""
        self.reads_emitted = 0

    @property
    def tier(self) -> str:
        return self.source.tier

    @property
    def total_bytes(self) -> int:
        return self.source.total_bytes

    @property
    def exhausted(self) -> bool:
        return self.consumed_bytes >= self.source.total_bytes

    def _decode(self, raw: bytes, final: bool) -> bytes:
        if self._gz is None:
            if not raw:
                self._gz = False
                return b""
            self._gz = raw[:2] == b"\x1f\x8b"
            if self._gz:
                self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                self._in_member = False
            else:
                lead = raw.lstrip()[:16].lower()
                if lead.startswith((b"<!doctype", b"<html", b"<?xml", b"<head", b"<body")):
                    raise ValueError(f"FASTQ source {self.run_id!r} appears to be HTML/XML rather than FASTQ")
                if raw[:4] in (b"\x28\xb5\x2f\xfd", b"CRAM") or raw[:3] == b"BZh" or raw[:6] == b"\xfd7zXZ\x00":
                    raise ValueError(f"Unsupported compressed/binary format in FASTQ source {self.run_id!r}")
        if not self._gz:
            return raw
        out = bytearray()
        data = raw
        try:
            while data:
                self._in_member = True
                out += self._d.decompress(data)
                if self._d.eof:
                    trailing = self._d.unused_data
                    self._d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    self._in_member = False
                    data = trailing
                else:
                    data = b""
            if final and self._d is not None:
                out += self._d.flush()
                if self._in_member and not self._d.eof:
                    raise ValueError(
                        f"Truncated gzip member in FASTQ source {self.run_id!r} at compressed byte {self.consumed_bytes}"
                    )
        except zlib.error as exc:
            raise ValueError(f"Corrupt gzip FASTQ stream in {self.run_id!r}: {exc}") from exc
        return bytes(out)

    def next_block(self) -> FastqBlock | None:
        if self.exhausted:
            return None
        start = self.consumed_bytes
        raw = self.source.read_range(start, start + self.block_bytes)
        self.consumed_bytes += len(raw)
        text = self._rem + self._decode(raw, self.exhausted)
        if not text and self.exhausted:
            idx = self.block_index
            self.block_index += 1
            part = "holdout" if is_holdout_block(self.run_id, idx, self.holdout_fraction, self.salt) else "discovery"
            return FastqBlock(self.run_id, idx, part, start, start + len(raw), [], [])
        lines = text.split(b"\n")
        if self.exhausted:
            while lines and lines[-1] in (b"", b"\r"):
                lines.pop()
            if len(lines) % 4 != 0:
                raise ValueError(
                    f"Truncated FASTQ record at EOF in {self.run_id!r} ({len(lines) % 4} trailing line(s))"
                )
            n_complete = len(lines) // 4
        else:
            n_complete = (len(lines) - 1) // 4
        used = n_complete * 4
        self._rem = b"\n".join(lines[used:]) if not self.exhausted else b""
        reads: list[str] = []
        names: list[str] = []
        for i in range(0, used, 4):
            hdr = lines[i].rstrip(b"\r")
            seq = lines[i + 1].rstrip(b"\r")
            plus = lines[i + 2].rstrip(b"\r")
            qual = lines[i + 3].rstrip(b"\r")
            if not hdr.startswith(b"@") or len(hdr) < 2:
                raise ValueError(f"Malformed FASTQ header line in {self.run_id!r}: {hdr[:80]!r}")
            if not plus.startswith(b"+"):
                raise ValueError(f"Malformed FASTQ '+' separator line in {self.run_id!r}: {plus[:80]!r}")
            if not seq or len(seq) != len(qual):
                raise ValueError(
                    f"FASTQ sequence/quality length mismatch in {self.run_id!r} ({len(seq)} vs {len(qual)})"
                )
            reads.append(seq.decode("ascii", "replace").upper())
            names.append(hdr[1:].split()[0].decode("ascii", "replace"))
        self.reads_emitted += len(reads)
        idx = self.block_index
        self.block_index += 1
        part = "holdout" if is_holdout_block(self.run_id, idx, self.holdout_fraction, self.salt) else "discovery"
        return FastqBlock(self.run_id, idx, part, start, start + len(raw), reads, names)

    def __iter__(self) -> Iterator[FastqBlock]:
        while True:
            b = self.next_block()
            if b is None:
                return
            yield b


def read_fasta(path: str | Path) -> list[tuple[str, str, dict[str, str]]]:
    """Parse FASTA (plain or gz) into (id, sequence, header key=value attributes)."""
    p = Path(path)
    opener = gzip.open if p.read_bytes()[:2] == b"\x1f\x8b" else open
    recs: list[tuple[str, str, dict[str, str]]] = []
    name, attrs, buf = None, {}, []
    with opener(p, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf).upper(), attrs))
                parts = line[1:].split()
                name = parts[0]
                attrs = dict(tok.split("=", 1) for tok in parts[1:] if "=" in tok)
                buf = []
            else:
                buf.append(line)
    if name is not None:
        recs.append((name, "".join(buf).upper(), attrs))
    return recs
