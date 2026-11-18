"""Zero-FASTQ-on-disk SRA streaming and spliced alignment pipelines.

* :class:`RunStream` keeps ONE ``xsra dump`` process per run alive across waves (back-pressure
  pauses it between slices) and cuts exact spot slices from its interleaved FASTQ stream. If a
  stream dies it is restarted and the already-consumed prefix is skipped (the re-streamed bytes are
  honestly counted as wire bytes).
* :func:`align_slice_short` : slice -> glcore sieve -> HISAT2 --dta -> glcore junc (counts + per-bin
  capping) -> compact BAM slice. Nothing uncompressed ever touches disk.
* :func:`align_slice_long` : slice -> minimap2 -ax splice:hq -uf -> glcore junc -> compact BAM.
* :func:`align_run_full` : Arm B full-archive batch path -> full sorted BAM per run on disk.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import random
import re
import select
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sieve_syncmer import build_glcore
from .telemetry import NicCounter, safe_env, wait_rusage


def _tool_path(name: str) -> str:
    tb = os.environ.get("GENELOOP_TOOL_BIN", "").strip()
    if tb:
        cand = Path(tb) / name
        if cand.exists():
            return str(cand)
    return shutil.which(name) or name


XSRA = _tool_path("xsra")


def _open_fastq_text(path: Path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        import gzip

        return gzip.open(path, "rb")
    return open(path, "rb")


def _count_fastq_records(path: Path) -> int:
    n_lines = 0
    with _open_fastq_text(path) as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            n_lines += chunk.count(b"\n")
    return n_lines // 4


class LocalFastqStream:
    """Drop-in local FASTQ(.gz) stream compatible with :class:`RunStream` for Galaxy History inputs."""

    def __init__(
        self,
        run_id: str,
        r1_path: Path,
        r2_path: Path | None = None,
        total_spots: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.r1_path = Path(r1_path)
        self.r2_path = Path(r2_path) if r2_path is not None else None
        self.segs = 2 if self.r2_path is not None else 1
        self.total_spots = int(total_spots) if total_spots is not None else _count_fastq_records(self.r1_path)
        self._fh1 = None
        self._fh2 = None
        self.records_consumed = 0
        self.eof = False
        self.restarts = 0
        self.cpu_user = 0.0
        self.cpu_sys = 0.0

    @property
    def spots_consumed(self) -> int:
        return self.records_consumed // self.segs

    def _ensure_open(self) -> None:
        if self._fh1 is None:
            self._fh1 = _open_fastq_text(self.r1_path)
        if self.r2_path is not None and self._fh2 is None:
            self._fh2 = _open_fastq_text(self.r2_path)

    @staticmethod
    def _read_rec(fh) -> tuple[bytes, bytes, bytes, bytes] | None:
        h = fh.readline()
        if not h:
            return None
        s = fh.readline()
        p = fh.readline()
        q = fh.readline()
        if not (s and p and q):
            return None
        return h, s, p, q

    def pump(self, n_spots: int, out) -> int:
        """Write up to ``n_spots`` spots (interleaved if paired) to ``out``; returns spots written."""
        if self.eof:
            return 0
        self._ensure_open()
        spots = 0
        buf: list[bytes] = []
        for idx in range(int(n_spots)):
            r1 = self._read_rec(self._fh1)
            if r1 is None:
                self.eof = True
                break
            if self._fh2 is not None:
                r2 = self._read_rec(self._fh2)
                if r2 is None:
                    self.eof = True
                    break
                spot_num = self.spots_consumed + spots + 1
                hdr1 = f"@{self.run_id}.{spot_num}/1\n".encode("ascii")
                hdr2 = f"@{self.run_id}.{spot_num}/2\n".encode("ascii")
                buf.extend((hdr1, r1[1], b"+\n", r1[3], hdr2, r2[1], b"+\n", r2[3]))
                self.records_consumed += 2
            else:
                buf.extend(r1)
                self.records_consumed += 1
            spots += 1
            if len(buf) >= 1024:
                out.write(b"".join(buf))
                buf.clear()
        if buf:
            out.write(b"".join(buf))
        if spots < n_spots:
            self.eof = True
        return spots

    def close(self) -> None:
        for fh in (self._fh1, self._fh2):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        self._fh1 = None
        self._fh2 = None


def describe_run(run_id: str, cache_dir: Path) -> dict:
    """Total spots and biological segment count (xsra describe, cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{run_id}.describe.json"
    if p.exists():
        return json.loads(p.read_text())
    out = subprocess.run([XSRA, "describe", run_id, "-l", "200"], capture_output=True, text=True, env=safe_env(), timeout=600)
    txt = out.stdout
    d = json.loads(txt[txt.index("{"):])
    segs = [s for s in d.get("stats", []) if s.get("segment_type", "Biological") == "Biological"]
    res = {
        "run_id": run_id,
        "total_spots": int(d["total_spots"]),
        "n_segments": max(1, len(segs)),
        "mean_lengths": [s.get("mean_length", 0) for s in segs],
    }
    p.write_text(json.dumps(res))
    return res


class RunStream:
    """Persistent prefix stream over one SRA run with exact spot slicing."""

    CHUNK = 4 << 20

    def __init__(self, run_id: str, n_segments: int, total_spots: int, log_dir: Path, threads: int = 4):
        self.run_id = run_id
        self.segs = n_segments
        self.total_spots = total_spots
        self.log_dir = log_dir
        self.threads = threads
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self.carry = b""
        self.records_consumed = 0  # records delivered to consumers
        self.records_read = 0  # records read from current process (incl. skipped prefix)
        self.eof = False
        self.restarts = 0
        self.cpu_user = 0.0
        self.cpu_sys = 0.0
        self.skipped_records_restream = 0

    @property
    def spots_consumed(self) -> int:
        return self.records_consumed // self.segs

    def _start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
        self._log_fh = open(self.log_dir / f"xsra_{self.run_id}.log", "a")
        self.proc = subprocess.Popen(
            [XSRA, "dump", "--skip-technical", self.run_id, "-T", str(self.threads)],
            stdout=subprocess.PIPE, stderr=self._log_fh, bufsize=0, env=safe_env(),
        )
        self.carry = b""
        self.records_read = 0
        if self.records_consumed:
            # Re-stream and discard the already-consumed prefix after an interrupted stream.
            self._skip(self.records_consumed)

    def _read_chunk(self, timeout_ms: int = 120_000) -> bytes:
        out = self.proc.stdout
        if not hasattr(out, "fileno"):
            return out.read(self.CHUNK)
        try:
            fd = out.fileno()
        except Exception:
            return out.read(self.CHUNK)
        poller = select.poll()
        poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        events = poller.poll(timeout_ms)
        if not events:
            # Stalled xsra socket: kill subprocess and return EOF so pump() can restart & skip.
            try:
                self.proc.kill()
            except Exception:
                pass
            return b""
        return out.read(self.CHUNK)

    def _skip(self, n_records: int) -> None:
        sink = _NullSink()
        self._emit(n_records, sink, count_consumed=False)
        self.skipped_records_restream += n_records

    def _emit(self, n_records: int, out, count_consumed: bool = True) -> int:
        written = 0
        buf = self.carry
        while written < n_records:
            if buf:
                arr = np.frombuffer(buf, dtype=np.uint8)
                nl = np.flatnonzero(arr == 10)
                complete = len(nl) // 4
                need = n_records - written
                if complete >= need:
                    cut = int(nl[4 * need - 1]) + 1
                    out.write(buf[:cut])
                    buf = buf[cut:]
                    written += need
                    break
                if complete:
                    cut = int(nl[4 * complete - 1]) + 1
                    out.write(buf[:cut])
                    buf = buf[cut:]
                    written += complete
            chunk = self._read_chunk()
            if not chunk:
                self.eof = True
                break
            buf = buf + chunk if buf else chunk
        self.carry = buf
        self.records_read += written
        if count_consumed:
            self.records_consumed += written
        return written

    def pump(self, n_spots: int, out) -> int:
        """Write the next ``n_spots`` spots to file-like ``out``; returns spots written."""
        if self.eof:
            return 0
        if self.proc is None:
            self._start()
        want = n_spots * self.segs
        got = 0
        tries = 0
        while got < want and not self.eof:
            try:
                got += self._emit(want - got, out)
            except (BrokenPipeError, OSError):
                raise
            if self.eof and self.records_consumed // self.segs < self.total_spots * 0.999 and tries < 3:
                # premature EOF: restart stream and skip the consumed prefix
                self._reap()
                self.restarts += 1
                tries += 1
                self.eof = False
                self._start()
        return got // self.segs

    def _reap(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
        _, u, s = wait_rusage(self.proc)
        self.cpu_user += u
        self.cpu_sys += s
        self.proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def close(self) -> None:
        self._reap()


class _NullSink:
    def write(self, b: bytes) -> int:
        return len(b)


def hisat_binary(idx: Path) -> str:
    """The hisat2 Perl wrapper rejects FIFOs (-f test); call the aligner binary directly."""
    large = Path(str(idx) + ".1.ht2l").exists()
    return _tool_path("hisat2-align-l" if large else "hisat2-align-s")


def _watchdog(procs: list, victims: list, stop: threading.Event) -> None:
    """If any pipeline member dies with an error, kill the FIFO writers so nothing blocks forever."""
    while not stop.is_set():
        for p in procs:
            rc = p.poll()
            if rc is not None and rc != 0:
                for v in victims:
                    try:
                        v.kill()
                    except Exception:
                        pass
                return
        stop.wait(0.5)


def _mkfifo(p: Path) -> None:
    if p.exists():
        p.unlink()
    os.mkfifo(p)


@dataclass
class SliceResult:
    run_id: str
    spots: int
    wall: float
    cpu: dict = field(default_factory=dict)
    sieve: dict = field(default_factory=dict)
    junc: dict = field(default_factory=dict)
    hisat_summary: str = ""
    wire_bytes: int = 0
    bam: str = ""
    junc_tsv: str = ""
    bins_tsv: str = ""


def _junc_args(glcore: Path, prefix: Path, cap_state: Path | None, caps: tuple[int, int, int] | None) -> list[str]:
    a = [str(glcore), "junc", "--junc", f"{prefix}.junc.tsv", "--bins", f"{prefix}.bins.tsv", "--stats", f"{prefix}.junc.json"]
    if cap_state is not None and caps is not None:
        a += ["--cap-state", str(cap_state), "--cap-u", str(caps[0]), "--cap-s", str(caps[1]), "--cap-c", str(caps[2])]
    return a


def align_slice_short(
    stream: RunStream,
    n_spots: int,
    hisat_idx: Path,
    work: Path,
    tag: str,
    active_bin: Path | None,
    cap_state: Path,
    caps: tuple[int, int, int],
    threads: int = 20,
    sieve_threads: int = 4,
    max_intron: int = 500000,
    nic: NicCounter | None = None,
    min_hits: int = 2,
    max_miss: int = 1,
) -> SliceResult:
    glc = build_glcore()
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / tag
    f1, f2 = work / f"{tag}.r1.fifo", work / f"{tag}.r2.fifo"
    _mkfifo(f1)
    _mkfifo(f2)
    env = safe_env()
    rx0 = nic.rx() if nic else 0
    t0 = time.time()
    sieve_cmd = [
        str(glc), "sieve", "-1", str(f1), "-2", str(f2),
        "--stats", f"{prefix}.sieve.json",
        "--min-hits", str(min_hits), "--max-miss", str(max_miss),
        "-t", str(sieve_threads),
    ]
    if active_bin is not None and active_bin.exists():
        sieve_cmd += ["-a", str(active_bin)]
    sieve = subprocess.Popen(sieve_cmd, stdin=subprocess.PIPE, env=env)
    hlog = open(f"{prefix}.hisat2.log", "w")
    hisat = subprocess.Popen(
        [hisat_binary(hisat_idx), "-p", str(threads), "--dta", "--no-unal", "--mm", "--max-intronlen", str(max_intron),
         "--new-summary", "--summary-file", f"{prefix}.hisat2.summary", "-x", str(hisat_idx), "-1", str(f1), "-2", str(f2)],
        stdout=subprocess.PIPE, stderr=hlog, env=env,
    )
    junc = subprocess.Popen(_junc_args(glc, prefix, cap_state, caps), stdin=hisat.stdout, stdout=subprocess.PIPE, env=env)
    hisat.stdout.close()
    bam = subprocess.Popen(["samtools", "view", "-b", "-@", "2", "-o", f"{prefix}.bam", "-"], stdin=junc.stdout, env=env)
    junc.stdout.close()
    spots = [0]
    err: list[BaseException] = []

    def feed():
        try:
            spots[0] = stream.pump(n_spots, sieve.stdin)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            try:
                sieve.stdin.close()
            except Exception:
                pass

    stop = threading.Event()
    wd = threading.Thread(target=_watchdog, args=([hisat, junc, bam], [sieve], stop), daemon=True)
    wd.start()
    th = threading.Thread(target=feed, daemon=True)
    th.start()
    th.join()
    cpu = {}
    for name, p in (("sieve", sieve), ("hisat2", hisat), ("junc", junc), ("samtools", bam)):
        rc, u, s = wait_rusage(p)
        cpu[name] = u + s
        if rc != 0:
            stop.set()
            raise RuntimeError(f"{name} failed rc={rc} in {prefix}")
    stop.set()
    hlog.close()
    if err:
        raise err[0]
    f1.unlink(missing_ok=True)
    f2.unlink(missing_ok=True)
    res = SliceResult(run_id=stream.run_id, spots=spots[0], wall=time.time() - t0, cpu=cpu)
    res.sieve = json.loads(Path(f"{prefix}.sieve.json").read_text())
    res.junc = json.loads(Path(f"{prefix}.junc.json").read_text())
    res.hisat_summary = Path(f"{prefix}.hisat2.summary").read_text() if Path(f"{prefix}.hisat2.summary").exists() else ""
    res.wire_bytes = (nic.rx() - rx0) if nic else 0
    res.bam, res.junc_tsv, res.bins_tsv = f"{prefix}.bam", f"{prefix}.junc.tsv", f"{prefix}.bins.tsv"
    return res


def align_slice_short_se(
    stream: RunStream | LocalFastqStream,
    n_spots: int,
    hisat_idx: Path,
    work: Path,
    tag: str,
    cap_state: Path,
    caps: tuple[int, int, int],
    threads: int = 8,
    max_intron: int = 500000,
    nic: NicCounter | None = None,
) -> SliceResult:
    """Single-end short-read slice -> HISAT2 --dta -> glcore junc -> capped BAM slice."""
    glc = build_glcore()
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / tag
    env = safe_env()
    rx0 = nic.rx() if nic else 0
    t0 = time.time()
    hlog = open(f"{prefix}.hisat2.log", "w")
    hisat = subprocess.Popen(
        [
            hisat_binary(hisat_idx), "-p", str(threads), "--dta", "--no-unal", "--mm",
            "--max-intronlen", str(max_intron), "--new-summary",
            "--summary-file", f"{prefix}.hisat2.summary", "-x", str(hisat_idx), "-U", "-",
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=hlog, env=env,
    )
    junc = subprocess.Popen(_junc_args(glc, prefix, cap_state, caps), stdin=hisat.stdout, stdout=subprocess.PIPE, env=env)
    hisat.stdout.close()
    bam = subprocess.Popen(["samtools", "view", "-b", "-@", "2", "-o", f"{prefix}.bam", "-"], stdin=junc.stdout, env=env)
    junc.stdout.close()
    spots = [0]
    err: list[BaseException] = []

    def feed():
        try:
            spots[0] = stream.pump(n_spots, hisat.stdin)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            try:
                hisat.stdin.close()
            except Exception:
                pass

    th = threading.Thread(target=feed, daemon=True)
    th.start()
    th.join()
    cpu = {}
    for name, p in (("hisat2", hisat), ("junc", junc), ("samtools", bam)):
        rc, u, s = wait_rusage(p)
        cpu[name] = u + s
        if rc != 0:
            raise RuntimeError(f"{name} failed rc={rc} in {prefix}")
    hlog.close()
    if err:
        raise err[0]
    res = SliceResult(run_id=stream.run_id, spots=spots[0], wall=time.time() - t0, cpu=cpu)
    res.sieve = {"pairs_in": spots[0], "pairs_out": spots[0], "pairs_dropped_locus": 0, "pairs_dropped_bait": 0}
    res.junc = json.loads(Path(f"{prefix}.junc.json").read_text())
    res.hisat_summary = Path(f"{prefix}.hisat2.summary").read_text() if Path(f"{prefix}.hisat2.summary").exists() else ""
    res.wire_bytes = (nic.rx() - rx0) if nic else 0
    res.bam, res.junc_tsv, res.bins_tsv = f"{prefix}.bam", f"{prefix}.junc.tsv", f"{prefix}.bins.tsv"
    return res


def ingest_bam_to_evidence(
    bam_path: Path,
    run_id: str,
    work: Path,
    tag: str,
    cap_state: Path | None,
    caps: tuple[int, int, int] | None,
    is_long_read: bool = False,
) -> SliceResult:
    """Stream an existing BAM through ``samtools view -h | glcore junc`` to extract junctions, bins, and capped BAM."""
    glc = build_glcore()
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / tag
    env = safe_env()
    t0 = time.time()
    sv = subprocess.Popen(["samtools", "view", "-h", str(bam_path)], stdout=subprocess.PIPE, env=env)
    jargs = _junc_args(glc, prefix, cap_state, caps)
    if is_long_read:
        jargs += ["--min-anchor", "8"]
    junc = subprocess.Popen(jargs, stdin=sv.stdout, stdout=subprocess.PIPE, env=env)
    sv.stdout.close()
    sbam = subprocess.Popen(["samtools", "view", "-b", "-@", "2", "-o", f"{prefix}.bam", "-"], stdin=junc.stdout, env=env)
    junc.stdout.close()
    cpu = {}
    for name, p in (("samtools_in", sv), ("junc", junc), ("samtools_out", sbam)):
        rc, u, s = wait_rusage(p)
        cpu[name] = u + s
        if rc != 0:
            raise RuntimeError(f"{name} failed rc={rc} for {bam_path}")
    res = SliceResult(run_id=run_id, spots=0, wall=time.time() - t0, cpu=cpu)
    res.junc = json.loads(Path(f"{prefix}.junc.json").read_text())
    res.spots = int(res.junc.get("primary", res.junc.get("records", 0)))
    res.bam, res.junc_tsv, res.bins_tsv = f"{prefix}.bam", f"{prefix}.junc.tsv", f"{prefix}.bins.tsv"
    return res


def align_slice_long(
    stream: RunStream,
    n_spots: int,
    mm2_idx: Path,
    work: Path,
    tag: str,
    cap_state: Path | None,
    caps: tuple[int, int, int] | None,
    threads: int = 20,
    nic: NicCounter | None = None,
) -> SliceResult:
    glc = build_glcore()
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / tag
    env = safe_env()
    rx0 = nic.rx() if nic else 0
    t0 = time.time()
    mlog = open(f"{prefix}.minimap2.log", "w")
    mm = subprocess.Popen(["minimap2", "-ax", "splice:hq", "-uf", "-t", str(threads), "--secondary=no", str(mm2_idx), "-"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=mlog, env=env)
    junc = subprocess.Popen(_junc_args(glc, prefix, cap_state, caps) + ["--min-anchor", "8"], stdin=mm.stdout, stdout=subprocess.PIPE, env=env)
    mm.stdout.close()
    bam = subprocess.Popen(["samtools", "view", "-b", "-@", "2", "-o", f"{prefix}.bam", "-"], stdin=junc.stdout, env=env)
    junc.stdout.close()
    spots = [0]
    err: list[BaseException] = []

    def feed():
        try:
            spots[0] = stream.pump(n_spots, mm.stdin)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            try:
                mm.stdin.close()
            except Exception:
                pass

    th = threading.Thread(target=feed, daemon=True)
    th.start()
    th.join()
    cpu = {}
    for name, p in (("minimap2", mm), ("junc", junc), ("samtools", bam)):
        rc, u, s = wait_rusage(p)
        cpu[name] = u + s
        if rc != 0:
            raise RuntimeError(f"{name} failed rc={rc} in {prefix}")
    mlog.close()
    if err:
        raise err[0]
    res = SliceResult(run_id=stream.run_id, spots=spots[0], wall=time.time() - t0, cpu=cpu)
    res.junc = json.loads(Path(f"{prefix}.junc.json").read_text())
    res.wire_bytes = (nic.rx() - rx0) if nic else 0
    res.bam, res.junc_tsv, res.bins_tsv = f"{prefix}.bam", f"{prefix}.junc.tsv", f"{prefix}.bins.tsv"
    return res


def align_run_full(
    run_id: str,
    long_read: bool,
    hisat_idx: Path,
    mm2_idx: Path | None,
    work: Path,
    threads: int = 18,
    xsra_threads: int = 4,
    sort_threads: int = 4,
    max_intron: int = 500000,
    nic: NicCounter | None = None,
) -> SliceResult:
    """Arm B: stream 100% of a run -> spliced aligner -> full coordinate-sorted BAM on disk."""
    glc = build_glcore()
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / run_id
    env = safe_env()
    rx0 = nic.rx() if nic else 0
    t0 = time.time()
    xlog = open(f"{prefix}.xsra.log", "w")
    xs = subprocess.Popen([XSRA, "dump", "--skip-technical", run_id, "-T", str(xsra_threads)], stdout=subprocess.PIPE, stderr=xlog, env=env)
    procs = [("xsra", xs)]
    if not long_read:
        f1, f2 = work / f"{run_id}.r1.fifo", work / f"{run_id}.r2.fifo"
        _mkfifo(f1)
        _mkfifo(f2)
        split = subprocess.Popen([str(glc), "sieve", "-1", str(f1), "-2", str(f2), "--stats", f"{prefix}.split.json", "-t", "1"],
                                 stdin=xs.stdout, env=env)
        xs.stdout.close()
        alog = open(f"{prefix}.hisat2.log", "w")
        al = subprocess.Popen(
            [hisat_binary(hisat_idx), "-p", str(threads), "--dta", "--no-unal", "--max-intronlen", str(max_intron), "--new-summary",
             "--summary-file", f"{prefix}.hisat2.summary", "-x", str(hisat_idx), "-1", str(f1), "-2", str(f2)],
            stdout=subprocess.PIPE, stderr=alog, env=env,
        )
        procs += [("split", split), ("hisat2", al)]
    else:
        alog = open(f"{prefix}.minimap2.log", "w")
        al = subprocess.Popen(["minimap2", "-ax", "splice:hq", "-uf", "-t", str(threads), "--secondary=no", str(mm2_idx), "-"],
                              stdin=xs.stdout, stdout=subprocess.PIPE, stderr=alog, env=env)
        xs.stdout.close()
        procs += [("minimap2", al)]
    junc = subprocess.Popen(_junc_args(glc, prefix, None, None), stdin=al.stdout, stdout=subprocess.PIPE, env=env)
    al.stdout.close()
    sort = subprocess.Popen(["samtools", "sort", "-@", str(sort_threads), "-m", "1500M", "-T", f"{prefix}.sorttmp",
                             "-o", f"{prefix}.bam", "-"], stdin=junc.stdout, env=env)
    junc.stdout.close()
    procs += [("junc", junc), ("samtools_sort", sort)]
    stop = threading.Event()
    victims = [p for n, p in procs if n in ("split", "xsra")]
    wd = threading.Thread(target=_watchdog, args=([p for n, p in procs if n not in ("split", "xsra")], victims, stop), daemon=True)
    wd.start()
    cpu = {}
    for name, p in procs:
        rc, u, s = wait_rusage(p)
        cpu[name] = u + s
        if rc != 0:
            stop.set()
            raise RuntimeError(f"{name} failed rc={rc} for {run_id}")
    stop.set()
    xlog.close()
    alog.close()
    if not long_read:
        f1.unlink(missing_ok=True)
        f2.unlink(missing_ok=True)
    res = SliceResult(run_id=run_id, spots=0, wall=time.time() - t0, cpu=cpu)
    res.junc = json.loads(Path(f"{prefix}.junc.json").read_text())
    if long_read:
        res.spots = res.junc.get("records", 0)  # minimap2 --secondary=no: one record per read (+ rare supplementary)
    else:
        res.sieve = json.loads(Path(f"{prefix}.split.json").read_text())
        res.spots = res.sieve.get("pairs_in", 0)
        res.hisat_summary = Path(f"{prefix}.hisat2.summary").read_text()
    res.wire_bytes = (nic.rx() - rx0) if nic else 0
    res.bam, res.junc_tsv, res.bins_tsv = f"{prefix}.bam", f"{prefix}.junc.tsv", f"{prefix}.bins.tsv"
    return res


# ----------------------------------------------------------------------------------------------
# Logan S3 Contig Probing, Cross-Species Auto-Detection, and Streaming Alignment (A1 / A2)
# ----------------------------------------------------------------------------------------------

LOGAN_CONTIGS_URL_TEMPLATE = "https://s3.amazonaws.com/logan-pub/c/{acc}/{acc}.contigs.fa.zst"
ENA_FILEREPORT_URL_TEMPLATE = (
    "https://www.ebi.ac.uk/ena/portal/api/filereport"
    "?accession={acc}&result=read_run&fields=run_accession,scientific_name,tax_id&format=json"
)


def logan_contigs_url(acc: str) -> str:
    clean = str(acc).strip()
    return LOGAN_CONTIGS_URL_TEMPLATE.format(acc=clean)


def _write_json_atomic_stream(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def probe_logan_contigs_url(acc: str, timeout: float = 8.0, retries: int = 4) -> dict:
    """Probe Logan S3 contigs object via HTTP HEAD with bounded retries on transient errors."""
    clean = str(acc).strip()
    url = logan_contigs_url(clean)
    max_attempts = max(1, int(retries) + 1)
    last_err = ""
    last_status = 0
    attempt = 0
    while attempt < max_attempts:
        req = urllib.request.Request(url, method="HEAD")
        throttled = False
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                headers = getattr(resp, "headers", {}) or {}
                cl_raw = headers.get("Content-Length") if hasattr(headers, "get") else None
                cl = int(cl_raw) if cl_raw is not None and str(cl_raw).isdigit() else 0
                return {
                    "run_id": clean,
                    "available": status == 200,
                    "http_status": status,
                    "content_length": cl,
                    "url": url,
                }
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            last_status = code
            last_err = f"HTTP {code}"
            if code in (400, 403, 404):
                return {
                    "run_id": clean,
                    "available": False,
                    "http_status": code,
                    "content_length": 0,
                    "url": url,
                }
            if code in (429, 500, 502, 503, 504):
                throttled = True
                max_attempts = max(max_attempts, 5)
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
            last_err = str(exc)
        if attempt + 1 < max_attempts:
            if throttled:
                backoff = min(4.0, 0.25 * (2**attempt)) * (0.75 + 0.5 * random.random())
            else:
                backoff = 0.25 * (attempt + 1)
            time.sleep(backoff)
        attempt += 1
    out = {
        "run_id": clean,
        "available": False,
        "http_status": last_status,
        "content_length": 0,
        "url": url,
    }
    if last_err:
        out["error"] = last_err
    return out


def probe_logan_runs(
    short_runs: list[str],
    cache_path: Path | None = None,
    timeout: float = 8.0,
    retries: int = 4,
    max_workers: int = 16,
) -> dict:
    """Partition short SRA accessions into HTTP-200 ``logan_runs`` and ``fallback_short_runs``."""
    ordered = [str(r).strip() for r in short_runs if str(r).strip()]
    unique_accs: list[str] = []
    seen: set[str] = set()
    for acc in ordered:
        if acc not in seen:
            seen.add(acc)
            unique_accs.append(acc)

    cached_probes: dict[str, dict] = {}
    cache_exact_hit = False
    if cache_path is not None and cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            if isinstance(raw, dict) and raw.get("schema_version") == 1 and isinstance(raw.get("probes"), dict):
                for k, v in raw["probes"].items():
                    if isinstance(v, dict) and int(v.get("http_status", 0)) in (200, 400, 403, 404):
                        cached_probes[str(k)] = v
                if list(raw.get("requested_short_runs", [])) == ordered and all(a in cached_probes for a in unique_accs):
                    cache_exact_hit = True
        except (OSError, ValueError, TypeError):
            cached_probes = {}

    if not cache_exact_hit:
        missing = [a for a in unique_accs if a not in cached_probes]
        if missing:
            workers = max(1, min(int(max_workers), len(missing)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(lambda a: probe_logan_contigs_url(a, timeout=timeout, retries=retries), missing):
                    cached_probes[res["run_id"]] = res

    probes_out = {a: cached_probes[a] for a in unique_accs}
    logan_runs = [a for a in ordered if probes_out[a].get("available", False)]
    fallback_short_runs = [a for a in ordered if not probes_out[a].get("available", False)]
    total_bytes = sum(int(probes_out[a].get("content_length", 0)) for a in unique_accs if probes_out[a].get("available"))
    payload = {
        "schema_version": 1,
        "requested_short_runs": ordered,
        "logan_runs": logan_runs,
        "fallback_short_runs": fallback_short_runs,
        "total_compressed_bytes": total_bytes,
        "probes": probes_out,
    }
    if cache_path is not None and not cache_exact_hit:
        _write_json_atomic_stream(cache_path, payload)
    return payload


def _canonical_binomial(species_str: str) -> str:
    """Normalize ``'Homo sapiens (chr19 + chr22 slice)'`` -> ``'homo sapiens'``."""
    base = re.sub(r"\(.*?\)", " ", str(species_str or "")).strip()
    toks = [t for t in re.split(r"\s+", base) if t]
    if len(toks) >= 2:
        return f"{toks[0].lower()} {toks[1].lower()}"
    return base.lower()


_CROSS_SPECIES_NOTE_RE = re.compile(r"\b(sister|cross[- ]species|congeneric|related\s+species|heterologous)\b", re.I)


def _fetch_ena_run_organism(acc: str, timeout: float = 5.0) -> dict:
    clean = str(acc).strip()
    url = ENA_FILEREPORT_URL_TEMPLATE.format(acc=clean)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            rows = json.loads(resp.read().decode("utf-8", errors="replace"))
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                return {
                    "run_id": clean,
                    "scientific_name": str(rows[0].get("scientific_name") or "").strip(),
                    "tax_id": str(rows[0].get("tax_id") or "").strip(),
                }
    except Exception:
        pass
    return {"run_id": clean, "scientific_name": "", "tax_id": ""}


def detect_cross_species_rna(
    gate: dict,
    run_ids: list[str],
    cache_dir: Path | None = None,
    allow_network: bool = True,
    timeout: float = 5.0,
) -> tuple[bool, dict]:
    """Auto-detect whether RNA accessions originate from a species distinct from ``gate['species']``."""
    if "rna_cross_species" in gate and gate["rna_cross_species"] is not None:
        flag = bool(gate["rna_cross_species"])
        return flag, {"cross_species": flag, "reason": "gate_rna_cross_species"}

    target_binom = _canonical_binomial(str(gate.get("species", "")))
    target_taxid = str(gate["taxid"]).strip() if gate.get("taxid") is not None else ""

    explicit_orgs: dict[str, str] = {}
    if isinstance(gate.get("sra_run_organisms"), dict):
        for k, v in gate["sra_run_organisms"].items():
            explicit_orgs[str(k).strip()] = str(v or "").strip()
    if isinstance(gate.get("runs"), list):
        for item in gate["runs"]:
            if isinstance(item, dict) and item.get("acc") and item.get("species"):
                explicit_orgs[str(item["acc"]).strip()] = str(item["species"]).strip()
    if explicit_orgs and target_binom:
        for r in run_ids:
            org = _canonical_binomial(explicit_orgs.get(str(r).strip(), ""))
            if org and org != target_binom:
                return True, {
                    "cross_species": True,
                    "reason": "manifest_run_organism_mismatch",
                    "run_id": r,
                    "run_organism": org,
                    "target_species": target_binom,
                }
        if all(str(r).strip() in explicit_orgs for r in run_ids if str(r).strip()):
            return False, {"cross_species": False, "reason": "manifest_run_organisms_match"}

    note = str(gate.get("rna_source_note", ""))
    if note and _CROSS_SPECIES_NOTE_RE.search(note):
        return True, {"cross_species": True, "reason": "rna_source_note"}

    if not run_ids or (not target_binom and not target_taxid):
        return False, {"cross_species": False, "reason": "no_species_metadata"}

    org_cache_p = (cache_dir / "run_organisms.json") if cache_dir is not None else None
    org_map: dict[str, dict] = {}
    cache_dirty = False
    if org_cache_p is not None and org_cache_p.exists():
        try:
            loaded = json.loads(org_cache_p.read_text())
            if isinstance(loaded, dict):
                org_map = {str(k): v for k, v in loaded.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            org_map = {}

    if allow_network:
        candidates: list[str] = []
        seen_pfx: set[str] = set()
        for r in run_ids:
            rc = str(r).strip()
            if not rc or rc in org_map:
                continue
            pfx = rc[:8]
            if pfx not in seen_pfx:
                seen_pfx.add(pfx)
                candidates.append(rc)
        for r in run_ids:
            if len(candidates) >= 32:
                break
            rc = str(r).strip()
            if rc and rc not in org_map and rc not in candidates:
                candidates.append(rc)
        if candidates:
            with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as ex:
                for info in ex.map(lambda a: _fetch_ena_run_organism(a, timeout=timeout), candidates):
                    if info.get("scientific_name") or info.get("tax_id"):
                        org_map[info["run_id"]] = info
                        cache_dirty = True
            if cache_dirty and org_cache_p is not None:
                _write_json_atomic_stream(org_cache_p, org_map)

    for r in run_ids:
        info = org_map.get(str(r).strip())
        if not info:
            continue
        rtax = str(info.get("tax_id") or "").strip()
        rsci = _canonical_binomial(str(info.get("scientific_name") or ""))
        if target_taxid and rtax and rtax != target_taxid:
            return True, {
                "cross_species": True,
                "reason": "ena_taxid_mismatch",
                "run_id": r,
                "run_tax_id": rtax,
                "target_tax_id": target_taxid,
                "run_organism": info.get("scientific_name", ""),
            }
        if target_binom and rsci and rsci != target_binom:
            return True, {
                "cross_species": True,
                "reason": "ena_species_mismatch",
                "run_id": r,
                "run_organism": info.get("scientific_name", ""),
                "target_species": target_binom,
            }
    return False, {"cross_species": False, "reason": "same_species"}


def fetch_logan_contig_file(
    acc: str,
    dst: Path,
    timeout: float = 30.0,
    retries: int = 2,
    expected_bytes: int = 0,
) -> int:
    """Download ``{acc}.contigs.fa.zst`` atomically into ``dst``; returns compressed byte count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        existing_sz = int(dst.stat().st_size)
        if existing_sz > 0 and (expected_bytes <= 0 or existing_sz == int(expected_bytes)):
            return existing_sz
        dst.unlink(missing_ok=True)
    url = logan_contigs_url(acc)
    attempts = max(1, int(retries) + 1)
    part = dst.with_name(f".{dst.name}.part.{os.getpid()}.{threading.get_ident()}")
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            expected_cl = int(expected_bytes) if expected_bytes > 0 else 0
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as fo:
                headers = getattr(resp, "headers", None)
                if headers is None and hasattr(resp, "info"):
                    headers = resp.info()
                cl_raw = None
                if headers is not None and hasattr(headers, "get"):
                    cl_raw = headers.get("Content-Length")
                    if cl_raw is None:
                        cl_raw = headers.get("content-length")
                if cl_raw is not None:
                    cl_str = str(cl_raw).strip()
                    if cl_str.isdigit() and int(cl_str) > 0:
                        expected_cl = int(cl_str)
                shutil.copyfileobj(resp, fo, length=1 << 20)
            sz = int(part.stat().st_size)
            if sz <= 0:
                raise RuntimeError(f"Empty Logan contigs payload for {acc} from {url}")
            if expected_cl > 0 and sz != expected_cl:
                raise RuntimeError(
                    f"Truncated Logan contigs payload for {acc} from {url}: got {sz} bytes, expected Content-Length={expected_cl}"
                )
            if expected_bytes > 0 and sz != int(expected_bytes):
                raise RuntimeError(
                    f"Size mismatch for Logan contigs payload {acc} from {url}: got {sz} bytes, expected {expected_bytes}"
                )
            os.replace(part, dst)
            return sz
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Failed to download Logan contigs for {acc} ({url}): {last_exc}") from last_exc


def align_logan_contigs(
    logan_runs: list[str],
    genome_fa: Path,
    mm2_flags: list[str],
    glcore_bin: Path,
    out_dir: Path,
    tmp_dir: Path,
    bam_out: Path,
    threads: int = 24,
    nic: NicCounter | None = None,
    local_zst_map: dict[str, Path] | None = None,
) -> dict:
    """Fetch Logan ``c/`` contigs into ``tmp_dir``, stream ``zstd -> minimap2 -y -> glcore logan -> samtools sort``, and purge ``tmp_dir``."""
    if not logan_runs:
        raise ValueError("align_logan_contigs requires at least one Logan accession")
    out_dir.mkdir(parents=True, exist_ok=True)
    bam_out.parent.mkdir(parents=True, exist_ok=True)
    bai_out = Path(f"{bam_out}.bai")
    j_unw = out_dir / "logan.per_run.unw.tsv"
    j_ka = out_dir / "logan.per_run.ka.tsv"
    b_unw = out_dir / "logan.bins.unw.tsv"
    b_ka = out_dir / "logan.bins.ka.tsv"
    stats_p = out_dir / "logan.align_stats.json"
    meta_p = out_dir / "logan.align_meta.json"

    if (
        bam_out.exists()
        and bai_out.exists()
        and j_unw.exists()
        and j_ka.exists()
        and b_unw.exists()
        and b_ka.exists()
        and meta_p.exists()
    ):
        try:
            cached_meta = json.loads(meta_p.read_text())
            if (
                cached_meta.get("logan_runs") == list(logan_runs)
                and cached_meta.get("mm2_flags") == list(mm2_flags)
            ):
                return cached_meta
        except (OSError, ValueError):
            pass

    env = safe_env()
    rx0 = nic.rx() if nic else 0
    t0 = time.time()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[Path] = []
    try:
        t_dl0 = time.time()
        zst_inputs: list[Path] = []
        dl_bytes = 0
        if local_zst_map is not None:
            for acc in logan_runs:
                p = Path(local_zst_map[acc])
                zst_inputs.append(p)
                dl_bytes += int(p.stat().st_size)
        else:
            unique_runs = list(dict.fromkeys(logan_runs))
            dl_workers = max(1, min(8, len(unique_runs)))

            def _dl_one(acc: str) -> tuple[str, Path, int]:
                dst = tmp_dir / f"{acc}.contigs.fa.zst"
                sz = fetch_logan_contig_file(acc, dst)
                return acc, dst, sz

            with ThreadPoolExecutor(max_workers=dl_workers) as ex:
                by_acc = {}
                for acc, dst, sz in ex.map(_dl_one, unique_runs):
                    by_acc[acc] = dst
                    downloaded_files.append(dst)
                    dl_bytes += sz
            zst_inputs = [by_acc[acc] for acc in logan_runs]
        dl_wall = round(time.time() - t_dl0, 3)

        zstd_bin = _tool_path("zstd")
        mm2_bin = _tool_path("minimap2")
        samtools_bin = _tool_path("samtools")
        sort_threads = min(16, max(2, int(threads) // 3))

        zstd_cmd = [zstd_bin, "-dcq"] + [str(p) for p in zst_inputs]
        mm2_cmd = [mm2_bin] + list(mm2_flags) + ["-y", "-t", str(max(1, int(threads))), str(genome_fa), "-"]
        lj_cmd = [
            str(glcore_bin),
            "logan",
            "--junc-unw", str(j_unw),
            "--junc-ka", str(j_ka),
            "--bins-unw", str(b_unw),
            "--bins-ka", str(b_ka),
            "--stats", str(stats_p),
        ]
        sort_cmd = [
            samtools_bin, "sort",
            "-@", str(sort_threads),
            "-m", "2G",
            "-T", f"{bam_out}.sorttmp",
            "-o", str(bam_out),
            "-",
        ]

        mm2_log_p = out_dir / "logan.minimap2.log"
        cpu_by_proc: dict[str, float] = {}
        p_zstd: subprocess.Popen | None = None
        p_mm2: subprocess.Popen | None = None
        p_lj: subprocess.Popen | None = None
        p_sort: subprocess.Popen | None = None
        reaped: set[int] = set()
        killed_by_us: set[int] = set()
        stop = threading.Event()
        wd: threading.Thread | None = None
        kill_lock = threading.Lock()

        def _terminate_proc(vp: subprocess.Popen) -> None:
            with kill_lock:
                if vp.poll() is None:
                    killed_by_us.add(id(vp))
                    try:
                        vp.terminate()
                    except OSError:
                        pass

        def _kill_proc(vp: subprocess.Popen) -> None:
            with kill_lock:
                if vp.poll() is None:
                    killed_by_us.add(id(vp))
                    try:
                        vp.kill()
                    except OSError:
                        pass

        with open(mm2_log_p, "w") as mm2_log:
            try:
                p_zstd = subprocess.Popen(zstd_cmd, stdout=subprocess.PIPE, env=env)
                p_mm2 = subprocess.Popen(mm2_cmd, stdin=p_zstd.stdout, stdout=subprocess.PIPE, stderr=mm2_log, env=env)
                p_zstd.stdout.close()
                p_lj = subprocess.Popen(lj_cmd, stdin=p_mm2.stdout, stdout=subprocess.PIPE, env=env)
                p_mm2.stdout.close()
                p_sort = subprocess.Popen(sort_cmd, stdin=p_lj.stdout, env=env)
                p_lj.stdout.close()

                stage_procs = [
                    ("zstd", p_zstd),
                    ("minimap2", p_mm2),
                    ("glcore_logan", p_lj),
                    ("samtools_sort", p_sort),
                ]

                def _logan_watchdog() -> None:
                    while not stop.is_set():
                        for _sname, pr in stage_procs:
                            rc_w = pr.poll()
                            if rc_w is not None and rc_w != 0:
                                for _vn, vp in stage_procs:
                                    if vp is not pr:
                                        _terminate_proc(vp)
                                time.sleep(0.05)
                                for _vn, vp in stage_procs:
                                    if vp is not pr:
                                        _kill_proc(vp)
                                return
                        stop.wait(0.05)

                wd = threading.Thread(target=_logan_watchdog, daemon=True)
                wd.start()

                stage_rcs: list[tuple[str, int, subprocess.Popen]] = []
                for name, pr in stage_procs:
                    rc, u, s = wait_rusage(pr)
                    reaped.add(id(pr))
                    cpu_by_proc[name] = round(u + s, 3)
                    if rc != 0:
                        stage_rcs.append((name, int(rc), pr))
                        for _rn, rp in stage_procs:
                            if rp is not pr:
                                _terminate_proc(rp)
                if stage_rcs:
                    stop.set()
                    with kill_lock:
                        self_failed = [
                            (nm, rc)
                            for nm, rc, pr in stage_rcs
                            if id(pr) not in killed_by_us and rc not in (-13, 141)
                        ]
                    bad_name, bad_rc = self_failed[-1] if self_failed else (stage_rcs[-1][0], stage_rcs[-1][1])
                    bam_out.unlink(missing_ok=True)
                    bai_out.unlink(missing_ok=True)
                    raise RuntimeError(f"{bad_name} failed with rc={bad_rc} during Logan contig alignment")
            finally:
                stop.set()
                if wd is not None:
                    wd.join(timeout=2.0)
                all_procs = [
                    ("zstd", p_zstd),
                    ("minimap2", p_mm2),
                    ("glcore_logan", p_lj),
                    ("samtools_sort", p_sort),
                ]
                for _nm, pr in all_procs:
                    if pr is not None:
                        if pr.stdout is not None and not pr.stdout.closed:
                            try:
                                pr.stdout.close()
                            except OSError:
                                pass
                        if pr.poll() is None:
                            try:
                                pr.terminate()
                            except OSError:
                                pass
                for _nm, pr in all_procs:
                    if pr is not None and pr.poll() is None:
                        try:
                            pr.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            try:
                                pr.kill()
                            except OSError:
                                pass
                for nm, pr in all_procs:
                    if pr is not None and id(pr) not in reaped:
                        try:
                            _rc, u, s = wait_rusage(pr)
                            reaped.add(id(pr))
                            cpu_by_proc[nm] = round(u + s, 3)
                        except Exception:
                            try:
                                pr.wait(timeout=1.0)
                            except Exception:
                                pass

        p_idx = subprocess.Popen([samtools_bin, "index", "-@", str(sort_threads), str(bam_out)], env=env)
        rc_i, u_i, s_i = wait_rusage(p_idx)
        if rc_i != 0:
            raise RuntimeError(f"samtools index failed rc={rc_i} for {bam_out}")
        cpu_by_proc["samtools_index"] = round(u_i + s_i, 3)
    finally:
        for fp in downloaded_files:
            fp.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    nic_delta = (nic.rx() - rx0) if nic else 0
    wire_bytes = max(int(dl_bytes), int(nic_delta))
    lj_stats = json.loads(stats_p.read_text()) if stats_p.exists() else {}
    total_cpu_s = round(sum(cpu_by_proc.values()), 3)
    meta = {
        "logan_runs": list(logan_runs),
        "n_logan_runs": len(logan_runs),
        "mm2_flags": list(mm2_flags),
        "bam_path": str(bam_out),
        "bam_bytes": int(bam_out.stat().st_size),
        "junc_unw_path": str(j_unw),
        "junc_ka_path": str(j_ka),
        "bins_unw_path": str(b_unw),
        "bins_ka_path": str(b_ka),
        "download_wall_s": dl_wall,
        "download_bytes": int(dl_bytes),
        "wire_bytes": wire_bytes,
        "wall_s": round(time.time() - t0, 3),
        "cpu_by_proc_s": cpu_by_proc,
        "total_cpu_s": total_cpu_s,
        "stream_stats": lj_stats,
    }
    _write_json_atomic_stream(meta_p, meta)
    return meta


def ingest_per_run_junc_tsv(js, path: Path, genome) -> list[tuple[tuple[str, int, int], int, int]]:
    """Ingest 7-column ``glcore logan`` per-run TSV: ``run_id, contig, start, end, n_total, n_ua, max_anchor``."""
    from .evidence import junction_motif

    added: list[tuple[tuple[str, int, int], int, int]] = []
    if not path.exists():
        return added
    seen_runs = set(js.run_ids)
    with open(path, "r") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            rid = f[0]
            if rid not in seen_runs:
                seen_runs.add(rid)
                js.run_ids.append(rid)
            k = (f[1], int(f[2]), int(f[3]))
            nt, nua, ma = int(f[4]), int(f[5]), int(f[6])
            d = js.ua.get(k)
            if d is None:
                d = {}
                js.ua[k] = d
                if genome is not None:
                    js.motif[k] = junction_motif(genome, *k)
            if nua > 0:
                d[rid] = d.get(rid, 0) + nua
            js.total[k] = js.total.get(k, 0) + nt
            if ma > js.max_anchor.get(k, 0):
                js.max_anchor[k] = ma
            added.append((k, nua, nt))
    return added


def load_unw_junc_stats(
    junc_unw_path: Path,
) -> tuple[dict[tuple[str, int, int], int], dict[tuple[str, int, int], int]]:
    """Return ``(unw_runs_count, unw_ua_count)`` per ``(contig, start, end)`` from ``glcore logan --junc-unw``."""
    runs: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    ua: dict[tuple[str, int, int], int] = defaultdict(int)
    if not junc_unw_path.exists():
        return {}, {}
    with open(junc_unw_path, "r") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            nua = int(f[5])
            if nua > 0:
                k = (f[1], int(f[2]), int(f[3]))
                runs[k].add(f[0])
                ua[k] += nua
    return {k: len(v) for k, v in runs.items()}, dict(ua)
