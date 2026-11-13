"""Stage 2: spot-block streamer with host sieve.

Sources (all emit FASTQ bytes on a pipe, nothing lands on disk):
  OdpXsra    AWS ODP via `xsra dump --skip-technical -Q -P https` (full Phred qualities, interleaved mates)
  EnaFastq   ENA paired FASTQ.gz over HTTPS, lockstep-verified and interleaved in memory (fallback without ODP)
  SangerCram Sanger Pf8 CRAM fetched with parallel HTTP ranges into `samtools collate | samtools fastq` (paired, last resort)

Wire bytes for xsra come from the kernel TCP counters (`ss -tinp`, bytes_received per socket of the pid),
for the HTTP fallbacks from response sizes.
"""
import gzip
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import requests
from urllib3.util.retry import Retry

from . import genome as G
from . import kernels as Kn

XSRA = os.environ.get("XSRA", "xsra")
SS_RE = re.compile(r'pid=(\d+),')
BYTES_RE = re.compile(r'bytes_received:(\d+)')


class SocketMeter(threading.Thread):
    """polls `ss -tinpH` and keeps the max bytes_received per (pid, socket) for the watched pids."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.pids = set()
        self.seen = {}
        self.lock = threading.Lock()
        self.alive = True

    def watch(self, pid):
        with self.lock:
            self.pids.add(int(pid))

    def poll(self):
        try:
            out = subprocess.run(["ss", "-tinpH"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return
        lines = out.splitlines()
        with self.lock:
            pids = set(self.pids)
        for i, line in enumerate(lines):
            m = SS_RE.search(line)
            if not m or int(m.group(1)) not in pids:
                continue
            key = (int(m.group(1)), " ".join(line.split()[2:4]))
            b = BYTES_RE.search(lines[i + 1]) if i + 1 < len(lines) else None
            if b:
                with self.lock:
                    self.seen[key] = max(self.seen.get(key, 0), int(b.group(1)))

    def total(self, pid):
        self.poll()
        with self.lock:
            return sum(v for (p, _), v in self.seen.items() if p == int(pid))

    def run(self):
        while self.alive:
            self.poll()
            time.sleep(self.interval)


class Source:
    paired = True
    kind = "base"

    def __init__(self):
        self.errors = []
        self._closed_early = False
        self._close_lock = threading.Lock()

    def _check_proc(self, proc, label):
        if proc is None:
            return
        rc = proc.wait()
        if rc != 0 and not (self._closed_early and rc in (-15, -13, -9, 141, 143)):
            self.errors.append(f"{label} exited with code {rc}")

    def _terminate_proc(self, proc, label, wait_s=1.0):
        """Terminate a streaming subprocess cleanly; close stdout and escalate to SIGKILL quickly."""
        if proc is None:
            return
        if proc.poll() is None:
            self._closed_early = True
            if getattr(proc, "stdout", None) is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            try:
                proc.terminate()
                proc.wait(wait_s)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass
        else:
            self._check_proc(proc, label)

    def read(self, n):
        raise NotImplementedError

    def wire_bytes(self):
        return 0

    def close(self, early=False):
        if early:
            self._closed_early = True


class OdpXsra(Source):
    """sequential xsra dumps of every ENA run of a sample (each run's spot order is the archive order)."""
    kind = "aws_odp_xsra"

    def __init__(self, runs, threads=2, log=None):
        super().__init__()
        self.runs = list(runs)
        self.threads = threads
        self.log = log or subprocess.DEVNULL
        self.meter = SocketMeter()
        self.meter.start()
        self.proc = None
        self.pids = []
        self.ri = -1
        self.done_bytes = 0
        self._next()

    def _next(self):
        if self._closed_early:
            return False
        if self.proc is not None:
            self._check_proc(self.proc, f"xsra dump {self.runs[self.ri]}")
        self.ri += 1
        if self.ri >= len(self.runs):
            self.proc = None
            return False
        cmd = [XSRA, "dump", self.runs[self.ri], "--skip-technical", "-Q", "-P", "https", "-T", str(self.threads)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.pids.append(self.proc.pid)
        self.meter.watch(self.proc.pid)
        return True

    def read(self, n):
        while not self._closed_early:
            proc = self.proc
            if proc is None:
                break
            try:
                b = proc.stdout.read(n)
            except (ValueError, OSError):
                return b""
            if b:
                return b
            if self._closed_early or not self._next():
                break
        return b""

    def wire_bytes(self):
        return sum(self.meter.total(p) for p in self.pids)

    def close(self, early=False):
        with self._close_lock:
            super().close(early=early)
            proc = self.proc
            self.proc = None
            ri = self.ri
            if proc is not None:
                label = f"xsra dump {self.runs[ri]}" if 0 <= ri < len(self.runs) else "xsra dump"
                self._terminate_proc(proc, label)
            self.meter.alive = False


ODP_URL = "https://sra-pub-run-odp.s3.amazonaws.com/sra/{0}/{0}"
CRAI_DIR = Path(os.environ.get("S", "/media/anton/hd2/malariavar_staging")) / "data" / "crai"
# CRAM 3.0 EOF container: htslib's constant, also the last 38 bytes of every complete Sanger Pf8 CRAM
CRAM_EOF = bytes.fromhex("0f000000ffffffff0fe0454f4600000000010005bdd94f0001000606010001000100ee63014b")
N_NUCLEAR = 14


_TL = threading.local()


def _session():
    """one persistent keep-alive session per thread: no TCP + TLS handshake per range."""
    ss = getattr(_TL, "s", None)
    if ss is None:
        ss = requests.Session()
        # servers drop idle keep-alive sockets (Sanger's RGW does): retry the request at once on a fresh one
        ad = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=Retry(
            total=3, connect=3, read=3, status=0, backoff_factor=0.1, raise_on_status=False))
        ss.mount("https://", ad)
        ss.mount("http://", ad)
        _TL.s = ss
    return ss


_POOLS = {}
_POOLS_LOCK = threading.Lock()


def shared_pool(name, streams):
    """long-lived per-process download pool: its threads, and so their sessions, stay warm across files
    (a pool per file starts every file on cold connections)."""
    key = (os.getpid(), name, streams)  # a forked child builds its own (threads do not survive fork)
    with _POOLS_LOCK:
        ex = _POOLS.get(key)
        if ex is None:
            ex = _POOLS[key] = ThreadPoolExecutor(streams, thread_name_prefix=f"dl-{name}")
        return ex


def head_size(url):
    r = _session().head(url, timeout=120)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def pull_range(url, fd, a, b, tries=8, piece=1 << 20, shift=0):
    """GET bytes a..b (inclusive) on this thread's warm session, pwrite each piece as it arrives at its source
    offset minus `shift` (memory stays at one piece per thread); a dropped connection resumes from the last
    byte written."""
    pos = a
    for t in range(tries):
        try:
            with _session().get(url, headers={"Range": f"bytes={pos}-{b}"}, timeout=(30, 120), stream=True) as r:
                r.raise_for_status()
                if r.status_code != 206:
                    raise IOError(f"range ignored (HTTP {r.status_code})")
                while pos <= b:
                    buf = r.raw.read(min(piece, b - pos + 1))
                    if not buf:
                        break
                    mv = memoryview(buf)
                    while mv:
                        n = os.pwrite(fd, mv, pos - shift)
                        pos += n
                        mv = mv[n:]
            if pos > b:
                return b - a + 1
            raise IOError(f"range {a}-{b} ended at {pos}")
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(min(30, 2 * (t + 1)))


def ranged_download(url, path, streams=8, chunk=16 << 20):
    """one object as parallel range GETs on this process's warm pool, written in place; returns bytes
    received (1.0x the object)."""
    size = head_size(url)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        ex = shared_pool("file", streams)
        futs = [ex.submit(pull_range, url, fd, a, min(size, a + chunk) - 1) for a in range(0, size, chunk)]
        wait(futs, return_when=FIRST_EXCEPTION)
        if any(f.done() and f.exception() for f in futs):
            for f in futs:
                f.cancel()
        wait(futs)  # never close the fd under a running pwrite
        got = sum(f.result() for f in futs)
    finally:
        os.close(fd)
    if got != size:
        raise IOError(f"short download {got}/{size} {url}")
    return got


# ------------------------------------------------------------------------- prefetch of staged ODP objects
def staged_runs(row, prefer=None):
    """ENA runs that make_source would stage from ODP (None when the isolate takes another source)."""
    runs = list(row.get("ena_runs") if row.get("ena_runs") is not None else [])
    if not (prefer in (None, "odp", "odp_stage") and bool(row.get("all_odp")) and runs):
        return None
    flow = row.get("order_hint") == "flowcell"
    d = row.get("mean_depth_est")
    h = row.get("unmapped_byte_frac")
    pf_depth = (d or 0) * (1 - (h if h is not None and np.isfinite(h) else 0.1)) if d is not None and np.isfinite(d) else 0
    if prefer != "odp_stage" and flow and pf_depth > 2.5 * 32:
        return None
    return runs


def claim(path):
    """atomically claim <path>.part; False when someone else is already fetching it."""
    try:
        os.close(os.open(str(path) + ".part", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
        return True
    except FileExistsError:
        return False


def fetch_claimed(url, path, streams=8):
    """download into the claimed <path>.part, then rename to <path> (readers only ever see whole objects)."""
    part = Path(str(path) + ".part")
    try:
        n = ranged_download(url, part, streams=streams)
        os.replace(part, path)
        return n
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def prefetcher_alive(pdir):
    """the cohort's prefetcher (pid in <pdir>/prefetcher.pid) is running (not exited, not a zombie)."""
    try:
        pid = int((Path(pdir) / "prefetcher.pid").read_text())
        st = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return False
    return st.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


class _Job:
    """one staged object fetched as fixed-size ranges on the shared pool into <dst>.part; whichever range
    finishes last closes the file and renames it into place (no polling), or leaves <dst>.err."""

    def __init__(self, run, url, dst, chunk, on_end):
        self.run, self.url, self.dst = run, url, Path(dst)
        self.part = Path(str(dst) + ".part")
        self.size = head_size(url)
        self.fd = os.open(str(self.part), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            os.posix_fallocate(self.fd, 0, self.size)
        except (AttributeError, OSError):
            pass
        self.ranges = [(a, min(self.size, a + chunk) - 1) for a in range(0, self.size, chunk)]
        self.left = len(self.ranges)
        self.failed = None
        self.lock = threading.Lock()
        self.on_end = on_end
        self.t0 = time.time()

    def submit(self, ex):
        for a, b in self.ranges:
            ex.submit(self._one, a, b)

    def _one(self, a, b):
        err = None
        if self.failed is None:
            try:
                pull_range(self.url, self.fd, a, b)
            except Exception as e:
                err = e
        with self.lock:
            if err is not None and self.failed is None:
                self.failed = err
            self.left -= 1
            last = self.left == 0
        if last:
            os.close(self.fd)
            if self.failed is None:
                os.replace(self.part, self.dst)
            else:
                self.part.unlink(missing_ok=True)
                Path(str(self.dst) + ".err").write_text(f"{type(self.failed).__name__}: {self.failed}\n")
            self.on_end(self)


def stage_bytes(root):
    """bytes of staged objects under `root` (prefetched, partial and held by workers; .part files are
    preallocated, so their size is the object size)."""
    tot = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".sra") or f.endswith(".sra.part"):
                try:
                    tot += os.stat(os.path.join(dirpath, f)).st_size
                except OSError:
                    pass
    return tot


def prefetch_loop(runs, pdir, ahead=28, in_flight=6, streams=32, chunk=16 << 20, stop_file=None,
                  url_fmt=ODP_URL, log_path=None, max_bytes=0, stage_root=None):
    """stage ODP objects in the order workers will need them on ONE long-lived pool of `streams` warm
    keep-alive connections: ranges of up to `in_flight` objects are queued FIFO, so the object at the head
    of the queue gets every connection and lands first; at most `ahead` objects (whole or partial) sit in
    pdir, and with `max_bytes` the next object starts only while everything staged under `stage_root`
    (prefetched + held by workers) plus its own size fits (a RAM disk must never fill). Workers consume
    (rename away) objects, which frees a slot."""
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "prefetcher.pid").write_text(str(os.getpid()))
    ex = ThreadPoolExecutor(streams, thread_name_prefix="prefetch")  # lives as long as the prefetcher
    slots = threading.BoundedSemaphore(in_flight)
    jobs = []
    logf = open(log_path, "a") if log_path else None
    loglock = threading.Lock()

    def stopped():
        return stop_file is not None and Path(stop_file).exists()

    def on_end(job):
        slots.release()
        if logf is not None:
            with loglock:
                logf.write(json.dumps(dict(run=job.run, bytes=job.size, s=round(time.time() - job.t0, 2),
                                           ok=job.failed is None, t_end=round(time.time(), 2))) + "\n")
                logf.flush()

    def staged():
        return sum(1 for p in pdir.iterdir() if p.name.endswith(".sra") or p.name.endswith(".sra.part"))

    for run in runs:
        dst = pdir / f"{run}.sra"
        if dst.exists() or (pdir / "taken" / run).exists():
            continue
        while staged() >= ahead and not stopped():
            time.sleep(0.05)
        while not slots.acquire(timeout=0.5):
            if stopped():
                break
        if stopped():
            break
        if (pdir / "taken" / run).exists() or not claim(dst):
            slots.release()
            continue
        if max_bytes:
            try:
                need = head_size(url_fmt.format(run))
            except Exception:
                need = 0
            root = stage_root or pdir
            while not stopped():  # an object larger than the cap still goes once nothing else is staged
                b = stage_bytes(root)
                if b == 0 or b + need <= max_bytes:
                    break
                time.sleep(0.05)
            if stopped():
                Path(str(dst) + ".part").unlink(missing_ok=True)
                break
        try:
            job = _Job(run, url_fmt.format(run), dst, chunk, on_end)
        except Exception as e:  # HEAD failed: the worker fetches this one itself
            Path(str(dst) + ".part").unlink(missing_ok=True)
            Path(str(dst) + ".err").write_text(f"{type(e).__name__}: {e}\n")
            slots.release()
            continue
        jobs.append(job)
        job.submit(ex)
    while not stopped() and any(j.left > 0 for j in jobs):
        time.sleep(0.2)
    ex.shutdown(wait=not stopped(), cancel_futures=stopped())


def preread(path, chunk=16 << 20):
    """sequential read so the following random-access decode is served from the page cache."""
    with open(path, "rb", buffering=0) as f:
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass
        while f.read(chunk):
            pass


class OdpStaged(Source):
    """coordinate-ordered ODP archives must be read end to end for a whole genome: fetch the normalised
    full-quality SRA object once with parallel ranges (1.0x wire bytes, xsra over HTTPS pulls ~2.3x),
    decode it locally with xsra, delete it as soon as the run is decoded."""
    kind = "aws_odp_staged"

    def __init__(self, runs, workdir, threads=4, streams=8, log=None, decode_sem=None, preread=False,
                 prefetch_dir=None):
        super().__init__()
        self.decode_sem = decode_sem
        self.preread = preread
        self.prefetch_dir = Path(prefetch_dir) if prefetch_dir else None
        self.prefetched = 0
        self.direct_fetches = 0
        self.holding = False
        self.t_decode_wait = 0.0
        self.runs = list(runs)
        self.workdir = Path(workdir)
        self.threads, self.streams = threads, streams
        self.log = log or subprocess.DEVNULL
        self.proc = None
        self.ri = -1
        self.got = 0
        self.peak = 0
        self.cur = None
        self.t_download = 0.0

    def _release(self):
        if self.holding and self.decode_sem is not None:
            self.decode_sem.release()
        self.holding = False

    def _next(self):
        if self._closed_early:
            return False
        if self.proc is not None:
            self._check_proc(self.proc, f"xsra dump {self.runs[self.ri]}")
            self.proc = None
        self._release()
        if self.cur is not None and self.cur.exists():
            self.cur.unlink()
        self.ri += 1
        if self.ri >= len(self.runs):
            return False
        self.workdir.mkdir(parents=True, exist_ok=True)
        run = self.runs[self.ri]
        self.cur = self.workdir / f"{run}.sra"
        t0 = time.time()
        n = self._obtain(run)
        self.t_download += time.time() - t0
        self.got += n
        self.peak = max(self.peak, n)
        # VDB reads column blobs at random: on NVMe staging no decode cap or pre-read is needed. On a spinning
        # disk pass decode_sem (cap concurrent decodes) and preread=True (pull the object into the page cache)
        t1 = time.time()
        if self.decode_sem is not None:
            self.decode_sem.acquire()
            self.holding = True
        self.t_decode_wait += time.time() - t1
        if self.preread:
            preread(self.cur)
        self.proc = subprocess.Popen([XSRA, "dump", str(self.cur), "--skip-technical", "-T", str(self.threads)],
                                     stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        return True

    def _direct(self, run):
        self.direct_fetches += 1
        return ranged_download(ODP_URL.format(run), self.cur, streams=self.streams)

    def _obtain(self, run):
        """object for `run` at self.cur. With a prefetcher the worker only waits (0.05 s poll) and takes the
        object by rename: it never opens competing S3 streams of its own (14 workers x 8 streams beside the
        prefetcher thrashed the link). It fetches directly only when the prefetcher gave up on the object
        (<run>.sra.err), is gone, or already handed this run out once (a retry)."""
        if self.prefetch_dir is None:
            return self._direct(run)
        pre = self.prefetch_dir / f"{run}.sra"
        err = Path(str(pre) + ".err")
        taken = self.prefetch_dir / "taken" / run
        if taken.exists():
            return self._direct(run)
        t_wait = time.time()
        while not pre.exists():
            if err.exists() or not prefetcher_alive(self.prefetch_dir):
                if pre.exists():
                    break
                taken.parent.mkdir(exist_ok=True)
                taken.touch()
                return self._direct(run)
            if time.time() - t_wait > 3600:
                raise IOError(f"prefetch of {run} stalled")
            time.sleep(0.05)
        self.prefetched += 1
        n = pre.stat().st_size
        taken.parent.mkdir(exist_ok=True)
        taken.touch()  # the prefetcher must not fetch it again
        os.replace(pre, self.cur)
        return n

    def read(self, n):
        while not self._closed_early:
            proc = self.proc
            if proc is None:
                if not self._next():
                    return b""
                proc = self.proc
                if proc is None:
                    return b""
            try:
                b = proc.stdout.read(n)
            except (ValueError, OSError):
                return b""
            if b:
                return b
            if self._closed_early or not self._next():
                return b""
        return b""

    def wire_bytes(self):
        return self.got

    def close(self, early=False):
        with self._close_lock:
            super().close(early=early)
            proc = self.proc
            self.proc = None
            ri = self.ri
            if proc is not None:
                label = f"xsra dump {self.runs[ri]}" if 0 <= ri < len(self.runs) else "xsra dump"
                self._terminate_proc(proc, label)
            self._release()
            if self.cur is not None and self.cur.exists():
                self.cur.unlink()
            try:
                self.workdir.rmdir()
            except OSError:
                pass


class _CountedPipe(io.RawIOBase):
    def __init__(self, raw, on_bytes):
        self._raw = raw
        self._on_bytes = on_bytes

    def readable(self):
        return True

    def readinto(self, b):
        data = self._raw.read(len(b))
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        self._on_bytes(n)
        return n


def _qname_stem(hdr):
    tok = hdr[1:].rstrip(b"\r\n").split(None, 1)[0]
    if len(tok) > 2 and tok[-2:] in (b"/1", b"/2", b".1", b".2", b".0"):
        return tok[:-2]
    return tok


class EnaFastq(Source):
    kind = "ena_fastq"

    def __init__(self, url_pairs, log=None):
        super().__init__()
        self.pairs = list(url_pairs)
        self.log = log or subprocess.DEVNULL
        self.procs = []
        self._gz = None
        self._buf = bytearray()
        self.got = 0
        self.i = -1
        self._next()

    def _add_bytes(self, n):
        self.got += n

    def _finish_pair(self):
        if self._gz is not None:
            for gz in self._gz:
                try:
                    gz.close()
                except Exception:
                    pass
            self._gz = None
        if self.procs:
            r1, r2 = self.pairs[self.i]
            for p, u in zip(self.procs, (r1, r2)):
                if p.poll() is None and self._closed_early:
                    p.terminate()
                    try:
                        p.wait(5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                else:
                    self._check_proc(p, f"curl {u}")
            self.procs = []

    def _next(self):
        self._finish_pair()
        self.i += 1
        if self.i >= len(self.pairs) or self.errors:
            return False
        r1, r2 = self.pairs[self.i]
        p1 = subprocess.Popen(["curl", "-fsSL", "--retry", "5", str(r1)], stdout=subprocess.PIPE,
                              stderr=self.log, bufsize=0)
        p2 = subprocess.Popen(["curl", "-fsSL", "--retry", "5", str(r2)], stdout=subprocess.PIPE,
                              stderr=self.log, bufsize=0)
        self.procs = [p1, p2]
        self._gz = (gzip.GzipFile(fileobj=io.BufferedReader(_CountedPipe(p1.stdout, self._add_bytes), 65536)),
                    gzip.GzipFile(fileobj=io.BufferedReader(_CountedPipe(p2.stdout, self._add_bytes), 65536)))
        return True

    def _read_rec(self, gz):
        h = gz.readline()
        if not h:
            return None
        s = gz.readline()
        p = gz.readline()
        q = gz.readline()
        if not (s and p and q):
            raise EOFError("truncated 4-line FASTQ record")
        if not h.startswith(b"@") or not p.startswith(b"+"):
            raise ValueError("invalid FASTQ record framing")
        if len(s.rstrip(b"\r\n")) != len(q.rstrip(b"\r\n")):
            raise ValueError("FASTQ sequence/quality length mismatch")
        return h, s, p, q

    def read(self, n):
        while len(self._buf) < n and self._gz is not None:
            try:
                rec1 = self._read_rec(self._gz[0])
                rec2 = self._read_rec(self._gz[1])
            except (EOFError, OSError, ValueError) as e:
                self.errors.append(f"EnaFastq pair {self.i}: {type(e).__name__}: {e}")
                self._closed_early = True
                self._finish_pair()
                break
            if rec1 is None and rec2 is None:
                if not self._next():
                    break
                continue
            if (rec1 is None) != (rec2 is None):
                self.errors.append(f"EnaFastq pair {self.i}: premature EOF on {'R1' if rec1 is None else 'R2'}")
                self._closed_early = True
                self._finish_pair()
                break
            if _qname_stem(rec1[0]) != _qname_stem(rec2[0]):
                self.errors.append(
                    f"EnaFastq pair {self.i}: mate QNAME mismatch ({_qname_stem(rec1[0])!r} vs {_qname_stem(rec2[0])!r})")
                self._closed_early = True
                self._finish_pair()
                break
            for line in (*rec1, *rec2):
                self._buf.extend(line if line.endswith(b"\n") else line + b"\n")
        if not self._buf:
            return b""
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def wire_bytes(self):
        return self.got

    def close(self, early=False):
        super().close(early=early)
        if self.procs:
            self._closed_early = True
        self._finish_pair()


def crai_nuclear_end(crai, n_nuclear=N_NUCLEAR):
    """byte offset of the first container after the last container holding a nuclear slice
    (seq_id 0..n_nuclear-1): the CRAM prefix up to it holds every nuclear slice whole, and only the
    apicoplast, mitochondrion and unmapped containers follow. None when nothing follows.
    (container_offset + slice_offset + slice_size stops short by the container header: the slice offset
    counts from the end of the header.)"""
    raw = Path(crai).read_bytes()
    txt = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    a = np.loadtxt(io.StringIO(txt), dtype=np.int64, ndmin=2)
    if not len(a):
        return None
    nuc = (a[:, 0] >= 0) & (a[:, 0] < n_nuclear)
    if not nuc.any():
        return None
    after = a[a[:, 3] > a[nuc, 3].max(), 3]
    return int(after.min()) if len(after) else None


class SangerCram(Source):
    """Sanger CRAM through parallel ranged GETs (written in order) into `samtools collate | samtools fastq`,
    preserving paired-end mate adjacency for insert-size, overlap-dedup, and face-away CNV junction detection.
    With the .crai, only the nuclear prefix is fetched and a CRAM EOF container closes the stream.
    The server gives each connection ~0.05 MB/s and scales with connections (16: 0.9, 48: 2.0, 96: 3.1 MB/s),
    so the stream count, not the handshake, sets the speed; sessions are kept alive anyway."""
    kind = "sanger_cram"
    paired = True

    def __init__(self, url, size, streams=48, chunk=8 << 20, log=None, crai=None, ref=None, paired=True, tries=8):
        super().__init__()
        self.paired = bool(paired)
        self.tries = int(tries)
        self.url = url
        self.full_size = int(size)
        end = None
        if crai is not None and Path(crai).exists():
            try:
                end = crai_nuclear_end(crai)
            except Exception:
                end = None
        self.trim = end is not None and 0 < end < self.full_size
        self.size = end if self.trim else self.full_size
        self.trimmed_bytes = self.full_size - self.size
        self.chunk = chunk
        self.ahead = 2 * streams
        self.got = 0
        self.log = log or subprocess.DEVNULL
        tmp_base = os.environ.get("TMPDIR", "/tmp")
        Path(tmp_base).mkdir(parents=True, exist_ok=True)
        self._tmpdir = Path(tempfile.mkdtemp(prefix="sanger_cram_", dir=tmp_base))
        ref_q = shlex.quote(str(ref or G.REF_FASTA))
        coll_q = shlex.quote(str(self._tmpdir / "coll"))
        cmd = (f"samtools collate -u -O -r 50000 --reference {ref_q} - {coll_q} | "
               f"samtools fastq -F 0x900 -s /dev/null -N -")
        self.proc = subprocess.Popen(
            ["bash", "-o", "pipefail", "-c", cmd],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self._waited = False
        self.pool = shared_pool("sanger", streams)
        self.futs = {}
        self.alive = True
        self.feeder = threading.Thread(target=self._feed, daemon=True)
        self.feeder.start()

    def _get(self, a, b):
        for t in range(self.tries):
            if not self.alive:
                return b""
            try:
                with _session().get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=(30, 300),
                                    stream=True) as r:
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise IOError(f"range ignored (HTTP {r.status_code})")
                    data = r.raw.read()
                if len(data) != b - a:
                    raise IOError(f"short range {len(data)} != {b - a}")
                return data
            except Exception:
                if t == self.tries - 1:
                    raise
                time.sleep(min(30, 3 * (t + 1)))

    def _feed(self):
        try:
            starts = list(range(0, self.size, self.chunk))
            nxt = 0
            for i in range(min(self.ahead, len(starts))):
                self.futs[i] = self.pool.submit(self._get, starts[i], min(self.size, starts[i] + self.chunk))
            submitted = min(self.ahead, len(starts))
            while nxt < len(starts) and self.alive:
                data = self.futs.pop(nxt).result()
                self.got += len(data)
                self.proc.stdin.write(data)
                nxt += 1
                if submitted < len(starts):
                    self.futs[submitted] = self.pool.submit(self._get, starts[submitted],
                                                            min(self.size, starts[submitted] + self.chunk))
                    submitted += 1
            if self.trim and nxt == len(starts) and self.alive:
                self.proc.stdin.write(CRAM_EOF)  # nuclear prefix ends on a container boundary
        except (BrokenPipeError, ValueError) as e:
            if self.alive and not self._closed_early:
                self.errors.append(f"SangerCram pipe closed mid-stream: {type(e).__name__}")
        except Exception as e:
            if self.alive and not self._closed_early:
                self.errors.append(f"SangerCram feeder: {type(e).__name__}: {e}")
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def read(self, n):
        b = self.proc.stdout.read(n)
        if not b and not self._waited:
            self._waited = True
            self.feeder.join(timeout=10)
            self._check_proc(self.proc, "SangerCram samtools collate|fastq")
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        return b

    def wire_bytes(self):
        return self.got

    def close(self, early=False):
        super().close(early=early)
        self.alive = False
        for f in list(self.futs.values()):
            f.cancel()
        if self.proc.poll() is None:
            self._closed_early = True
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        elif not self._waited:
            self._check_proc(self.proc, "SangerCram samtools collate|fastq")
        self._waited = True
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def make_source(row, xsra_threads=2, log=None, prefer=None, stage_dir=None, decode_sem=None, prefetch_dir=None):
    """ODP first (flowcell-order archives streamed with xsra -Q -P https so the stop gate can cut them short,
    coordinate-ordered archives staged once and decoded locally); ENA FASTQ only without ODP; Sanger CRAM
    (nuclear prefix by its .crai) when the sample has no usable ENA run."""
    runs = list(row.get("ena_runs") if row.get("ena_runs") is not None else [])
    if prefer in (None, "odp", "odp_stream", "odp_stage") and bool(row.get("all_odp")) and runs:
        flow = row.get("order_hint") == "flowcell"
        d = row.get("mean_depth_est")
        h = row.get("unmapped_byte_frac")
        pf_depth = (d or 0) * (1 - (h if h is not None and np.isfinite(h) else 0.1)) if d is not None and np.isfinite(d) else 0
        # xsra over HTTPS pulls ~2.3x the object, so streaming only pays when the gate can stop before ~40%
        worth = pf_depth > 2.5 * 32
        if prefer == "odp_stream" or (prefer != "odp_stage" and flow and worth):
            return OdpXsra(runs, threads=xsra_threads, log=log)
        # per-process dir: two arms (or reruns) staging the same isolate never share a file
        return OdpStaged(runs, Path(stage_dir) / f"{row['Sample']}.{os.getpid()}", threads=xsra_threads, log=log,
                         decode_sem=decode_sem, prefetch_dir=prefetch_dir)
    if prefer in (None, "ena") and bool(row.get("all_paired_fastq")) and row.get("fastq_urls") is not None:
        urls = [u for u in list(row["fastq_urls"]) if u]
        pairs = []
        for u in urls:
            parts = [("https://" + x if not x.startswith("http") else x) for x in str(u).split(";")]
            r1 = [x for x in parts if x.endswith("_1.fastq.gz")]
            r2 = [x for x in parts if x.endswith("_2.fastq.gz")]
            if r1 and r2:
                pairs.append((r1[0], r2[0]))
        if pairs:
            return EnaFastq(pairs, log=log)
    crai = CRAI_DIR / f"{row['Sample']}.cram.crai"
    cb = row.get("crai_bytes")
    if not crai.exists() or (cb is not None and np.isfinite(cb) and crai.stat().st_size != int(cb)):
        crai = None
    return SangerCram(row["cram_url"], row["cram_bytes"], log=log, crai=crai)


class Sieve:
    """host sieve + spot subsampler over a FASTQ byte stream (wraps kernels.fastq_block or PLA-index sieve)."""

    def __init__(self, kset, paired, stride=8, need=2, trim=True, min_len=30, use_pla_sieve=False):
        self.ad = Kn.ADAPTER if trim else np.zeros(0, np.uint8)
        self.min_len = min_len
        self.kset = kset
        self.use_pla_sieve = bool(use_pla_sieve)
        self.paired = 1 if paired else 0
        self.stride, self.need = stride, need
        self.stats = np.zeros(10, np.int64)
        self.stats[8] = -1
        self.qhist = np.zeros(128, np.int64)
        self.carry = b""
        self.spot = 0
        self.p = 1.0

    def keep_thr(self):
        return np.uint64(min(int(self.p * (1 << 32)), (1 << 32)))

    def push(self, data, final=False, max_spots=0):
        buf = self.carry + data
        if final and buf and not buf.endswith(b"\n"):
            buf += b"\n"
        arr = np.frombuffer(buf, np.uint8)
        out = np.empty(2 * len(arr) + 4096, np.uint8)
        if self.use_pla_sieve:
            used, o, self.spot = Kn.pla_fastq_block(
                arr, len(arr), out, self.spot, int(self.keep_thr()), self.paired,
                self.kset, self.need, self.stats, self.qhist, self.ad, self.min_len, int(max_spots),
            )
        else:
            used, o, self.spot = Kn.fastq_block(arr, len(arr), out, self.spot, self.keep_thr(), self.paired,
                                                self.kset, self.stride, self.need, self.stats, self.qhist,
                                                self.ad, self.min_len, np.int64(max_spots))
        self.carry = buf[used:]
        return out[:o].tobytes()

    def quality_check(self):
        """real Phred qualities: >= 3 distinct symbols in the first block. SRA-lite is one flat value ('?', with '#'
        or '$' for filtered reads); NovaSeq bins real qualities into exactly three (',', ':', 'F')."""
        q = np.nonzero(self.qhist[33:127])[0]
        sym = [chr(33 + x) for x in q[:12]]
        flat = set(sym) <= {"?", "#", "$"}
        return (len(q) >= 3 and not flat), sym


def mapper_cmd(threads, paired, ref=None, mapper="strobealign"):
    ref = str(ref or G.REF_FASTA)
    t = str(threads)
    if mapper == "strobealign":
        cmd = ["strobealign", "-t", t]
        if paired:
            cmd.append("--interleaved")
        return cmd + [ref, "-"]
    if mapper in ("bwa-mem2", "bwa"):
        cmd = [mapper, "mem", "-t", t]
        if paired:
            cmd.append("-p")
        return cmd + [ref, "-"]
    if mapper == "minimap2":
        return ["minimap2", "-t", t, "-ax", "sr", ref, "-"]
    raise ValueError(f"unknown mapper: {mapper}")


if __name__ == "__main__":
    # cohort prefetcher: python streamer.py prefetch <queue.json> <dir> <ahead> <in_flight> [streams]
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "prefetch":
        q = json.loads(Path(sys.argv[2]).read_text())
        prefetch_loop(q["runs"], sys.argv[3], ahead=int(sys.argv[4]) if len(sys.argv) > 4 else 28,
                      in_flight=int(sys.argv[5]) if len(sys.argv) > 5 else 6,
                      streams=int(sys.argv[6]) if len(sys.argv) > 6 else 32,
                      max_bytes=int(sys.argv[7]) if len(sys.argv) > 7 else 0,
                      stage_root=sys.argv[8] if len(sys.argv) > 8 else None,
                      stop_file=q.get("stop_file"), log_path=q.get("log_path"))
