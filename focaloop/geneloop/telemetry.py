"""Systems telemetry: per-process CPU (wait4 rusage), GPU sampling, NIC wire bytes, peak scratch.

Every external tool GeneLoop launches goes through :func:`run_pipeline` so CPU seconds are exact
per process tree (``os.wait4`` rusage covers the child and every descendant it reaped), never
estimated from wall-clock.  Docker containers are measured from inside the container (the
docker CLI's own rusage does not include the containerised process).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _default_scratch_root() -> Path:
    env_sc = os.environ.get("GENELOOP_SCRATCH")
    if env_sc:
        return Path(env_sc)
    return Path(tempfile.gettempdir()) / "geneloop_runs"


def _default_tmp_root() -> Path:
    env_tmp = os.environ.get("TMPDIR")
    if env_tmp:
        return Path(env_tmp)
    env_sc = os.environ.get("GENELOOP_SCRATCH")
    if env_sc:
        return Path(env_sc) / "tmp"
    return Path(tempfile.gettempdir()) / "geneloop_tmp"


SCRATCH_ROOT = _default_scratch_root()
TMP_ROOT = _default_tmp_root()


def safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for every child with writable TMPDIR and optional GENELOOP_TOOL_BIN prefix."""
    env = dict(os.environ)
    eff_tmp = _default_tmp_root()
    try:
        eff_tmp.mkdir(parents=True, exist_ok=True)
    except OSError:
        eff_tmp = Path(tempfile.gettempdir()) / "geneloop_tmp"
        eff_tmp.mkdir(parents=True, exist_ok=True)
    tmp_s = str(eff_tmp)
    env["TMPDIR"] = tmp_s
    env["TMP"] = tmp_s
    env["TEMP"] = tmp_s
    tool_bin = os.environ.get("GENELOOP_TOOL_BIN", "").strip()
    if tool_bin and Path(tool_bin).is_dir() and tool_bin not in env.get("PATH", "").split(":"):
        env["PATH"] = tool_bin + ":" + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


def assert_not_root_disk(path: str | Path) -> None:
    """Optional scratch guard when GENELOOP_ENFORCE_SCRATCH_ROOT=1."""
    if os.environ.get("GENELOOP_ENFORCE_SCRATCH_ROOT") != "1":
        return
    p = Path(path).resolve()
    s = str(p)
    allowed = ("/dev/shm/",)
    scratch_prefix = str(SCRATCH_ROOT.resolve())
    if (
        not s.startswith(allowed)
        and not s.startswith(str(TMP_ROOT))
        and not (s == scratch_prefix or s.startswith(scratch_prefix + "/"))
    ):
        raise RuntimeError(f"Refusing to write scratch outside allowed scratch roots: {s}")


@dataclass
class StageRecord:
    name: str
    wall_seconds: float = 0.0
    cpu_user: float = 0.0
    cpu_sys: float = 0.0
    gpu_seconds: float = 0.0
    gpu_util_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    wire_bytes: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def cpu_seconds(self) -> float:
        return self.cpu_user + self.cpu_sys

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cpu_seconds"] = self.cpu_seconds
        return d


class StageLog:
    """Append-only JSON-lines log of stage records for one gate/arm working directory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, rec: StageRecord) -> None:
        line = json.dumps(rec.to_dict()) + "\n"
        with self._lock:
            with open(self.path, "a") as fh:
                fh.write(line)

    def load(self) -> list[dict]:
        with self._lock:
            if not self.path.exists():
                return []
            text = self.path.read_text()
        out = []
        for line in text.splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def latest(self, name: str) -> dict | None:
        recs = [r for r in self.load() if r["name"] == name]
        return recs[-1] if recs else None


def wait_rusage(p: subprocess.Popen) -> tuple[int, float, float]:
    """Reap ``p`` with wait4 and return (returncode, user_s, sys_s) for its whole process tree."""
    while True:
        try:
            _, status, ru = os.wait4(p.pid, 0)
            break
        except InterruptedError:
            continue
        except ChildProcessError:
            return (p.returncode if p.returncode is not None else 0), 0.0, 0.0
    if os.WIFEXITED(status):
        rc = os.WEXITSTATUS(status)
    elif os.WIFSIGNALED(status):
        rc = -os.WTERMSIG(status)
    else:
        rc = 0
    p.returncode = rc
    return rc, ru.ru_utime, ru.ru_stime


def run_cmd(
    cmd: list[str] | str,
    *,
    stdout=None,
    stdin=None,
    stderr=None,
    cwd: str | Path | None = None,
    env: dict | None = None,
    check: bool = True,
) -> tuple[int, float, float, float]:
    """Run one command; return (rc, wall, user, sys)."""
    shell = isinstance(cmd, str)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=shell, stdout=stdout, stdin=stdin, stderr=stderr, cwd=cwd, env=env or safe_env())
    rc, u, s = wait_rusage(p)
    wall = time.time() - t0
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd if shell else shlex.join(cmd))
    return rc, wall, u, s


class NicCounter:
    """Host NIC rx-byte counter (default-route interface)."""

    def __init__(self) -> None:
        self.iface = self._default_iface()

    @staticmethod
    def _default_iface() -> str:
        try:
            out = subprocess.run(["ip", "route", "get", "8.8.8.8"], capture_output=True, text=True).stdout.split()
            return out[out.index("dev") + 1]
        except Exception:
            if os.path.isdir("/sys/class/net"):
                for cand in sorted(os.listdir("/sys/class/net")):
                    if cand != "lo":
                        return cand
            return "lo"

    def rx(self) -> int:
        try:
            return int(Path(f"/sys/class/net/{self.iface}/statistics/rx_bytes").read_text())
        except Exception:
            return 0


class GpuSampler:
    """Background nvidia-smi sampler (memory.used MB, utilization %)."""

    def __init__(self, out_path: Path, interval_ms: int = 500):
        self.out_path = Path(out_path)
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.baseline_mb = 0.0

    def __enter__(self) -> "GpuSampler":
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            )
            self.baseline_mb = float(q.stdout.strip().splitlines()[0])
        except Exception:
            self.baseline_mb = 0.0
        self.fh = open(self.out_path, "w")
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=self.fh,
            stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self.fh.close()

    def summary(self) -> dict:
        peak = 0.0
        util_s = 0.0
        n = 0
        dt = self.interval_ms / 1000.0
        if self.out_path.exists():
            for line in self.out_path.read_text().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    mem = float(parts[1])
                    util = float(parts[2])
                except ValueError:
                    continue
                n += 1
                peak = max(peak, mem)
                util_s += util / 100.0 * dt
        return {
            "peak_vram_mb": peak,
            "peak_vram_mb_above_baseline": max(0.0, peak - self.baseline_mb),
            "baseline_vram_mb": self.baseline_mb,
            "gpu_util_seconds": util_s,
            "samples": n,
        }


def dir_bytes(path: Path) -> int:
    """Allocated bytes under ``path`` (st_blocks, follows nothing)."""
    total = 0
    stack = [Path(path)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_blocks * 512
                    except FileNotFoundError:
                        continue
        except (FileNotFoundError, NotADirectoryError):
            continue
    return total


class ScratchMonitor:
    """Samples allocated bytes of a directory tree every ``interval`` seconds; tracks the peak."""

    def __init__(self, path: Path, interval: float = 5.0):
        self.path = Path(path)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def sample(self) -> int:
        b = dir_bytes(self.path)
        self.peak = max(self.peak, b)
        return b

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "ScratchMonitor":
        self.path.mkdir(parents=True, exist_ok=True)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=30)
        self.sample()
