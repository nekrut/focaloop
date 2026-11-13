"""Native C shared library compiler and loader for MalariaVar upgrades.

Supports pre-compiled libraries inside read-only Apptainer ``.sif`` images
(``/opt/focaloop/native_cache`` or ``NATIVE_DIR``), portable x86-64 ISA builds
(``FOCALOOP_PORTABLE_ISA=1`` -> ``-msse4.2 -mpopcnt`` for Ivy Bridge compatibility),
and automatic ``/tmp`` fallback when ``NATIVE_DIR`` is read-only.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

NATIVE_DIR = Path(__file__).resolve().parent


def _lib_ext() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _compiler() -> str:
    for cand in (os.environ.get("CC"), "gcc", "clang", "cc"):
        if cand and shutil.which(cand):
            return cand
    raise RuntimeError("No C compiler (gcc/clang/cc) found in PATH")


def ensure_shared_lib(
    lib_stem: str,
    sources: list[str],
    headers: list[str] | None = None,
    use_openmp: bool = True,
) -> Path:
    """Compile a shared library if missing or stale, returning its path."""
    ext = _lib_ext()
    arch = platform.machine().lower()
    fname = f"{lib_stem}_{sys.platform}_{arch}{ext}"

    for pre_dir in (os.environ.get("FOCALOOP_NATIVE_CACHE"), "/opt/focaloop/native_cache"):
        if pre_dir:
            pre_cand = Path(pre_dir) / fname
            if pre_cand.exists():
                return pre_cand

    src_paths = [NATIVE_DIR / s for s in sources]
    dep_paths = src_paths + [NATIVE_DIR / h for h in (headers or [])] + [Path(__file__).resolve()]
    newest_src = max(p.stat().st_mtime for p in dep_paths if p.exists())

    in_tree = NATIVE_DIR / fname
    if in_tree.exists():
        if not os.access(NATIVE_DIR, os.W_OK) or in_tree.stat().st_mtime >= newest_src:
            return in_tree

    build_dir_env = os.environ.get("FOCALOOP_BUILD_NATIVE_DIR")
    if build_dir_env:
        out_dir = Path(build_dir_env)
    elif os.access(NATIVE_DIR, os.W_OK):
        out_dir = NATIVE_DIR
    else:
        out_dir = Path(tempfile.gettempdir()) / f"focaloop_malariavar_native_{os.getuid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / fname
    if target.exists() and target.stat().st_mtime >= newest_src:
        return target

    cc = _compiler()
    base_cmd = [cc, "-O3", "-shared", "-fPIC", "-Wall", "-Wno-unused-function", "-I", str(NATIVE_DIR)]
    if os.environ.get("FOCALOOP_PORTABLE_ISA") == "1":
        if arch in ("x86_64", "amd64"):
            base_cmd.extend(["-msse4.2", "-mpopcnt"])
    elif arch in ("x86_64", "amd64", "arm64", "aarch64"):
        base_cmd.append("-march=native")

    omp_flags = ["-fopenmp"] if (use_openmp and sys.platform != "darwin") else []
    tmp_out = out_dir / f".tmp_{lib_stem}_{os.getpid()}{ext}"
    attempts = [omp_flags, []] if omp_flags else [[]]
    last_err = ""
    try:
        for extra in attempts:
            cmd = base_cmd + extra + [str(p) for p in src_paths] + ["-lm", "-o", str(tmp_out)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and tmp_out.exists():
                os.replace(tmp_out, target)
                return target
            last_err = res.stderr
    finally:
        tmp_out.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to compile {target.name} with {cc}: {last_err}")


def load_shared_lib(
    lib_stem: str,
    sources: list[str],
    headers: list[str] | None = None,
    use_openmp: bool = True,
) -> ctypes.CDLL:
    """Ensure shared library is compiled cleanly for current platform and open via ctypes."""
    path = ensure_shared_lib(lib_stem, sources, headers=headers, use_openmp=use_openmp)
    return ctypes.CDLL(str(path))
