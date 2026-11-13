"""Cross-platform on-demand C99 shared-library compiler for Medvedev extensions.

Checks pre-compiled container cache (``FOCALOOP_NATIVE_CACHE`` / ``/opt/focaloop/native_cache``)
first so read-only Apptainer ``.sif`` images load immediately with zero runtime compiler calls,
falling back to a user-scoped temp cache directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

_LOCK = threading.Lock()
_LOADED: dict[str, ctypes.CDLL] = {}


def find_c_compiler() -> str | None:
    """Return a working C compiler executable path, or None."""
    env_cc = os.environ.get("CC")
    candidates = [env_cc] if env_cc else []
    candidates.extend(["clang", "gcc", "cc"])
    for cand in candidates:
        if cand:
            exe = shutil.which(cand)
            if exe:
                return exe
    return None


def _sanitize_isa_flags(flags: list[str] | None) -> list[str]:
    out = [f for f in (flags or []) if not f.startswith("-march=")]
    arch = platform.machine().lower()
    if os.environ.get("FOCALOOP_PORTABLE_ISA") == "1":
        if arch in ("x86_64", "amd64"):
            out.extend(["-msse4.2", "-mpopcnt"])
    elif flags:
        return list(flags)
    return out


def compile_c_lib(name: str, c_path: str | Path, extra_flags: list[str] | None = None) -> ctypes.CDLL:
    """Compile (if not cached by SHA-256) and load a C99 shared library cleanly."""
    src = Path(c_path)
    src_bytes = src.read_bytes()
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    key = f"{name}:{digest}"
    with _LOCK:
        cached = _LOADED.get(key)
        if cached is not None:
            return cached
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib_name = f"lib{name}_{digest}{ext}"
        for pre_dir in (
            os.environ.get("FOCALOOP_NATIVE_CACHE"),
            "/opt/focaloop/native_cache",
            os.environ.get("PANEXPR_CACHE_DIR"),
        ):
            if pre_dir:
                cand = Path(pre_dir) / lib_name
                if cand.exists():
                    lib = ctypes.CDLL(str(cand))
                    _LOADED[key] = lib
                    return lib
        cc = find_c_compiler()
        if not cc:
            raise RuntimeError(f"No C compiler found to build {src.name}")
        target_dir_env = os.environ.get("FOCALOOP_BUILD_NATIVE_DIR")
        if target_dir_env:
            cache_dir = Path(target_dir_env)
        else:
            cache_dir = Path(os.environ.get("PANEXPR_CACHE_DIR") or tempfile.gettempdir()) / f"panexpr_medvedev_{os.getuid()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / lib_name
        if not out_path.exists():
            tmp_path = cache_dir / f"lib{name}_{digest}_{os.getpid()}{ext}"
            flags = _sanitize_isa_flags(extra_flags)
            cmd = [cc, "-O3", "-fPIC", "-shared", *flags, str(src), "-lm", "-o", str(tmp_path)]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                os.replace(tmp_path, out_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        lib = ctypes.CDLL(str(out_path))
        _LOADED[key] = lib
        return lib
