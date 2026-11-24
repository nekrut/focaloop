"""Cross-platform on-demand C99 shared-library compiler for Medvedev extensions.

Checks pre-compiled container/prefix cache (``FOCALOOP_NATIVE_CACHE``,
``sys.prefix/share/focaloop/native_cache``, ``/opt/focaloop/native_cache``) first
so read-only environments load immediately with zero runtime compiler calls,
falling back to an owner-verified ``0700`` user-scoped cache directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import stat
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
    if os.environ.get("FOCALOOP_PORTABLE_ISA", "1") != "0":
        if sys.platform != "darwin" and arch in ("x86_64", "amd64"):
            out.extend(["-msse4.2", "-mpopcnt"])
    elif flags:
        return list(flags)
    return out


def prebuilt_native_dirs() -> tuple[Path, ...]:
    """Return trusted prebuilt native cache directories in lookup priority order."""
    out: list[Path] = []
    for raw in (
        os.environ.get("FOCALOOP_NATIVE_CACHE"),
        str(Path(sys.prefix) / "share" / "focaloop" / "native_cache"),
        "/opt/focaloop/native_cache",
    ):
        if raw:
            p = Path(raw)
            if p not in out:
                out.append(p)
    return tuple(out)


def verify_native_artifact(path: Path, *, require_current_uid: bool) -> bool:
    """Verify a cached shared library or binary is a non-symlink regular file with safe ownership/mode."""
    try:
        lst = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(lst.st_mode):
        if require_current_uid:
            return False
        try:
            st = os.stat(path)
        except OSError:
            return False
    else:
        st = lst
    if not stat.S_ISREG(st.st_mode):
        return False
    uid = os.getuid()
    if require_current_uid:
        if st.st_uid != uid:
            return False
        if (st.st_mode & 0o022) != 0:
            return False
    else:
        # Trusted prebuilt/prefix location: allow current user or root, never world-writable
        if st.st_uid not in (uid, 0):
            return False
        if (st.st_mode & 0o002) != 0:
            return False
    return True


def ensure_user_native_cache_dir(
    explicit_build_dir: str | Path | None = None,
    explicit_cache_root: str | Path | None = None,
    subdir_stem: str = "focaloop_panexpr_native",
) -> tuple[Path, bool]:
    """Return ``(cache_dir, is_build_dir)`` enforcing ``0700`` and ``st_uid == getuid()`` on user caches."""
    if explicit_build_dir:
        bdir = Path(explicit_build_dir)
        if os.path.islink(bdir):
            raise RuntimeError(f"Refusing symlink native build directory {bdir}")
        bdir.mkdir(parents=True, exist_ok=True)
        st = os.lstat(bdir)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise RuntimeError(f"Unsafe ownership on native build directory {bdir}")
        return bdir, True

    if explicit_cache_root:
        root = Path(explicit_cache_root)
        cache_dir = root / f"{subdir_stem}_{os.getuid()}"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir())
        cache_dir = base / f"{subdir_stem}_{os.getuid()}"

    if os.path.islink(cache_dir):
        raise RuntimeError(f"Refusing symlink native cache directory {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = os.lstat(cache_dir)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"Native cache path {cache_dir} is not a regular directory")
    if st.st_uid != os.getuid():
        raise RuntimeError(
            f"Refusing native cache directory {cache_dir} owned by uid {st.st_uid} (expected {os.getuid()})"
        )
    if (st.st_mode & 0o077) != 0:
        os.chmod(cache_dir, 0o700)
        st2 = os.lstat(cache_dir)
        if (st2.st_mode & 0o077) != 0:
            raise RuntimeError(f"Could not secure permissions (0700) on native cache directory {cache_dir}")
    return cache_dir, False


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
        for pre_dir in prebuilt_native_dirs():
            cand = pre_dir / lib_name
            if verify_native_artifact(cand, require_current_uid=False):
                lib = ctypes.CDLL(str(cand))
                _LOADED[key] = lib
                return lib

        cache_dir, is_build_dir = ensure_user_native_cache_dir(
            explicit_build_dir=os.environ.get("FOCALOOP_BUILD_NATIVE_DIR"),
            explicit_cache_root=os.environ.get("PANEXPR_CACHE_DIR") or os.environ.get("PANEXPR_C_CACHE"),
            subdir_stem="panexpr_medvedev",
        )
        out_path = cache_dir / lib_name
        if out_path.exists() and not verify_native_artifact(out_path, require_current_uid=True):
            lst = os.lstat(out_path)
            if not stat.S_ISLNK(lst.st_mode) and stat.S_ISREG(lst.st_mode) and lst.st_uid == os.getuid():
                out_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"Refusing untrusted cached shared library {out_path}")
        if not out_path.exists():
            cc = find_c_compiler()
            if not cc:
                raise RuntimeError(f"No C compiler found to build {src.name}")
            tmp_path = cache_dir / f".lib{name}_{digest}_{os.getpid()}_{threading.get_ident()}{ext}"
            flags = _sanitize_isa_flags(extra_flags)
            cmd = [cc, "-O3", "-fPIC", "-shared", *flags, str(src), "-lm", "-o", str(tmp_path)]
            try:
                res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
                if res.returncode != 0:
                    raise RuntimeError(f"Failed compiling {src.name}: {res.stderr.strip()}")
                os.chmod(tmp_path, 0o755 if is_build_dir else 0o700)
                os.replace(tmp_path, out_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        if not verify_native_artifact(out_path, require_current_uid=True):
            raise RuntimeError(f"Compiled shared library {out_path} failed ownership/permission check")
        lib = ctypes.CDLL(str(out_path))
        _LOADED[key] = lib
        return lib
