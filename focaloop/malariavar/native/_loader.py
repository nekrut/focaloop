"""Auto-compile and load native C shared libraries with portable ISA defaults and hashed per-user cache."""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

NATIVE_DIR = Path(__file__).resolve().parent
_BUILD_LOCK = threading.Lock()


def _detect_cc() -> str:
    env_cc = os.environ.get("CC")
    if env_cc:
        found_env = shutil.which(env_cc)
        if found_env:
            return found_env
        p = Path(env_cc)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise RuntimeError(f"Configured C compiler CC={env_cc!r} not found or not executable")
    for cand in ("gcc", "clang", "cc"):
        found = shutil.which(cand)
        if found:
            return found
    raise RuntimeError("No C compiler (gcc/clang/cc) found in PATH")


def _is_portable_isa() -> bool:
    """Default to portable baseline ISA unless FOCALOOP_PORTABLE_ISA is explicitly '0'."""
    return os.environ.get("FOCALOOP_PORTABLE_ISA", "1").strip() != "0"


def _isa_flags(portable: bool) -> list[str]:
    arch = platform.machine().lower()
    if portable:
        if arch in ("x86_64", "amd64"):
            return ["-msse4.2", "-mpopcnt"]
        return []
    if arch in ("x86_64", "amd64"):
        return ["-march=native", "-mpopcnt"]
    if arch in ("aarch64", "arm64"):
        if sys.platform == "darwin":
            return ["-mcpu=apple-m1"]
        return ["-march=native"]
    return []


def _omp_flags(cc: str) -> list[str]:
    if sys.platform == "darwin" and "clang" in os.path.basename(cc):
        for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.isdir(prefix):
                return [
                    "-Xpreprocessor",
                    "-fopenmp",
                    f"-I{prefix}/include",
                    f"-L{prefix}/lib",
                    "-lomp",
                ]
        return []
    return ["-fopenmp"]


def _compute_build_hash(c_paths: list[Path], hdr_paths: list[Path], flags: list[str]) -> str:
    h = hashlib.sha256()
    h.update(b"focaloop-mv-native-v2\0")
    h.update(" ".join(flags).encode("utf-8") + b"\0")
    loader_path = Path(__file__).resolve()
    seen: set[Path] = set()
    for p in [*c_paths, *hdr_paths, loader_path]:
        rp = p.resolve()
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        h.update(rp.name.encode("utf-8") + b"\0")
        h.update(rp.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def _ensure_private_user_cache_dir() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    base = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    cache_dir = base / f"focaloop_native_{uid}"
    cache_dir.mkdir(mode=0o700, exist_ok=True)
    st = os.lstat(cache_dir)
    if stat.S_ISLNK(st.st_mode):
        raise RuntimeError(f"Refusing symlinked native cache directory: {cache_dir}")
    if hasattr(os, "getuid") and st.st_uid != uid:
        raise RuntimeError(f"Native cache directory {cache_dir} is not owned by current user (uid={uid})")
    if (st.st_mode & 0o777) != 0o700:
        os.chmod(cache_dir, 0o700)
    return cache_dir


def _prebuilt_search_dirs() -> list[Path]:
    out: list[Path] = []
    for raw in (
        os.environ.get("FOCALOOP_NATIVE_CACHE", ""),
        str(Path(sys.prefix) / "share" / "focaloop" / "native_cache"),
        "/opt/focaloop/native_cache",
    ):
        if raw:
            p = Path(raw)
            if p not in out:
                out.append(p)
    return out


def ensure_shared_lib(
    lib_stem: str,
    sources: list[str],
    headers: list[str] | None = None,
    use_openmp: bool = True,
    extra_cflags: list[str] | None = None,
) -> Path:
    """Compile a shared library if missing from portable prebuilt cache or per-user SHA-256 cache."""
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    arch = platform.machine().lower()
    portable = _is_portable_isa()
    unhashed_fname = f"{lib_stem}_{sys.platform}_{arch}{ext}"

    c_paths = [NATIVE_DIR / s for s in sources]
    hdr_paths = [NATIVE_DIR / h for h in (headers or [])] + sorted(NATIVE_DIR.glob("*.h"))
    base_flags = ["-O3", "-fPIC", "-shared", "-Wall", "-Wno-unused-function", *_isa_flags(portable)]
    if extra_cflags:
        base_flags.extend(extra_cflags)
    if use_openmp:
        base_flags.append("OPENMP=1")
    digest = _compute_build_hash(c_paths, hdr_paths, base_flags)
    hashed_fname = f"{lib_stem}_{sys.platform}_{arch}_{digest}{ext}"

    explicit_build_dir = os.environ.get("FOCALOOP_BUILD_NATIVE_DIR", "").strip()
    if not explicit_build_dir and portable and not extra_cflags:
        for pre_dir in _prebuilt_search_dirs():
            for cand_name in (hashed_fname, unhashed_fname):
                cand = pre_dir / cand_name
                if cand.is_file():
                    return cand

    with _BUILD_LOCK:
        if explicit_build_dir:
            out_dir = Path(explicit_build_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            lib_path = out_dir / (hashed_fname if extra_cflags else unhashed_fname)
        else:
            out_dir = _ensure_private_user_cache_dir()
            lib_path = out_dir / hashed_fname
            if lib_path.is_file():
                return lib_path

        cc = _detect_cc()
        cmd_prefix = [
            cc,
            "-O3",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wno-unused-function",
            "-I",
            str(NATIVE_DIR),
            *_isa_flags(portable),
        ]
        if extra_cflags:
            cmd_prefix.extend(extra_cflags)

        omp = _omp_flags(cc) if use_openmp else []
        attempts = [omp, []] if omp else [[]]
        tmp_out = out_dir / f".{lib_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        last_err = ""
        try:
            for omp_try in attempts:
                omp_compile = [f for f in omp_try if not f.startswith(("-l", "-L"))]
                omp_link = [f for f in omp_try if f.startswith(("-l", "-L"))]
                cmd = [
                    *cmd_prefix,
                    *omp_compile,
                    *(str(p) for p in c_paths),
                    *omp_link,
                    "-lpthread",
                    "-lm",
                    "-o",
                    str(tmp_out),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and tmp_out.is_file():
                    os.replace(tmp_out, lib_path)
                    return lib_path
                last_err = res.stderr
        finally:
            tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to compile {lib_path.name} with {cc}: {last_err}")


def load_shared_lib(
    lib_stem: str,
    sources: list[str],
    headers: list[str] | None = None,
    use_openmp: bool = True,
    extra_cflags: list[str] | None = None,
) -> ctypes.CDLL:
    """Ensure shared library is compiled cleanly and open via ctypes."""
    path = ensure_shared_lib(
        lib_stem, sources, headers=headers, use_openmp=use_openmp, extra_cflags=extra_cflags
    )
    return ctypes.CDLL(str(path))


load_cdll = load_shared_lib
