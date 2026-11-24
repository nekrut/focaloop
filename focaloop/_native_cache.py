"""Safe native C/OpenMP artifact cache resolution, verification, and atomic compilation.

Provides:
  1. Portable-by-default ISA selection (``FOCALOOP_PORTABLE_ISA != "0"`` -> SSE4.2+POPCNT on x86-64,
     baseline ARMv8-A on aarch64/arm64; ``-march=native`` only when ``FOCALOOP_PORTABLE_ISA=0``).
  2. Automatic prebuilt lookup in ``sys.prefix / "share" / "focaloop" / "native_cache"`` (so
     ``make install-native`` works out of the box without environment variables) and container caches.
  3. Per-user ``0700`` runtime cache directories outside the package source tree, preventing both
     shared-filesystem CPU-ISA poisoning and ``/tmp`` pre-planting attacks.
  4. Strict ownership and permission checks before ``ctypes.CDLL`` (``dlopen``) or ``exec``.
  5. SHA-256 ``(source + flags)`` cache keys and PID-stamped atomic compilation + ``os.replace``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Iterable, Mapping, Sequence

_COMPILE_LOCK = threading.Lock()


def lib_ext(is_darwin: bool | None = None) -> str:
    eff_darwin = (sys.platform == "darwin") if is_darwin is None else bool(is_darwin)
    return ".dylib" if eff_darwin else ".so"


def use_portable_isa(env: Mapping[str, str] | None = None) -> bool:
    """Return True unless ``FOCALOOP_PORTABLE_ISA`` explicitly opts into host-specific ``-march=native``."""
    src = os.environ if env is None else env
    raw = str(src.get("FOCALOOP_PORTABLE_ISA", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def portable_isa_flags(
    arch: str | None = None,
    is_darwin: bool | None = None,
    *,
    portable: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return ISA compiler flags honoring the portable-by-default policy."""
    eff_arch = (arch or platform.machine()).strip().lower()
    eff_darwin = (sys.platform == "darwin") if is_darwin is None else bool(is_darwin)
    eff_portable = use_portable_isa(env) if portable is None else bool(portable)
    if eff_portable:
        if eff_arch in ("x86_64", "amd64"):
            return ["-msse4.2", "-mpopcnt"]
        return []
    if eff_arch in ("x86_64", "amd64", "aarch64", "arm64") and not eff_darwin:
        return ["-march=native"]
    return []


def pick_c_compiler(env: Mapping[str, str] | None = None) -> list[str]:
    """Locate a working C compiler command vector (``CC``, ``gcc``, ``clang``, ``cc``)."""
    src = os.environ if env is None else env
    for cand in (src.get("CC"), "gcc", "clang", "cc"):
        if cand:
            toks = shlex.split(cand)
            if toks and (shutil.which(toks[0]) or Path(toks[0]).is_file()):
                return toks
    raise RuntimeError("No C compiler (CC/gcc/clang/cc) found in PATH")


def prebuilt_search_dirs(extra_dirs: Iterable[str | Path | None] = ()) -> list[Path]:
    """Return ordered prebuilt native cache directories (checking ``sys.prefix`` automatically)."""
    raw_cands: list[str | Path | None] = [
        os.environ.get("FOCALOOP_NATIVE_CACHE"),
        os.environ.get("FOCALOOP_NATIVE_DIR"),
        os.environ.get("FOCALOOP_BIN_CACHE"),
        Path(sys.prefix) / "share" / "focaloop" / "native_cache",
        Path("/opt/focaloop/native_cache"),
        *extra_dirs,
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for item in raw_cands:
        if not item:
            continue
        s = str(item).strip()
        if not s:
            continue
        p = Path(s)
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _allowed_prebuilt_uids(target: Path) -> set[int]:
    euid = os.geteuid()
    allowed = {euid, 0}
    try:
        prefix_resolved = Path(sys.prefix).resolve()
        target_resolved = target.resolve()
        if target_resolved == prefix_resolved or prefix_resolved in target_resolved.parents:
            allowed.add(prefix_resolved.stat().st_uid)
    except OSError:
        pass
    return allowed


def verify_safe_artifact(
    path: str | Path,
    *,
    require_per_user_mode: bool = False,
    require_exec: bool = False,
) -> Path:
    """Verify ownership and permissions of a shared library or helper binary before ``dlopen``/``exec``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Native artifact does not exist: {p}")

    euid = os.geteuid()
    egid = os.getegid()

    if require_per_user_mode:
        lst = p.lstat()
        if stat.S_ISLNK(lst.st_mode):
            raise PermissionError(f"Refusing symlinked native artifact in runtime cache: {p}")
        if not stat.S_ISREG(lst.st_mode):
            raise PermissionError(f"Native artifact is not a regular file: {p}")
        if lst.st_uid != euid:
            raise PermissionError(
                f"Refusing native artifact {p} owned by uid={lst.st_uid} (expected uid={euid})"
            )
        if (lst.st_mode & 0o077) != 0:
            raise PermissionError(
                f"Refusing native artifact {p} with group/world permissions { oct(lst.st_mode & 0o777) } (expected 0700/0600)"
            )
        parent = p.parent
        plst = parent.lstat()
        if stat.S_ISLNK(plst.st_mode) or not stat.S_ISDIR(plst.st_mode):
            raise PermissionError(f"Refusing non-directory or symlinked runtime cache parent: {parent}")
        if plst.st_uid != euid:
            raise PermissionError(
                f"Refusing runtime cache directory {parent} owned by uid={plst.st_uid} (expected uid={euid})"
            )
        if (plst.st_mode & 0o077) != 0:
            raise PermissionError(
                f"Refusing runtime cache directory {parent} with mode {oct(plst.st_mode & 0o777)} (expected 0700)"
            )
    else:
        resolved = p.resolve()
        st = resolved.stat()
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"Native artifact is not a regular file: {resolved}")
        allowed_uids = _allowed_prebuilt_uids(resolved)
        if st.st_uid not in allowed_uids:
            raise PermissionError(
                f"Refusing native artifact {resolved} owned by untrusted uid={st.st_uid} (allowed={sorted(allowed_uids)})"
            )
        if (st.st_mode & 0o002) != 0:
            raise PermissionError(f"Refusing world-writable native artifact: {resolved}")
        if (st.st_mode & 0o020) != 0 and not (st.st_uid == euid and st.st_gid == egid):
            raise PermissionError(f"Refusing group-writable native artifact: {resolved}")
        pst = resolved.parent.stat()
        if pst.st_uid not in allowed_uids:
            raise PermissionError(
                f"Refusing native artifact directory {resolved.parent} owned by untrusted uid={pst.st_uid}"
            )
        if (pst.st_mode & 0o002) != 0:
            raise PermissionError(f"Refusing native artifact inside world-writable directory: {resolved.parent}")

    if require_exec and not os.access(p, os.X_OK):
        raise PermissionError(f"Native binary is not executable: {p}")
    return p


def _ensure_private_dir(path: Path) -> Path:
    """Create or validate a directory owned by ``os.geteuid()`` with strict ``0700`` permissions."""
    euid = os.geteuid()
    if path.is_symlink():
        raise PermissionError(f"Refusing symlinked native cache directory: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            pass
    lst = path.lstat()
    if stat.S_ISLNK(lst.st_mode) or not stat.S_ISDIR(lst.st_mode):
        raise PermissionError(f"Invalid native cache directory (not a real directory): {path}")
    if lst.st_uid != euid:
        raise PermissionError(
            f"Refusing native cache directory {path} owned by uid={lst.st_uid} (expected uid={euid})"
        )
    if (lst.st_mode & 0o077) != 0:
        os.chmod(path, 0o700)
        lst = path.lstat()
        if (lst.st_mode & 0o077) != 0:
            raise PermissionError(f"Could not enforce 0700 permissions on native cache directory: {path}")
    return path


def resolve_user_cache_dir(
    subdir: str = "native_cache",
    *,
    env_vars: Sequence[str] = (),
) -> Path:
    """Resolve and verify a per-user ``0700`` runtime compile cache directory without touching ``~/``."""
    for ev in ("FOCALOOP_BUILD_NATIVE_DIR", *env_vars):
        val = os.environ.get(ev, "").strip()
        if val:
            return _ensure_private_dir(Path(val))

    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        root = _ensure_private_dir(Path(xdg) / f"focaloop_{os.geteuid()}")
    else:
        root = _ensure_private_dir(Path(tempfile.gettempdir()) / f"focaloop_native_{os.geteuid()}")
    if not subdir:
        return root
    return _ensure_private_dir(root / subdir)


def compute_artifact_key(
    sources: Sequence[str | Path] = (),
    *,
    inline_sources: Sequence[str | bytes] = (),
    flags: Sequence[str] = (),
    extra: str = "",
) -> str:
    """Compute a 16-hex-char SHA-256 cache key over ``(platform, arch, flags, sources)``."""
    h = hashlib.sha256()
    h.update(f"{sys.platform}:{platform.machine().lower()}\0".encode("utf-8"))
    h.update(("\0".join(str(f) for f in flags) + "\0").encode("utf-8"))
    if extra:
        h.update((extra + "\0").encode("utf-8"))
    for sp in sources:
        p = Path(sp)
        h.update(f"file:{p.name}\0".encode("utf-8"))
        h.update(p.read_bytes())
        h.update(b"\0")
    for idx, src in enumerate(inline_sources):
        h.update(f"inline:{idx}\0".encode("utf-8"))
        h.update(src.encode("utf-8") if isinstance(src, str) else src)
        h.update(b"\0")
    return h.hexdigest()[:16]


def find_prebuilt_artifact(
    candidate_names: Sequence[str],
    *,
    require_exec: bool = False,
    extra_dirs: Iterable[str | Path | None] = (),
) -> Path | None:
    """Locate a verified prebuilt artifact in ``prebuilt_search_dirs``."""
    for d in prebuilt_search_dirs(extra_dirs):
        for name in candidate_names:
            cand = d / name
            if cand.exists() or cand.is_symlink():
                verify_safe_artifact(cand, require_per_user_mode=False, require_exec=require_exec)
                return cand
    return None


def ensure_compiled_artifact(
    stem: str,
    *,
    sources: Sequence[str | Path] = (),
    inline_source: str | None = None,
    headers: Sequence[str | Path] = (),
    include_dirs: Sequence[str | Path] = (),
    shared: bool = True,
    use_openmp: bool = False,
    extra_cflags: Sequence[str] = (),
    libs: Sequence[str] = ("-lm",),
    prebuilt_names: Sequence[str] = (),
    cache_subdir: str = "native_cache",
    cache_env_vars: Sequence[str] = (),
) -> Path:
    """Return a verified prebuilt or atomically compiled native shared library / executable."""
    is_darwin = sys.platform == "darwin"
    ext = lib_ext(is_darwin) if shared else ""
    portable = use_portable_isa()
    isa = portable_isa_flags(is_darwin=is_darwin, portable=portable)
    cflags_env = shlex.split(os.environ.get("CFLAGS", ""))
    ldflags_env = shlex.split(os.environ.get("LDFLAGS", ""))

    key_flags = [
        "-O3",
        *(["-shared", "-fPIC"] if shared else []),
        *isa,
        *cflags_env,
        *extra_cflags,
        *(["openmp"] if use_openmp else []),
        *ldflags_env,
        *libs,
    ]
    key = compute_artifact_key(
        [*sources, *headers],
        inline_sources=([inline_source] if inline_source is not None else ()),
        flags=key_flags,
        extra=stem,
    )
    hashed_name = f"{stem}_{key}{ext}"

    lookup_names = [hashed_name]
    if portable:
        for pn in prebuilt_names:
            if pn not in lookup_names:
                lookup_names.append(pn)

    pre = find_prebuilt_artifact(lookup_names, require_exec=not shared)
    if pre is not None:
        return pre

    cache_dir = resolve_user_cache_dir(cache_subdir, env_vars=cache_env_vars)
    out_path = cache_dir / hashed_name
    if out_path.exists() or out_path.is_symlink():
        return verify_safe_artifact(out_path, require_per_user_mode=True, require_exec=not shared)

    with _COMPILE_LOCK:
        if out_path.exists() or out_path.is_symlink():
            return verify_safe_artifact(out_path, require_per_user_mode=True, require_exec=not shared)

        cc_cmd = pick_c_compiler()
        inc_args: list[str] = []
        for d in include_dirs:
            inc_args.extend(["-I", str(d)])

        omp_attempts: list[list[str]] = [[]]
        if use_openmp:
            omp_attempts = [["-Xpreprocessor", "-fopenmp", "-lomp"], ["-fopenmp"], []] if is_darwin else [["-fopenmp"], []]

        stamp = f"{os.getpid()}_{threading.get_ident()}_{os.urandom(4).hex()}"
        tmp_c: Path | None = None
        tmp_out = cache_dir / f".tmp_{stem}_{key}_{stamp}{ext}"
        src_args = [str(s) for s in sources]
        try:
            if inline_source is not None:
                tmp_c = cache_dir / f".tmp_{stem}_{key}_{stamp}.c"
                fd = os.open(str(tmp_c), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(inline_source)
                src_args.append(str(tmp_c))

            sh_flags = ["-shared", "-fPIC"] if shared else []
            last_err = ""
            for omp in omp_attempts:
                cmd = [
                    *cc_cmd,
                    *cflags_env,
                    "-O3",
                    *sh_flags,
                    *isa,
                    *extra_cflags,
                    *inc_args,
                    *(f for f in omp if not f.startswith("-l")),
                    *src_args,
                    *ldflags_env,
                    *(f for f in omp if f.startswith("-l")),
                    *libs,
                    "-o",
                    str(tmp_out),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if res.returncode == 0 and tmp_out.is_file() and not tmp_out.is_symlink():
                    os.chmod(tmp_out, 0o700)
                    os.replace(tmp_out, out_path)
                    return verify_safe_artifact(out_path, require_per_user_mode=True, require_exec=not shared)
                last_err = res.stderr.strip()
            raise RuntimeError(f"Failed compiling {out_path.name} with {cc_cmd[0]}: {last_err}")
        finally:
            if tmp_c is not None:
                tmp_c.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)


def load_verified_cdll(
    path: str | Path,
    *,
    require_per_user_mode: bool = False,
) -> ctypes.CDLL:
    """Verify ownership and permissions of ``path`` before loading it via ``ctypes.CDLL``."""
    verified = verify_safe_artifact(path, require_per_user_mode=require_per_user_mode, require_exec=False)
    return ctypes.CDLL(str(verified))
