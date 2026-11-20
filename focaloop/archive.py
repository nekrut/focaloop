"""Deterministic ``.tar.zst`` packing of a Focaloop state directory (Galaxy-friendly single dataset)."""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 16 * 1024**3
_CHUNK = 1 << 20


class _CappedStream:
    """File-like wrapper over a decompressed stream that caps total bytes read (zstd-bomb guard)."""

    def __init__(self, inner, max_bytes: int, label: str) -> None:
        self._inner = inner
        self._max_bytes = max_bytes
        self._label = label
        self._total = 0

    def read(self, n: int = -1) -> bytes:
        want = _CHUNK if (n is None or n < 0) else min(n, _CHUNK)
        rem = self._max_bytes - self._total
        chunk = self._inner.read(min(want, rem + 1))
        if not chunk:
            return b""
        self._total += len(chunk)
        if self._total > self._max_bytes:
            raise ValueError(f"{self._label}: state archive inflates past {self._max_bytes} bytes; refusing to unpack")
        return chunk


def pack_state(state_dir: str | Path, out_path: str | Path) -> Path:
    state_dir, out_path = Path(state_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.CompressedOutputStream(str(out_path), "zstd") as fh:
        with tarfile.open(fileobj=fh, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(state_dir.rglob("*")):
                if not p.is_file():
                    continue
                info = tarfile.TarInfo(name=f"focaloop_state/{p.relative_to(state_dir).as_posix()}")
                info.size = p.stat().st_size
                info.mtime = 0
                info.mode = 0o644
                with open(p, "rb") as src_f:
                    tar.addfile(info, src_f)
    return out_path


def is_state_archive(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as fh:
        return fh.read(4) == ZSTD_MAGIC


def unpack_state(archive: str | Path, dest: str | Path, max_bytes: int = MAX_ARCHIVE_UNCOMPRESSED_BYTES) -> Path:
    dest = Path(dest)
    existed = dest.exists()
    try:
        with pa.CompressedInputStream(str(archive), "zstd") as fh:
            capped = _CappedStream(fh, max_bytes, str(archive))
            with tarfile.open(fileobj=capped, mode="r|") as tar:
                for m in tar:
                    name = Path(m.name)
                    if m.isdir() or not m.isfile():
                        continue
                    if name.is_absolute() or ".." in name.parts or name.parts[0] != "focaloop_state":
                        raise ValueError(f"unsafe member in state archive: {m.name}")
                    out = dest.joinpath(*name.parts[1:])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tar.extractfile(m)
                    if extracted is not None:
                        with open(out, "wb") as dst_f:
                            shutil.copyfileobj(extracted, dst_f, length=_CHUNK)
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    except Exception:
        if not existed and dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise


def materialize_state(ref: str | Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return a state directory for ``ref`` (directory or ``.tar.zst``); caller keeps the tmp handle alive."""
    p = Path(ref)
    if p.is_dir():
        return p, None
    if is_state_archive(p):
        tmp = tempfile.TemporaryDirectory(prefix="focaloop_state_")
        return unpack_state(p, tmp.name), tmp
    raise FileNotFoundError(f"{ref}: neither a state directory nor a .tar.zst state archive")
