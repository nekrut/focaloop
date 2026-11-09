"""Deterministic ``.tar.zst`` packing of a Focaloop state directory (Galaxy-friendly single dataset)."""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3
_CHUNK = 1 << 20


def pack_state(state_dir: str | Path, out_path: str | Path) -> Path:
    state_dir, out_path = Path(state_dir), Path(out_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in sorted(state_dir.rglob("*")):
            if not p.is_file():
                continue
            info = tarfile.TarInfo(name=f"focaloop_state/{p.relative_to(state_dir).as_posix()}")
            data = p.read_bytes()
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    with pa.CompressedOutputStream(str(out_path), "zstd") as fh:
        fh.write(buf.getvalue())
    return out_path


def is_state_archive(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as fh:
        return fh.read(4) == ZSTD_MAGIC


def _read_capped(archive: str | Path, max_bytes: int) -> bytes:
    """Decompress in bounded chunks; refuse archives that inflate past ``max_bytes`` (zstd bombs)."""
    buf = io.BytesIO()
    total = 0
    with pa.CompressedInputStream(str(archive), "zstd") as fh:
        while True:
            chunk = fh.read(min(_CHUNK, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{archive}: state archive inflates past {max_bytes} bytes; refusing to unpack")
            buf.write(chunk)
    return buf.getvalue()


def unpack_state(archive: str | Path, dest: str | Path, max_bytes: int = MAX_ARCHIVE_UNCOMPRESSED_BYTES) -> Path:
    dest = Path(dest)
    raw = _read_capped(archive, max_bytes)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        for m in tar.getmembers():
            name = Path(m.name)
            if m.isdir() or not m.isfile():
                continue
            if name.is_absolute() or ".." in name.parts or name.parts[0] != "focaloop_state":
                raise ValueError(f"unsafe member in state archive: {m.name}")
            out = dest.joinpath(*name.parts[1:])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tar.extractfile(m).read())
    return dest


def materialize_state(ref: str | Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return a state directory for ``ref`` (directory or ``.tar.zst``); caller keeps the tmp handle alive."""
    p = Path(ref)
    if p.is_dir():
        return p, None
    if is_state_archive(p):
        tmp = tempfile.TemporaryDirectory(prefix="focaloop_state_")
        return unpack_state(p, tmp.name), tmp
    raise FileNotFoundError(f"{ref}: neither a state directory nor a .tar.zst state archive")
