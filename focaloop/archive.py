"""Deterministic ``.tar.zst`` packing of a Focaloop state directory (Galaxy-friendly single dataset)."""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


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


def unpack_state(archive: str | Path, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with pa.CompressedInputStream(str(archive), "zstd") as fh:
        raw = fh.read()
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
