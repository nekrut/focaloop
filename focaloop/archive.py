"""Deterministic, atomic ``.tar.zst`` packing and integrity-verified unpacking of Focaloop state bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

import pyarrow as pa

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 16 * 1024**3
_CHUNK = 1 << 20
ARCHIVE_MANIFEST_NAME = "_archive_manifest.json"


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


def _hash_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(_CHUNK)
            if not blk:
                break
            total += len(blk)
            h.update(blk)
    return total, h.hexdigest()


def pack_state(state_dir: str | Path, out_path: str | Path) -> Path:
    """Pack ``state_dir`` into ``out_path`` atomically with a trailing SHA-256 member manifest."""
    state_dir, out_path = Path(state_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files_to_pack: list[tuple[str, Path, int, str]] = []
    for p in sorted(state_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(state_dir).as_posix()
        if rel == ARCHIVE_MANIFEST_NAME:
            continue
        sz, digest = _hash_file(p)
        files_to_pack.append((rel, p, sz, digest))

    manifest_doc = {
        "format_version": 1,
        "member_count": len(files_to_pack),
        "files": {rel: {"size": sz, "sha256": digest} for rel, _, sz, digest in files_to_pack},
    }
    manifest_bytes = (json.dumps(manifest_doc, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".part", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with pa.CompressedOutputStream(str(tmp_path), "zstd") as fh:
            with tarfile.open(fileobj=fh, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for rel, p, sz, _ in files_to_pack:
                    info = tarfile.TarInfo(name=f"focaloop_state/{rel}")
                    info.size = sz
                    info.mtime = 0
                    info.mode = 0o644
                    with open(p, "rb") as src_f:
                        tar.addfile(info, src_f)
                m_info = tarfile.TarInfo(name=f"focaloop_state/{ARCHIVE_MANIFEST_NAME}")
                m_info.size = len(manifest_bytes)
                m_info.mtime = 0
                m_info.mode = 0o644
                tar.addfile(m_info, io.BytesIO(manifest_bytes))
        os.replace(tmp_path, out_path)
        return out_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def is_state_archive(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as fh:
        return fh.read(4) == ZSTD_MAGIC


def _verify_required_engine_members(dest: Path, extracted: dict[str, tuple[int, str]], label: str) -> None:
    meta_file = dest / "focaloop_state.json"
    if not meta_file.is_file():
        raise ValueError(f"{label}: incomplete or corrupt state archive (missing focaloop_state.json)")
    try:
        meta: dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label}: corrupt focaloop_state.json in state archive: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{label}: invalid focaloop_state.json root type")

    eng = meta.get("engine")
    rel_set = set(extracted)
    if eng in ("panexpr", "expressionloop"):
        has_cat = any(r.startswith("catalog/") for r in rel_set)
        if not has_cat or "panexpr_state/target_counts.parquet" not in rel_set:
            raise ValueError(
                f"{label}: incomplete {eng} state archive (missing catalog/ or panexpr_state/target_counts.parquet)"
            )
    elif eng == "variationloop":
        if "variants.parquet" not in rel_set:
            raise ValueError(f"{label}: incomplete variationloop state archive (missing variants.parquet)")
    elif eng == "malariavar":
        for req in ("cohort/warm/samples.parquet", "cohort/warm/wsaf_u8.npy"):
            if req not in rel_set:
                raise ValueError(f"{label}: incomplete malariavar state archive (missing {req})")
    elif eng is None:
        # Legacy blueprint state archive
        tf_name = meta.get("targets_file")
        if tf_name and str(tf_name) not in rel_set:
            raise ValueError(f"{label}: incomplete blueprint state archive (missing {tf_name})")
        if "target_accumulators.parquet" not in rel_set and "allele_counts.parquet" not in rel_set:
            raise ValueError(
                f"{label}: incomplete blueprint state archive (missing target_accumulators.parquet / allele_counts.parquet)"
            )


def unpack_state(archive: str | Path, dest: str | Path, max_bytes: int = MAX_ARCHIVE_UNCOMPRESSED_BYTES) -> Path:
    dest = Path(dest)
    existed = dest.exists()
    label = str(archive)
    extracted: dict[str, tuple[int, str]] = {}
    manifest_raw: bytes | None = None
    try:
        dest.mkdir(parents=True, exist_ok=True)
        with pa.CompressedInputStream(str(archive), "zstd") as fh:
            capped = _CappedStream(fh, max_bytes, label)
            with tarfile.open(fileobj=capped, mode="r|") as tar:
                for m in tar:
                    name = Path(m.name)
                    if m.isdir() or not m.isfile():
                        continue
                    if name.is_absolute() or ".." in name.parts or name.parts[0] != "focaloop_state" or len(name.parts) < 2:
                        raise ValueError(f"unsafe member in state archive: {m.name}")
                    rel = "/".join(name.parts[1:])
                    src_stream = tar.extractfile(m)
                    if src_stream is None:
                        continue
                    if rel == ARCHIVE_MANIFEST_NAME:
                        data = src_stream.read()
                        if len(data) != m.size:
                            raise ValueError(f"{label}: truncated archive manifest member")
                        manifest_raw = data
                        continue

                    out = dest.joinpath(*name.parts[1:])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    written = 0
                    with open(out, "wb") as dst_f:
                        while True:
                            blk = src_stream.read(_CHUNK)
                            if not blk:
                                break
                            dst_f.write(blk)
                            written += len(blk)
                            h.update(blk)
                    if written != m.size:
                        raise ValueError(
                            f"{label}: truncated member {rel!r} in state archive (got {written} bytes, expected {m.size})"
                        )
                    extracted[rel] = (written, h.hexdigest())

        if manifest_raw is not None:
            try:
                mdoc = json.loads(manifest_raw.decode("utf-8"))
                expected_files = mdoc["files"]
                if not isinstance(expected_files, dict):
                    raise TypeError("files must be a mapping")
            except Exception as exc:
                raise ValueError(f"{label}: corrupt archive manifest in state archive: {exc}") from exc
            missing = sorted(set(expected_files) - set(extracted))
            extra = sorted(set(extracted) - set(expected_files))
            if missing or extra:
                raise ValueError(
                    f"{label}: archive member mismatch against manifest (missing={missing[:5]}, unexpected={extra[:5]})"
                )
            for rel, spec in expected_files.items():
                got_sz, got_sha = extracted[rel]
                if int(spec.get("size", -1)) != got_sz or str(spec.get("sha256", "")) != got_sha:
                    raise ValueError(f"{label}: integrity check failed for archive member {rel!r}")

        if manifest_raw is None or (dest / "focaloop_state.json").exists():
            _verify_required_engine_members(dest, extracted, label)
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
        try:
            return unpack_state(p, tmp.name), tmp
        except Exception:
            tmp.cleanup()
            raise
    raise FileNotFoundError(f"{ref}: neither a state directory nor a .tar.zst state archive")
