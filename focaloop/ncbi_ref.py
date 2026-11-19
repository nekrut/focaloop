"""Resolve and download NCBI RefSeq/GenBank genome FASTA and matching GFF3 annotation by assembly accession."""

from __future__ import annotations

import os
from pathlib import Path
import re
import socket
import time
from typing import Any

_NCBI_ACC_RE = re.compile(r"^(GC[AF])_(\d{3})(\d{3})(\d{3})(?:\.(\d+))?$", re.IGNORECASE)
_NCBI_FTP_ALL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"


def _ipv4_session():
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    sess = requests.Session()
    sess.headers["User-Agent"] = "focaloop/0.3.0"
    return sess


def _fetch_text(url: str, timeout: float = 45.0, retries: int = 4) -> str:
    sess = _ipv4_session()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _download_file(url: str, dest: Path, timeout: float = 180.0, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    sess = _ipv4_session()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def resolve_ncbi_assembly_dir(accession: str) -> tuple[str, str]:
    """Return ``(assembly_dir_name, base_https_url)`` on NCBI FTP for a GCF_/GCA_ accession."""
    acc = accession.strip()
    m = _NCBI_ACC_RE.match(acc)
    if not m:
        raise ValueError(
            f"Invalid NCBI assembly accession {accession!r}. "
            "Expected a RefSeq or GenBank accession such as GCF_000005845.2 or GCA_000001405.29."
        )
    prefix = m.group(1).upper()
    d1, d2, d3 = m.group(2), m.group(3), m.group(4)
    ver = m.group(5)
    parent_url = f"{_NCBI_FTP_ALL}/{prefix}/{d1}/{d2}/{d3}/"
    listing = _fetch_text(parent_url)
    base_acc = f"{prefix}_{d1}{d2}{d3}"
    if ver:
        pat = re.compile(rf'href="({re.escape(base_acc)}\.{re.escape(ver)}_[^"/]+)/"')
    else:
        pat = re.compile(rf'href="({re.escape(base_acc)}\.\d+_[^"/]+)/"')
    def _ver_key(name: str) -> tuple[int, str]:
        vm = re.search(r"\.(\d+)_", name)
        return (int(vm.group(1)) if vm else 0, name)

    matches = sorted(set(pat.findall(listing)), key=_ver_key)
    if not matches:
        raise ValueError(
            f"NCBI assembly {accession!r} was not found at {parent_url}. "
            "Verify the accession and version number."
        )
    asm_dir = matches[-1]
    return asm_dir, f"{parent_url}{asm_dir}"


def fetch_ncbi_assembly(
    accession: str,
    dest_dir: str | Path,
    *,
    need_annotation: bool = True,
) -> dict[str, Any]:
    """Download genomic FASTA (and optionally matching GFF3 annotation) for an NCBI assembly accession."""
    asm_dir, base_url = resolve_ncbi_assembly_dir(accession)
    cache_env = os.environ.get("FOCALOOP_NCBI_CACHE_DIR", "").strip()
    root = (Path(cache_env) / asm_dir) if cache_env else (Path(dest_dir) / asm_dir)
    root.mkdir(parents=True, exist_ok=True)

    fna_gz = root / f"{asm_dir}_genomic.fna.gz"
    if not (fna_gz.is_file() and fna_gz.stat().st_size > 0):
        _download_file(f"{base_url}/{asm_dir}_genomic.fna.gz", fna_gz)

    gff_gz: Path | None = None
    if need_annotation:
        gff_gz = root / f"{asm_dir}_genomic.gff.gz"
        if not (gff_gz.is_file() and gff_gz.stat().st_size > 0):
            try:
                _download_file(f"{base_url}/{asm_dir}_genomic.gff.gz", gff_gz)
            except RuntimeError as exc:
                raise ValueError(
                    f"NCBI assembly {accession!r} ({asm_dir}) does not provide a matching genomic GFF3 file "
                    f"at {base_url}. Provide a custom GTF/GFF3 override or use the corresponding RefSeq (GCF_) accession."
                ) from exc

    return {
        "accession": accession.strip().upper(),
        "assembly_dir": asm_dir,
        "base_url": base_url,
        "fasta": fna_gz,
        "gff": gff_gz,
    }
