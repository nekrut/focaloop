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
_NCBI_DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome"


def _fetch_json(url: str, params: dict[str, Any] | None = None, timeout: float = 30.0, retries: int = 3) -> dict[str, Any]:
    sess = _ipv4_session()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to query NCBI Datasets API {url}: {last_err}")


def lookup_ncbi_assembly_taxon(accession: str) -> dict[str, Any]:
    """Resolve species name, tax_id, and strain for an NCBI assembly accession."""
    acc = accession.strip().upper()
    if not _NCBI_ACC_RE.match(acc):
        raise ValueError(f"Invalid NCBI assembly accession {accession!r}")
    data = _fetch_json(f"{_NCBI_DATASETS_API}/accession/{acc}/dataset_report")
    reports = data.get("reports") or []
    if not reports:
        raise ValueError(f"NCBI Datasets returned no metadata for accession {accession!r}")
    rep = reports[0]
    org = rep.get("organism") or {}
    org_name = str(org.get("organism_name") or "").strip()
    parts = org_name.split()
    species_name = " ".join(parts[:2]) if len(parts) >= 2 else org_name
    infra = org.get("infraspecific_names") or {}
    strain = str(infra.get("strain") or infra.get("isolate") or "").strip()
    return {
        "accession": acc,
        "tax_id": org.get("tax_id"),
        "organism_name": org_name,
        "species_name": species_name,
        "strain": strain,
    }


def _norm_strain(val: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(val or "").lower())


def discover_related_ncbi_assemblies(
    taxon_or_species: str,
    *,
    max_strains: int = 3,
    exclude_accessions: tuple[str, ...] | list[str] | set[str] = (),
    exclude_strains: tuple[str, ...] | list[str] | set[str] = (),
    candidate_pool: int = 12,
) -> list[dict[str, Any]]:
    """Discover strain-deduplicated RefSeq annotated genomes for a species or NCBI taxon ID."""
    import urllib.parse

    q = str(taxon_or_species or "").strip()
    if not q:
        raise ValueError("Provide a species name or NCBI taxonomy ID to auto-fetch related RefSeq strains.")
    want_pool = max(int(max_strains) * 3, int(candidate_pool))
    excl_accs: set[str] = set()
    for a in exclude_accessions:
        base_a = str(a).strip().upper().split(".")[0]
        if base_a:
            excl_accs.add(base_a)
            if base_a.startswith("GCA_"):
                excl_accs.add("GCF_" + base_a[4:])
            elif base_a.startswith("GCF_"):
                excl_accs.add("GCA_" + base_a[4:])
    excl_strains = {_norm_strain(s) for s in exclude_strains if _norm_strain(s)}

    encoded_q = urllib.parse.quote(q, safe="")
    url = f"{_NCBI_DATASETS_API}/taxon/{encoded_q}/dataset_report"

    # Try complete genomes first, then broaden to chromosome/scaffold if needed
    raw_reports: list[dict[str, Any]] = []
    for level in ("complete_genome", None):
        params: dict[str, Any] = {
            "filters.assembly_source": "refseq",
            "filters.has_annotation": "true",
            "filters.exclude_paired_reports": "true",
            "page_size": min(100, max(50, want_pool * 3)),
        }
        if level:
            params["filters.assembly_level"] = level
        try:
            payload = _fetch_json(url, params=params)
            reps = payload.get("reports") or []
            raw_reports.extend(reps)
        except Exception:
            if level is None and not raw_reports:
                raise
        if len(raw_reports) >= want_pool * 2:
            break

    # Prioritize reference/representative genomes, then complete > chromosome > scaffold
    level_rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}

    def _sort_key(rep: dict[str, Any]) -> tuple[int, int, str]:
        ainfo = rep.get("assembly_info") or {}
        cat = str(ainfo.get("refseq_category") or "").lower()
        cat_r = 0 if "reference" in cat else (1 if "representative" in cat else 2)
        lvl_r = level_rank.get(str(ainfo.get("assembly_level") or ""), 4)
        return (cat_r, lvl_r, str(rep.get("accession") or ""))

    raw_reports.sort(key=_sort_key)

    selected: list[dict[str, Any]] = []
    seen_base_acc: set[str] = set(excl_accs)
    seen_strains: set[str] = set(excl_strains)

    for rep in raw_reports:
        acc = str(rep.get("accession") or "").strip().upper()
        if not acc or not _NCBI_ACC_RE.match(acc):
            continue
        base_acc = acc.split(".")[0]
        if base_acc in seen_base_acc:
            continue
        org = rep.get("organism") or {}
        infra = org.get("infraspecific_names") or {}
        strain = str(infra.get("strain") or infra.get("isolate") or "").strip()
        snorm = _norm_strain(strain)
        if snorm and snorm in seen_strains:
            continue
        seen_base_acc.add(base_acc)
        if snorm:
            seen_strains.add(snorm)
        ainfo = rep.get("assembly_info") or {}
        selected.append(
            {
                "accession": acc,
                "strain": strain,
                "assembly_name": str(ainfo.get("assembly_name") or ""),
                "assembly_level": str(ainfo.get("assembly_level") or ""),
                "refseq_category": str(ainfo.get("refseq_category") or ""),
            }
        )
        if len(selected) >= want_pool:
            break

    return selected


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
