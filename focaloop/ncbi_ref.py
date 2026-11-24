"""Resolve and download NCBI RefSeq/GenBank genome FASTA and matching GFF3 annotation by assembly accession."""

from __future__ import annotations

import os
from pathlib import Path
import re
import socket
import time
from typing import Any
import urllib.parse

_NCBI_ACC_RE = re.compile(r"^(GC[AF])_(\d{3})(\d{3})(\d{3})(?:\.(\d+))?$", re.IGNORECASE)
_NCBI_FTP_ALL = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
_NCBI_DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome"
_NCBI_TAXONOMY_API = "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon"

_UNCLASSIFIED_SPECIES_RE = re.compile(
    r"(?:^|\s)(?:sp\.?|spp\.?|bacterium|archaeon|uncultured|unidentified|endosymbiont|symbiont|metagenome)(?:\s|$)",
    re.IGNORECASE,
)
_BAD_EPITHET_TOKENS = frozenset(
    {
        "sp",
        "sp.",
        "spp",
        "spp.",
        "cf",
        "cf.",
        "aff",
        "aff.",
        "bacterium",
        "archaeon",
        "endosymbiont",
        "symbiont",
        "phage",
        "virus",
        "uncultured",
        "unidentified",
    }
)


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


def _is_unclassified_species_name(name: str) -> bool:
    s = re.sub(r"\s+", " ", str(name or "").strip())
    if not s:
        return True
    return bool(_UNCLASSIFIED_SPECIES_RE.search(s))


def _parse_species_binomial(org_name: str) -> str:
    """Extract a resolved named species ('Genus epithet' or 'Candidatus Genus epithet'), or '' if unresolvable."""
    s = re.sub(r"\s+", " ", str(org_name or "").strip())
    if not s or _is_unclassified_species_name(s):
        return ""
    parts = s.split()
    if parts[0].lower() == "candidatus":
        if len(parts) < 3:
            return ""
        genus = parts[1].strip("[]")
        epithet = parts[2]
        if (
            not re.match(r"^[A-Z][A-Za-z-]+$", genus)
            or epithet.lower() in _BAD_EPITHET_TOKENS
            or not re.match(r"^[a-z][a-z-]{1,}$", epithet)
        ):
            return ""
        return f"Candidatus {genus} {epithet}"
    if len(parts) < 2:
        return ""
    genus = parts[0].strip("[]")
    epithet = parts[1]
    if (
        not re.match(r"^[A-Z][A-Za-z-]+$", genus)
        or epithet.lower() in _BAD_EPITHET_TOKENS
        or not re.match(r"^[a-z][a-z-]{1,}$", epithet)
    ):
        return ""
    return f"{parts[0]} {epithet}" if parts[0].startswith("[") else f"{genus} {epithet}"


def _resolve_species_from_taxonomy(query: str | int) -> tuple[int | None, str, bool]:
    """Resolve ``query`` via NCBI Datasets taxonomy API to ``(species_tax_id, species_name, taxonomy_queried)``.

    Returns ``(None, '', False)`` only when ``_fetch_json`` is stubbed without ``taxonomy_nodes`` in offline unit tests.
    Raises ``ValueError`` when NCBI taxonomy confirms the taxon has no named species-rank identity (e.g. genus or 'Genus sp.').
    """
    q = str(query or "").strip()
    if not q:
        raise ValueError("Empty taxon query; provide a species name or species-rank NCBI taxonomy ID.")
    enc = urllib.parse.quote(q, safe="")
    payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{enc}")
    if not isinstance(payload, dict) or "taxonomy_nodes" not in payload:
        return None, "", False
    nodes = payload.get("taxonomy_nodes") or []
    if not nodes:
        raise ValueError(
            f"NCBI Datasets taxonomy could not resolve {query!r} to a valid species. "
            "Provide an explicit species name or species-rank NCBI taxonomy ID in --auto-expand-taxon."
        )
    tax = (nodes[0] or {}).get("taxonomy") or {}
    tid = tax.get("tax_id")
    tname = str(tax.get("organism_name") or "").strip()
    rank = str(tax.get("rank") or "").upper()
    if not tid:
        raise ValueError(
            f"NCBI Datasets taxonomy returned no tax_id for {query!r}. "
            "Provide an explicit species name or species-rank NCBI taxonomy ID in --auto-expand-taxon."
        )
    if rank == "SPECIES":
        if _is_unclassified_species_name(tname):
            raise ValueError(
                f"Organism {tname!r} (tax_id={tid}) is an unclassified/placeholder taxon rather than a named species. "
                "Specify an explicit target species or taxonomy ID in --auto-expand-taxon."
            )
        return int(tid), tname, True

    lineage = [str(x) for x in (tax.get("lineage") or []) if str(x).isdigit() and int(x) > 1]
    if lineage:
        lin_csv = urllib.parse.quote(",".join(lineage[-30:]), safe=",")
        lin_payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{lin_csv}")
        for lnode in lin_payload.get("taxonomy_nodes") or []:
            lt = (lnode or {}).get("taxonomy") or {}
            if str(lt.get("rank") or "").upper() == "SPECIES" and lt.get("tax_id"):
                anc_name = str(lt.get("organism_name") or "").strip()
                anc_tid = int(lt["tax_id"])
                if _is_unclassified_species_name(anc_name):
                    raise ValueError(
                        f"Organism {tname or query!r} resolves to unclassified taxon {anc_name!r} (tax_id={anc_tid}). "
                        "Specify an explicit target species or taxonomy ID in --auto-expand-taxon."
                    )
                return anc_tid, anc_name, True

    raise ValueError(
        f"Taxon {tname or query!r} (rank={rank or 'unknown'}, tax_id={tid}) does not resolve to a species-rank NCBI taxon. "
        "Specify an explicit species name or species-rank NCBI taxonomy ID in --auto-expand-taxon."
    )


def _batch_resolve_species_taxids(tax_ids: set[int]) -> dict[int, int | None]:
    """Map ``{organism_tax_id: species_tax_id}`` for candidate assembly organisms via NCBI Datasets taxonomy."""
    if not tax_ids:
        return {}
    ordered = sorted(tax_ids)
    out: dict[int, int | None] = {}
    need_lineage: dict[int, list[int]] = {}
    all_anc: set[int] = set()

    for i in range(0, len(ordered), 50):
        chunk = ordered[i : i + 50]
        csv_ids = urllib.parse.quote(",".join(str(x) for x in chunk), safe=",")
        try:
            payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{csv_ids}")
        except Exception:
            return {}
        if not isinstance(payload, dict) or "taxonomy_nodes" not in payload:
            return {}
        for node in payload.get("taxonomy_nodes") or []:
            tx = (node or {}).get("taxonomy") or {}
            tid_raw = tx.get("tax_id")
            if not tid_raw:
                continue
            tid = int(tid_raw)
            rank = str(tx.get("rank") or "").upper()
            tname = str(tx.get("organism_name") or "")
            if rank == "SPECIES":
                out[tid] = None if _is_unclassified_species_name(tname) else tid
            else:
                lin = [int(x) for x in (tx.get("lineage") or []) if str(x).isdigit() and int(x) > 1]
                need_lineage[tid] = lin
                all_anc.update(lin[-25:])

    if all_anc:
        anc_species: set[int] = set()
        anc_list = sorted(all_anc)
        for i in range(0, len(anc_list), 50):
            chunk = anc_list[i : i + 50]
            csv_ids = urllib.parse.quote(",".join(str(x) for x in chunk), safe=",")
            try:
                payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{csv_ids}")
            except Exception:
                break
            for node in payload.get("taxonomy_nodes") or []:
                tx = (node or {}).get("taxonomy") or {}
                if str(tx.get("rank") or "").upper() == "SPECIES" and tx.get("tax_id"):
                    if not _is_unclassified_species_name(str(tx.get("organism_name") or "")):
                        anc_species.add(int(tx["tax_id"]))
        for tid, lin in need_lineage.items():
            sp_hit = next((a for a in reversed(lin) if a in anc_species), None)
            out[tid] = sp_hit
    return out


def lookup_ncbi_assembly_taxon(accession: str) -> dict[str, Any]:
    """Resolve species-rank tax_id, species_name (handling Candidatus / Genus sp.), and strain for an NCBI assembly."""
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
    raw_tax_id = org.get("tax_id")
    infra = org.get("infraspecific_names") or {}
    strain = str(infra.get("strain") or infra.get("isolate") or "").strip()

    sp_tax_id: int | None = None
    species_name = ""
    tax_queried = False
    if raw_tax_id is not None or org_name:
        sp_tax_id, species_name, tax_queried = _resolve_species_from_taxonomy(
            raw_tax_id if raw_tax_id is not None else org_name
        )
    if not tax_queried:
        species_name = _parse_species_binomial(org_name)
        if not species_name:
            raise ValueError(
                f"Could not resolve a named species from organism {org_name!r} for assembly {acc}. "
                "Provide an explicit species name or NCBI taxonomy ID in --auto-expand-taxon."
            )
        if raw_tax_id is not None and str(raw_tax_id).isdigit():
            sp_tax_id = int(raw_tax_id)

    return {
        "accession": acc,
        "tax_id": sp_tax_id if sp_tax_id is not None else raw_tax_id,
        "species_tax_id": sp_tax_id,
        "organism_tax_id": raw_tax_id,
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
    """Discover strain-deduplicated RefSeq annotated genomes strictly matching a resolved species taxon."""
    q = str(taxon_or_species or "").strip()
    if not q:
        raise ValueError("Provide a species name or NCBI taxonomy ID to auto-fetch related RefSeq strains.")
    if not q.isdigit() and _is_unclassified_species_name(q):
        raise ValueError(
            f"Cannot auto-expand strains for unclassified/placeholder taxon {q!r}. "
            "Provide a named species or species-rank NCBI taxonomy ID in --auto-expand-taxon."
        )

    target_sp_taxid, target_sp_name, tax_queried = _resolve_species_from_taxonomy(q)
    if not tax_queried:
        if q.isdigit():
            target_sp_taxid = int(q)
            target_sp_name = ""
        else:
            target_sp_name = _parse_species_binomial(q)
            if not target_sp_name:
                raise ValueError(
                    f"Taxon query {q!r} is not a resolvable species name (for example, 'Escherichia coli' or "
                    "'Candidatus Liberibacter asiaticus') or numeric NCBI taxonomy ID."
                )

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

    query_token = str(target_sp_taxid) if target_sp_taxid is not None else (target_sp_name or q)
    encoded_q = urllib.parse.quote(query_token, safe="")
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

    cand_org_tids: set[int] = set()
    if tax_queried and target_sp_taxid is not None:
        for rep in raw_reports:
            rtid = (rep.get("organism") or {}).get("tax_id")
            if rtid is not None and str(rtid).isdigit() and int(rtid) != target_sp_taxid:
                cand_org_tids.add(int(rtid))
    tid_to_sp: dict[int, int | None] = (
        _batch_resolve_species_taxids(cand_org_tids) if cand_org_tids else {}
    )
    if target_sp_taxid is not None:
        tid_to_sp[target_sp_taxid] = target_sp_taxid
    target_sp_norm = target_sp_name.lower()

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
        rep_org_name = str(org.get("organism_name") or "").strip()
        rep_tid_raw = org.get("tax_id")
        rep_tid = int(rep_tid_raw) if (rep_tid_raw is not None and str(rep_tid_raw).isdigit()) else None

        # Strictly enforce species-level match so congeneric species (e.g. Ca. L. solanacearum vs Ca. L. asiaticus) never leak in
        species_matches = True
        if target_sp_taxid is not None and rep_tid is not None and rep_tid in tid_to_sp:
            species_matches = tid_to_sp[rep_tid] == target_sp_taxid
        elif target_sp_norm and rep_org_name:
            cand_sp = _parse_species_binomial(rep_org_name).lower()
            species_matches = bool(cand_sp) and cand_sp == target_sp_norm
        elif target_sp_taxid is not None and rep_tid is not None and not tax_queried:
            species_matches = rep_tid == target_sp_taxid
        if not species_matches:
            continue

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
                "species_tax_id": target_sp_taxid if target_sp_taxid is not None else rep_tid,
                "organism_name": rep_org_name,
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
    sess.headers["User-Agent"] = "focaloop/0.4.0"
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
