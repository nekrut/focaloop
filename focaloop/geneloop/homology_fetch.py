"""Automatic related-species protein homology retrieval for GeneLoop (NCBI RefSeq & OrthoDB v12)."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import re
from typing import Any, Callable
import urllib.parse

from ..cvmfs_annot import infer_species_from_cached_meta
from ..ncbi_ref import (
    _NCBI_ACC_RE,
    _NCBI_DATASETS_API,
    _download_file,
    _fetch_json,
    _ipv4_session,
    lookup_ncbi_assembly_taxon,
    resolve_ncbi_assembly_dir,
)

_NCBI_TAXONOMY_API = "https://api.ncbi.nlm.nih.gov/datasets/v2/taxonomy/taxon"
_ORTHODB_V12_API = "https://data.orthodb.org/v12"
_GREIFSWALD_ODB12_BASE = "https://bioinf.uni-greifswald.de/bioinf/partitioned_odb12"

CLADE_HOMOLOGY_DEFAULTS: dict[str, dict[str, str]] = {
    "vertebrates": {"taxid": "7742", "taxon_name": "Vertebrata", "partition": "Vertebrata"},
    "mammalia_softmasking_v2": {"taxid": "40674", "taxon_name": "Mammalia", "partition": "Vertebrata"},
    "mammalia_unmasked": {"taxid": "40674", "taxon_name": "Mammalia", "partition": "Vertebrata"},
    "mammalia_nosoftmasking_v2": {"taxid": "40674", "taxon_name": "Mammalia", "partition": "Vertebrata"},
    "mammalia_nosofttmasking_v2": {"taxid": "40674", "taxon_name": "Mammalia", "partition": "Vertebrata"},
    "insecta": {"taxid": "50557", "taxon_name": "Insecta", "partition": "Arthropoda"},
    "angiosperms": {"taxid": "3398", "taxon_name": "Magnoliopsida", "partition": "Viridiplantae"},
    "fungi": {"taxid": "4751", "taxon_name": "Fungi", "partition": "Fungi"},
    "chlorophyta": {"taxid": "3041", "taxon_name": "Chlorophyta", "partition": "Viridiplantae"},
    "diatoms": {"taxid": "33634", "taxon_name": "Stramenopiles", "partition": "Stramenopiles"},
}

VALID_ODB12_PARTITIONS = {
    "Vertebrata",
    "Arthropoda",
    "Viridiplantae",
    "Fungi",
    "Stramenopiles",
    "Alveolata",
    "Amoebozoa",
    "Euglenozoa",
    "Metazoa",
    "Eukaryota",
}


def _emit(log: Callable[[str], None] | None, msg: str) -> None:
    if log is not None:
        log(msg)


def _binomial(name: str) -> str:
    parts = [p for p in str(name or "").strip().split() if p]
    return f"{parts[0]} {parts[1]}" if len(parts) >= 2 else (parts[0] if parts else "")


def _genus_of(name: str) -> str:
    parts = [p for p in str(name or "").strip().split() if p]
    return parts[0] if parts else ""


def resolve_taxon_info(taxon_query: str) -> dict[str, Any]:
    """Resolve a taxon name or numeric ID to ``{tax_id, organism_name, rank, family_taxid, order_taxid}``."""
    q = str(taxon_query or "").strip()
    if not q:
        return {}
    enc = urllib.parse.quote(q, safe="")
    payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{enc}")
    nodes = payload.get("taxonomy_nodes") or []
    if not nodes:
        return {}
    tax = (nodes[0] or {}).get("taxonomy") or {}
    tid = tax.get("tax_id")
    if not tid:
        return {}
    out: dict[str, Any] = {
        "tax_id": str(tid),
        "organism_name": str(tax.get("organism_name") or q),
        "rank": str(tax.get("rank") or "").upper(),
        "family_taxid": "",
        "family_name": "",
        "order_taxid": "",
        "order_name": "",
    }
    lineage = [str(x) for x in (tax.get("lineage") or []) if str(x).isdigit() and int(x) > 1]
    if lineage:
        try:
            lin_csv = urllib.parse.quote(",".join(lineage[-25:]), safe=",")
            lin_payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{lin_csv}")
            for lnode in lin_payload.get("taxonomy_nodes") or []:
                lt = (lnode or {}).get("taxonomy") or {}
                lrank = str(lt.get("rank") or "").upper()
                if lrank == "FAMILY" and lt.get("tax_id"):
                    out["family_taxid"] = str(lt["tax_id"])
                    out["family_name"] = str(lt.get("organism_name") or "")
                elif lrank == "ORDER" and lt.get("tax_id"):
                    out["order_taxid"] = str(lt["tax_id"])
                    out["order_name"] = str(lt.get("organism_name") or "")
        except Exception:
            pass
    return out


def infer_target_context(
    *,
    ncbi_accession: str = "",
    genome_dbkey: str = "",
    genome_name: str = "",
    genome_value: str = "",
) -> dict[str, str]:
    """Infer target species, genus, assembly accession, family taxid, and order taxid when available."""
    ctx = {
        "accession": "",
        "species_name": "",
        "genus": "",
        "family_taxid": "",
        "family_name": "",
        "order_taxid": "",
        "order_name": "",
    }
    acc = str(ncbi_accession or "").strip().upper()
    if acc and _NCBI_ACC_RE.match(acc):
        ctx["accession"] = acc
        try:
            meta = lookup_ncbi_assembly_taxon(acc)
            ctx["species_name"] = _binomial(meta.get("species_name") or meta.get("organism_name") or "")
        except Exception:
            pass
    elif any(str(x or "").strip() for x in (genome_dbkey, genome_name, genome_value)):
        try:
            inferred = infer_species_from_cached_meta(
                dbkey=genome_dbkey, display_name=genome_name, value=genome_value
            )
            ctx["accession"] = str(inferred.get("ncbi_accession_hint") or "").upper()
            ctx["species_name"] = _binomial(inferred.get("species_name") or "")
            if not ctx["species_name"] and ctx["accession"]:
                meta = lookup_ncbi_assembly_taxon(ctx["accession"])
                ctx["species_name"] = _binomial(meta.get("species_name") or meta.get("organism_name") or "")
        except Exception:
            pass

    ctx["genus"] = _genus_of(ctx["species_name"])
    if ctx["species_name"]:
        try:
            tinfo = resolve_taxon_info(ctx["species_name"])
            for k in ("family_taxid", "family_name", "order_taxid", "order_name"):
                ctx[k] = str(tinfo.get(k) or "")
        except Exception:
            pass
    return ctx


def _open_faa_text(path: Path):
    with open(path, "rb") as raw:
        magic = raw.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _append_clean_fasta(src: Path, dst_fh, prefix_tag: str = "") -> int:
    """Append non-empty FASTA lines to ``dst_fh``, optionally prefixing sequence IDs for uniqueness."""
    n_seqs = 0
    with _open_faa_text(src) as fh:
        for raw in fh:
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                n_seqs += 1
                hdr = ln[1:].strip()
                if prefix_tag:
                    parts = hdr.split(None, 1)
                    sid = f"{prefix_tag}|{parts[0]}"
                    rest = f" {parts[1]}" if len(parts) > 1 else ""
                    dst_fh.write(f">{sid}{rest}\n")
                else:
                    dst_fh.write(f">{hdr}\n")
            else:
                dst_fh.write(ln + "\n")
    return n_seqs


def fetch_ncbi_assembly_proteins(accession: str, dest_dir: str | Path) -> dict[str, Any]:
    """Download ``{asm_dir}_protein.faa.gz`` from NCBI FTP for a RefSeq/GenBank assembly accession."""
    asm_dir, base_url = resolve_ncbi_assembly_dir(accession)
    cache_env = os.environ.get("FOCALOOP_NCBI_CACHE_DIR", "").strip()
    root = (Path(cache_env) / asm_dir) if cache_env else (Path(dest_dir) / asm_dir)
    root.mkdir(parents=True, exist_ok=True)
    faa_gz = root / f"{asm_dir}_protein.faa.gz"
    if not (faa_gz.is_file() and faa_gz.stat().st_size > 0):
        _download_file(f"{base_url}/{asm_dir}_protein.faa.gz", faa_gz)
    return {
        "accession": accession.strip().upper(),
        "assembly_dir": asm_dir,
        "protein_faa": faa_gz,
    }


def _discover_species_diverse_refseq_proteomes(
    taxon_queries: list[str],
    *,
    max_species: int,
    exclude_accessions: set[str],
    exclude_species: set[str],
    exclude_genus: str = "",
) -> list[dict[str, str]]:
    """Discover up to ``max_species`` annotated RefSeq assemblies across distinct species and genera."""
    level_rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
    excl_base = set()
    for a in exclude_accessions:
        b = str(a or "").strip().upper().split(".")[0]
        if b:
            excl_base.add(b)
            if b.startswith("GCA_"):
                excl_base.add("GCF_" + b[4:])
            elif b.startswith("GCF_"):
                excl_base.add("GCA_" + b[4:])
    excl_sp = {s.lower() for s in exclude_species if s}
    excl_gen = exclude_genus.strip().lower()

    chosen: list[dict[str, str]] = []
    seen_species: set[str] = set(excl_sp)
    seen_base: set[str] = set(excl_base)

    for tq in taxon_queries:
        q = str(tq or "").strip()
        if not q or len(chosen) >= max_species:
            continue
        enc = urllib.parse.quote(q, safe="")
        url = f"{_NCBI_DATASETS_API}/taxon/{enc}/dataset_report"
        try:
            payload = _fetch_json(
                url,
                params={
                    "filters.assembly_source": "refseq",
                    "filters.has_annotation": "true",
                    "filters.exclude_paired_reports": "true",
                    "page_size": 100,
                },
            )
        except Exception:
            continue
        reps = list(payload.get("reports") or [])

        def _skey(rep: dict[str, Any]) -> tuple[int, int, str]:
            ai = rep.get("assembly_info") or {}
            cat = str(ai.get("refseq_category") or "").lower()
            cr = 0 if "reference" in cat else (1 if "representative" in cat else 2)
            lr = level_rank.get(str(ai.get("assembly_level") or ""), 4)
            return (cr, lr, str(rep.get("accession") or ""))

        reps.sort(key=_skey)

        # Pass 1: distinct genera first, Pass 2: allow additional species in already-represented genera
        for require_new_genus in (True, False):
            used_genera = {_genus_of(c["species_name"]).lower() for c in chosen}
            for rep in reps:
                if len(chosen) >= max_species:
                    break
                acc = str(rep.get("accession") or "").strip().upper()
                if not acc.startswith("GCF_") or not _NCBI_ACC_RE.match(acc):
                    continue
                bacc = acc.split(".")[0]
                if bacc in seen_base:
                    continue
                org = rep.get("organism") or {}
                sp = _binomial(str(org.get("organism_name") or ""))
                if not sp or sp.lower() in seen_species:
                    continue
                gen = _genus_of(sp).lower()
                if excl_gen and gen == excl_gen:
                    continue
                if require_new_genus and gen in used_genera:
                    continue
                seen_base.add(bacc)
                seen_species.add(sp.lower())
                used_genera.add(gen)
                chosen.append({"accession": acc, "species_name": sp})
    return chosen


def fetch_ncbi_related_proteomes(
    *,
    clade: str,
    dest_dir: str | Path,
    protein_query: str = "",
    max_species: int = 3,
    target_ctx: dict[str, str] | None = None,
    exclude_genus: str = "",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch and merge related-species RefSeq proteomes from NCBI into a clean FASTA file."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ctx = target_ctx or {}
    clade_def = CLADE_HOMOLOGY_DEFAULTS.get(clade, CLADE_HOMOLOGY_DEFAULTS["vertebrates"])
    max_sp = max(1, min(15, int(max_species)))

    raw_q = str(protein_query or "").strip()
    tokens = [t.strip() for t in re.split(r"[\s,;]+", raw_q) if t.strip()]
    explicit_accs = [t.upper() for t in tokens if _NCBI_ACC_RE.match(t)]

    selected: list[dict[str, str]] = []
    if explicit_accs:
        for a in dict.fromkeys(explicit_accs):
            selected.append({"accession": a, "species_name": a})
    else:
        if raw_q:
            taxon_chain = [raw_q]
        else:
            taxon_chain = [
                t
                for t in (ctx.get("family_taxid"), ctx.get("order_taxid"), clade_def["taxid"])
                if t
            ]
        excl_accs = {ctx["accession"]} if ctx.get("accession") else set()
        excl_sp = {ctx["species_name"]} if ctx.get("species_name") else set()
        selected = _discover_species_diverse_refseq_proteomes(
            taxon_chain,
            max_species=max_sp,
            exclude_accessions=excl_accs,
            exclude_species=excl_sp,
            exclude_genus=exclude_genus,
        )
        if not selected and excl_sp:
            # If the user's taxon was a single species matching target, allow fallback without excluding species
            selected = _discover_species_diverse_refseq_proteomes(
                taxon_chain,
                max_species=max_sp,
                exclude_accessions=excl_accs,
                exclude_species=set(),
                exclude_genus=exclude_genus,
            )

    if not selected:
        raise ValueError(
            f"No annotated RefSeq proteomes found on NCBI for query={raw_q or clade_def['taxon_name']!r}. "
            "Specify a broader taxonomic group (for example Diptera, Brassicaceae, or Fungi) or supply a protein FASTA from History."
        )

    merged_faa = dest / "ncbi_related_proteins.faa"
    total_seqs = 0
    used_records: list[dict[str, Any]] = []
    with open(merged_faa, "w", encoding="utf-8") as out_fh:
        for item in selected:
            acc = item["accession"]
            _emit(log, f"[focaloop-geneloop] Downloading NCBI RefSeq proteome {acc} ({item['species_name']})...")
            dl = fetch_ncbi_assembly_proteins(acc, dest / "ncbi_proteomes")
            n_s = _append_clean_fasta(dl["protein_faa"], out_fh, prefix_tag=acc)
            total_seqs += n_s
            used_records.append({"accession": acc, "species_name": item["species_name"], "proteins": n_s})

    if total_seqs == 0:
        raise RuntimeError("Downloaded NCBI RefSeq proteome archives contained 0 protein sequences.")

    return {
        "source": "ncbi_refseq",
        "proteins_fasta": merged_faa,
        "n_species": len(used_records),
        "n_sequences": total_seqs,
        "species": used_records,
    }


def _find_cached_orthodb_partition(partition: str) -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get("FOCALOOP_ORTHODB_CACHE_DIR", "").strip()
    if env_dir:
        candidates.extend([Path(env_dir) / f"{partition}.fa.gz", Path(env_dir) / f"{partition}.fa"])
    for base in ("/opt/focaloop/orthodb", "/cvmfs/data.galaxyproject.org/byhand/orthodb"):
        candidates.extend([Path(base) / f"{partition}.fa.gz", Path(base) / f"{partition}.fa"])
    for p in candidates:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def fetch_orthodb_clade_proteins(
    *,
    clade: str,
    dest_dir: str | Path,
    protein_query: str = "",
    max_species: int = 5,
    orthodb_mode: str = "representatives",
    target_ctx: dict[str, str] | None = None,
    exclude_genus: str = "",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch OrthoDB v12 homology proteins (local/CVMFS partition cache, representative species API, or full partition)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ctx = target_ctx or {}
    clade_def = CLADE_HOMOLOGY_DEFAULTS.get(clade, CLADE_HOMOLOGY_DEFAULTS["vertebrates"])
    raw_q = str(protein_query or "").strip()

    partition = raw_q if raw_q in VALID_ODB12_PARTITIONS else clade_def["partition"]

    # 1. Reuse local/CVMFS pre-staged partitioned OrthoDB archive immediately when available and no custom taxon override
    if not raw_q or raw_q in VALID_ODB12_PARTITIONS:
        cached_p = _find_cached_orthodb_partition(partition)
        if cached_p is not None:
            _emit(log, f"[focaloop-geneloop] Using cached OrthoDB v12 partition {cached_p}...")
            return {
                "source": f"orthodb_cached:{partition}",
                "proteins_fasta": cached_p,
                "n_species": -1,
                "n_sequences": -1,
                "species": [{"partition": partition, "path": str(cached_p)}],
            }

    # 2. Explicit full Greifswald/BRAKER3 partitioned clade archive download
    if orthodb_mode == "full_partition":
        cache_env = os.environ.get("FOCALOOP_ORTHODB_CACHE_DIR", "").strip()
        dl_dir = Path(cache_env) if cache_env else (dest / "orthodb_partitions")
        dl_dir.mkdir(parents=True, exist_ok=True)
        out_gz = dl_dir / f"{partition}.fa.gz"
        if not (out_gz.is_file() and out_gz.stat().st_size > 0):
            url = f"{_GREIFSWALD_ODB12_BASE}/{partition}.fa.gz"
            _emit(log, f"[focaloop-geneloop] Downloading OrthoDB v12 partitioned clade archive {partition}.fa.gz...")
            _download_file(url, out_gz, timeout=600.0)
        return {
            "source": f"orthodb_partition:{partition}",
            "proteins_fasta": out_gz,
            "n_species": -1,
            "n_sequences": -1,
            "species": [{"partition": partition, "path": str(out_gz)}],
        }

    # 3. Representative diverse OrthoDB v12 clade proteomes via OrthoDB REST API
    max_sp = max(1, min(20, int(max_species)))
    level_candidates: list[str] = []
    if raw_q and raw_q not in VALID_ODB12_PARTITIONS:
        if raw_q.isdigit():
            level_candidates.append(raw_q)
        else:
            tinfo = resolve_taxon_info(raw_q)
            if tinfo.get("tax_id"):
                level_candidates.append(str(tinfo["tax_id"]))
    else:
        for tid in (ctx.get("family_taxid"), ctx.get("order_taxid"), clade_def["taxid"]):
            if tid and tid not in level_candidates:
                level_candidates.append(tid)

    excl_sp = (ctx.get("species_name") or "").strip().lower()
    excl_gen = exclude_genus.strip().lower()
    excl_acc = (ctx.get("accession") or "").strip().upper().split(".")[0]

    sess = _ipv4_session()
    chosen: list[dict[str, str]] = []
    seen_sp: set[str] = {excl_sp} if excl_sp else set()

    for lvl in level_candidates:
        if len(chosen) >= max_sp:
            break
        try:
            r = sess.get(f"{_ORTHODB_V12_API}/species", params={"level": lvl}, timeout=45.0)
            r.raise_for_status()
            sp_list = list((r.json() or {}).get("data") or [])
        except Exception:
            continue

        # Sort GCF_ RefSeq assemblies ahead of GCA_ GenBank assemblies for cleanest gene models
        sp_list.sort(
            key=lambda d: (
                0 if str(d.get("organism_id") or "").upper().startswith("GCF_") else 1,
                str(d.get("sciname") or ""),
            )
        )

        for require_new_genus in (True, False):
            used_genera = {_genus_of(c["species_name"]).lower() for c in chosen}
            for entry in sp_list:
                if len(chosen) >= max_sp:
                    break
                odb_id = str(entry.get("taxon_id") or "").strip()
                sci = _binomial(str(entry.get("sciname") or ""))
                org_acc = str(entry.get("organism_id") or "").strip().upper().split(".")[0]
                if not odb_id or not sci or sci.lower() in seen_sp:
                    continue
                if excl_acc and org_acc and org_acc == excl_acc:
                    continue
                gen = _genus_of(sci).lower()
                if excl_gen and gen == excl_gen:
                    continue
                if require_new_genus and gen in used_genera:
                    continue
                seen_sp.add(sci.lower())
                used_genera.add(gen)
                chosen.append({"odb_taxon_id": odb_id, "species_name": sci, "assembly": str(entry.get("organism_id") or "")})

    if not chosen:
        raise ValueError(
            f"OrthoDB v12 returned no matching representative proteomes for query={raw_q or clade_def['taxon_name']!r}. "
            "Try a broader taxonomic name or switch to NCBI RefSeq proteomes."
        )

    merged_faa = dest / "orthodb_v12_proteins.faa"
    total_seqs = 0
    used_records: list[dict[str, Any]] = []
    sp_dir = dest / "orthodb_species"
    sp_dir.mkdir(parents=True, exist_ok=True)
    with open(merged_faa, "w", encoding="utf-8") as out_fh:
        for item in chosen:
            oid = item["odb_taxon_id"]
            _emit(log, f"[focaloop-geneloop] Downloading OrthoDB v12 proteome {oid} ({item['species_name']})...")
            sp_fa = sp_dir / f"{oid}.faa"
            if not (sp_fa.is_file() and sp_fa.stat().st_size > 0):
                url = f"{_ORTHODB_V12_API}/fasta?species={urllib.parse.quote(oid, safe='')}"
                _download_file(url, sp_fa, timeout=180.0)
            n_s = _append_clean_fasta(sp_fa, out_fh)
            sp_fa.unlink(missing_ok=True)
            total_seqs += n_s
            used_records.append({**item, "proteins": n_s})

    if total_seqs == 0:
        raise RuntimeError("OrthoDB v12 FASTA stream returned 0 protein sequences.")

    return {
        "source": "orthodb_v12_representatives",
        "proteins_fasta": merged_faa,
        "n_species": len(used_records),
        "n_sequences": total_seqs,
        "species": used_records,
    }
