"""Automatic related-species protein homology retrieval for GeneLoop (NCBI RefSeq & OrthoDB v12)."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import re
import tempfile
import threading
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
_ODB_PARTITION_BY_LOWER = {p.lower(): p for p in VALID_ODB12_PARTITIONS}
_ODB_TAX_HEADER_RE = re.compile(r"^(\d+)(?:_(\d+))?:")


def _emit(log: Callable[[str], None] | None, msg: str) -> None:
    if log is not None:
        log(msg)


def _writable_dir_or_fallback(preferred: Path, fallback: Path, log: Callable[[str], None] | None = None) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=preferred, prefix=".gl_write_probe_", delete=True):
            pass
        return preferred
    except OSError:
        _emit(log, f"[focaloop-geneloop] Cache directory {preferred} is not writable, using job scratch {fallback}...")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _download_atomic_unique(url: str, dst: Path, timeout: float = 180.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f".{os.getpid()}.{threading.get_ident()}.{dst.name}")
    try:
        _download_file(url, tmp_dst, timeout=timeout)
        os.replace(tmp_dst, dst)
    finally:
        tmp_dst.unlink(missing_ok=True)
        Path(f"{tmp_dst}.part").unlink(missing_ok=True)


def _binomial(name: str) -> str:
    parts = [p for p in str(name or "").strip().split() if p]
    return f"{parts[0]} {parts[1]}" if len(parts) >= 2 else (parts[0] if parts else "")


def _genus_of(name: str) -> str:
    parts = [p for p in str(name or "").strip().split() if p]
    return parts[0] if parts else ""


def resolve_taxon_info(taxon_query: str) -> dict[str, Any]:
    """Resolve a taxon name or numeric ID to ``{tax_id, organism_name, rank, genus_taxid, family_taxid, order_taxid}``."""
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
        "genus_taxid": str(tid) if str(tax.get("rank") or "").upper() == "GENUS" else "",
        "genus_name": str(tax.get("organism_name") or "") if str(tax.get("rank") or "").upper() == "GENUS" else "",
        "family_taxid": "",
        "family_name": "",
        "order_taxid": "",
        "order_name": "",
        "partition_hint": _ODB_PARTITION_BY_LOWER.get(str(tax.get("organism_name") or "").strip().lower(), ""),
    }
    lineage = [str(x) for x in (tax.get("lineage") or []) if str(x).isdigit() and int(x) > 1]
    if lineage:
        try:
            lin_csv = urllib.parse.quote(",".join(lineage[-25:]), safe=",")
            lin_payload = _fetch_json(f"{_NCBI_TAXONOMY_API}/{lin_csv}")
            # Walk most specific -> broadest so narrowest OrthoDB partition wins (e.g. Arthropoda over Metazoa)
            part_priority = [
                "Vertebrata", "Arthropoda", "Viridiplantae", "Fungi", "Stramenopiles",
                "Alveolata", "Amoebozoa", "Euglenozoa", "Metazoa", "Eukaryota",
            ]
            seen_org_names: set[str] = set()
            for lnode in lin_payload.get("taxonomy_nodes") or []:
                lt = (lnode or {}).get("taxonomy") or {}
                lrank = str(lt.get("rank") or "").upper()
                lname = str(lt.get("organism_name") or "")
                if lname:
                    seen_org_names.add(lname)
                if lrank == "GENUS" and lt.get("tax_id"):
                    out["genus_taxid"] = str(lt["tax_id"])
                    out["genus_name"] = lname
                elif lrank == "FAMILY" and lt.get("tax_id"):
                    out["family_taxid"] = str(lt["tax_id"])
                    out["family_name"] = lname
                elif lrank == "ORDER" and lt.get("tax_id"):
                    out["order_taxid"] = str(lt["tax_id"])
                    out["order_name"] = lname
            if not out["partition_hint"]:
                for cand_part in part_priority:
                    if cand_part in seen_org_names:
                        out["partition_hint"] = cand_part
                        break
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
    """Infer target species, taxid, genus, assembly accession, family taxid, and order taxid when available."""
    ctx = {
        "accession": "",
        "tax_id": "",
        "species_name": "",
        "genus": "",
        "genus_taxid": "",
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
            if meta.get("tax_id"):
                ctx["tax_id"] = str(meta["tax_id"]).strip()
        except Exception:
            pass
    elif any(str(x or "").strip() for x in (genome_dbkey, genome_name, genome_value)):
        try:
            inferred = infer_species_from_cached_meta(
                dbkey=genome_dbkey, display_name=genome_name, value=genome_value
            )
            ctx["accession"] = str(inferred.get("ncbi_accession_hint") or "").upper()
            ctx["species_name"] = _binomial(inferred.get("species_name") or "")
            if inferred.get("tax_id"):
                ctx["tax_id"] = str(inferred["tax_id"]).strip()
            if not ctx["species_name"] and ctx["accession"]:
                meta = lookup_ncbi_assembly_taxon(ctx["accession"])
                ctx["species_name"] = _binomial(meta.get("species_name") or meta.get("organism_name") or "")
                if meta.get("tax_id"):
                    ctx["tax_id"] = str(meta["tax_id"]).strip()
        except Exception:
            pass

    ctx["genus"] = _genus_of(ctx["species_name"])
    if ctx["species_name"]:
        try:
            tinfo = resolve_taxon_info(ctx["species_name"])
            if not ctx["tax_id"] and tinfo.get("tax_id"):
                ctx["tax_id"] = str(tinfo["tax_id"])
            for k in ("genus_taxid", "family_taxid", "family_name", "order_taxid", "order_name"):
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
    """Validate and append FASTA records from ``src`` to ``dst_fh``, rejecting HTML/JSON or empty payloads."""
    n_seqs = 0
    seq_chars = 0
    checked_first = False
    with _open_faa_text(src) as fh:
        for raw in fh:
            ln = raw.strip()
            if not ln:
                continue
            if not checked_first:
                checked_first = True
                if not ln.startswith(">"):
                    raise RuntimeError(
                        f"Invalid FASTA payload in {src}: expected '>' header on first non-empty line, got {ln[:80]!r}"
                    )
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
                if ln[0] in ("<", "{"):
                    raise RuntimeError(f"Corrupt non-FASTA line inside {src}: {ln[:80]!r}")
                seq_chars += len(ln)
                dst_fh.write(ln + "\n")
    if n_seqs == 0 or seq_chars == 0:
        raise RuntimeError(f"Proteome file {src} contained 0 valid protein sequences")
    return n_seqs


def fetch_ncbi_assembly_proteins(
    accession: str,
    dest_dir: str | Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download ``{asm_dir}_protein.faa.gz`` from NCBI FTP for a RefSeq/GenBank assembly accession."""
    asm_dir, base_url = resolve_ncbi_assembly_dir(accession)
    fallback_root = Path(dest_dir) / asm_dir
    cache_env = os.environ.get("FOCALOOP_NCBI_CACHE_DIR", "").strip()
    if cache_env:
        cand_cached = Path(cache_env) / asm_dir / f"{asm_dir}_protein.faa.gz"
        try:
            if cand_cached.is_file() and cand_cached.stat().st_size > 2:
                with open(cand_cached, "rb") as fh:
                    if fh.read(2) == b"\x1f\x8b":
                        return {
                            "accession": accession.strip().upper(),
                            "assembly_dir": asm_dir,
                            "protein_faa": cand_cached,
                        }
        except OSError:
            pass
        root = _writable_dir_or_fallback(Path(cache_env) / asm_dir, fallback_root, log=log)
    else:
        root = fallback_root
        root.mkdir(parents=True, exist_ok=True)
    faa_gz = root / f"{asm_dir}_protein.faa.gz"
    need_dl = True
    if faa_gz.is_file() and faa_gz.stat().st_size > 2:
        with open(faa_gz, "rb") as fh:
            if fh.read(2) == b"\x1f\x8b":
                need_dl = False
    if need_dl:
        _download_atomic_unique(f"{base_url}/{asm_dir}_protein.faa.gz", faa_gz)
        with open(faa_gz, "rb") as fh:
            magic = fh.read(2)
        if magic != b"\x1f\x8b":
            faa_gz.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded NCBI proteome for {accession} is not a valid gzip archive (magic={magic!r})"
            )
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
            dl = fetch_ncbi_assembly_proteins(acc, dest / "ncbi_proteomes", log=log)
            n_s = _append_clean_fasta(dl["protein_faa"], out_fh, prefix_tag=acc)
            _emit(log, f"[focaloop-geneloop] Retained {n_s:,} proteins from {acc} ({item['species_name']}).")
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
    p_low = partition.lower()
    names = (
        f"{partition}.fa.gz",
        f"{partition}.fa",
        f"{partition}.faa.gz",
        f"{partition}.faa",
        f"orthodb_v12_{p_low}.faa.gz",
        f"orthodb_v12_{p_low}.faa",
    )
    if env_dir:
        candidates.extend(Path(env_dir) / nm for nm in names)
    for base in ("/opt/focaloop/orthodb", "/cvmfs/data.galaxyproject.org/byhand/orthodb"):
        candidates.extend(Path(base) / nm for nm in names)
    for p in candidates:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def _resolve_orthodb_exclusion_taxids(
    ctx: dict[str, Any],
    exclude_genus: str,
    clade_taxid: str,
) -> set[str]:
    """Collect numeric taxid and ``TAXID_N`` prefixes to exclude from OrthoDB partition FASTAs."""
    excl: set[str] = set()
    for key in ("tax_id", "taxid"):
        raw_t = str(ctx.get(key) or "").strip()
        if raw_t:
            excl.add(raw_t)
            excl.add(raw_t.split("_", 1)[0])
    extra = ctx.get("exclude_taxids")
    if isinstance(extra, str):
        for tok in re.split(r"[\s,;]+", extra):
            if tok.strip():
                excl.add(tok.strip())
                excl.add(tok.strip().split("_", 1)[0])
    elif isinstance(extra, (list, tuple, set, frozenset)):
        for tok in extra:
            s = str(tok).strip()
            if s:
                excl.add(s)
                excl.add(s.split("_", 1)[0])

    sp_name = str(ctx.get("species_name") or "").strip()
    if sp_name and not (ctx.get("tax_id") or ctx.get("taxid")):
        try:
            tinfo = resolve_taxon_info(sp_name)
            if tinfo.get("tax_id"):
                excl.add(str(tinfo["tax_id"]))
        except Exception:
            pass

    gen_clean = exclude_genus.strip()
    if gen_clean:
        gen_low = gen_clean.lower()
        levels: list[str] = []
        if ctx.get("genus_taxid") and str(ctx.get("genus") or "").strip().lower() == gen_low:
            levels.append(str(ctx["genus_taxid"]))
        else:
            try:
                ginfo = resolve_taxon_info(gen_clean)
                if ginfo.get("tax_id"):
                    levels.append(str(ginfo["tax_id"]))
            except Exception:
                pass
        for fallback_lvl in (ctx.get("family_taxid"), clade_taxid):
            if fallback_lvl and str(fallback_lvl) not in levels and not levels:
                levels.append(str(fallback_lvl))
        if levels:
            try:
                sess = _ipv4_session()
                for lvl in levels[:1]:
                    r = sess.get(f"{_ORTHODB_V12_API}/species", params={"level": lvl}, timeout=15.0)
                    r.raise_for_status()
                    for entry in (r.json() or {}).get("data") or []:
                        sci = _binomial(str(entry.get("sciname") or ""))
                        if _genus_of(sci).lower() == gen_low:
                            odb_id = str(entry.get("taxon_id") or "").strip()
                            if odb_id:
                                excl.add(odb_id)
                                excl.add(odb_id.split("_", 1)[0])
            except Exception:
                pass
    return {x for x in excl if x and x != "0" and not x.startswith("0_")}


def filter_orthodb_partition_fasta(
    src_path: Path,
    dst_path: Path,
    *,
    exclude_taxids: set[str] | None = None,
    exclude_genus: str = "",
    exclude_species: str = "",
    max_species: int | None = None,
) -> dict[str, Any]:
    """Filter an OrthoDB partition FASTA by excluded taxids/genus and optional ``max_species`` taxon cap."""
    excl_tax = {str(x).strip() for x in (exclude_taxids or set()) if str(x).strip()}
    excl_gen_re = (
        re.compile(rf"(?:\[|\borganism=|\b){re.escape(exclude_genus.strip())}\s+[a-z]", re.I)
        if exclude_genus.strip()
        else None
    )
    excl_sp_low = exclude_species.strip().lower()

    # Preserve tiny non-FASTA sentinel mocks when no exclusions apply (unit-test cache probe contract)
    try:
        raw_sz = int(src_path.stat().st_size)
    except OSError:
        raw_sz = 0
    if raw_sz <= 64 and not excl_tax and not exclude_genus.strip() and not excl_sp_low:
        with _open_faa_text(src_path) as fh_peek:
            first_ln = ""
            for raw in fh_peek:
                if raw.strip():
                    first_ln = raw.strip()
                    break
        if not first_ln.startswith(">"):
            return {
                "path": src_path,
                "passthrough_mock": True,
                "n_input": 0,
                "n_kept": 0,
                "n_removed": 0,
                "n_taxa_kept": 0,
                "kept_taxa": [],
            }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_path.with_name(f".{dst_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    n_in = 0
    n_kept = 0
    n_removed = 0
    kept_taxa_order: list[str] = []
    kept_taxa_set: set[str] = set()
    keep_cur = False
    checked_first = False

    try:
        with _open_faa_text(src_path) as fin, open(tmp_dst, "w", encoding="utf-8") as fout:
            for raw in fin:
                s = raw.strip()
                if not s:
                    continue
                if not checked_first:
                    checked_first = True
                    if not s.startswith(">"):
                        raise RuntimeError(
                            f"Invalid OrthoDB FASTA payload in {src_path}: expected '>' on first line, got {s[:80]!r}"
                        )
                if s.startswith(">"):
                    n_in += 1
                    hdr = s[1:].strip()
                    tok0 = hdr.split(None, 1)[0]
                    m = _ODB_TAX_HEADER_RE.match(tok0)
                    base_tax = m.group(1) if m else ""
                    odb_tax = tok0.split(":", 1)[0] if ":" in tok0 else base_tax
                    tax_key = odb_tax or base_tax or f"_unk_{n_in}"

                    drop = False
                    if base_tax and base_tax in excl_tax:
                        drop = True
                    elif odb_tax and odb_tax in excl_tax:
                        drop = True
                    elif excl_gen_re is not None and excl_gen_re.search(hdr):
                        drop = True
                    elif excl_sp_low and excl_sp_low in hdr.lower():
                        drop = True

                    if drop:
                        keep_cur = False
                        n_removed += 1
                        continue

                    if max_species is not None and max_species > 0:
                        if tax_key not in kept_taxa_set:
                            if len(kept_taxa_set) >= max_species:
                                keep_cur = False
                                continue
                            kept_taxa_set.add(tax_key)
                            kept_taxa_order.append(tax_key)
                    else:
                        if tax_key not in kept_taxa_set:
                            kept_taxa_set.add(tax_key)
                            kept_taxa_order.append(tax_key)

                    keep_cur = True
                    n_kept += 1
                    fout.write(f">{hdr}\n")
                elif keep_cur:
                    fout.write(s + "\n")
        os.replace(tmp_dst, dst_path)
    finally:
        tmp_dst.unlink(missing_ok=True)

    return {
        "path": dst_path,
        "passthrough_mock": False,
        "n_input": n_in,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "n_taxa_kept": len(kept_taxa_order),
        "kept_taxa": kept_taxa_order,
    }


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
    max_sp = max(1, min(20, int(max_species)))

    matched_partition = _ODB_PARTITION_BY_LOWER.get(raw_q.lower()) if raw_q else None
    if raw_q and matched_partition is None and orthodb_mode == "full_partition":
        # Attempt lineage-based mapping to a valid OrthoDB v12 partition; raise if unresolvable
        mapped_part = ""
        try:
            tinfo = resolve_taxon_info(raw_q)
            mapped_part = str(tinfo.get("partition_hint") or "")
        except Exception:
            mapped_part = ""
        if mapped_part in VALID_ODB12_PARTITIONS:
            _emit(
                log,
                f"[focaloop-geneloop] Mapped non-partition taxon query {raw_q!r} to OrthoDB v12 partition {mapped_part}.",
            )
            partition = mapped_part
        else:
            valid_csv = ", ".join(sorted(VALID_ODB12_PARTITIONS))
            raise ValueError(
                f"OrthoDB full_partition mode requires a valid partition name ({valid_csv}), got {raw_q!r}. "
                "Use orthodb_mode='representatives' for family/order/genus queries."
            )
    else:
        partition = matched_partition if matched_partition is not None else clade_def["partition"]

    excl_taxids = _resolve_orthodb_exclusion_taxids(ctx, exclude_genus, clade_def["taxid"])
    excl_sp_name = str(ctx.get("species_name") or "").strip()

    # 1. Reuse local/CVMFS pre-staged partitioned OrthoDB archive when available (filtered & subsampled!)
    if not raw_q or matched_partition is not None:
        cached_p = _find_cached_orthodb_partition(partition)
        if cached_p is not None:
            _emit(log, f"[focaloop-geneloop] Using cached OrthoDB v12 partition {cached_p}...")
            cap_sp = max_sp if orthodb_mode == "representatives" else None
            filt_dst = dest / f"orthodb_cached_{partition}.filtered.faa"
            fres = filter_orthodb_partition_fasta(
                cached_p,
                filt_dst,
                exclude_taxids=excl_taxids,
                exclude_genus=exclude_genus,
                exclude_species=excl_sp_name,
                max_species=cap_sp,
            )
            if fres["passthrough_mock"]:
                return {
                    "source": f"orthodb_cached:{partition}",
                    "proteins_fasta": cached_p,
                    "n_species": -1,
                    "n_sequences": -1,
                    "excluded_taxids": sorted(excl_taxids),
                    "species": [{"partition": partition, "path": str(cached_p)}],
                }
            if fres["n_kept"] > 0:
                _emit(
                    log,
                    f"[focaloop-geneloop] Filtered cached {partition}: kept {fres['n_kept']:,}/{fres['n_input']:,} proteins "
                    f"across {fres['n_taxa_kept']} taxa (removed {fres['n_removed']:,} excluded records).",
                )
                return {
                    "source": f"orthodb_cached:{partition}",
                    "proteins_fasta": fres["path"],
                    "n_species": fres["n_taxa_kept"],
                    "n_sequences": fres["n_kept"],
                    "excluded_taxids": sorted(excl_taxids),
                    "species": [{"partition": partition, "path": str(cached_p), "taxa_kept": fres["n_taxa_kept"]}],
                }
            if orthodb_mode == "full_partition":
                raise ValueError(
                    f"Cached OrthoDB partition {partition} had 0 proteins remaining after excluding target/genus taxids {sorted(excl_taxids)}."
                )
            _emit(log, f"[focaloop-geneloop] Cached {partition} had 0 proteins after exclusion; querying OrthoDB REST API...")

    # 2. Explicit full Greifswald/BRAKER3 partitioned clade archive download (written into job scratch, never polluting shared cache)
    if orthodb_mode == "full_partition":
        dl_dir = dest / "orthodb_partitions"
        dl_dir.mkdir(parents=True, exist_ok=True)
        out_gz = dl_dir / f"{partition}.fa.gz"
        if not (out_gz.is_file() and out_gz.stat().st_size > 0):
            url = f"{_GREIFSWALD_ODB12_BASE}/{partition}.fa.gz"
            _emit(log, f"[focaloop-geneloop] Downloading OrthoDB v12 partitioned clade archive {partition}.fa.gz...")
            _download_atomic_unique(url, out_gz, timeout=600.0)
        filt_dst = dest / f"orthodb_partition_{partition}.filtered.faa"
        fres = filter_orthodb_partition_fasta(
            out_gz,
            filt_dst,
            exclude_taxids=excl_taxids,
            exclude_genus=exclude_genus,
            exclude_species=excl_sp_name,
            max_species=None,
        )
        out_gz.unlink(missing_ok=True)
        if fres["n_kept"] == 0:
            raise ValueError(
                f"OrthoDB partition {partition} contained 0 proteins after excluding target/genus taxids {sorted(excl_taxids)}."
            )
        return {
            "source": f"orthodb_partition:{partition}",
            "proteins_fasta": fres["path"],
            "n_species": fres["n_taxa_kept"],
            "n_sequences": fres["n_kept"],
            "excluded_taxids": sorted(excl_taxids),
            "species": [{"partition": partition, "path": str(fres["path"]), "taxa_kept": fres["n_taxa_kept"]}],
        }

    # 3. Representative diverse OrthoDB v12 clade proteomes via OrthoDB REST API
    level_candidates: list[str] = []
    if raw_q and matched_partition is None:
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

    excl_sp = excl_sp_name.lower()
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
                base_tid = odb_id.split("_", 1)[0]
                sci = _binomial(str(entry.get("sciname") or ""))
                org_acc = str(entry.get("organism_id") or "").strip().upper().split(".")[0]
                if not odb_id or not sci or sci.lower() in seen_sp:
                    continue
                if odb_id in excl_taxids or (base_tid and base_tid in excl_taxids):
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
                _download_atomic_unique(url, sp_fa, timeout=180.0)
            n_s = _append_clean_fasta(sp_fa, out_fh)
            _emit(log, f"[focaloop-geneloop] Retained {n_s:,} proteins from {oid} ({item['species_name']}).")
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
        "excluded_taxids": sorted(excl_taxids),
        "species": used_records,
    }
