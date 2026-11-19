"""Resolve and verify matching RefSeq gene annotations for Galaxy CVMFS built-in genomes.

Two-tier strategy:
  1. UCSC-keyed builds (e.g. human, mouse, fruit fly, yeast, worm, zebrafish, rat):
     Verify exact chromosome-length agreement against UCSC chromInfo, then fetch
     UCSC's prebuilt NCBI RefSeq GTF (whose chromosome names already match CVMFS).
  2. Non-UCSC and microbial builds (e.g. Arabidopsis, Pseudomonas, E. coli):
     Infer species/strain from Galaxy's build metadata, discover candidate RefSeq
     assemblies via NCBI Datasets, select the assembly revision whose contig lengths
     match the cached FASTA to the exact base pair using NCBI assembly reports,
     download its GFF3, and remap RefSeq contig accessions to the FASTA contig names.

Fail-fast safety guarantee: never silently align to shifted coordinates or mismatched
contigs. If an older cached genome cannot be matched unambiguously within exact
contig lengths, raise ValueError with actionable instructions.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from .ncbi_ref import (
    _NCBI_ACC_RE,
    _NCBI_DATASETS_API,
    _NCBI_FTP_ALL,
    _download_file,
    _fetch_json,
    _fetch_text,
    _ipv4_session,
)

_UCSC_API = "https://api.genome.ucsc.edu"
_UCSC_GOLDENPATH = "https://hgdownload.soe.ucsc.edu/goldenPath"

_UCSC_GENOMES_CACHE: dict[str, dict[str, Any]] | None = None

_SPECIES_TYPOS = {
    "pseudomonas aeruginos": "Pseudomonas aeruginosa",
}

_MT_ALIASES = {"mt", "chrm", "m", "mito", "mitochondrion", "mtdna"}
_PLTD_ALIASES = {"pltd", "chrc", "c", "chloroplast", "plastid", "cp", "pt"}


def _emit(log: Callable[[str], None] | None, msg: str) -> None:
    if log is not None:
        log(msg)


def _norm_token(val: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(val or "").lower())


def read_fasta_contig_lengths(fasta_path: str | Path) -> dict[str, int]:
    """Return ``{contig_name: length_bp}`` using nearby ``.fai`` indexes first, else streaming FASTA."""
    fp = Path(fasta_path)
    fai_candidates = [
        Path(str(fp) + ".fai"),
        fp.parent.parent / "sam_index" / f"{fp.name}.fai",
        fp.parent.parent / "sam_index" / f"{fp.stem}.fa.fai",
    ]
    for fai in fai_candidates:
        try:
            if fai.is_file() and fai.stat().st_size > 0:
                lengths: dict[str, int] = {}
                with open(fai, "rt", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 2 and parts[1].isdigit():
                            lengths[parts[0]] = int(parts[1])
                if lengths:
                    return lengths
        except OSError:
            continue

    with open(fp, "rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    lengths = {}
    cur_name: str | None = None
    cur_len = 0
    with opener(fp, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line:
                continue
            if line[0] == ">":
                if cur_name is not None:
                    lengths[cur_name] = cur_len
                cur_name = line[1:].strip().split()[0]
                cur_len = 0
            else:
                cur_len += len(line.strip())
    if cur_name is not None:
        lengths[cur_name] = cur_len
    return lengths


def fetch_ucsc_genomes(timeout: float = 20.0) -> dict[str, dict[str, Any]]:
    """Fetch UCSC genome metadata keyed by dbkey (cached in-process)."""
    global _UCSC_GENOMES_CACHE
    if _UCSC_GENOMES_CACHE is not None:
        return _UCSC_GENOMES_CACHE
    payload = _fetch_json(f"{_UCSC_API}/list/ucscGenomes", timeout=timeout)
    genomes = payload.get("ucscGenomes") or {}
    if isinstance(genomes, dict):
        _UCSC_GENOMES_CACHE = genomes
        return genomes
    return {}


def fetch_ucsc_chrom_lengths(ucsc_dbkey: str, timeout: float = 20.0) -> dict[str, int]:
    """Fetch exact chromosome lengths for a UCSC genome build."""
    payload = _fetch_json(f"{_UCSC_API}/list/chromosomes", params={"genome": ucsc_dbkey}, timeout=timeout)
    chroms = payload.get("chromosomes") or {}
    return {str(k): int(v) for k, v in chroms.items() if isinstance(v, (int, float)) or str(v).isdigit()}


def _clean_species_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", str(raw or "").strip())
    low = s.lower()
    for bad, good in _SPECIES_TYPOS.items():
        if low.startswith(bad):
            s = good + s[len(bad):]
            break
    # Keep binomial "Genus species" if trailing subspecies/strain markers exist
    s = re.split(r"\b(?:subsp\.?|ssp\.?|str\.?|strain|var\.?|isolate|L\.)\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:-()")
    parts = [p for p in s.split() if re.match(r"^[A-Za-z][A-Za-z.-]*$", p)]
    if len(parts) >= 2 and parts[0][0].isupper():
        return f"{parts[0]} {parts[1]}"
    return " ".join(parts[:2])


def infer_species_from_cached_meta(
    dbkey: str = "",
    display_name: str = "",
    value: str = "",
    *,
    ucsc_genomes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Infer UCSC build key, species name, strain, and any accession hint from Galaxy all_fasta metadata."""
    dbk = str(dbkey or "").strip()
    val = str(value or "").strip()
    name = str(display_name or "").strip()

    out = {
        "species_name": "",
        "strain": "",
        "ucsc_dbkey": "",
        "assembly_hint": "",
        "ncbi_accession_hint": "",
    }

    # 1. Explicit NCBI assembly accession anywhere in metadata
    m_acc = re.search(r"\b(GC[AF]_\d{9}(?:\.\d+)?)\b", f"{name} {dbk} {val}", re.IGNORECASE)
    if m_acc:
        out["ncbi_accession_hint"] = m_acc.group(1).upper()

    # 2. Match against UCSC genome catalog (handles hg38, hg38canon, dm6, sacCer3, mm10, etc.)
    if ucsc_genomes is None:
        try:
            ucsc_genomes = fetch_ucsc_genomes()
        except Exception:
            ucsc_genomes = {}

    stripped_val = re.sub(r"(?:canon|full|female|male|chrUn|hap|patch\d*)$", "", val, flags=re.IGNORECASE)
    for cand in (dbk, val, stripped_val):
        if cand and cand in ucsc_genomes:
            info = ucsc_genomes[cand]
            out["ucsc_dbkey"] = cand
            sci = str(info.get("scientificName") or "").strip()
            if sci:
                out["species_name"] = _clean_species_name(sci)
            src_name = str(info.get("sourceName") or "")
            if not out["ncbi_accession_hint"]:
                ms = re.search(r"\b(GC[AF]_\d{9}(?:\.\d+)?)\b", src_name, re.IGNORECASE)
                if ms:
                    out["ncbi_accession_hint"] = ms.group(1).upper()
            break

    # 3. Parse scientific name, strain, and assembly hint from Galaxy display_name
    if name:
        head = name.split(":", 1)[0].strip()
        tail = name.split(":", 1)[1].strip() if ":" in name else ""

        # Extract parenthesized chunks: could be "(Homo sapiens)" or "(str. K-12 substr. MG1655)" or "(PAO1)"
        parens = re.findall(r"\(([^)]+)\)", head)
        head_no_paren = re.sub(r"\([^)]*\)", " ", head).strip()

        for ptxt in parens:
            pt = ptxt.strip()
            # Strain / substrain inside parens
            m_sub = re.search(r"\bsubstr\.?\s*([A-Za-z0-9_.-]+)", pt, re.IGNORECASE)
            m_str = re.search(r"\b(?:str\.?|strain)\s*([A-Za-z0-9_.-]+)", pt, re.IGNORECASE)
            if m_sub:
                out["strain"] = m_sub.group(1)
            elif m_str:
                out["strain"] = m_str.group(1)
            elif not out["species_name"] and re.match(r"^[A-Z][a-z]+\s+[a-z]{2,}\b", pt):
                out["species_name"] = _clean_species_name(pt)
            elif not out["strain"] and re.match(r"^[A-Z0-9][A-Za-z0-9_.-]{1,20}$", pt):
                out["strain"] = pt

        # Strain outside parens, e.g. "Bacillus subtilis subsp. subtilis str. 168"
        if not out["strain"]:
            m_sub = re.search(r"\bsubstr\.?\s*([A-Za-z0-9_.-]+)", head_no_paren, re.IGNORECASE)
            m_str = re.search(r"\b(?:str\.?|strain)\s*([A-Za-z0-9_.-]+)", head_no_paren, re.IGNORECASE)
            if m_sub:
                out["strain"] = m_sub.group(1)
            elif m_str:
                out["strain"] = m_str.group(1)

        if not out["species_name"] and head_no_paren:
            cleaned = _clean_species_name(head_no_paren)
            if len(cleaned.split()) >= 2:
                out["species_name"] = cleaned

        # Assembly hint from tail or value (e.g. TAIR10, IRGSP4.0, R64)
        for token_src in (tail, val, dbk):
            m_asm = re.search(r"(TAIR\d+(?:\.\d+)?|IRGSP[\d.]+|GRCh\d+|GRCm\d+|BDGP\d+|WBcel\d+)", token_src, re.IGNORECASE)
            if m_asm:
                out["assembly_hint"] = m_asm.group(1)
                break

    return out


_GTF_GID_RE = re.compile(r'gene_id "([^"]+)"')
_GFF_KEY_RES = (
    re.compile(r"(?:^|;)locus_tag=([^;]+)"),
    re.compile(r"(?:^|;)gene_id=([^;]+)"),
    re.compile(r"(?:^|;)gene=([^;]+)"),
    re.compile(r"(?:^|;)Parent=([^;]+)"),
    re.compile(r"(?:^|;)ID=([^;]+)"),
)


def _extract_gene_key(attrs: str, ctg: str, start: str, end: str) -> str:
    mg = _GTF_GID_RE.search(attrs)
    if mg:
        return mg.group(1)
    for pat in _GFF_KEY_RES:
        mf = pat.search(attrs)
        if mf:
            return mf.group(1).split(",")[0]
    return f"{ctg}:{start}-{end}"


def verify_annotation_against_fasta(
    annot_path: str | Path,
    fasta_lengths: dict[str, int],
    *,
    min_genes: int = 10,
    min_gene_frac: float = 0.80,
    raw_total_genes: int | None = None,
) -> dict[str, Any]:
    """Verify contig overlap and coordinate bounds of a GTF/GFF3 annotation against FASTA lengths."""
    ap = Path(annot_path)
    with open(ap, "rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open

    all_genes: set[str] = set()
    matched_genes: set[str] = set()
    unmatched_contigs: set[str] = set()
    n_features = 0
    n_matched_features = 0
    n_oob = 0
    oob_examples: list[str] = []

    with opener(ap, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2]
            if ftype not in ("exon", "CDS", "gene", "transcript", "mRNA"):
                continue
            ctg = parts[0]
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            gid = _extract_gene_key(parts[8], ctg, parts[3], parts[4])

            n_features += 1
            all_genes.add(gid)
            ctg_len = fasta_lengths.get(ctg)
            if ctg_len is None:
                unmatched_contigs.add(ctg)
                continue
            if start < 1 or end < start or end > ctg_len:
                n_oob += 1
                if len(oob_examples) < 5:
                    oob_examples.append(f"{ctg}:{start}-{end} (contig_len={ctg_len})")
                continue
            n_matched_features += 1
            matched_genes.add(gid)

    denom = max(raw_total_genes or 0, len(all_genes))
    matched_frac = (len(matched_genes) / denom) if denom > 0 else 0.0
    eff_min_genes = min(min_genes, denom) if denom > 0 else min_genes
    ok = (n_oob == 0) and (len(matched_genes) >= max(1, eff_min_genes)) and (matched_frac >= min_gene_frac)
    return {
        "ok": ok,
        "n_features": n_features,
        "n_matched_features": n_matched_features,
        "n_total_genes": denom,
        "n_matched_genes": len(matched_genes),
        "matched_gene_frac": round(matched_frac, 4),
        "n_oob": n_oob,
        "oob_examples": oob_examples,
        "unmatched_contigs_count": len(unmatched_contigs),
    }


def _url_exists(url: str, timeout: float = 15.0) -> bool:
    sess = _ipv4_session()
    try:
        r = sess.head(url, allow_redirects=True, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _try_ucsc_refseq_gtf(
    ucsc_dbkey: str,
    fasta_lengths: dict[str, int],
    dest_dir: Path,
    *,
    min_bp_frac: float = 0.90,
    min_gene_frac: float = 0.80,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify UCSC chrom lengths and fetch prebuilt UCSC RefSeq GTF."""
    try:
        ucsc_chroms = fetch_ucsc_chrom_lengths(ucsc_dbkey)
    except Exception as exc:
        return None, f"Could not query UCSC chromosome lengths for {ucsc_dbkey} ({exc})"
    if not ucsc_chroms:
        return None, f"UCSC returned no chromosome lengths for {ucsc_dbkey}"

    total_fa_bp = sum(fasta_lengths.values())
    if total_fa_bp <= 0:
        return None, "Cached FASTA contains zero bases"

    matched_bp = 0
    mismatches: list[str] = []
    for ctg, flen in fasta_lengths.items():
        ulen = ucsc_chroms.get(ctg)
        if ulen is None:
            continue
        if ulen != flen:
            mismatches.append(f"{ctg} (cached {flen:,} bp vs UCSC {ulen:,} bp)")
        else:
            matched_bp += flen

    if mismatches:
        sample_mm = ", ".join(mismatches[:3])
        return None, f"Chromosome length mismatch against UCSC {ucsc_dbkey}: {sample_mm}"

    bp_frac = matched_bp / total_fa_bp
    if bp_frac < min_bp_frac:
        return None, (
            f"Only {100.0 * bp_frac:.1f}% of cached FASTA bases match UCSC {ucsc_dbkey} contig names and lengths "
            f"(minimum required is {100.0 * min_bp_frac:.0f}%)"
        )

    cache_env = os.environ.get("FOCALOOP_NCBI_CACHE_DIR", "").strip()
    root = (Path(cache_env) / "ucsc" / ucsc_dbkey) if cache_env else (dest_dir / "ucsc" / ucsc_dbkey)
    root.mkdir(parents=True, exist_ok=True)

    tracks = (
        ("ncbiRefSeq", f"{ucsc_dbkey}.ncbiRefSeq.gtf.gz"),
        ("refGene", f"{ucsc_dbkey}.refGene.gtf.gz"),
        ("ensGene", f"{ucsc_dbkey}.ensGene.gtf.gz"),
    )
    chosen_track: str | None = None
    chosen_path: Path | None = None
    for track_name, fname in tracks:
        local_p = root / fname
        if local_p.is_file() and local_p.stat().st_size > 0:
            chosen_track = track_name
            chosen_path = local_p
            break
        url = f"{_UCSC_GOLDENPATH}/{ucsc_dbkey}/bigZips/genes/{fname}"
        if _url_exists(url):
            _emit(log, f"Downloading UCSC {track_name} GTF for {ucsc_dbkey}...")
            _download_file(url, local_p)
            chosen_track = track_name
            chosen_path = local_p
            break

    if chosen_path is None or chosen_track is None:
        return None, f"No prebuilt UCSC gene GTF track is available for {ucsc_dbkey}"

    check = verify_annotation_against_fasta(chosen_path, fasta_lengths, min_gene_frac=min_gene_frac)
    if not check["ok"]:
        return None, (
            f"UCSC {ucsc_dbkey} {chosen_track} annotation failed FASTA verification "
            f"(matched_genes={check['n_matched_genes']}/{check['n_total_genes']}, "
            f"out_of_bounds={check['n_oob']})"
        )

    return {
        "gtf_path": chosen_path,
        "source": f"ucsc:{ucsc_dbkey}:{chosen_track}",
        "ucsc_dbkey": ucsc_dbkey,
        "matched_bp_frac": round(bp_frac, 4),
        "verification": check,
    }, ""


def _parse_assembly_report(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 9:
            continue
        seq_len_s = parts[8].strip()
        if not seq_len_s.isdigit():
            continue
        rows.append(
            {
                "seq_name": parts[0].strip(),
                "seq_role": parts[1].strip(),
                "assigned_mol": parts[2].strip(),
                "genbank_accn": parts[4].strip(),
                "refseq_accn": parts[6].strip(),
                "seq_len": int(seq_len_s),
                "ucsc_name": parts[9].strip() if len(parts) > 9 else "na",
            }
        )
    return rows


def _build_exact_contig_map(
    report_rows: list[dict[str, Any]],
    fasta_lengths: dict[str, int],
) -> tuple[dict[str, str], float, float, list[str]]:
    """Build ``{ncbi_seqid: fasta_contig}`` strictly enforcing exact base-pair length equality."""
    fa_lower = {k.lower(): k for k in fasta_lengths}
    fa_by_len: dict[int, list[str]] = {}
    for k, ln in fasta_lengths.items():
        fa_by_len.setdefault(ln, []).append(k)

    contig_map: dict[str, str] = {}
    matched_fa: set[str] = set()
    used_rows: set[int] = set()
    length_mismatches: list[str] = []

    # Pass 1: match by synonym AND exact length
    for idx, row in enumerate(report_rows):
        slen = row["seq_len"]
        rs = row["refseq_accn"]
        gb = row["genbank_accn"]
        sn = row["seq_name"]
        uc = row["ucsc_name"]

        cand_names: list[str] = []
        for raw in (rs, uc, gb, sn):
            if raw and raw.lower() != "na":
                cand_names.append(raw)
        if sn and sn.lower() != "na":
            if not sn.lower().startswith("chr"):
                cand_names.append(f"chr{sn}")
            if sn.lower() in _MT_ALIASES:
                cand_names.extend(["chrM", "MT", "M", "Mito", "chrMT"])
            elif sn.lower() in _PLTD_ALIASES:
                cand_names.extend(["chrC", "Pltd", "C", "Chloroplast", "chrPt"])

        matched_ctg: str | None = None
        for cname in cand_names:
            f_ctg = fa_lower.get(cname.lower())
            if f_ctg is None:
                continue
            if fasta_lengths[f_ctg] == slen:
                matched_ctg = f_ctg
                break
            length_mismatches.append(f"{f_ctg} (cached {fasta_lengths[f_ctg]:,} bp vs NCBI {rs or sn} {slen:,} bp)")

        if matched_ctg is not None:
            used_rows.add(idx)
            matched_fa.add(matched_ctg)
            for acc in (rs, gb, uc, sn):
                if acc and acc.lower() != "na":
                    contig_map[acc] = matched_ctg

    # Pass 2: unique exact-length match for unnamed microbial chromosomes (e.g. CVMFS 'chr')
    unmatched_rows_by_len: dict[int, list[int]] = {}
    for idx, row in enumerate(report_rows):
        if idx in used_rows:
            continue
        unmatched_rows_by_len.setdefault(row["seq_len"], []).append(idx)

    for f_ctg, flen in fasta_lengths.items():
        if f_ctg in matched_fa or flen < 1000:
            continue
        same_len_fa = [c for c in fa_by_len.get(flen, []) if c not in matched_fa]
        same_len_rows = unmatched_rows_by_len.get(flen, [])
        if len(same_len_fa) == 1 and len(same_len_rows) == 1:
            r_idx = same_len_rows[0]
            row = report_rows[r_idx]
            used_rows.add(r_idx)
            matched_fa.add(f_ctg)
            for acc in (row["refseq_accn"], row["genbank_accn"], row["ucsc_name"], row["seq_name"]):
                if acc and acc.lower() != "na":
                    contig_map[acc] = f_ctg

    # Record primary single-contig length mismatch if nothing matched (super helpful for microbial builds!)
    if not matched_fa and len(fasta_lengths) == 1 and report_rows:
        f_ctg, flen = next(iter(fasta_lengths.items()))
        longest_row = max(report_rows, key=lambda r: r["seq_len"])
        rs_lbl = longest_row["refseq_accn"] if longest_row["refseq_accn"] != "na" else longest_row["seq_name"]
        length_mismatches.append(
            f"{f_ctg} (cached {flen:,} bp vs NCBI {rs_lbl} {longest_row['seq_len']:,} bp)"
        )

    total_fa_bp = sum(fasta_lengths.values()) or 1
    matched_bp = sum(fasta_lengths[c] for c in matched_fa)
    bp_frac = matched_bp / total_fa_bp
    ctg_frac = len(matched_fa) / max(1, len(fasta_lengths))
    return contig_map, bp_frac, ctg_frac, length_mismatches


def _list_ncbi_assembly_versions(base_acc: str) -> list[tuple[int, str, str]]:
    """List all version directories under an NCBI FTP base accession (GCF_xxxxxxxxx), newest first."""
    m = _NCBI_ACC_RE.match(base_acc.strip().upper())
    if not m:
        return []
    prefix, d1, d2, d3 = m.group(1).upper(), m.group(2), m.group(3), m.group(4)
    parent_url = f"{_NCBI_FTP_ALL}/{prefix}/{d1}/{d2}/{d3}/"
    try:
        listing = _fetch_text(parent_url, timeout=25.0, retries=2)
    except Exception:
        return []
    base = f"{prefix}_{d1}{d2}{d3}"
    pat = re.compile(rf'href="({re.escape(base)}\.(\d+)_[^"/]+)/"')
    found: dict[str, tuple[int, str, str]] = {}
    for asm_dir, ver_s in pat.findall(listing):
        found[asm_dir] = (int(ver_s), asm_dir, f"{parent_url}{asm_dir}")
    return sorted(found.values(), key=lambda x: (x[0], x[1]), reverse=True)


def _discover_candidate_gcf_bases(meta: dict[str, str], max_candidates: int = 5) -> list[tuple[str, str]]:
    """Return ordered ``[(GCF_base, full_latest_acc)]`` candidates for a cached genome."""
    import urllib.parse

    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(acc: str) -> None:
        a = acc.strip().upper()
        if not _NCBI_ACC_RE.match(a):
            return
        if a.startswith("GCA_"):
            a = "GCF_" + a[4:]
        base = a.split(".")[0]
        if base not in seen:
            seen.add(base)
            out.append((base, a))

    if meta.get("ncbi_accession_hint"):
        _add(meta["ncbi_accession_hint"])

    species = meta.get("species_name", "").strip()
    if species:
        encoded = urllib.parse.quote(species, safe="")
        url = f"{_NCBI_DATASETS_API}/taxon/{encoded}/dataset_report"
        params = {
            "filters.assembly_source": "refseq",
            "filters.has_annotation": "true",
            "filters.exclude_paired_reports": "true",
            "page_size": 100,
        }
        try:
            payload = _fetch_json(url, params=params, timeout=25.0, retries=2)
            reps = payload.get("reports") or []
        except Exception:
            reps = []

        strain_norm = _norm_token(meta.get("strain", ""))
        asm_norm = _norm_token(meta.get("assembly_hint", ""))
        lvl_rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}

        def _rank(rep: dict[str, Any]) -> tuple[int, int, int, int, str]:
            ainfo = rep.get("assembly_info") or {}
            org = rep.get("organism") or {}
            infra = org.get("infraspecific_names") or {}
            r_strain = _norm_token(infra.get("strain") or infra.get("isolate") or "")
            r_asm = _norm_token(ainfo.get("assembly_name") or "")
            # Exact or substring strain match gets top priority
            s_hit = 2
            if strain_norm and r_strain:
                if strain_norm == r_strain:
                    s_hit = 0
                elif strain_norm in r_strain or r_strain in strain_norm:
                    s_hit = 1
            a_hit = 0 if (asm_norm and asm_norm in r_asm) else 1
            cat = str(ainfo.get("refseq_category") or "").lower()
            cat_r = 0 if "reference" in cat else (1 if "representative" in cat else 2)
            lvl_r = lvl_rank.get(str(ainfo.get("assembly_level") or ""), 4)
            return (s_hit, a_hit, cat_r, lvl_r, str(rep.get("accession") or ""))

        reps.sort(key=_rank)
        for rep in reps:
            acc = str(rep.get("accession") or "").strip().upper()
            if acc:
                _add(acc)
            if len(out) >= max_candidates:
                break

    return out[:max_candidates]


def _remap_gff3_contigs(
    src_gff_gz: Path,
    dst_gff3: Path,
    contig_map: dict[str, str],
) -> dict[str, int]:
    """Rewrite a gzipped NCBI GFF3 replacing column-1 accessions with exact matched FASTA contigs."""
    dst_gff3.parent.mkdir(parents=True, exist_ok=True)
    with open(src_gff_gz, "rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open

    raw_genes: set[str] = set()
    kept_genes: set[str] = set()
    kept_lines = 0
    dropped_lines = 0

    with opener(src_gff_gz, "rt", encoding="utf-8", errors="replace") as fi, open(dst_gff3, "wt", encoding="utf-8") as fo:
        for line in fi:
            if not line:
                continue
            if line.startswith("##sequence-region "):
                parts = line.strip().split()
                if len(parts) >= 4 and parts[1] in contig_map:
                    parts[1] = contig_map[parts[1]]
                    fo.write(" ".join(parts) + "\n")
                continue
            if line.startswith("#"):
                fo.write(line)
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 9:
                continue
            old_ctg = parts[0]
            ftype = parts[2]
            gid = ""
            if ftype in ("gene", "CDS", "exon", "mRNA", "transcript"):
                gid = _extract_gene_key(parts[8], old_ctg, parts[3], parts[4])
                raw_genes.add(gid)
            new_ctg = contig_map.get(old_ctg)
            if new_ctg is None:
                dropped_lines += 1
                continue
            parts[0] = new_ctg
            fo.write("\t".join(parts) + "\n")
            kept_lines += 1
            if gid:
                kept_genes.add(gid)

    return {
        "kept_lines": kept_lines,
        "dropped_lines": dropped_lines,
        "raw_total_genes": len(raw_genes),
        "kept_genes": len(kept_genes),
    }


def _try_ncbi_refseq_gff(
    meta: dict[str, str],
    fasta_lengths: dict[str, int],
    dest_dir: Path,
    *,
    min_bp_frac: float = 0.90,
    min_gene_frac: float = 0.80,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Find exact-length matching NCBI RefSeq assembly, download GFF3, and remap contigs."""
    cands = _discover_candidate_gcf_bases(meta)
    if not cands:
        return None, (
            f"Could not map built-in genome ({meta.get('species_name') or 'unknown organism'}) "
            "to an annotated NCBI RefSeq assembly"
        )

    best_match: dict[str, Any] | None = None
    diagnostics: list[str] = []

    for base_acc, latest_acc in cands:
        versions = _list_ncbi_assembly_versions(base_acc)
        if not versions:
            continue
        for ver_num, asm_dir, base_url in versions:
            rep_url = f"{base_url}/{asm_dir}_assembly_report.txt"
            try:
                rep_txt = _fetch_text(rep_url, timeout=20.0, retries=2)
            except Exception:
                continue
            rows = _parse_assembly_report(rep_txt)
            if not rows:
                continue
            cmap, bp_frac, ctg_frac, mm = _build_exact_contig_map(rows, fasta_lengths)
            if bp_frac < min_bp_frac:
                if mm and len(diagnostics) < 3:
                    diagnostics.append(f"{latest_acc}: {mm[0]}")
                continue

            gff_url = f"{base_url}/{asm_dir}_genomic.gff.gz"
            if not _url_exists(gff_url):
                diagnostics.append(
                    f"{asm_dir} matches contig lengths ({100.0 * bp_frac:.1f}% bp) but NCBI no longer hosts its GFF3 "
                    f"(latest annotated RefSeq revision is {latest_acc})"
                )
                continue

            score = (round(bp_frac, 6), round(ctg_frac, 6), ver_num)
            if best_match is None or score > best_match["score"]:
                best_match = {
                    "score": score,
                    "asm_dir": asm_dir,
                    "base_url": base_url,
                    "gff_url": gff_url,
                    "accession": asm_dir.split("_", 2)[0] + "_" + asm_dir.split("_", 2)[1],
                    "contig_map": cmap,
                    "bp_frac": bp_frac,
                    "ctg_frac": ctg_frac,
                }
            # Perfect 100% bp & 100% contig match on newest revision -> stop early
            if bp_frac >= 0.999999 and ctg_frac >= 0.999999:
                break
        if best_match is not None and best_match["bp_frac"] >= 0.999999 and best_match["ctg_frac"] >= 0.999999:
            break

    if best_match is None:
        detail = (" Details: " + " | ".join(diagnostics[:3])) if diagnostics else ""
        rec_acc = cands[0][1]
        return None, (
            f"No annotated NCBI RefSeq revision matches the cached genome contig lengths exactly.{detail} "
            f"Recommended fix: switch Reference source to 'Fetch automatically by NCBI assembly accession' "
            f"with {rec_acc} to download a matched FASTA and GFF3 pair, or supply a matching GTF/GFF3 from your History."
        )

    asm_dir = best_match["asm_dir"]
    cache_env = os.environ.get("FOCALOOP_NCBI_CACHE_DIR", "").strip()
    cache_root = (Path(cache_env) / asm_dir) if cache_env else (dest_dir / "ncbi" / asm_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_gff_gz = cache_root / f"{asm_dir}_genomic.gff.gz"
    if not (raw_gff_gz.is_file() and raw_gff_gz.stat().st_size > 0):
        _emit(log, f"Downloading matched NCBI RefSeq GFF3 ({asm_dir})...")
        _download_file(best_match["gff_url"], raw_gff_gz)

    remapped_gff3 = dest_dir / f"{asm_dir}.remapped.gff3"
    remap_stats = _remap_gff3_contigs(raw_gff_gz, remapped_gff3, best_match["contig_map"])
    check = verify_annotation_against_fasta(
        remapped_gff3,
        fasta_lengths,
        min_gene_frac=min_gene_frac,
        raw_total_genes=remap_stats["raw_total_genes"],
    )
    if not check["ok"]:
        return None, (
            f"Remapped NCBI GFF3 ({asm_dir}) failed FASTA verification "
            f"(matched_genes={check['n_matched_genes']}/{check['n_total_genes']}, "
            f"out_of_bounds={check['n_oob']})"
        )

    return {
        "gtf_path": remapped_gff3,
        "source": f"ncbi:{asm_dir}",
        "ncbi_accession": best_match["accession"],
        "matched_bp_frac": round(best_match["bp_frac"], 4),
        "remap_stats": remap_stats,
        "verification": check,
    }, ""


def resolve_cvmfs_annotation(
    fasta_path: str | Path,
    dest_dir: str | Path,
    *,
    dbkey: str = "",
    display_name: str = "",
    value: str = "",
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Automatically resolve, fetch, contig-normalize, and verify a RefSeq annotation for a built-in CVMFS FASTA."""
    t0 = time.time()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    fasta_lengths = read_fasta_contig_lengths(fasta_path)
    if not fasta_lengths:
        raise ValueError(f"Built-in reference genome FASTA {fasta_path} contains no sequences.")

    meta = infer_species_from_cached_meta(dbkey=dbkey, display_name=display_name, value=value)
    label_desc = display_name or dbkey or value or Path(fasta_path).name
    _emit(
        log,
        f"Resolving RefSeq annotation for built-in genome {label_desc!r} "
        f"(contigs={len(fasta_lengths)}, total_bp={sum(fasta_lengths.values()):,}, "
        f"ucsc_dbkey={meta['ucsc_dbkey'] or 'none'}, species={meta['species_name'] or 'unknown'})...",
    )

    reasons: list[str] = []

    # Tier 1: UCSC RefSeq GTF for UCSC-keyed builds
    if meta["ucsc_dbkey"]:
        res, err = _try_ucsc_refseq_gtf(meta["ucsc_dbkey"], fasta_lengths, dest, log=log)
        if res is not None:
            res.update(
                {
                    "species_name": meta["species_name"],
                    "strain": meta["strain"],
                    "ncbi_accession": meta.get("ncbi_accession_hint", ""),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            _emit(
                log,
                f"Auto-annotation verified via {res['source']} "
                f"({res['verification']['n_matched_genes']:,} genes, "
                f"{100.0 * res['matched_bp_frac']:.1f}% FASTA bp matched, {res['elapsed_s']:.1f}s)",
            )
            return res
        reasons.append(f"UCSC lookup failed: {err}")

    # Tier 2: NCBI Datasets + exact-length version selection + GFF3 contig remap
    res, err = _try_ncbi_refseq_gff(meta, fasta_lengths, dest, log=log)
    if res is not None:
        res.update(
            {
                "species_name": meta["species_name"],
                "strain": meta["strain"],
                "ucsc_dbkey": meta["ucsc_dbkey"],
                "elapsed_s": round(time.time() - t0, 2),
            }
        )
        _emit(
            log,
            f"Auto-annotation verified via {res['source']} "
            f"({res['verification']['n_matched_genes']:,} genes, "
            f"{100.0 * res['matched_bp_frac']:.1f}% FASTA bp matched, {res['elapsed_s']:.1f}s)",
        )
        return res
    reasons.append(f"NCBI lookup failed: {err}")

    summary = " / ".join(reasons)
    raise ValueError(
        f"Could not automatically resolve a verified RefSeq gene annotation for built-in genome {label_desc!r}. "
        f"{summary} "
        "To proceed safely without shifted coordinates, either select 'From your Galaxy History (GTF or GFF3)' "
        "for Gene annotation source or switch Reference genome source to 'Fetch automatically by NCBI assembly accession'."
    )
