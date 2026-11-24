"""Structural ncRNA annotation pipeline (EGAPx-parity mode: tRNAscan-SE + barrnap + 2-stage Rfam cmsearch).

Implements Iteration 2' strictly post-Seed using genome-only inputs (zero target leakage, zero models
emitted from ``data/euk_rrna_core.fa`` bait):
  1. Pinned Rfam 15.1 verification (A4) & 3-class (``rRNA``, ``snRNA``, ``snoRNA``) taxon-filtered
     paired ``INFERNAL1/a + HMMER3/f`` subset extraction (N2, N4).
  2. Sharded ``tRNAscan-SE -E -X 55 -Q -q`` (no ``-b``) with spliced exon lifting on both strands
     (Blocker 8) and ``CTrnaAnnot::s_RepeatFilter`` ``std::multimap`` front-erasure parity (N1, I1).
  3. ``barrnap --kingdom euk --reject 0.25`` for ``18S``/``28S``/``5.8S`` (A2: ``>=0.80`` intact,
     ``0.25..0.80`` ``rRNA_pseudogene``, ``5S`` discarded).
  4. Vectorized ``nhmmer -Z 160 -E 10.0`` prefilter -> ``cmsearch --rfam --cut_ga --nohmmonly`` with
     Rfam clan/class competition, terminal-N trimming, and strict A1 model-end completeness.
  5. Dynamic post-coding overlap exclusions inside ``synthesize()`` retaining intronic ``snoRNA``/``snRNA``
     and ``SNHG`` snoRNA-host ``lncRNA``s.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from . import seed as seedmod
from .evidence import LocusIndex
from .gffio import Transcript, interval_overlap_bp, merge_intervals, parse_attrs
from .sequtil import Genome
from .telemetry import StageLog, StageRecord, run_cmd, safe_env

NCRNA_RAW_VERSION = "v3"
RFAM_PINNED_VERSION = "15.1"
ALLOWED_RFAM_CLASSES = frozenset({"rRNA", "snRNA", "snoRNA"})
EXCLUDED_RFAM_MODELS = frozenset({"SSU_rRNA_eukarya", "LSU_rRNA_eukarya"})

PINNED_RFAM_ASSETS: dict[str, str] = {
    "Rfam.cm.gz": "e2e636e9ce138dae506769aa74d067ce35466e94002818cfe125ea2a3cab8168",
    "rfam151_amendments.xml": "8adeea9e920e30e5a31855882997f2837f6a5f5d6406a0ddf0d2737c339373af",
    "clan_membership.txt.gz": "2f199aaaa0a1e340145acb8fa20639427d1c0a38f4dedce29901888f222afbec",
}

DEFAULT_BARRNAP_EUK_LENG: dict[str, int] = {
    "18S_rRNA": 1869,
    "28S_rRNA": 3401,
    "5_8S_rRNA": 154,
}

TAXON_GROUP_ALLOWED_TAXNAMES: dict[str, frozenset[str]] = {
    "insecta": frozenset({"", "Eukaryota", "Metazoa", "Drosophila", "Drosophila melanogaster"}),
    "viridiplantae": frozenset({"", "Eukaryota", "Viridiplantae", "Brassica rapa"}),
    "metazoa": frozenset({"", "Eukaryota", "Metazoa"}),
}

_INSECT_TOKENS = ("insecta", "diptera", "arthropoda", "endopterygota", "hymenoptera", "lepidoptera", "coleoptera")
_PLANT_TOKENS = (
    "embryophyta",
    "viridiplantae",
    "eudicot",
    "liliopsida",
    "chlorophyta",
    "brassica",
    "angiosperm",
    "poales",
    "solanales",
    "fabales",
)

_ASSET_SHA_CACHE: dict[tuple[str, int, int, int], str] = {}
_CMSEARCH_OK_CACHE: dict[tuple[str, int, int], bool] = {}
_N100_RE = re.compile(r"[Nn]{100,}")


@dataclass(frozen=True)
class RfamEntry:
    identification: str
    gene_symbol: str
    rna_class: str
    ga_threshold: float
    end_prox_bp: int
    cutoff2: float
    product: str
    taxname: str


def default_rfam_dir() -> Path:
    env_dir = os.environ.get("GENELOOP_RFAM_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    for cand in (
        Path("/opt/focaloop/rfam"),
        Path("/cvmfs/data.galaxyproject.org/byhand/rfam"),
        seedmod.BUILD_ROOT / "rfam",
    ):
        if cand.is_dir():
            return cand
    return seedmod.BUILD_ROOT / "rfam"


def _read_log_tail(log_path: Path, max_lines: int = 40, max_chars: int = 4096) -> str:
    try:
        if not log_path.is_file():
            return ""
        txt = log_path.read_text(encoding="utf-8", errors="replace").strip()
        if not txt:
            return ""
        lines = txt.splitlines()[-max_lines:]
        joined = "\n".join(lines)
        return joined[-max_chars:]
    except OSError:
        return ""


def _run_ncrna_cmd(
    cmd: list[str],
    *,
    log_path: Path,
    stage_label: str,
    stdout_path: Path | None = None,
) -> tuple[int, float, float, float]:
    """Execute a structural ncRNA tool and surface log tail to stderr + RuntimeError before temp cleanup."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stdout_path is not None:
        with open(stdout_path, "w") as f_out, open(log_path, "w") as f_err:
            rc, w, u, s = run_cmd(cmd, stdout=f_out, stderr=f_err, check=False)
    else:
        with open(log_path, "w") as f_log:
            rc, w, u, s = run_cmd(cmd, stdout=f_log, stderr=f_log, check=False)
    if rc != 0:
        tail = _read_log_tail(log_path)
        cmd_str = shlex.join(cmd)
        msg = f"{stage_label} failed (rc={rc}): {cmd_str}"
        if tail:
            msg = f"{msg}\n--- {stage_label} stderr/log tail ---\n{tail}"
        print(f"[focaloop-geneloop] {msg}", file=sys.stderr)
        raise RuntimeError(msg)
    return rc, w, u, s


def _sha256_cached(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))
    cached = _ASSET_SHA_CACHE.get(key)
    if cached:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    _ASSET_SHA_CACHE[key] = digest
    return digest


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _sanitize_gff_val(val: str) -> str:
    return (
        str(val)
        .replace("\t", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace(";", "%3B")
        .replace("=", "%3D")
        .strip()
    )


def _cmsearch_supports_rfam(bin_path: str) -> bool:
    p = Path(bin_path)
    if not (p.is_file() and os.access(p, os.X_OK)):
        return False
    st = p.stat()
    ck = (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
    if ck in _CMSEARCH_OK_CACHE:
        return _CMSEARCH_OK_CACHE[ck]
    try:
        r = subprocess.run([str(p), "-h"], capture_output=True, text=True, timeout=10, env=safe_env())
        ok = "--rfam" in ((r.stdout or "") + (r.stderr or ""))
    except OSError:
        ok = False
    _CMSEARCH_OK_CACHE[ck] = ok
    return ok


def _find_infernal_cmsearch(search_path: str) -> str | None:
    cands: list[str] = []
    env_cm = os.environ.get("GENELOOP_CMSEARCH_BIN")
    if env_cm:
        cands.append(env_cm)
    cands.append("/usr/lib/infernal/cmsearch")
    w_cm = shutil.which("cmsearch", path=search_path)
    if w_cm:
        cands.append(w_cm)
    seen: set[str] = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        if _cmsearch_supports_rfam(c):
            return str(Path(c).resolve())
    return None


def check_ncrna_assets(rfam_dir: Path | None = None, mode: str = "auto") -> dict:
    """Verify pinned Rfam 15.1 files (SHA256-checked) and required structural ncRNA binaries.

    ``mode`` must be ``"auto" | "on" | "off"``:
      - ``"off"``: returns ``status="disabled"`` immediately.
      - ``"on"``: raises ``FileNotFoundError`` if any asset/binary is missing.
      - ``"auto"``: returns ``status="skipped_missing_assets"`` cleanly when assets/binaries are absent.
      - Existing Rfam files with mismatched SHA256 ALWAYS raise ``RuntimeError`` (A4).
    """
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"Invalid enable_structural_ncrna={mode!r}; expected 'auto', 'on', or 'off'")
    rdir = Path(rfam_dir) if rfam_dir is not None else default_rfam_dir()
    if mode == "off":
        return {
            "available": False,
            "status": "disabled",
            "stages": {"trnascan": False, "barrnap": False, "rfam": False},
            "rfam_dir": str(rdir),
            "rfam_version": RFAM_PINNED_VERSION,
            "missing": [],
            "binaries": {},
            "asset_sha256": {},
        }

    s_env = safe_env()
    search_path = os.pathsep.join(
        [
            s_env.get("PATH", ""),
            "/usr/lib/infernal",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    missing: list[str] = []
    asset_shas: dict[str, str] = {}
    for fname, exp_sha in PINNED_RFAM_ASSETS.items():
        fp = rdir / fname
        if not fp.is_file():
            missing.append(f"file:{fp}")
            continue
        got_sha = _sha256_cached(fp)
        if got_sha != exp_sha:
            raise RuntimeError(
                f"Pinned Rfam {RFAM_PINNED_VERSION} SHA256 mismatch for {fp}: "
                f"expected {exp_sha}, got {got_sha}"
            )
        asset_shas[fname] = got_sha

    bins: dict[str, str] = {}
    for tool in ("tRNAscan-SE", "nhmmer", "barrnap"):
        resolved = shutil.which(tool, path=search_path)
        if resolved:
            bins[tool] = str(Path(resolved).resolve())
        else:
            missing.append(f"binary:{tool}")
    cm_bin = _find_infernal_cmsearch(search_path)
    if cm_bin:
        bins["cmsearch"] = cm_bin
    else:
        missing.append("binary:cmsearch(Infernal>=1.1 --rfam)")

    stages = {
        "trnascan": "tRNAscan-SE" in bins,
        "barrnap": "barrnap" in bins,
        "rfam": (
            "nhmmer" in bins
            and "cmsearch" in bins
            and len(asset_shas) == len(PINNED_RFAM_ASSETS)
        ),
    }

    if missing:
        if mode == "on":
            raise FileNotFoundError(
                f"enable_structural_ncrna='on' requires Rfam {RFAM_PINNED_VERSION} assets and binaries; "
                f"missing: {', '.join(missing)}"
            )
        if any(stages.values()):
            return {
                "available": True,
                "status": "partial",
                "stages": stages,
                "rfam_dir": str(rdir),
                "rfam_version": RFAM_PINNED_VERSION,
                "missing": missing,
                "binaries": bins,
                "asset_sha256": asset_shas,
            }
        return {
            "available": False,
            "status": "skipped_missing_assets",
            "stages": stages,
            "rfam_dir": str(rdir),
            "rfam_version": RFAM_PINNED_VERSION,
            "missing": missing,
            "binaries": bins,
            "asset_sha256": asset_shas,
        }

    return {
        "available": True,
        "status": "ok",
        "stages": stages,
        "rfam_dir": str(rdir),
        "rfam_version": RFAM_PINNED_VERSION,
        "missing": [],
        "binaries": bins,
        "asset_sha256": asset_shas,
    }


def resolve_rfam_taxon_group(gate_or_lineage: dict | str | None) -> str:
    """Map gate lineage/config to ``'insecta'``, ``'viridiplantae'``, or ``'metazoa'``."""
    if isinstance(gate_or_lineage, dict):
        raw = " ".join(
            str(gate_or_lineage.get(k) or "")
            for k in ("busco_lineage", "compleasm_lineage", "tiberius_model_cfg")
        ).lower()
    else:
        raw = str(gate_or_lineage or "").lower()
    if any(tok in raw for tok in _INSECT_TOKENS):
        return "insecta"
    if any(tok in raw for tok in _PLANT_TOKENS):
        return "viridiplantae"
    return "metazoa"


def load_rfam_amendments(
    xml_path: str | Path,
    gate_or_lineage: dict | str | None = "metazoa",
) -> dict[str, RfamEntry]:
    """Parse ``rfam151_amendments.xml`` restricted to 3 allowed classes and lineage taxnames (§3.1)."""
    taxon_group = (
        gate_or_lineage
        if isinstance(gate_or_lineage, str) and gate_or_lineage in TAXON_GROUP_ALLOWED_TAXNAMES
        else resolve_rfam_taxon_group(gate_or_lineage)
    )
    allowed_tax = TAXON_GROUP_ALLOWED_TAXNAMES[taxon_group]
    tree = ET.parse(xml_path)
    out: dict[str, RfamEntry] = {}
    for entry in tree.getroot().findall(".//Entry"):
        ident = (entry.get("identification") or "").strip()
        if not ident or ident in EXCLUDED_RFAM_MODELS:
            continue
        rna_class = (entry.findtext("ncRNA_class") or "").strip() or (entry.findtext("feature_type") or "").strip()
        if rna_class not in ALLOWED_RFAM_CLASSES:
            continue
        taxname = (entry.findtext("organism_restriction_taxname") or "").strip()
        if taxname not in allowed_tax:
            continue
        gsym = (entry.findtext("gene_symbol") or ident).strip()
        ga = float((entry.findtext("GatheringMethodThreshold") or "0").strip())
        ep = int((entry.findtext("ContigEndProximityThreshold") or "9").strip())
        c2 = float((entry.findtext("cutoff2") or "20.0").strip())
        prod = (entry.findtext("product") or "").strip()
        out[ident] = RfamEntry(
            identification=ident,
            gene_symbol=gsym,
            rna_class=rna_class,
            ga_threshold=ga,
            end_prox_bp=ep,
            cutoff2=c2,
            product=prod,
            taxname=taxname,
        )
    return out


def load_clan_membership(clan_gz_path: str | Path) -> dict[str, str]:
    """Load Rfam accession -> clan accession mapping from ``clan_membership.txt.gz``."""
    clan_by_acc: dict[str, str] = {}
    p = Path(clan_gz_path)
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                clan_by_acc[parts[1]] = parts[0]
    return clan_by_acc


def ensure_rfam_subset_files(
    rfam_cm_gz: str | Path,
    out_dir: Path,
    selected_entries: dict[str, RfamEntry],
) -> tuple[Path, Path, dict[str, int], dict[str, str]]:
    """Extract paired ``INFERNAL1/a + HMMER3/f`` blocks to ``sub.cm`` and ``HMMER3/f`` to ``sub.hmm`` (A4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_cm = out_dir / "rfam_subset.cm"
    sub_hmm = out_dir / "rfam_subset.hmm"
    cm_tmp = out_dir / f".rfam_subset.cm.tmp.{os.getpid()}"
    hmm_tmp = out_dir / f".rfam_subset.hmm.tmp.{os.getpid()}"

    cm_clen: dict[str, int] = {}
    cm_acc: dict[str, str] = {}
    hmm_leng: dict[str, int] = {}

    p = Path(rfam_cm_gz)
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt") as fin, open(cm_tmp, "w") as f_cm, open(hmm_tmp, "w") as f_hmm:
        block: list[str] = []
        kind = ""
        name = ""
        acc = ""
        length: int | None = None
        for line in fin:
            if not block:
                if line.startswith("INFERNAL1/a"):
                    kind = "cm"
                elif line.startswith("HMMER3/f"):
                    kind = "hmm"
                else:
                    kind = ""
                name = ""
                acc = ""
                length = None
            block.append(line)
            if line.startswith("NAME "):
                name = line.split()[1]
            elif line.startswith("ACC "):
                acc = line.split()[1]
            elif kind == "cm" and line.startswith("CLEN "):
                length = int(line.split()[1])
            elif kind == "hmm" and line.startswith("LENG "):
                length = int(line.split()[1])
            elif line.strip() == "//":
                if name in selected_entries:
                    if kind == "cm":
                        if length is None:
                            raise RuntimeError(f"Missing CLEN in INFERNAL1/a block for {name!r}")
                        f_cm.writelines(block)
                        cm_clen[name] = length
                        cm_acc[name] = acc
                    elif kind == "hmm":
                        if length is None:
                            raise RuntimeError(f"Missing LENG in HMMER3/f block for {name!r}")
                        f_cm.writelines(block)
                        f_hmm.writelines(block)
                        hmm_leng[name] = length
                block = []

    if set(selected_entries) != set(cm_clen) or set(cm_clen) != set(hmm_leng):
        miss_cm = sorted(set(selected_entries) - set(cm_clen))
        miss_hmm = sorted(set(selected_entries) - set(hmm_leng))
        cm_tmp.unlink(missing_ok=True)
        hmm_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Rfam subset extraction mismatch (A4): selected={len(selected_entries)}, "
            f"cm={len(cm_clen)} (missing={miss_cm[:10]}), hmm={len(hmm_leng)} (missing={miss_hmm[:10]})"
        )
    os.replace(cm_tmp, sub_cm)
    os.replace(hmm_tmp, sub_hmm)
    return sub_cm, sub_hmm, cm_clen, cm_acc


# ----------------------------------------------------------------------------------------------
# Contig chunking & LPT sharding
# ----------------------------------------------------------------------------------------------


def split_nuclear_chunks(
    genome: Genome,
    nuclear_contigs: list[str],
    max_chunk_bp: int = 25_000_000,
    overlap_bp: int = 2_000,
) -> list[dict]:
    """Split >25 Mb nuclear contigs at N>=100 bp assembly gaps or fixed sequence overlaps."""
    chunks: list[dict] = []
    idx = 0
    for ctg in nuclear_contigs:
        L = int(genome.lengths.get(ctg, 0))
        if L <= 0:
            continue
        if L <= max_chunk_bp:
            chunks.append(
                {
                    "chunk_id": f"glchunk_{idx}",
                    "contig": ctg,
                    "lo": 0,
                    "hi": L,
                    "left_synthetic": False,
                    "right_synthetic": False,
                }
            )
            idx += 1
            continue
        pos = 0
        left_syn = False
        while pos < L:
            if L - pos <= max_chunk_bp:
                chunks.append(
                    {
                        "chunk_id": f"glchunk_{idx}",
                        "contig": ctg,
                        "lo": pos,
                        "hi": L,
                        "left_synthetic": left_syn,
                        "right_synthetic": False,
                    }
                )
                idx += 1
                break
            target = pos + max_chunk_bp
            search_lo = max(pos + max_chunk_bp // 2, target - 2_000_000)
            search_hi = min(L, target + 500_000)
            sub_seq = genome.fetch(ctg, search_lo, search_hi)
            m_gap = _N100_RE.search(sub_seq)
            if m_gap is not None:
                cut_hi = search_lo + m_gap.start()
                next_pos = search_lo + m_gap.end()
                if cut_hi > pos:
                    chunks.append(
                        {
                            "chunk_id": f"glchunk_{idx}",
                            "contig": ctg,
                            "lo": pos,
                            "hi": cut_hi,
                            "left_synthetic": left_syn,
                            "right_synthetic": False,
                        }
                    )
                    idx += 1
                pos = next_pos
                left_syn = False
            else:
                cut_hi = min(L, target)
                chunks.append(
                    {
                        "chunk_id": f"glchunk_{idx}",
                        "contig": ctg,
                        "lo": pos,
                        "hi": cut_hi,
                        "left_synthetic": left_syn,
                        "right_synthetic": cut_hi < L,
                    }
                )
                idx += 1
                if cut_hi >= L:
                    break
                pos = max(pos + 1, cut_hi - overlap_bp)
                left_syn = True
    return chunks


def pack_chunks_lpt(chunks: list[dict], n_shards: int) -> list[list[dict]]:
    """Deterministic LPT shard packing over nuclear sequence chunks."""
    if not chunks:
        return []
    k = max(1, min(int(n_shards), len(chunks)))
    buckets: list[list[dict]] = [[] for _ in range(k)]
    loads = [0] * k
    ordered = sorted(chunks, key=lambda c: (-(int(c["hi"]) - int(c["lo"])), str(c["contig"]), int(c["lo"])))
    for ch in ordered:
        bi = min(range(k), key=lambda i: (loads[i], i))
        buckets[bi].append(ch)
        loads[bi] += max(0, int(ch["hi"]) - int(ch["lo"]))
    return [b for b in buckets if b]


def _write_chunk_fasta(genome: Genome, chunks: list[dict], out_fa: Path) -> None:
    with open(out_fa, "w") as fh:
        for ch in chunks:
            seq = genome.fetch(ch["contig"], int(ch["lo"]), int(ch["hi"]))
            fh.write(f">{ch['chunk_id']}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


# ----------------------------------------------------------------------------------------------
# §3.2 tRNAscan-SE 2.0.12 + CTrnaAnnot::s_RepeatFilter parity (N1, I1, Blocker 8)
# ----------------------------------------------------------------------------------------------


def build_trnascan_cmd(
    trnascan_bin: str,
    threads_per_worker: int,
    gff_out: Path,
    tsv_out: Path,
    fasta_in: Path,
) -> list[str]:
    """Build direct tRNAscan-SE 2.0+ CLI invocation (strictly omitting -b, which is --bed in 2.0.12)."""
    return [
        str(trnascan_bin),
        "-E",
        "-X",
        "55",
        "-Q",
        "-q",
        "--thread",
        str(max(1, int(threads_per_worker))),
        "--gff",
        str(gff_out),
        "-o",
        str(tsv_out),
        str(fasta_in),
    ]


def parse_trnascan_gff(
    gff_path: str | Path,
    chunk_lookup: dict[str, dict] | None = None,
) -> list[dict]:
    """Parse ``tRNAscan-SE --gff`` dropping pseudogenes/Sup/Undet/<55 and lifting spliced exons on both strands."""
    p = Path(gff_path)
    if not p.exists() or p.stat().st_size == 0:
        return []

    parents: dict[str, dict] = {}
    parent_order: list[str] = []
    exons_by_parent: dict[str, list[tuple[int, int]]] = {}
    dropped_ids: set[str] = set()

    with open(p, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                continue
            seqid, _src, ftype, s_str, e_str, score_str, strand, _phase, attr_str = cols[:9]
            attrs = parse_attrs(attr_str, False)
            fid = attrs.get("ID", "")
            s1, e1 = int(s_str), int(e_str)
            if (
                ftype == "pseudogene"
                or attrs.get("pseudo", "").lower() == "true"
                or attrs.get("gene_biotype", "").lower() == "pseudogene"
            ):
                if fid:
                    dropped_ids.add(fid)
                continue
            if ftype == "tRNA":
                if strand not in ("+", "-") or score_str in ("", "."):
                    if fid:
                        dropped_ids.add(fid)
                    continue
                score = float(score_str)
                isotype = attrs.get("isotype", "").strip()
                anticodon = attrs.get("anticodon", "").strip().upper().replace("T", "U")
                note_txt = (attrs.get("Note", "") + " " + attrs.get("note", "")).lower()
                if score < 55.0 or isotype in ("", "Undet", "Sup") or not anticodon or "pseudo" in note_txt:
                    if fid:
                        dropped_ids.add(fid)
                    continue
                key_id = fid or f"{seqid}:{s1}-{e1}:{strand}"
                parents[key_id] = {
                    "seqid": seqid,
                    "strand": strand,
                    "s1": min(s1, e1),
                    "e1": max(s1, e1),
                    "score": score,
                    "isotype": isotype,
                    "anticodon": anticodon,
                }
                parent_order.append(key_id)
            elif ftype == "exon":
                pid = attrs.get("Parent", "")
                if not pid or pid in dropped_ids:
                    continue
                exons_by_parent.setdefault(pid, []).append((min(s1, e1), max(s1, e1)))

    out: list[dict] = []
    for fid in parent_order:
        pinfo = parents[fid]
        seqid = pinfo["seqid"]
        ch = chunk_lookup.get(seqid) if chunk_lookup is not None else None
        if ch is not None:
            orig_ctg = str(ch["contig"])
            offset = int(ch["lo"])
            chunk_len = int(ch["hi"]) - offset
            if ch.get("left_synthetic") and pinfo["s1"] <= 1:
                continue
            if ch.get("right_synthetic") and pinfo["e1"] >= chunk_len:
                continue
        else:
            orig_ctg = seqid
            offset = 0

        raw_ex = exons_by_parent.get(fid) or [(pinfo["s1"], pinfo["e1"])]
        exons = sorted((offset + es - 1, offset + ee) for es, ee in raw_ex if ee >= es)
        if not exons:
            continue
        start, end = exons[0][0], exons[-1][1]
        isotype = pinfo["isotype"]
        ac = pinfo["anticodon"]
        out.append(
            {
                "source_tool": "tRNAscan-SE",
                "kind": "functional",
                "biotype": "tRNA",
                "contig": orig_ctg,
                "strand": pinfo["strand"],
                "start": start,
                "end": end,
                "exons": [[a, b] for a, b in exons],
                "score": pinfo["score"],
                "isotype": isotype,
                "anticodon": ac,
                "rna_name": f"tRNA-{isotype}",
                "attrs": {
                    "isotype": _sanitize_gff_val(isotype),
                    "anticodon": _sanitize_gff_val(ac),
                    "product": _sanitize_gff_val(f"tRNA-{isotype}"),
                    "score": f"{pinfo['score']:.1f}",
                },
            }
        )
    return out


def dedup_trna_overlap_hits(hits: list[dict]) -> list[dict]:
    """Deduplicate identical/overlapping tRNA calls across 2 kb synthetic chunk overlaps."""
    if len(hits) <= 1:
        return list(hits)
    ordered = sorted(
        hits,
        key=lambda h: (
            str(h["contig"]),
            str(h["strand"]),
            -float(h["score"]),
            -(int(h["end"]) - int(h["start"])),
            int(h["start"]),
            int(h["end"]),
        ),
    )
    kept_by_cs: dict[tuple[str, str], list[dict]] = {}
    for h in ordered:
        cs = (str(h["contig"]), str(h["strand"]))
        s, e = int(h["start"]), int(h["end"])
        span = max(1, e - s)
        dup = False
        for prev in kept_by_cs.get(cs, ()):
            ps, pe = int(prev["start"]), int(prev["end"])
            ov = max(0, min(e, pe) - max(s, ps))
            if ov > 0 and ov / min(span, max(1, pe - ps)) >= 0.80:
                dup = True
                break
        if not dup:
            kept_by_cs.setdefault(cs, []).append(h)
    out = [h for v in kept_by_cs.values() for h in v]
    out.sort(key=lambda h: (str(h["contig"]), int(h["start"]), int(h["end"]), str(h["strand"])))
    return out


def apply_trnascan_repeat_filter(
    hits: list[dict],
    contig_order: dict[str, int] | list[str] | None = None,
    repeat_filter_factor: float = 3.0,
    repeat_filter_min: int = 20,
) -> tuple[list[dict], dict]:
    """EGAPx ``trnascan_dump`` genome-wide anticodon repeat filter with ``std::multimap`` tie-break parity (N1, I1)."""
    if not hits:
        return [], {
            "n_initial": 0,
            "n_kept": 0,
            "gmean": 0.0,
            "max_per_anticodon": int(repeat_filter_min),
        }
    if isinstance(contig_order, dict):
        c_ord = dict(contig_order)
    elif contig_order is not None:
        c_ord = {str(c): idx for idx, c in enumerate(contig_order)}
    else:
        c_ord = {}
    for h in hits:
        c = str(h["contig"])
        if c not in c_ord:
            c_ord[c] = len(c_ord)

    by_ac: dict[str, list[dict]] = {}
    for h in hits:
        ac = str(h.get("anticodon", "")).strip().upper().replace("T", "U")
        if not ac:
            continue
        norm_h = {**h, "anticodon": ac}
        by_ac.setdefault(ac, []).append(norm_h)

    if not by_ac:
        return [], {
            "n_initial": len(hits),
            "n_kept": 0,
            "gmean": 0.0,
            "max_per_anticodon": int(repeat_filter_min),
        }

    gmean = math.exp(sum(math.log(len(v)) for v in by_ac.values()) / len(by_ac))
    max_per_anticodon = max(int(repeat_filter_min), int(math.floor(float(repeat_filter_factor) * gmean)))

    kept: list[dict] = []
    for _ac, v in sorted(by_ac.items()):
        v_sorted = sorted(
            v,
            key=lambda h: (
                float(h["score"]),
                c_ord[str(h["contig"])],
                0 if h["strand"] == "+" else 1,
                int(h["start"]) if h["strand"] == "+" else -int(h["end"]),
            ),
        )
        if len(v_sorted) > max_per_anticodon:
            kept.extend(v_sorted[-max_per_anticodon:])
        else:
            kept.extend(v_sorted)

    kept.sort(key=lambda h: (c_ord[str(h["contig"])], int(h["start"]), int(h["end"]), str(h["strand"])))
    return kept, {
        "n_initial": sum(len(v) for v in by_ac.values()),
        "n_kept": len(kept),
        "gmean": round(gmean, 5),
        "max_per_anticodon": max_per_anticodon,
    }


# ----------------------------------------------------------------------------------------------
# §3.3 Step A (barrnap), Step B (nhmmer), Step C (cmsearch + strict A1)
# ----------------------------------------------------------------------------------------------


def read_barrnap_euk_lengths(barrnap_bin: str | Path | None = None) -> dict[str, int]:
    """Read exact ``LENG`` values for ``18S_rRNA``, ``28S_rRNA``, ``5_8S_rRNA`` from barrnap's ``euk.hmm``."""
    out = dict(DEFAULT_BARRNAP_EUK_LENG)
    cand_paths: list[Path] = []
    if barrnap_bin:
        bp = Path(barrnap_bin).resolve()
        cand_paths.append(bp.parent.parent / "db" / "euk.hmm")
        cand_paths.append(bp.parent.parent / "share" / "barrnap" / "db" / "euk.hmm")
    cand_paths.append(Path("/usr/share/barrnap/db/euk.hmm"))
    for hmmp in cand_paths:
        if not hmmp.is_file():
            continue
        name = ""
        with open(hmmp, "r") as fh:
            for line in fh:
                if line.startswith("NAME "):
                    name = line.split()[1]
                elif line.startswith("LENG ") and name in out:
                    out[name] = int(line.split()[1])
                elif line.strip() == "//":
                    name = ""
        break
    return out


def parse_barrnap_gff(
    gff_text: str,
    euk_hmm_leng: dict[str, int] | None = None,
) -> list[dict]:
    """Parse barrnap ``--kingdom euk --reject 0.25`` GFF3, discarding ``5S_rRNA`` and applying A2."""
    leng_map = euk_hmm_leng or DEFAULT_BARRNAP_EUK_LENG
    out: list[dict] = []
    for raw_line in gff_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 9 or cols[2] != "rRNA":
            continue
        ctg, _src, _ft, s_str, e_str, score_str, strand, _ph, attr_str = cols[:9]
        if strand not in ("+", "-"):
            continue
        attrs = parse_attrs(attr_str, False)
        name = attrs.get("Name", "").strip()
        if name == "5S_rRNA" or name not in leng_map:
            continue
        s1, e1 = int(s_str), int(e_str)
        start, end = min(s1, e1) - 1, max(s1, e1)
        if end <= start:
            continue
        cov = (end - start) / float(leng_map[name])
        if cov < 0.25:
            continue
        prod = attrs.get("product", name).split(" (partial)")[0].strip()
        if cov >= 0.80:
            out.append(
                {
                    "source_tool": "barrnap",
                    "kind": "functional",
                    "biotype": "rRNA",
                    "contig": ctg,
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "exons": [[start, end]],
                    "score": 0.0 if score_str in ("", ".") else float(score_str),
                    "cov": round(cov, 4),
                    "rna_name": name,
                    "attrs": {
                        "rna_name": _sanitize_gff_val(name),
                        "product": _sanitize_gff_val(prod),
                        "evidence": "barrnap",
                    },
                }
            )
        else:
            out.append(
                {
                    "source_tool": "barrnap",
                    "kind": "pseudo_ncrna",
                    "biotype": "rRNA_pseudogene",
                    "contig": ctg,
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "exons": [[start, end]],
                    "score": 0.0 if score_str in ("", ".") else float(score_str),
                    "cov": round(cov, 4),
                    "rna_name": name,
                    "attrs": {
                        "pseudo": "true",
                        "gbkey": "rRNA",
                        "rna_name": _sanitize_gff_val(name),
                        "product": _sanitize_gff_val(prod),
                        "evidence": "barrnap_partial",
                    },
                }
            )
    return out


def parse_nhmmer_tblout_to_windows(
    tbl_paths: list[Path],
    chunk_lookup: dict[str, dict],
    model_clen: dict[str, int],
    contig_lengths: dict[str, int],
) -> list[tuple[str, int, int]]:
    """Lift Stage-1 nhmmer envelopes, pad by ``max(250, clen + 100)``, and merge per contig (§3.3B)."""
    raw_by_ctg: dict[str, list[tuple[int, int]]] = {}
    for tp in tbl_paths:
        if not tp.exists():
            continue
        with open(tp, "r") as fh:
            for raw_line in fh:
                if not raw_line or raw_line.startswith("#"):
                    continue
                parts = raw_line.split(None, 15)
                if len(parts) < 15:
                    continue
                target_id = parts[0]
                qname = parts[2]
                if qname not in model_clen:
                    continue
                ch = chunk_lookup.get(target_id)
                orig_ctg = str(ch["contig"]) if ch is not None else target_id
                offset = int(ch["lo"]) if ch is not None else 0
                L = int(contig_lengths.get(orig_ctg, 0))
                if L <= 0:
                    continue
                env1, env2 = int(parts[8]), int(parts[9])
                s0 = offset + min(env1, env2) - 1
                e0 = offset + max(env1, env2)
                pad = max(250, int(model_clen[qname]) + 100)
                w_lo = max(0, s0 - pad)
                w_hi = min(L, e0 + pad)
                if w_hi > w_lo:
                    raw_by_ctg.setdefault(orig_ctg, []).append((w_lo, w_hi))

    merged_windows: list[tuple[str, int, int]] = []
    for ctg in contig_lengths:
        ivs = raw_by_ctg.get(ctg)
        if not ivs:
            continue
        for a, b in merge_intervals(ivs):
            merged_windows.append((ctg, a, b))
    return merged_windows


_CM_HIT_HDR_RE = re.compile(
    r"^\s*\(\d+\)\s+[!?]\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s+\d+\s+\S+\s+(\d+)\s+(\d+)\s+([+-])\s+"
)
_CM_ELISION_RE = re.compile(r"\*\[\s*(\d+)\s*\]\*")


def build_nhmmer_cmd(
    nhmmer_bin: str,
    threads: int,
    tbl_out: Path,
    sub_hmm: Path,
    fasta_in: Path,
) -> list[str]:
    """Build Stage-1 nhmmer CLI with explicit ``--rna`` alphabet flag (F4: prevents 0%-C AT-satellite abort)."""
    return [
        str(nhmmer_bin),
        "--rna",
        "--noali",
        "-Z",
        "160",
        "-E",
        "10.0",
        "--cpu",
        str(max(1, int(threads))),
        "--tblout",
        str(tbl_out),
        "-o",
        "/dev/null",
        str(sub_hmm),
        str(fasta_in),
    ]


def build_cmsearch_cmd(
    cmsearch_bin: str,
    threads: int,
    tbl_out: Path,
    cm_out: Path,
    sub_cm: Path,
    fasta_in: Path,
) -> list[str]:
    """Build Stage-2 cmsearch CLI capturing ``-o <cm_out>`` for F5 local-end ``[~~]`` elision detection."""
    return [
        str(cmsearch_bin),
        "--rfam",
        "--cut_ga",
        "--nohmmonly",
        "--tblout",
        str(tbl_out),
        "-o",
        str(cm_out),
        "--cpu",
        str(max(1, int(threads))),
        str(sub_cm),
        str(fasta_in),
    ]


def parse_cmsearch_local_end_elisions(
    cm_out_paths: list[Path],
    window_lookup: dict[str, tuple[str, int, int]] | None = None,
) -> set[tuple[str, str, str, int, int]]:
    """Parse ``cmsearch -o`` alignment blocks for internal local-end ``[~~]`` / ``*[ N ]*`` elisions (F5).

    Returns a set of ``(orig_contig, strand, mdl_name, orig_start, orig_end)`` 0-based half-open keys matching
    ``parse_and_classify_cmsearch_hits`` ``dkey`` tuples.
    """
    elided: set[tuple[str, str, str, int, int]] = set()
    for cp in cm_out_paths:
        if not cp.exists():
            continue
        cur_mdl = ""
        cur_target = ""
        in_alignments = False
        cur_hit_key: tuple[str, str, str, int, int] | None = None
        with open(cp, "r", errors="replace") as fh:
            for raw_line in fh:
                if raw_line.startswith("Query:"):
                    parts = raw_line.split()
                    cur_mdl = parts[1] if len(parts) >= 2 else ""
                    cur_target = ""
                    in_alignments = False
                    cur_hit_key = None
                    continue
                if raw_line.startswith("//") or raw_line.startswith("Internal CM pipeline statistics"):
                    in_alignments = False
                    cur_target = ""
                    cur_hit_key = None
                    if raw_line.startswith("//"):
                        cur_mdl = ""
                    continue
                if raw_line.startswith("Hit alignments:"):
                    in_alignments = True
                    cur_target = ""
                    cur_hit_key = None
                    continue
                if not in_alignments:
                    continue
                if raw_line.startswith(">>"):
                    t_parts = raw_line[2:].strip().split()
                    cur_target = t_parts[0] if t_parts else ""
                    cur_hit_key = None
                    continue
                if raw_line.lstrip().startswith("("):
                    m = _CM_HIT_HDR_RE.match(raw_line)
                    if m and cur_mdl and cur_target:
                        s1, s2 = int(m.group(1)), int(m.group(2))
                        strand = m.group(3)
                        if window_lookup is not None and cur_target in window_lookup:
                            orig_ctg, w_lo, _w_hi = window_lookup[cur_target]
                        else:
                            orig_ctg, w_lo = cur_target, 0
                        orig_start = w_lo + min(s1, s2) - 1
                        orig_end = w_lo + max(s1, s2)
                        cur_hit_key = (orig_ctg, strand, cur_mdl, orig_start, orig_end)
                        continue
                if cur_hit_key is not None:
                    if "*[" in raw_line and _CM_ELISION_RE.search(raw_line):
                        elided.add(cur_hit_key)
    return elided


def is_cmsearch_hit_complete_a1(
    *,
    mdl_from: int,
    mdl_to: int,
    clen: int,
    trunc: str,
    strand: str,
    orig_start: int,
    orig_end: int,
    contig_len: int,
    end_prox_bp: int = 9,
    has_local_end_elision: bool = False,
) -> bool:
    """Exact A1 model-end completeness rule + F5 internal local-end ``[~~]`` elision check."""
    if has_local_end_elision:
        return False
    dist_5p = orig_start if strand == "+" else (contig_len - orig_end)
    dist_3p = (contig_len - orig_end) if strand == "+" else orig_start
    complete_5p = (mdl_from == 1) or (dist_5p <= end_prox_bp)
    complete_3p = (mdl_to == clen) or (dist_3p <= end_prox_bp)
    return (trunc == "no" or dist_5p <= end_prox_bp or dist_3p <= end_prox_bp) and complete_5p and complete_3p


def parse_and_classify_cmsearch_hits(
    tbl_paths: list[Path],
    window_lookup: dict[str, tuple[str, int, int]] | None,
    selected_entries: dict[str, RfamEntry],
    model_clen: dict[str, int],
    clan_by_acc: dict[str, str],
    genome: Genome,
    cm_out_paths: list[Path] | None = None,
    local_end_elisions: set[tuple[str, str, str, int, int]] | None = None,
) -> list[dict]:
    """Parse Stage-2 cmsearch ``--tblout``, apply GA/5S/clan rules, trim terminal Ns, and classify via A1+F5."""
    elided_keys: set[tuple[str, str, str, int, int]] = set(local_end_elisions or ())
    if cm_out_paths:
        elided_keys |= parse_cmsearch_local_end_elisions(cm_out_paths, window_lookup=window_lookup)
    dedup_raw: dict[tuple[str, str, str, int, int], dict] = {}
    for tp in tbl_paths:
        if not tp.exists():
            continue
        with open(tp, "r") as fh:
            for raw_line in fh:
                if not raw_line or raw_line.startswith("#"):
                    continue
                parts = raw_line.split(None, 17)
                if len(parts) < 16:
                    continue
                target_id = parts[0]
                mdl_name = parts[2]
                mdl_acc = parts[3]
                entry = selected_entries.get(mdl_name)
                if entry is None or mdl_name not in model_clen:
                    continue
                strand = parts[9]
                if strand not in ("+", "-"):
                    continue
                bitscore = float(parts[14])
                evalue = float(parts[15])
                if bitscore < entry.ga_threshold:
                    continue
                if (mdl_name == "5S_rRNA" or mdl_acc == "RF00001") and bitscore < 75.0:
                    continue

                if window_lookup is not None and target_id in window_lookup:
                    orig_ctg, w_lo, _w_hi = window_lookup[target_id]
                else:
                    orig_ctg, w_lo = target_id, 0
                if orig_ctg not in genome.lengths:
                    continue

                m1, m2 = int(parts[5]), int(parts[6])
                mdl_from, mdl_to = min(m1, m2), max(m1, m2)
                s1, s2 = int(parts[7]), int(parts[8])
                orig_start = w_lo + min(s1, s2) - 1
                orig_end = w_lo + max(s1, s2)
                if orig_end <= orig_start:
                    continue
                trunc = parts[10]
                dkey = (orig_ctg, strand, mdl_name, orig_start, orig_end)
                has_elision = dkey in elided_keys
                cand = {
                    "contig": orig_ctg,
                    "strand": strand,
                    "mdl_name": mdl_name,
                    "mdl_acc": mdl_acc,
                    "mdl_from": mdl_from,
                    "mdl_to": mdl_to,
                    "orig_start": orig_start,
                    "orig_end": orig_end,
                    "trunc": trunc,
                    "has_local_end_elision": has_elision,
                    "bitscore": bitscore,
                    "evalue": evalue,
                    "rna_class": entry.rna_class,
                    "clan": clan_by_acc.get(mdl_acc, ""),
                }
                prev = dedup_raw.get(dkey)
                if prev is None or (
                    (0 if (trunc == "no" and not has_elision) else 1, -bitscore, evalue)
                    < (
                        0 if (prev["trunc"] == "no" and not prev.get("has_local_end_elision", False)) else 1,
                        -prev["bitscore"],
                        prev["evalue"],
                    )
                ):
                    dedup_raw[dkey] = cand

    ranked = sorted(
        dedup_raw.values(),
        key=lambda h: (-h["bitscore"], h["evalue"], h["mdl_name"], h["orig_start"], h["orig_end"], h["strand"]),
    )
    accepted_by_cs: dict[tuple[str, str], list[dict]] = {}
    accepted: list[dict] = []
    for h in ranked:
        cs = (h["contig"], h["strand"])
        hs, he = h["orig_start"], h["orig_end"]
        h_len = max(1, he - hs)
        h_clan = h["clan"]
        h_cls = h["rna_class"]
        conflict = False
        for acc_h in accepted_by_cs.get(cs, ()):
            if not ((h_clan and h_clan == acc_h["clan"]) or (h_cls == acc_h["rna_class"])):
                continue
            ov = max(0, min(he, acc_h["orig_end"]) - max(hs, acc_h["orig_start"]))
            if ov <= 0:
                continue
            shorter = min(h_len, max(1, acc_h["orig_end"] - acc_h["orig_start"]))
            if ov / shorter >= 0.20:
                conflict = True
                break
        if not conflict:
            accepted_by_cs.setdefault(cs, []).append(h)
            accepted.append(h)

    classified: list[dict] = []
    for h in accepted:
        ctg = h["contig"]
        L = int(genome.lengths[ctg])
        raw_s, raw_e = max(0, int(h["orig_start"])), min(L, int(h["orig_end"]))
        seq = genome.fetch(ctg, raw_s, raw_e)
        if not seq:
            continue
        l_trim = len(seq) - len(seq.lstrip("Nn"))
        r_trim = len(seq) - len(seq.rstrip("Nn"))
        start = raw_s + l_trim
        end = raw_e - r_trim
        if end <= start:
            continue

        mdl_name = h["mdl_name"]
        mdl_acc = h["mdl_acc"]
        entry = selected_entries[mdl_name]
        clen = int(model_clen[mdl_name])
        is_complete = is_cmsearch_hit_complete_a1(
            mdl_from=int(h["mdl_from"]),
            mdl_to=int(h["mdl_to"]),
            clen=clen,
            trunc=str(h["trunc"]),
            strand=str(h["strand"]),
            orig_start=start,
            orig_end=end,
            contig_len=L,
            end_prox_bp=int(entry.end_prox_bp),
            has_local_end_elision=bool(h.get("has_local_end_elision", False)),
        )
        bitscore = float(h["bitscore"])
        rna_class = entry.rna_class
        prod = _sanitize_gff_val(entry.product or mdl_name)
        rfam_attr = _sanitize_gff_val(f"{mdl_acc}:{mdl_name}" if mdl_acc and mdl_acc != "-" else mdl_name)

        if mdl_name == "5S_rRNA" or mdl_acc == "RF00001":
            if bitscore >= 100.0 and is_complete:
                kind = "functional"
                biotype = "rRNA"
                attrs = {
                    "rna_name": "5S_rRNA",
                    "rfam": rfam_attr,
                    "product": prod,
                    "score": f"{bitscore:.1f}",
                }
            else:
                kind = "pseudo_ncrna"
                biotype = "rRNA_pseudogene"
                attrs = {
                    "pseudo": "true",
                    "gbkey": "rRNA",
                    "rna_name": "5S_rRNA",
                    "rfam": rfam_attr,
                    "product": prod,
                    "score": f"{bitscore:.1f}",
                }
        elif mdl_name == "5_8S_rRNA" or mdl_acc == "RF00002":
            if is_complete:
                kind = "functional"
                biotype = "rRNA"
                attrs = {
                    "rna_name": "5_8S_rRNA",
                    "rfam": rfam_attr,
                    "product": prod,
                    "score": f"{bitscore:.1f}",
                }
            else:
                kind = "pseudo_ncrna"
                biotype = "rRNA_pseudogene"
                attrs = {
                    "pseudo": "true",
                    "gbkey": "rRNA",
                    "rna_name": "5_8S_rRNA",
                    "rfam": rfam_attr,
                    "product": prod,
                    "score": f"{bitscore:.1f}",
                }
        else:
            if is_complete:
                kind = "functional"
                biotype = rna_class
                attrs = {
                    "rna_name": _sanitize_gff_val(entry.gene_symbol or mdl_name),
                    "rfam": rfam_attr,
                    "product": prod,
                    "score": f"{bitscore:.1f}",
                }
            else:
                kind = "pseudo_ncrna"
                biotype = "ncRNA_pseudogene"
                attrs = {
                    "pseudo": "true",
                    "gbkey": "ncRNA",
                    "rna_name": _sanitize_gff_val(entry.gene_symbol or mdl_name),
                    "rfam": rfam_attr,
                    "product": prod,
                    "score": f"{bitscore:.1f}",
                }

        classified.append(
            {
                "source_tool": "cmsearch",
                "kind": kind,
                "biotype": biotype,
                "contig": ctg,
                "strand": h["strand"],
                "start": start,
                "end": end,
                "exons": [[start, end]],
                "score": bitscore,
                "mdl_from": int(h["mdl_from"]),
                "mdl_to": int(h["mdl_to"]),
                "clen": clen,
                "trunc": str(h["trunc"]),
                "has_local_end_elision": bool(h.get("has_local_end_elision", False)),
                "is_complete": is_complete,
                "rna_name": mdl_name,
                "attrs": attrs,
            }
        )
    classified.sort(key=lambda c: (str(c["contig"]), int(c["start"]), int(c["end"]), str(c["strand"]), str(c["rna_name"])))
    return classified


def reconcile_rrna_and_rfam_candidates(barrnap_hits: list[dict], cmsearch_hits: list[dict]) -> list[dict]:
    """Reconcile barrnap 18S/28S/5.8S with cmsearch 5S/5.8S/snRNA/snoRNA (§3.3C)."""
    cm_5_8s_by_cs: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for h in cmsearch_hits:
        if h.get("rna_name") == "5_8S_rRNA":
            cm_5_8s_by_cs.setdefault((str(h["contig"]), str(h["strand"])), []).append((int(h["start"]), int(h["end"])))

    intact_large_by_cs: dict[tuple[str, str], list[tuple[int, int]]] = {}
    kept_barrnap: list[dict] = []
    for bh in barrnap_hits:
        cs = (str(bh["contig"]), str(bh["strand"]))
        bs, be = int(bh["start"]), int(bh["end"])
        blen = max(1, be - bs)
        rname = bh.get("rna_name", "")
        if rname in ("18S_rRNA", "28S_rRNA") and bh.get("kind") == "functional":
            intact_large_by_cs.setdefault(cs, []).append((bs, be))
            kept_barrnap.append(bh)
            continue
        if rname == "5_8S_rRNA":
            dup_cm = any(
                max(0, min(be, ce) - max(bs, cs_)) / min(blen, max(1, ce - cs_)) >= 0.20
                for cs_, ce in cm_5_8s_by_cs.get(cs, ())
            )
            if dup_cm:
                continue
        kept_barrnap.append(bh)

    def _embedded_in_intact_large(h: dict) -> bool:
        if h.get("rna_name") not in ("5S_rRNA", "5_8S_rRNA"):
            return False
        cs = (str(h["contig"]), str(h["strand"]))
        hs, he = int(h["start"]), int(h["end"])
        hlen = max(1, he - hs)
        return any(
            max(0, min(he, le) - max(hs, ls)) / hlen >= 0.50
            for ls, le in intact_large_by_cs.get(cs, ())
        )

    out = [h for h in kept_barrnap if not _embedded_in_intact_large(h)] + [
        h for h in cmsearch_hits if not _embedded_in_intact_large(h)
    ]
    out.sort(key=lambda c: (str(c["contig"]), int(c["start"]), int(c["end"]), str(c["strand"]), str(c["biotype"])))
    return out


# ----------------------------------------------------------------------------------------------
# §3.4 Post-Seed Raw Runner/Cache & Dynamic Synthesis Exclusions
# ----------------------------------------------------------------------------------------------


def compute_ncrna_raw_cache_key(
    genome_identity: dict,
    taxon_group: str,
    stages: dict[str, bool] | None = None,
) -> str:
    eff_stages = {"trnascan": True, "barrnap": True, "rfam": True} if stages is None else dict(stages)
    payload: dict[str, object] = {
        "ncrna_raw_version": NCRNA_RAW_VERSION,
        "rfam_version": RFAM_PINNED_VERSION,
        "allowed_rfam_classes": sorted(ALLOWED_RFAM_CLASSES),
        "excluded_rfam_models": sorted(EXCLUDED_RFAM_MODELS),
        "pinned_rfam_assets": PINNED_RFAM_ASSETS,
        "taxon_group": taxon_group,
        "genome_identity": genome_identity,
    }
    if not all(eff_stages.values()):
        payload["stages"] = {k: bool(eff_stages.get(k, False)) for k in ("trnascan", "barrnap", "rfam")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def ensure_structural_ncrna_raw(
    *,
    genome: Genome,
    genome_fa: Path,
    genome_identity: dict,
    organelle_contigs: set[str],
    gate: dict,
    out_dir: Path,
    threads: int = 24,
    mode: str = "auto",
    rfam_dir: Path | None = None,
    log: StageLog | None = None,
) -> dict:
    """Run (or reuse cached) post-Seed genome-only structural ncRNA raw pipeline in ``seed/ncrna_raw/``."""
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"Invalid enable_structural_ncrna={mode!r}; expected 'auto', 'on', or 'off'")
    if mode == "off":
        return {
            "status": "disabled",
            "stages": {"trnascan": False, "barrnap": False, "rfam": False},
            "stages_run": [],
            "rfam_version": RFAM_PINNED_VERSION,
            "ncrna_raw_version": NCRNA_RAW_VERSION,
            "missing": [],
            "raw_candidates_path": "",
            "raw_candidates_sha256": "disabled",
            "n_raw_candidates": 0,
        }

    rdir_cand = Path(rfam_dir) if rfam_dir is not None else default_rfam_dir()
    for fname, exp_sha in PINNED_RFAM_ASSETS.items():
        fp = rdir_cand / fname
        if fp.is_file():
            got_sha = _sha256_cached(fp)
            if got_sha != exp_sha:
                raise RuntimeError(
                    f"Pinned Rfam {RFAM_PINNED_VERSION} SHA256 mismatch for {fp}: "
                    f"expected {exp_sha}, got {got_sha}"
                )

    asset_info = check_ncrna_assets(rfam_dir=rfam_dir, mode=mode)
    stages = asset_info.get("stages") or {"trnascan": True, "barrnap": True, "rfam": True}
    stages_run = [k for k in ("trnascan", "barrnap", "rfam") if stages.get(k)]
    if not asset_info["available"]:
        return {
            "status": asset_info["status"],
            "stages": stages,
            "stages_run": [],
            "rfam_version": RFAM_PINNED_VERSION,
            "ncrna_raw_version": NCRNA_RAW_VERSION,
            "missing": asset_info["missing"],
            "raw_candidates_path": "",
            "raw_candidates_sha256": asset_info["status"],
            "n_raw_candidates": 0,
        }

    raw_json_p = out_dir / "raw_ncrna_candidates.json"
    done_json_p = out_dir / "done.json"
    taxon_group = resolve_rfam_taxon_group(gate)
    cache_key = compute_ncrna_raw_cache_key(genome_identity, taxon_group, stages=stages)
    expected_shas = PINNED_RFAM_ASSETS if stages.get("rfam") else asset_info["asset_sha256"]

    if done_json_p.exists() and raw_json_p.exists():
        try:
            done_old = json.loads(done_json_p.read_text())
        except (OSError, ValueError):
            done_old = {}
        if (
            isinstance(done_old, dict)
            and done_old.get("cache_key") == cache_key
            and done_old.get("asset_sha256") == expected_shas
        ):
            actual_sha = _sha256_cached(raw_json_p)
            if done_old.get("raw_candidates_sha256") == actual_sha:
                return {
                    **done_old,
                    "status": asset_info["status"],
                    "cached": True,
                    "raw_candidates_path": str(raw_json_p),
                }

    out_dir.mkdir(parents=True, exist_ok=True)

    done_json_p.unlink(missing_ok=True)
    work_dir = out_dir / f"_work_{os.getpid()}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    t_total0 = time.time()
    n_threads = max(1, int(threads))
    rdir = Path(asset_info["rfam_dir"])
    bins = asset_info["binaries"]
    nuclear_contigs = [c for c in genome.lengths if c not in organelle_contigs and genome.lengths[c] > 0]
    contig_order = {c: idx for idx, c in enumerate(nuclear_contigs)}
    nuclear_bp = sum(int(genome.lengths[c]) for c in nuclear_contigs)

    try:
        if not nuclear_contigs:
            raw_candidates: list[dict] = []
            stage_stats = {
                "n_selected_models": 0,
                "trnascan_wall_s": 0.0,
                "trnascan_cpu_s": 0.0,
                "barrnap_wall_s": 0.0,
                "barrnap_cpu_s": 0.0,
                "nhmmer_wall_s": 0.0,
                "nhmmer_cpu_s": 0.0,
                "cmsearch_wall_s": 0.0,
                "cmsearch_cpu_s": 0.0,
                "candidate_windows_n": 0,
                "candidate_bp": 0,
                "candidate_pct": 0.0,
                "trna_repeat_filter": {"n_initial": 0, "n_kept": 0, "gmean": 0.0, "max_per_anticodon": 20},
            }
            tot_u = 0.0
            tot_s = 0.0
        else:
            # 1. Sharded tRNAscan-SE 2.0.12 + genome-wide anticodon repeat filter (when available)
            trna_wall = 0.0
            trna_u = 0.0
            trna_s = 0.0
            kept_trnas: list[dict] = []
            trna_rf_stats = {"n_initial": 0, "n_kept": 0, "gmean": 0.0, "max_per_anticodon": 20}
            if stages.get("trnascan"):
                t0 = time.time()
                trna_chunks = split_nuclear_chunks(genome, nuclear_contigs, max_chunk_bp=25_000_000, overlap_bp=2_000)
                trna_lookup = {ch["chunk_id"]: ch for ch in trna_chunks}
                w_trna = min(14, max(1, n_threads // 4))
                trna_shards = pack_chunks_lpt(trna_chunks, w_trna)
                w_trna_eff = max(1, len(trna_shards))
                t_per_trna = max(1, min(4, n_threads // w_trna_eff))
                trna_dir = work_dir / "trnascan"
                trna_dir.mkdir(parents=True, exist_ok=True)

                def _run_trna_shard(item: tuple[int, list[dict]]) -> tuple[Path, float, float]:
                    si, schunks = item
                    s_fa = trna_dir / f"shard_{si}.fa"
                    s_gff = trna_dir / f"shard_{si}.gff3"
                    s_tsv = trna_dir / f"shard_{si}.tsv"
                    _write_chunk_fasta(genome, schunks, s_fa)
                    cmd = build_trnascan_cmd(bins["tRNAscan-SE"], t_per_trna, s_gff, s_tsv, s_fa)
                    try:
                        _, _w, u, s = _run_ncrna_cmd(
                            cmd,
                            log_path=trna_dir / f"shard_{si}.log",
                            stage_label=f"tRNAscan-SE[shard_{si}]",
                        )
                    finally:
                        s_fa.unlink(missing_ok=True)
                    return s_gff, u, s

                with ThreadPoolExecutor(max_workers=w_trna_eff) as ex:
                    trna_outs = list(ex.map(_run_trna_shard, enumerate(trna_shards)))
                trna_wall = time.time() - t0
                trna_u = sum(x[1] for x in trna_outs)
                trna_s = sum(x[2] for x in trna_outs)
                raw_trnas: list[dict] = []
                for s_gff, _u, _s in trna_outs:
                    raw_trnas.extend(parse_trnascan_gff(s_gff, chunk_lookup=trna_lookup))
                dedup_trnas = dedup_trna_overlap_hits(raw_trnas)
                kept_trnas, trna_rf_stats = apply_trnascan_repeat_filter(dedup_trnas, contig_order=contig_order)

            # 2. Step A: Unsplit 18S / 28S / 5.8S scan via barrnap (when available)
            bar_wall = 0.0
            bar_u = 0.0
            bar_s = 0.0
            barrnap_hits: list[dict] = []
            if stages.get("barrnap"):
                t1 = time.time()
                t_bar = max(1, min(16, n_threads))
                if organelle_contigs:
                    nuc_fa = work_dir / "nuclear.fa"
                    with open(nuc_fa, "w") as fh:
                        for ctg in nuclear_contigs:
                            seq = genome.fetch(ctg, 0, genome.lengths[ctg])
                            fh.write(f">{ctg}\n")
                            for i in range(0, len(seq), 80):
                                fh.write(seq[i : i + 80] + "\n")
                else:
                    nuc_fa = genome_fa
                bar_gff = work_dir / "barrnap.gff3"
                bar_err = work_dir / "barrnap.err"
                try:
                    _, _wb, bar_u, bar_s = _run_ncrna_cmd(
                        [
                            bins["barrnap"],
                            "--kingdom",
                            "euk",
                            "--threads",
                            str(t_bar),
                            "--reject",
                            "0.25",
                            "--quiet",
                            str(nuc_fa),
                        ],
                        stdout_path=bar_gff,
                        log_path=bar_err,
                        stage_label="barrnap",
                    )
                finally:
                    if nuc_fa != genome_fa:
                        nuc_fa.unlink(missing_ok=True)
                bar_wall = time.time() - t1
                euk_leng = read_barrnap_euk_lengths(bins["barrnap"])
                barrnap_hits = parse_barrnap_gff(bar_gff.read_text(), euk_hmm_leng=euk_leng)

            # 3. Step B + Step C: Rfam 15.1 nhmmer prefilter + cmsearch (when Rfam assets + Infernal/HMMER available)
            n_selected_models = 0
            hmm_wall = 0.0
            hmm_u = 0.0
            hmm_s = 0.0
            cm_wall = 0.0
            cm_u = 0.0
            cm_s = 0.0
            merged_windows: list[tuple[str, int, int]] = []
            cand_bp = 0
            cm_hits: list[dict] = []
            if stages.get("rfam"):
                selected_entries = load_rfam_amendments(rdir / "rfam151_amendments.xml", taxon_group)
                n_selected_models = len(selected_entries)
                clan_by_acc = load_clan_membership(rdir / "clan_membership.txt.gz")
                sub_cm, sub_hmm, model_clen, _model_acc = ensure_rfam_subset_files(
                    rdir / "Rfam.cm.gz", work_dir / "models", selected_entries
                )

                t2 = time.time()
                hmm_chunks = split_nuclear_chunks(genome, nuclear_contigs, max_chunk_bp=25_000_000, overlap_bp=2_500)
                hmm_lookup = {ch["chunk_id"]: ch for ch in hmm_chunks}
                w_hmm = min(7, max(1, n_threads // 8))
                hmm_shards = pack_chunks_lpt(hmm_chunks, w_hmm)
                w_hmm_eff = max(1, len(hmm_shards))
                t_hmm = max(1, min(8, n_threads // w_hmm_eff))
                hmm_dir = work_dir / "nhmmer"
                hmm_dir.mkdir(parents=True, exist_ok=True)

                def _run_nhmmer_shard(item: tuple[int, list[dict]]) -> tuple[Path, float, float]:
                    si, schunks = item
                    s_fa = hmm_dir / f"shard_{si}.fa"
                    s_tbl = hmm_dir / f"shard_{si}.nhmmer.tbl"
                    _write_chunk_fasta(genome, schunks, s_fa)
                    cmd = build_nhmmer_cmd(bins["nhmmer"], t_hmm, s_tbl, sub_hmm, s_fa)
                    try:
                        _, _wh, u, s = _run_ncrna_cmd(
                            cmd,
                            log_path=hmm_dir / f"shard_{si}.err",
                            stage_label=f"nhmmer[shard_{si}]",
                        )
                    finally:
                        s_fa.unlink(missing_ok=True)
                    return s_tbl, u, s

                with ThreadPoolExecutor(max_workers=w_hmm_eff) as ex:
                    hmm_outs = list(ex.map(_run_nhmmer_shard, enumerate(hmm_shards)))
                hmm_wall = time.time() - t2
                hmm_u = sum(x[1] for x in hmm_outs)
                hmm_s = sum(x[2] for x in hmm_outs)
                nuc_lengths = {c: int(genome.lengths[c]) for c in nuclear_contigs}
                merged_windows = parse_nhmmer_tblout_to_windows(
                    [x[0] for x in hmm_outs], hmm_lookup, model_clen, nuc_lengths
                )
                cand_bp = sum(b - a for _, a, b in merged_windows)

                t3 = time.time()
                if merged_windows:
                    w_cm = min(4, max(1, n_threads // 14)) if len(merged_windows) >= 64 else 1
                    win_dicts = [
                        {"chunk_id": f"glwin_{wi}", "contig": ctg, "lo": a, "hi": b}
                        for wi, (ctg, a, b) in enumerate(merged_windows)
                    ]
                    win_lookup = {wd["chunk_id"]: (wd["contig"], int(wd["lo"]), int(wd["hi"])) for wd in win_dicts}
                    cm_shards = pack_chunks_lpt(win_dicts, w_cm)
                    w_cm_eff = max(1, len(cm_shards))
                    t_cm = max(1, min(16, n_threads // w_cm_eff))
                    cm_dir = work_dir / "cmsearch"
                    cm_dir.mkdir(parents=True, exist_ok=True)

                    def _run_cm_shard(
                        item: tuple[int, list[dict]],
                    ) -> tuple[Path, set[tuple[str, str, str, int, int]], float, float]:
                        si, swins = item
                        s_fa = cm_dir / f"batch_{si}.fa"
                        s_tbl = cm_dir / f"batch_{si}.cm.tbl"
                        s_out = cm_dir / f"batch_{si}.cm.out"
                        _write_chunk_fasta(genome, swins, s_fa)
                        cmd = build_cmsearch_cmd(bins["cmsearch"], t_cm, s_tbl, s_out, sub_cm, s_fa)
                        try:
                            _, _wc, u, s = _run_ncrna_cmd(
                                cmd,
                                log_path=cm_dir / f"batch_{si}.err",
                                stage_label=f"cmsearch[batch_{si}]",
                            )
                            elisions = parse_cmsearch_local_end_elisions([s_out], window_lookup=win_lookup)
                        finally:
                            s_fa.unlink(missing_ok=True)
                            s_out.unlink(missing_ok=True)
                        return s_tbl, elisions, u, s

                    with ThreadPoolExecutor(max_workers=w_cm_eff) as ex:
                        cm_outs = list(ex.map(_run_cm_shard, enumerate(cm_shards)))
                    cm_u = sum(x[2] for x in cm_outs)
                    cm_s = sum(x[3] for x in cm_outs)
                    all_elisions: set[tuple[str, str, str, int, int]] = set()
                    for _tbl, elis, _u, _s in cm_outs:
                        all_elisions |= elis
                    cm_hits = parse_and_classify_cmsearch_hits(
                        [x[0] for x in cm_outs],
                        win_lookup,
                        selected_entries,
                        model_clen,
                        clan_by_acc,
                        genome,
                        local_end_elisions=all_elisions,
                    )
                cm_wall = time.time() - t3

            rrna_and_rfam = reconcile_rrna_and_rfam_candidates(barrnap_hits, cm_hits)
            raw_candidates = sorted(
                kept_trnas + rrna_and_rfam,
                key=lambda c: (
                    contig_order.get(str(c["contig"]), 10**9),
                    int(c["start"]),
                    int(c["end"]),
                    str(c["strand"]),
                    str(c["biotype"]),
                ),
            )
            tot_u = trna_u + bar_u + hmm_u + cm_u
            tot_s = trna_s + bar_s + hmm_s + cm_s
            stage_stats = {
                "n_selected_models": n_selected_models,
                "trnascan_wall_s": round(trna_wall, 3),
                "trnascan_cpu_s": round(trna_u + trna_s, 3),
                "barrnap_wall_s": round(bar_wall, 3),
                "barrnap_cpu_s": round(bar_u + bar_s, 3),
                "nhmmer_wall_s": round(hmm_wall, 3),
                "nhmmer_cpu_s": round(hmm_u + hmm_s, 3),
                "cmsearch_wall_s": round(cm_wall, 3),
                "cmsearch_cpu_s": round(cm_u + cm_s, 3),
                "candidate_windows_n": len(merged_windows),
                "candidate_bp": cand_bp,
                "candidate_pct": round(100.0 * cand_bp / max(1, nuclear_bp), 4),
                "trna_repeat_filter": trna_rf_stats,
            }

        counts_by_bt: dict[str, int] = {}
        for c in raw_candidates:
            bt = str(c["biotype"])
            counts_by_bt[bt] = counts_by_bt.get(bt, 0) + 1

        _write_json_atomic(raw_json_p, raw_candidates)
        raw_sha = _sha256_cached(raw_json_p)
        tot_wall = time.time() - t_total0
        done_payload = {
            "status": asset_info["status"],
            "stages": stages,
            "stages_run": stages_run,
            "missing": asset_info["missing"],
            "ncrna_raw_version": NCRNA_RAW_VERSION,
            "rfam_version": RFAM_PINNED_VERSION,
            "taxon_group": taxon_group,
            "cache_key": cache_key,
            "asset_sha256": asset_info["asset_sha256"],
            "raw_candidates_sha256": raw_sha,
            "n_raw_candidates": len(raw_candidates),
            "counts_by_biotype": counts_by_bt,
            "nuclear_bp": nuclear_bp,
            "total_wall_s": round(tot_wall, 3),
            "total_cpu_s": round(tot_u + tot_s, 3),
            **stage_stats,
        }
        _write_json_atomic(done_json_p, done_payload)
        if log is not None:
            log.append(
                StageRecord(
                    name="structural_ncrna_raw",
                    wall_seconds=tot_wall,
                    cpu_user=tot_u,
                    cpu_sys=tot_s,
                    extra=done_payload,
                )
            )
        return {
            **done_payload,
            "cached": False,
            "raw_candidates_path": str(raw_json_p),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def filter_and_materialize_structural_ncrnas(
    raw_candidates: list[dict] | None,
    coding_models: list[Transcript],
    protein_pseudo_models: list[Transcript],
    organelle_contigs: set[str] | None = None,
    prefix: str = "GL",
    emit_pseudogenes: bool = True,
) -> tuple[list[Transcript], list[Transcript], dict]:
    """Apply §3.4 / F1 dynamic overlap exclusions against coding CDS & protein-pseudogene models.

    Rules:
      - Reject structural candidates overlapping (>= 1 bp) same-strand coding ``t.cds`` (NOT over-extended
        StringTie UTR exons with 0 bp CDS overlap) or same-strand protein-pseudogene exons.
      - For ``tRNA``, ``rRNA``, ``rRNA_pseudogene``, ``ncRNA_pseudogene``, also reject opposite-strand
        coding ``t.cds`` overlaps (>= 1 bp).
      - Retain intronic and UTR-only-adjacent ``snoRNA`` / ``snRNA`` / ``tRNA`` (0 bp CDS overlap).
    """
    diag = {
        "n_raw": len(raw_candidates or ()),
        "n_rejected_same_strand_exon": 0,
        "n_rejected_opp_strand_cds": 0,
        "n_functional": 0,
        "n_pseudo_ncrna": 0,
        "by_biotype": {"tRNA": 0, "rRNA": 0, "snRNA": 0, "snoRNA": 0},
        "pseudo_by_biotype": {"rRNA_pseudogene": 0, "ncRNA_pseudogene": 0},
    }
    if not raw_candidates:
        return [], [], diag

    org = organelle_contigs or set()
    same_exon_items: list[tuple[str, int, int, tuple[str, tuple[tuple[int, int], ...]]]] = []
    opp_cds_items: list[tuple[str, int, int, tuple[str, tuple[tuple[int, int], ...]]]] = []
    for tx in coding_models:
        if tx.cds:
            cds_t = tuple(sorted(tx.cds))
            same_exon_items.append((tx.contig, cds_t[0][0], cds_t[-1][1], (tx.strand, cds_t)))
            opp_cds_items.append((tx.contig, cds_t[0][0], cds_t[-1][1], (tx.strand, cds_t)))
    for pm in protein_pseudo_models:
        if pm.exons:
            ex_p = tuple(sorted(pm.exons))
            same_exon_items.append((pm.contig, ex_p[0][0], ex_p[-1][1], (pm.strand, ex_p)))

    same_exon_idx = LocusIndex(same_exon_items)
    opp_cds_idx = LocusIndex(opp_cds_items)
    opp_veto_biotypes = frozenset({"tRNA", "rRNA", "rRNA_pseudogene", "ncRNA_pseudogene"})

    kept_func: list[tuple[str, int, int, str, str, list[tuple[int, int]], dict[str, str]]] = []
    kept_pseudo: list[tuple[str, int, int, str, str, list[tuple[int, int]], dict[str, str]]] = []

    for cand in raw_candidates:
        ctg = str(cand["contig"])
        if ctg in org:
            continue
        strand = str(cand["strand"])
        if strand not in ("+", "-"):
            continue
        raw_ex = cand.get("exons") or [[int(cand["start"]), int(cand["end"])]]
        exons = sorted((int(a), int(b)) for a, b in raw_ex if int(b) > int(a))
        if not exons:
            continue
        ex_tup = tuple(exons)
        c_lo, c_hi = exons[0][0], exons[-1][1]
        bt = str(cand["biotype"])

        if any(
            ov_st == strand and interval_overlap_bp(ex_tup, ov_ex) >= 1
            for ov_st, ov_ex in same_exon_idx.overlapping(ctg, c_lo, c_hi)
        ):
            diag["n_rejected_same_strand_exon"] += 1
            continue

        if bt in opp_veto_biotypes and any(
            ov_st != strand and interval_overlap_bp(ex_tup, ov_cds) >= 1
            for ov_st, ov_cds in opp_cds_idx.overlapping(ctg, c_lo, c_hi)
        ):
            diag["n_rejected_opp_strand_cds"] += 1
            continue

        cattrs = {str(k): _sanitize_gff_val(v) for k, v in (cand.get("attrs") or {}).items()}
        if cand.get("kind") == "pseudo_ncrna" or bt.endswith("pseudogene"):
            if not emit_pseudogenes:
                continue
            cattrs["pseudo"] = "true"
            if "gbkey" not in cattrs:
                cattrs["gbkey"] = "rRNA" if bt == "rRNA_pseudogene" else "ncRNA"
            kept_pseudo.append((ctg, c_lo, c_hi, strand, bt, exons, cattrs))
        else:
            kept_func.append((ctg, c_lo, c_hi, strand, bt, exons, cattrs))

    kept_func.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
    kept_pseudo.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

    snc_pfx = "GLS" if prefix.startswith("GL") else f"{prefix}S"
    pnc_pfx = "GLNP" if prefix.startswith("GL") else f"{prefix}NP"

    func_models: list[Transcript] = []
    for i, (ctg, _lo, _hi, strand, bt, exons, cattrs) in enumerate(kept_func, 1):
        gid = f"{snc_pfx}{i:06d}"
        func_models.append(
            Transcript(
                tid=f"{gid}.t1",
                contig=ctg,
                strand=strand,
                exons=exons,
                cds=[],
                gid=gid,
                source="GeneLoop",
                attrs=cattrs,
                biotype=bt,
            )
        )
        diag["by_biotype"][bt] = diag["by_biotype"].get(bt, 0) + 1

    pseudo_models: list[Transcript] = []
    for i, (ctg, _lo, _hi, strand, bt, exons, cattrs) in enumerate(kept_pseudo, 1):
        gid = f"{pnc_pfx}{i:06d}"
        pseudo_models.append(
            Transcript(
                tid=f"{gid}.t1",
                contig=ctg,
                strand=strand,
                exons=exons,
                cds=[],
                gid=gid,
                source="GeneLoop",
                attrs=cattrs,
                biotype=bt,
            )
        )
        diag["pseudo_by_biotype"][bt] = diag["pseudo_by_biotype"].get(bt, 0) + 1

    diag["n_functional"] = len(func_models)
    diag["n_pseudo_ncrna"] = len(pseudo_models)
    return func_models, pseudo_models, diag
