"""Wave 0 seed: Tiberius (GPU) + genus-stripped miniprot + LOCKED/UNSETTLED partition.

Anti-leakage: the annotation engine never opens a RefSeq ``.gff`` truth file. This module only
reads the genome FASTA, the clade protein FASTA (target genus/taxid stripped and audited) and the
outputs of Tiberius / miniprot.
"""

from __future__ import annotations

import bisect
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import CANONICAL, LocusIndex, junction_motif
from .gffio import Transcript, _phases, interval_overlap_bp, introns_of, merge_intervals, parse_attrs, read_models
from .sequtil import Genome, is_scaffold_edge_partial_cds, orf_status, repair_orf, revcomp, translate
from .telemetry import GpuSampler, StageLog, StageRecord, run_cmd, safe_env

def _default_build_root() -> Path:
    env_br = os.environ.get("GENELOOP_BUILD_ROOT")
    if env_br:
        return Path(env_br)
    for cand in ("/opt/focaloop/geneloop", "/cvmfs/data.galaxyproject.org/byhand/focaloop/geneloop"):
        if Path(cand).exists():
            return Path(cand)
    return Path("/opt/focaloop/geneloop")


BUILD_ROOT = _default_build_root()
TIBERIUS_IMAGE = "gaiusaugustus/tiberius:2.0.9"

GENOME_KEYS = {
    "gate0_atha_120mb": "atha_GCF_000001735.4",
    "gate1_dmel_144mb": "dmel_GCF_000001215.4",
    "gate2_ggal_1050mb": "ggal_GCF_016700215.2",
    "gate3_hrus_vgp_1110mb": "hrus_GCF_015227805.2",
    "gate4a_hsap_chr19_chr22_110mb": "hsap_chr19_chr22_GCF_000001405.40",
    "gate4b_hsap_3100mb": "hsap_GCF_000001405.40",
    "gate4c_hsap_chr1_19_22_358mb": "hsap_chr1_chr19_chr22_GCF_000001405.40",
}


def load_manifest(path: str | Path | None = None) -> dict:
    if path is None:
        env_m = os.environ.get("GENELOOP_MANIFEST")
        p = Path(env_m) if env_m else Path(__file__).parent / "manifests" / "benchmark_panel.json"
    else:
        p = Path(path)
    return json.loads(p.read_text())


def gate_config(gate_id: str, manifest: dict | None = None) -> dict:
    m = manifest or load_manifest()
    for g in m["gates"]:
        if g["gate_id"] == gate_id:
            return g
    raise KeyError(gate_id)


def _resolve_build_rel(val: str) -> Path:
    p = Path(val)
    return p if p.is_absolute() else BUILD_ROOT / p


def _open_maybe_gz(path: Path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def genome_fasta_gz(gate_id: str, gate: dict | None = None) -> Path:
    g = gate if gate is not None else gate_config(gate_id)
    if g.get("fasta_path"):
        return Path(g["fasta_path"])
    if g.get("genome_fasta_gz"):
        return _resolve_build_rel(g["genome_fasta_gz"])
    k = GENOME_KEYS[gate_id]
    return BUILD_ROOT / "genomes" / k / f"{k}_genomic.fna.gz"


def truth_gff_gz(gate_id: str, gate: dict | None = None) -> Path:
    """ONLY for eval.py. The engine must never call this."""
    g = gate if gate is not None else gate_config(gate_id)
    if g.get("truth_gff_gz"):
        return _resolve_build_rel(g["truth_gff_gz"])
    k = GENOME_KEYS[gate_id]
    return BUILD_ROOT / "genomes" / k / f"{k}_genomic.gff.gz"


# ----------------------------------------------------------------------------------------------
# Genome preparation
# ----------------------------------------------------------------------------------------------

ORGANELLE_RE = re.compile(r"mitochondri|chloroplast|plastid|\bchrM\b|\bMT\b|apicoplast", re.I)


def _run_faidx(fa_path: Path) -> tuple[float, float]:
    if shutil.which("samtools"):
        _, _, u, s = run_cmd(["samtools", "faidx", str(fa_path)])
        return u, s
    import pysam

    pysam.faidx(str(fa_path))
    return 0.0, 0.0


def prepare_genome(gate_id: str, gate_dir: Path, log: StageLog, gate: dict | None = None) -> dict:
    gdir = gate_dir / "genome"
    gdir.mkdir(parents=True, exist_ok=True)
    fa = gdir / "genome.fna"
    meta_p = gdir / "genome_meta.json"
    if meta_p.exists() and fa.exists() and (gdir / "genome.fna.fai").exists():
        return json.loads(meta_p.read_text())
    t0 = time.time()
    gate = gate if gate is not None else gate_config(gate_id)
    src = genome_fasta_gz(gate_id, gate=gate)
    excl = re.compile(gate["exclude_contig_header_regex"]) if gate.get("exclude_contig_header_regex") else None
    organelles = []
    names = []
    excluded = []
    keep = True
    with _open_maybe_gz(src) as fin, open(fa, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                name = line[1:].split()[0]
                keep = not (excl and excl.search(line))
                if not keep:
                    excluded.append(name)
                    continue
                names.append(name)
                if ORGANELLE_RE.search(line):
                    organelles.append(name)
                fout.write(">" + name + "\n")
            elif keep:
                fout.write(line)
    rec = StageRecord(name="prepare_genome")
    u, s = _run_faidx(fa)
    rec.wall_seconds = time.time() - t0
    rec.cpu_user, rec.cpu_sys = u, s
    lengths = {}
    for line in (gdir / "genome.fna.fai").read_text().splitlines():
        f = line.split("\t")
        lengths[f[0]] = int(f[1])
    meta = {
        "fasta": str(fa),
        "n_contigs": len(names),
        "genome_bp": sum(lengths.values()),
        "organelle_contigs": organelles,
        "excluded_contigs": len(excluded),
        "exclude_regex": gate.get("exclude_contig_header_regex", ""),
        "source": str(src),
    }
    meta_p.write_text(json.dumps(meta, indent=1))
    rec.extra = {k: v for k, v in meta.items() if k != "source"}
    log.append(rec)
    return meta


# ----------------------------------------------------------------------------------------------
# Anti-leakage protein stripping
# ----------------------------------------------------------------------------------------------

_TAXID_RE = re.compile(r"\[taxid=(\d+)\]")
_ORG_RE = re.compile(r"\[organism=([^\]]+)\]")
_ODB_HEADER_TAX_RE = re.compile(r"^>?(\d+)(?:_(\d+))?:")


def seed_protein_path(seed_dir: Path, gate: dict) -> Path:
    """Canonical stripped-protein FASTA path under ``seed_dir`` (defaults blank ``exclude_genus`` to ``'target'``)."""
    excl = str(gate.get("exclude_genus") or "").strip() or "target"
    return Path(seed_dir) / f"proteins_no_{excl}.faa"


def _normalize_exclude_taxid_tokens(raw_taxids: set[int] | set[str] | list | tuple | None, gate: dict | None = None) -> set[str]:
    tokens: set[str] = set()
    for item in (raw_taxids or set()):
        s = str(item).strip()
        if s:
            tokens.add(s)
            tokens.add(s.split("_", 1)[0])
    if gate:
        for key in ("taxid", "tax_id"):
            if gate.get(key) is not None and str(gate[key]).strip():
                t = str(gate[key]).strip()
                tokens.add(t)
                tokens.add(t.split("_", 1)[0])
        extra = gate.get("exclude_taxids")
        if isinstance(extra, str):
            for tok in re.split(r"[\s,;]+", extra):
                if tok.strip():
                    tokens.add(tok.strip())
                    tokens.add(tok.strip().split("_", 1)[0])
        elif isinstance(extra, (list, tuple, set, frozenset)):
            for tok in extra:
                s = str(tok).strip()
                if s:
                    tokens.add(s)
                    tokens.add(s.split("_", 1)[0])
    return {x for x in tokens if x and x != "0" and not x.startswith("0_")}


def strip_genus_proteins(
    src_faa_gz: Path,
    out_faa: Path,
    exclude_genus: str,
    exclude_taxids: set[int] | set[str],
    species_name: str,
    include_taxons: list[str] | None = None,
    readthrough_ids_out: set[str] | None = None,
) -> dict:
    """Filter protein bag to ``include_taxons`` and strip target genus/species/taxids (including OrthoDB ``TAXID_N:``).

    A header is removed if ANY of:
      * ``include_taxons`` is non-empty and none of the taxon names match ``[<taxon>]`` or ``[organism=<taxon>]``
      * ``[organism=<genus> ...]``                  (organism field genus match)
      * ``[taxid=N]`` or ``>TAXID_M:`` in ``exclude_taxids``
      * the token ``<genus>`` appears as a whole word anywhere in the header (when ``exclude_genus`` is non-empty)
      * the binomial species name appears anywhere in the header
    """
    excl_gen_clean = (exclude_genus or "").strip()
    genus_word = re.compile(r"\b" + re.escape(excl_gen_clean) + r"\b") if excl_gen_clean else None
    excl_tax_strs = _normalize_exclude_taxid_tokens(exclude_taxids)
    species_l = (species_name or "").strip().lower()
    inc_re = (
        re.compile(r"\[(?:organism=)?(?:" + "|".join(re.escape(t) for t in include_taxons) + r")\]")
        if include_taxons
        else None
    )
    n_in = n_out = 0
    n_odb_tax_headers_kept = 0
    n_org_headers_kept = 0
    by_reason = {
        "not_in_filter_taxons": 0,
        "organism_genus": 0,
        "taxid": 0,
        "orthodb_taxid_prefix": 0,
        "header_genus_token": 0,
        "species_name": 0,
    }
    removed_taxids: dict[str, int] = {}
    keep = False
    with _open_maybe_gz(src_faa_gz) as fin, open(out_faa, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                n_in += 1
                hdr = line
                tok0 = hdr[1:].split()[0] if hdr[1:].strip() else ""
                acc = tok0.split("|")[1] if tok0.startswith("ref|") and "|" in tok0 else tok0
                odb_m = _ODB_HEADER_TAX_RE.match(tok0)
                odb_base_tax = odb_m.group(1) if odb_m else ""
                odb_full_tax = tok0.split(":", 1)[0] if (odb_m and ":" in tok0) else ""

                reason = None
                if inc_re is not None and not inc_re.search(hdr):
                    reason = "not_in_filter_taxons"
                m = _ORG_RE.search(hdr)
                if reason is None and excl_gen_clean and m and m.group(1).split()[0].lower() == excl_gen_clean.lower():
                    reason = "organism_genus"
                if reason is None and excl_tax_strs:
                    tm = _TAXID_RE.search(hdr)
                    if tm and tm.group(1) in excl_tax_strs:
                        reason = "taxid"
                    elif (odb_base_tax and odb_base_tax in excl_tax_strs) or (odb_full_tax and odb_full_tax in excl_tax_strs):
                        reason = "orthodb_taxid_prefix"
                if reason is None and genus_word is not None and genus_word.search(hdr):
                    reason = "header_genus_token"
                if reason is None and species_l and species_l in hdr.lower():
                    reason = "species_name"
                keep = reason is None
                if keep:
                    n_out += 1
                    if odb_m:
                        n_odb_tax_headers_kept += 1
                    if m is not None:
                        n_org_headers_kept += 1
                    if readthrough_ids_out is not None and "readthrough" in hdr.lower() and acc:
                        readthrough_ids_out.add(acc)
                    fout.write(">" + acc + "\n")
                else:
                    by_reason[reason] += 1
                    if reason != "not_in_filter_taxons":
                        tm = _TAXID_RE.search(hdr)
                        rec_tid = tm.group(1) if tm else (odb_base_tax or "")
                        if rec_tid:
                            removed_taxids[rec_tid] = removed_taxids.get(rec_tid, 0) + 1
            elif keep:
                fout.write(line)
    # Post-hoc verification pass: zero remaining headers may match excluded genus or excluded TAXID_N: prefixes.
    residual_genus = 0
    residual_taxid = 0
    with open(out_faa) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            if genus_word is not None and genus_word.search(line):
                residual_genus += 1
            if excl_tax_strs:
                tok = line[1:].strip().split()[0] if line[1:].strip() else ""
                om = _ODB_HEADER_TAX_RE.match(tok)
                if om:
                    bt = om.group(1)
                    ft = tok.split(":", 1)[0]
                    if bt in excl_tax_strs or ft in excl_tax_strs:
                        residual_taxid += 1
    residual = residual_genus + residual_taxid
    if residual > 0:
        audit_status = "leakage_detected"
    elif n_out > 0 and n_odb_tax_headers_kept > 0 and n_org_headers_kept == 0 and not excl_tax_strs:
        audit_status = "unverifiable"
    else:
        audit_status = "verified_clean"
    return {
        "source": str(src_faa_gz),
        "exclude_genus": excl_gen_clean,
        "exclude_taxids": sorted(excl_tax_strs),
        "include_taxons": list(include_taxons) if include_taxons else None,
        "proteins_pre": n_in,
        "proteins_post": n_out,
        "proteins_removed": n_in - n_out,
        "removed_by_reason": by_reason,
        "removed_taxid_counts_top": dict(sorted(removed_taxids.items(), key=lambda kv: -kv[1])[:20]),
        "residual_genus_headers_after_strip": residual_genus,
        "residual_taxid_headers_after_strip": residual_taxid,
        "leakage_audit_status": audit_status,
    }


def leakage_audit(gate: dict, gate_dir: Path, log: StageLog) -> dict:
    seed_dir = gate_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_faa = seed_protein_path(seed_dir, gate)
    audit_p = seed_dir / "leakage_audit.json"
    rt_json_p = seed_dir / "readthrough_targets.json"
    if audit_p.exists() and out_faa.exists() and rt_json_p.exists():
        return json.loads(audit_p.read_text())
    t0 = time.time()
    rt_ids: set[str] = set()
    excl_tax_tokens = _normalize_exclude_taxid_tokens(None, gate=gate)
    raw_excl_genus = str(gate.get("exclude_genus") or "").strip()
    sp_clean = str(gate.get("species") or "").split(" (")[0].strip()

    if gate.get("proteins_path"):
        src = Path(gate["proteins_path"])
        res = strip_genus_proteins(
            src,
            out_faa,
            raw_excl_genus,
            excl_tax_tokens,
            sp_clean,
            include_taxons=gate.get("proteins_filter_taxons"),
            readthrough_ids_out=rt_ids,
        )
    elif gate.get("egapx_protein_bin"):
        src = BUILD_ROOT / "proteins" / gate["egapx_protein_bin"]
        res = strip_genus_proteins(
            src,
            out_faa,
            raw_excl_genus,
            excl_tax_tokens,
            sp_clean,
            include_taxons=gate.get("proteins_filter_taxons"),
            readthrough_ids_out=rt_ids,
        )
    else:
        out_faa.write_text("")
        res = {
            "source": "",
            "exclude_genus": raw_excl_genus or "target",
            "exclude_taxids": sorted(excl_tax_tokens),
            "include_taxons": None,
            "proteins_pre": 0,
            "proteins_post": 0,
            "proteins_removed": 0,
            "removed_by_reason": {},
            "removed_taxid_counts_top": {},
            "residual_genus_headers_after_strip": 0,
            "residual_taxid_headers_after_strip": 0,
            "leakage_audit_status": "verified_clean",
        }
    res["readthrough_proteins"] = len(rt_ids)
    res["self_protein_leakage"] = (
        res.get("residual_genus_headers_after_strip", 0) + res.get("residual_taxid_headers_after_strip", 0)
    ) != 0
    if res["self_protein_leakage"]:
        raise RuntimeError(
            f"Leakage audit detected residual excluded headers after stripping {out_faa}: {res}"
        )
    res["tiberius_training_audit"] = tiberius_training_audit(gate)
    if out_faa.stat().st_size > 0:
        _run_faidx(out_faa)
    else:
        Path(str(out_faa) + ".fai").write_text("")
    rt_tmp = rt_json_p.with_name(f".{rt_json_p.name}.tmp.{os.getpid()}")
    rt_tmp.write_text(json.dumps({"ids": sorted(rt_ids)}, indent=1))
    os.replace(rt_tmp, rt_json_p)
    audit_p.write_text(json.dumps(res, indent=1))
    rec = StageRecord(name="leakage_strip", wall_seconds=time.time() - t0, extra=res)
    log.append(rec)
    return res


def tiberius_training_audit(gate: dict) -> dict:
    cfg = BUILD_ROOT / "Tiberius" / "model_cfg" / f"{gate.get('tiberius_model_cfg', '')}.yaml"
    if not cfg.is_file():
        return {
            "model_cfg": gate.get("tiberius_model_cfg", ""),
            "n_training_species": 0,
            "target_species_in_training": False,
            "same_genus_training_species": [],
            "cfg_available": False,
        }
    txt = cfg.read_text()
    block = txt[txt.find("training_species:"):]
    names = re.findall(r"\b([A-Z][a-z]+_[a-z_]+)\b", block)
    species = gate.get("species", "").split(" (")[0].replace(" ", "_")
    genus = gate.get("exclude_genus", "")
    return {
        "model_cfg": gate.get("tiberius_model_cfg", ""),
        "n_training_species": len(names),
        "target_species_in_training": species in names,
        "same_genus_training_species": [n for n in names if genus and n.split("_")[0] == genus],
        "cfg_available": True,
    }


# ----------------------------------------------------------------------------------------------
# Tiberius (GPU, docker)
# ----------------------------------------------------------------------------------------------

_RUSAGE_WRAPPER = (
    "import resource,subprocess,sys,json,time;t=time.time();"
    "r=subprocess.run(sys.argv[2:]).returncode;"
    "ru=resource.getrusage(resource.RUSAGE_CHILDREN);"
    "json.dump({'rc':r,'user':ru.ru_utime,'sys':ru.ru_stime,'wall':time.time()-t,'maxrss_kb':ru.ru_maxrss},"
    "open(sys.argv[1],'w'));sys.exit(r)"
)

# Work around TensorFlow/oneDNN 2.17 CPU stalls (~8.5 min per chunk) when batch==1 by padding to batch==2 and slicing [:1].
_CPU_PAD_EXEC = "\n".join([
    "import runpy, sys, numpy as np",
    "if '/opt/Tiberius' not in sys.path: sys.path.insert(0, '/opt/Tiberius')",
    "from tiberius.eval_model_class import PredictionGTF",
    "_orig = PredictionGTF.predict_lstm_batch",
    "def _pad(self, b):",
    "    a = b[0] if isinstance(b, (list, tuple)) else b",
    "    if getattr(a, 'shape', (0,))[0] != 1:",
    "        return _orig(self, b)",
    "    b2 = type(b)(np.concatenate([x, x], axis=0) for x in b) if isinstance(b, (list, tuple)) else np.concatenate([b, b], axis=0)",
    "    out = _orig(self, b2)",
    "    return type(out)(r[:1] for r in out) if isinstance(out, (list, tuple)) else out[:1]",
    "PredictionGTF.predict_lstm_batch = _pad",
    "sys.argv = sys.argv[1:]",
    "runpy.run_path(sys.argv[0], run_name='__main__')",
])

_TIBERIUS_IMG_ID: str | None = None


def _tiberius_image_id() -> str:
    """Return cached Docker image digest/ID for TIBERIUS_IMAGE ('unknown' if unavailable)."""
    global _TIBERIUS_IMG_ID
    if _TIBERIUS_IMG_ID is None:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", TIBERIUS_IMAGE],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            _TIBERIUS_IMG_ID = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
        except Exception:
            _TIBERIUS_IMG_ID = "unknown"
    return _TIBERIUS_IMG_ID


def _compute_tiberius_input_sha256(
    genome_fa: Path,
    model_cfg: str,
    batch_size: int,
    seq_len: int | None,
    cpu_only: bool = False,
) -> str:
    """Compute G-A1/R2/Y5 input SHA256 over genome FASTA bytes, image digest, model_cfg, batch_size, seq_len, cpu_only."""
    h = hashlib.sha256()
    if genome_fa.exists():
        with open(genome_fa, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    suffix = f"|{TIBERIUS_IMAGE}|{_tiberius_image_id()}|{model_cfg}|{batch_size}|{seq_len}|{bool(cpu_only)}".encode("utf-8")
    h.update(suffix)
    return h.hexdigest()


def _compute_manifest_tiberius_input_sha256(
    genome: Genome,
    manifest: dict,
    model_cfg: str,
    batch_size: int,
    seq_len: int | None,
    cpu_only: bool = True,
) -> str:
    """Recompute exact G-A1/X1 supercontig input SHA256 from genome + manifest without writing temporary FASTA files."""
    h = hashlib.sha256()
    spacer_bytes = b"N" * int(manifest.get("spacer_bp", 4_500))
    for sid, info in (manifest.get("super_contigs") or {}).items():
        segs = sorted(info.get("segments") or (), key=lambda s: int(s["super_lo"]))
        if not segs:
            continue
        h.update(f">{sid}\n".encode("utf-8"))
        for idx, seg in enumerate(segs):
            if idx > 0:
                h.update(spacer_bytes)
            h.update(
                genome.fetch(str(seg["orig_contig"]), int(seg["orig_lo"]), int(seg["orig_hi"])).encode("utf-8")
            )
        h.update(b"\n")
    suffix = f"|{TIBERIUS_IMAGE}|{_tiberius_image_id()}|{model_cfg}|{batch_size}|{seq_len}|{bool(cpu_only)}".encode("utf-8")
    h.update(suffix)
    return h.hexdigest()


_TIBERIUS_WEIGHT_NAMES: dict[str, tuple[str, ...]] = {
    "vertebrates": ("vertebrates_weights",),
    "mammalia_nosofttmasking_v2": ("tiberius_nosm_weights_v2",),
    "mammalia_softmasking_v2": ("tiberius_weights_v2", "mammalia_softmasking_v2"),
    "mammalia_clamsa_v2": ("tiberius_clamsa_weights_v2", "mammalia_clamsa_v2"),
    "insecta": ("insecta_unmasked_v2",),
    "angiosperms": ("angiosperms_weights", "angiosperms"),
    "fungi": ("fungi_weights", "fungi"),
    "chlorophyta": ("chlorophyta_weights", "chlorophyta"),
    "diatoms": ("diatoms_weights", "diatoms"),
}

_TIBERIUS_UNMASKED_CFGS = frozenset({"vertebrates", "insecta", "mammalia_nosofttmasking_v2"})
_TIBERIUS_DEFAULT_SEQLEN = {"vertebrates": 400_050}


def gpu_available() -> bool:
    """Return True iff at least one NVIDIA GPU is usable via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip()) and "Failed" not in r.stdout
    except Exception:
        return False


def _safe_extract_tar(tf, dest: Path) -> None:
    """Extract only validated regular files and directories strictly inside ``dest`` (rejecting traversal/links)."""
    dest_resolved = dest.resolve()
    members = tf.getmembers()
    for m in members:
        name = m.name or ""
        if not name or name.startswith("/") or name.startswith("\\"):
            raise ValueError(f"Unsafe absolute path in Tiberius weights archive: {name!r}")
        rel_parts = Path(name).parts
        if ".." in rel_parts:
            raise ValueError(f"Unsafe parent-directory traversal in Tiberius weights archive: {name!r}")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"Disallowed non-regular tar member (symlink/hardlink/device) in weights archive: {name!r}")
        target_p = (dest_resolved / name).resolve()
        if target_p != dest_resolved and dest_resolved not in target_p.parents:
            raise ValueError(f"Tar member escapes destination directory: {name!r}")
    try:
        tf.extractall(dest_resolved, members=members, filter="data")
    except TypeError:
        for m in members:
            out_p = (dest_resolved / m.name).resolve()
            if m.isdir():
                out_p.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                out_p.parent.mkdir(parents=True, exist_ok=True)
                src_fh = tf.extractfile(m)
                if src_fh is not None:
                    with src_fh, open(out_p, "wb") as dst_fh:
                        shutil.copyfileobj(src_fh, dst_fh)


def discover_tiberius_weights(model_cfg: str, explicit_weights: str | Path | None = None) -> Path | None:
    """Discover unpacked Tiberius model weights directory (with weights.h5 / model_config.json)."""
    cands_names = _TIBERIUS_WEIGHT_NAMES.get(model_cfg, (model_cfg, f"{model_cfg}_weights"))

    def _is_weights_dir(p: Path) -> bool:
        return p.is_dir() and ((p / "weights.h5").is_file() or (p / "model_config.json").is_file())

    if explicit_weights:
        ep = Path(explicit_weights)
        if _is_weights_dir(ep):
            return ep.resolve()
        if ep.is_dir():
            for nm in cands_names:
                sub = ep / nm
                if _is_weights_dir(sub):
                    return sub.resolve()
        if ep.is_file():
            import tarfile
            import tempfile

            if tarfile.is_tarfile(ep):
                dest = Path(tempfile.mkdtemp(prefix="tiberius_weights_"))
                try:
                    with tarfile.open(ep, "r:*") as tf:
                        _safe_extract_tar(tf, dest)
                except Exception:
                    shutil.rmtree(dest, ignore_errors=True)
                    raise
                if _is_weights_dir(dest):
                    return dest.resolve()
                for sub in dest.iterdir():
                    if _is_weights_dir(sub):
                        return sub.resolve()

    search_roots: list[Path] = []
    for env_k in ("GENELOOP_TIBERIUS_WEIGHTS", "TIBERIUS_WEIGHTS_DIR"):
        v = os.environ.get(env_k, "").strip()
        if v:
            search_roots.append(Path(v))
    search_roots.extend([
        Path("/opt/Tiberius/model_weights"),
        Path("/opt/focaloop/tiberius_weights"),
        Path("/cvmfs/data.galaxyproject.org/byhand/tiberius_weights"),
        BUILD_ROOT / "weights",
    ])
    for root in search_roots:
        if _is_weights_dir(root) and any(nm in root.name for nm in cands_names):
            return root.resolve()
        if root.is_dir():
            for nm in cands_names:
                sub = root / nm
                if _is_weights_dir(sub):
                    return sub.resolve()
    return None


def detect_tiberius_runtime() -> dict[str, Any]:
    """Detect available Tiberius execution backend: native python, apptainer sif, or docker."""
    env_cmd = os.environ.get("GENELOOP_TIBERIUS_CMD", "").strip()
    if env_cmd and Path(env_cmd).is_file():
        return {"backend": "native", "script": str(Path(env_cmd).resolve()), "python": sys.executable}
    for cand in (shutil.which("tiberius.py"), "/opt/Tiberius/tiberius.py"):
        if cand and Path(cand).is_file():
            try:
                import importlib.util

                if importlib.util.find_spec("tensorflow") is not None:
                    return {"backend": "native", "script": str(Path(cand).resolve()), "python": sys.executable}
            except Exception:
                pass
    env_sif = os.environ.get("GENELOOP_TIBERIUS_SIF", "").strip()
    sif_cands = [Path(env_sif)] if env_sif else []
    sif_cands.extend([
        BUILD_ROOT / "sif" / "tiberius_2.0.9.sif",
        Path("/opt/focaloop/sif/tiberius_2.0.9.sif"),
    ])
    app_exe = shutil.which("apptainer") or shutil.which("singularity")
    if app_exe:
        for sif in sif_cands:
            if sif.is_file():
                return {"backend": "apptainer", "exe": app_exe, "sif": str(sif.resolve())}
    if shutil.which("docker"):
        try:
            r = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False, timeout=4)
            if r.returncode == 0:
                return {"backend": "docker", "image": TIBERIUS_IMAGE}
        except Exception:
            pass
    return {"backend": "none"}


def run_tiberius(
    gate: dict,
    genome_fa: Path,
    out_dir: Path,
    log: StageLog,
    batch_size: int = 4,
    tag: str = "",
    seq_len: int | None = None,
    cpu_only: bool = False,
    cpu_threads: int = 24,
    cpuset_cpus: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_meta: dict | None = None,
) -> StageRecord:
    out_dir = Path(out_dir).resolve()
    genome_fa = Path(genome_fa).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"bs{batch_size}"
    gtf = out_dir / f"tiberius_{tag}.gtf"
    done = out_dir / f"tiberius_{tag}.done.json"
    model_cfg = gate.get("tiberius_model_cfg", "")
    input_sha256 = (
        _compute_tiberius_input_sha256(genome_fa, model_cfg, batch_size, seq_len, cpu_only=cpu_only)
        if genome_fa.exists()
        else ""
    )
    if done.exists() and gtf.exists() and input_sha256:
        try:
            done_data = json.loads(done.read_text())
        except (OSError, ValueError):
            done_data = {}
        cached_sha = done_data.get("input_sha256") or (done_data.get("extra") or {}).get("input_sha256")
        if cached_sha == input_sha256:
            rec_cached = StageRecord(**{k: v for k, v in done_data.items() if k in StageRecord.__dataclass_fields__})
            if extra_meta:
                rec_cached.extra = {**(rec_cached.extra or {}), **extra_meta}
            return rec_cached
    has_seq = genome_fa.exists() and genome_fa.stat().st_size > 0
    if has_seq:
        with genome_fa.open() as fh:
            has_seq = any(ln.strip() and not ln.startswith(">") for ln in fh)
    if not has_seq:
        gtf.write_text("")
        (out_dir / f"tiberius_{tag}.faa").write_text("")
        rec = StageRecord(
            name=f"tiberius_{tag}",
            extra={
                "empty_input": True,
                "rc": 0,
                "batch_size": batch_size,
                "cpu_only": cpu_only,
                "seq_len": seq_len,
                "gtf": str(gtf),
                "input_sha256": input_sha256,
                **(extra_meta or {}),
            },
        )
        log.append(rec)
        done.write_text(json.dumps({**rec.to_dict(), "input_sha256": input_sha256}))
        return rec

    rt = detect_tiberius_runtime()
    if rt["backend"] == "none":
        raise RuntimeError(
            "Tiberius runtime is not available in this environment (neither native tiberius.py+tensorflow, "
            "Apptainer SIF, nor Docker). Use --mode fast_evidence or install Tiberius + weights."
        )

    ru_json = out_dir / f"tiberius_{tag}.rusage.json"
    base_tmp = out_dir / ".tib_tmp"
    base_tmp.mkdir(parents=True, exist_ok=True)
    tmp = base_tmp / f"tib_{tag}_{os.getpid()}_{threading.get_ident()}"
    tmp.mkdir(parents=True, exist_ok=True)

    env_map: dict[str, str] = {
        "TMPDIR": str(tmp),
        "HOME": str(tmp),
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_USE_LEGACY_KERAS": "0",
    }
    if cpu_only:
        onednn = "0" if model_cfg == "vertebrates" else "1"
        env_map.update({
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(cpu_threads),
            "TF_NUM_INTRAOP_THREADS": str(cpu_threads),
            "TF_NUM_INTEROP_THREADS": "2",
            "TF_ENABLE_ONEDNN_OPTS": onednn,
        })
        if cpu_threads <= 24:
            env_map["OMP_WAIT_POLICY"] = "ACTIVE"
            env_map["KMP_BLOCKTIME"] = "200"
    if extra_env:
        env_map.update({k: str(v) for k, v in extra_env.items()})

    weights_dir = discover_tiberius_weights(model_cfg, explicit_weights=gate.get("tiberius_weights"))
    eff_seq_len = seq_len or _TIBERIUS_DEFAULT_SEQLEN.get(model_cfg)
    tib_script = rt.get("script", "/opt/Tiberius/tiberius.py")
    tib_py = rt.get("python", "python3")
    tib_exec = [tib_py, "-c", _CPU_PAD_EXEC, tib_script] if cpu_only else [tib_py, tib_script]
    tib_args = [
        "--genome", str(genome_fa),
        "--batch_size", str(batch_size),
        "--out", str(gtf),
        "--protseq", str(out_dir / f"tiberius_{tag}.faa"),
    ]
    if weights_dir is not None and rt["backend"] != "docker":
        tib_args += ["--model", str(weights_dir)]
        if model_cfg in _TIBERIUS_UNMASKED_CFGS:
            tib_args.append("--no_softmasking")
    else:
        tib_args += ["--model_cfg", model_cfg]
    if eff_seq_len:
        tib_args += ["--seq_len", str(eff_seq_len)]

    child_env: dict[str, str] | None = None
    if rt["backend"] == "native":
        cmd = [
            "python3", "-c", _RUSAGE_WRAPPER, str(ru_json),
            *tib_exec, *tib_args,
        ]
        child_env = safe_env(env_map)
    elif rt["backend"] == "apptainer":
        cmd = [rt["exe"], "exec", "--cleanenv"]
        if not cpu_only:
            cmd.append("--nv")
        for ek, ev in env_map.items():
            cmd += ["--env", f"{ek}={ev}"]
        binds = [str(out_dir.resolve()), str(genome_fa.resolve().parent), str(tmp.resolve())]
        if weights_dir is not None:
            binds.append(str(weights_dir.resolve()))
        for bp in dict.fromkeys(binds):
            cmd += ["-B", f"{bp}:{bp}"]
        cmd += [rt["sif"], "python3", "-c", _RUSAGE_WRAPPER, str(ru_json), *tib_exec, *tib_args]
    else:
        uid, gid = os.getuid(), os.getgid()
        cmd = ["docker", "run", "--rm"]
        if cpuset_cpus:
            cmd += ["--cpuset-cpus", cpuset_cpus]
        if not cpu_only:
            cmd += ["--gpus", "all"]
        cmd += ["--ipc=host", "-u", f"{uid}:{gid}"]
        for ek, ev in env_map.items():
            cmd += ["-e", f"{ek}={ev}"]
        w_root = (weights_dir.parent if weights_dir is not None else BUILD_ROOT / "weights")
        if w_root.exists():
            cmd += ["-v", f"{w_root}:/opt/Tiberius/model_weights:ro"]
        mounted: list[Path] = []
        for cand_p in (out_dir.resolve(), genome_fa.resolve().parent, tmp.resolve()):
            if any(cand_p == m or str(cand_p).startswith(str(m) + "/") for m in mounted):
                continue
            cmd += ["-v", f"{cand_p}:{cand_p}"]
            mounted.append(cand_p)
        cmd += [TIBERIUS_IMAGE, "python3", "-c", _RUSAGE_WRAPPER, str(ru_json), *tib_exec, *tib_args]

    rec = StageRecord(name=f"tiberius_{tag}")
    t0 = time.time()
    try:
        if cpu_only:
            with open(out_dir / f"tiberius_{tag}.log", "w") as lf:
                rc, wall, u, s = run_cmd(cmd, stdout=lf, stderr=subprocess.STDOUT, env=child_env, check=False)
            gsum = {"peak_vram_mb": 0.0, "gpu_util_seconds": 0.0, "samples": 0}
            rec.gpu_seconds = 0.0
        else:
            with GpuSampler(out_dir / f"tiberius_{tag}.gpu.csv") as gs:
                with open(out_dir / f"tiberius_{tag}.log", "w") as lf:
                    rc, wall, u, s = run_cmd(cmd, stdout=lf, stderr=subprocess.STDOUT, env=child_env, check=False)
            gsum = gs.summary()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec.wall_seconds = time.time() - t0
    inner = json.loads(ru_json.read_text()) if ru_json.exists() else {}
    rec.cpu_user = inner.get("user", 0.0) + u
    rec.cpu_sys = inner.get("sys", 0.0) + s
    if not cpu_only:
        rec.gpu_seconds = inner.get("wall", rec.wall_seconds)
    rec.gpu_util_seconds = gsum["gpu_util_seconds"]
    rec.peak_vram_mb = gsum["peak_vram_mb"]
    rec.extra = {
        "backend": rt["backend"],
        "batch_size": batch_size,
        "cpu_only": cpu_only,
        "seq_len": eff_seq_len,
        "cpu_threads": cpu_threads if cpu_only else None,
        "cpuset_cpus": cpuset_cpus,
        "env": env_map if env_map else None,
        "rc": rc,
        "gpu": gsum,
        "inner_maxrss_kb": inner.get("maxrss_kb"),
        "gtf": str(gtf),
        "input_sha256": input_sha256,
        **(extra_meta or {}),
    }
    if rc != 0:
        log.append(rec)
        raise RuntimeError(f"Tiberius failed rc={rc}; see {out_dir}/tiberius_{tag}.log")
    if not gtf.is_file():
        log.append(rec)
        raise RuntimeError(
            f"Tiberius exited with rc=0 without writing output GTF {gtf}; check Tiberius runtime and {out_dir}/tiberius_{tag}.log"
        )
    rec.extra["n_transcripts"] = len(load_tiberius(gtf))
    log.append(rec)
    done.write_text(json.dumps({**rec.to_dict(), "input_sha256": input_sha256}))
    return rec


# ----------------------------------------------------------------------------------------------
# miniprot
# ----------------------------------------------------------------------------------------------


def _shard_contigs(lengths: dict[str, int], max_shard_bp: int) -> list[list[str]]:
    """Pack contigs into size-bounded shards (largest-first greedy bin packing)."""
    shards: list[list[str]] = []
    shard_bp: list[int] = []
    for ctg, clen in sorted(lengths.items(), key=lambda kv: (-kv[1], kv[0])):
        placed = False
        for i in range(len(shards)):
            if shard_bp[i] + clen <= max_shard_bp:
                shards[i].append(ctg)
                shard_bp[i] += clen
                placed = True
                break
        if not placed:
            shards.append([ctg])
            shard_bp.append(clen)
    return shards


def wrap_taskset_cmd(cmd: list[str], cpus: list[int] | None) -> tuple[list[str], bool]:
    """Prefix cmd with ``taskset -c <cpus>`` when cpus is non-empty and taskset is installed."""
    if not cpus:
        return cmd, False
    if shutil.which("taskset") is None:
        return cmd, True
    cpu_arg = ",".join(str(int(c)) for c in cpus)
    return ["taskset", "-c", cpu_arg, *cmd], False


def run_miniprot(
    genome_fa: Path,
    proteins: Path,
    out_dir: Path,
    log: StageLog,
    threads: int = 24,
    max_shard_bp: int = 700_000_000,
    outs: float | None = None,
    outn: int | None = None,
    cpuset: list[int] | None = None,
) -> StageRecord:
    out_dir = Path(out_dir).resolve()
    genome_fa = Path(genome_fa).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gff = out_dir / "miniprot.gff3"
    done = out_dir / "miniprot.done.json"
    if done.exists() and gff.exists():
        return StageRecord(**{k: v for k, v in json.loads(done.read_text()).items() if k in StageRecord.__dataclass_fields__})
    rec = StageRecord(name="miniprot")
    t0 = time.time()
    g = Genome(genome_fa)
    genome_bp = sum(g.lengths.values())
    shards = _shard_contigs(g.lengths, max_shard_bp) if max_shard_bp > 0 else [list(g.lengths)]
    eff_outs = float(outs) if outs is not None else 0.97
    eff_outn = int(outn) if outn is not None else 1000
    eff_intron = miniprot_max_intron(genome_bp)
    extra_mp = ["--outs", str(eff_outs), "--outn", str(eff_outn)]
    eff_threads = len(cpuset) if cpuset else threads
    taskset_unavail = False
    if not Path(proteins).is_file() or Path(proteins).stat().st_size == 0 or genome_bp == 0:
        gff.write_text("##gff-version 3\n")
        rec.wall_seconds = time.time() - t0
        rec.extra = {"mode": "full", "n_shards": 0, "threads": eff_threads, "gff_bytes": gff.stat().st_size, "n_alignments": 0}
        log.append(rec)
        done.write_text(json.dumps(rec.to_dict()))
        return rec
    if (max_shard_bp <= 0) or (genome_bp <= max_shard_bp) or (len(shards) <= 1):
        mpi = out_dir / "genome.mpi"
        cmd1, unav1 = wrap_taskset_cmd(["miniprot", "-t", str(eff_threads), "-d", str(mpi), str(genome_fa)], cpuset)
        cmd2, unav2 = wrap_taskset_cmd(
            ["miniprot", "-t", str(eff_threads), "-G", str(eff_intron), *extra_mp, "--gff-only", str(mpi), str(proteins)],
            cpuset,
        )
        taskset_unavail = unav1 or unav2
        with open(out_dir / "miniprot_index.log", "w") as lf:
            _, _, u1, s1 = run_cmd(cmd1, stderr=lf)
        with open(gff, "w") as fo, open(out_dir / "miniprot.log", "w") as lf:
            _, _, u2, s2 = run_cmd(cmd2, stdout=fo, stderr=lf)
        mpi.unlink(missing_ok=True)
        rec.cpu_user, rec.cpu_sys = u1 + u2, s1 + s2
        rec.extra = {
            "mode": "full",
            "n_shards": 1,
            "threads": eff_threads,
            "outs": eff_outs,
            "outn": eff_outn,
            "eff_max_intron": eff_intron,
            "gff_bytes": gff.stat().st_size,
        }
        if taskset_unavail:
            rec.extra["taskset_unavailable"] = True
    else:
        prepared_shards: list[tuple[int, Path, Path, Path, dict[str, tuple[str, int, int]]]] = []
        for si, ctgs in enumerate(shards):
            s_fa = out_dir / f"shard_{si}.fna"
            s_mpi = out_dir / f"shard_{si}.mpi"
            s_gff = out_dir / f"shard_{si}.gff3"
            wmap: dict[str, tuple[str, int, int]] = {}
            with open(s_fa, "w") as fh:
                for c in ctgs:
                    clen = g.lengths[c]
                    wmap[c] = (c, 0, clen)
                    seq = g.fetch(c, 0, clen)
                    fh.write(f">{c}\n")
                    for pos in range(0, len(seq), 80):
                        fh.write(seq[pos : pos + 80] + "\n")
            prepared_shards.append((si, s_fa, s_mpi, s_gff, wmap))

        shard_results: dict[int, tuple[float, float, bool]] = {}

        def _exec_shard(item: tuple[int, Path, Path, Path, dict], s_cpus: list[int] | None, s_threads: int) -> None:
            si, s_fa, s_mpi, s_gff, _wmap = item
            c1, un1 = wrap_taskset_cmd(["miniprot", "-t", str(s_threads), "-d", str(s_mpi), str(s_fa)], s_cpus)
            c2, un2 = wrap_taskset_cmd(
                ["miniprot", "-t", str(s_threads), "-G", str(eff_intron), *extra_mp, "--gff-only", str(s_mpi), str(proteins)],
                s_cpus,
            )
            with open(out_dir / f"miniprot_index_s{si}.log", "w") as lf:
                _, _, u1, s1 = run_cmd(c1, stderr=lf)
            with open(s_gff, "w") as fo, open(out_dir / f"miniprot_s{si}.log", "w") as lf:
                _, _, u2, s2 = run_cmd(c2, stdout=fo, stderr=lf)
            s_mpi.unlink(missing_ok=True)
            s_fa.unlink(missing_ok=True)
            shard_results[si] = (u1 + u2, s1 + s2, un1 or un2)

        concurrent_shards = 1
        if cpuset and len(cpuset) >= 32 and len(prepared_shards) >= 2:
            concurrent_shards = 2
            half = len(cpuset) // 2
            halves = [cpuset[:half], cpuset[half : 2 * half]]
            import queue as _qmod

            sq: _qmod.Queue[int] = _qmod.Queue()
            sq.put(0)
            sq.put(1)

            def _run_slot(it):
                sl = sq.get()
                try:
                    _exec_shard(it, halves[sl], len(halves[sl]))
                finally:
                    sq.put(sl)

            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(_run_slot, prepared_shards))
        else:
            for it in prepared_shards:
                _exec_shard(it, cpuset, eff_threads)

        bucket_gffs = [(s_gff, wmap) for _si, _s_fa, _s_mpi, s_gff, wmap in prepared_shards]
        u_tot = sum(v[0] for v in shard_results.values())
        s_tot = sum(v[1] for v in shard_results.values())
        taskset_unavail = any(v[2] for v in shard_results.values())
        n_kept = lift_miniprot_gff3(bucket_gffs, gff, sec_ratio=eff_outs, max_rank=eff_outn)
        for s_gff, _ in bucket_gffs:
            s_gff.unlink(missing_ok=True)
        rec.cpu_user, rec.cpu_sys = u_tot, s_tot
        rec.extra = {
            "mode": "sharded",
            "n_shards": len(shards),
            "concurrent_shards": concurrent_shards,
            "max_shard_bp": max_shard_bp,
            "outs": eff_outs,
            "outn": eff_outn,
            "eff_max_intron": eff_intron,
            "n_kept_alignments": n_kept,
            "gff_bytes": gff.stat().st_size,
        }
        if taskset_unavail:
            rec.extra["taskset_unavailable"] = True
    rec.wall_seconds = time.time() - t0
    log.append(rec)
    done.write_text(json.dumps(rec.to_dict()))
    return rec


def miniprot_max_intron(genome_bp: int) -> int:
    """Match ``miniprot -I`` max intron length (3.6 * sqrt(refLen), clamped to [10k, 200k]) on full genome_bp."""
    return min(200_000, max(10_000, int(round(3.6 * math.sqrt(max(1, genome_bp))))))


def _read_fasta_dict(faa: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur_id = ""
    buf: list[str] = []
    with open(faa) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(buf)
                cur_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur_id:
        seqs[cur_id] = "".join(buf)
    return seqs


def _select_blastp_hits_tiled(
    rows: list[tuple[str, float, int, int]],
    top_global: int = 8,
    top_per_domain: int = 6,
    min_novel_aa: int = 25,
) -> set[str]:
    """Select top global hits plus domain-tiling hits so fused multi-domain Tiberius queries keep all partner homologs."""
    if not rows:
        return set()
    rows_sorted = sorted(rows, key=lambda r: -r[1])
    chosen: set[str] = set()
    max_qend = max(r[3] for r in rows_sorted)
    cov = [0] * (max_qend + 2)

    def _mark(qs: int, qe: int) -> None:
        a, b = max(1, min(qs, qe)), min(max_qend, max(qs, qe))
        for p in range(a, b + 1):
            cov[p] += 1

    for sseqid, _bits, qs, qe in rows_sorted[:top_global]:
        chosen.add(sseqid)
        _mark(qs, qe)
    for sseqid, _bits, qs, qe in rows_sorted[top_global:]:
        if sseqid in chosen:
            continue
        a, b = max(1, min(qs, qe)), min(max_qend, max(qs, qe))
        under = sum(1 for p in range(a, b + 1) if cov[p] < top_per_domain)
        if under >= min_novel_aa:
            chosen.add(sseqid)
            _mark(qs, qe)
    return chosen


def _write_cds_masked_intergenic_segments(
    genome: Genome,
    tiberius: list[Transcript],
    out_fna: Path,
    chunk_bp: int = 50_000,
    overlap_bp: int = 600,
    min_seg_bp: int = 90,
) -> tuple[dict[str, tuple[str, int]], int]:
    """Mask Tiberius CDS intervals with 'N' and emit non-N genomic chunks for fast ``diamond blastx``."""
    cds_by_ctg: dict[str, list[tuple[int, int]]] = {}
    for t in tiberius:
        if t.cds:
            cds_by_ctg.setdefault(t.contig, []).extend(t.cds)
    seg_map: dict[str, tuple[str, int]] = {}
    total_bp = 0
    seg_idx = 0
    n_run_re = re.compile(r"N{200,}", re.I)
    step = max(1000, chunk_bp - overlap_bp)
    with open(out_fna, "w") as out:
        for ctg, ctg_len in genome.lengths.items():
            raw = genome.fetch(ctg, 0, ctg_len)
            intervals = merge_intervals(cds_by_ctg.get(ctg, []))
            if intervals:
                arr = bytearray(raw.encode("ascii"))
                for a, b in intervals:
                    a0, b0 = max(0, a), min(ctg_len, b)
                    if a0 < b0:
                        arr[a0:b0] = b"N" * (b0 - a0)
                masked = arr.decode("ascii")
            else:
                masked = raw
            pos = 0
            for m in n_run_re.finditer(masked):
                if m.start() - pos >= min_seg_bp:
                    sub = masked[pos : m.start()]
                    for off in range(0, len(sub), step):
                        piece = sub[off : off + chunk_bp]
                        if len(piece) - piece.upper().count("N") < min_seg_bp:
                            continue
                        sid = f"S{seg_idx}"
                        seg_idx += 1
                        seg_map[sid] = (ctg, pos + off)
                        out.write(f">{sid}\n{piece}\n")
                        total_bp += len(piece)
                        if off + chunk_bp >= len(sub):
                            break
                pos = m.end()
            if len(masked) - pos >= min_seg_bp:
                sub = masked[pos:]
                for off in range(0, len(sub), step):
                    piece = sub[off : off + chunk_bp]
                    if len(piece) - piece.upper().count("N") < min_seg_bp:
                        continue
                    sid = f"S{seg_idx}"
                    seg_idx += 1
                    seg_map[sid] = (ctg, pos + off)
                    out.write(f">{sid}\n{piece}\n")
                    total_bp += len(piece)
                    if off + chunk_bp >= len(sub):
                        break
    return seg_map, total_bp


def _expand_cluster_members(
    reps: set[str],
    clusters: dict[str, list[str]],
    prot_seqs: dict[str, str],
    max_extra: int = 2,
) -> list[str]:
    """Include centroid plus up to ``max_extra`` length-diverse cluster members to preserve >=0.80 coverage."""
    selected: dict[str, None] = {}
    for rep in sorted(reps):
        if rep in prot_seqs:
            selected[rep] = None
        members = [m for m in clusters.get(rep, ()) if m != rep and m in prot_seqs]
        if not members or max_extra <= 0:
            continue
        rep_len = max(1, len(prot_seqs.get(rep, "")))
        members.sort(key=lambda m: (len(prot_seqs[m]), m))
        # Prefer a slightly shorter conserved ortholog (>=0.80 * rep_len) and the longest full-length member
        ge80 = [m for m in members if len(prot_seqs[m]) >= 0.8 * rep_len]
        picks: list[str] = []
        if ge80:
            picks.append(ge80[0])
        if members[-1] not in picks:
            picks.append(members[-1])
        for m in picks[:max_extra]:
            selected[m] = None
    return list(selected)


def _rewrite_attr(attr_str: str, new_id: str, new_rank: int, is_mrna: bool) -> str:
    parts = []
    for kv in attr_str.split(";"):
        if "=" not in kv:
            if kv:
                parts.append(kv)
            continue
        k, v = kv.split("=", 1)
        if is_mrna and k == "ID":
            parts.append(f"ID={new_id}")
        elif not is_mrna and k == "Parent":
            parts.append(f"Parent={new_id}")
        elif k == "Rank":
            parts.append(f"Rank={new_rank}")
        else:
            parts.append(kv)
    return ";".join(parts)


def lift_miniprot_gff3(
    bucket_gffs: list[tuple[Path, dict[str, tuple[str, int, int]]]],
    out_gff: Path,
    sec_ratio: float = 0.7,
    max_rank: int = 30,
) -> int:
    """Two-pass byte-offset lift of window-local miniprot GFF3 records back to parent contig coordinates."""
    by_tid: dict[str, list[tuple[float, str, int, int, int, int]]] = {}
    seen_exact: set[bytes] = set()

    def _index_block(
        b_idx: int,
        byte_start: int,
        byte_end: int,
        cur_mrna: list[str] | None,
        cds_local: list[tuple[int, int]],
        has_children: bool,
        wmap: dict[str, tuple[str, int, int]],
    ) -> None:
        if cur_mrna is None or not has_children or not cds_local:
            return
        wid = cur_mrna[0]
        if wid not in wmap:
            return
        ctg, w_off, _ = wmap[wid]
        m_s = int(cur_mrna[3]) + w_off
        score = float(cur_mrna[5]) if cur_mrna[5] != "." else 0.0
        strand = cur_mrna[6]
        tid = ""
        for kv in cur_mrna[8].split(";"):
            if kv.startswith("Target="):
                tparts = kv[7:].split()
                if tparts:
                    tid = tparts[0]
                break
        if not tid:
            return
        chain = merge_intervals((cs + w_off - 1, ce + w_off) for cs, ce in cds_local)
        if not chain:
            return
        h = hashlib.blake2b(digest_size=12)
        h.update(f"{ctg}\0{strand}\0{tid}\0".encode("utf-8"))
        for a, b in chain:
            h.update(f"{a}:{b},".encode("ascii"))
        dkey = h.digest()
        if dkey in seen_exact:
            return
        seen_exact.add(dkey)
        by_tid.setdefault(tid, []).append((score, ctg, m_s, b_idx, byte_start, byte_end - byte_start))

    for b_idx, (gff_p, wmap) in enumerate(bucket_gffs):
        if not gff_p.exists():
            continue
        cur_m: list[str] | None = None
        cds_loc: list[tuple[int, int]] = []
        has_ch = False
        blk_off = 0
        with open(gff_p, "rb") as fh:
            while True:
                line_off = fh.tell()
                raw = fh.readline()
                if not raw:
                    end_off = fh.tell()
                    _index_block(b_idx, blk_off, end_off, cur_m, cds_loc, has_ch, wmap)
                    break
                if raw.startswith(b"#") or not raw.strip():
                    continue
                line = raw.decode("utf-8", errors="replace")
                f = line.rstrip("\r\n").split("\t")
                if len(f) < 9:
                    continue
                if f[2] == "mRNA":
                    _index_block(b_idx, blk_off, line_off, cur_m, cds_loc, has_ch, wmap)
                    cur_m = f
                    cds_loc = []
                    has_ch = False
                    blk_off = line_off
                elif f[2] in ("CDS", "stop_codon", "exon"):
                    has_ch = True
                    if f[2] in ("CDS", "stop_codon"):
                        cds_loc.append((int(f[3]), int(f[4])))

    seen_exact.clear()
    kept: list[tuple[str, int, float, str, int, int, int, int]] = []
    for tid, blks in by_tid.items():
        blks.sort(key=lambda b: (-b[0], b[1], b[2]))
        best = blks[0][0]
        rank = 0
        for score, ctg, m_s, b_idx, byte_off, byte_len in blks:
            if best > 0 and score < sec_ratio * best:
                continue
            rank += 1
            if rank > max_rank:
                break
            kept.append((ctg, m_s, -score, tid, rank, b_idx, byte_off, byte_len))
    by_tid.clear()

    kept.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    open_handles: dict[int, Any] = {}
    try:
        with open(out_gff, "w") as out:
            out.write("##gff-version 3\n")
            for idx, (_ctg, _ms, _neg_sc, _tid, rank, b_idx, byte_off, byte_len) in enumerate(kept, 1):
                fh_b = open_handles.get(b_idx)
                if fh_b is None:
                    fh_b = open(bucket_gffs[b_idx][0], "rb")
                    open_handles[b_idx] = fh_b
                wmap = bucket_gffs[b_idx][1]
                fh_b.seek(byte_off)
                chunk_txt = fh_b.read(byte_len).decode("utf-8", errors="replace")
                mp_id = f"MP{idx:06d}"
                for raw_ln in chunk_txt.splitlines():
                    if not raw_ln or raw_ln[0] == "#":
                        continue
                    f = raw_ln.split("\t")
                    if len(f) < 9:
                        continue
                    ft = f[2]
                    if ft not in ("mRNA", "CDS", "stop_codon", "exon"):
                        continue
                    ctg_real, w_off, _ = wmap[f[0]]
                    f[0] = ctg_real
                    f[3] = str(int(f[3]) + w_off)
                    f[4] = str(int(f[4]) + w_off)
                    f[8] = _rewrite_attr(f[8], mp_id, rank, is_mrna=(ft == "mRNA"))
                    out.write("\t".join(f) + "\n")
    finally:
        for fh_b in open_handles.values():
            try:
                fh_b.close()
            except OSError:
                pass
    return len(kept)


def run_targeted_miniprot(
    genome_fa: Path,
    proteins: Path,
    tiberius_gtf: Path,
    out_dir: Path,
    log: StageLog,
    threads: int = 24,
    pad_locus: int | None = None,
    pad_intergenic: int | None = None,
    merge_gap: int = 15_000,
    linclust_id: float = 90.0,
    linclust_cov: float = 80.0,
    min_linclust_proteins: int = 200,
    max_buckets: int = 16,
    target_bucket_bp: int = 25_000_000,
    intergenic_blastx: bool = True,
    cluster_expand: int = 2,
) -> StageRecord:
    """Two-stage homology seed (DIAMOND linclust + blastp/blastx routing -> windowed miniprot + lift-back)."""
    out_dir = Path(out_dir).resolve()
    genome_fa = Path(genome_fa).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gff = out_dir / "miniprot.gff3"
    done = out_dir / "miniprot.done.json"
    if done.exists() and gff.exists():
        return StageRecord(**{k: v for k, v in json.loads(done.read_text()).items() if k in StageRecord.__dataclass_fields__})

    t0 = time.time()
    rec = StageRecord(name="miniprot")
    cpu_u = cpu_s = 0.0
    sub_wall: dict[str, float] = {}
    sub_cpu: dict[str, float] = {}

    def _acct(step: str, w: float, u: float, s: float) -> None:
        nonlocal cpu_u, cpu_s
        cpu_u += u
        cpu_s += s
        sub_wall[step] = sub_wall.get(step, 0.0) + w
        sub_cpu[step] = sub_cpu.get(step, 0.0) + u + s

    genome = Genome(genome_fa)
    genome_bp = sum(genome.lengths.values())
    g_max = miniprot_max_intron(genome_bp)
    if pad_locus is None:
        pad_locus = 20_000 if genome_bp < 1_500_000_000 else 35_000
    if pad_intergenic is None:
        pad_intergenic = 35_000 if genome_bp < 1_500_000_000 else 50_000

    prot_seqs = _read_fasta_dict(proteins)
    tiberius = load_tiberius(tiberius_gtf) if tiberius_gtf.exists() else []
    if not prot_seqs or genome_bp == 0:
        gff.write_text("##gff-version 3\n")
        rec.wall_seconds = time.time() - t0
        rec.extra = {"mode": "targeted", "gff_bytes": gff.stat().st_size, "n_alignments": 0}
        log.append(rec)
        done.write_text(json.dumps(rec.to_dict()))
        return rec

    work = out_dir / "_targeted_seed_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    dtmp = work / "dtmp"
    dtmp.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Collapse clade protein redundancy via diamond linclust
        clusters: dict[str, list[str]] = {}
        if len(prot_seqs) >= min_linclust_proteins:
            reps_faa = work / "proteins_nr.faa"
            clust_tsv = work / "linclust.tsv"
            _, w, u, s = run_cmd([
                "diamond", "linclust",
                "--db", str(proteins),
                "--out", str(clust_tsv),
                "--reps", str(reps_faa),
                "--approx-id", str(linclust_id),
                "--member-cover", str(linclust_cov),
                "--tmpdir", str(dtmp),
                "--threads", str(threads),
                "--ignore-warnings",
                "--quiet",
            ])
            _acct("diamond_linclust", w, u, s)
            with open(clust_tsv) as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        clusters.setdefault(parts[0], []).append(parts[1])
            n_proteins_nr = len(clusters)
        else:
            reps_faa = proteins
            clusters = {pid: [pid] for pid in prot_seqs}
            n_proteins_nr = len(prot_seqs)

        # 2. Build DIAMOND DB on non-redundant clade centroids
        dmnd = work / "clade_nr.dmnd"
        _, w, u, s = run_cmd([
            "diamond", "makedb",
            "--in", str(reps_faa),
            "-d", str(dmnd),
            "--threads", str(threads),
            "--ignore-warnings",
            "--quiet",
        ])
        _acct("diamond_makedb", w, u, s)

        # 3. Query Tiberius predicted proteins against clade_nr.dmnd (domain-tiled selection)
        raw_intervals: dict[str, list[tuple[int, int, set[str]]]] = {}
        tib_routed_reps: set[str] = set()
        if tiberius:
            tib_faa = work / "tiberius_queries.faa"
            valid_tib: list[Transcript] = []
            with open(tib_faa, "w") as fh:
                for t in tiberius:
                    ph = int(getattr(t, "cds_phase", 0) or 0) % 3
                    aa = translate(genome.spliced(t.contig, t.strand, t.cds)[ph:]).rstrip("*").replace("*", "X")
                    if len(aa) >= 15:
                        qi = len(valid_tib)
                        valid_tib.append(t)
                        fh.write(f">Q{qi}\n{aa}\n")
            if valid_tib:
                bp_tsv = work / "tib_blastp.tsv"
                _, w, u, s = run_cmd([
                    "diamond", "blastp",
                    "-q", str(tib_faa),
                    "-d", str(dmnd),
                    "-o", str(bp_tsv),
                    "--sensitive",
                    "--max-target-seqs", "25",
                    "--evalue", "1e-2",
                    "--outfmt", "6", "qseqid", "sseqid", "bitscore", "qstart", "qend",
                    "-p", str(threads),
                    "--tmpdir", str(dtmp),
                    "--ignore-warnings",
                    "--quiet",
                ])
                _acct("diamond_blastp", w, u, s)
                hits_by_q: dict[int, list[tuple[str, float, int, int]]] = {}
                with open(bp_tsv) as fh:
                    for line in fh:
                        f = line.rstrip("\n").split("\t")
                        if len(f) >= 5:
                            qi = int(f[0][1:])
                            hits_by_q.setdefault(qi, []).append((f[1], float(f[2]), int(f[3]), int(f[4])))
                for qi, t in enumerate(valid_tib):
                    reps_q = _select_blastp_hits_tiled(hits_by_q.get(qi, []))
                    if reps_q:
                        tib_routed_reps.update(reps_q)
                        ctg_len = genome.lengths[t.contig]
                        lo = max(0, t.cds_start - pad_locus)
                        hi = min(ctg_len, t.cds_end + pad_locus)
                        raw_intervals.setdefault(t.contig, []).append((lo, hi, reps_q))

        # 4. Coarse intergenic / unannotated scan on Tiberius-CDS-masked genome via diamond blastx
        bx_routed_reps: set[str] = set()
        if intergenic_blastx:
            ig_fna = work / "intergenic_segments.fna"
            t_mask0 = time.time()
            seg_map, seg_bp = _write_cds_masked_intergenic_segments(genome, tiberius, ig_fna)
            sub_wall["mask_intergenic"] = time.time() - t_mask0
            if seg_map and seg_bp > 0:
                bx_tsv = work / "intergenic_blastx.tsv"
                bx_mode = "--sensitive" if seg_bp < 10_000_000 else "--fast"
                bx_max_tgt = "20" if not tiberius else "6"
                _, w, u, s = run_cmd([
                    "diamond", "blastx",
                    "-q", str(ig_fna),
                    "-d", str(dmnd),
                    "-o", str(bx_tsv),
                    bx_mode,
                    "--max-target-seqs", bx_max_tgt,
                    "--evalue", "1e-4",
                    "--outfmt", "6", "qseqid", "qstart", "qend", "sseqid", "pident", "bitscore",
                    "-p", str(threads),
                    "--tmpdir", str(dtmp),
                    "--ignore-warnings",
                    "--quiet",
                ])
                _acct("diamond_blastx", w, u, s)
                # Group intergenic hits into 5-kb bins (cap 12 when tiberius=[], else 6)
                bin_cap = 12 if not tiberius else 6
                span_top = 16 if not tiberius else 8
                bin_hits: dict[tuple[str, int], list[tuple[float, int, int, str]]] = {}
                with open(bx_tsv) as fh:
                    for line in fh:
                        f = line.rstrip("\n").split("\t")
                        if len(f) < 6 or f[0] not in seg_map:
                            continue
                        ctg, seg_off = seg_map[f[0]]
                        qs, qe = int(f[1]), int(f[2])
                        h_lo = seg_off + min(qs, qe) - 1
                        h_hi = seg_off + max(qs, qe)
                        bits = float(f[5])
                        bkey = (ctg, h_lo // 5_000)
                        bin_hits.setdefault(bkey, []).append((bits, h_lo, h_hi, f[3]))
                for (ctg, _bin), hlist in bin_hits.items():
                    hlist.sort(key=lambda x: -x[0])
                    reps_b: set[str] = set()
                    lo_b = min(x[1] for x in hlist[:span_top])
                    hi_b = max(x[2] for x in hlist[:span_top])
                    for _bits, _l, _h, sseqid in hlist:
                        reps_b.add(sseqid)
                        if len(reps_b) >= bin_cap:
                            break
                    bx_routed_reps.update(reps_b)
                    ctg_len = genome.lengths[ctg]
                    raw_intervals.setdefault(ctg, []).append(
                        (max(0, lo_b - pad_intergenic), min(ctg_len, hi_b + pad_intergenic), reps_b)
                    )

        # 5. Merge overlapping/nearby windows per contig + bridge-merge intervals sharing homologs within G_max
        merged_windows: list[tuple[str, int, int, set[str]]] = []
        for ctg in genome.lengths:
            ivs = raw_intervals.get(ctg, [])
            if not ivs:
                continue
            ivs.sort(key=lambda x: (x[0], x[1]))
            step1: list[list] = []
            for lo, hi, rset in ivs:
                if step1 and lo <= step1[-1][1] + merge_gap:
                    step1[-1][1] = max(step1[-1][1], hi)
                    step1[-1][2].update(rset)
                else:
                    step1.append([lo, hi, set(rset)])
            # Bridge adjacent windows within g_max if they share >=1 routed centroid (prevents splitting missed exons)
            step2: list[list] = []
            for lo, hi, rset in step1:
                if (
                    step2
                    and lo - step2[-1][1] <= g_max
                    and hi - step2[-1][0] <= 1_000_000
                    and (step2[-1][2] & rset)
                ):
                    step2[-1][1] = max(step2[-1][1], hi)
                    step2[-1][2].update(rset)
                else:
                    step2.append([lo, hi, rset])
            for lo, hi, rset in step2:
                if rset and hi > lo:
                    merged_windows.append((ctg, lo, hi, rset))

        total_window_bp = sum(hi - lo for _c, lo, hi, _r in merged_windows)
        if not merged_windows or total_window_bp == 0:
            gff.write_text("##gff-version 3\n")
            rec.wall_seconds = time.time() - t0
            rec.cpu_user, rec.cpu_sys = cpu_u, cpu_s
            rec.extra = {"mode": "targeted", "gff_bytes": gff.stat().st_size, "n_alignments": 0, "window_bp": 0}
            log.append(rec)
            done.write_text(json.dumps(rec.to_dict()))
            return rec

        # 6. Contig-ordered balanced bucketing + windowed miniprot + exact coordinate lift-back
        n_buckets = max(1, min(max_buckets, len(merged_windows), max(1, round(total_window_bp / target_bucket_bp))))
        bucket_target = math.ceil(total_window_bp / n_buckets)
        buckets: list[list[tuple[str, int, int, set[str]]]] = [[]]
        cur_bp = 0
        for w_item in merged_windows:
            wbp = w_item[2] - w_item[1]
            if cur_bp > 0 and cur_bp + wbp > bucket_target and len(buckets) < n_buckets:
                buckets.append([])
                cur_bp = 0
            buckets[-1].append(w_item)
            cur_bp += wbp

        bucket_outputs: list[tuple[Path, dict[str, tuple[str, int, int]]]] = []
        all_expanded_prots: set[str] = set()
        win_counter = 0
        for bi, b_wins in enumerate(buckets):
            if not b_wins:
                continue
            b_fna = work / f"bucket_{bi:02d}.fna"
            b_mpi = work / f"bucket_{bi:02d}.mpi"
            b_faa = work / f"bucket_{bi:02d}.faa"
            b_gff = work / f"bucket_{bi:02d}.gff3"
            wmap: dict[str, tuple[str, int, int]] = {}
            b_reps: set[str] = set()
            with open(b_fna, "w") as fh:
                for ctg, lo, hi, rset in b_wins:
                    win_counter += 1
                    wid = f"W{win_counter:06d}"
                    wmap[wid] = (ctg, lo, hi)
                    b_reps.update(rset)
                    fh.write(f">{wid}\n{genome.fetch(ctg, lo, hi)}\n")
            b_prots = _expand_cluster_members(b_reps, clusters, prot_seqs, max_extra=cluster_expand)
            if not b_prots:
                continue
            all_expanded_prots.update(b_prots)
            with open(b_faa, "w") as fh:
                for pid in b_prots:
                    fh.write(f">{pid}\n{prot_seqs[pid]}\n")
            with open(out_dir / "miniprot_index.log", "a") as lf:
                _, w1, u1, s1 = run_cmd(["miniprot", "-t", str(threads), "-d", str(b_mpi), str(b_fna)], stderr=lf)
            _acct("miniprot_index", w1, u1, s1)
            with open(b_gff, "w") as fo, open(out_dir / "miniprot.log", "a") as lf:
                _, w2, u2, s2 = run_cmd(
                    ["miniprot", "-t", str(threads), "-G", str(g_max), "--outs", "0.7", "--outn", "30", "--gff-only", str(b_mpi), str(b_faa)],
                    stdout=fo,
                    stderr=lf,
                )
            _acct("miniprot_align", w2, u2, s2)
            b_mpi.unlink(missing_ok=True)
            b_fna.unlink(missing_ok=True)
            b_faa.unlink(missing_ok=True)
            bucket_outputs.append((b_gff, wmap))

        t_lift0 = time.time()
        n_aln = lift_miniprot_gff3(bucket_outputs, gff)
        sub_wall["lift_gff3"] = time.time() - t_lift0
    finally:
        shutil.rmtree(work, ignore_errors=True)

    rec.wall_seconds = time.time() - t0
    rec.cpu_user, rec.cpu_sys = cpu_u, cpu_s
    rec.extra = {
        "mode": "targeted",
        "gff_bytes": gff.stat().st_size,
        "n_alignments": n_aln,
        "n_proteins_in": len(prot_seqs),
        "n_proteins_nr": n_proteins_nr,
        "n_reps_blastp": len(tib_routed_reps),
        "n_reps_blastx": len(bx_routed_reps),
        "n_proteins_routed_unique": len(all_expanded_prots),
        "genome_bp": genome_bp,
        "window_bp": total_window_bp,
        "window_frac": round(total_window_bp / max(1, genome_bp), 4),
        "n_windows": len(merged_windows),
        "n_buckets": len(bucket_outputs),
        "max_intron_g": g_max,
        "wall_by_substep_s": {k: round(v, 3) for k, v in sub_wall.items()},
        "cpu_by_substep_s": {k: round(v, 3) for k, v in sub_cpu.items()},
    }
    log.append(rec)
    done.write_text(json.dumps(rec.to_dict()))
    return rec


@dataclass
class ProteinHit:
    contig: str
    strand: str
    cds: tuple[tuple[int, int], ...]
    score: float
    identity: float
    positive: float
    coverage: float
    target: str
    frameshift: int = 0
    stop_codon: int = 0
    n_proteins: int = 1
    n_rank1: int = 0
    targets: list[str] = field(default_factory=list)
    n_query_ids: int = 0
    n_disabled_query_ids: int = 0
    n_intact_query_ids: int = 0
    max_positive: float = 0.0

    @property
    def start(self) -> int:
        return self.cds[0][0]

    @property
    def end(self) -> int:
        return self.cds[-1][1]

    @property
    def introns(self) -> tuple[tuple[int, int], ...]:
        return introns_of(list(self.cds))

    @property
    def quality(self) -> float:
        return self.positive * min(1.0, self.coverage)


@dataclass(frozen=True)
class RawMiniprotAlignment:
    contig: str
    strand: str
    cds: tuple[tuple[int, int], ...]
    tid: str
    score: float
    identity: float
    positive: float
    coverage: float
    fs: int
    sc: int

    @property
    def start(self) -> int:
        return self.cds[0][0]

    @property
    def end(self) -> int:
        return self.cds[-1][1]

    @property
    def quality(self) -> float:
        return self.positive * min(1.0, self.coverage)

    @property
    def cds_len(self) -> int:
        return sum(b - a for a, b in self.cds)


def _iter_miniprot_records(
    gff: Path,
    prot_len: dict[str, int],
    min_positive: float = 0.4,
    min_cov: float = 0.3,
):
    """Yield (ctg, strand, chain, score, ident, pos, cov, tid, rank, fs, sc) for qualifying miniprot alignments."""
    cur = None
    cur_cds: list[tuple[int, int]] = []

    def _parse_one():
        if cur is None or not cur_cds:
            return None
        ctg, strand, attrs, score = cur
        tgt = attrs.get("Target", "").split()
        if not tgt:
            return None
        tid = tgt[0]
        ts, te = int(tgt[1]), int(tgt[2])
        plen = prot_len.get(tid, 0)
        cov = (te - ts + 1) / plen if plen else 0.0
        pos = float(attrs.get("Positive", 0))
        ident = float(attrs.get("Identity", 0))
        if pos < min_positive or cov < min_cov:
            return None
        chain = tuple(merge_intervals(cur_cds))
        rank = int(attrs.get("Rank", 1))
        fs = int(attrs.get("Frameshift", 0))
        sc = int(attrs.get("StopCodon", 0))
        return ctg, strand, chain, score, ident, pos, cov, tid, rank, fs, sc

    with open(gff) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] == "mRNA":
                rec = _parse_one()
                if rec is not None:
                    yield rec
                attrs = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
                cur = (f[0], f[6], attrs, float(f[5]) if f[5] != "." else 0.0)
                cur_cds = []
            elif f[2] in ("CDS", "stop_codon"):
                cur_cds.append((int(f[3]) - 1, int(f[4])))
    rec = _parse_one()
    if rec is not None:
        yield rec


def read_miniprot(gff: Path, prot_len: dict[str, int], min_positive: float = 0.4, min_cov: float = 0.3) -> list[ProteinHit]:
    """Stream miniprot GFF3; collapse identical CDS chains; keep best attributes and support counts."""
    hits: dict[tuple, ProteinHit] = {}
    q_all: dict[tuple, set[str]] = {}
    q_dis: dict[tuple, set[str]] = {}
    q_int: dict[tuple, set[str]] = {}
    for ctg, strand, chain, score, ident, pos, cov, tid, rank, fs, sc in _iter_miniprot_records(
        gff, prot_len, min_positive=min_positive, min_cov=min_cov
    ):
        key = (ctg, strand, chain)
        h = hits.get(key)
        if h is None:
            hits[key] = ProteinHit(
                ctg, strand, chain, score, ident, pos, cov, tid, fs, sc, 1, int(rank == 1), [tid], max_positive=pos
            )
            q_all[key] = set()
            q_dis[key] = set()
            q_int[key] = set()
        else:
            h.n_proteins += 1
            h.n_rank1 += int(rank == 1)
            if pos > h.max_positive:
                h.max_positive = pos
            if len(h.targets) < 8:
                h.targets.append(tid)
            if pos * min(1.0, cov) > h.positive * min(1.0, h.coverage):
                h.score, h.identity, h.positive, h.coverage, h.target = score, ident, pos, cov, tid
                h.frameshift, h.stop_codon = fs, sc
        q_all[key].add(tid)
        if fs > 0 or sc > 0:
            q_dis[key].add(tid)
        else:
            q_int[key].add(tid)
    for key, h in hits.items():
        h.n_query_ids = len(q_all[key])
        h.n_disabled_query_ids = len(q_dis[key])
        h.n_intact_query_ids = len(q_int[key])
    return list(hits.values())


def read_miniprot_raw_alignments(
    gff: Path,
    prot_len: dict[str, int],
    min_positive: float = 0.4,
    min_cov: float = 0.3,
) -> list[RawMiniprotAlignment]:
    """Read uncollapsed qualifying miniprot alignments in memory without altering seed partition files."""
    return [
        RawMiniprotAlignment(ctg, strand, chain, tid, score, ident, pos, cov, fs, sc)
        for ctg, strand, chain, score, ident, pos, cov, tid, _rank, fs, sc in _iter_miniprot_records(
            gff, prot_len, min_positive=min_positive, min_cov=min_cov
        )
    ]


def scan_readthrough_protein_ids(fasta_path: Path, cache_json: Path | None = None) -> set[str]:
    """Return protein IDs whose FASTA header contains 'readthrough' (case-insensitive), optionally cached."""
    p = Path(fasta_path)
    if not p.exists():
        return set()
    st = p.stat()
    fp = {"source_path": str(p.resolve()), "source_mtime_ns": st.st_mtime_ns, "source_size": st.st_size}
    if cache_json is not None and cache_json.exists():
        try:
            cached = json.loads(cache_json.read_text())
            if all(cached.get(k) == v for k, v in fp.items()):
                return set(cached.get("ids", ()))
        except (OSError, ValueError):
            pass
    ids: set[str] = set()
    with _open_maybe_gz(p) as fh:
        for line in fh:
            if line.startswith(">") and "readthrough" in line.lower():
                acc = line[1:].split()[0]
                if acc.startswith("ref|"):
                    acc = acc.split("|")[1]
                ids.add(acc)
    if cache_json is not None:
        try:
            cache_json.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_json.with_name(f".{cache_json.name}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps({**fp, "ids": sorted(ids)}, indent=1))
            os.replace(tmp, cache_json)
        except OSError:
            pass
    return ids


# ----------------------------------------------------------------------------------------------
# LOCKED_HIGH_CONF / UNSETTLED_LOCI partition
# ----------------------------------------------------------------------------------------------

LOCKED = "LOCKED_HIGH_CONF"
UNSETTLED = "UNSETTLED_LOCI"


@dataclass
class Locus:
    locus_id: str
    contig: str
    strand: str
    start: int
    end: int
    status: str
    category: str
    tiberius: list[Transcript] = field(default_factory=list)
    hits: list[ProteinHit] = field(default_factory=list)
    locked_model: Transcript | None = None


def _ovl(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _cds_block_components(items: list, get_key, get_cds) -> list[list]:
    """Group items by ``get_key(item)`` (contig, strand) and partition into CDS-block-overlapping components."""
    by_cs: dict[tuple[str, str], list] = {}
    for it in items:
        by_cs.setdefault(get_key(it), []).append(it)
    out: list[list] = []
    for cs in sorted(by_cs):
        grp = by_cs[cs]
        m = len(grp)
        if m == 1:
            out.append(grp)
            continue
        parent = list(range(m))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            rx, ry = _find(x), _find(y)
            if rx != ry:
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry

        ex_list: list[tuple[int, int, int]] = []
        for idx, it in enumerate(grp):
            for a, b in get_cds(it):
                if b > a:
                    ex_list.append((a, b, idx))
        ex_list.sort(key=lambda r: (r[0], r[1]))
        cur_end = -1
        cur_owner = -1
        for a, b, idx in ex_list:
            if cur_owner >= 0 and a < cur_end:
                _union(cur_owner, idx)
                if b > cur_end:
                    cur_end = b
            else:
                cur_owner = idx
                cur_end = b
        comps_by_root: dict[int, list] = {}
        for idx, it in enumerate(grp):
            comps_by_root.setdefault(_find(idx), []).append(it)
        comps = sorted(
            comps_by_root.values(),
            key=lambda cl: (min(a for it in cl for a, _b in get_cds(it)), max(b for it in cl for _a, b in get_cds(it))),
        )
        out.extend(comps)
    return out


def partition_loci(
    tiberius: list[Transcript],
    hits: list[ProteinHit],
    genome: Genome,
    min_hit_positive: float = 0.6,
    min_hit_cov: float = 0.8,
    miniprot_only_min_quality: float = 0.6,
) -> list[Locus]:
    """Cluster Tiberius CDS models and miniprot chains per strand by CDS-block overlap; classify each cluster."""
    items: list[tuple[str, str, int, int, str, object, tuple[tuple[int, int], ...]]] = []
    for t in tiberius:
        if t.cds:
            items.append((t.contig, t.strand, t.cds_start, t.cds_end, "T", t, tuple(sorted(t.cds))))
    for h in hits:
        items.append((h.contig, h.strand, h.start, h.end, "M", h, tuple(sorted(h.cds))))
    clusters = _cds_block_components(items, lambda x: (x[0], x[1]), lambda x: x[6])

    loci: list[Locus] = []
    n = 0
    for cl in clusters:
        tibs = [x[5] for x in cl if x[4] == "T"]
        hs = [x[5] for x in cl if x[4] == "M"]
        ctg, strand = cl[0][0], cl[0][1]
        start = min(x[2] for x in cl)
        end = max(x[3] for x in cl)
        good_hits = [h for h in hs if h.positive >= min_hit_positive and h.coverage >= min_hit_cov]
        if not tibs:
            best = max(hs, key=lambda h: h.quality)
            if best.quality < miniprot_only_min_quality:
                continue  # weak protein-only hit: not a locus candidate
            n += 1
            loci.append(Locus(f"L{n:06d}", ctg, strand, start, end, UNSETTLED, "miniprot_only", [], hs))
            continue
        # Split/fusion: a Tiberius model overlapping >=2 mutually non-overlapping good protein chains,
        # or >=2 Tiberius models overlapped by one good protein chain (by CDS-block overlap).
        category = None
        locked_model = None
        for t in tibs:
            ov = [h for h in good_hits if interval_overlap_bp(t.cds, h.cds) > 0]
            ov.sort(key=lambda h: h.start)
            disjoint = 0
            last_end = -1
            for h in ov:
                if h.start >= last_end:
                    disjoint += 1
                    last_end = h.end
            if disjoint >= 2:
                category = "split_fusion"
        for h in good_hits:
            ov_t = [t for t in tibs if interval_overlap_bp(t.cds, h.cds) > 0]
            if len(ov_t) >= 2:
                category = "split_fusion"
        if category is None:
            for t in tibs:
                st = orf_status(genome.spliced(t.contig, t.strand, t.cds))
                single = len(t.cds) == 1
                if single:
                    category = category or "single_exon"
                    continue
                if not st["valid"]:
                    category = "frameshift_partial"
                    continue
                chain = t.cds_introns
                match = [h for h in good_hits if h.introns == chain and h.frameshift == 0 and h.stop_codon == 0]
                if match:
                    locked_model = t
                    category = None
                    break
                if any(h.introns == chain for h in hs):
                    category = category or "frameshift_partial"
                elif hs:
                    category = category or "discordant_chain"
                else:
                    category = category or "abinitio_only"
        n += 1
        if locked_model is not None and len(tibs) == 1:
            loci.append(Locus(f"L{n:06d}", ctg, strand, start, end, LOCKED, "intron_chain_match", tibs, hs, locked_model))
        else:
            if locked_model is not None:
                category = "multi_model_cluster"
            loci.append(Locus(f"L{n:06d}", ctg, strand, start, end, UNSETTLED, category or "discordant_chain", tibs, hs))
    return loci


def partition_summary(loci: list[Locus]) -> dict:
    out: dict = {"n_loci": len(loci), LOCKED: 0, UNSETTLED: 0, "categories": {}}
    for l in loci:
        out[l.status] += 1
        out["categories"][l.category] = out["categories"].get(l.category, 0) + 1
    out["locked_frac"] = out[LOCKED] / max(1, len(loci))
    return out


def load_tiberius(gtf: Path) -> list[Transcript]:
    txs = read_models(gtf, fmt="gtf")
    for t in txs:
        t.source = "tiberius"
    return [t for t in txs if t.cds]


def protein_lengths(faa: Path) -> dict[str, int]:
    fai = Path(str(faa) + ".fai")
    out = {}
    for line in fai.read_text().splitlines():
        f = line.split("\t")
        out[f[0]] = int(f[1])
    return out


# ----------------------------------------------------------------------------------------------
# Inverted Stage-0 helpers: super-contig packing/lifting, SAFE_LOCK, and zoom windows
# ----------------------------------------------------------------------------------------------


def pack_windows_to_supercontigs(
    genome: Genome,
    windows: list[tuple[str, int, int]],
    out_fa: Path,
    out_manifest: Path,
    target_super_bp: int = 10_000_000,
    spacer_bp: int = 4_500,
) -> dict:
    """Concatenate genomic zoom windows into ~10 Mb stitched super-contigs separated by 4.5 kb 'N' spacers."""
    clean = sorted((ctg, max(0, int(lo)), min(genome.lengths.get(ctg, 0), int(hi))) for ctg, lo, hi in windows if hi > lo)
    merged: list[tuple[str, int, int]] = []
    for ctg, lo, hi in clean:
        if merged and merged[-1][0] == ctg and lo <= merged[-1][2]:
            merged[-1] = (ctg, merged[-1][1], max(merged[-1][2], hi))
        else:
            merged.append((ctg, lo, hi))

    super_contigs: dict[str, dict] = {}
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    spacer = "N" * spacer_bp

    cur_idx = 1
    cur_parts: list[str] = []
    cur_segs: list[dict] = []
    cur_len = 0

    def _flush(fh) -> None:
        nonlocal cur_idx, cur_parts, cur_segs, cur_len
        if not cur_segs:
            return
        sid = f"SUPER_{cur_idx:04d}"
        seq = "".join(cur_parts)
        fh.write(f">{sid}\n{seq}\n")
        super_contigs[sid] = {"length": len(seq), "segments": cur_segs}
        cur_idx += 1
        cur_parts = []
        cur_segs = []
        cur_len = 0

    total_win_bp = 0
    with open(out_fa, "w") as fh:
        for ctg, lo, hi in merged:
            seg_len = hi - lo
            total_win_bp += seg_len
            add_len = seg_len if not cur_segs else (spacer_bp + seg_len)
            if cur_segs and cur_len + add_len > target_super_bp:
                _flush(fh)
            if cur_segs:
                cur_parts.append(spacer)
                cur_len += spacer_bp
            seq = genome.fetch(ctg, lo, hi)
            super_lo = cur_len
            super_hi = cur_len + len(seq)
            cur_parts.append(seq)
            cur_segs.append({
                "super_lo": super_lo,
                "super_hi": super_hi,
                "orig_contig": ctg,
                "orig_lo": lo,
                "orig_hi": hi,
            })
            cur_len = super_hi
        _flush(fh)

    manifest = {
        "spacer_bp": spacer_bp,
        "target_super_bp": target_super_bp,
        "n_windows": len(merged),
        "n_super_contigs": len(super_contigs),
        "total_window_bp": total_win_bp,
        "total_super_bp": sum(v["length"] for v in super_contigs.values()),
        "contig_lengths": {c: int(genome.lengths[c]) for c in sorted({w[0] for w in merged}) if c in genome.lengths},
        "super_contigs": super_contigs,
    }
    out_manifest.write_text(json.dumps(manifest, indent=1))
    return manifest


def _append_gtf_attrs(col9: str, extra_kv: list[tuple[str, str]]) -> str:
    base = col9.strip().rstrip(";").strip()
    parts = [f'{k} "{v}"' for k, v in extra_kv]
    if base:
        return f"{base}; " + "; ".join(parts) + ";"
    return "; ".join(parts) + ";"


def _build_trimmed_gtf_rows(
    orig_ctg: str,
    src: str,
    strand: str,
    gid: str,
    tid: str,
    rep_cds: list[tuple[int, int]],
    orig_lo: int,
    orig_hi: int,
    clip_left: bool,
    clip_right: bool,
) -> list[list[str]]:
    segs = sorted(rep_cds)
    tx_tmp = Transcript(tid=tid, contig=orig_ctg, strand=strand, exons=segs, cds=segs, gid=gid, source=src)
    phases = _phases(tx_tmp)
    base_id = f'gene_id "{gid}"; transcript_id "{tid}";'
    tx_col9 = _append_gtf_attrs(
        base_id,
        [
            ("trimmed_spacer", "1"),
            ("seg_lo", str(orig_lo)),
            ("seg_hi", str(orig_hi)),
            ("clip_left", "1" if clip_left else "0"),
            ("clip_right", "1" if clip_right else "0"),
        ],
    )
    rows: list[list[str]] = [
        [orig_ctg, src, "transcript", str(segs[0][0] + 1), str(segs[-1][1]), ".", strand, ".", tx_col9]
    ]
    for (a, b), ph in zip(segs, phases):
        rows.append([orig_ctg, src, "exon", str(a + 1), str(b), ".", strand, ".", base_id])
        rows.append([orig_ctg, src, "CDS", str(a + 1), str(b), ".", strand, str(ph), base_id])
    return rows


def lift_tiberius_gtf(
    super_gtf: Path,
    manifest: dict | Path,
    out_gtf: Path,
    genome: Genome | None = None,
    out_faa: Path | None = None,
    trim_spacer_bridges: bool = True,
) -> dict:
    """Lift super-contig Tiberius GTF coordinates back to original contigs with B1 clip flags & B2 spacer trimming."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())

    contig_lengths: dict[str, int] = {str(k): int(v) for k, v in (manifest.get("contig_lengths") or {}).items()}
    if genome is not None:
        contig_lengths.update({str(k): int(v) for k, v in genome.lengths.items()})

    sc_map = manifest.get("super_contigs", {})
    seg_lookup: dict[str, tuple[list[int], list[dict]]] = {}
    for sid, info in sc_map.items():
        segs = sorted(info["segments"], key=lambda s: s["super_lo"])
        seg_lookup[sid] = ([s["super_lo"] for s in segs], segs)

    tx_order: list[str] = []
    tx_lines: dict[str, list[list[str]]] = {}
    tx_gid: dict[str, str] = {}
    gene_order: list[str] = []
    gene_meta: dict[str, list[str]] = {}

    if super_gtf.exists():
        with open(super_gtf) as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9:
                    continue
                at = parse_attrs(f[8], is_gtf=True)
                gid = at.get("gene_id", "")
                tid = at.get("transcript_id", "")
                if f[2] == "gene" and gid and not tid:
                    if gid not in gene_meta:
                        gene_order.append(gid)
                        gene_meta[gid] = f
                    continue
                key = tid or gid or f"line_{len(tx_order)}"
                if key not in tx_lines:
                    tx_order.append(key)
                    tx_lines[key] = []
                    tx_gid[key] = gid
                tx_lines[key].append(f)

    def _lift_coord(sid: str, s1: int, e1: int) -> tuple[int, str, int, int] | None:
        entry = seg_lookup.get(sid)
        if not entry:
            return None
        starts, segs = entry
        a0, b0 = s1 - 1, e1
        idx = bisect.bisect_right(starts, a0) - 1
        if idx < 0:
            return None
        seg = segs[idx]
        if a0 < seg["super_lo"] or b0 > seg["super_hi"]:
            return None
        orig_a0 = seg["orig_lo"] + (a0 - seg["super_lo"])
        orig_b0 = seg["orig_lo"] + (b0 - seg["super_lo"])
        return idx, seg["orig_contig"], orig_a0 + 1, orig_b0

    lifted_by_gid: dict[str, list[tuple[str, list[list[str]]]]] = {}
    orphan_tx: list[tuple[str, list[list[str]]]] = []
    n_discarded_spacer = 0
    n_trimmed_spacer = 0
    n_lifted_tx = 0

    for tid in tx_order:
        lines = tx_lines[tid]
        lifted_rows: list[list[str]] = []
        ref_seg: tuple[str, int] | None = None
        bad = False
        for row in lines:
            hit = _lift_coord(row[0], int(row[3]), int(row[4]))
            if hit is None:
                bad = True
                break
            seg_idx, orig_ctg, orig_s1, orig_e1 = hit
            if ref_seg is None:
                ref_seg = (row[0], seg_idx)
            elif ref_seg != (row[0], seg_idx):
                bad = True
                break
            nrow = list(row)
            nrow[0] = orig_ctg
            nrow[3] = str(orig_s1)
            nrow[4] = str(orig_e1)
            lifted_rows.append(nrow)

        if not bad and lifted_rows and ref_seg is not None:
            sid, seg_idx = ref_seg
            seg = seg_lookup[sid][1][seg_idx]
            orig_ctg = seg["orig_contig"]
            orig_lo = int(seg["orig_lo"])
            orig_hi = int(seg["orig_hi"])
            L = contig_lengths.get(orig_ctg, orig_hi)
            cds_ivs = [(int(r[3]) - 1, int(r[4])) for r in lifted_rows if r[2] in ("CDS", "stop_codon")]
            if not cds_ivs:
                cds_ivs = [(int(r[3]) - 1, int(r[4])) for r in lifted_rows]
            cds_lo = min(a for a, _ in cds_ivs)
            cds_hi = max(b for _, b in cds_ivs)
            c_left = (orig_lo > 0) and (cds_lo - orig_lo <= 1_500)
            c_right = (orig_hi < L) and (orig_hi - cds_hi <= 1_500)
            if c_left or c_right:
                extra_kv = [
                    ("seg_lo", str(orig_lo)),
                    ("seg_hi", str(orig_hi)),
                    ("clip_left", "1" if c_left else "0"),
                    ("clip_right", "1" if c_right else "0"),
                ]
                has_tx_row = False
                for r in lifted_rows:
                    if r[2] == "transcript":
                        r[8] = _append_gtf_attrs(r[8], extra_kv)
                        has_tx_row = True
                if not has_tx_row:
                    eff_gid = tx_gid.get(tid, "") or tid
                    tx_col9 = _append_gtf_attrs(f'gene_id "{eff_gid}"; transcript_id "{tid}";', extra_kv)
                    tx_lo = min(int(r[3]) for r in lifted_rows)
                    tx_hi = max(int(r[4]) for r in lifted_rows)
                    lifted_rows.insert(
                        0, [orig_ctg, lifted_rows[0][1], "transcript", str(tx_lo), str(tx_hi), ".", lifted_rows[0][6], ".", tx_col9]
                    )
            n_lifted_tx += 1
            gid = tx_gid.get(tid, "")
            if gid:
                lifted_by_gid.setdefault(gid, []).append((tid, lifted_rows))
            else:
                orphan_tx.append((tid, lifted_rows))
            continue

        trimmed_rows: list[list[str]] | None = None
        if genome is not None and trim_spacer_bridges and lines:
            sids = {r[0] for r in lines}
            sid = lines[0][0]
            if len(sids) == 1 and sid in seg_lookup:
                segs = seg_lookup[sid][1]
                seg_cds_by_idx: dict[int, list[tuple[int, int]]] = {}
                raw_row_spans: list[tuple[int, int, int | None]] = []
                for row in lines:
                    s0, e0 = int(row[3]) - 1, int(row[4])
                    hit = _lift_coord(sid, int(row[3]), int(row[4]))
                    hseg = hit[0] if hit is not None else None
                    raw_row_spans.append((s0, e0, hseg))
                    if hit is not None and row[2] in ("CDS", "stop_codon"):
                        seg_cds_by_idx.setdefault(hit[0], []).append((hit[2] - 1, hit[3]))
                if seg_cds_by_idx:
                    best_idx = max(
                        seg_cds_by_idx,
                        key=lambda idx: (sum(b - a for a, b in merge_intervals(seg_cds_by_idx[idx])), -idx),
                    )
                    best_cds = merge_intervals(seg_cds_by_idx[best_idx])
                    best_bp = sum(b - a for a, b in best_cds)
                    if best_bp >= 180:
                        seg = segs[best_idx]
                        orig_ctg = seg["orig_contig"]
                        orig_lo = int(seg["orig_lo"])
                        orig_hi = int(seg["orig_hi"])
                        L = contig_lengths.get(orig_ctg, orig_hi)
                        strand = lines[0][6]
                        if orf_status(genome.spliced(orig_ctg, strand, list(best_cds)))["valid"]:
                            rep_cds: list[tuple[int, int]] | None = list(best_cds)
                        else:
                            rep_cds = repair_orf(
                                genome, orig_ctg, strand, list(best_cds), max_ext=150, bounds=(orig_lo, orig_hi)
                            )
                        if rep_cds:
                            disc_left = any(s0 < seg["super_lo"] for s0, _e0, hseg in raw_row_spans if hseg != best_idx)
                            disc_right = any(e0 > seg["super_hi"] for _s0, e0, hseg in raw_row_spans if hseg != best_idx)
                            c_left = (orig_lo > 0) and (disc_left or (rep_cds[0][0] - orig_lo <= 1_500))
                            c_right = (orig_hi < L) and (disc_right or (orig_hi - rep_cds[-1][1] <= 1_500))
                            eff_gid = tx_gid.get(tid, "") or tid
                            trimmed_rows = _build_trimmed_gtf_rows(
                                orig_ctg, lines[0][1], strand, eff_gid, tid, rep_cds, orig_lo, orig_hi, c_left, c_right
                            )

        if trimmed_rows is None:
            n_discarded_spacer += 1
            continue
        n_trimmed_spacer += 1
        n_lifted_tx += 1
        gid = tx_gid.get(tid, "")
        if gid:
            lifted_by_gid.setdefault(gid, []).append((tid, trimmed_rows))
        else:
            orphan_tx.append((tid, trimmed_rows))

    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_gtf, "w") as out:
        seen_gids: set[str] = set()
        for gid in gene_order:
            txs = lifted_by_gid.get(gid)
            if not txs:
                continue
            seen_gids.add(gid)
            g_min = min(int(r[3]) for _, rows in txs for r in rows)
            g_max = max(int(r[4]) for _, rows in txs for r in rows)
            grow = list(gene_meta[gid])
            grow[0] = txs[0][1][0][0]
            grow[3] = str(g_min)
            grow[4] = str(g_max)
            out.write("\t".join(grow) + "\n")
            for _, rows in txs:
                for r in rows:
                    out.write("\t".join(r) + "\n")
        for gid, txs in lifted_by_gid.items():
            if gid in seen_gids:
                continue
            for _, rows in txs:
                for r in rows:
                    out.write("\t".join(r) + "\n")
        for _, rows in orphan_tx:
            for r in rows:
                out.write("\t".join(r) + "\n")

    if out_faa is not None and genome is not None:
        txs = load_tiberius(out_gtf)
        with open(out_faa, "w") as fh:
            for t in txs:
                ph = int(getattr(t, "cds_phase", 0) or 0) % 3
                aa = translate(genome.spliced(t.contig, t.strand, t.cds)[ph:]).rstrip("*")
                if aa:
                    fh.write(f">{t.tid}\n{aa}\n")

    return {
        "n_input_tx": len(tx_order),
        "n_lifted_tx": n_lifted_tx,
        "n_trimmed_spacer_tx": n_trimmed_spacer,
        "n_discarded_spacer_tx": n_discarded_spacer,
        "out_gtf": str(out_gtf),
    }


def merge_tiberius_gtfs(gtf_paths: list[Path], out_gtf: Path, genome: Genome | None = None, out_faa: Path | None = None) -> int:
    """Concatenate multiple lifted Tiberius GTFs with unique gene/transcript prefixes."""
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    n_tx = 0
    with open(out_gtf, "w") as out:
        for idx, gp in enumerate(gtf_paths):
            if not gp.exists():
                continue
            pfx = f"S{idx}_"
            with open(gp) as fh:
                for line in fh:
                    if not line or line[0] == "#":
                        continue
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 9:
                        continue
                    if f[2] == "transcript":
                        n_tx += 1
                    col9 = re.sub(r'(\b(?:gene_id|transcript_id)\s+")([^"]+)(")', lambda m: f"{m.group(1)}{pfx}{m.group(2)}{m.group(3)}", f[8])
                    f[8] = col9
                    out.write("\t".join(f) + "\n")
    if out_faa is not None and genome is not None:
        txs = load_tiberius(out_gtf)
        with open(out_faa, "w") as fh:
            for t in txs:
                ph = int(getattr(t, "cds_phase", 0) or 0) % 3
                aa = translate(genome.spliced(t.contig, t.strand, t.cds)[ph:]).rstrip("*")
                if aa:
                    fh.write(f">{t.tid}\n{aa}\n")
    return n_tx


# Y6: tight_v2 is an explicit CLI/config alias for the tight_v1 windowing preset (5 kb merge_gap,
# tight homology/RNA safe-lock floors, 15 kb default spacer in run_vgp_4taxa); B0/B1/B2 apply to all presets.
_VALID_ZOOM_PRESETS = ("default", "tight_v1", "tight_v2")


def _has_genomic_orf(seq: str, min_nt: int = 300) -> bool:
    """Return True if any of the 6 reading frames contains an uninterrupted ORF >= min_nt bases."""
    u = seq.upper()
    n = len(u)
    for st in (u, u.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]):
        for frame in range(3):
            run = 0
            i = frame
            while i + 3 <= n:
                cod = st[i : i + 3]
                if cod in ("TAA", "TAG", "TGA"):
                    if run >= min_nt:
                        return True
                    run = 0
                else:
                    run += 3
                i += 3
            if run >= min_nt:
                return True
    return False


def _is_broken_retrocopy(h: ProteinHit) -> bool:
    """True for single-exon miniprot hits carrying internal frameshifts or premature stop codons."""
    return len(h.cds) == 1 and (h.frameshift > 0 or h.stop_codon > 0)


def safe_lock_miniprot_loci(
    hits: list[ProteinHit],
    genome: Genome,
    scout_juncs=None,
    min_hit_positive: float = 0.60,
    min_hit_cov: float = 0.80,
    zoom_preset: str = "default",
) -> tuple[list[Locus], list[ProteinHit], list[ProteinHit]]:
    """Promote clean conserved multi-exon miniprot clusters directly to LOCKED_HIGH_CONF without Tiberius."""
    if zoom_preset not in _VALID_ZOOM_PRESETS:
        raise ValueError(f"Unsupported zoom_preset={zoom_preset!r}; expected one of {_VALID_ZOOM_PRESETS}")
    tight = zoom_preset in ("tight_v1", "tight_v2")

    base_hits = [h for h in hits if (not tight or h.quality >= 0.50)]
    clusters = _cds_block_components(base_hits, lambda h: (h.contig, h.strand), lambda h: h.cds)

    strong_juncs_by_ctg = (
        scout_juncs.by_contig(min_support=5, canonical_only=True)
        if (tight and scout_juncs is not None)
        else {}
    )
    strong_junc_starts = {
        ctg: [ja for ja, _jb, _jsup, _jst in jlist]
        for ctg, jlist in strong_juncs_by_ctg.items()
    }
    ext_junc_idx = (
        LocusIndex(
            [
                (ctg, ja, jb, (ja, jb, jsup, jst))
                for ctg, jlist in scout_juncs.by_contig(min_support=2, canonical_only=True).items()
                for ja, jb, jsup, jst in jlist
                if jst in ("+", "-")
            ],
            pad=5,
        )
        if scout_juncs is not None
        else None
    )

    def _has_open_upstream_exon(contig: str, strand: str, ja: int, jb: int) -> bool:
        """True if >=15 bp immediately upstream of the 5'-abutting intron stays open in-frame with hit."""
        if strand != "-":
            up = genome.fetch(contig, max(0, ja - 60), ja).upper()
        else:
            L_ctg = genome.lengths.get(contig, jb + 60)
            up = revcomp(genome.fetch(contig, jb, min(L_ctg, jb + 60))).upper()
        up_cod = up[len(up) % 3 :]
        if len(up_cod) < 15:
            return False
        return all(up_cod[i : i + 3] not in ("TAA", "TAG", "TGA") for i in range(0, len(up_cod), 3))

    def _has_extending_or_conflicting_junc(hit: ProteinHit, intr_set: set[tuple[int, int]]) -> bool:
        if ext_junc_idx is None:
            return False
        for ja, jb, jsup, jst in ext_junc_idx.overlapping(hit.contig, hit.start, hit.end):
            if jst != hit.strand or (ja, jb) in intr_set:
                continue
            if ja < hit.start < jb or ja < hit.end < jb:
                return True  # splice intron crosses hit boundary -> partial fragment inside longer gene
            if hit.start <= ja and jb <= hit.end:
                if jsup >= 5 or (jsup >= 3 and any(cx < ja and jb < cy for cx, cy in hit.cds)):
                    return True
            elif jsup >= 3 and ((hit.strand != "-" and jb == hit.start) or (hit.strand == "-" and ja == hit.end)):
                if _has_open_upstream_exon(hit.contig, hit.strand, ja, jb):
                    return True
        return False

    safe_loci: list[Locus] = []
    safe_hits: list[ProteinHit] = []
    unsettled_hits: list[ProteinHit] = []
    if tight:
        # Preserve low-quality hits excluded from clustering in unsettled_hits so accounting stays exact.
        unsettled_hits.extend(h for h in hits if h.quality < 0.50)
    safe_idx = 0

    pos_floor = 0.75 if tight else 0.72
    cov_floor = 0.92 if tight else 0.90

    for hs in clusters:
        good = [h for h in hs if h.positive >= min_hit_positive and h.coverage >= min_hit_cov]
        good_sorted = sorted(good, key=lambda h: h.start)
        disjoint_good = 0
        last_end = -1
        for h in good_sorted:
            if h.start >= last_end:
                disjoint_good += 1
                last_end = h.end

        best = max(hs, key=lambda h: (h.quality, h.n_rank1, h.n_proteins))
        is_safe = False
        if (
            disjoint_good == 1
            and len(best.cds) >= 2
            and best.frameshift == 0
            and best.stop_codon == 0
            and best.positive >= pos_floor
            and best.coverage >= cov_floor
        ):
            st = orf_status(genome.spliced(best.contig, best.strand, list(best.cds)))
            if st["valid"]:
                introns = best.introns
                if tight:
                    genomic_canon_frac = (
                        sum(1 for a, b in introns if junction_motif(genome, best.contig, a, b)[0] in CANONICAL)
                        / max(1, len(introns))
                    )
                    rna_ok = False
                    if scout_juncs is not None and best.positive >= 0.75 and best.coverage >= 0.92:
                        sups = [scout_juncs.support((best.contig, a, b)) for a, b in introns]
                        term_ok = bool(sups) and sups[0] >= 2 and sups[-1] >= 2
                        sup_ok_frac = (
                            sum(
                                1
                                for (a, b), s in zip(introns, sups)
                                if s >= 2 and scout_juncs.canonical((best.contig, a, b))
                            )
                            / max(1, len(introns))
                        )
                        no_conflict = True
                        sj_list = strong_juncs_by_ctg.get(best.contig)
                        if sj_list:
                            sj_starts = strong_junc_starts[best.contig]
                            idx0 = bisect.bisect_left(sj_starts, best.start)
                            intr_set = set(introns)
                            for ja, jb, _jsup, _jst in sj_list[idx0:]:
                                if ja > best.end:
                                    break
                                if jb <= best.end and (ja, jb) not in intr_set:
                                    no_conflict = False
                                    break
                        rna_ok = sup_ok_frac >= 0.80 and genomic_canon_frac == 1.0 and term_ok and no_conflict
                    hom_ok = (
                        best.positive >= 0.80
                        and best.coverage >= 0.92
                        and genomic_canon_frac == 1.0
                        and (best.n_rank1 >= 2 or (best.n_proteins >= 3 and best.positive >= 0.85))
                    )
                else:
                    rna_ok = (
                        scout_juncs is not None
                        and all(
                            scout_juncs.support((best.contig, a, b)) >= 2
                            and scout_juncs.canonical((best.contig, a, b))
                            for a, b in introns
                        )
                    )
                    hom_ok = best.positive >= 0.85 and best.coverage >= 0.95 and best.n_rank1 >= 2
                if (rna_ok or hom_ok) and not _has_extending_or_conflicting_junc(best, set(introns)):
                    is_safe = True

        if is_safe:
            safe_idx += 1
            sid = f"SAFE_{safe_idx:06d}"
            tx = Transcript(
                sid,
                best.contig,
                best.strand,
                list(best.cds),
                list(best.cds),
                gid=sid,
                source="miniprot_safelock",
            )
            start = best.start if tight else min(h.start for h in hs)
            end = best.end if tight else max(h.end for h in hs)
            safe_loci.append(Locus(sid, best.contig, best.strand, start, end, LOCKED, "safe_locked_miniprot", [], hs, tx))
            safe_hits.extend(hs)
        else:
            unsettled_hits.extend(hs)

    return safe_loci, safe_hits, unsettled_hits


def filter_non_safelock_tiberius(
    tiberius: list[Transcript],
    safe_loci: list[Locus],
    genome: Genome | None = None,
) -> list[Transcript]:
    """Drop Tiberius models redundant with ``safe_loci``, preserving compatible extensions of partial safe_loci."""
    if not safe_loci or not tiberius:
        return list(tiberius)
    safe_idx = LocusIndex([(l.contig, l.start, l.end, l) for l in safe_loci])
    kept: list[Transcript] = []
    for t in tiberius:
        ov_safe = [
            l
            for l in safe_idx.overlapping(t.contig, t.cds_start, t.cds_end)
            if l.strand == t.strand
            and (
                getattr(l, "locked_model", None) is None
                or not l.locked_model.cds
                or interval_overlap_bp(t.cds, l.locked_model.cds) > 0
            )
        ]
        if not ov_safe:
            kept.append(t)
            continue
        if len(ov_safe) >= 2 or len(t.cds) < 3:
            continue
        if genome is not None:
            cds_seq = genome.spliced(t.contig, t.strand, t.cds)
            ph = int(getattr(t, "cds_phase", 0) or 0)
            if not (
                orf_status(cds_seq)["valid"]
                or is_scaffold_edge_partial_cds(genome, t.contig, t.strand, t.cds, cds_phase=ph)
            ):
                continue
        t_cds = sorted(t.cds)
        t_len = sum(b - a for a, b in t_cds)
        t_introns = t.cds_introns
        t_intron_set = set(t_introns)
        l = ov_safe[0]
        lm = getattr(l, "locked_model", None)
        if lm is None or not lm.cds:
            continue
        lm_cds = sorted(lm.cds)
        lm_introns = set(lm.cds_introns)
        lm_len = sum(b - a for a, b in lm_cds)
        if not lm_introns.issubset(t_intron_set):
            continue
        if any(l.start <= ia and ib <= l.end and (ia, ib) not in lm_introns for ia, ib in t_introns):
            continue
        if interval_overlap_bp(t_cds, lm_cds) < lm_len:
            continue
        if genome is not None and orf_status(genome.spliced(lm.contig, lm.strand, lm_cds))["valid"]:
            continue
        if any(ia < l.start or ib > l.end for ia, ib in t_introns) and len(t_cds) > len(lm_cds) and t_len > lm_len:
            kept.append(t)
    return kept


def build_tiberius_zoom_windows(
    genome: Genome,
    unsettled_hits: list[ProteinHit],
    safe_loci: list[Locus],
    scout_juncs=None,
    scout_bins=None,
    bait_intervals: dict[str, list[tuple[int, int]]] | None = None,
    organelle_contigs: set[str] | None = None,
    pad_tib: int | None = None,
    merge_gap: int | None = None,
    zoom_preset: str = "default",
) -> dict:
    """Compute merged Tiberius zoom windows over uncertain miniprot hits, RNA junctions, and transcribed islands."""
    if zoom_preset not in _VALID_ZOOM_PRESETS:
        raise ValueError(f"Unsupported zoom_preset={zoom_preset!r}; expected one of {_VALID_ZOOM_PRESETS}")
    tight = zoom_preset in ("tight_v1", "tight_v2")

    org = organelle_contigs or set()
    bait_map = bait_intervals or {}
    nuclear_bp = sum(L for c, L in genome.lengths.items() if c not in org)
    if pad_tib is None:
        pad_tib = 12_000 if nuclear_bp < 250_000_000 else 25_000
    eff_merge_gap = merge_gap if merge_gap is not None else (5_000 if tight else 10_000)

    pad_intact = 8_000 if tight else pad_tib
    pad_trunc = 22_000 if tight else pad_tib
    pad_junc = 10_000 if tight else pad_tib
    pad_bin = 6_000 if tight else pad_tib

    safe_by_ctg: dict[str, list[tuple[int, int]]] = {}
    safe_by_ctg_strand: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for l in safe_loci:
        safe_by_ctg.setdefault(l.contig, []).append((l.start, l.end))
        safe_by_ctg_strand.setdefault((l.contig, l.strand), []).append((l.start, l.end))

    raw_by_ctg: dict[str, list[tuple[int, int]]] = {}

    for h in unsettled_hits:
        if h.contig in org or h.contig not in genome.lengths:
            continue
        if tight and (h.quality < 0.50 or _is_broken_retrocopy(h)):
            continue
        L = genome.lengths[h.contig]
        is_intact = (h.coverage >= 0.90 and h.frameshift == 0 and h.stop_codon == 0)
        pad = (pad_intact if is_intact else pad_trunc) if tight else pad_tib
        raw_by_ctg.setdefault(h.contig, []).append((max(0, h.start - pad), min(L, h.end + pad)))

    if scout_juncs is not None:
        for ctg, jlist in scout_juncs.by_contig(min_support=2, canonical_only=True).items():
            if ctg in org or ctg not in genome.lengths:
                continue
            L = genome.lengths[ctg]
            baits = bait_map.get(ctg, ())
            for a, b, _sup, jst in jlist:
                slocs = (
                    safe_by_ctg_strand.get((ctg, jst), ())
                    if (tight and jst in "+-")
                    else safe_by_ctg.get(ctg, ())
                )
                if any(sa <= a and b <= sb for sa, sb in slocs):
                    continue
                if any(ba < b and a < bb for ba, bb in baits):
                    continue
                raw_by_ctg.setdefault(ctg, []).append((max(0, a - pad_junc), min(L, b + pad_junc)))

    if scout_bins is not None:
        min_reads = 15 if tight else 5
        for ctg, arr in scout_bins.arr.items():
            if ctg in org or ctg not in genome.lengths:
                continue
            L = genome.lengths[ctg]
            slocs = safe_by_ctg.get(ctg, ())
            baits = bait_map.get(ctg, ())
            for bidx in np.flatnonzero(arr >= min_reads):
                a = int(bidx) * scout_bins.bin
                b = min(L, (int(bidx) + 1) * scout_bins.bin)
                if any(sa < b and a < sb for sa, sb in slocs):
                    continue
                if any(ba < b and a < bb for ba, bb in baits):
                    continue
                if tight and not _has_genomic_orf(genome.fetch(ctg, max(0, a - 1000), min(L, b + 1000)), 300):
                    continue
                raw_by_ctg.setdefault(ctg, []).append((max(0, a - pad_bin), min(L, b + pad_bin)))

    merged_windows: list[tuple[str, int, int]] = []
    by_ctg: dict[str, list[tuple[int, int]]] = {}
    for ctg in genome.lengths:
        ivs = sorted(raw_by_ctg.get(ctg, []))
        if not ivs:
            continue
        m: list[list[int]] = []
        for a, b in ivs:
            if m and a <= m[-1][1] + eff_merge_gap:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        ctg_ivs = [(a, b) for a, b in m if b > a]
        if ctg_ivs:
            by_ctg[ctg] = ctg_ivs
            for a, b in ctg_ivs:
                merged_windows.append((ctg, a, b))

    tib_window_bp = sum(b - a for _c, a, b in merged_windows)
    return {
        "windows": merged_windows,
        "by_contig": by_ctg,
        "tib_window_bp": tib_window_bp,
        "nuclear_genome_bp": nuclear_bp,
        "tib_window_frac": round(tib_window_bp / max(1, nuclear_bp), 6),
        "pad_tib": pad_tib,
        "merge_gap": eff_merge_gap,
        "zoom_preset": zoom_preset,
    }
