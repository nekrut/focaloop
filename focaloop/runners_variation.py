"""Production Galaxy runner for Focaloop VariationLoop (Pathogen + Diploid Human VarLoop)."""

from __future__ import annotations

import csv
import gzip
import html
import io
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .archive import pack_state
from .engine import STATE_META, write_json
from .hooks import HypothesisLedger
from .state import FocaloopStateStore

_INSDC_RUN_RE = re.compile(r"^[SED]RR\d+$", re.IGNORECASE)
_READ_EXT_RE = re.compile(r"(\.f(ast)?qsanger(\.gz|\.bz2)?|\.f(ast)?q(\.gz|\.bz2)?|\.gz|\.bz2)+$", re.IGNORECASE)
_MATE_SUFFIX_RE = re.compile(r"(_R?[12]|_[12])$", re.IGNORECASE)

_BASE_CSS = """
<style>
  :root {
    --bg: #f8fafc; --card: #ffffff; --ink: #0f172a; --muted: #475569;
    --border: #e2e8f0; --accent: #0f766e; --accent-soft: #ccfbf1;
    --blue: #1d4ed8; --blue-soft: #dbeafe; --amber: #b45309; --amber-soft: #fef3c7;
    --rose: #be123c; --rose-soft: #ffe4e6; --green: #15803d; --green-soft: #dcfce7;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--ink); line-height: 1.45; font-size: 14px; }
  .wrap { max-width: 1240px; margin: 0 auto; }
  .hero { background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%); color: #fff;
          border-radius: 12px; padding: 22px 26px; margin-bottom: 18px; }
  .hero h1 { margin: 0 0 6px 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }
  .hero p { margin: 0; color: #cbd5e1; font-size: 13.5px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .kpi .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .kpi .val { font-size: 22px; font-weight: 700; margin-top: 4px; color: var(--ink); }
  .kpi .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
             padding: 18px 20px; margin-bottom: 16px; }
  .section h2 { margin: 0 0 10px 0; font-size: 16px; font-weight: 650;
                border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 9px; border-bottom: 1px solid var(--border); }
  th { background: #f1f5f9; font-weight: 600; color: #334155; }
  tr:hover td { background: #f8fafc; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .b-teal { background: var(--accent-soft); color: var(--accent); }
  .b-blue { background: var(--blue-soft); color: var(--blue); }
  .b-amber { background: var(--amber-soft); color: var(--amber); }
  .b-rose { background: var(--rose-soft); color: var(--rose); }
  .b-green { background: var(--green-soft); color: var(--green); }
  .bar-bg { width: 110px; height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; display: inline-block; vertical-align: middle; }
  .bar-fg { height: 100%; background: var(--accent); }
  code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
"""


_AGGRESSIVE_ERR_RE = re.compile(r"(error|exception)\s*:", re.IGNORECASE)


def _log(msg: str) -> None:
    safe_msg = _AGGRESSIVE_ERR_RE.sub(r"\1 -", str(msg))
    print(f"[focaloop-variationloop] {safe_msg}", file=sys.stderr, flush=True)


def _esc(val: Any) -> str:
    return html.escape("" if val is None else str(val))


def _fmt_num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
        if not np.isfinite(fv):
            return "—"
        if abs(fv) > 0 and (abs(fv) < 1e-3 or abs(fv) >= 1e5):
            return f"{fv:.2e}"
        return f"{fv:.{digits}f}"
    except Exception:
        return _esc(v)


def _gunzip_if_needed(src: str | Path, dst_dir: Path, out_name: str) -> Path:
    src_p = Path(src).resolve()
    dst = dst_dir / out_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(src_p, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(src_p, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        return dst
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src_p, dst)
    return dst


def _is_gzip_file(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _stage_local_fastq(src: Path, dst_dir: Path, stem: str) -> Path:
    """Ensure local/Galaxy `.dat` FASTQs expose a `.fastq.gz` or `.fastq` suffix for aligners like strobealign."""
    src_p = src.resolve()
    is_gz = _is_gzip_file(src_p)
    low = src_p.name.lower()
    if is_gz and low.endswith((".fq.gz", ".fastq.gz", ".fastqsanger.gz")):
        return src_p
    if (not is_gz) and low.endswith((".fq", ".fastq", ".fastqsanger")):
        return src_p
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext = ".fastq.gz" if is_gz else ".fastq"
    link = dst_dir / f"{stem}{ext}"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(src_p, link)
    return link


def _detect_aln_format(src: Path) -> str | None:
    """Detect CRAM or BAM by magic bytes (handles Galaxy `.dat` datasets)."""
    if not src.is_file():
        return None
    with open(src, "rb") as fh:
        head = fh.read(4)
    if head == b"CRAM":
        return "cram"
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(src, "rb") as gz:
                if gz.read(4) == b"BAM\x01":
                    return "bam"
        except Exception:
            pass
    return None


def _stage_local_aln(src: Path, dst_dir: Path, stem: str, fmt: str) -> Path:
    src_p = src.resolve()
    if src_p.name.lower().endswith(f".{fmt}"):
        return src_p
    dst_dir.mkdir(parents=True, exist_ok=True)
    link = dst_dir / f"{stem}.{fmt}"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(src_p, link)
    return link


def _ensure_faidx(fa_path: Path) -> None:
    if Path(str(fa_path) + ".fai").exists():
        return
    if shutil.which("samtools"):
        subprocess.run(["samtools", "faidx", str(fa_path)], check=False)
    else:
        try:
            import pysam
            pysam.faidx(str(fa_path))
        except Exception:
            pass


def _read_manifest_table(manifest_path: str | Path) -> list[dict[str, str]]:
    text = Path(manifest_path).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{manifest_path}: manifest is empty")
    delim = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
    rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{manifest_path}: manifest has header but zero sample rows")
    return rows


def _pick_col(row: dict[str, str], candidates: Sequence[str], default: str = "") -> str:
    lower_map = {k.lower(): v for k, v in row.items()}
    for c in candidates:
        if c in row and row[c]:
            return row[c]
        if c.lower() in lower_map and lower_map[c.lower()]:
            return lower_map[c.lower()]
    return default


def _ipv4_session():
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    s = requests.Session()
    s.headers["User-Agent"] = "focaloop/0.3.0"
    return s


def _head_content_length(url: str, timeout: float = 45.0) -> int:
    s = _ipv4_session()
    try:
        r = s.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code < 400 and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    r = s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
    try:
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            tot = cr.rsplit("/", 1)[-1]
            if tot.isdigit():
                return int(tot)
        if r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    finally:
        r.close()
    raise RuntimeError(f"Could not determine remote byte length for {url}")


def _fetch_crai_file(crai_uri: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(crai_uri).exists():
        shutil.copy2(crai_uri, dst)
        return dst
    s = _ipv4_session()
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = s.get(crai_uri, timeout=60.0)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return dst
        except Exception as exc:
            last_err = exc
            try:
                out = subprocess.run(
                    ["curl", "-4", "-sSL", "--fail", "--max-time", "60", crai_uri],
                    check=True, capture_output=True, timeout=75,
                ).stdout
                dst.write_bytes(out)
                return dst
            except Exception as exc2:
                last_err = exc2
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch CRAI {crai_uri}: {last_err}")


def _resolve_ena_or_odp(acc: str) -> dict[str, Any]:
    acc = acc.strip().upper()
    odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
    all_odp = False
    odp_bytes = np.nan
    try:
        odp_bytes = float(_head_content_length(odp_url, timeout=20.0))
        all_odp = True
    except Exception:
        pass

    fastq_urls: list[str] = []
    fastq_bytes = np.nan
    order_hint = "flowcell"
    s = _ipv4_session()
    try:
        r = s.get(
            "https://www.ebi.ac.uk/ena/portal/api/filereport",
            params={
                "accession": acc,
                "result": "read_run",
                "fields": "run_accession,fastq_ftp,fastq_bytes,submitted_format",
                "format": "tsv",
            },
            timeout=30.0,
        )
        if r.status_code == 200:
            parsed = list(csv.DictReader(io.StringIO(r.text), delimiter="\t"))
            if parsed:
                row0 = parsed[0]
                fq = row0.get("fastq_ftp") or ""
                if fq:
                    fastq_urls = [fq]
                fb = [int(x) for x in (row0.get("fastq_bytes") or "").split(";") if x.isdigit()]
                if fb:
                    fastq_bytes = float(sum(fb))
                sfmt = (row0.get("submitted_format") or "").upper()
                if sfmt and "SRF" not in sfmt and ("BAM" in sfmt or "CRAM" in sfmt):
                    order_hint = "sorted"
    except Exception:
        pass
    return {
        "ena_runs": [acc],
        "all_odp": all_odp,
        "odp_bytes": odp_bytes,
        "all_paired_fastq": bool(fastq_urls and "_1.fastq.gz" in fastq_urls[0] and "_2.fastq.gz" in fastq_urls[0]),
        "fastq_urls": fastq_urls,
        "fastq_bytes": fastq_bytes,
        "order_hint": order_hint,
    }


def _mapper_runs_cleanly(cand: str) -> bool:
    exe = shutil.which(cand)
    if not exe:
        return False
    try:
        if cand in ("strobealign", "minimap2"):
            with tempfile.TemporaryDirectory(prefix=f"focaloop_{cand}_chk_") as td:
                fa = Path(td) / "r.fa"
                fq = Path(td) / "q.fq"
                seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                fa.write_text(f">chr1\n{seq * 4}\n")
                fq.write_text(f"@r1\n{seq}\n+\n{'I' * len(seq)}\n")
                cmd = [exe, "-t", "1", str(fa), str(fq)] if cand == "strobealign" else [exe, "-t", "1", "-ax", "sr", str(fa), str(fq)]
                rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode
                return rc == 0
        args = ["version"] if cand == "bwa-mem2" else ([] if cand == "bwa" else ["--version"])
        rc = subprocess.run([exe, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode
        return rc >= 0 and rc not in (126, 127, 132, 134, 139)
    except Exception:
        return False


def _select_mapper(requested: str) -> str:
    if requested and requested != "auto":
        if _mapper_runs_cleanly(requested):
            return requested
        _log(f"Requested mapper {requested!r} failed runtime probe; falling back automatically")
    for cand in ("strobealign", "minimap2", "bwa-mem2", "bwa"):
        if _mapper_runs_cleanly(cand):
            return cand
    raise RuntimeError("No supported short-read mapper (strobealign/minimap2/bwa-mem2/bwa) found in PATH")


def _build_local_reads_map(reads: Sequence[str] | None, read_ids: Sequence[str] | None) -> dict[str, list[Path]]:
    local_map: dict[str, list[Path]] = {}
    if not reads:
        return local_map
    ids = list(read_ids) if read_ids and len(read_ids) == len(reads) else [Path(r).name for r in reads]
    for rid, rpath in zip(ids, reads):
        p = Path(rpath).resolve()
        stem = _READ_EXT_RE.sub("", rid)
        clean_id = _MATE_SUFFIX_RE.sub("", stem)
        for key in (rid, stem, clean_id, p.name, _MATE_SUFFIX_RE.sub("", _READ_EXT_RE.sub("", p.name))):
            if key:
                local_map.setdefault(key, [])
                if p not in local_map[key]:
                    local_map[key].append(p)
    return local_map


def _parse_bed_regions(bed_path: str | Path | None, contig_lengths: dict[str, int] | None = None) -> list[tuple[str, int, int]]:
    if not bed_path:
        return []
    regs: list[tuple[str, int, int]] = []
    with open(bed_path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            parts = ln.rstrip("\n").split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            s0 = max(0, int(parts[1]))
            e0 = int(parts[2])
            if contig_lengths is not None:
                if chrom not in contig_lengths:
                    continue
                e0 = min(contig_lengths[chrom], e0)
            if e0 > s0:
                regs.append((chrom, s0, e0))
    return regs


def _pf3d7_assets_ready(ref_fasta: str | Path | None) -> bool:
    from .malariavar import genome as G
    from .malariavar import prepare as P

    has_pf_ref = (G.REFDIR / "genome_cache.npz").exists() or (G.GFF.exists() and G.REGIONS.exists())
    if not (has_pf_ref and P.PANEL.exists()):
        return False
    if not ref_fasta:
        return True
    # Check whether provided FASTA matches Pf3D7 contig names
    try:
        names, _ = G.read_fasta(ref_fasta)
        return any(n.startswith("Pf3D7_") for n in names)
    except Exception:
        return False


def _materialize_vcf_output(src_vcf_gz: Path, out_vcf: Path) -> None:
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    if str(out_vcf).endswith(".gz"):
        shutil.copy2(src_vcf_gz, out_vcf)
        tbi = Path(str(src_vcf_gz) + ".tbi")
        if tbi.exists():
            shutil.copy2(tbi, Path(str(out_vcf) + ".tbi"))
        return
    with gzip.open(src_vcf_gz, "rb") as fin, open(out_vcf, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def _vcf_to_dataframe(vcf_path: Path, sample_override: str | None = None) -> pd.DataFrame:
    cols = ["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "gq", "dp", "ad_ref", "ad_alt", "qual", "source"]
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    samples: list[str] = []
    with opener(vcf_path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("##"):
                continue
            if ln.startswith("#CHROM"):
                hdr = ln.rstrip("\n").split("\t")
                samples = hdr[9:] if len(hdr) > 9 else [sample_override or "SAMPLE"]
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos_s, _, ref, alt, qual_s = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            alts = alt.split(",")
            kind = "SNV" if (len(ref) == 1 and all(len(a) == 1 for a in alts)) else "INDEL"
            fmt_keys = parts[8].split(":") if len(parts) > 8 else []
            for s_idx, s_name in enumerate(samples or [sample_override or "SAMPLE"]):
                vals = parts[9 + s_idx].split(":") if len(parts) > 9 + s_idx else []
                fmap = dict(zip(fmt_keys, vals))
                gt = fmap.get("GT", "./.")
                if gt in ("0/0", "./."):
                    continue
                gq = int(fmap["GQ"]) if fmap.get("GQ", ".").isdigit() else np.nan
                dp = int(fmap["DP"]) if fmap.get("DP", ".").isdigit() else np.nan
                ad_parts = fmap.get("AD", "").split(",")
                ad_ref = int(ad_parts[0]) if len(ad_parts) >= 1 and ad_parts[0].isdigit() else np.nan
                ad_alt = ",".join(ad_parts[1:]) if len(ad_parts) >= 2 else ""
                rows.append(
                    {
                        "sample_id": sample_override or s_name,
                        "chrom": chrom,
                        "pos": int(pos_s),
                        "ref": ref,
                        "alt": alt,
                        "kind": kind,
                        "gt": gt,
                        "gq": gq,
                        "dp": dp,
                        "ad_ref": ad_ref,
                        "ad_alt": ad_alt,
                        "qual": float(qual_s) if qual_s not in (".", "") else np.nan,
                        "source": "variationloop",
                    }
                )
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _write_pathogen_vcf(
    g: Any,
    calls_df: pd.DataFrame,
    out_vcf: Path,
) -> int:
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    samples = sorted(calls_df["sample_id"].astype(str).unique().tolist()) if not calls_df.empty else ["SAMPLE"]
    lines: list[str] = [
        "##fileformat=VCFv4.2",
        "##source=Focaloop_VariationLoop_v0.3.0",
        '##FILTER=<ID=PASS,Description="All filters passed">',
    ]
    for cid in range(len(g.names)):
        lines.append(f"##contig=<ID={g.names[cid]},length={int(g.lengths[cid])}>")
    lines.extend(
        [
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">',
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
            '##FORMAT=<ID=WSAF,Number=1,Type=Float,Description="Within-sample allele frequency">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
        ]
    )
    n_written = 0
    if not calls_df.empty:
        grouped = calls_df.groupby(["chrom", "pos", "ref", "alt"], sort=True)
        for (chrom, pos, ref, alt), sub in grouped:
            by_s = sub.set_index("sample_id")
            quals = pd.to_numeric(sub.get("qual"), errors="coerce").dropna()
            q_val = float(quals.max()) if not quals.empty else 30.0
            s_cells = []
            for s in samples:
                if s in by_s.index:
                    row = by_s.loc[s]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    gt_i = int(row.get("gt", 1))
                    gt_str = "0/1" if gt_i == 1 else ("1/1" if gt_i >= 2 else "0/0")
                    dp_i = int(row.get("dp", 0) or 0)
                    ar = int(row.get("ad_ref", 0) or 0)
                    aa = int(row.get("ad_alt", 0) or 0)
                    ws = float(row.get("wsaf", 0.0) or 0.0)
                    s_cells.append(f"{gt_str}:{dp_i}:{ar},{aa}:{ws:.3f}")
                else:
                    s_cells.append("./.:0:0,0:0.000")
            lines.append(f"{chrom}\t{int(pos)}\t.\t{ref}\t{alt}\t{q_val:.1f}\tPASS\t.\tGT:DP:AD:WSAF\t" + "\t".join(s_cells))
            n_written += 1

    payload = "\n".join(lines) + "\n"
    if str(out_vcf).endswith(".gz"):
        with gzip.open(out_vcf, "wt", encoding="utf-8") as gz:
            gz.write(payload)
    else:
        out_vcf.write_text(payload, encoding="utf-8")
    return n_written


def _format_malariavar_drm_table(
    tel_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    from .malariavar import who_resistance as WR

    cnv_piv = (
        cnv_df.pivot_table(index="sample", columns="locus", values="call", aggfunc="first")
        if not cnv_df.empty
        else pd.DataFrame()
    )
    meta_by_s = meta_df.set_index("Sample").to_dict(orient="index")
    out_rows = []

    def _cnv_str(val: Any, is_del: bool = False) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "undetermined"
        iv = int(float(val))
        if iv < 0:
            return "undetermined"
        if is_del:
            return "deleted" if iv == 1 else "intact"
        return "amplified" if iv == 1 else "single_copy"

    for _, r in tel_df.iterrows():
        s = str(r["sample"])
        m = meta_by_s.get(s, {})
        c_row = cnv_piv.loc[s] if (not cnv_piv.empty and s in cnv_piv.index) else {}
        mdr1_call = c_row.get("mdr1", np.nan) if hasattr(c_row, "get") else np.nan
        pm23_call = c_row.get("pm2_pm3", np.nan) if hasattr(c_row, "get") else np.nan
        gch1_call = c_row.get("gch1", np.nan) if hasattr(c_row, "get") else np.nan
        crt_call = c_row.get("crt", np.nan) if hasattr(c_row, "get") else np.nan
        hrp2_call = c_row.get("hrp2", np.nan) if hasattr(c_row, "get") else np.nan
        hrp3_call = c_row.get("hrp3", np.nan) if hasattr(c_row, "get") else np.nan

        k13_raw = r.get("k13_ns")
        k13_str = "WT" if (k13_raw is None or str(k13_raw).strip() in ("", "none", "None", "nan")) else str(k13_raw)
        who = WR.classify_sample(
            {
                "crt_76": r.get("aa_crt_76"),
                "dhfr_51": r.get("aa_dhfr_51"),
                "dhfr_59": r.get("aa_dhfr_59"),
                "dhfr_108": r.get("aa_dhfr_108"),
                "dhfr_164": r.get("aa_dhfr_164"),
                "dhps_437": r.get("aa_dhps_437"),
                "dhps_540": r.get("aa_dhps_540"),
                "dhps_581": r.get("aa_dhps_581"),
                "dhps_613": r.get("aa_dhps_613"),
                "k13_ns": r.get("k13_ns"),
                "mdr1_amp": mdr1_call,
                "pm2_pm3_amp": pm23_call,
            }
        )
        wire_mb = float(r.get("wire_bytes") or 0.0) / 1e6
        arch_b = r.get("archive_bytes")
        arch_mb = (float(arch_b) / 1e6) if (arch_b is not None and pd.notna(arch_b) and float(arch_b) > 0) else np.nan
        pct_arch = (100.0 * wire_mb / arch_mb) if (pd.notna(arch_mb) and arch_mb > 0) else np.nan
        hfrac = r.get("host_frac")
        host_pct = (100.0 * float(hfrac)) if (hfrac is not None and pd.notna(hfrac)) else 0.0
        fws_val = float(r.get("fws")) if (r.get("fws") is not None and pd.notna(r.get("fws"))) else np.nan
        clonality = "monoclonal" if (np.isfinite(fws_val) and fws_val >= 0.95) else ("polyclonal" if np.isfinite(fws_val) else "unknown")

        dhfr_hap = "".join(str(r.get(f"aa_dhfr_{c}") or "-") for c in (51, 59, 108, 164))
        dhps_hap = "".join(str(r.get(f"aa_dhps_{c}") or "-") for c in (436, 437, 540, 581, 613))
        src_mode = "focused_cram" if str(r.get("source", "")).startswith("sanger_crai") else str(r.get("source", "wgs_stream"))

        out_rows.append(
            {
                "sample_id": s,
                "country_or_group": m.get("Country", ""),
                "population": m.get("Population", m.get("Country", "")),
                "mode": src_mode,
                "stop_reason": str(r.get("stop_reason", "")),
                "wire_mb": round(wire_mb, 3),
                "archive_mb": round(arch_mb, 2) if np.isfinite(arch_mb) else np.nan,
                "pct_archive": round(pct_arch, 3) if np.isfinite(pct_arch) else np.nan,
                "host_depletion_reject_pct": round(host_pct, 2),
                "median_depth": round(float(r.get("median_core_depth") or 0.0), 2),
                "fws": round(fws_val, 4) if np.isfinite(fws_val) else np.nan,
                "clonality": clonality,
                "kelch13_mutations": k13_str,
                "crt_76": str(r.get("aa_crt_76") or "-"),
                "mdr1_86": str(r.get("aa_mdr1_86") or "-"),
                "mdr1_184": str(r.get("aa_mdr1_184") or "-"),
                "dhfr_hap": dhfr_hap,
                "dhps_hap": dhps_hap,
                "hrp2_deletion": _cnv_str(hrp2_call, is_del=True),
                "hrp3_deletion": _cnv_str(hrp3_call, is_del=True),
                "mdr1_cnv": _cnv_str(mdr1_call),
                "pm2_pm3_cnv": _cnv_str(pm23_call),
                "gch1_cnv": _cnv_str(gch1_call),
                "crt_cnv": _cnv_str(crt_call),
                "who_artemisinin": who["Artemisinin"],
                "who_chloroquine": who["Chloroquine"],
                "who_piperaquine": who["Piperaquine"],
                "who_mefloquine": who["Mefloquine"],
                "who_pyrimethamine": who["Pyrimethamine"],
                "who_sulfadoxine": who["Sulfadoxine"],
                "who_sp_uncomplicated": who["SP (uncomplicated)"],
                "who_sp_iptp": who["SP (IPTp)"],
                "who_as_mq": who["AS-MQ"],
                "who_dha_ppq": who["DHA-PPQ"],
            }
        )
    return pd.DataFrame(out_rows)


def render_variationloop_report(
    out_path: str | Path,
    *,
    analysis_mode: str,
    summary_stats: dict[str, Any],
    variants_df: pd.DataFrame,
    drm_df: pd.DataFrame | None = None,
    sample_telemetry: list[dict[str, Any]] | None = None,
) -> Path:
    """Render self-contained HTML report for VariationLoop across all modes."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if drm_df is not None and not drm_df.empty:
        n_iso = len(drm_df)
        wire_mb = float(drm_df["wire_mb"].fillna(0).sum()) if "wire_mb" in drm_df.columns else 0.0
        arch_mb = float(drm_df["archive_mb"].fillna(0).sum()) if "archive_mb" in drm_df.columns else 0.0
        pct_arch = (100.0 * wire_mb / arch_mb) if arch_mb > 0 else 0.0
        n_poly = int((drm_df["clonality"] == "polyclonal").sum()) if "clonality" in drm_df.columns else 0
        n_mono = int((drm_df["clonality"] == "monoclonal").sum()) if "clonality" in drm_df.columns else 0
        med_dep = float(drm_df["host_depletion_reject_pct"].dropna().median()) if (
            "host_depletion_reject_pct" in drm_df.columns and drm_df["host_depletion_reject_pct"].notna().any()
        ) else 0.0

        prev_rows = []
        if "country_or_group" in drm_df.columns:
            for grp, sub in drm_df.groupby("country_or_group", sort=False):
                n_g = len(sub)
                art_r = int((sub.get("who_artemisinin") == "Resistant").sum())
                cq_r = int((sub.get("who_chloroquine") == "Resistant").sum())
                ppq_r = int((sub.get("who_piperaquine") == "Resistant").sum())
                sp_r = int((sub.get("who_sp_uncomplicated") == "Resistant").sum())
                hrp2_d = int((sub.get("hrp2_deletion") == "deleted").sum())
                hrp3_d = int((sub.get("hrp3_deletion") == "deleted").sum())
                poly_g = int((sub.get("clonality") == "polyclonal").sum())
                prev_rows.append(
                    f"<tr><td><strong>{_esc(grp)}</strong></td><td class='num'>{n_g}</td>"
                    f"<td class='num'>{art_r}/{n_g} ({100*art_r/max(n_g,1):.0f}%)</td>"
                    f"<td class='num'>{cq_r}/{n_g} ({100*cq_r/max(n_g,1):.0f}%)</td>"
                    f"<td class='num'>{ppq_r}/{n_g} ({100*ppq_r/max(n_g,1):.0f}%)</td>"
                    f"<td class='num'>{sp_r}/{n_g} ({100*sp_r/max(n_g,1):.0f}%)</td>"
                    f"<td class='num'>{hrp2_d}/{n_g}</td>"
                    f"<td class='num'>{hrp3_d}/{n_g}</td>"
                    f"<td class='num'>{poly_g}/{n_g}</td></tr>"
                )

        iso_rows = []
        for _, r in drm_df.iterrows():
            pct = float(r.get("pct_archive") or 0.0)
            bar_w = min(max(pct, 1.0), 100.0) if np.isfinite(pct) else 0.0
            art = str(r.get("who_artemisinin", "Undetermined"))
            art_cls = "b-rose" if art == "Resistant" else ("b-green" if art == "Sensitive" else "b-amber")
            clon = str(r.get("clonality", "unknown"))
            clon_cls = "b-amber" if clon == "polyclonal" else "b-teal"
            iso_rows.append(
                f"<tr><td><code>{_esc(r.get('sample_id'))}</code></td>"
                f"<td>{_esc(r.get('country_or_group'))}</td>"
                f"<td><code>{_esc(r.get('mode'))}</code></td>"
                f"<td class='num'>{_fmt_num(r.get('wire_mb'), 1)} / {_fmt_num(r.get('archive_mb'), 0)} MB "
                f"<span class='bar-bg'><span class='bar-fg' style='width:{bar_w:.1f}%'></span></span></td>"
                f"<td class='num'>{_fmt_num(r.get('host_depletion_reject_pct'), 1)}%</td>"
                f"<td class='num'>{_fmt_num(r.get('median_depth'), 1)}x</td>"
                f"<td class='num'>{_fmt_num(r.get('fws'), 3)} <span class='badge {clon_cls}'>{_esc(clon)}</span></td>"
                f"<td><code>{_esc(r.get('kelch13_mutations'))}</code> <span class='badge {art_cls}'>{_esc(art)}</span></td>"
                f"<td><code>{_esc(r.get('crt_76'))}</code></td>"
                f"<td><code>{_esc(r.get('dhfr_hap'))}</code> / <code>{_esc(r.get('dhps_hap'))}</code></td>"
                f"<td>mdr1:{_esc(r.get('mdr1_cnv'))} pm2/3:{_esc(r.get('pm2_pm3_cnv'))} gch1:{_esc(r.get('gch1_cnv'))}</td>"
                f"<td>hrp2:{_esc(r.get('hrp2_deletion'))} hrp3:{_esc(r.get('hrp3_deletion'))}</td></tr>"
            )

        body_sections = f"""
  <div class="kpis">
    <div class="kpi"><div class="label">Isolates Genotyped</div><div class="val">{n_iso}</div><div class="sub">{n_mono} monoclonal · {n_poly} polyclonal (F_WS &lt; 0.95)</div></div>
    <div class="kpi"><div class="label">Bytes Transferred</div><div class="val">{wire_mb:.1f} MB</div><div class="sub">{pct_arch:.2f}% of {arch_mb / 1000.0:.2f} GB full archives</div></div>
    <div class="kpi"><div class="label">Host Read Depletion</div><div class="val">{med_dep:.1f}%</div><div class="sub">Median non-target spots filtered on stream</div></div>
    <div class="kpi"><div class="label">Artemisinin Resistant</div><div class="val">{int((drm_df.get('who_artemisinin') == 'Resistant').sum())}/{n_iso}</div><div class="sub">WHO validated/candidate kelch13 349–726</div></div>
    <div class="kpi"><div class="label">Variant Calls</div><div class="val">{len(variants_df):,}</div><div class="sub">hrp2/3 del: {int((drm_df.get('hrp2_deletion') == 'deleted').sum())} / {int((drm_df.get('hrp3_deletion') == 'deleted').sum())}</div></div>
  </div>
  <div class="section">
    <h2>Cohort WHO Resistance &amp; Diagnostic-Escape Prevalence by Group</h2>
    <table>
      <thead><tr><th>Country / Group</th><th class="num">N</th><th class="num">Artemisinin Res</th><th class="num">Chloroquine Res</th><th class="num">Piperaquine Res</th><th class="num">SP (uncomp) Res</th><th class="num">hrp2 Del</th><th class="num">hrp3 Del</th><th class="num">Polyclonal</th></tr></thead>
      <tbody>{''.join(prev_rows)}</tbody>
    </table>
  </div>
  <div class="section">
    <h2>Per-Isolate Streaming Telemetry, Clonality &amp; Drug-Resistance Genotypes</h2>
    <table>
      <thead><tr><th>Isolate</th><th>Group</th><th>Mode</th><th class="num">Wire / Archive</th><th class="num">Host Depletion</th><th class="num">Depth</th><th class="num">F_WS</th><th>kelch13 (WHO Art)</th><th>crt 76</th><th>dhfr / dhps</th><th>CNVs</th><th>Deletions</th></tr></thead>
      <tbody>{''.join(iso_rows)}</tbody>
    </table>
  </div>
"""
    else:
        tel_rows = sample_telemetry or []
        n_samp = len(tel_rows) if tel_rows else int(variants_df["sample_id"].nunique() if not variants_df.empty else 0)
        n_tot = len(variants_df)
        n_snv = int((variants_df["kind"] == "SNV").sum()) if ("kind" in variants_df.columns and not variants_df.empty) else 0
        n_indel = n_tot - n_snv
        n_asm = sum(int(t.get("n_omnitig_assembled", 0) or 0) for t in tel_rows)
        n_dup = sum(int(t.get("pcr_dups_skipped", 0) or 0) for t in tel_rows)
        acc_mb = sum(float(t.get("acc_total_bytes", 0) or 0) for t in tel_rows) / 1e6

        samp_html = []
        for t in tel_rows:
            samp_html.append(
                f"<tr><td><code>{_esc(t.get('sample'))}</code></td>"
                f"<td><code>{_esc(t.get('mapper', t.get('mode', analysis_mode)))}</code></td>"
                f"<td class='num'>{int(t.get('reads_counted', t.get('sam_records', 0)) or 0):,}</td>"
                f"<td class='num'>{int(t.get('pcr_dups_skipped', 0) or 0):,}</td>"
                f"<td class='num'>{int(t.get('n_snv', 0) or 0):,}</td>"
                f"<td class='num'>{int(t.get('n_indel', 0) or 0):,}</td>"
                f"<td class='num'>{int(t.get('n_omnitig_assembled', 0) or 0):,}</td>"
                f"<td class='num'>{_fmt_num(float(t.get('acc_total_bytes', 0) or 0) / 1e6, 2)} MB</td>"
                f"<td class='num'>{_fmt_num(t.get('total_wall_s', t.get('wall_s', 0.0)), 2)}s</td></tr>"
            )

        var_preview = variants_df.head(50)
        var_html = []
        for _, v in var_preview.iterrows():
            var_html.append(
                f"<tr><td><code>{_esc(v.get('sample_id'))}</code></td>"
                f"<td><code>{_esc(v.get('chrom'))}:{_esc(v.get('pos'))}</code></td>"
                f"<td><code>{_esc(v.get('ref'))}</code> &rarr; <code>{_esc(v.get('alt'))}</code></td>"
                f"<td><span class='badge b-blue'>{_esc(v.get('kind'))}</span></td>"
                f"<td><code>{_esc(v.get('gt'))}</code></td>"
                f"<td class='num'>{_fmt_num(v.get('qual'), 1)}</td>"
                f"<td class='num'>{_esc(v.get('dp'))}</td>"
                f"<td class='num'>{_esc(v.get('ad_ref'))},{_esc(v.get('ad_alt'))}</td>"
                f"<td><code>{_esc(v.get('source', 'variationloop'))}</code></td></tr>"
            )

        body_sections = f"""
  <div class="kpis">
    <div class="kpi"><div class="label">Samples Analyzed</div><div class="val">{n_samp}</div><div class="sub">Mode: <code>{_esc(analysis_mode)}</code></div></div>
    <div class="kpi"><div class="label">Total Variants</div><div class="val">{n_tot:,}</div><div class="sub">{n_snv:,} SNVs · {n_indel:,} Indels</div></div>
    <div class="kpi"><div class="label">Omnitig Rescued</div><div class="val">{n_asm:,}</div><div class="sub">Multi-k local haplotype reassembly</div></div>
    <div class="kpi"><div class="label">PCR Duplicates Filtered</div><div class="val">{n_dup:,}</div><div class="sub">62-bit exact coordinate+TLEN dedup</div></div>
    <div class="kpi"><div class="label">Paged Accumulator RAM</div><div class="val">{acc_mb:.2f} MB</div><div class="sub">64 kb lazy-paged C accumulator</div></div>
  </div>
  <div class="section">
    <h2>Per-Sample Streaming &amp; Calling Telemetry</h2>
    <table>
      <thead><tr><th>Sample</th><th>Engine / Mapper</th><th class="num">Reads Counted</th><th class="num">PCR Dups Skipped</th><th class="num">SNVs</th><th class="num">Indels</th><th class="num">Omnitig Calls</th><th class="num">Accumulator</th><th class="num">Wall Time</th></tr></thead>
      <tbody>{''.join(samp_html) or '<tr><td colspan="9">No per-sample telemetry recorded.</td></tr>'}</tbody>
    </table>
  </div>
  <div class="section">
    <h2>Variant Calls Preview ({min(len(variants_df), 50)} of {len(variants_df):,})</h2>
    <table>
      <thead><tr><th>Sample</th><th>Locus</th><th>Alleles</th><th>Type</th><th>GT</th><th class="num">QUAL</th><th class="num">DP</th><th class="num">AD</th><th>Source</th></tr></thead>
      <tbody>{''.join(var_html) or '<tr><td colspan="9">No variants called above quality thresholds.</td></tr>'}</tbody>
    </table>
  </div>
"""

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Focaloop VariationLoop Report</title>{_BASE_CSS}</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Focaloop VariationLoop — Streaming Variation &amp; Genotyping Report</h1>
    <p>Analysis Mode: <code>{_esc(analysis_mode)}</code> &nbsp;·&nbsp; Reference: <code>{_esc(summary_stats.get('reference_label', 'Custom'))}</code> &nbsp;·&nbsp; Wall-clock: <strong>{float(summary_stats.get('wall_s', 0.0)):.2f}s</strong></p>
  </div>
  {body_sections}
</div>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out


def _resolve_fastq_pair_for_row(
    row: dict[str, str],
    local_map: dict[str, list[Path]],
    work_dir: Path,
    sample_id: str,
) -> tuple[str, str | None, str]:
    """Resolve a manifest row's source into (r1_path_or_url, r2_path_or_url, kind)."""
    src = _pick_col(row, ("source_uri_or_srr", "run_or_url", "fastq_1", "accession", "source", "url"))
    mate2 = _pick_col(row, ("mate2_url", "fastq_2", "r2", "mate2"))
    cand_local = local_map.get(src) or local_map.get(sample_id)
    if cand_local:
        ordered = sorted(cand_local, key=lambda p: p.name)[:2]
        return str(ordered[0]), (str(ordered[1]) if len(ordered) > 1 else None), "local_fastq"
    if src and Path(src).exists():
        p1 = str(Path(src).resolve())
        p2 = str(Path(mate2).resolve()) if (mate2 and Path(mate2).exists()) else None
        return p1, p2, "local_fastq"
    if _INSDC_RUN_RE.match(src):
        info = _resolve_ena_or_odp(src)
        fq_raw = info.get("fastq_urls") or []
        if fq_raw:
            urls = [u.strip() for u in fq_raw[0].split(";") if u.strip()]
            urls = [("https://" + u[6:] if u.startswith("ftp://") else (u if u.startswith("http") else "https://" + u)) for u in urls]
            if len(urls) >= 2:
                return urls[0], urls[1], "remote_fastq"
            return urls[0], None, "remote_fastq"
        return src.upper(), None, "insdc_odp"
    if src.startswith(("http://", "https://", "ftp://")):
        parts = [u.strip() for u in src.split(";") if u.strip()]
        if mate2:
            parts.append(mate2.strip())
        parts = [("https://" + u[6:] if u.startswith("ftp://") else u) for u in parts]
        return parts[0], (parts[1] if len(parts) > 1 else None), "remote_fastq"
    raise ValueError(f"Sample {sample_id!r}: could not resolve FASTQ source {src!r}")


def _materialize_remote_input(uri_or_acc: str, dst: Path, kind: str, threads: int = 4) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if kind == "insdc_odp":
        if not shutil.which("xsra"):
            raise RuntimeError(f"INSDC accession {uri_or_acc!r} requires 'xsra' in PATH when ENA FASTQ URLs are unavailable")
        odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{uri_or_acc}/{uri_or_acc}"
        out_fq = dst.with_suffix(".interleaved.fq")
        with open(out_fq, "wb") as fh:
            subprocess.run(["xsra", "dump", "-Q", "-P", odp_url, "-t", str(max(1, threads))], stdout=fh, check=True)
        return out_fq
    s = _ipv4_session()
    with s.get(uri_or_acc, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return dst


def _run_pf3d7_surveillance(
    *,
    mode_req: str,
    rows_raw: list[dict[str, str]],
    local_map: dict[str, list[Path]],
    work_dir: Path,
    state_root: Path,
    out_variants: Path,
    out_vcf: Path | None,
    out_drm: Path | None,
    out_report: Path,
    ref_fasta: str | Path | None,
    targets_bed: str | Path | None,
    target_depth: float,
    mapper: str,
    threads: int,
    streams: int,
    sample_timeout_s: float,
    use_host_depletion: bool,
    state_archive: str | Path | None,
    t_start: float,
) -> dict[str, Any]:
    os.environ["MALARIAVAR_SCRATCH"] = str(work_dir / "scratch")
    os.environ["MALARIAVAR_STAGE_DIR"] = str(work_dir / "stage")
    os.environ["MALARIAVAR_LOGDIR"] = str(work_dir / "logs")

    from .malariavar import engine as E
    from .malariavar import focused_cohort as FC
    from .malariavar import genome as G
    from .malariavar import prepare as P
    from .malariavar import state_store as SS
    from .malariavar import streamer as ST

    E.LOGDIR = work_dir / "logs"
    E.STAGE = work_dir / "stage_stream"
    FC.STAGE = work_dir / "stage_focused"
    for d in (E.LOGDIR, E.STAGE, FC.STAGE):
        d.mkdir(parents=True, exist_ok=True)

    cohort_dir = state_root / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    if ref_fasta:
        user_fa = _gunzip_if_needed(ref_fasta, work_dir / "ref", "Pf3D7_user.fa")
        G.REF_FASTA = user_fa
        _ensure_faidx(user_fa)

    g = G.load()
    meta_rows: list[dict[str, Any]] = []
    focused_jobs: list[dict[str, Any]] = []
    stream_jobs: list[dict[str, Any]] = []

    for idx_r, row in enumerate(rows_raw):
        sid = _pick_col(row, ("sample_id", "Sample", "sample"), f"isolate_{idx_r + 1}")
        grp = _pick_col(row, ("country_or_group", "Country", "group", "Population"), "cohort")
        pop = _pick_col(row, ("population", "Population"), grp)
        src = _pick_col(row, ("source_uri_or_srr", "cram_url", "accession", "ena_runs", "source", "url"))
        crai_u = _pick_col(row, ("crai_uri", "crai_uri_optional", "crai_url", "crai"))
        stype = _pick_col(row, ("sample_type", "Sample_type"), "gDNA")
        year_s = _pick_col(row, ("year", "Year"), "2020")
        year = int(year_s) if year_s.isdigit() else 2020

        if not src and sid in local_map:
            src = sid
        if not src:
            raise ValueError(f"Isolate {sid!r}: missing source_uri_or_srr")

        row_mode = mode_req
        if row_mode == "auto":
            row_mode = "focused_cram" if src.split("?", 1)[0].endswith(".cram") else "wgs_stream"

        meta_rows.append({"Sample": sid, "Country": grp, "Population": pop, "Year": year, "Sample_type": stype})
        if row_mode == "focused_cram":
            cram_bytes_s = _pick_col(row, ("cram_bytes", "archive_bytes"))
            cram_bytes = int(float(cram_bytes_s)) if cram_bytes_s else None
            focused_jobs.append(
                {
                    "Sample": sid,
                    "cram_url": src,
                    "crai_uri": crai_u or (src + ".crai"),
                    "cram_bytes": cram_bytes,
                    "Sample_type": stype,
                }
            )
        else:
            cand_local = local_map.get(src) or local_map.get(sid)
            if cand_local:
                ordered_l = sorted(cand_local, key=lambda p: p.name)[:2]
                staged_l = [
                    _stage_local_fastq(p, work_dir / "fq_links", f"{sid}_R{i + 1}")
                    for i, p in enumerate(ordered_l)
                ]
                urls = ";".join(str(p) for p in staged_l)
                stream_jobs.append(
                    {
                        "Sample": sid,
                        "Sample_type": stype,
                        "ena_runs": [],
                        "all_odp": False,
                        "all_paired_fastq": len(staged_l) >= 2,
                        "fastq_urls": [urls],
                        "fastq_bytes": np.nan,
                        "order_hint": "flowcell",
                    }
                )
            else:
                accs = [a.strip() for a in re.split(r"[;,]", src) if a.strip()]
                if all(_INSDC_RUN_RE.match(a) for a in accs):
                    info = _resolve_ena_or_odp(accs[0])
                    if len(accs) > 1:
                        info["ena_runs"] = [a.upper() for a in accs]
                    stream_jobs.append({"Sample": sid, "Sample_type": stype, **info})
                else:
                    mate2 = _pick_col(row, ("mate2_url", "fastq_2", "r2", "mate2"))
                    fq_src = f"{src};{mate2}" if mate2 else src
                    stream_jobs.append(
                        {
                            "Sample": sid,
                            "Sample_type": stype,
                            "ena_runs": [],
                            "all_odp": False,
                            "all_paired_fastq": ";" in fq_src,
                            "fastq_urls": [fq_src],
                            "fastq_bytes": np.nan,
                            "order_hint": "flowcell",
                        }
                    )

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(cohort_dir / "manifest.parquet", index=False)

    if focused_jobs:
        _log(f"Running Mode A (focused_cram) on {len(focused_jobs)} isolates...")
        regs = FC.regions(g)
        if targets_bed:
            cidx = {n: i for i, n in enumerate(g.names)}
            for cn, s0, e0 in _parse_bed_regions(targets_bed, dict(zip(g.names, [int(x) for x in g.lengths]))):
                regs.append((cidx[cn], s0 + 1, e0, f"bed:{cn}:{s0}"))
        FC._init_worker(use_omnitig_zoom=False)
        crai_cache = work_dir / "crai"
        dl_pool = ThreadPoolExecutor(max_workers=streams, thread_name_prefix="crai-range")
        try:
            for idx_j, fj in enumerate(focused_jobs, 1):
                sid = fj["Sample"]
                url = fj["cram_url"]
                crai_local = _fetch_crai_file(fj["crai_uri"], crai_cache / f"{sid}.cram.crai")
                cram_bytes = fj["cram_bytes"]
                if not cram_bytes:
                    cram_bytes = Path(url).stat().st_size if Path(url).exists() else _head_content_length(url)
                    fj["cram_bytes"] = cram_bytes
                hdr_end, rr, nc, spans = FC.plan_ranges(crai_local, regs)
                staged_cram = FC.STAGE / f"{sid}.cram"
                if Path(url).exists():
                    with open(url, "rb") as fi, open(staged_cram, "wb") as fo:
                        fo.write(fi.read(hdr_end))
                        for a, b in rr:
                            fi.seek(a)
                            fo.write(fi.read(b - a))
                        fo.write(ST.CRAM_EOF)
                    fetch_bytes = staged_cram.stat().st_size
                    fetch_s = 0.01
                else:
                    done_ev = threading.Event()
                    fjob = FC.FetchJob(sid, url, hdr_end, rr, staged_cram, lambda j, ev=done_ev: ev.set())
                    fjob.submit(dl_pool)
                    done_ev.wait(timeout=sample_timeout_s)
                    if not done_ev.is_set():
                        raise TimeoutError(f"Timed out fetching focused CRAM ranges for {sid}")
                    if fjob.failed is not None:
                        raise RuntimeError(f"Focused CRAM range fetch failed for {sid}: {fjob.failed}")
                    fetch_bytes = fjob.total + len(ST.CRAM_EOF)
                    fetch_s = (fjob.t1 or time.time()) - fjob.t0
                info = {"bytes": fetch_bytes, "n_ranges": len(rr) + 1, "n_containers": nc, "fetch_s": fetch_s}
                tel = FC.process_one((sid, str(staged_cram), fj, str(cohort_dir), spans.tolist(), info))
                if tel.get("stop_reason") == "crash" or tel.get("error"):
                    raise RuntimeError(f"Focused CRAM processing failed for {sid}: {tel.get('error')}")
                _log(f"  [{idx_j}/{len(focused_jobs)}] {sid}: {fetch_bytes / 1e6:.2f} MB ({100.0 * fetch_bytes / max(cram_bytes, 1):.2f}%)")
        finally:
            dl_pool.shutdown(wait=False, cancel_futures=True)

    if stream_jobs:
        chosen_mapper = _select_mapper(mapper)
        use_pla = P.PLA_SIEVE.exists()
        if not use_pla and not P.KSET.exists():
            _log("Building Pf3D7 PLA host-depletion index...")
            P.build_pla_sieve(out_path=work_dir / "pf_pla_sieve.pla", num_threads=min(threads, 8))
            pla_path = str(work_dir / "pf_pla_sieve.pla")
            use_pla = True
        else:
            pla_path = str(P.PLA_SIEVE) if use_pla else None

        if chosen_mapper in ("bwa-mem2", "bwa"):
            ext_check = ".bwt.2bit.64" if chosen_mapper == "bwa-mem2" else ".bwt"
            if not Path(str(G.REF_FASTA) + ext_check).exists():
                idx_fa = work_dir / "mapper_ref" / G.REF_FASTA.name
                idx_fa.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(G.REF_FASTA, idx_fa)
                subprocess.run([chosen_mapper, "index", str(idx_fa)], check=True)
                G.REF_FASTA = idx_fa

        _log(f"Running Mode B (wgs_stream) on {len(stream_jobs)} isolates with mapper={chosen_mapper}, depletion={use_host_depletion}...")
        cfg = {
            "map_threads": max(2, min(threads, 6)),
            "xsra_threads": max(2, min(threads // 2, 4)),
            "prefer": "odp_stream",
            "store_candidates": False,
            "mapper": chosen_mapper,
            "min_depth": float(target_depth),
            "min_depth_poly": float(max(target_depth + 8.0, target_depth * 1.25)),
            "use_pla_sieve": use_pla,
            "pla_sieve_path": pla_path,
            "depletion_need": 2 if use_host_depletion else 0,
            "use_omnitig_zoom": True,
            "sample_timeout_s": float(sample_timeout_s),
        }
        for idx_s, srow in enumerate(stream_jobs, 1):
            sid = srow["Sample"]
            tel = E.run_sample(srow, str(cohort_dir), cfg)
            if tel.get("stop_reason") == "error":
                raise RuntimeError(f"WGS streaming failed for {sid}: {tel.get('error')}")
            _log(f"  [{idx_s}/{len(stream_jobs)}] {sid}: depth={float(tel.get('median_core_depth', 0)):.1f}x")

    all_samples = [r["Sample"] for r in meta_rows]
    SS.seal_shard(cohort_dir, 0, all_samples)
    warm_info = SS.build_warm(cohort_dir, meta=meta_df, use_dothash_knn=True, use_esscolor_panel=True, n_threads=min(threads, 4))

    tel_df = SS.read_table(cohort_dir, "sample_telemetry")
    cnv_df = SS.read_table(cohort_dir, "cnv_del_calls")
    drm_df = _format_malariavar_drm_table(tel_df, cnv_df, meta_df)
    if out_drm:
        out_drm.parent.mkdir(parents=True, exist_ok=True)
        drm_df.to_csv(out_drm, sep="\t", index=False, float_format="%.6g")

    calls_df = SS.read_table(cohort_dir, "variant_calls")
    if not calls_df.empty:
        calls_df = calls_df[calls_df["gt"] > 0].copy()
        contig_names = list(g.names)
        calls_df["chrom"] = [contig_names[int(c)] if 0 <= int(c) < len(contig_names) else str(c) for c in calls_df["chrom"]]
        calls_df = calls_df.rename(columns={"sample": "sample_id"})
        cols_v = ["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "ad_ref", "ad_alt", "dp", "wsaf", "qual", "core"]
        calls_df = calls_df[[c for c in cols_v if c in calls_df.columns]].sort_values(["sample_id", "chrom", "pos"])
    else:
        calls_df = pd.DataFrame(columns=["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "ad_ref", "ad_alt", "dp", "wsaf", "qual", "core"])
    out_variants.parent.mkdir(parents=True, exist_ok=True)
    calls_df.to_csv(out_variants, sep="\t", index=False, float_format="%.6g")
    if out_vcf:
        _write_pathogen_vcf(g, calls_df, out_vcf)

    eff_mode = "focused_cram+wgs_stream" if (focused_jobs and stream_jobs) else ("focused_cram" if focused_jobs else "wgs_stream")
    summary_stats = {"mode": eff_mode, "reference_label": "Pf3D7 (PlasmoDB-54)", "n_isolates": len(drm_df), "wall_s": round(time.time() - t_start, 2), "warm": warm_info}
    render_variationloop_report(out_report, analysis_mode=eff_mode, summary_stats=summary_stats, variants_df=calls_df, drm_df=drm_df)

    store = FocaloopStateStore(state_root)
    HypothesisLedger.from_store(store, alpha_budget=0.05)
    drm_df.to_parquet(state_root / "drm_calls.parquet", index=False)
    state_meta = {
        "engine": "malariavar",
        "version": "0.3.0",
        "blueprint_id": "malariavar",
        "mode": eff_mode,
        "summary": summary_stats,
        "samples": [
            {
                "sample": r["Sample"],
                "condition": r["Country"],
                "country_or_group": r["Country"],
                "Country": r["Country"],
                "Population": r["Population"],
            }
            for r in meta_rows
        ],
    }
    write_json(state_root / STATE_META, state_meta)
    if state_archive:
        pack_state(state_root, state_archive)
    return state_meta


def run_variationloop_galaxy(
    *,
    analysis_mode: str = "focused_cram",
    out_variants: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    out_vcf: str | Path | None = None,
    out_drm: str | Path | None = None,
    manifest_path: str | Path | None = None,
    ref_fasta: str | Path | None = None,
    targets_bed: str | Path | None = None,
    cram_pad_bp: int = 1000,
    reads: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
    reads_r1: str | Path | None = None,
    reads_r2: str | Path | None = None,
    bam_path: str | Path | None = None,
    crai_path: str | Path | None = None,
    sample_name: str = "SAMPLE",
    target_depth: float = 24.0,
    subsample_fraction: float = 1.0,
    mapper: str = "auto",
    threads: int = 8,
    streams: int = 24,
    sample_timeout_s: float = 420.0,
    use_host_depletion: bool = True,
    enable_reassembly: bool = True,
    min_mapq: int = 20,
    min_bq: int = 13,
    snv_min_qual: float = 20.0,
    indel_min_qual: float = 15.0,
    dedup_bits: int | None = None,
    reference_label: str | None = None,
    surveillance_panel: str = "none",
) -> dict[str, Any]:
    """Execute Focaloop VariationLoop across diploid WGS/targeted, CRAM range slicing, or target-filtered FASTQ calling."""
    if analysis_mode not in ("focused_cram", "wgs_stream", "diploid_wgs"):
        raise ValueError(f"Unsupported analysis_mode {analysis_mode!r}")

    t_start = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix="focaloop_varloop_"))
    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    out_variants_p = Path(out_variants)
    out_report_p = Path(out_report)
    out_vcf_p = Path(out_vcf) if out_vcf else None
    out_drm_p = Path(out_drm) if out_drm else None

    local_map = _build_local_reads_map(reads, read_ids)
    if reads_r1:
        r1_p = Path(reads_r1).resolve()
        pair = [r1_p] + ([Path(reads_r2).resolve()] if reads_r2 else [])
        local_map[sample_name] = pair

    try:
        # Specialized P. falciparum surveillance branch only when explicitly opted in (or legacy call with no ref_fasta)
        use_pf3d7 = (
            analysis_mode in ("focused_cram", "wgs_stream")
            and (surveillance_panel == "pf3d7" or ref_fasta is None)
            and _pf3d7_assets_ready(ref_fasta)
        )
        if use_pf3d7:
            if manifest_path:
                rows_raw = _read_manifest_table(manifest_path)
            elif bam_path and analysis_mode == "focused_cram":
                rows_raw = [
                    {
                        "sample_id": sample_name,
                        "country_or_group": "cohort",
                        "source_uri_or_srr": str(Path(bam_path).resolve()),
                        "crai_uri": str(Path(crai_path).resolve()) if crai_path else "",
                    }
                ]
            elif reads_r1 or (reads and len(reads) >= 1):
                pair_paths = [Path(reads_r1).resolve()] + ([Path(reads_r2).resolve()] if reads_r2 else []) if reads_r1 else [Path(p).resolve() for p in reads[:2]]
                local_map[sample_name] = pair_paths
                rows_raw = [{"sample_id": sample_name, "country_or_group": "cohort", "source_uri_or_srr": sample_name}]
            else:
                raise ValueError(f"Mode {analysis_mode!r} requires a sample manifest or History reads/CRAM input")

            return _run_pf3d7_surveillance(
                mode_req=analysis_mode,
                rows_raw=rows_raw,
                local_map=local_map,
                work_dir=work_dir,
                state_root=state_root,
                out_variants=out_variants_p,
                out_vcf=out_vcf_p,
                out_drm=out_drm_p,
                out_report=out_report_p,
                ref_fasta=ref_fasta,
                targets_bed=targets_bed,
                target_depth=target_depth,
                mapper=mapper,
                threads=threads,
                streams=streams,
                sample_timeout_s=sample_timeout_s,
                use_host_depletion=use_host_depletion,
                state_archive=state_archive,
                t_start=t_start,
            )

        # Custom reference / Portable Mode A & B fallback OR Mode C Diploid Human VariationLoop
        if not ref_fasta:
            raise ValueError(
                "Reference genome FASTA (--ref-fasta) is required for diploid VariationLoop "
                "and when bundled Pf3D7 surveillance assets are absent"
            )

        from .malariavar import genome as G
        from .malariavar import varloop_giab as VG

        eff_ref = _gunzip_if_needed(ref_fasta, work_dir / "ref", "reference.fa")
        _ensure_faidx(eff_ref)
        g = G.from_fasta(eff_ref, bed=targets_bed, need_core=True)
        ctg_lens = dict(zip(g.names, [int(x) for x in g.lengths]))
        bed_regions = _parse_bed_regions(targets_bed, ctg_lens) if targets_bed else []

        # Build sample jobs uniformly from manifest, History BAM/CRAM, or History FASTQs
        jobs: list[dict[str, Any]] = []
        if manifest_path:
            for idx_r, row in enumerate( _read_manifest_table(manifest_path)):
                sid = _pick_col(row, ("sample_id", "Sample", "sample"), f"sample_{idx_r + 1}")
                grp = _pick_col(row, ("country_or_group", "population", "group", "condition"), "cohort")
                src = _pick_col(row, ("source_uri_or_srr", "cram_url", "bam_url", "run_or_url", "fastq_1", "accession", "source", "url"))
                crai_u = _pick_col(row, ("crai_uri", "crai_uri_optional", "crai_url", "crai"))
                mate2 = _pick_col(row, ("mate2_url", "fastq_2", "r2", "mate2"))
                if not src and sid in local_map:
                    src = sid
                if not src:
                    raise ValueError(f"Sample {sid!r}: missing input source column in manifest")
                jobs.append({"sample": sid, "group": grp, "src": src, "mate2": mate2, "crai": crai_u, "row": row})
        elif bam_path:
            jobs.append(
                {
                    "sample": sample_name,
                    "group": "cohort",
                    "src": str(Path(bam_path).resolve()),
                    "mate2": "",
                    "crai": str(Path(crai_path).resolve()) if crai_path else "",
                    "row": {},
                }
            )
        elif reads_r1:
            jobs.append(
                {
                    "sample": sample_name,
                    "group": "cohort",
                    "src": str(Path(reads_r1).resolve()),
                    "mate2": str(Path(reads_r2).resolve()) if reads_r2 else "",
                    "crai": "",
                    "row": {},
                }
            )
        elif reads:
            if len(reads) == 2 and Path(reads[0]).exists() and Path(reads[1]).exists():
                jobs.append(
                    {
                        "sample": sample_name,
                        "group": "cohort",
                        "src": str(Path(reads[0]).resolve()),
                        "mate2": str(Path(reads[1]).resolve()),
                        "crai": "",
                        "row": {},
                    }
                )
            else:
                for idx_r, rp in enumerate(reads):
                    sid = (read_ids[idx_r] if read_ids and idx_r < len(read_ids) else Path(rp).stem)
                    jobs.append({"sample": sid, "group": "cohort", "src": str(Path(rp).resolve()), "mate2": "", "crai": "", "row": {}})
        else:
            raise ValueError("Provide a sample manifest, History paired FASTQ files, or a History BAM/CRAM file")

        total_genome_bp = sum(ctg_lens.values())
        if dedup_bits is None:
            if total_genome_bp < 1_000_000:
                eff_dedup_bits = 18
            elif total_genome_bp < 100_000_000:
                eff_dedup_bits = 20
            elif total_genome_bp < 1_000_000_000:
                eff_dedup_bits = 22
            else:
                eff_dedup_bits = 24
        else:
            eff_dedup_bits = int(dedup_bits)

        dp_params = VG.DiploidParams(
            min_mapq=int(min_mapq),
            min_bq=int(min_bq),
            snv_min_qual=float(snv_min_qual),
            indel_min_qual=float(indel_min_qual),
            enable_reassembly=bool(enable_reassembly),
            dedup_bits=eff_dedup_bits,
        )
        chosen_mapper: str | None = None
        sample_tels: list[dict[str, Any]] = []
        var_frames: list[pd.DataFrame] = []
        per_sample_vcfs: list[Path] = []
        p_frac = float(np.clip(subsample_fraction, 1e-4, 1.0))

        # Optional host/off-target 31-mer retention filter for target-filtered FASTQ calling
        kset_tab = None
        if analysis_mode == "wgs_stream" and use_host_depletion:
            regs_k = bed_regions or [(cn, 0, ln) for cn, ln in ctg_lens.items()]
            target_span = sum(max(0, e - s) for _, s, e in regs_k)
            if target_span > 45_000_000:
                raise ValueError(
                    f"Target 31-mer read filter span ({target_span / 1e6:.1f} Mb) exceeds the 45 Mb in-memory limit. "
                    "Provide a target BED file covering your regions of interest or disable the 31-mer pre-filter."
                )
            if target_span < 500_000:
                bits_k = 20
            elif target_span < 8_000_000:
                bits_k = 24
            else:
                bits_k = 26
            kset_tab, kmeta = VG.build_target_kset(g, regs_k, bits=bits_k)
            _log(f"Built target 31-mer pre-filter table ({kmeta['inserted_kmers']:,} k-mers across {target_span / 1e6:.2f} Mb)")

        for idx_j, job in enumerate(jobs, 1):
            sid = job["sample"]
            src = job["src"]
            s_vcf_gz = work_dir / "vcfs" / f"{idx_j:03d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', sid)}.vcf.gz"
            src_clean = src.split("?", 1)[0].lower()
            aln_fmt = None
            if src_clean.endswith(".cram"):
                aln_fmt = "cram"
            elif src_clean.endswith(".bam"):
                aln_fmt = "bam"
            elif not src.startswith(("http://", "https://")) and Path(src).is_file():
                aln_fmt = _detect_aln_format(Path(src))

            if aln_fmt in ("cram", "bam"):
                crai_src = job["crai"] or (src + ".crai" if aln_fmt == "cram" else "")
                if analysis_mode == "focused_cram" and aln_fmt == "cram" and crai_src:
                    if src.startswith(("http://", "https://")):
                        cram_ref = src
                    else:
                        cram_ref = str(_stage_local_aln(Path(src), work_dir / "aln_links", sid, "cram"))
                    crai_local = _fetch_crai_file(crai_src, work_dir / "crai" / f"{sid}.cram.crai")
                    regs_use = bed_regions or [(cn, 0, ln) for cn, ln in ctg_lens.items()]
                    tel = VG.run_varloop_crai_focused(
                        g, eff_ref, cram_ref, crai_local, regs_use, s_vcf_gz,
                        params=dp_params, sample=sid, pad_bp=int(cram_pad_bp), zoom_assisted=bool(enable_reassembly),
                    )
                    tel["sample"] = sid
                else:
                    if src.startswith(("http://", "https://")):
                        local_aln = _materialize_remote_input(src, work_dir / "aln" / f"{sid}.{aln_fmt}", "remote_aln", threads=threads)
                    else:
                        local_aln = _stage_local_aln(Path(src), work_dir / "aln_links", sid, aln_fmt)
                        if crai_src and Path(crai_src).is_file():
                            idx_ext = ".cram.crai" if aln_fmt == "cram" else ".bam.bai"
                            idx_link = local_aln.parent / f"{sid}{idx_ext}"
                            if idx_link.exists() or idx_link.is_symlink():
                                idx_link.unlink()
                            os.symlink(Path(crai_src).resolve(), idx_link)
                    tel = VG.run_varloop_bam_cram(
                        g, eff_ref, local_aln, s_vcf_gz,
                        params=dp_params, target_regions=bed_regions or None, zoom_regions=bed_regions or None,
                        sample=sid, threads=threads, call_workers=min(4, max(1, threads // 2)),
                        targets_bed=targets_bed,
                    )
            else:
                if chosen_mapper is None:
                    chosen_mapper = _select_mapper(mapper)
                    if chosen_mapper in ("bwa-mem2", "bwa"):
                        ext_c = ".bwt.2bit.64" if chosen_mapper == "bwa-mem2" else ".bwt"
                        if not Path(str(eff_ref) + ext_c).exists():
                            idx_fa = work_dir / "mapper_ref" / eff_ref.name
                            idx_fa.parent.mkdir(parents=True, exist_ok=True)
                            os.symlink(eff_ref, idx_fa)
                            subprocess.run([chosen_mapper, "index", str(idx_fa)], check=True)
                            eff_ref = idx_fa

                r1_ref, r2_ref, rkind = _resolve_fastq_pair_for_row(job["row"] or {"source": src, "mate2_url": job["mate2"]}, local_map, work_dir, sid)
                if rkind != "local_fastq":
                    r1_local = _materialize_remote_input(r1_ref, work_dir / "fq" / f"{sid}_R1.fq.gz", rkind, threads=threads)
                    r2_local = _materialize_remote_input(r2_ref, work_dir / "fq" / f"{sid}_R2.fq.gz", rkind, threads=threads) if r2_ref else None
                else:
                    r1_local = _stage_local_fastq(Path(r1_ref), work_dir / "fq_links", f"{sid}_R1")
                    r2_local = _stage_local_fastq(Path(r2_ref), work_dir / "fq_links", f"{sid}_R2") if r2_ref else None

                if r2_local is None:
                    raise ValueError(f"Sample {sid!r}: paired-end R1 and R2 FASTQ inputs are required for {analysis_mode!r}")

                if kset_tab is not None:
                    filt_r1 = work_dir / "fq_filt" / f"{sid}_R1.fq.gz"
                    filt_r2 = work_dir / "fq_filt" / f"{sid}_R2.fq.gz"
                    fstats = VG.extract_sieved_fastq_pair(
                        r1_local, r2_local, filt_r1, filt_r2,
                        kset_tab=kset_tab, stride=5, need=1, p_sample=p_frac,
                        decomp_threads=min(4, threads), comp_threads=min(4, threads),
                    )
                    _log(f"  Depletion filter {sid}: kept {fstats['spots_emitted']}/{fstats['spots_seen']} spots")
                    r1_use, r2_use, p_run = filt_r1, filt_r2, 1.0
                else:
                    r1_use, r2_use, p_run = r1_local, r2_local, p_frac

                tel = VG.run_varloop_fastq(
                    g, eff_ref, r1_use, r2_use, s_vcf_gz,
                    params=dp_params, map_threads=max(1, threads),
                    target_regions=bed_regions or None, zoom_regions=bed_regions or None,
                    sample=sid, p_sample=p_run, call_workers=min(4, max(1, threads // 2)),
                    mapper_name=chosen_mapper,
                )

            sample_tels.append(tel)
            per_sample_vcfs.append(s_vcf_gz)
            var_frames.append(_vcf_to_dataframe(s_vcf_gz, sample_override=sid))
            _log(f"  [{idx_j}/{len(jobs)}] {sid}: {tel.get('n_variants', 0)} variants ({tel.get('total_wall_s', tel.get('wall_s', 0)):.2f}s)")

        variants_df = pd.concat(var_frames, ignore_index=True) if var_frames else _vcf_to_dataframe(per_sample_vcfs[0])
        out_variants_p.parent.mkdir(parents=True, exist_ok=True)
        variants_df.to_csv(out_variants_p, sep="\t", index=False, float_format="%.6g")

        if out_vcf_p:
            if len(per_sample_vcfs) == 1:
                _materialize_vcf_output(per_sample_vcfs[0], out_vcf_p)
            elif shutil.which("bcftools"):
                merged_gz = work_dir / "merged.vcf.gz"
                subprocess.run(["bcftools", "merge", "-Oz", "-o", str(merged_gz), *[str(p) for p in per_sample_vcfs]], check=True)
                _materialize_vcf_output(merged_gz, out_vcf_p)
            else:
                _materialize_vcf_output(per_sample_vcfs[0], out_vcf_p)

        if out_drm_p:
            out_drm_p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "sample_id": t.get("sample"),
                        "analysis_mode": analysis_mode,
                        "n_variants": t.get("n_variants", 0),
                        "n_snv": t.get("n_snv", 0),
                        "n_indel": t.get("n_indel", 0),
                        "note": "WHO DRM panel applies to Pf3D7 pathogen surveillance cohorts",
                    }
                    for t in sample_tels
                ]
            ).to_csv(out_drm_p, sep="\t", index=False)

        ref_lbl = (reference_label or "").strip()
        if not ref_lbl:
            raw_ref_name = Path(ref_fasta).name
            ref_lbl = "reference_genome" if raw_ref_name.endswith(".dat") else raw_ref_name

        summary_stats = {
            "mode": analysis_mode,
            "reference_label": ref_lbl,
            "n_samples": len(jobs),
            "n_variants": len(variants_df),
            "wall_s": round(time.time() - t_start, 2),
        }
        render_variationloop_report(
            out_report_p,
            analysis_mode=analysis_mode,
            summary_stats=summary_stats,
            variants_df=variants_df,
            sample_telemetry=sample_tels,
        )

        store = FocaloopStateStore(state_root)
        HypothesisLedger.from_store(store, alpha_budget=0.05)
        variants_df.to_parquet(state_root / "variants.parquet", index=False)
        pd.DataFrame(sample_tels).to_parquet(state_root / "sample_telemetry.parquet", index=False)
        sample_records = []
        for j in jobs:
            rec = {"sample": j["sample"], "group": j["group"], "condition": j["group"]}
            for k, v in (j.get("row") or {}).items():
                if k and k not in rec and v:
                    rec[k] = v
            sample_records.append(rec)
        state_meta = {
            "engine": "variationloop",
            "version": "0.3.0",
            "blueprint_id": "variationloop",
            "mode": analysis_mode,
            "summary": summary_stats,
            "samples": sample_records,
        }
        write_json(state_root / STATE_META, state_meta)
        if state_archive:
            pack_state(state_root, state_archive)
        return state_meta
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
