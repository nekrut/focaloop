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
_READ_EXT_RE = re.compile(r"(\.f(ast)?qsanger(\.gz|\.bz2)?|\.f(ast)?q(\.gz|\.bz2)?|\.bam|\.cram|\.sam|\.gz|\.bz2)+$", re.IGNORECASE)
_MATE_SUFFIX_RE = re.compile(r"(_R?[12]|_[12])$", re.IGNORECASE)


def _as_str_list(val: str | Path | Sequence[str | Path] | None, *, keep_empty: bool = False) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (str, Path)):
        s = str(val).strip()
        return [s] if (keep_empty or s) else []
    out: list[str] = []
    for item in val:
        s = str(item or "").strip()
        if keep_empty or s:
            out.append(s)
    return out


def _clean_sample_name(raw: str | Path | None, fallback: str = "sample_1", *, strip_mate_suffix: bool = True) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return fallback
    looks_like_path = (
        isinstance(raw, Path)
        or txt.startswith(("/", "./", "../", "~"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", txt))
        or bool(_READ_EXT_RE.search(txt))
        or (txt.lower().endswith(".dat") and "dataset_" in txt.lower())
    )
    base = Path(txt).name if (looks_like_path and ("/" in txt or "\\" in txt)) else txt
    if base.lower().endswith(".dat") and base.lower().startswith("dataset_"):
        return fallback
    had_fastq_ext = bool(re.search(r"\.(?:fastqsanger|fastq|fq)(?:\.gz|\.bz2|\.zst)?$", base, re.I))
    stem = _READ_EXT_RE.sub("", base)
    if strip_mate_suffix and (had_fastq_ext or re.search(r"(?:_R[12]|_R[12]_001)$", stem, re.I)):
        stem = _MATE_SUFFIX_RE.sub("", stem)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.").lstrip("-").strip("_.")
    if not cleaned:
        fb = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fallback)).strip("_.").lstrip("-").strip("_.")
        return fb or "sample_1"
    return cleaned


def _dedup_sample_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    renamed: list[str] = []
    for raw in names:
        n = (raw or "sample").lstrip("-") or "sample"
        cnt = seen.get(n, 0) + 1
        seen[n] = cnt
        final_n = n if cnt == 1 else f"{n}_{cnt}"
        if cnt > 1:
            renamed.append(f"{n}->{final_n}")
        out.append(final_n)
    if renamed:
        _log(f"Deduplicated colliding sample identifiers: {', '.join(renamed[:10])}")
    return out


def _vcf_has_variants(vcf_path: Path) -> bool:
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln and not ln.startswith("#"):
                return True
    return False


def _bcftools_merge_batched(
    vcf_paths: Sequence[Path],
    merged_gz: Path,
    work_dir: Path,
    threads: int = 4,
    batch_size: int = 150,
) -> None:
    """Hierarchically merge bgzipped+tabix VCFs in batches <= batch_size to avoid ulimit -n exhaustion."""
    cur = [Path(p) for p in vcf_paths]
    if not any(_vcf_has_variants(p) for p in cur):
        _merge_vcfs_pure_python(cur, merged_gz)
        if shutil.which("tabix") and str(merged_gz).endswith(".gz"):
            subprocess.run(["tabix", "-f", "-p", "vcf", str(merged_gz)], check=False)
        return
    merge_threads = str(max(1, min(int(threads), 8)))
    round_idx = 0
    while len(cur) > batch_size:
        round_idx += 1
        next_round: list[Path] = []
        bdir = work_dir / f"merge_r{round_idx}"
        bdir.mkdir(parents=True, exist_ok=True)
        for bi in range(0, len(cur), batch_size):
            chunk = cur[bi : bi + batch_size]
            if len(chunk) == 1:
                next_round.append(chunk[0])
                continue
            out_b = bdir / f"batch_{bi // batch_size:04d}.vcf.gz"
            if not any(_vcf_has_variants(p) for p in chunk):
                plain_b = out_b.with_suffix("")
                _merge_vcfs_pure_python(chunk, plain_b)
                subprocess.run(["bgzip", "-f", str(plain_b)], check=True)
                subprocess.run(["tabix", "-f", "-p", "vcf", str(out_b)], check=True)
                next_round.append(out_b)
                continue
            fofn = bdir / f"batch_{bi // batch_size:04d}.fofn"
            fofn.write_text("\n".join(str(p) for p in chunk) + "\n", encoding="utf-8")
            subprocess.run(
                ["bcftools", "merge", "--no-version", "--threads", merge_threads, "-l", str(fofn), "-Oz", "-o", str(out_b)],
                check=True,
            )
            subprocess.run(["tabix", "-f", "-p", "vcf", str(out_b)], check=True)
            next_round.append(out_b)
        cur = next_round
    fofn_final = work_dir / "merge_final.fofn"
    fofn_final.write_text("\n".join(str(p) for p in cur) + "\n", encoding="utf-8")
    subprocess.run(
        ["bcftools", "merge", "--no-version", "--threads", merge_threads, "-l", str(fofn_final), "-Oz", "-o", str(merged_gz)],
        check=True,
    )


def _merge_vcfs_pure_python(vcf_paths: Sequence[Path], out_vcf: Path) -> None:
    """Merge per-sample VCF(.gz) files into one multi-sample VCF when bcftools/tabix are absent."""
    meta_lines: list[str] = []
    seen_meta: set[str] = set()
    sample_names: list[str] = []
    # key = (chrom_ord, chrom, pos, ref, alt) -> {sample_idx: sample_col_str, qual, filt, info, fmt}
    contig_order: dict[str, int] = {}
    records: dict[tuple[str, int, str, str], dict[str, Any]] = {}

    for s_idx, vp in enumerate(vcf_paths):
        opener = gzip.open if str(vp).endswith(".gz") else open
        with opener(vp, "rt", encoding="utf-8") as fh:
            sname = f"sample_{s_idx + 1}"
            for ln in fh:
                if ln.startswith("##"):
                    if ln not in seen_meta:
                        seen_meta.add(ln)
                        meta_lines.append(ln)
                        m_ctg = re.match(r"##contig=<ID=([^,>]+)", ln)
                        if m_ctg and m_ctg.group(1) not in contig_order:
                            contig_order[m_ctg.group(1)] = len(contig_order)
                    continue
                if ln.startswith("#CHROM"):
                    hdr = ln.rstrip("\n").split("\t")
                    if len(hdr) > 9 and hdr[9].strip():
                        sname = hdr[9].strip()
                    sample_names.append(sname)
                    continue
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue
                chrom, pos_s, vid, ref, alt, qual, flt, info, fmt, scol = parts[:10]
                if chrom not in contig_order:
                    contig_order[chrom] = len(contig_order)
                key = (chrom, int(pos_s), ref, alt)
                rec = records.get(key)
                if rec is None:
                    rec = {"id": vid, "qual": qual, "flt": flt, "info": info, "fmt": fmt, "samples": {}}
                    records[key] = rec
                else:
                    try:
                        if float(qual) > float(rec["qual"]):
                            rec["qual"] = qual
                    except Exception:
                        pass
                rec["samples"][s_idx] = scol

    sample_names = _dedup_sample_names(sample_names)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    out_opener = gzip.open if str(out_vcf).endswith(".gz") else open
    with out_opener(out_vcf, "wt", encoding="utf-8") as fo:
        for ml in meta_lines:
            fo.write(ml)
        fo.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names) + "\n")
        for key in sorted(records, key=lambda k: (contig_order.get(k[0], 10**9), k[1], k[2], k[3])):
            chrom, pos, ref, alt = key
            rec = records[key]
            n_fmt = max(1, len(rec["fmt"].split(":")))
            sample_vals = list(rec["samples"].values())
            is_haploid = bool(sample_vals) and all(
                ("/" not in sv.split(":", 1)[0]) and ("|" not in sv.split(":", 1)[0]) for sv in sample_vals
            )
            miss_gt = "." if is_haploid else "./."
            miss_cell = ":".join([miss_gt] + ["."] * (n_fmt - 1))
            scols = [rec["samples"].get(i, miss_cell) for i in range(len(sample_names))]
            fo.write(
                f"{chrom}\t{pos}\t{rec['id']}\t{ref}\t{alt}\t{rec['qual']}\t{rec['flt']}\t{rec['info']}\t{rec['fmt']}\t"
                + "\t".join(scols)
                + "\n"
            )

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
  .hero code { background: rgba(255, 255, 255, 0.16); color: #f8fafc; padding: 2px 6px; border-radius: 4px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; min-width: 0; }
  .kpi .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .kpi .val { font-size: 20px; font-weight: 700; margin-top: 4px; color: var(--ink); overflow-wrap: anywhere; word-break: break-word; line-height: 1.25; }
  .kpi .sub { font-size: 12px; color: var(--muted); margin-top: 3px; }
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
             padding: 18px 20px; margin-bottom: 16px; }
  .warn-box { border-color: #f59e0b; background: #fffbeb; }
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
    text = Path(manifest_path).read_text(encoding="utf-8-sig")
    non_blank = [ln for ln in text.splitlines() if ln.strip()]
    if not non_blank:
        raise ValueError(f"{Path(manifest_path).name}: manifest is empty")
    lines: list[str] = []
    for idx_ln, ln in enumerate(non_blank):
        s_ln = ln.lstrip()
        if s_ln.startswith("#"):
            # Allow a leading '#' on the header row (e.g. `#sample_id\tgroup`) while skipping comment lines (L36)
            cand_hdr = s_ln.lstrip("#").strip()
            if not lines and ("\t" in cand_hdr or "," in cand_hdr) and re.search(
                r"\b(?:sample|sample_id|id|group|population|country|country_or_group|source|accession|url)\b",
                cand_hdr,
                re.I,
            ):
                lines.append(cand_hdr)
            continue
        lines.append(ln)
    if not lines:
        raise ValueError(f"{Path(manifest_path).name}: manifest contains no header or sample rows")
    delim = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
    rows = [{(k or "").strip().lstrip("\ufeff"): (v or "").strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{Path(manifest_path).name}: manifest has header but zero sample rows")
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
    s.headers["User-Agent"] = "focaloop/0.4.0"
    return s


def _head_content_length(url: str, timeout: float = 45.0, retries: int = 3) -> int:
    s = _ipv4_session()
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            r = s.head(url, allow_redirects=True, timeout=timeout)
            if 200 <= r.status_code < 300 and r.headers.get("Content-Length"):
                return int(r.headers["Content-Length"])
            if r.status_code in (401, 403, 404, 410):
                r.raise_for_status()
        except Exception as exc:
            last_err = exc
            if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403, 404, 410):
                raise
        try:
            with s.get(url, headers={"Range": "bytes=0-0"}, stream=True, allow_redirects=True, timeout=timeout) as rg:
                rg.raise_for_status()
                if rg.status_code == 206:
                    cr = rg.headers.get("Content-Range", "")
                    if "/" in cr:
                        tot = cr.rsplit("/", 1)[-1].strip()
                        if tot.isdigit():
                            return int(tot)
                elif rg.status_code == 200 and rg.headers.get("Content-Length"):
                    return int(rg.headers["Content-Length"])
        except Exception as exc:
            last_err = exc
            if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403, 404, 410):
                raise
        if attempt + 1 < retries:
            time.sleep(1.0 * (2 ** attempt))
    raise RuntimeError(f"Could not determine remote byte length for {url}: {last_err}")


def _assert_manifest_uri_allowed(
    uri: str,
    sample_id: str,
    field_name: str,
    *,
    allow_local: bool,
    local_map: dict[str, list[Path]] | None = None,
) -> None:
    if not uri or allow_local:
        return
    if local_map and (uri in local_map or sample_id in local_map):
        return
    if uri.startswith(("http://", "https://", "ftp://")):
        return
    accs = [a.strip() for a in re.split(r"[;,]", uri) if a.strip()]
    if accs and all(_INSDC_RUN_RE.match(a) for a in accs):
        return
    raise ValueError(
        f"Sample {sample_id!r}: local filesystem path in manifest {field_name} is disallowed "
        "(pass --allow-local-manifest-paths on CLI or supply Galaxy History datasets)"
    )


def _fetch_crai_file(crai_uri: str, dst: Path, *, allow_local: bool = True) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not crai_uri.startswith(("http://", "https://", "ftp://")):
        if not allow_local:
            raise ValueError("Local CRAI path in manifest requires --allow-local-manifest-paths")
        if Path(crai_uri).is_file():
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


def _match_ena_paired_urls(urls: Sequence[str]) -> tuple[str | None, str | None]:
    """Select _1/_2 paired FASTQ URLs explicitly even when ENA lists 3 files (ACC.fastq.gz + _1 + _2) (#21)."""
    clean = [u.strip() for u in urls if u and u.strip()]
    if not clean:
        return None, None
    r1_m = [u for u in clean if re.search(r"_(?:R?1)(?:_001)?\.f(?:ast)?q(?:\.gz)?$", u.split("?", 1)[0], re.I)]
    r2_m = [u for u in clean if re.search(r"_(?:R?2)(?:_001)?\.f(?:ast)?q(?:\.gz)?$", u.split("?", 1)[0], re.I)]
    if r1_m and r2_m:
        return r1_m[0], r2_m[0]
    if len(clean) == 2:
        return clean[0], clean[1]
    return clean[0], None


def _resolve_ena_or_odp(acc: str, retries: int = 4) -> dict[str, Any]:
    acc = acc.strip().upper()
    odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
    all_odp = False
    odp_bytes = np.nan
    try:
        odp_bytes = float(_head_content_length(odp_url, timeout=20.0, retries=2))
        all_odp = True
    except Exception:
        pass

    fastq_urls: list[str] = []
    fastq_bytes = np.nan
    order_hint = "flowcell"
    s = _ipv4_session()
    ena_ok = False
    ena_err: Exception | None = None
    for attempt in range(max(1, retries)):
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
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"ENA API returned HTTP {r.status_code}")
            r.raise_for_status()
            ena_ok = True
            if r.text.strip():
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
            break
        except Exception as exc:
            ena_err = exc
            if attempt + 1 < retries:
                time.sleep(1.0 * (2 ** attempt))
    if not ena_ok and not all_odp:
        raise RuntimeError(f"Failed to resolve INSDC accession {acc!r} via ENA or SRA ODP: {ena_err}")

    is_paired = False
    if fastq_urls:
        u1, u2 = _match_ena_paired_urls(fastq_urls[0].split(";"))
        is_paired = bool(u1 and u2)
    return {
        "ena_runs": [acc],
        "all_odp": all_odp,
        "odp_bytes": odp_bytes,
        "all_paired_fastq": is_paired,
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


def _validate_and_parse_bed(
    bed_path: str | Path,
    contig_lengths: dict[str, int],
    ref_label: str = "reference FASTA",
) -> list[tuple[str, int, int]]:
    """Parse, validate, and merge 0-based BED intervals against `contig_lengths` (#28, #29)."""
    raw_rows = 0
    bed_ctgs: list[str] = []
    skipped_ctgs: list[str] = []
    by_ctg: dict[str, list[tuple[int, int]]] = {}
    with open(bed_path, encoding="utf-8-sig") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            parts = ln.rstrip("\n").split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                s0 = max(0, int(parts[1]))
                e0 = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"Malformed BED coordinates in {Path(bed_path).name!r}: {ln.strip()!r}") from exc
            if e0 <= s0:
                continue
            raw_rows += 1
            if chrom not in bed_ctgs and len(bed_ctgs) < 8:
                bed_ctgs.append(chrom)
            if chrom not in contig_lengths:
                if chrom not in skipped_ctgs:
                    skipped_ctgs.append(chrom)
                continue
            e_clip = min(contig_lengths[chrom], e0)
            if e_clip > s0:
                by_ctg.setdefault(chrom, []).append((s0, e_clip))

    if raw_rows == 0:
        raise ValueError(f"Target BED {Path(bed_path).name!r} contains zero valid (start < end) intervals")
    if not by_ctg:
        raise ValueError(
            f"Target BED {Path(bed_path).name!r} matched 0 contigs in {ref_label} "
            f"(BED contigs: {bed_ctgs[:5]}, reference contigs: {list(contig_lengths.keys())[:5]})"
        )
    if skipped_ctgs:
        _log(
            f"Warning: {len(skipped_ctgs)} BED contig(s) absent from {ref_label} were skipped "
            f"({', '.join(skipped_ctgs[:5])})"
        )

    merged_out: list[tuple[str, int, int]] = []
    for chrom in contig_lengths:
        ivs = by_ctg.get(chrom)
        if not ivs:
            continue
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for a, b in ivs[1:]:
            if a <= cur_e:
                cur_e = max(cur_e, b)
            else:
                merged_out.append((chrom, cur_s, cur_e))
                cur_s, cur_e = a, b
        merged_out.append((chrom, cur_s, cur_e))
    return merged_out


def _pad_and_merge_regions(
    regions: Sequence[tuple[str, int, int]],
    contig_lengths: dict[str, int],
    pad_bp: int,
) -> list[tuple[str, int, int]]:
    by_ctg: dict[str, list[tuple[int, int]]] = {}
    for chrom, s0, e0 in regions:
        if chrom not in contig_lengths:
            continue
        clen = contig_lengths[chrom]
        a = max(0, int(s0) - int(pad_bp))
        b = min(clen, int(e0) + int(pad_bp))
        if b > a:
            by_ctg.setdefault(chrom, []).append((a, b))
    out: list[tuple[str, int, int]] = []
    for chrom in contig_lengths:
        ivs = by_ctg.get(chrom)
        if not ivs:
            continue
        ivs.sort()
        cs, ce = ivs[0]
        for a, b in ivs[1:]:
            if a <= ce:
                ce = max(ce, b)
            else:
                out.append((chrom, cs, ce))
                cs, ce = a, b
        out.append((chrom, cs, ce))
    return out


def _parse_bed_regions(bed_path: str | Path | None, contig_lengths: dict[str, int] | None = None) -> list[tuple[str, int, int]]:
    if not bed_path:
        return []
    if contig_lengths is not None:
        return _validate_and_parse_bed(bed_path, contig_lengths)
    regs: list[tuple[str, int, int]] = []
    with open(bed_path, encoding="utf-8-sig") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            parts = ln.rstrip("\n").split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            s0 = max(0, int(parts[1]))
            e0 = int(parts[2])
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
    cols = ["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "gq", "dp", "ad_ref", "ad_alt", "wsaf", "qual", "source"]
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
                if gt in ("0/0", "./.", "0", ".", "0|0", ".|."):
                    continue
                gq = int(fmap["GQ"]) if fmap.get("GQ", ".").isdigit() else np.nan
                dp = int(fmap["DP"]) if fmap.get("DP", ".").isdigit() else np.nan
                ad_parts = fmap.get("AD", "").split(",")
                ad_ref = int(ad_parts[0]) if len(ad_parts) >= 1 and ad_parts[0].isdigit() else np.nan
                ad_alt = ",".join(ad_parts[1:]) if len(ad_parts) >= 2 else ""
                wsaf_val = np.nan
                raw_ws = fmap.get("WSAF", "")
                if raw_ws and raw_ws != ".":
                    ws_toks = []
                    for t in raw_ws.split(","):
                        if t and t != ".":
                            try:
                                ws_toks.append(float(t))
                            except ValueError:
                                pass
                    if ws_toks:
                        wsaf_val = float(max(ws_toks))
                if not np.isfinite(wsaf_val):
                    try:
                        alt_nums = [int(x) for x in ad_parts[1:] if x.isdigit()]
                        if alt_nums:
                            ref_n = int(ad_parts[0]) if ad_parts and ad_parts[0].isdigit() else 0
                            denom = max(ref_n + sum(alt_nums), int(dp) if np.isfinite(dp) else 0, 1)
                            wsaf_val = float(max(alt_nums)) / float(denom)
                    except Exception:
                        pass
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
                        "wsaf": round(wsaf_val, 4) if np.isfinite(wsaf_val) else np.nan,
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
        "##source=Focaloop_VariationLoop_v0.4.0",
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
        k13_miss_raw = r.get("aa_k13_missing_codons", r.get("k13_missing_codons", 0))
        try:
            k13_miss_cnt = int(float(k13_miss_raw or 0)) if pd.notna(k13_miss_raw) else 0
        except (TypeError, ValueError):
            k13_miss_cnt = 0
        is_wt_k13 = k13_raw is None or str(k13_raw).strip() in ("", "none", "None", "nan")
        if is_wt_k13 and k13_miss_cnt > 0:
            k13_str = f"WT ({k13_miss_cnt} codons uncovered)"
        else:
            k13_str = "WT" if is_wt_k13 else str(k13_raw)
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
                "k13_has_missing": k13_miss_cnt > 0,
                "k13_missing_codons": k13_miss_cnt,
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
        n_tot = int(summary_stats.get("n_variants", len(variants_df)))
        if "n_snv" in summary_stats:
            n_snv = int(summary_stats["n_snv"])
            n_indel = int(summary_stats.get("n_indel", max(0, n_tot - n_snv)))
        else:
            n_snv = int((variants_df["kind"] == "SNV").sum()) if ("kind" in variants_df.columns and not variants_df.empty) else 0
            n_indel = max(0, n_tot - n_snv)
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
                f"<td class='num'>{_fmt_num(v.get('wsaf'), 3)}</td>"
                f"<td class='num'>{_fmt_num(v.get('qual'), 1)}</td>"
                f"<td class='num'>{_esc(v.get('dp'))}</td>"
                f"<td class='num'>{_esc(v.get('ad_ref'))},{_esc(v.get('ad_alt'))}</td>"
                f"<td><code>{_esc(v.get('source', 'variationloop'))}</code></td></tr>"
            )

        ploidy_lbl = str(summary_stats.get("ploidy", "diploid"))
        dup_warn_samples = summary_stats.get("high_dup_samples") or []
        dup_banner = ""
        if dup_warn_samples:
            dup_banner = (
                '<div class="section warn-box">'
                '<h2>High PCR Duplicate Rate Notice</h2>'
                f'<p>More than 95% of mapped reads were marked as coordinate+TLEN duplicates in: '
                f'<code>{_esc(", ".join(str(x) for x in dup_warn_samples[:10]))}</code>. '
                'If these libraries come from targeted PCR amplicon sequencing, re-run with PCR duplicate removal set to Off.</p>'
                '</div>'
            )
        body_sections = f"""
  {dup_banner}
  <div class="kpis">
    <div class="kpi"><div class="label">Samples Analyzed</div><div class="val">{n_samp}</div><div class="sub">Mode: <code>{_esc(analysis_mode)}</code> · Model: <code>{_esc(ploidy_lbl)}</code></div></div>
    <div class="kpi"><div class="label">Total Variants</div><div class="val">{n_tot:,}</div><div class="sub">{n_snv:,} SNVs · {n_indel:,} Indels</div></div>
    <div class="kpi"><div class="label">Omnitig Rescued</div><div class="val">{n_asm:,}</div><div class="sub">Multi-k local haplotype reassembly</div></div>
    <div class="kpi"><div class="label">PCR Duplicates Filtered</div><div class="val">{n_dup:,}</div><div class="sub">64-bit collision-free coordinate+TLEN dedup</div></div>
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
    <h2>Variant Calls Preview ({min(len(variants_df), 50)} of {n_tot:,})</h2>
    <table>
      <thead><tr><th>Sample</th><th>Locus</th><th>Alleles</th><th>Type</th><th>GT</th><th class="num">WSAF</th><th class="num">QUAL</th><th class="num">DP</th><th class="num">AD</th><th>Source</th></tr></thead>
      <tbody>{''.join(var_html) or '<tr><td colspan="10">No variants called above quality thresholds.</td></tr>'}</tbody>
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
    *,
    allow_local: bool = True,
) -> tuple[str, str | None, str]:
    """Resolve a manifest row's source into (r1_path_or_url, r2_path_or_url, kind)."""
    src = _pick_col(row, ("source_uri_or_srr", "run_or_url", "fastq_1", "accession", "source", "url"))
    mate2 = _pick_col(row, ("mate2_url", "fastq_2", "r2", "mate2"))
    cand_local = local_map.get(src) or local_map.get(sample_id)
    if cand_local:
        ordered = sorted(cand_local, key=lambda p: p.name)[:2]
        return str(ordered[0]), (str(ordered[1]) if len(ordered) > 1 else None), "local_fastq"
    if _INSDC_RUN_RE.match(src):
        info = _resolve_ena_or_odp(src)
        fq_raw = info.get("fastq_urls") or []
        if fq_raw:
            urls = [u.strip() for u in fq_raw[0].split(";") if u.strip()]
            urls = [("https://" + u[6:] if u.startswith("ftp://") else (u if u.startswith("http") else "https://" + u)) for u in urls]
            u1, u2 = _match_ena_paired_urls(urls)
            if u1:
                return u1, u2, "remote_fastq"
        return src.upper(), None, "insdc_odp"
    if src.startswith(("http://", "https://", "ftp://")):
        parts = [u.strip() for u in src.split(";") if u.strip()]
        if mate2:
            _assert_manifest_uri_allowed(mate2, sample_id, "mate2", allow_local=allow_local, local_map=local_map)
            parts.append(mate2.strip())
        parts = [("https://" + u[6:] if u.startswith("ftp://") else u) for u in parts]
        u1, u2 = _match_ena_paired_urls(parts)
        return (u1 or parts[0]), u2, "remote_fastq"
    if not allow_local:
        raise ValueError(
            f"Sample {sample_id!r}: local filesystem path in manifest is disallowed "
            "(pass --allow-local-manifest-paths on CLI or supply Galaxy History datasets)"
        )
    if src and Path(src).exists():
        p1 = str(Path(src).resolve())
        p2 = str(Path(mate2).resolve()) if (mate2 and Path(mate2).exists()) else None
        return p1, p2, "local_fastq"
    raise ValueError(f"Sample {sample_id!r}: could not resolve FASTQ source {src!r}")


def _materialize_odp_paired(acc: str, dst_r1: Path, dst_r2: Path, threads: int = 4) -> tuple[Path, Path]:
    """Stream interleaved SRA ODP FASTQ via xsra and split lockstep mates into paired R1/R2 .fq.gz files (L31)."""
    if not shutil.which("xsra"):
        raise RuntimeError(f"INSDC accession {acc!r} requires 'xsra' in PATH when ENA FASTQ URLs are unavailable")
    dst_r1.parent.mkdir(parents=True, exist_ok=True)
    dst_r2.parent.mkdir(parents=True, exist_ok=True)
    odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
    proc = subprocess.Popen(
        ["xsra", "dump", "-Q", "-P", odp_url, "-t", str(max(1, threads))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    n_pairs = 0
    try:
        assert proc.stdout is not None
        with gzip.open(dst_r1, "wb", compresslevel=1) as f1, gzip.open(dst_r2, "wb", compresslevel=1) as f2:
            while True:
                h1 = proc.stdout.readline()
                if not h1:
                    break
                s1 = proc.stdout.readline()
                p1 = proc.stdout.readline()
                q1 = proc.stdout.readline()
                h2 = proc.stdout.readline()
                s2 = proc.stdout.readline()
                p2 = proc.stdout.readline()
                q2 = proc.stdout.readline()
                if not (s1 and p1 and q1 and h2 and s2 and p2 and q2):
                    raise RuntimeError(f"Truncated interleaved xsra FASTQ stream for {acc!r}")
                f1.write(h1 + s1 + p1 + q1)
                f2.write(h2 + s2 + p2 + q2)
                n_pairs += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        err_txt = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"xsra dump failed for {acc!r} (rc={rc}): {err_txt[-300:]}")
    if n_pairs == 0:
        raise RuntimeError(f"xsra dump produced 0 paired reads for {acc!r}")
    return dst_r1, dst_r2


def _materialize_remote_input(uri_or_acc: str, dst: Path, kind: str, threads: int = 4, retries: int = 5) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if kind == "insdc_odp":
        r2_dst = dst.with_name(dst.name.replace("_R1", "_R2"))
        if r2_dst == dst:
            r2_dst = dst.with_suffix(".r2.fq.gz")
        _materialize_odp_paired(uri_or_acc, dst, r2_dst, threads=threads)
        return dst
    part = dst.with_name(f".{dst.name}.{os.getpid()}.part")
    part.unlink(missing_ok=True)
    s = _ipv4_session()
    expected_total: int | None = None
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        offset = part.stat().st_size if part.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        try:
            with s.get(uri_or_acc, headers=headers, stream=True, timeout=(30, 180)) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code} downloading {uri_or_acc}")
                r.raise_for_status()
                if offset > 0 and r.status_code == 200:
                    # Server ignored Range resume: restart from byte 0
                    offset = 0
                if expected_total is None:
                    if r.status_code == 200 and r.headers.get("Content-Length"):
                        expected_total = int(r.headers["Content-Length"])
                    elif r.status_code == 206 and "/" in r.headers.get("Content-Range", ""):
                        tot_s = r.headers["Content-Range"].rsplit("/", 1)[-1].strip()
                        if tot_s.isdigit():
                            expected_total = int(tot_s)
                mode = "ab" if offset > 0 else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            got_sz = part.stat().st_size if part.is_file() else 0
            if expected_total is not None and got_sz != expected_total:
                raise RuntimeError(f"Incomplete download for {uri_or_acc}: got {got_sz} bytes, expected {expected_total}")
            if got_sz == 0:
                raise RuntimeError(f"Downloaded empty (0-byte) file from {uri_or_acc}")
            os.replace(part, dst)
            return dst
        except Exception as exc:
            last_err = exc
            if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403, 404, 410):
                part.unlink(missing_ok=True)
                raise
            if attempt + 1 < retries:
                time.sleep(min(20.0, 1.5 * (2 ** attempt)))
    part.unlink(missing_ok=True)
    raise RuntimeError(f"Failed downloading {uri_or_acc} after {retries} attempts: {last_err}")


def _sanitize_state_sample_record(
    sample_id: str,
    group: str,
    extra_row: dict[str, Any] | None,
    reads_counted: int,
    median_depth: float,
) -> dict[str, Any]:
    """Build portable sample metadata entry with reads_counted, median_depth, and zero local filesystem paths (#72, L41)."""
    PATH_LIKE_KEYS = {
        "source", "source_uri_or_srr", "cram_url", "bam_url", "run_or_url",
        "fastq_1", "fastq_2", "r2", "mate2", "mate2_url",
        "crai", "crai_uri", "crai_url", "crai_uri_optional", "url",
        "vcf_path", "pileup_vcf_path",
    }
    rec: dict[str, Any] = {
        "sample": sample_id,
        "group": group,
        "condition": group,
        "reads_counted": int(reads_counted),
        "median_depth": round(float(median_depth), 2) if np.isfinite(median_depth) else 0.0,
    }
    for k, v in (extra_row or {}).items():
        if not k or k in rec or v is None or v == "":
            continue
        sv = str(v).strip()
        if not sv:
            continue
        if k.lower() in PATH_LIKE_KEYS:
            if _INSDC_RUN_RE.match(sv):
                rec["accession"] = sv.upper()
            continue
        if ("/" in sv or "\\" in sv) and not sv.startswith(("http://", "https://", "ftp://")):
            continue
        rec[k] = v
    return rec


def _write_site_nocalls_parquet(
    out_pq: Path,
    nocall_rows: list[tuple[str, int, str]],
) -> None:
    """Write `site_nocalls.parquet` (`chrom`, `pos`, `sample_id`) for uncovered/no-call samples at cohort sites (#62)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("sample_id", pa.string()),
        ]
    )
    if nocall_rows:
        nocall_rows.sort(key=lambda r: (r[0], r[1], r[2]))
        tbl = pa.Table.from_pydict(
            {
                "chrom": [r[0] for r in nocall_rows],
                "pos": [int(r[1]) for r in nocall_rows],
                "sample_id": [r[2] for r in nocall_rows],
            },
            schema=schema,
        )
    else:
        tbl = pa.Table.from_pydict({"chrom": [], "pos": [], "sample_id": []}, schema=schema)
    pq.write_table(tbl, str(out_pq), compression="zstd")


def _load_sample_covered_intervals(
    cov_npz: Path | None,
    cov_tsv: Path | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if cov_npz is not None and cov_npz.is_file():
        with np.load(cov_npz) as z:
            for k in z.files:
                if k.startswith("s_"):
                    cn = k[2:]
                    ek = f"e_{cn}"
                    if ek in z.files:
                        out[cn] = (np.asarray(z[k], dtype=np.int32), np.asarray(z[ek], dtype=np.int32))
        return out
    if cov_tsv is not None and cov_tsv.is_file():
        by_c: dict[str, tuple[list[int], list[int]]] = {}
        with open(cov_tsv, "r", encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    s_l, e_l = by_c.setdefault(parts[0], ([], []))
                    s_l.append(int(parts[1]))
                    e_l.append(int(parts[2]))
        for cn, (s_l, e_l) in by_c.items():
            out[cn] = (np.asarray(s_l, dtype=np.int32), np.asarray(e_l, dtype=np.int32))
    return out


def _is_pos0_covered(cov_map: dict[str, tuple[np.ndarray, np.ndarray]], chrom: str, pos0: int) -> bool:
    ivs = cov_map.get(chrom)
    if ivs is None:
        return False
    s_arr, e_arr = ivs
    if len(e_arr) == 0:
        return False
    idx = int(np.searchsorted(e_arr, pos0, side="right"))
    return idx < len(s_arr) and int(s_arr[idx]) <= pos0


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
    dedup_mode: str,
    allow_local_manifest: bool,
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
    seen_sids: set[str] = set()

    for idx_r, row in enumerate(rows_raw):
        sid = _clean_sample_name(_pick_col(row, ("sample_id", "Sample", "sample"), f"isolate_{idx_r + 1}"), fallback=f"isolate_{idx_r + 1}")
        if sid in seen_sids:
            raise ValueError(f"Duplicate sample_id {sid!r} in surveillance manifest")
        seen_sids.add(sid)
        grp = _pick_col(row, ("country_or_group", "Country", "group", "Population"), "cohort")
        pop = _pick_col(row, ("population", "Population"), grp)
        src = _pick_col(row, ("source_uri_or_srr", "cram_url", "accession", "ena_runs", "source", "url"))
        crai_u = _pick_col(row, ("crai_uri", "crai_uri_optional", "crai_url", "crai"))
        stype = _pick_col(row, ("sample_type", "Sample_type"), "gDNA")
        year_s = _pick_col(row, ("year", "Year"), "2020")
        year = int(year_s) if year_s.isdigit() else 2020

        from_direct = bool(row.get("_from_direct_args"))
        eff_allow_row = allow_local_manifest or from_direct

        if not src and sid in local_map:
            src = sid
        if not src:
            raise ValueError(f"Isolate {sid!r}: missing source_uri_or_srr")
        _assert_manifest_uri_allowed(src, sid, "source", allow_local=eff_allow_row, local_map=local_map)
        if crai_u:
            _assert_manifest_uri_allowed(crai_u, sid, "crai", allow_local=eff_allow_row, local_map=local_map)

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
                    "_allow_local": eff_allow_row,
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
                    if mate2:
                        _assert_manifest_uri_allowed(mate2, sid, "mate2", allow_local=eff_allow_row, local_map=local_map)
                    fq_src = f"{src};{mate2}" if mate2 else src
                    u1, u2 = _match_ena_paired_urls(fq_src.split(";"))
                    stream_jobs.append(
                        {
                            "Sample": sid,
                            "Sample_type": stype,
                            "ena_runs": [],
                            "all_odp": False,
                            "all_paired_fastq": bool(u1 and u2),
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
            for cn, s0, e0 in _validate_and_parse_bed(targets_bed, dict(zip(g.names, [int(x) for x in g.lengths]))):
                regs.append((cidx[cn], s0 + 1, e0, f"bed:{cn}:{s0}"))
        FC._init_worker(use_omnitig_zoom=False)
        FC._WS["cp"].dedup = (dedup_mode != "off")
        crai_cache = work_dir / "crai"
        dl_pool = ThreadPoolExecutor(max_workers=streams, thread_name_prefix="crai-range")
        try:
            for idx_j, fj in enumerate(focused_jobs, 1):
                sid = fj["Sample"]
                url = fj["cram_url"]
                crai_local = _fetch_crai_file(fj["crai_uri"], crai_cache / f"{sid}.cram.crai", allow_local=bool(fj.get("_allow_local", True)))
                cram_bytes = fj["cram_bytes"]
                is_local_u = (not url.startswith(("http://", "https://", "ftp://"))) and bool(fj.get("_allow_local")) and Path(url).is_file()
                if not cram_bytes:
                    cram_bytes = Path(url).stat().st_size if is_local_u else _head_content_length(url)
                    fj["cram_bytes"] = cram_bytes
                hdr_end, rr, nc, spans = FC.plan_ranges(crai_local, regs, cram_size=cram_bytes)
                staged_cram = FC.STAGE / f"{sid}.cram"
                if is_local_u:
                    with open(url, "rb") as fi, open(staged_cram, "wb") as fo:
                        hdr_b = fi.read(hdr_end)
                        if len(hdr_b) != hdr_end:
                            raise RuntimeError(f"Short read on local CRAM header for {sid}")
                        fo.write(hdr_b)
                        for a, b in rr:
                            fi.seek(a)
                            blk = fi.read(b - a)
                            if len(blk) != (b - a):
                                raise RuntimeError(f"Truncated local CRAM range [{a}, {b}) for {sid}")
                            fo.write(blk)
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
                if int(tel.get("reads_counted") or 0) <= 0:
                    raise RuntimeError(f"Focused CRAM isolate {sid!r}: zero usable reads counted")
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
        for idx_s, srow in enumerate(stream_jobs, 1):
            sid = srow["Sample"]
            cfg = {
                "map_threads": max(2, min(threads, 6)),
                "xsra_threads": max(2, min(threads // 2, 4)),
                "prefer": "odp_stream" if bool(srow.get("all_odp")) else "ena",
                "store_candidates": False,
                "mapper": chosen_mapper,
                "min_depth": float(target_depth),
                "min_depth_poly": float(max(target_depth + 8.0, target_depth * 1.25)),
                "use_pla_sieve": use_pla,
                "pla_sieve_path": pla_path,
                "depletion_need": 2 if use_host_depletion else 0,
                "dedup": (dedup_mode != "off"),
                "use_omnitig_zoom": True,
                "sample_timeout_s": float(sample_timeout_s),
            }
            tel = E.run_sample(srow, str(cohort_dir), cfg)
            if tel.get("stop_reason") == "error":
                raise RuntimeError(f"WGS streaming failed for {sid}: {tel.get('error')}")
            if int(tel.get("reads_counted") or 0) <= 0:
                raise RuntimeError(f"WGS streaming isolate {sid!r}: zero usable reads counted")
            pcr_d = int(tel.get("pcr_dups_skipped") or 0)
            sam_r = int(tel.get("sam_records") or 0)
            if dedup_mode != "off" and sam_r >= 100 and pcr_d > 0.95 * sam_r:
                _log(
                    f"WARNING: Isolate {sid!r} had {100.0 * pcr_d / max(sam_r, 1):.1f}% PCR duplicates skipped. "
                    "Use --dedup off (--dedup-bits 0) for targeted PCR amplicon libraries."
                )
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
    contig_names = list(g.names)
    if not calls_df.empty:
        calls_df = calls_df[calls_df["gt"] > 0].copy()
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

    # Build site_nocalls.parquet from callable_mask intervals vs cohort called variant sites (#62)
    callable_all = SS.read_table(cohort_dir, "callable_mask")
    nocalls_pf: list[tuple[str, int, str]] = []
    if not calls_df.empty:
        cohort_sites_pf = sorted({(str(c), int(p)) for c, p in zip(calls_df["chrom"], calls_df["pos"])})
        called_set_pf = {(str(c), int(p), str(s)) for c, p, s in zip(calls_df["chrom"], calls_df["pos"], calls_df["sample_id"])}
        cov_by_s_pf: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        if not callable_all.empty:
            for s_id, grp_c in callable_all.groupby("sample"):
                m_c: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for c_idx, sub_c in grp_c.groupby("chrom"):
                    cn = contig_names[int(c_idx)] if 0 <= int(c_idx) < len(contig_names) else str(c_idx)
                    sub_sorted = sub_c.sort_values("start")
                    s_arr = (sub_sorted["start"].to_numpy(dtype=np.int32) - 1)
                    e_arr = sub_sorted["end"].to_numpy(dtype=np.int32)
                    m_c[cn] = (s_arr, e_arr)
                cov_by_s_pf[str(s_id)] = m_c
        for s_id in all_samples:
            cm_s = cov_by_s_pf.get(s_id, {})
            for ch, p1 in cohort_sites_pf:
                if (ch, p1, s_id) not in called_set_pf and not _is_pos0_covered(cm_s, ch, p1 - 1):
                    nocalls_pf.append((ch, p1, s_id))
    _write_site_nocalls_parquet(state_root / "site_nocalls.parquet", nocalls_pf)

    eff_mode = "focused_cram+wgs_stream" if (focused_jobs and stream_jobs) else ("focused_cram" if focused_jobs else "wgs_stream")
    summary_stats = {"mode": eff_mode, "reference_label": "Pf3D7 (PlasmoDB-54)", "n_isolates": len(drm_df), "wall_s": round(time.time() - t_start, 2), "warm": warm_info}
    render_variationloop_report(out_report, analysis_mode=eff_mode, summary_stats=summary_stats, variants_df=calls_df, drm_df=drm_df)

    store = FocaloopStateStore(state_root)
    HypothesisLedger.from_store(store, alpha_budget=0.05)
    drm_df.to_parquet(state_root / "drm_calls.parquet", index=False)
    tel_by_sample = tel_df.drop_duplicates("sample", keep="last").set_index("sample").to_dict(orient="index") if not tel_df.empty else {}
    state_meta = {
        "engine": "malariavar",
        "version": "0.4.0",
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
                "reads_counted": int((tel_by_sample.get(r["Sample"]) or {}).get("reads_counted") or 0),
                "median_depth": round(float((tel_by_sample.get(r["Sample"]) or {}).get("median_core_depth") or 0.0), 2),
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
    analysis_mode: str = "auto",
    out_variants: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    out_vcf: str | Path | None = None,
    out_drm: str | Path | None = None,
    manifest_path: str | Path | None = None,
    ref_fasta: str | Path | None = None,
    ncbi_accession: str | None = None,
    sra_accessions: str | None = None,
    logan_mode: str = "auto",
    logan_min_ka: str = "adapt",
    logan_min_wsaf: float = 0.25,
    ploidy: str = "diploid",
    min_wsaf: float | None = None,
    targets_bed: str | Path | None = None,
    cram_pad_bp: int = 1000,
    reads: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
    reads_r1: str | Path | Sequence[str | Path] | None = None,
    reads_r2: str | Path | Sequence[str | Path] | None = None,
    bam_path: str | Path | Sequence[str | Path] | None = None,
    bam_ids: Sequence[str] | None = None,
    crai_path: str | Path | Sequence[str | Path] | None = None,
    sample_name: str | None = None,
    target_depth: float = 24.0,
    subsample_fraction: float = 1.0,
    mapper: str = "auto",
    threads: int = 8,
    streams: int = 24,
    sample_timeout_s: float = 420.0,
    use_host_depletion: bool = True,
    host_depletion_mode: str | None = None,
    enable_reassembly: bool = True,
    min_mapq: int = 20,
    min_bq: int = 13,
    snv_min_qual: float = 20.0,
    indel_min_qual: float = 15.0,
    dedup_bits: int | None = None,
    dedup: str | bool | None = None,
    allow_local_manifest_paths: bool | None = None,
    reference_label: str | None = None,
    surveillance_panel: str = "none",
) -> dict[str, Any]:
    """Execute Focaloop VariationLoop across paired collections, BAM/CRAM, indexed CRAM slicing, target-filtered FASTQ, or Logan unitigs."""
    if analysis_mode not in ("auto", "focused_cram", "wgs_stream", "diploid_wgs", "logan_unitigs"):
        raise ValueError(f"Unsupported analysis_mode {analysis_mode!r}")
    eff_logan_mode = str(logan_mode or "auto").strip().lower()
    if eff_logan_mode not in ("auto", "logan_only", "off"):
        raise ValueError(f"Unsupported logan_mode {logan_mode!r}; expected 'auto', 'logan_only', or 'off'")
    eff_ploidy = str(ploidy or "diploid").strip().lower()
    if eff_ploidy not in ("haploid", "diploid"):
        raise ValueError(f"Unsupported ploidy {ploidy!r}; expected 'haploid' or 'diploid'")
    eff_min_wsaf = max(0.05, min(0.99, float(min_wsaf if min_wsaf is not None else logan_min_wsaf)))

    # Resolve PCR duplicate removal policy (#71, L43)
    raw_dedup = os.environ.get("FOCALOOP_DEDUP_MODE", "auto") if dedup is None else dedup
    if raw_dedup is False or (isinstance(raw_dedup, str) and raw_dedup.strip().lower() in ("off", "false", "0", "no")) or dedup_bits == 0:
        eff_dedup_mode = "off"
    elif raw_dedup is True or (isinstance(raw_dedup, str) and raw_dedup.strip().lower() in ("on", "true", "1", "yes")):
        eff_dedup_mode = "on"
    elif isinstance(raw_dedup, str) and raw_dedup.strip().lower() == "auto":
        eff_dedup_mode = "auto"
    else:
        raise ValueError(f"Unsupported dedup setting {dedup!r} (expected 'auto', 'on', or 'off')")

    # Resolve manifest local-path policy (#72): block arbitrary local filesystem paths unless explicitly opted in
    if allow_local_manifest_paths is None:
        eff_allow_local_manifest = os.environ.get("FOCALOOP_ALLOW_LOCAL_MANIFEST_PATHS", "").strip() == "1"
        allow_logan_local_unitigs = (analysis_mode == "logan_unitigs")
    else:
        eff_allow_local_manifest = bool(allow_local_manifest_paths)
        allow_logan_local_unitigs = bool(allow_local_manifest_paths)

    if host_depletion_mode is not None:
        eff_dep_mode = str(host_depletion_mode).strip().lower()
        if eff_dep_mode not in ("auto", "on", "off"):
            raise ValueError(f"Unsupported host_depletion_mode {host_depletion_mode!r}; expected 'auto', 'on', or 'off'")
    elif analysis_mode in ("wgs_stream", "logan_unitigs"):
        eff_dep_mode = "on" if use_host_depletion else "off"
    else:
        eff_dep_mode = "off"

    t_start = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix="focaloop_varloop_"))
    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    out_variants_p = Path(out_variants)
    out_report_p = Path(out_report)
    out_vcf_p = Path(out_vcf) if out_vcf else None
    out_drm_p = Path(out_drm) if out_drm else None

    r1_list = _as_str_list(reads_r1)
    r2_list = _as_str_list(reads_r2)
    bam_list = _as_str_list(bam_path)
    crai_list = _as_str_list(crai_path, keep_empty=True)
    read_id_list = _as_str_list(read_ids)
    bam_id_list = _as_str_list(bam_ids)

    if r1_list and r2_list and len(r1_list) != len(r2_list):
        raise ValueError(f"Mismatched paired FASTQ counts: {len(r1_list)} R1 files vs {len(r2_list)} R2 files")
    if r1_list and read_id_list and len(read_id_list) != len(r1_list):
        raise ValueError(f"Mismatched --read-ids count ({len(read_id_list)}) vs --reads-r1 count ({len(r1_list)})")
    if bam_list and bam_id_list and len(bam_id_list) != len(bam_list):
        raise ValueError(f"Mismatched --bam-ids count ({len(bam_id_list)}) vs --bam count ({len(bam_list)})")

    eff_single_sample = _clean_sample_name(
        sample_name
        or (read_id_list[0] if read_id_list else None)
        or (bam_id_list[0] if bam_id_list else None)
        or (r1_list[0] if r1_list else None)
        or (bam_list[0] if bam_list else None),
        fallback="sample_1",
        strip_mate_suffix=not bool(r1_list and read_id_list),
    )

    local_map = _build_local_reads_map(reads, read_ids)
    r1_dedup_sids: list[str] = []
    if r1_list:
        raw_r1_sids = [
            _clean_sample_name(
                sample_name
                if (len(r1_list) == 1 and sample_name)
                else (read_id_list[i] if i < len(read_id_list) else r1_s),
                fallback=f"sample_{i + 1}",
                strip_mate_suffix=not (i < len(read_id_list)),
            )
            for i, r1_s in enumerate(r1_list)
        ]
        r1_dedup_sids = _dedup_sample_names(raw_r1_sids)
        for i, (sid_final, r1_s) in enumerate(zip(r1_dedup_sids, r1_list)):
            r1_p = Path(r1_s).resolve()
            pair = [r1_p] + ([Path(r2_list[i]).resolve()] if i < len(r2_list) else [])
            local_map[sid_final] = pair

    try:
        if ncbi_accession and str(ncbi_accession).strip():
            from .ncbi_ref import fetch_ncbi_assembly

            acc_ref = str(ncbi_accession).strip()
            _log(f"Fetching NCBI reference assembly {acc_ref}...")
            fetched = fetch_ncbi_assembly(acc_ref, work_dir / "ncbi_ref", need_annotation=False)
            ref_fasta = fetched["fasta"]
            if not reference_label:
                reference_label = acc_ref

        # Specialized P. falciparum surveillance branch only when explicitly opted in (or legacy call with no ref_fasta)
        use_pf3d7 = (
            analysis_mode in ("focused_cram", "wgs_stream")
            and (surveillance_panel == "pf3d7" or ref_fasta is None)
            and _pf3d7_assets_ready(ref_fasta)
        )
        if use_pf3d7:
            if manifest_path and not (r1_list or bam_list):
                rows_raw = _read_manifest_table(manifest_path)
            elif bam_list and analysis_mode == "focused_cram":
                bam_sids_pf = _dedup_sample_names([
                    _clean_sample_name(
                        sample_name if (len(bam_list) == 1 and sample_name) else (bam_id_list[i] if i < len(bam_id_list) else bp),
                        fallback=f"sample_{i + 1}",
                        strip_mate_suffix=False,
                    )
                    for i, bp in enumerate(bam_list)
                ])
                rows_raw = [
                    {
                        "sample_id": bam_sids_pf[i],
                        "country_or_group": "cohort",
                        "source_uri_or_srr": str(Path(bp).resolve()),
                        "crai_uri": str(Path(crai_list[i]).resolve()) if (i < len(crai_list) and crai_list[i]) else "",
                        "_from_direct_args": "1",
                    }
                    for i, bp in enumerate(bam_list)
                ]
            elif r1_list or (reads and len(reads) >= 1):
                if r1_list:
                    rows_raw = [
                        {"sample_id": sid, "country_or_group": "cohort", "source_uri_or_srr": sid, "_from_direct_args": "1"}
                        for sid in r1_dedup_sids
                    ]
                else:
                    pair_paths = [Path(p).resolve() for p in reads[:2]]
                    local_map[eff_single_sample] = pair_paths
                    rows_raw = [{"sample_id": eff_single_sample, "country_or_group": "cohort", "source_uri_or_srr": eff_single_sample, "_from_direct_args": "1"}]
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
                use_host_depletion=(eff_dep_mode != "off"),
                dedup_mode=eff_dedup_mode,
                allow_local_manifest=eff_allow_local_manifest,
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
        g = G.from_fasta(eff_ref, bed=targets_bed, need_core=False)
        ctg_lens = dict(zip(g.names, [int(x) for x in g.lengths]))
        bed_regions = _validate_and_parse_bed(targets_bed, ctg_lens, ref_label=Path(ref_fasta).name) if targets_bed else []

        # Optional group/metadata table when paired collections or BAM/CRAM lists also pass --manifest (L36)
        group_lookup: dict[str, tuple[str, dict[str, str]]] = {}
        if manifest_path and (r1_list or bam_list):
            gs_rows = _read_manifest_table(manifest_path)
            gs_cols_low = {k.lower() for k in gs_rows[0].keys()}
            if not (gs_cols_low & {"sample_id", "sample"}):
                raise ValueError(
                    f"Group sheet {Path(manifest_path).name!r} is missing a required 'sample_id' (or 'sample') column"
                )
            if not (gs_cols_low & {"country_or_group", "population", "group", "condition"}):
                raise ValueError(
                    f"Group sheet {Path(manifest_path).name!r} is missing a required 'group' (or 'population'/'condition') column"
                )
            for row in gs_rows:
                raw_sid = _pick_col(row, ("sample_id", "Sample", "sample"), "")
                grp = _pick_col(row, ("country_or_group", "population", "group", "condition"), "")
                if not raw_sid or not grp:
                    raise ValueError(f"Group sheet {Path(manifest_path).name!r} contains an empty sample_id or group cell")
                group_lookup[raw_sid] = (grp, row)
                group_lookup[_clean_sample_name(raw_sid, fallback=raw_sid, strip_mate_suffix=False)] = (grp, row)
                group_lookup[_clean_sample_name(raw_sid, fallback=raw_sid, strip_mate_suffix=True)] = (grp, row)

        # Build sample jobs uniformly from paired collections, BAM/CRAM lists, manifest, or SRA accessions
        jobs: list[dict[str, Any]] = []
        if r1_list:
            matched_gs = 0
            for i, (sid, r1_s) in enumerate(zip(r1_dedup_sids, r1_list)):
                raw_key = read_id_list[i] if i < len(read_id_list) else sid
                hit = group_lookup.get(raw_key) or group_lookup.get(sid)
                if hit is not None:
                    matched_gs += 1
                    grp, extra_row = hit
                else:
                    grp, extra_row = ("cohort", {})
                r1_res = str(Path(r1_s).resolve())
                r2_res = str(Path(r2_list[i]).resolve()) if i < len(r2_list) else ""
                jobs.append({
                    "sample": sid, "group": grp, "src": r1_res, "mate2": r2_res, "crai": "",
                    "row": extra_row, "from_direct_args": True, "direct_pair": (r1_res, r2_res),
                })
            if group_lookup and matched_gs == 0:
                raise ValueError(
                    f"Group sheet {Path(manifest_path).name!r} matched 0 collection samples "
                    f"(collection IDs: {r1_dedup_sids[:5]})"
                )
        elif bam_list:
            raw_sids = [
                _clean_sample_name(
                    sample_name if (len(bam_list) == 1 and sample_name) else (bam_id_list[i] if i < len(bam_id_list) else bp),
                    fallback=f"sample_{i + 1}",
                    strip_mate_suffix=False,
                )
                for i, bp in enumerate(bam_list)
            ]
            sids = _dedup_sample_names(raw_sids)
            matched_gs = 0
            for i, (sid, bp) in enumerate(zip(sids, bam_list)):
                raw_key = bam_id_list[i] if i < len(bam_id_list) else sid
                hit = group_lookup.get(raw_key) or group_lookup.get(sid)
                if hit is not None:
                    matched_gs += 1
                    grp, extra_row = hit
                else:
                    grp, extra_row = ("cohort", {})
                crai_s = str(Path(crai_list[i]).resolve()) if (i < len(crai_list) and crai_list[i]) else ""
                jobs.append({
                    "sample": sid, "group": grp, "src": str(Path(bp).resolve()), "mate2": "", "crai": crai_s,
                    "row": extra_row, "from_direct_args": True,
                })
            if group_lookup and matched_gs == 0:
                raise ValueError(
                    f"Group sheet {Path(manifest_path).name!r} matched 0 BAM/CRAM samples "
                    f"(sample IDs: {sids[:5]})"
                )
        elif manifest_path:
            seen_man_sids: set[str] = set()
            for idx_r, row in enumerate(_read_manifest_table(manifest_path)):
                sid = _clean_sample_name(_pick_col(row, ("sample_id", "Sample", "sample"), f"sample_{idx_r + 1}"), fallback=f"sample_{idx_r + 1}")
                if sid in seen_man_sids:
                    raise ValueError(f"Duplicate sample_id {sid!r} in manifest {Path(manifest_path).name!r}")
                seen_man_sids.add(sid)
                grp = _pick_col(row, ("country_or_group", "population", "group", "condition"), "cohort")
                src = _pick_col(row, ("source_uri_or_srr", "cram_url", "bam_url", "run_or_url", "fastq_1", "accession", "source", "url"))
                crai_u = _pick_col(row, ("crai_uri", "crai_uri_optional", "crai_url", "crai"))
                mate2 = _pick_col(row, ("mate2_url", "fastq_2", "r2", "mate2"))
                if not src and sid in local_map:
                    src = sid
                if not src:
                    raise ValueError(f"Sample {sid!r}: missing input source column in manifest")
                allow_row_local = eff_allow_local_manifest or allow_logan_local_unitigs
                _assert_manifest_uri_allowed(src, sid, "source", allow_local=allow_row_local, local_map=local_map)
                if mate2:
                    _assert_manifest_uri_allowed(mate2, sid, "mate2", allow_local=eff_allow_local_manifest, local_map=local_map)
                if crai_u:
                    _assert_manifest_uri_allowed(crai_u, sid, "crai", allow_local=eff_allow_local_manifest, local_map=local_map)
                jobs.append({"sample": sid, "group": grp, "src": src, "mate2": mate2, "crai": crai_u, "row": row, "from_direct_args": False})
        elif sra_accessions and str(sra_accessions).strip():
            raw_txt = str(sra_accessions).replace("__cn__", " ").replace("__cr__", " ").replace("__tt__", " ")
            tokens = [t.strip().upper() for t in re.split(r"[\s,;]+", raw_txt.strip()) if t.strip()]
            seen_toks: set[str] = set()
            for tok in tokens:
                if tok in seen_toks:
                    continue
                seen_toks.add(tok)
                jobs.append({"sample": tok, "group": "cohort", "src": tok, "mate2": "", "crai": "", "row": {}, "from_direct_args": False})
        elif reads:
            if len(reads) == 2 and Path(reads[0]).exists() and Path(reads[1]).exists():
                r1_res = str(Path(reads[0]).resolve())
                r2_res = str(Path(reads[1]).resolve())
                jobs.append(
                    {
                        "sample": eff_single_sample,
                        "group": "cohort",
                        "src": r1_res,
                        "mate2": r2_res,
                        "crai": "",
                        "row": {},
                        "from_direct_args": True,
                        "direct_pair": (r1_res, r2_res),
                    }
                )
            else:
                raw_read_sids = [
                    _clean_sample_name(read_ids[idx_r] if read_ids and idx_r < len(read_ids) else Path(rp).stem, fallback=f"sample_{idx_r + 1}")
                    for idx_r, rp in enumerate(reads)
                ]
                ded_read_sids = _dedup_sample_names(raw_read_sids)
                for sid, rp in zip(ded_read_sids, reads):
                    jobs.append({"sample": sid, "group": "cohort", "src": str(Path(rp).resolve()), "mate2": "", "crai": "", "row": {}, "from_direct_args": True})
        else:
            raise ValueError("Provide a paired FASTQ collection, paired R1/R2 FASTQ files, BAM/CRAM file(s), public SRA accessions, or a cohort manifest")

        total_genome_bp = sum(ctg_lens.values())
        if eff_dedup_mode == "off":
            eff_dedup_bits = 0
        elif dedup_bits is None:
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
            ploidy=eff_ploidy,
            haploid_min_wsaf=eff_min_wsaf,
        )
        chosen_mapper: str | None = None
        sample_tels: list[dict[str, Any]] = []
        per_sample_vcfs: list[Path] = []
        used_engines: set[str] = set()
        p_frac = float(np.clip(subsample_fraction, 1e-4, 1.0))

        # 0.4.1 Logan unitig probe, 400 Mb span gate (#41), RAM-aware aligner (#44), & optional k=15 bitset
        logan_avail: set[str] = set()
        k15_bit_path: Path | None = None
        logan_pileup_regs = _pad_and_merge_regions(bed_regions, ctg_lens, pad_bp=500) if bed_regions else [(cn, 0, ln) for cn, ln in ctg_lens.items()]
        logan_pileup_span = sum(max(0, e - s) for _, s, e in logan_pileup_regs)
        logan_aligner_selected = "bwa"

        if analysis_mode == "logan_unitigs" or eff_logan_mode in ("auto", "logan_only"):
            from .logan import is_insdc_run, probe_logan_runs
            from .malariavar import logan_var as LV

            cand_accs = [
                j["src"].strip().upper()
                for j in jobs
                if is_insdc_run(j["src"]) and j["src"] not in local_map and j["sample"] not in local_map
            ]
            if cand_accs and logan_pileup_span > 400_000_000 and analysis_mode == "auto" and eff_logan_mode == "auto":
                _log(
                    f"Reference pileup span ({logan_pileup_span / 1e6:.1f} Mb) exceeds the 400 Mb Logan pileup limit "
                    "without --targets-bed, falling back automatically to raw read streaming"
                )
                cand_accs = []
            elif (cand_accs or analysis_mode == "logan_unitigs") and logan_pileup_span > 400_000_000:
                raise ValueError(
                    f"Logan unitig calling on reference span {logan_pileup_span / 1e6:.1f} Mb (> 400 Mb) requires "
                    "--targets-bed restricting target intervals to <= 400 Mb (or switch Logan mode to Automatic)"
                )

            if cand_accs:
                lprobe = probe_logan_runs(cand_accs, kind="unitigs", max_workers=min(32, max(8, int(threads))))
                logan_avail = set(lprobe["logan_runs"])
                _log(
                    f"Logan probe: {len(logan_avail)}/{len(lprobe['probes'])} accessions available on logan-pub/u/ "
                    f"({lprobe['total_logan_bytes'] / 1e6:.1f} MB compressed)"
                )
                if (analysis_mode == "logan_unitigs" or eff_logan_mode == "logan_only") and lprobe["fallback_runs"]:
                    missing = ", ".join(lprobe["fallback_runs"])
                    raise ValueError(
                        f"Logan unitigs are not available for accession(s): {missing}. "
                        "Switch Logan mode to Automatic to fall back to raw read streaming."
                    )
            if logan_avail or analysis_mode == "logan_unitigs":
                prep_bin, _ = LV.ensure_var_logan_bins()
                try:
                    _aln_name, _aln_bin = LV._select_var_aligner(mapper, ref_bp=total_genome_bp)
                except TypeError:
                    _aln_name, _aln_bin = LV._select_var_aligner(mapper)
                idx_logan_ref = LV.ensure_var_aligner_index(
                    _aln_name, _aln_bin, eff_ref, work_dir, _log, allow_bwa_fallback=(mapper == "auto")
                )
                if (
                    _aln_name == "bwa-mem2"
                    and isinstance(idx_logan_ref, Path)
                    and not Path(str(idx_logan_ref) + ".bwt.2bit.64").is_file()
                    and Path(str(idx_logan_ref) + ".bwt").is_file()
                ):
                    logan_aligner_selected = "bwa"
                else:
                    logan_aligner_selected = _aln_name
                logan_span = sum(max(0, e - s) for _, s, e in bed_regions) if bed_regions else total_genome_bp
                want_k15 = (eff_dep_mode == "on") or (eff_dep_mode == "auto" and logan_span <= 50_000_000)
                if want_k15:
                    k15_bit_path = LV.build_k15_reference_bitset(
                        eff_ref,
                        work_dir / "logan_k15.bit",
                        prep_bin=prep_bin,
                        targets_bed=Path(targets_bed) if targets_bed else None,
                        log=_log,
                    )

        def _job_allows_local(j_dict: dict[str, Any]) -> bool:
            return bool(j_dict.get("from_direct_args")) or eff_allow_local_manifest

        # Optional host/off-target 31-mer retention filter for raw FASTQ calling (padded +-150 bp around narrow BEDs, #28)
        kset_tab = None
        has_raw_fq_jobs = any(
            not (j["src"].strip().upper() in logan_avail and j["src"] not in local_map and j["sample"] not in local_map)
            and not (analysis_mode == "logan_unitigs" and allow_logan_local_unitigs and Path(j["src"]).is_file())
            and not (
                j["src"].split("?", 1)[0].lower().endswith((".bam", ".cram"))
                or (_job_allows_local(j) and not j["src"].startswith(("http://", "https://")) and Path(j["src"]).is_file() and _detect_aln_format(Path(j["src"])) in ("bam", "cram"))
            )
            for j in jobs
        )
        if has_raw_fq_jobs and eff_dep_mode != "off":
            regs_k = _pad_and_merge_regions(bed_regions, ctg_lens, pad_bp=150) if bed_regions else [(cn, 0, ln) for cn, ln in ctg_lens.items()]
            target_span = sum(max(0, e - s) for _, s, e in regs_k)
            want_fq_kset = (eff_dep_mode == "on") or (eff_dep_mode == "auto" and bool(bed_regions) and target_span <= 45_000_000)
            if want_fq_kset:
                if target_span > 45_000_000:
                    raise ValueError(
                        f"Target 31-mer read filter span ({target_span / 1e6:.1f} Mb) exceeds the 45 Mb in-memory limit. "
                        "Provide a target BED file covering your regions of interest or set Host/off-target pre-filtering to Off."
                    )
                if target_span < 500_000:
                    bits_k = 20
                elif target_span < 8_000_000:
                    bits_k = 24
                else:
                    bits_k = 26
                kset_tab, kmeta = VG.build_target_kset(g, regs_k, bits=bits_k)
                if int(kmeta["inserted_kmers"]) == 0:
                    _log(
                        "Warning: target intervals yielded 0 non-repetitive 31-mers — skipping read pre-filter "
                        "so target alignments are not discarded"
                    )
                    kset_tab = None
                else:
                    _log(f"Built target 31-mer pre-filter table ({kmeta['inserted_kmers']:,} k-mers across {target_span / 1e6:.2f} Mb)")

        cov_dir = work_dir / "cov"
        cov_dir.mkdir(parents=True, exist_ok=True)
        cov_files_by_idx: dict[int, tuple[Path | None, Path | None]] = {}

        logan_indexed: list[tuple[int, dict[str, Any], Path]] = []
        other_indexed: list[tuple[int, dict[str, Any], Path]] = []
        for idx_j, job in enumerate(jobs, 1):
            sid = job["sample"]
            src = job["src"]
            s_vcf_gz = work_dir / "vcfs" / f"{idx_j:04d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', sid)}.vcf.gz"
            src_u = src.strip().upper()
            use_logan_for_job = (
                (src_u in logan_avail and src not in local_map and sid not in local_map)
                or (analysis_mode == "logan_unitigs" and (allow_logan_local_unitigs or _job_allows_local(job)) and not src.startswith(("http://", "https://")) and Path(src).exists())
            )
            if use_logan_for_job:
                logan_indexed.append((idx_j, job, s_vcf_gz))
            else:
                other_indexed.append((idx_j, job, s_vcf_gz))

        results_by_idx: dict[int, tuple[dict[str, Any], Path, str]] = {}
        log_lock = threading.Lock()

        if logan_indexed:
            from .malariavar import logan_var as LV

            avail_ram = LV.available_ram_bytes()
            bed_loaded_bp = sum(ctg_lens[cn] for cn in {c for c, _, _ in bed_regions}) if bed_regions else total_genome_bp
            idx_ram_per_w = (6 * total_genome_bp) if logan_aligner_selected == "bwa-mem2" else (2 * total_genome_bp)
            per_w_ram = idx_ram_per_w + bed_loaded_bp + 20 * logan_pileup_span + (256 << 20)
            ram_cap_w = max(1, int((0.75 * avail_ram) // max(per_w_ram, 256 << 20)))
            logan_workers = min(len(logan_indexed), max(1, int(threads) // 4), ram_cap_w)
            logan_per_threads = max(1, int(threads) // logan_workers)
            if logan_workers > 1:
                _log(
                    f"Running {len(logan_indexed)} Logan samples across {logan_workers} concurrent workers "
                    f"({logan_per_threads} aligner threads/worker)"
                )

            def _run_logan_item(item: tuple[int, dict[str, Any], Path]) -> tuple[int, dict[str, Any], Path, str, Path]:
                idx_j, job, s_vcf_gz = item
                sid = job["sample"]
                src = job["src"]
                src_u = src.strip().upper()
                is_local_u = (
                    analysis_mode == "logan_unitigs"
                    and (allow_logan_local_unitigs or _job_allows_local(job))
                    and not src.startswith(("http://", "https://"))
                    and Path(src).exists()
                )
                local_u = Path(src) if is_local_u else None
                cov_tsv = cov_dir / f"{idx_j:04d}.cov.tsv"
                tel = LV.run_varloop_logan_unitigs(
                    acc=src_u,
                    sample=sid,
                    ref_fa=eff_ref,
                    ctg_lens=ctg_lens,
                    out_vcf_gz=s_vcf_gz,
                    work_dir=work_dir,
                    targets_bed=Path(targets_bed) if targets_bed else None,
                    k15_bit=k15_bit_path,
                    mapper=mapper,
                    threads=logan_per_threads,
                    min_mapq=int(min_mapq),
                    snv_min_qual=float(snv_min_qual),
                    indel_min_qual=float(indel_min_qual),
                    min_ka_mode=str(logan_min_ka or "adapt"),
                    min_wsaf=eff_min_wsaf,
                    ploidy=eff_ploidy,
                    local_unitigs_path=local_u,
                    out_cov_tsv=cov_tsv,
                    log=lambda m: _log(f"  [{idx_j}/{len(jobs)}] {m}"),
                )
                with log_lock:
                    _log(
                        f"  [{idx_j}/{len(jobs)}] {sid} (Logan): {tel.get('n_variants', 0)} variants, "
                        f"{tel.get('wire_bytes', 0) / 1e6:.2f} MB wire ({tel.get('wall_s', 0):.2f}s)"
                    )
                return idx_j, tel, s_vcf_gz, sid, cov_tsv

            if logan_workers > 1:
                with ThreadPoolExecutor(max_workers=logan_workers) as ex:
                    for idx_j, tel, s_vcf_gz, sid, cov_tsv in ex.map(_run_logan_item, logan_indexed):
                        results_by_idx[idx_j] = (tel, s_vcf_gz, sid)
                        cov_files_by_idx[idx_j] = (None, cov_tsv)
            else:
                for item in logan_indexed:
                    idx_j, tel, s_vcf_gz, sid, cov_tsv = _run_logan_item(item)
                    results_by_idx[idx_j] = (tel, s_vcf_gz, sid)
                    cov_files_by_idx[idx_j] = (None, cov_tsv)
            used_engines.add("logan_unitigs")

        for idx_j, job, s_vcf_gz in other_indexed:
            sid = job["sample"]
            src = job["src"]
            allow_loc = _job_allows_local(job)
            cov_npz = cov_dir / f"{idx_j:04d}.cov.npz"
            cleanup_paths: list[Path] = []
            try:
                src_clean = src.split("?", 1)[0].lower()
                aln_fmt = None
                if src_clean.endswith(".cram"):
                    aln_fmt = "cram"
                elif src_clean.endswith(".bam"):
                    aln_fmt = "bam"
                elif allow_loc and not src.startswith(("http://", "https://")) and Path(src).is_file():
                    aln_fmt = _detect_aln_format(Path(src))

                if aln_fmt in ("cram", "bam"):
                    if not src.startswith(("http://", "https://", "ftp://")) and not allow_loc:
                        raise ValueError(
                            f"Sample {sid!r}: local BAM/CRAM path in manifest requires --allow-local-manifest-paths"
                        )
                    crai_src = job["crai"] or (
                        src + ".crai"
                        if (aln_fmt == "cram" and (src.startswith(("http://", "https://")) or (allow_loc and Path(src + ".crai").is_file())))
                        else ""
                    )
                    use_cram_slicing = analysis_mode in ("auto", "focused_cram") and aln_fmt == "cram" and bool(crai_src)
                    if use_cram_slicing:
                        if src.startswith(("http://", "https://")):
                            cram_ref = src
                        else:
                            cram_ref = str(_stage_local_aln(Path(src), work_dir / "aln_links", sid, "cram"))
                        crai_dst = work_dir / "crai" / f"{idx_j:04d}_{sid}.cram.crai"
                        crai_local = _fetch_crai_file(crai_src, crai_dst, allow_local=allow_loc)
                        cleanup_paths.append(crai_local)
                        regs_use = bed_regions or [(cn, 0, ln) for cn, ln in ctg_lens.items()]
                        tel = VG.run_varloop_crai_focused(
                            g, eff_ref, cram_ref, crai_local, regs_use, s_vcf_gz,
                            params=dp_params, sample=sid, pad_bp=int(cram_pad_bp), zoom_assisted=bool(enable_reassembly),
                            clip_regions=bed_regions or None,
                            callable_npz_out=cov_npz,
                        )
                        tel["sample"] = sid
                        used_engines.add("focused_cram")
                    else:
                        if src.startswith(("http://", "https://")):
                            local_aln = _materialize_remote_input(src, work_dir / "aln" / f"{idx_j:04d}_{sid}.{aln_fmt}", "remote_aln", threads=threads)
                            cleanup_paths.append(local_aln)
                        else:
                            local_aln = _stage_local_aln(Path(src), work_dir / "aln_links", sid, aln_fmt)
                            if crai_src and allow_loc and Path(crai_src).is_file():
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
                            clip_regions=bed_regions or None,
                            callable_npz_out=cov_npz,
                        )
                        used_engines.add("diploid_wgs")
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

                    if job.get("direct_pair"):
                        r1_ref, r2_ref = job["direct_pair"]
                        rkind = "local_fastq"
                    else:
                        r1_ref, r2_ref, rkind = _resolve_fastq_pair_for_row(
                            job["row"] or {"source": src, "mate2_url": job["mate2"]},
                            local_map,
                            work_dir,
                            sid,
                            allow_local=allow_loc,
                        )
                    if rkind == "insdc_odp":
                        r1_dst = work_dir / "fq" / f"{idx_j:04d}_{sid}_R1.fq.gz"
                        r2_dst = work_dir / "fq" / f"{idx_j:04d}_{sid}_R2.fq.gz"
                        r1_local, r2_local = _materialize_odp_paired(r1_ref, r1_dst, r2_dst, threads=threads)
                        cleanup_paths.extend([r1_local, r2_local])
                    elif rkind != "local_fastq":
                        if not r2_ref:
                            raise ValueError(f"Sample {sid!r}: paired-end R1 and R2 FASTQ inputs are required for variant calling")
                        r1_local = _materialize_remote_input(r1_ref, work_dir / "fq" / f"{idx_j:04d}_{sid}_R1.fq.gz", rkind, threads=threads)
                        cleanup_paths.append(r1_local)
                        r2_local = _materialize_remote_input(r2_ref, work_dir / "fq" / f"{idx_j:04d}_{sid}_R2.fq.gz", rkind, threads=threads)
                        cleanup_paths.append(r2_local)
                    else:
                        if not r2_ref:
                            raise ValueError(f"Sample {sid!r}: paired-end R1 and R2 FASTQ inputs are required for variant calling")
                        r1_local = _stage_local_fastq(Path(r1_ref), work_dir / "fq_links", f"{idx_j:04d}_{sid}_R1")
                        r2_local = _stage_local_fastq(Path(r2_ref), work_dir / "fq_links", f"{idx_j:04d}_{sid}_R2")

                    if kset_tab is not None:
                        VG._validate_fastq_input(Path(r1_local))
                        VG._validate_fastq_input(Path(r2_local))
                        filt_r1 = work_dir / "fq_filt" / f"{idx_j:04d}_{sid}_R1.fq.gz"
                        filt_r2 = work_dir / "fq_filt" / f"{idx_j:04d}_{sid}_R2.fq.gz"
                        cleanup_paths.extend([filt_r1, filt_r2])
                        fstats = VG.extract_sieved_fastq_pair(
                            r1_local, r2_local, filt_r1, filt_r2,
                            kset_tab=kset_tab, stride=5, need=1, p_sample=p_frac,
                            decomp_threads=min(4, threads), comp_threads=min(4, threads),
                        )
                        _log(f"  Target pre-filter {sid}: kept {fstats['spots_emitted']}/{fstats['spots_seen']} read pairs")
                        if int(fstats["spots_seen"]) <= 0 or int(fstats["spots_emitted"]) <= 0:
                            raise RuntimeError(
                                f"Sample {sid!r}: zero read pairs retained after target 31-mer pre-filter "
                                f"({fstats['spots_emitted']}/{fstats['spots_seen']})"
                            )
                        r1_use, r2_use, p_run = filt_r1, filt_r2, 1.0
                        used_engines.add("wgs_stream")
                    else:
                        r1_use, r2_use, p_run = r1_local, r2_local, p_frac
                        used_engines.add("diploid_wgs")

                    tel = VG.run_varloop_fastq(
                        g, eff_ref, r1_use, r2_use, s_vcf_gz,
                        params=dp_params, map_threads=max(1, threads),
                        target_regions=bed_regions or None, zoom_regions=bed_regions or None,
                        sample=sid, p_sample=p_run, call_workers=min(4, max(1, threads // 2)),
                        mapper_name=chosen_mapper,
                        clip_regions=bed_regions or None,
                        callable_npz_out=cov_npz,
                    )
            finally:
                # Immediately unlink per-sample materialized FASTQ/BAM scratch files (#43)
                for cp_tmp in cleanup_paths:
                    cp_tmp.unlink(missing_ok=True)

            results_by_idx[idx_j] = (tel, s_vcf_gz, sid)
            cov_files_by_idx[idx_j] = (cov_npz, None)
            _log(f"  [{idx_j}/{len(jobs)}] {sid}: {tel.get('n_variants', 0)} variants ({tel.get('total_wall_s', tel.get('wall_s', 0)):.2f}s)")

        import pyarrow as pa
        import pyarrow.parquet as pq

        var_cols = ["sample_id", "chrom", "pos", "ref", "alt", "kind", "gt", "gq", "dp", "ad_ref", "ad_alt", "wsaf", "qual", "source"]
        var_schema = pa.schema(
            [
                ("sample_id", pa.string()),
                ("chrom", pa.string()),
                ("pos", pa.int64()),
                ("ref", pa.string()),
                ("alt", pa.string()),
                ("kind", pa.string()),
                ("gt", pa.string()),
                ("gq", pa.float64()),
                ("dp", pa.float64()),
                ("ad_ref", pa.float64()),
                ("ad_alt", pa.string()),
                ("wsaf", pa.float64()),
                ("qual", pa.float64()),
                ("source", pa.string()),
            ]
        )

        out_variants_p.parent.mkdir(parents=True, exist_ok=True)
        pq_path = state_root / "variants.parquet"
        preview_chunks: list[pd.DataFrame] = []
        preview_n = 0
        total_variants = 0
        total_snv = 0
        total_indel = 0
        wrote_tsv_header = False
        cohort_sites_set: set[tuple[str, int]] = set()
        called_sample_sites: set[tuple[str, int, str]] = set()
        high_dup_samples: list[str] = []

        with pq.ParquetWriter(str(pq_path), var_schema, compression="zstd") as pq_writer:
            for idx_j in range(1, len(jobs) + 1):
                raw_tel, s_vcf_gz, sid = results_by_idx[idx_j]
                if "reads_counted" in raw_tel:
                    reads_cnt_j = int(raw_tel.get("reads_counted") or 0)
                else:
                    reads_cnt_j = int(
                        raw_tel.get("counted_unitigs")
                        or raw_tel.get("sam_records")
                        or raw_tel.get("n_variants")
                        or 0
                    )
                if reads_cnt_j <= 0:
                    raise RuntimeError(f"Sample {sid!r}: zero reads/unitigs counted — refusing to pack invalid sample state")
                raw_tel["reads_counted"] = reads_cnt_j
                prim_j = int(raw_tel.get("primary_mapped") or raw_tel.get("sam_records") or 0)
                dup_frac_j = float(raw_tel.get("pcr_dup_fraction") or 0.0)
                if eff_dedup_mode != "off" and prim_j >= 100 and dup_frac_j > 0.95:
                    high_dup_samples.append(sid)
                    _log(
                        f"WARNING: Sample {sid!r} had {100.0 * dup_frac_j:.1f}% PCR duplicates skipped "
                        f"({int(raw_tel.get('pcr_dups_skipped') or 0)}/{prim_j}). "
                        "If this is a targeted PCR amplicon library, set PCR duplicate removal to Off (--dedup-bits 0)."
                    )
                clean_tel = {
                    k: v
                    for k, v in raw_tel.items()
                    if k not in ("vcf_path", "pileup_vcf_path")
                }
                sample_tels.append(clean_tel)
                per_sample_vcfs.append(s_vcf_gz)
                df_s = _vcf_to_dataframe(s_vcf_gz, sample_override=sid)
                n_s = len(df_s)
                if n_s > 0:
                    for ch_v, pos_v in zip(df_s["chrom"], df_s["pos"]):
                        ch_s = str(ch_v)
                        pos_i = int(pos_v)
                        cohort_sites_set.add((ch_s, pos_i))
                        called_sample_sites.add((ch_s, pos_i, sid))
                    n_s_snv = int((df_s["kind"] == "SNV").sum())
                    total_variants += n_s
                    total_snv += n_s_snv
                    total_indel += n_s - n_s_snv
                    if preview_n < 50:
                        take = df_s.head(50 - preview_n).copy()
                        preview_chunks.append(take)
                        preview_n += len(take)
                    df_s.to_csv(
                        out_variants_p,
                        sep="\t",
                        index=False,
                        float_format="%.6g",
                        mode="a" if wrote_tsv_header else "w",
                        header=not wrote_tsv_header,
                    )
                    wrote_tsv_header = True
                    df_typed = pd.DataFrame(
                        {
                            "sample_id": df_s["sample_id"].astype(str),
                            "chrom": df_s["chrom"].astype(str),
                            "pos": pd.to_numeric(df_s["pos"], errors="coerce").fillna(0).astype("int64"),
                            "ref": df_s["ref"].astype(str),
                            "alt": df_s["alt"].astype(str),
                            "kind": df_s["kind"].astype(str),
                            "gt": df_s["gt"].astype(str),
                            "gq": pd.to_numeric(df_s["gq"], errors="coerce").astype("float64"),
                            "dp": pd.to_numeric(df_s["dp"], errors="coerce").astype("float64"),
                            "ad_ref": pd.to_numeric(df_s["ad_ref"], errors="coerce").astype("float64"),
                            "ad_alt": df_s["ad_alt"].astype(str),
                            "wsaf": pd.to_numeric(df_s["wsaf"], errors="coerce").astype("float64"),
                            "qual": pd.to_numeric(df_s["qual"], errors="coerce").astype("float64"),
                            "source": df_s["source"].astype(str),
                        }
                    )
                    pq_writer.write_table(pa.Table.from_pandas(df_typed, schema=var_schema, preserve_index=False))
            if not wrote_tsv_header:
                pd.DataFrame(columns=var_cols).to_csv(out_variants_p, sep="\t", index=False)
                pq_writer.write_table(pa.Table.from_pydict({f.name: [] for f in var_schema}, schema=var_schema))

        # Write site_nocalls.parquet recording uncovered/no-call samples at called cohort variant sites (#62)
        nocall_rows: list[tuple[str, int, str]] = []
        if cohort_sites_set:
            ordered_sites = sorted(cohort_sites_set)
            for idx_j in range(1, len(jobs) + 1):
                _, _, sid = results_by_idx[idx_j]
                c_npz, c_tsv = cov_files_by_idx.get(idx_j, (None, None))
                cov_map = _load_sample_covered_intervals(c_npz, c_tsv)
                for ch_s, pos_i in ordered_sites:
                    if (ch_s, pos_i, sid) not in called_sample_sites:
                        if not _is_pos0_covered(cov_map, ch_s, pos_i - 1):
                            nocall_rows.append((ch_s, pos_i, sid))
        _write_site_nocalls_parquet(state_root / "site_nocalls.parquet", nocall_rows)

        variants_preview_df = pd.concat(preview_chunks, ignore_index=True) if preview_chunks else pd.DataFrame(columns=var_cols)

        eff_mode = (
            analysis_mode
            if analysis_mode != "auto"
            else (next(iter(used_engines)) if len(used_engines) == 1 else ("hybrid_cohort" if used_engines else "diploid_wgs"))
        )

        if out_vcf_p:
            if len(per_sample_vcfs) == 1:
                _materialize_vcf_output(per_sample_vcfs[0], out_vcf_p)
            elif total_variants > 0 and shutil.which("bcftools") and all(Path(str(p) + ".tbi").is_file() for p in per_sample_vcfs):
                merged_gz = work_dir / "merged.vcf.gz"
                _bcftools_merge_batched(per_sample_vcfs, merged_gz, work_dir, threads=threads)
                _materialize_vcf_output(merged_gz, out_vcf_p)
            else:
                # Bypass bcftools merge on 0-variant cohorts to avoid bcftools 1.15.1 SIGSEGV (#37)
                _merge_vcfs_pure_python(per_sample_vcfs, out_vcf_p)

        if out_drm_p:
            out_drm_p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "sample_id": t.get("sample"),
                        "analysis_mode": eff_mode,
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
            "mode": eff_mode,
            "ploidy": eff_ploidy,
            "min_wsaf": eff_min_wsaf,
            "dedup": eff_dedup_mode,
            "reference_label": ref_lbl,
            "n_samples": len(jobs),
            "n_variants": total_variants,
            "n_snv": total_snv,
            "n_indel": total_indel,
            "wall_s": round(time.time() - t_start, 2),
        }
        if high_dup_samples:
            summary_stats["high_dup_samples"] = high_dup_samples
        render_variationloop_report(
            out_report_p,
            analysis_mode=eff_mode,
            summary_stats=summary_stats,
            variants_df=variants_preview_df,
            sample_telemetry=sample_tels,
        )

        store = FocaloopStateStore(state_root)
        HypothesisLedger.from_store(store, alpha_budget=0.05)
        pd.DataFrame(sample_tels).to_parquet(state_root / "sample_telemetry.parquet", index=False)
        sample_records = []
        for idx_j, j in enumerate(jobs, 1):
            tel_j, _, _ = results_by_idx[idx_j]
            sample_records.append(
                _sanitize_state_sample_record(
                    j["sample"],
                    j["group"],
                    j.get("row"),
                    reads_counted=int(tel_j.get("reads_counted") or 0),
                    median_depth=float(tel_j.get("median_depth") or 0.0),
                )
            )
        state_meta = {
            "engine": "variationloop",
            "version": "0.4.0",
            "blueprint_id": "variationloop",
            "mode": eff_mode,
            "ploidy": eff_ploidy,
            "min_wsaf": eff_min_wsaf,
            "summary": summary_stats,
            "samples": sample_records,
        }
        write_json(state_root / STATE_META, state_meta)
        if state_archive:
            pack_state(state_root, state_archive)
        return state_meta
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
