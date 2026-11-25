"""Production Galaxy runner for ExpressionLoop (PanExpr v2.4 adaptive RNA-seq engine)."""

from __future__ import annotations

import csv
from dataclasses import asdict
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .archive import pack_state
from .engine import STATE_META, write_json
from .hooks import HypothesisLedger
from .reports import render_panexpr_report
from .state import FocaloopStateStore, HYPOTHESIS_LEDGER_SCHEMA

_INSDC_RUN_RE = re.compile(r"^[SED]RR\d+$", re.IGNORECASE)
_READ_EXT_RE = re.compile(r"(\.f(ast)?qsanger(\.gz|\.bz2)?|\.f(ast)?q(\.gz|\.bz2)?|\.gz|\.bz2)+$", re.IGNORECASE)
_EXPLICIT_R_MATE_RE = re.compile(r"^(.+?)(?:[._])R([12])$", re.IGNORECASE)
_AGGRESSIVE_ERR_RE = re.compile(r"(error|exception)\s*:", re.IGNORECASE)


def _log(msg: str) -> None:
    # Avoid tripping Galaxy detect_errors="aggressive" on non-fatal tournament probe messages.
    safe_msg = _AGGRESSIVE_ERR_RE.sub(r"\1 -", str(msg))
    print(f"[focaloop-expressionloop] {safe_msg}", file=sys.stderr, flush=True)


def _sanitize_leading_dash(val: str) -> str:
    """Normalize leading '-' on sample/read IDs so CLI parsers and aligners never see option flags (L94)."""
    s = str(val or "").strip()
    if s.startswith("-"):
        return "_" + s[1:]
    return s


def _sniff_annotation_ext(path: Path) -> str:
    """Detect GFF3 vs GTF by content (works even when Galaxy stages files as dataset_*.dat)."""
    try:
        with open(path, "rb") as fh:
            magic = fh.read(2)
        opener = gzip.open if magic == b"\x1f\x8b" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as tf:
            for _ in range(200):
                ln = tf.readline()
                if not ln:
                    break
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("##gff-version 3"):
                    return ".gff3"
                if s.startswith("#"):
                    continue
                parts = s.split("\t")
                if len(parts) >= 9:
                    attr = parts[8]
                    if ("ID=" in attr or "Parent=" in attr) and 'gene_id "' not in attr:
                        return ".gff3"
                    return ".gtf"
    except OSError:
        pass
    return ".gff3" if str(path).endswith((".gff", ".gff3", ".gff.gz", ".gff3.gz")) else ".gtf"


def _gunzip_if_needed(src: str | Path, dst_dir: Path, out_name: str) -> Path:
    src_p = Path(src)
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
    os.symlink(src_p.resolve(), dst)
    return dst


def _detect_organism_mode(gtf_path: Path, requested: str) -> str:
    if requested in ("eukaryotic", "prokaryotic"):
        return requested
    exon_by_tx: dict[str, int] = {}
    n_exon = 0
    n_cds = 0
    from .panexpr.catalog import iter_features

    for i, feat in enumerate(iter_features(gtf_path)):
        if i > 40_000:
            break
        if feat.ftype == "exon":
            n_exon += 1
            if feat.attrs.get("transcript_id"):
                txs = [feat.attrs["transcript_id"]]
            else:
                txs = [p.strip() for p in (feat.attrs.get("Parent") or "").split(",") if p.strip()]
            for tx in txs:
                exon_by_tx[tx] = exon_by_tx.get(tx, 0) + 1
        elif feat.ftype == "CDS":
            n_cds += 1
    multi_exon_tx = sum(1 for v in exon_by_tx.values() if v >= 2)
    if multi_exon_tx >= 3:
        return "eukaryotic"
    if n_cds > 0 and n_exon < max(5, int(0.25 * n_cds)):
        return "prokaryotic"
    if n_exon > 0:
        return "eukaryotic"
    return "prokaryotic" if n_cds > 0 else "eukaryotic"


_SAMPLE_COL_CANDS = ("sample_id", "sample", "sample_alias", "Sample")
_COND_COL_CANDS = ("condition", "group", "Condition")


def _read_manifest_table(manifest_path: str | Path) -> list[dict[str, str]]:
    text = Path(manifest_path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{manifest_path}: manifest is empty")
    delim = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
    fieldnames = [(fn or "").strip() for fn in (reader.fieldnames or []) if (fn or "").strip()]
    lower_fields = {fn.lower() for fn in fieldnames}
    if not any(c.lower() in lower_fields for c in _SAMPLE_COL_CANDS):
        raise ValueError(
            f"{manifest_path}: manifest header is missing a sample identifier column "
            f"(expected one of {list(_SAMPLE_COL_CANDS)}, got {fieldnames})"
        )
    if not any(c.lower() in lower_fields for c in _COND_COL_CANDS):
        raise ValueError(
            f"{manifest_path}: manifest header is missing a condition/group column "
            f"(expected one of {list(_COND_COL_CANDS)}, got {fieldnames})"
        )
    rows: list[dict[str, str]] = []
    for row_num, raw_row in enumerate(reader, start=1):
        clean_row = {(k or "").strip(): (v or "").strip() for k, v in raw_row.items() if (k or "").strip()}
        if not any(clean_row.values()):
            continue
        sid = _pick_col(clean_row, _SAMPLE_COL_CANDS, "")
        cnd = _pick_col(clean_row, _COND_COL_CANDS, "")
        if not sid:
            raise ValueError(f"{manifest_path}: row {row_num} has an empty sample_id cell")
        if not cnd:
            raise ValueError(f"{manifest_path}: row {row_num} ({sid!r}) has an empty condition/group cell")
        rows.append(clean_row)
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


def parse_expression_contrast(contrast: str, default_factor: str = "condition") -> list[str]:
    text = contrast.strip()
    if "=" in text and ":" in text:
        fac, rest = text.split("=", 1)
        a, b = rest.split(":", 1)
        return [fac.strip(), a.strip(), b.strip()]
    parts = text.split(":")
    if len(parts) == 3:
        return [p.strip() for p in parts]
    if len(parts) == 2:
        return [default_factor, parts[0].strip(), parts[1].strip()]
    if "_vs_" in text:
        a, b = text.split("_vs_", 1)
        return [default_factor, a.strip(), b.strip()]
    raise ValueError(f"Invalid contrast {contrast!r} - expected 'condition:treated:control' or 'treated:control'")


def _parse_followup_contrasts(raw: Sequence[str] | str | None) -> list[list[str]]:
    if not raw:
        return []
    tokens: list[str] = []
    if isinstance(raw, str):
        for part in re.split(r"[,;\s]+", raw.strip()):
            if part:
                tokens.append(part)
    else:
        for item in raw:
            for part in re.split(r"[,;\s]+", str(item).strip()):
                if part:
                    tokens.append(part)
    return [parse_expression_contrast(tok) for tok in tokens]


def _ipv4_session():
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    s = requests.Session()
    s.headers["User-Agent"] = "focaloop/0.4.1"
    return s


def _head_content_length(url: str, timeout: float = 45.0, retries: int = 3) -> int:
    """Resolve remote archive byte length strictly from HTTP 200 Content-Length or HTTP 206 Content-Range (L11)."""
    s = _ipv4_session()
    retry_base = float(os.environ.get("FOCALOOP_HEAD_RETRY_BASE_S", "1.0"))
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            r_head = s.head(url, allow_redirects=True, timeout=timeout)
            if r_head.status_code in (401, 403, 404, 410):
                r_head.raise_for_status()
            if r_head.status_code == 200:
                cl = (r_head.headers.get("Content-Length") or "").strip()
                if cl.isdigit() and int(cl) > 0:
                    return int(cl)
            elif r_head.status_code in (429, 500, 502, 503, 504):
                r_head.raise_for_status()

            r_get = s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
            try:
                if r_get.status_code >= 400:
                    r_get.raise_for_status()
                if r_get.status_code == 206:
                    cr = (r_get.headers.get("Content-Range") or "").strip()
                    if "/" in cr:
                        tot = cr.rsplit("/", 1)[-1].strip()
                        if tot.isdigit() and int(tot) > 0:
                            return int(tot)
                    raise RuntimeError(
                        f"HTTP 206 response for {url} lacked a valid total size in Content-Range ({cr!r})"
                    )
                if r_get.status_code == 200:
                    cl_get = (r_get.headers.get("Content-Length") or "").strip()
                    if cl_get.isdigit() and int(cl_get) > 0:
                        return int(cl_get)
                raise RuntimeError(
                    f"Unexpected HTTP {r_get.status_code} without usable byte length for {url}"
                )
            finally:
                r_get.close()
        except Exception as exc:
            last_err = exc
            resp = getattr(exc, "response", None)
            code = getattr(resp, "status_code", None)
            if code in (401, 403, 404, 410):
                raise
            if attempt + 1 < max(1, retries):
                if retry_base > 0:
                    time.sleep(retry_base * (attempt + 1))
                continue
    if last_err is not None:
        raise RuntimeError(f"Could not determine remote byte length for {url}: {last_err}") from last_err
    raise RuntimeError(f"Could not determine remote byte length for {url}")


def _strip_ext_only(raw_id: str) -> str:
    return _READ_EXT_RE.sub("", _sanitize_leading_dash(raw_id))


def _ena_row_to_meta(erow: dict[str, str], files_len: int, files_bytes: int, read_count: int) -> dict[str, Any]:
    layout_raw = str(erow.get("library_layout") or "").strip().upper()
    is_pe = ("PAIRED" in layout_raw) or (layout_raw == "PE") or (files_len == 2)
    mates = 2 if is_pe else 1
    bc_raw = str(erow.get("base_count") or "0").strip()
    base_count = int(bc_raw) if bc_raw.isdigit() else 0
    if read_count > 0 and base_count > 0:
        rlen = max(31.0, float(base_count) / float(read_count * mates))
        kpf = max(2.0, float(mates) * max(1.0, rlen - 30.0))
    else:
        rlen = 100.0
        kpf = float(mates * 70.0)
    return {
        "read_count": int(read_count),
        "base_count": int(base_count),
        "fastq_bytes": int(files_bytes),
        "layout": "PE" if is_pe else "SE",
        "mates": mates,
        "read_len": round(rlen, 2),
        "kmers_per_frag": round(kpf, 2),
    }


def resolve_expression_manifest(
    manifest_path: str | Path,
    reads: Sequence[str] | None = None,
    reads_r1: Sequence[str] | None = None,
    reads_r2: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
) -> tuple[Any, pd.DataFrame, bool]:
    """Resolve sample sheet + optional Galaxy History FASTQs (SE or paired collection) into StudyManifest."""
    from .panexpr.manifest import FastqFile, Run, StudyManifest, fetch_ena_filereport

    rows = _read_manifest_table(manifest_path)

    paired_map: dict[str, tuple[Path, Path]] = {}
    if reads_r1 or reads_r2:
        if not (reads_r1 and reads_r2) or len(reads_r1) != len(reads_r2):
            raise ValueError("Paired History inputs require equal-length --reads-r1 and --reads-r2 lists")
        if read_ids is not None and len(read_ids) != len(reads_r1):
            raise ValueError(
                f"Length mismatch between --read-ids ({len(read_ids)}) and --reads-r1 ({len(reads_r1)})"
            )
        ids = (
            [_sanitize_leading_dash(x) for x in read_ids]
            if read_ids is not None
            else [_strip_ext_only(Path(p).name) for p in reads_r1]
        )
        for rid, p1_raw, p2_raw in zip(ids, reads_r1, reads_r2):
            p1 = Path(p1_raw).resolve()
            p2 = Path(p2_raw).resolve()
            paired_map[rid] = (p1, p2)
            no_ext = _strip_ext_only(rid)
            if no_ext and no_ext not in paired_map:
                paired_map[no_ext] = (p1, p2)

    # Single-end / flat History dataset map: exact or extension-stripped keys ONLY (never strip _1/_2! #3)
    local_map: dict[str, list[Path]] = {}
    explicit_r_mates: dict[str, dict[str, Path]] = {}
    if reads:
        if read_ids is not None and len(read_ids) != len(reads):
            raise ValueError(
                f"Length mismatch between --read-ids ({len(read_ids)}) and --reads ({len(reads)})"
            )
        ids = (
            [_sanitize_leading_dash(x) for x in read_ids]
            if read_ids is not None
            else [_sanitize_leading_dash(Path(r).name) for r in reads]
        )
        for rid, rpath in zip(ids, reads):
            resolved = Path(rpath).resolve()
            local_map.setdefault(rid, []).append(resolved)
            no_ext = _strip_ext_only(rid)
            if no_ext and no_ext != rid:
                local_map.setdefault(no_ext, []).append(resolved)
            m_rmate = _EXPLICIT_R_MATE_RE.match(no_ext)
            if m_rmate:
                stem, mate_num = m_rmate.group(1), m_rmate.group(2)
                explicit_r_mates.setdefault(stem, {})[mate_num] = resolved
        # Only register <stem> when BOTH explicit _R1 and _R2 files were supplied
        for stem, mdict in explicit_r_mates.items():
            if "1" in mdict and "2" in mdict and stem not in local_map:
                local_map[stem] = [mdict["1"], mdict["2"]]

    def _lookup_paired(key: str) -> tuple[Path, Path] | None:
        if not key:
            return None
        k_norm = _sanitize_leading_dash(key)
        k_noext = _strip_ext_only(k_norm)
        return paired_map.get(k_norm) or paired_map.get(k_noext)

    def _lookup_local(key: str) -> list[Path] | None:
        if not key:
            return None
        k_norm = _sanitize_leading_dash(key)
        k_noext = _strip_ext_only(k_norm)
        return local_map.get(k_norm) or local_map.get(k_noext)

    # Batch-resolve any INSDC run accessions via ENA filereport when not backed by History FASTQs
    ena_cache: dict[str, dict[str, str]] = {}
    accs_needed = []
    for row in rows:
        sample_id = _sanitize_leading_dash(_pick_col(row, _SAMPLE_COL_CANDS))
        src = _pick_col(row, ("run_or_url", "run_accession", "accession", "source", "fastq_1", "url"))
        if (
            _INSDC_RUN_RE.match(src)
            and _lookup_paired(src) is None
            and _lookup_local(src) is None
            and _lookup_paired(sample_id) is None
            and _lookup_local(sample_id) is None
            and not row.get("fastq_ftp")
        ):
            accs_needed.append(src.upper())
    unique_accs = list(dict.fromkeys(accs_needed))
    for i_chunk in range(0, len(unique_accs), 40):
        chunk = unique_accs[i_chunk : i_chunk + 40]
        batch_ok = False
        for attempt in range(3):
            try:
                tsv = fetch_ena_filereport(",".join(chunk), timeout=60.0)
                parsed = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
                if parsed:
                    for prow in parsed:
                        ra = str(prow.get("run_accession") or "").strip().upper()
                        if ra:
                            ena_cache[ra] = prow
                    batch_ok = True
                    break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if not batch_ok:
            for acc in chunk:
                if acc in ena_cache:
                    continue
                for attempt in range(3):
                    try:
                        tsv = fetch_ena_filereport(acc, timeout=60.0)
                        parsed = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
                        if parsed:
                            ena_cache[acc] = parsed[0]
                            break
                    except Exception:
                        time.sleep(1.5 * (attempt + 1))
    for acc in unique_accs:
        if acc not in ena_cache:
            odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}"
            odp_sz = _head_content_length(odp_url)
            layout_guess = "PE"
            for r_chk in rows:
                if _pick_col(r_chk, ("run_or_url", "run_accession", "accession", "source")).upper() == acc:
                    layout_guess = _pick_col(r_chk, ("layout", "library_layout"), "PE").upper()
                    break
            ena_cache[acc] = {
                "run_accession": acc,
                "library_layout": layout_guess,
                "library_strategy": "RNA-Seq",
                "read_count": str(max(1, odp_sz // 100)),
                "base_count": "0",
                "fastq_ftp": f"{odp_url};{odp_url}" if layout_guess == "PE" else odp_url,
                "fastq_bytes": f"{odp_sz // 2};{odp_sz - odp_sz // 2}" if layout_guess == "PE" else str(odp_sz),
                "fastq_md5": "",
            }

    # Pre-count rows per biological sample_id so multi-lane sheets auto-disambiguate run_id (#1)
    sample_row_totals: dict[str, int] = {}
    seen_explicit_rids: set[str] = set()
    for row in rows:
        sid_pre = _sanitize_leading_dash(_pick_col(row, _SAMPLE_COL_CANDS))
        sample_row_totals[sid_pre] = sample_row_totals.get(sid_pre, 0) + 1
        exp_rid = _sanitize_leading_dash(_pick_col(row, ("run_id",), ""))
        if exp_rid:
            if exp_rid in seen_explicit_rids:
                raise ValueError(f"{manifest_path}: duplicate explicit run_id {exp_rid!r} in manifest")
            seen_explicit_rids.add(exp_rid)

    sample_lane_seq: dict[str, int] = {}
    used_run_ids: set[str] = set()
    claimed_local_files: dict[Path, str] = {}

    runs: list[Run] = []
    meta_records: list[dict[str, Any]] = []
    all_local = True

    def _claim_files(paths: Sequence[Path], row_label: str) -> None:
        for lp in paths:
            prev_owner = claimed_local_files.get(lp)
            if prev_owner is not None:
                raise ValueError(
                    f"History FASTQ file {lp.name!r} was matched by multiple manifest rows "
                    f"({prev_owner!r} and {row_label!r}). Check sample_id / run_or_url names."
                )
            claimed_local_files[lp] = row_label

    for idx_r, row in enumerate(rows):
        sample_id = _sanitize_leading_dash(_pick_col(row, _SAMPLE_COL_CANDS, f"sample_{idx_r + 1}"))
        cond = _pick_col(row, _COND_COL_CANDS)
        src = _pick_col(row, ("run_or_url", "run_accession", "accession", "source", "fastq_1", "url"))
        mate2 = _pick_col(row, ("mate2_url", "fastq_2", "mate2"))
        strategy = _pick_col(row, ("strategy", "library_strategy"), "RNA-Seq")
        read_count_raw = _pick_col(row, ("read_count", "spots", "n_reads"), "0")
        read_count = int(read_count_raw) if read_count_raw.isdigit() else 0
        explicit_run_id = _sanitize_leading_dash(_pick_col(row, ("run_id",), ""))

        sample_lane_seq[sample_id] = sample_lane_seq.get(sample_id, 0) + 1
        lane_num = sample_lane_seq[sample_id]
        is_multi_lane = sample_row_totals.get(sample_id, 1) > 1

        if not src and (_lookup_paired(sample_id) is not None or _lookup_local(sample_id) is not None):
            src = sample_id

        files: list[FastqFile] = []
        source_kind = "unknown"
        ena_meta_entry: dict[str, Any] | None = None

        cand_pair = _lookup_paired(src) if src else None
        if cand_pair is None and not src:
            cand_pair = _lookup_paired(sample_id)
        elif cand_pair is None and src == sample_id:
            cand_pair = _lookup_paired(sample_id)
        elif cand_pair is None and reads_r1 and not _INSDC_RUN_RE.match(src) and not src.startswith(("http://", "https://", "ftp://")) and not (src and Path(src).exists()):
            # Only fall back from missing/empty src to sample_id when src wasn't a distinct unmatched token
            pass

        cand_local = _lookup_local(src) if src else None
        if cand_local is None and src == sample_id:
            cand_local = _lookup_local(sample_id)

        # Allow sample_id lookup in History mode when manifest has no run_or_url column (src == "")
        if not src:
            cand_pair = cand_pair or _lookup_paired(sample_id)
            cand_local = cand_local or _lookup_local(sample_id)

        default_local_rid = (
            explicit_run_id
            or (_sanitize_leading_dash(src) if (src and _INSDC_RUN_RE.match(src) and not is_multi_lane) else "")
            or (f"{sample_id}_L{lane_num}" if is_multi_lane else sample_id)
        )
        row_tag = f"{sample_id} (lane {lane_num})" if is_multi_lane else sample_id

        if cand_pair is not None:
            _claim_files(cand_pair, row_tag)
            for lp in cand_pair:
                files.append(FastqFile(url=str(lp), bytes=lp.stat().st_size, local_path=str(lp)))
            run_id = explicit_run_id or default_local_rid
            source_kind = "history_paired"
        elif cand_local is not None:
            ordered = list(dict.fromkeys(cand_local))[:2]
            _claim_files(ordered, row_tag)
            for lp in ordered:
                files.append(FastqFile(url=str(lp), bytes=lp.stat().st_size, local_path=str(lp)))
            run_id = explicit_run_id or default_local_rid
            source_kind = "history_single"
        elif src and Path(src).exists():
            p1 = Path(src).resolve()
            local_ps = [p1]
            if mate2:
                p2_cand = Path(mate2)
                if not p2_cand.exists():
                    raise ValueError(f"Row {row_tag!r}: mate2_url file {mate2!r} does not exist")
                local_ps.append(p2_cand.resolve())
            for lp in local_ps:
                files.append(FastqFile(url=str(lp), bytes=lp.stat().st_size, local_path=str(lp)))
            run_id = explicit_run_id or default_local_rid
            source_kind = "local_path"
        elif (_INSDC_RUN_RE.match(src) or row.get("fastq_ftp")) and not src.startswith(("http://", "https://", "ftp://")):
            all_local = False
            erow = row if row.get("fastq_ftp") else ena_cache[src.upper()]
            acc_id = (erow.get("run_accession") or src).strip().upper()
            run_id = explicit_run_id or acc_id
            rc_ena = erow.get("read_count") or "0"
            if str(rc_ena).isdigit() and int(rc_ena) > 0:
                read_count = int(rc_ena)
            raw_urls = [u for u in (erow.get("fastq_ftp") or "").split(";") if u]
            raw_sizes = [int(x) for x in (erow.get("fastq_bytes") or "").split(";") if x and str(x).isdigit()]
            raw_md5 = [m for m in (erow.get("fastq_md5") or "").split(";") if m]
            while len(raw_md5) < len(raw_urls):
                raw_md5.append("")
            if raw_urls:
                paired_idx = [i for i, u in enumerate(raw_urls) if u.endswith("_1.fastq.gz") or u.endswith("_2.fastq.gz")]
                use_idx = paired_idx if len(paired_idx) == 2 else ([0, 1] if len(raw_urls) == 2 else [0])
                for i in use_idx:
                    u = raw_urls[i]
                    if u.startswith("ftp://"):
                        u = "https://" + u[len("ftp://"):]
                    elif not u.startswith("http"):
                        u = "https://" + u
                    sz = raw_sizes[i] if i < len(raw_sizes) else _head_content_length(u)
                    files.append(FastqFile(url=u, bytes=sz, md5=raw_md5[i]))
            else:
                odp_url = f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc_id}/{acc_id}"
                odp_sz = _head_content_length(odp_url)
                files.append(FastqFile(url=odp_url, bytes=odp_sz))
            strategy = erow.get("library_strategy") or strategy
            source_kind = "insdc" if _INSDC_RUN_RE.match(acc_id) else "url"
            ena_meta_entry = _ena_row_to_meta(
                erow,
                files_len=len(files),
                files_bytes=sum(f.bytes for f in files),
                read_count=read_count,
            )
        elif src.startswith(("http://", "https://", "ftp://")):
            all_local = False
            url_list = [u.strip() for u in src.split(";") if u.strip()]
            if mate2:
                url_list.append(mate2.strip())
            for u in url_list[:2]:
                if u.startswith("ftp://"):
                    u = "https://" + u[len("ftp://"):]
                sz = _head_content_length(u)
                files.append(FastqFile(url=u, bytes=sz))
            run_id = explicit_run_id or (f"{sample_id}_L{lane_num}" if is_multi_lane else sample_id)
            source_kind = "url"
        else:
            raise ValueError(
                f"Row {sample_id!r}: could not resolve run_or_url={src!r} as INSDC accession, URL, or History FASTQ"
            )

        if run_id in used_run_ids:
            raise ValueError(
                f"Duplicate run_id {run_id!r} resolved from manifest (each sequencing run row must have a distinct run_id or accession)"
            )
        used_run_ids.add(run_id)

        if read_count <= 0 and files:
            read_count = max(1, sum(f.bytes for f in files) // 100)

        layout = "PE" if len(files) == 2 else "SE"
        # Retain 'group' even when 'condition' is absent so '--design ~group' and '--contrast group:A:B' work (L10)
        skip_cols = {"sample_id", "sample", "condition", "run_or_url", "fastq_ftp", "fastq_bytes", "fastq_md5", "mate2_url", "run_id"}
        cov_extra: dict[str, Any] = {k: v for k, v in row.items() if k not in skip_cols and v != ""}
        run_extra: dict[str, Any] = dict(cov_extra)
        run_extra["_source_kind"] = source_kind
        if ena_meta_entry is not None:
            run_extra["_ena_meta"] = ena_meta_entry

        runs.append(
            Run(
                run_id=run_id,
                sample=sample_id,
                condition=cond,
                layout=layout,
                strategy=strategy,
                read_count=read_count,
                files=tuple(files),
                extra=run_extra,
            )
        )
        cov_row: dict[str, Any] = {"run_id": run_id, "sample": sample_id, "condition": cond}
        cov_row.update(cov_extra)
        meta_records.append(cov_row)

    manifest = StudyManifest(study_id=Path(manifest_path).stem, accession="galaxy_cohort", runs=runs)
    meta_df = pd.DataFrame(meta_records).fillna("")
    return manifest, meta_df, all_local


def _prepare_and_validate_design_metadata(
    sample_meta_df: pd.DataFrame,
    manifest: Any,
    design: str,
    contrast_list: list[str],
    followups: list[list[str]],
) -> pd.DataFrame:
    """Validate additive design, blank covariate cells, numeric casting, and full rank up front (L1, L3, L12, L13)."""
    from .panexpr.solver import parse_design_terms

    parsed_terms = parse_design_terms(design)
    design_cols = [c for c, _ in parsed_terms]
    explicit_cats = {c for c, is_cat in parsed_terms if is_cat}

    fac, lvl_a, lvl_b = contrast_list
    if fac not in sample_meta_df.columns:
        raise ValueError(
            f"Primary contrast factor {fac!r} is missing from manifest columns: {list(sample_meta_df.columns)}"
        )
    if fac not in design_cols:
        raise ValueError(
            f"Primary contrast factor {fac!r} must appear in design formula {design!r} (design columns: {design_cols})"
        )

    followup_factors = {fc[0] for fc in followups}
    for fc in followups:
        f_fac, f_a, f_b = fc
        if f_fac not in sample_meta_df.columns:
            raise ValueError(
                f"Follow-up contrast factor {f_fac!r} is missing from manifest columns: {list(sample_meta_df.columns)}"
            )
        if f_fac not in design_cols:
            raise ValueError(
                f"Follow-up contrast factor {f_fac!r} must appear in design formula {design!r} (design columns: {design_cols})"
            )
        f_lvls = set(sample_meta_df[f_fac].astype(str))
        for lvl in (f_a, f_b):
            if lvl not in f_lvls:
                raise ValueError(f"Follow-up contrast level {lvl!r} not present in column {f_fac!r}: {sorted(f_lvls)}")

    for col in design_cols:
        if col not in sample_meta_df.columns:
            raise ValueError(
                f"Design covariate {col!r} in {design!r} is missing from manifest columns: {list(sample_meta_df.columns)}"
            )
        vals = sample_meta_df[col]
        blank_mask = vals.isna() | (vals.astype(str).str.strip() == "")
        if bool(blank_mask.any()):
            bad_samples = list(sample_meta_df.loc[blank_mask, "sample"].astype(str).unique()[:5])
            raise ValueError(
                f"Design column {col!r} contains empty/missing values for sample(s): {', '.join(bad_samples)}"
            )

    present_lvls = set(sample_meta_df[fac].astype(str))
    for lvl in (lvl_a, lvl_b):
        if lvl not in present_lvls:
            raise ValueError(f"Contrast level {lvl!r} not present in manifest column {fac!r}: {sorted(present_lvls)}")
    if lvl_a == lvl_b:
        raise ValueError(f"Contrast test and reference levels must differ (both were {lvl_a!r})")

    rna_sample_ids = {r.sample for r in manifest.rna_runs}
    out_df = sample_meta_df.copy()

    # Cast non-contrast covariates whose values all parse as finite numbers to float64 unless wrapped in C(...) (L13)
    factor_cols = {fac} | followup_factors | explicit_cats
    for col in design_cols:
        if col in factor_cols:
            out_df[col] = out_df[col].astype(str)
            continue
        raw_strs = [str(v).strip() for v in out_df[col].tolist()]
        all_num = True
        parsed_floats: list[float] = []
        for sv in raw_strs:
            try:
                fv = float(sv)
                if not math.isfinite(fv):
                    all_num = False
                    break
                parsed_floats.append(fv)
            except ValueError:
                all_num = False
                break
        if all_num and parsed_floats:
            out_df[col] = np.asarray(parsed_floats, dtype=np.float64)
            # Propagate numeric covariate into Run.extra so Focaloop._enrich_meta sees float64 too
            val_by_s = dict(zip(out_df["sample"].astype(str), parsed_floats))
            for r in manifest.rna_runs:
                if r.sample in val_by_s and r.extra is not None:
                    r.extra[col] = float(val_by_s[r.sample])

    # Check biological sample design matrix rank, collinearity, and residual degrees of freedom up front
    bio_df = out_df[out_df["sample"].isin(rna_sample_ids)].drop_duplicates("sample").set_index("sample")
    if len(bio_df) < 2:
        raise ValueError(f"Differential expression requires at least 2 biological RNA samples (got {len(bio_df)})")

    if len(design_cols) == 1 and design_cols[0] == fac:
        sub_ab = bio_df[bio_df[fac].astype(str).isin([str(lvl_a), str(lvl_b)])]
        if len(sub_ab) <= 2:
            raise ValueError(
                f"Insufficient biological replicates ({len(sub_ab)} samples across {lvl_a!r} vs {lvl_b!r}) "
                "to estimate dispersion with positive residual degrees of freedom."
            )
    else:
        n_b = len(bio_df)
        cols_x = [np.ones(n_b, dtype=float)]
        for col in design_cols:
            s_col = bio_df[col]
            if col not in factor_cols and pd.api.types.is_numeric_dtype(s_col):
                v_num = s_col.to_numpy(dtype=float)
                if float(np.std(v_num)) <= 0.0:
                    raise ValueError(f"Numeric design covariate {col!r} is constant across all samples")
                cols_x.append(v_num)
            else:
                lvls = sorted(set(s_col.astype(str).tolist()))
                if len(lvls) < 2:
                    raise ValueError(f"Categorical design factor {col!r} has fewer than 2 levels ({lvls})")
                for lv in lvls[1:]:
                    cols_x.append((s_col.astype(str) == lv).to_numpy(dtype=float))
        X_full = np.column_stack(cols_x)
        rk = int(np.linalg.matrix_rank(X_full))
        if rk < X_full.shape[1]:
            raise ValueError(
                f"Design matrix for {design!r} is rank-deficient (rank {rk} < {X_full.shape[1]} columns) - "
                "one or more covariates are collinear or confounded with the experimental group."
            )
        if n_b <= X_full.shape[1]:
            raise ValueError(
                f"Design {design!r} uses {X_full.shape[1]} parameters across {n_b} biological samples "
                "(0 residual degrees of freedom). Remove over-parameterized covariates or wrap categorical codes carefully."
            )
    return out_df


def _build_ortholog_bridge_map(catalog: Any) -> dict[str, str]:
    prim = catalog.primary_assembly
    bridge: dict[str, list[str]] = {}
    for _, ts in catalog.groups.items():
        prim_genes = sorted({t.gene_id for t in ts if t.assembly_id == prim})
        sec_genes = sorted({f"{t.assembly_id}:{t.gene_id}" for t in ts if t.assembly_id != prim})
        if prim_genes and sec_genes:
            joined = ",".join(sec_genes)
            for pg in prim_genes:
                bridge.setdefault(pg, [])
                if joined not in bridge[pg]:
                    bridge[pg].append(joined)
        elif not prim_genes and sec_genes:
            joined = ",".join(sec_genes)
            for t in ts:
                bridge.setdefault(t.gene_id, [])
                if joined not in bridge[t.gene_id]:
                    bridge[t.gene_id].append(joined)
    return {k: ";".join(v) for k, v in bridge.items()}


def _format_de_table(raw_de: pd.DataFrame, bridge_map: dict[str, str]) -> pd.DataFrame:
    if raw_de.empty:
        return pd.DataFrame(
            columns=[
                "gene_id", "gene_name", "ortholog_bridge", "baseMean", "log2FoldChange",
                "lfcSE_deseq2", "lfcSE_total", "ci95_low", "ci95_high", "pvalue", "padj",
                "placement_var_frac", "tech_var_frac", "streamed_reads", "dna_present_prob", "zoomed", "status",
            ]
        )
    df = raw_de.copy()
    se_tot2 = np.square(df["lfc_se_total"].to_numpy(dtype=float))
    var_amb = df["var_placement_ambiguity"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pfrac = np.where(se_tot2 > 0, np.clip(var_amb / se_tot2, 0.0, 1.0), 0.0)
    out = pd.DataFrame(
        {
            "gene_id": df["gene_id"].astype(str),
            "gene_name": df["symbol"].fillna("").astype(str),
            "ortholog_bridge": [bridge_map.get(str(g), "") for g in df["gene_id"]],
            "baseMean": df["base_mean"].astype(float),
            "log2FoldChange": df["log2fc"].astype(float),
            "lfcSE_deseq2": df["lfc_se_deseq2"].astype(float),
            "lfcSE_total": df["lfc_se_total"].astype(float),
            "ci95_low": df["ci95_low"].astype(float),
            "ci95_high": df["ci95_high"].astype(float),
            "pvalue": df["pvalue"].astype(float),
            "padj": df["padj"].astype(float),
            "placement_var_frac": pfrac,
            "tech_var_frac": df["tech_var_fraction"].astype(float),
            "streamed_reads": df["streamed_reads"].astype(float),
            "dna_present_prob": df["dna_present_prob"].astype(float),
            "zoomed": df["status"].astype(str).str.contains("zoom", case=False, na=False),
            "status": df["status"].astype(str),
        }
    )
    return out.sort_values(["padj", "pvalue", "gene_id"], ascending=[True, True, True], na_position="last").reset_index(drop=True)


def _init_empty_ledger(store: FocaloopStateStore, alpha_budget: float = 0.05) -> HypothesisLedger:
    """Initialize unspent alpha-investing ledger without pre-seeding holdout==discovery wealth (#61)."""
    if not store.exists("hypothesis_ledger"):
        store.write(
            "hypothesis_ledger",
            pa.Table.from_pylist([], schema=HYPOTHESIS_LEDGER_SCHEMA),
            schema=HYPOTHESIS_LEDGER_SCHEMA,
        )
    return HypothesisLedger.from_store(store, alpha_budget=alpha_budget)


def run_expressionloop_galaxy(
    *,
    manifest_path: str | Path,
    ref_fasta: str | Path | None = None,
    ref_gtf: str | Path | None = None,
    ref_dbkey: str = "",
    ref_name: str = "",
    ref_value: str = "",
    auto_ref_gtf: bool = False,
    ncbi_accession: str | None = None,
    contrast: str,
    out_de: str | Path,
    out_report: str | Path,
    state_dir: str | Path,
    state_archive: str | Path | None = None,
    out_transcripts: str | Path | None = None,
    secondary_fasta: str | Path | None = None,
    secondary_gtf: str | Path | None = None,
    primary_label: str = "primary",
    secondary_label: str = "secondary",
    extra_references: Sequence[dict[str, Any]] | None = None,
    auto_expand_strains: int = 0,
    auto_expand_taxon: str | None = None,
    reads: Sequence[str] | None = None,
    reads_r1: Sequence[str] | None = None,
    reads_r2: Sequence[str] | None = None,
    read_ids: Sequence[str] | None = None,
    organism_mode: str = "auto",
    source_mode: str = "auto",
    logan_mode: str = "auto",
    logan_min_ka: float = 0.0,
    design: str = "~condition",
    followup_contrasts: Sequence[str] | str | None = None,
    max_byte_fraction: float = 0.12,
    v_tech_target: float = 0.25,
    delta_lfc_stop: float = 0.05,
    min_mapped_per_sample: int | None = None,
    enable_zoom: bool = True,
    enable_abundant_filter: bool = True,
    filter_mode: str = "auto",
    threads: int = 8,
    kernel_names: Sequence[str] | None = None,
    n_tournament_reads: int = 25_000,
) -> dict[str, Any]:
    """Execute full PanExpr v2.4 adaptive RNA-seq DE loop and emit Galaxy outputs & warm state bundle."""
    from .panexpr.catalog import AssemblySpec, build_catalog
    from .panexpr.kernels import Kernel, RunAccumulator, StarKernel, default_kernels, map_pooled
    from .panexpr.loop import Focaloop, LoopConfig
    from .panexpr.regime import AssemblyInput, available_ram_bytes, classify
    from .panexpr.solver import quantify_cohort, strand_mode
    from .panexpr.sra_odp import XsraRunStream, xsra_runnable
    from .panexpr.state import TARGET_COUNTS_SCHEMA, StateStore as PanStateStore
    from .panexpr.streamer import open_run_stream, sample_reads
    from .panexpr.tournament import run_tournament

    t_start = time.time()
    state_root = Path(state_dir).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="focaloop_expression_"))

    try:
        manifest, sample_meta_df, all_local = resolve_expression_manifest(
            manifest_path,
            reads=reads,
            reads_r1=reads_r1,
            reads_r2=reads_r2,
            read_ids=read_ids,
        )
        contrast_list = parse_expression_contrast(contrast)
        fac, lvl_a, lvl_b = contrast_list
        followups = _parse_followup_contrasts(followup_contrasts)
        sample_meta_df = _prepare_and_validate_design_metadata(
            sample_meta_df, manifest, design, contrast_list, followups,
        )

        if source_mode == "sra_odp" and not xsra_runnable():
            raise RuntimeError(
                "Requested AWS SRA Open Data streaming (sra_odp), but 'xsra' binary is missing or incompatible "
                "with this CPU. Select 'Auto' or 'ENA HTTPS 206 byte-range' instead."
            )

        if source_mode == "auto":
            eff_source_mode = "local_bytes" if all_local else "auto"
        else:
            eff_source_mode = source_mode

        refs_dir = work_dir / "refs"
        primary_cvmfs_meta: dict[str, Any] = {}
        primary_label = _sanitize_leading_dash(primary_label) or "primary"
        secondary_label = _sanitize_leading_dash(secondary_label) or "secondary"

        if ncbi_accession and str(ncbi_accession).strip():
            from .ncbi_ref import fetch_ncbi_assembly

            acc_clean = str(ncbi_accession).strip()
            need_gff = ref_gtf is None or not str(ref_gtf).strip()
            _log(f"Fetching NCBI assembly {acc_clean} (annotation={'auto' if need_gff else 'custom override'})...")
            fetched = fetch_ncbi_assembly(acc_clean, refs_dir / "ncbi", need_annotation=need_gff)
            ref_fasta = fetched["fasta"]
            if need_gff:
                ref_gtf = fetched["gff"]
            if primary_label == "primary":
                primary_label = acc_clean
        elif auto_ref_gtf and ref_fasta and (ref_gtf is None or not str(ref_gtf).strip()):
            from .cvmfs_annot import resolve_cvmfs_annotation

            primary_cvmfs_meta = resolve_cvmfs_annotation(
                ref_fasta,
                refs_dir / "cvmfs_annot" / "primary",
                dbkey=ref_dbkey,
                display_name=ref_name,
                value=ref_value,
                log=_log,
            )
            ref_gtf = primary_cvmfs_meta["gtf_path"]
            if primary_label == "primary" and (ref_value or ref_dbkey):
                primary_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ref_value or ref_dbkey).strip()).strip("._-") or "primary"

        if not ref_fasta or not ref_gtf:
            raise ValueError(
                "Provide either a reference genome FASTA and GTF/GFF3 annotation (or --auto-ref-gtf for a built-in genome), or an NCBI assembly accession (--ncbi-accession)."
            )

        p_fna = _gunzip_if_needed(ref_fasta, refs_dir, f"{primary_label}.fna")
        gtf_ext = _sniff_annotation_ext(Path(ref_gtf))
        p_gtf = _gunzip_if_needed(ref_gtf, refs_dir, f"{primary_label}{gtf_ext}")
        org_mode = _detect_organism_mode(p_gtf, organism_mode)

        specs = [AssemblySpec(primary_label, p_fna, p_gtf)]
        used_labels: set[str] = {primary_label}
        used_ncbi_accs: list[str] = []
        if ncbi_accession and str(ncbi_accession).strip():
            used_ncbi_accs.append(str(ncbi_accession).strip().upper())
        if primary_cvmfs_meta.get("ncbi_accession"):
            used_ncbi_accs.append(str(primary_cvmfs_meta["ncbi_accession"]).strip().upper())

        def _unique_label(raw: str, fallback: str) -> str:
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "").strip()).strip("._-") or fallback
            cand = base
            idx_u = 2
            while cand in used_labels:
                cand = f"{base}_{idx_u}"
                idx_u += 1
            used_labels.add(cand)
            return cand

        if bool(secondary_fasta) ^ bool(secondary_gtf):
            raise ValueError("Both --secondary-fasta and --secondary-gtf must be provided together.")
        if secondary_fasta and secondary_gtf:
            sec_lbl = _unique_label(secondary_label, "secondary")
            s_fna = _gunzip_if_needed(secondary_fasta, refs_dir, f"{sec_lbl}.fna")
            s_ext = _sniff_annotation_ext(Path(secondary_gtf))
            s_gtf = _gunzip_if_needed(secondary_gtf, refs_dir, f"{sec_lbl}{s_ext}")
            specs.append(AssemblySpec(sec_lbl, s_fna, s_gtf))

        for idx_e, eref in enumerate(extra_references or [], 1):
            e_acc = str(eref.get("ncbi_accession") or "").strip()
            e_fa = str(eref.get("fasta") or "").strip()
            e_gtf = str(eref.get("gtf") or "").strip()
            e_auto_gtf = bool(eref.get("auto_gtf"))
            e_dbkey = str(eref.get("dbkey") or "").strip()
            e_name = str(eref.get("ref_name") or "").strip()
            e_val = str(eref.get("value") or "").strip()
            e_lbl_raw = str(eref.get("label") or "").strip()
            if e_acc:
                from .ncbi_ref import fetch_ncbi_assembly

                need_e_gff = not e_gtf
                _log(f"Fetching extra NCBI reference {e_acc} (annotation={'auto' if need_e_gff else 'custom override'})...")
                efetch = fetch_ncbi_assembly(e_acc, refs_dir / "ncbi", need_annotation=need_e_gff)
                e_fa = str(efetch["fasta"])
                if need_e_gff:
                    e_gtf = str(efetch["gff"])
                used_ncbi_accs.append(e_acc.upper())
                if not e_lbl_raw:
                    e_lbl_raw = e_acc.upper()
            elif e_auto_gtf and e_fa and not e_gtf:
                from .cvmfs_annot import resolve_cvmfs_annotation

                ecvmfs = resolve_cvmfs_annotation(
                    e_fa,
                    refs_dir / "cvmfs_annot" / f"extra_{idx_e}",
                    dbkey=e_dbkey,
                    display_name=e_name,
                    value=e_val,
                    log=_log,
                )
                e_gtf = str(ecvmfs["gtf_path"])
                if ecvmfs.get("ncbi_accession"):
                    used_ncbi_accs.append(str(ecvmfs["ncbi_accession"]).strip().upper())
                if not e_lbl_raw and (e_val or e_dbkey):
                    e_lbl_raw = e_val or e_dbkey
            if not e_fa or not e_gtf:
                raise ValueError(
                    f"Additional reference #{idx_e} requires either an NCBI assembly accession, a built-in genome with auto-annotation, or both a genome FASTA and GTF/GFF3 annotation."
                )
            e_lbl = _unique_label(e_lbl_raw, f"extra_{idx_e}")
            efna_p = _gunzip_if_needed(e_fa, refs_dir, f"{e_lbl}.fna")
            eext = _sniff_annotation_ext(Path(e_gtf))
            egtf_p = _gunzip_if_needed(e_gtf, refs_dir, f"{e_lbl}{eext}")
            specs.append(AssemblySpec(e_lbl, efna_p, egtf_p))

        auto_expanded_info: list[dict[str, Any]] = []
        n_expand = int(auto_expand_strains or 0)
        if n_expand > 0:
            from .ncbi_ref import discover_related_ncbi_assemblies, fetch_ncbi_assembly, lookup_ncbi_assembly_taxon

            taxon_q = str(auto_expand_taxon or "").strip()
            excl_strains: list[str] = []
            if not taxon_q:
                if ncbi_accession and str(ncbi_accession).strip():
                    tax_meta = lookup_ncbi_assembly_taxon(str(ncbi_accession).strip())
                    taxon_q = tax_meta["species_name"] or str(tax_meta.get("tax_id") or "")
                    if tax_meta.get("strain"):
                        excl_strains.append(tax_meta["strain"])
                    _log(
                        f"Auto-expand inferred species {taxon_q!r} from primary accession {ncbi_accession!r}"
                        + (f" (excluding primary strain {tax_meta['strain']!r})" if tax_meta.get("strain") else "")
                    )
                elif ref_dbkey or ref_name or ref_value:
                    from .cvmfs_annot import infer_species_from_cached_meta

                    cmeta = (
                        primary_cvmfs_meta
                        if primary_cvmfs_meta.get("species_name")
                        else infer_species_from_cached_meta(dbkey=ref_dbkey, display_name=ref_name, value=ref_value)
                    )
                    taxon_q = str(cmeta.get("species_name") or "").strip()
                    c_strain = str(cmeta.get("strain") or "").strip()
                    if c_strain:
                        excl_strains.append(c_strain)
                    if cmeta.get("ncbi_accession_hint"):
                        used_ncbi_accs.append(str(cmeta["ncbi_accession_hint"]).strip().upper())
                    if not taxon_q:
                        raise ValueError(
                            f"Could not infer species automatically from built-in genome {ref_name or ref_dbkey or ref_value!r}. "
                            "Enter a species name or NCBI taxonomy ID in the Auto-expand taxon field."
                        )
                    _log(
                        f"Auto-expand inferred species {taxon_q!r} from built-in genome {ref_name or ref_dbkey or ref_value!r}"
                        + (f" (excluding primary strain {c_strain!r})" if c_strain else "")
                    )
                else:
                    raise ValueError(
                        "Auto-expanding related RefSeq strains requires a species name or NCBI taxonomy ID "
                        "when the primary genome comes from your Galaxy History."
                    )
            cands = discover_related_ncbi_assemblies(
                taxon_q,
                max_strains=n_expand,
                exclude_accessions=used_ncbi_accs,
                exclude_strains=excl_strains,
            )
            if not cands:
                _log(f"Auto-expand found no additional annotated RefSeq assemblies for {taxon_q!r} - continuing with {len(specs)} reference(s).")
            else:
                added = 0
                for cand in cands:
                    if added >= n_expand:
                        break
                    c_acc = cand["accession"]
                    try:
                        cfetch = fetch_ncbi_assembly(c_acc, refs_dir / "ncbi", need_annotation=True)
                    except Exception as exc:
                        _log(f"Skipping candidate {c_acc} ({type(exc).__name__}: {exc})")
                        continue
                    c_lbl = _unique_label(c_acc, f"strain_{added + 1}")
                    cfna_p = _gunzip_if_needed(cfetch["fasta"], refs_dir, f"{c_lbl}.fna")
                    cext = _sniff_annotation_ext(Path(cfetch["gff"]))
                    cgtf_p = _gunzip_if_needed(cfetch["gff"], refs_dir, f"{c_lbl}{cext}")
                    specs.append(AssemblySpec(c_lbl, cfna_p, cgtf_p))
                    used_ncbi_accs.append(c_acc)
                    auto_expanded_info.append({**cand, "label": c_lbl})
                    added += 1
                _log(
                    f"Auto-expanded catalog with {added} distinct RefSeq strain(s) for {taxon_q!r}: "
                    f"{[x['accession'] for x in auto_expanded_info]}"
                )

        _log(f"Building {org_mode} catalog across {[s.assembly_id for s in specs]}...")
        catalog = build_catalog(specs, org_mode, primary_label, work_dir / "tmp", decoys="auto")
        cat_dir = state_root / "catalog"
        catalog.save(cat_dir)
        write_json(cat_dir / "link_stats.json", catalog.link_stats)

        # 0.4.1 Logan unitig streaming branch for genuine public INSDC SRA/ENA cohorts (L14)
        eff_logan_mode = str(logan_mode or "auto").strip().lower()
        if eff_logan_mode not in ("auto", "logan_only", "off"):
            raise ValueError(f"Unsupported logan_mode {logan_mode!r} - expected 'auto', 'logan_only', or 'off'")

        if eff_logan_mode in ("auto", "logan_only") and not all_local:
            from .logan import is_insdc_run, probe_logan_runs
            from .panexpr.logan_expr import LoganZeroMappingError, run_logan_expression_cohort

            rna_runs = manifest.rna_runs
            rna_run_ids = [r.run_id for r in rna_runs]
            all_insdc = bool(rna_runs) and all(
                (r.extra or {}).get("_source_kind") == "insdc" and is_insdc_run(r.run_id)
                for r in rna_runs
            )
            if not all_insdc:
                if eff_logan_mode == "logan_only":
                    raise ValueError(
                        "Logan-only mode requires public SRA/ENA run accessions (SRR, ERR, or DRR) for every sample in the manifest."
                    )
            else:
                probe = probe_logan_runs(rna_run_ids, kind="unitigs", max_workers=min(32, max(8, int(threads) * 2)))
                n_avail = len(probe["logan_runs"])
                n_total = len(probe["probes"])
                stub_runs = probe.get("stub_runs") or []
                if n_avail == n_total and n_total > 0:
                    _log(
                        f"Logan probe: {n_avail}/{n_total} runs available on logan-pub/u/ "
                        f"({probe['total_logan_bytes'] / 1e6:.1f} MB total compressed unitigs)"
                    )
                    pan_state_dir = state_root / "panexpr_state"
                    logan_res: dict[str, Any] | None = None
                    try:
                        logan_res = run_logan_expression_cohort(
                            catalog=catalog,
                            manifest=manifest,
                            sample_meta_df=sample_meta_df,
                            contrast_list=contrast_list,
                            design=design,
                            gtf_path=p_gtf,
                            work_dir=work_dir,
                            pan_state_dir=pan_state_dir,
                            threads=threads,
                            min_ka=float(logan_min_ka),
                            write_transcript_table=False,
                            out_gene_abundances_tsv=(Path(out_transcripts) if out_transcripts else None),
                            log=_log,
                        )
                    except LoganZeroMappingError as z_exc:
                        if eff_logan_mode == "logan_only":
                            raise
                        _log(
                            f"Logan unitig fallback triggered ({z_exc}) - "
                            "switching cohort uniformly to raw-read streaming"
                        )
                        shutil.rmtree(pan_state_dir, ignore_errors=True)
                        logan_res = None

                    if logan_res is not None:
                        bridge_map = _build_ortholog_bridge_map(catalog)
                        de_df = _format_de_table(logan_res["de_raw"], bridge_map)
                        Path(out_de).parent.mkdir(parents=True, exist_ok=True)
                        de_df.to_csv(out_de, sep="\t", index=False, float_format="%.6g")

                        cat_summary = {
                            "primary_assembly": catalog.primary_assembly,
                            "assemblies": [s.assembly_id for s in specs],
                            "n_targets": len(catalog),
                            "n_groups": len(catalog.groups),
                            "link_stats": catalog.link_stats,
                        }
                        loop_summary = {
                            "source_mode": "logan_unitigs",
                            "logan_mode": eff_logan_mode,
                            "strand_mode": "unstranded (Logan k=31 unitigs)",
                            "tournament_reads": 0,
                            "tournament_seconds": 0.0,
                            "winner": logan_res["aligner"],
                            "iterations": 1,
                            "stop_reason": f"LOGAN_UNITIGS_COMPLETE ({logan_res['wire_reduction_x']:.1f}x wire reduction)",
                            "total_archive_bytes": logan_res["total_archive_bytes"],
                            "bytes_streamed_global": logan_res["logan_wire_bytes"],
                            "bytes_streamed_zoom": 0,
                            "bytes_streamed_total": logan_res["logan_wire_bytes"],
                            "pct_archive_streamed": logan_res["pct_archive"],
                            "wire_reduction_x": logan_res["wire_reduction_x"],
                            "wall_seconds": round(time.time() - t_start, 2),
                            "v_tech_target": v_tech_target,
                            "delta_lfc_stop": delta_lfc_stop,
                            "assembly_prior": {catalog.primary_assembly: 1.0},
                            "telemetry": [],
                            "zoom": {
                                "n_antisense_genes": logan_res["n_antisense_genes"],
                                "n_zoom_candidates": logan_res["n_zoom_candidates"],
                            },
                            "zoom_ran": False,
                            "holdout_saved": False,
                            "kpf_fallback_runs": logan_res.get("kpf_fallback_runs", []),
                            "logan_runs": logan_res["run_stats"],
                        }
                        contrast_pretty = f"{fac}: {lvl_a} vs {lvl_b}"
                        render_panexpr_report(
                            out_report,
                            contrast_str=contrast_pretty,
                            winner=logan_res["aligner"],
                            scoreboard=[],
                            per_sample_rows=logan_res["per_sample_rows"],
                            catalog_summary=cat_summary,
                            loop_summary=loop_summary,
                            de_df=de_df,
                        )
                        sample_meta_df.to_parquet(state_root / "sample_metadata.parquet", index=False)
                        store = FocaloopStateStore(state_root)
                        _init_empty_ledger(store, alpha_budget=0.05)
                        state_meta = {
                            "engine": "panexpr",
                            "version": "0.4.1",
                            "blueprint_id": "panexpr",
                            "primary_contrast": contrast_list,
                            "design": design,
                            "organism_mode": org_mode,
                            "length_offsets": False,
                            "holdout_saved": False,
                            "catalog_summary": cat_summary,
                            "loop_summary": loop_summary,
                            "samples": sample_meta_df.to_dict(orient="records"),
                        }
                        write_json(state_root / STATE_META, state_meta)
                        if state_archive:
                            pack_state(state_root, state_archive)
                        return state_meta
                elif eff_logan_mode == "logan_only":
                    missing = ", ".join(probe["fallback_runs"])
                    stub_note = f" (empty <=100 B stubs: {', '.join(stub_runs)})" if stub_runs else ""
                    raise ValueError(
                        f"Logan unitigs are not available for all requested accessions (missing: {missing}){stub_note}. "
                        "Remove stub/missing accessions or switch Logan mode to Automatic to fall back to raw SRA/ENA streaming."
                    )
                else:
                    stub_note = f" ({len(stub_runs)} empty <=100 B stub(s): {', '.join(stub_runs[:5])})" if stub_runs else ""
                    _log(
                        f"Logan probe: {n_avail}/{n_total} runs usable{stub_note} - falling back to uniform raw-read streaming "
                        "to keep quantification consistent across all samples."
                    )
        elif eff_logan_mode == "logan_only" and all_local:
            raise ValueError(
                "Logan-only mode applies to public SRA/ENA run accessions, not local Galaxy History FASTQ files."
            )

        stream_cache = work_dir / "stream_cache"
        stream_cache.mkdir(parents=True, exist_ok=True)
        first_run = manifest.rna_runs[0]
        r1, r2, sample_bytes = sample_reads(first_run, n_tournament_reads, eff_source_mode, cache_dir=stream_cache)
        _log(
            f"Stage 1 tournament sample: {len(r1):,} {first_run.layout} reads "
            f"from {first_run.run_id} ({sample_bytes / 1e6:.2f} MB)"
        )

        fna_targets = catalog.write_fasta(work_dir / "targets.fna")
        idx_dir = work_dir / "kernel_idx"
        idx_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "tmp").mkdir(parents=True, exist_ok=True)
        cands = default_kernels(threads, tmp_dir=work_dir / "tmp", include_star=False)
        explicit_kernels = bool(kernel_names)
        if explicit_kernels:
            allow = set(kernel_names or [])
            cands = [k for k in cands if k.name in allow]
            if not cands:
                raise ValueError(f"Unknown --kernels selection: {list(kernel_names or [])}")

        ai = [AssemblyInput(a.assembly_id, a.fasta, a.annotation) for a in specs]
        regime_dec = classify(
            ai,
            [s for s, _ in r1],
            [s for s, _ in r2] if r2 else None,
            kernels=[k.name for k in cands],
            use_island_pc_router=True,
            ipc_index_dir=None,
            transcript_fasta=(fna_targets if org_mode == "eukaryotic" else None),
            persist_ipc_cache=False,
        )
        if not kernel_names and regime_dec.excluded:
            pruned = [k for k in cands if k.name not in regime_dec.excluded]
            if not pruned:
                reasons_str = ", ".join(f"{kn}: {why}" for kn, why in regime_dec.excluded.items())
                raise ValueError(
                    f"All candidate aligners {[k.name for k in cands]} were excluded by pre-tournament regime checks ({reasons_str})"
                )
            cands = pruned
            _log(
                f"Regime classifier selected {[k.name for k in cands]} "
                f"(RAM budget {regime_dec.ram_budget_bytes / (1 << 30):.1f} GiB)"
            )
        elif kernel_names and regime_dec.excluded:
            pruned = [k for k in cands if k.name not in regime_dec.excluded]
            if not pruned:
                reasons_str = ", ".join(f"{kn}: {why}" for kn, why in regime_dec.excluded.items())
                raise ValueError(
                    f"Explicitly requested aligners {[k.name for k in cands]} were excluded by regime checks ({reasons_str})"
                )
            cands = pruned

        live: list[Kernel] = []
        for k in cands:
            if not k.available():
                if explicit_kernels and len(cands) == 1:
                    raise RuntimeError(f"Requested aligner kernel {k.name!r} is not installed on PATH")
                continue
            try:
                if isinstance(k, StarKernel):
                    k.ensure_index(fna_targets, idx_dir, catalog=catalog, genome_fasta=p_fna)
                else:
                    k.ensure_index(fna_targets, idx_dir)
                live.append(k)
            except Exception as exc:
                if explicit_kernels:
                    raise RuntimeError(f"Explicitly requested aligner {k.name!r} failed index build: {exc}") from exc
                _log(f"Kernel {k.name} index skipped ({type(exc).__name__} - {exc})")

        if not live:
            raise RuntimeError(f"No mapping kernels available from candidates {[k.name for k in cands]}")

        n_synth = min(600, max(120, len(catalog.gene_targets) // 4))
        t_tour0 = time.time()
        winner, board = run_tournament(
            live, catalog, r1, r2, first_run.layout, n_synthetic_per_regime=n_synth, regime=regime_dec, log=_log
        )
        t_tour = time.time() - t_tour0
        win_kernel = next(k for k in live if k.name == winner)
        del live, cands
        catalog.release_decoy_sequences()
        import gc
        gc.collect()
        _log(f"Stage 1 winner: {winner} ({t_tour:.1f}s)")

        max_pct = float(max_byte_fraction) * 100.0 if max_byte_fraction <= 1.0 else float(max_byte_fraction)
        init_frac = min(0.004, max_pct / 400.0) if max_pct < 100.0 else 0.25
        floor_val = int(min_mapped_per_sample) if (min_mapped_per_sample is not None and min_mapped_per_sample > 0) else None
        loop_cfg = LoopConfig(
            contrast=contrast_list,
            design=design,
            source_mode=eff_source_mode,
            cache_dir=stream_cache,
            initial_fraction=max(0.0015, init_frac),
            growth=2.2,
            max_iterations=12,
            min_iterations=2,
            max_bytes_pct=max_pct,
            delta_lfc_stop=delta_lfc_stop,
            tech_var_stop=v_tech_target,
            zoom=enable_zoom,
            zoom_extra_fraction=min(0.01, max_pct / 100.0 * 0.15),
            length_offsets=(org_mode == "eukaryotic"),
            followup_contrasts=followups,
            fetch_workers=min(8, max(1, threads)),
            sample_metadata=sample_meta_df,
            min_mapped_per_sample=floor_val,
            abundant_sieve=bool(enable_abundant_filter),
            sieve_mode=filter_mode if filter_mode in ("auto", "include", "deplete") else "auto",
        )

        pan_state_dir = state_root / "panexpr_state"
        loop = Focaloop(catalog, manifest, win_kernel, loop_cfg, pan_state_dir, log=_log)
        holdout_saved = False
        bytes_by_run: dict[str, int] = {}
        try:
            rep = loop.run(close_streams=False)
            for rid, acc in loop.accs.items():
                st_obj = loop.streams.get(rid)
                bytes_by_run[rid] = int(st_obj.consumed_bytes) if st_obj is not None else int(acc.bytes_consumed)

            # Collect a bounded independent holdout slice directly from the open streams past discovery+zoom (#16, #61)
            max_hold_wave_bytes = min(16 << 20, max(262_144, available_ram_bytes() // 64))
            decoy_set = catalog.decoy_ids
            hold_accs: dict[str, RunAccumulator] = {}
            pending_wave: list[tuple[str, list, list | None, Any]] = []
            pending_wave_bytes = 0

            def _flush_holdout_wave() -> None:
                nonlocal pending_wave_bytes
                if not pending_wave:
                    return
                for paired_flag in (True, False):
                    sub_w = [b for b in pending_wave if (b[2] is not None) == paired_flag]
                    if sub_w:
                        mapped_w = map_pooled(win_kernel, sub_w)
                        for rid_w, mres_w in mapped_w.items():
                            hold_accs[rid_w].add(mres_w)
                pending_wave.clear()
                pending_wave_bytes = 0

            for r in manifest.rna_runs:
                rs = loop.streams.get(r.run_id)
                owned_rs = False
                try:
                    if rs is None:
                        used_b = bytes_by_run.get(r.run_id, 0)
                        if used_b + 131_072 >= r.total_bytes:
                            continue
                        rs = open_run_stream(r, eff_source_mode, stream_cache)
                        owned_rs = True
                        if isinstance(rs, XsraRunStream):
                            continue
                        rs.skip_bytes(used_b)
                    elif isinstance(rs, XsraRunStream):
                        continue

                    rem_after = rs.total_bytes - rs.consumed_bytes
                    if rs.exhausted or rem_after <= 131_072:
                        continue
                    hb_target = max(131_072, min(int(r.total_bytes * 0.005), 8 << 20, rem_after))
                    hr1: list = []
                    hr2: list | None = None
                    rem_h = hb_target
                    while rem_h > 0 and not rs.exhausted:
                        step_h = min(rem_h, max_hold_wave_bytes)
                        prev_h = rs.consumed_bytes
                        c1_h, c2_h = rs.advance_bytes(step_h)
                        delta_h = rs.consumed_bytes - prev_h
                        if not c1_h and delta_h <= 0:
                            break
                        rem_h -= max(step_h, delta_h)
                        if c1_h:
                            hr1.extend(c1_h)
                            if c2_h is not None:
                                if hr2 is None:
                                    hr2 = []
                                hr2.extend(c2_h)
                except (OSError, RuntimeError) as io_exc:
                    _log(f"Holdout stream read skipped for {r.run_id} ({type(io_exc).__name__} - {io_exc})")
                    continue
                finally:
                    if owned_rs and rs is not None and hasattr(rs, "close"):
                        rs.close()

                if hr1:
                    acc_h = RunAccumulator(run=r, decoy_ids=decoy_set)
                    hold_accs[r.run_id] = acc_h
                    est_batch_b = sum(len(s) * 2 for s, _ in hr1) * (2 if hr2 is not None else 1)
                    if pending_wave and (pending_wave_bytes + est_batch_b > max_hold_wave_bytes):
                        _flush_holdout_wave()
                    pending_wave.append((r.run_id, hr1, hr2, acc_h.calibrator))
                    pending_wave_bytes += est_batch_b

            _flush_holdout_wave()
            if len(hold_accs) >= 2:
                tc_hold, _ = quantify_cohort(
                    hold_accs,
                    catalog,
                    dna_prob={},
                    dna_counts={},
                    strand_priors=rep.strand_priors,
                    em_iters=4,
                    include_posteriors=False,
                    include_decoys=False,
                )
                PanStateStore(pan_state_dir).write("target_counts_holdout", tc_hold, TARGET_COUNTS_SCHEMA)
                holdout_saved = True
        finally:
            loop.close_streams()

        # Ensure zoom_context.json contract is persisted whenever target_counts_zoom.parquet was written (#66)
        zoom_pq = pan_state_dir / "target_counts_zoom.parquet"
        zoom_ctx_file = pan_state_dir / "zoom_context.json"
        if zoom_pq.is_file() and not zoom_ctx_file.is_file():
            zctx = getattr(loop, "_last_zoom_ctx", None) or {}
            ds_obj = zctx.get("depth_scale")
            ds_dict = {str(k): float(v) for k, v in ds_obj.items()} if ds_obj is not None else {}
            write_json(
                zoom_ctx_file,
                {
                    "zoomed_groups": [str(g) for g in zctx.get("groups", [])],
                    "sample_depth_scale": ds_dict,
                },
            )

        bridge_map = _build_ortholog_bridge_map(catalog)
        de_raw = rep.de_postzoom if (rep.de_postzoom is not None and not rep.de_postzoom.empty) else rep.de_prezoom
        de_df = _format_de_table(de_raw, bridge_map)
        Path(out_de).parent.mkdir(parents=True, exist_ok=True)
        de_df.to_csv(out_de, sep="\t", index=False, float_format="%.6g")

        if out_transcripts:
            tx_pq = pan_state_dir / "transcript_abundances.parquet"
            tx_df = pq.read_table(tx_pq).to_pandas() if tx_pq.exists() else pd.DataFrame()
            Path(out_transcripts).parent.mkdir(parents=True, exist_ok=True)
            tx_df.to_csv(out_transcripts, sep="\t", index=False, float_format="%.6g")

        # Column-projected Arrow aggregation without loading 50M-row DataFrame into pandas (#18)
        tc_proj = pq.read_table(pan_state_dir / "target_counts.parquet", columns=["run_id", "effective_count"])
        if tc_proj.num_rows > 0:
            tc_agg = tc_proj.group_by("run_id").aggregate([("effective_count", "sum")])
            mapped_by_run = {
                str(rid): float(val or 0.0)
                for rid, val in zip(tc_agg.column("run_id").to_pylist(), tc_agg.column("effective_count_sum").to_pylist())
            }
        else:
            mapped_by_run = {}
        del tc_proj

        per_sample_rows = []
        for r in manifest.runs:
            sb = bytes_by_run.get(r.run_id, 0)
            tb = max(r.total_bytes, 1)
            per_sample_rows.append(
                {
                    "sample_id": r.sample,
                    "condition": r.condition,
                    "run_id": r.run_id,
                    "layout": r.layout,
                    "archive_mb": r.total_bytes / 1e6,
                    "streamed_mb": sb / 1e6,
                    "pct_streamed": 100.0 * sb / tb,
                    "mapped_reads": int(round(mapped_by_run.get(r.run_id, 0.0))),
                }
            )

        # Require >0 mapped reads across EVERY RNA sample in the cohort, not only primary contrast levels (L9)
        mapped_by_sample: dict[str, float] = {}
        for r in manifest.rna_runs:
            z_acc = getattr(loop.accs[r.run_id], "zoom", None)
            z_mapped = float(z_acc.n_mapped) if z_acc is not None else 0.0
            mapped_by_sample[r.sample] = mapped_by_sample.get(r.sample, 0.0) + float(loop.accs[r.run_id].n_mapped) + z_mapped
        empty_samples = [s for s, m in mapped_by_sample.items() if m <= 0]
        if empty_samples:
            raise RuntimeError(
                f"Zero mapped reads in RNA sample(s): {', '.join(sorted(empty_samples))}. "
                "Verify FASTQ inputs, reference assembly match, or remote archive availability."
            )

        strand_p = float(np.mean([v for k, v in rep.strand_priors.items() if not manifest.run(k).is_wgs])) if rep.strand_priors else 0.5
        strand_label = strand_mode(strand_p)
        cat_summary = {
            "primary_assembly": catalog.primary_assembly,
            "assemblies": [s.assembly_id for s in specs],
            "n_targets": len(catalog),
            "n_groups": len(catalog.groups),
            "link_stats": catalog.link_stats,
        }
        zoom_dict = asdict(rep.zoom) if rep.zoom is not None else None
        if zoom_dict and "sieve_mode" in zoom_dict:
            zoom_dict["filter_mode"] = zoom_dict.pop("sieve_mode")
        loop_summary = {
            "source_mode": eff_source_mode,
            "strand_mode": strand_label,
            "tournament_reads": len(r1),
            "tournament_seconds": round(t_tour, 2),
            "winner": winner,
            "iterations": rep.iterations,
            "stop_reason": rep.stop_reason,
            "total_archive_bytes": manifest.total_archive_bytes,
            "bytes_streamed_global": rep.bytes_global,
            "bytes_streamed_zoom": rep.bytes_zoom,
            "bytes_streamed_total": rep.bytes_global + rep.bytes_zoom,
            "pct_archive_streamed": rep.pct_archive,
            "wall_seconds": round(time.time() - t_start, 2),
            "v_tech_target": v_tech_target,
            "delta_lfc_stop": delta_lfc_stop,
            "assembly_prior": {k: round(float(v), 4) for k, v in (rep.assembly_prior or {}).items()},
            "telemetry": rep.telemetry,
            "zoom": zoom_dict,
            "zoom_ran": bool(rep.zoom_ran),
            "followups": rep.followups,
            "malformed_skipped": rep.malformed_skipped,
            "skipped_unpaired_mates": rep.skipped_unpaired_mates,
            "holdout_saved": bool(holdout_saved),
        }
        contrast_pretty = f"{fac}: {lvl_a} vs {lvl_b}"
        render_panexpr_report(
            out_report,
            contrast_str=contrast_pretty,
            winner=winner,
            scoreboard=board,
            per_sample_rows=per_sample_rows,
            catalog_summary=cat_summary,
            loop_summary=loop_summary,
            de_df=de_df,
        )

        sample_meta_df.to_parquet(state_root / "sample_metadata.parquet", index=False)
        store = FocaloopStateStore(state_root)
        _init_empty_ledger(store, alpha_budget=0.05)
        state_meta = {
            "engine": "panexpr",
            "version": "0.4.1",
            "blueprint_id": "panexpr",
            "primary_contrast": contrast_list,
            "design": design,
            "organism_mode": org_mode,
            "length_offsets": (org_mode == "eukaryotic"),
            "holdout_saved": bool(holdout_saved),
            "catalog_summary": cat_summary,
            "loop_summary": loop_summary,
            "samples": sample_meta_df.to_dict(orient="records"),
        }
        write_json(state_root / STATE_META, state_meta)
        if state_archive:
            pack_state(state_root, state_archive)
        return state_meta
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def prebuild_expressionloop_native_libs() -> dict[str, str]:
    """Compile and load all C99 shared libraries used by ExpressionLoop (PanExpr v2.4)."""
    from .panexpr import _fast_c
    from .panexpr._medvedev_compile import compile_c_lib

    pan_dir = Path(_fast_c.__file__).resolve().parent
    built: dict[str, str] = {}
    em_lib = _fast_c.get_fast_lib()
    if em_lib is None:
        raise RuntimeError("Failed to compile panexpr v2.4 _fast_c shared library")
    built["panexpr_fast_c"] = str(getattr(em_lib, "_name", "loaded"))
    for name, src in (
        ("island_pc", "island_pc_core.c"),
        ("ust", "ust_kernel.c"),
        ("sam_unordered", "sam_unordered_core.c"),
        ("dothash", "dothash_sketch.c"),
    ):
        lib = compile_c_lib(name, pan_dir / src)
        built[f"panexpr_{name}"] = str(getattr(lib, "_name", "loaded"))
    from .panexpr.logan_expr import ensure_expr_logan_bins

    prep_bin, ec_bin = ensure_expr_logan_bins()
    built["panexpr_prep_expr_unitigs"] = str(prep_bin)
    built["panexpr_expr_sam_ec_v2"] = str(ec_bin)
    return built
