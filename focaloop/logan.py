"""Shared public Logan archive (unitigs & contigs) probing and ENA run-length metadata helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import socket
import time
from typing import Any, Sequence

_INSDC_RUN_RE = re.compile(r"^[SED]RR\d+$", re.IGNORECASE)

LOGAN_UNITIGS_URL_TEMPLATE = "https://s3.amazonaws.com/logan-pub/u/{acc}/{acc}.unitigs.fa.zst"
LOGAN_CONTIGS_URL_TEMPLATE = "https://s3.amazonaws.com/logan-pub/c/{acc}/{acc}.contigs.fa.zst"
# Empty zstd frames in logan-pub are ~55 bytes and decompress to 0 bytes; reject <=100 B stubs
MIN_LOGAN_ARCHIVE_BYTES = 100


def is_insdc_run(text: str) -> bool:
    return bool(_INSDC_RUN_RE.match(str(text or "").strip()))


def logan_unitigs_url(acc: str) -> str:
    clean = acc.strip().upper()
    return LOGAN_UNITIGS_URL_TEMPLATE.format(acc=clean)


def logan_contigs_url(acc: str) -> str:
    clean = acc.strip().upper()
    return LOGAN_CONTIGS_URL_TEMPLATE.format(acc=clean)


def _ipv4_session():
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    sess = requests.Session()
    sess.headers["User-Agent"] = "focaloop/0.4.1"
    return sess


def probe_logan_url(
    acc: str,
    *,
    kind: str = "unitigs",
    timeout: float = 8.0,
    retries: int = 3,
) -> dict[str, Any]:
    """HTTP HEAD check against public Logan S3 bucket for unitigs (u/) or contigs (c/)."""
    clean = acc.strip().upper()
    if not is_insdc_run(clean):
        return {
            "accession": clean,
            "kind": kind,
            "available": False,
            "content_length": 0,
            "status": 0,
            "url": "",
        }
    url = logan_unitigs_url(clean) if kind == "unitigs" else logan_contigs_url(clean)
    sess = _ipv4_session()
    last_status = 0
    for attempt in range(max(1, retries)):
        try:
            r = sess.head(url, timeout=timeout, allow_redirects=True)
            last_status = int(r.status_code)
            if r.status_code == 200:
                cl = int(r.headers.get("Content-Length") or 0)
                is_stub = 0 < cl <= MIN_LOGAN_ARCHIVE_BYTES
                return {
                    "accession": clean,
                    "kind": kind,
                    "available": cl > MIN_LOGAN_ARCHIVE_BYTES,
                    "is_stub": is_stub,
                    "content_length": cl,
                    "status": 200,
                    "url": url,
                }
            if r.status_code in (403, 404):
                return {
                    "accession": clean,
                    "kind": kind,
                    "available": False,
                    "content_length": 0,
                    "status": last_status,
                    "url": url,
                }
        except Exception:
            pass
        if attempt + 1 < retries:
            time.sleep(0.5 * (attempt + 1))
    return {
        "accession": clean,
        "kind": kind,
        "available": False,
        "content_length": 0,
        "status": last_status,
        "url": url,
    }


def probe_logan_runs(
    accessions: Sequence[str],
    *,
    kind: str = "unitigs",
    max_workers: int = 8,
    timeout: float = 8.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Partition run accessions into available Logan runs and fallback runs."""
    ordered: list[str] = []
    seen: set[str] = set()
    for a in accessions:
        c = str(a or "").strip().upper()
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    probes: dict[str, dict[str, Any]] = {}
    if ordered:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(ordered))) as ex:
            for res in ex.map(
                lambda x: probe_logan_url(x, kind=kind, timeout=timeout, retries=retries),
                ordered,
            ):
                probes[res["accession"]] = res
    logan_runs = [a for a in ordered if probes.get(a, {}).get("available")]
    fallback_runs = [a for a in ordered if not probes.get(a, {}).get("available")]
    stub_runs = [a for a in ordered if probes.get(a, {}).get("is_stub")]
    total_logan_bytes = sum(int(probes[a].get("content_length", 0)) for a in logan_runs)
    return {
        "kind": kind,
        "logan_runs": logan_runs,
        "fallback_runs": fallback_runs,
        "stub_runs": stub_runs,
        "probes": probes,
        "total_logan_bytes": total_logan_bytes,
    }


def fetch_ena_run_lengths(
    runs: Sequence[str],
    timeout: float = 30.0,
    *,
    retries: int = 3,
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """Query ENA filereport one accession per request with retries/backoff and safe per-row parsing."""
    clean_runs: list[str] = []
    seen: set[str] = set()
    for r in runs:
        s = str(r or "").strip().upper()
        if is_insdc_run(s) and s not in seen:
            seen.add(s)
            clean_runs.append(s)
    if not clean_runs:
        return {}

    def _parse_rows(text: str) -> dict[str, dict[str, Any]]:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            raise ValueError("ENA filereport returned header-only or empty response")
        hdr = [h.strip() for h in lines[0].split("\t")]
        idx = {h: j for j, h in enumerate(hdr)}
        if "run_accession" not in idx:
            raise ValueError("ENA filereport missing run_accession header")
        parsed_rows: dict[str, dict[str, Any]] = {}
        for ln in lines[1:]:
            p = ln.split("\t")
            if len(p) <= idx["run_accession"]:
                continue
            try:
                acc = p[idx["run_accession"]].strip().upper()
                if not is_insdc_run(acc):
                    continue
                rc_s = p[idx["read_count"]].strip() if "read_count" in idx and idx["read_count"] < len(p) else ""
                bc_s = p[idx["base_count"]].strip() if "base_count" in idx and idx["base_count"] < len(p) else ""
                rc = int(float(rc_s)) if rc_s else 0
                bc = int(float(bc_s)) if bc_s else 0
                lay = (
                    p[idx["library_layout"]].strip().upper()
                    if "library_layout" in idx and idx["library_layout"] < len(p)
                    else ""
                )
                fb_raw = (
                    p[idx["fastq_bytes"]].strip()
                    if "fastq_bytes" in idx and idx["fastq_bytes"] < len(p)
                    else ""
                )
                fb = sum(int(x) for x in re.split(r"[;,]", fb_raw) if x.strip().isdigit())
                mates = 2 if lay == "PAIRED" else 1
                read_len = (bc / (rc * mates)) if (rc > 0 and bc > 0) else 100.0
                kmers_per_frag = max(2.0, mates * max(1.0, read_len - 30.0))
                parsed_rows[acc] = {
                    "read_count": rc,
                    "base_count": bc,
                    "fastq_bytes": fb,
                    "layout": "PE" if mates == 2 else "SE",
                    "mates": mates,
                    "read_len": round(read_len, 2),
                    "kmers_per_frag": round(kmers_per_frag, 2),
                }
            except (ValueError, TypeError, IndexError):
                continue
        if not parsed_rows:
            raise ValueError("ENA filereport contained zero valid run rows")
        return parsed_rows

    def _fetch_single(acc: str) -> dict[str, dict[str, Any]]:
        url = (
            "https://www.ebi.ac.uk/ena/portal/api/filereport?"
            f"accession={acc}&result=read_run"
            "&fields=run_accession,read_count,base_count,library_layout,fastq_bytes&format=tsv"
        )
        sess = _ipv4_session()
        n_tries = max(1, int(retries))
        for attempt in range(n_tries):
            try:
                r = sess.get(url, timeout=timeout)
                r.raise_for_status()
                return _parse_rows(r.text)
            except Exception:
                if attempt + 1 < n_tries:
                    time.sleep(0.5 * (attempt + 1))
        return {}

    out: dict[str, dict[str, Any]] = {}
    if len(clean_runs) == 1 or max_workers <= 1:
        for acc in clean_runs:
            out.update(_fetch_single(acc))
    else:
        with ThreadPoolExecutor(max_workers=min(int(max_workers), len(clean_runs))) as ex:
            for item in ex.map(_fetch_single, clean_runs):
                out.update(item)
    return out
