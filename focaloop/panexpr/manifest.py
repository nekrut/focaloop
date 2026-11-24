"""Run manifests: ENA accessions, layouts (PE/SE), exact byte sizes and URLs.

A :class:`Run` is one sequencing run with one (SE) or two (PE) FASTQ files.
Manifests are built either from a saved ENA ``filereport`` TSV or fetched
live from the ENA portal API, and may be annotated with local staged paths.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
import io
from pathlib import Path
import socket

ENA_FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = (
    "run_accession,sample_alias,sample_title,library_layout,library_strategy,"
    "read_count,base_count,fastq_bytes,fastq_ftp,fastq_md5"
)


@dataclass(frozen=True)
class FastqFile:
    url: str
    bytes: int
    md5: str = ""
    local_path: str | None = None

    @property
    def basename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Run:
    run_id: str
    sample: str
    condition: str
    layout: str                      # "PE" or "SE"
    strategy: str                    # "RNA-Seq" or "WGS"
    read_count: int
    files: tuple[FastqFile, ...]
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.layout not in ("PE", "SE"):
            raise ValueError(f"layout must be PE or SE, got {self.layout!r}")
        if self.layout == "PE" and len(self.files) != 2:
            raise ValueError(f"PE run {self.run_id} needs 2 files, got {len(self.files)}")
        if self.layout == "SE" and len(self.files) != 1:
            raise ValueError(f"SE run {self.run_id} needs 1 file, got {len(self.files)}")

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def is_wgs(self) -> bool:
        return self.strategy.upper() == "WGS"

    def with_local_dir(self, directory: str | Path) -> "Run":
        directory = Path(directory)
        return replace(self, files=tuple(replace(f, local_path=str(directory / f.basename)) for f in self.files))


@dataclass
class StudyManifest:
    study_id: str
    accession: str
    runs: list[Run]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for r in self.runs:
            if r.run_id in seen:
                raise ValueError(f"Duplicate run_id {r.run_id!r} in StudyManifest")
            seen.add(r.run_id)

    @property
    def total_archive_bytes(self) -> int:
        return sum(r.total_bytes for r in self.runs)

    @property
    def rna_runs(self) -> list[Run]:
        return [r for r in self.runs if not r.is_wgs]

    @property
    def wgs_runs(self) -> list[Run]:
        return [r for r in self.runs if r.is_wgs]

    @property
    def layout(self) -> str:
        layouts = {r.layout for r in self.rna_runs}
        return "PE" if layouts == {"PE"} else ("SE" if layouts == {"SE"} else "MIXED")

    def run(self, run_id: str) -> Run:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise KeyError(run_id)

    def with_local_dir(self, directory: str | Path) -> "StudyManifest":
        return StudyManifest(self.study_id, self.accession, [r.with_local_dir(directory) for r in self.runs])


def _ipv4_session():
    """requests session forced to IPv4 (compute nodes have no IPv6 route to EBI)."""
    import requests
    from urllib3.util import connection as _conn

    _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    s = requests.Session()
    s.headers["User-Agent"] = "panexpr/0.1"
    return s


def fetch_ena_filereport(accession: str, timeout: float = 60.0) -> str:
    """Fetch the ENA read_run filereport TSV for a project/run accession."""
    s = _ipv4_session()
    r = s.get(
        ENA_FILEREPORT,
        params={"accession": accession, "result": "read_run", "fields": ENA_FIELDS, "format": "tsv"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def parse_ena_filereport(
    tsv_text: str,
    study_id: str,
    accession: str,
    condition_of: "callable | dict[str, str] | None" = None,
    sample_of: "callable | None" = None,
) -> StudyManifest:
    """Parse an ENA filereport TSV into a StudyManifest.

    ``condition_of`` maps a run row (dict) -> condition label; a dict keyed by
    run_accession or sample alias also works. ``sample_of`` maps row -> sample name.
    """
    rows = list(csv.DictReader(io.StringIO(tsv_text), delimiter="\t"))
    runs: list[Run] = []
    for row in rows:
        urls = [u for u in (row.get("fastq_ftp") or "").split(";") if u]
        sizes = [int(x) for x in (row.get("fastq_bytes") or "").split(";") if x]
        md5s = [m for m in (row.get("fastq_md5") or "").split(";") if m]
        if not urls:
            continue
        if len(md5s) < len(urls):
            md5s += [""] * (len(urls) - len(md5s))
        # ENA sometimes lists an unsplit file first for PE runs; keep _1/_2 when present
        paired = [i for i, u in enumerate(urls) if u.endswith("_1.fastq.gz") or u.endswith("_2.fastq.gz")]
        if len(paired) == 2:
            idx = paired
        else:
            idx = [0]
        files = tuple(
            FastqFile(url=("https://" + urls[i]) if not urls[i].startswith("http") else urls[i], bytes=sizes[i], md5=md5s[i])
            for i in idx
        )
        layout = "PE" if len(files) == 2 else "SE"
        acc = row["run_accession"]
        sample = sample_of(row) if sample_of else (row.get("sample_alias") or row.get("sample_title") or acc)
        if callable(condition_of):
            cond = condition_of(row)
        elif isinstance(condition_of, dict):
            cond = condition_of.get(acc) or condition_of.get(sample) or "NA"
        else:
            cond = "NA"
        rc_val = int(row.get("read_count") or 0)
        bc_val = int(row.get("base_count") or 0)
        extra_meta: dict = {}
        if rc_val > 0 and bc_val > 0:
            mates = 2 if layout == "PE" else 1
            rlen = bc_val / (rc_val * mates)
            extra_meta["_ena_meta"] = {
                "read_count": rc_val,
                "base_count": bc_val,
                "fastq_bytes": sum(sizes),
                "layout": layout,
                "mates": mates,
                "read_len": round(rlen, 2),
                "kmers_per_frag": round(max(2.0, mates * max(1.0, rlen - 30.0)), 2),
            }
        runs.append(
            Run(
                run_id=acc,
                sample=sample,
                condition=cond,
                layout=layout,
                strategy=row.get("library_strategy") or "RNA-Seq",
                read_count=rc_val,
                files=files,
                extra=extra_meta,
            )
        )
    runs.sort(key=lambda r: r.run_id)
    return StudyManifest(study_id=study_id, accession=accession, runs=runs)


def load_ena_tsv(path: str | Path, study_id: str, accession: str, **kw) -> StudyManifest:
    return parse_ena_filereport(Path(path).read_text(), study_id, accession, **kw)


def manifest_to_rows(m: StudyManifest) -> list[dict]:
    out = []
    for r in m.runs:
        for i, f in enumerate(r.files, start=1):
            out.append({
                "run_id": r.run_id, "sample": r.sample, "condition": r.condition, "layout": r.layout,
                "strategy": r.strategy, "mate": i, "url": f.url, "bytes": f.bytes, "md5": f.md5,
                "local_path": f.local_path or "",
            })
    return out
