"""3-tier resolver, byte-accounted block reader, holdout partitioning, residual spans."""

from __future__ import annotations

import gzip
import os
import stat
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from focaloop.state import FocaloopStateStore
from focaloop.streamer import (
    FastqBlockReader,
    HttpRangeSource,
    LocalFileSource,
    coalesce_residual_spans,
    ena_fastq_urls,
    is_holdout_block,
    XsraSource,
    open_source,
    resolve_tiers,
    s3_odp_url,
    xsra_argv,
)

from .conftest import BLUEPRINTS

FQ = BLUEPRINTS / "panexpr" / "test-data" / "ctrl1.fastq.gz"


def test_resolver_orders_three_tiers() -> None:
    plans = resolve_tiers("ERR1234567", layout="PE", cram_url="https://example.org/ERR1234567.cram")
    assert [p.tier for p in plans] == ["cram_range", "s3_odp_spot", "ena_https_range"]
    assert plans[1].urls == ("https://sra-pub-run-odp.s3.amazonaws.com/sra/ERR1234567/ERR1234567",)
    assert ena_fastq_urls("ERR1234567") == [
        "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR123/007/ERR1234567/ERR1234567_1.fastq.gz",
        "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR123/007/ERR1234567/ERR1234567_2.fastq.gz",
    ]
    assert ena_fastq_urls("SRR123456", "SE") == ["https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123456/SRR123456.fastq.gz"]
    assert "/045/" in ena_fastq_urls("SRR12345645")[0]
    assert [p.tier for p in resolve_tiers("DRR000001")] == ["s3_odp_spot", "ena_https_range"]
    with pytest.raises(ValueError):
        resolve_tiers("not_an_accession")


def test_local_gzip_blocks_account_every_byte_and_record() -> None:
    rd = FastqBlockReader(LocalFileSource(FQ), "ctrl1", block_bytes=4096, holdout_fraction=0.25)
    blocks = list(rd)
    assert rd.consumed_bytes == FQ.stat().st_size == blocks[-1].byte_end
    assert [b.byte_start for b in blocks[1:]] == [b.byte_end for b in blocks[:-1]]
    with gzip.open(FQ, "rt") as fh:
        n_records = sum(1 for _ in fh) // 4
    assert sum(len(b.reads) for b in blocks) == n_records == rd.reads_emitted
    assert {b.partition for b in blocks} == {"discovery", "holdout"}
    expected = ["holdout" if is_holdout_block("ctrl1", i, 0.25) else "discovery" for i in range(len(blocks))]
    assert [b.partition for b in blocks] == expected


def test_plain_fastq_and_holdout_determinism(tmp_path: Path) -> None:
    plain = tmp_path / "x.fastq"
    with gzip.open(FQ, "rb") as fh:
        plain.write_bytes(fh.read())
    rd = FastqBlockReader(open_source(plain)[0], "x", block_bytes=1000, holdout_fraction=0.0)
    blocks = list(rd)
    assert all(b.partition == "discovery" for b in blocks)
    assert sum(len(b.reads) for b in blocks) == plain.read_text().count("\n+\n")
    split = [is_holdout_block("SRR1", i, 0.10) for i in range(5000)]
    assert split == [is_holdout_block("SRR1", i, 0.10) for i in range(5000)]
    assert 0.08 < sum(split) / 5000 < 0.12


class _RangeHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a) -> None:  # silence
        pass

    def do_GET(self) -> None:  # minimal RFC 7233 single-range support
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, b = rng[6:].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            body = data[a:b + 1]
        else:
            self.send_response(200)
            body = data
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_http_206_range_source_matches_local_reader() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(FQ.parent)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/{FQ.name}"
        src = open_source(url)[0]
        assert isinstance(src, HttpRangeSource) and src.total_bytes == FQ.stat().st_size
        remote = list(FastqBlockReader(src, "ctrl1", block_bytes=8192))
        local = list(FastqBlockReader(LocalFileSource(FQ), "ctrl1", block_bytes=8192))
        assert [b.reads for b in remote] == [b.reads for b in local]
        assert src.bytes_read == FQ.stat().st_size and src.requests_made == len(remote)
    finally:
        srv.shutdown()


def test_residual_spans_coalesce_and_persist(tmp_path: Path) -> None:
    store = FocaloopStateStore(tmp_path)
    seq = "ACGT" * 20
    spans = coalesce_residual_spans("r1", "local_file", [(0, 100, seq), (150, 300, seq), (400_000, 400_100, seq)],
                                    merge_gap_bytes=65536, store=store)
    assert [(s.byte_start, s.byte_end, s.n_reads) for s in spans] == [(0, 300, 2), (400_000, 400_100, 1)]
    assert store.read("residual_spans").num_rows == 2


def _fake_xsra(bindir: Path, body: str) -> Path:
    bindir.mkdir(parents=True, exist_ok=True)
    exe = bindir / "xsra"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_tier2_odp_url_and_full_quality_argv() -> None:
    assert s3_odp_url("SRR27592687") == "https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR27592687/SRR27592687"
    argv = xsra_argv("SRR27592687")
    assert argv == ["xsra", "dump", "-Q", "-P", "https", "SRR27592687"]
    assert "-Q" in argv  # full Illumina qualities, never SRA-lite '?'
    with pytest.raises(ValueError):
        s3_odp_url("not_a_run")


def test_open_source_prefers_xsra_on_path(tmp_path: Path, monkeypatch) -> None:
    fq = tmp_path / "plain.fastq"
    fq.write_bytes(gzip.decompress(FQ.read_bytes()))
    log = tmp_path / "argv.txt"
    _fake_xsra(tmp_path / "bin", f'echo "$@" > {log}\ncat {fq}\n')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    (src,) = open_source("SRR27592687")
    assert isinstance(src, XsraSource) and src.tier == "s3_odp_spot"
    rd = FastqBlockReader(src, "SRR27592687", block_bytes=4096, holdout_fraction=0.0)
    n = sum(len(b.reads) for b in rd)
    assert log.read_text().split() == ["dump", "-Q", "-P", "https", "SRR27592687"]
    assert rd.exhausted and src.total_bytes == fq.stat().st_size == rd.consumed_bytes
    ref = FastqBlockReader(LocalFileSource(fq), "x", block_bytes=4096, holdout_fraction=0.0)
    assert n == sum(len(b.reads) for b in ref) > 0


@pytest.mark.parametrize("body", [None, "echo 'error: lite-only object' >&2\nexit 3\n"])
def test_open_source_falls_back_to_ena(tmp_path: Path, monkeypatch, body) -> None:
    if body is None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no xsra anywhere
    else:
        _fake_xsra(tmp_path / "bin", body)
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(HttpRangeSource, "_probe_size", lambda self: 1234)  # no network in tests
    srcs = open_source("SRR123456")
    assert [type(s) for s in srcs] == [HttpRangeSource]
    assert srcs[0].tier == "ena_https_range" and srcs[0].url == ena_fastq_urls("SRR123456", "SE")[0]


def test_xsra_truncated_stream_raises(tmp_path: Path, monkeypatch) -> None:
    _fake_xsra(tmp_path / "bin", "printf '@r1\\nACGT\\n+\\nIIII\\n'\nexit 2\n")
    src = XsraSource("SRR123456", str(tmp_path / "bin" / "xsra"))
    rd = FastqBlockReader(src, "SRR123456", block_bytes=4096, holdout_fraction=0.0)
    with pytest.raises(RuntimeError, match="rc=2"):
        rd.next_block()
