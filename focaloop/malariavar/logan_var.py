"""VariationLoop 0.4.1 organism-agnostic Logan unitig variant calling engine."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import pandas as pd

from ..logan import logan_unitigs_url

_COMPILE_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()


def available_ram_bytes() -> int:
    """Detect effective memory limit across cgroup v2/v1 (including ancestors), Galaxy/Slurm env, and host RAM."""
    candidates: list[int] = []

    for env_mb in ("GALAXY_MEMORY_MB", "SLURM_MEM_PER_NODE"):
        raw = os.environ.get(env_mb, "").strip()
        if raw.isdigit() and int(raw) > 0:
            candidates.append(int(raw) * (1024 * 1024))

    # Walk cgroup v2/v1 hierarchy up to /sys/fs/cgroup so parent-slice limits are respected
    try:
        cg_text = Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="ignore")
        for ln in cg_text.splitlines():
            parts = ln.split(":", 2)
            if len(parts) != 3:
                continue
            _, subsys, rel = parts
            rel_clean = rel.lstrip("/")
            if subsys == "":
                cur = Path("/sys/fs/cgroup") / rel_clean if rel_clean else Path("/sys/fs/cgroup")
                root = Path("/sys/fs/cgroup").resolve()
                while True:
                    mmax = cur / "memory.max"
                    if mmax.is_file():
                        val = mmax.read_text(encoding="utf-8").strip()
                        if val.isdigit():
                            v = int(val)
                            if 0 < v < (1 << 60):
                                candidates.append(v)
                    if cur.resolve() == root or cur.parent == cur:
                        break
                    cur = cur.parent
            elif "memory" in subsys.split(","):
                cur = Path("/sys/fs/cgroup/memory") / rel_clean if rel_clean else Path("/sys/fs/cgroup/memory")
                root = Path("/sys/fs/cgroup/memory").resolve()
                while True:
                    lim = cur / "memory.limit_in_bytes"
                    if lim.is_file():
                        val = lim.read_text(encoding="utf-8").strip()
                        if val.isdigit():
                            v = int(val)
                            if 0 < v < (1 << 60):
                                candidates.append(v)
                    if cur.resolve() == root or cur.parent == cur:
                        break
                    cur = cur.parent
    except OSError:
        pass

    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
        for key in ("MemAvailable:", "MemTotal:"):
            for ln in meminfo.splitlines():
                if ln.startswith(key):
                    kb = int(ln.split()[1])
                    if kb > 0:
                        candidates.append(kb * 1024)
                    break
            if candidates:
                break
    except (OSError, ValueError, IndexError):
        pass

    if not candidates:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            psize = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and psize > 0:
                candidates.append(int(pages) * int(psize))
        except (ValueError, OSError, AttributeError):
            pass

    return min(candidates) if candidates else (8 << 30)


def _is_portable_isa() -> bool:
    return os.environ.get("FOCALOOP_PORTABLE_ISA", "1").strip() != "0"


def _bin_cache_dir() -> Path:
    env_dir = os.environ.get("FOCALOOP_BIN_CACHE", "").strip()
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    uid = os.getuid() if hasattr(os, "getuid") else 0
    base = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"focaloop_bins_{uid}"
    p.mkdir(mode=0o700, exist_ok=True)
    st = os.lstat(p)
    if stat.S_ISLNK(st.st_mode):
        raise RuntimeError(f"Refusing symlinked binary cache directory: {p}")
    if hasattr(os, "getuid") and st.st_uid != uid:
        raise RuntimeError(f"Binary cache directory {p} is not owned by current user (uid={uid})")
    if (st.st_mode & 0o777) != 0o700:
        os.chmod(p, 0o700)
    return p


def _ensure_c_binary(name: str, src_path: Path) -> Path:
    portable = _is_portable_isa()
    arch = platform.machine().lower()
    isa_flags: list[str] = []
    if portable:
        if arch in ("x86_64", "amd64"):
            isa_flags = ["-msse4.2", "-mpopcnt"]
    elif arch in ("x86_64", "amd64", "aarch64", "arm64"):
        isa_flags = ["-march=native"]

    if portable:
        for pre_dir in (
            os.environ.get("FOCALOOP_NATIVE_CACHE", ""),
            str(Path(sys.prefix) / "share" / "focaloop" / "native_cache"),
            "/opt/focaloop/native_cache",
        ):
            if pre_dir:
                cand = Path(pre_dir) / name
                if cand.is_file() and os.access(cand, os.X_OK):
                    return cand
        pre_path = shutil.which(name)
        if pre_path:
            return Path(pre_path)

    h_obj = hashlib.sha256()
    h_obj.update(b"focaloop-logan-bin-v2\0")
    h_obj.update(" ".join(isa_flags).encode("utf-8") + b"\0")
    h_obj.update(src_path.read_bytes())
    digest = h_obj.hexdigest()[:16]
    out_bin = _bin_cache_dir() / f"{name}_{digest}"
    if out_bin.is_file() and os.access(out_bin, os.X_OK):
        return out_bin
    with _COMPILE_LOCK:
        if out_bin.is_file() and os.access(out_bin, os.X_OK):
            return out_bin
        cc_env = os.environ.get("CC")
        cc = (shutil.which(cc_env) if cc_env else None) or shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        if not cc:
            raise RuntimeError(f"C compiler required to build {name} from {src_path}")
        tmp_out = out_bin.with_name(f".{out_bin.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        flags = [cc, "-O3", *isa_flags, str(src_path), "-o", str(tmp_out), "-lm"]
        res = subprocess.run(flags, capture_output=True, text=True)
        if res.returncode != 0:
            tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to compile {name}: {res.stderr.strip()}")
        os.chmod(tmp_out, 0o755)
        os.replace(tmp_out, out_bin)
    return out_bin


def ensure_var_logan_bins() -> tuple[Path, Path]:
    nat_dir = Path(__file__).resolve().parent / "native"
    prep = _ensure_c_binary("prep_unitigs_filt_adapt", nat_dir / "prep_unitigs_filt_adapt.c")
    pile = _ensure_c_binary("logan_var_pileup", nat_dir / "logan_var_pileup.c")
    return prep, pile


def _select_var_aligner(requested: str = "auto", ref_bp: int = 0, avail_ram: int | None = None) -> tuple[str, str]:
    ram = available_ram_bytes() if avail_ram is None else int(avail_ram)
    if requested in ("bwa-mem2", "bwa", "minimap2"):
        order = [requested, "bwa", "bwa-mem2", "minimap2"]
    else:
        # bwa-mem2 index needs ~28 B * ref_bp RAM (e.g. ~88 GB on GRCh38); choose classic bwa when RAM-constrained (#44)
        if ref_bp > 0 and (28 * int(ref_bp)) > int(0.70 * ram) and shutil.which("bwa"):
            order = ["bwa", "minimap2", "bwa-mem2"]
        else:
            order = ["bwa-mem2", "bwa", "minimap2"]
    seen: set[str] = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        p = shutil.which(name)
        if p:
            return name, p
    raise RuntimeError("Logan VariationLoop requires bwa-mem2, bwa, or minimap2 on PATH")


def ensure_var_aligner_index(
    aligner_name: str,
    aligner_bin: str,
    ref_fa: Path,
    work_dir: Path,
    log: Callable[[str], None],
    allow_bwa_fallback: bool = True,
) -> Path:
    """Ensure aligner index exists, returning `indexed_ref_path` (falling back to classic `bwa` if `bwa-mem2` fails)."""
    if aligner_name == "bwa-mem2":
        marker = Path(str(ref_fa) + ".bwt.2bit.64")
        if marker.is_file() and marker.stat().st_size > 0:
            return ref_fa
        idx_fa = work_dir / "logan_idx" / ref_fa.name
        with _INDEX_LOCK:
            idx_fa.parent.mkdir(parents=True, exist_ok=True)
            if not idx_fa.exists():
                os.symlink(ref_fa.resolve(), idx_fa)
            if not Path(str(idx_fa) + ".bwt.2bit.64").is_file():
                if Path(str(idx_fa) + ".bwt").is_file() and allow_bwa_fallback and shutil.which("bwa"):
                    return idx_fa
                t0 = time.time()
                log(f"Building bwa-mem2 reference index for Logan unitigs ({ref_fa.stat().st_size / 1e6:.1f} MB)...")
                res = subprocess.run([aligner_bin, "index", str(idx_fa)], capture_output=True, text=True)
                if res.returncode != 0:
                    for f_part in idx_fa.parent.glob(f"{ref_fa.name}.*"):
                        f_part.unlink(missing_ok=True)
                    bwa_bin = shutil.which("bwa")
                    if allow_bwa_fallback and bwa_bin:
                        log(
                            f"bwa-mem2 index failed (rc={res.returncode}), falling back to classic bwa index..."
                        )
                        return ensure_var_aligner_index("bwa", bwa_bin, ref_fa, work_dir, log, allow_bwa_fallback=False)
                    raise RuntimeError(
                        f"bwa-mem2 index failed (rc={res.returncode}): {res.stderr.strip()[-400:]}"
                    )
                log(f"bwa-mem2 index ready in {time.time() - t0:.1f}s")
        return idx_fa
    if aligner_name == "bwa":
        marker = Path(str(ref_fa) + ".bwt")
        if marker.is_file() and marker.stat().st_size > 0:
            return ref_fa
        idx_fa = work_dir / "logan_idx" / ref_fa.name
        with _INDEX_LOCK:
            idx_fa.parent.mkdir(parents=True, exist_ok=True)
            if not idx_fa.exists():
                os.symlink(ref_fa.resolve(), idx_fa)
            if not Path(str(idx_fa) + ".bwt").is_file():
                t0 = time.time()
                log(f"Building bwa reference index for Logan unitigs ({ref_fa.stat().st_size / 1e6:.1f} MB)...")
                res = subprocess.run([aligner_bin, "index", str(idx_fa)], capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"bwa index failed (rc={res.returncode}): {res.stderr.strip()[-400:]}")
                log(f"bwa index ready in {time.time() - t0:.1f}s")
        return idx_fa
    return ref_fa


def build_k15_reference_bitset(
    ref_fa: Path,
    out_bit: Path,
    *,
    prep_bin: Path,
    targets_bed: Path | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> Path | None:
    """Build a 128 MiB canonical k=15 bitset from reference (or BED intervals) for fast host/off-target rejection."""
    if out_bit.is_file() and out_bit.stat().st_size == (1 << 27):
        return out_bit
    out_bit.parent.mkdir(parents=True, exist_ok=True)
    fa_for_bit = ref_fa
    if targets_bed is not None and targets_bed.is_file():
        want_ctgs: set[str] = set()
        bed_rows: list[tuple[str, int, int]] = []
        with open(targets_bed, "r", encoding="utf-8-sig") as bf:
            for ln in bf:
                if not ln.strip() or ln.startswith(("#", "track", "browser")):
                    continue
                parts = ln.split()
                if len(parts) >= 3:
                    cn, s0, e0 = parts[0], int(parts[1]), int(parts[2])
                    if e0 > s0:
                        want_ctgs.add(cn)
                        bed_rows.append((cn, s0, e0))
        if not bed_rows:
            return None
        seqs: dict[str, str] = {}
        cur = None
        keep_cur = False
        chunks: list[str] = []
        with open(ref_fa, "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith(">"):
                    if cur is not None and keep_cur:
                        seqs[cur] = "".join(chunks)
                    cur = ln[1:].split()[0]
                    keep_cur = cur in want_ctgs
                    chunks = []
                elif cur is not None and keep_cur:
                    chunks.append(ln.strip())
            if cur is not None and keep_cur:
                seqs[cur] = "".join(chunks)
        sub_fa = out_bit.with_suffix(".targets.fa")
        span = 0
        with open(sub_fa, "w", encoding="utf-8") as out_f:
            for idx, (cn, s0, e0) in enumerate(bed_rows):
                cseq = seqs.get(cn)
                if not cseq:
                    continue
                a = max(0, s0 - 200)
                b = min(len(cseq), e0 + 200)
                if b > a:
                    out_f.write(f">t{idx}\n{cseq[a:b]}\n")
                    span += b - a
        if span == 0 or span > 50_000_000:
            sub_fa.unlink(missing_ok=True)
            return None
        fa_for_bit = sub_fa
    else:
        if ref_fa.stat().st_size > 55_000_000:
            return None
    t0 = time.time()
    res = subprocess.run([str(prep_bin), "--build", str(fa_for_bit), str(out_bit)], capture_output=True, text=True)
    if fa_for_bit != ref_fa:
        fa_for_bit.unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to build k=15 reference bitset: {res.stderr.strip()}")
    log(f"Built k=15 host/off-target filter bitset in {time.time() - t0:.2f}s")
    return out_bit


def _wsaf_qual(ac: float, rc: float, err: float = 0.005) -> float:
    """Phred-scaled likelihood ratio against sequencing error at rate `err`."""
    tot = ac + rc
    if ac <= 0.0 or tot <= 0.0:
        return 0.0
    p_hat = min(max(ac / tot, err), 1.0 - 1e-12)
    if p_hat <= err:
        return 0.0
    ll_alt = ac * math.log(p_hat) + rc * math.log(max(1.0 - p_hat, 1e-12))
    ll_err = ac * math.log(err) + rc * math.log(1.0 - err)
    llr = max(0.0, ll_alt - ll_err)
    return min(999.0, 10.0 * llr / math.log(10.0))


def _genotype_and_gq(ad_non_alt: float, ad_alt: float, err: float = 0.005) -> tuple[str, int]:
    """Compute diploid genotype (0/1 or 1/1) and Phred GQ from weighted non-alt/alt depths."""
    l00 = ad_non_alt * math.log(1.0 - err) + ad_alt * math.log(err)
    l01 = (ad_non_alt + ad_alt) * math.log(0.5)
    l11 = ad_non_alt * math.log(err) + ad_alt * math.log(1.0 - err)
    scores = sorted([("0/0", l00), ("0/1", l01), ("1/1", l11)], key=lambda kv: kv[1], reverse=True)
    best_gt, best_ll = scores[0]
    second_ll = scores[1][1]
    gq = int(min(99, round(10.0 * max(0.0, best_ll - second_ll) / math.log(10.0))))
    if best_gt == "0/0":
        best_gt = "0/1"
    return best_gt, gq


def _multiallelic_diploid_gt(ad_ref: float, ad_alt1: float, ad_alt2: float, err: float = 0.005) -> tuple[str, int]:
    """Evaluate 0/1, 0/2, 1/1, 2/2, and 1/2 log-likelihoods for two co-located alt alleles (L33)."""
    tot = ad_ref + ad_alt1 + ad_alt2
    l00 = ad_ref * math.log(1.0 - err) + (ad_alt1 + ad_alt2) * math.log(err)
    l01 = (ad_ref + ad_alt1) * math.log(0.5) + ad_alt2 * math.log(err)
    l11 = (tot - ad_alt1) * math.log(err) + ad_alt1 * math.log(1.0 - err)
    l02 = (ad_ref + ad_alt2) * math.log(0.5) + ad_alt1 * math.log(err)
    l22 = (tot - ad_alt2) * math.log(err) + ad_alt2 * math.log(1.0 - err)
    l12 = (ad_alt1 + ad_alt2) * math.log(0.5) + ad_ref * math.log(err)
    scores = sorted(
        [("0/0", l00), ("0/1", l01), ("1/1", l11), ("0/2", l02), ("2/2", l22), ("1/2", l12)],
        key=lambda kv: kv[1],
        reverse=True,
    )
    best_gt, best_ll = scores[0]
    gq = int(min(99, round(10.0 * max(0.0, best_ll - scores[1][1]) / math.log(10.0))))
    return best_gt, gq


def _load_bed_filter(targets_bed: Path | None) -> dict[str, list[tuple[int, int]]]:
    if targets_bed is None or not targets_bed.is_file():
        return {}
    out: dict[str, list[tuple[int, int]]] = {}
    with open(targets_bed, "r", encoding="utf-8-sig") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.split()
            if len(p) >= 3:
                out.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    return out


def _pos_in_bed(chrom: str, pos1: int, ref_len: int, bed_map: dict[str, list[tuple[int, int]]]) -> bool:
    if not bed_map:
        return True
    ivs = bed_map.get(chrom)
    if not ivs:
        return False
    p0 = pos1 - 1
    p1 = p0 + max(1, ref_len)
    return any(p0 < e and p1 > s for s, e in ivs)


def _validate_local_unitigs_file(lp: Path) -> bytes:
    """Validate non-empty FASTA(.gz/.zst) unitigs input and reject HTML, FASTQ, bz2, xz, or BAM/CRAM (#36)."""
    if not lp.is_file():
        raise FileNotFoundError(f"Unitigs file does not exist: {lp.name}")
    if lp.stat().st_size == 0:
        raise ValueError(f"Unitigs file {lp.name!r} is empty (0 bytes)")
    with open(lp, "rb") as fh:
        head = fh.read(256)
    if head.startswith(b"BZh"):
        raise ValueError(f"Unsupported bzip2-compressed input {lp.name!r} (expected FASTA, .fa.gz, or .fa.zst)")
    if head.startswith(b"\xfd7zXZ\x00"):
        raise ValueError(f"Unsupported xz-compressed input {lp.name!r} (expected FASTA, .fa.gz, or .fa.zst)")
    if head.startswith((b"BAM\x01", b"CRAM")):
        raise ValueError(f"Alignment file {lp.name!r} supplied where Logan unitig FASTA was expected")
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(lp, "rb") as gzf:
                inner = gzf.read(256)
        except Exception as exc:
            raise ValueError(f"Corrupt or truncated gzip unitig archive {lp.name!r}: {exc}") from exc
        stripped = inner.lstrip()
    elif head[:4] == b"\x28\xb5\x2f\xfd":
        return head[:4]
    else:
        stripped = head.lstrip()

    low_prefix = stripped[:64].lower()
    if low_prefix.startswith((b"<!doctype", b"<html", b"<?xml", b"<error")):
        raise ValueError(f"Input {lp.name!r} is an HTML/XML error document rather than FASTA unitigs")
    if stripped.startswith(b"@"):
        raise ValueError(
            f"Input {lp.name!r} is FASTQ ('@' header) rather than Logan FASTA unitigs ('>' header with ka:f: abundance)"
        )
    if not stripped.startswith(b">"):
        raise ValueError(f"Input {lp.name!r} is not a valid FASTA unitigs file (missing '>' header)")
    return head[:4]


def _write_vcf_gz(
    vcf_gz: Path,
    sample_name: str,
    ctg_lens: dict[str, int],
    records: list[dict[str, Any]],
) -> None:
    vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    plain_vcf = vcf_gz.with_suffix("") if vcf_gz.suffix == ".gz" else vcf_gz
    with open(plain_vcf, "w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##source=FocaloopVariationLoopLogan-0.4.1\n")
        for cn, ln in ctg_lens.items():
            fh.write(f"##contig=<ID={cn},length={ln}>\n")
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Weighted k-mer abundance depth">\n')
        fh.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Weighted alternate allele fraction">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
        fh.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Weighted depth">\n')
        fh.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Weighted ref and alt allele depths">\n')
        fh.write('##FORMAT=<ID=WSAF,Number=A,Type=Float,Description="Within-sample alternate allele fraction">\n')
        fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n")
        for rec in records:
            ws_val = rec["wsaf"]
            if isinstance(ws_val, (list, tuple)):
                ws_str = ",".join(f"{float(x):.4f}" for x in ws_val)
            else:
                ws_str = f"{float(ws_val):.4f}"
            ad_alt_val = rec["ad_alt"]
            if isinstance(ad_alt_val, (list, tuple)):
                ad_alt_str = ",".join(str(int(x)) for x in ad_alt_val)
            else:
                ad_alt_str = str(ad_alt_val)
            info = f"DP={rec['dp']};AF={ws_str}"
            fmt = f"{rec['gt']}:{rec['gq']}:{rec['dp']}:{rec['ad_ref']},{ad_alt_str}:{ws_str}"
            fh.write(
                f"{rec['chrom']}\t{rec['pos']}\t.\t{rec['ref']}\t{rec['alt']}\t"
                f"{rec['qual']:.1f}\tPASS\t{info}\tGT:GQ:DP:AD:WSAF\t{fmt}\n"
            )
    if vcf_gz.suffix == ".gz":
        if shutil.which("bgzip"):
            subprocess.run(["bgzip", "-f", str(plain_vcf)], check=True)
            if shutil.which("tabix"):
                subprocess.run(["tabix", "-f", "-p", "vcf", str(vcf_gz)], check=True)
        else:
            with open(plain_vcf, "rb") as fi, gzip.open(vcf_gz, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            plain_vcf.unlink(missing_ok=True)


def run_varloop_logan_unitigs(
    *,
    acc: str,
    sample: str,
    ref_fa: Path,
    ctg_lens: dict[str, int],
    out_vcf_gz: Path,
    work_dir: Path,
    targets_bed: Path | None = None,
    k15_bit: Path | None = None,
    mapper: str = "auto",
    threads: int = 4,
    min_mapq: int = 20,
    snv_min_qual: float = 20.0,
    indel_min_qual: float = 15.0,
    min_ka_mode: str = "adapt",
    max_ka_adapt: float = 3.0,
    min_wsaf: float = 0.25,
    ploidy: str = "diploid",
    local_unitigs_path: Path | None = None,
    out_cov_tsv: Path | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Stream one sample's Logan unitigs -> depth-adaptive k=15 host filter -> align -> ka-weighted pileup -> VCF."""
    t0 = time.time()
    magic = b""
    if local_unitigs_path is not None:
        magic = _validate_local_unitigs_file(Path(local_unitigs_path))

    total_ref_bp = sum(int(v) for v in ctg_lens.values())
    prep_bin, pile_bin = ensure_var_logan_bins()
    aligner_name, aligner_bin = _select_var_aligner(mapper, ref_bp=total_ref_bp)
    idx_ref = ensure_var_aligner_index(
        aligner_name, aligner_bin, ref_fa, work_dir, log, allow_bwa_fallback=(mapper == "auto")
    )
    if (
        aligner_name == "bwa-mem2"
        and not Path(str(idx_ref) + ".bwt.2bit.64").is_file()
        and Path(str(idx_ref) + ".bwt").is_file()
        and shutil.which("bwa")
    ):
        aligner_name = "bwa"
        aligner_bin = shutil.which("bwa") or aligner_bin

    sample_work = work_dir / f"logan_var_{out_vcf_gz.stem}"
    sample_work.mkdir(parents=True, exist_ok=True)
    cand_tsv = sample_work / "candidates.tsv"
    curl_err = sample_work / "curl.err"
    prep_err = sample_work / "prep.err"
    aln_err = sample_work / "aln.err"
    pile_err = sample_work / "pile.err"

    bit_arg = str(k15_bit) if (k15_bit is not None and k15_bit.is_file()) else "-"
    min_hits = "2" if bit_arg != "-" else "0"
    relax_ka = "5.0" if bit_arg != "-" else "0"
    bed_arg = str(targets_bed) if (targets_bed is not None and targets_bed.is_file()) else "-"
    cov_arg = str(out_cov_tsv) if out_cov_tsv is not None else "-"

    def _build_aln_cmd(a_name: str, a_bin: str, a_idx: Path) -> list[str]:
        if a_name in ("bwa-mem2", "bwa"):
            return [a_bin, "mem", "-K", "1000000", "-k", "15", "-T", "22", "-t", str(max(1, threads)), str(a_idx), "-"]
        return [a_bin, "-K", "1M", "-x", "sr", "-k", "15", "-w", "5", "-a", "-t", str(max(1, threads)), str(a_idx), "-"]

    def _run_pipeline(a_name: str, a_bin: str, a_idx: Path) -> tuple[list[int], int]:
        aln_cmd = _build_aln_cmd(a_name, a_bin, a_idx)
        wbytes = 0
        procs: list[subprocess.Popen] = []

        def _pipe_and_count(src_stream, dst_stream) -> None:
            nonlocal wbytes
            try:
                while True:
                    chunk = src_stream.read(1 << 18)
                    if not chunk:
                        break
                    wbytes += len(chunk)
                    dst_stream.write(chunk)
            except (BrokenPipeError, OSError):
                pass
            finally:
                for st in (dst_stream, src_stream):
                    try:
                        st.close()
                    except Exception:
                        pass

        with (
            open(curl_err, "wb") as f_cerr,
            open(prep_err, "wb") as f_perr,
            open(aln_err, "wb") as f_aerr,
            open(pile_err, "wb") as f_lerr,
        ):
            p_curl: subprocess.Popen | None = None
            p_decomp: subprocess.Popen | None = None
            feeder: threading.Thread | None = None
            if local_unitigs_path is not None:
                lp = Path(local_unitigs_path)
                wbytes = lp.stat().st_size
                if magic == b"\x28\xb5\x2f\xfd":
                    p_decomp = subprocess.Popen(["zstd", "-dc", str(lp)], stdout=subprocess.PIPE, stderr=f_cerr)
                    prep_stdin = p_decomp.stdout
                    procs.append(p_decomp)
                elif magic[:2] == b"\x1f\x8b":
                    p_decomp = subprocess.Popen(["gzip", "-dc", str(lp)], stdout=subprocess.PIPE, stderr=f_cerr)
                    prep_stdin = p_decomp.stdout
                    procs.append(p_decomp)
                else:
                    prep_stdin = open(lp, "rb")
            else:
                url = logan_unitigs_url(acc)
                p_curl = subprocess.Popen(
                    [
                        "curl", "-4", "-sSL", "--show-error", "--fail",
                        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120", url,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=f_cerr,
                )
                p_decomp = subprocess.Popen(["zstd", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=f_cerr)
                procs.extend([p_curl, p_decomp])
                feeder = threading.Thread(target=_pipe_and_count, args=(p_curl.stdout, p_decomp.stdin), daemon=True)
                feeder.start()
                prep_stdin = p_decomp.stdout

            p_prep = subprocess.Popen(
                [str(prep_bin), str(min_ka_mode), "0", bit_arg, min_hits, relax_ka, str(max_ka_adapt)],
                stdin=prep_stdin,
                stdout=subprocess.PIPE,
                stderr=f_perr,
            )
            p_aln = subprocess.Popen(
                aln_cmd,
                stdin=p_prep.stdout,
                stdout=subprocess.PIPE,
                stderr=f_aerr,
            )
            p_pile = subprocess.Popen(
                [str(pile_bin), str(ref_fa), bed_arg, str(cand_tsv), str(int(min_mapq)), "0.12", "1", cov_arg],
                stdin=p_aln.stdout,
                stdout=subprocess.DEVNULL,
                stderr=f_lerr,
            )
            procs.extend([p_prep, p_aln, p_pile])

            if p_decomp is not None and p_decomp.stdout is not None:
                p_decomp.stdout.close()
            elif local_unitigs_path is not None and hasattr(prep_stdin, "close"):
                prep_stdin.close()
            if p_prep.stdout is not None:
                p_prep.stdout.close()
            if p_aln.stdout is not None:
                p_aln.stdout.close()

            if feeder is not None:
                feeder.join()
            rcs_rev = []
            for proc in reversed(procs):
                if proc is p_curl and proc.poll() is None:
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                rcs_rev.append(proc.wait())
            return list(reversed(rcs_rev)), wbytes

    def _is_retryable_curl(rc0: int) -> bool:
        if rc0 in (6, 7, 18, 28, 35, 52, 56, 92):
            return True
        if rc0 == 22 and curl_err.is_file():
            msg = curl_err.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\b(?:429|500|502|503|504)\b", msg):
                return True
        return False

    try:
        rcs, wire_bytes = _run_pipeline(aligner_name, aligner_bin, idx_ref)
        for retry_i in range(3):
            if local_unitigs_path is None and len(rcs) == 5 and _is_retryable_curl(rcs[0]):
                log(
                    f"Transient Logan download error for {sample} ({acc}, curl rc={rcs[0]}), "
                    f"retrying ({retry_i + 1}/3)..."
                )
                time.sleep(2.0 * (2 ** retry_i))
                rcs, wire_bytes = _run_pipeline(aligner_name, aligner_bin, idx_ref)
            else:
                break

        # Fallback to classic bwa ONLY when bwa-mem2 itself crashed with a non-SIGPIPE signal (L35)
        aln_rc_idx = len(rcs) - 2
        pile_rc = rcs[-1] if rcs else 0
        if (
            any(rc != 0 for rc in rcs)
            and pile_rc == 0
            and aln_rc_idx >= 0
            and rcs[aln_rc_idx] < 0
            and rcs[aln_rc_idx] != -13
            and aligner_name == "bwa-mem2"
            and mapper == "auto"
            and shutil.which("bwa")
        ):
            bwa_bin = shutil.which("bwa")
            assert bwa_bin is not None
            log(f"bwa-mem2 exited with signal {rcs[aln_rc_idx]} on {sample} ({acc}), retrying with classic bwa...")
            aligner_name = "bwa"
            aligner_bin = bwa_bin
            idx_ref = ensure_var_aligner_index(
                "bwa", bwa_bin, ref_fa, work_dir, log, allow_bwa_fallback=False
            )
            rcs, wire_bytes = _run_pipeline(aligner_name, aligner_bin, idx_ref)

        if any(rc != 0 for rc in rcs):
            pile_msg = pile_err.read_text(encoding="utf-8", errors="replace").strip()
            curl_msg = curl_err.read_text(encoding="utf-8", errors="replace").strip()
            aln_tail = " | ".join(aln_err.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-4:])
            raise RuntimeError(
                f"Logan VariationLoop stream failed for {sample} ({acc}): exit_codes={rcs} {pile_msg}"
                + (f" decomp/curl={curl_msg[-200:]}" if curl_msg else "")
                + (f" aln_err={aln_tail}" if aln_tail else "")
            )

        pt = prep_err.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        lt = pile_err.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not pt or not lt:
            raise RuntimeError(f"Logan VariationLoop stream produced incomplete stage metrics for {sample} ({acc})")
        prep_m: dict[str, Any] = json.loads(pt[-1])
        pile_m: dict[str, Any] = json.loads(lt[-1])

        n_unitigs_seen = int(prep_m.get("n_unitigs", 0))
        kept_unitigs = int(prep_m.get("kept_unitigs", 0))
        sam_records = int(pile_m.get("sam_records", 0))
        counted_unitigs = int(pile_m.get("counted_unitigs", 0))
        if n_unitigs_seen <= 0:
            raise ValueError(
                f"Sample {sample!r} ({acc}): zero FASTA unitigs decoded from input stream"
            )
        if kept_unitigs <= 0 or sam_records <= 0 or counted_unitigs <= 0:
            raise RuntimeError(
                f"Sample {sample!r} ({acc}): zero usable unitigs aligned to reference "
                f"(n_unitigs={n_unitigs_seen}, kept_unitigs={kept_unitigs}, "
                f"sam_records={sam_records}, counted_unitigs={counted_unitigs})"
            )

        bed_map = _load_bed_filter(targets_bed)
        cand_df = (
            pd.read_csv(
                cand_tsv,
                sep="\t",
                dtype={"chrom": str, "ref": str, "alt": str, "kind": str},
                keep_default_na=False,
                na_filter=False,
            )
            if cand_tsv.is_file()
            else pd.DataFrame()
        )
        records: list[dict[str, Any]] = []
        n_snv = 0
        n_indel = 0
        ctg_order = {cn: idx for idx, cn in enumerate(ctg_lens.keys())}

        eff_ploidy = "haploid" if str(ploidy or "diploid").strip().lower() == "haploid" else "diploid"
        eff_min_wsaf = max(0.05, min(0.99, float(min_wsaf)))
        eff_indel_min_wsaf = max(eff_min_wsaf, 0.15)

        passing: list[dict[str, Any]] = []
        if not cand_df.empty:
            for row in cand_df.itertuples(index=False):
                chrom = str(row.chrom)
                pos = int(row.pos)
                ref_a = str(row.ref).upper()
                alt_a = str(row.alt).upper()
                if not _pos_in_bed(chrom, pos, len(ref_a), bed_map):
                    continue
                kind = str(row.kind).upper()
                dp_f = float(row.dp)
                ad_ref_f = float(row.ad_ref)
                ad_alt_f = float(row.ad_alt)
                wsaf = float(row.wsaf)
                non_alt_f = max(dp_f - ad_alt_f, 0.0)

                if kind == "SNV":
                    qual = _wsaf_qual(ad_alt_f, non_alt_f, err=0.005)
                    if qual < float(snv_min_qual) or ad_alt_f < 2.0 or wsaf < eff_min_wsaf:
                        continue
                else:
                    qual = _wsaf_qual(ad_alt_f, non_alt_f, err=0.015)
                    if qual < float(indel_min_qual) or ad_alt_f < 2.0 or wsaf < eff_indel_min_wsaf:
                        continue

                passing.append(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref_a,
                        "alt": alt_a,
                        "kind": kind,
                        "dp_f": dp_f,
                        "ad_ref_f": ad_ref_f,
                        "ad_alt_f": ad_alt_f,
                        "non_alt_f": non_alt_f,
                        "wsaf": wsaf,
                        "qual": qual,
                    }
                )

        if eff_ploidy == "haploid":
            for item in passing:
                if item["kind"] == "SNV":
                    n_snv += 1
                else:
                    n_indel += 1
                records.append(
                    {
                        "chrom": item["chrom"],
                        "pos": item["pos"],
                        "ref": item["ref"],
                        "alt": item["alt"],
                        "kind": item["kind"],
                        "dp": max(1, int(round(item["dp_f"]))),
                        "ad_ref": max(0, int(round(item["ad_ref_f"]))),
                        "ad_alt": max(1, int(round(item["ad_alt_f"]))),
                        "wsaf": item["wsaf"],
                        "qual": item["qual"],
                        "gt": "1",
                        "gq": int(min(99, max(0, round(item["qual"])))),
                    }
                )
        else:
            # Group co-located alleles at (chrom, pos, ref, kind) to emit 1/2 multiallelic diploid calls cleanly (L33)
            grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
            for item in passing:
                grouped.setdefault((item["chrom"], item["pos"], item["ref"], item["kind"]), []).append(item)
            for (chrom, pos, ref_a, kind), grp in grouped.items():
                grp.sort(key=lambda x: -x["ad_alt_f"])
                err_rate = 0.005 if kind == "SNV" else 0.015
                if len(grp) >= 2:
                    g0, g1 = grp[0], grp[1]
                    dp_comb = max(g0["dp_f"], g1["dp_f"], g0["ad_alt_f"] + g1["ad_alt_f"])
                    ref_comb = max(0.0, dp_comb - g0["ad_alt_f"] - g1["ad_alt_f"])
                    m_gt, m_gq = _multiallelic_diploid_gt(ref_comb, g0["ad_alt_f"], g1["ad_alt_f"], err=err_rate)
                    if m_gt == "1/2":
                        if kind == "SNV":
                            n_snv += 1
                        else:
                            n_indel += 1
                        records.append(
                            {
                                "chrom": chrom,
                                "pos": pos,
                                "ref": ref_a,
                                "alt": f"{g0['alt']},{g1['alt']}",
                                "kind": kind,
                                "dp": max(1, int(round(dp_comb))),
                                "ad_ref": max(0, int(round(ref_comb))),
                                "ad_alt": (max(1, int(round(g0["ad_alt_f"]))), max(1, int(round(g1["ad_alt_f"])))),
                                "wsaf": (g0["wsaf"], g1["wsaf"]),
                                "qual": max(g0["qual"], g1["qual"]),
                                "gt": "1/2",
                                "gq": m_gq,
                            }
                        )
                        continue
                g0 = grp[0]
                gt, gq = _genotype_and_gq(g0["non_alt_f"], g0["ad_alt_f"], err=err_rate)
                if kind == "SNV":
                    n_snv += 1
                else:
                    n_indel += 1
                records.append(
                    {
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref_a,
                        "alt": g0["alt"],
                        "kind": kind,
                        "dp": max(1, int(round(g0["dp_f"]))),
                        "ad_ref": max(0, int(round(g0["ad_ref_f"]))),
                        "ad_alt": max(1, int(round(g0["ad_alt_f"]))),
                        "wsaf": g0["wsaf"],
                        "qual": g0["qual"],
                        "gt": gt,
                        "gq": gq,
                    }
                )

        records.sort(key=lambda r: (ctg_order.get(r["chrom"], 10**9), r["pos"], r["ref"], r["alt"]))
        _write_vcf_gz(out_vcf_gz, sample, ctg_lens, records)
    finally:
        shutil.rmtree(sample_work, ignore_errors=True)

    wall_s = round(time.time() - t0, 2)
    return {
        "sample": sample,
        "accession": acc,
        "source": "logan_unitigs",
        "aligner": aligner_name,
        "n_variants": len(records),
        "n_snv": n_snv,
        "n_indel": n_indel,
        "wire_bytes": int(wire_bytes),
        "n_unitigs": n_unitigs_seen,
        "kept_unitigs": kept_unitigs,
        "drop_ka": int(prep_m.get("drop_ka", 0)),
        "drop_host": int(prep_m.get("drop_host", 0)),
        "eff_min_ka": float(prep_m.get("eff_min_ka", 1.5)),
        "ploidy": eff_ploidy,
        "min_wsaf": float(eff_min_wsaf),
        "med_len80_ka": float(prep_m.get("med_len80_ka", -1.0)),
        "sam_records": sam_records,
        "reads_counted": counted_unitigs,
        "median_depth": float(pile_m.get("median_depth", 0.0)),
        "wall_s": wall_s,
        "total_wall_s": wall_s,
    }
