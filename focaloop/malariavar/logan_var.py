"""VariationLoop 0.4.0 organism-agnostic Logan unitig variant calling engine."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..logan import logan_unitigs_url

_COMPILE_LOCK = threading.Lock()


def _bin_cache_dir() -> Path:
    env_dir = os.environ.get("FOCALOOP_BIN_CACHE", "").strip()
    p = Path(env_dir) if env_dir else (Path(tempfile.gettempdir()) / f"focaloop_bins_{os.getuid()}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_c_binary(name: str, src_path: Path) -> Path:
    pre = shutil.which(name)
    if pre:
        return Path(pre)
    pkg_bin = src_path.parent / name
    if pkg_bin.is_file() and os.access(pkg_bin, os.X_OK):
        return pkg_bin
    h = hashlib.sha256(src_path.read_bytes()).hexdigest()[:12]
    out_bin = _bin_cache_dir() / f"{name}_{h}"
    if out_bin.is_file() and os.access(out_bin, os.X_OK):
        return out_bin
    with _COMPILE_LOCK:
        if out_bin.is_file() and os.access(out_bin, os.X_OK):
            return out_bin
        cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        if not cc:
            raise RuntimeError(f"C compiler required to build {name} from {src_path}")
        tmp_out = out_bin.with_suffix(".tmp")
        flags = [cc, "-O3", str(src_path), "-o", str(tmp_out), "-lm"]
        if os.environ.get("FOCALOOP_PORTABLE_ISA") == "1" and os.uname().machine in ("x86_64", "AMD64"):
            flags[2:2] = ["-msse4.2", "-mpopcnt"]
        res = subprocess.run(flags, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to compile {name}: {res.stderr.strip()}")
        os.chmod(tmp_out, 0o755)
        tmp_out.replace(out_bin)
    return out_bin


def ensure_var_logan_bins() -> tuple[Path, Path]:
    nat_dir = Path(__file__).resolve().parent / "native"
    prep = _ensure_c_binary("prep_unitigs_filt_adapt", nat_dir / "prep_unitigs_filt_adapt.c")
    pile = _ensure_c_binary("logan_var_pileup", nat_dir / "logan_var_pileup.c")
    return prep, pile


def _select_var_aligner(requested: str = "auto") -> tuple[str, str]:
    order = (
        [requested, "bwa-mem2", "bwa", "minimap2"]
        if requested in ("bwa-mem2", "bwa", "minimap2")
        else ["bwa-mem2", "bwa", "minimap2"]
    )
    for name in order:
        p = shutil.which(name)
        if p:
            return name, p
    raise RuntimeError("Logan VariationLoop requires bwa-mem2, bwa, or minimap2 on PATH")


def ensure_var_aligner_index(aligner_name: str, aligner_bin: str, ref_fa: Path, work_dir: Path, log: Callable[[str], None]) -> Path:
    if aligner_name == "bwa-mem2":
        marker = Path(str(ref_fa) + ".bwt.2bit.64")
        if marker.is_file() and marker.stat().st_size > 0:
            return ref_fa
        idx_fa = work_dir / "logan_idx" / ref_fa.name
        idx_fa.parent.mkdir(parents=True, exist_ok=True)
        if not idx_fa.exists():
            os.symlink(ref_fa.resolve(), idx_fa)
        if not Path(str(idx_fa) + ".bwt.2bit.64").is_file():
            t0 = time.time()
            log(f"Building bwa-mem2 reference index for Logan unitigs ({ref_fa.stat().st_size / 1e6:.1f} MB)...")
            subprocess.run([aligner_bin, "index", str(idx_fa)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"bwa-mem2 index ready in {time.time() - t0:.1f}s")
        return idx_fa
    if aligner_name == "bwa":
        marker = Path(str(ref_fa) + ".bwt")
        if marker.is_file() and marker.stat().st_size > 0:
            return ref_fa
        idx_fa = work_dir / "logan_idx" / ref_fa.name
        idx_fa.parent.mkdir(parents=True, exist_ok=True)
        if not idx_fa.exists():
            os.symlink(ref_fa.resolve(), idx_fa)
        if not Path(str(idx_fa) + ".bwt").is_file():
            subprocess.run([aligner_bin, "index", str(idx_fa)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        # Extract target regions (+-200 bp) into a small FASTA so large genomes can still use k=15 host filtering
        seqs: dict[str, str] = {}
        cur = None
        chunks: list[str] = []
        with open(ref_fa, "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith(">"):
                    if cur is not None:
                        seqs[cur] = "".join(chunks)
                    cur = ln[1:].split()[0]
                    chunks = []
                elif cur is not None:
                    chunks.append(ln.strip())
            if cur is not None:
                seqs[cur] = "".join(chunks)
        sub_fa = out_bit.with_suffix(".targets.fa")
        span = 0
        with open(targets_bed, "r", encoding="utf-8") as bf, open(sub_fa, "w", encoding="utf-8") as out_f:
            for idx, ln in enumerate(bf):
                if not ln.strip() or ln.startswith(("#", "track", "browser")):
                    continue
                parts = ln.split()
                if len(parts) < 3:
                    continue
                cn, s0, e0 = parts[0], int(parts[1]), int(parts[2])
                cseq = seqs.get(cn)
                if not cseq:
                    continue
                a = max(0, s0 - 200)
                b = min(len(cseq), e0 + 200)
                if b > a:
                    out_f.write(f">t{idx}\n{cseq[a:b]}\n")
                    span += b - a
        if span == 0 or span > 50_000_000:
            return None
        fa_for_bit = sub_fa
    else:
        if ref_fa.stat().st_size > 55_000_000:
            return None
    t0 = time.time()
    res = subprocess.run([str(prep_bin), "--build", str(fa_for_bit), str(out_bit)], capture_output=True, text=True)
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


def _genotype_and_gq(ad_ref: float, ad_alt: float, err: float = 0.005) -> tuple[str, int]:
    """Compute diploid genotype (0/1 or 1/1) and Phred GQ from weighted ref/alt depths."""
    l00 = ad_ref * math.log(1.0 - err) + ad_alt * math.log(err)
    l01 = (ad_ref + ad_alt) * math.log(0.5)
    l11 = ad_ref * math.log(err) + ad_alt * math.log(1.0 - err)
    scores = sorted([("0/0", l00), ("0/1", l01), ("1/1", l11)], key=lambda kv: kv[1], reverse=True)
    best_gt, best_ll = scores[0]
    second_ll = scores[1][1]
    gq = int(min(99, round(10.0 * max(0.0, best_ll - second_ll) / math.log(10.0))))
    if best_gt == "0/0":
        best_gt = "0/1"
    return best_gt, gq


def _load_bed_filter(targets_bed: Path | None) -> dict[str, list[tuple[int, int]]]:
    if targets_bed is None or not targets_bed.is_file():
        return {}
    out: dict[str, list[tuple[int, int]]] = {}
    with open(targets_bed, "r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.split()
            if len(p) >= 3:
                out.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    return out


def _pos_in_bed(chrom: str, pos1: int, bed_map: dict[str, list[tuple[int, int]]]) -> bool:
    if not bed_map:
        return True
    ivs = bed_map.get(chrom)
    if not ivs:
        return False
    p0 = pos1 - 1
    return any(s <= p0 < e for s, e in ivs)


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
        fh.write("##source=FocaloopVariationLoopLogan-0.4.0\n")
        for cn, ln in ctg_lens.items():
            fh.write(f"##contig=<ID={cn},length={ln}>\n")
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Weighted k-mer abundance depth">\n')
        fh.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Weighted alternate allele fraction">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
        fh.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Weighted depth">\n')
        fh.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Weighted ref and alt allele depths">\n')
        fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n")
        for rec in records:
            info = f"DP={rec['dp']};AF={rec['wsaf']:.4f}"
            fmt = f"{rec['gt']}:{rec['gq']}:{rec['dp']}:{rec['ad_ref']},{rec['ad_alt']}"
            fh.write(
                f"{rec['chrom']}\t{rec['pos']}\t.\t{rec['ref']}\t{rec['alt']}\t"
                f"{rec['qual']:.1f}\tPASS\t{info}\tGT:GQ:DP:AD\t{fmt}\n"
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
    local_unitigs_path: Path | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Stream one sample's Logan unitigs -> depth-adaptive k=15 host filter -> align -> ka-weighted pileup -> VCF."""
    t0 = time.time()
    prep_bin, pile_bin = ensure_var_logan_bins()
    aligner_name, aligner_bin = _select_var_aligner(mapper)
    idx_ref = ensure_var_aligner_index(aligner_name, aligner_bin, ref_fa, work_dir, log)

    sample_work = work_dir / f"logan_var_{sample}"
    sample_work.mkdir(parents=True, exist_ok=True)
    cand_tsv = sample_work / "candidates.tsv"
    prep_err = sample_work / "prep.err"
    pile_err = sample_work / "pile.err"

    bit_arg = str(k15_bit) if (k15_bit is not None and k15_bit.is_file()) else "-"
    min_hits = "2" if bit_arg != "-" else "0"
    relax_ka = "5.0" if bit_arg != "-" else "0"
    bed_arg = str(targets_bed) if (targets_bed is not None and targets_bed.is_file()) else "-"

    if aligner_name in ("bwa-mem2", "bwa"):
        aln_cmd = [aligner_bin, "mem", "-k", "15", "-T", "22", "-t", str(max(1, threads)), str(idx_ref), "-"]
    else:
        aln_cmd = [aligner_bin, "-x", "sr", "-k", "15", "-w", "5", "-a", "-t", str(max(1, threads)), str(idx_ref), "-"]

    wire_bytes = 0
    procs: list[subprocess.Popen] = []

    def _pipe_and_count(src_stream, dst_stream) -> None:
        nonlocal wire_bytes
        try:
            while True:
                chunk = src_stream.read(1 << 18)
                if not chunk:
                    break
                wire_bytes += len(chunk)
                dst_stream.write(chunk)
        finally:
            try:
                dst_stream.close()
            except Exception:
                pass

    with open(prep_err, "wb") as f_perr, open(pile_err, "wb") as f_lerr:
        p_decomp: subprocess.Popen | None = None
        feeder: threading.Thread | None = None
        if local_unitigs_path is not None:
            lp = Path(local_unitigs_path)
            wire_bytes = lp.stat().st_size
            with open(lp, "rb") as fh:
                magic = fh.read(4)
            if magic == b"\x28\xb5\x2f\xfd":
                p_decomp = subprocess.Popen(["zstd", "-dc", str(lp)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                prep_stdin = p_decomp.stdout
                procs.append(p_decomp)
            elif magic[:2] == b"\x1f\x8b":
                p_decomp = subprocess.Popen(["gzip", "-dc", str(lp)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                prep_stdin = p_decomp.stdout
                procs.append(p_decomp)
            else:
                prep_stdin = open(lp, "rb")
        else:
            url = logan_unitigs_url(acc)
            p_curl = subprocess.Popen(
                ["curl", "-4", "-sSL", "--fail", "--retry", "5", "--retry-delay", "2", "--max-time", "600", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            p_decomp = subprocess.Popen(["zstd", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
            stderr=subprocess.DEVNULL,
        )
        p_pile = subprocess.Popen(
            [str(pile_bin), str(ref_fa), bed_arg, str(cand_tsv), str(int(min_mapq)), "0.12", "1"],
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
        rcs = [p.wait() for p in procs]
        if any(rc != 0 for rc in rcs):
            pile_msg = pile_err.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(f"Logan VariationLoop stream failed for {sample} ({acc}): exit_codes={rcs} {pile_msg}")

    prep_m: dict[str, Any] = {}
    pile_m: dict[str, Any] = {}
    try:
        pt = prep_err.read_text(encoding="utf-8").strip().splitlines()
        if pt:
            prep_m = json.loads(pt[-1])
    except Exception:
        pass
    try:
        lt = pile_err.read_text(encoding="utf-8").strip().splitlines()
        if lt:
            pile_m = json.loads(lt[-1])
    except Exception:
        pass

    bed_map = _load_bed_filter(targets_bed)
    cand_df = pd.read_csv(cand_tsv, sep="\t") if cand_tsv.is_file() else pd.DataFrame()
    records: list[dict[str, Any]] = []
    n_snv = 0
    n_indel = 0
    ctg_order = {cn: idx for idx, cn in enumerate(ctg_lens.keys())}

    if not cand_df.empty:
        # Collapse any duplicate (chrom, pos, ref, alt) rows and score
        for row in cand_df.itertuples(index=False):
            chrom = str(row.chrom)
            pos = int(row.pos)
            if not _pos_in_bed(chrom, pos, bed_map):
                continue
            ref_a = str(row.ref).upper()
            alt_a = str(row.alt).upper()
            kind = str(row.kind).upper()
            dp_f = float(row.dp)
            ad_ref_f = float(row.ad_ref)
            ad_alt_f = float(row.ad_alt)
            wsaf = float(row.wsaf)

            if kind == "SNV":
                qual = _wsaf_qual(ad_alt_f, max(dp_f - ad_alt_f, 0.0), err=0.005)
                if qual < float(snv_min_qual) or ad_alt_f < 2.0 or wsaf < 0.10:
                    continue
                n_snv += 1
            else:
                qual = _wsaf_qual(ad_alt_f, max(dp_f - ad_alt_f, 0.0), err=0.015)
                if qual < float(indel_min_qual) or ad_alt_f < 2.0 or wsaf < 0.15:
                    continue
                n_indel += 1

            gt, gq = _genotype_and_gq(ad_ref_f, ad_alt_f, err=0.005 if kind == "SNV" else 0.015)
            records.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref_a,
                    "alt": alt_a,
                    "kind": kind,
                    "dp": max(1, int(round(dp_f))),
                    "ad_ref": max(0, int(round(ad_ref_f))),
                    "ad_alt": max(1, int(round(ad_alt_f))),
                    "wsaf": wsaf,
                    "qual": qual,
                    "gt": gt,
                    "gq": gq,
                }
            )

    records.sort(key=lambda r: (ctg_order.get(r["chrom"], 10**9), r["pos"], r["ref"], r["alt"]))
    _write_vcf_gz(out_vcf_gz, sample, ctg_lens, records)
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
        "n_unitigs": int(prep_m.get("n_unitigs", 0)),
        "kept_unitigs": int(prep_m.get("kept_unitigs", 0)),
        "drop_ka": int(prep_m.get("drop_ka", 0)),
        "drop_host": int(prep_m.get("drop_host", 0)),
        "eff_min_ka": float(prep_m.get("eff_min_ka", 1.5)),
        "med_len80_ka": float(prep_m.get("med_len80_ka", -1.0)),
        "sam_records": int(pile_m.get("sam_records", 0)),
        "reads_counted": int(pile_m.get("counted_unitigs", 0)),
        "median_depth": float(pile_m.get("median_depth", 0.0)),
        "wall_s": wall_s,
        "total_wall_s": wall_s,
    }
