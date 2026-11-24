"""Gate 3 (focused): every QC-pass Pf8 isolate at the curated loci only, from Sanger CRAM containers.

Whole-genome streaming of 24,409 isolates is ~35 TB. This mode fetches, per isolate, only the CRAM containers
(located through the cached `.crai`) that cover:
  - the 9 drug-resistance genes (kelch13, crt, mdr1, dhfr, dhps, exo, arps10, fd, mdr2), gene body +- 2 kb
  - the CNV genes (crt, mdr1, gch1, plasmepsin 2/3) +- 2 kb and the left end of every tandem-duplication
    breakpoint Pf8 observed (the face-away read sits there; its mate position is in the record)
  - hrp2 / hrp3 gene body +- 2 kb
The Sanger CRAMs are aligned to Pf3D7 v3 (BWA-MEM, duplicates marked), so the containers are decoded with
`samtools view` and fed straight into the engine's pileup kernel: no re-mapping, read pairs kept (face-away
junction pairs for sWGA CNV calls). Calls reuse the engine's code (phased codons, kelch13 349-726 scan, junction
clusters, F_WS on the Pf8 site panel inside the fetched containers). Coverage CNV / hrp ratios are normalised
against every fetched core bin more than 250 kb from a CNV or hrp locus (the whole-genome engine's 50-250 kb
same-contig flanks would add ~0.37 TB).

Isolates already streamed whole-genome (Gate 1 / Gate 2 cohorts) are projected from their saved state: 0 bytes.

    python focused_cohort.py plan
    python focused_cohort.py run --cohort $W/cohort_24409_focused --exclude manifests/pf8_truth_200.parquet \
        --exclude manifests/pf8_raw_stream_2000.parquet --workers 4 --streams 96
    python focused_cohort.py project --cohort $W/cohort_24409_focused --source $W/truth_200_fast_v2 --source $W/cohort_2000
    python focused_cohort.py eval --cohort $W/cohort_24409_focused
"""
import argparse
import gzip
import io
import json
import multiprocessing as mp
import os
import queue
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import caller as C
from . import genome as G
from . import prepare as P
from . import state_store as SS
from . import streamer as ST

import tempfile

S = G.default_staging_dir()
W = G.default_work_dir()
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifests" / "pf8_cram_24409.parquet"
CRAI_DIR = S / "data" / "crai"
RELEASE = (G.REFDIR.parent / "data" / "pf8_release") if (G.REFDIR.parent / "data" / "pf8_release").exists() else (S / "data" / "pf8_release")
BREAKPOINTS = RELEASE / "Pf8_tandem_duplication_breakpoints.tsv"
OBSERVED_BP_JSON = RELEASE / "observed_breakpoints.json"
CNV_TABLE = RELEASE / "Pf8_cnv_calls.tsv"
STAGE = Path(os.environ.get("MALARIAVAR_STAGE_DIR") or (Path(tempfile.gettempdir()) / f"focaloop_focused_{os.getuid()}"))
DRM_GENES = ["kelch13", "crt", "mdr1", "dhfr", "dhps", "exo", "arps10", "fd", "mdr2"]
CNV_GENES = ["crt", "mdr1", "gch1", "pm2", "pm3"]
DEL_GENES = ["hrp2", "hrp3"]
GENE_FLANK = 2000
BP_FLANK = 500
BASELINE_EXCL = 250_000   # bins this close to a CNV / hrp gene never enter the copy-number baseline
SITE_WIN = 30_000         # panel sites this close to a target window go into the warm matrices
PIECE = 1 << 20           # range GET size (Sanger gives each connection ~0.05-0.3 MB/s: many small GETs)
CHUNK = 4 << 20
FOCUS_VERSION = "focused_v1"
# gch1 coverage ratios on ~100 baseline bins are noisier than the other loci: on the 200 Gate 1 isolates
# (whole-genome accumulators masked to each isolate's focused bins) 1.5 gave 14 false gDNA gch1 amplifications,
# 1.7 gave 2 (one missed); the 22,232 streamed Gate 3 isolates are out of sample for this setting
AMP_RATIO = {"gch1": 1.7}
# 8 barcode windows: F_WS r vs Pf8 0.818 -> 0.925, clonal agreement 0.796 -> 0.958 (200 Gate 1 isolates,
# whole-genome WSAF restricted to the focused sites) for +12 MB per isolate; 16 windows: 0.936 / 0.964, +23 MB
BARCODE_K = 8
BARCODE_WIN = 15_000
# hrp2 exon 2 is a repeat: BWA leaves MAPQ >= 20 bases in deleted exons and intact isolates go as low as
# 0.005 there, so a focused hrp2 deletion is an emptied 1 kb gene bin alone (deleted <= 0.046, intact >= 0.29
# on the 200 Gate 1 isolates); hrp3 keeps the engine rule (bin < 0.3 and exon window < 0.05)
DEL_BIN_ONLY = {"hrp2": 0.15}
SOURCE = "sanger_crai_focused"


# ------------------------------------------------------------------------------------------------ regions
def observed_breakpoints():
    if OBSERVED_BP_JSON.exists():
        return set(json.loads(OBSERVED_BP_JSON.read_text())) - {"-", ""}
    if not CNV_TABLE.exists():
        bp = pd.read_csv(BREAKPOINTS, sep="\t")
        return set(bp["breakpoint_id"].dropna().astype(str)) - {"-", ""}
    c = pd.read_csv(CNV_TABLE, sep="\t", usecols=["CRT_breakpoint", "GCH1_breakpoint", "MDR1_breakpoint",
                                                   "PM2_PM3_breakpoint"])
    ids = set()
    for col in c.columns:
        ids |= set(c[col].dropna().astype(str))
    return ids - {"-", ""}


def bp_key(x):
    """the CNV table and the breakpoint table spell IDs differently (PfCRT_dup_1 / CRT_dup_1,
    PfGCH1_dup_01 / PfGCH1_dup_1, PfPlasmepsin_dup_1 / PfPlasmepsin_1)."""
    k = str(x).lower()
    k = k[2:] if k.startswith("pf") else k
    k = k.replace("dup_", "")
    head, _, num = k.rpartition("_")
    return f"{head}_{int(num)}" if num.isdigit() else k


def barcode_windows(g, panel, k=BARCODE_K, win=BARCODE_WIN, min_maf=0.1):
    """the k core 15 kb windows with the most Pf8 panel sites at population MAF >= 0.1 (at most
    ceil(k / 14) per chromosome), for F_WS and F_ST: sites near the drug genes carry little diversity
    (65 informative panel sites in the gene containers). Deterministic from the panel."""
    goff, af = panel["goff"], panel["af"]
    inf = np.minimum(af, 1 - af) >= min_maf
    cap = -(-k // 14)
    cand = []
    for ci in range(14):
        L, a0 = int(g.lengths[ci]), int(g.offsets[ci])
        sel = (goff >= a0) & (goff < a0 + L)
        gs = goff[sel][inf[sel]]
        starts = np.arange(1, L - win, win // 2)
        cnt = np.searchsorted(gs, a0 + starts - 1 + win) - np.searchsorted(gs, a0 + starts - 1)
        core = np.array([g.core[a0 + s_ - 1: a0 + s_ - 1 + win].mean() for s_ in starts])
        cand += [(-int(c), ci, int(s_)) for c, s_, cf in zip(cnt, starts, core) if cf >= 0.9]
    cand.sort()
    out, per = [], {}
    for _, ci, s_ in cand:
        if len(out) >= k:
            break
        if per.get(ci, 0) >= cap or any(c2 == ci and abs(s2 - s_) < win for c2, s2, _ in out):
            continue
        out.append((ci, s_, s_ + win - 1))
        per[ci] = per.get(ci, 0) + 1
    return out


def regions(g=None, n_barcode=None):
    """[(contig index, start, end, label)], 1-based inclusive target windows (the same for every isolate)."""
    g = g or G.load()
    out = []
    for k in dict.fromkeys(DRM_GENES + CNV_GENES + DEL_GENES):
        x = g.gene(k)
        out.append((g.cidx(x["contig"]), max(1, x["start"] - GENE_FLANK), x["end"] + GENE_FLANK, f"gene:{k}"))
    bp = pd.read_csv(BREAKPOINTS, sep="\t")
    seen = {bp_key(x) for x in observed_breakpoints()}
    for r in bp.itertuples():
        if bp_key(r.breakpoint_id) in seen:
            out.append((g.cidx(r.chrom), max(1, int(r.first_breakpoint_start) - BP_FLANK),
                        int(r.first_breakpoint_end) + BP_FLANK, f"bp:{r.breakpoint_id}"))
    for ci, s_, e_ in barcode_windows(g, P.load_panel(), BARCODE_K if n_barcode is None else n_barcode):
        out.append((ci, s_, e_, f"barcode:{g.names[ci]}:{s_}"))
    return out


def parse_crai(path):
    raw = Path(path).read_bytes()
    txt = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    return np.loadtxt(io.StringIO(txt), dtype=np.int64, ndmin=2)


def plan_ranges(crai, regs, read_pad=0, cram_size=None):
    """(header end, [(a, b)] coalesced byte ranges of whole containers, n containers, covered spans).
    A container is needed when one of its slices overlaps a window: a slice's alignment span already runs to
    the end of its last read, so a read that starts before the window and reaches into it is caught by its own
    slice. Spans are the (contig, start, end) alignment extents of every slice in the chosen containers."""
    a = crai if isinstance(crai, np.ndarray) else parse_crai(crai)
    if len(a) == 0:
        raise ValueError("Empty CRAI index")
    offs = np.unique(a[:, 3])
    hdr_end = int(offs[0])
    nxt = {int(o): int(n) for o, n in zip(offs[:-1], offs[1:])}
    seq, st, sp = a[:, 0], a[:, 1], a[:, 2]
    need = np.zeros(len(a), bool)
    for ci, s, e, _ in regs:
        need |= (seq == ci) & (st <= e) & (st + sp - 1 >= s - read_pad)
    cont = sorted(set(a[need, 3].tolist()))
    ranges = []
    eof_len = len(ST.CRAM_EOF)
    for o in cont:
        end = nxt.get(int(o))
        if end is None:
            if cram_size is None or not np.isfinite(cram_size) or int(cram_size) <= int(o):
                raise ValueError(
                    f"cram_size is required when target regions overlap the final CRAM container at offset {int(o)}"
                )
            cs = int(cram_size)
            end = cs - eof_len if cs > int(o) + eof_len else cs
            if end <= int(o):
                raise ValueError(f"Invalid final CRAM container bounds [{int(o)}, {end}) for cram_size={cs}")
        if ranges and ranges[-1][1] == o:
            ranges[-1][1] = end
        else:
            ranges.append([int(o), end])
    sl = a[np.isin(a[:, 3], cont) & (a[:, 0] >= 0)]
    spans = np.stack([sl[:, 0], sl[:, 1], sl[:, 1] + sl[:, 2] - 1], 1) if len(sl) else np.zeros((0, 3), np.int64)
    return hdr_end, [tuple(r) for r in ranges], len(cont), spans


def valid_bins(g, spans, lead=200, tail=300):
    """1 kb bins whose reads are all inside the fetched containers: a read starting in the unfetched previous
    container can reach `lead` bp into a span, and reads of the unfetched next container start up to the
    last read start (~`tail` bp before the span end)."""
    vb = np.zeros(int(g.bin_offsets[-1]), bool)
    if not len(spans):
        return vb
    for ci in np.unique(spans[:, 0]):
        s = spans[spans[:, 0] == ci]
        s = s[np.argsort(s[:, 1])]
        merged = []
        for _, a, b in s:
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            lo, hi = a + lead, b - tail
            b0 = (lo - 1 + G.BIN - 1) // G.BIN          # first bin starting at or after lo
            b1 = hi // G.BIN                               # bins ending at or before hi
            if b1 > b0:
                vb[int(g.bin_offsets[ci]) + b0: int(g.bin_offsets[ci]) + b1] = True
    return vb


def baseline_exclusion(g):
    ex = np.zeros(int(g.bin_offsets[-1]), bool)
    for k in CNV_GENES + DEL_GENES:
        x = g.gene(k)
        ci = g.cidx(x["contig"])
        b0, b1 = g.bin_range(x["contig"], max(1, x["start"] - BASELINE_EXCL),
                             min(int(g.lengths[ci]), x["end"] + BASELINE_EXCL))
        ex[b0:b1] = True
    return ex


def site_subset(g, panel, regs):
    """panel site indices within SITE_WIN of a target window (the only sites a focused isolate can cover)."""
    goff = panel["goff"]
    keep = np.zeros(len(goff), bool)
    for ci, s, e, _ in regs:
        a = g.goff(g.names[ci], max(1, s - SITE_WIN))
        b = g.goff(g.names[ci], min(int(g.lengths[ci]), e + SITE_WIN))
        keep |= (goff >= a) & (goff <= b)
    return np.where(keep)[0]


_REGS = None


def _plan_one(item):
    s, cbytes = item
    h, rr, nc, _ = plan_ranges(CRAI_DIR / f"{s}.cram.crai", _REGS, cram_size=cbytes)
    return s, h, sum(b - a_ for a_, b in rr), len(rr), nc


def cmd_plan(a):
    """bytes per isolate for the region set, over every cached .crai (no network)."""
    global _REGS
    g = G.load()
    regs = _REGS = regions(g)
    m = pd.read_parquet(MANIFEST, columns=["Sample", "cram_bytes"])
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_plan_one, list(zip(m.Sample.tolist(), m.cram_bytes.tolist())), chunksize=64)
    df = pd.DataFrame(rows, columns=["Sample", "header_bytes", "range_bytes", "n_ranges", "n_containers"])
    df = df.merge(m, on="Sample")
    df.to_parquet(a.out, index=False)
    tot = (df.header_bytes + df.range_bytes).sum()
    print(json.dumps(dict(n=len(df), n_regions=len(regs), total_TB=tot / 1e12, cram_TB=df.cram_bytes.sum() / 1e12,
                          reduction=df.cram_bytes.sum() / tot,
                          median_MB=float((df.header_bytes + df.range_bytes).median() / 1e6),
                          median_containers=float(df.n_containers.median())), indent=1))


# ------------------------------------------------------------------------------------------------ fetch
class FetchJob:
    """CRAM header + the chosen containers, written contiguously into <path>.part with 1 MB range GETs on the
    shared keep-alive pool, then the CRAM 3.0 EOF container; renamed when the last piece lands."""

    def __init__(self, sample, url, hdr_end, ranges, path, on_end):
        self.sample, self.url, self.path = sample, url, Path(path)
        self.part = Path(str(path) + ".part")
        pieces, dst = [], 0
        for a, b in [(0, hdr_end)] + list(ranges):
            for x in range(a, b, PIECE):
                y = min(b, x + PIECE)
                pieces.append((x, y - 1, x - (dst + (x - a))))
            dst += b - a
        self.total = dst
        self.pieces = pieces
        self.left = len(pieces)
        self.failed = None
        self.lock = threading.Lock()
        self.on_end = on_end
        self.fd = os.open(str(self.part), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self.t0 = time.time()
        self.t1 = None

    def submit(self, ex):
        for a, b, shift in self.pieces:
            ex.submit(self._one, a, b, shift)

    def _one(self, a, b, shift):
        err = None
        if self.failed is None:
            try:
                ST.pull_range(self.url, self.fd, a, b, shift=shift)
            except Exception as e:
                err = e
        with self.lock:
            if err is not None and self.failed is None:
                self.failed = err
            self.left -= 1
            last = self.left == 0
        if last:
            try:
                if self.failed is None:
                    os.pwrite(self.fd, ST.CRAM_EOF, self.total)
                os.close(self.fd)
                if self.failed is None:
                    os.replace(self.part, self.path)
                else:
                    self.part.unlink(missing_ok=True)
            finally:
                self.t1 = time.time()
                self.on_end(self)


# ------------------------------------------------------------------------------------------------ worker
_WS = {}


def _init_worker(use_omnitig_zoom=False):
    g = G.load()
    from . import engine as E
    _WS.update(g=g, panel=P.load_panel(), cp=C.CallParams(use_omnitig_zoom=bool(use_omnitig_zoom)),
               zm=E.zoom_mask(g), excl=baseline_exclusion(g), drm=E.DRM_CODONS)


def regroup(sam):
    """mates adjacent and QNAME -> integer spot id (the pileup kernel parses QNAME as the spot number and
    counts overlapping mate bases once when mates are consecutive). Returns (SAM bytes, records, spots)."""
    lines = sam.split(b"\n")
    if lines and not lines[-1]:
        lines.pop()
    lines = [x for x in lines if x and x[:1] != b"@"]
    keys = [x[:x.find(b"\t")] for x in lines]
    order = sorted(range(len(lines)), key=keys.__getitem__)
    out, prev, sid = [], None, 0
    for i in order:
        k = keys[i]
        if k != prev:
            sid += 1
            prev = k
        out.append(b"%d" % sid + lines[i][len(k):])
    return (b"\n".join(out) + b"\n") if out else b"", len(lines), sid


def gc_ratio(g, depth, base, deg=2, iters=3):
    """depth / expected depth at the bin's GC, scaled so baseline bins have median 1. Expected depth is a
    robust polynomial fit of log depth on GC over the ~100 baseline bins (Pf gDNA libraries lose >10x depth
    from 25% to 10% GC; 1% GC strata need far more bins than a focused isolate has). GC is clamped to the
    baseline's 2-98% range, so the fit never extrapolates."""
    med = float(np.median(depth[base])) if base.any() else np.nan
    gc = g.bin_gc
    x, y = gc[base], np.log(depth[base])
    if len(x) < 30:
        return np.full(len(depth), np.nan), med
    keep = np.ones(len(x), bool)
    for _ in range(iters):
        coef = np.polyfit(x[keep], y[keep], deg)
        res = y - np.polyval(coef, x)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826 + 1e-9
        keep = np.abs(res - np.median(res[keep])) < 3 * mad
    lo, hi = np.percentile(x, [2, 98])
    e = np.exp(np.polyval(coef, np.clip(gc, lo, hi)))
    ratio = np.where(e > 0, depth / e, np.nan)
    norm = np.nanmedian(ratio[base])
    return ratio / norm, med


def focused_cnv(g, acc, p, sample_type, vb, excl):
    """amplification (crt, mdr1, gch1, pm2/3) and hrp2 / hrp3 deletion calls on a focused isolate: the engine's
    rules (coverage ratio >= 1.5 on gDNA or >= 3 face-away pairs spanning the gene; hrp gDNA deletion = a MAPQ>=1
    gene bin below 0.3 and a 300 bp exon window below 0.05), with ratios relative to the fetched baseline bins."""
    depth = C.bin_depth(g, acc)
    base = vb & (g.bin_core >= 0.9) & (depth > 0) & ~excl
    gdna = C._normalize_sample_type(sample_type) == "gdna"
    rows = []
    if base.sum() < 50:
        ratio, med = np.full(len(depth), np.nan), np.nan
    else:
        ratio, med = gc_ratio(g, depth, base)
    rf = acc.rf[:int(acc.rf_n[0])]
    for name, keys in G.CNV_LOCI.items():
        genes = [g.gene(k) for k in keys]
        contig = genes[0]["contig"]
        b0, b1 = g.bin_range(contig, min(x["start"] for x in genes), max(x["end"] for x in genes))
        v = ratio[b0:b1][vb[b0:b1]]
        v = v[np.isfinite(v)]
        r = float(np.median(v)) if len(v) else np.nan
        ci = g.cidx(contig)
        sel = rf[rf[:, 0] == ci] if len(rf) else rf
        fa = C.junction_support(sel, genes, p)
        cov = bool(gdna and np.isfinite(r) and r >= AMP_RATIO.get(name, p.amp_ratio))
        bad = (not np.isfinite(med)) or med < 5 or (gdna and not np.isfinite(r) and fa == 0)
        rows.append(dict(locus=name, kind="amp", ratio=r, se=np.nan, n_bins=int(len(v)),
                         call=-1 if bad else int(cov or fa >= p.faceaway_min),
                         method=("coverage+junction" if gdna else "junction_only") + "_focused_v1",
                         faceaway_pairs=float(fa)))
    for name, key in G.DEL_LOCI.items():
        gg = g.gene(key)
        b0, b1 = g.bin_range(gg["contig"], gg["start"], gg["end"])
        v = ratio[b0:b1][vb[b0:b1]]
        v = v[np.isfinite(v)]
        rmin = float(v.min()) if len(v) else np.nan
        rgene = float(np.median(v)) if len(v) else np.nan
        ex = max(gg["exons"] or gg["cds"], key=lambda e: e[1] - e[0])
        a_, b_ = g.goff(gg["contig"], ex[0]), g.goff(gg["contig"], ex[1]) + 1
        d = (acc.counts[a_:b_].sum(1) + acc.delc[a_:b_]).astype(np.float64)
        wmin = np.nan
        if len(d) >= 300 and np.isfinite(med) and med > 0:
            cs = np.concatenate([[0], np.cumsum(d)])
            wmin = float(((cs[300:] - cs[:-300]) / 300).min() / med)
        if not gdna:
            call = -1
        else:
            if name in DEL_BIN_ONLY:
                call = -1 if (not np.isfinite(rmin) or not np.isfinite(med) or med < 5) else int(rmin < DEL_BIN_ONLY[name])
            else:
                call = -1 if (not np.isfinite(rmin) or not np.isfinite(wmin) or not np.isfinite(med) or med < 5) else \
                    int(rmin < p.del_ratio and wmin < p.del_window)
        rows.append(dict(locus=name, kind="del", ratio=rmin, se=np.nan, n_bins=int(len(v)), call=call,
                         method="bin_and_window_focused_v1" if gdna else "no_call_amplified_library",
                         ratio_bins=rgene, ratio_window=wmin, ratio_telomere=np.nan, faceaway_pairs=np.nan))
    return rows, ratio, med, int(base.sum())


def process_one(job):
    """decode one focused CRAM, pile it up, call, write the sample's state; returns telemetry."""
    sample, path, row, cohort, spans, info = job
    t0 = time.time()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    rc0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    g, panel, cp = _WS["g"], _WS["panel"], _WS["cp"]
    try:
        pr = subprocess.run(["samtools", "view", "--no-PG", "-T", str(G.REF_FASTA), str(path)], capture_output=True)
        if pr.returncode != 0:
            raise RuntimeError(f"samtools view rc={pr.returncode}: {pr.stderr.decode()[-300:]}")
        t_dec = time.time()
        buf, n_rec, n_spots = regroup(pr.stdout)
        del pr
        if int(info.get("n_containers", 0)) > 0 and n_rec == 0:
            raise RuntimeError(
                f"Focused CRAM decode yielded 0 SAM records across {info['n_containers']} planned containers"
            )
        acc = C.Accumulator(g, cp, zoom_mask=_WS["zm"])
        i = 0
        while i < len(buf):
            j = min(len(buf), i + CHUNK)
            if j < len(buf):
                j = buf.rfind(b"\n", i, j) + 1
            arr = np.frombuffer(buf, np.uint8, count=j - i, offset=i)
            acc.feed(arr, len(arr))
            i = j
        del buf
        t_pile = time.time()
        vb = valid_bins(g, np.asarray(spans, np.int64))
        w, d = C.wsaf_at(acc, panel)
        fws, n_fws = C.fws(w, d, panel["af"])
        clonal = bool(np.isfinite(fws) and fws >= cp.clonal_fws)
        st = row.get("Sample_type") or "gDNA"
        codons, k13, _, _, _, _ = C.curated_calls(g, acc, _WS["drm"], None, cp, sample_type=st)
        codons["k13_ns"] = k13
        cnv, ratio, med, n_base = focused_cnv(g, acc, cp, st, vb, _WS["excl"])
        snv = C.call_snvs(g, acc, cp, clonal=clonal)
        keep = snv["gt"] > 0
        vc = {k: snv[k][keep] for k in ("goff", "kind", "ref", "alt", "alen", "ad_ref", "ad_alt", "dp", "wsaf", "qual", "gt")}
        ci, pos, refs, alts = C.vcf_alleles(g, vc)
        t_call = time.time()
        variant_calls = pd.DataFrame(dict(sample=sample, chrom=ci, pos=pos, ref=refs, alt=alts, kind=vc["kind"],
                                          gt=vc["gt"], ad_ref=vc["ad_ref"].astype(np.int32),
                                          ad_alt=vc["ad_alt"].astype(np.int32), dp=vc["dp"].astype(np.int32),
                                          wsaf=vc["wsaf"], qual=vc["qual"], core=g.core[vc["goff"]], zoom=np.int8(0)))
        wu8 = np.where(d >= 3, np.round(np.nan_to_num(w) * 200), 255).astype(np.uint8)
        du8 = np.minimum(d, 254).astype(np.uint8)
        wsaf_sketch = pd.DataFrame(dict(sample=[sample], n_sites=[len(w)], wsaf_u8=[wu8.tobytes()], dp_u8=[du8.tobytes()]))
        depth_bins = pd.DataFrame(dict(sample=[sample], ratio_f16=[ratio.astype(np.float16).tobytes()],
                                       bin_median=[float(med) if np.isfinite(med) else np.nan]))
        cnv_df = pd.DataFrame([dict(sample=sample, **c) for c in cnv])
        gene_depth = {}
        for k in ("kelch13", "crt", "mdr1", "dhfr", "dhps"):
            x = g.gene(k)
            a_, b_ = g.goff(x["contig"], x["start"]), g.goff(x["contig"], x["end"]) + 1
            gene_depth[k] = float(np.median(acc.counts[a_:b_].sum(1)))
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        rc1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        tel = dict(sample=sample, engine=FOCUS_VERSION, source=SOURCE, stop_reason="focused", error=None,
                   sample_type=st, wire_bytes=float(info["bytes"]), archive_bytes=float(row.get("cram_bytes") or np.nan),
                   n_ranges=int(info["n_ranges"]), n_containers=int(info["n_containers"]), sam_records=int(n_rec),
                   spots=int(n_spots), reads_counted=int(acc.stats[2]), faceaway_pairs=int(acc.rf_n[0]),
                   baseline_bins=n_base, baseline_median_depth=float(med) if np.isfinite(med) else np.nan,
                   median_core_depth=float(med) if np.isfinite(med) else np.nan,
                   depth_kelch13=gene_depth["kelch13"], depth_crt=gene_depth["crt"], depth_mdr1=gene_depth["mdr1"],
                   depth_dhfr=gene_depth["dhfr"], depth_dhps=gene_depth["dhps"],
                   fws=fws, fws_sites=int(n_fws), clonal_policy="clonal" if clonal else "mixed",
                   n_snv=int(keep.sum()), fetch_s=float(info["fetch_s"]), decode_s=t_dec - t0, pileup_s=t_pile - t_dec,
                   call_s=t_call - t_pile, wall_s=float(info["fetch_s"]) + (time.time() - t0),
                   cpu_self_s=(r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime),
                   cpu_children_s=(rc1.ru_utime + rc1.ru_stime) - (rc0.ru_utime + rc0.ru_stime),
                   k13_ns=k13, **{f"aa_{k}": v for k, v in codons.items()})
        SS.write_sample(cohort, sample, dict(variant_calls=variant_calls, wsaf_sketch=wsaf_sketch,
                                             depth_1kb_bins=depth_bins, cnv_del_calls=cnv_df,
                                             sample_telemetry=pd.DataFrame([tel])))
        return tel
    except Exception as e:
        return dict(sample=sample, stop_reason="crash", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
    finally:
        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ run
def net_rx():
    tot = 0
    for line in open("/proc/net/dev").read().splitlines()[2:]:
        name, rest = line.split(":", 1)
        if name.strip() != "lo":
            tot += int(rest.split()[0])
    return tot


def cohort_manifest(cohort, shard_size=1000):
    """all 24,409 isolates, sharded by manifest position (streamed and projected isolates share shards)."""
    cohort = Path(cohort)
    p = cohort / "manifest.parquet"
    if p.exists():
        return pd.read_parquet(p)
    m = pd.read_parquet(MANIFEST, columns=["Sample", "Country", "Population", "Year", "Sample_type"])
    m["shard"] = np.arange(len(m)) // shard_size
    cohort.mkdir(parents=True, exist_ok=True)
    m.to_parquet(p, index=False)
    return m


def seal_ready(cohort, cm):
    done = SS.done_samples(cohort)
    out = []
    for sid, grp in cm.groupby("shard"):
        sh = Path(cohort) / "shards" / f"shard_{sid:03d}" / "SEALED"
        if not sh.exists() and grp.Sample.isin(done).all() and all(SS.is_done(cohort, s) for s in grp.Sample):
            SS.seal_shard(cohort, int(sid), grp.Sample.tolist())
            out.append(int(sid))
    return out


def cmd_run(a):
    cohort = Path(a.cohort)
    cm = cohort_manifest(cohort)
    meta = pd.read_parquet(MANIFEST, columns=["Sample", "cram_url", "cram_bytes", "Sample_type"])
    excl = set()
    for f in a.exclude or []:
        excl |= set(pd.read_parquet(f, columns=["Sample"]).Sample)
    rows = meta[~meta.Sample.isin(excl)]
    if a.samples:
        rows = rows[rows.Sample.isin(set(pd.read_parquet(a.samples, columns=["Sample"]).Sample)
                                     if a.samples.endswith(".parquet") else set(a.samples.split(",")))]
    done = SS.done_samples(cohort)
    rows = rows[~rows.Sample.isin(done)]
    if a.limit:
        rows = rows.head(a.limit)
    g = G.load()
    regs = regions(g)
    STAGE.mkdir(parents=True, exist_ok=True)
    stage = STAGE / f"{cohort.name}.{os.getpid()}"
    stage.mkdir()
    (cohort / "regions.json").write_text(json.dumps([dict(contig=g.names[c], start=int(s), end=int(e), label=l)
                                                     for c, s, e, l in regs], indent=0))
    log = open(cohort / "run_log.jsonl", "a")
    t0 = time.time()
    rx0 = net_rx()
    print(f"{time.strftime('%H:%M:%S')} focused cohort {cohort}: todo {len(rows)}, done {len(done)}, "
          f"{len(regs)} windows, streams {a.streams}, in flight {a.in_flight}, workers {a.workers}", flush=True)
    nic_stop = threading.Event()

    def nic_sampler(path=cohort / f"nic_rx_{int(t0)}.tsv", every=10.0):
        with open(path, "w") as f:
            f.write("t_s\trx_bytes\tdone\n")
            while True:
                f.write(f"{time.time() - t0:.1f}\t{net_rx() - rx0}\t{nd[0]}\n")
                f.flush()
                if nic_stop.wait(every):
                    break

    nd = [0]
    threading.Thread(target=nic_sampler, daemon=True).start()
    ex = ThreadPoolExecutor(a.streams, thread_name_prefix="crai")
    use_omni = bool(getattr(a, "omnitig_zoom", False))
    pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(use_omni,), maxtasksperchild=200)
    results = queue.Queue()
    slots = threading.BoundedSemaphore(a.in_flight)
    backlog = threading.BoundedSemaphore(a.max_staged)
    fetched_bytes = [0]

    def on_end(job, row, spans, nr, nc):
        slots.release()
        if job.failed is not None:
            backlog.release()
            results.put(dict(sample=job.sample, stop_reason="crash", error=f"fetch: {type(job.failed).__name__}: {job.failed}"))
            return
        fetched_bytes[0] += job.total
        info = dict(bytes=job.total + len(ST.CRAM_EOF), n_ranges=nr, n_containers=nc, fetch_s=job.t1 - job.t0)

        def done_cb(tel):
            backlog.release()
            results.put(tel)
        pool.apply_async(process_one, ((job.sample, str(job.path), row, str(cohort), spans.tolist(), info),),
                         callback=done_cb, error_callback=lambda e, s=job.sample: (backlog.release(), results.put(
                             dict(sample=s, stop_reason="crash", error=f"{type(e).__name__}: {e}"))))

    def scheduler():
        for r in rows.itertuples():
            row = dict(Sample=r.Sample, cram_bytes=r.cram_bytes, Sample_type=r.Sample_type)
            try:
                hdr, rr, nc, spans = plan_ranges(
                    CRAI_DIR / f"{r.Sample}.cram.crai", regs, cram_size=r.cram_bytes
                )
            except Exception as e:
                results.put(dict(sample=r.Sample, stop_reason="crash", error=f"plan: {type(e).__name__}: {e}"))
                continue
            backlog.acquire()
            slots.acquire()
            job = FetchJob(r.Sample, r.cram_url, hdr, rr, stage / f"{r.Sample}.cram",
                           lambda j, row=row, spans=spans, nr=len(rr) + 1, nc=nc: on_end(j, row, spans, nr, nc))
            job.submit(ex)

    th = threading.Thread(target=scheduler, daemon=True)
    th.start()
    n = 0
    crashes = 0
    try:
        while n < len(rows):
            tel = results.get()
            n += 1
            nd[0] = n
            tel["t_done_s"] = time.time() - t0
            if tel.get("stop_reason") == "crash":
                crashes += 1
            log.write(json.dumps(tel, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
            log.flush()
            if n % a.report == 0 or n == len(rows) or tel.get("stop_reason") == "crash":
                el = time.time() - t0
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(rows)} {tel.get('sample')} {tel.get('stop_reason')} "
                      f"{(tel.get('error') or '')[:120]} useful={fetched_bytes[0] / el / 1e6:.2f}MB/s "
                      f"nic={(net_rx() - rx0) / el / 1e6:.2f}MB/s rate={n / el * 3600:.0f}/h crashes={crashes}", flush=True)
            if n % 500 == 0:
                for sid in seal_ready(cohort, cm):
                    print(f"{time.strftime('%H:%M:%S')} sealed shard {sid}", flush=True)
    finally:
        pool.close() if n >= len(rows) else pool.terminate()
        pool.join()
        ex.shutdown(wait=False, cancel_futures=True)
        nic_stop.set()
        shutil.rmtree(stage, ignore_errors=True)
    for sid in seal_ready(cohort, cm):
        print(f"{time.strftime('%H:%M:%S')} sealed shard {sid}", flush=True)
    summary = dict(cohort=str(cohort), n=len(rows), crashes=crashes, wall_s=time.time() - t0,
                   nic_rx_bytes=net_rx() - rx0, useful_bytes=fetched_bytes[0])
    (cohort / f"run_summary_{int(t0)}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------------------------------------ projection
PROJ_TEL = ["sample", "engine", "source", "stop_reason", "error", "sample_type", "wire_bytes", "archive_bytes",
            "median_core_depth", "fws", "fws_sites", "clonal_policy", "faceaway_pairs", "wall_s", "cpu_self_s",
            "cpu_children_s"]


def cmd_project(a):
    """isolates streamed whole-genome (Gate 1 / Gate 2) enter the focused cohort from their saved state:
    telemetry with curated codon calls, CNV / hrp calls, WSAF sketch. 0 network bytes."""
    cohort = Path(a.cohort)
    cm = cohort_manifest(cohort)
    want = set(cm.Sample)
    done = SS.done_samples(cohort)
    n = 0
    for src in a.source:
        src = Path(src)
        name = src.name
        tel = SS.read_table(src, "sample_telemetry")
        tel = tel[tel["sample"].isin(want) & ~tel["sample"].isin(done)]
        if not len(tel):
            continue
        rp = src / "replay"
        cur = pd.read_parquet(rp / "curated.parquet").set_index("sample") if (rp / "curated.parquet").exists() else None
        cnv = SS.read_table(src, "cnv_del_calls")
        ws = SS.read_table(src, "wsaf_sketch")
        cnv = cnv[cnv["sample"].isin(set(tel["sample"]))]
        ws = ws[ws["sample"].isin(set(tel["sample"]))].set_index("sample")
        for _, t in tel.iterrows():
            s = t["sample"]
            row = {k: t.get(k) for k in PROJ_TEL if k in t.index}
            row.update({k: t[k] for k in t.index if k.startswith("aa_")})
            if cur is not None and s in cur.index:
                row.update({k: cur.at[s, k] for k in cur.columns if k.startswith("aa_")})
            row["k13_ns"] = row.get("aa_k13_ns")
            row["source"] = f"projected:{name}:{t.get('source')}"
            row["projected_wire_bytes"] = float(t.get("wire_bytes") or 0)
            row["wire_bytes"] = 0.0
            row["engine"] = f"{t.get('engine')}+{FOCUS_VERSION}"
            SS.write_sample(cohort, s, dict(sample_telemetry=pd.DataFrame([row]),
                                            cnv_del_calls=cnv[cnv["sample"] == s].copy(),
                                            wsaf_sketch=ws.loc[[s]].reset_index() if s in ws.index else None))
            done.add(s)
            n += 1
        print(f"{time.strftime('%H:%M:%S')} projected {len(tel)} from {src}", flush=True)
    sealed = seal_ready(cohort, cm)
    print(json.dumps(dict(projected=n, sealed=sealed, done=len(SS.done_samples(cohort)), of=len(cm))))


# ------------------------------------------------------------------------------------------------ eval + queries
def build_focused_warm(cohort):
    """warm matrices for the focused cohort: WSAF / depth at the panel sites a focused isolate can cover."""
    cohort = Path(cohort)
    warm = cohort / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    g = G.load()
    panel = P.load_panel()
    idx = site_subset(g, panel, regions(g))
    t0 = time.time()
    tel = SS.read_table(cohort, "sample_telemetry")
    meta = pd.read_parquet(MANIFEST, columns=["Sample", "Country", "Population", "Year", "F_WS", "Sample_type"])
    order, wl, dl = [], [], []
    for f in SS.table_files(cohort, "wsaf_sketch"):
        t = pd.read_parquet(f)
        for s, wb, db in zip(t["sample"], t["wsaf_u8"], t["dp_u8"]):
            order.append(s)
            wl.append(np.frombuffer(wb, np.uint8)[idx])
            dl.append(np.frombuffer(db, np.uint8)[idx])
    np.save(warm / "wsaf_u8.npy", np.stack(wl))
    np.save(warm / "dp_u8.npy", np.stack(dl))
    np.save(warm / "panel_idx.npy", idx)
    s = tel.drop_duplicates("sample").set_index("sample").reindex(order).reset_index().rename(columns={"index": "sample"})
    s = s.merge(meta.rename(columns={"F_WS": "F_WS_pf8", "Sample_type": "Sample_type_meta"}), left_on="sample",
                right_on="Sample", how="left")
    s.to_parquet(warm / "samples.parquet", index=False)
    SS.read_table(cohort, "cnv_del_calls").to_parquet(warm / "cnv.parquet", index=False)
    return dict(samples=len(order), sites=int(len(idx)), build_s=time.time() - t0,
                warm_bytes=int(sum(f.stat().st_size for f in warm.iterdir())))


def run_queries(cohort):
    """the Gate 2 query set on the focused warm state (24,409 isolates), sockets disabled in-process."""
    from . import query as Q
    Q.no_network()

    class FocusedWarm(Q.Warm):
        def __init__(self, cohort):
            w = Path(cohort) / "warm"
            t0 = time.time()
            self.samples = pd.read_parquet(w / "samples.parquet")
            self.wsaf = np.load(w / "wsaf_u8.npy", mmap_mode="r")
            self.dp = np.load(w / "dp_u8.npy", mmap_mode="r")
            self.cnv = pd.read_parquet(w / "cnv.parquet")
            full = P.load_panel()
            idx = np.load(w / "panel_idx.npy")
            self.panel = {k: (v[idx] if hasattr(v, "__len__") and len(v) == len(full["goff"]) else v)
                          for k, v in full.items()}
            self.g = G.load()
            self.genes = pd.read_parquet(G.REFDIR / "genes_all.parquet")
            self.load_s = time.time() - t0

    t0 = time.time()
    Wm = FocusedWarm(cohort)
    out = dict(cohort=str(cohort), n_samples=int(len(Wm.samples)), load_s=Wm.load_s, queries=[])
    out["queries"].append(q_spatiotemporal(Wm))
    out["queries"].append(Q.q_temporal_k13_pm23(Wm))
    out["queries"].append(Q.q_hrp(Wm))
    out["queries"].append(Q.q_fws(Wm))
    out["queries"].append(Q.q_drug(Wm))
    out["queries"].append(Q.q_fst(Wm, "AS-SE-E", "AF-W", "focused-site F_ST: eastern SE Asia vs West Africa"))
    out["queries"].append(Q.q_fst(Wm, "AF-E", "AF-W", "focused-site F_ST: East vs West Africa"))
    out["queries"].append(Q.q_poly_contrast(Wm))
    out["total_s"] = time.time() - t0
    out["max_query_s"] = max(q["seconds"] for q in out["queries"])
    return out


def q_spatiotemporal(Wm):
    """resistance-marker and CNV prevalence by country x 3-year period, all 24,409 isolates."""
    from . import query as Q
    t0 = time.time()
    s = Wm.samples
    f = Q.mutant_flags(s)
    c = Q.cnv_matrix(Wm)
    for k in ("mdr1", "pm2_pm3", "gch1", "crt", "hrp2", "hrp3"):
        f[k] = c[k].values if k in c else np.nan
    f["country"] = s.Country.values
    yr = pd.to_numeric(s.Year, errors="coerce")
    f["period"] = (yr // 3 * 3).astype("Int64").astype(str).values
    tab = f.groupby(["country", "period"]).agg(n=("k13_any", "size"), k13=("k13_any", "mean"), c580y=("k13_C580Y", "mean"),
                                              crt_K76T=("crt_K76T", "mean"), dhfr_triple=("dhfr_triple", "mean"),
                                              dhps_K540E=("dhps_K540E", "mean"), mdr1_amp=("mdr1", "mean"),
                                              pm23_amp=("pm2_pm3", "mean"), hrp2_del=("hrp2", "mean")).reset_index()
    return dict(query="resistance and CNV prevalence by country x 3-year period", n_cells=int(len(tab)),
                n_countries=int(tab.country.nunique()), table=tab.round(4).to_dict("records"), seconds=time.time() - t0)


def k13_marker(series, alt):
    v = series.astype(object)
    return v.map(lambda x: np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(alt in str(x).split(",")))


def cmd_queries(a):
    print(json.dumps(run_queries(a.cohort), default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def compare_whole_genome(tel, wg_cohort, meta, cohort):
    """focused calls against the whole-genome engine's calls on the same isolates (codons, CNV / hrp, F_WS)."""
    wg = SS.read_table(wg_cohort, "sample_telemetry").drop_duplicates("sample").set_index("sample")
    f = tel[tel.source == SOURCE].set_index("sample")
    common = f.index.intersection(wg.index)
    aa = [c for c in f.columns if c.startswith("aa_") and c in wg.columns and c != "aa_k13_ns"]
    a_, b_ = f.loc[common, aa].astype(str), wg.loc[common, aa].astype(str)
    called = ~a_.isin(["None", "nan"]) & ~b_.isin(["None", "nan"])
    out = dict(n=int(len(common)), codon_calls_both=int(called.values.sum()),
               codon_identity=float(((a_ == b_) & called).values.sum() / max(called.values.sum(), 1)),
               codon_called_focused_only=int((~a_.isin(["None", "nan"]) & b_.isin(["None", "nan"])).values.sum()),
               codon_called_wg_only=int((a_.isin(["None", "nan"]) & ~b_.isin(["None", "nan"])).values.sum()))
    k1, k2 = f.loc[common, "aa_k13_ns"].astype(str), wg.loc[common, "aa_k13_ns"].astype(str)
    out["k13_ns_identity"] = float((k1 == k2).mean())
    x = pd.DataFrame(dict(a=f.loc[common, "fws"], b=wg.loc[common, "fws"])).dropna()
    out["fws"] = dict(n=int(len(x)), pearson_r=float(np.corrcoef(x.a, x.b)[0, 1]) if len(x) > 2 else None,
                      mean_abs_diff=float((x.a - x.b).abs().mean()),
                      clonal_agreement=float(((x.a >= 0.95) == (x.b >= 0.95)).mean()))
    cf = SS.read_table(cohort, "cnv_del_calls")
    if len(cf):
        cw = SS.read_table(wg_cohort, "cnv_del_calls")
        j = cf.merge(cw, on=["sample", "locus"], suffixes=("_f", "_w"))
        j = j[j["sample"].isin(set(common))]
        both = j[(j.call_f >= 0) & (j.call_w >= 0)]
        out["cnv_hrp"] = {loc: dict(n=int(len(g_)), identical=int((g_.call_f == g_.call_w).sum()),
                                    focused_only_pos=int(((g_.call_f == 1) & (g_.call_w == 0)).sum()),
                                    wg_only_pos=int(((g_.call_f == 0) & (g_.call_w == 1)).sum()))
                          for loc, g_ in both.groupby("locus")}
    out["wire_bytes"] = dict(focused=float(f.loc[common, "wire_bytes"].sum()),
                             whole_genome=float(wg.loc[common, "wire_bytes"].sum()))
    out["wire_bytes"]["reduction"] = out["wire_bytes"]["whole_genome"] / max(out["wire_bytes"]["focused"], 1)
    return out


def cmd_eval(a):
    import gates as GT
    cohort = Path(a.cohort)
    meta = pd.read_parquet(MANIFEST).set_index("Sample")
    tel = SS.read_table(cohort, "sample_telemetry")
    tel = tel[tel.stop_reason != "crash"].drop_duplicates("sample")
    tel["kind"] = np.where(tel.source == SOURCE, "focused", "projected")
    res = dict(cohort=str(cohort), n_manifest=int(len(meta)), n_done=int(len(tel)),
               by_kind=tel.kind.value_counts().to_dict())
    kinds = {"all": tel, "focused": tel[tel.kind == "focused"], "projected": tel[tel.kind == "projected"]}
    cnv_all = SS.read_table(cohort, "cnv_del_calls")
    res["drm"], res["cnv"], res["hrp"], res["fws"], res["k13_validated"] = {}, {}, {}, {}, {}
    for k, t in kinds.items():
        if not len(t):
            continue
        dd, res["drm"][k] = GT.drm_table(t, meta)
        if k == "all":
            dd.to_parquet(HERE / "results" / f"{a.results_name}_drm_calls.parquet", index=False)
        sub = cnv_all[cnv_all["sample"].isin(set(t["sample"]))]
        if len(sub):
            cd, cres = GT.cnv_table(cohort, meta, sub)
            res["cnv"][k], res["hrp"][k] = cres["cnv"], cres["hrp"]
        st = meta.loc[t["sample"], "Sample_type"].values
        for stype in ("gDNA", "sWGA"):
            sub = cnv_all[cnv_all["sample"].isin(set(t["sample"][st == stype]))]
            if len(sub):
                _, cr = GT.cnv_table(cohort, meta, sub)
                res["cnv"][f"{k}_{stype}"] = cr["cnv"]
                res["hrp"][f"{k}_{stype}"] = cr["hrp"]
        x = pd.DataFrame(dict(ours=t.set_index("sample").fws, pf8=meta.loc[t["sample"], "F_WS"].values)).dropna()
        if len(x) > 2:
            res["fws"][k] = dict(n=int(len(x)), pearson_r=float(np.corrcoef(x.ours, x.pf8)[0, 1]),
                                 mean_abs_diff=float((x.ours - x.pf8).abs().mean()),
                                 clonal_agreement=float(((x.ours >= 0.95) == (x.pf8 >= 0.95)).mean()),
                                 n_no_fws=int(t.fws.isna().sum()))
        kv = {}
        for col, aa_col, alt in (("k13_C580Y", "aa_kelch13_580", "Y"), ("k13_R539T", "aa_kelch13_539", "T")):
            if aa_col not in t:
                continue
            ours = k13_marker(t.set_index("sample")[aa_col], alt)
            truth = meta.loc[t["sample"], col].map({"WT": 0.0, "hom": 1.0, "het": 1.0})
            j = pd.DataFrame(dict(o=ours.values, t=truth.values)).dropna()
            kv[col] = dict(n=int(len(j)), concordance=float((j.o == j.t).mean()) if len(j) else None,
                           tp=int(((j.o == 1) & (j.t == 1)).sum()), fn=int(((j.o == 0) & (j.t == 1)).sum()),
                           fp=int(((j.o == 1) & (j.t == 0)).sum()))
        res["k13_validated"][k] = kv
    f = kinds["focused"]
    if len(f):
        cram = meta.loc[f["sample"], "cram_bytes"].astype(float)
        res["cost_focused"] = dict(n=int(len(f)), wire_bytes_total=float(f.wire_bytes.sum()),
                                   wire_bytes_median=float(f.wire_bytes.median()), cram_bytes_total=float(cram.sum()),
                                   reduction_vs_cram=float(cram.sum() / f.wire_bytes.sum()),
                                   cpu_s_median=float((f.cpu_self_s + f.cpu_children_s).median()),
                                   cpu_s_total=float((f.cpu_self_s + f.cpu_children_s).sum()),
                                   fetch_s_median=float(f.fetch_s.median()),
                                   baseline_median_depth_median=float(f.baseline_median_depth.median()),
                                   fws_missing=int(f.fws.isna().sum()))
        sums = sorted(cohort.glob("run_summary_*.json"))
        if sums:
            ss = [json.loads(p.read_text()) for p in sums]
            res["cost_focused"]["runs"] = ss
    t1 = time.time()
    res["warm"] = build_focused_warm(cohort)
    out = HERE / "results" / f"{a.results_name.replace('24409_focused', 'queries')}_queries.json" \
        if a.results_name != "gate3_24409_focused" else HERE / "results" / "gate3_queries.json"
    trace = S / "logs" / "gate3_query_strace.txt"
    rx0 = net_rx()
    tq = time.time()
    subprocess.run(["strace", "-f", "-qq", "-c", "-e", "trace=%network", "-o", str(trace), sys.executable, "-B",
                    str(Path(__file__).resolve()), "queries", "--cohort", str(cohort)],
                   check=True, stdout=open(out, "w"))
    q = json.loads(out.read_text())
    res["queries"] = dict(wall_s=time.time() - tq, nic_rx_bytes=net_rx() - rx0,
                          network_syscalls=trace.read_text().strip() or "none", max_query_s=q["max_query_s"],
                          all_under_30s=bool(q["max_query_s"] < 30), load_s=q["load_s"],
                          per_query={qq["query"]: qq["seconds"] for qq in q["queries"]})
    res["eval_s"] = time.time() - t1
    if a.compare:
        res["vs_whole_genome"] = compare_whole_genome(tel, Path(a.compare), meta, cohort)
    js = json.dumps(GT.clean(res), indent=1, default=str)
    (HERE / "results" / f"{a.results_name}.json").write_text(js)
    if a.results_name != "gate3_24409_focused":
        print(js[:4000])
        return
    GT.write_status("gate3_24409_focused_done.json", dict(
        gate=3, mode="focused .crai loci", status="done", results="beyond_workflows/malariavar/results/gate3_24409_focused.json",
        n_done=res["n_done"], by_kind=res["by_kind"], drm=res["drm"].get("all", {}).get("concordance"),
        cnv=res["cnv"].get("all", {}).get("concordance"), hrp=res["hrp"].get("all", {}).get("concordance"),
        fws=res["fws"].get("all"), k13=res["k13_validated"].get("all"), queries=res["queries"],
        cost=res.get("cost_focused", {})))
    print(js[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "project", "eval", "queries"])
    ap.add_argument("--cohort")
    ap.add_argument("--out", default=str(S / "data" / "focused_plan.parquet"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--streams", type=int, default=96, help="concurrent range GETs to Sanger")
    ap.add_argument("--in-flight", type=int, default=48, help="isolates downloading at once")
    ap.add_argument("--max-staged", type=int, default=96, help="isolates fetched or downloading but not yet called")
    ap.add_argument("--exclude", action="append", help="manifest(s) of isolates not to stream (projected instead)")
    ap.add_argument("--samples", help="comma list or parquet with a Sample column (pilot)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report", type=int, default=50)
    ap.add_argument("--source", action="append", help="whole-genome cohort(s) to project from")
    ap.add_argument("--results-name", default="gate3_24409_focused", help="results/<name>.json (pilots: no status file)")
    ap.add_argument("--compare", help="whole-genome cohort of the same isolates: call identity")
    ap.add_argument("--omnitig-zoom", action="store_true", help="enable Simple Omnitigs + VarGeno QV + WSAF MLE in zoom_genes/curated_calls")
    a = ap.parse_args()
    if a.cmd != "plan":
        for v in ("TMPDIR",):
            val = os.environ.get(v, "")
            if val and not (val.startswith(str(W)) or val.startswith(tempfile.gettempdir())):
                sys.exit(f"{v} must point into {W} or {tempfile.gettempdir()}")
    {"plan": cmd_plan, "run": cmd_run, "project": cmd_project, "eval": cmd_eval, "queries": cmd_queries}[a.cmd](a)


if __name__ == "__main__":
    main()
