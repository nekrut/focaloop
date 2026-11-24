"""Stage 3: whole-genome pileup accumulation and calling (no BAM on disk).

Accumulator   per-base A/C/G/T counts, deletion / insertion counts, indel events, 1 kb depth bins
call_snvs     genome-wide SNPs with a binomial WSAF likelihood (hom-alt / mixed), all candidates kept
call_indels   left-normalised indel events with the same likelihood (homopolymer-aware error)
wsaf_at       WSAF + depth at the Pf8 site panel (sketch and F_WS input)
fws           moimix-style F_WS = 1 - slope(Hw ~ Hs) over PLAF bins
cnv_calls     GC-normalised 1 kb depth ratios, curated CNV amplifications and hrp2/hrp3 deletions
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import genome as G
from . import kernels as Kn

LN10 = np.log(10.0)


@dataclass
class CallParams:
    min_mapq: int = 20
    min_bq: int = 13
    bin_mapq: int = 1
    max_mm_frac: float = 0.20        # dev-set sweep: 0.12 -> 0.20 +0.0011 mono F1 (0.3 / off identical)
    snv_err: float = 0.005
    snv_min_alt: int = 2
    snv_min_wsaf: float = 0.7        # major allele (calibrated on 8 replays: 0.7 > 0.5, 0.8)
    snv_min_qual: float = 30.0
    snv_minor_err: float = 0.04      # minor alleles (mixed infections): conservative error model
    snv_minor_alt: int = 4
    snv_minor_wsaf: float = 0.10
    snv_novel_minor_penalty: float = 10.0  # extra Phred penalty for minor alleles off the Pf8 polymorphic site panel
    clonal_fws: float = 0.95         # sample F_WS reference point for clonality classification
    indel_err: float = 0.03          # base indel error rate
    indel_hp_err: float = 0.012      # per-bp homopolymer slippage increment beyond 4 bp
    indel_min_alt: int = 3
    indel_min_wsaf: float = 0.25
    indel_clonal_wsaf: float = 0.50  # clonal samples require major-allele WSAF on indels too
    indel_min_qual: float = 30.0
    hom_wsaf: float = 0.80
    callable_dp: int = 5
    cand_min_alt: int = 2           # candidates persisted for cohort WSAF even below call threshold
    cand_min_wsaf: float = 0.03
    amp_ratio: float = 1.5
    del_ratio: float = 0.30          # min 1 kb bin over the gene (MAPQ >= 1 depth), locally normalised
    del_window: float = 0.05         # and min 300 bp exon window (MAPQ >= 20 bases) essentially empty
    cnv_max_se: float = 0.08
    faceaway_min: int = 3            # face-away pairs in one junction cluster spanning a curated gene
    junction_tol: int = 1000         # cluster tolerance on both pair ends (gDNA)
    junction_tol_swga: int = 500     # tighter cluster tolerance on sWGA/MDA libraries
    junction_max_span: int = 100_000
    flank_lo: int = 50_000           # local normalisation flanks (both sides, same chromosome)
    flank_hi: int = 250_000
    # BAQ-style read mask (0 = off): see kernels._base_mask. Indel-flank and mismatch-cluster masks lowered F1
    mask_indel_flank: int = 0
    mask_clip_flank: int = 10
    mask_end_trim: int = 3          # dev-set sweep with clip flank 10: poly F1 0.929 -> 0.936, mono +0.0008
    mask_mm_win: int = 0
    mask_mm_n: int = 0
    dedup: bool = True              # streaming coordinate + TLEN PCR/optical duplicate filter
    use_omnitig_zoom: bool = False  # Simple Omnitigs + VarGeno QV + constrained WSAF simplex MLE


class Accumulator:
    def __init__(self, g: G.Genome, p: CallParams, zoom_mask=None, ev_cap=2_000_000):
        L = g.L
        self.g, self.p = g, p
        self.flt = np.array([p.mask_indel_flank, p.mask_clip_flank, p.mask_end_trim, p.mask_mm_win, p.mask_mm_n],
                            np.int64)
        self.counts = np.zeros((L, 4), np.uint16)
        self.delc = np.zeros(L, np.uint16)
        self.insc = np.zeros(L, np.uint16)
        self.binbases = np.zeros(int(g.bin_offsets[-1]), np.int64)
        self.ev = np.zeros((ev_cap, 4), np.int64)
        self.ev_n = np.zeros(1, np.int64)
        self.ev_keys = np.zeros((0, 4), np.int64)
        self.ev_cnt = np.zeros(0, np.int64)
        self.ccount = np.zeros(len(g.names), np.int64)
        self.prev = np.full(4, -1, np.int64)
        self.stats = np.zeros(11, np.int64)
        names = [n.encode() for n in g.names]
        self.names = np.frombuffer(b"".join(names), np.uint8)
        self.name_off = np.concatenate([[0], np.cumsum([len(n) for n in names])]).astype(np.int64)
        self.coff = g.offsets.astype(np.int64)
        self.boff = g.bin_offsets.astype(np.int64)
        self.zmask = zoom_mask if zoom_mask is not None else np.zeros(L, bool)
        self.zbuf = np.zeros(48 << 20, np.uint8)
        self.zmeta = np.zeros((400_000, 3), np.int64)
        self.zn = np.zeros(2, np.int64)
        self.rf = np.zeros((200_000, 3), np.int64)
        self.rf_n = np.zeros(1, np.int64)
        self.ins_buf = np.zeros(4 << 20, np.uint8)
        self.ins_meta = np.zeros((16_384, 3), np.int64)
        self.ins_n = np.zeros(2, np.int64)
        self.ins_seqs = {}
        self.dedup_tab = np.zeros(65_536 if p.dedup else 0, np.uint64)

    def _flush_ins_seqs(self):
        n = int(self.ins_n[0])
        for i in range(n):
            ic, off, ln = int(self.ins_meta[i, 0]), int(self.ins_meta[i, 1]), int(self.ins_meta[i, 2])
            if ic not in self.ins_seqs:
                self.ins_seqs[ic] = self.ins_buf[off:off + ln].tobytes().decode("ascii", errors="replace")
        self.ins_n[:] = 0

    def feed(self, buf, n):
        used = Kn.sam_block(buf, n, self.names, self.name_off, self.coff, self.boff, self.g.seq, self.counts,
                            self.delc, self.insc, self.binbases, self.ev, self.ev_n, self.ccount, self.prev,
                            self.stats, self.p.min_mapq, self.p.min_bq, self.p.bin_mapq, self.p.max_mm_frac,
                            self.zmask, self.zbuf, self.zmeta, self.zn, self.rf, self.rf_n, self.flt,
                            self.ins_buf, self.ins_meta, self.ins_n, self.dedup_tab)
        if self.ins_n[0] > 0:
            self._flush_ins_seqs()
        if self.ev_n[0] > 0.8 * len(self.ev):
            self.compact()
        return used

    def compact(self):
        if self.ins_n[0] > 0:
            self._flush_ins_seqs()
        n = int(self.ev_n[0])
        if n == 0:
            return
        keys = np.concatenate([self.ev_keys, self.ev[:n]])
        cnt = np.concatenate([self.ev_cnt, np.ones(n, np.int64)])
        u, inv = np.unique(keys, axis=0, return_inverse=True)
        self.ev_keys = u
        self.ev_cnt = np.bincount(inv.ravel(), weights=cnt, minlength=len(u)).astype(np.int64)
        self.ev_n[0] = 0

    ARRAYS = ("counts", "delc", "insc", "binbases", "ev_keys", "ev_cnt", "ccount", "stats", "zmeta", "zbuf", "zn", "rf")

    def dump(self, path):
        """compressed replay archive of the accumulators (re-call without network bytes)."""
        import numcodecs
        self.compact()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        codec = numcodecs.Blosc("zstd", 3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        n = int(self.zn[0])
        nb = int(self.zn[1])
        arrs = dict(counts=self.counts, delc=self.delc, insc=self.insc, binbases=self.binbases,
                    ev_keys=self.ev_keys, ev_cnt=self.ev_cnt, ccount=self.ccount, stats=self.stats,
                    zmeta=self.zmeta[:n], zbuf=self.zbuf[:nb], zn=self.zn, rf=self.rf[:int(self.rf_n[0])])
        meta = {}
        for k, a in arrs.items():
            a = np.ascontiguousarray(a)
            if a.size == 0:  # blosc cannot round-trip empty buffers
                a = np.zeros((1,) + a.shape[1:], a.dtype)
                meta[k + "_empty"] = True
            (path / f"{k}.blosc").write_bytes(codec.encode(a))
            meta[k] = [str(a.dtype), list(a.shape)] if not meta.get(k + "_empty") else [str(a.dtype), [0] + list(a.shape[1:])]
        if getattr(self, "ins_seqs", None):
            (path / "ins_seqs.json").write_text(json.dumps({str(k): v for k, v in self.ins_seqs.items()}))
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, g, p, path):
        import numcodecs
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        codec = numcodecs.Blosc()
        acc = cls.__new__(cls)
        acc.g, acc.p = g, p
        for k, v in meta.items():
            if k.endswith("_empty"):
                continue
            dt, shape = v
            if meta.get(k + "_empty"):
                setattr(acc, k, np.zeros(shape, dt))
                continue
            a = np.frombuffer(codec.decode((path / f"{k}.blosc").read_bytes()), dtype=dt).reshape(shape)
            setattr(acc, k, a.copy())
        if not hasattr(acc, "rf"):  # dumps from before face-away capture: no junction evidence
            acc.rf = np.zeros((0, 3), np.int64)
        acc.rf_n = np.array([len(acc.rf)], np.int64)
        acc.ev = np.zeros((0, 4), np.int64)
        acc.ev_n = np.zeros(1, np.int64)
        acc.ins_buf = np.zeros(0, np.uint8)
        acc.ins_meta = np.zeros((0, 3), np.int64)
        acc.ins_n = np.zeros(2, np.int64)
        ins_f = path / "ins_seqs.json"
        acc.ins_seqs = {int(k): v for k, v in json.loads(ins_f.read_text()).items()} if ins_f.exists() else {}
        acc.dedup_tab = np.zeros(0, np.uint64)
        return acc

    def depth(self):
        return self.counts.sum(1, dtype=np.int32) + self.delc

    def zoom_reads(self):
        k = min(int(self.zn[0]), len(self.zmeta))
        return self.zmeta[:k].copy(), self.zbuf[:int(self.zn[1])].copy()


# ----------------------------------------------------------------------------------------------- likelihood
def wsaf_qual(ac, rc, err):
    """phred-scaled LR of the constrained ML WSAF in [0, 1] vs the error-only (hom-ref) model."""
    ac = np.asarray(ac, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    n = np.maximum(ac + rc, 1.0)
    f = ac / n
    e = np.clip(np.asarray(err, dtype=np.float64), 1e-9, 0.70)
    # conditional on observing ref or this specific alt (total mass 1 - 2e/3), solve p(w) = f * (1 - 2e/3)
    w_mle = np.clip((f * (1.0 - 2.0 * e / 3.0) - e / 3.0) / np.maximum(1.0 - 4.0 * e / 3.0, 1e-9), 0.0, 1.0)
    pa1 = w_mle * (1.0 - e) + (1.0 - w_mle) * e / 3.0
    pr1 = (1.0 - w_mle) * (1.0 - e) + w_mle * e / 3.0
    ll1 = ac * np.log(pa1) + rc * np.log(pr1)
    ll0 = ac * np.log(e / 3.0) + rc * np.log(1.0 - e)
    return np.maximum(0.0, 10.0 * (ll1 - ll0) / LN10)


def _minor_prior(p: CallParams, clonal: bool, fws=None):
    """continuous F_WS-aware minor-allele threshold (eliminates step cliff at F_WS = 0.95)."""
    if fws is None or not np.isfinite(fws):
        return (np.inf, 1.0) if clonal else (p.snv_min_qual, p.snv_minor_wsaf)
    if fws >= 0.985:
        return np.inf, 1.0
    mix_prior = float(np.clip((0.985 - float(fws)) / 0.185, 1e-6, 1.0))
    qual_thr = p.snv_min_qual - 10.0 * np.log10(mix_prior)
    wsaf_thr = p.snv_minor_wsaf + 0.30 * (1.0 - mix_prior)
    return qual_thr, wsaf_thr


def snv_call_mask(ad_alt, ad_ref, wsaf, qual, p: CallParams, clonal, fws=None, on_panel=None):
    """major alleles always; minor alleles weighted continuously by within-host diversity (1 - F_WS) and site prior."""
    ad_alt = np.asarray(ad_alt)
    ad_ref = np.asarray(ad_ref)
    wsaf = np.asarray(wsaf)
    qual = np.asarray(qual)
    major = (ad_alt >= p.snv_min_alt) & (wsaf >= p.snv_min_wsaf) & (qual >= p.snv_min_qual)
    q_thr, w_thr = _minor_prior(p, clonal, fws=fws)
    if not np.isfinite(q_thr):
        return major
    qm = wsaf_qual(ad_alt, ad_ref, p.snv_minor_err)
    req_q = np.full(len(ad_alt), q_thr, dtype=np.float64)
    if on_panel is not None:
        req_q += np.where(np.asarray(on_panel, dtype=bool), 0.0, p.snv_novel_minor_penalty)
    return major | ((ad_alt >= p.snv_minor_alt) & (wsaf >= w_thr) & (qm >= req_q))


def call_snvs(g, acc: Accumulator, p: CallParams, clonal=False, fws=None, panel_goffs=None):
    counts = acc.counts
    ref = g.seq
    valid = ref < 4
    tot = counts.sum(1, dtype=np.int32)
    rc_all = np.zeros(g.L, np.int32)
    rc_all[valid] = counts[np.where(valid)[0], ref[valid]]
    nonref = tot - rc_all
    cand = np.where(valid & (nonref >= p.cand_min_alt))[0]
    pmask = None
    if panel_goffs is not None and len(cand):
        pm = np.zeros(g.L, bool)
        pm[np.asarray(panel_goffs, dtype=np.int64)] = True
        pmask = pm[cand]
    c = counts[cand].astype(np.int32)
    r = ref[cand]
    c_masked = c.copy()
    c_masked[np.arange(len(cand)), r] = -1
    order = np.argsort(-c_masked, axis=1)
    rows = []
    dp = tot[cand] + acc.delc[cand]
    tot_c = np.maximum(tot[cand], 1)
    for k in (0, 1):  # top two alternate bases
        alt = order[:, k]
        ac = c[np.arange(len(cand)), alt]
        rc = c[np.arange(len(cand)), r]
        # include any third non-ref allele in the denominator so triallelic sites never overstate WSAF
        other = np.maximum(tot_c - ac, 0)
        w = ac / tot_c
        keep = (ac >= p.cand_min_alt) & (alt != r) & (w >= p.cand_min_wsaf)
        q = wsaf_qual(ac[keep], other[keep], p.snv_err)
        on_p = pmask[keep] if pmask is not None else None
        called = snv_call_mask(ac[keep], other[keep], w[keep], q, p, clonal, fws=fws, on_panel=on_p)
        gt = np.where(called, np.where(w[keep] >= p.hom_wsaf, 2, 1), 0).astype(np.int8)
        rows.append(dict(goff=cand[keep], kind=np.zeros(keep.sum(), np.int8), ref=r[keep].astype(np.int64),
                         alt=alt[keep].astype(np.int64), alen=np.zeros(keep.sum(), np.int32), ad_ref=rc[keep],
                         ad_alt=ac[keep], dp=dp[keep], wsaf=w[keep].astype(np.float32), qual=q.astype(np.float32),
                         gt=gt))
    return {k: np.concatenate([x[k] for x in rows]) for k in rows[0]}


def homopolymer_len(seq, goff):
    """max homopolymer tract length flanking goff (both upstream through goff and downstream from goff+1)."""
    n = len(seq)
    out = np.ones(len(goff), np.int32)
    for k in range(len(goff)):
        g0 = int(goff[k])
        best = 1
        if 0 <= g0 < n and seq[g0] < 4:
            b = seq[g0]
            lo = g0
            while lo > 0 and seq[lo - 1] == b and g0 - lo < 60:
                lo -= 1
            hi = g0 + 1
            while hi < n and seq[hi] == b and hi - lo < 60:
                hi += 1
            best = max(best, hi - lo)
        a = g0 + 1
        if 0 <= a < n and seq[a] < 4:
            b = seq[a]
            lo = a
            while lo > 0 and seq[lo - 1] == b and a - lo < 60:
                lo -= 1
            hi = a + 1
            while hi < n and seq[hi] == b and hi - lo < 60:
                hi += 1
            best = max(best, hi - lo)
        out[k] = best
    return out


def _decode_ins(code, ln, ins_seqs=None):
    if ln > 31:
        return ins_seqs.get(int(code)) if ins_seqs else None
    s = []
    for t in range(ln):
        s.append("ACGT"[(code >> (2 * (ln - 1 - t))) & 3])
    return "".join(s)


def _fnv63(seq_str):
    code = np.uint64(1469598103934665603)
    with np.errstate(over="ignore"):
        for ch in seq_str.encode("ascii"):
            code = (code ^ np.uint64(ch)) * np.uint64(1099511628211)
    return int(code & np.uint64(0x7FFFFFFFFFFFFFFF))


def normalize_indels(g, keys, cnt, ins_seqs=None):
    """left-align deletions/insertions through repeats without crossing contig starts, merge duplicates."""
    seq = g.seq
    goff = keys[:, 0].copy()
    kind = keys[:, 1]
    ln = keys[:, 2]
    code = keys[:, 3].copy()
    cstart = g.offsets[np.searchsorted(g.offsets, goff, side="right") - 1]
    for i in range(len(goff)):
        if kind[i] == 1:  # deletion of seq[goff : goff+ln]; VCF anchor is goff - 1 >= cstart[i]
            a = max(int(goff[i]), int(cstart[i]) + 1)
            L = int(ln[i])
            while a - 1 > cstart[i] and seq[a - 1] == seq[a - 1 + L] and seq[a - 1] < 4:
                a -= 1
            goff[i] = a
        elif kind[i] == 2:  # insertion after anchor goff >= cstart[i]
            s = _decode_ins(int(code[i]), int(ln[i]), ins_seqs=ins_seqs)
            if s is None:
                continue
            a = max(int(goff[i]), int(cstart[i]))
            shifted = False
            while a > cstart[i] and seq[a] < 4 and "ACGT"[seq[a]] == s[-1]:
                s = "ACGT"[seq[a]] + s[:-1]
                a -= 1
                shifted = True
            goff[i] = a
            if ln[i] <= 31:
                v = 0
                for ch in s:
                    v = (v << 2) | "ACGT".index(ch)
                code[i] = v
            elif shifted and ins_seqs is not None:
                ic = _fnv63(s)
                ins_seqs[ic] = s
                code[i] = ic
    k2 = np.stack([goff, kind, ln, code], 1)
    u, inv = np.unique(k2, axis=0, return_inverse=True)
    return u, np.bincount(inv.ravel(), weights=cnt, minlength=len(u)).astype(np.int64)


def call_indels(g, acc: Accumulator, p: CallParams, clonal=False, fws=None):
    acc.compact()
    if len(acc.ev_keys) == 0:
        return None
    ins_seqs = getattr(acc, "ins_seqs", None)
    keys, cnt = normalize_indels(g, acc.ev_keys, acc.ev_cnt, ins_seqs=ins_seqs)
    keep = cnt >= p.cand_min_alt
    keys, cnt = keys[keep], cnt[keep]
    goff = keys[:, 0]
    kind = keys[:, 1]
    cstart = g.offsets[np.searchsorted(g.offsets, goff, side="right") - 1]
    dpos = goff
    dp = acc.counts[dpos].sum(1, dtype=np.int32) + acc.delc[dpos]
    dp = np.maximum(dp, cnt)
    anchor = np.where(kind == 1, np.maximum(goff - 1, cstart), goff)
    hp = homopolymer_len(g.seq, anchor)
    err = p.indel_err + p.indel_hp_err * np.maximum(0, hp - 4)
    err = np.minimum(err, 0.3)
    ac = cnt
    rc = np.maximum(dp - ac, 0)
    w = ac / np.maximum(dp, 1)
    keepc = w >= p.cand_min_wsaf
    q = wsaf_qual(ac, rc, err)
    is_clonal = bool(clonal or (fws is not None and np.isfinite(fws) and fws >= p.clonal_fws))
    min_w = p.indel_clonal_wsaf if is_clonal else p.indel_min_wsaf
    called = (ac >= p.indel_min_alt) & (w >= min_w) & (q >= p.indel_min_qual)
    gt = np.where(called, np.where(w >= p.hom_wsaf, 2, 1), 0).astype(np.int8)
    m = keepc
    return dict(goff=anchor[m], kind=kind[m].astype(np.int8), ref=np.zeros(m.sum(), np.int64),
                alt=keys[m, 3], alen=keys[m, 2].astype(np.int32), ad_ref=rc[m].astype(np.int32),
                ad_alt=ac[m].astype(np.int32), dp=dp[m], wsaf=w[m].astype(np.float32),
                qual=q[m].astype(np.float32), gt=gt[m], hp=hp[m])


def vcf_alleles(g, rec, ins_seqs=None):
    """(contig idx, POS, REF, ALT) strings for call dicts (SNV kind 0, DEL 1 anchor-based, INS 2 anchor-based)."""
    c, pos = g.locate(rec["goff"])
    refs, alts = [], []
    for i in range(len(rec["goff"])):
        k = rec["kind"][i]
        a = int(rec["goff"][i])
        if k == 0:
            refs.append("ACGTN"[g.seq[a]])
            alts.append("ACGTN"[int(rec["alt"][i])])
        elif k == 1:
            L = int(rec["alen"][i])
            refs.append(g.bases(a, a + L + 1))
            alts.append(g.bases(a, a + 1))
        else:
            L = int(rec["alen"][i])
            s = _decode_ins(int(rec["alt"][i]), L, ins_seqs=ins_seqs)
            refs.append(g.bases(a, a + 1))
            alts.append(g.bases(a, a + 1) + (s if s is not None else "N" * L))
    return c.astype(np.int16), pos.astype(np.int32), refs, alts


# ----------------------------------------------------------------------------------------------- WSAF / F_WS
def wsaf_at(acc: Accumulator, panel):
    goff, rb, ab = panel["goff"], panel["ref"], panel["alt"]
    c = acc.counts[goff].astype(np.int32)
    ac = c[np.arange(len(goff)), ab]
    d = c.sum(1, dtype=np.int32)
    w = np.where(d > 0, ac / np.maximum(d, 1), np.nan).astype(np.float32)
    return w, d


def fws(w, d, plaf, min_dp=10, nbins=10):
    """F_WS (Manske 2012 / moimix): regress mean within-host Hw on population Hs across 10 MAF bins."""
    ok = (d >= min_dp) & np.isfinite(w)
    if ok.sum() < 200:
        return np.nan, int(ok.sum())
    maf = np.minimum(plaf[ok], 1 - plaf[ok])
    hs = 2 * maf * (1 - maf)
    hw = 2 * w[ok] * (1 - w[ok])
    edges = np.linspace(0, 0.5, nbins + 1)
    b = np.clip(np.digitize(maf, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for k in range(nbins):
        m = b == k
        if m.sum() >= 10:
            xs.append(hs[m].mean())
            ys.append(hw[m].mean())
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 3 or (xs ** 2).sum() == 0:
        return np.nan, int(ok.sum())
    slope = (xs * ys).sum() / (xs ** 2).sum()
    return float(np.clip(1 - slope, 0, 1)), int(ok.sum())


# ----------------------------------------------------------------------------------------------- depth / CNV
def bin_depth(g, acc: Accumulator):
    nb = len(g.bin_gc)
    blen = np.full(nb, G.BIN, np.float64)
    for c in range(len(g.names)):
        last = g.bin_offsets[c + 1] - 1
        blen[last] = g.lengths[c] - G.BIN * (g.bin_offsets[c + 1] - g.bin_offsets[c] - 1)
    return acc.binbases / blen


def gc_normalize(g, depth):
    """ratio = depth / expected(GC), expected from the median of core bins in 1% GC strata (smoothed)."""
    core = (g.bin_core >= 0.9) & (depth > 0)
    gc = g.bin_gc
    ratio = np.full(len(depth), np.nan)
    if core.sum() < 100:
        return ratio, np.nan
    med_all = np.median(depth[core])
    strata = np.clip((gc * 100).astype(int), 0, 60)
    exp = np.full(61, np.nan)
    for s in range(61):
        m = core & (strata == s)
        if m.sum() >= 30:
            exp[s] = np.median(depth[m])
    good = np.isfinite(exp)
    xs = np.arange(61)
    if good.sum() >= 2:
        exp = np.interp(xs, xs[good], exp[good])
    else:
        exp[:] = med_all
    sm = exp.copy()
    for s in range(61):
        sm[s] = np.mean(exp[max(0, s - 1):s + 2])
    e = sm[strata]
    ratio = np.where(e > 0, depth / e, np.nan)
    norm = np.nanmedian(ratio[core])
    return ratio / norm, float(med_all)


def locus_ratio(g, ratio, keys):
    vals = []
    for k in keys:
        gg = g.gene(k)
        b0, b1 = g.bin_range(gg["contig"], gg["start"], gg["end"])
        vals.append(ratio[b0:b1])
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.inf, 0
    se = 1.2533 * (np.std(v) / np.sqrt(len(v))) if len(v) > 1 else np.inf
    return float(np.median(v)), float(se), int(len(v))


def exon_ratio(g, acc: Accumulator, key, genome_median):
    """per-base depth over the largest exon (hrp2/hrp3 exon 2) / genome core median depth."""
    gg = g.gene(key)
    ex = max(gg["exons"] or gg["cds"], key=lambda e: e[1] - e[0])
    a, b = g.goff(gg["contig"], ex[0]), g.goff(gg["contig"], ex[1]) + 1
    d = acc.counts[a:b].sum(1) + acc.delc[a:b]
    if not genome_median or genome_median <= 0:
        return np.nan, np.inf
    r = float(np.mean(d) / genome_median)
    se = float(np.sqrt(max(np.mean(d), 1) / max(len(d) / 150.0, 1)) / genome_median)
    return r, se


def cnv_calls(g, acc: Accumulator, p: CallParams, core_median_depth):
    depth = bin_depth(g, acc)
    ratio, bin_median = gc_normalize(g, depth)
    bad_med = (not np.isfinite(bin_median)) or bin_median < 5
    bad_core = (core_median_depth is None) or (not np.isfinite(core_median_depth)) or core_median_depth < 5
    out = []
    for name, keys in G.CNV_LOCI.items():
        r, se, nb = locus_ratio(g, ratio, keys)
        call = -1 if (not np.isfinite(r) or se > p.cnv_max_se or bad_med) else int(r >= p.amp_ratio)
        out.append(dict(locus=name, kind="amp", ratio=r, se=se, n_bins=nb, call=call, method="coverage"))
    for name, key in G.DEL_LOCI.items():
        r_bin, se_bin, nb = locus_ratio(g, ratio, [key])
        r_ex, se_ex = exon_ratio(g, acc, key, core_median_depth)
        r = r_ex if np.isfinite(r_ex) else r_bin
        call = -1 if (not np.isfinite(r) or bad_core) else int(r < p.del_ratio)
        out.append(dict(locus=name, kind="del", ratio=r, se=se_ex, n_bins=nb, call=call, method="coverage",
                        ratio_bins=r_bin))
    return out, depth, ratio, bin_median


def callable_runs(g, acc: Accumulator, p: CallParams):
    d = acc.depth()
    return Kn.runs_ge(d, p.callable_dp, g.offsets.astype(np.int64))


def codon_call(g, acc: Accumulator, gene, codon, min_frac=0.10, min_alt=2):
    """amino acids present at a codon from per-base counts (major plus minor alleles >= min_frac)."""
    gs, strand = g.codon_goffs(gene, codon)
    alle = []
    for x in gs:
        c = acc.counts[x].astype(np.int64)
        tot = c.sum()
        if tot < 5:
            return None
        major = int(np.argmax(c))
        opts = [(major, c[major])]
        for b in range(4):
            if b != major and c[b] >= min_alt and c[b] / tot >= min_frac:
                opts.append((b, c[b]))
        alle.append(opts)
    aas = set()
    major_bases = ["ACGT"[o[0][0]] for o in alle]
    aas.add(g.codon_aa(gene, codon, major_bases))
    for k in range(3):
        for b, _ in alle[k][1:]:
            bb = list(major_bases)
            bb[k] = "ACGT"[b]
            aas.add(g.codon_aa(gene, codon, bb))
    return aas


AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His",
       "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
       "Y": "Tyr", "V": "Val", "*": "Ter"}


def k13_ns_changes(g, acc, lo=349, hi=726, min_frac=0.10):
    """Pf8-style kelch13 propeller summary: non-synonymous changes in codons 349-726, upper case when the
    mutant is the only allele (hom), lower case when mixed with the reference; 'None' when wild type."""
    out = []
    missing = 0
    for codon in range(lo, hi + 1):
        gs, _ = g.codon_goffs("kelch13", codon)
        ref = g.codon_aa("kelch13", codon, [g.bases(x, x + 1) for x in gs])
        aas = codon_call(g, acc, "kelch13", codon, min_frac=min_frac)
        if aas is None:
            missing += 1
            continue
        for aa in sorted(aas - {ref}):
            tag = f"{ref}{codon}{aa}"
            out.append(tag if ref not in aas else tag.lower())
    if missing > (hi - lo + 1) * 0.2:
        return "-"
    return ",".join(out) if out else "None"



# ----------------------------------------------------------------------------------------------- v2: p-aware, local
def p_corrected_depth(g, acc, bin_p=None):
    """1 kb bin depth divided by the sampling rate in force when the stream covered each bin."""
    depth = bin_depth(g, acc)
    if bin_p is None:
        return depth
    bp = np.where(np.isfinite(bin_p) & (bin_p > 0), bin_p, np.nanmedian(bin_p) if np.isfinite(bin_p).any() else 1.0)
    return depth / bp


def local_ratio(g, ratio, contig, start, end, p: CallParams):
    """median GC-normalised ratio over [start,end] divided by the median of same-contig flanks 50-250 kb away."""
    c0, c1 = g.bin_range(contig, 1, int(g.lengths[g.cidx(contig)]))
    b0, b1 = g.bin_range(contig, start, end)
    fl = []
    for a, b in ((start - p.flank_hi, start - p.flank_lo), (end + p.flank_lo, end + p.flank_hi)):
        a = max(1, a)
        b = min(int(g.lengths[g.cidx(contig)]), b)
        if b > a:
            x0, x1 = g.bin_range(contig, a, b)
            fl.append(ratio[x0:x1][g.bin_core[x0:x1] >= 0.5])
    fl = np.concatenate(fl) if fl else np.zeros(0)
    fl = fl[np.isfinite(fl) & (fl > 0)]
    v = ratio[b0:b1]
    v = v[np.isfinite(v)]
    if len(fl) < 20 or len(v) == 0:
        return np.nan, len(v)
    return float(np.median(v) / np.median(fl)), len(v)


def hrp_window_ratio(g, acc, key, bin_p, p: CallParams, win=300):
    """minimum 300 bp window depth over the largest exon relative to the local flank depth (p-corrected)."""
    gg = g.gene(key)
    ex = max(gg["exons"] or gg["cds"], key=lambda e: e[1] - e[0])
    a, b = g.goff(gg["contig"], ex[0]), g.goff(gg["contig"], ex[1]) + 1
    d = (acc.counts[a:b].sum(1) + acc.delc[a:b]).astype(np.float64)
    bins = g.bin_of(np.arange(a, b))
    if bin_p is not None:
        bp = bin_p[bins]
        d = d / np.where(np.isfinite(bp) & (bp > 0), bp, 1.0)
    depth = p_corrected_depth(g, acc, bin_p)
    flank = []
    L = int(g.lengths[g.cidx(gg["contig"])])
    for x0, x1 in ((ex[0] - p.flank_hi, ex[0] - p.flank_lo), (ex[1] + p.flank_lo, ex[1] + p.flank_hi)):
        x0, x1 = max(1, x0), min(L, x1)
        if x1 > x0:
            c0, c1 = g.bin_range(gg["contig"], x0, x1)
            seg = depth[c0:c1][g.bin_core[c0:c1] >= 0.5]
            flank.append(seg[seg > 0])
    flank = np.concatenate(flank) if flank else np.zeros(0)
    if len(flank) < 20 or len(d) < win:
        return np.nan, np.nan, np.nan
    ref = float(np.median(flank))
    cs = np.concatenate([[0], np.cumsum(d)])
    wins = (cs[win:] - cs[:-win]) / win
    return float(wins.min() / ref), float(d.mean() / ref), ref


def junction_support(sel, genes, p: CallParams, tol=None):
    """largest cluster of face-away pairs (both ends within tol) whose span covers a gene of the locus."""
    tol = p.junction_tol if tol is None else tol
    best = 0
    for x in genes:
        m = (sel[:, 1] <= x["start"] + 200) & (sel[:, 2] >= x["end"] - 200) & \
            (sel[:, 2] - sel[:, 1] <= p.junction_max_span) if len(sel) else np.zeros(0, bool)
        pts = sel[m][:, 1:] if len(sel) else np.zeros((0, 2))
        if len(pts) == 0:
            continue
        pts = pts[np.argsort(pts[:, 0])]
        for i in range(len(pts)):
            near = (np.abs(pts[:, 0] - pts[i, 0]) <= tol) & (np.abs(pts[:, 1] - pts[i, 1]) <= tol)
            best = max(best, int(near.sum()))
    return best


def _normalize_sample_type(sample_type) -> str:
    raw = "gDNA" if sample_type is None else str(sample_type).strip()
    if not raw:
        return "gdna"
    low = raw.lower()
    if low not in ("gdna", "swga", "mda"):
        raise ValueError(f"Unsupported sample_type {sample_type!r} (expected gDNA, sWGA, or MDA)")
    return low


def cnv_calls_v2(g, acc, p: CallParams, bin_p=None, sample_type="gDNA"):
    """CNV amplifications and hrp2/hrp3 deletions with p-corrected, GC-normalised, locally normalised depth.

    Amplification: a junction cluster spanning a curated gene (with chimeric-background scaling on sWGA/MDA),
    or (gDNA libraries only) a local coverage ratio >= 1.5."""
    stype_low = _normalize_sample_type(sample_type)
    gdna = (stype_low == "gdna")
    depth = p_corrected_depth(g, acc, bin_p)
    ratio, bin_median = gc_normalize(g, depth)
    bad_med = (not np.isfinite(bin_median)) or bin_median < 5
    out = []
    rf = getattr(acc, "rf", np.zeros((0, 3), np.int64))
    if hasattr(acc, "rf_n"):
        rf = acc.rf[:int(acc.rf_n[0])]
    tol = p.junction_tol if gdna else p.junction_tol_swga
    # phi29 MDA / sWGA libraries produce genome-wide random face-away chimeras; scale the junction floor above background
    fa_floor = p.faceaway_min if gdna else max(p.faceaway_min, int(p.faceaway_min + 0.02 * max(0, len(rf) - 20)))
    for name, keys in G.CNV_LOCI.items():
        genes = [g.gene(k) for k in keys]
        contig = genes[0]["contig"]
        r, nb = local_ratio(g, ratio, contig, min(x["start"] for x in genes), max(x["end"] for x in genes), p)
        ci = g.cidx(contig)
        sel = rf[rf[:, 0] == ci] if len(rf) else rf
        fa = junction_support(sel, genes, p, tol=tol)
        cov = bool(gdna and np.isfinite(r) and r >= p.amp_ratio)
        call = -1 if (bad_med or (gdna and not np.isfinite(r) and fa == 0)) else int(cov or fa >= fa_floor)
        out.append(dict(locus=name, kind="amp", ratio=r, se=np.nan, n_bins=nb, call=call,
                        method=("coverage+junction" if gdna else "junction_only") + "_v4", faceaway_pairs=fa))
    for name, key in G.DEL_LOCI.items():
        # MAPQ >= 1 bins: the hrp2 / hrp3 repeat region gets MAPQ 0 from any aligner, so MAPQ >= 20 base counts
        # look 'deleted' in intact genes; a real (telomere-healing) deletion empties at least one gene bin
        gg = g.gene(key)
        b0, b1 = g.bin_range(gg["contig"], gg["start"], gg["end"])
        rmin, nb = np.nan, b1 - b0
        rgene, _ = local_ratio(g, ratio, gg["contig"], gg["start"], gg["end"], p)
        vals = []
        for b in range(b0, b1):
            pos1 = int((b - g.bin_offsets[g.cidx(gg["contig"])]) * G.BIN + 1)
            r1, _ = local_ratio(g, ratio, gg["contig"], pos1, pos1 + G.BIN - 1, p)
            vals.append(r1)
        vals = [v for v in vals if np.isfinite(v)]
        rmin = float(min(vals)) if vals else np.nan
        wmin, _, wref = hrp_window_ratio(g, acc, key, bin_p, p)
        # hrp2 / hrp3 deletions are terminal (telomere healing, chromosome breakage): the telomere-side flank goes too,
        # while a divergent but present gene (repeat polymorphism the aligner cannot place) leaves the flank intact
        L = int(g.lengths[g.cidx(gg["contig"])])
        if L - gg["end"] < gg["start"]:
            t0_, t1_ = gg["end"] + 2000, min(L, gg["end"] + 20000)
        else:
            t0_, t1_ = max(1, gg["start"] - 20000), gg["start"] - 2000
        rtel, _ = local_ratio(g, ratio, gg["contig"], t0_, t1_, p)
        # gDNA: a MAPQ >= 1 gene bin mostly empty AND a 300 bp exon window (MAPQ >= 20 bases) essentially empty.
        # sWGA / MDA coverage is too uneven for deletion calls (Pf8 leaves 77 of 79 such panel isolates undetermined):
        # no call. The telomere-side ratio is reported, not required (it cost 9 true gDNA hrp3 deletions).
        if not gdna:
            call = -1
        else:
            call = -1 if (not np.isfinite(rmin) or not np.isfinite(wmin) or bad_med) else \
                int(rmin < p.del_ratio and wmin < p.del_window)
        out.append(dict(locus=name, kind="del", ratio=rmin, se=np.nan, n_bins=nb, call=call,
                        method="bin_and_window_v5" if gdna else "no_call_amplified_library",
                        ratio_bins=rgene, ratio_window=wmin, ratio_telomere=rtel))
    return out, depth, ratio, bin_median


def read_offsets(g, zmeta, zbuf, max_shift=40):
    """align each captured read to the best reference diagonal near its start (soft clips shift the start)."""
    if len(zmeta) == 0:
        return []
    start, ident = Kn.best_diagonals(g.seq, np.ascontiguousarray(zmeta), zbuf, max_shift)
    out = []
    for (g0, o, ln), st, idt in zip(zmeta, start, ident):
        out.append((int(st), Kn.ENC[zbuf[o:o + ln]], float(idt)))
    return out


def codon_call_reads(g, aligned, gene, codon, min_frac=0.10, min_reads=2, flank=12, max_flank_mm=4):
    """amino acids at a codon counted from whole reads (phased: multi-base changes stay on one haplotype)."""
    gs, strand = g.codon_goffs(gene, codon)
    lo, hi = min(gs), max(gs)
    cnt = {}
    n = 0
    for start, rc, ident in aligned:
        if ident < 0.85:
            continue
        i0 = lo - flank - start
        i1 = hi + flank + 1 - start
        if i0 < 0 or i1 > len(rc):
            continue
        seg = rc[i0:i1]
        refseg = g.seq[lo - flank:hi + flank + 1]
        fl = np.r_[0:flank, len(seg) - flank:len(seg)]
        if (seg[fl] != refseg[fl]).sum() > max_flank_mm or (seg > 3).any():
            continue  # neighbouring codons may carry the haplotype's own changes (crt CVIET): tolerate a few
        bases = ["ACGT"[seg[x - lo + flank]] for x in gs]
        aa = g.codon_aa(gene, codon, bases)
        cnt[aa] = cnt.get(aa, 0) + 1
        n += 1
    if n < 5:
        return None, n
    return {aa for aa, c in cnt.items() if c >= min_reads and c / n >= min_frac}, n



def haplotype_call_reads(g, aligned, gene, codons_list, min_frac=0.10, min_reads=2, flank=8, max_flank_mm=4):
    """phased multi-codon haplotype calls (e.g. crt 72-76 CVMNK/CVIET, arps10 127-128 VD/VH/MH) from single reads."""
    codon_gs = [g.codon_goffs(gene, c)[0] for c in codons_list]
    all_gs = [x for gs in codon_gs for x in gs]
    lo, hi = min(all_gs), max(all_gs)
    cnt = {}
    n = 0
    for start, rc, ident in aligned:
        if ident < 0.85:
            continue
        i0 = lo - flank - start
        i1 = hi + flank + 1 - start
        if i0 < 0 or i1 > len(rc):
            continue
        seg = rc[i0:i1]
        if (seg > 3).any():
            continue
        refseg = g.seq[lo - flank:hi + flank + 1]
        fl = np.r_[0:flank, len(seg) - flank:len(seg)]
        if (seg[fl] != refseg[fl]).sum() > max_flank_mm:
            continue
        hap = "".join(g.codon_aa(gene, c, ["ACGT"[seg[x - lo + flank]] for x in gs])
                      for c, gs in zip(codons_list, codon_gs))
        cnt[hap] = cnt.get(hap, 0) + 1
        n += 1
    if n < 5:
        return None
    return {h for h, c in cnt.items() if c >= min_reads and c / n >= min_frac}


def _fallback_haplotype(codons, gene, codons_list):
    """assemble multi-codon haplotype from per-codon calls when at most one codon is mixed."""
    vals = [codons.get(f"{gene}_{c}") for c in codons_list]
    if any(v is None for v in vals):
        return None
    sets = [v.split(",") for v in vals]
    if sum(len(s) > 1 for s in sets) > 1:
        return None
    out = [""]
    for s in sets:
        out = [p + a for p in out for a in s]
    return set(out)


def curated_calls(g, acc, drm_codons, bin_p=None, p: CallParams = None, sample_type="gDNA"):
    """v2 curated calls: phased codons + multi-codon haplotypes from captured reads (per-base fallback),
    kelch13 349-726 scan, p-aware locally normalised CNV / hrp."""
    p = p or CallParams()
    zm, zb = acc.zoom_reads()
    aligned = read_offsets(g, zm, zb) if len(zm) else []
    zoom = set(G.ZOOM_GENES)
    Z = None
    if p.use_omnitig_zoom and len(zm):
        from . import kmer_zoom as Z
    codons = {}
    for gene, codon in drm_codons:
        aa = None
        if gene in zoom and aligned:
            aa, _ = codon_call_reads(g, aligned, gene, codon)
        if Z is not None and gene in zoom:
            omni_aa = Z.omnitig_codon_calls(g, zm, zb, gene, [codon], min_wsaf=0.05)
            if omni_aa:
                aa = (aa | omni_aa) if aa is not None else omni_aa
        if aa is None:
            aa = codon_call(g, acc, gene, codon)
        codons[f"{gene}_{codon}"] = ",".join(sorted(aa)) if aa else None
    for gene, tag, clist in (("crt", "crt_72_76", [72, 73, 74, 75, 76]), ("arps10", "arps10_127_128", [127, 128])):
        haps = Z.omnitig_codon_calls(g, zm, zb, gene, clist, min_wsaf=0.05) if (Z is not None and gene in zoom) else None
        if not haps:
            haps = haplotype_call_reads(g, aligned, gene, clist) if (gene in zoom and aligned) else None
        if not haps:
            haps = _fallback_haplotype(codons, gene, clist)
        codons[tag] = ",".join(sorted(haps)) if haps else None
    # kelch13 propeller: per-base screen, phased confirmation of codons with any alternate evidence
    out, missing = [], 0
    for codon in range(349, 727):
        gs, _ = g.codon_goffs("kelch13", codon)
        ref = g.codon_aa("kelch13", codon, [g.bases(x, x + 1) for x in gs])
        aa = codon_call(g, acc, "kelch13", codon)
        if aa is None:
            missing += 1
            continue
        has_alt_cand = (aa != {ref})
        if Z is not None and not has_alt_cand:
            # Check raw non-ref pileup counts (>= 2 reads) to rescue minor clones >= 5% WSAF via omnitig assembly.
            for x in gs:
                rb = g.seq[x]
                if rb < 4 and (int(acc.counts[x, :4].sum()) - int(acc.counts[x, rb])) >= 2:
                    has_alt_cand = True
                    break
        if has_alt_cand and aligned:
            ph, _ = codon_call_reads(g, aligned, "kelch13", codon)
            if Z is not None:
                omni_ph = Z.omnitig_codon_calls(g, zm, zb, "kelch13", [codon], min_wsaf=0.05)
                if omni_ph:
                    ph = (ph | omni_ph) if ph is not None else omni_ph
            aa = ph if ph is not None else aa
        for x in sorted(aa - {ref}):
            tag = f"{ref}{codon}{x}"
            out.append(tag if ref not in aa else tag.lower())
    codons["k13_missing_codons"] = int(missing)
    k13 = "-" if missing > 378 * 0.2 else (",".join(out) if out else "None")
    cnv, depth, ratio, bmed = cnv_calls_v2(g, acc, p, bin_p, sample_type)
    return codons, k13, cnv, depth, ratio, bmed
