"""Startup empirical kernel micro-tournament (0 LLM calls).

Samples real reads from the study plus synthetic ground-truth reads generated
from the study's own catalog in six regimes (the sixth only for intron-bearing catalogs):

  normal_0.5pct_err, divergent_3pct_SNP, adapter_overhang (TruSeq and Nextera, alternating),
  short_edge_or_splice_junction, paralog_or_strain_tie,
  unannotated_exon_skip_junction (intron-bearing catalogs only: exon_i spliced to exon_i+2 where
  that junction is absent from every annotated isoform of the gene)

and scores every available kernel on real mapping rate, synthetic recall,
throughput and the **mate-tie symmetry gate**: on reads that are genuinely tied
between two targets (identical sequence in two strains or paralogs) a kernel
must report both candidates.  A kernel whose paired-end rescue silently breaks
the tie (as strobealign's does) leaks Pass-2 strain/paralog weights, so it is
penalised whenever multi-assembly or paralog conditioning is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import re
import time

from .catalog import Catalog, Target, revcomp
from .kernels import Kernel, MapResult, Read

EXON_SKIP = "unannotated_exon_skip_junction"
REGIMES = ("normal_0.5pct_err", "divergent_3pct_SNP", "adapter_overhang", "short_edge_or_splice_junction", "paralog_or_strain_tie", EXON_SKIP)
CORE_RECALL_REGIMES = ("normal_0.5pct_err", "divergent_3pct_SNP", "adapter_overhang", "short_edge_or_splice_junction")
ADAPTER_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCACATCACGATCTCGTATGCCGTCTTCTGCTTG" * 3
ADAPTER_R2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCTCGGTGGTCGCCGTATCATTAAAAAA" * 3
NEXTERA_R1 = "CTGTCTCTTATACACATCTCCGAGCCCACGAGACTAAGGCGAATCTCGTATGCCGTCTTCTGCTTG" * 3
NEXTERA_R2 = "CTGTCTCTTATACACATCTGACGCTGCCGACGAGCGATCTAGTGTAGATCTCGGTGGTCGCCGTATCATT" * 3
ADAPTER_KITS = (("truseq", ADAPTER_R1, ADAPTER_R2), ("nextera", NEXTERA_R1, NEXTERA_R2))
MIN_JUNCTION_ANCHOR = 15


@dataclass
class SyntheticCase:
    regime: str
    truth: frozenset[str]          # acceptable target ids (the true target's group members)
    r1: str
    r2: str | None
    tie_pair: tuple[str, str] | None = None


def _mutate(rng: random.Random, s: str, rate: float) -> str:
    if rate <= 0:
        return s
    b = list(s)
    for i, ch in enumerate(b):
        if rng.random() < rate:
            b[i] = rng.choice([x for x in "ACGT" if x != ch])
    return "".join(b)


def _make_pair(frag: str, read_len: int, layout: str, orientation: str = "reverse") -> tuple[str, str | None]:
    """Generate synthetic read (pair) in dUTP ('reverse', ISR/SR) or 'forward' (ISF/SF) orientation."""
    if orientation == "forward":
        r1 = frag[:read_len]
        if layout == "SE":
            return r1, None
        r2 = revcomp(frag[-read_len:]) if len(frag) >= read_len else revcomp(frag)
        return r1, r2
    r1 = revcomp(frag[-read_len:]) if len(frag) >= read_len else revcomp(frag)
    if layout == "SE":
        return r1, None
    r2 = frag[:read_len]
    return r1, r2


def find_tie_pairs(catalog: Catalog, window: int, max_pairs: int = 400, seed: int = 7, tmp_dir: str | None = None) -> list[tuple[str, str, int, int]]:
    """Pairs of targets (different group or different assembly) sharing an identical ``window``-bp block.

    Every gene target is cut into ``window``-bp chunks (step window/2) that are mapped
    back onto the gene set with libminimap2 (short-read preset, both strands); a chunk
    that aligns end-to-end with zero mismatches to a *different* target marks a tie.
    Returns (target_a, target_b, offset_a, offset_b); offset_b is in target_b's forward
    coordinates (the block may lie on its reverse strand).
    """
    import tempfile

    import mappy

    genes = [t for t in catalog.gene_targets if t.length >= window + 2]
    if len(genes) < 2:
        return []
    if len(genes) > 5000:
        return _find_tie_pairs_seed_hash(catalog, genes, window, max_pairs=max_pairs, seed=seed)
    tg = catalog.targets
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        fa = f"{td}/self.fna"
        with open(fa, "w") as fh:
            for t in genes:
                fh.write(f">{t.target_id}\n{t.seq}\n")
        idx = mappy.Aligner(fa, preset="sr", best_n=20)
        pairs: list[tuple[str, str, int, int]] = []
        seen: set[tuple[str, str]] = set()
        order = list(genes)
        rng.shuffle(order)
        step = max(1, window // 2)
        tbuf = mappy.ThreadBuffer()
        for A in order:
            for off in range(0, A.length - window + 1, step):
                chunk = A.seq[off:off + window]
                for h in idx.map(chunk, buf=tbuf):
                    if h.ctg == A.target_id or h.NM != 0 or h.mlen < window:
                        continue
                    B = tg.get(h.ctg)
                    if B is None or (A.group_id == B.group_id and A.assembly_id == B.assembly_id):
                        continue
                    key = (A.target_id, B.target_id) if A.target_id < B.target_id else (B.target_id, A.target_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((A.target_id, B.target_id, off, h.r_st))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def _find_tie_pairs_seed_hash(
    catalog: Catalog,
    genes: list[Target],
    window: int,
    max_pairs: int = 400,
    seed: int = 7,
    max_genes: int = 12_000,
) -> list[tuple[str, str, int, int]]:
    """Exact O(N) zero-mismatch window tie finder for large eukaryotic catalogs (0 minimap2 index build)."""
    rng = random.Random(seed)
    order = list(genes)
    rng.shuffle(order)
    order = order[:max_genes]
    anchor_k = min(31, window)
    step = max(1, window // 2)
    seen_seeds: dict[str, list[tuple[Target, int, bool]]] = {}
    pairs: list[tuple[str, str, int, int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for A in order:
        seq = A.seq
        for off in range(0, A.length - window + 1, step):
            blk = seq[off:off + window]
            if "N" in blk:
                continue
            rc_blk = revcomp(blk)
            canon_is_rc = rc_blk < blk
            key_seed = (rc_blk if canon_is_rc else blk)[:anchor_k]
            hits = seen_seeds.get(key_seed)
            if hits:
                for B, b_off, b_rc in hits:
                    if B.target_id == A.target_id or (A.group_id == B.group_id and A.assembly_id == B.assembly_id):
                        continue
                    pkey = (A.target_id, B.target_id) if A.target_id < B.target_id else (B.target_id, A.target_id)
                    if pkey in seen_pairs:
                        continue
                    b_blk = B.seq[b_off:b_off + window]
                    if b_blk == blk or b_blk == rc_blk:
                        seen_pairs.add(pkey)
                        pairs.append((A.target_id, B.target_id, off, b_off))
                        if len(pairs) >= max_pairs:
                            return pairs
            if hits is None:
                seen_seeds[key_seed] = [(A, off, canon_is_rc)]
            elif len(hits) < 8:
                hits.append((A, off, canon_is_rc))
    return pairs


def default_frag_len(read_len: int) -> int:
    return max(read_len + 40, min(2 * read_len + 20, 320))


def estimate_insert_size(r1: list[Read], r2: list[Read] | None, n: int = 4000, seed_len: int = 20, max_mm_frac: float = 0.05) -> dict:
    """Empirical insert size from mate overlap alone (no index, no alignment).

    For a fragment shorter than the two mates combined, the reverse complement of one mate starts inside the
    other.  A ``seed_len`` exact seed locates that offset and the overlap is accepted when it has at most
    ``max_mm_frac`` mismatches.  Both mate orientations are tried, so stranded and unstranded libraries work.
    Fragments longer than ``2 * read_len - seed_len`` do not overlap; their share is reported, and the median is
    only called ``measurable`` when at least half of the pairs overlap.
    """
    if r2 is None:
        return {"layout": "SE", "measurable": False}
    sizes: list[int] = []
    m = min(n, len(r1), len(r2))
    readthrough = 0
    for i in range(m):
        a, b = r1[i][0], r2[i][0]
        best = None
        for x, y in ((b, revcomp(a)), (a, revcomp(b))):
            if len(y) < seed_len:
                continue
            p = x.find(y[:seed_len])
            if p < 0:
                continue
            o = min(len(x) - p, len(y))
            mm = sum(1 for u, v in zip(x[p:p + o], y[:o]) if u != v)
            if mm <= max_mm_frac * o:
                best = p + len(y)
                break
        if best is None:
            # mate starting inside its partner's reverse complement: read-through into adapter (insert < read length)
            for x, y in ((revcomp(a), b), (revcomp(b), a)):
                if y.find(x[-seed_len:]) >= 0 and y.find(x[-seed_len:]) + seed_len < len(y) - 5:
                    readthrough += 1
                    break
            continue
        sizes.append(best)
    sizes.sort()
    frac = len(sizes) / max(1, m)
    out = {"layout": "PE", "n_pairs": m, "overlap_frac": round(frac, 4), "readthrough_frac": round(readthrough / max(1, m), 4),
           "read_len": int(sorted(len(s) for s, _ in r1[:m])[m // 2]) if m else 0}
    if sizes:
        out.update(median_overlapping=sizes[len(sizes) // 2], p10_overlapping=sizes[len(sizes) // 10], p90_overlapping=sizes[(9 * len(sizes)) // 10])
    out["measurable"] = frac >= 0.5
    out["median_insert"] = sizes[len(sizes) // 2] if out["measurable"] else None
    out["lower_bound_insert_if_not_measurable"] = None if out["measurable"] else 2 * out["read_len"] - seed_len
    return out


def _introns(exons: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ivs = sorted(exons)
    return {(e1 + 1, s2 - 1) for (_s1, e1), (s2, _e2) in zip(ivs, ivs[1:]) if s2 > e1 + 1}


def exon_skip_candidates(catalog: Catalog) -> list[tuple[Target, int]]:
    """(transcript, j) where genomic exon j spliced to exon j+2 forms an intron absent from every annotated isoform of the gene."""
    annotated: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for t in catalog.gene_targets:
        if len(t.exons) >= 2:
            annotated.setdefault((t.assembly_id, t.gene_id), set()).update(_introns(t.exons))
    out: list[tuple[Target, int]] = []
    for t in catalog.gene_targets:
        ivs = sorted(t.exons)
        if len(ivs) < 3 or sum(e - s + 1 for s, e in ivs) != t.length:
            continue
        known = annotated.get((t.assembly_id, t.gene_id), set())
        for j in range(len(ivs) - 2):
            novel = (ivs[j][1] + 1, ivs[j + 2][0] - 1)
            if novel not in known:
                out.append((t, j))
    return out


def exon_skip_fragment(t: Target, j: int, frag_len: int, read_len: int, layout: str, rng: random.Random) -> str | None:
    """Fragment across the exon_j|exon_j+2 junction with >= MIN_JUNCTION_ANCHOR bp on each side inside one read."""
    ivs = sorted(t.exons)
    lens = [e - s + 1 for s, e in ivs]
    m = j + 1                                           # skipped exon (genomic order)
    a = sum(lens[:m]) if t.strand != "-" else sum(lens[m + 1:])
    skipped = t.seq[:a] + t.seq[a + lens[m]:]
    F = frag_len
    rl = min(read_len, F)
    if len(skipped) < F:
        return None
    windows = [(F - rl + MIN_JUNCTION_ANCHOR, F - MIN_JUNCTION_ANCHOR)]          # inside R1 = revcomp(frag[-rl:])
    if layout == "PE":
        windows.append((MIN_JUNCTION_ANCHOR, rl - MIN_JUNCTION_ANCHOR))          # inside R2 = frag[:rl]
    feasible = []
    for lo, hi in windows:
        d_lo = max(lo, a - (len(skipped) - F))     # st = a - d must lie in [0, len(skipped) - F]
        d_hi = min(hi, a)
        if d_lo <= d_hi:
            feasible.append((d_lo, d_hi))
    if not feasible:
        return None
    d_lo, d_hi = rng.choice(feasible)
    st = a - rng.randint(d_lo, d_hi)
    return skipped[st:st + F]


def build_synthetic_truth(
    catalog: Catalog,
    layout: str,
    read_len: int,
    n_per_regime: int = 600,
    seed: int = 42,
    frag_len: int | None = None,
    orientation: str = "reverse",
) -> list[SyntheticCase]:
    rng = random.Random(seed)
    frag_len = int(frag_len) if frag_len else default_frag_len(read_len)
    genes = [t for t in catalog.gene_targets if t.length >= frag_len + 20 and t.assembly_id == catalog.primary_assembly]
    if not genes:
        genes = [t for t in catalog.gene_targets if t.length >= frag_len + 20]
    truth_of = {t.target_id: frozenset(x.target_id for x in catalog.groups.get(t.group_id, [t])) for t in genes}
    cases: list[SyntheticCase] = []

    def emit(regime: str, t: Target, frag: str, rl: int = read_len) -> None:
        r1, r2 = _make_pair(frag, rl, layout, orientation=orientation)
        cases.append(SyntheticCase(regime, truth_of.get(t.target_id, frozenset([t.target_id])), r1, r2))

    for _ in range(n_per_regime):
        t = rng.choice(genes)
        st = rng.randint(0, t.length - frag_len)
        emit("normal_0.5pct_err", t, _mutate(rng, t.seq[st:st + frag_len], 0.005))
    for _ in range(n_per_regime):
        t = rng.choice(genes)
        st = rng.randint(0, t.length - frag_len)
        emit("divergent_3pct_SNP", t, _mutate(rng, t.seq[st:st + frag_len], 0.03))
    for i in range(n_per_regime):
        t = rng.choice(genes)
        insert = max(40, min(read_len - 30, int(read_len * 0.6)))
        st = rng.randint(0, t.length - insert)
        frag = _mutate(rng, t.seq[st:st + insert], 0.005)
        _kit, ad1, ad2 = ADAPTER_KITS[i % 2]          # TruSeq on even cases, Nextera on odd cases
        if orientation == "forward":
            r1 = (frag + ad1)[:read_len]
            r2 = (revcomp(frag) + ad2)[:read_len] if layout == "PE" else None
        else:
            r1 = (revcomp(frag) + ad1)[:read_len]
            r2 = (frag + ad2)[:read_len] if layout == "PE" else None
        cases.append(SyntheticCase("adapter_overhang", truth_of[t.target_id], r1, r2))
    # short edge / splice junction
    spliced = [t for t in genes if len(t.exons) >= 2] if catalog.mode == "eukaryotic" else []
    short_len = max(36, min(55, read_len))
    for _ in range(n_per_regime):
        if spliced:
            t = rng.choice(spliced)
            # cDNA junction positions: cumulative exon lengths (exons are genomic-ordered; reverse for '-')
            ex_lens = [e - s + 1 for s, e in t.exons]
            if t.strand == "-":
                ex_lens = ex_lens[::-1]
            cuts = []
            acc = 0
            for L in ex_lens[:-1]:
                acc += L
                cuts.append(acc)
            j = rng.choice(cuts)
            st = max(0, min(t.length - frag_len, j - rng.randint(15, max(16, frag_len - 15))))
            emit("short_edge_or_splice_junction", t, _mutate(rng, t.seq[st:st + frag_len], 0.005))
        else:
            t = rng.choice(genes)
            edge = rng.random() < 0.5
            seg = t.seq[:short_len * 2] if edge else t.seq[-short_len * 2:]
            emit("short_edge_or_splice_junction", t, _mutate(rng, seg, 0.02), short_len)
    ties = find_tie_pairs(catalog, frag_len)
    for _ in range(n_per_regime if ties else 0):
        ta, tb, oa, ob = rng.choice(ties)
        A = catalog.targets[ta]
        frag = _mutate(rng, A.seq[oa:oa + frag_len], 0.003)
        r1, r2 = _make_pair(frag, read_len, layout, orientation=orientation)
        cases.append(SyntheticCase("paralog_or_strain_tie", frozenset([ta, tb]), r1, r2, tie_pair=(ta, tb)))
    # unannotated exon-skip junctions (only when the catalog carries genes with >= 3 exons)
    if catalog.mode == "eukaryotic":
        cands = exon_skip_candidates(catalog)
        prim = [c for c in cands if c[0].assembly_id == catalog.primary_assembly] or cands
        made, tries = 0, 0
        while prim and made < n_per_regime and tries < 50 * n_per_regime:
            tries += 1
            t, j = rng.choice(prim)
            frag = exon_skip_fragment(t, j, frag_len, read_len, layout, rng)
            if frag is None:
                continue
            truth = frozenset(x.target_id for x in catalog.groups.get(t.group_id, [t]))
            r1, r2 = _make_pair(_mutate(rng, frag, 0.005), read_len, layout, orientation=orientation)
            cases.append(SyntheticCase(EXON_SKIP, truth, r1, r2))
            made += 1
    return cases


@dataclass
class KernelScore:
    kernel: str
    label: str
    real_mapped_pct: float
    dominant_strand_purity_pct: float
    cross_assembly_tie_retention_pct: float
    recall: dict[str, float]
    overall_synthetic_recall_pct: float
    mate_tie_symmetry: float
    reads_per_sec: float
    wall_sec: float
    gate_passed: bool
    composite_raw: float
    optimality_score: float
    error: str = ""
    core4_recall_pct: float = 0.0

    @property
    def exon_skip_recall_pct(self) -> float | None:
        return self.recall.get(EXON_SKIP)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["recall"] = {k: round(v, 2) for k, v in self.recall.items()}
        for k in ("real_mapped_pct", "dominant_strand_purity_pct", "cross_assembly_tie_retention_pct", "overall_synthetic_recall_pct", "reads_per_sec", "wall_sec", "core4_recall_pct"):
            d[k] = round(d[k], 2)
        for k in ("mate_tie_symmetry", "composite_raw", "optimality_score"):
            d[k] = round(d[k], 4)
        return d


def _recall_and_symmetry(per: dict, cases: list[SyntheticCase], offset: int) -> tuple[dict[str, float], float]:
    hits: dict[str, int] = {}
    tot: dict[str, int] = {}
    sym_n, sym_ok = 0, 0
    for i, c in enumerate(cases):
        tot[c.regime] = tot.get(c.regime, 0) + 1
        found = per.get(offset + i)
        targets = set(found[0]) if found else set()
        if targets & c.truth:
            hits[c.regime] = hits.get(c.regime, 0) + 1
        if c.tie_pair is not None and targets:
            sym_n += 1
            if c.tie_pair[0] in targets and c.tie_pair[1] in targets:
                sym_ok += 1
    recall = {r: 100.0 * hits.get(r, 0) / n for r, n in tot.items()}
    symmetry = (sym_ok / sym_n) if sym_n else float("nan")
    return recall, symmetry


def evaluate_kernel(
    kernel: Kernel,
    catalog: Catalog,
    real_r1: list[Read],
    real_r2: list[Read] | None,
    cases: list[SyntheticCase],
    conditioning_active: bool,
) -> KernelScore:
    """Map real reads and synthetic cases in ONE batch (so insert-size estimation and pair
    rescue behave as they do in production), then score from the per-read trace."""
    n_real = len(real_r1)
    syn_r1 = [(c.r1, "I" * len(c.r1)) for c in cases]
    syn_r2 = [(c.r2, "I" * len(c.r2)) for c in cases] if (real_r2 is not None) else None
    batch_r1 = real_r1 + syn_r1
    batch_r2 = (real_r2 + syn_r2) if real_r2 is not None else None
    t0 = time.time()
    res = kernel.map_batch(batch_r1, batch_r2, trace=True)
    wall = time.time() - t0
    if getattr(kernel, "trace_costs_extra_pass", False) and res.wall_seconds:
        wall = res.wall_seconds          # per-read tracing needed an extra pass; time the production pass only
    per = res.per_read or {}
    tg = catalog.targets
    real_mapped = sum(1 for i in range(n_real) if i in per)
    map_pct = 100.0 * real_mapped / max(1, n_real)
    a_sum = b_sum = 0.0
    cross = 0
    for i in range(n_real):
        e = per.get(i)
        if not e:
            continue
        ts, fl = e
        if len(ts) == 1:
            if ts[0] in tg and not tg[ts[0]].is_decoy:
                if fl[0]:
                    a_sum += 1
                else:
                    b_sum += 1
        elif len({tg[t].assembly_id for t in ts if t in tg}) >= 2:
            cross += 1
    purity = 100.0 * max(a_sum, b_sum) / max(1.0, a_sum + b_sum)
    tie_ret = 100.0 * cross / max(1, real_mapped)
    recall, symmetry = _recall_and_symmetry(per, cases, n_real)
    non_tie = [v for k, v in recall.items() if k != "paralog_or_strain_tie"]
    overall = sum(non_tie) / len(non_tie) if non_tie else 0.0
    core = [recall[k] for k in CORE_RECALL_REGIMES if k in recall]
    core4 = sum(core) / len(core) if core else 0.0
    rps = len(batch_r1) / max(wall, 1e-3)
    sym_eff = 1.0 if math.isnan(symmetry) else symmetry
    tie_term = sym_eff if (conditioning_active and not math.isnan(symmetry)) else min(1.0, tie_ret / 60.0)
    composite = 0.35 * map_pct / 100.0 + 0.40 * overall / 100.0 + 0.15 * tie_term + 0.10 * min(1.0, math.log10(max(10.0, rps)) / 5.0)
    gate = True
    score = composite
    if conditioning_active and not math.isnan(symmetry):
        if symmetry < 0.80:
            gate = False
        score = composite * (0.5 + 0.5 * symmetry) if symmetry < 0.95 else composite
    if len(catalog.gene_targets) >= 25_000 and rps < 2_500.0:
        # On mammalian-scale catalogs, penalize sub-2.5k r/s SAM projection mappers so they cannot beat 30k+ r/s cDBG mappers
        score *= min(1.0, (max(rps, 10.0) / 2_500.0) ** 0.35)
    return KernelScore(kernel.name, kernel.label, map_pct, purity, tie_ret, recall, overall, sym_eff, rps, wall, gate, composite, score,
                       core4_recall_pct=core4)


def _probe_library_orientation(catalog: Catalog, real_r1: list[Read], max_reads: int = 1000) -> str:
    """Return 'forward' when real R1 reads predominantly map sense (+) on catalog transcripts, else 'reverse'."""
    if not real_r1:
        return "reverse"
    import tempfile

    try:
        import mappy
    except ImportError:
        return "reverse"
    genes = [t for t in catalog.gene_targets if t.assembly_id == catalog.primary_assembly][:2500] or catalog.gene_targets[:2500]
    if not genes:
        return "reverse"
    n_rev = n_fwd = 0
    with tempfile.TemporaryDirectory() as td:
        fa = f"{td}/probe.fa"
        with open(fa, "w") as fh:
            for t in genes:
                fh.write(f">{t.target_id}\n{t.seq}\n")
        idx = mappy.Aligner(fa, preset="sr", best_n=2)
        tbuf = mappy.ThreadBuffer()
        for seq, _q in real_r1[:max_reads]:
            hits = [h for h in idx.map(seq, buf=tbuf) if h.is_primary]
            if len(hits) != 1:
                continue
            if hits[0].strand < 0:
                n_rev += 1
            else:
                n_fwd += 1
    if (n_rev + n_fwd) >= 20 and n_rev / (n_rev + n_fwd) <= 0.20:
        return "forward"
    return "reverse"


def run_tournament(
    kernels: list[Kernel],
    catalog: Catalog,
    real_r1: list[Read],
    real_r2: list[Read] | None,
    layout: str,
    read_len: int | None = None,
    n_synthetic_per_regime: int = 600,
    conditioning_active: bool | None = None,
    log=print,
    match_insert: bool = False,
    cases: list[SyntheticCase] | None = None,
    regime=None,
) -> tuple[str, list[dict]]:
    """Score all kernels; returns (winner_name, scoreboard). Kernels that error are recorded and skipped."""
    if read_len is None:
        read_len = int(sorted(len(s) for s, _ in real_r1[:2000])[len(real_r1[:2000]) // 2]) if real_r1 else 100
    pre_excluded: dict[str, str] = {}
    if regime is not None:
        # panexpr.regime.RegimeDecision: kernels it excluded are never indexed or mapped; its tie gate wins
        pre_excluded = {k.name: regime.excluded[k.name] for k in kernels if k.name in regime.excluded}
        kernels = [k for k in kernels if k.name not in pre_excluded]
        if conditioning_active is None:
            conditioning_active = bool(regime.strict_tie_gate)
    if conditioning_active is None:
        conditioning_active = len(catalog.assemblies) > 1 or any(len(v) > 1 for v in catalog.groups.values())
    insert = estimate_insert_size(real_r1, real_r2) if layout == "PE" else {"layout": layout, "measurable": False}
    frag_len = default_frag_len(read_len)
    if match_insert and insert.get("measurable"):
        frag_len = max(read_len + 1, int(insert["median_insert"]))
    syn_orientation = _probe_library_orientation(catalog, real_r1) if cases is None else "custom"
    if cases is None:
        cases = build_synthetic_truth(catalog, layout, read_len, n_synthetic_per_regime, frag_len=frag_len, orientation=syn_orientation)
    log(f"  [tournament] synthetic frag_len={frag_len} (default {default_frag_len(read_len)}) orientation={syn_orientation}; empirical insert {insert}")
    n_tie = sum(1 for c in cases if c.regime == "paralog_or_strain_tie")
    board: list[KernelScore] = []
    for k in kernels:
        if not k.available():
            continue
        try:
            sc = evaluate_kernel(k, catalog, real_r1, real_r2, cases, conditioning_active and n_tie > 0)
        except Exception as exc:  # a broken kernel must not abort the study
            sc = KernelScore(k.name, k.label, 0, 0, 0, {}, 0, 0, 0, 0, False, 0, 0, error=f"{type(exc).__name__}: {exc}")
        board.append(sc)
        log(f"  [tournament] {k.label:55s} mapped={sc.real_mapped_pct:5.1f}% recall={sc.overall_synthetic_recall_pct:5.1f}% "
            f"tie_sym={sc.mate_tie_symmetry:.3f} gate={'ok' if sc.gate_passed else 'FAIL'} {sc.reads_per_sec:8.0f} r/s score={sc.optimality_score:.4f} {sc.error}")
    board.sort(key=lambda s: s.optimality_score, reverse=True)
    passing = [s for s in board if s.gate_passed and not s.error]
    winner = (passing[0] if passing else board[0]).kernel
    rows = [s.as_dict() for s in board]
    for name, why in pre_excluded.items():
        log(f"  [tournament] {name:55s} pre-excluded by regime classifier: {why}")
        rows.append({"kernel": name, "error": f"pre-excluded: {why}", "optimality_score": 0.0, "gate_passed": False, "pre_excluded": True})
    for r in rows:
        r["n_synthetic_cases"] = len(cases)
        r["n_tie_cases"] = n_tie
        r["conditioning_active"] = bool(conditioning_active and n_tie > 0)
        r["synthetic_frag_len"] = frag_len
        r["synthetic_orientation"] = syn_orientation
        r["empirical_insert"] = insert
        r["n_exon_skip_cases"] = sum(1 for c in cases if c.regime == EXON_SKIP)
    return winner, rows
