"""LocusZoom targeted resolvers (<1.5% discordant genomic bases):

1. Neural / regulated micro-exon (3-27 nt) junction-gap scanner:
   Rescues <15 nt exons systematically skipped by seed-and-extend aligners (STAR, minimap2, miniprot)
   when an aligned transcript/protein displays a 3-27 nt unaligned query gap across an intron.
2. Bayesian 3-feature single-exon gene vs retrotransposed processed-pseudogene classifier:
   Separates functional intronless genes (olfactory receptors, SOX/GPCRs, polyA-less canonical histones)
   from intronless retrocopies of multi-exon parent genes without egapx's blunt single-exon purge.
"""

from __future__ import annotations

from dataclasses import dataclass

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
RESCUE_CODE_VERSION = "v2"

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    " GCT".strip(): "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class MicroExonHit:
    intron_offset_start: int
    intron_offset_end: int
    exon_seq: str
    ppt_pyrimidine_frac: float
    identity: float
    score: float


def _pyrimidine_tract_fraction(upstream_seq: str) -> float:
    """Measure polypyrimidine (C/T) enrichment in the [-20, -3] acceptor window."""
    if not upstream_seq:
        return 0.0
    window = upstream_seq[-20:-3] if len(upstream_seq) >= 20 else upstream_seq[:-2]
    if not window:
        return 0.0
    ct = sum(1 for b in window.upper() if b in ("C", "T"))
    return ct / float(len(window))


def scan_junction_gap_micro_exon(
    intron_seq: str,
    unaligned_query_nt: str,
    allow_mismatches: int = 1,
) -> MicroExonHit | None:
    """Scan a single intron (5'->3' coding orientation) for a 3-27 nt micro-exon matching unaligned_query_nt.

    Requires canonical upstream '...AG' acceptor and downstream 'GT...' (or 'GC...') donor,
    plus absence of in-frame stop codons when len(query) is a multiple of 3.
    """
    q = unaligned_query_nt.upper()
    m = len(q)
    if m < 3 or m > 27:
        return None
    u = intron_seq.upper()
    n = len(u)
    if n < m + 20:
        return None

    best: MicroExonHit | None = None
    # Leave >=12 bp 5' donor margin and >=4 bp 3' acceptor margin inside the intron
    max_pos = n - m - 4
    if allow_mismatches == 0:
        if m % 3 == 0 and any(q[i : i + 3] in STOP_CODONS for i in range(0, m, 3)):
            return None
        pos = u.find(q, 12)
        while 0 <= pos < max_pos:
            acceptor = u[pos - 2 : pos]
            donor = u[pos + m : pos + m + 2]
            if acceptor == "AG" and donor in ("GT", "GC"):
                ppt = _pyrimidine_tract_fraction(u[max(0, pos - 22) : pos])
                donor_bonus = 0.05 if donor == "GT" else 0.0
                score = 0.65 + 0.30 * ppt + donor_bonus
                hit = MicroExonHit(
                    intron_offset_start=pos,
                    intron_offset_end=pos + m,
                    exon_seq=q,
                    ppt_pyrimidine_frac=ppt,
                    identity=1.0,
                    score=score,
                )
                if best is None or hit.score > best.score:
                    best = hit
            pos = u.find(q, pos + 1)
        return best

    for pos in range(12, max_pos):
        acceptor = u[pos - 2 : pos]
        donor = u[pos + m : pos + m + 2]
        if acceptor != "AG" or donor not in ("GT", "GC"):
            continue
        cand = u[pos : pos + m]
        mismatches = sum(1 for a, b in zip(cand, q) if a != b)
        if mismatches > allow_mismatches:
            continue
        if m % 3 == 0:
            if any(cand[i : i + 3] in STOP_CODONS for i in range(0, m, 3)):
                continue
        ident = (m - mismatches) / float(m)
        ppt = _pyrimidine_tract_fraction(u[max(0, pos - 22) : pos])
        donor_bonus = 0.05 if donor == "GT" else 0.0
        score = 0.65 * ident + 0.30 * ppt + donor_bonus
        hit = MicroExonHit(
            intron_offset_start=pos,
            intron_offset_end=pos + m,
            exon_seq=cand,
            ppt_pyrimidine_frac=ppt,
            identity=ident,
            score=score,
        )
        if best is None or hit.score > best.score:
            best = hit
    return best


@dataclass(frozen=True)
class SingleExonCandidate:
    candidate_id: str
    orf_length_nt: int
    has_internal_stop_or_frameshift: bool
    parent_multi_exon_intron_count: int
    unique_kmer_rna_tpm: float
    has_3prime_genomic_polya_tract: bool
    tiberius_coding_prob: float
    miniprot_identity: float
    has_histone_stem_loop: bool = False


def classify_single_exon_candidate(cand: SingleExonCandidate) -> str:
    """Classify an intronless ORF as 'functional_single_exon_gene', 'processed_pseudogene', or 'discard_noise'.

    Explicitly protects:
    - Replication-dependent canonical histones (0 TPM in polyA RNA-seq, stem-loop +, parent_introns == 0)
    - Transcriptionally silent intact Olfactory / GPCR receptors (0 TPM, high Tiberius + miniprot, parent_introns == 0)
    While rejecting intronless retrocopies derived from multi-exon parent genes.
    """
    if cand.has_internal_stop_or_frameshift:
        if cand.parent_multi_exon_intron_count >= 2 or cand.miniprot_identity >= 0.4:
            return "processed_pseudogene"
        return "discard_noise"

    # Replication-dependent histones first: they DO have multi-exon paralogs (H3.3, H2A.Z/H2Av variants),
    # but their 3' stem-loop is absent from polyadenylated variant-derived retrocopies.
    if cand.has_histone_stem_loop and cand.orf_length_nt >= 300:
        return "functional_single_exon_gene"

    # Retrotransposed copy of a multi-exon gene in the same genome
    if cand.parent_multi_exon_intron_count >= 2:
        if cand.has_3prime_genomic_polya_tract or cand.unique_kmer_rna_tpm < 1.0:
            return "processed_pseudogene"
        # Rare expressed functional retrogenes require strong unique RNA + coding prior + no genomic polyA artifact
        if cand.unique_kmer_rna_tpm >= 5.0 and cand.tiberius_coding_prob >= 0.8:
            return "functional_single_exon_gene"
        return "processed_pseudogene"

    # True intronless gene families (ORs, SOX, Type I IFNs)
    if cand.orf_length_nt >= 600 and (cand.tiberius_coding_prob >= 0.75 or cand.miniprot_identity >= 0.55):
        return "functional_single_exon_gene"
    if cand.orf_length_nt >= 300 and cand.tiberius_coding_prob >= 0.85 and cand.miniprot_identity >= 0.50:
        return "functional_single_exon_gene"
    if cand.orf_length_nt >= 450 and cand.unique_kmer_rna_tpm >= 3.0 and cand.tiberius_coding_prob >= 0.60:
        return "functional_single_exon_gene"
    # Short expressed intronless genes (antimicrobial / secreted peptides, small ribosomal-like ORFs):
    # strong unique-RNA expression plus an ab initio coding call rescues ORFs below the length floors.
    if cand.orf_length_nt >= 150 and cand.unique_kmer_rna_tpm >= 5.0 and cand.tiberius_coding_prob >= 0.75:
        return "functional_single_exon_gene"

    return "discard_noise"


# ----------------------------------------------------------------------------------------------
# Stage 5 driver: micro-exon rescue from compact RNA evidence (junction table + capped BAM slices)
# ----------------------------------------------------------------------------------------------

_RC = str.maketrans("ACGTN", "TGCAN")


def _rc(s: str) -> str:
    return s.translate(_RC)[::-1]


def _mismatches(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def microexons_from_junction_pairs(juncs: dict, min_support: int = 2, max_len: int = 27) -> dict:
    """Micro-exons already split by the aligner: supported (a,x) + (y,b) with 3 <= y-x <= max_len inside supported (a,b).

    ``juncs``: contig -> sorted [(a, b, support, strand)].  Returns {(contig, a, b): [(x, y, support)]}.
    """
    out: dict = {}
    for ctg, v in juncs.items():
        by_a: dict[int, list] = {}
        by_b: dict[int, list] = {}
        for a, b, s, st in v:
            if s >= min_support:
                by_a.setdefault(a, []).append((b, s, st))
                by_b.setdefault(b, []).append((a, s, st))
        for a, b, s, st in v:
            if s < min_support:
                continue
            for x, s1, st1 in by_a.get(a, []):
                if x >= b:
                    continue
                # junctions (y, b) closing a 3..max_len nt exon (x, y) inside (a, b)
                for y, s2, st2 in by_b.get(b, []):
                    if x < y and 3 <= y - x <= max_len and st1 == st2 == st:
                        out.setdefault((ctg, a, b), []).append((x, y, min(s1, s2)))
    return out


def _scan_contig_softclips_on_handles(
    handles: list,
    ctg: str,
    donor_set: set[int],
    acceptor_set: set[int],
    min_clip: int,
    max_reads_per_site: int,
) -> tuple[list[dict[int, list[str]]], list[dict[int, list[str]]]]:
    """Stream ``ctg`` once per handle, collecting capped donor/acceptor clip strings (G-D1..G-D3, D-1, D-2, I2)."""
    d_by_h: list[dict[int, list[str]]] = []
    a_by_h: list[dict[int, list[str]]] = []
    if max_reads_per_site <= 0 or (not donor_set and not acceptor_set):
        return [{} for _ in handles], [{} for _ in handles]
    for h in handles:
        d_clips: dict[int, list[str]] = {}
        a_clips: dict[int, list[str]] = {}
        for r in h.fetch(ctg):
            if r.is_secondary or r.is_supplementary or r.cigartuples is None:
                continue
            rend = r.reference_end
            if rend is not None and rend in donor_set and r.reference_start < rend:
                if len(d_clips.get(rend, ())) < max_reads_per_site and r.cigartuples[-1][0] == 4:
                    qseq = r.query_sequence
                    if qseq is not None:
                        clip = qseq[-r.cigartuples[-1][1] :]
                        if len(clip) >= min_clip:
                            d_clips.setdefault(rend, []).append(clip)
            rstart = r.reference_start
            if rstart in acceptor_set:
                if len(a_clips.get(rstart, ())) < max_reads_per_site and r.cigartuples[0][0] == 4:
                    qseq = r.query_sequence
                    if qseq is not None:
                        clip = qseq[: r.cigartuples[0][1]]
                        if len(clip) >= min_clip:
                            a_clips.setdefault(rstart, []).append(clip)
        d_by_h.append(d_clips)
        a_by_h.append(a_clips)
    return d_by_h, a_by_h


def _scan_contig_softclips_worker(
    args: tuple[list[str], str, set[int], set[int], int, int],
) -> tuple[str, list[dict[int, list[str]]], list[dict[int, list[str]]]]:
    """Dedicated-handle per-contig worker for ProcessPoolExecutor (G-D4)."""
    import pysam

    bams_clean, ctg, donor_set, acceptor_set, min_clip, max_reads_per_site = args
    handles = [pysam.AlignmentFile(b) for b in bams_clean]
    try:
        d_by_h, a_by_h = _scan_contig_softclips_on_handles(
            handles, ctg, donor_set, acceptor_set, min_clip, max_reads_per_site
        )
    finally:
        for h in handles:
            h.close()
    return ctg, d_by_h, a_by_h


def rescue_microexons_from_bam(
    bams: list[str],
    juncs: dict,
    genome,
    min_support: int = 3,
    min_intron: int = 80,
    min_clip: int = 5,
    max_reads_per_site: int = 200,
    max_workers: int = 1,
) -> dict:
    """Single-pass soft-clip split search per contig with exact random-fetch oracle parity.

    Streams each contig once per BAM handle (instead of ``2 * N_junctions`` BGZF random seeks),
    preserving per-handle site caps (G-D1), exact vote insertion order (G-D2), string-only clip
    storage (G-D3), placed-unmapped acceptor handling (D-1), ``max_reads_per_site <= 0`` (D-2),
    and slice-before-length semantics (I2). Optional multi-contig ``ProcessPoolExecutor`` (G-D4)
    opens dedicated ``pysam.AlignmentFile`` handles when ``>= 5_000`` junctions span ``>= 2`` contigs.

    Returns ``{(contig, a, b): [(ms, me, support)]}`` in genomic coordinates.
    """
    import pysam

    bams_clean = [str(b) for b in bams if b]
    if not bams_clean or not juncs:
        return {}

    contig_sets: dict[str, tuple[set[int], set[int]]] = {}
    for ctg, v in juncs.items():
        donor_set = {a for a, b, s, st in v if s >= min_support and b - a >= min_intron and st in "+-" and a > 0}
        acceptor_set = {b for a, b, s, st in v if s >= min_support and b - a >= min_intron and st in "+-" and b >= 0}
        if donor_set or acceptor_set:
            contig_sets[ctg] = (donor_set, acceptor_set)

    if not contig_sets:
        return {}

    total_juncs = sum(len(v) for v in juncs.values())
    use_pool = int(max_workers) > 1 and total_juncs >= 5_000 and len(contig_sets) >= 2
    clips_by_ctg: dict[str, tuple[list[dict[int, list[str]]], list[dict[int, list[str]]]]] = {}

    if use_pool:
        from concurrent.futures import ProcessPoolExecutor

        n_w = min(int(max_workers), len(contig_sets))
        tasks = [
            (bams_clean, ctg, dset, aset, min_clip, max_reads_per_site)
            for ctg, (dset, aset) in contig_sets.items()
        ]
        with ProcessPoolExecutor(max_workers=n_w) as ex:
            for ctg_r, d_by_h, a_by_h in ex.map(_scan_contig_softclips_worker, tasks):
                clips_by_ctg[ctg_r] = (d_by_h, a_by_h)

    votes: dict = {}
    handles = None if use_pool else [pysam.AlignmentFile(b) for b in bams_clean]
    try:
        n_h = len(bams_clean)
        for ctg, v in juncs.items():
            if ctg not in contig_sets:
                continue
            if use_pool:
                d_by_h, a_by_h = clips_by_ctg.pop(ctg)
            else:
                dset, aset = contig_sets[ctg]
                d_by_h, a_by_h = _scan_contig_softclips_on_handles(
                    handles, ctg, dset, aset, min_clip, max_reads_per_site
                )
            for a, b, s, strand in v:
                if s < min_support or b - a < min_intron or strand not in "+-":
                    continue
                intron_oriented: str | None = None
                right_flank: str | None = None
                left_flank: str | None = None
                for hi in range(n_h):
                    d_list = d_by_h[hi].get(a)
                    if d_list:
                        if intron_oriented is None:
                            raw_intr = genome.fetch(ctg, a, b).upper()
                            intron_oriented = raw_intr if strand == "+" else _rc(raw_intr)
                        if right_flank is None:
                            right_flank = genome.fetch(ctg, b, b + 512).upper()
                        for clip in d_list:
                            _vote_right(votes, ctg, a, b, intron_oriented, clip, right_flank, genome, strand)
                    a_list = a_by_h[hi].get(b)
                    if a_list:
                        if intron_oriented is None:
                            raw_intr = genome.fetch(ctg, a, b).upper()
                            intron_oriented = raw_intr if strand == "+" else _rc(raw_intr)
                        if left_flank is None:
                            left_flank = genome.fetch(ctg, max(0, a - 512), a).upper()
                        for clip in a_list:
                            _vote_left(votes, ctg, a, b, intron_oriented, clip, left_flank, genome, strand)
    finally:
        if handles is not None:
            for h in handles:
                h.close()

    out: dict = {}
    for (ctg, a, b, ms, me), cnt in votes.items():
        if cnt >= 2:
            out.setdefault((ctg, a, b), []).append((ms, me, cnt))
    return out


def _vote_right(votes, ctg, a, b, intron_oriented, clip, right_flank, genome, strand):
    """Genomic-left exon ends at a; clip = micro-exon (m nt) + start of exon at b."""
    L = len(clip)
    clip_u = clip.upper()
    for m in range(3, min(27, L) + 1):
        rest = clip_u[m:]
        rlen = len(rest)
        if rest:
            ref = right_flank[:rlen] if rlen <= 512 else genome.fetch(ctg, b, b + rlen).upper()
            if _mismatches(rest, ref) > max(1, rlen // 10):
                continue
        if rlen < 3 and L < 8:
            continue
        _scan_and_vote(votes, ctg, a, b, intron_oriented, clip_u[:m], strand)


def _vote_left(votes, ctg, a, b, intron_oriented, clip, left_flank, genome, strand):
    """Genomic-right exon starts at b; clip = end of exon at a + micro-exon (m nt)."""
    L = len(clip)
    clip_u = clip.upper()
    flank_len = len(left_flank)
    for m in range(3, min(27, L) + 1):
        rest = clip_u[: L - m]
        rlen = len(rest)
        if rest:
            ref = left_flank[max(0, flank_len - rlen) :] if rlen <= 512 else genome.fetch(ctg, a - rlen, a).upper()
            if _mismatches(rest, ref) > max(1, rlen // 10):
                continue
        if rlen < 3 and L < 8:
            continue
        _scan_and_vote(votes, ctg, a, b, intron_oriented, clip_u[L - m :], strand)


def _scan_and_vote(votes, ctg, a, b, intron_oriented, q, strand):
    query = q if strand == "+" else _rc(q)
    hit = scan_junction_gap_micro_exon(intron_oriented, query, allow_mismatches=0)
    if hit is None:
        return
    if strand == "+":
        ms, me = a + hit.intron_offset_start, a + hit.intron_offset_end
    else:
        n = b - a
        ms, me = a + (n - hit.intron_offset_end), a + (n - hit.intron_offset_start)
    k = (ctg, a, b, ms, me)
    votes[k] = votes.get(k, 0) + 1


def merge_microexon_maps(*maps: dict) -> dict:
    out: dict = {}
    for m in maps:
        for k, v in m.items():
            cur = {(x, y): s for x, y, s in out.get(k, [])}
            for x, y, s in v:
                cur[(x, y)] = cur.get((x, y), 0) + s
            out[k] = [(x, y, s) for (x, y), s in cur.items()]
    return out


def microexons_from_models(models, max_len: int = 27) -> dict:
    """Micro-exons (3..max_len nt internal CDS exons) called by Tiberius / miniprot chains, keyed by the
    spanning intron (a, b) that a spliced aligner reports when it skips them."""
    out: dict = {}
    for m in models:
        cds = sorted(m.cds) if hasattr(m, "cds") else []
        for i in range(1, len(cds) - 1):
            x, y = cds[i]
            if 3 <= y - x <= max_len:
                a, b = cds[i - 1][1], cds[i + 1][0]
                out.setdefault((m.contig, a, b), []).append((x, y, 2))
    return out


def candidate_exonic_bridge_gaps(rna_tx: list, max_gap: int = 1_000) -> list[tuple[str, int, int]]:
    """Enumerate inter-fragment genomic gaps (3..max_gap bp) between multi-exon same-strand RNA transcripts."""
    by_cs: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for t in rna_tx:
        if getattr(t, "strand", ".") not in ("+", "-"):
            continue
        ex = sorted(getattr(t, "exons", ()))
        if len(ex) < 3:
            continue
        by_cs.setdefault((t.contig, t.strand), []).append((ex[0][0], max(y for _, y in ex)))
    gaps: set[tuple[str, int, int]] = set()
    for (ctg, _strand), spans in by_cs.items():
        uniq = sorted(set(spans))
        for i in range(len(uniq)):
            _a0, b0 = uniq[i]
            for j in range(i + 1, len(uniq)):
                a1, _b1 = uniq[j]
                if a1 <= b0:
                    continue
                gap = a1 - b0
                if gap > max_gap:
                    break
                if gap >= 3:
                    gaps.add((ctg, b0, a1))
    return sorted(gaps)


def count_gap_interior_nonspliced_reads(
    bam_paths: list[str],
    gaps: list[tuple[str, int, int]],
    min_mapq: int = 20,
) -> dict[tuple[str, int, int], int]:
    """Count primary non-duplicate MAPQ>=min_mapq non-spliced reads overlapping strict gap interior [ga+1, gb-1)."""
    if not bam_paths or not gaps:
        return {}
    import pysam

    handles = [pysam.AlignmentFile(str(p), "rb") for p in bam_paths]
    out: dict[tuple[str, int, int], int] = {}
    try:
        for h in handles:
            refs = set(h.references)
            for ctg, ga, gb in gaps:
                if gb - ga < 3 or ctg not in refs:
                    continue
                cnt = 0
                for r in h.fetch(ctg, ga + 1, gb - 1):
                    if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate:
                        continue
                    if r.mapping_quality < min_mapq or r.cigartuples is None:
                        continue
                    if any(op == 3 for op, _ln in r.cigartuples):
                        continue
                    if r.reference_start is not None and r.reference_end is not None:
                        if r.reference_start < gb - 1 and r.reference_end > ga + 1:
                            cnt += 1
                if cnt > 0:
                    k = (ctg, int(ga), int(gb))
                    out[k] = out.get(k, 0) + cnt
    finally:
        for h in handles:
            h.close()
    return out
