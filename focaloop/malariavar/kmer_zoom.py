"""Stage 5: targeted 31-mer local assembly and breakpoint zoom.

assemble     greedy 31-mer de Bruijn walk (major + first-branch minor path) between reference anchors
zoom_genes   re-derive indel alleles in kelch13 / crt / mdr1 / dhfr / dhps from the reads captured during
             streaming, confirm or reject pileup indel calls, and add alleles only assembly sees
crai_zoom    single-container Sanger CRAM fetch (local .crai) to settle borderline CNV / hrp2 / hrp3 calls
"""
import os
from collections import Counter
from pathlib import Path

import numpy as np

from . import genome as G

K = 31
CRAI_DIR = G.default_staging_dir() / "data" / "crai"


def kmer_counts(reads, k=K):
    c = Counter()
    for r in reads:
        for i in range(len(r) - k + 1):
            km = r[i:i + k]
            if "N" not in km:
                c[km] += 1
    return c


def _walk(start, end_kmer, counts, max_len, min_count, forced=None):
    path = start
    cur = start[-K:]
    steps = 0
    first_branch = None
    used = {cur}
    while steps < max_len:
        if path.endswith(end_kmer) and len(path) > len(start):
            return path, first_branch
        opts = sorted(((counts.get(cur[1:] + b, 0), b) for b in "ACGT"), reverse=True)
        if forced is not None and steps == forced[0]:
            nb = forced[1]
        else:
            if opts[0][0] < min_count:
                return None, first_branch
            nb = opts[0][1]
            if first_branch is None and opts[1][0] >= max(min_count, 0.15 * opts[0][0]):
                first_branch = (steps, opts[1][1], opts[1][0], opts[0][0])
        cur = cur[1:] + nb
        if cur in used:
            return None, first_branch  # cycle (tandem repeat longer than k)
        used.add(cur)
        path += nb
        steps += 1
    return None, first_branch


def assemble(ref_window, reads, min_count=2, slack=80, use_omnitig_zoom=False):
    """paths from the left to the right reference anchor: [(seq, min k-mer support)], major first."""
    if use_omnitig_zoom:
        from . import omnitig_zoom as OZ
        return OZ.assemble(ref_window, reads, min_count=min_count, slack=slack)
    counts = kmer_counts(reads)
    left, right = ref_window[:K], ref_window[-K:]
    if counts.get(left, 0) < min_count or counts.get(right, 0) < min_count:
        return []
    max_len = len(ref_window) - K + slack
    major, br = _walk(left, right, counts, max_len, min_count)
    out = []
    if major:
        out.append((major, _support(major, counts)))
    if br is not None:
        minor, _ = _walk(left, right, counts, max_len, min_count, forced=(br[0], br[1]))
        if minor and minor != major:
            out.append((minor, _support(minor, counts)))
    return out


def omnitig_codon_calls(
    g,
    zmeta,
    zbuf,
    gene,
    codons_list,
    min_count=2,
    min_wsaf=0.05,
    flank=45,
):
    """Assemble local Simple Omnitig paths across codons_list with VarGeno weights & WSAF MLE.

    Returns sorted set of AA strings (1-letter for single codon, multi-letter haplotype for list)
    for haplotypes with marginal WSAF >= min_wsaf and support >= min_count, or None if unassembled.
    """
    if len(zmeta) == 0 or not codons_list:
        return None
    from . import omnitig_zoom as OZ

    codon_gs = [g.codon_goffs(gene, c)[0] for c in codons_list]
    all_gs = [x for gs in codon_gs for x in gs]
    lo = max(0, min(all_gs) - flank)
    hi = min(g.L, max(all_gs) + flank + 1)
    if hi - lo < K + 4:
        return None
    starts = zmeta[:, 0]
    idx = np.where((starts > lo - 350) & (starts < hi))[0]
    if len(idx) < min_count:
        return None
    reads = []
    for i in idx:
        _, o, ln = zmeta[i]
        reads.append(zbuf[o:o + ln].tobytes().decode())
    ref = g.bases(lo, hi)
    haps = OZ.assemble_omnitig_wsaf(ref, reads, min_count=min_count, slack=max(40, flank))
    if not haps:
        return None

    left_anchor, right_anchor = ref[:K], ref[-K:]
    rel_gs = [[x - lo for x in gs] for gs in codon_gs]
    min_rel = min(x for r in rel_gs for x in r)
    max_rel = max(x for r in rel_gs for x in r)

    out_aas = set()
    for h in haps:
        if h.wsaf < min_wsaf or h.min_support < min_count:
            continue
        seq = h.seq
        # Align omnitig onto ref using unique start or end k-mer anchor.
        p0 = ref.find(seq[:K]) if len(seq) >= K else -1
        if p0 < 0 or ref.find(seq[:K], p0 + 1) >= 0:
            p1 = ref.find(seq[-K:]) if len(seq) >= K else -1
            if p1 >= 0 and ref.find(seq[-K:], p1 + 1) < 0:
                p0 = p1 + K - len(seq)
            elif h.is_anchor_to_anchor and seq.startswith(left_anchor) and seq.endswith(right_anchor):
                p0 = 0
            else:
                continue
        if p0 < 0 or min_rel < p0 or max_rel >= p0 + len(seq):
            continue
        # Guard against large frameshifting indels shifting codon coordinates vs ref.
        if abs(len(seq) - (len(ref) - p0 if seq.endswith(right_anchor) else len(seq))) > 15:
            continue
        # Locate local delta between omnitig [0..len(seq)] and ref[p0..p0+len(seq)] if lengths differ.
        if p0 + len(seq) <= len(ref) and not (h.is_anchor_to_anchor and len(seq) != len(ref)):
            off_in_seq = 0
            # Verify anchors / handle exact SNV-level or same-length path
            if all(0 <= r - p0 < len(seq) for g_list in rel_gs for r in g_list):
                try:
                    hap_aa = "".join(
                        g.codon_aa(gene, c, [seq[r - p0] for r in rlist])
                        for c, rlist in zip(codons_list, rel_gs)
                    )
                    if "*" not in hap_aa or len(hap_aa) == 1:
                        out_aas.add(hap_aa)
                except KeyError:
                    continue
        else:
            # Handle in-frame indels inside window: adjust positions past indel via diff_alleles
            d = diff_alleles(ref[p0:], seq) if p0 == 0 else None
            if d is None:
                continue
            doff, dr, da = d
            shift = len(da) - len(dr)
            if shift % 3 != 0:
                continue
            ok = True
            bases_per_c = []
            for rlist in rel_gs:
                cur_b = []
                for r in rlist:
                    idx_s = r - p0
                    if idx_s >= doff + len(dr):
                        idx_s += shift
                    elif idx_s >= doff:
                        ok = False
                        break
                    if not (0 <= idx_s < len(seq)):
                        ok = False
                        break
                    cur_b.append(seq[idx_s])
                if not ok:
                    break
                bases_per_c.append(cur_b)
            if ok:
                try:
                    hap_aa = "".join(
                        g.codon_aa(gene, c, bs) for c, bs in zip(codons_list, bases_per_c)
                    )
                    out_aas.add(hap_aa)
                except KeyError:
                    continue
    return out_aas or None


def _support(seq, counts):
    return min(counts.get(seq[i:i + K], 0) for i in range(len(seq) - K + 1))


def diff_alleles(ref, alt):
    """minimal (offset, REF, ALT) with one anchor base, VCF style, for a single contiguous difference."""
    if ref == alt:
        return None
    p = 0
    while p < min(len(ref), len(alt)) and ref[p] == alt[p]:
        p += 1
    s = 0
    while s < min(len(ref), len(alt)) - p and ref[-1 - s] == alt[-1 - s]:
        s += 1
    r = ref[p:len(ref) - s]
    a = alt[p:len(alt) - s]
    if len(r) != len(a) or len(r) == 0:
        p -= 1  # anchor base for indels
        r = ref[p:len(ref) - s]
        a = alt[p:len(alt) - s]
    return p, r, a


def zoom_genes(g, zmeta, zbuf, indel_calls, genes=G.ZOOM_GENES, flank=70, min_count=3, use_omnitig_zoom=False):
    """returns rows: gene, goff (anchor), REF, ALT, kmer_support, status in {confirmed, assembly_only, rejected}."""
    starts = zmeta[:, 0]
    rows = []
    if indel_calls is None:
        return rows
    from .caller import vcf_alleles
    _, _, crefs, calts = vcf_alleles(g, indel_calls)
    for gene in genes:
        gg = g.gene(gene)
        a0, a1 = g.goff(gg["contig"], gg["start"]), g.goff(gg["contig"], gg["end"])
        sel = np.where((indel_calls["goff"] >= a0) & (indel_calls["goff"] <= a1) & (indel_calls["gt"] > 0))[0]
        windows = sorted(set(int(indel_calls["goff"][i]) for i in sel))
        # also probe windows where many captured reads carry soft clips is out of scope; indel sites drive windows
        for w in windows:
            lo, hi = w - flank, w + flank + 1
            idx = np.where((starts > lo - 400) & (starts < hi))[0]
            reads = []
            for i in idx:
                s, o, ln = zmeta[i]
                seq = zbuf[o:o + ln].tobytes().decode()
                reads.append(seq)
            ref = g.bases(lo, hi)
            called = []
            for i in sel:
                if int(indel_calls["goff"][i]) == w:
                    off_c = w - lo
                    rc, ac = crefs[i], calts[i]
                    hap_c = ref[:off_c] + ac + ref[off_c + len(rc):] if 0 <= off_c <= len(ref) - len(rc) else None
                    called.append((w, rc, ac, hap_c))
            matched_called = set()
            if use_omnitig_zoom:
                from . import omnitig_zoom as OZ
                haps = OZ.assemble_omnitig_wsaf(ref, reads, min_count=min_count)
                has_paths = bool(haps)
                seen_vars = {}
                for h in haps:
                    if h.wsaf < 0.05:
                        continue
                    for v_off, vr, va in h.variants:
                        key = (lo + v_off, vr, va)
                        prev = seen_vars.get(key)
                        if prev is None or h.mean_support > prev[0]:
                            seen_vars[key] = (h.mean_support, h.wsaf, h.seq)
                for (pos_g, r, a), (sup, _wsaf, seq) in seen_vars.items():
                    hit = [idx_c for idx_c, (cw, rc, ac, hap_c) in enumerate(called)
                           if (cw == pos_g and rc == r and ac == a) or (hap_c is not None and hap_c == seq)]
                    if hit:
                        for idx_c in hit:
                            matched_called.add(idx_c)
                            cw, rc, ac, _ = called[idx_c]
                            rows.append(dict(gene=gene, goff=cw, ref=rc, alt=ac, kmer_support=int(round(sup)),
                                             status="confirmed", n_reads=len(reads)))
                    else:
                        rows.append(dict(gene=gene, goff=pos_g, ref=r, alt=a, kmer_support=int(round(sup)),
                                         status="assembly_only", n_reads=len(reads)))
            else:
                paths = assemble(ref, reads, min_count=min_count)
                has_paths = bool(paths)
                for seq, sup in paths:
                    d = diff_alleles(ref, seq)
                    if d is None:
                        continue
                    off, r, a = d
                    pos_g = lo + off
                    hit = [idx_c for idx_c, (cw, rc, ac, hap_c) in enumerate(called)
                           if (cw == pos_g and rc == r and ac == a) or (hap_c is not None and hap_c == seq)]
                    if hit:
                        for idx_c in hit:
                            matched_called.add(idx_c)
                            cw, rc, ac, _ = called[idx_c]
                            rows.append(dict(gene=gene, goff=cw, ref=rc, alt=ac, kmer_support=int(sup),
                                             status="confirmed", n_reads=len(reads)))
                    else:
                        rows.append(dict(gene=gene, goff=pos_g, ref=r, alt=a, kmer_support=int(sup),
                                         status="assembly_only", n_reads=len(reads)))
            for idx_c, (cw, rc, ac, _) in enumerate(called):
                if idx_c not in matched_called:
                    rows.append(dict(gene=gene, goff=cw, ref=rc, alt=ac, kmer_support=0,
                                     status="rejected" if has_paths else "unresolved", n_reads=len(reads)))
    return rows


def containers_bytes(crai, regions):
    """bytes htslib pulls for the regions: container start through the end of the last touched slice."""
    import gzip
    import io
    arr = np.loadtxt(io.StringIO(gzip.decompress(Path(crai).read_bytes()).decode()), dtype=np.int64, ndmin=2)
    g = G.load()
    need = {}
    for c, s, e in regions:
        sid = g.cidx(c)
        sel = arr[(arr[:, 0] == sid) & (arr[:, 1] <= e) & (arr[:, 1] + arr[:, 2] - 1 >= s)]
        for row in sel:
            need[int(row[3])] = max(need.get(int(row[3]), 0), int(row[4] + row[5]))
    return int(sum(need.values()))


def crai_zoom(cram_url, sample, contig, start, end, flank=20000, min_mapq=1):
    """read-depth ratio of [start,end] vs the flanks from the Sanger CRAM, fetching only the containers needed.

    Returns ratio, reads in locus, reads in flanks, bytes of the containers touched (from the .crai).
    """
    import pysam
    crai = CRAI_DIR / f"{sample}.cram.crai"
    if not crai.exists():
        return None
    regions = [(contig, max(1, start - flank), start - 1), (contig, start, end), (contig, end + 1, end + flank)]
    nbytes = containers_bytes(crai, regions)
    cnt = []
    with pysam.AlignmentFile(cram_url, "rc", index_filename=str(crai), reference_filename=str(G.REF_FASTA)) as f:
        for c, s, e in regions:
            n = 0
            for r in f.fetch(c, s - 1, e):
                if r.is_secondary or r.is_supplementary or r.is_unmapped or r.mapping_quality < min_mapq:
                    continue
                n += 1
            cnt.append(n / max(1, e - s + 1))
    flank_d = (cnt[0] + cnt[2]) / 2
    return dict(ratio=(cnt[1] / flank_d) if flank_d > 0 else np.nan, locus_density=cnt[1],
                flank_density=flank_d, bytes=int(nbytes))
