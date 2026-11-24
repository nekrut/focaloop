"""Numba kernels for the streaming hot path (no Python per read).

fastq_block   interleaved FASTQ -> spot-hash subsample -> Pf 31-mer host sieve -> renamed FASTQ for the mapper
sam_block     mapper SAM -> per-base A/C/G/T counts, deletion/insertion counts, indel events, 1 kb depth bins
kmer table    open-addressing set of canonical Pf 31-mers with low-complexity (DUST-like) k-mers removed
"""
import numpy as np
from numba import njit

K = 31
KMASK = (1 << (2 * K)) - 1
EMPTY = np.uint64(0)

ENC = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    ENC[_c] = _i
    ENC[_c + 32] = _i


@njit(cache=True, inline="always")
def mix64(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


# ----------------------------------------------------------------------------------------------- k-mer set
@njit(cache=True)
def dust_score(seq, i):
    """triplet-repeat score of seq[i:i+31] (DUST-style, higher = lower complexity)."""
    cnt = np.zeros(64, np.int32)
    s = 0
    for j in range(i, i + K - 2):
        a, b, c = seq[j], seq[j + 1], seq[j + 2]
        if a > 3 or b > 3 or c > 3:
            continue
        t = a * 16 + b * 4 + c
        s += cnt[t]
        cnt[t] += 1
    return s


@njit(cache=True)
def kset_insert(tab, key):
    m = np.uint64(len(tab) - 1)
    h = mix64(key) & m
    k1 = key + np.uint64(1)
    while True:
        v = tab[h]
        if v == EMPTY:
            tab[h] = k1
            return 1
        if v == k1:
            return 0
        h = (h + np.uint64(1)) & m


@njit(cache=True, inline="always")
def kset_has(tab, key):
    m = np.uint64(len(tab) - 1)
    h = mix64(key) & m
    k1 = key + np.uint64(1)
    while True:
        v = tab[h]
        if v == k1:
            return True
        if v == EMPTY:
            return False
        h = (h + np.uint64(1)) & m


@njit(cache=True)
def build_kset(seq, contig_ends, tab, dust_max):
    """insert canonical 31-mers of the reference (codes 0..3) that do not cross contig ends or Ns."""
    n_ins = 0
    n_dust = 0
    start = 0
    for ce in contig_ends:
        fw = np.uint64(0)
        rv = np.uint64(0)
        run = 0
        for i in range(start, ce):
            c = seq[i]
            if c > 3:
                run = 0
                continue
            fw = ((fw << np.uint64(2)) | np.uint64(c)) & np.uint64(KMASK)
            rv = (rv >> np.uint64(2)) | (np.uint64(3 - c) << np.uint64(2 * (K - 1)))
            run += 1
            if run >= K:
                if dust_score(seq, i - K + 1) > dust_max:
                    n_dust += 1
                    continue
                n_ins += kset_insert(tab, min(fw, rv))
        start = ce
    return n_ins, n_dust


@njit(cache=True)
def read_pf_hits(buf, s0, ln, tab, stride, need):
    """count Pf k-mer hits at positions 0, stride, 2*stride ... of one read (early exit at `need`)."""
    hits = 0
    if ln < K:
        return 0
    p = 0
    while p + K <= ln:
        fw = np.uint64(0)
        rv = np.uint64(0)
        ok = True
        for j in range(K):
            c = ENC_T[buf[s0 + p + j]]
            if c > 3:
                ok = False
                break
            fw = (fw << np.uint64(2)) | np.uint64(c)
            rv = rv | (np.uint64(3 - c) << np.uint64(2 * j))
        if ok and kset_has(tab, min(fw, rv)):
            hits += 1
            if hits >= need:
                return hits
        p += stride
    return hits


def read_pf_hits_pla(buf, s0, ln, pla_idx, need):
    """Count Pf splitmix64 open-syncmer hits in buf[s0:s0+ln] against mmap PLA-index (early exit at `need`)."""
    return pla_idx.read_pf_hits(buf, s0, ln, need)


def pla_fastq_block(buf, n, out, spot0, keep_thr, paired, pla_idx, need, stats, qhist, ad, min_len, max_spots=0):
    """Consume complete FASTQ records via native C pla_fastq_block (zero FPR/FNR mmap PLA-index)."""
    return pla_idx.fastq_block(buf, n, out, spot0, keep_thr, paired, need, stats, qhist, ad, min_len, max_spots)


ENC_T = ENC  # global constant array captured by numba


# ----------------------------------------------------------------------------------------------- FASTQ
@njit(cache=True, inline="always")
def _line_end(buf, i, n):
    while i < n and buf[i] != 10:
        i += 1
    return i


@njit(cache=True)
def _stem_len(buf, s, e):
    """length of the read-name stem: truncate at first space/tab, then strip trailing .0/.1/.2 or /1 /2 marker."""
    w = s
    while w < e and buf[w] != 32 and buf[w] != 9:
        w += 1
    if w - s >= 2 and (buf[w - 2] == 46 or buf[w - 2] == 47) and (buf[w - 1] == 48 or buf[w - 1] == 49 or buf[w - 1] == 50):
        return (w - 2) - s
    j = w - 1
    while j > s and buf[j] != 46 and buf[j] != 47:
        j -= 1
    if j > s and buf[j] == 47 and j + 1 < w:
        all_dig = True
        for k in range(j + 1, w):
            if buf[k] < 48 or buf[k] > 57:
                all_dig = False
                break
        if all_dig:
            return j - s
    return w - s


@njit(cache=True)
def _write_int(out, o, v):
    if v == 0:
        out[o] = 48
        return o + 1
    tmp = np.empty(20, np.uint8)
    k = 0
    while v > 0:
        tmp[k] = 48 + v % 10
        v //= 10
        k += 1
    for j in range(k):
        out[o + j] = tmp[k - 1 - j]
    return o + k


ADAPTER = np.frombuffer(b"AGATCGGAAGAGC", dtype=np.uint8)


@njit(cache=True)
def adapter_pos(buf, s, e, ad):
    """first offset of the Illumina adapter stem in buf[s:e] (read length when absent)."""
    m = len(ad)
    for i in range(s, e - m + 1):
        ok = True
        for j in range(m):
            if buf[i + j] != ad[j]:
                ok = False
                break
        if ok:
            return i - s
    return e - s


@njit(cache=True)
def _emit(out, o, spot, buf, ss, se, qs, qe):
    out[o] = 64
    o = _write_int(out, o + 1, spot)
    out[o] = 10
    o += 1
    for j in range(ss, se):
        out[o] = buf[j]
        o += 1
    out[o] = 10
    out[o + 1] = 43
    out[o + 2] = 10
    o += 3
    for j in range(qs, qe):
        out[o] = buf[j]
        o += 1
    out[o] = 10
    return o + 1


@njit(cache=True, inline="always")
def _trim_cr(buf, s, e):
    return e - 1 if (e > s and buf[e - 1] == 13) else e


@njit(cache=True)
def fastq_block(buf, n, out, spot0, keep_thr, paired, tab, stride, need, stats, qhist, ad, min_len, max_spots=0):
    """Consume complete records (pairs when paired=1) from buf[:n].

    A spot is kept when (mix64(spot) >> 32) < keep_thr (keep_thr = p * 2**32), then passes the host
    sieve when its mates carry >= `need` Pf 31-mer hits (need = 0 disables the sieve).
    stats[0] spots seen, [1] kept by subsample, [2] passing the Pf sieve, [3] bases seen,
         [4] bases emitted, [5] unpaired/malformed records dropped, [6] max read length, [7] mates adapter-trimmed
         (optional [8]: last emitted spot index, or -1 if none yet)
    Returns (bytes consumed, bytes written to out, next spot index). out must hold >= 2 * n bytes.
    """
    i = 0
    o = 0
    spot = spot0
    nrec = 2 if paired else 1
    hs = np.zeros(2, np.int64)
    he = np.zeros(2, np.int64)
    ss = np.zeros(2, np.int64)
    se = np.zeros(2, np.int64)
    qs = np.zeros(2, np.int64)
    qe = np.zeros(2, np.int64)
    re = np.zeros(2, np.int64)
    while i < n:
        if max_spots > 0 and spot >= max_spots:
            break
        if buf[i] != 64:  # not at a record start: skip one line
            e = _line_end(buf, i, n)
            if e >= n:
                break
            i = e + 1
            continue
        j = i
        complete = True
        malformed = False
        for r in range(nrec):
            if j >= n:
                complete = False
                break
            if buf[j] != 64:
                malformed = True
                break
            e1 = _line_end(buf, j, n)
            if e1 >= n:
                complete = False
                break
            e2 = _line_end(buf, e1 + 1, n)
            if e2 >= n:
                complete = False
                break
            e3 = _line_end(buf, e2 + 1, n)
            if e3 >= n:
                complete = False
                break
            e4 = _line_end(buf, e3 + 1, n)
            if e4 >= n:
                complete = False
                break
            he_r = _trim_cr(buf, j + 1, e1)
            se_r = _trim_cr(buf, e1 + 1, e2)
            qe_r = _trim_cr(buf, e3 + 1, e4)
            # strict 4-line FASTQ framing: '@' header, '+' separator on line 3, equal seq & qual lengths
            if he_r <= j + 1 or e2 + 1 >= e3 or buf[e2 + 1] != 43 or (se_r - (e1 + 1)) != (qe_r - (e3 + 1)):
                malformed = True
                break
            hs[r] = j + 1
            he[r] = he_r
            ss[r] = e1 + 1
            se[r] = se_r
            qs[r] = e3 + 1
            qe[r] = qe_r
            re[r] = e4
            j = e4 + 1
        if not complete:
            break
        if malformed:
            stats[5] += 1
            e = _line_end(buf, i, n)
            if e >= n:
                break
            i = e + 1
            continue
        if nrec == 2:
            l0 = _stem_len(buf, hs[0], he[0])
            same = l0 == _stem_len(buf, hs[1], he[1])
            if same:
                for t in range(l0):
                    if buf[hs[0] + t] != buf[hs[1] + t]:
                        same = False
                        break
            if not same:  # drop the first record, resync on the second
                stats[5] += 1
                i = re[0] + 1
                continue
            # 3-segment multiplexed spots (biological R1 + short index barcode + biological R2): skip the barcode
            if (se[0] - ss[0]) >= min_len and (se[1] - ss[1]) < 20 and j < n:
                if buf[j] == 64:
                    te1 = _line_end(buf, j, n)
                    te2 = _line_end(buf, te1 + 1, n) if te1 < n else n
                    te3 = _line_end(buf, te2 + 1, n) if te2 < n else n
                    te4 = _line_end(buf, te3 + 1, n) if te3 < n else n
                    if te4 >= n:
                        break
                    the = _trim_cr(buf, j + 1, te1)
                    tse = _trim_cr(buf, te1 + 1, te2)
                    tqe = _trim_cr(buf, te3 + 1, te4)
                    if (the > j + 1 and te2 + 1 < te3 and buf[te2 + 1] == 43
                            and (tse - (te1 + 1)) == (tqe - (te3 + 1))
                            and _stem_len(buf, j + 1, the) == l0):
                        same3 = True
                        for t in range(l0):
                            if buf[hs[0] + t] != buf[j + 1 + t]:
                                same3 = False
                                break
                        if same3:
                            hs[1] = j + 1
                            he[1] = the
                            ss[1] = te1 + 1
                            se[1] = tse
                            qs[1] = te3 + 1
                            qe[1] = tqe
                            re[1] = te4
                            j = te4 + 1
                            stats[9] += 1
        i = j
        stats[0] += 1
        blen = 0
        for r in range(nrec):
            ln = se[r] - ss[r]
            blen += ln
            if ln > stats[6]:
                stats[6] = ln
            if stats[0] <= 2000:
                for t in range(qs[r], qe[r]):
                    qhist[buf[t] & 127] += 1
        stats[3] += blen
        if (mix64(np.uint64(spot)) >> np.uint64(32)) >= np.uint64(keep_thr):
            spot += 1
            continue
        stats[1] += 1
        if len(ad) > 0:  # adapter read-through: trim both mates, drop pairs left too short to place
            short = False
            for r in range(nrec):
                t = adapter_pos(buf, ss[r], se[r], ad)
                if t < se[r] - ss[r]:
                    stats[7] += 1
                    se[r] = ss[r] + t
                    qe[r] = qs[r] + t
                if t < min_len:
                    short = True
            if short:
                spot += 1
                continue
        hits = 0
        if need > 0:
            for r in range(nrec):
                hits += read_pf_hits(buf, ss[r], se[r] - ss[r], tab, stride, need - hits)
                if hits >= need:
                    break
        if need == 0 or hits >= need:
            stats[2] += 1
            stats[4] += blen
            if len(stats) > 8:
                stats[8] = spot
            for r in range(nrec):
                o = _emit(out, o, spot, buf, ss[r], se[r], qs[r], qe[r])
        spot += 1
    return i, o, spot


# ----------------------------------------------------------------------------------------------- SAM
@njit(cache=True, inline="always")
def _atoi(buf, s, e):
    v = 0
    neg = False
    if s < e and buf[s] == 45:
        neg = True
        s += 1
    for j in range(s, e):
        v = v * 10 + (buf[j] - 48)
    return -v if neg else v


@njit(cache=True)
def _contig(buf, s, e, names, name_off, last):
    """contig index of RNAME buf[s:e]; checks the cached index first."""
    nn = len(name_off) - 1
    for k in range(nn + 1):
        c = last if k == 0 else k - 1
        if c < 0 or c >= nn:
            continue
        a, b = name_off[c], name_off[c + 1]
        if b - a != e - s:
            continue
        ok = True
        for t in range(b - a):
            if names[a + t] != buf[s + t]:
                ok = False
                break
        if ok:
            return c
    return -1


@njit(cache=True)
def _base_mask(buf, cs, ce, qs, qlen, g0, ref, flt, mask, mm):
    """BAQ-style read mask: mask[q] = True for bases not to count.

    flt = [indel_flank, clip_flank, end_trim, mm_win, mm_n] (0 disables each)
    indel_flank   bases within this many query bases of an insertion or deletion in the same read
    clip_flank    bases next to a soft clip
    end_trim      the first / last bases of the read
    mm_win, mm_n  >= mm_n mismatches against the reference within mm_win bp: mask from first to last of the cluster
    Returns the number of masked bases."""
    for q in range(qlen):
        mask[q] = False
    fi = flt[0]
    fc = flt[1]
    fe = flt[2]
    mw = flt[3]
    mn = flt[4]
    num = 0
    qi = 0
    g = g0
    nmm = 0
    lead = 0
    trail = 0
    for j in range(cs, ce):
        ch = buf[j]
        if ch >= 48 and ch <= 57:
            num = num * 10 + (ch - 48)
            continue
        if ch == 77 or ch == 61 or ch == 88:
            if mn > 0:
                for t in range(num):
                    b = ENC_T[buf[qs + qi + t]]
                    r = ref[g + t]
                    if b < 4 and r < 4 and b != r and nmm < len(mm):
                        mm[nmm] = qi + t
                        nmm += 1
            qi += num
            g += num
        elif ch == 73:
            if fi > 0:
                for q in range(max(0, qi - fi), min(qlen, qi + num + fi)):
                    mask[q] = True
            qi += num
        elif ch == 68:
            if fi > 0:
                for q in range(max(0, qi - fi), min(qlen, qi + fi)):
                    mask[q] = True
            g += num
        elif ch == 78:
            g += num
        elif ch == 83:
            if qi == 0:
                lead = num
            else:
                trail = num
            qi += num
        num = 0
    if fc > 0:
        if lead > 0:
            for q in range(lead, min(qlen, lead + fc)):
                mask[q] = True
        if trail > 0:
            for q in range(max(0, qlen - trail - fc), qlen - trail):
                mask[q] = True
    if fe > 0:
        for q in range(min(fe, qlen)):
            mask[q] = True
        for q in range(max(0, qlen - fe), qlen):
            mask[q] = True
    if mn > 1 and nmm >= mn:
        a = 0
        for b in range(nmm):
            while mm[b] - mm[a] > mw:
                a += 1
            if b - a + 1 >= mn:
                for q in range(mm[a], mm[b] + 1):
                    mask[q] = True
    k = 0
    for q in range(qlen):
        if mask[q]:
            k += 1
    return k


@njit(cache=True)
def sam_block(buf, n, names, name_off, coff, boff, ref, counts, delc, insc, binbases,
              ev, ev_n, ccount, prev, stats, minmapq, minbq, binmapq, max_mm_frac, zmask, zbuf, zmeta, zn, rf, rf_n,
              flt, ins_buf, ins_meta, ins_n, dedup_tab):
    """Accumulate SAM records in buf[:n] (complete lines only). Returns bytes consumed.

    counts[g, 0..3]  base counts (MAPQ>=minmapq, BQ>=minbq, proper pair, mismatch filter, mate overlap removed)
    delc/insc        deletion-spanning / insertion-after counts
    binbases[b]      aligned bases of MAPQ>=binmapq primary reads (1 kb depth bins)
    ev[k] = (goff, kind 1 del / 2 ins, length, ins-seq 2-bit code or FNV hash)   ev_n[0] = used rows
    ccount[c]        primary MAPQ>=minmapq reads per contig (stream-order diagnostics)
    prev             [spot, contig, start, end] of the previous primary record (mate overlap)
    stats            [0 records, 1 primary mapped, 2 counted reads, 3 counted bases, 4 max spot seen,
                      5 mismatch-filtered, 6 events dropped (buffer full), 7 overlap bases skipped, 8 zoom reads,
                      9 bases masked by the BAQ-style read mask, (optional 10: PCR duplicate reads skipped)]
    flt              [indel_flank, clip_flank, end_trim, mm_win, mm_n]: see _base_mask (all 0 = off)
    zmask[g]         True on curated zoom loci: reads touching them are copied to zbuf for Stage 5
    rf[k] = (contig, left pos, right mate pos)  face-away (RF) pairs 300 bp - 300 kb apart, MAPQ >= minmapq:
                     the junction signature of tandem duplications (recorded once, from the leftmost mate)
    """
    i = 0
    last = 0
    f_s = np.zeros(12, np.int64)
    f_e = np.zeros(12, np.int64)
    evcap = ev.shape[0]
    bmask = np.zeros(4096, np.bool_)
    mmbuf = np.zeros(512, np.int64)
    use_mask = flt[0] > 0 or flt[1] > 0 or flt[2] > 0 or flt[4] > 0
    dmask = np.uint64(len(dedup_tab) - 1) if len(dedup_tab) > 0 else np.uint64(0)
    while i < n:
        e = _line_end(buf, i, n)
        if e >= n:
            break
        if buf[i] == 64:
            i = e + 1
            continue
        # split first 11 fields + tag region
        k = 0
        s = i
        for j in range(i, e + 1):
            if j == e or buf[j] == 9:
                if k < 12:
                    f_s[k] = s
                    f_e[k] = j
                k += 1
                s = j + 1
                if k >= 12:
                    break
        if k < 11:
            i = e + 1
            continue
        tag_s = f_s[11] if k >= 12 else e
        stats[0] += 1
        spot = _atoi(buf, f_s[0], f_e[0])
        if spot > stats[4]:
            stats[4] = spot
        flag = _atoi(buf, f_s[1], f_e[1])
        if flag & 0xF04:  # unmapped, secondary, qcfail, duplicate, supplementary
            i = e + 1
            continue
        c = _contig(buf, f_s[2], f_e[2], names, name_off, last)
        if c < 0:
            i = e + 1
            continue
        last = c
        pos = _atoi(buf, f_s[3], f_e[3])
        mapq = _atoi(buf, f_s[4], f_e[4])
        stats[1] += 1
        g0 = coff[c] + pos - 1
        # walk CIGAR once for reference span, indel bases, and leading/trailing soft clips
        rspan = 0
        indel = 0
        num = 0
        q_cur = 0
        lead_clip = 0
        trail_clip = 0
        for j in range(f_s[5], f_e[5]):
            ch = buf[j]
            if ch >= 48 and ch <= 57:
                num = num * 10 + (ch - 48)
                continue
            if ch == 77 or ch == 61 or ch == 88 or ch == 68 or ch == 78:
                rspan += num
            if ch == 73 or ch == 68:
                indel += num
            if ch == 77 or ch == 61 or ch == 88 or ch == 73:
                q_cur += num
            elif ch == 83:
                if q_cur == 0:
                    lead_clip = num
                else:
                    trail_clip = num
                q_cur += num
            num = 0
        paired = (flag & 1) != 0
        same_spot = prev[0] == spot
        same_chr = f_e[6] - f_s[6] == 1 and buf[f_s[6]] == 61
        pn = _atoi(buf, f_s[7], f_e[7]) if same_chr else 0
        tlen = _atoi(buf, f_s[8], f_e[8])
        # full 64-bit collision-free streaming coordinate + template-span PCR/optical duplicate filter
        if len(dedup_tab) > 0 and paired and same_chr and tlen != 0 and mapq >= minmapq and (flag & 2) != 0:
            left_p = np.uint64((pos if tlen > 0 else pn) & 0xFFFFFFFF)
            span_u = np.uint64(abs(tlen) & 0xFFFFFFFF)
            side_u = np.uint64(0x9E3779B97F4A7C15) if tlen > 0 else np.uint64(0xBF58476D1CE4E5B9)
            dk = (mix64(mix64(np.uint64(c & 0xFFFFFFFF) ^ side_u) ^ mix64((left_p << np.uint64(32)) | span_u))
                  | np.uint64(1))
            ds = mix64(dk) & dmask
            if dedup_tab[ds] == dk:
                if len(stats) > 10:
                    stats[10] += 1
                if same_spot:
                    prev[0] = -1
                    prev[1] = -1
                    prev[2] = 0
                    prev[3] = 0
                i = e + 1
                continue
            dedup_tab[ds] = dk
        # 1 kb bins (MAPQ >= binmapq, accumulated after PCR duplicate removal when enabled)
        if mapq >= binmapq and rspan > 0:
            ce = coff[c + 1]
            a = g0
            b = min(g0 + rspan, ce)
            pa = a - coff[c]
            while a < b:
                bi = boff[c] + pa // 1000
                nxt = min(b, a + (1000 - pa % 1000))
                binbases[bi] += nxt - a
                pa += nxt - a
                a = nxt
        # face-away pair at a tandem-duplication junction: leftmost mate reverse, its mate forward
        if paired and mapq >= minmapq and (flag & 0x8) == 0 and (flag & 0x10) != 0 and (flag & 0x20) == 0 and same_chr:
            if pn > pos + 300 and pn < pos + 300000 and rf_n[0] < rf.shape[0]:
                rf[rf_n[0], 0] = c
                rf[rf_n[0], 1] = pos
                rf[rf_n[0], 2] = pn
                rf_n[0] += 1
        if mapq < minmapq or (paired and (flag & 2) == 0):
            if same_spot:
                prev[0] = -1
                prev[1] = -1
                prev[2] = 0
                prev[3] = 0
            i = e + 1
            continue
        ccount[c] += 1
        # mismatch filter from NM
        nm = -1
        j = tag_s
        while j + 5 < e:
            if buf[j] == 78 and buf[j + 1] == 77 and buf[j + 2] == 58 and buf[j + 3] == 105:
                q = j + 5
                while q < e and buf[q] != 9:
                    q += 1
                nm = _atoi(buf, j + 5, q)
                break
            while j < e and buf[j] != 9:
                j += 1
            j += 1
        if nm >= 0 and rspan > 0 and (nm - indel) > max_mm_frac * rspan:
            stats[5] += 1
            if same_spot:
                prev[0] = -1
                prev[1] = -1
                prev[2] = 0
                prev[3] = 0
            i = e + 1
            continue
        # mate overlap window to skip (second mate of a pair overlapping the first)
        ov_a = 0
        ov_b = 0
        if paired and same_spot and prev[1] == c:
            ov_a = max(prev[2], g0)
            ov_b = min(prev[3], g0 + rspan)
        prev[0] = spot
        prev[1] = c
        prev[2] = g0
        prev[3] = g0 + rspan
        # zoom capture
        if rspan > 0 and (zmask[g0] or zmask[min(g0 + rspan - 1, len(zmask) - 1)]):
            ln = f_e[9] - f_s[9]
            if zn[1] + ln < len(zbuf) and zn[0] < zmeta.shape[0]:
                zmeta[zn[0], 0] = g0
                zmeta[zn[0], 1] = zn[1]
                zmeta[zn[0], 2] = ln
                for t in range(ln):
                    zbuf[zn[1] + t] = buf[f_s[9] + t]
                zn[1] += ln
                zn[0] += 1
                stats[8] += 1
        # BAQ-style mask over query positions + QUAL == '*' handling (#24)
        qlen = f_e[9] - f_s[9]
        quallen = f_e[10] - f_s[10]
        qual_star = (quallen == 1 and buf[f_s[10]] == 42)
        if qlen <= 0 or (qlen == 1 and buf[f_s[9]] == 42) or (not qual_star and quallen != qlen):
            i = e + 1
            continue
        masked = use_mask and qlen <= 4096
        if masked:
            stats[9] += _base_mask(buf, f_s[5], f_e[5], f_s[9], qlen, g0, ref, flt, bmask, mmbuf)
        # indel boundary limits: exclude indels inside end_trim or soft-clip flanks, while preserving mask_indel_flank
        ind_lo = max(flt[2], lead_clip + (flt[1] if lead_clip > 0 else 0))
        ind_hi = min(qlen - flt[2], qlen - trail_clip - (flt[1] if trail_clip > 0 else 0))
        # walk CIGAR with bases
        qp = f_s[9]
        qq = f_s[10]
        g = g0
        num = 0
        counted = 0
        for j in range(f_s[5], f_e[5]):
            ch = buf[j]
            if ch >= 48 and ch <= 57:
                num = num * 10 + (ch - 48)
                continue
            if ch == 77 or ch == 61 or ch == 88:  # M = X
                for t in range(num):
                    gg = g + t
                    if gg >= ov_a and gg < ov_b:
                        stats[7] += 1
                        continue
                    if masked and bmask[qp + t - f_s[9]]:
                        continue
                    b = ENC_T[buf[qp + t]]
                    bq_val = 30 if qual_star else (buf[qq + t] - 33)
                    if b < 4 and bq_val >= minbq:
                        if counts[gg, b] < 65535:
                            counts[gg, b] += 1
                        counted += 1
                qp += num
                if not qual_star:
                    qq += num
                g += num
            elif ch == 73:  # I
                qi = qp - f_s[9]
                bq_ok = qi >= ind_lo and (qi + num) <= ind_hi
                if bq_ok:
                    for t in range(num):
                        bq_val = 30 if qual_star else (buf[qq + t] - 33)
                        if ENC_T[buf[qp + t]] > 3 or bq_val < minbq:
                            bq_ok = False
                            break
                if bq_ok and not (g - 1 >= ov_a and g - 1 < ov_b) and g > coff[c]:
                    if insc[g - 1] < 65535:
                        insc[g - 1] += 1
                    if ev_n[0] < evcap:
                        code = np.uint64(0)
                        if num <= 31:
                            for t in range(num):
                                code = (code << np.uint64(2)) | np.uint64(ENC_T[buf[qp + t]] & 3)
                            ic = np.int64(code & np.uint64(0x7FFFFFFFFFFFFFFF))
                        else:
                            code = np.uint64(1469598103934665603)
                            for t in range(num):
                                code = (code ^ np.uint64(buf[qp + t])) * np.uint64(1099511628211)
                            ic = np.int64(code & np.uint64(0x7FFFFFFFFFFFFFFF))
                            if ins_n[0] < ins_meta.shape[0] and ins_n[1] + num <= len(ins_buf):
                                if ins_n[0] == 0 or ins_meta[ins_n[0] - 1, 0] != ic:
                                    ins_meta[ins_n[0], 0] = ic
                                    ins_meta[ins_n[0], 1] = ins_n[1]
                                    ins_meta[ins_n[0], 2] = num
                                    for t in range(num):
                                        ins_buf[ins_n[1] + t] = buf[qp + t]
                                    ins_n[1] += num
                                    ins_n[0] += 1
                        ev[ev_n[0], 0] = g - 1
                        ev[ev_n[0], 1] = 2
                        ev[ev_n[0], 2] = num
                        ev[ev_n[0], 3] = ic
                        ev_n[0] += 1
                    else:
                        stats[6] += 1
                qp += num
                if not qual_star:
                    qq += num
            elif ch == 68:  # D
                qi = qp - f_s[9]
                bq_ok = qi >= ind_lo and qi <= ind_hi
                if bq_ok and not qual_star:
                    if qi > 0 and buf[qq - 1] - 33 < minbq:
                        bq_ok = False
                    if qi < qlen and buf[qq] - 33 < minbq:
                        bq_ok = False
                if bq_ok and not (g >= ov_a and g < ov_b):
                    for t in range(num):
                        if delc[g + t] < 65535:
                            delc[g + t] += 1
                    if ev_n[0] < evcap:
                        ev[ev_n[0], 0] = g
                        ev[ev_n[0], 1] = 1
                        ev[ev_n[0], 2] = num
                        ev[ev_n[0], 3] = 0
                        ev_n[0] += 1
                    else:
                        stats[6] += 1
                g += num
            elif ch == 78:
                g += num
            elif ch == 83:  # S
                qp += num
                qq += num
            num = 0
        stats[2] += 1
        stats[3] += counted
        i = e + 1
    return i


# ----------------------------------------------------------------------------------------------- helpers
@njit(cache=True)
def runs_ge(depth, thr, offsets):
    """intervals [start, end) of depth >= thr that never cross contig boundaries."""
    n_runs = 0
    for c in range(len(offsets) - 1):
        a = offsets[c]
        b = offsets[c + 1]
        inrun = False
        for g in range(a, b):
            if depth[g] >= thr:
                if not inrun:
                    inrun = True
                    n_runs += 1
            elif inrun:
                inrun = False
    out = np.empty((n_runs, 2), np.int64)
    if n_runs == 0:
        return out
    k = 0
    for c in range(len(offsets) - 1):
        a = offsets[c]
        b = offsets[c + 1]
        inrun = False
        s = a
        for g in range(a, b):
            if depth[g] >= thr:
                if not inrun:
                    inrun = True
                    s = g
            elif inrun:
                inrun = False
                out[k, 0] = s
                out[k, 1] = g
                k += 1
        if inrun:
            out[k, 0] = s
            out[k, 1] = b
            k += 1
    return out[:k]



@njit(cache=True)
def best_diagonals(gseq, zmeta, zbuf, max_shift):
    """per captured read: reference start (alignment start + best shift) and identity on that diagonal."""
    n = zmeta.shape[0]
    start = np.zeros(n, np.int64)
    ident = np.zeros(n, np.float64)
    L = len(gseq)
    for r in range(n):
        g0 = zmeta[r, 0]
        o = zmeta[r, 1]
        ln = zmeta[r, 2]
        best = -1.0
        bd = 0
        for d in range(-max_shift, max_shift + 1):
            a = g0 + d
            if a < 0 or a + ln > L:
                continue
            m = 0
            for i in range(ln):
                if gseq[a + i] == ENC_T[zbuf[o + i]]:
                    m += 1
            f = m / max(ln, 1)
            if f > best:
                best = f
                bd = d
        start[r] = g0 + bd
        ident[r] = best
    return start, ident
