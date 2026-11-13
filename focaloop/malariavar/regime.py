"""Stage 1 regimes: synthetic paired reads with known origin on the 80.6% AT Pf3D7 genome.

err05        0.5% substitution errors at random core positions
at_indel     1-3 bp indels inside A/T homopolymers (>= 8 bp) and (AT)n tandem repeats
minor20      polyclonal mixture: 20% of pairs from a haplotype carrying SNPs (WSAF 0.20 at those sites)
adapter      inserts shorter than the read so reads run into Illumina adapter sequence
paralog      reads from a core segment duplicated exactly in a decoy contig (mate-tie symmetry gate)

Read names carry the truth: <regime>:<id>:<contig>:<leftmost goff of mate1 genomic part>:<same for mate2>:<extra>
"""
import numpy as np

from . import genome as G

ADAPTER1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCACATCTCGTATGCCGTCTTCTGCTTG"
ADAPTER2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCTCGGTGGTCGCCGTATCATT"
PARALOG_NAME = "PARALOG_decoy"
COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(COMP)[::-1]


def qual(n, rng):
    q = rng.choice(np.frombuffer(b"FFFJJJJJJJ<A", np.uint8), n)
    return q.tobytes().decode()


def mutate(s, err, rng):
    if err <= 0:
        return s
    a = np.frombuffer(s.encode(), np.uint8).copy()
    hit = np.where(rng.random(len(a)) < err)[0]
    for i in hit:
        choices = [c for c in b"ACGT" if c != a[i]]
        a[i] = choices[rng.integers(0, 3)]
    return a.tobytes().decode()


def paralog_segment(g, length=6000, seed=5):
    """a core segment (no N, inside one contig) duplicated as the decoy contig."""
    rng = np.random.default_rng(seed)
    core = np.where(g.core)[0]
    while True:
        a = int(core[rng.integers(0, len(core))])
        c, _ = g.locate(np.array([a]))
        if g.offsets[c[0] + 1] - a > length + 10 and g.core[a:a + length].all() and (g.seq[a:a + length] < 4).all():
            return a, a + length


def tournament_reference(g, path):
    a, b = paralog_segment(g)
    with open(path, "w") as f:
        for i, n in enumerate(g.names):
            f.write(f">{n}\n")
            s = g.bases(int(g.offsets[i]), int(g.offsets[i + 1]))
            for j in range(0, len(s), 80):
                f.write(s[j:j + 80] + "\n")
        f.write(f">{PARALOG_NAME}\n")
        s = g.bases(a, b)
        for j in range(0, len(s), 80):
            f.write(s[j:j + 80] + "\n")
    return a, b


def _pair(g, hap, a, ins, rl, err, rng):
    """fragment [a, a+ins) of haplotype string hap (ref coords == hap coords for SNP-only haplotypes)."""
    frag = hap[a:a + ins]
    r1 = frag[:rl]
    r2 = rc(frag)[:rl]
    if ins < rl:  # adapter read-through
        r1 = (frag + ADAPTER1)[:rl]
        r2 = (rc(frag) + ADAPTER2)[:rl]
    return mutate(r1, err, rng), mutate(r2, err, rng)


def sample_starts(g, n, rng, span, avoid=None):
    core = np.where(g.core)[0]
    out = []
    while len(out) < n:
        a = int(core[rng.integers(0, len(core))])
        if a + span >= len(g.seq) or not g.core[a:a + span].all() or (g.seq[a:a + span] > 3).any():
            continue
        c, _ = g.locate(np.array([a, a + span]))
        if c[0] != c[1]:
            continue
        if avoid is not None and avoid[0] - span <= a < avoid[1]:
            continue
        out.append(a)
    return out


def records(g, regime, n_pairs, seed, rl=150, avoid=None):
    """list of (name, r1, q1, r2, q2) for one regime."""
    rng = np.random.default_rng(seed)
    out = []
    ref = None
    if regime in ("err05", "adapter", "minor20"):
        ref = g.bases(0, g.L)
    if regime == "err05":
        for i, a in enumerate(sample_starts(g, n_pairs, rng, 700, avoid)):
            ins = int(np.clip(rng.normal(320, 50), 200, 600))
            r1, r2 = _pair(g, ref, a, ins, rl, 0.005, rng)
            out.append((f"err05:{i}:{a}:{a + ins - rl}:0", r1, r2))
    elif regime == "adapter":
        for i, a in enumerate(sample_starts(g, n_pairs, rng, 700, avoid)):
            ins = int(rng.integers(60, 140))
            r1, r2 = _pair(g, ref, a, ins, rl, 0.002, rng)
            out.append((f"adapter:{i}:{a}:{a}:{ins}", r1, r2))
    elif regime == "minor20":
        starts = sample_starts(g, 10, rng, 2000, avoid)
        sites = []
        for a in starts:
            for k in range(5):
                x = a + 300 + 280 * k
                b = int(g.seq[x])
                alt = (b + 1 + int(rng.integers(0, 3))) % 4
                sites.append((x, alt))
        hap = list(ref)
        for x, alt in sites:
            hap[x] = "ACGT"[alt]
        hapb = "".join(hap)
        per = max(1, n_pairs // len(starts))
        i = 0
        for a in starts:
            for _ in range(per):
                s0 = a + int(rng.integers(0, 2000 - 650))
                ins = int(np.clip(rng.normal(320, 40), 220, 600))
                h = 1 if rng.random() < 0.2 else 0
                r1, r2 = _pair(g, hapb if h else ref, s0, ins, rl, 0.002, rng)
                out.append((f"minor20:{i}:{s0}:{s0 + ins - rl}:{h}", r1, r2))
                i += 1
        return out, sites
    elif regime == "at_indel":
        seq = g.seq
        events = []
        core = np.where(g.core)[0]
        while len(events) < n_pairs // 4:
            x = int(core[rng.integers(0, len(core))])
            b = seq[x]
            if b not in (0, 3):
                continue
            j = x
            while j < len(seq) and seq[j] == b:
                j += 1
            hp = j - x
            at = all(seq[x + 2 * k] == seq[x] and seq[x + 2 * k + 1] == 3 - seq[x] for k in range(6))
            if hp >= 8 or at:
                if not g.core[x - 800:x + 800].all() or (seq[x - 800:x + 800] > 3).any():
                    continue
                if avoid is not None and avoid[0] - 1600 <= x < avoid[1] + 800:
                    continue
                L = int(rng.integers(1, 4))
                kind = "del" if rng.random() < 0.5 else "ins"
                events.append((x, kind, L, hp if hp >= 8 else 12))
        i = 0
        for x, kind, L, span in events:
            lo = x - 700
            window = g.bases(lo, x + 900)
            off = x - lo + span // 2
            if kind == "del":
                hapw = window[:off] + window[off + L:]
            else:
                hapw = window[:off] + "ACGT"[g.seq[x]] * L + window[off:]
            for _ in range(4):
                ins = int(np.clip(rng.normal(320, 40), 220, 520))
                s0 = int(rng.integers(max(0, off - ins + 40), off - 40))
                frag = hapw[s0:s0 + ins]
                r1 = mutate(frag[:rl], 0.002, rng)
                r2 = mutate(rc(frag)[:rl], 0.002, rng)
                p2 = lo + s0 + ins - rl - (L if kind == "ins" and s0 + ins - rl > off else 0) + \
                    (L if kind == "del" and s0 + ins - rl > off else 0)
                out.append((f"at_indel:{i}:{lo + s0}:{p2}:{kind}{L}@{x}", r1, r2))
                i += 1
    elif regime == "paralog":
        a, b = avoid
        ref = g.bases(a, b)
        for i in range(n_pairs):
            ins = int(np.clip(rng.normal(320, 40), 220, 520))
            s0 = int(rng.integers(0, (b - a) - ins))
            r1, r2 = _pair(g, ref, s0, ins, rl, 0.002, rng)
            out.append((f"paralog:{i}:{a + s0}:{a + s0 + ins - rl}:{s0}", r1, r2))
    return out, None


def write_fastq(recs, p1, p2, seed=0):
    rng = np.random.default_rng(seed)
    with open(p1, "w") as f1, open(p2, "w") as f2:
        for name, r1, r2 in recs:
            f1.write(f"@{name}\n{r1}\n+\n{qual(len(r1), rng)}\n")
            f2.write(f"@{name}\n{r2}\n+\n{qual(len(r2), rng)}\n")


def genome():
    return G.load()
