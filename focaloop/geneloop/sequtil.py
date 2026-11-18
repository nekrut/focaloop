"""Genome sequence access, translation and ORF utilities."""

from __future__ import annotations

from pathlib import Path
import threading

from .locus_zoom import _CODON_TABLE, STOP_CODONS

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_CANONICAL_SPLICE = {"GT-AG", "GC-AG", "AT-AC"}


def _cds_introns_canonical(
    genome: Genome,
    contig: str,
    strand: str,
    cds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    """True when all internal introns of ``cds`` carry canonical splice motifs on ``strand``."""
    blocks = sorted(cds)
    for ( _, a), (b, _) in zip(blocks, blocks[1:]):
        if b - a < 4:
            return False
        d = genome.fetch(contig, a, a + 2).upper()
        ac = genome.fetch(contig, b - 2, b).upper()
        motif = f"{d}-{ac}" if strand != "-" else f"{revcomp(ac)}-{revcomp(d)}"
        if motif not in _CANONICAL_SPLICE:
            return False
    return True


def _edge_trunc_flags(
    genome: Genome,
    contig: str,
    strand: str,
    blocks: list[tuple[int, int]],
    edge_margin: int,
) -> tuple[bool, bool]:
    """Return ``(trunc_5p, trunc_3p)`` indicating whether ``blocks`` reach contig boundaries within ``edge_margin``."""
    L = genome.lengths.get(contig, 0)
    at_left = blocks[0][0] <= edge_margin
    at_right = L > 0 and blocks[-1][1] >= L - edge_margin
    return (at_left, at_right) if strand != "-" else (at_right, at_left)


def _both_trunc_open_span(u: str, min_cds_bp: int) -> tuple[int, int] | None:
    """Return ``(off, end_i)`` of the first valid open frame across a both-truncated spliced CDS, or None."""
    n = len(u)
    if n % 3 == 0 and n >= min_cds_bp and orf_status(u)["internal_stops"] == 0:
        return 0, n
    for off in range(3):
        end_i = n - ((n - off) % 3)
        sub = u[off:end_i]
        if len(sub) >= min_cds_bp and all(sub[i : i + 3] not in STOP_CODONS for i in range(0, len(sub), 3)):
            return off, end_i
    return None


def is_scaffold_edge_partial_cds(
    genome: Genome,
    contig: str,
    strand: str,
    cds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    min_cds_bp: int = 150,
    min_exons: int = 2,
    cds_phase: int = 0,
) -> bool:
    """True for clean multi-exon coding models truncated directly at contig boundaries (start<=3 or end>=L-3)."""
    blocks = sorted(cds)
    if len(blocks) < min_exons:
        return False
    if sum(b - a for a, b in blocks) < min_cds_bp:
        return False
    u = genome.spliced(contig, strand, blocks).upper()
    rem = len(u) % 3
    edge_margin = 5 if rem == 0 else 3
    trunc_5p, trunc_3p = _edge_trunc_flags(genome, contig, strand, blocks, edge_margin)
    if not (trunc_5p or trunc_3p):
        return False
    if not _cds_introns_canonical(genome, contig, strand, blocks):
        return False
    ph = int(cds_phase or 0) % 3
    if ph != 0:
        if not trunc_5p:
            return False
        u_ph = u[ph:]
        rem_ph = len(u_ph) % 3
        if not trunc_3p:
            return (
                rem_ph == 0
                and len(u_ph) >= min_cds_bp
                and u_ph[-3:] in STOP_CODONS
                and all(u_ph[i : i + 3] not in STOP_CODONS for i in range(0, len(u_ph) - 3, 3))
            )
        if rem_ph == 0 and len(u_ph) >= min_cds_bp and orf_status(u_ph)["internal_stops"] == 0:
            return True
        sub = u_ph[: len(u_ph) - rem_ph]
        return len(sub) >= min_cds_bp and all(sub[i : i + 3] not in STOP_CODONS for i in range(0, len(sub), 3))
    if trunc_5p and trunc_3p:
        return _both_trunc_open_span(u, min_cds_bp) is not None
    if rem == 0:
        st = orf_status(u)
        return (
            st["internal_stops"] == 0
            and (st["has_start"] or trunc_5p)
            and (st["has_stop"] or trunc_3p)
        )
    if trunc_3p and not trunc_5p:
        sub = u[: len(u) - rem]
        if len(sub) < min_cds_bp or sub[:3] != "ATG":
            return False
        return all(sub[i : i + 3] not in STOP_CODONS for i in range(0, len(sub), 3))
    sub = u[rem:]
    if len(sub) < min_cds_bp or sub[-3:] not in STOP_CODONS:
        return False
    return all(sub[i : i + 3] not in STOP_CODONS for i in range(0, len(sub) - 3, 3))


def trim_partial_edge_cds(
    genome: Genome,
    contig: str,
    strand: str,
    cds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    cds_phase: int = 0,
    min_cds_bp: int = 150,
    min_exons: int = 2,
) -> list[tuple[int, int]] | None:
    """Trim partial-codon overhang (and non-zero GTF phase) from a valid scaffold-edge partial CDS to frame=0."""
    blocks = sorted(cds)
    if not is_scaffold_edge_partial_cds(
        genome, contig, strand, blocks, min_cds_bp=min_cds_bp, min_exons=min_exons, cds_phase=cds_phase
    ):
        return None
    u = genome.spliced(contig, strand, blocks).upper()
    n = len(u)
    rem = n % 3
    ph = int(cds_phase or 0) % 3
    edge_margin = 5 if rem == 0 else 3
    trunc_5p, trunc_3p = _edge_trunc_flags(genome, contig, strand, blocks, edge_margin)
    if ph != 0:
        end_i = n - ((n - ph) % 3)
        out = project_tx_interval(blocks, strand, ph, end_i)
    elif trunc_5p and trunc_3p:
        span = _both_trunc_open_span(u, min_cds_bp)
        if span is None:
            return None
        out = blocks if span == (0, n) else project_tx_interval(blocks, strand, span[0], span[1])
    elif rem == 0:
        out = blocks
    elif trunc_3p and not trunc_5p:
        out = project_tx_interval(blocks, strand, 0, n - rem)
    else:
        out = project_tx_interval(blocks, strand, rem, n)
    if len(out) < min_exons:
        return None
    return out if is_scaffold_edge_partial_cds(
        genome, contig, strand, out, min_cds_bp=min_cds_bp, min_exons=min_exons, cds_phase=0
    ) else None


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def translate(nt: str) -> str:
    u = nt.upper()
    return "".join(_CODON_TABLE.get(u[i : i + 3], "X") for i in range(0, len(u) - 2, 3))


class Genome:
    """Indexed FASTA (pysam.FastaFile) with a tiny per-contig cache for synthetic tests."""

    def __init__(self, fasta: str | Path | None = None, seqs: dict[str, str] | None = None):
        self._seqs = seqs
        self._fa = None
        self._lock = threading.Lock()
        if fasta is not None:
            import pysam

            self._fa = pysam.FastaFile(str(fasta))
            self.lengths = dict(zip(self._fa.references, self._fa.lengths))
        else:
            self.lengths = {k: len(v) for k, v in (seqs or {}).items()}

    def fetch(self, contig: str, a: int, b: int) -> str:
        a = max(0, a)
        b = min(self.lengths.get(contig, b), b)
        if b <= a:
            return ""
        if self._seqs is not None:
            return self._seqs[contig][a:b]
        with self._lock:
            return self._fa.fetch(contig, a, b)

    def spliced(self, contig: str, strand: str, blocks: list[tuple[int, int]]) -> str:
        s = "".join(self.fetch(contig, a, b) for a, b in sorted(blocks))
        return revcomp(s) if strand == "-" else s


def orf_status(cds_nt: str) -> dict:
    """Validity of a CDS that should include its stop codon."""
    u = cds_nt.upper()
    n = len(u)
    has_start = u[:3] == "ATG"
    has_stop = n >= 3 and u[-3:] in STOP_CODONS
    internal = 0
    for i in range(0, n - 3, 3):
        if u[i : i + 3] in STOP_CODONS:
            internal += 1
    return {
        "len": n,
        "frame_ok": n % 3 == 0,
        "has_start": has_start,
        "has_stop": has_stop,
        "internal_stops": internal,
        "valid": has_start and has_stop and internal == 0 and n % 3 == 0 and n >= 6,
    }


def longest_orf(nt: str, min_aa: int = 50, require_atg: bool = True) -> tuple[int, int] | None:
    """Longest ATG..stop ORF on the given strand of ``nt``; returns (start, end incl stop) or None."""
    u = nt.upper()
    n = len(u)
    best = None
    best_len = 0
    for frame in range(3):
        start = None
        i = frame
        while i + 3 <= n:
            cod = u[i : i + 3]
            if start is None:
                if cod == "ATG" or (not require_atg and i == frame):
                    start = i
            if cod in STOP_CODONS:
                if start is not None:
                    ln = i + 3 - start
                    if ln > best_len:
                        best_len = ln
                        best = (start, i + 3)
                start = None
            i += 3
    if best is None or best_len // 3 - 1 < min_aa:
        return None
    return best


def best_open_orf(nt: str) -> tuple[int, int, int, int]:
    """Longest contiguous non-stop codon run across frames 0..2 (open 5'/3' ends) -> (aa_len, frame, tx_start, tx_end)."""
    u = nt.upper()
    n = len(u)
    best_aa = 0
    best_frame = 0
    best_s = 0
    best_e = 0
    for frame in range(3):
        run = 0
        run_s = frame
        i = frame
        while i + 3 <= n:
            if u[i : i + 3] in STOP_CODONS:
                if run > best_aa:
                    best_aa, best_frame, best_s, best_e = run, frame, run_s, i
                run = 0
                run_s = i + 3
            else:
                run += 1
            i += 3
        if run > best_aa:
            best_aa, best_frame, best_s, best_e = run, frame, run_s, run_s + 3 * run
    return best_aa, best_frame, best_s, best_e


def max_open_orf_aa(nt: str) -> int:
    """Maximum contiguous non-stop codon run across frames 0..2, allowing open 5' and 3' ends."""
    return best_open_orf(nt)[0]


def project_tx_interval(blocks: list[tuple[int, int]], strand: str, t0: int, t1: int) -> list[tuple[int, int]]:
    """Map transcript-coordinate interval [t0, t1) (5'->3' of the transcript) to genomic blocks."""
    segs = sorted(blocks)
    order = segs if strand != "-" else list(reversed(segs))
    out = []
    pos = 0
    for a, b in order:
        ln = b - a
        s0, s1 = max(t0, pos), min(t1, pos + ln)
        if s0 < s1:
            if strand != "-":
                out.append((a + (s0 - pos), a + (s1 - pos)))
            else:
                out.append((b - (s1 - pos), b - (s0 - pos)))
        pos += ln
    return sorted(out)


def genomic_to_tx(blocks: list[tuple[int, int]], strand: str, g: int) -> int | None:
    """Transcript offset of genomic base g (None if g not in blocks)."""
    segs = sorted(blocks)
    order = segs if strand != "-" else list(reversed(segs))
    pos = 0
    for a, b in order:
        if a <= g < b:
            return pos + (g - a if strand != "-" else b - 1 - g)
        pos += b - a
    return None


def _extend_to_stop(
    genome: Genome,
    contig: str,
    strand: str,
    blocks: list[list[int]],
    max_ext: int,
    lo_bound: int,
    hi_bound: int,
) -> bool:
    """Extend ``blocks`` downstream in-frame up to ``max_ext`` bp to hit a stop codon; mutates ``blocks`` on success."""
    for _step in range(0, max_ext, 3):
        if strand != "-":
            a = blocks[-1][1]
            if a + 3 > hi_bound:
                return False
            cod = genome.fetch(contig, a, a + 3).upper()
            if len(cod) < 3:
                return False
            blocks[-1][1] = a + 3
        else:
            a = blocks[0][0]
            if a - 3 < lo_bound:
                return False
            cod = revcomp(genome.fetch(contig, a - 3, a)).upper()
            if len(cod) < 3:
                return False
            blocks[0][0] = a - 3
        if cod in STOP_CODONS:
            return True
    return False


def _repair_start(
    genome: Genome,
    contig: str,
    strand: str,
    blocks: list[list[int]],
    seq: str,
    max_ext: int,
    lo_bound: int,
    hi_bound: int,
    partial_edge: bool = False,
) -> list[list[int]] | None:
    """Ensure ``blocks`` begins with an in-frame ATG by upstream extension or short forward trim."""
    if seq[:3].upper() == "ATG":
        return blocks
    found = False
    hit_open_edge = False
    L = genome.lengths.get(contig, 0)
    if strand != "-":
        a = blocks[0][0]
        for step in range(3, max_ext + 3, 3):
            if a - step < lo_bound:
                if lo_bound <= 3:
                    hit_open_edge = True
                break
            cod = genome.fetch(contig, a - step, a - step + 3).upper()
            if len(cod) < 3 or cod in STOP_CODONS:
                break
            if cod == "ATG":
                blocks[0][0] = a - step
                found = True
                break
    else:
        b = blocks[-1][1]
        for step in range(3, max_ext + 3, 3):
            if b + step > hi_bound:
                if L > 0 and hi_bound >= L - 3:
                    hit_open_edge = True
                break
            cod = revcomp(genome.fetch(contig, b + step - 3, b + step)).upper()
            if len(cod) < 3 or cod in STOP_CODONS:
                break
            if cod == "ATG":
                blocks[-1][1] = b + step
                found = True
                break
    if found:
        return blocks
    s = seq.upper()
    pos = None
    for i in range(3, min(len(s) - 3, 180), 3):
        if s[i : i + 3] == "ATG":
            pos = i
            break
    if pos is None:
        return None
    at_5p_edge = (blocks[0][0] <= 3) if strand != "-" else (L > 0 and blocks[-1][1] >= L - 3)
    if partial_edge and hit_open_edge and at_5p_edge and pos > 3:
        return None
    tx_blocks = [tuple(b) for b in blocks]
    return [list(b) for b in project_tx_interval(tx_blocks, strand, pos, len(s))]


def _repair_edge_partial_orf(
    genome: Genome,
    contig: str,
    strand: str,
    orig_blocks: list[list[int]],
    max_ext: int,
    lo_bound: int,
    hi_bound: int,
) -> list[tuple[int, int]] | None:
    """Validate/repair a multi-exon partial coding chain truncated directly at contig boundary."""
    blocks = [list(b) for b in orig_blocks]
    if len(blocks) < 2 or sum(b - a for a, b in blocks) < 150:
        return None
    L = genome.lengths.get(contig, 0)
    at_left = blocks[0][0] <= 3 and lo_bound <= 3
    at_right = L > 0 and blocks[-1][1] >= L - 3 and hi_bound >= L - 3
    trunc_5p = at_left if strand != "-" else at_right
    trunc_3p = at_right if strand != "-" else at_left
    if not (trunc_5p or trunc_3p):
        return None
    if not _cds_introns_canonical(genome, contig, strand, [tuple(b) for b in blocks]):
        return None

    seq = genome.spliced(contig, strand, [tuple(b) for b in blocks])
    rem = len(seq) % 3
    if trunc_3p and not trunc_5p:
        if rem:
            if strand != "-":
                if blocks[-1][1] - rem <= blocks[-1][0]:
                    return None
                blocks[-1][1] -= rem
            else:
                if blocks[0][0] + rem >= blocks[0][1]:
                    return None
                blocks[0][0] += rem
            seq = genome.spliced(contig, strand, [tuple(b) for b in blocks])
        repaired = _repair_start(genome, contig, strand, blocks, seq, max_ext, lo_bound, hi_bound)
        if repaired is None:
            return None
        blocks = repaired
    elif trunc_5p and not trunc_3p:
        if rem:
            if strand != "-":
                if blocks[0][0] + rem >= blocks[0][1]:
                    return None
                blocks[0][0] += rem
            else:
                if blocks[-1][1] - rem <= blocks[-1][0]:
                    return None
                blocks[-1][1] -= rem
            seq = genome.spliced(contig, strand, [tuple(b) for b in blocks])
        if seq[-3:].upper() not in STOP_CODONS:
            if not _extend_to_stop(genome, contig, strand, blocks, max_ext, lo_bound, hi_bound):
                return None
    else:
        span = _both_trunc_open_span(seq.upper(), 150)
        if span is None:
            return None
        chosen = (
            [tuple(b) for b in blocks]
            if span == (0, len(seq))
            else project_tx_interval([tuple(b) for b in blocks], strand, span[0], span[1])
        )
        blocks = [list(b) for b in chosen]

    out = [tuple(b) for b in blocks]
    if len(out) < 2 or out[0][0] < lo_bound or (hi_bound > 0 and out[-1][1] > hi_bound):
        return None
    return out if is_scaffold_edge_partial_cds(genome, contig, strand, out) else None


def repair_orf(
    genome: Genome,
    contig: str,
    strand: str,
    cds: list[tuple[int, int]],
    max_ext: int = 600,
    bounds: tuple[int, int] | None = None,
    partial_edge: bool = False,
) -> list[tuple[int, int]] | None:
    """Extend a CDS chain to an in-frame ATG upstream and stop codon downstream (clamped to bounds when set)."""
    blocks = [list(b) for b in sorted(cds)]
    if not blocks:
        return None
    L = genome.lengths.get(contig, 0)
    lo_bound = 0 if bounds is None else max(0, int(bounds[0]))
    hi_bound = L if bounds is None else (min(L, int(bounds[1])) if L > 0 else int(bounds[1]))
    if bounds is not None and (blocks[0][0] < lo_bound or blocks[-1][1] > hi_bound):
        return None
    orig_blocks = [list(b) for b in blocks]
    seq = genome.spliced(contig, strand, [tuple(b) for b in blocks])
    if len(seq) % 3:
        return _repair_edge_partial_orf(genome, contig, strand, orig_blocks, max_ext, lo_bound, hi_bound) if partial_edge else None
    if seq[-3:].upper() not in STOP_CODONS:
        if not _extend_to_stop(genome, contig, strand, blocks, max_ext, lo_bound, hi_bound):
            return _repair_edge_partial_orf(genome, contig, strand, orig_blocks, max_ext, lo_bound, hi_bound) if partial_edge else None
    seq = genome.spliced(contig, strand, [tuple(b) for b in blocks])
    repaired_start = _repair_start(genome, contig, strand, blocks, seq, max_ext, lo_bound, hi_bound, partial_edge=partial_edge)
    if repaired_start is None:
        return _repair_edge_partial_orf(genome, contig, strand, orig_blocks, max_ext, lo_bound, hi_bound) if partial_edge else None
    out = [tuple(b) for b in repaired_start]
    if bounds is not None and (not out or out[0][0] < lo_bound or out[-1][1] > hi_bound):
        return None
    st = orf_status(genome.spliced(contig, strand, out))
    if st["valid"]:
        return out
    return _repair_edge_partial_orf(genome, contig, strand, orig_blocks, max_ext, lo_bound, hi_bound) if partial_edge else None
