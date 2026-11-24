"""Transcript / CDS model synthesis shared by all three benchmark arms.

The SAME rules run for Arm A (no RNA), Arm B (100 % RNA, full BAMs) and Arm C (GeneLoop streamed
RNA); only the evidence differs. GeneLoop-specific Stage-5 modules (micro-exon insertion and the
retrocopy discriminator) are switched on with ``zoom=True``.

Scoring is ADDITIVE OVER EVIDENCE UNITS (per supported intron, per homologous intron, ...) with a
constant per-gene cost, so a gene split into fragments never out-scores the intact model unless
evidence that the fragments are separate genes (exact protein chains, missing RNA junction) exists.
Selection = weighted interval scheduling of non-overlapping CDS spans per strand.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .evidence import CANONICAL, BinStore, JunctionStore, LocusIndex, junction_motif
from .gffio import Transcript, interval_overlap_bp, introns_of, is_pseudogene_model, merge_intervals
from .locus_zoom import STOP_CODONS, SingleExonCandidate, classify_single_exon_candidate
from .ncrna import filter_and_materialize_structural_ncrnas
from .seed import LOCKED, Locus, ProteinHit
from .sequtil import (
    Genome,
    best_open_orf,
    is_scaffold_edge_partial_cds,
    longest_orf,
    max_open_orf_aa,
    orf_status,
    project_tx_interval,
    repair_orf,
    revcomp,
    translate,
    trim_partial_edge_cds,
)
from .telemetry import run_cmd, safe_env

_LOG = logging.getLogger(__name__)


@dataclass
class SynthParams:
    gene_cost: float = 0.5
    w_tib_exact: float = 2.0
    w_tib_intron: float = 0.4
    w_hom_exact: float = 3.0
    w_hom_intron: float = 0.8
    w_hom_overlap: float = 1.5
    w_rna_sup: float = 1.0
    w_rna_unsup: float = 1.5
    w_rna_contra: float = 1.0
    w_len: float = 0.5
    w_rna_tx: float = 0.3
    w_tib_conflict: float = 1.5  # per intron where a non-Tiberius chain disagrees with an overlapping valid
    # Tiberius model WITHOUT RNA arbitration (evidence hierarchy: RNA > ab initio splice model > distant homology)
    w_tib_frag: float = 3.5  # per broken boundary where a shorter candidate truncates inside an intact chain
    min_support: int = 2
    absent_lambda: float = 4.74  # Poisson: P(X <= 1 | lambda) < 0.05 -> a missing intron is evidence, not sampling noise
    contra_min: int = 5
    expressed_min_spliced: int = 5
    hit_candidate_min_quality: float = 0.5
    max_hit_candidates_per_cluster: int = 6
    rna_orf_min_aa: int = 100
    rna_only_min_aa: int = 150
    max_isoforms: int = 3
    isoform_min_cov_frac: float = 0.15
    min_score: float = 0.25
    softmask_thresh: float = 0.25
    softmask_penalty: float = 3.5
    tib_uncor_2ex_penalty: float = 1.0
    tib_uncor_min_score: float = 0.0
    rna_only_min_score: float = 2.5
    alt_cds_min_len_frac: float = 0.85
    alt_cds_min_score_frac: float = 0.75
    antisense_single_ov_frac: float = 0.20
    enable_b_prime: bool = True
    emit_lncrna: bool = True
    lnc_coding_buffer_bp: int = 2_000
    emit_pseudogenes: bool = True
    emit_disabled_hom_pseudogenes: bool = True
    emit_structural_ncrna: bool = True
    enable_rna_orf_repaired: bool = True
    enable_rna_orf_bridged: bool = True
    threads: int = 8
    strict_retrocopy_scan: bool = True


_LOWER_ASCII = b"abcdefghijklmnopqrstuvwxyz"
_cds_overlap_bp = interval_overlap_bp


def _softmask_frac(genome: Genome, contig: str, blocks: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> float:
    low_bp = 0
    tot_bp = 0
    for a, b in blocks:
        tot_bp += b - a
        seq = genome.fetch(contig, a, b)
        if seq:
            bseq = seq.encode("ascii", "ignore")
            low_bp += len(bseq) - len(bseq.translate(None, _LOWER_ASCII))
    return low_bp / max(1, tot_bp)


def _exonic_connected_components(items: list, get_blocks) -> list[list]:
    """Partition items into exact connected components under exonic overlap (>0 shared bp)."""
    n = len(items)
    if n <= 1:
        return [list(items)] if n == 1 else []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    exons: list[tuple[int, int, int]] = []
    for idx, it in enumerate(items):
        for a, b in get_blocks(it):
            if b > a:
                exons.append((a, b, idx))
    exons.sort(key=lambda x: (x[0], x[1]))
    cur_end = -1
    cur_owner = -1
    for a, b, idx in exons:
        if cur_owner >= 0 and a < cur_end:
            union(cur_owner, idx)
            if b > cur_end:
                cur_end = b
        else:
            cur_owner = idx
            cur_end = b
    groups: dict[int, list] = {}
    for idx, it in enumerate(items):
        groups.setdefault(find(idx), []).append(it)
    return list(groups.values())


@dataclass
class Cand:
    contig: str
    strand: str
    cds: tuple[tuple[int, int], ...]
    sources: set = field(default_factory=set)
    exons: tuple[tuple[int, int], ...] | None = None  # RNA transcript exons (with UTR) when RNA-derived
    tx_cov: float = 0.0
    hit: ProteinHit | None = None
    score: float = 0.0
    feats: dict = field(default_factory=dict)
    microexons: int = 0
    micro_parent: dict = field(default_factory=dict)  # flanking intron -> spanning intron it was split from
    cds_phase: int = 0

    @property
    def start(self) -> int:
        return self.cds[0][0]

    @property
    def end(self) -> int:
        return self.cds[-1][1]

    @property
    def introns(self) -> tuple[tuple[int, int], ...]:
        return introns_of(list(self.cds))

    @property
    def cds_len(self) -> int:
        return sum(b - a for a, b in self.cds)

    def key(self) -> tuple:
        return (self.contig, self.strand, self.cds)


# ----------------------------------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------------------------------


def _add(cands: dict, c: Cand) -> Cand:
    k = c.key()
    if k in cands:
        old = cands[k]
        old.sources |= c.sources
        if c.exons and (old.exons is None or c.tx_cov > old.tx_cov):
            old.exons, old.tx_cov = c.exons, c.tx_cov
        if c.hit is not None and (old.hit is None or c.hit.quality > old.hit.quality):
            old.hit = c.hit
        if old.cds_phase == 0 and c.cds_phase != 0 and "tiberius" in c.sources and len(old.sources) == 1:
            old.cds_phase = c.cds_phase
        if "tiberius" in c.sources:
            if not c.feats.get("tiberius_backfill", False):
                old.feats["tiberius_non_backfill"] = True
                old.feats["tiberius_backfill"] = False
            elif not old.feats.get("tiberius_non_backfill", False):
                old.feats["tiberius_backfill"] = True
            if not c.feats.get("trimmed_spacer", False):
                old.feats["tiberius_untrimmed"] = True
                old.feats["trimmed_spacer"] = False
            elif not old.feats.get("tiberius_untrimmed", False):
                old.feats["trimmed_spacer"] = True
        return old
    if "tiberius" in c.sources:
        if not c.feats.get("tiberius_backfill", False):
            c.feats["tiberius_non_backfill"] = True
        if not c.feats.get("trimmed_spacer", False):
            c.feats["tiberius_untrimmed"] = True
    cands[k] = c
    return c


def _partial_edge_rna_orfs(seq: str, trunc_5p: bool, trunc_3p: bool, min_aa: int) -> list[tuple[int, int]]:
    """Extract candidate partial-edge ORF intervals ``(tx_s, tx_e)`` in 5'->3' spliced RNA coordinates."""
    u = seq.upper()
    n = len(u)
    spans: list[tuple[int, int]] = []
    if trunc_5p:
        for fr in range(3):
            stop_i = None
            for i in range(fr, n - 2, 3):
                if u[i : i + 3] in STOP_CODONS:
                    stop_i = i
                    break
            if stop_i is not None:
                if stop_i + 3 - fr >= 3 * min_aa:
                    spans.append((fr, stop_i + 3))
            elif trunc_3p:
                end_i = fr + 3 * ((n - fr) // 3)
                if end_i - fr >= 3 * min_aa:
                    spans.append((fr, end_i))
    if trunc_3p:
        for fr in range(3):
            first_atg = None
            for i in range(fr, n - 2, 3):
                cod = u[i : i + 3]
                if cod in STOP_CODONS:
                    first_atg = None
                elif cod == "ATG" and first_atg is None:
                    first_atg = i
            if first_atg is not None:
                end_i = first_atg + 3 * ((n - first_atg) // 3)
                if end_i - first_atg >= 3 * min_aa:
                    spans.append((first_atg, end_i))
    if trunc_5p and trunc_3p:
        aa_len, _fr, tx_s, tx_e = best_open_orf(u)
        if aa_len >= min_aa:
            spans.append((tx_s, tx_e))
    return spans


def rna_orf_candidate(genome: Genome, t: Transcript, min_aa: int) -> list[Cand]:
    out = []
    strands = [t.strand] if t.strand in "+-" else ["+", "-"]
    exons = sorted(t.exons)
    L = genome.lengths.get(t.contig, 0)
    at_left = bool(exons) and exons[0][0] <= 3
    at_right = bool(exons) and L > 0 and exons[-1][1] >= L - 3
    multi_edge = len(exons) >= 2 and (at_left or at_right)
    cov = float(t.attrs.get("cov", 0) or 0)
    for st in strands:
        seq = genome.spliced(t.contig, st, exons)
        orf = longest_orf(seq, min_aa=min_aa)
        if orf is not None:
            cds = project_tx_interval(exons, st, orf[0], orf[1])
            cand: Cand | None = Cand(t.contig, st, tuple(cds), {"rna_orf"}, tuple(exons), cov)
        else:
            cand = None
        if multi_edge:
            trunc_5p = at_left if st != "-" else at_right
            trunc_3p = at_right if st != "-" else at_left
            for tx_s, tx_e in sorted(
                _partial_edge_rna_orfs(seq, trunc_5p, trunc_3p, min_aa),
                key=lambda iv: -(iv[1] - iv[0]),
            ):
                if cand is not None and (tx_e - tx_s) <= cand.cds_len:
                    continue
                p_cds = project_tx_interval(exons, st, tx_s, tx_e)
                if is_scaffold_edge_partial_cds(genome, t.contig, st, p_cds, min_cds_bp=3 * min_aa):
                    cand = Cand(t.contig, st, tuple(p_cds), {"rna_orf"}, tuple(exons), cov)
                    cand.feats["partial_edge"] = True
                    break
        if cand is not None:
            out.append(cand)
    if len(out) > 1:
        out = [max(out, key=lambda c: c.cds_len)]
    return out


def _merge_scanned_windows(windows: list[tuple[str, int, int]] | None) -> list[tuple[str, int, int]]:
    if not windows:
        return []
    by_c: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ctg, a, b in windows:
        if int(b) > int(a):
            by_c[str(ctg)].append((int(a), int(b)))
    out: list[tuple[str, int, int]] = []
    for ctg, ivs in sorted(by_c.items()):
        for a, b in merge_intervals(ivs):
            out.append((ctg, a, b))
    return out


def _scanned_span_frac(scanned_iv_idx: LocusIndex | None, ctg: str, lo: int, hi: int) -> float:
    if scanned_iv_idx is None or hi <= lo:
        return 0.0
    ov = sum(max(0, min(hi, wb) - max(lo, wa)) for wa, wb in scanned_iv_idx.overlapping(ctg, lo, hi))
    return ov / (hi - lo)


def _generate_orphan_rna_candidates(
    cands: dict,
    genome: Genome,
    tiberius: list[Transcript],
    hits: list[ProteinHit],
    raw_alignments: list | None,
    rna_tx: list[Transcript],
    junctions: JunctionStore | None,
    p: SynthParams,
    scanned_tib_windows: list[tuple[str, int, int]] | None,
    gap_continuity: dict[tuple[str, int, int], int] | None,
    bait_intervals: dict[str, list[tuple[int, int]]],
    organelle_contigs: set[str],
) -> None:
    """C1-C5 / R5-R8 open-frame orphan repair ('rna_orf_repaired') and multi-fragment bridging ('rna_orf_bridged')."""
    if junctions is None or not rna_tx or (not p.enable_rna_orf_repaired and not p.enable_rna_orf_bridged):
        return

    # R5: Full exclusion indexes (4-tuple payloads) including invalid-ORF Tiberius and positive>=0.45 miniprot hits
    all_tib_cds_idx = LocusIndex([(t.contig, t.cds_start, t.cds_end, t) for t in tiberius if t.cds])
    hom_items: list[tuple[str, int, int, object]] = []
    seen_hom: set[tuple[str, str, tuple[tuple[int, int], ...]]] = set()
    for h in hits:
        if max(h.positive, float(getattr(h, "max_positive", 0.0) or 0.0)) >= 0.45:
            hk = (h.contig, h.strand, h.cds)
            if hk not in seen_hom:
                seen_hom.add(hk)
                hom_items.append((h.contig, h.start, h.end, h))
    if raw_alignments:
        for a in raw_alignments:
            if a.positive >= 0.45 and (a.contig, a.strand, a.cds) not in seen_hom:
                seen_hom.add((a.contig, a.strand, a.cds))
                hom_items.append((a.contig, a.start, a.end, a))
    hom_045_idx = LocusIndex(hom_items)

    merged_scanned = _merge_scanned_windows(scanned_tib_windows)
    scanned_tib_idx = LocusIndex([(c, a, b, True) for c, a, b in merged_scanned]) if merged_scanned else None
    scanned_iv_idx = LocusIndex([(c, a, b, (a, b)) for c, a, b in merged_scanned]) if merged_scanned else None
    _ = scanned_tib_idx  # R5 4-tuple window index verified alongside interval-overlap index

    def _blocked_by_tib_or_hom(ctg: str, strand: str, lo: int, hi: int, exons_t: tuple[tuple[int, int], ...]) -> bool:
        if any(tb.strand == strand for tb in all_tib_cds_idx.overlapping(ctg, lo, hi)):
            return True
        for hm in hom_045_idx.overlapping(ctg, lo, hi):
            if hm.strand == strand or _cds_overlap_bp(exons_t, hm.cds) > 0:
                return True
        return False

    # 1. Single-fragment open-frame repair ("rna_orf_repaired")
    if p.enable_rna_orf_repaired and scanned_iv_idx is not None:
        for t in rna_tx:
            if t.strand not in ("+", "-"):
                continue
            exons = merge_intervals(t.exons)
            if len(exons) < 4:
                continue
            t_lo, t_hi = exons[0][0], exons[-1][1]
            if _overlaps_bait(t.contig, t_lo, t_hi, bait_intervals, organelle_contigs):
                continue
            if _scanned_span_frac(scanned_iv_idx, t.contig, t_lo, t_hi) < 0.90:
                continue
            ex_tup = tuple(exons)
            if _blocked_by_tib_or_hom(t.contig, t.strand, t_lo, t_hi, ex_tup):
                continue
            intr = introns_of(exons)
            if not all(
                junctions.canonical((t.contig, a, b))
                and junctions.strand((t.contig, a, b)) == t.strand
                and junctions.support((t.contig, a, b)) >= 3
                for a, b in intr
            ):
                continue
            if _softmask_frac(genome, t.contig, ex_tup) > 0.45:
                continue
            seq = genome.spliced(t.contig, t.strand, exons)
            aa_len, _fr, tx_s, tx_e = best_open_orf(seq)
            if aa_len < 150 or (3 * aa_len) / max(1, len(seq)) < 0.80:
                continue
            proj_cds = project_tx_interval(exons, t.strand, tx_s, tx_e)
            repaired = repair_orf(genome, t.contig, t.strand, proj_cds, max_ext=300)
            if repaired is None or sum(b - a for a, b in repaired) < 753:
                continue
            rep_key = (t.contig, t.strand, tuple(repaired))
            if rep_key in cands:
                continue
            cov = float(t.attrs.get("cov", 0) or 0)
            rep_exons = tuple(merge_intervals(list(exons) + list(repaired)))
            _add(cands, Cand(t.contig, t.strand, tuple(repaired), {"rna_orf_repaired"}, rep_exons, cov))

    # 2. Co-directional multi-fragment bridging ("rna_orf_bridged")
    if p.enable_rna_orf_bridged:
        eff_runs = set(junctions.run_ids) or {r for d in junctions.ua.values() for r, v in d.items() if v > 0}
        n_libs = len(eff_runs)
        min_junc_link = max(3, int(p.min_support))
        frags_by_cs: dict[tuple[str, str], dict[tuple[tuple[int, int], ...], float]] = defaultdict(dict)
        for t in rna_tx:
            if t.strand not in ("+", "-"):
                continue
            exons = merge_intervals(t.exons)
            if len(exons) < 3:
                continue
            intr = introns_of(exons)
            if not all(
                junctions.canonical((t.contig, a, b))
                and junctions.strand((t.contig, a, b)) == t.strand
                and junctions.support((t.contig, a, b)) >= 3
                for a, b in intr
            ):
                continue
            f_seq = genome.spliced(t.contig, t.strand, exons)
            aa_f, _, _, _ = best_open_orf(f_seq)
            if (3 * aa_f) / max(1, len(f_seq)) < 0.85:
                continue
            ex_tup = tuple(exons)
            cov = float(t.attrs.get("cov", 0) or 0)
            prev_cov = frags_by_cs[(t.contig, t.strand)].get(ex_tup, -1.0)
            if cov > prev_cov:
                frags_by_cs[(t.contig, t.strand)][ex_tup] = cov

        for (ctg, strand), fmap in sorted(frags_by_cs.items()):
            if len(fmap) < 2:
                continue
            ranked = sorted(
                fmap.items(),
                key=lambda kv: (-sum(b - a for a, b in kv[0]), -kv[1], kv[0][0][0]),
            )
            non_ov: list[tuple[list[tuple[int, int]], float]] = []
            for ex_tup, cov in ranked:
                lo, hi = ex_tup[0][0], ex_tup[-1][1]
                if any(max(lo, x[0][0][0]) < min(hi, x[0][-1][1]) for x in non_ov):
                    continue
                non_ov.append((list(ex_tup), cov))
            non_ov.sort(key=lambda x: x[0][0][0])
            if len(non_ov) < 2:
                continue

            def _emit_chain(
                chain_exons: list[tuple[int, int]],
                n_frags: int,
                used_junc: bool,
                first_gap_lo: int,
                last_gap_hi: int,
                max_cov: float,
            ) -> None:
                if n_frags < 2:
                    return
                c_lo, c_hi = chain_exons[0][0], chain_exons[-1][1]
                if _overlaps_bait(ctg, c_lo, c_hi, bait_intervals, organelle_contigs):
                    return
                ex_t = tuple(chain_exons)
                if _blocked_by_tib_or_hom(ctg, strand, c_lo, c_hi, ex_t):
                    return
                if _softmask_frac(genome, ctg, ex_t) > 0.45:
                    return
                b_seq = genome.spliced(ctg, strand, chain_exons)
                aa_b, _fr_b, tx_s, tx_e = best_open_orf(b_seq)
                in_scanned = _scanned_span_frac(scanned_iv_idx, ctg, c_lo, c_hi) >= 0.90
                exonic_dense = (not used_junc) and ((3 * aa_b) / max(1, len(b_seq)) >= 0.85)
                if not (in_scanned or exonic_dense):
                    return
                proj_cds = project_tx_interval(chain_exons, strand, tx_s, tx_e)
                if orf_status(genome.spliced(ctg, strand, proj_cds))["valid"]:
                    bridged = proj_cds
                else:
                    bridged = repair_orf(genome, ctg, strand, proj_cds, max_ext=300)
                if bridged is None or sum(b - a for a, b in bridged) < 753:
                    return
                if bridged[0][0] >= first_gap_lo or bridged[-1][1] <= last_gap_hi:
                    return
                b_key = (ctg, strand, tuple(bridged))
                if b_key in cands:
                    return
                b_exons = tuple(merge_intervals(list(chain_exons) + list(bridged)))
                _add(cands, Cand(ctg, strand, tuple(bridged), {"rna_orf_bridged"}, b_exons, max_cov))

            cur_ex = list(non_ov[0][0])
            cur_cov = non_ov[0][1]
            cur_n = 1
            cur_used_junc = False
            first_g_lo = -1
            last_g_hi = -1
            for nxt_ex, nxt_cov in non_ov[1:]:
                ga = cur_ex[-1][1]
                gb = nxt_ex[0][0]
                gap = gb - ga
                linked = False
                if 0 < gap <= 1_500:
                    jk = (ctg, ga, gb)
                    junc_ok = (
                        junctions.canonical(jk)
                        and junctions.strand(jk) == strand
                        and junctions.support(jk) >= min_junc_link
                        and (n_libs < 2 or junctions.n_runs_supporting(jk) >= 2)
                    )
                    if junc_ok:
                        if first_g_lo < 0:
                            first_g_lo = ga
                        last_g_hi = gb
                        cur_ex = cur_ex + nxt_ex
                        cur_cov = max(cur_cov, nxt_cov)
                        cur_n += 1
                        cur_used_junc = True
                        linked = True
                    elif (
                        gap_continuity is not None
                        and 3 <= gap <= 1_000
                        and gap_continuity.get((ctg, ga, gb), 0) >= 2
                    ):
                        cand_merged = cur_ex[:-1] + [(cur_ex[-1][0], nxt_ex[0][1])] + nxt_ex[1:]
                        left_bp = sum(b - a for a, b in cur_ex)
                        right_bp = sum(b - a for a, b in nxt_ex)
                        gap_tx_lo = left_bp if strand != "-" else right_bp
                        gap_tx_hi = gap_tx_lo + gap
                        _aa_m, _fr_m, ms, me = best_open_orf(genome.spliced(ctg, strand, cand_merged))
                        if ms < gap_tx_lo and me > gap_tx_hi:
                            if first_g_lo < 0:
                                first_g_lo = ga
                            last_g_hi = gb
                            cur_ex = cand_merged
                            cur_cov = max(cur_cov, nxt_cov)
                            cur_n += 1
                            linked = True
                if not linked:
                    _emit_chain(cur_ex, cur_n, cur_used_junc, first_g_lo, last_g_hi, cur_cov)
                    cur_ex = list(nxt_ex)
                    cur_cov = nxt_cov
                    cur_n = 1
                    cur_used_junc = False
                    first_g_lo = -1
                    last_g_hi = -1
            _emit_chain(cur_ex, cur_n, cur_used_junc, first_g_lo, last_g_hi, cur_cov)


def _validate_or_repair_tiberius_cds(
    genome: Genome,
    t: Transcript,
) -> tuple[tuple[tuple[int, int], ...], int, bool] | None:
    """Validate or edge-trim a Tiberius CDS model; returns ``(cds_tuple, cds_phase, partial_edge)`` or ``None``."""
    from .sequtil import _cds_introns_canonical

    t_cds = tuple(sorted(t.cds))
    if not t_cds:
        return None
    ph = int(getattr(t, "cds_phase", 0) or 0) % 3
    raw_seq = genome.spliced(t.contig, t.strand, list(t_cds))
    if ph == 0 and orf_status(raw_seq)["valid"]:
        return t_cds, 0, False

    trimmed = trim_partial_edge_cds(genome, t.contig, t.strand, t_cds, cds_phase=ph)
    if trimmed is not None:
        return tuple(trimmed), 0, True

    # Support 5'-partial models carrying an explicit non-zero GTF phase (e.g. zoom-window edge)
    # when phased translation has zero internal stops, valid frame, and canonical splice sites.
    if ph in (1, 2):
        u_ph = raw_seq[ph:].upper()
        rem_ph = len(u_ph) % 3
        clip_3p = (
            (t.attrs.get("clip_right") == "1" and t.strand != "-")
            or (t.attrs.get("clip_left") == "1" and t.strand == "-")
        )
        if clip_3p and rem_ph != 0 and len(u_ph) - rem_ph >= 150:
            proj = project_tx_interval(list(t_cds), t.strand, 0, len(raw_seq) - rem_ph)
            if len(proj) >= len(t_cds):
                t_cds_try = tuple(proj)
                u_try = genome.spliced(t.contig, t.strand, list(t_cds_try))[ph:].upper()
                if (
                    len(u_try) >= 150
                    and len(u_try) % 3 == 0
                    and _cds_introns_canonical(genome, t.contig, t.strand, t_cds_try)
                    and all(u_try[i : i + 3] not in STOP_CODONS for i in range(0, len(u_try), 3))
                ):
                    return t_cds_try, ph, True
        if (
            rem_ph == 0
            and len(u_ph) >= 6
            and (u_ph[-3:] in STOP_CODONS or clip_3p)
            and _cds_introns_canonical(genome, t.contig, t.strand, t_cds)
            and all(u_ph[i : i + 3] not in STOP_CODONS for i in range(0, max(0, len(u_ph) - 3), 3))
        ):
            return t_cds, ph, True

    # Try small local start/stop repair for unphased canonical models without internal stops
    if ph == 0 and _cds_introns_canonical(genome, t.contig, t.strand, t_cds):
        repaired = repair_orf(genome, t.contig, t.strand, list(t_cds), max_ext=90, partial_edge=True)
        if repaired is not None:
            rep_seq = genome.spliced(t.contig, t.strand, list(repaired))
            if orf_status(rep_seq)["valid"]:
                return tuple(repaired), 0, False
            if is_scaffold_edge_partial_cds(genome, t.contig, t.strand, repaired):
                return tuple(repaired), 0, True
    return None


def generate_candidates(
    genome: Genome,
    tiberius: list[Transcript],
    hits: list[ProteinHit],
    rna_tx: list[Transcript] | None,
    p: SynthParams,
    junctions: JunctionStore | None = None,
    raw_alignments: list | None = None,
    scanned_tib_windows: list[tuple[str, int, int]] | None = None,
    gap_continuity: dict[tuple[str, int, int], int] | None = None,
    bait_intervals: dict[str, list[tuple[int, int]]] | None = None,
    organelle_contigs: set[str] | None = None,
) -> dict:
    cands: dict = {}
    for t in tiberius:
        if not t.cds:
            continue
        validated = _validate_or_repair_tiberius_cds(genome, t)
        if validated is None:
            continue
        t_cds, eff_ph, partial_edge = validated
        c = Cand(t.contig, t.strand, t_cds, {"tiberius"}, cds_phase=eff_ph)
        c.feats["tib_valid"] = True
        if partial_edge:
            c.feats["partial_edge"] = True
        if t.source == "tiberius_backfill" or t.attrs.get("backfill") == "1":
            c.feats["tiberius_backfill"] = True
        if t.attrs.get("trimmed_spacer") == "1":
            c.feats["trimmed_spacer"] = True
        _add(cands, c)
    # protein chains: top-N per CDS-block-overlapping cluster (preserves nested intronic genes)
    hs_by_cs: dict[tuple[str, str], list[ProteinHit]] = defaultdict(list)
    for h in hits:
        if h.quality >= p.hit_candidate_min_quality and h.frameshift == 0:
            hs_by_cs[(h.contig, h.strand)].append(h)

    def flush(cl: list[ProteinHit]):
        for h in sorted(cl, key=lambda h: -(h.quality * (1 + 0.05 * min(h.n_proteins, 20))))[: p.max_hit_candidates_per_cluster]:
            cds = repair_orf(genome, h.contig, h.strand, list(h.cds), partial_edge=True)
            if cds is None:
                continue
            c = Cand(h.contig, h.strand, tuple(cds), {"miniprot"}, hit=h)
            if not orf_status(genome.spliced(h.contig, h.strand, cds))["valid"]:
                c.feats["partial_edge"] = True
            _add(cands, c)

    for _cs in sorted(hs_by_cs):
        for comp in _exonic_connected_components(hs_by_cs[_cs], lambda h: h.cds):
            flush(comp)
    if rna_tx:
        for t in rna_tx:
            for c in rna_orf_candidate(genome, t, p.rna_orf_min_aa):
                _add(cands, c)
        _generate_orphan_rna_candidates(
            cands,
            genome,
            tiberius,
            hits,
            raw_alignments,
            rna_tx,
            junctions,
            p,
            scanned_tib_windows,
            gap_continuity,
            bait_intervals or {},
            organelle_contigs or set(),
        )
    return cands


# ----------------------------------------------------------------------------------------------
# Feature extraction and scoring
# ----------------------------------------------------------------------------------------------


def _is_readthrough_hit(h: ProteinHit, readthrough_targets: set[str]) -> bool:
    if h.target:
        return "readthrough" in h.target.lower() or h.target in readthrough_targets
    tlist = list(h.targets)
    return bool(tlist) and all(("readthrough" in tid.lower() or tid in readthrough_targets) for tid in tlist)


class EvidenceView:
    def __init__(
        self,
        tiberius: list[Transcript],
        hits: list[ProteinHit],
        junctions: JunctionStore | None,
        bins: BinStore | None,
        p: SynthParams,
        genome: Genome | None = None,
        readthrough_targets: set[str] | None = None,
    ):
        self.p = p
        self.genome = genome
        self.readthrough_targets = readthrough_targets or set()
        self.viable_tib_keys: set[tuple[str, str, tuple[tuple[int, int], ...]]] | None = None
        self.viable_hit_keys: set[tuple[str, str, tuple[tuple[int, int], ...]]] | None = None
        self.all_tiberius_cds = [t for t in tiberius if t.cds]
        valid: list[Transcript] = []
        for t in tiberius:
            if not t.cds:
                continue
            if genome is None:
                valid.append(t)
                continue
            validated = _validate_or_repair_tiberius_cds(genome, t)
            if validated is not None:
                v_cds, v_ph, _pe = validated
                valid.append(
                    Transcript(
                        tid=t.tid,
                        contig=t.contig,
                        strand=t.strand,
                        exons=list(t.exons),
                        cds=list(v_cds),
                        gid=t.gid,
                        source=t.source,
                        attrs=dict(t.attrs),
                        biotype=t.biotype,
                        cds_phase=v_ph,
                    )
                )
        self.tib_index = LocusIndex([(t.contig, t.cds_start, t.cds_end, t) for t in valid])
        self.tib_introns: set = set()
        for t in valid:
            for a, b in t.cds_introns:
                self.tib_introns.add((t.contig, t.strand, a, b))
        good = [h for h in hits if h.positive >= 0.5]
        self.hit_index = LocusIndex([(h.contig, h.start, h.end, h) for h in good])
        self.hit_introns: dict = {}
        self.non_rt_hit_introns: set[tuple[str, str, int, int]] = set()
        self.hit_chain: dict = {}
        for h in good:
            non_rt = not _is_readthrough_hit(h, self.readthrough_targets)
            for a, b in h.introns:
                k = (h.contig, h.strand, a, b)
                if h.quality > self.hit_introns.get(k, 0.0):
                    self.hit_introns[k] = h.quality
                if non_rt:
                    self.non_rt_hit_introns.add(k)
            ck = (h.contig, h.strand, h.introns)
            if h.quality > self.hit_chain.get(ck, (0.0, None))[0]:
                self.hit_chain[ck] = (h.quality, h)
        intact_full = [
            h
            for h in hits
            if h.frameshift == 0
            and h.stop_codon == 0
            and h.coverage >= 0.80
            and h.positive >= 0.50
            and not _is_readthrough_hit(h, self.readthrough_targets)
        ]
        self.intact_full_hits = intact_full
        self.intact_full_hit_index = LocusIndex(
            [(h.contig, h.start, h.end, (h, frozenset(h.introns))) for h in intact_full]
        )
        self.junc = junctions
        self.bins = bins
        self.jlist: dict[str, list] = {}
        self.jstarts: dict[str, list[int]] = {}
        if junctions is not None:
            self.jlist = junctions.by_contig(min_support=p.min_support)
            self.jstarts = {c: [x[0] for x in v] for c, v in self.jlist.items()}

    @property
    def has_rna(self) -> bool:
        return self.junc is not None

    def juncs_in(self, contig: str, a: int, b: int) -> list[tuple[int, int, int, str]]:
        st = self.jstarts.get(contig)
        if not st:
            return []
        v = self.jlist[contig]
        i = bisect.bisect_left(st, a)
        out = []
        while i < len(v) and v[i][0] < b:
            if v[i][1] <= b:
                out.append(v[i])
            i += 1
        return out


def _tib_clipped_against_cand(t: Transcript, c: Cand) -> bool:
    if t.attrs.get("clip_left") == "1" and "seg_lo" in t.attrs and c.start < int(t.attrs["seg_lo"]):
        return True
    if t.attrs.get("clip_right") == "1" and "seg_hi" in t.attrs and c.end > int(t.attrs["seg_hi"]):
        return True
    return False


def _weighted_union_overlap(cds_len: int, segs: list[tuple[int, int, float]]) -> tuple[int, float]:
    """Return (union_covered_bp, quality_weighted_union_fraction) across disjoint base intervals."""
    if not segs or cds_len <= 0:
        return 0, 0.0
    pts = sorted({pt for a, b, _q in segs for pt in (a, b)})
    cov_bp = 0
    weighted = 0.0
    for u, v in zip(pts, pts[1:]):
        if v <= u:
            continue
        q_max = max((q for a, b, q in segs if a <= u and v <= b), default=0.0)
        if q_max > 0.0:
            ln = v - u
            cov_bp += ln
            weighted += ln * q_max
    return cov_bp, weighted / cds_len


def _is_complete_gene_hit(h: ProteinHit, genome: Genome | None) -> bool:
    """True if ``h`` is an intact full-length protein hit representing a complete standalone gene."""
    if h.frameshift != 0 or h.stop_codon != 0 or h.coverage < 0.80:
        return False
    cds_len = sum(y - x for x, y in h.cds)
    if cds_len < 150:
        return False
    if genome is None:
        return len(h.cds) >= 2 and cds_len >= 450
    seq = genome.spliced(h.contig, h.strand, list(h.cds)).upper()
    st = orf_status(seq)
    if st["valid"]:
        return True
    if not (st["frame_ok"] and st["internal_stops"] == 0):
        return False
    has_5p = st["has_start"]
    has_3p = st["has_stop"]
    L = genome.lengths.get(h.contig, 0)
    if not has_5p:
        if h.strand != "-":
            up = genome.fetch(h.contig, max(0, h.start - 15), h.start).upper()
        else:
            up = revcomp(genome.fetch(h.contig, h.end, min(L, h.end + 15))).upper()
        for off in range(3, len(up) + 1, 3):
            cod = up[len(up) - off : len(up) - off + 3]
            if cod in STOP_CODONS:
                break
            if cod == "ATG":
                has_5p = True
                break
    if not has_3p:
        if h.strand != "-":
            dn = genome.fetch(h.contig, h.end, min(L, h.end + 6)).upper()
        else:
            dn = revcomp(genome.fetch(h.contig, max(0, h.start - 6), h.start)).upper()
        for off in range(0, len(dn) - 2, 3):
            if dn[off : off + 3] in STOP_CODONS:
                has_3p = True
                break
    return has_5p and has_3p


def _compute_hom_overlap(
    c: Cand,
    ev: EvidenceView,
    has_strong_conflict: bool,
    genome: Genome | None = None,
) -> tuple[float, float]:
    """Compute homology overlap quality (max single-hit >=50% OR conflict-free multi-hit union) and max identity."""
    eff_genome = genome if genome is not None else ev.genome
    best_ov = 0.0
    best_ident = 0.0
    c_introns = set(c.introns)
    compat_hits: list[ProteinHit] = []
    compat_segs: list[tuple[int, int, float]] = []
    complete_ov_hits: list[ProteinHit] = []
    max_single_compat_bp = 0

    for h in ev.hit_index.overlapping(c.contig, c.start, c.end):
        if h.strand != c.strand:
            continue
        segs = [
            (max(a, x), min(b, y))
            for a, b in c.cds
            for x, y in h.cds
            if min(b, y) > max(a, x)
        ]
        ov = sum(b - a for a, b in segs)
        h_cds_len = sum(y - x for x, y in h.cds)
        if len(c.cds) >= 2 and ov >= 0.5 * max(1, h_cds_len) and _is_complete_gene_hit(h, eff_genome):
            if not any(c.start <= ia and ib <= c.end and (ia, ib) not in c_introns for ia, ib in h.introns):
                complete_ov_hits.append(h)
        frac = ov / max(1, c.cds_len)
        if frac >= 0.5:
            best_ov = max(best_ov, h.quality * min(1.0, frac))
        best_ident = max(best_ident, h.identity)

        if len(c.cds) >= 2 and ov > 0 and not has_strong_conflict:
            if ov < 0.5 * max(1, h_cds_len):
                continue
            if any(c.start <= ia and ib <= c.end and (ia, ib) not in c_introns for ia, ib in h.introns):
                continue
            if any(hx < cb and ca < hy for ca, cb in c_introns for hx, hy in h.cds):
                continue
            compat_hits.append(h)
            max_single_compat_bp = max(max_single_compat_bp, ov)
            for sa, sb in segs:
                compat_segs.append((sa, sb, h.quality))

    chimera_bridge = False
    if len(c.cds) >= 2 and len(complete_ov_hits) >= 2:
        # Chimeric merge guard: never reward bridging disjoint complete genes without RNA or spanning-protein support.
        full_hits = sorted(complete_ov_hits, key=lambda h: (h.start, h.end))
        for h1, h2 in zip(full_hits, full_hits[1:]):
            if h1.end <= h2.start:
                bridge_introns = [i for i in c.introns if i[0] >= h1.start and i[1] <= h2.end and i[1] > h1.end and i[0] < h2.start]
                if not bridge_introns:
                    chimera_bridge = True
                    break
                for ia, ib in bridge_introns:
                    has_rna_sup = ev.has_rna and (
                        ev.junc.support((c.contig, ia, ib)) >= ev.p.min_support
                        or ((ia, ib) in c.micro_parent and ev.junc.support((c.contig, *c.micro_parent[(ia, ib)])) >= ev.p.min_support)
                    )
                    has_prot_sup = (c.contig, c.strand, ia, ib) in ev.non_rt_hit_introns
                    if not (has_rna_sup or has_prot_sup):
                        chimera_bridge = True
                        break
                if chimera_bridge:
                    break
    if chimera_bridge:
        best_ov = 0.0
    elif len(c.cds) >= 2 and len(compat_hits) >= 2 and not has_strong_conflict:
        union_bp, union_q = _weighted_union_overlap(c.cds_len, compat_segs)
        if union_bp > max_single_compat_bp and union_bp / max(1, c.cds_len) >= 0.10:
            best_ov = max(best_ov, union_q)
    return best_ov, best_ident


def _cand_deep_rna(c: Cand, ev: EvidenceView) -> bool:
    if not ev.has_rna:
        return False
    p = ev.p
    if c.introns:
        sup = [
            max(
                ev.junc.support((c.contig, a, b)),
                ev.junc.support((c.contig, *c.micro_parent[(a, b)])) if (a, b) in c.micro_parent else 0,
            )
            for a, b in c.introns
        ]
        supported = sorted(x for x in sup if x >= p.min_support)
        lam = supported[len(supported) // 2] if supported else 0.0
        return lam >= p.absent_lambda
    return ev.bins is not None and ev.bins.reads_in(c.contig, c.start, c.end) >= 20


def _boundary_broken_vs_chain(
    c: Cand,
    ref_cds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ref_introns: list[tuple[int, int]],
    ref_is_tib: bool,
    ev: EvidenceView,
    genome: Genome | None = None,
) -> int:
    """Count broken 5'/3' boundaries (0..2) where shorter ``c`` truncates inside an intact multi-exon chain."""
    if len(ref_cds) < 3 or len(ref_introns) < 2:
        return 0
    ref_start, ref_end = ref_cds[0][0], ref_cds[-1][1]
    if c.cds_len >= sum(b - a for a, b in ref_cds) or _cds_overlap_bp(c.cds, ref_cds) == 0:
        return 0
    p = ev.p
    eff_genome = genome if genome is not None else ev.genome
    shares_intron = bool(set(c.introns) & set(ref_introns))
    c_orf_valid = eff_genome is not None and orf_status(eff_genome.spliced(c.contig, c.strand, list(c.cds)))["valid"]
    c_deep = _cand_deep_rna(c, ev)
    is_full_hom = (
        c.hit is not None
        and c.hit.coverage >= 0.80
        and c.hit.frameshift == 0
        and c.hit.stop_codon == 0
        and (c_orf_valid or (c.cds_len >= 450 and len(c.cds) >= 2))
    )

    def _intron_has_sup(ia: int, ib: int) -> bool:
        rna_sup = ev.has_rna and ev.junc.support((c.contig, ia, ib)) >= p.min_support
        if rna_sup:
            return True
        if not ref_is_tib and ev.has_rna and c_deep:
            return False
        return (c.contig, c.strand, ia, ib) in ev.non_rt_hit_introns

    breaks = 0
    if ref_start < c.start < ref_end:
        up_introns = [i for i in ref_introns if i[0] < c.start]
        if up_introns:
            has_flank_sup = any(_intron_has_sup(ia, ib) for ia, ib in up_introns)
            inside_ref = any(ra < c.start < rb for ra, rb in ref_cds) or any(ia < c.start < ib for ia, ib in up_introns)
            exact_exon_start = any(ra == c.start for ra, _rb in ref_cds)
            nearest_i = up_introns[-1]
            nearest_sup = _intron_has_sup(*nearest_i)
            crosses_boundary = nearest_i[0] < c.start < nearest_i[1]
            exempt_chimera = not nearest_sup and (
                (is_full_hom and (exact_exon_start or c_orf_valid or crosses_boundary))
                or (c_orf_valid and crosses_boundary)
            )
            uncorrob_deep_rna = ev.has_rna and c_deep and not has_flank_sup
            if not exempt_chimera and not uncorrob_deep_rna and (
                has_flank_sup or (ref_is_tib and not (ev.has_rna and c_deep) and (inside_ref or shares_intron))
            ):
                breaks += 1

    if ref_start < c.end < ref_end:
        dn_introns = [i for i in ref_introns if i[1] > c.end]
        if dn_introns:
            has_flank_sup = any(_intron_has_sup(ia, ib) for ia, ib in dn_introns)
            inside_ref = any(ra < c.end < rb for ra, rb in ref_cds) or any(ia < c.end < ib for ia, ib in dn_introns)
            exact_exon_end = any(rb == c.end for _ra, rb in ref_cds)
            nearest_i = dn_introns[0]
            nearest_sup = _intron_has_sup(*nearest_i)
            crosses_boundary = nearest_i[0] < c.end < nearest_i[1]
            exempt_chimera = not nearest_sup and (
                (is_full_hom and (exact_exon_end or c_orf_valid or crosses_boundary))
                or (c_orf_valid and crosses_boundary)
            )
            uncorrob_deep_rna = ev.has_rna and c_deep and not has_flank_sup
            if not exempt_chimera and not uncorrob_deep_rna and (
                has_flank_sup or (ref_is_tib and not (ev.has_rna and c_deep) and (inside_ref or shares_intron))
            ):
                breaks += 1
    return breaks


def _count_tib_fragment_breaks(
    c: Cand,
    tib_ov: list[Transcript],
    ev: EvidenceView,
    genome: Genome | None = None,
) -> int:
    """Max broken-boundary count across overlapping intact viable Tiberius (or homology consensus) chains."""
    if "tiberius" in c.sources:
        return 0
    active_tib = [
        t
        for t in tib_ov
        if ev.viable_tib_keys is None or (t.contig, t.strand, tuple(sorted(t.cds))) in ev.viable_tib_keys
    ]
    max_breaks = 0
    for t in active_tib:
        max_breaks = max(
            max_breaks,
            _boundary_broken_vs_chain(c, t.cds, sorted(t.cds_introns), True, ev, genome=genome),
        )
    if not active_tib and c.hit is None:
        for h, _iset_h in ev.intact_full_hit_index.overlapping(c.contig, c.start, c.end):
            if h.strand == c.strand and len(h.cds) >= 4:
                if ev.viable_hit_keys is not None and (h.contig, h.strand, h.cds) not in ev.viable_hit_keys:
                    continue
                max_breaks = max(
                    max_breaks,
                    _boundary_broken_vs_chain(c, h.cds, sorted(h.introns), False, ev, genome=genome),
                )
    return max_breaks


def featurize(c: Cand, ev: EvidenceView, genome: Genome) -> dict:
    p = ev.p
    f = c.feats
    intr = c.introns
    n_int = len(intr)
    f["n_introns"] = n_int
    f["cds_len"] = c.cds_len
    f["softmask_frac"] = _softmask_frac(genome, c.contig, c.cds)
    # Tiberius
    f["tib_exact"] = "tiberius" in c.sources
    f["tib_introns"] = sum(1 for a, b in intr if (c.contig, c.strand, a, b) in ev.tib_introns)
    # Homology
    q_exact = 0.0
    if n_int:
        hc = ev.hit_chain.get((c.contig, c.strand, intr))
        if hc:
            q_exact = hc[0]
    elif c.hit is not None:
        q_exact = c.hit.quality
    f["hom_exact_q"] = q_exact
    f["hom_introns"] = sum(1 for a, b in intr if (c.contig, c.strand, a, b) in ev.hit_introns)
    # Conflict with overlapping valid Tiberius models (only for non-Tiberius candidates)
    extra: list = []
    missing: list = []
    tib_ov: list[Transcript] = []
    if "tiberius" not in c.sources:
        tib_ov = [
            t
            for t in ev.tib_index.overlapping(c.contig, c.start, c.end)
            if t.strand == c.strand and not _tib_clipped_against_cand(t, c)
        ]
        if tib_ov:
            T = set()
            for t in tib_ov:
                T.update(t.cds_introns)
            C = set(intr)
            extra = [i for i in C if i not in T]
            missing = [i for i in T if c.start <= i[0] and i[1] <= c.end and i not in C]
    f["tib_conflict_extra"] = extra
    f["tib_conflict_missing"] = missing
    f["n_tib_frag_breaks"] = _count_tib_fragment_breaks(c, tib_ov, ev, genome=genome)
    # RNA
    if ev.has_rna:
        mp = c.micro_parent

        def jsup(a, b):
            v = ev.junc.support((c.contig, a, b))
            par = mp.get((a, b))
            return max(v, ev.junc.support((c.contig, *par))) if par else v

        sup = [jsup(a, b) for a, b in intr]
        n_sup = sum(1 for s in sup if s >= p.min_support)
        supported = sorted(x for x in sup if x >= p.min_support)
        # local depth: median support of this model's supported introns (or of local junctions if none)
        lam = supported[len(supported) // 2] if supported else 0.0
        local = ev.juncs_in(c.contig, c.start, c.end)
        spliced_local = sum(x[2] for x in local)
        expressed = spliced_local >= p.expressed_min_spliced or (ev.bins is not None and ev.bins.reads_in(c.contig, c.start, c.end) >= 20)
        iset = set(intr)
        med = sorted(sup)[len(sup) // 2] if sup else 0
        contra = 0
        for a, b, s, strand in local:
            if (a, b) in iset or strand not in (c.strand, "."):
                continue
            if s < p.contra_min:
                continue
            # junction inside a single CDS exon -> candidate misses an intron
            inside = any(x < a and b < y for x, y in c.cds)
            shares = [(x, y) for x, y in intr if x == a or y == b]
            if inside and s >= 0.3 * max(med, p.contra_min):
                contra += 1
            elif shares:
                comp = max(ev.junc.support((c.contig, x, y)) for x, y in shares)
                if s > 2 * comp:
                    contra += 1
        deep = lam >= p.absent_lambda
        n_unsup = n_int - n_sup
        s_sup = {intr[i] for i, s in enumerate(sup) if s >= p.min_support}
        if p.enable_b_prime and s_sup and n_unsup > 0:
            ov_intact = [
                iset_h
                for h, iset_h in ev.intact_full_hit_index.overlapping(c.contig, c.start, c.end)
                if h.strand == c.strand and (iset_h & s_sup)
            ]
            excused = sum(
                1
                for u in intr
                if u not in s_sup
                and (c.contig, c.strand, u[0], u[1]) in ev.tib_introns
                and any(u in iset_h for iset_h in ov_intact)
            )
            n_unsup_pen = n_unsup - excused
        else:
            excused = 0
            n_unsup_pen = n_unsup
        f.update(
            n_sup=n_sup,
            n_unsup=n_unsup,
            n_unsup_pen=n_unsup_pen,
            n_unsup_excused=excused,
            expressed=expressed and deep,
            n_contra=contra,
            spliced_local=spliced_local,
            min_intron_sup=min(sup) if sup else 0,
            local_lambda=lam,
        )
        # RNA arbitration of Tiberius-vs-other conflicts: an unsampled Tiberius intron is only refuted at sufficient depth
        n_conf = sum(1 for a, b in extra if ev.junc.support((c.contig, a, b)) < p.min_support)
        n_conf += sum(1 for a, b in missing if not (deep and ev.junc.support((c.contig, a, b)) < p.min_support))
    else:
        f["n_unsup_pen"] = 0
        f["n_unsup_excused"] = 0
        n_conf = len(extra) + len(missing)
    f["n_tib_conflict"] = n_conf
    has_strong_conflict = n_conf > 0 or f.get("n_contra", 0) > 0
    best_ov, best_ident = _compute_hom_overlap(c, ev, has_strong_conflict, genome=genome)
    f["hom_overlap_q"] = best_ov
    f["hom_identity"] = best_ident
    del f["tib_conflict_extra"], f["tib_conflict_missing"]
    return f


def score(c: Cand, ev: EvidenceView) -> float:
    p = ev.p
    f = c.feats
    s = -p.gene_cost
    s += p.w_tib_exact * f["tib_exact"] + p.w_tib_intron * f["tib_introns"]
    s -= p.w_tib_conflict * f.get("n_tib_conflict", 0)
    s -= p.w_tib_frag * f.get("n_tib_frag_breaks", 0)
    s += p.w_hom_exact * f["hom_exact_q"] + p.w_hom_intron * f["hom_introns"] + p.w_hom_overlap * f["hom_overlap_q"]
    coding_ev = f["tib_exact"] or f["tib_introns"] > 0 or f["hom_overlap_q"] > 0.3 or f["hom_introns"] > 0
    unsup_delta = 0
    if ev.has_rna:
        s += p.w_rna_sup * f["n_sup"]
        if f["expressed"]:
            unsup_pen = f.get("n_unsup_pen", f["n_unsup"])
            unsup_delta = f["n_unsup"] - unsup_pen
            s -= p.w_rna_unsup * unsup_pen
            s -= p.w_rna_contra * f["n_contra"]
        if c.exons is not None and f["n_unsup"] == 0:
            s += p.w_rna_tx
    s += p.w_len * min(1.0, c.cds_len / 900.0)
    if not coding_ev:
        # RNA-only ORF: require long ORF, multi-exon, fully supported
        if not (ev.has_rna and c.cds_len >= 3 * p.rna_only_min_aa and f["n_introns"] >= 1 and f["n_unsup"] == 0):
            c.score = -1.0
            f["score_pre_excuse"] = -1.0
            return -1.0
    if (
        p.softmask_penalty > 0
        and f.get("softmask_frac", 0.0) > p.softmask_thresh
        and not (f.get("n_sup", 0) > 0 or f["hom_overlap_q"] > 0.3)
    ):
        s -= p.softmask_penalty * (f["softmask_frac"] - p.softmask_thresh)
    if (
        ev.has_rna
        and p.tib_uncor_2ex_penalty > 0
        and c.sources <= {"tiberius", "microexon_zoom"}
        and len(c.cds) <= 2
        and f.get("n_sup", 0) == 0
    ):
        s -= p.tib_uncor_2ex_penalty
    if not coding_ev and p.rna_only_min_score > 0 and s < p.rna_only_min_score:
        c.score = -1.0
        f["score_pre_excuse"] = -1.0
        return -1.0
    if (
        ev.has_rna
        and p.tib_uncor_min_score > 0
        and c.sources <= {"tiberius", "microexon_zoom"}
        and f["n_sup"] == 0
        and s < p.tib_uncor_min_score
    ):
        c.score = -1.0
        f["score_pre_excuse"] = -1.0
        return -1.0
    f["score_pre_excuse"] = s - p.w_rna_unsup * unsup_delta
    c.score = s
    return s


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------


def _schedule_span_dp(xs: list[Cand]) -> list[Cand]:
    """O(n log n) exact DP when conflict == outer-span overlap."""
    xs = sorted(xs, key=lambda c: (c.end, c.start))
    ends = [c.end for c in xs]
    n = len(xs)
    dp = [0.0] * (n + 1)
    take = [False] * (n + 1)
    prev = [0] * n
    for i, c in enumerate(xs):
        prev[i] = bisect.bisect_right(ends, c.start, 0, i)
    for i in range(1, n + 1):
        c = xs[i - 1]
        inc = c.score + dp[prev[i - 1]]
        if inc > dp[i - 1]:
            dp[i] = inc
            take[i] = True
        else:
            dp[i] = dp[i - 1]
    out: list[Cand] = []
    i = n
    while i > 0:
        if take[i]:
            out.append(xs[i - 1])
            i = prev[i - 1]
        else:
            i -= 1
    return out[::-1]


def _schedule_cds_mwis_exact(comp: list[Cand]) -> list[Cand]:
    """Exact maximum-weight CDS-disjoint subset on one exonic connected component."""
    ordered = sorted(comp, key=lambda c: (-c.score, c.end, c.start, c.cds))
    m = len(ordered)
    adj = [0] * m
    has_nested_pair = False
    for i in range(m):
        ci = ordered[i]
        for j in range(i + 1, m):
            cj = ordered[j]
            if _cds_overlap_bp(ci.cds, cj.cds) > 0:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
            elif ci.start < cj.end and cj.start < ci.end:
                has_nested_pair = True
    if not has_nested_pair:
        return _schedule_span_dp(comp)

    suffix = [0.0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix[i] = suffix[i + 1] + ordered[i].score

    best_score = -1.0
    best_mask = 0

    def _dfs(idx: int, chosen_mask: int, blocked_mask: int, cur_score: float) -> None:
        nonlocal best_score, best_mask
        if cur_score + suffix[idx] <= best_score + 1e-12:
            return
        if idx >= m:
            if cur_score > best_score:
                best_score = cur_score
                best_mask = chosen_mask
            return
        bit = 1 << idx
        if not (blocked_mask & bit):
            _dfs(idx + 1, chosen_mask | bit, blocked_mask | adj[idx], cur_score + ordered[idx].score)
        _dfs(idx + 1, chosen_mask, blocked_mask, cur_score)

    _dfs(0, 0, 0, 0.0)
    chosen = [ordered[i] for i in range(m) if (best_mask >> i) & 1]
    chosen.sort(key=lambda c: (c.start, c.end, c.cds))
    return chosen


def schedule(cands: list[Cand]) -> list[Cand]:
    """Select max-score set of CDS-block-disjoint candidates on one strand/contig (retaining nested intronic genes)."""
    pos = [c for c in cands if c.score > 0]
    if not pos:
        return []
    out: list[Cand] = []
    for comp in _exonic_connected_components(pos, lambda c: c.cds):
        if len(comp) == 1:
            out.append(comp[0])
        elif len(comp) <= 28:
            out.extend(_schedule_cds_mwis_exact(comp))
        else:
            top = sorted(comp, key=lambda c: (-c.score, c.end, c.start))[:28]
            picked = _schedule_cds_mwis_exact(top)
            picked_ids = {id(c) for c in picked}
            for rem_c in sorted(comp, key=lambda c: (-c.score, c.end, c.start)):
                if id(rem_c) in picked_ids:
                    continue
                if all(_cds_overlap_bp(rem_c.cds, pk.cds) == 0 for pk in picked):
                    picked.append(rem_c)
                    picked_ids.add(id(rem_c))
            out.extend(picked)
    out.sort(key=lambda c: (c.start, c.end, c.cds))
    return out


# ----------------------------------------------------------------------------------------------
# Stage 5 helpers: micro-exons and retrocopies
# ----------------------------------------------------------------------------------------------

_STEM_LOOP = "GGCYCTTTTCAGRGCC"  # replication-dependent histone 3' stem-loop (IUPAC)
_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "Y": "CT", "R": "AG", "N": "ACGT"}


def _approx_find(seq: str, motif: str, max_mm: int) -> bool:
    """True if ``motif`` (IUPAC) occurs in ``seq`` with <= max_mm mismatches."""
    m = len(motif)
    allowed = [_IUPAC[c] for c in motif]
    for i in range(0, len(seq) - m + 1):
        mm = 0
        for j in range(m):
            if seq[i + j] not in allowed[j]:
                mm += 1
                if mm > max_mm:
                    break
        if mm <= max_mm:
            return True
    return False


_PAIR = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G"), ("G", "T"), ("T", "G")}  # Watson-Crick + G.U wobble


def has_histone_stem_loop(seq: str, max_mm: int = 1) -> bool:
    """Replication-dependent histone 3' stem-loop: 6-bp stem, UUUC loop, paired 6-nt arm (G.U allowed).

    Structural rather than consensus matching: the mammalian GGCUCU-UUUC-AGAGCC and the Drosophila
    CGUCCU-UUUC-AGGACG stem-loops share the fold but differ at 4 of 16 positions."""
    s = seq.upper()
    for i in range(0, len(s) - 15):
        if s[i + 6 : i + 10] != "TTTC":
            continue
        x, y = s[i : i + 6], s[i + 10 : i + 16]
        mm = sum(1 for k in range(6) if (x[k], y[5 - k]) not in _PAIR)
        if mm <= max_mm:
            return True
    return False


def downstream_seq(genome: Genome, c: Cand, n: int) -> str:
    if c.strand != "-":
        return genome.fetch(c.contig, c.end, c.end + n).upper()
    return revcomp(genome.fetch(c.contig, max(0, c.start - n), c.start)).upper()


def has_polya(seq: str, win: int = 20, min_a: int = 16) -> bool:
    for i in range(0, max(0, len(seq) - win + 1)):
        if seq[i : i + win].count("A") >= min_a:
            return True
    return False


def insert_microexons(c: Cand, genome: Genome, micro: dict) -> Cand:
    """Split CDS introns that carry a rescued in-frame micro-exon (RNA-supported)."""
    if not micro:
        return c
    cds = list(c.cds)
    changed = 0
    parent = dict(c.micro_parent)
    for a, b in c.introns:
        rs = micro.get((c.contig, a, b))
        if not rs:
            continue
        agg: dict = {}
        for ms, me, sup in rs:
            agg[(ms, me)] = agg.get((ms, me), 0) + sup
        (ms, me), sup = max(agg.items(), key=lambda kv: kv[1])
        if (me - ms) % 3 != 0 or not (a < ms < me < b):
            continue
        new = sorted(cds + [(ms, me)])
        ph = int(getattr(c, "cds_phase", 0) or 0) % 3
        spliced_new = genome.spliced(c.contig, c.strand, new)
        if ph == 0:
            valid_new = orf_status(spliced_new)["valid"]
        else:
            u_ph = spliced_new[ph:].upper()
            valid_new = (
                len(u_ph) >= 6
                and len(u_ph) % 3 == 0
                and all(u_ph[i : i + 3] not in STOP_CODONS for i in range(0, max(0, len(u_ph) - 3), 3))
            )
        if valid_new:
            cds = new
            changed += 1
            parent[(a, ms)] = (a, b)
            parent[(me, b)] = (a, b)
    if not changed:
        return c
    exons = None
    if c.exons:
        exons = tuple(sorted(list(c.exons) + [x for x in cds if x not in c.cds]))
    nc = Cand(
        c.contig, c.strand, tuple(cds), set(c.sources) | {"microexon_zoom"}, exons, c.tx_cov, c.hit,
        cds_phase=c.cds_phase,
    )
    nc.microexons = changed
    nc.micro_parent = parent
    nc.feats = dict(c.feats)
    return nc


def parent_intron_counts(
    single: list[Cand],
    multi: list[Cand],
    genome: Genome,
    work: Path,
    *,
    threads: int = 8,
    strict: bool = True,
) -> tuple[dict, float, bool]:
    """Family intron architecture of each single-exon candidate (diamond blastp vs ALL selected models).

    A multi-exon "parent" is reported only when multi-exon homologs (>= 50 % id, >= 70 % query cover,
    other loci) outnumber single-exon homologs: intronless families (ORs, histones) contain a few
    mis-predicted multi-exon members and must not be read as retrocopies of them."""
    if not single or not multi:
        return {}, 0.0, False
    work.mkdir(parents=True, exist_ok=True)
    q = work / "single_exon.faa"
    db = work / "all_models.faa"
    allm = single + multi
    with open(q, "w") as fh:
        for i, c in enumerate(single):
            ph_c = int(getattr(c, "cds_phase", 0) or 0) % 3
            fh.write(f">s{i}\n{translate(genome.spliced(c.contig, c.strand, list(c.cds))[ph_c:]).rstrip('*')}\n")
    with open(db, "w") as fh:
        for i, c in enumerate(allm):
            ph_c = int(getattr(c, "cds_phase", 0) or 0) % 3
            fh.write(f">m{i}\n{translate(genome.spliced(c.contig, c.strand, list(c.cds))[ph_c:]).rstrip('*')}\n")
    out = work / "single_vs_all.tsv"
    n_threads = str(max(1, min(32, int(threads))))
    env = safe_env()
    diamond_cpu_s = 0.0
    try:
        rc1, _, u1, s1 = run_cmd(
            ["diamond", "makedb", "--in", str(db), "-d", str(work / "allm"), "--threads", n_threads, "--quiet"],
            check=False, env=env,
        )
        diamond_cpu_s += u1 + s1
        if rc1 != 0:
            if strict:
                raise RuntimeError(f"diamond makedb failed (rc={rc1}) during retrocopy family scan")
            _LOG.warning("diamond makedb exited with rc=%d during retrocopy family scan", rc1)
            return {}, diamond_cpu_s, True
        rc2, _, u2, s2 = run_cmd(
            ["diamond", "blastp", "-q", str(q), "-d", str(work / "allm"), "-o", str(out), "--quiet",
             "--outfmt", "6", "qseqid", "sseqid", "pident", "qcovhsp", "-k", "25", "--id", "50",
             "--query-cover", "70", "-p", n_threads],
            check=False, env=env,
        )
        diamond_cpu_s += u2 + s2
        if rc2 != 0:
            if strict:
                raise RuntimeError(f"diamond blastp failed (rc={rc2}) during retrocopy family scan")
            _LOG.warning("diamond blastp exited with rc=%d during retrocopy family scan", rc2)
            return {}, diamond_cpu_s, True
    except FileNotFoundError as exc:
        if strict:
            raise RuntimeError(
                "diamond binary not found on PATH during retrocopy family scan; install diamond or disable zoom"
            ) from exc
        _LOG.warning("diamond binary not found; skipping retrocopy family scan")
        return {}, diamond_cpu_s, True
    fam: dict[int, dict] = {}
    for line in out.read_text().splitlines():
        qs, ss, pid, qc = line.split("\t")
        qi, mi = int(qs[1:]), int(ss[1:])
        sc, mc = single[qi], allm[mi]
        if sc.contig == mc.contig and _cds_overlap_bp(sc.cds, mc.cds) > 0:
            continue  # self / CDS-overlapping locus
        f = fam.setdefault(qi, {"multi": 0, "single": 0, "max_introns": 0, "seen": set()})
        if mi in f["seen"]:
            continue
        f["seen"].add(mi)
        ni = len(mc.introns)
        if ni:
            f["multi"] += 1
            f["max_introns"] = max(f["max_introns"], ni)
        else:
            f["single"] += 1
    return {qi: (f["max_introns"] if f["multi"] > f["single"] else 0) for qi, f in fam.items()}, diamond_cpu_s, False


def _overlaps_bait(
    contig: str,
    a: int,
    b: int,
    bait_intervals: dict[str, list[tuple[int, int]]],
    organelle_contigs: set[str],
) -> bool:
    if contig in organelle_contigs:
        return True
    return any(ba < b and a < bb for ba, bb in bait_intervals.get(contig, ()))


def _has_downstream_histone_stem_loop(genome: Genome, contig: str, strand: str, start: int, end: int) -> bool:
    if strand != "-":
        ds = genome.fetch(contig, end, end + 150).upper()
    else:
        ds = revcomp(genome.fetch(contig, max(0, start - 150), start)).upper()
    return has_histone_stem_loop(ds) or _approx_find(ds, _STEM_LOOP, 2)


def retrocopy_filter(
    selected: list[Cand],
    genome: Genome,
    ev: EvidenceView,
    work: Path,
    bait_intervals: dict[str, list[tuple[int, int]]] | None = None,
    organelle_contigs: set[str] | None = None,
    *,
    threads: int | None = None,
    strict: bool | None = None,
) -> tuple[list[Cand], list[dict], list[Cand], float, bool]:
    single = [c for c in selected if len(c.cds) == 1]
    multi = [c for c in selected if len(c.cds) > 1]
    if not single:
        return selected, [], [], 0.0, False
    bmap = bait_intervals or {}
    org = organelle_contigs or set()
    eff_threads = int(threads if threads is not None else ev.p.threads)
    eff_strict = bool(ev.p.strict_retrocopy_scan if strict is None else strict)
    parents, diamond_cpu_s, scan_failed = parent_intron_counts(
        single, multi, genome, work, threads=eff_threads, strict=eff_strict
    )
    drop = set()
    calls = []
    d1_emit: list[Cand] = []
    for i, c in enumerate(single):
        ds = downstream_seq(genome, c, 300)
        tpm = ev.bins.rpkm(c.contig, c.start, c.end) if ev.bins is not None else 0.0
        ph_c = int(getattr(c, "cds_phase", 0) or 0) % 3
        st = orf_status(genome.spliced(c.contig, c.strand, list(c.cds))[ph_c:])
        tib_p = 0.9 if c.feats.get("tib_exact") else (0.6 if c.feats.get("tib_introns") else 0.3)
        stem_loop = has_histone_stem_loop(ds[:150]) or _approx_find(ds[:150], _STEM_LOOP, 2)
        sec = SingleExonCandidate(
            candidate_id=f"{c.contig}:{c.start}-{c.end}{c.strand}",
            orf_length_nt=c.cds_len,
            has_internal_stop_or_frameshift=not st["valid"],
            parent_multi_exon_intron_count=parents.get(i, 0),
            unique_kmer_rna_tpm=tpm,
            has_3prime_genomic_polya_tract=has_polya(ds[:200]),
            tiberius_coding_prob=tib_p,
            miniprot_identity=c.feats.get("hom_identity", 0.0),
            has_histone_stem_loop=stem_loop,
        )
        call = classify_single_exon_candidate(sec)
        real_disruption = (st["internal_stops"] >= 1) or (not st["frame_ok"])
        emit_pseudo = (
            ev.p.emit_pseudogenes
            and call == "processed_pseudogene"
            and real_disruption
            and not sec.has_histone_stem_loop
            and (sec.parent_multi_exon_intron_count >= 2 or sec.miniprot_identity >= 0.50)
            and not _overlaps_bait(c.contig, c.start, c.end, bmap, org)
        )
        calls.append({**sec.__dict__, "call": call, "real_disruption": real_disruption, "emitted_pseudogene": emit_pseudo})
        if call != "functional_single_exon_gene":
            drop.add(id(c))
            if emit_pseudo:
                d1_emit.append(c)
    return [c for c in selected if id(c) not in drop], calls, d1_emit, diamond_cpu_s, scan_failed


# ----------------------------------------------------------------------------------------------
# UTRs and isoforms
# ----------------------------------------------------------------------------------------------


def compatible(t_exons: tuple, cds: tuple) -> bool:
    a0, b0 = cds[0][0], cds[-1][1]
    clipped = []
    for a, b in t_exons:
        x, y = max(a, a0), min(b, b0)
        if x < y:
            clipped.append((x, y))
    return tuple(merge_intervals(clipped)) == tuple(cds) and t_exons[0][0] <= a0 and t_exons[-1][1] >= b0


def _best_utr_exons(c: Cand, rna_index: LocusIndex | None) -> tuple[tuple[int, int], ...]:
    """Attach the highest-coverage compatible empirical RNA UTR exons, or fall back to c's own exons/CDS."""
    if rna_index is not None:
        best_ex = None
        best_cov = -1.0
        for t in rna_index.overlapping(c.contig, c.start, c.end):
            if t.strand not in (c.strand, "."):
                continue
            ex = tuple(sorted(t.exons))
            if compatible(ex, c.cds):
                cov = float(t.attrs.get("cov", 0) or 0)
                if cov > best_cov:
                    best_cov = cov
                    best_ex = ex
        if best_ex is not None:
            return best_ex
    if c.exons and compatible(c.exons, c.cds):
        return tuple(sorted(c.exons))
    return c.cds


def _unify_bridged_coding_groups(
    selected_sorted: list[Cand],
    raw_isos_list: list[list[tuple[tuple, tuple, Cand]]],
    max_isoforms: int = 3,
) -> list[tuple[Cand, list[tuple[tuple, tuple, Cand]]]]:
    """Consolidate same-strand scheduled fragments ONLY when a single n_unsup==0 isoform bridges their primary CDSs (U1).

    Merges primaries A and B only when an attached isoform owned by one overlaps the primary CDS of all merged
    members (e.g., 43-exon clasp1a bridging 36 + 7 exons). Never merges on alt-alt overlaps or transitive A-B-C
    chains lacking a common spanning transcript, enforces inter-gene CDS non-overlap (N1), and caps at max_isoforms.
    """
    n = len(selected_sorted)
    cap = max(1, int(max_isoforms))
    if n == 0:
        return []
    if n == 1:
        return [(selected_sorted[0], raw_isos_list[0][:cap])]

    prim_idx = LocusIndex(
        [(c.contig, c.start, c.end, (i, c.strand, c.cds)) for i, c in enumerate(selected_sorted)]
    )
    bridge_by_key: dict[tuple[str, str, tuple], tuple[tuple[int, ...], tuple, tuple, Cand]] = {}
    for i, isos in enumerate(raw_isos_list):
        c_i = selected_sorted[i]
        for ex, cds, ow in isos:
            bridged = tuple(
                sorted(
                    j
                    for j, st_j, p_cds_j in prim_idx.overlapping(c_i.contig, cds[0][0], cds[-1][1])
                    if st_j == c_i.strand and _cds_overlap_bp(cds, p_cds_j) > 0
                )
            )
            if len(bridged) < 2:
                continue
            bkey = (c_i.contig, c_i.strand, cds)
            prev = bridge_by_key.get(bkey)
            if prev is None or (ow.score, sum(b - a for a, b in ex)) > (
                prev[3].score,
                sum(b - a for a, b in prev[1]),
            ):
                bridge_by_key[bkey] = (bridged, ex, cds, ow)

    ranked_bridges = sorted(
        bridge_by_key.values(),
        key=lambda item: (
            -len(item[0]),
            -sum(b - a for a, b in item[2]),
            -item[3].score,
            item[2][0][0],
            item[2][-1][1],
            item[2],
        ),
    )

    assigned: set[int] = set()
    group_specs: list[tuple[tuple[int, ...], tuple | None]] = []
    for bridged, _ex_b, cds_b, _ow_b in ranked_bridges:
        if any(j in assigned for j in bridged):
            continue
        assigned.update(bridged)
        group_specs.append((bridged, cds_b))
    for i in range(n):
        if i not in assigned:
            group_specs.append(((i,), None))

    # Build fixed core CDSs per group (member primary CDSs + anchor bridge CDS); core CDSs are disjoint across groups.
    core_cds_sets: list[set[tuple]] = []
    accepted_ivs_by_group: list[list[tuple[int, int]]] = []
    group_meta: list[tuple[str, str]] = []
    for members, anchor_cds in group_specs:
        c0 = selected_sorted[members[0]]
        group_meta.append((c0.contig, c0.strand))
        cset = {selected_sorted[m].cds for m in members}
        if anchor_cds is not None:
            cset.add(anchor_cds)
        core_cds_sets.append(cset)
        accepted_ivs_by_group.append(merge_intervals([iv for cds in cset for iv in cds]))

    def _overlaps_other_group(g_id: int, ctg: str, strand: str, cds: tuple) -> bool:
        lo, hi = cds[0][0], cds[-1][1]
        for g2, ivs2 in enumerate(accepted_ivs_by_group):
            if g2 == g_id or not ivs2:
                continue
            ctg2, st2 = group_meta[g2]
            if ctg2 == ctg and st2 == strand and ivs2[0][0] < hi and lo < ivs2[-1][1]:
                if _cds_overlap_bp(cds, ivs2) > 0:
                    return True
        return False

    # Greedily accept non-core alt isoforms across groups by score so alt-alt overlaps never merge OR leak inter-gene CDS overlap.
    alt_pool: list[tuple[int, int, tuple, tuple, Cand]] = []
    for g_id, (members, _anchor_cds) in enumerate(group_specs):
        cset = core_cds_sets[g_id]
        for mi in members:
            for ord_k, (ex, cds, ow) in enumerate(raw_isos_list[mi]):
                if cds not in cset:
                    alt_pool.append((g_id, ord_k, ex, cds, ow))
    alt_pool.sort(
        key=lambda x: (-x[4].score, -sum(b - a for a, b in x[3]), x[3][0][0], x[3][-1][1], x[0], x[1])
    )
    allowed_alts_by_group: list[set[tuple]] = [set() for _ in group_specs]
    for g_id, _ord_k, _ex, cds, _ow in alt_pool:
        ctg, strand = group_meta[g_id]
        if cds in allowed_alts_by_group[g_id]:
            continue
        if _overlaps_other_group(g_id, ctg, strand, cds):
            continue
        allowed_alts_by_group[g_id].add(cds)
        accepted_ivs_by_group[g_id] = merge_intervals(list(accepted_ivs_by_group[g_id]) + list(cds))

    unified: list[tuple[Cand, list[tuple[tuple, tuple, Cand]]]] = []
    for g_id, (members, anchor_cds) in enumerate(group_specs):
        keep_cds = core_cds_sets[g_id] | allowed_alts_by_group[g_id]
        if len(members) == 1:
            m0 = members[0]
            kept = [item for item in raw_isos_list[m0] if item[1] in keep_cds][:cap]
            unified.append((selected_sorted[m0], kept))
            continue
        by_cds: dict[tuple, tuple[tuple, tuple, Cand]] = {}
        for mi in members:
            for ex, cds, ow in raw_isos_list[mi]:
                if cds not in keep_cds:
                    continue
                prev = by_cds.get(cds)
                if prev is None or (ow.score, sum(b - a for a, b in ex)) > (
                    prev[2].score,
                    sum(b - a for a, b in prev[0]),
                ):
                    by_cds[cds] = (ex, cds, ow)
        dedup_isos = sorted(
            by_cds.values(),
            key=lambda item: (
                0 if item[1] == anchor_cds else 1,
                -sum(b - a for a, b in item[1]),
                -item[2].score,
                item[1][0][0],
                item[1][-1][1],
                item[1],
            ),
        )[:cap]
        rep_c = dedup_isos[0][2]
        unified.append((rep_c, dedup_isos))

    unified.sort(key=lambda pair: (pair[0].contig, min(cds[0][0] for _ex, cds, _ow in pair[1]), pair[0].strand))
    return unified


def attach_transcripts(
    sel: Cand,
    rna_index: LocusIndex | None,
    all_cands_index: LocusIndex | None,
    junc: JunctionStore | None,
    p: SynthParams,
    selected_primaries_index: LocusIndex | None = None,
) -> list[tuple[tuple, tuple, Cand]]:
    """Return [(exons, cds, owner_cand)] isoforms for one selected CDS model (primary first, deduplicated per CDS)."""
    out: list[tuple[tuple, tuple, Cand]] = [(_best_utr_exons(sel, rna_index), sel.cds, sel)]
    seen_cds = {sel.cds}
    if all_cands_index is not None and junc is not None and p.max_isoforms > 1:
        alts: list[Cand] = []
        sel_base = sel.feats.get("score_pre_excuse", sel.score)
        for c in all_cands_index.overlapping(sel.contig, sel.start, sel.end):
            if c is sel or c.strand != sel.strand or c.cds in seen_cds:
                continue
            if c.score < p.min_score or c.feats.get("n_unsup", 1) != 0:
                continue
            c_base = c.feats.get("score_pre_excuse", c.score)
            if c.cds_len < p.alt_cds_min_len_frac * sel.cds_len or c_base < p.alt_cds_min_score_frac * sel_base:
                continue
            if _cds_overlap_bp(c.cds, sel.cds) < 0.5 * min(c.cds_len, sel.cds_len):
                continue
            if selected_primaries_index is not None and any(
                other is not sel and other.strand == sel.strand and _cds_overlap_bp(c.cds, other.cds) > 0
                for other in selected_primaries_index.overlapping(c.contig, c.start, c.end)
            ):
                continue
            alts.append(c)
        alts.sort(key=lambda c: (-c.score, -c.cds_len))
        for c in alts:
            if c.cds in seen_cds:
                continue
            seen_cds.add(c.cds)
            out.append((_best_utr_exons(c, rna_index), c.cds, c))
            if len(out) >= p.max_isoforms:
                break
    return out


# ----------------------------------------------------------------------------------------------
# Pseudogene (D2') and spliced lncRNA (C') post-selection passes
# ----------------------------------------------------------------------------------------------


def synthesize_disabled_pseudogenes(
    genome: Genome,
    raw_alignments: list,
    coding_models: list[Transcript],
    d1_cands: list[Cand],
    p: SynthParams,
    bait_intervals: dict[str, list[tuple[int, int]]],
    organelle_contigs: set[str],
) -> list[tuple[str, str, list[tuple[int, int]], dict]]:
    """Cluster-level consensus disabled homology pseudogene pass (D2'; zero effect on coding selection)."""
    if not p.emit_pseudogenes or not p.emit_disabled_hom_pseudogenes or not raw_alignments:
        return []
    d2_alns = [a for a in raw_alignments if a.positive >= 0.40 and a.coverage >= 0.30]
    if not d2_alns:
        return []
    forb_spans: list[tuple[str, int, int, str]] = [
        (tx.contig, max(0, tx.start - 1_000), tx.end + 1_000, tx.strand) for tx in coding_models if tx.cds
    ] + [
        (c.contig, max(0, c.start - 1_000), c.end + 1_000, c.strand) for c in d1_cands
    ]
    forb_idx = LocusIndex(forb_spans)

    intact_any = [a for a in d2_alns if a.fs == 0 and a.sc == 0]
    intact_any_idx = LocusIndex([(a.contig, a.start, a.end, a) for a in intact_any])
    intact_cov80_idx = LocusIndex([(a.contig, a.start, a.end, a) for a in intact_any if a.coverage >= 0.80])

    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for a in d2_alns:
        if a.contig not in organelle_contigs:
            by_key[(a.contig, a.strand)].append(a)

    out: list[tuple[str, str, list[tuple[int, int]], dict]] = []
    for (ctg, strand), alns in sorted(by_key.items()):
        for comp in _exonic_connected_components(alns, lambda a: a.cds):
            comp_tids = {a.tid for a in comp}
            if len(comp_tids) < 2:
                continue
            if not all((a.fs > 0 or a.sc > 0) for a in comp):
                continue
            comp_lo = min(a.start for a in comp)
            comp_hi = max(a.end for a in comp)
            if any(ia.strand == strand and ia.tid in comp_tids for ia in intact_any_idx.overlapping(ctg, comp_lo, comp_hi)):
                continue
            if any(ia.strand == strand for ia in intact_cov80_idx.overlapping(ctg, comp_lo, comp_hi)):
                continue
            best = max(comp, key=lambda a: (a.quality, a.score, a.cds_len, a.tid, -a.start, -a.end, a.cds))
            if best.positive < 0.60 or best.coverage < 0.70 or best.cds_len < 300:
                continue
            if any(st == strand for st in forb_idx.overlapping(ctg, best.start, best.end)):
                continue
            if _overlaps_bait(ctg, best.start, best.end, bait_intervals, organelle_contigs):
                continue
            if _softmask_frac(genome, ctg, best.cds) > 0.45:
                continue
            if _has_downstream_histone_stem_loop(genome, ctg, strand, best.start, best.end):
                continue
            out.append((
                ctg,
                strand,
                list(best.cds),
                {"pseudo": "true", "evidence": "miniprot_disabled", "n_queries": str(len(comp_tids))},
            ))
    return out


def _is_long_read_tx(t: Transcript) -> bool:
    return t.attrs.get("long_read") == "1" or "long" in str(t.source or "").lower()


def synthesize_lncrnas(
    genome: Genome,
    rna_tx: list[Transcript],
    ev: EvidenceView,
    hits: list[ProteinHit],
    raw_alignments: list | None,
    coding_models: list[Transcript],
    pseudo_models: list[Transcript],
    structural_ncrnas: list[Transcript] | None,
    p: SynthParams,
    bait_intervals: dict[str, list[tuple[int, int]]],
    organelle_contigs: set[str],
    prefix: str = "GL",
    tiberius: list[Transcript] | None = None,
    lnc_eligible_introns: set[tuple[str, int, int]] | None = None,
) -> tuple[list[Transcript], dict]:
    """Select high-confidence spliced StringTie lncRNAs (C') strictly isolated from coding/pseudogene loci."""
    diag = {"lnc_candidates_lt100aa": 0, "lnc_candidates_pass_sup4": 0, "lnc_genes": 0, "lnc_transcripts": 0}
    has_long_tx = any(_is_long_read_tx(t) for t in (rna_tx or ()))
    if not p.emit_lncrna or not rna_tx or (not ev.has_rna and not has_long_tx):
        return [], diag

    eff_runs = (
        set(ev.junc.run_ids) or {r for d in ev.junc.ua.values() for r, v in d.items() if v > 0}
        if ev.junc is not None
        else set()
    )
    n_libs = len(eff_runs)
    if n_libs == 0 and not has_long_tx:
        return [], diag
    min_junc_sup = 8 if n_libs == 1 else 4
    min_runs = 1 if n_libs <= 1 else 2

    buf = max(0, int(p.lnc_coding_buffer_bp))
    cod_span_idx = LocusIndex(
        [(tx.contig, max(0, tx.start - buf), tx.end + buf, tx.strand) for tx in coding_models if tx.cds]
    )
    pseudo_span_idx = LocusIndex([(ps.contig, ps.start, ps.end, ps.strand) for ps in pseudo_models])
    tib_source = tiberius if tiberius is not None else ev.all_tiberius_cds
    all_tib_idx = LocusIndex([(tb.contig, tb.cds_start, tb.cds_end, tb.strand) for tb in tib_source if tb.cds])

    hom45_items: list[tuple[str, int, int, tuple[str, tuple[tuple[int, int], ...]]]] = []
    seen_hom: set[tuple[str, str, tuple[tuple[int, int], ...]]] = set()
    for h in hits:
        if max(h.positive, float(getattr(h, "max_positive", 0.0) or 0.0)) >= 0.45:
            hk = (h.contig, h.strand, h.cds)
            if hk not in seen_hom:
                seen_hom.add(hk)
                hom45_items.append((h.contig, h.start, h.end, (h.strand, h.cds)))
    if raw_alignments:
        for a in raw_alignments:
            if a.positive >= 0.45 and (a.contig, a.strand, a.cds) not in seen_hom:
                seen_hom.add((a.contig, a.strand, a.cds))
                hom45_items.append((a.contig, a.start, a.end, (a.strand, a.cds)))
    hom45_idx = LocusIndex(hom45_items)

    snc_idx = (
        LocusIndex([(nc.contig, nc.start, nc.end, (nc.strand, tuple(sorted(nc.exons)))) for nc in structural_ncrnas if nc.exons])
        if structural_ncrnas
        else None
    )

    kept_by_chain: dict[tuple[str, str, tuple[tuple[int, int], ...]], tuple[float, int, bool]] = {}
    for t in rna_tx:
        if t.strand not in ("+", "-"):
            continue
        is_long = _is_long_read_tx(t)
        if not is_long and (ev.junc is None or n_libs == 0):
            continue
        exons = tuple(merge_intervals(t.exons))
        if len(exons) < 2:
            continue
        spliced_len = sum(b - a for a, b in exons)
        if spliced_len < 200:
            continue
        cov = float(t.attrs.get("cov", 0) or 0)
        if cov < (2.0 if (is_long or len(exons) >= 3) else 2.5):
            continue
        t_lo, t_hi = exons[0][0], exons[-1][1]
        if _overlaps_bait(t.contig, t_lo, t_hi, bait_intervals, organelle_contigs):
            continue
        if any(st == t.strand for st in cod_span_idx.overlapping(t.contig, t_lo, t_hi)):
            continue
        if any(st == t.strand for st in pseudo_span_idx.overlapping(t.contig, t_lo, t_hi)):
            continue
        if any(tb_st == t.strand for tb_st in all_tib_idx.overlapping(t.contig, t_lo, t_hi)):
            continue
        hom_conflict = False
        for h_strand, h_cds in hom45_idx.overlapping(t.contig, t_lo, t_hi):
            if h_strand == t.strand or _cds_overlap_bp(exons, h_cds) > 0:
                hom_conflict = True
                break
        if hom_conflict:
            continue
        if snc_idx is not None:
            if any(nc_st == t.strand and _cds_overlap_bp(exons, nc_ex) > 0 for nc_st, nc_ex in snc_idx.overlapping(t.contig, t_lo, t_hi)):
                continue

        intr = introns_of(list(exons))
        junc_ok_sup4 = True
        junc_ok_strict = True
        for a, b in intr:
            jk = (t.contig, a, b)
            if is_long:
                motif_str, motif_st = junction_motif(genome, t.contig, a, b)
                if motif_str not in CANONICAL or motif_st != t.strand:
                    junc_ok_sup4 = False
                    junc_ok_strict = False
                    break
                continue
            if not ev.junc.canonical(jk) or ev.junc.strand(jk) != t.strand:
                junc_ok_sup4 = False
                junc_ok_strict = False
                break
            in_v3_whitelist = lnc_eligible_introns is not None and jk in lnc_eligible_introns
            n_runs = ev.junc.n_runs_supporting(jk)
            if not in_v3_whitelist and min_runs >= 2 and n_runs < min_runs:
                junc_ok_sup4 = False
                junc_ok_strict = False
                break
            sup = ev.junc.support(jk)
            motif_str = ev.junc.motif.get(jk, ("", "."))[0]
            multi_run_gt_ag = 2 <= n_libs < 10 and n_runs >= 2 and sup >= 2 and motif_str == "GT-AG"
            single_lib_multi_intr = n_libs == 1 and len(intr) >= 2 and sup >= 4 and motif_str == "GT-AG"
            if not ((in_v3_whitelist and sup >= 2) or multi_run_gt_ag or sup >= 4):
                junc_ok_sup4 = False
                junc_ok_strict = False
                break
            if not ((in_v3_whitelist and sup >= 2) or multi_run_gt_ag or single_lib_multi_intr or sup >= min_junc_sup):
                junc_ok_strict = False
        if not junc_ok_sup4:
            continue
        if _softmask_frac(genome, t.contig, exons) > 0.45:
            continue
        spliced_seq = genome.spliced(t.contig, t.strand, list(exons))
        max_aa = max_open_orf_aa(spliced_seq)
        if (3 * max_aa) / spliced_len >= 0.60:
            continue
        if max_aa < 100 and junc_ok_strict:
            diag["lnc_candidates_lt100aa"] += 1
        if max_aa >= 80:
            continue
        diag["lnc_candidates_pass_sup4"] += 1
        if not junc_ok_strict:
            continue
        ckey = (t.contig, t.strand, exons)
        prev = kept_by_chain.get(ckey)
        if prev is None or cov > prev[0] or (is_long and not prev[2]):
            kept_by_chain[ckey] = (max(cov, prev[0]) if prev is not None else cov, spliced_len, bool(prev[2] if prev is not None else False) or is_long)

    by_cs: dict[tuple[str, str], list[tuple[tuple[tuple[int, int], ...], float, int, bool]]] = defaultdict(list)
    for (ctg, strand, exons), (cov, slen, is_long) in kept_by_chain.items():
        by_cs[(ctg, strand)].append((exons, cov, slen, is_long))

    clusters: list[tuple[str, int, str, list[tuple[tuple[tuple[int, int], ...], float, int, bool]]]] = []
    for (ctg, strand), items in sorted(by_cs.items()):
        for comp in _exonic_connected_components(items, lambda x: x[0]):
            comp.sort(key=lambda x: (-x[1], -x[2], x[0]))
            top = comp[:2]
            min_s = min(ex[0][0] for ex, _cov, _sl, _lr in top)
            clusters.append((ctg, min_s, strand, top))
    clusters.sort(key=lambda x: (x[0], x[1], x[2]))

    lnc_pfx = "GLNC" if prefix.startswith("GL") else f"{prefix}NC"
    lnc_models: list[Transcript] = []
    for li, (ctg, _min_s, strand, top) in enumerate(clusters, 1):
        gid = f"{lnc_pfx}{li:06d}"
        for ti, (exons, cov, _slen, is_long) in enumerate(top, 1):
            lnc_attrs = {"evidence": "stringtie_lncrna", "cov": f"{cov:.2f}"}
            if is_long:
                lnc_attrs["long_read"] = "1"
            lnc_models.append(
                Transcript(
                    tid=f"{gid}.t{ti}",
                    contig=ctg,
                    strand=strand,
                    exons=list(exons),
                    cds=[],
                    gid=gid,
                    source="GeneLoop",
                    attrs=lnc_attrs,
                    biotype="lncRNA",
                )
            )
    diag["lnc_genes"] = len(clusters)
    diag["lnc_transcripts"] = len(lnc_models)
    return lnc_models, diag


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


@dataclass
class SynthResult:
    models: list[Transcript]
    selected: list[Cand]
    n_candidates: int
    retro_calls: list[dict]
    stats: dict


def synthesize(
    genome: Genome,
    tiberius: list[Transcript],
    hits: list[ProteinHit],
    loci: list[Locus] | None = None,
    junctions: JunctionStore | None = None,
    rna_tx: list[Transcript] | None = None,
    bins: BinStore | None = None,
    microexons: dict | None = None,
    zoom: bool = False,
    work: Path | None = None,
    params: SynthParams | None = None,
    prefix: str = "GL",
    raw_alignments: list | None = None,
    bait_intervals: dict[str, list[tuple[int, int]]] | None = None,
    organelle_contigs: set[str] | None = None,
    readthrough_targets: set[str] | None = None,
    structural_ncrnas: list[Transcript] | None = None,
    raw_ncrna_candidates: list[dict] | None = None,
    scanned_tib_windows: list[tuple[str, int, int]] | None = None,
    gap_continuity: dict[tuple[str, int, int], int] | None = None,
    lnc_eligible_introns: set[tuple[str, int, int]] | None = None,
    threads: int | None = None,
) -> SynthResult:
    p = params or SynthParams()
    bmap = bait_intervals or {}
    org = organelle_contigs or set()
    # Y7: When Tiberius ran genome-wide (non-inverted mode, no windowed seg_lo attributes), pass whole-contig nuclear windows.
    eff_scanned_wins = scanned_tib_windows
    if eff_scanned_wins is None and tiberius and not any("seg_lo" in t.attrs for t in tiberius):
        eff_scanned_wins = [(c, 0, int(L)) for c, L in sorted(genome.lengths.items()) if c not in org]
    cands = generate_candidates(
        genome,
        tiberius,
        hits,
        rna_tx,
        p,
        junctions=junctions,
        raw_alignments=raw_alignments,
        scanned_tib_windows=eff_scanned_wins,
        gap_continuity=gap_continuity,
        bait_intervals=bmap,
        organelle_contigs=org,
    )
    ev = EvidenceView(tiberius, hits, junctions, bins, p, genome, readthrough_targets=readthrough_targets)
    clist = list(cands.values())
    if zoom and microexons:
        extra = []
        for c in clist:
            nc = insert_microexons(c, genome, microexons)
            if nc is not c:
                extra.append(nc)
        for nc in extra:
            _add(cands, nc)
        clist = list(cands.values())
    tib_cands = [c for c in clist if "tiberius" in c.sources]
    mp_cands = [c for c in clist if "tiberius" not in c.sources and c.hit is not None]
    other_cands = [c for c in clist if "tiberius" not in c.sources and c.hit is None]

    def _run_score_batch(batch: list[Cand]) -> None:
        for c in batch:
            featurize(c, ev, genome)
            score(c, ev)
            if c.microexons:
                if c.score > 0:
                    c.score += 0.3 * c.microexons
                if c.feats.get("score_pre_excuse", -1.0) > 0:
                    c.feats["score_pre_excuse"] += 0.3 * c.microexons

    _run_score_batch(tib_cands)
    ev.viable_tib_keys = {c.key() for c in tib_cands if c.score >= p.min_score}
    _run_score_batch(mp_cands)
    ev.viable_hit_keys = {
        (c.hit.contig, c.hit.strand, c.hit.cds)
        for c in mp_cands
        if c.hit is not None and c.score >= p.min_score and c.feats.get("n_contra", 0) == 0
    }
    _run_score_batch(other_cands)
    by = defaultdict(list)
    for c in clist:
        by[(c.contig, c.strand)].append(c)
    selected: list[Cand] = []
    for k in sorted(by):
        selected.extend(schedule(by[k]))
    selected = [c for c in selected if c.score >= p.min_score]
    if p.antisense_single_ov_frac > 0 and selected:
        sel_idx = LocusIndex([(c.contig, c.start, c.end, c) for c in selected])
        antisense_kept: list[Cand] = []
        for c in selected:
            non_zoom = c.sources - {"microexon_zoom"}
            is_rna_only = "tiberius" not in non_zoom and "miniprot" not in non_zoom
            is_single = len(non_zoom) == 1
            drop = False
            if is_rna_only or is_single:
                for o in sel_idx.overlapping(c.contig, c.start, c.end):
                    if o.strand == c.strand or o.score <= c.score:
                        continue
                    ov = _cds_overlap_bp(c.cds, o.cds)
                    if ov > 0 and (is_rna_only or ov >= p.antisense_single_ov_frac * c.cds_len):
                        drop = True
                        break
            if not drop:
                antisense_kept.append(c)
        selected = antisense_kept

    retro_calls: list[dict] = []
    d1_cands: list[Cand] = []
    diamond_cpu_s = 0.0
    retro_scan_failed = False
    eff_threads = int(threads if threads is not None else p.threads)
    if zoom:
        selected, retro_calls, d1_cands, diamond_cpu_s, retro_scan_failed = retrocopy_filter(
            selected,
            genome,
            ev,
            (work or Path(".")) / "retro",
            bait_intervals=bmap,
            organelle_contigs=org,
            threads=eff_threads,
            strict=p.strict_retrocopy_scan,
        )

    # Locus status lookup (for attributes only, requiring CDS overlap when locked_model exists)
    lidx = LocusIndex([(l.contig, l.start, l.end, l) for l in loci]) if loci else None
    rna_index = LocusIndex([(t.contig, t.start, t.end, t) for t in rna_tx]) if rna_tx else None
    cand_index = LocusIndex([(c.contig, c.start, c.end, c) for c in clist]) if rna_tx else None
    selected_sorted = sorted(selected, key=lambda c: (c.contig, c.start, c.end, c.strand))
    prim_index = LocusIndex([(c.contig, c.start, c.end, c) for c in selected_sorted]) if rna_tx else None
    raw_isos_list = [
        attach_transcripts(c, rna_index, cand_index, junctions, p, selected_primaries_index=prim_index)
        for c in selected_sorted
    ]
    unified_groups = _unify_bridged_coding_groups(
        selected_sorted, raw_isos_list, max_isoforms=p.max_isoforms
    )
    n_consolidated_split_genes = len(selected_sorted) - len(unified_groups)

    coding_models: list[Transcript] = []
    kept_selected: list[Cand] = []
    emitted_tx_by_source: dict[str, int] = {}
    n_utr = 0
    n_b_prime_genes = 0
    n_trimmed_spacer_tx = 0
    n_backfill_tx = 0
    n_dropped_invalid_coding = 0
    gene_idx = 0
    for c, isos in unified_groups:
        valid_isos: list[tuple[tuple, tuple, Cand]] = []
        for ex, cds, owner_c in isos:
            ph_o = int(getattr(owner_c, "cds_phase", 0) or 0) % 3
            spliced_nt = genome.spliced(c.contig, c.strand, list(cds))[ph_o:].upper()
            if not spliced_nt or len(spliced_nt) % 3 != 0:
                n_dropped_invalid_coding += 1
                continue
            aa = translate(spliced_nt)
            aa_body = aa[:-1] if aa.endswith("*") else aa
            if not aa_body or "*" in aa_body:
                n_dropped_invalid_coding += 1
                continue
            valid_isos.append((ex, cds, owner_c))
        if not valid_isos:
            continue
        gene_idx += 1
        kept_selected.append(c)
        gid = f"{prefix}{gene_idx:06d}"
        status = ""
        if lidx is not None:
            ls = [
                l
                for l in lidx.overlapping(c.contig, c.start, c.end)
                if l.strand == c.strand
                and (
                    getattr(l, "locked_model", None) is None
                    or not l.locked_model.cds
                    or _cds_overlap_bp(c.cds, l.locked_model.cds) > 0
                )
            ]
            status = ls[0].status if ls else "NOVEL"
        if int(c.feats.get("n_unsup_excused", 0)) > 0:
            n_b_prime_genes += 1
        for ti, (ex, cds, owner_c) in enumerate(valid_isos, 1):
            t = Transcript(
                tid=f"{gid}.t{ti}",
                contig=c.contig,
                strand=c.strand,
                exons=list(ex),
                cds=list(cds),
                gid=gid,
                source="GeneLoop",
                biotype="protein_coding",
                cds_phase=int(getattr(owner_c, "cds_phase", 0) or 0) % 3,
            )
            if tuple(ex) != tuple(cds):
                n_utr += 1
            ev_label = "+".join(sorted(owner_c.sources))
            emitted_tx_by_source[ev_label] = emitted_tx_by_source.get(ev_label, 0) + 1
            t.attrs = {
                "evidence": ev_label,
                "score": f"{owner_c.score:.2f}",
                "locus_status": status,
            }
            if owner_c.microexons:
                t.attrs["microexons"] = str(owner_c.microexons)
            owner_excused = int(owner_c.feats.get("n_unsup_excused", 0))
            if owner_excused > 0:
                t.attrs["b_prime_excused"] = str(owner_excused)
            if owner_c.feats.get("trimmed_spacer", False):
                t.attrs["trimmed_spacer"] = "1"
                n_trimmed_spacer_tx += 1
            if owner_c.feats.get("tiberius_backfill", False):
                t.attrs["backfill"] = "1"
                n_backfill_tx += 1
            if owner_c.feats.get("partial_edge", False):
                t.attrs["partial_edge"] = "1"
            coding_models.append(t)
    selected = kept_selected

    pseudo_raw: list[tuple[str, str, list[tuple[int, int]], dict]] = [
        (c.contig, c.strand, list(c.cds), {"pseudo": "true", "evidence": "retrocopy_disabled", "score": f"{c.score:.2f}"})
        for c in d1_cands
    ]
    d2_raw = (
        synthesize_disabled_pseudogenes(genome, raw_alignments, coding_models, d1_cands, p, bmap, org)
        if raw_alignments
        else []
    )
    pseudo_raw.extend(d2_raw)
    pseudo_raw.sort(key=lambda r: (r[0], r[2][0][0], r[2][-1][1], r[1]))

    pseudo_pfx = "GLP" if prefix.startswith("GL") else f"{prefix}P"
    protein_pseudo_models: list[Transcript] = []
    for pi, (ctg, strand, ex_blocks, pattrs) in enumerate(pseudo_raw, 1):
        gid = f"{pseudo_pfx}{pi:06d}"
        protein_pseudo_models.append(
            Transcript(
                tid=f"{gid}.t1",
                contig=ctg,
                strand=strand,
                exons=ex_blocks,
                cds=[],
                gid=gid,
                source="GeneLoop",
                attrs=pattrs,
                biotype="pseudogene",
            )
        )

    if p.emit_structural_ncrna and raw_ncrna_candidates:
        func_snc_models, pseudo_ncrna_models, ncrna_diag = filter_and_materialize_structural_ncrnas(
            raw_ncrna_candidates,
            coding_models,
            protein_pseudo_models,
            organelle_contigs=org,
            prefix=prefix,
            emit_pseudogenes=p.emit_pseudogenes,
        )
    else:
        func_snc_models = []
        pseudo_ncrna_models = []
        ncrna_diag = {
            "n_raw": 0,
            "n_rejected_same_strand_exon": 0,
            "n_rejected_opp_strand_cds": 0,
            "n_functional": 0,
            "n_pseudo_ncrna": 0,
            "by_biotype": {"tRNA": 0, "rRNA": 0, "snRNA": 0, "snoRNA": 0},
            "pseudo_by_biotype": {"rRNA_pseudogene": 0, "ncRNA_pseudogene": 0},
        }

    eff_snc_veto = (
        ([nc for nc in structural_ncrnas if not is_pseudogene_model(nc)] if structural_ncrnas else [])
        + func_snc_models
    ) or None

    lnc_models, lnc_diag = synthesize_lncrnas(
        genome,
        rna_tx or [],
        ev,
        hits,
        raw_alignments,
        coding_models,
        protein_pseudo_models,
        eff_snc_veto,
        p,
        bmap,
        org,
        prefix=prefix,
        tiberius=tiberius,
        lnc_eligible_introns=lnc_eligible_introns,
    )
    lnc_gate_stats: dict | None = None
    if lnc_eligible_introns is not None:
        lnc_models, lnc_gate_stats = gate_lncrna_models_posthoc(
            lnc_models,
            eligible_introns=lnc_eligible_introns,
            prefix=prefix,
        )
        lnc_diag["lnc_genes"] = lnc_gate_stats["n_lnc_genes_kept"]
        lnc_diag["lnc_transcripts"] = lnc_gate_stats["n_lnc_kept"]

    pseudo_models = protein_pseudo_models + pseudo_ncrna_models
    models = coding_models + pseudo_models + func_snc_models + lnc_models
    stats = {
        "n_candidates": len(clist),
        "n_genes": len(selected) + len(pseudo_models) + len(func_snc_models) + lnc_diag["lnc_genes"],
        "n_coding_genes": len(selected),
        "n_transcripts": len(models),
        "n_coding_transcripts": len(coding_models),
        "n_with_utr": n_utr,
        "n_single_exon_genes": sum(1 for c in selected if len(c.cds) == 1),
        "n_microexon_models": sum(1 for c in selected if c.microexons),
        "n_b_prime_genes": n_b_prime_genes,
        "n_consolidated_split_genes": n_consolidated_split_genes,
        "n_trimmed_spacer_tx": n_trimmed_spacer_tx,
        "n_tiberius_backfill_tx": n_backfill_tx,
        "retro_rejected": sum(1 for r in retro_calls if r["call"] == "processed_pseudogene"),
        "retro_emitted_pseudogenes": len(d1_cands),
        "retro_scan_failed": retro_scan_failed,
        "dropped_invalid_coding_tx": n_dropped_invalid_coding,
        "disabled_hom_pseudogenes": len(d2_raw),
        "n_pseudo_genes": len(pseudo_models),
        "n_pseudo_protein_genes": len(protein_pseudo_models),
        "n_pseudo_ncrna_genes": len(pseudo_ncrna_models),
        "n_structural_ncrna_genes": len(func_snc_models),
        "n_structural_ncrna_by_biotype": ncrna_diag["by_biotype"],
        "structural_ncrna_filter": ncrna_diag,
        "n_lnc_genes": lnc_diag["lnc_genes"],
        "n_lnc_transcripts": lnc_diag["lnc_transcripts"],
        "lnc_candidates_lt100aa": lnc_diag["lnc_candidates_lt100aa"],
        "lnc_candidates_pass_sup4": lnc_diag["lnc_candidates_pass_sup4"],
        "diamond_cpu_s": round(diamond_cpu_s, 4),
        "single_exon_noise_rejected": sum(1 for r in retro_calls if r["call"] == "discard_noise"),
        "selected_by_source": _source_counts(selected),
        "emitted_tx_by_source": emitted_tx_by_source,
    }
    if lnc_gate_stats is not None:
        stats["logan_lnc_gate"] = lnc_gate_stats
    return SynthResult(models, selected, len(clist), retro_calls, stats)


def _source_counts(sel: list[Cand]) -> dict:
    out: dict[str, int] = {}
    for c in sel:
        k = "+".join(sorted(c.sources))
        out[k] = out.get(k, 0) + 1
    return out


# ----------------------------------------------------------------------------------------------
# Logan v3 Wide-Cohort Guards (>= 32 Logan runs)
# ----------------------------------------------------------------------------------------------

LOGAN_WIDE_COHORT_MIN_RUNS = 32


def compute_logan_lnc_thresholds(n_logan_runs: int) -> tuple[int, int]:
    """Return ``(min_lnc_runs, min_lnc_ua)`` for wide Logan cohorts."""
    min_lnc_runs = max(3, min(30, round(0.15 * int(n_logan_runs))))
    return min_lnc_runs, 2 * min_lnc_runs


def build_safe_locked_miniprot_ref_models(seed_or_loci: dict | list) -> list[Transcript]:
    """Build pairwise non-overlapping ``safe_locked_miniprot`` ortholog models (NEVER Tiberius)."""
    loci = seed_or_loci.get("loci", ()) if isinstance(seed_or_loci, dict) else seed_or_loci
    cands: list[Transcript] = []
    for loc in loci:
        if getattr(loc, "category", "") == "safe_locked_miniprot" and getattr(loc, "locked_model", None) is not None:
            lm = loc.locked_model
            if lm.cds:
                cands.append(
                    Transcript(
                        tid=f"SL_{loc.locus_id}",
                        gid=f"SL_{loc.locus_id}",
                        contig=lm.contig,
                        strand=lm.strand,
                        source="safe_locked_miniprot",
                        exons=list(lm.cds),
                        cds=list(lm.cds),
                    )
                )
    by_cs: dict[tuple[str, str], list[Transcript]] = {}
    for t in cands:
        by_cs.setdefault((t.contig, t.strand), []).append(t)
    clean: list[Transcript] = []
    for cs in sorted(by_cs):
        txs = sorted(by_cs[cs], key=lambda x: (x.cds_end, x.cds_start, x.tid))
        kept: list[Transcript] = []
        for t in txs:
            if all(interval_overlap_bp(t.cds, k.cds) == 0 for k in kept):
                kept.append(t)
        clean.extend(kept)
    return clean


def split_safe_ortholog_chimeras(
    rna_tx: list[Transcript],
    safe_mp_ref: list[Transcript],
    min_locus_ov_bp: int = 100,
) -> tuple[list[Transcript], dict]:
    """Split chimeric RNA transcripts bridging >=2 ``safe_locked_miniprot`` orthologs (NEVER Tiberius).

    Preserves outer 5'/3' UTR exons upstream of the first ortholog and downstream of the last ortholog
    while dropping inter-locus bridge exons.
    """
    anchor_spans: dict[int, tuple[str, str, int, int]] = {}
    anchor_cds: dict[int, tuple[tuple[int, int], ...]] = {}
    anchor_items = []
    for lid, m in enumerate(safe_mp_ref, 1):
        src_low = str(m.source or "").lower()
        if src_low.startswith("tiberius"):
            continue
        if m.cds:
            cds_blks = tuple(merge_intervals(m.cds))
            if not cds_blks:
                continue
            a, b = cds_blks[0][0], cds_blks[-1][1]
            anchor_spans[lid] = (m.contig, m.strand, a, b)
            anchor_cds[lid] = cds_blks
            anchor_items.append((m.contig, a, b, (m.strand, lid)))
    anchor_idx = LocusIndex(anchor_items)

    out_tx: list[Transcript] = []
    n_chimeric_split = 0
    n_fragments_emitted = 0

    for tx in rna_tx:
        exons = tuple(merge_intervals(tx.exons))
        if not exons:
            continue
        t_lo, t_hi = exons[0][0], exons[-1][1]
        hit_lids: dict[int, int] = defaultdict(int)
        for st, x_lid in anchor_idx.overlapping(tx.contig, t_lo, t_hi):
            if st != tx.strand:
                continue
            ov = interval_overlap_bp(exons, anchor_cds[x_lid])
            if ov >= min_locus_ov_bp:
                hit_lids[x_lid] += ov

        if len(hit_lids) >= 2:
            n_chimeric_split += 1
            loci_sorted = sorted(hit_lids.keys(), key=lambda x: (anchor_spans[x][2], anchor_spans[x][3], x))
            first_lid = loci_sorted[0]
            first_la = anchor_spans[first_lid][2]
            last_lid = max(loci_sorted, key=lambda x: (anchor_spans[x][3], anchor_spans[x][2], x))
            last_lb = anchor_spans[last_lid][3]
            buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for ea, eb in exons:
                cds_hits: list[tuple[int, list[tuple[int, int]]]] = []
                for x_lid in loci_sorted:
                    hit_blks = [(ca, cb) for ca, cb in anchor_cds[x_lid] if cb > ea and ca < eb]
                    if hit_blks:
                        cds_hits.append((x_lid, hit_blks))
                if len(cds_hits) == 1:
                    buckets[cds_hits[0][0]].append((ea, eb))
                elif len(cds_hits) >= 2:
                    for x_lid, hit_blks in cds_hits:
                        sa = ea if (x_lid == first_lid and ea < first_la and hit_blks[0][0] == first_la) else max(ea, hit_blks[0][0])
                        sb = eb if (x_lid == last_lid and eb > last_lb and hit_blks[-1][1] == last_lb) else min(eb, hit_blks[-1][1])
                        if sb > sa:
                            buckets[x_lid].append((sa, sb))
                else:
                    if eb <= first_la:
                        buckets[first_lid].append((ea, eb))
                    elif ea >= last_lb:
                        buckets[last_lid].append((ea, eb))
            emitted_idx = 0
            for x_lid in loci_sorted:
                bex = buckets.get(x_lid)
                if not bex:
                    continue
                sp_tid = f"{tx.tid}_sp{emitted_idx}"
                sp_gid = f"{tx.gid}_sp{emitted_idx}" if tx.gid else sp_tid
                new_attrs = dict(tx.attrs)
                if "transcript_id" in new_attrs:
                    new_attrs["transcript_id"] = sp_tid
                if "gene_id" in new_attrs:
                    new_attrs["gene_id"] = sp_gid
                out_tx.append(
                    Transcript(
                        tid=sp_tid,
                        gid=sp_gid,
                        contig=tx.contig,
                        strand=tx.strand,
                        source=tx.source,
                        exons=merge_intervals(bex),
                        attrs=new_attrs,
                        biotype=tx.biotype,
                    )
                )
                emitted_idx += 1
                n_fragments_emitted += 1
        else:
            out_tx.append(tx)

    stats = {
        "n_input_tx": len(rna_tx),
        "n_output_tx": len(out_tx),
        "n_chimeric_split": n_chimeric_split,
        "n_fragments_emitted": n_fragments_emitted,
    }
    return out_tx, stats


def compute_eligible_lnc_introns(
    unw_runs: dict[tuple[str, int, int], int],
    unw_ua: dict[tuple[str, int, int], int],
    min_lnc_runs: int,
    min_lnc_ua: int,
    fb_js: JunctionStore | None = None,
    fb_min_ua: int = 4,
) -> set[tuple[str, int, int]]:
    """Return introns passing wide-cohort unweighted Logan run/contig support OR fallback support >= 4."""
    eligible: set[tuple[str, int, int]] = set()
    for jk, nr in unw_runs.items():
        if nr >= min_lnc_runs and unw_ua.get(jk, 0) >= min_lnc_ua:
            eligible.add(jk)
    if fb_js is not None:
        for jk in fb_js.ua:
            if fb_js.support(jk) >= fb_min_ua:
                eligible.add(jk)
    return eligible


def gate_lncrna_models_posthoc(
    models: list[Transcript],
    unw_runs: dict[tuple[str, int, int], int] | None = None,
    unw_ua: dict[tuple[str, int, int], int] | None = None,
    min_lnc_runs: int = 3,
    min_lnc_ua: int = 6,
    fb_js: JunctionStore | None = None,
    fb_min_ua: int = 4,
    *,
    eligible_introns: set[tuple[str, int, int]] | None = None,
    prefix: str | None = None,
) -> tuple[list[Transcript], dict]:
    """Filter ``lncRNA`` models post-synthesis using unweighted Logan run/contig counts or fallback >= 4.

    Leaves all coding, pseudogene, and structural ncRNA models untouched.
    """
    eff_eligible = (
        eligible_introns
        if eligible_introns is not None
        else compute_eligible_lnc_introns(
            unw_runs or {},
            unw_ua or {},
            min_lnc_runs=min_lnc_runs,
            min_lnc_ua=min_lnc_ua,
            fb_js=fb_js,
            fb_min_ua=fb_min_ua,
        )
    )
    kept_non_lnc: list[Transcript] = []
    kept_lnc: list[Transcript] = []
    n_lnc_in = 0
    n_lnc_kept = 0
    n_lnc_dropped = 0
    out_ordered: list[tuple[bool, Transcript]] = []

    for m in models:
        bt = m.biotype or m.attrs.get("gene_biotype", "") or m.attrs.get("biotype", "")
        if bt != "lncRNA":
            kept_non_lnc.append(m)
            out_ordered.append((False, m))
            continue
        n_lnc_in += 1
        ex = merge_intervals(m.exons)
        intr = introns_of(ex)
        ok = (m.attrs.get("long_read") == "1" and bool(intr)) or (
            bool(intr) and all((m.contig, a, b) in eff_eligible for a, b in intr)
        )
        if ok:
            kept_lnc.append(m)
            out_ordered.append((True, m))
            n_lnc_kept += 1
        else:
            n_lnc_dropped += 1

    if prefix is not None and kept_lnc:
        lnc_pfx = "GLL" if prefix.startswith("GL") else f"{prefix}L"
        by_old_gid: dict[str, list[Transcript]] = {}
        old_gid_order: list[str] = []
        for m in kept_lnc:
            og = m.gid or m.tid
            if og not in by_old_gid:
                by_old_gid[og] = []
                old_gid_order.append(og)
            by_old_gid[og].append(m)
        remap: dict[int, Transcript] = {}
        for gi, og in enumerate(old_gid_order, 1):
            new_gid = f"{lnc_pfx}{gi:06d}"
            for ti, m in enumerate(by_old_gid[og], 1):
                remap[id(m)] = Transcript(
                    tid=f"{new_gid}.t{ti}",
                    contig=m.contig,
                    strand=m.strand,
                    exons=list(m.exons),
                    cds=list(m.cds),
                    gid=new_gid,
                    source=m.source,
                    attrs=dict(m.attrs),
                    biotype=m.biotype,
                )
        final_models = [remap[id(m)] if is_lnc else m for is_lnc, m in out_ordered]
        n_lnc_genes_kept = len(old_gid_order)
    else:
        final_models = [m for _is_lnc, m in out_ordered]
        n_lnc_genes_kept = len({m.gid or m.tid for m in kept_lnc})

    return final_models, {
        "n_lnc_in": n_lnc_in,
        "n_lnc_kept": n_lnc_kept,
        "n_lnc_dropped": n_lnc_dropped,
        "n_lnc_genes_kept": n_lnc_genes_kept,
    }
