"""Minimal, dependency-free GTF/GFF3 transcript model I/O.

Coordinates are 0-based half-open internally; files are 1-based closed.
CDS intervals always INCLUDE the stop codon internally (RefSeq GFF3 convention), so a separate
``stop_codon`` feature from GTF writers (Tiberius/AUGUSTUS style) is merged into the CDS.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

Interval = tuple[int, int]


def interval_overlap_bp(a: Sequence[Interval], b: Sequence[Interval]) -> int:
    """Return overlapping base pairs between two sorted disjoint interval sequences."""
    tot = i = j = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        ax, ay = a[i]
        bx, by = b[j]
        lo = ax if ax > bx else bx
        hi = ay if ay < by else by
        if hi > lo:
            tot += hi - lo
        if ay < by:
            i += 1
        else:
            j += 1
    return tot


def _open(path: str | Path):
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def merge_intervals(iv: Iterable[Interval]) -> list[Interval]:
    s = sorted(iv)
    out: list[list[int]] = []
    for a, b in s:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def introns_of(blocks: list[Interval]) -> tuple[Interval, ...]:
    b = sorted(blocks)
    return tuple((b[i][1], b[i + 1][0]) for i in range(len(b) - 1))


@dataclass
class Transcript:
    tid: str
    contig: str
    strand: str
    exons: list[Interval] = field(default_factory=list)
    cds: list[Interval] = field(default_factory=list)
    gid: str = ""
    source: str = ""
    attrs: dict = field(default_factory=dict)
    biotype: str = ""
    cds_phase: int = 0

    @property
    def start(self) -> int:
        e = self.exons or self.cds
        return min(a for a, _ in e)

    @property
    def end(self) -> int:
        e = self.exons or self.cds
        return max(b for _, b in e)

    @property
    def cds_start(self) -> int:
        return min(a for a, _ in self.cds)

    @property
    def cds_end(self) -> int:
        return max(b for _, b in self.cds)

    @property
    def introns(self) -> tuple[Interval, ...]:
        return introns_of(self.exons)

    @property
    def cds_introns(self) -> tuple[Interval, ...]:
        return introns_of(self.cds)

    @property
    def cds_len(self) -> int:
        return sum(b - a for a, b in self.cds)

    def is_coding(self) -> bool:
        return bool(self.cds)

    def normalize(self) -> None:
        if self.strand not in ("+", "-", "."):
            self.strand = "."  # RefSeq '?' (e.g. trans-spliced mod(mdg4)) -> unknown
        self.exons = merge_intervals(self.exons)
        self.cds = merge_intervals(self.cds)
        if not self.exons and self.cds:
            self.exons = list(self.cds)


_GTF_ATTR = re.compile(r'\s*([^\s"]+)\s+"?([^";]*)"?\s*;?')


def parse_attrs(col9: str, is_gtf: bool) -> dict:
    out: dict = {}
    if is_gtf:
        for part in col9.strip().split(";"):
            part = part.strip()
            if not part:
                continue
            if " " in part:
                k, v = part.split(" ", 1)
                out[k] = v.strip().strip('"')
            else:
                out[part] = ""
    else:
        for part in col9.strip().split(";"):
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
            else:
                out[part.strip()] = ""
    return out


TRANSCRIPT_TYPES = {
    "mRNA", "transcript", "lnc_RNA", "lncRNA", "ncRNA", "rRNA", "tRNA", "snRNA", "snoRNA", "miRNA",
    "primary_transcript", "C_gene_segment", "V_gene_segment", "D_gene_segment", "J_gene_segment",
    "antisense_RNA", "guide_RNA", "scRNA", "scaRNA", "RNase_MRP_RNA", "RNase_P_RNA", "telomerase_RNA",
    "vault_RNA", "Y_RNA", "piRNA", "SRP_RNA", "unconfirmed_transcript",
    "pseudogenic_transcript", "pseudogenic_rRNA", "pseudogenic_tRNA",
}

STRUCTURAL_NCRNA_BIOTYPES = ("tRNA", "rRNA", "snRNA", "snoRNA")
EVAL_NCRNA_BIOTYPES = ("lncRNA", "tRNA", "rRNA", "snRNA", "snoRNA")
PROTEIN_PSEUDO_BIOTYPES = frozenset({
    "pseudogene",
    "transcribed_pseudogene",
    "processed_pseudogene",
    "unprocessed_pseudogene",
    "unitary_pseudogene",
    "polymorphic_pseudogene",
})
NCRNA_PSEUDO_BIOTYPES = frozenset({
    "rRNA_pseudogene",
    "ncRNA_pseudogene",
    "tRNA_pseudogene",
    "snRNA_pseudogene",
    "snoRNA_pseudogene",
    "miRNA_pseudogene",
    "misc_RNA_pseudogene",
})


def normalize_biotype(biotype: str) -> str:
    """Canonicalize biotype labels across RefSeq and EGAPx conventions."""
    bt = (biotype or "").strip()
    if bt == "lnc_RNA":
        return "lncRNA"
    if bt == "gene":
        return ""
    return bt


def _pseudo_flag(t: Transcript) -> bool:
    """True if transcript or parent gene carries pseudogene metadata (ignoring CDS presence)."""
    bt = normalize_biotype(t.biotype)
    gb = normalize_biotype(t.attrs.get("gene_biotype", ""))
    ttype = t.attrs.get("_type", "")
    return (
        t.attrs.get("_gene_type") == "pseudogene"
        or bt.endswith("pseudogene")
        or gb.endswith("pseudogene")
        or t.attrs.get("pseudo", "").lower() == "true"
        or ttype.startswith("pseudogenic_")
    )


def is_pseudogene_model(t: Transcript) -> bool:
    """True for CDS-free pseudogene records under RefSeq, EGAPx, or SO GFF3 conventions."""
    return not t.cds and _pseudo_flag(t)


def effective_biotype(t: Transcript) -> str:
    """Return canonical biotype for a transcript model."""
    bt = normalize_biotype(t.biotype or t.attrs.get("gene_biotype", ""))
    if t.cds:
        return bt or "protein_coding"
    ttype = t.attrs.get("_type", "")
    if is_pseudogene_model(t):
        if bt.endswith("pseudogene"):
            return bt
        gbkey = t.attrs.get("gbkey", "")
        if ttype in ("pseudogenic_rRNA", "rRNA") or gbkey == "rRNA":
            return "rRNA_pseudogene"
        if ttype in ("pseudogenic_tRNA", "tRNA") or gbkey == "tRNA":
            return "tRNA_pseudogene"
        if ttype in ("snRNA", "snoRNA"):
            return f"{ttype}_pseudogene"
        if ttype == "ncRNA":
            return "ncRNA_pseudogene"
        return "pseudogene"
    if bt:
        return bt
    if ttype in ("lnc_RNA", "lncRNA"):
        return "lncRNA"
    if ttype and ttype not in ("mRNA", "transcript"):
        return ttype
    return "ncRNA"


def pseudogene_bucket(t: Transcript) -> str | None:
    """Classify a pseudogene model into 'pseudo_protein' or 'pseudo_ncrna' (or None if not pseudogene)."""
    if not is_pseudogene_model(t):
        return None
    bt = effective_biotype(t)
    if bt in PROTEIN_PSEUDO_BIOTYPES:
        return "pseudo_protein"
    ttype = t.attrs.get("_type", "")
    gbkey = t.attrs.get("gbkey", "")
    if (
        bt in NCRNA_PSEUDO_BIOTYPES
        or bt.startswith(("rRNA_", "tRNA_", "snRNA_", "snoRNA_", "ncRNA_", "miRNA_"))
        or ttype in ("pseudogenic_rRNA", "pseudogenic_tRNA", "rRNA", "tRNA", "snRNA", "snoRNA", "ncRNA")
        or gbkey in ("rRNA", "tRNA")
    ):
        return "pseudo_ncrna"
    return "pseudo_protein"


def count_models_by_biotype(models: Iterable[Transcript]) -> dict:
    """Count genes and transcripts cleanly partitioned across coding, ncRNA biotypes, and pseudogene buckets."""
    tx_list = list(models)
    genes: dict[str, list[Transcript]] = {}
    for t in tx_list:
        genes.setdefault(t.gid or t.tid, []).append(t)
    n_coding_genes = 0
    n_coding_tx = sum(1 for t in tx_list if t.cds)
    ncrna_genes_by_bt: dict[str, int] = {bt: 0 for bt in EVAL_NCRNA_BIOTYPES}
    ncrna_tx_by_bt: dict[str, int] = {bt: 0 for bt in EVAL_NCRNA_BIOTYPES}
    n_pseudo_prot_genes = 0
    n_pseudo_prot_tx = 0
    n_pseudo_ncrna_genes = 0
    n_pseudo_ncrna_tx = 0
    for txs in genes.values():
        if any(t.cds for t in txs):
            n_coding_genes += 1
            continue
        pb = pseudogene_bucket(txs[0])
        if pb == "pseudo_protein":
            n_pseudo_prot_genes += 1
            n_pseudo_prot_tx += len(txs)
        elif pb == "pseudo_ncrna":
            n_pseudo_ncrna_genes += 1
            n_pseudo_ncrna_tx += len(txs)
        else:
            bt = effective_biotype(txs[0])
            ncrna_genes_by_bt[bt] = ncrna_genes_by_bt.get(bt, 0) + 1
            ncrna_tx_by_bt[bt] = ncrna_tx_by_bt.get(bt, 0) + len(txs)
    return {
        "n_genes": len(genes),
        "n_transcripts": len(tx_list),
        "n_coding_genes": n_coding_genes,
        "n_coding_transcripts": n_coding_tx,
        "n_lnc_genes": ncrna_genes_by_bt["lncRNA"],
        "n_lnc_transcripts": ncrna_tx_by_bt["lncRNA"],
        "n_ncrna_genes": sum(ncrna_genes_by_bt.values()),
        "n_ncrna_transcripts": sum(ncrna_tx_by_bt.values()),
        "n_ncrna_by_biotype": ncrna_genes_by_bt,
        "n_ncrna_transcripts_by_biotype": ncrna_tx_by_bt,
        "n_pseudo_genes": n_pseudo_prot_genes + n_pseudo_ncrna_genes,
        "n_pseudo_transcripts": n_pseudo_prot_tx + n_pseudo_ncrna_tx,
        "n_pseudo_protein_genes": n_pseudo_prot_genes,
        "n_pseudo_protein_transcripts": n_pseudo_prot_tx,
        "n_pseudo_ncrna_genes": n_pseudo_ncrna_genes,
        "n_pseudo_ncrna_transcripts": n_pseudo_ncrna_tx,
    }


def read_models(path: str | Path, fmt: str | None = None, keep_types: set[str] | None = None) -> list[Transcript]:
    """Read transcript models from GTF or GFF3 (auto-detected by extension when fmt is None)."""
    p = str(path)
    if fmt is None:
        base = p[:-3] if p.endswith(".gz") else p
        fmt = "gtf" if base.endswith(".gtf") else "gff3"
    is_gtf = fmt == "gtf"
    tx: dict[str, Transcript] = {}
    order: list[str] = []
    stops: dict[str, list[Interval]] = {}
    cds_phases: dict[str, list[tuple[int, int, int]]] = {}
    gene_biotype: dict[str, str] = {}
    gene_ftype: dict[str, str] = {}
    gene_pseudo: dict[str, bool] = {}
    parent_of: dict[str, str] = {}
    pseudo_gene_records: dict[str, tuple[str, str, str, int, int, dict[str, str]]] = {}
    pseudo_gene_order: list[str] = []
    with _open(p) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            ctg, src, ftype, s, e, _score, strand, _phase, col9 = f
            a, b = int(s) - 1, int(e)
            at = parse_attrs(col9, is_gtf)
            if is_gtf:
                tid = at.get("transcript_id")
                if not tid:
                    continue
                t = tx.get(tid)
                if t is None:
                    t = Transcript(tid=tid, contig=ctg, strand=strand, gid=at.get("gene_id", tid), source=src)
                    tx[tid] = t
                    order.append(tid)
                gbt = normalize_biotype(
                    at.get("transcript_biotype", "") or at.get("gene_biotype", "") or at.get("biotype", "")
                )
                if gbt and not t.biotype:
                    t.biotype = gbt
                if at.get("pseudo", "").lower() == "true":
                    t.attrs["pseudo"] = "true"
                if ftype == "exon":
                    t.exons.append((a, b))
                elif ftype == "CDS":
                    t.cds.append((a, b))
                    if _phase in ("0", "1", "2"):
                        cds_phases.setdefault(tid, []).append((a, b, int(_phase)))
                elif ftype == "stop_codon":
                    stops.setdefault(tid, []).append((a, b))
                elif ftype == "transcript":
                    t.attrs.update(at)
                    if gbt:
                        t.biotype = gbt
            else:
                fid = at.get("ID", "")
                if fid and "Parent" in at:
                    parent_of[fid] = at["Parent"].split(",", 1)[0]
                if ftype in ("gene", "pseudogene"):
                    if fid:
                        gene_ftype[fid] = ftype
                        raw_gb = normalize_biotype(at.get("gene_biotype", ""))
                        is_gp = (ftype == "pseudogene") or (at.get("pseudo", "").lower() == "true") or raw_gb.endswith("pseudogene")
                        gene_pseudo[fid] = is_gp
                        gene_biotype[fid] = raw_gb
                        if ftype == "pseudogene" and fid not in pseudo_gene_records:
                            pseudo_gene_records[fid] = (ctg, src, strand, a, b, dict(at))
                            pseudo_gene_order.append(fid)
                    continue
                if keep_types is not None and ftype in TRANSCRIPT_TYPES and ftype not in keep_types:
                    continue
                if ftype in TRANSCRIPT_TYPES or (ftype not in ("exon", "CDS", "stop_codon", "start_codon",
                                                                 "five_prime_UTR", "three_prime_UTR", "intron")
                                                  and "Parent" in at and "ID" in at and ftype.endswith(("RNA", "transcript"))):
                    tid = at.get("ID")
                    if not tid:
                        continue
                    t = tx.get(tid)
                    if t is None:
                        t = Transcript(tid=tid, contig=ctg, strand=strand, source=src)
                        tx[tid] = t
                        order.append(tid)
                    t.gid = at.get("Parent", tid)
                    t.attrs.update(at)
                    t.attrs["_type"] = ftype
                    continue
                if ftype in ("exon", "CDS", "stop_codon"):
                    parents = at.get("Parent", "")
                    if not parents:
                        continue
                    for par in parents.split(","):
                        t = tx.get(par)
                        if t is None:
                            # Feature appears before its parent (rare) or parent is a gene (CDS/exon directly under gene)
                            t = Transcript(tid=par, contig=ctg, strand=strand, source=src)
                            tx[par] = t
                            order.append(par)
                        if at.get("pseudo", "").lower() == "true":
                            t.attrs.setdefault("pseudo", "true")
                        if at.get("gbkey") and "gbkey" not in t.attrs:
                            t.attrs["gbkey"] = at["gbkey"]
                        if ftype == "exon":
                            t.exons.append((a, b))
                        elif ftype == "CDS":
                            t.cds.append((a, b))
                            if _phase in ("0", "1", "2"):
                                cds_phases.setdefault(par, []).append((a, b, int(_phase)))
                        else:
                            stops.setdefault(par, []).append((a, b))

    def _root_gene_id(gid: str) -> str:
        cur = gid
        seen: set[str] = set()
        while cur and cur not in gene_ftype and cur in parent_of and cur not in seen:
            seen.add(cur)
            cur = parent_of[cur]
        return cur

    out = []
    for tid in order:
        t = tx[tid]
        if not t.exons and not t.cds:
            continue
        # Merge separate stop codon into CDS (GTF convention excludes it)
        if t.cds and tid in stops:
            cs = merge_intervals(t.cds)
            for sa, sb in stops[tid]:
                inside = any(a <= sa and sb <= b for a, b in cs)
                if not inside:
                    cs.append((sa, sb))
            t.cds = cs
        t.normalize()
        if t.cds and tid in cds_phases:
            ph_segs = cds_phases[tid]
            first_seg = max(ph_segs, key=lambda x: (x[1], x[0])) if t.strand == "-" else min(ph_segs, key=lambda x: (x[0], x[1]))
            t.cds_phase = first_seg[2]
        if t.cds and t.exons:
            # Ensure exons cover CDS (CDS-only GTF from ab initio predictors)
            covered = all(any(ea <= a and b <= eb for ea, eb in t.exons) for a, b in t.cds)
            if not covered:
                t.exons = merge_intervals(t.exons + t.cds)
        if not t.gid:
            t.gid = t.tid
        rgid = _root_gene_id(t.gid)
        if rgid in gene_ftype:
            t.attrs["_gene_type"] = gene_ftype[rgid]
        if gene_pseudo.get(rgid, False):
            t.attrs.setdefault("pseudo", "true")
        if rgid in gene_biotype and gene_biotype[rgid]:
            t.biotype = gene_biotype[rgid]
        elif t.biotype:
            t.biotype = normalize_biotype(t.biotype)
        else:
            t.biotype = normalize_biotype(
                t.attrs.get("transcript_biotype", "") or t.attrs.get("gene_biotype", "") or t.attrs.get("biotype", "")
            )
        if not t.cds and is_pseudogene_model(t) and not t.biotype.endswith("pseudogene"):
            t.biotype = effective_biotype(t)
        elif not t.cds and not t.biotype:
            ttype = t.attrs.get("_type", "")
            if ttype in ("lnc_RNA", "lncRNA"):
                t.biotype = "lncRNA"
            elif ttype in STRUCTURAL_NCRNA_BIOTYPES or (ttype in TRANSCRIPT_TYPES and ttype not in ("mRNA", "transcript")):
                t.biotype = ttype
        out.append(t)
    if not is_gtf and pseudo_gene_order and (keep_types is None or "pseudogene" in keep_types):
        represented_genes: set[str] = set()
        for t in out:
            represented_genes.add(t.tid)
            represented_genes.add(t.gid)
            represented_genes.add(_root_gene_id(t.gid))
        for fid in pseudo_gene_order:
            if fid in represented_genes:
                continue
            ctg_p, src_p, strand_p, a_p, b_p, at_p = pseudo_gene_records[fid]
            if b_p <= a_p:
                continue
            raw_gb = gene_biotype.get(fid, "") or normalize_biotype(at_p.get("gene_biotype", "") or at_p.get("biotype", ""))
            mat_attrs = {**at_p, "_gene_type": "pseudogene"}
            mat_attrs.setdefault("pseudo", "true")
            pt = Transcript(
                tid=fid,
                contig=ctg_p,
                strand=strand_p,
                exons=[(a_p, b_p)],
                cds=[],
                gid=fid,
                source=src_p,
                attrs=mat_attrs,
                biotype=raw_gb,
            )
            if not pt.biotype.endswith("pseudogene"):
                pt.biotype = effective_biotype(pt)
            out.append(pt)
    return out


def _phases(t: Transcript) -> list[int]:
    """GFF3 phase per CDS segment in genomic order."""
    segs = sorted(t.cds)
    order = segs if t.strand != "-" else list(reversed(segs))
    ph: dict[Interval, int] = {}
    cur_ph = int(getattr(t, "cds_phase", 0) or 0) % 3
    for seg in order:
        ph[seg] = cur_ph
        cur_ph = (cur_ph - (seg[1] - seg[0])) % 3
    return [ph[s] for s in segs]


_NCRNA_SO_FEATURE = {
    "lncRNA": "lnc_RNA",
    "tRNA": "tRNA",
    "rRNA": "rRNA",
    "snRNA": "snRNA",
    "snoRNA": "snoRNA",
    "miRNA": "miRNA",
    "antisense_RNA": "antisense_RNA",
    "scaRNA": "scaRNA",
    "scRNA": "scRNA",
    "RNase_MRP_RNA": "RNase_MRP_RNA",
    "RNase_P_RNA": "RNase_P_RNA",
    "SRP_RNA": "SRP_RNA",
    "telomerase_RNA": "telomerase_RNA",
    "vault_RNA": "vault_RNA",
    "Y_RNA": "Y_RNA",
    "piRNA": "piRNA",
    "guide_RNA": "guide_RNA",
}
EXPLICIT_NONCODING_BIOTYPES = frozenset({"ncRNA", "misc_RNA", *_NCRNA_SO_FEATURE.keys()})


def write_gff3(models: Iterable[Transcript], path: str | Path, source: str = "GeneLoop") -> int:
    """Write multi-biotype GFF3 (protein_coding, lncRNA, structural ncRNAs, pseudogene).

    Enforces two strict invariants:
      1. Non-coding and pseudogene transcripts must never carry CDS features.
      2. A gene (gid) must never mix CDS-bearing and CDS-free transcripts or multiple biotypes.
    """
    genes: dict[str, list[Transcript]] = {}
    for t in models:
        raw_bt = normalize_biotype(t.biotype or t.attrs.get("gene_biotype", ""))
        if t.cds and (_pseudo_flag(t) or raw_bt in EXPLICIT_NONCODING_BIOTYPES):
            raise ValueError(f"Transcript {t.tid!r} with biotype={raw_bt or 'pseudogene'!r} must not carry CDS features")
        if not t.cds and raw_bt == "protein_coding":
            raise ValueError(f"Transcript {t.tid!r} with biotype='protein_coding' has empty CDS")
        genes.setdefault(t.gid or t.tid, []).append(t)

    gl = sorted(genes.items(), key=lambda kv: (kv[1][0].contig, min(x.start for x in kv[1])))
    n = 0
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for gid, txs in gl:
            has_cds = [bool(t.cds) for t in txs]
            if any(has_cds) and not all(has_cds):
                raise ValueError(f"Gene {gid!r} mixes CDS-bearing and CDS-free transcripts")
            bts = {effective_biotype(t) for t in txs}
            if len(bts) > 1:
                raise ValueError(f"Gene {gid!r} mixes multiple biotypes: {sorted(bts)}")
            biot = next(iter(bts))
            is_pseudo = is_pseudogene_model(txs[0])
            ctg = txs[0].contig
            strand = txs[0].strand
            gs = min(t.start for t in txs)
            ge = max(t.end for t in txs)
            if is_pseudo:
                fh.write(f"{ctg}\t{source}\tpseudogene\t{gs + 1}\t{ge}\t.\t{strand}\t.\tID={gid};gene_biotype={biot};pseudo=true\n")
            else:
                fh.write(f"{ctg}\t{source}\tgene\t{gs + 1}\t{ge}\t.\t{strand}\t.\tID={gid};gene_biotype={biot}\n")
            for t in txs:
                skip_keys = {"ID", "Parent", "gene_biotype", "pseudo"} if is_pseudo else {"ID", "Parent", "gene_biotype"}
                extra = "".join(f";{k}={v}" for k, v in t.attrs.items() if not k.startswith("_") and k not in skip_keys)
                if t.cds:
                    ttype = "mRNA"
                    t_attrs = f"ID={t.tid};Parent={gid}{extra}"
                elif is_pseudo:
                    ttype = "pseudogenic_transcript"
                    t_attrs = f"ID={t.tid};Parent={gid};pseudo=true{extra}"
                else:
                    ttype = _NCRNA_SO_FEATURE.get(biot, "transcript")
                    t_attrs = f"ID={t.tid};Parent={gid}{extra}"
                fh.write(f"{ctg}\t{t.source or source}\t{ttype}\t{t.start + 1}\t{t.end}\t.\t{strand}\t.\t{t_attrs}\n")
                for i, (a, b) in enumerate(sorted(t.exons)):
                    fh.write(f"{ctg}\t{t.source or source}\texon\t{a + 1}\t{b}\t.\t{strand}\t.\tID={t.tid}.exon{i + 1};Parent={t.tid}\n")
                if t.cds:
                    for (a, b), ph in zip(sorted(t.cds), _phases(t)):
                        fh.write(f"{ctg}\t{t.source or source}\tCDS\t{a + 1}\t{b}\t.\t{strand}\t{ph}\tID={t.tid}.cds;Parent={t.tid}\n")
                n += 1
    return n


def write_gtf(models: Iterable[Transcript], path: str | Path, cds_only: bool = False, source: str = "GeneLoop") -> int:
    """Write GTF with gene_biotype, transcript_biotype, and pseudo attributes."""
    n = 0
    with open(path, "w") as fh:
        for t in sorted(models, key=lambda x: (x.contig, x.start)):
            blocks = sorted(t.cds) if cds_only else sorted(t.exons)
            if not blocks:
                continue
            gid = t.gid or t.tid
            a0, b0 = blocks[0][0], blocks[-1][1]
            biot = effective_biotype(t)
            g_biot = normalize_biotype(t.attrs.get("gene_biotype", "")) or biot
            is_pseudo = is_pseudogene_model(t)
            ga = (
                f'gene_id "{gid}"; transcript_id "{t.tid}"; '
                f'gene_biotype "{g_biot}"; transcript_biotype "{biot}";'
            )
            if is_pseudo:
                ga += ' pseudo "true";'
            fh.write(f"{t.contig}\t{source}\ttranscript\t{a0 + 1}\t{b0}\t.\t{t.strand}\t.\t{ga}\n")
            for a, b in blocks:
                fh.write(f"{t.contig}\t{source}\texon\t{a + 1}\t{b}\t.\t{t.strand}\t.\t{ga}\n")
            if not cds_only and t.cds:
                for (a, b), ph in zip(sorted(t.cds), _phases(t)):
                    fh.write(f"{t.contig}\t{source}\tCDS\t{a + 1}\t{b}\t.\t{t.strand}\t{ph}\t{ga}\n")
            n += 1
    return n


def iter_overlaps(sorted_models: list[Transcript]) -> Iterator[list[Transcript]]:
    """Yield clusters of transcripts with overlapping spans (same contig + strand), input sorted by (contig,strand,start)."""
    cur: list[Transcript] = []
    cur_end = -1
    key = None
    for t in sorted_models:
        k = (t.contig, t.strand)
        if cur and (k != key or t.start >= cur_end):
            yield cur
            cur = []
            cur_end = -1
        cur.append(t)
        key = k
        cur_end = max(cur_end, t.end)
    if cur:
        yield cur
