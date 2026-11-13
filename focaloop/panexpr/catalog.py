"""Unified target catalog: multi-strain prokaryotic CDS + decoys, or spliced eukaryotic cDNA + decoys.

Targets carry an ``assembly_id`` (strain / assembly of origin), a ``gene_id``, a
``transcript_id`` and a cross-assembly ``group_id`` (ortholog group, the
``tx2gene`` generalisation).  Grouping is done by exact sequence identity first,
then by reciprocal-best-hit alignment (libminimap2 ``asm20``), or from a
user-supplied TSV.  Decoys (intergenic, intron, genomic) absorb reads that do
not come from annotated transcripts so they cannot inflate gene counts.

Also provides the Soneson et al. 2015 ``tximport`` gene-level effective-length
offsets used by the DESeq2 adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_GTF_ATTR_RE = re.compile(r'(\S+)\s+"([^"]*)"')
_COMP = str.maketrans("ACGTNacgtnRYKMSWBDHV", "TGCANtgcanYRMKSWVHDB")
DECOY_BIOTYPES = ("decoy_intergenic", "decoy_intron", "decoy_genomic")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def _open_text(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load_fasta(path: str | Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    cur = None
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip().upper())
    return {k: "".join(v) for k, v in seqs.items()}


def write_fasta(path: str | Path, records: Iterable[tuple[str, str]], width: int = 0) -> None:
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            if width and width > 0:
                for i in range(0, len(seq), width):
                    fh.write(seq[i:i + width] + "\n")
            else:
                fh.write(seq + "\n")


@dataclass
class Feature:
    contig: str
    ftype: str
    start: int      # 1-based inclusive
    end: int
    strand: str
    attrs: dict[str, str]


def _parse_attrs(attr_s: str, is_gff3: bool) -> dict[str, str]:
    if is_gff3:
        out: dict[str, str] = {}
        for part in attr_s.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out
    return dict(_GTF_ATTR_RE.findall(attr_s))


def iter_features(path: str | Path) -> Iterator[Feature]:
    """Iterate GTF or GFF3 features (gz-aware). Format is sniffed from the attribute column."""
    is_gff3 = str(path).endswith((".gff", ".gff3", ".gff.gz", ".gff3.gz"))
    with _open_text(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            attrs = _parse_attrs(parts[8], is_gff3 or ("=" in parts[8] and '"' not in parts[8]))
            yield Feature(parts[0], parts[2], int(parts[3]), int(parts[4]), parts[6], attrs)


@dataclass
class Target:
    target_id: str
    assembly_id: str
    gene_id: str
    transcript_id: str
    group_id: str
    symbol: str
    biotype: str
    contig: str
    start_1b: int
    end_1b: int
    strand: str
    seq: str
    product: str = ""
    exons: list[tuple[int, int]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def is_decoy(self) -> bool:
        return self.biotype in DECOY_BIOTYPES


@dataclass
class AssemblySpec:
    assembly_id: str
    fasta: str | Path
    annotation: str | Path
    only_genes: set[str] | None = None
    symbol_overrides: dict[str, str] = field(default_factory=dict)
    id_attr: tuple[str, ...] = ("locus_tag", "gene_id", "ID", "Name")


@dataclass
class ExtraWindow:
    """An explicit genomic window target (e.g. an operon intergenic junction)."""
    assembly_id: str
    contig: str
    start_1b: int
    end_1b: int
    strand: str
    target_name: str
    symbol: str
    biotype: str = "window"
    product: str = ""


def _gene_id_from(attrs: dict[str, str], id_attr: tuple[str, ...]) -> str:
    for k in id_attr:
        v = attrs.get(k)
        if v:
            return v.split(":")[-1] if k == "ID" and v.startswith("gene:") else v
    return ""


# ---------------------------------------------------------------------------
# Prokaryotic mode
# ---------------------------------------------------------------------------

def build_prokaryotic_targets(spec: AssemblySpec) -> list[Target]:
    """One target per annotated gene (CDS/rRNA/tRNA/...): whole gene body, strand-oriented."""
    genome = load_fasta(spec.fasta)
    genes: dict[str, Feature] = {}
    products: dict[str, str] = {}
    for f in iter_features(spec.annotation):
        gid = _gene_id_from(f.attrs, spec.id_attr)
        if not gid:
            continue
        if spec.only_genes is not None and gid not in spec.only_genes:
            continue
        if f.ftype == "gene":
            genes[gid] = f
        elif f.ftype in ("CDS", "transcript", "exon", "mRNA", "rRNA", "tRNA") and "product" in f.attrs and gid not in products:
            products[gid] = f.attrs["product"]
        elif f.ftype in ("CDS",) and gid not in genes:
            genes[gid] = f  # annotation with CDS but no gene lines
    out: list[Target] = []
    for gid, f in genes.items():
        ctg = genome.get(f.contig)
        if not ctg:
            continue
        raw = ctg[max(0, f.start - 1):f.end]
        if not raw:
            continue
        seq = revcomp(raw) if f.strand == "-" else raw
        biotype = f.attrs.get("gene_biotype") or f.attrs.get("biotype") or ("protein_coding" if f.ftype in ("gene", "CDS") else f.ftype)
        sym = spec.symbol_overrides.get(gid) or f.attrs.get("gene") or f.attrs.get("Name") or f.attrs.get("gene_name") or gid
        out.append(Target(
            target_id=f"{spec.assembly_id}|{gid}", assembly_id=spec.assembly_id, gene_id=gid, transcript_id=gid,
            group_id="", symbol=sym, biotype=biotype, contig=f.contig, start_1b=f.start, end_1b=f.end,
            strand=f.strand, seq=seq, product=products.get(gid, ""),
        ))
    return out


def build_intergenic_decoys(
    spec: AssemblySpec,
    gene_targets: list[Target],
    min_len: int = 150,
    chunk: int = 1500,
    use_ust_decoys: bool = False,
) -> list[Target]:
    """Intergenic windows of the assembly (both strands implicit) as decoy targets."""
    genome = load_fasta(spec.fasta)
    by_ctg: dict[str, list[tuple[int, int]]] = {}
    tx_seqs: list[str] = []
    for t in gene_targets:
        if t.assembly_id == spec.assembly_id:
            by_ctg.setdefault(t.contig, []).append((t.start_1b, t.end_1b))
            tx_seqs.append(t.seq)
    if use_ust_decoys:
        from .ust_simplitigs import build_ust_decoy_simplitigs_chunked, pack_n_spacer_super_decoys

        raw_gaps: list[str] = []
        for ctg, seq in genome.items():
            ivs = sorted(by_ctg.get(ctg, []))
            cur = 1
            for s, e in ivs:
                if s - cur >= 31:
                    raw_gaps.append(seq[cur - 1:s - 1])
                cur = max(cur, e + 1)
            if len(seq) - cur + 1 >= 31:
                raw_gaps.append(seq[cur - 1:])
        simplitigs, _ = build_ust_decoy_simplitigs_chunked(raw_gaps, tx_seqs, k=31)
        super_seqs = pack_n_spacer_super_decoys(simplitigs, spacer_len=64, super_chunk_bp=262_144)
        out_ust: list[Target] = []
        for idx, s in enumerate(super_seqs):
            name = f"DECOY_UST_IG_{idx}"
            out_ust.append(Target(
                target_id=f"{spec.assembly_id}|{name}", assembly_id=spec.assembly_id, gene_id=name, transcript_id=name,
                group_id=f"DECOY_{spec.assembly_id}_{name}", symbol=name, biotype="decoy_intergenic", contig="UST",
                start_1b=1, end_1b=len(s), strand="+", seq=s,
            ))
        return out_ust
    out: list[Target] = []
    for ctg, seq in genome.items():
        ivs = sorted(by_ctg.get(ctg, []))
        cur = 1
        gaps: list[tuple[int, int]] = []
        for s, e in ivs:
            if s - cur >= min_len:
                gaps.append((cur, s - 1))
            cur = max(cur, e + 1)
        if len(seq) - cur + 1 >= min_len:
            gaps.append((cur, len(seq)))
        for gs, ge in gaps:
            for cs in range(gs, ge + 1, chunk):
                ce = min(ge, cs + chunk - 1)
                if ce - cs + 1 < min_len:
                    continue
                name = f"DECOY_IG_{ctg}_{cs}_{ce}"
                out.append(Target(
                    target_id=f"{spec.assembly_id}|{name}", assembly_id=spec.assembly_id, gene_id=name, transcript_id=name,
                    group_id=f"DECOY_{spec.assembly_id}_{name}", symbol=name, biotype="decoy_intergenic", contig=ctg,
                    start_1b=cs, end_1b=ce, strand="+", seq=seq[cs - 1:ce],
                ))
    return out


# ---------------------------------------------------------------------------
# Spliced eukaryotic mode
# ---------------------------------------------------------------------------

def _tx_key(attrs: dict[str, str]) -> str:
    return attrs.get("transcript_id") or attrs.get("Parent") or ""


def build_spliced_transcripts(spec: AssemblySpec, min_exon_total: int = 30) -> list[Target]:
    """Mature spliced cDNA for every transcript (pure-Python exon splicer, +/- strand safe)."""
    genome = load_fasta(spec.fasta)
    exons: dict[str, list[Feature]] = {}
    tx_gene: dict[str, str] = {}
    tx_biotype: dict[str, str] = {}
    gene_sym: dict[str, str] = {}
    gene_biotype: dict[str, str] = {}
    gene_product: dict[str, str] = {}
    mrna_parent: dict[str, str] = {}  # GFF3: transcript ID -> gene ID
    gff3_gene_ids: dict[str, str] = {}
    for f in iter_features(spec.annotation):
        a = f.attrs
        if f.ftype == "gene":
            gid = _gene_id_from(a, spec.id_attr)
            if gid:
                gene_sym[gid] = spec.symbol_overrides.get(gid) or a.get("gene") or a.get("gene_name") or a.get("Name") or gid
                gene_biotype[gid] = a.get("gene_biotype") or a.get("biotype") or "protein_coding"
                if "ID" in a:
                    gff3_gene_ids[a["ID"]] = gid
        elif f.ftype in ("mRNA", "transcript", "rRNA", "tRNA", "ncRNA", "lnc_RNA", "snoRNA", "snRNA") and "ID" in a and "Parent" in a:
            mrna_parent[a["ID"]] = a["Parent"]
            if "product" in a:
                gene_product.setdefault(a["Parent"], a["product"])
        elif f.ftype == "exon":
            tk = _tx_key(a)
            if not tk:
                continue
            exons.setdefault(tk, []).append(f)
            gid = a.get("gene_id") or ""
            if gid:
                tx_gene[tk] = gid
            if "transcript_biotype" in a:
                tx_biotype[tk] = a["transcript_biotype"]
        elif f.ftype == "CDS" and "product" in a:
            gid = a.get("gene_id") or mrna_parent.get(a.get("Parent", ""), "")
            if gid:
                gene_product.setdefault(gid, a["product"])
    out: list[Target] = []
    for tk, exs in exons.items():
        gid = tx_gene.get(tk)
        if not gid:
            parent = mrna_parent.get(tk, "")
            gid = gff3_gene_ids.get(parent, parent) or tk
        gid = gid.split(":")[-1] if gid.startswith("gene:") else gid
        if spec.only_genes is not None and gid not in spec.only_genes:
            continue
        ctg_seq = genome.get(exs[0].contig)
        if not ctg_seq:
            continue
        strand = exs[0].strand
        ivs = sorted({(e.start, e.end) for e in exs})
        pieces = [ctg_seq[s - 1:e] for s, e in ivs]
        cdna = "".join(pieces)
        if len(cdna) < min_exon_total:
            continue
        if strand == "-":
            cdna = revcomp(cdna)
        tid = tk.split(":")[-1] if tk.startswith(("rna:", "transcript:")) else tk
        safe_tid = tid.replace("|", "_").replace(" ", "_")
        out.append(Target(
            target_id=f"{spec.assembly_id}|{safe_tid}", assembly_id=spec.assembly_id, gene_id=gid, transcript_id=safe_tid,
            group_id="", symbol=gene_sym.get(gid, gid), biotype=tx_biotype.get(tk) or gene_biotype.get(gid, "protein_coding"),
            contig=exs[0].contig, start_1b=ivs[0][0], end_1b=ivs[-1][1], strand=strand, seq=cdna,
            product=gene_product.get(gid, ""), exons=ivs,
        ))
    return out


def build_intron_decoys(spec: AssemblySpec, transcripts: list[Target], min_len: int = 60, chunk: int = 2000, flank: int = 20) -> list[Target]:
    """Intron (retention) decoys: each intron with ``flank`` bp of adjoining exon on either side."""
    genome = load_fasta(spec.fasta)
    seen: set[tuple[str, int, int]] = set()
    out: list[Target] = []
    for t in transcripts:
        if t.assembly_id != spec.assembly_id or len(t.exons) < 2:
            continue
        ctg = genome.get(t.contig)
        if not ctg:
            continue
        for (s1, e1), (s2, e2) in zip(t.exons[:-1], t.exons[1:]):
            istart, iend = e1 + 1, s2 - 1
            if iend - istart + 1 < min_len:
                continue
            key = (t.contig, istart, iend)
            if key in seen:
                continue
            seen.add(key)
            ws, we = max(1, istart - flank), min(len(ctg), iend + flank)
            for cs in range(ws, we + 1, chunk):
                ce = min(we, cs + chunk - 1)
                if ce - cs + 1 < min_len:
                    continue
                name = f"DECOY_INTRON_{t.contig}_{cs}_{ce}"
                out.append(Target(
                    target_id=f"{spec.assembly_id}|{name}", assembly_id=spec.assembly_id, gene_id=name, transcript_id=name,
                    group_id=f"DECOY_{spec.assembly_id}_{name}", symbol=name, biotype="decoy_intron", contig=t.contig,
                    start_1b=cs, end_1b=ce, strand="+", seq=ctg[cs - 1:ce],
                ))
    return out


def _merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ivs:
        return []
    s_ivs = sorted(ivs)
    merged = [s_ivs[0]]
    for a, b in s_ivs[1:]:
        if a <= merged[-1][1] + 1:
            if b > merged[-1][1]:
                merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _subtract_intervals(spans: list[tuple[int, int]], masks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans or not masks:
        return list(spans)
    out: list[tuple[int, int]] = []
    j = 0
    n_m = len(masks)
    for a, b in spans:
        cur = a
        while j < n_m and masks[j][1] < cur:
            j += 1
        k = j
        while k < n_m and masks[k][0] <= b:
            ma, mb = masks[k]
            if ma > cur:
                out.append((cur, min(b, ma - 1)))
            cur = max(cur, mb + 1)
            if cur > b:
                break
            k += 1
        if cur <= b:
            out.append((cur, b))
    return out


def build_merged_intron_decoys(
    spec: AssemblySpec,
    transcripts: list[Target],
    min_len: int = 60,
    super_chunk_bp: int = 262_144,
    flank: int = 20,
    spacer_len: int = 64,
    use_ust_decoys: bool = False,
) -> list[Target]:
    """v2.1 compact exon-subtracted intronic super-decoys.

    Overlapping isoform introns on each contig are merged, strictly internal annotated exonic cores
    (shrunk by ``flank`` bp so splice-junction overhangs remain in the decoy) are subtracted so
    alternatively spliced exons never collide with decoys, and adjacent intronic spans are packed
    into ``super_chunk_bp`` super-targets separated by ``spacer_len`` Ns (> k=31).  Exact 1-based
    genomic intervals are recorded in ``Target.exons`` for genome-projection kernels.

    When ``use_ust_decoys=True``, subtracts 100% of canonical transcript 31-mers (zero leaked
    exonic 31-mers) and compacts surviving intronic/flank 31-mers into minimal bidirected
    Rahman-Medvedev UST simplitigs before packing into N-spacer super-targets.
    """
    genome = load_fasta(spec.fasta)
    raw_introns: dict[str, list[tuple[int, int]]] = {}
    raw_exons: dict[str, list[tuple[int, int]]] = {}
    tx_seqs: list[str] = []
    for t in transcripts:
        if t.assembly_id != spec.assembly_id:
            continue
        tx_seqs.append(t.seq)
        if not t.exons:
            continue
        raw_exons.setdefault(t.contig, []).extend(t.exons)
        if len(t.exons) < 2:
            continue
        for (_s1, e1), (s2, _e2) in zip(t.exons[:-1], t.exons[1:]):
            istart, iend = e1 + 1, s2 - 1
            if iend - istart + 1 >= min_len:
                raw_introns.setdefault(t.contig, []).append((max(1, istart - flank), iend + flank))
    if use_ust_decoys:
        from .ust_simplitigs import build_ust_decoy_simplitigs_chunked, pack_n_spacer_super_decoys

        intron_frags: list[str] = []
        for ctg in sorted(raw_introns):
            seq = genome.get(ctg)
            if not seq:
                continue
            ctg_len = len(seq)
            clamped = [(max(1, a), min(ctg_len, b)) for a, b in raw_introns[ctg] if a <= ctg_len]
            for a, b in _merge_intervals(clamped):
                if b - a + 1 >= 31:
                    intron_frags.append(seq[a - 1:b])
        simplitigs, _ = build_ust_decoy_simplitigs_chunked(intron_frags, tx_seqs, k=31)
        super_seqs = pack_n_spacer_super_decoys(simplitigs, spacer_len=spacer_len, super_chunk_bp=super_chunk_bp)
        out_ust: list[Target] = []
        for idx, s in enumerate(super_seqs):
            name = f"DECOY_UST_INTRON_{idx}"
            out_ust.append(Target(
                target_id=f"{spec.assembly_id}|{name}", assembly_id=spec.assembly_id,
                gene_id=name, transcript_id=name, group_id=f"DECOY_{spec.assembly_id}_{name}",
                symbol=name, biotype="decoy_intron", contig="UST",
                start_1b=1, end_1b=len(s), strand="+", seq=s,
            ))
        return out_ust
    spacer = "N" * spacer_len
    out: list[Target] = []
    for ctg in sorted(raw_introns):
        seq = genome.get(ctg)
        if not seq:
            continue
        ctg_len = len(seq)
        clamped = [(max(1, a), min(ctg_len, b)) for a, b in raw_introns[ctg] if a <= ctg_len]
        merged_int = _merge_intervals(clamped)
        exon_cores = [
            (a + flank, b - flank)
            for a, b in _merge_intervals(raw_exons.get(ctg, []))
            if b - a + 1 > 2 * flank
        ]
        pure_int = [
            (a, b)
            for a, b in _subtract_intervals(merged_int, exon_cores)
            if b - a + 1 >= min_len
        ]
        pieces: list[tuple[int, int]] = []
        for a, b in pure_int:
            if b - a + 1 <= super_chunk_bp:
                pieces.append((a, b))
            else:
                for cs in range(a, b + 1, super_chunk_bp):
                    ce = min(b, cs + super_chunk_bp - 1)
                    if ce - cs + 1 >= min_len:
                        pieces.append((cs, ce))
        cur_ivs: list[tuple[int, int]] = []
        cur_bp = 0
        idx = 0

        def _flush(ivs: list[tuple[int, int]], chunk_idx: int) -> None:
            if not ivs:
                return
            s0, e1 = ivs[0][0], ivs[-1][1]
            name = f"DECOY_INTRON_{ctg}_{s0}_{e1}_{chunk_idx}"
            cat_seq = spacer.join(seq[a - 1:b] for a, b in ivs)
            out.append(Target(
                target_id=f"{spec.assembly_id}|{name}", assembly_id=spec.assembly_id,
                gene_id=name, transcript_id=name, group_id=f"DECOY_{spec.assembly_id}_{name}",
                symbol=name, biotype="decoy_intron", contig=ctg,
                start_1b=s0, end_1b=e1, strand="+", seq=cat_seq, exons=list(ivs),
            ))

        for a, b in pieces:
            seg_len = b - a + 1
            add_len = seg_len + (spacer_len if cur_ivs else 0)
            if cur_ivs and cur_bp + add_len > super_chunk_bp:
                _flush(cur_ivs, idx)
                idx += 1
                cur_ivs = [(a, b)]
                cur_bp = seg_len
            else:
                cur_ivs.append((a, b))
                cur_bp += add_len
        if cur_ivs:
            _flush(cur_ivs, idx)
    return out


def build_window_targets(windows: list[ExtraWindow], fasta_by_assembly: dict[str, str | Path]) -> list[Target]:
    genomes: dict[str, dict[str, str]] = {}
    out: list[Target] = []
    for w in windows:
        if w.assembly_id not in genomes:
            genomes[w.assembly_id] = load_fasta(fasta_by_assembly[w.assembly_id])
        ctg = genomes[w.assembly_id][w.contig]
        raw = ctg[w.start_1b - 1:w.end_1b]
        seq = revcomp(raw) if w.strand == "-" else raw
        out.append(Target(
            target_id=f"{w.assembly_id}|{w.target_name}", assembly_id=w.assembly_id, gene_id=w.target_name,
            transcript_id=w.target_name, group_id=f"OG_WINDOW_{w.target_name}", symbol=w.symbol, biotype=w.biotype,
            contig=w.contig, start_1b=w.start_1b, end_1b=w.end_1b, strand=w.strand, seq=seq, product=w.product,
        ))
    return out


# ---------------------------------------------------------------------------
# Cross-assembly ortholog grouping
# ---------------------------------------------------------------------------

def _gene_representatives(targets: list[Target]) -> dict[str, Target]:
    """Longest non-decoy target per (assembly, gene)."""
    rep: dict[str, Target] = {}
    for t in targets:
        if t.is_decoy or t.biotype == "window":
            continue
        key = f"{t.assembly_id}|{t.gene_id}"
        if key not in rep or t.length > rep[key].length:
            rep[key] = t
    return rep


def _rbh_pairs(a: list[Target], b: list[Target], tmp_dir: Path, min_frac: float = 0.5) -> dict[str, str]:
    """Reciprocal best hits a.gene -> b.gene by libminimap2 asm20 alignment of representatives."""
    import mappy

    tmp_dir.mkdir(parents=True, exist_ok=True)
    fa_a = tmp_dir / "_rbh_a.fna"
    fa_b = tmp_dir / "_rbh_b.fna"
    write_fasta(fa_a, ((t.gene_id, t.seq) for t in a))
    write_fasta(fa_b, ((t.gene_id, t.seq) for t in b))
    idx_b = mappy.Aligner(str(fa_b), preset="asm20")
    idx_a = mappy.Aligner(str(fa_a), preset="asm20")

    def best(idx, recs):
        out: dict[str, tuple[str, int]] = {}
        for t in recs:
            bc, bm = "", 0
            for h in idx.map(t.seq):
                if h.strand == 1 and h.mlen > bm and h.mlen >= min_frac * min(len(t.seq), h.ctg_len):
                    bc, bm = h.ctg, h.mlen
            if bc:
                out[t.gene_id] = (bc, bm)
        return out

    a2b = best(idx_b, a)
    b2a = best(idx_a, b)
    fa_a.unlink(missing_ok=True)
    fa_b.unlink(missing_ok=True)
    return {ga: gb for ga, (gb, _) in a2b.items() if b2a.get(gb, ("", 0))[0] == ga}


def link_orthologs(
    targets: list[Target],
    primary_assembly: str,
    tmp_dir: str | Path,
    provided_map: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Assign ``group_id`` across assemblies.

    Order of evidence: (1) provided map ``"{assembly}|{gene}" -> primary gene``,
    (2) exact sequence hash identity of gene representatives, (3) reciprocal-best-hit
    alignment of the remaining representatives against the primary assembly.
    Returns per-assembly linking statistics.
    """
    tmp_dir = Path(tmp_dir)
    reps = _gene_representatives(targets)
    assemblies = sorted({t.assembly_id for t in targets})
    prim = [t for k, t in reps.items() if t.assembly_id == primary_assembly]
    prim_by_hash: dict[str, str] = {}
    for t in prim:
        prim_by_hash.setdefault(hashlib.sha1(t.seq.encode()).hexdigest(), t.gene_id)
    gene_to_group: dict[str, str] = {f"{primary_assembly}|{t.gene_id}": f"OG_{t.gene_id}" for t in prim}
    stats: dict[str, dict] = {primary_assembly: {"n_genes": len(prim), "linked_provided": 0, "linked_identical": 0, "linked_rbh": 0, "unlinked": 0}}
    for asm in assemblies:
        if asm == primary_assembly:
            continue
        others = [t for k, t in reps.items() if t.assembly_id == asm]
        st = {"n_genes": len(others), "linked_provided": 0, "linked_identical": 0, "linked_rbh": 0, "unlinked": 0}
        remaining: list[Target] = []
        for t in others:
            key = f"{asm}|{t.gene_id}"
            if provided_map and key in provided_map and f"{primary_assembly}|{provided_map[key]}" in gene_to_group:
                gene_to_group[key] = gene_to_group[f"{primary_assembly}|{provided_map[key]}"]
                st["linked_provided"] += 1
                continue
            h = hashlib.sha1(t.seq.encode()).hexdigest()
            if h in prim_by_hash:
                gene_to_group[key] = f"OG_{prim_by_hash[h]}"
                st["linked_identical"] += 1
                continue
            remaining.append(t)
        if remaining and prim:
            linked_prim = {g.split("|", 1)[1] for g in gene_to_group if g.startswith(asm + "|")}
            pairs = _rbh_pairs(remaining, prim, tmp_dir)
            for ga, gp in pairs.items():
                gene_to_group[f"{asm}|{ga}"] = f"OG_{gp}"
                st["linked_rbh"] += 1
        for t in others:
            key = f"{asm}|{t.gene_id}"
            if key not in gene_to_group:
                gene_to_group[key] = f"OG_{asm}_ONLY_{t.gene_id}"
                st["unlinked"] += 1
        stats[asm] = st
    for t in targets:
        if t.is_decoy or t.group_id:
            continue
        t.group_id = gene_to_group.get(f"{t.assembly_id}|{t.gene_id}", f"OG_{t.assembly_id}_ONLY_{t.gene_id}")
    # Harmonise symbols within a group when one member is informative
    sym_by_group: dict[str, str] = {}
    for t in targets:
        if not t.is_decoy and t.symbol != t.gene_id:
            sym_by_group.setdefault(t.group_id, t.symbol)
    for t in targets:
        if not t.is_decoy and t.symbol == t.gene_id and t.group_id in sym_by_group:
            t.symbol = sym_by_group[t.group_id]
    return stats


# ---------------------------------------------------------------------------
# Catalog container
# ---------------------------------------------------------------------------

def _parse_exons(s: str | None) -> list[tuple[int, int]]:
    if not s:
        return []
    out = []
    for iv in s.split(","):
        a, b = iv.split("-")
        out.append((int(a), int(b)))
    return out


def restore_exons_from_annotation(catalog: "Catalog", annotation: str | Path, assembly_id: str | None = None) -> int:
    """Re-attach genomic exon intervals to spliced targets of a catalog saved before exons were persisted.

    Targets are matched on ``transcript_id`` with the same key rules ``build_spliced_transcripts`` used.
    Returns the number of targets that received exons."""
    ex: dict[str, set[tuple[int, int]]] = {}
    for f in iter_features(annotation):
        if f.ftype != "exon":
            continue
        tk = _tx_key(f.attrs)
        if not tk:
            continue
        tid = tk.split(":")[-1] if tk.startswith(("rna:", "transcript:")) else tk
        ex.setdefault(tid.replace("|", "_").replace(" ", "_"), set()).add((f.start, f.end))
    n = 0
    for t in catalog.gene_targets:
        if assembly_id is not None and t.assembly_id != assembly_id:
            continue
        e = ex.get(t.transcript_id)
        if e and sum(b - a + 1 for a, b in e) == t.length:
            t.exons = sorted(e)
            n += 1
    return n


class Catalog:
    def __init__(self, targets: list[Target], primary_assembly: str, mode: str) -> None:
        self.targets: dict[str, Target] = {t.target_id: t for t in targets}
        self.primary_assembly = primary_assembly
        self.mode = mode
        self.assemblies = sorted({t.assembly_id for t in targets})
        self.link_stats: dict = {}
        self._gene_targets: list[Target] = [t for t in self.targets.values() if not t.is_decoy]
        self._decoy_ids: frozenset[str] = frozenset(t.target_id for t in self.targets.values() if t.is_decoy)
        self._groups_cache: dict[str, list[Target]] | None = None
        self._primary_gene_cache: dict[str, str] | None = None

    def invalidate_caches(self) -> None:
        self._gene_targets = [t for t in self.targets.values() if not t.is_decoy]
        self._decoy_ids = frozenset(t.target_id for t in self.targets.values() if t.is_decoy)
        self._groups_cache = None
        self._primary_gene_cache = None

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def decoy_ids(self) -> frozenset[str]:
        if len(self._gene_targets) + len(self._decoy_ids) != len(self.targets):
            self.invalidate_caches()
        return self._decoy_ids

    @property
    def gene_targets(self) -> list[Target]:
        if len(self._gene_targets) + len(self._decoy_ids) != len(self.targets):
            self.invalidate_caches()
        return self._gene_targets

    @property
    def groups(self) -> dict[str, list[Target]]:
        gt = self.gene_targets
        if self._groups_cache is not None:
            return self._groups_cache
        g: dict[str, list[Target]] = {}
        all_grouped = True
        for t in gt:
            if not t.group_id:
                all_grouped = False
            g.setdefault(t.group_id, []).append(t)
        if all_grouped:
            self._groups_cache = g
        return g

    def primary_gene_of_group(self) -> dict[str, str]:
        if self._primary_gene_cache is not None and self._groups_cache is not None:
            return self._primary_gene_cache
        out: dict[str, str] = {}
        for gid, ts in self.groups.items():
            prim = [t for t in ts if t.assembly_id == self.primary_assembly]
            out[gid] = prim[0].gene_id if prim else ts[0].gene_id
        if self._groups_cache is not None:
            self._primary_gene_cache = out
        return out

    def write_fasta(self, path: str | Path) -> Path:
        write_fasta(path, ((t.target_id, t.seq) for t in self.targets.values()))
        return Path(path)

    def write_gentrome(self, path: str | Path, genome_fastas: list[str | Path]) -> tuple[Path, Path]:
        """Salmon-style gentrome: targets followed by whole-genome decoy contigs + decoys.txt."""
        path = Path(path)
        decoys_txt = path.with_suffix(".decoys.txt")
        names: list[str] = []
        with open(path, "w") as fh:
            for t in self.targets.values():
                fh.write(f">{t.target_id}\n{t.seq}\n")
            for gf in genome_fastas:
                for ctg, seq in load_fasta(gf).items():
                    nm = f"DECOY_GENOME_{ctg}"
                    names.append(nm)
                    fh.write(f">{nm}\n{seq}\n")
        decoys_txt.write_text("\n".join(names) + "\n")
        return path, decoys_txt

    def to_table(self) -> pa.Table:
        ts = list(self.targets.values())
        return pa.table({
            "target_id": [t.target_id for t in ts],
            "assembly_id": [t.assembly_id for t in ts],
            "gene_id": [t.gene_id for t in ts],
            "transcript_id": [t.transcript_id for t in ts],
            "group_id": [t.group_id for t in ts],
            "symbol": [t.symbol for t in ts],
            "biotype": [t.biotype for t in ts],
            "contig": [t.contig for t in ts],
            "start_1b": pa.array([t.start_1b for t in ts], pa.int64()),
            "end_1b": pa.array([t.end_1b for t in ts], pa.int64()),
            "strand": [t.strand for t in ts],
            "length_bp": pa.array([t.length for t in ts], pa.int32()),
            "product": [t.product for t in ts],
            "is_decoy": [t.is_decoy for t in ts],
            # genomic exon intervals "s-e,s-e" (empty for unspliced targets); older catalogs lack the column
            "exons": [",".join(f"{a}-{b}" for a, b in t.exons) for t in ts],
        })

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fna = self.write_fasta(out_dir / "targets.fna")
        meta = out_dir / "catalog.parquet"
        pq.write_table(self.to_table(), meta, compression="zstd")
        (out_dir / "catalog_info.txt").write_text(f"primary_assembly={self.primary_assembly}\nmode={self.mode}\n")
        return {"fasta": fna, "parquet": meta}

    @classmethod
    def load(cls, out_dir: str | Path) -> "Catalog":
        out_dir = Path(out_dir)
        info = dict(line.split("=", 1) for line in (out_dir / "catalog_info.txt").read_text().splitlines() if "=" in line)
        seqs = load_fasta(out_dir / "targets.fna")
        df = pq.read_table(out_dir / "catalog.parquet").to_pandas()
        has_exons = "exons" in df.columns
        ts = [
            Target(r.target_id, r.assembly_id, r.gene_id, r.transcript_id, r.group_id, r.symbol, r.biotype, r.contig,
                   int(r.start_1b), int(r.end_1b), r.strand, seqs[r.target_id], r.product,
                   _parse_exons(r.exons) if has_exons else [])
            for r in df.itertuples(index=False)
        ]
        return cls(ts, info["primary_assembly"], info["mode"])

    def tx2gene(self) -> pd.DataFrame:
        rows = [(t.target_id, t.group_id, t.gene_id, t.assembly_id) for t in self.gene_targets]
        return pd.DataFrame(rows, columns=["target_id", "group_id", "gene_id", "assembly_id"])

    def ensure_island_pc_index(self, out_dir: str | Path, compute_dense_d1: bool = False):
        """Load or offline-build and pre-serialize ``island_pc.idx`` alongside the catalog."""
        from .island_pc_router import ReferenceIslandPCIndex

        idx_path = Path(out_dir) / "island_pc.idx"
        if idx_path.exists():
            return ReferenceIslandPCIndex.load(idx_path)
        seqs = [t.seq for t in self.gene_targets if len(t.seq) >= 31]
        if not seqs:
            seqs = [t.seq for t in self.targets.values() if len(t.seq) >= 31]
        idx = ReferenceIslandPCIndex(seqs, compute_dense_d1=compute_dense_d1)
        idx.save(idx_path)
        return idx


def build_catalog(
    assemblies: list[AssemblySpec],
    mode: str,
    primary_assembly: str | None = None,
    tmp_dir: str | Path = "/tmp",
    decoys: str = "auto",
    extra_windows: list[ExtraWindow] | None = None,
    provided_ortholog_map: dict[str, str] | None = None,
    use_ust_decoys: bool = False,
) -> Catalog:
    """Build a Catalog in ``mode`` = "prokaryotic" or "eukaryotic".

    decoys: "auto" (intergenic for prokaryotic, intron for eukaryotic), "none".
    """
    if mode not in ("prokaryotic", "eukaryotic"):
        raise ValueError(mode)
    if decoys not in ("auto", "intergenic", "intron", "merged_intron", "none"):
        raise ValueError(f"unknown decoys={decoys!r}")
    primary_assembly = primary_assembly or assemblies[0].assembly_id
    targets: list[Target] = []
    for spec in assemblies:
        if mode == "prokaryotic":
            genes = build_prokaryotic_targets(spec)
            targets.extend(genes)
            if decoys in ("auto", "intergenic") and spec.assembly_id == primary_assembly:
                targets.extend(build_intergenic_decoys(spec, genes, use_ust_decoys=use_ust_decoys))
        else:
            txs = build_spliced_transcripts(spec)
            targets.extend(txs)
            if spec.assembly_id == primary_assembly and decoys != "none":
                if use_ust_decoys or decoys == "merged_intron" or (decoys == "auto" and len(txs) >= 25_000):
                    targets.extend(build_merged_intron_decoys(spec, txs, use_ust_decoys=use_ust_decoys))
                elif decoys in ("auto", "intron"):
                    targets.extend(build_intron_decoys(spec, txs))
    if extra_windows:
        targets.extend(build_window_targets(extra_windows, {s.assembly_id: s.fasta for s in assemblies}))
    cat = Catalog(targets, primary_assembly, mode)
    cat.link_stats = link_orthologs(targets, primary_assembly, tmp_dir, provided_ortholog_map)
    cat.invalidate_caches()
    return cat


# ---------------------------------------------------------------------------
# Effective lengths & tximport offsets (Soneson et al. 2015)
# ---------------------------------------------------------------------------

def effective_length(length: int, frag_mean: float, frag_sd: float) -> float:
    """Salmon/kallisto style effective length: length minus the conditional mean fragment length."""
    if length <= 0:
        return 1.0
    if frag_sd <= 0:
        return max(1.0, length - frag_mean) if length > frag_mean else max(1.0, length / 2.0)
    # Conditional mean of truncated normal fragment length distribution on [1, length]
    xs = np.arange(1, length + 1, dtype=np.float64)
    w = np.exp(-0.5 * ((xs - frag_mean) / frag_sd) ** 2)
    if w.sum() <= 0:
        return 1.0
    cond_mean = float((xs * w).sum() / w.sum())
    return max(1.0, length - cond_mean + 1.0)


def tximport_gene_level(
    tx_counts: pd.DataFrame,
    tx_lengths: pd.DataFrame,
    tx2gene: pd.Series | dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Gene-level aggregation with per-sample effective-length offsets.

    tx_counts, tx_lengths: transcript x sample.  Returns dict with
    ``counts`` (summed), ``length`` (abundance-weighted gene effective length),
    ``norm_mat`` (length offsets: L / exp(rowMeans(log L)), the DESeq2 normalizationFactors
    numerator), ``abundance`` (gene TPM) and ``counts_length_scaled`` (tximport
    countsFromAbundance = "lengthScaledTPM", for tools without offset support).
    """
    t2g = pd.Series(tx2gene) if not isinstance(tx2gene, pd.Series) else tx2gene
    tx_idx = tx_counts.index
    cols = tx_counts.columns
    cnt_mat = tx_counts.to_numpy(dtype=np.float64, copy=False)
    len_mat = np.clip(tx_lengths.reindex(tx_idx).to_numpy(dtype=np.float64, copy=False), 1.0, None)
    rate_mat = cnt_mat / len_mat
    rate_sum = rate_mat.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpm_mat = np.where(rate_sum[None, :] > 0.0, rate_mat / np.where(rate_sum[None, :] > 0.0, rate_sum[None, :], 1.0) * 1e6, 0.0)
    genes = t2g.reindex(tx_idx).fillna("NA")
    g_codes, g_uniques = pd.factorize(genes, sort=True)
    n_g = len(g_uniques)
    n_s = len(cols)
    g_cnt_mat = np.zeros((n_g, n_s), dtype=np.float64)
    g_ab_mat = np.zeros((n_g, n_s), dtype=np.float64)
    w_len_mat = np.zeros((n_g, n_s), dtype=np.float64)
    mean_len_mat = np.zeros((n_g, n_s), dtype=np.float64)
    tx_per_g = np.maximum(1, np.bincount(g_codes, minlength=n_g)).astype(np.float64)
    w_tx_mat = tpm_mat * len_mat
    for j in range(n_s):
        g_cnt_mat[:, j] = np.bincount(g_codes, weights=cnt_mat[:, j], minlength=n_g)
        g_ab_mat[:, j] = np.bincount(g_codes, weights=tpm_mat[:, j], minlength=n_g)
        w_len_mat[:, j] = np.bincount(g_codes, weights=w_tx_mat[:, j], minlength=n_g)
        mean_len_mat[:, j] = np.bincount(g_codes, weights=len_mat[:, j], minlength=n_g) / tx_per_g
    with np.errstate(divide="ignore", invalid="ignore"):
        g_len_mat = np.where(g_ab_mat > 0.0, w_len_mat / np.where(g_ab_mat > 0.0, g_ab_mat, 1.0), mean_len_mat)
    log_l = np.log(np.clip(g_len_mat, 1.0, None))
    norm_mat_arr = np.exp(log_l - log_l.mean(axis=1, keepdims=True))
    lib = g_cnt_mat.sum(axis=0)
    mean_gene_len = g_len_mat.mean(axis=1)
    scaled_raw = g_ab_mat * mean_gene_len[:, None]
    scaled_sum = scaled_raw.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_arr = np.where(scaled_sum[None, :] > 0.0, scaled_raw / np.where(scaled_sum[None, :] > 0.0, scaled_sum[None, :], 1.0) * lib[None, :], 0.0)
    g_idx = pd.Index(g_uniques, name=genes.name)
    return {
        "counts": pd.DataFrame(g_cnt_mat, index=g_idx, columns=cols),
        "length": pd.DataFrame(g_len_mat, index=g_idx, columns=cols),
        "norm_mat": pd.DataFrame(norm_mat_arr, index=g_idx, columns=cols),
        "abundance": pd.DataFrame(g_ab_mat, index=g_idx, columns=cols),
        "counts_length_scaled": pd.DataFrame(scaled_arr, index=g_idx, columns=cols),
    }
