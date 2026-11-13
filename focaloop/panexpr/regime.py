"""Pre-tournament regime classifier (0 LLM calls, 0 taxonomy lookups).

Before any index is built or the 28,000-read tournament starts, inspect only:

  * the reference FASTA(s) (``.fai`` when present, else a byte scan or block sample),
  * the GTF/GFF3 annotation(s) (exact parse when small, evenly spaced byte chunks when large),
  * the first 1,000 reads of one run,
  * the RAM available to this process (cgroup / Slurm / ``MemAvailable``),

and decide (a) how the target catalog is built and (b) which kernels are eligible for the
tournament.  Five regimes are measured:

  1. intron / splicing        multi_exon_gene_frac, introns_per_gene, intron_bp_per_exon_bp, genome_bp
  2. isoform multiplicity     tx_per_gene, n_transcripts, cdna_bp
  3. redundancy               n_assemblies, within-assembly duplicate k-mer fraction, cross-assembly shared k-mer fraction
  4. RAM ceiling              predicted index-build MaxRSS per kernel vs the available budget
  5. read profile             read length, layout, empirical 3' Illumina adapter k-mer rate

Rules (all thresholds are module constants so they can be audited and tested):

  * ``multi_exon_gene_frac < 0.01``  -> unspliced gene-body catalog + intergenic decoys, spliced aligners excluded.
    Otherwise spliced cDNA catalog + (merged) intron decoys, spliced aligners kept.  A spliced genome with
    ~1 transcript per gene (fungal / non-model annotation) keeps spliced aligners *and* records
    ``unannotated_junction_risk`` because junctions missing from the annotation can only be recovered by them.
  * ``tx_per_gene >= 1.5`` or ``n_transcripts >= 100,000`` -> high-isoform regime: unitig / cDBG indexers stay
    eligible, merged intron decoys are required and tximport effective-length offsets apply.
  * ``n_assemblies >= 2`` or ``tx_per_gene > 1.05`` or duplicate k-mer fraction ``>= 0.01`` -> strict
    mate-tie symmetry gate (>= 0.80); kernels whose paired mode collapses ties by construction are dropped
    before indexing.
  * predicted index-build RAM > budget -> kernel dropped before indexing.
  * adapter rate > 2 % -> kernels that align end-to-end without soft clipping are dropped.
  * median read length > 400 -> long-read regime: short-read-only kernels dropped, mappy switches preset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
import os
from pathlib import Path
import re
import statistics
import time
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
SPLICED_MIN_MULTI_EXON_FRAC = 0.01
HIGH_ISOFORM_TX_PER_GENE = 1.5
HIGH_ISOFORM_N_TRANSCRIPTS = 100_000
FAST_EUK_MIN_TRANSCRIPTS = 25_000
FAST_EUK_MIN_GENOME_BP = 100_000_000
FAST_EUK_KERNELS = frozenset({"piscem", "salmon"})
STRICT_TIE_TX_PER_GENE = 1.05
STRICT_TIE_DUP_KMER_FRAC = 0.01
ADAPTER_RATE_MAX = 0.02
LONG_READ_MEDIAN = 400
RAM_SAFETY = 0.85                    # use at most this share of the budget for one index build
ANNOT_EXACT_MAX_BYTES = 24 << 20     # parse the whole annotation below this size
ANNOT_SAMPLE_CHUNKS = 48
ANNOT_CHUNK_BYTES = 192 << 10
GENOME_FULL_SCAN_MAX_BP = 80_000_000
GENOME_BLOCKS = 192
GENOME_BLOCK_BYTES = 32 << 10
KMER = 21
ANCHORS = ("ACGTA", "TGCAG", "GATCC", "CATGA", "AGTCT", "TCAGG")   # content-defined k-mer sample (~6/1024 of k-mers)

ADAPTERS = {"truseq": "AGATCGGAAGAG", "nextera": "CTGTCTCTTATA"}

# ---------------------------------------------------------------------------
# Kernel traits.  ``ram_b_per_bp`` / ``ram_base_mb`` model index-BUILD MaxRSS against the bp of the
# sequence that kernel indexes (catalog FASTA, or the genome for genome-indexing kernels), fitted on the
# measured builds in beyond_workflows/panexpr_benchmarks/mapper_survey (scoreboard_all_studies.json,
# ``index_ram_calibration``); see 14_open_source_mapper_survey.md.  ``end_to_end`` marks kernels whose mean measured
# adapter_overhang recall over Studies 1-7 is below 30 % (bowtie2 global, salmon, both HISAT2 and both STAR kernels).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelTraits:
    family: str                   # seed_extend | spliced | kmer_ec | graph
    spliced: bool = False         # spliced aligner (needs introns to be useful)
    needs_genome: bool = False    # indexes the genome, not the catalog (STAR TranscriptomeSAM)
    end_to_end: bool = False      # no soft clipping: adapter read-through costs the read
    pair_collapse: bool = False   # paired mode collapses equal-score ties by construction
    long_read_ok: bool = False
    ram_b_per_bp: float = 10.0
    ram_base_mb: float = 200.0


KERNEL_TRAITS: dict[str, KernelTraits] = {
    # ram_b_per_bp / ram_base_mb: two-point fit of measured index-build MaxRSS (Study 3 Af293+A1163 catalog, 34.7 Mb, and the
    # GRCh38 catalog, 2,568.5 Mb; genome-indexing kernels use the 29.4 Mb Af293 and 3.1 Gb GRCh38 genomes), slope rounded up.
    "mappy": KernelTraits("seed_extend", long_read_ok=True, ram_b_per_bp=7.5, ram_base_mb=100),
    "bwa_mem2_spa": KernelTraits("seed_extend", ram_b_per_bp=24.0, ram_base_mb=100),
    "bwa_mem2_naive_pa": KernelTraits("seed_extend", pair_collapse=True, ram_b_per_bp=24.0, ram_base_mb=100),
    "bwa_spa": KernelTraits("seed_extend", ram_b_per_bp=1.6, ram_base_mb=100),
    "bwa_meme_spa": KernelTraits("seed_extend", ram_b_per_bp=35.0, ram_base_mb=2200),
    "bowtie2_local_k16": KernelTraits("seed_extend", ram_b_per_bp=2.4, ram_base_mb=200),
    "bowtie2_global_k16": KernelTraits("seed_extend", end_to_end=True, ram_b_per_bp=2.4, ram_base_mb=200),
    "strobealign_N16": KernelTraits("seed_extend", ram_b_per_bp=4.0, ram_base_mb=100),
    "chromap": KernelTraits("seed_extend", ram_b_per_bp=10.0, ram_base_mb=100),
    "snap": KernelTraits("seed_extend", ram_b_per_bp=18.0, ram_base_mb=300),
    "dragmap": KernelTraits("seed_extend", ram_b_per_bp=22.0, ram_base_mb=3500),
    "hisat2_k16": KernelTraits("spliced", spliced=True, end_to_end=True, ram_b_per_bp=3.6, ram_base_mb=100),
    "subread": KernelTraits("seed_extend", ram_b_per_bp=1.8, ram_base_mb=2000),
    "subjunc": KernelTraits("spliced", spliced=True, ram_b_per_bp=1.8, ram_base_mb=2000),
    "gsnap_k16": KernelTraits("spliced", spliced=True, ram_b_per_bp=1.8, ram_base_mb=400),
    "segemehl": KernelTraits("spliced", spliced=True, ram_b_per_bp=24.0, ram_base_mb=300),
    "star": KernelTraits("spliced", spliced=True, needs_genome=True, end_to_end=True, ram_b_per_bp=13.0, ram_base_mb=400),
    "hisat2_genome_k16": KernelTraits("spliced", spliced=True, needs_genome=True, end_to_end=True, ram_b_per_bp=2.1, ram_base_mb=250),
    "star_genome_m16": KernelTraits("spliced", spliced=True, needs_genome=True, end_to_end=True, ram_b_per_bp=13.0, ram_base_mb=400),
    "salmon": KernelTraits("kmer_ec", end_to_end=True, ram_b_per_bp=4.6, ram_base_mb=900),
    "kallisto": KernelTraits("kmer_ec", ram_b_per_bp=24.5, ram_base_mb=100),
    "piscem": KernelTraits("kmer_ec", ram_b_per_bp=4.6, ram_base_mb=900),
    "themisto": KernelTraits("kmer_ec", ram_b_per_bp=6.0, ram_base_mb=1000),
    "vg_giraffe": KernelTraits("graph", ram_b_per_bp=70.0, ram_base_mb=300),
}


def traits_of(kernel: str) -> KernelTraits:
    return KERNEL_TRAITS.get(kernel, KernelTraits("unknown"))


# ---------------------------------------------------------------------------
# Feature containers
# ---------------------------------------------------------------------------


@dataclass
class AnnotationFeatures:
    path: str
    file_bytes: int
    sampled: bool
    bytes_parsed: int
    n_genes: int
    n_transcripts: int
    multi_exon_gene_frac: float
    introns_per_gene: float
    intron_bp_per_exon_bp: float
    tx_per_gene: float
    exons_per_tx: float
    cdna_bp: int
    intron_bp: int
    genic_span_bp: int


@dataclass
class GenomeFeatures:
    path: str
    genome_bp: int
    n_contigs: int
    method: str           # fai | scan | estimate


@dataclass
class ReadProfile:
    n_reads: int
    layout: str
    read_len_median: float
    read_len_max: int
    adapter_rate: float
    adapter_hits: dict[str, int]
    tail_partial_rate: float
    read_class: str


@dataclass
class RedundancyFeatures:
    n_assemblies: int
    method: str
    sampled_bp: int
    dup_kmer_frac_primary: float
    cross_assembly_shared_frac: dict[str, float]


@dataclass
class RegimeDecision:
    catalog_mode: str                 # prokaryotic (gene body) | eukaryotic (spliced cDNA)
    decoys: str                       # intergenic | intron | merged_intron
    spliced_regime: bool
    unannotated_junction_risk: bool
    high_isoform: bool
    tximport_offsets: bool
    strict_tie_gate: bool
    read_class: str
    mappy_preset: str
    ram_budget_bytes: int
    predicted_catalog_bp: int
    predicted_index_ram_bytes: dict[str, int]
    candidates: list[str]
    excluded: dict[str, str]
    reasons: list[str]
    wall_seconds: float
    features: dict = field(default_factory=dict)
    router_verdict: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

_GTF_TX = re.compile(r'transcript_id "([^"]*)"')
_GTF_GENE = re.compile(r'gene_id "([^"]*)"')
_GFF_ID = re.compile(r"(?:^|;)ID=([^;]+)")
_GFF_PARENT = re.compile(r"(?:^|;)Parent=([^;]+)")


def _open_bytes(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def _annotation_chunks(path: Path) -> tuple[list[bytes], bool, int]:
    """Whole file when small (or gzipped: a prefix), else evenly spaced chunks cut at line boundaries."""
    size = path.stat().st_size
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            data = fh.read(ANNOT_EXACT_MAX_BYTES)
            more = bool(fh.read(1))
        if more:
            data = data[: data.rfind(b"\n") + 1]
        return [data], more, size
    if size <= ANNOT_EXACT_MAX_BYTES:
        return [path.read_bytes()], False, size
    chunks = []
    with open(path, "rb") as fh:
        for i in range(ANNOT_SAMPLE_CHUNKS):
            off = int(i * (size - ANNOT_CHUNK_BYTES) / max(1, ANNOT_SAMPLE_CHUNKS - 1))
            fh.seek(off)
            buf = fh.read(ANNOT_CHUNK_BYTES)
            a = buf.find(b"\n") + 1 if off else 0
            b = buf.rfind(b"\n") + 1
            chunks.append(buf[a:b])
    return chunks, True, size


def _parse_chunk(text: str, is_gff3: bool, edge_trim: bool):
    """Return {tx: (gene, [(s, e), ...])} for one chunk.  Exons win over CDS when a transcript has both."""
    exons: dict[str, list[tuple[int, int]]] = {}
    cds: dict[str, list[tuple[int, int]]] = {}
    tx_gene: dict[str, str] = {}
    order: list[str] = []
    for line in text.split("\n"):
        if not line or line[0] == "#":
            continue
        p = line.split("\t")
        if len(p) < 9:
            continue
        ftype = p[2]
        if is_gff3:
            if ftype in ("mRNA", "transcript", "ncRNA", "lnc_RNA", "rRNA", "tRNA", "snRNA", "snoRNA", "primary_transcript"):
                m = _GFF_ID.search(p[8]); g = _GFF_PARENT.search(p[8])
                if m:
                    tx_gene[m.group(1)] = g.group(1).split(",")[0] if g else m.group(1)
                continue
            if ftype not in ("exon", "CDS"):
                continue
            m = _GFF_PARENT.search(p[8])
            if not m:
                continue
            tx = m.group(1).split(",")[0]
        else:
            if ftype not in ("exon", "CDS"):
                continue
            m = _GTF_TX.search(p[8]); g = _GTF_GENE.search(p[8])
            gene = g.group(1) if g else ""
            tx = m.group(1) if (m and m.group(1)) else gene
            if not tx:
                continue
            tx_gene.setdefault(tx, gene or tx)
        try:
            iv = (int(p[3]), int(p[4]))
        except ValueError:
            continue
        key = f"{p[0]}|{tx}"
        if key not in exons and key not in cds:
            order.append(key)
        (exons if ftype == "exon" else cds).setdefault(key, []).append(iv)
    if edge_trim and len(order) > 2:      # first / last transcript of a sampled chunk may be cut
        order = order[1:-1]
    out = {}
    for key in order:
        ivs = exons.get(key) or cds.get(key)
        if not ivs:
            continue
        tx = key.split("|", 1)[1]
        out[key] = (tx_gene.get(tx, tx), sorted(set(ivs)))
    return out


def annotation_features(path: str | Path) -> AnnotationFeatures:
    path = Path(path)
    is_gff3 = any(s in path.name.lower() for s in (".gff3", ".gff"))
    chunks, sampled, size = _annotation_chunks(path)
    parsed_bytes = sum(len(c) for c in chunks)
    tx: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    for c in chunks:
        tx.update(_parse_chunk(c.decode("ascii", "replace"), is_gff3, sampled))
    genes: dict[str, list[list[tuple[int, int]]]] = {}
    contig_of: dict[str, str] = {}
    for key, (g, ivs) in tx.items():
        gk = f"{key.split('|', 1)[0]}|{g}"
        genes.setdefault(gk, []).append(ivs)
    n_tx = len(tx)
    n_genes = len(genes)
    multi = 0
    introns_total = 0
    exon_bp_tx = 0
    intron_bp_tx = 0
    exons_total = 0
    intron_union_bp = 0
    span_bp = 0
    for gk, txs in genes.items():
        has_multi = False
        intr: set[tuple[int, int]] = set()
        lo, hi = None, None
        for ivs in txs:
            exons_total += len(ivs)
            exon_bp_tx += sum(e - s + 1 for s, e in ivs)
            lo = ivs[0][0] if lo is None else min(lo, ivs[0][0])
            hi = max(e for _, e in ivs) if hi is None else max(hi, max(e for _, e in ivs))
            if len(ivs) >= 2:
                has_multi = True
                for (s1, e1), (s2, _e2) in zip(ivs, ivs[1:]):
                    if s2 > e1 + 1:
                        intr.add((e1 + 1, s2 - 1))
                        intron_bp_tx += s2 - 1 - e1
        multi += has_multi
        introns_total += len(intr)
        span_bp += (hi - lo + 1) if lo is not None else 0
        if intr:
            iv = sorted(intr)
            cs, ce = iv[0]
            for s, e in iv[1:]:
                if s > ce:
                    intron_union_bp += ce - cs + 1
                    cs, ce = s, e
                else:
                    ce = max(ce, e)
            intron_union_bp += ce - cs + 1
    scale = (size / parsed_bytes) if sampled and parsed_bytes else 1.0
    if sampled and str(path).endswith(".gz"):
        # gzip prefix: approximate the uncompressed size from the compressed share consumed
        scale = 1.0
        try:
            with gzip.open(path, "rb") as fh:
                fh.read(parsed_bytes)
                consumed = fh.fileobj.tell()  # type: ignore[attr-defined]
            scale = size / max(1, consumed)
        except Exception:
            pass
    return AnnotationFeatures(
        path=str(path), file_bytes=size, sampled=sampled, bytes_parsed=parsed_bytes,
        n_genes=int(round(n_genes * scale)), n_transcripts=int(round(n_tx * scale)),
        multi_exon_gene_frac=multi / n_genes if n_genes else 0.0,
        introns_per_gene=introns_total / n_genes if n_genes else 0.0,
        intron_bp_per_exon_bp=intron_bp_tx / exon_bp_tx if exon_bp_tx else 0.0,
        tx_per_gene=n_tx / n_genes if n_genes else 0.0,
        exons_per_tx=exons_total / n_tx if n_tx else 0.0,
        cdna_bp=int(round(exon_bp_tx * scale)),
        intron_bp=int(round(intron_union_bp * scale)),
        genic_span_bp=int(round(span_bp * scale)),
    )


# ---------------------------------------------------------------------------
# Genome size + redundancy sketch
# ---------------------------------------------------------------------------


def genome_features(path: str | Path) -> GenomeFeatures:
    path = Path(path)
    fai = Path(str(path) + ".fai")
    if fai.exists():
        n, bp = 0, 0
        for line in fai.read_text().splitlines():
            p = line.split("\t")
            if len(p) >= 2:
                n += 1
                bp += int(p[1])
        return GenomeFeatures(str(path), bp, n, "fai")
    size = path.stat().st_size
    if not str(path).endswith(".gz") and size <= GENOME_FULL_SCAN_MAX_BP * 1.1:
        data = path.read_bytes()
        n = data.count(b">")
        hdr = sum(len(m) for m in re.findall(rb">[^\n]*\n", data))
        return GenomeFeatures(str(path), len(data) - hdr - data.count(b"\n") + n, n, "scan")
    with _open_bytes(path) as fh:
        head = fh.read(1 << 20)
    nl = head.count(b"\n") / max(1, len(head))
    est = int(size * (1 - nl)) if not str(path).endswith(".gz") else int(size * 3.4 * (1 - nl))
    return GenomeFeatures(str(path), est, head.count(b">"), "estimate")


_RC = bytes.maketrans(b"ACGT", b"TGCA")


def _read_sequence_sample(path: Path, full: bool) -> bytes:
    """Upper-case sequence bytes ('N' joins contigs).  Full read or GENOME_BLOCKS evenly spaced blocks."""
    if full:
        with _open_bytes(path) as fh:
            data = fh.read()
        parts = re.split(rb">[^\n]*\n", data)
        return b"N".join(p.replace(b"\n", b"").replace(b"\r", b"") for p in parts).upper()
    size = path.stat().st_size
    if size <= GENOME_BLOCK_BYTES:
        return _read_sequence_sample(path, full=True)
    out = []
    with open(path, "rb") as fh:
        for i in range(GENOME_BLOCKS):
            fh.seek(int(i * (size - GENOME_BLOCK_BYTES) / max(1, GENOME_BLOCKS - 1)))
            buf = fh.read(GENOME_BLOCK_BYTES)
            buf = re.sub(rb">[^\n]*\n", b"N", buf)
            out.append(buf.replace(b"\n", b"").replace(b"\r", b"").upper())
    return b"N".join(out)


def _anchor_kmers(seq: bytes, k: int = KMER) -> list[bytes]:
    """All k-mers whose canonical orientation starts with one of ANCHORS (strand symmetric, position free)."""
    out: list[bytes] = []
    for a in ANCHORS:
        ab = a.encode()
        arc = ab.translate(_RC)[::-1]
        i = seq.find(ab)
        while i != -1:
            km = seq[i:i + k]
            if len(km) == k and b"N" not in km:
                out.append(km)
            i = seq.find(ab, i + 1)
        if arc != ab:
            j = seq.find(arc)
            while j != -1:
                st = j + len(arc) - k
                if st >= 0:
                    km = seq[st:j + len(arc)]
                    if b"N" not in km:
                        out.append(km.translate(_RC)[::-1])
                j = seq.find(arc, j + 1)
    return out


def redundancy_features(genomes: Sequence[str | Path], genome_bp: Sequence[int]) -> RedundancyFeatures:
    full = sum(genome_bp) <= GENOME_FULL_SCAN_MAX_BP
    sketches = []
    sampled_bp = 0
    for g in genomes:
        seq = _read_sequence_sample(Path(g), full or str(g).endswith(".gz"))
        sampled_bp += len(seq)
        sketches.append(_anchor_kmers(seq))
    counts: dict[bytes, int] = {}
    for km in sketches[0]:
        counts[km] = counts.get(km, 0) + 1
    dup = sum(1 for c in counts.values() if c >= 2) / max(1, len(counts))
    prim = set(counts)
    cross = {}
    for g, sk in zip(genomes[1:], sketches[1:]):
        s = set(sk)
        cross[Path(g).name] = round(len(s & prim) / max(1, len(s)), 4) if full else float("nan")
    return RedundancyFeatures(len(genomes), "full_scan" if full else "block_sample", sampled_bp, round(dup, 5), cross)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _first_reads(path: str | Path, n: int) -> list[str]:
    seqs = []
    with _open_bytes(Path(path)) as fh:
        while len(seqs) < n:
            h = fh.readline()
            if not h:
                break
            s = fh.readline().strip()
            fh.readline(); fh.readline()
            seqs.append(s.decode("ascii", "replace").upper())
    return seqs


def read_profile(r1: str | Path | list[str], r2: str | Path | list[str] | None = None, n: int = 1000) -> ReadProfile:
    s1 = r1 if isinstance(r1, list) else _first_reads(r1, n)
    s2 = (r2 if isinstance(r2, list) else _first_reads(r2, n)) if r2 is not None else None
    s1 = s1[:n]
    s2 = s2[:n] if s2 is not None else None
    lens = [len(s) for s in s1] + ([len(s) for s in s2] if s2 else [])
    hits = {k: 0 for k in ADAPTERS}
    any_hit = 0
    tail = 0
    for i, s in enumerate(s1):
        mates = [s] + ([s2[i]] if s2 and i < len(s2) else [])
        hit = False
        for m in mates:
            for name, ad in ADAPTERS.items():
                if ad in m:
                    hits[name] += 1
                    hit = True
        any_hit += hit
        if not hit:
            for m in mates:
                if any(m.endswith(ad[:L]) for ad in ADAPTERS.values() for L in range(8, len(ad))):
                    tail += 1
                    break
    med = statistics.median(lens) if lens else 0.0
    return ReadProfile(len(s1), "PE" if s2 else "SE", float(med), max(lens) if lens else 0,
                       round(any_hit / max(1, len(s1)), 4), hits, round(tail / max(1, len(s1)), 4),
                       "long" if med > LONG_READ_MEDIAN else "short")


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------


def available_ram_bytes() -> int:
    """min(cgroup memory.max, SLURM_MEM_PER_NODE, MemAvailable)."""
    vals = []
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit() and int(v) < (1 << 60):
                vals.append(int(v))
        except OSError:
            pass
    try:
        cg = Path("/proc/self/cgroup").read_text().strip().splitlines()
        for line in cg:
            rel = line.split(":", 2)[-1]
            p = Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.max"
            if p.exists():
                v = p.read_text().strip()
                if v.isdigit():
                    vals.append(int(v))
    except OSError:
        pass
    sm = os.environ.get("SLURM_MEM_PER_NODE")
    if sm and sm.isdigit():
        vals.append(int(sm) << 20)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                vals.append(int(line.split()[1]) << 10)
    except OSError:
        pass
    return min(vals) if vals else 8 << 30


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass
class AssemblyInput:
    assembly_id: str
    fasta: str | Path
    annotation: str | Path


def predict_catalog_bp(annots: list[AnnotationFeatures], genomes: list[GenomeFeatures], spliced: bool) -> int:
    """cDNA (or gene bodies) of every assembly + decoys of the primary (merged introns or intergenic)."""
    if spliced:
        return int(sum(a.cdna_bp for a in annots) + (annots[0].intron_bp if annots else 0))
    genic = sum(min(a.genic_span_bp, g.genome_bp) for a, g in zip(annots, genomes))
    inter = max(0, genomes[0].genome_bp - min(annots[0].genic_span_bp, genomes[0].genome_bp)) if genomes else 0
    return int(genic + inter)


def _first_reads_with_quals(path: str | Path, n: int) -> tuple[list[str], list[str] | None]:
    seqs: list[str] = []
    quals: list[str] = []
    with _open_bytes(Path(path)) as fh:
        while len(seqs) < n:
            h = fh.readline()
            if not h:
                break
            s = fh.readline().strip()
            fh.readline()
            q = fh.readline().strip()
            seqs.append(s.decode("ascii", "replace").upper())
            if len(q) == len(s) and q:
                quals.append(q.decode("ascii", "replace"))
    return seqs, (quals if len(quals) == len(seqs) and seqs else None)


def _read_all_fasta_contigs(path: Path, max_total_bp: int | None = None) -> list[str]:
    """Read sequences from a plain or gzipped FASTA without loading arbitrary 3 Gb files unless allowed."""
    contigs: list[str] = []
    cur: list[str] = []
    total = 0
    with _open_bytes(path) as fh:
        for raw in fh:
            if raw.startswith(b">"):
                if cur:
                    seq = "".join(cur).upper()
                    total += len(seq)
                    if max_total_bp is not None and total > max_total_bp:
                        return []
                    contigs.append(seq)
                    cur.clear()
            else:
                s = raw.strip()
                if s:
                    cur.append(s.decode("ascii", "replace"))
        if cur:
            seq = "".join(cur).upper()
            total += len(seq)
            if max_total_bp is not None and total > max_total_bp:
                return []
            contigs.append(seq)
    return contigs


def _resolve_probe_reads(
    r1: str | Path | list[str],
    r2: str | Path | list[str] | None,
    n_probe: int = 2000,
) -> tuple[list[str], list[str] | None]:
    if isinstance(r1, list):
        seqs = [s.upper() for s in r1[:n_probe]]
        if isinstance(r2, list) and len(seqs) < n_probe:
            seqs.extend(s.upper() for s in r2[: n_probe - len(seqs)])
        return seqs, None
    if r2 is not None and not isinstance(r2, list):
        n_half = max(1, n_probe // 2)
        s1, q1 = _first_reads_with_quals(r1, n_half)
        s2, q2 = _first_reads_with_quals(r2, n_probe - len(s1))
        seqs = s1 + s2
        quals = (q1 + q2) if (q1 is not None and q2 is not None) else None
        return seqs, quals
    return _first_reads_with_quals(r1, n_probe)


def _find_transcript_sibling(fasta_path: Path, ipc_index_dir: Path | None) -> Path | None:
    candidates = [
        fasta_path.parent / "transcripts.fa",
        fasta_path.parent / "transcripts.fa.gz",
        fasta_path.parent / "cdna.fa",
        fasta_path.parent / "cdna.fa.gz",
    ]
    if ipc_index_dir is not None:
        candidates.extend([
            ipc_index_dir / "transcripts.fa",
            ipc_index_dir.parent / "transcripts.fa",
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _run_island_pc_router(
    assemblies: Sequence[AssemblyInput],
    genomes: list[GenomeFeatures],
    r1: str | Path | list[str],
    r2: str | Path | list[str] | None,
    *,
    spliced: bool,
    strict: bool,
    n_probe: int = 2000,
    ipc_index: object = None,
    ipc_index_dir: str | Path | None = None,
) -> dict:
    from .island_pc_router import (
        ReferenceIslandPCIndex,
        pack_read_batch,
        route_pangenome_wave0,
    )

    t0 = time.perf_counter()
    seqs, quals = _resolve_probe_reads(r1, r2, n_probe=n_probe)
    if not seqs:
        return {"status": "skipped_no_reads", "pseudoaligners_disqualified": False}
    batch = pack_read_batch(seqs, quals=quals)
    idx_dir = Path(ipc_index_dir) if ipc_index_dir is not None else None
    if idx_dir is not None:
        idx_dir.mkdir(parents=True, exist_ok=True)

    loaded_indices: dict[str, ReferenceIslandPCIndex] = {}
    owned_indices: list[ReferenceIslandPCIndex] = []
    index_source = "none"
    max_auto_build_bp = 120_000_000

    try:
        if isinstance(ipc_index, ReferenceIslandPCIndex):
            loaded_indices[assemblies[0].assembly_id] = ipc_index
            index_source = "in_memory"
        elif isinstance(ipc_index, (str, Path)):
            idx = ReferenceIslandPCIndex.load(ipc_index)
            loaded_indices[assemblies[0].assembly_id] = idx
            owned_indices.append(idx)
            index_source = "pre_serialized"
        elif isinstance(ipc_index, dict):
            for asm_id, val in ipc_index.items():
                if isinstance(val, ReferenceIslandPCIndex):
                    loaded_indices[asm_id] = val
                else:
                    idx = ReferenceIslandPCIndex.load(val)
                    loaded_indices[asm_id] = idx
                    owned_indices.append(idx)
            index_source = "pre_serialized_map"
        else:
            # Check pre-serialized cache first (<5 ms load), otherwise build when compact.
            single_idx = (idx_dir / "island_pc.idx") if (idx_dir is not None and len(assemblies) == 1) else None
            all_cached = idx_dir is not None and all(
                (idx_dir / f"{a.assembly_id}.ipc.idx").exists() for a in assemblies
            )
            if all_cached and idx_dir is not None:
                for a in assemblies:
                    idx = ReferenceIslandPCIndex.load(idx_dir / f"{a.assembly_id}.ipc.idx")
                    loaded_indices[a.assembly_id] = idx
                    owned_indices.append(idx)
                index_source = "pre_serialized_cache"
            elif single_idx is not None and single_idx.exists():
                idx = ReferenceIslandPCIndex.load(single_idx)
                loaded_indices[assemblies[0].assembly_id] = idx
                owned_indices.append(idx)
                index_source = "pre_serialized_cache"
            elif all(g.genome_bp <= max_auto_build_bp for g in genomes):
                for a in assemblies:
                    cache_p = (idx_dir / f"{a.assembly_id}.ipc.idx") if idx_dir is not None else None
                    if cache_p is not None and cache_p.exists():
                        idx = ReferenceIslandPCIndex.load(cache_p)
                    else:
                        contigs = _read_all_fasta_contigs(Path(a.fasta), max_total_bp=max_auto_build_bp)
                        if not contigs:
                            continue
                        idx = ReferenceIslandPCIndex(contigs, compute_dense_d1=False)
                        if cache_p is not None:
                            idx.save(cache_p)
                    loaded_indices[a.assembly_id] = idx
                    owned_indices.append(idx)
                index_source = "built_compact_genome"
            elif spliced:
                prim_a = assemblies[0]
                cache_p = (idx_dir / f"{prim_a.assembly_id}.ipc.idx") if idx_dir is not None else None
                if cache_p is not None and cache_p.exists():
                    idx = ReferenceIslandPCIndex.load(cache_p)
                    loaded_indices[prim_a.assembly_id] = idx
                    owned_indices.append(idx)
                    index_source = "pre_serialized_cache"
                else:
                    tx_fa = _find_transcript_sibling(Path(prim_a.fasta), idx_dir)
                    if tx_fa is not None:
                        contigs = _read_all_fasta_contigs(tx_fa, max_total_bp=700_000_000)
                        if contigs:
                            idx = ReferenceIslandPCIndex(contigs, compute_dense_d1=False)
                            if cache_p is not None:
                                idx.save(cache_p)
                            loaded_indices[prim_a.assembly_id] = idx
                            owned_indices.append(idx)
                            index_source = "built_transcript_sibling"

        if not loaded_indices:
            return {
                "status": "skipped_large_genome_no_preserialized_index",
                "pseudoaligners_disqualified": False,
                "wall_ms": round((time.perf_counter() - t0) * 1e3, 3),
            }

        pangenome_verdict = None
        if len(loaded_indices) > 1:
            pg = route_pangenome_wave0(loaded_indices, batch, spliced=spliced)
            summary = pg["leader_summary"]
            pangenome_verdict = {
                "leader": pg["leader"],
                "retained_assemblies": list(pg["retained_assemblies"]),
                "eliminated_assemblies": list(pg["eliminated_assemblies"]),
                "strictly_separated": bool(pg["strictly_separated"]),
                "multi_assembly_zoom_triggered": bool(pg["multi_assembly_zoom_triggered"]),
                "assemblies": {
                    aid: {
                        "r_hat": float(st["r_hat"]),
                        "ci_low": float(st["ci_low"]),
                        "ci_high": float(st["ci_high"]),
                        "ci_overlap": [float(st["ci_overlap"][0]), float(st["ci_overlap"][1])],
                        "ci_binom": [float(st["ci_binom"][0]), float(st["ci_binom"][1])],
                    }
                    for aid, st in pg["assemblies"].items()
                },
            }
        else:
            only_idx = next(iter(loaded_indices.values()))
            summary = only_idx.classify_reads(batch, spliced=spliced, strict_tie_gate=strict)

        verdict_dict = summary.as_dict()
        verdict_dict["status"] = "ok"
        verdict_dict["index_source"] = index_source
        verdict_dict["total_router_wall_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        verdict_dict["pseudoaligners_disqualified"] = bool(summary.routing.pseudoaligners_disqualified)
        if pangenome_verdict is not None:
            verdict_dict["pangenome_wave0"] = pangenome_verdict
        return verdict_dict
    finally:
        for idx in owned_indices:
            idx.close()


def classify(
    assemblies: Sequence[AssemblyInput],
    r1: str | Path | list[str],
    r2: str | Path | list[str] | None = None,
    kernels: Iterable[str] | None = None,
    ram_budget_bytes: int | None = None,
    n_reads: int = 1000,
    use_island_pc_router: bool = False,
    ipc_index: object = None,
    ipc_index_dir: str | Path | None = None,
) -> RegimeDecision:
    """Measure the five regimes and return catalog settings + the pre-tournament candidate list."""
    t0 = time.time()
    if not assemblies:
        raise ValueError("at least one assembly is required")
    kernels = list(kernels) if kernels is not None else list(KERNEL_TRAITS)
    annots = [annotation_features(a.annotation) for a in assemblies]
    genomes = [genome_features(a.fasta) for a in assemblies]
    red = redundancy_features([a.fasta for a in assemblies], [g.genome_bp for g in genomes])
    reads = read_profile(r1, r2, n_reads)
    budget = int(ram_budget_bytes if ram_budget_bytes is not None else available_ram_bytes())
    prim = annots[0]
    reasons: list[str] = []

    spliced = prim.multi_exon_gene_frac >= SPLICED_MIN_MULTI_EXON_FRAC
    high_iso = prim.tx_per_gene >= HIGH_ISOFORM_TX_PER_GENE or prim.n_transcripts >= HIGH_ISOFORM_N_TRANSCRIPTS
    junction_risk = spliced and prim.tx_per_gene < 1.2
    catalog_mode = "eukaryotic" if spliced else "prokaryotic"
    decoys = ("merged_intron" if high_iso else "intron") if spliced else "intergenic"
    reasons.append(f"multi_exon_gene_frac={prim.multi_exon_gene_frac:.4f} -> {'spliced cDNA + ' + decoys if spliced else 'gene-body + intergenic'} decoys")
    if junction_risk:
        reasons.append(f"tx_per_gene={prim.tx_per_gene:.3f} with introns: unannotated junctions possible, spliced aligners kept")
    if high_iso:
        reasons.append(f"tx_per_gene={prim.tx_per_gene:.3f}, n_transcripts~{prim.n_transcripts}: high-isoform, cDBG indexers + tximport offsets")
    strict = red.n_assemblies >= 2 or prim.tx_per_gene > STRICT_TIE_TX_PER_GENE or red.dup_kmer_frac_primary >= STRICT_TIE_DUP_KMER_FRAC
    reasons.append(f"n_assemblies={red.n_assemblies}, dup_kmer_frac={red.dup_kmer_frac_primary:.4f}, tx_per_gene={prim.tx_per_gene:.3f} -> "
                   f"strict mate-tie gate {'ON' if strict else 'off'}")
    cat_bp = predict_catalog_bp(annots, genomes, spliced)
    genome_bp = genomes[0].genome_bp
    long_reads = reads.read_class == "long"
    fast_euk = (
        spliced
        and high_iso
        and not junction_risk
        and not long_reads
        and (prim.n_transcripts >= FAST_EUK_MIN_TRANSCRIPTS or genome_bp >= FAST_EUK_MIN_GENOME_BP)
        and any(k in FAST_EUK_KERNELS for k in kernels)
    )
    router_verdict: dict | None = None
    pseudoaligners_disqualified = False
    if use_island_pc_router:
        router_verdict = _run_island_pc_router(
            assemblies,
            genomes,
            r1,
            r2,
            spliced=spliced,
            strict=strict,
            n_probe=max(2000, int(n_reads)),
            ipc_index=ipc_index,
            ipc_index_dir=ipc_index_dir,
        )
        pseudoaligners_disqualified = bool(router_verdict.get("pseudoaligners_disqualified", False))
        if router_verdict.get("status") == "ok":
            routing_info = router_verdict.get("routing", {})
            reasons.append(
                f"island_pc_router: p_indel={router_verdict.get('p_indel_hat', 0.0):.5f}, "
                f"kr={router_verdict.get('kr_hat', 0.0):.4f}, "
                f"pseudoaligners_disqualified={pseudoaligners_disqualified}, "
                f"rec={routing_info.get('panexpr_recommended', '?')}"
            )
            if pseudoaligners_disqualified and fast_euk:
                fast_euk = False
                reasons.append("island_pc_router disqualified pseudoaligners -> keeping alignment fallback kernels")
    if fast_euk:
        reasons.append(
            f"large high-isoform eukaryote (n_tx~{prim.n_transcripts}, genome_bp={genome_bp:,}) -> "
            f"pruned to fast cDBG equivalence-class pool {sorted(FAST_EUK_KERNELS)}"
        )
    pred: dict[str, int] = {}
    excluded: dict[str, str] = {}
    cands: list[str] = []
    for k in kernels:
        tr = traits_of(k)
        bp = genome_bp if tr.needs_genome else cat_bp
        need = int(tr.ram_base_mb * (1 << 20) + tr.ram_b_per_bp * bp)
        pred[k] = need
        if pseudoaligners_disqualified and tr.family == "kmer_ec":
            excluded[k] = (
                f"island_pc_router disqualified pseudoaligner "
                f"(p_indel={router_verdict.get('p_indel_hat', 0.0):.5f}, kr={router_verdict.get('kr_hat', 0.0):.4f})"
            )
        elif tr.spliced and not spliced:
            excluded[k] = f"spliced aligner, multi_exon_gene_frac={prim.multi_exon_gene_frac:.4f} < {SPLICED_MIN_MULTI_EXON_FRAC}"
        elif strict and tr.pair_collapse:
            excluded[k] = "paired mode collapses equal-score ties by construction; strict mate-tie gate is on"
        elif reads.adapter_rate > ADAPTER_RATE_MAX and tr.end_to_end:
            excluded[k] = f"end-to-end aligner, adapter rate {reads.adapter_rate:.3f} > {ADAPTER_RATE_MAX}"
        elif long_reads and not tr.long_read_ok:
            excluded[k] = f"short-read kernel, median read length {reads.read_len_median:.0f} > {LONG_READ_MEDIAN}"
        elif need > budget * RAM_SAFETY:
            excluded[k] = f"predicted index RAM {need / 2**30:.1f} GiB > {RAM_SAFETY:.0%} of budget {budget / 2**30:.1f} GiB"
        elif fast_euk and k not in FAST_EUK_KERNELS:
            excluded[k] = f"pruned on large high-isoform eukaryotic catalog (n_tx~{prim.n_transcripts}, genome_bp={genome_bp:,})"
        else:
            cands.append(k)
    feats = {
        "annotation": [asdict(a) for a in annots],
        "genome": [asdict(g) for g in genomes],
        "redundancy": asdict(red),
        "reads": asdict(reads),
    }
    if router_verdict is not None:
        feats["island_pc_router"] = router_verdict
    return RegimeDecision(
        catalog_mode=catalog_mode, decoys=decoys, spliced_regime=spliced, unannotated_junction_risk=junction_risk,
        high_isoform=high_iso, tximport_offsets=spliced and high_iso, strict_tie_gate=strict,
        read_class=reads.read_class, mappy_preset="map-ont" if long_reads else "sr", ram_budget_bytes=budget,
        predicted_catalog_bp=cat_bp, predicted_index_ram_bytes=pred, candidates=cands, excluded=excluded,
        reasons=reasons, wall_seconds=round(time.time() - t0, 4), features=feats,
        router_verdict=router_verdict,
    )
