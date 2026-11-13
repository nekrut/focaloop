"""Pf3D7 PlasmoDB-54 reference model shared by every MalariaVar stage.

One flat coordinate space (``goff``) over the 16 contigs, 2-bit base codes, the Pf3k/Pf8 core-genome
mask, 1 kb bins with GC content, curated loci from the PlasmoDB-55 GFF, and codon lookups.
"""
import gzip
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

_DEFAULT_PF3D7 = Path("/opt/focaloop/pf3d7_ref")
S = Path(os.environ.get("S") or (str(_DEFAULT_PF3D7) if _DEFAULT_PF3D7.exists() else "/media/anton/hd2/malariavar_staging"))
REFDIR = Path(
    os.environ.get("FOCALOOP_PF3D7_DIR")
    or os.environ.get("MALARIAVAR_REFDIR")
    or (str(_DEFAULT_PF3D7 / "ref") if (_DEFAULT_PF3D7 / "ref").exists() else str(S / "ref"))
)
REF_FASTA = REFDIR / "Pf3D7_v54.fa"
GFF = (REFDIR.parent / "data" / "PlasmoDB-55_Pfalciparum3D7.gff.gz") if (REFDIR.parent / "data").exists() else (S / "data" / "PlasmoDB-55_Pfalciparum3D7.gff.gz")
REGIONS = (REFDIR.parent / "data" / "regions-20130225.onebased.txt") if (REFDIR.parent / "data").exists() else (S / "data" / "regions-20130225.onebased.txt")
BIN = 1000
NUCLEAR = [f"Pf3D7_{i:02d}_v3" for i in range(1, 15)]

# curated loci (PlasmoDB gene IDs)
GENES = {
    "kelch13": "PF3D7_1343700", "crt": "PF3D7_0709000", "mdr1": "PF3D7_0523000",
    "dhfr": "PF3D7_0417200", "dhps": "PF3D7_0810800", "gch1": "PF3D7_1224000",
    "pm2": "PF3D7_1408000", "pm3": "PF3D7_1408100", "hrp2": "PF3D7_0831800",
    "hrp3": "PF3D7_1372200", "aat1": "PF3D7_0629500", "pfs47": "PF3D7_1346800",
    "exo": "PF3D7_1362500", "arps10": "PF3D7_1460900", "fd": "PF3D7_1318100",
    "mdr2": "PF3D7_1447900", "ama1": "PF3D7_1133400", "csp": "PF3D7_0304600",
}
CNV_LOCI = {"crt": ["crt"], "mdr1": ["mdr1"], "gch1": ["gch1"], "pm2_pm3": ["pm2", "pm3"]}
DEL_LOCI = {"hrp2": "hrp2", "hrp3": "hrp3"}
ZOOM_GENES = ["kelch13", "crt", "mdr1", "dhfr", "dhps", "exo", "arps10", "fd", "mdr2", "aat1"]

CODE = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    CODE[_c] = _i
    CODE[ord(chr(_c).lower())] = _i
BASES = np.frombuffer(b"ACGTN", dtype=np.uint8)
COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
CODON = {}
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for _x in range(4):
    for _y in range(4):
        for _z in range(4):
            CODON[_b[_x] + _b[_y] + _b[_z]] = _aa[16 * _x + 4 * _y + _z]


def revcomp(s):
    return "".join(COMP[c] for c in reversed(s))


def read_fasta(path):
    names, seqs, cur = [], [], []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                if cur:
                    seqs.append(b"".join(cur))
                names.append(line[1:].split()[0].decode())
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append(b"".join(cur))
    return names, seqs


@dataclass
class Genome:
    names: list
    lengths: np.ndarray
    offsets: np.ndarray          # goff of each contig start (len = n + 1)
    seq: np.ndarray              # uint8 codes 0..3, 4 = N
    core: np.ndarray             # bool per base, Pf3k "Core" regions
    bin_offsets: np.ndarray      # first bin id of each contig (len = n + 1)
    bin_gc: np.ndarray           # GC fraction per 1 kb bin
    bin_core: np.ndarray         # fraction of core bases per bin
    genes: dict = field(default_factory=dict)

    @property
    def L(self):
        return int(self.offsets[-1])

    def cidx(self, name):
        return self.names.index(name)

    def goff(self, contig, pos1):
        """1-based contig position -> flat 0-based offset."""
        return int(self.offsets[self.cidx(contig)]) + int(pos1) - 1

    def locate(self, g):
        """flat offset(s) -> (contig index, 1-based pos)."""
        g = np.asarray(g)
        c = np.searchsorted(self.offsets, g, side="right") - 1
        return c, g - self.offsets[c] + 1

    def bin_of(self, g):
        c, p = self.locate(g)
        return self.bin_offsets[c] + (p - 1) // BIN

    def bin_range(self, contig, start1, end1):
        c = self.cidx(contig)
        b0 = self.bin_offsets[c]
        return int(b0 + (start1 - 1) // BIN), int(b0 + (end1 - 1) // BIN) + 1

    def nuclear_mask(self):
        m = np.zeros(self.L, bool)
        for n in NUCLEAR:
            c = self.cidx(n)
            m[self.offsets[c]:self.offsets[c + 1]] = True
        return m

    def bases(self, g0, g1):
        return BASES[self.seq[g0:g1]].tobytes().decode()

    def gene(self, key):
        return self.genes[GENES.get(key, key)]

    def codon_goffs(self, key, codon):
        """flat offsets of the three codon bases in transcript order plus strand."""
        g = self.gene(key)
        cds = sorted(g["cds"], key=lambda e: e[0])
        pos = []
        for s, e in cds:
            pos.extend(range(s, e + 1))
        if g["strand"] == "-":
            pos = pos[::-1]
        i = 3 * (int(codon) - 1)
        c0 = self.offsets[self.cidx(g["contig"])]
        return [int(c0 + p - 1) for p in pos[i:i + 3]], g["strand"]

    def codon_aa(self, key, codon, bases3):
        """translate transcript-order bases (genomic strand letters) to an amino acid."""
        _, strand = self.codon_goffs(key, codon)
        s = "".join(bases3)
        if strand == "-":
            s = "".join(COMP[c] for c in s)
        return CODON.get(s, "X")


def _parse_gff(names, gff=None):
    gff = gff or GFF
    genes = {}
    want = set(GENES.values())
    with gzip.open(gff, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.rstrip("\n").split("\t")
            if len(t) < 9 or t[0] not in names:
                continue
            attr = dict(kv.split("=", 1) for kv in t[8].split(";") if "=" in kv)
            if t[2] in ("protein_coding_gene", "gene", "ncRNA_gene", "pseudogene") and attr.get("ID") in want:
                genes.setdefault(attr["ID"], {"cds": [], "exons": []}).update(
                    {"contig": t[0], "start": int(t[3]), "end": int(t[4]), "strand": t[6]})
            elif t[2] == "CDS" and attr.get("gene_id") in want and attr.get("Parent", "").endswith(".1"):
                genes.setdefault(attr["gene_id"], {"cds": [], "exons": []})["cds"].append((int(t[3]), int(t[4])))
            elif t[2] == "exon" and attr.get("gene_id") in want and attr.get("Parent", "").endswith(".1"):
                genes.setdefault(attr["gene_id"], {"cds": [], "exons": []})["exons"].append((int(t[3]), int(t[4])))
    return genes


def _source_fingerprint(fasta=None, gff=None, regions=None):
    fasta, gff, regions = Path(fasta or REF_FASTA), Path(gff or GFF), Path(regions or REGIONS)
    meta = {"genes": sorted(GENES.items()), "bin": BIN}
    for key, p in (("fasta", fasta), ("gff", gff), ("regions", regions)):
        if p.exists():
            st = p.stat()
            meta[key] = [int(st.st_size), int(st.st_mtime_ns)]
    return json.dumps(meta, sort_keys=True)


def build(cache=REFDIR / "genome_cache.npz", fasta=None, gff=None, regions=None):
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    fasta, gff, regions = Path(fasta or REF_FASTA), Path(gff or GFF), Path(regions or REGIONS)
    names, seqs = read_fasta(fasta)
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    seq = CODE[np.frombuffer(b"".join(seqs), dtype=np.uint8)]
    core = np.zeros(int(offsets[-1]), bool)
    with open(regions) as f:
        for line in f:
            c, s, e, kind = line.split()[:4]
            if kind == "Core" and c in names:
                i = names.index(c)
                core[offsets[i] + int(s) - 1: offsets[i] + int(e)] = True
    nb = (lengths + BIN - 1) // BIN
    bin_offsets = np.concatenate([[0], np.cumsum(nb)])
    gc = np.zeros(int(bin_offsets[-1]), np.float32)
    bcore = np.zeros_like(gc)
    for i in range(len(names)):
        s = seq[offsets[i]:offsets[i + 1]]
        cm = core[offsets[i]:offsets[i + 1]]
        for b in range(int(nb[i])):
            w = s[b * BIN:(b + 1) * BIN]
            acgt = w[w < 4]
            gc[bin_offsets[i] + b] = np.isin(acgt, (1, 2)).mean() if len(acgt) else 0
            bcore[bin_offsets[i] + b] = cm[b * BIN:(b + 1) * BIN].mean()
    genes = _parse_gff(names, gff=gff)
    fp = _source_fingerprint(fasta, gff, regions)
    np.savez(cache, names=np.array(names), lengths=lengths, offsets=offsets, seq=seq, core=core,
             bin_offsets=bin_offsets, bin_gc=gc, bin_core=bcore, fingerprint=np.array(fp))
    cache.with_name("genes.json").write_text(json.dumps(genes, indent=1))
    return _read_cache_npz(cache)


def _cache_is_stale(cache, genes_path, z, genes, fasta, gff, regions):
    if not os.access(cache.parent, os.W_OK):
        return False
    sources = [Path(p) for p in (fasta, gff, regions)]
    have_all_sources = all(p.exists() for p in sources)
    if not have_all_sources:
        return False
    if "fingerprint" in z.files:
        return str(z["fingerprint"]) != _source_fingerprint(fasta, gff, regions)
    cache_mtime = min(cache.stat().st_mtime_ns, genes_path.stat().st_mtime_ns)
    if any(p.stat().st_mtime_ns > cache_mtime for p in sources):
        return True
    return not set(GENES.values()).issubset(genes.keys())


def _read_cache_npz(cache: Path):
    genes_path = cache.with_name("genes.json")
    with np.load(cache) as z:
        genes = json.loads(genes_path.read_text())
        g = Genome(names=[str(x) for x in z["names"]], lengths=z["lengths"], offsets=z["offsets"], seq=z["seq"],
                   core=z["core"], bin_offsets=z["bin_offsets"], bin_gc=z["bin_gc"], bin_core=z["bin_core"])
    g.genes = genes
    return g


@lru_cache(maxsize=8)
def _load_keyed(cache: Path, fasta: Path, gff: Path, regions: Path, fp: str):
    genes_path = cache.with_name("genes.json")
    if not cache.exists() or not genes_path.exists():
        return build(cache, fasta=fasta, gff=gff, regions=regions)
    with np.load(cache) as z:
        genes = json.loads(genes_path.read_text())
        stale = _cache_is_stale(cache, genes_path, z, genes, fasta, gff, regions)
    if stale:
        return build(cache, fasta=fasta, gff=gff, regions=regions)
    return _read_cache_npz(cache)


def load(cache=REFDIR / "genome_cache.npz", fasta=None, gff=None, regions=None):
    cache = Path(cache)
    fasta, gff, regions = Path(fasta or REF_FASTA), Path(gff or GFF), Path(regions or REGIONS)
    return _load_keyed(cache, fasta, gff, regions, _source_fingerprint(fasta, gff, regions))


def verify_against_cram_header(header_text):
    """True when every @SQ name/length in a Pf8 CRAM header matches the local reference."""
    g = load()
    sq = {}
    for line in header_text.splitlines():
        if line.startswith("@SQ"):
            d = dict(x.split(":", 1) for x in line.split("\t")[1:] if ":" in x)
            sq[d["SN"]] = int(d["LN"])
    ref = dict(zip(g.names, (int(x) for x in g.lengths)))
    return sq == ref, sq, ref
