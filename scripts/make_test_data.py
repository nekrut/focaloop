#!/usr/bin/env python3
"""Regenerate the compact deterministic synthetic fixtures under blueprints/*/test-data/.

    python scripts/make_test_data.py            # fixtures for panexpr, malariavar, focaloop_bait
    python scripts/make_test_data.py --state    # also the focaloop_query warm-state archive (runs focaloop)

Everything is seeded; gzip members are written with mtime=0 so reruns are byte-identical.
"""

from __future__ import annotations

import argparse
import gzip
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMP = str.maketrans("ACGT", "TGCA")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def mutate(rng: random.Random, s: str, rate: float) -> str:
    b = list(s)
    for i, ch in enumerate(b):
        if rng.random() < rate:
            b[i] = rng.choice([x for x in "ACGT" if x != ch])
    return "".join(b)


def write_fasta(path: Path, recs: list[tuple[str, str, dict[str, str]]]) -> None:
    with open(path, "w", encoding="ascii") as fh:
        for name, seq, attrs in recs:
            hdr = " ".join([name] + [f"{k}={v}" for k, v in attrs.items()])
            fh.write(f">{hdr}\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i:i + 70] + "\n")


def write_fastq(path: Path, prefix: str, reads: list[str], gz: bool = True) -> None:
    buf = io.StringIO()
    for i, r in enumerate(reads):
        buf.write(f"@{prefix}.{i + 1}\n{r}\n+\n{'I' * len(r)}\n")
    data = buf.getvalue().encode("ascii")
    if gz:
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as fh:
            fh.write(data)
    else:
        path.write_bytes(data)


def sample_reads(rng: random.Random, seqs: dict[str, str], weights: dict[str, float], n: int, read_len: int = 100,
                 err: float = 0.002, antisense: float = 0.85) -> list[str]:
    names = [t for t in weights if weights[t] > 0]
    w = [weights[t] for t in names]
    out = []
    for _ in range(n):
        t = rng.choices(names, w)[0]
        s = seqs[t]
        st = rng.randint(0, len(s) - read_len)
        frag = mutate(rng, s[st:st + read_len], err)
        out.append(rc(frag) if rng.random() < antisense else frag)
    return out


# ---------------------------------------------------------------------------
def make_panexpr(out: Path) -> None:
    rng = random.Random(20261109)
    out.mkdir(parents=True, exist_ok=True)
    base = {f"g{i}": rand_seq(rng, 900) for i in range(1, 9)}
    shared = rand_seq(rng, 160)  # paralog block shared verbatim by g7 and g8 (different groups)
    base["g7"] = base["g7"][:300] + shared + base["g7"][460:]
    base["g8"] = base["g8"][:500] + shared + base["g8"][660:]
    recs, seqs, groups = [], {}, {}
    for g, s in base.items():
        tid = f"{g}.strainA"
        recs.append((tid, s, {"group": g, "assembly": "strainA", "kind": "transcript", "kingdom": "bacteria"}))
        seqs[tid], groups[tid] = s, g
    for g in ("g1", "g2", "g3"):
        s = mutate(rng, base[g], 0.03)
        tid = f"{g}.strainB"
        recs.append((tid, s, {"group": g, "assembly": "strainB", "kind": "transcript", "kingdom": "bacteria"}))
        seqs[tid], groups[tid] = s, g
    vir = rand_seq(rng, 900)
    recs.append(("phage_v1", vir, {"group": "phage_v1", "assembly": "phage", "kind": "transcript", "kingdom": "virus"}))
    seqs["phage_v1"], groups["phage_v1"] = vir, "phage_v1"
    write_fasta(out / "targets.fa", recs)

    abundance = {"g1": 10, "g2": 6, "g3": 8, "g4": 12, "g5": 5, "g6": 7, "g7": 6, "g8": 6, "phage_v1": 4}
    design = [("ctrl1", "control", "b1"), ("ctrl2", "control", "b2"), ("ctrl3", "control", "b1"),
              ("trt1", "treated", "b1"), ("trt2", "treated", "b2"), ("trt3", "treated", "b2")]
    sheet = ["sample\tcondition\tbatch"]
    for sample, cond, batch in design:
        w = {}
        for tid in seqs:
            g = groups[tid]
            a = abundance[g] * rng.uniform(0.85, 1.15)
            if cond == "treated" and g == "g2":
                a *= 5.0
            if cond == "treated" and g == "g4":
                a /= 5.0
            if tid.endswith("strainB"):
                a *= 0.4
            w[tid] = a
        reads = sample_reads(rng, seqs, w, 1400) + [rand_seq(rng, 100) for _ in range(30)]  # ~2% unmappable
        rng.shuffle(reads)
        write_fastq(out / f"{sample}.fastq.gz", sample, reads)
        sheet.append(f"{sample}\t{cond}\t{batch}")
    (out / "samples.tsv").write_text("\n".join(sheet) + "\n", encoding="ascii")
    spec = ["targets: targets.fa", "contrast: [treated, control]", "samples:"]
    for sample, cond, batch in design:
        spec += [f"  - sample: {sample}", f"    condition: {cond}", f"    batch: {batch}",
                 f"    reads: {sample}.fastq.gz"]
    (out / "spec.yml").write_text("\n".join(spec) + "\n", encoding="ascii")
    bed = [f"chrA\t{i * 1000}\t{i * 1000 + 900}\t{g}" for i, g in enumerate(list(base) + ["phage_v1"])]
    (out / "targets.bed").write_text("\n".join(bed) + "\n", encoding="ascii")
    (out / "priors.tsv").write_text(
        "group_id\ttarget_id\tsample\tmodality\tlog_prior_weight\tconfidence\tsource_uri\n"
        "g1\tg1.strainB\t*\twgs_copy_number\t-2.0\t0.9\tsynthetic:wgs\n", encoding="ascii")


# ---------------------------------------------------------------------------
def make_malariavar(out: Path) -> None:
    rng = random.Random(8)
    out.mkdir(parents=True, exist_ok=True)
    genes = ["pfcrt", "pfmdr1", "gch1", "pm2", "pm3", "hrp2", "hrp3", "ctrl_1", "ctrl_2", "ctrl_3"]
    seqs = {g: rand_seq(rng, 600) for g in genes}
    block = rand_seq(rng, 140)
    seqs["pm2"] = seqs["pm2"][:200] + block + seqs["pm2"][340:]
    seqs["pm3"] = seqs["pm3"][:250] + block + seqs["pm3"][390:]
    human = rand_seq(rng, 1500)
    recs = [(g, seqs[g], {"group": g, "assembly": "Pf3D7", "kind": "gene_body", "kingdom": "protist"}) for g in genes]
    recs.append(("human_decoy", human, {"group": "human_decoy", "assembly": "GRCh38", "kind": "host_decoy", "kingdom": "host"}))
    write_fasta(out / "targets.fa", recs)

    def alt_base(b: str) -> str:
        return {"A": "C", "C": "G", "G": "T", "T": "A"}[b]

    sites = [
        ("pfcrt_K76T", "pfcrt", 228, None), ("pfmdr1_N86Y", "pfmdr1", 256, None),
        ("ctrl1_snp", "ctrl_1", 300, None), ("ctrl2_snp", "ctrl_2", 310, None),
        ("ctrl3_del", "ctrl_3", 280, "del"),
    ]
    rows = ["site_id\ttarget_id\tpos\tref\talt\tkind"]
    site_alleles = {}
    for sid, g, pos, kind in sites:
        if kind == "del":
            ref = seqs[g][pos:pos + 3]
            alt = ref[0]
            rows.append(f"{sid}\t{g}\t{pos}\t{ref}\t{alt}\tindel")
        else:
            ref = seqs[g][pos]
            alt = alt_base(ref)
            rows.append(f"{sid}\t{g}\t{pos}\t{ref}\t{alt}\tsnp")
        site_alleles[sid] = (g, pos, ref, alt)
    (out / "sites.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")

    def with_alts(alts: set[str]) -> dict[str, str]:
        s = dict(seqs)
        for sid in alts:
            g, pos, ref, alt = site_alleles[sid]
            s[g] = s[g][:pos] + alt + s[g][pos + len(ref):]
        return s

    # (sample, site, year, clone mixture [(fraction, alt set)], copy multipliers, deleted genes)
    design = [
        ("ML02", "Mali", "2019", [(1.0, set())], {}, set()),
        ("ML03", "Mali", "2019", [(0.6, set()), (0.4, {"ctrl1_snp", "ctrl2_snp", "pfmdr1_N86Y"})], {}, set()),
        ("GH01", "Ghana", "2021", [(1.0, {"pfcrt_K76T", "ctrl3_del", "ctrl1_snp"})], {"pfmdr1": 2.0}, set()),
        ("GH05", "Ghana", "2021", [(1.0, {"pfcrt_K76T", "ctrl2_snp"})], {}, {"hrp2"}),
    ]
    sheet = ["sample\tcondition\tsite\tyear"]
    for sample, site, year, clones, cn, deleted in design:
        reads: list[str] = []
        for frac, alts in clones:
            s = with_alts(alts)
            w = {g: (0.0 if g in deleted else cn.get(g, 1.0)) for g in genes}
            reads += sample_reads(rng, s, w, int(1200 * frac), err=0.001, antisense=0.5)
        reads += sample_reads(rng, {"human_decoy": human}, {"human_decoy": 1.0}, 150, err=0.001, antisense=0.5)
        rng.shuffle(reads)
        write_fastq(out / f"{sample}.fastq.gz", sample, reads)
        sheet.append(f"{sample}\t{site}\t{site}\t{year}")
    (out / "samples.tsv").write_text("\n".join(sheet) + "\n", encoding="ascii")
    spec = ["targets: targets.fa", "variant_sites: sites.tsv", "samples:"]
    for sample, site, year, *_ in design:
        spec += [f"  - sample: {sample}", f"    condition: {site}", f"    site: {site}", f"    year: '{year}'",
                 f"    reads: {sample}.fastq.gz"]
    (out / "spec.yml").write_text("\n".join(spec) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
def make_bait(out: Path) -> None:
    rng = random.Random(31)
    out.mkdir(parents=True, exist_ok=True)
    kelch = rand_seq(rng, 360)
    write_fasta(out / "query.fa", [("pfk13_propeller", kelch, {})])
    background = rand_seq(rng, 5000)
    bg = {"bg": background}
    hit = sample_reads(rng, bg, {"bg": 1.0}, 300, read_len=100) + \
        sample_reads(rng, {"k": kelch}, {"k": 1.0}, 40, read_len=100)
    rng.shuffle(hit)
    write_fastq(out / "run_hit.fastq.gz", "run_hit", hit)
    write_fastq(out / "run_miss.fastq.gz", "run_miss", sample_reads(rng, bg, {"bg": 1.0}, 340, read_len=100))
    weak = sample_reads(rng, bg, {"bg": 1.0}, 300, read_len=100) + \
        sample_reads(rng, {"k": kelch}, {"k": 1.0}, 6, read_len=100)
    rng.shuffle(weak)
    write_fastq(out / "run_weak.fastq", "run_weak", weak, gz=False)


def make_query_state(out: Path) -> None:
    import os

    os.environ["FOCALOOP_FIXED_CLOCK"] = "1"  # wall-clock fields are the only non-deterministic bytes
    sys.path.insert(0, str(ROOT))
    from focaloop.cli import main

    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / "blueprints" / "panexpr" / "test-data"
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rc_ = main(["run", "--blueprint", str(ROOT / "blueprints" / "panexpr" / "blueprint.focaloop.yml"),
                    "--spec", str(src / "spec.yml"), "--state-dir", f"{td}/state",
                    "--state-archive", str(out / "panexpr_state.tar.zst"),
                    "--out-table", f"{td}/t.tsv", "--out-telemetry", f"{td}/tel.json"])
        if rc_ != 0:
            raise SystemExit(f"focaloop run failed with {rc_}")
        rc_ = main(["run", "--blueprint", str(ROOT / "blueprints" / "malariavar" / "blueprint.focaloop.yml"),
                    "--spec", str(ROOT / "blueprints" / "malariavar" / "test-data" / "spec.yml"),
                    "--state-dir", f"{td}/mstate", "--state-archive", str(out / "malariavar_state.tar.zst"),
                    "--out-table", f"{td}/m.tsv", "--out-telemetry", f"{td}/mtel.json"])
        if rc_ != 0:
            raise SystemExit(f"focaloop run (malariavar) failed with {rc_}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", action="store_true", help="also regenerate focaloop_query warm-state archives")
    a = ap.parse_args()
    bp = ROOT / "blueprints"
    make_panexpr(bp / "panexpr" / "test-data")
    make_malariavar(bp / "malariavar" / "test-data")
    make_bait(bp / "focaloop_bait" / "test-data")
    if a.state:
        make_query_state(bp / "focaloop_query" / "test-data")


if __name__ == "__main__":
    main()
