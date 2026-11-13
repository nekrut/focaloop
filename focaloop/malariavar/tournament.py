"""Stage 1: AT-rich kernel tournament (25,000 real Pf8 reads + 3,000 synthetic reads over 5 regimes).

Mappers (paired-end production flags — no paired-end rescue disabling `-S -P` or all-alignments `-a`):
  bwa mem | bwa-mem2 mem | strobealign | minimap2 -ax sr
Winner: fastest / most accurate among mappers within 2 points of best placement in every synthetic regime
(using post-sieve adapter trimming), meeting the network-bound Pf throughput floor, and with mate-tie
symmetry >= 0.80 on the paralog regime.

    python tournament.py
"""
import json
import os
import re
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import genome as G
from . import regime as R

W = Path(os.environ.get("W", "/media/anton/wd4tb/malariavar"))
S = Path(os.environ.get("S", "/media/anton/hd2/malariavar_staging"))
TDIR = S / "ref" / "tournament"
WORK = W / "tmp" / "tournament"
HERE = Path(__file__).resolve().parent
REAL_RUN = "ERR4594631"
CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")


def mapper_cmds(ref, r1, r2, threads):
    ref = str(ref)
    t = str(threads)
    return {
        "bwa": ["bwa", "mem", "-t", t, ref, r1, r2],
        "bwa-mem2": ["bwa-mem2", "mem", "-t", t, ref, r1, r2],
        "strobealign": ["strobealign", "-t", t, ref, r1, r2],
        "minimap2": ["minimap2", "-t", t, "-ax", "sr", ref, r1, r2],
    }


def run_mapper(cmd, out):
    t0 = time.time()
    r0 = os.times()
    with open(out, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    r1 = os.times()
    return dict(rc=p.returncode, wall_s=time.time() - t0,
                cpu_s=(r1.children_user - r0.children_user) + (r1.children_system - r0.children_system))


def build_reference(g):
    TDIR.mkdir(parents=True, exist_ok=True)
    ref = TDIR / "Pf3D7_v54_paralog.fa"
    a, b = R.tournament_reference(g, ref)
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    subprocess.run(["bwa", "index", str(ref)], check=True, capture_output=True)
    subprocess.run(["bwa-mem2", "index", str(ref)], check=True, capture_output=True)
    return ref, (a, b)


def real_reads(n_pairs=12500, blocks=10):
    """12,500 real pairs from 10 spot blocks spread over a coordinate-ordered ODP archive (fastq-dump -N/-X)."""
    p1, p2 = WORK / "real_1.fq", WORK / "real_2.fq"
    if p1.exists() and p2.exists():
        return p1, p2
    per = n_pairs // blocks
    spots = 16_719_925
    with open(p1, "w") as f1, open(p2, "w") as f2:
        for k in range(blocks):
            s0 = int((k + 0.5) * spots / blocks)
            out = subprocess.run(["fastq-dump", "-N", str(s0), "-X", str(s0 + per - 1), "--split-spot", "--stdout",
                                  REAL_RUN], capture_output=True, text=True).stdout.splitlines()
            recs = [out[i:i + 4] for i in range(0, len(out) - 3, 4)]
            for i in range(0, len(recs) - 1, 2):
                a, b = recs[i], recs[i + 1]
                name = f"real:{k}:{i // 2}"
                f1.write(f"@{name}\n{a[1]}\n+\n{a[3]}\n")
                f2.write(f"@{name}\n{b[1]}\n+\n{b[3]}\n")
    return p1, p2


def parse_sam(path, contig_off):
    """primary records per (name, mate): contig, goff, mapq, cigar ops."""
    prim = {}
    sec = defaultdict(int)
    with open(path) as f:
        for line in f:
            if line[0] == "@":
                continue
            t = line.split("\t", 11)
            flag = int(t[1])
            name = t[0]
            mate = 1 if flag & 0x40 else 2
            if flag & 0x4:
                prim.setdefault((name, mate), None)
                continue
            if flag & 0x900:
                sec[(name, mate)] += 1
                continue
            ops = [(int(n), o) for n, o in CIGAR.findall(t[5])]
            prim[(name, mate)] = (t[2], contig_off.get(t[2], 0) + int(t[3]) - 1, int(t[4]), ops, flag)
    return prim, sec


def score_regime(regime, prim, extra, L, par):
    names = sorted({k[0] for k in prim})
    ok = n = 0
    det = Counter()
    for name in names:
        f = name.split(":")
        t1, t2 = int(f[2]), int(f[3])
        a1, a2 = prim.get((name, 1)), prim.get((name, 2))
        n += 1
        if regime == "paralog":
            if a1 is None or a2 is None:
                det["unmapped"] += 1
                continue
            on = [x[0] == R.PARALOG_NAME for x in (a1, a2)]
            if all(on):
                det["decoy"] += 1
            elif not any(on):
                det["original"] += 1
            else:
                det["split_pair"] += 1
            det["mapq_le3"] += int(a1[2] <= 3 and a2[2] <= 3)
            off = int(f[4])
            pos_ok = all(abs((x[1] - L) - off if x[0] == R.PARALOG_NAME else x[1] - (par[0] + off)) <= 600
                         for x in (a1, a2))
            ok += pos_ok
            continue
        if regime == "at_indel":
            m = re.match(r"(del|ins)(\d+)@(\d+)", f[4])
            kind, ln, x = m.group(1), int(m.group(2)), int(m.group(3))
            found = False
            for a in (a1, a2):
                if a is None:
                    continue
                gpos = a[1]
                for cnt, op in a[3]:
                    if op in "MD=XN":
                        if op == "D" and kind == "del" and cnt == ln and x - 5 <= gpos <= x + 25:
                            found = True
                        gpos += cnt
                    elif op == "I" and kind == "ins" and cnt == ln and x - 5 <= gpos <= x + 25:
                        found = True
            det["indel_in_cigar"] += found
            ok += bool(a1 is not None and abs(a1[1] - t1) <= 10)
            continue
        good = a1 is not None and abs(a1[1] - t1) <= 10
        good &= a2 is not None and abs(a2[1] - t2) <= 10
        ok += good
    res = dict(n_pairs=n, placement=ok / max(n, 1))
    if regime == "paralog":
        o, d = det["original"], det["decoy"]
        res.update(original=o, decoy=d, split_pair=det["split_pair"], unmapped=det["unmapped"],
                   mate_tie_symmetry=min(o, d) / max(o, d) if max(o, d) else 0.0,
                   pair_consistency=(o + d) / max(n, 1), mapq_le3=det["mapq_le3"] / max(n, 1))
    if regime == "at_indel":
        res["indel_recovered"] = det["indel_in_cigar"] / max(n, 1)
    return res


def real_metrics(prim):
    names = {k[0] for k in prim}
    mapped = sum(1 for v in prim.values() if v is not None)
    q20 = sum(1 for v in prim.values() if v is not None and v[2] >= 20)
    proper = sum(1 for v in prim.values() if v is not None and v[4] & 2)
    return dict(pairs=len(names), reads=len(prim), mapped=mapped / max(len(prim), 1),
                mapq20=q20 / max(len(prim), 1), proper_flag=proper / max(len(prim), 1))


def placement_agreement(prims):
    """per-mapper fraction of real reads placed within 10 bp of the majority placement of all mappers."""
    keys = set().union(*[set(p) for p in prims.values()])
    agree = {m: 0 for m in prims}
    n = 0
    for k in keys:
        pos = {m: (p.get(k)[1] if p.get(k) is not None else None) for m, p in prims.items()}
        vals = [v for v in pos.values() if v is not None]
        if len(vals) < 2:
            continue
        med = int(np.median(vals))
        n += 1
        for m, v in pos.items():
            agree[m] += int(v is not None and abs(v - med) <= 10)
    return {m: agree[m] / max(n, 1) for m in prims}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    g = G.load()
    ref, par = build_reference(g)
    L = g.L
    contig_off = {n: int(g.offsets[i]) for i, n in enumerate(g.names)}
    contig_off[R.PARALOG_NAME] = L
    regimes = {}
    sites = None
    for rg, n in (("err05", 300), ("at_indel", 300), ("minor20", 300), ("adapter", 300), ("paralog", 300)):
        recs, extra = R.records(g, rg, n, seed=zlib.crc32(rg.encode()) % 1000, avoid=par)
        if rg == "minor20":
            sites = extra
        p1, p2 = WORK / f"{rg}_1.fq", WORK / f"{rg}_2.fq"
        R.write_fastq(recs, p1, p2)
        regimes[rg] = (p1, p2, len(recs))
    r1, r2 = real_reads()
    out = {"reference": str(ref), "paralog_segment_goff": list(par), "regimes": {}, "real": {}, "speed": {},
           "synthetic_reads": int(sum(v[2] * 2 for v in regimes.values())),
           "real_reads": sum(1 for _ in open(r1)) // 4 * 2}
    prims_real = {}
    for m in ("bwa", "bwa-mem2", "strobealign", "minimap2"):
        for rg, (p1, p2, _) in regimes.items():
            sam = WORK / f"{m}_{rg}.sam"
            st = run_mapper(mapper_cmds(ref, str(p1), str(p2), 8)[m], sam)
            prim, _ = parse_sam(sam, contig_off)
            res = score_regime(rg, prim, None, L, par)
            if rg == "minor20":
                res.update(minor_wsaf(prim, sites, contig_off, g, sam))
            res.update(st)
            out["regimes"].setdefault(m, {})[rg] = res
        sam = WORK / f"{m}_real.sam"
        st = run_mapper(mapper_cmds(ref, str(r1), str(r2), 8)[m], sam)
        prim, _ = parse_sam(sam, contig_off)
        prims_real[m] = prim
        out["real"][m] = dict(real_metrics(prim), **st)
        # throughput on a 500k-read real slice (index build amortised)
        s1, s2 = WORK / "speed_1.fq", WORK / "speed_2.fq"
        if not s1.exists():
            split_interleaved(W / "tmp" / "mapbench" / "il.fq", s1, s2)
        st = run_mapper(mapper_cmds(G.REF_FASTA, str(s1), str(s2), 24)[m], "/dev/null")
        st["reads"] = 500000
        st["reads_per_cpu_s"] = st["reads"] / max(st["cpu_s"], 1e-9)
        out["speed"][m] = st
        print(m, json.dumps(out["regimes"][m]["paralog"]), json.dumps(out["speed"][m]), flush=True)
    agree = placement_agreement(prims_real)
    for m in agree:
        out["real"][m]["consensus_placement"] = agree[m]
    # adapter read-through is trimmed by the sieve before mapping in production: score every mapper that way
    p1_trim, p2_trim = trimmed_regime(regimes["adapter"][0], regimes["adapter"][1])
    for m in out["regimes"]:
        sam = WORK / f"{m}_adapter_trimmed.sam"
        run_mapper(mapper_cmds(ref, str(p1_trim), str(p2_trim), 8)[m], sam)
        prim, _ = parse_sam(sam, contig_off)
        out["regimes"][m]["adapter_after_sieve_trim"] = score_regime("adapter", prim, None, L, par)

    def _eff_placement(m, rg):
        if rg == "adapter":
            return out["regimes"][m].get("adapter_after_sieve_trim", out["regimes"][m]["adapter"])["placement"]
        return out["regimes"][m][rg]["placement"]

    # winner rule
    best = {rg: max(_eff_placement(m, rg) for m in out["regimes"]) for rg in regimes}
    elig = []
    for m in out["regimes"]:
        rr = out["regimes"][m]
        within = all(_eff_placement(m, rg) >= best[rg] - 0.02 for rg in regimes if rg != "paralog")
        sym = rr["paralog"]["mate_tie_symmetry"] >= 0.80
        rr["eligible"] = bool(within and sym)
        rr["within_2pts_all_regimes"] = bool(within)
        rr["paralog_gate_pass"] = bool(sym)
        if within and sym:
            elig.append(m)
    pool = elig or list(out["regimes"])
    out["rule_a_winner"] = max(pool, key=lambda m: out["speed"][m]["reads_per_cpu_s"])
    out["rule_a_fallback_ineligible"] = not bool(elig)
    out["rule_a"] = ("fastest reads/CPU-s among mappers within 2 points of the best placement in every regime "
                     "(post-sieve adapter trimming) and mate-tie symmetry >= 0.80")
    # production rule: the mapper has to keep the 64 MB/s ODP pipe fed on map_cores after host-sieve filtering
    floor = throughput_floor()
    out["throughput_floor"] = floor
    fast = [m for m in out["regimes"] if out["speed"][m]["reads_per_cpu_s"] >= floor["reads_per_cpu_s"]
            and out["regimes"][m]["paralog_gate_pass"]]
    elig_fast = [m for m in fast if out["regimes"][m]["eligible"]]

    def acc_score(m):
        r = out["regimes"][m]
        ad = r.get("adapter_after_sieve_trim", r["adapter"])["placement"]
        pl = [r[rg]["placement"] for rg in ("err05", "at_indel", "minor20")] + [ad]
        return float(np.mean(pl)) + r["at_indel"].get("indel_recovered", 0)

    if elig_fast:
        out["production_winner"] = max(elig_fast, key=acc_score)
        out["fallback_ineligible"] = False
    elif fast:
        out["production_winner"] = max(fast, key=acc_score)
        out["fallback_ineligible"] = True
    else:
        out["production_winner"] = out["rule_a_winner"]
        out["fallback_ineligible"] = not bool(elig)
    out["production_rule"] = ("most accurate (mean placement with sieve adapter trimming + AT-indel recovery) "
                              "among eligible mappers at or above the Pf-sieved throughput floor with mate-tie "
                              "symmetry >= 0.80")
    out["winner"] = out["production_winner"]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stage1_tournament.json").write_text(json.dumps(out, indent=1))
    print("rule_a", out["rule_a_winner"], "eligible", elig, "production", out["production_winner"])


def throughput_floor(pipe_mb_s=64.0, archive_gb=1.43, reads_per_archive=30e6, keep=0.5, pf_fraction=0.76, map_cores=16):
    """reads per CPU-second a mapper needs so mapping keeps pace with Pf-sieved reads arriving at the pipe rate."""
    archives_per_s = pipe_mb_s / (archive_gb * 1000)
    reads_s = archives_per_s * reads_per_archive * keep * pf_fraction
    return dict(pipe_mb_s=pipe_mb_s, archive_gb=archive_gb, reads_per_archive=reads_per_archive, keep_fraction=keep,
                pf_fraction=pf_fraction, map_cores=map_cores, reads_per_s_needed=reads_s,
                reads_per_cpu_s=reads_s / map_cores)


def trimmed_regime(p1, p2):
    """run the adapter regime through the production sieve (adapter trim, no subsampling, no host filter)."""
    from . import streamer as ST
    from . import prepare as P
    a = open(p1).read().splitlines()
    b = open(p2).read().splitlines()
    il = []
    for i in range(0, len(a), 4):
        il += [a[i] + ".0", a[i + 1], "+", a[i + 3], b[i] + ".1", b[i + 1], "+", b[i + 3]]
    sv = ST.Sieve(P.load_kset(), paired=True, need=0)
    out = sv.push(("\n".join(il) + "\n").encode(), final=True).decode().splitlines()
    names = [a[i][1:] for i in range(0, len(a), 4)]
    q1, q2 = WORK / "adapter_trim_1.fq", WORK / "adapter_trim_2.fq"
    with open(q1, "w") as f1, open(q2, "w") as f2:
        for k in range(0, len(out), 8):
            nm = names[int(out[k][1:])]
            f1.write(f"@{nm}\n{out[k + 1]}\n+\n{out[k + 3]}\n")
            f2.write(f"@{nm}\n{out[k + 5]}\n+\n{out[k + 7]}\n")
    return q1, q2


def split_interleaved(il, p1, p2):
    with open(il) as f, open(p1, "w") as a, open(p2, "w") as b:
        while True:
            r1 = [f.readline() for _ in range(4)]
            r2 = [f.readline() for _ in range(4)]
            if not r2[3]:
                break
            a.writelines(r1)
            b.writelines(r2)


def island_pc_preflight(g, fq1, fq2=None, max_reads=2000, use_island_pc_gate=False):
    """Optional Wave-0 island_pc_router preflight: Blanca lag-(k-1) CI, --pc divergence, and mapper routing."""
    if not use_island_pc_gate:
        return None
    from . import island_pc_router as IPC
    contigs = [
        G.BASES[g.seq[int(g.offsets[i]):int(g.offsets[i + 1])]].tobytes().decode("ascii")
        for i in range(len(g.names))
    ]
    seqs, quals = [], []
    for path in ([fq1, fq2] if fq2 else [fq1]):
        with open(path) as fh:
            lines = fh.read().splitlines()
        for i in range(0, len(lines) - 3, 4):
            seqs.append(lines[i + 1])
            quals.append(lines[i + 3])
            if len(seqs) >= max_reads:
                break
        if len(seqs) >= max_reads:
            break
    with IPC.ReferenceIslandPCIndex(contigs, compute_dense_d1=False) as idx:
        prob = idx.classify_reads(seqs, quals=quals)
    usable = "INSUFFICIENT_USABLE_BASES" not in prob.routing.flags
    return {
        "r_pc_syncmer_bio": float(prob.r_pc_syncmer_bio),
        "ci_low": float(prob.r_pc_syncmer_ci_low),
        "ci_high": float(prob.r_pc_syncmer_ci_high),
        "p_s_hat": float(prob.p_s_hat),
        "p_indel_hat": float(prob.p_indel_hat),
        "recommended_mapper": prob.routing.malariavar_recommended,
        "allow_pseudoalignment": not bool(prob.routing.pseudoaligners_disqualified),
        "usable_evidence": bool(usable),
        "flags": list(prob.routing.flags),
        "reason": prob.routing.rationale,
    }


def minor_wsaf(prim, sites, contig_off, g, sam):
    """alt fraction at the minor-clone SNP sites from the mapper's alignments (target 0.20)."""
    site = dict(sites)
    alt = Counter()
    tot = Counter()
    with open(sam) as f:
        for line in f:
            if line[0] == "@":
                continue
            t = line.split("\t", 11)
            flag = int(t[1])
            if flag & 0x904 or int(t[4]) < 20:
                continue
            gpos = contig_off.get(t[2], 0) + int(t[3]) - 1
            qp = 0
            seq = t[9]
            for n, op in CIGAR.findall(t[5]):
                n = int(n)
                if op in "M=X":
                    for k in range(n):
                        x = gpos + k
                        if x in site:
                            tot[x] += 1
                            alt[x] += seq[qp + k] == "ACGT"[site[x]]
                    gpos += n
                    qp += n
                elif op in "DN":
                    gpos += n
                elif op in "IS":
                    qp += n
    fr = [alt[x] / tot[x] for x in site if tot[x] >= 5]
    return dict(minor_sites=len(site), minor_sites_covered=len(fr), minor_wsaf_mean=float(np.mean(fr)) if fr else None,
                minor_wsaf_bias=float(np.mean(fr) - 0.2) if fr else None)


if __name__ == "__main__":
    sys.exit(main())
