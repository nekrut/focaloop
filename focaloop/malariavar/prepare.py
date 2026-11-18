"""Build every static input the engine needs into $S/ref (small, NVMe).

    python prepare.py            # genome cache, Pf 31-mer host-sieve set, F_WS/WSAF site panel, CRAM header check
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import genome as G
from . import kernels as Kn

W = G.default_work_dir()
KSET = G.REFDIR / "pf_kset31.u64"
KSET_BITS = 26
DUST_MAX = 80
PLA_SIEVE = Path(os.environ.get("MALARIAVAR_PLA_SIEVE") or (G.REFDIR / "pf_pla_sieve.pla"))
PANEL = G.REFDIR / "site_panel.npz"
TRUTH_VARIANTS = W / "truth_200" / "variants.zarr"


def build_pla_sieve(out_path=None, eps=8, dust_max=DUST_MAX, human_decoy_fasta=None, use_splitmix=True, num_threads=8):
    """Build splitmix64-permuted mmap PLA-index over Pf3D7 canonical 15-mer open syncmers (K=31, S=15, t*=8)."""
    from . import pla_sieve as PS
    g = G.load()
    target = Path(out_path or PLA_SIEVE)
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_keys = []
    # Also compute exact raw vs splitmix64 open-syncmer density across nuclear Pf3D7 genome via island_pc_router
    try:
        from . import island_pc_router as IPC
        measure_fn = IPC.measure_syncmer_density
    except Exception:
        measure_fn = None

    raw_kmers_tot = raw_sync_tot = sm_sync_tot = 0
    for c_idx in range(len(g.names)):
        s_off = int(g.offsets[c_idx])
        e_off = int(g.offsets[c_idx + 1])
        seq_b = G.BASES[g.seq[s_off:e_off]].tobytes()
        ks = PS.extract_open_syncmers_u64(
            seq_b, k=Kn.K, s=15, skip_lower=False, use_splitmix=use_splitmix, dust_max=dust_max
        )
        all_keys.append(ks)
        if measure_fn is not None:
            r_d = measure_fn(seq_b, k=Kn.K, s=15, use_splitmix=False)
            s_d = measure_fn(seq_b, k=Kn.K, s=15, use_splitmix=True)
            raw_kmers_tot += int(r_d["n_kmers"])
            raw_sync_tot += int(r_d["n_syncmers"])
            sm_sync_tot += int(s_d["n_syncmers"])

    merged = np.unique(np.concatenate(all_keys)) if all_keys else np.empty(0, dtype=np.uint64)
    n_decoy_removed = 0
    if human_decoy_fasta is not None and Path(human_decoy_fasta).exists():
        import gzip
        p_dec = Path(human_decoy_fasta)
        opener = gzip.open if p_dec.name.endswith(".gz") else open
        decoy_chunks = []
        with opener(p_dec, "rb") as fh:
            buf = bytearray()
            for line in fh:
                if line.startswith(b">"):
                    if buf:
                        decoy_chunks.append(
                            PS.extract_open_syncmers_u64(
                                bytes(buf), k=Kn.K, s=15, skip_lower=False,
                                use_splitmix=use_splitmix, dust_max=-1,
                            )
                        )
                        buf.clear()
                else:
                    buf.extend(line.strip())
            if buf:
                decoy_chunks.append(
                    PS.extract_open_syncmers_u64(
                        bytes(buf), k=Kn.K, s=15, skip_lower=False,
                        use_splitmix=use_splitmix, dust_max=-1,
                    )
                )
        if decoy_chunks:
            dec_u = np.unique(np.concatenate(decoy_chunks))
            n_before = len(merged)
            merged = np.setdiff1d(merged, dec_u, assume_unique=True)
            n_decoy_removed = n_before - len(merged)

    with PS.PLAIndex.build(
        merged, eps=eps, use_perm=True, k=Kn.K, s=15,
        num_threads=num_threads, use_splitmix_syncmer=use_splitmix,
    ) as idx:
        idx.save_mmap(target)
        st = idx.stats()

    raw_dens = (raw_sync_tot / raw_kmers_tot) if raw_kmers_tot else None
    sm_dens = (sm_sync_tot / raw_kmers_tot) if raw_kmers_tot else None
    lift_pct = ((sm_dens - raw_dens) / raw_dens * 100.0) if (raw_dens and sm_dens) else None
    meta = {
        "path": str(target),
        "k": Kn.K,
        "s": 15,
        "t_star": 8,
        "eps": int(eps),
        "use_splitmix_syncmer": bool(use_splitmix),
        "dust_max": int(dust_max),
        "n_keys": int(st.n_keys),
        "n_segs": int(st.n_segs),
        "max_abs_err": int(st.max_abs_err),
        "total_bytes": int(st.total_bytes),
        "total_mb": float(st.total_bytes / (1024 * 1024)),
        "bytes_per_key": float(st.total_bytes_per_key),
        "decoy_removed": int(n_decoy_removed),
        "syncmer_density_raw": raw_dens,
        "syncmer_density_splitmix": sm_dens,
        "syncmer_density_lift_pct": lift_pct,
        "build_s": float(time.time() - t0),
    }
    target.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def load_pla_sieve(path=None, prefault_spline=True):
    from . import pla_sieve as PS
    target = Path(path or PLA_SIEVE)
    return PS.PLAIndex.open_mmap(target, map_private=False, prefault_spline=prefault_spline)


def build_kset():
    g = G.load()
    tab = np.zeros(1 << KSET_BITS, np.uint64)
    t0 = time.time()
    n_ins, n_dust = Kn.build_kset(g.seq, g.offsets[1:].astype(np.int64), tab, DUST_MAX)
    tab.tofile(KSET)
    meta = {"k": Kn.K, "slots": len(tab), "inserted": int(n_ins), "dust_masked": int(n_dust),
            "dust_max": DUST_MAX, "load": float(n_ins / len(tab)), "build_s": time.time() - t0}
    (G.REFDIR / "pf_kset31.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_kset():
    return np.memmap(KSET, dtype=np.uint64, mode="r")


def build_panel():
    """Pf8 PASS core biallelic SNP sites with callset-wide AF (used as PLAF for F_WS and the WSAF sketch)."""
    import zarr
    g = G.load()
    vz = zarr.open_group(str(TRUTH_VARIANTS), mode="r")
    chrom = vz["CHROM"][:]
    pos = vz["POS"][:]
    keep = vz["FILTER_PASS"][:] & vz["is_snp"][:] & (vz["numalt"][:] == 1)
    af = vz["AF"][:, 0]
    keep &= (af >= 0.0003) & (af <= 0.9997)
    idx = np.where(keep)[0]
    cidx = {n: i for i, n in enumerate(g.names)}
    ci = np.array([cidx.get(c, -1) for c in chrom[idx]])
    ok = ci >= 0
    idx, ci = idx[ok], ci[ok]
    goff = g.offsets[ci] + pos[idx] - 1
    core = g.core[goff]
    idx, goff = idx[core], goff[core]
    ref = vz["REF"].get_orthogonal_selection(idx)
    alt = vz["ALT"].get_orthogonal_selection((idx, slice(0, 1)))[:, 0]
    rc = G.CODE[np.frombuffer("".join(ref).encode(), np.uint8)]
    ac = G.CODE[np.frombuffer("".join(alt).encode(), np.uint8)]
    agree = rc == g.seq[goff]
    np.savez(PANEL, goff=goff[agree].astype(np.int64), ref=rc[agree], alt=ac[agree],
             af=af[idx][agree].astype(np.float32), vidx=idx[agree].astype(np.int64))
    meta = {"n_sites": int(agree.sum()), "ref_mismatch": int((~agree).sum()),
            "rule": "FILTER_PASS & is_snp & numalt==1 & 0.0003<=AF<=0.9997 (>= ~10 alleles in 33,325 samples) & Pf3k core"}
    (G.REFDIR / "site_panel.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_panel():
    z = np.load(PANEL)
    return {k: z[k] for k in z.files}


def check_cram_header(url):
    h = subprocess.run(["samtools", "view", "-H", url], capture_output=True, text=True, timeout=300, check=True).stdout
    ok, sq, ref = G.verify_against_cram_header(h)
    md5 = {}
    for line in h.splitlines():
        if line.startswith("@SQ"):
            d = dict(x.split(":", 1) for x in line.split("\t")[1:] if ":" in x)
            md5[d["SN"]] = d.get("M5")
    out = subprocess.run(["samtools", "dict", str(G.REF_FASTA)], capture_output=True, text=True, check=True).stdout
    local = {}
    for line in out.splitlines():
        if line.startswith("@SQ"):
            d = dict(x.split(":", 1) for x in line.split("\t")[1:] if ":" in x)
            local[d["SN"]] = d.get("M5")
    return {"cram": url, "names_lengths_match": ok, "md5_match": md5 == local, "n_contigs": len(sq),
            "genome_bp": int(sum(ref.values())), "nuclear_bp": int(sum(v for k, v in ref.items() if k in G.NUCLEAR))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pla-sieve", action="store_true", help="also build splitmix64 mmap PLA-index sieve")
    ap.add_argument("--pla-only", action="store_true", help="only build the PLA-index sieve without touching kset/panel")
    ap.add_argument("--pla-out", type=Path, default=None)
    ap.add_argument("--human-decoy", type=Path, default=None)
    a = ap.parse_args()
    if a.pla_only:
        out = {"pla_sieve": build_pla_sieve(out_path=a.pla_out, human_decoy_fasta=a.human_decoy)}
        print(json.dumps(out, indent=1))
        return 0
    G.build()
    out = {"kset": build_kset(), "panel": build_panel()}
    if a.pla_sieve:
        out["pla_sieve"] = build_pla_sieve(out_path=a.pla_out, human_decoy_fasta=a.human_decoy)
    import pandas as pd
    t = pd.read_parquet(Path(__file__).parent / "manifests" / "pf8_truth_200.parquet", columns=["cram_url"])
    out["reference"] = check_cram_header(t.cram_url.iloc[0])
    g = G.load()
    out["reference"]["core_bp"] = int(g.core.sum())
    (G.REFDIR / "prepare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
