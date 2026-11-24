"""Warm-state query engine over a cohort FocaloopState (no network: sockets are disabled in-process).

    python query.py --cohort $W/cohort_2000 all --out results/gate2_queries.json
    python query.py --cohort $W/cohort_2000 fst --a AS-SE-E --b AF-W

Queries read only the warm matrices built by state_store.build_warm (WSAF sketch at 233k Pf8 core sites,
GC-normalised 1 kb depth ratios, called variants, CNV / deletion calls, per-sample telemetry + metadata).
"""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import genome as G
from . import prepare as P
from . import state_store as SS
from . import ust_simplitigs as UST

KNOWN = {"crt": "PF3D7_0709000", "dhfr": "PF3D7_0417200", "dhps": "PF3D7_0810800", "kelch13": "PF3D7_1343700",
         "aat1": "PF3D7_0629500", "pfs47": "PF3D7_1346800", "mdr1": "PF3D7_0523000", "gch1": "PF3D7_1224000"}
SEA = ["Cambodia", "Vietnam", "Laos", "Thailand", "Myanmar"]
HORN = ["Ethiopia", "Sudan", "Eritrea", "Somalia", "Djibouti"]
SOUTH_AM = ["Peru", "Colombia", "Brazil", "Venezuela", "French Guiana", "Ecuador"]


def no_network():
    """any socket creation from here on raises: a query cannot pull bytes (strace confirms zero network calls)."""
    class Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise RuntimeError("network disabled in query engine")

    def blocked(*a, **k):
        raise RuntimeError("network disabled in query engine")
    socket.socket = Blocked
    socket.create_connection = blocked


class Warm:
    def __init__(self, cohort, load_dothash: bool = False, load_esscolor: bool = False):
        import hashlib
        w = Path(cohort) / "warm"
        t0 = time.time()
        self.cohort = Path(cohort)
        self.samples = pd.read_parquet(w / "samples.parquet")
        self.wsaf = np.load(w / "wsaf_u8.npy", mmap_mode="r")
        self.dp = np.load(w / "dp_u8.npy", mmap_mode="r")
        self.ratio = np.load(w / "ratio_f16.npy", mmap_mode="r")
        if len(self.samples) != self.wsaf.shape[0] or len(self.samples) != self.dp.shape[0]:
            raise ValueError(
                f"Corrupt warm state in {w}: samples={len(self.samples)}, wsaf={self.wsaf.shape}, dp={self.dp.shape}"
            )
        self.cnv = pd.read_parquet(w / "cnv.parquet")
        self.calls_path = w / "calls.parquet"
        bman = json.loads((w / "bundle_manifest.json").read_text()) if (w / "bundle_manifest.json").exists() else {}

        if (w / "panel.npz").exists():
            if bman.get("panel_sha256"):
                actual = hashlib.sha256((w / "panel.npz").read_bytes()).hexdigest()
                if actual != bman["panel_sha256"]:
                    raise ValueError(f"Warm panel.npz checksum mismatch in {w}")
            with np.load(w / "panel.npz") as pz:
                self.panel = {k: pz[k] for k in pz.files}
        else:
            self.panel = P.load_panel()

        if self.wsaf.ndim == 2 and self.wsaf.shape[1] > 0 and len(self.panel["goff"]) != self.wsaf.shape[1]:
            raise ValueError(
                f"Panel site count ({len(self.panel['goff'])}) does not match wsaf_u8 width ({self.wsaf.shape[1]})"
            )

        if (w / "genome_meta.npz").exists():
            if bman.get("genome_sha256"):
                actual_g = hashlib.sha256((w / "genome_meta.npz").read_bytes()).hexdigest()
                if actual_g != bman["genome_sha256"]:
                    raise ValueError(f"Warm genome_meta.npz checksum mismatch in {w}")
            with np.load(w / "genome_meta.npz") as gz:
                g_genes = (
                    json.loads((w / "genes_curated.json").read_text())
                    if (w / "genes_curated.json").exists()
                    else {}
                )
                self.g = G.Genome(
                    names=[str(x) for x in gz["names"]],
                    lengths=gz["lengths"],
                    offsets=gz["offsets"],
                    seq=np.empty(0, dtype=np.uint8),
                    core=np.empty(0, dtype=bool),
                    bin_offsets=gz["bin_offsets"],
                    bin_gc=gz["bin_gc"],
                    bin_core=gz["bin_core"],
                    genes=g_genes,
                )
        else:
            self.g = G.load()

        if (w / "genes_all.parquet").exists():
            self.genes = pd.read_parquet(w / "genes_all.parquet")
        else:
            self.genes = pd.read_parquet(G.REFDIR / "genes_all.parquet")

        self.dothash = SS.load_dothash_index(cohort) if (load_dothash and (w / "dothash_i16.npy").exists()) else None
        self.esscolor, self.esscolor_meta = (
            SS.load_esscolor_drm_panel(cohort)
            if (load_esscolor and (w / "esscolor_drm_panel.bin").exists())
            else (None, None)
        )
        self.load_s = time.time() - t0

    def group(self, **kw):
        m = np.ones(len(self.samples), bool)
        s = self.samples
        for k, v in kw.items():
            if v is None:
                continue
            col = s[k]
            if isinstance(v, (list, tuple, set)):
                m &= col.isin(list(v)).values
            elif isinstance(v, tuple) and len(v) == 2:
                m &= col.between(*v).values
            else:
                m &= (col == v).values
        return np.where(m)[0]

    def freqs(self, idx, min_dp=5):
        """population allele frequency per panel site = mean WSAF over samples with depth >= min_dp."""
        w = np.asarray(self.wsaf[idx], dtype=np.float32)
        d = np.asarray(self.dp[idx])
        ok = (w != 255) & (d >= min_dp)
        wf = np.where(ok, w / 200.0, 0.0)
        n = ok.sum(0)
        return wf.sum(0) / np.maximum(n, 1), n

    def annotate(self, goff):
        c, p = self.g.locate(np.array([goff]))
        contig = self.g.names[int(c[0])]
        gg = self.genes[(self.genes.contig == contig)]
        mid = (gg.start + gg.end) / 2
        k = (mid - int(p[0])).abs().idxmin()
        return contig, int(p[0]), gg.loc[k, "gene_id"], gg.loc[k, "name"]


def hudson_fst(p1, n1, p2, n2):
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / np.maximum(n1 - 1, 1) - p2 * (1 - p2) / np.maximum(n2 - 1, 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    return num, den


def q_fst(W, a, b, label, win=20, top=40, min_n=10):
    t0 = time.time()
    ia = W.group(Population=a) if isinstance(a, str) else W.group(Country=a)
    ib = W.group(Population=b) if isinstance(b, str) else W.group(Country=b)
    p1, n1 = W.freqs(ia)
    p2, n2 = W.freqs(ib)
    ok = (n1 >= min_n) & (n2 >= min_n) & ((p1 + p2) > 0) & ((p1 + p2) < 2)
    num, den = hudson_fst(p1[ok], n1[ok], p2[ok], n2[ok])
    goff = W.panel["goff"][ok]
    den_sum = float(den.sum()) if len(den) else 0.0
    genome_fst = float(num.sum() / den_sum) if den_sum > 0 else None
    # windows of `win` consecutive informative sites inside a contig
    c, _ = W.g.locate(goff) if len(goff) else (np.zeros(0, int), np.zeros(0, int))
    rows = []
    for ci in np.unique(c):
        sel = np.where(c == ci)[0]
        for k in range(0, len(sel) - win + 1, win // 2):
            s = sel[k:k + win]
            w_den = float(den[s].sum())
            if w_den > 0:
                rows.append((float(num[s].sum() / w_den), int(goff[s[0]]), int(goff[s[-1]])))
    wdf = pd.DataFrame(rows, columns=["fst", "g0", "g1"]).sort_values("fst", ascending=False)
    wdf["rank"] = np.arange(1, len(wdf) + 1)
    hits = {}
    for name, gid in KNOWN.items():
        gg = W.g.genes[gid]
        x0 = W.g.goff(gg["contig"], gg["start"]) - 25_000
        x1 = W.g.goff(gg["contig"], gg["end"]) + 25_000
        near = wdf[(wdf.g1 >= x0) & (wdf.g0 <= x1)]
        if len(near):
            best = near.iloc[0]
            hits[name] = dict(best_window_fst=float(best.fst), rank=int(best["rank"]),
                              percentile=float(1 - best["rank"] / len(wdf)))
    peaks = []
    for _, r in wdf.head(top).iterrows():
        contig, pos, gid, nm = W.annotate((r.g0 + r.g1) // 2)
        peaks.append(dict(contig=contig, pos=pos, fst=float(r.fst), nearest_gene=gid, name=nm))
    return dict(query=label, group_a=a, group_b=b, n_a=int(len(ia)), n_b=int(len(ib)), sites=int(ok.sum()),
                windows=int(len(wdf)), genome_fst=genome_fst, known_loci=hits, top_peaks=peaks[:15],
                seconds=time.time() - t0)


def _aa_token_flag(series, alt):
    if series is None:
        return np.nan
    target = str(alt).strip().upper()
    out = np.full(len(series), np.nan, dtype=float)
    for i, val in enumerate(series):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        sv = str(val).strip()
        if sv in ("", "-", "None", "none", "nan", "NaN", "*", "!"):
            continue
        toks = {t.strip().upper() for t in sv.replace("*", "").replace("!", "").split(",") if t.strip()}
        if not toks or toks <= {"NONE", "-"}:
            continue
        out[i] = float(target in toks)
    return out


def _k13_flags(s):
    raw = s["k13_ns"] if "k13_ns" in s.columns else (
        s["aa_k13_ns"] if "aa_k13_ns" in s.columns else pd.Series([None] * len(s), index=s.index))
    any_mut = np.full(len(s), np.nan, dtype=float)
    c580y = np.full(len(s), np.nan, dtype=float)
    for i, val in enumerate(raw):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        sv = str(val).strip()
        if sv in ("", "-", "nan", "NaN"):
            continue
        if sv.lower() == "none":
            any_mut[i] = 0.0
            c580y[i] = 0.0
            continue
        toks = {t.strip().upper() for t in sv.replace("*", "").replace("!", "").split(",") if t.strip()}
        any_mut[i] = float(bool(toks - {"NONE", "-"}))
        c580y[i] = float("C580Y" in toks)
    return any_mut, c580y


def mutant_flags(s):
    out = pd.DataFrame(index=s.index)
    out["k13_any"], out["k13_C580Y"] = _k13_flags(s)
    for col, alt, name in (("aa_crt_76", "T", "crt_K76T"), ("aa_mdr1_86", "Y", "mdr1_N86Y"),
                           ("aa_dhfr_108", "N", "dhfr_S108N"), ("aa_dhfr_51", "I", "dhfr_N51I"),
                           ("aa_dhfr_59", "R", "dhfr_C59R"), ("aa_dhps_437", "G", "dhps_A437G"),
                           ("aa_dhps_540", "E", "dhps_K540E"), ("aa_dhps_581", "G", "dhps_A581G")):
        out[name] = _aa_token_flag(s.get(col), alt)
    out["dhfr_triple"] = out[["dhfr_S108N", "dhfr_N51I", "dhfr_C59R"]].min(axis=1, skipna=False)
    return out


def cnv_matrix(W):
    c = W.cnv.pivot_table(index="sample", columns="locus", values="call", aggfunc="first")
    c = c.reindex(W.samples["sample"]).where(lambda x: x >= 0)
    return c


def _mantel_haenszel_or(df, stratum_cols, x_col="k13", y_col="pm23"):
    num = 0.0
    den = 0.0
    used = 0
    for _, g in df.dropna(subset=[*stratum_cols, x_col, y_col]).groupby(list(stratum_cols)):
        n = len(g)
        if n < 2:
            continue
        x1 = g[x_col] == 1
        y1 = g[y_col] == 1
        a = float((x1 & y1).sum())
        b = float((x1 & ~y1).sum())
        c = float((~x1 & y1).sum())
        d = float((~x1 & ~y1).sum())
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        num += (a * d) / n
        den += (b * c) / n
        used += 1
    if used == 0:
        return None, 0
    if den == 0:
        return None, used
    return float(num / den), used


def q_temporal_k13_pm23(W):
    t0 = time.time()
    s = W.samples
    f = mutant_flags(s)
    c = cnv_matrix(W)
    d = pd.DataFrame(dict(country=s.Country.values, year=s.Year.values, k13=f.k13_any.values,
                          c580y=f.k13_C580Y.values, pm23=c["pm2_pm3"].values if "pm2_pm3" in c else np.nan))
    d = d[d.country.isin(SEA)]
    tab = d.groupby("year").agg(n=("k13", "size"), k13_prev=("k13", "mean"), c580y_prev=("c580y", "mean"),
                                pm23_amp_prev=("pm23", "mean")).reset_index()
    both = d.dropna(subset=["k13", "pm23"]).copy()
    co = pd.crosstab(both.k13, both.pm23) if len(both) else pd.DataFrame()
    odds = None
    if co.shape == (2, 2):
        odds = float((co.loc[1, 1] + 0.5) * (co.loc[0, 0] + 0.5) / ((co.loc[1, 0] + 0.5) * (co.loc[0, 1] + 0.5)))
    odds_mh, mh_strata = None, 0
    if len(both):
        both["period"] = (pd.to_numeric(both["year"], errors="coerce") // 2) * 2
        odds_mh, mh_strata = _mantel_haenszel_or(both, ("country", "period"))
    return dict(query="temporal kelch13 + plasmepsin 2/3 co-evolution (Greater Mekong)",
                by_year=tab.round(4).to_dict("records"),
                pm23_given_k13=float(both[both.k13 == 1].pm23.mean()) if len(both[both.k13 == 1]) else None,
                pm23_given_wt=float(both[both.k13 == 0].pm23.mean()) if len(both[both.k13 == 0]) else None,
                odds_ratio=odds, odds_ratio_mh=odds_mh, mh_strata_used=mh_strata,
                seconds=time.time() - t0)


def q_hrp(W):
    t0 = time.time()
    s = W.samples
    c = cnv_matrix(W)
    d = pd.DataFrame(dict(country=s.Country.values, pop=s.Population.values,
                          hrp2=c["hrp2"].values if "hrp2" in c else np.nan,
                          hrp3=c["hrp3"].values if "hrp3" in c else np.nan))
    d["both"] = d[["hrp2", "hrp3"]].min(axis=1, skipna=False)
    by = d.groupby("country").agg(n=("hrp2", "size"), hrp2_del=("hrp2", "mean"), hrp3_del=("hrp3", "mean"),
                                  hrp2_hrp3_del=("both", "mean")).reset_index()
    focus = by[by.country.isin(HORN + SOUTH_AM)]
    return dict(query="hrp2 / hrp3 deletion prevalence (Horn of Africa, South America, all countries)",
                focus=focus.round(4).to_dict("records"), all=by.round(4).to_dict("records"), seconds=time.time() - t0)


def q_fws(W):
    t0 = time.time()
    s = W.samples
    d = s[["Population", "Country", "fws"]].copy()
    d["poly"] = (d.fws < 0.95).astype(float)
    by = d.groupby("Population").agg(n=("fws", "size"), fws_median=("fws", "median"), polyclonal=("poly", "mean"))
    return dict(query="regional F_WS transmission-intensity gradient",
                by_population=by.sort_values("polyclonal", ascending=False).round(4).reset_index().to_dict("records"),
                seconds=time.time() - t0)


def q_drug(W):
    t0 = time.time()
    s = W.samples
    f = mutant_flags(s)
    f["Population"] = s.Population.values
    c = cnv_matrix(W)
    for k in ("crt", "mdr1", "gch1", "pm2_pm3"):
        if k in c:
            f[f"{k}_amp"] = c[k].values
    by = f.groupby("Population").mean(numeric_only=True)
    return dict(query="drug-resistance marker and CNV prevalence by population", by_population=by.round(4).reset_index().to_dict("records"),
                seconds=time.time() - t0)


def q_poly_contrast(W):
    """polyclonal vs monoclonal infections inside the same populations: F_WS split and resistance enrichment."""
    t0 = time.time()
    s = W.samples
    f = mutant_flags(s)
    d = pd.concat([s[["Population", "fws"]].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    d["clonality"] = np.where(d.fws < 0.95, "polyclonal", "monoclonal")
    by = d.groupby(["Population", "clonality"]).agg(n=("fws", "size"), fws_median=("fws", "median"),
                                                    crt_K76T=("crt_K76T", "mean"), dhfr_triple=("dhfr_triple", "mean"),
                                                    k13_any=("k13_any", "mean")).reset_index()
    return dict(query="polyclonal vs monoclonal contrast within populations", table=by.round(4).to_dict("records"),
                seconds=time.time() - t0)


def q_dothash_knn(W, sample_or_idx=0, k_top=5, n_threads=1):
    """Zero-network 1-vs-N DotHash kNN similarity query over cohort hypervectors."""
    t0 = time.time()
    if W.dothash is None:
        return {"query": "dothash_knn", "error": "dothash index not loaded", "seconds": time.time() - t0}
    if isinstance(sample_or_idx, str):
        matches = np.nonzero(W.samples["sample"].astype(str).values == sample_or_idx)[0]
        if len(matches) == 0:
            return {"query": "dothash_knn", "error": f"unknown sample {sample_or_idx}", "seconds": time.time() - t0}
        q_idx = int(matches[0])
    else:
        q_idx = int(sample_or_idx)
        if q_idx < 0 or q_idx >= len(W.samples):
            return {"query": "dothash_knn", "error": f"query index {q_idx} out of range (n={len(W.samples)})", "seconds": time.time() - t0}
    eff_k = max(0, int(k_top))
    q_hv = W.dothash.matrix[q_idx]
    q_card = float(W.dothash.cardinalities[q_idx])
    top_idx, top_sim = W.dothash.knn_1_vs_n(q_hv, k_top=eff_k + 1, query_card=q_card, n_threads=n_threads)
    hits = []
    for idx_i, sim in zip(top_idx, top_sim):
        ii = int(idx_i)
        if ii < 0 or ii == q_idx or ii >= len(W.samples):
            continue
        row = W.samples.iloc[ii]
        hits.append({
            "sample": str(row.get("sample", ii)),
            "population": str(row.get("Population", "")) if "Population" in row else None,
            "country": str(row.get("Country", "")) if "Country" in row else None,
            "jaccard_est": float(sim),
        })
        if len(hits) >= eff_k:
            break
    return {
        "query": "dothash_knn",
        "query_sample": str(W.samples.iloc[q_idx].get("sample", q_idx)),
        "query_population": str(W.samples.iloc[q_idx].get("Population", "")) if "Population" in W.samples.columns else None,
        "k_top": int(k_top),
        "n_threads": int(n_threads),
        "ram_bytes": int(W.dothash.ram_bytes),
        "neighbors": hits,
        "seconds": time.time() - t0,
    }


def q_esscolor_drm(W, hap_idx=0):
    """Zero-network ESSColor DRM haplotype k-mer reconstruction & size summary."""
    t0 = time.time()
    if W.esscolor is None:
        return {"query": "esscolor_drm", "error": "esscolor panel not loaded", "seconds": time.time() - t0}
    kmers = UST.reconstruct_esscolor_haplotype_kmers(W.esscolor, int(hap_idx))
    sigs = (W.esscolor_meta or {}).get("signatures", [])
    return {
        "query": "esscolor_drm",
        "hap_idx": int(hap_idx),
        "n_haplotypes": int(W.esscolor.n_haplotypes),
        "n_backbone_kmers": int(W.esscolor.n_backbone_kmers),
        "reconstructed_kmers": int(len(kmers)),
        "signature": sigs[int(hap_idx)] if 0 <= int(hap_idx) < len(sigs) else None,
        "compressed_panel_bytes": int(len(W.esscolor.packed_zstd_blob)),
        "seconds": time.time() - t0,
    }


def run_all(cohort, use_dothash_knn: bool = False, use_esscolor_panel: bool = False, threads: int = 1):
    no_network()
    t0 = time.time()
    W = Warm(cohort, load_dothash=use_dothash_knn, load_esscolor=use_esscolor_panel)
    out = dict(cohort=str(cohort), n_samples=int(len(W.samples)), load_s=W.load_s, queries=[])
    out["queries"].append(q_fst(W, "AS-SE-E", "AF-W", "genome-wide F_ST: eastern SE Asia vs West Africa"))
    out["queries"].append(q_temporal_k13_pm23(W))
    out["queries"].append(q_hrp(W))
    out["queries"].append(q_fws(W))
    out["queries"].append(q_drug(W))
    out["queries"].append(q_fst(W, "AF-E", "AF-W", "genome-wide F_ST: East vs West Africa"))
    out["queries"].append(q_poly_contrast(W))
    if use_dothash_knn and len(W.samples):
        out["queries"].append(q_dothash_knn(W, 0, k_top=5, n_threads=threads))
    if use_esscolor_panel and W.esscolor is not None:
        out["queries"].append(q_esscolor_drm(W, 0))
    out["total_s"] = time.time() - t0
    out["max_query_s"] = max(q["seconds"] for q in out["queries"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("cmd", choices=["all", "fst", "knn"])
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--out")
    ap.add_argument("--dothash-knn", action="store_true")
    ap.add_argument("--esscolor-panel", action="store_true")
    ap.add_argument("--knn-sample", default="0")
    ap.add_argument("--k-top", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "all":
        res = run_all(a.cohort, use_dothash_knn=a.dothash_knn, use_esscolor_panel=a.esscolor_panel, threads=a.threads)
    elif a.cmd == "knn":
        no_network()
        q_ref = int(a.knn_sample) if str(a.knn_sample).isdigit() else str(a.knn_sample)
        res = q_dothash_knn(Warm(a.cohort, load_dothash=True), q_ref, k_top=a.k_top, n_threads=a.threads)
    else:
        no_network()
        res = q_fst(Warm(a.cohort), a.a, a.b, f"F_ST {a.a} vs {a.b}")
    js = json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if a.out:
        Path(a.out).write_text(js)
    print(js[:4000])


if __name__ == "__main__":
    main()
