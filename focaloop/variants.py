"""Variant-modality accumulators: allele probe k-mers, CNV ratios, deletions, F_WS.

Sites (SNPs or indels) come from a TSV ``site_id, target_id, pos (0-based), ref, alt, kind``.
Each allele becomes a k-mer centred on the site (flanks from the target), so a
read supports an allele when it contains that allele's canonical k-mer. Copy
number is effective depth per kb relative to the median of single-copy control
genes; a deletion is called when that ratio stays below ``deletion_ratio``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from .kernels import encode_kmers_stranded

ALLELE_COUNTS_SCHEMA = pa.schema([
    ("sample", pa.string()),
    ("partition", pa.string()),
    ("site_id", pa.string()),
    ("target_id", pa.string()),
    ("kind", pa.string()),  # snp | indel
    ("ref_reads", pa.int64()),
    ("alt_reads", pa.int64()),
])


@dataclass(frozen=True)
class Site:
    site_id: str
    target_id: str
    pos: int
    ref: str
    alt: str
    kind: str


def load_sites(path: str | Path) -> list[Site]:
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    out = []
    for r in df.itertuples():
        kind = getattr(r, "kind", None) or ("snp" if len(r.ref) == len(r.alt) == 1 else "indel")
        out.append(Site(r.site_id, r.target_id, int(r.pos), r.ref.upper(), r.alt.upper(), kind))
    return out


class AlleleProbes:
    def __init__(self, sites: Sequence[Site], target_seqs: Mapping[str, str], k: int = 31) -> None:
        self.k = k
        self.sites = list(sites)
        codes: list[int] = []
        owner: list[tuple[int, int]] = []  # (site index, 0=ref / 1=alt)
        for i, s in enumerate(self.sites):
            seq = target_seqs[s.target_id]
            if seq[s.pos:s.pos + len(s.ref)] != s.ref:
                raise ValueError(f"site {s.site_id}: ref {s.ref!r} does not match target at {s.pos}")
            for a, allele in enumerate((s.ref, s.alt)):
                left_n = (k - len(allele)) // 2
                right_n = k - len(allele) - left_n
                if left_n < 0:
                    raise ValueError(f"site {s.site_id}: allele longer than k")
                left = seq[max(0, s.pos - left_n):s.pos]
                right = seq[s.pos + len(s.ref):s.pos + len(s.ref) + right_n]
                probe = left + allele + right
                if len(probe) != k:
                    raise ValueError(f"site {s.site_id}: too close to target edge for a {k}-mer probe")
                c, ok, _ = encode_kmers_stranded([probe], k)
                codes.append(int(c[0, 0]))
                owner.append((i, a))
        arr = np.array(codes, dtype=np.uint64)
        order = np.argsort(arr)
        self.codes = arr[order]
        self.owner = [owner[j] for j in order]

    def count(self, reads: Sequence[str]) -> np.ndarray:
        """(n_sites, 2) read counts supporting ref / alt."""
        out = np.zeros((len(self.sites), 2), dtype=np.int64)
        if not reads or not self.sites:
            return out
        c, ok, _ = encode_kmers_stranded(reads, self.k)
        if c.shape[1] == 0:
            return out
        hit = np.isin(c, self.codes) & ok
        rows, cols = np.nonzero(hit)
        seen: set[tuple[int, int]] = set()
        for r, col in zip(rows, cols):
            j = int(np.searchsorted(self.codes, c[r, col]))
            si, a = self.owner[j]
            if (int(r), si * 2 + a) in seen:
                continue
            seen.add((int(r), si * 2 + a))
            out[si, a] += 1
        return out


def fws_by_sample(alleles: pd.DataFrame, min_depth: int = 10) -> dict[str, float]:
    """Within-host diversity F_WS = 1 - H_w / H_s (Manske et al. 2012), SNP sites only."""
    df = alleles[alleles["kind"] == "snp"].groupby(["sample", "site_id"], as_index=False)[["ref_reads", "alt_reads"]].sum()
    df["depth"] = df["ref_reads"] + df["alt_reads"]
    df = df[df["depth"] >= min_depth]
    if df.empty:
        return {}
    df["p"] = df["alt_reads"] / df["depth"]
    pop = df.groupby("site_id")["p"].mean()
    hs_site = 2 * pop * (1 - pop)
    out: dict[str, float] = {}
    for s, g in df.groupby("sample"):
        hs = hs_site.reindex(g["site_id"]).to_numpy()
        hw = (2 * g["p"] * (1 - g["p"])).to_numpy()
        keep = hs > 0
        if not keep.any():
            out[s] = 1.0
            continue
        out[s] = float(max(0.0, 1.0 - hw[keep].mean() / hs[keep].mean()))
    return out


def copy_number(eff: pd.DataFrame, lengths: Mapping[str, int], controls: Sequence[str]) -> pd.DataFrame:
    """``eff`` columns sample, group_id, effective_count -> copy ratio vs single-copy control median."""
    rows = []
    for s, g in eff.groupby("sample"):
        depth = {t: c / max(1e-9, lengths.get(t, 1000) / 1000.0) for t, c in zip(g["group_id"], g["effective_count"])}
        ctrl = [depth[c] for c in controls if c in depth and depth[c] > 0]
        base = float(np.median(ctrl)) if ctrl else float("nan")
        for t, d in depth.items():
            ratio = d / base if base and not math.isnan(base) and base > 0 else float("nan")
            rows.append({"sample": s, "group_id": t, "depth_per_kb": d, "copy_ratio": ratio})
    return pd.DataFrame(rows)


def variant_table(alleles: pd.DataFrame, eff: pd.DataFrame, lengths: Mapping[str, int], cfg: Mapping) -> pd.DataFrame:
    """Primary TSV for the variants modality: one row per (sample, feature)."""
    min_alt = int(cfg.get("min_alt_reads", 3))
    rows: list[dict] = []
    if not alleles.empty:
        a = alleles.groupby(["sample", "site_id", "target_id", "kind"], as_index=False)[["ref_reads", "alt_reads"]].sum()
        for r in a.itertuples():
            depth = r.ref_reads + r.alt_reads
            af = r.alt_reads / depth if depth else float("nan")
            call = "no_coverage" if depth == 0 else ("alt" if af >= 0.9 else ("mixed" if r.alt_reads >= min_alt else "ref"))
            rows.append({"sample": r.sample, "feature_type": r.kind, "feature_id": r.site_id, "target_id": r.target_id,
                         "value": af, "ref_reads": r.ref_reads, "alt_reads": r.alt_reads, "call": call})
    cn = copy_number(eff, lengths, cfg.get("single_copy_controls", []))
    cnv_genes = set(cfg.get("cnv_genes", []))
    del_genes = set(cfg.get("deletion_genes", []))
    for r in cn.itertuples():
        if r.group_id in cnv_genes:
            cp = r.copy_ratio
            call = "amplified" if cp >= 1.5 else ("single_copy" if cp >= 0.5 else "low")
            rows.append({"sample": r.sample, "feature_type": "cnv", "feature_id": r.group_id, "target_id": r.group_id,
                         "value": cp, "ref_reads": 0, "alt_reads": 0, "call": call})
        if r.group_id in del_genes:
            call = "deleted" if (r.copy_ratio < 0.1) else "present"
            rows.append({"sample": r.sample, "feature_type": "deletion", "feature_id": r.group_id,
                         "target_id": r.group_id, "value": r.copy_ratio, "ref_reads": 0, "alt_reads": 0, "call": call})
    for s, f in fws_by_sample(alleles, int(cfg.get("fws", {}).get("min_depth", 10)) if isinstance(cfg.get("fws"), dict) else 10).items():
        rows.append({"sample": s, "feature_type": "fws", "feature_id": "F_WS", "target_id": "",
                     "value": f, "ref_reads": 0, "alt_reads": 0, "call": "monoclonal" if f >= 0.95 else "polyclonal"})
    cols = ["sample", "feature_type", "feature_id", "target_id", "value", "ref_reads", "alt_reads", "call"]
    return pd.DataFrame(rows, columns=cols).sort_values(["sample", "feature_type", "feature_id"]).reset_index(drop=True)
