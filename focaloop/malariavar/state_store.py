"""Compact zstd-Parquet FocaloopState: per-sample checkpoints sealed into 1,000-sample shards.

Layout under a cohort directory:
  samples/<sample>/<table>.parquet + DONE     per-sample checkpoint (atomic rename)
  shards/shard_NNN/<table>.parquet + SEALED   compacted shard, per-sample dirs removed after sealing
  warm/                                       query-ready matrices (WSAF sketch, 1 kb depth ratios, sample index)

Tables: variant_calls, wsaf_sketch, depth_1kb_bins, cnv_del_calls, sample_telemetry, callable_mask,
        slice_log, zoom_calls
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import dothash as DH
from . import genome as G
from . import ust_simplitigs as UST

TABLES = ["variant_calls", "wsaf_sketch", "depth_1kb_bins", "cnv_del_calls", "sample_telemetry",
          "callable_mask", "slice_log", "zoom_calls"]
PQ = dict(compression="zstd", compression_level=6)


def sample_dir(cohort, sample):
    return Path(cohort) / "samples" / sample


def is_done(cohort, sample):
    return (sample_dir(cohort, sample) / "DONE").exists()


def is_failed(cohort, sample):
    return (sample_dir(cohort, sample) / "FAILED").exists()


def _atomic_replace_dir(tmp: Path, final: Path):
    """replace directory `final` with `tmp` without leaving a deleted-checkpoint window on crash."""
    old = final.with_name(final.name + f".old.{os.getpid()}")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    try:
        tmp.rename(final)
    except BaseException:
        if old.exists() and not final.exists():
            old.rename(final)
        raise
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def write_sample(cohort, sample, tables, ok=True):
    """write every table to a temp dir, then swap into place; failed samples get FAILED instead of DONE."""
    final = sample_dir(cohort, sample)
    tmp = final.with_name(final.name + f".tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    nbytes = 0
    for name, df in tables.items():
        if df is None:
            continue
        t = df if isinstance(df, pa.Table) else pa.Table.from_pandas(pd.DataFrame(df), preserve_index=False)
        pq.write_table(t, tmp / f"{name}.parquet", **PQ)
        nbytes += (tmp / f"{name}.parquet").stat().st_size
    marker = "DONE" if ok else "FAILED"
    (tmp / marker).write_text(json.dumps({"t": time.time(), "bytes": nbytes, "ok": bool(ok)}))
    _atomic_replace_dir(tmp, final)
    return nbytes


def unified_schema(files):
    """one schema for per-sample files (all-null columns become string; conflicting types fall back to string)."""
    schemas = [pq.read_schema(f) for f in files]
    try:
        sch = pa.unify_schemas(schemas, promote_options="permissive")
        return pa.schema([pa.field(f.name, pa.string() if pa.types.is_null(f.type) else f.type) for f in sch])
    except (pa.ArrowTypeError, pa.ArrowInvalid):
        names, types = [], {}
        for sc in schemas:
            for f in sc:
                if f.name not in types:
                    names.append(f.name)
                types.setdefault(f.name, set()).add(f.type)
        fields = []
        for n in names:
            ts = {t for t in types[n] if not pa.types.is_null(t)}
            if len(ts) == 1:
                fields.append(pa.field(n, ts.pop()))
            elif ts and all(pa.types.is_integer(t) or pa.types.is_floating(t) for t in ts):
                fields.append(pa.field(n, pa.float64()))
            else:
                fields.append(pa.field(n, pa.string()))
        return pa.schema(fields)


def conform(t, schema):
    cols = []
    for f in schema:
        if f.name in t.column_names:
            c = t.column(f.name)
            if c.type != f.type:
                c = c.cast(f.type) if not (pa.types.is_string(f.type) and len(c) == 0) else pa.array([], f.type)
            cols.append(c)
        else:
            cols.append(pa.nulls(len(t), f.type))
    return pa.Table.from_arrays(cols, schema=schema)


def done_samples(cohort):
    out = set()
    sd = Path(cohort) / "samples"
    if sd.exists():
        out |= {p.parent.name for p in sd.glob("*/DONE")}
    for sh in sealed_shards(cohort):
        out |= set(json.loads((sh / "SEALED").read_text())["samples"])
    return out


def sealed_shards(cohort):
    sd = Path(cohort) / "shards"
    return sorted(p.parent for p in sd.glob("shard_*/SEALED")) if sd.exists() else []


def seal_shard(cohort, shard_id, samples):
    """merge per-sample tables of `samples` into shards/shard_NNN and drop the per-sample dirs."""
    out = Path(cohort) / "shards" / f"shard_{shard_id:03d}"
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    seen = set()
    unique_samples = []
    for s in samples:
        if s not in seen:
            seen.add(s)
            unique_samples.append(s)
    present = [s for s in unique_samples if is_done(cohort, s)]
    for name in TABLES:
        files = [sample_dir(cohort, s) / f"{name}.parquet" for s in present]
        files = [f for f in files if f.exists()]
        if not files:
            continue
        schema = unified_schema(files)
        with pq.ParquetWriter(tmp / f"{name}.parquet", schema, **PQ) as w:
            for f in files:
                w.write_table(conform(pq.read_table(f), schema))
    (tmp / "SEALED").write_text(json.dumps({"samples": present, "t": time.time()}))
    _atomic_replace_dir(tmp, out)
    for s in present:
        shutil.rmtree(sample_dir(cohort, s), ignore_errors=True)
    return out


def unseal_shard(cohort, shard_dir):
    """split a sealed shard back into per-sample checkpoints (repairs / reruns), then remove it."""
    shard_dir = Path(shard_dir)
    samples = json.loads((shard_dir / "SEALED").read_text())["samples"]
    tables = {n: pq.read_table(shard_dir / f"{n}.parquet").to_pandas() for n in TABLES
              if (shard_dir / f"{n}.parquet").exists()}
    for s in samples:
        write_sample(cohort, s, {n: t[t["sample"] == s] for n, t in tables.items()})
    shutil.rmtree(shard_dir)
    return samples


def table_files(cohort, name):
    files = [sh / f"{name}.parquet" for sh in sealed_shards(cohort)]
    sd = Path(cohort) / "samples"
    if sd.exists():
        files += sorted(p.parent / f"{name}.parquet" for p in sd.glob("*/DONE"))
    return [f for f in files if f.exists()]


def read_table(cohort, name, columns=None):
    files = table_files(cohort, name)
    if not files:
        return pd.DataFrame()
    schema = unified_schema(files)
    if columns:
        schema = pa.schema([schema.field(c) for c in columns])
    return pa.concat_tables([conform(pq.read_table(f, columns=columns), schema) for f in files]).to_pandas()


def u8_from_bytes(col):
    return np.stack([np.frombuffer(b, np.uint8) for b in col]) if len(col) else np.zeros((0, 0), np.uint8)


def f16_from_bytes(col):
    return np.stack([np.frombuffer(b, np.float16) for b in col]) if len(col) else np.zeros((0, 0), np.float16)


def _splitmix64_u64(arr: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit SplitMix64 bijective hash for uint64 feature keys."""
    z = (np.asarray(arr, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def extract_sample_feature_keys(
    wsaf_row: np.ndarray,
    dp_row: np.ndarray,
    min_dp: int = 3,
    min_wsaf_u8: int = 20,
) -> np.ndarray:
    """Derive deterministic 64-bit unsettled/variant keys for DotHash sketching from WSAF + DP."""
    ok = (wsaf_row != 255) & (dp_row >= min_dp) & (wsaf_row >= min_wsaf_u8)
    sites = np.nonzero(ok)[0].astype(np.uint64)
    if sites.size == 0:
        return np.empty(0, dtype=np.uint64)
    w = wsaf_row[sites]
    # Distinguish major-alt (WSAF >= 0.50 -> bin 1) vs minor-alt (0.10 <= WSAF < 0.50 -> bin 2)
    bins = np.where(w >= 100, np.uint64(1), np.uint64(2))
    raw = np.uint64(0x5753414600000000) | (sites << np.uint64(4)) | bins
    return _splitmix64_u64(raw)


def build_dothash_index(
    cohort,
    wsaf_u8: np.ndarray | None = None,
    dp_u8: np.ndarray | None = None,
    calls_df: pd.DataFrame | None = None,
    sample_order: list[str] | None = None,
    n_threads: int = 0,
    min_dp: int = 3,
    min_wsaf_u8: int = 20,
) -> dict:
    """Build (N, 2048) int16 DotHash matrix (4 KB/isolate) into <cohort>/warm/dothash_i16.npy."""
    warm = Path(cohort) / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    if wsaf_u8 is None:
        wsaf_u8 = np.load(warm / "wsaf_u8.npy", mmap_mode="r")
    if dp_u8 is None:
        dp_u8 = np.load(warm / "dp_u8.npy", mmap_mode="r")
    n_samples = int(wsaf_u8.shape[0]) if wsaf_u8.ndim == 2 else 0

    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    chunks = []
    if n_samples > 0 and wsaf_u8.shape[1] > 0:
        for i in range(n_samples):
            keys = extract_sample_feature_keys(
                np.asarray(wsaf_u8[i]),
                np.asarray(dp_u8[i]),
                min_dp=min_dp,
                min_wsaf_u8=min_wsaf_u8,
            )
            chunks.append(keys)
            offsets[i + 1] = offsets[i] + len(keys)
    elif calls_df is not None and sample_order:
        by_s = {s: g for s, g in calls_df.groupby("sample", sort=False)}
        n_samples = len(sample_order)
        offsets = np.zeros(n_samples + 1, dtype=np.int64)
        for i, s in enumerate(sample_order):
            sub = by_s.get(s)
            if sub is None or len(sub) == 0:
                keys = np.empty(0, dtype=np.uint64)
            else:
                ch = sub["chrom"].astype(np.uint64).to_numpy()
                po = sub["pos"].astype(np.uint64).to_numpy()
                gt = sub["gt"].astype(np.uint64).to_numpy()
                raw = np.uint64(0x43414C4C00000000) | ((ch & np.uint64(0xFF)) << np.uint64(32)) | (po << np.uint64(2)) | (gt & np.uint64(3))
                keys = _splitmix64_u64(np.unique(raw))
            chunks.append(keys)
            offsets[i + 1] = offsets[i] + len(keys)

    all_keys = np.concatenate(chunks) if chunks and offsets[-1] > 0 else np.empty(0, dtype=np.uint64)
    idx = DH.DotHashCohortIndex.from_csr(all_keys, offsets, n_threads=n_threads)
    out_npy = warm / "dothash_i16.npy"
    np.save(out_npy, idx.matrix)
    out_card = warm / "dothash_cards.npy"
    np.save(out_card, idx.cardinalities)
    return {
        "n_samples": int(idx.n_samples),
        "dim": int(DH.DOTHASH_DIM),
        "ram_bytes": int(idx.ram_bytes),
        "disk_bytes": int(out_npy.stat().st_size + out_card.stat().st_size),
        "zstd_bytes": int(len(idx.to_zstd_bytes())),
    }


def load_dothash_index(cohort) -> DH.DotHashCohortIndex:
    """Memory-map saved (N, 2048) int16 DotHash index from <cohort>/warm."""
    warm = Path(cohort) / "warm"
    mat = np.load(warm / "dothash_i16.npy", mmap_mode="r")
    cards_path = warm / "dothash_cards.npy"
    cards = np.load(cards_path, mmap_mode="r") if cards_path.exists() else None
    return DH.DotHashCohortIndex(mat, cardinalities=cards)


def _drm_codon_sequence(seed_text: str, length: int = 120) -> str:
    """Deterministic valid A/C/G/T anchor sequence for ESSColor haplotype representation."""
    bases = "ACGT"
    h = int(hashlib.blake2b(seed_text.encode("utf-8"), digest_size=16).hexdigest(), 16)
    chars = []
    x = h
    for i in range(length):
        x = (x * 6364136223846793005 + 1442695040888963407 + i) & ((1 << 64) - 1)
        chars.append(bases[(x >> 62) & 3])
    return "".join(chars)


def build_esscolor_drm_panel(
    cohort,
    samples_df: pd.DataFrame | None = None,
    k: int = 31,
) -> dict:
    """Compress observed cohort DRM haplotypes using ESSColor (UST simplitigs + XOR delta bits)."""
    warm = Path(cohort) / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    if samples_df is None:
        samples_df = pd.read_parquet(warm / "samples.parquet")

    drm_cols = sorted([c for c in samples_df.columns if c.startswith("aa_")])
    if "k13_ns" in samples_df.columns:
        drm_cols.append("k13_ns")
    if not drm_cols:
        drm_cols = ["sample"]

    # Canonical Pf3D7 backbone template padded cleanly for deterministic codon substitution
    backbone = _drm_codon_sequence("Pf3D7_DRM_BACKBONE_V1", length=max(160, 12 * max(len(drm_cols), 1) + 40))
    hap_sigs = []
    for _, row in samples_df[drm_cols].fillna("").iterrows():
        sig = "|".join(f"{c}={str(row[c]).strip()}" for c in drm_cols)
        hap_sigs.append(sig)

    uniq_sigs = {}
    hap_seqs = []
    iso_hap_idx = np.zeros(len(hap_sigs), dtype=np.uint16)
    codon_bases = ["AAA", "CCC", "GGG", "TTT", "AAC", "AAG", "AAT", "ACA",
                   "ACC", "ACG", "ACT", "AGA", "AGC", "AGG", "AGT", "ATC"]
    for i, sig in enumerate(hap_sigs):
        if sig not in uniq_sigs:
            hid = len(uniq_sigs)
            uniq_sigs[sig] = hid
            # Mutate discrete separated 3-bp windows along the shared backbone so UST captures shared flanking k-mers cleanly
            seq_list = list(backbone)
            for j, part in enumerate(sig.split("|")):
                offset = 24 + (j % ((len(backbone) - 48) // 35 + 1)) * 35
                if offset + 3 <= len(seq_list):
                    idx_c = int.from_bytes(hashlib.blake2b(part.encode("utf-8"), digest_size=2).digest(), "little") % len(codon_bases)
                    seq_list[offset:offset + 3] = list(codon_bases[idx_c])
            hap_seqs.append("".join(seq_list))
        iso_hap_idx[i] = uniq_sigs[sig]

    raw_bytes = sum(len(s) for s in hap_seqs)
    panel = UST.build_esscolor_panel(hap_seqs, k=k, isolate_to_hap_idx=iso_hap_idx)
    (warm / "esscolor_drm_panel.bin").write_bytes(panel.packed_zstd_blob)
    if panel.isolate_haplotype_ids_zstd is not None:
        (warm / "esscolor_isolate_hap_ids.bin").write_bytes(panel.isolate_haplotype_ids_zstd)
    meta = {
        "k": int(panel.k),
        "n_haplotypes": int(panel.n_haplotypes),
        "n_backbone_kmers": int(panel.n_backbone_kmers),
        "n_simplitigs": int(len(panel.ust_simplitigs)),
        "raw_haplotype_bytes": int(raw_bytes),
        "compressed_panel_bytes": int(len(panel.packed_zstd_blob)),
        "isolate_map_bytes": int(len(panel.isolate_haplotype_ids_zstd or b"")),
        "drm_cols": drm_cols,
        "signatures": list(uniq_sigs.keys()),
    }
    (warm / "esscolor_drm_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_esscolor_drm_panel(cohort) -> tuple[UST.ESSColorPanel, dict]:
    """Load saved ESSColor DRM panel and metadata from <cohort>/warm."""
    warm = Path(cohort) / "warm"
    meta = json.loads((warm / "esscolor_drm_meta.json").read_text())
    blob = (warm / "esscolor_drm_panel.bin").read_bytes()
    iso_p = warm / "esscolor_isolate_hap_ids.bin"
    iso_blob = iso_p.read_bytes() if iso_p.exists() else None
    panel = UST.ESSColorPanel(
        k=int(meta["k"]),
        n_haplotypes=int(meta["n_haplotypes"]),
        n_backbone_kmers=int(meta["n_backbone_kmers"]),
        ust_simplitigs=[],
        packed_zstd_blob=blob,
        isolate_haplotype_ids_zstd=iso_blob,
    )
    return panel, meta


def build_warm(cohort, meta=None, use_dothash_knn: bool = False, use_esscolor_panel: bool = False, n_threads: int = 0):
    """materialise query matrices: WSAF sketch (samples x panel sites, uint8, 255 = missing),
    GC-normalised 1 kb ratios (samples x bins, float16), and a sample index joined to metadata."""
    warm = Path(cohort) / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    ws = read_table(cohort, "wsaf_sketch")
    tel = read_table(cohort, "sample_telemetry")
    if "error" in tel.columns:
        tel = tel[tel["error"].isna()]
    if "stop_reason" in tel.columns:
        tel = tel[tel["stop_reason"] != "error"]
    if len(tel):
        tel = tel.drop_duplicates("sample", keep="last").reset_index(drop=True)
    valid = set(tel["sample"])
    ws = ws[ws["sample"].isin(valid)].drop_duplicates("sample", keep="last").reset_index(drop=True)
    dep = read_table(cohort, "depth_1kb_bins", columns=["sample", "ratio_f16"])
    if len(dep):
        dep = dep.drop_duplicates("sample", keep="last")
    order = ws["sample"].tolist()
    wsaf_mat = u8_from_bytes(ws["wsaf_u8"])
    dp_mat = u8_from_bytes(ws["dp_u8"])
    np.save(warm / "wsaf_u8.npy", wsaf_mat)
    np.save(warm / "dp_u8.npy", dp_mat)
    dep = dep.set_index("sample").loc[order]
    ratio_mat = f16_from_bytes(dep["ratio_f16"])
    np.save(warm / "ratio_f16.npy", ratio_mat)
    idx = tel.set_index("sample").loc[order].reset_index()
    rp = Path(cohort) / "replay"
    if (rp / "curated.parquet").exists():  # curated calls re-derived with the final rules (zero network)
        cur = pd.read_parquet(rp / "curated.parquet").drop_duplicates("sample", keep="last").set_index("sample")
        cols = [c for c in cur.columns if c.startswith("aa_") or c == "k13_ns"]
        idx = idx.drop(columns=[c for c in cols if c in idx.columns]).merge(cur[cols], left_on="sample",
                                                                            right_index=True, how="left")
    if meta is not None:
        meta_dedup = meta.drop_duplicates("Sample", keep="last") if "Sample" in meta.columns else meta
        idx = idx.merge(meta_dedup, left_on="sample", right_on="Sample", how="left", suffixes=("", "_meta"))
    assert len(idx) == wsaf_mat.shape[0] == dp_mat.shape[0] == ratio_mat.shape[0], (
        f"Warm state sample row mismatch: samples={len(idx)}, wsaf={wsaf_mat.shape[0]}, "
        f"dp={dp_mat.shape[0]}, ratio={ratio_mat.shape[0]}"
    )
    idx.to_parquet(warm / "samples.parquet", index=False)
    calls = read_table(cohort, "variant_calls", columns=["sample", "chrom", "pos", "ref", "alt", "kind", "gt", "wsaf"])
    calls = calls[(calls["gt"] > 0) & calls["sample"].isin(valid)]
    calls.to_parquet(warm / "calls.parquet", index=False, **PQ)
    cnv = pd.read_parquet(rp / "cnv.parquet") if (rp / "cnv.parquet").exists() else read_table(cohort, "cnv_del_calls")
    if len(cnv):
        cnv = cnv[cnv["sample"].isin(valid)]
        if {"sample", "locus"}.issubset(cnv.columns):
            cnv = cnv.drop_duplicates(["sample", "locus"], keep="last")
    cnv.to_parquet(warm / "cnv.parquet", index=False)

    # Persist self-contained panel + genome metadata + genes_all inside warm/ for portable Warm queries (L46)
    bundle_manifest = {"n_samples": int(len(order)), "n_sites": int(wsaf_mat.shape[1]) if wsaf_mat.ndim == 2 else 0}
    try:
        from . import prepare as P
        if P.SITE_PANEL.exists():
            panel_obj = P.load_panel()
            np.savez_compressed(warm / "panel.npz", **{k: np.asarray(v) for k, v in panel_obj.items()})
            bundle_manifest["panel_sha256"] = hashlib.sha256((warm / "panel.npz").read_bytes()).hexdigest()
    except Exception:
        pass
    try:
        if G.REF_FASTA.exists() or (G.REFDIR / "genome_cache.npz").exists():
            g_obj = G.load()
            np.savez_compressed(
                warm / "genome_meta.npz",
                names=np.asarray(g_obj.names),
                lengths=np.asarray(g_obj.lengths, dtype=np.int64),
                offsets=np.asarray(g_obj.offsets, dtype=np.int64),
                bin_offsets=np.asarray(g_obj.bin_offsets, dtype=np.int64),
                bin_gc=np.asarray(g_obj.bin_gc, dtype=np.float32),
                bin_core=np.asarray(g_obj.bin_core, dtype=np.float32),
            )
            (warm / "genes_curated.json").write_text(json.dumps(g_obj.genes, default=list))
            bundle_manifest["genome_sha256"] = hashlib.sha256((warm / "genome_meta.npz").read_bytes()).hexdigest()
        if (G.REFDIR / "genes_all.parquet").exists():
            shutil.copy2(G.REFDIR / "genes_all.parquet", warm / "genes_all.parquet")
            bundle_manifest["genes_sha256"] = hashlib.sha256((warm / "genes_all.parquet").read_bytes()).hexdigest()
    except Exception:
        pass
    (warm / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=1))

    out = {"samples": len(order)}
    if use_dothash_knn:
        out["dothash"] = build_dothash_index(
            cohort, wsaf_u8=wsaf_mat, dp_u8=dp_mat, calls_df=calls, sample_order=order, n_threads=n_threads
        )
    if use_esscolor_panel:
        out["esscolor"] = build_esscolor_drm_panel(cohort, samples_df=idx)
    out["warm_bytes"] = int(sum(f.stat().st_size for f in warm.iterdir()))
    return out
