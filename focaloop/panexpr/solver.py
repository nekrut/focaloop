"""Pass-2 backward conditioning, strandedness, gene-level DE and calibrated uncertainty.

* WGS conditioning: per-target P(DNA present) and assembly priors from a WGS run
  (Poisson presence test against background), used to down-weight alleles that
  are not in the sequenced isolate (e.g. hblCDAB absent from ATCC 10987).
* Cohort EM: multi-mapped equivalence classes are re-weighted with abundance
  priors pooled across the cohort (Dirichlet pseudo-counts), inferred library
  strandedness and P(DNA).
* Gene-level DE: PyDESeq2 on group (tx2gene / ortholog) counts, with placement
  ambiguity variance propagated into ``lfc_se_total`` and 95% CIs, plus the
  technical-vs-biological variance decomposition used for stopping.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import itertools
import math
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from ._fast_c import c_em_sparse
from .catalog import Catalog, Target, tximport_gene_level
from .kernels import RunAccumulator
from .state import TARGET_COUNTS_SCHEMA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LN2_SQ = math.log(2.0) ** 2


def _sigmoid(x: float) -> float:
    if x >= 35.0:
        return 1.0
    if x <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# WGS / assembly priors
# ---------------------------------------------------------------------------

def infer_dna_presence(
    wgs_accs: list[RunAccumulator],
    catalog: Catalog,
    em_iters: int = 4,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """P(DNA present) per target, DNA read counts, and assembly prior from WGS evidence.

    Without any WGS run every target gets P(DNA)=1 and the assembly prior is flat.
    """
    targets = catalog.targets
    assemblies = catalog.assemblies
    if not wgs_accs or all(a.n_mapped == 0 for a in wgs_accs):
        flat = {a: 1.0 / len(assemblies) for a in assemblies}
        return dict.fromkeys(targets, 1.0), dict.fromkeys(targets, 0.0), flat
    dna_counts = dict.fromkeys(targets, 0.0)
    for acc in wgs_accs:
        for tid, (c0, c1) in acc.unique_counts.items():
            if tid in dna_counts:
                dna_counts[tid] += c0 + c1
    evidence = {a: 1.0 for a in assemblies}
    for tid, c in dna_counts.items():
        evidence[targets[tid].assembly_id] += c
    for acc in wgs_accs:
        for (ts, _s, lls), cnt in acc.eq_classes.items():
            best = {targets[t].assembly_id for t, ll in zip(ts, lls) if ll >= -0.5 and t in targets}
            if len(best) == 1:
                evidence[next(iter(best))] += cnt
    tot = sum(evidence.values())
    assembly_prior = {a: max(1e-5, v / tot) for a, v in evidence.items()}
    # EM starts from unambiguous evidence: an allele with no uniquely supporting WGS reads
    # only keeps tied reads if no better-supported allele can explain them
    unique_dna = {t: 0.0 for t in targets}
    for acc in wgs_accs:
        for tid, (c0, c1) in acc.unique_counts.items():
            if tid in unique_dna:
                unique_dna[tid] += c0 + c1
    theta = {t: assembly_prior[targets[t].assembly_id] * (unique_dna[t] + 0.2) / max(0.1, targets[t].length / 1000.0) for t in targets}
    for _ in range(max(em_iters, 8)):
        assigned = {t: 0.0 for t in targets}
        for acc in wgs_accs:
            for tid, (c0, c1) in acc.unique_counts.items():
                if tid in assigned:
                    assigned[tid] += c0 + c1
            for (ts, _s, lls), cnt in acc.eq_classes.items():
                raw = [theta.get(t, 1e-6) * math.exp(ll) for t, ll in zip(ts, lls)]
                z = sum(raw) or 1e-12
                for t, rw in zip(ts, raw):
                    if t in assigned:
                        assigned[t] += cnt * rw / z
        dna_counts = assigned
        for t, rec in targets.items():
            kb = max(0.1, rec.length / 1000.0)
            theta[t] = assembly_prior[rec.assembly_id] * (assigned[t] + 0.2) / kb
    # Poisson presence posterior relative to the primary assembly's median coverage
    prim_rates = [dna_counts[t] / max(0.2, r.length / 1000.0) for t, r in targets.items()
                  if r.assembly_id == catalog.primary_assembly and not r.is_decoy]
    median_rate = max(0.5, float(np.median(prim_rates))) if prim_rates else 2.0
    dna_prob: dict[str, float] = {}
    for t, r in targets.items():
        kb = max(0.15, r.length / 1000.0)
        mu1 = max(0.3, median_rate * kb)
        mu0 = max(0.002, 0.01 * mu1)
        sp = min(0.9999, max(1e-5, assembly_prior[r.assembly_id]))
        llr = math.log(sp / (1.0 - sp)) + dna_counts[t] * math.log(mu1 / mu0) - (mu1 - mu0)
        dna_prob[t] = _sigmoid(llr)
    return dna_prob, dna_counts, assembly_prior


def infer_strandedness(acc: RunAccumulator, catalog: Catalog) -> float:
    """P(dUTP-like orientation) for one run from unique protein-coding hits (0.5 = unstranded)."""
    a_sum, b_sum = 1.0, 1.0
    targets = catalog.targets
    for tid, (c0, c1) in acc.unique_counts.items():
        t = targets.get(tid)
        if t is not None and not t.is_decoy:
            a_sum += c0
            b_sum += c1
    return a_sum / (a_sum + b_sum)


def strand_mode(p_dutp: float) -> str:
    if p_dutp >= 0.8:
        return "reverse"
    if p_dutp <= 0.2:
        return "forward"
    return "unstranded"


# ---------------------------------------------------------------------------
# Pass-1 / Pass-2 quantification
# ---------------------------------------------------------------------------

def _vectorized_group_var(
    ec_idx: np.ndarray,
    grp_code: np.ndarray,
    ws: np.ndarray,
    mask: np.ndarray,
    ec_cnt: np.ndarray,
    grp_labels: list[str],
    n_ec: int,
) -> dict[str, float]:
    """Vectorized cross-group ambiguity variance matching add_group_var() exactly."""
    valid = mask & (grp_code >= 0)
    if not np.any(valid):
        return {}
    v_ec = ec_idx[valid]
    v_gr = grp_code[valid]
    v_ws = ws[valid]
    n_g = max(1, len(grp_labels))
    comb = v_ec.astype(np.int64) * n_g + v_gr.astype(np.int64)
    order = np.argsort(comb, kind="mergesort")
    s_comb = comb[order]
    s_ws = v_ws[order]
    uniq_comb, seg_starts = np.unique(s_comb, return_index=True)
    wg = np.add.reduceat(s_ws, seg_starts)
    seg_ec = uniq_comb // n_g
    seg_gr = uniq_comb % n_g
    n_grp_per_ec = np.bincount(seg_ec, minlength=n_ec)
    cross = n_grp_per_ec[seg_ec] > 1
    if not np.any(cross):
        return {}
    c_gr = seg_gr[cross]
    c_wg = wg[cross]
    c_cnt = ec_cnt[seg_ec[cross]]
    contrib = c_cnt * c_wg * (1.0 - c_wg)
    sums = np.bincount(c_gr, weights=contrib, minlength=n_g)
    return {grp_labels[i]: float(sums[i]) for i in np.nonzero(sums)[0]}


def _quantify_run_sparse(
    acc: RunAccumulator,
    catalog: Catalog,
    dna_prob: dict[str, float] | None = None,
    p_dutp: float = 0.5,
    cohort_theta: dict[str, float] | None = None,
    two_way: bool = True,
    em_iters: int = 6,
    dirichlet_alpha: float = 0.5,
    include_posteriors: bool = True,
) -> tuple[dict[str, float], dict[str, float], dict[int, list[float]], dict[str, float]]:
    """Sparse active-target EM returning (active_eff, active_var, post, gvar)."""
    targets = catalog.targets
    mode = strand_mode(p_dutp)
    dna_prob = dna_prob or {}
    post: dict[int, list[float]] = {}
    gvar: dict[str, float] = {}

    act: set[str] = set(acc.unique_counts)
    for ts, _f, _l in acc.eq_classes:
        act.update(ts)
    active = [t for t in act if t in targets]
    if not active and not acc.eq_classes:
        return {}, {}, post, gvar

    M = len(active)
    t_to_i = {t: i for i, t in enumerate(active)}
    grp_map: dict[str, int] = {}
    grp_labels: list[str] = []
    t_grp = np.empty(M + 1, dtype=np.int32)
    for i, t in enumerate(active):
        g = targets[t].group_id
        gi = grp_map.get(g)
        if gi is None:
            gi = len(grp_labels)
            grp_map[g] = gi
            grp_labels.append(g)
        t_grp[i] = gi
    t_grp[M] = -1

    K = len(acc.eq_classes)
    if K == 0:
        cur_eff: dict[str, float] = {}
        for tid, (c0, c1) in acc.unique_counts.items():
            if tid in t_to_i:
                if not two_way or mode == "unstranded":
                    cur_eff[tid] = float(c0 + c1)
                else:
                    cur_eff[tid] = float(c0 if mode == "reverse" else c1)
        if two_way and em_iters <= 0:
            cur_eff = {}
        return cur_eff, {}, post, gvar

    # Flatten equivalence classes into CSR arrays once
    ec_items = list(acc.eq_classes.items())
    lens = np.fromiter((len(k[0]) for k, _ in ec_items), dtype=np.int64, count=K)
    starts = np.zeros(K, dtype=np.int64)
    if K > 1:
        np.cumsum(lens[:-1], out=starts[1:])
    L = int(starts[-1] + lens[-1])
    ec_cnt = np.fromiter((float(c) for _, c in ec_items), dtype=np.float64, count=K)

    flat_tids = list(itertools.chain.from_iterable(k[0] for k, _ in ec_items))
    ec_ti = np.fromiter((t_to_i.get(t, M) for t in flat_tids), dtype=np.int32, count=L)
    flat_lls = np.fromiter(itertools.chain.from_iterable(k[2] for k, _ in ec_items), dtype=np.float64, count=L)
    exp_lls = np.exp(flat_lls)

    if not two_way:
        valid_t = ec_ti < M
        eff_arr = np.zeros(M, dtype=np.float64)
        for tid, (c0, c1) in acc.unique_counts.items():
            ti = t_to_i.get(tid)
            if ti is not None:
                eff_arr[ti] += c0 + c1
        dummy_m = np.zeros(M, dtype=np.float64)
        c_res = c_em_sparse(
            M, K, L, len(grp_labels), False, 1, dirichlet_alpha,
            starts, lens, ec_cnt, ec_ti, t_grp, exp_lls, valid_t,
            eff_arr, dummy_m, dummy_m, dummy_m,
        )
        if c_res is not None:
            eff_arr, var_arr, ws, gvar_arr = c_res
            gvar = {grp_labels[i]: float(gvar_arr[i]) for i in np.nonzero(gvar_arr)[0]}
        else:
            ec_idx = np.repeat(np.arange(K, dtype=np.int64), lens)
            ec_cnt_rep = np.repeat(ec_cnt, lens)
            z = np.add.reduceat(exp_lls, starts)
            z_rep = np.repeat(np.where(z != 0.0, z, 1.0), lens)
            ws = exp_lls / z_rep
            eff_arr = eff_arr + np.bincount(ec_ti[valid_t], weights=(ec_cnt_rep * ws)[valid_t], minlength=M)
            var_arr = np.bincount(ec_ti[valid_t], weights=(ec_cnt_rep * ws * (1.0 - ws))[valid_t], minlength=M)
            gvar = _vectorized_group_var(ec_idx, t_grp[ec_ti], ws, valid_t, ec_cnt, grp_labels, K)
        if include_posteriors:
            ws_list = ws.tolist()
            for idx in range(K):
                s0 = int(starts[idx])
                post[idx] = ws_list[s0:s0 + int(lens[idx])]
        return (
            {active[i]: float(eff_arr[i]) for i in np.nonzero(eff_arr)[0]},
            {active[i]: float(var_arr[i]) for i in np.nonzero(var_arr)[0]},
            post,
            gvar,
        )

    if em_iters <= 0:
        return {}, {}, post, gvar

    w_a = max(0.005, min(0.995, p_dutp))
    w_b = 1.0 - w_a
    flat_flags = np.fromiter(itertools.chain.from_iterable(k[1] for k, _ in ec_items), dtype=bool, count=L)
    if mode == "unstranded":
        strand_w_arr = np.full(L, 0.5, dtype=np.float64)
        sense_arr = np.ones(L, dtype=bool)
    elif mode == "reverse":
        strand_w_arr = np.where(flat_flags, w_a, w_b)
        sense_arr = flat_flags
    else:
        strand_w_arr = np.where(flat_flags, w_a, w_b)
        sense_arr = ~flat_flags

    if dna_prob:
        cand_dna = np.fromiter((max(1e-7, dna_prob.get(t, 1.0)) for t in flat_tids), dtype=np.float64, count=L)
        ec_const = cand_dna * strand_w_arr * exp_lls
        base_prob = np.fromiter((max(1e-8, dna_prob.get(t, 1.0)) for t in active), dtype=np.float64, count=M)
    else:
        ec_const = strand_w_arr * exp_lls
        base_prob = np.ones(M, dtype=np.float64)

    inv_len = np.fromiter((1.0 / max(100.0, float(targets[t].length)) for t in active), dtype=np.float64, count=M)
    prior_arr = (
        np.fromiter((cohort_theta.get(t, 0.0) for t in active), dtype=np.float64, count=M)
        if cohort_theta
        else np.zeros(M, dtype=np.float64)
    )
    uniq_arr = np.zeros(M, dtype=np.float64)
    for tid, (c0, c1) in acc.unique_counts.items():
        ti = t_to_i.get(tid)
        if ti is not None:
            if mode == "unstranded":
                uniq_arr[ti] += c0 + c1
            else:
                uniq_arr[ti] += c0 if mode == "reverse" else c1

    valid_sense = (ec_ti < M) & sense_arr
    c_res = c_em_sparse(
        M, K, L, len(grp_labels), True, em_iters, dirichlet_alpha,
        starts, lens, ec_cnt, ec_ti, t_grp, ec_const, valid_sense,
        uniq_arr, base_prob, prior_arr, inv_len,
    )
    if c_res is not None:
        cur_arr, var_arr, ws, gvar_arr = c_res
        gvar = {grp_labels[i]: float(gvar_arr[i]) for i in np.nonzero(gvar_arr)[0]}
    else:
        ec_idx = np.repeat(np.arange(K, dtype=np.int64), lens)
        ec_cnt_rep = np.repeat(ec_cnt, lens)
        inv_lens_rep = np.repeat(1.0 / np.maximum(1, lens), lens)
        theta_arr = np.empty(M + 1, dtype=np.float64)
        theta_arr[:M] = base_prob * (prior_arr + dirichlet_alpha) * inv_len
        theta_arr[M] = 1e-8
        valid_ti = ec_ti[valid_sense]
        valid_cnt = ec_cnt_rep[valid_sense]
        cur_arr = uniq_arr
        ws = inv_lens_rep
        for _it in range(em_iters):
            raw = ec_const * theta_arr[ec_ti]
            z = np.add.reduceat(raw, starts)
            z_rep = np.repeat(z, lens)
            pos = z_rep > 0.0
            ws = np.where(pos, raw / np.where(pos, z_rep, 1.0), inv_lens_rep)
            w_valid = ws[valid_sense]
            cur_arr = uniq_arr + np.bincount(valid_ti, weights=valid_cnt * w_valid, minlength=M)
            theta_arr[:M] = base_prob * (cur_arr + 0.05 * prior_arr + 0.05 * dirichlet_alpha) * inv_len
        w_valid = ws[valid_sense]
        var_arr = np.bincount(valid_ti, weights=valid_cnt * w_valid * (1.0 - w_valid), minlength=M)
        gvar = _vectorized_group_var(ec_idx, t_grp[ec_ti], ws, valid_sense, ec_cnt, grp_labels, K)
    if include_posteriors:
        ws_list = ws.tolist()
        for idx in range(K):
            s0 = int(starts[idx])
            post[idx] = ws_list[s0:s0 + int(lens[idx])]
    return (
        {active[i]: float(cur_arr[i]) for i in np.nonzero(cur_arr)[0]},
        {active[i]: float(var_arr[i]) for i in np.nonzero(var_arr)[0]},
        post,
        gvar,
    )


def quantify_run(
    acc: RunAccumulator,
    catalog: Catalog,
    dna_prob: dict[str, float] | None = None,
    p_dutp: float = 0.5,
    cohort_theta: dict[str, float] | None = None,
    two_way: bool = True,
    em_iters: int = 6,
    dirichlet_alpha: float = 0.5,
    include_posteriors: bool = True,
) -> tuple[dict[str, float], dict[str, float], dict[int, list[float]], dict[str, float]]:
    """Effective counts, target assignment variance, eq-class posteriors and group-level
    (cross-group) assignment variance for one run.
    """
    act_eff, act_var, post, gvar = _quantify_run_sparse(
        acc, catalog, dna_prob, p_dutp, cohort_theta, two_way, em_iters, dirichlet_alpha, include_posteriors
    )
    eff = dict.fromkeys(catalog.targets, 0.0)
    var = dict.fromkeys(catalog.targets, 0.0)
    eff.update(act_eff)
    var.update(act_var)
    return eff, var, post, gvar


def cohort_abundance_prior(accs: list[RunAccumulator], catalog: Catalog) -> dict[str, float]:
    """Pooled unique-read abundance across the cohort (sparse over touched targets)."""
    tot: dict[str, float] = {}
    targets = catalog.targets
    n_runs = 0
    for acc in accs:
        if acc.run.is_wgs:
            continue
        n_runs += 1
        for tid, (c0, c1) in acc.unique_counts.items():
            if tid in targets:
                tot[tid] = tot.get(tid, 0.0) + c0 + c1
    denom = max(1, n_runs)
    return {t: v / denom for t, v in tot.items()}


def _catalog_arrow_static(catalog: Catalog, include_decoys: bool) -> dict:
    """Build and cache static catalog Arrow columns and target/group row maps once per catalog."""
    cache = getattr(catalog, "_solver_arrow_cache", None)
    if cache is None:
        cache = {}
        setattr(catalog, "_solver_arrow_cache", cache)
    key = (len(catalog.targets), bool(include_decoys))
    hit = cache.get(key)
    if hit is not None:
        return hit
    recs = list(catalog.targets.values()) if include_decoys else list(catalog.gene_targets)
    tids = [r.target_id for r in recs]
    t_to_row = {t: i for i, t in enumerate(tids)}
    n = len(tids)
    base_static = {
        "target_id": pa.array(tids, pa.string()),
        "assembly_id": pa.array([r.assembly_id for r in recs], pa.string()),
        "gene_id": pa.array([r.gene_id for r in recs], pa.string()),
        "group_id": pa.array([r.group_id for r in recs], pa.string()),
        "symbol": pa.array([r.symbol for r in recs], pa.string()),
        "biotype": pa.array([r.biotype for r in recs], pa.string()),
        "length_bp": pa.array([r.length for r in recs], pa.int32()),
    }
    grp_tmp: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        grp_tmp.setdefault(r.group_id, []).append(i)
    grp_to_rows = {g: np.asarray(rows, dtype=np.int32) for g, rows in grp_tmp.items()}
    entry = {
        "tids": tids,
        "t_to_row": t_to_row,
        "n": n,
        "base_static": base_static,
        "grp_to_rows": grp_to_rows,
    }
    cache[key] = entry
    return entry


def quantify_cohort(
    accs: dict[str, RunAccumulator],
    catalog: Catalog,
    dna_prob: dict[str, float],
    dna_counts: dict[str, float],
    strand_priors: dict[str, float],
    two_way: bool = True,
    em_iters: int = 6,
    cohort_rounds: int = 2,
    include_posteriors: bool = True,
    include_decoys: bool = True,
) -> tuple[pa.Table, dict[tuple[str, int], list[float]]]:
    """Quantify every RNA run; returns a TARGET_COUNTS table and eq-class posteriors."""
    rna = [a for a in accs.values() if not a.run.is_wgs]
    cohort_theta = cohort_abundance_prior(rna, catalog) if two_way else None
    results: dict[str, tuple[dict[str, float], dict[str, float], dict[int, list[float]], dict[str, float]]] = {}
    n_rna = max(1, len(rna))
    max_workers = min(len(rna), int(os.environ.get("PANEXPR_EM_WORKERS", "8")))
    for rnd in range(cohort_rounds if two_way else 1):
        is_last_rnd = (rnd == (cohort_rounds - 1 if two_way else 0))
        want_post = include_posteriors and is_last_rnd
        if len(rna) >= 2 and max_workers > 1:
            def _run_one(acc: RunAccumulator):
                return acc.run.run_id, _quantify_run_sparse(
                    acc, catalog, dna_prob, strand_priors.get(acc.run.run_id, 0.5),
                    cohort_theta, two_way, em_iters, include_posteriors=want_post,
                )
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for rid, val in ex.map(_run_one, rna):
                    results[rid] = val
        else:
            for acc in rna:
                results[acc.run.run_id] = _quantify_run_sparse(
                    acc, catalog, dna_prob, strand_priors.get(acc.run.run_id, 0.5),
                    cohort_theta, two_way, em_iters, include_posteriors=want_post,
                )
        if two_way and not is_last_rnd:
            pooled_sparse: dict[str, float] = {}
            for act_eff, _v, _p, _g in results.values():
                for t, val in act_eff.items():
                    pooled_sparse[t] = pooled_sparse.get(t, 0.0) + val
            cohort_theta = {t: val / n_rna for t, val in pooled_sparse.items()}

    cached = _catalog_arrow_static(catalog, include_decoys)
    tids = cached["tids"]
    t_to_row = cached["t_to_row"]
    n = cached["n"]
    grp_to_rows = cached["grp_to_rows"]
    wgs_arr = (
        pa.array(np.fromiter((float(dna_counts.get(t, 0.0)) for t in tids), dtype=np.float64, count=n))
        if dna_counts
        else pa.array(np.zeros(n, dtype=np.float64))
    )
    dna_arr = (
        pa.array(np.fromiter((float(dna_prob.get(t, 1.0)) for t in tids), dtype=np.float64, count=n))
        if dna_prob
        else pa.array(np.ones(n, dtype=np.float64))
    )
    static = {
        **cached["base_static"],
        "wgs_dna_reads": wgs_arr,
        "dna_present_prob": dna_arr,
    }

    chunks: dict[str, list] = {k: [] for k in TARGET_COUNTS_SCHEMA.names}
    post_lookup: dict[tuple[str, int], list[float]] = {}
    for acc in rna:
        act_eff, act_var, post, gvar = results[acc.run.run_id]
        if include_posteriors:
            for idx, ws in post.items():
                post_lookup[(acc.run.run_id, idx)] = ws
        p = strand_priors.get(acc.run.run_id, 0.5)
        mode = strand_mode(p)
        u_s = np.zeros(n, dtype=np.float64)
        u_a = np.zeros(n, dtype=np.float64)
        fwd = (mode == "forward")
        for t, (c0, c1) in acc.unique_counts.items():
            ri = t_to_row.get(t)
            if ri is not None:
                if fwd:
                    u_s[ri] = c1
                    u_a[ri] = c0
                else:
                    u_s[ri] = c0
                    u_a[ri] = c1
        eff_col = np.zeros(n, dtype=np.float64)
        for t, val in act_eff.items():
            ri = t_to_row.get(t)
            if ri is not None:
                eff_col[ri] = val
        var_col = np.zeros(n, dtype=np.float64)
        for t, val in act_var.items():
            ri = t_to_row.get(t)
            if ri is not None:
                var_col[ri] = val
        gvar_col = np.zeros(n, dtype=np.float64)
        for g, val in gvar.items():
            rows = grp_to_rows.get(g)
            if rows is not None:
                gvar_col[rows] = val

        chunks["run_id"].append(pa.repeat(pa.scalar(acc.run.run_id, pa.string()), n))
        chunks["sample"].append(pa.repeat(pa.scalar(acc.run.sample, pa.string()), n))
        chunks["condition"].append(pa.repeat(pa.scalar(acc.run.condition, pa.string()), n))
        chunks["unique_sense"].append(pa.array(u_s))
        chunks["unique_antisense"].append(pa.array(u_a))
        chunks["effective_count"].append(pa.array(eff_col))
        chunks["assignment_variance"].append(pa.array(var_col))
        chunks["group_assignment_variance"].append(pa.array(gvar_col))
        for k, arr in static.items():
            chunks[k].append(arr)
    cols = {k: pa.chunked_array(chunks[k], TARGET_COUNTS_SCHEMA.field(k).type) for k in TARGET_COUNTS_SCHEMA.names}
    return pa.table(cols).cast(TARGET_COUNTS_SCHEMA), post_lookup


def assembly_assignment_summary(table: pa.Table) -> dict[str, float]:
    """Effective RNA counts per assembly (used for cross-strain leak metrics; zero-pandas Arrow path)."""
    import pyarrow.compute as pc
    if table.num_rows == 0:
        return {}
    bt = table.column("biotype")
    t = table.filter(pc.invert(pc.starts_with(bt, "decoy"))).select(["assembly_id", "effective_count"])
    if t.num_rows == 0:
        return {}
    agg = t.group_by("assembly_id").aggregate([("effective_count", "sum")])
    return {a: float(v) for a, v in zip(agg.column("assembly_id").to_pylist(), agg.column("effective_count_sum").to_pylist())}


# ---------------------------------------------------------------------------
# Gene-level aggregation
# ---------------------------------------------------------------------------

DE_EXCLUDED_BIOTYPES = ("rRNA", "tRNA", "tmRNA", "RNase_P_RNA", "SRP_RNA", "window")


def gene_level_tables(
    table: pa.Table,
    catalog: Catalog,
    length_offsets: bool = False,
    frag_mean: float = 200.0,
    exclude_biotypes: tuple[str, ...] = DE_EXCLUDED_BIOTYPES,
) -> dict:
    """Aggregate target counts to groups (tx2gene / ortholog groups).

    Returns counts (sample x group), assignment variance (sample x group),
    sample metadata, per-group P(DNA), and optionally length-scaled counts.
    """
    import pyarrow.compute as pc
    bt = table.column("biotype")
    keep = pc.and_(pc.invert(pc.starts_with(bt, "decoy")), pc.invert(pc.is_in(bt, value_set=pa.array(list(exclude_biotypes), pa.string()))))
    cols_sel = ["sample", "condition", "run_id", "group_id", "effective_count", "group_assignment_variance", "dna_present_prob"]
    if length_offsets and catalog.mode == "eukaryotic":
        cols_sel += ["target_id", "length_bp"]
    t = table.filter(keep).select(cols_sel)
    s_codes, s_lab = _sorted_codes(t.column("sample"))
    g_codes, g_lab = _sorted_codes(t.column("group_id"))
    n_s, n_g = len(s_lab), len(g_lab)
    eff_np = t.column("effective_count").to_numpy(zero_copy_only=False)
    gav_np = t.column("group_assignment_variance").to_numpy(zero_copy_only=False)
    dna_np = t.column("dna_present_prob").to_numpy(zero_copy_only=False)
    if n_s > 0 and n_g > 0 and len(s_codes) > 0:
        flat_sg = s_codes * n_g + g_codes
        cnt_mat = np.bincount(flat_sg, weights=eff_np, minlength=n_s * n_g).reshape(n_s, n_g)
        avar_mat = np.zeros((n_s, n_g), dtype=np.float64)
        nz_gav = gav_np != 0.0
        if np.any(nz_gav):
            np.maximum.at(avar_mat, (s_codes[nz_gav], g_codes[nz_gav]), gav_np[nz_gav])
        dna_arr = np.full(n_g, -np.inf, dtype=np.float64)
        np.maximum.at(dna_arr, g_codes, dna_np)
        dna_arr = np.where(np.isfinite(dna_arr), dna_arr, 1.0)
    else:
        cnt_mat = np.zeros((n_s, n_g), dtype=np.float64)
        avar_mat = np.zeros((n_s, n_g), dtype=np.float64)
        dna_arr = np.ones(n_g, dtype=np.float64)
    s_idx = pd.Index(s_lab, name="sample")
    g_idx = pd.Index(g_lab, name="group_id")
    counts = pd.DataFrame(cnt_mat, index=s_idx, columns=g_idx)
    avar = pd.DataFrame(avar_mat, index=s_idx, columns=g_idx)
    # drop_duplicates("sample") semantics: first row of each sample, in order of first appearance
    _u, first_rows = np.unique(s_codes, return_index=True)
    first_rows = np.sort(first_rows)
    meta = pd.DataFrame({"condition": t.column("condition").take(pa.array(first_rows)).to_pylist(),
                         "run_id": t.column("run_id").take(pa.array(first_rows)).to_pylist()},
                        index=pd.Index([s_lab[i] for i in s_codes[first_rows]], name="sample"))
    dna = pd.Series(dna_arr, index=g_idx, name="dna_present_prob")
    out = {"counts": counts, "avar": avar, "meta": meta, "dna_prob": dna}
    if length_offsets and catalog.mode == "eukaryotic":
        tx_codes, tx_lab = _sorted_codes(t.column("target_id"))
        n_tx, n_s = len(tx_lab), len(s_lab)
        tx_mat = np.zeros((n_tx, n_s), dtype=np.float64)
        np.add.at(tx_mat, (tx_codes, s_codes), t.column("effective_count").to_numpy())
        _, tx_first = np.unique(tx_codes, return_index=True)
        len_raw = t.column("length_bp").take(pa.array(tx_first)).to_numpy(zero_copy_only=False).astype(np.float64)
        tx_first_codes = tx_codes[tx_first]
        lengths_arr = np.ones(n_tx, dtype=np.float64)
        lengths_arr[tx_first_codes] = len_raw
        eff_arr = np.where(
            lengths_arr > frag_mean,
            np.maximum(1.0, lengths_arr - frag_mean),
            np.maximum(1.0, lengths_arr / 2.0),
        )
        grp_first = g_codes[tx_first]
        t2g_arr = np.empty(n_tx, dtype=np.int64)
        t2g_arr[tx_first_codes] = grp_first
        t2g_series = pd.Series([g_lab[i] for i in t2g_arr], index=pd.Index(tx_lab, name="target_id"))
        tx_counts_df = pd.DataFrame(tx_mat, index=t2g_series.index, columns=pd.Index(s_lab, name="sample"))
        tx_len_df = pd.DataFrame(np.broadcast_to(eff_arr[:, None], (n_tx, n_s)), index=t2g_series.index, columns=tx_counts_df.columns)
        txi = tximport_gene_level(tx_counts_df, tx_len_df, t2g_series)
        out["counts_length_scaled"] = txi["counts_length_scaled"].T.reindex(counts.index).fillna(0.0)
        out["norm_mat"] = txi["norm_mat"].T.reindex(counts.index)
    return out


def _sorted_codes(col) -> tuple[np.ndarray, list[str]]:
    """Integer codes whose order equals Python's sort order of the labels (what pivot_table's sorted axes use)."""
    enc = col.combine_chunks().dictionary_encode() if hasattr(col, "combine_chunks") else col.dictionary_encode()
    labels = enc.dictionary.to_pylist()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    rank = np.empty(len(labels), dtype=np.int64)
    rank[order] = np.arange(len(labels))
    return rank[enc.indices.to_numpy(zero_copy_only=False)], [labels[i] for i in order]


def _gene_level_tables_pandas(table: pa.Table, catalog: Catalog, frag_mean: float, exclude_biotypes: tuple[str, ...]) -> dict:
    """Original pandas path, kept for the length-offset (tximport) variant."""
    df = table.to_pandas()
    df = df[~df["biotype"].str.startswith("decoy") & ~df["biotype"].isin(exclude_biotypes)]
    counts = df.pivot_table(index="sample", columns="group_id", values="effective_count", aggfunc="sum").fillna(0.0)
    avar = df.pivot_table(index="sample", columns="group_id", values="group_assignment_variance", aggfunc="max").fillna(0.0)
    meta = df[["sample", "condition", "run_id"]].drop_duplicates("sample").set_index("sample")
    dna = df.groupby("group_id")["dna_present_prob"].max()
    out = {"counts": counts, "avar": avar, "meta": meta, "dna_prob": dna}
    tx_counts = df.pivot_table(index="target_id", columns="sample", values="effective_count", aggfunc="sum").fillna(0.0)
    lengths = df.drop_duplicates("target_id").set_index("target_id")["length_bp"].astype(float)
    eff_len = (lengths - frag_mean).clip(lower=1.0)
    eff_len = eff_len.where(lengths > frag_mean, (lengths / 2.0).clip(lower=1.0))
    tx_len = pd.DataFrame({s: eff_len for s in tx_counts.columns}).reindex(tx_counts.index)
    t2g = df.drop_duplicates("target_id").set_index("target_id")["group_id"]
    txi = tximport_gene_level(tx_counts, tx_len, t2g)
    out["counts_length_scaled"] = txi["counts_length_scaled"].T.reindex(counts.index).fillna(0.0)
    out["norm_mat"] = txi["norm_mat"].T.reindex(counts.index)
    return out


# ---------------------------------------------------------------------------
# PyDESeq2 adapter and calibrated uncertainty
# ---------------------------------------------------------------------------

def run_pydeseq2(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    contrast: list[str],
    design: str = "~condition",
    min_total: int = 2,
    n_cpus: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Negative-binomial GLM (PyDESeq2). counts: sample x gene. Returns (results, dispersions, size_factors)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    if n_cpus is None:
        n_cpus = int(os.environ.get("PANEXPR_DESEQ2_CPUS", "1"))   # 1 = the locked behaviour; >1 crashed loky in the SIF
    meta_aligned = meta.loc[counts.index].copy()
    keep = counts.sum(axis=0) >= min_total
    sub = counts.loc[:, keep].round().astype(int)
    dds = DeseqDataSet(counts=sub, metadata=meta_aligned, design=design, refit_cooks=True, quiet=True, n_cpus=n_cpus)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=contrast, alpha=0.05, cooks_filter=True, independent_filter=True, quiet=True, n_cpus=n_cpus)
    ds.summary()
    disp = pd.Series(np.asarray(dds.var["dispersions"]), index=dds.var_names)
    sf = pd.Series(np.asarray(dds.obs["size_factors"]), index=dds.obs_names)
    return ds.results_df.copy(), disp, sf


def _assemble_de_dataframe(
    counts: pd.DataFrame,
    avar: pd.DataFrame,
    meta: pd.DataFrame,
    contrast: list[str],
    res: pd.DataFrame,
    disp: pd.Series,
    sf: pd.Series,
    dna_prob: pd.Series | None,
    symbols: dict[str, str] | None,
    gene_ids: dict[str, str] | None,
    iteration: int,
    min_expressed_reads: float,
) -> pd.DataFrame:
    """Vectorized uncertainty propagation, tech-vs-bio variance fraction, and BH adjustment."""
    factor, lvl_a, lvl_b = contrast
    cols = counts.columns
    n_cols = len(cols)
    if n_cols == 0:
        return pd.DataFrame(columns=[
            "iteration", "group_id", "gene_id", "symbol", "dna_present_prob", "streamed_reads",
            "base_mean", "log2fc", "lfc_se_deseq2", "var_placement_ambiguity", "lfc_se_total",
            "ci95_low", "ci95_high", "pvalue", "padj", "tech_var_fraction", "demand_score", "status",
        ])
    sf_s = sf.reindex(counts.index)
    norm_c = counts.div(sf_s, axis=0)
    norm_v = avar.reindex(index=counts.index, columns=cols).fillna(0.0).div(sf_s ** 2, axis=0)
    cond_col = meta.loc[counts.index, factor]
    a_s = [s for s in counts.index if cond_col.loc[s] == lvl_a]
    b_s = [s for s in counts.index if cond_col.loc[s] == lvl_b]
    n_a, n_b = max(1, len(a_s)), max(1, len(b_s))
    mean_a = (norm_c.loc[a_s].mean(axis=0) + 0.5).to_numpy(dtype=np.float64)
    mean_b = (norm_c.loc[b_s].mean(axis=0) + 0.5).to_numpy(dtype=np.float64)
    v_a = norm_v.loc[a_s].sum(axis=0).to_numpy(dtype=np.float64) / (n_a ** 2)
    v_b = norm_v.loc[b_s].sum(axis=0).to_numpy(dtype=np.float64) / (n_b ** 2)
    pv_raw = (v_a / (mean_a ** 2) + v_b / (mean_b ** 2)) / LN2_SQ
    streamed_arr = counts.loc[a_s + b_s].sum(axis=0).to_numpy(dtype=np.float64)
    col_mean_arr = norm_c.mean(axis=0).to_numpy(dtype=np.float64)

    cols_list = list(cols)
    p_dna = (
        dna_prob.reindex(cols).fillna(1.0).to_numpy(dtype=np.float64)
        if dna_prob is not None
        else np.ones(n_cols, dtype=np.float64)
    )
    gene_id_col = [gene_ids.get(g, g) for g in cols_list] if gene_ids else cols_list
    symbol_col = [symbols.get(g, g) for g in cols_list] if symbols else cols_list

    is_absent = p_dna < 0.05
    in_res = cols.isin(res.index) if len(res) else np.zeros(n_cols, dtype=bool)
    is_unexpr = (~is_absent) & (~in_res)
    is_tested = (~is_absent) & in_res

    if len(res):
        res_re = res.reindex(cols)
        res_bm = res_re["baseMean"].to_numpy(dtype=np.float64)
        res_lfc = res_re["log2FoldChange"].to_numpy(dtype=np.float64)
        res_se = res_re["lfcSE"].to_numpy(dtype=np.float64)
        res_padj_notna = res_re["padj"].notna().to_numpy(dtype=bool)
    else:
        res_bm = np.full(n_cols, np.nan, dtype=np.float64)
        res_lfc = np.full(n_cols, np.nan, dtype=np.float64)
        res_se = np.full(n_cols, np.nan, dtype=np.float64)
        res_padj_notna = np.zeros(n_cols, dtype=bool)

    disp_arr = np.maximum(1e-4, disp.reindex(cols).fillna(0.1).to_numpy(dtype=np.float64)) if len(disp) else np.full(n_cols, 0.1, dtype=np.float64)

    base_mean = np.where(is_absent, 0.0, np.where(is_unexpr, col_mean_arr, res_bm))
    log2fc = np.where(is_tested, res_lfc, np.nan)
    lfc_se_deseq2 = np.where(is_tested, res_se, np.nan)
    var_pa = np.where(is_absent, 0.0, pv_raw)
    se_tot = np.where(is_tested & (~np.isnan(res_se)), np.sqrt(res_se ** 2 + pv_raw), np.nan)

    pos_se = is_tested & (~np.isnan(se_tot)) & (se_tot > 0.0)
    z_arr = np.zeros(n_cols, dtype=np.float64)
    z_arr[pos_se] = np.abs(log2fc[pos_se] / se_tot[pos_se])
    p_tot = np.full(n_cols, np.nan, dtype=np.float64)
    z_nan = np.isnan(z_arr)
    t_ok = is_tested & (~z_nan)
    p_tot[is_tested & z_nan] = 1.0
    if np.any(t_ok):
        p_tot[t_ok] = 2.0 * norm.sf(z_arr[t_ok])

    ci95_low = log2fc - 1.96 * se_tot
    ci95_high = log2fc + 1.96 * se_tot

    v_tech = (1.0 / (n_a * mean_a) + 1.0 / (n_b * mean_b)) / LN2_SQ
    v_bio = disp_arr * (1.0 / n_a + 1.0 / n_b) / LN2_SQ
    v_sum = v_tech + v_bio
    tf_tested = np.where(v_sum > 0.0, v_tech / np.where(v_sum > 0.0, v_sum, 1.0), 0.0)
    tech_frac = np.where(is_absent, 0.0, np.where(is_unexpr, 1.0, tf_tested))
    boundary = np.exp(-0.5 * ((z_arr - 2.2) / 1.1) ** 2)
    demand = np.where(is_tested, tf_tested * boundary, 0.0)

    valid = is_tested & res_padj_notna
    padj = np.full(n_cols, np.nan, dtype=np.float64)
    if np.any(valid):
        _, q, _, _ = multipletests(p_tot[valid], method="fdr_bh")
        padj[valid] = q

    status = np.full(n_cols, "pending", dtype=object)
    status[is_absent] = "genetically_absent"
    status[is_unexpr] = "unexpressed"
    sig = padj < 0.05
    status[is_tested & sig] = "DE"
    status[is_tested & (~sig)] = "non_DE"
    status[(~is_absent) & (~is_unexpr) & (streamed_arr < min_expressed_reads)] = "low_depth"

    return pd.DataFrame({
        "iteration": np.full(n_cols, int(iteration), dtype=np.int64),
        "group_id": cols_list,
        "gene_id": gene_id_col,
        "symbol": symbol_col,
        "dna_present_prob": p_dna,
        "streamed_reads": streamed_arr,
        "base_mean": base_mean,
        "log2fc": log2fc,
        "lfc_se_deseq2": lfc_se_deseq2,
        "var_placement_ambiguity": var_pa,
        "lfc_se_total": se_tot,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "pvalue": p_tot,
        "padj": padj,
        "tech_var_fraction": tech_frac,
        "demand_score": demand,
        "status": status,
    })


def _deseq2_size_factors_fast(C: np.ndarray) -> np.ndarray:
    """Closed-form DESeq2 median-of-ratios size factors for count matrix C (n_samples x n_genes)."""
    n_s = C.shape[0]
    if n_s == 0:
        return np.ones(0, dtype=np.float64)
    if C.shape[1] == 0:
        return np.ones(n_s, dtype=np.float64)
    pos_all = np.all(C > 0.0, axis=0)
    if np.any(pos_all):
        lc = np.log(C[:, pos_all])
        geo = np.mean(lc, axis=0, keepdims=True)
        sf = np.exp(np.median(lc - geo, axis=1))
    else:
        lib = np.sum(C, axis=1)
        pos_lib = lib > 0.0
        if np.any(pos_lib):
            gm = float(np.exp(np.mean(np.log(lib[pos_lib]))))
            sf = np.where(pos_lib, lib / max(1e-8, gm), 1.0)
        else:
            sf = np.ones(n_s, dtype=np.float64)
    sf = np.where(np.isfinite(sf) & (sf > 0.0), sf, 1.0)
    return sf


def fast_nb_mom_de(
    counts: pd.DataFrame,
    avar: pd.DataFrame,
    meta: pd.DataFrame,
    contrast: list[str],
    design: str = "~condition",
    dna_prob: pd.Series | None = None,
    symbols: dict[str, str] | None = None,
    gene_ids: dict[str, str] | None = None,
    iteration: int = 0,
    min_expressed_reads: float = 20.0,
    min_total: int = 2,
    irls_iters: int = 8,
) -> pd.DataFrame:
    """Closed-form NB Method-of-Moments + batched BLAS Fisher-scoring DE for intermediate loop iterations.

    Runs in ~15 ms on 60,000 mammalian gene groups (vs 45-90 s for full PyDESeq2 L-BFGS-B per iteration)
    while supporting arbitrary additive designs (including donor-blocked ``~ block + condition``).
    Full ``de_with_uncertainty`` (PyDESeq2) is always executed once the stopping rule triggers.
    """
    meta_aligned = meta.loc[counts.index].copy()
    keep_g = (counts.sum(axis=0) >= min_total).to_numpy(dtype=bool)
    C_round = np.rint(counts.to_numpy(dtype=np.float64))
    sf_vals = _deseq2_size_factors_fast(C_round[:, keep_g] if np.any(keep_g) else C_round)
    sf_series = pd.Series(sf_vals, index=counts.index)

    if not np.any(keep_g):
        empty_res = pd.DataFrame(columns=["baseMean", "log2FoldChange", "lfcSE", "padj"])
        return _assemble_de_dataframe(
            counts, avar, meta_aligned, contrast, empty_res, pd.Series(dtype=float), sf_series,
            dna_prob, symbols, gene_ids, iteration, min_expressed_reads,
        )

    sub_cols = counts.columns[keep_g]
    X, c_vec, keep_s = _design_matrix_and_contrast(meta_aligned, design, contrast)
    X_k = X[keep_s]
    N_s, p = X_k.shape
    M_sub = int(np.sum(keep_g))
    Y = C_round[np.ix_(keep_s, keep_g)]
    sf_k = np.maximum(1e-8, sf_vals[keep_s])
    off_k = np.log(sf_k)

    # Initial log-rate OLS + Method-of-Moments dispersion with parametric trend shrinkage
    q_obs = (Y / sf_k[:, None]).T  # (M_sub, N_s)
    Y_log = np.log(Y / sf_k[:, None] + 0.5)
    B = (np.linalg.pinv(X_k) @ Y_log).T  # (M_sub, p)
    mu0 = np.exp(np.clip(B @ X_k.T + off_k[None, :], -25.0, 25.0))
    q_hat = mu0 / sf_k[None, :]
    df = max(1.0, float(N_s - p))
    var_q = np.sum((q_obs - q_hat) ** 2, axis=1) / df
    q_bar = np.maximum(1e-6, np.mean(q_obs, axis=1))
    mean_inv_sf = float(np.mean(1.0 / sf_k))
    alpha_mom = np.clip((var_q - q_bar * mean_inv_sf) / (q_bar ** 2), 1e-4, 10.0)
    hi = q_bar >= 5.0
    if int(np.sum(hi)) >= 20:
        q_hi = q_bar[hi]
        a_hi = alpha_mom[hi]
        med_q = float(np.median(q_hi))
        a0 = max(0.01, float(np.median(a_hi[q_hi >= med_q])))
        a1 = max(0.05, float(np.median(np.maximum(0.0, a_hi - a0) * q_hi)))
    else:
        a0, a1 = 0.05, 1.0
    alpha_tr = np.clip(a0 + a1 / q_bar, 1e-4, 10.0)
    w_eb = df / (df + 4.0)
    alpha = np.exp(w_eb * np.log(alpha_mom) + (1.0 - w_eb) * np.log(alpha_tr))

    # Batched NB Fisher scoring across all M_sub genes simultaneously via BLAS DGEMM
    XX = X_k[:, :, None] * X_k[:, None, :]  # (N_s, p, p)
    reg = np.zeros((p, p), dtype=np.float64)
    np.fill_diagonal(reg, 1e-6)
    if p > 1:
        for j in range(1, p):
            reg[j, j] += 1e-4
    YT = Y.T  # (M_sub, N_s)
    for _ in range(max(1, irls_iters)):
        eta = np.clip(B @ X_k.T + off_k[None, :], -25.0, 25.0)
        mu = np.exp(eta)
        W = mu / (1.0 + alpha[:, None] * mu)
        Z = (B @ X_k.T) + (YT - mu) / np.maximum(1e-12, mu)
        H = np.tensordot(W, XX, axes=([1], [0])) + reg[None, :, :]
        rhs = (W * Z) @ X_k
        B_new = np.linalg.solve(H, rhs[:, :, None])[:, :, 0]
        step = B_new - B
        max_s = np.max(np.abs(step), axis=1, keepdims=True)
        scale = np.where(max_s > 2.0, 2.0 / np.maximum(max_s, 1e-12), 1.0)
        B = B + step * scale

    eta = np.clip(B @ X_k.T + off_k[None, :], -25.0, 25.0)
    mu = np.exp(eta)
    W = mu / (1.0 + alpha[:, None] * mu)
    H = np.tensordot(W, XX, axes=([1], [0])) + reg[None, :, :]
    c_tile = np.broadcast_to(c_vec[None, :, None], (M_sub, p, 1))
    u = np.linalg.solve(H, c_tile)[:, :, 0]
    ln2 = math.log(2.0)
    lfc_sub = (B @ c_vec) / ln2
    se_sub = np.sqrt(np.maximum(0.0, u @ c_vec)) / ln2
    bm_sub = np.mean(counts.to_numpy(dtype=np.float64)[:, keep_g] / sf_vals[:, None], axis=0)

    # Approximate independent filtering threshold matching DESeq2 (exclude near-zero mean genes from BH)
    res_df = pd.DataFrame({
        "baseMean": bm_sub,
        "log2FoldChange": lfc_sub,
        "lfcSE": se_sub,
        "padj": np.where(bm_sub >= 0.5, 1.0, np.nan),
    }, index=sub_cols)
    disp_series = pd.Series(alpha, index=sub_cols)
    return _assemble_de_dataframe(
        counts, avar, meta_aligned, contrast, res_df, disp_series, sf_series,
        dna_prob, symbols, gene_ids, iteration, min_expressed_reads,
    )


def de_with_uncertainty(
    counts: pd.DataFrame,
    avar: pd.DataFrame,
    meta: pd.DataFrame,
    contrast: list[str],
    design: str = "~condition",
    dna_prob: pd.Series | None = None,
    symbols: dict[str, str] | None = None,
    gene_ids: dict[str, str] | None = None,
    iteration: int = 0,
    min_expressed_reads: float = 20.0,
    return_fit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series, pd.Series]:
    """DE with placement-ambiguity variance propagated into lfc_se_total, 95% CIs and BH padj.

    Also returns the technical (Poisson read-sampling) vs biological (NB dispersion)
    variance fraction per gene and a demand score peaking at the decision boundary.
    """
    res, disp, sf = run_pydeseq2(counts, meta, contrast, design)
    df = _assemble_de_dataframe(
        counts, avar, meta, contrast, res, disp, sf,
        dna_prob, symbols, gene_ids, iteration, min_expressed_reads,
    )
    return (df, disp, sf) if return_fit else df


# ---------------------------------------------------------------------------
# Per-gene NB GLM refit with depth offsets (used after the borderline zoom)
# ---------------------------------------------------------------------------

def nb_glm_wald(y: np.ndarray, X: np.ndarray, offset: np.ndarray, alpha: float, max_iter: int = 100,
                prior_sd_log2: float = 4.0, return_cov: bool = False):
    """Fisher-scoring fit of a negative-binomial GLM (log link, known dispersion alpha).

    Non-intercept coefficients carry a weak Gaussian (ridge) prior with sd ``prior_sd_log2``
    log2 units so genes with zero counts in one group get a finite, bounded fold change
    (the same role as DESeq2's zero-centred LFC prior).  Returns (beta, se) by default, or
    (beta, se, cov) when ``return_cov=True``; ``offset`` is log(size_factor / depth_scale) per sample.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = math.log(max(1e-3, float(np.mean(y / np.exp(offset)))))
    alpha = max(1e-8, float(alpha))
    lam = np.zeros(p)
    lam[1:] = 1.0 / (prior_sd_log2 * math.log(2.0)) ** 2
    L = np.diag(lam)
    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        w = mu / (1.0 + alpha * mu)
        z = (X @ beta) + (y - mu) / mu
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X + L + 1e-8 * np.eye(p), XtW @ z)
        except np.linalg.LinAlgError:
            break
        step = new - beta
        if np.max(np.abs(step)) > 2.0:  # damped step for stability far from the optimum
            step *= 2.0 / np.max(np.abs(step))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ beta + offset
    mu = np.exp(np.clip(eta, -30, 30))
    w = mu / (1.0 + alpha * mu)
    cov = np.linalg.pinv((X.T * w) @ X + L)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return (beta, se, cov) if return_cov else (beta, se)


def _design_matrix_and_contrast(
    meta: pd.DataFrame,
    design: str,
    contrast: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full-rank model matrix ``X`` (n x p), contrast vector ``c`` (p,), and sample mask ``keep``.

    Supports arbitrary additive formulas ``~ f1 + f2 + ...`` (categorical or numeric terms, with
    optional ``C(col)`` wrappers) while matching the exact 2-column ``[1, cond == lvl_a]`` layout
    on simple 2-group ``~condition`` designs.
    """
    import re
    factor, lvl_a, lvl_b = contrast
    n = len(meta)
    cond = meta[factor].astype(str)
    is_a = (cond == str(lvl_a)).values
    is_b = (cond == str(lvl_b)).values
    rhs = design.split("~", 1)[-1].strip() if "~" in design else design.strip()
    raw_terms = [t.strip() for t in rhs.split("+") if t.strip() and t.strip() not in ("1", "0")]
    terms: list[str] = []
    for t in raw_terms:
        m = re.match(r"^C\(\s*([^,\)]+)\s*(?:,.*)?\)$", t)
        col = m.group(1).strip() if m else t
        if col in meta.columns and col not in terms:
            terms.append(col)
    if factor not in terms:
        terms.append(factor)

    # Exact fast path for single-factor 2-group design: subset to (lvl_a | lvl_b) with X = [1, cond == lvl_a]
    if terms == [factor]:
        keep = is_a | is_b
        X = np.column_stack([np.ones(n, dtype=float), is_a.astype(float)])
        c_vec = np.array([0.0, 1.0], dtype=float)
        return X, c_vec, keep

    # Multi-factor design: fit across all samples with complete covariates so block/batch effects use full cohort
    keep_mask = np.ones(n, dtype=bool)
    for col in terms:
        keep_mask &= meta[col].notna().values
    cols = [np.ones(n, dtype=float)]
    col_names = ["Intercept"]
    factor_col_idx: dict[str, int] = {}
    factor_ref: str | None = None
    for col in terms:
        s = meta[col]
        if col != factor and pd.api.types.is_numeric_dtype(s):
            vals = s.astype(float).fillna(0.0).values
            if np.std(vals[keep_mask]) > 0:
                cols.append(vals)
                col_names.append(col)
            continue
        s_str = s.astype(str)
        levels = sorted(set(s_str.values[keep_mask]))
        if not levels:
            continue
        # For the contrast factor, anchor reference at lvl_b when present so lvl_a has a direct +1 coefficient
        ref = str(lvl_b) if (col == factor and str(lvl_b) in levels) else levels[0]
        if col == factor:
            factor_ref = ref
        for lv in levels:
            if lv == ref:
                continue
            ind = (s_str == lv).astype(float).values
            if ind[keep_mask].sum() == 0:
                continue
            idx = len(cols)
            cols.append(ind)
            col_names.append(f"{col}[T.{lv}]")
            if col == factor:
                factor_col_idx[lv] = idx

    X_raw = np.column_stack(cols)
    c_raw = np.zeros(X_raw.shape[1], dtype=float)
    if str(lvl_a) != factor_ref and str(lvl_a) in factor_col_idx:
        c_raw[factor_col_idx[str(lvl_a)]] += 1.0
    if str(lvl_b) != factor_ref and str(lvl_b) in factor_col_idx:
        c_raw[factor_col_idx[str(lvl_b)]] -= 1.0

    # Prune any collinear nuisance columns while preserving Intercept and contrast columns
    protected = {0} | set(factor_col_idx.values())
    sel: list[int] = []
    cur_rank = 0
    for j in sorted(protected) + [k for k in range(X_raw.shape[1]) if k not in protected]:
        cand = sel + [j]
        rk = int(np.linalg.matrix_rank(X_raw[keep_mask][:, cand]))
        if rk > cur_rank:
            sel.append(j)
            cur_rank = rk
    sel.sort()
    X = X_raw[:, sel]
    c_vec = c_raw[sel]
    return X, c_vec, keep_mask


def combine_zoom_gene_tables(
    gl: dict,
    table_global: pa.Table,
    ztable: pa.Table,
    groups: list[str],
    depth_scale: pd.Series,
    catalog: Catalog,
    length_offsets: bool = False,
    frag_mean: float = 200.0,
    exclude_biotypes: tuple[str, ...] = DE_EXCLUDED_BIOTYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build consistent ``(counts_global, combined_zoom_counts, combined_avar)`` for post-zoom refits.

    When ``length_offsets=True`` on a eukaryotic catalog, ``counts_global`` is ``gl['counts_length_scaled']``
    and zoomed groups in ``combined_zoom_counts`` are scaled on the exact same Soneson et al. (2015)
    ``lengthScaledTPM`` scale (using depth-normalized transcript counts to weight isoform effective lengths
    and the global library length-scaling factor per sample), preventing length-scaled and unscaled counts
    from ever mixing in ``de_postzoom``.
    """
    import pyarrow.compute as pc
    raw_counts = gl["counts"]
    samples = list(raw_counts.index)
    zgl = gene_level_tables(ztable, catalog, length_offsets=False, frag_mean=frag_mean, exclude_biotypes=exclude_biotypes)
    zc = zgl["counts"].reindex(index=raw_counts.index, columns=raw_counts.columns).fillna(0.0)
    zv = zgl["avar"].reindex(index=raw_counts.index, columns=raw_counts.columns).fillna(0.0)
    avar_comb = gl["avar"].reindex(index=raw_counts.index, columns=raw_counts.columns).fillna(0.0).copy()
    gset = [g for g in groups if g in raw_counts.columns]
    if gset:
        avar_comb[gset] = avar_comb[gset] + zv[gset]

    if not (length_offsets and catalog.mode == "eukaryotic" and "counts_length_scaled" in gl):
        combined = raw_counts.copy()
        if gset:
            combined[gset] = raw_counts[gset] + zc[gset]
        return raw_counts, combined, avar_comb

    counts_scaled = gl["counts_length_scaled"].reindex(index=raw_counts.index, columns=raw_counts.columns).fillna(0.0)
    combined_scaled = counts_scaled.copy()
    if not gset:
        return counts_scaled, combined_scaled, avar_comb

    # Global sample factor alpha_s = lib_s / sum_{g'} (R_{g',s}^{global} * mean_L_{g'}^{global})
    # Note: for every gene g in counts_scaled, counts_scaled[s, g] = R_{g,s}^{global} * mean_L_g^{global} * alpha_s.
    # Extract transcript x sample global and zoom counts on gset to compute consistent length-scaled zoomed counts.
    gset_arr = pa.array(list(set(gset)), pa.string())

    def _tx_sub(tbl: pa.Table) -> pd.DataFrame:
        bt = tbl.column("biotype")
        keep = pc.and_(
            pc.and_(pc.invert(pc.starts_with(bt, "decoy")), pc.invert(pc.is_in(bt, value_set=pa.array(list(exclude_biotypes), pa.string())))),
            pc.is_in(tbl.column("group_id"), value_set=gset_arr),
        )
        sub = tbl.filter(keep).select(["sample", "target_id", "effective_count"])
        if sub.num_rows == 0:
            return pd.DataFrame(0.0, index=pd.Index([], name="target_id"), columns=samples)
        df = pd.DataFrame({
            "sample": sub.column("sample").to_pylist(),
            "target_id": sub.column("target_id").to_pylist(),
            "eff": sub.column("effective_count").to_numpy(zero_copy_only=False),
        })
        return df.groupby(["target_id", "sample"])["eff"].sum().unstack("sample", fill_value=0.0).reindex(columns=samples, fill_value=0.0)

    tx_g = _tx_sub(table_global)
    tx_z = _tx_sub(ztable)
    all_tids = sorted(set(tx_g.index) | set(tx_z.index))
    if not all_tids:
        return counts_scaled, combined_scaled, avar_comb

    tx_g = tx_g.reindex(index=all_tids, columns=samples, fill_value=0.0)
    tx_z = tx_z.reindex(index=all_tids, columns=samples, fill_value=0.0)
    ds_vec = depth_scale.reindex(samples).fillna(1.0).values.astype(float)

    eff_lens = np.array([
        max(1.0, catalog.targets[tid].length - frag_mean) if catalog.targets[tid].length > frag_mean else max(1.0, catalog.targets[tid].length / 2.0)
        for tid in all_tids
    ], dtype=float)
    t2g = pd.Series([catalog.targets[tid].group_id for tid in all_tids], index=pd.Index(all_tids, name="target_id"))

    # Global rate * mean_L on gset to recover exact per-sample global factor alpha_s (defaults to 1.0 if gset had 0 global counts)
    rate_g = tx_g.div(eff_lens, axis=0)
    g_rate_g = rate_g.groupby(t2g).sum()
    g_cnt_g = tx_g.groupby(t2g).sum()
    w_len_g = tx_g.groupby(t2g).sum()  # rate_g * eff_lens == tx_g
    fallback_len = pd.Series(eff_lens, index=all_tids).groupby(t2g).mean()
    g_len_g = (w_len_g / g_rate_g.replace(0.0, np.nan)).T.fillna(fallback_len).T
    mean_len_g = g_len_g.mean(axis=1)
    unnorm_g = g_rate_g.mul(mean_len_g, axis=0)  # group x sample

    # Recover alpha_s from non-zero entries of counts_scaled (which covers the whole transcriptome); fallback to 1.0
    # Using any expressed gene in gset gives the exact scalar alpha_s used by tximport_gene_level on table_global!
    alpha_s = np.ones(len(samples), dtype=float)
    for si, s in enumerate(samples):
        denom = float(unnorm_g[s].sum())
        numer = float(counts_scaled.loc[s, unnorm_g.index].sum()) if len(unnorm_g.index) else 0.0
        if denom > 1e-9 and numer > 0.0:
            alpha_s[si] = numer / denom
        else:
            tot_raw = float(raw_counts.loc[s].sum())
            tot_scl = float(counts_scaled.loc[s].sum())
            alpha_s[si] = (tot_scl / tot_raw) if tot_raw > 0 else 1.0

    # Depth-normalized counts determine unbiased isoform effective lengths across samples; raw (global+zoom) counts carry zoom precision
    tx_raw_comb = tx_g + tx_z
    tx_norm_comb = tx_raw_comb.mul(ds_vec, axis=1)
    rate_norm = tx_norm_comb.div(eff_lens, axis=0)
    g_rate_norm = rate_norm.groupby(t2g).sum()
    g_len_comb = (tx_norm_comb.groupby(t2g).sum() / g_rate_norm.replace(0.0, np.nan)).T.fillna(fallback_len).T
    mean_len_comb = g_len_comb.mean(axis=1)

    g_rate_raw_comb = tx_raw_comb.div(eff_lens, axis=0).groupby(t2g).sum()
    scaled_zoom_genes = g_rate_raw_comb.mul(mean_len_comb, axis=0).mul(alpha_s, axis=1).T  # sample x group
    for g in scaled_zoom_genes.columns:
        if g in combined_scaled.columns:
            combined_scaled[g] = scaled_zoom_genes[g].reindex(samples).fillna(0.0)
    return counts_scaled, combined_scaled, avar_comb


def refit_genes_with_depth_offsets(
    de: pd.DataFrame,
    counts_global: pd.DataFrame,
    meta: pd.DataFrame,
    contrast: list[str],
    genes: list[str],
    zoomed_counts: pd.DataFrame,
    depth_scale: pd.Series,
    avar_zoomed: pd.DataFrame | None = None,
    design: str = "~condition",
    disp: pd.Series | None = None,
    sf: pd.Series | None = None,
) -> pd.DataFrame:
    """Replace the DE rows of ``genes`` using (global + zoom) counts with per-sample depth offsets.

    depth_scale[s] = global_bytes_s / (global_bytes_s + zoom_bytes_s), i.e. the zoomed gene was
    observed at 1/depth_scale times the global depth in sample s.  Dispersions and size factors
    come from the global PyDESeq2 fit under ``design``; BH is recomputed over all genes.
    """
    factor, lvl_a, lvl_b = contrast
    if disp is None or sf is None:
        _res, disp, sf = run_pydeseq2(counts_global, meta, contrast, design=design)
    samples = list(counts_global.index)
    meta_s = meta.loc[samples]
    cond = meta_s[factor].astype(str)
    is_a = (cond == str(lvl_a)).values
    is_b = (cond == str(lvl_b)).values
    X, c_vec, keep = _design_matrix_and_contrast(meta_s, design, contrast)
    out = de.set_index("group_id").copy()
    for g in genes:
        if g not in zoomed_counts.columns or g not in out.index:
            continue
        y = zoomed_counts.loc[samples, g].values.astype(float)
        off = np.log(sf.reindex(samples).values / depth_scale.reindex(samples).fillna(1.0).values)
        alpha = float(disp.get(g, np.nan))
        if not np.isfinite(alpha):
            alpha = float(np.nanmedian(disp.values)) if len(disp) else 0.1
        beta, _se, cov = nb_glm_wald(y[keep], X[keep], off[keep], alpha, return_cov=True)
        lfc = float((c_vec @ beta) / math.log(2.0))
        se_l = float(math.sqrt(max(0.0, float(c_vec @ cov @ c_vec))) / math.log(2.0))
        pv = float(out.loc[g, "var_placement_ambiguity"]) if pd.notna(out.loc[g, "var_placement_ambiguity"]) else 0.0
        if avar_zoomed is not None and g in avar_zoomed.columns and is_a.any() and is_b.any():
            norm_y = y / np.exp(off)
            v = avar_zoomed.loc[samples, g].values / np.exp(off) ** 2
            ma = norm_y[is_a].mean() + 0.5
            mb = norm_y[is_b].mean() + 0.5
            pv = float((v[is_a].sum() / is_a.sum() ** 2 / ma ** 2 + v[is_b].sum() / is_b.sum() ** 2 / mb ** 2) / LN2_SQ)
        se_tot = math.sqrt(se_l ** 2 + pv)
        z = abs(lfc / se_tot) if se_tot > 0 else 0.0
        out.loc[g, ["log2fc", "lfc_se_deseq2", "var_placement_ambiguity", "lfc_se_total"]] = [lfc, se_l, pv, se_tot]
        out.loc[g, ["ci95_low", "ci95_high"]] = [lfc - 1.96 * se_tot, lfc + 1.96 * se_tot]
        out.loc[g, "pvalue"] = float(2.0 * norm.sf(z))
        out.loc[g, "streamed_reads"] = float(y[is_a | is_b].sum())
        out.loc[g, "base_mean"] = float(np.mean(y[keep] / np.exp(off[keep])))
        out.loc[g, "status"] = "zoomed"
    valid = out["pvalue"].notna() & out["status"].isin(["DE", "non_DE", "zoomed", "pending", "low_depth"])
    if valid.any():
        _, q, _, _ = multipletests(out.loc[valid, "pvalue"].values, method="fdr_bh")
        out.loc[valid, "padj"] = q
    zoomed = out["status"] == "zoomed"
    out.loc[zoomed & (out["padj"] < 0.05), "status"] = "DE_after_zoom"
    out.loc[zoomed & ~(out["padj"] < 0.05), "status"] = "non_DE_after_zoom"
    return out.reset_index()
