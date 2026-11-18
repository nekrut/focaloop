"""MalariaVar whole-genome streaming engine: per-isolate pipeline + checkpointed cohort runner.

    python engine.py run --manifest manifests/pf8_truth_200.parquet --cohort $W/truth_200_stream --workers 16
    python engine.py one --manifest manifests/pf8_truth_200.parquet --sample PA0409-C --cohort $W/debug

Per isolate: source (ODP xsra -Q / ENA FASTQ / Sanger CRAM) -> host sieve + spot subsampler -> strobealign
-> numba pileup accumulators -> stop gate every 250,000 spots -> whole-genome calls -> zstd Parquet state.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import caller as C
from . import genome as G
from . import kmer_zoom as Z
from . import prepare as P
from . import state_store as SS
from . import stop_gate as SG
from . import streamer as ST

import tempfile

HERE = Path(__file__).resolve().parent
W = Path(os.environ.get("W") or tempfile.gettempdir())
_SCRATCH_BASE = os.environ.get("MALARIAVAR_SCRATCH") or (str(Path(tempfile.gettempdir()) / f"focaloop_mvar_{os.getuid()}"))
LOGDIR = Path(os.environ.get("MALARIAVAR_LOGDIR") or Path(_SCRATCH_BASE) / "logs" / "engine")
CHUNK = 4 << 20
ENGINE_VERSION = "1.1"


def _cpu():
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime, c.ru_utime + c.ru_stime


def _runs(row):
    r = row.get("ena_runs")
    return [str(x) for x in r] if r is not None else []


def tournament_mapper():
    f = HERE / "results" / "stage1_tournament.json"
    if f.exists():
        return json.loads(f.read_text()).get("winner", "strobealign")
    return "strobealign"


def prior_p(row, target):
    d = row.get("mean_depth_est")
    h = row.get("unmapped_byte_frac")
    if d is None or not np.isfinite(d) or d <= 0:
        return 0.35
    pf = d * (1 - (h if h is not None and np.isfinite(h) else 0.1))
    return float(np.clip(1.25 * target / max(pf, 1e-6), 0.02, 1.0))


def zoom_mask(g):
    m = np.zeros(g.L, bool)
    for k in G.ZOOM_GENES:
        gg = g.gene(k)
        m[g.goff(gg["contig"], gg["start"]) - 300: g.goff(gg["contig"], gg["end"]) + 300] = True
    return m


def detect_order_mode(ccount, lengths, binbases=None, min_primary_bp=100_000):
    """Classify stream order as 'sorted' or 'flowcell' for any reference karyotype (1 to 10,000+ contigs).

    Replaces the hardcoded `ccount[:14]` P. falciparum assumption:
      - Filters to primary nuclear contigs (`lengths >= min_primary_bp`, or all contigs if none qualify).
      - On multi-contig references, compares observed contig read share `p_obs` against length-proportional
        random/flowcell expectation `p_exp` via total variation distance `tvd` and max 2-adjacent-contig share
        (handling slices that straddle a chromosome boundary such as chr19 -> chr20).
      - On single-contig references (e.g. standalone chr20), checks spatial concentration of covered 1-kb bins.
    """
    cc = np.asarray(ccount, dtype=np.float64)
    ln = np.asarray(lengths, dtype=np.float64)
    pri = ln >= min_primary_bp
    if not pri.any():
        pri = np.ones(len(ln), dtype=bool)
    cc_p = cc[pri]
    ln_p = ln[pri]
    tot = float(cc_p.sum())
    if tot <= 0:
        return "flowcell"
    if len(cc_p) == 1 or float(ln_p.max() / max(ln_p.sum(), 1.0)) > 0.90:
        if binbases is not None and len(binbases) >= 200:
            bb = np.asarray(binbases, dtype=np.float64)
            b_tot = float(bb.sum())
            if b_tot > 0:
                win = max(1, len(bb) // 4)
                csum = np.concatenate([[0.0], np.cumsum(bb)])
                max_q = float((csum[win:] - csum[:-win]).max() / b_tot)
                cov_frac = float((bb > 0).mean())
                if max_q > 0.80 and cov_frac < 0.50:
                    return "sorted"
        return "flowcell"
    p_obs = cc_p / tot
    p_exp = ln_p / float(ln_p.sum())
    tvd = 0.5 * float(np.abs(p_obs - p_exp).sum())
    adj2_obs = float(max(p_obs.max(), (p_obs[:-1] + p_obs[1:]).max()))
    adj2_exp = float(max(p_exp.max(), (p_exp[:-1] + p_exp[1:]).max()))
    if tvd > 0.35 and adj2_obs > max(0.50, min(0.95, 2.0 * adj2_exp)):
        return "sorted"
    return "flowcell"


def run_sample(row, cohort, cfg):
    """stream + call one isolate; writes its FocaloopState tables and returns telemetry."""
    sample = row["Sample"]
    t0 = time.time()
    cpu0 = _cpu()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = open(LOGDIR / f"{sample}.log", "ab")
    g = G.load()
    panel = P.load_panel()
    use_pla_sieve = bool(cfg.get("use_pla_sieve", False))
    kset = P.load_pla_sieve(cfg.get("pla_sieve_path")) if use_pla_sieve else P.load_kset()
    cp = C.CallParams(min_bq=cfg.get("min_bq", 13), min_mapq=cfg.get("min_mapq", 20),
                      max_mm_frac=cfg.get("max_mm_frac", C.CallParams().max_mm_frac),
                      use_omnitig_zoom=bool(cfg.get("use_omnitig_zoom", False)))
    gp = SG.GateParams(slice_spots=cfg.get("slice_spots", 250_000), min_depth=cfg.get("min_depth", 22.0),
                       min_depth_poly=cfg.get("min_depth_poly", 32.0),
                       use_island_pc_gate=bool(cfg.get("use_island_pc_gate", False)))
    acc = C.Accumulator(g, cp, zoom_mask=zoom_mask(g))
    gate = SG.StopGate(g, panel, gp)
    src = ST.make_source(row, xsra_threads=cfg.get("xsra_threads", 2), log=log, prefer=cfg.get("prefer"),
                         stage_dir=STAGE, decode_sem=_DECODE_SLOTS, prefetch_dir=cfg.get("prefetch_dir"))
    if src.kind != "aws_odp_xsra":
        # staged ODP objects are already on the wire; CRAM / ENA FASTQ are coordinate-sorted and need the whole file
        gate.p.max_wire_bytes = float("inf")
    srf = bool(row.get("order_hint") == "flowcell")
    sieve = ST.Sieve(kset, paired=src.paired, need=int(cfg.get("depletion_need", 2)), use_pla_sieve=use_pla_sieve)
    sieve.p = 1.0 if srf else prior_p(row, gp.target_depth)
    if cfg.get("p_fixed"):
        sieve.p = float(cfg["p_fixed"])
    p_log = [(0, sieve.p)]
    mapper_name = cfg.get("mapper") or "strobealign"
    mapper = subprocess.Popen(ST.mapper_cmd(cfg.get("map_threads", 2), src.paired, mapper=mapper_name),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0)
    tees = {}
    if cfg.get("save_dir"):  # dev set: sieved FASTQ (mapper input) and SAM (mapper output), zstd
        sd = Path(cfg["save_dir"])
        sd.mkdir(parents=True, exist_ok=True)
        for kind in ("fq", "sam"):
            tees[kind] = subprocess.Popen(["zstd", "-q", "-3", "-T2", "-f", "-o", str(sd / f"{sample}.{kind}.zst")],
                                          stdin=subprocess.PIPE, stderr=log)
    stop = threading.Event()
    bounds = []                 # (slice k, last emitted spot index in slice k, p used for slice k)
    lock = threading.Lock()
    slice_cv = threading.Condition(lock)
    k_done_ref = [-1]
    bytes_ahead = [0]
    qc = {}
    err = []
    wave0_buf = bytearray()
    wave0_done = [False]

    def _maybe_wave0(out):
        if not gp.use_island_pc_gate or wave0_done[0] or not out:
            return
        need = 512 * 1024 - len(wave0_buf)
        if need > 0:
            wave0_buf.extend(out[:need])
        if len(wave0_buf) >= 256 * 1024:
            wave0_done[0] = True
            gate.assess_wave0_pc(bytes(wave0_buf))

    def _write_out(out):
        if not out:
            return
        _maybe_wave0(out)
        mapper.stdin.write(out)
        if "fq" in tees:
            tees["fq"].stdin.write(out)
        with lock:
            bytes_ahead[0] += len(out)

    def feeder():
        k_next = gp.slice_spots
        first = True
        try:
            while not stop.is_set():
                data = src.read(CHUNK)
                if not data:
                    break
                out = sieve.push(data, max_spots=k_next)
                if first and sieve.stats[0] >= 1000:
                    ok, sym = sieve.quality_check()
                    qc.update(real_quals=ok, symbols="".join(sym))
                    first = False
                    if not ok:
                        err.append("flat qualities (SRA-lite) on first block")
                        stop.set()
                        break
                _write_out(out)
                while not stop.is_set() and sieve.stats[0] >= k_next:
                    with slice_cv:
                        k_idx = len(bounds)
                        bounds.append((k_idx, int(sieve.stats[8]), float(sieve.p)))
                        while (not stop.is_set() and k_idx - k_done_ref[0] > 1
                               and bytes_ahead[0] >= (4 << 20) and mapper.poll() is None):
                            slice_cv.wait(timeout=0.05)
                    k_next += gp.slice_spots
                    if stop.is_set() or not sieve.carry:
                        break
                    out = sieve.push(b"", max_spots=k_next)
                    _write_out(out)
            if not stop.is_set():
                while not stop.is_set() and sieve.carry:
                    out = sieve.push(b"", final=True, max_spots=k_next)
                    _write_out(out)
                    if sieve.stats[0] >= k_next:
                        with slice_cv:
                            bounds.append((len(bounds), int(sieve.stats[8]), float(sieve.p)))
                        k_next += gp.slice_spots
                    else:
                        break
            if gp.use_island_pc_gate and not wave0_done[0] and wave0_buf:
                wave0_done[0] = True
                gate.assess_wave0_pc(bytes(wave0_buf))
        except BrokenPipeError:
            if not stop.is_set():
                err.append("feeder: mapper stdin closed unexpectedly (BrokenPipeError)")
        except Exception as e:  # source/network failure surfaces as an error row
            err.append(f"feeder: {type(e).__name__}: {e}")
        finally:
            src.close(early=stop.is_set())
            try:
                mapper.stdin.close()
            except Exception:
                pass

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    timed_out = threading.Event()
    timeout_s = float(cfg.get("sample_timeout_s") or 0)
    watchdog = None
    if timeout_s > 0:
        def _on_timeout():
            timed_out.set()
            stop.set()
            src.close(early=True)
            try:
                mapper.kill()
            except Exception:
                pass
        watchdog = threading.Timer(timeout_s, _on_timeout)
        watchdog.daemon = True
        watchdog.start()
    bin_p = np.full(len(g.bin_gc), np.nan)  # sampling rate in force when each 1 kb bin was first covered
    gate.bin_p = bin_p
    k_done = -1
    stop_reason = "archive_end"
    order_mode = None
    gate_rows = []
    carry = b""
    try:
        while True:
            data = mapper.stdout.read(CHUNK)
            if not data:
                break
            if "sam" in tees:
                tees["sam"].stdin.write(data)
            buf = carry + data
            arr = np.frombuffer(buf, np.uint8)
            used = acc.feed(arr, len(arr))
            carry = buf[used:]
            with lock:
                pending = [b for b in bounds if b[0] > k_done]
            for k, last_spot, slice_p in pending:
                if last_spot >= 0 and acc.stats[4] < last_spot:
                    break
                k_done = k
                newly = (acc.binbases > 0) & np.isnan(bin_p)
                bin_p[newly] = slice_p
                if order_mode is None:
                    order_mode = detect_order_mode(acc.ccount, g.lengths, acc.binbases)
                    if (order_mode == "flowcell" and src.kind == "aws_odp_xsra" and sieve.p < 1.0
                            and not cfg.get("p_fixed")):
                        with lock:
                            sieve.p = 1.0
                        p_log.append((k, 1.0))
                stop_now, why, m = gate.evaluate(acc, k, src.wire_bytes())
                m.update(spots=int(sieve.stats[0]), kept=int(sieve.stats[1]), pf=int(sieve.stats[2]), p=slice_p,
                         t=time.time() - t0)
                gate_rows.append({kk: vv for kk, vv in m.items() if not isinstance(vv, dict)})
                if (order_mode == "sorted" or src.kind != "aws_odp_xsra") and not cfg.get("p_fixed"):
                    p_new = gate.adapt_p(sieve.p, acc)
                    if p_new != sieve.p:
                        with lock:
                            sieve.p = p_new
                        p_log.append((k, p_new))
                if stop_now and not stop.is_set():
                    stop_reason = why
                    stop.set()
                    threading.Thread(target=src.close, kwargs={"early": True}, daemon=True).start()
                with slice_cv:
                    k_done_ref[0] = k_done
                    bytes_ahead[0] = 0
                    slice_cv.notify_all()
    finally:
        if watchdog is not None:
            watchdog.cancel()
        with slice_cv:
            slice_cv.notify_all()
    if timed_out.is_set():
        err.append(f"sample timeout after {timeout_s:.0f}s")
    if carry:
        arr = np.frombuffer(carry + b"\n", np.uint8)
        acc.feed(arr, len(arr))
    rc_map = mapper.wait()
    th.join(timeout=60)
    if rc_map != 0 and not (stop.is_set() and rc_map in (-15, -13, -9, 141, 143)):
        err.append(f"mapper ({mapper_name}) exited with code {rc_map}")
    for kind, tp in tees.items():
        try:
            tp.stdin.close()
        except Exception:
            pass
        rc_tee = tp.wait()
        if rc_tee != 0:
            err.append(f"tee ({kind}) exited with code {rc_tee}")
    if src.errors:
        err.extend(src.errors)
    if sieve.stats[0] == 0 and not err:
        err.append("zero spots read from source")
    if tees:
        (Path(cfg["save_dir"]) / f"{sample}.p.json").write_text(json.dumps(dict(p_log=p_log, p_final=sieve.p)))
    t_stream = time.time() - t0
    wire = src.wire_bytes()
    if ((sieve.stats[0] == 0 or err) and src.kind == "aws_odp_xsra" and not cfg.get("_no_fallback")
            and not any("flat qualities" in e for e in err)):
        # old or transiently failing xsra HTTPS runs: fall back to the staged normalised ODP object
        log.close()
        return run_sample(row, cohort, dict(cfg, prefer="odp_stage", _no_fallback=True))
    if cfg.get("dump_acc"):
        acc.dump(Path(cohort) / "acc" / sample)
        np.save(Path(cohort) / "acc" / sample / "bin_p.npy", bin_p)
    if err:
        stop_reason = "error"
    # ---------------------------------------------------------------- Stage 3 + 5 finalisation
    t1 = time.time()
    m_final = gate.metrics(acc, cnv_loci=False)
    med = m_final["median_depth"]
    w, d = C.wsaf_at(acc, panel)
    fws, n_fws = C.fws(w, d, panel["af"])
    clonal = bool(np.isfinite(fws) and fws >= cp.clonal_fws)
    snv = C.call_snvs(g, acc, cp, clonal=clonal, fws=fws, panel_goffs=panel["goff"])
    ind = C.call_indels(g, acc, cp, clonal=clonal, fws=fws)
    zmeta, zbuf = acc.zoom_reads()
    zoom_rows = Z.zoom_genes(g, zmeta, zbuf, ind, use_omnitig_zoom=cp.use_omnitig_zoom)
    bin_p[(acc.binbases > 0) & np.isnan(bin_p)] = sieve.p
    codons, k13, cnv, depth, ratio, bin_med = C.curated_calls(g, acc, DRM_CODONS, bin_p, cp,
                                                              sample_type=row.get("Sample_type") or "gDNA")
    codons["k13_ns"] = k13
    runs = C.callable_runs(g, acc, cp)
    t_call = time.time() - t1
    # ---------------------------------------------------------------- tables
    rows = [snv] + ([ind] if ind is not None else [])
    vc = {k: np.concatenate([r[k] for r in rows]) for k in ("goff", "kind", "ref", "alt", "alen", "ad_ref",
                                                            "ad_alt", "dp", "wsaf", "qual", "gt")}
    store_all = cfg.get("store_candidates", True)
    keep = (vc["gt"] > 0) | (vc["qual"] >= (0 if store_all else 10))
    vc = {k: v[keep] for k, v in vc.items()}
    ci, pos, refs, alts = C.vcf_alleles(g, vc, ins_seqs=acc.ins_seqs)
    zmap = {}
    for z in zoom_rows:
        zmap[(int(z["goff"]), z["ref"], z["alt"])] = z["status"]
    zflag = np.array([{"confirmed": 1, "rejected": -1, "unresolved": 0}.get(
        zmap.get((int(a), r, al), ""), 0) for a, r, al in zip(vc["goff"], refs, alts)], np.int8)
    variant_calls = pd.DataFrame(dict(sample=sample, chrom=ci, pos=pos, ref=refs, alt=alts, kind=vc["kind"],
                                      gt=vc["gt"], ad_ref=vc["ad_ref"].astype(np.int32),
                                      ad_alt=vc["ad_alt"].astype(np.int32), dp=vc["dp"].astype(np.int32),
                                      wsaf=vc["wsaf"], qual=vc["qual"], core=g.core[vc["goff"]], zoom=zflag))
    extra = [z for z in zoom_rows if z["status"] == "assembly_only"]
    if extra:  # alleles only the local assembly sees
        c_, p_ = g.locate(np.array([z["goff"] for z in extra]))
        variant_calls = pd.concat([variant_calls, pd.DataFrame(dict(
            sample=sample, chrom=c_.astype(np.int16), pos=p_.astype(np.int32), ref=[z["ref"] for z in extra],
            alt=[z["alt"] for z in extra], kind=np.int8(3), gt=np.int8(1), ad_ref=np.int32(0),
            ad_alt=np.array([z["kmer_support"] for z in extra], np.int32), dp=np.int32(0), wsaf=np.float32(np.nan),
            qual=np.float32(0), core=g.core[[z["goff"] for z in extra]], zoom=np.int8(2)))], ignore_index=True)
    wu8 = np.where(d >= 3, np.round(np.nan_to_num(w) * 200), 255).astype(np.uint8)
    du8 = np.minimum(d, 254).astype(np.uint8)
    wsaf_sketch = pd.DataFrame(dict(sample=[sample], n_sites=[len(w)], wsaf_u8=[wu8.tobytes()],
                                    dp_u8=[du8.tobytes()]))
    depth_bins = pd.DataFrame(dict(sample=[sample], depth_f32=[depth.astype(np.float32).tobytes()],
                                   ratio_f16=[ratio.astype(np.float16).tobytes()], bin_median=[bin_med]))
    cnv_df = pd.DataFrame([dict(sample=sample, **c) for c in cnv])
    c_r, p_r = g.locate(runs[:, 0]) if len(runs) else (np.zeros(0, int), np.zeros(0, int))
    callable_df = pd.DataFrame(dict(sample=sample, chrom=c_r.astype(np.int16), start=p_r.astype(np.int32),
                                    end=(p_r + (runs[:, 1] - runs[:, 0]) - 1).astype(np.int32)))
    cpu1 = _cpu()
    core_callable = float(sum(g.core[a:b].sum() for a, b in runs) / g.core.sum()) if len(runs) else 0.0
    tel = dict(sample=sample, engine=ENGINE_VERSION, mapper=mapper_name, source=src.kind,
               order_mode=order_mode or "unknown",
               order_hint=row.get("order_hint"), runs=",".join(_runs(row)),
               stop_reason=stop_reason, error="; ".join(err) if err else None,
               slices=k_done + 1, spots_seen=int(sieve.stats[0]), spots_kept=int(sieve.stats[1]),
               spots_pf=int(sieve.stats[2]), bases_seen=int(sieve.stats[3]), bases_mapped_in=int(sieve.stats[4]),
               inline_barcodes_skipped=int(sieve.stats[9]),
               host_frac=float(1 - sieve.stats[2] / max(sieve.stats[1], 1)), p_final=float(sieve.p),
               p_log=json.dumps(p_log), read_len_max=int(sieve.stats[6]), real_quals=qc.get("real_quals"),
               qual_symbols=qc.get("symbols"), wire_bytes=float(wire),
               archive_bytes=float(row.get("odp_bytes") if src.kind.startswith("aws_odp") else
                                   (row.get("fastq_bytes") if src.kind == "ena_fastq" else row.get("cram_bytes"))
                                   or np.nan),
               sam_records=int(acc.stats[0]), reads_counted=int(acc.stats[2]), bases_counted=int(acc.stats[3]),
               mm_filtered=int(acc.stats[5]), overlap_bases_skipped=int(acc.stats[7]), zoom_reads=int(acc.stats[8]),
               pcr_dups_skipped=int(acc.stats[10]),
               median_core_depth=med, breadth10=m_final["breadth10"], core_callable_frac=core_callable,
               fws=fws, fws_sites=n_fws, clonal_policy=clonal, sample_type=row.get("Sample_type"),
               faceaway_pairs=int(acc.rf_n[0]), prefetched=int(getattr(src, "prefetched", 0)),
               direct_fetches=int(getattr(src, "direct_fetches", 0)), trimmed_bytes=int(getattr(src, "trimmed_bytes", 0)), fws_prov=gate.st.fws_prov, depth_full_est=gate.st.depth_full,
               n_snv=int(((variant_calls["kind"] == 0) & (variant_calls["gt"] > 0)).sum()),
               n_indel=int(((variant_calls["kind"].isin([1, 2])) & (variant_calls["gt"] > 0)).sum()),
               wall_s=time.time() - t0, stream_s=t_stream, call_s=t_call,
               cpu_self_s=cpu1[0] - cpu0[0], cpu_children_s=cpu1[1] - cpu0[1],
               peak_scratch_bytes=int(getattr(src, "peak", 0)), t_download=float(getattr(src, "t_download", 0.0)),
               t_decode_wait=float(getattr(src, "t_decode_wait", 0.0)),
               use_omnitig_zoom=cp.use_omnitig_zoom, use_pla_sieve=use_pla_sieve,
               use_island_pc_gate=gp.use_island_pc_gate,
               pc_route_status=(gate.st.pc_summary or {}).get("status") if gate.st.pc_summary else None,
               pc_est=(gate.st.pc_summary or {}).get("pc_est") if gate.st.pc_summary else None,
               pc_ci_halfwidth=(gate.st.pc_summary or {}).get("pc_ci_halfwidth") if gate.st.pc_summary else None,
               k13_ns=k13,
               **{f"aa_{k}": v for k, v in codons.items()})
    ok_sample = (stop_reason != "error") and not err
    nbytes = SS.write_sample(cohort, sample, dict(
        variant_calls=variant_calls, wsaf_sketch=wsaf_sketch, depth_1kb_bins=depth_bins, cnv_del_calls=cnv_df,
        sample_telemetry=pd.DataFrame([tel]), callable_mask=callable_df,
        slice_log=pd.DataFrame(gate_rows).assign(sample=sample) if gate_rows else None,
        zoom_calls=pd.DataFrame(zoom_rows).assign(sample=sample) if zoom_rows else None), ok=ok_sample)
    tel["state_bytes"] = nbytes
    log.close()
    return tel


DRM_CODONS = [("crt", c) for c in (72, 73, 74, 75, 76, 93, 97, 145, 218, 220, 271, 326, 333, 343, 350, 353, 356, 371)] + \
             [("dhfr", c) for c in (16, 51, 59, 108, 164, 306)] + \
             [("dhps", c) for c in (436, 437, 540, 581, 613)] + \
             [("mdr1", c) for c in (86, 184, 1034, 1042, 1226, 1246)] + \
             [("kelch13", c) for c in (441, 446, 449, 458, 469, 476, 481, 493, 515, 527, 537, 538, 539, 543, 553,
                                      561, 568, 574, 578, 580, 622, 675)] + \
             [("exo", 415), ("arps10", 127), ("arps10", 128), ("fd", 193), ("mdr2", 484)]


# ------------------------------------------------------------------------------------------- cohort runner
_CRAM_SLOTS = None
_DECODE_SLOTS = None
# transient staged ODP objects: NVMe, deleted as soon as each run is decoded (cohort outputs stay on $W)
STAGE = Path(os.environ.get("MALARIAVAR_STAGE_DIR") or Path(_SCRATCH_BASE) / "tmp_stage" / "stream")


def _init(sem, dsem=None):
    global _CRAM_SLOTS, _DECODE_SLOTS
    _CRAM_SLOTS = sem
    _DECODE_SLOTS = dsem


def _worker(args):
    row, cohort, cfg = args
    sanger = not bool(row.get("all_odp")) and not bool(row.get("all_paired_fastq"))
    for attempt in range(cfg.get("retries", 2) + 1):
        try:
            if sanger and _CRAM_SLOTS is not None:
                with _CRAM_SLOTS:  # Sanger serves ~6-8 MB/s in total: at most a few CRAM streams at once
                    tel = run_sample(row, cohort, cfg)
            else:
                tel = run_sample(row, cohort, cfg)
            if tel["stop_reason"] != "error" or attempt == cfg.get("retries", 2):
                return tel
        except Exception as e:
            tb = traceback.format_exc()
            if attempt == cfg.get("retries", 2):
                return dict(sample=row["Sample"], stop_reason="crash", error=f"{type(e).__name__}: {e}", tb=tb)
        time.sleep(10 * (attempt + 1))
    return dict(sample=row["Sample"], stop_reason="crash")


def load_manifest(path):
    m = pd.read_parquet(path)
    if "order_hint" in m.columns:  # precomputed (shard manifests ship without ena_runs.parquet)
        return m
    ena_path = G.default_staging_dir() / "data" / "ena_runs.parquet"
    if not ena_path.exists():
        raise FileNotFoundError(
            f"Missing {ena_path}. Set S or FOCALOOP_PF3D7_DIR to a staging directory containing data/ena_runs.parquet, "
            "or include an order_hint column in the manifest."
        )
    e = pd.read_parquet(ena_path, columns=["run_accession", "submitted_format"]).set_index("run_accession")
    fmt = m.ena_runs.apply(lambda r: [str(e.submitted_format.get(x, "")).split(";")[0] for x in (list(r) if r is not None else [])])
    m["order_hint"] = fmt.apply(lambda f: "flowcell" if f and all(x == "SRF" for x in f) else "sorted")
    return m


def run_cohort(manifest, cohort, workers, cfg, shard_size=1000, limit=None, samples=None, shard_base=0):
    cohort = Path(cohort)
    cohort.mkdir(parents=True, exist_ok=True)
    m = load_manifest(manifest)
    if samples:
        m = m[m.Sample.isin(samples)]
    if limit:
        m = m.head(limit)
    m = m.reset_index(drop=True)
    m["shard"] = shard_base + np.arange(len(m)) // shard_size  # distinct bases let several hosts' shards merge
    if not (cohort / "manifest.parquet").exists():
        m[["Sample", "Country", "Population", "Year", "shard", "order_hint"]].to_parquet(cohort / "manifest.parquet")
    done = SS.done_samples(cohort)
    todo = m[~m.Sample.isin(done)]
    # ODP queue: the `head_small` smallest objects first, so every worker starts mapping within the first
    # minute while larger objects prefetch behind them; then manifest order
    odp = todo[todo.all_odp.astype(bool)]
    slow = todo[~todo.all_odp.astype(bool)]
    hs = min(cfg.get("head_small", 16), len(odp))
    if hs:
        head = odp.odp_bytes.astype(float).fillna(np.inf).nsmallest(hs).index
        odp = pd.concat([odp.loc[head], odp.drop(head)])
    log = open(cohort / "run_log.jsonl", "a")
    t0 = time.time()
    rx0 = net_rx()
    print(f"{time.strftime('%H:%M:%S')} cohort {cohort} todo {len(todo)} (odp {len(odp)}, other {len(slow)}) "
          f"done {len(done)}", flush=True)
    nic_stop = threading.Event()

    def nic_sampler(path=cohort / f"nic_rx_{int(t0)}.tsv", every=5.0):
        with open(path, "w") as f:
            f.write("t_s\trx_bytes\n")
            while True:
                f.write(f"{time.time() - t0:.1f}\t{net_rx() - rx0}\n")
                f.flush()
                if nic_stop.wait(every):
                    break
    threading.Thread(target=nic_sampler, daemon=True).start()
    # prefetcher process: staged ODP objects fetched in submission order on one warm connection pool
    pre_proc = None
    if cfg.get("prefetch_ahead", 0) > 0:
        pdir = STAGE / "prefetch" / f"{cohort.name}.{os.getpid()}"
        shutil.rmtree(pdir, ignore_errors=True)
        pdir.mkdir(parents=True)
        q = [r for _, x in odp.iterrows() for r in (ST.staged_runs(dict(x), cfg.get("prefer")) or [])]
        qf = pdir.parent / f"{pdir.name}.queue.json"
        qf.write_text(json.dumps(dict(runs=q, stop_file=str(pdir.parent / f"{pdir.name}.stop"),
                                      log_path=str(cohort / f"prefetch_{int(t0)}.jsonl"))))
        pre_proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve().parent / "streamer.py"),
                                     "prefetch", str(qf), str(pdir), str(cfg["prefetch_ahead"]),
                                     str(cfg.get("prefetch_in_flight", 6)), str(cfg.get("prefetch_streams", 32)),
                                     str(int(cfg.get("stage_max_gb", 0) * 1e9)), str(STAGE)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=open(LOGDIR / f"prefetch_{cohort.name}.log", "ab"), cwd=os.getcwd())
        (pdir / "prefetcher.pid").write_text(str(pre_proc.pid))  # before any worker looks for it
        cfg = dict(cfg, prefetch_dir=str(pdir))
    n = 0
    ntot = len(odp) + len(slow)
    results = queue.Queue()
    t_last = {}
    ctx = mp.get_context("fork")
    # Sanger CRAMs are bound by the remote server (~0.05 MB/s per connection): they get their own
    # `cram_slots` processes, so they never hold a compute worker while waiting for a slot
    pools = [ctx.Pool(workers, maxtasksperchild=8, initializer=_init,
                      initargs=(None, ctx.BoundedSemaphore(cfg["decode_slots"]) if cfg.get("decode_slots") else None))]
    if len(slow):
        pools.append(ctx.Pool(max(1, cfg.get("cram_slots", 2)), maxtasksperchild=8, initializer=_init,
                              initargs=(None, None)))
    for pool, part in zip(pools, (odp, slow)):
        for _, x in part.iterrows():
            pool.apply_async(_worker, ((dict(x), str(cohort), cfg),), callback=results.put,
                             error_callback=lambda e, s=x["Sample"]: results.put(
                                 dict(sample=s, stop_reason="crash", error=f"{type(e).__name__}: {e}")))
    try:
        while n < ntot:
            tel = results.get()
            n += 1
            tel = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in tel.items()}
            tel["t_done_s"] = time.time() - t0  # completion time within the cohort run
            t_last[str(tel.get("source"))] = tel["t_done_s"]
            log.write(json.dumps(tel, default=str) + "\n")
            log.flush()
            el = time.time() - t0
            print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} {tel.get('sample')} {tel.get('stop_reason')} "
                  f"src={tel.get('source')} mode={tel.get('order_mode')} depth={tel.get('median_core_depth', 0):.1f} "
                  f"wire={tel.get('wire_bytes', 0) / 1e6:.0f}MB wall={tel.get('wall_s', 0):.0f}s "
                  f"agg={(net_rx() - rx0) / max(el, 1) / 1e6:.1f}MB/s", flush=True)
            # seal any shard whose samples are all done
            done = SS.done_samples(cohort)
            for sid, grp in m.groupby("shard"):
                sh = cohort / "shards" / f"shard_{sid:03d}" / "SEALED"
                if not sh.exists() and grp.Sample.isin(done).all() and \
                        all(SS.is_done(cohort, s) for s in grp.Sample):
                    SS.seal_shard(cohort, int(sid), grp.Sample.tolist())
                    print(f"{time.strftime('%H:%M:%S')} sealed shard {sid}", flush=True)
    finally:
        for pool in pools:
            pool.close() if n >= ntot else pool.terminate()
        for pool in pools:
            pool.join()
        nic_stop.set()
    if pre_proc is not None:  # stop the prefetcher and drop anything it fetched that no worker consumed
        Path(pdir.parent / f"{pdir.name}.stop").touch()
        pre_proc.terminate()
        try:
            pre_proc.wait(30)
        except subprocess.TimeoutExpired:
            pre_proc.kill()
        shutil.rmtree(pdir, ignore_errors=True)
        for f in (pdir.parent / f"{pdir.name}.stop", pdir.parent / f"{pdir.name}.queue.json"):
            f.unlink(missing_ok=True)
    summary = dict(cohort=str(cohort), n=len(m), wall_s=time.time() - t0, nic_rx_bytes=net_rx() - rx0,
                   last_done_s_by_source=t_last)
    (cohort / f"run_summary_{int(t0)}.json").write_text(json.dumps(summary, indent=1))
    return summary


def net_rx():
    tot = 0
    for line in open("/proc/net/dev").read().splitlines()[2:]:
        name, rest = line.split(":", 1)
        if name.strip() != "lo":
            tot += int(rest.split()[0])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--shard-base", type=int, default=0, help="first shard id (one base per host: shards merge)")
    ap.add_argument("--stage-max-gb", type=float, default=0,
                    help="cap on staged ODP bytes (prefetched + held by workers); 0 = count cap only")
    ap.add_argument("--map-threads", type=int, default=2)
    ap.add_argument("--xsra-threads", type=int, default=2)
    ap.add_argument("--prefer")
    ap.add_argument("--no-candidates", action="store_true")
    ap.add_argument("--dump-acc", action="store_true", help="keep a compressed accumulator replay archive")
    ap.add_argument("--rerun", action="append", help="re-stream these samples even if done (unseals their shard)")
    ap.add_argument("--decode-slots", type=int, default=0,
                    help="cap on concurrent local decodes of staged archives (0 = none; NVMe staging needs none)")
    ap.add_argument("--save-dir", help="dev set: keep each sample's sieved FASTQ and mapper SAM (zstd) here")
    ap.add_argument("--prefetch-ahead", type=int, default=0,
                    help="staged ODP objects fetched ahead of the workers (0 = off; e.g. workers + 4)")
    ap.add_argument("--prefetch-in-flight", "--prefetch-parallel", dest="prefetch_in_flight", type=int, default=6,
                    help="objects downloading at once on the prefetch pool")
    ap.add_argument("--prefetch-streams", type=int, default=32, help="keep-alive connections in the prefetch pool")
    ap.add_argument("--head-small", type=int, default=16, help="smallest ODP objects queued first")
    ap.add_argument("--cram-slots", type=int, default=2, help="Sanger CRAM isolates streamed at once (own pool)")
    ap.add_argument("--p-fixed", type=float, help="fixed spot sampling rate (disables adaptive p)")
    ap.add_argument("--min-bq", type=int, default=13)
    ap.add_argument("--min-mapq", type=int, default=20)
    ap.add_argument("--max-mm-frac", type=float, default=0.20)
    ap.add_argument("--min-depth", type=float, default=22.0)
    ap.add_argument("--min-depth-poly", type=float, default=32.0)
    ap.add_argument("--omnitig-zoom", action="store_true",
                    help="Upgrade 1: Simple Omnitig + VarGeno QV local assembly & DRM haplotype rescue")
    ap.add_argument("--pla-sieve", action="store_true",
                    help="Upgrade 2: splitmix64 open syncmers + mmap PLA-index host sieve")
    ap.add_argument("--pla-path", help="path to prebuilt .pla index (default: prepare.PLA_SIEVE)")
    ap.add_argument("--island-pc-gate", action="store_true",
                    help="Upgrade 3: Blanca overlap-variance Island-PC Wave-0 routing + early stop gate")
    ap.add_argument("--dothash-knn", action="store_true",
                    help="Upgrade 4: record/use 4 KB DotHash kNN population sketches")
    ap.add_argument("--esscolor-panel", action="store_true",
                    help="Upgrade 4: record/use ESSColor-compressed DRM haplotype panel")
    ap.add_argument("--sample-timeout", type=float, default=0.0,
                    help="Per-isolate streaming timeout in seconds (0 = disabled)")
    a = ap.parse_args()
    ok = [str(W), os.environ.get("S", ""), os.environ.get("MALARIAVAR_SCRATCH", "")]
    for v in ("TMPDIR", "XDG_CACHE_HOME"):
        if not any(p and os.environ.get(v, "").startswith(p) for p in ok):
            sys.exit(f"{v} must point into $W, $S or $MALARIAVAR_SCRATCH, never the system default (source env.sh)")
    a.manifest = str(Path(a.manifest).resolve())
    a.cohort = str(Path(a.cohort).resolve())
    os.chdir(os.environ["TMPDIR"])
    cfg = dict(map_threads=a.map_threads, xsra_threads=a.xsra_threads, prefer=a.prefer,
               store_candidates=not a.no_candidates, mapper=tournament_mapper(), dump_acc=a.dump_acc,
               p_fixed=a.p_fixed, min_bq=a.min_bq, min_mapq=a.min_mapq, max_mm_frac=a.max_mm_frac, min_depth=a.min_depth,
               min_depth_poly=a.min_depth_poly, decode_slots=a.decode_slots, save_dir=a.save_dir,
               prefetch_ahead=a.prefetch_ahead, prefetch_in_flight=a.prefetch_in_flight,
               prefetch_streams=a.prefetch_streams, head_small=a.head_small, cram_slots=a.cram_slots,
               stage_max_gb=a.stage_max_gb, use_omnitig_zoom=a.omnitig_zoom,
               use_pla_sieve=a.pla_sieve, pla_sieve_path=a.pla_path,
               use_island_pc_gate=a.island_pc_gate,
               use_dothash_knn=a.dothash_knn, use_esscolor_panel=a.esscolor_panel,
               sample_timeout_s=a.sample_timeout)
    if a.cmd == "one":
        m = load_manifest(a.manifest)
        row = dict(m[m.Sample == a.sample[0]].iloc[0])
        tel = run_sample(row, a.cohort, cfg)
        print(json.dumps(tel, indent=1, default=str))
    else:
        if a.rerun:
            for sh in SS.sealed_shards(a.cohort):
                if set(json.loads((sh / "SEALED").read_text())["samples"]) & set(a.rerun):
                    SS.unseal_shard(a.cohort, sh)
            for s_ in a.rerun:
                import shutil
                shutil.rmtree(SS.sample_dir(a.cohort, s_), ignore_errors=True)
                shutil.rmtree(Path(a.cohort) / "acc" / s_, ignore_errors=True)
        print(json.dumps(run_cohort(a.manifest, a.cohort, a.workers, cfg, a.shard_size, a.limit, a.sample,
                                    shard_base=a.shard_base), indent=1))


if __name__ == "__main__":
    sys.exit(main())
