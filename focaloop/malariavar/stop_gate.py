"""Stage 4: adaptive per-isolate whole-genome sufficiency stop gate.

Evaluated after every completed slice (250,000 spots). Stop when all hold:
  (a) core median depth >= 22x (>= 32x when provisional F_WS < 0.90)
  (b) core breadth at >= 10x changed < 0.5% over the last slice
  (c) WSAF class flip rate at polymorphic panel sites < 0.002 between slices k-1 and k
      (sites with >= 10x in both slices; a flip needs a class change and a WSAF move > 0.2)
  (d) curated CNV / deletion depth ratios stable: max relative change between slices < 0.05
      (the bin-dispersion SE of AT-rich 1 kb bins does not shrink with depth, so stability is the test)
  (e) every nuclear chromosome observed (>= 98% of core 1 kb bins covered)
or when the wire-byte cap is hit (high-human trap) or the archive ends.

Also steers the spot-subsampling rate p for coordinate-ordered archives, where a prefix only
covers the first chromosomes and the whole archive has to pass through: p tracks the target depth.
"""
from dataclasses import dataclass, field

import numpy as np

from . import caller as C
from . import genome as G


@dataclass
class GateParams:
    slice_spots: int = 250_000
    min_depth: float = 22.0
    min_depth_poly: float = 32.0
    fws_poly: float = 0.90
    max_breadth_delta: float = 0.005
    max_flip: float = 0.002
    max_cnv_se: float = 0.05
    min_bin_cover: float = 0.98
    max_wire_bytes: float = 2.5e9
    min_slices: int = 2
    target_depth: float = 60.0      # archives read end to end: aim p here (calibrated: F1 flat beyond ~60x)
    target_depth_poly: float = 90.0
    p_floor: float = 0.02
    use_island_pc_gate: bool = False
    max_pc_ci_halfwidth: float = 0.005


@dataclass
class GateState:
    prev_class: np.ndarray = None
    prev_ok: np.ndarray = None
    prev_w: np.ndarray = None
    prev_loci: dict = None
    prev_breadth: float = 0.0
    fws_prov: float = np.nan
    history: list = field(default_factory=list)
    bin_p: np.ndarray = None
    bin_age: np.ndarray = None
    depth_full: float = np.nan
    pc_summary: dict = None


class StopGate:
    _IPC_INDEX_CACHE = {}

    @classmethod
    def get_ipc_index(cls, g: G.Genome):
        key = (int(g.L), len(g.names))
        cached = cls._IPC_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        from . import island_pc_router as IPC
        contigs = [
            G.BASES[g.seq[int(g.offsets[i]):int(g.offsets[i + 1])]].tobytes().decode("ascii")
            for i in range(len(g.names))
        ]
        # CAV-04: skip expensive O(N*k) dense Hamming-1 table when syncmer CI path is active.
        idx = IPC.ReferenceIslandPCIndex(contigs, compute_dense_d1=False)
        cls._IPC_INDEX_CACHE[key] = idx
        return idx

    def assess_wave0_pc(self, fastq_bytes: bytes, max_reads: int = 2000):
        """Run Wave-0 island_pc_router probe on raw sieved FASTQ slice: repeats/Blanca CI + routing."""
        if not self.p.use_island_pc_gate or not fastq_bytes:
            return None
        lines = fastq_bytes.splitlines()
        seqs, quals = [], []
        for i in range(0, len(lines) - 3, 4):
            if lines[i].startswith(b"@") and lines[i + 2].startswith(b"+"):
                s = lines[i + 1].decode("ascii", errors="ignore")
                q = lines[i + 3].decode("ascii", errors="ignore")
                if len(s) >= 31 and len(s) == len(q):
                    seqs.append(s)
                    quals.append(q)
                    if len(seqs) >= max_reads:
                        break
        if not seqs:
            self.st.pc_summary = {
                "usable": False,
                "status": "INSUFFICIENT_USABLE_BASES",
                "n_reads": 0,
                "r_hat": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "ci_halfwidth": np.nan,
                "p_indel_hat": np.nan,
                "routing_verdict": "INSUFFICIENT_USABLE_BASES",
            }
            return self.st.pc_summary
        idx = self.get_ipc_index(self.g)
        prob = idx.classify_reads(seqs, quals=quals)
        ci_lo = float(prob.r_pc_syncmer_ci_low)
        ci_hi = float(prob.r_pc_syncmer_ci_high)
        hw = 0.5 * max(0.0, ci_hi - ci_lo) if (np.isfinite(ci_lo) and np.isfinite(ci_hi)) else np.nan
        usable = "INSUFFICIENT_USABLE_BASES" not in prob.routing.flags
        status = "OK" if usable else "INSUFFICIENT_USABLE_BASES"
        if prob.routing.pseudoaligners_disqualified and usable:
            status = "DISQUALIFY_PSEUDOALIGNERS"
        self.st.pc_summary = {
            "usable": bool(usable),
            "status": status,
            "flags": list(prob.routing.flags),
            "n_reads": int(prob.n_reads_processed),
            "syncmer_probe_total": int(prob.syncmer_probe_total),
            "syncmer_probe_novel": int(prob.syncmer_probe_novel),
            "r_hat": float(prob.r_pc_syncmer_bio),
            "pc_est": float(prob.r_pc_syncmer_bio),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "ci_halfwidth": hw,
            "pc_ci_halfwidth": hw,
            "p_s_hat": float(prob.p_s_hat),
            "p_d_hat": float(prob.p_d_hat),
            "p_i_hat": float(prob.p_i_hat),
            "p_indel_hat": float(prob.p_indel_hat),
            "recommended_mapper": prob.routing.malariavar_recommended,
            "allow_pseudoalignment": not bool(prob.routing.pseudoaligners_disqualified),
            "routing_reason": prob.routing.rationale,
        }
        return self.st.pc_summary

    def __init__(self, g: G.Genome, panel, params: GateParams = None, seed=11):
        self.g = g
        self.p = params or GateParams()
        self.panel = panel
        core_idx = np.where(g.core)[0]
        self.depth_idx = core_idx[::53]                      # ~392k sampled core positions
        self.core_bins = np.where(g.bin_core >= 0.5)[0]
        self.st = GateState()
        self.bin_p = None  # engine shares its per-bin sampling-rate array

    def _classes(self, acc):
        w, d = C.wsaf_at(acc, self.panel)
        cls = np.zeros(len(w), np.int8)
        cls[w >= 0.1] = 1
        cls[w >= 0.9] = 2
        return cls, d >= 5, w, d

    def metrics(self, acc, cnv_loci=True):
        d = acc.counts[self.depth_idx].sum(1, dtype=np.int32) + acc.delc[self.depth_idx]
        med = float(np.median(d))
        breadth = float((d >= 10).mean())
        cls, ok, w, dd = self._classes(acc)
        ok = dd >= 10
        flip = np.nan
        n_poly = 0
        if self.st.prev_class is not None:
            both = ok & self.st.prev_ok
            poly = both & ((cls > 0) | (self.st.prev_class > 0))
            n_poly = int(poly.sum())
            moved = (cls != self.st.prev_class) & (np.abs(np.nan_to_num(w) - np.nan_to_num(self.st.prev_w)) > 0.2)
            flip = float(moved[poly].mean()) if n_poly else np.nan
        self.st.prev_w = w
        fws, n_fws = C.fws(w, dd, self.panel["af"])
        bins_cov = float((acc.binbases[self.core_bins] > 0).mean())
        se = {}
        if cnv_loci:
            rows, *_ = C.cnv_calls_v2(self.g, acc, C.CallParams(), self.bin_p)
            loci = {r["locus"]: r["ratio"] for r in rows}
            prev = self.st.prev_loci
            if prev is None:
                for k, v in loci.items():
                    if np.isfinite(v):
                        se[k] = np.inf
            else:
                for k, v in loci.items():
                    pv = prev.get(k, np.nan)
                    if np.isfinite(v) and np.isfinite(pv):
                        se[k] = abs(v - pv) / max(abs(pv), 0.5)
                    elif np.isfinite(v) != np.isfinite(pv):
                        se[k] = np.inf
                    # loci whose local flanks remain NaN across slices are unmeasurable, not unstable
            self.st.prev_loci = loci
        return dict(median_depth=med, breadth10=breadth, flip=flip, n_poly=n_poly, fws=fws, n_fws=n_fws,
                    bins_covered=bins_cov, cnv_se_max=float(max(se.values())) if se else np.nan,
                    cnv_measurable=len(se), cnv_se=se, _cls=cls, _ok=ok)

    def evaluate(self, acc, k, wire_bytes, eof=False):
        """returns (stop, reason, metrics) after slice k completed."""
        had_prev = self.st.prev_loci is not None
        m = self.metrics(acc)
        if np.isfinite(m["fws"]) and m["n_fws"] >= 2000:
            self.st.fws_prov = m["fws"]
        poly = np.isfinite(self.st.fws_prov) and self.st.fws_prov < self.p.fws_poly
        need = self.p.min_depth_poly if poly else self.p.min_depth
        db = abs(m["breadth10"] - self.st.prev_breadth)
        a = m["median_depth"] >= need
        b = db < self.p.max_breadth_delta
        c = np.isfinite(m["flip"]) and m["flip"] < self.p.max_flip
        e = m["bins_covered"] >= self.p.min_bin_cover
        if m["cnv_measurable"] > 0:
            d = np.isfinite(m["cnv_se_max"]) and m["cnv_se_max"] < self.p.max_cnv_se
        else:
            d = bool(had_prev and e)
        self.st.prev_class, self.st.prev_ok = m.pop("_cls"), m.pop("_ok")
        self.st.prev_breadth = m["breadth10"]
        pc_ok = True
        if self.p.use_island_pc_gate and self.st.pc_summary is not None:
            m["pc_summary"] = dict(self.st.pc_summary)
            hw = self.st.pc_summary.get("ci_halfwidth", np.nan)
            usable = self.st.pc_summary.get("usable", True)
            if not usable or (np.isfinite(hw) and hw > self.p.max_pc_ci_halfwidth and k < 4):
                pc_ok = False
            m["pc_gate_pass"] = bool(pc_ok)
        m.update(dict(slice=k, need_depth=need, polyclonal_prov=bool(poly), breadth_delta=db,
                      a_depth=bool(a), b_breadth=bool(b), c_flip=bool(c), d_cnv=bool(d), e_cover=bool(e),
                      wire_bytes=float(wire_bytes)))
        self.st.history.append({kk: (vv if not isinstance(vv, dict) else dict(vv)) for kk, vv in m.items()})
        if eof:
            return True, "archive_end", m
        if k + 1 >= self.p.min_slices and a and b and c and d and e and pc_ok:
            return True, "sufficient", m
        if wire_bytes >= self.p.max_wire_bytes:
            return True, "byte_cap", m
        return False, "", m

    def adapt_p(self, p_now, acc):
        """sorted archives: estimate full-archive depth from bins covered at a known p, aim p at target.

        Each core bin remembers the p in force when it was first covered; bins first covered in the most
        recent slice may be partial and are left out. depth_full = median(bin depth / bin p).
        """
        cov = acc.binbases[self.core_bins] / 1000.0
        if self.st.bin_p is None:
            self.st.bin_p = np.full(len(self.core_bins), np.nan)
            self.st.bin_age = np.zeros(len(self.core_bins), np.int32)
        new = (cov > 0) & np.isnan(self.st.bin_p)
        self.st.bin_p[new] = p_now
        self.st.bin_age[cov > 0] += 1
        settled = (self.st.bin_age >= 2) & np.isfinite(self.st.bin_p)
        if settled.sum() < 100:
            return p_now
        d_full = float(np.median(cov[settled] / self.st.bin_p[settled]))
        self.st.depth_full = d_full
        poly = np.isfinite(self.st.fws_prov) and self.st.fws_prov < self.p.fws_poly
        target = self.p.target_depth_poly if poly else self.p.target_depth
        if d_full <= 0:
            return 1.0
        p_new = float(np.clip(target / d_full, self.p.p_floor, 1.0))
        if abs(p_new - p_now) / max(p_now, 1e-9) < 0.10:
            return p_now
        return p_new
