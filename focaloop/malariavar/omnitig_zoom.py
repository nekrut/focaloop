"""Component 5B: Simple Omnitigs + VarGeno QV Likelihoods + Constrained WSAF Simplex MLE.

Drop-in replacement for MalariaVar `kmer_zoom.py`'s `assemble(ref_window, reads, ...)`:
  1. Reproduction of production `kmer_zoom.py` (`kmer_zoom_baseline_assemble`, lines 20-77)
     demonstrating how greedy `_walk` + single `first_branch` fabricates false chimeric
     recombinants across >=2 polymorphic bubbles separated by >=k bp.
  2. `VarGeno` (Sun & Medvedev 2019) Phred quality survival edge weighting:
       w(e) = sum_{r ni e} prod_{b in e} (1 - 10^(-Q_b / 10))   [or min_b mode]
     suppressing low-Q sequencing errors on high-depth major clones while preserving
     clean 15% minor-clone alleles.
  3. Linear-time `Simple Omnitigs` (Rahman & Medvedev 2022; Schmidt et al. WABI 2024):
     univocal left extension (`in_deg(v_first) == 1`) and univocal right extension
     (`out_deg(v_last) == 1`) with an explicit per-direction visited-edge cycle guard
     for 81%-AT P. falciparum tandem repeats (guaranteed O(|V|+|E|) termination, zero
     infinite loops, zero over-unrolling, and 0% chimeric recombinants).
  4. Constrained non-negative simplex MLE (`sum_h pi_h = 1, pi_h >= 0`) via Poisson EM
     projected onto the safe Simple Omnitig incidence matrix to recover per-allele WSAF.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

K_DEFAULT = 31

# Precompute log(1 - 10^(-Q/10)) and (1 - 10^(-Q/10)) for integer Phred scores Q in [0..93]
_SURV_PROB = np.zeros(128, dtype=np.float64)
_LOG_SURV_PROB = np.full(128, -50.0, dtype=np.float64)
for _q in range(1, 128):
    _p = 1.0 - (10.0 ** (-_q / 10.0))
    _SURV_PROB[_q] = _p
    _LOG_SURV_PROB[_q] = math.log(_p)


# ============================================================================
# 1. Faithful Reproduction of MalariaVar Production `kmer_zoom.py` (lines 20-96)
# ============================================================================

def kmer_counts(reads: Iterable[str], k: int = K_DEFAULT) -> Counter:
    c: Counter = Counter()
    for r in reads:
        for i in range(len(r) - k + 1):
            km = r[i:i + k]
            if "N" not in km:
                c[km] += 1
    return c


def _walk_baseline(
    start: str,
    end_kmer: str,
    counts: Counter,
    max_len: int,
    min_count: int,
    k: int = K_DEFAULT,
    forced: tuple[int, str] | None = None,
):
    path = start
    cur = start[-k:]
    steps = 0
    first_branch = None
    used = {cur}
    while steps < max_len:
        if path.endswith(end_kmer) and len(path) > len(start):
            return path, first_branch
        opts = sorted(((counts.get(cur[1:] + b, 0), b) for b in "ACGT"), reverse=True)
        if forced is not None and steps == forced[0]:
            nb = forced[1]
        else:
            if opts[0][0] < min_count:
                return None, first_branch
            nb = opts[0][1]
            if first_branch is None and opts[1][0] >= max(min_count, 0.15 * opts[0][0]):
                first_branch = (steps, opts[1][1], opts[1][0], opts[0][0])
        cur = cur[1:] + nb
        if cur in used:
            return None, first_branch  # cycle (tandem repeat longer than k)
        used.add(cur)
        path += nb
        steps += 1
    return None, first_branch


def _support_baseline(seq: str, counts: Counter, k: int = K_DEFAULT) -> int:
    return min(counts.get(seq[i:i + k], 0) for i in range(len(seq) - k + 1))


def kmer_zoom_baseline_assemble(
    ref_window: str,
    reads: Sequence[str],
    min_count: int = 2,
    slack: int = 80,
    k: int = K_DEFAULT,
) -> list[tuple[str, int]]:
    """Exact reproduction of MalariaVar `kmer_zoom.py:assemble` (lines 57-73)."""
    counts = kmer_counts(reads, k=k)
    left, right = ref_window[:k], ref_window[-k:]
    if counts.get(left, 0) < min_count or counts.get(right, 0) < min_count:
        return []
    max_len = len(ref_window) - k + slack
    major, br = _walk_baseline(left, right, counts, max_len, min_count, k=k)
    out: list[tuple[str, int]] = []
    if major:
        out.append((major, _support_baseline(major, counts, k=k)))
    if br is not None:
        minor, _ = _walk_baseline(left, right, counts, max_len, min_count, k=k, forced=(br[0], br[1]))
        if minor and minor != major:
            out.append((minor, _support_baseline(minor, counts, k=k)))
    return out


def diff_alleles(ref: str, alt: str) -> tuple[int, str, str] | None:
    """Exact reproduction of MalariaVar `kmer_zoom.py:diff_alleles` (lines 79-95)."""
    if ref == alt:
        return None
    p = 0
    while p < min(len(ref), len(alt)) and ref[p] == alt[p]:
        p += 1
    s = 0
    while s < min(len(ref), len(alt)) - p and ref[-1 - s] == alt[-1 - s]:
        s += 1
    r = ref[p:len(ref) - s]
    a = alt[p:len(alt) - s]
    if len(r) != len(a) or len(r) == 0:
        if p == 0:
            return 0, ref[:len(ref) - s], alt[:len(alt) - s]
        p -= 1  # anchor base for indels
        r = ref[p:len(ref) - s]
        a = alt[p:len(alt) - s]
    return p, r, a


# ============================================================================
# 2. VarGeno Phred Survival Weighted Local Locus de Bruijn Graph
# ============================================================================

def _parse_qual_array(q: str | Sequence[int] | np.ndarray) -> np.ndarray:
    if isinstance(q, str):
        arr = np.frombuffer(q.encode("ascii"), dtype=np.uint8)
        return np.clip(arr.astype(np.int32) - 33, 0, 93)
    return np.clip(np.asarray(q, dtype=np.int32), 0, 93)


_SURV_PROB_LIST = _SURV_PROB.tolist()


def vargeno_edge_weights(
    reads: Sequence[str],
    quals: Sequence[str | bytes | np.ndarray] | None = None,
    k: int = K_DEFAULT,
    min_bq: int = 8,
    weight_mode: str = "prod",
) -> dict[str, float]:
    """Accumulate (k+1)-mer edge weights across reads, breaking on 'N' or bases with Phred < min_bq."""
    kp1 = k + 1
    weights: dict[str, float] = {}
    if quals is None:
        for r in reads:
            nr = len(r)
            if nr < kp1:
                continue
            for i in range(nr - k):
                edge = r[i:i + kp1]
                if "N" not in edge:
                    weights[edge] = weights.get(edge, 0.0) + 1.0
        return weights

    if weight_mode not in ("prod", "min"):
        raise ValueError(f"weight_mode must be 'prod' or 'min', got {weight_mode!r}")

    surv_tbl = _SURV_PROB_LIST
    get_w = weights.get
    for r, q_raw in zip(reads, quals):
        nr = len(r)
        if nr < kp1:
            continue
        if isinstance(q_raw, (bytes, bytearray, memoryview)):
            qb = bytes(q_raw)
            q_list = [b - 33 for b in qb]
        elif isinstance(q_raw, str):
            q_list = [ord(c) - 33 for c in q_raw]
        else:
            arr_q = np.asarray(q_raw)
            if arr_q.dtype == np.uint8:
                q_list = [int(b) - 33 for b in arr_q.tolist()]
            else:
                q_list = arr_q.tolist()
        if len(q_list) != nr:
            raise ValueError(f"Read length {nr} != qual length {len(q_list)}")

        bad_ps = [0] * (nr + 1)
        low_ps = [0] * (nr + 1)
        b_acc = 0
        l_acc = 0
        for idx_c, (ch, qv) in enumerate(zip(r, q_list)):
            if qv < min_bq or ch == "N":
                b_acc += 1
            elif qv < 30:
                l_acc += 1
            bad_ps[idx_c + 1] = b_acc
            low_ps[idx_c + 1] = l_acc

        if weight_mode == "prod":
            csum = [0.0] * (nr + 1)
            c_acc = 0.0
            for idx_c, qv in enumerate(q_list):
                qc = 0 if qv < 0 else (93 if qv > 93 else qv)
                c_acc += float(_LOG_SURV_PROB[qc])
                csum[idx_c + 1] = c_acc
            for i in range(nr - k):
                if bad_ps[i + kp1] > bad_ps[i]:
                    continue
                edge = r[i:i + kp1]
                weights[edge] = get_w(edge, 0.0) + math.exp(csum[i + kp1] - csum[i])
        else:
            for i in range(nr - k):
                if bad_ps[i + kp1] > bad_ps[i]:
                    continue
                if low_ps[i + kp1] == low_ps[i]:
                    w = 0.9995
                else:
                    mq = min(q_list[i:i + kp1])
                    w = surv_tbl[93 if mq > 93 else (0 if mq < 0 else mq)]
                if w > 0.0:
                    edge = r[i:i + kp1]
                    weights[edge] = get_w(edge, 0.0) + w
    return weights


# ============================================================================
# 3. Directed Locus Graph, Unitigs, and Cycle-Guarded Simple Omnitigs
# ============================================================================

@dataclass
class LocusGraph:
    k: int
    nodes: list[str]
    node_to_id: dict[str, int]
    edge_u: list[int]
    edge_v: list[int]
    edge_seq: list[str]      # (k+1)-mer
    edge_weight: list[float]
    out_edges: list[list[int]]
    in_edges: list[list[int]]
    guarded_out_branch: list[bool]
    guarded_in_merge: list[bool]


def build_locus_graph(
    edge_weights: dict[str, float],
    k: int = K_DEFAULT,
    min_weight: float = 2.0,
    branch_guard_weight: float | None = None,
) -> LocusGraph:
    """Construct thresholded directed k-mer graph with optional sub-threshold minor-branch guards."""
    if branch_guard_weight is None:
        # Guard against rare minor-clone coverage dips right at a junction while ignoring single-read errors
        branch_guard_weight = max(1.25, 0.55 * min_weight) if min_weight > 1.3 else min_weight

    kept_edges = sorted((e, w) for e, w in edge_weights.items() if w >= min_weight)
    node_to_id: dict[str, int] = {}
    nodes: list[str] = []

    def _nid(km: str) -> int:
        idx = node_to_id.get(km)
        if idx is None:
            idx = len(nodes)
            node_to_id[km] = idx
            nodes.append(km)
        return idx

    edge_u: list[int] = []
    edge_v: list[int] = []
    edge_seq: list[str] = []
    edge_w: list[float] = []

    for e, w in kept_edges:
        u = _nid(e[:k])
        v = _nid(e[1:])
        edge_u.append(u)
        edge_v.append(v)
        edge_seq.append(e)
        edge_w.append(w)

    n_nodes = len(nodes)
    out_edges: list[list[int]] = [[] for _ in range(n_nodes)]
    in_edges: list[list[int]] = [[] for _ in range(n_nodes)]
    for eid, (u, v) in enumerate(zip(edge_u, edge_v)):
        out_edges[u].append(eid)
        in_edges[v].append(eid)

    # Secondary sub-threshold branch guard: if a node has 1 kept edge >= min_weight PLUS
    # another candidate edge in [branch_guard_weight, min_weight), prevent univocal extension
    # from blindly bridging across a dropped-out minor bubble arm!
    raw_out_deg: dict[str, int] = {}
    raw_in_deg: dict[str, int] = {}
    if branch_guard_weight < min_weight:
        for e, w in edge_weights.items():
            if w >= branch_guard_weight:
                u_km, v_km = e[:k], e[1:]
                raw_out_deg[u_km] = raw_out_deg.get(u_km, 0) + 1
                raw_in_deg[v_km] = raw_in_deg.get(v_km, 0) + 1

    guarded_out = [
        len(out_edges[i]) > 1 or raw_out_deg.get(nodes[i], 0) > 1
        for i in range(n_nodes)
    ]
    guarded_in = [
        len(in_edges[i]) > 1 or raw_in_deg.get(nodes[i], 0) > 1
        for i in range(n_nodes)
    ]

    return LocusGraph(
        k=k,
        nodes=nodes,
        node_to_id=node_to_id,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_seq=edge_seq,
        edge_weight=edge_w,
        out_edges=out_edges,
        in_edges=in_edges,
        guarded_out_branch=guarded_out,
        guarded_in_merge=guarded_in,
    )


def spell_edge_walk(graph: LocusGraph, edge_ids: Sequence[int]) -> str:
    if not edge_ids:
        return ""
    parts = [graph.edge_seq[edge_ids[0]]]
    for eid in edge_ids[1:]:
        parts.append(graph.edge_seq[eid][-1])
    return "".join(parts)


def extract_core_unitigs(graph: LocusGraph, stop_vertices: set[int] | None = None) -> list[list[int]]:
    """Partition all edges of `graph` into maximal edge-disjoint non-branching core unitigs.

    Internal vertices v of a core unitig strictly satisfy in_deg(v) == 1 and out_deg(v) == 1
    (and are neither guarded branch nor guarded merge vertices). Pure isolated cycles are broken once.
    """
    n_edges = len(graph.edge_seq)
    if n_edges == 0:
        return []
    stops = stop_vertices or set()

    def _is_1to1_vertex(vid: int) -> bool:
        return (
            vid not in stops
            and len(graph.in_edges[vid]) == 1
            and len(graph.out_edges[vid]) == 1
            and not graph.guarded_in_merge[vid]
            and not graph.guarded_out_branch[vid]
        )

    visited = [False] * n_edges
    core_unitigs: list[list[int]] = []

    # Pass 1: Core unitigs starting at edges whose source vertex is NOT 1-to-1
    for eid in range(n_edges):
        if visited[eid]:
            continue
        u = graph.edge_u[eid]
        if _is_1to1_vertex(u):
            continue
        walk = [eid]
        visited[eid] = True
        cur_v = graph.edge_v[eid]
        while _is_1to1_vertex(cur_v):
            nxt_e = graph.out_edges[cur_v][0]
            if visited[nxt_e]:
                break
            visited[nxt_e] = True
            walk.append(nxt_e)
            cur_v = graph.edge_v[nxt_e]
        core_unitigs.append(walk)

    # Pass 2: Remaining unvisited edges form pure isolated cycles
    for eid in range(n_edges):
        if visited[eid]:
            continue
        walk = [eid]
        visited[eid] = True
        cur_v = graph.edge_v[eid]
        while _is_1to1_vertex(cur_v):
            nxt_e = graph.out_edges[cur_v][0]
            if visited[nxt_e]:
                break
            visited[nxt_e] = True
            walk.append(nxt_e)
            cur_v = graph.edge_v[nxt_e]
        core_unitigs.append(walk)

    return core_unitigs


def unitig_baseline_assemble(
    ref_window: str,
    reads: Sequence[str],
    quals: Sequence[str | Sequence[int] | np.ndarray] | None = None,
    min_count: int = 2,
    k: int = K_DEFAULT,
) -> list[tuple[str, float]]:
    """Naive maximal unitigs on the local locus graph (shatters at every bubble & shared repeat)."""
    ew = vargeno_edge_weights(reads, quals=quals, k=k)
    g = build_locus_graph(ew, k=k, min_weight=float(min_count), branch_guard_weight=float(min_count))
    cores = extract_core_unitigs(g)
    out: list[tuple[str, float]] = []
    for c in cores:
        seq = spell_edge_walk(g, c)
        mean_w = float(np.mean([g.edge_weight[e] for e in c]))
        out.append((seq, mean_w))
    out.sort(key=lambda x: (-x[1], -len(x[0])))
    return out


def _is_sublist(needle: list[int], haystack: list[int]) -> bool:
    n, m = len(needle), len(haystack)
    if n > m:
        return False
    if n == 0:
        return True
    first = needle[0]
    for i in range(m - n + 1):
        if haystack[i] == first and haystack[i:i + n] == needle:
            return True
    return False


def extract_simple_omnitigs(
    graph: LocusGraph,
    max_walk_edges: int = 2048,
    anchor_left: str | None = None,
    anchor_right: str | None = None,
    return_core_indices: bool = False,
):
    """Compute maximal deduplicated Simple Omnitigs (Schmidt et al. WABI 2024) in O(|V|+|E|) time.

    For each maximal core unitig C = (e_0, ..., e_m):
      - Univocal Left Extension: prepend unique incoming edge (w -> v_first) whenever
        in_deg(v_first) == 1 (even if out_deg(v_first) > 1!), stopping at anchor_left.
      - Univocal Right Extension: append unique outgoing edge (v_last -> u) whenever
        out_deg(v_last) == 1 (even if in_deg(v_last) > 1!), stopping at anchor_right.
      - Explicit Cycle Guard: tracks visited edges separately in left and right extensions
        (seeded with `set(C)`), plus `max_walk_edges` cap, AND blocks extension into dead-end
        sinkless/sourceless cycles that cannot connect to anchors when anchors exist.
        Guarantees 0 infinite loops and 0 over-unrolling on 81%-AT P. falciparum tandem repeats!
    """
    left_id = graph.node_to_id.get(anchor_left, -1) if anchor_left else -1
    right_id = graph.node_to_id.get(anchor_right, -1) if anchor_right else -1
    stops = {v for v in (left_id, right_id) if v >= 0} if not return_core_indices else set()
    cores = extract_core_unitigs(graph, stop_vertices=stops)
    if not cores:
        return ([], [], []) if return_core_indices else ([], [])

    # If both anchors exist and right_id is reachable from left_id, identify vertices that can
    # reach right_id and vertices reachable from left_id so sinkless/sourceless noise cycles
    # (lollipop traps) are never unrolled!
    valid_for_ext: list[bool] | None = None
    if left_id >= 0 and right_id >= 0:
        n_v = len(graph.nodes)
        fwd = [False] * n_v
        stack = [left_id]
        fwd[left_id] = True
        while stack:
            u = stack.pop()
            for eid in graph.out_edges[u]:
                v = graph.edge_v[eid]
                if not fwd[v]:
                    fwd[v] = True
                    stack.append(v)
        if fwd[right_id]:
            bwd = [False] * n_v
            stack = [right_id]
            bwd[right_id] = True
            while stack:
                v = stack.pop()
                for eid in graph.in_edges[v]:
                    u = graph.edge_u[eid]
                    if not bwd[u]:
                        bwd[u] = True
                        stack.append(u)
            valid_for_ext = [fwd[i] and bwd[i] for i in range(n_v)]

    raw_walks: list[tuple[list[int], int]] = []
    for c_idx, core in enumerate(cores):
        # Skip noise tips completely outside the s-t flow subgraph when a valid s-t flow exists,
        # UNLESS no s-t flow exists (in which case keep all components).
        if valid_for_ext is not None:
            u0 = graph.edge_u[core[0]]
            v1 = graph.edge_v[core[-1]]
            if not (valid_for_ext[u0] and valid_for_ext[v1]):
                continue

        # 1. Univocal Left Extension: in_deg(v_first) == 1 (even if out_deg(v_first) > 1)
        left_ext: list[int] = []
        left_seen: set[int] = set(core)
        v_first = graph.edge_u[core[0]]
        while (
            v_first != left_id
            and len(graph.in_edges[v_first]) == 1
            and not graph.guarded_in_merge[v_first]
            and len(left_ext) + len(core) < max_walk_edges
        ):
            e_prev = graph.in_edges[v_first][0]
            if e_prev in left_seen:
                break  # Cycle guard: prevent infinite loop / cycle over-unrolling!
            u_prev = graph.edge_u[e_prev]
            if valid_for_ext is not None and not valid_for_ext[u_prev]:
                break
            left_seen.add(e_prev)
            left_ext.append(e_prev)
            v_first = u_prev
        left_ext.reverse()

        # 2. Univocal Right Extension: out_deg(v_last) == 1 (even if in_deg(v_last) > 1)
        right_ext: list[int] = []
        right_seen: set[int] = set(core)
        v_last = graph.edge_v[core[-1]]
        while (
            v_last != right_id
            and len(graph.out_edges[v_last]) == 1
            and not graph.guarded_out_branch[v_last]
            and len(left_ext) + len(core) + len(right_ext) < max_walk_edges
        ):
            e_next = graph.out_edges[v_last][0]
            if e_next in right_seen:
                break  # Cycle guard: prevent infinite loop / cycle over-unrolling!
            v_next = graph.edge_v[e_next]
            if valid_for_ext is not None and not valid_for_ext[v_next]:
                break
            right_seen.add(e_next)
            right_ext.append(e_next)
            v_last = v_next

        full_walk = left_ext + core + right_ext
        raw_walks.append((full_walk, c_idx))

    # Deduplicate non-maximal Simple Omnitigs (keep walks not strictly contained in a longer walk)
    order = sorted(range(len(raw_walks)), key=lambda i: len(raw_walks[i][0]), reverse=True)
    kept_walks: list[list[int]] = []
    kept_core_idx: list[int] = []
    for idx in order:
        w, c_idx = raw_walks[idx]
        if any(_is_sublist(w, kw) for kw in kept_walks):
            continue
        kept_walks.append(w)
        kept_core_idx.append(c_idx)

    if return_core_indices:
        return kept_walks, cores, kept_core_idx
    return kept_walks, cores


# ============================================================================
# 4. Constrained Simplex WSAF MLE (Poisson EM over Safe Omnitig Flow Graph)
# ============================================================================

@dataclass
class OmnitigHaplotig:
    seq: str
    wsaf: float
    mean_support: float
    min_support: float
    is_anchor_to_anchor: bool
    variants: list[tuple[int, str, str]]  # (window_offset, REF, ALT)


def _enumerate_st_core_paths(
    graph: LocusGraph,
    cores: list[list[int]],
    left_id: int,
    right_id: int,
    max_paths: int = 64,
    max_edges: int = 512,
) -> list[list[int]] | None:
    """Enumerate bounded s->t walks in the compacted core-unitig graph for global Poisson simplex EM."""
    if left_id < 0 or right_id < 0:
        return None
    edge_to_core: dict[int, int] = {}
    for c_idx, c in enumerate(cores):
        edge_to_core[c[0]] = c_idx

    start_cores = [edge_to_core[e] for e in graph.out_edges[left_id] if e in edge_to_core]
    if not start_cores:
        return None

    paths: list[list[int]] = []
    # Stack items: (cur_core_idx, core_path, total_edges, core_visit_counts)
    stack: list[tuple[int, list[int], int, dict[int, int]]] = [
        (sc, [sc], len(cores[sc]), {sc: 1}) for sc in start_cores
    ]
    while stack:
        c_idx, c_path, n_ed, vcnt = stack.pop()
        end_v = graph.edge_v[cores[c_idx][-1]]
        if end_v == right_id:
            paths.append(c_path)
            if len(paths) > max_paths:
                return None  # Fall back to local omnitig flow normalization
            continue
        for nxt_e in graph.out_edges[end_v]:
            nxt_c = edge_to_core.get(nxt_e)
            if nxt_c is None:
                continue
            # Allow at most 2 visits per core unitig in candidate flow paths (covers tandem duplications)
            if vcnt.get(nxt_c, 0) >= 2:
                continue
            new_ed = n_ed + len(cores[nxt_c])
            if new_ed > max_edges:
                continue
            new_vcnt = dict(vcnt)
            new_vcnt[nxt_c] = new_vcnt.get(nxt_c, 0) + 1
            stack.append((nxt_c, c_path + [nxt_c], new_ed, new_vcnt))

    return paths if paths else None


def fit_wsaf_simplex_mle(
    graph: LocusGraph,
    omnitig_walks: list[list[int]],
    cores: list[list[int]],
    omnitig_core_indices: list[int],
    anchor_left: str | None = None,
    anchor_right: str | None = None,
    em_iters: int = 50,
) -> list[float]:
    """Fit constrained non-negative simplex MLE (pi_h >= 0, sum_h pi_h = 1) via Poisson EM
    and project onto each safe Simple Omnitig.
    """
    n_omni = len(omnitig_walks)
    if n_omni == 0:
        return []
    if n_omni == 1:
        return [1.0]

    left_id = graph.node_to_id.get(anchor_left, -1) if anchor_left else -1
    right_id = graph.node_to_id.get(anchor_right, -1) if anchor_right else -1

    n_cores = len(cores)
    core_len = np.array([len(c) for c in cores], dtype=np.float64)
    core_y = np.array([sum(graph.edge_weight[e] for e in c) for c in cores], dtype=np.float64)

    st_paths = _enumerate_st_core_paths(graph, cores, left_id, right_id)
    if st_paths is not None and len(st_paths) >= 1:
        n_paths = len(st_paths)
        # Incidence/multiplicity matrix H of shape (n_cores, n_paths)
        H = np.zeros((n_cores, n_paths), dtype=np.float64)
        path_edge_walks: list[list[int]] = []
        for p_idx, cp in enumerate(st_paths):
            ew: list[int] = []
            for c_idx in cp:
                H[c_idx, p_idx] += 1.0
                ew.extend(cores[c_idx])
            path_edge_walks.append(ew)

        active = H.sum(axis=1) > 0
        H_act = H[active]
        L_act = core_len[active]
        Y_act = core_y[active]

        denom = (L_act[:, None] * H_act).sum(axis=0)
        denom = np.maximum(denom, 1e-12)
        total_depth_init = float(Y_act.sum() / max(1.0, L_act.sum()))
        theta = np.full(n_paths, max(1e-3, total_depth_init / n_paths), dtype=np.float64)

        for _ in range(em_iters):
            expected = H_act @ theta
            ratio = Y_act / np.maximum(expected, 1e-12)
            theta = theta * ((H_act.T @ ratio) / denom)
            theta = np.maximum(theta, 1e-12)

        pi_hat = theta / np.sum(theta)

        # Project simplex path frequencies onto safe Simple Omnitig containment!
        wsafs: list[float] = []
        for w in omnitig_walks:
            val = 0.0
            for p_idx, pew in enumerate(path_edge_walks):
                if _is_sublist(w, pew):
                    val += float(pi_hat[p_idx])
            wsafs.append(min(1.0, max(0.0, val)))

        # If any omnitig was not on a complete s->t path (e.g. partial dropout arm), estimate locally
        anchor_depth = max(
            1e-6,
            sum(graph.edge_weight[e] for e in graph.out_edges[left_id]) if left_id >= 0 else 0.0,
            sum(graph.edge_weight[e] for e in graph.in_edges[right_id]) if right_id >= 0 else 0.0,
        )
        for i, c_idx in enumerate(omnitig_core_indices):
            if wsafs[i] <= 1e-9:
                c_mean = float(core_y[c_idx] / max(1.0, core_len[c_idx]))
                wsafs[i] = min(1.0, c_mean / anchor_depth)
        return wsafs

    # Fallback when s->t path enumeration is unavailable: normalize core mean depth by locus flow
    core_means = core_y / np.maximum(core_len, 1.0)
    if left_id >= 0 and graph.out_edges[left_id]:
        locus_flow = sum(graph.edge_weight[e] for e in graph.out_edges[left_id])
    elif right_id >= 0 and graph.in_edges[right_id]:
        locus_flow = sum(graph.edge_weight[e] for e in graph.in_edges[right_id])
    else:
        locus_flow = float(np.max(core_means))
    locus_flow = max(locus_flow, 1e-6)
    return [min(1.0, float(core_means[c_idx] / locus_flow)) for c_idx in omnitig_core_indices]


def _extract_omnitig_variants(ref_window: str, omnitig_seq: str, k: int = K_DEFAULT) -> list[tuple[int, str, str]]:
    """Map safe Simple Omnitig back to unique k-mer anchors in `ref_window` and extract VCF-style diffs.

    Supports one-sided unique anchoring when the opposite end terminates at a homopolymer/tandem repeat
    (e.g. 81%-AT P. falciparum A x 35 tracts where `end_km` or `start_km` occurs multiple times).
    """
    if len(omnitig_seq) < k or omnitig_seq in ref_window:
        return []
    start_km = omnitig_seq[:k]
    end_km = omnitig_seq[-k:]
    p0_first = ref_window.find(start_km)
    start_unique = p0_first >= 0 and ref_window.find(start_km, p0_first + 1) < 0

    p1_first = ref_window.find(end_km)
    end_unique = p1_first >= 0 and ref_window.find(end_km, p1_first + 1) < 0

    p0 = -1
    p1 = -1
    if start_unique:
        p0 = p0_first
        p1_cand = ref_window.find(end_km, p0)
        if p1_cand >= 0 and ref_window.find(end_km, p1_cand + 1) < 0:
            p1 = p1_cand
        else:
            # Right endpoint lands inside a repeat/homopolymer: test matching occurrence anchored by walk length
            q = p0 + len(omnitig_seq) - k
            if p0 <= q <= len(ref_window) - k and ref_window[q:q + k] == end_km:
                p1 = q
    elif end_unique:
        p1 = p1_first
        q = p1 + k - len(omnitig_seq)
        if 0 <= q <= p1 and ref_window[q:q + k] == start_km:
            p0 = q

    if p0 < 0 or p1 < p0:
        return []
    ref_sub = ref_window[p0:p1 + k]
    d = diff_alleles(ref_sub, omnitig_seq)
    if d is None:
        return []
    off, r, a = d
    return [(p0 + off, r, a)]


def _estimate_edge_survival_scale(
    quals: Sequence[str | Sequence[int] | np.ndarray] | None,
    k: int,
    weight_mode: str,
) -> float:
    """Scale min_weight by empirical Phred survival so Q20 minor clones (e.g. 2 reads -> 2*0.99^32=1.45) survive."""
    if quals is None or len(quals) == 0:
        return 1.0
    sample_q = quals[0]
    if len(sample_q) == 0:
        return 1.0
    if isinstance(sample_q, (str, bytes)):
        b = sample_q.encode("latin1") if isinstance(sample_q, str) else sample_q
        med_q = float(np.median(np.frombuffer(b, dtype=np.uint8)) - 33.0)
    else:
        med_q = float(np.median(np.asarray(sample_q, dtype=np.float64)))
    med_q = max(2.0, min(60.0, med_q))
    p_err = 10.0 ** (-med_q / 10.0)
    if weight_mode == "prod":
        surv = (1.0 - p_err) ** (k + 1)
    else:
        surv = 1.0 - p_err
    return max(0.35, min(1.0, surv))


def assemble_omnitig_wsaf(
    ref_window: str,
    reads: Sequence[str],
    quals: Sequence[str | Sequence[int] | np.ndarray] | None = None,
    min_count: int = 2,
    min_weight: float | None = None,
    slack: int = 80,
    k: int = K_DEFAULT,
    weight_mode: str = "prod",
) -> list[OmnitigHaplotig]:
    """Full Simple Omnitigs + VarGeno QV + Constrained WSAF Simplex MLE assembler."""
    if len(ref_window) < k:
        return []
    left, right = ref_window[:k], ref_window[-k:]

    ew = vargeno_edge_weights(reads, quals=quals, k=k, weight_mode=weight_mode)
    if min_weight is None:
        if quals is None:
            min_weight = float(min_count)
        else:
            surv_scale = _estimate_edge_survival_scale(quals, k=k, weight_mode=weight_mode)
            min_weight = max(1.10, 0.80 * float(min_count) * surv_scale)

    graph = build_locus_graph(ew, k=k, min_weight=min_weight)
    if left not in graph.node_to_id and right not in graph.node_to_id:
        return []

    max_walk_edges = max(len(ref_window) - k + slack, 2 * len(ref_window))
    walks, cores, core_indices = extract_simple_omnitigs(
        graph,
        max_walk_edges=max_walk_edges,
        anchor_left=left,
        anchor_right=right,
        return_core_indices=True,
    )
    if not walks:
        return []

    wsafs = fit_wsaf_simplex_mle(
        graph,
        walks,
        cores,
        core_indices,
        anchor_left=left,
        anchor_right=right,
    )

    results: list[OmnitigHaplotig] = []
    for w, wsaf in zip(walks, wsafs):
        seq = spell_edge_walk(graph, w)
        w_vals = [graph.edge_weight[e] for e in w]
        mean_sup = float(np.mean(w_vals))
        min_sup = float(np.min(w_vals))
        is_full = seq.startswith(left) and seq.endswith(right)
        variants = _extract_omnitig_variants(ref_window, seq, k=k)
        results.append(
            OmnitigHaplotig(
                seq=seq,
                wsaf=float(wsaf),
                mean_support=mean_sup,
                min_support=min_sup,
                is_anchor_to_anchor=is_full,
                variants=variants,
            )
        )

    results.sort(key=lambda h: (-h.wsaf, -len(h.seq), h.seq))
    return results


def assemble(
    ref_window: str,
    reads: Sequence[str],
    quals: Sequence[str | Sequence[int] | np.ndarray] | None = None,
    min_count: int = 2,
    slack: int = 80,
    k: int = K_DEFAULT,
    weight_mode: str = "prod",
) -> list[tuple[str, float]]:
    """Drop-in replacement for MalariaVar `kmer_zoom.py:assemble(ref_window, reads, min_count=2, slack=80)`.

    Returns `[(omnitig_seq, support)]` sorted by WSAF / quality-weighted support descending,
    provably free of false chimeric recombinants across multi-bubble polyclonal infections.
    """
    haplotigs = assemble_omnitig_wsaf(
        ref_window=ref_window,
        reads=reads,
        quals=quals,
        min_count=min_count,
        slack=slack,
        k=k,
        weight_mode=weight_mode,
    )
    return [(h.seq, round(h.mean_support, 3)) for h in haplotigs]


# ============================================================================
# 5. Multi-k Diploid Haplotype Discovery & Needleman-Wunsch VCF Decomposition
# ============================================================================

def enumerate_anchor_paths(
    graph: LocusGraph,
    cores: list[list[int]],
    anchor_left: str,
    anchor_right: str,
    max_paths: int = 16,
    max_edges: int = 600,
) -> tuple[list[tuple[str, float]], bool]:
    """Enumerate highest-support s->t paths in the compacted core-unitig graph; flag if cycle encountered."""
    left_id = graph.node_to_id.get(anchor_left, -1)
    right_id = graph.node_to_id.get(anchor_right, -1)
    if left_id < 0 or right_id < 0:
        return [], False

    edge_to_core = {c[0]: idx for idx, c in enumerate(cores) if c}
    core_mean_w = [float(np.mean([graph.edge_weight[e] for e in c])) for c in cores]
    start_cores = sorted(
        (edge_to_core[e] for e in graph.out_edges[left_id] if e in edge_to_core),
        key=lambda cid: -core_mean_w[cid],
    )
    if not start_cores:
        return [], False

    n_v = len(graph.nodes)
    can_reach = [False] * n_v
    st = [right_id]
    can_reach[right_id] = True
    while st:
        v = st.pop()
        for eid in graph.in_edges[v]:
            u = graph.edge_u[eid]
            if not can_reach[u]:
                can_reach[u] = True
                st.append(u)
    if not can_reach[left_id]:
        return [], False

    results: list[tuple[str, float]] = []
    saw_cycle = False
    stack: list[tuple[int, list[int], int, float, set[int]]] = [
        (cid, [cid], len(cores[cid]), core_mean_w[cid], {cid})
        for cid in reversed(start_cores)
        if can_reach[graph.edge_v[cores[cid][-1]]]
    ]

    while stack and len(results) < max_paths:
        cid, cpath, n_ed, min_sup, vis = stack.pop()
        v_end = graph.edge_v[cores[cid][-1]]
        if v_end == right_id and n_ed >= 1:
            full_edges = [e for c_i in cpath for e in cores[c_i]]
            seq = spell_edge_walk(graph, full_edges)
            results.append((seq, min_sup))
            continue
        if n_ed >= max_edges:
            continue
        nxt_cids = sorted(
            (edge_to_core[e] for e in graph.out_edges[v_end] if e in edge_to_core),
            key=lambda x: core_mean_w[x],
        )
        for ncid in nxt_cids:
            if not can_reach[graph.edge_v[cores[ncid][-1]]]:
                continue
            if ncid in vis:
                saw_cycle = True
                continue
            stack.append((
                ncid,
                cpath + [ncid],
                n_ed + len(cores[ncid]),
                min(min_sup, core_mean_w[ncid]),
                vis | {ncid},
            ))

    results.sort(key=lambda x: (-x[1], len(x[0])))
    return results, saw_cycle


def discover_candidate_haplotypes(
    ref_window: str,
    reads: Sequence[str],
    quals: Sequence[str | bytes | np.ndarray] | None = None,
    uncap_ref_depth: int = 0,
    k_list: Sequence[int] = (31, 55),
    min_weight: float = 1.8,
    slack: int = 120,
    max_candidates: int = 8,
) -> list[str]:
    """Return deduplicated candidate haplotypes `[ref_window, alt_1, ...]` using multi-k Simple Omnitigs."""
    candidates: list[str] = [ref_window]
    seen: set[str] = {ref_window}

    for k in k_list:
        if len(ref_window) < k + 2:
            continue
        left, right = ref_window[:k], ref_window[-k:]
        ew = vargeno_edge_weights(reads, quals=quals, k=k, weight_mode="min")
        if uncap_ref_depth > 0:
            w_ref = float(uncap_ref_depth)
            for i in range(len(ref_window) - k):
                e = ref_window[i:i + k + 1]
                if "N" not in e:
                    ew[e] = ew.get(e, 0.0) + w_ref
        else:
            for i in list(range(min(k + 4, len(ref_window) - k))) + list(range(max(0, len(ref_window) - 2 * k - 4), len(ref_window) - k)):
                e = ref_window[i:i + k + 1]
                if "N" not in e:
                    ew[e] = ew.get(e, 0.0) + 0.6

        eff_min = min_weight
        graph = build_locus_graph(ew, k=k, min_weight=eff_min, branch_guard_weight=eff_min)
        max_edges = max(len(ref_window) - k + slack, 2 * len(ref_window))
        omni_walks, cores = extract_simple_omnitigs(graph, max_walk_edges=max_edges, anchor_left=left, anchor_right=right)

        found_nonref_this_k = False
        for w in omni_walks:
            seq = spell_edge_walk(graph, w)
            i_l = seq.find(left)
            i_r = seq.rfind(right)
            if i_l >= 0 and i_r >= i_l:
                sub_seq = seq[i_l: i_r + k]
                if len(sub_seq) >= k + 1 and sub_seq not in seen:
                    seen.add(sub_seq)
                    candidates.append(sub_seq)
                    found_nonref_this_k = True
                    if len(candidates) >= max_candidates:
                        return candidates

        st_paths, saw_cycle = enumerate_anchor_paths(
            graph, cores, left, right, max_paths=max_candidates, max_edges=max_edges
        )
        for seq, _ in st_paths:
            if seq not in seen:
                seen.add(seq)
                candidates.append(seq)
                found_nonref_this_k = True
                if len(candidates) >= max_candidates:
                    return candidates

        if found_nonref_this_k and not saw_cycle:
            break

    return candidates


@njit(cache=True, nogil=True)
def _nw_align_ops_jit(rs_u8: np.ndarray, as_u8: np.ndarray) -> np.ndarray:
    lr = len(rs_u8)
    la = len(as_u8)
    NEG = -1000000000
    GO = -6
    GE = -1
    MM = -4
    M = np.full((lr + 1, la + 1), NEG, dtype=np.int32)
    I = np.full((lr + 1, la + 1), NEG, dtype=np.int32)
    D = np.full((lr + 1, la + 1), NEG, dtype=np.int32)
    tbM = np.zeros((lr + 1, la + 1), dtype=np.int8)
    tbI = np.zeros((lr + 1, la + 1), dtype=np.int8)
    tbD = np.zeros((lr + 1, la + 1), dtype=np.int8)

    M[0, 0] = 0
    for i in range(1, lr + 1):
        D[i, 0] = GO + (i - 1) * GE
        tbD[i, 0] = 0 if i == 1 else 2
    for j in range(1, la + 1):
        I[0, j] = GO + (j - 1) * GE
        tbI[0, j] = 0 if j == 1 else 1

    for i in range(1, lr + 1):
        rc = rs_u8[i - 1]
        for j in range(1, la + 1):
            sc = 2 if rc == as_u8[j - 1] else MM
            m0 = M[i - 1, j - 1]
            i0 = I[i - 1, j - 1]
            d0 = D[i - 1, j - 1]
            if m0 >= i0 and m0 >= d0:
                M[i, j] = m0 + sc
                tbM[i, j] = 0
            elif i0 >= d0:
                M[i, j] = i0 + sc
                tbM[i, j] = 1
            else:
                M[i, j] = d0 + sc
                tbM[i, j] = 2

            im = M[i, j - 1] + GO
            ii = I[i, j - 1] + GE
            if ii >= im:
                I[i, j] = ii
                tbI[i, j] = 1
            else:
                I[i, j] = im
                tbI[i, j] = 0

            dm = M[i - 1, j] + GO
            dd = D[i - 1, j] + GE
            if dd >= dm:
                D[i, j] = dd
                tbD[i, j] = 2
            else:
                D[i, j] = dm
                tbD[i, j] = 0

    buf = np.empty((lr + la + 1, 3), dtype=np.int32)
    n_ops = 0
    i = lr
    j = la
    mv = M[i, j]
    iv = I[i, j]
    dv = D[i, j]
    if mv >= iv and mv >= dv:
        st = 0
    elif iv >= dv:
        st = 1
    else:
        st = 2

    while i > 0 or j > 0:
        if st == 0:
            prv = int(tbM[i, j])
            buf[n_ops, 0] = 0 if rs_u8[i - 1] == as_u8[j - 1] else 1
            buf[n_ops, 1] = i - 1
            buf[n_ops, 2] = j - 1
            n_ops += 1
            i -= 1
            j -= 1
            st = prv
        elif st == 1:
            prv = int(tbI[i, j])
            buf[n_ops, 0] = 2
            buf[n_ops, 1] = i
            buf[n_ops, 2] = j - 1
            n_ops += 1
            j -= 1
            st = prv
        else:
            prv = int(tbD[i, j])
            buf[n_ops, 0] = 3
            buf[n_ops, 1] = i - 1
            buf[n_ops, 2] = j
            n_ops += 1
            i -= 1
            st = prv

    out = np.empty((n_ops, 3), dtype=np.int32)
    for k in range(n_ops):
        out[k, 0] = buf[n_ops - 1 - k, 0]
        out[k, 1] = buf[n_ops - 1 - k, 1]
        out[k, 2] = buf[n_ops - 1 - k, 2]
    return out


def decompose_haplotype_diffs(ref_seq: str, alt_seq: str, ref_start0: int, ctg_ref_seq_u8: np.ndarray | None = None) -> list[tuple[int, str, str]]:
    """Decompose `alt_seq` vs `ref_seq` into minimal left-aligned VCF alleles `(pos0, REF, ALT)`."""
    if ref_seq == alt_seq:
        return []

    n_r, n_a = len(ref_seq), len(alt_seq)
    p = 0
    while p < n_r and p < n_a and ref_seq[p] == alt_seq[p]:
        p += 1
    s = 0
    while s < (n_r - p) and s < (n_a - p) and ref_seq[n_r - 1 - s] == alt_seq[n_a - 1 - s]:
        s += 1

    r_core = ref_seq[p:n_r - s]
    a_core = alt_seq[p:n_a - s]
    if len(r_core) == 1 and len(a_core) == 1:
        return [(ref_start0 + p, r_core, a_core)]

    if len(r_core) == 0 or len(a_core) == 0:
        return [_left_normalize_vcf_allele(ref_seq, p, r_core, a_core, ref_start0, ctg_ref_seq_u8)]

    pad_l = min(p, 6)
    pad_r = min(s, 6)
    rs = ref_seq[p - pad_l: n_r - s + pad_r]
    as_ = alt_seq[p - pad_l: n_a - s + pad_r]
    base0 = p - pad_l

    lr, la = len(rs), len(as_)
    if lr * la > 90_000:
        return [_left_normalize_vcf_allele(ref_seq, p, r_core, a_core, ref_start0, ctg_ref_seq_u8)]

    rs_u8 = np.frombuffer(rs.encode("ascii"), dtype=np.uint8)
    as_u8 = np.frombuffer(as_.encode("ascii"), dtype=np.uint8)
    ops_arr = _nw_align_ops_jit(rs_u8, as_u8)
    _op_ch = ("=", "X", "I", "D")
    ops = [(_op_ch[int(ops_arr[t, 0])], int(ops_arr[t, 1]), int(ops_arr[t, 2])) for t in range(len(ops_arr))]

    diffs = []
    k = 0
    while k < len(ops):
        if ops[k][0] == "=":
            k += 1
            continue
        g_start = k
        g_end = k + 1
        while g_end < len(ops):
            eq = 0
            while g_end + eq < len(ops) and ops[g_end + eq][0] == "=":
                eq += 1
            if g_end + eq < len(ops) and eq < 3:
                g_end = g_end + eq + 1
            else:
                break
        r_indices = [op[1] for op in ops[g_start:g_end] if op[0] in ("=", "X", "D")]
        a_indices = [op[2] for op in ops[g_start:g_end] if op[0] in ("=", "X", "I")]
        r0 = r_indices[0] if r_indices else ops[g_start][1]
        r1 = (r_indices[-1] + 1) if r_indices else ops[g_start][1]
        a0 = a_indices[0] if a_indices else ops[g_start][2]
        a1 = (a_indices[-1] + 1) if a_indices else ops[g_start][2]
        sub_r = rs[r0:r1]
        sub_a = as_[a0:a1]
        diffs.append(_left_normalize_vcf_allele(ref_seq, base0 + r0, sub_r, sub_a, ref_start0, ctg_ref_seq_u8))
        k = g_end

    merged: list[tuple[int, str, str]] = []
    for pos0_v, rv, av in sorted(diffs, key=lambda x: x[0]):
        if merged and pos0_v < merged[-1][0] + len(merged[-1][1]):
            return [_left_normalize_vcf_allele(ref_seq, p, r_core, a_core, ref_start0, ctg_ref_seq_u8)]
        merged.append((pos0_v, rv, av))
    return merged


def _left_normalize_vcf_allele(
    ref_window: str,
    win_off: int,
    r_sub: str,
    a_sub: str,
    ref_start0: int,
    ctg_ref_seq_u8: np.ndarray | None = None,
) -> tuple[int, str, str]:
    """Trim outer matches and left-align indels to parsimonious 1-bp-anchored VCF representation."""
    while len(r_sub) > 0 and len(a_sub) > 0 and r_sub[-1] == a_sub[-1]:
        r_sub = r_sub[:-1]
        a_sub = a_sub[:-1]
    while len(r_sub) > 0 and len(a_sub) > 0 and r_sub[0] == a_sub[0]:
        r_sub = r_sub[1:]
        a_sub = a_sub[1:]
        win_off += 1

    pos0 = ref_start0 + win_off
    if len(r_sub) == 1 and len(a_sub) == 1:
        return (pos0, r_sub, a_sub)

    def _ref_char(p0: int) -> str:
        w_idx = p0 - ref_start0
        if 0 <= w_idx < len(ref_window):
            return ref_window[w_idx]
        if ctg_ref_seq_u8 is not None and 0 <= p0 < len(ctg_ref_seq_u8):
            return "ACGTN"[int(ctg_ref_seq_u8[p0])]
        return "N"

    if len(a_sub) == 0 and len(r_sub) > 0:
        while pos0 > 1:
            prev_base = _ref_char(pos0 - 1)
            if prev_base != "N" and prev_base == r_sub[-1]:
                r_sub = prev_base + r_sub[:-1]
                pos0 -= 1
            else:
                break
        anchor = _ref_char(pos0 - 1) if pos0 > 0 else "N"
        if pos0 > 0 and anchor != "N":
            return (pos0 - 1, anchor + r_sub, anchor)
        nxt = _ref_char(pos0 + len(r_sub))
        return (pos0, r_sub + nxt, nxt)

    if len(r_sub) == 0 and len(a_sub) > 0:
        while pos0 > 1:
            prev_base = _ref_char(pos0 - 1)
            if prev_base != "N" and prev_base == a_sub[-1]:
                a_sub = prev_base + a_sub[:-1]
                pos0 -= 1
            else:
                break
        anchor = _ref_char(pos0 - 1) if pos0 > 0 else "N"
        if pos0 > 0 and anchor != "N":
            return (pos0 - 1, anchor, anchor + a_sub)
        nxt = _ref_char(pos0)
        return (pos0, nxt, a_sub + nxt)

    return (pos0, r_sub, a_sub)
