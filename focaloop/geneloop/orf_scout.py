"""Native C/OpenMP orphan-ORF scout for rescuing silent lineage-specific coding loci."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .gffio import merge_intervals
from .sequtil import Genome

ORF_SCOUT_SRC = Path(__file__).parent / "csrc" / "orf_scout.c"
_LIB_EXT = ".dylib" if platform.system() == "Darwin" else ".so"
ORF_SCOUT_LIB = Path(
    os.environ.get("ORF_SCOUT_LIB", str(Path(__file__).parent / "csrc" / "build" / f"liborf_scout{_LIB_EXT}"))
)


class _COrfCandidate(ctypes.Structure):
    _fields_ = [
        ("contig_idx", ctypes.c_int),
        ("start", ctypes.c_int),
        ("end", ctypes.c_int),
        ("strand", ctypes.c_int),
        ("gc", ctypes.c_float),
        ("dinuc_entropy", ctypes.c_float),
        ("soft_n_frac", ctypes.c_float),
        ("frac_lowcopy_syncmers", ctypes.c_float),
        ("n_syncmers", ctypes.c_int),
    ]


@dataclass
class OrphanOrf:
    contig: str
    start: int
    end: int
    strand: str
    gc: float
    dinuc_entropy: float
    soft_n_frac: float
    frac_lowcopy_syncmers: float
    n_syncmers: int


def _pick_c_compiler() -> tuple[str, bool]:
    """Return (compiler_exe, supports_openmp)."""
    env_cc = os.environ.get("CC")
    if env_cc:
        return env_cc, ("clang" not in Path(env_cc).name)
    if platform.system() == "Darwin":
        for cand in ("gcc-15", "gcc-14", "gcc-13", "/opt/homebrew/bin/gcc-15", "/opt/homebrew/bin/gcc-14"):
            if shutil.which(cand) or Path(cand).exists():
                return cand, True
        return "clang", False
    return "gcc", True


def build_orf_scout(force: bool = False) -> Path:
    """Compile ``orf_scout.c`` into ``liborf_scout_<platform>_<arch>.so/.dylib`` (portable ISA when requested)."""
    import sys
    import tempfile

    env_lib = os.environ.get("ORF_SCOUT_LIB")
    if env_lib and not force and Path(env_lib).is_file():
        return Path(env_lib)

    arch = platform.machine().lower()
    portable_name = f"liborf_scout_{sys.platform}_{arch}{_LIB_EXT}"
    if not force:
        for pre_dir in (
            os.environ.get("FOCALOOP_NATIVE_CACHE"),
            os.environ.get("FOCALOOP_NATIVE_DIR"),
            str(Path(sys.prefix) / "share" / "focaloop" / "native_cache"),
            "/opt/focaloop/native_cache",
        ):
            if pre_dir:
                for cand_name in (portable_name, f"liborf_scout{_LIB_EXT}"):
                    cand = Path(pre_dir) / cand_name
                    if cand.is_file():
                        return cand

    build_dir_env = os.environ.get("FOCALOOP_BUILD_NATIVE_DIR")
    if build_dir_env:
        target = Path(build_dir_env) / portable_name
    elif env_lib:
        target = Path(env_lib)
    elif os.access(ORF_SCOUT_LIB.parent, os.W_OK) or (not ORF_SCOUT_LIB.parent.exists() and os.access(ORF_SCOUT_LIB.parent.parent, os.W_OK)):
        target = ORF_SCOUT_LIB
    else:
        target = Path(tempfile.gettempdir()) / f"focaloop_geneloop_native_{os.getuid()}" / portable_name

    target.parent.mkdir(parents=True, exist_ok=True)
    if force or not target.exists() or target.stat().st_mtime < ORF_SCOUT_SRC.stat().st_mtime:
        cc, has_omp = _pick_c_compiler()
        cmd = [cc, "-O3", "-shared", "-fPIC", "-Wall"]
        if platform.system() == "Darwin" and "SDKROOT" not in os.environ and "clang" not in Path(cc).name:
            sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=False).stdout.strip()
            if sdk:
                cmd += ["-isysroot", sdk]
        if os.environ.get("FOCALOOP_PORTABLE_ISA") == "1":
            if arch in ("x86_64", "amd64") and sys.platform != "darwin":
                cmd.extend(["-msse4.2", "-mpopcnt"])
        elif arch in ("x86_64", "amd64"):
            cmd.append("-march=native")
        attempts = [["-fopenmp"], []] if has_omp else [[]]
        tmp_out = target.parent / f".tmp_orf_scout_{os.getpid()}{_LIB_EXT}"
        last_err = ""
        try:
            for omp_f in attempts:
                res = subprocess.run(cmd + omp_f + ["-o", str(tmp_out), str(ORF_SCOUT_SRC), "-lm"], capture_output=True, text=True)
                if res.returncode == 0 and tmp_out.exists():
                    os.replace(tmp_out, target)
                    return target
                last_err = res.stderr
        finally:
            tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to compile {target.name} with {cc}: {last_err}")
    return target


_LOADED_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LOADED_LIB
    lib_path = build_orf_scout()
    if _LOADED_LIB is None:
        lib = ctypes.CDLL(str(lib_path))
        lib.scan_orphan_orfs.argtypes = [
            ctypes.c_int,                                  # n_contigs
            ctypes.POINTER(ctypes.c_char_p),               # contig_seqs
            ctypes.POINTER(ctypes.c_int),                  # contig_lens
            ctypes.POINTER(ctypes.c_int),                  # n_excl_per_contig
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # excl_starts
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # excl_ends
            ctypes.c_int,                                  # min_orf_bp
            ctypes.c_float,                                # max_soft_n_frac
            ctypes.c_float,                                # min_gc
            ctypes.c_float,                                # max_gc
            ctypes.c_float,                                # min_entropy
            ctypes.c_int,                                  # sync_k
            ctypes.c_int,                                  # sync_s
            ctypes.c_int,                                  # max_sync_mult
            ctypes.c_float,                                # min_lowcopy_frac
            ctypes.c_int,                                  # threads
            ctypes.POINTER(ctypes.POINTER(_COrfCandidate)),  # out_cands
            ctypes.POINTER(ctypes.c_int),                  # out_count
        ]
        lib.scan_orphan_orfs.restype = ctypes.c_int
        lib.free_orf_candidates.argtypes = [ctypes.POINTER(_COrfCandidate)]
        lib.free_orf_candidates.restype = None
        _LOADED_LIB = lib
    return _LOADED_LIB


def scan_orphan_orfs(
    genome: Genome,
    excluded_by_contig: dict[str, list[tuple[int, int]]] | None = None,
    organelle_contigs: set[str] | None = None,
    min_orf_bp: int = 450,
    max_soft_n_frac: float = 0.20,
    min_gc: float = 0.25,
    max_gc: float = 0.75,
    min_entropy: float = 3.30,
    sync_k: int = 31,
    sync_s: int = 15,
    max_sync_mult: int = 3,
    min_lowcopy_frac: float = 0.70,
    threads: int = 24,
) -> list[OrphanOrf]:
    """Scan nuclear contigs outside ``excluded_by_contig`` for unique/low-copy long canonical ORFs."""
    lib = _load_lib()
    org = organelle_contigs or set()
    excl_map = excluded_by_contig or {}
    contigs = [c for c in genome.lengths if c not in org]
    if not contigs:
        return []

    n_contigs = len(contigs)
    seq_bytes: list[bytes] = []
    lens = (ctypes.c_int * n_contigs)()
    n_excl = (ctypes.c_int * n_contigs)()
    excl_starts_arr = (ctypes.POINTER(ctypes.c_int) * n_contigs)()
    excl_ends_arr = (ctypes.POINTER(ctypes.c_int) * n_contigs)()
    keep_alive: list[object] = []

    for i, ctg in enumerate(contigs):
        L = genome.lengths[ctg]
        s_b = genome.fetch(ctg, 0, L).encode("ascii")
        seq_bytes.append(s_b)
        lens[i] = len(s_b)
        ivs = merge_intervals(excl_map.get(ctg, []))
        n_excl[i] = len(ivs)
        if ivs:
            sa = (ctypes.c_int * len(ivs))(*(a for a, _ in ivs))
            ea = (ctypes.c_int * len(ivs))(*(b for _, b in ivs))
            keep_alive.append(sa)
            keep_alive.append(ea)
            excl_starts_arr[i] = sa
            excl_ends_arr[i] = ea
        else:
            excl_starts_arr[i] = ctypes.POINTER(ctypes.c_int)()
            excl_ends_arr[i] = ctypes.POINTER(ctypes.c_int)()

    c_seqs = (ctypes.c_char_p * n_contigs)(*seq_bytes)
    out_ptr = ctypes.POINTER(_COrfCandidate)()
    out_cnt = ctypes.c_int(0)

    rc = lib.scan_orphan_orfs(
        n_contigs,
        c_seqs,
        lens,
        n_excl,
        excl_starts_arr,
        excl_ends_arr,
        int(min_orf_bp),
        float(max_soft_n_frac),
        float(min_gc),
        float(max_gc),
        float(min_entropy),
        int(sync_k),
        int(sync_s),
        int(max_sync_mult),
        float(min_lowcopy_frac),
        int(threads),
        ctypes.byref(out_ptr),
        ctypes.byref(out_cnt),
    )
    if rc != 0:
        raise RuntimeError(f"scan_orphan_orfs failed with rc={rc}")

    res: list[OrphanOrf] = []
    try:
        for idx in range(out_cnt.value):
            c = out_ptr[idx]
            res.append(
                OrphanOrf(
                    contig=contigs[c.contig_idx],
                    start=int(c.start),
                    end=int(c.end),
                    strand="+" if c.strand > 0 else "-",
                    gc=float(c.gc),
                    dinuc_entropy=float(c.dinuc_entropy),
                    soft_n_frac=float(c.soft_n_frac),
                    frac_lowcopy_syncmers=float(c.frac_lowcopy_syncmers),
                    n_syncmers=int(c.n_syncmers),
                )
            )
    finally:
        if out_ptr:
            lib.free_orf_candidates(out_ptr)
    return res


def orphan_orf_delta_windows(
    orfs: list[OrphanOrf],
    genome_lengths: dict[str, int],
    excluded_by_contig: dict[str, list[tuple[int, int]]] | None = None,
    pad_bp: int = 10_000,
    merge_gap: int = 10_000,
) -> list[tuple[str, int, int]]:
    """Pad surviving orphan ORFs by ``±pad_bp``, merge within ``merge_gap``, and subtract existing windows."""
    by_ctg: dict[str, list[tuple[int, int]]] = {}
    for o in orfs:
        L = genome_lengths.get(o.contig, 0)
        if L <= 0:
            continue
        by_ctg.setdefault(o.contig, []).append((max(0, o.start - pad_bp), min(L, o.end + pad_bp)))

    excl_map = excluded_by_contig or {}
    out: list[tuple[str, int, int]] = []
    for ctg in genome_lengths:
        raw = sorted(by_ctg.get(ctg, []))
        if not raw:
            continue
        merged: list[list[int]] = []
        for a, b in raw:
            if merged and a <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        # Avoid re-running Tiberius over intervals already covered by merged_tib_windows / SAFE_LOCKED
        excl = merge_intervals(excl_map.get(ctg, []))
        for a, b in merged:
            cur = a
            for ea, eb in excl:
                if eb <= cur:
                    continue
                if ea >= b:
                    break
                if ea > cur:
                    out.append((ctg, cur, ea))
                cur = max(cur, eb)
                if cur >= b:
                    break
            if cur < b:
                out.append((ctg, cur, b))
    return [(c, a, b) for c, a, b in out if b - a >= 450]
