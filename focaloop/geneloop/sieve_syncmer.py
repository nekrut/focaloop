"""Unique-multiplicity open-syncmer Depletion Sieve for TranscriptEvidenceLoop.

Safety invariant:
- Only syncmers with assembly-wide genomic multiplicity == 1 (or organellar/rDNA bait syncmers)
  may ever enter the active depletion set.
- Syncmers shared across paralogs or gene families (multiplicity > 1) are NEVER depleted at Sieve,
  preventing high-expression housekeeping paralogs from blinding low-expression family members.
- A locus enters depletion only after clearing BOTH total junction depth (>= min_junction_depth)
  AND Good-Turing local isoform-branch convergence (singleton_ratio <= max_singleton_ratio).
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def canonical_kmer(kmer: str) -> str:
    u = kmer.upper()
    rc = revcomp(u)
    return u if u <= rc else rc


_ACGT = frozenset("ACGT")


def extract_open_syncmers(seq: str, k: int = 31, s: int = 15) -> list[str]:
    """Extract canonical open syncmers (first minimal s-mer of the canonical k-mer at offset t = (k - s) // 2).

    Reference implementation; ``glcore`` (C) reproduces it bit-for-bit (A<C<G<T order, first minimum,
    k-mers with any non-ACGT base skipped).
    """
    u = seq.upper()
    n = len(u)
    if n < k or s > k:
        return []
    target_offset = (k - s) // 2
    n_s = k - s + 1
    out: list[str] = []
    for i in range(n - k + 1):
        kmer = u[i : i + k]
        if not set(kmer) <= _ACGT:
            continue
        # Canonical orientation for deterministic syncmer selection
        ck = canonical_kmer(kmer)
        best_pos = 0
        best_smer = ck[0:s]
        for j in range(1, n_s):
            sm = ck[j : j + s]
            if sm < best_smer:
                best_smer = sm
                best_pos = j
        if best_pos == target_offset:
            out.append(ck)
    return out


@dataclass
class LocusSaturationState:
    locus_id: str
    min_exon_depth: float
    min_junction_depth: int
    total_spliced_obs: int
    singleton_junctions: int
    has_unresolved_paralog_eq: bool = False

    @property
    def singleton_ratio(self) -> float:
        if self.total_spliced_obs <= 0:
            return 1.0
        return self.singleton_junctions / float(self.total_spliced_obs)


class OpenSyncmerDepletionSieve:
    """In-flight pre-alignment read filter backed by unique-multiplicity genomic syncmers."""

    def __init__(
        self,
        k: int = 31,
        s: int = 15,
        min_exon_depth: float = 40.0,
        min_junction_depth: int = 50,
        max_singleton_ratio: float = 0.01,
        max_genomic_multiplicity: int = 1,
    ) -> None:
        self.k = k
        self.s = s
        self.min_exon_depth = min_exon_depth
        self.min_junction_depth = min_junction_depth
        self.max_singleton_ratio = max_singleton_ratio
        self.max_genomic_multiplicity = max_genomic_multiplicity
        self.genomic_multiplicity: dict[str, int] = {}
        self.active_depletion: set[str] = set()
        self.saturated_loci: set[str] = set()

    def index_genome_contigs(self, contigs: dict[str, str], organellar_ids: set[str] | None = None) -> None:
        """Count syncmer OCCURRENCES across nuclear contigs; auto-deplete organellar/rDNA contigs.

        Occurrences (not per-contig presence) are counted so tandem paralogs on the SAME chromosome
        (KRAB-ZNF clusters on chr19, OR arrays, histone clusters) are correctly multi-copy.
        """
        organellar = organellar_ids or set()
        for cid, seq in contigs.items():
            syncs = extract_open_syncmers(seq, self.k, self.s)
            if cid in organellar:
                self.active_depletion.update(syncs)
                continue
            for sm in syncs:
                self.genomic_multiplicity[sm] = self.genomic_multiplicity.get(sm, 0) + 1

    def locus_ready(self, state: LocusSaturationState) -> bool:
        """Saturation gate: depth AND Good-Turing branch convergence AND no unresolved paralog ambiguity."""
        if state.has_unresolved_paralog_eq:
            return False
        if state.min_exon_depth < self.min_exon_depth:
            return False
        if state.min_junction_depth < self.min_junction_depth:
            return False
        if state.singleton_ratio > self.max_singleton_ratio:
            return False
        return True

    def try_saturate_locus(self, state: LocusSaturationState, exonic_seq: str) -> int:
        """Promote unique exonic syncmers of a structurally converged locus into active_depletion."""
        if state.locus_id in self.saturated_loci:
            return 0
        if not self.locus_ready(state):
            return 0

        added = 0
        for sm in set(extract_open_syncmers(exonic_seq, self.k, self.s)):
            mult = self.genomic_multiplicity.get(sm, 0)
            if 1 <= mult <= self.max_genomic_multiplicity and sm not in self.active_depletion:
                self.active_depletion.add(sm)
                added += 1
        self.saturated_loci.add(state.locus_id)
        return added

    def should_deplete_read(self, read_seq: str, min_hits: int = 2) -> bool:
        """Return True iff read_seq hits >= min_hits active unique/organellar depletion syncmers."""
        if not self.active_depletion:
            return False
        hits = 0
        for sm in extract_open_syncmers(read_seq, self.k, self.s):
            if sm in self.active_depletion:
                hits += 1
                if hits >= min_hits:
                    return True
        return False


# ----------------------------------------------------------------------------------------------
# Native (C/OpenMP) depletion filter: glcore
# ----------------------------------------------------------------------------------------------

OpenSyncmerDepletionFilter = OpenSyncmerDepletionSieve

GLCORE_SRC = Path(__file__).parent / "glcore" / "glcore.c"
GLCORE_BIN = Path(os.environ.get("GLCORE_BIN", str(Path(__file__).parent / "glcore" / "build" / "glcore")))


def _glcore_pre_dirs() -> tuple[str | None, ...]:
    which_gl = shutil.which("glcore")
    return (
        str(Path(sys.prefix) / "share" / "focaloop" / "native_cache"),
        os.environ.get("FOCALOOP_NATIVE_CACHE"),
        os.environ.get("FOCALOOP_NATIVE_DIR"),
        "/opt/focaloop/native_cache",
        str(Path(which_gl).parent) if which_gl else None,
    )


def _portable_isa_enabled() -> bool:
    return os.environ.get("FOCALOOP_PORTABLE_ISA", "1").strip() != "0"


def _ensure_private_cache_dir(base: Path) -> Path:
    uid = os.getuid()
    if base.is_symlink():
        raise RuntimeError(f"Native cache path must not be a symlink: {base}")
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = os.lstat(base)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"Native cache path is not a regular directory: {base}")
    if st.st_uid != uid:
        raise RuntimeError(f"Native cache directory {base} is not owned by uid {uid}")
    if (st.st_mode & 0o777) != 0o700:
        os.chmod(base, 0o700)
        st = os.lstat(base)
        if (st.st_mode & 0o777) != 0o700:
            raise RuntimeError(f"Native cache directory {base} has unsafe mode {oct(st.st_mode & 0o777)}")
    return base


def _secure_user_native_cache() -> Path:
    uid = os.getuid()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(xdg) if xdg else Path(tempfile.gettempdir())
    return _ensure_private_cache_dir(root / f"focaloop_geneloop_native_{uid}")


def _verify_cached_exec(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return False
    if st.st_uid != os.getuid():
        return False
    if st.st_mode & 0o022:
        return False
    return os.access(path, os.X_OK)


def build_glcore(force: bool = False) -> Path:
    """Compile glcore into an owner-verified per-user cached binary (portable ISA unless FOCALOOP_PORTABLE_ISA=0)."""
    env_bin = os.environ.get("GLCORE_BIN")
    if env_bin and not force and Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
        return Path(env_bin)

    portable = _portable_isa_enabled()
    build_dir_env = os.environ.get("FOCALOOP_BUILD_NATIVE_DIR")

    if not force and not build_dir_env and portable:
        for pre_dir in _glcore_pre_dirs():
            if pre_dir:
                cand = Path(pre_dir) / "glcore"
                if cand.is_file() and os.access(cand, os.X_OK):
                    return cand

    arch = platform.machine().lower()
    isa_flags: list[str] = []
    if portable:
        if arch in ("x86_64", "amd64"):
            isa_flags = ["-msse4.2", "-mpopcnt"]
    elif arch in ("x86_64", "amd64"):
        isa_flags = ["-march=native"]

    src_bytes = GLCORE_SRC.read_bytes()
    key_blob = src_bytes + b"\0" + " ".join([sys.platform, arch, *isa_flags]).encode("utf-8")
    digest = hashlib.sha256(key_blob).hexdigest()[:16]

    if build_dir_env:
        target = Path(build_dir_env) / "glcore"
        target.parent.mkdir(parents=True, exist_ok=True)
        cache_hit = not force and target.is_file() and os.access(target, os.X_OK) and target.stat().st_mtime >= GLCORE_SRC.stat().st_mtime
    elif env_bin:
        target = Path(env_bin)
        target.parent.mkdir(parents=True, exist_ok=True)
        cache_hit = not force and target.is_file() and os.access(target, os.X_OK) and target.stat().st_mtime >= GLCORE_SRC.stat().st_mtime
    else:
        cache_dir = _secure_user_native_cache()
        target = cache_dir / f"glcore_{digest}"
        cache_hit = not force and _verify_cached_exec(target)

    if not cache_hit:
        cc = os.environ.get("CC") or "gcc"
        use_omp = True
        extra: list[str] = []
        if platform.system() == "Darwin" and not os.environ.get("CC"):
            for cand in ("gcc-15", "gcc-14", "gcc-13", "/opt/homebrew/bin/gcc-15", "/opt/homebrew/bin/gcc-14"):
                if shutil.which(cand) or Path(cand).exists():
                    cc = cand
                    break
            else:
                cc = shutil.which("clang") or "clang"
                use_omp = False
        if platform.system() == "Darwin" and "SDKROOT" not in os.environ and "clang" not in Path(cc).name:
            sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=False).stdout.strip()
            if sdk:
                extra += ["-isysroot", sdk]
        base_cmd = [cc, "-O3", *isa_flags, *extra, "-Wall"]
        attempts = [["-fopenmp"], []] if use_omp else [[]]
        tmp_out = target.parent / f".tmp_glcore_{os.getpid()}_{digest}"
        last_err = ""
        try:
            for omp_f in attempts:
                cmd = base_cmd + omp_f + ["-o", str(tmp_out), str(GLCORE_SRC), "-lm"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and tmp_out.exists():
                    os.chmod(tmp_out, 0o700)
                    os.replace(tmp_out, target)
                    return target
                last_err = res.stderr
        finally:
            tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to compile {target.name} with {cc}: {last_err}")
    return target


def native_syncmers(seqs: list[str], k: int = 31, s: int = 15) -> list[list[str]]:
    """Run ``glcore syncmers`` (parity check against :func:`extract_open_syncmers`)."""
    exe = build_glcore()
    out = subprocess.run([str(exe), "syncmers", "-k", str(k), "-s", str(s)], input="\n".join(seqs) + "\n",
                         capture_output=True, text=True, check=True).stdout
    res: list[list[str]] = [[]]
    for line in out.splitlines():
        if line == "//":
            res.append([])
        else:
            res[-1].append(line)
    return res[: len(seqs)]
