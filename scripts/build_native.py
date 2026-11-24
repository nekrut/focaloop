#!/usr/bin/env python3
"""Pure-stdlib build script for all 16 Focaloop C99/OpenMP binaries and shared libraries.

Designed for upstream ``Makefile``, ``conda-build`` (Bioconda), Docker, and Apptainer builds
where third-party runtime packages (numpy/pandas/pyarrow) may not be installed in the host
build environment. Honors ``CC``, ``CFLAGS``, ``LDFLAGS``, ``target_platform``, and
``FOCALOOP_PORTABLE_ISA=1``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def _detect_target(target_platform: str) -> tuple[str, str, str, bool]:
    tp = (target_platform or os.environ.get("target_platform") or "").strip().lower()
    if tp.startswith("osx-"):
        plat_tag = "darwin"
        arch = "arm64" if "arm64" in tp else "x86_64"
        ext = ".dylib"
        is_darwin = True
    elif tp.startswith("linux-"):
        plat_tag = "linux"
        if "aarch64" in tp or "arm64" in tp:
            arch = "aarch64"
        else:
            arch = "x86_64"
        ext = ".so"
        is_darwin = False
    else:
        plat_tag = sys.platform
        arch = platform.machine().lower()
        is_darwin = plat_tag == "darwin"
        ext = ".dylib" if is_darwin else ".so"
    return plat_tag, arch, ext, is_darwin


def _pick_compiler() -> str:
    for cand in (os.environ.get("CC"), "gcc", "clang", "cc"):
        if cand:
            toks = shlex.split(cand)
            if toks and (shutil.which(toks[0]) or Path(toks[0]).is_file()):
                return cand
    raise RuntimeError("No C compiler (CC/gcc/clang/cc) found")


def _extract_panexpr_fast_c_source(fast_c_py: Path) -> str:
    tree = ast.parse(fast_c_py.read_text(encoding="utf-8"), filename=str(fast_c_py))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_C_SOURCE":
                    val = ast.literal_eval(node.value)
                    if isinstance(val, str) and "panexpr_em_sparse" in val:
                        return val
    raise RuntimeError(f"Could not extract _C_SOURCE from {fast_c_py}")


def _compile(
    cc_cmd: list[str],
    cflags: list[str],
    ldflags: list[str],
    isa_flags: list[str],
    sources: list[Path],
    output: Path,
    *,
    shared: bool,
    use_openmp: bool,
    is_darwin: bool,
    include_dirs: list[Path] | None = None,
    libs: list[str] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    inc = []
    for d in include_dirs or []:
        inc.extend(["-I", str(d)])
    sh_flags = ["-shared", "-fPIC"] if shared else []
    link_libs = libs if libs is not None else ["-lm"]

    omp_attempts: list[list[str]] = [[]]
    if use_openmp:
        if is_darwin:
            omp_attempts = [["-Xpreprocessor", "-fopenmp", "-lomp"], ["-fopenmp"], []]
        else:
            omp_attempts = [["-fopenmp"], []]

    last_err = ""
    tmp_out = output.parent / f".tmp_{output.name}_{os.getpid()}"
    try:
        for omp in omp_attempts:
            cmd = [
                *cc_cmd,
                *cflags,
                "-O3",
                *sh_flags,
                *isa_flags,
                *inc,
                *(f for f in omp if not f.startswith("-l")),
                *(str(s) for s in sources),
                *ldflags,
                *(f for f in omp if f.startswith("-l")),
                *link_libs,
                "-o",
                str(tmp_out),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and tmp_out.is_file():
                os.chmod(tmp_out, 0o755)
                os.replace(tmp_out, output)
                return
            last_err = res.stderr.strip()
    finally:
        tmp_out.unlink(missing_ok=True)
    raise RuntimeError(f"Failed compiling {output.name}:\n{last_err}")


def build_all(prefix: Path | None, cache_dir: Path | None, bin_dir: Path | None, portable_isa: bool, target_platform: str) -> list[Path]:
    plat_tag, arch, ext, is_darwin = _detect_target(target_platform)
    cc_raw = _pick_compiler()
    cc_cmd = shlex.split(cc_raw)
    cflags = shlex.split(os.environ.get("CFLAGS", ""))
    ldflags = shlex.split(os.environ.get("LDFLAGS", ""))

    isa_flags: list[str] = []
    if portable_isa:
        if arch in ("x86_64", "amd64") and not is_darwin:
            isa_flags = ["-msse4.2", "-mpopcnt"]
        elif arch in ("x86_64", "amd64") and is_darwin:
            isa_flags = ["-msse4.2", "-mpopcnt"]

    if cache_dir is None:
        if prefix is None:
            prefix = Path(sys.prefix)
        cache_dir = prefix / "share" / "focaloop" / "native_cache"
    if bin_dir is None and prefix is not None:
        bin_dir = prefix / "bin"

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o755)
    if bin_dir is not None:
        bin_dir.mkdir(parents=True, exist_ok=True)

    built: list[Path] = []

    # 1. GeneLoop C/OpenMP core executable + ORF scout shared library
    glcore_src = REPO_ROOT / "focaloop" / "geneloop" / "glcore" / "glcore.c"
    glcore_out = cache_dir / "glcore"
    _compile(cc_cmd, cflags, ldflags, isa_flags, [glcore_src], glcore_out, shared=False, use_openmp=True, is_darwin=is_darwin, libs=["-lpthread", "-lm"])
    built.append(glcore_out)

    orf_src = REPO_ROOT / "focaloop" / "geneloop" / "csrc" / "orf_scout.c"
    orf_out = cache_dir / f"liborf_scout_{plat_tag}_{arch}{ext}"
    _compile(cc_cmd, cflags, ldflags, isa_flags, [orf_src], orf_out, shared=True, use_openmp=True, is_darwin=is_darwin, libs=["-lm"])
    built.append(orf_out)

    # 2. VariationLoop C shared libraries + 2 Logan streaming executables
    mv_dir = REPO_ROOT / "focaloop" / "malariavar" / "native"
    for stem, src_name, omp in (
        ("libpla_sieve", "pla_sieve.c", True),
        ("libdothash", "dothash_sketch.c", True),
        ("libisland_pc", "island_pc_core.c", False),
        ("libust", "ust_kernel.c", False),
        ("varloop_core", "varloop_core.c", False),
    ):
        out_so = cache_dir / f"{stem}_{plat_tag}_{arch}{ext}"
        _compile(
            cc_cmd, cflags, ldflags, isa_flags, [mv_dir / src_name], out_so,
            shared=True, use_openmp=omp, is_darwin=is_darwin, include_dirs=[mv_dir], libs=["-lpthread", "-lm"],
        )
        built.append(out_so)

    for bin_name in ("prep_unitigs_filt_adapt", "logan_var_pileup"):
        out_bin = cache_dir / bin_name
        _compile(cc_cmd, cflags, ldflags, isa_flags, [mv_dir / f"{bin_name}.c"], out_bin, shared=False, use_openmp=False, is_darwin=is_darwin, libs=["-lm"])
        built.append(out_bin)

    # 3. ExpressionLoop C99 shared libraries (SHA-256 keyed) + 2 Logan streaming executables
    pan_dir = REPO_ROOT / "focaloop" / "panexpr"
    fast_c_src_text = _extract_panexpr_fast_c_source(pan_dir / "_fast_c.py")
    fast_digest = hashlib.sha256(fast_c_src_text.encode("utf-8")).hexdigest()[:16]
    fast_tmp_c = cache_dir / f".panexpr_fast_{fast_digest}.c"
    fast_out = cache_dir / f"panexpr_fast_{fast_digest}{ext}"
    try:
        fast_tmp_c.write_text(fast_c_src_text, encoding="utf-8")
        _compile(cc_cmd, cflags, ldflags, isa_flags, [fast_tmp_c], fast_out, shared=True, use_openmp=False, is_darwin=is_darwin, libs=["-lpthread", "-lm"])
    finally:
        fast_tmp_c.unlink(missing_ok=True)
    built.append(fast_out)

    for name, src_name in (
        ("island_pc", "island_pc_core.c"),
        ("ust", "ust_kernel.c"),
        ("sam_unordered", "sam_unordered_core.c"),
        ("dothash", "dothash_sketch.c"),
    ):
        src_p = pan_dir / src_name
        digest = hashlib.sha256(src_p.read_bytes()).hexdigest()[:16]
        out_so = cache_dir / f"lib{name}_{digest}{ext}"
        _compile(cc_cmd, cflags, ldflags, isa_flags, [src_p], out_so, shared=True, use_openmp=False, is_darwin=is_darwin, libs=["-lm"])
        built.append(out_so)

    for bin_name in ("prep_expr_unitigs", "expr_sam_ec_v2"):
        out_bin = cache_dir / bin_name
        _compile(cc_cmd, cflags, ldflags, isa_flags, [pan_dir / "csrc" / f"{bin_name}.c"], out_bin, shared=False, use_openmp=False, is_darwin=is_darwin, libs=["-lm"])
        built.append(out_bin)

    # Copy all 5 standalone executables to bin_dir when installing into a prefix
    if bin_dir is not None and bin_dir.resolve() != cache_dir.resolve():
        for bin_name in ("glcore", "prep_unitigs_filt_adapt", "logan_var_pileup", "prep_expr_unitigs", "expr_sam_ec_v2"):
            dst = bin_dir / bin_name
            shutil.copy2(cache_dir / bin_name, dst)
            os.chmod(dst, 0o755)

    return built


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", type=Path, default=None, help="Installation prefix (installs to PREFIX/bin and PREFIX/share/focaloop/native_cache)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Explicit native_cache output directory")
    ap.add_argument("--bin-dir", type=Path, default=None, help="Optional directory to install standalone C binaries")
    ap.add_argument("--target-platform", default="", help="Optional conda-build target_platform override")
    ap.add_argument("--native-isa", action="store_true", help="Disable portable x86-64 SSE4.2/POPCNT baseline")
    args = ap.parse_args()

    portable = not args.native_isa and os.environ.get("FOCALOOP_PORTABLE_ISA", "1") != "0"
    built = build_all(args.prefix, args.cache_dir, args.bin_dir, portable, args.target_platform)
    print(f"Built {len(built)} native Focaloop artifacts in {built[0].parent}")


if __name__ == "__main__":
    main()
