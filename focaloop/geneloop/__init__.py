"""GeneLoop (FocalAnnot): Clean-sheet 5-stage eukaryotic structural annotation Focaloop.

Combines:
1. O(|G|) global priors once (Tiberius 2.0 CNN+BiLSTM+HMM + miniprot clade proteins + compleasm anchors)
   to protect zero-RNA single-exon & silent loci (olfactory receptors, canonical histones, developmental TFs).
2. Loop 1 — TranscriptEvidenceLoop: multi-run INSDC SRA or History FASTQ streaming + unique-multiplicity
   open-syncmer depletion filter + Good-Turing unseen-intron Judge + Two-Way EM retrocopy conditioning.
3. Loop 2 — LocusZoom: 3-frame codon/frameshift DP + 3-27 nt micro-exon scanner + structural ncRNA rescue
   restricted to <1.5% discordant genomic bases.
"""

from .sieve_syncmer import OpenSyncmerDepletionFilter, OpenSyncmerDepletionSieve, LocusSaturationState, build_glcore
from .intron_judge import GoodTuringUCBJudge, IntronObservation, LibraryIntronTracker, MultiRunIntronBanditJudge, WaveJunction
from .locus_zoom import (
    MicroExonHit,
    scan_junction_gap_micro_exon,
    SingleExonCandidate,
    classify_single_exon_candidate,
)
from .orf_scout import OrphanOrf, build_orf_scout, orphan_orf_delta_windows, scan_orphan_orfs
from .seed import (
    build_tiberius_zoom_windows,
    detect_tiberius_runtime,
    discover_tiberius_weights,
    gpu_available,
    lift_miniprot_gff3,
    lift_tiberius_gtf,
    miniprot_max_intron,
    pack_windows_to_supercontigs,
    run_targeted_miniprot,
    safe_lock_miniprot_loci,
)

__version__ = "0.3.0"


def prebuild_native_libs() -> dict[str, str]:
    """Precompile GeneLoop C/OpenMP native core (`glcore` and `liborf_scout`)."""
    glc = build_glcore()
    orf = build_orf_scout()
    return {"geneloop_glcore": str(glc), "geneloop_orf_scout": str(orf)}


__all__ = [
    "__version__",
    "prebuild_native_libs",
    "OpenSyncmerDepletionFilter",
    "OpenSyncmerDepletionSieve",
    "LocusSaturationState",
    "IntronObservation",
    "LibraryIntronTracker",
    "MultiRunIntronBanditJudge",
    "GoodTuringUCBJudge",
    "WaveJunction",
    "MicroExonHit",
    "scan_junction_gap_micro_exon",
    "SingleExonCandidate",
    "classify_single_exon_candidate",
    "OrphanOrf",
    "scan_orphan_orfs",
    "orphan_orf_delta_windows",
    "detect_tiberius_runtime",
    "discover_tiberius_weights",
    "gpu_available",
    "lift_miniprot_gff3",
    "miniprot_max_intron",
    "run_targeted_miniprot",
    "pack_windows_to_supercontigs",
    "lift_tiberius_gtf",
    "safe_lock_miniprot_loci",
    "build_tiberius_zoom_windows",
]
