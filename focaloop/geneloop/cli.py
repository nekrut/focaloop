"""Standalone GeneLoop CLI (`python -m focaloop.geneloop.cli`) with `--long-stranded` support."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from ..runners_geneloop import run_geneloop_galaxy


def _default_threads() -> int:
    raw = os.environ.get("GALAXY_SLOTS") or ""
    if raw.strip().isdigit():
        return max(1, int(raw.strip()))
    return max(1, min(8, os.cpu_count() or 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="focaloop-geneloop",
        description="Adaptive eukaryotic structural genome annotation",
    )
    p.add_argument("--genome", help="target genome assembly FASTA(.gz)")
    p.add_argument("--ncbi-accession", help="optional NCBI assembly accession (GCF_.../GCA_...)")
    p.add_argument("--genome-dbkey", default="", help="optional Galaxy cached genome dbkey")
    p.add_argument("--genome-name", default="", help="optional Galaxy cached genome display name")
    p.add_argument("--genome-value", default="", help="optional Galaxy cached genome value")
    p.add_argument("--clade", default="vertebrates", help="organism clade preset")
    p.add_argument("--mode", choices=("auto", "fast_evidence", "cpu_tiberius", "gpu_tiberius"), default="auto")
    p.add_argument("--proteins", help="optional related-species homology protein FASTA(.gz)")
    p.add_argument("--protein-source", choices=("", "none", "history", "ncbi", "orthodb"), default="")
    p.add_argument("--protein-query", default="")
    p.add_argument("--protein-max-species", type=int, default=0)
    p.add_argument("--orthodb-mode", choices=("representatives", "full_partition"), default="representatives")
    p.add_argument("--exclude-genus", default="")
    p.add_argument("--tiberius-weights", help="optional Tiberius weights directory or .tar.gz archive")
    p.add_argument("--rna-manifest", help="optional SRA/ENA RNA-seq run manifest")
    p.add_argument("--sra-accessions", help="optional comma/space-separated public SRA/ENA run accessions")
    p.add_argument("--short-reads-1", nargs="+")
    p.add_argument("--short-reads-2", nargs="+")
    p.add_argument("--short-read-ids", nargs="+")
    p.add_argument("--long-reads", nargs="+")
    p.add_argument("--long-read-ids", nargs="+")
    p.add_argument("--long-stranded", action="store_true", dest="long_stranded", default=False,
                   help="treat long-read RNA-seq as strand-oriented (e.g. ONT direct-RNA -> minimap2 -uf)")
    p.add_argument("--no-long-stranded", action="store_false", dest="long_stranded",
                   help="treat long-read RNA-seq as unoriented cDNA/Iso-Seq (default -> minimap2 -ub)")
    p.add_argument("--short-bams", nargs="+")
    p.add_argument("--long-bams", nargs="+")
    p.add_argument("--short-bam-ids", nargs="+")
    p.add_argument("--long-bam-ids", nargs="+")
    p.add_argument("--logan-mode", choices=("auto", "logan_only", "off"), default="auto")
    p.add_argument("--max-frac-spots", type=float, default=0.35)
    p.add_argument("--p0-stop", type=float, default=0.005)
    p.add_argument("--wave-spots", type=int, default=1_000_000)
    p.add_argument("--structural-ncrna", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--rfam-dir", help="optional Rfam CM directory")
    p.add_argument("--zoom", action="store_true", dest="enable_zoom", default=True)
    p.add_argument("--no-zoom", action="store_false", dest="enable_zoom")
    p.add_argument("--threads", type=int, default=_default_threads())
    p.add_argument("--state-dir", required=True)
    p.add_argument("--state-archive")
    p.add_argument("--out-gff3", required=True)
    p.add_argument("--out-proteins", required=True)
    p.add_argument("--out-gtf", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--out-report", required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    try:
        meta = run_geneloop_galaxy(
            genome_fasta=a.genome,
            ncbi_accession=a.ncbi_accession,
            genome_dbkey=a.genome_dbkey,
            genome_name=a.genome_name,
            genome_value=a.genome_value,
            out_gff3=a.out_gff3,
            out_proteins=a.out_proteins,
            out_gtf=a.out_gtf,
            out_summary=a.out_summary,
            out_report=a.out_report,
            state_dir=a.state_dir,
            state_archive=a.state_archive,
            proteins_fasta=a.proteins,
            protein_source=a.protein_source,
            protein_query=a.protein_query,
            protein_max_species=a.protein_max_species,
            orthodb_mode=a.orthodb_mode,
            exclude_genus=a.exclude_genus,
            clade=a.clade,
            mode=a.mode,
            tiberius_weights=a.tiberius_weights,
            rna_manifest=a.rna_manifest,
            sra_accessions=a.sra_accessions,
            short_reads_1=a.short_reads_1,
            short_reads_2=a.short_reads_2,
            short_read_ids=a.short_read_ids,
            long_reads=a.long_reads,
            long_read_ids=a.long_read_ids,
            short_bams=a.short_bams,
            long_bams=a.long_bams,
            short_bam_ids=a.short_bam_ids,
            long_bam_ids=a.long_bam_ids,
            logan_mode=a.logan_mode,
            max_frac_spots=a.max_frac_spots,
            p0_stop=a.p0_stop,
            wave_spots=a.wave_spots,
            structural_ncrna=a.structural_ncrna,
            rfam_dir=a.rfam_dir,
            enable_zoom=a.enable_zoom,
            long_stranded=a.long_stranded,
            threads=a.threads,
        )
    except Exception as exc:
        print(f"focaloop-geneloop: error: {exc}", file=sys.stderr)
        return 1
    bio = meta.get("biotypes") or {}
    print(
        f"focaloop geneloop: mode={meta.get('effective_mode')} clade={meta.get('clade')} "
        f"coding={bio.get('n_coding_genes', 0)} proteins={meta.get('proteins_emitted', 0)} "
        f"wall={meta.get('wall_seconds', 0.0):.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
