"""`focaloop` command-line entry point: validate | run | expressionloop | variationloop | geneloop | panexpr | malariavar | query | bait | steer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from . import __version__
from .blueprint import BlueprintError, load_blueprint, resolve_blueprint_path, validate_blueprint
from .engine import EXIT_ERROR, EXIT_OK, FocaloopRun, RunSpec


def _err(msg: str) -> None:
    print(f"focaloop: error: {msg}", file=sys.stderr)


def cmd_validate(a: argparse.Namespace) -> int:
    rc = EXIT_OK
    for ref in a.blueprints:
        try:
            path = resolve_blueprint_path(ref)
            doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            _err(f"{ref}: {exc}")
            rc = EXIT_ERROR
            continue
        errors = validate_blueprint(doc)
        if errors:
            rc = EXIT_ERROR
            print(f"INVALID {path}")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK {path} ({doc['id']} {doc['version']}, modality={doc['modality']})")
    return rc


def cmd_run(a: argparse.Namespace) -> int:
    try:
        bp = load_blueprint(a.blueprint)
    except (BlueprintError, FileNotFoundError) as exc:
        _err(str(exc))
        return EXIT_ERROR
    if bp.modality in ("query", "bait"):
        _err(f"blueprint modality {bp.modality!r} is served by 'focaloop {bp.modality}', not 'run'")
        return EXIT_ERROR
    contrast = tuple(a.contrast.split(":", 1)) if a.contrast else None
    try:
        if a.spec:
            spec = RunSpec.load(a.spec)
            if contrast:
                spec.contrast = contrast
        else:
            if not (a.targets and a.sample_sheet and a.reads):
                _err("either --spec or (--targets, --sample-sheet, --reads) is required")
                return EXIT_ERROR
            spec = RunSpec.from_sheet(a.targets, a.sample_sheet, a.reads, contrast, a.priors, a.variant_sites,
                                      a.read_ids)
        if a.priors and a.spec:
            spec.priors = Path(a.priors)
    except (OSError, ValueError, KeyError) as exc:
        _err(f"spec: {exc}")
        return EXIT_ERROR
    run = FocaloopRun(bp, spec, a.state_dir, a.out_table, a.out_telemetry, a.state_archive, a.max_bytes)
    rc = run.execute()
    tel = run.telemetry
    print(f"focaloop run {bp.id}: status={tel['status']} kernel={tel.get('kernel')} "
          f"bytes={tel.get('total_consumed_bytes')} stop={tel.get('stop_reason')} traps={tel['trap_codes']}",
          file=sys.stderr)
    return rc


def cmd_panexpr(a: argparse.Namespace) -> int:
    from .galaxy_runners import run_panexpr_galaxy

    try:
        meta = run_panexpr_galaxy(
            manifest_path=a.manifest,
            ref_fasta=a.ref_fasta,
            ref_gtf=a.ref_gtf,
            contrast=a.contrast,
            out_de=a.out_de,
            out_report=a.out_report,
            state_dir=a.state_dir,
            state_archive=a.state_archive,
            secondary_fasta=a.secondary_fasta,
            secondary_gtf=a.secondary_gtf,
            primary_label=a.primary_label,
            secondary_label=a.secondary_label,
            reads=a.reads,
            read_ids=a.read_ids,
            organism_mode=a.organism_mode,
            source_mode=a.source_mode,
            design=a.design,
            max_byte_fraction=a.max_byte_fraction,
            v_tech_target=a.v_tech_target,
            delta_lfc_stop=a.delta_lfc_stop,
            enable_zoom=not a.no_zoom,
            threads=a.threads,
            kernel_names=a.kernels,
            n_tournament_reads=a.tournament_reads,
        )
    except Exception as exc:
        _err(str(exc))
        return EXIT_ERROR
    ls = meta["loop_summary"]
    print(
        f"focaloop panexpr: winner={ls['winner']} pct_streamed={ls['pct_archive_streamed']:.2f}% "
        f"stop={ls['stop_reason']} wall={ls['wall_seconds']:.1f}s",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_malariavar(a: argparse.Namespace) -> int:
    from .galaxy_runners import run_malariavar_galaxy

    try:
        meta = run_malariavar_galaxy(
            manifest_path=a.manifest,
            out_drm=a.out_drm,
            out_variants=a.out_variants,
            out_report=a.out_report,
            state_dir=a.state_dir,
            state_archive=a.state_archive,
            ref_fasta=a.ref_fasta,
            mode=a.mode,
            target_depth=a.target_depth,
            mapper=a.mapper,
            threads=a.threads,
            streams=a.streams,
            sample_timeout_s=a.sample_timeout,
        )
    except Exception as exc:
        _err(str(exc))
        return EXIT_ERROR
    sm = meta["summary"]
    print(
        f"focaloop malariavar: mode={sm['mode']} isolates={sm['n_isolates']} wall={sm['wall_s']:.1f}s",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_expressionloop(a: argparse.Namespace) -> int:
    import json
    from .runners_expression import run_expressionloop_galaxy

    extra_refs: list[dict[str, str]] = []
    if a.extra_refs_json:
        raw_j = json.loads(Path(a.extra_refs_json).read_text(encoding="utf-8"))
        if isinstance(raw_j, list):
            extra_refs.extend(raw_j)
    for acc in a.extra_ncbi_accession or []:
        if str(acc).strip():
            extra_refs.append({"ncbi_accession": str(acc).strip()})

    try:
        meta = run_expressionloop_galaxy(
            manifest_path=a.manifest,
            ref_fasta=a.ref_fasta,
            ref_gtf=a.ref_gtf,
            ref_dbkey=a.ref_dbkey,
            ref_name=a.ref_name,
            ref_value=a.ref_value,
            auto_ref_gtf=a.auto_ref_gtf,
            ncbi_accession=a.ncbi_accession,
            contrast=a.contrast,
            out_de=a.out_de,
            out_report=a.out_report,
            state_dir=a.state_dir,
            state_archive=a.state_archive,
            out_transcripts=a.out_transcripts,
            secondary_fasta=a.secondary_fasta,
            secondary_gtf=a.secondary_gtf,
            primary_label=a.primary_label,
            secondary_label=a.secondary_label,
            extra_references=extra_refs or None,
            auto_expand_strains=a.auto_expand_strains,
            auto_expand_taxon=a.auto_expand_taxon,
            reads=a.reads,
            reads_r1=a.reads_r1,
            reads_r2=a.reads_r2,
            read_ids=a.read_ids,
            organism_mode=a.organism_mode,
            source_mode=a.source_mode,
            logan_mode=a.logan_mode,
            logan_min_ka=a.logan_min_ka,
            design=a.design,
            followup_contrasts=a.followup_contrasts,
            max_byte_fraction=a.max_byte_fraction,
            v_tech_target=a.v_tech_target,
            delta_lfc_stop=a.delta_lfc_stop,
            min_mapped_per_sample=a.min_mapped_per_sample,
            enable_zoom=not a.no_zoom,
            enable_abundant_filter=not a.no_abundant_filter,
            filter_mode=a.filter_mode,
            threads=a.threads,
            kernel_names=a.kernels,
            n_tournament_reads=a.tournament_reads,
        )
    except Exception as exc:
        _err(str(exc))
        return EXIT_ERROR
    ls = meta["loop_summary"]
    print(
        f"focaloop expressionloop: winner={ls['winner']} pct_streamed={ls['pct_archive_streamed']:.2f}% "
        f"stop={ls['stop_reason']} wall={ls['wall_seconds']:.1f}s",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_variationloop(a: argparse.Namespace) -> int:
    from .runners_variation import run_variationloop_galaxy

    try:
        meta = run_variationloop_galaxy(
            analysis_mode=a.analysis_mode,
            out_variants=a.out_variants,
            out_report=a.out_report,
            state_dir=a.state_dir,
            state_archive=a.state_archive,
            out_vcf=a.out_vcf,
            out_drm=a.out_drm,
            manifest_path=a.manifest,
            ref_fasta=a.ref_fasta,
            ncbi_accession=a.ncbi_accession,
            sra_accessions=a.sra_accessions,
            logan_mode=a.logan_mode,
            logan_min_ka=a.logan_min_ka,
            logan_min_wsaf=a.min_wsaf,
            ploidy=a.ploidy,
            min_wsaf=a.min_wsaf,
            targets_bed=a.targets_bed,
            cram_pad_bp=a.cram_pad_bp,
            reads=a.reads,
            read_ids=a.read_ids,
            reads_r1=a.reads_r1,
            reads_r2=a.reads_r2,
            bam_path=a.bam,
            bam_ids=a.bam_ids,
            crai_path=a.crai,
            sample_name=a.sample_name,
            target_depth=a.target_depth,
            subsample_fraction=a.subsample_fraction,
            mapper=a.mapper,
            threads=a.threads,
            streams=a.streams,
            sample_timeout_s=a.sample_timeout,
            use_host_depletion=not a.no_host_depletion,
            host_depletion_mode=a.host_depletion_mode,
            enable_reassembly=not a.no_reassembly,
            min_mapq=a.min_mapq,
            min_bq=a.min_bq,
            snv_min_qual=a.snv_min_qual,
            indel_min_qual=a.indel_min_qual,
            dedup_bits=a.dedup_bits,
            reference_label=a.reference_label,
            surveillance_panel=a.surveillance_panel,
        )
    except Exception as exc:
        _err(str(exc))
        return EXIT_ERROR
    sm = meta["summary"]
    n_cnt = sm.get("n_isolates", sm.get("n_samples", 0))
    print(
        f"focaloop variationloop: mode={sm['mode']} samples={n_cnt} wall={sm['wall_s']:.1f}s",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_geneloop(a: argparse.Namespace) -> int:
    from .runners_geneloop import run_geneloop_galaxy

    try:
        meta = run_geneloop_galaxy(
            genome_fasta=a.genome,
            ncbi_accession=a.ncbi_accession,
            out_gff3=a.out_gff3,
            out_proteins=a.out_proteins,
            out_gtf=a.out_gtf,
            out_summary=a.out_summary,
            out_report=a.out_report,
            state_dir=a.state_dir,
            state_archive=a.state_archive,
            proteins_fasta=a.proteins,
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
            threads=a.threads,
        )
    except Exception as exc:
        _err(str(exc))
        return EXIT_ERROR
    bio = meta.get("biotypes") or {}
    print(
        f"focaloop geneloop: mode={meta.get('effective_mode')} clade={meta.get('clade')} "
        f"coding={bio.get('n_coding_genes', 0)} proteins={meta.get('proteins_emitted', 0)} "
        f"wall={meta.get('wall_seconds', 0.0):.1f}s",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_query(a: argparse.Namespace) -> int:
    import json
    from .ops import run_query

    contrast = a.contrast
    actor = a.actor or "human"
    hypothesis = a.hypothesis
    design = a.design
    subset = a.subset
    if a.args_json:
        payload = json.loads(Path(a.args_json).read_text(encoding="utf-8"))
        contrast = contrast or payload.get("contrast") or None
        actor = payload.get("actor") or actor
        hypothesis = hypothesis or payload.get("hypothesis") or None
        design = design or payload.get("design") or None
        subset = subset or payload.get("subset") or None
    if not contrast:
        _err("--contrast is required (via CLI flags or --args-json)")
        return EXIT_ERROR

    try:
        out = run_query(a.state_dir, contrast, actor, a.out_table, a.out_verdict, hypothesis, a.mode,
                        a.out_state_archive, a.alpha_budget, a.n_perm, out_report=a.out_report, design=design,
                        subset=subset)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        _err(str(exc))
        return EXIT_ERROR
    v = out["verdict"]
    print(f"focaloop query {v['query_id']}: confirmed={v['confirmed']} wealth={v['alpha_wealth_after']:.4f} new_bytes=0",
          file=sys.stderr)
    return EXIT_OK


def cmd_bait(a: argparse.Namespace) -> int:
    from .ops import run_bait

    try:
        df = run_bait(a.query_fasta, a.inputs, a.state_dir, a.out_tsv, a.probe_bytes, a.min_kmer_hits,
                      a.min_hit_reads, run_ids=a.run_ids)
    except (OSError, ValueError, RuntimeError) as exc:
        _err(str(exc))
        return EXIT_ERROR
    print(f"focaloop bait: {int(df['pass_gate'].sum())}/{len(df)} runs pass", file=sys.stderr)
    return EXIT_OK


def cmd_steer(a: argparse.Namespace) -> int:
    from .ops import run_steer

    try:
        out = run_steer(a.state_dir, a.coords_bed, a.out_bed, a.out_telemetry, a.min_precision,
                        out_state_archive=a.out_state_archive)
    except (OSError, ValueError, KeyError) as exc:
        _err(str(exc))
        return EXIT_ERROR
    print(f"focaloop steer: enrich={out['n_enrich_targets']} deplete={out['n_deplete_targets']} halt_run={out['halt_run']}",
          file=sys.stderr)
    return EXIT_OK


def _default_threads() -> int:
    raw = os.environ.get("GALAXY_SLOTS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 8


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="focaloop", description=__doc__)
    p.add_argument("--version", action="version", version=f"focaloop {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="strictly validate one or more blueprint.focaloop.yml files")
    v.add_argument("blueprints", nargs="+")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="stream, map, condition and gate one study")
    r.add_argument("--blueprint", required=True, help="blueprint file or builtin id")
    r.add_argument("--spec", help="run spec YAML (targets, samples, contrast)")
    r.add_argument("--targets", help="target FASTA (alternative to --spec)")
    r.add_argument("--sample-sheet", help="TSV: sample, condition[, batch, run_id, ...] in --reads order")
    r.add_argument("--reads", nargs="+", help="FASTQ(.gz) paths, URLs or run accessions, one per sheet row")
    r.add_argument("--read-ids", nargs="+", help="sample ids for --reads (match sheet rows by name, not order)")
    r.add_argument("--contrast", help="A:B condition contrast")
    r.add_argument("--priors", help="external priors TSV (EXTERNAL_PRIORS_SCHEMA columns)")
    r.add_argument("--variant-sites", help="variant sites TSV (variants modality)")
    r.add_argument("--state-dir", required=True)
    r.add_argument("--state-archive", help="also write the state directory as .tar.zst")
    r.add_argument("--out-table", required=True)
    r.add_argument("--out-telemetry", required=True)
    r.add_argument("--max-bytes", type=int, help="override stop_gate.max_bytes")
    r.set_defaults(func=cmd_run)

    pe = sub.add_parser("panexpr", help="production 5-stage adaptive RNA-seq differential expression loop")
    pe.add_argument("--manifest", required=True, help="TSV/CSV sample manifest (sample_id, condition, run_or_url)")
    pe.add_argument("--ref-fasta", required=True, help="primary reference genome FASTA(.gz)")
    pe.add_argument("--ref-gtf", required=True, help="primary reference annotation GTF/GFF3(.gz)")
    pe.add_argument("--secondary-fasta", help="optional secondary strain/assembly/virus FASTA(.gz)")
    pe.add_argument("--secondary-gtf", help="optional secondary strain/assembly/virus GTF/GFF3(.gz)")
    pe.add_argument("--primary-label", default="primary")
    pe.add_argument("--secondary-label", default="secondary")
    pe.add_argument("--reads", nargs="+", help="optional local/Galaxy History FASTQ(.gz) files")
    pe.add_argument("--read-ids", nargs="+", help="optional element identifiers matching --reads")
    pe.add_argument("--contrast", required=True, help="factor:treated:control or treated:control")
    pe.add_argument("--design", default="~condition", help="Wilkinson formula (default: ~condition)")
    pe.add_argument("--organism-mode", choices=("auto", "eukaryotic", "prokaryotic"), default="auto")
    pe.add_argument("--source-mode", choices=("auto", "http_range", "sra_odp", "local_bytes"), default="auto")
    pe.add_argument("--max-byte-fraction", type=float, default=0.12, help="max archive fraction (0.12 or 12.0)")
    pe.add_argument("--v-tech-target", type=float, default=0.25, help="median V_tech/(V_tech+V_bio) stop gate")
    pe.add_argument("--delta-lfc-stop", type=float, default=0.05)
    pe.add_argument("--no-zoom", action="store_true", help="disable Stage-5 31-mer borderline zoom")
    pe.add_argument("--threads", type=int, default=_default_threads())
    pe.add_argument("--kernels", nargs="+", help="optional subset of mapping kernels")
    pe.add_argument("--tournament-reads", type=int, default=25_000)
    pe.add_argument("--state-dir", required=True)
    pe.add_argument("--state-archive", help="output .focaloop.tar.zst state bundle")
    pe.add_argument("--out-de", required=True, help="output differential expression TSV")
    pe.add_argument("--out-report", required=True, help="output self-contained HTML report")
    pe.set_defaults(func=cmd_panexpr)

    mv = sub.add_parser("malariavar", help="production Plasmodium variation & WHO drug-resistance loop")
    mv.add_argument("--manifest", required=True, help="TSV/CSV isolate manifest (sample_id, country_or_group, source_uri_or_srr[, crai_uri])")
    mv.add_argument("--ref-fasta", help="optional Pf3D7 reference FASTA(.gz) (defaults to container Pf3D7 bundle)")
    mv.add_argument("--mode", choices=("auto", "focused_cram", "wgs_stream"), default="auto")
    mv.add_argument("--target-depth", type=float, default=24.0, help="target median core depth for wgs_stream stop gate")
    mv.add_argument("--mapper", choices=("auto", "strobealign", "minimap2", "bwa-mem2", "bwa"), default="auto")
    mv.add_argument("--threads", type=int, default=_default_threads())
    mv.add_argument("--streams", type=int, default=24)
    mv.add_argument("--sample-timeout", type=float, default=420.0)
    mv.add_argument("--state-dir", required=True)
    mv.add_argument("--state-archive", help="output .focaloop.tar.zst state bundle")
    mv.add_argument("--out-drm", required=True, help="output per-isolate WHO DRM/CNV/deletion/F_WS TSV")
    mv.add_argument("--out-variants", required=True, help="output variant calls TSV")
    mv.add_argument("--out-report", required=True, help="output self-contained HTML report")
    mv.set_defaults(func=cmd_malariavar)

    el = sub.add_parser("expressionloop", help="adaptive RNA-seq differential gene and isoform expression")
    el.add_argument("--manifest", required=True, help="TSV/CSV sample manifest")
    el.add_argument("--ref-fasta", help="primary reference genome FASTA(.gz)")
    el.add_argument("--ref-gtf", help="primary reference annotation GTF/GFF3(.gz)")
    el.add_argument("--ref-dbkey", default="", help="optional Galaxy dbkey for built-in CVMFS genome metadata")
    el.add_argument("--ref-name", default="", help="optional Galaxy display name for built-in CVMFS genome metadata")
    el.add_argument("--ref-value", default="", help="optional Galaxy data-table value for built-in CVMFS genome metadata")
    el.add_argument("--auto-ref-gtf", action="store_true", help="automatically fetch and verify matching RefSeq annotation for a built-in CVMFS genome")
    el.add_argument("--ncbi-accession", help="optional NCBI assembly accession (GCF_.../GCA_...) to fetch genome FASTA and matching GFF3")
    el.add_argument("--secondary-fasta", help="optional secondary strain/assembly/pathogen FASTA(.gz)")
    el.add_argument("--secondary-gtf", help="optional secondary strain/assembly/pathogen GTF/GFF3(.gz)")
    el.add_argument("--primary-label", default="primary")
    el.add_argument("--secondary-label", default="secondary")
    el.add_argument("--extra-refs-json", help="optional JSON list of additional reference dicts ({label, fasta, gtf, ncbi_accession})")
    el.add_argument("--extra-ncbi-accession", action="append", default=[], help="repeatable additional NCBI assembly accession(s) to join into ortholog catalog")
    el.add_argument("--auto-expand-strains", type=int, default=0, help="automatically fetch up to K distinct annotated RefSeq strains from NCBI (default: 0 = off)")
    el.add_argument("--auto-expand-taxon", default="", help="species name or NCBI taxonomy ID for --auto-expand-strains (inferred from primary --ncbi-accession if omitted)")
    el.add_argument("--reads", nargs="+", help="optional single-end FASTQ(.gz) files")
    el.add_argument("--reads-r1", nargs="+", help="optional paired R1 FASTQ(.gz) files")
    el.add_argument("--reads-r2", nargs="+", help="optional paired R2 FASTQ(.gz) files")
    el.add_argument("--read-ids", nargs="+", help="optional sample identifiers matching --reads or --reads-r1/--reads-r2")
    el.add_argument("--contrast", required=True, help="factor:groupA:groupB or groupA:groupB")
    el.add_argument("--design", default="~condition", help="statistical model formula (default: ~condition)")
    el.add_argument("--followup-contrasts", nargs="*", help="optional comma-separated extra contrasts")
    el.add_argument("--organism-mode", choices=("auto", "eukaryotic", "prokaryotic"), default="auto")
    el.add_argument("--source-mode", choices=("auto", "http_range", "sra_odp", "local_bytes"), default="auto")
    el.add_argument("--logan-mode", choices=("auto", "logan_only", "off"), default="auto", help="stream pre-assembled Logan unitigs for public SRA/ENA runs when available")
    el.add_argument("--logan-min-ka", type=float, default=0.0, help="minimum k-mer abundance (ka) cutoff for Logan unitigs (default: 0.0)")
    el.add_argument("--max-byte-fraction", type=float, default=0.12, help="max file fraction to read (0.005 to 1.0)")
    el.add_argument("--v-tech-target", type=float, default=0.25, help="target sequencing-noise share of total variance")
    el.add_argument("--delta-lfc-stop", type=float, default=0.05)
    el.add_argument("--min-mapped-per-sample", type=int, default=None)
    el.add_argument("--zoom", action="store_false", dest="no_zoom", default=False, help="enable extra scan for borderline genes")
    el.add_argument("--no-zoom", action="store_true", dest="no_zoom", help="disable extra scan for borderline genes")
    el.add_argument("--abundant-filter", action="store_false", dest="no_abundant_filter", default=False, help="skip saturated abundant transcripts during borderline scan")
    el.add_argument("--no-abundant-filter", action="store_true", dest="no_abundant_filter", help="disable abundant-transcript filter")
    el.add_argument("--filter-mode", choices=("auto", "include", "deplete"), default="auto")
    el.add_argument("--threads", type=int, default=_default_threads())
    el.add_argument("--kernels", nargs="*", help="optional subset of aligners")
    el.add_argument("--tournament-reads", type=int, default=25_000)
    el.add_argument("--state-dir", required=True)
    el.add_argument("--state-archive", help="output .focaloop.tar.zst state bundle")
    el.add_argument("--out-de", required=True, help="output differential expression TSV")
    el.add_argument("--out-report", required=True, help="output self-contained HTML report")
    el.add_argument("--out-transcripts", help="optional output transcript abundance TSV")
    el.set_defaults(func=cmd_expressionloop)

    vl = sub.add_parser("variationloop", help="organism-agnostic variant calling from FASTQ, BAM, indexed CRAM, or Logan unitigs")
    vl.add_argument("--analysis-mode", choices=("auto", "focused_cram", "wgs_stream", "diploid_wgs", "logan_unitigs"), default="auto")
    vl.add_argument("--manifest", help="TSV/CSV sample manifest or optional group metadata sheet")
    vl.add_argument("--sra-accessions", help="optional newline/comma/space-separated public SRA/ENA run accessions for direct Logan/streaming calling")
    vl.add_argument("--logan-mode", choices=("auto", "logan_only", "off"), default="auto", help="use pre-assembled Logan unitigs for public SRA/ENA runs when available")
    vl.add_argument("--logan-min-ka", default="adapt", help="Logan unitig k-mer abundance floor ('adapt' or numeric threshold)")
    vl.add_argument("--ploidy", choices=("haploid", "diploid"), default="diploid", help="calling model ploidy: 'haploid' for continuous WSAF mixtures (haploid parasites, bacteria, viruses, fungi) or 'diploid' for 0/1 and 1/1 genotypes (default: diploid)")
    vl.add_argument("--min-wsaf", "--logan-min-wsaf", dest="min_wsaf", type=float, default=0.25, help="minimum within-sample allele fraction (WSAF) for haploid/mixed-clone calling and Logan unitigs (default: 0.25)")
    vl.add_argument("--reads", nargs="+", help="optional local FASTQ(.gz) files for manifest matching")
    vl.add_argument("--read-ids", nargs="+", help="sample identifiers matching --reads or --reads-r1/--reads-r2")
    vl.add_argument("--reads-r1", nargs="+", help="one or more forward R1 FASTQ(.gz) files")
    vl.add_argument("--reads-r2", nargs="+", help="one or more reverse R2 FASTQ(.gz) files")
    vl.add_argument("--bam", nargs="+", help="one or more aligned BAM or CRAM files")
    vl.add_argument("--bam-ids", nargs="+", help="sample identifiers matching --bam")
    vl.add_argument("--crai", nargs="*", help="optional CRAM/BAM index (.crai or .bai) paths matching --bam")
    vl.add_argument("--sample-name", default=None, help="optional explicit sample name (auto-derived from input identifier when omitted)")
    vl.add_argument("--ref-fasta", help="reference genome FASTA(.gz)")
    vl.add_argument("--ncbi-accession", help="optional NCBI assembly accession (GCF_.../GCA_...) to fetch reference genome FASTA")
    vl.add_argument("--reference-label", help="display label for reference genome in reports")
    vl.add_argument("--targets-bed", help="optional target intervals BED")
    vl.add_argument("--cram-pad-bp", type=int, default=1000)
    vl.add_argument("--streams", type=int, default=24)
    vl.add_argument("--target-depth", type=float, default=24.0, help="target median coverage depth")
    vl.add_argument("--host-depletion-mode", choices=("auto", "on", "off"), default=None, help="host/off-target read or unitig pre-filtering policy")
    vl.add_argument("--host-depletion", action="store_false", dest="no_host_depletion", default=False, help="enable 31-mer target retention / host filtering")
    vl.add_argument("--no-host-depletion", action="store_true", dest="no_host_depletion", help="disable 31-mer host/off-target read filter")
    vl.add_argument("--mapper", choices=("auto", "strobealign", "minimap2", "bwa-mem2", "bwa"), default="auto")
    vl.add_argument("--subsample-fraction", type=float, default=1.0)
    vl.add_argument("--reassembly", action="store_false", dest="no_reassembly", default=False, help="enable local haplotype reassembly")
    vl.add_argument("--no-reassembly", action="store_true", dest="no_reassembly", help="disable local haplotype reassembly")
    vl.add_argument("--min-mapq", type=int, default=20)
    vl.add_argument("--min-bq", type=int, default=13)
    vl.add_argument("--snv-min-qual", type=float, default=20.0)
    vl.add_argument("--indel-min-qual", type=float, default=15.0)
    vl.add_argument("--dedup-bits", type=int, default=None, help="duplicate filter table size (default: automatic from genome length)")
    vl.add_argument("--sample-timeout", type=float, default=420.0)
    vl.add_argument("--surveillance-panel", choices=("none", "pf3d7"), default="none", help="optional organism-specific surveillance panel")
    vl.add_argument("--threads", type=int, default=_default_threads())
    vl.add_argument("--state-dir", required=True)
    vl.add_argument("--state-archive", help="output .focaloop.tar.zst state bundle")
    vl.add_argument("--out-variants", required=True, help="output variant calls TSV")
    vl.add_argument("--out-vcf", help="optional output VCF")
    vl.add_argument("--out-drm", help="optional output drug-resistance panel TSV")
    vl.add_argument("--out-report", required=True, help="output self-contained HTML report")
    vl.set_defaults(func=cmd_variationloop)

    gl = sub.add_parser("geneloop", help="adaptive eukaryotic structural genome annotation")
    gl.add_argument("--genome", help="target genome assembly FASTA(.gz)")
    gl.add_argument("--ncbi-accession", help="optional NCBI assembly accession (GCF_.../GCA_...) to fetch target genome FASTA")
    gl.add_argument("--clade", default="vertebrates", help="organism clade preset")
    gl.add_argument("--mode", choices=("auto", "fast_evidence", "cpu_tiberius", "gpu_tiberius"), default="auto")
    gl.add_argument("--proteins", help="optional related-species homology protein FASTA(.gz)")
    gl.add_argument("--exclude-genus", default="", help="optional genus to strip from homology proteins")
    gl.add_argument("--tiberius-weights", help="optional Tiberius weights directory or .tar.gz archive")
    gl.add_argument("--rna-manifest", help="optional SRA/ENA RNA-seq run manifest")
    gl.add_argument("--sra-accessions", help="optional comma/space-separated public SRA/ENA RNA-seq run accessions")
    gl.add_argument("--short-reads-1", nargs="+", help="optional local short-read R1 or single-end FASTQs")
    gl.add_argument("--short-reads-2", nargs="+", help="optional local short-read R2 FASTQs")
    gl.add_argument("--short-read-ids", nargs="+", help="sample identifiers matching short-read FASTQs")
    gl.add_argument("--long-reads", nargs="+", help="optional local long-read FASTQs")
    gl.add_argument("--long-read-ids", nargs="+", help="sample identifiers matching long-read FASTQs")
    gl.add_argument("--short-bams", nargs="+", help="optional pre-aligned short-read RNA-seq BAMs")
    gl.add_argument("--long-bams", nargs="+", help="optional pre-aligned long-read RNA-seq BAMs")
    gl.add_argument("--short-bam-ids", nargs="+", help="sample identifiers matching short-read BAMs")
    gl.add_argument("--long-bam-ids", nargs="+", help="sample identifiers matching long-read BAMs")
    gl.add_argument("--logan-mode", choices=("auto", "logan_only", "off"), default="auto")
    gl.add_argument("--max-frac-spots", type=float, default=0.35)
    gl.add_argument("--p0-stop", type=float, default=0.005)
    gl.add_argument("--wave-spots", type=int, default=1_000_000)
    gl.add_argument("--structural-ncrna", choices=("auto", "on", "off"), default="auto")
    gl.add_argument("--rfam-dir", help="optional Rfam CM directory")
    gl.add_argument("--zoom", action="store_true", dest="enable_zoom", default=True,
                    help="enable micro-exon rescue and retrocopy filtering")
    gl.add_argument("--no-zoom", action="store_false", dest="enable_zoom",
                    help="disable micro-exon rescue and retrocopy filtering")
    gl.add_argument("--threads", type=int, default=_default_threads())
    gl.add_argument("--state-dir", required=True)
    gl.add_argument("--state-archive", help="output .focaloop.tar.zst state bundle")
    gl.add_argument("--out-gff3", required=True, help="output annotation GFF3")
    gl.add_argument("--out-proteins", required=True, help="output predicted proteins FASTA")
    gl.add_argument("--out-gtf", required=True, help="output transcripts GTF")
    gl.add_argument("--out-summary", required=True, help="output summary metrics TSV")
    gl.add_argument("--out-report", required=True, help="output self-contained HTML report")
    gl.set_defaults(func=cmd_geneloop)

    q = sub.add_parser("query", help="instant follow-up comparison on a saved Focaloop state bundle")
    q.add_argument("--state", "--state-dir", dest="state_dir", required=True, help="saved Focaloop state directory or .focaloop.tar.zst archive")
    q.add_argument("--args-json", help="optional JSON configfile containing contrast/actor/hypothesis/design")
    q.add_argument("--contrast", help="groupA:groupB, factor:groupA:groupB, or field=groupA:groupB")
    q.add_argument("--actor", default="human", help="audit tag recorded in the statistical test ledger (default: human)")
    q.add_argument("--hypothesis", help="specific gene ID or chromosome/region to test (default: top hit)")
    q.add_argument("--design", help="optional statistical model formula override for ExpressionLoop queries")
    q.add_argument("--subset", help="optional space-separated sample filter clauses like 'plate_id=P1 timepoint=60m'")
    q.add_argument("--mode", choices=("auto", "contrast", "fst"), default="auto")
    q.add_argument("--alpha-budget", type=float, default=0.05)
    q.add_argument("--n-perm", type=int, default=199)
    q.add_argument("--out-table", required=True)
    q.add_argument("--out-verdict", required=True)
    q.add_argument("--out-report", help="optional self-contained HTML query report")
    q.add_argument("--out-state-archive", help="write the updated state archive (.focaloop.tar.zst)")
    q.set_defaults(func=cmd_query)

    b = sub.add_parser("bait", help="Stage-0 canonical 31-mer pan-repository screen")
    b.add_argument("--query-fasta", required=True)
    b.add_argument("--inputs", nargs="+", required=True, help="FASTQ(.gz) paths, URLs or run accessions")
    b.add_argument("--run-ids", nargs="+", help="run ids matching --inputs (default: file stem)")
    b.add_argument("--state-dir", required=True)
    b.add_argument("--out-tsv", required=True)
    b.add_argument("--probe-bytes", type=int, default=1 << 20)
    b.add_argument("--min-kmer-hits", type=int, default=2)
    b.add_argument("--min-hit-reads", type=int, default=2)
    b.set_defaults(func=cmd_bait)

    s = sub.add_parser("steer", help="export ONT ReadUntil enrich/deplete BED + halt telemetry")
    s.add_argument("--state-dir", required=True, help="state directory or .tar.zst archive")
    s.add_argument("--coords-bed", required=True)
    s.add_argument("--out-bed", required=True)
    s.add_argument("--out-telemetry")
    s.add_argument("--min-precision", type=float)
    s.add_argument("--out-state-archive")
    s.set_defaults(func=cmd_steer)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
