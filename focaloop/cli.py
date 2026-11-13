"""`focaloop` command-line entry point: validate | run | panexpr | malariavar | query | bait | steer."""

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


def cmd_query(a: argparse.Namespace) -> int:
    import json
    from .ops import run_query

    contrast = a.contrast
    actor = a.actor
    hypothesis = a.hypothesis
    design = a.design
    if a.args_json:
        payload = json.loads(Path(a.args_json).read_text(encoding="utf-8"))
        contrast = contrast or payload.get("contrast") or None
        actor = actor or payload.get("actor") or None
        hypothesis = hypothesis or payload.get("hypothesis") or None
        design = design or payload.get("design") or None
    if not contrast or not actor:
        _err("--contrast and --actor are required (via CLI flags or --args-json)")
        return EXIT_ERROR

    try:
        out = run_query(a.state_dir, contrast, actor, a.out_table, a.out_verdict, hypothesis, a.mode,
                        a.out_state_archive, a.alpha_budget, a.n_perm, out_report=a.out_report, design=design)
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

    q = sub.add_parser("query", help="zero-new-byte warm-state contrast / F_ST query")
    q.add_argument("--state-dir", required=True, help="state directory or .tar.zst archive")
    q.add_argument("--args-json", help="optional JSON configfile containing contrast/actor/hypothesis/design")
    q.add_argument("--contrast", help="A:B or field:A:B or field=A:B")
    q.add_argument("--actor", help="'human' or 'agent:<name>'")
    q.add_argument("--hypothesis", help="group/gene id to test (default: top discovery hit)")
    q.add_argument("--design", help="optional multi-factor formula override for PanExpr warm query")
    q.add_argument("--mode", choices=("auto", "contrast", "fst"), default="auto")
    q.add_argument("--alpha-budget", type=float, default=0.05)
    q.add_argument("--n-perm", type=int, default=199)
    q.add_argument("--out-table", required=True)
    q.add_argument("--out-verdict", required=True)
    q.add_argument("--out-report", help="optional self-contained HTML query report")
    q.add_argument("--out-state-archive", help="write the updated state (with ledger row) as .tar.zst")
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
