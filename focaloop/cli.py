"""`focaloop` command-line entry point: validate | run | query | bait | steer."""

from __future__ import annotations

import argparse
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


def cmd_query(a: argparse.Namespace) -> int:
    from .ops import run_query

    try:
        out = run_query(a.state_dir, a.contrast, a.actor, a.out_table, a.out_verdict, a.hypothesis, a.mode,
                        a.out_state_archive, a.alpha_budget, a.n_perm)
    except (OSError, ValueError, KeyError) as exc:
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

    q = sub.add_parser("query", help="zero-new-byte warm-state contrast / F_ST query")
    q.add_argument("--state-dir", required=True, help="state directory or .tar.zst archive")
    q.add_argument("--contrast", required=True, help="A:B or field=A:B")
    q.add_argument("--actor", required=True, help="'human' or 'agent:<name>'")
    q.add_argument("--hypothesis", help="group id to test (default: top discovery hit)")
    q.add_argument("--mode", choices=("auto", "contrast", "fst"), default="auto")
    q.add_argument("--alpha-budget", type=float, default=0.05)
    q.add_argument("--n-perm", type=int, default=199)
    q.add_argument("--out-table", required=True)
    q.add_argument("--out-verdict", required=True)
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
