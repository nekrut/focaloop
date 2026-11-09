---
name: focaloop-run
description: Use when running a focaloop blueprint (panexpr, malariavar, or a distilled one) on reads -- choosing inputs, the byte budget and contrast, invoking `focaloop run` locally or the Galaxy UDT/XML tool, and reading the telemetry, trap codes and stop reason.
version: 0.1.0
---

# Running a focal loop

`focaloop run` streams reads in fixed byte blocks, picks a mapping kernel by tournament, accumulates
tie-preserving equivalence classes, re-solves them with backward EM, and stops when the stop gate says
more bytes stop changing the answer. It writes a primary TSV, `telemetry.json`, and a Zstd-Parquet state.

## Inputs

1. **Blueprint**: a builtin id (`panexpr`, `malariavar`) or a path to `blueprint.focaloop.yml`.
   Validate first: `focaloop validate <blueprint>`.
2. **Data**, either
   - `--spec spec.yml` (`targets`, `samples[{sample, condition, batch, reads, ...}]`, `contrast`, optional
     `priors`, `variant_sites`), or
   - `--targets T.fa --sample-sheet S.tsv --reads R1 R2 ...` (+ `--read-ids` to match by name).
   `reads` may be local FASTQ(.gz), an `http(s)` URL (HTTP 206 ranges), or an `SRR/ERR/DRR` accession (ENA tier 3).
3. **Target headers** carry `group=`, `assembly=`, `kind=` (`host_decoy` sieves host reads), `kingdom=`.

## Command

```bash
focaloop run --blueprint panexpr --spec spec.yml \
  --state-dir state --state-archive state.tar.zst \
  --out-table de.tsv --out-telemetry telemetry.json [--max-bytes N]
```

## Reading the result

| Telemetry field | Meaning |
|-----------------|---------|
| `status` | `ok`, `ok_with_warnings` (warn traps fired) or `trap` (abort; exit code 3) |
| `trap_codes` | any of `LOW_MAP_RATE`, `TIE_SYMMETRY_FAIL`, `BYTE_CAP_UNCONVERGED`, `CONFOUNDED_DESIGN` |
| `tournament.board` | per-kernel map rate, per-regime recall, `mate_tie_symmetry`, `gate_passed` |
| `stop_reason` | `converged`, `byte_cap_unconverged`, `exhausted_unconverged` |
| `iterations[]` | cumulative bytes, map rate, posterior precision, tech-variance fraction, delta |

## Trap playbook

- `LOW_MAP_RATE`: wrong catalog or contaminated library; inspect `residual_spans.parquet` sketches before adding targets.
- `TIE_SYMMETRY_FAIL`: every kernel collapses strain/paralog ties; add `kmer_eq` (or `salmon`) to `kernel_pool`.
- `CONFOUNDED_DESIGN`: condition equals batch; fix the design, do not override.
- `BYTE_CAP_UNCONVERGED`: raise `--max-bytes` or accept wider intervals; the state is still valid for queries.

## Galaxy

Use the `focaloop_panexpr` / `focaloop_malariavar` XML tools, or create the UDT from
`blueprints/<id>/galaxy_udt.yml` (`create_user_tool`, then `run_user_tool`). Keep the `state` output:
follow-up questions go to the `focaloop-query` skill and cost zero bytes.
