# focaloop

`focaloop` answers a sequencing question from the fewest bytes that settle it. It streams FASTQ blocks
from local files or public INSDC archives (HTTP `Range`), picks a mapping kernel by tournament, keeps
strain/paralog ties as equivalence classes, re-solves them with backward expectation-maximization (EM),
and stops once a replicate-variance / posterior-precision gate says more bytes stop changing the answer.
Each study is a declarative `.focaloop.yml` **blueprint**, packaged for Galaxy both as a user-defined tool
(UDT) and as a ToolShed XML wrapper.

## Install and test

```bash
uv run --with numpy --with pandas --with pyarrow --with scipy --with pyyaml --with pytest pytest -v
pip install .            # provides the `focaloop` entry point
```

## Command line

| Command | Purpose |
|---------|---------|
| `focaloop validate <blueprint>...` | strict blueprint validation (all violations at once) |
| `focaloop run --blueprint B --spec S --state-dir D --out-table T --out-telemetry J [--state-archive A]` | stream, map, condition, gate; exit 3 on an abort trap |
| `focaloop query --state-dir D\|A --contrast A:B --actor X --out-table T --out-verdict V` | zero-new-byte contrast / F_ST query guarded by the hypothesis ledger |
| `focaloop bait --query-fasta Q --inputs ... --state-dir D --out-tsv T` | Stage-0 canonical 31-mer screen of many runs |
| `focaloop steer --state-dir D --coords-bed B --out-bed O [--out-telemetry J]` | ONT ReadUntil enrich/deplete BED + `halt_run` |

## Layout

| Path | Contents |
|------|----------|
| `focaloop/state.py` | 9 Zstd-Parquet schemas + `FocaloopStateStore` |
| `focaloop/hooks.py` | 5 capability hooks: `BaitPrefilter`, `EqClassBackendProtocol` + `HolobiontLocusRegistry`, `coalesce_residual_spans` + `apply_external_priors`, `SteeringMask`, `HypothesisLedger` |
| `focaloop/blueprint.py` | typed blueprint model, `load_blueprint`, `validate_blueprint`, regime rules |
| `focaloop/streamer.py` | 3-tier resolver (CRAM/BAM range → S3 `sra-pub-run-odp` → ENA HTTPS 206) + local files; byte-accounted blocks, discovery/holdout split |
| `focaloop/kernels.py`, `tournament.py` | k-mer kernels + external aligner registry; 25,000 real + 5 × 600 synthetic read tournament with the ≥ 0.80 mate-tie symmetry gate |
| `focaloop/conditioner.py`, `stop_gate.py` | backward EM with external log-priors; stop gate and design checks |
| `focaloop/engine.py`, `ops.py`, `cli.py` | `run`; `query`, `bait`, `steer`; CLI |
| `focaloop/variants.py`, `stats.py` | allele probes, CNV, deletions, F_WS; contrasts, BH, Hudson F_ST |
| `focaloop/galaxy_lint.py` | offline strict UDT and XML checks (+ official `galaxy-tool-util` when installed) |
| `blueprints/<id>/` | `blueprint.focaloop.yml`, `galaxy_udt.yml`, `galaxy_tool.xml`, `test-data/` |
| `container/` | `Dockerfile` + pinned `environment.yml` for `ghcr.io/galaxyproject/focaloop:0.1.0` |
| `galaxy/` | dedicated Galaxy 25.0+ config, tool panel, setup script, deploy notes |
| `skills/` | agent skills: `focaloop-run`, `focaloop-query`, `focaloop-distill` |
| `scripts/make_test_data.py` | regenerates all seeded fixtures (`--state` also rebuilds query states) |

## Blueprints

| id | Modality | Question |
|----|----------|----------|
| `panexpr` | expression | bulk multi-strain, multi-release, host-virus differential expression |
| `malariavar` | variants | *P. falciparum* SNP/indel, F_WS, *crt*/*mdr1*/*gch1*/*plasmepsin2/3* CNV, *hrp2/3* deletion, host-decoy sieve |
| `focaloop_query` | query | zero-byte warm-state contrast / spatiotemporal F_ST with ledger verification |
| `focaloop_bait` | bait | Stage-0 31-mer pan-repository SRA/ENA scan |

## Traps

| Code | Fires when | Default action |
|------|-----------|----------------|
| `LOW_MAP_RATE` | discovery-partition non-decoy map rate below threshold (first iteration with discovery reads, and at stop); holdout reads never count | abort |
| `TIE_SYMMETRY_FAIL` | every kernel keeps < 80% of strain/paralog ties while conditioning is active | abort |
| `BYTE_CAP_UNCONVERGED` | byte budget hit before the stop gate converged | warn |
| `CONFOUNDED_DESIGN` | contrast level missing, or condition perfectly confounded with batch | abort |

## Status and limits

- Builtin kernels (`kmer_eq`, `kmer_besthit`) are pure numpy. External aligners are registered by name and
  report unavailable until SAM/PAF adapters land; the tournament skips them and records a warning.
- Tier 1 (CRAM/BAM) and tier 2 (S3 SRA objects) are resolved but not yet decoded; streaming reads FASTQ
  via local files, plain URLs and ENA tier 3.
- The container image and the `focaloop` bioconda recipe are not published yet; the Galaxy XML
  `<requirement>` resolves only once they are.

License: MIT.
