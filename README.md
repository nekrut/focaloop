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
| `focaloop panexpr --manifest M --ref-fasta F --ref-gtf G ...` | 5-stage adaptive RNA-seq differential expression loop (Stage-1 empirical aligner tournament, Stage-2 inverse-depth HTTP 206 / `xsra` streaming, Stage-3 C99 CSR EM + multi-reference orthology bridge, Stage-4 `PyDESeq2` + `V_tech/(V_tech+V_bio)` stop gate, Stage-5 31-mer zoom + `.focaloop.tar.zst` warm state) |
| `focaloop malariavar --manifest M [--ref-fasta F] --mode focused_cram\|wgs_stream ...` | *P. falciparum* variation & WHO drug-resistance loop (Mode A 55-window `.crai` CRAM HTTP Range surveillance or Mode B raw SRA streaming + C 31-mer/PLA human host sieve + depth stop gate) |
| `focaloop query --state-dir D\|A (--contrast A:B \| --args-json J) --out-table T --out-verdict V [--out-report R]` | zero-network-byte warm-state `PyDESeq2` contrast or sub-cohort WHO prevalence & Hudson $F_{ST}$ query guarded by `HypothesisLedger` |
| `focaloop validate <blueprint>...` | strict blueprint validation (all violations at once) |
| `focaloop run --blueprint B --spec S --state-dir D --out-table T --out-telemetry J [--state-archive A]` | declarative blueprint runner; exit 3 on an abort trap |
| `focaloop bait --query-fasta Q --inputs ... --state-dir D --out-tsv T` | Stage-0 canonical 31-mer screen of many runs |
| `focaloop steer --state-dir D --coords-bed B --out-bed O [--out-telemetry J]` | ONT ReadUntil enrich/deplete BED + `halt_run` |

## Layout

| Path | Contents |
|------|----------|
| `focaloop/panexpr/` | production 5-stage RNA-seq engine (`piscem`/`salmon`/`strobealign`/`bwa-mem2`/`hisat2`/`minimap2` tournament, C99 CSR EM `_em_accel.c` + `_fast_c.c`, `PyDESeq2` variance-ratio stop gate, 31-mer borderline zoom) |
| `focaloop/malariavar/` | production *P. falciparum* surveillance engine (55-window `.crai` CRAM HTTP Range fetcher, `xsra`/HTTPS streamer, C canonical 31-mer/PLA human host sieve `_sieve.c` + `_pla_sieve.c`, Dirichlet-multinomial diploid/polyclonal caller, WHO DRM + CNV + *hrp2/3* deletion + $F_{WS}$) |
| `focaloop/galaxy_runners.py`, `reports.py` | ergonomic manifest-driven Galaxy orchestrators, self-contained HTML telemetry reports, `.focaloop.tar.zst` state packaging, and zero-network-byte warm queries |
| `focaloop/state.py` | 9 Zstd-Parquet schemas + `FocaloopStateStore` |
| `focaloop/hooks.py` | 5 capability hooks: `BaitPrefilter`, `EqClassBackendProtocol` + `HolobiontLocusRegistry`, `coalesce_residual_spans` + `apply_external_priors`, `SteeringMask`, `HypothesisLedger` |
| `focaloop/blueprint.py` | typed blueprint model, `load_blueprint`, `validate_blueprint`, regime rules |
| `focaloop/streamer.py` | 3-tier resolver (CRAM/BAM range → AWS ODP via `xsra` → ENA HTTPS 206 fallback) + local files |
| `focaloop/kernels.py`, `tournament.py` | k-mer fallback kernels + tournament helpers |
| `focaloop/conditioner.py`, `stop_gate.py` | blueprint backward EM + stop gate |
| `focaloop/engine.py`, `ops.py`, `cli.py` | blueprint engine, `query`/`bait`/`steer`, unified `focaloop` CLI |
| `focaloop/galaxy_lint.py` | offline strict UDT and XML checks (+ official `galaxy-tool-util` when installed) |
| `blueprints/<id>/` | `blueprint.focaloop.yml`, `galaxy_udt.yml` (`GalaxyUserTool`), `galaxy_tool.xml` (Tool Panel XML), `test-data/` |
| `container/` | `Dockerfile`, `Apptainer.def`, and pinned `environment.yml` for `ghcr.io/galaxyproject/focaloop:0.2.0` |
| `galaxy/` | Galaxy 25.0+/27.0+ config, tool panel, TPV rules, and deployment notes |
| `skills/` | agent skills: `focaloop-run`, `focaloop-query`, `focaloop-distill` |
| `scripts/make_test_data.py` | regenerates seeded fixtures |

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

## Galaxy deployment (`0.2.0`)

- **Dual tool surfaces:** Every blueprint ships both a zero-admin `class: GalaxyUserTool` (`galaxy_udt.yml`, deployable via `POST /api/unprivileged_tools`) and a classic Galaxy Tool Panel XML wrapper (`galaxy_tool.xml`, profile `25.0`).
- **Production engines:** `focaloop panexpr` (5-stage RNA-seq differential expression with external aligner tournament, C99 CSR EM, `PyDESeq2` variance-ratio stop gate, and 31-mer borderline zoom) and `focaloop malariavar` (55-window `.crai` CRAM HTTP Range surveillance + raw SRA `xsra`/HTTPS streaming through a C canonical 31-mer/PLA human host sieve, WHO DRM/CNV/*hrp2-3*/$F_{WS}$ caller) emit compact `.focaloop.tar.zst` warm-state bundles queried in seconds with `0` network bytes by `focaloop query`.
- **Container resolution:** Build locally via `container/Dockerfile` or `container/Apptainer.def`; on Slurm + Apptainer clusters without public registry pushes, point TPV `singularity_container_id_override` at the shared `.sif` image.

License: MIT.
