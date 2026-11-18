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
| `focaloop expressionloop --manifest M --ref-fasta F --ref-gtf G ...` | 5-stage adaptive RNA-seq differential expression loop (Stage-1 empirical aligner tournament, Stage-2 inverse-depth HTTP 206 / `xsra` streaming, Stage-3 C99 CSR EM + multi-reference orthology bridge, Stage-4 `PyDESeq2` + `V_tech/(V_tech+V_bio)` stop gate, Stage-5 31-mer zoom + `.focaloop.tar.zst` warm state) |
| `focaloop variationloop --analysis-mode focused_cram\|wgs_stream\|diploid_wgs ...` | Pathogen CRAM range / WGS host-depleted surveillance & diploid human VariationLoop calling (64-kb paged C accumulator + multi-k Simple Omnitig reassembly + WHO DRM/CNV/*hrp2/3*/$F_{WS}$) |
| `focaloop geneloop --genome G --clade V [--proteins P] ...` | 5-stage adaptive eukaryotic genome annotation loop (homology/Tiberius seed, OpenMP open-syncmer depletion filter, Good-Turing UCB RNA-seq streaming, micro-exon & structural ncRNA rescue, GFF3/FAA/GTF + warm state) |
| `focaloop query --state-dir D\|A (--contrast A:B \| --args-json J) --out-table T --out-verdict V [--out-report R]` | zero-network-byte warm-state `PyDESeq2` contrast or sub-cohort WHO prevalence & Hudson $F_{ST}$ query guarded by `HypothesisLedger` |
| `focaloop validate <blueprint>...` | strict blueprint validation (all violations at once) |
| `focaloop run --blueprint B --spec S --state-dir D --out-table T --out-telemetry J [--state-archive A]` | declarative blueprint runner; exit 3 on an abort trap |
| `focaloop bait --query-fasta Q --inputs ... --state-dir D --out-tsv T` | Stage-0 canonical 31-mer screen of many runs |
| `focaloop steer --state-dir D --coords-bed B --out-bed O [--out-telemetry J]` | ONT ReadUntil enrich/deplete BED + `halt_run` |

## Layout

| Path | Contents |
|------|----------|
| `focaloop/panexpr/` | production 5-stage RNA-seq engine (`piscem`/`salmon`/`strobealign`/`bwa-mem2`/`hisat2`/`minimap2` tournament, C99 CSR EM + 31-mer filter `_fast_c.py`, `PyDESeq2` variance-ratio stop gate, isoform-aware EM) |
| `focaloop/malariavar/` | production variation & surveillance engine (`varloop_core.c` 64-kb lazy-paged C accumulator + multi-k Simple Omnitig reassembly, 55-window `.crai` CRAM HTTP Range fetcher, `xsra`/HTTPS streamer, C canonical 31-mer/PLA host depletion filter, diploid + WHO DRM/CNV/*hrp2/3*/$F_{WS}$) |
| `focaloop/geneloop/` | production 5-stage eukaryotic genome annotation engine (`glcore.c` OpenMP open-syncmer depletion filter, `orf_scout.c`, Good-Turing UCB intron judge, micro-exon + structural ncRNA + retrocopy rescue) |
| `focaloop/runners_*.py`, `reports.py` | Galaxy orchestrators (`runners_expression.py`, `runners_variation.py`, `runners_geneloop.py`, `galaxy_runners.py`), self-contained HTML telemetry reports, `.focaloop.tar.zst` state packaging, and zero-network-byte warm queries |
| `focaloop/state.py` | 9 Zstd-Parquet schemas + `FocaloopStateStore` |
| `focaloop/hooks.py` | 5 capability hooks: `BaitPrefilter`, `EqClassBackendProtocol` + `HolobiontLocusRegistry`, `coalesce_residual_spans` + `apply_external_priors`, `SteeringMask`, `HypothesisLedger` |
| `focaloop/blueprint.py` | typed blueprint model, `load_blueprint`, `validate_blueprint`, regime rules |
| `focaloop/streamer.py` | 3-tier resolver (CRAM/BAM range → AWS ODP via `xsra` → ENA HTTPS 206 fallback) + local files |
| `focaloop/kernels.py`, `tournament.py` | k-mer fallback kernels + tournament helpers |
| `focaloop/conditioner.py`, `stop_gate.py` | blueprint backward EM + stop gate |
| `focaloop/engine.py`, `ops.py`, `cli.py` | blueprint engine, `query`/`bait`/`steer`, unified `focaloop` CLI |
| `focaloop/galaxy_lint.py` | offline strict UDT and XML checks (+ official `galaxy-tool-util` when installed) |
| `tools/focaloop/` | tools-iuc wrappers (`expressionloop.xml`, `variationloop.xml`, `geneloop.xml`, `focaloop_query.xml`, `macros.xml`, `.shed.yml`, `test-data/`) |
| `blueprints/<id>/` | `blueprint.focaloop.yml`, `galaxy_udt.yml` (`GalaxyUserTool`), `galaxy_tool.xml` (Tool Panel XML), `test-data/` |
| `container/` | `Dockerfile`, `Apptainer.def`, and pinned `environment.yml` for `ghcr.io/galaxyproject/focaloop:0.3.0` |
| `galaxy/` | Galaxy 25.0+/27.0+ config, tool panel, TPV rules, and deployment notes |
| `skills/` | agent skills: `focaloop-run`, `focaloop-query`, `focaloop-distill` |
| `scripts/make_test_data.py` | regenerates seeded fixtures |

## Blueprints

| id | Modality | Question |
|----|----------|----------|
| `panexpr` | expression | bulk multi-strain, multi-release, host-virus differential expression |
| `malariavar` | variants | *P. falciparum* SNP/indel, F_WS, *crt*/*mdr1*/*gch1*/*plasmepsin2/3* CNV, *hrp2/3* deletion, host-decoy filter |
| `geneloop` | annotation | eukaryotic structural gene annotation, open-syncmer depletion filter, Good-Turing UCB RNA-seq streaming, ncRNA & micro-exon rescue |
| `focaloop_query` | query | zero-byte warm-state contrast / spatiotemporal F_ST with ledger verification |
| `focaloop_bait` | bait | Stage-0 31-mer pan-repository SRA/ENA scan |

## Traps

| Code | Fires when | Default action |
|------|-----------|----------------|
| `LOW_MAP_RATE` | discovery-partition non-decoy map rate below threshold (first iteration with discovery reads, and at stop); holdout reads never count | abort |
| `TIE_SYMMETRY_FAIL` | every kernel keeps < 80% of strain/paralog ties while conditioning is active | abort |
| `BYTE_CAP_UNCONVERGED` | byte budget hit before the stop gate converged | warn |
| `CONFOUNDED_DESIGN` | contrast level missing, or condition perfectly confounded with batch | abort |

## Galaxy deployment (`0.3.0`)

- **Dual tool surfaces:** tools-iuc XML wrappers live under `tools/focaloop/` (`expressionloop.xml`, `variationloop.xml`, `geneloop.xml`, `focaloop_query.xml`, `macros.xml`, `.shed.yml`), alongside `class: GalaxyUserTool` YAML definitions under `blueprints/*/galaxy_udt.yml`.
- **Production engines:** `focaloop expressionloop`, `focaloop variationloop`, and `focaloop geneloop` emit compact `.focaloop.tar.zst` warm-state bundles queried in seconds with `0` network bytes by `focaloop query`.
- **Container resolution:** Build locally via `container/Dockerfile` or `container/Apptainer.def`; on Slurm + Apptainer clusters without public registry pushes, point TPV `singularity_container_id_override` at the shared `.sif` image.

License: MIT.
