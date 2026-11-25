# focaloop

Standard sequencing workflows treat every public dataset as an all-or-nothing batch job—pulling full raw archives over the network, writing hundreds of gigabytes of disposable alignments, and discarding the statistical state the moment a summary table appears. `focaloop` turns sequencing analysis into a byte-budgeted feedback loop. It streams only the data required to settle a biological question—starting from compact pre-assembled Logan unitigs when available and falling back to AWS Open Data (`xsra`) or HTTP `Range` slices—updates posterior uncertainty after each wave, and stops once technical sampling noise drops below biological variation. Download stops when the answer stops moving. Every run emits a self-contained `.focaloop.tar.zst` state bundle, allowing researchers and automated agents to run new contrasts, sub-cohort filters, and locus checks in seconds with zero additional downloads.

Version `0.4.1` unifies three domain loops—`expressionloop` for RNA-seq differential expression, `variationloop` for diploid and mixed-clone variant calling, and `geneloop` for eukaryotic structural gene annotation—with `query` for instant post-hoc interrogation of saved state bundles. All three loops share a three-way reference layer accepting built-in Galaxy CVMFS assemblies, local FASTA and annotation files, or automatic NCBI RefSeq and GenBank retrieval. Multi-reference orthology catalogs in `expressionloop` also expand related microbial or viral strains automatically from NCBI. Both diploid genotypes and continuous within-sample allele fractions are called across multi-sample cohorts in `variationloop`. Sister-species homology proteomes are retrieved automatically from NCBI RefSeq or OrthoDB v12 in `geneloop` alongside deep-learning gene prediction and adaptive RNA-seq splice refinement. The same code runs on the command line, in standalone containers, and on public Galaxy servers.

## Installation and containers

Install the complete environment and precompiled C99/OpenMP binaries from Bioconda.

```bash
mamba install -c conda-forge -c bioconda focaloop=0.4.1
```

Build from source to compile all 16 portable C99/OpenMP binaries and shared libraries into `PREFIX/bin` and `PREFIX/share/focaloop/native_cache` (`FOCALOOP_PORTABLE_ISA=1`).

```bash
make install PREFIX=/usr/local PYTHON=python3
pytest -q
```

Pull the standalone container image—which adds `xsra`, `themisto`, CPU `Tiberius`, and all seven clade weight packs for offline eukaryotic gene prediction—or download the prebuilt Apptainer `.sif` from the `v0.4.1` GitHub Release.

```bash
apptainer build focaloop_0.4.1.sif docker://ghcr.io/nekrut/focaloop:0.4.1
# or download focaloop_0.4.1.sif + focaloop_0.4.1.sif.sha256 from releases/tag/v0.4.1
```

## Command line

| Command | Purpose |
|---|---|
| `focaloop expressionloop --manifest M --contrast A:B ...` | Five-stage adaptive RNA-seq loop: empirical mapper tournament, Logan unitig or inverse-depth `xsra`/HTTP `206` streaming, C99 CSR expectation-maximization with multi-strain orthology bridging, `PyDESeq2` variance-ratio stopping rule, and 31-mer borderline zoom |
| `focaloop variationloop --ploidy diploid\|haploid ...` | Cohort and single-sample variant calling from `list:paired` FASTQ, BAM, indexed CRAM ranges, or public SRA/Logan unitigs, using a 64-kb paged C accumulator, local haplotype reassembly, continuous `FORMAT/WSAF` or diploid genotyping, and parallel `bcftools` merging |
| `focaloop geneloop --clade C --protein-source ncbi\|orthodb\|history\|none ...` | Five-stage eukaryotic genome annotation loop: automatic NCBI RefSeq or OrthoDB v12 homology retrieval, `Tiberius` deep-learning seed (`auto` fallback to `fast_evidence`), OpenMP open-syncmer transcript filtering, Good-Turing UCB RNA-seq streaming, micro-exon rescue, and structural ncRNA calls |
| `focaloop query --state S.focaloop.tar.zst --contrast A:B [--subset ...] ...` | Zero-download follow-up `PyDESeq2` contrast (with metadata `--subset` filtering and `--design` overrides) or sub-cohort `WSAF` and Hudson $F_{ST}$ comparison tracked by `HypothesisLedger` |
| `focaloop validate <blueprint>...` | Strict validation of declarative `.focaloop.yml` blueprints |
| `focaloop run --blueprint B --spec S --state-dir D ...` | Declarative blueprint runner (exits `3` when an integrity check aborts execution) |
| `focaloop bait --query-fasta Q --inputs ... --out-tsv T` | Canonical 31-mer pre-screen across local files, URLs, or public run accessions |
| `focaloop steer --state-dir D --coords-bed B --out-bed O` | Oxford Nanopore ReadUntil enrich/deplete BED export and stopping telemetry |

## Key `0.4.1` capabilities

- **Logan-first public streaming (`--logan-mode auto|logan_only|off`):** Public SRA/ENA accessions stream pre-assembled Logan unitigs or transcript contigs (`10–27x` smaller than raw reads) before falling back to `xsra` on AWS Open Data or ENA HTTP `206` byte ranges.
- **Three-way reference and annotation resolution:** Every loop accepts Galaxy `all_fasta` CVMFS paths (with verified RefSeq GFF3/GTF auto-retrieval via `--auto-ref-gtf`), local files, or `--ncbi-accession` assemblies. `expressionloop` also joins extra strain genomes (`--extra-ncbi-accession`, `--auto-expand-strains`).
- **Automatic homology proteomes in `geneloop`:** `--protein-source ncbi` and `--protein-source orthodb` infer the target taxonomic family from CVMFS metadata or NCBI accessions, exclude the target species to prevent self-leakage, and retrieve diverse sister-species proteomes (`--orthodb-mode representatives|full_partition`).
- **Haploid and mixed-infection variant calling in `variationloop`:** `--ploidy haploid` models clonal mixtures and microbial populations via per-sample `FORMAT/WSAF` (`--min-wsaf 0.25`) alongside standard `--ploidy diploid` genotyping and optional pathogen surveillance panels (`--surveillance-panel pf3d7`).
- **Compendium-scale warm queries in `query`:** `--subset` filters large multi-perturbation state bundles (for example, `'plate_id=P1 timepoint=60m'`) and re-fits contrasts in seconds with zero network traffic.

## Repository layout

| Path | Contents |
|---|---|
| `Makefile`, `scripts/build_native.py` | Pure-stdlib portable C99/OpenMP build system (`make install-native`) for Bioconda, Docker, Apptainer, and source installs |
| `focaloop/panexpr/`, `runners_expression.py` | Adaptive RNA-seq engine, C99 CSR EM solver, multi-reference orthology catalog, and Logan unitig expression quantifier |
| `focaloop/malariavar/`, `runners_variation.py` | Paged C pileup accumulator (`varloop_core.c`), multi-$k$ local reassembly, Logan unitig variant caller, CRAM `.crai` range fetcher, and `WSAF`/diploid cohort caller |
| `focaloop/geneloop/`, `runners_geneloop.py` | `glcore.c` OpenMP syncmer engine, `orf_scout.c`, `homology_fetch.py` (NCBI RefSeq and OrthoDB v12), `Tiberius` integration, and splice/micro-exon refinement |
| `focaloop/logan.py`, `cvmfs_annot.py`, `ncbi_ref.py` | Public Logan S3 probes, Galaxy CVMFS genome/taxon resolvers, and verified NCBI RefSeq/GenBank fetchers |
| `focaloop/state.py`, `archive.py`, `hooks.py`, `ops.py` | Zstd-Parquet schemas, `.focaloop.tar.zst` archiver (2 GiB decompression guard), `HypothesisLedger`, and zero-download `query`/`bait`/`steer` operations |
| `tools/focaloop/` | Galaxy XML wrapper suite (`0.4.1+galaxy0`, profile `25.0`): `expressionloop.xml`, `variationloop.xml`, `geneloop.xml`, `focaloop_query.xml`, `macros.xml`, `.shed.yml`, and `test-data/` |
| `container/` | Reproducible `Dockerfile`, `Apptainer.def`, and `environment.yml` for `ghcr.io/nekrut/focaloop:0.4.1` |
| `skills/` | Agent skills (`focaloop-run`, `focaloop-query`, `focaloop-distill`) |
| `blueprints/` | Frozen `0.1–0.2` declarative `.focaloop.yml` and `GalaxyUserTool` archival prototypes |

## Integrity checks

| Code | Condition | Default action |
|---|---|---|
| `LOW_MAP_RATE` | Discovery-partition non-decoy mapping rate falls below threshold (holdout reads stay isolated) | abort |
| `TIE_SYMMETRY_FAIL` | Every candidate mapper retains fewer than 80% of strain or paralog ties while conditioning is active | abort |
| `BYTE_CAP_UNCONVERGED` | Archive byte cap reached before the variance-ratio stopping threshold is met | warn |
| `CONFOUNDED_DESIGN` | Contrast group missing or experimental factor confounded with batch | abort |

## Galaxy deployment (`0.4.1+galaxy0`)

The four Galaxy wrappers in `tools/focaloop/` (`suite_focaloop` `0.4.1+galaxy0` on the Main Galaxy Tool Shed) support the full three-source architecture—local Galaxy History collections, public SRA/ENA Logan and raw-read streaming, and CVMFS or NCBI reference selection—and emit self-contained interactive HTML reports alongside tabular, VCF, and GFF3 outputs plus `.focaloop.tar.zst` state bundles. Configure `TIBERIUS_WEIGHTS_DIR`, `FOCALOOP_ORTHODB_CACHE_DIR`, and read-only `/cvmfs` mounts via Total Perspective Vortex (TPV), and add the four tool IDs to Galaxy's `sanitize_allowlist.txt` so interactive HTML reports render in place without stripping embedded plots. No workstation-specific fallback paths remain anywhere in the package, and `geneloop` falls back automatically to `fast_evidence` whenever deep-learning weights are absent on a target cluster.

License: MIT.
