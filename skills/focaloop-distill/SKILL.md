---
name: focaloop-distill
description: Use when turning an existing IWC Galaxy workflow (.ga / .gxwf.yml) or an nf-core pipeline into a focaloop blueprint -- producing a tested blueprint.focaloop.yml, galaxy_udt.yml, galaxy_tool.xml and test-data that reproduce the workflow's primary result from a byte budget.
version: 0.1.0
---

# Distilling a workflow into a focaloop blueprint

A focaloop blueprint keeps the *question* a workflow answers and replaces its fixed full-file pipeline
with streamed blocks, a kernel tournament, backward EM and a stop gate. Distillation is done when the
blueprint reproduces the workflow's primary table on shared test data and all three packagings pass.

## Step-by-step protocol

1. **Summarize the source.**
   - IWC: read the `.ga` / `.gxwf.yml`; list inputs (formats, collections), steps (tool ids + key
     parameters), and the output(s) users actually consume (the "primary result").
   - nf-core: read `main.nf`, `nextflow_schema.json`, `conf/test.config`, `modules.json`; same list.
2. **Fix the question and modality.** One primary result per blueprint. Map it to a `modality`:
   counts/DE → `expression`; SNP/indel/CNV/deletion → `variants`; taxa/genes in mixed communities →
   `metagenomics`; peaks/accessibility → `chromatin`; presence screen → `bait`; follow-up on saved
   state → `query`.
3. **Collapse preprocessing into the kernel pool.** Trimming, alignment and quantification steps become
   `kernel_pool.kernels` candidates (e.g. HISAT2/STAR/Salmon steps → `salmon`, `kmer_eq`, `bwa-mem2`).
   Always include a tie-retaining kernel (`kmer_eq` or `salmon`) or `TIE_SYMMETRY_FAIL` will abort
   multi-strain catalogs.
4. **Translate decisions into `regime_rules`.** Workflow conditionals and parameter presets become
   restricted predicates over run facts (`n_assemblies > 1`, `has_decoy == true`, `always`).
5. **Set the stop gate from the workflow's tolerance.** Choose `min_posterior_precision`,
   `max_tech_var_fraction`, `delta_tol`, `block_bytes`, `max_bytes`, `holdout_fraction` (>= 0.1 when
   follow-up queries are expected).
6. **Declare all four traps** with actions (`abort` for invariants that make results wrong, `warn`
   for budget exhaustion). `TIE_SYMMETRY_FAIL.threshold` must be `>= 0.80`.
7. **Write `blueprint.focaloop.yml`** and run `focaloop validate` until it prints `OK`.
8. **Build compact test-data** (< 1 MB per file): a target FASTA with group/assembly attributes, a
   sample sheet, small FASTQ(.gz) with planted effects, and a `spec.yml`. Prefer extending
   `scripts/make_test_data.py` so fixtures stay seeded and byte-identical.
9. **Check equivalence.** Run the source workflow (Galaxy or `nextflow run -profile test`) and
   `focaloop run` on the same test data; compare primary results (sign and rank of effects, calls).
   Record the comparison in the blueprint `description`.
10. **Package for Galaxy.**
    - `galaxy_udt.yml`: `class: GalaxyUserTool`, `shell_command` with `\` line continuations, quoted
      `$(inputs.x)` text, every output claimed by `from_work_dir`, a real `help` object; validate with
      the udt-authoring `scripts/validate.py` (or `focaloop.galaxy_lint.official_udt_errors`).
    - `galaxy_tool.xml`: `<requirements>`, `<command detect_errors="exit_code">`, inputs/outputs,
      `<tests>` with `expect_num_outputs` and assertions against `test-data/`, markdown `<help>`,
      `<citations>`; run `planemo lint`.
11. **Add tests** to `tests/`: blueprint validation, UDT + XML lint (the parametrized tests pick up a
    new `blueprints/<id>/` automatically once its id is added to `BLUEPRINT_IDS`), and one CLI run
    asserting the planted effect.
12. **Done means proven**: `pytest` green, `planemo lint` clean, and the equivalence numbers written down.
