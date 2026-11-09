"""Zstd-compressed Arrow/Parquet state schemas shared across Focaloop modalities.

Includes core equivalence-class + iteration telemetry tables and first-class
extension tables for:
  1. Stage-0 pan-repository 31-mer bait screening (`bait_prefilter`)
  2. Pangenome graph / colored de Bruijn / holobiont locus mapping (`graph_locus_map`)
  3. Retroactive unmapped/clipped byte-span index (`residual_spans`) and
     cross-modal external priors (`external_priors`)
  4. Real-time instrument steering masks & halt telemetry (`steering_decisions`)
  5. Sequential human/AI hypothesis ledger with online FDR wealth (`hypothesis_ledger`)
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TARGET_ACCUMULATOR_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("sample", pa.string()),
    ("condition", pa.string()),
    ("partition", pa.string()),  # "discovery" | "holdout"
    ("target_id", pa.string()),
    ("assembly_id", pa.string()),
    ("group_id", pa.string()),
    ("length_bp", pa.int32()),
    ("unique_primary", pa.float64()),
    ("unique_alt_or_antisense", pa.float64()),
    ("effective_count", pa.float64()),
    ("assignment_variance", pa.float64()),
])

AMBIGUOUS_EQ_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("partition", pa.string()),  # "discovery" | "holdout"
    ("eq_class_id", pa.int32()),
    ("fragment_count", pa.int32()),
    ("candidate_targets", pa.list_(pa.string())),
    ("candidate_assemblies", pa.list_(pa.string())),
    ("candidate_orientations", pa.list_(pa.bool_())),
    ("aln_loglik", pa.list_(pa.float64())),
    ("posterior_weights", pa.list_(pa.float64())),
])

ITERATION_TELEMETRY_SCHEMA = pa.schema([
    ("iteration", pa.int32()),
    ("modality", pa.string()),
    ("stage", pa.string()),
    ("cum_bytes", pa.int64()),
    ("pct_archive", pa.float64()),
    ("total_reads", pa.int64()),
    ("mapped_reads", pa.int64()),
    ("n_active_targets", pa.int64()),
    ("median_abs_delta_estimate", pa.float64()),
    ("median_tech_var_fraction", pa.float64()),
    ("wall_seconds", pa.float64()),
    ("stop_reason", pa.string()),
])

# Hook 1: Stage-0 Pan-Repository Bait Pre-Filter
BAIT_PREFILTER_SCHEMA = pa.schema([
    ("query_name", pa.string()),
    ("run_id", pa.string()),
    ("stream_tier", pa.string()),
    ("probe_bytes", pa.int64()),
    ("probe_reads", pa.int64()),
    ("hit_reads", pa.int64()),
    ("hit_rate", pa.float64()),
    ("wilson_ci95_low", pa.float64()),
    ("pass_gate", pa.bool_()),
])

# Hook 2: Colored de Bruijn / Pangenome Graph / Holobiont Locus Registry
GRAPH_LOCUS_MAP_SCHEMA = pa.schema([
    ("unit_id", pa.string()),
    ("unit_kind", pa.string()),  # "transcript" | "gene_body" | "cdbg_color" | "vg_snarl" | "plasmid_contig"
    ("group_id", pa.string()),
    ("assembly_id", pa.string()),
    ("kingdom", pa.string()),  # "host" | "bacteria" | "fungi" | "virus" | "protist" | "archaea"
    ("hgt_mobile_flag", pa.bool_()),
    ("length_bp", pa.int32()),
])

# Hook 3a: Residual Unmapped / Soft-Clipped Byte-Range Index
RESIDUAL_SPANS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("stream_tier", pa.string()),
    ("byte_start", pa.int64()),
    ("byte_end", pa.int64()),
    ("residual_kind", pa.string()),  # "unmapped" | "soft_clipped" | "chimeric_split"
    ("n_reads", pa.int32()),
    ("kmer_sketch_u64", pa.list_(pa.uint64())),
])

# Hook 3b: Cross-Modal External Priors (WGS, Proteomics, Chromatin, Perturbation)
EXTERNAL_PRIORS_SCHEMA = pa.schema([
    ("group_id", pa.string()),
    ("target_id", pa.string()),
    ("sample", pa.string()),
    ("modality", pa.string()),  # "wgs_copy_number" | "proteomics_ms" | "atac_chromatin" | "chip_histone"
    ("log_prior_weight", pa.float64()),
    ("confidence", pa.float64()),
    ("source_uri", pa.string()),
])

# Hook 4: Real-Time Instrument Steering Telemetry
STEERING_DECISION_SCHEMA = pa.schema([
    ("iteration", pa.int32()),
    ("n_enrich_targets", pa.int32()),
    ("n_deplete_targets", pa.int32()),
    ("n_enrich_kmers", pa.int64()),
    ("n_deplete_kmers", pa.int64()),
    ("halt_run", pa.bool_()),
    ("reason", pa.string()),
])

# Hook 5: Interactive Human/AI Hypothesis Ledger (Online FDR + Reusable Holdout)
HYPOTHESIS_LEDGER_SCHEMA = pa.schema([
    ("query_index", pa.int32()),
    ("query_id", pa.string()),
    ("actor", pa.string()),  # "human" | "agent:<name>"
    ("hypothesis_spec", pa.string()),
    ("discovery_estimate", pa.float64()),
    ("discovery_pvalue", pa.float64()),
    ("holdout_estimate", pa.float64()),
    ("holdout_pvalue", pa.float64()),
    ("alpha_allocated", pa.float64()),
    ("alpha_wealth_after", pa.float64()),
    ("confirmed", pa.bool_()),
])

SCHEMA_REGISTRY: dict[str, pa.Schema] = {
    "target_accumulators": TARGET_ACCUMULATOR_SCHEMA,
    "ambiguous_eq": AMBIGUOUS_EQ_SCHEMA,
    "iteration_telemetry": ITERATION_TELEMETRY_SCHEMA,
    "bait_prefilter": BAIT_PREFILTER_SCHEMA,
    "graph_locus_map": GRAPH_LOCUS_MAP_SCHEMA,
    "residual_spans": RESIDUAL_SPANS_SCHEMA,
    "external_priors": EXTERNAL_PRIORS_SCHEMA,
    "steering_decisions": STEERING_DECISION_SCHEMA,
    "hypothesis_ledger": HYPOTHESIS_LEDGER_SCHEMA,
}


class FocaloopStateStore:
    """Typed Zstd-Parquet state directory shared across all Focaloop stages and hooks."""

    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.parquet"

    def write(self, name: str, table: pa.Table, schema: pa.Schema | None = None) -> Path:
        target_schema = schema or SCHEMA_REGISTRY.get(name)
        if target_schema is not None:
            table = table.select(target_schema.names).cast(target_schema)
        out = self.path(name)
        pq.write_table(table, out, compression="zstd")
        return out

    def append(self, name: str, rows: dict[str, list], schema: pa.Schema | None = None) -> Path:
        target_schema = schema or SCHEMA_REGISTRY[name]
        new_tbl = pa.Table.from_pydict(rows, schema=target_schema)
        if self.exists(name):
            prev = self.read(name).select(target_schema.names).cast(target_schema)
            new_tbl = pa.concat_tables([prev, new_tbl])
        return self.write(name, new_tbl, schema=target_schema)

    def read(self, name: str) -> pa.Table:
        return pq.read_table(self.path(name))

    def exists(self, name: str) -> bool:
        return self.path(name).exists()
