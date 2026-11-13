"""Typed Arrow/Parquet state tables shared by the mapping, solving and DE stages."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TARGET_COUNTS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("sample", pa.string()),
    ("condition", pa.string()),
    ("target_id", pa.string()),
    ("assembly_id", pa.string()),
    ("gene_id", pa.string()),
    ("group_id", pa.string()),
    ("symbol", pa.string()),
    ("biotype", pa.string()),
    ("length_bp", pa.int32()),
    ("unique_sense", pa.float64()),
    ("unique_antisense", pa.float64()),
    ("effective_count", pa.float64()),
    ("assignment_variance", pa.float64()),
    ("group_assignment_variance", pa.float64()),
    ("wgs_dna_reads", pa.float64()),
    ("dna_present_prob", pa.float64()),
])

AMBIGUOUS_READS_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("eq_class_id", pa.int32()),
    ("fragment_count", pa.int32()),
    ("candidate_targets", pa.list_(pa.string())),
    ("candidate_assemblies", pa.list_(pa.string())),
    ("candidate_is_sense", pa.list_(pa.bool_())),
    ("aln_loglik", pa.list_(pa.float64())),
    ("posterior_weights", pa.list_(pa.float64())),
])

GENE_DE_SCHEMA = pa.schema([
    ("iteration", pa.int32()),
    ("group_id", pa.string()),
    ("gene_id", pa.string()),
    ("symbol", pa.string()),
    ("dna_present_prob", pa.float64()),
    ("base_mean", pa.float64()),
    ("streamed_reads", pa.float64()),
    ("log2fc", pa.float64()),
    ("lfc_se_deseq2", pa.float64()),
    ("var_placement_ambiguity", pa.float64()),
    ("lfc_se_total", pa.float64()),
    ("ci95_low", pa.float64()),
    ("ci95_high", pa.float64()),
    ("pvalue", pa.float64()),
    ("padj", pa.float64()),
    ("tech_var_fraction", pa.float64()),
    ("demand_score", pa.float64()),
    ("status", pa.string()),
])

ITERATION_TELEMETRY_SCHEMA = pa.schema([
    ("iteration", pa.int32()),
    ("stage", pa.string()),
    ("cum_bytes", pa.int64()),
    ("pct_archive", pa.float64()),
    ("total_reads", pa.int64()),
    ("mapped_reads", pa.int64()),
    ("n_expressed_genes", pa.int64()),
    ("median_abs_delta_lfc", pa.float64()),
    ("p90_abs_delta_lfc", pa.float64()),
    ("p95_abs_delta_lfc", pa.float64()),
    ("median_tech_var_fraction", pa.float64()),
    ("n_deg_fdr05", pa.int64()),
    ("wall_seconds", pa.float64()),
    ("stop_reason", pa.string()),
])

DOTHASH_SKETCH_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("sample", pa.string()),
    ("condition", pa.string()),
    ("dim", pa.int32()),
    ("n_reads_sketched", pa.int64()),
    ("n_distinct_syncmers_seen", pa.int64()),
    ("max_within_cond_cosine", pa.float64()),
    ("max_within_cond_containment", pa.float64()),
    ("dothash_weight_scale", pa.float64()),
    ("sketch_int16", pa.binary(4096)),
])


class StateStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.parquet"

    def write(self, name: str, table: pa.Table, schema: pa.Schema | None = None) -> Path:
        if schema is not None:
            table = table.select(schema.names).cast(schema)
        p = self.path(name)
        pq.write_table(table, p, compression="zstd")
        return p

    def read(self, name: str) -> pa.Table:
        return pq.read_table(self.path(name))

    def exists(self, name: str) -> bool:
        return self.path(name).exists()
