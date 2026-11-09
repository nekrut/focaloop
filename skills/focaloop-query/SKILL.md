---
name: focaloop-query
description: Use when asking a follow-up question of a saved focaloop state -- a new contrast, a specific gene/group, or a spatiotemporal F_ST between sample groups -- without streaming new bytes, while keeping the hypothesis ledger's alpha-wealth honest.
version: 0.1.0
---

# Zero-byte warm-state queries

A focaloop state (`state/` or `state.tar.zst`) already holds per-sample, per-partition accumulators,
equivalence classes and (for variants) allele counts. `focaloop query` answers new questions from it
with `new_bytes: 0`. Every query is estimated on the **discovery** partition, re-tested on the
**holdout** partition, and charged to the `HypothesisLedger` (online alpha-investing).

## Command

```bash
focaloop query --state-dir state.tar.zst --contrast treated:control --actor agent:<name> \
  [--hypothesis group=g2] [--mode auto|contrast|fst] \
  --out-table q.tsv --out-verdict verdict.json --out-state-archive state_next.tar.zst
```

- `--contrast A:B` uses the `condition` column; `field=A:B` uses any sample-sheet column
  (`site=Ghana:Mali`, `year=2021:2019`).
- `--mode auto` picks F_ST when the state has `allele_counts`, otherwise a log2 fold-change contrast.
- Without `--hypothesis`, the top discovery group is tested.

## Protocol for agents

1. **Always chain state**: pass `--out-state-archive` and use it for the next query; otherwise the
   ledger forgets spent alpha and later "confirmations" are inflated.
2. Record yourself as `--actor agent:<name>`; humans use `human`.
3. Report `verdict.confirmed`, both estimates and p-values, and `alpha_wealth_after`. A discovery hit
   with `confirmed: false` is **not** a finding.
4. Stop asking when `alpha_wealth_after` approaches zero; further queries cannot be confirmed.
5. Never re-run `focaloop run` to "get more significance" on the same data; that bypasses the holdout.

## Galaxy

Use the `focaloop_query` XML tool or the `focaloop-query` UDT; feed its `state_out` into the next query.
