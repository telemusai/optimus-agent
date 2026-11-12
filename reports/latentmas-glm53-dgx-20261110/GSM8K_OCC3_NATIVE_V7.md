# GSM8K native OCC-3 sequential batch (v7), 2026-11-10

## Result

The v7 protocol batch covered GSM8K test items 2-19 (18 items) with the
four-role OCC-3 native-handoff condition (`occ3_agent_native_v7`).

- 17/18 items reached `complete` with `transport_valid: true` and
  `cleanup_confirmed: true` (proof and cleanup gates passed on every item).
- Item 19 never started: `blocked_before_start`,
  reason `insufficient_immutable_authority_or_deadline`. It requires fresh
  authorization; it was not silently retried.
- Derived offline join: **16/17 correct** on the completed items. The single
  miss is item 12 (predicted 12, gold 13) - the same item 12 the historical
  14-item run and both API conditions missed on their overlap.
- Wrapper exited 2 (batch stopped, not crashed); zero failed inference
  terminals. Total item wall time 48.9 min (mean 172.5 s, median 171.7 s,
  range 102.9-264.5 s).
- Item 0 was validated separately (see `results/gsm8k-native-v7-item00-*`);
  item 1 failed setup in the prior batch and was not rerun. Both remain
  outside this batch's denominator.

The join is a derived readout, not the runner's own scoring: the plan records
`scoring: offline_separate_runner_groups_no_correctness_join`. The method and
per-item rows are in `results/native_v7/derived-join.json`: the last
`FINAL_ANSWER:` numeric from each judger transcript (`item.json`), normalized
(string-compare on comma-stripped integers) against the manifest gold. All 17
judger transcripts ended in a parseable `FINAL_ANSWER`.

## Provenance

| Item | Value |
|---|---|
| Batch root | `/home/nvidia/occ3-deploy/native-v7-batch-20261110t175110z` |
| Started | 2026-11-10 17:51:10 UTC |
| Terminal | 2026-11-10 18:52:03 UTC (`batch_terminal`, stopped) |
| Ledger | `batch-ledger.jsonl`, 55 rows, sha256 `d49f86e40d787dd9fd57380d0ddc59e4cad5d2e1f41a6c371eebae6354a4cb60` |
| Outcome | `batch-outcome.json`, sha256 `28397ece186bae2444c02e63ca3c20ad02b52e00857b0ae8836d7f09a69969d3` |
| Resume | forbidden; per-item restart requires fresh immutable authorization |
| Scoring | offline, separate (no automatic correctness join in the runner) |

Every completed item carries sha256 artifact hashes for its journal, outcome,
raw judger transcript, and the standby role records; see
`results/native_v7/batch-outcome.json`. The standby roles run with reviewed
context only (no benchmark answers), single-item scope, and GPU-quiet
admission.

## Comparison limits

- The OCC-3 condition differs from the API conditions in channel, client,
  and output budget (see the main README). It is not a budget-matched
  comparison with the GSM8K API rows.
- `transport_valid` proves the native protocol completed its gates per item.
  It is not a claim that every hop avoided text fallback; the per-hop
  admitted/completed split is in each item journal, not summarized here.
- 16/17 is a single sample on one benchmark; no significance test is claimed
  against the 92.9% historical adjudicated 14-item run (overlapping items,
  different protocol version).
- Item 19 remains NOT-RUN.
