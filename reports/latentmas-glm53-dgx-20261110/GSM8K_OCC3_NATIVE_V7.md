# GSM8K native OCC-3 sequential batch (v7), 2026-11-10

## Result

The v7 protocol batch covered GSM8K test items 2-19 (18 items) with the
four-role OCC-3 native-handoff condition (`occ3_agent_native_v7`).

- 17/18 items reached `complete` with `transport_valid: true` and
  `cleanup_confirmed: true` (proof and cleanup gates passed on every item).
- Item 19 never started: `blocked_before_start`,
  reason `insufficient_immutable_authority_or_deadline`. It requires fresh
  authorization; it was not silently retried.
- Pinned-scorer offline result (below): **15/17 correct** on this batch's
  completed items; **17/19 correct** across the full native cohort (item 0 and
  the v7 item-19 retry included). The two misses on this batch are item 12
  (predicted 12, gold 13 - the same item 12 the historical 14-item run and both
  API conditions missed on their overlap) and item 2, a formatting loss: the
  final line read `FINAL_ANSWER: $70,000` and the pinned upstream extractor
  (last-number regex, no boxed form) read `000`. Both API conditions emitted
  `\boxed{70000}` and parsed correctly. The same pinned rule scored all
  conditions; item 2 stays strict-incorrect and is not excluded. The earlier
  informal transcript join said 16/17; the pinned scorer is authoritative and
  the difference is exactly this item 2.
- Wrapper exited 2 (batch stopped, not crashed); zero failed inference
  terminals. Total item wall time 48.9 min (mean 172.5 s, median 171.7 s,
  range 102.9-264.5 s).
- Item 0 was validated separately (see `results/gsm8k-native-v7-item00-*`);
  item 1 failed setup in the prior batch and was not rerun in this batch.
  Both remain outside this batch's denominator.

The final numbers come from pinned offline scorers, not the runner (the plan
records `scoring: offline_separate_runner_groups_no_correctness_join`):
`score_native_gsm8k_v3.py` for the item-0 run and this batch (0.1.41-era
artifacts) and `score_native_gsm8k_v7.py` for the item-19 v7 retry (0.1.45
daemon, schemaRevision 47). Each scorer re-reads the frozen run directories,
validates the daemon/protocol identity, per-item approval bindings, context
inventories, standby records and cleanup gates, then applies the upstream
`extract_gsm8k_answer`/`normalize_answer` functions to the raw judger final and
string-compares against the manifest gold. Per-item counts and hashes (no
predictions) are in `results/native_v7/score-public-*.json`; the combined safe
rows and summary are `results/glm53_gsm8k_native_occ3_v7_safe.json` and
`results/glm53_gsm8k_native_occ3_v7_summary.json`. The earlier
`derived-join.json` (last `FINAL_ANSWER:` numeric from the judger transcript)
is retained for provenance; it agreed with the pinned scorer on every item
except item 2.

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
- 17/19 is a single sample on one benchmark. Exact McNemar against the paired
  20-item historical cohort (19 shared items: native 17, baseline 18,
  TextMAS-sequential 18) is p=1.0 - no significant difference.
- Item 1 did not run in this cohort (setup failure in the item-1 batch). It
  ran separately on 2026-11-12; see the last section. The 19-item cohort
  numbers above do not include it.

## Later item-19 attempts (2026-11-11)

Four further single-use authorizations were consumed on item 19 after the
daemon was upgraded to 0.1.45. The first three produced no score; the fourth
(v7) completed the item and is included in the 17/19 cohort number above. All
are recorded here so the batch-level readout above stays tied to its original
batch.

- v4 retry batch `native-v7-batch-20261111t045500z`: rejected
  `approval_expired_or_too_long` before any work (2 h expiry exceeds the 1 h
  cap). Re-armed attempt rejected `RemoteRejected` because `--old-root` was
  passed as a path instead of the session id. No model work ran.
- v5 retry batch `native-v7-batch-20261111t052800z`: fresh 55-minute
  authorization, all prior batches pinned. The item ran 137.7 s with all four
  roles spawned and real native work observed, then ended
  `failed_uncertain` with the runner's `item_exit_or_work_cleanup_bound`
  terminal (the item journal records a role generation failure/truncation).
  Cleanup was confirmed and the run artifacts are retained on the DGX under
  `runs/item-19/`. Authorization is single-use and terminal, so the item
  requires another fresh authorization; it was not retried automatically.

- v6 retry batch `native-v7-batch-gsm8k-item19-v6-20261111t055343z`: blocked
  before start. Root cause: the v6 controller variant pinned daemon
  `schemaRevision` 46, and the upgraded 0.1.45 daemon reports 47, so the
  controller's `DaemonClient.handshake` rejected with
  `unreviewed_schema_revision` before any model work. No authorization was
  consumed by model work; the batch is terminal with reason `Stop`.
- v7 retry batch `native-v7-batch-gsm8k-item19-v7-20261111t055343z`:
  regenerated controller/scorer variants with the schemaRevision-47 gate,
  validated offline (contract report, prior-scope pins over the full v6->v5->
  v4->v3->item1->item0 chain, live read-only daemon probe) and then ran.
  **Item 19 completed in 192.5 s with all transport and cleanup gates green,
  and the pinned scorer marked it correct.** The 17/19 cohort number above
  includes this item. Full artifact set retained on the DGX (111 role-cycle
  snapshots, standby records, cleanup proof).

## Item 1 (2026-11-12)

Item 1 ran once on 2026-11-12 in a one-item batch (`native-v7-batch-gsm8k-item1-20261112t091814z`)
with a fresh plan and authorization. It used the GSM8K v7 controller and
runner, prior root `native-v7-batch-gsm8k-item19-v7-20261111t055343z`, and the
pinned scorer `score_native_gsm8k_v7.py` (sha256
`ddb0f3ea854374bb637711bfcda67a62c41fa8aa45be206b9f303042fe597c1c`), the same scorer as the item-19 v7
retry. The server ran the PR #284 server-extension update (capture-busy fix,
deployed 2026-11-12 07:49 UTC).

- Outcome: `complete`, all transport, cleanup and format gates valid; scored
  **correct**. Item wall time 116.4 s; provider-reported tokens
  95,623 input + 6,346 output over 26 role records.
- First 20 items: **18/20** (wrong: 2, 12), by arithmetic across
  disjoint score reports, not one combined scorer run.
- Paired on items 0-19: native 18, baseline 19 (b=1, c=0, p=1.0),
  TextMAS sequential 19 (b=1, c=0, p=1.0). Exact McNemar.
- Item 1's only earlier attempt (batch `native-v7-batch-20261110t161939z`,
  2026-11-10) ended `blocked` with `standby_contract_not_met` before any score.
  It stays excluded. No other attempt on item 1 ran.
- Public score report: `results/native_v7/score-public-gsm8k-item01.json`.
  Source-bound totals: `results/glm53_gsm8k_native_occ3_v7_continuation_summary.json`.
