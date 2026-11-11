# LatentMAS benchmarks on GLM-5.3 (DGX, 2026-11-10): interim results

These are interim results from running the LatentMAS evaluation suite
([Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS) at `9a9e4d3`) against
GLM-5.3. The API conditions use the upstream prompts and scoring. The OCC-3
condition adapts the role prompts and includes the answer recoveries below. The model is served by SGLang with the OCC-3
bridge. The suite adds an Optimus OCC-3 native-handoff condition. The run was paused at
2026-11-10 09:29 UTC so that PR #279 could be deployed. Reruns on the updated stack will
be added to this directory. All numbers are measured. Untested items are listed under
NOT-RUN. The post-update smoke failed; see [smoke-validation.md](smoke-validation.md).
It is diagnostic evidence, not an additional benchmark sample.

Update: GPQA baseline is now complete (73/198). The separate 38-item continuation
on the PR #281 overlay is documented below; the historical setup and OCC-3
results remain unchanged. TextMAS sequential is also complete at **103/198
(52.02%)**, with four calls per item; it is not a budget-matched comparison. See
[GPQA_TEXTMAS_SEQUENTIAL.md](GPQA_TEXTMAS_SEQUENTIAL.md). TextMAS hierarchical
is complete at **10/198 (5.05%)**. The low result is preserved without answer
repair or rerun; source-contract review found no discrepancy. See
[GPQA_TEXTMAS_HIERARCHICAL.md](GPQA_TEXTMAS_HIERARCHICAL.md).
MedQA300 baseline is complete at **174/300 (58.00%)**; see
[MEDQA_BASELINE.md](MEDQA_BASELINE.md). TextMAS sequential is complete at
**268/300 (89.33%)**; see [MEDQA_TEXTMAS_SEQUENTIAL.md](MEDQA_TEXTMAS_SEQUENTIAL.md).
The GSM8K native OCC-3 v7 batch (items 2-19) is complete at **16/17 correct
on completed items** with item 19 never started; see
[GSM8K_OCC3_NATIVE_V7.md](GSM8K_OCC3_NATIVE_V7.md).
Four calls per item versus one baseline call are not budget matched.
A code-task sweep on the same server started 2026-11-11 with the
`runner.py` harness: HumanEval+ (n=164) and MBPP+ (n=378) are complete for all
three conditions (MBPP+ TextMAS hierarchical 323/378, 85.45%). ARC-Easy baseline
and TextMAS sequential are complete (285/300, 95.0% and 290/300, 96.67%);
ARC-Easy hierarchical and all ARC-Challenge conditions remain in flight on a
restarted runner and are listed under NOT-RUN. See the Code-task sweep and
ARC-Easy sections below. A second item-19 retry attempt under the v7
protocol ran 2026-11-11 and ended `failed_uncertain` without a score; see
[GSM8K_OCC3_NATIVE_V7.md](GSM8K_OCC3_NATIVE_V7.md).

## Setup

This table describes the historical runs. The GPQA continuation records its
updated runtime separately in [GPQA_COMPLETION.md](GPQA_COMPLETION.md). The new
sequential and hierarchical results record their execution and scoring contracts
in [GPQA_TEXTMAS_SEQUENTIAL.md](GPQA_TEXTMAS_SEQUENTIAL.md) and
[GPQA_TEXTMAS_HIERARCHICAL.md](GPQA_TEXTMAS_HIERARCHICAL.md). Historical statistics
and OCC-3 labels below do not describe these new HTTP TextMAS runs.

| Item | Value |
|---|---|
| Model | GLM-5.3 FP8 (`GlmMoeDsaForCausalLM`, 78 layers, MTP speculative decoding), TP 8 on 8x B300, FP8 E4M3 KV cache |
| Server | SGLang v0.5.21 (`e00930c5`) with the OCC-3 bridge plugin, image `9728d909511b` (built 2026-11-08, before PR #279) |
| Endpoint | OpenAI-compatible chat completions on loopback `127.0.0.1:30001` |
| Client for OCC-3 runs | Optimus Agent 0.1.44 at `62677f014` (merge of PR #277, before PR #279) |
| Sampling, API conditions | temperature 0.6, top_p 0.95, max_tokens 4096 per model call, one sample per item (pass@1), up to 16 requests in flight |
| Sampling, OCC-3 condition | Optimus provider defaults: no explicit temperature or top_p, output budget 131,072 tokens per call (`defaultMaxTokens`) |
| Items | LatentMAS repo loaders. Sampled subsets use seed 42. Manifest SHA-256 values are in `MANIFESTS.json` |
| Statistics | Wilson 95% intervals; exact McNemar test against the baseline on paired items (b = baseline-only correct, c = condition-only correct) |

Conditions:

- `baseline`: one agent with the LatentMAS baseline prompt (API call).
- `text_mas_sequential`, `text_mas_hierarchical`: the LatentMAS TextMAS four-agent
  prompts. Text context is passed between agents. Each agent call is a separate API call
  with its own 4096-token cap.
- `occ3_sequential`: the same four roles (planner, critic, refiner, judger) as Optimus
  subagents on `dgx-glm53-occ3/glm-5.3`:
  - The GLM-5.3 Optimus root session on the DGX ran the items. It spawned the children
    with the prompts in `harness/occ3_v3_prompts.py`.
  - Role wording comes from the LatentMAS latent-method prompts. A required handoff
    protocol is added: arm, emit the role output, publish.
  - Each hop attempts native `occ3_handoff`; admitted and completed are distinct.
    The judger is instructed to emit `FINAL_ANSWER: <answer>`.
  - Items ran one at a time, with no other benchmark traffic.
  - A busy decline was retried up to 3 times. After that, the hop used a text fallback
    marked `FALLBACK_USED`.
  - The children are full Optimus agents, with the Optimus system prompt and tools.
- `latent_mas`: NOT RUN. It needs in-process hidden-state and KV access, which a served
  API cannot provide.

## Results

### AIME 2024 and AIME 2025 (full sets, n = 30 each)

| Task | Condition | Correct | Accuracy | 95% CI | McNemar vs baseline | Truncated | Completion tokens/item | Mean latency |
|---|---|---|---|---|---|---|---|---|
| AIME 2024 | baseline | 11/30 | 36.7% | 21.9-54.5% | - | 63% | 3,167 | 24.0 s |
| AIME 2024 | TextMAS sequential | 16/30 | 53.3% | 36.1-69.8% | b=0, c=5, p=0.0625 | 83% | 13,088 | 99.4 s |
| AIME 2024 | TextMAS hierarchical | 18/30 | 60.0% | 42.3-75.4% | b=0, c=7, p=0.0156 | 60% | 11,236 | 79.0 s |
| AIME 2025 | baseline | 10/30 | 33.3% | 19.2-51.2% | - | 67% | 3,336 | 23.5 s |
| AIME 2025 | TextMAS sequential | 11/30 | 36.7% | 21.9-54.5% | b=0, c=1, p=1.0 | 80% | 13,587 | 273.1 s |
| AIME 2025 | TextMAS hierarchical | 13/30 | 43.3% | 27.4-60.8% | b=1, c=4, p=0.375 | 70% | 12,539 | 93.1 s |

- "Truncated" is the share of items in which at least one model call stopped at the
  4096-token cap.
- Latency is the per-item time with up to 16 requests in flight.

### Code-task sweep (HumanEval+, MBPP+; interim)

Run 2026-11-11 with `harness/runner.py` on the same GLM-5.3 SGLang endpoint.
Prompts and scoring mirror the repo: one agent for baseline, the four-agent
TextMAS prompts for sequential/hierarchical, the last ```python fenced block
extracted and executed with the gold tests in the offline `latentmas-sandbox:py311`
docker container (no network, read-only, uid 65534). One sample per item.

| Task | Condition | Correct | Accuracy | 95% CI | McNemar vs baseline | Truncated | Completion tokens/item | Mean latency |
|---|---|---|---|---|---|---|---|---|
| HumanEval+ | baseline | 142/164 | 86.6% | 80.5-91.0% | - | 10.4% | 1,404 | 10.6 s |
| HumanEval+ | TextMAS sequential | 143/164 | 87.2% | 81.2-91.5% | b=5, c=6, p=1.0 | 29.3% | 6,890 | 51.2 s |
| HumanEval+ | TextMAS hierarchical | 149/164 | 90.9% | 85.5-94.4% | b=5, c=12, p=0.143 | 10.4% | 3,813 | 25.8 s |
| MBPP+ | baseline | 303/378 | 80.2% | 75.8-83.9% | - | 8.5% | 1,162 | 8.4 s |
| MBPP+ | TextMAS sequential | 303/378 | 80.2% | 75.8-83.9% | b=13, c=13, p=1.0 | 25.7% | 6,415 | 48.8 s |
| MBPP+ | TextMAS hierarchical | 323/378 | 85.4% | 81.5-88.6% | b=6, c=26, p=0.000535 | 9.5% | 3,295 | 21.4 s |

- "Truncated" is the share of items in which at least one model call stopped
  at the 4096-token cap; HumanEval+ sequential truncation is visible in its
  29.3% rate and its mean latency.
- HumanEval+ hierarchical is above its baseline Wilson interval overlap:
  +4.3 points, McNemar p=0.143 (not significant at 5%). MBPP+ hierarchical is
  +5.3 points above its 80.2% baseline and above the baseline's 75.8-83.9%
  Wilson interval; McNemar b=6, c=26, p=0.000535 (significant at 5%).
- Raw model outputs are not committed (11 MB per file); per-item correctness,
  token and latency rows plus source sha256 values are in
  `results/*_safe.json`, and aggregate metrics in `results/*_summary.json`.
- ARC-Easy hierarchical and all ARC-Challenge conditions (restarted runner
  finishing them) are NOT-RUN and excluded from the tables.

### ARC-Easy (interim, 300 items)

From the same 2026-11-11 `runner.py` sweep on the same GLM-5.3 SGLang endpoint,
run on the ARC-Easy test split. One sample per item, upstream LatentMAS prompts
and scoring; gold answers are letters. ARC-Easy TextMAS hierarchical and all
ARC-Challenge conditions are NOT-RUN (restarted runner finishing them) and
excluded.

| Task | Condition | Correct | Accuracy | 95% CI | McNemar vs baseline | Truncated | Completion tokens/item | Mean latency |
|---|---|---|---|---|---|---|---|---|
| ARC-Easy | baseline | 285/300 | 95.0% | 91.9-96.9% | - | 0.7% | 392 | 2.8 s |
| ARC-Easy | TextMAS sequential | 290/300 | 96.7% | 94.0-98.2% | b=6, c=11, p=0.332 | 2.0% | 3,834 | 31.3 s |

### GSM8K (first 20 test items)

| Condition | Items | Correct | Accuracy | 95% CI | Tokens (prompt + completion, all items) |
|---|---|---|---|---|---|
| baseline | 0-19 | 19/20 | 95.0% | 76.4-99.1% | 2,773 + 4,979 |
| TextMAS sequential | 0-19 | 19/20 | 95.0% | 76.4-99.1% | 62,160 + 52,531 |
| baseline (same items as OCC-3) | 0-13 | 13/14 | 92.9% | 68.5-98.7% | - |
| TextMAS sequential (same items) | 0-13 | 13/14 | 92.9% | 68.5-98.7% | - |
| OCC-3 sequential (adjudicated) | 0-13 | 13/14 | 92.9% | 68.5-98.7% | not recorded |

On items 0-13, all three conditions miss only item 12 (McNemar OCC-3 vs baseline:
b=0, c=0). The GSM8K API rows were produced with `bench_lib.run_baseline` and
`bench_lib.run_text_mas`. The driver command for these rows was not saved.

A later 18-item native OCC-3 v7 batch (items 2-19) completed 17 items with all
transport gates valid and a derived offline join of 16/17 correct; item 12 is
again the only miss. Item 19 was blocked before start. See
[GSM8K_OCC3_NATIVE_V7.md](GSM8K_OCC3_NATIVE_V7.md).

OCC-3 protocol labels on the 14 items in the historical `clean` file (that
filename does not certify isolated scoring or completed native use):

- Admission-reported hops: 38/42 (90.5%, descriptive Wilson 95% CI 77.9-96.2%).
  Hops are clustered within items, not independent observations.
- Items with all three hops reported admitted: 10/14, all adjudicated correct (10/10).
- Only 31 receipt-id entries accompany the 38 admitted labels. Seven are missing,
  and item 13 has a shortened planner-to-critic id. The published artifacts do not
  establish a completed-native-use rate; do not interpret this as either 38/42
  verified completions or proof of zero completions.
- Items with one text-fallback hop: 4 (items 0, 8, 10, 12), 3 correct.
- Wall time per item: mean 294 s, median 274 s, range 212-442 s. This includes the root
  session's fixed 150-180 s waits between checks, so it is not comparable with the API
  latencies above.

### GPQA-Diamond baseline (complete, 198 items)

| Segment | Indices | Correct | Accuracy | Length-limited |
|---|---:|---:|---:|---:|
| Historical baseline | 0-159 | 48/160 | 30.00% | 103/160 |
| PR #281 overlay continuation | 160-197 | 25/38 | 65.79% | 12/38 |
| Combined | 0-197 | **73/198** | **36.87%** | **115/198 (58.08%)** |

The remaining 38 items completed at 2026-11-10 14:01 UTC. The original 160 rows
were preserved and their joins/scoring rechecked offline. All capped responses
remain in the denominator; no 16k rerun or answer repair was performed.
`results/glm53_gpqa_baseline.jsonl` now contains all 198 rows: the preserved
160 historical rows plus the 38 continuation rows joined by index, with the
continuation correctness taken from the scored fragment (no re-derivation).
Segment provenance and aggregate metrics are in
`results/glm53_gpqa_baseline_summary.json`.

The continuation kept the upstream prompts, actual `glm-5.3` model, 4096-token
cap, temperature 0.6, top_p 0.95 and concurrency 16. It used a single HTTP attempt
per item instead of the historical retry loop. The literal Qwen identity in the
upstream prompt was retained for prompt equivalence; no Qwen model was run.
The two question segments and runtime versions differ, so their accuracy gap is
not evidence of an overlay quality improvement. Timing is segmented, not pooled.
See [GPQA_COMPLETION.md](GPQA_COMPLETION.md) for provenance, token accounting,
validation and the sanitized continuation artifact.

Historical-only extraction diagnostics for indices 0-159:
- 48/160 correct (30.0%, 95% CI 23.4-37.5%).
- 103/160 stopped at the 4096-token cap.
- 103/160 have no extracted answer; 102 of these were truncated.
- 57 have an extracted answer: 49 option letters (48 correct) and 8 numbers
  (all wrong). The unboxed numeric outputs used the extractor's last-number path.

This ordinary baseline completion does not prove native OCC-3 transport or
retirement behavior. Native acceptance and the new GSM8K run remain separate.

### GPQA-Diamond TextMAS sequential (complete, 198 items)

| Condition | Correct | Strict accuracy | Calls per item | Items with any length-limited role |
|---|---:|---:|---:|---:|
| TextMAS sequential | **103/198** | **52.02%** | 4 | **160/198 (80.81%)** |

All 792 role requests have confirmed result receipts. Failed, incomplete and
missing item counts are zero. Strict scoring preserves the upstream extractor;
there was no answer repair, retry or higher-cap rerun. The 160 length-limited
pipelines remain in the denominator. This counts pipelines with any capped role,
not individual capped calls or necessarily capped Judger calls.

The difference from the preserved 73/198 baseline is descriptive only. Four
4096-token calls per item are not budget matched to one baseline call, and the
baseline spans two runtime segments. No native OCC-3, causal quality,
token-efficiency, speedup or paired-significance claim follows from this result.
Provider totals are 633,590 prompt and 2,385,777 completion tokens. Missing cache
and nested reasoning details remain unknown. The 1511.858-second driver wall
interval is separate from overlapping role latency sums and is not GPU time.
See [GPQA_TEXTMAS_SEQUENTIAL.md](GPQA_TEXTMAS_SEQUENTIAL.md) for exact accounting,
provenance, limitations and the sanitized artifact.

### GPQA-Diamond TextMAS hierarchical (complete, 198 items)

| Condition | Correct | Strict accuracy | Calls per item | Items with any length-limited role |
|---|---:|---:|---:|---:|
| TextMAS hierarchical | **10/198** | **5.05%** | 4 | **139/198 (70.20%)** |

All 198 items and 792 role-result receipts are complete. Other item states and
receipt gaps are zero. The unchanged strict score includes capped pipelines;
there was no answer repair, alternate extraction or repeat model run.

A source-only differential review found no mismatch with the historical runner.
The first three GPQA role prompts each use the question and ignore prior context;
only the Judger receives the accumulated specialist content with the original
`context[:-1]` slice. These are upstream semantics, not a proven explanation for
the low score. No causal, significance, native OCC-3 or efficiency claim follows.
The 139 any-role length flags are not a Judger-only cap count.

Provider totals are 255,321 prompt and 2,255,471 completion tokens. Missing cache
and nested reasoning details remain unknown. Driver wall time is 1609.768 seconds,
separate from overlapping request-latency sums and not GPU time.
See [GPQA_TEXTMAS_HIERARCHICAL.md](GPQA_TEXTMAS_HIERARCHICAL.md) for the source
checks, accounting, comparison limits and sanitized artifact.

### MedQA300 baseline (complete, 300 items)

| Condition | Correct | Strict accuracy | Calls per item | Length-limited items |
|---|---:|---:|---:|---:|
| Single-agent baseline | **174/300** | **58.00%** | 1 | **27/300 (9.00%)** |

All 300 items and single-request result receipts are complete. Failed,
incomplete, missing and unconfirmed counts are zero. The historical baseline
retains raw normal response content and uses the exact upstream extractor and
normalizer against unchanged gold after sealed offline audit. No answer repair,
alternate extraction, cap change or rerun was performed. All capped outputs
remain in the denominator.

Provider totals are 97,628 prompt and 389,509 completion tokens. Missing cache
and nested reasoning details remain unknown. Driver wall time is 368.978 seconds,
separate from the overlapping request-latency sum and not isolated GPU time.
This is ordinary HTTP inference, not native OCC-3 or a clinical reliability claim.
The separate sequential TextMAS condition is complete below; hierarchical is not run.
See [MEDQA_BASELINE.md](MEDQA_BASELINE.md) for cohort/source bindings, accounting,
metric limits and sanitized metadata.

### MedQA300 TextMAS sequential (complete, 300 items)

| Condition | Correct | Strict accuracy | Calls per item | Any-role length-limited pipelines |
|---|---:|---:|---:|---:|
| Sequential TextMAS | **268/300** | **89.33%** | 4 | **67/300 (22.33%)** |

All 300 pipelines and 1,200 role-result receipts are complete. Failed, incomplete,
missing, unstarted and unconfirmed counts are zero. All capped pipelines remain
in the denominator; the length flag means any role, not necessarily the Judger.
The historical context slice and stripped Judger normal-content final are preserved.
No answer repair, alternate extraction, cap change or rerun was performed.

The baseline remains 174/300. The descriptive difference is 94 correct items or
31.33 percentage points on the same cohort, not a paired-win count, causal effect
or significance result. Four calls versus one are **not budget matched**.
Provider totals are 1,637,784 prompt and 1,925,873 completion tokens. Missing cache
and nested reasoning details remain unknown. Driver wall time is 1525.444 seconds,
separate from the overlapping request-latency sum and not isolated GPU time.
This is ordinary HTTP TextMAS, not native OCC-3 or a clinical reliability claim.
See [MEDQA_TEXTMAS_SEQUENTIAL.md](MEDQA_TEXTMAS_SEQUENTIAL.md) for source/cohort
bindings, strict scoring, accounting and sanitized evidence.

## Caveats

1. Token cap. 4096 completion tokens per call truncates most single-agent AIME and GPQA
   answers. TextMAS uses 3.5-4.1 times the completion tokens of the baseline, because it
   makes four capped calls. So the AIME differences between TextMAS and the baseline are
   confounded with the total token budget. A separately labelled rerun at a higher cap
   has not been run.
2. The OCC-3 condition differs from the API conditions in more than the channel:
   - its output budget is 131,072 tokens per call instead of 4096;
   - it uses Optimus provider sampling defaults;
   - its children have the Optimus system prompt and tools.

   This is not a channel-only comparison and cannot establish token efficiency,
   speedup, or equivalence to LatentMAS latent reasoning. On GSM8K all conditions
   are near the ceiling (13/14). Gold-answer isolation in the historical OCC-3
   run is unverified. Full-agent observation tools can expose parent transcripts
   and files; the later smoke reported access to a parent transcript containing
   scoring context. This does not prove historical answer leakage.
3. OCC-3 rows do not record token counts. Their wall time includes fixed waits.
4. Five OCC-3 items needed answer recovery, recorded in each row's `scoring_note`:
   - item 2 used a comma-aware parse of `$70,000` (the verbatim LatentMAS extractor reads
     only the digits before the comma);
   - item 3: the judger answered in an OCC/2-encoded message;
   - items 7, 8 and 9: the judger's last continuation echoed the format instruction, so the
     answer came from its earlier `FINAL_ANSWER` and `\boxed{}` reports.

   All five recovered answers match the gold answer. If these five items are scored as
   wrong, the result is 8/14. The OCC-3 team guidance in the same PR addresses this
   failure mode (plain-text final answers). The historical rows omit judger/role
   outputs, so the reported recoveries cannot be independently re-extracted from
   this publication. Treat 13/14 as adjudicated accuracy; 8/14 is a conservative
   no-recovery sensitivity score, not a fresh blinded run.
5. The historical prompts permit solving from the question alone after several
   continuations. The rows do not establish native-state-before-answer ordering.
   An admission or receipt id does not prove receiver completion; that requires
   a matched completed-native-use receipt. Even verified use does not prove
   that native state improved the answer. The current team recipe removes the
   question-alone escape, but has not been live-validated as a runtime fix.
6. Excluded rows:
   - `glm53_gsm8k_occ3.jsonl` (3 rows) ran under concurrent benchmark load and is marked
     `invalid_concurrent_load`.
   - Item 14 of the clean run was halted for the stack update and is marked
     `invalid_stack_update_halt`.
7. The AIME/GSM8K conditions use one sample on 14-30 items (API temperature
   0.6). The completed GPQA baseline has 198 items across two runtime segments. The small-set intervals are wide.
8. The GPQA authors ask that examples not be published in plain text. So
   `results/glm53_gpqa_baseline.jsonl` omits question text, options and model outputs.
   Per-item scoring fields remain. Numeric predictions are replaced by
   `"prediction_kind": "non_letter"`.

## NOT-RUN

- The `latent_mas` condition, because it needs model internals.
- ARC-Challenge and GSM8K-300 have no published result here. HumanEval+, MBPP+
  and ARC-Easy interim results are in the sections above; ARC-Easy hierarchical
  is in flight on a restarted runner. MedQA baseline and TextMAS sequential are
  complete above; its hierarchical condition is not run. Manifests are listed in
  `MANIFESTS.json`.
- OCC-3 items 14-19, and a completed rerun of items 0-13 on the PR #279 stack.
  The later item-0 smoke failed and remains separate from these denominators.
- AIME and GPQA at a higher token cap.
- Token accounting for OCC-3 rows.

## Files

| Path | Contents |
|---|---|
| `results/*.jsonl` | Per-item records; API rows include model outputs except GPQA; OCC-3 rows omit role/judger outputs |
| `results/*_summary.json*` | Per-condition summaries written by the runner |
| `results/glm53_gsm8k_occ3_clean.jsonl` | OCC-3 rows: hop states, receipt ids, native/total hops, wall time, scoring notes |
| `harness/runner.py`, `harness/bench_lib.py` | Resumable runner and OpenAI-compatible client adapter for the LatentMAS prompts |
| `harness/occ3_lib.py`, `harness/occ3_v3_prompts.py` | Historical OCC-3 helpers/prompts, retained unchanged; not a validated team recipe |
| `smoke-validation.md`, `server-smoke-excerpt.txt` | Post-update smoke failure and bounded server-log evidence |
| `results/glm53_gsm8k_occ3_v6.jsonl` | Original diagnostic-only item-0 smoke row; its `native_hops` field counts admission, not proven completion |
| `harness/sandbox-img/Dockerfile` | Code-task sandbox image (not used by any completed task yet) |
| `MANIFESTS.json` | Item manifest SHA-256 values, sources, counts and sampled indices |
| `GPQA_COMPLETION.md` | Completed GPQA baseline aggregate and segmented continuation provenance |
| `results/gpqa_baseline_160_197_safe.json` | Continuation index, truncation, usage and timing only; no answer content |
| `GPQA_TEXTMAS_SEQUENTIAL.md` | Completed 198-item HTTP TextMAS sequential result, strict scoring and unequal-budget limits |
| `results/gpqa_textmas_sequential_safe.json` | Sanitized sequential score, coverage, hashes, timing and usage; no answer content |
| `GPQA_TEXTMAS_HIERARCHICAL.md` | Completed 198-item HTTP TextMAS hierarchical result, source recheck and unresolved low-score diagnosis |
| `results/gpqa_textmas_hierarchical_safe.json` | Sanitized hierarchical score, coverage, hashes, timing and usage; no answer content |
| `MEDQA_BASELINE.md` | Completed 300-item single-agent baseline, strict historical scoring and accounting |
| `results/medqa_baseline_safe.json` | Sanitized MedQA baseline score, coverage, hashes, timing and usage; no answer content |
| `MEDQA_TEXTMAS_SEQUENTIAL.md` | Completed 300-item HTTP TextMAS sequential result, strict historical scoring and unequal-budget limits |
| `results/medqa_textmas_sequential_safe.json` | Sanitized MedQA sequential score, coverage, hashes, timing and usage; no answer content |
| `occ3-activation-notes.md` | Root-cause notes for OCC-3 activation on the DGX (settings keys and the per-chat `occ.json` preference) |

## Reproduce

The commands below describe the preserved historical harness. They do not
reproduce the new GPQA sequential/hierarchical driver
(`6b7eaa11be5394870b41cdd7dbc241bdd2c85ee711570738adaef7ad7847e1af`),
which uses question-only inputs, one-attempt receipts and a separate sealed
offline scorer. Its exact source and artifact pins are in
[GPQA_TEXTMAS_SEQUENTIAL.md](GPQA_TEXTMAS_SEQUENTIAL.md) and
[GPQA_TEXTMAS_HIERARCHICAL.md](GPQA_TEXTMAS_HIERARCHICAL.md). Do not use the
historical retry/resume driver to overwrite or supplement these frozen results.
The new MedQA baseline and sequential conditions likewise use a separate
source-pinned, question-only noncode driver with one-attempt receipts and sealed
offline scoring. Exact pins and differences are in [MEDQA_BASELINE.md](MEDQA_BASELINE.md)
and [MEDQA_TEXTMAS_SEQUENTIAL.md](MEDQA_TEXTMAS_SEQUENTIAL.md).

- Historical main conditions:
  1. Put the LatentMAS repo at `9a9e4d3` in `../LatentMAS`.
  2. Regenerate the manifests with the repo loaders and check them against
     `MANIFESTS.json`.
  3. Run `python harness/runner.py [task,...]`. The runner skips rows that are already
     recorded.
- The OCC-3 condition is run from an Optimus root session with the prompts in `harness/`.
  `occ3-activation-notes.md` lists historical activation observations, not a
  recommended agent-side settings-edit procedure. Use the current agent-message
  skill for setup and operator-owned configuration changes. A clean quality
  rerun also needs isolated scoring, fixed extraction, matched budgets/sampling,
  no alternate observation path to answers, per-role timestamps, token accounting,
  and receipt-correlated completion outcomes.
