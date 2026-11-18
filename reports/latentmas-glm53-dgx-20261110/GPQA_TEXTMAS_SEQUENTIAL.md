# GPQA TextMAS sequential completion

## Result

GLM-5.3 TextMAS sequential completed all **198 items** in the pinned GPQA
manifest. The strict offline score is **103/198 (52.02%)**. All 792 role requests
have confirmed result receipts in the supplied audit summary. There are no
failed, incomplete, missing or unstarted items, and no missing role receipts.
The [sanitized score artifact](results/gpqa_textmas_sequential_safe.json) contains
per-index status, correctness, truncation, hashes, timing and usage, but no answer
content.

| Condition | Correct / total | Strict accuracy | Planned calls per item | Items with at least one length-limited call |
|---|---:|---:|---:|---:|
| Preserved combined baseline | 73 / 198 | 36.87% | 1 | 115 / 198 (58.08%) |
| TextMAS sequential | **103 / 198** | **52.02%** | **4** | **160 / 198 (80.81%)** |

The descriptive difference is 30 more correct items, or 15.15 percentage points.
This is **not a budget-matched or causal comparison**. TextMAS has four serial
4096-token calls per item, versus one planned 4096-token baseline call. The
baseline also spans historical and continuation runtime segments; it is not a
single fresh-stack control run. The baseline result and historical rows remain
unchanged. See [GPQA baseline completion](GPQA_COMPLETION.md) for that segmented
provenance. No paired significance test, token-efficiency gain or speedup is
claimed here.

This report uses only the hash-pinned sanitized score summary, a sanitized
completion probe, and published reference documentation. Preparation checked
coverage, arithmetic, usage sums and corresponding source hashes. It did **not**
read private questions, options, gold, model responses or raw result files, and
did not independently re-extract or rescore answers.

## Condition and execution contract

- Actual API model: `glm-5.3`. Condition: `text_mas_sequential`.
- One sample per item. Every call used `max_tokens=4096`, `temperature=0.6`, and
  `top_p=0.95`. No generation seed was sent.
- Roles: Planner, Critic, Refiner, Judger. Each item made four serial ordinary
  chat-completion requests. At most 16 item pipelines/HTTP calls were in flight.
- Text context, not native state, passed between roles. The historical pipeline
  strips each role's content, accumulates the non-Judger outputs, and retains
  the upstream default `context[:text_mas_context_length]` with value `-1`.
  That slice drops the last character; it was preserved, not corrected.
- Prompt source: LatentMAS revision
  `9a9e4d331eb11430bd9e64754c6b252b06d73031`. Reviewed harness label: `e9067141a`.
  The literal system identity was retained for prompt equivalence:
  `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
  **No Qwen model was used.**
- The operator identifies the execution context as the PR #281 overlay at
  `0209c1d`. No exact server image, binary identity or runtime generation is
  asserted from these safe inputs. The historical setup table in README is
  not treated as the current runtime's identity.
- The driver used **one HTTP attempt per role**, without the historical client's
  automatic retry loop. Exactly 792 starts and 792 confirmed role results were
  reported, with 198 results for each role.
- The safety transport changes were explicit: durable pre-dispatch intents,
  bounded requests/responses, private result journals, source/hash checks,
  disabled redirects/proxy-environment use, and refusal to replay pending or
  uncertain work. A failed item would stop its own pipeline; admitted pipelines
  would drain before stopping admission of another chunk. None failed in this
  run. These controls differ from the historical recovery policy, while the
  successful-request prompts, context propagation and scoring remain equivalent.
- The local endpoint lock excludes cooperating drivers only. Operator no-overlap
  controls are not an attested global GPU reservation.

This is an ordinary HTTP TextMAS result. It is **not a native OCC-3 benchmark**,
not evidence of completed native handoffs or source-retirement safety, and not an
in-process `latent_mas` run. Using upstream prompts with GLM-5.3 does not make this
an official model-matched LatentMAS reproduction.

## Gold separation and strict scoring

Generation received a closed question-only projection: item index, exact
question string and its hash, plus cohort identity. Gold and solution fields
were excluded. The generation path did not import the manifest reader or scorer.
It compiled only source-pinned prompt/role definitions; earlier reasoning text
was not substituted for normal content when constructing role context.

After the driver settled, the offline scorer validated the explicit terminal
seal, provenance, source/code pins, question hashes, intent/result/terminal
bindings and source-rebuilt payload/context sequence **before** reading the
original manifest for the gold join. Scoring used the exact upstream
`normalize_answer(extract_gsm8k_answer(final_text))` on the historical stripped
Judger content, compared with the untouched manifest gold.

No format repair, alternate extractor, earlier-role answer fallback,
supplemental score, gold-dependent retry or larger-budget rerun was applied.
All 160 length-limited pipelines remain in the 198-item denominator. They were
not discarded or rerun at a 16k per-call cap. A pipeline's length-limited flag
means **at least one** of its four calls ended `length`; 160 is not the count of
individual capped calls. That flag alone does not imply an incorrect answer.

The supplied safe summary supports the final counts. This report does not turn
those metadata checks into an independent review of private answer content.

## Observed time and provider usage

| Timing metric | Recorded value |
|---|---:|
| Driver invocation wall interval | **1511.858 s** |
| Sum of 792 role-request latency intervals | **17622.182 s** |
| Role latency records missing | 0 |

The wall interval is one recorded driver invocation, not pooled historical and
continuation timing. It excludes earlier wrapper/source setup and run preparation.
The summed role latencies overlap under concurrency and include client/server
waiting; they are **not elapsed wall time or GPU execution time**. Neither value
establishes isolated device latency or a speedup over the segmented baseline.

The supplied wrapper probe reports start `2026-11-10T15:54:03.405035Z`, finish
`2026-11-10T16:19:15.328878Z`, and driver exit0. Its interval has a different scope
from the driver's invocation timer and is not substituted for it. The wrapper
receipt reports an invocation ending; it is not itself a scoring or native-use
receipt.

| Independent provider field | Sum reported | Known role records | Unknown role records |
|---|---:|---:|---:|
| `prompt_tokens` | 633,590 | 792 | 0 |
| `completion_tokens` | 2,385,777 | 792 | 0 |
| `total_tokens` | 3,019,367 | 792 | 0 |
| `reasoning_tokens` | 2,155,734 | 792 | 0 |
| `prompt_tokens_details.cached_tokens` | unavailable | 0 | 792 |
| `completion_tokens_details.reasoning_tokens` | unavailable | 0 | 792 |

The aggregate total equals prompt plus completion tokens. Provider-reported
reasoning is a separate counter and is **not added again** to total or completion
tokens. The top-level reasoning counter is known; the nested reasoning-detail
field is not. Cache attribution and absent detail counters are **unknown, not
zero**. No uncached-token estimate, provider-billing estimate or inferred visible
output-token total is made.

| Role | Confirmed records | Prompt tokens | Completion tokens | Provider total tokens | Separate top-level reasoning tokens |
|---|---:|---:|---:|---:|---:|
| Planner | 198 | 63,394 | 509,974 | 573,368 | 452,596 |
| Critic | 198 | 126,574 | 769,372 | 895,946 | 710,950 |
| Refiner | 198 | 188,188 | 608,123 | 796,311 | 545,622 |
| Judger | 198 | 255,434 | 498,308 | 753,742 | 446,566 |

## Artifact and source identity

Run ID: `0a15d23df01e4502842964d8e69254a0`.
The sanitized summary uses schema `gpqa.textmas.score_summary.v1` and was created
at `2026-11-10T16:24:08.205373Z`.

| Artifact or identity | SHA-256 |
|---|---|
| GPQA manifest, file bytes | `4f6347e51a6d0489b5bc6e8ef2e1e32c7424ecf64224a2fd707d0826df25b0ad` |
| Question projection, canonical JSON object | `0cc6d5074b5834975e4e5131502114d5bef0eea6a35c414effbfc17e48cdc084` |
| Paired cohort identity, canonical JSON object | `0c7beb1c4e10c50f70b1a5881180dfba36c8d9a6a431f143979acb6261490fe8` |
| Provenance, canonical JSON object | `bf69319d5ba1c384720ff471c24efec3832b75545e8fce64ec9b0f2f8221399d` |
| Invocation terminal, operator-supplied file-byte digest | `e766d2a85e695c55c8626fa2d87eba47fdc9265baacb2a92f84ce140e268c0c8` |
| Private `journal.jsonl`, file bytes | `fa31cecdf0e36949ec28b1bb148540d5482563331e5b797e28bd9dedbd7f1f4a` |
| Private `roles.jsonl`, file bytes | `70ad2bc48271cfa358050449bdc1830e4b4b838624ca86f6a58c41a416a59acc` |
| Private `items.jsonl`, file bytes | `0129a55052c13ba271c6e266fbeb0fc2392d9a029534dfa5244b0c712feaad15` |
| `bench_gpqa_textmas.py`, generation driver | `6b7eaa11be5394870b41cdd7dbc241bdd2c85ee711570738adaef7ad7847e1af` |
| `bench_gpqa_textmas_common.py`, shared contracts | `55dffad169036a22246f00bdec2dab8b4508377c1ab9f5e73949f27b2c572478` |
| `bench_gpqa_textmas_offline.py`, projection/scorer | `777d9c5b12215afaa1ea0253dad109c89647ebc107d709fd3f4ef39ba5e76c8f` |
| `gpqa-textmas-seq-score-safe.json`, supplied sanitized file | `9fabd71f20813f32b360fe05153b71ef31fadc29dd88fe9993bceba03e1dd1e6` |
| `gpqa-textmas-seq-completion-probe.json`, local sanitized probe bytes | `a8ba44be1c9e82990aaaf227a3d1112d85ef3eb430ec1209ae6594be713845fb` |

The projection, paired-cohort and provenance hashes are canonical-object hashes,
not file-byte hashes. These encodings are not interchangeable. Private result
files and the terminal were not opened during report preparation; their digests
are retained for operator reconciliation from the safe summary and supplied
terminal identity. The sanitized probe's hash identifies the local probe file,
not the original remote wrapper-completion file bytes.

| Pinned historical source | SHA-256 |
|---|---|
| `harness/bench_lib.py` | `d3612e197ef2d666d4fca9db8abb291c58ea0508da95f1ce065c40a703ae6545` |
| `harness/runner.py` | `a9ba8cb40415f552630b79ff297a5565c8676ebdb4c3964f887670a874f3f505` |
| `upstream/methods/__init__.py` | `eb00b86252532810d444b2449aefd0b005903e23b8fc8df54610ced26b26c6ae` |
| `upstream/methods/text_mas.py` | `fecbea12eb74bdb3e59177fb06384702408b4d13c39a782165b1c9fb74ddf860` |
| `upstream/prompts.py` | `74e7e940f1cf21fcc21ed44fe75ef53741408b5ae95ba3b7d00f88d25ca55ff0` |
| `upstream/utils.py` | `36a905981170d464e479b2f0b0c6efda874abc84788abc7ff13aa4e4a9560721` |

Hashes identify the supplied source and artifacts. They are **not signatures,
cryptographic daemon authentication or independent runtime attestation**. They
do not prove GPU exclusivity or native transport use. No private artifact is
included in this report.

## Local sanitized review and limits

- Exact coverage 0–197, no duplicate indices: checked.
- All 198 safe rows completed, with four role records each: checked.
- Safe-row correctness sum 103 and length-limited sum 160 match the aggregate.
- 792 attempted starts, observed results and confirmed results agree. Failed,
  incomplete, missing, unstarted and missing-receipt counts are all zero.
- Per-row and per-role usage aggregates agree with the provider-field totals;
  unknown counts remain explicit.
- Driver/common hashes agree between the safe score summary and completion probe.
- The report was prepared without Git, network, remote commands, live inference,
  private dataset reads or raw-output access. Existing source/results were not
  edited. Publication is a separate operator action.

This adds only the sequential condition. It does not revise the preserved
baseline, publish answer content, establish the current status of another
condition, or support a claim that a specific runtime change caused the accuracy
difference. One stochastic sample per item, unequal role/token budgets and
segmented baseline provenance remain material limitations.
