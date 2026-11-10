# GPQA TextMAS hierarchical completion

## Strict result

The pinned GPQA hierarchical run completed all **198 items** and received
**792 confirmed role-result receipts**. The unchanged historical scorer reports
**10/198 correct (5.05%)**. This low score is preserved without repair or rerun.
All 198 items remain in the denominator, including **139 pipelines (70.20%)**
with at least one role ending at the length limit.

| Condition | Correct / total | Strict accuracy | Planned calls per item | Items with at least one length-limited call |
|---|---:|---:|---:|---:|
| Preserved combined baseline | 73 / 198 | 36.87% | 1 | 115 / 198 (58.08%) |
| Preserved TextMAS sequential | 103 / 198 | 52.02% | 4 | 160 / 198 (80.81%) |
| TextMAS hierarchical | **10 / 198** | **5.05%** | **4** | **139 / 198 (70.20%)** |

These are descriptive results, not causal or significance claims. Four capped
calls are not budget-matched to a single baseline call. The baseline spans
historical and continuation runtime segments; its latency is not pooled here.
Sequential and hierarchical share the pinned cohort and per-call parameters,
but differ in prompts/context routing. One stochastic sample per item does not
establish why their scores differ. All existing results remain unchanged.

The safe inventory reports no failed, incomplete, missing or unstarted items;
792 role starts, observed results and confirmed results agree, with no receipt
gaps. This report checks sanitized metadata and frozen source contracts. It
**does not independently rescore private answers** or infer that an unsuccessful
answer was caused by a transport defect, prompt wording, truncation or extraction.

## Source-contract recheck

The unexpectedly low aggregate prompted a source-only recheck, not a new model
run. The actual frozen driver/common/scorer and historical source bytes matched
the hashes below. The hierarchical behavior matches
`remote-harness/runner.py::run_item` and the pinned upstream builder:

1. All four requests execute serially per item, in Planner, Critic, Refiner,
   Judger order. At most 16 item pipelines/HTTP calls are in flight.
2. In the GPQA branch of `build_agent_messages_hierarchical_text_mas`
   (`prompts.py:557–595`), the first three prompts assign math, science and code
   roles. Each includes the question but **ignores the supplied prior context**.
   Serial execution does not mean those first three prompts consume one
   another's outputs. This is the historical builder's behavior.
3. Each response's normal content is stripped. The first three contents are
   accumulated using the literal display labels `Math Agent`, `Science Agent`
   and `Code Agent`. The preserved map also contains the spelling
   `Task Summrizer`; the Judger's output is not appended to prior context.
4. The Judger prompt receives the question plus the accumulated context sliced
   by `context[:args.text_mas_context_length]`, with the historical default `-1`.
   The last character is therefore dropped. This was not silently corrected.
5. Only the last Judger's normal `content.strip()` becomes `final_text`.
   Reasoning text, earlier specialist outputs and alternate messages are not
   answer fallbacks.
6. The scorer uses the exact historical non-code branch:
   `normalize_answer(extract_gsm8k_answer(final_text))`, compared with untouched
   manifest gold when both are truthy. No new letter parser, comma repair,
   numeric conversion, gold normalization or supplemental extraction is used.

The original 24 network-forbidden GPQA regression tests passed again. An
additional synthetic differential check executed the original runner's scoped
AST with a fake client and compared it against the frozen hierarchical driver:
all four exact prompt payloads, role order/display tracing, final content,
truncation and strict scoring matched. It also checked that the first three
prompts ignore context and that the Judger retains the minus1 slice. No original
manifest or actual model output was accessed by these checks.

**No source-contract discrepancy was found in that review.** This is not proof
that any particular real answer was formatted correctly, nor an explanation of
the low score. The safe summary lacks final-answer content, extractor failure
categories and the Judger-only finish-reason distribution. Those causes remain
unresolved by this report. The 139 any-role flags are **not** a count of capped
Judger calls or individual capped requests.

## Run, isolation and scoring policy

- Actual model: `glm-5.3`; condition: `text_mas_hierarchical`.
- Parameters per request: `max_tokens=4096`, `temperature=0.6`, `top_p=0.95`.
  No generation seed was sent. There was one sample per item.
- Upstream revision: `9a9e4d331eb11430bd9e64754c6b252b06d73031`.
  Historical harness label: `e9067141a`.
- The literal system identity remains
  `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
  It was preserved for prompt equivalence; no Qwen model was used.
- Generation used a closed question-only projection without gold or solutions.
  The inference path did not import the original manifest reader or scorer.
- After settlement, the offline scorer validated terminal/artifact seals,
  provenance, source/code pins, question hashes, intent/result bindings and
  source-rebuilt payload/context/final content before the private manifest join.
- One HTTP attempt per role was permitted. The historical automatic retry loop
  was not used. This deliberate safety-policy difference does not alter the
  successful-request prompt/context/scoring contract.
- Durable pre-dispatch records, bounded IO, private journals, disabled redirects
  and proxy-environment use, and refusal to replay uncertain/pending work were
  retained. On failure an item would stop, admitted pipelines would drain, and
  no next chunk would start. No such failure is reported in this run.
- All capped outputs remain in the strict denominator. No answer repair,
  earlier-role substitution, larger per-call cap, score-dependent retry or
  repeat model run was performed for this result.

This is ordinary HTTP TextMAS, **not native OCC-3** and not an in-process
`latent_mas` run. It does not prove native handoff completion or retirement
behavior. Using the upstream prompts on GLM-5.3 is not an official model-matched
LatentMAS reproduction. Source hashes do not establish the current server image,
runtime generation or exact hardware state; none is inferred from an older
README setup table. A cooperating endpoint lock is not a global GPU reservation.

## Observed timing and provider counters

| Timing metric | Recorded value |
|---|---:|
| Driver invocation wall interval | **1609.768 s** |
| Sum of 792 role-request latency intervals | **17494.444 s** |
| Unknown role latency records | 0 |

The driver's wall interval covers one invocation after its initial preparation.
The summed request intervals overlap under concurrency and include waiting.
They are neither elapsed wall time nor isolated GPU execution time. No speedup,
throughput attribution or token-efficiency conclusion follows from these values.

The supplied wrapper completion reports start `2026-11-10T16:38:32.594320Z`,
finish `2026-11-10T17:05:22.455538Z`, and driver exit0. Its interval includes a
different setup/publication scope; it does not replace the driver timer. The
wrapper completion signals invocation termination, not correctness or native use.

| Independent provider field | Reported sum | Known role records | Unknown role records |
|---|---:|---:|---:|
| `prompt_tokens` | 255,321 | 792 | 0 |
| `completion_tokens` | 2,255,471 | 792 | 0 |
| `total_tokens` | 2,510,792 | 792 | 0 |
| `reasoning_tokens` | 2,187,265 | 792 | 0 |
| `prompt_tokens_details.cached_tokens` | unavailable | 0 | 792 |
| `completion_tokens_details.reasoning_tokens` | unavailable | 0 | 792 |

Provider total equals prompt plus completion tokens in the aggregate. Top-level
reasoning tokens remain a separate reported counter, not an additional amount
to add to total or completion tokens. The absent nested reasoning and cache
fields remain **unknown, not zero**. No billing, uncached-token or visible-output
token count is estimated from those missing fields.

| Role | Confirmed records | Prompt tokens | Completion tokens | Provider total tokens | Separate top-level reasoning tokens |
|---|---:|---:|---:|---:|---:|
| Planner | 198 | 52,486 | 596,796 | 649,282 | 581,794 |
| Critic | 198 | 52,521 | 606,155 | 658,676 | 591,753 |
| Refiner | 198 | 52,684 | 592,950 | 645,634 | 580,921 |
| Judger | 198 | 97,630 | 459,570 | 557,200 | 432,797 |

## Exact identity references

Run ID: `a13838efc22342f497fd206dde397ae3`.
Safe score schema: `gpqa.textmas.score_summary.v1`.
Scored-summary timestamp: `2026-11-10T17:06:41.892679Z`.
Terminal name: `invocation-terminal-956da3aa59e3412cae3664a28285aa8f.json`.
The terminal digest supplied by the operator agrees with the final safe progress
record. The terminal/private journals themselves were not opened for this report.

| Artifact or identity | SHA-256 |
|---|---|
| GPQA manifest, file bytes | `4f6347e51a6d0489b5bc6e8ef2e1e32c7424ecf64224a2fd707d0826df25b0ad` |
| Question projection, canonical JSON object | `99ea793070730aa9068d28158424bf558cbc0cfcd871c4946823ab9630c0eadf` |
| Paired cohort, canonical JSON object | `0c7beb1c4e10c50f70b1a5881180dfba36c8d9a6a431f143979acb6261490fe8` |
| Provenance, canonical JSON object | `c5e003f48fadaf52429e69cf3631f50e148ff00a73dc68114642baec43a7444d` |
| Invocation terminal, operator-supplied file-byte digest | `78383ff08901a95c1918c449b3e50a4e8bd81af11ebf73861fb7acf1324e7902` |
| Private `items.jsonl`, file bytes | `1b3489cc1c26832fa5f3d2b5b65311edc64af6e7338ab5f1b42bf72ec3d11053` |
| Private `journal.jsonl`, file bytes | `d8c109376d937cb146887816e3f214dcac55bd0e2fb27ef1da7a8fee0b9069bd` |
| Private `roles.jsonl`, file bytes | `41cee5568a97a85818cb3e7f48632c156104fad4d948dcfc99982c83befea03e` |
| `bench_gpqa_textmas.py`, generation driver | `6b7eaa11be5394870b41cdd7dbc241bdd2c85ee711570738adaef7ad7847e1af` |
| `bench_gpqa_textmas_common.py`, shared contracts | `55dffad169036a22246f00bdec2dab8b4508377c1ab9f5e73949f27b2c572478` |
| `bench_gpqa_textmas_offline.py`, projection/scorer | `777d9c5b12215afaa1ea0253dad109c89647ebc107d709fd3f4ef39ba5e76c8f` |
| `gpqa-textmas-hier-score-safe.json`, sanitized summary bytes | `286bae2a902b9bff12560c84b15783c102fd165a9821bf935df09e460b99f280` |
| `gpqa-textmas-hier-completion.json`, supplied safe completion bytes | `646bebeb76eea42b20072dbd94dea136350d3dc4e09c24a3af4107861fa12c0b` |
| `gpqa-textmas-hier-progress-final-safe.json`, sanitized progress bytes | `82dc823750bac989762c0abc05764fbe182e8c06f898650bfd93a9f86dcf89e6` |

Projection, cohort and provenance digests are canonical JSON object hashes, not
file-byte hashes. The encodings are not interchangeable. The safe completion and
progress digests identify the supplied local files; no claim about uninspected
remote serialization is inferred. Private-file digests are reconciliation
references from the safe summary, not copied private artifacts.

| Pinned historical source | SHA-256 |
|---|---|
| `harness/bench_lib.py` | `d3612e197ef2d666d4fca9db8abb291c58ea0508da95f1ce065c40a703ae6545` |
| `harness/runner.py` | `a9ba8cb40415f552630b79ff297a5565c8676ebdb4c3964f887670a874f3f505` |
| `upstream/methods/__init__.py` | `eb00b86252532810d444b2449aefd0b005903e23b8fc8df54610ced26b26c6ae` |
| `upstream/methods/text_mas.py` | `fecbea12eb74bdb3e59177fb06384702408b4d13c39a782165b1c9fb74ddf860` |
| `upstream/prompts.py` | `74e7e940f1cf21fcc21ed44fe75ef53741408b5ae95ba3b7d00f88d25ca55ff0` |
| `upstream/utils.py` | `36a905981170d464e479b2f0b0c6efda874abc84788abc7ff13aa4e4a9560721` |

These hashes are identity/integrity references, **not signatures, cryptographic
daemon authentication or independent runtime attestations**. They do not certify
GPU exclusivity, answer quality or native transport use.

## Sanitized review and unresolved diagnosis

- Exact sorted index coverage 0–197 and 198 completed four-role rows: checked.
- Safe-row correctness sum10 and any-role-length sum139 match the aggregate.
- All 792 starts/results/confirmed receipts agree; other states and gaps are zero.
- Per-row and per-role provider counters and known/unknown counts agree with
  the aggregate. Driver/common hashes agree with the wrapper completion.
- Final safe progress reports the same condition/run, 198 admitted items,
  completed status, and the operator-supplied terminal name/hash.
- No questions, answer options, gold, predictions, raw responses, reasoning text
  or individual answer examples are included here. Only sanitized aggregates,
  hashes and source semantics were used. No model, remote, network, Git or
  private-artifact operation was used during report preparation.

If the operator separately authorizes deeper diagnosis, a bounded private audit
could emit aggregate-only final-empty/nonempty counts, boxed-match counts,
exact-extractor missing/nonmissing counts and output-type categories, plus
Judger finish-reason cross-tabs. It must not expose answer values or substitute a
new extractor. That diagnostic is **not performed or implied by this report**.
No proposed explanation is accepted merely because the score is low. The strict
10/198 result remains unchanged.

Sanitized per-item metadata and aggregates: [results/gpqa_textmas_hierarchical_safe.json](results/gpqa_textmas_hierarchical_safe.json).
