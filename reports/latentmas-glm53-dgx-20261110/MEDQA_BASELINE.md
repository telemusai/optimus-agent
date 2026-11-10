# MedQA300 — single-agent baseline

## Result and scope

The frozen offline scorer reports **174 / 300 correct (58.00%)** for the
`baseline` condition. All 300 items completed. All 300 single-agent request
results have matching recorded start and terminal receipts.

| Measure | Count |
| --- | ---: |
| Fixed cohort / strict-score denominator | 300 |
| Completed items | 300 |
| Strict correct | 174 |
| Strict incorrect | 126 |
| Failed / incomplete / missing items | 0 / 0 / 0 |
| Request starts / observed results / confirmed results | 300 / 300 / 300 |
| Unconfirmed or missing request results | 0 |
| Unstarted request slots | 0 |
| Completed items with a provider `length` finish | 27 (9.00%) |

The 27 length-limited outputs remain in the denominator and are scored by the
same extractor. They are not removed, repaired, replaced, or rerun. A completed
request receipt does not imply that its answer is correct or that generation
finished without a length limit.

This report independently checks sanitized metadata, arithmetic, cohort hashes,
and frozen source contracts. It is **not an independent re-score of private
answers**. The private manifest, question projection, generated answers, raw
journals, and private terminal file were not opened for this report. The operator
ran the frozen offline audit and score step before supplying the safe summary.

- Task: `medqa`.
- Condition: `baseline`.
- Run ID: `5cd287db2b7d45498b1dda97bea87123`.
- Cohort: all indices 0–299 of the exact pinned MedQA300 manifest below.
- Score-summary timestamp: `2026-11-10T17:19:09.488407Z`.
- No supplementary score or answer-repair metric is reported.

## Prompt and execution contract

Each item uses the original `build_agent_messages_single_agent` function with
`task=medqa` and `method=baseline`. The task-specific multiple-choice prompt is
preserved. There is one `SingleAgent` / `singleagent` request per item and no
Planner, Critic, Refiner, or Judger pipeline in this condition.

The inherited system prompt literally identifies Qwen and Alibaba Cloud. This
is static prompt wording preserved for source comparability. The actual request
model field is **`glm-5.3`**, not Qwen. The source adapter's Qwen-named sentinel
satisfies an upstream prompt assertion; it does not select a different model.

| Request setting | Value |
| --- | --- |
| Model field | `glm-5.3` |
| Maximum generation tokens per request | 4096 |
| Temperature | 0.6 |
| Top-p | 0.95 |
| Generation seed sent | None |
| Requests per completed item | 1 |
| Maximum concurrent item requests | 16 |
| Item chunk size | Up to 16 |
| Attempt policy | One HTTP attempt per role; no automatic replay |

Requests use the ordinary configured loopback chat-completion endpoint. This is
**not a native Optimus child-agent or OCC3 condition**. The source hashes below
identify the preserved prompt and scoring sources. They do not establish an
official model-matched LatentMAS reproduction or a cryptographic runtime
attestation.

The replacement driver preserves prompt construction, generation parameters,
and historical final-content selection. It deliberately changes transport and
recording behavior: bounded I/O, durable per-request intent/results, private
artifacts, one attempt instead of the historical client's retries, and refusal
to replay uncertain dispatch. A failed item stops; already admitted bounded
work in its current chunk drains, and no next chunk starts. No failures were
reported in this run. No resume was requested.

The endpoint lock coordinates cooperating processes only. It is not a global
GPU reservation or proof that no other host process can issue inference.

## Exact scoring and gold separation

The generation process receives a closed, question-only projection with item
indices and question hashes. It has no manifest, gold-answer, solution, or
scorer input. The inference module loads only the hash-pinned prompt helpers
needed for this task. The scorer joins the pinned manifest offline, after the
operator's sealed-terminal and request/provenance audit.

Historical baseline final selection is the **entire normal response content**,
unchanged. It is not stripped by the generation pipeline. The scorer applies the
source-pinned `extract_gsm8k_answer`, despite that function's name, followed by
`normalize_answer`, exactly as the historical runner does for MedQA:

1. Use the last boxed match when present; inside it, use the first signed
   decimal match, or the stripped box contents if no decimal is present.
2. Otherwise, use the last signed decimal in the entire final text.
3. Normalize the extracted prediction with `strip().lower()` only; preserve
   a missing extraction as missing.
4. Compare that prediction to the **unchanged manifest gold value**, requiring
   both values to be nonempty. Do not normalize gold again.

There is no alternative letter extractor, reasoning-field fallback, earlier
message fallback, comma repair, answer coercion, or correctness-driven retry.
This metric can be sensitive to response formatting; that is part of the
preserved scoring contract, not permission to alter this result.

The frozen offline audit checks item/request identity, prompt reconstruction,
request and response hashes, recorded wire bytes and parsed response agreement,
role start/terminal pairing, source pins, and the bound invocation terminal
before scoring. This report checks the resulting safe summary and metadata; it
does not independently repeat those checks against private payloads.

## Usage and elapsed time

These are independent provider-reported counters from all 300 recorded role
results. Unknown nested fields remain unknown, not zero.

| Provider field | Sum of known values | Known requests | Unknown requests |
| --- | ---: | ---: | ---: |
| `prompt_tokens` | 97,628 | 300 | 0 |
| `completion_tokens` | 389,509 | 300 | 0 |
| `total_tokens` | 487,137 | 300 | 0 |
| Top-level `reasoning_tokens` | 290,729 | 300 | 0 |
| `prompt_tokens_details.cached_tokens` | Unknown | 0 | 300 |
| `completion_tokens_details.reasoning_tokens` | Unknown | 0 | 300 |

The prompt and completion sums add to the reported total. Reasoning and cache
fields are not added to that total, subtracted from it, or treated as extra
billable tokens. Missing nested reasoning data does not invalidate the distinct
reported top-level reasoning counter. No visible-token, cache-savings, cost, or
energy estimate is derived. The `singleagent` role totals equal the aggregate.

- Recorded driver-invocation wall time: **368.978 seconds**
  (unrounded: 368.9784489502199).
- Sum of 300 known request latencies: **2,495.787 seconds**
  (unrounded: 2495.7872684355825); unknown latency count: 0.
- Wrapper start: `2026-11-10T17:09:35.927662Z`.
- Wrapper finish: `2026-11-10T17:15:44.973414Z`.

Concurrent request latencies overlap. Their sum is not wall time, exclusive GPU
compute time, or service-level latency. Wrapper timestamps bound a different
scope from the driver's invocation measurement. These measurements exclude the
later offline score step. No cross-condition speed or token-efficiency claim is
made. The score summary's `budget_matched_to_baseline=true` only identifies this
baseline condition; it does not make a four-request TextMAS condition budget
matched.

## Audited bindings

The sanitized rows cover exactly 300 unique sorted indices. Their correctness
and length flags sum to 174 and 27. All rows contain one role record. Per-row
usage sums and known/unknown counts match the aggregate. The paired cohort hash
was independently rebuilt from the safe per-row question hashes and the pinned
manifest hash, without reading question text.

The safe final progress reports `completed` and matches the run ID, condition,
request totals, and terminal hash. The wrapper reports exit code 0 and
`SystemExit`, no error code, one driver invocation, and no resume. Its completion
receipt is only an invocation-status record, not model proof or a score.

The bound terminal is
`invocation-terminal-770d571df244478389df90a0d29c0999.json`.
Its file hash is supplied by the operator and corroborated by sanitized final
progress. The private terminal and private artifact hashes below are metadata
from that audit, not private files independently opened by this report.

### Run and implementation hashes

Canonical JSON object hashes are explicitly labeled. Other entries hash file
bytes. Source and artifact digests support content identity and consistency;
they are not signatures, cryptographic daemon authenticity, or protection
against a malicious same-user host rewriting all evidence together.

| Artifact | SHA-256 |
| --- | --- |
| Pinned MedQA300 manifest (operator-verified file bytes) | `8734b1d3e08b3a9525fd9b628476cb285857d46fb576e87be42f9b600817fde0` |
| Question-only projection (canonical JSON object) | `4f91d84cdc8b1352c797ff13895a2990cc8e54e892182b6cc9a4a7a0415fba6a` |
| Paired cohort (canonical JSON object) | `56ccd3e7884e0f98adebe30dc521a86fbe9e8f7396bd8d6ee7c1d78cdedfa79d` |
| Run provenance (canonical JSON object) | `03c307c6d0acdd112477ad8d24059184e9bddeb7c3d122164cf96b8869c77fe0` |
| Private items.jsonl (operator-audited file bytes) | `1ef240d78f2aa4c84877edd004f6d114c28ae512384ecb0cf61c6a07581a458c` |
| Private journal.jsonl (operator-audited file bytes) | `b93a8836ae1ab9843e52aecc333d65f3a156cb9851b637fae75a7afeccaa424e` |
| Private roles.jsonl (operator-audited file bytes) | `b80fa3b2b324e279eb361f3e624605843320b5c94258b6ce02c7296f55106bec` |
| Invocation terminal (operator-supplied file bytes) | `7e383153bb3491df29d14f28c0cf995c69fd6b57167f1eafdbd49346e92bb59c` |
| Sanitized score summary (locally verified file bytes) | `9ce9fa64ad97069509313742a23f4057aadbbb0d39b4bd9b206fec375c69fb1d` |
| Sanitized wrapper completion (locally verified file bytes) | `7594e7e150ec80cd2f3a81a094cb478dd9bbc4feb6c4a95ffe6611cec208365a` |
| Sanitized final progress (locally verified file bytes) | `f1d6bb5b75f028132d86f49afacbcf88ee3b211e380fd0357f28a64eeec7f02f` |
| Reviewed baseline launch spec (locally verified file bytes) | `c70a453c29ba5580783f0582bc93c49487df441a23b3ea7b7d3c1e7747b0a94b` |
| Frozen noncode driver | `f829c4660930a79506629f0e333d8cc42aac876576976b3695b0598643e6cb53` |
| Frozen noncode common module | `a8e426f0991bb147b6933906bc0644a343b722b8c773a9abf36dc70b391fffce` |
| Frozen offline scorer | `ef81ad75fddefbd09be5a07657f746865d0ceb5fbd8b35d397b4ffcee0d61f90` |
| Verified dependency bench_gpqa_textmas.py | `6b7eaa11be5394870b41cdd7dbc241bdd2c85ee711570738adaef7ad7847e1af` |
| Verified dependency bench_gpqa_textmas_common.py | `55dffad169036a22246f00bdec2dab8b4508377c1ab9f5e73949f27b2c572478` |

### Preserved historical source hashes

| Source | SHA-256 |
| --- | --- |
| `harness/bench_lib.py` | `d3612e197ef2d666d4fca9db8abb291c58ea0508da95f1ce065c40a703ae6545` |
| `harness/runner.py` | `a9ba8cb40415f552630b79ff297a5565c8676ebdb4c3964f887670a874f3f505` |
| `upstream/methods/__init__.py` | `eb00b86252532810d444b2449aefd0b005903e23b8fc8df54610ced26b26c6ae` |
| `upstream/methods/text_mas.py` | `fecbea12eb74bdb3e59177fb06384702408b4d13c39a782165b1c9fb74ddf860` |
| `upstream/prompts.py` | `74e7e940f1cf21fcc21ed44fe75ef53741408b5ae95ba3b7d00f88d25ca55ff0` |
| `upstream/utils.py` | `36a905981170d464e479b2f0b0c6efda874abc84788abc7ff13aa4e4a9560721` |

## Limits and privacy

This is one sampled run on one fixed 300-item cohort. No repeated-run variance,
confidence interval, statistical significance, causal benefit, or clinical
reliability is claimed. Other conditions and other task cohorts must remain
separate. No result is overwritten or selectively rerun based on correctness.

This report contains no item questions, options, model answers, extracted
predictions, gold values, reasoning, raw request bodies, credentials, or private
transcripts. It does not infer an unverified model image or service revision
from older reports. Publication and any later live run remain separate operator
actions.

Sanitized per-item metadata and aggregates: [results/medqa_baseline_safe.json](results/medqa_baseline_safe.json).
