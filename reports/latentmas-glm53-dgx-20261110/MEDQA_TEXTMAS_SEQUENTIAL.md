# MedQA300 — sequential TextMAS

## Strict result

The frozen offline scorer reports **268 / 300 correct (89.33%)** for
`text_mas_sequential`. All 300 item pipelines completed, with **1,200 confirmed
role results**. There are no failed, incomplete, missing, unconfirmed, or
unstarted entries in the supplied safe summary.

| Measure | Count |
| --- | ---: |
| Fixed cohort / strict-score denominator | 300 |
| Completed item pipelines | 300 |
| Strict correct / incorrect | 268 / 32 |
| Failed / incomplete / missing items | 0 / 0 / 0 |
| Role starts / observed results / confirmed results | 1,200 / 1,200 / 1,200 |
| Unconfirmed or missing role results | 0 |
| Unstarted role slots | 0 |
| Pipelines with at least one provider `length` finish | 67 (22.33%) |

The 67 length flags count **items with any length-limited role**, not
length-limited Judger calls or a count of individual truncated requests. All
300 completed pipelines remain in the strict denominator. Length-limited
outputs are scored without repair, exclusion, a larger cap, or rerun.

- Task: `medqa`.
- Condition: `text_mas_sequential`.
- Run ID: `8c103b757c094001bd1bc9f9d89bcd4e`.
- Safe score timestamp: `2026-11-10T17:51:15.834646Z`.
- Cohort: all indices 0–299 of the exact pinned MedQA300 manifest below.
- No supplementary score or alternative extraction metric is reported.

### Descriptive baseline comparison

| Condition | Correct | Strict accuracy | Requests per item | Confirmed requests |
| --- | ---: | ---: | ---: | ---: |
| Preserved single-agent baseline | 174 / 300 | 58.00% | 1 | 300 |
| Sequential TextMAS | 268 / 300 | 89.33% | 4 | 1,200 |

The aggregate difference is **94 more correct, or 31.33 percentage points**.
The question hashes and paired-cohort hash agree across the two supplied safe
summaries. This is a descriptive comparison, not a causal estimate, paired-win
count, significance test, or repeated-run result. Four requests versus one are
**not budget matched**. The baseline result is preserved, not replaced.

## Prompt and generation contract

The role order is the original `default_agents()` sequence:
**Planner → Critic → Refiner → Judger**. Each pipeline makes four serial HTTP
requests. Up to 16 item pipelines run concurrently, with no more than 16 role
requests in flight. Each admitted chunk has at most 16 items.

The frozen driver uses the source-pinned
`build_agent_messages_sequential_text_mas` with `task=medqa`. Each non-Judger
normal response is stripped and appended to cumulative context with its
original role-name header and two trailing newlines. The upstream default
`text_mas_context_length=-1` is preserved: the prompt builder uses
`context[:args.text_mas_context_length]`, which is **`context[:-1]`**. It removes
the last character of nonempty context; it is not treated as an unlimited-context
sentinel or silently corrected during this run.

The final scored text is **only the last Judger normal content after
`strip()`**, exactly as in the historical runner. The separate reasoning field,
earlier role answers, and earlier messages are not fallback answers.

The inherited system prompt literally identifies Qwen and Alibaba Cloud. This
static wording is retained for source comparability. The actual request model
field is **`glm-5.3`**. The adapter's Qwen-containing model-name sentinel only
satisfies a prompt-builder assertion; it is not model selection.

| Request setting | Value |
| --- | --- |
| Model field | `glm-5.3` |
| Maximum generation tokens per role request | 4096 |
| Temperature | 0.6 |
| Top-p | 0.95 |
| Generation seed sent | None |
| Role requests per completed item | 4, serial |
| Maximum concurrent item pipelines / role requests | 16 / 16 |
| Request attempt policy | One attempt; no automatic replay |

These are ordinary loopback HTTP chat-completion calls. This is **not a native
Optimus/OCC3 handoff run** and is not an official model-matched LatentMAS
reproduction. It does not establish clinical reliability.

## Gold separation, audit, and scoring

Generation consumes a closed question-only projection with item indices and
question hashes, not the original manifest, gold values, solutions, or scorer.
The prompt helpers are loaded from pinned source. The offline scorer joins gold
only after the operator's sealed-terminal and request/provenance audit. No score
feedback is used for generation, retry, or item selection.

The noncode scorer preserves the historical runner's use of
`extract_gsm8k_answer` for this task, despite that helper's name. It selects the
last boxed match when present, then the first signed decimal within the box or
the stripped box contents if no decimal exists. Otherwise it takes the last
signed decimal in the final text. `normalize_answer` applies `strip().lower()`
to the extracted prediction and preserves a missing extraction. The nonempty
prediction is compared to the **unchanged manifest gold**. Gold is not normalized
again. There is no alternative letter parser, comma repair, answer coercion, or
format repair. Formatting sensitivity remains part of this strict metric.

The safety wrapper and driver change transport and recording behavior, not the
preserved prompt/context/final-selection rules. They add bounded I/O, private
durable intents and results, one attempt instead of the historical client's
retries, and explicit refusal to replay uncertain work. A failed item stops;
already admitted pipelines in its chunk drain, and no next chunk starts. No
failures or resume requests were reported here. The shared endpoint lock covers
cooperating processes only; it is not a global GPU reservation.

The operator's frozen offline audit binds source and projection hashes, rebuilt
request prompts, request/response hashes, recorded wire bytes and parsed response
agreement, ordered role start/result/terminal records, item finals, and the
invocation terminal before scoring. **This report is a metadata and source
review, not an independent re-extraction of private answers.** It does not open
the original manifest, projection, raw journals, generated text, or private
terminal file.

## Token accounting

All counters below are independent provider-reported fields from the 1,200
recorded role results. Missing nested fields stay unknown, not zero.

| Provider field | Sum of known values | Known requests | Unknown requests |
| --- | ---: | ---: | ---: |
| `prompt_tokens` | 1,637,784 | 1,200 | 0 |
| `completion_tokens` | 1,925,873 | 1,200 | 0 |
| `total_tokens` | 3,563,657 | 1,200 | 0 |
| Top-level `reasoning_tokens` | 1,156,910 | 1,200 | 0 |
| `prompt_tokens_details.cached_tokens` | Unknown | 0 | 1,200 |
| `completion_tokens_details.reasoning_tokens` | Unknown | 0 | 1,200 |

| Role | Results | Prompt tokens | Completion tokens | Total tokens | Top-level reasoning tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Planner | 300 | 98,228 | 329,869 | 428,097 | 198,290 |
| Critic | 300 | 238,413 | 897,900 | 1,136,313 | 594,819 |
| Refiner | 300 | 545,853 | 395,010 | 940,863 | 192,665 |
| Judger | 300 | 755,290 | 303,094 | 1,058,384 | 171,136 |

Each role has 300 known top-level counters and 300 unknown entries for each of
the two nested fields. Per-row and per-role sums match the aggregate. Prompt
plus completion tokens equals reported total tokens. Reasoning and cached tokens
are not added to or subtracted from that total. No billable-token, visible-token,
cache-savings, energy, or cost estimate is derived.

## Timing and terminal status

- Driver-invocation wall time: **1,525.444 seconds**
  (unrounded: 1525.4444866259582).
- Sum of 1,200 known role latencies: **14,805.374 seconds**
  (unrounded: 14805.374327109661); unknown latency count: 0.
- The 300 supplied finite item-wall values sum to 14807.638186655007 seconds.
  They include pipeline overhead and overlap across items; this is not another
  end-to-end wall-time estimate.
- Wrapper start: `2026-11-10T17:23:52.429217Z`.
- Wrapper finish: `2026-11-10T17:49:18.031555Z`.

Overlapping request latency sums are not invocation wall time, exclusive GPU
compute time, or an efficiency claim. Individual role latency values are not in
the sanitized summary; their aggregate is reported by the frozen scorer, not
independently recomputed here. Wrapper timestamps cover a different scope from
the driver's invocation timing. Offline scoring occurs later and is not included
in these measurements.

The wrapper reports exit code 0 with `SystemExit`, no error code, one invocation,
and no resume. The sanitized progress ends with `completed` for the same run and
condition. Its terminal name/hash agree with the operator-supplied binding:
`invocation-terminal-461152dc105745148ba536a03c46c2e1.json`.
A wrapper completion is only invocation-status metadata, not a native receipt,
model attestation, or score.

## Evidence bindings

The safe rows are exactly 300 unique sorted indices. Their strict-correct flags
sum to 268, their any-role-length flags to 67, and each has four role records.
Provider totals and known/unknown counts reconcile from both item and role
summaries. The paired-cohort hash was rebuilt from safe question hashes and the
pinned manifest hash, without reading any question text. The projection hash
matches the prior safe inspection metadata.

Canonical JSON object hashes are labeled separately from file-byte hashes.
Private artifact and terminal hashes below come from the operator's audit; they
are not files independently opened for this report. In particular, the
sanitized completion hash identifies the **local watcher file bytes**, not the
original remote completion-file byte sequence. Digests support content identity
and consistency; they are not cryptographic daemon authenticity, signatures, or
protection against a malicious same-user host rewriting all evidence together.

### Run and implementation hashes

| Artifact | SHA-256 |
| --- | --- |
| Pinned MedQA300 manifest (operator-verified file bytes) | `8734b1d3e08b3a9525fd9b628476cb285857d46fb576e87be42f9b600817fde0` |
| Question-only sequential projection (canonical JSON object) | `920e83a39444950c8b97dfbaf739b0076d448722ed77bbc5dc70655a88dc524f` |
| Paired cohort shared with baseline (canonical JSON object) | `56ccd3e7884e0f98adebe30dc521a86fbe9e8f7396bd8d6ee7c1d78cdedfa79d` |
| Sequential run provenance (canonical JSON object) | `4a475ddf6ea69042c94b76a26277ae3e926d86fbff6a3c71f73358f57b1fd71e` |
| Private items.jsonl (operator-audited file bytes) | `e7c3d9830aafab2471223b40c42526b70dd5d0e62a8005245ce187e01a3b6475` |
| Private journal.jsonl (operator-audited file bytes) | `2823816826412b8dda0aa327d547ba44cf1c7af899f5f23d386dd28d3fb61b3c` |
| Private roles.jsonl (operator-audited file bytes) | `b496849dd6d12a350e1a618de24d0d8eff63ad91488054ab3472bc5837559cbc` |
| Invocation terminal (operator-supplied file bytes) | `ebb3f28f2f19a547c5a2383744ca2715017ee7d1cfa6b0471c175f1cd62d2384` |
| Sanitized score summary (locally verified file bytes) | `37ce77832256c0876413c0128abf322d83525893ba0d00bde7f9e74ade13f2a9` |
| Sanitized completion (local watcher file bytes, not original remote bytes) | `45d361366eb37648e26a271a558f0592d4253fb0e3fd23fe925184b9a277b9f1` |
| Sanitized final progress (local watcher file bytes) | `73a4dc48ca1fd44b378f7aed5684c895d098c1ef2225333527a2e46f670ce75e` |
| Reviewed sequential launch spec (local file bytes) | `908a196e9b18e9bc2ecf574365514ba61a417ccaf3253b8940882de436352522` |
| Frozen noncode driver | `f829c4660930a79506629f0e333d8cc42aac876576976b3695b0598643e6cb53` |
| Frozen noncode common module | `a8e426f0991bb147b6933906bc0644a343b722b8c773a9abf36dc70b391fffce` |
| Frozen offline scorer | `ef81ad75fddefbd09be5a07657f746865d0ceb5fbd8b35d397b4ffcee0d61f90` |
| Frozen foreground wrapper | `6dee4e0af80de64a57c2e2872e04700f7957dd9ffc1a0c96946aeec808cbc9d9` |
| Verified dependency bench_gpqa_textmas.py | `6b7eaa11be5394870b41cdd7dbc241bdd2c85ee711570738adaef7ad7847e1af` |
| Verified dependency bench_gpqa_textmas_common.py | `55dffad169036a22246f00bdec2dab8b4508377c1ab9f5e73949f27b2c572478` |
| Preserved baseline safe summary (file bytes) | `9ce9fa64ad97069509313742a23f4057aadbbb0d39b4bd9b206fec375c69fb1d` |
| Published MEDQA_BASELINE.md (verified file bytes) | `24e96b48242bd95d1a8ffc60a62795a0a6e8593417b3e7fb452d9c8544d141b2` |

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

This report describes one sampled run on one fixed cohort. No confidence
interval, statistical significance, causal improvement, or clinical performance
claim follows from it. The preserved one-request baseline and four-request
sequential condition have different inference budgets. No outcome is repaired
or rerun because of its score, and no unverified model image or runtime revision
is inferred from older reports.

The report contains no item questions, options, raw model answers, extracted
predictions, gold values, reasoning text, credentials, raw request payloads, or
private transcripts. Existing results and frozen implementation files remain
unchanged. Publication and any future inference remain separate operator actions.

Local artifacts: [safe sequential summary](results/medqa_textmas_sequential_safe.json), [preserved baseline](MEDQA_BASELINE.md), and [report index](README.md).
