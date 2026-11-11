# GPQA baseline completion

## Result

The GLM-5.3 baseline is complete for all 198 items in the pinned GPQA manifest.
The continuation added indices 160–197. It did not replace the historical rows.

| Segment | Indices | Correct / total | Accuracy | Length-limited |
|---|---:|---:|---:|---:|
| Historical baseline | 0–159 | 48 / 160 | 30.00% | 103 / 160 |
| Continuation | 160–197 | 25 / 38 | 65.79% | 12 / 38 |
| Combined | 0–197 | **73 / 198** | **36.87%** | **115 / 198 (58.08%)** |

The continuation journal contains 38 starts and 38 completions. All 38 responses
have HTTP 200 and `response_received` status. The terminal inventory has no failed
or unstarted indices. The offline score summary reports no missing, failed, or
incomplete rows. The operator's aggregate records that the historical file stayed
unchanged after completion.

This report uses only sanitized metadata and aggregate scoring evidence. It does
not independently rescore private answers. It checks index coverage, arithmetic,
terminal counts, and the supplied artifact-hash relationships.

## Condition and segmented provenance

- Actual model: `glm-5.3`.
- Condition: single-agent baseline; no TextMAS or native OCC-3 handoff is involved.
- Request settings: `max_tokens=4096`, `temperature=0.6`, `top_p=0.95`.
- The continuation preserved the original 16-request chunk/concurrency limit.
- Prompt source: LatentMAS `9a9e4d331eb11430bd9e64754c6b252b06d73031`.
  The original system prompt's literal Qwen identity was retained for prompt
  equivalence. No Qwen model was used.
- Reviewed harness snapshot: `e9067141a`.
- Historical segment: preserved baseline stack; its exact runtime is not
  established by the supplied safe evidence.
- Continuation segment: operator-labeled PR #281 overlay, run
  `gpqa-resume-20261110T135919Z`.
- The continuation used **one HTTP attempt per item**, rather than the historical
  client's automatic retry loop. This deliberate recovery-policy change is not
  hidden. The journal shows exactly 38 starts for the 38 items.
- Scoring remained the upstream extractor/normalizer on the completed final
  assistant response. No answer repair, earlier-message substitution, or
  gold-dependent retry was used by the continuation scorer.

The combined score is an aggregate across two runtime segments, not a single
fresh-stack rerun. The segments contain different questions. Their accuracy
difference is not evidence that the overlay improved model quality. Do not pool
historical and continuation latency into one homogeneous performance claim.

## Continuation-only cost and timing

| Metric | Value |
|---|---:|
| Items | 38 |
| Invocation wall time | 78.225 s |
| Mean request latency | 13.071 s |
| Median request latency | 9.380 s |
| Request latency range | 2.323–29.333 s |
| Prompt tokens | 10,492 |
| Completion tokens | 78,801 |
| Provider total tokens | 89,293 |
| Provider-reported reasoning tokens | 68,593 |

Terminal timestamp: `2026-11-10T14:01:52.912753Z`.
The wall time is the recorded invocation interval, not the sum of concurrent
request latencies. Every row satisfies `total_tokens = prompt_tokens +
completion_tokens`. Reasoning tokens are a separate provider-reported counter;
they must **not** be added again to total tokens. On nine rows that counter is
1–5 tokens larger than completion tokens. The sanitized rows preserve those raw
counts without adjustment. `prompt_tokens_details` is null on all rows, so
cache-read/write attribution is unavailable, not zero.

## Truncation and scope limits

- All 115 length-limited results remain in the combined denominator. They have
  not been discarded or rescored with a larger output budget.
- The cap stayed at 4096. **No 16k rerun was performed.**
- A length-limited response is not automatically equivalent to an incorrect
  answer. The supplied metadata does not establish the per-item relationship.
- Per-index correctness is absent from the safe source inventory. The companion
  JSON deliberately omits `correct`; it does not guess which 25 continuation
  rows were correct.
- Aggregate accuracy does not establish native transport use, native source
  retirement safety, or completed-use handoff behavior. Native source/receiver
  acceptance evidence is separate and unrelated to this ordinary baseline run.
- The local endpoint lock protects cooperating drivers only. The continuation's
  requested no-overlap policy is an operator control, not a GPU reservation
  attested by this report.

## Privacy and report artifacts

`gpqa_baseline_160_197_safe.json` contains exactly 38 sorted rows. Its only row
fields are `idx`, `truncated`, `usage`, and `latency_s`. It contains no question,
answer options, gold answer, prediction, reasoning text, or model output.

Safe rows SHA-256: `3ae402c3e3e8503baf156c415f629eb0afc65dcf8963aa81963b41a2bf9181cc`.

Private artifacts were not copied into this report directory. Their hashes are
retained below for operator reconciliation. A content digest is an integrity
reference, not a signature or independent runtime attestation.

| Artifact or identity | SHA-256 |
|---|---|
| GPQA manifest | `4f6347e51a6d0489b5bc6e8ef2e1e32c7424ecf64224a2fd707d0826df25b0ad` |
| Historical baseline JSONL (unchanged) | `866bd2ab33519f961995ab28b434b1c3af866ff4658ba3221e58fdaefa241545` |
| Continuation raw-results JSONL | `a24ea2243d085b4bd97f7bd21c978a293c1d00d75f36f1ccf0a4b52a314f0a1e` |
| Continuation journal JSONL | `de3ac55b59780a190a9659d9c8ff9b0439410c29942453c93e42242b3fbbd185` |
| Continuation provenance file bytes | `ac22e2c6d8ba6a61de9d22c19ffa3c56168e22aeed0818e0708ab9006b9e8f22` |
| Continuation provenance canonical JSON object | `2b03abb125250845f286f33a7060e1bfa4b570950438dc466588f069b9662dc4` |
| Private scored artifact (operator-supplied digest) | `3e9fe8a8943df22e0f4b8d70c176ed2798adb0e04827231bf06d7a49b2adca58` |
| Frozen offline projection/scorer driver | `6a21a5bfd1f4ffa405516a0f8ce41632d5a76ff5c71cb11e432c3395856726c5` |
| Frozen continuation driver | `90f47cfdca93f6fd0cd9c7124ba1d695730372239613a23a4c0b32b2066af88e` |

The provenance file-byte hash and canonical-object hash intentionally use
different encodings; they are not interchangeable.

### Safe evidence inputs

| Local evidence artifact | SHA-256 |
|---|---|
| `gpqa-aggregate-safe.json` | `99fe138a66e02a9a9b7b14db80dec2be7ad58e66f928c9e493aa95bae3bc7b33` |
| `gpqa-private-run-inventory.json` | `f7481a167a9d4be54200aee10da0ed5db656605684a6d4c474970078efc5f51a` |
| `gpqa-score-counts.json` | `7120f89ec9ccb9709b6dec9e7e89a5195d3d7066b76abbde2dbf4143ac30d7fe` |
| `gpqa-parent-review.json` | `2d2d408d47e0adfcf6c9cf201bf8a6b05063c72503de79e8943e73be8c733057` |

`gpqa-parent-review.json` is the pre-run review record. Its
`scored_results_not_yet_available` flag describes that earlier stage; the later
score-count and aggregate artifacts provide the completion result.

## Local review

- Exact index coverage 160–197, no duplicate indices: passed.
- Completed journal/terminal/HTTP counts agree at 38: passed.
- Continuation truncation count 12; combined count 103 + 12 = 115: passed.
- Aggregate correctness 48 + 25 = 73 out of 198: passed.
- Raw-results, journal, and canonical provenance digests agree across the safe
  inventory, terminal record, and aggregate where corresponding fields exist.
- Frozen drivers remain unchanged. Their recorded authoring checks were 16
  offline projection/scorer checks and 26 mock-HTTP driver checks.
- No Git, remote command, live inference, private dataset, or raw-output access
  was used to prepare this report.
