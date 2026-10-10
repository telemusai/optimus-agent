# Memory evidence and reference architecture checkpoint

## Scope and status

This is a corrected checkpoint, not a final campaign report.
The source baseline is [`b8695ec81016028a8137229588807333ec4ea074`][optimus-base].
It includes merged PRs [#124][pr124], [#125][pr125] and [#126][pr126].
Product correctness fixes are in [PR #127][pr127], **open, not merged**, at commit [`43b1993ebb701aa5ae3da812af6ffaf418348aa0`][correctness-commit].
That PR contains nine product/test/changelog files. Offline replay/comparison tools are documented [separately](memory-benchmark-tools.md). Driver changes and the portable evaluator remain separate review work; the evaluator is held for license and protocol repairs.
This is not an installed-release or production-profile rollout claim.

**No Jev:** current work and every planned build, configuration and experiment exclude it.
This applies to chat, tools and every benchmark arm, and supersedes earlier phase plans.
Reference repositories were read-only, untrusted study inputs. No reference installs or scripts were run for this review.
This document copies no reference code, prompts, datasets, private histories or credentials.
No new paid task benchmark was run for this checkpoint.
Correctness tests do not establish task-quality gains or diminishing returns.

## Seven pinned references

These are the exact seven study repositories, not seven adopted memory implementations.
Licenses below describe the inspected root license at the pinned commit; nested assets can have separate terms.

| Reference | Repository URL | Exact commit | License at pin |
|---|---|---|---|
| R1 MemoryLake benchmark | https://github.com/memorylake-ai/memorylake-locomo-benchmark | `4ede2a6a37eebc6897c585a7a7bb1cffa287b8b6` | No license found; study only, no copying |
| R2 Mem0 | https://github.com/mem0ai/mem0 | `b7ad69afda6b6ed030347c66d48a13e4de9dec08` | [Apache-2.0][mem0-license] |
| R3 Independent memory-benchmarks | https://github.com/rellocode/memory-benchmarks | `9ccc46877aab856dd15833c46d1f4b9c9d13632e` | [Apache-2.0][bench-license] |
| R4 EverMemOS / EverOS | https://github.com/EverMind-AI/EverMemOS | `d2aa9494da062246e665a21e3f045583d81df90f` | [Apache-2.0][everos-license] |
| R5 Graphiti | https://github.com/getzep/graphiti | `1026ae7ae25e7e4cfcc1c7ebe00347b8a90f52a0` | [Apache-2.0][graphiti-license] |
| R6 MemOS | https://github.com/MemTensor/MemOS | `a7367d07e55db61099f7b4e2c1108bc5831a24f3` | [Apache-2.0][memos-license] |
| R7 LoCoMo | https://github.com/snap-research/locomo | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | [CC-BY-NC-4.0][locomo-license] |

R7 corrects the earlier metadata's `NOASSERTION`: `LICENSE.txt` explicitly contains Attribution-NonCommercial 4.0.
Do not treat that material as Apache-licensed or assume commercial redistribution rights.
Root licenses alone do not clear inherited evaluator code; retain attribution and review each source's terms before distribution.
EverOS is the renamed repository behind the supplied EverMemOS URL, not a replacement reference.

## Practical feature comparison

This compares source behavior, not product quality or leaderboard rank.
“Optimus” means the baseline plus the limited continuation corrections identified below.

| System | Capture and persistent state | Retrieval and temporal behavior | Practical lesson and cost boundary |
|---|---|---|---|
| Optimus | Evidence-cited refinement/import; JSON stores, revision history, source metadata and non-destructive supersession | Scoped IDF lexical ranking; bounded injection and freshness checks; optional model-assisted recall | Keep the local deterministic core. Fix provenance, eligibility and activation before changing storage or ranking architecture. |
| R1 MemoryLake benchmark | Evaluation harness/results, not evidence for a locally deployable storage design | Reported LoCoMo protocol differs from ours | Methodology comparison only. No-license source is not an implementation source. |
| R2 Mem0 | Current OSS V3 additive extraction, batch hash dedup and entity links; explicit observation-date context | Vector/keyword retrieval and optional reranking | Study date authority, detail retention and dedup context. Adds embedding/provider/store dependencies; do not assume the old update-decision loop is the current default. |
| R3 memory-benchmarks | Independent comparison harness | Different answer/judge protocol and category coverage | Pin models, metrics, exclusions and denominators. Its headline is not an Optimus target measured under the same conditions. |
| R4 EverOS | Boundary detection into MemCells; user episodes and agent cases/skills; markdown-first state with LanceDB synchronization | Typed recall, hybrid/agentic paths; merged episodes deprecate originals | Study linked episode/fact/case provenance and retryable derived indexes. Do not import its service/index stack without a measured need. |
| R5 Graphiti | Entity/episode graph, deduplication and time-aware invalidation | BM25, vectors and graph traversal; selectable fusion/reranking; queryable validity intervals | Distinguish event validity from ingestion time. Graph/embedding/reranker infrastructure is not a minimal lexical-memory patch. |
| R6 MemOS local plugin | Skill, experience, trace and world-memory paths | Deterministic query construction, lexical/vector/identifier channels, bounded LLM filter | Study query-source separation and safe fallbacks. Its multi-channel/vector design and injection policy are not drop-in defaults. |
| R7 LoCoMo | Dataset and evaluation protocol, not a memory service | Token-F1 evaluation; distinct question categories | Preserve metric/category semantics and dataset terms. Do not compare F1 with binary judge accuracy. |

### Source corrections that change the design reading

- **Mem0:** [`mem0/memory/main.py`][mem0-add] selects `ADDITIVE_EXTRACTION_PROMPT` in its V3 add path.
  Legacy ADD/UPDATE/DELETE prompts still exist, but they do not describe that default path.
  The [current prompt][mem0-prompts] favors context-rich statements, uses an observation date, and covers both speakers.
- **EverOS:** [`service/memorize.py`][everos-memorize] routes MemCells through user/agent pipelines.
  [`memory/reflection/orchestrator.py`][everos-reflection] deprecates source episodes after consolidation.
  The earlier MemCube/MAVen description does not map to this pinned implementation.
- **Graphiti:** [`Graphiti.search_`][graphiti-search] defaults to the combined cross-encoder recipe, not plain RRF.
  Its [RRF helper][graphiti-rrf] uses `rank_const=1`; MemOS's [keyword helper][memos-keyword] uses `k=60`.
  These constants and candidate requirements cannot be assumed equivalent.
- **MemOS:** the [query builder][memos-query] is deterministic; its [LLM filter][memos-filter] falls back to a bounded mechanical cutoff.
  A valid empty selection stays empty. Optimus's lexical fallback has different semantics and remains deliberate.

## Production defaults are not benchmark recipes

At the baseline, IDF ranking and import-apply hardening are real product capabilities.
The full capture recipe used by the campaign is not the shipped default.
See `crates/pi-coding-agent/src/core/memory/{store,service,jobs,evidence,search}.rs`
and `crates/pi-coding-agent/examples/memory_bench.rs` at the baseline pin.

| Setting or behavior | Baseline product | Historical benchmark recipe |
|---|---|---|
| Recall / learning | Both enabled; automatic refinement is a separate production path | Explicit import workflow; automatic refinement disabled by driver default |
| Recall limits | 6000 characters, 6 entries | 8f/8g/8h kept these limits; 12000 characters was a separate trial |
| Import chunk / extraction limit | 40000 characters / 4096 output tokens | Base: 15000 / 32000; 8f/8g/8h: 8000 / 32000 |
| Import instructions | No configured override; entry points have different durable-project-fact instructions | `CAP_INSTR` was an explicit host-import override, not a default prompt |
| Speaker origins | User and assistant evidence remain distinct | Driver relabels supplied transcript events as user evidence; this is not permission to trust actual assistant assertions |
| Query distillation / rerank | Declared default-off; baseline settings overlay dropped configured values | 8h saved both flags as true, but effective flags stayed false |
| Python evaluation suite | Absent from the baseline Git tree | Retained campaign tooling; portable draft is held from publication for license review, not something PR #124 shipped |

The continuation preserves these defaults and limits.
Fixing flag propagation means an existing explicit `true` setting can now cause helper calls.
That compatibility correction needs a cost/latency notice; it is not evidence to enable either feature by default.
Host-import instruction overrides also must not be described as uniform terminal-import or automatic-learning behavior.

## Accepted minimal corrections and architecture reasoning

The write path remains evidence -> chunked extraction -> validated proposal -> receipt-backed apply -> JSON state/history.
The read path remains query -> eligible corpus -> lexical rank -> bounded selection -> fallible-context injection -> diagnostics.
No destructive migration, new database/server dependency, or mandatory embedding service is introduced.
Existing IDs, scopes, persisted records and public commands remain in place.

| Correction and delivery boundary | Concrete defect addressed | Why this boundary is minimal |
|---|---|---|
| PR #127: settings propagation in `core/memory/store.rs` | Validated and persisted recall flags never reached effective settings | Copy the existing optional flags through the overlay. Keep false defaults and test the real configure/read path. |
| PR #127: `memory.apply` source intent across Python, service and store | Omitted `sources` became `[]`, clearing provenance; receipts collapsed preserve and clear | Preserve omission, keep explicit clearing, and distinguish new receipts. Accept legacy hashes read-only; old receipts cannot recover the lost distinction. |
| PR #127: scope-consistent IDF in `core/memory/search.rs` | Ineligible foreign/inactive records changed eligible scores | Compute DF and corpus size over the same eligible population used for scoring. Preserve formula, field weights and tie-breaks. This is correctness, not a measured ranking gain. |
| PR #127: recall helper runtime in `core/extensions/builtin/memory.rs` | Implicit retries, cancellation/error handling, malformed output, cache reuse and candidate identity failures | Bound dispatch, preserve cancellation and deterministic fallback, cache the original turn, and select exact transmitted top-20 records. Reject ambiguous scoped-ID collisions. |
| Review only: feature realization in `examples/memory_bench.rs` | Requested settings and historical execution were conflated | Fail early on requested/effective mismatch; record effective settings, product/driver identity and optional helper observations. No historical backfill. |
| Review only: offline comparator, snapshot exporter and native recall probe | Mixed variants, missing rows, mutable stores and uncertain artifact joins obscured comparisons | Make selection, hashes, scope inclusion and settings authority explicit. The native probe is lexical-only; it does not validate model-helper quality. |
| Review only: offline tool tests using synthetic inputs | Missing/error/variant handling and snapshot safety lacked regression coverage | Verify the tools without distributing histories or contacting providers. Tests do not establish benchmark validity. |

Relevant repository paths are `resources/agent/skills/memory/src/memory/__init__.py`,
`crates/pi-coding-agent/src/core/memory/{service,store,search}.rs`,
`crates/pi-coding-agent/src/core/extensions/builtin/memory.rs`, the two memory examples,
`scripts/compare_memory_bench.py`, and `scripts/export_memory_bench_snapshot.py`.
`recallHelpers.*.attempted` records dispatch, not completion, billable usage or task success.
Missing old diagnostics remain unknown, not inferred active or inactive from new code.

**Publication hold:** the portable `bench/` draft contains inherited evaluator material with unresolved licensing.
It is excluded from the product PR pending resolution, including unresolved LoCoMo-Plus prompt/parser rights.
Source-only packaging and passing portability tests do not remove those restrictions.
No upstream prompt/algorithm parity or completed harness delivery is claimed.

## Corrected historical task results

These are development-set observations from saved artifacts, not new results from the continuation fixes.
Historical answerer/model-judge runs used DGX GLM-5.3; LoCoMo scoring was deterministic.
`U/R/A` means score-universe rows / selected run records / nonempty answers.
Scores below use recorded-qid denominators, retaining historical flagged zeros; they are not unflagged-answer-only estimates.
`E`, `D`, `J` mean empty answer, driver error, evaluator/judge error. Unrun rows are listed separately.
Wilson intervals describe thresholded recorded scores, including flagged zeros; they do not cover fractional F1 or a causal effect.

| Family / arm | U/R/A | Score on recorded qids | Binary 95% Wilson interval | Failures among records; unrun |
|---|---:|---|---|---|
| LoCoMo no memory | 300/300/300 | mean F1 .2257; accuracy@.5 68/300 = 22.67% | 18.3–27.7% | none; 0 |
| LoCoMo base | 300/300/300 | mean F1 .2891; accuracy@.5 80/300 = 26.67% | 22.0–31.9% | none; 0 |
| LoCoMo capture 8f | 300/300/299 | mean F1 .3395; accuracy@.5 99/300 = 33.00% | 27.9–38.5% | 1 E; 0 |
| LoCoMo IDF 8g | 300/300/299 | mean F1 .3363; accuracy@.5 98/300 = 32.67% | 27.6–38.2% | 1 D; 0 |
| LoCoMo-Plus base | 120/120/120 | binary 50/120 = 41.67% | 33.2–50.6% | 4 J; 0 |
| LoCoMo-Plus no memory | 120/120/120 | binary 41/120 = 34.17% | 26.3–43.0% | 5 J; 0 |
| LoCoMo-Plus IDF 8g subset | 120/15/15 | binary 8/15 = 53.33% | 30.1–75.2% | 1 J; 105 |
| LoCoMo-Plus 8h subset | 120/15/15 | binary 3/15 = 20.00% | 7.0–45.2% | 1 J; 105 |
| LongMemEval base | 100/50/50 | binary 17/50 = 34.00% | 22.4–47.8% | none; 50 |
| LongMemEval no memory | 100/50/50 | binary 4/50 = 8.00% | 3.2–18.8% | none; 50 |
| LME-v2 base | 120/120/108 | binary 36/120 = 30.00% | 22.5–38.7% | 11 E + 1 D + 1 J; 0 |
| LME-v2 no memory | 120/120/116 | binary 22/120 = 18.33% | 12.4–26.2% | 4 E + 1 J; 0 |

The LongMemEval headline is **34% versus 8%, n=50 answered in each arm**, not 17% versus 4% on 100 answers.
The saved 17%/4% aggregates zero-filled the same 50 unrun questions.
Likewise, saved 8g/8h LoCoMo-Plus aggregates of 6.67%/2.50% used 120 rows, not the 15 answered subset.
Do not use shard-only or mixed-variant aggregates as complete-arm results.

### Paired uncertainty and metric definitions

Wins/losses/ties favor the first named arm. Tests are two-sided exact, exploratory and not corrected for repeated comparisons.
McNemar applies to binary correctness; a sign test applies to raw fractional-F1 differences.
Question-level tests/intervals do not account for clustering within shared conversations or environments.

| Comparison | Common qids | Binary W/L/T; McNemar p | Raw F1 W/L/T; sign-test p |
|---|---:|---|---|
| LoCoMo base vs no memory | 300 | 17/5/278; .01690 | 67/5/228; 6.39e-15 |
| LoCoMo 8f vs base | 300 | 33/14/253; .00794 | 69/50/181; .09852 |
| LoCoMo 8g vs 8f | 300 | 5/6/289; 1.00000 | 23/31/246; .34089 |
| LoCoMo-Plus base vs no memory | 120 | 26/17/77; **.22205** | Not fractional |
| LoCoMo-Plus 8h vs 8g | 15 | 0/5/10; .06250 | Not fractional |
| LongMemEval base vs no memory | 50 | 13/0/37; .000244 | Not fractional |
| LME-v2 base vs no memory | 120 | 19/5/96; .006611 | Not fractional |

The 23/31/246 and 5/6/289 LoCoMo comparisons are both correct: they measure different outcomes.
Neither p-values nor narrow code tests remove protocol, store, model-sampling or judge confounds.
A nonsignificant result, including LoCoMo-Plus p=.222, is not proof of no effect.

### Withdrawn interpretations and bounded diagnostics

- **8h never activated the intended treatment.** At the baseline and PR #126 branch tip, `apply_settings` omitted both flags.
  The saved true flags therefore did not enable the gated calls. The mechanism is **unmeasured**, not shown harmful.
  Equal direct/injected recall IDs on 15/15 rows support the source finding; latency alone is not activation proof.
  Independent extraction produced different stores, and answer/judge failures also differ. The lower score has no isolated cause.
  The earlier token-budget-exhaustion explanation is withdrawn: those helper budgets were not exercised.
- **`lp_107` is not evidence for an IDF gain or an 8h retrieval regression.** Its gold was stored but not injected or in top-50 search in the audited arms.
  The credited 8g answer was generic inference; 8h produced narration. The authoritative 8g record resolves to the 841-entry primary store, not the superseded 952-entry split.
- **LoCoMo-Plus narration is a judge/protocol risk.** A heuristic audit found credited narration in 12 base and 2 no-memory answers.
  Removing that credit gives 38/120 versus 39/120 as a diagnostic sensitivity analysis only.
  The labels were not blinded/validated; this does not replace the recorded 50/120 versus 41/120 scores.
- **LoCoMo categories were mislabeled.** Correct mapping: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
  Base-to-8f single-hop accuracy was 13/127 -> 33/127; temporal remained 1/48 -> 1/48.
- **Historical adversarial option labels were biased.** The converter shuffled already-labeled alternatives, leaving the correct abstention letter `a` on every category-5 item. This is not the upstream random label assignment. Preserve those old scores as protocol-specific artifacts; corrected options require a new protocol and new matched runs, not silent relabeling.
  This is a bundled capture-recipe observation, not proof of one prompt, role or chunk-size change.
- **LME-v2 used a 2000-character accessibility-tree cap, not 8000.** The audit found cap-removed phrase text in 7/52 phrase-set rows; 20/52 lacked that text even in the raw haystack.
  These are text-coverage diagnostics, not a semantic answerability proof. Screenshots were not ingested despite 29 image-referencing questions.
  The enterprise base store retained 582/1648 entries from the older protocol era after re-import; its 57-question shard is not a clean capture baseline.
  The 36/120 versus 22/120 aggregate is retained as historical observation, not a controlled architecture result.
- **Historical lineage is incomplete.** Reused run IDs, duplicate qids, last-wins merges and rewritten stores need explicit selection rules.
  A hash of today's saved executable does not authenticate every old row; old build fingerprints also omitted example source.
  Import job identity hashes project plus raw transcript bytes, not chunk settings. Changed role bytes created new generations; chunk-size changes were not the diagnosed cause.
  Per-environment import totals can omit superseded work or count already-completed chunks. They are not reliable marginal-cost evidence.

External 91–94% headlines are not comparable to these rows.
For example, the MemoryLake and independent comparison harnesses exclude adversarial category 5 (1540/1986 questions remain),
and use different models and/or binary judges rather than this campaign's F1-plus-abstention protocol.
No quality ordering among the seven references and Optimus is established here.

## What still needs controlled and heldout evaluation

1. **Freeze provenance before spending.** Pin source, driver binary, model/judge configuration, question/input hashes, extraction instructions, effective settings and store snapshots.
   Stop on mismatches. Record feature dispatch/outcome, fallback, errors, retries, latency and usage separately; prove No Jev configuration for every arm.
2. **Separate retrieval from capture.** Retrieval arms must use byte-identical frozen stores with identical included scopes, source files and question protocol.
   Compare lexical control, distill-only, rerank-only and both; first confirm activation. Preserve the current deterministic fallback and test latency/cost as well as answers.
   Capture arms need separate fresh stores, fixed inputs and repeated independent ingestions; do not change ranking at the same time.
3. **Repair the task protocol before interpreting gains.** Pin a conversational answer instruction, validate narration/judge labels blind, and report missing answers separately from missing judgments.
   Keep unmodified historical scores plus explicitly named sensitivities. Review LME-v2 truncation, screenshots, clean protocol-era ingestion and abstention parsing.
4. **Test source-grounded temporal capture and task memory as hypotheses.** The current driver uses synthetic times and evidence labels lack a trusted date anchor.
   Only dataset/source-authoritative dates may ground facts; absent dates must stay absent. Never promote a synthetic ordering timestamp to truth.
   Evaluate coverage/specificity, near-duplicate control, procedural cases, updates/conflicts and abstention separately before changing production prompts or schemas.
5. **Use untouched heldout material.** Pre-register each family's primary metric, paired analysis, minimum relevant effect and stopping rule.
   Report total/attempted/answered/judged/common qids, all failure types, interval assumptions and category coverage.
   Use clustered/repeated-run uncertainty where appropriate. Keep development tuning out of final heldout estimates.
6. **Measure real operation separately.** Imported-history QA does not measure ongoing automatic learning, tool-use success or deployment reliability.
   Verify legacy state/settings loading, source integrity and cancellation with correctness tests, then measure task success and cost under an authorized operational protocol.

The next decision is whether these isolated changes improve heldout tasks at acceptable cost.
That question remains open. This checkpoint makes no final-success or diminishing-returns claim.

## Evidence map and validation boundary

Repository regression locations: `crates/pi-coding-agent/tests/{memory_compatibility,memory_source_integrity}.rs`,
the runtime/search/example test modules, and `tests/test_memory_{skill_payloads,bench_comparison,bench_snapshot_export,bench_portability}.py`.
These use synthetic/offline fixtures. The coordinator reports the following correctness checks, separately from task benchmarks:

| Check | Result |
|---|---|
| `bash scripts/check.sh` | Passed |
| Rust recall runtime / compatibility / source integrity | 18/18 / 12/12 / 10/10 passed |
| Rust core memory / examples | 55/55 / 22/22 passed |
| Python memory suites | 118/118 passed |

These checks cover the review worktree; example/tooling files are not all in PR #127.
They establish neither live-provider quality nor heldout task gains. No new task scores are attributed to that PR.

Retained campaign artifacts are not bundled here. The audit trail uses relative names:
- `scores/locomo_{base,nomem,cap8f_full,idf8g}.json`, `scores/locomo_plus_{base_full,nomem_full}.json`, `scores/loco_plus_{idf8g,8h}.json`.
- `scores/longmemeval_{base,nomem}.json`, `scores/lme_v2_{base,nomem}.json`, and the selected run JSONL files named by those artifacts.
- `reports/continuation/{benchmark-evidence-review,production-defaults-audit,capture-reference-review,retrieval-reference-review,import-resume-audit}.md`.
- `reports/continuation/evidence-docs-recalculation.json`: score-byte hashes, selected answer counts, exact paired recalculation and binary intervals for this checkpoint.
Earlier audit drafts contain superseded interpretations. The explicit corrections above take precedence over stale summaries.

[optimus-base]: https://github.com/telemusai/optimus-agent/tree/b8695ec81016028a8137229588807333ec4ea074
[pr124]: https://github.com/telemusai/optimus-agent/pull/124
[pr125]: https://github.com/telemusai/optimus-agent/pull/125
[pr126]: https://github.com/telemusai/optimus-agent/pull/126
[pr127]: https://github.com/telemusai/optimus-agent/pull/127
[correctness-commit]: https://github.com/telemusai/optimus-agent/commit/43b1993ebb701aa5ae3da812af6ffaf418348aa0
[mem0-license]: https://github.com/mem0ai/mem0/blob/b7ad69afda6b6ed030347c66d48a13e4de9dec08/LICENSE
[bench-license]: https://github.com/rellocode/memory-benchmarks/blob/9ccc46877aab856dd15833c46d1f4b9c9d13632e/LICENSE
[everos-license]: https://github.com/EverMind-AI/EverMemOS/blob/d2aa9494da062246e665a21e3f045583d81df90f/LICENSE
[graphiti-license]: https://github.com/getzep/graphiti/blob/1026ae7ae25e7e4cfcc1c7ebe00347b8a90f52a0/LICENSE
[memos-license]: https://github.com/MemTensor/MemOS/blob/a7367d07e55db61099f7b4e2c1108bc5831a24f3/LICENSE
[locomo-license]: https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt
[mem0-add]: https://github.com/mem0ai/mem0/blob/b7ad69afda6b6ed030347c66d48a13e4de9dec08/mem0/memory/main.py#L881-L1213
[mem0-prompts]: https://github.com/mem0ai/mem0/blob/b7ad69afda6b6ed030347c66d48a13e4de9dec08/mem0/configs/prompts.py#L468-L942
[everos-memorize]: https://github.com/EverMind-AI/EverMemOS/blob/d2aa9494da062246e665a21e3f045583d81df90f/src/everos/service/memorize.py
[everos-reflection]: https://github.com/EverMind-AI/EverMemOS/blob/d2aa9494da062246e665a21e3f045583d81df90f/src/everos/memory/reflection/orchestrator.py
[graphiti-search]: https://github.com/getzep/graphiti/blob/1026ae7ae25e7e4cfcc1c7ebe00347b8a90f52a0/graphiti_core/graphiti.py#L1729-L1740
[graphiti-rrf]: https://github.com/getzep/graphiti/blob/1026ae7ae25e7e4cfcc1c7ebe00347b8a90f52a0/graphiti_core/search/search_utils.py#L1775-L1790
[memos-keyword]: https://github.com/MemTensor/MemOS/blob/a7367d07e55db61099f7b4e2c1108bc5831a24f3/apps/memos-local-plugin/core/storage/keyword.ts#L192-L194
[memos-query]: https://github.com/MemTensor/MemOS/blob/a7367d07e55db61099f7b4e2c1108bc5831a24f3/apps/memos-local-plugin/core/retrieval/query-builder.ts
[memos-filter]: https://github.com/MemTensor/MemOS/blob/a7367d07e55db61099f7b4e2c1108bc5831a24f3/apps/memos-local-plugin/core/retrieval/llm-filter.ts
