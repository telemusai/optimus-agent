# Offline memory benchmark tools

These tools inspect explicit artifacts. They do not run a model, Jev, ingestion,
or a live agent session. They do not need the separate `bench/` evaluator package.
See [the evidence checkpoint](memory-evidence-review.md) for historical limits.

## Privacy and input ownership

Outputs are **private, content-bearing evidence by default**. Memory exports and
probe output include full memory content, source URIs, project/host identifiers,
settings, history and paths. Comparison reports can also contain local paths.
The exporter's filename restrictions are not a content-redaction guarantee.
Do not publish raw bundles, caches, runs or reports without a separate review.
If a public redacted derivative is needed, record its own hash and provenance;
do not alter the original evidence or claim the original hash covers redaction.

Use only authorized, stopped benchmark inputs. `--frozen` is the caller's
attestation, not a transactional snapshot or protection against hostile writers.
Do not point these tools at the installed assistant profile. No reference
repository is imported, executed or downloaded by these tools.

## Export a frozen native store

Create an empty parent output directory. Each output bundle directory below must
be new. Select the original project identity and source revision explicitly:

```text
python3 scripts/export_memory_bench_snapshot.py snapshot --run-dir FROZEN_RUN --run-id RUN_ID --env-id ENV_ID --project-id PROJECT_ID --project-root ORIGINAL_ROOT --code-revision SOURCE_REVISION --capture-stage final_state --frozen --persisted-settings-for-current-replay --output-dir OUTPUT/snapshot
```

The exporter reads the selected `envs/ENV_ID` native files, not entry-only
`env.jsonl` snapshots. It never exports `models.json`, `auth.json`, credential
files or the complete application settings file. Optional global, session and
shared corpora require explicit `--include-scope` choices. Included and omitted
corpora are recorded; changing the corpus can change IDF.

File-origin evidence needs explicit `--allow-source-root` authorization. Unsafe,
hard-linked, non-UTF-8, oversized, unreadable or changing sources fail rather than
being silently replaced. Only confirmed absence is exported as missing.

Choose settings authority explicitly:
- `--persisted-settings-for-current-replay`: replay saved requested memory
  settings with current code. Historical effective settings remain unverified.
- `--effective-memory-settings FILE --settings-provenance LABEL`: use an explicit
  caller-supplied complete effective settings object. This is an attestation,
  not proof of historical helper execution.

The fixture and sidecar record byte hashes and selected source provenance.

## Export gold-free queries and replay native lexical recall

```text
python3 scripts/export_memory_bench_snapshot.py queries --input QUESTIONS.jsonl --output-dir OUTPUT/queries
cargo build --locked -p pi-coding-agent --example memory_recall_probe
memory_recall_probe OUTPUT/snapshot/fixture.json OUTPUT/queries/queries.jsonl > OUTPUT/recall.jsonl
```

Use the executable under the Cargo target directory (with `.exe` on Windows).
Query export keeps literal query text, identity and supported options. It drops
answer/gold/evidence fields; it does not infer or strip an instruction wrapper
inside the literal text. Run/environment selectors are required when present.

The probe materializes only private temporary native files and runs actual
`MemoryService::search` and `recall_memory`. Original paths in the fixture are
provenance labels, not live inputs to open. The exported files are not changed.
Its strict query schema rejects gold/unknown fields and duplicate query/variant
pairs. It reports scope-filtered candidates, native scores, rendered IDs/text,
budgets, settings and implementation/input hashes. Check process exit status:
a late invalid query can fail after earlier output rows have been written.

This is lexical search and rendering only. `injected_ids` is the renderer's
selection, not proof of injection into a running agent. Distillation/reranking
are never executed even if persisted settings request them. No accuracy or
historical LLM activation is established by a probe result.

## Compare scored answer artifacts

```text
python3 scripts/compare_memory_bench.py --arm control CONTROL_RUN.jsonl CONTROL_SCORES.json --arm candidate CANDIDATE_RUN.jsonl CANDIDATE_SCORES.json --out COMPARISON.json
```

Each arm is explicit. Variant selection and any duplicate policy must be
explicit when needed; duplicates fail by default. Existing output paths that
alias inputs through a hard link fail before writing. Input inspection errors
also fail closed.

The report separates attempted coverage, answered common-qid pairing, driver
failures, score flags and a recorded-flag sensitivity analysis. A 120-row score
file for a 15-question run does not make 120 questions answered. Missing scores
are not silently imputed by the comparator. The paired answered denominator is
not an intention-to-treat estimate or a randomized causal comparison.

A supplied `run_sha256` must match the exact input bytes. This verifies only run
content, not the judge, question/gold data, evaluator parameters, or which
prediction a scorer chose from duplicate records. Historical files without it
remain explicitly metadata-only. Do not pair changed question sets or use
`--dedup first` for scores generated with a last-record policy. Require a separate
question/evaluator/selected-record binding before making strict efficacy claims.

No official evaluator, dataset, judge prompt or licensed reference implementation
is bundled with these standalone tools. Import scores only from a separately
reviewed, explicitly configured scoring protocol. Preserve older artifacts when
a protocol changes; do not silently relabel them as a corrected experiment.
