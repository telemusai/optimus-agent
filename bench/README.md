# Versioned memory evaluation core

This is a **new local protocol**, not an official four-benchmark evaluator.
It bundles original scaffolding, independent local metrics, and specifically
licensed MIT/Apache ports. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[PROVENANCE.json](PROVENANCE.json), and [ORIGINAL_SCORING.md](ORIGINAL_SCORING.md).
Real data, historical runs/scores/caches, credentials, reference repos and the
privately held old harness are not distributed. No MemoryLake code is included.

## Offline environment and tests

Python >=3.11; `pyproject.toml` declares no third-party dependencies. Run from
the repository root, not a reference repo or the agent kernel:

```sh
python -m venv --without-pip bench/.venv
# Windows; on POSIX use bench/.venv/bin/python instead.
bench/.venv/Scripts/python.exe -B -W error -m unittest discover -s tests -p test_memory_bench_portability.py -v
```

Set `TEMP`, `TMP`, and `TMPDIR` to a separate writable test directory. No model,
network, dependency install or installed profile is needed for the tests. The
suite includes the independent scoring module's 74 synthetic tests. Those tests
validate math, syntax and wiring, not actual model quality or benchmark gains.

## Convert explicit local inputs

Replace `python` below with the isolated interpreter. Invented fixtures are in
`bench/fixtures/raw/`; their answers are not benchmark evidence.

```sh
python -m bench.converters.run_converters --data-root bench/fixtures/raw --out-dir bench/outputs/common --full
python -m bench.eval.evaluate --bench locomo --run bench/fixtures/sample-run.jsonl --variant base --dedup last --qs-dir bench/outputs/common/locomo --out bench/outputs/scores.json
```

Expected invented LoCoMo example: five questions, mean 1.0, zero score flags.
`sample-run.jsonl` is a newly constructed synthetic declaration fixture with
run ID `synthetic-declaration-v1`, not a relabeled historical/model run. Its
hashes match the default fixture conversion, not arbitrary converted data.
`--family` accepts `longmemeval`, `locomo`, `lme_v2`, `locomo_plus`.

| Family | Local input filenames under `--data-root` |
| --- | --- |
| LongMemEval-S | `longmemeval_s.json` |
| LoCoMo-shaped | `locomo10.json` |
| LME-V2 | `lme_v2_questions.jsonl`, `lme_v2_small_haystack.json`, `trajectories.jsonl` |
| Normalized cue adapter | `locomo_plus_normalized.json` |

Data acquisition and licensing are separate. Nothing is downloaded by the CLI.
Roots default to repository-relative `data/` and `data/common/` for convenience.
`MEMORY_BENCH_DATA_ROOT` / `MEMORY_BENCH_COMMON_ROOT` or `--data-root` / `--out-dir`
select explicit alternatives. Use an English-compatible locale for date fields.

**Use a fresh output root.** Every conversion replaces generated `env_*.json`
and `qs_*.jsonl` for the chosen family. `--full` regenerates dev outputs first,
then emits full outputs. A failed conversion can leave old generated files
removed. LoCoMo/LME-V2 full question sets reuse parent environment files.

### New normalized cue boundary

Unlicensed LoCoMo-Plus prompts, parser, duration parsing and stitching are absent.
The adapter accepts a list of records with:

- `qid`, `question`, `evidence_cue`, and optional `category` strings;
- `env`: a caller-normalized common environment with `env_id`, ordered `sessions`
  and `events`, and optional `ground_truth_refs`.

See `bench/fixtures/raw/locomo_plus_normalized.json`. Supply already licensed,
ordered contexts with isolated environment IDs. Raw `time_gap` strings are not
parsed. Raw `locomo_plus.json` is not supported and gives an explicit boundary
error. The package never downloads/imports a third-party adapter. Sampling is a
stable local qid sample, not the old relation/rank stratification.

## Protocol migration: never silently rescore old artifacts

Evaluator identity: **`optimus-memory-evaluation/2.0.0`**.
LoCoMo/cue metric identity: **`optimus-local-scoring/1.0.0`**.

- LoCoMo-shaped F1 uses independently authored token mathematics plus an
  optionally applied licensed NLTK stemmer. Local normalization removes a/an/the,
  **not “and”**. This is not upstream F1 parity. Comma parts and category-3 gold
  truncation are explicit local adapter rules.
- Category labels: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
  5 adversarial. Category-5 alternatives are shuffled **before** letters are
  assigned. The exact option map and abstention label are recorded. There is no
  uniform-correct-`a` rule or old date-hint presentation text. Local choice
  scoring accepts the recorded correct label, its parenthesized form, or the
  exact abstention option text; it does not use old substring heuristics.
- Cue semantic scoring uses a new independent rubric and strict bare JSON
  `{correct: boolean, reason: string}`. There is no Cognitive/partial-score or
  substring-parser parity. Narration about work is not itself an answer.
- MIT LongMemEval template constants and Apache LME-V2 metrics/prompts remain
  under their licenses. They use separate local adapter identities. LongMemEval
  abstention detection and LME-V2 eval-option parsing still have documented
  campaign differences; the whole stack is not upstream-equivalent. LME-V2 is
  text-only, ingests trajectories in file order, caps accessibility trees at
  `LME_V2_AXTREE_CAP` (default 8000), and uses boxed-answer/UNKNOWN handling.

Regenerate **common questions and new model runs** before evaluating the new
local protocols. Evaluating files without the expected question/scorer versions
fails. Do not mutate old scores or claim historical predictions used corrected
choice association. The private old source archive/caches retain their history
outside this repository; no migration or automatic rescore is performed.

## Prepare NEW gold-free question declarations

```sh
python -m bench.prepare_questions --bench locomo --qs-dir COMMON/locomo --out NEW-DRIVER-QUESTIONS.jsonl
```

This offline command takes only explicitly named scoring questions. It emits
exactly `qid`, `env_id`, `question`, `question_protocol`, `scoring_protocol`, and
`question_sha256`. Visible `question` is the exact `presented_question` when
present, otherwise `question`. No `answer`, `eval`, `evidence`, or prediction is
emitted. Existing output paths are refused. This is question preparation, not
a complete native manifest/seal builder or a model runner.

The SHA256 is an **opaque caller declaration** of the full canonical scoring
question, including scoring-only fields. A gold-free driver must not recompute
it from gold. For new strict runs the driver echoes all three declarations and
sets `question_binding_source="caller_declared_scoring_question"`. Protocol
strings must be nonblank, at most 256 Unicode scalar values; the digest must be
64 ASCII hex characters (case is retained by the driver and compared as hex).
The evaluator requires exact known protocols, visible text, qid/environment,
benchmark and hash agreement. Every present selected run row also needs a
nonblank `run_id`, `benchmark`, `variant`, and `env_id`.

Drivers without these fields remain a **legacy input boundary**, not strict
scoring inputs. Use a driver build that supports the declaration contract for
new authorized runs; preserve its separate gold-free validation and frozen
manifest/seal requirements. Never add declarations to old predictions or use
this command to rewrite historical pilot rows. Missing/conflicting declarations
reject strict evaluation before scoring or judge calls.

## Scoring identities and attribution

```sh
python -m bench.eval.evaluate --run RUN.jsonl --bench locomo --variant base --dedup last --qs-dir COMMON/locomo --out SCORES.json
python -m bench.analysis.attribute --runs RUN.jsonl --envs ENVS.jsonl --bench locomo --variant base --dedup last --data-common COMMON --scores SCORES.json --out ATTRIBUTION.json
```

Run `answer_text` is the prediction. `answer` may be echoed gold and is never
used as prediction. Variant filtering is strict; mixed variants require an
explicit selection. `--dedup error|first|last` records duplicate selection;
`last` remains the default for append-style run compatibility.

`memory-bench-scores/1` retains numeric score/flag/aggregate fields and adds
`memory-bench-scoring-provenance/2`: canonical **full question-set hash**, raw
question-file hashes, evaluator/source/protocol/effective budget identity,
explicit requested/effective variant, dedup/selected qids, selected-record
hashes, and `memory-bench-caller-question-declaration/1`. Each result also binds
its question and selected record. Whole-run SHA256 uses exact raw bytes.
Downstream tools must validate these fields, not infer them from qids.

Attribution validates those bindings, the echoed caller declaration, exact
visible text, environment/eval/category metadata, finite scores and score/native
failure flags **before joining**. Snapshot joins require the exact selected
record tuple **`(run_id, benchmark, variant, env_id)`**. Omitting `--variant`
uses the unique selected run variant; it never grants an env-only fallback.
Required snapshots with missing/foreign identity, duplicate exact identity, or
missing `store_snapshot.memory` objects reject attribution. Run `--dedup` does
not resolve snapshot ambiguity. Different run IDs can share an environment ID;
all four identity parts must match each selected row. Explicit empty memory is
valid; absent memory is not invented. Flagged/native-failure rows and declared
`nomem` rows have no snapshot inference.

The success label is **`verified-local-artifact-consistency`**, not independent
verification of actual question presentation, ground truth, model/provider,
activation, or history. A caller can declare an identity; only separate runtime
traces can support presentation claims. Overlap diagnostics stay **noncausal
text heuristics**, not proof of capture, retrieval, injection or reasoning.
Low-level metric functions alone do not perform artifact validation.

`--allow-legacy-unverified` is the only legacy path: old scores with no scoring
provenance are labeled unverified and produce no overlap/failure-stage inference,
even if old rows/snapshots lack declarations and run identity. Contradictory new
bindings cannot be bypassed with this flag. Missing stores are never “nomem.”

The legacy aggregate denominator remains all selected questions, with flagged
rows at zero. Missing/error flags are not judged failures. Wilson intervals use
`score >= 0.5`, not continuous-F1 uncertainty. Native stop/timeout diagnostics
stay separate. Attribution excludes native failures from overlap inference.

## Explicit DGX-only network access

All benchmark driver/judge LLMs must be **DGX GLM-5.3; no Jev**. No driver or
agent launcher is included here. No import or default action contacts a model.
An offline cache miss becomes a flagged `eval_error`, not judge evidence.

Select a private config with `--models-json FILE` or `MEMORY_BENCH_MODELS_JSON`.
It alone grants no network access; authorized live calls also need
`--allow-network`. Config structure:

```json
{"providers":{"dgx-glm53":{
  "baseUrl":"https://YOUR-DGX-ENDPOINT/v1",
  "apiKey":{"env":"MEMORY_BENCH_API_KEY"},
  "headers":{"X-Pomerium-Authorization":{"env":"MEMORY_BENCH_SESSION_TOKEN"}}
}}}
```

Explicit literal secrets are supported but env references are preferred. No
credential command, profile/netrc discovery, proxy autodiscovery, or HTTP
redirect is used. HTTPS is required. The provider/model strings are policy
constraints, **not authentication of a caller-selected remote server**.

`--judge-cache DIR` / `MEMORY_BENCH_JUDGE_CACHE` opts into local caching; otherwise
no cache is written. Cache hashes now include the evaluator-version namespace,
actual messages, judge, model, temperature and token budget. **Old campaign
cache entries are not reused.** LongMemEval budget is 2048; LME-V2 is 4096;
local cue budget is 512 or explicit `LP_JUDGE_MAX_TOKENS`. Concurrent misses can
still cause duplicate requests. No provider health/activation test was run.

Outputs, snapshots and caches are **private content-bearing evidence**, not
universal secret/path scrubbers. Known config credentials are redacted from
judge outputs and diagnostics omit private errors, but source/model/gold text
may remain. Public redactions require separate derivative artifacts/hashes.
Input-alias checks and atomic output writes protect named local inputs.
