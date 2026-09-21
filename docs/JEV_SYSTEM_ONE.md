# Bounded Jev System One integration

Jev is optional. Installing or configuring a credential does not enable decisions,
filtering, or compaction. No Jev decision can execute a tool, generate tool
arguments, change permissions, select a primary/child model, spawn children,
raise retry limits, or stop an agent.

Question catalogs, evaluator thresholds, and suppression floors are documented
in [JEV_QUESTIONS_AND_THRESHOLDS.md](JEV_QUESTIONS_AND_THRESHOLDS.md).

## Independent controls

| Command | Compare | Active |
| --- | --- | --- |
| `/jev off` | Off | Off |
| `/jev compare` or `/jev on` | On | Off |
| `/jev active` | Off | On |
| `/jev compare-active` | On | On |

`/jev compact on`, `/jev compact off`, and `/jev compact status` control a
separate per-session setting. Compaction can run with decisions Off. Changing
one axis does not change the other. New feature gates and compaction are off by
default. Existing tool-requirement and complexity effects retain their defaults.

`/jev feature <name> on|off` changes a feature policy. An enabled Active feature
still needs Active mode and an accepted, current answer. Compare never changes
the request. Result sufficiency, loop control, retry classification, verification,
trace assessment, and model/subagent routing remain advisory.

## Requested Jev model and catalog

`/jev model status`, `/jev model set <id>`, and `/jev model reset` are local
settings operations. They make no catalog or decision request. `set` preserves
the exact safe identifier; whitespace, controls, bidi/invisible formatting and
credential-shaped values are rejected without echoing the supplied value.
`reset` selects the documented native default, `jev-latest`. Model settings do
not enable Jev or change the full-jev overlay.

Every decision lane captures mode, requested model, feature/compaction policy
and the durable settings identity from one authoritative snapshot before any
await. The captured model is sent on that request. A later settings write,
including model A-to-B-to-A, makes the held result stale and prevents its effect.

`/jev models` is the only catalog command. It performs at most one bounded,
read-only catalog fetch when a credential is available. It never auto-selects a
model and never writes settings. With no credential it reports unavailability
without a fetch. Catalog output is bounded and hostile identifiers or prose are
not repeated.

## Integration boundaries

- `pi-jev` owns typed questions, acceptance policy, validation, transport,
  redaction, bounded state, and records. It does not own agent runtime handles.
- `jev_bridge.rs` binds the shared client to a session and lifecycle boundary.
  Combined mode retains one response and produces separate comparison and
  Active records. Baselines are captured before any request mutation.
- `jev_active.rs` changes only the advertised provider request. The original
  tool registry and tool execution path remain authoritative. Optional pruning
  only considers configured optional tools; required, unknown, and forced tools
  remain available. The native `ipython` tool is never an optional pruning
  candidate; extension tools need an explicit optional-tool allowlist.
  The existing `tool_requirement == none` effect is separate
  and retains PR #56's full catalog withdrawal behavior.
  Complexity supports both Chat `reasoning_effort` and Responses
  `reasoning.effort`, preserving summary fields and the model registry's limits.
- Retrieval filtering works on temporary, identified candidates. Memory filtering
  only considers eligible retrieved automatic memories, not user/manual entries,
  session state, prompt notes, or entries with unknown provenance. Context
  filtering considers successful paired historical read/search results, excluding
  the newest six messages, the current user-request suffix, and protected content.
  It replaces eligible output with a marker only when that reduces its size.
  Neither filter edits persistent memory or the saved transcript. Records include
  candidate counts and local byte-based token estimates, not billed token counts.
- Native compaction is a **request-local context projection**. The durable
  session transcript is not rewritten. Disabling it restores normal context
  construction. Existing automatic/overflow/manual summary and provider-native
  checkpoint compaction remain unchanged; `/compact` is not replaced by the
  Claude Code plugin.

## Compaction design

The algorithm is inspired by
[`tamaratran/fast-jev-compaction`](https://github.com/tamaratran/fast-jev-compaction),
not its Claude-specific hook or message types. Optimus pairs native calls and
results, preserves pinned/recent and authoritative content, and asks separate
keep-call and keep-result questions. Retained content stays verbatim. A call can
be retained with a bounded result head; deleting a call deletes its paired result.
Invalid pairs, unsupported rich/signed content, or missing/ambiguous decisions
cannot produce partial context edits.

The configured state/request budgets, preserved recent-message count, keep
threshold, result-head length, and minimum reduction ratio are bounded and
validated. Large raw tool results and tool arguments are not sent to Jev. The
service sees bounded, redacted state and tool/result metadata. Pattern-based
redaction is not a guarantee that all private task text is safe to disclose.

The native policy is deliberately narrower than the reference. Known successful
read/search results can be eligible. Unknown tools, shell commands, edits, errors,
images, instruction files, and opaque provider checkpoints stay pinned. Native
Python calls remain present; only a positively recognized source-read pattern
can have its old text output truncated. Arbitrary Python output stays intact.
The adapter evaluates at most eight old pairs per request. Hard byte caps apply
in addition to the configurable token estimates.

A provider checkpoint pins the prefix through that checkpoint. Complete eligible
tool pairs after it can still be compacted; opaque items are never sent to Jev or
rewritten. A pair crossing the boundary remains intact. The minimum reduction
threshold applies to the eligible suffix, excluding opaque checkpoint bytes.
Records distinguish the total serialized reduction from `eligible_reduction_ratio`
and report `protected_messages`. No eligible old output means no network request;
enabling compaction does not force deletion of required or recent context.

A missing credential, timeout, cancellation, breaker, invalid response, stale
settings, state-fitting failure, or insufficient reduction keeps the original
context. Built-in compaction still operates under its existing policy. A failed
optional optimization must not block the main provider request.

### Search/evidence composition

Line-find and citation checks are advisory request-local annotations. When both
are enabled, they may assess the same otherwise-eligible native source-read
result. Eligibility and provenance stay anchored to the immutable original text
block and its captured fingerprint; one annotation cannot turn arbitrary
pre-existing multi-block, image, error, mutation or unsigned content into an
eligible source. Each accepted annotation is a separate bounded block on the
provider-request copy. Source code and saved history remain unchanged.

Native compaction treats recognized line-find, retrieval-safety and citation
annotations as protected. It does not erase those annotations later in the same
provider-context pipeline. Missing, malformed, stale, cancelled, oversized or
deadline-expired assessments fail open to the unmodified input. This does not
create a source artifact, perform another source read, verify a claim, or add
new drop or execution authority.

## Experimental code-search relevance

Search retrieves; Jev evaluates retrieved candidates. The Rust host has a separate
`code_search_relevance` category and feature, independent of historical
`context_relevance`. Both search switches default to off:

```text
/jev feature code_search_relevance on
/jev feature code_search_filtering off
/jev compare-active
```

This scores without changing search context. After benchmarking, enable
`/jev feature code_search_filtering on` in Active or Compare + Active to allow
high-confidence pruning. Compare always preserves every result, even when the
filtering switch is on. No search-planning or tool-execution authority is granted.

In the Python REPL, capture deterministic search output using the normal `bash()`
interface, then present candidates in a cell with no other output:

```python
from rlm.code_search import from_ripgrep, present
candidates = from_ripgrep(result.output)  # result from rg --json
present(candidates)
```

AST, symbol, filename, reference and test searches can pass their own dictionaries
to `present`: `kind` (`file`, `symbol`, `grep`, `reference`, `test`), `path`, optional
`line`, `snippet`, and `mandatory`. Keep full results in a named Python variable.
The helper validates the envelope and performs no search or network request.

At the provider boundary the host scores at most 64 optional candidates in batches
of at most eight, with at most two concurrent Active requests and a 2.5-second
overall deadline. Unscored candidates, instruction files and mandatory evidence
remain intact. Missing, stale, malformed or uncertain decisions retain the
affected batch. Removing every candidate is refused. Original result order is
preserved; this version prunes, it does not impose a fixed top-K cutoff.
Only complete successful `ipython` presentations are eligible; arbitrary stdout,
mixed output, errors and mutations stay intact. Durable transcripts and Python
variables are unchanged. Turning the switches off restores the original context.

Records use stage `code_search` and category `code_search_relevance`, with ordinal
candidate IDs, probabilities and candidate-only size estimates. They do not store
paths or source snippets. Compare records are suitable for collecting recommendations;
they are not task-success labels. No production quality or cost gain is established.

## Records and measurement

`jev.compare/1` remains observational (`applied: false`). `jev.active/1` records
recommendation, policy acceptance, actual effects, refusal, and fallback as
separate facts. Combined rows share a request ID. Count network requests by
request ID, not by record line or question. An accepted decision with no eligible
field to change is not an applied change.
Reasoning telemetry reports the actual provider field, including nested Responses
effort. Single-tool catalogs skip the tool-selection comparison. Tool suitability
requires task evidence; end-of-turn recommendations include the task and explicitly
distinguish loop termination from successful completion.

Pre-mutation request metadata is not proof of what an unmodified model would
have done. A missing semantic baseline is noncomparable. A successful tool call
is not proof that tests passed or that the task was completed. Unknown model
cost, task quality, and avoided model calls must not be reported as zero savings.

Summarize local records without printing prompt/result content:

```sh
python3 scripts/summarize_jev_records.py /path/to/agent-dir/jev/records.jsonl
```

Enable `trace_observer` explicitly to collect bounded local run measurements.
With decisions Off it records metadata only, without Jev calls. `jev/runs.jsonl`
contains observed turns, assistant completions, available usage, and elapsed time
between agent-start and agent-end events. Logical assistant completions are not
raw provider transport attempts. Missing usage stays unknown. Configuration
changes and incomplete observation make a run ineligible for matched benchmarks.
Read this separate file with `--runs /path/to/agent-dir/jev/runs.jsonl`.

The reader is bounded and reports malformed or truncated input. Historical rows
without a compaction-setting stamp are grouped as `unknown`, not guessed.

### Matched evaluation protocol

1. Use eight isolated agent directories: four decision modes times compaction
   Off/On. Never reuse a live production session as a benchmark fixture.
2. Hold task, repository revision, primary provider/model, effort, tool allowlist,
   memory snapshot, budgets, feature policy, and success checks constant.
3. Repeat each task in each configuration. Randomize order to reduce provider
   latency/cache bias. Record aborts and failed runs; do not discard them.
4. Measure Jev requests/questions/latency, accepted/applied/refused decisions,
   fallback reasons, tool/context reduction, primary model calls/tokens,
   wall-clock duration, test outcomes, and independently scored task quality.
5. Compare distributions and failure rates. Context reduction alone is not an
   improvement if retries, task failures, total cost, or elapsed time increase.

The regression suite uses mock transports and synthetic messages. It establishes
safety and behavior, **not production quality, latency, or cost savings**. No live
paid benchmark is run automatically.

### Early decisions

The request catalog and final context are known reliably at their native
boundaries. Starting a decision earlier is only valid if the same state and
policy still apply at consumption. This change reuses answers at shared
boundaries; it does not speculate across changed request/context snapshots.
Latency records provide the baseline for a later fingerprint-bound prefetch
experiment. No latency improvement from speculative execution is claimed.

## CONTROL and terminal status

CONTROL decisions require the full-jev profile and an Active-capable mode. The
host owns fixed corrective-feedback text, per-session durable budgets, epoch
correlation and every effect. A result can request a queued follow-up only after
a fresh authoritative apply check. It cannot execute tools, refill a budget,
widen retries, terminate a tool mid-call, or mark a goal complete.

Verification states are honest: `unknown`, `not_applicable`, `unverified`,
`verified`, or `failed`. Verified/failed require explicit correlated evidence;
a successful tool name or model answer is not enough. Existing terminal
annotation, pause, escalation/attention and verification state are exposed as
bounded worker status. These fields report the resolver outcome and never
fabricate completion or execution authority.

## Daemon and live-service limits

Daemon Jev settings and status are capability- and schema-gated. Older peers
keep the legacy command behavior. Worker-local telemetry is returned by the
session worker; a supervisor must not fabricate an empty healthy snapshot.
Catalog selection remains local except for the explicit `/jev models` fetch.

The checked regression suite uses injected transports, local faux providers and
mock Jev responses. It does not establish compatibility, availability, latency
or quality for the live Jev service. Live service health and compatibility are
unverified in this candidate. No live inference or catalog probe, installation,
restart, activation or cutover is part of these instructions or receipts.

## Full-jev global overlay (opt-in; ROOT-CONTRACT v1)

`/jev full-jev on` installs a persisted named overlay profile for this agentDir;
`/jev full-jev off` removes it; `/jev full-jev status` reports truth. The overlay
is NOT a new `JevMode` and it never edits the built-in defaults, the saved global
fields or the per-session map: while installed it is resolved ABOVE every saved
session, global and inherited override (mode Compare + Active, all feature gates
on including the new `code_search_reranking` and `line_find`, compaction on).
Removing it restores the exact prior resolution because nothing was overwritten.
Off does NOT delete the persisted block text: it is retained as an INACTIVE
tombstone (enabled: false) carrying a durable revision, so the activation
history survives restarts. An inactive block never masks anything and the
resolution equals the exact pre-overlay result; the retained revision only
prevents ABA reuse — a later `/jev full-jev on` reactivates with a NEW
revision so stale stamps from the earlier activation (cheap caches, in-flight
decisions) invalidate exactly like a key rotation.

- Emergency exit: `/jev off` while the overlay is active atomically removes the
  overlay AND sets the issuing session's decisions Off and compaction false;
  other sessions return to their saved values.
- While the overlay is active, mode/feature/compaction/default change requests
  are refused with an explicit no-change message pointing to `/jev full-jev off`;
  status/key operations still work. No success is reported for a change the
  overlay would hide.
- Staleness: every effect re-checks current settings, credentials and the
  overlay revision (`full_jev_stamp`) at apply time — a toggle invalidates
  in-flight work exactly like a key rotation; stale pre-toggle decisions cannot
  apply. Same-process refresh is immediate; a different process picks the change
  up at its next settings read, bounded by the 250 ms settings cache TTL.
- Inheritance: children inherit the overlay-EXCLUDED baseline only; overlay
  values are never materialized into sessions or children, which resolve the
  live overlay like every other session.
- Versioning: the overlay is understood by THIS source build. An already-running
  older executable does not gain these features because settings JSON changed;
  upgrade requires the new binary (mixed-version writers are outside the
  supported live-reload contract).
- Credentials: absent key means the footer reports unavailable and every
  decision fails closed until `/jev key`; enabled configuration never fabricates
  answers.

## Reference installation

The reference npm library can be installed in an isolated checkout for algorithm
inspection and offline tests. Its Claude Code function hook is not an Optimus
plugin and must not be installed into Optimus's extension registry. Native
compaction uses the Rust implementation above. Do not activate Claude hooks or
restart a running Optimus daemon to validate this feature.
