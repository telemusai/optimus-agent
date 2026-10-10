# Memory capture grounding and diagnostics

## Internal observation time

Memory evidence can carry the source message's observed timestamp in Unix
milliseconds. The extractor sees a validated UTC timestamp label beside the
unchanged source text. This is **observed time**, not the event's valid time:
a message observed today can describe an event from another date. Transport
does not resolve relative dates or certify the asserted fact.

Missing, zero, nonfinite or out-of-range anchors are not emitted. Import uses the
nested source message timestamp, never the session row's append clock or the
current wall clock. Existing saved import jobs are read without timestamp
backfills. Evidence splitting and reload retain valid internal anchors.

`MemorySource` and saved source IDs/digests retain their existing shapes.
`import_chunk` removes timestamps from a cloned response only; stored jobs and
extractor evidence keep their anchors. No daemon command, capability or schema
revision is changed.

## Opt-in benchmark capture

The Rust example is `crates/pi-coding-agent/examples/memory_bench.rs`.
New capture input uses
`captureProtocol: "optimus-memory-capture/temporal-authority/1.0.0"` and a fresh
output directory. Valid source `event.ts` values are the only timestamp input.
Naive ISO values use an explicitly recorded UTC convention. A converter's
source session-date anchor is not a measured per-message timestamp.

LoCoMo-shaped external human speaker labels can be imported as user evidence.
Other roles remain unchanged. LME-V2 revision 1 requires an explicit positive
accessibility-tree cap. Its browser observations are `toolResult`; task metadata,
actions and thoughts remain assistant-origin and undated. They are not promoted
to user facts. Legacy unversioned capture remains separately labeled and writes
no invented timeline. Historical frozen replay is not relabeled or repaired.

The driver pins a scratch control profile and checks explicit NoJev isolation
before runtime work. Provider configuration must be supplied explicitly; there
is no installed-profile fallback. Frozen replay checks its project-only corpus,
manifest, fixture and settings, disables learning and ingestion, and permits one
gold-free question per fresh attempt. Main-answer thinking settings are explicit
benchmark controls, not changes to production helper reasoning settings.

New question declarations bind exact visible text to caller-supplied question
and scoring protocols plus an opaque full-scoring-question hash. The driver
echoes them without deriving them from gold. The Python evaluator rejects
conflicting declarations before scoring. Attribution joins snapshots by exact
`(run_id, benchmark, variant, env_id)`. These checks establish local artifact
consistency, not truthful gold, actual model presentation or provider identity.

The [portable bench README](../../../bench/README.md) describes offline setup,
synthetic fixtures and strict evaluation. It retains the identity
`optimus-local-scoring/1.0.0`; it does not claim official benchmark parity.
MIT/Apache license texts and third-party notices remain part of that package.
Real datasets, model configurations, runs, caches and snapshots are not shipped.

## Helper diagnostics and unchanged controls

Optional recall-helper metadata adds normalized stop reason, emitted text
character count, monotonic dispatch latency, observed cancellation/error flags,
and provider-observed usage counters. It records no prompt, completion,
reasoning text, raw error, model identity or credential. Missing observations
remain null; explicit provider zero remains zero. Totals and reasoning tokens
are not inferred or summed. Cache reuse does not repeat usage or latency.

These are backward-compatible opaque custom-entry fields, not new daemon wire
fields. An attempted helper means a stream dispatch, not confirmed completion
or final remote usage. Diagnostics do not form a complete cost ledger.

Token-cap experiments were **not adopted**. Query distillation remains capped
at 64 output tokens and reranking at 512. Helpers remain off by default. This
change does not add retries, increase helper budgets, remove dependencies or
change provider controls.
