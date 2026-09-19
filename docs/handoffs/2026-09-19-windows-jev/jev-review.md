# Codex independent Jev review — 2026-09-19

Status: FROZEN UNFINISHED at explicit user stop (2026-09-19). DRAFT handoff only; NOT cutover-ready. Frozen team source is preserved. Parent authorised repairs only in codex-wrapup/source; tests/builds remain parent-owned. The findings below describe the frozen candidate; the repair/gate section records subsequent combined-tree changes. Findings are source-proven, not runtime-reproduced by this reviewer.

## Blocking findings in the frozen candidate

1. **JEV-C01 — First-use Compare is inert until extension runner recreation.** `crates/pi-coding-agent/src/core/jev_bridge.rs:93` conditionally registers once using existing settings; `/jev compare` only writes a file and publishes UI, never registers the missing observer. Default-Off startup therefore cannot start observing in the same chat. Repair: register a dormant read-only adapter unconditionally; retain cheap Off event gate and lazy client/tasks. Test: same session Off -> Compare -> Off -> Compare, with mock transport and unchanged primary requests.
2. **JEV-C02 — Cancellation is not shared and can be lost.** `crates/pi-jev/src/scheduler.rs:184` clones the cancellation AtomicBool by value. `cancel_session` cancels the map's clone while workers consult a different flag; Notify is not a persistent state. Cancellation during the min-interval select is consumed and execution then falls through into the real request. Repair: one Arc-owned cancellation state, check after waits, close the enqueue/cancel registration race, and gate dispatch and completion against current mode/credential generation. Tests: cancel before waiter, during rate wait, during request; no dispatch after Off; Compare can rearm.
3. **JEV-C03 — Observer replacement leaks workers and old credentials.** `scheduler.rs:154,242` workers own Shared, Shared owns the queue sender, and no Drop calls shutdown. `jev_bridge.rs:296` replaces observers without closing old schedulers. Worker tasks remain waiting forever and retain old SystemOne/credentials. Repair: explicit scheduler Drop/shutdown, drain cancellation terminals, stop replaced observers. Tests: weak handle releases after drop/rotation; every queued request settles once.
4. **JEV-C04 — Credential rotation watches the wrong file.** `jev_bridge.rs:311` watches `<agent>/jev/jev-credential.json`; `credential.rs:296` actually writes `<agent>/jev/typesafe.jev-credential.json`. After first observer construction, save/remove/replace never changes the cheap stamp. Env rotation only checks presence booleans. Repair: use actual named path and content-version stamp; invalidate stale in-flight results, without decrypting at every event. Tests: synthetic envelope create/replace/remove changes stamp; no raw values enter it.
5. **JEV-C05 — Compare clones the full saved transcript on every event.** `jev_bridge.rs:591` calls `get_branch().len()`; `agent_session/runtime_members.rs:74` implements it by `get_entries()` plus transforming the whole vector. This directly contradicts bounded observation and reintroduces long-chat overhead. Repair: constant-time optional entry count; never clone history for metadata; keep unknown null. Test: a ReadonlySessionManager whose get_branch panics still permits all bridge observations.
6. **JEV-C06 — Status/footer are not connected to live pipeline telemetry.** `jev_host.rs:198,222` always supply default pipeline / `JevStatusReport::local_only`; `jev_menu.rs:262` explicitly says pipeline is unwired. Counters are always zero, last success never, and footer is published only on /jev actions, not attached-chat startup or request transitions. Repair owner: parent/UI integration. Add worker-owned bounded status snapshot and capability-gated retrieval; no per-render network polling; unavailable != zero. This is a required user feature, not cosmetic.
7. **JEV-C07 — Credential-entry cancellation claims exceed implementation.** `jev_host.rs:389` async validation body contains only synchronous DPAPI/store I/O and no await. `select!` cannot pre-empt its first poll, so Escape cannot cancel slow persistence; the code calls this asynchronous validation although no TypeSafe validation is done. The UI honestly discloses no live validation after storage. Repair owner: parent/UI; move preparatory work off executor and guard any final persistence/ack by cancellation generation; never claim nothing stored after a commit raced cancellation.

## Coverage and safety assessment

- No Jev-originated primary model/provider/effort or child lifecycle/task/message authority found in inspected paths. Extension has no tools/commands/flags; handlers return None; Active is refused; records hard-write applied=false. This is positive source evidence, not a full runtime acceptance pass.
- All eleven evaluators exist, but category 6 has no allowlist and category 8 no memory input in the actual bridge. They always skip. Category 7 receives message count and current prompt rather than actual context evidence, so a context-relevance score is not meaningful evidence of safe filtering. Do not claim all eleven features operational from synthetic evaluator tests.
- API endpoint/auth and current documented Noul/Choice/Score shapes checked against https://docs.typesafe.ai/api and https://docs.typesafe.ai/primitives (2026-09-19). No live API call made. The current Score type includes required instructions; frozen bridge's Score pattern omits the new field and requires `..` for compilation.
- HTTP client uses no redirects and never logs provider error bodies. DPAPI uses current-user scope with no plaintext fallback. Saved-secret rotation and task lifetime still require repairs above.
- Additional unaccepted issues: client Retry-After is clamped shorter than server request; client in-flight counter increments/decrements without cancellation RAII; settings read-modify-write uses fixed shared temp filename and no cross-process transaction; these need parent triage rather than silent readiness.

## Execution boundary

No production/session/gateway/credential action; no model API call; no test/build execution by reviewer. Use parent-recorded results for final readiness. No persistent memory facts were used.

## Combined-tree repairs and integration gate

Implemented in `codex-wrapup/source`, not the frozen source:

- JEV-C01: unconditional dormant adapter, lazy Compare construction, same-session Off -> Compare -> Off -> Compare end-to-end regression.
- JEV-C02/C03: shared persistent cancellation state; dispatch/completion mode and credential-generation gates; cancellation registration race closure; scheduler Drop/shutdown draining; old observer shutdown on rotation. Mock-only lifecycle regressions cover cancellation before/during waits, stale result suppression, rearming, and terminal settlement/release.
- JEV-C04: actual named credential-envelope path, nanosecond metadata generation and environment fingerprints. Rewriting the same key now rebuilds the captured generation gate instead of silently black-holing all later requests. An inert synthetic-envelope/mock-transport regression exercises both first generation and rewrite without DPAPI or external calls.
- JEV-C05: bridge uses the core reviewer's optional `get_entry_count()` interface, not cloned transcript content. Turn correlation uses actual observed turn indices.
- JEV-C06: bounded worker-local per-session status registry: real queued/in-flight/success/failure/drop/last-success/latency/model values, `applied=false`; absent observations remain unknown. Model metadata is sanitized, control-stripped and capped. Observer publishes footer status at semantic boundaries and terminal completion; UI /jev status awaits an optional capability-gated worker snapshot, with a bounded 2-second timeout and no render polling. Parent/adapter reviewer own daemon route, initial-attach publication, and optional connection getter.
- JEV-C07: parent implemented cancellable off-executor credential preparation with an atomic commit gate and truthful cancellation-after-commit messaging. UI input is length-bounded, masks all text, ignores resubmit/paste during save, and explicitly describes secure storage rather than an unperformed live API validation.
- Compare disclosure is now displayed before mode selection and on enable. No new primary-model/subagent/agent-loop mutation authority was added.
- Repaired frozen test compile failures, restored missing Off test annotation, and repaired impossible source-test assertions that classified all dotted String/Vec calls as session mutations. Boundary assertions continue to permit only identity/telemetry reads and reject forbidden model/child control calls. Late-result regression now permits terminal cancellation records while requiring no additional accepted result or transport call.

Parent gate evidence available at this update: `logs/jev-suite-2.log` shows pi-jev unit 9/9 and client 74/74 passed, comparison 26/28 passed. Two comparison failures were stale expectations (zero cancellation terminal records; wrong redaction-marker spelling) and have been corrected for rerun. Newest source/UI/real-session mock tests still require parent acceptance. This is NOT a final passing gate or cutover authorization.

Known boundaries still requiring honest handoff: categories 6/8 remain explicit skips for missing safe input; category 7 lacks actual context evidence. No real API/key validation has been performed. A native UI Off write is observed by worker dispatch/completion generation gates but does not itself synchronously cancel an already-dispatched worker future; it is bounded by the existing request deadline unless another worker event cancels it. Parent is triaging immediate mode-control RPC and the client/settings issues listed above.

## Explicit stop: exact unfinished state

The user stopped expansion because of token cost. All reviewer-owned source is left at a syntactically complete write boundary. No further source edits, audits, tests or builds were performed after that stop. Root plans a DRAFT PR only, with no cutover.

Reviewer-owned changed files (relative to combined source):

- `crates/pi-jev/src/scheduler.rs`
- `crates/pi-jev/src/hooks.rs`
- `crates/pi-jev/tests/comparison_tests.rs`
- `crates/pi-coding-agent/src/core/jev_bridge.rs`
- `crates/pi-coding-agent/src/modes/interactive/jev_menu.rs`
- `crates/pi-coding-agent/src/modes/interactive/jev_host.rs` (includes parent credential-cancellation edits)
- `crates/pi-coding-agent/src/modes/interactive/jev_menu_component.rs`
- `crates/pi-coding-agent/tests/jev_compare_tests.rs`
- `crates/pi-coding-agent/tests/jev_ui_tests.rs`

Important OPEN safety/performance issue discovered immediately before stop: `bridge_event` still serializes the entire tool input before taking 400 characters, and forwards raw argument excerpts that can include unrelated API keys/passwords. `summarize_agent_end` still clones/joins all assistant text blocks before truncation. Task/result excerpts are bounded in final payload but not yet sanitized before construction. Proposed omission of raw tool args plus bounded/redacted task/result construction WAS NOT IMPLEMENTED; no redaction regression was added. Do not enable real Compare on sensitive tasks until this is repaired and outgoing-request tests prove it.

Known UNVERIFIED test details: same-key generation regression creates an inert synthetic envelope in its isolated temp agent directory; it currently leaves it until TempDir cleanup, while the final existing e2e owned-files assertion allows only settings/records and will need to account for or remove that test fixture. UI source-string guards may still contain brittle expectations; the latest UI test repairs have not been executed. Most recent observed parent log remains `jev-suite-2.log` (9 unit + 74 client pass, 26 comparison pass, 2 stale-assertion failures since patched but not rerun by reviewer). New generation-rewrite, worker-status UI, and actual session parity changes are not accepted as passing.

Suggested next owner commands (NOT executed by reviewer; use root's isolated environment/target and offline locked toolchain):

```text
cargo test --offline --locked -p pi-jev
cargo test --offline --locked -p pi-coding-agent --test jev_compare_tests
cargo test --offline --locked -p pi-coding-agent --test jev_ui_tests
cargo test --offline --locked -p pi-coding-agent no_subagent_control_tests
```

Before those gates, finish the outgoing-data minimization/redaction repair and regressions; reconcile the inert e2e envelope assertion; ensure parent/adapter getter and initial-attach footer patches compile together. Check latest root logs for any results newer than this reviewer snapshot. No real Jev call, user credential read/write, production restart, cutover, or git operation was performed by this reviewer.
