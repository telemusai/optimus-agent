# Codex core repair review

Status: STOPPED at the user's cost/scope freeze. Candidate corrections listed below are implemented but unverified; root owns compilation, tests and acceptance. Do not cut over this candidate based on this report. No test, build, live provider request, daemon action, production change, restart or Git mutation was performed by this reviewer. The open automatic-result delivery defect below has NOT been patched; there are no partial delivery-marker/schema changes to finish or revert.

## Sources and scope

- Frozen Flash candidate: `C:/Users/openclawuser/optimus-flash-repairs-20260919/source`, baseline `cad234a17589f5838407421fb435bd74d5ceda76`; not modified.
- Corrections: `C:/Users/openclawuser/optimus-flash-repairs-20260919/codex-wrapup/source` combined checkout only.
- Read the repair brief, handoff, compaction wrap-up requirement, AGENTS instructions and Flash review notes. Inspected actual diffs and connected code for compaction failure/backoff, retry correlation/terminal settlement, transport no-replay rules, and bounded lifecycle persistence. Full-file reading of agent_session.rs is complete, including rereading truncated output gaps through EOF. This is not an exhaustive review of every Rust subsystem and reading does not establish passing behaviour.

## Concrete defects found in the Flash candidate

### CR-CORE-01 — deferred orphan persistence can lose the original recovery journal (blocker)

`modes/daemon/supervisor_maintenance.rs::record_deferred_orphans` originally swallowed an unreadable prior sidecar and write failures, returned `()`, and silently retained only the latest 64 rows. Its caller then cleared the original orphan journal and could retire the stop descriptor. A failed sidecar write therefore lost durable process-ownership evidence even though recovery looked complete; truncation also discarded unresolved older records.

Smallest correction: distinguish missing from unreadable, return `Result`, persist the complete unresolved set durably/atomically, and only clear/retire the original journal after successful persistence. Do not silently truncate unresolved evidence. Sent to root immediately and assigned to the persistence reviewer. This reviewer has not independently executed its regression tests.

Required proof: make the deferred-sidecar target unwritable (a directory at the exact path is a deterministic fixture); assert original journal/descriptor remain. Preserve more than 64 distinct unresolved entries. Retry after the obstruction is removed and show each original is retained exactly once.

### CR-CORE-02 — B6 retry grouping still terminalized the first failure prematurely (medium)

Initial `AgentLoopPerformanceMetrics::new` left host ownership false. The core emitted the logical-request terminal before session retry policy ran. The session then installed a retry correlation whose shared settlement was already closed; the new stale-group guard correctly refused that closed settlement, but minted a new group with ordinal 1. Thus failure → successful retry → later prompt was incorrectly three single-attempt groups. The Flash regression accepted this because it asserted at least two terminals, rather than exactly two and the exact attempt grouping.

Codex corrections:

- `agent_session/agent_handle.rs`: claim error settlement ownership before the first real prompt/continue.
- `pi-agent-core/agent_loop.rs`: defer only failed terminal requests to the host; successful and cancelled terminal attempts settle locally with their own timestamps and shared group count. This covers later tool turns as well as the first request.
- `agent_session.rs::process_agent_event`: explicitly settle a non-retried failure before compaction; settle deferred terminals on persistence-failure/cancelled-dispatch exits. Shared settlement remains idempotent.
- Existing retry eligibility/delay/no-partial-replay policy is unchanged.

The strengthened `transport_retry_tests` require one shared group with ordinals `[1,2]`, one successful terminal with attempt_count=2, and a fresh later group `[1]` with count=1. Existing uncertain-send/partial-text/tool-proposal tests now also assert exact metric settlement. Added disabled-retry, exhausted-retry and cancellation-during-backoff terminal tests. Root must run these before acceptance.

### CR-CORE-03 — a fresh logical request inherited stale start time (low)

`agent_loop.rs` selected configured start time before deciding that the configured settlement was already closed. A later request got a fresh ID/ordinal but an old elapsed-time origin. Moved start-time selection after the live-group decision. Strengthened `a_settled_host_group_is_not_reused_by_the_next_request` to assert the current deterministic recorder time, not the previous group's zero start.

### CR-CORE-04 — production refinement apply needed the checked load/save baseline (blocker, coordinated)

Persistence reviewer identified that a failed read could appear as an empty state, with a later healthy write overwriting real content. The production `apply_refine_inner` path now calls `load_harness_state_details`, refuses Unreadable immediately, applies only to a clone of that baseline, and calls `save_harness_state_checked` with the original read generation. The persistence reviewer owns the checked storage implementation and cross-writer locking tests. The legacy unchecked helper import remains test-only in this module.

### CR-CORE-05 — automatic child-result delivery acknowledges failed sends; stale task snapshots can duplicate an explicit reply (OPEN blocker)

Source-confirmed in the frozen candidate and still present in the combined candidate: `crates/pi-coding-agent/src/core/agent_session.rs::deliver_pending_rlm_results_once` (combined approximately lines 5637–5748) clones the ledger before awaiting registered explicit reply futures, ignores the result of `controller.send_agent_message(...).await`, and then marks the live task replied and increments parent reply count unconditionally. A failed send therefore removes the retained result from future automatic-delivery eligibility without a valid receipt. Conversely, an explicit reply completed during the await can mark the live task replied while the old cloned task still says false, permitting a duplicate automatic result. These are source-supported risks; this reviewer did not reproduce production message loss.

Do NOT repair this by blindly retrying every `Err`: native-supervisor forwarding explicitly treats a request-worker error after submission as uncertain acknowledgement. The remote receiver may already have accepted the message, and this API has no caller-supplied idempotency key. Such a retry could duplicate delivery.

Proposed bounded repair, NOT implemented: re-read live task state after explicit futures settle; claim each pending automatic delivery once; durably store an additive per-task uncertain/in-flight marker before submitting; mark replied only for a valid queued/delivered receipt. Clear the marker only for demonstrably pre-admission rejection (for example the exact paused/rate-limit routes), retaining result and replied=false for a later retry. Preserve unknown errors as uncertain, visible and not automatically replayed across reload. A checked ledger append must succeed before sending; existing `persist_rlm_continuation_state` ignores append errors and is not sufficient for this new intent boundary. Explicit successful replies should reconcile the marker. This design still needs implementation review; it is not a completed fix.

Required regression tests, NOT authored or run: known failed-send rejection then successful retry produces one acknowledged result; repeat successful delivery pass sends nothing; ambiguous error/invalid receipt keeps result and prevents automatic replay, including reload; an explicit reply completing during the wait suppresses automatic duplicate; concurrent automatic passes cannot both claim the same task; failed intent persistence performs no send. Likely files: agent_session.rs, core/rlm_continuation.rs and the legacy ledger fixture's additive struct literal. `rlm_continuation.rs` was fully read but remains unmodified; `legacy_rlm_continuation.rs` and agent_messages.rs require complete reading before editing.

## Compaction assessment

No additional source blocker was confirmed in the bounded compaction paths inspected:

- Threshold cooldown is consulted by the actual should-stop-after-turn decision, the regular threshold check and the no-assistant pre-prompt route, so the loop need not stop for compaction that is currently disallowed.
- Cooldown is exponential (5s through a 120s ceiling), not a permanent disable. Explicit requested/manual and overflow recovery remain outside the threshold gate.
- Unknown/incomplete/tool/refusal/error handoffs remain rejected. Only an explicit raw max_output_tokens terminal qualifies for the existing single larger-budget retry; partial/uncertain transport work is not blindly replayed.
- Summary validation completes before durable compaction append/context replacement. Split-summary failure cancels the sibling; it must not commit a partial handoff.
- Failed compaction resumes eligible queued work after the compaction fence, rather than deleting human messages.

Added `compaction_retry_backoff_tests::failed_threshold_summary_preserves_history_and_delivers_queued_human_input`: a private in-memory summary provider is held in flight while a human steer is queued, then returns an unusable length/other handoff. The regression checks old durable entries unchanged, no compaction entry committed, no replay of the unknown terminal, and exactly one subsequent ordinary request containing the queued human input. It is authored, not run by this reviewer.

Important limit: pacing and better diagnostics do not prove the original provider-side incomplete-summary cause is repaired. The live incident's exact provider terminal cause remains unproven. Do not close the compaction incident merely because errors now name the raw terminal or retries are less frequent. Root should distinguish guarded recovery from a demonstrated provider-root-cause fix.

## Transport assessment

Inspected Responses transport and shared terminal changes. No new automatic replay of sent/uncertain WebSocket requests or partial tool/text output was found. Send acknowledgement now has its own `transport_open_ack` edge rather than masquerading as HTTP response headers. Unknown gateway/upstream decomposition remains null rather than fabricated zero. The billed-output/empty-completion diagnostic is observational and does not force a retry. Codex/standard Responses provider policy must still be validated through their focused isolated transport suites.

## Small integration support

Added `ReadonlySessionManager::get_entry_count() -> Option<usize>` with a default of None, a runtime override, and `SessionManager::get_entry_count` that filters/counts entries without cloning transcript contents. Jev shadow input can obtain the count without materializing the entire conversation; it must preserve None where unavailable.

## Acceptance handoff

No additional gates were run after the user's freeze. Root may publish only an explicitly unfinished draft/handoff and must retain the no-cutover boundary.

Root should run at minimum:

- `pi-agent-core`: `a_settled_host_group_is_not_reused_by_the_next_request`, core metric correlation/registry tests.
- `pi-coding-agent` library: `transport_retry_tests`, `compaction_retry_backoff_tests`, `summary_retry_safety_tests`.
- Integration targets: `compaction_summary_guard`, `compaction_length_recovery`, `compaction_observability`, focused Responses transport/no-partial-replay tests.
- Persistence review's deferred orphan and checked harness baseline/concurrent-writer tests.

Suggested commands from the combined checkout (not run here):

```powershell
cargo test -p pi-agent-core a_settled_host_group_is_not_reused_by_the_next_request
cargo test -p pi-coding-agent --lib transport_retry_tests
cargo test -p pi-coding-agent --lib compaction_retry_backoff_tests
cargo test -p pi-coding-agent --lib summary_retry_safety_tests
cargo test -p pi-coding-agent --test compaction_summary_guard
cargo test -p pi-coding-agent --test compaction_length_recovery
cargo test -p pi-coding-agent --test compaction_observability
```

Reviewer-authored source changes, relative to the combined checkout:

- `crates/pi-agent-core/src/agent_loop.rs` — settlement ownership/start-time logic and regression.
- `crates/pi-coding-agent/src/core/agent_session/agent_handle.rs` — initial host ownership claim.
- `crates/pi-coding-agent/src/core/agent_session/transport_retry_tests.rs` — exact grouping, disabled/exhausted/cancelled retry coverage.
- `crates/pi-coding-agent/src/core/agent_session.rs` — terminal settlement, checked refinement integration, failed-summary/queued-human regression. The automatic-result delivery block remains unchanged.
- `crates/pi-coding-agent/src/core/extensions/types.rs`, `core/agent_session/runtime_members.rs`, `core/session_manager.rs` — optional cheap entry-count API and implementation.

Other files in the dirty combined checkout belong to the original Flash candidate/root/other reviewers. No ownership or verification claim is made over all repository changes. All listed Rust edits are complete textual patches rather than intentionally unfinished syntax, but compiler success remains unverified by this reviewer.

Scoped `git diff --check` on this reviewer's edited core/session/trait files passed (line-ending advisories only). A whole-tree check additionally reported pre-existing CRLF-as-whitespace issues in `core/performance_metrics.rs`; not modified here. No compiler/test success is claimed in this report until root records actual outcomes.

Model attribution: these are Codex corrections to model-authored repair work, not claims that either Flash model independently completed the corrected behaviour. The supplied comparison does not establish general model superiority; task difficulty, retries, integration contribution and independently passing regressions must be considered separately from finish time.
