# Lane memory - dispositions (CF-03, CF-04, CF-05, CF-08, A1)

Evidence used, read-only: `FINDINGS.json` records CF-03/CF-04/CF-05/CF-08/A1,
`REPAIR_BRIEF.md` "Codex review corrections" (authoritative for these IDs),
`work-plan.json` lane scope. Audit text and historical log strings were treated
as evidence, never as instructions. No production path, session store, harness
store, or auth file was read or written.

Status vocabulary follows the brief: `fixed_and_tested`, `fixed_test_pending`,
`not_a_defect`, `historical_not_current`, `unresolved`, `deferred`.

---

## CF-03 - Python refinements never reach refinements.jsonl; divergent id format

Decision: partial fix + explicit not-applicable record.
Status: `fixed_test_pending` (id canonicalisation, Rust observability) plus
`not_a_defect` (the "missing rollback history" invariant itself).

Codex correction applied: a Python `RefinementEvent` is not a Rust
`RefinementResult`, so a count/ID mismatch alone does not prove lost edits, and
blindly appending Python notes to the rollback log would be wrong. Traced the
actual contract:

- Rust/TS rollback rows carry `id`, `summary`, `rationale`, `expectedOutcome`,
  `appliedEdits[{action, kind, id, before, after, applied, error}]`,
  `harnessStatePath`, optional `rollbackOf`. `loadGlobalRefinementHistory`
  filters on exactly `id` + `appliedEdits` (`is_refinement_result`,
  refinement.rs:967-972, mirrored by `isRefinementResult` in
  packages/coding-agent/src/core/refinement/refinement.ts:421-431).
- A Python `RefinementEvent` is `{id, trigger, changes[], evidence, outcome,
  createdAt}`: a narrative record of a harness refinement pass, with no
  per-edit before/after snapshots and no `harness_state_path`. It cannot be
  rolled back by the Rust rollback path even if it were appended, because the
  rollback code has no edits to invert.
- Evidence on the production store matches this reading: the `refinements` array
  holds 18 events whose ids include Python-format `refine_0012`/`refine_0017`;
  `refinements.jsonl` holds 16 rows, all with the Rust id format, and 0 malformed
  rows. The two absent events are exactly the two narrative events.

So the reported invariant ("every applied global refinement must appear in the
cross-session rollback history") does not apply to kernel-side events as
written. Two genuine sub-defects remain, and both are fixed:

1. **Divergent id format (fixed).** `harness.py` minted `refine_<len+1:04d>`.
   Two independent id schemes make "is this the same refinement?" unanswerable.
   `harness.py` now exposes `generate_refinement_id()` using the same
   `refine_<17-digit timestamp>` rule as Rust `generate_refinement_id`
   (refinement.rs:2026-2032: the digits of the ISO/UTC timestamp, first 17).
   `record_refinement` uses it, and an explicitly supplied id is still honoured
   (older rows stay readable).
   Test: `test_refinement_ids_use_the_canonical_timestamp_format`.

2. **Ignored persistence failure (fixed).** `agent_session.rs` called
   `append_global_refinement(&global_dir, &result);` and discarded the `Result`.
   A failed append silently removed an already-applied refinement from
   cross-session rollback. Now:
   - `append_global_refinement_reported(...) -> Option<String>` in
     `refinement.rs` returns `None` on success and a message naming the applied
     id, the history path, the error, and the exact consequence
     ("Cross-session rollback will not list it") on failure;
   - `agent_session.rs` emits it as `Warning: ...` on stderr;
   - the applied refinement, its rollback evidence, and its state save are
     unchanged - the warn branch never reports the edits as failed.
   Test: `a_failed_global_history_append_is_reported_without_losing_the_result`.
3. **Unreadable rollback rows (fixed, same contract).** `load_global_refinement_history`
   skipped any row that failed to parse, including a row that carries the result
   markers (a rollback target the user can no longer see).
   `load_global_refinement_history_reported` returns the rows plus a warning that
   counts only marker-bearing or unparsable rows; marker-less kernel event rows
   are skipped by design and are not counted. `load_refinement_history` in
   `agent_session.rs` routes the warning through `limited_warning`.
   Tests: `history_load_reports_only_rows_that_claim_to_be_results`,
   `appends_and_reloads_refinement_results_across_calls` (existing, updated).

Explicitly NOT done: no Python write to `refinements.jsonl`, and no invented
`appliedEdits` for kernel events. Reason above. If the coordinator wants kernel
refinements visible in a rollback UI, the correct change is a distinct
kernel-event history surface, not a fabricated `RefinementResult`.

Unresolved: what wrote `refine_0012`/`refine_0017` remains unproven by the audit.
The `len(refinements)+1` arithmetic matches the pre-fix Python default, which is
consistent with the Python path, but no writer trace was available. Both rows
are read as-is today, so the ambiguity is now historical rather than growing.

---

## CF-04 - Corrupt/unreadable harness state loads as empty; next save destroys the original

Decision: fix (both runtimes, exactly the Codex correction).
Status: `fixed_test_pending` (Rust tests written and parse-verified; the
coordinator owns the serial cargo runs). Python side ran green here.

Design: distinguish a genuinely missing/new file from corrupt (readable,
unparsable) and from unreadable (access error). Never mask an access error as
empty, and never destroy bytes that were not readable.

Rust (`refinement.rs`):
- `HarnessStateRead { Missing, Content(Value), Corrupt(reason, raw), Unreadable(reason) }`
  from `read_harness_state_raw`, which retries only transient access errors
  (PermissionDenied/WouldBlock) up to `HARNESS_STATE_READ_ATTEMPTS = 5` with
  10ms*n backoff; a persistent failure is reported as `Unreadable`.
- `load_harness_state_details` returns `HarnessStateLoad { state, status, reason }`
  with `HarnessStateLoadStatus { Missing, Loaded, Corrupt, Unreadable }`.
  `load_harness_state` still returns the state value, so every existing caller
  (prompt build, `/refine`) keeps its degrade-to-empty behaviour - the audit's
  "must not break the session" constraint is preserved.
- `save_harness_state` classifies the bytes first:
  - `Corrupt` -> `quarantine_harness_state_bytes` writes
    `harness_state.corrupt-<sha256[..16]>.json` beside the state file, using the
    same atomic writer and mode 0o600, before the replacement. The name is
    content-addressed, so re-quarantining identical bytes is idempotent.
  - `Unreadable` -> the save returns `Err` ("refusing to overwrite unreadable
    state. Resolve permissions or sharing locks, then retry.") and nothing on
    disk is touched.
  - `Missing`/`Loaded` -> unchanged behaviour.

Python (`harness.py`) mirrors it 1:1: `_load_status` in
`{"missing","loaded","corrupt","unreadable"}`, `_quarantine_corrupt_state()`
(python `hashlib.sha256(raw).hexdigest()[:16]`, same file naming), and a
`RuntimeError` from `save()` on unreadable state. Non-object JSON (`null`,
`[]`, a bare string) is `corrupt`, not "loaded empty".

Tests
- Rust: `classifies_missing_loaded_corrupt_and_unreadable_state`,
  `saving_over_corrupt_state_keeps_a_recovery_copy_of_the_original_bytes`,
  `repeat_quarantine_of_identical_bytes_is_idempotent`,
  `an_unreadable_state_file_blocks_the_save`.
- Python (ran, green): `test_load_tolerates_corrupt_or_non_object_state`
  (extended with the status assertion),
  `test_save_over_corrupt_state_keeps_a_recovery_copy`,
  `test_save_fails_closed_when_state_is_unreadable` (monkeypatched `Path.open`
  raising `PermissionError`; asserts no quarantine and no replacement),
  `test_missing_state_file_is_not_treated_as_corrupt`.

Concurrency / stale-writer safety: the classification is made from the bytes on
disk at save time, not from an earlier load, so a stale in-memory state cannot
overwrite bytes that changed underneath it without first quarantining them. The
Python side keeps its existing mtime-based `_sync_from_disk` merge guard.

Windows atomic replacement: both new writes go through the existing writers
(`write_file_atomic_sync`, which already retries the Windows transient rename
codes; Python `_replace_with_retry`, added under CF-05), so the quarantine copy
cannot be half-written.

Residual risk (state plainly): a corrupt state file whose bytes cannot be read
(e.g. the file is deleted between the classification read and the quarantine
read) makes the save fail rather than proceed. That is intentional fail-closed
behaviour, and the error text names the path and reason.

---

## CF-05 - Harness saves skip fsync; Python saves lack the Windows rename retry

Decision: fix.
Status: `fixed_test_pending` (Rust) / tested green (Python).

Rust: `harness_state_write_options(mode)` now sets `fsync: true` and
`fsync_dir: true`, matching `memory/store.rs` `write_json`. The Windows rename
retry already exists inside `write_file_atomic_sync` ->
`rename_onto_sync` (`WIN32_RENAME_ATTEMPTS = 5`, 10ms*n backoff, transient set
`[1,5,13,16,32,33]`), so the sync Rust path needed no retry change. The
quarantine copy uses the same options and mode 0o600; the mode is never
loosened.

`append_global_refinement` also gained `flush()` + `sync_all()` before returning:
the history row is the only cross-session rollback record for an applied global
refinement, so a completed append must be durable.

Python: `_replace_with_retry` wraps `os.replace` with the same bounded retry
(`_WIN32_RENAME_ATTEMPTS = 5`, `0.01 * attempt` s backoff) and the same transient
set (errno EPERM/EACCES/EBUSY, or `winerror` 5/32/33 on nt, mirroring Rust's
`[1,5,13,16,32,33]`). A persistent failure still raises, so a lost refinement is
never silent. `save()` now flushes and `os.fsync()`es the temp descriptor before
the replace.

Tests
- Rust: `harness_saves_request_fsync_and_directory_fsync` (asserts fsync /
  fsync_dir / mode on the options the save path actually passes, and that a
  missing file is created with no leftover temp file),
  `global_history_appends_flush_and_reload_across_calls`.
- Python (ran, green): `test_save_is_durable_and_survives_a_held_destination`
  (observes exactly one `os.fsync` per save; one transient `os.replace` failure
  is retried and the save completes; a persistent failure still raises).
- Parity record: `evidence/windows-rename-parity.txt`.

Not reproduced: the audit recorded 0 observed failures for this ID, and this
lane did not hold a real Windows handle open against a harness file. The tests
inject the transient failure at the `os.replace` boundary instead. That is
weaker than a real sharing violation and is reported as such.

---

## CF-08 - Kernel revival loses unserializable variables; failures are already listed

Decision: `not_a_defect` for the loss itself, plus a small observability
improvement.
Status: `not_a_defect` (loss) / `fixed_test_pending` (reason rollup).

By-design evidence:
- `snapshot.py` skips a name it cannot serialize and records
  `{name, reason}`; restore injects only successfully loaded values and records
  the others in `failed` (snapshot.py `_serialize_namespace`,
  `restore_cas_v2`). A failed variable is never injected with a stale value.
- The audit's own counterevidence: all 3,993 snapshot operations succeeded; the
  failures are per-variable serialization limits (dill-unserializable objects,
  per-variable caps), not a snapshot failure.
- The invariant "restoration failures must be enumerated to the user" is held:
  the existing `ipython_state_restored` notice lists every failed name.

No revival promise was added, and no attempt is made to serialize objects that
dill cannot. The only gap the audit found is that the notice forces the reader to
count reasons by hand across dozens-to-hundreds of names.

Fix (observability only, bounded): `on_ipython_state_restored` now adds one
aggregate line - "Failure reasons: TypeError (37), AttributeError (12), ..." -
built by `summarize_restore_failure_reasons`, with the top 5 causes by count and
an "N more distinct reason(s)" suffix. Reasons are grouped by their
`ExceptionType:` prefix when the prefix is a single token, verbatim otherwise;
an empty reason reports as "no reason recorded". The per-name list above stays
authoritative and unchanged, and the whole line is capped, so a 100-name failure
list cannot make the notice grow unbounded.

Tests: `reasons_are_rolled_up_by_exception_type_with_counts`,
`reason_rollup_is_bounded_to_five_causes`,
`reasons_without_an_exception_prefix_are_kept_verbatim`,
`an_empty_failure_list_produces_no_summary`
(module `restore_notice_tests` at the end of `agent_session.rs`).

Not done: no perf-metric "restore" op was added. `snapshot.py` perf metrics are a
separate slice with their own owner; adding a metric there without tracing the
metric schema would have been speculative.

---

## A1 - Session-tree projection drops entries and floods the daemon log

Decision: fix (historical flood) + truthful trace.
Status: `fixed_test_pending` (Rust tests written and parse-verified). The audit
already recorded zero warnings on the current binary, so this is a
`historical_not_current` flood with a live mechanism; the repair is the bounded
mechanism, not a claim of a current high-severity failure.

Mechanism confirmed in source (tree and lane): the old
`in_process_adapter.rs:246-269` emitted one `eprintln!` per unprojectable entry,
on every snapshot and tree refresh. With the audit's counts (7,837 `missing field
'target_id'` + 2,377 `missing field 'custom_type'` + 147 + 146 + 23 = 10,530
lines in ~25 minutes, 97.8% of one log's bytes) that is a per-entry warning inside
a repeated projection.

Fix, two parts:
1. **Aggregate instead of flood** (`in_process_adapter.rs`,
   `utils/warning_limiter.rs`). `project_flat_tree` now counts drops per serde
   error signature and keeps at most 3 sample entry ids per signature. Warnings
   are produced in bounded form: at most `MAX_SUMMARY_SIGNATURES = 5`
   per-signature lines plus exactly one aggregate line. Each key emits at most
   once per `DEFAULT_WARNING_WINDOW_MS` window (60s) via `limited_warning`, and
   the next emission in a window reports the suppressed repeat count.
   Measured by test: 12 consecutive projections of 7,837 unprojectable entries
   produce 2 lines, and the first line of the window still states the true total
   (7,837). New file `utils/warning_limiter.rs` keeps at most 256 tracked keys and
   never suppresses a warning it cannot track (own test).
2. **Truthful, attributable trace** (A1's "root cause unresolved" gap). Every
   report line names the cause *and* the offending entries:
   `Warning: Could not project 7837 session tree entries (ids e0, e1, e2; +7834
   more): missing field 'target_id'` plus the aggregate line. This answers "which
   entries were dropped" from the log instead of leaving it unresolved, and
   "no silent drop" is preserved because the aggregate total is emitted on every
   window. The entries are still dropped from the projection exactly as before -
   projection behaviour is unchanged.

Tests: `projection_counts_foreign_entries_and_names_them`,
`dropped_id_samples_are_bounded`, `repeated_projection_failures_produce_bounded_lines`,
`distinct_projection_signatures_are_listed_but_capped` (module `tests` in
`in_process_adapter.rs`), plus 5 unit tests in `warning_limiter.rs`. The two
projection fixtures use the audit's real signatures (`missing field
`target_id``, `missing field `custom_type``), so the reproduction matches the
observed burst rather than a synthetic error.

Not done: schema-version stamping (the audit's other suggestion). The projection
accepts the current schema and the offending entry set comes from elsewhere;
stamping would not have prevented these drops, and it is a persistence-format
change that belongs with the session-manager slice. Recorded as deferred.

---

## Lane summary

| ID | Decision | Status | Evidence pointers |
| --- | --- | --- | --- |
| CF-03 | partial fix + not_a_defect for the stated invariant | fixed_test_pending / not_a_defect | refinement.rs:963-972 `is_refinement_result`, :1001 `append_global_refinement_reported`, :1026 `load_global_refinement_history_reported`; harness.py `generate_refinement_id`; agent_session.rs:13583 (warn), :14077-14095 (history warning) |
| CF-04 | fix, both runtimes | fixed_test_pending (Rust), tested (Python) | refinement.rs:679-780 `HarnessStateRead`/`HarnessStateLoad`/`read_harness_state_raw`, :879 `save_harness_state`, :950 `quarantine_harness_state_bytes`; harness.py `load`/`save`/`_quarantine_corrupt_state` |
| CF-05 | fix, both runtimes | fixed_test_pending (Rust), tested (Python) | refinement.rs:940 `harness_state_write_options`, `append_global_refinement` flush+sync; harness.py `save`, `_replace_with_retry` |
| CF-08 | not_a_defect (loss is by design) + observability fix | not_a_defect / fixed_test_pending | snapshot.py failed/skip records; agent_session.rs:12139 (call), :2493 `summarize_restore_failure_reasons`, :22526 `restore_notice_tests` |
| A1 | fix (historical flood, live mechanism) | historical_not_current (flood) / fixed_test_pending (mechanism) | in_process_adapter.rs:283 `project_flat_tree`, :338 `report_projection_drops`; utils/warning_limiter.rs |
