//! W48 post-wave adversarial review: red-team tests for the D21-D29 fix
//! code itself (integration tip 71e770c5e).
//!
//! W35 red-teamed the typed implementation; the fixes landed in waves
//! (D21 render caps, D22 render direction, D23 grammar, D24 stable ids,
//! D25 step boundary + durable cap, D26 salvage, D27 fabricated-claim,
//! D28 native carry-through, D29 provenance pairing). This suite attacks
//! the FIXES: one adversarial test per surface from the W48 brief.
//! Tests that pass pin the fixed guarantees. Tests that fail are new
//! defects recorded in reports/context-rot-r2/BUG-LEDGER.md (D31+) and
//! carry an `#[ignore = "Dnn: ..."]` note, runnable red with
//! `cargo test -- --ignored`.
//!
//! Attack map:
//! 1. D25 retired_refs vs the fabricated-claim check: a summary quoting a
//!    retired id in a verified-evidence claim (not reference resolution).
//! 2. D25 cap interplay: a ledger at the 512 cap where the allocator's
//!    keep order and cap_drop_rank disagree on tie-breaks.
//! 3. D24 + D29: an evolving tool record (stable id) whose text moves into
//!    a correction's subject across three cycles.
//! 4. D26 salvage: record slots carrying unknown fields (additive schema
//!    evolution), not missing ones.
//! 5. D29 over-fire: interleaved decisions and corrections (D1 C1 D2 C2).
//! 6. D29 token tiers: a correction sharing tokens with both a decision
//!    and an earlier correction (the W9-F4 shape).
//! 7. D28 + D26: native provider compaction over a malformed prior ledger.
//! 8. D22: a supersession chain (A retired by B, B retired by C) rendered
//!    end to end.
//! 9. D33 (W54): a restatement-absorbed record's citations, the E5-RM
//!    TASK-E cycle-2 shape, resolved through retired_refs tombstones.
//!
//! No network. Unit-level attacks drive `typed_state` directly; the D28
//! attack drives the real native arm of `compact_with_metrics` through a
//! fixture compaction provider (same shape as rot2_native_typed_chain.rs).

use std::pin::Pin;
use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::api_registry::{register_api_provider_simple, ApiProviderSimple};
use pi_ai::compaction::{CompactionRequestError, ProviderCompactionCheckpoint, ProviderCompactionResult};
use pi_ai::types::{
    AssistantMessage, ContentBlock, Context, Message, Model, TextContent, ToolResultMessage,
    UserContent, UserMessage,
};
use pi_coding_agent::core::compaction::compaction::{
    compact, default_compaction_settings, default_summary_call_runner, prepare_compaction,
    CompactionSessionEntry, CompactionSettings, TASK_STATE_MODE_TYPED,
};
use pi_coding_agent::core::compaction::typed_state::{
    allocate, build_typed_state, merge, render_hot_state, verify, MAX_DURABLE_RECORDS,
    RecordStatus, SupersessionPath, TypedInformationKind, TypedStateLedger, TypedStateRecord,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Entry helpers (typed_state window shapes, mirroring rot2_typed_adversarial.rs)
// ---------------------------------------------------------------------------

fn user_entry(id: &str, text: &str) -> CompactionSessionEntry {
    CompactionSessionEntry::Message {
        id: id.to_string(),
        parent_id: None,
        message: AgentMessage::Message(Message::User(UserMessage::new(
            UserContent::Text(text.into()),
            0,
        ))),
    }
}

fn assistant_entry(id: &str, text: &str) -> CompactionSessionEntry {
    CompactionSessionEntry::Message {
        id: id.to_string(),
        parent_id: None,
        message: AgentMessage::Message(Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::Text(TextContent::new(text))],
            stop_reason: "stop".into(),
            ..Default::default()
        })),
    }
}

fn tool_entry(id: &str, text: &str) -> CompactionSessionEntry {
    CompactionSessionEntry::Message {
        id: id.to_string(),
        parent_id: None,
        message: AgentMessage::Message(Message::ToolResult(ToolResultMessage::new(
            "call-1",
            "bash",
            vec![pi_ai::types::ImageOrTextContent::Text(TextContent::new(text))],
            false,
            0,
        ))),
    }
}

/// A previous compaction entry with hand-chosen details, standing in for a
/// persisted prior cycle (same shape as rot2_typed_adversarial.rs).
fn compaction_marker(id: &str, first_kept_entry_id: &str, details: &Value) -> CompactionSessionEntry {
    CompactionSessionEntry::Compaction {
        id: id.to_string(),
        parent_id: None,
        summary: "## Goal\n- Prior work summarized by the previous cycle.\n\n\
                   ## Constraints & Preferences\n- See the durable ledger.\n\n\
                   ## Progress\n### Done\n- [x] Cycle complete.\n\n\
                   ### In Progress\n- [ ] Continuing.\n\n### Blocked\n- None.\n\n\
                   ## Key Decisions\n- None.\n\n## Next Steps\n1. Continue.\n\n\
                   ## Critical Context\n- None."
            .to_string(),
        first_kept_entry_id: first_kept_entry_id.to_string(),
        tokens_before: 10_000.0,
        details: Some(details.clone()),
        from_hook: None,
        custom_instructions: None,
        harness_digest: None,
        timestamp: String::new(),
    }
}

fn typed_settings() -> CompactionSettings {
    let mut settings = default_compaction_settings();
    settings.keep_recent_tokens = 20.0;
    settings.task_state = Some(TASK_STATE_MODE_TYPED.to_string());
    settings
}

fn record_with_text<'a>(ledger: &'a TypedStateLedger, needle: &str) -> &'a TypedStateRecord {
    ledger
        .records
        .iter()
        .find(|record| record.text.contains(needle))
        .unwrap_or_else(|| panic!("no record containing {needle}"))
}

// ---------------------------------------------------------------------------
// Attack 1: D25 retired_refs vs the fabricated-claim check
// ---------------------------------------------------------------------------

/// A retired record id keeps bracketed references resolvable (the D25 fix),
/// but it must not count as EVIDENCE: a summary line that claims verified
/// material while citing only a retired id maps to no record the ledger
/// vouches for, so the fabricated-claim check (the D27 surface) must fire.
/// The retirements here come from the D25 step boundary: a pending action
/// from the first window that the second window does not restate.
#[test]
fn retired_ids_do_not_vouch_for_verified_claims() {
    let first = build_typed_state(&[user_entry(
        "u1",
        "TASK GOAL: migrate the billing service R2-W48-A1. \
Also don't forget to update the changelog after the migration R2-W48-PA.",
    )]);
    let pending = record_with_text(&first, "changelog after the migration").clone();
    assert_eq!(pending.kind, TypedInformationKind::PendingAction);

    let second = build_typed_state(&[
        user_entry(
            "u2",
            "CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.",
        ),
        user_entry("u3", "status update: second window filler content."),
    ]);
    let merged = merge(Some(&first), second);
    // The step boundary retired the pending action: the record is gone and
    // its id is a tombstone in retired_refs.
    assert!(
        !merged.records.iter().any(|record| record.id == pending.id),
        "the pending action left the ledger at the merge"
    );
    assert!(merged.retired_refs.contains(&pending.id));

    let task = merged
        .records
        .iter()
        .find(|record| record.kind == TypedInformationKind::CurrentTask)
        .expect("the task survives the merge")
        .clone();
    let constraint = record_with_text(&merged, "ARCH-2-CHANGELOG").clone();

    // A compliant summary (required material present by id) plus one line
    // that claims verified evidence and cites only the retired id.
    let artifact = format!(
        "## Goal\n- [{}] TASK GOAL: migrate the billing service R2-W48-A1.\n\n\
         ## Constraints & Preferences\n- [{}] CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.\n\n\
         ## Critical Context\n- Verified: the changelog update was confirmed by the completed tool run [{}].\n",
        task.id, constraint.id, pending.id
    );
    let verification = verify(&artifact, &merged);
    assert!(
        !verification.fabricated.is_empty(),
        "a retired id must not vouch for a verified claim: {verification:?}"
    );
    assert!(!verification.passed);

    // The same laundering through the retired source entry id quoted
    // without the @ sigil: also not evidence for a verified claim.
    let artifact_entry_ref = format!(
        "## Goal\n- [{}] TASK GOAL: migrate the billing service R2-W48-A1.\n\n\
         ## Constraints & Preferences\n- [{}] CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.\n\n\
         ## Critical Context\n- Verified: the migration was confirmed [{}].\n",
        task.id, constraint.id, pending.source_entry_id
    );
    let verification_entry_ref = verify(&artifact_entry_ref, &merged);
    assert!(
        !verification_entry_ref.fabricated.is_empty(),
        "a retired entry id must not vouch for a verified claim: {verification_entry_ref:?}"
    );

    // Control: the same line citing the live task record maps legitimately.
    let artifact_control = format!(
        "## Goal\n- [{}] TASK GOAL: migrate the billing service R2-W48-A1.\n\n\
         ## Constraints & Preferences\n- [{}] CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.\n\n\
         ## Critical Context\n- Verified: the migration goal is recorded [{}].\n",
        task.id, constraint.id, task.id
    );
    let verification_control = verify(&artifact_control, &merged);
    assert!(
        verification_control.fabricated.is_empty(),
        "a live verified record is legitimate evidence: {verification_control:?}"
    );
}

/// Positive pin for the D31 fix: the fabricated-claim check's resolver is
/// restricted to the records the ledger carries, so a "verified" line
/// citing a LIVE record still maps while tombstones sit in the same
/// ledger's retired_refs. Both renderer-grammar reference forms count:
/// `[id]` for a record, `[@entry]` for the source entry of a carried
/// record.
#[test]
fn live_record_ids_still_vouch_for_verified_claims() {
    let first = build_typed_state(&[user_entry(
        "u1",
        "TASK GOAL: migrate the billing service R2-W48-A1. \
Also don't forget to update the changelog after the migration R2-W48-PA.",
    )]);
    let pending = record_with_text(&first, "changelog after the migration").clone();
    let second = build_typed_state(&[user_entry(
        "u2",
        "CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.",
    )]);
    let merged = merge(Some(&first), second);
    assert!(merged.retired_refs.contains(&pending.id));

    let task = merged
        .records
        .iter()
        .find(|record| record.kind == TypedInformationKind::CurrentTask)
        .expect("the task survives the merge")
        .clone();
    let constraint = record_with_text(&merged, "ARCH-2-CHANGELOG").clone();

    // The record-id form: a verified line citing the live task record.
    let artifact = format!(
        "## Goal\n- [{}] TASK GOAL: migrate the billing service R2-W48-A1.\n\n\
         ## Constraints & Preferences\n- [{}] CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.\n\n\
         ## Critical Context\n- Verified: the migration goal is recorded [{}].\n",
        task.id, constraint.id, task.id
    );
    let verification = verify(&artifact, &merged);
    assert!(
        verification.fabricated.is_empty(),
        "a live record id still vouches: {verification:?}"
    );

    // The @-sigil form: a verified line citing the source entry of a
    // carried record.
    let artifact_entry = format!(
        "## Goal\n- [{}] TASK GOAL: migrate the billing service R2-W48-A1.\n\n\
         ## Constraints & Preferences\n- [{}] CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG.\n\n\
         ## Critical Context\n- Verified: the constraint was recorded [@{}].\n",
        task.id, constraint.id, constraint.source_entry_id
    );
    let verification_entry = verify(&artifact_entry, &merged);
    assert!(
        verification_entry.fabricated.is_empty(),
        "a carried entry reference still vouches: {verification_entry:?}"
    );
}

// ---------------------------------------------------------------------------
// Attack 2: D25 cap interplay at the 512 boundary
// ---------------------------------------------------------------------------

/// A ledger at the durable cap where the allocator's keep order and
/// cap_drop_rank disagree on tie-breaks (the allocator prefers strongest
/// verification then newest within a tier; the cap drops oldest first and
/// ignores verification). The invariant under attack: a required record is
/// never dropped by the cap while a superseded record survives and renders.
///
/// Window one classifies 512 user task statements: task succession retires
/// each statement in favor of the next, so the ledger sits exactly at the
/// cap with 511 superseded records and one required CURRENT_TASK.
#[test]
fn cap_never_drops_a_required_record_while_a_superseded_one_renders() {
    let mut tasks = Vec::new();
    for index in 0..MAX_DURABLE_RECORDS {
        tasks.push(user_entry(
            &format!("task-{index:03}"),
            &format!("TASK GOAL: migrate the billing service stage {index:03}."),
        ));
    }
    let first = build_typed_state(&tasks);
    assert_eq!(first.records.len(), MAX_DURABLE_RECORDS);
    let current_task = first
        .records
        .iter()
        .find(|record| record.status == RecordStatus::Current)
        .expect("the newest task stays current")
        .clone();
    assert_eq!(current_task.kind, TypedInformationKind::CurrentTask);
    assert_eq!(
        first
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::Superseded)
            .count(),
        MAX_DURABLE_RECORDS - 1
    );

    // Sub-case A: a small merge pushes the ledger two records past the
    // cap. Only superseded history may drop; the required task survives
    // and the allocator renders it ahead of every superseded line.
    let small = build_typed_state(&[
        user_entry("u2a", "CONSTRAINT: keep the canary rollout ARCH-A."),
        user_entry("u3a", "CONSTRAINT: keep the runbook pinned ARCH-B."),
    ]);
    let merged_small = merge(Some(&first), small);
    assert_eq!(merged_small.records.len(), MAX_DURABLE_RECORDS);
    assert_eq!(
        merged_small.dropped_by_cap, 2,
        "512 carried + 2 fresh records drop the two oldest superseded lines"
    );
    assert!(
        merged_small
            .records
            .iter()
            .any(|record| record.id == current_task.id),
        "the required task must survive a merge that drops history"
    );
    let allocation = allocate(&merged_small, usize::MAX);
    let task_line = allocation
        .included
        .iter()
        .position(|item| item.id == current_task.id)
        .expect("the required task renders");
    let first_superseded = allocation
        .included
        .iter()
        .position(|item| item.status == RecordStatus::Superseded)
        .expect("superseded history renders at the unlimited budget");
    assert!(
        task_line < first_superseded,
        "the allocator must render required material before superseded history"
    );
    let dropped_superseded: Vec<&String> = first
        .records
        .iter()
        .filter(|record| record.status == RecordStatus::Superseded)
        .map(|record| &record.id)
        .filter(|id| !merged_small.records.iter().any(|record| &record.id == *id))
        .collect();
    assert_eq!(dropped_superseded.len(), 2);
    for id in &dropped_superseded {
        assert!(merged_small.retired_refs.contains(id));
    }

    // Sub-case B: the worst case. 513 fresh required constraints force the
    // cap past every superseded record and past the old required task. When
    // a required record drops, no superseded record may survive: the drop
    // order empties the history tier before it touches tier 0.
    let fresh: Vec<CompactionSessionEntry> = (0..=MAX_DURABLE_RECORDS)
        .map(|index| {
            user_entry(
                &format!("u2b-{index:03}"),
                &format!("CONSTRAINT: keep the marker arch{index:03} intact."),
            )
        })
        .collect();
    let big = build_typed_state(&fresh);
    let merged_big = merge(Some(&first), big);
    assert_eq!(merged_big.records.len(), MAX_DURABLE_RECORDS);
    assert_eq!(
        merged_big.dropped_by_cap,
        MAX_DURABLE_RECORDS + 1,
        "511 superseded + the old task + one fresh constraint drop"
    );
    assert!(
        !merged_big
            .records
            .iter()
            .any(|record| record.id == current_task.id),
        "the old required task is the oldest tier-0 record and drops"
    );
    assert!(
        merged_big
            .records
            .iter()
            .all(|record| record.status == RecordStatus::Current),
        "a required record may drop only after every superseded record dropped"
    );
    assert!(merged_big.retired_refs.contains(&current_task.id));
    assert!(merged_big.retired_refs.contains(&current_task.source_entry_id));
    // The capped render stays honest: no SUPERSEDED section, and the cap
    // line is visible (the D21/D25 render-cap idiom).
    let render = render_hot_state(&merged_big);
    assert!(!render.contains("SUPERSEDED (HISTORY, NOT CURRENT)"));
    assert!(render.contains(&format!(
        "{} record(s) omitted by the durable ledger cap",
        MAX_DURABLE_RECORDS + 1
    )));
    let verification = verify(&render, &merged_big);
    assert!(verification.passed, "findings: {verification:?}");
}

// ---------------------------------------------------------------------------
// Attack 3: D24 + D29 combined (evolving tool state + provenance pairing)
// ---------------------------------------------------------------------------

/// An evolving tool record keeps its stable id across merges (D24) while a
/// zero-overlap user correction retires the stale decision by provenance
/// (D29). Across a third cycle the tool record's text moves onto the
/// correction's subject: the retired decision must stay retired (no
/// resurrection through the evolving text) and the correction must keep
/// exactly one pairing (no double retirement).
#[test]
fn evolving_tool_state_with_correction_provenance_neither_double_retires_nor_resurrects() {
    let first = build_typed_state(&[
        user_entry("u1", "TASK GOAL: migrate the billing service to the new ledger R2-W48-A3."),
        assistant_entry(
            "a1",
            "Decision: persistence goes through the postgres driver connection pool.",
        ),
        tool_entry("t1", "on branch rot2-w48; 3 files changed, 12 insertions(+)"),
    ]);
    let repo = record_with_text(&first, "files changed").clone();
    assert_eq!(repo.kind, TypedInformationKind::RepositoryState);
    let decision = record_with_text(&first, "postgres driver").clone();
    assert_eq!(decision.kind, TypedInformationKind::VerifiedDecision);

    // Second window: the repository state evolves (same record, new text)
    // and a zero-overlap correction follows the decision.
    let second = build_typed_state(&[
        tool_entry("t2", "on branch rot2-w48; 5 files changed, 20 insertions(+)"),
        user_entry(
            "u2",
            "Correction: scrap that; use sqlite embedded engine instead.",
        ),
    ]);
    let merged_second = merge(Some(&first), second);
    let repo_second = record_with_text(&merged_second, "5 files changed").clone();
    assert_eq!(repo_second.id, repo.id, "the evolving record keeps its id");
    let decision_second = merged_second
        .records
        .iter()
        .find(|record| record.id == decision.id)
        .expect("the decision is retained as history")
        .clone();
    let correction = record_with_text(&merged_second, "sqlite embedded engine").clone();
    assert_eq!(correction.kind, TypedInformationKind::UserCorrection);
    assert_eq!(decision_second.status, RecordStatus::Superseded);
    assert_eq!(decision_second.link.as_deref(), Some(correction.id.as_str()));
    assert_eq!(decision_second.retire_path, Some(SupersessionPath::Provenance));
    assert_eq!(correction.link.as_deref(), Some(decision.id.as_str()));
    assert_eq!(correction.status, RecordStatus::Current);
    assert_eq!(
        merged_second
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::Superseded)
            .count(),
        1
    );
    let render_second = render_hot_state(&merged_second);
    assert!(verify(&render_second, &merged_second).passed);

    // Third window: the repository state evolves onto the correction's
    // subject and a fresh decision arrives. The retired decision must not
    // come back and the correction must not re-pair.
    let third = build_typed_state(&[
        tool_entry(
            "t3",
            "on branch rot2-w48; 7 files changed, 30 insertions(+)",
        ),
        assistant_entry(
            "a3",
            "Decision: the reporting module will use sqlite engine as well.",
        ),
    ]);
    let merged_third = merge(Some(&merged_second), third);
    let repo_third = record_with_text(&merged_third, "7 files changed").clone();
    assert_eq!(repo_third.id, repo.id, "the id stays stable across both merges");
    let decision_third = merged_third
        .records
        .iter()
        .find(|record| record.id == decision.id)
        .expect("the retired decision is still history")
        .clone();
    assert_eq!(
        decision_third.status,
        RecordStatus::Superseded,
        "no resurrection through the evolving tool text"
    );
    assert_eq!(decision_third.link.as_deref(), Some(correction.id.as_str()));
    let correction_third = merged_third
        .records
        .iter()
        .find(|record| record.id == correction.id)
        .expect("the correction is retained")
        .clone();
    assert_eq!(correction_third.status, RecordStatus::Current);
    assert_eq!(
        correction_third.link.as_deref(),
        Some(decision.id.as_str()),
        "the correction keeps its single pairing"
    );
    assert_eq!(
        merged_third
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::Superseded)
            .count(),
        1,
        "no double retirement"
    );
    let render_third = render_hot_state(&merged_third);
    assert!(verify(&render_third, &merged_third).passed);
}

// ---------------------------------------------------------------------------
// Attack 4: D26 salvage vs unknown fields (additive schema evolution)
// ---------------------------------------------------------------------------

/// Record slots carrying unknown FIELDS (a future schema version adds a
/// field) must not be rejected by the strict parse: additive evolution is
/// the compatibility direction the salvage exists to protect. If serde
/// rejected unknown fields, every slot would fall to the salvage path and
/// non-repairable slots would drop, which is wrong for additive drift.
#[test]
fn unknown_payload_fields_are_tolerated_by_the_strict_parse() {
    use pi_coding_agent::core::compaction::typed_state::parse_typed_state;

    let ledger = build_typed_state(&[
        user_entry("u1", "CONSTRAINT: never delete the payments ledger ARCH-W48-KEEP."),
        user_entry("u2", "CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG."),
        user_entry("u3", "Last week we tried the nightly export batch ARCH-3."),
    ]);
    assert_eq!(ledger.records.len(), 3);
    let mut payload = serde_json::to_value(&ledger).expect("serialize the typed payload");
    // Unknown record-level fields (a newer schema wrote extra data) plus an
    // unknown ledger-level field.
    payload["records"][0]["provenanceMetadata"] = json!({"window": "cycle-9", "author": "v3"});
    payload["records"][1]["confidence"] = json!(0.9);
    payload["schemaVersion"] = json!(3);
    let parsed = parse_typed_state(&payload);
    assert_eq!(parsed.records.len(), 3, "no record may drop");
    assert!(
        parsed.salvage.is_none(),
        "unknown fields must not force the salvage path: {:?}",
        parsed.salvage
    );
    assert!(parsed.records.iter().any(|record| record.text.contains("ARCH-W48-KEEP")));

    // Mixed drift: an unknown field together with a type-drifted positional
    // field. The salvage repairs the positional field in place and the
    // unknown field must not block the repair.
    let mut drifted = serde_json::to_value(&ledger).expect("serialize the typed payload");
    drifted["records"][0]["ordinal"] = json!("zero");
    drifted["records"][0]["provenanceMetadata"] = json!({"window": "cycle-9"});
    let salvaged = parse_typed_state(&drifted);
    assert_eq!(salvaged.records.len(), 3);
    let salvage = salvaged
        .salvage
        .as_ref()
        .expect("the drifted positional field is salvaged");
    assert_eq!(salvage.dropped_records, 0);
    assert_eq!(
        salvage.repaired_fields,
        vec!["records[0].ordinal".to_string()],
        "the unknown field does not widen the repair"
    );
    assert!(salvaged
        .records
        .iter()
        .any(|record| record.text.contains("ARCH-W48-KEEP")));
}

// ---------------------------------------------------------------------------
// Attack 5: D29 over-fire (interleaved decisions and corrections)
// ---------------------------------------------------------------------------

/// Two decisions and two corrections interleaved (D1 C1 D2 C2) with all
/// token overlaps below the pairing threshold: each correction must pair
/// with the decision it positionally replaces (C1 -> D1, C2 -> D2). The
/// mis-pairing to watch for is C1 -> D2, which would leave D1 current and
/// C2 unpaired (the intervening-decision guard refuses the second pairing).
#[test]
fn interleaved_decisions_and_corrections_pair_by_position() {
    let ledger = build_typed_state(&[
        assistant_entry("a1", "Decision: persistence goes through the postgres driver connection pool."),
        user_entry("u1", "Correction: scrap that; use sqlite embedded engine instead."),
        assistant_entry("a2", "Decision: the reporting pipeline will use the batch worker nightly."),
        user_entry("u2", "Correction: change of plan; streaming runner takes over from now on."),
    ]);
    let first_decision = record_with_text(&ledger, "postgres driver").clone();
    let second_decision = record_with_text(&ledger, "batch worker").clone();
    let first_correction = record_with_text(&ledger, "sqlite embedded engine").clone();
    let second_correction = record_with_text(&ledger, "streaming runner").clone();
    assert_eq!(first_decision.kind, TypedInformationKind::VerifiedDecision);
    assert_eq!(second_decision.kind, TypedInformationKind::VerifiedDecision);
    assert_eq!(first_correction.kind, TypedInformationKind::UserCorrection);
    assert_eq!(second_correction.kind, TypedInformationKind::UserCorrection);
    // C1 pairs with D1, C2 with D2: positional adjacency, no crossing.
    assert_eq!(first_decision.status, RecordStatus::Superseded);
    assert_eq!(
        first_decision.link.as_deref(),
        Some(first_correction.id.as_str()),
        "C1 pairs with the decision it follows"
    );
    assert_eq!(first_decision.retire_path, Some(SupersessionPath::Provenance));
    assert_eq!(second_decision.status, RecordStatus::Superseded);
    assert_eq!(
        second_decision.link.as_deref(),
        Some(second_correction.id.as_str()),
        "C2 pairs with the decision it follows, not with D1"
    );
    assert_eq!(second_decision.retire_path, Some(SupersessionPath::Provenance));
    assert_eq!(first_correction.status, RecordStatus::Current);
    assert_eq!(first_correction.link.as_deref(), Some(first_decision.id.as_str()));
    assert_eq!(second_correction.status, RecordStatus::Current);
    assert_eq!(second_correction.link.as_deref(), Some(second_decision.id.as_str()));
    assert_eq!(
        ledger
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::Superseded)
            .count(),
        2,
        "exactly the two decisions retire; no correction is unpaired or over-fired"
    );
    let render = render_hot_state(&ledger);
    assert!(verify(&render, &ledger).passed);
}

// ---------------------------------------------------------------------------
// Attack 6: D29 token tiers (decision vs sibling correction as target)
// ---------------------------------------------------------------------------

/// A later correction shares MORE tokens with an earlier correction than
/// with the current decision. The non-correction-outranks-corrections rule
/// (the W9-F4 fix) must hold: the token path retires the decision, never
/// the sibling correction, and no two corrections retire each other.
#[test]
fn token_pairing_prefers_the_decision_over_the_sibling_correction() {
    let ledger = build_typed_state(&[
        assistant_entry("a1", "Decision: the billing_store uses the postgres driver."),
        user_entry(
            "u1",
            "Correction: the billing_store uses the sqlite driver instead.",
        ),
        assistant_entry("a2", "Decision: the audit_store runs on the mysql cluster."),
        user_entry(
            "u2",
            "Correction: the billing_store uses the sqlite driver for the audit_store as well now.",
        ),
    ]);
    let first_decision = record_with_text(&ledger, "postgres driver").clone();
    let second_decision = record_with_text(&ledger, "mysql cluster").clone();
    let first_correction = record_with_text(&ledger, "sqlite driver instead").clone();
    let second_correction = record_with_text(&ledger, "audit_store as well").clone();
    // The first correction retired the first decision (token pairing).
    assert_eq!(first_decision.status, RecordStatus::Superseded);
    assert_eq!(first_decision.link.as_deref(), Some(first_correction.id.as_str()));
    assert_eq!(first_correction.status, RecordStatus::Current);
    // The later correction retires the CURRENT decision by the token path,
    // even though the sibling correction is the higher-scoring token match.
    assert_eq!(second_decision.status, RecordStatus::Superseded);
    assert_eq!(
        second_decision.link.as_deref(),
        Some(second_correction.id.as_str()),
        "the decision is the target, not the sibling correction"
    );
    assert_eq!(second_decision.retire_path, Some(SupersessionPath::Token));
    assert_eq!(second_correction.link.as_deref(), Some(second_decision.id.as_str()));
    assert_eq!(second_correction.retire_path, Some(SupersessionPath::Token));
    // W9-F4 stays dead: no correction is superseded, no mutual pairing.
    assert_eq!(
        first_correction.status,
        RecordStatus::Current,
        "a correction must never retire a sibling correction while a decision qualifies"
    );
    assert!(ledger
        .records
        .iter()
        .filter(|record| record.kind == TypedInformationKind::UserCorrection)
        .all(|record| record.status == RecordStatus::Current));
    assert_eq!(
        ledger
            .records
            .iter()
            .filter(|record| record.status == RecordStatus::Superseded)
            .count(),
        2
    );
    let render = render_hot_state(&ledger);
    assert!(verify(&render, &ledger).passed);
}

// ---------------------------------------------------------------------------
// Attack 7: D28 + D26 (native compaction over a malformed prior ledger)
// ---------------------------------------------------------------------------

/// A native provider compaction whose previous cycle persisted a malformed
/// `details.typedState` payload. The salvaged records must ride through
/// the native path (D28 carry-through), and the D26 diagnostics must not
/// vanish with them: the native episode observed the malformation, so
/// `details.typedVerification.malformedPriorLedger` must carry the repair
/// exactly as the text-summarizer path does.
#[tokio::test]
async fn native_compaction_reports_the_salvage_of_a_malformed_prior_ledger() {
    let first = build_typed_state(&[user_entry(
        "u1",
        "CONSTRAINT: never delete the payments ledger ARCH-W48-KEEP.",
    )]);
    let mut payload = serde_json::to_value(&first).expect("serialize the typed payload");
    payload["records"][0]["ordinal"] = json!("zero");
    let details = json!({"typedState": payload});

    let entries = vec![
        user_entry("u1", "CONSTRAINT: never delete the payments ledger ARCH-W48-KEEP."),
        compaction_marker("c1", "u2", &details),
        user_entry("u2", "CONSTRAINT: always keep the changelog frozen ARCH-2-CHANGELOG."),
        user_entry("u3", "status update: second cycle filler content for the window."),
        user_entry("u4", "status update: more second cycle filler content for the window."),
    ];
    let preparation =
        prepare_compaction(&entries, &typed_settings(), &|_| 40_000.0).expect("prepare");
    // The malformed prior payload was observed and salvaged on the way in.
    assert!(preparation
        .typed_state
        .as_ref()
        .and_then(|state| state.salvage.as_ref())
        .is_some());

    let model = native_fixture("rot2-w48-native-salvage");
    let context = Context::default();
    let result = compact(
        &preparation,
        &model,
        "unused",
        None,
        None,
        None,
        default_summary_call_runner(None),
        None,
        Some((&context, None)),
    )
    .await
    .expect("the native episode must succeed");
    let details = serde_json::to_value(result.details.as_ref().expect("details"))
        .expect("serialize details");

    // The salvaged durable record rides through the native path.
    let persisted = details
        .get("typedState")
        .and_then(|value| value.get("records"))
        .and_then(|records| records.as_array())
        .expect("the native path persists the typed ledger");
    assert!(
        persisted
            .iter()
            .any(|record| record["text"]
                .as_str()
                .is_some_and(|text| text.contains("ARCH-W48-KEEP"))),
        "the salvaged record carries through the native boundary"
    );

    // The D26 diagnostics must surface on the native path too.
    let malformed = details
        .get("typedVerification")
        .and_then(|value| value.get("malformedPriorLedger"))
        .expect("the native episode observed the malformed prior payload");
    assert_eq!(malformed["repairedFields"], json!(["records[0].ordinal"]));
    assert_eq!(malformed["droppedRecords"], json!(0));
}

/// A compaction-capable fixture API (same shape as rot2_native_typed_chain.rs):
/// every native call answers with a checkpoint the kernel accepts.
fn native_fixture(api: &str) -> Model {
    register_api_provider_simple(
        ApiProviderSimple {
            api: api.into(),
            stream: Arc::new(|_, _, _| panic!("unexpected base stream")),
            stream_simple: Arc::new(|_, _, _| panic!("unexpected text summarizer call")),
            compact: Some(Arc::new(|model, _context, _options| {
                let model = model.clone();
                Box::pin(async move {
                    Ok(Some(ProviderCompactionResult {
                        checkpoint: ProviderCompactionCheckpoint {
                            api: model.api.clone(),
                            base_url: "https://fixture.invalid".into(),
                            endpoint: None,
                            estimated_tokens: 64.0,
                            items: vec![json!({
                                "role": "assistant",
                                "content": "server-side compaction of the cycle window",
                            })
                            .as_object()
                            .unwrap()
                            .clone()],
                            model: model.id.clone(),
                            provider: model.provider.clone(),
                            version: 1,
                        },
                        usage: None,
                    }))
                })
                    as Pin<Box<dyn std::future::Future<Output = Result<Option<ProviderCompactionResult>, CompactionRequestError>> + Send>>
            })),
            supports_compaction: Some(Arc::new(|_| true)),
        },
        None,
    );
    let mut model = Model::new(api, api, api, "fixture", "https://fixture.invalid");
    model.context_window = 1_000_000.0;
    model.max_tokens = 64_000.0;
    model
}

// ---------------------------------------------------------------------------
// Attack 8: D22 render direction through a supersession chain
// ---------------------------------------------------------------------------

/// A supersession chain (A retired by B, B retired by C). The rendered
/// "(superseded by)" clause on A must point at its DIRECT superseder B, not
/// at the current record C: the render teaches the summarizer the actual
/// pairing, and the chain stays walkable. Task succession builds the chain
/// (three user task statements retire their predecessors in order).
#[test]
fn superseded_chain_renders_the_direct_superseder() {
    let ledger = build_typed_state(&[
        user_entry("u1", "TASK GOAL: build the parser service."),
        user_entry("u2", "TASK GOAL: build the validator service."),
        user_entry("u3", "TASK GOAL: build the exporter service."),
    ]);
    let first_task = record_with_text(&ledger, "parser service").clone();
    let second_task = record_with_text(&ledger, "validator service").clone();
    let third_task = record_with_text(&ledger, "exporter service").clone();
    for record in [&first_task, &second_task, &third_task] {
        assert_eq!(record.kind, TypedInformationKind::CurrentTask);
    }
    // The id-level chain: A <- B <- C (the W35 attack-4 shape).
    assert_eq!(first_task.status, RecordStatus::Superseded);
    assert_eq!(first_task.link.as_deref(), Some(second_task.id.as_str()));
    assert_eq!(second_task.status, RecordStatus::Superseded);
    assert_eq!(second_task.link.as_deref(), Some(third_task.id.as_str()));
    assert_eq!(third_task.status, RecordStatus::Current);
    assert!(third_task.link.is_none());

    // The render pins the direction: A names B, B names C, and neither
    // names the wrong record. A superseded line naming C would invert or
    // collapse the chain (the D22 defect shape).
    let render = render_hot_state(&ledger);
    let line_of = |id: &str| -> String {
        render
            .lines()
            .find(|line| line.contains(&format!("[{id}]")))
            .unwrap_or_else(|| panic!("no rendered line for {id}"))
            .to_string()
    };
    let first_line = line_of(&first_task.id);
    assert!(
        first_line.contains(&format!("(superseded by [{}])", second_task.id)),
        "A must name its direct superseder B: {first_line}"
    );
    assert!(
        !first_line.contains(&third_task.id),
        "A must not name the current record C: {first_line}"
    );
    let second_line = line_of(&second_task.id);
    assert!(
        second_line.contains(&format!("(superseded by [{}])", third_task.id)),
        "B must name its direct superseder C: {second_line}"
    );
    let third_line = line_of(&third_task.id);
    assert!(
        !third_line.contains("superseded by"),
        "the current record renders as current: {third_line}"
    );
    assert!(third_line.contains("[@"));
    // The chain render verifies: each superseded line is presented together
    // with the record that retired it.
    let verification = verify(&render, &ledger);
    assert!(verification.passed, "findings: {verification:?}");
}

// ---------------------------------------------------------------------------
// Attack 9: D33 restatement absorption strands the absorbed record's refs
// ---------------------------------------------------------------------------

/// The E5-RM TASK-E cycle-2 shape (W47 offline cell,
/// research/context_rot_r2/live/e5/rm/): three telemetry tool lines and a
/// multiline failing-suite result in window one. The failing line
/// ("test result: 55 passed; 1 failed", a 4-token content key) is a subset
/// restatement of each telemetry line's 8-token key, so reconcile absorbs
/// it into the first telemetry slot: that slot is the sentinel (D24 keeps
/// its id, the tool text becomes the failing line), and the content-derived
/// ids reproduce the field observation exactly (VT-1cfcd1 sentinel,
/// VT-25d467 tests_01, VT-174e04 tests_02). The cycle-one render cites all
/// three. At the cycle-two merge the carried tests_01 record is now a
/// superset restatement of the sentinel's failing-line text, so reconcile
/// absorbs it into the sentinel: the absorbed id and source entry leave the
/// ledger. Before the D33 fix exactly [VT-25d467] and [@t01] failed
/// unresolved-ref resolution; the fix retires them into retired_refs, and a
/// cycle-two summary that preserves the cycle-one citations verifies clean.
#[test]
fn absorption_then_citation_resolves() {
    let tests_00_line = "cargo test --lib telemetry::tests_00: test result: 18 passed; 0 failed";
    let tests_01_line = "cargo test --lib telemetry::tests_01: test result: 18 passed; 0 failed";
    let tests_02_line = "cargo test --lib telemetry::tests_02: test result: 18 passed; 0 failed";
    let first = build_typed_state(&[
        tool_entry("t00", tests_00_line),
        tool_entry("t01", tests_01_line),
        tool_entry("t02", tests_02_line),
        tool_entry(
            "t03",
            "cargo test --lib\n\
             running 56 tests\n\
             test result: 55 passed; 1 failed\n\
             \n\
             failures:\n\
             \n\
             ---- parser::tests::parse_nested stdout ----\n\
             thread 'parser::tests::parse_nested' panicked at src/parser.rs:198:\n\
             assertion failed: depth == 4096 (got 3072)\n\
             note: run with `RUST_BACKTRACE=1` for a full backtrace",
        ),
    ]);
    // Window one: the failing line absorbed into the tests_00 slot (the
    // sentinel holds it under the id derived from the telemetry shape),
    // and tests_01/tests_02 stay separate records.
    let sentinel = record_with_text(&first, "55 passed; 1 failed").clone();
    assert_eq!(sentinel.kind, TypedInformationKind::VerifiedToolResult);
    assert_eq!(sentinel.id, "VT-1cfcd1");
    assert_eq!(sentinel.source_entry_id, "t00");
    let tests_01 = record_with_text(&first, "telemetry::tests_01").clone();
    let tests_02 = record_with_text(&first, "telemetry::tests_02").clone();
    assert_eq!(tests_01.id, "VT-25d467");
    assert_ne!(tests_02.id, sentinel.id);
    let render_one = render_hot_state(&first);
    assert!(render_one.contains(&format!("- [{}] [@t01]", tests_01.id)));
    assert!(verify(&render_one, &first).passed);

    // Cycle two: the merge absorbs the carried tests_01 record into the
    // sentinel (its 8-token key supersets the sentinel's 4-token failing
    // key). The sentinel keeps its id and takes the tests_01 text (D24);
    // the absorbed record leaves the ledger as a tombstone.
    let merged = merge(
        Some(&first),
        build_typed_state(&[user_entry(
            "u9",
            "status update: second window filler content.",
        )]),
    );
    let sentinel_two = merged
        .records
        .iter()
        .find(|record| record.id == sentinel.id)
        .expect("the sentinel keeps its id across the merge (D24)")
        .clone();
    assert_eq!(sentinel_two.text, tests_01_line);
    assert!(merged
        .records
        .iter()
        .any(|record| record.id == tests_02.id));
    assert!(!merged.records.iter().any(|record| record.id == tests_01.id));
    assert!(merged.retired_refs.contains(&tests_01.id));
    assert!(merged.retired_refs.contains(&"t01".to_string()));

    // A cycle-two summary that preserves the cycle-one citations (the
    // E5-RM live shape) verifies clean: both absorbed references resolve
    // through the tombstones.
    let verification = verify(&render_one, &merged);
    assert!(
        verification.unresolved_refs.is_empty(),
        "the absorbed record's citations must resolve: {verification:?}"
    );
    assert!(verification.passed, "{verification:?}");
    let citation = format!("Prior evidence: [{}] [@t01] {}", tests_01.id, tests_01_line);
    let verification = verify(&citation, &merged);
    assert!(
        verification.unresolved_refs.is_empty(),
        "the isolated absorbed citations must resolve: {verification:?}"
    );
    assert!(verification.passed, "{verification:?}");

    // A third cycle carries the tombstones forward (the E5-RM chain had
    // zero stranded references from cycle three on), and the cycle-two
    // render still verifies against it.
    let merged_three = merge(
        Some(&merged),
        build_typed_state(&[user_entry(
            "u10",
            "status update: third window filler content.",
        )]),
    );
    assert!(merged_three.retired_refs.contains(&tests_01.id));
    assert!(merged_three.retired_refs.contains(&"t01".to_string()));
    let render_two = render_hot_state(&merged);
    let verification = verify(&render_two, &merged_three);
    assert!(verification.passed, "findings: {verification:?}");
}
