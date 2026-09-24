//! CONTROL-lane host tests for `pi_coding_agent::core::jev_control`.
//!
//! Pure state-machine + durable-accounting tests: no transport, no SystemOne
//! calls, no provider mocks. They pin the ROOT CONTRACT v1 host semantics and
//! the ROOT-CONTROL-ACCOUNTING-BLOCKERS repairs:
//! - real-user-task epochs activate ONLY for interactive/rpc sources with the
//!   full profile active; internal steers, synthesized inputs, resumes and
//!   compaction never move the epoch and never write the ledger;
//! - budgets survive eviction and restart through the bounded JSONL ledger,
//!   recovered by the PRODUCTION snapshot/consume path (no reload helper),
//!   and never leak across sessions;
//! - every spend is reserved and committed durably under the per-ledger
//!   transaction lock BEFORE an effect is authorized; interleaved writers
//!   never lose writes;
//! - a failed/corrupt/missing/truncated/oversized/above-max ledger refuses
//!   new effects (fail-closed, zero headroom) instead of minting maxima;
//! - monotonic ledger-wide sequence authority: reordered records cannot
//!   restore older higher remaining counts;
//! - once-per-attempt veto consults are durable across books;
//! - compaction bounds drop sessions only into fail-closed refusal, never a
//!   re-initialized active budget;
//! - a decision bound to an older expected epoch can never spend whichever
//!   epoch happens to be current.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use pi_ai::types::{
    AssistantMessage, ContentBlock, ImageOrTextContent, TextContent, ToolCall, ToolResultMessage,
};
use pi_agent_core::types::CustomMessageContent;
use pi_coding_agent::core::jev_control::{
    eligible_questions, nonprogress_feedback, resolve_agent_end, resolve_retry_veto,
    resolve_turn_end, result_gap_feedback, turn_signature, verification_missing_feedback,
    ControlBook, VetoConsultState, CONTROL_FEEDBACK_CUSTOM_TYPE, LEDGER_RECORD_VERSION,
    MAX_LEDGER_BYTES, MAX_TRACKED_SESSIONS, REAL_USER_INPUT_SOURCES,
};
use pi_jev::active::AnswerCandidate;
use pi_jev::config::JevMode;
use pi_jev::control::{
    ControlBudgetKind, ControlBudgetSnapshot, ControlFeatures, ControlPolicy, ControlRefusal,
    ControlVerificationState, ControlVerdict, FeedbackKind, HostControlFacts,
    NonprogressVerdict, PauseReason,
};
use pi_jev::observation::RetryFailureKind;
use pi_jev::snapshot::{SnapshotStage, StateSnapshot};
use pi_jev::types::DecisionCategory;
use serde_json::{json, Map};

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_book(tag: &str) -> (ControlBook, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "jev-control-host-{}-{}-{}",
        tag,
        std::process::id(),
        DIR_SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::create_dir_all(&dir);
    (ControlBook::new(dir.clone()), dir)
}

fn ledger_path(dir: &Path) -> PathBuf {
    dir.join("jev").join("control-budgets.jsonl")
}

fn ledger_lines(dir: &Path) -> Vec<String> {
    std::fs::read_to_string(ledger_path(dir))
        .expect("ledger must exist")
        .lines()
        .map(str::to_string)
        .collect()
}

fn write_ledger(dir: &Path, body: &str) {
    write_ledger_bytes(dir, body.as_bytes());
}

fn write_ledger_bytes(dir: &Path, bytes: &[u8]) {
    let path = ledger_path(dir);
    std::fs::create_dir_all(path.parent().expect("ledger parent")).expect("writable temp dir");
    std::fs::write(path, bytes).expect("write crafted ledger");
}

/// One valid v2 record line with explicit fields (crafted-ledger tests).
fn record_line(
    session: &str,
    epoch: u64,
    seq: u64,
    feedback: u8,
    verification: u8,
    nonprogress: u8,
    veto: u8,
    consults: &[u32],
) -> String {
    json!({
        "v": LEDGER_RECORD_VERSION,
        "session": session,
        "epoch": epoch,
        "seq": seq,
        "anchor": "",
        "feedback": feedback,
        "verification": verification,
        "nonprogress": nonprogress,
        "veto": veto,
        "consults": consults,
    })
    .to_string()
}

fn assert_unavailable(snapshot: &ControlBudgetSnapshot) {
    assert!(
        !snapshot.available,
        "snapshot must be fail-closed zero headroom"
    );
    assert_eq!(snapshot.feedback_remaining, 0);
    assert_eq!(snapshot.verification_remaining, 0);
    assert_eq!(snapshot.nonprogress_remaining, 0);
    assert_eq!(snapshot.veto_remaining, 0);
}

fn features() -> ControlFeatures {
    ControlFeatures {
        result_sufficiency: true,
        loop_control: true,
        verification: true,
        retry_classification: true,
        full_jev_active: true,
    }
}

fn facts(session: &str, turn: u64, question_ids: &[&str]) -> HostControlFacts {
    HostControlFacts {
        now: SystemTime::now(),
        session_id: session.to_string(),
        turn,
        expected_question_ids: question_ids.iter().map(|id| id.to_string()).collect(),
        policy_generation: "gen-1".to_string(),
        full_jev_stamp: "stamp-1".to_string(),
        prompt_version: "jev-control-prompts/1".to_string(),
        epoch_id: format!("{session}:1"),
        request_id: "req-1".to_string(),
    }
}

fn candidate(
    category: DecisionCategory,
    question_id: &str,
    value: &str,
    confidence: f64,
    turn: u64,
) -> AnswerCandidate {
    AnswerCandidate {
        category,
        question_id: question_id.to_string(),
        value: Some(value.to_string()),
        confidence: Some(confidence),
        response_model: Some("test-model".to_string()),
        request_id: "req-1".to_string(),
        turn,
        // Stamped strictly BEFORE the helper-built facts.now (facts() runs
        // first in every test): decided_at must be in the past of the host
        // clock or freshness would refuse with Stale by construction.
        decided_at: SystemTime::now() - std::time::Duration::from_millis(100),
    }
}

fn assistant_with_tool_call(name: &str, command: &str) -> AssistantMessage {
    let mut arguments = Map::new();
    arguments.insert("command".to_string(), json!(command));
    AssistantMessage {
        content: vec![ContentBlock::ToolCall(ToolCall::new("call-1", name, arguments))],
        ..Default::default()
    }
}

fn tool_result(name: &str, body: &str, is_error: bool) -> ToolResultMessage {
    ToolResultMessage::new(
        "call-1",
        name,
        vec![ImageOrTextContent::Text(TextContent::new(body))],
        is_error,
        1,
    )
}

#[test]
fn real_user_sources_constant_matches_contract() {
    assert_eq!(REAL_USER_INPUT_SOURCES, ["interactive", "rpc"]);
}

#[test]
fn epochs_bump_only_for_real_user_sources() {
    let (book, dir) = temp_book("epochs");

    let first = book.note_real_user_input("sess", "interactive", "do task A", true);
    assert_eq!(first.epoch_id, "sess:1");
    assert_eq!(first.feedback_remaining, 2);
    assert!(first.available);

    // Internal steers, extension inputs, resumes and compaction never bump
    // and never write the ledger.
    let lines_before = ledger_lines(&dir).len();
    for source in ["extension", "internal", "synthesized", "compaction", ""] {
        let unchanged = book.note_real_user_input("sess", source, "steer", true);
        assert_eq!(unchanged.epoch_id, "sess:1", "source {source} must not bump");
    }
    assert_eq!(
        ledger_lines(&dir).len(),
        lines_before,
        "internal inputs must not append ledger records"
    );

    // Consumption inside the epoch is preserved across non-user inputs.
    book.consume("sess", "sess:1", ControlBudgetKind::NonprogressCorrection)
        .expect("first correction allowed");
    let after_steers = book.note_real_user_input("sess", "internal", "resume", true);
    assert_eq!(after_steers.epoch_id, "sess:1");
    assert_eq!(after_steers.nonprogress_remaining, 0);

    // The next real user task starts a fresh epoch with fresh budgets.
    let second = book.note_real_user_input("sess", "rpc", "do task B", true);
    assert_eq!(second.epoch_id, "sess:2");
    assert_eq!(second.feedback_remaining, 2);
    assert_eq!(second.nonprogress_remaining, 1);
    assert_eq!(second.veto_remaining, 2);

    // The stale epoch-1 binding is dead after B's delivery.
    assert!(book
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());

    // Empty session ids never create state.
    let empty = book.note_real_user_input("", "interactive", "x", true);
    assert!(!empty.available);
    assert_eq!(empty.epoch_id, "");

    // Inactive full profile: no ledger is written at all (H-CONTROL-1).
    let (off_book, off_dir) = temp_book("epochs-off");
    let off = off_book.note_real_user_input("sess", "interactive", "task", false);
    assert!(!off.available);
    assert!(!ledger_path(&off_dir).exists(), "no control ledger when full is off");
}

#[test]
fn budgets_persist_across_book_recreation_without_reload_helper() {
    let (book, dir) = temp_book("persist");
    book.note_real_user_input("sess", "interactive", "task", true);
    let after_first = book
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .expect("first feedback allowed");
    assert_eq!(after_first.feedback_remaining, 1);
    book.consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .expect("first veto allowed");

    // A new book over the same agent dir (restart/eviction) rebuilds through
    // the PRODUCTION snapshot/consume path — no reload helper exists anymore.
    let reborn = ControlBook::new(dir.clone());
    let restored = reborn.snapshot("sess");
    assert!(restored.available);
    assert_eq!(restored.epoch_id, "sess:1");
    assert_eq!(restored.feedback_remaining, 1);
    assert_eq!(restored.veto_remaining, 1);

    // Consumption continues from the restored values and persists again.
    let again = reborn
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .expect("second feedback allowed");
    assert_eq!(again.feedback_remaining, 0);
    assert!(reborn
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .is_none());
    // The first book observes the durable truth on its next transaction.
    assert!(book
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .is_none());
}

#[test]
fn budgets_never_leak_across_sessions() {
    let (book, _dir) = temp_book("isolation");
    book.note_real_user_input("sess-a", "interactive", "task A", true);
    let initial_b = book.note_real_user_input("sess-b", "interactive", "task B", true);
    book.consume("sess-a", "sess-a:1", ControlBudgetKind::RetryVeto)
        .expect("veto budget available");
    book.consume("sess-a", "sess-a:1", ControlBudgetKind::RetryVeto)
        .expect("second veto budget available");

    let b = book.snapshot("sess-b");
    assert_eq!(b.epoch_id, initial_b.epoch_id);
    assert_eq!(b.veto_remaining, 2);
    assert_eq!(b.feedback_remaining, 2);
    assert!(book
        .consume("sess-a", "sess-a:1", ControlBudgetKind::RetryVeto)
        .is_none());
    assert!(book
        .consume("sess-b", &b.epoch_id, ControlBudgetKind::RetryVeto)
        .is_some());
}

#[test]
fn two_books_interleave_spends_without_lost_writes() {
    let (book_a, dir) = temp_book("two-store");
    let book_b = ControlBook::new(dir.clone());
    book_a.note_real_user_input("sess", "interactive", "task", true);

    // Interleaved consumes across two independent books (two stores/processes
    // simulated): each transaction reads the current durable state under the
    // ledger lock, so no writer can lose or decrease another's spends.
    let a1 = book_a
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .expect("veto budget available");
    assert_eq!(a1.veto_remaining, 1);
    let b1 = book_b
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .expect("second veto from the other store");
    assert_eq!(b1.veto_remaining, 0, "book B must see book A's durable spend");
    assert!(book_a
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());
    assert!(book_b
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());

    let b2 = book_b
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .expect("feedback from the other store");
    assert_eq!(b2.feedback_remaining, 1);
    let a2 = book_a
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .expect("second feedback");
    assert_eq!(a2.feedback_remaining, 0);

    // Durable truth afterwards: book A's fresh cache agrees, and book B (whose
    // in-memory cache may lag behind another writer) still reconciles through
    // the durable transaction on its next spend.
    assert_eq!(book_a.snapshot("sess").feedback_remaining, 0);
    assert!(book_b
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .is_none());
}

#[test]
fn held_lock_fails_closed_and_is_never_stolen() {
    let (book, dir) = temp_book("held-lock");
    book.note_real_user_input("sess", "interactive", "task", true);
    let lock_path = ledger_path(&dir).with_extension("jsonl.lock");
    assert!(
        lock_path.exists(),
        "stable lock file persists after transactions"
    );
    let ledger_before = std::fs::read(ledger_path(&dir)).expect("ledger bytes");

    // A live external holder (simulated second writer/process) keeps the OS
    // lock on the STABLE lock file. The book must fail closed immediately:
    // no steal, no mtime takeover, no unlink, no sleeping retry loop.
    let holder = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .expect("open stable lock file");
    holder
        .try_lock()
        .expect("external holder acquires the OS lock");

    assert!(book
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());
    assert_eq!(
        book.veto_consult_state("sess", 1),
        VetoConsultState::Unavailable
    );
    // The refused transactions did not touch the ledger.
    assert_eq!(
        std::fs::read(ledger_path(&dir)).expect("ledger bytes"),
        ledger_before
    );

    // Releasing the holder restores transactions; the lock file was never
    // unlinked (stable inode) so no successor lock can be stolen.
    drop(holder);
    assert!(book
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_some());
    assert!(
        lock_path.exists(),
        "stable lock file must never be unlinked"
    );
}

#[test]
fn veto_consult_is_durable_across_books() {
    let (book, dir) = temp_book("veto-consult");
    book.note_real_user_input("sess", "interactive", "task", true);
    assert_eq!(book.veto_consult_state("sess", 1), VetoConsultState::Fresh);
    assert_eq!(book.veto_consult_state("sess", 1), VetoConsultState::Repeated);
    assert!(book.note_veto_consult("sess", 2));

    // Consult identity survives eviction/restart (durable record).
    let reborn = ControlBook::new(dir.clone());
    assert_eq!(reborn.veto_consult_state("sess", 1), VetoConsultState::Repeated);
    assert_eq!(reborn.veto_consult_state("sess", 2), VetoConsultState::Repeated);
    assert_eq!(reborn.veto_consult_state("sess", 3), VetoConsultState::Fresh);

    // A new real user task starts a fresh consult set.
    reborn.note_real_user_input("sess", "interactive", "task 2", true);
    assert_eq!(reborn.veto_consult_state("sess", 1), VetoConsultState::Fresh);

    // Without durable accounting the consult refuses (baseline proceeds).
    let (broken, broken_dir) = temp_book("veto-consult-broken");
    std::fs::create_dir_all(ledger_path(&broken_dir)).expect("make ledger path a directory");
    assert_eq!(
        broken.veto_consult_state("sess", 1),
        VetoConsultState::Unavailable
    );
}

#[test]
fn nonprogress_verdicts_over_recorded_signatures() {
    let (book, _dir) = temp_book("nonprogress");
    book.note_real_user_input("sess", "interactive", "task", true);
    let identical = assistant_with_tool_call("bash", "ls");
    book.record_turn_signature("sess", &identical, &[], true);
    assert_eq!(book.nonprogress("sess"), NonprogressVerdict::None);
    book.record_turn_signature("sess", &identical, &[], true);
    assert_eq!(
        book.nonprogress("sess"),
        NonprogressVerdict::Candidate {
            identical_turns: 2
        }
    );
    // New evidence breaks the chain (same tool name, different content).
    let changed = assistant_with_tool_call("bash", "ls -la");
    book.record_turn_signature("sess", &changed, &[], true);
    assert_eq!(book.nonprogress("sess"), NonprogressVerdict::None);

    // Guard: with the full profile inactive nothing is recorded at all.
    let (off_book, _off_dir) = temp_book("nonprogress-off");
    off_book.record_turn_signature("sess", &identical, &[], false);
    off_book.record_turn_signature("sess", &identical, &[], false);
    assert_eq!(off_book.nonprogress("sess"), NonprogressVerdict::None);
}

#[test]
fn turn_signature_digests_content_not_names() {
    let same_args = turn_signature(&assistant_with_tool_call("bash", "ls"), &[]);
    let same_again = turn_signature(&assistant_with_tool_call("bash", "ls"), &[]);
    assert_eq!(same_args, same_again);

    // Same tool name, different arguments: different signature.
    let different_args = turn_signature(&assistant_with_tool_call("bash", "ls -la"), &[]);
    assert_ne!(same_args, different_args);

    // Different tool name: different signature.
    let different_name = turn_signature(&assistant_with_tool_call("ipython", "ls"), &[]);
    assert_ne!(same_args, different_name);

    // Result content enters the digest, including the error flag.
    let with_result = turn_signature(
        &assistant_with_tool_call("bash", "ls"),
        &[tool_result("bash", "file-a", false)],
    );
    let other_result = turn_signature(
        &assistant_with_tool_call("bash", "ls"),
        &[tool_result("bash", "file-b", false)],
    );
    assert_ne!(with_result, other_result);
    let error_result = turn_signature(
        &assistant_with_tool_call("bash", "ls"),
        &[tool_result("bash", "file-a", true)],
    );
    assert_ne!(with_result, error_result);

    // Text and thinking content change the text digest; empty turns are
    // equal to each other but never identical to content turns.
    let empty_a = turn_signature(&AssistantMessage::default(), &[]);
    let empty_b = turn_signature(&AssistantMessage::default(), &[]);
    assert_eq!(empty_a, empty_b);
    assert_ne!(empty_a, same_args);
}

#[test]
fn resolve_agent_end_defers_on_insufficient_results() {
    let (book, _dir) = temp_book("agent-end-gap");
    book.note_real_user_input("sess", "interactive", "task", true);
    let fact = facts("sess", 7, &["result_sufficiency.0"]);
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );

    let result = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[insufficient.clone()],
        false,
        "turns=2, tool_results=1",
    );
    assert!(result.feedback.is_some());
    assert!(result.defer_goal_finish);
    assert_eq!(result.verification_state, ControlVerificationState::Unknown);
    assert_eq!(book.snapshot("sess").feedback_remaining, 1);

    // A different decision in the SAME epoch is a repeated insistence on one
    // logical task: at most one result-gap continuation per user task
    // (approved 2026-09-24). The answer is respected; still-insufficient
    // non-terminal work stays deferred and truthfully paused, never
    // finished as complete.
    let mut next_facts = fact.clone();
    next_facts.request_id = "req-2".to_string();
    let mut next_insufficient = insufficient.clone();
    next_insufficient.request_id = next_facts.request_id.clone();
    let second = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &next_facts,
        &[next_insufficient],
        false,
        "turns=3",
    );
    assert!(second.feedback.is_none());
    // Deduped feedback never finishes still-insufficient work: truthful
    // paused state, goal stays deferred, no success or verification claim.
    assert!(second.defer_goal_finish);
    assert_eq!(second.pause, Some(PauseReason::BudgetExhausted));
    assert_eq!(second.verification_state, ControlVerificationState::Unknown);
    assert_eq!(second.terminal_annotation, Some("duplicate_control_feedback"));
    assert!(second.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::TriggerNotMet("duplicate_control_feedback"))
    )));
    // The refusal spends nothing: the shared budget keeps its remaining unit
    // for the one verification request the epoch may still deliver.
    assert_eq!(book.snapshot("sess").feedback_remaining, 1);

    // The replay of the ORIGINAL decision also stays refused (same state).
    let replay = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[insufficient],
        false,
        "turns=2, tool_results=1",
    );
    assert!(replay.feedback.is_none());
    assert!(!replay.defer_goal_finish);
    assert!(replay.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::TriggerNotMet("duplicate_control_feedback"))
    )));
    assert_eq!(book.snapshot("sess").feedback_remaining, 1);

    // A genuinely exhausted budget (direct consumption) still refuses with
    // the truthful budget annotation, never a silent success.
    let spent = book.consume(
        "sess",
        &book.snapshot("sess").epoch_id,
        ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
    );
    assert!(spent.is_some());
    assert_eq!(book.snapshot("sess").feedback_remaining, 0);
    let mut third_facts = fact.clone();
    third_facts.request_id = "req-3".to_string();
    let mut third_insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    third_insufficient.request_id = third_facts.request_id.clone();
    let third = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &third_facts,
        &[third_insufficient],
        false,
        "turns=4",
    );
    assert!(third.feedback.is_none());
    assert!(!third.defer_goal_finish);
    assert_eq!(third.terminal_annotation, Some("budget_exhausted_unverified"));
    assert!(third.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::BudgetExhausted(_))
    )));

    // A pending continuation owns the task: no second feedback. Fresh
    // session so the feedback budget is available and the pending-continuation
    // refusal is exercised on its own.
    let pending_budget = book.note_real_user_input("sess-pending", "interactive", "task", true);
    let mut pending_facts = facts("sess-pending", 7, &["result_sufficiency.0"]);
    pending_facts.epoch_id = pending_budget.epoch_id;
    let pending_insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let pending = resolve_agent_end(
        &book,
        "sess-pending",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &pending_facts,
        &[pending_insufficient],
        true,
        "turns=5",
    );
    assert!(pending.feedback.is_none());
    assert!(!pending.defer_goal_finish);
    assert!(pending.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::ContinuationPending)
    )));
    // Budget untouched by the refused path.
    assert_eq!(book.snapshot("sess-pending").feedback_remaining, 2);
}

#[test]
fn resolve_agent_end_verification_request_and_pause_fallback() {
    let (book, _dir) = temp_book("agent-end-verify");
    book.note_real_user_input("sess", "interactive", "task", true);
    let fact = facts("sess", 9, &["first_pass_verification.0"]);
    let verify = candidate(
        DecisionCategory::FirstPassVerification,
        "first_pass_verification.0",
        "verify",
        0.9,
        9,
    );

    // Budget-1 verification request: feedback continuation, honest Unverified.
    let first = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[verify.clone()],
        false,
        "no tests observed",
    );
    assert!(first.feedback.is_some());
    assert!(first.defer_goal_finish);
    assert_eq!(first.verification_state, ControlVerificationState::Unverified);
    assert_eq!(book.snapshot("sess").verification_remaining, 0);

    // A different recommendation cannot be delivered: truthful pause.
    let mut next_facts = fact.clone();
    next_facts.request_id = "req-2".to_string();
    let mut verify = verify;
    verify.request_id = next_facts.request_id.clone();
    let second = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &next_facts,
        &[verify],
        false,
        "still no tests",
    );
    assert!(second.feedback.is_none());
    assert_eq!(second.pause, Some(PauseReason::VerificationUnconfirmed));
    assert!(second.defer_goal_finish);
    assert_eq!(second.verification_state, ControlVerificationState::Unverified);
    assert_eq!(second.terminal_annotation, Some("verification_unconfirmed"));
}

#[test]
fn ctrl001_repeated_insufficiency_in_one_epoch_never_reopens_the_answer() {
    // The observed recurrence pattern: an approval-required or read-only
    // answer is restated on each automatic continuation. One result-gap
    // correction per logical task epoch (approved 2026-09-24); the restated
    // answer is respected, the duplicate refusal spends nothing, no success
    // or verification is ever claimed, and still-insufficient non-terminal
    // work never finishes merely because feedback was deduped: the goal
    // stays deferred with a truthful paused state.
    let (book, _dir) = temp_book("epoch-repeat-gap");
    let budget = book.note_real_user_input("sess", "interactive", "Diagnose only; do not start the repair", true);
    let mut fact = facts("sess", 7, &["result_sufficiency.0"]);
    fact.epoch_id = budget.epoch_id.clone();
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );

    // First agent end: the single corrective continuation is delivered.
    let first = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &[insufficient.clone()], false, "turns=7");
    assert!(first.feedback.is_some());
    assert!(first.defer_goal_finish);

    // The queued continuation answers at a NEW turn: same epoch, new decision.
    let mut continued = fact.clone();
    continued.turn = 8;
    continued.request_id = "req-continued".to_string();
    let mut restated = insufficient.clone();
    restated.request_id = continued.request_id.clone();
    restated.turn = 8;
    let second = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &continued, &[restated], false, "turns=8");
    assert!(second.feedback.is_none(), "no second insistence on one logical task");
    assert!(second.defer_goal_finish, "still-insufficient work is never finished by dedupe");
    assert_eq!(second.pause, Some(PauseReason::BudgetExhausted));
    assert_eq!(second.verification_state, ControlVerificationState::Unknown);
    assert_eq!(second.terminal_annotation, Some("duplicate_control_feedback"));
    assert_eq!(book.snapshot("sess").feedback_remaining, 1);

    // A genuinely new real-user task re-arms the budget: fresh correction.
    // Correlation requires candidate.turn == facts.turn AND
    // candidate.request_id == facts.request_id: the third decision keeps them
    // in sync (a mismatch refuses the candidate with Baseline(NoAnswer) before
    // the epoch re-arm path is ever exercised — that was the C8s failure, a
    // test defect, not a product defect in the epoch gate).
    let reborn = book.note_real_user_input("sess", "interactive", "New task: implement it", true);
    let mut fresh = facts("sess", 7, &["result_sufficiency.0"]);
    fresh.epoch_id = reborn.epoch_id.clone();
    fresh.request_id = "req-new-epoch".to_string();
    let mut fresh_insufficient = insufficient.clone();
    fresh_insufficient.request_id = fresh.request_id.clone();
    let third = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fresh, &[fresh_insufficient], false, "turns=9");
    assert!(third.feedback.is_some(), "a new logical task may be corrected again");
}

#[test]
fn ctrl001_duplicate_result_gap_maps_not_required_without_success_claim() {
    // When the epoch-duplicate refusal fires and the evaluator itself
    // recommended no verification, verification maps to NotApplicable (no
    // verification needed) — that speaks ONLY to verification, never to task
    // sufficiency. Still-insufficient work stays deferred and paused; the
    // state is never Verified and never a success claim.
    let (book, _dir) = temp_book("epoch-repeat-not-applicable");
    let budget = book.note_real_user_input("sess", "interactive", "Status only", true);
    let mut fact = facts("sess", 7, &["result_sufficiency.0", "first_pass_verification.0"]);
    fact.epoch_id = budget.epoch_id.clone();
    let gap = candidate(DecisionCategory::ResultSufficiency, "result_sufficiency.0", "insufficient", 0.9, 7);
    let no_verification = candidate(DecisionCategory::FirstPassVerification, "first_pass_verification.0", "none", 0.9, 7);

    let first = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &[gap.clone(), no_verification.clone()], false, "turns=7");
    assert!(first.feedback.is_some());

    let mut continued = fact.clone();
    continued.turn = 8;
    continued.request_id = "req-continued".to_string();
    let mut restated_gap = gap.clone();
    restated_gap.request_id = continued.request_id.clone();
    restated_gap.turn = 8;
    let mut restated_none = no_verification.clone();
    restated_none.request_id = continued.request_id.clone();
    restated_none.turn = 8;
    let second = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &continued, &[restated_gap, restated_none], false, "turns=8");
    assert!(second.feedback.is_none());
    assert!(second.defer_goal_finish);
    assert_eq!(second.pause, Some(PauseReason::BudgetExhausted));
    assert_eq!(second.verification_state, ControlVerificationState::NotApplicable);
    assert_eq!(second.terminal_annotation, Some("duplicate_control_feedback"));
    assert_eq!(book.snapshot("sess").feedback_remaining, 1);
}

#[test]
fn resolve_agent_end_not_applicable_for_normal_non_test_tasks() {
    let (book, _dir) = temp_book("agent-end-na");
    book.note_real_user_input("sess", "interactive", "answer a question", true);
    let fact = facts(
        "sess",
        9,
        &["result_sufficiency.0", "first_pass_verification.0"],
    );
    let sufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "sufficient",
        0.9,
        9,
    );
    let no_verification = candidate(
        DecisionCategory::FirstPassVerification,
        "first_pass_verification.0",
        "none",
        0.9,
        9,
    );
    let result = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[sufficient, no_verification],
        false,
        "turns=1, tool_results=0",
    );
    assert!(result.feedback.is_none());
    assert!(!result.defer_goal_finish);
    assert_eq!(result.pause, None);
    assert_eq!(result.verification_state, ControlVerificationState::NotApplicable);
    // No budget was touched.
    assert_eq!(book.snapshot("sess").feedback_remaining, 2);
}

#[test]
fn resolve_agent_end_escalate_defers_and_flags() {
    let (book, _dir) = temp_book("agent-end-escalate");
    book.note_real_user_input("sess", "interactive", "task", true);
    let fact = facts("sess", 9, &["continue_stop_escalate.0"]);
    let escalate = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "escalate",
        0.9,
        9,
    );
    let result = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[escalate],
        false,
        "turns=2",
    );
    assert!(result.escalate);
    assert!(result.defer_goal_finish);
    assert!(result.feedback.is_none());
}

#[test]
fn resolve_agent_end_gates_closed_outside_full_profile() {
    let (book, _dir) = temp_book("agent-end-gates");
    book.note_real_user_input("sess", "interactive", "task", true);
    let fact = facts("sess", 7, &["result_sufficiency.0"]);
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let mut gated = features();
    gated.full_jev_active = false;
    let result = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &gated,
        JevMode::Active,
        &fact,
        &[insufficient],
        false,
        "turns=2",
    );
    assert!(result.feedback.is_none());
    assert!(!result.defer_goal_finish);
    assert!(result.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::GatesClosed(_))
    )));
}

#[test]
fn resolve_agent_end_accounting_unavailable_refuses_effects() {
    let (book, dir) = temp_book("agent-end-unavailable");
    // The ledger path is a DIRECTORY: every durable read/write fails.
    std::fs::create_dir_all(ledger_path(&dir)).expect("make ledger path a directory");
    let snap = book.note_real_user_input("sess", "interactive", "task", true);
    assert_unavailable(&snap);

    let fact = facts("sess", 7, &["result_sufficiency.0"]);
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let result = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[insufficient],
        false,
        "turns=2",
    );
    // No unaccounted effect, no refill, no deferred success loop.
    assert!(result.feedback.is_none());
    assert!(!result.defer_goal_finish);
    assert_eq!(result.terminal_annotation, Some("accounting_unavailable"));
    assert!(result.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::AccountingUnavailable(_))
    )));

    // TurnEnd gate refuses the same way (no stop, no correction).
    let turn_facts = facts("sess", 8, &["continue_stop_escalate.0"]);
    let stop = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "stop",
        0.9,
        8,
    );
    let turn_result = resolve_turn_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &turn_facts,
        &[stop],
        &NonprogressVerdict::Candidate {
            identical_turns: 2,
        },
    );
    assert!(!turn_result.stop_loop);
    assert!(turn_result.feedback.is_none());
    assert!(turn_result.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::AccountingUnavailable(_))
    )));

    // Consume and veto consults refuse while accounting is unavailable.
    assert!(book
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());

    // Repaired ledger + restart: a genuine real delivered user task alone
    // initializes a new budget (no refill of the uncommitted epoch).
    std::fs::remove_dir_all(ledger_path(&dir)).expect("repair ledger path");
    let reborn = ControlBook::new(dir);
    let fresh = reborn.note_real_user_input("sess", "interactive", "task", true);
    assert!(fresh.available);
    assert_eq!(fresh.epoch_id, "sess:1");
    assert_eq!(fresh.feedback_remaining, 2);
    assert!(reborn
        .consume("sess", "sess:1", ControlBudgetKind::NonprogressCorrection)
        .is_some());
}

#[test]
fn resolve_turn_end_correction_then_pause() {
    let (book, _dir) = temp_book("turn-end");
    book.note_real_user_input("sess", "interactive", "task", true);
    let fact = facts("sess", 4, &["continue_stop_escalate.0"]);
    let stop = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "stop",
        0.9,
        4,
    );
    let candidate_verdict = NonprogressVerdict::Candidate {
        identical_turns: 2,
    };

    // No nonprogress trigger: nothing happens.
    let idle = resolve_turn_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[],
        &NonprogressVerdict::None,
    );
    assert!(idle.feedback.is_none());
    assert!(!idle.stop_loop);
    assert!(idle.verdicts.iter().any(|verdict| matches!(
        verdict,
        ControlVerdict::Refused(ControlRefusal::TriggerNotMet(_))
    )));

    // First nonprogress: the bounded correction feedback, not a pause.
    let first = resolve_turn_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[stop.clone()],
        &candidate_verdict,
    );
    assert!(first.feedback.is_some());
    assert!(!first.stop_loop);
    assert!(!first.escalate);
    assert_eq!(book.snapshot("sess").nonprogress_remaining, 0);

    // Correction spent: an accepted stop now pauses truthfully.
    let second = resolve_turn_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[stop.clone()],
        &candidate_verdict,
    );
    assert!(second.feedback.is_none());
    assert!(second.stop_loop);

    // Escalate still lands even with the correction budget spent.
    let escalate = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "escalate",
        0.9,
        4,
    );
    let third = resolve_turn_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[escalate],
        &candidate_verdict,
    );
    assert!(third.stop_loop);
    assert!(third.escalate);
}

#[test]
fn resolve_retry_veto_is_bounded_and_once_per_attempt() {
    let (book, _dir) = temp_book("retry-veto");
    book.note_real_user_input("sess", "interactive", "task", true);
    let fact = facts("sess", 2, &["retry_classification.0"]);
    let fatal = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "fatal",
        0.9,
        2,
    );
    let transient = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "transient",
        0.9,
        2,
    );

    // Transient failures never veto: baseline retries proceed.
    let transient_verdict = resolve_retry_veto(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[transient.clone()],
        1,
    );
    assert!(!transient_verdict.veto);
    assert_eq!(transient_verdict.classification, None);
    assert_eq!(transient_verdict.reason, "baseline");

    // Fatal at the high floor vetoes the host-planned attempt.
    let fatal_verdict = resolve_retry_veto(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[fatal.clone()],
        2,
    );
    assert!(fatal_verdict.veto);
    assert_eq!(fatal_verdict.classification, Some(RetryFailureKind::Fatal));

    // Second veto of the epoch is allowed (max 2)...
    let second = resolve_retry_veto(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[fatal.clone()],
        3,
    );
    assert!(second.veto);
    assert_eq!(book.snapshot("sess").veto_remaining, 0);

    // ...the third is refused: the veto can never extend ceilings.
    let third = resolve_retry_veto(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[fatal],
        4,
    );
    assert!(!third.veto);
    assert_eq!(third.reason, "veto_budget_exhausted");

    // Same attempt ordinal cannot be consulted twice.
    let repeat = resolve_retry_veto(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[transient],
        4,
    );
    assert_eq!(repeat.reason, "already_consulted");
}

#[test]
fn stale_decision_cannot_spend_the_new_epoch() {
    let (book, _dir) = temp_book("queued-b");
    // Task A delivered (epoch 1) and spends one feedback continuation.
    book.note_real_user_input("sess", "interactive", "task A", true);
    book.consume(
        "sess",
        "sess:1",
        ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
    )
    .expect("A feedback allowed");

    // Task B delivered while A is still finishing: epoch activates at B's
    // delivery with fresh budgets; A's in-flight decision keeps epoch 1.
    book.note_real_user_input("sess", "rpc", "task B", true);
    let b = book.snapshot("sess");
    assert_eq!(b.epoch_id, "sess:2");
    assert_eq!(b.feedback_remaining, 2);

    // A's reservation is bound to the expected epoch: refused, and B's fresh
    // budget is untouched by the stale attempt.
    assert!(book
        .consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .is_none());
    assert_eq!(book.snapshot("sess").feedback_remaining, 2);

    // Resolver level: facts captured pre-dispatch say epoch 1; the current
    // budget says epoch 2; correlation refuses and nothing is applied.
    let fact = facts("sess", 7, &["result_sufficiency.0"]);
    assert_eq!(fact.epoch_id, "sess:1");
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let result = resolve_agent_end(
        &book,
        "sess",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &fact,
        &[insufficient],
        false,
        "stale decision",
    );
    assert!(result.feedback.is_none());
    assert!(!result.defer_goal_finish);
    assert_eq!(book.snapshot("sess").feedback_remaining, 2);
}

#[test]
fn missing_or_corrupt_or_oversized_records_fail_closed() {
    // Durable state vanished after a spend: refuse new effects, never refill.
    {
        let (book, dir) = temp_book("missing-after-spend");
        book.note_real_user_input("sess", "interactive", "task", true);
        book.consume(
            "sess",
            "sess:1",
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
        )
        .expect("feedback allowed");
        std::fs::remove_file(ledger_path(&dir)).expect("remove ledger");
        let reborn = ControlBook::new(dir);
        assert_unavailable(&reborn.snapshot("sess"));
        assert!(reborn
            .consume(
                "sess",
                "sess:1",
                ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
            )
            .is_none());
    }

    // The ENTIRE file is refused when any part of it is invalid: a healthy
    // record plus a corrupt newer line must not restore the healthy (older)
    // record into authority, and malformed consults must not silently erase
    // consult trust. Oversized files are refused whole (no tail salvage).
    let seventeen: Vec<u32> = (1..=17).collect();
    let healthy = record_line("sess", 1, 1, 2, 1, 1, 2, &[]);
    let cases: Vec<(&str, String)> = vec![
        (
            "corrupt",
            "this is not json\n{\"v\": 2, \"session\": \"sess\", \"epoch\": 1, \"seq\": \"x\"\n"
                .to_string(),
        ),
        ("truncated", {
            let full = record_line("sess", 1, 1, 2, 1, 1, 2, &[]);
            let cut = &full[..full.len() * 3 / 5];
            format!("{cut}{cut}")
        }),
        (
            "above-max",
            record_line("sess", 1, 1, 200, 9, 1, 2, &[]),
        ),
        // A valid older record followed by a corrupt newer one: the whole
        // set is refused; the older record is NOT salvaged into authority.
        (
            "healthy-plus-corrupt",
            format!("{healthy}\nthis line is not json\n"),
        ),
        // Malformed or missing consult arrays erase trust => refused.
        (
            "missing-consults",
            json!({
                "v": LEDGER_RECORD_VERSION,
                "session": "sess",
                "epoch": 1,
                "seq": 1,
                "anchor": "",
                "feedback": 2,
                "verification": 1,
                "nonprogress": 1,
                "veto": 2,
            })
            .to_string(),
        ),
        (
            "consults-not-array",
            json!({
                "v": LEDGER_RECORD_VERSION,
                "session": "sess",
                "epoch": 1,
                "seq": 1,
                "anchor": "",
                "feedback": 2,
                "verification": 1,
                "nonprogress": 1,
                "veto": 2,
                "consults": 5,
            })
            .to_string(),
        ),
        (
            "consults-duplicate",
            record_line("sess", 1, 1, 2, 1, 1, 2, &[1, 1]),
        ),
        (
            "consults-oversized",
            record_line("sess", 1, 1, 2, 1, 1, 2, &seventeen),
        ),
    ];
    for (tag, body) in cases {
        let (book, dir) = temp_book(tag);
        write_ledger(&dir, &body);
        assert_unavailable(&book.snapshot("sess"));
        assert!(book
            .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
            .is_none());
        assert!(!book.note_veto_consult("sess", 1));
    }

    // Invalid UTF-8 anywhere in the file refuses the whole ledger.
    {
        let (book, dir) = temp_book("invalid-utf8");
        let mut bytes = healthy.into_bytes();
        bytes.extend_from_slice(b"\n\xff\xfe broken bytes\n");
        write_ledger_bytes(&dir, &bytes);
        assert_unavailable(&book.snapshot("sess"));
        assert!(book
            .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
            .is_none());
    }

    // Oversized ledger (beyond the byte cap): refused ENTIRELY — including a
    // valid newest record at the tail. No bounded-read salvage into authority.
    {
        let (book, dir) = temp_book("oversized-garbage");
        let garbage = format!(
            "{}\n{}\n{}\n",
            "x".repeat(120_000),
            "x".repeat(120_000),
            "x".repeat(120_000)
        );
        write_ledger(&dir, &garbage);
        assert!(
            ledger_path(&dir).metadata().expect("ledger size").len()
                > MAX_LEDGER_BYTES as u64
        );
        assert_unavailable(&book.snapshot("sess"));
    }
    {
        let (book, dir) = temp_book("oversized-valid-tail");
        let body = format!(
            "{}\n{}\n{}\n{}\n",
            record_line("sess", 1, 1, 2, 1, 1, 2, &[]),
            "x".repeat(150_000),
            "x".repeat(150_000),
            record_line("sess", 1, 2, 1, 1, 1, 2, &[]),
        );
        write_ledger(&dir, &body);
        assert!(
            ledger_path(&dir).metadata().expect("ledger size").len()
                > MAX_LEDGER_BYTES as u64
        );
        // Even though the file ENDS with a valid record, an oversized ledger
        // is refused whole: no partial authority.
        assert_unavailable(&book.snapshot("sess"));
        assert!(book
            .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
            .is_none());
    }

    // A record for a different session is never applied to this one (and a
    // valid record IS attributed to its own session only).
    {
        let (book, dir) = temp_book("wrong-session");
        write_ledger(&dir, &record_line("other-session", 1, 1, 2, 1, 1, 2, &[]));
        assert_unavailable(&book.snapshot("sess"));
        assert!(book
            .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
            .is_none());
        let foreign = book.snapshot("other-session");
        assert!(foreign.available);
        assert_eq!(foreign.epoch_id, "other-session:1");
        assert_eq!(foreign.feedback_remaining, 2);
    }
}

#[test]
fn conflicting_duplicate_records_fail_closed() {
    let (book, dir) = temp_book("duplicate-records");
    // Two records for the SAME session with conflicting state: our writer
    // emits exactly one current record per session, so a duplicate record set
    // means the file cannot be trusted. It must fail closed ENTIRELY instead
    // of picking the newer seq or the higher headroom (no rollback refund,
    // no salvage of an older valid record over a corrupt newer one).
    let spent = record_line("sess", 1, 7, 0, 0, 0, 0, &[1, 2]);
    let stale_higher = record_line("sess", 1, 3, 2, 1, 1, 2, &[]);
    write_ledger(&dir, &format!("{spent}\n{stale_higher}\n"));

    let reborn = ControlBook::new(dir);
    assert_unavailable(&reborn.snapshot("sess"));
    assert!(reborn
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());
    assert_eq!(
        reborn.veto_consult_state("sess", 1),
        VetoConsultState::Unavailable
    );
    // Even initialization refuses while the durable file is untrusted: no
    // continuation maxima can be minted from it.
    let refused_init = reborn.note_real_user_input("sess", "interactive", "task again", true);
    assert_unavailable(&refused_init);
}

#[test]
fn compaction_bounds_drop_only_into_fail_closed() {
    let (book, dir) = temp_book("compaction");
    let mut epochs = Vec::new();
    for index in 0..(MAX_TRACKED_SESSIONS as u64 + 6) {
        let session = format!("sess-{index}");
        let initial = book.note_real_user_input(&session, "interactive", "task", true);
        assert!(initial.available);
        epochs.push((session, initial.epoch_id));
    }
    let (latest_session, latest_epoch) = epochs.last().unwrap();
    book.consume(latest_session, latest_epoch, ControlBudgetKind::RetryVeto)
        .expect("veto available");
    let lines = ledger_lines(&dir);
    assert!(
        lines.len() <= MAX_TRACKED_SESSIONS,
        "ledger must stay bounded, got {}",
        lines.len()
    );

    // The most recent session recovers truthfully after restart.
    let reborn = ControlBook::new(dir.clone());
    let latest = reborn.snapshot(latest_session);
    assert!(latest.available);
    assert_eq!(&latest.epoch_id, latest_epoch);
    assert_eq!(latest.veto_remaining, 1);

    // A session dropped by the bound is fail-closed, NEVER re-minted.
    let dropped = reborn.snapshot("sess-0");
    assert_unavailable(&dropped);
    assert!(reborn
        .consume("sess-0", "sess-0:1", ControlBudgetKind::RetryVeto)
        .is_none());

    // Restart removes any lucky surviving cache entry: a new real user
    // delivery must never reuse the evicted task's identity.
    drop(book);
    let renewed = reborn.note_real_user_input("sess-0", "interactive", "task again", true);
    assert!(renewed.available);
    assert_ne!(renewed.epoch_id, epochs[0].1);
    assert_eq!(renewed.feedback_remaining, 2);
    assert!(reborn
        .consume("sess-0", &epochs[0].1, ControlBudgetKind::RetryVeto)
        .is_none());
    let stale_facts = facts("sess-0", 7, &["result_sufficiency.0"]);
    let result = resolve_agent_end(
        &reborn,
        "sess-0",
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        &stale_facts,
        &[candidate(
            DecisionCategory::ResultSufficiency,
            "result_sufficiency.0",
            "insufficient",
            0.9,
            7,
        )],
        false,
        "stale decision after eviction",
    );
    assert!(result.feedback.is_none());
    assert!(!result.defer_goal_finish);
    assert_eq!(reborn.snapshot("sess-0").feedback_remaining, 2);
    let restarted = ControlBook::new(dir);
    assert_eq!(
        restarted.current_epoch_id("sess-0"),
        Some(renewed.epoch_id.clone())
    );
    assert!(restarted
        .consume("sess-0", &renewed.epoch_id, ControlBudgetKind::RetryVeto)
        .is_some());
}

#[test]
fn new_session_survives_a_full_ledger_of_updated_sessions() {
    let (book, dir) = temp_book("ledger-high-water");
    let mut previous_seq = 0;
    for index in 0..MAX_TRACKED_SESSIONS {
        let session = format!("busy-{index}");
        let initial = book.note_real_user_input(&session, "interactive", "task", true);
        assert!(initial.available);
        book.consume(&session, &initial.epoch_id, ControlBudgetKind::RetryVeto)
            .expect("first veto");
        assert_eq!(book.veto_consult_state(&session, 1), VetoConsultState::Fresh);
        let row = ledger_lines(&dir)
            .into_iter()
            .map(|line| serde_json::from_str::<serde_json::Value>(&line).unwrap())
            .find(|row| row["session"] == session)
            .unwrap();
        let seq = row["seq"].as_u64().unwrap();
        assert_eq!(
            seq,
            previous_seq + 3,
            "activation, spend and consult each advance the ledger"
        );
        previous_seq = seq;
    }
    let reborn = ControlBook::new(dir.clone());
    let fresh = reborn.note_real_user_input("new-session", "rpc", "new task", true);
    assert!(
        fresh.available,
        "an updated older session must not evict the new write"
    );
    assert_eq!(fresh.epoch_id, format!("new-session:{}", previous_seq + 1));
    assert_eq!(ledger_lines(&dir).len(), MAX_TRACKED_SESSIONS);
    assert_unavailable(&reborn.snapshot("busy-0"));
}

#[test]
fn exhausted_ledger_sequence_refuses_mutations_without_reusing_identity() {
    let (book, dir) = temp_book("sequence-exhausted");
    write_ledger(&dir, &record_line("sess", 1, u64::MAX, 2, 1, 1, 2, &[]));
    let before = std::fs::read(ledger_path(&dir)).unwrap();
    assert!(book
        .consume("sess", "sess:1", ControlBudgetKind::RetryVeto)
        .is_none());
    assert_eq!(book.veto_consult_state("sess", 1), VetoConsultState::Unavailable);
    assert_unavailable(&book.note_real_user_input("new-session", "rpc", "task", true));
    assert_unavailable(&book.note_real_user_input("sess", "rpc", "new task", true));
    assert_eq!(std::fs::read(ledger_path(&dir)).unwrap(), before);
}

#[test]
fn ledger_restores_many_sessions_across_restart() {
    let (book, dir) = temp_book("restart-many");
    let mut epochs = Vec::new();
    // Many sessions, one consumption each: the ledger must rebuild each
    // session's freshest snapshot independently after a restart.
    for epoch in 0..40u64 {
        let session = format!("sess-{epoch}");
        let initial = book.note_real_user_input(&session, "interactive", "task", true);
        book.consume(
            &session,
            &initial.epoch_id,
            ControlBudgetKind::RetryVeto,
        )
        .expect("veto budget available");
        epochs.push(initial.epoch_id);
    }
    let reborn = ControlBook::new(dir);
    let restored = reborn.snapshot("sess-39");
    assert_eq!(restored.epoch_id, epochs[39]);
    assert_eq!(restored.veto_remaining, 1);
    let untouched = reborn.snapshot("sess-0");
    assert_eq!(untouched.epoch_id, "sess-0:1");
    assert_eq!(untouched.veto_remaining, 1);
    assert_eq!(untouched.feedback_remaining, 2);
    // Consumption continues without any reload helper.
    assert!(reborn
        .consume("sess-0", "sess-0:1", ControlBudgetKind::RetryVeto)
        .is_some());
}

#[test]
fn eligible_questions_mirror_legacy_gating() {
    let state = json!({
        "features": {
            "result_sufficiency": true,
            "loop_control": true,
            "first_pass_verification": true,
            "retry_classification": true
        },
        "result_excerpt": "the change was applied",
        "observation": {
            "turns": 3,
            "tool_results": 2,
            "failure_kind": "rate_limited"
        }
    });
    let snapshot =
        StateSnapshot::new(SnapshotStage::AgentEnd, "sess", 7, 1, None, state, vec![])
            .expect("bounded snapshot");

    let all_on = features();
    let questions = eligible_questions(&snapshot, &all_on, true);
    assert!(questions
        .iter()
        .any(|question| question.question_id.starts_with("result_sufficiency.")));
    assert!(questions
        .iter()
        .any(|question| question.question_id.starts_with("retry_classification.")));

    // Feature off in the host truth removes the category, regardless of the
    // snapshot state.
    let mut sufficiency_off = features();
    sufficiency_off.result_sufficiency = false;
    let questions = eligible_questions(&snapshot, &sufficiency_off, true);
    assert!(!questions
        .iter()
        .any(|question| question.question_id.starts_with("result_sufficiency.")));

    // No failure observed: retry classification stays out of the bundle.
    let questions = eligible_questions(&snapshot, &all_on, false);
    assert!(!questions
        .iter()
        .any(|question| question.question_id.starts_with("retry_classification.")));
}

#[test]
fn ctrl001_terminal_scope_wins_without_claiming_checks_passed() {
    for task in ["Thanks", "Report status only", "Give me the link", "List the files", "Stop work"] {
        let (book, _dir) = temp_book("terminal-scope");
        book.note_real_user_input("sess", "interactive", task, true);
        let fact = facts("sess", 9, &["result_sufficiency.0", "first_pass_verification.0", "continue_stop_escalate.0"]);
        let decisions = [
            candidate(DecisionCategory::ResultSufficiency, "result_sufficiency.0", "insufficient", 0.9, 9),
            candidate(DecisionCategory::FirstPassVerification, "first_pass_verification.0", "none", 0.9, 9),
            candidate(DecisionCategory::ContinueStopEscalate, "continue_stop_escalate.0", "stop", 0.9, 9),
        ];
        let result = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &decisions, false, "verification=Unknown");
        assert!(result.feedback.is_none(), "{task}");
        assert!(!result.defer_goal_finish);
        assert_eq!(result.verification_state, ControlVerificationState::NotApplicable);
        assert_eq!(book.snapshot("sess").feedback_remaining, 2);
    }
}

#[test]
fn ctrl001_blocked_or_stopped_implementation_never_claims_verified() {
    for (scope, escalation) in [("stop", false), ("escalate", true)] {
        let (book, _dir) = temp_book("blocked-scope");
        book.note_real_user_input("sess", "interactive", "Implement the change", true);
        let fact = facts("sess", 9, &["first_pass_verification.0", "continue_stop_escalate.0"]);
        let decisions = [
            candidate(DecisionCategory::FirstPassVerification, "first_pass_verification.0", "verify", 0.9, 9),
            candidate(DecisionCategory::ContinueStopEscalate, "continue_stop_escalate.0", scope, 0.9, 9),
        ];
        let result = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &decisions, false, "verification=Unknown");
        assert!(result.feedback.is_none());
        assert_eq!(result.escalate, escalation);
        assert_eq!(result.verification_state, ControlVerificationState::Unknown);
        assert_eq!(book.snapshot("sess").verification_remaining, 1);
    }
}

#[test]
fn ctrl001_duplicate_notices_survive_restart_without_spending_again() {
    for (category, id, value) in [
        (DecisionCategory::ResultSufficiency, "result_sufficiency.0", "insufficient"),
        (DecisionCategory::FirstPassVerification, "first_pass_verification.0", "verify"),
    ] {
        let (book, dir) = temp_book("notice-dedupe");
        book.note_real_user_input("sess", "interactive", "Implement and check", true);
        let fact = facts("sess", 9, &[id]);
        let decisions = [candidate(category, id, value, 0.9, 9)];
        let first = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &decisions, false, "unknown");
        assert!(first.feedback.is_some());
        let remaining = book.snapshot("sess");
        let reborn = ControlBook::new(dir);
        let duplicate = resolve_agent_end(&reborn, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &decisions, false, "unknown");
        assert!(duplicate.feedback.is_none());
        assert!(duplicate.pause.is_none());
        assert!(!duplicate.defer_goal_finish);
        assert_eq!(reborn.snapshot("sess"), remaining);
        assert!(duplicate.verdicts.iter().any(|verdict| matches!(verdict,
            ControlVerdict::Refused(ControlRefusal::TriggerNotMet("duplicate_control_feedback")))));
    }
}


#[test]
fn ctrl001_duplicate_feedback_never_suppresses_a_new_terminal_safety_decision() {
    for (terminal, escalate) in [("escalate", true), ("stop", false)] {
        let (book, dir) = temp_book("dedupe-terminal-safety");
        book.note_real_user_input("sess", "interactive", "Implement and check", true);
        let fact = facts("sess", 9, &["result_sufficiency.0", "continue_stop_escalate.0"]);
        let gap = candidate(DecisionCategory::ResultSufficiency, "result_sufficiency.0", "insufficient", 0.9, 9);
        let first = resolve_agent_end(&book, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &[gap.clone()], false, "unknown");
        assert!(first.feedback.is_some());
        let remaining = book.snapshot("sess");
        let reborn = ControlBook::new(dir);
        let mut safety = candidate(DecisionCategory::ContinueStopEscalate, "continue_stop_escalate.0", terminal, 0.95, 9);
        // Deterministic freshness: stamp decided_at from the matching facts
        // clock (minus 100ms) instead of the later wall clock, so a
        // parallel-load stall between the facts construction and this
        // candidate can never future-stamp decided_at past facts.now (the
        // C8 parallel-only escalate failure was exactly that wall-clock
        // skew; freshness itself is unchanged).
        safety.decided_at = fact.now - std::time::Duration::from_millis(100);
        let result = resolve_agent_end(&reborn, "sess", &ControlPolicy::default(), &features(), JevMode::Active, &fact, &[gap, safety], false, "concrete blocker or stop");
        assert!(result.feedback.is_none());
        assert_eq!(result.escalate, escalate);
        assert_eq!(result.defer_goal_finish, escalate);
        assert_eq!(result.verification_state, ControlVerificationState::Unknown);
        assert_eq!(reborn.snapshot("sess"), remaining);
        if !escalate {
            assert_eq!(result.terminal_annotation, Some("no_authorized_follow_up"));
        }
    }
}

#[test]
fn ctrl001_new_task_drops_old_nonprogress_evidence() {
    let (book, _dir) = temp_book("task-scope");
    book.note_real_user_input("sess", "interactive", "task A", true);
    let message = assistant_with_tool_call("ipython", "print('same')");
    book.record_turn_signature("sess", &message, &[], true);
    book.record_turn_signature("sess", &message, &[], true);
    assert!(matches!(book.nonprogress("sess"), NonprogressVerdict::Candidate { .. }));
    book.note_real_user_input("sess", "interactive", "task B", true);
    assert_eq!(book.nonprogress("sess"), NonprogressVerdict::None);
}

#[test]
fn feedback_templates_are_fixed_host_text() {
    let gap = result_gap_feedback("insufficient", "turns=2, tool_results=1");
    assert_eq!(gap.custom_type, CONTROL_FEEDBACK_CUSTOM_TYPE);
    assert!(gap.display);
    assert_eq!(gap.role, "custom");
    assert!(gap.timestamp > 0);
    match &gap.content {
        CustomMessageContent::Text(body) => {
            assert!(body.contains("not yet complete"));
            assert!(body.contains("insufficient"));
            assert!(body.contains("turns=2"));
        }
        CustomMessageContent::Blocks(_) => panic!("feedback must be plain text"),
    }

    let verification = verification_missing_feedback("no tests observed");
    match &verification.content {
        CustomMessageContent::Text(body) => assert!(body.contains("unreported, not failed")),
        CustomMessageContent::Blocks(_) => panic!("feedback must be plain text"),
    }

    let nonprogress = nonprogress_feedback(2);
    match &nonprogress.content {
        CustomMessageContent::Text(body) => assert!(body.contains("2 turns")),
        CustomMessageContent::Blocks(_) => panic!("feedback must be plain text"),
    }
}
