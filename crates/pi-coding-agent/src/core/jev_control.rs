//! Host adapter for applied bounded control (full-jev profile, ROOT CONTRACT v1).
//!
//! This module is CONTROL-lane owned. It owns:
//! - the REAL-user-task epoch budget state machine (contract: budgets are
//!   maxima per real user task; they must NOT reset on internal steer,
//!   synthesized input, automatic resume, compaction, or eviction/recreation
//!   of an active task; state never leaks across sessions);
//! - bounded, content-signature-based nonprogress detection (repeated tool
//!   NAME alone never counts; a single reasoning-only/empty turn is never
//!   nonprogress);
//! - host-authored FIXED feedback templates (Jev never authors conversation
//!   text, tool arguments, or success claims). Each feedback KIND is
//!   delivered at most once per real-user task epoch (approved 2026-09-24):
//!   a repeated insistence on the same logical task refuses as
//!   `duplicate_control_feedback`, never reopens the answer, and — while
//!   results remain insufficient for non-terminal work — keeps the goal
//!   deferred with a truthful paused state instead of finishing it, while
//!   a repeated verification need keeps the truthful `Unverified` pause;
//! - the gate resolvers the integrator's call sites use to turn accepted Jev
//!   answers into typed effects at safe boundaries.
//!
//! Accounting durability (ROOT-CONTROL-ACCOUNTING-BLOCKERS C-BUDGET-1..3):
//! budgets are reserved and committed durably under a per-ledger transaction
//! lock BEFORE any effect is authorized; snapshots never mint maxima (a
//! session without trusted durable accounting is zero-headroom and control
//! effects stay refused while the ordinary baseline continues); records are
//! bounded, content-free, validated, and selected by a monotonic ledger-wide
//! sequence instead of timestamps.
//!
//! It deliberately does NOT call the observer/bridge deeper than the
//! integration points in reports/API-HANDOFF.md: the integrator builds the
//! question bundle, obtains the decision, and hands the answers here as
//! `pi_jev::active::AnswerCandidate` values. All settings truth (mode, feature
//! gates, full-jev overlay state) arrives as parameters; nothing is read from
//! decision payloads.
//!
//! Verification honesty (contract + early review C1): no verification state
//! is ever derived from tool transport success. Without a reliable correlated
//! evidence adapter, the only paths are `Unknown`, `NotApplicable` (accepted
//! `none` recommendation), and `Unverified` (a bounded verification request
//! was delivered). `Verified`/`Failed` require
//! [`pi_jev::control::CorrelatedVerificationEvidence`], which no host call
//! site constructs in this batch.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use pi_agent_core::types::CustomMessageContent;
use pi_jev::active::AnswerCandidate;
use pi_jev::config::JevMode;
use pi_jev::control::{
    combine_sufficiency, control_gates_open, evaluate_control_answer, nonprogress_verdict,
    verification_need, ControlBoundary, ControlBudgetKind, ControlBudgetSnapshot, ControlBudgets,
    ControlEffectKind, ControlFeatures, ControlPolicy, ControlRefusal, ControlVerificationState,
    ControlVerdict, FeedbackKind, HostControlFacts, NonprogressVerdict, PauseReason,
    SufficiencyVerdict, TurnSignature, VerificationNeed, CONTROL_NONPROGRESS_WINDOW,
};
use pi_jev::evaluators::{for_boundary, EvaluatorOutput, PreparedQuestion};
use pi_jev::observation::RetryFailureKind;
use pi_jev::snapshot::StateSnapshot;
use pi_jev::types::DecisionCategory;
use serde_json::{json, Value};
use sha2::{Digest as ShaDigest, Sha256};

use crate::config::get_agent_dir;
use crate::core::messages::CustomMessage;

/// Custom message type for control feedback (host-authored, fixed templates).
pub const CONTROL_FEEDBACK_CUSTOM_TYPE: &str = "jevControl";

/// Prompt provenance carried into every acceptance record.
pub const CONTROL_PROMPT_VERSION: &str = "jev-control-prompts/1";

/// Bounded per-session tracking (same bound as the bridge SessionBook).
pub const MAX_TRACKED_SESSIONS: usize = 64;

/// Ledger bounds: the durable write set is compacted to the latest record per
/// session, at most this many lines, and at most [`MAX_LEDGER_BYTES`] bytes.
pub const MAX_LEDGER_LINES: usize = 512;

/// Hard byte bound for the durable ledger. A file larger than this is
/// refused ENTIRELY (no tail salvage, no partial authority); writes are
/// compacted to stay within it.
pub const MAX_LEDGER_BYTES: usize = 256 * 1024;

/// Durable record format version. Older or unknown formats are ignored and
/// leave the session fail-closed (never re-initialized from foreign data).
pub const LEDGER_RECORD_VERSION: u64 = 2;

/// Retry-veto consult ordinals retained per session record. Ordinals only
/// grow within an epoch, so keeping the largest ones cannot re-enable a
/// duplicate consult for a future attempt.
pub const MAX_RECORDED_VETO_CONSULTS: usize = 16;

/// Session-id sanity bound for durable attribution.
pub const MAX_SESSION_ID_CHARS: usize = 256;

/// Maximum characters of a turn's text/thinking/result content entering a
/// digest input (digests never retain content).
pub const MAX_DIGEST_INPUT_CHARS: usize = 400;

/// Input sources that identify a REAL user task (contract: epoch must not
/// move for internal steers, synthesized or extension inputs, resumes, or
/// compaction).
pub const REAL_USER_INPUT_SOURCES: [&str; 2] = ["interactive", "rpc"];

/// Delivery state of one feedback KIND within the current task epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeedbackDeliveryState {
    /// No notice of this kind delivered in this epoch yet.
    New,
    /// This exact decision already delivered its notice (replay guard).
    SameDecision,
    /// Another decision in this epoch already delivered this kind of notice.
    DeliveredInEpoch,
}

/// One session's in-memory control bookkeeping. `budget` is `Some` only when
/// a trusted durable record backs it; the ledger is always the authority and
/// the in-memory copy is a cache that can never mint or refill a budget.
#[derive(Debug, Clone, Default)]
struct SessionControl {
    epoch: u64,
    seq: u64,
    anchor_digest: String,
    budget: Option<ControlBudgetSnapshot>,
    signatures: Vec<TurnSignature>,
    consulted_veto_attempts: Vec<u32>,
}

/// Task-epoch budget state machine with a bounded durable ledger so budgets
/// survive eviction of the in-memory map and process restarts.
///
/// Accounting invariants (ROOT-CONTROL-ACCOUNTING-BLOCKERS C-BUDGET-1..3):
/// - Budgets are NEVER minted. A snapshot without a trusted durable record is
///   zero-headroom (`available: false`) and authorizes no effects; the
///   ordinary baseline continues. Fresh maxima exist only through a durably
///   committed activation of a REAL delivered user task.
/// - Every state change (epoch activation, budget consumption, veto consult)
///   is reserved and committed durably under one per-ledger transaction lock
///   BEFORE the effect is authorized. A failed commit leaves the previous
///   durable state authoritative and reports "no budget"; the in-memory copy
///   never becomes authoritative after a durable failure.
/// - The ledger stores one latest record per session with a monotonic
///   ledger-wide `seq`. Compaction retains the highest `seq` — never a
///   timestamp — so clock rollback or reordered records cannot restore older
///   higher remaining counts.
/// - Records are validated on load (version, session bounds, epoch/seq >= 1,
///   counters <= contract maxima, bounded consult list). Invalid,
///   truncated, oversized, or foreign-session records are ignored and leave
///   the session fail-closed.
/// - Compaction bounds the write set to the freshest sessions. A session
///   dropped by the bound becomes fail-closed on its next access; it is
///   never re-initialized without a genuinely new real user delivery.
pub struct ControlBook {
    sessions: Mutex<HashMap<String, SessionControl>>,
    /// In-process serialization for ledger transactions; the file lock below
    /// covers cross-process writers (same pattern as the cron-jobs state
    /// lock). Lock order: ledger transaction -> sessions mutex. Sessions
    /// critical sections never take the ledger lock.
    io_mutex: Mutex<()>,
    ledger_path: PathBuf,
    lock_path: PathBuf,
}

impl ControlBook {
    pub fn new(agent_dir: impl Into<PathBuf>) -> Self {
        let mut ledger_path: PathBuf = agent_dir.into();
        ledger_path.push("jev");
        ledger_path.push("control-budgets.jsonl");
        let mut lock_path = ledger_path.clone();
        lock_path.set_extension("jsonl.lock");
        Self {
            sessions: Mutex::new(HashMap::new()),
            io_mutex: Mutex::new(()),
            ledger_path,
            lock_path,
        }
    }

    /// Process-global book wired by the integrator at the input boundary.
    pub fn global() -> &'static ControlBook {
        static BOOK: OnceLock<ControlBook> = OnceLock::new();
        BOOK.get_or_init(|| ControlBook::new(get_agent_dir()))
    }

    /// Record a real user input; activates a NEW task epoch ONLY for real
    /// user sources and only when the full profile is active. The fresh
    /// budget is committed durably BEFORE it is returned: when the durable
    /// commit fails the snapshot is zero-headroom and control effects stay
    /// refused (fail-closed) until a later commit succeeds — no refill, no
    /// retry loop. Steers, extension inputs, internal prompts, resumes and
    /// compaction never pass a real user source and never touch the ledger.
    /// When `full_jev_active` is false nothing is hashed, written, or stored.
    pub fn note_real_user_input(
        &self,
        session_id: &str,
        source: &str,
        text: &str,
        full_jev_active: bool,
    ) -> ControlBudgetSnapshot {
        if !full_jev_active {
            return ControlBudgetSnapshot::zero_headroom(String::new());
        }
        if !REAL_USER_INPUT_SOURCES.contains(&source) {
            return self.snapshot(session_id);
        }
        if !valid_session_id(session_id) {
            return ControlBudgetSnapshot::zero_headroom(String::new());
        }
        let anchor = digest_text(text);
        match self.commit_activation(session_id, &anchor) {
            Ok(snapshot) => snapshot,
            Err(()) => {
                // The epoch activation never committed: advance the in-memory
                // counter so the next activation starts a NEW epoch, keep zero
                // budget, and never retry in a loop.
                let epoch = self.advance_uncommitted_epoch(session_id, &anchor);
                ControlBudgetSnapshot::zero_headroom(epoch_id(session_id, epoch))
            }
        }
    }

    /// Current budget snapshot; recovers from the durable ledger when the
    /// in-memory entry was evicted (budgets survive eviction/restart through
    /// the PRODUCTION path — no test helper). NEVER mints maxima: without a
    /// trusted durable record this is zero-headroom (`available: false`).
    pub fn snapshot(&self, session_id: &str) -> ControlBudgetSnapshot {
        if !valid_session_id(session_id) {
            return ControlBudgetSnapshot::zero_headroom(String::new());
        }
        if let Some(trusted) = self.fast_trusted(session_id) {
            return trusted;
        }
        match self.recover(session_id) {
            Ok(Recovery::Trusted(record)) => record.budget_snapshot(),
            Ok(Recovery::Uncommitted) | Ok(Recovery::Missing) | Err(()) => {
                ControlBudgetSnapshot::zero_headroom(epoch_id(
                    session_id,
                    self.in_memory_epoch(session_id),
                ))
            }
        }
    }

    /// Consume one budget unit at acceptance time, bound to the epoch the
    /// decision was made for. The reservation is committed durably (under the
    /// ledger transaction lock) BEFORE `Some` is returned, so no effect is
    /// ever authorized without accounted budget. A stale expected epoch, an
    /// exhausted budget, or any durable failure returns `None` and leaves the
    /// durable state untouched.
    pub fn consume(
        &self,
        session_id: &str,
        expected_epoch_id: &str,
        kind: ControlBudgetKind,
    ) -> Option<ControlBudgetSnapshot> {
        self.consume_with_notices(session_id, expected_epoch_id, kind, &[])
    }

    /// Consume one budget unit, recording every delivered notice key
    /// durably. A key already present refuses the whole consumption (race
    /// safety), so a delivered notice can never be spent twice.
    fn consume_with_notices(
        &self,
        session_id: &str,
        expected_epoch_id: &str,
        kind: ControlBudgetKind,
        notices: &[&str],
    ) -> Option<ControlBudgetSnapshot> {
        if !valid_session_id(session_id) {
            return None;
        }
        let outcome = self.ledger_txn(session_id, true, |records: &mut Vec<DurableRecord>| {
            let entry_epoch = self.in_memory_epoch(session_id);
            let index = match records.iter().position(|record| record.session == session_id) {
                Some(index) => index,
                // No durable accounting: nothing may spend.
                None => return Ok((None, Vec::new(), false)),
            };
            let record = records[index].clone();
            if record.epoch < entry_epoch {
                // The in-memory epoch never committed durably: nothing may
                // spend, and the older record must not be adopted.
                return Ok((None, Vec::new(), false));
            }
            if epoch_id(session_id, record.epoch) != expected_epoch_id {
                // A newer epoch is current (a real user delivered a new
                // task): the stale decision may not spend whichever epoch
                // happens to be current. Refresh the cache from durable
                // truth only.
                if record.epoch > entry_epoch {
                    return Ok((None, vec![record.clone()], false));
                }
                return Ok((None, Vec::new(), false));
            }
            let current = record.budget_snapshot();
            if !current.allows(kind)
                || notices
                    .iter()
                    .any(|key| record.feedback_notices.iter().any(|seen| seen == *key))
            {
                return Ok((None, Vec::new(), false));
            }
            let next = current.consume(kind);
            let mut updated = record.clone();
            updated.seq = next_ledger_sequence(records, 0)?;
            updated.feedback_remaining = next.feedback_remaining;
            updated.verification_remaining = next.verification_remaining;
            updated.nonprogress_remaining = next.nonprogress_remaining;
            updated.veto_remaining = next.veto_remaining;
            for notice in notices {
                updated.feedback_notices.push(notice.to_string());
            }
            records[index] = updated.clone();
            // Cache install happens only after the durable commit succeeds.
            Ok((Some(next), vec![updated], true))
        });
        match outcome {
            Ok(inner) => inner,
            Err(()) => None,
        }
    }

    /// How this feedback KIND stands for the CURRENT logical task (epoch):
    /// - [`FeedbackDeliveryState::New`]: nothing delivered yet in this epoch;
    /// - [`FeedbackDeliveryState::SameDecision`]: THIS exact decision (same
    ///   request and turn) already delivered its notice (replay guard);
    /// - [`FeedbackDeliveryState::DeliveredInEpoch`]: another agent end in
    ///   the same epoch already delivered this KIND of notice. One corrective
    ///   notice per kind per real-user task: repeated insistence on the same
    ///   logical task is refused, never re-queued.
    fn feedback_delivery_state(
        &self,
        session_id: &str,
        facts: &HostControlFacts,
        kind: FeedbackKind,
    ) -> FeedbackDeliveryState {
        let exact_key = feedback_notice_key(facts, kind);
        let epoch_key = feedback_epoch_notice_key(&facts.epoch_id, kind);
        let outcome = self.ledger_txn(session_id, false, |records| {
            let mut state = FeedbackDeliveryState::New;
            for record in records.iter() {
                if record.session == session_id
                    && epoch_id(&record.session, record.epoch) == facts.epoch_id
                {
                    if record.feedback_notices.iter().any(|seen| *seen == exact_key) {
                        state = FeedbackDeliveryState::SameDecision;
                        break;
                    }
                    if record.feedback_notices.iter().any(|seen| *seen == epoch_key) {
                        state = FeedbackDeliveryState::DeliveredInEpoch;
                    }
                }
            }
            Ok((state, Vec::new(), false))
        });
        outcome.unwrap_or(FeedbackDeliveryState::New)
    }

    /// Record one bounded turn signature for the nonprogress predicate. The
    /// guard happens BEFORE any digest work: when the full profile is
    /// inactive nothing is hashed, recorded, or persisted (H-CONTROL-1).
    /// `has_new_evidence` is computed at record time (differs from the
    /// previous recorded signature; the first turn of a session differs from
    /// nothing). Signatures are in-memory only: losing them to eviction can
    /// only delay a nonprogress pause, never reset a budget.
    pub fn record_turn_signature(
        &self,
        session_id: &str,
        message: &pi_ai::types::AssistantMessage,
        tool_results: &[pi_ai::types::ToolResultMessage],
        full_jev_active: bool,
    ) {
        if !full_jev_active || !valid_session_id(session_id) {
            return;
        }
        let mut signature = turn_signature(message, tool_results);
        let mut sessions = self.lock_sessions();
        let entry = self.entry_or_evict(&mut sessions, session_id);
        signature.has_new_evidence = match entry.signatures.last() {
            Some(previous) => {
                previous.tool_calls_digest != signature.tool_calls_digest
                    || previous.text_digest != signature.text_digest
                    || previous.results_digest != signature.results_digest
            }
            None => false,
        };
        entry.signatures.push(signature);
        let excess = entry.signatures.len().saturating_sub(CONTROL_NONPROGRESS_WINDOW);
        if excess > 0 {
            entry.signatures.drain(0..excess);
        }
    }

    /// Current nonprogress verdict over the recorded signatures.
    pub fn nonprogress(&self, session_id: &str) -> NonprogressVerdict {
        let sessions = self.lock_sessions();
        match sessions.get(session_id) {
            Some(entry) => nonprogress_verdict(&entry.signatures),
            None => NonprogressVerdict::None,
        }
    }

    /// Idempotency guard for the retry veto consult: one consult per attempt
    /// ordinal, DURABLY recorded so a restart or eviction cannot consult the
    /// same attempt twice into a duplicate veto. Returns false when this
    /// attempt was already consulted or when durable accounting is
    /// unavailable.
    pub fn note_veto_consult(&self, session_id: &str, attempt: u32) -> bool {
        matches!(
            self.veto_consult_state(session_id, attempt),
            VetoConsultState::Fresh
        )
    }

    /// Consult state with truthful unavailability reporting.
    pub fn veto_consult_state(&self, session_id: &str, attempt: u32) -> VetoConsultState {
        if !valid_session_id(session_id) || attempt == 0 {
            return VetoConsultState::Unavailable;
        }
        let outcome = self.ledger_txn(session_id, true, |records: &mut Vec<DurableRecord>| {
            let entry_epoch = self.in_memory_epoch(session_id);
            let index = match records.iter().position(|record| record.session == session_id) {
                Some(index) => index,
                None => return Ok((VetoConsultState::Unavailable, Vec::new(), false)),
            };
            let record = records[index].clone();
            if record.epoch < entry_epoch {
                return Ok((VetoConsultState::Unavailable, Vec::new(), false));
            }
            if record.veto_attempts.contains(&attempt) {
                return Ok((VetoConsultState::Repeated, Vec::new(), false));
            }
            let mut attempts = record.veto_attempts.clone();
            attempts.push(attempt);
            let mut updated = record.clone();
            updated.seq = next_ledger_sequence(records, 0)?;
            updated.veto_attempts = trim_consults(attempts);
            records[index] = updated.clone();
            // Cache install happens only after the durable commit succeeds.
            Ok((VetoConsultState::Fresh, vec![updated], true))
        });
        match outcome {
            Ok(state) => state,
            Err(()) => VetoConsultState::Unavailable,
        }
    }

    // -----------------------------------------------------------------------
    // Durable ledger transaction machinery (std-only; same discipline as the
    // cron-jobs state lock: in-process coordinator plus create_new file lock
    // with bounded retries and stale takeover).
    // -----------------------------------------------------------------------

    /// Run one bounded ledger transaction. `mutate` transactions compact and
    /// rewrite the bounded file under the lock ONLY when the closure changed
    /// state; read-only transactions only parse. The OS lock handle is held
    /// across load, validation, and commit. Cache installs happen ONLY after
    /// a successful durable commit (or from freshly read durable truth), so
    /// no failed commit or failed cache path can leave the cache authorizing
    /// an effect. The sessions mutex may be taken INSIDE this transaction
    /// (ledger -> memory lock order only).
    fn ledger_txn<T>(
        &self,
        session_id: &str,
        mutate: bool,
        f: impl FnOnce(&mut Vec<DurableRecord>) -> Result<(T, Vec<DurableRecord>, bool), ()>,
    ) -> Result<T, ()> {
        let _io_guard = self
            .io_mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _lock_handle = acquire_ledger_lock(&self.lock_path)?;
        let mut records = self.load_records()?;
        let (outcome, installs, changed) = f(&mut records)?;
        if mutate && changed {
            compact_for_write(&mut records, session_id)?;
            self.write_records(&records)?;
        }
        for record in installs {
            self.install(&record);
        }
        Ok(outcome)
    }

    /// Load and strictly validate the WHOLE ledger. Trustworthy only when the
    /// byte size is within the cap, the content is valid UTF-8, EVERY line is
    /// a valid v2 record, and every session appears exactly once. Our writer
    /// emits exactly one current record per session atomically, so any
    /// irregularity (corrupt, truncated, oversized, invalid UTF-8, malformed
    /// consults, foreign or duplicate records) means the durable state cannot
    /// be trusted: refuse the ENTIRE file rather than salvaging an older
    /// valid record into authority. A missing file is empty durable state.
    fn load_records(&self) -> Result<Vec<DurableRecord>, ()> {
        let mut file = match File::open(&self.ledger_path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(_) => return Err(()),
        };
        if file.metadata().map_err(|_| ())?.len() > MAX_LEDGER_BYTES as u64 {
            return Err(());
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|_| ())?;
        let text = String::from_utf8(bytes).map_err(|_| ())?;
        let mut records: Vec<DurableRecord> = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                // Our writer never emits blank or whitespace lines.
                return Err(());
            }
            let record = DurableRecord::parse(line).ok_or(())?;
            if records
                .iter()
                .any(|existing| existing.session == record.session)
            {
                // Conflicting duplicate record set: refuse the whole file.
                return Err(());
            }
            records.push(record);
        }
        Ok(records)
    }

    /// Durable write: temp file + sync + atomic rename under the transaction
    /// lock. Any failure fails the transaction (no partial authority).
    fn write_records(&self, records: &[DurableRecord]) -> Result<(), ()> {
        if let Some(parent) = self.ledger_path.parent() {
            fs::create_dir_all(parent).map_err(|_| ())?;
        }
        let mut payload = String::new();
        for record in records {
            payload.push_str(&record.to_line());
            payload.push('\n');
        }
        if payload.len() > MAX_LEDGER_BYTES {
            return Err(());
        }
        let temp_path = self.ledger_path.with_extension("jsonl.tmp");
        let written: std::io::Result<()> = (|| {
            let mut file = File::create(&temp_path)?;
            file.write_all(payload.as_bytes())?;
            file.sync_all()?;
            Ok(())
        })();
        if written.is_err() {
            let _ = fs::remove_file(&temp_path);
            return Err(());
        }
        if fs::rename(&temp_path, &self.ledger_path).is_err() {
            let _ = fs::remove_file(&temp_path);
            return Err(());
        }
        Ok(())
    }

    /// Recovery of one session's durable state. `Trusted` when a valid record
    /// exists at or ahead of the in-memory epoch (installed into memory);
    /// `Uncommitted` when the in-memory epoch advanced past the durable one
    /// (its activation failed; nothing may spend); `Missing` when the ledger
    /// holds no record for the session. Read failures are `Err`.
    fn recover(&self, session_id: &str) -> Result<Recovery, ()> {
        self.ledger_txn(session_id, false, |records: &mut Vec<DurableRecord>| {
            let entry_epoch = self.in_memory_epoch(session_id);
            match records.iter().find(|record| record.session == session_id) {
                // Install only freshly read durable truth into the cache.
                Some(record) if record.epoch >= entry_epoch => Ok((
                    Recovery::Trusted(record.clone()),
                    vec![record.clone()],
                    false,
                )),
                Some(_) => Ok((Recovery::Uncommitted, Vec::new(), false)),
                None => Ok((Recovery::Missing, Vec::new(), false)),
            }
        })
    }

    /// Current DURABLE task-epoch identity for the session, read through a
    /// ledger transaction and never from the in-memory cache. Resolvers use
    /// this to refuse effects whose captured epoch is no longer the current
    /// durable epoch (cross-process authority check). `None` when no trusted
    /// durable record exists or the ledger is unreadable.
    pub fn current_epoch_id(&self, session_id: &str) -> Option<String> {
        if !valid_session_id(session_id) {
            return None;
        }
        let outcome = self.ledger_txn(session_id, false, |records: &mut Vec<DurableRecord>| {
            let current = records
                .iter()
                .find(|record| record.session == session_id)
                .map(|record| epoch_id(&record.session, record.epoch));
            Ok((current, Vec::new(), false))
        });
        match outcome {
            Ok(current) => current,
            Err(()) => None,
        }
    }

    /// Commit a NEW epoch activation with fresh maxima. The epoch ordinal is
    /// monotonic across in-memory and durable state; only a successful
    /// durable commit returns a trusted snapshot.
    fn commit_activation(
        &self,
        session_id: &str,
        anchor: &str,
    ) -> Result<ControlBudgetSnapshot, ()> {
        self.ledger_txn(session_id, true, |records: &mut Vec<DurableRecord>| {
            let (entry_epoch, entry_seq) = self.in_memory_progress(session_id);
            let durable = records
                .iter()
                .find(|record| record.session == session_id);
            let seq = next_ledger_sequence(records, entry_seq.max(entry_epoch))?;
            let epoch = match durable {
                Some(record) => record.epoch.max(entry_epoch).checked_add(1).ok_or(())?,
                // Compaction may have removed this session's entire history,
                // including its cached epoch. The retained ledger high-water
                // mark prevents an old decision from matching a new task.
                None => seq,
            };
            let snapshot = ControlBudgetSnapshot::fresh_maxima(epoch_id(session_id, epoch));
            let record = DurableRecord {
                session: session_id.to_string(),
                epoch,
                seq,
                anchor_digest: anchor.to_string(),
                feedback_remaining: snapshot.feedback_remaining,
                verification_remaining: snapshot.verification_remaining,
                nonprogress_remaining: snapshot.nonprogress_remaining,
                veto_remaining: snapshot.veto_remaining,
                veto_attempts: Vec::new(),
                feedback_notices: Vec::new(),
            };
            match records
                .iter()
                .position(|existing| existing.session == session_id)
            {
                Some(index) => records[index] = record.clone(),
                None => records.push(record.clone()),
            }
            // Cache install happens only after the durable commit succeeds.
            Ok((record.budget_snapshot(), vec![record], true))
        })
    }

    /// Advance the in-memory epoch after a failed activation commit. No
    /// budget is held; the next activation starts a NEW epoch.
    fn advance_uncommitted_epoch(&self, session_id: &str, anchor: &str) -> u64 {
        let mut sessions = self.lock_sessions();
        let entry = self.entry_or_evict(&mut sessions, session_id);
        entry.epoch = entry.epoch.saturating_add(1);
        entry.seq = entry.seq.saturating_add(1);
        entry.anchor_digest = anchor.to_string();
        entry.budget = None;
        entry.consulted_veto_attempts.clear();
        entry.epoch
    }

    /// Install a durable record into the in-memory cache. Never downgrades
    /// the epoch (an uncommitted in-memory epoch stays untrusted) and never
    /// touches turn signatures (in-memory only).
    fn install(&self, record: &DurableRecord) {
        let mut sessions = self.lock_sessions();
        let entry = self.entry_or_evict(&mut sessions, &record.session);
        if record.epoch >= entry.epoch {
            if record.epoch != entry.epoch {
                entry.signatures.clear();
            }
            entry.epoch = record.epoch;
            entry.seq = entry.seq.max(record.seq);
            entry.anchor_digest = record.anchor_digest.clone();
            entry.budget = Some(record.budget_snapshot());
            entry.consulted_veto_attempts = record.veto_attempts.clone();
        }
    }

    fn fast_trusted(&self, session_id: &str) -> Option<ControlBudgetSnapshot> {
        let sessions = self.lock_sessions();
        let entry = sessions.get(session_id)?;
        let budget = entry.budget.as_ref()?;
        if budget.available && budget.epoch_id == epoch_id(session_id, entry.epoch) {
            Some(budget.clone())
        } else {
            None
        }
    }

    fn in_memory_epoch(&self, session_id: &str) -> u64 {
        let sessions = self.lock_sessions();
        sessions
            .get(session_id)
            .map(|entry| entry.epoch)
            .unwrap_or(0)
    }

    fn in_memory_progress(&self, session_id: &str) -> (u64, u64) {
        let sessions = self.lock_sessions();
        match sessions.get(session_id) {
            Some(entry) => (entry.epoch, entry.seq),
            None => (0, 0),
        }
    }

    fn entry_or_evict<'a>(
        &self,
        sessions: &'a mut HashMap<String, SessionControl>,
        session_id: &str,
    ) -> &'a mut SessionControl {
        if sessions.len() >= MAX_TRACKED_SESSIONS && !sessions.contains_key(session_id) {
            // Evict the least-recent entry; durable state heals the next
            // access, and eviction itself never resets an epoch.
            if let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, entry)| (entry.seq, entry.epoch))
                .map(|(key, _)| key.clone())
            {
                sessions.remove(&oldest);
            }
        }
        sessions.entry(session_id.to_string()).or_default()
    }

    fn lock_sessions(&self) -> std::sync::MutexGuard<'_, HashMap<String, SessionControl>> {
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// In-memory epoch recovery outcome for one session.
enum Recovery {
    Trusted(DurableRecord),
    Uncommitted,
    Missing,
}

/// Truthful veto-consult state (once-per-attempt durability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VetoConsultState {
    /// This attempt ordinal was not consulted yet; the consult is now durably
    /// recorded.
    Fresh,
    /// This attempt ordinal was already consulted (across books/restarts).
    Repeated,
    /// Durable accounting is unavailable; the consult is refused and the
    /// baseline retry policy proceeds.
    Unavailable,
}

/// One validated durable budget record (ledger v2). Every field is bounded
/// and content-free: session attribution, monotonic sequence, epoch ordinal,
/// remaining counters, and veto-consult ordinals.
#[derive(Debug, Clone)]
struct DurableRecord {
    session: String,
    epoch: u64,
    seq: u64,
    anchor_digest: String,
    feedback_remaining: u8,
    verification_remaining: u8,
    nonprogress_remaining: u8,
    veto_remaining: u8,
    veto_attempts: Vec<u32>,
    feedback_notices: Vec<String>,
}

impl DurableRecord {
    /// Trustworthy record validation: version, session bounds, epoch/seq
    /// ordinals, counters at or below the contract maxima, and a bounded
    /// consult list. Any miss rejects the record (fail-closed), never clamps
    /// it into authority.
    fn parse(line: &str) -> Option<DurableRecord> {
        let value: Value = serde_json::from_str(line).ok()?;
        if value.get("v").and_then(Value::as_u64) != Some(LEDGER_RECORD_VERSION) {
            return None;
        }
        let session = value.get("session").and_then(Value::as_str)?;
        if !valid_session_id(session) {
            return None;
        }
        let epoch = value.get("epoch").and_then(Value::as_u64)?;
        if epoch == 0 {
            return None;
        }
        let seq = value.get("seq").and_then(Value::as_u64)?;
        if seq == 0 {
            return None;
        }
        let anchor = value
            .get("anchor")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if anchor.chars().count() > 64 || !anchor.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let maxima = ControlBudgets::maxima();
        let counter = |key: &str, max: u8| -> Option<u8> {
            let raw = value.get(key).and_then(Value::as_u64)?;
            u8::try_from(raw).ok().filter(|count| *count <= max)
        };
        let feedback_remaining = counter("feedback", maxima.feedback)?;
        let verification_remaining = counter("verification", maxima.verification_requests)?;
        let nonprogress_remaining = counter("nonprogress", maxima.nonprogress_corrections)?;
        let veto_remaining = counter("veto", maxima.retry_vetoes)?;
        // Consults must be a PRESENT, bounded, valid array: a missing or
        // non-array field must not silently erase consult trust, and
        // load-time truncation/deduplication must not rewrite history into
        // authority. (Our writer always emits the array, sorted, deduplicated
        // and bounded.)
        let list = value.get("consults").and_then(Value::as_array)?;
        if list.len() > MAX_RECORDED_VETO_CONSULTS {
            return None;
        }
        let mut veto_attempts: Vec<u32> = Vec::new();
        for entry in list {
            let raw = entry.as_u64()?;
            let ordinal = u32::try_from(raw).ok()?;
            if ordinal == 0 || veto_attempts.contains(&ordinal) {
                return None;
            }
            veto_attempts.push(ordinal);
        }
        // Older v2 records have no notice ids. They retain their spent budgets.
        // Each delivered notice stores two keys (exact-decision + epoch-kind),
        // and at most `feedback` deliveries can happen in one epoch, so the
        // validated bound is twice the feedback maximum.
        let feedback_notices: Vec<String> = match value.get("feedback_notices") {
            None => Vec::new(),
            Some(value) => serde_json::from_value(value.clone()).ok()?,
        };
        if feedback_notices.len() > usize::from(maxima.feedback) * 2
            || feedback_notices.iter().any(|key| key.len() != 64 || !key.bytes().all(|c| c.is_ascii_hexdigit()))
        {
            return None;
        }
        Some(DurableRecord {
            session: session.to_string(),
            epoch,
            seq,
            anchor_digest: anchor.to_string(),
            feedback_remaining,
            verification_remaining,
            nonprogress_remaining,
            veto_remaining,
            veto_attempts,
            feedback_notices,
        })
    }

    fn to_line(&self) -> String {
        json!({
            "v": LEDGER_RECORD_VERSION,
            "session": self.session,
            "epoch": self.epoch,
            "seq": self.seq,
            "anchor": self.anchor_digest,
            "feedback": self.feedback_remaining,
            "verification": self.verification_remaining,
            "nonprogress": self.nonprogress_remaining,
            "veto": self.veto_remaining,
            "consults": self.veto_attempts,
            "feedback_notices": self.feedback_notices,
        })
        .to_string()
    }

    fn budget_snapshot(&self) -> ControlBudgetSnapshot {
        ControlBudgetSnapshot {
            epoch_id: epoch_id(&self.session, self.epoch),
            feedback_remaining: self.feedback_remaining,
            verification_remaining: self.verification_remaining,
            nonprogress_remaining: self.nonprogress_remaining,
            veto_remaining: self.veto_remaining,
            available: true,
        }
    }
}

/// Every committed mutation advances a ledger-wide high-water mark. The
/// newest record survives compaction, so missing sessions can allocate an
/// epoch above every prior committed epoch without unbounded tombstones.
/// Exhaustion refuses the transaction instead of reusing an identity.
fn next_ledger_sequence(records: &[DurableRecord], floor: u64) -> Result<u64, ()> {
    records
        .iter()
        .map(|record| record.seq.max(record.epoch))
        .chain(std::iter::once(floor))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(())
}

/// Bound the write set: one latest record per session, the
/// [`MAX_TRACKED_SESSIONS`] freshest sessions by seq, and at most
/// [`MAX_LEDGER_BYTES`] bytes. The session being written always survives;
/// anything dropped becomes fail-closed on its next access (never
/// re-initialized, never minted).
fn compact_for_write(records: &mut Vec<DurableRecord>, session_id: &str) -> Result<(), ()> {
    // The vec arrives in file order (chronologically oldest first) with the
    // record being committed appended last. Reverse FIRST so the STABLE
    // seq-descending sort keeps the NEWEST record at the front of every
    // equal-seq tie group in older ledgers. New writes allocate a ledger-wide
    // sequence, so the current record is newest and preserves the high-water
    // mark even after every other session has been evicted.
    records.reverse();
    records.sort_by(|a, b| b.seq.cmp(&a.seq));
    records.truncate(MAX_TRACKED_SESSIONS);
    if !records.iter().any(|record| record.session == session_id) {
        return Err(());
    }
    while records.len() > 1 {
        let total: usize = records
            .iter()
            .map(|record| record.to_line().len() + 1)
            .sum();
        if total <= MAX_LEDGER_BYTES {
            break;
        }
        let dropped = records.pop();
        if dropped.as_ref().map(|record| record.session.as_str()) == Some(session_id) {
            return Err(());
        }
    }
    let total: usize = records
        .iter()
        .map(|record| record.to_line().len() + 1)
        .sum();
    if total > MAX_LEDGER_BYTES {
        return Err(());
    }
    if records.len() > MAX_LEDGER_LINES {
        records.truncate(MAX_LEDGER_LINES);
    }
    records.reverse(); // chronological order in the file
    Ok(())
}

/// Cross-process transaction lock: the SAME stable-file OS try_lock pattern
/// as the settings store (pi-jev/src/config.rs save). The `<ledger>.lock`
/// file is created once and NEVER deleted — deleting it would let two
/// processes acquire different locks — and the returned File handle holds
/// the OS lock until it is dropped, so a crashed writer releases the lock
/// through the OS and a live (even suspended) writer can never have its lock
/// stolen by an mtime heuristic. Busy => immediate fail-closed error; the
/// transaction reports unavailable accounting and never retries in a loop.
fn acquire_ledger_lock(lock_path: &Path) -> Result<File, ()> {
    // Mirror the settings-store save pattern (pi-jev/src/config.rs): the
    // lock file's parent directory must exist before the stable lock file
    // can be opened, otherwise every transaction would fail closed on a
    // fresh directory.
    let parent = lock_path.parent().ok_or(())?;
    fs::create_dir_all(parent).map_err(|_| ())?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|_| ())?;
    lock.try_lock().map_err(|_| ())?;
    Ok(lock)
}

fn valid_session_id(session_id: &str) -> bool {
    !session_id.is_empty()
        && session_id.chars().count() <= MAX_SESSION_ID_CHARS
        && !session_id.chars().any(char::is_control)
}

fn trim_consults(mut attempts: Vec<u32>) -> Vec<u32> {
    attempts.sort_unstable();
    attempts.dedup();
    if attempts.len() > MAX_RECORDED_VETO_CONSULTS {
        attempts = attempts[attempts.len() - MAX_RECORDED_VETO_CONSULTS..].to_vec();
    }
    attempts
}

fn epoch_id(session_id: &str, epoch: u64) -> String {
    format!("{session_id}:{epoch}")
}

fn now_ms_i64() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn digest_text(text: &str) -> String {
    let bounded: String = text.chars().take(MAX_DIGEST_INPUT_CHARS).collect();
    let mut hasher = Sha256::new();
    hasher.update(bounded.len().to_le_bytes());
    hasher.update(bounded.as_bytes());
    hex(&hasher.finalize())
}

fn digest_parts(parts: &[String]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.len().to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Bounded, privacy-safe content signature of one turn (contract: repeated
/// tool NAMES alone never count; digests cover names, arguments, text,
/// thinking, and tool results; nothing raw is retained).
pub fn turn_signature(
    message: &pi_ai::types::AssistantMessage,
    tool_results: &[pi_ai::types::ToolResultMessage],
) -> TurnSignature {
    use pi_ai::types::ContentBlock;

    let mut tool_call_parts: Vec<String> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::ToolCall(call) => {
                let arguments = pi_jev::snapshot::bound_json(
                    serde_json::to_value(&call.arguments).unwrap_or(Value::Null),
                    0,
                );
                tool_call_parts.push(format!(
                    "{}\u{1}{}",
                    call.name,
                    digest_parts(&[serde_json::to_string(&arguments).unwrap_or_default()])
                ));
            }
            ContentBlock::Text(text) => {
                text_parts.push(digest_text(&text.text));
            }
            ContentBlock::Thinking(thinking) => {
                text_parts.push(digest_text(&thinking.thinking));
            }
        }
    }
    tool_call_parts.sort();

    let mut result_parts: Vec<String> = Vec::new();
    for result in tool_results {
        let mut content_parts: Vec<String> = Vec::new();
        for block in &result.content {
            match block {
                pi_ai::types::ImageOrTextContent::Text(text) => {
                    content_parts.push(digest_text(&text.text));
                }
                pi_ai::types::ImageOrTextContent::Image(_) => {
                    content_parts.push("image".to_string());
                }
            }
        }
        content_parts.sort();
        result_parts.push(format!(
            "{}\u{1}{}\u{1}{}",
            result.tool_name,
            if result.is_error { "error" } else { "ok" },
            digest_parts(&content_parts),
        ));
    }
    result_parts.sort();

    TurnSignature {
        tool_calls_digest: digest_parts(&tool_call_parts),
        text_digest: digest_parts(&text_parts),
        results_digest: digest_parts(&result_parts),
        // Marked by the book when recorded: true when any component differs
        // from the previous turn's signature.
        has_new_evidence: false,
    }
}

/// The control question set eligible at a boundary, mirroring the legacy
/// observer gating exactly (questions themselves are unchanged).
pub fn eligible_questions(
    snapshot: &StateSnapshot,
    features: &ControlFeatures,
    failure_kind_present: bool,
) -> Vec<PreparedQuestion> {
    let stage = snapshot.stage;
    let mut questions = Vec::new();
    for evaluator in for_boundary(stage) {
        let enabled = match evaluator.category().as_str() {
            "result_sufficiency" => features.result_sufficiency,
            "continue_stop_escalate" => features.loop_control,
            "first_pass_verification" => features.verification,
            "retry_classification" => features.retry_classification && failure_kind_present,
            _ => false,
        };
        if enabled {
            if let EvaluatorOutput::Questions(mut prepared) = evaluator.evaluate(snapshot) {
                questions.append(&mut prepared);
            }
        }
    }
    questions
}

/// Result of the AgentEnd control gate.
#[derive(Debug, Clone)]
pub struct ControlAgentEndResult {
    pub verdicts: Vec<ControlVerdict>,
    /// Host-authored fixed feedback to queue via the existing steer surface.
    pub feedback: Option<CustomMessage>,
    /// Defer goal finish while a corrective/verification continuation is
    /// pending (contract: no automatic success from Jev).
    pub defer_goal_finish: bool,
    /// Escalate: safe turn-boundary pause plus user-attention flag.
    pub escalate: bool,
    /// Safe terminal pause (contract fallback: verification recommended but
    /// the budget-1 verification request is already spent). Never success,
    /// never mid-tool.
    pub pause: Option<PauseReason>,
    /// Honest verification state; never synthesized from transport success.
    pub verification_state: ControlVerificationState,
    /// Truthful terminal annotation when budgets are exhausted (never a
    /// success claim).
    pub terminal_annotation: Option<&'static str>,
}

/// Result of the TurnEnd control gate.
#[derive(Debug, Clone)]
pub struct ControlTurnEndResult {
    pub verdicts: Vec<ControlVerdict>,
    pub feedback: Option<CustomMessage>,
    /// Stop the loop at the turn boundary (truthful pause; never success,
    /// never mid-tool).
    pub stop_loop: bool,
    pub escalate: bool,
}

/// Result of the retry veto consult.
#[derive(Debug, Clone)]
pub struct ControlRetryVerdict {
    pub verdicts: Vec<ControlVerdict>,
    /// True only for an accepted {bad_arguments, fatal} classification at the
    /// high-impact floor with veto budget remaining, once per attempt.
    pub veto: bool,
    pub classification: Option<RetryFailureKind>,
    pub confidence: Option<f64>,
    pub reason: &'static str,
}

/// AgentEnd gate. The integrator calls this with the answers from ONE
/// fanned-out decision at `agent_end`; `continuation_pending` is host truth
/// (another continuation or compaction retry already queued).
pub fn resolve_agent_end(
    book: &ControlBook,
    session_id: &str,
    policy: &ControlPolicy,
    features: &ControlFeatures,
    mode: JevMode,
    facts: &HostControlFacts,
    candidates: &[AnswerCandidate],
    continuation_pending: bool,
    evidence_description: &str,
) -> ControlAgentEndResult {
    let mut result = ControlAgentEndResult {
        verdicts: Vec::new(),
        feedback: None,
        defer_goal_finish: false,
        escalate: false,
        pause: None,
        verification_state: ControlVerificationState::Unknown,
        terminal_annotation: None,
    };
    // Gates first: closed gates never touch the book or the ledger
    // (H-CONTROL-1: no control state outside the full profile).
    if !control_gates_open(features, mode) {
        result.verdicts.push(ControlVerdict::Refused(
            ControlRefusal::GatesClosed("gates_closed"),
        ));
        return result;
    }
    let budget = book.snapshot(session_id);
    let mut sufficiency = SufficiencyVerdict::Unknown;
    let mut verification = VerificationNeed::Unknown;
    let mut insufficient_budget_refused = false;
    let mut verification_budget_refused = false;
    let mut terminal_stop = false;
    for candidate in candidates {
        let verdict = evaluate_control_answer(
            policy,
            features,
            mode,
            ControlBoundary::AgentEnd,
            candidate,
            facts,
            &budget,
        );
        let applied_effect = verdict.applied().map(|acceptance| acceptance.effect.clone());
        let budget_refused = matches!(
            verdict,
            ControlVerdict::Refused(ControlRefusal::BudgetExhausted(_))
        );
        match candidate.category {
            DecisionCategory::ResultSufficiency => match applied_effect {
                Some(ControlEffectKind::Feedback(FeedbackKind::ResultGap)) => {
                    sufficiency = SufficiencyVerdict::Insufficient;
                }
                Some(ControlEffectKind::Continue) => {
                    // A sufficient/complete answer raises the verdict only
                    // when no accepted answer called the result insufficient.
                    if sufficiency != SufficiencyVerdict::Insufficient {
                        if let Some(value) = candidate.value.as_deref() {
                            if combine_sufficiency(Some(value), None)
                                == SufficiencyVerdict::Sufficient
                            {
                                sufficiency = SufficiencyVerdict::Sufficient;
                            }
                        }
                    }
                }
                // Jev said insufficient but the feedback budget is spent: the
                // refusal is the record; annotate truthfully, never silently.
                _ => insufficient_budget_refused |= budget_refused,
            },
            DecisionCategory::FirstPassVerification => match applied_effect {
                Some(ControlEffectKind::Feedback(FeedbackKind::VerificationMissing)) => {
                    verification = VerificationNeed::Request;
                }
                Some(ControlEffectKind::Continue) => {
                    verification = verification_need(candidate.value.as_deref());
                }
                Some(ControlEffectKind::Escalate) => {
                    verification = VerificationNeed::Escalate;
                    result.escalate = true;
                }
                // Verification recommended but the budget-1 request is spent.
                _ => verification_budget_refused |= budget_refused,
            },
            DecisionCategory::ContinueStopEscalate => {
                // At AgentEnd the loop has already stopped. An accepted stop
                // must not be overridden by another category reopening it.
                terminal_stop |= matches!(applied_effect, Some(ControlEffectKind::Continue))
                    && candidate.value.as_deref().is_some_and(|value| value.eq_ignore_ascii_case("stop"));
                if matches!(applied_effect, Some(ControlEffectKind::Escalate)) {
                    result.escalate = true;
                }
            }
            _ => {}
        }
        result.verdicts.push(verdict);
    }
    // Untrusted durable accounting: refuse NEW control effects, keep the
    // ordinary baseline available, annotate truthfully (never a success
    // claim), and never loop (C-BUDGET-1).
    if result.verdicts.iter().any(|verdict| {
        matches!(
            verdict,
            ControlVerdict::Refused(ControlRefusal::AccountingUnavailable(_))
        )
    }) {
        result.terminal_annotation = Some("accounting_unavailable");
        return result;
    }
    // A decision may act only while its captured epoch is still the CURRENT
    // durable epoch (cross-process authority check; single-process
    // transactions keep them equal by construction).
    if book.current_epoch_id(session_id).as_deref() != Some(budget.epoch_id.as_str()) {
        result
            .verdicts
            .push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet(
                "epoch_moved",
            )));
        return result;
    }
    // Terminal safety decisions must survive a replayed feedback notice.
    // Dedupe only the corrective continuation, never a stop or escalation.
    if result.escalate {
        result.defer_goal_finish = true;
        return result;
    }
    if terminal_stop {
        // A stop assessment is not evidence that implementation or tests passed.
        if verification == VerificationNeed::NotRequired {
            result.verification_state = ControlVerificationState::NotApplicable;
        }
        result.terminal_annotation = Some("no_authorized_follow_up");
        return result;
    }
    // Replayed-decision guard (both feedback kinds): when THIS exact
    // decision already delivered its notice — including after a restart —
    // nothing new may be said about it. Silent duplicate refusal, never a
    // pause and never a success claim.
    if book.feedback_delivery_state(session_id, facts, FeedbackKind::ResultGap)
        == FeedbackDeliveryState::SameDecision
        || book.feedback_delivery_state(session_id, facts, FeedbackKind::VerificationMissing)
            == FeedbackDeliveryState::SameDecision
    {
        result
            .verdicts
            .push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet(
                "duplicate_control_feedback",
            )));
        return result;
    }
    if sufficiency == SufficiencyVerdict::Insufficient {
        if continuation_pending {
            result.verdicts.push(ControlVerdict::Refused(
                ControlRefusal::ContinuationPending,
            ));
            return result;
        }
        // One corrective result-gap notice per logical task. A repeated
        // insistence on the same epoch must not reopen the task (no second
        // message), but still-insufficient NON-TERMINAL work is never
        // finished either: the goal stays deferred with a truthful paused
        // state (attention required). This is a refusal to continue, never
        // a success, sufficiency, or verification claim; NotRequired maps
        // to NotApplicable only (verification not needed, not task
        // sufficient).
        if book.feedback_delivery_state(
            session_id,
            facts,
            FeedbackKind::ResultGap,
        ) != FeedbackDeliveryState::New
        {
            result.verdicts.push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet(
                "duplicate_control_feedback",
            )));
            if verification == VerificationNeed::NotRequired {
                // Not-applicable mirrors the terminal-stop mapping: the
                // evaluator's own recommendation, never a passed check and
                // never a sufficiency claim.
                result.verification_state = ControlVerificationState::NotApplicable;
            }
            result.pause = Some(PauseReason::BudgetExhausted);
            result.defer_goal_finish = true;
            result.terminal_annotation = Some("duplicate_control_feedback");
            return result;
        }
        let exact_notice = feedback_notice_key(facts, FeedbackKind::ResultGap);
        let epoch_notice = feedback_epoch_notice_key(&budget.epoch_id, FeedbackKind::ResultGap);
        let notices = [exact_notice.as_str(), epoch_notice.as_str()];
        match book.consume_with_notices(
            session_id,
            &budget.epoch_id,
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap),
            &notices,
        ) {
            Some(_) => {
                result.feedback = Some(result_gap_feedback(
                    "insufficient",
                    evidence_description,
                ));
                result.defer_goal_finish = true;
            }
            None => {
                // Unreachable in practice (evaluate refuses first); kept for
                // race safety and truthful annotation.
                result.terminal_annotation = Some("budget_exhausted_unverified");
            }
        }
        return result;
    }
    if insufficient_budget_refused {
        // Feedback continuations already spent this epoch: truthful final
        // state, never silent success, no pause loop.
        result.terminal_annotation = Some("budget_exhausted_unverified");
        return result;
    }
    match verification {
        VerificationNeed::Request => {
            if continuation_pending {
                result
                    .verdicts
                    .push(ControlVerdict::Refused(ControlRefusal::ContinuationPending));
                return result;
            }
            // One verification request per logical task. A replayed decision
            // stays silent (its notice was delivered); a NEW agent end in the
            // same epoch gets no second message, only the truthful pause.
            match book.feedback_delivery_state(
                session_id,
                facts,
                FeedbackKind::VerificationMissing,
            ) {
                FeedbackDeliveryState::SameDecision => {
                    result
                        .verdicts
                        .push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet(
                            "duplicate_control_feedback",
                        )));
                    return result;
                }
                FeedbackDeliveryState::DeliveredInEpoch => {
                    // Verification was already requested once in this epoch
                    // and remains unconfirmed: no nag, no success claim; the
                    // user decides. Unverified is preserved truthfully.
                    result
                        .verdicts
                        .push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet(
                            "duplicate_control_feedback",
                        )));
                    result.pause = Some(PauseReason::VerificationUnconfirmed);
                    result.defer_goal_finish = true;
                    result.verification_state = ControlVerificationState::Unverified;
                    result.terminal_annotation = Some("verification_unconfirmed");
                    return result;
                }
                FeedbackDeliveryState::New => {}
            }
            let exact_notice = feedback_notice_key(facts, FeedbackKind::VerificationMissing);
            let epoch_notice =
                feedback_epoch_notice_key(&budget.epoch_id, FeedbackKind::VerificationMissing);
            let notices = [exact_notice.as_str(), epoch_notice.as_str()];
            match book.consume_with_notices(
                session_id,
                &budget.epoch_id,
                ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing),
                &notices,
            ) {
                Some(_) => {
                    result.feedback =
                        Some(verification_missing_feedback(evidence_description));
                    result.defer_goal_finish = true;
                    result.verification_state = ControlVerificationState::Unverified;
                }
                None => {
                    // Race fallback (evaluate normally refuses first): the
                    // request could not be delivered. Truthful pause; the
                    // control lane never panics into the agent.
                    result.pause = Some(PauseReason::VerificationUnconfirmed);
                    result.defer_goal_finish = true;
                    result.verification_state = ControlVerificationState::Unverified;
                    result.terminal_annotation = Some("verification_unconfirmed");
                }
            }
        }
        VerificationNeed::NotRequired => {
            if sufficiency == SufficiencyVerdict::Sufficient {
                result.verification_state = ControlVerificationState::NotApplicable;
            }
        }
        VerificationNeed::Escalate => {
            // Unreachable: an accepted escalate answer returns early above.
        }
        VerificationNeed::Unknown => {}
    }
    if verification_budget_refused {
        // Contract fallback: verification was recommended but the budget-1
        // request is spent. Truthful terminal pause; the user decides.
        result.pause = Some(PauseReason::VerificationUnconfirmed);
        result.defer_goal_finish = true;
        result.verification_state = ControlVerificationState::Unverified;
        result.terminal_annotation = Some("verification_unconfirmed");
    }
    result
}

/// Exact-decision notice key: epoch + request + turn + kind. A replayed
/// decision (same facts) can never deliver its notice twice.
fn feedback_notice_key(facts: &HostControlFacts, kind: FeedbackKind) -> String {
    digest_parts(&[
        facts.epoch_id.clone(),
        facts.request_id.clone(),
        facts.turn.to_string(),
        kind.as_str().to_string(),
    ])
}

/// Epoch-scoped notice key: epoch + kind. Once a KIND of corrective notice
/// was delivered for one real-user task, no later agent end in that task
/// delivers the same kind again (feedback finite per logical task).
fn feedback_epoch_notice_key(epoch_id: &str, kind: FeedbackKind) -> String {
    digest_parts(&[epoch_id.to_string(), kind.as_str().to_string()])
}

/// TurnEnd gate. The integrator calls this with the answers from ONE
/// fanned-out decision at `turn_end` when the host nonprogress predicate
/// triggered (no decision is requested without a trigger).
pub fn resolve_turn_end(
    book: &ControlBook,
    session_id: &str,
    policy: &ControlPolicy,
    features: &ControlFeatures,
    mode: JevMode,
    facts: &HostControlFacts,
    candidates: &[AnswerCandidate],
    nonprogress: &NonprogressVerdict,
) -> ControlTurnEndResult {
    let mut result = ControlTurnEndResult {
        verdicts: Vec::new(),
        feedback: None,
        stop_loop: false,
        escalate: false,
    };
    if !control_gates_open(features, mode) {
        result.verdicts.push(ControlVerdict::Refused(
            ControlRefusal::GatesClosed("gates_closed"),
        ));
        return result;
    }
    let NonprogressVerdict::Candidate { identical_turns } = nonprogress else {
        result
            .verdicts
            .push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet("no_nonprogress")));
        return result;
    };
    let budget = book.snapshot(session_id);
    if !budget.available {
        // Durable accounting untrusted: refuse new control effects; the
        // baseline loop continues (C-BUDGET-1, no refill, no loop).
        result
            .verdicts
            .push(ControlVerdict::Refused(ControlRefusal::AccountingUnavailable(
                "accounting_unavailable",
            )));
        return result;
    }
    // A decision may act only while its captured epoch is still the CURRENT
    // durable epoch (cross-process authority check).
    if book.current_epoch_id(session_id).as_deref() != Some(budget.epoch_id.as_str()) {
        result
            .verdicts
            .push(ControlVerdict::Refused(ControlRefusal::TriggerNotMet(
                "epoch_moved",
            )));
        return result;
    }
    let mut accepted_stop = false;
    let mut accepted_escalate = false;
    for candidate in candidates {
        let verdict = evaluate_control_answer(
            policy,
            features,
            mode,
            ControlBoundary::TurnEnd,
            candidate,
            facts,
            &budget,
        );
        match verdict.applied().map(|acceptance| acceptance.effect.clone()) {
            Some(ControlEffectKind::Pause(PauseReason::NonprogressUncorrected)) => {
                accepted_stop = true;
                result.verdicts.push(verdict);
            }
            Some(ControlEffectKind::Escalate) => {
                accepted_escalate = true;
                result.verdicts.push(verdict);
            }
            _ => result.verdicts.push(verdict),
        }
    }
    if accepted_escalate {
        result.stop_loop = true;
        result.escalate = true;
        return result;
    }
    // Correction first (contract: nonprogress correction then truthful pause).
    if book
        .consume(
            session_id,
            &budget.epoch_id,
            ControlBudgetKind::NonprogressCorrection,
        )
        .is_some()
    {
        result.feedback = Some(nonprogress_feedback(*identical_turns));
        return result;
    }
    result
        .verdicts
        .push(ControlVerdict::Refused(ControlRefusal::BudgetExhausted(
            ControlBudgetKind::NonprogressCorrection,
        )));
    if accepted_stop {
        result.stop_loop = true;
    }
    result
}

/// Retry veto consult. Call once per retry attempt ordinal at the host's
/// existing provider retry decision; the veto only SKIPS one host-planned
/// attempt within existing ceilings.
pub fn resolve_retry_veto(
    book: &ControlBook,
    session_id: &str,
    policy: &ControlPolicy,
    features: &ControlFeatures,
    mode: JevMode,
    facts: &HostControlFacts,
    candidates: &[AnswerCandidate],
    attempt: u32,
) -> ControlRetryVerdict {
    let mut result = ControlRetryVerdict {
        verdicts: Vec::new(),
        veto: false,
        classification: None,
        confidence: None,
        reason: "baseline",
    };
    // Gates first: closed gates never touch the consult ledger (H-CONTROL-1).
    if !control_gates_open(features, mode) {
        result.reason = "gates_closed";
        result.verdicts.push(ControlVerdict::Refused(
            ControlRefusal::GatesClosed("gates_closed"),
        ));
        return result;
    }
    let budget = book.snapshot(session_id);
    if !budget.available {
        // Durable accounting untrusted: refuse the consult and the veto; the
        // baseline retry policy proceeds untouched (C-BUDGET-1).
        result.reason = "accounting_unavailable";
        result.verdicts.push(ControlVerdict::Refused(
            ControlRefusal::AccountingUnavailable("accounting_unavailable"),
        ));
        return result;
    }
    // A decision may act only while its captured epoch is still the CURRENT
    // durable epoch: no consult is written into a foreign epoch either.
    if book.current_epoch_id(session_id).as_deref() != Some(budget.epoch_id.as_str()) {
        result.reason = "epoch_moved";
        result.verdicts.push(ControlVerdict::Refused(
            ControlRefusal::TriggerNotMet("epoch_moved"),
        ));
        return result;
    }
    match book.veto_consult_state(session_id, attempt) {
        VetoConsultState::Fresh => {}
        VetoConsultState::Repeated => {
            result.reason = "already_consulted";
            result.verdicts.push(ControlVerdict::Refused(
                ControlRefusal::TriggerNotMet("already_consulted"),
            ));
            return result;
        }
        VetoConsultState::Unavailable => {
            result.reason = "accounting_unavailable";
            result.verdicts.push(ControlVerdict::Refused(
                ControlRefusal::AccountingUnavailable("accounting_unavailable"),
            ));
            return result;
        }
    }
    let mut budget_refused = 0usize;
    for candidate in candidates {
        let verdict = evaluate_control_answer(
            policy,
            features,
            mode,
            ControlBoundary::RetryDecision,
            candidate,
            facts,
            &budget,
        );
        match verdict.applied().map(|acceptance| acceptance.effect.clone()) {
            Some(ControlEffectKind::RetryVeto(kind)) => {
                match book.consume(session_id, &budget.epoch_id, ControlBudgetKind::RetryVeto) {
                    Some(_) => {
                        result.veto = true;
                        result.classification = Some(kind);
                        result.confidence = verdict.applied().map(|a| a.confidence);
                        result.reason = "veto";
                    }
                    None => {
                        result.reason = "veto_budget_exhausted";
                        result.verdicts.push(ControlVerdict::Refused(
                            ControlRefusal::BudgetExhausted(ControlBudgetKind::RetryVeto),
                        ));
                    }
                }
                result.verdicts.push(verdict);
            }
            _ => {
                if matches!(
                    verdict,
                    ControlVerdict::Refused(ControlRefusal::BudgetExhausted(_))
                ) {
                    budget_refused += 1;
                }
                result.verdicts.push(verdict);
            }
        }
    }
    // The veto budget for this epoch is spent and NO candidate could apply:
    // report the truthful budget-exhausted reason instead of a silent
    // baseline (the veto can never extend ceilings).
    if !result.veto && budget_refused > 0 && candidates.len() == budget_refused {
        result.reason = "veto_budget_exhausted";
    }
    result
}

/// Stable annotation string for a paused task (truthful, never a success
/// claim); the integrator surfaces it in records/UI when a pause lands.
pub fn annotation_for_pause(reason: PauseReason) -> &'static str {
    match reason {
        PauseReason::NonprogressUncorrected => "nonprogress_uncorrected",
        PauseReason::EscalateRecommended => "escalate_recommended",
        PauseReason::VerificationUnconfirmed => "verification_unconfirmed",
        PauseReason::BudgetExhausted => "budget_exhausted",
    }
}

// ---------------------------------------------------------------------------
// Host-authored fixed feedback templates (Jev never authors these).
// ---------------------------------------------------------------------------

fn control_feedback_message(body: String) -> CustomMessage {
    CustomMessage {
        role: "custom".to_string(),
        custom_type: CONTROL_FEEDBACK_CUSTOM_TYPE.to_string(),
        content: CustomMessageContent::Text(body),
        display: true,
        details: None,
        timestamp: now_ms_i64(),
    }
}

/// Fixed template: result gap correction. Only the bounded assessment value
/// and the bounded trace evidence description are interpolated.
pub fn result_gap_feedback(assessment: &str, evidence_description: &str) -> CustomMessage {
    control_feedback_message(format!(
        "The bounded result check found this task not yet complete (assessment: {assessment}). \
Continue only the unfinished work already requested and authorized. Do not reopen a completed \
conversation, status, link, or list answer. An explicit stop or a concrete blocker wins: report \
it without restarting work. Do not infer missing tools, permissions, or a need to log in from \
model prose. Use the actual request/tool outcome. Observed evidence: {evidence_description}"
    ))
}

/// Fixed template: verification request. The host asks the main model to run
/// the actual verification; success still comes only from real outcomes.
pub fn verification_missing_feedback(evidence_description: &str) -> CustomMessage {
    control_feedback_message(format!(
        "Verification is unreported, not failed. Run only verification applicable to the requested \
work and already authorized. A completed conversational, status, link, or list answer needs \
no extra implementation or test run. Respect an explicit stop or concrete blocker. Report \
actual outcomes; never infer missing tools or login requirements from model prose. Observed \
evidence: {evidence_description}"
    ))
}

/// Fixed template: nonprogress correction.
pub fn nonprogress_feedback(identical_turns: u32) -> CustomMessage {
    control_feedback_message(format!(
        "The last {identical_turns} turns produced no new evidence. Change the approach or report \
the concrete blocker. Do not repeat the same step."
    ))
}

/// Convenience for integrator wiring: the global book.
pub fn global_book() -> &'static ControlBook {
    ControlBook::global()
}
