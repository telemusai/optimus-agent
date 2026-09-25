//! Native System One adapter. Compare observes without changing execution.
//! Active and combined modes share bounded provider/retrieval decisions with
//! request-local transformations. Compaction has a separate opt-in gate.
//! Credentials, cancellation and captured policy generations remain local.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio_util::sync::CancellationToken;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use pi_jev::config::{JevMode, JevSettings};
use pi_jev::hooks::JevObserver;
use pi_jev::mock::MockJevTransport;
use serde_json::{json, Value};

use crate::config::get_agent_dir;
use crate::core::extensions::types::SharedExtension;
use crate::core::extensions::types::{
    Extension, ExtensionContext, ExtensionEvent, ExtensionHandler, ToolExecutionEndPayload,
};
use crate::core::memory::search::MemoryHit;
use crate::core::skills::Skill;

/// Path of the internal observer extension (stable, easy to spot in logs).
pub const JEV_OBSERVER_PATH: &str = "<jev-observer-internal>";

pub(crate) mod dynamic;

/// Ownership-only stop fence, independent of credentials or observer replacement.
/// The host persists retention and combines this with its other owned domains.
pub fn request_session_retain_stop(session_id: &str) -> pi_jev::scheduler::SessionSettlementStatus {
    pi_jev::scheduler::request_session_retain_stop(session_id)
}

pub fn session_retain_status(session_id: &str) -> pi_jev::scheduler::SessionSettlementStatus {
    pi_jev::scheduler::session_retain_status(session_id)
}

pub async fn settle_session_retain_stop(session_id: &str, timeout: Duration) -> pi_jev::scheduler::SessionSettlementStatus {
    pi_jev::scheduler::settle_session_retain_stop(session_id, timeout).await
}

/// Only an explicit authorized host generation may clear a retained fence.
pub fn begin_session_retain_generation(session_id: &str, expected_generation: u64) -> bool {
    pi_jev::scheduler::begin_session_retain_generation(session_id, expected_generation)
}


/// Observed-change component of the settings identity: advances only when a
/// reload observes a settings VALUE that differs from the previous snapshot
/// (never on unchanged TTL re-reads). Combined with the DURABLE persisted
/// `write_revision` this gives hints/decisions a host settings identity that
/// moves on every authoritative write — including model A->B->A where the
/// serialized values return to identical bytes — and on any in-process
/// change this process observes.
static SETTINGS_REVISION: AtomicU64 = AtomicU64::new(1);

/// The current settings revision label for hint stamps: the durable
/// persisted write revision plus this process's observed-change counter.
pub fn settings_revision() -> String {
    let durable = load_settings_cached().write_revision;
    format!("w{}:o{}", durable, SETTINGS_REVISION.load(Ordering::SeqCst))
}

    /// ROOT-CONTRACT v7 (Agent-guidance lane) — the awaited pre-context
    /// assessment seam, called from the session's `before_request` agent hook
    /// (one assessment per provider request at the boundary that BUILDS the
    /// request). Active modes only: the decide is awaited here, the store
    /// + listener refresh happen BEFORE the loop builds the request context,
    /// so the very request this boundary serves carries the assessed hint.
    /// The stamp is CAPTURED before the await and re-checked post-await
    /// against FRESHLY loaded settings and the consumer-noted roster; the
    /// origin stamp stays immutable. Per-loop-turn cadence: one assessment
    /// per provider request, replacing the old per-provider-boundary decide.
    pub async fn before_request_assessment(
        session_id: &str,
        request_turn: u64,
        signal: Option<CancellationToken>,
    ) {
        let Some(core) = bridge_for_session(session_id) else { return };
        // Request identity must advance even without a skill assessment. Queued
        // TurnStart events can lag this awaited boundary and must not overwrite it.
        core.note_request_turn(session_id, request_turn);
        let settings_at_capture = load_settings_cached();
        let features_at_capture = settings_at_capture.effective_features(session_id);
        if !features_at_capture.skill_suggestion {
            // ROOT-CONTRACT v7: feature off — drop any cached hint so the
            // system prompt cannot keep rendering one that is no longer
            // governed. No decide runs while the feature is off.
            core.store_skill_hint(session_id, request_turn, None);
            core.notify_hint_listener(session_id);
            return;
        }
        let mode_at_capture = settings_at_capture.effective_mode(session_id);
        if !mode_at_capture.is_enabled() {
            // Off mode: clear and stop (Off removes an old hint).
            core.store_skill_hint(session_id, request_turn, None);
            core.notify_hint_listener(session_id);
            return;
        }
        if !mode_at_capture.allows_active() {
            // Compare-only: the TurnStart battery records; the hint lane
            // never installs from compare-only observations.
            return;
        }
        let roster_at_capture = core.cached_skill_roster();
        if roster_at_capture.is_empty() { return; }
        let Some(prepared) = crate::core::jev_agent_guidance::prepare_skill_suggestion(
            core.task_excerpt(session_id).as_deref().unwrap_or(""), &roster_at_capture,
        ) else { return; };
        let Some(observer) = core.observer(session_id, None) else { return };
        let dispatch_payload = serde_json::json!({
            "session_id": session_id,
            "turn": request_turn,
            // The verified state IS the outgoing state: the exact bytes
            // verify_guidance_request validated with these questions.
            "state": prepared.state,
            "policy_generation": decision_policy_generation(&settings_at_capture, session_id),
        });
        let policy = pi_jev::active::ActivationPolicy {
            enabled_categories: Default::default(),
            ..Default::default()
        };
        // The stamp CAPTURED here identifies the REQUEST inputs (revision,
        // mode, feature, turn, task, catalog) BEFORE the decide is awaited;
        // the hint carries THIS captured stamp, immutable.
        let request_stamp = core.current_hint_stamp_with(
            session_id,
            &settings_at_capture,
            features_at_capture.skill_suggestion,
            request_turn,
            &roster_at_capture,
        );
        let decide = observer.decide_prepared(&dispatch_payload, "skill_suggestion", prepared.questions, &policy);
        let outcome = if let Some(signal) = signal {
            tokio::select! { biased;
                _ = signal.cancelled() => {
                    observer.cancel_decisions(session_id);
                    // Best-effort bounded state event. The outer loop may win
                    // the same abort race and drop this hook before it writes.
                    observer.correlator().record_skipped_category(
                        session_id,
                        request_turn,
                        "skill_suggestion",
                        pi_jev::types::DecisionCategory::SkillSuggestion,
                        "assessment_cancelled",
                        pi_jev::hooks::PROMPT_VERSION,
                        mode_at_capture.as_str(),
                    );
                    return;
                }
                outcome = decide => outcome,
            }
        } else { decide.await };
        let fresh = observer.can_apply(&outcome) && outcome.turn == request_turn;
        let budget_ok = outcome.unavailable.is_none();
        // At store time the CURRENT facts are FRESHLY resolved (durable
        // settings load, not the TTL cache; the consumer-noted roster) and
        // compared against the captured origin stamp: a settings flip
        // (Off->On or A->B->A), task switch or roster change between the
        // capture and the store means the answers belong to a superseded
        // request and install nothing (NoHint clears). The catalog identity
        // is authoritatively re-checked at the consumer seam too
        // (`skill_hint_for_render` against the ACTUAL loaded skills).
        let fresh_settings = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
        let fresh_roster = core.cached_skill_roster();
        let facts_stable = core.current_hint_stamp_with(
            session_id,
            &fresh_settings,
            fresh_settings.effective_features(session_id).skill_suggestion,
            request_turn,
            &fresh_roster,
        ) == request_stamp;
        let records: &[pi_jev::types::DecisionRecord] = outcome.raw.as_ref()
            .map(|raw| raw.records.as_slice()).unwrap_or(&[]);
        match crate::core::jev_agent_guidance::skill_hint_from_raw(
            records,
            fresh && facts_stable,
            budget_ok,
            &fresh_roster,
            pi_jev::agent_guidance::CapturedSkillHintStamp::capture(request_stamp),
        ) {
            pi_jev::agent_guidance::SkillHintOutcome::Hint(hint) => {
                core.store_skill_hint(session_id, request_turn, Some(hint));
                core.notify_hint_listener(session_id);
                // This row records only the bounded host-state update. It is
                // not evidence of prompt rendering, provider receipt, or tool
                // execution; those boundaries have separate native captures.
                observer.correlator().record_skipped_category(
                    session_id,
                    request_turn,
                    "skill_suggestion",
                    pi_jev::types::DecisionCategory::SkillSuggestion,
                    "hint_state_updated",
                    pi_jev::hooks::PROMPT_VERSION,
                    mode_at_capture.as_str(),
                );
            }
            pi_jev::agent_guidance::SkillHintOutcome::NoHint(_) => {
                core.store_skill_hint(session_id, request_turn, None);
                core.notify_hint_listener(session_id);
                observer.correlator().record_skipped_category(
                    session_id,
                    request_turn,
                    "skill_suggestion",
                    pi_jev::types::DecisionCategory::SkillSuggestion,
                    "no_hint_state_updated",
                    pi_jev::hooks::PROMPT_VERSION,
                    mode_at_capture.as_str(),
                );
            }
        }
    }

fn live_bridges() -> &'static Mutex<Vec<Weak<JevBridgeCore>>> {
    static BRIDGES: OnceLock<Mutex<Vec<Weak<JevBridgeCore>>>> = OnceLock::new();
    BRIDGES.get_or_init(|| Mutex::new(Vec::new()))
}

fn control_terminal_statuses() -> &'static Mutex<HashMap<String, Value>> {
    static STATUS: OnceLock<Mutex<HashMap<String, Value>>> = OnceLock::new();
    STATUS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Surface the existing typed CONTROL terminal outcome without inventing
/// verification, completion, or execution authority. Values are bounded enums
/// and booleans only; no prompt or result content is retained.
pub(crate) fn note_control_terminal_outcome(
    session_id: &str,
    epoch_id: &str,
    outcome: &crate::core::jev_control::ControlAgentEndResult,
) {
    if session_id.is_empty() || session_id.len() > 128 {
        return;
    }
    if let Some(core) = bridge_for_session(session_id) {
        let mut sessions = core.sessions.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(book) = sessions.get_mut(session_id) {
            if book.control_epoch_id.as_deref() != Some(epoch_id) {
                return;
            }
            // These are task assessments, not proof of a completed test run.
            let observed = match &outcome.verification_state {
                pi_jev::control::ControlVerificationState::NotApplicable => Some(pi_jev::observation::VerificationEvidence::NotNeeded),
                pi_jev::control::ControlVerificationState::Unverified => Some(pi_jev::observation::VerificationEvidence::NotRun),
                _ => None,
            };
            if let Some(outcome) = observed {
                book.trace.record(pi_jev::observation::TraceEvent::VerificationObserved { outcome });
            }
        }
    }
    let verification_state = match &outcome.verification_state {
        pi_jev::control::ControlVerificationState::Unknown => "unknown",
        pi_jev::control::ControlVerificationState::NotApplicable => "not_applicable",
        pi_jev::control::ControlVerificationState::Unverified => "unverified",
        pi_jev::control::ControlVerificationState::Verified(_) => "verified",
        pi_jev::control::ControlVerificationState::Failed(_) => "failed",
    };
    let pause = outcome.pause.map(pi_jev::control::PauseReason::as_str);
    let attention_required = outcome.escalate || pause.is_some();
    let status = json!({
        "verification_state": verification_state,
        "terminal_annotation": outcome.terminal_annotation,
        "attention_required": attention_required,
        "pause": pause,
    });
    let mut statuses = control_terminal_statuses()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if statuses.len() >= MAX_TRACKED_SESSIONS && !statuses.contains_key(session_id) {
        if let Some(oldest) = statuses.keys().next().cloned() {
            statuses.remove(&oldest);
        }
    }
    statuses.insert(session_id.to_string(), status);
}

/// Worker-local read-only telemetry. None means no observed request in this
/// process, not a fabricated healthy/zero snapshot. Daemon callers must use
/// the session worker's result rather than the supervisor's empty registry.
pub fn session_status_snapshot(session_id: &str) -> Option<Value> {
    let cores: Vec<_> = live_bridges().lock().unwrap_or_else(|p| p.into_inner())
        .iter().filter_map(Weak::upgrade).collect();
    let mut result = None;
    for core in cores {
        if let Some(build) = core.observer.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            if let Some(mut status) = build.observer.session_status(session_id) {
                status["active_breaker_open"] = json!(build.observer.active_breaker_open());
                result = Some(status);
                break;
            }
        }
    }
    if let Some(compaction) = crate::core::jev_compaction::session_status(session_id) {
        let status = result.get_or_insert_with(|| json!({}));
        status["compaction"] = compaction;
    }
    if let Some(control) = control_terminal_statuses()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(session_id)
        .cloned()
    {
        let status = result.get_or_insert_with(|| json!({}));
        status["controlTerminal"] = control;
    }
    // TOOL-001: truthful offered-tool/availability diagnostics. Historical
    // observations (names, staleness, counts) only; they appear only when
    // the session already has status content, so the absent-session
    // semantics stay exactly as before. No commands, results, or secrets.
    if result.is_some() {
        if let Some(core) = bridge_for_session(session_id) {
            let diagnostics = core.tool_diagnostics(session_id);
            if !diagnostics.is_null() {
                let status = result.get_or_insert_with(|| json!({}));
                status["toolDiagnostics"] = diagnostics;
            }
        }
    }
    // Full-jev overlay status truth: presence plus its persisted stamp and
    // how many saved session values are currently masked. Read from the same
    // cached settings every other consumer uses, so the footer, daemon view
    // and decisions cannot disagree. The None/no-activity semantics for a
    // non-overlay Off run are preserved: the block only materializes when
    // there is already status content or the overlay is actually on.
    {
        let settings = load_settings_cached();
        if result.is_some() || settings.full_jev_active() {
            let masked = settings.full_jev_masked_sessions();
            let status = result.get_or_insert_with(|| json!({}));
            status["fullJev"] = json!({
                "enabled": settings.full_jev_active(),
                "stamp": settings.full_jev_stamp(),
                "maskedSessionCount": masked.len(),
            });
        }
    }
    result
}

/// The decision-segment state for a session: settings truth, credential
/// PRESENCE and worker telemetry, mapped onto the pure module's footer states.
/// Metadata only: this never creates a client or task.
fn footer_decision_state(
    settings: &pi_jev::config::JevSettings,
    session_id: &str,
) -> crate::modes::interactive::native_host::JevFooterState {
    use crate::modes::interactive::native_host::JevFooterState;
    let mode = settings.effective_mode(session_id);
    if !mode.is_enabled() {
        return JevFooterState::Off;
    }
    let present = pi_jev::config::jev_dir_for(get_agent_dir())
        .join(format!(
            "{}.{}",
            pi_jev::config::DEFAULT_KEY_ID,
            pi_jev::credential::CREDENTIAL_FILE_NAME
        ))
        .is_file()
        || pi_jev::config::EnvKeyPresence::from_env() != pi_jev::config::EnvKeyPresence::default();
    let status = session_status_snapshot(session_id);
    if !present
        && !(cfg!(debug_assertions)
            && settings
                .transport
                .as_deref()
                .is_some_and(|value| value.starts_with("mock")))
    {
        return JevFooterState::Unavailable;
    }
    if status
        .as_ref()
        .is_some_and(|value| value["in_flight"].as_u64().unwrap_or(0) > 0)
    {
        return JevFooterState::Checking;
    }
    if status.as_ref().is_some_and(|value| {
        value["fallback_reason"]
            .as_str()
            .is_some_and(|reason| !reason.is_empty())
    }) {
        return JevFooterState::Fallback;
    }
    if status.as_ref().is_some_and(|value| {
        value["active"]["last_reason"]
            .as_str()
            .is_some_and(|reason| !reason.is_empty())
    }) {
        return JevFooterState::Fallback;
    }
    match mode {
        JevMode::Compare => JevFooterState::Compare,
        JevMode::Active => JevFooterState::Active,
        JevMode::CompareAndActive => JevFooterState::CompareAndActive,
        JevMode::Off => JevFooterState::Off,
    }
}

/// Metadata-only decision footer text (labelled form).
///
/// Labels, dot glyph and colours come from the same pure module the interactive
/// tray renders, so a daemon-pushed footer and a locally published one can never
/// disagree about wording: healthy operative modes are green and name their
/// effective mode, Off is red, degraded states stay amber.
pub fn footer_status_text(session_id: &str) -> String {
    use crate::modes::interactive::native_host::footer_color_key;
    let settings = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
    let state = footer_decision_state(&settings, session_id);
    crate::modes::interactive::theme::theme::theme().fg(
        footer_color_key(state),
        &crate::modes::interactive::native_host::footer_text(state),
    )
}

/// All four themed footer segment texts for a session: decision (full + short
/// labelled narrow form) and the independent compaction state (full + short
/// labelled narrow form).
///
/// Built from ONE settings snapshot: a concurrent setting change can never make
/// the two segments on the same row disagree. The in-process publisher reads one
/// snapshot the same way, so the one-snapshot claim holds on every publish path.
/// Metadata only: no client, no task, no network.
pub struct FooterStatusForms {
    pub decision_text: String,
    pub decision_compact_text: String,
    pub compaction_text: String,
    pub compaction_compact_text: String,
}

pub fn footer_status_forms(session_id: &str) -> FooterStatusForms {
    use crate::modes::interactive::native_host::{
        compaction_state, footer_color_key, footer_compact_text, footer_compaction_compact_text,
        footer_compaction_text, footer_text,
    };
    let settings = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
    let decision_state = footer_decision_state(&settings, session_id);
    let compaction = compaction_state(&settings, session_id);
    FooterStatusForms {
        decision_text: crate::modes::interactive::theme::theme::theme().fg(
            footer_color_key(decision_state),
            &footer_text(decision_state),
        ),
        decision_compact_text: crate::modes::interactive::theme::theme::theme().fg(
            footer_color_key(decision_state),
            &footer_compact_text(decision_state),
        ),
        compaction_text: crate::modes::interactive::theme::theme::theme()
            .fg(compaction.color_key(), &footer_compaction_text(compaction)),
        compaction_compact_text: crate::modes::interactive::theme::theme::theme().fg(
            compaction.color_key(),
            &footer_compaction_compact_text(compaction),
        ),
    }
}

/// The one event the Active handler subscribes to. Registering it makes the
/// provider path dispatch one bounded extension call per request; in Off and
/// Compare the handler returns the untouched payload immediately.
pub const JEV_ACTIVE_EVENT: &str = "before_provider_request";

/// Settings-file cache TTL: one cheap stat/read per interval per process.
const SETTINGS_TTL: Duration = Duration::from_millis(250);

/// Events the observer extension subscribes to. Every handler here is
/// observe-only and always returns `None`; no session_before_* event and no
/// mutating surface is touched. `input` is bookkeeping-only (task text
/// capture) and returns `None`, which the input pipeline maps to "Continue"
/// with the original text.
///
/// Active mode adds one separate handler on `before_provider_request`
/// (see [`JEV_ACTIVE_EVENT`]). It is registered in addition to these, and it
/// is the only Jev handler in the process that may return a modified value.
pub const JEV_EVENTS: [&str; 12] = [
    "session_start",
    "agent_start",
    "turn_start",
    "turn_end",
    "input",
    "tool_call",
    "tool_execution_start",
    "tool_execution_end",
    "message_end",
    "agent_end",
    "model_select",
    "session_shutdown",
];

// ---------------------------------------------------------------------------
// Settings cache (cheap mode checks)
// ---------------------------------------------------------------------------

struct CachedSettings {
    loaded: Instant,
    settings: JevSettings,
}

fn settings_cache() -> &'static Mutex<Option<CachedSettings>> {
    static CACHE: OnceLock<Mutex<Option<CachedSettings>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Load Jev settings with a tiny TTL cache. Off-mode cost: one cached read
/// per event; no client, no network, no tasks.
fn load_settings_cached() -> JevSettings {
    let (settings, resync_handlers) = {
        let mut cache = settings_cache().lock().unwrap_or_else(|p| p.into_inner());
        if let Some(cached) = cache.as_ref() {
            if cached.loaded.elapsed() < SETTINGS_TTL {
                return cached.settings.clone();
            }
        }
        let agent_dir = get_agent_dir();
        let settings = pi_jev::config::JevSettingsStore::new(&agent_dir).load();
        // Host-authoritative settings revision: advances ONLY when the
        // reloaded settings VALUE differs from the previously observed one.
        // An external settings change (including Off->On with no consumer
        // visit during Off) therefore always moves the revision, while TTL
        // re-reads of unchanged settings do not invalidate live stamps.
        let changed = cache.as_ref().is_none_or(|cached| cached.settings != settings);
        if changed {
            SETTINGS_REVISION.fetch_add(1, Ordering::SeqCst);
        }
        // Cross-process toggle: another process may have changed Active
        // presence between the expired snapshot and this fresh read. The
        // comparison happens BEFORE the cache is replaced, and the resync
        // runs only after the settings lock is dropped, so the nested
        // `active_mode_requested` finds the fresh value: no recursion and
        // no lock held during the sync.
        let was_active = cache.as_ref().is_some_and(|cached| active_requested_from(&cached.settings));
        let now_active = active_requested_from(&settings);
        let resync_handlers = was_active != now_active;
        *cache = Some(CachedSettings {
            loaded: Instant::now(),
            settings: settings.clone(),
        });
        (settings, resync_handlers)
    };
    if resync_handlers {
        sync_active_handlers();
    }
    settings
}

/// Session-resolved full-jev overlay truth for host bookkeeping gates. Reads
/// the same cached settings truth every other consumer (decide_control, footer,
/// daemon view) uses, so bookkeeping can never disagree with decisions.
pub(crate) fn session_full_jev_active() -> bool {
    load_settings_cached().full_jev_active()
}

/// Invalidate the settings cache; the /jev UI lane can call this after
/// writing settings so a mode change is visible immediately. It also re-syncs
/// Active-handler presence, so `/jev active` takes effect without a restart.
pub fn invalidate_settings_cache() {
    *settings_cache().lock().unwrap_or_else(|p| p.into_inner()) = None;
    sync_active_handlers();
}

/// True when Active is requested through any scope: the full-jev overlay
/// (CompareAndActive while on), the global default, or a saved session
/// override. The overlay sits above every saved scope, so it is checked
/// first; handler presence follows settings.
fn active_requested_from(settings: &JevSettings) -> bool {
    settings.full_jev_active()
        || settings.global_default.is_some_and(JevMode::allows_active)
        || settings
            .sessions
            .values()
            .any(|session| session.mode.is_some_and(JevMode::allows_active))
}

fn active_mode_requested() -> bool {
    active_requested_from(&load_settings_cached())
}

/// Add or remove the Active provider-request handler on every live bridge.
///
/// The handler is the only Jev surface that can change a provider request, and
/// the runner treats "a `before_provider_request` handler exists" as "the
/// request body may change", which also decides whether a retry may reuse the
/// previous turn's semantic edges. Registering it while every session is Off or
/// Compare would change retry behavior for a user who never enabled Jev, so
/// handler presence follows the setting instead of the process.
fn sync_active_handlers() {
    let cores: Vec<_> = live_bridges()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .filter_map(Weak::upgrade)
        .collect();
    let wanted = active_mode_requested();
    for core in cores {
        core.set_active_handler(&core, wanted);
    }
}

/// A dormant adapter must exist even at default-Off startup so `/jev compare`
/// can take effect in the same session. Client and tasks remain lazy.
pub fn maybe_register_jev_observer(extensions: &mut Vec<SharedExtension>) {
    let settings = load_settings_cached();
    let core = Arc::new(JevBridgeCore::new(settings));
    {
        let mut bridges = live_bridges().lock().unwrap_or_else(|p| p.into_inner());
        bridges.retain(|core| core.strong_count() > 0);
        bridges.push(Arc::downgrade(&core));
    }
    let extension = build_observer_extension(Arc::clone(&core));
    core.attach_extension(&extension);
    core.set_active_handler(&core, active_mode_requested());
    extensions.push(extension);
}

// ---------------------------------------------------------------------------
// Bridge core
// ---------------------------------------------------------------------------

/// Bounded per-session bookkeeping. Only a capped task-text excerpt is kept
/// (for the next TurnStart snapshot); it never reaches records or logs.
#[derive(Clone, Default)]
struct SessionBook {
    last_task_excerpt: Option<String>,
    observed_tools: Vec<String>,
    turn: u64,
    /// Authoritative provider index once the awaited request hook has run.
    request_turn: Option<u64>,
    trace: pi_jev::observation::TraceObserver,
    control_epoch_id: Option<String>,
    /// TOOL-001 diagnostics: tool names advertised by the MOST RECENTLY
    /// OBSERVED provider request, plus the turn that request belonged to.
    /// This is a historical observation, not a guarantee of current
    /// availability; consumers must read `advertised_turn` for staleness.
    last_advertised_tools: Vec<String>,
    advertised_turn: Option<u64>,
    /// Bounded retention window of recent observed provider requests: 1 when
    /// the request advertised at least one tool, 0 otherwise.
    recent_tool_requests: std::collections::VecDeque<bool>,
    /// Last observed tool RESULT state and the turn it arrived in.
    last_tool_result_ok: Option<bool>,
    last_tool_result_turn: Option<u64>,
    /// CTRL-001: a designated task check FAILED in the current task epoch.
    /// Sticky within the epoch: a later unrelated passed check can never
    /// flip the epoch's verification evidence back to Passed. Reset when
    /// the host commits a new real-user task epoch (with the trace).
    verification_failed_in_epoch: bool,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): latest advisory skill hint.
    /// Assessment only — it never loads or executes a skill.
    skill_hint: Option<pi_jev::agent_guidance::SkillHint>,
    /// The turn the stored hint was assessed for. A hint is served to the
    /// system prompt only while the session is still on that turn, so a new
    /// task never inherits the previous task's hint (root wiring contract).
    skill_hint_turn: u64,
    /// ROOT blocker-3: local fingerprint of the FULL host task text (the raw
    /// Input text, hashed in-process; no raw text is stored or sent). The
    /// hint stamp binds to this identity, so two different tasks that share
    /// a bounded-excerpt prefix hash apart.
    task_identity: Option<String>,
    /// ROOT blocker-3 delivery identity: the host delivery (prepared turn
    /// action) id currently executing. Two genuinely new deliveries with
    /// byte-identical text can never reuse the previous hint.
    delivery_id: Option<String>,
}

impl SessionBook {
    fn note_control_epoch(&mut self, epoch_id: &str) {
        if !epoch_id.is_empty() && self.control_epoch_id.as_deref() != Some(epoch_id) {
            self.trace.reset();
            self.verification_failed_in_epoch = false;
            self.control_epoch_id = Some(epoch_id.to_string());
        }
    }

    fn current_turn(&self) -> u64 {
        self.request_turn.unwrap_or(self.turn)
    }

    /// TOOL-001 diagnostics: record the tool names one observed provider
    /// request actually advertised. Names only, bounded; a tool schema is
    /// public by definition (it is already on its way to the provider).
    fn note_advertised_tools(&mut self, turn: u64, names: &[String]) {
        const MAX_ADVERTISED_HISTORY: usize = 16;
        self.last_advertised_tools = names.to_vec();
        self.last_advertised_tools.truncate(MAX_ADVERTISED_HISTORY);
        self.advertised_turn = Some(turn);
        self.recent_tool_requests.push_back(!names.is_empty());
        while self.recent_tool_requests.len() > MAX_ADVERTISED_HISTORY {
            self.recent_tool_requests.pop_front();
        }
    }

    /// Record the last observed tool result state for diagnostics.
    fn note_tool_result_state(&mut self, turn: u64, is_error: bool) {
        self.last_tool_result_ok = Some(!is_error);
        self.last_tool_result_turn = Some(turn);
    }
}

struct JevBridgeCore {
    /// Cached observer keyed by the credential fingerprint it was built
    /// with; a credential rotation (or transport change) rebuilds it.
    observer: Mutex<Option<ObserverBuild>>,
    sessions: Mutex<HashMap<String, SessionBook>>,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): the session's already-loaded
    /// model-visible skill roster, pushed by the host before prompt build.
    /// Catalog metadata only — no filesystem authority, no load capability.
    skill_roster: Mutex<Vec<Skill>>,
    run_metrics: crate::core::jev_run_metrics::JevRunMetrics,
    /// Weak handle to this core's registered extension, so Active-handler
    /// presence can follow the setting at runtime.
    extension: Mutex<Weak<Mutex<Extension>>>,
    /// Per-session hint listeners registered by the host session (root
    /// blocker-1 seam): invoked whenever a session's stored hint state
    /// changes, so the session re-derives its request system prompt and the
    /// NEXT provider request consumes the new state.
    hint_listeners: Mutex<HashMap<String, Arc<dyn Fn() + Send + Sync>>>,
}

struct ObserverBuild {
    /// Cheap change probe matched BEFORE any DPAPI/client work on the hot path.
    cheap_stamp: String,
    observer: Arc<JevObserver>,
}

const MAX_TRACKED_SESSIONS: usize = 64;
const MAX_OBSERVED_TOOLS: usize = 16;

impl JevBridgeCore {
    fn new(settings: JevSettings) -> Self {
        let _ = settings;
        Self {
            observer: Mutex::new(None),
            sessions: Mutex::new(HashMap::new()),
            skill_roster: Mutex::new(Vec::new()),
            run_metrics: crate::core::jev_run_metrics::JevRunMetrics::new(std::path::PathBuf::from(get_agent_dir())),
            extension: Mutex::new(Weak::new()),
            hint_listeners: Mutex::new(HashMap::new()),
        }
    }

    /// Remember the registered extension so Active-handler presence can be
    /// added or removed later. Called once, at registration.
    fn attach_extension(&self, extension: &SharedExtension) {
        *self.extension.lock().unwrap_or_else(|p| p.into_inner()) = Arc::downgrade(extension);
    }

    /// Make Active-handler presence match `wanted`. This is the only place the
    /// provider-request handler is installed, and it is installed for every
    /// registered bridge rather than for one session: an extension handler list
    /// is process-wide, while modes are per session. Presence is therefore the
    /// conservative union ("some session is Active"), and the handler itself
    /// re-checks the effective mode of the session it is called for.
    fn set_active_handler(&self, core: &Arc<JevBridgeCore>, wanted: bool) {
        let Some(extension) = self
            .extension
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .upgrade()
        else {
            return;
        };
        let mut guard = extension.lock().unwrap_or_else(|p| p.into_inner());
        let handlers = guard
            .handlers
            .entry(JEV_ACTIVE_EVENT.to_string())
            .or_default();
        if wanted {
            if handlers.is_empty() {
                handlers.push(make_active_handler(Arc::clone(core)));
            }
        } else {
            handlers.clear();
        }
    }

    fn remember_task_text(&self, session_id: &str, text: &str) {
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if sessions.len() > MAX_TRACKED_SESSIONS && !sessions.contains_key(session_id) {
            if let Some(oldest) = sessions.keys().next().cloned() {
                sessions.remove(&oldest);
            }
        }
        let book = sessions.entry(session_id.to_string()).or_default();
        // Redacted while the excerpt is built: this text is later sent to
        // SystemOne as `user_text_excerpt`.
        book.last_task_excerpt = Some(pi_jev::redact::bounded_excerpt(text, 400));
        // Root blocker-3: the STAMP binds to the FULL task text, fingerprinted
        // locally (the raw text never leaves the process in the stamp).
        book.task_identity = Some(pi_jev::agent_guidance::guidance_stamp_hash(&[text]));
    }

    /// Root blocker-3: the full host task text's local fingerprint (the
    /// stamp's task identity), recorded at Input time.
    fn task_identity(&self, session_id: &str) -> Option<String> {
        self.sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session_id)
            .and_then(|book| book.task_identity.clone())
    }

    /// Root blocker-3: the CURRENT host delivery id for stamp binding.
    fn delivery_id(&self, session_id: &str) -> String {
        self.sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session_id)
            .and_then(|book| book.delivery_id.clone())
            .unwrap_or_default()
    }

    fn task_excerpt(&self, session_id: &str) -> Option<String> {
        let sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sessions
            .get(session_id)
            .and_then(|book| book.last_task_excerpt.clone())
    }

    fn note_tool_call(&self, session_id: &str, tool_name: &str) {
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let book = sessions.entry(session_id.to_string()).or_default();
        if !book.observed_tools.iter().any(|name| name == tool_name) {
            book.observed_tools.push(tool_name.to_string());
            book.observed_tools.truncate(MAX_OBSERVED_TOOLS);
        }
    }

    /// CTRL-001: reset epoch-scoped verification bookkeeping (and the
    /// observation trace) when the host commits a new real-user task epoch.
    /// Evidence from a previous task never bleeds into the new one.
    fn note_control_epoch(&self, session_id: &str, epoch_id: &str) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(book) = sessions.get_mut(session_id) {
            book.note_control_epoch(epoch_id);
        }
    }

    /// TOOL-001: remember which tool names one observed provider request
    /// actually advertised, with the request turn for staleness reporting.
    fn note_advertised_tools(&self, session_id: &str, turn: u64, names: &[String]) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        sessions
            .entry(session_id.to_string())
            .or_default()
            .note_advertised_tools(turn, names);
    }

    /// TOOL-001: bounded names from the last OBSERVED provider request.
    fn last_advertised_tools(&self, session_id: &str) -> Vec<String> {
        self.sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session_id)
            .map(|book| book.last_advertised_tools.clone())
            .unwrap_or_default()
    }

    /// TOOL-001: whether the recorded advertisement belongs to the current
    /// turn. False means stale (from an earlier request), never "missing".
    fn advertised_tools_is_current(&self, session_id: &str) -> bool {
        self.sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session_id)
            .is_some_and(|book| {
                book.advertised_turn.is_some_and(|turn| turn == book.current_turn())
            })
    }

    /// TOOL-001: the offered-tools fact for CONTROL evidence text. Bounded
    /// names of the last OBSERVED advertisement plus explicit staleness;
    /// never a guarantee of current availability, and never derived from a
    /// blanket model claim.
    fn advertised_tools_fact(&self, session_id: &str) -> Option<String> {
        let sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        let book = sessions.get(session_id)?;
        if book.last_advertised_tools.is_empty() {
            return None;
        }
        let names = book.last_advertised_tools.join(",");
        let current = book
            .advertised_turn
            .is_some_and(|turn| turn == book.current_turn());
        Some(if current {
            format!("advertised_tools_last_request={names}")
        } else {
            format!("advertised_tools_last_request={names} (stale, from an earlier request)")
        })
    }

    fn observed_tools(&self, session_id: &str) -> Vec<String> {
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(session_id)
            .map(|book| book.observed_tools.clone())
            .unwrap_or_default()
    }

    /// ROOT-CONTRACT v7 (Agent-guidance lane): cache the session's
    /// model-visible roster (already-loaded metadata; bounded defensively).
    fn set_skill_roster(&self, skills: &[Skill]) {
        const MAX_ROSTER_CACHE: usize = 128;
        let mut roster = self.skill_roster.lock().unwrap_or_else(|p| p.into_inner());
        *roster = skills.iter().take(MAX_ROSTER_CACHE).cloned().collect();
    }

    fn cached_skill_roster(&self) -> Vec<Skill> {
        self.skill_roster.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn store_skill_hint(&self, session_id: &str, turn: u64, hint: Option<pi_jev::agent_guidance::SkillHint>) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        if sessions.len() >= MAX_TRACKED_SESSIONS && !sessions.contains_key(session_id) { return; }
        let book = sessions.entry(session_id.to_string()).or_default();
        book.skill_hint = hint;
        book.skill_hint_turn = turn;
    }

    /// Root blocker-1 seam, host side: the session registers a listener so a
    /// hint-state change can refresh the request-local prompt derivation.
    fn set_hint_listener(&self, session_id: &str, listener: Arc<dyn Fn() + Send + Sync>) {
        self.hint_listeners
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(session_id.to_string(), listener);
    }

    fn notify_hint_listener(&self, session_id: &str) {
        let listener = self
            .hint_listeners
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session_id)
            .cloned();
        if let Some(listener) = listener {
            listener();
        }
    }

    /// The hint is served only while the session is still on the turn it was
    /// assessed for: a new task (turn advanced) never inherits it.
    fn skill_hint_for(&self, session_id: &str) -> Option<pi_jev::agent_guidance::SkillHint> {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner()).get(session_id)
            .filter(|book| book.current_turn() == book.skill_hint_turn)
            .and_then(|book| book.skill_hint.clone())
    }

    /// Advisory skill suggestion over the already-loaded roster: Compare only
    /// OBSERVES the closed-set question; Active additionally DECIDES it with an
    /// empty appliable policy (nothing can ever apply) and stores the hint for
    /// the system prompt. Assessment only — never loads or executes a skill.
    /// The CURRENT authoritative hint-stamp facts for this session (root
    /// wiring contract). Production captures it BEFORE the decide is awaited
    /// and compares a freshly resolved copy at store; consumption re-derives
    /// it per request. Strict equality only: any drift fails open to NoHint.
    fn current_hint_stamp_with(
        &self,
        session_id: &str,
        settings: &JevSettings,
        feature_enabled: bool,
        turn: u64,
        roster: &[Skill],
    ) -> pi_jev::agent_guidance::SkillHintStamp {
        // Root blocker-3: the catalog identity covers id AND description
        // (the assessed entries), and the task identity is the FULL host
        // task text's local fingerprint, not the bounded presentation excerpt.
        // The revision label derives from the PASSED settings' durable write
        // revision plus the process observed-change counter, so a post-await
        // re-derivation from a FRESH durable load catches any authoritative
        // write (including A->B->A) that happened during the await.
        let catalog =
            crate::core::jev_agent_guidance::skill_roster_view_with_skips(roster).0;
        let task_identity = self.task_identity(session_id).unwrap_or_default();
        let revision = format!("w{}:o{}", settings.write_revision, SETTINGS_REVISION.load(Ordering::SeqCst));
        crate::core::jev_agent_guidance::skill_hint_stamp(
            &revision,
            settings.effective_mode(session_id).as_str(),
            feature_enabled,
            turn,
            &self.delivery_id(session_id),
            &task_identity,
            &catalog,
        )
    }

    /// ROOT-CONTRACT v7 (Agent-guidance lane), compare-only battery: record the
    /// skill-suggestion assessment as a record-only comparison. The outgoing
    /// state IS the adapter's verified state (`PreparedGuidance.state`) — the
    /// exact bytes `verify_guidance_request` validated — merged into a clone of
    /// the real event envelope (session/turn/correlation untouched).
    async fn observe_skill_suggestion_compare(
        &self,
        session_id: &str,
        payload: &Value,
        observer: &Arc<JevObserver>,
        mode: JevMode,
    ) {
        let roster = self.cached_skill_roster();
        if roster.is_empty() { return; }
        let Some(prepared) = crate::core::jev_agent_guidance::prepare_skill_suggestion(
            self.task_excerpt(session_id).as_deref().unwrap_or(""), &roster,
        ) else { return; };
        let mut dispatch_payload = payload.clone();
        dispatch_payload["state"] = prepared.state.clone();
        observer.observe_prepared(&dispatch_payload, "skill_suggestion", prepared.questions);
    }


    fn forget_session(&self, session_id: &str) {
        self.sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(session_id);
    }

    /// Root blocker-3: note the CURRENT host delivery id (prepared turn
    /// action) so the hint stamp binds to the genuine delivery, not just the
    /// task text. Called at the commit seam before each real run.
    fn note_delivery(&self, session_id: &str, delivery_id: &str) {
        self.sessions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(session_id.to_string())
            .or_default()
            .delivery_id = Some(delivery_id.to_string());
    }

    fn note_turn(&self, session_id: &str, turn: u64) {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner())
            .entry(session_id.to_string()).or_default().turn = turn;
    }

    fn note_request_turn(&self, session_id: &str, turn: u64) {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner())
            .entry(session_id.to_string()).or_default().request_turn = Some(turn);
    }

    fn turn(&self, session_id: &str) -> u64 {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner())
            .get(session_id).map(SessionBook::current_turn).unwrap_or(0)
    }

    fn note_observation(&self, session_id: &str, event: &ExtensionEvent) {
        use pi_jev::observation::{ObservedStopReason, TraceEvent, VerificationEvidence};
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        let book = sessions.entry(session_id.to_string()).or_default();
        match event {
            ExtensionEvent::TurnStart(_) => book.trace.record(TraceEvent::TurnStarted),
            ExtensionEvent::ToolExecutionEnd(payload) => {
                book.trace.record(TraceEvent::ToolEnded { is_error: payload.is_error });
                book.note_tool_result_state(book.current_turn(), payload.is_error);
                // CTRL-001 evidence adapter: only the native ipython tool —
                // the sole supported producer of the typed script-report
                // pipe — can record verification evidence, and only from
                // strictly-shaped, explicitly designated task checks. The
                // outcome is the host-measured, validated script report:
                // never transport success, printed text, activity counts, or
                // an executionReports-shaped payload echoed by another tool.
                if payload.tool_name == SCRIPT_REPORT_TOOL_NAME {
                    if let Some(reports) = designated_verification_reports(payload) {
                        if reports.iter().any(|report| report.failed()) {
                            book.verification_failed_in_epoch = true;
                            book.trace.record(TraceEvent::VerificationObserved {
                                outcome: VerificationEvidence::Failed,
                            });
                        } else if !book.verification_failed_in_epoch {
                            // A failed designated check is sticky within the
                            // task epoch: a later unrelated passed check can
                            // never flip the epoch's evidence to Passed.
                            book.trace.record(TraceEvent::VerificationObserved {
                                outcome: VerificationEvidence::Passed,
                            });
                        }
                    }
                }
            }
            ExtensionEvent::MessageEnd(payload) if payload.message.get("role").and_then(Value::as_str) == Some("assistant") => {
                let stop = payload.message.get("stopReason").and_then(Value::as_str).unwrap_or("");
                let kind = observed_failure_kind(&payload.message);
                book.trace.record(TraceEvent::AssistantEnded { stop_reason: ObservedStopReason::from_stop_reason(stop), failure_kind: kind });
            }
            _ => {}
        }
    }

    /// TOOL-001: bounded, truthful offered-tool/availability diagnostics for
    /// one session. Historical observations with explicit staleness data;
    /// `None` bookkeeping stays absent (never a fabricated healthy state).
    /// Tool NAMES only — no commands, no results, no secrets.
    fn tool_diagnostics(&self, session_id: &str) -> Value {
        let sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        let Some(book) = sessions.get(session_id) else {
            return Value::Null;
        };
        let advertised_turn = book.advertised_turn;
        let current_turn = book.current_turn();
        let with_tools = book.recent_tool_requests.iter().filter(|flag| **flag).count();
        let observed_requests = book.recent_tool_requests.len();
        let verification = match book.trace.summary().verification {
            pi_jev::observation::VerificationEvidence::Passed => "passed",
            pi_jev::observation::VerificationEvidence::Failed => "failed",
            pi_jev::observation::VerificationEvidence::NotRun => "not_run",
            pi_jev::observation::VerificationEvidence::NotNeeded => "not_needed",
            pi_jev::observation::VerificationEvidence::Unknown => "unknown",
        };
        json!({
            // Last OBSERVED advertisement, never a guaranteed current list.
            "lastAdvertisedTools": book.last_advertised_tools.clone(),
            "advertisedAtTurn": advertised_turn,
            "currentTurn": current_turn,
            "isCurrentTurn": advertised_turn.is_some_and(|turn| turn == current_turn),
            "recentRequestsRetainingTools": format!("{with_tools}/{observed_requests}"),
            "lastToolResult": match book.last_tool_result_ok {
                Some(false) => "error",
                Some(true) => "ok",
                None => "none",
            },
            "turnsSinceLastToolResult": book
                .last_tool_result_turn
                .map(|turn| current_turn.saturating_sub(turn)),
            "verificationEvidence": verification,
        })
    }

    fn observation(&self, session_id: &str) -> pi_jev::observation::TraceSummary {
        self.sessions.lock().unwrap_or_else(|p| p.into_inner()).get(session_id)
            .map(|book| book.trace.summary()).unwrap_or_default()
    }

    async fn observe_optional(&self, session_id: &str, event: &ExtensionEvent,
        ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>, handler_event: &'static str) {
        let settings = load_settings_cached();
        let features = settings.effective_features(session_id);
        let stage = match event {
            ExtensionEvent::TurnEnd(_) if features.loop_control || features.retry_classification => pi_jev::snapshot::SnapshotStage::TurnEnd,
            ExtensionEvent::AgentEnd(_) if features.result_sufficiency || features.loop_control || features.verification
                || features.retry_classification || features.trace_observer => pi_jev::snapshot::SnapshotStage::AgentEnd,
            _ => return,
        };
        // Only explicitly enabled observational categories run in Active-only mode.
        let Some(core) = bridge_for_session(session_id) else { return; };
        let Some((_, mut payload)) = bridge_event(&core, event, ctx, session_id, handler_event, &settings) else { return; };
        payload["policy_generation"] = json!(decision_policy_generation(&settings,session_id));
        payload["state"]["features"] = json!(features);
        let Some(observer) = self.observer(session_id, Some(ctx.ui())) else { return; };
        let state = payload.get("state").cloned().unwrap_or(Value::Null);
        let Ok(snapshot) = pi_jev::snapshot::StateSnapshot::new(stage, session_id, self.turn(session_id), 0, None, state, Vec::new()) else { return; };
        let mut questions = Vec::new();
        for evaluator in pi_jev::evaluators::for_boundary(stage) {
            let enabled = match evaluator.category().as_str() {
                "result_sufficiency" => features.result_sufficiency,
                "continue_stop_escalate" => features.loop_control,
                "first_pass_verification" => features.verification,
                "retry_classification" => features.retry_classification,
                "trace_assessment" => features.trace_observer,
                _ => false,
            };
            if enabled {
                if let pi_jev::evaluators::EvaluatorOutput::Questions(mut prepared) = evaluator.evaluate(&snapshot) { questions.append(&mut prepared); }
            }
        }
        let policy = pi_jev::active::ActivationPolicy { enabled_categories: Default::default(), ..Default::default() };
        let outcome = observer.decide_prepared(&payload, stage.as_str(), questions, &policy).await;
        observer.record_active(&outcome, &BTreeMap::new());
    }

    /// Off-mode immediate effect: cancel queued/in-flight comparison work
    /// for the session and drop its bounded bookkeeping. The observer slot
    /// itself is kept (rebuilt lazily on the next Compare event).
    fn drop_session_work(&self, session_id: &str) {
        if let Some(build) = self
            .observer
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            build.observer.cancel_session(session_id);
        }
        self.forget_session(session_id);
    }

    fn drop_decision_work(&self, session_id: &str) {
        if let Some(build) = self.observer.lock().unwrap_or_else(|p| p.into_inner()).as_ref() { build.observer.cancel_decisions(session_id); }
        if load_settings_cached().effective_compaction_enabled(session_id) {
            if let Some(book) = self.sessions.lock().unwrap_or_else(|p| p.into_inner()).get_mut(session_id) { *book = SessionBook::default(); }
        } else { self.forget_session(session_id); }
    }

    fn own_session(&self, session_id: &str) {
        let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
        if sessions.len() >= MAX_TRACKED_SESSIONS && !sessions.contains_key(session_id) { return; }
        sessions.entry(session_id.to_string()).or_default();
    }

    /// Effective mode for one session (cheap cached settings read).
    fn effective_mode(&self, session_id: Option<&str>) -> JevMode {
        load_settings_cached().effective_mode(session_id.unwrap_or(""))
    }

    /// Lazily construct the observer on the first Compare-mode event, and
    /// rebuild it when the effective credential changes (rotation). In Off
    /// nothing is ever constructed.
    fn observer(&self, session_id: &str, ui: Option<Arc<dyn crate::core::extensions::types::ExtensionUiContext>>) -> Option<Arc<JevObserver>> {
        let settings = load_settings_cached();
        if !settings.effective_mode(session_id).is_enabled() && !settings.effective_compaction_enabled(session_id) {
            return None;
        }
        // Cached fast path: the cheap change probe (transport selector,
        // credential envelope stamp, env presence booleans) before any
        // credential read or client build. No DPAPI decrypt and no reqwest
        // client on the hot path when nothing changed.
        let cheap = cheap_credential_stamp(&settings);
        if let Some(cached) = self
            .observer
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            if cached.cheap_stamp == cheap {
                return Some(Arc::clone(&cached.observer));
            }
        }
        // A changed envelope generation must rebuild even if it decrypts to
        // the same key: the dispatch gate captures the generation, not only
        // key identity. Reusing that observer would reject all later work.
        let (transport, credential, _fingerprint) = build_transport(&settings);
        let effective = settings.effective_mode(session_id);
        // The transport is data-only. Independent compaction can use it while decisions are Off.
        let client_mode = if effective.is_enabled() { effective } else { JevMode::Compare };
        let system_one: Arc<dyn pi_jev::types::SystemOne> = match
            pi_jev::client::JevSystemOne::new(
                client_mode,
                credential,
                transport,
                pi_jev::client::JevLimits::default(),
                Arc::new(pi_jev::client::JevStats::default()),
            ) {
            Ok(client) => Arc::new(client),
            // Construction can only fail on an unusable credential here;
            // comparison requests then fail closed to logged skips and no
            // network object is used.
            Err(_) => Arc::new(pi_jev::client::DisabledSystemOne::new(effective)),
        };
        // The observer re-checks the effective mode itself; the gate reads the
        // same cached settings the handlers use (one cheap read per event).
        let observed_stamp = cheap.clone();
        let independent_stamp = cheap.clone();
        let authoritative_stamp = cheap.clone();
        let footer_session = session_id.to_string();
        let on_terminal = ui.map(|ui| {
            let footer_session = footer_session.clone();
            Arc::new(move |session_id: &str| {
                if session_id == footer_session {
                    ui.set_status("jev".into(), Some(footer_status_text(session_id)));
                }
            }) as Arc<dyn Fn(&str) + Send + Sync>
        });
        let config = pi_jev::hooks::JevObserverConfig {
            mode_gate: Arc::new(move |session_id: Option<&str>| {
                // Dispatch and completion use current settings, not the event
                // cache: a queued request cannot outlive Off/key rotation.
                // ROOT-CONTRACT v9: the SAME single load also resolves the
                // requested Jev model, so a request's mode gate and its
                // `model` field can never disagree; both are captured at the
                // boundary BEFORE any await. No late global getter is mixed
                // into an old payload; the durable write_revision invalidates
                // held work across model A->B->A at the apply boundary.
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                if cheap_credential_stamp(&current) != observed_stamp {
                    return (JevMode::Off, current.requested_model_or_default().to_string());
                }
                (
                    current.effective_mode(session_id.unwrap_or("")),
                    current.requested_model_or_default().to_string(),
                )
            }),
            policy_generation: Arc::new(|session_id, independent| {
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                if independent {
                    compaction_policy_generation(&current, session_id)
                } else {
                    decision_policy_generation(&current, session_id)
                }
            }),
            authoritative_gate: Some(Arc::new(move |session_id, independent| {
                // One durable snapshot supplies mode, requested model, feature/
                // compaction generation and permission. This prevents a newer
                // model from being combined with features/generation captured
                // by an older payload.
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                let session_id = session_id.unwrap_or("");
                let credential_current =
                    cheap_credential_stamp(&current) == authoritative_stamp;
                let mode = if credential_current {
                    current.effective_mode(session_id)
                } else {
                    JevMode::Off
                };
                let policy_generation = if independent {
                    compaction_policy_generation(&current, session_id)
                } else {
                    decision_policy_generation(&current, session_id)
                };
                let allowed = if independent {
                    credential_current && current.effective_compaction_enabled(session_id)
                } else {
                    credential_current && mode.is_enabled()
                };
                pi_jev::hooks::JevRequestGate {
                    mode,
                    requested_model: current.requested_model_or_default().to_string(),
                    policy_generation,
                    allowed,
                }
            })),
            independent_gate: Arc::new(move |session_id| {
                let current = pi_jev::config::JevSettingsStore::new(get_agent_dir()).load();
                cheap_credential_stamp(&current) == independent_stamp && current.effective_compaction_enabled(session_id)
            }),
            on_terminal,
            ..pi_jev::hooks::JevObserverConfig::default()
        };
        let records_path = std::path::Path::new(&get_agent_dir())
            .join("jev")
            .join("records.jsonl");
        let observer = JevObserver::new(config, system_one, records_path);
        let build = ObserverBuild {
            cheap_stamp: cheap,
            observer: Arc::clone(&observer),
        };
        let previous = self.observer.lock().unwrap_or_else(|p| p.into_inner()).replace(build);
        if let Some(previous) = previous {
            previous.observer.shutdown();
        }
        Some(observer)
    }
}


/// Cheap change probe for the observer cache: the transport selector, the
/// credential envelope's metadata stamp and the env presence booleans. Reads
/// no secret and performs no DPAPI work; a changed stamp triggers the full
/// transport build (whose fingerprint includes the decrypted key's hash).
fn cheap_credential_stamp(settings: &JevSettings) -> String {
    credential_stamp_at(settings, &std::path::PathBuf::from(get_agent_dir()))
}

fn credential_stamp_at(settings: &JevSettings, agent_dir: &std::path::Path) -> String {
    let transport_selector = settings.transport.clone().unwrap_or_default();
    let envelope = std::fs::metadata(
        pi_jev::config::jev_dir_for(agent_dir).join(format!("{}.{}",
            pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME)),
    )
    .ok()
    .map(|meta| {
        let modified = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        (modified, meta.len())
    })
    .unwrap_or((0, 0));
    // Only hashes leave this scope; changing an already-present env key must
    // invalidate the old client just like replacing the saved envelope.
    let env_fingerprint = |name: &str| std::env::var(name).ok()
        .map(|value| pi_jev::credential::SecretString::new(value).key_fingerprint())
        .unwrap_or_default();
    format!(
        "cheap:{transport_selector}:{modified}:{len}:{typesafe}:{jev}:{full_jev}",
        modified = envelope.0,
        len = envelope.1,
        typesafe = env_fingerprint(pi_jev::config::ENV_TYPESAFE_API_KEY),
        jev = env_fingerprint(pi_jev::config::ENV_JEV_API_KEY),
        // The full-jev overlay changes every effective resolution; a toggle
        // must invalidate in-flight work exactly like a key rotation.
        full_jev = settings.full_jev_stamp(),
    )
}

/// Binding credential order (DESIGN.md 3.1): saved credential first, then
/// env `TYPESAFE_API_KEY`, then alias `JEV_API_KEY`. The value is wrapped in
/// a SecretString and never logged, stored or copied elsewhere.
fn load_credential() -> Option<pi_jev::credential::SecretString> {
    if let Some(saved) = load_saved_credential() {
        return Some(saved);
    }
    for env_name in ["TYPESAFE_API_KEY", "JEV_API_KEY"] {
        if let Ok(value) = std::env::var(env_name) {
            let trimmed = value.trim().to_string();
            if !trimmed.is_empty() {
                return Some(pi_jev::credential::SecretString::new(trimmed));
            }
        }
    }
    None
}

/// Selects the transport per settings. Mock selection is a development/test
/// aid and only honored in debug builds; production release builds always
/// use the real endpoint path (saved credential, then env fallback).
fn build_transport(
    settings: &JevSettings,
) -> (
    Arc<dyn pi_jev::types::Transport>,
    pi_jev::credential::SecretString,
    String,
) {
    let synthetic_credential =
        || pi_jev::credential::SecretString::new("jev-mock-credential-not-a-real-key");
    // An UNKNOWN transport value falls through to the real endpoint below
    // (fail toward honesty, never toward a silent mock); a warning-level note
    // is unnecessary because /jev status shows the effective source.
    if cfg!(debug_assertions) {
        match settings.transport.as_deref() {
            Some("mock") => {
                return (
                    Arc::new(MockJevTransport::all_valid()),
                    synthetic_credential(),
                    "mock".to_string(),
                );
            }
            // Development-only hostile variant: valid, maximum-confidence
            // answers chosen adversarially. Proves nothing Jev says can act.
            Some("mock-hostile") => {
                return (
                    Arc::new(HostileTestTransport),
                    synthetic_credential(),
                    "mock-hostile".to_string(),
                );
            }
            // Test-only: answers arrive AFTER the turn boundary (stale) and
            // can only ever become applied=false records.
            Some("mock-delayed") => {
                return (
                    Arc::new(MockJevTransport::scripted(vec![
                        pi_jev::mock::MockStep::SlowResponse { delay_ms: 600 },
                    ])),
                    synthetic_credential(),
                    "mock-delayed".to_string(),
                );
            }
            // Test-only: protocol-violating (malformed) answers. The client
            // validation rejects them; they land as logged skips and cannot
            // influence the agent loop.
            Some("mock-malformed") => {
                return (
                    Arc::new(MockJevTransport::scripted(vec![
                        pi_jev::mock::MockStep::Body(TRUNCATED_MALFORMED_BODY.to_string()),
                    ])),
                    synthetic_credential(),
                    "mock-malformed".to_string(),
                );
            }
            // Test-only fixture for the native skill-hint wiring capture: a
            // REQUEST-AWARE transport that answers suggestion decides with a
            // fixed hint/no-hint/hint sequence (need high, inverse low, act
            // high, decisive rank on alpha-skill; then rank `none`; then hint)
            // and answers every OTHER shadow-battery request with valid
            // per-type fixture answers. Debug assertions only; production
            // ignores unknown transport values entirely.
            Some("mock-hint") => {
                return (
                    Arc::new(HintFixtureTransport::default()),
                    synthetic_credential(),
                    "mock-hint".to_string(),
                );
            }
            // Test-only fixture for the native CONTROL/compaction held-identity
            // regressions: a REQUEST-AWARE transport. The test installs an
            // optional response callback (used to perform REAL authoritative
            // settings saves from inside the decide await) and every call is
            // recorded (question ids + the v9 requested-model stamp only; no
            // raw state content is retained). Without a callback it answers
            // every request with valid per-type fixture answers, exactly like
            // the hint fixture's shadow batteries. Debug assertions only;
            // production ignores unknown transport values entirely.
            Some("mock-control") => {
                return (
                    Arc::new(ControlFixtureTransport),
                    synthetic_credential(),
                    "mock-control".to_string(),
                );
            }
            _ => {}
        }
    }
    // The real endpoint needs a credential; without one, comparison requests
    // fail closed to logged skips.
    match load_credential() {
        Some(secret) => {
            let fingerprint = format!("saved:{}", secret.key_fingerprint());
            match pi_jev::client::JevHttpTransport::new(
                pi_jev::client::JevLimits::default(),
                secret.clone(),
            ) {
                Ok(transport) => (Arc::new(transport), secret, fingerprint),
                Err(_) => (
                    Arc::new(NoCredentialTransport),
                    synthetic_credential(),
                    "unavailable".to_string(),
                ),
            }
        }
        None => (
            Arc::new(NoCredentialTransport),
            synthetic_credential(),
            "no_credential".to_string(),
        ),
    }
}

#[cfg(test)]
static TEST_CATALOG_TRANSPORT: OnceLock<Mutex<Option<Arc<dyn pi_jev::types::Transport>>>> =
    OnceLock::new();

/// Install or clear the native-command catalog transport in library tests.
/// This seam is compiled out of non-test builds; it exists only so the real
/// `/jev models` dispatch can prove one successful provider call offline.
#[cfg(test)]
pub(crate) fn set_test_catalog_transport(
    transport: Option<Arc<dyn pi_jev::types::Transport>>,
) {
    *TEST_CATALOG_TRANSPORT
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = transport;
}

/// ROOT-CONTRACT v9: transport + limits for the EXPLICIT `/jev models` catalog
/// command ONLY. Reuses the single `build_transport` selection path: debug
/// mock selectors stay debug-only (a scripted mock either serves a catalog
/// step or reports an honest `Internal`), and release builds use the SAME
/// credential order (saved credential, then env) against the documented
/// endpoint with the default limits. Returns `None` when no usable credential
/// is configured so the command reports honest unavailability WITHOUT any
/// network attempt. This helper performs no I/O of its own; the single bounded
/// catalog fetch happens only in the explicit command handler. Nothing here
/// selects a model or writes settings.
pub(crate) fn catalog_transport_for_command(
    settings: &JevSettings,
) -> Option<(
    Arc<dyn pi_jev::types::Transport>,
    pi_jev::client::JevLimits,
)> {
    #[cfg(test)]
    if let Some(transport) = TEST_CATALOG_TRANSPORT
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
    {
        return Some((transport, pi_jev::client::JevLimits::default()));
    }
    let (transport, _credential, fingerprint) = build_transport(settings);
    if fingerprint == "no_credential" || fingerprint == "unavailable" {
        return None;
    }
    Some((transport, pi_jev::client::JevLimits::default()))
}

/// ROOT-CONTRACT v9: the effective credential for the `/jev model set`
/// credential-overlap refusal ONLY. The value stays in memory inside the
/// `SecretString`; it is never logged, echoed or persisted. Returns `None`
/// when no credential is configured — the overlap check then cannot fire,
/// which is safe because there is no secret to overlap.
pub(crate) fn credential_for_model_overlap() -> Option<pi_jev::credential::SecretString> {
    load_credential()
}

/// Development/test-only transport: valid-shaped, maximum-confidence
/// adversarial answers (mock-hostile). Proves nothing Jev says can act.
/// Selected only via the settings `transport` override; never by default.
struct HostileTestTransport;

impl pi_jev::types::Transport for HostileTestTransport {
    fn post(
        &self,
        request: &pi_jev::types::SystemOneRequest,
        _timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<pi_jev::types::SystemOneResponse, pi_jev::error::JevError>>
    {
        let response = hostile_response_for(request);
        Box::pin(async move { Ok(response) })
    }
}

/// Fixed decide-answer bodies for the debug-only `mock-hint` transport:
/// a HINT-shaped set (decisive rank on alpha-skill; need high, inverse low,
/// act high, fit high) and a NO-HINT set (rank abstains to `none`).
/// Test-only request-aware fixture transport for `mock-hint` (debug builds).
/// The suggestion decide is keyed on the request's ACTUAL assessed state —
/// `request.state["user_text_excerpt"]`, the verified `PreparedGuidance.state`
/// the adapter disclosed — so the fixture answers by task identity, never by
/// call order: an excerpt containing "alpha retry policy" gets the
/// hint-shaped answer set, "beta cleanup flow" gets the rank-`none` set, and
/// every other (shadow-battery) request gets valid per-type fixture answers.
#[derive(Default)]
struct HintFixtureTransport;

/// Test-only scratchpad for the `mock-hint` fixture transport (debug builds):
/// the state and question ids of the most recent skill-suggestion decide that
/// reached the transport.
#[cfg(debug_assertions)]
static HINT_FIXTURE_LAST_SUGGESTION: std::sync::Mutex<Option<(serde_json::Value, Vec<String>)>> =
    std::sync::Mutex::new(None);

impl HintFixtureTransport {
    /// The rank answer is built from the request's ACTUAL criteria: the
    /// catalog is whatever roster the consumer seam noted (project skills plus
    /// any user-scope skills), so the fixture can never assume fixed ids. The
    /// hint case ranks the first catalog id decisively; the none case ranks
    /// `none`. Both keep the documented shape: choice = argmax, distribution
    /// keys = criteria keys, sum 1, confidence in range.
    fn suggestion_body(excerpt: &str, criteria_keys: &[String]) -> serde_json::Value {
        let hint = excerpt.contains("alpha retry policy");
        let ranked: String = criteria_keys
            .iter()
            .filter(|id| id.as_str() != "none")
            .next()
            .cloned()
            .unwrap_or_else(|| "none".to_string());
        // Every criteria key is present; the mass concentrates on the choice
        // so the documented argmax rule holds and the sum is exactly 1.
        let mut probabilities = std::collections::BTreeMap::new();
        for id in criteria_keys {
            probabilities.insert(id.clone(), 0.0_f64);
        }
        if hint {
            *probabilities.entry(ranked.clone()).or_insert(0.0) = 0.9;
            *probabilities.entry("none".to_string()).or_insert(0.0) = 0.1;
        } else {
            *probabilities.entry("none".to_string()).or_insert(0.0) = 1.0;
        }
        let choice = if hint { ranked } else { "none".to_string() };
        // The adapter prepares rank + three gates (ids .0-.3); the fit second
        // pass (.4) is never prepared by the adapter, so it is never answered.
        json!({
            "model": "jev-mock-hint/1",
            "answers": {
                "skill_suggestion.0": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": 0.9
                },
                "skill_suggestion.1": {"type": "noul", "noul": 0.9},
                "skill_suggestion.2": {"type": "noul", "noul": 0.1},
                "skill_suggestion.3": {"type": "noul", "noul": 0.9}
            },
            "usage": {"input_tokens": 5, "output_tokens": 5}
        })
    }
}

impl pi_jev::types::Transport for HintFixtureTransport {
    fn post(
        &self,
        request: &pi_jev::types::SystemOneRequest,
        _timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<pi_jev::types::SystemOneResponse, pi_jev::error::JevError>> {
        let is_suggestion = request
            .questions
            .keys()
            .any(|id| id.starts_with("skill_suggestion."));
        if is_suggestion {
            let excerpt = request
                .state
                .get("user_text_excerpt")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            #[cfg(debug_assertions)]
            {
                *HINT_FIXTURE_LAST_SUGGESTION
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) =
                    Some((request.state.clone(), request.questions.keys().cloned().collect()));
                HINT_FIXTURE_SUGGESTION_CALLS
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Clone under the fixture mutex, then drop the guard before
                // invoking user test code. The callback may inspect fixture
                // state or schedule an abort and must never self-deadlock.
                let callback = {
                    HINT_FIXTURE_ON_SUGGESTION
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .clone()
                };
                if let Some(callback) = callback {
                    callback();
                }
            }
            let criteria_keys: Vec<String> = request
                .questions
                .get("skill_suggestion.0")
                .and_then(|spec| match spec {
                    pi_jev::types::QuestionSpec::Choice { criteria, .. } => {
                        Some(criteria.keys().cloned().collect())
                    }
                    _ => None,
                })
                .unwrap_or_default();
            let response = {
                let raw = Self::suggestion_body(&excerpt, &criteria_keys);
                pi_jev::client::parse_systemone_body(raw.to_string().as_bytes())
                    .unwrap_or_else(|_| pi_jev::types::SystemOneResponse::default())
            };
            #[cfg(debug_assertions)]
            let hold = std::time::Duration::from_millis(
                HINT_FIXTURE_HOLD_MS.load(std::sync::atomic::Ordering::SeqCst),
            );
            return Box::pin(async move {
                #[cfg(debug_assertions)]
                if !hold.is_zero() {
                    tokio::time::sleep(hold).await;
                }
                Ok(response)
            });
        }
        // Shadow batteries: valid per-type fixture answers for the actual
        // request's question specs (never the suggestion bodies).
        let mut answers = std::collections::BTreeMap::new();
        for (id, spec) in &request.questions {
            answers.insert(id.clone(), pi_jev::mock::valid_answer_for(spec));
        }
        let response = pi_jev::types::SystemOneResponse {
            model: "jev-mock-hint/1".to_string(),
            answers,
            usage: pi_jev::types::Usage {
                input_tokens: Some(5),
                output_tokens: Some(5),
            },
            ..pi_jev::types::SystemOneResponse::default()
        };
        Box::pin(async move { Ok(response) })
    }
}

/// Test-only accessors for the `mock-hint` fixture transport of the LIVE
/// bridge core (debug builds): the state and question ids of the most recent
/// skill-suggestion decide that reached the transport. The hint-lane capture
/// test asserts the wire state equals the adapter's verified state and that
/// the request survived native bounding.
#[cfg(debug_assertions)]
pub fn debug_hint_fixture_last_suggestion() -> Option<(serde_json::Value, Vec<String>)> {
    HINT_FIXTURE_LAST_SUGGESTION
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
}

/// Test-only cancellation window for the `mock-hint` fixture transport
/// (debug builds). The timer-backed hold is explicitly capped, defaults to
/// zero, and changes no production cadence in release builds.
#[cfg(debug_assertions)]
static HINT_FIXTURE_ON_SUGGESTION:
    std::sync::Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>> =
    std::sync::Mutex::new(None);
#[cfg(debug_assertions)]
const HINT_FIXTURE_MAX_HOLD_MS: u64 = 5_000;
#[cfg(debug_assertions)]
static HINT_FIXTURE_HOLD_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(debug_assertions)]
static HINT_FIXTURE_SUGGESTION_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(debug_assertions)]
pub fn debug_hint_fixture_on_suggestion(
    callback: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
) {
    *HINT_FIXTURE_ON_SUGGESTION
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = callback;
}

/// Set a debug-only post-callback fixture hold. Values above five seconds are
/// capped; zero restores the unchanged default behavior.
#[cfg(debug_assertions)]
pub fn debug_hint_fixture_set_hold_ms(ms: u64) {
    HINT_FIXTURE_HOLD_MS.store(
        ms.min(HINT_FIXTURE_MAX_HOLD_MS),
        std::sync::atomic::Ordering::SeqCst,
    );
}

#[cfg(debug_assertions)]
pub fn debug_hint_fixture_suggestion_calls() -> u64 {
    HINT_FIXTURE_SUGGESTION_CALLS.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(debug_assertions)]
pub fn debug_hint_fixture_reset() {
    *HINT_FIXTURE_LAST_SUGGESTION
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = None;
    *HINT_FIXTURE_ON_SUGGESTION
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = None;
    HINT_FIXTURE_HOLD_MS.store(0, std::sync::atomic::Ordering::SeqCst);
    HINT_FIXTURE_SUGGESTION_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
}

/// Debug-build disclosure derived from the ACTUAL model-visible roster most
/// recently pushed by the session consumer seam. Uses the production
/// visibility, safe-id and truncation rules rather than a second loader.
#[cfg(debug_assertions)]
pub fn debug_hint_fixture_roster_disclosure(
    session_id: &str,
) -> Option<(usize, bool, usize)> {
    let core = bridge_for_session(session_id)?;
    let roster = core.cached_skill_roster();
    let (entries, truncated, skipped) =
        crate::core::jev_agent_guidance::skill_roster_view_with_skips(&roster);
    Some((entries.len(), truncated, skipped))
}

/// Bounded scheduler counters for native hint fixture settlement checks. No
/// request state, roster text, prompt content, or credential is exposed.
#[cfg(debug_assertions)]
pub fn debug_hint_fixture_observer_status(session_id: &str) -> Option<Value> {
    let core = bridge_for_session(session_id)?;
    let observer = core.observer(session_id, None)?;
    observer.session_status(session_id)
}

/// Test-only scratchpad for the `mock-control` fixture transport (debug
/// builds): an optional response callback plus one bounded capture entry per
/// call (question ids + the v9 requested-model stamp). No raw state content,
/// no credential, no ordering assumptions — the test reads the capture after
/// the turn settles.
#[cfg(debug_assertions)]
struct ControlFixtureScript {
    respond: Option<
        Arc<
            dyn Fn(
                    &pi_jev::types::SystemOneRequest,
                ) -> Option<pi_jev::types::SystemOneResponse>
                + Send
                + Sync,
        >,
    >,
    calls: Vec<serde_json::Value>,
}

#[cfg(debug_assertions)]
static CONTROL_FIXTURE: std::sync::Mutex<Option<ControlFixtureScript>> =
    std::sync::Mutex::new(None);

/// Fixture transport for the CONTROL/compaction held-identity regressions.
/// Answers from the test callback when installed, otherwise with valid
/// per-type fixture answers (shadow-battery discipline). Never fabricated
/// provenance: it is data-only, like every mock selector here.
#[derive(Default)]
struct ControlFixtureTransport;

impl pi_jev::types::Transport for ControlFixtureTransport {
    fn post(
        &self,
        request: &pi_jev::types::SystemOneRequest,
        _timeout: std::time::Duration,
    ) -> pi_jev::types::BoxFuture<
        Result<pi_jev::types::SystemOneResponse, pi_jev::error::JevError>,
    > {
        #[cfg(debug_assertions)]
        {
            let respond = {
                let mut guard = CONTROL_FIXTURE
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let script = guard.get_or_insert_with(|| ControlFixtureScript {
                    respond: None,
                    calls: Vec::new(),
                });
                // Bounded capture: question ids + requested-model identity.
                script.calls.push(serde_json::json!({
                    "question_ids": request
                        .questions
                        .keys()
                        .cloned()
                        .collect::<Vec<String>>(),
                    "model": request.model.clone(),
                    "fixture_lane": request.state
                        .get("_jev_fixture_lane")
                        .and_then(serde_json::Value::as_str),
                }));
                script.respond.clone()
            };
            // The callback can inspect capture and save authoritative settings,
            // so never invoke it while the fixture scratchpad lock is held.
            if let Some(response) = respond.and_then(|respond| respond(request)) {
                return Box::pin(async move { Ok(response) });
            }
        }
        let mut answers = std::collections::BTreeMap::new();
        for (id, spec) in &request.questions {
            answers.insert(id.clone(), pi_jev::mock::valid_answer_for(spec));
        }
        let response = pi_jev::types::SystemOneResponse {
            model: "jev-mock-control/1".to_string(),
            answers,
            usage: pi_jev::types::Usage {
                input_tokens: Some(5),
                output_tokens: Some(5),
            },
            ..pi_jev::types::SystemOneResponse::default()
        };
        Box::pin(async move { Ok(response) })
    }
}

/// Install the test's response callback for the `mock-control` transport
/// (debug builds). The callback may perform REAL authoritative settings saves
/// through independent store instances — this is the mid-await write window
/// the held-identity regressions exercise. Pass `None` to use the valid
/// per-type fallback answers only.
#[cfg(debug_assertions)]
pub fn debug_control_fixture_set_respond(
    respond: Option<
        Box<
            dyn Fn(
                    &pi_jev::types::SystemOneRequest,
                ) -> Option<pi_jev::types::SystemOneResponse>
                + Send
                + Sync,
        >,
    >,
) {
    let mut guard = CONTROL_FIXTURE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let script = guard.get_or_insert_with(|| ControlFixtureScript {
        respond: None,
        calls: Vec::new(),
    });
    script.respond = respond.map(Arc::from);
}

/// The bounded call capture so far (question ids + requested-model stamp).
#[cfg(debug_assertions)]
pub fn debug_control_fixture_calls() -> Vec<serde_json::Value> {
    CONTROL_FIXTURE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_ref()
        .map(|script| script.calls.clone())
        .unwrap_or_default()
}

/// Clear the callback and the capture between tests.
#[cfg(debug_assertions)]
pub fn debug_control_fixture_reset() {
    *CONTROL_FIXTURE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

/// Deterministic truncated payload for the mock-malformed test transport./// Deterministic truncated payload for the mock-malformed test transport./// Deterministic truncated payload for the mock-malformed test transport.
const TRUNCATED_MALFORMED_BODY: &str =
    r#"{"model":"jev-latest","answers":{"task_classification.0":{"#;

/// Loads the saved credential through the agent-dir store. The value is
/// wrapped and never logged, stored or copied elsewhere; `None` when absent
/// or the store is unavailable (fail closed).
fn load_saved_credential() -> Option<pi_jev::credential::SecretString> {
    let store = pi_jev::credential::default_credential_store(get_agent_dir());
    match store.get(pi_jev::config::DEFAULT_KEY_ID) {
        Ok(Some(key)) => Some(pi_jev::credential::SecretString::new(key)),
        _ => None,
    }
}

/// Transport used when no credential is configured: fails closed to a logged
/// skip, never a network call.
struct NoCredentialTransport;

impl pi_jev::types::Transport for NoCredentialTransport {
    fn post(
        &self,
        _request: &pi_jev::types::SystemOneRequest,
        _timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<pi_jev::types::SystemOneResponse, pi_jev::error::JevError>>
    {
        Box::pin(async {
            // The store is fine; there is simply no credential yet, so the
            // skip reason must say exactly that.
            Err(pi_jev::error::JevError::MissingCredential)
        })
    }
}

// ---------------------------------------------------------------------------
// Internal extension construction
// ---------------------------------------------------------------------------

/// Build the programmatic internal observer Extension. Handlers mirror the
/// runner's `ExtensionHandler` signature exactly.
fn build_observer_extension(core: Arc<JevBridgeCore>) -> SharedExtension {
    let mut handlers: HashMap<String, Vec<ExtensionHandler>> = HashMap::new();
    for event_type in JEV_EVENTS {
        let handler = make_handler(Arc::clone(&core), event_type);
        handlers
            .entry(event_type.to_string())
            .or_default()
            .push(handler);
    }
    // The Active handler ([`JEV_ACTIVE_EVENT`]) is NOT registered here.
    // [`sync_active_handlers`] installs it when some session is actually Active,
    // because its presence alone changes retry behavior for the whole process.
    // The caller stores the extension on the core via
    // [`JevBridgeCore::attach_extension`].
    Arc::new(Mutex::new(Extension {
        path: JEV_OBSERVER_PATH.to_string(),
        resolved_path: JEV_OBSERVER_PATH.to_string(),
        source_info: crate::core::source_info::create_synthetic_source_info(
            JEV_OBSERVER_PATH,
            &crate::core::source_info::SyntheticSourceInfoOptions {
                source: "internal".to_string(),
                ..Default::default()
            },
        ),
        handlers,
        tools: HashMap::new(),
        message_renderers: HashMap::new(),
        commands: indexmap::IndexMap::new(),
        flags: HashMap::new(),
        shortcuts: HashMap::new(),
    }))
}

/// Build one observe-only handler. Off cost: one cheap mode check, then
/// return `None` (the runner never sees a result from Jev).
fn make_handler(
    core: Arc<JevBridgeCore>,
    handler_event: &'static str,
) -> ExtensionHandler {
    Arc::new(
        move |event: ExtensionEvent,
              ctx: Arc<dyn crate::core::extensions::types::ExtensionContext>| {
            let core = Arc::clone(&core);
            Box::pin(async move {
                let session_id = ctx.session_manager().get_session_id();
                let settings = load_settings_cached();
                let mode = settings.effective_mode(&session_id);
                let _ = core.run_metrics.observe(&session_id, &event, mode, settings.effective_compaction_enabled(&session_id), settings.effective_features(&session_id));
                if handler_event == "session_shutdown" {
                    crate::core::jev_compaction::clear_session_status(&session_id);
                    core.drop_session_work(&session_id);
                    return None::<Value>;
                }
                if mode.is_enabled() || load_settings_cached().effective_compaction_enabled(&session_id) { core.own_session(&session_id); }
                if !mode.is_enabled() {
                    // Off takes effect immediately: cancel/forget this
                    // session's queued and in-flight comparison work so no
                    // late result can surface after the mode changed.
                    core.drop_decision_work(&session_id);
                    // ROOT-CONTRACT v7: an Off session never keeps an advisory
                    // skill hint cached — the next On activation starts clean.
                    core.store_skill_hint(&session_id, 0, None);
                    if handler_event == "session_start" {
                        ctx.ui().set_status("jev".into(), Some(footer_status_text(&session_id)));
                    }
                    return None::<Value>;
                }
                // Bounded per-session bookkeeping (never leaves the
                // process except as a capped excerpt inside a snapshot).
                //
                // This runs in Active too: the Active decision at the provider
                // boundary asks the same questions about the same state, so it
                // needs the same task excerpt and observed tool names.
                match &event {
                    ExtensionEvent::Input(payload) => {
                        core.remember_task_text(&session_id, &payload.text);
                    }
                    ExtensionEvent::ToolCall(tool_call) => {
                        core.note_tool_call(&session_id, tool_call.tool_name());
                    }
                    ExtensionEvent::TurnStart(payload) => {
                        core.note_turn(&session_id, payload.turn_index as u64);
                    }
                    ExtensionEvent::SessionShutdown(_) => {
                        core.forget_session(&session_id);
                    }
                    _ => {}
                }
                core.note_observation(&session_id, &event);
                // The shared actionable bundle replaces the duplicate shadow turn-start call.
                if mode.allows_active() && handler_event == "turn_start" { return None::<Value>; }
                if !mode.allows_compare() {
                    core.observe_optional(&session_id, &event, &ctx, handler_event).await;
                    return None::<Value>;
                }
                let Some(observer) = core.observer(&session_id, Some(ctx.ui())) else {
                    return None::<Value>;
                };
                if let Some((event_type, mut payload)) =
                    bridge_event(&core, &event, &ctx, &session_id, handler_event, &settings)
                {
                    observer.observe(&event_type, &payload);
                    // ROOT-CONTRACT v7 (Agent-guidance lane): the guardrail
                    // batteries ride the existing boundary dispatch as their
                    // OWN explicit requests (scheduler-cap aware; Compare and
                    // CompareAndActive observe; nothing ever applies).
                    let features = settings.effective_features(&session_id);
                    payload["policy_generation"] = json!(decision_policy_generation(&settings, &session_id));
                    if features.guardrails_input
                        && matches!(event, ExtensionEvent::TurnStart(_) | ExtensionEvent::TurnEnd(_))
                    {
                        if let Some(excerpt) = payload.get("state")
                            .and_then(|state| state.get("user_text_excerpt"))
                            .and_then(Value::as_str)
                        {
                            if let Some(prepared) = crate::core::jev_agent_guidance::prepare_guardrail_battery("guardrails_input", excerpt) {
                                observer.observe_prepared(&payload, "guardrails_input", prepared.questions);
                            }
                        }
                    }
                    if features.guardrails_output && matches!(event, ExtensionEvent::AgentEnd(_)) {
                        if let Some(excerpt) = payload.get("state")
                            .and_then(|state| state.get("result_excerpt"))
                            .and_then(Value::as_str)
                        {
                            if let Some(prepared) = crate::core::jev_agent_guidance::prepare_guardrail_battery("guardrails_output", excerpt) {
                                observer.observe_prepared(&payload, "guardrails_output", prepared.questions);
                            }
                        }
                    }
                    if features.skill_suggestion && matches!(event, ExtensionEvent::TurnStart(_)) {
                        core.observe_skill_suggestion_compare(&session_id, &payload, &observer, mode).await;
                    }
                    ctx.ui().set_status("jev".into(), Some(footer_status_text(&session_id)));
                }
                None::<Value>
            })
        },
    )
}

/// Build the one Active-mode handler.
///
/// Registered in addition to the observe-only handlers, and only reached when
/// the effective mode allows Active. In Off and Compare it
/// returns the untouched payload, so the provider path is unchanged.
fn make_active_handler(core: Arc<JevBridgeCore>) -> ExtensionHandler {
    Arc::new(
        move |event: ExtensionEvent,
              ctx: Arc<dyn crate::core::extensions::types::ExtensionContext>| {
            let core = Arc::clone(&core);
            Box::pin(async move {
                let ExtensionEvent::BeforeProviderRequest(payload) = event else {
                    return None::<Value>;
                };
                let session_id = ctx.session_manager().get_session_id();
                if !core.effective_mode(Some(&session_id)).allows_active() {
                    return None::<Value>;
                }
                // TOOL-001 diagnostics: remember the tool names THIS request
                // actually advertises (public schema names only), so a later
                // "no tools attached" claim can be compared with facts.
                // Observation only — it never changes the request.
                core.note_advertised_tools(
                    &session_id,
                    core.turn(&session_id),
                    &advertised_tool_names(&payload.payload),
                );
                let Some(observer) = core.observer(&session_id, Some(ctx.ui())) else {
                    return None::<Value>;
                };
                let mut params = payload.payload;
                // One decision boundary per provider request. The snapshot is
                // built from what the adapter already tracks plus the tool
                // catalog actually advertised in this request.
                let settings = load_settings_cached();
                let mut state = active_request_state(&core, &ctx, &session_id, &params, &settings);
                let features = settings.effective_features(&session_id);
                let tool_plan = features.tool_candidates.then(|| crate::core::jev_active::prepare_tool_pruning(
                    &params, core.task_excerpt(&session_id).as_deref().unwrap_or(""), &settings.filtering));
                if let Some(plan) = &tool_plan { state["state"]["optional_tools"] = plan.state["optional_tools"].clone(); }
                state["optional_min_confidence"] = json!(settings.filtering.min_confidence);
                state["optional_max_decision_age_ms"] = json!(settings.filtering.max_decision_age_ms);
                let policy = activation_policy(features);
                let call = observer.decide_active(&state, pi_jev::snapshot::SnapshotStage::TurnStart, &policy);
                let outcome = if let Some(signal) = ctx.signal() {
                    tokio::select! { biased;
                        _ = signal.cancelled() => { observer.cancel_decisions(&session_id); return None::<Value>; },
                        outcome = call => outcome,
                    }
                } else { call.await };
                let mut effects: BTreeMap<String, Vec<pi_jev::active::AppliedEffect>> =
                    BTreeMap::new();
                for decision in &outcome.decisions {
                    if !observer.can_apply(&outcome) || decision.turn != core.turn(&session_id)
                        || !decision.is_fresh(std::time::SystemTime::now(), &policy) { break; }
                    let before = request_action(&params);
                    let changes = crate::core::jev_active::apply_model_decision(
                        &mut params,
                        decision.category.as_str(),
                        &decision.value,
                        ctx.model().as_ref(),
                    );
                    if changes.is_empty() {
                        continue;
                    }
                    effects.insert(
                        decision.category.as_str().to_string(),
                        changes
                            .iter()
                            .map(|change| {
                                pi_jev::active::AppliedEffect::new(
                                    change.key.clone(),
                                    before.get(&change.key).cloned(),
                                    request_action(&params).get(&change.key).cloned(),
                                )
                            })
                            .collect(),
                    );
                }
                if let Some(plan) = &tool_plan {
                    if observer.can_apply(&outcome) {
                        let changes = plan.apply(&mut params, &outcome.decisions,
                            outcome.request_id.as_deref().unwrap_or(""), core.turn(&session_id));
                        let mut dropped: Vec<_> = outcome.decisions.iter().filter(|decision|
                            decision.category == pi_jev::types::DecisionCategory::ToolCandidates && decision.value == "drop").collect();
                        dropped.sort_by_key(|decision| decision.question_id.rsplit_once('.').and_then(|(_, suffix)| suffix.parse::<usize>().ok()));
                        for (decision, change) in dropped.into_iter().zip(changes) {
                            effects.insert(decision.question_id.clone(), vec![pi_jev::active::AppliedEffect::new(
                                "optional_tool", change.from, Some("omitted_for_request".to_string()))]);
                        }
                    }
                }
                if !observer.can_apply(&outcome) { effects.clear(); }
                observer.record_active_with_action(&outcome, &effects, &request_action(&params));
                // ROOT-CONTRACT v7 (Agent-guidance lane): the skill suggestion
                // is its OWN awaited request at the pre-context
                // `before_request` seam (agent_session hook), NOT here: this
                // late boundary cannot change the request context already
                // built for this request, and one assessment per provider
                // request lives at that seam only.
                ctx.ui()
                    .set_status("jev".into(), Some(footer_status_text(&session_id)));
                if effects.is_empty() {
                    // Nothing applied: hand the payload back untouched rather
                    // than claim a change the request never had.
                    None::<Value>
                } else {
                    Some(params)
                }
            })
        },
    )
}

fn decision_policy_generation(settings: &JevSettings, session_id: &str) -> String {
    // Bound to the SAME supplied authoritative snapshot (control-lane ABA
    // finding): the durable write revision moves on every authoritative save,
    // so an Off->On cycle or a model A->B->A reset changes this generation
    // even when the serialized feature/filtering values return to identical
    // bytes, and can_apply's fresh recheck refuses to apply any decision that
    // crossed an Off window. The overlay stamp is included so overlay
    // installs/removals are visible. No separate process counter and no fresh
    // global getter is mixed into this snapshot; callers that need a fresh
    // apply check load a fresh snapshot and call this with it.
    json!([
        settings.write_revision,
        settings.full_jev_stamp(),
        settings.effective_features(session_id),
        settings.filtering,
        settings.requested_model_or_default()
    ]).to_string()
}

pub fn compaction_policy_generation(settings: &JevSettings, session_id: &str) -> String {
    // Same-snapshot identity (control-lane ABA finding): the durable write
    // revision + overlay stamp ride the independent-compaction generation so
    // an Off->On or A->B->A authoritative write invalidates held independent
    // work even when the serialized values return to identical bytes.
    json!([
        settings.write_revision,
        settings.full_jev_stamp(),
        settings.effective_compaction_enabled(session_id),
        settings.compaction,
        settings.requested_model_or_default()
    ]).to_string()
}

/// ROOT-CONTRACT v7 (Agent-guidance lane): the host pushes the session's
/// already-loaded model-visible skill roster before the prompt build. Catalog
/// metadata only; the suggestion pool equals the prompt roster.
pub fn note_session_skill_roster(session_id: &str, skills: &[Skill]) {
    let Some(core) = bridge_for_session(session_id) else { return; };
    core.set_skill_roster(skills);
}

/// The latest advisory skill hint for the session, if any. Assessment only.
/// Served only while the guidance feature is enabled for the session and the
/// hint belongs to the CURRENT turn; a disabled feature clears the cache so
/// an Off->On activation never resurrects an ungoverned hint.
pub fn current_skill_hint(session_id: &str) -> Option<pi_jev::agent_guidance::SkillHint> {
    if !load_settings_cached().effective_features(session_id).skill_suggestion {
        if let Some(core) = bridge_for_session(session_id) {
            core.store_skill_hint(session_id, 0, None);
        }
        return None;
    }
    let core = bridge_for_session(session_id)?;
    core.skill_hint_for(session_id)
}

/// The host notes the CURRENT delivery id (prepared turn action) at the
/// commit seam (root blocker-3): the hint stamp binds to the genuine
/// delivery, so two new deliveries with byte-identical task text can never
/// reuse the previous request's hint.
/// Called only after a real user task epoch is committed by the host.
/// Automatic follow-ups keep the same observation scope and verification state.
pub fn note_control_task_epoch(session_id: &str, epoch_id: &str) {
    if let Some(core) = bridge_for_session(session_id) {
        core.note_control_epoch(session_id, epoch_id);
    }
}

pub fn note_session_delivery(session_id: &str, delivery_id: &str) {
    let Some(core) = bridge_for_session(session_id) else { return; };
    core.note_delivery(session_id, delivery_id);
}

/// The host session registers a prompt-refresh listener for the session
/// (root blocker-1): the bridge invokes it whenever the stored hint state
/// changes, so the session re-derives the request prompt and the next
/// provider request consumes it.
pub fn set_skill_hint_listener(
    session_id: &str,
    listener: Arc<dyn Fn() + Send + Sync>,
) {
    let Some(core) = bridge_for_session(session_id) else { return; };
    core.set_hint_listener(session_id, listener);
}

/// The CURRENT authoritative hint-stamp facts for the session (root wiring
/// contract): settings revision, mode, feature flag, turn, bounded task
/// excerpt and roster-identity hash. Production and consumption derive this
/// the same way, so a stored hint renders only while every fact still
/// matches; any drift fails open to no hint.
pub fn current_skill_hint_stamp(
    session_id: &str,
    skills: &[crate::core::skills::Skill],
) -> pi_jev::agent_guidance::SkillHintStamp {
    let settings = load_settings_cached();
    let features = settings.effective_features(session_id);
    let mode = settings.effective_mode(session_id);
    let revision = settings_revision();
    // Root blocker-3: the catalog identity derives from the ACTUAL loaded
    // skills the caller passes (the same list rendering the roster block),
    // never from a possibly-stale bridge cache.
    let catalog = crate::core::jev_agent_guidance::skill_roster_view_with_skips(skills).0;
    let Some(core) = bridge_for_session(session_id) else {
        return crate::core::jev_agent_guidance::skill_hint_stamp(
            &revision,
            mode.as_str(),
            features.skill_suggestion,
            0,
            "",
            "",
            &catalog,
        );
    };
    let task_identity = core.task_identity(session_id).unwrap_or_default();
    crate::core::jev_agent_guidance::skill_hint_stamp(
        &revision,
        mode.as_str(),
        features.skill_suggestion,
        core.turn(session_id),
        &core.delivery_id(session_id),
        &task_identity,
        &catalog,
    )
}

/// v5 usage knownness for compaction stats: Some(n) — including a measured
/// zero — is emitted; None (UNKNOWN) is omitted. Never fabricates a zero.
fn record_usage_stats(usage: &pi_jev::types::Usage, stats: &mut Value) {
    if let Some(tokens) = usage.input_tokens { stats["jev_input_tokens"] = json!(tokens); }
    if let Some(tokens) = usage.output_tokens { stats["jev_output_tokens"] = json!(tokens); }
}

pub fn record_compaction_skip(ctx: Arc<dyn ExtensionContext>, reason: &str, stats: Value) {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    if !settings.effective_compaction_enabled(&session_id) { return; }
    let capture = pi_jev::scheduler::RequestContext {
        request_id:format!("jev-{}",uuid::Uuid::new_v4()), session_id:session_id.clone(),
        turn:bridge_for_session(&session_id).map(|core| core.turn(&session_id)).unwrap_or(0),
        stage:"compaction".to_string(),state_fingerprint:String::new(),
        state_schema_version:pi_jev::snapshot::STATE_SCHEMA_VERSION.to_string(),
        prompt_version:pi_jev::hooks::PROMPT_VERSION.to_string(), mode:settings.effective_mode(&session_id).as_str().to_string(),
        requested_model:settings.requested_model_or_default().to_string(),
        policy_generation:compaction_policy_generation(&settings, &session_id),
        questions:Vec::new(),baselines:BTreeMap::new(),request_start_ts:pi_jev::client::utc_now_rfc3339(),
    };
    let records=std::path::PathBuf::from(get_agent_dir()).join("jev").join("records.jsonl");
    pi_jev::correlate::Correlator::new(records,0.7).record_compaction(&capture,None,0,true,None,&stats,Some(reason));
}

pub struct CompactionDecision {
    pub outcome: pi_jev::types::DecisionOutcome,
    observer: Arc<JevObserver>,
    boundary: pi_jev::hooks::ActiveDecideOutcome,
    signal: Option<tokio_util::sync::CancellationToken>,
}

impl CompactionDecision {
    pub fn can_apply(&self) -> bool {
        self.signal.as_ref().is_none_or(|signal| !signal.is_cancelled()) && self.observer.can_apply(&self.boundary)
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.boundary.terminal_reason.as_deref().or_else(|| self.outcome.skips.first().map(|(_, reason)| *reason))
    }

    pub fn record_compaction(&self, mut stats: Value, fallback: Option<&str>) {
        let Some(ctx) = &self.boundary.context else { return; };
        let reason = if self.can_apply() { self.fallback_reason().or(fallback) } else { Some("cancelled_or_policy_changed") };
        // v5 usage knownness: Some(n) (including a measured zero) is emitted, None (UNKNOWN) is omitted.
        record_usage_stats(&self.outcome.usage, &mut stats);
        let attempts = self.boundary.raw.as_ref().map(|raw|raw.attempts).unwrap_or(u32::from(self.boundary.dispatched));
        let known = self.boundary.raw.is_some() || !self.boundary.dispatched;
        self.observer.correlator().record_compaction(ctx, self.outcome.response_model.as_deref(), attempts, known,
            self.boundary.duration_ms, &stats, reason);
    }
}

pub async fn decide_compaction(ctx: Arc<dyn ExtensionContext>, mut bundle: pi_jev::types::DecisionBundle,
    signal: Option<tokio_util::sync::CancellationToken>) -> Option<CompactionDecision> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    if !settings.effective_compaction_enabled(&session_id) || signal.as_ref().is_some_and(|signal| signal.is_cancelled()) { return None; }
    let core = bridge_for_session(&session_id)?;
    let observer = core.observer(&session_id, Some(ctx.ui()))?;
    let captured_generation = bundle.state.as_object_mut().and_then(|state| state.remove("_jev_policy_generation"))
        .and_then(|value| value.as_str().map(str::to_string))?;
    let payload = json!({"session_id":session_id, "turn":core.turn(&session_id), "state":bundle.state,
        "compaction_enabled":true, "policy_generation":captured_generation});
    let questions = bundle.questions.into_iter().map(|(question_id, spec)| pi_jev::evaluators::PreparedQuestion { question_id, spec }).collect();
    let call = observer.decide_independent(&payload, "compaction", questions);
    let boundary = if let Some(signal) = &signal {
        tokio::select! { biased; _ = signal.cancelled() => return None, result = call => result }
    } else { call.await };
    let outcome = boundary.raw.clone().unwrap_or_else(|| pi_jev::types::DecisionOutcome::skipped_all("unavailable"));
    Some(CompactionDecision { outcome, observer, boundary, signal })
}

async fn relevance_decision(ctx: &Arc<dyn ExtensionContext>, prepared: &crate::core::jev_retrieval::PreparedRelevance, settings: &JevSettings)
    -> Option<(Arc<JevObserver>, pi_jev::hooks::ActiveDecideOutcome, Vec<usize>)> {
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let core = bridge_for_session(&session_id)?;
    let observer = core.observer(&session_id, Some(ctx.ui()))?;
    let payload = json!({"session_id":session_id, "turn":core.turn(&session_id), "state":prepared.state,
        "baseline_action":prepared.action_metadata(&[]),
        "compaction_enabled":settings.effective_compaction_enabled(&session_id),
        "policy_generation":decision_policy_generation(settings, &session_id)});
    let questions = prepared.questions();
    if questions.is_empty() { return None; }
    if !mode.allows_active() {
        if mode.allows_compare() { observer.observe_prepared(&payload, "retrieval", questions); }
        return None;
    }
    let mut policy = activation_policy(settings.effective_features(&session_id));
    policy.min_confidence = settings.filtering.min_confidence;
    policy.max_decision_age = Duration::from_millis(settings.filtering.max_decision_age_ms);
    let call = observer.decide_prepared(&payload, "retrieval", questions, &policy);
    let outcome = if let Some(signal) = ctx.signal() {
        tokio::select! { biased;
            _ = signal.cancelled() => { observer.cancel_decisions(&session_id); return None; },
            outcome = call => outcome,
        }
    } else { call.await };
    let removals = if observer.can_apply(&outcome) && core.turn(&session_id) == outcome.turn {
        prepared.removals(&outcome.decisions, outcome.request_id.as_deref().unwrap_or(""), outcome.turn)
    } else { Vec::new() };
    Some((observer, outcome, removals))
}

fn relevance_effects(prepared: &crate::core::jev_retrieval::PreparedRelevance, removals: &[usize]) -> BTreeMap<String, Vec<pi_jev::active::AppliedEffect>> {
    prepared.candidate_indices.iter().enumerate().filter(|(_, index)| removals.contains(index)).map(|(ordinal, _)| {
        let id = format!("{}.{}", prepared.category.as_str(), ordinal);
        (id.clone(), vec![pi_jev::active::AppliedEffect::new(id, Some("included".to_string()), Some("omitted_for_request".to_string()))])
    }).collect()
}

pub async fn filter_memory_candidates(ctx: Arc<dyn ExtensionContext>, query: &str, hits: Vec<MemoryHit>) -> Vec<MemoryHit> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    if !settings.effective_mode(&session_id).is_enabled() || !settings.effective_features(&session_id).memory_relevance { return hits; }
    let mut prepared = crate::core::jev_retrieval::prepare_memory(&hits, query);
    prepared.configure(&settings.filtering);
    let Some((observer, outcome, mut removals)) = relevance_decision(&ctx, &prepared, &settings).await else { return hits; };
    if !observer.can_apply(&outcome) { removals.clear(); }
    let effects = relevance_effects(&prepared, &removals);
    observer.record_active_with_action(&outcome, &effects, &prepared.action_metadata(&removals));
    crate::core::jev_retrieval::apply_memory(hits, &removals)
}

pub async fn filter_context_candidates(ctx: Arc<dyn ExtensionContext>, messages: Vec<Value>, budget: &pi_jev::search::SearchBudget) -> Vec<Value> {
    let session_id = ctx.session_manager().get_session_id();
    let settings = load_settings_cached();
    // Capture citation provenance from the strict, unannotated input before
    // any request-local Jev stage can add a prefix-shaped advisory block.
    let citation_basis = prepare_citation_basis(&ctx, &messages, &settings);
    // ROOT CONTRACT v1 (Search): ONE shared provider-context deadline across
    // filtering, reranking and line matching. No independent stage stacks.
    let messages = filter_code_search_candidates(&ctx, messages, &settings, budget).await;
    let messages = annotate_line_find_candidates(&ctx, messages, &settings, budget).await;
    // ROOT-CONTRACT v6 (Evidence lane): bounded citation assessment over the
    // ACTUAL ipython source-read path. At most ONE citation judgment per
    // provider request, funded from the ONE shared provider-context deadline.
    // Advisory only: never verification, never a gate, never a drop.
    let messages = annotate_citation_check(&ctx, messages, &settings, budget, citation_basis).await;
    if !settings.effective_mode(&session_id).is_enabled() || !settings.effective_features(&session_id).context_relevance { return messages; }
    let mut prepared = crate::core::jev_retrieval::prepare_context(&messages);
    prepared.configure(&settings.filtering);
    let Some((observer, outcome, mut removals)) = relevance_decision(&ctx, &prepared, &settings).await else { return messages; };
    if !observer.can_apply(&outcome) { removals.clear(); }
    let effects = relevance_effects(&prepared, &removals);
    observer.record_active_with_action(&outcome, &effects, &prepared.action_metadata(&removals));
    crate::core::jev_retrieval::apply_context(messages, &removals)
}

async fn filter_code_search_candidates(ctx: &Arc<dyn ExtensionContext>, mut messages: Vec<Value>, settings: &JevSettings, budget: &pi_jev::search::SearchBudget) -> Vec<Value> {
    use futures::StreamExt;
    use crate::core::jev_code_search;
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let features = settings.effective_features(&session_id);
    if !mode.is_enabled() || !features.code_search_relevance { return messages; }
    if ctx.signal().is_some_and(|signal| signal.is_cancelled()) { return messages; }
    let Some(core) = bridge_for_session(&session_id) else { return messages; };
    let Some(observer) = core.observer(&session_id, Some(ctx.ui())) else { return messages; };
    let query = crate::core::jev_retrieval::query_from_messages(&messages);
    let query = if query.is_empty() { core.task_excerpt(&session_id).unwrap_or_default() } else { query };
    let presentations = jev_code_search::prepare(&messages, &query, &settings.filtering);
    let generation = decision_policy_generation(settings, &session_id);
    let stamp = cheap_credential_stamp(settings);
    for presentation in presentations {
        let cache_key = pi_jev::snapshot::fingerprint_of(&json!([session_id, core.turn(&session_id), mode.as_str(), generation, stamp, presentation.fingerprint]));
        if !mode.allows_active() && jev_code_search::already_observed(&cache_key) {
            continue;
        }
        let payloads: Vec<_> = presentation.batches.iter().map(|batch| json!({
            "session_id":session_id, "turn":core.turn(&session_id), "state":batch.state,
            "baseline_action":batch.action_metadata(&[]), "policy_generation":generation,
            "compaction_enabled":settings.effective_compaction_enabled(&session_id)
        })).collect();
        if !mode.allows_active() {
            for (batch, payload) in presentation.batches.iter().zip(&payloads) {
                observer.observe_prepared(payload, "code_search", batch.questions());
            }
            // ROOT-CONTRACT v6 (Evidence lane): Compare observes the safety
            // battery too — as its own explicit request under the scheduler
            // cap; nothing applies.
            if features.retrieval_safety {
                let battery_is_cancelled = || ctx.signal().is_some_and(|signal| signal.is_cancelled());
                let battery_inputs = crate::core::jev_evidence::BatteryInputs {
                    observer: Arc::clone(&observer),
                    session_id: session_id.as_str(),
                    turn: core.turn(&session_id),
                    mode,
                    policy_generation: generation.as_str(),
                    max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
                    budget,
                    is_cancelled: &battery_is_cancelled,
                };
                crate::core::jev_evidence::observe_safety_battery(&battery_inputs, &presentation);
            }
            if features.code_search_reranking {
                // Compare observes the rerank questions too; nothing applies.
                let inputs = presentation.rerank_inputs(pi_jev::search::MAX_RERANK_CANDIDATES, &[]);
                for chunk in inputs.chunks(pi_jev::search::RERANK_BATCH) {
                    let excerpts: Vec<String> = chunk.iter().map(|(_, excerpt)| excerpt.clone()).collect();
                    let state = pi_jev::search::rerank_batch_state(&query, &excerpts);
                    if let Some(questions) = pi_jev::search::rerank_batch_questions(&state) {
                        if !questions.is_empty() {
                            observer.observe_prepared(&json!({
                                "session_id": session_id, "turn": core.turn(&session_id),
                                "state": state, "policy_generation": generation
                            }), "code_search_rerank", questions);
                        }
                    }
                }
            }
            jev_code_search::remember_observation(cache_key);
            continue;
        }
        let policy = pi_jev::active::ActivationPolicy {
            enabled_categories: if features.code_search_filtering { [pi_jev::types::DecisionCategory::CodeSearchRelevance].into_iter().collect() } else { Default::default() },
            min_confidence: settings.filtering.min_confidence,
            max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
        };
        let signal = ctx.signal();
        let calls: Vec<_> = payloads.into_iter().enumerate().map(|(index, payload)| {
            let observer = observer.clone();
            let policy = policy.clone();
            let signal = signal.clone();
            let questions = presentation.batches[index].questions();
            async move {
                let remaining = budget.remaining();
                let mut payload = payload;
                payload["decision_timeout_ms"] = json!(if signal.as_ref().is_some_and(|signal| signal.is_cancelled()) {
                    0
                } else { remaining.as_millis() as u64 });
                let call = observer.decide_prepared(&payload, "code_search", questions, &policy);
                tokio::pin!(call);
                let outcome = if let Some(signal) = signal {
                    tokio::select! { biased;
                        // Register the request token before cancelling it, including
                        // when cancellation races the first poll of this future.
                        result = &mut call => result,
                        _ = signal.cancelled() => {
                            observer.cancel_decisions(payload["session_id"].as_str().unwrap_or_default());
                            call.await
                        }
                    }
                } else { call.await };
                (index, outcome)
            }
        }).collect();
        let outcomes: Vec<_> = futures::stream::iter(calls).buffer_unordered(2).collect().await;
        let complete = outcomes.iter().all(|(_, outcome)|
            observer.can_apply(outcome) && outcome.turn == core.turn(&session_id) && outcome.unavailable.is_none()
                && outcome.raw.as_ref().is_some_and(|raw| raw.skips.is_empty()))
            && signal.as_ref().is_none_or(|signal| !signal.is_cancelled());
        let removals: Vec<_> = if complete && features.code_search_filtering {
            outcomes.iter().flat_map(|(index, outcome)| {
                let batch = &presentation.batches[*index];
                batch.removals(&outcome.decisions, outcome.request_id.as_deref().unwrap_or(""), outcome.turn)
            }).collect()
        } else { Vec::new() };
        // ROOT-CONTRACT v6 (Evidence lane): the safety battery is its OWN
        // typed request under the scheduler's per-request question cap — it
        // never rides the filter request (a combined request would be
        // silently refused and kill the filter under the flag). Planned
        // drops are covered first (the veto targets), then retained
        // candidates up to the explicit battery cap; the subset and its
        // coverage are disclosed in the annotation and metadata. Fail-open:
        // any refused, skipped, stale or cancelled battery leaves the
        // removals untouched.
        let battery_is_cancelled = || ctx.signal().is_some_and(|signal| signal.is_cancelled());
        let battery_inputs = crate::core::jev_evidence::BatteryInputs {
            observer: Arc::clone(&observer),
            session_id: session_id.as_str(),
            turn: core.turn(&session_id),
            mode,
            policy_generation: generation.as_str(),
            max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
            budget,
            is_cancelled: &battery_is_cancelled,
        };
        // ROOT-CONTRACT v6 (Evidence lane): the correlated battery outcome is
        // RETURNED so the effect site can fold its freshness into
        // still_current and record the attach state. Refused rounds carry
        // None and stay fail-open; removals are never increased.
        let (removals, safety_annotation, safety_metadata, battery_outcome) = if features.retrieval_safety {
            crate::core::jev_evidence::run_safety_veto(&battery_inputs, &presentation, removals).await
        } else { (removals, None, BTreeMap::new(), None) };
        // Rerank stage (ROOT CONTRACT v1, Search): absolute per-pair Noul
        // scores over the RETAINED scored candidates; every failure keeps the
        // original surviving order. The stage name is in the observation
        // cache key so rerank never shadows the filter.
        let mut order: Option<Vec<usize>> = None;
        let mut rerank_model: Option<String> = None;
        let mut rerank_records: Vec<(std::collections::BTreeMap<String, String>, pi_jev::hooks::ActiveDecideOutcome)> = Vec::new();
        if complete && features.code_search_reranking && !budget.expired() {
            let inputs = presentation.rerank_inputs(pi_jev::search::MAX_RERANK_CANDIDATES, &removals);
            let mut scores: Vec<pi_jev::search::ScoredCandidate> = Vec::new();
            let mut rerank_complete = true;
            for (batch_index, chunk) in inputs.chunks(pi_jev::search::RERANK_BATCH).enumerate() {
                if budget.expired() || ctx.signal().is_some_and(|signal| signal.is_cancelled()) {
                    rerank_complete = false;
                    break;
                }
                let excerpts: Vec<String> = chunk.iter().map(|(_, excerpt)| excerpt.clone()).collect();
                let state = pi_jev::search::rerank_batch_state(&query, &excerpts);
                let questions = pi_jev::search::rerank_batch_questions(&state).unwrap_or_default();
                if questions.is_empty() { rerank_complete = false; break; }
                let candidate_indices: Vec<usize> = chunk.iter().map(|(ordinal, _)| *ordinal).collect();
                let mut payload = json!({
                    "session_id": session_id, "turn": core.turn(&session_id), "state": state,
                    "policy_generation": generation
                });
                payload["decision_timeout_ms"] = json!(budget.remaining_ms());
                let rerank_policy = pi_jev::active::ActivationPolicy {
                    enabled_categories: [pi_jev::types::DecisionCategory::CodeSearchRerank].into_iter().collect(),
                    // Typed acceptance; a Noul is never confidence-gated.
                    min_confidence: 0.0,
                    max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
                };
                let outcome = observer.decide_prepared(&payload, "code_search_rerank", questions, &rerank_policy).await;
                let usable = observer.can_apply(&outcome)
                    && outcome.unavailable.is_none()
                    && outcome.raw.as_ref().is_some_and(|raw| raw.skips.is_empty());
                if !usable { rerank_complete = false; break; }
                scores.extend(pi_jev::search::scored_candidates_from_decisions(
                    &outcome.decisions, &candidate_indices, pi_jev::types::DecisionCategory::CodeSearchRerank,
                ));
                if rerank_model.is_none() { rerank_model = outcome.response_model.clone(); }
                rerank_records.push((std::collections::BTreeMap::from([
                    ("rerank_batch".to_string(), batch_index.to_string()),
                    ("rerank_scored".to_string(), chunk.len().to_string()),
                ]), outcome));
            }
            if rerank_complete && !inputs.is_empty() && scores.len() == inputs.len() {
                order = pi_jev::search::reranked_order(
                    &scores, inputs.len(), core.turn(&session_id),
                    std::time::SystemTime::now(),
                    Duration::from_millis(settings.filtering.max_decision_age_ms),
                );
            }
        }
        // Truthful scope: when the shared candidate cap means only a prefix
        // of the retained scored set was scored, the label must say "prefix".
        let scope = if order.as_ref().is_some_and(|order| {
            order.len() < presentation.rerank_inputs(usize::MAX, &removals).len()
        }) {
            "prefix"
        } else {
            "all"
        };
        let projection = presentation.project_with_order(&removals, order.as_deref(), rerank_model.as_deref(), order.as_ref().map(|_| scope));
        let applied = if projection.is_some() { removals.clone() } else { Vec::new() };
        // ROOT CONTRACT v1: before ANY effect, re-check the current gates. A
        // settings/feature toggle off->on (policy generation), a mode flip, or
        // cancellation between decision and application drops the effect; the
        // original request copy stays untouched (fail open). ROOT-CONTRACT v6
        // (Evidence lane): the correlated battery outcome folds into the same
        // gate so a stale battery decision can never un-drop at projection
        // time; a refused battery (None) imposes no freshness constraint.
        let still_current = complete
            && outcomes.iter().all(|(_, outcome)| observer.can_apply(outcome))
            && rerank_records.iter().all(|(_, outcome)| observer.can_apply(outcome))
            && battery_outcome.as_ref().map(|outcome| observer.can_apply(outcome)).unwrap_or(true)
            && !ctx.signal().is_some_and(|signal| signal.is_cancelled());
        // ROOT-CONTRACT v6 (Evidence lane): additive advisory safety
        // annotation on the request copy only. Same one-annotation discipline
        // as line-find (single-block toolResult guard); originals are
        // preserved and nothing is dropped or rewritten. The attach result is
        // recorded, never assumed.
        let mut safety_attached = false;
        if still_current {
            if let Some(content) = projection { messages[presentation.message_index]["content"] = content; }
            if features.retrieval_safety {
                if let Some(block) = safety_annotation.clone() {
                    safety_attached = crate::core::jev_evidence::attach_safety_block(
                        &mut messages, presentation.message_index, block);
                }
            }
        }
        // ROOT-CONTRACT v6 (Evidence lane): the battery's own correlated
        // record carries the attach state distinctly (attached /
        // skipped_multi_block / withheld_stale_gates) — assessment and
        // annotation are never conflated and no attached claim is made when
        // the effect gates went stale.
        if let Some(outcome) = &battery_outcome {
            let mut battery_record = safety_metadata.clone();
            if safety_annotation.is_some() {
                battery_record.insert(
                    "jev_safety_annotation".to_string(),
                    if !still_current {
                        "withheld_stale_gates".to_string()
                    } else if safety_attached {
                        "attached".to_string()
                    } else {
                        "skipped_multi_block".to_string()
                    },
                );
            }
            observer.record_active_with_action(outcome, &std::collections::BTreeMap::new(), &battery_record);
        }
        for (index, outcome) in &outcomes {
            let batch = &presentation.batches[*index];
            let mut filter_metadata = batch.action_metadata(&applied);
            if battery_outcome.is_none() {
                // A refused battery round is disclosed on the filter record
                // that funded it; flag-off rounds merge an empty map (no-op).
                filter_metadata.extend(safety_metadata.clone());
            }
            observer.record_active_with_action(outcome, &relevance_effects(batch, &applied), &filter_metadata);
        }
        let rerank_metadata = presentation.rerank_action_metadata(&removals, order.as_deref(), scope);
        for (batch_metadata, outcome) in &rerank_records {
            let mut merged = batch_metadata.clone();
            merged.extend(rerank_metadata.clone());
            observer.record_active_with_action(outcome, &std::collections::BTreeMap::new(), &merged);
        }
    }
    messages
}

// ROOT CONTRACT v1 (Search): line-level semantic find over explicitly supplied
// text. At most ONE judgment per provider request, funded from the ONE shared
// provider-context deadline. Fail-open everywhere: nothing is removed or
// rewritten; the annotation is additive to the request copy only, and session
// history keeps the original. In Compare mode the questions are observed but
// nothing is applied; for windowed texts Compare observes the window question
// (the second pass depends on the unasked window answer, so it is not observed).

async fn annotate_line_find_candidates(
    ctx: &Arc<dyn ExtensionContext>,
    mut messages: Vec<Value>,
    settings: &JevSettings,
    budget: &pi_jev::search::SearchBudget,
) -> Vec<Value> {
    use crate::core::{jev_code_search, jev_line_find};
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let features = settings.effective_features(&session_id);
    if !mode.is_enabled() || !features.line_find { return messages; }
    if ctx.signal().is_some_and(|signal| signal.is_cancelled()) { return messages; }
    let Some(core) = bridge_for_session(&session_id) else { return messages; };
    let Some(observer) = core.observer(&session_id, Some(ctx.ui())) else { return messages; };
    let query = crate::core::jev_retrieval::query_from_messages(&messages);
    let query = if query.is_empty() { core.task_excerpt(&session_id).unwrap_or_default() } else { query };
    let options = pi_jev::search::LineFindOptions::default();
    if options.validate().is_err() { return messages; }
    // Primary reachability: the deterministic ipython source-read idiom
    // (this runtime has no read/read_file tools; source is read via the REPL).
    let mut presentations = jev_line_find::prepare(&messages, &query, &options);
    // Secondary: the top file candidate of the CURRENT (filtered+reranked)
    // code-search envelope - an explicit presentation snippet.
    if presentations.is_empty() && features.code_search_relevance {
        // The CURRENT (filtered+reranked) envelope: the projection adds
        // disclosure keys, so the strict two-key `prepare` cannot see it. The
        // top candidate is the first NON-pinned `file` candidate; pinned
        // anchors are never line-matched (jev_line_find re-checks too).
        if let Some((index, envelope)) = jev_code_search::recent_code_search_envelope(&messages) {
            let candidate = envelope["candidates"].as_array().and_then(|items| {
                items
                    .iter()
                    .find(|candidate| candidate["kind"] == "file" && !jev_code_search::pinned(candidate))
                    .cloned()
            });
            if let Some(candidate) = candidate {
                if let Some(presentation) = jev_line_find::prepare_from_snippet(&candidate, index, &query, &options) {
                    presentations.push(presentation);
                }
            }
        }
    }
    // ONE line-find judgment per provider request (ROOT CONTRACT bound).
    let Some(presentation) = presentations.first() else { return messages; };
    let generation = decision_policy_generation(settings, &session_id);
    let stamp = cheap_credential_stamp(settings);
    let cache_key = pi_jev::snapshot::fingerprint_of(&json!([
        session_id, core.turn(&session_id), mode.as_str(), generation, stamp,
        "code_line_find", presentation.fingerprint
    ]));
    if !mode.allows_active() && jev_code_search::already_observed(&cache_key) { return messages; }
    let policy = pi_jev::active::ActivationPolicy {
        enabled_categories: [pi_jev::types::DecisionCategory::CodeLineFind].into_iter().collect(),
        // Typed acceptance; the existence Noul is never confidence-gated.
        min_confidence: 0.0,
        max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
    };
    let payload_for = |state: &Value| {
        let mut payload = json!({
            "session_id": session_id, "turn": core.turn(&session_id),
            "policy_generation": generation
        });
        payload["state"] = state.clone();
        payload["decision_timeout_ms"] = json!(budget.remaining_ms());
        payload
    };
    // Cascade pass 1 (windows) when the supplied text exceeds one Choice request.
    let window = if presentation.needs_window_pass() {
        let Some((state, questions)) = presentation.pass1() else { return messages; };
        if !mode.allows_active() {
            observer.observe_prepared(&payload_for(&state), "code_line_find", questions);
            jev_code_search::remember_observation(cache_key);
            return messages;
        }
        if budget.expired() { return messages; }
        let first = observer.decide_prepared(&payload_for(&state), "code_line_find", questions, &policy).await;
        // ROOT CONTRACT v1 (Search): the pass-1 window Choice is consumed at
        // set level from the RAW record (the typed pair acceptance covers only
        // the pass-2 pair), and only a complete finite normalized distribution
        // over exactly the supplied windows may narrow the cascade.
        let window_distribution = first.raw.as_ref()
            .and_then(|raw| raw.records.iter().find(|record| record.question_id == pi_jev::search::WINDOW_QUESTION_ID))
            .and_then(|record| match &record.answer {
                pi_jev::Answer::Choice { probabilities, .. } => Some(probabilities.clone()),
                _ => None,
            })
            .and_then(|probabilities| pi_jev::search::validate_window_distribution(&probabilities, presentation.text.windows()));
        let usable = !ctx.signal().is_some_and(|signal| signal.is_cancelled())
            && observer.can_apply(&first)
            && first.unavailable.is_none()
            && first.raw.as_ref().is_some_and(|raw| raw.skips.is_empty())
            && window_distribution.is_some();
        if !usable { return messages; }
        match first.raw.as_ref()
            .and_then(|raw| raw.records.iter().find(|record| record.question_id == pi_jev::search::WINDOW_QUESTION_ID))
            .map(|record| jev_line_find::narrow(&record.answer.selected_value()))
        {
            Some(Some(window)) => Some(window),
            _ => return messages,
        }
    } else { None };
    // Pass 2: judge ONLY the inspected window (or the whole small text).
    let Some((state, questions)) = presentation.pass2(window) else { return messages; };
    let Some(entries) = presentation.pass2_entries(window) else { return messages; };
    if !mode.allows_active() {
        observer.observe_prepared(&payload_for(&state), "code_line_find", questions);
        jev_code_search::remember_observation(cache_key);
        return messages;
    }
    if budget.expired() { return messages; }
    let outcome = observer.decide_prepared(&payload_for(&state), "code_line_find", questions, &policy).await;
    let usable = !ctx.signal().is_some_and(|signal| signal.is_cancelled())
        && observer.can_apply(&outcome)
        && outcome.unavailable.is_none()
        && outcome.raw.as_ref().is_some_and(|raw| raw.skips.is_empty());
    if !usable { return messages; }
    let where_decision = outcome.decisions.iter()
        .find(|decision| decision.question_id == pi_jev::search::WHERE_QUESTION_ID);
    let exists_decision = outcome.decisions.iter()
        .find(|decision| decision.question_id == pi_jev::search::EXISTS_QUESTION_ID);
    // The full Choice distribution comes from the raw outcome record.
    let where_probabilities = outcome.raw.as_ref()
        .and_then(|raw| raw.records.iter().find(|record| record.question_id == pi_jev::search::WHERE_QUESTION_ID))
        .and_then(|record| match &record.answer {
            pi_jev::Answer::Choice { probabilities, .. } => Some(probabilities.clone()),
            _ => None,
        });
    let (Some(where_decision), Some(exists_decision), Some(where_probabilities)) =
        (where_decision, exists_decision, where_probabilities)
    else { return messages; };
    let Some(request_id) = outcome.request_id.clone() else { return messages; };
    let Some(annotation) = presentation.annotate(
        where_decision,
        exists_decision,
        Some(&where_probabilities),
        &entries,
        &request_id,
        outcome.turn,
        std::time::SystemTime::now(),
        policy.max_decision_age,
        outcome.response_model.as_deref(),
        &options,
    ) else { return messages; };
    let block = match serde_json::to_string(&annotation) {
        Ok(text) => json!({"type": "text", "text": text}),
        Err(_) => return messages,
    };
    let verdict_text = annotation["jev_line_find"]["verdict"].as_str().unwrap_or_default();
    let Some(verdict) = pi_jev::search::LineFindVerdict::parse(verdict_text) else { return messages; };
    let exists_noul = annotation["jev_line_find"]["exists_noul"].as_f64().unwrap_or_default();
    let top = annotation["jev_line_find"]["top_lines"].as_array()
        .and_then(|lines| lines.first())
        .and_then(|line| line["id"].as_str());
    let metadata = presentation.action_metadata(verdict, exists_noul, top, entries.windowed());
    if presentation.attach(&mut messages, block).is_none() { return messages; }
    observer.record_active_with_action(&outcome, &std::collections::BTreeMap::new(), &metadata);
    messages
}


/// CONTROL seam (API-HANDOFF B7): one applied-control decision at a turn
/// boundary, returning the typed answers for the CONTROL resolvers.
/// None whenever gates are closed, no questions are eligible, or the
/// decision could not complete. Effects are never guessed here.
pub struct ControlDecision {
    pub answers: Vec<pi_jev::active::AnswerCandidate>,
    pub question_ids: Vec<String>,
    pub request_id: String,
    pub turn: u64,
    pub evidence_description: String,
    pub features: pi_jev::control::ControlFeatures,
    pub mode: pi_jev::config::JevMode,
    pub epoch_id: String,
    pub policy_generation: String,
    pub full_jev_stamp: String,
}

/// Capture a citation basis before code-search, line-find, or safety can add
/// request-local annotation blocks. This is local provenance only: no read,
/// provider call, budget, or capability is added.
fn prepare_citation_basis(
    ctx: &Arc<dyn ExtensionContext>,
    messages: &[Value],
    settings: &JevSettings,
) -> Option<(String, crate::core::jev_evidence::CitationPresentation)> {
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let features = settings.effective_features(&session_id);
    if !mode.is_enabled()
        || !features.citation_check
        || ctx.signal().is_some_and(|signal| signal.is_cancelled())
    {
        return None;
    }
    let core = bridge_for_session(&session_id)?;
    let claim = crate::core::jev_retrieval::query_from_messages(messages);
    let claim = if claim.is_empty() {
        core.task_excerpt(&session_id).unwrap_or_default()
    } else {
        claim
    };
    if claim.trim().is_empty() {
        return None;
    }
    let presentation = crate::core::jev_evidence::prepare_original_citation(messages, &claim)?;
    Some((claim, presentation))
}

/// ROOT-CONTRACT v6 (Evidence lane): bounded citation assessment over the
/// ACTUAL ipython source-read path (the same deterministic `print(open(..)
/// .read())` idiom line-find recognizes, mirrored by jev_evidence's own
/// reachability copy). One judgment per provider request, funded from the ONE
/// shared provider-context deadline. Advisory only: the annotation never
/// verifies, never gates an effect, never claims document-wide absence, and
/// the claim and the supplied span stay untrusted data. Fail-open everywhere.
async fn annotate_citation_check(
    ctx: &Arc<dyn ExtensionContext>,
    messages: Vec<Value>,
    settings: &JevSettings,
    budget: &pi_jev::search::SearchBudget,
    citation_basis: Option<(String, crate::core::jev_evidence::CitationPresentation)>,
) -> Vec<Value> {
    let session_id = ctx.session_manager().get_session_id();
    let mode = settings.effective_mode(&session_id);
    let features = settings.effective_features(&session_id);
    if !mode.is_enabled() || !features.citation_check { return messages; }
    if ctx.signal().is_some_and(|signal| signal.is_cancelled()) { return messages; }
    let Some((claim, presentation)) = citation_basis else { return messages; };
    let Some(core) = bridge_for_session(&session_id) else { return messages; };
    let Some(observer) = core.observer(&session_id, Some(ctx.ui())) else { return messages; };
    let generation = decision_policy_generation(settings, &session_id);
    let is_cancelled = || ctx.signal().is_some_and(|signal| signal.is_cancelled());
    let inputs = crate::core::jev_evidence::CitationStageInputs {
        observer,
        session_id: session_id.as_str(),
        turn: core.turn(&session_id),
        mode,
        policy_generation: generation.as_str(),
        max_decision_age: Duration::from_millis(settings.filtering.max_decision_age_ms),
        budget,
        claim: claim.as_str(),
        pre_annotation_presentation: Some(presentation),
        is_cancelled: &is_cancelled,
    };
    crate::core::jev_evidence::annotate_citation_check(messages, &inputs).await
}

pub async fn decide_control(
    session_id: &str,
    event: &ExtensionEvent,
    ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>,
    only_categories: &[&str],
) -> Option<ControlDecision> {
    let settings = load_settings_cached();
    let raw_features = settings.effective_features(session_id);
    let mode = settings.effective_mode(session_id);
    let features = pi_jev::control::ControlFeatures {
        result_sufficiency: raw_features.result_sufficiency,
        loop_control: raw_features.loop_control,
        verification: raw_features.verification,
        retry_classification: raw_features.retry_classification,
        full_jev_active: settings.full_jev_active(),
    };
    if !pi_jev::control::control_gates_open(&features, mode) { return None; }
    let stage = match event {
        ExtensionEvent::TurnEnd(_) if raw_features.loop_control || raw_features.retry_classification
            => pi_jev::snapshot::SnapshotStage::TurnEnd,
        ExtensionEvent::AgentEnd(_)
            if raw_features.result_sufficiency || raw_features.loop_control
                || raw_features.verification || raw_features.retry_classification
            => pi_jev::snapshot::SnapshotStage::AgentEnd,
        _ => return None,
    };
    let core = bridge_for_session(session_id)?;
    let Some((_, mut payload)) = bridge_event(&core, event, ctx, session_id, "decide_control", &settings) else { return None; };
    #[cfg(debug_assertions)]
    if settings.transport.as_deref() == Some("mock-control") {
        payload["state"]["_jev_fixture_lane"] = json!("explicit_control");
    }
    payload["policy_generation"] = json!(decision_policy_generation(&settings, session_id));
    let observer = core.observer(session_id, Some(ctx.ui()))?;
    let state = payload.get("state").cloned().unwrap_or(Value::Null);
    let turn = payload.get("turn").and_then(Value::as_u64).unwrap_or(0);
    let Ok(snapshot) = pi_jev::snapshot::StateSnapshot::new(stage, session_id, turn, 0, None, state, Vec::new()) else { return None; };
    let mut questions = Vec::new();
    for evaluator in pi_jev::evaluators::for_boundary(stage) {
        let category = evaluator.category().as_str();
        let enabled = match category {
            "result_sufficiency" => features.result_sufficiency,
            "continue_stop_escalate" => features.loop_control,
            "first_pass_verification" => features.verification,
            "retry_classification" => features.retry_classification,
            _ => false,
        } && (only_categories.is_empty() || only_categories.contains(&category));
        if enabled {
            if let pi_jev::evaluators::EvaluatorOutput::Questions(mut prepared) = evaluator.evaluate(&snapshot) {
                questions.append(&mut prepared);
            }
        }
    }
    if questions.is_empty() { return None; }
    // H-CONTROL-3: capture the immutable task epoch BEFORE the provider await.
    // The decision is bound to this epoch; consumption re-reads current durable
    // state and refuses any mismatch, so a stale decision can never spend
    // whichever epoch happens to be current at consume time.
    let captured_epoch_id = crate::core::jev_control::ControlBook::global().snapshot(session_id).epoch_id;
    let policy = pi_jev::active::ActivationPolicy { enabled_categories: Default::default(), ..Default::default() };
    let outcome = observer.decide_prepared(&payload, stage.as_str(), questions.clone(), &policy).await;
    observer.record_active(&outcome, &std::collections::BTreeMap::new());
    // Staleness re-check at consume time (N2): current mode/settings must
    // still open the gates and the outcome must still be applicable.
    if !observer.can_apply(&outcome) { return None; }
    let raw = outcome.raw.as_ref()?;
    let request_id = outcome.request_id.clone()?;
    let question_ids: Vec<String> = questions.iter().map(|q| q.question_id.clone()).collect();
    let now = std::time::SystemTime::now();
    // Inline of hooks::policy_confidence (private there): optional
    // categories report min(answer confidence, selected probability).
    let answers = raw.records.iter().map(|record| {
        let mut confidence = record.answer.confidence();
        if pi_jev::active::OPTIONAL_APPLIABLE_CATEGORIES.contains(&record.category) {
            if let pi_jev::types::Answer::Choice { choice, probabilities, .. } = &record.answer {
                confidence = probabilities.get(choice).copied().map(|p| confidence.unwrap_or(p).min(p));
            } else {
                confidence = None;
            }
        }
        pi_jev::active::AnswerCandidate {
            category: record.category,
            question_id: record.question_id.clone(),
            value: Some(record.answer.selected_value()),
            confidence,
            response_model: record.response_model.clone(),
            request_id: request_id.clone(),
            turn: outcome.turn,
            decided_at: now,
        }
    }).collect();
    let mut evidence_description = core.observation(session_id).evidence_description();
    // TOOL-001: append the truthful offered-tool fact so a corrective
    // feedback message answers a "no tools attached" claim with the actual
    // last observed advertisement instead of login/reconnect advice.
    if let Some(fact) = core.advertised_tools_fact(session_id) {
        evidence_description.push_str("; ");
        evidence_description.push_str(&fact);
    }
    Some(ControlDecision {
        answers,
        question_ids,
        request_id,
        turn: outcome.turn,
        evidence_description,
        features,
        mode,
        epoch_id: captured_epoch_id,
        policy_generation: outcome.policy_generation.clone(),
        full_jev_stamp: settings.full_jev_stamp(),
    })
}

/// CTRL-001 trusted-producer gate: verification-designated reports are
/// accepted ONLY from the native ipython tool, the sole supported producer
/// of the typed script-report pipe (`create_ipython_tool_definition`,
/// `core::tools::ipython`). An `executionReports`-shaped details payload
/// echoed by any other tool is ignored: schema shape alone never
/// establishes host-measured execution.
const SCRIPT_REPORT_TOOL_NAME: &str = "ipython";

/// CTRL-001: verification-designated reports from one observed ipython tool
/// result, parsed through the SUPPORTED structured report path only (callers
/// must first pass the `SCRIPT_REPORT_TOOL_NAME` provenance gate). Strictly
/// validated (a malformed designation invalidates its whole report), and a
/// designation is required: a bare exit code is never verification.
fn designated_verification_reports(
    payload: &ToolExecutionEndPayload,
) -> Option<Vec<crate::core::kernel::shared::ScriptExecutionReport>> {
    let reports_value = payload
        .result
        .get("details")?
        .get(crate::core::kernel::shared::EXECUTION_REPORTS_DETAILS_KEY)?;
    let reports = crate::core::kernel::shared::parse_execution_reports(Some(reports_value))
        .ok()??;
    let designated: Vec<_> = reports
        .into_iter()
        .filter(|report| report.verification_label().is_some())
        .collect();
    (!designated.is_empty()).then_some(designated)
}

fn observed_failure_kind(message: &Value) -> Option<pi_jev::observation::RetryFailureKind> {
    use pi_jev::observation::RetryFailureKind;
    for diagnostic in message.get("diagnostics")?.as_array()?.iter().take(32) {
        match diagnostic.get("type").and_then(Value::as_str) {
            Some("agent_lifecycle_failure") => return Some(RetryFailureKind::Fatal),
            Some("provider_stream_failure") => {
                let kind = diagnostic.get("details").and_then(|details| details.get("kind")).and_then(Value::as_str).unwrap_or("");
                return Some(RetryFailureKind::from_provider_kind(kind));
            }
            _ => {}
        }
    }
    None
}

fn activation_policy(features: pi_jev::config::JevFeatures) -> pi_jev::active::ActivationPolicy {
    use pi_jev::types::DecisionCategory;
    let enabled_categories = [
        (features.tool_requirement, DecisionCategory::ToolRequirement),
        (features.complexity, DecisionCategory::Complexity),
        (features.tool_candidates, DecisionCategory::ToolCandidates),
        (features.context_relevance, DecisionCategory::ContextRelevance),
        (features.memory_relevance, DecisionCategory::MemoryRelevance),
    ].into_iter().filter_map(|(enabled, category)| enabled.then_some(category)).collect();
    pi_jev::active::ActivationPolicy { enabled_categories, ..Default::default() }
}

fn request_action(params: &Value) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let tools = params.get("tools").and_then(Value::as_array);
    result.insert("tools".to_string(), tools.map(|tools| format!("count:{}", tools.len())).unwrap_or_else(|| "absent".to_string()));
    result.insert("execution_tool".to_string(), if tools.is_some_and(|tools| tools.iter().any(|tool| tool.get("name").or_else(|| tool.get("function").and_then(|function| function.get("name"))).and_then(Value::as_str) == Some("ipython"))) { "ipython_advertised" } else { "ipython_not_advertised" }.to_string());
    result.insert("tool_state_scope".to_string(), "local_request_not_provider_receipt".to_string());
    if let Some((key, effort)) = crate::core::jev_active::reasoning_effort(params) {
        result.insert(key.to_string(), effort.to_string());
    }
    let choice = match params.get("tool_choice") {
        None => "absent",
        Some(Value::String(value)) => match value.as_str() {
            "auto" => "auto",
            "none" => "none",
            "required" => "required",
            _ => "unknown",
        },
        Some(Value::Object(value)) => match value.get("type").and_then(Value::as_str) {
            Some("function") => "function",
            Some("allowed_tools") => "allowed_tools",
            Some("custom") => "custom",
            _ => "unknown",
        },
        _ => "unknown",
    };
    result.insert("tool_choice".to_string(), choice.to_string());
    result
}

fn bridge_for_session(session_id: &str) -> Option<Arc<JevBridgeCore>> {
    let cores: Vec<_> = live_bridges().lock().unwrap_or_else(|p| p.into_inner()).iter().filter_map(Weak::upgrade).collect();
    cores.iter().find(|core| core.sessions.lock().unwrap_or_else(|p| p.into_inner()).contains_key(session_id)).cloned()
}

/// Bounded state for one Active decision boundary at the provider edge.
///
/// Reuses the same field names the Compare path sends at `turn_start`, so both
/// modes ask the same questions of the same snapshot. The tool catalog is the
/// one this request actually advertises, which is what the tool-candidate and
/// tool-requirement evaluators need; the observed-tool history is the fallback.
fn active_request_state(
    core: &Arc<JevBridgeCore>,
    ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>,
    session_id: &str,
    params: &Value,
    settings: &JevSettings,
) -> Value {
    // Request-local availability must not fall back to tools from old turns.
    let observed = advertised_tool_names(params);
    let model_id = ctx.model().map(|model| model.id.clone());
    json!({
        "session_id": session_id,
        "turn": core.turn(session_id),
        "model": model_id,
        "baseline_action": request_action(params),
        "compaction_enabled": settings.effective_compaction_enabled(session_id),
        "policy_generation": decision_policy_generation(settings, session_id),
        "state": {
            "features": settings.effective_features(session_id),
            "observation": core.observation(session_id),
            "user_text_excerpt": core.task_excerpt(session_id),
            "observed_tools": observed,
            "message_count": ctx.session_manager().get_entry_count(),
            "model": model_id,
            "model_allowlist": Vec::<String>::new(),
            "cwd_name": std::path::Path::new(&ctx.cwd())
                .file_name()
                .map(|name| name.to_string_lossy().to_string()),
        },
    })
}

/// Tool names advertised in one provider request body, in order, bounded.
///
/// Only `function.name` entries are read. A tool schema is public by
/// definition: it is already on its way to the model provider.
fn advertised_tool_names(params: &Value) -> Vec<String> {
    const MAX_ADVERTISED_TOOLS: usize = 16;
    const MAX_ADVERTISED_NAME_CHARS: usize = 128;
    let Some(tools) = params.get(crate::core::jev_active::TOOLS_KEY).and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    for tool in tools {
        let name = tool
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| tool.get("name").and_then(Value::as_str));
        if let Some(name) = name {
            if !name.is_empty()
                && name.chars().count() <= MAX_ADVERTISED_NAME_CHARS
                && !names.iter().any(|seen| seen == name)
            {
                names.push(name.to_string());
            }
        }
        if names.len() >= MAX_ADVERTISED_TOOLS {
            break;
        }
    }
    names
}

/// Convert an `ExtensionEvent` into the bounded `(event_type, payload)` DTO
/// the transport-agnostic observer consumes. Everything here is bounded:
/// message text is excerpt-capped, tool args excerpted, no full transcripts.
fn bridge_event(
    core: &Arc<JevBridgeCore>,
    event: &ExtensionEvent,
    ctx: &Arc<dyn crate::core::extensions::types::ExtensionContext>,
    session_id: &str,
    handler_event: &'static str,
    settings: &JevSettings,
) -> Option<(String, Value)> {
    let _ = handler_event;
    let model_id = ctx.model().map(|model| model.id.clone());
    let message_count = ctx.session_manager().get_entry_count();
    let turn = core.turn(session_id);
    // No user-approved allowlist source exists in this build; category 6
    // records an explicit `no_model_allowlist` skip instead of inventing
    // candidates. The selected model id is still observed for the record.
    let allowlist: Vec<String> = Vec::new();
    let mut result = match event {
        ExtensionEvent::SessionStart(_) => Some((
            "session_start".to_string(),
            json!({ "session_id": session_id, "reason": "startup" }),
        )),
        ExtensionEvent::AgentStart => Some((
            "agent_start".to_string(),
            json!({ "session_id": session_id }),
        )),
        ExtensionEvent::TurnStart(payload) => Some((
            "turn_start".to_string(),
            json!({
                "session_id": session_id,
                "turn": payload.turn_index as u64,
                "model": model_id,
                "state": {
                    "user_text_excerpt": core.task_excerpt(session_id),
                    "observed_tools": core.observed_tools(session_id),
                    "message_count": message_count,
                    "model": model_id,
                    "model_allowlist": allowlist,
                    "cwd_name": std::path::Path::new(&ctx.cwd())
                        .file_name()
                        .map(|name| name.to_string_lossy().to_string()),
                },
            }),
        )),
        ExtensionEvent::Input(payload) => Some((
            "input".to_string(),
            json!({
                "session_id": session_id,
                "turn": turn,
                "state": {
                    "user_text_excerpt": pi_jev::redact::bounded_excerpt(&payload.text, 400),
                    "message_count": message_count,
                    "model": model_id,
                    "model_allowlist": allowlist,
                },
            }),
        )),
        ExtensionEvent::ToolCall(tool_call) => {
            let tool_name = tool_call.tool_name().to_string();
            let tool_call_id = tool_call.tool_call_id().to_string();
            let mut state = tool_call_observation(&tool_name, &tool_call_id, &allowlist);
            state["user_text_excerpt"] = json!(core.task_excerpt(session_id));
            Some((
                "tool_call".to_string(),
                json!({
                    "session_id": session_id,
                    "turn": turn,
                    "tool_name": tool_name,
                    "model": model_id,
                    "state": state,
                }),
            ))
        }
        ExtensionEvent::ToolExecutionStart(payload) => Some((
            "tool_execution_start".to_string(),
            json!({
                "session_id": session_id,
                "tool_name": pi_jev::snapshot::truncate_text(&payload.tool_name, 120).0,
            }),
        )),
        ExtensionEvent::ToolExecutionEnd(payload) => Some((
            "tool_execution_end".to_string(),
            json!({
                "session_id": session_id,
                "tool_name": pi_jev::snapshot::truncate_text(&payload.tool_name, 120).0,
                "is_error": payload.is_error,
            }),
        )),
        ExtensionEvent::MessageEnd(payload) => Some((
            "message_end".to_string(),
            json!({
                "session_id": session_id,
                "role": payload.message.get("role").cloned().unwrap_or(Value::Null),
            }),
        )),
        ExtensionEvent::ModelSelect(payload) => Some((
            "model_select".to_string(),
            json!({
                "session_id": session_id,
                "selected_model": payload.model.id.clone(),
                "model": payload.model.id.clone(),
                "state": {
                    "model_allowlist": allowlist,
                    "selected_model": payload.model.id,
                    "previous_model": payload
                        .previous_model
                        .as_ref()
                        .map(|model| model.id.clone()),
                },
            }),
        )),
        ExtensionEvent::TurnEnd(payload) => Some(("turn_end".to_string(), json!({
            "session_id": session_id, "turn": payload.turn_index as u64, "model": model_id,
            "state": { "result_excerpt": bounded_assistant_excerpt(&payload.message),
                "user_text_excerpt": core.task_excerpt(session_id), "model_allowlist": allowlist },
        }))),
        ExtensionEvent::AgentEnd(payload) => {
            // Bounded result summary: stop reason + excerpt of the last
            // assistant text; never full transcripts.
            let summary = summarize_agent_end(&payload.messages);
            Some((
                "agent_end".to_string(),
                json!({
                    "session_id": session_id,
                    "turn": turn,
                    "model": model_id,
                    "state": {
                        "result_excerpt": summary.result_excerpt,
                        "user_text_excerpt": core.task_excerpt(session_id),
                        "stop_reason": summary.stop_reason,
                        "message_count": message_count,
                        "model_allowlist": allowlist,
                        // TOOL-001: observed advertisement facts for the
                        // records; the last OBSERVED request only, with
                        // explicit staleness (never a current guarantee).
                        "offered_tools": core.last_advertised_tools(session_id),
                        "offered_tools_current": core.advertised_tools_is_current(session_id),
                    },
                }),
            ))
        }
        ExtensionEvent::SessionShutdown(payload) => Some((
            "session_shutdown".to_string(),
            json!({
                "session_id": session_id,
                "reason": payload.reason,
            }),
        )),
        _ => None,
    };
    if let Some((_, payload)) = result.as_mut() {
        // All request identity and feature bytes derive from the caller's ONE
        // settings snapshot. Hooks compare this generation with the single-load
        // authoritative gate before any dispatch.
        payload["compaction_enabled"] = json!(settings.effective_compaction_enabled(session_id));
        payload["policy_generation"] = json!(decision_policy_generation(settings, session_id));
        if let Some(state) = payload.get_mut("state").and_then(Value::as_object_mut) {
            state.insert("features".to_string(), json!(settings.effective_features(session_id)));
            state.insert("observation".to_string(), json!(core.observation(session_id)));
        }
    }
    result
}

/// Observation state for one tool call.
///
/// Tool ARGUMENTS are never observed. Callers pass credentials, connection
/// strings and file bodies through tool input, and the tool-choice evaluators
/// only need tool identity. Serializing `input()` here - even truncated right
/// afterwards - would copy a potentially multi-megabyte payload and could
/// disclose a credential that sits in its first characters.
fn tool_call_observation(tool_name: &str, tool_call_id: &str, allowlist: &[String]) -> Value {
    json!({
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "args_omitted": true,
        "model_allowlist": allowlist,
    })
}

struct AgentEndSummary {
    result_excerpt: Option<String>,
    stop_reason: Option<String>,
}

/// Bounded, redacted summary of the final assistant message.
///
/// The excerpt is accumulated character by character and stops at the redaction
/// scan window, so a multi-megabyte final message is never cloned or joined in
/// full just to keep 400 characters. Redaction happens before the value reaches
/// the snapshot state.
fn summarize_agent_end(messages: &[Value]) -> AgentEndSummary {
    let last_assistant = messages
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("assistant"));
    let Some(message) = last_assistant else {
        return AgentEndSummary {
            result_excerpt: None,
            stop_reason: None,
        };
    };
    AgentEndSummary {
        result_excerpt: Some(bounded_assistant_excerpt(message)),
        stop_reason: message
            .get("stopReason")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// Incremental excerpt of one assistant message's text blocks.
fn bounded_assistant_excerpt(message: &Value) -> String {
    let mut collected = String::new();
    let mut budget = pi_jev::redact::MAX_SCAN_CHARS;
    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
        for block in blocks {
            if budget == 0 {
                break;
            }
            let Some(text) = block.get("text").and_then(Value::as_str) else {
                continue;
            };
            if !collected.is_empty() && budget > 1 {
                collected.push(' ');
                budget -= 1;
            }
            for character in text.chars() {
                if budget == 0 {
                    break;
                }
                collected.push(character);
                budget -= 1;
            }
        }
    }
    pi_jev::redact::bounded_excerpt(&collected, 400)
}

/// High-confidence valid answers that pick the most "act now"-looking option
/// in every question (e.g. escalate, insufficient). These can only land in
/// records: no code path consumes them.
fn hostile_response_for(request: &pi_jev::types::SystemOneRequest) -> pi_jev::types::SystemOneResponse {
    let mut answers = std::collections::BTreeMap::new();
    for (id, spec) in &request.questions {
        let answer = match spec {
            pi_jev::types::QuestionSpec::Noul { .. } => pi_jev::types::Answer::Noul { noul: 1.0 },
            pi_jev::types::QuestionSpec::Choice { criteria, .. } => {
                let choice = criteria
                    .keys()
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "none".to_string());
                let mut probabilities = std::collections::BTreeMap::new();
                for key in criteria.keys() {
                    probabilities.insert(key.clone(), 0.0);
                }
                probabilities.insert(choice.clone(), 1.0);
                pi_jev::types::Answer::Choice {
                    choice,
                    probabilities,
                    confidence: 1.0,
                }
            }
            pi_jev::types::QuestionSpec::Score { criteria, .. } => {
                let mut probabilities = std::collections::BTreeMap::new();
                for index in 0..criteria.len() {
                    probabilities.insert(index.to_string(), 0.0);
                }
                if let Some(last) = criteria.len().checked_sub(1) {
                    probabilities.insert(last.to_string(), 1.0);
                }
                let weighted: f64 = probabilities
                    .iter()
                    .map(|(k, v)| k.parse::<usize>().map(|i| v * i as f64).unwrap_or(0.0))
                    .sum();
                let mut legend = std::collections::BTreeMap::new();
                for (index, level) in criteria.iter().enumerate() {
                    legend.insert(index.to_string(), level.clone());
                }
                pi_jev::types::Answer::Score {
                    score: weighted,
                    legend,
                    probabilities,
                    confidence: 1.0,
                }
            }
        };
        answers.insert(id.clone(), answer);
    }
    pi_jev::types::SystemOneResponse {
        model: "jev-mock-hostile/1".to_string(),
        answers,
        usage: pi_jev::types::Usage {
            input_tokens: Some(1),
            output_tokens: Some(1),
        },
        // The hostile fixture parses every answer; no recorded parse skips and
        // no captured server request id.
        answer_parse_skips: Vec::new(),
        server_request_id: None,
    }
}

// ---------------------------------------------------------------------------
// Negative-capability unit tests (no subagent control, ever)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod no_subagent_control_tests {
    use super::*;

    #[test]
    fn credential_stamp_tracks_the_actual_named_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let settings = JevSettings::default();
        let before = credential_stamp_at(&settings, dir.path());
        let jev_dir = dir.path().join("jev");
        std::fs::create_dir_all(&jev_dir).unwrap();
        let envelope = jev_dir.join(format!("{}.{}",
            pi_jev::config::DEFAULT_KEY_ID, pi_jev::credential::CREDENTIAL_FILE_NAME));
        std::fs::write(&envelope, b"synthetic-encrypted-envelope").unwrap();
        let stored = credential_stamp_at(&settings, dir.path());
        assert_ne!(before, stored);
        std::fs::write(&envelope, b"different-synthetic-encrypted-envelope").unwrap();
        assert_ne!(stored, credential_stamp_at(&settings, dir.path()));
        std::fs::remove_file(envelope).unwrap();
        assert_eq!(before, credential_stamp_at(&settings, dir.path()));
        assert!(!stored.contains("synthetic"));
    }

    /// The internal observer extension must expose NO callable capability
    /// surface into the runtime: zero tools, commands, shortcuts, flags and
    /// message renderers. Handlers on observe-only events only.
    #[test]
    fn observer_extension_has_no_mutable_capability_surface() {
        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        let extension = build_observer_extension(core);
        let extension = extension.lock().unwrap();
        assert!(extension.tools.is_empty(), "Jev must register no tools");
        assert!(extension.commands.is_empty(), "Jev must register no commands");
        assert!(extension.shortcuts.is_empty(), "Jev must register no shortcuts");
        assert!(extension.flags.is_empty(), "Jev must register no flags");
        assert!(
            extension.message_renderers.is_empty(),
            "Jev must register no message renderers"
        );
        let mut mutating = 0usize;
        for event_type in extension.handlers.keys() {
            assert!(
                !event_type.starts_with("session_before_")
                    && !matches!(
                        event_type.as_str(),
                        "context"
                            | "before_provider_request"
                            | "before_agent_start"
                            | "tool_result"
                            | "user_bash"
                            | "message_update"
                    ),
                "Jev must never subscribe to mutating surface {event_type}"
            );
            let _ = &mut mutating;
        }
        // Every registered event is one of the documented observe-only set.
        for event_type in JEV_EVENTS {
            assert!(
                extension.handlers.contains_key(event_type),
                "expected handler for {event_type}"
            );
        }
    }

    /// The documented event list itself must stay free of mutating surfaces.
    #[test]
    fn jev_event_list_excludes_all_mutating_surfaces() {
        for event_type in JEV_EVENTS {
            assert!(
                !event_type.starts_with("session_before_"),
                "session_before_* is decision-returning: {event_type}"
            );
            assert!(
                !matches!(
                    event_type,
                    "context"
                        | "before_provider_request"
                        | "before_agent_start"
                        | "tool_result"
                        | "user_bash"
                        | "message_update"
                ),
                "mutating surface {event_type} must never be observed"
            );
        }
    }

    /// Active is operative: it wants the observer, keeps the exact per-session
    /// value (never coerced to Compare or Off), and is not a Compare mode.
    #[test]
    fn active_settings_select_active_and_never_compare() {
        let mut settings = JevSettings::default();
        settings.global_default = Some(JevMode::Active);
        settings.sessions.insert(
            "any-session".to_string(),
            pi_jev::config::PersistedSessionMode {
                mode: Some(JevMode::Active),
                inherited_from: None,
                ..Default::default()
            },
        );
        assert!(settings.wants_observer());
        assert_eq!(settings.effective_mode("any-session"), JevMode::Active);
        assert!(!JevMode::Active.allows_compare());
        // A per-session override always wins over the global default, in both
        // directions, so an Active session never silently becomes Compare.
        settings.sessions.insert(
            "any-session".to_string(),
            pi_jev::config::PersistedSessionMode {
                mode: Some(JevMode::Off),
                inherited_from: None,
                ..Default::default()
            },
        );
        assert_eq!(settings.effective_mode("any-session"), JevMode::Off);
    }

    #[cfg(test)]
mod full_jev_overlay_wiring_tests {
    use super::*;
    use pi_jev::config::{JevSettings, JevSettingsStore};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    /// Serializes tests that mutate the process-wide agent-dir environment
    /// variable and the global settings cache (the migrations.rs ENV_LOCK
    /// pattern): one env-mutating bridge test at a time.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn handler_count(extension: &SharedExtension) -> usize {
        extension
            .lock()
            .unwrap()
            .handlers
            .get(JEV_ACTIVE_EVENT)
            .map(|handlers| handlers.len())
            .unwrap_or(0)
    }

    fn test_settings(agent_dir: &Path) -> JevSettings {
        JevSettingsStore::new(agent_dir).load()
    }

    /// ROOT-CONTRACT staleness condition (root ABA finding): the persisted
    /// overlay identity must be FRESH on every off->on transition. A consumer
    /// that never observed the intermediate Off must still reject work that
    /// was captured under a previous activation. Verified across store
    /// reloads, because a separate process only ever sees persisted state.
    #[test]
    fn full_stamp_differs_between_separate_activations() {
        let dir = tempfile::tempdir().expect("tempdir");
        let agent_dir = dir.path();

        // First activation through a real persisted store.
        let mut first = test_settings(agent_dir);
        assert!(first.full_jev_install());
        JevSettingsStore::new(agent_dir).save(&first).expect("save first");
        let first_stamp = first.full_jev_stamp();
        let first_cheap = credential_stamp_at(&first, agent_dir);

        // Off through a distinct loaded snapshot (a writer that missed the
        // first activation, e.g. another process).
        let mut off = test_settings(agent_dir);
        assert!(off.full_jev_remove());
        JevSettingsStore::new(agent_dir).save(&off).expect("save off");

        // Second activation, again from a fresh persisted load.
        let mut second = test_settings(agent_dir);
        assert!(second.full_jev_install());
        JevSettingsStore::new(agent_dir).save(&second).expect("save second");
        let second_stamp = second.full_jev_stamp();
        let second_cheap = credential_stamp_at(&second, agent_dir);

        assert_ne!(
            first_stamp, second_stamp,
            "off->on must mint a fresh activation identity; equal stamps let a consumer that missed the Off accept stale first-activation work"
        );
        assert_ne!(
            first_cheap, second_cheap,
            "the bridge cheap stamp must change across an activation cycle so in-flight work is invalidated like a key rotation"
        );
    }

    /// Idempotent already-on installs must not churn the stamp, and removing
    /// twice is a truthful no-op.
    #[test]
    fn full_jev_install_remove_idempotency_keeps_stamps_stable() {
        let mut settings = JevSettings::default();
        assert!(settings.full_jev_install());
        let stamp_on = settings.full_jev_stamp();
        assert!(!settings.full_jev_install(), "already-active install is a no-op");
        assert_eq!(settings.full_jev_stamp(), stamp_on, "no-op install must not churn the stamp");
        assert!(settings.full_jev_remove());
        assert!(!settings.full_jev_remove(), "second remove is a no-op");
        assert_eq!(settings.full_jev_stamp(), "full-jev:0");
    }

    /// Cross-process Off->Full: a distinct store instance installs the overlay
    /// (simulating another process writing settings). After the 250ms settings
    /// TTL expires, the next settings read must resync Active-handler presence
    /// on live bridges WITHOUT any same-process invalidate call.
    #[test]
    fn ttl_reload_resyncs_handlers_off_to_full() {
        let _guard = env_lock().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let agent_dir = dir.path().to_path_buf();
        std::env::set_var(crate::config::env_agent_dir(), agent_dir.clone());
        invalidate_settings_cache();

        // Base: everything Off, saved once.
        let store = JevSettingsStore::new(&agent_dir);
        store.save(&JevSettings::default()).expect("save base");
        invalidate_settings_cache();

        let mut extensions: Vec<SharedExtension> = Vec::new();
        maybe_register_jev_observer(&mut extensions);
        let extension = extensions
            .first()
            .expect("observer extension registered")
            .clone();
        assert_eq!(handler_count(&extension), 0, "Off baseline installs no handler");

        // A DISTINCT store instance (another process) installs full-jev.
        let mut writer = JevSettingsStore::new(&agent_dir).load();
        assert!(writer.full_jev_install());
        JevSettingsStore::new(&agent_dir).save(&writer).expect("save full");
        // No invalidate_settings_cache() here: the TTL path is under test.

        std::thread::sleep(SETTINGS_TTL + Duration::from_millis(40));
        assert!(active_mode_requested(), "fresh read sees the overlay");
        assert_eq!(
            handler_count(&extension),
            1,
            "TTL reload must resync handler presence without a same-process invalidate"
        );

        invalidate_settings_cache();
        std::env::remove_var(crate::config::env_agent_dir());
    }

    /// Cross-process Full->Off: the reverse direction of the same contract.
    #[test]
    fn ttl_reload_resyncs_handlers_full_to_off() {
        let _guard = env_lock().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let agent_dir = dir.path().to_path_buf();
        std::env::set_var(crate::config::env_agent_dir(), agent_dir.clone());
        invalidate_settings_cache();

        // Base: full-jev already installed by a previous writer.
        let mut initial = JevSettings::default();
        assert!(initial.full_jev_install());
        let store = JevSettingsStore::new(&agent_dir);
        store.save(&initial).expect("save full base");
        invalidate_settings_cache();

        let mut extensions: Vec<SharedExtension> = Vec::new();
        maybe_register_jev_observer(&mut extensions);
        let extension = extensions
            .first()
            .expect("observer extension registered")
            .clone();
        assert_eq!(handler_count(&extension), 1, "overlay-active baseline installs the handler");

        // A DISTINCT store instance removes full-jev.
        let mut writer = JevSettingsStore::new(&agent_dir).load();
        assert!(writer.full_jev_remove());
        JevSettingsStore::new(&agent_dir).save(&writer).expect("save off");

        std::thread::sleep(SETTINGS_TTL + Duration::from_millis(40));
        assert!(!active_mode_requested(), "fresh read sees the removal");
        assert_eq!(
            handler_count(&extension),
            0,
            "TTL reload must remove handler presence after a cross-process full-off"
        );

        invalidate_settings_cache();
        std::env::remove_var(crate::config::env_agent_dir());
    }
}

/// The provider-request handler exists only while Active is requested.
    ///
    /// Its presence alone tells the runner that a request body may change,
    /// which also gates retry reuse of semantic edges. A process where nobody
    /// enabled Active must therefore keep the handler absent.
    #[test]
    fn active_handler_presence_follows_the_setting() {
        fn active_handlers(extension: &SharedExtension) -> usize {
            let guard = extension.lock().unwrap();
            guard
                .handlers
                .get(JEV_ACTIVE_EVENT)
                .map(|handlers| handlers.len())
                .unwrap_or(0)
        }

        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        // A default-Off build registers no provider-request handler at all.
        let extension = build_observer_extension(Arc::clone(&core));
        assert_eq!(active_handlers(&extension), 0);
        core.attach_extension(&extension);
        // Installing is idempotent; removing leaves an empty entry.
        core.set_active_handler(&core, true);
        assert_eq!(active_handlers(&extension), 1);
        core.set_active_handler(&core, true);
        assert_eq!(active_handlers(&extension), 1);
        core.set_active_handler(&core, false);
        assert_eq!(active_handlers(&extension), 0);
    }
}


#[cfg(test)]
mod observation_redaction_tests {
    use super::*;
    use serde_json::json;

    /// v5 usage knownness: compaction stats keep measured zeros, omit UNKNOWN.
    #[test]
    fn compaction_usage_stats_keep_known_zeros_and_omit_unknown() {
        let mut stats = json!({"reason": "x"});
        record_usage_stats(&pi_jev::types::Usage::unknown(), &mut stats);
        assert!(stats.get("jev_input_tokens").is_none(), "UNKNOWN usage is omitted, not fabricated as 0");
        assert!(stats.get("jev_output_tokens").is_none(), "UNKNOWN usage is omitted, not fabricated as 0");
        record_usage_stats(&pi_jev::types::Usage { input_tokens: Some(0), output_tokens: Some(7) }, &mut stats);
        assert_eq!(stats["jev_input_tokens"], json!(0), "a measured zero is a known zero and is kept");
        assert_eq!(stats["jev_output_tokens"], json!(7));
    }

    #[test]
    fn diagnostic_observation_uses_native_shape_without_error_text() {
        use pi_jev::observation::RetryFailureKind;
        let message=json!({"role":"assistant","stopReason":"error","errorMessage":"secret raw provider error",
            "diagnostics":[{"type":"provider_stream_failure","details":{"kind":"rate_limit","body":"do not copy"}}]});
        assert_eq!(observed_failure_kind(&message),Some(RetryFailureKind::RateLimited));
        let lifecycle=json!({"diagnostics":[{"type":"agent_lifecycle_failure","details":{"kind":"do not infer transient"}}]});
        assert_eq!(observed_failure_kind(&lifecycle),Some(RetryFailureKind::Fatal));
        assert_eq!(observed_failure_kind(&json!({"errorKind":"rate_limit"})),None);
        let mut many=vec![json!({"type":"other"});32];
        many.push(json!({"type":"provider_stream_failure","details":{"kind":"rate_limit"}}));
        assert_eq!(observed_failure_kind(&json!({"diagnostics":many})),None);
    }

    #[test]
    fn session_ownership_never_falls_back_to_an_unrelated_bridge() {
        let first=Arc::new(JevBridgeCore::new(JevSettings::default()));
        let second=Arc::new(JevBridgeCore::new(JevSettings::default()));
        first.own_session("ownership-first-unique");
        second.own_session("ownership-second-unique");
        {
            let mut bridges=live_bridges().lock().unwrap();
            bridges.push(Arc::downgrade(&first)); bridges.push(Arc::downgrade(&second));
        }
        assert!(Arc::ptr_eq(&bridge_for_session("ownership-first-unique").unwrap(),&first));
        assert!(Arc::ptr_eq(&bridge_for_session("ownership-second-unique").unwrap(),&second));
        first.forget_session("ownership-first-unique");
        assert!(bridge_for_session("ownership-first-unique").is_none());
        assert!(bridge_for_session("never-registered-unique").is_none());
    }

    /// Fixtures used across these tests: values that must never reach a
    /// SystemOne request built from a shadow observation.
    const SECRETS: [&str; 6] = [
        "sk-live-abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        "hunter2-the-password",
        "dXNlcjpwYXNzd29yZA==",
        "AKIAIOSFODNN7EXAMPLE",
        "MIIEowIBAAKCAQEAprivatekeymaterial",
    ];

    fn assert_no_secret(payload: &Value) {
        let text = serde_json::to_string(payload).unwrap();
        for secret in SECRETS {
            assert!(!text.contains(secret), "{secret} reached the payload: {text}");
        }
    }

    #[test]
    fn tool_arguments_are_never_observed_or_serialized() {
        let state = tool_call_observation("bash", "call-1", &[]);
        assert_eq!(state["args_omitted"], json!(true));
        assert!(state.get("args_excerpt").is_none(), "{state}");
        // Source guard: serializing tool input anywhere in the PRODUCTION
        // bridge would reintroduce the raw-argument capture this repair
        // removed. The scan stops at the first `#[cfg(test)]` module, because
        // test code must be able to name the field it forbids. Tokens are
        // split so the guard cannot match its own source text.
        let source = include_str!("jev_bridge.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or(source);
        let raw_args_field = concat!("args", "_exc", "erpt");
        let raw_input_call = concat!("tool_call", ".in", "put()");
        assert!(
            !production.contains(raw_args_field) && !production.contains(raw_input_call),
            "the bridge serializes raw tool arguments again"
        );
    }

    #[test]
    fn secret_bearing_task_text_is_redacted_before_the_payload() {
        let raw = "deploy with TYPESAFE_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz \
                  and Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz";
        let state = json!({ "user_text_excerpt": pi_jev::redact::bounded_excerpt(raw, 400) });
        assert_no_secret(&state);
        assert!(serde_json::to_string(&state).unwrap().contains(pi_jev::redact::REDACTED));
    }

    #[test]
    fn secret_bearing_final_message_is_redacted_in_the_end_of_turn_state() {
        let messages = vec![json!({
            "role": "assistant",
            "stopReason": "end_turn",
            "content": [
                { "type": "text", "text": "wrote the config" },
                { "type": "text", "text": "password=hunter2-the-password" },
                { "type": "text", "text": "token AKIAIOSFODNN7EXAMPLE ok" },
            ],
        })];
        let summary = summarize_agent_end(&messages);
        let excerpt = summary.result_excerpt.unwrap_or_default();
        for secret in SECRETS {
            assert!(!excerpt.contains(secret), "{secret} survived: {excerpt}");
        }
        assert!(excerpt.contains("wrote the config"), "{excerpt}");
    }

    #[test]
    fn a_multi_megabyte_final_message_is_bounded_without_a_full_copy() {
        // 4 MiB assistant message ending in a credential: the excerpt must stay
        // bounded and must not disclose the credential.
        let big = format!(
            "{}{}",
            "x".repeat(4 * 1024 * 1024),
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAprivatekeymaterial"
        );
        let messages = vec![json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": big }],
        })];
        let summary = summarize_agent_end(&messages);
        let excerpt = summary.result_excerpt.unwrap_or_default();
        assert!(excerpt.chars().count() <= 400);
        for secret in SECRETS {
            assert!(!excerpt.contains(secret), "{secret} survived: {excerpt}");
        }
    }

    #[test]
    fn observation_state_never_carries_a_raw_allowlist_or_transcript() {
        let allowlist: Vec<String> = Vec::new();
        let state = tool_call_observation("read_file", "call-2", &allowlist);
        assert_eq!(state["model_allowlist"], json!([]));
        assert_eq!(state["tool_name"], json!("read_file"));
    }
}

#[cfg(test)]
mod provider_action_tests {
    use super::*;

    #[test]
    fn ctrl001_task_epoch_resets_trace_once_not_on_internal_followups() {
        use pi_jev::observation::{TraceEvent, VerificationEvidence};
        let mut book = SessionBook::default();
        book.note_control_epoch("session:1");
        book.trace.record(TraceEvent::ToolEnded { is_error: false });
        book.trace.record(TraceEvent::VerificationObserved { outcome: VerificationEvidence::NotNeeded });
        book.note_control_epoch("session:1");
        book.trace.record(TraceEvent::TurnStarted);
        assert_eq!(book.trace.summary().tool_results, 1);
        assert_eq!(book.trace.summary().verification, VerificationEvidence::NotNeeded);
        book.note_control_epoch("session:2");
        assert_eq!(book.trace.summary().tool_results, 0);
        assert_eq!(book.trace.summary().verification, VerificationEvidence::Unknown);
    }

    #[test]
    fn tool001_request_diagnostics_distinguish_retained_disabled_and_absent() {
        for choice in ["auto", "none", "required"] {
            let params = json!({"tools":[{"type":"function","name":"ipython","parameters":{"secret":"DO_NOT_LOG"}}],"tool_choice":choice});
            let action = request_action(&params);
            assert_eq!(action["tools"], "count:1");
            assert_eq!(action["execution_tool"], "ipython_advertised");
            assert_eq!(action["tool_choice"], choice);
            assert_eq!(action["tool_state_scope"], "local_request_not_provider_receipt");
            assert!(!format!("{action:?}").contains("DO_NOT_LOG"));
        }
        let absent = request_action(&json!({}));
        assert_eq!(absent["tools"], "absent");
        assert_eq!(absent["tool_choice"], "absent");
        let function = request_action(&json!({"tools":[],"tool_choice":{"type":"function","name":"DO_NOT_LOG"}}));
        assert_eq!(function["tools"], "count:0");
        assert_eq!(function["tool_choice"], "function");
        assert!(!format!("{function:?}").contains("DO_NOT_LOG"));
    }
    #[test]
    fn telemetry_tracks_the_responses_field_that_actually_changes() {
        let mut params=json!({"reasoning":{"effort":"low","summary":"auto"}});
        let before=request_action(&params);
        let changes=crate::core::jev_active::apply_decision(&mut params,"complexity","high");
        let after=request_action(&params);
        assert_eq!(before[&changes[0].key],"low");
        assert_eq!(after[&changes[0].key],"medium");
        assert!(!after.contains_key("reasoning_effort"));
    }

    fn tool_end_with_reports(tool_name: &str, reports: Value, is_error: bool) -> ExtensionEvent {
        ExtensionEvent::ToolExecutionEnd(ToolExecutionEndPayload {
            tool_call_id: "call-1".to_string(),
            tool_name: tool_name.to_string(),
            result: json!({
                "content": [{ "type": "text", "text": "done" }],
                "details": { "executionReports": reports },
                "isError": is_error,
            }),
            is_error,
        })
    }

    fn designated_report(label: &str, exit_code: i64, expected: Value, is_error: bool) -> Value {
        json!([{
            "schema": "optimus.script-result.v1",
            "stage": "process",
            "scriptId": "s1",
            "exitCode": exit_code,
            "durationSeconds": 0.1,
            "expectedExitCodes": expected,
            "isError": is_error,
            "receipt": { "verification": { "kind": "task_check", "label": label } },
        }])
    }

    #[test]
    fn ctrl001_designated_task_check_records_real_verification_evidence() {
        use pi_jev::observation::VerificationEvidence;
        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        // A designated, non-failed report records Passed evidence.
        core.note_observation("sess-verify", &tool_end_with_reports("ipython",
            designated_report("focused scope: status check", 0, json!([0]), false), false));
        assert_eq!(core.observation("sess-verify").verification, VerificationEvidence::Passed);
        // A designated failed report records Failed evidence.
        core.note_observation("sess-verify", &tool_end_with_reports("ipython",
            designated_report("focused scope: status check", 1, json!([0]), true), true));
        assert_eq!(core.observation("sess-verify").verification, VerificationEvidence::Failed);
        // A failed designated check is sticky for the task epoch: a later
        // unrelated passed check can never flip the evidence to Passed.
        core.note_observation("sess-verify", &tool_end_with_reports("ipython",
            designated_report("unrelated later check", 0, json!([0]), false), false));
        assert_eq!(core.observation("sess-verify").verification, VerificationEvidence::Failed,
            "a later passed check must not un-fail the epoch");
        // A new real-user task epoch resets the scope: the sticky failure and
        // the trace evidence do not bleed into the new task, and a fresh
        // designated passed check is recorded truthfully.
        core.note_control_epoch("sess-verify", "sess-verify:2");
        assert_eq!(core.observation("sess-verify").verification, VerificationEvidence::Unknown);
        core.note_observation("sess-verify", &tool_end_with_reports("ipython",
            designated_report("new task check", 0, json!([0]), false), false));
        assert_eq!(core.observation("sess-verify").verification, VerificationEvidence::Passed);
    }

    #[test]
    fn ctrl001_undesignated_reports_and_transport_success_never_become_verification() {
        use pi_jev::observation::VerificationEvidence;
        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        // Plain successful reports (no designation) stay Unknown.
        core.note_observation("sess-plain", &tool_end_with_reports("ipython", json!([{
            "schema": "optimus.script-result.v1", "stage": "process", "scriptId": "s1",
            "exitCode": 0, "durationSeconds": 0.1, "expectedExitCodes": [0],
            "isError": false, "receipt": null,
        }]), false));
        assert_eq!(core.observation("sess-plain").verification, VerificationEvidence::Unknown);
        // Tool transport success without reports is not verification either.
        core.note_observation("sess-plain", &tool_end_with_reports("ipython", Value::Null, false));
        assert_eq!(core.observation("sess-plain").verification, VerificationEvidence::Unknown);
        // A malformed designation invalidates the whole report: no evidence.
        core.note_observation("sess-plain", &tool_end_with_reports("ipython", json!([{
            "schema": "optimus.script-result.v1", "stage": "process", "scriptId": "s1",
            "exitCode": 0, "durationSeconds": 0.1, "expectedExitCodes": [0],
            "isError": false, "receipt": { "verification": { "kind": "assert", "label": "fake" } },
        }]), false));
        assert_eq!(core.observation("sess-plain").verification, VerificationEvidence::Unknown);
        // Provenance gate: executionReports-shaped details from any tool
        // other than the native ipython producer never become verification
        // evidence, designated or not.
        core.note_observation("sess-plain", &tool_end_with_reports("bash",
            designated_report("echoed by another tool", 0, json!([0]), false), false));
        assert_eq!(core.observation("sess-plain").verification, VerificationEvidence::Unknown,
            "schema shape alone is not host-measured provenance");
    }

    #[test]
    fn tool001_diagnostics_report_last_advertisement_with_staleness() {
        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        // No observation yet: absent, never a fabricated healthy state.
        assert!(core.tool_diagnostics("sess-diag").is_null());
        core.note_turn("sess-diag", 5);
        core.note_advertised_tools("sess-diag", 5, &["ipython".to_string(), "bash".to_string()]);
        core.note_observation("sess-diag", &tool_end_with_reports("ipython", Value::Null, false));
        let fresh = core.tool_diagnostics("sess-diag");
        assert_eq!(fresh["lastAdvertisedTools"], json!(["ipython", "bash"]));
        assert_eq!(fresh["advertisedAtTurn"], json!(5));
        assert_eq!(fresh["isCurrentTurn"], json!(true));
        assert_eq!(fresh["recentRequestsRetainingTools"], json!("1/1"));
        assert_eq!(fresh["lastToolResult"], json!("ok"));
        assert_eq!(fresh["turnsSinceLastToolResult"], json!(0));
        assert_eq!(fresh["verificationEvidence"], json!("unknown"));
        // A later turn makes the SAME observation explicitly stale; the
        // list is never relabelled as current availability.
        core.note_turn("sess-diag", 9);
        let stale = core.tool_diagnostics("sess-diag");
        assert_eq!(stale["isCurrentTurn"], json!(false));
        assert_eq!(stale["currentTurn"], json!(9));
        assert_eq!(stale["turnsSinceLastToolResult"], json!(4));
        assert_eq!(core.advertised_tools_fact("sess-diag").unwrap(),
            "advertised_tools_last_request=ipython,bash (stale, from an earlier request)");
        // Only bounded names ever leave bookkeeping: no command bodies.
        assert!(!format!("{stale:?}").contains("secret"));
    }

    #[test]
    fn tool001_fact_and_event_state_carry_only_public_tool_names() {
        let core = Arc::new(JevBridgeCore::new(JevSettings::default()));
        core.note_turn("sess-fact", 3);
        core.note_advertised_tools("sess-fact", 3, &["ipython".to_string()]);
        assert_eq!(core.advertised_tools_fact("sess-fact").unwrap(),
            "advertised_tools_last_request=ipython");
        assert_eq!(core.last_advertised_tools("sess-fact"), vec!["ipython".to_string()]);
        assert!(core.advertised_tools_is_current("sess-fact"));
        assert!(core.advertised_tools_fact("sess-fact-none").is_none());
        // Oversized or duplicate names are dropped/bounded, never a flood.
        let long_name = "x".repeat(129);
        let names = advertised_tool_names(&json!({"tools": [
            {"type":"function","name":"ipython"},
            {"type":"function","name":"ipython"},
            {"type":"function","name": long_name},
        ]}));
        assert_eq!(names, vec!["ipython".to_string()]);
    }
}

#[cfg(test)]
#[path = "jev_bridge/tool_error_bridge_tests.rs"]
mod tool_error_bridge_tests;
