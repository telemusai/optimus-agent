//! Jev command surface + mode store bridge for the interactive UI.
//!
//! This module owns three things and nothing else:
//!
//! 1. `/jev` argument parsing (`/jev`, `/jev off`, `/jev compare`, `/jev active`,
//!    `/jev on`, `/jev status`, `/jev key`, `/jev help`).
//! 2. The mode get/set bridge into the `pi-jev` config store, including scope
//!    reporting (explicit session overrides vs global defaults).
//! 3. The truthful status text the UI renders for `/jev status`.
//!
//! DESIGN.md binding constraints this module implements:
//!
//! * section 0 / 10.2 - `/jev` is the only mode control; key presence NEVER
//!   enables Jev; explicit per-session mode > explicit global default >
//!   built-in `Off`.
//! * section 11 - NO PRIMARY-model control: the user's primary model,
//!   provider and effort stay authoritative. ROOT-CONTRACT v9 adds exactly
//!   one bounded, operator-initiated Jev-model surface here: the requested
//!   JEV SystemOne request model (`/jev model status|set <id>|reset`, agent
//!   dir-persistent, default `jev-latest`, zero network) and the explicit
//!   `/jev models` catalog query (the only networked model command). Nothing
//!   here reads or writes the primary chat model, provider or effort, and
//!   nothing selects a model automatically.
//! * section 12 - NO subagent control: this module has no spawn/delete/cancel/
//!   task/message/budget surface. Category 5/6 assessments are advisory
//!   records in Compare and are never converted into commands.

use std::path::{Path, PathBuf};

use pi_tui::keybindings::get_keybindings;

use pi_jev::config::{
    resolve_credential_source, CredentialSource, EnvKeyPresence, JevFeature, JevFeatures, JevSettings, JevSettingsStore,
    ModeScope, DEFAULT_KEY_ID, ENV_JEV_API_KEY, ENV_TYPESAFE_API_KEY,
};
use pi_jev::credential::{CredentialStore, SecretString};
use pi_jev::error::JevError;
use pi_jev::types::JevMode;

/// Canonical builtin command name (added to `core/slash_commands.rs`).
pub const JEV_COMMAND_NAME: &str = "jev";

/// Autocomplete argument hint. `on` is the short form of `compare`; only the
/// explicit `active` spelling selects the mode that changes a request.
pub const JEV_ARGUMENT_HINT: &str =
    "[off|compare|active|compare-active|on|compact|feature|default|full-jev|status|key|models|model]";

/// Autocomplete description. It names the modes, the key entry and status.
pub const JEV_COMMAND_DESCRIPTION: &str = "Jev System One: Off, Compare, Active, Compare + Active, compaction, feature gates, the global full-jev overlay, API key, the requested Jev model (model status/set/reset) and the model catalog";

/// The one line `/jev on` adds after the Compare confirmation, so the shorthand
/// cannot be mistaken for the request-changing mode.
pub const JEV_ON_COMPARE_NOTICE: &str = "`on` selects Compare: answers are recorded, never applied. Use `/jev active` to let an accepted answer change the next provider request.";

/// True when the typed argument was the `on` shorthand, which selects Compare and
/// must say so. No key or case is hardcoded.
pub fn is_on_shorthand(args: &str) -> bool {
    args.trim().eq_ignore_ascii_case("on")
}

/// Why an Active session can show no counters at all.
pub const JEV_ACTIVE_UNKNOWN_NOTE: &str =
    "no Active boundary has been observed in this worker yet; unknown is not zero";

/// The exact notice an `active` mode change prints. Active is a real, operative
/// mode in this release; `/jev on` stays Compare and says so instead.
pub const JEV_ACTIVE_NOTICE: &str = "Jev Active applies accepted, feature-gated decisions at bounded native boundaries. Tool requirement and complexity are enabled by default; optional tool/retrieval filtering, observers and compaction need separate opt-in. A refused answer, failure or timeout keeps the baseline unchanged.";

/// Documented SystemOne endpoint (DESIGN.md section 2). Recorded for display only:
/// this UI lane never calls it.
pub const JEV_API_ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";

/// Documented default response model (DESIGN.md section 2).
pub const JEV_DEFAULT_MODEL: &str = "jev-latest";

/// The first-use disclosure wording (DESIGN.md data-handling requirement).
pub const JEV_DISCLOSURE_NOTICE: &str = "Disclosure: Jev decisions and independently enabled compaction send selected, bounded prompt/context excerpts to TypeSafe. Pattern-based redaction cannot find every confidential item. Compare records recommendations without applying them. Active applies only accepted decisions allowed by local feature gates. Combined mode records both outcomes from one boundary request.";

pub const JEV_BOUNDARY_NOTICE: &str = "Jev never controls the primary model, provider, permissions, subagents, agent messages, depth, concurrency or budgets. It never deletes durable memory or transcript history. Compaction is request-local and separately controlled; continuation and verification remain advisory.";

/// What `/jev full-jev` (and its explicit `on` spelling) print on install.
/// Bare full-jev is explicit consent (ROOT-CONTRACT v1): no extra modal.
pub const JEV_FULL_JEV_ON_NOTICE: &str = "Full-jev is now active globally for this agent dir: mode Compare + Active, every feature gate on (including candidate reranking and line-level semantic find), and independent request-local compaction on. The overlay resolves ABOVE every saved per-session, global and inherited override; those saved values stay on disk unchanged and resolve again the moment you run /jev full-jev off.";

/// What `/jev full-jev off` prints when the overlay was removed.
pub const JEV_FULL_JEV_OFF_NOTICE: &str = "Full-jev is now off. The overlay was removed without touching any saved setting, so every chat resolves from its own saved values again.";

/// What `/jev full-jev off` prints when nothing was active: a truthful
/// no-change message, never a silent success.
pub const JEV_FULL_JEV_ALREADY_OFF_NOTICE: &str = "Full-jev was not active; nothing was changed.";

/// The rejection every conflicting mode/feature/compaction/default change
/// gets while the overlay is active (ROOT-CONTRACT v1): an explicit
/// no-change message with the recovery path, never a hidden success.
pub const JEV_FULL_JEV_REJECTION: &str = "Full-jev is active, so this saved setting is currently masked by the global overlay and nothing was changed. Turn the overlay off first with /jev full-jev off, then set per-session values again. (/jev key, /jev status and /jev full-jev status still work.)";

/// What `/jev off` prints when it acts as the emergency exit while full-jev
/// is active (ROOT-CONTRACT v1).
pub const JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE: &str = "Emergency exit: full-jev was active, so /jev off disabled the global overlay and set THIS chat to decisions Off and compaction off in one atomic write. Other chats now resolve from their own saved settings again. Use /jev compact on to re-enable compaction for this chat; the overlay stays off until /jev full-jev.";

/// One parsed `/jev` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JevRequest {
    /// `/jev` with no argument: open the menu.
    Menu,
    /// `/jev off`, `/jev compare`, `/jev on`, `/jev active`.
    SetMode(JevMode),
    SetDefaultMode(JevMode),
    SetFeature(JevFeature, bool),
    SetCompaction(bool),
    SetDefaultCompaction(bool),
    CompactionStatus,
    /// `/jev status`.
    Status,
    /// `/jev full-jev` / `/jev full-jev on`: install the global overlay.
    /// `/jev full-jev off`: remove it.
    SetFullJev(bool),
    /// `/jev full-jev status`: the overlay's local-only truth panel.
    FullJevStatus,
    /// `/jev key`: masked credential entry.
    InputKey,
    /// `/jev key clear`: delete the stored credential.
    ClearKey,
    /// `/jev models`: EXPLICIT operator-initiated read-only catalog query —
    /// the ONLY `/jev` command that may touch the network (one bounded
    /// single-attempt `GET /v1/models`). Never automatic, never a selection,
    /// never a settings write.
    Models,
    /// `/jev model` / `/jev model status`: the requested-model panel. Local
    /// settings truth plus the in-process comparison snapshot when this
    /// process holds it; no RPC, no catalog, no network.
    ModelStatus,
    /// `/jev model set <id>`: durable requested-Jev-model write. Local only;
    /// validates the exact identifier; no probe, no availability claim, no
    /// budget effect, no overlay interaction.
    ModelSet(String),
    /// `/jev model reset`: remove the explicit selection; the native default
    /// `jev-latest` governs again. Local only.
    ModelReset,
    /// `/jev help`.
    Help,
    /// An unrecognised argument, with the usage text to show.
    Unknown(String),
}

/// Usage line for unknown arguments. Built from the same constants the registry
/// quotes, so the two cannot drift.
pub fn jev_usage() -> String {
    format!(
        "Usage: /{JEV_COMMAND_NAME} {JEV_ARGUMENT_HINT}  (or /{JEV_COMMAND_NAME} for the menu)"
    )
}

/// Parse the `/jev` argument text.
///
/// `on` stays the short alias for `Compare` (the repo rule: no shorthand arms a
/// request-changing mode), and the caller must explain the difference with
/// [`JEV_ON_COMPARE_NOTICE`]. Only the explicit `active` spelling selects
/// [`JevMode::Active`] and prints [`JEV_ACTIVE_NOTICE`]. No form is silently
/// rewritten to another mode.
pub fn parse_jev_request(args: &str) -> JevRequest {
    // ROOT-CONTRACT v9: the requested model id is an EXACT identifier, so the
    // `model set` arm never case-folds, trims or otherwise normalizes it. Only
    // leading command whitespace is ignored. Everything after the literal
    // `model set ` delimiter is passed byte-for-byte to the safe-id validator;
    // any whitespace in the id is refused without echo.
    // `to_ascii_lowercase` is byte-preserving, so this byte arithmetic is exact.
    let command = args.trim_start();
    let lowered_command = command.to_ascii_lowercase();
    if lowered_command == "model set" {
        return JevRequest::ModelSet(String::new());
    }
    if lowered_command.starts_with("model set ") {
        let id = &command["model set ".len()..];
        return JevRequest::ModelSet(id.to_string());
    }
    let normalized = args.trim().to_ascii_lowercase();
    let parts: Vec<&str> = normalized.split(' ').filter(|part| !part.is_empty()).collect();
    let toggle = |value: &str| match value { "on" => Some(true), "off" => Some(false), _ => None };
    match parts.as_slice() {
        [] => JevRequest::Menu,
        ["off"] => JevRequest::SetMode(JevMode::Off),
        ["compare" | "on"] => JevRequest::SetMode(JevMode::Compare),
        ["active"] => JevRequest::SetMode(JevMode::Active),
        ["compare-active" | "compare-and-active" | "compare_and_active" | "both"] => JevRequest::SetMode(JevMode::CompareAndActive),
        ["status"] => JevRequest::Status,
        ["compact" | "compaction"] | ["compact" | "compaction", "status"] => JevRequest::CompactionStatus,
        ["compact" | "compaction", value] if toggle(value).is_some() => JevRequest::SetCompaction(toggle(value).unwrap()),
        ["default", "compact" | "compaction", value] if toggle(value).is_some() => JevRequest::SetDefaultCompaction(toggle(value).unwrap()),
        ["default", mode] if JevMode::parse(mode).is_some() => JevRequest::SetDefaultMode(JevMode::parse(mode).unwrap()),
        ["feature", feature, value] if JevFeature::parse(feature).is_some() && toggle(value).is_some() => JevRequest::SetFeature(JevFeature::parse(feature).unwrap(), toggle(value).unwrap()),
        // Bare `full-jev` means ON, exactly like `on` means Compare: no silent
        // shorthand arms a request-changing state (ROOT-CONTRACT v1).
        ["full-jev" | "fulljev" | "full_jev"] => JevRequest::SetFullJev(true),
        ["full-jev" | "fulljev" | "full_jev", "on"] => JevRequest::SetFullJev(true),
        ["full-jev" | "fulljev" | "full_jev", "off"] => JevRequest::SetFullJev(false),
        ["full-jev" | "fulljev" | "full_jev", "status"] => JevRequest::FullJevStatus,
        ["models"] => JevRequest::Models,
        ["model"] | ["model", "status"] => JevRequest::ModelStatus,
        ["model", "reset"] => JevRequest::ModelReset,
        ["key" | "key-input"] => JevRequest::InputKey,
        ["key", "clear"] | ["key-clear" | "clear-key"] => JevRequest::ClearKey,
        ["help" | "-h" | "--help"] => JevRequest::Help,
        _ => JevRequest::Unknown(normalized),
    }
}

/// Environment presence read through a caller-supplied accessor, so tests never
/// need a real environment. Absent/empty counts as not present.
pub fn env_presence(read_env: &dyn Fn(&str) -> Option<String>) -> EnvKeyPresence {
    let present = |name: &str| read_env(name).is_some_and(|value| !value.trim().is_empty());
    EnvKeyPresence {
        typesafe_api_key: present(ENV_TYPESAFE_API_KEY),
        jev_api_key: present(ENV_JEV_API_KEY),
    }
}

/// Presence from the PROCESS environment, through the same accessor shape the
/// tests use. One path for both, so the tested one is the shipped one.
pub fn process_env_presence() -> EnvKeyPresence {
    env_presence(&|name| std::env::var(name).ok())
}

/// Credential state the status panel reports: presence and source only, never
/// the secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CredentialStatus {
    pub source: CredentialSource,
    pub saved_present: bool,
    pub env: EnvKeyPresence,
}

impl CredentialStatus {
    pub fn resolve(saved_present: bool, env_typesafe: bool, env_jev: bool) -> Self {
        let env = EnvKeyPresence {
            typesafe_api_key: env_typesafe,
            jev_api_key: env_jev,
        };
        Self {
            source: resolve_credential_source(saved_present, env_typesafe, env_jev),
            saved_present,
            env,
        }
    }

    /// True when BOTH environment variables are present, which status must report
    /// because the primary one wins and the alias is ignored.
    pub fn env_conflict(&self) -> bool {
        self.env.has_conflict()
    }

    /// Wording only; never a secret value.
    pub fn describe(&self) -> String {
        let mut text = format!(
            "Credential: {} ({})",
            self.source.as_str(),
            if self.present() { "present" } else { "absent" }
        );
        if self.env_conflict() {
            if self.source == CredentialSource::Saved {
                text.push_str(&format!(
                    ". A saved credential exists, so {ENV_TYPESAFE_API_KEY} and {ENV_JEV_API_KEY} are ignored"
                ));
            } else {
                text.push_str(&format!(
                    ". Both {ENV_TYPESAFE_API_KEY} and {ENV_JEV_API_KEY} are set; {ENV_TYPESAFE_API_KEY} wins"
                ));
            }
        }
        text
    }

    pub fn present(&self) -> bool {
        self.source.is_configured()
    }
}

/// Client-visible health of the in-flight comparison pipeline.
///
/// `/jev status` must distinguish these states instead of inferring health from
/// key presence (DESIGN.md section 0).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JevPipelineStatus {
    /// True only for a worker-sourced snapshot; absent telemetry is unknown.
    pub observed: bool,
    /// A request/response cycle is currently scheduled or in flight.
    pub in_flight: u32,
    /// Bounded queue depth and capacity.
    pub queue_depth: u32,
    pub queue_capacity: u32,
    /// Wall clock of the last successful call, RFC3339, plus its latency.
    pub last_success_at: Option<String>,
    pub last_latency_ms: Option<u64>,
    pub success_count: u64,
    pub failure_count: u64,
    /// Comparisons dropped because the bounded queue was full.
    pub dropped_comparisons: u64,
    /// Categories skipped, with their explicit reason.
    pub skipped_categories: Vec<(String, String)>,
    /// Redacted fallback reason; empty when there was no fallback.
    pub fallback_reason: String,
    /// Response model the server reported (config drift is visible).
    pub response_model: Option<String>,
    /// Real Active-mode counters for this session, present only when the worker
    /// observed an Active provider boundary. Absent in Off and Compare, so an
    /// absent block is unknown and never rendered as zero.
    pub active: Option<ActiveCounters>,
    pub compaction: Option<CompactionStatus>,
}

/// Latest request-local compaction metadata. Missing fields remain unknown.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionStatus {
    pub applied: Option<bool>,
    pub estimated_tokens_before: Option<u64>,
    pub estimated_tokens_after: Option<u64>,
    pub calls_evaluated: Option<u64>,
    pub calls_removed: Option<u64>,
    pub results_removed: Option<u64>,
    pub results_truncated: Option<u64>,
    pub reduction_percent: Option<String>,
    pub fallback_reason: Option<String>,
    pub breaker_state: Option<String>,
}

impl CompactionStatus {
    pub fn from_snapshot(value: Option<&serde_json::Value>) -> Option<Self> {
        let value = value.filter(|value| value.is_object())?;
        let text = |key: &str| value.get(key).and_then(serde_json::Value::as_str)
            .map(|text| pi_jev::correlate::sanitize_text(text, 120));
        let applied = value.get("applied").and_then(serde_json::Value::as_bool);
        let fallback_reason = text("fallback_reason");
        if applied.is_none() && fallback_reason.is_none() { return None; }
        let number = |key: &str| value.get(key).and_then(serde_json::Value::as_u64);
        Some(Self {
            applied,
            estimated_tokens_before: number("estimated_tokens_before"),
            estimated_tokens_after: number("estimated_tokens_after"),
            calls_evaluated: number("calls_evaluated"),
            calls_removed: number("calls_removed"),
            results_removed: number("results_removed"),
            results_truncated: number("results_truncated"),
            reduction_percent: value.get("reduction_ratio").and_then(serde_json::Value::as_f64)
                .filter(|ratio| ratio.is_finite() && (0.0..=1.0).contains(ratio))
                .map(|ratio| format!("{:.1}%", ratio * 100.0)),
            fallback_reason,
            breaker_state: text("breaker_state"),
        })
    }
}

pub fn render_compaction_observation(status: Option<&CompactionStatus>) -> String {
    let Some(status) = status else {
        return "Compaction last: unknown (no worker boundary observed)\nCompaction breaker: unknown\n".into();
    };
    let count = |value: Option<u64>| value.map(|value| value.to_string()).unwrap_or_else(|| "unknown".into());
    format!("Compaction last: {}\nCompaction estimated tokens: {} -> {} (reduction {})\nCompaction candidates: {} evaluated, {} calls removed, {} results removed, {} results truncated\nCompaction fallback: {}\nCompaction breaker: {}\n",
        match status.applied { Some(true) => "applied", Some(false) => "not applied", None => "unknown" },
        count(status.estimated_tokens_before), count(status.estimated_tokens_after),
        status.reduction_percent.as_deref().unwrap_or("unknown"),
        count(status.calls_evaluated), count(status.calls_removed), count(status.results_removed), count(status.results_truncated),
        status.fallback_reason.as_deref().unwrap_or("none reported"),
        status.breaker_state.as_deref().unwrap_or("unknown"))
}

/// The `active` block of one worker snapshot.
///
/// `applied` counts provider-request boundaries where a request field actually
/// changed (not answers), `accepted_no_effect` counts accepted answers with no
/// reversible effect, `refused` counts refused answers, and `unavailable`
/// counts boundaries with no usable answer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActiveCounters {
    pub applied: u64,
    pub accepted_no_effect: u64,
    pub refused: u64,
    pub unavailable: u64,
    pub last_reason: Option<String>,
    pub last_category: Option<String>,
}

impl ActiveCounters {
    /// Parse the block when the snapshot carries one. An absent or malformed
    /// block is `None`: "no Active boundary seen" is not the same as all-zero.
    pub fn from_snapshot(value: Option<&serde_json::Value>) -> Option<Self> {
        let value = value.filter(|value| value.is_object())?;
        let number = |key: &str| value.get(key).and_then(serde_json::Value::as_u64);
        let text = |key: &str| value.get(key).and_then(serde_json::Value::as_str)
            .map(|text| pi_jev::correlate::sanitize_text(text, 120));
        Some(Self {
            applied: number("applied")?,
            accepted_no_effect: number("accepted_no_effect")?,
            refused: number("refused")?,
            unavailable: number("unavailable")?,
            last_reason: text("last_reason"),
            last_category: text("last_category"),
        })
    }
}

impl JevPipelineStatus {
    /// Accept either snapshot shape: the Compare/scheduler counters, the Active
    /// counters, or both. A snapshot with neither is unknown.
    pub fn from_snapshot(snapshot: Option<&serde_json::Value>) -> Self {
        let Some(value) = snapshot else { return Self::default(); };
        let active = ActiveCounters::from_snapshot(value.get("active"));
        let compaction = CompactionStatus::from_snapshot(value.get("compaction"));
        let compare_counters = ["success_count", "failure_count", "queue_capacity"]
            .iter().all(|key| value.get(key).and_then(serde_json::Value::as_u64).is_some());
        if !compare_counters {
            // An Active-only snapshot has no scheduler counters. They stay at
            // their default (unknown) values instead of being invented as zero.
            return Self { active, compaction, ..Self::default() };
        }
        let number = |key: &str| value.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0);
        let optional_text = |key: &str| value.get(key).and_then(serde_json::Value::as_str)
            .map(|text| pi_jev::correlate::sanitize_text(text, 120));
        Self {
            observed: true,
            in_flight: number("in_flight").min(u32::MAX as u64) as u32,
            queue_depth: number("queue_depth").min(u32::MAX as u64) as u32,
            queue_capacity: number("queue_capacity").min(u32::MAX as u64) as u32,
            last_success_at: value.get("last_success_ms").and_then(serde_json::Value::as_u64)
                .and_then(|ms| i64::try_from(ms).ok()).and_then(chrono::DateTime::from_timestamp_millis)
                .map(|time| time.to_rfc3339()),
            last_latency_ms: value.get("last_latency_ms").and_then(serde_json::Value::as_u64),
            success_count: number("success_count"), failure_count: number("failure_count"),
            dropped_comparisons: number("dropped_comparisons"),
            skipped_categories: value.get("skipped_categories").and_then(serde_json::Value::as_object)
                .map(|categories| categories.iter().filter_map(|(category, reason)| reason.as_str()
                    .map(|reason| (pi_jev::correlate::sanitize_text(category, 80), pi_jev::correlate::sanitize_text(reason, 120))))
                    .take(13).collect()).unwrap_or_default(),
            fallback_reason: optional_text("fallback_reason").unwrap_or_default(),
            response_model: optional_text("response_model"),
            active,
            compaction,
        }
    }

    /// True when the snapshot carried the Compare/scheduler counters. The zeroes
    /// of an Active-only snapshot are NOT counter facts and must not be rendered
    /// as one.
    pub fn counters_known(&self) -> bool {
        self.observed || self.in_flight > 0 || self.queue_capacity > 0
            || self.last_success_at.is_some() || self.success_count > 0 || self.failure_count > 0
    }

    /// True when anything at all was observed for this session, in either mode.
    pub fn known(&self) -> bool {
        self.counters_known() || self.active.is_some() || self.compaction.is_some()
    }
    /// In-flight work makes the footer amber/checking rather than green.
    pub fn checking(&self) -> bool {
        self.in_flight > 0
    }

    pub fn degraded(&self) -> bool {
        !self.fallback_reason.is_empty() || (self.failure_count > 0 && self.last_success_at.is_none())
    }
}

/// Everything `/jev status` needs. Assembled by the caller from the store plus
/// whatever the comparison pipeline exposes; no field requires a network call.
#[derive(Debug, Clone, PartialEq)]
pub struct JevStatusReport {
    pub mode: JevMode,
    pub scope: ModeScope,
    pub features: JevFeatures,
    pub compaction_enabled: bool,
    pub compaction_scope: ModeScope,
    pub compaction_config: pi_jev::compaction::CompactionConfig,
    pub credential: CredentialStatus,
    pub pipeline: JevPipelineStatus,
    /// API/model identity.
    pub api_endpoint: String,
    pub requested_model: String,
    /// Compare keeps this at 0: nothing is applied. In Active the live figure is
    /// [`JevPipelineStatus::active`]`::applied`, which counts boundaries where a
    /// request field actually changed.
    pub applied_decisions: u64,
    pub hypothetical_only: bool,
    /// Where the pipeline figures came from, or why they are empty. Kept in the
    /// report so an empty counter set is never read as "all healthy".
    pub pipeline_note: Option<String>,
    /// True while the global full-jev overlay is active, so the panel can say
    /// so instead of letting the scope line carry it alone.
    pub full_jev_active: bool,
}

impl JevStatusReport {
    /// No worker telemetry means unknown counters, not measured zeroes.
    pub fn local_only(mode: JevMode, scope: ModeScope, credential: CredentialStatus) -> Self {
        Self {
            mode,
            scope,
            features: JevFeatures::default(),
            compaction_enabled: false,
            compaction_scope: ModeScope::BuiltIn,
            compaction_config: pi_jev::compaction::CompactionConfig::default(),
            credential,
            pipeline: JevPipelineStatus::default(),
            api_endpoint: JEV_API_ENDPOINT.to_string(),
            requested_model: JEV_DEFAULT_MODEL.to_string(),
            applied_decisions: 0,
            hypothetical_only: true,
            pipeline_note: Some(
                "Worker telemetry is unavailable or no comparison has been observed in this worker yet. Unknown is not zero."
                    .to_string(),
            ),
            full_jev_active: false,
        }
    }

    pub fn with_settings(mut self, settings: &JevSettings, session_id: &str) -> Self {
        self.features = settings.effective_features(session_id);
        self.compaction_config = settings.compaction.clone();
        (self.compaction_enabled, self.compaction_scope) = settings.effective_compaction_with_scope(session_id);
        self.full_jev_active = settings.full_jev_active();
        // ROOT-CONTRACT v9: the requested model is the persisted explicit
        // selection or the native default, resolved from the SAME settings
        // snapshot the rest of this panel reads. Local only.
        self.requested_model = settings.requested_model_or_default().to_string();
        self
    }

    pub fn with_snapshot(mut self, snapshot: Option<&serde_json::Value>) -> Self {
        self.pipeline = JevPipelineStatus::from_snapshot(snapshot);
        if self.pipeline.observed { self.pipeline_note = Some("Live session-worker comparison telemetry".into()); }
        self
    }
}

/// Render the status panel. Pure: same input, same text, no I/O and no polling.
pub fn render_status(report: &JevStatusReport) -> String {
    let mut text = String::new();
    text.push_str("Jev Status\n\n");
    text.push_str(&format!("Mode: {}\n", mode_label(report.mode)));
    text.push_str(&format!(
        "Scope: {} ({})\n",
        report.scope.as_str(),
        match report.scope {
            ModeScope::Session => "an explicit per-session setting",
            ModeScope::GlobalDefault => "no per-session setting; this is the global default",
            ModeScope::BuiltIn => "no per-session setting and no global default; this is the built-in default",
            ModeScope::FullJevOverlay =>
                "the global full-jev overlay; it resolves above every saved setting",
        }
    ));
    if report.full_jev_active {
        text.push_str("Full-jev overlay: active; run /jev full-jev status for the overlay panel\n");
    }
    if report.mode.allows_active() {
        text.push_str(&format!("{JEV_ACTIVE_NOTICE}\n"));
    }
    text.push_str(&render_compaction_status(report.compaction_enabled, report.compaction_scope));
    text.push_str(&format!("Compaction policy: keep_threshold={}, preserve_recent_messages={}, max_state_tokens={}, max_request_tokens={}, truncate_head_chars={}, minimum_reduction_ratio={}\n",
        report.compaction_config.keep_threshold, report.compaction_config.preserve_recent_messages,
        report.compaction_config.max_state_tokens, report.compaction_config.max_request_tokens,
        report.compaction_config.truncate_head_chars, report.compaction_config.minimum_reduction_ratio));
    text.push_str("Feature gates (configured; not proof of an applied decision):\n");
    for feature in JevFeature::ALL {
        text.push_str(&format!("  {}: {}\n", feature.as_str(), if report.features.enabled(feature) { "on" } else { "off" }));
    }
    text.push_str(&format!("{}\n", report.credential.describe()));
    text.push_str(&format!("API: {}\n", report.api_endpoint));
    text.push_str(&format!("Model requested: {}\n", report.requested_model));
    text.push_str(&format!(
        "Model reported by server: {}\n",
        report
            .pipeline
            .response_model
            .clone()
            .unwrap_or_else(|| "none yet (no successful call)".to_string())
    ));
    text.push_str(&format!("Configured: {}\n", if report.credential.present() { "yes" } else { "no credential" }));
    text.push_str(&format!(
        "State: {}\n",
        if report.mode == JevMode::Off && report.compaction_enabled {
            "decision mode off (request-local compaction independently enabled)"
        } else if report.mode == JevMode::Off {
            "idle (Off: no scheduling, no client, no network)"
        } else if report.mode.allows_active() && !report.pipeline.known() {
            "unknown (no Active boundary observed in this worker yet)"
        } else if report.mode.allows_active() {
            "active (accepted decisions remain feature-gated and bounded)"
        } else if !report.pipeline.known() {
            "unknown (worker telemetry unavailable or no observation yet)"
        } else if report.pipeline.checking() {
            "checking (comparisons in flight)"
        } else if report.pipeline.degraded() {
            "degraded (a fallback or failure was recorded)"
        } else if report.pipeline.last_success_at.is_some() {
            "last-success recorded"
        } else {
            "no successful call yet"
        }
    ));
    text.push_str(&format!(
        "Last success: {} (latency {})\n",
        report.pipeline.last_success_at.clone().unwrap_or_else(|| if report.pipeline.counters_known() { "never" } else { "unknown" }.to_string()),
        report
            .pipeline
            .last_latency_ms
            .map(|ms| format!("{ms} ms"))
            .unwrap_or_else(|| "unknown".to_string())
    ));
    if report.pipeline.counters_known() {
    text.push_str(&format!(
        "Counters: {} ok, {} failed\n",
        report.pipeline.success_count, report.pipeline.failure_count
    ));
    text.push_str(&format!(
        "Queue: {}/{} (in flight {})\n",
        report.pipeline.queue_depth, report.pipeline.queue_capacity, report.pipeline.in_flight
    ));
    text.push_str(&format!(
        "Dropped comparisons: {}\n",
        report.pipeline.dropped_comparisons
    ));
    } else {
        text.push_str("Counters: unknown\nQueue: unknown\nDropped comparisons: unknown\n");
    }
    if report.pipeline.skipped_categories.is_empty() {
        text.push_str("Skipped categories: none recorded\n");
    } else {
        text.push_str("Skipped categories:\n");
        for (category, reason) in &report.pipeline.skipped_categories {
            text.push_str(&format!("  - {category}: {reason}\n"));
        }
    }
    text.push_str(&format!(
        "Fallback reason: {}\n",
        if report.pipeline.fallback_reason.is_empty() {
            "none".to_string()
        } else {
            report.pipeline.fallback_reason.clone()
        }
    ));
    if let Some(note) = &report.pipeline_note {
        text.push_str(&format!("Pipeline source: {note}\n"));
    }
    text.push_str(&format!(
        "Decisions applied: {} ({})\n",
        if report.mode.allows_active() {
            report.pipeline.active.as_ref().map(|counters| counters.applied.to_string()).unwrap_or_else(|| "unknown".into())
        } else {
            report.applied_decisions.to_string()
        },
        if report.mode.allows_active() {
            "Active counts boundaries with an actual applied effect"
        } else if report.mode.allows_compare() {
            "Compare is shadow-only; potential savings are hypothetical, never measured"
        } else {
            "nothing is applied while Jev is Off"
        }
    ));
    if let Some(counters) = report.pipeline.active.as_ref() {
        text.push_str(&format!(
            "Active boundaries: {} applied, {} accepted with no effect, {} refused, {} without a usable answer\n",
            counters.applied, counters.accepted_no_effect, counters.refused, counters.unavailable
        ));
        text.push_str(&format!(
            "Active last: {} ({})\n",
            counters.last_category.clone().unwrap_or_else(|| "none".to_string()),
            counters.last_reason.clone().unwrap_or_else(|| "none".to_string())
        ));
    } else if report.mode.allows_active() {
        text.push_str(&format!(
            "Active boundaries: unknown ({JEV_ACTIVE_UNKNOWN_NOTE})\n"
        ));
    }
    text.push_str(&render_compaction_observation(report.pipeline.compaction.as_ref()));
    text.push_str(&format!("Footer: {JEV_FOOTER_RULE_NOTICE}\n"));
    text.push('\n');
    text.push_str(JEV_DISCLOSURE_NOTICE);
    text.push_str("\n\n");
    text.push_str(JEV_BOUNDARY_NOTICE);
    text.push('\n');
    text
}

/// The `/jev full-jev status` panel. Pure and local-only: one settings
/// snapshot in, text out — no network call, no secret, no worker telemetry.
/// It reports the overlay truth, this chat's effective values, and the saved
/// decisions the overlay is currently masking (they return on removal).
pub fn render_full_jev_status(
    settings: &JevSettings,
    session_id: &str,
    credential: CredentialStatus,
) -> String {
    let active = settings.full_jev_active();
    let resolution = settings.effective_mode_with_scope(session_id);
    let masked = settings.full_jev_masked_sessions();
    let mut text = String::new();
    text.push_str("Jev full-jev Status\n\n");
    if active {
        let revision = settings
            .full_jev
            .as_ref()
            .map(|profile| profile.revision)
            .unwrap_or(0);
        text.push_str(&format!("Profile: active (revision {revision})\n"));
        text.push_str("Resolves above every saved session, global and inherited override:\n");
        text.push_str(&format!(
            "  Mode: {}\n",
            mode_label(pi_jev::types::JevMode::CompareAndActive)
        ));
        text.push_str("  Feature gates: all on\n");
        text.push_str("  Compaction: on (request-local, independent of the decision mode)\n");
    } else {
        text.push_str("Profile: not installed\n");
        text.push_str(
            "While active it would resolve: mode Compare + Active, every feature gate on, compaction on.\n",
        );
    }
    text.push_str(&format!(
        "This chat resolves: {} (scope: {})\n",
        mode_label(resolution.mode),
        resolution.scope.as_str()
    ));
    let (compaction_enabled, compaction_scope) = settings.effective_compaction_with_scope(session_id);
    text.push_str(&render_compaction_status(compaction_enabled, compaction_scope));
    if active {
        if masked.is_empty() {
            text.push_str("Saved decisions masked by the overlay: none\n");
        } else {
            text.push_str(&format!(
                "Saved decisions masked by the overlay: {} (they resolve again when the overlay is removed)\n",
                masked.len()
            ));
            for session in masked.iter().take(8) {
                text.push_str(&format!("  - {session}\n"));
            }
            if masked.len() > 8 {
                text.push_str(&format!("  - ... and {} more\n", masked.len() - 8));
            }
        }
    }
    text.push_str(&format!("{}\n", credential.describe()));
    if !credential.present() {
        text.push_str(
            "No API key is configured: the footer shows unavailable and every decision fails closed until /jev key.\n",
        );
    }
    text.push_str("\n");
    text.push_str(JEV_DISCLOSURE_NOTICE);
    text.push_str("\n\n");
    text.push_str(JEV_BOUNDARY_NOTICE);
    text.push('\n');
    text
}

// ---------------------------------------------------------------------------
// Requested Jev model + explicit catalog (ROOT-CONTRACT v9)
// ---------------------------------------------------------------------------

/// HOST POLICY display cap for `/jev models` (presentation ONLY: the parsed
/// catalog is already capped at [`pi_jev::models::MAX_MODEL_CATALOG_ENTRIES`];
/// this caps only what the panel renders). Entries beyond the cap are
/// disclosed with a truthful overflow note, never dropped silently.
pub const MAX_MODEL_CATALOG_DISPLAY: usize = 32;

/// Renders the parsed catalog for `/jev models`. Pure presentation of
/// ALREADY-sanitized data: entries carry host-bounded prose, names are exact
/// identifiers (never normalized), and rejected reasons are bounded strings
/// that never echo the rejected text. No network, no settings write, no
/// selection (ROOT-CONTRACT v9).
pub fn render_model_catalog(catalog: &pi_jev::models::ModelCatalog) -> String {
    let mut text = String::new();
    text.push_str("Jev model catalog\n\n");
    if catalog.models.is_empty() {
        text.push_str(
            "No selectable models were returned. Nothing was selected automatically.\n",
        );
    }
    for (index, card) in catalog.models.iter().enumerate() {
        if index == MAX_MODEL_CATALOG_DISPLAY {
            text.push_str(&format!(
                "... and {} more entries (host display cap {MAX_MODEL_CATALOG_DISPLAY}; the parsed list is bounded at {})\n",
                catalog.models.len() - index,
                pi_jev::models::MAX_MODEL_CATALOG_ENTRIES,
            ));
            break;
        }
        text.push_str(&format!(
            "- {} — {} (released {})\n",
            card.name, card.description, card.release_date
        ));
    }
    if !catalog.rejected.is_empty() {
        text.push_str("\nRejected entries (bounded reasons; the unsafe values are never shown):\n");
        for (index, reason) in catalog.rejected.iter().enumerate() {
            if index == MAX_MODEL_CATALOG_DISPLAY {
                text.push_str(&format!(
                    "... and {} more rejection reasons (host display cap {MAX_MODEL_CATALOG_DISPLAY})\n",
                    catalog.rejected.len() - index,
                ));
                break;
            }
            text.push_str(&format!("- {reason}\n"));
        }
    }
    text.push_str("\nIDs are exact server-listed identifiers; a listed alias is not a resolved version. Set one explicitly with /jev model set <id>. This command only lists: nothing was selected, written or probed.\n");
    text
}

/// Everything `/jev model` (model status) needs. LOCAL only: settings truth
/// plus the in-process comparison snapshot when this process holds it; no
/// RPC, no catalog, no network of any kind (ROOT-CONTRACT v9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevModelStatusReport {
    /// The resolved requested model: the explicit selection or the native
    /// default. This is the id native SystemOne requests carry.
    pub requested: String,
    /// The explicit selection exactly as persisted, when one is set.
    pub explicit: Option<String>,
    /// Durable write identity of the settings snapshot the requested model
    /// came from (advances on every authoritative save, so held decisions
    /// can be invalidated by model A->B->A).
    pub write_revision: u64,
    /// Last server-reported model id from the in-process comparison status,
    /// when THIS process observed any. Unknown otherwise, never fabricated.
    pub reported: Option<String>,
}

/// Renders the requested-model panel. Pure: same input, same text, no I/O.
pub fn render_model_status(report: &JevModelStatusReport) -> String {
    let mut text = String::new();
    text.push_str("Jev Model\n\n");
    match report.explicit.as_deref() {
        Some(id) => text.push_str(&format!(
            "Requested Jev model: {id} (explicit /jev model set selection)\n"
        )),
        None => text.push_str(&format!(
            "Requested Jev model: {} (native default; nothing explicitly set)\n",
            report.requested
        )),
    }
    text.push_str(&format!(
        "Durable settings write revision: {}\n",
        report.write_revision
    ));
    text.push_str(&format!(
        "Model reported by server: {}\n",
        report
            .reported
            .clone()
            .unwrap_or_else(|| "unknown (no comparison observed in this worker)".to_string())
    ));
    if let Some(reported) = report.reported.as_deref() {
        if reported != report.requested {
            text.push_str(
                "Note: the server-reported id differs from the requested id; it is the response model of the last observed comparison, and identifier equality would not prove identical behavior anyway.\n",
            );
        }
    }
    text.push_str(
        "\nBoundaries: this selection is the Jev SystemOne request model only; it never selects the primary chat model, provider or effort. status/set/reset perform no network call, no probe and no budget change; set/reset write only this field atomically and are idempotent.\n",
    );
    text
}

/// Help/hotkey text: the modes and their real scope, with no model or
/// subagent claims (sections 11 and 12).
pub fn render_help() -> String {
    format!("{JEV_COMMAND_DESCRIPTION}\n\n\
Off                 No feature decision calls; compaction is independent.\n\
Compare             Shadow-only; no decisions applied.\n\
Active              Accepted, feature-gated native effects.\n\
Compare + Active    Comparison and application from one boundary request.\n\
{JEV_ACTIVE_NOTICE}\n\n\
Commands: /jev off|compare|active|compare-active|on|status|key|help\n\
/jev compact on|off|status    Independent request-local compaction control.\n\
/jev feature <name> on|off   Set one feature gate for this chat.\n\
/jev default <mode>          Default for sessions without a mode override.\n\
/jev default compact on|off  Default independent compaction toggle.\n\
/jev key clear              Remove the saved credential.\n\
/jev models                  Explicit one-shot model-catalog query (the only networked /jev model command).\n\
/jev model [status|set <id>|reset]  Requested Jev model: local panel, durable set, reset to native jev-latest (zero network).\n\
Numeric compaction/filtering policy is configured in jev-settings.json.\n\
{JEV_ON_COMPARE_NOTICE}\nFooter: {JEV_FOOTER_RULE_NOTICE}\n\n{JEV_BOUNDARY_NOTICE}\n")
}

pub fn render_compaction_status(enabled: bool, scope: ModeScope) -> String {
    format!("Compaction: {} (scope: {}; {})\n",
        if enabled { "on" } else { "off" }, scope.as_str(),
        if !enabled { "independent toggle is off" }
        else { "request-local; independent of decision mode" })
}

pub fn render_compaction_settings(settings: &JevSettings, session_id: &str) -> String {
    let (enabled, scope) = settings.effective_compaction_with_scope(session_id);
    let config = &settings.compaction;
    format!("{}keep_threshold: {}\npreserve_recent_messages: {}\nmax_state_tokens: {}\nmax_request_tokens: {}\ntruncate_head_chars: {}\nminimum_reduction_ratio: {}\nHard request and candidate caps also apply. No durable history is deleted.\n",
        render_compaction_status(enabled, scope),
        config.keep_threshold, config.preserve_recent_messages, config.max_state_tokens,
        config.max_request_tokens, config.truncate_head_chars, config.minimum_reduction_ratio)
}

/// Human label for a mode. Active is a real mode, so it is labelled plainly; the
/// wording that bounds what it does lives in [`JEV_ACTIVE_NOTICE`].
pub fn mode_label(mode: JevMode) -> &'static str {
    match mode {
        JevMode::Off => "Off",
        JevMode::Compare => "Compare",
        JevMode::Active => "Active",
        JevMode::CompareAndActive => "Compare + Active",
    }
}

/// The message a mode change produces, including scope and the notice that
/// belongs to the mode that was written.
pub fn mode_change_message(change: &ModeChange) -> String {
    match change {
        ModeChange::Applied { mode, scope } => format!(
            "Jev mode: {} (scope: {}){}",
            mode_label(*mode),
            match scope {
                ModeScope::Session => "this chat",
                ModeScope::GlobalDefault => "default for sessions without an override",
                ModeScope::BuiltIn => "built-in default",
                ModeScope::FullJevOverlay =>
                    "the global full-jev overlay (above every saved setting)",
            },
            match mode {
                JevMode::Compare => format!("\n{JEV_DISCLOSURE_NOTICE}"),
                JevMode::Active => format!("\n{JEV_ACTIVE_NOTICE}\n{JEV_BOUNDARY_NOTICE}"),
                JevMode::CompareAndActive => format!("\n{JEV_DISCLOSURE_NOTICE}\n{JEV_ACTIVE_NOTICE}\n{JEV_BOUNDARY_NOTICE}"),
                JevMode::Off => String::new(),
            },
        ),
        ModeChange::EmergencyExit { mode } => format!(
            "Jev mode: {} (scope: this chat)\n{JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE}",
            mode_label(*mode)
        ),
    }
}

/// The outcome of one mode-set request.
///
/// `Applied` means the requested value was written to the store. Every mode is
/// writable, so there is no refusal variant: a mode is never silently rewritten
/// into another one. The one compound case is the full-jev emergency exit:
/// `/jev off` while the overlay is active disables the overlay globally and
/// writes this session Off (decisions and compaction) in the same atomic save
/// (ROOT-CONTRACT v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeChange {
    Applied { mode: JevMode, scope: ModeScope },
    /// `/jev off` acted as the emergency exit: the global full-jev overlay was
    /// removed AND this session's decisions and compaction were set off in
    /// one atomic write. Other sessions resolve from saved settings again.
    EmergencyExit {
        mode: JevMode,
    },
}

/// The outcome of one `/jev full-jev` request. Never a silent success: the
/// `already_active`/`was_active` flags say whether anything was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullJevChange {
    /// The overlay install request. `already_active == true` means it was
    /// already on and NOTHING was written (idempotent install).
    Installed { already_active: bool },
    /// The overlay removal request. `was_active == false` means it was
    /// already off and NOTHING was written.
    Removed { was_active: bool },
}

/// The outcome of one `/jev model set|reset` (ROOT-CONTRACT v9). `written`
/// is false for a truthful no-op (the selection was already in the requested
/// state); no save happened in that case, so the durable write revision did
/// not move and held decisions were not invalidated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSetOutcome {
    pub written: bool,
    /// The requested id exactly as it will persist (`DEFAULT_MODEL` for a
    /// reset). Never secret material: the setter refuses credential-shaped
    /// ids before anything is stored or rendered.
    pub requested: String,
}

pub fn require_feature_support(supported: bool) -> Result<(), String> {
    if supported { Ok(()) } else {
        Err("The attached worker does not support Jev System One features. No settings were changed; update the daemon before enabling combined mode, feature gates or compaction.".into())
    }
}

/// Thin bridge the host uses for mode get/set, over lane A's
/// [`JevSettingsStore`] (`<agent_dir>/jev/jev-settings.json`).
///
/// It owns no network client and no scheduler, so a mode change needs no restart
/// and never blocks the UI. Every write refuses to persist anything that looks
/// like a secret, because that guard lives in the owner's `save`.
#[derive(Debug, Clone)]
pub struct JevModeBridge {
    store: JevSettingsStore,
}

impl JevModeBridge {
    pub fn new(agent_dir: &Path) -> Self {
        Self {
            store: JevSettingsStore::new(agent_dir),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.store.path().to_path_buf()
    }

    pub fn settings(&self) -> JevSettings {
        self.store.load()
    }

    pub fn effective_mode(&self, session_id: &str) -> JevMode {
        self.store.load().effective_mode(session_id)
    }

    /// The deciding scope for the status panel.
    pub fn scope(&self, session_id: &str) -> ModeScope {
        self.store.load().effective_mode_with_scope(session_id).scope
    }

    pub fn set_session_mode_supported(&self, session_id: &str, requested: JevMode, supported: bool) -> Result<ModeChange, String> {
        if requested == JevMode::CompareAndActive { require_feature_support(supported)?; }
        self.set_session_mode(session_id, requested)
    }

    pub fn set_global_default_supported(&self, requested: JevMode, supported: bool) -> Result<ModeChange, String> {
        if requested == JevMode::CompareAndActive { require_feature_support(supported)?; }
        self.set_global_default(requested)
    }

    /// One bounded load->mutate->save transaction over the generation-checked
    /// store (ROOT-CONTRACT v1: prove atomic/conflict-safe writes). A concurrent
    /// writer either moves the settings generation or holds the advisory lock;
    /// both surface as store errors, so instead of losing the change or
    /// overwriting the newer write, the bounded ceiling reloads and re-applies.
    /// `mutate` returns whether anything changed; a no-change request never
    /// writes. Returns true when a write happened.
    fn write_with_retry(&self, mut mutate: impl FnMut(&mut JevSettings) -> bool) -> Result<bool, String> {
        const WRITE_ATTEMPTS: usize = 3;
        let mut last_error = String::new();
        for _ in 0..WRITE_ATTEMPTS {
            let mut settings = self.store.load();
            if !mutate(&mut settings) {
                return Ok(false);
            }
            match self.store.save(&settings) {
                Ok(()) => return Ok(true),
                Err(error) => last_error = describe_error(error),
            }
        }
        Err(last_error)
    }

    /// True when the global full-jev overlay is active; used to reject
    /// conflicting writes it would mask (never a hidden success).
    fn full_jev_would_mask(&self) -> bool {
        self.store.load().full_jev_active()
    }

    /// Install or remove the global full-jev overlay (ROOT-CONTRACT v1).
    ///
    /// The overlay is a persisted block the resolver reads ABOVE every saved
    /// session, global and inherited override; the base settings and the
    /// sessions map are never rewritten, so removal restores the prior
    /// resolution exactly. Both directions are idempotent and the result
    /// reports whether anything was written.
    pub fn set_full_jev(&self, enabled: bool) -> Result<FullJevChange, String> {
        let written = self.write_with_retry(|settings| {
            if enabled {
                settings.full_jev_install()
            } else {
                settings.full_jev_remove()
            }
        })?;
        Ok(if enabled {
            FullJevChange::Installed {
                already_active: !written,
            }
        } else {
            FullJevChange::Removed {
                was_active: written,
            }
        })
    }

    /// Write an explicit per-session mode. Every mode is written, including
    /// `Active`: the store is the only source of truth, and no request is
    /// silently redirected to a different mode.
    ///
    /// Two full-jev rules (ROOT-CONTRACT v1) live here so the command path
    /// and the menu path can never disagree:
    /// - `Off` while the overlay is active is the EMERGENCY EXIT: remove the
    ///   overlay globally and set THIS session's decisions Off and compaction
    ///   false in one atomic save.
    /// - Any other mode while the overlay is active is REJECTED with the
    ///   no-change message: the overlay would mask the write, so reporting
    ///   success would be a lie.
    pub fn set_session_mode(
        &self,
        session_id: &str,
        requested: JevMode,
    ) -> Result<ModeChange, String> {
        if requested == JevMode::Off && self.full_jev_would_mask() {
            // Every required mutation contributes to the changed flag. If a
            // concurrent writer removes the overlay between the guard and the
            // fresh load, full_jev_remove() is false, and reporting the
            // emergency exit without committing the session writes would be a
            // success with zero writes (root F1): the changed flag is the OR
            // of the actual changes, never only the overlay removal.
            self.write_with_retry(|settings| {
                let removed = settings.full_jev_remove();
                let mode_written = settings.session_mode(session_id) != Some(JevMode::Off);
                let compaction_written =
                    settings.session_compaction_enabled(session_id) != Some(false);
                settings.set_session_mode(session_id, JevMode::Off);
                settings.set_session_compaction_enabled(session_id, false);
                removed || mode_written || compaction_written
            })?;
            return Ok(ModeChange::EmergencyExit { mode: JevMode::Off });
        }
        if requested != JevMode::Off && self.full_jev_would_mask() {
            return Err(JEV_FULL_JEV_REJECTION.to_string());
        }
        let mut settings = self.store.load();
        settings.set_session_mode(session_id, requested);
        self.store.save(&settings).map_err(describe_error)?;
        Ok(ModeChange::Applied {
            mode: requested,
            scope: ModeScope::Session,
        })
    }

    /// Write the global default for sessions with no explicit value. `Active`
    /// is written like every other mode. Rejected while the full-jev overlay
    /// is active: it would mask the write, so success would be a lie.
    pub fn set_global_default(&self, requested: JevMode) -> Result<ModeChange, String> {
        if self.full_jev_would_mask() {
            return Err(JEV_FULL_JEV_REJECTION.to_string());
        }
        let mut settings = self.store.load();
        settings.global_default = Some(requested);
        self.store.save(&settings).map_err(describe_error)?;
        Ok(ModeChange::Applied {
            mode: requested,
            scope: ModeScope::GlobalDefault,
        })
    }

    pub fn set_feature(
        &self,
        session_id: &str,
        feature: JevFeature,
        enabled: bool,
    ) -> Result<(), String> {
        if self.full_jev_would_mask() {
            return Err(JEV_FULL_JEV_REJECTION.to_string());
        }
        let mut settings = self.store.load();
        settings.set_session_feature(session_id, feature, enabled);
        self.store.save(&settings).map_err(describe_error)
    }

    pub fn set_compaction(&self, session_id: &str, enabled: bool) -> Result<(), String> {
        if self.full_jev_would_mask() {
            return Err(JEV_FULL_JEV_REJECTION.to_string());
        }
        let mut settings = self.store.load();
        settings.set_session_compaction_enabled(session_id, enabled);
        self.store.save(&settings).map_err(describe_error)
    }

    pub fn set_default_compaction(&self, enabled: bool) -> Result<(), String> {
        if self.full_jev_would_mask() {
            return Err(JEV_FULL_JEV_REJECTION.to_string());
        }
        let mut settings = self.store.load();
        settings.compaction_enabled = enabled;
        self.store.save(&settings).map_err(describe_error)
    }

    /// ROOT-CONTRACT v9: set the requested Jev model. Zero network, zero
    /// probe, zero availability claim; the only effects are validation and at
    /// most ONE atomic generation-checked durable write. The id is validated
    /// with the catalog parser's exact-identifier rule; when the EFFECTIVE
    /// credential is available, a credential-overlap id is refused with a
    /// GENERIC reason (never echoing the supplied value, never persisting or
    /// displaying it). Identical writes short-circuit BEFORE the save (no
    /// revision churn). The full-jev overlay neither masks nor selects this
    /// field — an operator selection is independent and deliberate — and
    /// model changes never touch control budgets.
    pub fn set_requested_model(
        &self,
        raw: &str,
        credential: Option<&SecretString>,
    ) -> Result<ModelSetOutcome, String> {
        if let Err(reason) = pi_jev::models::validate_requested_model_id(raw) {
            return Err(format!("Jev model id refused: {reason}"));
        }
        if let Some(secret) = credential {
            if pi_jev::models::id_overlaps_credential(raw, secret) {
                return Err(
                    "Jev model id refused: the supplied id looks like a credential; nothing was saved or displayed."
                        .to_string(),
                );
            }
        }
        let written = self.write_with_retry(|settings| {
            let changed = settings.requested_model.as_deref() != Some(raw);
            if changed {
                settings.requested_model = Some(raw.to_string());
            }
            changed
        })?;
        Ok(ModelSetOutcome {
            written,
            requested: raw.to_string(),
        })
    }

    /// ROOT-CONTRACT v9: reset to the native default `jev-latest` (a durable
    /// tombstone; a real write advances the persisted write revision, so held
    /// decisions are invalidated exactly like any authoritative save). Zero
    /// network, zero probe; an already-default reset is a truthful no-op.
    pub fn reset_requested_model(&self) -> Result<ModelSetOutcome, String> {
        let written = self.write_with_retry(|settings| settings.clear_requested_model())?;
        Ok(ModelSetOutcome {
            written,
            requested: pi_jev::types::DEFAULT_MODEL.to_string(),
        })
    }

    /// Clear the explicit per-session mode so the global default applies
    /// again. Rejected while the full-jev overlay is active: the overlay
    /// decides the effective mode right now, so a "cleared" report would be
    /// a hidden no-change success.
    pub fn clear_session_mode(&self, session_id: &str) -> Result<(), String> {
        if self.full_jev_would_mask() {
            return Err(JEV_FULL_JEV_REJECTION.to_string());
        }
        let mut settings = self.store.load();
        settings.clear_session_mode(session_id);
        self.store.save(&settings).map_err(describe_error)
    }

    /// Child inheritance at creation, through the owner's `inherit_mode`.
    ///
    /// The child snapshots the parent's EFFECTIVE mode, so a later global change
    /// cannot silently alter an existing child chat. A missing parent session
    /// degrades to the parent's resolved default (built-in `Off` when nothing is
    /// set) and NEVER to a silent `Compare`.
    pub fn inherit_into_child(
        &self,
        child_session_id: &str,
        parent_session_id: &str,
        child_override: Option<JevMode>,
    ) -> Result<JevMode, String> {
        let mut settings = self.store.load();
        let inherited =
            pi_jev::config::inherit_mode(&mut settings, child_session_id, parent_session_id, child_override);
        self.store.save(&settings).map_err(describe_error)?;
        Ok(inherited)
    }
}

/// Lane A's error text already excludes secrets and prompt content; this keeps
/// the shape the UI needs (a `String` the status line can show).
pub fn describe_error(error: JevError) -> String {
    error.log_line()
}

/// Operative modes are green (`success`); Off is red (`error`). The label text
/// always names the state too, so colour is never the sole indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevFooterState {
    Off,
    Compare,
    Active,
    CompareAndActive,
    Unavailable,
    Checking,
    Fallback,
}

impl JevFooterState {
    /// Text label. Text AND colour both carry the meaning. A healthy operative
    /// mode says `Jev On` and names its truthful effective mode, so the label
    /// can never claim a mode the session is not in.
    pub fn label(self) -> &'static str {
        match self {
            JevFooterState::Off => "Jev Off",
            JevFooterState::Compare => "Jev On (Compare)",
            JevFooterState::Active => "Jev On (Active)",
            JevFooterState::CompareAndActive => "Jev On (Compare + Active)",
            JevFooterState::Unavailable => "Jev unavailable",
            JevFooterState::Checking => "Jev checking",
            JevFooterState::Fallback => "Jev fallback",
        }
    }

    /// Theme colour key. Healthy operative modes are green (`success`); a
    /// credential-less or degraded operative mode falls to the amber ladder;
    /// Off is red (`error`).
    pub fn color_key(self) -> &'static str {
        match self {
            JevFooterState::Off => "error",
            JevFooterState::Compare | JevFooterState::Active | JevFooterState::CompareAndActive => {
                "success"
            }
            JevFooterState::Unavailable | JevFooterState::Checking | JevFooterState::Fallback => {
                "warning"
            }
        }
    }

    /// Short labelled form for narrow rows. Still names the state in text, so
    /// the identity and the On/Off meaning survive when the full label does not
    /// fit: a bare dot would be indistinguishable from the compaction dot.
    /// Degraded states keep their full label; they are rare and amber.
    pub fn compact_label(self) -> &'static str {
        match self {
            JevFooterState::Off => "Jev Off",
            JevFooterState::Compare => "Jev C On",
            JevFooterState::Active => "Jev A On",
            JevFooterState::CompareAndActive => "Jev C+A On",
            JevFooterState::Unavailable => "Jev unavailable",
            JevFooterState::Checking => "Jev checking",
            JevFooterState::Fallback => "Jev fallback",
        }
    }

    /// The dot glyph prefix.
    pub fn dot(self) -> &'static str {
        "\u{25cf}"
    }

    /// True for a healthy operative mode (the green dot states). Degraded
    /// operative modes (`Unavailable`/`Checking`/`Fallback`) stay amber, and Off
    /// stays red.
    pub fn is_green(self) -> bool {
        matches!(
            self,
            JevFooterState::Compare | JevFooterState::Active | JevFooterState::CompareAndActive
        )
    }
}

/// The independent Jev compaction state the second footer dot shows.
///
/// This is the CONFIGURED effective state for the current session (explicit
/// session override first, else the global default), never an inference from
/// the decision mode: `/jev off` does not imply compaction off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevCompactionState {
    On,
    Off,
    /// Only when no settings source could resolve the state. Never rendered as
    /// on or off.
    Unknown,
}

impl JevCompactionState {
    /// Text label. Text AND colour both carry the meaning.
    pub fn label(self) -> &'static str {
        match self {
            JevCompactionState::On => "Jev compact on",
            JevCompactionState::Off => "Jev compact off",
            JevCompactionState::Unknown => "Jev compact unknown",
        }
    }

    /// Short labelled form for narrow rows. `Jev Cmp on/off` stays distinct
    /// from the decision segment's `Jev C On` / `Jev A On` / `Jev C+A On`, so
    /// the two dots can never be confused by width pressure. Unknown keeps its
    /// full label.
    pub fn compact_label(self) -> &'static str {
        match self {
            JevCompactionState::On => "Jev Cmp on",
            JevCompactionState::Off => "Jev Cmp off",
            JevCompactionState::Unknown => "Jev compact unknown",
        }
    }

    /// Theme colour key: on is green, off is red, unknown is amber.
    pub fn color_key(self) -> &'static str {
        match self {
            JevCompactionState::On => "success",
            JevCompactionState::Off => "error",
            JevCompactionState::Unknown => "warning",
        }
    }

    /// The dot glyph prefix (same glyph as the decision segment).
    pub fn dot(self) -> &'static str {
        "\u{25cf}"
    }

    /// True only for the on state.
    pub fn is_green(self) -> bool {
        matches!(self, JevCompactionState::On)
    }
}

/// The effective compaction state from already-loaded settings.
///
/// The store resolves an explicit session override first, then the global
/// default; the standard (non-Jev) auto-compaction setting is a different
/// surface and is never read here.
pub fn compaction_state(settings: &JevSettings, session_id: &str) -> JevCompactionState {
    match settings.effective_compaction_with_scope(session_id) {
        (true, _) => JevCompactionState::On,
        (false, _) => JevCompactionState::Off,
    }
}

/// Derive the footer state from mode + credential + pipeline truth.
///
/// Rules, in order:
/// * `Off` -> red dot + `Jev Off`, always. Off does no work, so the credential and
///   the pipeline cannot change that.
/// * `Active` -> green `Jev On (Active)` once the credential and the pipeline
///   allow it; otherwise the same amber unavailable / checking / fallback states
///   Compare uses.
/// * `Compare` -> green `Jev On (Compare)` under the same credential and
///   pipeline rules.
/// * `CompareAndActive` -> green `Jev On (Compare + Active)` under the same rules.
/// * No credential -> amber `Jev unavailable`.
/// * In-flight work -> amber `Jev checking`.
/// * Recorded fallback/failure -> amber `Jev fallback`.
pub fn footer_state(
    mode: JevMode,
    credential: &CredentialStatus,
    pipeline: &JevPipelineStatus,
) -> JevFooterState {
    let healthy = match mode {
        JevMode::Off => return JevFooterState::Off,
        JevMode::Compare => JevFooterState::Compare,
        JevMode::Active => JevFooterState::Active,
        JevMode::CompareAndActive => JevFooterState::CompareAndActive,
    };
    if !credential.present() {
        return JevFooterState::Unavailable;
    }
    if pipeline.checking() {
        return JevFooterState::Checking;
    }
    if pipeline.degraded() {
        return JevFooterState::Fallback;
    }
    healthy
}

/// Plain-text footer segment: `\u{25cf} Jev Off`. No ANSI, so the caller can
/// measure width and truncate before colouring.
pub fn footer_text(state: JevFooterState) -> String {
    format!("{} {}", state.dot(), state.label())
}

/// Narrow decision segment: `\u{25cf} Jev C On`. A SHORT LABELLED form, never a
/// bare dot: the state and the mode stay readable when the full label cannot
/// fit, and the segment stays distinguishable from the compaction segment.
pub fn footer_compact_text(state: JevFooterState) -> String {
    format!("{} {}", state.dot(), state.compact_label())
}

/// Plain-text compaction segment: `\u{25cf} Jev compact on`. No ANSI.
pub fn footer_compaction_text(state: JevCompactionState) -> String {
    format!("{} {}", state.dot(), state.label())
}

/// Narrow compaction segment: `\u{25cf} Jev Cmp on`. Short labelled form, see
/// [`footer_compact_text`].
pub fn footer_compaction_compact_text(state: JevCompactionState) -> String {
    format!("{} {}", state.dot(), state.compact_label())
}

/// Minimum terminal columns needed to render the labelled form; below this the
/// caller keeps only the dot so narrow terminals do not lose layout.
pub const FOOTER_LABEL_MIN_COLUMNS: usize = 40;

/// Narrow-terminal-safe render: full text when it fits, dot-only when it does not.
///
/// The live status row owns its own truncation, so this measured form has no
/// caller inside the binary yet: it is the contract a caller that CAN measure
/// uses, and `tests/jev_ui_tests.rs` pins it.
#[allow(dead_code)]
pub fn footer_segment(state: JevFooterState, terminal_columns: usize) -> String {
    let text = footer_text(state);
    if terminal_columns < FOOTER_LABEL_MIN_COLUMNS {
        state.dot().to_string()
    } else {
        text
    }
}

/// Narrow-terminal-safe compaction render, same rule as [`footer_segment`].
#[allow(dead_code)]
pub fn footer_compaction_segment(state: JevCompactionState, terminal_columns: usize) -> String {
    if terminal_columns < FOOTER_LABEL_MIN_COLUMNS {
        state.dot().to_string()
    } else {
        footer_compaction_text(state)
    }
}



// ===========================================================================
// Menu rows and state machine (pure: external crates only, so the integration
// test in `tests/jev_ui_tests.rs` can include this file with `#[path]`)
// ===========================================================================

/// Modes, independent compaction controls, credentials and status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevMenuRow {
    Off,
    Compare,
    Active,
    CompareAndActive,
    CompactionOn,
    CompactionOff,
    InputKey,
    Status,
}

impl JevMenuRow {
    /// Declaration order; the selector walks this.
    pub const ALL: [JevMenuRow; 8] = [
        JevMenuRow::Off,
        JevMenuRow::Compare,
        JevMenuRow::Active,
        JevMenuRow::CompareAndActive,
        JevMenuRow::CompactionOn,
        JevMenuRow::CompactionOff,
        JevMenuRow::InputKey,
        JevMenuRow::Status,
    ];

    pub fn title(self, active: JevMode) -> String {
        let marker = match self {
            JevMenuRow::Off if active == JevMode::Off => " (current)",
            JevMenuRow::Compare if active == JevMode::Compare => " (current)",
            JevMenuRow::Active if active == JevMode::Active => " (current)",
            JevMenuRow::CompareAndActive if active == JevMode::CompareAndActive => " (current)",
            _ => "",
        };
        match self {
            // The row is the plain mode name: Active is a real, selectable mode.
            JevMenuRow::Off => format!("Off{marker}"),
            JevMenuRow::Compare => format!("Compare{marker}"),
            JevMenuRow::Active => format!("Active{marker}"),
            JevMenuRow::CompareAndActive => format!("Compare + Active{marker}"),
            JevMenuRow::CompactionOn => "Compaction on".to_string(),
            JevMenuRow::CompactionOff => "Compaction off".to_string(),
            JevMenuRow::InputKey => "Input API key".to_string(),
            JevMenuRow::Status => "Status".to_string(),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            JevMenuRow::Off => "Disable feature decisions. Independent compaction keeps its setting.",
            JevMenuRow::Compare => "Shadow-only observations; no decisions applied.",
            JevMenuRow::Active => JEV_ACTIVE_NOTICE,
            JevMenuRow::CompareAndActive => "Compare and apply accepted, gated decisions from the same boundary request.",
            JevMenuRow::CompactionOn => "Enable request-local compaction independently of the decision mode.",
            JevMenuRow::CompactionOff => "Disable compaction only. Other Jev features and mode stay unchanged.",
            JevMenuRow::InputKey => "Enter the TypeSafe key (masked, never echoed).",
            JevMenuRow::Status => "Mode, scope, credential source, counters, queue, skips.",
        }
    }
}

/// What the owner must do after one keypress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JevMenuAction {
    None,
    /// Cancel the dialog immediately.
    Cancel,
    /// Set the session mode to this value. Every row that selects a mode reports
    /// this action; the owner writes it.
    SetMode(JevMode),
    SetCompaction(bool),
    /// Open the masked key-entry dialog.
    InputKey,
    /// Show the status panel.
    ShowStatus,
}

/// The menu state machine.
///
/// Pure and testable: it owns the selection index and the explanation line, and
/// every transition is a total function of (key, state).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevMenuState {
    pub selected: usize,
    pub active_mode: JevMode,
    /// The last explanation shown under the list (errors, async notes).
    pub message: Option<String>,
    /// True while the owner is doing async work for this dialog.
    pub busy: bool,
    /// Set once the dialog must close; the owner hides the overlay.
    pub closed: bool,
}

impl Default for JevMenuState {
    fn default() -> Self {
        Self::new(JevMode::Off)
    }
}

impl JevMenuState {
    pub fn new(active_mode: JevMode) -> Self {
        // Preselect the row of the effective mode, so the current mode is always
        // visible as `(current)` under the cursor.
        let selected = match active_mode {
            JevMode::Compare => 1,
            JevMode::Active => 2,
            JevMode::CompareAndActive => 3,
            _ => 0,
        };
        Self {
            selected,
            active_mode,
            message: None,
            busy: false,
            closed: false,
        }
    }

    pub fn rows(&self) -> [JevMenuRow; 8] {
        JevMenuRow::ALL
    }

    pub fn current_row(&self) -> JevMenuRow {
        JevMenuRow::ALL[self.selected.min(JevMenuRow::ALL.len() - 1)]
    }

    pub fn move_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn move_down(&mut self) {
        self.selected = (self.selected + 1).min(JevMenuRow::ALL.len() - 1);
    }

    /// Cancel is available at every moment, including while async validation or a
    /// status refresh is pending. It never blocks on that work.
    pub fn cancel(&mut self) {
        self.closed = true;
        self.busy = false;
        self.message = None;
    }

    /// Accept the current row.
    pub fn accept(&mut self) -> JevMenuAction {
        if self.closed {
            return JevMenuAction::None;
        }
        match self.current_row() {
            JevMenuRow::Off => {
                self.active_mode = JevMode::Off;
                self.closed = true;
                JevMenuAction::SetMode(JevMode::Off)
            }
            JevMenuRow::Compare => {
                self.active_mode = JevMode::Compare;
                self.closed = true;
                JevMenuAction::SetMode(JevMode::Compare)
            }
            JevMenuRow::Active => {
                // Active is a real mode: the state changes and the owner writes it,
                // exactly like Off and Compare.
                self.active_mode = JevMode::Active;
                self.closed = true;
                JevMenuAction::SetMode(JevMode::Active)
            }
            JevMenuRow::CompareAndActive => {
                self.active_mode = JevMode::CompareAndActive;
                self.closed = true;
                JevMenuAction::SetMode(JevMode::CompareAndActive)
            }
            JevMenuRow::CompactionOn | JevMenuRow::CompactionOff => {
                let enabled = self.current_row() == JevMenuRow::CompactionOn;
                self.closed = true;
                JevMenuAction::SetCompaction(enabled)
            }
            JevMenuRow::InputKey => {
                self.closed = true;
                JevMenuAction::InputKey
            }
            JevMenuRow::Status => {
                // The status panel is rendered by the handler, so the overlay
                // closes first. This keeps the panel code in one place.
                self.closed = true;
                JevMenuAction::ShowStatus
            }
        }
    }

    /// Handle one input chunk through the configurable bindings.
    ///
    /// Precedence: cancel first (so Escape and Ctrl+C always work, including
    /// during async validation), then navigation, then submit.
    pub fn handle_key(&mut self, data: &str) -> JevMenuAction {
        if is_cancel_key(data) {
            self.cancel();
            return JevMenuAction::Cancel;
        }
        let keybindings = get_keybindings();
        if keybindings.matches(data, "tui.select.up") {
            self.move_up();
            return JevMenuAction::None;
        }
        if keybindings.matches(data, "tui.select.down") {
            self.move_down();
            return JevMenuAction::None;
        }
        if is_submit_key(data) {
            return self.accept();
        }
        JevMenuAction::None
    }

    /// Called by the owner when async work starts or finishes, so the dialog can
    /// show a busy line without blocking input.
    pub fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
    }

    /// The hint line the owner renders. Resolved from the LIVE bindings, so a
    /// user override changes the hint too; no key is hardcoded.
    pub fn hint(&self) -> String {
        format!(
            "{}  {}  {}",
            binding_hint("tui.select.up", "move"),
            binding_hint("tui.select.confirm", "select"),
            binding_hint("tui.select.cancel", "cancel")
        )
    }

    /// The body lines for the menu, before theme colouring.
    pub fn body_lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for (index, row) in JevMenuRow::ALL.iter().enumerate() {
            let marker = if index == self.selected { "> " } else { "  " };
            lines.push(format!("{marker}{}", row.title(self.active_mode)));
            if index == self.selected {
                lines.push(format!("     {}", row.description()));
            }
        }
        if self.busy {
            lines.push("Working... (cancel is always available)".to_string());
        }
        if let Some(message) = &self.message {
            lines.push(message.clone());
        }
        lines.push(self.hint());
        lines
    }
}



// ===========================================================================
// Masked key-entry logic (pure) + footer payload
// ===========================================================================

/// The mask character. Fixed width, so the real length is not observable.
pub const MASK_CHAR: char = '\u{2022}';

/// Fixed mask width: the value length is never published by the UI.
pub const MASK_LENGTH: usize = 12;

/// The prompt line for masked entry.
pub const KEY_INPUT_PROMPT: &str = "TypeSafe API key (masked)";

/// The always-visible explanation under the masked line.
pub const KEY_INPUT_HINT: &str =
    "Paste or type the key. It is never echoed, never stored in history, and never printed.";

/// The masked rendering of a value. The result never contains an input
/// character, so it is safe to render, log, diff or put in a bug report.
pub fn mask_value(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    std::iter::repeat(MASK_CHAR).take(MASK_LENGTH).collect()
}

/// Redact key-like tokens from a server failure reason.
///
/// A reason is shown to the user, so it must not echo the credential back. Long
/// opaque tokens (>= 16 chars of alphanumerics/`-_.`) are replaced, while the
/// useful part (status code, category, short words) survives.
pub fn redact_reason(reason: &str) -> String {
    reason
        .split_whitespace()
        .map(|token| {
            let trimmed = token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric());
            let opaque = trimmed.len() >= 16
                && trimmed
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.');
            if opaque { "[redacted]" } else { token }
        })
        .collect::<Vec<&str>>()
        .join(" ")
}

/// What the host must do next with the buffered key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyInputState {
    /// Collecting input; no validation requested yet.
    Editing,
    /// Submitted; asynchronous validation is in flight.
    Validating,
    /// Validation succeeded and the caller consumed the buffer.
    Validated,
    /// Validation failed with a redacted reason (never the key itself).
    Failed(String),
    /// Cancelled by the user or the caller.
    Cancelled,
}

/// The masked key buffer.
///
/// `Debug` is manual and redacts the buffer, so an accidental `{:?}` (a panic
/// payload, a test failure dump) cannot leak the key. The buffer is taken exactly
/// once by [`JevKeyInputState::take_for_validation`] and cleared on cancel, so
/// the secret has the shortest possible lifetime.
#[derive(Clone, Default)]
pub struct JevKeyInputState {
    value: String,
    state: Option<KeyInputState>,
    /// Monotonic submission token. A result carrying a superseded token is a
    /// STALE result (the user cancelled, re-submitted, or switched session) and
    /// must never be applied.
    generation: u64,
}

impl std::fmt::Debug for JevKeyInputState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JevKeyInputState")
            .field("value", &"<redacted>")
            .field("value_present", &!self.value.is_empty())
            .field("state", &self.state())
            .field("generation", &self.generation)
            .finish()
    }
}

impl JevKeyInputState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> KeyInputState {
        self.state.clone().unwrap_or(KeyInputState::Editing)
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.state(),
            KeyInputState::Validated | KeyInputState::Failed(_) | KeyInputState::Cancelled
        )
    }

    /// Buffer length only; never the value.
    pub fn value_len(&self) -> usize {
        self.value.chars().count()
    }

    pub fn value_is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// The current submission token. A caller that performs asynchronous work
    /// keeps this value and passes it back to [`Self::apply_validation`].
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Hand the secret to the caller exactly once, when validation starts.
    ///
    /// The buffer is moved out (never copied), so a secret has one owner at a
    /// time. The submission token in [`Self::generation`] is bumped, which makes
    /// every earlier in-flight result stale.
    pub fn take_for_validation(&mut self) -> Option<String> {
        if self.is_done() || matches!(self.state(), KeyInputState::Validating) {
            return None;
        }
        let value = std::mem::take(&mut self.value);
        if value.is_empty() {
            // Nothing to validate: stay in the editing state so the user can retry.
            return None;
        }
        self.generation = self.generation.wrapping_add(1);
        self.state = Some(KeyInputState::Validating);
        Some(value)
    }

    /// Apply an asynchronous validation result.
    ///
    /// Returns `false` when the token is stale: the result belongs to a
    /// superseded submission (cancel, retry, or session switch) and is ignored,
    /// so a late answer can never change the state of a newer attempt.
    pub fn apply_validation(&mut self, token: u64, result: Result<(), String>) -> bool {
        if token != self.generation {
            return false;
        }
        self.value.clear();
        self.state = Some(match result {
            Ok(()) => KeyInputState::Validated,
            Err(reason) => KeyInputState::Failed(redact_reason(&reason)),
        });
        true
    }

    /// Immediate cancel: nothing is submitted, the buffer is dropped, and the
    /// token moves on so any in-flight validation is stale.
    pub fn cancel(&mut self) {
        self.value.clear();
        self.generation = self.generation.wrapping_add(1);
        self.state = Some(KeyInputState::Cancelled);
    }

    /// Strip newlines and tabs so a paste cannot break the single-line contract
    /// or inject a control sequence into the prompt.
    pub fn sanitize(chunk: &str) -> String {
        chunk
            .chars()
            .filter(|ch| *ch != '\n' && *ch != '\r' && *ch != '\t')
            .collect()
    }

    /// Number of characters a chunk contributes after sanitising.
    pub fn paste(&mut self, chunk: &str) {
        if self.is_done() || matches!(self.state(), KeyInputState::Validating) {
            return;
        }
        let chunk = chunk.strip_prefix("\x1b[200~").unwrap_or(chunk);
        let chunk = chunk.strip_suffix("\x1b[201~").unwrap_or(chunk);
        for character in chunk.chars().filter(|ch| !ch.is_control()) {
            if self.value.len().saturating_add(character.len_utf8()) > pi_jev::credential::MAX_SECRET_LEN {
                self.value.clear();
                self.state = Some(KeyInputState::Failed("Key exceeds the allowed length".into()));
                return;
            }
            self.value.push(character);
        }
    }

    /// Handle one input chunk. Cancel is checked FIRST, so Escape/Ctrl+C work
    /// during asynchronous validation as well.
    pub fn handle_key(&mut self, data: &str) -> KeyInputState {
        if is_cancel_key(data) {
            self.cancel();
            return self.state();
        }
        if self.is_done() || matches!(self.state(), KeyInputState::Validating) {
            return self.state();
        }
        if is_submit_key(data) {
            // Submission only records intent; the owner calls take_for_validation.
            return self.state();
        }
        if data.chars().count() > 1 {
            self.paste(data);
            return self.state();
        }
        self.paste(data);
        self.state()
    }

    /// The status line the dialog shows. Never contains the value.
    pub fn status_line(&self) -> String {
        match self.state() {
            KeyInputState::Editing if self.value.is_empty() => KEY_INPUT_HINT.to_string(),
            KeyInputState::Editing => {
                format!("{} characters entered (masked)", self.value_len())
            }
            KeyInputState::Validating => {
                format!("Saving to the secure credential store... ({})", binding_hint("tui.select.cancel", "cancel"))
            }
            KeyInputState::Validated => {
                "Key accepted and stored in the platform credential store.".to_string()
            }
            KeyInputState::Failed(reason) => format!("Key rejected: {reason}"),
            KeyInputState::Cancelled => "Cancelled. Nothing was stored.".to_string(),
        }
    }

    /// The masked line the dialog renders.
    pub fn masked_line(&self) -> String {
        format!("> {}", mask_value(&self.value))
    }
}

/// True when the key data matches the configured cancel OR interrupt binding.
/// Both ids are user-configurable; no key is hardcoded here.
pub fn is_cancel_key(data: &str) -> bool {
    let keybindings = pi_tui::keybindings::get_keybindings();
    keybindings.matches(data, "tui.select.cancel") || keybindings.matches(data, "app.interrupt")
        || keybindings.matches(data, "app.jev.cancel")
}

/// True when the key data matches the configured submit binding.
pub fn is_submit_key(data: &str) -> bool {
    pi_tui::keybindings::get_keybindings().matches(data, "tui.input.submit")
}

/// A `Key+Key label` hint resolved from the live binding, e.g. `Esc cancel`.
pub fn binding_hint(keybinding: &str, description: &str) -> String {
    let keys = pi_tui::keybindings::get_keybindings().get_keys(keybinding);
    if keys.is_empty() {
        return description.to_string();
    }
    format!("{} {}", keys.join("/"), description)
}

// ===========================================================================
// Footer payload
// ===========================================================================

/// The extension status key the Jev footer publishes under.
pub const JEV_STATUS_KEY: &str = "jev";

/// The extension status key the INDEPENDENT Jev compaction dot publishes under.
///
/// A separate key on purpose: a decision-mode refresh must never clobber the
/// compaction state and vice versa, and `Jev Off` must never remove the
/// compaction dot.
pub const JEV_COMPACT_STATUS_KEY: &str = "jev-compact";

/// Documentation of the footer colour rule (asserted by the tests). The callers
/// prefix it with `Footer: `, so it does not repeat that word itself.
pub const JEV_FOOTER_RULE_NOTICE: &str =
    "green \"Jev On (Compare)\"/\"Jev On (Active)\"/\"Jev On (Compare + Active)\", red \"Jev Off\", amber \"Jev unavailable\"/\"Jev checking\"/\"Jev fallback\". A second dot shows compaction: green \"Jev compact on\", red \"Jev compact off\". The two are independent: turning one off never turns the other off. On narrow rows the short forms \"Jev C On\"/\"Jev A On\"/\"Jev C+A On\" and \"Jev Cmp on\"/\"Jev Cmp off\" keep both states readable in text; never two bare dots.";

/// The theme colour key for the footer segment.
///
/// `success` is returned only when [`JevFooterState::is_green`] says the state is
/// a green one (a healthy operative mode). Degraded operative modes stay amber,
/// so a red or amber state can never masquerade as healthy.
pub fn footer_color_key(state: JevFooterState) -> &'static str {
    if state.is_green() {
        "success"
    } else {
        state.color_key()
    }
}

/// The `setStatus` payload that publishes the footer segment.
///
/// The host routes it into `extension_surfaces.set_status`, and
/// `native_host_extensions::Statuses` renders it ON the model/effort tray row
/// (after the effort label), so the segment needs no new host surface. Kept as a
/// `serde_json::Value` so the tests pin the exact payload the host receives. The
/// live publisher in `jev_footer.rs` uses the labelled form, because the dispatch
/// task cannot measure the terminal; this measured form is the contract for a
/// caller that can. `statusCompactText` is the optional SHORT LABELLED narrow form
/// (`\u{25cf} Jev C On`): receivers that do not know it ignore the field, and
/// senders that omit it degrade to left-truncation instead of segment
/// compaction. Never a bare dot: two bare dots cannot be told apart.
#[allow(dead_code)]
pub fn footer_status_payload(state: JevFooterState, terminal_columns: usize) -> serde_json::Value {
    serde_json::json!({
        "statusKey": JEV_STATUS_KEY,
        "statusText": footer_segment(state, terminal_columns),
        "statusCompactText": footer_compact_text(state),
    })
}

/// The `setStatus` payload that removes the footer segment.
pub fn footer_clear_payload() -> serde_json::Value {
    serde_json::json!({
        "statusKey": JEV_STATUS_KEY,
        "statusText": serde_json::Value::Null,
    })
}

/// The `setStatus` payload for the independent compaction dot. Same optional
/// `statusCompactText` contract as [`footer_status_payload`].
#[allow(dead_code)]
pub fn footer_compaction_status_payload(state: JevCompactionState, terminal_columns: usize) -> serde_json::Value {
    serde_json::json!({
        "statusKey": JEV_COMPACT_STATUS_KEY,
        "statusText": footer_compaction_segment(state, terminal_columns),
        "statusCompactText": footer_compaction_compact_text(state),
    })
}

/// The `setStatus` payload that removes the compaction dot.
pub fn footer_compaction_clear_payload() -> serde_json::Value {
    serde_json::json!({
        "statusKey": JEV_COMPACT_STATUS_KEY,
        "statusText": serde_json::Value::Null,
    })
}


// ===========================================================================
// Secret handling: the only type allowed to carry key material
// ===========================================================================

/// The entered API key, wrapped so it cannot leak by accident.
///
/// * `Debug` is manual and prints `<redacted>`, so a panic payload, a test
///   failure dump or a `{:?}` in a log never carries the secret.
/// * `Display` is NOT implemented on purpose: `format!("{secret}")` must not
///   compile, so no string interpolation can put the key into a message.
/// * `expose` is the only accessor and it borrows, so the caller cannot store a
///   copy without asking for one explicitly.
#[derive(Clone, PartialEq, Eq)]
pub struct JevSecret(String);

impl std::fmt::Debug for JevSecret {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("JevSecret(<redacted>)")
    }
}

impl JevSecret {
    /// Take ownership of an entered value. The value is not copied.
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Hand the value to the owner's [`SecretString`], which is the only type the
    /// credential store accepts. `SecretString` has no `Display` either.
    pub fn into_secret_string(self) -> SecretString {
        SecretString::new(self.0)
    }

    /// Borrow the value for a length/placeholder check only. The credential store
    /// call goes through [`Self::into_secret_string`], so no other path can pass
    /// the raw string to a sink.
    pub fn peek(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.chars().count()
    }

    /// True when the value looks like a placeholder rather than a real key.
    pub fn looks_like_placeholder(&self) -> bool {
        let value = self.0.trim();
        value.eq_ignore_ascii_case("test")
            || value.eq_ignore_ascii_case("changeme")
            || value.starts_with("sk-xxx")
    }

    /// Cheap shape check the UI runs BEFORE the store call, so an obvious typo is
    /// rejected without a round trip. It is intentionally permissive: the server
    /// remains the only authority on a key's validity.
    pub fn looks_malformed(&self) -> bool {
        let value = self.0.trim();
        value.len() < 8 || value.chars().any(char::is_whitespace)
    }
}

/// Store an entered key through the shared [`CredentialStore`] (DESIGN.md section
/// 6). This is the ONLY write path the UI has:
///
/// * the value never becomes a `String` argument, so it cannot be logged;
/// * a store that reports `is_available() == false` is refused BEFORE the write,
///   so the UI never degrades to plaintext;
/// * the owner's `save`/`store` guard is the second line of defence.
pub fn store_secret(
    store: &dyn CredentialStore,
    secret: JevSecret,
) -> Result<(), JevError> {
    if secret.is_empty() {
        return Err(JevError::validation("the API key is empty"));
    }
    if secret.looks_like_placeholder() {
        return Err(JevError::validation(
            "that looks like a placeholder, not an API key",
        ));
    }
    if secret.looks_malformed() {
        return Err(JevError::validation(
            "that does not look like an API key (too short, or it contains whitespace)",
        ));
    }
    if !store.is_available() {
        return Err(JevError::Unavailable {
            reason: "the platform credential store is not available".to_string(),
        });
    }
    let secret = secret.into_secret_string();
    store.store(DEFAULT_KEY_ID, secret.expose())
}

/// Delete the stored key. Kept beside the write so the UI's only credential
/// mutation is paired with its inverse.
pub fn clear_secret(store: &dyn CredentialStore) -> Result<(), JevError> {
    store.delete(DEFAULT_KEY_ID)
}

#[cfg(test)]
mod worker_status_tests {
    use super::*;

    #[test]
    fn unavailable_worker_telemetry_is_not_reported_as_zero() {
        let report = JevStatusReport::local_only(JevMode::Compare, ModeScope::Session,
            CredentialStatus::resolve(true, false, false));
        let rendered = render_status(&report.with_snapshot(None));
        assert!(rendered.contains("Counters: unknown"));
        assert!(rendered.contains("Last success: unknown"));
        assert!(!rendered.contains("Counters: 0 ok, 0 failed"));
        assert!(!JevPipelineStatus::from_snapshot(Some(&serde_json::json!({}))).observed);
    }

    #[test]
    fn worker_snapshot_populates_real_status_fields() {
        let snapshot = serde_json::json!({
            "success_count": 12, "failure_count": 2, "queue_capacity": 128,
            "queue_depth": 3, "in_flight": 1, "last_success_ms": 1000,
            "last_latency_ms": 47, "response_model": "jev-test-model",
            "dropped_comparisons": 4, "fallback_reason": null,
            "skipped_categories": {"memory_relevance": "no_memory_state"}
        });
        let pipeline = JevPipelineStatus::from_snapshot(Some(&snapshot));
        assert!(pipeline.observed);
        assert!(pipeline.checking());
        assert!(!pipeline.degraded(), "past failures do not erase later recovery");
        assert_eq!(pipeline.success_count, 12);
        assert_eq!(pipeline.failure_count, 2);
        assert_eq!(pipeline.last_latency_ms, Some(47));
        assert_eq!(pipeline.response_model.as_deref(), Some("jev-test-model"));
        assert_eq!(pipeline.skipped_categories.len(), 1);
    }

    #[test]
    fn secure_input_is_bounded_and_cannot_resubmit_while_saving() {
        let mut input = JevKeyInputState::new();
        input.paste("synthetic-credential");
        assert!(input.take_for_validation().is_some());
        input.paste("second-secret");
        assert!(input.take_for_validation().is_none());
        assert!(input.value_is_empty());
        input.cancel();
        assert_eq!(input.state(), KeyInputState::Cancelled);

        let mut oversized = JevKeyInputState::new();
        oversized.paste(&"x".repeat(pi_jev::credential::MAX_SECRET_LEN + 1));
        assert!(oversized.value_is_empty());
        assert!(matches!(oversized.state(), KeyInputState::Failed(_)));
    }
}

#[cfg(test)]
mod full_jev_write_retry_tests {
    use super::*;

    /// Deterministic interleaving (root F2): a concurrent writer lands BETWEEN
    /// this bridge's load and its save — exactly like a second process racing
    /// the settings file. The bounded retry must re-apply on a fresh load
    /// without losing either writer's fields.
    #[test]
    fn write_with_retry_reapplies_after_a_concurrent_writer_moves_the_generation() {
        let directory = tempfile::tempdir().expect("tempdir");
        let bridge = JevModeBridge::new(directory.path());
        let racer = JevSettingsStore::new(directory.path());
        let mut injected = false;
        let wrote = bridge
            .write_with_retry(|settings| {
                if !injected {
                    injected = true;
                    let mut winner = racer.load();
                    winner.set_session_mode("racer", JevMode::Compare);
                    racer
                        .save(&winner)
                        .expect("concurrent writer wins the race");
                }
                settings.set_session_mode("target", JevMode::Off);
                true
            })
            .expect("the bounded retry commits");
        assert!(wrote);
        let settled = racer.load();
        assert_eq!(
            settled.session_mode("racer"),
            Some(JevMode::Compare),
            "the concurrent writer's change survives the retry"
        );
        assert_eq!(
            settled.session_mode("target"),
            Some(JevMode::Off),
            "the retried write is re-applied on the fresh load"
        );
    }

    /// Root F1 regression: the emergency-exit mutation shape (the closure in
    /// [`JevModeBridge::set_session_mode`]) must commit the session Off and
    /// compaction-off writes even when a concurrent writer removes the
    /// overlay between the guard and the fresh load, so the reported
    /// `ModeChange::EmergencyExit` never describes zero writes.
    #[test]
    fn emergency_exit_commits_session_writes_when_the_overlay_vanishes_mid_flight() {
        let directory = tempfile::tempdir().expect("tempdir");
        let bridge = JevModeBridge::new(directory.path());
        let racer = JevSettingsStore::new(directory.path());
        // alpha holds an explicit saved Compare + compaction on that the exit
        // must flip; the overlay is active on disk.
        let mut base = racer.load();
        base.set_session_mode("alpha", JevMode::Compare);
        base.set_session_compaction_enabled("alpha", true);
        assert!(base.full_jev_install());
        racer.save(&base).expect("base save");
        let mut injected = false;
        let wrote = bridge
            .write_with_retry(|settings| {
                if !injected {
                    injected = true;
                    // The concurrent removal lands between this closure's
                    // fresh load and its save.
                    let mut other = racer.load();
                    assert!(other.full_jev_remove());
                    racer.save(&other).expect("racer save");
                }
                // The production emergency-exit closure shape: the changed
                // flag is the OR of every required mutation.
                let removed = settings.full_jev_remove();
                let mode_written = settings.session_mode("alpha") != Some(JevMode::Off);
                let compaction_written =
                    settings.session_compaction_enabled("alpha") != Some(false);
                settings.set_session_mode("alpha", JevMode::Off);
                settings.set_session_compaction_enabled("alpha", false);
                removed || mode_written || compaction_written
            })
            .expect("the bounded retry commits");
        assert!(wrote, "the session writes are required mutations");
        let settled = racer.load();
        assert!(
            !settled.full_jev_active(),
            "the concurrent removal stands; the retry does not resurrect the overlay"
        );
        assert_eq!(settled.session_mode("alpha"), Some(JevMode::Off));
        assert_eq!(settled.session_compaction_enabled("alpha"), Some(false));
    }
}
