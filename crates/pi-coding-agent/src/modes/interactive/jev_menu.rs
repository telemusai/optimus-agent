//! Jev command surface + mode store bridge for the interactive UI.
//!
//! This module owns three things and nothing else:
//!
//! 1. `/jev` argument parsing (`/jev`, `/jev off`, `/jev compare`, `/jev active`,
//!    `/jev on`, `/jev status`, `/jev key`, `/jev help`).
//! 2. The mode get/set bridge into the `pi-jev` config store, including scope
//!    reporting ("this chat" vs "defaults for new chats").
//! 3. The truthful status text the UI renders for `/jev status`.
//!
//! DESIGN.md binding constraints this module implements:
//!
//! * section 0 / 10.2 - `/jev` is the only mode control; key presence NEVER
//!   enables Jev; explicit per-session mode > explicit global default >
//!   built-in `Off`.
//! * section 11 - NO model control: the user's primary model/provider/effort
//!   stays authoritative. Nothing here reads or writes a model.
//! * section 12 - NO subagent control: this module has no spawn/delete/cancel/
//!   model/task/message/budget surface. Category 5/6 assessments are advisory
//!   records in Compare and are never converted into commands.

use std::path::{Path, PathBuf};

use pi_tui::keybindings::get_keybindings;

use pi_jev::config::{
    resolve_credential_source, CredentialSource, EnvKeyPresence, JevSettings, JevSettingsStore,
    ModeScope, DEFAULT_KEY_ID, ENV_JEV_API_KEY, ENV_TYPESAFE_API_KEY,
};
use pi_jev::credential::{CredentialStore, SecretString};
use pi_jev::error::JevError;
use pi_jev::types::JevMode;

/// Canonical builtin command name (added to `core/slash_commands.rs`).
pub const JEV_COMMAND_NAME: &str = "jev";

/// Autocomplete argument hint. `on` is the short form of `compare`; only the
/// explicit `active` spelling selects the mode that changes a request.
pub const JEV_ARGUMENT_HINT: &str = "[off|compare|active|on|status|key]";

/// Autocomplete description. It names the modes, the key entry and status.
pub const JEV_COMMAND_DESCRIPTION: &str =
    "Jev comparison mode: Off, Compare (shadow-only), Active (applied to the next provider request), Input API key, Status";

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
pub const JEV_ACTIVE_NOTICE: &str = "Jev Active: an accepted answer is applied to the next provider request. The tool catalog is withdrawn for a request whose task needs no tools, and an already-set reasoning effort may move one step. A refused answer, failure or timeout leaves the request unchanged.";

/// Documented SystemOne endpoint (DESIGN.md section 2). Recorded for display only:
/// this UI lane never calls it.
pub const JEV_API_ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";

/// Documented default response model (DESIGN.md section 2).
pub const JEV_DEFAULT_MODEL: &str = "jev-latest";

/// The first-use disclosure wording (DESIGN.md data-handling requirement).
pub const JEV_DISCLOSURE_NOTICE: &str = "Disclosure: in Compare mode selected prompt/context excerpts are sent to TypeSafe \
for bounded, explicit decision categories. Redaction cannot guarantee that every confidential business item is found; \
treat data minimisation as the user's protection. In Compare mode nothing Jev returns is applied. In Active mode an \
accepted answer may change at most one field of one outgoing provider request, as described by /jev status; the primary \
model keeps full control.";

/// Section 11/12 boundary wording appended to help and status.
///
/// This is also the permanent limit of Active: the notice above is the whole of
/// what an accepted answer may change, and nothing here is ever in reach.
pub const JEV_BOUNDARY_NOTICE: &str = "Jev never controls the primary model, provider, permissions, context, memory, compaction, continuation, subagents, agent messages, depth, concurrency or budgets.";

/// One parsed `/jev` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JevRequest {
    /// `/jev` with no argument: open the menu.
    Menu,
    /// `/jev off`, `/jev compare`, `/jev on`, `/jev active`.
    SetMode(JevMode),
    /// `/jev status`.
    Status,
    /// `/jev key`: masked credential entry.
    InputKey,
    /// `/jev key clear`: delete the stored credential.
    ClearKey,
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
    match args.trim().to_ascii_lowercase().as_str() {
        "" => JevRequest::Menu,
        "off" => JevRequest::SetMode(JevMode::Off),
        "compare" | "on" => JevRequest::SetMode(JevMode::Compare),
        "active" => JevRequest::SetMode(JevMode::Active),
        "status" => JevRequest::Status,
        "key" | "key-input" => JevRequest::InputKey,
        "key clear" | "key-clear" | "clear-key" => JevRequest::ClearKey,
        "help" | "-h" | "--help" => JevRequest::Help,
        other => JevRequest::Unknown(other.to_string()),
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
        let number = |key: &str| value.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0);
        let text = |key: &str| value.get(key).and_then(serde_json::Value::as_str)
            .map(|text| pi_jev::correlate::sanitize_text(text, 120));
        Some(Self {
            applied: number("applied"),
            accepted_no_effect: number("accepted_no_effect"),
            refused: number("refused"),
            unavailable: number("unavailable"),
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
        let compare_counters = ["success_count", "failure_count", "queue_capacity"]
            .iter().all(|key| value.get(key).and_then(serde_json::Value::as_u64).is_some());
        if !compare_counters {
            // An Active-only snapshot has no scheduler counters. They stay at
            // their default (unknown) values instead of being invented as zero.
            return Self { active, ..Self::default() };
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
                    .take(11).collect()).unwrap_or_default(),
            fallback_reason: optional_text("fallback_reason").unwrap_or_default(),
            response_model: optional_text("response_model"),
            active,
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
        self.counters_known() || self.active.is_some()
    }
    /// True while Jev is working: this is what makes the footer amber/checking
    /// rather than green.
    pub fn checking(&self) -> bool {
        self.in_flight > 0
    }

    pub fn degraded(&self) -> bool {
        !self.fallback_reason.is_empty() || (self.failure_count > 0 && self.last_success_at.is_none())
    }
}

/// Everything `/jev status` needs. Assembled by the caller from the store plus
/// whatever the comparison pipeline exposes; no field requires a network call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JevStatusReport {
    pub mode: JevMode,
    pub scope: ModeScope,
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
}

impl JevStatusReport {
    /// The report for a UI that has no comparison-pipeline telemetry yet: the
    /// counters are truthful zeros and the note says so.
    pub fn local_only(mode: JevMode, scope: ModeScope, credential: CredentialStatus) -> Self {
        Self {
            mode,
            scope,
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
        }
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
        }
    ));
    if report.mode == JevMode::Active {
        text.push_str(&format!("{JEV_ACTIVE_NOTICE}\n"));
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
        if report.mode == JevMode::Off {
            "idle (Off: no scheduling, no client, no network)"
        } else if report.mode == JevMode::Active && !report.pipeline.known() {
            "unknown (no Active boundary observed in this worker yet)"
        } else if report.mode == JevMode::Active {
            "active (an accepted answer changes at most one request field)"
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
        if report.mode == JevMode::Active {
            report.pipeline.active.as_ref().map(|counters| counters.applied).unwrap_or(0)
        } else {
            report.applied_decisions
        },
        if report.mode == JevMode::Active {
            "Active counts one boundary per provider request whose body actually changed"
        } else if report.hypothetical_only {
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
    } else if report.mode == JevMode::Active {
        text.push_str(&format!(
            "Active boundaries: unknown ({JEV_ACTIVE_UNKNOWN_NOTE})\n"
        ));
    }
    text.push_str(&format!("Footer: {JEV_FOOTER_RULE_NOTICE}\n"));
    text.push('\n');
    text.push_str(JEV_DISCLOSURE_NOTICE);
    text.push_str("\n\n");
    text.push_str(JEV_BOUNDARY_NOTICE);
    text.push('\n');
    text
}

/// Help/hotkey text: the modes and their real scope, with no model or
/// subagent claims (sections 11 and 12).
pub fn render_help() -> String {
    let mut text = String::new();
    text.push_str(&format!("{JEV_COMMAND_DESCRIPTION}\n\n"));
    text.push_str("Off            Jev is disabled (default). No client, no scheduling, no network.\n");
    text.push_str("Compare        Shadow-only. Jev observes and records; Optimus alone decides.\n");
    text.push_str("Active         Operative. An accepted answer changes at most one field of the next\n");
    text.push_str("               provider request.\n");
    text.push_str(JEV_ACTIVE_NOTICE);
    text.push('\n');
    text.push_str("Input API key  Enter a TypeSafe key (masked; never stored in the transcript).\n");
    text.push_str("Status         Mode, scope, credential source, counters, queue, skips.\n");
    text.push_str(&format!("Footer         {JEV_FOOTER_RULE_NOTICE}\n\n"));
    text.push_str(&format!(
        "Commands: /{JEV_COMMAND_NAME}, /{JEV_COMMAND_NAME} off, /{JEV_COMMAND_NAME} compare, /{JEV_COMMAND_NAME} active, /{JEV_COMMAND_NAME} on, /{JEV_COMMAND_NAME} status, /{JEV_COMMAND_NAME} key, /{JEV_COMMAND_NAME} key clear, /{JEV_COMMAND_NAME} help\n"
    ));
    text.push_str("Note: /jev compare and its short form /jev on stay shadow-only and apply nothing.\n");
    text.push_str(JEV_ON_COMPARE_NOTICE);
    text.push_str("\n\n");
    text.push_str(JEV_BOUNDARY_NOTICE);
    text.push('\n');
    text
}

/// Human label for a mode. Active is a real mode, so it is labelled plainly; the
/// wording that bounds what it does lives in [`JEV_ACTIVE_NOTICE`].
pub fn mode_label(mode: JevMode) -> &'static str {
    match mode {
        JevMode::Off => "Off",
        JevMode::Compare => "Compare",
        JevMode::Active => "Active",
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
                ModeScope::GlobalDefault => "defaults for new chats",
                ModeScope::BuiltIn => "built-in default",
            },
            match mode {
                JevMode::Compare => format!("\n{JEV_DISCLOSURE_NOTICE}"),
                JevMode::Active => format!("\n{JEV_ACTIVE_NOTICE}\n{JEV_BOUNDARY_NOTICE}"),
                JevMode::Off => String::new(),
            },
        ),
    }
}

/// The outcome of one mode-set request.
///
/// `Applied` means the requested value was written to the store. Every mode is
/// writable, so there is no refusal variant: a mode is never silently rewritten
/// into another one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeChange {
    Applied { mode: JevMode, scope: ModeScope },
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

    /// The deciding scope, so status can say "this chat" or "defaults for new chats".
    pub fn scope(&self, session_id: &str) -> ModeScope {
        self.store.load().effective_mode_with_scope(session_id).scope
    }

    /// Write an explicit per-session mode. Every mode is written, including
    /// `Active`: the store is the only source of truth, and no request is
    /// silently redirected to a different mode.
    pub fn set_session_mode(
        &self,
        session_id: &str,
        requested: JevMode,
    ) -> Result<ModeChange, String> {
        let mut settings = self.store.load();
        settings.set_session_mode(session_id, requested);
        self.store.save(&settings).map_err(describe_error)?;
        Ok(ModeChange::Applied {
            mode: requested,
            scope: ModeScope::Session,
        })
    }

    /// Write the global default for sessions with no explicit value. `Active`
    /// is written like every other mode.
    pub fn set_global_default(&self, requested: JevMode) -> Result<ModeChange, String> {
        let mut settings = self.store.load();
        settings.global_default = Some(requested);
        self.store.save(&settings).map_err(describe_error)?;
        Ok(ModeChange::Applied {
            mode: requested,
            scope: ModeScope::GlobalDefault,
        })
    }

    /// Clear the explicit per-session mode so the global default applies again.
    pub fn clear_session_mode(&self, session_id: &str) -> Result<(), String> {
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

/// The footer states this release can show. Both Compare and Active are accent
/// (working) states; no state in this release is green.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevFooterState {
    Off,
    Compare,
    Active,
    Unavailable,
    Checking,
    Fallback,
}

impl JevFooterState {
    /// Text label. Text AND colour both carry the meaning.
    pub fn label(self) -> &'static str {
        match self {
            JevFooterState::Off => "Jev Off",
            JevFooterState::Compare => "Jev Compare",
            JevFooterState::Active => "Jev Active",
            JevFooterState::Unavailable => "Jev unavailable",
            JevFooterState::Checking => "Jev checking",
            JevFooterState::Fallback => "Jev fallback",
        }
    }

    /// Theme colour key. Green (`success`) is never returned here: Active is an
    /// accent state, not a green one.
    pub fn color_key(self) -> &'static str {
        match self {
            JevFooterState::Off => "error",
            JevFooterState::Compare | JevFooterState::Active => "accent",
            JevFooterState::Unavailable | JevFooterState::Checking | JevFooterState::Fallback => {
                "warning"
            }
        }
    }

    /// The dot glyph prefix.
    pub fn dot(self) -> &'static str {
        "\u{25cf}"
    }

    /// True only for a state that is both active and healthy-green. This release
    /// marks no state green, so the rule holds by construction and
    /// [`footer_color_key`] can never return `success`.
    pub fn is_green(self) -> bool {
        false
    }
}

/// Derive the footer state from mode + credential + pipeline truth.
///
/// Rules, in order:
/// * `Off` -> red dot + `Jev Off`, always. Off does no work, so the credential and
///   the pipeline cannot change that.
/// * `Active` -> accent `Jev Active` once the credential and the pipeline allow it;
///   otherwise the same amber unavailable / checking / fallback states Compare uses.
/// * `Compare` -> accent `Jev Compare` under the same credential and pipeline rules.
/// * No credential -> amber `Jev unavailable`.
/// * In-flight work -> amber `Jev checking`.
/// * Recorded fallback/failure -> amber `Jev fallback`.
/// Green is never produced.
pub fn footer_state(mode: JevMode, credential: &CredentialStatus, pipeline: &JevPipelineStatus) -> JevFooterState {
    let healthy = match mode {
        JevMode::Off => return JevFooterState::Off,
        JevMode::Compare => JevFooterState::Compare,
        JevMode::Active => JevFooterState::Active,
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



// ===========================================================================
// Menu rows and state machine (pure: external crates only, so the integration
// test in `tests/jev_ui_tests.rs` can include this file with `#[path]`)
// ===========================================================================

/// The five menu rows, in the order the brief fixes: Off, Compare, Active,
/// Input API key, Status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevMenuRow {
    Off,
    Compare,
    Active,
    InputKey,
    Status,
}

impl JevMenuRow {
    /// Declaration order; the selector walks this.
    pub const ALL: [JevMenuRow; 5] = [
        JevMenuRow::Off,
        JevMenuRow::Compare,
        JevMenuRow::Active,
        JevMenuRow::InputKey,
        JevMenuRow::Status,
    ];

    pub fn title(self, active: JevMode) -> String {
        let marker = match self {
            JevMenuRow::Off if active == JevMode::Off => " (current)",
            JevMenuRow::Compare if active == JevMode::Compare => " (current)",
            JevMenuRow::Active if active == JevMode::Active => " (current)",
            _ => "",
        };
        match self {
            // The row is the plain mode name: Active is a real, selectable mode.
            JevMenuRow::Off => format!("Off{marker}"),
            JevMenuRow::Compare => format!("Compare{marker}"),
            JevMenuRow::Active => format!("Active{marker}"),
            JevMenuRow::InputKey => "Input API key".to_string(),
            JevMenuRow::Status => "Status".to_string(),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            JevMenuRow::Off => "Disable Jev. No client, no scheduling, no network.",
            JevMenuRow::Compare => "Shadow-only observations; no decisions applied.",
            JevMenuRow::Active => JEV_ACTIVE_NOTICE,
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

    pub fn rows(&self) -> [JevMenuRow; 5] {
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

/// Documentation of the footer colour rule (asserted by the tests). The callers
/// prefix it with `Footer: `, so it does not repeat that word itself.
pub const JEV_FOOTER_RULE_NOTICE: &str =
    "red \"Jev Off\", accent \"Jev Compare\", accent \"Jev Active\", amber \"Jev unavailable\"/\"Jev checking\"/\"Jev fallback\". No green \"Jev On\" state is produced by this release.";

/// The theme colour key for the footer segment.
///
/// This is the one place the no-green rule is enforced in the LIVE path:
/// `success` is returned only when [`JevFooterState::is_green`] says the state is
/// a green one, and that method is a constant `false` in this release. Active is
/// an accent state, so a green footer would require changing the pure module.
pub fn footer_color_key(state: JevFooterState) -> &'static str {
    if state.is_green() {
        "success"
    } else {
        state.color_key()
    }
}

/// The `setStatus` payload that publishes the footer segment.
///
/// `native_host.rs:2327-2331` routes it into `extension_surfaces.set_status`, and
/// `native_host_extensions::Statuses` renders it directly under the model/effort
/// tray, so the segment needs no new host surface. Kept as a `serde_json::Value`
/// so the tests pin the exact payload the host receives. The live publisher in
/// `jev_footer.rs` uses the labelled form, because the dispatch task cannot
/// measure the terminal; this measured form is the contract for a caller that can.
#[allow(dead_code)]
pub fn footer_status_payload(state: JevFooterState, terminal_columns: usize) -> serde_json::Value {
    serde_json::json!({
        "statusKey": JEV_STATUS_KEY,
        "statusText": footer_segment(state, terminal_columns),
    })
}

/// The `setStatus` payload that removes the footer segment.
pub fn footer_clear_payload() -> serde_json::Value {
    serde_json::json!({
        "statusKey": JEV_STATUS_KEY,
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
