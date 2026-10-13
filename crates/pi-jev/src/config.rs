//! Mode + settings + credential-source resolution (DESIGN.md sections 0, 3.1, 10.2, 11).
//!
//! Precedence (binding, DESIGN.md 10.2):
//!   resolve_effective_mode(explicit_session, inherited_or_global_default)
//!       = explicit_session.or(global_default).unwrap_or(Off)
//! Explicit per-session mode always wins. API-key presence never influences any mode.
//! Feature and compaction controls resolve independently of mode.
//!
//! Credential source order (binding, DESIGN.md 3.1): saved > TYPESAFE_API_KEY > JEV_API_KEY.
//! Key values are never written to the settings file; only key-presence metadata is.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Session control. Credentials never enable a mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JevMode {
    Off,
    Compare,
    Active,
    #[serde(rename = "compare-active", alias = "compare_and_active", alias = "compare-and-active")]
    CompareAndActive,
}

impl JevMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Compare => "compare",
            Self::Active => "active",
            Self::CompareAndActive => "compare-active",
        }
    }

    /// `on` always selects shadow-only Compare.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "false" | "no" | "disabled" => Some(Self::Off),
            "compare" | "comparison" | "on" | "true" | "yes" | "shadow" => Some(Self::Compare),
            "active" | "enable" => Some(Self::Active),
            "compare-active" | "compare-and-active" | "compare_and_active" | "both" => Some(Self::CompareAndActive),
            _ => None,
        }
    }

    pub fn allows_compare(self) -> bool {
        matches!(self, Self::Compare | Self::CompareAndActive)
    }

    pub fn allows_active(self) -> bool {
        matches!(self, Self::Active | Self::CompareAndActive)
    }

    pub fn is_enabled(self) -> bool {
        self != Self::Off
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Jev Off",
            Self::Compare => "Jev Compare",
            Self::Active => "Jev Active",
            Self::CompareAndActive => "Jev Compare + Active",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Off => "Jev decision mode is off. Independent compaction keeps its own setting.",
            Self::Compare => "Jev Compare records recommendations only. Nothing is applied.",
            Self::Active => "Jev Active applies accepted, feature-gated decisions at bounded native boundaries. Failure leaves the baseline unchanged.",
            Self::CompareAndActive => "Jev Compare + Active records comparisons and applies accepted, feature-gated decisions from the same boundary request.",
        }
    }
}

/// Operator gates for Active effects and supplemental observations. These never
/// remove the existing Compare categories or grant model/permission control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct JevFeatures {
    pub tool_requirement: bool,
    pub complexity: bool,
    pub tool_candidates: bool,
    pub context_relevance: bool,
    pub code_search_relevance: bool,
    pub code_search_filtering: bool,
    /// Additive semantic recall; normal memory retrieval always runs alongside it.
    pub memory: bool,
    pub memory_relevance: bool,
    pub result_sufficiency: bool,
    pub loop_control: bool,
    pub retry_classification: bool,
    pub verification: bool,
    pub trace_observer: bool,
    /// Full-jev search flag (ROOT-CONTRACT v1): re-rank code-search candidates
    /// with real per-query scores. Defaults to `false`; the full-jev overlay
    /// forces it on.
    pub code_search_reranking: bool,
    /// Full-jev search flag (ROOT-CONTRACT v1): line-level semantic find over
    /// the actual native source-read path. Defaults to `false`; the full-jev
    /// overlay forces it on.
    pub line_find: bool,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): advisory skill hint from the
    /// already-loaded roster. Defaults to `false`; the full-jev overlay
    /// forces it on. Never loads or executes a skill.
    pub skill_suggestion: bool,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): input guardrail battery.
    /// Defaults to `false`; the full-jev overlay forces it on.
    pub guardrails_input: bool,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): output guardrail battery
    /// (post-hoc assessment). Defaults to `false`; the full-jev overlay
    /// forces it on.
    pub guardrails_output: bool,
    /// ROOT-CONTRACT v6 (Evidence lane): candidate-bound retrieval-safety
    /// battery over the code-search filter request. Defaults to `false`; the
    /// full-jev overlay forces it on.
    pub retrieval_safety: bool,
    /// ROOT-CONTRACT v6 (Evidence lane): advisory citation check over the
    /// actual supplied source span. Defaults to `false`; the full-jev
    /// overlay forces it on.
    pub citation_check: bool,
    /// Agent-authored typed questions through the jev_decide tool.
    pub dynamic: bool,
}

impl Default for JevFeatures {
    fn default() -> Self {
        Self {
            tool_requirement: true,
            complexity: true,
            tool_candidates: false,
            context_relevance: false,
            code_search_relevance: false,
            code_search_filtering: false,
            memory: false,
            memory_relevance: false,
            result_sufficiency: false,
            loop_control: false,
            retry_classification: false,
            verification: false,
            trace_observer: false,
            code_search_reranking: false,
            line_find: false,
            skill_suggestion: false,
            guardrails_input: false,
            guardrails_output: false,
            retrieval_safety: false,
            citation_check: false,
            dynamic: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JevFeature {
    ToolRequirement,
    Complexity,
    ToolCandidates,
    ContextRelevance,
    CodeSearchRelevance,
    CodeSearchFiltering,
    Memory,
    MemoryRelevance,
    ResultSufficiency,
    LoopControl,
    RetryClassification,
    Verification,
    TraceObserver,
    CodeSearchReranking,
    LineFind,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): skill suggestion hint.
    SkillSuggestion,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): input guardrail battery.
    GuardrailsInput,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): output guardrail battery.
    GuardrailsOutput,
    /// ROOT-CONTRACT v6 (Evidence lane): retrieval-safety battery.
    RetrievalSafety,
    /// ROOT-CONTRACT v6 (Evidence lane): citation check.
    CitationCheck,
    Dynamic,
}

impl JevFeature {
    pub const ALL: [Self; 21] = [
        Self::ToolRequirement,
        Self::Complexity,
        Self::ToolCandidates,
        Self::ContextRelevance,
        Self::CodeSearchRelevance,
        Self::CodeSearchFiltering,
        Self::Memory,
        Self::MemoryRelevance,
        Self::ResultSufficiency,
        Self::LoopControl,
        Self::RetryClassification,
        Self::Verification,
        Self::TraceObserver,
        Self::CodeSearchReranking,
        Self::LineFind,
        Self::SkillSuggestion,
        Self::GuardrailsInput,
        Self::GuardrailsOutput,
        Self::RetrievalSafety,
        Self::CitationCheck,
        Self::Dynamic,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolRequirement => "tool_requirement",
            Self::Complexity => "complexity",
            Self::ToolCandidates => "tool_candidates",
            Self::ContextRelevance => "context_relevance",
            Self::CodeSearchRelevance => "code_search_relevance",
            Self::CodeSearchFiltering => "code_search_filtering",
            Self::Memory => "memory",
            Self::MemoryRelevance => "memory_relevance",
            Self::ResultSufficiency => "result_sufficiency",
            Self::LoopControl => "loop_control",
            Self::RetryClassification => "retry_classification",
            Self::Verification => "verification",
            Self::TraceObserver => "trace_observer",
            Self::CodeSearchReranking => "code_search_reranking",
            Self::LineFind => "line_find",
            Self::SkillSuggestion => "skill_suggestion",
            Self::GuardrailsInput => "guardrails_input",
            Self::GuardrailsOutput => "guardrails_output",
            Self::RetrievalSafety => "retrieval_safety",
            Self::CitationCheck => "citation_check",
            Self::Dynamic => "dynamic",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
        Self::ALL.into_iter().find(|feature| feature.as_str() == normalized)
    }
}

impl JevFeatures {
    pub fn enabled(self, feature: JevFeature) -> bool {
        match feature {
            JevFeature::ToolRequirement => self.tool_requirement,
            JevFeature::Complexity => self.complexity,
            JevFeature::ToolCandidates => self.tool_candidates,
            JevFeature::ContextRelevance => self.context_relevance,
            JevFeature::CodeSearchRelevance => self.code_search_relevance,
            JevFeature::CodeSearchFiltering => self.code_search_filtering,
            JevFeature::Memory => self.memory,
            JevFeature::MemoryRelevance => self.memory_relevance,
            JevFeature::ResultSufficiency => self.result_sufficiency,
            JevFeature::LoopControl => self.loop_control,
            JevFeature::RetryClassification => self.retry_classification,
            JevFeature::Verification => self.verification,
            JevFeature::TraceObserver => self.trace_observer,
            JevFeature::CodeSearchReranking => self.code_search_reranking,
            JevFeature::LineFind => self.line_find,
            JevFeature::SkillSuggestion => self.skill_suggestion,
            JevFeature::GuardrailsInput => self.guardrails_input,
            JevFeature::GuardrailsOutput => self.guardrails_output,
            JevFeature::RetrievalSafety => self.retrieval_safety,
            JevFeature::CitationCheck => self.citation_check,
            JevFeature::Dynamic => self.dynamic,
        }
    }

    pub fn set(&mut self, feature: JevFeature, enabled: bool) {
        match feature {
            JevFeature::ToolRequirement => self.tool_requirement = enabled,
            JevFeature::Complexity => self.complexity = enabled,
            JevFeature::ToolCandidates => self.tool_candidates = enabled,
            JevFeature::ContextRelevance => self.context_relevance = enabled,
            JevFeature::CodeSearchRelevance => self.code_search_relevance = enabled,
            JevFeature::CodeSearchFiltering => self.code_search_filtering = enabled,
            JevFeature::Memory => self.memory = enabled,
            JevFeature::MemoryRelevance => self.memory_relevance = enabled,
            JevFeature::ResultSufficiency => self.result_sufficiency = enabled,
            JevFeature::LoopControl => self.loop_control = enabled,
            JevFeature::RetryClassification => self.retry_classification = enabled,
            JevFeature::Verification => self.verification = enabled,
            JevFeature::TraceObserver => self.trace_observer = enabled,
            JevFeature::CodeSearchReranking => self.code_search_reranking = enabled,
            JevFeature::LineFind => self.line_find = enabled,
            JevFeature::SkillSuggestion => self.skill_suggestion = enabled,
            JevFeature::GuardrailsInput => self.guardrails_input = enabled,
            JevFeature::GuardrailsOutput => self.guardrails_output = enabled,
            JevFeature::RetrievalSafety => self.retrieval_safety = enabled,
            JevFeature::CitationCheck => self.citation_check = enabled,
            JevFeature::Dynamic => self.dynamic = enabled,
        }
    }
}

impl fmt::Display for JevMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Saved-credential key id used by both the client lane and the UI lane.
pub const DEFAULT_KEY_ID: &str = "typesafe";

/// Primary environment variable name.
pub const ENV_TYPESAFE_API_KEY: &str = "TYPESAFE_API_KEY";

/// Documented alias, consulted only when the primary variable is absent.
pub const ENV_JEV_API_KEY: &str = "JEV_API_KEY";

/// Agent-dir override used by the repo for isolated state.
pub const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";

/// Built-in default agent directory (mirrors the repo's `.prime/agent`).
pub const DEFAULT_AGENT_DIR_NAME: &str = ".prime/agent";

/// Settings file name inside the agent dir.
pub const SETTINGS_FILE_NAME: &str = "jev-settings.json";

/// Schema version of the persisted settings file.
pub const SETTINGS_SCHEMA_VERSION: u32 = 1;

/// Where the effective credential came from. Never carries the secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialSource {
    /// A key from the credential store. Wins over every environment variable.
    Saved,
    /// `TYPESAFE_API_KEY`.
    EnvTypesafe,
    /// `JEV_API_KEY` (alias; used only when `TYPESAFE_API_KEY` is absent).
    EnvJev,
    /// Nothing configured.
    None,
}

impl CredentialSource {
    /// Stable name for status output.
    pub fn as_str(self) -> &'static str {
        match self {
            CredentialSource::Saved => "saved",
            CredentialSource::EnvTypesafe => ENV_TYPESAFE_API_KEY,
            CredentialSource::EnvJev => ENV_JEV_API_KEY,
            CredentialSource::None => "none",
        }
    }

    /// True when a key of this source exists.
    pub fn is_configured(self) -> bool {
        !matches!(self, CredentialSource::None)
    }
}

/// Which scope supplied a resolved setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeScope {
    /// An explicit value chosen for this session.
    Session,
    /// The global default used when no session override exists.
    GlobalDefault,
    /// The built-in default (Off).
    BuiltIn,
    /// The `/jev full-jev` overlay, which resolves ABOVE every session,
    /// global and inherited override while it is installed.
    FullJevOverlay,
}

impl ModeScope {
    pub fn as_str(self) -> &'static str {
        match self {
            ModeScope::Session => "session",
            ModeScope::GlobalDefault => "global_default",
            ModeScope::BuiltIn => "built_in_default",
            ModeScope::FullJevOverlay => "full_jev_overlay",
        }
    }
}

/// The `/jev full-jev` global overlay (ROOT-CONTRACT v1): a persisted, named
/// profile that resolves ABOVE every per-session, global and inherited
/// override while it is installed. The base settings — `global_default`,
/// `features`, `compaction_enabled` and the whole `sessions` map — are never
/// rewritten by full-on or full-off, so removing the overlay restores the
/// pre-existing resolution exactly.
///
/// Resolution reads the FIXED [`FULL_JEV_MODE`], [`FULL_JEV_FEATURES`] and
/// [`FULL_JEV_COMPACTION_ENABLED`] constants, never these persisted fields,
/// so a hand-edited or downgraded file can never weaken what "full" means;
/// the stored fields exist for truthful status and forward compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullJevProfile {
    /// True while the overlay is installed. Removal retains the block with
    /// `enabled: false`; `None` on `JevSettings` means the profile was never
    /// installed. Resolution and status treat both states as "off".
    pub enabled: bool,
    /// The operative mode the overlay resolves to: always [`FULL_JEV_MODE`].
    pub mode: JevMode,
    /// Every registered feature gate the overlay forces on.
    pub features: JevFeatures,
    /// Independent compaction the overlay forces on.
    pub compaction_enabled: bool,
    /// Install counter for the cheap change stamp: every full-on/full-off
    /// cycle moves it, so cached work stamped with an older revision can be
    /// invalidated the same way a credential rotation is. No secrets.
    pub revision: u64,
}

/// The mode the full-jev overlay resolves to. Not a new `JevMode` variant
/// (ROOT-CONTRACT v1): the overlay means combined mode.
pub const FULL_JEV_MODE: JevMode = JevMode::CompareAndActive;

/// Every feature gate the full-jev overlay forces on, including the two
/// full-jev flags that default to `false` outside the overlay.
pub const FULL_JEV_FEATURES: JevFeatures = JevFeatures {
    tool_requirement: true,
    complexity: true,
    tool_candidates: true,
    context_relevance: true,
    code_search_relevance: true,
    code_search_filtering: true,
    memory: true,
    memory_relevance: true,
    result_sufficiency: true,
    loop_control: true,
    retry_classification: true,
    verification: true,
    trace_observer: true,
    code_search_reranking: true,
    line_find: true,
    skill_suggestion: true,
    guardrails_input: true,
    guardrails_output: true,
    retrieval_safety: true,
    citation_check: true,
    dynamic: true,
};

/// Compaction under the full-jev overlay: on, still request-local and
/// independent of the decision mode.
pub const FULL_JEV_COMPACTION_ENABLED: bool = true;

/// Result of mode resolution, including which scope decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeResolution {
    pub mode: JevMode,
    pub scope: ModeScope,
}

/// BINDING precedence (DESIGN.md 10.2): explicit session > inherited/global default > Off.
///
/// An explicit per-session `Off` wins over a global `Compare`, and an explicit per-session
/// `Compare` wins over a global `Off`. Credentials are not an input here at all.
pub fn resolve_effective_mode(
    explicit_session: Option<JevMode>,
    inherited_or_global_default: Option<JevMode>,
) -> JevMode {
    resolve_mode(explicit_session, inherited_or_global_default).mode
}

/// Same resolution with the deciding scope reported.
pub fn resolve_mode(
    explicit_session: Option<JevMode>,
    inherited_or_global_default: Option<JevMode>,
) -> ModeResolution {
    match explicit_session {
        Some(mode) => ModeResolution {
            mode,
            scope: ModeScope::Session,
        },
        None => match inherited_or_global_default {
            Some(mode) => ModeResolution {
                mode,
                scope: ModeScope::GlobalDefault,
            },
            None => ModeResolution {
                mode: JevMode::Off,
                scope: ModeScope::BuiltIn,
            },
        },
    }
}

/// BINDING credential order (DESIGN.md 3.1): saved > TYPESAFE_API_KEY > JEV_API_KEY.
pub fn resolve_credential_source(
    saved_present: bool,
    env_typesafe_present: bool,
    env_jev_present: bool,
) -> CredentialSource {
    if saved_present {
        CredentialSource::Saved
    } else if env_typesafe_present {
        CredentialSource::EnvTypesafe
    } else if env_jev_present {
        CredentialSource::EnvJev
    } else {
        CredentialSource::None
    }
}

/// Resolves the effective credential source from a saved-presence flag and env presence.
///
/// Convenience wrapper so callers do not re-implement the documented order.
pub fn resolve_credential_source_from_presence(
    saved_present: bool,
    env: EnvKeyPresence,
) -> CredentialSource {
    resolve_credential_source(saved_present, env.typesafe_api_key, env.jev_api_key)
}

/// One-line credential status for `/jev status`. Never contains the secret.
///
/// When both environment variables are present and no saved credential exists, the line says
/// explicitly that `TYPESAFE_API_KEY` wins and the alias is ignored. When a saved credential
/// exists, the line says the environment variables are ignored entirely.
pub fn credential_status_line(saved_present: bool, env: EnvKeyPresence) -> String {
    let source = resolve_credential_source_from_presence(saved_present, env);
    let mut line = match source {
        CredentialSource::Saved => "credential: saved".to_string(),
        CredentialSource::EnvTypesafe => {
            format!("credential: {ENV_TYPESAFE_API_KEY}")
        }
        CredentialSource::EnvJev => {
            format!("credential: {ENV_JEV_API_KEY} (alias; {ENV_TYPESAFE_API_KEY} absent)")
        }
        CredentialSource::None => "credential: none (set /jev to add one)".to_string(),
    };
    if env.has_conflict() && source != CredentialSource::Saved {
        line.push_str(&format!(
            "; both environment variables are set, {ENV_TYPESAFE_API_KEY} wins and {ENV_JEV_API_KEY} is ignored"
        ));
    }
    if env.has_conflict() && source == CredentialSource::Saved {
        line.push_str(&format!(
            "; environment variables are ignored while a saved credential exists ({ENV_TYPESAFE_API_KEY}, {ENV_JEV_API_KEY})"
        ));
    }
    line
}

/// Which environment variables are present. Booleans only; values are never read here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvKeyPresence {
    pub typesafe_api_key: bool,
    pub jev_api_key: bool,
}

impl EnvKeyPresence {
    /// Reads key PRESENCE from the process environment. Values are never copied or logged.
    pub fn from_env() -> Self {
        Self {
            typesafe_api_key: env_var_non_empty(ENV_TYPESAFE_API_KEY),
            jev_api_key: env_var_non_empty(ENV_JEV_API_KEY),
        }
    }

    /// True when the primary variable is present and the alias is also present.
    /// Status says so explicitly, because the primary wins.
    pub fn has_conflict(&self) -> bool {
        self.typesafe_api_key && self.jev_api_key
    }
}

fn env_var_non_empty(name: &str) -> bool {
    std::env::var(name).map(|value| !value.trim().is_empty()).unwrap_or(false)
}

/// Per-session mode override. Absent means "inherit / use the global default".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedSessionMode {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<JevMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<JevFeatures>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_enabled: Option<bool>,
    /// Opaque parent session id when the value was inherited at child creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<String>,
}

/// Persisted settings. Contains NO secret material; only presence metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JevSettings {
    /// Read generation for checked saves; never part of the persisted schema.
    #[serde(skip)]
    pub loaded_generation: Option<SettingsGeneration>,
    pub schema_version: u32,
    /// Default for sessions without an explicit mode. `None` means built-in Off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_default: Option<JevMode>,
    #[serde(default)]
    pub features: JevFeatures,
    #[serde(default)]
    pub compaction_enabled: bool,
    #[serde(default)]
    pub compaction: crate::compaction::CompactionConfig,
    #[serde(default)]
    pub filtering: crate::filtering::FilteringOptions,
    /// Per-session explicit overrides, keyed by opaque session id.
    #[serde(default)]
    pub sessions: std::collections::BTreeMap<String, PersistedSessionMode>,
    /// Advisory metadata only: whether a saved credential existed at last write.
    #[serde(default)]
    pub credential_configured: bool,
    /// Advisory metadata only: which source was effective at last write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_source: Option<CredentialSource>,
    /// Whether the first-use disclosure has been shown.
    #[serde(default)]
    pub disclosure_shown: bool,
    /// Development/test-only transport selector among the crate's built-in
    /// deterministic mock transports ("mock", "mock-hostile", "mock-delayed",
    /// "mock-malformed"). `None` (production default) uses the saved
    /// credential against the real endpoint; a mock never reads a credential.
    /// This selects a transport ONLY; it never changes a mode and never
    /// grants any capability (DESIGN.md 11/12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
    /// The `/jev full-jev` overlay: absent while the profile was never
    /// installed; RETAINED with `enabled: false` after `/jev full-jev off` so
    /// the persisted `revision` keeps its identity role across off->on cycles.
    /// See [`FullJevProfile`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_jev: Option<FullJevProfile>,
    /// ROOT-CONTRACT v9 durable write identity: every authoritative save
    /// persists the previous file's revision + 1 (1 for the first write).
    /// Monotonic across processes; a reader that misses intermediate writes
    /// still observes a strictly larger revision on return, so work or hints
    /// stamped with an older revision are rejected even for A->B->A where the
    /// serialized VALUES return to identical bytes. Hand-edited files keep
    /// their stated revision; only the save path advances authority.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub write_revision: u64,
    /// ROOT-CONTRACT v9: the explicitly requested Jev SystemOne model for this
    /// agent dir, set by `/jev model set <id>` and removed by `/jev model
    /// reset`. `None` means the built-in native default
    /// [`crate::types::DEFAULT_MODEL`] (`jev-latest`). This selects the
    /// `model` field of JEV SystemOne requests ONLY: it is never the user's
    /// primary chat model, provider or effort (DESIGN.md section 11), it is
    /// never selected automatically (no startup probe, no catalog auto-pick,
    /// no live call), and the `/jev full-jev` overlay neither reads, writes
    /// nor masks it — an explicit operator selection is independent and
    /// deliberate. Model writes never refill or reset control budgets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model: Option<String>,
}

/// serde helper for the durable write revision default skip.
fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

impl Default for JevSettings {
    fn default() -> Self {
        Self {
            loaded_generation: None,
            schema_version: SETTINGS_SCHEMA_VERSION,
            global_default: None,
            features: JevFeatures::default(),
            compaction_enabled: false,
            compaction: crate::compaction::CompactionConfig::default(),
            filtering: crate::filtering::FilteringOptions::default(),
            sessions: std::collections::BTreeMap::new(),
            credential_configured: false,
            credential_source: None,
            disclosure_shown: false,
            transport: None,
            full_jev: None,
            write_revision: 0,
            requested_model: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsGeneration(Option<[u8; 32]>);

fn settings_generation(bytes: Option<&[u8]>) -> SettingsGeneration {
    SettingsGeneration(bytes.map(|bytes| Sha256::digest(bytes).into()))
}

impl JevSettings {
    /// True when any scope resolves to an operative mode. Both `Compare`
    /// (records only) and `Active` (may apply accepted answers) need the
    /// observer and its client; `Off` never does.
    pub fn wants_observer(&self) -> bool {
        if self.full_jev_active() {
            return true;
        }
        let operative = |mode: Option<JevMode>| mode.is_some_and(JevMode::is_enabled);
        if operative(self.global_default) {
            return true;
        }
        self.sessions.values().any(|entry| operative(entry.mode))
    }

    pub fn validate(&self) -> Result<(), crate::error::JevError> {
        self.compaction.validate().map_err(|_| crate::error::JevError::config("invalid Jev compaction settings"))?;
        self.filtering.validate().map_err(|_| crate::error::JevError::config("invalid Jev filtering settings"))?;
        // ROOT-CONTRACT v9: a persisted requested-model id must satisfy the
        // SAME exact-identifier rule the catalog parser and the setter use.
        // A hand-edited hostile value (control characters, bidi controls,
        // ambiguous/invisible whitespace, credential echoes, empty, over-cap)
        // makes the WHOLE file load as defaults — the documented corrupt-file
        // fail-safe: the native default `jev-latest` then governs, nothing
        // blocks startup, and no value is normalized into a different
        // selectable id. The bounded reason never echoes the hostile text.
        if let Some(id) = self.requested_model.as_deref() {
            crate::models::validate_requested_model_id(id).map_err(|reason| {
                crate::error::JevError::config(format!(
                    "invalid requested Jev model selection: {reason}"
                ))
            })?;
        }
        Ok(())
    }

    /// Settings with an explicit global default.
    pub fn with_global_default(mode: JevMode) -> Self {
        Self {
            global_default: Some(mode),
            ..Self::default()
        }
    }

    /// Explicit per-session override, if any.
    pub fn session_mode(&self, session_id: &str) -> Option<JevMode> {
        self.sessions.get(session_id).and_then(|entry| entry.mode)
    }

    /// The session's SAVED compaction decision, if any. This is the raw
    /// persisted value, not the overlay-aware resolution.
    pub fn session_compaction_enabled(&self, session_id: &str) -> Option<bool> {
        self.sessions
            .get(session_id)
            .and_then(|entry| entry.compaction_enabled)
    }

    /// Sets an explicit per-session override.
    pub fn set_session_mode(&mut self, session_id: &str, mode: JevMode) {
        self.sessions
            .entry(session_id.to_string())
            .or_default()
            .mode = Some(mode);
    }

    /// Clears an explicit per-session override, so the session falls back to the default.
    pub fn clear_session_mode(&mut self, session_id: &str) {
        if let Some(entry) = self.sessions.get_mut(session_id) {
            entry.mode = None;
        }
    }

    /// True while the full-jev overlay is installed and active. Cheap: one
    /// `Option` check, no allocation, callable on hot paths.
    pub fn full_jev_active(&self) -> bool {
        self.full_jev
            .as_ref()
            .is_some_and(|profile| profile.enabled)
    }

    /// Stable cheap change stamp for the full-jev overlay: presence plus the
    /// install revision, e.g. `full-jev:0` / `full-jev:3`. Equality on this
    /// string answers "did the overlay state change since this was stamped?"
    /// with no secrets and no hashing on the hot path. A full toggle must
    /// invalidate cached decisions the same way a credential rotation does
    /// (ROOT-CONTRACT v1), so the bridge folds this into its cheap stamp.
    pub fn full_jev_stamp(&self) -> String {
        match &self.full_jev {
            Some(profile) if profile.enabled => format!("full-jev:{}", profile.revision),
            _ => "full-jev:0".to_string(),
        }
    }

    /// Install the full-jev overlay. Idempotent: an already-active profile is
    /// left untouched and `false` is returned, so nothing is rewritten and no
    /// revision is churned. Each fresh activation bumps `revision`, which the
    /// removal path RETAINS disabled, so every off->on cycle yields a new
    /// persisted activation identity (no ABA for stamped cached work, even
    /// when a consumer misses the intermediate Off). The base settings fields
    /// are NEVER touched; the documented one-way exception is the first-use
    /// notice flag.
    /// Returns `true` when this call installed the overlay.
    pub fn full_jev_install(&mut self) -> bool {
        if self.full_jev_active() {
            return false;
        }
        let revision = self
            .full_jev
            .as_ref()
            .map_or(1, |profile| profile.revision + 1);
        self.full_jev = Some(FullJevProfile {
            enabled: true,
            mode: FULL_JEV_MODE,
            features: FULL_JEV_FEATURES,
            compaction_enabled: FULL_JEV_COMPACTION_ENABLED,
            revision,
        });
        self.disclosure_shown = true;
        true
    }

    /// Remove the full-jev overlay. Idempotent: returns `false` when it was
    /// not active. The profile block is RETAINED DISABLED instead of deleted:
    /// its persisted `revision` survives off->on cycles, so every fresh
    /// activation gets a new durable identity (an off->on cycle never repeats a
    /// stamp; a consumer that misses the Off cannot accept first-activation
    /// work in the second activation). Resolution ignores the block while
    /// disabled, so the pre-overlay resolution returns exactly and no base
    /// field is touched.
    /// Returns `true` when this call deactivated the overlay.
    pub fn full_jev_remove(&mut self) -> bool {
        match &mut self.full_jev {
            Some(profile) if profile.enabled => {
                profile.enabled = false;
                true
            }
            _ => false,
        }
    }

    /// The explicit per-session decisions the overlay is currently masking,
    /// as opaque session ids, for truthful status. Empty when the overlay is
    /// not active. Masked values are still SAVED — they resolve again the
    /// moment the overlay is removed.
    pub fn full_jev_masked_sessions(&self) -> Vec<&str> {
        if !self.full_jev_active() {
            return Vec::new();
        }
        self.sessions
            .iter()
            .filter(|(_, entry)| {
                entry.mode.is_some()
                    || entry.features.is_some()
                    || entry.compaction_enabled.is_some()
            })
            .map(|(session_id, _)| session_id.as_str())
            .collect()
    }

    /// Effective mode for a session. See `resolve_effective_mode`. The active
    /// full-jev overlay resolves ABOVE every session, global and inherited
    /// override.
    pub fn effective_mode(&self, session_id: &str) -> JevMode {
        if self.full_jev_active() {
            return FULL_JEV_MODE;
        }
        resolve_effective_mode(self.session_mode(session_id), self.global_default)
    }

    /// Effective mode with the deciding scope.
    pub fn effective_mode_with_scope(&self, session_id: &str) -> ModeResolution {
        if self.full_jev_active() {
            return ModeResolution {
                mode: FULL_JEV_MODE,
                scope: ModeScope::FullJevOverlay,
            };
        }
        resolve_mode(self.session_mode(session_id), self.global_default)
    }

    /// Effective features for a session. The active full-jev overlay forces
    /// every gate on, even where saved session settings say off.
    pub fn effective_features(&self, session_id: &str) -> JevFeatures {
        if self.full_jev_active() {
            return FULL_JEV_FEATURES;
        }
        self.sessions
            .get(session_id)
            .and_then(|entry| entry.features)
            .unwrap_or(self.features)
    }

    /// The session's own effective mode with the full-jev overlay EXCLUDED:
    /// the baseline a child inherits or a masked-override status reports.
    /// Overlay settings are never materialized into children or the base
    /// fields (ROOT-CONTRACT v1).
    pub fn base_effective_mode(&self, session_id: &str) -> JevMode {
        resolve_effective_mode(self.session_mode(session_id), self.global_default)
    }

    /// Overlay-excluded features baseline. See [`Self::base_effective_mode`].
    pub fn base_effective_features(&self, session_id: &str) -> JevFeatures {
        self.sessions
            .get(session_id)
            .and_then(|entry| entry.features)
            .unwrap_or(self.features)
    }

    /// Overlay-excluded compaction baseline. See [`Self::base_effective_mode`].
    pub fn base_effective_compaction_enabled(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .and_then(|entry| entry.compaction_enabled)
            .unwrap_or(self.compaction_enabled)
    }

    pub fn set_session_feature(&mut self, session_id: &str, feature: JevFeature, enabled: bool) {
        // Snapshot the BASE features: overlay values must never be
        // materialized into a session entry (ROOT-CONTRACT v1).
        let mut features = self.base_effective_features(session_id);
        features.set(feature, enabled);
        self.sessions
            .entry(session_id.to_string())
            .or_default()
            .features = Some(features);
    }

    /// Resolves compaction independently of the Jev decision mode.
    pub fn effective_compaction_enabled(&self, session_id: &str) -> bool {
        self.effective_compaction_with_scope(session_id).0
    }

    pub fn effective_compaction_with_scope(&self, session_id: &str) -> (bool, ModeScope) {
        if self.full_jev_active() {
            return (FULL_JEV_COMPACTION_ENABLED, ModeScope::FullJevOverlay);
        }
        match self
            .sessions
            .get(session_id)
            .and_then(|entry| entry.compaction_enabled)
        {
            Some(enabled) => (enabled, ModeScope::Session),
            None => (self.compaction_enabled, ModeScope::GlobalDefault),
        }
    }

    pub fn set_session_compaction_enabled(&mut self, session_id: &str, enabled: bool) {
        self.sessions.entry(session_id.to_string()).or_default().compaction_enabled = Some(enabled);
    }

    /// Records key presence metadata. Never stores a key value.
    pub fn set_credential_metadata(&mut self, configured: bool, source: CredentialSource) {
        self.credential_configured = configured;
        self.credential_source = Some(source);
    }

    /// ROOT-CONTRACT v9: the requested Jev model native SystemOne requests
    /// carry — the explicit `/jev model set <id>` selection when present,
    /// otherwise the documented native default `jev-latest`
    /// ([`crate::types::DEFAULT_MODEL`]). Resolution is PURE and LOCAL: no
    /// network, no probe, no availability claim. Callers MUST capture this
    /// value from the SAME authoritative settings snapshot they resolve the
    /// mode/gates/revision from, BEFORE any await, and must not re-read it
    /// late into an already-captured payload.
    pub fn requested_model_or_default(&self) -> &str {
        self.requested_model.as_deref().unwrap_or(crate::types::DEFAULT_MODEL)
    }

    /// Sets the requested Jev model to an already-captured operator value
    /// (the `/jev model set <id>` path). Validates with the SAME
    /// exact-identifier rule the catalog parser uses
    /// ([`crate::models::validate_requested_model_id`]); the bounded refusal
    /// reason NEVER contains the supplied text. Returns `Ok(true)` when the
    /// value changed and `Ok(false)` when it was already set (an idempotent
    /// no-op: the caller's save is then skipped, so the durable write
    /// revision does not move). Credential-overlap protection against the
    /// EFFECTIVE credential is a CALLER duty
    /// ([`crate::models::id_overlaps_credential`]) because this method never
    /// loads a credential. Zero network, zero probe, no availability claim,
    /// no budget effect, no overlay interaction.
    pub fn set_requested_model(&mut self, raw: &str) -> Result<bool, crate::error::JevError> {
        crate::models::validate_requested_model_id(raw).map_err(|reason| {
            crate::error::JevError::config(format!("requested Jev model refused: {reason}"))
        })?;
        let changed = self.requested_model.as_deref() != Some(raw);
        if changed {
            self.requested_model = Some(raw.to_string());
        }
        Ok(changed)
    }

    /// Removes the explicit requested-model selection (the `/jev model reset`
    /// tombstone): the native default `jev-latest` governs again. Returns
    /// `true` when an explicit selection was actually removed and `false`
    /// when none was set (the caller's save is then skipped). No network, no
    /// probe, no budget effect, no overlay interaction.
    pub fn clear_requested_model(&mut self) -> bool {
        self.requested_model.take().is_some()
    }

    /// True when the serialized form looks like it carries credential material.
    ///
    /// `JevSettings` has no field able to hold a key; this guard exists so a later field
    /// cannot silently introduce one, and it is asserted by the settings tests.
    pub fn looks_like_it_contains_a_secret(&self) -> bool {
        let Ok(serialized) = serde_json::to_string(self) else {
            return false;
        };
        let lower = serialized.to_ascii_lowercase();
        if lower.contains("bearer ") {
            return true;
        }
        const FORBIDDEN_KEYS: [&str; 7] = [
            "\"api_key\"",
            "\"apikey\"",
            "\"api-key\"",
            "\"secret\"",
            "\"token\"",
            "\"password\"",
            "\"authorization\"",
        ];
        if FORBIDDEN_KEYS.iter().any(|key| lower.contains(key)) {
            return true;
        }
        // Session ids stay opaque and id-shaped; anything else is treated as secret material.
        self.sessions
            .keys()
            .chain(self.sessions.values().filter_map(|entry| entry.inherited_from.as_ref()))
            .any(|id| !is_opaque_local_id(id))
    }
}

/// Accepts only opaque local ids (session ids): bounded, id-shaped, no whitespace.
fn is_opaque_local_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Child inheritance: the parent's EFFECTIVE mode is snapshotted at creation time.
///
/// An explicit child override wins and is recorded; otherwise the child stores the parent's
/// effective mode as its own explicit value, so a later global change cannot silently alter
/// an existing chat. (DESIGN.md section 0: existing chats must not silently change.)
pub fn inherit_mode(
    settings: &mut JevSettings,
    child_session_id: &str,
    parent_session_id: &str,
    explicit_child_override: Option<JevMode>,
) -> JevMode {
    // Baseline inheritance (ROOT-CONTRACT v1): the child snapshots the
    // parent's overlay-EXCLUDED effective values, so full-jev settings are
    // never materialized into the child's saved entry. While the overlay is
    // active the child still RESOLVES through it like every other session;
    // removing the overlay returns the child to this inherited baseline.
    let inherited =
        explicit_child_override.unwrap_or_else(|| settings.base_effective_mode(parent_session_id));
    let features = settings.base_effective_features(parent_session_id);
    let compaction_enabled = settings.base_effective_compaction_enabled(parent_session_id);
    let entry = settings.sessions.entry(child_session_id.to_string()).or_default();
    entry.features.get_or_insert(features);
    entry.compaction_enabled.get_or_insert(compaction_enabled);
    entry.mode = Some(inherited);
    entry.inherited_from = Some(parent_session_id.to_string());
    inherited
}

/// Agent dir for Jev state: the injected value wins, otherwise the documented env override,
/// otherwise `$HOME/.prime/agent`.
pub fn default_agent_dir(injected: Option<&Path>) -> PathBuf {
    if let Some(dir) = injected {
        return dir.to_path_buf();
    }
    if let Ok(value) = std::env::var(ENV_AGENT_DIR) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(DEFAULT_AGENT_DIR_NAME)
}

/// Jev state directory inside the agent dir.
pub fn jev_dir_for(agent_dir: impl AsRef<Path>) -> PathBuf {
    agent_dir.as_ref().join("jev")
}

/// Cheap change identity of a settings file: data modification time in
/// nanoseconds since the Unix epoch, plus the file length. RELIES ON THE
/// WRITE DISCIPLINE: saves go through a fresh temp file + atomic rename, so
/// every save leaves a new mtime behind, and NTFS stamps writes at 100 ns
/// granularity - two different contents colliding on BOTH (mtime_ns, len)
/// is therefore assumed impossible. On filesystems with coarser timestamps
/// (e.g. FAT's 2 s) that assumption is weaker; the agent dir is not expected
/// to live on one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SettingsFileStat {
    mtime_ns: u64,
    len: u64,
}

fn settings_file_stat(metadata: &std::fs::Metadata) -> Option<SettingsFileStat> {
    // Unreadable or pre-epoch mtimes cannot be trusted to move on rewrite,
    // so they opt out of the cache entirely (always a full read).
    let since_epoch = metadata
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some(SettingsFileStat {
        mtime_ns: since_epoch.as_nanos() as u64,
        len: metadata.len(),
    })
}

/// One cached settings snapshot behind interior mutability, shared by every
/// [`JevSettingsStore`] instance for the same settings file. The
/// `(stat, snapshot)` pair swap happens under one mutex, so a reader
/// observes either the old or the new pair, never a mix.
struct SharedSettingsCache {
    entry: Mutex<Option<(SettingsFileStat, Arc<JevSettings>)>>,
    /// Full read+parse passes served by this cache (unit-test evidence only).
    #[cfg(test)]
    full_loads: std::sync::atomic::AtomicU64,
}

impl SharedSettingsCache {
    fn shared() -> Arc<Self> {
        Arc::new(Self {
            entry: Mutex::new(None),
            #[cfg(test)]
            full_loads: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Drops the cached snapshot. Called after every write attempt through
    /// any store sharing this cache (safety rule: a store must never serve
    /// its own pre-save view; the stat check in `load` covers every other
    /// writer, including other processes).
    fn invalidate(&self) {
        *self.entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }
}

/// Live per-file caches. The hot call sites build a fresh store per call
/// (`JevSettingsStore::new(get_agent_dir()).load()`), so a strictly
/// per-object cache could never serve them; the registry lets every store
/// instance for one settings path share one cache instead. Keys are the raw
/// store paths: a key miss (two spellings of one file) only duplicates the
/// cached snapshot, never correctness - every load still stats the real
/// file. `Weak` values let an unused cache be reclaimed with its last store.
static SETTINGS_CACHES: OnceLock<Mutex<std::collections::HashMap<PathBuf, Weak<SharedSettingsCache>>>> =
    OnceLock::new();

fn settings_cache_for_path(path: &Path) -> Arc<SharedSettingsCache> {
    let registry = SETTINGS_CACHES.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut caches = registry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cache) = caches.get(path).and_then(Weak::upgrade) {
        return cache;
    }
    // Opportunistic sweep: long-lived processes (and the test binary's many
    // temp dirs) never accumulate dead keys.
    caches.retain(|_, cache| cache.strong_count() > 0);
    let cache = SharedSettingsCache::shared();
    caches.insert(path.to_path_buf(), Arc::downgrade(&cache));
    cache
}

/// Settings store bound to one agent dir. `agent_dir` is injected to keep it testable and to
/// keep every test's state inside its own temporary directory.
///
/// Change-aware cache: every instance for the same settings file shares one
/// cached snapshot (see [`SETTINGS_CACHES`]). `load` re-reads and re-parses
/// the file only when its `(mtime_ns, len)` identity moves, and `save` drops
/// the snapshot so a store never serves its own pre-save view.
pub struct JevSettingsStore {
    path: PathBuf,
    cache: Arc<SharedSettingsCache>,
}

impl Clone for JevSettingsStore {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            cache: Arc::clone(&self.cache),
        }
    }
}

impl fmt::Debug for JevSettingsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors the pre-cache derive output: the cached snapshot itself
        // would be megabytes of log noise.
        f.debug_struct("JevSettingsStore")
            .field("path", &self.path)
            .finish()
    }
}

impl JevSettingsStore {
    /// Store for an explicit agent dir.
    pub fn new(agent_dir: impl AsRef<Path>) -> Self {
        Self::from_path(jev_dir_for(agent_dir).join(SETTINGS_FILE_NAME))
    }

    /// Store for an explicit settings file path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        Self {
            cache: settings_cache_for_path(&path),
            path,
        }
    }

    /// Settings file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Loads settings. A missing file yields defaults; a corrupt file yields defaults and
    /// never panics (a broken settings file must not block startup).
    ///
    /// Change-aware cache: the file is stat'ed first, and a `(mtime_ns,
    /// len)` that matches the cached snapshot returns a clone of it,
    /// skipping the full read + parse + validate. Safety rules, each
    /// covered by tests: (a) a stat that differs NEVER serves the cache, so
    /// a disable or credential revocation is never masked; (b) a malformed
    /// or unreadable file keeps the defaults behavior and is never cached,
    /// so the next good write is always seen; (c) a missing file yields
    /// defaults and clears the cache; (d) a save through any store sharing
    /// this cache invalidates it; (e) the snapshot is only cached when the
    /// stat is stable across the read (double-stat), so a write racing this
    /// load can never leave old bytes paired with a new stat.
    pub fn load(&self) -> JevSettings {
        // Fast path: stat only.
        let stat_before = std::fs::metadata(&self.path).ok().as_ref().and_then(settings_file_stat);
        if let Some(stat) = stat_before {
            let cached = self.cache
                .entry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
                .and_then(|(cached_stat, snapshot)| {
                    (*cached_stat == stat).then(|| (**snapshot).clone())
                });
            if let Some(settings) = cached {
                return settings;
            }
        }
        #[cfg(test)]
        self.cache.full_loads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Slow path: the original full read + parse + validate.
        let bytes = std::fs::read(&self.path).ok();
        let parsed = bytes
            .as_deref()
            .and_then(|bytes| serde_json::from_slice::<JevSettings>(bytes).ok())
            .filter(|settings| settings.validate().is_ok());
        let mut settings = parsed.clone().unwrap_or_default();
        settings.loaded_generation = Some(settings_generation(bytes.as_deref()));
        // Cache refresh, per the safety rules above.
        match std::fs::metadata(&self.path) {
            Ok(metadata) => {
                if let Some(after) = settings_file_stat(&metadata) {
                    if parsed.is_some() && stat_before == Some(after) {
                        *self.cache.entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
                            Some((after, Arc::new(settings.clone())));
                    }
                }
            }
            // A missing file yields defaults and clears the cache: the next
            // good write (recreating the file) must never be masked.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => self.cache.invalidate(),
            // Unreadable metadata: defaults, cache untouched (never cached).
            Err(_) => {}
        }
        settings
    }

    /// Saves settings. Refuses to write a file that appears to contain a secret.
    pub fn save(&self, settings: &JevSettings) -> Result<(), crate::error::JevError> {
        settings.validate()?;
        if settings.looks_like_it_contains_a_secret() {
            return Err(crate::error::JevError::CredentialStore {
                detail: "refusing to persist settings that appear to contain a secret".to_string(),
            });
        }
        let Some(parent) = self.path.parent() else {
            return Err(crate::error::JevError::config("settings path has no parent directory"));
        };
        std::fs::create_dir_all(parent).map_err(|error| {
            crate::error::JevError::config(format!("cannot create settings dir: {}", error.kind()))
        })?;
        // Keep a stable lock inode: deleting this file would allow two processes
        // to acquire different locks. Dropping the handle releases a crashed writer.
        let lock = std::fs::OpenOptions::new().read(true).write(true).create(true)
            .truncate(false).open(self.path.with_extension("json.lock"))
            .map_err(|_| crate::error::JevError::config("cannot open settings lock"))?;
        lock.try_lock().map_err(|_| crate::error::JevError::config("settings busy; retry the change"))?;
        let current = match std::fs::read(&self.path) {
            Ok(bytes) => {
                serde_json::from_slice::<JevSettings>(&bytes)
                    .map_err(|_| crate::error::JevError::config("settings are corrupt; refusing to overwrite"))?
                    .validate()?;
                Some(bytes)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(_) => return Err(crate::error::JevError::config("cannot read settings; refusing to overwrite")),
        };
        if settings.loaded_generation.as_ref().is_some_and(|generation|
            *generation != settings_generation(current.as_deref())) {
            return Err(crate::error::JevError::config("settings changed; reload and retry the change"));
        }
        // ROOT-CONTRACT v9: every authoritative save advances the durable
        // write revision (previous file's revision + 1; 1 when absent). This
        // is the settings identity stamps compare against: it moves on EVERY
        // authoritative write, including model A->B->A and reset tombstones,
        // regardless of whether the serialized values change.
        let previous_revision = current.as_deref()
            .and_then(|bytes| serde_json::from_slice::<JevSettings>(bytes).ok())
            .map(|settings| settings.write_revision)
            .unwrap_or(0);
        // The caller's value stays untouched; the PERSISTED file carries the
        // advanced revision, observed by the next authoritative load.
        let mut persisted = settings.clone();
        persisted.write_revision = previous_revision + 1;
        let serialized = serde_json::to_vec_pretty(&persisted).map_err(|_| {
            crate::error::JevError::config("cannot serialize settings")
        })?;
        let temp = self.path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = std::fs::File::create_new(&temp)?;
            file.write_all(&serialized)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, &self.path)
        })();
        // SAFETY RULE (d): every completed write attempt drops the cached
        // snapshot for this file, so this store (and every store sharing its
        // cache) never serves its own pre-save view. Writers in OTHER
        // processes are covered by the stat check in `load`: the atomic
        // rename leaves a fresh mtime behind.
        self.cache.invalidate();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result.map_err(|error| crate::error::JevError::config(format!("cannot save settings: {}", error.kind())))
    }
}

/// Convenience: shared settings store for an agent dir.
pub fn settings_store(agent_dir: impl AsRef<Path>) -> Arc<JevSettingsStore> {
    Arc::new(JevSettingsStore::new(agent_dir))
}

/// Safety rules for the change-aware settings cache in [`JevSettingsStore`].
/// Each test maps to one rule from the load/save doc comments.
#[cfg(test)]
mod settings_cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// Full read+parse passes served by the shared cache of this path.
    fn full_loads(store: &JevSettingsStore) -> u64 {
        store.cache.full_loads.load(Ordering::Relaxed)
    }

    fn snapshot_is_cached(store: &JevSettingsStore) -> bool {
        store
            .cache
            .entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// Writes the file directly, like an operator or an external process.
    fn write_file(path: &Path, contents: &[u8]) {
        std::fs::create_dir_all(path.parent().expect("settings parent")).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn unchanged_file_is_served_from_the_cache_without_reparse() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let mut settings = JevSettings::default();
        settings.set_session_mode("alpha", JevMode::Compare);
        store.save(&settings).unwrap();
        let first = store.load();
        assert_eq!(first.session_mode("alpha"), Some(JevMode::Compare));
        let baseline = full_loads(&store);
        assert!(baseline >= 1, "the first load after a save is a full read");
        assert!(snapshot_is_cached(&store), "a good load primes the cache");
        // 50 loads of an unchanged file: same value, zero full re-parses.
        for _ in 0..50 {
            let cached = store.load();
            assert_eq!(cached.session_mode("alpha"), Some(JevMode::Compare));
        }
        assert_eq!(
            full_loads(&store),
            baseline,
            "an unchanged (mtime_ns, len) must skip the full read+parse"
        );
        // A rewrite (new mtime) re-parses and serves the fresh value.
        let mut next = store.load();
        next.set_session_mode("alpha", JevMode::Off);
        store.save(&next).unwrap();
        let after = store.load();
        assert_eq!(after.session_mode("alpha"), Some(JevMode::Off));
        assert!(
            full_loads(&store) > baseline,
            "a changed stat must fall back to the full read"
        );
    }

    #[test]
    fn fresh_store_instances_for_one_path_share_the_cache() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let mut settings = JevSettings::default();
        settings.set_session_mode("alpha", JevMode::Compare);
        store.save(&settings).unwrap();
        let _warm = store.load();
        let baseline = full_loads(&store);
        // The hot callers build a fresh store per call; they must hit the
        // same shared cache instead of re-parsing every time.
        for _ in 0..10 {
            let hot_caller = JevSettingsStore::new(directory.path());
            assert_eq!(hot_caller.load().session_mode("alpha"), Some(JevMode::Compare));
        }
        assert_eq!(
            full_loads(&store),
            baseline,
            "fresh instances on the same path must be served by the shared cache"
        );
    }

    #[test]
    fn disable_after_enable_is_visible_across_store_instances() {
        let directory = tempfile::tempdir().unwrap();
        let store_a = JevSettingsStore::new(directory.path());
        let store_b = JevSettingsStore::new(directory.path());
        let mut enable = JevSettings::default();
        enable.set_session_mode("alpha", JevMode::Compare);
        store_b.save(&enable).unwrap();
        assert_eq!(
            store_a.load().session_mode("alpha"),
            Some(JevMode::Compare),
            "an enable through B is seen by A"
        );
        // B disables; A's cached enable must NEVER be served again.
        let mut disable = store_b.load();
        disable.set_session_mode("alpha", JevMode::Off);
        store_b.save(&disable).unwrap();
        assert_eq!(
            store_a.load().session_mode("alpha"),
            Some(JevMode::Off),
            "a disable must not be masked by A's cache"
        );
    }

    #[test]
    fn malformed_file_is_never_cached_and_a_fix_is_seen() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        write_file(store.path(), b"broken settings");
        for _ in 0..3 {
            let loaded = store.load();
            assert_eq!(
                loaded.effective_mode("alpha"),
                JevMode::Off,
                "a malformed file yields defaults"
            );
        }
        assert!(
            !snapshot_is_cached(&store),
            "malformed content must never be cached"
        );
        assert!(
            full_loads(&store) >= 3,
            "every malformed load is a real read, so the next good write is observable"
        );
        // An operator fixes the file by hand; the next load must pick it up.
        let mut fixed = JevSettings::default();
        fixed.set_session_mode("alpha", JevMode::Active);
        write_file(store.path(), &serde_json::to_vec_pretty(&fixed).unwrap());
        let healed = store.load();
        assert_eq!(healed.session_mode("alpha"), Some(JevMode::Active));
        assert!(snapshot_is_cached(&store), "the healed file is cacheable again");
    }

    #[test]
    fn missing_file_yields_defaults_clears_the_cache_then_a_new_file_is_seen() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let mut settings = JevSettings::default();
        settings.set_session_mode("alpha", JevMode::Compare);
        store.save(&settings).unwrap();
        assert!(store.load().session_mode("alpha").is_some());
        assert!(snapshot_is_cached(&store), "a good load primes the cache");
        std::fs::remove_file(store.path()).unwrap();
        let missing = store.load();
        assert_eq!(
            missing.effective_mode("alpha"),
            JevMode::Off,
            "a missing file yields defaults"
        );
        assert!(missing.sessions.is_empty());
        assert!(
            !snapshot_is_cached(&store),
            "a missing file clears the cache"
        );
        // A NEW file at the same path must be picked up, never stale state.
        let mut fresh = JevSettings::default();
        fresh.set_session_mode("alpha", JevMode::Active);
        store.save(&fresh).unwrap();
        let reloaded = store.load();
        assert_eq!(reloaded.session_mode("alpha"), Some(JevMode::Active));
    }

    #[test]
    fn save_invalidates_the_stores_own_cache_view() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let other = JevSettingsStore::new(directory.path());
        let mut settings = JevSettings::default();
        settings.set_session_mode("alpha", JevMode::Compare);
        store.save(&settings).unwrap();
        assert_eq!(store.load().session_mode("alpha"), Some(JevMode::Compare));
        let mut next = store.load();
        next.set_session_mode("alpha", JevMode::Off);
        store.save(&next).unwrap();
        assert_eq!(
            store.load().session_mode("alpha"),
            Some(JevMode::Off),
            "save-then-load on the same store returns fresh values"
        );
        assert_eq!(
            other.load().session_mode("alpha"),
            Some(JevMode::Off),
            "a second instance sharing the cache sees the save too"
        );
    }

    #[test]
    fn concurrent_loads_and_saves_stay_coherent() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let mut base = JevSettings::default();
        base.set_session_mode("alpha", JevMode::Off);
        store.save(&base).unwrap();
        // A snapshot loaded BEFORE the race; saving it afterwards must be
        // refused by the generation guard.
        let stale = store.load();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer = {
            let store = store.clone();
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                for round in 0..40u64 {
                    let mode = if round % 2 == 0 { JevMode::Compare } else { JevMode::Off };
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    loop {
                        let mut settings = store.load();
                        settings.set_session_mode("alpha", mode);
                        match store.save(&settings) {
                            Ok(()) => break,
                            Err(_) if std::time::Instant::now() < deadline => continue,
                            Err(error) => panic!("writer failed to settle: {error}"),
                        }
                    }
                }
                stop.store(true, Ordering::SeqCst);
            })
        };
        let mut readers = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            let stop = Arc::clone(&stop);
            readers.push(std::thread::spawn(move || {
                let mut last_revision = 0u64;
                while !stop.load(Ordering::SeqCst) {
                    let settings = store.load();
                    // No torn state: each load is one coherent snapshot with
                    // a valid schema and a mode/revision pair that could have
                    // been produced by one single save (base: Off at
                    // revision 1, then Compare at even, Off at odd).
                    assert!(settings.validate().is_ok());
                    let mode = settings.session_mode("alpha").expect("session always set");
                    let revision = settings.write_revision;
                    match mode {
                        JevMode::Compare => assert_eq!(
                            revision % 2,
                            0,
                            "Compare snapshots carry an even revision"
                        ),
                        JevMode::Off => assert_eq!(
                            revision % 2,
                            1,
                            "Off snapshots carry an odd revision"
                        ),
                        other => panic!("unexpected mode: {other:?}"),
                    }
                    assert!(
                        revision >= last_revision,
                        "revisions never move backwards for one reader"
                    );
                    last_revision = revision;
                }
            }));
        }
        writer.join().expect("writer thread");
        for reader in readers {
            reader.join().expect("reader thread");
        }
        // The generation guard still rejects the pre-race snapshot.
        let refused = store.save(&stale);
        assert!(
            refused.is_err(),
            "the generation guard must reject the stale pre-race save"
        );
        // The final state is a coherent last write.
        let final_settings = store.load();
        assert!(matches!(
            final_settings.session_mode("alpha"),
            Some(JevMode::Compare) | Some(JevMode::Off)
        ));
        assert!(final_settings.validate().is_ok());
    }
}

#[cfg(test)]
mod full_jev_tests {
    use super::*;

    /// A settings value with saved decisions the overlay must mask:
    /// session "alpha" explicitly Off with compaction off and verification
    /// explicitly on, plus a global default of Compare.
    fn saved_baseline() -> JevSettings {
        let mut settings = JevSettings::default();
        settings.global_default = Some(JevMode::Compare);
        settings.set_session_mode("alpha", JevMode::Off);
        settings.set_session_compaction_enabled("alpha", false);
        settings.set_session_feature("alpha", JevFeature::Verification, true);
        settings
    }

    #[test]
    fn full_jev_install_resolves_above_every_saved_override() {
        let mut settings = saved_baseline();
        assert!(settings.full_jev_install());
        // The overlay masks the explicit Off, the explicit compaction off and
        // the global default: everything resolves to the fixed constants.
        assert_eq!(settings.effective_mode("alpha"), JevMode::CompareAndActive);
        assert_eq!(
            settings.effective_mode_with_scope("alpha").scope,
            ModeScope::FullJevOverlay
        );
        for feature in JevFeature::ALL {
            assert!(
                settings.effective_features("alpha").enabled(feature),
                "{}",
                feature.as_str()
            );
        }
        assert!(settings.effective_features("alpha").code_search_reranking);
        assert!(settings.effective_features("alpha").line_find);
        assert!(settings.effective_compaction_enabled("alpha"));
        assert_eq!(
            settings.effective_compaction_with_scope("alpha").1,
            ModeScope::FullJevOverlay
        );
        assert!(settings.wants_observer());
    }

    #[test]
    fn full_jev_remove_restores_saved_resolution_exactly() {
        let mut settings = saved_baseline();
        let before = settings.clone();
        assert!(settings.full_jev_install());
        assert!(settings.full_jev_remove());
        // The base fields and the sessions map are untouched by full-on/off,
        // so removal restores the pre-existing resolution byte-for-byte.
        assert_eq!(settings.global_default, before.global_default);
        assert_eq!(settings.features, before.features);
        assert_eq!(settings.compaction_enabled, before.compaction_enabled);
        assert_eq!(settings.sessions, before.sessions);
        assert_eq!(settings.effective_mode("alpha"), JevMode::Off);
        assert!(!settings.effective_compaction_enabled("alpha"));
        assert!(settings.effective_features("alpha").verification);
        assert!(!settings.effective_features("alpha").code_search_reranking);
        assert!(!settings.effective_features("alpha").line_find);
        assert_eq!(
            settings.effective_mode_with_scope("alpha").scope,
            ModeScope::Session
        );
    }

    #[test]
    fn full_jev_install_is_idempotent_and_keeps_one_revision() {
        let mut settings = JevSettings::default();
        assert!(settings.full_jev_install());
        let revision = settings.full_jev.as_ref().expect("profile").revision;
        assert!(settings.disclosure_shown);
        assert!(!settings.full_jev_install());
        assert_eq!(
            settings.full_jev.as_ref().expect("profile").revision,
            revision
        );
        assert_eq!(settings.full_jev_stamp(), format!("full-jev:{revision}"));
    }

    #[test]
    fn full_jev_remove_without_profile_is_a_truthful_noop() {
        let mut settings = saved_baseline();
        let before = settings.clone();
        assert!(!settings.full_jev_remove());
        assert_eq!(settings, before);
        assert_eq!(settings.full_jev_stamp(), "full-jev:0");
    }

    #[test]
    fn full_jev_revision_and_stamp_move_across_cycles() {
        let mut settings = JevSettings::default();
        let stamps = |settings: &JevSettings| {
            (
                settings.full_jev_stamp(),
                settings.full_jev.as_ref().map(|profile| profile.revision),
            )
        };
        assert_eq!(settings.full_jev_stamp(), "full-jev:0");
        assert!(settings.full_jev_install());
        let (on_one, rev_one) = stamps(&settings);
        assert_eq!(rev_one, Some(1));
        assert!(settings.full_jev_remove());
        assert_eq!(settings.full_jev_stamp(), "full-jev:0");
        assert!(settings.full_jev_install());
        let (on_two, rev_two) = stamps(&settings);
        assert_eq!(rev_two, Some(2));
        assert_ne!(on_one, on_two);
    }

    #[test]
    fn full_jev_masked_sessions_reports_saved_decisions_only() {
        let mut settings = saved_baseline();
        assert!(settings.full_jev_masked_sessions().is_empty());
        assert!(settings.full_jev_install());
        let masked = settings.full_jev_masked_sessions();
        assert_eq!(masked, vec!["alpha"]);
        // Inherited-only entries with no explicit value stay unreported:
        // nothing of theirs is masked.
        settings
            .sessions
            .entry("beta".to_string())
            .or_default()
            .inherited_from = Some("alpha".to_string());
        assert_eq!(settings.full_jev_masked_sessions(), vec!["alpha"]);
        assert!(settings.full_jev_remove());
        assert!(settings.full_jev_masked_sessions().is_empty());
    }

    #[test]
    fn full_jev_never_materializes_into_children_or_session_entries() {
        let mut settings = saved_baseline();
        assert!(settings.full_jev_install());
        // The child inherits the parent's BASELINE, not the overlay values.
        let inherited = inherit_mode(&mut settings, "child", "alpha", None);
        assert_eq!(inherited, JevMode::Off);
        let child = settings.sessions.get("child").expect("child entry");
        assert_eq!(child.mode, Some(JevMode::Off));
        // alpha's saved baseline has verification ON, and the child inherits
        // the BASELINE, never the overlay values.
        assert!(child.features.expect("features").verification);
        assert_eq!(child.compaction_enabled, Some(false));
        // While the overlay is active the child still RESOLVES through it...
        assert_eq!(settings.effective_mode("child"), JevMode::CompareAndActive);
        assert!(settings.effective_compaction_enabled("child"));
        // ...and falls back to the inherited baseline once it is removed.
        assert!(settings.full_jev_remove());
        assert_eq!(settings.effective_mode("child"), JevMode::Off);
        assert!(!settings.effective_compaction_enabled("child"));
    }

    #[test]
    fn full_jev_set_session_feature_snapshots_the_baseline() {
        let mut settings = JevSettings::default();
        assert!(settings.full_jev_install());
        // A feature write during the overlay records a BASE features value,
        // never the overlay's forced-on set.
        settings.set_session_feature("alpha", JevFeature::ToolCandidates, true);
        let saved = settings
            .sessions
            .get("alpha")
            .expect("entry")
            .features
            .expect("features");
        assert!(saved.tool_candidates);
        assert!(!saved.code_search_reranking);
        assert!(!saved.line_find);
        // The snapshot is the BASE features: tool_requirement defaults TRUE
        // (JevFeatures::default), so it is snapshotted TRUE, not overlay-on.
        assert!(saved.tool_requirement);
    }

    #[test]
    fn new_search_flags_default_off_and_round_trip_by_name() {
        let features = JevFeatures::default();
        assert!(!features.code_search_reranking);
        assert!(!features.line_find);
        assert_eq!(
            JevFeature::parse("code_search_reranking"),
            Some(JevFeature::CodeSearchReranking)
        );
        assert_eq!(
            JevFeature::parse("code-search-reranking"),
            Some(JevFeature::CodeSearchReranking)
        );
        assert_eq!(JevFeature::parse("line_find"), Some(JevFeature::LineFind));
        assert_eq!(JevFeature::parse("line-find"), Some(JevFeature::LineFind));
        assert_eq!(JevFeature::parse("nope"), None);
        let serialized = serde_json::to_value(features).unwrap();
        let fields = serialized.as_object().unwrap();
        assert_eq!(JevFeature::ALL.len(), fields.len());
        // FULL_JEV_FEATURES turns every gate on, including the new two.
        for feature in JevFeature::ALL {
            assert!(fields.contains_key(feature.as_str()));
            assert_eq!(JevFeature::parse(feature.as_str()), Some(feature));
            assert!(FULL_JEV_FEATURES.enabled(feature), "{}", feature.as_str());
        }
        assert_eq!(FULL_JEV_MODE, JevMode::CompareAndActive);
        assert!(FULL_JEV_COMPACTION_ENABLED);
    }

    #[test]
    fn full_jev_settings_survive_store_reload_and_absent_field_loads_as_none() {
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let mut settings = saved_baseline();
        assert!(settings.full_jev_install());
        store.save(&settings).unwrap();
        let loaded = store.load();
        assert!(loaded.full_jev_active());
        assert_eq!(loaded.effective_mode("alpha"), JevMode::CompareAndActive);
        assert_eq!(loaded.full_jev.as_ref().expect("profile").revision, 1);
        // Old-reader tolerance: a file without the field loads with None.
        let mut legacy = serde_json::to_value(&settings).unwrap();
        assert!(legacy.as_object_mut().unwrap().remove("full_jev").is_some());
        std::fs::write(
            directory.path().join("jev").join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&legacy).unwrap(),
        )
        .unwrap();
        let old_loaded = store.load();
        assert!(!old_loaded.full_jev_active());
        assert!(old_loaded.full_jev.is_none());
        assert_eq!(old_loaded.effective_mode("alpha"), JevMode::Off);
        // skip_serializing_if: a None overlay never writes the field.
        assert!(!serde_json::to_string(&JevSettings::default())
            .unwrap()
            .contains("full_jev"));
    }

    #[test]
    fn concurrent_writers_through_separate_stores_settle_on_one_generation() {
        // ROOT-CONTRACT v1: prove atomic/conflict-safe writes. Two snapshots
        // LOADED from ONE starting generation race through separate store
        // instances; exactly one save wins, the other is refused by the
        // generation check, and the bounded-retry discipline (reload,
        // re-apply, save) lands the loser's change WITHOUT overwriting the
        // winner's fields.
        let directory = tempfile::tempdir().unwrap();
        let store_a = JevSettingsStore::new(directory.path());
        let store_b = JevSettingsStore::new(directory.path());
        let mut base = saved_baseline();
        store_a.save(&base).unwrap();
        // Both writers hold loaded snapshots of the SAME generation.
        let mut writer_a = store_a.load();
        let mut writer_b = store_b.load();
        assert!(writer_a.full_jev_install(), "A installs the overlay");
        writer_b.global_default = Some(JevMode::Off);
        // One save wins the generation race...
        store_a.save(&writer_a).expect("the first save wins");
        // ...the other is refused, never silently overwriting the winner.
        let refused = store_b.save(&writer_b);
        assert!(
            refused.is_err(),
            "the moved generation must refuse the stale save"
        );
        // The bounded retry: reload, re-apply, save.
        let mut retried = store_b.load();
        assert!(retried.full_jev_active(), "the winner's overlay survived");
        retried.global_default = Some(JevMode::Off);
        store_b.save(&retried).expect("the fresh retry saves");
        // BOTH changes are on disk; neither overwrote the other.
        let settled = store_a.load();
        assert!(settled.full_jev_active());
        assert_eq!(settled.global_default, Some(JevMode::Off));
        assert_eq!(settled.effective_mode("alpha"), JevMode::CompareAndActive);
    }

    #[test]
    fn stale_loaded_snapshot_is_refused_until_the_retry_reloads() {
        // Sequential form of the conflict proof: a LOADED snapshot (with a
        // real loaded generation, unlike an in-memory construct) is refused
        // after another writer moves the generation, then the reload-retry
        // succeeds and preserves the winner's overlay.
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        let mut base = saved_baseline();
        store.save(&base).unwrap();
        // A snapshot LOADED at the starting generation.
        let mut stale = store.load();
        // Another writer moves the generation (the overlay toggle).
        let mut writer = store.load();
        assert!(writer.full_jev_install());
        store.save(&writer).unwrap();
        // The stale save is refused...
        assert!(
            store.save(&stale).is_err(),
            "a stale LOADED snapshot must not overwrite the winner"
        );
        // ...and the bounded-retry discipline re-applies it cleanly.
        let mut retried = store.load();
        retried.global_default = Some(JevMode::Off);
        store.save(&retried).unwrap();
        let settled = store.load();
        assert!(settled.full_jev_active());
        assert_eq!(settled.global_default, Some(JevMode::Off));
    }

    #[test]
    fn full_jev_activation_identity_is_fresh_across_off_on_and_reloads() {
        // Root B1 (activation-stamp ABA): an off->on cycle must NEVER repeat
        // a stamp, even for a consumer that never observes the intermediate
        // Off, and the identity must survive store reloads.
        let directory = tempfile::tempdir().unwrap();
        let store = JevSettingsStore::new(directory.path());
        store.save(&JevSettings::default()).unwrap();
        // First activation.
        let mut activation_one = store.load();
        assert!(activation_one.full_jev_install());
        let stamp_one = activation_one.full_jev_stamp();
        assert_eq!(stamp_one, "full-jev:1");
        store.save(&activation_one).unwrap();
        // Idempotent already-on: the stamp is stable, nothing is rewritten.
        let mut already_on = store.load();
        assert!(!already_on.full_jev_install());
        assert_eq!(already_on.full_jev_stamp(), stamp_one);
        // The Off cycle, observed only through later reloads.
        let mut off_cycle = store.load();
        assert!(off_cycle.full_jev_remove());
        store.save(&off_cycle).unwrap();
        let after_off = store.load();
        assert!(!after_off.full_jev_active());
        assert_eq!(after_off.full_jev_stamp(), "full-jev:0");
        // Second activation: fresh persisted identity, visible to a consumer
        // that captured stamp_one and never saw the Off state.
        let mut activation_two = store.load();
        assert!(activation_two.full_jev_install());
        let stamp_two = activation_two.full_jev_stamp();
        assert_ne!(stamp_two, stamp_one, "no activation may repeat a stamp");
        assert_ne!(stamp_two, "full-jev:0");
        store.save(&activation_two).unwrap();
        assert_eq!(
            store.load().full_jev_stamp(),
            stamp_two,
            "identity survives reloads"
        );
        // Third cycle proves the identity keeps moving.
        let mut third = store.load();
        assert!(third.full_jev_remove());
        store.save(&third).unwrap();
        let mut activation_three = store.load();
        assert!(activation_three.full_jev_install());
        assert_ne!(activation_three.full_jev_stamp(), stamp_two);
        assert_ne!(activation_three.full_jev_stamp(), stamp_one);
    }
}
