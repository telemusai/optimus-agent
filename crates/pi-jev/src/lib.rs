//! Bounded Jev / TypeSafe System One transport, policies, and observation records.
//!
//! Crate direction rule (DESIGN.md section 10.1): pi-jev is framework-neutral. It must never
//! depend on pi-coding-agent or pi-ai; the Extension/AgentSession adapter lives in
//! pi-coding-agent and calls into this crate.

pub mod active;
pub mod agent_guidance;
pub mod client;
pub mod compaction;
pub mod config;
pub mod control;
pub mod correlate;
pub mod credential;
pub mod error;
pub mod dynamic;
pub mod evidence;
pub mod evaluators;
pub mod filtering;
pub mod hooks;
pub mod mock;
pub mod models;
pub mod observation;
pub mod redact;
pub mod report;
pub mod scheduler;
pub mod search;
pub mod snapshot;
pub mod telemetry;
pub mod types;

pub use evidence::{
    CitationQuotePresence, CitationVerdict, SafetyLabels, SAFETY_QUESTIONS_PER_CANDIDATE,
    safety_questions,
};
pub use agent_guidance::{
    choice_abstention, compose_fixed_weighted, guardrail_route, guidance_stamp_hash, noul_band,
    skill_hint, skill_hint_is_current, ChoiceAbstention, GuardrailAssessment, GuardrailInput,
    GuardrailPolicy, GuardrailRoute, GuidanceThresholds, NoHintReason, NoulBand, SkillAnswerSet,
    CapturedSkillHintStamp, SkillCatalogEntry, SkillHint, SkillHintOutcome, SkillHintStamp,
    TimingLabel,
    WeightedComposition, MAX_GUIDANCE_CATALOG, MAX_GUIDANCE_TEXT_CHARS,
};
pub use models::{
    fetch_model_catalog, parse_model_catalog, ModelCard, ModelCatalog, MODEL_CATALOG_PATH,
    MAX_MODEL_CATALOG_ENTRIES, MAX_MODEL_DATE_CHARS, MAX_MODEL_DESCRIPTION_CHARS,
    MAX_MODEL_NAME_CHARS,
};
pub use active::{
    evaluate_answer, Acceptance, ActivationPolicy, ActiveDecision, AnswerCandidate, AppliedEffect,
    FallbackReason, DEFAULT_APPLIABLE_CATEGORIES, MAX_EFFECT_CHARS, MAX_VALUE_CHARS,
};
pub use client::{
    accepted_answers, backoff_policy_line, build_system_one, bundle_with_questions, decide_with,
    parse_retry_after, parse_systemone_body, retry_decision, ClientHandle, DisabledSystemOne,
    JevHttpTransport, JevLimits, JevStats, JevStatsSnapshot, JevSystemOne, RetryDecision,
    DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, MAX_RETRY_AFTER, RETRYABLE_STATUSES,
    STATUS_OVERLOADED,
};
pub use compaction::CompactionConfig;
pub use config::{
    credential_status_line, default_agent_dir, inherit_mode, jev_dir_for,
    resolve_credential_source, resolve_credential_source_from_presence, resolve_effective_mode,
    resolve_mode, settings_store, CredentialSource, EnvKeyPresence, JevFeature, JevFeatures,
    JevMode, JevSettings, JevSettingsStore, ModeResolution, ModeScope, PersistedSessionMode,
    DEFAULT_AGENT_DIR_NAME, DEFAULT_KEY_ID, ENV_AGENT_DIR, ENV_JEV_API_KEY, ENV_TYPESAFE_API_KEY,
    SETTINGS_FILE_NAME, SETTINGS_SCHEMA_VERSION,
};
pub use control::{
    combine_sufficiency, control_gates_open, evaluate_control_answer, nonprogress_verdict,
    verification_need, CorrelatedVerificationEvidence, ControlAcceptance, ControlBoundary,
    ControlBudgetKind, ControlBudgetSnapshot, ControlBudgets, ControlEffectKind, ControlFeatures,
    ControlPolicy, ControlProvenance, ControlRefusal, ControlVerificationState, ControlVerdict,
    FeedbackKind, HostControlFacts, NonprogressVerdict, PauseReason, SufficiencyVerdict,
    TurnSignature, VerificationNeed, VerificationSourceKind, CONTROL_ACT_MIN_CONFIDENCE,
    CONTROL_HIGH_IMPACT_MIN_CONFIDENCE, CONTROL_MAX_DECISION_AGE, CONTROL_MAX_FEEDBACK,
    CONTROL_MAX_NONPROGRESS_CORRECTIONS, CONTROL_MAX_RETRY_VETOES,
    CONTROL_MAX_VERIFICATION_REQUESTS, CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS,
    CONTROL_NONPROGRESS_WINDOW,
};
pub use credential::{
    default_credential_store,
    validate_key_id,
    CredentialStore,
    DpapiCredentialStore,
    InMemoryCredentialStore,
    SecretString,
    UnavailableCredentialStore,
    CREDENTIAL_ENVELOPE_VERSION,
    CREDENTIAL_FILE_NAME,
    MAX_SECRET_LEN,
    // `secure_store_available` is intentionally reachable as `credential::secure_store_available`
    // so lane C can gate the key-input UI before the first save.
};
pub use error::{
    redact_authorization, sanitize_detail, sanitize_url, JevError, MAX_DETAIL_CHARS, REDACTED,
};
pub use filtering::FilteringOptions;
pub use mock::{
    canonicalize, choice_question, fingerprint, low_confidence_response_for, mutate, noul_question,
    question_ids, raw_huge_extra_ids_body, raw_injection_body, raw_noul_non_finite_body,
    raw_truncated_body, raw_wrong_shape_body, score_question, valid_answer_for, valid_response_for,
    MockJevTransport, MockMutation, MockStep, RecordedCall, MOCK_RESPONSE_MODEL,
};
pub use observation::{ResultAssessment, RetryFailureKind, TraceAssessment};
pub use types::{
    compare_default_categories, model_drift, refuse_subagent_control, validate_answer,
    validate_request_shape, validate_response, Answer, AnswerIssue, AnswerType, BoxFuture,
    Complexity, ContinueDecision, DecisionBundle, DecisionCategory, DecisionOutcome,
    DecisionRecord, ModelDrift, NoulCriteria, QuestionSpec, QuestionType, ResponseValidation,
    SubagentControlRequest, SubagentObservation, Sufficient, SystemOne, SystemOneRequest,
    SystemOneResponse, TaskType, ToolRequirement, Transport, Usage, VerificationRecommendation,
    DEFAULT_MODEL, FORBIDDEN_SUBAGENT_CAPABILITIES, MAX_QUESTIONS_PER_REQUEST,
    PROBABILITY_TOLERANCE, SYSTEMONE_PATH,
};

pub use search::{
    build_line_find_text, line_find_questions, line_state, rerank_batch_questions,
    rerank_batch_state, reranked_order, scored_candidates_from_decisions, top_lines,
    validate_where_distribution, validate_window_distribution, verdict, window_questions,
    window_state, LineEntry,
    LineFindOptions, LineFindText, LineFindVerdict, RerankOptions, ScoredCandidate,
    SearchBudget, WindowEntries, EXISTS_ABSENT, EXISTS_FOUND, EXISTS_QUESTION_ID,
    LINE_EXCERPT_CHARS, LINE_FIND_CATEGORY, LINE_WINDOW, MAX_LINE_FIND_LINES,
    MAX_LINE_STATE_BYTES, MAX_RERANK_CANDIDATES, PROMPT_VERSION_SEARCH, RERANK_BATCH,
    RERANK_CATEGORY, SHARED_SEARCH_BUDGET_MS, WHERE_QUESTION_ID, WINDOW_QUESTION_ID,
};
pub use types::{LineFindAssessment, RerankAssessment};
