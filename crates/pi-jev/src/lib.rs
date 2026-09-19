//! pi-jev: isolated Jev / TypeSafe System One client plus comparison-mode scaffolding.
//!
//! Crate direction rule (DESIGN.md section 10.1): pi-jev is framework-neutral. It must never
//! depend on pi-coding-agent or pi-ai; the Extension/AgentSession adapter lives in
//! pi-coding-agent and calls into this crate.
//!
//! Lane A (jev-client) owns: lib.rs, types.rs, error.rs, client.rs, mock.rs, config.rs,
//! credential.rs, tests/client_tests.rs.
//! Lane B (jev-comparison) owns: snapshot.rs, scheduler.rs, evaluators.rs, evaluators/*.rs,
//! hooks.rs, correlate.rs, report.rs and fills the placeholder modules declared below.

pub mod client;
pub mod config;
pub mod credential;
pub mod error;
pub mod mock;
pub mod types;

// Lane B modules. Created as empty placeholders so the crate compiles before lane B lands.
pub mod correlate;
pub mod evaluators;
pub mod hooks;
pub mod report;
pub mod scheduler;
pub mod snapshot;

pub use client::{
    accepted_answers, backoff_policy_line, build_system_one, bundle_with_questions, decide_with, parse_systemone_body,
    retry_decision, ClientHandle, DisabledSystemOne, JevHttpTransport, JevLimits, JevStats, JevStatsSnapshot,
    parse_retry_after, JevSystemOne, RetryDecision, DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT,
    MAX_RETRY_AFTER, RETRYABLE_STATUSES, STATUS_OVERLOADED,
};
pub use config::{
    credential_status_line, default_agent_dir, inherit_mode, jev_dir_for, resolve_credential_source,
    resolve_credential_source_from_presence, resolve_effective_mode, resolve_mode,
    settings_store, CredentialSource, EnvKeyPresence, JevMode, JevSettings, JevSettingsStore, ModeResolution,
    ModeScope, PersistedSessionMode, DEFAULT_AGENT_DIR_NAME, DEFAULT_KEY_ID, ENV_AGENT_DIR, ENV_JEV_API_KEY,
    ENV_TYPESAFE_API_KEY, SETTINGS_FILE_NAME, SETTINGS_SCHEMA_VERSION,
};
pub use credential::{
    default_credential_store, validate_key_id, CredentialStore, DpapiCredentialStore, InMemoryCredentialStore,
    SecretString, UnavailableCredentialStore, CREDENTIAL_ENVELOPE_VERSION, CREDENTIAL_FILE_NAME, MAX_SECRET_LEN,
    // `secure_store_available` is intentionally reachable as `credential::secure_store_available`
    // so lane C can gate the key-input UI before the first save.
};
pub use error::{redact_authorization, sanitize_detail, sanitize_url, JevError, MAX_DETAIL_CHARS, REDACTED};
pub use mock::{
    canonicalize, choice_question, fingerprint, low_confidence_response_for, mutate, noul_question, question_ids,
    raw_huge_extra_ids_body, raw_injection_body, raw_noul_non_finite_body, raw_truncated_body, raw_wrong_shape_body,
    score_question, valid_answer_for, valid_response_for, MockJevTransport, MockMutation, MockStep, RecordedCall,
    MOCK_RESPONSE_MODEL,
};
pub use types::{
    compare_default_categories, model_drift, refuse_subagent_control, validate_answer, validate_request_shape,
    validate_response, Answer, AnswerIssue, AnswerType, BoxFuture, Complexity, ContinueDecision, DecisionBundle,
    DecisionCategory, DecisionOutcome, DecisionRecord, ModelDrift, NoulCriteria, QuestionSpec, QuestionType,
    ResponseValidation, SubagentControlRequest, SubagentObservation, Sufficient, SystemOne, SystemOneRequest,
    SystemOneResponse, TaskType, ToolRequirement, Transport, Usage, VerificationRecommendation,
    FORBIDDEN_SUBAGENT_CAPABILITIES, DEFAULT_MODEL, MAX_QUESTIONS_PER_REQUEST, PROBABILITY_TOLERANCE,
    SYSTEMONE_PATH,
};
