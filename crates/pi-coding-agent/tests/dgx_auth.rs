//! Integration tests for the DGX auth file mechanism (`optimus-dgx-auth/1`).
//!
//! Every credential in this suite is synthetic: values carry a
//! `TEST-SECRET-NEVER-REAL` marker and the gateway endpoint is the reserved
//! `https://dgx.test/v1`, so nothing here can reach or reveal a real DGX
//! allocation. The tests pin the `core::dgx_auth` contract: export/import
//! round trips, validation rejections, force semantics, atomic failure
//! behavior, redaction, registry visibility of the three DGX models, the
//! optional credential-refresh command, expiry classification, and the
//! portability contract for command-backed credentials (export resolves
//! them to inline values, import rejects files that still reference helper
//! commands).
//!
//! Two tests are requirement-literal rather than mechanism-aware on purpose:
//! - `bundled_dgx_models_list_in_a_fresh_agent_dir` asserts the bundled DGX
//!   registrations list before any import (the shipped-registration
//!   requirement).
//! - `cli_dispatch_smoke_runs_dgx_auth_status` asserts the public command
//!   layer dispatches `dgx-auth status`.
//! Both fail in isolation when the sibling registration or CLI work has not
//! landed yet; the core mechanism itself is covered by the direct-API tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pi_ai::types::{Compat, InputModality};
use serde_json::json;

use pi_coding_agent::cli::public_command::{PublicCommandIo, handle_public_command};
use pi_coding_agent::config::ENV_AGENT_DIR;
use pi_coding_agent::core::auth_storage::AuthStorage;
use pi_coding_agent::core::dgx_auth::{
    DGX_AUTH_FORMAT_ID, DGX_AUTH_REFRESH_COMMAND_ENV, DgxAuthFile, DgxGateway,
    dgx_credentials_expired, export_dgx_auth, import_dgx_auth, redact_secret,
    refresh_dgx_credentials, validate_dgx_auth,
};
use pi_coding_agent::core::model_registry::{
    ModelDefinition, ModelRegistry, ModelsConfig, ProviderConfig,
};

/// Reserved, never-routable gateway endpoint for the whole suite.
const GATEWAY_BASE_URL: &str = "https://dgx.test/v1";
const POMERIUM_HEADER: &str = "X-Pomerium-Authorization";

// Synthetic credentials. Every value carries the NEVER-REAL marker so any leak
// into an error message, report, or CLI output is unambiguous in test output.
// The fixture `tests/fixtures/dgx-auth-sample.json` uses the same values; the
// fixture import test cross-checks that they stay in sync.
const SECRET_GLM53_KEY: &str = "TEST-SECRET-NEVER-REAL-GLM53-KEY-1";
const SECRET_K3_KEY: &str = "TEST-SECRET-NEVER-REAL-K3-KEY-2";
const SECRET_DEEPSEEK_KEY: &str = "TEST-SECRET-NEVER-REAL-DEEPSEEK-KEY-3";
const SECRET_JWT: &str = "TEST-SECRET-NEVER-REAL-POMERIUM-JWT-4";
const SECRET_GLM53_KEY_ALT: &str = "TEST-SECRET-NEVER-REAL-GLM53-KEY-ALT-5";
const SECRET_JWT_ALT: &str = "TEST-SECRET-NEVER-REAL-POMERIUM-JWT-ALT-6";
const SECRET_SOURCE_RELAY_KEY: &str = "TEST-SECRET-NEVER-REAL-SOURCE-RELAY-7";
const SECRET_FRESH_RELAY_KEY: &str = "TEST-SECRET-NEVER-REAL-FRESH-RELAY-8";
// Inline values produced by fake `!`-prefixed credential helpers: the export
// side must write exactly these into the auth file, and no error or report
// may ever name them.
const SECRET_RESOLVED_KEY: &str = "TEST-SECRET-NEVER-REAL-RESOLVED-KEY-9";
const SECRET_RESOLVED_JWT: &str = "TEST-SECRET-NEVER-REAL-RESOLVED-JWT-10";
const SECRET_RESOLVED_MODEL_HEADER: &str = "TEST-SECRET-NEVER-REAL-RESOLVED-MODEL-HDR-11";

/// Every synthetic credential used anywhere in this suite.
const SECRETS: &[&str] = &[
    SECRET_GLM53_KEY,
    SECRET_K3_KEY,
    SECRET_DEEPSEEK_KEY,
    SECRET_JWT,
    SECRET_GLM53_KEY_ALT,
    SECRET_JWT_ALT,
    SECRET_SOURCE_RELAY_KEY,
    SECRET_FRESH_RELAY_KEY,
    SECRET_RESOLVED_KEY,
    SECRET_RESOLVED_JWT,
    SECRET_RESOLVED_MODEL_HEADER,
];

/// The agent-dir environment is process-global, so tests that point it at a
/// temp dir are serialized through this lock. Poisoning is tolerated: one
/// failing env test must not cascade into unrelated failures.
static AGENT_DIR_LOCK: Mutex<()> = Mutex::new(());

fn agent_dir_lock() -> std::sync::MutexGuard<'static, ()> {
    AGENT_DIR_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The env var holding the optional DGX credential-refresh command. When set,
/// `refresh_dgx_credentials` runs it (`cmd /C` on Windows, `sh -c` elsewhere)
/// and a zero exit code means the on-disk credentials were renewed.
const REFRESH_CMD_ENV: &str = "OPTIMUS_DGX_AUTH_REFRESH_CMD";

/// Sets or clears an env var for the duration of the guard. Drop restores the
/// previous value even on panic, so one failing test cannot leak the variable
/// into its siblings. Callers must hold `AGENT_DIR_LOCK`; the environment is
/// process-global.
struct EnvVarGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var_os(name);
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(previous) => std::env::set_var(self.name, previous),
            None => std::env::remove_var(self.name),
        }
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dgx-auth-sample.json")
}

fn assert_no_secrets(text: &str) {
    for (index, secret) in SECRETS.iter().enumerate() {
        assert!(
            !text.contains(secret),
            "synthetic secret #{index} leaked into: {text}"
        );
    }
}

fn pom_headers(jwt: &str) -> IndexMap<String, String> {
    IndexMap::from([(POMERIUM_HEADER.to_string(), jwt.to_string())])
}

fn glm53_compat() -> serde_json::Value {
    json!({
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "supportsLongCacheRetention": false,
        "sglangTokenBudget": true,
        "thinkingFormat": "openai",
        "requiresThinkingAsText": false,
        "requiresReasoningContentOnAssistantMessages": true
    })
}

fn k3_compat() -> serde_json::Value {
    json!({
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "supportsLongCacheRetention": false
    })
}

fn dgx_model(id: &str, name: &str, input: &[&str]) -> ModelDefinition {
    ModelDefinition {
        id: id.to_string(),
        name: Some(name.to_string()),
        reasoning: Some(true),
        input: Some(input.iter().map(|value| value.to_string()).collect()),
        context_window: Some(1048576.0),
        max_tokens: Some(262144.0),
        ..Default::default()
    }
}

fn dgx_block(
    name: &str,
    api_key: &str,
    jwt: &str,
    compat: Option<serde_json::Value>,
    models: Vec<ModelDefinition>,
) -> ProviderConfig {
    ProviderConfig {
        name: Some(name.to_string()),
        base_url: Some(GATEWAY_BASE_URL.to_string()),
        api_key: Some(api_key.to_string()),
        api: Some("openai-completions".to_string()),
        headers: Some(pom_headers(jwt)),
        compat: compat.map(|value| serde_json::from_value::<Compat>(value).expect("valid compat")),
        models: Some(models),
        ..Default::default()
    }
}

fn glm53_block(api_key: &str, jwt: &str) -> ProviderConfig {
    dgx_block(
        "DGX GLM 5.3",
        api_key,
        jwt,
        Some(glm53_compat()),
        vec![dgx_model("glm-5.3", "GLM-5.3 - DGX MTP", &["text"])],
    )
}

fn k3_block(api_key: &str, jwt: &str) -> ProviderConfig {
    dgx_block(
        "DGX K3",
        api_key,
        jwt,
        Some(k3_compat()),
        vec![dgx_model(
            "kimi-k3",
            "Kimi K3 - DGX MAX",
            &["text", "image"],
        )],
    )
}

fn deepseek_block(api_key: &str, jwt: &str) -> ProviderConfig {
    dgx_block(
        "DGX DeepSeek 4.1",
        api_key,
        jwt,
        None,
        vec![dgx_model(
            "deepseek-v4.1-flash-1017",
            "DeepSeek V4.1 Flash 1017 - DGX",
            &["text", "image"],
        )],
    )
}

fn dgx_providers() -> IndexMap<String, ProviderConfig> {
    IndexMap::from([
        (
            "dgx-glm53".to_string(),
            glm53_block(SECRET_GLM53_KEY, SECRET_JWT),
        ),
        ("dgx-k3".to_string(), k3_block(SECRET_K3_KEY, SECRET_JWT)),
        (
            "dgx-deepseek41".to_string(),
            deepseek_block(SECRET_DEEPSEEK_KEY, SECRET_JWT),
        ),
    ])
}

fn relay_block(name: &str, api_key: &str) -> ProviderConfig {
    ProviderConfig {
        name: Some(name.to_string()),
        base_url: Some("https://relay.test/v1".to_string()),
        api_key: Some(api_key.to_string()),
        api: Some("openai-completions".to_string()),
        models: Some(vec![dgx_model("relay-mini", "Relay Mini", &["text"])]),
        ..Default::default()
    }
}

fn write_models_config(path: &Path, providers: Vec<(&str, ProviderConfig)>) -> ModelsConfig {
    let config = ModelsConfig {
        providers: providers
            .into_iter()
            .map(|(id, block)| (id.to_string(), block))
            .collect(),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    config
}

fn read_models_config(path: &Path) -> ModelsConfig {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn write_auth_file(path: &Path, file: &DgxAuthFile) {
    fs::write(path, serde_json::to_string_pretty(file).unwrap()).unwrap();
}

fn read_auth_file(path: &Path) -> DgxAuthFile {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn sample_auth_file() -> DgxAuthFile {
    DgxAuthFile {
        format: DGX_AUTH_FORMAT_ID.to_string(),
        created_at: "2026-11-02T00:00:00Z".to_string(),
        expires_at: Some("2026-11-03T12:00:00Z".to_string()),
        gateway: DgxGateway {
            base_url: GATEWAY_BASE_URL.to_string(),
            headers: pom_headers(SECRET_JWT),
        },
        providers: dgx_providers(),
    }
}

fn sorted(mut ids: Vec<String>) -> Vec<String> {
    ids.sort();
    ids
}

/// Sets `ENV_AGENT_DIR` for the duration of the guard. Callers must hold
/// `AGENT_DIR_LOCK`; the environment is process-global.
struct AgentDirEnv(Option<std::ffi::OsString>);

impl AgentDirEnv {
    fn set(path: &Path) -> Self {
        let previous = std::env::var_os(ENV_AGENT_DIR);
        std::env::set_var(ENV_AGENT_DIR, path);
        Self(previous)
    }
}

impl Drop for AgentDirEnv {
    fn drop(&mut self) {
        match self.0.take() {
            Some(previous) => std::env::set_var(ENV_AGENT_DIR, previous),
            None => std::env::remove_var(ENV_AGENT_DIR),
        }
    }
}

/// The console/error/exit-code seam for one `handle_public_command` run. The
/// callbacks are owned fields (not `&|...|` temporaries) so `io()` can hand
/// out references that outlive the call, matching the harness the
/// `public_command` unit tests use.
struct CliHarness {
    logged: Arc<Mutex<Vec<String>>>,
    errors: Arc<Mutex<Vec<String>>>,
    log_fn: Box<dyn Fn(&str)>,
    error_fn: Box<dyn Fn(&str)>,
    exit_fn: Box<dyn Fn(i32)>,
    prompt_fn: Box<dyn Fn(&str) -> bool>,
}

impl CliHarness {
    fn new() -> Self {
        let logged = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let log_fn: Box<dyn Fn(&str)> = {
            let logged = logged.clone();
            Box::new(move |line: &str| logged.lock().unwrap().push(line.to_string()))
        };
        let error_fn: Box<dyn Fn(&str)> = {
            let errors = errors.clone();
            Box::new(move |line: &str| errors.lock().unwrap().push(line.to_string()))
        };
        Self {
            logged,
            errors,
            log_fn,
            error_fn,
            exit_fn: Box::new(|_code: i32| {}),
            prompt_fn: Box::new(|_message: &str| false),
        }
    }

    fn io(&self, cwd: &Path) -> PublicCommandIo<'_> {
        PublicCommandIo {
            log: &*self.log_fn,
            error: &*self.error_fn,
            set_exit_code: &*self.exit_fn,
            stdin_is_tty: None,
            prompt_yes_no: &*self.prompt_fn,
            cwd: cwd.to_string_lossy().into_owned(),
        }
    }

    fn output(&self) -> String {
        let mut output = self.logged.lock().unwrap().join("\n");
        output.push('\n');
        output.push_str(&self.errors.lock().unwrap().join("\n"));
        output
    }
}

#[test]
fn round_trip_export_import_preserves_dgx_provider_blocks() {
    let directory = tempfile::tempdir().unwrap();
    let source_models = directory.path().join("source-models.json");
    let auth_path = directory.path().join("team.json");
    let fresh_models = directory.path().join("fresh-models.json");

    // The sharer's live models.json: three DGX providers plus an unrelated
    // relay that must not travel with the auth file.
    let source = write_models_config(
        &source_models,
        vec![
            ("dgx-glm53", glm53_block(SECRET_GLM53_KEY, SECRET_JWT)),
            ("dgx-k3", k3_block(SECRET_K3_KEY, SECRET_JWT)),
            (
                "dgx-deepseek41",
                deepseek_block(SECRET_DEEPSEEK_KEY, SECRET_JWT),
            ),
            (
                "source-relay",
                relay_block("Source Relay", SECRET_SOURCE_RELAY_KEY),
            ),
        ],
    );
    // The recipient starts from a models.json that only knows an unrelated relay.
    write_models_config(
        &fresh_models,
        vec![(
            "fresh-relay",
            relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
        )],
    );

    let exported = export_dgx_auth(&source_models, &auth_path, None).unwrap();
    assert_eq!(exported.format, DGX_AUTH_FORMAT_ID);
    // The opaque synthetic session header has no decodable exp claim, so the
    // export must omit expiresAt rather than guess one.
    assert!(exported.expires_at.is_none());
    // createdAt is an RFC3339 UTC timestamp.
    chrono::DateTime::parse_from_rfc3339(&exported.created_at)
        .unwrap_or_else(|error| panic!("createdAt is not RFC3339: {error}"));
    // The gateway block mirrors the DGX providers' endpoint and session header.
    assert_eq!(exported.gateway.base_url, GATEWAY_BASE_URL);
    assert_eq!(exported.gateway.headers, pom_headers(SECRET_JWT));
    // Only the dgx-* providers travel; the relay does not.
    assert_eq!(
        sorted(exported.providers.keys().cloned().collect()),
        vec!["dgx-deepseek41", "dgx-glm53", "dgx-k3"]
    );

    // The written file is the same document and never mentions the relay's key.
    let raw_auth = fs::read_to_string(&auth_path).unwrap();
    assert!(!raw_auth.contains(SECRET_SOURCE_RELAY_KEY));
    let reparsed = read_auth_file(&auth_path);
    assert_eq!(reparsed.format, DGX_AUTH_FORMAT_ID);
    assert_eq!(
        reparsed.providers.keys().cloned().collect::<Vec<_>>(),
        exported.providers.keys().cloned().collect::<Vec<_>>()
    );
    assert!(validate_dgx_auth(&reparsed).is_ok());

    let report = import_dgx_auth(&auth_path, &fresh_models, false).unwrap();
    assert_eq!(
        sorted(report.imported.clone()),
        vec!["dgx-deepseek41", "dgx-glm53", "dgx-k3"]
    );
    assert!(report.replaced.is_empty());
    assert!(report.skipped_unchanged.is_empty());
    assert_no_secrets(&format!("{report:?}"));

    let merged = read_models_config(&fresh_models);
    assert_eq!(
        sorted(merged.providers.keys().cloned().collect()),
        vec!["dgx-deepseek41", "dgx-glm53", "dgx-k3", "fresh-relay"]
    );
    for id in ["dgx-glm53", "dgx-k3", "dgx-deepseek41"] {
        let source_block = source.providers.get(id).unwrap();
        let imported = merged
            .providers
            .get(id)
            .unwrap_or_else(|| panic!("{id} missing after import"));
        assert_eq!(
            imported.api_key, source_block.api_key,
            "{id} apiKey mismatch"
        );
        assert_eq!(
            imported.headers, source_block.headers,
            "{id} headers mismatch"
        );
        assert_eq!(imported.models, source_block.models, "{id} models mismatch");
        assert_eq!(imported.compat, source_block.compat, "{id} compat mismatch");
        assert_eq!(
            imported.base_url, source_block.base_url,
            "{id} baseUrl mismatch"
        );
    }
    // The unrelated relay survived the merge untouched.
    assert_eq!(
        merged.providers["fresh-relay"].api_key.as_deref(),
        Some(SECRET_FRESH_RELAY_KEY)
    );

    // The atomic write leaves no temp files behind.
    let leftovers: Vec<String> = fs::read_dir(directory.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("tmp") || name.ends_with(".temp"))
        .collect();
    assert!(leftovers.is_empty(), "stray temp files: {leftovers:?}");
}

#[test]
fn export_selects_the_requested_providers_only() {
    let directory = tempfile::tempdir().unwrap();
    let source_models = directory.path().join("models.json");
    let auth_path = directory.path().join("k3-only.json");
    write_models_config(
        &source_models,
        vec![
            ("dgx-glm53", glm53_block(SECRET_GLM53_KEY, SECRET_JWT)),
            ("dgx-k3", k3_block(SECRET_K3_KEY, SECRET_JWT)),
            (
                "dgx-deepseek41",
                deepseek_block(SECRET_DEEPSEEK_KEY, SECRET_JWT),
            ),
        ],
    );

    let provider_ids = vec!["dgx-k3".to_string()];
    let exported =
        export_dgx_auth(&source_models, &auth_path, Some(provider_ids.as_slice())).unwrap();
    assert_eq!(
        exported.providers.keys().cloned().collect::<Vec<_>>(),
        vec!["dgx-k3"]
    );
    assert_eq!(
        exported.providers["dgx-k3"].api_key.as_deref(),
        Some(SECRET_K3_KEY)
    );
}

#[test]
fn export_resolves_command_backed_credentials_to_inline_values() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    let auth_path = directory.path().join("auth.json");

    // The sharer's live credentials are helper commands, not literals: the
    // gateway key, the shared session header, and a model-level header. The
    // commands stay quote-free because the resolver spawns them through the
    // configured shell or the `cmd /C` fallback, where inner quotes do not
    // survive.
    let mut glm53 = glm53_block(
        &format!("!echo {SECRET_RESOLVED_KEY}"),
        &format!("!echo {SECRET_RESOLVED_JWT}"),
    );
    glm53.models.as_mut().unwrap()[0].headers = Some(IndexMap::from([(
        "X-DGX-Model-Header".to_string(),
        format!("!echo {SECRET_RESOLVED_MODEL_HEADER}"),
    )]));
    // k3 keeps an inline key: it must pass through untouched. Both providers
    // share one session-header command, so the resolved headers still
    // compare equal and a shared gateway can be exported.
    let k3 = k3_block(SECRET_K3_KEY, &format!("!echo {SECRET_RESOLVED_JWT}"));
    write_models_config(
        &models,
        vec![("dgx-glm53", glm53), ("dgx-k3", k3)],
    );

    let exported = export_dgx_auth(&models, &auth_path, None).unwrap();

    // The command references became inline values...
    assert_eq!(
        exported.providers["dgx-glm53"].api_key.as_deref(),
        Some(SECRET_RESOLVED_KEY)
    );
    assert_eq!(
        exported.providers["dgx-glm53"].headers,
        Some(pom_headers(SECRET_RESOLVED_JWT))
    );
    assert_eq!(
        exported.providers["dgx-glm53"].models.as_ref().unwrap()[0]
            .headers
            .as_ref()
            .and_then(|headers| headers.get("X-DGX-Model-Header").map(String::as_str)),
        Some(SECRET_RESOLVED_MODEL_HEADER)
    );
    // ...the gateway mirrors the resolved session header...
    assert_eq!(exported.gateway.headers, pom_headers(SECRET_RESOLVED_JWT));
    // ...and inline credentials pass through unchanged.
    assert_eq!(
        exported.providers["dgx-k3"].api_key.as_deref(),
        Some(SECRET_K3_KEY)
    );

    // The written file is self-contained: the resolved values are inline and
    // no command reference survives.
    let raw = fs::read_to_string(&auth_path).unwrap();
    assert!(raw.contains(SECRET_RESOLVED_KEY));
    assert!(raw.contains(SECRET_RESOLVED_JWT));
    assert!(
        !raw.contains("!echo"),
        "command references must not survive into the auth file: {raw}"
    );
    assert!(validate_dgx_auth(&exported).is_ok());

    // The resolved file imports on a machine without the helper commands.
    let fresh_models = directory.path().join("fresh-models.json");
    write_models_config(
        &fresh_models,
        vec![(
            "fresh-relay",
            relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
        )],
    );
    let report = import_dgx_auth(&auth_path, &fresh_models, false).unwrap();
    assert_eq!(report.imported, vec!["dgx-glm53", "dgx-k3"]);
    assert_no_secrets(&format!("{report:?}"));
    let merged = read_models_config(&fresh_models);
    assert_eq!(
        merged.providers["dgx-glm53"].api_key.as_deref(),
        Some(SECRET_RESOLVED_KEY)
    );
    assert_eq!(
        merged.providers["dgx-k3"].api_key.as_deref(),
        Some(SECRET_K3_KEY)
    );
    assert_eq!(
        merged.providers["dgx-glm53"].headers,
        Some(pom_headers(SECRET_RESOLVED_JWT))
    );
}

#[test]
fn export_fails_when_a_command_backed_credential_does_not_resolve() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    let auth_path = directory.path().join("auth.json");

    // `exit 3` is a builtin of every shell the resolver spawns (the
    // configured shell, `cmd /C`, `sh -c`): the helper runs, exits nonzero,
    // and prints nothing.
    write_models_config(
        &models,
        vec![("dgx-glm53", glm53_block("!exit 3", SECRET_JWT))],
    );
    let error = export_dgx_auth(&models, &auth_path, None).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("dgx-glm53"),
        "the refusal must name the provider id: {message}"
    );
    assert!(
        message.contains("apiKey"),
        "the refusal must name the field kind: {message}"
    );
    assert!(
        !message.contains("exit 3"),
        "the command text must not leak: {message}"
    );
    assert_no_secrets(&message);
    assert_no_secrets(&format!("{error:?}"));
    assert!(
        !auth_path.exists(),
        "no file may be written when a credential does not resolve"
    );

    // A helper that exits 0 but prints nothing is equally unresolvable.
    write_models_config(
        &models,
        vec![("dgx-glm53", glm53_block("!exit 0", SECRET_JWT))],
    );
    let message = export_dgx_auth(&models, &auth_path, None)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("dgx-glm53"),
        "empty helper output must fail the export too: {message}"
    );
    assert!(!auth_path.exists());

    // The same refusal covers command-backed header values, naming the
    // header field instead of the apiKey.
    write_models_config(
        &models,
        vec![("dgx-glm53", glm53_block(SECRET_GLM53_KEY, "!exit 4"))],
    );
    let error = export_dgx_auth(&models, &auth_path, None).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("dgx-glm53"),
        "the refusal must name the provider id: {message}"
    );
    assert!(
        message.contains("header"),
        "the refusal must name the field kind: {message}"
    );
    assert!(
        !message.contains("exit 4"),
        "the command text must not leak: {message}"
    );
    assert_no_secrets(&message);
    assert!(!auth_path.exists());
}

#[test]
fn validation_rejects_wrong_format_id() {
    let mut file = sample_auth_file();
    file.format = "optimus-dgx-auth/2".to_string();
    let errors = validate_dgx_auth(&file).unwrap_err();
    let joined = errors.join("\n");
    assert!(
        joined.contains(DGX_AUTH_FORMAT_ID),
        "must name the expected format id: {joined}"
    );
    assert_no_secrets(&joined);
}

#[test]
fn validation_rejects_empty_providers() {
    let mut file = sample_auth_file();
    file.providers = IndexMap::new();
    let errors = validate_dgx_auth(&file).unwrap_err();
    let joined = errors.join("\n");
    assert!(
        joined.to_lowercase().contains("provider"),
        "must name the providers field: {joined}"
    );
    assert_no_secrets(&joined);
}

#[test]
fn validation_rejects_invalid_gateway_base_url() {
    for bad in ["", "ftp://dgx.test/v1", "dgx.test/v1"] {
        let mut file = sample_auth_file();
        file.gateway.base_url = bad.to_string();
        let errors = validate_dgx_auth(&file).unwrap_err();
        let joined = errors.join("\n");
        let lowered = joined.to_lowercase();
        assert!(
            lowered.contains("baseurl")
                || lowered.contains("base url")
                || lowered.contains("base_url")
                || lowered.contains("gateway"),
            "must name the gateway baseUrl for {bad:?}: {joined}"
        );
        assert_no_secrets(&joined);
    }
}

#[test]
fn validation_rejects_provider_without_api_or_models() {
    let mut file = sample_auth_file();
    // Credentials alone do not make a provider importable.
    file.providers.insert(
        "dgx-bare".to_string(),
        ProviderConfig {
            api_key: Some(SECRET_GLM53_KEY_ALT.to_string()),
            ..Default::default()
        },
    );
    let errors = validate_dgx_auth(&file).unwrap_err();
    let joined = errors.join("\n");
    assert!(
        joined.contains("dgx-bare"),
        "must name the provider id: {joined}"
    );
    assert_no_secrets(&joined);
}

#[test]
fn validation_collects_every_error_class_at_once() {
    let mut file = sample_auth_file();
    file.format = "optimus-dgx-auth/0".to_string();
    file.gateway.base_url = "ftp://dgx.test/v1".to_string();
    file.providers
        .insert("dgx-bare".to_string(), ProviderConfig::default());
    let errors = validate_dgx_auth(&file).unwrap_err();
    assert!(
        errors.len() >= 3,
        "expected every error class, got: {errors:?}"
    );
    let joined = errors.join("\n");
    assert!(joined.contains(DGX_AUTH_FORMAT_ID));
    assert!(joined.contains("dgx-bare"));
    assert_no_secrets(&joined);
}

#[test]
fn import_refuses_conflicting_credentials_without_force() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    // glm53 has a different key; k3 has the same key but a rotated session
    // header. Both count as different credentials.
    write_models_config(
        &models,
        vec![
            (
                "dgx-glm53",
                glm53_block(SECRET_GLM53_KEY_ALT, SECRET_JWT_ALT),
            ),
            ("dgx-k3", k3_block(SECRET_K3_KEY, SECRET_JWT_ALT)),
            (
                "fresh-relay",
                relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
            ),
        ],
    );
    let auth_path = directory.path().join("auth.json");
    write_auth_file(&auth_path, &sample_auth_file());

    let error = import_dgx_auth(&auth_path, &models, false).unwrap_err();
    let message = error.to_string();
    // The refusal names the conflicting provider ids...
    assert!(
        message.contains("dgx-glm53"),
        "refusal must name dgx-glm53: {message}"
    );
    assert!(
        message.contains("dgx-k3"),
        "refusal must name dgx-k3: {message}"
    );
    // ...and only the ids: no credential values from either side.
    assert_no_secrets(&message);
    assert_no_secrets(&format!("{error:?}"));
}

#[test]
fn import_with_force_replaces_conflicting_providers() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    // glm53 identical (skip), k3 conflicting (replace), deepseek41 absent (insert).
    write_models_config(
        &models,
        vec![
            ("dgx-glm53", glm53_block(SECRET_GLM53_KEY, SECRET_JWT)),
            ("dgx-k3", k3_block(SECRET_K3_KEY, SECRET_JWT_ALT)),
            (
                "fresh-relay",
                relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
            ),
        ],
    );
    let auth_path = directory.path().join("auth.json");
    write_auth_file(&auth_path, &sample_auth_file());

    let report = import_dgx_auth(&auth_path, &models, true).unwrap();
    assert_eq!(report.imported, vec!["dgx-deepseek41"]);
    assert_eq!(report.replaced, vec!["dgx-k3"]);
    assert_eq!(report.skipped_unchanged, vec!["dgx-glm53"]);
    assert_no_secrets(&format!("{report:?}"));

    let merged = read_models_config(&models);
    // Replaced: the auth file's session header won.
    let k3 = &merged.providers["dgx-k3"];
    assert_eq!(k3.api_key.as_deref(), Some(SECRET_K3_KEY));
    assert_eq!(k3.headers, Some(pom_headers(SECRET_JWT)));
    // Inserted: deepseek41 arrived with its block intact.
    let deepseek = &merged.providers["dgx-deepseek41"];
    assert_eq!(deepseek.api_key.as_deref(), Some(SECRET_DEEPSEEK_KEY));
    assert_eq!(
        deepseek
            .models
            .as_ref()
            .and_then(|models| models.first())
            .map(|model| model.id.as_str()),
        Some("deepseek-v4.1-flash-1017")
    );
    // Skipped and unrelated providers keep their credentials.
    assert_eq!(
        merged.providers["dgx-glm53"].api_key.as_deref(),
        Some(SECRET_GLM53_KEY)
    );
    assert_eq!(
        merged.providers["fresh-relay"].api_key.as_deref(),
        Some(SECRET_FRESH_RELAY_KEY)
    );
}

#[test]
fn import_skips_providers_with_identical_credentials() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    // Identical apiKey+headers, but a locally edited model name: the skip
    // rule is decided by credentials, so the local block stays as it is.
    let mut k3 = k3_block(SECRET_K3_KEY, SECRET_JWT);
    if let Some(model) = k3.models.as_mut().and_then(|models| models.first_mut()) {
        model.name = Some("Kimi K3 - locally edited".to_string());
    }
    write_models_config(
        &models,
        vec![
            ("dgx-glm53", glm53_block(SECRET_GLM53_KEY, SECRET_JWT)),
            ("dgx-k3", k3),
            (
                "dgx-deepseek41",
                deepseek_block(SECRET_DEEPSEEK_KEY, SECRET_JWT),
            ),
        ],
    );
    let auth_path = directory.path().join("auth.json");
    write_auth_file(&auth_path, &sample_auth_file());

    let report = import_dgx_auth(&auth_path, &models, false).unwrap();
    assert_eq!(
        sorted(report.skipped_unchanged.clone()),
        vec!["dgx-deepseek41", "dgx-glm53", "dgx-k3"]
    );
    assert!(report.imported.is_empty());
    assert!(report.replaced.is_empty());
    assert_no_secrets(&format!("{report:?}"));

    let merged = read_models_config(&models);
    let k3 = &merged.providers["dgx-k3"];
    assert_eq!(k3.api_key.as_deref(), Some(SECRET_K3_KEY));
    assert_eq!(
        k3.models
            .as_ref()
            .and_then(|models| models.first())
            .and_then(|model| model.name.as_deref()),
        Some("Kimi K3 - locally edited")
    );
}

#[test]
fn import_rejects_auth_files_that_still_reference_commands() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    write_models_config(
        &models,
        vec![(
            "fresh-relay",
            relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
        )],
    );
    let before = fs::read(&models).unwrap();

    // A hand-built file: the glm53 apiKey and the k3 session header are
    // still command references, so the file is not portable. deepseek41 keeps
    // inline credentials and must not be named.
    let mut file = sample_auth_file();
    file.providers["dgx-glm53"].api_key = Some(format!("!echo {SECRET_RESOLVED_KEY}"));
    file.providers["dgx-k3"].headers = Some(pom_headers(&format!(
        "!echo {SECRET_RESOLVED_JWT}"
    )));
    let auth_path = directory.path().join("auth.json");
    write_auth_file(&auth_path, &file);

    let error = import_dgx_auth(&auth_path, &models, false).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("dgx-glm53") && message.contains("dgx-k3"),
        "the rejection must name the offending provider ids: {message}"
    );
    assert!(
        !message.contains("dgx-deepseek41"),
        "providers with inline credentials must not be named: {message}"
    );
    assert_no_secrets(&message);
    assert_no_secrets(&format!("{error:?}"));

    // The rejection is structural, so --force does not bypass it, and
    // models.json stays byte-identical either way.
    let error = import_dgx_auth(&auth_path, &models, true).unwrap_err();
    assert_no_secrets(&error.to_string());
    assert_eq!(
        fs::read(&models).unwrap(),
        before,
        "models.json must stay byte-identical"
    );
}

#[test]
fn failed_import_leaves_models_json_byte_identical() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    write_models_config(
        &models,
        vec![
            ("dgx-k3", k3_block(SECRET_K3_KEY, SECRET_JWT_ALT)),
            (
                "fresh-relay",
                relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
            ),
        ],
    );
    let before = fs::read(&models).unwrap();

    // (a) credential conflict without --force.
    let auth_path = directory.path().join("auth.json");
    write_auth_file(&auth_path, &sample_auth_file());
    let error = import_dgx_auth(&auth_path, &models, false).unwrap_err();
    assert_no_secrets(&error.to_string());
    assert_no_secrets(&format!("{error:?}"));
    assert_eq!(
        fs::read(&models).unwrap(),
        before,
        "models.json must stay byte-identical"
    );

    // (b) validation failure (wrong format id).
    let mut invalid = sample_auth_file();
    invalid.format = "optimus-dgx-auth/2".to_string();
    let invalid_path = directory.path().join("invalid.json");
    write_auth_file(&invalid_path, &invalid);
    let error = import_dgx_auth(&invalid_path, &models, false).unwrap_err();
    assert_no_secrets(&error.to_string());
    assert_eq!(fs::read(&models).unwrap(), before);

    // (c) malformed JSON.
    let malformed_path = directory.path().join("malformed.json");
    fs::write(&malformed_path, "{\"format\":").unwrap();
    let error = import_dgx_auth(&malformed_path, &models, false).unwrap_err();
    assert_no_secrets(&error.to_string());
    assert_eq!(fs::read(&models).unwrap(), before);

    // (d) missing auth file.
    let missing_path = directory.path().join("missing.json");
    let error = import_dgx_auth(&missing_path, &models, false).unwrap_err();
    assert_no_secrets(&error.to_string());
    assert_eq!(fs::read(&models).unwrap(), before);
}

#[test]
fn imports_the_sample_fixture_and_lists_dgx_models_in_the_registry() {
    let _guard = agent_dir_lock();
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    write_models_config(
        &models,
        vec![(
            "fresh-relay",
            relay_block("Fresh Relay", SECRET_FRESH_RELAY_KEY),
        )],
    );

    let report = import_dgx_auth(&fixture_path(), &models, false).unwrap();
    assert_eq!(
        sorted(report.imported.clone()),
        vec!["dgx-deepseek41", "dgx-glm53", "dgx-k3"]
    );
    assert_no_secrets(&format!("{report:?}"));

    let merged = read_models_config(&models);
    // Fixture sync: the file's values are this suite's synthetic constants.
    let glm53 = &merged.providers["dgx-glm53"];
    assert_eq!(glm53.api_key.as_deref(), Some(SECRET_GLM53_KEY));
    assert_eq!(glm53.headers, Some(pom_headers(SECRET_JWT)));
    assert_eq!(glm53.base_url.as_deref(), Some(GATEWAY_BASE_URL));
    assert_eq!(
        glm53
            .models
            .as_ref()
            .and_then(|models| models.first())
            .map(|model| model.id.as_str()),
        Some("glm-5.3")
    );
    assert_eq!(
        glm53
            .compat
            .as_ref()
            .and_then(|compat| compat.as_completions())
            .and_then(|compat| compat.max_tokens_field.as_deref()),
        Some("max_tokens")
    );
    assert_eq!(
        merged.providers["fresh-relay"].api_key.as_deref(),
        Some(SECRET_FRESH_RELAY_KEY)
    );

    // The fixture itself validates and matches the pinned shape.
    let parsed = read_auth_file(&fixture_path());
    assert_eq!(parsed.format, DGX_AUTH_FORMAT_ID);
    assert!(validate_dgx_auth(&parsed).is_ok());

    // End to end: after the import, the recipient's registry lists the three
    // DGX models with the gateway endpoint applied.
    let _agent_dir = AgentDirEnv::set(directory.path());
    let registry = ModelRegistry::create(AuthStorage::in_memory(Default::default(), None), None);
    for (provider, id) in [
        ("dgx-glm53", "glm-5.3"),
        ("dgx-k3", "kimi-k3"),
        ("dgx-deepseek41", "deepseek-v4.1-flash-1017"),
    ] {
        let model = registry
            .find(provider, id)
            .unwrap_or_else(|| panic!("{provider}/{id} missing from the registry"));
        assert_eq!(model.base_url, GATEWAY_BASE_URL);
        assert!(model.reasoning);
    }
    let kimi = registry.find("dgx-k3", "kimi-k3").unwrap();
    assert_eq!(kimi.context_window, 1048576.0);
    assert!(kimi.input.contains(&InputModality::Image));
}

#[test]
fn bundled_dgx_models_list_in_a_fresh_agent_dir() {
    // Requirement-literal: the bundled DGX registrations list in a fresh
    // agent dir with no models.json and no import. This pins the shipped
    // registration requirement itself, not any particular mechanism.
    let _guard = agent_dir_lock();
    let directory = tempfile::tempdir().unwrap();
    let _agent_dir = AgentDirEnv::set(directory.path());
    let registry = ModelRegistry::create(AuthStorage::in_memory(Default::default(), None), None);
    for (provider, id) in [
        ("dgx-glm53", "glm-5.3"),
        ("dgx-k3", "kimi-k3"),
        ("dgx-deepseek41", "deepseek-v4.1-flash-1017"),
    ] {
        assert!(
            registry.find(provider, id).is_some(),
            "bundled model {provider}/{id} must list in a fresh agent dir"
        );
    }
}

#[test]
fn export_decodes_the_session_expiry_from_a_decodable_header() {
    use base64::Engine as _;
    // 2026-12-01T00:00:00Z. A syntactically decodable JWT with a plain exp
    // claim; no real signature, no real session.
    let exp = 1_796_083_200i64;
    let header = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#);
    let payload =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(format!(r#"{{"exp":{exp}}}"#));
    let signature = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(b"TEST-SECRET-NEVER-REAL-SIGNATURE");
    let session = format!("{header}.{payload}.{signature}");

    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    let auth_path = directory.path().join("auth.json");
    write_models_config(
        &models,
        vec![
            ("dgx-glm53", glm53_block(SECRET_GLM53_KEY, &session)),
            ("dgx-k3", k3_block(SECRET_K3_KEY, &session)),
            (
                "dgx-deepseek41",
                deepseek_block(SECRET_DEEPSEEK_KEY, &session),
            ),
        ],
    );

    let exported = export_dgx_auth(&models, &auth_path, None).unwrap();
    let expires_at = exported
        .expires_at
        .expect("decodable exp claim must surface as expiresAt");
    let parsed = chrono::DateTime::parse_from_rfc3339(&expires_at)
        .unwrap_or_else(|error| panic!("expiresAt is not RFC3339 ({expires_at}): {error}"));
    let expected = chrono::DateTime::from_timestamp(exp, 0).unwrap();
    let drift = (parsed.timestamp() - expected.timestamp()).abs();
    assert!(
        drift <= 2,
        "expiresAt {parsed} drifts from the exp claim by {drift}s"
    );
}

#[test]
fn redact_secret_reports_the_length_and_not_the_value() {
    let redacted = redact_secret(SECRET_GLM53_KEY);
    assert_ne!(redacted, SECRET_GLM53_KEY);
    assert!(!redacted.contains(SECRET_GLM53_KEY), "{redacted}");
    // No prefix of the value survives either.
    assert!(!redacted.contains("TEST-SECRET"), "{redacted}");
    assert!(redacted.contains("<redacted"), "{redacted}");
    assert!(
        redacted.contains(&SECRET_GLM53_KEY.len().to_string()),
        "must report the length: {redacted}"
    );
}

#[test]
fn cli_dispatch_smoke_runs_dgx_auth_status() {
    // Requirement-literal dispatch smoke for the pinned CLI surface
    // (`optimus-agent dgx-auth status`). Depends on the command registration
    // landing in the CLI registry; the core mechanism is covered by the
    // direct-API tests above.
    let _guard = agent_dir_lock();
    let directory = tempfile::tempdir().unwrap();
    write_models_config(
        &directory.path().join("models.json"),
        vec![
            ("dgx-glm53", glm53_block(SECRET_GLM53_KEY, SECRET_JWT)),
            ("dgx-k3", k3_block(SECRET_K3_KEY, SECRET_JWT)),
            (
                "dgx-deepseek41",
                deepseek_block(SECRET_DEEPSEEK_KEY, SECRET_JWT),
            ),
        ],
    );
    let _agent_dir = AgentDirEnv::set(directory.path());

    let harness = CliHarness::new();
    let args = vec!["dgx-auth".to_string(), "status".to_string()];
    let result =
        futures::executor::block_on(handle_public_command(&args, &harness.io(directory.path())));
    assert!(
        result.handled,
        "dgx-auth status must be dispatched by the public command layer"
    );

    let output = harness.output();
    assert!(!output.contains("Unknown command"), "{output}");
    assert_no_secrets(&output);
}

#[test]
fn refresh_without_a_configured_command_is_a_no_op() {
    let _guard = agent_dir_lock();
    let _env = EnvVarGuard::set(REFRESH_CMD_ENV, None);
    // The suite pins the documented env var name; the core exports the same
    // constant, so an accidental rename fails here instead of silently
    // following it.
    assert_eq!(REFRESH_CMD_ENV, DGX_AUTH_REFRESH_COMMAND_ENV);
    // No refresh command in the environment: nothing ran, so no error and no
    // claimed refresh.
    assert!(!refresh_dgx_credentials().unwrap());
}

#[test]
fn refresh_runs_the_configured_command_and_reports_success() {
    let _guard = agent_dir_lock();
    // `exit 0` is a builtin on both `cmd /C` and `sh -c`, so the same fake
    // command works everywhere.
    let _env = EnvVarGuard::set(REFRESH_CMD_ENV, Some("exit 0"));
    assert!(refresh_dgx_credentials().unwrap());
}

#[test]
fn refresh_failure_reports_the_exit_code_without_secrets() {
    let _guard = agent_dir_lock();
    let _env = EnvVarGuard::set(REFRESH_CMD_ENV, Some("exit 3"));
    let error = refresh_dgx_credentials().unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains('3'),
        "the failure must mention the exit code: {message}"
    );
    assert_no_secrets(&message);
    assert_no_secrets(&format!("{error:?}"));
}

#[test]
fn refresh_then_export_reads_the_rotated_credentials() {
    let _guard = agent_dir_lock();
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models.json");
    let staged = directory.path().join("staged-models.json");
    // Live state: glm53 with the original key.
    write_models_config(
        &models,
        vec![("dgx-glm53", glm53_block(SECRET_GLM53_KEY, SECRET_JWT))],
    );
    // The staged rotation: the gateway session was renewed, so both the key
    // and the session header changed.
    write_models_config(
        &staged,
        vec![(
            "dgx-glm53",
            glm53_block(SECRET_GLM53_KEY_ALT, SECRET_JWT_ALT),
        )],
    );

    // One outer quote pair survives every reasonable `cmd /C` spawn strategy;
    // inner quotes only survive raw-argument spawns, so the paths stay
    // unquoted (temp paths carry no spaces on the platforms this suite runs).
    let refresh_command = if cfg!(windows) {
        format!("copy /Y {} {}", staged.display(), models.display())
    } else {
        format!("cp {} {}", staged.display(), models.display())
    };
    let _env = EnvVarGuard::set(REFRESH_CMD_ENV, Some(&refresh_command));

    assert!(refresh_dgx_credentials().unwrap());
    // The refresh command rewrote models.json on disk...
    let after = read_models_config(&models);
    assert_eq!(
        after.providers["dgx-glm53"].api_key.as_deref(),
        Some(SECRET_GLM53_KEY_ALT)
    );
    assert_eq!(
        after.providers["dgx-glm53"].headers,
        Some(pom_headers(SECRET_JWT_ALT))
    );

    // ...and a subsequent export reads the rotated credentials from disk
    // instead of any pre-refresh snapshot.
    let auth_path = directory.path().join("rotated.json");
    let exported = export_dgx_auth(&models, &auth_path, None).unwrap();
    assert_eq!(
        exported.providers["dgx-glm53"].api_key.as_deref(),
        Some(SECRET_GLM53_KEY_ALT)
    );
    assert_eq!(exported.gateway.headers, pom_headers(SECRET_JWT_ALT));
    let raw = fs::read_to_string(&auth_path).unwrap();
    assert!(raw.contains(SECRET_GLM53_KEY_ALT));
    assert!(!raw.contains(SECRET_GLM53_KEY));
    assert!(!raw.contains(SECRET_JWT));
    assert!(validate_dgx_auth(&exported).is_ok());
}

#[test]
fn expired_credentials_classify_the_expires_at_claim() {
    // Best-effort classification of the decoded exp claim: only a parseable
    // past timestamp counts as expired.
    let now = chrono::Utc::now();
    let mut file = sample_auth_file();

    file.expires_at = Some((now - chrono::Duration::hours(24)).to_rfc3339());
    assert!(
        dgx_credentials_expired(&file),
        "a past expiry must count as expired"
    );

    file.expires_at = Some((now + chrono::Duration::hours(24)).to_rfc3339());
    assert!(
        !dgx_credentials_expired(&file),
        "a future expiry must not count as expired"
    );

    // An undecodable session header leaves expiresAt absent: unknown, not
    // expired.
    file.expires_at = None;
    assert!(!dgx_credentials_expired(&file));

    file.expires_at = Some("not-a-timestamp".to_string());
    assert!(
        !dgx_credentials_expired(&file),
        "an unparseable expiry must not count as expired"
    );
}
