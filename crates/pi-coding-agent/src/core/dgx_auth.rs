//! DGX auth files: self-contained, shareable credentials for the `dgx-*`
//! custom providers in models.json.
//!
//! `export_dgx_auth` snapshots the live `dgx-*` providers (gateway key,
//! Pomerium headers, model definitions) into an `optimus-dgx-auth/1` JSON
//! file. `import_dgx_auth` merges those providers into another user's
//! models.json, so the recipient gets working DGX models until the gateway
//! session expires. models.json credentials may be command-backed
//! (`"!"`-prefixed values that run a helper command on the exporting
//! machine); export resolves those to their inline output so the file is
//! self-contained, and import rejects files that still carry command
//! references. Secret values never appear in errors, reports, or
//! messages; only provider ids, presence flags, and JSON positions are
//! reported.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use chrono::{DateTime, SecondsFormat, Utc};
use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::model_registry::{ModelsConfig, ProviderConfig, strip_json_comments};
use crate::core::resolve_config_value::resolve_config_value_uncached;
use crate::utils::atomic_file::{
    WriteFileAtomicOptions, realpath_if_present_sync, write_file_atomic_sync,
};

/// Format marker written into every exported auth file.
pub const DGX_AUTH_FORMAT_ID: &str = "optimus-dgx-auth/1";

/// Environment variable holding the command that renews the DGX gateway
/// credentials (for example a Pomerium login helper that rewrites
/// models.json). Its output is never captured or printed: a misbehaving
/// script may echo credential material.
pub const DGX_AUTH_REFRESH_COMMAND_ENV: &str = "OPTIMUS_DGX_AUTH_REFRESH_CMD";

/// Login can involve a human, so the refresh command gets a generous fixed
/// window before it is killed.
const REFRESH_TIMEOUT_SECS: u64 = 35 * 60;

/// Case-insensitive name of the header carrying the Pomerium gateway session JWT.
const POMERIUM_HEADER: &str = "x-pomerium-authorization";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DgxAuthFile {
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub gateway: DgxGateway,
    #[serde(default)]
    pub providers: IndexMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DgxGateway {
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub headers: IndexMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportReport {
    pub imported: Vec<String>,
    pub replaced: Vec<String>,
    pub skipped_unchanged: Vec<String>,
}

/// Snapshot the `dgx-*` providers of `models_json_path` into a shareable auth
/// file at `out_path`. `provider_ids` restricts the export to a subset; `None`
/// exports every `dgx-*` provider. Command-backed credentials
/// (`"!"`-prefixed apiKey/header values that run a helper command on this
/// machine) are resolved to their inline output first, so the file is
/// self-contained; a value that does not resolve fails the export before any
/// file is written. Returns the file that was written.
pub fn export_dgx_auth(
    models_json_path: &Path,
    out_path: &Path,
    provider_ids: Option<&[String]>,
) -> anyhow::Result<DgxAuthFile> {
    if same_file(models_json_path, out_path) {
        bail!("The DGX auth export destination must not be the models.json source");
    }
    let content = read_file_content(models_json_path, "models.json")?;
    let config: ModelsConfig = parse_json_or_empty("models.json", &content)?;
    let selected = select_dgx_providers(&config, provider_ids)?;
    // Command-backed credentials must not travel as references to helper
    // commands on this machine. Resolving before the gateway is built also
    // inlines the gateway headers, which come from the first provider.
    let selected = resolve_command_backed_credentials(&selected)?;
    let gateway = shared_gateway(&selected)?;

    let file = DgxAuthFile {
        format: DGX_AUTH_FORMAT_ID.to_string(),
        created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        expires_at: pomerium_expiry_hint(Some(&gateway.headers)),
        gateway,
        providers: selected.into_iter().collect(),
    };
    if let Err(errors) = validate_dgx_auth(&file) {
        bail!(
            "Refusing to export an invalid DGX auth file:\n{}",
            errors.join("\n")
        );
    }
    write_auth_file(out_path, &file)?;
    Ok(file)
}

/// Merge the providers of the auth file at `auth_file_path` into
/// `models_json_path`, creating that file when it does not exist. Providers
/// whose credentials (apiKey + headers) already match are skipped; conflicting
/// providers are refused unless `force` replaces them.
pub fn import_dgx_auth(
    auth_file_path: &Path,
    models_json_path: &Path,
    force: bool,
) -> anyhow::Result<ImportReport> {
    if same_file(auth_file_path, models_json_path) {
        bail!("The models.json destination must differ from the imported DGX auth file");
    }
    let content = read_file_content(auth_file_path, "DGX auth file")?;
    let file: DgxAuthFile = parse_json_strict("DGX auth file", &content)?;
    if let Err(errors) = validate_dgx_auth(&file) {
        bail!("Invalid DGX auth file:\n{}", errors.join("\n"));
    }
    // Auth files only ever carry `dgx-*` providers; anything else is a
    // hand-edited file, and importing arbitrary provider ids would let a
    // crafted file reroute unrelated providers.
    let non_dgx: Vec<&str> = file
        .providers
        .keys()
        .map(String::as_str)
        .filter(|id| !is_dgx_provider_id(id))
        .collect();
    if !non_dgx.is_empty() {
        bail!(
            "A DGX auth file may only declare \"dgx-\" providers; it also declares: {}",
            non_dgx.join(", ")
        );
    }
    // Export resolves command-backed credentials to their inline output, so a
    // surviving `"!"`-prefixed value means a hand-built or downgraded file.
    // Importing it would embed the sharer's helper commands in the
    // recipient's models.json, and every request would then fail to resolve
    // them. Only the offending provider ids are reported.
    let command_backed = command_backed_provider_ids(&file);
    if !command_backed.is_empty() {
        bail!(
            "The DGX auth file is not portable: providers {} still carry command references \
             (\"!\"-prefixed apiKey or header values) instead of inline credentials. \
             Re-export the file on the machine where those commands resolve",
            command_backed.join(", ")
        );
    }

    let mut root = read_models_root(models_json_path)?;
    let object = root.as_object_mut().ok_or_else(|| {
        anyhow!(
            "models.json at {} is not a JSON object",
            models_json_path.display()
        )
    })?;
    let providers_value = object
        .entry("providers".to_string())
        .or_insert_with(|| serde_json::json!({}));
    let providers = match providers_value.as_object_mut() {
        Some(providers) => providers,
        None => bail!("models.json \"providers\" must be a JSON object of provider configs"),
    };

    let mut report = ImportReport::default();
    let mut conflicts: Vec<String> = Vec::new();
    for (id, provider) in &file.providers {
        let effective = apply_gateway(provider, &file.gateway);
        let write = match providers.get(id.as_str()) {
            None => {
                report.imported.push(id.clone());
                true
            }
            Some(existing) if credentials_match(existing, &effective) => {
                report.skipped_unchanged.push(id.clone());
                false
            }
            Some(_) if !force => {
                conflicts.push(id.clone());
                false
            }
            Some(_) => {
                report.replaced.push(id.clone());
                true
            }
        };
        if write {
            providers.insert(
                id.clone(),
                serde_json::to_value(&effective)
                    .context("Failed to serialize an imported provider")?,
            );
        }
    }
    if !conflicts.is_empty() {
        bail!(
            "Refusing to import: models.json already holds different credentials for: {}. \
             Re-run with --force to replace them",
            conflicts.join(", ")
        );
    }

    write_models_config(models_json_path, &root)?;

    // The imported file embeds live gateway credentials; tighten it on unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(auth_file_path, std::fs::Permissions::from_mode(0o600));
    }

    Ok(report)
}

/// Collect every structural problem of an auth file at once: the exact format
/// id, at least one provider, a non-empty http(s) gateway baseUrl, and an
/// `api` or `models` field on every provider.
pub fn validate_dgx_auth(file: &DgxAuthFile) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if file.format != DGX_AUTH_FORMAT_ID {
        errors.push(format!("format: expected \"{DGX_AUTH_FORMAT_ID}\""));
    }
    if file.providers.is_empty() {
        errors.push("providers: at least one provider is required".to_string());
    }
    let base_url = file.gateway.base_url.trim();
    if base_url.is_empty() {
        errors.push("gateway baseUrl: must not be empty".to_string());
    } else if !is_http_url(base_url) {
        errors.push("gateway baseUrl: must be an http or https URL".to_string());
    }
    for (id, provider) in &file.providers {
        let has_api = provider
            .api
            .as_deref()
            .map(|api| !api.trim().is_empty())
            .unwrap_or(false);
        let has_models = provider
            .models
            .as_ref()
            .map(|models| !models.is_empty())
            .unwrap_or(false);
        if !has_api && !has_models {
            errors.push(format!("provider {id}: must define \"api\" or \"models\""));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Value-free stand-in for a credential, for CLI output: reports only the length.
pub fn redact_secret(value: &str) -> String {
    format!("<redacted len={}>", value.chars().count())
}

/// True when the file's best-effort expiry decodes to a past timestamp. A
/// missing or unparseable `expiresAt` is never reported as expired: the
/// gateway session bound stays unknown, and callers decide their messaging.
pub fn dgx_credentials_expired(file: &DgxAuthFile) -> bool {
    match file
        .expires_at
        .as_deref()
        .and_then(|expires_at| DateTime::parse_from_rfc3339(expires_at).ok())
    {
        Some(expires_at) => expires_at < Utc::now(),
        None => false,
    }
}

/// Run the credential refresh command configured in
/// [`DGX_AUTH_REFRESH_COMMAND_ENV`]. `Ok(false)` means no command is
/// configured; `Ok(true)` means the command exited `0` and callers should
/// re-read models.json before exporting. Stdout and stderr are discarded
/// without capture: a misbehaving script may echo credential material.
pub fn refresh_dgx_credentials() -> anyhow::Result<bool> {
    refresh_dgx_credentials_within(Duration::from_secs(REFRESH_TIMEOUT_SECS))
}

/// The refresh engine, parameterized only so tests can pin the timeout; the
/// production entry point fixes it at [`REFRESH_TIMEOUT_SECS`].
fn refresh_dgx_credentials_within(timeout: Duration) -> anyhow::Result<bool> {
    let command = std::env::var(DGX_AUTH_REFRESH_COMMAND_ENV).unwrap_or_default();
    if command.trim().is_empty() {
        return Ok(false);
    }

    let started = Instant::now();
    let program;
    let mut arguments: Vec<String> = Vec::new();
    #[cfg(unix)]
    {
        program = "sh";
        arguments.push("-c".to_string());
    }
    #[cfg(not(unix))]
    {
        program = "cmd";
        arguments.push("/C".to_string());
    }
    arguments.push(command);

    let mut process = Command::new(program)
        .args(&arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to launch the DGX credential refresh command")?;

    loop {
        if let Some(status) = process.try_wait()? {
            return if status.success() {
                Ok(true)
            } else {
                bail!(
                    "DGX credential refresh command failed (exit {})",
                    status.code().unwrap_or(-1)
                )
            };
        }
        if started.elapsed() >= timeout {
            let _ = process.kill();
            // Reap so the timed-out child does not linger as a zombie.
            let _ = process.wait();
            bail!("DGX credential refresh command timed out");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Whether the headers carry an `X-Pomerium-Authorization` value (any casing).
pub fn has_pomerium_header(headers: Option<&IndexMap<String, String>>) -> bool {
    pomerium_token(headers).is_some()
}

/// Best-effort expiry of the `X-Pomerium-Authorization` JWT: decodes the
/// payload `exp` claim without signature verification. `None` on any failure.
pub fn pomerium_expiry_hint(headers: Option<&IndexMap<String, String>>) -> Option<String> {
    pomerium_token(headers).and_then(jwt_expiry_rfc3339)
}

fn pomerium_token(headers: Option<&IndexMap<String, String>>) -> Option<&str> {
    let headers = headers?;
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(POMERIUM_HEADER))
        .map(|(_, value)| value.as_str())
}

fn jwt_expiry_rfc3339(token: &str) -> Option<String> {
    use base64::Engine;
    let token = token.strip_prefix("Bearer ").unwrap_or(token).trim();
    let payload = token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;
    let expires_at = claims.get("exp")?.as_i64()?;
    let timestamp = DateTime::<Utc>::from_timestamp(expires_at, 0)?;
    Some(timestamp.to_rfc3339_opts(SecondsFormat::Secs, true))
}

fn is_dgx_provider_id(id: &str) -> bool {
    id.starts_with("dgx-")
}

fn is_http_url(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    lowered.starts_with("http://") || lowered.starts_with("https://")
}

fn same_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn read_file_content(path: &Path, label: &str) -> anyhow::Result<String> {
    if !path.exists() {
        bail!("{label} not found at {}", path.display());
    }
    std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {label} at {}", path.display()))
}

/// serde diagnostics can quote the offending value verbatim, and any value in
/// these files may be a credential, so only the position and error class are
/// reported.
fn redacted_parse_error(label: &str, error: serde_json::Error) -> anyhow::Error {
    anyhow!(
        "{label}: invalid JSON structure at line {} column {} ({:?})",
        error.line(),
        error.column(),
        error.classify()
    )
}

fn parse_json_strict<T: DeserializeOwned>(label: &str, content: &str) -> anyhow::Result<T> {
    serde_json::from_str(&strip_json_comments(content))
        .map_err(|error| redacted_parse_error(label, error))
}

fn parse_json_or_empty<T: DeserializeOwned + Default>(
    label: &str,
    content: &str,
) -> anyhow::Result<T> {
    let stripped = strip_json_comments(content);
    if stripped.trim().is_empty() {
        return Ok(T::default());
    }
    serde_json::from_str(&stripped).map_err(|error| redacted_parse_error(label, error))
}

fn select_dgx_providers(
    config: &ModelsConfig,
    requested: Option<&[String]>,
) -> anyhow::Result<Vec<(String, ProviderConfig)>> {
    let available: Vec<&str> = config
        .providers
        .keys()
        .map(String::as_str)
        .filter(|id| is_dgx_provider_id(id))
        .collect();
    let selected: Vec<&String> = match requested {
        None => {
            if available.is_empty() {
                bail!("No DGX providers (ids starting with \"dgx-\") are defined in models.json");
            }
            config
                .providers
                .keys()
                .filter(|id| is_dgx_provider_id(id))
                .collect()
        }
        Some(ids) => {
            if ids.is_empty() {
                bail!("No DGX providers selected; pass at least one provider id");
            }
            let mut unique: Vec<&str> = Vec::new();
            for id in ids {
                let id = id.as_str();
                if !unique.contains(&id) {
                    unique.push(id);
                }
                match config.providers.get(id) {
                    Some(_) if is_dgx_provider_id(id) => {}
                    Some(_) => bail!(
                        "Provider \"{id}\" in models.json is not a DGX provider (ids must start with \"dgx-\"). \
                         Available DGX providers: {}",
                        format_available(&available)
                    ),
                    None => bail!(
                        "DGX provider \"{id}\" not found in models.json. Available DGX providers: {}",
                        format_available(&available)
                    ),
                }
            }
            config
                .providers
                .keys()
                .filter(|id| unique.contains(&id.as_str()))
                .collect()
        }
    };

    Ok(selected
        .into_iter()
        .map(|id| (id.clone(), config.providers[id].clone()))
        .collect())
}

fn format_available(available: &[&str]) -> String {
    if available.is_empty() {
        "(none defined in models.json)".to_string()
    } else {
        available.join(", ")
    }
}

/// Resolve the command-backed credentials of the selected providers into
/// inline values. models.json blocks may carry `"!"`-prefixed apiKey/header
/// values that run a credential helper on the machine where models.json
/// lives (`core::resolve_config_value`); copying those verbatim into the
/// auth file would export this machine's helper paths, and every recipient
/// request would then fail to resolve them. Inline values pass through
/// unchanged, and the gateway headers are taken from the first provider's
/// resolved headers, so they are covered too. Unresolvable values are
/// collected and fail the export before any file is written; the errors name
/// only provider ids and field kinds, never the commands or their output
/// (both may be secrets).
fn resolve_command_backed_credentials(
    selected: &[(String, ProviderConfig)],
) -> anyhow::Result<Vec<(String, ProviderConfig)>> {
    let mut resolved: Vec<(String, ProviderConfig)> = Vec::with_capacity(selected.len());
    let mut unresolvable: Vec<String> = Vec::new();
    for (id, provider) in selected {
        let mut provider = provider.clone();
        if let Some(api_key) = provider.api_key.as_deref() {
            match resolve_command_backed_value(api_key) {
                Some(value) => provider.api_key = Some(value),
                None => unresolvable.push(format!("apiKey for provider {id}")),
            }
        }
        if let Some(headers) = provider.headers.as_mut() {
            unresolvable.extend(resolve_command_backed_headers(headers).into_iter().map(
                |name| format!("header \"{name}\" for provider {id}"),
            ));
        }
        for model in provider.models.iter_mut().flatten() {
            if let Some(headers) = model.headers.as_mut() {
                unresolvable.extend(resolve_command_backed_headers(headers).into_iter().map(
                    |name| {
                        format!("model header \"{name}\" for model {} of provider {id}", model.id)
                    },
                ));
            }
        }
        for (model_id, model_override) in provider.model_overrides.iter_mut().flatten() {
            if let Some(headers) = model_override.headers.as_mut() {
                unresolvable.extend(resolve_command_backed_headers(headers).into_iter().map(
                    |name| {
                        format!(
                            "model override header \"{name}\" for model {model_id} of provider {id}"
                        )
                    },
                ));
            }
        }
        resolved.push((id.clone(), provider));
    }
    if !unresolvable.is_empty() {
        bail!(
            "Failed to resolve command-backed {}; cannot build a portable auth file",
            unresolvable.join(", ")
        );
    }
    Ok(resolved)
}

/// Resolve one credential value for the auth file: `"!"`-prefixed values run
/// their helper command through the shared resolver, anything else passes
/// through verbatim. Uncached on purpose: an export is a one-shot snapshot
/// taken after an optional credential refresh, so a helper must be
/// re-executed rather than served from a per-request cache. `None` means the
/// helper failed, timed out, or printed nothing.
fn resolve_command_backed_value(value: &str) -> Option<String> {
    if is_command_reference(value) {
        resolve_config_value_uncached(value)
    } else {
        Some(value.to_string())
    }
}

/// Resolve every `"!"`-prefixed header value in place, returning the names
/// of the headers whose helper produced no value.
fn resolve_command_backed_headers(headers: &mut IndexMap<String, String>) -> Vec<String> {
    let mut unresolvable: Vec<String> = Vec::new();
    for (name, value) in headers.iter_mut() {
        match resolve_command_backed_value(value.as_str()) {
            Some(resolved) => *value = resolved,
            None => unresolvable.push(name.clone()),
        }
    }
    unresolvable
}

/// The exported gateway must be shared: every selected provider has to carry
/// the same baseUrl and headers, or the file could not re-apply one gateway.
fn shared_gateway(selected: &[(String, ProviderConfig)]) -> anyhow::Result<DgxGateway> {
    let (first_id, first) = &selected[0];
    let Some(base_url) = first
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
    else {
        bail!("DGX provider \"{first_id}\" has no baseUrl, so a shared gateway cannot be exported");
    };

    let mut conflicting: Vec<&str> = Vec::new();
    for (id, provider) in &selected[1..] {
        let other_base_url = provider
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|url| !url.is_empty());
        if other_base_url != Some(base_url)
            || !headers_equivalent(first.headers.as_ref(), provider.headers.as_ref())
        {
            conflicting.push(id);
        }
    }
    if !conflicting.is_empty() {
        // The first id is the reference config; every listed id disagrees with it.
        let mut ids = vec![first_id.as_str()];
        ids.extend(conflicting);
        bail!(
            "DGX providers disagree on gateway baseUrl or headers, so no shared gateway can be exported: {}",
            ids.join(", ")
        );
    }

    Ok(DgxGateway {
        base_url: base_url.to_string(),
        headers: first.headers.clone().unwrap_or_default(),
    })
}

/// Order-insensitive header-map equality; `None` and an empty map are equal.
fn headers_equivalent(
    left: Option<&IndexMap<String, String>>,
    right: Option<&IndexMap<String, String>>,
) -> bool {
    let left = left.filter(|headers| !headers.is_empty());
    let right = right.filter(|headers| !headers.is_empty());
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.len() == right.len()
                && left.iter().all(|(name, value)| {
                    right.get(name).map(String::as_str) == Some(value.as_str())
                })
        }
        _ => false,
    }
}

/// The provider an import writes: the file's provider with the gateway
/// baseUrl and the gateway headers applied on top of its own.
fn apply_gateway(provider: &ProviderConfig, gateway: &DgxGateway) -> ProviderConfig {
    let mut merged = provider.clone();
    merged.base_url = Some(gateway.base_url.clone());
    let mut headers = merged.headers.take().unwrap_or_default();
    for (name, value) in &gateway.headers {
        headers.insert(name.clone(), value.clone());
    }
    merged.headers = (!headers.is_empty()).then_some(headers);
    merged
}

/// Whether the existing models.json entry already carries the imported
/// credentials (apiKey + headers). Only credentials are compared; malformed
/// values never match, so a conflict is reported instead of a silent skip.
fn credentials_match(existing: &Value, effective: &ProviderConfig) -> bool {
    let existing_key = existing.get("apiKey").filter(|value| !value.is_null());
    let key_matches = match (existing_key, effective.api_key.as_deref()) {
        (None, None) => true,
        (Some(Value::String(existing)), Some(effective)) => existing == effective,
        _ => false,
    };
    let existing_headers = existing
        .get("headers")
        .and_then(Value::as_object)
        .filter(|headers| !headers.is_empty());
    let effective_headers = effective
        .headers
        .as_ref()
        .filter(|headers| !headers.is_empty());
    let headers_match = match (existing_headers, effective_headers) {
        (None, None) => true,
        (Some(existing), Some(effective)) => {
            existing.len() == effective.len()
                && effective.iter().all(|(name, value)| {
                    existing.get(name).and_then(Value::as_str) == Some(value.as_str())
                })
        }
        _ => false,
    };
    key_matches && headers_match
}

/// Whether the value is a command reference: `"!"`-prefixed config values
/// execute a helper command instead of being used literally
/// (`core::resolve_config_value`).
fn is_command_reference(value: &str) -> bool {
    value.starts_with('!')
}

/// Whether any header value in the map is a command reference.
fn headers_reference_commands(headers: Option<&IndexMap<String, String>>) -> bool {
    headers.is_some_and(|headers| headers.values().any(|value| is_command_reference(value)))
}

/// Whether any credential of the provider is still a command reference: the
/// apiKey, the provider headers, or any model-level or model-override header
/// value.
fn provider_references_commands(provider: &ProviderConfig) -> bool {
    provider.api_key.as_deref().is_some_and(is_command_reference)
        || headers_reference_commands(provider.headers.as_ref())
        || provider.models.as_ref().is_some_and(|models| {
            models
                .iter()
                .any(|model| headers_reference_commands(model.headers.as_ref()))
        })
        || provider.model_overrides.as_ref().is_some_and(|overrides| {
            overrides
                .values()
                .any(|model_override| headers_reference_commands(model_override.headers.as_ref()))
        })
}

/// Provider ids whose imported credentials would still be command references.
/// The gateway headers are applied to every provider on import, so a
/// command-backed gateway value flags every provider in the file.
fn command_backed_provider_ids(file: &DgxAuthFile) -> Vec<&str> {
    let gateway_references_commands = file
        .gateway
        .headers
        .values()
        .any(|value| is_command_reference(value));
    file.providers
        .iter()
        .filter(|(_, provider)| {
            gateway_references_commands || provider_references_commands(provider)
        })
        .map(|(id, _)| id.as_str())
        .collect()
}

/// The models.json root an import merges into: the parsed file when present,
/// an empty object when it is missing or blank.
fn read_models_root(models_json_path: &Path) -> anyhow::Result<Value> {
    if !models_json_path.exists() {
        return Ok(serde_json::json!({}));
    }
    let content = std::fs::read_to_string(models_json_path).with_context(|| {
        format!(
            "Failed to read models.json at {}",
            models_json_path.display()
        )
    })?;
    if strip_json_comments(&content).trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    let root: Value = parse_json_strict("models.json", &content)?;
    if !root.is_object() {
        bail!(
            "models.json at {} is not a JSON object",
            models_json_path.display()
        );
    }
    Ok(root)
}

fn atomic_target(path: &Path) -> String {
    let path = path.to_string_lossy();
    realpath_if_present_sync(&path).unwrap_or_else(|_| path.to_string())
}

fn write_auth_file(out_path: &Path, file: &DgxAuthFile) -> anyhow::Result<()> {
    let mut json =
        serde_json::to_string_pretty(file).context("Failed to serialize the DGX auth file")?;
    json.push('\n');
    // The file embeds live gateway credentials, so it is written owner-only.
    write_file_atomic_sync(
        &atomic_target(out_path),
        &json,
        WriteFileAtomicOptions {
            mode: Some(0o600),
            fsync: true,
            ..Default::default()
        },
    )
    .with_context(|| {
        format!(
            "Failed to write the DGX auth file at {}",
            out_path.display()
        )
    })
}

fn write_models_config(models_json_path: &Path, root: &Value) -> anyhow::Result<()> {
    let mut json = serde_json::to_string_pretty(root).context("Failed to serialize models.json")?;
    json.push('\n');
    let target = atomic_target(models_json_path);
    // Keep the permissions of an existing models.json; a newly created one
    // carries credentials, so it starts owner-only.
    #[cfg(unix)]
    let mode = {
        use std::os::unix::fs::PermissionsExt;
        Some(
            std::fs::metadata(&target)
                .map(|metadata| metadata.permissions().mode() & 0o777)
                .unwrap_or(0o600),
        )
    };
    #[cfg(not(unix))]
    let mode: Option<u32> = None;
    write_file_atomic_sync(
        &target,
        &json,
        WriteFileAtomicOptions {
            mode,
            fsync: true,
            ..Default::default()
        },
    )
    .with_context(|| {
        format!(
            "Failed to write models.json at {}",
            models_json_path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GATEWAY_URL: &str = "https://dgx-gateway.example.test/v1";
    const SECRET_KEY: &str = "TEST-SECRET-gateway-key";
    const SECRET_SIGNATURE: &str = "TEST-SECRET-signature";

    fn test_jwt(expires_at: Option<i64>) -> String {
        use base64::Engine;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let claims = match expires_at {
            Some(exp) => json!({"exp": exp}).to_string(),
            None => json!({"sub": "TEST-SECRET-subject"}).to_string(),
        };
        format!(
            "{}.{}.{}",
            engine.encode("{\"alg\":\"none\"}"),
            engine.encode(claims),
            SECRET_SIGNATURE
        )
    }

    fn dgx_provider(base_url: &str, api_key: &str, token: &str) -> ProviderConfig {
        serde_json::from_value(json!({
            "name": "DGX TEST",
            "baseUrl": base_url,
            "apiKey": api_key,
            "api": "openai-completions",
            "headers": {"X-Pomerium-Authorization": token},
            "models": [{
                "id": "glm-5.3",
                "name": "GLM-5.3 - DGX MTP",
                "reasoning": true,
                "input": ["text"],
                "contextWindow": 1048576,
                "maxTokens": 262144
            }]
        }))
        .unwrap()
    }

    fn write_models_json(dir: &Path, providers: Value) -> std::path::PathBuf {
        let path = dir.join("models.json");
        std::fs::write(&path, json!({"providers": providers}).to_string()).unwrap();
        path
    }

    fn live_models(dir: &Path) -> std::path::PathBuf {
        write_models_json(
            dir,
            json!({
                "other": {"api": "openai-completions", "apiKey": "TEST-SECRET-other"},
                "dgx-a": dgx_provider(GATEWAY_URL, SECRET_KEY, &test_jwt(Some(1760000000))),
                "dgx-b": dgx_provider(GATEWAY_URL, SECRET_KEY, &test_jwt(Some(1760000000)))
            }),
        )
    }

    fn auth_file_base(providers: Value) -> DgxAuthFile {
        serde_json::from_value(json!({
            "format": "optimus-dgx-auth/1",
            "createdAt": "2026-11-02T00:00:00Z",
            "gateway": {
                "baseUrl": GATEWAY_URL,
                "headers": {"X-Pomerium-Authorization": "TEST-SECRET-jwt"}
            },
            "providers": providers
        }))
        .unwrap()
    }

    #[test]
    fn redact_secret_reports_only_the_length() {
        assert_eq!(redact_secret("TEST-SECRET-abc"), "<redacted len=15>");
        assert!(!redact_secret("TEST-SECRET-abc").contains("TEST"));
    }

    #[test]
    fn jwt_expiry_decodes_the_exp_claim_best_effort() {
        assert_eq!(
            jwt_expiry_rfc3339(&test_jwt(Some(1760000000))).as_deref(),
            Some("2025-10-09T08:53:20Z")
        );
        assert_eq!(
            jwt_expiry_rfc3339(&format!("Bearer {}", test_jwt(Some(1760000000)))).as_deref(),
            Some("2025-10-09T08:53:20Z")
        );
        assert_eq!(jwt_expiry_rfc3339(&test_jwt(None)), None);
        assert_eq!(jwt_expiry_rfc3339("TEST-SECRET-not-a-jwt"), None);
    }

    #[test]
    fn validate_accepts_a_well_formed_file() {
        let file = auth_file_base(
            json!({"dgx-a": dgx_provider(GATEWAY_URL, SECRET_KEY, "TEST-SECRET-jwt")}),
        );
        assert_eq!(validate_dgx_auth(&file), Ok(()));
    }

    #[test]
    fn validate_collects_every_error_at_once() {
        let mut file = auth_file_base(json!({}));
        file.format = "optimus-dgx-auth/2".to_string();
        file.gateway.base_url = "ftp://gateway.example.test".to_string();
        let errors = validate_dgx_auth(&file).unwrap_err();
        assert_eq!(errors.len(), 3, "{errors:?}");
        assert!(errors.iter().any(|error| error.contains("format")));
        assert!(errors.iter().any(|error| error.contains("baseUrl")));

        let mut provider_only = auth_file_base(json!({"dgx-empty": {"baseUrl": GATEWAY_URL}}));
        provider_only.gateway.base_url = String::new();
        let errors = validate_dgx_auth(&provider_only).unwrap_err();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors.iter().any(|error| error.contains("dgx-empty")));
    }

    #[test]
    fn export_snapshots_the_shared_gateway_and_dgx_providers() {
        let dir = tempfile::tempdir().unwrap();
        let models_json = live_models(dir.path());
        let out = dir.path().join("dgx-auth.json");

        let file = export_dgx_auth(&models_json, &out, None).unwrap();

        assert_eq!(file.format, "optimus-dgx-auth/1");
        assert_eq!(file.gateway.base_url, GATEWAY_URL);
        assert_eq!(
            file.gateway
                .headers
                .get("X-Pomerium-Authorization")
                .map(String::as_str),
            Some(test_jwt(Some(1760000000)).as_str())
        );
        assert_eq!(file.expires_at.as_deref(), Some("2025-10-09T08:53:20Z"));
        let ids: Vec<&str> = file.providers.keys().map(String::as_str).collect();
        assert_eq!(ids, vec!["dgx-a", "dgx-b"]);
        assert!(!file.providers.contains_key("other"));

        let written: DgxAuthFile =
            parse_json_strict("DGX auth file", &std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(written, file);
    }

    #[test]
    fn export_honors_the_provider_subset_and_reports_missing_ids() {
        let dir = tempfile::tempdir().unwrap();
        let models_json = live_models(dir.path());
        let out = dir.path().join("dgx-auth.json");

        let file = export_dgx_auth(&models_json, &out, Some(&["dgx-b".to_string()])).unwrap();
        assert_eq!(file.providers.len(), 1);
        assert!(file.providers.contains_key("dgx-b"));

        let missing = export_dgx_auth(&models_json, &out, Some(&["dgx-nope".to_string()]))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("dgx-nope"));
        assert!(missing.contains("dgx-a") && missing.contains("dgx-b"));
        assert!(!missing.contains(SECRET_KEY));

        let not_dgx = export_dgx_auth(&models_json, &out, Some(&["other".to_string()]))
            .unwrap_err()
            .to_string();
        assert!(not_dgx.contains("not a DGX provider"));

        let none = export_dgx_auth(&models_json, &out, Some(&[]))
            .unwrap_err()
            .to_string();
        assert!(none.contains("at least one"));
    }

    #[test]
    fn export_refuses_providers_that_disagree_on_the_gateway() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dgx-auth.json");

        let models_json = write_models_json(
            dir.path(),
            json!({
                "dgx-a": dgx_provider(GATEWAY_URL, SECRET_KEY, &test_jwt(Some(1760000000))),
                "dgx-b": dgx_provider("https://other-gateway.example.test/v1", SECRET_KEY, &test_jwt(Some(1760000000)))
            }),
        );
        let error = export_dgx_auth(&models_json, &out, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("dgx-a") && error.contains("dgx-b"));
        assert!(!error.contains("other-gateway"));
        assert!(!error.contains(SECRET_SIGNATURE));

        let models_json = write_models_json(
            dir.path(),
            json!({
                "dgx-a": dgx_provider(GATEWAY_URL, SECRET_KEY, &test_jwt(Some(1760000000))),
                "dgx-b": dgx_provider(GATEWAY_URL, SECRET_KEY, "TEST-SECRET-other-jwt")
            }),
        );
        let error = export_dgx_auth(&models_json, &out, None)
            .unwrap_err()
            .to_string();
        assert!(!error.contains("TEST-SECRET"));
        assert!(!out.exists());
    }

    #[test]
    fn export_refuses_to_write_an_invalid_auth_file() {
        let dir = tempfile::tempdir().unwrap();
        let models_json = write_models_json(
            dir.path(),
            json!({
                "dgx-a": {
                    "baseUrl": GATEWAY_URL,
                    "apiKey": SECRET_KEY,
                    "headers": {"X-Pomerium-Authorization": "TEST-SECRET-jwt"}
                }
            }),
        );
        let out = dir.path().join("dgx-auth.json");
        let error = export_dgx_auth(&models_json, &out, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("dgx-a"));
        assert!(!out.exists());
    }

    #[test]
    fn export_refuses_to_overwrite_the_models_json_source() {
        let dir = tempfile::tempdir().unwrap();
        let models_json = live_models(dir.path());
        let error = export_dgx_auth(&models_json, &models_json, None).unwrap_err();
        assert!(error.to_string().contains("must not be"));
    }

    #[test]
    fn import_creates_models_json_and_reports_inserted_ids() {
        let source = tempfile::tempdir().unwrap();
        let models_json = live_models(source.path());
        let auth_file = source.path().join("dgx-auth.json");
        export_dgx_auth(&models_json, &auth_file, None).unwrap();

        let target = tempfile::tempdir().unwrap();
        let target_models = target.path().join("models.json");
        let report = import_dgx_auth(&auth_file, &target_models, false).unwrap();
        assert_eq!(
            report.imported,
            vec!["dgx-a".to_string(), "dgx-b".to_string()]
        );
        assert!(report.replaced.is_empty() && report.skipped_unchanged.is_empty());

        let root: Value = parse_json_strict(
            "models.json",
            &std::fs::read_to_string(&target_models).unwrap(),
        )
        .unwrap();
        let providers = root.get("providers").unwrap();
        for id in ["dgx-a", "dgx-b"] {
            let provider = providers.get(id).unwrap();
            assert_eq!(
                provider.get("baseUrl").and_then(Value::as_str),
                Some(GATEWAY_URL)
            );
            assert_eq!(
                provider.get("apiKey").and_then(Value::as_str),
                Some(SECRET_KEY)
            );
            assert_eq!(provider.get("models").unwrap().as_array().unwrap().len(), 1);
        }
        assert!(providers.get("other").is_none());
    }

    #[test]
    fn import_skips_unchanged_providers_and_preserves_unrelated_ones() {
        let source = tempfile::tempdir().unwrap();
        let models_json = live_models(source.path());
        let auth_file = source.path().join("dgx-auth.json");
        export_dgx_auth(&models_json, &auth_file, None).unwrap();

        let target = tempfile::tempdir().unwrap();
        let target_models = write_models_json(
            target.path(),
            json!({
                "other": {
                    "api": "openai-completions",
                    "apiKey": "TEST-SECRET-other",
                    "x-custom-field": "keep-me"
                }
            }),
        );
        import_dgx_auth(&auth_file, &target_models, false).unwrap();
        let report = import_dgx_auth(&auth_file, &target_models, false).unwrap();
        assert_eq!(
            report.skipped_unchanged,
            vec!["dgx-a".to_string(), "dgx-b".to_string()]
        );
        assert!(report.imported.is_empty() && report.replaced.is_empty());

        let root: Value = parse_json_strict(
            "models.json",
            &std::fs::read_to_string(&target_models).unwrap(),
        )
        .unwrap();
        let other = root.get("providers").unwrap().get("other").unwrap();
        assert_eq!(
            other.get("x-custom-field").and_then(Value::as_str),
            Some("keep-me")
        );
    }

    #[test]
    fn import_refuses_conflicting_credentials_unless_forced() {
        let source = tempfile::tempdir().unwrap();
        let models_json = live_models(source.path());
        let auth_file = source.path().join("dgx-auth.json");
        export_dgx_auth(&models_json, &auth_file, None).unwrap();

        let target = tempfile::tempdir().unwrap();
        let target_models = write_models_json(
            target.path(),
            json!({
                "dgx-a": dgx_provider("https://old-gateway.example.test/v1", "TEST-SECRET-old-key", "TEST-SECRET-old-jwt"),
                "dgx-b": dgx_provider(GATEWAY_URL, SECRET_KEY, &test_jwt(Some(1760000000)))
            }),
        );

        let error = import_dgx_auth(&auth_file, &target_models, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("dgx-a"));
        assert!(!error.contains("dgx-b"));
        assert!(!error.contains("TEST-SECRET"));

        let report = import_dgx_auth(&auth_file, &target_models, true).unwrap();
        assert_eq!(report.replaced, vec!["dgx-a".to_string()]);
        assert_eq!(report.skipped_unchanged, vec!["dgx-b".to_string()]);

        let root: Value = parse_json_strict(
            "models.json",
            &std::fs::read_to_string(&target_models).unwrap(),
        )
        .unwrap();
        let provider = root.get("providers").unwrap().get("dgx-a").unwrap();
        assert_eq!(
            provider.get("baseUrl").and_then(Value::as_str),
            Some(GATEWAY_URL)
        );
    }

    #[test]
    fn import_rejects_auth_files_with_non_dgx_providers() {
        let dir = tempfile::tempdir().unwrap();
        let file = auth_file_base(
            json!({"not-dgx": dgx_provider(GATEWAY_URL, SECRET_KEY, "TEST-SECRET-jwt")}),
        );
        let path = dir.path().join("auth.json");
        std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();
        let target = dir.path().join("models.json");
        let error = import_dgx_auth(&path, &target, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not-dgx"));
        assert!(!target.exists());
    }

    #[test]
    fn import_rejects_invalid_auth_files() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("models.json");
        let mut file = auth_file_base(
            json!({"dgx-a": dgx_provider(GATEWAY_URL, SECRET_KEY, "TEST-SECRET-jwt")}),
        );
        file.format = "optimus-dgx-auth/2".to_string();
        let path = dir.path().join("auth.json");
        std::fs::write(&path, serde_json::to_string(&file).unwrap()).unwrap();
        let error = import_dgx_auth(&path, &target, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("optimus-dgx-auth/1"));
        assert!(!target.exists());
    }

    #[test]
    fn import_treats_a_blank_models_json_as_empty() {
        let source = tempfile::tempdir().unwrap();
        let models_json = live_models(source.path());
        let auth_file = source.path().join("dgx-auth.json");
        export_dgx_auth(&models_json, &auth_file, None).unwrap();

        let target = tempfile::tempdir().unwrap();
        let target_models = target.path().join("models.json");
        std::fs::write(&target_models, "   \n").unwrap();
        let report = import_dgx_auth(&auth_file, &target_models, false).unwrap();
        assert_eq!(report.imported.len(), 2);
    }

    #[test]
    fn pomerium_helpers_treat_header_names_case_insensitively() {
        let mut headers = IndexMap::new();
        headers.insert(
            "x-pomerium-authorization".to_string(),
            test_jwt(Some(1760000000)),
        );
        assert!(has_pomerium_header(Some(&headers)));
        assert_eq!(
            pomerium_expiry_hint(Some(&headers)).as_deref(),
            Some("2025-10-09T08:53:20Z")
        );
        assert!(!has_pomerium_header(None));
        assert_eq!(pomerium_expiry_hint(None), None);

        let mut upper = IndexMap::new();
        upper.insert(
            "X-POMERIUM-AUTHORIZATION".to_string(),
            test_jwt(Some(1760000000)),
        );
        assert_eq!(
            pomerium_expiry_hint(Some(&upper)).as_deref(),
            Some("2025-10-09T08:53:20Z")
        );
    }

    #[test]
    fn credentials_expired_classifies_only_a_decodable_past_timestamp() {
        let mut file = auth_file_base(
            json!({"dgx-a": dgx_provider(GATEWAY_URL, SECRET_KEY, "TEST-SECRET-jwt")}),
        );

        // A future exp claim is not expired.
        file.expires_at = Some("2999-01-01T00:00:00Z".to_string());
        assert!(!dgx_credentials_expired(&file));

        // A past exp claim is expired.
        file.expires_at = Some("2020-01-01T00:00:00Z".to_string());
        assert!(dgx_credentials_expired(&file));

        // Missing or unparseable expiry is never reported as expired.
        file.expires_at = None;
        assert!(!dgx_credentials_expired(&file));
        file.expires_at = Some("not-a-timestamp".to_string());
        assert!(!dgx_credentials_expired(&file));
    }

    #[test]
    fn refresh_reports_no_command_when_the_env_var_is_unset_or_empty() {
        let previous = std::env::var(DGX_AUTH_REFRESH_COMMAND_ENV).ok();
        std::env::remove_var(DGX_AUTH_REFRESH_COMMAND_ENV);
        assert!(!refresh_dgx_credentials().unwrap());
        std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, "   ");
        assert!(!refresh_dgx_credentials().unwrap());
        match previous {
            Some(value) => std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, value),
            None => std::env::remove_var(DGX_AUTH_REFRESH_COMMAND_ENV),
        }
    }

    #[test]
    fn refresh_runs_the_command_line_through_the_platform_shell() {
        let previous = std::env::var(DGX_AUTH_REFRESH_COMMAND_ENV).ok();
        // A fake command that always exits 0; the interpreter path varies by
        // platform, so the platform shell resolves the binary.
        // `exit 0` is a builtin on both `cmd /C` and `sh -c`.
        std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, "exit 0");
        assert!(refresh_dgx_credentials().unwrap());
        match previous {
            Some(value) => std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, value),
            None => std::env::remove_var(DGX_AUTH_REFRESH_COMMAND_ENV),
        }
    }

    #[test]
    fn refresh_reports_the_exit_code_of_a_failing_command() {
        let previous = std::env::var(DGX_AUTH_REFRESH_COMMAND_ENV).ok();
        std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, "exit 7");
        let error = refresh_dgx_credentials().unwrap_err().to_string();
        assert!(error.contains("failed (exit 7)"), "{error}");
        match previous {
            Some(value) => std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, value),
            None => std::env::remove_var(DGX_AUTH_REFRESH_COMMAND_ENV),
        }
    }

    #[cfg(unix)]
    #[test]
    fn refresh_times_out_a_hung_command_and_kills_it() {
        let previous = std::env::var(DGX_AUTH_REFRESH_COMMAND_ENV).ok();
        std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, "sleep 600");
        let error = refresh_dgx_credentials_within(Duration::from_millis(200))
            .unwrap_err()
            .to_string();
        assert!(error.contains("timed out"), "{error}");
        match previous {
            Some(value) => std::env::set_var(DGX_AUTH_REFRESH_COMMAND_ENV, value),
            None => std::env::remove_var(DGX_AUTH_REFRESH_COMMAND_ENV),
        }
    }

    #[cfg(unix)]
    #[test]
    fn unix_files_carry_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;

        let source = tempfile::tempdir().unwrap();
        let models_json = live_models(source.path());
        let auth_file = source.path().join("dgx-auth.json");
        export_dgx_auth(&models_json, &auth_file, None).unwrap();
        assert_eq!(mode(&auth_file), 0o600);

        // A loosely permissioned auth file is tightened after import.
        let loose = source.path().join("loose-auth.json");
        let file: DgxAuthFile = parse_json_strict(
            "DGX auth file",
            &std::fs::read_to_string(&auth_file).unwrap(),
        )
        .unwrap();
        std::fs::write(&loose, serde_json::to_string(&file).unwrap()).unwrap();
        std::fs::set_permissions(&loose, std::fs::Permissions::from_mode(0o644)).unwrap();

        let target = tempfile::tempdir().unwrap();
        let target_models = target.path().join("models.json");
        import_dgx_auth(&loose, &target_models, false).unwrap();
        assert_eq!(mode(&loose), 0o600);
        assert_eq!(mode(&target_models), 0o600);
    }
}
