//! `dgx-auth` command group: export, import, and inspect the DGX gateway
//! credentials that back the `dgx-*` custom model providers.
//!
//! Rust-native command group with no TypeScript counterpart. It follows the
//! `mcp` command pattern: `cli/public_command.rs` owns the `"dgx-auth"`
//! dispatch arm, and this module owns argument parsing, output, and the calls
//! into the pinned core API in `core/dgx_auth.rs`. Output carries provider
//! ids, paths, and credential presence only; secret values never reach the
//! console.
//!
//! Export refreshes the gateway credentials first unless `--no-refresh` is
//! passed: `OPTIMUS_DGX_AUTH_REFRESH_CMD` (see `core/dgx_auth.rs`) renews them
//! before the snapshot is taken, so a shared file authenticates for as long as
//! the refreshed session lives.

use std::path::{Path, PathBuf};

use crate::core::dgx_auth::{
    DgxAuthFile, ImportReport, dgx_credentials_expired, export_dgx_auth, import_dgx_auth,
    pomerium_expiry_hint, refresh_dgx_credentials,
};
use crate::core::model_registry::{ModelsConfig, ProviderConfig, strip_json_comments};

/// Case-insensitive match for the gateway session header in models.json
/// provider blocks. `crates/pi-ai` treats values of this header as
/// credentials, and so does every code path here.
const POMERIUM_AUTHORIZATION_HEADER: &str = "x-pomerium-authorization";
/// Display form used in status output (presence only, never the value).
const POMERIUM_AUTHORIZATION_DISPLAY: &str = "X-Pomerium-Authorization";

const EXPORT_USAGE: &str =
    "Usage: dgx-auth export [--out <path>] [--providers <ids>] [--no-refresh]";
const IMPORT_USAGE: &str = "Usage: dgx-auth import <file> [--force]";

/// Callback surface for `console.log`, mirroring the seams
/// `cli/public_command.rs` and `cli/list_models.rs` inject.
pub struct DgxAuthCommandIo<'a> {
    pub log: &'a dyn Fn(&str),
    pub cwd: String,
}

/// Entry point for the `"dgx-auth"` dispatch arm; `args` excludes the command
/// name itself (same contract as `runMcpManagementCommand`).
pub fn run_dgx_auth_command(args: &[String], io: &DgxAuthCommandIo<'_>) -> Result<(), String> {
    let subcommand = args.first().map(String::as_str).unwrap_or_default();
    let rest: &[String] = args.get(1..).unwrap_or(&[]);
    match subcommand {
        "export" => run_export(rest, &models_json_path(), io),
        "import" => run_import(rest, &models_json_path(), io),
        "status" => {
            if !rest.is_empty() {
                return Err("Usage: dgx-auth status.".to_string());
            }
            run_status(&models_json_path(), io)
        }
        _ => Err("Usage: dgx-auth <export|import|status>.".to_string()),
    }
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

/// `ParsedDgxAuthExportArgs`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedExportArgs {
    pub out: Option<String>,
    pub provider_ids: Option<Vec<String>>,
    pub no_refresh: bool,
}

/// Parse `dgx-auth export [--out <path>] [--providers <ids>] [--no-refresh]`.
pub fn parse_export_args(args: &[String]) -> Result<ParsedExportArgs, String> {
    let mut parsed = ParsedExportArgs::default();
    let mut seen: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let option = args[index].clone();
        if !option.starts_with('-') {
            return Err(EXPORT_USAGE.to_string());
        }
        if option == "--no-refresh" {
            if parsed.no_refresh {
                return Err("Duplicate dgx-auth export option: --no-refresh".to_string());
            }
            parsed.no_refresh = true;
            index += 1;
            continue;
        }
        if option != "--out" && option != "--providers" {
            return Err(format!("Unknown dgx-auth export option: {option}"));
        }
        if seen.contains(&option) {
            return Err(format!("Duplicate dgx-auth export option: {option}"));
        }
        seen.push(option.clone());
        index += 1;
        let Some(value) = args.get(index) else {
            return Err(format!("{option} requires a value."));
        };
        if value.is_empty() {
            return Err(format!("{option} requires a value."));
        }
        if option == "--out" {
            parsed.out = Some(value.clone());
        } else {
            parsed.provider_ids = Some(split_provider_ids(value)?);
        }
        index += 1;
    }
    Ok(parsed)
}

/// `dgx-auth export`.
fn run_export(
    args: &[String],
    models_path: &Path,
    io: &DgxAuthCommandIo<'_>,
) -> Result<(), String> {
    run_export_with_refresh(args, models_path, io, &default_refresh)
}

/// The production refresh: the core helper driven by
/// `OPTIMUS_DGX_AUTH_REFRESH_CMD`, mapped onto this module's `String` errors.
fn default_refresh() -> Result<bool, String> {
    refresh_dgx_credentials().map_err(|error| error.to_string())
}

/// Export with the refresh step injected, so the parsing and output stay
/// unit-testable without running the configured refresh command.
fn run_export_with_refresh(
    args: &[String],
    models_path: &Path,
    io: &DgxAuthCommandIo<'_>,
    refresh: &dyn Fn() -> Result<bool, String>,
) -> Result<(), String> {
    let parsed = parse_export_args(args)?;
    if parsed.no_refresh {
        (io.log)("Skipping credential refresh (--no-refresh)");
    } else {
        match refresh() {
            Ok(true) => (io.log)("Credentials refreshed via OPTIMUS_DGX_AUTH_REFRESH_CMD"),
            Ok(false) => (io.log)(
                "No refresh command configured (OPTIMUS_DGX_AUTH_REFRESH_CMD); exporting current credentials",
            ),
            Err(error) => return Err(format!("Failed to refresh DGX credentials: {error}")),
        }
    }
    let out_path = match parsed.out {
        Some(out) => PathBuf::from(out),
        None => default_export_path(&io.cwd),
    };
    let auth_file: DgxAuthFile =
        export_dgx_auth(models_path, &out_path, parsed.provider_ids.as_deref())
            .map_err(|error| format!("Failed to export DGX auth file: {error}"))?;
    (io.log)(&format!("DGX auth file written to {}", out_path.display()));
    (io.log)(&format!(
        "Providers: {}",
        join_ids(auth_file.providers.keys().map(String::as_str))
    ));
    (io.log)(&match auth_file.expires_at.as_deref() {
        Some(expires_at) => format!("Expires: {expires_at}"),
        None => "Expires: expiry unknown (bounded by the gateway session)".to_string(),
    });
    if dgx_credentials_expired(&auth_file) {
        (io.log)(
            "WARNING: the gateway credentials in this file are already expired — the file will not authenticate. Renew credentials and re-export.",
        );
    }
    (io.log)("This file contains live DGX credentials. Share it only with trusted users.");
    Ok(())
}

/// Default export destination: `dgx-auth-YYYYMMDD-HHMMSS.json` in the cwd.
fn default_export_path(cwd: &str) -> PathBuf {
    Path::new(cwd).join(default_export_file_name(&chrono::Utc::now()))
}

/// The timestamped default file name, separate so tests can pin the format.
fn default_export_file_name(now: &chrono::DateTime<chrono::Utc>) -> String {
    format!("dgx-auth-{}.json", now.format("%Y%m%d-%H%M%S"))
}

/// Split a comma-separated `--providers` value into trimmed, deduplicated ids.
fn split_provider_ids(value: &str) -> Result<Vec<String>, String> {
    let mut ids: Vec<String> = Vec::new();
    for id in value.split(',').map(str::trim) {
        if id.is_empty() {
            continue;
        }
        if !ids.iter().any(|seen| seen == id) {
            ids.push(id.to_string());
        }
    }
    if ids.is_empty() {
        return Err("--providers must be a comma-separated list of provider ids.".to_string());
    }
    Ok(ids)
}

// ---------------------------------------------------------------------------
// import
// ---------------------------------------------------------------------------

/// `ParsedDgxAuthImportArgs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedImportArgs {
    pub file: String,
    pub force: bool,
}

/// Parse `dgx-auth import <file> [--force]`.
pub fn parse_import_args(args: &[String]) -> Result<ParsedImportArgs, String> {
    let mut file: Option<String> = None;
    let mut force = false;
    for arg in args {
        if arg == "--force" {
            if force {
                return Err("Duplicate dgx-auth import option: --force".to_string());
            }
            force = true;
            continue;
        }
        if arg.starts_with('-') {
            return Err(format!("Unknown dgx-auth import option: {arg}"));
        }
        if file.is_some() {
            return Err(IMPORT_USAGE.to_string());
        }
        file = Some(arg.clone());
    }
    let Some(file) = file else {
        return Err(IMPORT_USAGE.to_string());
    };
    Ok(ParsedImportArgs { file, force })
}

/// `dgx-auth import`.
fn run_import(
    args: &[String],
    models_path: &Path,
    io: &DgxAuthCommandIo<'_>,
) -> Result<(), String> {
    let parsed = parse_import_args(args)?;
    let report = import_dgx_auth(Path::new(&parsed.file), models_path, parsed.force)
        .map_err(|error| format!("Failed to import DGX auth file: {error}"))?;
    for line in format_import_report(&report) {
        (io.log)(&line);
    }
    Ok(())
}

/// ImportReport ids plus the one-line security note; ids only, never values.
fn format_import_report(report: &ImportReport) -> Vec<String> {
    let mut lines = Vec::new();
    if !report.imported.is_empty() {
        lines.push(format!(
            "Imported: {}",
            join_ids(report.imported.iter().map(String::as_str))
        ));
    }
    if !report.replaced.is_empty() {
        lines.push(format!(
            "Replaced: {}",
            join_ids(report.replaced.iter().map(String::as_str))
        ));
    }
    if !report.skipped_unchanged.is_empty() {
        lines.push(format!(
            "Unchanged: {}",
            join_ids(report.skipped_unchanged.iter().map(String::as_str))
        ));
    }
    lines.push(
        "This file grants access to the sharer's DGX allocation until the gateway credentials expire."
            .to_string(),
    );
    lines
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

/// `dgx-auth status`.
fn run_status(models_path: &Path, io: &DgxAuthCommandIo<'_>) -> Result<(), String> {
    if !models_path.exists() {
        (io.log)(&format!("No models.json at {}.", models_path.display()));
        return Ok(());
    }
    let providers = read_dgx_providers(models_path)?;
    if providers.is_empty() {
        (io.log)(&format!("No DGX providers in {}.", models_path.display()));
        return Ok(());
    }
    (io.log)(&format!("DGX providers in {}:", models_path.display()));
    for (id, config) in &providers {
        for line in format_provider_status(id, config) {
            (io.log)(&line);
        }
    }
    Ok(())
}

/// Read models.json and return its DGX providers sorted by id.
fn read_dgx_providers(models_path: &Path) -> Result<Vec<(String, ProviderConfig)>, String> {
    let content = std::fs::read_to_string(models_path)
        .map_err(|error| format!("Failed to read models.json: {error}"))?;
    let config: ModelsConfig = serde_json::from_str(&strip_json_comments(&content))
        .map_err(|error| format!("Failed to parse models.json: {error}"))?;
    let mut providers: Vec<(String, ProviderConfig)> = config
        .providers
        .into_iter()
        .filter(|(id, _)| is_dgx_provider_id(id))
        .collect();
    providers.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(providers)
}

/// The DGX gateway registrations live under the `dgx-` provider-id prefix,
/// the same rule the core export/import selection uses.
fn is_dgx_provider_id(id: &str) -> bool {
    id.starts_with("dgx-")
}

/// Presence-only status lines for one provider; never includes values.
fn format_provider_status(id: &str, config: &ProviderConfig) -> Vec<String> {
    let mut lines = vec![id.to_string()];
    let api_key_set = config
        .api_key
        .as_deref()
        .map(|key| !key.trim().is_empty())
        .unwrap_or(false);
    lines.push(format!(
        "  apiKey: {}",
        if api_key_set { "set" } else { "not set" }
    ));
    match gateway_authorization(config) {
        Some(header) if !header.trim().is_empty() => {
            lines.push(format!("  {POMERIUM_AUTHORIZATION_DISPLAY}: set"));
            // The core helper decodes the gateway JWT `exp` claim without
            // signature verification; anything else falls back to the bound.
            lines.push(match pomerium_expiry_hint(config.headers.as_ref()) {
                Some(expires_at) => format!("  expiresAt: {expires_at}"),
                None => "  expiresAt: expiry unknown (bounded by the gateway session)".to_string(),
            });
        }
        _ => lines.push(format!("  {POMERIUM_AUTHORIZATION_DISPLAY}: not set")),
    }
    lines
}

/// The gateway session header value, if the provider carries one.
fn gateway_authorization(config: &ProviderConfig) -> Option<String> {
    config.headers.as_ref().and_then(|headers| {
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(POMERIUM_AUTHORIZATION_HEADER))
            .map(|(_, value)| value.clone())
    })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Same default the registry resolves (`ModelRegistry::create`):
/// `<agent dir>/models.json`.
fn models_json_path() -> PathBuf {
    Path::new(&crate::config::get_agent_dir()).join("models.json")
}

/// Sorted, comma-joined ids; "(none)" keeps output readable for an empty set.
fn join_ids<'a>(ids: impl Iterator<Item = &'a str>) -> String {
    let mut ids: Vec<&str> = ids.collect();
    ids.sort_unstable();
    if ids.is_empty() {
        return "(none)".to_string();
    }
    ids.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dgx_auth::redact_secret;
    use std::sync::{Arc, Mutex};

    /// The `console.log` seam for one run, mirroring the public_command tests.
    struct Harness {
        logged: Arc<Mutex<Vec<String>>>,
        log_fn: Box<dyn Fn(&str)>,
        cwd: String,
    }

    impl Harness {
        fn new() -> Self {
            Self::with_cwd("/work")
        }

        fn with_cwd(cwd: &str) -> Self {
            let logged = Arc::new(Mutex::new(Vec::new()));
            let log_fn: Box<dyn Fn(&str)> = {
                let logged = logged.clone();
                Box::new(move |line: &str| logged.lock().unwrap().push(line.to_string()))
            };
            Self {
                logged,
                log_fn,
                cwd: cwd.to_string(),
            }
        }

        fn io(&self) -> DgxAuthCommandIo<'_> {
            DgxAuthCommandIo {
                log: &*self.log_fn,
                cwd: self.cwd.clone(),
            }
        }

        fn logged(&self) -> Vec<String> {
            self.logged.lock().unwrap().clone()
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    /// A gateway JWT with a decodable `exp` claim. The header and signature
    /// segments stay opaque so assertions can prove they never print.
    fn test_gateway_jwt(exp: i64) -> String {
        use base64::Engine;
        let payload = serde_json::json!({ "exp": exp }).to_string();
        format!(
            "TEST-SECRET-header.{}.TEST-SECRET-signature",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload)
        )
    }

    /// models.json fixture: the three DGX registrations plus one unrelated
    /// provider. Every DGX provider shares the same gateway (baseUrl +
    /// X-Pomerium-Authorization), the way a live registration does, with
    /// obvious fake credentials only.
    /// Far-future `exp` (2100-01-01T00:00:00Z) so the not-expired fixtures stay
    /// valid regardless of when the suite runs.
    const FUTURE_EXP: i64 = 4102444800;

    fn write_models_fixture(path: &Path) {
        let models = serde_json::json!({
            "providers": {
                "openai": { "apiKey": "TEST-SECRET-openai" },
                "dgx-glm53": {
                    "api": "openai-completions",
                    "baseUrl": "https://dgx.example.test/v1",
                    "apiKey": "TEST-SECRET-gateway-key",
                    "headers": { "X-Pomerium-Authorization": test_gateway_jwt(FUTURE_EXP) },
                    "models": [
                        {
                            "id": "glm-5.3",
                            "name": "GLM-5.3 - DGX MTP",
                            "reasoning": true,
                            "input": ["text"],
                            "contextWindow": 1048576,
                            "maxTokens": 262144
                        }
                    ]
                },
                "dgx-k3": {
                    "name": "DGX K3",
                    "api": "openai-completions",
                    "baseUrl": "https://dgx.example.test/v1",
                    "apiKey": "TEST-SECRET-gateway-key",
                    "headers": { "X-Pomerium-Authorization": test_gateway_jwt(FUTURE_EXP) }
                },
                "dgx-deepseek41": {
                    "api": "openai-completions",
                    "baseUrl": "https://dgx.example.test/v1",
                    "apiKey": "TEST-SECRET-gateway-key",
                    "headers": { "X-Pomerium-Authorization": test_gateway_jwt(FUTURE_EXP) }
                }
            }
        });
        std::fs::write(path, serde_json::to_string_pretty(&models).unwrap()).unwrap();
    }

    #[test]
    fn routing_requires_a_known_subcommand() {
        let harness = Harness::new();
        let usage = "Usage: dgx-auth <export|import|status>.";
        assert_eq!(
            run_dgx_auth_command(&strings(&[]), &harness.io()).unwrap_err(),
            usage
        );
        assert_eq!(
            run_dgx_auth_command(&strings(&["nonsense"]), &harness.io()).unwrap_err(),
            usage
        );
        assert_eq!(
            run_dgx_auth_command(&strings(&["status", "extra"]), &harness.io()).unwrap_err(),
            "Usage: dgx-auth status."
        );
    }

    #[test]
    fn parse_export_args_matches_the_surface() {
        assert_eq!(
            parse_export_args(&strings(&[])).unwrap(),
            ParsedExportArgs::default()
        );

        let parsed = parse_export_args(&strings(&["--out", "/tmp/dgx-auth.json"])).unwrap();
        assert_eq!(parsed.out.as_deref(), Some("/tmp/dgx-auth.json"));
        assert!(parsed.provider_ids.is_none());

        let parsed =
            parse_export_args(&strings(&["--providers", "dgx-k3, dgx-glm53,dgx-k3"])).unwrap();
        assert_eq!(parsed.provider_ids, Some(strings(&["dgx-k3", "dgx-glm53"])));

        let parsed =
            parse_export_args(&strings(&["--providers", "dgx-k3", "--out", "auth.json"])).unwrap();
        assert_eq!(parsed.out.as_deref(), Some("auth.json"));
        assert_eq!(parsed.provider_ids, Some(strings(&["dgx-k3"])));

        let parsed = parse_export_args(&strings(&["--no-refresh"])).unwrap();
        assert!(parsed.no_refresh);

        let parsed = parse_export_args(&strings(&[
            "--no-refresh",
            "--out",
            "auth.json",
            "--providers",
            "dgx-k3",
        ]))
        .unwrap();
        assert!(parsed.no_refresh);
        assert_eq!(parsed.out.as_deref(), Some("auth.json"));
        assert_eq!(parsed.provider_ids, Some(strings(&["dgx-k3"])));
    }

    #[test]
    fn parse_export_args_rejects_invalid_input() {
        assert_eq!(
            parse_export_args(&strings(&["--nope"])).unwrap_err(),
            "Unknown dgx-auth export option: --nope"
        );
        assert_eq!(
            parse_export_args(&strings(&["stray"])).unwrap_err(),
            EXPORT_USAGE
        );
        assert_eq!(
            parse_export_args(&strings(&["--out"])).unwrap_err(),
            "--out requires a value."
        );
        assert_eq!(
            parse_export_args(&strings(&["--out", ""])).unwrap_err(),
            "--out requires a value."
        );
        assert_eq!(
            parse_export_args(&strings(&["--providers", ""])).unwrap_err(),
            "--providers requires a value."
        );
        assert_eq!(
            parse_export_args(&strings(&["--providers", ", ,"])).unwrap_err(),
            "--providers must be a comma-separated list of provider ids."
        );
        assert_eq!(
            parse_export_args(&strings(&["--out", "a.json", "--out", "b.json"])).unwrap_err(),
            "Duplicate dgx-auth export option: --out"
        );
        assert_eq!(
            parse_export_args(&strings(&["--no-refresh", "--no-refresh"])).unwrap_err(),
            "Duplicate dgx-auth export option: --no-refresh"
        );
        assert_eq!(
            parse_export_args(&strings(&["--no-refresh", "stray"])).unwrap_err(),
            EXPORT_USAGE
        );
    }

    #[test]
    fn parse_import_args_matches_the_surface() {
        let parsed = parse_import_args(&strings(&["auth.json"])).unwrap();
        assert_eq!(parsed.file, "auth.json");
        assert!(!parsed.force);

        let parsed = parse_import_args(&strings(&["--force", "auth.json"])).unwrap();
        assert_eq!(parsed.file, "auth.json");
        assert!(parsed.force);

        let parsed = parse_import_args(&strings(&["auth.json", "--force"])).unwrap();
        assert_eq!(parsed.file, "auth.json");
        assert!(parsed.force);
    }

    #[test]
    fn parse_import_args_rejects_invalid_input() {
        assert_eq!(parse_import_args(&strings(&[])).unwrap_err(), IMPORT_USAGE);
        assert_eq!(
            parse_import_args(&strings(&["a.json", "b.json"])).unwrap_err(),
            IMPORT_USAGE
        );
        assert_eq!(
            parse_import_args(&strings(&["--json"])).unwrap_err(),
            "Unknown dgx-auth import option: --json"
        );
        assert_eq!(
            parse_import_args(&strings(&["--force", "--force"])).unwrap_err(),
            "Duplicate dgx-auth import option: --force"
        );
    }

    #[test]
    fn default_export_names_stamp_utc_time_into_the_file_name() {
        use chrono::TimeZone;
        let now = chrono::Utc
            .with_ymd_and_hms(2026, 11, 2, 15, 30, 5)
            .unwrap();
        assert_eq!(
            default_export_file_name(&now),
            "dgx-auth-20261102-153005.json"
        );

        let path = default_export_path("dir");
        assert_eq!(path.parent(), Some(Path::new("dir")));
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("dgx-auth-2"), "{name}");
        assert!(name.ends_with(".json"), "{name}");
    }

    #[test]
    fn status_reports_presence_only_and_never_secret_values() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        let models = serde_json::json!({
            "providers": {
                "openai": { "apiKey": "TEST-SECRET-openai" },
                "dgx-glm53": {
                    "apiKey": "TEST-SECRET-gateway-key",
                    "headers": { "X-Pomerium-Authorization": test_gateway_jwt(FUTURE_EXP) }
                },
                "dgx-k3": {
                    "apiKey": "TEST-SECRET-gateway-key",
                    "headers": { "x-pomerium-authorization": "TEST-SECRET-opaque" }
                },
                "dgx-deepseek41": { "apiKey": "TEST-SECRET-gateway-key" }
            }
        });
        std::fs::write(&models_path, serde_json::to_string_pretty(&models).unwrap()).unwrap();

        let harness = Harness::new();
        run_status(&models_path, &harness.io()).unwrap();

        let logged = harness.logged();
        assert_eq!(
            logged,
            vec![
                format!("DGX providers in {}:", models_path.display()),
                "dgx-deepseek41".to_string(),
                "  apiKey: set".to_string(),
                "  X-Pomerium-Authorization: not set".to_string(),
                "dgx-glm53".to_string(),
                "  apiKey: set".to_string(),
                "  X-Pomerium-Authorization: set".to_string(),
                "  expiresAt: 2100-01-01T00:00:00Z".to_string(),
                "dgx-k3".to_string(),
                "  apiKey: set".to_string(),
                "  X-Pomerium-Authorization: set".to_string(),
                "  expiresAt: expiry unknown (bounded by the gateway session)".to_string(),
            ]
        );
        let printed = logged.join("\n");
        assert!(!printed.contains("TEST-SECRET"), "{printed}");
        assert!(!printed.contains("openai"), "{printed}");
    }

    #[test]
    fn status_treats_blank_credentials_as_not_set() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        let models = serde_json::json!({
            "providers": {
                "dgx-k3": {
                    "apiKey": "   ",
                    "headers": { "x-pomerium-authorization": "   " }
                }
            }
        });
        std::fs::write(&models_path, serde_json::to_string(&models).unwrap()).unwrap();

        let harness = Harness::new();
        run_status(&models_path, &harness.io()).unwrap();
        let printed = harness.logged().join("\n");
        assert!(printed.contains("  apiKey: not set"), "{printed}");
        assert!(
            printed.contains("  X-Pomerium-Authorization: not set"),
            "{printed}"
        );
    }

    #[test]
    fn status_accepts_commented_models_json() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        std::fs::write(
            &models_path,
            "{\n  // custom registry\n  \"providers\": { \"dgx-k3\": { \"apiKey\": \"TEST-SECRET-key\" } }\n}\n",
        )
        .unwrap();

        let harness = Harness::new();
        run_status(&models_path, &harness.io()).unwrap();
        let printed = harness.logged().join("\n");
        assert!(printed.contains("dgx-k3"), "{printed}");
        assert!(printed.contains("  apiKey: set"), "{printed}");
        assert!(!printed.contains("TEST-SECRET"), "{printed}");
    }

    #[test]
    fn status_reports_missing_files_and_absent_providers_without_failing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("models.json");
        let harness = Harness::new();
        run_status(&missing, &harness.io()).unwrap();
        assert!(
            harness.logged()[0].starts_with("No models.json at "),
            "{:?}",
            harness.logged()
        );

        let models_path = dir.path().join("other-models.json");
        std::fs::write(
            &models_path,
            serde_json::to_string(&serde_json::json!({ "providers": { "openai": {} } })).unwrap(),
        )
        .unwrap();
        let harness = Harness::new();
        run_status(&models_path, &harness.io()).unwrap();
        assert!(
            harness.logged()[0].starts_with("No DGX providers in "),
            "{:?}",
            harness.logged()
        );
    }

    #[test]
    fn status_fails_with_a_read_or_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        std::fs::write(&models_path, "{ not json").unwrap();
        let harness = Harness::new();
        let error = run_status(&models_path, &harness.io()).unwrap_err();
        assert!(error.starts_with("Failed to parse models.json:"), "{error}");
    }

    #[test]
    fn export_writes_the_auth_file_and_reports_ids_and_expiry_only() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let out_path = dir.path().join("shared-dgx.json");

        let harness = Harness::new();
        run_export_with_refresh(
            &strings(&["--out", &out_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
            &|| Ok(true),
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(printed.contains(&format!("DGX auth file written to {}", out_path.display())));
        assert!(
            printed.contains("Providers: dgx-deepseek41, dgx-glm53, dgx-k3"),
            "{printed}"
        );
        // RFC3339 UTC per the auth-file format.
        assert!(
            printed.contains("Expires: 2100-01-01T00:00:00Z"),
            "{printed}"
        );
        assert!(
            printed.contains("This file contains live DGX credentials."),
            "{printed}"
        );
        assert!(!printed.contains("TEST-SECRET"), "{printed}");
        assert!(!printed.contains("WARNING"), "{printed}");
        assert!(out_path.exists());
    }

    #[test]
    fn export_runs_the_configured_refresh_command_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let out_path = dir.path().join("shared-dgx.json");

        let harness = Harness::new();
        run_export_with_refresh(
            &strings(&["--out", &out_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
            &|| Ok(true),
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(
            printed.contains("Credentials refreshed via OPTIMUS_DGX_AUTH_REFRESH_CMD"),
            "{printed}"
        );
        assert!(
            !printed.contains("No refresh command configured"),
            "{printed}"
        );
        assert!(
            !printed.contains("Skipping credential refresh"),
            "{printed}"
        );
        assert!(out_path.exists());
    }

    #[test]
    fn export_continues_without_a_configured_refresh_command() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let out_path = dir.path().join("shared-dgx.json");

        let harness = Harness::new();
        run_export_with_refresh(
            &strings(&["--out", &out_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
            &|| Ok(false),
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(
            printed.contains(
                "No refresh command configured (OPTIMUS_DGX_AUTH_REFRESH_CMD); exporting current credentials"
            ),
            "{printed}"
        );
        assert!(out_path.exists());
    }

    #[test]
    fn export_skips_the_refresh_command_with_no_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let out_path = dir.path().join("shared-dgx.json");

        // The injected refresh fails the test when it runs: --no-refresh must
        // not call it.
        let refresh = || -> Result<bool, String> { panic!("--no-refresh must skip the refresh") };
        let harness = Harness::new();
        run_export_with_refresh(
            &strings(&["--no-refresh", "--out", &out_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
            &refresh,
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(
            printed.contains("Skipping credential refresh (--no-refresh)"),
            "{printed}"
        );
        assert!(
            !printed.contains("OPTIMUS_DGX_AUTH_REFRESH_CMD;"),
            "{printed}"
        );
        assert!(out_path.exists());
    }

    #[test]
    fn export_aborts_without_writing_a_file_when_the_refresh_fails() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let out_path = dir.path().join("shared-dgx.json");

        let harness = Harness::new();
        let error = run_export_with_refresh(
            &strings(&["--out", &out_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
            &|| Err("TEST-SECRET-refresh-command-failed".to_string()),
        )
        .unwrap_err();
        assert!(
            error.contains("Failed to refresh DGX credentials"),
            "{error}"
        );
        assert!(
            error.contains("TEST-SECRET-refresh-command-failed"),
            "{error}"
        );
        assert!(!out_path.exists());
        assert!(harness.logged().is_empty());
    }

    #[test]
    fn export_warns_when_the_exported_credentials_are_already_expired() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        let models = serde_json::json!({
            "providers": {
                "dgx-k3": {
                    "api": "openai-completions",
                    "baseUrl": "https://dgx.example.test/v1",
                    "apiKey": "TEST-SECRET-gateway-key",
                    "headers": { "X-Pomerium-Authorization": test_gateway_jwt(1) }
                }
            }
        });
        std::fs::write(&models_path, serde_json::to_string_pretty(&models).unwrap()).unwrap();
        let out_path = dir.path().join("shared-dgx.json");

        let harness = Harness::new();
        run_export_with_refresh(
            &strings(&["--no-refresh", "--out", &out_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
            &|| Ok(true),
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(
            printed.contains(
                "WARNING: the gateway credentials in this file are already expired — the file will not authenticate. Renew credentials and re-export."
            ),
            "{printed}"
        );
        assert!(out_path.exists());
    }

    #[test]
    fn export_defaults_to_a_timestamped_file_in_the_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);

        let harness = Harness::with_cwd(&dir.path().to_string_lossy());
        run_export_with_refresh(&[], &models_path, &harness.io(), &|| Ok(false)).unwrap();

        let written = harness
            .logged()
            .iter()
            .find(|line| line.starts_with("DGX auth file written to "))
            .map(|line| {
                line.strip_prefix("DGX auth file written to ")
                    .unwrap()
                    .to_string()
            })
            .expect("written line");
        let name = Path::new(&written)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(name.starts_with("dgx-auth-2"), "{name}");
        assert!(name.ends_with(".json"), "{name}");
        assert!(Path::new(&written).is_file(), "{written}");
    }

    #[test]
    fn import_reports_provider_ids_and_the_security_note() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        std::fs::write(
            &models_path,
            serde_json::to_string(&serde_json::json!({ "providers": {} })).unwrap(),
        )
        .unwrap();
        let source = dir.path().join("source-models.json");
        write_models_fixture(&source);
        let auth_path = dir.path().join("auth.json");
        export_dgx_auth(&source, &auth_path, None).unwrap();

        let harness = Harness::new();
        run_import(
            &strings(&[&auth_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(
            printed.contains("Imported: dgx-deepseek41, dgx-glm53, dgx-k3"),
            "{printed}"
        );
        assert!(
            printed.contains(
                "This file grants access to the sharer's DGX allocation until the gateway credentials expire."
            ),
            "{printed}"
        );
        assert!(!printed.contains("TEST-SECRET"), "{printed}");
    }

    #[test]
    fn import_refuses_conflicting_credentials_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let source = dir.path().join("source-models.json");
        let conflicting = serde_json::json!({
            "providers": {
                "dgx-glm53": {
                    "api": "openai-completions",
                    "baseUrl": "https://dgx.example.test/v1",
                    "apiKey": "TEST-SECRET-other-key",
                    "headers": { "X-Pomerium-Authorization": test_gateway_jwt(FUTURE_EXP) }
                }
            }
        });
        std::fs::write(&source, serde_json::to_string(&conflicting).unwrap()).unwrap();
        let auth_path = dir.path().join("auth.json");
        export_dgx_auth(&source, &auth_path, None).unwrap();

        let harness = Harness::new();
        let auth_file_arg = auth_path.to_string_lossy().to_string();
        let error =
            run_import(&strings(&[&auth_file_arg]), &models_path, &harness.io()).unwrap_err();
        assert!(error.contains("dgx-glm53"), "{error}");
        assert!(error.contains("--force"), "{error}");
        assert!(!error.contains("TEST-SECRET"), "{error}");

        run_import(
            &strings(&[&auth_file_arg, "--force"]),
            &models_path,
            &harness.io(),
        )
        .unwrap();
        let printed = harness.logged().join("\n");
        assert!(printed.contains("Replaced: dgx-glm53"), "{printed}");
        assert!(!printed.contains("TEST-SECRET"), "{printed}");
    }

    #[test]
    fn import_reports_unchanged_providers_without_values() {
        let dir = tempfile::tempdir().unwrap();
        let models_path = dir.path().join("models.json");
        write_models_fixture(&models_path);
        let auth_path = dir.path().join("auth.json");
        export_dgx_auth(&models_path, &auth_path, None).unwrap();

        let harness = Harness::new();
        run_import(
            &strings(&[&auth_path.to_string_lossy()]),
            &models_path,
            &harness.io(),
        )
        .unwrap();

        let printed = harness.logged().join("\n");
        assert!(
            printed.contains("Unchanged: dgx-deepseek41, dgx-glm53, dgx-k3"),
            "{printed}"
        );
        assert!(
            printed.contains(
                "This file grants access to the sharer's DGX allocation until the gateway credentials expire."
            ),
            "{printed}"
        );
        assert!(!printed.contains("TEST-SECRET"), "{printed}");
    }

    #[test]
    fn format_import_report_lists_each_non_empty_group_once() {
        let report = ImportReport {
            imported: strings(&["dgx-k3"]),
            replaced: strings(&["dgx-glm53", "dgx-deepseek41"]),
            skipped_unchanged: Vec::new(),
        };
        assert_eq!(
            format_import_report(&report),
            vec![
                "Imported: dgx-k3".to_string(),
                "Replaced: dgx-deepseek41, dgx-glm53".to_string(),
                "This file grants access to the sharer's DGX allocation until the gateway credentials expire."
                    .to_string(),
            ]
        );
    }

    #[test]
    fn join_ids_sorts_and_handles_empty_sets() {
        assert_eq!(
            join_ids(["dgx-k3", "dgx-glm53"].into_iter()),
            "dgx-glm53, dgx-k3"
        );
        assert_eq!(join_ids(Vec::<&str>::new().into_iter()), "(none)");
    }

    #[test]
    fn is_dgx_provider_id_matches_the_gateway_prefix() {
        assert!(is_dgx_provider_id("dgx-glm53"));
        assert!(is_dgx_provider_id("dgx-k3"));
        assert!(is_dgx_provider_id("dgx-deepseek41"));
        assert!(!is_dgx_provider_id("dgx"));
        assert!(!is_dgx_provider_id("openai"));
        assert!(!is_dgx_provider_id("my-dgx-clone"));
    }

    #[test]
    fn redact_secret_never_returns_the_value() {
        let redacted = redact_secret("TEST-SECRET-gateway-key");
        assert!(!redacted.contains("TEST-SECRET-gateway-key"));
        assert!(!redacted.is_empty());
    }
}
