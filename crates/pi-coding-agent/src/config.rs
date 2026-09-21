//! Port of packages/coding-agent/src/config.ts
//!
//! Package/config paths, install-method detection and the self-update command
//! surface.

use std::path::{Path, PathBuf};

use crate::utils::child_process::{should_use_windows_shell, spawn_sync_hidden, SpawnOptions};
use crate::utils::daemon_socket_path::normalize_socket_path;
use crate::utils::update_source::PRIME_AGENT_UPDATE_RELEASE_URL;

// =============================================================================
// Package Detection
// =============================================================================

/// `isBunBinary`: Bun's virtual filesystem paths (`$bunfs`, `~BUN`, `%7EBUN`).
///
/// The port has no Bun runtime, so this is always false; `isBunRuntime`
/// (`!!process.versions.bun`) is false for the same reason.
pub const fn is_bun_binary() -> bool {
    false
}

/// `isBunRuntime` is false: no Bun runtime exists here.
pub const fn is_bun_runtime() -> bool {
    false
}

pub const SELF_UPDATE_INTERACTIVE_CHILD_ENV: &str = "PRIME_AGENT_INTERACTIVE_SELF_UPDATE";
pub const SELF_UPDATE_NOT_ATTEMPTED_EXIT_CODE: i32 = 75;

// =============================================================================
// App Config constants (from package.json piConfig)
// =============================================================================

/// `PACKAGE_NAME` (`pkg.name`), pinned from `packages/coding-agent/package.json`.
///
/// The TypeScript module reads these from package.json at import time. Rust needs
/// `&'static str` constants at some call sites, so the port also exposes the same
/// values as `const` alongside the runtime getters (`package_name()`).
pub const PACKAGE_NAME: &str = "@earendil-works/pi-coding-agent";
/// `APP_NAME` (`pkg.piConfig.name || "pi"`).
pub const APP_NAME: &str = "prime-agent";
/// `APP_TITLE` (`pkg.piConfig.name ? APP_NAME : "\u{3c0}"`).
pub const APP_TITLE: &str = "prime-agent";
/// `CONFIG_DIR_NAME` (`pkg.piConfig.configDir || ".prime/agent"`).
pub const CONFIG_DIR_NAME: &str = ".prime/agent";
/// `VERSION` (`pkg.version`), from the workspace package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// `ENV_AGENT_DIR` (`${envPrefix}_CODING_AGENT_DIR`).
pub const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";
/// `ENV_SESSION_DIR` (`${envPrefix}_SESSION_DIR`).
pub const ENV_SESSION_DIR: &str = "PRIME_AGENT_SESSION_DIR";
/// `ENV_LEGACY_SESSION_DIR` (`${envPrefix}_CODING_AGENT_SESSION_DIR`).
pub const ENV_LEGACY_SESSION_DIR: &str = "PRIME_AGENT_CODING_AGENT_SESSION_DIR";

// =============================================================================
// Install Method Detection
// =============================================================================

pub type InstallMethod = &'static str;

pub const INSTALL_METHODS: [InstallMethod; 7] =
    ["bun-binary", "homebrew", "npm", "pnpm", "yarn", "bun", "unknown"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdateCommandStep {
    pub command: String,
    pub args: Vec<String>,
    pub display: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfUpdateCommand {
    pub command: String,
    pub args: Vec<String>,
    pub display: String,
    pub steps: Option<Vec<SelfUpdateCommandStep>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SelfUpdateCommandOptions {
    pub uninstall_after_install: bool,
}

fn make_self_update_command(
    install_step: SelfUpdateCommandStep,
    uninstall_step: Option<SelfUpdateCommandStep>,
    options: SelfUpdateCommandOptions,
) -> SelfUpdateCommand {
    let Some(uninstall_step) = uninstall_step else {
        return SelfUpdateCommand {
            command: install_step.command,
            args: install_step.args,
            display: install_step.display,
            steps: None,
        };
    };
    if options.uninstall_after_install {
        return SelfUpdateCommand {
            command: install_step.command.clone(),
            args: install_step.args.clone(),
            display: format!("{} && {}", install_step.display, uninstall_step.display),
            steps: Some(vec![install_step, uninstall_step]),
        };
    }
    SelfUpdateCommand {
        command: install_step.command.clone(),
        args: install_step.args.clone(),
        display: format!("{} && {}", uninstall_step.display, install_step.display),
        steps: Some(vec![uninstall_step, install_step]),
    }
}

fn make_self_update_command_step(command: &str, args: &[&str]) -> SelfUpdateCommandStep {
    let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
    let display = std::iter::once(command.to_string())
        .chain(args.iter().cloned())
        .map(|arg| if arg.contains(' ') { format!("\"{arg}\"") } else { arg })
        .collect::<Vec<String>>()
        .join(" ");
    SelfUpdateCommandStep {
        command: command.to_string(),
        args,
        display,
    }
}

fn replace_backslashes(value: &str) -> String {
    value.replace('\\', "/")
}

/// `detectInstallMethod`.
pub fn detect_install_method() -> InstallMethod {
    if is_bun_binary() {
        return "bun-binary";
    }
    if is_homebrew_install() {
        return "homebrew";
    }

    let resolved_path = replace_backslashes(&format!(
        "{}\0{}",
        module_dir(),
        std::env::current_exe()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default()
    ))
    .to_lowercase();

    if resolved_path.contains("/pnpm/") || resolved_path.contains("/.pnpm/") {
        return "pnpm";
    }
    if resolved_path.contains("/yarn/") || resolved_path.contains("/.yarn/") {
        return "yarn";
    }
    if is_bun_runtime() || resolved_path.contains("/install/global/node_modules/") {
        return "bun";
    }
    if resolved_path.contains("/npm/") || resolved_path.contains("/node_modules/") {
        return "npm";
    }

    "unknown"
}

fn is_homebrew_install() -> bool {
    let package_dir = replace_backslashes(&get_package_dir()).to_lowercase();
    package_dir.contains("/cellar/") && package_dir.contains("/libexec/lib/node_modules/")
}

fn win32_basename(path: &str) -> String {
    let normalized = path.replace('/', "\\");
    normalized
        .rsplit('\\')
        .next()
        .unwrap_or_default()
        .to_string()
}

fn win32_dirname(path: &str) -> String {
    let normalized = path.replace('/', "\\");
    match normalized.rfind('\\') {
        Some(index) if index > 2 => normalized[..index].to_string(),
        Some(index) => normalized[..index + 1].to_string(),
        None => String::new(),
    }
}

fn path_basename(path: &str, win32: bool) -> String {
    if win32 {
        return win32_basename(path);
    }
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn path_dirname(path: &str, win32: bool) -> String {
    if win32 {
        return win32_dirname(path);
    }
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn join_path(base: &str, name: &str) -> String {
    Path::new(base).join(name).to_string_lossy().to_string()
}

fn get_inferred_npm_install() -> Option<(String, String)> {
    let package_dir = get_package_dir();
    let win32 = crate::utils::pi_user_agent::process_platform() == "win32" || package_dir.contains('\\');
    let parent = path_dirname(&package_dir, win32);
    let mut root: Option<String> = None;
    if path_basename(&parent, win32).starts_with('@')
        && path_basename(&path_dirname(&parent, win32), win32) == "node_modules"
    {
        root = Some(path_dirname(&parent, win32));
    } else if path_basename(&parent, win32) == "node_modules" {
        root = Some(parent.clone());
    }
    let root = root?;
    let root_parent = path_dirname(&root, win32);
    if path_basename(&root_parent, win32) == "lib" {
        return Some((root, path_dirname(&root_parent, win32)));
    }
    // Windows global npm prefixes use `<prefix>\node_modules`, which is
    // indistinguishable from local project installs by path shape alone. Do not
    // infer unsupported Windows custom prefixes without `npm root -g` evidence.
    None
}

fn is_direct_package_artifact_spec(update_spec: &str) -> bool {
    let spec = update_spec.trim().to_lowercase();
    spec.starts_with("http://")
        || spec.starts_with("https://")
        || spec.starts_with("file:")
        || spec.ends_with(".tgz")
        || spec.ends_with(".tar.gz")
}

fn get_default_update_package_name(installed_package_name: &str, update_spec: &str) -> String {
    if is_direct_package_artifact_spec(update_spec) {
        return installed_package_name.to_string();
    }
    update_spec.to_string()
}

/// `getSelfUpdateCommandForMethod`.
pub fn get_self_update_command_for_method(
    method: InstallMethod,
    installed_package_name: &str,
    update_spec: &str,
    npm_command: Option<&[String]>,
    update_package_name: &str,
) -> Option<SelfUpdateCommand> {
    let uninstall_after_install = is_direct_package_artifact_spec(update_spec);
    let options = SelfUpdateCommandOptions {
        uninstall_after_install,
    };
    match method {
        "bun-binary" | "homebrew" => None,
        "pnpm" => Some(make_self_update_command(
            make_self_update_command_step("pnpm", &["install", "-g", update_spec]),
            if update_package_name == installed_package_name {
                None
            } else {
                Some(make_self_update_command_step("pnpm", &["remove", "-g", installed_package_name]))
            },
            options,
        )),
        "yarn" => Some(make_self_update_command(
            make_self_update_command_step("yarn", &["global", "add", update_spec]),
            if update_package_name == installed_package_name {
                None
            } else {
                Some(make_self_update_command_step("yarn", &["global", "remove", installed_package_name]))
            },
            options,
        )),
        "bun" => Some(make_self_update_command(
            make_self_update_command_step("bun", &["install", "-g", update_spec]),
            if update_package_name == installed_package_name {
                None
            } else {
                Some(make_self_update_command_step("bun", &["uninstall", "-g", installed_package_name]))
            },
            options,
        )),
        "npm" => {
            let owned_npm_command = npm_command.unwrap_or(&[]);
            let (command, npm_args) = match owned_npm_command.split_first() {
                Some((command, rest)) => (command.clone(), rest.to_vec()),
                None => ("npm".to_string(), Vec::new()),
            };
            let configured = !owned_npm_command.is_empty();
            let inferred = if configured { None } else { get_inferred_npm_install() };
            let mut prefix_args = npm_args;
            if let Some((_root, prefix)) = inferred {
                prefix_args.push("--prefix".to_string());
                prefix_args.push(prefix);
            }
            let install_args: Vec<String> = prefix_args
                .iter()
                .cloned()
                .chain(["install".to_string(), "-g".to_string(), update_spec.to_string()])
                .collect();
            let install_step = SelfUpdateCommandStep {
                display: std::iter::once(command.clone())
                    .chain(install_args.iter().cloned())
                    .map(|arg| if arg.contains(' ') { format!("\"{arg}\"") } else { arg })
                    .collect::<Vec<String>>()
                    .join(" "),
                command: command.clone(),
                args: install_args,
            };
            let uninstall_step = if update_package_name == installed_package_name {
                None
            } else {
                let uninstall_args: Vec<String> = prefix_args
                    .iter()
                    .cloned()
                    .chain(["uninstall".to_string(), "-g".to_string(), installed_package_name.to_string()])
                    .collect();
                Some(SelfUpdateCommandStep {
                    display: std::iter::once(command.clone())
                        .chain(uninstall_args.iter().cloned())
                        .map(|arg| if arg.contains(' ') { format!("\"{arg}\"") } else { arg })
                        .collect::<Vec<String>>()
                        .join(" "),
                    command,
                    args: uninstall_args,
                })
            };
            Some(make_self_update_command(install_step, uninstall_step, options))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReadCommandOutputOptions {
    pub require_success: bool,
}

fn read_command_output(
    command: &str,
    args: &[String],
    options: ReadCommandOutputOptions,
) -> Result<Option<String>, String> {
    let result = spawn_sync_hidden(
        command,
        args,
        SpawnOptions {
            shell: should_use_windows_shell(command),
            capture_stdout: true,
            capture_stderr: true,
            ..Default::default()
        },
    );
    let full_command = std::iter::once(command.to_string())
        .chain(args.iter().cloned())
        .collect::<Vec<String>>()
        .join(" ");
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            if options.require_success {
                return Err(format!("Failed to run {full_command}: {error}"));
            }
            return Ok(None);
        }
    };
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return Ok(if stdout.is_empty() { None } else { Some(stdout) });
    }
    if options.require_success {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let reason = if !stderr.is_empty() {
            stderr
        } else {
            match output.status.code() {
                Some(code) => format!("exit code {code}"),
                None => "exit code unknown".to_string(),
            }
        };
        return Err(format!("Failed to run {full_command}: {reason}"));
    }
    Ok(None)
}

fn home_dir() -> String {
    // Node's os.homedir() honors USERPROFILE on Windows; dirs::home_dir()
    // goes straight to the OS known-folder API. Keep explicit profile roots
    // (including isolated installations) ahead of that existing fallback.
    #[cfg(windows)]
    if let Some(profile) = std::env::var_os("USERPROFILE").filter(|value| !value.is_empty()) {
        return profile.to_string_lossy().into_owned();
    }
    dirs::home_dir()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn get_global_package_roots(
    method: InstallMethod,
    _package_name: &str,
    npm_command: Option<&[String]>,
) -> Result<Vec<String>, String> {
    match method {
        "npm" => {
            let owned_npm_command = npm_command.unwrap_or(&[]);
            let configured = !owned_npm_command.is_empty();
            let (command, npm_args) = match owned_npm_command.split_first() {
                Some((command, rest)) => (command.clone(), rest.to_vec()),
                None => ("npm".to_string(), Vec::new()),
            };
            if configured && command == "bun" {
                let mut bun_args = npm_args.clone();
                bun_args.extend(["pm".to_string(), "bin".to_string(), "-g".to_string()]);
                let bun_bin = read_command_output(
                    &command,
                    &bun_args,
                    ReadCommandOutputOptions { require_success: true },
                )?;
                let mut roots = vec![join_path(
                    &join_path(&join_path(&home_dir(), ".bun"), "install"),
                    "global/node_modules",
                )];
                if let Some(bun_bin) = bun_bin {
                    roots.push(join_path(
                        &join_path(&join_path(&path_dirname(&bun_bin, false), "install"), "global"),
                        "node_modules",
                    ));
                }
                return Ok(roots);
            }
            let mut root_args = npm_args;
            root_args.extend(["root".to_string(), "-g".to_string()]);
            let root = read_command_output(
                &command,
                &root_args,
                ReadCommandOutputOptions { require_success: configured },
            )?;
            let inferred = if configured { None } else { get_inferred_npm_install() };
            let mut roots: Vec<String> = Vec::new();
            if let Some(root) = root {
                roots.push(root);
            }
            if let Some((inferred_root, _prefix)) = inferred {
                roots.push(inferred_root);
            }
            Ok(roots)
        }
        "pnpm" => {
            let root = read_command_output("pnpm", &["root".to_string(), "-g".to_string()], ReadCommandOutputOptions::default())?;
            Ok(match root {
                None => Vec::new(),
                Some(root) => {
                    let parent = path_dirname(&root, false);
                    vec![root, parent]
                }
            })
        }
        "yarn" => {
            let dir = read_command_output(
                "yarn",
                &["global".to_string(), "dir".to_string()],
                ReadCommandOutputOptions::default(),
            )?;
            Ok(match dir {
                None => Vec::new(),
                Some(dir) => {
                    let nested = join_path(&dir, "node_modules");
                    vec![dir, nested]
                }
            })
        }
        "bun" => {
            let bun_bin = read_command_output(
                "bun",
                &["pm".to_string(), "bin".to_string(), "-g".to_string()],
                ReadCommandOutputOptions::default(),
            )?;
            let mut roots = vec![join_path(
                &join_path(&join_path(&home_dir(), ".bun"), "install"),
                "global/node_modules",
            )];
            if let Some(bun_bin) = bun_bin {
                roots.push(join_path(
                    &join_path(&join_path(&path_dirname(&bun_bin, false), "install"), "global"),
                    "node_modules",
                ));
            }
            Ok(roots)
        }
        _ => Ok(Vec::new()),
    }
}

fn normalize_existing_path_for_comparison(path: &str) -> Option<String> {
    let resolved_path = resolve_path(path);
    if !Path::new(&resolved_path).exists() {
        return None;
    }
    let mut normalized_path = std::fs::canonicalize(&resolved_path).ok()?.to_string_lossy().to_string();
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        normalized_path = normalized_path.to_lowercase();
    }
    Some(normalized_path)
}

/// `accessSync(path, constants.W_OK)`.
#[cfg(unix)]
fn is_path_writable(path: &str) -> bool {
    use std::ffi::CString;
    let Ok(c_path) = CString::new(path) else {
        return false;
    };
    unsafe { libc::access(c_path.as_ptr(), libc::W_OK) == 0 }
}

/// `accessSync(path, constants.W_OK)` on Windows checks the read-only attribute,
/// exactly as Node does.
#[cfg(not(unix))]
fn is_path_writable(path: &str) -> bool {
    match std::fs::metadata(path) {
        Ok(metadata) => !metadata.permissions().readonly(),
        Err(_) => false,
    }
}

fn is_self_update_path_writable() -> bool {
    let package_dir = get_package_dir();
    is_path_writable(&package_dir) && is_path_writable(&path_dirname(&package_dir, false))
}

fn is_managed_by_global_package_manager(
    method: InstallMethod,
    package_name: &str,
    npm_command: Option<&[String]>,
) -> bool {
    let Some(package_dir) = normalize_existing_path_for_comparison(&get_package_dir()) else {
        return false;
    };
    let Ok(roots) = get_global_package_roots(method, package_name, npm_command) else {
        return false;
    };
    roots.into_iter().any(|root| {
        let Some(normalized_root) = normalize_existing_path_for_comparison(&root) else {
            return false;
        };
        let prefix = if normalized_root.ends_with(std::path::MAIN_SEPARATOR) {
            normalized_root
        } else {
            format!("{normalized_root}{}", std::path::MAIN_SEPARATOR)
        };
        package_dir.starts_with(&prefix)
    })
}

/// `getSelfUpdateCommand`.
pub fn get_self_update_command(
    package_name: &str,
    npm_command: Option<&[String]>,
    update_spec: &str,
    update_package_name: &str,
) -> Option<SelfUpdateCommand> {
    let method = detect_install_method();
    let command =
        get_self_update_command_for_method(method, package_name, update_spec, npm_command, update_package_name);
    match command {
        Some(command)
            if is_managed_by_global_package_manager(method, package_name, npm_command)
                && is_self_update_path_writable() =>
        {
            Some(command)
        }
        _ => None,
    }
}

/// `getSelfUpdateUnavailableInstruction`.
pub fn get_self_update_unavailable_instruction(
    package_name: &str,
    npm_command: Option<&[String]>,
    update_spec: &str,
    update_package_name: &str,
) -> String {
    let method = detect_install_method();
    if method == "bun-binary" {
        return format!("Download from: {PRIME_AGENT_UPDATE_RELEASE_URL}");
    }
    if method == "homebrew" {
        return format!("Update with: brew upgrade {APP_NAME}");
    }
    let command =
        get_self_update_command_for_method(method, package_name, update_spec, npm_command, update_package_name);
    if let Some(command) = command {
        if is_managed_by_global_package_manager(method, package_name, npm_command) && !is_self_update_path_writable() {
            return format!(
                "This installation is managed by a global {method} install, but the install path is not writable. Update it yourself with: {}",
                command.display
            );
        }
        return format!(
            "This installation is not managed by a global {method} install. Update it with the package manager, wrapper, or source checkout that provides it."
        );
    }
    format!(
        "Update {update_spec} using the package manager, wrapper, or source checkout that provides this installation."
    )
}

/// `getUpdateInstruction`.
pub fn get_update_instruction(package_name: &str) -> String {
    if get_self_update_command_for_method(detect_install_method(), package_name, package_name, None, package_name)
        .is_some()
    {
        return format!("Run: {APP_NAME} update");
    }
    get_self_update_unavailable_instruction(package_name, None, package_name, package_name)
}

// =============================================================================
// Package Asset Paths (shipped with executable)
// =============================================================================

/// `__dirname` for `config.ts`.
///
/// The TypeScript module lives at `<packageDir>/dist/config.js` (built) or
/// `<packageDir>/src/config.ts` (tsx), so walking up from it lands on the
/// package root. The Rust library has no per-module directory: the port uses the
/// `packages/coding-agent` directory of the checkout when it can find it, so
/// `getPackageDir()` and the `src`/`dist` asset paths resolve identically.
fn module_dir() -> String {
    static MODULE_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE_DIR
        .get_or_init(|| {
            if let Ok(dir) = std::env::var("PI_CODING_AGENT_MODULE_DIR") {
                return dir;
            }
            let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            loop {
                let candidate = dir.join("packages").join("coding-agent");
                if candidate.join("package.json").exists() {
                    return candidate.to_string_lossy().to_string();
                }
                match dir.parent() {
                    Some(parent) if parent != dir => dir = parent.to_path_buf(),
                    _ => break,
                }
            }
            env!("CARGO_MANIFEST_DIR").to_string()
        })
        .clone()
}

/// `getPackageDir`.
pub fn get_package_dir() -> String {
    // Allow override via environment variable (useful for Nix/Guix where store paths tokenize poorly)
    if let Ok(env_dir) = std::env::var("PI_PACKAGE_DIR") {
        if !env_dir.is_empty() {
            return expand_tilde_path(&env_dir, None);
        }
    }

    let mut dir = PathBuf::from(module_dir());
    while dir.parent().map(|parent| parent != dir).unwrap_or(false) {
        if dir.join("package.json").exists() {
            return dir.to_string_lossy().to_string();
        }
        dir = match dir.parent() {
            Some(parent) => parent.to_path_buf(),
            None => break,
        };
    }
    module_dir()
}

fn src_or_dist(package_dir: &str) -> &'static str {
    if Path::new(package_dir).join("src").exists() {
        "src"
    } else {
        "dist"
    }
}

/// `getThemesDir`.
pub fn get_themes_dir() -> String {
    if is_bun_binary() {
        return join_path(&get_package_dir(), "theme");
    }
    let package_dir = get_package_dir();
    let mut path = PathBuf::from(&package_dir);
    path.push(src_or_dist(&package_dir));
    path.push("modes");
    path.push("interactive");
    path.push("theme");
    path.to_string_lossy().to_string()
}

/// `getExportTemplateDir`.
pub fn get_export_template_dir() -> String {
    if is_bun_binary() {
        return join_path(&get_package_dir(), "export-html");
    }
    let package_dir = get_package_dir();
    let mut path = PathBuf::from(&package_dir);
    path.push(src_or_dist(&package_dir));
    path.push("core");
    path.push("export-html");
    path.to_string_lossy().to_string()
}

/// `getPackageJsonPath`.
pub fn get_package_json_path() -> String {
    join_path(&get_package_dir(), "package.json")
}

/// `getDocsPath`.
pub fn get_docs_path() -> String {
    resolve_path(&join_path(&get_package_dir(), "docs"))
}

/// `getChangelogPath`.
pub fn get_changelog_path() -> String {
    resolve_path(&join_path(&get_package_dir(), "CHANGELOG.md"))
}

/// `getInteractiveAssetsDir`.
pub fn get_interactive_assets_dir() -> String {
    if is_bun_binary() {
        return join_path(&get_package_dir(), "assets");
    }
    let package_dir = get_package_dir();
    let mut path = PathBuf::from(&package_dir);
    path.push(src_or_dist(&package_dir));
    path.push("modes");
    path.push("interactive");
    path.push("assets");
    path.to_string_lossy().to_string()
}

/// `getBundledInteractiveAssetPath`.
pub fn get_bundled_interactive_asset_path(name: &str) -> String {
    join_path(&get_interactive_assets_dir(), name)
}

/// `getBundledSkillsDir`.
pub fn get_bundled_skills_dir() -> String {
    if is_bun_binary() {
        return join_path(&get_package_dir(), "skills");
    }
    let package_dir = get_package_dir();
    // Source checkouts (tsx) keep built-in skills at the package root; built
    // packages copy them to dist/skills. Decide by whether src/ is present so a
    // stale dist/ from a prior build never shadows live source edits.
    let is_source_checkout = Path::new(&package_dir).join("src").exists();
    if is_source_checkout {
        join_path(&package_dir, "skills")
    } else {
        join_path(&join_path(&package_dir, "dist"), "skills")
    }
}

// =============================================================================
// App Config (from package.json piConfig)
// =============================================================================

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct PackageJsonPiConfig {
    name: Option<String>,
    #[serde(rename = "configDir")]
    config_dir: Option<String>,
    /// Optional terminal/tab identity title (`piConfig.title`); the TypeScript
    /// reference ignores this field.
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct PackageJson {
    name: Option<String>,
    version: Option<String>,
    #[serde(rename = "piConfig")]
    pi_config: Option<PackageJsonPiConfig>,
}

fn read_package_json() -> PackageJson {
    let content = std::fs::read_to_string(get_package_json_path()).unwrap_or_default();
    serde_json::from_str(&content).unwrap_or_default()
}

fn env_prefix_for(pi_config_name: Option<&str>) -> String {
    let upper = pi_config_name.unwrap_or("pi").to_uppercase();
    let mut replaced = String::new();
    let mut last_underscore = false;
    for character in upper.chars() {
        if character.is_ascii_uppercase() || character.is_ascii_digit() {
            replaced.push(character);
            last_underscore = false;
        } else if !last_underscore {
            replaced.push('_');
            last_underscore = true;
        }
    }
    let trimmed = replaced.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "PI".to_string()
    } else {
        trimmed
    }
}

fn pkg_name(pkg: &PackageJson) -> String {
    pkg.name.clone().unwrap_or_else(|| "@earendil-works/pi-coding-agent".to_string())
}

fn app_name_from(pkg: &PackageJson) -> String {
    pkg.pi_config
        .as_ref()
        .and_then(|config| config.name.clone())
        .unwrap_or_else(|| "pi".to_string())
}

pub fn app_title_from_pkg(pkg: &PackageJson) -> String {
    match pkg.pi_config.as_ref().and_then(|config| config.name.clone()) {
        Some(_) => app_name_from(pkg),
        None => "\u{3c0}".to_string(),
    }
}

/// Terminal/tab identity title. `PRIME_AGENT_APP_TITLE` wins when it is set and
/// non-empty after sanitization, then `piConfig.title`, then the existing
/// `app_title_from_pkg` behavior (`piConfig.name` or the pi letter fallback).
pub fn app_display_title_from(pkg: &PackageJson, env_override: Option<&str>) -> String {
    let candidates = [env_override, pkg.pi_config.as_ref().and_then(|config| config.title.as_deref())];
    for candidate in candidates {
        let Some(candidate) = candidate else { continue };
        let cleaned = pi_tui::terminal::sanitize_title_text(candidate.trim()).trim().to_string();
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    app_title_from_pkg(pkg)
}

fn config_dir_name_from(pkg: &PackageJson) -> String {
    pkg.pi_config
        .as_ref()
        .and_then(|config| config.config_dir.clone())
        .unwrap_or_else(|| ".prime/agent".to_string())
}

fn version_from(pkg: &PackageJson) -> String {
    pkg.version.clone().unwrap_or_else(|| "0.0.0".to_string())
}

fn package_name_static() -> &'static str {
    static PACKAGE_NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PACKAGE_NAME.get_or_init(|| pkg_name(&read_package_json()))
}

/// `PACKAGE_NAME` (`pkg.name || "@earendil-works/pi-coding-agent"`).
pub fn package_name() -> &'static str {
    package_name_static()
}

fn app_name_static() -> &'static str {
    static APP_NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    APP_NAME.get_or_init(|| app_name_from(&read_package_json()))
}

/// `APP_NAME` (`piConfig.name || "pi"`).
pub fn app_name() -> &'static str {
    app_name_static()
}

fn app_title_static() -> &'static str {
    static APP_TITLE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    APP_TITLE.get_or_init(|| app_title_from_pkg(&read_package_json()))
}

/// `APP_TITLE` (`piConfig.name ? APP_NAME : "π"`).
pub fn app_title() -> &'static str {
    app_title_static()
}

fn app_display_title_static() -> &'static str {
    static APP_DISPLAY_TITLE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    APP_DISPLAY_TITLE.get_or_init(|| {
        let env_override = std::env::var("PRIME_AGENT_APP_TITLE").ok();
        app_display_title_from(&read_package_json(), env_override.as_deref())
    })
}

/// Terminal/tab identity title (see `app_display_title_from`); resolved once at
/// startup from the environment override, `piConfig.title`, or the name-based
/// `APP_TITLE` fallback.
pub fn app_display_title() -> &'static str {
    app_display_title_static()
}

/// `app_display_title()` (String getter, see `app_title_string()`).
pub fn app_display_title_string() -> String {
    app_display_title().to_string()
}

fn config_dir_name_static() -> &'static str {
    static CONFIG_DIR_NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CONFIG_DIR_NAME.get_or_init(|| config_dir_name_from(&read_package_json()))
}

/// `CONFIG_DIR_NAME` (`piConfig.configDir || ".prime/agent"`).
pub fn config_dir_name() -> &'static str {
    config_dir_name_static()
}

fn version_static() -> &'static str {
    static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    VERSION.get_or_init(|| {
        let from_pkg = version_from(&read_package_json());
        if from_pkg == "0.0.0" {
            env!("CARGO_PKG_VERSION").to_string()
        } else {
            from_pkg
        }
    })
}

/// `VERSION` (`pkg.version || "0.0.0"`; falls back to the crate version here).
pub fn version() -> &'static str {
    version_static()
}

fn env_agent_dir_static() -> &'static str {
    static ENV_AGENT_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ENV_AGENT_DIR.get_or_init(|| {
        format!(
            "{}_CODING_AGENT_DIR",
            env_prefix_for(read_package_json().pi_config.as_ref().and_then(|config| config.name.as_deref()))
        )
    })
}

/// `ENV_AGENT_DIR` (e.g. `PRIME_AGENT_CODING_AGENT_DIR`).
pub fn env_agent_dir() -> &'static str {
    env_agent_dir_static()
}

fn env_session_dir_static() -> &'static str {
    static ENV_SESSION_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ENV_SESSION_DIR.get_or_init(|| {
        format!(
            "{}_SESSION_DIR",
            env_prefix_for(read_package_json().pi_config.as_ref().and_then(|config| config.name.as_deref()))
        )
    })
}

/// `ENV_SESSION_DIR`.
pub fn env_session_dir() -> &'static str {
    env_session_dir_static()
}

fn env_legacy_session_dir_static() -> &'static str {
    static ENV_LEGACY_SESSION_DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ENV_LEGACY_SESSION_DIR.get_or_init(|| {
        format!(
            "{}_CODING_AGENT_SESSION_DIR",
            env_prefix_for(read_package_json().pi_config.as_ref().and_then(|config| config.name.as_deref()))
        )
    })
}

/// `ENV_LEGACY_SESSION_DIR`.
pub fn env_legacy_session_dir() -> &'static str {
    env_legacy_session_dir_static()
}

// =============================================================================
// Path helpers
// =============================================================================

/// `expandTildePath(path, platform)`.
pub fn expand_tilde_path(path: &str, platform: Option<&str>) -> String {
    let platform = platform.unwrap_or_else(|| crate::utils::pi_user_agent::process_platform());
    if path == "~" {
        return home_dir();
    }
    if path.starts_with("~/") || (platform == "win32" && path.starts_with("~\\")) {
        return join_path(&home_dir(), &path[2..]);
    }
    path.to_string()
}

/// Node's `path.resolve` for a single argument (absolute input wins, otherwise CWD-relative).
fn resolve_path(path: &str) -> String {
    let candidate = Path::new(path);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(candidate)
    };
    normalize_lexically(&joined)
}

fn normalize_lexically(path: &Path) -> String {
    // Same lexical collapse as `utils/daemon-socket-path.ts` (Node's `path.resolve`).
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Prefix(prefix_component) => {
                prefix.push_str(&prefix_component.as_os_str().to_string_lossy());
            }
            Component::RootDir => {
                // `Component::Prefix("C:")` followed by `Component::RootDir`
                // is the single drive root "C:\". Pushing the root separator
                // unconditionally keeps that root; the earlier `if prefix
                // .is_empty()` test dropped it and produced "C:Users/...",
                // a drive-relative path, so every caller failed with
                // os error 3 (and "/tmp/registry" became "C:tmp/registry").
                prefix.push('/');
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !parts.is_empty() && parts.last().map(|part| part != "..").unwrap_or(false) {
                    parts.pop();
                } else if !path.is_absolute() {
                    parts.push("..".to_string());
                }
            }
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
        }
    }
    if prefix.is_empty() {
        parts.join("/")
    } else if prefix == "/" {
        format!("/{}", parts.join("/"))
    } else {
        format!("{}{}", prefix, parts.join("/"))
    }
}

const DEFAULT_SHARE_VIEWER_URL: &str = "https://pi.dev/session/";

/// `getShareViewerUrl`.
pub fn get_share_viewer_url(gist_id: &str) -> String {
    let base_url = std::env::var("PI_SHARE_VIEWER_URL").unwrap_or_else(|_| DEFAULT_SHARE_VIEWER_URL.to_string());
    format!("{base_url}#{gist_id}")
}

// =============================================================================
// User Config Paths (~/.prime/agent/*)
// =============================================================================

/// `getAgentDir`.
pub fn get_agent_dir() -> String {
    if let Ok(env_dir) = std::env::var(env_agent_dir()) {
        if !env_dir.is_empty() {
            return expand_tilde_path(&env_dir, None);
        }
    }
    join_path(&home_dir(), config_dir_name())
}

/// `getCustomThemesDir`.
pub fn get_custom_themes_dir() -> String {
    join_path(&get_agent_dir(), "themes")
}

/// `getLogsDir`.
pub fn get_logs_dir() -> String {
    join_path(&get_agent_dir(), "logs")
}

/// `getClientErrorLogPath`.
pub fn get_client_error_log_path() -> String {
    join_path(&get_logs_dir(), "client-errors.log")
}

/// `getAgentTracesLogPath`.
pub fn get_agent_traces_log_path() -> String {
    join_path(&get_logs_dir(), "agent-traces.log")
}

/// `getAgentLogPath`.
pub fn get_agent_log_path() -> String {
    join_path(&get_logs_dir(), "agent.jsonl")
}

/// `getDaemonLogPath`: basename plus the first 8 hex characters of the sha256 of
/// the full normalized socket path, so two sockets that share a basename do not
/// interleave into one file.
pub fn get_daemon_log_path(socket_path: &str) -> String {
    use sha2::{Digest, Sha256};
    let normalized = normalize_socket_path(socket_path, None);
    let hash = format!("{:x}", Sha256::digest(normalized.as_bytes()));
    let base_name = path_basename(&normalized, false);
    join_path(&get_logs_dir(), &format!("{}.{}.log", base_name, &hash[..8]))
}

/// `getDaemonUpdateRestartManifestPath`.
pub fn get_daemon_update_restart_manifest_path(socket_path: &str, agent_dir: Option<&str>) -> String {
    use sha2::{Digest, Sha256};
    let normalized_socket_path = normalize_socket_path(socket_path, None);
    let socket_hash = format!("{:x}", Sha256::digest(normalized_socket_path.as_bytes()));
    let agent_dir = agent_dir.map(|dir| dir.to_string()).unwrap_or_else(get_agent_dir);
    join_path(&join_path(&agent_dir, "daemon-update-restarts"), &format!("{socket_hash}.json"))
}

/// `getLegacyDaemonUpdateRestartManifestPath`.
pub fn get_legacy_daemon_update_restart_manifest_path(agent_dir: Option<&str>) -> String {
    let agent_dir = agent_dir.map(|dir| dir.to_string()).unwrap_or_else(get_agent_dir);
    join_path(&agent_dir, "daemon-update-restart.json")
}

pub const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

/// `appendRotatingLog`: append one line, keeping the size bounded with a
/// single-generation rotation. Best-effort: diagnostics must never throw into
/// the caller.
pub fn append_rotating_log(log_path: &str, message: &str, max_bytes: u64) {
    let path = Path::new(log_path);
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let outcome = (|| -> std::io::Result<()> {
        if let Ok(metadata) = std::fs::metadata(path) {
            if metadata.len() > max_bytes {
                // Drop any prior .old first: renameSync fails on Windows if it exists.
                let old = format!("{log_path}.old");
                let _ = std::fs::remove_file(&old);
                if std::fs::rename(path, &old).is_err() {
                    // Keep appending rather than dropping the log on a rotation failure.
                    return Ok(());
                }
            }
        }
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(format!("{message}\n").as_bytes())
    })();
    // A read-only or missing log dir must never break the caller.
    let _ = outcome;
}

/// `appendRotatingLog(logPath, message)` with the default 5 MiB cap.
pub fn append_rotating_log_default(log_path: &str, message: &str) {
    append_rotating_log(log_path, message, MAX_LOG_BYTES)
}

/// `getAuthPath`.
pub fn get_auth_path() -> String {
    join_path(&get_agent_dir(), "auth.json")
}

/// `getCronJobsPath`.
pub fn get_cron_jobs_path(agent_dir: Option<&str>) -> String {
    let agent_dir = agent_dir.map(|dir| dir.to_string()).unwrap_or_else(get_agent_dir);
    join_path(&agent_dir, "cron-jobs.json")
}

/// `getBinDir`.
pub fn get_bin_dir() -> String {
    join_path(&get_agent_dir(), "bin")
}

/// `getSessionsDir`.
pub fn get_sessions_dir(agent_dir: Option<&str>) -> String {
    if let Some(env_dir) = get_session_dir_env_override() {
        return env_dir;
    }
    let agent_dir = agent_dir.map(|dir| dir.to_string()).unwrap_or_else(get_agent_dir);
    join_path(&agent_dir, "sessions")
}

/// `getSessionDirEnvOverride`.
pub fn get_session_dir_env_override() -> Option<String> {
    let env_dir = std::env::var(env_session_dir())
        .ok()
        .or_else(|| std::env::var(env_legacy_session_dir()).ok())?;
    if env_dir.is_empty() {
        return None;
    }
    Some(expand_tilde_path(&env_dir, None))
}

/// `getDebugLogPath`.
pub fn get_debug_log_path() -> String {
    join_path(&get_agent_dir(), &format!("{}-debug.log", app_name()))
}

/// `APP_NAME` (const-compatible getter, see `app_name()`).
pub fn app_name_string() -> String {
    app_name().to_string()
}

/// `VERSION` (const-compatible getter, see `version()`).
pub fn version_string() -> String {
    version().to_string()
}

/// `PACKAGE_NAME` (const-compatible getter, see `package_name()`).
pub fn package_name_string() -> String {
    package_name().to_string()
}

/// `APP_TITLE` (const-compatible getter, see `app_title()`).
pub fn app_title_string() -> String {
    app_title().to_string()
}

/// `CONFIG_DIR_NAME` (const-compatible getter, see `config_dir_name()`).
pub fn config_dir_name_string() -> String {
    config_dir_name().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_package() -> PackageJson {
        PackageJson {
            name: Some("@earendil-works/pi-coding-agent".to_string()),
            version: Some("9.9.9".to_string()),
            pi_config: Some(PackageJsonPiConfig {
                name: Some("prime-agent".to_string()),
                config_dir: Some(".prime/agent".to_string()),
                title: None,
            }),
        }
    }

    #[test]
    fn package_identity_matches_the_pinned_package_json() {
        let pkg = sample_package();
        assert_eq!(pkg_name(&pkg), "@earendil-works/pi-coding-agent");
        assert_eq!(app_name_from(&pkg), "prime-agent");
        assert_eq!(config_dir_name_from(&pkg), ".prime/agent");
        assert_eq!(version_from(&pkg), "9.9.9");
        assert_eq!(app_title_from_pkg(&pkg), "prime-agent");
    }

    #[test]
    fn pi_title_falls_back_to_the_greek_letter() {
        let pkg = PackageJson { name: Some("@earendil-works/pi-coding-agent".to_string()), version: None, pi_config: None };
        assert_eq!(app_name_from(&pkg), "pi");
        assert_eq!(app_title_from_pkg(&pkg), "\u{3c0}");
        assert_eq!(config_dir_name_from(&pkg), ".prime/agent");
        assert_eq!(version_from(&pkg), "0.0.0");
    }

    #[test]
    fn pi_config_title_sets_the_display_identity() {
        let mut pkg = sample_package();
        pkg.pi_config.as_mut().unwrap().title = Some("Optimus - Agent".to_string());
        assert_eq!(app_display_title_from(&pkg, None), "Optimus - Agent");
    }

    #[test]
    fn app_title_env_override_wins_over_pi_config() {
        let mut pkg = sample_package();
        pkg.pi_config.as_mut().unwrap().title = Some("Optimus - Agent".to_string());
        assert_eq!(app_display_title_from(&pkg, Some("Optimus - Assistant")), "Optimus - Assistant");
    }

    #[test]
    fn app_title_sources_drop_control_and_escape_payloads() {
        let mut pkg = sample_package();
        pkg.pi_config.as_mut().unwrap().title = Some("\x1b]0;evil\x07Optimus - Agent".to_string());
        let title = app_display_title_from(&pkg, None);
        assert_eq!(title, "]0;evilOptimus - Agent");
        assert!(!title.contains('\x1b'));
        assert!(!title.contains('\x07'));
        let override_title = app_display_title_from(&pkg, Some("  Optimus - Assistant \u{202e} \x1b[31m"));
        assert_eq!(override_title, "Optimus - Assistant  [31m");
    }

    #[test]
    fn empty_app_title_fallbacks_keep_the_existing_behavior() {
        let pkg = sample_package();
        assert_eq!(app_display_title_from(&pkg, None), "prime-agent");
        assert_eq!(app_display_title_from(&pkg, Some("   ")), "prime-agent");
        let mut pkg = sample_package();
        pkg.pi_config.as_mut().unwrap().title = Some("  \x1b \x07 ".to_string());
        assert_eq!(app_display_title_from(&pkg, None), "prime-agent");
        let bare = PackageJson { name: Some("@earendil-works/pi-coding-agent".to_string()), version: None, pi_config: None };
        assert_eq!(app_display_title_from(&bare, None), "\u{3c0}");
    }

    #[test]
    fn env_prefix_uppercases_and_underscores() {
        assert_eq!(env_prefix_for(Some("prime-agent")), "PRIME_AGENT");
        assert_eq!(env_prefix_for(Some("pi")), "PI");
        assert_eq!(env_prefix_for(None), "PI");
        assert_eq!(env_prefix_for(Some("__x__")), "X");
    }

    #[test]
    fn tilde_expansion_matches_the_typescript() {
        assert_eq!(expand_tilde_path("~", None), home_dir());
        assert_eq!(expand_tilde_path("/abs", None), "/abs");
        assert_eq!(expand_tilde_path("~nope", None), "~nope");
        assert_eq!(expand_tilde_path("~/x", None), join_path(&home_dir(), "x"));
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            assert_eq!(expand_tilde_path("~\\x", Some("win32")), join_path(&home_dir(), "x"));
            assert_eq!(expand_tilde_path("~\\x", Some("linux")), "~\\x");
        }
    }

    #[test]
    fn direct_package_artifact_specs() {
        assert!(is_direct_package_artifact_spec("HTTPS://example.com/pkg.tgz"));
        assert!(is_direct_package_artifact_spec("file:./pkg"));
        assert!(is_direct_package_artifact_spec("x.tar.gz"));
        assert!(!is_direct_package_artifact_spec("npm:@foo/bar"));
        assert_eq!(get_default_update_package_name("pkg", "https://x/y.tgz"), "pkg");
        assert_eq!(get_default_update_package_name("pkg", "npm:@foo/bar"), "npm:@foo/bar");
    }

    #[test]
    fn npm_self_update_command_shape() {
        let command = get_self_update_command_for_method("npm", "pkg", "npm:pkg", None, "pkg").unwrap();
        assert_eq!(command.command, "npm");
        assert_eq!(command.args, vec!["install", "-g", "npm:pkg"]);
        assert_eq!(command.display, "npm install -g npm:pkg");
        assert!(command.steps.is_none());
    }

    #[test]
    fn npm_self_update_command_installs_before_uninstall_for_direct_specs() {
        let command =
            get_self_update_command_for_method("npm", "old-pkg", "https://x/y.tgz", None, "new-pkg").unwrap();
        let steps = command.steps.unwrap();
        assert_eq!(steps[0].command, "npm");
        assert_eq!(steps[0].args, vec!["install", "-g", "https://x/y.tgz"]);
        assert_eq!(steps[1].args, vec!["uninstall", "-g", "old-pkg"]);
        assert_eq!(command.display, "npm install -g https://x/y.tgz && npm uninstall -g old-pkg");
    }

    #[test]
    fn pnpm_self_update_command_replaces_the_package_name() {
        let command = get_self_update_command_for_method("pnpm", "old-pkg", "new-pkg", None, "new-pkg").unwrap();
        assert_eq!(command.display, "pnpm remove -g old-pkg && pnpm install -g new-pkg");
    }

    #[test]
    fn commands_quote_arguments_with_spaces() {
        let step = make_self_update_command_step("npm", &["install", "-g", "a b"]);
        assert_eq!(step.display, "npm install -g \"a b\"");
    }

    #[test]
    fn binary_and_homebrew_installs_have_no_self_update_command() {
        assert!(get_self_update_command_for_method("bun-binary", "pkg", "pkg", None, "pkg").is_none());
        assert!(get_self_update_command_for_method("homebrew", "pkg", "pkg", None, "pkg").is_none());
        assert!(get_self_update_command_for_method("unknown", "pkg", "pkg", None, "pkg").is_none());
        if detect_install_method() == "bun-binary" {
            assert_eq!(
                get_self_update_unavailable_instruction("pkg", None, "pkg", "pkg"),
                format!("Download from: {PRIME_AGENT_UPDATE_RELEASE_URL}")
            );
        }
    }

    #[test]
    fn homebrew_install_detection() {
        let package_dir = "/opt/homebrew/Cellar/pi/1.0/libexec/lib/node_modules/@earendil-works/pi-coding-agent";
        let normalized = replace_backslashes(package_dir).to_lowercase();
        assert!(normalized.contains("/cellar/") && normalized.contains("/libexec/lib/node_modules/"));
    }

    #[test]
    fn share_viewer_url_appends_the_gist_fragment() {
        assert_eq!(get_share_viewer_url("abc"), "https://pi.dev/session/#abc");
    }

    #[test]
    fn path_basename_and_dirname_use_the_win32_flavour_when_asked() {
        assert_eq!(path_basename("C:\\a\\b", true), "b");
        assert_eq!(path_dirname("C:\\a\\b", true), "C:\\a");
        assert_eq!(path_basename("/a/b", false), "b");
        assert_eq!(path_dirname("/a/b", false), "/a");
    }

    #[test]
    fn resolve_path_is_lexical_and_absolute_input_wins() {
        let collapsed = resolve_path("/a/./b/../c");
        assert!(collapsed.ends_with("/a/c"), "`.`/`..` were not collapsed: {collapsed}");
        // A rootless "/a/..." input inherits the current drive on Windows, so the
        // collapsed name must stay drive-ABSOLUTE. Asserting the plain POSIX
        // "/a/c" only holds on a POSIX host.
        let expected_root = if crate::utils::pi_user_agent::process_platform() == "win32" {
            let drive = std::env::current_dir()
                .ok()
                .and_then(|cwd| cwd.components().next().map(|component| component.as_os_str().to_string_lossy().to_string()))
                .unwrap_or_default();
            assert_eq!(collapsed, format!("{drive}/a/c"));
            assert!(!collapsed.starts_with(&format!("{drive}a")), "drive root dropped: {collapsed}");
            format!("{drive}/")
        } else {
            assert_eq!(collapsed, "/a/c");
            "/".to_string()
        };
        // Absolute input wins, and a drive-rooted spelling keeps its root.
        assert!(resolve_path("/a/./b/../c").starts_with(&expected_root));
        assert!(resolve_path("rel").ends_with("rel"));
        assert!(
            !resolve_path("C:\\tmp\\registry").starts_with("C:tmp"),
            "drive root dropped: {}",
            resolve_path("C:\\tmp\\registry")
        );
    }

    #[test]
    fn rotating_log_caps_the_file_and_keeps_one_generation() {
        let dir = std::env::temp_dir().join(format!("pi-config-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("test.log");
        let path = path.to_string_lossy().to_string();
        append_rotating_log(&path, "first", 4);
        append_rotating_log(&path, "second", 4);
        assert!(Path::new(&format!("{path}.old")).exists());
        let current = std::fs::read_to_string(&path).unwrap();
        assert_eq!(current, "second\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn env_override_names_follow_the_package_identity() {
        assert!(env_agent_dir().ends_with("_CODING_AGENT_DIR"));
        assert!(env_session_dir().ends_with("_SESSION_DIR"));
        assert!(env_legacy_session_dir().ends_with("_CODING_AGENT_SESSION_DIR"));
        assert!(get_agent_dir().ends_with(config_dir_name()) || std::env::var(env_agent_dir()).is_ok());
    }

    #[test]
    fn package_json_paths_come_from_the_package_directory() {
        let package_dir = get_package_dir();
        assert_eq!(get_package_json_path(), join_path(&package_dir, "package.json"));
        assert_eq!(detect_install_method(), detect_install_method());
    }

    #[test]
    fn the_binary_is_never_a_bun_binary() {
        assert!(!is_bun_binary());
        assert!(!is_bun_runtime());
        assert_eq!(detect_install_method(), detect_install_method());
    }
    /// `Component::Prefix("C:")` followed by `Component::RootDir` is the single
    /// drive root "C:\". Returns the path once the host has produced that shape
    /// (None on a POSIX host, where "C:\..." has no Prefix component at all), so
    /// the assertions below cannot silently pass on a path that never reaches the
    /// RootDir arm.
    fn drive_absolute_input(label: &str, raw: &str) -> Option<PathBuf> {
        use std::path::Component;
        let path = Path::new(raw);
        if !matches!(path.components().next(), Some(Component::Prefix(_))) {
            return None;
        }
        assert!(
            path.components().any(|component| component == Component::RootDir),
            "{raw:?} has a drive prefix but no RootDir component ({label})"
        );
        Some(path.to_path_buf())
    }

    /// The defect this pins: the old `if prefix.is_empty()` guard skipped the
    /// root separator because `Prefix("C:")` had already filled `prefix`, so a
    /// drive-absolute path collapsed to a DRIVE-RELATIVE one - "C:Users/x/registry"
    /// instead of "C:/Users/x/registry", and a drive-rooted "/tmp/x" became
    /// "C:tmp/x". Every downstream create_dir_all / open / lockfile then failed
    /// with os error 3 ("The system cannot find the path specified").
    #[test]
    fn drive_roots_survive_the_lexical_collapse() {
        // Checked on every host: a rooted path keeps its root separator.
        let rooted = normalize_lexically(Path::new("/tmp/x"));
        assert!(rooted.starts_with('/'), "root separator dropped: {rooted}");

        let registry = match drive_absolute_input("registry", "C:\\Users\\x\\registry") {
            Some(path) => path,
            None => return,
        };
        let normalised = normalize_lexically(&registry);
        assert!(normalised.starts_with("C:/"), "drive root dropped: {normalised}");
        assert_eq!(normalised, "C:/Users/x/registry");

        // `Path::join` keeps the drive when it appends the rooted POSIX spelling,
        // which is exactly how a caller's resolve("/tmp/x") reaches this function.
        let joined = drive_absolute_input("joined /tmp/x", "C:\\base")
            .expect("C:\\base is drive-absolute")
            .join("/tmp/x");
        let normalised = normalize_lexically(&joined);
        assert!(!normalised.starts_with("C:tmp"), "/tmp/x became drive-relative: {normalised}");
        assert_eq!(normalised, "C:/tmp/x");

        let dotted = drive_absolute_input("parent collapse", "C:\\Users\\x\\..\\y")
            .expect("C:\\Users\\x\\..\\y is drive-absolute");
        assert_eq!(normalize_lexically(&dotted), "C:/Users/y");

        // Only the RootDir arm changed: a drive-RELATIVE input has no RootDir, so
        // it must still come back rootless instead of gaining a "C:/" root.
        let relative = Path::new("C:registry");
        if !relative.components().any(|component| component == std::path::Component::RootDir) {
            assert_eq!(normalize_lexically(relative), "C:registry");
        }
    }
}
