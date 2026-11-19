//! Port of packages/coding-agent/src/core/package-manager.ts
//!
//! Package management for npm/git/local sources with filtering.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::core::diagnostics::{
    ResourceDiagnostic, RESOURCE_DIAGNOSTIC_WARNING,
};
use crate::core::output_guard::is_stdout_taken_over;
use crate::core::settings_manager::{FilteredPackageSource, PackageSource, SettingsManager, Settings};
use crate::utils::child_process::{
    should_use_windows_shell, signal_process_group_or_process, spawn_hidden, spawn_sync_hidden,
    wait_for_child_process, Signal, SpawnOptions,
};
use crate::utils::git::{parse_git_url, GitSource};
use crate::utils::paths::{canonicalize_path, is_local_path};

/// `CONFIG_DIR_NAME` from config.ts.
/// blocked_on: needs crate::config::CONFIG_DIR_NAME
const CONFIG_DIR_NAME: &str = ".prime/agent";

/// `getBundledSkillsDir()` from config.ts.
/// blocked_on: needs crate::config::get_bundled_skills_dir
fn get_bundled_skills_dir() -> String {
    let package_dir = match std::env::var("PI_PACKAGE_DIR") {
        Ok(dir) if !dir.is_empty() => dir,
        _ => std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(|parent| parent.to_string_lossy().to_string()))
            .unwrap_or_default(),
    };
    let source_checkout = Path::new(&join_path(&package_dir, "src")).exists();
    if source_checkout {
        join_path(&package_dir, "skills")
    } else {
        join_path(&join_path(&package_dir, "dist"), "skills")
    }
}

pub const NETWORK_TIMEOUT_MS: f64 = 10000.0;
/// Node's `setTimeout` cap (`TIMEOUT_MAX` = 2**31 - 1).
const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
pub const UPDATE_CHECK_CONCURRENCY: usize = 4;
pub const GIT_UPDATE_CONCURRENCY: usize = 4;

/// `getEnv()`: an empty `process.env` in a Linux sandbox is filled from
/// `/proc/self/environ`; the Rust process env is read the same way.
fn get_env() -> Vec<(String, String)> {
    if !cfg!(target_os = "linux") || !std::env::vars().next().is_none() {
        return std::env::vars().collect();
    }
    match std::fs::read_to_string("/proc/self/environ") {
        Ok(data) => {
            let mut env = Vec::new();
            for entry in data.split('\0') {
                if let Some(index) = entry.find('=') {
                    if index > 0 {
                        env.push((entry[..index].to_string(), entry[index + 1..].to_string()));
                    }
                }
            }
            env
        }
        Err(_) => std::env::vars().collect(),
    }
}

pub fn is_offline_mode_enabled() -> bool {
    offline_mode_from_value(std::env::var("PI_OFFLINE").ok().as_deref())
}

fn offline_mode_from_value(value: Option<&str>) -> bool {
    matches!(value, Some(value) if value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathMetadata {
    pub source: String,
    pub scope: String,
    pub origin: String,
    pub base_dir: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedResource {
    pub path: String,
    pub enabled: bool,
    pub metadata: PathMetadata,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResolvedPaths {
    pub extensions: Vec<ResolvedResource>,
    pub skills: Vec<ResolvedResource>,
    pub prompts: Vec<ResolvedResource>,
    pub themes: Vec<ResolvedResource>,
    pub diagnostics: Vec<ResourceDiagnostic>,
}

pub const MISSING_SOURCE_ACTION_INSTALL: &str = "install";
pub const MISSING_SOURCE_ACTION_SKIP: &str = "skip";
pub const MISSING_SOURCE_ACTION_ERROR: &str = "error";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEvent {
    pub event_type: String,
    pub action: String,
    pub source: String,
    pub message: Option<String>,
}

pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageUpdate {
    pub source: String,
    pub display_name: String,
    pub update_type: String,
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfiguredPackage {
    pub source: String,
    pub scope: String,
    pub filtered: bool,
    pub installed_path: Option<String>,
}

#[derive(Clone)]
pub struct PackageManagerOptions {
    pub cwd: String,
    pub agent_dir: String,
    pub settings_manager: Arc<std::sync::Mutex<SettingsManager>>,
    /// Directory of built-in skills shipped with the package. Defaults to the bundled skills dir; pass `None` to disable.
    pub bundled_skills_dir: Option<Option<String>>,
    /// Extra force-exclude patterns for built-in skills (e.g. unauthenticated MCP integrations).
    pub extra_builtin_skill_overrides: Option<Arc<dyn Fn() -> Vec<String> + Send + Sync>>,
}

pub const SOURCE_SCOPE_USER: &str = "user";
pub const SOURCE_SCOPE_PROJECT: &str = "project";
pub const SOURCE_SCOPE_TEMPORARY: &str = "temporary";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpmSource {
    pub spec: String,
    pub name: String,
    pub pinned: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GitInstallCompletion {
    repo: String,
    reference: Option<String>,
    head: String,
    package_json_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSource {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedSource {
    Npm(NpmSource),
    Git(GitSource),
    Local(LocalSource),
}

impl ParsedSource {
    fn pinned(&self) -> bool {
        match self {
            ParsedSource::Npm(source) => source.pinned,
            ParsedSource::Git(source) => source.pinned,
            ParsedSource::Local(_) => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PiManifest {
    pub extensions: Option<Vec<String>>,
    pub skills: Option<Vec<String>>,
    pub prompts: Option<Vec<String>>,
    pub themes: Option<Vec<String>>,
}

impl PiManifest {
    fn get(&self, resource_type: &str) -> Option<&Vec<String>> {
        match resource_type {
            "extensions" => self.extensions.as_ref(),
            "skills" => self.skills.as_ref(),
            "prompts" => self.prompts.as_ref(),
            "themes" => self.themes.as_ref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PackageFilter {
    pub extensions: Option<Vec<String>>,
    pub skills: Option<Vec<String>>,
    pub prompts: Option<Vec<String>>,
    pub themes: Option<Vec<String>>,
}

impl PackageFilter {
    fn get(&self, resource_type: &str) -> Option<&Vec<String>> {
        match resource_type {
            "extensions" => self.extensions.as_ref(),
            "skills" => self.skills.as_ref(),
            "prompts" => self.prompts.as_ref(),
            "themes" => self.themes.as_ref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResourceAccumulator {
    pub extensions: BTreeMap<String, (PathMetadata, bool)>,
    pub skills: BTreeMap<String, (PathMetadata, bool)>,
    pub prompts: BTreeMap<String, (PathMetadata, bool)>,
    pub themes: BTreeMap<String, (PathMetadata, bool)>,
    pub diagnostics: Vec<ResourceDiagnostic>,
}

pub const RESOURCE_TYPES: [&str; 4] = ["extensions", "skills", "prompts", "themes"];

/// Compute a numeric precedence rank for a resource based on its metadata.
/// Lower rank = higher precedence.
fn resource_precedence_rank(metadata: &PathMetadata) -> i64 {
    if metadata.source == "builtin" {
        return 5;
    }
    if metadata.origin == "package" {
        return 4;
    }
    let scope_base = if metadata.scope == "project" { 0 } else { 2 };
    scope_base + if metadata.source == "local" { 0 } else { 1 }
}

const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

fn to_posix_path(path: &str) -> String {
    path.replace(std::path::MAIN_SEPARATOR, "/")
}

fn get_home_dir() -> String {
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => home,
        _ => dirs::home_dir()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default(),
    }
}

fn prefix_ignore_pattern(line: &str, prefix: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('#') && !trimmed.starts_with("\\#") {
        return None;
    }

    let mut pattern = line.to_string();
    let mut negated = false;

    if pattern.starts_with('!') {
        negated = true;
        pattern = pattern[1..].to_string();
    } else if pattern.starts_with("\\!") {
        pattern = pattern[1..].to_string();
    }

    if pattern.starts_with('/') {
        pattern = pattern[1..].to_string();
    }

    let prefixed = if prefix.is_empty() {
        pattern
    } else {
        format!("{prefix}{pattern}")
    };
    Some(if negated { format!("!{prefixed}") } else { prefixed })
}

/// `ignore()` matcher with the accumulated gitignore rules.
#[derive(Debug, Clone, Default)]
struct IgnoreMatcher {
    patterns: Vec<String>,
    compiled: Option<ignore::gitignore::Gitignore>,
}

impl IgnoreMatcher {
    fn new() -> Self {
        IgnoreMatcher::default()
    }

    /// Append rules and recompile (rules only grow, so a rebuild is enough).
    fn add(&mut self, patterns: &[String]) {
        self.patterns.extend(patterns.iter().cloned());
        let mut builder = ignore::gitignore::GitignoreBuilder::new("");
        for pattern in &self.patterns {
            let _ = builder.add_line(None, pattern);
        }
        self.compiled = builder.build().ok();
    }

    fn ignores(&self, path: &str) -> bool {
        self.compiled
            .as_ref()
            .map(|matcher| matcher.matched_path_or_any_parents(path, false).is_ignore())
            .unwrap_or(false)
    }
}

fn add_ignore_rules(ig: &mut IgnoreMatcher, dir: &str, root_dir: &str) {
    let relative_dir = relative_path(root_dir, dir);
    let prefix = if relative_dir.is_empty() {
        String::new()
    } else {
        format!("{}/", to_posix_path(&relative_dir))
    };

    for filename in IGNORE_FILE_NAMES {
        let ignore_path = join_path(dir, filename);
        if !Path::new(&ignore_path).exists() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&ignore_path) else {
            continue;
        };
        // Unreadable ignore file: skip it rather than failing resource loading.
        let patterns: Vec<String> = content
            .split('\n')
            .map(|line| line.trim_end_matches('\r'))
            .filter_map(|line| prefix_ignore_pattern(line, &prefix))
            .collect();
        if !patterns.is_empty() {
            ig.add(&patterns);
        }
    }
}

fn is_pattern(value: &str) -> bool {
    value.starts_with('!')
        || value.starts_with('+')
        || value.starts_with('-')
        || value.contains('*')
        || value.contains('?')
}

fn is_override_pattern(value: &str) -> bool {
    value.starts_with('!') || value.starts_with('+') || value.starts_with('-')
}

fn has_glob_pattern(value: &str) -> bool {
    value.contains('*') || value.contains('?')
}

fn split_patterns(entries: &[String]) -> (Vec<String>, Vec<String>) {
    let mut plain = Vec::new();
    let mut patterns = Vec::new();
    for entry in entries {
        if is_pattern(entry) {
            patterns.push(entry.clone());
        } else {
            plain.push(entry.clone());
        }
    }
    (plain, patterns)
}

/// `FILE_PATTERNS[resourceType]` as an extension test.
fn file_pattern_matches(resource_type: &str, name: &str) -> bool {
    match resource_type {
        "extensions" => name.ends_with(".ts") || name.ends_with(".js"),
        "skills" | "prompts" => name.ends_with(".md"),
        "themes" => name.ends_with(".json"),
        _ => false,
    }
}

fn collect_files(
    dir: &str,
    resource_type: &str,
    skip_node_modules: bool,
    ignore_matcher: Option<&mut IgnoreMatcher>,
    root_dir: Option<&str>,
) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return files;
    }

    let root = root_dir.map(str::to_string).unwrap_or_else(|| dir.to_string());
    let mut owned = IgnoreMatcher::new();
    let ig: &mut IgnoreMatcher = match ignore_matcher {
        Some(matcher) => matcher,
        None => &mut owned,
    };
    add_ignore_rules(ig, dir, &root);

    let Ok(entries) = std::fs::read_dir(dir) else {
        // Ignore unreadable directories during file discovery.
        return files;
    };
    let mut sorted: Vec<std::fs::DirEntry> = entries.flatten().collect();
    sorted.sort_by_key(|entry| entry.file_name());

    for entry in sorted {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if skip_node_modules && name == "node_modules" {
            continue;
        }

        let full_path = join_path(dir, &name);
        let mut is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
        let mut is_file = entry.file_type().map(|kind| kind.is_file()).unwrap_or(false);

        if entry.file_type().map(|kind| kind.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => {
                    is_dir = stats.is_dir();
                    is_file = stats.is_file();
                }
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&relative_path(&root, &full_path));
        let ignore_path = if is_dir { format!("{rel_path}/") } else { rel_path };
        if ig.ignores(&ignore_path) {
            continue;
        }

        if is_dir {
            files.extend(collect_files(
                &full_path,
                resource_type,
                skip_node_modules,
                Some(&mut *ig),
                Some(&root),
            ));
        } else if is_file && file_pattern_matches(resource_type, &name) {
            files.push(full_path);
        }
    }

    files
}

fn collect_skill_entries(
    dir: &str,
    mode: &str,
    ignore_matcher: Option<&mut IgnoreMatcher>,
    root_dir: Option<&str>,
) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }

    let root = root_dir.map(str::to_string).unwrap_or_else(|| dir.to_string());
    let mut owned = IgnoreMatcher::new();
    let ig: &mut IgnoreMatcher = match ignore_matcher {
        Some(matcher) => matcher,
        None => &mut owned,
    };
    add_ignore_rules(ig, dir, &root);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        // Ignore unreadable directories during skill discovery.
        return entries;
    };
    let mut sorted: Vec<std::fs::DirEntry> = dir_entries.flatten().collect();
    sorted.sort_by_key(|entry| entry.file_name());

    for entry in &sorted {
        let name = entry.file_name().to_string_lossy().to_string();
        if name != "SKILL.md" {
            continue;
        }
        let full_path = join_path(dir, &name);
        let mut is_file = entry.file_type().map(|kind| kind.is_file()).unwrap_or(false);
        if entry.file_type().map(|kind| kind.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => is_file = stats.is_file(),
                Err(_) => continue,
            }
        }
        let rel_path = to_posix_path(&relative_path(&root, &full_path));
        if is_file && !ig.ignores(&rel_path) {
            entries.push(full_path);
            return entries;
        }
    }

    for entry in &sorted {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if name == "node_modules" {
            continue;
        }

        let full_path = join_path(dir, &name);
        let mut is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
        let mut is_file = entry.file_type().map(|kind| kind.is_file()).unwrap_or(false);

        if entry.file_type().map(|kind| kind.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => {
                    is_dir = stats.is_dir();
                    is_file = stats.is_file();
                }
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&relative_path(&root, &full_path));
        if mode == "pi" && dir == root.as_str() && is_file && name.ends_with(".md") && !ig.ignores(&rel_path) {
            entries.push(full_path);
            continue;
        }

        if !is_dir {
            continue;
        }
        if ig.ignores(&format!("{rel_path}/")) {
            continue;
        }

        entries.extend(collect_skill_entries(&full_path, mode, Some(&mut *ig), Some(&root)));
    }

    entries
}

fn collect_auto_skill_entries(dir: &str, mode: &str) -> Vec<String> {
    collect_skill_entries(dir, mode, None, None)
}

fn find_git_repo_root(start_dir: &str) -> Option<String> {
    let mut dir = absolute_path(start_dir);
    loop {
        if Path::new(&join_path(&dir, ".git")).exists() {
            return Some(dir);
        }
        let parent = parent_path(&dir);
        if parent == dir {
            return None;
        }
        dir = parent;
    }
}

fn collect_ancestor_agents_skill_dirs(start_dir: &str) -> Vec<String> {
    let mut skill_dirs: Vec<String> = Vec::new();
    let resolved_start_dir = absolute_path(start_dir);
    let git_repo_root = find_git_repo_root(&resolved_start_dir);

    let mut dir = resolved_start_dir;
    loop {
        skill_dirs.push(join_path(&join_path(&dir, ".agents"), "skills"));
        if git_repo_root.as_deref() == Some(dir.as_str()) {
            break;
        }
        let parent = parent_path(&dir);
        if parent == dir {
            break;
        }
        dir = parent;
    }

    skill_dirs
}

fn collect_auto_prompt_entries(dir: &str) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }

    let mut ig = IgnoreMatcher::new();
    add_ignore_rules(&mut ig, dir, dir);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        // Ignore unreadable directories during prompt discovery.
        return entries;
    };
    let mut sorted: Vec<std::fs::DirEntry> = dir_entries.flatten().collect();
    sorted.sort_by_key(|entry| entry.file_name());

    for entry in sorted {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let full_path = join_path(dir, &name);
        let mut is_file = entry.file_type().map(|kind| kind.is_file()).unwrap_or(false);
        if entry.file_type().map(|kind| kind.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => is_file = stats.is_file(),
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&relative_path(dir, &full_path));
        if ig.ignores(&rel_path) {
            continue;
        }

        if is_file && name.ends_with(".md") {
            entries.push(full_path);
        }
    }

    entries
}

fn collect_auto_theme_entries(dir: &str) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }

    let mut ig = IgnoreMatcher::new();
    add_ignore_rules(&mut ig, dir, dir);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        // Ignore unreadable directories during theme discovery.
        return entries;
    };
    let mut sorted: Vec<std::fs::DirEntry> = dir_entries.flatten().collect();
    sorted.sort_by_key(|entry| entry.file_name());

    for entry in sorted {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let full_path = join_path(dir, &name);
        let mut is_file = entry.file_type().map(|kind| kind.is_file()).unwrap_or(false);
        if entry.file_type().map(|kind| kind.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => is_file = stats.is_file(),
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&relative_path(dir, &full_path));
        if ig.ignores(&rel_path) {
            continue;
        }

        if is_file && name.ends_with(".json") {
            entries.push(full_path);
        }
    }

    entries
}

fn read_pi_manifest_file(package_json_path: &str) -> Option<PiManifest> {
    let content = std::fs::read_to_string(package_json_path).ok()?;
    let pkg: Value = serde_json::from_str(&content).ok()?;
    let pi = pkg.get("pi")?;
    if pi.is_null() {
        return None;
    }
    Some(parse_pi_manifest(pi))
}

fn parse_pi_manifest(value: &Value) -> PiManifest {
    let strings = |key: &str| -> Option<Vec<String>> {
        value.get(key).and_then(Value::as_array).map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
    };
    PiManifest {
        extensions: strings("extensions"),
        skills: strings("skills"),
        prompts: strings("prompts"),
        themes: strings("themes"),
    }
}

fn resolve_extension_entries(dir: &str) -> Option<Vec<String>> {
    let package_json_path = join_path(dir, "package.json");
    if Path::new(&package_json_path).exists() {
        if let Some(manifest) = read_pi_manifest_file(&package_json_path) {
            if let Some(extensions) = manifest.extensions.as_ref().filter(|entries| !entries.is_empty()) {
                let mut entries: Vec<String> = Vec::new();
                for ext_path in extensions {
                    let resolved_ext_path = join_path(dir, ext_path);
                    if Path::new(&resolved_ext_path).exists() {
                        entries.push(resolved_ext_path);
                    }
                }
                if !entries.is_empty() {
                    return Some(entries);
                }
            }
        }
    }

    let index_ts = join_path(dir, "index.ts");
    let index_js = join_path(dir, "index.js");
    if Path::new(&index_ts).exists() {
        return Some(vec![index_ts]);
    }
    if Path::new(&index_js).exists() {
        return Some(vec![index_js]);
    }

    None
}

fn collect_auto_extension_entries(dir: &str) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    if !Path::new(dir).exists() {
        return entries;
    }
    if let Some(root_entries) = resolve_extension_entries(dir) {
        return root_entries;
    }
    let mut ig = IgnoreMatcher::new();
    add_ignore_rules(&mut ig, dir, dir);

    let Ok(dir_entries) = std::fs::read_dir(dir) else {
        // Ignore unreadable directories during extension discovery.
        return entries;
    };
    let mut sorted: Vec<std::fs::DirEntry> = dir_entries.flatten().collect();
    sorted.sort_by_key(|entry| entry.file_name());

    for entry in sorted {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }

        let full_path = join_path(dir, &name);
        let mut is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
        let mut is_file = entry.file_type().map(|kind| kind.is_file()).unwrap_or(false);

        if entry.file_type().map(|kind| kind.is_symlink()).unwrap_or(false) {
            match std::fs::metadata(&full_path) {
                Ok(stats) => {
                    is_dir = stats.is_dir();
                    is_file = stats.is_file();
                }
                Err(_) => continue,
            }
        }

        let rel_path = to_posix_path(&relative_path(dir, &full_path));
        let ignore_path = if is_dir { format!("{rel_path}/") } else { rel_path };
        if ig.ignores(&ignore_path) {
            continue;
        }

        if is_file && (name.ends_with(".ts") || name.ends_with(".js")) {
            entries.push(full_path);
        } else if is_dir {
            if let Some(resolved_entries) = resolve_extension_entries(&full_path) {
                entries.extend(resolved_entries);
            }
        }
    }

    entries
}

/// Collect resource files from a directory based on resource type.
fn collect_resource_files(dir: &str, resource_type: &str) -> Vec<String> {
    if resource_type == "skills" {
        return collect_skill_entries(dir, "pi", None, None);
    }
    if resource_type == "extensions" {
        return collect_auto_extension_entries(dir);
    }
    collect_files(dir, resource_type, true, None, None)
}

/// `minimatch(path, pattern)` with the default options (`*` does not cross `/`).
fn minimatch(path: &str, pattern: &str) -> bool {
    let Ok(glob) = globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .backslash_escape(true)
        .build()
    else {
        return false;
    };
    glob.compile_matcher().is_match(path)
}

fn matches_any_pattern(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    let rel = to_posix_path(&relative_path(base_dir, file_path));
    let name = basename(file_path);
    let file_path_posix = to_posix_path(file_path);
    let is_skill_file = name == "SKILL.md";
    let parent_dir = if is_skill_file {
        Some(parent_path(file_path))
    } else {
        None
    };
    let parent_rel = parent_dir
        .as_ref()
        .map(|dir| to_posix_path(&relative_path(base_dir, dir)));
    let parent_name = parent_dir.as_ref().map(|dir| basename(dir));
    let parent_dir_posix = parent_dir.as_ref().map(|dir| to_posix_path(dir));

    patterns.iter().any(|pattern| {
        let normalized_pattern = to_posix_path(pattern);
        if minimatch(&rel, &normalized_pattern)
            || minimatch(&name, &normalized_pattern)
            || minimatch(&file_path_posix, &normalized_pattern)
        {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        minimatch(parent_rel.as_deref().unwrap_or(""), &normalized_pattern)
            || minimatch(parent_name.as_deref().unwrap_or(""), &normalized_pattern)
            || minimatch(parent_dir_posix.as_deref().unwrap_or(""), &normalized_pattern)
    })
}

fn normalize_exact_pattern(pattern: &str) -> String {
    let normalized = if pattern.starts_with("./") || pattern.starts_with(".\\") {
        &pattern[2..]
    } else {
        pattern
    };
    to_posix_path(normalized)
}

fn matches_any_exact_pattern(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    if patterns.is_empty() {
        return false;
    }
    let rel = to_posix_path(&relative_path(base_dir, file_path));
    let name = basename(file_path);
    let file_path_posix = to_posix_path(file_path);
    let is_skill_file = name == "SKILL.md";
    let parent_dir = if is_skill_file {
        Some(parent_path(file_path))
    } else {
        None
    };
    let parent_rel = parent_dir
        .as_ref()
        .map(|dir| to_posix_path(&relative_path(base_dir, dir)));
    let parent_dir_posix = parent_dir.as_ref().map(|dir| to_posix_path(dir));

    patterns.iter().any(|pattern| {
        let normalized = normalize_exact_pattern(pattern);
        if normalized == rel || normalized == file_path_posix {
            return true;
        }
        if !is_skill_file {
            return false;
        }
        Some(&normalized) == parent_rel.as_ref() || Some(&normalized) == parent_dir_posix.as_ref()
    })
}

fn get_override_patterns(entries: &[String]) -> Vec<String> {
    entries
        .iter()
        .filter(|pattern| is_override_pattern(pattern))
        .cloned()
        .collect()
}

fn is_enabled_by_overrides(file_path: &str, patterns: &[String], base_dir: &str) -> bool {
    let overrides = get_override_patterns(patterns);
    let excludes: Vec<String> = overrides
        .iter()
        .filter(|pattern| pattern.starts_with('!'))
        .map(|pattern| pattern[1..].to_string())
        .collect();
    let force_includes: Vec<String> = overrides
        .iter()
        .filter(|pattern| pattern.starts_with('+'))
        .map(|pattern| pattern[1..].to_string())
        .collect();
    let force_excludes: Vec<String> = overrides
        .iter()
        .filter(|pattern| pattern.starts_with('-'))
        .map(|pattern| pattern[1..].to_string())
        .collect();

    let mut enabled = true;
    if !excludes.is_empty() && matches_any_pattern(file_path, &excludes, base_dir) {
        enabled = false;
    }
    if !force_includes.is_empty() && matches_any_exact_pattern(file_path, &force_includes, base_dir) {
        enabled = true;
    }
    if !force_excludes.is_empty() && matches_any_exact_pattern(file_path, &force_excludes, base_dir) {
        enabled = false;
    }
    enabled
}

/// Apply patterns to paths and return the set of enabled paths.
fn apply_patterns(all_paths: &[String], patterns: &[String], base_dir: &str) -> HashSet<String> {
    let mut includes: Vec<String> = Vec::new();
    let mut excludes: Vec<String> = Vec::new();
    let mut force_includes: Vec<String> = Vec::new();
    let mut force_excludes: Vec<String> = Vec::new();

    for pattern in patterns {
        if let Some(value) = pattern.strip_prefix('+') {
            force_includes.push(value.to_string());
        } else if let Some(value) = pattern.strip_prefix('-') {
            force_excludes.push(value.to_string());
        } else if let Some(value) = pattern.strip_prefix('!') {
            excludes.push(value.to_string());
        } else {
            includes.push(pattern.clone());
        }
    }

    // Apply patterns in order: includes, excludes, force-includes, then force-excludes.
    let mut result: Vec<String> = if includes.is_empty() {
        all_paths.to_vec()
    } else {
        all_paths
            .iter()
            .filter(|file_path| matches_any_pattern(file_path, &includes, base_dir))
            .cloned()
            .collect()
    };
    if !excludes.is_empty() {
        result.retain(|file_path| !matches_any_pattern(file_path, &excludes, base_dir));
    }
    if !force_includes.is_empty() {
        for file_path in all_paths {
            if !result.contains(file_path) && matches_any_exact_pattern(file_path, &force_includes, base_dir) {
                result.push(file_path.clone());
            }
        }
    }
    if !force_excludes.is_empty() {
        result.retain(|file_path| !matches_any_exact_pattern(file_path, &force_excludes, base_dir));
    }

    result.into_iter().collect()
}

/// `globSync(entry, { cwd: root, absolute: true, dot: false, nodir: false })`.
fn glob_sync(pattern: &str, root: &str) -> Vec<String> {
    let mut results: Vec<String> = Vec::new();
    let Ok(builder) = globset::GlobBuilder::new(&to_posix_path(pattern))
        .literal_separator(true)
        .backslash_escape(true)
        .build()
    else {
        return results;
    };
    let matcher = builder.compile_matcher();
    let mut stack = vec![root.to_string()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let full = join_path(&dir, &name);
            let is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
            let rel = to_posix_path(&relative_path(root, &full));
            if matcher.is_match(&rel) {
                results.push(absolute_path(&full));
            }
            if is_dir {
                stack.push(full);
            }
        }
    }
    results.sort();
    results
}

// =============================================================================
// DefaultPackageManager
// =============================================================================

#[derive(Debug, Clone)]
struct ConfiguredUpdateSource {
    source: String,
    scope: String,
}

#[derive(Debug, Clone)]
struct NpmUpdateTarget {
    source: String,
    scope: String,
    parsed: NpmSource,
}

#[derive(Debug, Clone)]
struct GitUpdateTarget {
    source: String,
    scope: String,
    parsed: GitSource,
}

pub struct DefaultPackageManager {
    cwd: String,
    agent_dir: String,
    settings_manager: Arc<std::sync::Mutex<SettingsManager>>,
    bundled_skills_dir: Option<String>,
    extra_builtin_skill_overrides: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
    /// `globalNpmRoot` cache keyed by the npm command it was resolved for.
    global_npm_root: std::sync::Mutex<Option<String>>,
    global_npm_root_command_key: std::sync::Mutex<Option<String>>,
    progress_callback: Option<ProgressCallback>,
}

impl DefaultPackageManager {
    pub fn new(options: PackageManagerOptions) -> Self {
        let bundled_skills_dir = match options.bundled_skills_dir {
            None => Some(get_bundled_skills_dir()),
            Some(inner) => inner,
        };
        DefaultPackageManager {
            cwd: options.cwd,
            agent_dir: options.agent_dir,
            settings_manager: options.settings_manager,
            bundled_skills_dir,
            extra_builtin_skill_overrides: options
                .extra_builtin_skill_overrides
                .unwrap_or_else(|| Arc::new(Vec::new)),
            global_npm_root: std::sync::Mutex::new(None),
            global_npm_root_command_key: std::sync::Mutex::new(None),
            progress_callback: None,
        }
    }

    pub fn set_progress_callback(&mut self, callback: Option<ProgressCallback>) {
        self.progress_callback = callback;
    }

    fn settings(&self) -> std::sync::MutexGuard<'_, SettingsManager> {
        self.settings_manager.lock().expect("settings manager")
    }

    pub fn add_source_to_settings(&self, source: &str, local: bool) -> bool {
        let scope = if local { SOURCE_SCOPE_PROJECT } else { SOURCE_SCOPE_USER };
        let mut settings_manager = self.settings();
        let current_settings = if scope == SOURCE_SCOPE_PROJECT {
            settings_manager.get_project_settings()
        } else {
            settings_manager.get_global_settings()
        };
        let current_packages = package_list(&current_settings);
        let normalized_source = self.normalize_package_source_for_settings(source, scope);
        let exists = current_packages
            .iter()
            .any(|existing| self.package_sources_match(existing, source, scope));
        if exists {
            return false;
        }
        let mut next_packages = current_packages;
        next_packages.push(Value::String(normalized_source));
        if scope == SOURCE_SCOPE_PROJECT {
            settings_manager.set_project_packages(next_packages);
        } else {
            settings_manager.set_packages(next_packages);
        }
        true
    }

    pub fn remove_source_from_settings(&self, source: &str, local: bool) -> bool {
        let scope = if local { SOURCE_SCOPE_PROJECT } else { SOURCE_SCOPE_USER };
        let mut settings_manager = self.settings();
        let current_settings = if scope == SOURCE_SCOPE_PROJECT {
            settings_manager.get_project_settings()
        } else {
            settings_manager.get_global_settings()
        };
        let current_packages = package_list(&current_settings);
        let next_packages: Vec<Value> = current_packages
            .iter()
            .filter(|existing| !self.package_sources_match(existing, source, scope))
            .cloned()
            .collect();
        let changed = next_packages.len() != current_packages.len();
        if !changed {
            return false;
        }
        if scope == SOURCE_SCOPE_PROJECT {
            settings_manager.set_project_packages(next_packages);
        } else {
            settings_manager.set_packages(next_packages);
        }
        true
    }

    pub fn get_installed_path(&self, source: &str, scope: &str) -> Option<String> {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => {
                let path = self.get_npm_install_path(&parsed, scope);
                if Path::new(&path).exists() {
                    Some(path)
                } else {
                    None
                }
            }
            ParsedSource::Git(parsed) => {
                let path = self.validated_git_install_path(&parsed, scope).ok()?;
                if Path::new(&path).exists() {
                    Some(path)
                } else {
                    None
                }
            }
            ParsedSource::Local(parsed) => {
                let base_dir = self.get_base_dir_for_scope(scope);
                let path = self.resolve_path_from_base(&parsed.path, &base_dir);
                if Path::new(&path).exists() {
                    Some(path)
                } else {
                    None
                }
            }
        }
    }

    fn emit_progress(&self, event: ProgressEvent) {
        if let Some(callback) = &self.progress_callback {
            callback(event);
        }
    }

    /// `withProgress(action, source, message, operation)`: the operation is only
    /// polled after the `start` event, so a boxed future matches the TypeScript.
    async fn with_progress(
        &self,
        action: &str,
        source: &str,
        message: &str,
        operation: futures::future::BoxFuture<'_, Result<(), String>>,
    ) -> Result<(), String> {
        self.emit_progress(ProgressEvent {
            event_type: "start".to_string(),
            action: action.to_string(),
            source: source.to_string(),
            message: Some(message.to_string()),
        });
        match operation.await {
            Ok(()) => {
                self.emit_progress(ProgressEvent {
                    event_type: "complete".to_string(),
                    action: action.to_string(),
                    source: source.to_string(),
                    message: None,
                });
                Ok(())
            }
            Err(error) => {
                self.emit_progress(ProgressEvent {
                    event_type: "error".to_string(),
                    action: action.to_string(),
                    source: source.to_string(),
                    message: Some(error.clone()),
                });
                Err(error)
            }
        }
    }

    pub async fn resolve(
        &self,
        on_missing: Option<&(dyn Fn(String) -> futures::future::BoxFuture<'static, String> + Send + Sync)>,
    ) -> Result<ResolvedPaths, String> {
        let mut accumulator = ResourceAccumulator::default();
        let global_settings = self.settings().get_global_settings();
        let project_settings = self.settings().get_project_settings();
        // Project resources win collisions with global resources.
        let mut all_packages: Vec<(Value, String)> = Vec::new();
        for pkg in package_list(&project_settings) {
            all_packages.push((pkg, SOURCE_SCOPE_PROJECT.to_string()));
        }
        for pkg in package_list(&global_settings) {
            all_packages.push((pkg, SOURCE_SCOPE_USER.to_string()));
        }
        // Deduplicate by package identity after recording project precedence.
        let package_sources = self.dedupe_packages(&all_packages);
        self.resolve_package_sources(&package_sources, &mut accumulator, on_missing)
            .await?;

        let global_base_dir = self.agent_dir.clone();
        let project_base_dir = join_path(&self.cwd, CONFIG_DIR_NAME);

        for resource_type in RESOURCE_TYPES {
            let global_entries = string_list(&global_settings, resource_type);
            let project_entries = string_list(&project_settings, resource_type);
            self.resolve_local_entries(
                &project_entries,
                resource_type,
                &mut accumulator,
                PathMetadata {
                    source: "local".to_string(),
                    scope: SOURCE_SCOPE_PROJECT.to_string(),
                    origin: "top-level".to_string(),
                    base_dir: None,
                },
                &project_base_dir,
            );
            self.resolve_local_entries(
                &global_entries,
                resource_type,
                &mut accumulator,
                PathMetadata {
                    source: "local".to_string(),
                    scope: SOURCE_SCOPE_USER.to_string(),
                    origin: "top-level".to_string(),
                    base_dir: None,
                },
                &global_base_dir,
            );
        }

        self.add_auto_discovered_resources(
            &mut accumulator,
            &global_settings,
            &project_settings,
            &global_base_dir,
            &project_base_dir,
        );

        Ok(self.to_resolved_paths(&accumulator))
    }

    pub async fn resolve_extension_sources(
        &self,
        sources: &[String],
        local: bool,
        temporary: bool,
    ) -> Result<ResolvedPaths, String> {
        let mut accumulator = ResourceAccumulator::default();
        let scope = if temporary {
            SOURCE_SCOPE_TEMPORARY
        } else if local {
            SOURCE_SCOPE_PROJECT
        } else {
            SOURCE_SCOPE_USER
        };
        let package_sources: Vec<(Value, String)> = sources
            .iter()
            .map(|source| (Value::String(source.clone()), scope.to_string()))
            .collect();
        self.resolve_package_sources(&package_sources, &mut accumulator, None)
            .await?;
        Ok(self.to_resolved_paths(&accumulator))
    }

    pub fn list_configured_packages(&self) -> Vec<ConfiguredPackage> {
        let global_settings = self.settings().get_global_settings();
        let project_settings = self.settings().get_project_settings();
        let mut configured_packages: Vec<ConfiguredPackage> = Vec::new();

        for pkg in package_list(&global_settings) {
            let source = package_source_string(&pkg);
            let installed_path = self.get_installed_path(&source, SOURCE_SCOPE_USER);
            configured_packages.push(ConfiguredPackage {
                source,
                scope: SOURCE_SCOPE_USER.to_string(),
                filtered: pkg.is_object(),
                installed_path,
            });
        }

        for pkg in package_list(&project_settings) {
            let source = package_source_string(&pkg);
            let installed_path = self.get_installed_path(&source, SOURCE_SCOPE_PROJECT);
            configured_packages.push(ConfiguredPackage {
                source,
                scope: SOURCE_SCOPE_PROJECT.to_string(),
                filtered: pkg.is_object(),
                installed_path,
            });
        }

        configured_packages
    }

    pub async fn install(&self, source: &str, local: bool) -> Result<(), String> {
        let parsed = self.parse_source(source);
        let scope = if local { SOURCE_SCOPE_PROJECT } else { SOURCE_SCOPE_USER };
        let message = format!("Installing {source}...");
        self.with_progress("install", source, &message, Box::pin(async {
            match &parsed {
                ParsedSource::Npm(npm) => {
                    self.install_npm(npm, scope, false).await?;
                    Ok(())
                }
                ParsedSource::Git(git) => {
                    self.install_git(git, scope).await?;
                    Ok(())
                }
                ParsedSource::Local(local_source) => {
                    let resolved = self.resolve_path(&local_source.path);
                    if !Path::new(&resolved).exists() {
                        return Err(format!("Path does not exist: {resolved}"));
                    }
                    Ok(())
                }
            }
        }))
        .await
    }

    pub async fn install_and_persist(&self, source: &str, local: bool) -> Result<(), String> {
        self.install(source, local).await?;
        self.add_source_to_settings(source, local);
        Ok(())
    }

    pub async fn remove(&self, source: &str, local: bool) -> Result<(), String> {
        let parsed = self.parse_source(source);
        let scope = if local { SOURCE_SCOPE_PROJECT } else { SOURCE_SCOPE_USER };
        let message = format!("Removing {source}...");
        self.with_progress("remove", source, &message, Box::pin(async {
            match &parsed {
                ParsedSource::Npm(npm) => {
                    self.uninstall_npm(npm, scope).await?;
                    Ok(())
                }
                ParsedSource::Git(git) => self.remove_git(git, scope),
                ParsedSource::Local(_) => Ok(()),
            }
        }))
        .await
    }

    pub async fn remove_and_persist(&self, source: &str, local: bool) -> Result<bool, String> {
        self.remove(source, local).await?;
        Ok(self.remove_source_from_settings(source, local))
    }

    pub async fn update(&self, source: Option<&str>) -> Result<(), String> {
        let global_settings = self.settings().get_global_settings();
        let project_settings = self.settings().get_project_settings();
        let identity = source.map(|source| self.get_package_identity(source, None));
        let mut matched = false;
        let mut update_sources: Vec<ConfiguredUpdateSource> = Vec::new();

        for pkg in package_list(&global_settings) {
            let source_str = package_source_string(&pkg);
            if let Some(identity) = &identity {
                if self.get_package_identity(&source_str, Some(SOURCE_SCOPE_USER)).as_str() != identity.as_str() {
                    continue;
                }
            }
            matched = true;
            update_sources.push(ConfiguredUpdateSource {
                source: source_str,
                scope: SOURCE_SCOPE_USER.to_string(),
            });
        }
        for pkg in package_list(&project_settings) {
            let source_str = package_source_string(&pkg);
            if let Some(identity) = &identity {
                if self.get_package_identity(&source_str, Some(SOURCE_SCOPE_PROJECT)).as_str() != identity.as_str() {
                    continue;
                }
            }
            matched = true;
            update_sources.push(ConfiguredUpdateSource {
                source: source_str,
                scope: SOURCE_SCOPE_PROJECT.to_string(),
            });
        }

        if let Some(source) = source {
            if !matched {
                let mut configured: Vec<Value> = package_list(&global_settings);
                configured.extend(package_list(&project_settings));
                return Err(self.build_no_matching_package_message(source, &configured));
            }
        }

        self.update_configured_sources(update_sources).await
    }

    async fn update_configured_sources(&self, sources: Vec<ConfiguredUpdateSource>) -> Result<(), String> {
        if is_offline_mode_enabled() || sources.is_empty() {
            return Ok(());
        }

        let mut npm_candidates: Vec<NpmUpdateTarget> = Vec::new();
        let mut git_candidates: Vec<GitUpdateTarget> = Vec::new();

        for entry in &sources {
            let parsed = self.parse_source(&entry.source);
            if parsed.pinned() {
                continue;
            }
            match parsed {
                ParsedSource::Local(_) => continue,
                ParsedSource::Npm(npm) => npm_candidates.push(NpmUpdateTarget {
                    source: entry.source.clone(),
                    scope: entry.scope.clone(),
                    parsed: npm,
                }),
                ParsedSource::Git(git) => git_candidates.push(GitUpdateTarget {
                    source: entry.source.clone(),
                    scope: entry.scope.clone(),
                    parsed: git,
                }),
            }
        }

        // `runWithConcurrency(npmCheckTasks, UPDATE_CHECK_CONCURRENCY)`.
        let mut npm_check_results: Vec<(NpmUpdateTarget, bool)> = Vec::new();
        for chunk in npm_candidates.chunks(UPDATE_CHECK_CONCURRENCY.max(1)) {
            let checks = chunk.iter().map(|entry| async move {
                (entry.clone(), self.should_update_npm_source(&entry.parsed, &entry.scope).await)
            });
            npm_check_results.extend(futures::future::join_all(checks).await);
        }

        let mut user_npm_updates: Vec<NpmUpdateTarget> = Vec::new();
        let mut project_npm_updates: Vec<NpmUpdateTarget> = Vec::new();
        for (entry, should_update) in npm_check_results {
            if !should_update {
                continue;
            }
            if entry.scope == SOURCE_SCOPE_USER {
                user_npm_updates.push(entry);
            } else {
                project_npm_updates.push(entry);
            }
        }

        let mut tasks: Vec<futures::future::BoxFuture<'_, Result<(), String>>> = Vec::new();
        if !user_npm_updates.is_empty() {
            tasks.push(Box::pin(self.update_npm_batch(user_npm_updates, SOURCE_SCOPE_USER)));
        }
        if !project_npm_updates.is_empty() {
            tasks.push(Box::pin(self.update_npm_batch(project_npm_updates, SOURCE_SCOPE_PROJECT)));
        }
        if !git_candidates.is_empty() {
            let git_tasks: Vec<(String, GitSource, String)> = git_candidates
                .into_iter()
                .map(|entry| (entry.source, entry.parsed, entry.scope))
                .collect();
            tasks.push(Box::pin(async move {
                let mut results: Vec<Result<(), String>> = Vec::new();
                for chunk in git_tasks.chunks(GIT_UPDATE_CONCURRENCY.max(1)) {
                    let mut chunk_tasks = Vec::new();
                    for (source, parsed, scope) in chunk {
                        chunk_tasks.push(async move {
                            let message = format!("Updating {source}...");
                            self.with_progress(
                                "update",
                                source,
                                &message,
                                Box::pin(async { self.update_git(parsed, scope).await }),
                            ).await
                        });
                    }
                    results.extend(futures::future::join_all(chunk_tasks).await);
                }
                for result in results {
                    result?;
                }
                Ok(())
            }));
        }

        for result in futures::future::join_all(tasks).await {
            result?;
        }
        Ok(())
    }

    async fn should_update_npm_source(&self, source: &NpmSource, scope: &str) -> bool {
        let installed_path = self.get_npm_install_path(source, scope);
        let installed_version = if Path::new(&installed_path).exists() {
            self.get_installed_npm_version(&installed_path)
        } else {
            None
        };
        let Some(installed_version) = installed_version else {
            return true;
        };

        match self.get_latest_npm_version(&source.name).await {
            Ok(latest_version) => latest_version != installed_version,
            // Preserve the existing update policy when version lookup fails.
            Err(_) => true,
        }
    }

    async fn update_npm_batch(&self, sources: Vec<NpmUpdateTarget>, scope: &str) -> Result<(), String> {
        if sources.is_empty() {
            return Ok(());
        }

        let source_label = if sources.len() == 1 {
            sources[0].source.clone()
        } else {
            format!("{scope} npm packages")
        };
        let message = if sources.len() == 1 {
            format!("Updating {}...", sources[0].source)
        } else {
            format!("Updating {scope} npm packages...")
        };
        let specs: Vec<String> = sources
            .iter()
            .map(|entry| format!("{}@latest", entry.parsed.name))
            .collect();

        self.with_progress(
            "update",
            &source_label,
            &message,
            Box::pin(async { self.install_npm_batch(&specs, scope).await }),
        )
        .await
    }

    async fn install_npm_batch(&self, specs: &[String], scope: &str) -> Result<(), String> {
        if scope == SOURCE_SCOPE_USER {
            let mut args: Vec<String> = vec!["install".to_string(), "-g".to_string()];
            args.extend(specs.iter().cloned());
            return self.run_npm_command(&args, None).await;
        }
        let install_root = self.get_npm_install_root(scope, false);
        self.ensure_npm_project(&install_root);
        let mut args: Vec<String> = vec!["install".to_string()];
        args.extend(specs.iter().cloned());
        args.push("--prefix".to_string());
        args.push(install_root);
        self.run_npm_command(&args, None).await
    }

    pub async fn check_for_available_updates(&self) -> Vec<PackageUpdate> {
        self.check_for_available_updates_with_offline(is_offline_mode_enabled()).await
    }

    async fn check_for_available_updates_with_offline(&self, offline: bool) -> Vec<PackageUpdate> {
        if offline {
            return Vec::new();
        }

        let global_settings = self.settings().get_global_settings();
        let project_settings = self.settings().get_project_settings();
        let mut all_packages: Vec<(Value, String)> = Vec::new();
        for pkg in package_list(&project_settings) {
            all_packages.push((pkg, SOURCE_SCOPE_PROJECT.to_string()));
        }
        for pkg in package_list(&global_settings) {
            all_packages.push((pkg, SOURCE_SCOPE_USER.to_string()));
        }

        let package_sources: Vec<(Value, String)> = self
            .dedupe_packages(&all_packages)
            .into_iter()
            .filter(|(_, scope)| scope != SOURCE_SCOPE_TEMPORARY)
            .collect();
        let mut results: Vec<Option<PackageUpdate>> = Vec::new();
        for chunk in package_sources.chunks(UPDATE_CHECK_CONCURRENCY.max(1)) {
            let checks = chunk.iter().map(|(pkg, scope)| async move {
                self.check_package_for_update(pkg, scope).await
            });
            results.extend(futures::future::join_all(checks).await);
        }
        results.into_iter().flatten().collect()
    }

    /// The per-package body of `checkForAvailableUpdates`.
    async fn check_package_for_update(&self, pkg: &Value, scope: &str) -> Option<PackageUpdate> {
        {
            let scope = scope.to_string();
            let source = package_source_string(pkg);
            let parsed = self.parse_source(&source);
            if parsed.pinned() {
                return None;
            }

            match parsed {
                ParsedSource::Local(_) => None,
                ParsedSource::Npm(npm) => {
                    let installed_path = self.get_npm_install_path(&npm, &scope);
                    if !Path::new(&installed_path).exists() {
                        return None;
                    }
                    if !self.npm_has_available_update(&npm, &installed_path).await {
                        return None;
                    }
                    Some(PackageUpdate {
                        source,
                        display_name: npm.name.clone(),
                        update_type: "npm".to_string(),
                        scope,
                    })
                }
                ParsedSource::Git(git) => {
                    let installed_path = self.validated_git_install_path(&git, &scope).ok()?;
                    if !Path::new(&installed_path).exists()
                        || !self.git_install_is_ready(&git, &installed_path).await
                    {
                        return None;
                    }
                    if !self.git_has_available_update(&installed_path).await {
                        return None;
                    }
                    Some(PackageUpdate {
                        source,
                        display_name: format!("{}/{}", git.host, git.path),
                        update_type: "git".to_string(),
                        scope,
                    })
                }
            }
        }
    }
}

impl DefaultPackageManager {
    async fn resolve_package_sources(
        &self,
        sources: &[(Value, String)],
        accumulator: &mut ResourceAccumulator,
        on_missing: Option<&(dyn Fn(String) -> futures::future::BoxFuture<'static, String> + Send + Sync)>,
    ) -> Result<(), String> {
        for (pkg, scope) in sources {
            let source_str = package_source_string(pkg);
            let filter = package_filter(pkg);
            let parsed = self.parse_source(&source_str);
            let metadata = PathMetadata {
                source: source_str.clone(),
                scope: scope.clone(),
                origin: "package".to_string(),
                base_dir: None,
            };

            if let ParsedSource::Local(local_source) = &parsed {
                let base_dir = self.get_base_dir_for_scope(scope);
                self.resolve_local_extension_source(local_source, accumulator, filter.as_ref(), metadata, &base_dir);
                continue;
            }

            let install_missing = || async {
                if is_offline_mode_enabled() {
                    return Ok(false);
                }
                if let Some(callback) = on_missing {
                    let action = callback(source_str.clone()).await;
                    if action == MISSING_SOURCE_ACTION_SKIP {
                        return Ok(false);
                    }
                    if action == MISSING_SOURCE_ACTION_ERROR {
                        return Err(format!("Missing source: {source_str}"));
                    }
                }
                self.install_parsed_source(&parsed, scope).await.map(|_| true)
            };

            match &parsed {
                ParsedSource::Npm(npm) => {
                    let installed_path = self.get_npm_install_path(npm, scope);
                    let needs_install = !Path::new(&installed_path).exists()
                        || (npm.pinned && !self.installed_npm_matches_pinned_version(npm, &installed_path));
                    if needs_install {
                        let installed = install_missing().await?;
                        if !installed {
                            continue;
                        }
                    }
                    let mut metadata = metadata;
                    metadata.base_dir = Some(installed_path.clone());
                    self.collect_package_resources(&installed_path, accumulator, filter.as_ref(), metadata);
                }
                ParsedSource::Git(git) => {
                    let installed_path = self.validated_git_install_path(git, scope)?;
                    if !Path::new(&installed_path).exists()
                        || !self.git_install_is_ready(git, &installed_path).await
                    {
                        let installed = install_missing().await?;
                        if !installed {
                            continue;
                        }
                    } else if scope == SOURCE_SCOPE_TEMPORARY && !git.pinned && !is_offline_mode_enabled() {
                        self.refresh_temporary_git_source(git, &source_str).await;
                    }
                    let mut metadata = metadata;
                    metadata.base_dir = Some(installed_path.clone());
                    self.collect_package_resources(&installed_path, accumulator, filter.as_ref(), metadata);
                }
                ParsedSource::Local(_) => {}
            }
        }
        Ok(())
    }

    fn resolve_local_extension_source(
        &self,
        source: &LocalSource,
        accumulator: &mut ResourceAccumulator,
        filter: Option<&PackageFilter>,
        mut metadata: PathMetadata,
        base_dir: &str,
    ) {
        let resolved = self.resolve_path_from_base(&source.path, base_dir);
        if !Path::new(&resolved).exists() {
            return;
        }

        let Ok(stats) = std::fs::metadata(&resolved) else {
            return;
        };
        if stats.is_file() {
            metadata.base_dir = Some(parent_path(&resolved));
            add_resource(&mut accumulator.extensions, &resolved, metadata, true);
            return;
        }
        if stats.is_dir() {
            metadata.base_dir = Some(resolved.clone());
            let resources = self.collect_package_resources(&resolved, accumulator, filter, metadata.clone());
            if !resources {
                add_resource(&mut accumulator.extensions, &resolved, metadata, true);
            }
        }
    }

    async fn install_parsed_source(&self, parsed: &ParsedSource, scope: &str) -> Result<(), String> {
        match parsed {
            ParsedSource::Npm(npm) => {
                self.install_npm(npm, scope, scope == SOURCE_SCOPE_TEMPORARY).await?;
                Ok(())
            }
            ParsedSource::Git(git) => {
                self.install_git(git, scope).await?;
                Ok(())
            }
            ParsedSource::Local(_) => Ok(()),
        }
    }

    fn get_source_match_key_for_input(&self, source: &str) -> String {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => format!("npm:{}", parsed.name),
            ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
            ParsedSource::Local(parsed) => format!("local:{}", self.resolve_path(&parsed.path)),
        }
    }

    fn get_source_match_key_for_settings(&self, source: &str, scope: &str) -> String {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => format!("npm:{}", parsed.name),
            ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
            ParsedSource::Local(parsed) => {
                let base_dir = self.get_base_dir_for_scope(scope);
                format!("local:{}", self.resolve_path_from_base(&parsed.path, &base_dir))
            }
        }
    }

    fn build_no_matching_package_message(&self, source: &str, configured_packages: &[Value]) -> String {
        match self.find_suggested_configured_source(source, configured_packages) {
            None => format!("No matching package found for {source}"),
            Some(suggestion) => format!("No matching package found for {source}. Did you mean {suggestion}?"),
        }
    }

    fn find_suggested_configured_source(&self, source: &str, configured_packages: &[Value]) -> Option<String> {
        let trimmed_source = source.trim();
        let mut suggestions: Vec<String> = Vec::new();

        for pkg in configured_packages {
            let source_str = package_source_string(pkg);
            match self.parse_source(&source_str) {
                ParsedSource::Npm(parsed) => {
                    if trimmed_source == parsed.name || trimmed_source == parsed.spec {
                        suggestions.push(source_str);
                    }
                }
                ParsedSource::Git(parsed) => {
                    let shorthand = format!("{}/{}", parsed.host, parsed.path);
                    let shorthand_with_ref = parsed
                    .reference
                    .as_ref()
                    .map(|reference| format!("{shorthand}@{reference}"));
                    if trimmed_source == shorthand
                        || shorthand_with_ref.as_deref() == Some(trimmed_source)
                    {
                        suggestions.push(source_str);
                    }
                }
                ParsedSource::Local(_) => {}
            }
        }

        suggestions.into_iter().next()
    }

    fn package_sources_match(&self, existing: &Value, input_source: &str, scope: &str) -> bool {
        let left = self.get_source_match_key_for_settings(&package_source_string(existing), scope);
        let right = self.get_source_match_key_for_input(input_source);
        left == right
    }

    fn normalize_package_source_for_settings(&self, source: &str, scope: &str) -> String {
        let parsed = self.parse_source(source);
        let ParsedSource::Local(local_source) = parsed else {
            return source.to_string();
        };
        let base_dir = self.get_base_dir_for_scope(scope);
        let resolved = self.resolve_path(&local_source.path);
        let rel = relative_path(&base_dir, &resolved);
        if rel.is_empty() {
            ".".to_string()
        } else {
            rel
        }
    }

    fn parse_source(&self, source: &str) -> ParsedSource {
        if let Some(rest) = source.strip_prefix("npm:") {
            let spec = rest.trim().to_string();
            let (name, version) = parse_npm_spec(&spec);
            return ParsedSource::Npm(NpmSource {
                spec,
                name,
                pinned: version.is_some(),
            });
        }

        if is_local_path(source) {
            return ParsedSource::Local(LocalSource {
                path: source.to_string(),
            });
        }
        if let Some(git_parsed) = parse_git_url(source) {
            return ParsedSource::Git(git_parsed);
        }

        ParsedSource::Local(LocalSource {
            path: source.to_string(),
        })
    }

    fn installed_npm_matches_pinned_version(&self, source: &NpmSource, installed_path: &str) -> bool {
        let Some(installed_version) = self.get_installed_npm_version(installed_path) else {
            return false;
        };

        let (_, pinned_version) = parse_npm_spec(&source.spec);
        let Some(pinned_version) = pinned_version else {
            return true;
        };

        installed_version == pinned_version
    }

    async fn npm_has_available_update(&self, source: &NpmSource, installed_path: &str) -> bool {
        if is_offline_mode_enabled() {
            return false;
        }

        let Some(installed_version) = self.get_installed_npm_version(installed_path) else {
            return false;
        };

        match self.get_latest_npm_version(&source.name).await {
            Ok(latest_version) => latest_version != installed_version,
            Err(_) => false,
        }
    }

    fn get_installed_npm_version(&self, installed_path: &str) -> Option<String> {
        let package_json_path = join_path(installed_path, "package.json");
        if !Path::new(&package_json_path).exists() {
            return None;
        }
        let content = std::fs::read_to_string(&package_json_path).ok()?;
        let pkg: Value = serde_json::from_str(&content).ok()?;
        pkg.get("version").and_then(Value::as_str).map(str::to_string)
    }

    async fn get_latest_npm_version(&self, package_name: &str) -> Result<String, String> {
        let npm_command = self.get_npm_command()?;
        let mut args = npm_command.args.clone();
        args.extend([
            "view".to_string(),
            package_name.to_string(),
            "version".to_string(),
            "--json".to_string(),
        ]);
        let stdout = self
            .run_command_capture(
                &npm_command.command,
                &args,
                Some(&self.cwd),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await?;
        let raw = stdout.trim();
        if raw.is_empty() {
            return Err("Empty response from npm view".to_string());
        }
        serde_json::from_str::<Value>(raw)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .ok_or_else(|| format!("Unexpected npm view response: {raw}"))
    }

    async fn git_has_available_update(&self, installed_path: &str) -> bool {
        if is_offline_mode_enabled() {
            return false;
        }

        let Ok(local_head) = self
            .run_command_capture(
                "git",
                &["rev-parse".to_string(), "HEAD".to_string()],
                Some(installed_path),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await
        else {
            return false;
        };
        match self.get_remote_git_head(installed_path).await {
            Ok(remote_head) => local_head.trim() != remote_head.trim(),
            Err(_) => false,
        }
    }

    async fn get_remote_git_head(&self, installed_path: &str) -> Result<String, String> {
        if let Some(upstream_ref) = self.get_git_upstream_ref(installed_path).await {
            let remote_head = self
                .run_git_remote_command(
                    installed_path,
                    &["ls-remote".to_string(), "origin".to_string(), upstream_ref],
                )
                .await?;
            if let Some(captured) = first_hex40_match(&remote_head) {
                return Ok(captured);
            }
        }

        let remote_head = self
            .run_git_remote_command(
                installed_path,
                &["ls-remote".to_string(), "origin".to_string(), "HEAD".to_string()],
            )
            .await?;
        match head_hex40_match(&remote_head) {
            Some(captured) => Ok(captured),
            None => Err("Failed to determine remote HEAD".to_string()),
        }
    }

    async fn get_local_git_update_target(
        &self,
        installed_path: &str,
    ) -> Result<(String, String, Vec<String>), String> {
        let upstream = self
            .run_command_capture(
                "git",
                &[
                    "rev-parse".to_string(),
                    "--abbrev-ref".to_string(),
                    "@{upstream}".to_string(),
                ],
                Some(installed_path),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await;
        if let Ok(upstream) = upstream {
            let trimmed_upstream = upstream.trim().to_string();
            if trimmed_upstream.starts_with("origin/") {
                let branch = trimmed_upstream["origin/".len()..].to_string();
                if !branch.is_empty() {
                    let head = self
                        .run_command_capture(
                            "git",
                            &["rev-parse".to_string(), "@{upstream}".to_string()],
                            Some(installed_path),
                            Some(NETWORK_TIMEOUT_MS),
                            None,
                        )
                        .await?;
                    return Ok((
                        "@{upstream}".to_string(),
                        head,
                        vec![
                            "fetch".to_string(),
                            "--prune".to_string(),
                            "--no-tags".to_string(),
                            "origin".to_string(),
                            format!("+refs/heads/{branch}:refs/remotes/origin/{branch}"),
                        ],
                    ));
                }
            }
        }

        let _ = self
            .run_command(
                "git",
                &[
                    "remote".to_string(),
                    "set-head".to_string(),
                    "origin".to_string(),
                    "-a".to_string(),
                ],
                Some(installed_path),
            )
            .await;
        let head = self
            .run_command_capture(
                "git",
                &["rev-parse".to_string(), "origin/HEAD".to_string()],
                Some(installed_path),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await?;
        let origin_head_ref = self
            .run_command_capture(
                "git",
                &[
                    "symbolic-ref".to_string(),
                    "refs/remotes/origin/HEAD".to_string(),
                ],
                Some(installed_path),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await
            .unwrap_or_default();
        let branch = origin_head_ref
            .trim()
            .strip_prefix("refs/remotes/origin/")
            .unwrap_or("")
            .to_string();
        if !branch.is_empty() {
            return Ok((
                "origin/HEAD".to_string(),
                head,
                vec![
                    "fetch".to_string(),
                    "--prune".to_string(),
                    "--no-tags".to_string(),
                    "origin".to_string(),
                    format!("+refs/heads/{branch}:refs/remotes/origin/{branch}"),
                ],
            ));
        }
        Ok((
            "origin/HEAD".to_string(),
            head,
            vec![
                "fetch".to_string(),
                "--prune".to_string(),
                "--no-tags".to_string(),
                "origin".to_string(),
                "+HEAD:refs/remotes/origin/HEAD".to_string(),
            ],
        ))
    }

    async fn get_git_upstream_ref(&self, installed_path: &str) -> Option<String> {
        let upstream = self
            .run_command_capture(
                "git",
                &[
                    "rev-parse".to_string(),
                    "--abbrev-ref".to_string(),
                    "@{upstream}".to_string(),
                ],
                Some(installed_path),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await
            .ok()?;
        let trimmed = upstream.trim().to_string();
        if !trimmed.starts_with("origin/") {
            return None;
        }
        let branch = trimmed["origin/".len()..].to_string();
        if branch.is_empty() {
            None
        } else {
            Some(format!("refs/heads/{branch}"))
        }
    }

    async fn run_git_remote_command(&self, installed_path: &str, args: &[String]) -> Result<String, String> {
        self.run_command_capture(
            "git",
            args,
            Some(installed_path),
            Some(NETWORK_TIMEOUT_MS),
            Some(vec![("GIT_TERMINAL_PROMPT".to_string(), "0".to_string())]),
        )
        .await
    }

    async fn run_with_concurrency<T, F>(&self, tasks: Vec<F>, limit: usize) -> Vec<T>
    where
        T: Send + 'static,
        F: FnOnce() -> futures::future::BoxFuture<'static, T> + Send + 'static,
    {
        if tasks.is_empty() {
            return Vec::new();
        }

        let total = tasks.len();
        let results: Arc<std::sync::Mutex<Vec<Option<T>>>> =
            Arc::new(std::sync::Mutex::new((0..total).map(|_| None).collect()));
        let queue: Arc<std::sync::Mutex<std::collections::VecDeque<(usize, F)>>> =
            Arc::new(std::sync::Mutex::new(tasks.into_iter().enumerate().collect()));
        let worker_count = std::cmp::max(1, std::cmp::min(limit, total));

        let mut workers = Vec::new();
        for _ in 0..worker_count {
            let queue = Arc::clone(&queue);
            let results = Arc::clone(&results);
            workers.push(tokio::spawn(async move {
                loop {
                    let next = queue.lock().expect("queue").pop_front();
                    let Some((index, task)) = next else {
                        return;
                    };
                    let value = task().await;
                    results.lock().expect("results")[index] = Some(value);
                }
            }));
        }
        for worker in workers {
            let _ = worker.await;
        }

        let mut collected = results.lock().expect("results");
        collected.drain(..).map(|value| value.expect("task result")).collect()
    }

    /// Get a unique identity for a package, ignoring version/ref.
    fn get_package_identity(&self, source: &str, scope: Option<&str>) -> String {
        match self.parse_source(source) {
            ParsedSource::Npm(parsed) => format!("npm:{}", parsed.name),
            // Use host/path for identity to normalize SSH and HTTPS
            ParsedSource::Git(parsed) => format!("git:{}/{}", parsed.host, parsed.path),
            ParsedSource::Local(parsed) => match scope {
                Some(scope) => {
                    let base_dir = self.get_base_dir_for_scope(scope);
                    format!("local:{}", self.resolve_path_from_base(&parsed.path, &base_dir))
                }
                None => format!("local:{}", self.resolve_path(&parsed.path)),
            },
        }
    }

    /// Dedupe packages: if the same package identity appears in both global and
    /// project, keep only the project one (project wins).
    fn dedupe_packages(&self, packages: &[(Value, String)]) -> Vec<(Value, String)> {
        let mut seen: Vec<(String, (Value, String))> = Vec::new();

        for entry in packages {
            let source_str = package_source_string(&entry.0);
            let identity = self.get_package_identity(&source_str, Some(&entry.1));

            match seen.iter_mut().find(|(key, _)| key == &identity) {
                None => seen.push((identity, entry.clone())),
                Some((_, existing)) => {
                    if entry.1 == SOURCE_SCOPE_PROJECT && existing.1 == SOURCE_SCOPE_USER {
                        *existing = entry.clone();
                    }
                }
            }
        }

        seen.into_iter().map(|(_, entry)| entry).collect()
    }

    fn get_npm_command(&self) -> Result<NpmCommand, String> {
        let configured_command = self.settings().get_npm_command();
        let Some(configured_command) = configured_command else {
            return Ok(NpmCommand {
                command: "npm".to_string(),
                args: Vec::new(),
            });
        };
        if configured_command.is_empty() {
            return Ok(NpmCommand {
                command: "npm".to_string(),
                args: Vec::new(),
            });
        }
        let mut iter = configured_command.into_iter();
        let command = iter.next().unwrap_or_default();
        if command.is_empty() {
            return Err("Invalid npmCommand: first array entry must be a non-empty command".to_string());
        }
        Ok(NpmCommand {
            command,
            args: iter.collect(),
        })
    }

    async fn run_npm_command(&self, args: &[String], cwd: Option<&str>) -> Result<(), String> {
        let npm_command = self.get_npm_command()?;
        let mut full: Vec<String> = npm_command.args.clone();
        full.extend(args.iter().cloned());
        self.run_command(&npm_command.command, &full, cwd).await
    }

    fn get_git_dependency_install_args(&self) -> Vec<String> {
        let configured_command = self.settings().get_npm_command();
        match configured_command {
            Some(command) if !command.is_empty() => vec!["install".to_string()],
            _ => vec!["install".to_string(), "--omit=dev".to_string()],
        }
    }

    fn run_npm_command_sync(&self, args: &[String]) -> Result<String, String> {
        let npm_command = self.get_npm_command()?;
        let mut full: Vec<String> = npm_command.args.clone();
        full.extend(args.iter().cloned());
        self.run_command_sync(&npm_command.command, &full)
    }

    async fn install_npm(&self, source: &NpmSource, scope: &str, temporary: bool) -> Result<(), String> {
        if scope == SOURCE_SCOPE_USER && !temporary {
            return self
                .run_npm_command(&["install".to_string(), "-g".to_string(), source.spec.clone()], None)
                .await;
        }
        let install_root = self.get_npm_install_root(scope, temporary);
        self.ensure_npm_project(&install_root);
        self.run_npm_command(
            &[
                "install".to_string(),
                source.spec.clone(),
                "--prefix".to_string(),
                install_root,
            ],
            None,
        )
        .await
    }

    async fn uninstall_npm(&self, source: &NpmSource, scope: &str) -> Result<(), String> {
        if scope == SOURCE_SCOPE_USER {
            return self
                .run_npm_command(&["uninstall".to_string(), "-g".to_string(), source.name.clone()], None)
                .await;
        }
        let install_root = self.get_npm_install_root(scope, false);
        if !Path::new(&install_root).exists() {
            return Ok(());
        }
        self.run_npm_command(
            &[
                "uninstall".to_string(),
                source.name.clone(),
                "--prefix".to_string(),
                install_root,
            ],
            None,
        )
        .await
    }

    fn git_install_completion_path(target_dir: &str) -> String {
        join_path(&join_path(target_dir, ".git"), "prime-agent-install.json")
    }

    fn git_package_json_hash(target_dir: &str) -> Option<String> {
        let bytes = std::fs::read(join_path(target_dir, "package.json")).ok()?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Some(hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
    }

    fn git_dependencies_ready(target_dir: &str) -> bool {
        let package_json_path = join_path(target_dir, "package.json");
        if !Path::new(&package_json_path).exists() {
            return true;
        }
        let Ok(package) = std::fs::read_to_string(package_json_path)
            .map_err(|error| error.to_string())
            .and_then(|text| serde_json::from_str::<Value>(&text).map_err(|error| error.to_string()))
        else {
            return false;
        };
        for section in ["dependencies"] {
            if let Some(dependencies) = package.get(section).and_then(Value::as_object) {
                for name in dependencies.keys() {
                    let dependency_manifest = Path::new(target_dir)
                        .join("node_modules")
                        .join(name)
                        .join("package.json");
                    if !dependency_manifest.is_file() {
                        return false;
                    }
                }
            }
        }
        true
    }

    async fn git_install_is_ready(&self, source: &GitSource, target_dir: &str) -> bool {
        let completion_path = Self::git_install_completion_path(target_dir);
        let Some(completion) = std::fs::read_to_string(completion_path)
            .ok()
            .and_then(|text| serde_json::from_str::<GitInstallCompletion>(&text).ok())
        else {
            return false;
        };
        if completion.repo != source.repo
            || completion.reference != source.reference
            || completion.package_json_sha256 != Self::git_package_json_hash(target_dir)
            || !Self::git_dependencies_ready(target_dir)
        {
            return false;
        }
        let Ok(head) = self
            .run_command_capture(
                "git",
                &["rev-parse".to_string(), "HEAD".to_string()],
                Some(target_dir),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await
        else {
            return false;
        };
        if head.trim() != completion.head {
            return false;
        }
        if let Some(reference) = &source.reference {
            let Ok(reference_head) = self
                .run_command_capture(
                    "git",
                    &[
                        "rev-parse".to_string(),
                        "--verify".to_string(),
                        "--end-of-options".to_string(),
                        format!("{reference}^{{commit}}"),
                    ],
                    Some(target_dir),
                    Some(NETWORK_TIMEOUT_MS),
                    None,
                )
                .await
            else {
                return false;
            };
            if reference_head.trim() != head.trim() {
                return false;
            }
        }
        true
    }

    async fn write_git_install_completion(
        &self,
        source: &GitSource,
        target_dir: &str,
    ) -> Result<(), String> {
        let head = self
            .run_command_capture(
                "git",
                &["rev-parse".to_string(), "HEAD".to_string()],
                Some(target_dir),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await?;
        if !Self::git_dependencies_ready(target_dir) {
            return Err("Git package dependencies are incomplete after installation".to_string());
        }
        let completion = GitInstallCompletion {
            repo: source.repo.clone(),
            reference: source.reference.clone(),
            head: head.trim().to_string(),
            package_json_sha256: Self::git_package_json_hash(target_dir),
        };
        let text = serde_json::to_string_pretty(&completion).map_err(|error| error.to_string())?;
        std::fs::write(Self::git_install_completion_path(target_dir), text)
            .map_err(|error| error.to_string())
    }

    fn remove_managed_git_target(target_dir: &str) -> Result<(), String> {
        let Ok(metadata) = std::fs::symlink_metadata(target_dir) else {
            return Ok(());
        };
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            std::fs::remove_dir_all(target_dir).map_err(|error| error.to_string())
        } else {
            std::fs::remove_file(target_dir).map_err(|error| error.to_string())
        }
    }

    async fn install_git(&self, source: &GitSource, scope: &str) -> Result<(), String> {
        let target_dir = self.validated_git_install_path(source, scope)?;
        if Path::new(&target_dir).exists() {
            if self.git_install_is_ready(source, &target_dir).await {
                return Ok(());
            }
            Self::remove_managed_git_target(&target_dir)?;
        }
        let git_root = self.get_git_install_root(scope);
        if let Some(git_root) = git_root {
            self.ensure_git_ignore(&git_root);
        }
        let _ = std::fs::create_dir_all(parent_path(&target_dir));

        let result = async {
            self.run_command(
                "git",
                &["clone".to_string(), source.repo.clone(), target_dir.clone()],
                None,
            )
            .await?;
            if let Some(reference) = &source.reference {
                self.run_command(
                    "git",
                    &["checkout".to_string(), reference.clone()],
                    Some(&target_dir),
                )
                .await?;
            }
            let package_json_path = join_path(&target_dir, "package.json");
            if Path::new(&package_json_path).exists() {
                let args = self.get_git_dependency_install_args();
                self.run_npm_command(&args, Some(&target_dir)).await?;
            }
            self.write_git_install_completion(source, &target_dir).await
        }
        .await;
        if result.is_err() {
            Self::remove_managed_git_target(&target_dir)?;
        }
        result
    }

    async fn update_git(&self, source: &GitSource, scope: &str) -> Result<(), String> {
        let target_dir = self.validated_git_install_path(source, scope)?;
        if Path::new(&target_dir).exists() && !self.git_install_is_ready(source, &target_dir).await {
            Self::remove_managed_git_target(&target_dir)?;
        }
        if !Path::new(&target_dir).exists() {
            return self.install_git(source, scope).await;
        }

        let (target_ref, target_head, fetch_args) = self.get_local_git_update_target(&target_dir).await?;

        self.run_command("git", &fetch_args, Some(&target_dir)).await?;

        let local_head = self
            .run_command_capture(
                "git",
                &["rev-parse".to_string(), "HEAD".to_string()],
                Some(&target_dir),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await?;
        let refreshed_target_head = self
            .run_command_capture(
                "git",
                &["rev-parse".to_string(), target_ref.clone()],
                Some(&target_dir),
                Some(NETWORK_TIMEOUT_MS),
                None,
            )
            .await?;
        let _ = target_head;
        if local_head.trim() == refreshed_target_head.trim() {
            return Ok(());
        }

        let _ = std::fs::remove_file(Self::git_install_completion_path(&target_dir));
        self.run_command(
            "git",
            &["reset".to_string(), "--hard".to_string(), target_ref],
            Some(&target_dir),
        )
        .await?;

        // Extension checkouts must be pristine after an update.
        self.run_command(
            "git",
            &["clean".to_string(), "-fdx".to_string()],
            Some(&target_dir),
        )
        .await?;

        let package_json_path = join_path(&target_dir, "package.json");
        if Path::new(&package_json_path).exists() {
            let args = self.get_git_dependency_install_args();
            self.run_npm_command(&args, Some(&target_dir)).await?;
        }
        self.write_git_install_completion(source, &target_dir).await
    }

    async fn refresh_temporary_git_source(&self, source: &GitSource, source_str: &str) {
        if is_offline_mode_enabled() {
            return;
        }
        let message = format!("Refreshing {source_str}...");
        // Keep the cached temporary checkout if refresh fails.
        let _ = self
            .with_progress(
                "pull",
                source_str,
                &message,
                Box::pin(async { self.update_git(source, SOURCE_SCOPE_TEMPORARY).await }),
            )
            .await;
    }

    fn remove_git(&self, source: &GitSource, scope: &str) -> Result<(), String> {
        let target_dir = self.validated_git_install_path(source, scope)?;
        if !Path::new(&target_dir).exists() {
            return Ok(());
        }
        Self::remove_managed_git_target(&target_dir)?;
        self.prune_empty_git_parents(&target_dir, self.get_git_install_root(scope).as_deref());
        Ok(())
    }

    fn prune_empty_git_parents(&self, target_dir: &str, install_root: Option<&str>) {
        let Some(install_root) = install_root else {
            return;
        };
        let resolved_root = absolute_path(install_root);
        let mut current = parent_path(target_dir);
        while current.starts_with(&resolved_root) && current != resolved_root {
            if !Path::new(&current).exists() {
                current = parent_path(&current);
                continue;
            }
            let entries = std::fs::read_dir(&current).map(|entries| entries.count()).unwrap_or(0);
            if entries > 0 {
                break;
            }
            if std::fs::remove_dir_all(&current).is_err() {
                break;
            }
            current = parent_path(&current);
        }
    }

    fn ensure_npm_project(&self, install_root: &str) {
        if !Path::new(install_root).exists() {
            let _ = std::fs::create_dir_all(install_root);
        }
        self.ensure_git_ignore(install_root);
        let package_json_path = join_path(install_root, "package.json");
        if !Path::new(&package_json_path).exists() {
            // `JSON.stringify({ name: "pi-extensions", private: true }, null, 2)`
            let pkg_json = serde_json::json!({"name": "pi-extensions", "private": true});
            let text = serde_json::to_string_pretty(&pkg_json).unwrap_or_default();
            let _ = std::fs::write(&package_json_path, text);
        }
    }

    fn ensure_git_ignore(&self, dir: &str) {
        if !Path::new(dir).exists() {
            let _ = std::fs::create_dir_all(dir);
        }
        let ignore_path = join_path(dir, ".gitignore");
        if !Path::new(&ignore_path).exists() {
            let _ = std::fs::write(&ignore_path, "*\n!.gitignore\n");
        }
    }

    fn get_npm_install_root(&self, scope: &str, temporary: bool) -> String {
        if temporary {
            return self.get_temporary_dir("npm", None);
        }

        if scope == SOURCE_SCOPE_PROJECT {
            return join_path(&join_path(&self.cwd, CONFIG_DIR_NAME), "npm");
        }
        join_path(&self.get_global_npm_root(), "..")
    }

    fn get_global_npm_root(&self) -> String {
        // Cached per npm command key; see `getNpmInstallRoot` in the TypeScript.
        let Ok(npm_command) = self.get_npm_command() else {
            return String::new();
        };
        let mut command_key_parts = vec![npm_command.command.clone()];
        command_key_parts.extend(npm_command.args.iter().cloned());
        let command_key = command_key_parts.join("\0");
        {
            let cached_root = self.global_npm_root.lock().expect("global npm root");
            let cached_key = self.global_npm_root_command_key.lock().expect("npm root key");
            if let Some(root) = cached_root.as_ref() {
                if cached_key.as_deref() == Some(command_key.as_str()) {
                    return root.clone();
                }
            }
        }
        let is_bun_package_manager = npm_command.command == "bun";
        let root = if is_bun_package_manager {
            let bin_dir = self
                .run_npm_command_sync(&["pm".to_string(), "bin".to_string(), "-g".to_string()])
                .unwrap_or_default()
                .trim()
                .to_string();
            join_path(&join_path(&parent_path(&bin_dir), "install"), "global/node_modules")
        } else {
            self.run_npm_command_sync(&["root".to_string(), "-g".to_string()])
                .unwrap_or_default()
                .trim()
                .to_string()
        };
        // `this.globalNpmRoot` is cached alongside the command key it was resolved for.
        *self.global_npm_root.lock().expect("global npm root") = Some(root.clone());
        *self.global_npm_root_command_key.lock().expect("npm root key") = Some(command_key);
        root
    }

    fn get_npm_install_path(&self, source: &NpmSource, scope: &str) -> String {
        if scope == SOURCE_SCOPE_TEMPORARY {
            return join_path(&join_path(&self.get_temporary_dir("npm", None), "node_modules"), &source.name);
        }
        if scope == SOURCE_SCOPE_PROJECT {
            let project_npm = join_path(&join_path(&self.cwd, CONFIG_DIR_NAME), "npm");
            return join_path(&join_path(&project_npm, "node_modules"), &source.name);
        }
        join_path(&self.get_global_npm_root(), &source.name)
    }

    fn get_git_install_path(&self, source: &GitSource, scope: &str) -> String {
        if scope == SOURCE_SCOPE_TEMPORARY {
            return self.get_temporary_dir(&format!("git-{}", source.host), Some(&source.path));
        }
        if scope == SOURCE_SCOPE_PROJECT {
            let project_git = join_path(&join_path(&self.cwd, CONFIG_DIR_NAME), "git");
            return join_path(&join_path(&project_git, &source.host), &source.path);
        }
        let user_git = join_path(&self.agent_dir, "git");
        join_path(&join_path(&user_git, &source.host), &source.path)
    }

    fn get_git_install_containment_anchor(&self, scope: &str) -> String {
        if scope == SOURCE_SCOPE_TEMPORARY {
            return join_path(&std::env::temp_dir().to_string_lossy(), "pi-extensions");
        }
        if scope == SOURCE_SCOPE_PROJECT {
            return self.cwd.clone();
        }
        self.agent_dir.clone()
    }

    fn get_git_install_containment_root(&self, source: &GitSource, scope: &str) -> String {
        if scope != SOURCE_SCOPE_TEMPORARY {
            return self.get_git_install_root(scope).unwrap_or_default();
        }
        let prefix = format!("git-{}", source.host);
        let hash = sha256_hex_8(&format!("{prefix}-{}", source.path));
        let temp_root = std::env::temp_dir().to_string_lossy().to_string();
        join_path(
            &join_path(&join_path(&temp_root, "pi-extensions"), &prefix),
            &hash,
        )
    }

    fn validated_git_install_path(&self, source: &GitSource, scope: &str) -> Result<String, String> {
        validate_git_source_path_parts(source)?;
        let anchor = self.get_git_install_containment_anchor(scope);
        let root = self.get_git_install_containment_root(source, scope);
        if anchor.is_empty() || root.is_empty() {
            return Err("Unsafe git package scope".to_string());
        }
        let target = self.get_git_install_path(source, scope);
        let lexical_anchor = lexical_absolute_path(Path::new(&anchor))?;
        let lexical_root = lexical_absolute_path(Path::new(&root))?;
        let lexical_target = lexical_absolute_path(Path::new(&target))?;
        if !lexical_root.starts_with(&lexical_anchor)
            || !lexical_target.starts_with(&lexical_root)
            || lexical_target == lexical_root
        {
            return Err(format!("Unsafe git package path: {}/{}", source.host, source.path));
        }
        let canonical_anchor = canonicalize_with_missing_components(&lexical_anchor)?;
        let canonical_root = canonicalize_with_missing_components(&lexical_root)?;
        let canonical_target = canonicalize_with_missing_components(&lexical_target)?;
        if !canonical_root.starts_with(&canonical_anchor)
            || !canonical_target.starts_with(&canonical_root)
            || canonical_target == canonical_root
        {
            return Err(format!("Git package path escapes its install root: {}/{}", source.host, source.path));
        }
        Ok(target)
    }

    fn get_git_install_root(&self, scope: &str) -> Option<String> {
        if scope == SOURCE_SCOPE_TEMPORARY {
            return None;
        }
        if scope == SOURCE_SCOPE_PROJECT {
            return Some(join_path(&join_path(&self.cwd, CONFIG_DIR_NAME), "git"));
        }
        Some(join_path(&self.agent_dir, "git"))
    }

    fn get_temporary_dir(&self, prefix: &str, suffix: Option<&str>) -> String {
        let hash = sha256_hex_8(&format!("{prefix}-{}", suffix.unwrap_or("")));
        let temp_root = std::env::temp_dir().to_string_lossy().to_string();
        let base = join_path(&join_path(&temp_root, "pi-extensions"), prefix);
        join_path(&join_path(&base, &hash), suffix.unwrap_or(""))
    }

    fn get_base_dir_for_scope(&self, scope: &str) -> String {
        if scope == SOURCE_SCOPE_PROJECT {
            return join_path(&self.cwd, CONFIG_DIR_NAME);
        }
        if scope == SOURCE_SCOPE_USER {
            return self.agent_dir.clone();
        }
        self.cwd.clone()
    }

    fn resolve_path(&self, input: &str) -> String {
        let trimmed = input.trim();
        if trimmed == "~" {
            return get_home_dir();
        }
        if let Some(rest) = trimmed.strip_prefix("~/") {
            return join_path(&get_home_dir(), rest);
        }
        if let Some(rest) = trimmed.strip_prefix('~') {
            return join_path(&get_home_dir(), rest);
        }
        absolute_join(&self.cwd, trimmed)
    }

    fn resolve_path_from_base(&self, input: &str, base_dir: &str) -> String {
        let trimmed = input.trim();
        if trimmed == "~" {
            return get_home_dir();
        }
        if let Some(rest) = trimmed.strip_prefix("~/") {
            return join_path(&get_home_dir(), rest);
        }
        if let Some(rest) = trimmed.strip_prefix('~') {
            return join_path(&get_home_dir(), rest);
        }
        absolute_join(base_dir, trimmed)
    }
}

impl DefaultPackageManager {
    fn collect_package_resources(
        &self,
        package_root: &str,
        accumulator: &mut ResourceAccumulator,
        filter: Option<&PackageFilter>,
        metadata: PathMetadata,
    ) -> bool {
        if let Some(filter) = filter {
            for resource_type in RESOURCE_TYPES {
                let patterns = filter.get(resource_type);
                match patterns {
                    Some(patterns) => {
                        self.apply_package_filter(package_root, patterns, resource_type, accumulator, metadata.clone())
                    }
                    None => {
                        self.collect_default_resources(package_root, resource_type, accumulator, metadata.clone())
                    }
                }
            }
            return true;
        }

        if let Some(manifest) = self.read_pi_manifest(package_root) {
            for resource_type in RESOURCE_TYPES {
                let entries = manifest.get(resource_type);
                self.add_manifest_entries(entries, package_root, resource_type, accumulator, metadata.clone());
            }
            return true;
        }

        let mut has_any_dir = false;
        for resource_type in RESOURCE_TYPES {
            let dir = join_path(package_root, resource_type);
            if Path::new(&dir).exists() {
                let files = collect_resource_files(&dir, resource_type);
                for file in files {
                    add_resource(target_map(accumulator, resource_type), &file, metadata.clone(), true);
                }
                has_any_dir = true;
            }
        }
        has_any_dir
    }

    fn collect_default_resources(
        &self,
        package_root: &str,
        resource_type: &str,
        accumulator: &mut ResourceAccumulator,
        metadata: PathMetadata,
    ) {
        let manifest = self.read_pi_manifest(package_root);
        let entries = manifest.as_ref().and_then(|manifest| manifest.get(resource_type));
        if let Some(entries) = entries {
            self.add_manifest_entries(Some(entries), package_root, resource_type, accumulator, metadata);
            return;
        }
        let dir = join_path(package_root, resource_type);
        if Path::new(&dir).exists() {
            let files = collect_resource_files(&dir, resource_type);
            for file in files {
                add_resource(target_map(accumulator, resource_type), &file, metadata.clone(), true);
            }
        }
    }

    fn apply_package_filter(
        &self,
        package_root: &str,
        user_patterns: &[String],
        resource_type: &str,
        accumulator: &mut ResourceAccumulator,
        metadata: PathMetadata,
    ) {
        let (all_files, _) = self.collect_manifest_files(package_root, resource_type);

        // An explicit empty list disables this resource type.
        if user_patterns.is_empty() {
            for file in &all_files {
                add_resource(target_map(accumulator, resource_type), file, metadata.clone(), false);
            }
            return;
        }
        let enabled_by_user = apply_patterns(&all_files, user_patterns, package_root);

        for file in &all_files {
            let enabled = enabled_by_user.contains(file);
            add_resource(target_map(accumulator, resource_type), file, metadata.clone(), enabled);
        }
    }

    /// Collect all files from a package for a resource type, applying manifest patterns.
    fn collect_manifest_files(&self, package_root: &str, resource_type: &str) -> (Vec<String>, HashSet<String>) {
        let manifest = self.read_pi_manifest(package_root);
        let entries = manifest.as_ref().and_then(|manifest| manifest.get(resource_type));
        if let Some(entries) = entries {
            if !entries.is_empty() {
                let all_files = self.collect_files_from_manifest_entries(entries, package_root, resource_type);
                let manifest_patterns = get_override_patterns(entries);
                let enabled_by_manifest = if !manifest_patterns.is_empty() {
                    apply_patterns(&all_files, &manifest_patterns, package_root)
                } else {
                    all_files.iter().cloned().collect()
                };
                let mut listed: Vec<String> = enabled_by_manifest.iter().cloned().collect();
                listed.sort();
                return (listed, enabled_by_manifest);
            }
        }

        let convention_dir = join_path(package_root, resource_type);
        if !Path::new(&convention_dir).exists() {
            return (Vec::new(), HashSet::new());
        }
        let all_files = collect_resource_files(&convention_dir, resource_type);
        let enabled: HashSet<String> = all_files.iter().cloned().collect();
        (all_files, enabled)
    }

    fn read_pi_manifest(&self, package_root: &str) -> Option<PiManifest> {
        let package_json_path = join_path(package_root, "package.json");
        if !Path::new(&package_json_path).exists() {
            return None;
        }

        let content = std::fs::read_to_string(&package_json_path).ok()?;
        let pkg: Value = serde_json::from_str(&content).ok()?;
        let pi = pkg.get("pi")?;
        if pi.is_null() {
            return None;
        }
        Some(parse_pi_manifest(pi))
    }

    fn add_manifest_entries(
        &self,
        entries: Option<&Vec<String>>,
        root: &str,
        resource_type: &str,
        accumulator: &mut ResourceAccumulator,
        metadata: PathMetadata,
    ) {
        let Some(entries) = entries else {
            return;
        };

        let all_files = self.collect_files_from_manifest_entries(entries, root, resource_type);
        let patterns = get_override_patterns(entries);
        let enabled_paths = apply_patterns(&all_files, &patterns, root);

        for file in &all_files {
            if enabled_paths.contains(file) {
                add_resource(target_map(accumulator, resource_type), file, metadata.clone(), true);
            }
        }
    }

    fn collect_files_from_manifest_entries(&self, entries: &[String], root: &str, resource_type: &str) -> Vec<String> {
        let source_entries: Vec<&String> = entries.iter().filter(|entry| !is_override_pattern(entry)).collect();
        let mut resolved: Vec<String> = Vec::new();
        for entry in source_entries {
            if !has_glob_pattern(entry) {
                resolved.push(absolute_join(root, entry));
                continue;
            }

            for matched in glob_sync(entry, root) {
                resolved.push(absolute_path(&matched));
            }
        }
        self.collect_files_from_paths(&resolved, resource_type)
    }

    fn resolve_local_entries(
        &self,
        entries: &[String],
        resource_type: &str,
        accumulator: &mut ResourceAccumulator,
        metadata: PathMetadata,
        base_dir: &str,
    ) {
        if entries.is_empty() {
            return;
        }
        let (plain, patterns) = split_patterns(entries);
        let resolved_plain: Vec<String> = plain
            .iter()
            .map(|entry| self.resolve_path_from_base(entry, base_dir))
            .collect();
        let all_files = self.collect_files_from_paths(&resolved_plain, resource_type);
        let enabled_paths = apply_patterns(&all_files, &patterns, base_dir);

        for file in &all_files {
            let enabled = enabled_paths.contains(file);
            add_resource(target_map(accumulator, resource_type), file, metadata.clone(), enabled);
        }
    }

    fn add_auto_discovered_resources(
        &self,
        accumulator: &mut ResourceAccumulator,
        global_settings: &Settings,
        project_settings: &Settings,
        global_base_dir: &str,
        project_base_dir: &str,
    ) {
        let user_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SOURCE_SCOPE_USER.to_string(),
            origin: "top-level".to_string(),
            base_dir: Some(global_base_dir.to_string()),
        };
        let project_metadata = PathMetadata {
            source: "auto".to_string(),
            scope: SOURCE_SCOPE_PROJECT.to_string(),
            origin: "top-level".to_string(),
            base_dir: Some(project_base_dir.to_string()),
        };

        let user_overrides: Vec<(String, Vec<String>)> = RESOURCE_TYPES
            .iter()
            .map(|resource_type| (resource_type.to_string(), string_list(global_settings, resource_type)))
            .collect();
        let project_overrides: Vec<(String, Vec<String>)> = RESOURCE_TYPES
            .iter()
            .map(|resource_type| (resource_type.to_string(), string_list(project_settings, resource_type)))
            .collect();
        let overrides_for = |overrides: &[(String, Vec<String>)], resource_type: &str| -> Vec<String> {
            overrides
                .iter()
                .find(|(key, _)| key == resource_type)
                .map(|(_, value)| value.clone())
                .unwrap_or_default()
        };

        let user_dirs: Vec<(String, String)> = RESOURCE_TYPES
            .iter()
            .map(|resource_type| (resource_type.to_string(), join_path(global_base_dir, resource_type)))
            .collect();
        let project_dirs: Vec<(String, String)> = RESOURCE_TYPES
            .iter()
            .map(|resource_type| (resource_type.to_string(), join_path(project_base_dir, resource_type)))
            .collect();
        let dir_for = |dirs: &[(String, String)], resource_type: &str| -> String {
            dirs.iter()
                .find(|(key, _)| key == resource_type)
                .map(|(_, value)| value.clone())
                .unwrap_or_default()
        };

        let user_agents_skills_dir = join_path(&join_path(&get_home_dir(), ".agents"), "skills");
        let project_agents_skill_dirs: Vec<String> = collect_ancestor_agents_skill_dirs(&self.cwd)
            .into_iter()
            .filter(|dir| absolute_path(dir) != absolute_path(&user_agents_skills_dir))
            .collect();

        let add_resources = |accumulator: &mut ResourceAccumulator,
                             resource_type: &str,
                             paths: Vec<String>,
                             metadata: PathMetadata,
                             overrides: Vec<String>,
                             base_dir: &str| {
            for path in paths {
                let enabled = is_enabled_by_overrides(&path, &overrides, base_dir);
                add_resource(target_map(accumulator, resource_type), &path, metadata.clone(), enabled);
            }
        };

        add_resources(
            accumulator,
            "extensions",
            collect_auto_extension_entries(&dir_for(&project_dirs, "extensions")),
            project_metadata.clone(),
            overrides_for(&project_overrides, "extensions"),
            project_base_dir,
        );
        let mut project_skills = collect_auto_skill_entries(&dir_for(&project_dirs, "skills"), "pi");
        for dir in &project_agents_skill_dirs {
            project_skills.extend(collect_auto_skill_entries(dir, "agents"));
        }
        add_resources(
            accumulator,
            "skills",
            project_skills,
            project_metadata.clone(),
            overrides_for(&project_overrides, "skills"),
            project_base_dir,
        );
        add_resources(
            accumulator,
            "prompts",
            collect_auto_prompt_entries(&dir_for(&project_dirs, "prompts")),
            project_metadata.clone(),
            overrides_for(&project_overrides, "prompts"),
            project_base_dir,
        );
        add_resources(
            accumulator,
            "themes",
            collect_auto_theme_entries(&dir_for(&project_dirs, "themes")),
            project_metadata,
            overrides_for(&project_overrides, "themes"),
            project_base_dir,
        );

        add_resources(
            accumulator,
            "extensions",
            collect_auto_extension_entries(&dir_for(&user_dirs, "extensions")),
            user_metadata.clone(),
            overrides_for(&user_overrides, "extensions"),
            global_base_dir,
        );
        let mut user_skills = collect_auto_skill_entries(&dir_for(&user_dirs, "skills"), "pi");
        user_skills.extend(collect_auto_skill_entries(&user_agents_skills_dir, "agents"));
        add_resources(
            accumulator,
            "skills",
            user_skills,
            user_metadata.clone(),
            overrides_for(&user_overrides, "skills"),
            global_base_dir,
        );

        if let Some(bundled_skills_dir) = &self.bundled_skills_dir {
            if self.settings().get_enable_builtin_skills() {
                let builtin_metadata = PathMetadata {
                    source: "builtin".to_string(),
                    scope: SOURCE_SCOPE_USER.to_string(),
                    origin: "top-level".to_string(),
                    base_dir: Some(bundled_skills_dir.clone()),
                };
                let builtin_entries = collect_auto_skill_entries(bundled_skills_dir, "pi");
                // Bundled skills must ship with the package; warn instead of silently exposing none.
                if builtin_entries.is_empty() {
                    accumulator.diagnostics.push(ResourceDiagnostic {
                        diagnostic_type: RESOURCE_DIAGNOSTIC_WARNING.to_string(),
                        message: if Path::new(bundled_skills_dir).exists() {
                            "built-in skills directory contains no skills; this build may be packaged incorrectly"
                                .to_string()
                        } else {
                            "built-in skills directory not found; this build may be packaged incorrectly"
                                .to_string()
                        },
                        path: Some(bundled_skills_dir.clone()),
                        collision: None,
                    });
                }
                let mut builtin_skill_overrides = overrides_for(&user_overrides, "skills");
                // Web search stays disabled until explicitly enabled.
                if !self.settings().get_bundled_websearch_enabled() {
                    builtin_skill_overrides.push("-websearch/SKILL.md".to_string());
                }
                // MCP integration skills stay disabled until their authentication is available.
                builtin_skill_overrides.extend((self.extra_builtin_skill_overrides)());
                add_resources(
                    accumulator,
                    "skills",
                    builtin_entries,
                    builtin_metadata,
                    builtin_skill_overrides,
                    bundled_skills_dir,
                );
            }
        }
        add_resources(
            accumulator,
            "prompts",
            collect_auto_prompt_entries(&dir_for(&user_dirs, "prompts")),
            user_metadata.clone(),
            overrides_for(&user_overrides, "prompts"),
            global_base_dir,
        );
        add_resources(
            accumulator,
            "themes",
            collect_auto_theme_entries(&dir_for(&user_dirs, "themes")),
            user_metadata,
            overrides_for(&user_overrides, "themes"),
            global_base_dir,
        );
    }

    fn collect_files_from_paths(&self, paths: &[String], resource_type: &str) -> Vec<String> {
        let mut files: Vec<String> = Vec::new();
        for path in paths {
            if !Path::new(path).exists() {
                continue;
            }

            let Ok(stats) = std::fs::metadata(path) else {
                // Ignore inaccessible resource paths.
                continue;
            };
            if stats.is_file() {
                files.push(path.clone());
            } else if stats.is_dir() {
                files.extend(collect_resource_files(path, resource_type));
            }
        }
        files
    }

    fn create_accumulator(&self) -> ResourceAccumulator {
        ResourceAccumulator::default()
    }

    fn to_resolved_paths(&self, accumulator: &ResourceAccumulator) -> ResolvedPaths {
        let map_to_resolved = |entries: &BTreeMap<String, (PathMetadata, bool)>| -> Vec<ResolvedResource> {
            let mut resolved: Vec<ResolvedResource> = entries
                .iter()
                .map(|(path, (metadata, enabled))| ResolvedResource {
                    path: path.clone(),
                    enabled: *enabled,
                    metadata: metadata.clone(),
                })
                .collect();
            resolved.sort_by_key(|entry| resource_precedence_rank(&entry.metadata));

            let mut seen: HashSet<String> = HashSet::new();
            resolved
                .into_iter()
                .filter(|entry| {
                    let canonical_path = canonicalize_path(&entry.path);
                    if seen.contains(&canonical_path) {
                        return false;
                    }
                    seen.insert(canonical_path);
                    true
                })
                .collect()
        };

        ResolvedPaths {
            extensions: map_to_resolved(&accumulator.extensions),
            skills: map_to_resolved(&accumulator.skills),
            prompts: map_to_resolved(&accumulator.prompts),
            themes: map_to_resolved(&accumulator.themes),
            diagnostics: accumulator.diagnostics.clone(),
        }
    }

    async fn run_command_capture(
        &self,
        command: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout_ms: Option<f64>,
        env: Option<Vec<(String, String)>>,
    ) -> Result<String, String> {
        let base_env = get_env();
        let merged_env = match env {
            Some(extra) => {
                let mut merged: BTreeMap<String, String> = base_env.into_iter().collect();
                for (key, value) in extra {
                    merged.insert(key, value);
                }
                merged.into_iter().collect()
            }
            None => base_env,
        };
        let handle = spawn_hidden(
            command,
            args,
            SpawnOptions {
                cwd: cwd.map(str::to_string),
                env: Some(merged_env),
                shell: should_use_windows_shell(command),
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
        )
        .map_err(|error| error.to_string())?;
        let mut child = handle.child;
        let pid = child.id().map(|pid| pid as i32);
        // The owner takes and drains both pipes while waiting. Borrowing the
        // owner also prevents cancellation of the first timeout from dropping
        // and force-killing the child before the documented SIGTERM path.
        let capture = async {
            let output = crate::utils::child_process::wait_for_child_process_with_output(&mut child).await?;
            let status = crate::utils::child_process::exit_code_for_status(&output.status);
            Ok::<_, std::io::Error>((
                String::from_utf8_lossy(&output.stdout).to_string(),
                String::from_utf8_lossy(&output.stderr).to_string(),
                status,
            ))
        };
        tokio::pin!(capture);


        // `const timeout = setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs)`
        // (package-manager.ts:2381-2387); the `close` handler then rejects with
        // `${command} ${args.join(" ")} timed out after ${options?.timeoutMs}ms` (2392-2397).
        let outcome = match capture_timeout_duration(timeout_ms) {
            Some(duration) => tokio::time::timeout(duration, &mut capture).await,
            None => Ok((&mut capture).await),
        };
        let (stdout, stderr, code) = match outcome {
            Ok(result) => result.map_err(|error| error.to_string())?,
            Err(_) => {
                // `child.kill()`: SIGTERM on POSIX, TerminateProcess on Windows.
                if let Some(pid) = pid {
                    signal_process_group_or_process(pid, Signal::Term);
                    // Resume the same capture so its readers and wait remain
                    // intact. Escalate only after the graceful bound expires,
                    // then bound the final kill/reap wait as well.
                    if tokio::time::timeout(
                        std::time::Duration::from_secs(1),
                        &mut capture,
                    )
                    .await
                    .is_err()
                    {
                        signal_process_group_or_process(pid, Signal::Kill);
                        let _ = tokio::time::timeout(
                            std::time::Duration::from_secs(3),
                            &mut capture,
                        )
                        .await;
                    }
                }
                return Err(format!(
                    "{} {} timed out after {}ms",
                    command,
                    args.join(" "),
                    js_number_text(timeout_ms.expect("the timer exists only for a numeric timeoutMs"))
                ));
            }
        };
        match code {
            Some(0) => Ok(stdout.trim().to_string()),
            Some(code) => Err(format!(
                "{} {} failed with code {code}: {}",
                command,
                args.join(" "),
                if stderr.is_empty() { stdout.clone() } else { stderr }
            )),
            None => Err(format!(
                "{} {} failed with signal unknown: {}",
                command,
                args.join(" "),
                if stderr.is_empty() { stdout.clone() } else { stderr }
            )),
        }
    }

    async fn run_command(&self, command: &str, args: &[String], cwd: Option<&str>) -> Result<(), String> {
        let handle = spawn_hidden(
            command,
            args,
            SpawnOptions {
                cwd: cwd.map(str::to_string),
                env: Some(get_env()),
                shell: should_use_windows_shell(command),
                // `stdio: isStdoutTakenOver() ? ["ignore", 2, 2] : "inherit"`.
                capture_stdout: is_stdout_taken_over(),
                capture_stderr: is_stdout_taken_over(),
                ..Default::default()
            },
        )
        .map_err(|error| error.to_string())?;
        let code = wait_for_child_process(handle.child)
            .await
            .map_err(|error| error.to_string())?;
        match code {
            Some(0) => Ok(()),
            Some(code) => Err(format!("{} {} failed with code {code}", command, args.join(" "))),
            None => Err(format!("{} {} failed with code null", command, args.join(" "))),
        }
    }

    fn run_command_sync(&self, command: &str, args: &[String]) -> Result<String, String> {
        let result = spawn_sync_hidden(
            command,
            args,
            SpawnOptions {
                env: Some(get_env()),
                shell: should_use_windows_shell(command),
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
        );
        match result {
            Err(error) => Err(format!(
                "Failed to run {} {}: {error}",
                command,
                args.join(" ")
            )),
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    return Err(format!(
                        "Failed to run {} {}: {}",
                        command,
                        args.join(" "),
                        if stderr.is_empty() { stdout } else { stderr }
                    ));
                }
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !stdout.is_empty() {
                    return Ok(stdout);
                }
                Ok(String::from_utf8_lossy(&output.stderr).trim().to_string())
            }
        }
    }
}

// =============================================================================
// Small helpers (the port of node:path and friends)
// =============================================================================

/// The `setTimeout(..., options.timeoutMs)` delay: absent for a missing `timeoutMs`
/// (`typeof options?.timeoutMs === "number"`, package-manager.ts:2384), otherwise the
/// delay Node accepts, clamped to `TIMEOUT_MAX` (larger delays fire immediately in Node;
/// NaN and negative delays behave like 0).
fn capture_timeout_duration(timeout_ms: Option<f64>) -> Option<std::time::Duration> {
    let timeout_ms = timeout_ms?;
    let clamped = if timeout_ms.is_nan() {
        0.0
    } else {
        timeout_ms.max(0.0).min(MAX_TIMEOUT_MS)
    };
    Some(std::time::Duration::from_millis(clamped.floor() as u64))
}

/// JS `String(number)` for the timeout text: integers print without a fractional part.
fn js_number_text(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

#[derive(Debug, Clone)]
struct NpmCommand {
    command: String,
    args: Vec<String>,
}

fn package_list(settings: &Settings) -> Vec<Value> {
    match settings.get("packages") {
        Some(Value::Array(items)) => items.clone(),
        _ => Vec::new(),
    }
}

fn string_list(settings: &Settings, resource_type: &str) -> Vec<String> {
    match settings.get(resource_type) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn package_source_string(pkg: &Value) -> String {
    match pkg {
        Value::String(source) => source.clone(),
        Value::Object(object) => object
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn package_filter(pkg: &Value) -> Option<PackageFilter> {
    let object = pkg.as_object()?;
    let strings = |key: &str| -> Option<Vec<String>> {
        object.get(key).and_then(Value::as_array).map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
    };
    Some(PackageFilter {
        extensions: strings("extensions"),
        skills: strings("skills"),
        prompts: strings("prompts"),
        themes: strings("themes"),
    })
}

fn target_map<'a>(accumulator: &'a mut ResourceAccumulator, resource_type: &str) -> &'a mut BTreeMap<String, (PathMetadata, bool)> {
    match resource_type {
        "extensions" => &mut accumulator.extensions,
        "skills" => &mut accumulator.skills,
        "prompts" => &mut accumulator.prompts,
        "themes" => &mut accumulator.themes,
        other => panic!("Unknown resource type: {other}"),
    }
}

fn add_resource(
    map: &mut BTreeMap<String, (PathMetadata, bool)>,
    path: &str,
    metadata: PathMetadata,
    enabled: bool,
) {
    if path.is_empty() {
        return;
    }
    map.entry(path.to_string()).or_insert((metadata, enabled));
}

/// `spec.match(/^(@?[^@]+(?:\/[^@]+)?)(?:@(.+))?$/)`.
fn parse_npm_spec(spec: &str) -> (String, Option<String>) {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        regex::Regex::new(r"^(@?[^@]+(?:/[^@]+)?)(?:@(.+))?$").expect("npm spec pattern")
    });
    match pattern.captures(spec) {
        Some(captures) => (
            captures.get(1).map_or(spec, |name| name.as_str()).to_string(),
            captures.get(2).map(|version| version.as_str().to_string()),
        ),
        None => (spec.to_string(), None),
    }
}

fn validate_git_source_path_parts(source: &GitSource) -> Result<(), String> {
    let invalid = || format!("Unsafe git package path: {}/{}", source.host, source.path);
    if source.host.is_empty()
        || source.host.trim() != source.host
        || source.host == "."
        || source.host == ".."
        || source.host.contains(['/', '\\', ':', '@'])
        || source.host.chars().any(char::is_control)
        || source.host.split('.').any(str::is_empty)
        || source.path.is_empty()
        || source.path.trim() != source.path
        || source.path.starts_with(['/', '\\'])
        || source.path.contains('\\')
        || source.path.chars().any(char::is_control)
    {
        return Err(invalid());
    }
    for segment in source.path.split('/') {
        if segment.is_empty()
            || segment == "."
            || segment == ".."
            || segment.contains(':')
            || segment.ends_with([' ', '.'])
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn lexical_absolute_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(format!("Path escapes its root: {}", absolute.display()));
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn canonicalize_with_missing_components(path: &Path) -> Result<PathBuf, String> {
    let mut existing = path.to_path_buf();
    let mut missing = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .ok_or_else(|| format!("Cannot resolve install path: {}", path.display()))?;
        missing.push(name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| format!("Cannot resolve install path: {}", path.display()))?
            .to_path_buf();
    }
    let mut resolved = std::fs::canonicalize(&existing).map_err(|error| {
        format!("Cannot resolve install path {}: {error}", existing.display())
    })?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn join_path(base: &str, name: &str) -> String {
    if name.is_empty() {
        return base.to_string();
    }
    PathBuf::from(base)
        .join(name)
        .to_string_lossy()
        .to_string()
}

fn absolute_join(base: &str, name: &str) -> String {
    let joined = join_path(base, name);
    absolute_path(&joined)
}

fn absolute_path(path: &str) -> String {
    if Path::new(path).is_absolute() {
        return path.to_string();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path).to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string())
}

fn parent_path(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| {
            let text = parent.to_string_lossy().to_string();
            if text.is_empty() {
                ".".to_string()
            } else {
                text
            }
        })
        .unwrap_or_else(|| ".".to_string())
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn relative_path(from: &str, to: &str) -> String {
    let from_path = Path::new(from);
    let to_path = Path::new(to);
    match pathdiff(from_path, to_path) {
        Some(relative) => relative,
        None => to.to_string(),
    }
}

/// `path.relative(from, to)` (empty string when both resolve to the same path).
fn pathdiff(from: &Path, to: &Path) -> Option<String> {
    let from_components: Vec<String> = from
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect();
    let to_components: Vec<String> = to
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect();

    let mut shared = 0;
    while shared < from_components.len()
        && shared < to_components.len()
        && from_components[shared] == to_components[shared]
    {
        shared += 1;
    }

    let mut parts: Vec<String> = Vec::new();
    for _ in shared..from_components.len() {
        parts.push("..".to_string());
    }
    parts.extend(to_components[shared..].iter().cloned());
    Some(parts.join("/"))
}

fn sha256_hex_8(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()[..8].to_string()
}

/// `/^([0-9a-f]{40})\s+/m` - the first 40-hex-char token on any line.
fn first_hex40_match(text: &str) -> Option<String> {
    for line in text.split('\n') {
        let trimmed = line.trim_start();
        if trimmed.len() >= 40 && trimmed[..40].chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
        {
            let candidate = &trimmed[..40];
            if trimmed[40..].starts_with(char::is_whitespace) {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// `/^([0-9a-f]{40})\s+HEAD$/m`.
fn head_hex40_match(text: &str) -> Option<String> {
    for line in text.split('\n') {
        let trimmed = line.trim_end_matches('\r');
        if trimmed.len() == 45 && trimmed.ends_with("HEAD") {
            let candidate = &trimmed[..40];
            if candidate.chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
                && trimmed[40..].starts_with(char::is_whitespace)
            {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::settings_manager::InMemorySettingsStorage;

    fn manager(cwd: &str, agent_dir: &str, settings: Value) -> DefaultPackageManager {
        let settings_manager = SettingsManager::in_memory(
            settings.as_object().unwrap().iter().map(|(key, value)| (key.clone(), value.clone())).collect(),
        );
        DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.to_string(),
            agent_dir: agent_dir.to_string(),
            settings_manager: Arc::new(std::sync::Mutex::new(settings_manager)),
            bundled_skills_dir: Some(None),
            extra_builtin_skill_overrides: None,
        })
    }

    fn temp_dir(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("pm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_string_lossy().to_string()
    }

    #[test]
    fn offline_mode_reads_the_env_flag() {
        assert!(!offline_mode_from_value(None));
        assert!(!offline_mode_from_value(Some("")));
        assert!(offline_mode_from_value(Some("1")));
        assert!(offline_mode_from_value(Some("TRUE")));
        assert!(offline_mode_from_value(Some("yes")));
        assert!(!offline_mode_from_value(Some("0")));
    }

    #[test]
    fn precedence_ranks_match_the_documented_order() {
        let metadata = |source: &str, scope: &str, origin: &str| PathMetadata {
            source: source.to_string(),
            scope: scope.to_string(),
            origin: origin.to_string(),
            base_dir: None,
        };
        assert_eq!(resource_precedence_rank(&metadata("local", "project", "top-level")), 0);
        assert_eq!(resource_precedence_rank(&metadata("auto", "project", "top-level")), 1);
        assert_eq!(resource_precedence_rank(&metadata("local", "user", "top-level")), 2);
        assert_eq!(resource_precedence_rank(&metadata("auto", "user", "top-level")), 3);
        assert_eq!(resource_precedence_rank(&metadata("npm:x", "user", "package")), 4);
        assert_eq!(resource_precedence_rank(&metadata("builtin", "user", "top-level")), 5);
    }

    #[test]
    fn parses_npm_specs() {
        assert_eq!(parse_npm_spec("pkg"), ("pkg".to_string(), None));
        assert_eq!(
            parse_npm_spec("pkg@1.2.3"),
            ("pkg".to_string(), Some("1.2.3".to_string()))
        );
        assert_eq!(
            parse_npm_spec("@scope/pkg@next"),
            ("@scope/pkg".to_string(), Some("next".to_string()))
        );
        assert_eq!(parse_npm_spec("@scope/pkg"), ("@scope/pkg".to_string(), None));
        assert_eq!(parse_npm_spec("pkg@"), ("pkg@".to_string(), None));
        assert_eq!(parse_npm_spec("@"), ("@".to_string(), None));
    }

    #[test]
    fn parses_source_forms() {
        let manager = manager("C:/tmp", "C:/tmp/agent", serde_json::json!({}));
        match manager.parse_source("npm:@scope/pkg@1.0.0") {
            ParsedSource::Npm(npm) => {
                assert_eq!(npm.name, "@scope/pkg");
                assert!(npm.pinned);
                assert_eq!(npm.spec, "@scope/pkg@1.0.0");
            }
            other => panic!("expected npm source, got {other:?}"),
        }
        assert!(matches!(
            manager.parse_source("git:github.com/user/repo"),
            ParsedSource::Git(_)
        ));
        match manager.parse_source("local:./ext") {
            ParsedSource::Local(local) => assert_eq!(local.path, "local:./ext"),
            other => panic!("expected local source, got {other:?}"),
        }
    }

    #[test]
    fn pattern_classification_matches_the_typescript() {
        assert!(is_pattern("!x"));
        assert!(is_pattern("+x"));
        assert!(is_pattern("-x"));
        assert!(is_pattern("a*b"));
        assert!(is_pattern("a?b"));
        assert!(!is_pattern("plain"));
        assert!(is_override_pattern("-x"));
        assert!(!is_override_pattern("*"));
        assert!(has_glob_pattern("a*"));
        assert!(!has_glob_pattern("plain"));

        let (plain, patterns) = split_patterns(&["a".to_string(), "!b".to_string()]);
        assert_eq!(plain, vec!["a"]);
        assert_eq!(patterns, vec!["!b"]);
    }

    #[test]
    fn prefix_ignore_pattern_matches_the_typescript() {
        assert_eq!(prefix_ignore_pattern("", ""), None);
        assert_eq!(prefix_ignore_pattern("# comment", ""), None);
        assert_eq!(prefix_ignore_pattern("\\# literal", ""), Some("\\# literal".to_string()));
        assert_eq!(prefix_ignore_pattern("!keep", "sub/"), Some("!sub/keep".to_string()));
        assert_eq!(prefix_ignore_pattern("/root", ""), Some("root".to_string()));
        assert_eq!(prefix_ignore_pattern("x", "sub/"), Some("sub/x".to_string()));
    }

    #[test]
    fn apply_patterns_follows_the_documented_order() {
        let base = "/root";
        let all = vec![
            "/root/a.ts".to_string(),
            "/root/b.ts".to_string(),
            "/root/c.js".to_string(),
        ];
        let enabled = apply_patterns(&all, &["*.ts".to_string()], base);
        assert!(enabled.contains("/root/a.ts"));
        assert!(!enabled.contains("/root/c.js"));

        let enabled = apply_patterns(&all, &["*.ts".to_string(), "!b.ts".to_string()], base);
        assert!(enabled.contains("/root/a.ts"));
        assert!(!enabled.contains("/root/b.ts"));

        let enabled = apply_patterns(&all, &["*.ts".to_string(), "-a.ts".to_string()], base);
        assert!(!enabled.contains("/root/a.ts"));

        let enabled = apply_patterns(&all, &["!*.ts".to_string(), "+a.ts".to_string()], base);
        assert!(enabled.contains("/root/a.ts"));
        assert!(!enabled.contains("/root/b.ts"));

        // An empty pattern list includes everything.
        let enabled = apply_patterns(&all, &[], base);
        assert_eq!(enabled.len(), 3);
    }

    #[test]
    fn exact_patterns_strip_dot_prefixes() {
        assert_eq!(normalize_exact_pattern("./a.ts"), "a.ts");
        assert!(matches_any_exact_pattern("/root/a.ts", &["./a.ts".to_string()], "/root"));
        assert!(!matches_any_exact_pattern("/root/b.ts", &["a.ts".to_string()], "/root"));
    }

    #[test]
    fn skill_patterns_also_match_the_parent_directory() {
        let patterns = vec!["websearch".to_string()];
        assert!(matches_any_pattern("/root/websearch/SKILL.md", &patterns, "/root"));
        assert!(!matches_any_pattern("/root/other/SKILL.md", &patterns, "/root"));
    }

    #[test]
    fn overrides_decide_enabled_state() {
        let overrides = vec!["!*.ts".to_string(), "+keep.ts".to_string()];
        assert!(is_enabled_by_overrides("/root/keep.ts", &overrides, "/root"));
        assert!(!is_enabled_by_overrides("/root/drop.ts", &overrides, "/root"));
        assert!(is_enabled_by_overrides("/root/other.md", &overrides, "/root"));
    }

    #[test]
    fn hex_matchers_read_git_ls_remote_output() {
        let head = "0123456789abcdef0123456789abcdef01234567\tHEAD\n";
        assert_eq!(
            head_hex40_match(head),
            Some("0123456789abcdef0123456789abcdef01234567".to_string())
        );
        let remote = "0123456789abcdef0123456789abcdef01234567\trefs/heads/main\n";
        assert_eq!(
            first_hex40_match(remote),
            Some("0123456789abcdef0123456789abcdef01234567".to_string())
        );
        assert_eq!(head_hex40_match("nope"), None);
        assert_eq!(first_hex40_match("ABCDEF\tx"), None);
    }

    #[test]
    fn npm_install_paths_follow_the_scope() {
        let cwd = temp_dir("npm-path");
        let agent = temp_dir("npm-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let source = NpmSource {
            spec: "pkg".to_string(),
            name: "pkg".to_string(),
            pinned: false,
        };
        let project = manager.get_npm_install_path(&source, SOURCE_SCOPE_PROJECT);
        assert!(project.ends_with(&format!("pkg")));
        assert!(project.contains(".prime"));
        let temporary = manager.get_npm_install_path(&source, SOURCE_SCOPE_TEMPORARY);
        assert!(temporary.contains("pi-extensions"));
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn git_install_paths_follow_the_scope() {
        let cwd = temp_dir("git-path");
        let agent = temp_dir("git-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let source = GitSource {
            kind: "git".to_string(),
            repo: "https://github.com/user/repo.git".to_string(),
            host: "github.com".to_string(),
            path: "user/repo".to_string(),
            reference: None,
            pinned: false,
        };
        let project = manager.get_git_install_path(&source, SOURCE_SCOPE_PROJECT);
        // `getGitInstallPath` is `join(this.cwd, CONFIG_DIR_NAME, "git", source.host, source.path)`
        // (package-manager.ts:1868), i.e. `node:path.join`. On win32 Node returns a
        // backslash path for every segment:
        //   node -e "console.log(require('path').join('C:\\cwd','.prime','git','github.com','user/repo'))"
        //   -> C:\cwd\.prime\git\github.com\user\repo
        // so the exact string "git/github.com/user/repo" is not what the TypeScript
        // produces on this host. The TypeScript suite compares these paths through
        // `normalizeForMatch` (test/package-manager.test.ts:11-17), which folds both
        // separators to "/", so assert the same directory shape the TypeScript
        // asserts instead of a byte-exact win32 string.
        assert!(to_posix_path(&project).ends_with("git/github.com/user/repo"));
        let user = manager.get_git_install_path(&source, SOURCE_SCOPE_USER);
        assert!(user.starts_with(&agent));
        assert_eq!(manager.get_git_install_root(SOURCE_SCOPE_TEMPORARY), None);
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn resolves_local_resource_entries_with_patterns() {
        let cwd = temp_dir("resolve-local");
        let agent = temp_dir("resolve-agent");
        let project_dir = Path::new(&agent).join("prompts");
        std::fs::create_dir_all(&project_dir).unwrap();
        std::fs::write(project_dir.join("keep.md"), "k").unwrap();
        std::fs::write(project_dir.join("drop.md"), "d").unwrap();

        let manager = manager(
            &cwd,
            &agent,
            serde_json::json!({"prompts": ["prompts", "!drop.md"]}),
        );
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        let paths: Vec<String> = resolved.prompts.iter().map(|entry| entry.path.clone()).collect();
        assert!(paths.iter().any(|path| path.ends_with("keep.md")));
        assert_eq!(resolved.prompts.len(), 2);
        assert!(resolved.prompts.iter().find(|entry| entry.path.ends_with("keep.md")).unwrap().enabled);
        assert!(!resolved.prompts.iter().find(|entry| entry.path.ends_with("drop.md")).unwrap().enabled);
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn resolves_auto_discovered_prompt_and_theme_files() {
        let cwd = temp_dir("resolve-auto");
        let agent = temp_dir("resolve-auto-agent");
        let project_prompts = Path::new(&cwd).join(CONFIG_DIR_NAME).join("prompts");
        std::fs::create_dir_all(&project_prompts).unwrap();
        std::fs::write(project_prompts.join("p.md"), "p").unwrap();
        let project_themes = Path::new(&cwd).join(CONFIG_DIR_NAME).join("themes");
        std::fs::create_dir_all(&project_themes).unwrap();
        std::fs::write(project_themes.join("t.json"), "{}").unwrap();
        std::fs::write(project_themes.join(".hidden.json"), "{}").unwrap();

        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        assert!(resolved.prompts.iter().any(|entry| entry.path.ends_with("p.md")));
        assert!(resolved.themes.iter().any(|entry| entry.path.ends_with("t.json")));
        assert!(!resolved
            .themes
            .iter()
            .any(|entry| entry.path.ends_with(".hidden.json")));
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn extension_discovery_prefers_index_ts() {
        let dir = temp_dir("ext-discovery");
        let sub = Path::new(&dir).join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("index.ts"), "").unwrap();
        std::fs::write(Path::new(&dir).join("a.js"), "").unwrap();
        std::fs::write(Path::new(&dir).join("notes.txt"), "").unwrap();

        let entries = collect_auto_extension_entries(&dir);
        assert!(entries.iter().any(|path| path.ends_with("a.js")));
        assert!(entries.iter().any(|path| path.ends_with("index.ts")));
        assert!(!entries.iter().any(|path| path.ends_with("notes.txt")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skill_discovery_finds_skill_files_and_agents_layout() {
        let dir = temp_dir("skill-discovery");
        let skill_dir = Path::new(&dir).join("websearch");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "").unwrap();
        std::fs::write(Path::new(&dir).join("top.md"), "").unwrap();

        let entries = collect_auto_skill_entries(&dir, "pi");
        assert!(entries.iter().any(|path| path.ends_with("SKILL.md")));
        assert!(entries.iter().any(|path| path.ends_with("top.md")));

        let agents = collect_auto_skill_entries(&dir, "agents");
        assert!(agents.iter().any(|path| path.ends_with("SKILL.md")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gitignore_rules_exclude_discovered_files() {
        let dir = temp_dir("gitignore");
        std::fs::write(Path::new(&dir).join(".gitignore"), "skip.md\n").unwrap();
        std::fs::write(Path::new(&dir).join("skip.md"), "").unwrap();
        std::fs::write(Path::new(&dir).join("keep.md"), "").unwrap();

        let entries = collect_auto_prompt_entries(&dir);
        assert!(entries.iter().any(|path| path.ends_with("keep.md")));
        assert!(!entries.iter().any(|path| path.ends_with("skip.md")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_entries_resolve_globs() {
        let dir = temp_dir("manifest");
        let ext_dir = Path::new(&dir).join("ext");
        std::fs::create_dir_all(&ext_dir).unwrap();
        std::fs::write(ext_dir.join("one.ts"), "").unwrap();
        std::fs::write(Path::new(&dir).join("package.json"), "{\"pi\":{\"extensions\":[\"ext/*.ts\"]}}").unwrap();

        let manager = manager("C:/tmp", "C:/tmp/agent", serde_json::json!({}));
        let manifest = manager.read_pi_manifest(&dir).expect("manifest");
        assert_eq!(manifest.extensions, Some(vec!["ext/*.ts".to_string()]));

        let files = manager.collect_files_from_manifest_entries(
            manifest.extensions.as_ref().unwrap(),
            &dir,
            "extensions",
        );
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("one.ts"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dedupe_keeps_the_project_entry() {
        let cwd = temp_dir("dedupe");
        let agent = temp_dir("dedupe-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let packages = vec![
            (Value::String("npm:pkg".to_string()), SOURCE_SCOPE_PROJECT.to_string()),
            (Value::String("npm:pkg@1.0.0".to_string()), SOURCE_SCOPE_USER.to_string()),
        ];
        let deduped = manager.dedupe_packages(&packages);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].1, SOURCE_SCOPE_PROJECT);
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn identity_normalizes_ssh_and_https() {
        let cwd = temp_dir("identity");
        let agent = temp_dir("identity-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let https = manager.get_package_identity("https://github.com/user/repo", None);
        let ssh = manager.get_package_identity("git:git@github.com:user/repo.git", None);
        assert_eq!(https, ssh);
        assert_eq!(https, "git:github.com/user/repo");
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn settings_sources_are_added_and_removed() {
        let cwd = temp_dir("settings");
        let agent = temp_dir("settings-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        assert!(manager.add_source_to_settings("npm:pkg", false));
        assert!(!manager.add_source_to_settings("npm:pkg", false));
        assert_eq!(manager.list_configured_packages().len(), 1);
        assert!(manager.remove_source_from_settings("npm:pkg", false));
        assert!(!manager.remove_source_from_settings("npm:pkg", false));
        assert!(manager.list_configured_packages().is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn no_matching_package_message_suggests_a_configured_source() {
        let cwd = temp_dir("suggest");
        let agent = temp_dir("suggest-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let configured = vec![Value::String("npm:@scope/pkg".to_string())];
        let message = manager.build_no_matching_package_message("@scope/pkg", &configured);
        assert_eq!(
            message,
            "No matching package found for @scope/pkg. Did you mean npm:@scope/pkg?"
        );
        let message = manager.build_no_matching_package_message("other", &configured);
        assert_eq!(message, "No matching package found for other");
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn update_reports_no_matching_package() {
        let cwd = temp_dir("update");
        let agent = temp_dir("update-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let error = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.update(Some("npm:missing")))
            .unwrap_err();
        assert_eq!(error, "No matching package found for npm:missing");
        // Updating an empty configured source set is a no-op.
        assert!(tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.update(None))
            .is_ok());
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn offline_update_check_returns_nothing() {
        let cwd = temp_dir("update-check");
        let agent = temp_dir("update-check-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({"packages": ["npm:pkg"]}));
        let updates = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.check_for_available_updates_with_offline(true));
        assert!(updates.is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn relative_paths_use_posix_separators() {
        assert_eq!(relative_path("/a/b", "/a/b/c.md"), "c.md");
        assert_eq!(relative_path("/a/b", "/a/b"), "");
        assert_eq!(to_posix_path(&format!("a{}b", std::path::MAIN_SEPARATOR)), "a/b");
    }

    #[test]
    fn npm_command_defaults_and_overrides() {
        let cwd = temp_dir("npm-command");
        let agent = temp_dir("npm-command-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let command = manager.get_npm_command().unwrap();
        assert_eq!(command.command, "npm");
        assert!(command.args.is_empty());
        assert_eq!(
            manager.get_git_dependency_install_args(),
            vec!["install".to_string(), "--omit=dev".to_string()]
        );
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn ensure_git_ignore_writes_the_ignore_file() {
        let dir = temp_dir("gitignore-write");
        let manager = manager(&dir, &dir, serde_json::json!({}));
        manager.ensure_git_ignore(&dir);
        let content = std::fs::read_to_string(Path::new(&dir).join(".gitignore")).unwrap();
        assert_eq!(content, "*\n!.gitignore\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_npm_project_writes_a_private_package_json() {
        let dir = temp_dir("npm-project");
        let manager = manager(&dir, &dir, serde_json::json!({}));
        let root = Path::new(&dir).join("npm");
        manager.ensure_npm_project(&root.to_string_lossy());
        let content = std::fs::read_to_string(root.join("package.json")).unwrap();
        assert!(content.contains("\"pi-extensions\""));
        assert!(content.contains("\"private\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_extension_sources_uses_temporary_scope() {
        let cwd = temp_dir("ext-sources");
        let agent = temp_dir("ext-sources-agent");
        let local_dir = Path::new(&cwd).join("local-ext");
        std::fs::create_dir_all(&local_dir).unwrap();
        std::fs::write(local_dir.join("index.ts"), "").unwrap();

        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve_extension_sources(&[local_dir.to_string_lossy().to_string()], false, true))
            .unwrap();
        assert_eq!(resolved.extensions.len(), 1);
        assert_eq!(resolved.extensions[0].metadata.scope, SOURCE_SCOPE_TEMPORARY);
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn missing_local_extension_source_is_skipped() {
        let cwd = temp_dir("missing-local");
        let agent = temp_dir("missing-local-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve_extension_sources(&["./nope".to_string()], true, false))
            .unwrap();
        assert!(resolved.extensions.is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn local_package_source_adds_extension_and_filtered_resources() {
        let cwd = temp_dir("local-package");
        let agent = temp_dir("local-package-agent");
        let package_dir = Path::new(&agent).join("pkg");
        let prompts = package_dir.join("prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        std::fs::write(prompts.join("a.md"), "").unwrap();
        std::fs::write(prompts.join("b.md"), "").unwrap();

        let manager = manager(
            &cwd,
            &agent,
            serde_json::json!({"packages": [{"source": "./pkg", "prompts": ["prompts/a.md"]}]}),
        );
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        let enabled: Vec<&ResolvedResource> = resolved.prompts.iter().filter(|entry| entry.enabled).collect();
        assert_eq!(enabled.len(), 1);
        assert!(enabled[0].path.ends_with("a.md"));
        let disabled: Vec<&ResolvedResource> = resolved.prompts.iter().filter(|entry| !entry.enabled).collect();
        assert_eq!(disabled.len(), 1);
        assert!(disabled[0].path.ends_with("b.md"));
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn local_package_without_resource_dirs_is_an_extension() {
        let cwd = temp_dir("local-ext-only");
        let agent = temp_dir("local-ext-only-agent");
        let package_dir = Path::new(&agent).join("pkg");
        std::fs::create_dir_all(&package_dir).unwrap();
        std::fs::write(package_dir.join("index.ts"), "").unwrap();

        let manager = manager(&cwd, &agent, serde_json::json!({"packages": ["./pkg"]}));
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        assert_eq!(resolved.extensions.len(), 1);
        assert_eq!(Path::new(&resolved.extensions[0].path), package_dir.as_path());
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn resolve_dedupes_canonical_paths() {
        let cwd = temp_dir("dedupe-paths");
        let agent = temp_dir("dedupe-paths-agent");
        let prompts = Path::new(&agent).join("prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        std::fs::write(prompts.join("p.md"), "").unwrap();

        let manager = manager(
            &cwd,
            &agent,
            serde_json::json!({"prompts": ["prompts", "prompts/p.md"]}),
        );
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        assert_eq!(resolved.prompts.len(), 1);
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn bundled_skills_are_discovered_with_overrides() {
        let cwd = temp_dir("bundled");
        let agent = temp_dir("bundled-agent");
        let skills = Path::new(&cwd).join("skills");
        std::fs::create_dir_all(skills.join("websearch")).unwrap();
        std::fs::write(skills.join("websearch").join("SKILL.md"), "").unwrap();
        std::fs::create_dir_all(skills.join("other")).unwrap();
        std::fs::write(skills.join("other").join("SKILL.md"), "").unwrap();

        let mut settings_manager = SettingsManager::from_storage(Arc::new(InMemorySettingsStorage::new()));
        settings_manager.set_enable_builtin_skills(true);
        settings_manager.apply_overrides(&[("bundledSkills".to_string(), serde_json::json!({"websearch": false}))].into_iter().collect());
        let manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.clone(),
            agent_dir: agent.clone(),
            settings_manager: Arc::new(std::sync::Mutex::new(settings_manager)),
            bundled_skills_dir: Some(Some(skills.to_string_lossy().to_string())),
            extra_builtin_skill_overrides: Some(Arc::new(|| vec!["-other/SKILL.md".to_string()])),
        });
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        let websearch = resolved
            .skills
            .iter()
            .find(|entry| entry.path.ends_with("websearch/SKILL.md") || entry.path.ends_with("websearch\\SKILL.md"))
            .expect("websearch skill");
        assert!(!websearch.enabled);
        let other = resolved
            .skills
            .iter()
            .find(|entry| entry.path.ends_with("other/SKILL.md") || entry.path.ends_with("other\\SKILL.md"))
            .expect("other skill");
        assert!(!other.enabled);
        assert!(resolved
            .skills
            .iter()
            .all(|entry| entry.metadata.source != "builtin" || !entry.enabled));
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn missing_bundled_skills_directory_warns() {
        let cwd = temp_dir("bundled-missing");
        let agent = temp_dir("bundled-missing-agent");
        let missing = Path::new(&cwd).join("nope");
        let mut settings_manager = SettingsManager::from_storage(Arc::new(InMemorySettingsStorage::new()));
        settings_manager.set_enable_builtin_skills(true);
        let manager = DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.clone(),
            agent_dir: agent.clone(),
            settings_manager: Arc::new(std::sync::Mutex::new(settings_manager)),
            bundled_skills_dir: Some(Some(missing.to_string_lossy().to_string())),
            extra_builtin_skill_overrides: None,
        });
        let resolved = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.resolve(None))
            .unwrap();
        assert_eq!(resolved.diagnostics.len(), 1);
        assert_eq!(resolved.diagnostics[0].diagnostic_type, RESOURCE_DIAGNOSTIC_WARNING);
        assert!(resolved.diagnostics[0]
            .message
            .contains("built-in skills directory not found"));
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn progress_events_are_emitted_for_installs() {
        let cwd = temp_dir("progress");
        let agent = temp_dir("progress-agent");
        let mut manager = manager(&cwd, &agent, serde_json::json!({}));
        let events: Arc<std::sync::Mutex<Vec<ProgressEvent>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        manager.set_progress_callback(Some(Arc::new(move |event| {
            captured.lock().unwrap().push(event);
        })));
        let error = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.install("./does-not-exist", true))
            .unwrap_err();
        assert!(error.starts_with("Path does not exist:"));
        let events = events.lock().unwrap();
        assert_eq!(events[0].event_type, "start");
        assert_eq!(events[0].action, "install");
        assert_eq!(events[1].event_type, "error");
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn local_install_succeeds_for_an_existing_path() {
        let cwd = temp_dir("local-install");
        let agent = temp_dir("local-install-agent");
        let target = Path::new(&cwd).join("ext");
        std::fs::create_dir_all(&target).unwrap();
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(manager.install(&target.to_string_lossy(), true))
            .unwrap();
        assert_eq!(
            manager.get_installed_path(&target.to_string_lossy(), SOURCE_SCOPE_PROJECT),
            Some(target.to_string_lossy().to_string())
        );
        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&agent);
    }

    #[test]
    fn filtered_package_source_parses_from_settings() {
        let filter = package_filter(&serde_json::json!({"source": "./x", "skills": []})).expect("filter");
        assert_eq!(filter.skills, Some(Vec::new()));
        assert!(filter.prompts.is_none());
        assert!(package_filter(&Value::String("./x".to_string())).is_none());
        let source = PackageSource::Filtered(FilteredPackageSource {
            source: "./x".to_string(),
            extensions: None,
            skills: Some(Vec::new()),
            prompts: None,
            themes: None,
        });
        let value = serde_json::to_value(&source).unwrap();
        assert_eq!(package_source_string(&value), "./x");
    }

    fn local_git_source(repo: &str, reference: Option<&str>) -> GitSource {
        GitSource {
            kind: "git".to_string(),
            repo: repo.to_string(),
            host: "example.com".to_string(),
            path: "team/repo".to_string(),
            reference: reference.map(str::to_string),
            pinned: reference.is_some(),
        }
    }

    fn git_fixture(name: &str, with_dependency: bool) -> (String, String, String) {
        let root = temp_dir(name);
        let repo = Path::new(&root).join("source");
        std::fs::create_dir_all(&repo).unwrap();
        let run = |args: &[&str]| {
            let output = std::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(args)
                .output()
                .unwrap();
            assert!(output.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&output.stderr));
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        };
        run(&["init", "-q", "-b", "main"]);
        run(&["config", "user.email", "synthetic@example.test"]);
        run(&["config", "user.name", "Synthetic Test"]);
        if with_dependency {
            std::fs::write(
                repo.join("package.json"),
                r#"{"name":"fixture","private":true,"dependencies":{"left-pad":"1.0.0"}}"#,
            )
            .unwrap();
        }
        std::fs::write(repo.join("index.ts"), "export const fixture = true;\n").unwrap();
        run(&["add", "-A"]);
        run(&["commit", "-q", "-m", "fixture"]);
        let head = run(&["rev-parse", "HEAD"]);
        run(&["branch", "exact-ref"]);
        (root, repo.to_string_lossy().to_string(), head)
    }

    fn failing_once_npm_settings(root: &str) -> Value {
        let counter = Path::new(root).join("npm-calls");
        let script = Path::new(root).join(if cfg!(windows) { "npm-fixture.cmd" } else { "npm-fixture.sh" });
        if cfg!(windows) {
            std::fs::write(
                &script,
                format!(
                    "@echo off\r\nsetlocal EnableExtensions\r\nset \"count=0\"\r\nif exist \"{}\" set /p count=<\"{}\"\r\nset /a count=count+1\r\n>\"{}\" echo %count%\r\nif \"%count%\"==\"1\" exit /b 19\r\nif not exist node_modules\\left-pad mkdir node_modules\\left-pad\r\n>node_modules\\left-pad\\package.json echo {{\"name\":\"left-pad\",\"version\":\"1.0.0\"}}\r\n",
                    counter.display(), counter.display(), counter.display()
                ),
            )
            .unwrap();
            serde_json::json!({"npmCommand": ["cmd", "/c", script.to_string_lossy()]})
        } else {
            std::fs::write(
                &script,
                format!(
                    "#!/bin/sh\ncount=0\nif [ -f '{0}' ]; then count=$(cat '{0}'); fi\ncount=$((count+1))\nprintf '%s' \"$count\" > '{0}'\nif [ \"$count\" -eq 1 ]; then exit 19; fi\nmkdir -p node_modules/left-pad\nprintf '%s' '{{\"name\":\"left-pad\",\"version\":\"1.0.0\"}}' > node_modules/left-pad/package.json\n",
                    counter.display()
                ),
            )
            .unwrap();
            serde_json::json!({"npmCommand": ["/bin/sh", script.to_string_lossy()]})
        }
    }

    #[tokio::test]
    async fn git_package_paths_reject_escapes_before_remove() {
        let cwd = temp_dir("unsafe-git-remove");
        let agent = temp_dir("unsafe-git-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let canary = Path::new(&cwd).join("outside-canary");
        std::fs::write(&canary, "keep").unwrap();
        for (host, path) in [
            ("/tmp", "team/repo"),
            ("..", "team/repo"),
            ("C:", "team/repo"),
            ("example.com", "../outside-canary"),
            ("example.com", "team/../../outside-canary"),
            ("example.com", "/tmp/repo"),
            ("example.com", "team\\..\\outside-canary"),
            ("example.com", "C:/outside-canary"),
            ("example.com", "team\nbad"),
        ] {
            let mut source = local_git_source("unused", None);
            source.host = host.to_string();
            source.path = path.to_string();
            let error = manager.remove_git(&source, SOURCE_SCOPE_PROJECT).unwrap_err();
            assert!(error.contains("Unsafe git package path"), "{host} {path}: {error}");
            assert_eq!(std::fs::read_to_string(&canary).unwrap(), "keep");
        }
        let error = manager
            .remove("git:example.com/team/../../outside-canary", true)
            .await
            .unwrap_err();
        assert!(error.contains("Unsafe git package path"));
        assert_eq!(std::fs::read_to_string(&canary).unwrap(), "keep");
        let scp = parse_git_url("git:git@github.com:user/repo").unwrap();
        assert!(manager.validated_git_install_path(&scp, SOURCE_SCOPE_PROJECT).is_ok());
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(agent);
    }

    #[cfg(unix)]
    #[test]
    fn git_package_paths_reject_existing_symlink_escapes() {
        use std::os::unix::fs::symlink;
        let cwd = temp_dir("unsafe-git-symlink");
        let agent = temp_dir("unsafe-git-symlink-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let root = Path::new(&cwd).join(CONFIG_DIR_NAME).join("git");
        std::fs::create_dir_all(&root).unwrap();
        let outside = Path::new(&cwd).join("outside");
        std::fs::create_dir_all(outside.join("team")).unwrap();
        let canary = outside.join("canary");
        std::fs::write(&canary, "keep").unwrap();
        symlink(&outside, root.join("example.com")).unwrap();
        let source = local_git_source("unused", None);
        assert!(manager.validated_git_install_path(&source, SOURCE_SCOPE_PROJECT).is_err());
        assert!(manager.remove_git(&source, SOURCE_SCOPE_PROJECT).is_err());
        assert_eq!(std::fs::read_to_string(&canary).unwrap(), "keep");
        std::fs::remove_file(root.join("example.com")).unwrap();
        std::fs::remove_dir(&root).unwrap();
        symlink(&outside, &root).unwrap();
        assert!(manager.validated_git_install_path(&source, SOURCE_SCOPE_PROJECT).is_err());
        assert!(manager.remove_git(&source, SOURCE_SCOPE_PROJECT).is_err());
        assert_eq!(std::fs::read_to_string(canary).unwrap(), "keep");
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(agent);
    }

    #[tokio::test]
    async fn failed_git_checkout_is_not_cached_and_retry_uses_the_exact_ref() {
        let (root, repo, expected_head) = git_fixture("git-checkout-retry", false);
        let cwd = temp_dir("git-checkout-cwd");
        let agent = temp_dir("git-checkout-agent");
        let manager = manager(&cwd, &agent, serde_json::json!({}));
        let missing = local_git_source(&repo, Some("missing-ref"));
        manager.install_git(&missing, SOURCE_SCOPE_PROJECT).await.unwrap_err();
        let target = manager.get_git_install_path(&missing, SOURCE_SCOPE_PROJECT);
        assert!(!Path::new(&target).exists());
        let exact = local_git_source(&repo, Some("exact-ref"));
        manager.install_git(&exact, SOURCE_SCOPE_PROJECT).await.unwrap();
        let output = std::process::Command::new("git")
            .args(["-C", &target, "rev-parse", "HEAD"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), expected_head);
        manager.install_git(&exact, SOURCE_SCOPE_PROJECT).await.unwrap();
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(agent);
    }

    #[tokio::test]
    async fn failed_git_dependency_install_is_cleaned_and_cache_revalidates_dependencies() {
        let (root, repo, expected_head) = git_fixture("git-npm-retry", true);
        let cwd = temp_dir("git-npm-cwd");
        let agent = temp_dir("git-npm-agent");
        let manager = manager(&cwd, &agent, failing_once_npm_settings(&root));
        let source = local_git_source(&repo, Some("exact-ref"));
        manager.install_git(&source, SOURCE_SCOPE_PROJECT).await.unwrap_err();
        let target = manager.get_git_install_path(&source, SOURCE_SCOPE_PROJECT);
        assert!(!Path::new(&target).exists());

        manager.install_git(&source, SOURCE_SCOPE_PROJECT).await.unwrap();
        let dependency = Path::new(&target).join("node_modules/left-pad/package.json");
        assert!(dependency.is_file());
        let calls = std::fs::read_to_string(Path::new(&root).join("npm-calls")).unwrap();
        assert_eq!(calls.trim(), "2");
        manager.install_git(&source, SOURCE_SCOPE_PROJECT).await.unwrap();
        let calls = std::fs::read_to_string(Path::new(&root).join("npm-calls")).unwrap();
        assert_eq!(calls.trim(), "2", "complete cache must not rerun npm");

        std::fs::remove_file(dependency).unwrap();
        manager.install_git(&source, SOURCE_SCOPE_PROJECT).await.unwrap();
        let calls = std::fs::read_to_string(Path::new(&root).join("npm-calls")).unwrap();
        assert_eq!(calls.trim(), "3", "incomplete cached dependencies must be repaired");
        let head = std::process::Command::new("git")
            .args(["-C", &target, "rev-parse", "HEAD"])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&head.stdout).trim(), expected_head);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(cwd);
        let _ = std::fs::remove_dir_all(agent);
    }

    // `${command} ${args.join(" ")} timed out after ${options?.timeoutMs}ms`
    // (package-manager.ts:2381-2403). Startup probes pass NETWORK_TIMEOUT_MS, so an
    // unbounded child would hang headless `--print` startup forever.
    #[tokio::test]
    async fn run_command_capture_times_out_and_kills_the_child() {
        let cwd = temp_dir("capture-timeout");
        let manager = manager(&cwd, &cwd, serde_json::json!({}));
        let command = if cfg!(windows) { "cmd" } else { "sh" };
        // Far longer than the 150ms timeout, short enough that a killed child cannot
        // keep the test binary waiting on its inherited pipes.
        let args = if cfg!(windows) {
            vec!["/c".to_string(), "ping -n 12 127.0.0.1 > NUL".to_string()]
        } else {
            vec!["-c".to_string(), "sleep 12".to_string()]
        };

        let started = std::time::Instant::now();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(20),
            manager.run_command_capture(command, &args, Some(&cwd), Some(150.0), None),
        )
        .await
        .expect("run_command_capture ignored its timeout: it did not return within 20s");
        let elapsed = started.elapsed();

        assert_eq!(
            result,
            Err(format!("{command} {} timed out after 150ms", args.join(" ")))
        );
        assert!(elapsed < std::time::Duration::from_secs(10), "timeout was not prompt: {elapsed:?}");
        let _ = std::fs::remove_dir_all(&cwd);
    }
}
