//! Port of packages/coding-agent/src/core/kernel/bootstrap.ts.
//!
//! Local stand-ins are marked `TODO(slice)` and listed in the slice status file:
//! child-process helpers, dir-lock, config::get_package_dir, uuid and sha2 have
//! no landed crate here yet, so minimal equivalents live in this module.
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;

use super::shared::{KernelError, KernelPythonSkill, SharedPromise};

const BOOTSTRAP_SCHEMA: f64 = 9.0;
const PYTHON_VERSION: &str = "3.11";
const RUNTIME_REQUIREMENT: &str = "prime-agent-runtime";
// Serializes the kernel's user namespace so it can be revived across session
// resume. Internal-only; intentionally not surfaced to the model as an import.
const STATE_SNAPSHOT_REQUIREMENT: &str = "dill";

struct RlmExtraPackage {
    uv_arg: &'static str,
    import_name: &'static str,
    prompt_label: &'static str,
}

const DEFAULT_RLM_EXTRA_PACKAGES: &[RlmExtraPackage] = &[
    RlmExtraPackage { uv_arg: "requests", import_name: "requests", prompt_label: "requests" },
    RlmExtraPackage { uv_arg: "httpx", import_name: "httpx", prompt_label: "httpx" },
    RlmExtraPackage { uv_arg: "pyyaml", import_name: "yaml", prompt_label: "yaml (PyYAML)" },
    RlmExtraPackage { uv_arg: "tomli", import_name: "tomli", prompt_label: "tomli" },
    RlmExtraPackage { uv_arg: "python-dotenv", import_name: "dotenv", prompt_label: "dotenv (python-dotenv)" },
    RlmExtraPackage { uv_arg: "pandas", import_name: "pandas", prompt_label: "pandas" },
    RlmExtraPackage { uv_arg: "numpy", import_name: "numpy", prompt_label: "numpy" },
    RlmExtraPackage { uv_arg: "scipy", import_name: "scipy", prompt_label: "scipy" },
    RlmExtraPackage { uv_arg: "beautifulsoup4", import_name: "bs4", prompt_label: "bs4 (Beautiful Soup)" },
    RlmExtraPackage { uv_arg: "lxml", import_name: "lxml", prompt_label: "lxml" },
    RlmExtraPackage { uv_arg: "pydantic", import_name: "pydantic", prompt_label: "pydantic" },
    RlmExtraPackage { uv_arg: "tyro", import_name: "tyro", prompt_label: "tyro" },
];

pub fn default_rlm_extra_uv_args() -> Vec<String> {
    DEFAULT_RLM_EXTRA_PACKAGES
        .iter()
        .map(|pkg| pkg.uv_arg.to_string())
        .collect()
}

pub fn default_rlm_extra_import_names() -> Vec<String> {
    DEFAULT_RLM_EXTRA_PACKAGES
        .iter()
        .map(|pkg| pkg.import_name.to_string())
        .collect()
}

pub fn default_rlm_extra_import_labels() -> Vec<String> {
    DEFAULT_RLM_EXTRA_PACKAGES
        .iter()
        .map(|pkg| pkg.prompt_label.to_string())
        .collect()
}

const WINDOWS_PATHEXT_DEFAULT: [&str; 4] = [".COM", ".EXE", ".BAT", ".CMD"];

fn windows_supported_executable_extensions() -> HashSet<String> {
    WINDOWS_PATHEXT_DEFAULT
        .iter()
        .map(|extension| extension.to_lowercase())
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchShimInvocation {
    pub args: Vec<String>,
    /// Full replacement environment (base env plus the token variables).
    pub env: Vec<(String, String)>,
}

/// Build a cmd.exe invocation without embedding user-controlled values in its command string.
pub fn build_batch_shim_invocation(
    command: &str,
    args: &[String],
    base_env: &[(String, String)],
    token: Option<String>,
) -> Result<BatchShimInvocation, KernelError> {
    let token = token.unwrap_or_else(random_token);
    if token.is_empty() || !token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(KernelError::new(
            "Windows batch shim token contains unsupported characters",
        ));
    }
    let mut values: Vec<String> = Vec::with_capacity(args.len() + 1);
    values.push(command.to_string());
    values.extend(args.iter().cloned());
    if values.iter().any(|value| {
        value
            .chars()
            .any(|c| c == '"' || c == '\0' || c == '\r' || c == '\n')
    }) {
        return Err(KernelError::new(
            "Windows batch shim paths and arguments cannot contain quotes, NUL, or line breaks",
        ));
    }
    let mut env = base_env.to_vec();
    let mut variables: Vec<String> = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let name = format!("PRIME_AGENT_BATCH_{token}_{index}");
        env.push((name.clone(), value.clone()));
        variables.push(format!("\"%{name}%\""));
    }
    Ok(BatchShimInvocation {
        args: vec![
            "/d".to_string(),
            "/v:off".to_string(),
            "/s".to_string(),
            "/c".to_string(),
            format!("\"{}\"", variables.join(" ")),
        ],
        env,
    })
}

const UV_INSTALL_COMMAND: &str = "curl -LsSf https://astral.sh/uv/install.sh | sh";
const REQUIRED_HARNESS_METHODS: [&str; 13] = [
    "create_memory",
    "update_memory",
    "delete_memory",
    "create_skill",
    "update_skill",
    "delete_skill",
    "create_subagent",
    "update_subagent",
    "delete_subagent",
    "create_prompt_note",
    "update_prompt_note",
    "delete_prompt_note",
    "record_refinement",
];

fn runtime_ready_check() -> String {
    let harness_methods = serde_json::to_string(&REQUIRED_HARNESS_METHODS).unwrap_or_else(|_| "[]".to_string());
    format!(
        "import inspect; import rlm; from rlm import McpIntegration; import rlm.mcp as mcp; from rlm.harness import HarnessEntry; _harness_methods = {harness_methods}; assert callable(mcp.list_tools); assert callable(mcp.call_tool); assert hasattr(rlm, 'run'); assert callable(rlm); assert hasattr(rlm, 'rlm'); assert callable(rlm.rlm); assert callable(rlm.host_request); assert callable(rlm.find_models); assert callable(rlm.rlm.find_models); assert callable(rlm.create_session); assert callable(rlm.rlm.create_session); assert hasattr(rlm, 'harness'); assert hasattr(rlm, 'get_harness_state'); assert hasattr(rlm.rlm, 'harness'); assert hasattr(rlm.rlm, 'get_harness_state'); assert all(callable(getattr(_harness, _method, None)) for _harness in (rlm.harness, rlm.rlm.harness) for _method in _harness_methods); assert 'reference' in HarnessEntry.__dataclass_fields__; assert 'scope' in HarnessEntry.__dataclass_fields__; assert 'reference' in inspect.signature(rlm.harness.create_skill).parameters; assert 'reference' in inspect.signature(rlm.harness.update_skill).parameters; assert 'global_' in inspect.signature(rlm.harness.create_memory).parameters; assert 'global_' in inspect.signature(rlm.get_harness_state).parameters; assert not hasattr(rlm, 'background'); assert not hasattr(rlm.rlm, 'background'); from rlm.bash import BashHandle, BashResult; assert callable(rlm.bash); assert all(callable(getattr(BashHandle, _m, None)) for _m in ('tail', 'output', 'poll', 'kill')); assert {{'exit_code', 'output', 'duration'}} <= set(BashResult.__dataclass_fields__); import rlm.repl as _repl; assert callable(_repl.main); assert callable(_repl.emit); assert callable(_repl.host_request); assert callable(_repl.is_active); assert _repl.PROTOCOL_VERSION == 4; assert callable(rlm.emit); assert not hasattr(rlm, 'HOST_COMM_TARGET'); assert not hasattr(mcp, 'install_shutdown_hook')"
    )
}

const BOOTSTRAP_VERSION_FILE: &str = ".bootstrap-version";
const BOOTSTRAP_LOCK_NAME: &str = ".bootstrap.lock";
const BOOTSTRAP_LOCK_RETRY_MS: u64 = 100;
const BOOTSTRAP_LOCK_STALE_WITHOUT_PID_MS: u64 = 30_000;

pub type KernelBootstrapProgressHandler = Arc<dyn Fn(&str) + Send + Sync>;

#[derive(Clone, Default)]
pub struct EnsureKernelPythonOptions {
    pub python_skills: Option<Vec<KernelPythonSkill>>,
    pub on_progress: Option<KernelBootstrapProgressHandler>,
}

impl std::fmt::Debug for EnsureKernelPythonOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsureKernelPythonOptions")
            .field("python_skills", &self.python_skills.as_ref().map(|s| s.len()))
            .field("on_progress", &self.on_progress.is_some())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BootstrapPythonSkill {
    import_name: String,
    package_path: String,
    pyproject_path: String,
    pyproject_hash: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BootstrapVersion {
    schema: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_uv_args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    python_skills: Option<Vec<BootstrapPythonSkill>>,
}

// ---------------------------------------------------------------------------
// Small process / fs helpers (TODO(slice): needs utils/child-process)
// ---------------------------------------------------------------------------

fn is_windows() -> bool {
    cfg!(windows)
}

fn path_delimiter() -> char {
    if is_windows() {
        ';'
    } else {
        ':'
    }
}

fn home_dir() -> PathBuf {
    match std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        Ok(value) if !value.is_empty() => PathBuf::from(value),
        _ => std::env::temp_dir(),
    }
}

/// A UUID-like 32-hex-character token; the TypeScript uses `randomUUID().replaceAll("-", "")`.
fn random_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64) << 32
        ^ COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut out = String::with_capacity(32);
    while out.len() < 32 {
        // xorshift64*: enough entropy for a lock/token name, no dependency needed.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push_str(&format!("{:016x}", state.wrapping_mul(0x2545_F491_4F6C_DD1D)));
    }
    out.truncate(32);
    out
}

fn exists(file_path: &str) -> bool {
    std::fs::metadata(file_path).is_ok()
}

fn is_executable(file_path: &str) -> bool {
    let Ok(metadata) = std::fs::metadata(file_path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        return metadata.permissions().mode() & 0o111 != 0;
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn expand_home(file_path: &str) -> String {
    if file_path == "~" {
        return home_dir().to_string_lossy().into_owned();
    }
    if let Some(rest) = file_path.strip_prefix("~/").or_else(|| file_path.strip_prefix("~\\")) {
        return home_dir().join(rest).to_string_lossy().into_owned();
    }
    file_path.to_string()
}

// ---------------------------------------------------------------------------
// SHA-256 (TODO(slice): the workspace has no sha2 dependency yet)
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

/// Streaming SHA-256 so a large runtime tree never has to be held in memory twice.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Sha256 {
    fn new() -> Self {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        let mut offset = 0;
        if self.buffered > 0 {
            let take = std::cmp::min(64 - self.buffered, data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            offset = take;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while offset + 64 <= data.len() {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[offset..offset + 64]);
            self.compress(&block);
            offset += 64;
        }
        if offset < data.len() {
            let rest = data.len() - offset;
            self.buffer[..rest].copy_from_slice(&data[offset..]);
            self.buffered = rest;
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for index in 0..16 {
            w[index] = u32::from_be_bytes([
                block[index * 4],
                block[index * 4 + 1],
                block[index * 4 + 2],
                block[index * 4 + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18) ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19) ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    fn finish(mut self) -> [u8; 32] {
        let bit_length = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buffered != 56 {
            self.update(&[0]);
        }
        let mut tail = [0u8; 8];
        tail.copy_from_slice(&bit_length.to_be_bytes());
        self.length = self.length.wrapping_sub(8);
        self.update(&tail);
        let mut out = [0u8; 32];
        for (index, word) in self.state.iter().enumerate() {
            out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(data);
    let digest = hash.finish();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn file_content_hash(file_path: &str) -> String {
    match std::fs::read(file_path) {
        Ok(bytes) => format!("sha256:{}", sha256_hex(&bytes)),
        Err(_) => "unreadable".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Python skill discovery
// ---------------------------------------------------------------------------

fn normalize_python_skills(python_skills: Option<&[KernelPythonSkill]>) -> Vec<BootstrapPythonSkill> {
    let mut by_key: Vec<(String, BootstrapPythonSkill)> = Vec::new();
    fn add_skill(
        skill: &KernelPythonSkill,
        by_key: &mut Vec<(String, BootstrapPythonSkill)>,
    ) {
        let package_path = PathBuf::from(&skill.package_path);
        let pyproject_path = PathBuf::from(&skill.pyproject_path);
        let package_path = if package_path.is_absolute() {
            package_path
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(package_path)
        };
        let pyproject_path = if pyproject_path.is_absolute() {
            pyproject_path
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(pyproject_path)
        };
        let key = format!(
            "{}\u{0}{}",
            skill.import_name,
            package_path.to_string_lossy()
        );
        if by_key.iter().any(|(existing, _)| *existing == key) {
            return;
        }
        let bootstrap_skill = BootstrapPythonSkill {
            import_name: skill.import_name.clone(),
            package_path: package_path.to_string_lossy().into_owned(),
            pyproject_path: pyproject_path.to_string_lossy().into_owned(),
            pyproject_hash: file_content_hash(&pyproject_path.to_string_lossy()),
        };
        by_key.push((key, bootstrap_skill.clone()));
        for dependency_name in read_python_skill_dependency_names(&bootstrap_skill) {
            if let Some(sibling) = resolve_sibling_python_skill_dependency(&bootstrap_skill, &dependency_name) {
                add_skill(
                    &KernelPythonSkill {
                        import_name: sibling.import_name,
                        package_path: sibling.package_path,
                        pyproject_path: sibling.pyproject_path,
                        name: String::new(),
                    },
                    by_key,
                );
            }
        }
    }
    for skill in python_skills.unwrap_or(&[]) {
        add_skill(skill, &mut by_key);
    }
    let mut skills: Vec<BootstrapPythonSkill> = by_key.into_iter().map(|(_, skill)| skill).collect();
    skills.sort_by(|a, b| {
        let package_compare = a.package_path.cmp(&b.package_path);
        if package_compare != std::cmp::Ordering::Equal {
            return package_compare;
        }
        a.import_name.cmp(&b.import_name)
    });
    skills
}

fn read_toml_project_section(pyproject_path: &str) -> Option<String> {
    let text = std::fs::read_to_string(pyproject_path).ok()?;
    let mut section_start: Option<usize> = None;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed == "[project]" {
            section_start = Some(offset + line.len());
            break;
        }
        offset += line.len();
    }
    let section_start = section_start?;
    let rest = &text[section_start..];
    // Next section header: a line whose first non-space character is '['.
    let mut next_section: Option<usize> = None;
    let mut offset = 0usize;
    for line in rest.split_inclusive('\n') {
        if line.trim_start().starts_with('[') {
            next_section = Some(offset);
            break;
        }
        offset += line.len();
    }
    Some(match next_section {
        Some(index) => rest[..index].to_string(),
        None => rest.to_string(),
    })
}

fn read_python_skill_project_name(skill: &BootstrapPythonSkill) -> String {
    let project_section = read_toml_project_section(&skill.pyproject_path);
    let name = project_section.as_deref().and_then(|section| {
        for line in section.split_inclusive('\n') {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("name") {
                continue;
            }
            let rest = trimmed["name".len()..].trim_start();
            let Some(rest) = rest.strip_prefix('=') else { continue };
            let rest = rest.trim_start();
            let quote = rest.chars().next()?;
            if quote != '"' && quote != '\'' {
                continue;
            }
            let after = &rest[1..];
            let end = after.find(quote)?;
            return Some(after[..end].to_string());
        }
        None
    });
    let name = name.unwrap_or_default();
    let trimmed = name.trim().to_string();
    if trimmed.is_empty() {
        skill.import_name.replace('_', "-")
    } else {
        trimmed
    }
}

fn parse_dependency_package_name(dependency: &str) -> Option<String> {
    let without_marker = dependency.split(';').next().unwrap_or("").trim();
    if without_marker.is_empty() {
        return None;
    }
    let name: String = without_marker
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '-')
        .collect();
    if name.is_empty() {
        return None;
    }
    Some(name.replace('_', "-").to_lowercase())
}

/// `findTomlArrayEnd(text, startIndex)`.
fn find_toml_array_end(text: &str, start_index: usize) -> Option<usize> {
    let bytes: Vec<char> = text.chars().collect();
    let mut in_quote: Option<char> = None;
    let mut escaped = false;
    let mut index = start_index;
    while index < bytes.len() {
        let character = bytes[index];
        if let Some(quote) = in_quote {
            if escaped {
                escaped = false;
                index += 1;
                continue;
            }
            if character == '\\' {
                escaped = true;
                index += 1;
                continue;
            }
            if character == quote {
                in_quote = None;
            }
            index += 1;
            continue;
        }
        if character == '"' || character == '\'' {
            in_quote = Some(character);
            index += 1;
            continue;
        }
        if character == ']' {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn read_python_skill_dependency_names(skill: &BootstrapPythonSkill) -> BTreeSet<String> {
    let Some(project_section) = read_toml_project_section(&skill.pyproject_path) else {
        return BTreeSet::new();
    };
    let Some(dependencies_start) = find_line_prefixed(&project_section, "dependencies", '[') else {
        return BTreeSet::new();
    };
    let Some(array_start) = project_section[dependencies_start..]
        .find('[')
        .map(|i| dependencies_start + i)
    else {
        return BTreeSet::new();
    };
    let Some(array_end) = find_toml_array_end(&project_section, array_start + 1) else {
        return BTreeSet::new();
    };
    let dependencies_array: String = project_section
        .chars()
        .skip(array_start)
        .take(array_end - array_start + 1)
        .collect();

    let mut dependencies = BTreeSet::new();
    for dependency in quoted_strings_in(&dependencies_array) {
        let dependency = dependency.replace("\\\"", "\"").replace("\\'", "'");
        if let Some(name) = parse_dependency_package_name(&dependency) {
            dependencies.insert(name);
        }
    }
    dependencies
}

/// Find the offset of a line matching `^\s*<key>\s*=\s*<marker>`.
fn find_line_prefixed(text: &str, key: &str, marker: char) -> Option<usize> {
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                if rest.trim_start().starts_with(marker) {
                    return Some(offset + line.len() - trimmed.len());
                }
            }
        }
        offset += line.len();
    }
    None
}

/// All double- or single-quoted strings in a TOML array body, escapes preserved.
fn quoted_strings_in(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let character = chars[index];
        if character != '"' && character != '\'' {
            index += 1;
            continue;
        }
        let quote = character;
        index += 1;
        let mut value = String::new();
        let mut closed = false;
        while index < chars.len() {
            let current = chars[index];
            if current == '\\' {
                if index + 1 < chars.len() {
                    value.push('\\');
                    value.push(chars[index + 1]);
                    index += 2;
                    continue;
                }
                index += 1;
                continue;
            }
            if current == quote {
                closed = true;
                index += 1;
                break;
            }
            value.push(current);
            index += 1;
        }
        if closed {
            out.push(value);
        }
    }
    out
}

fn resolve_sibling_python_skill_dependency(
    skill: &BootstrapPythonSkill,
    dependency_name: &str,
) -> Option<BootstrapPythonSkill> {
    let siblings_dir = Path::new(&skill.package_path).parent()?.to_path_buf();
    let entries = std::fs::read_dir(&siblings_dir).ok()?;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else { continue };
        if !file_type.is_dir() {
            continue;
        }
        let package_path = siblings_dir.join(entry.file_name());
        let pyproject_path = package_path.join("pyproject.toml");
        if !pyproject_path.exists() {
            continue;
        }
        let dependency = BootstrapPythonSkill {
            import_name: entry.file_name().to_string_lossy().replace('-', "_"),
            package_path: package_path.to_string_lossy().into_owned(),
            pyproject_path: pyproject_path.to_string_lossy().into_owned(),
            pyproject_hash: file_content_hash(&pyproject_path.to_string_lossy()),
        };
        if read_python_skill_project_name(&dependency)
            .replace('_', "-")
            .to_lowercase()
            == dependency_name
        {
            return Some(dependency);
        }
    }
    None
}

fn sort_python_skills_for_install(python_skills: &[BootstrapPythonSkill]) -> Vec<BootstrapPythonSkill> {
    let mut by_project_name: Vec<(String, BootstrapPythonSkill)> = Vec::new();
    let mut original_index: HashMap<usize, usize> = HashMap::new();
    for (index, skill) in python_skills.iter().enumerate() {
        original_index.insert(index, index);
        let key = read_python_skill_project_name(skill).replace('_', "-").to_lowercase();
        // TypeScript Map.set overwrites: the LAST duplicate project name wins.
        if let Some(existing) = by_project_name.iter_mut().find(|(existing, _)| *existing == key) {
            existing.1 = skill.clone();
        } else {
            by_project_name.push((key, skill.clone()));
        }
    }

    let index_of = |skill: &BootstrapPythonSkill| -> usize {
        python_skills
            .iter()
            .position(|candidate| candidate == skill)
            .unwrap_or(0)
    };

    let mut dependencies_by_skill: Vec<(usize, Vec<BootstrapPythonSkill>)> = Vec::new();
    for skill in python_skills {
        let mut dependencies = Vec::new();
        for dependency_name in read_python_skill_dependency_names(skill) {
            let found = by_project_name
                .iter()
                .find(|(name, _)| *name == dependency_name)
                .map(|(_, skill)| skill.clone())
                .or_else(|| resolve_sibling_python_skill_dependency(skill, &dependency_name));
            if let Some(found) = found {
                dependencies.push(found);
            }
        }
        dependencies_by_skill.push((index_of(skill), dependencies));
    }

    let mut pending: Vec<usize> = (0..python_skills.len()).collect();
    let mut sorted: Vec<BootstrapPythonSkill> = Vec::new();
    while !pending.is_empty() {
        let mut progressed = false;
        let ordered: Vec<usize> = {
            let mut ordered = pending.clone();
            ordered.sort_by_key(|index| original_index.get(index).copied().unwrap_or(0));
            ordered
        };
        for index in ordered {
            let dependencies = dependencies_by_skill
                .iter()
                .find(|(owner, _)| *owner == index)
                .map(|(_, dependencies)| dependencies.clone())
                .unwrap_or_default();
            let blocked = dependencies.iter().any(|dependency| {
                pending
                    .iter()
                    .any(|candidate| python_skills[*candidate] == *dependency)
            });
            if blocked {
                continue;
            }
            sorted.push(python_skills[index].clone());
            pending.retain(|candidate| *candidate != index);
            progressed = true;
        }
        if !progressed {
            // Cyclic local skill dependencies cannot be topologically ordered; keep a
            // deterministic order and let uv surface the packaging error if needed.
            let mut remaining: Vec<BootstrapPythonSkill> =
                pending.iter().map(|index| python_skills[*index].clone()).collect();
            remaining.sort_by(|a, b| a.package_path.cmp(&b.package_path));
            sorted.extend(remaining);
            break;
        }
    }
    sorted
}

fn format_python_skill_install_args(skill: &BootstrapPythonSkill) -> Vec<String> {
    vec!["--editable".to_string(), skill.package_path.clone()]
}

fn ensure_kernel_python_key(python_skills: &[BootstrapPythonSkill]) -> String {
    [
        std::env::var("PRIME_AGENT_KERNEL_PYTHON").unwrap_or_default(),
        std::env::var("PRIME_AGENT_KERNEL_VENV").unwrap_or_default(),
        std::env::var("HOME").unwrap_or_default(),
        std::env::var("XDG_DATA_HOME").unwrap_or_default(),
        serde_json::to_string(python_skills).unwrap_or_default(),
    ]
    .join("\u{0}")
}

pub fn get_kernel_venv_dir() -> String {
    if let Ok(override_value) = std::env::var("PRIME_AGENT_KERNEL_VENV") {
        if !override_value.is_empty() {
            return resolve_path(&expand_home(&override_value)).to_string_lossy().into_owned();
        }
    }
    home_dir()
        .join(".prime")
        .join("agent")
        .join("kernel-venv")
        .to_string_lossy()
        .into_owned()
}

fn resolve_path(value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn get_xdg_kernel_venv_dir() -> String {
    let data_home = match std::env::var("XDG_DATA_HOME") {
        Ok(value) if !value.is_empty() => resolve_path(&expand_home(&value)),
        _ => home_dir().join(".local").join("share"),
    };
    data_home
        .join("prime")
        .join("agent")
        .join("kernel-venv")
        .to_string_lossy()
        .into_owned()
}

async fn resolve_writable_kernel_venv_dir() -> Result<String, KernelError> {
    let primary = get_kernel_venv_dir();
    if let Some(parent) = Path::new(&primary).parent() {
        if tokio::fs::create_dir_all(parent).await.is_ok() {
            return Ok(primary);
        }
    }
    if std::env::var("PRIME_AGENT_KERNEL_VENV")
        .map(|value| !value.is_empty())
        .unwrap_or(false)
    {
        return Err(KernelError::new(format!(
            "couldn't create kernel venv parent directory for {primary}"
        )));
    }

    let fallback = get_xdg_kernel_venv_dir();
    if let Some(parent) = Path::new(&fallback).parent() {
        if tokio::fs::create_dir_all(parent).await.is_ok() {
            return Ok(fallback);
        }
    }
    Err(KernelError::new(format!(
        "couldn't create kernel venv directory at {primary} or {fallback}; set PRIME_AGENT_KERNEL_PYTHON to a python with a current prime-agent-runtime installed."
    )))
}

fn is_batch_shim(command: &str) -> bool {
    if !is_windows() {
        return false;
    }
    let lower = command.to_lowercase();
    lower.ends_with(".cmd") || lower.ends_with(".bat")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunStdio {
    Ignore,
    Inherit,
}

/// `run(command, args, options)`.
async fn run(command: &str, args: Vec<String>, stdio: RunStdio) -> Result<(), KernelError> {
    // CPython must read UTF-8 .pth files even under a Windows legacy code page.
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    if is_windows() {
        env.push(("PYTHONUTF8".to_string(), "1".to_string()));
    }
    let batch = if is_batch_shim(command) {
        Some(build_batch_shim_invocation(command, &args, &env, None)?)
    } else {
        None
    };

    let (program, program_args, program_env) = match &batch {
        Some(batch) => (
            std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".to_string()),
            batch.args.clone(),
            batch.env.clone(),
        ),
        None => (command.to_string(), args.clone(), env.clone()),
    };

    let command_display = command.to_string();
    let args_display = args.join(" ");
    let outcome = tokio::task::spawn_blocking(move || {
        let mut process = std::process::Command::new(&program);
        process.env_clear();
        for (key, value) in &program_env {
            process.env(key, value);
        }
        process.stdin(Stdio::null());
        match stdio {
            RunStdio::Ignore => {
                process.stdout(Stdio::null());
                process.stderr(Stdio::null());
                // Bootstrap helpers (uv/python probes) are noninteractive;
                // headless parents must not allocate a console window.
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                    process.creation_flags(CREATE_NO_WINDOW);
                }
            }
            RunStdio::Inherit => {
                process.stdout(Stdio::inherit());
                process.stderr(Stdio::inherit());
            }
        }
        if let Some(batch) = &batch {
            // windowsVerbatimArguments: pass the cmd string through untouched.
            for (index, arg) in program_args.iter().enumerate() {
                if index + 1 == program_args.len() {
                    #[cfg(windows)]
                    {
                        use std::os::windows::process::CommandExt;
                        process.raw_arg(arg);
                    }
                    #[cfg(not(windows))]
                    {
                        process.arg(arg);
                    }
                } else {
                    process.arg(arg);
                }
            }
            let _ = batch;
        } else {
            process.args(&program_args);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            process.creation_flags(CREATE_NO_WINDOW);
        }
        process.status()
    })
    .await
    .map_err(|error| KernelError::new(error.to_string()))?;

    let status = outcome.map_err(|error| KernelError::new(error.to_string()))?;
    if status.success() {
        return Ok(());
    }
    let reason = match status.code() {
        Some(code) => format!("exit code {code}"),
        None => "signal".to_string(),
    };
    Err(KernelError::new(format!(
        "{command_display} {args_display} failed with {reason}"
    )))
}

async fn python_imports(python: &str, module_name: &str) -> bool {
    run(
        python,
        vec!["-c".to_string(), format!("import {module_name}")],
        RunStdio::Ignore,
    )
    .await
    .is_ok()
}

async fn has_prime_agent_runtime(python: &str) -> bool {
    run(
        python,
        vec!["-c".to_string(), runtime_ready_check()],
        RunStdio::Ignore,
    )
    .await
    .is_ok()
}

async fn missing_rlm_extra_import_labels(python: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for pkg in DEFAULT_RLM_EXTRA_PACKAGES {
        if !python_imports(python, pkg.import_name).await {
            missing.push(pkg.prompt_label.to_string());
        }
    }
    missing
}

async fn missing_python_skill_import_labels(
    python: &str,
    python_skills: &[KernelPythonSkill],
) -> Vec<String> {
    let mut missing = Vec::new();
    for skill in python_skills {
        if !python_imports(python, &skill.import_name).await {
            missing.push(format!("{} ({})", skill.name, skill.import_name));
        }
    }
    missing
}

fn report_progress(options: &EnsureKernelPythonOptions, message: &str) {
    if let Some(on_progress) = &options.on_progress {
        on_progress(message);
        return;
    }
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{message}");
}

fn bootstrap_lock_dir(venv: &str) -> String {
    let path = Path::new(venv);
    let name = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    match path.parent() {
        Some(parent) => parent
            .join(format!("{name}{BOOTSTRAP_LOCK_NAME}"))
            .to_string_lossy()
            .into_owned(),
        None => format!("{name}{BOOTSTRAP_LOCK_NAME}"),
    }
}

async fn lock_missing_pid_is_stale(lock_dir: &str) -> bool {
    match tokio::fs::metadata(lock_dir).await {
        Ok(metadata) => {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|value| value.as_millis() as f64)
                .unwrap_or(0.0);
            super::shared::now_ms() - modified > BOOTSTRAP_LOCK_STALE_WITHOUT_PID_MS as f64
        }
        Err(_) => false,
    }
}

/// `tryAcquireDirLock(lockDir, ownerAlive)` (TODO(slice): needs utils/dir-lock).
async fn acquire_bootstrap_lock(
    venv: &str,
) -> Result<Arc<dyn Fn() -> BoxFutureLocal + Send + Sync>, KernelError> {
    let lock_dir = bootstrap_lock_dir(venv);
    if let Some(parent) = Path::new(&lock_dir).parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    // G-01/G-02/G-04: delegate to the landed utils/dir_lock port, which reads
    // the REAL file identity (GetFileInformationByHandle) so stale locks are
    // reclaimable on NTFS, propagates hard I/O errors (fail fast), and writes
    // 0o600 owner files on POSIX. A dir/symlink at the lock path is a
    // documented protected difference: it is never reclaimed.
    let lock_dir_static: &'static str = Box::leak(lock_dir.clone().into_boxed_str());
    for _ in 0..u32::MAX {
        let owner_alive = move |owner_pid: Option<i32>| {
            async move {
                match owner_pid {
                    Some(pid) => is_process_alive(pid as u32).await,
                    None => !lock_missing_pid_is_stale(lock_dir_static).await,
                }
            }
        };
        match crate::utils::dir_lock::try_acquire_dir_lock(&lock_dir, owner_alive).await {
            Ok(crate::utils::dir_lock::DirLockAttempt::Acquired) => {
                let lock_dir = lock_dir.clone();
                return Ok(Arc::new(move || {
                    let lock_dir = lock_dir.clone();
                    Box::pin(async move {
                        let _ = tokio::fs::remove_dir_all(&lock_dir).await;
                        let _ = tokio::fs::remove_file(&lock_dir).await;
                    })
                }));
            }
            Ok(crate::utils::dir_lock::DirLockAttempt::Held) => {
                tokio::time::sleep(std::time::Duration::from_millis(BOOTSTRAP_LOCK_RETRY_MS)).await;
            }
            Ok(crate::utils::dir_lock::DirLockAttempt::Reclaimed) => {}
            // G-02: a hard I/O failure (e.g. access-denied lock directory)
            // must fail fast with an actionable error, not spin as Held.
            Err(error) => return Err(KernelError::new(error.to_string())),
        }
    }
    Err(KernelError::new("could not acquire bootstrap lock"))
}

type BoxFutureLocal = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
/// `kill(0)` probe plus the zombie check.
async fn is_process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        let stat_path = format!("/proc/{pid}/stat");
        match std::fs::read_to_string(&stat_path) {
            Ok(stat) => {
                let after = stat.rfind(')').map(|index| &stat[index + 2..]).unwrap_or("");
                return !after.starts_with('Z');
            }
            Err(_) => return false,
        }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let output = tokio::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "stat="])
            .output()
            .await;
        return match output {
            Ok(output) => {
                let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
                !state.is_empty() && !state.starts_with('Z')
            }
            Err(_) => false,
        };
    }
    #[cfg(windows)]
    {
        let mut tasklist = tokio::process::Command::new("tasklist");
        tasklist
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"]);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            tasklist.creation_flags(CREATE_NO_WINDOW);
        }
        let output = tasklist.output().await;
        return match output {
            Ok(output) => {
                let text = String::from_utf8_lossy(&output.stdout);
                text.contains(&format!("\"{pid}\""))
            }
            Err(_) => false,
        };
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}
/// Try a bare command followed by supported PATHEXT extensions in the configured order.
pub fn windows_executable_candidates(name: &str, pathext: Option<&str>) -> Vec<String> {
    let supported = windows_supported_executable_extensions();
    let extensions: Vec<String> = pathext
        .unwrap_or("")
        .split(';')
        .map(|ext| ext.trim().to_lowercase())
        .filter(|ext| supported.contains(ext))
        .collect();
    let lower_name = name.to_lowercase();
    if WINDOWS_PATHEXT_DEFAULT
        .iter()
        .any(|ext| lower_name.ends_with(&ext.to_lowercase()))
    {
        return vec![name.to_string()];
    }
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(lower_name);
    let mut candidates = vec![name.to_string()];
    let extension_list: Vec<String> = if extensions.is_empty() {
        WINDOWS_PATHEXT_DEFAULT.iter().map(|ext| ext.to_lowercase()).collect()
    } else {
        extensions
    };
    for ext in extension_list {
        let candidate = format!("{name}{ext}");
        if seen.contains(&candidate.to_lowercase()) {
            continue;
        }
        seen.insert(candidate.to_lowercase());
        candidates.push(candidate);
    }
    candidates
}

async fn find_executable(name: &str) -> Option<String> {
    let path_value = std::env::var("PATH").ok()?;
    let candidates = if is_windows() {
        windows_executable_candidates(name, std::env::var("PATHEXT").ok().as_deref())
    } else {
        vec![name.to_string()]
    };
    for dir in path_value.split(path_delimiter()) {
        if dir.is_empty() {
            continue;
        }
        for candidate in &candidates {
            let full_path = Path::new(dir).join(candidate);
            let full_path = full_path.to_string_lossy().into_owned();
            if is_executable(&full_path) {
                return Some(full_path);
            }
        }
    }
    None
}

async fn ensure_uv(options: &EnsureKernelPythonOptions) -> Result<String, KernelError> {
    if let Some(from_path) = find_executable("uv").await {
        return Ok(from_path);
    }

    let local_uv = home_dir()
        .join(".local")
        .join("bin")
        .join(if is_windows() { "uv.exe" } else { "uv" })
        .to_string_lossy()
        .into_owned();
    if is_executable(&local_uv) {
        return Ok(local_uv);
    }

    let install_flag = std::env::var("PRIME_AGENT_INSTALL_UV").unwrap_or_default();
    let should_install_uv = install_flag == "1" || (options.on_progress.is_none() && confirm_uv_install().await);
    if !should_install_uv {
        return Err(KernelError::new(format!(
            "uv is required to set up the Python kernel. Install uv yourself: {UV_INSTALL_COMMAND}, or set PRIME_AGENT_INSTALL_UV=1 to let Optimus run that installer."
        )));
    }

    report_progress(options, "\u{203a} installing uv (one-time)\u{2026}");
    if let Err(error) = run(
        "sh",
        vec!["-c".to_string(), UV_INSTALL_COMMAND.to_string()],
        if options.on_progress.is_some() {
            RunStdio::Ignore
        } else {
            RunStdio::Inherit
        },
    )
    .await
    {
        return Err(KernelError::new(format!(
            "couldn't install uv from astral.sh; install it yourself: {UV_INSTALL_COMMAND}, then re-run optimus-agent. {error}"
        )));
    }

    if is_executable(&local_uv) {
        return Ok(local_uv);
    }
    if let Some(installed_from_path) = find_executable("uv").await {
        return Ok(installed_from_path);
    }
    Err(KernelError::new(
        "uv install completed but binary not found at ~/.local/bin/uv",
    ))
}

async fn confirm_uv_install() -> bool {
    if std::env::var("PRIME_AGENT_INSTALL_UV").unwrap_or_default() == "0" {
        return false;
    }
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return false;
    }
    use std::io::Write;
    let _ = write!(
        std::io::stderr(),
        "Optimus needs uv to set up Python. Install uv from astral.sh now? [Y/n] "
    );
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    let answer = line.trim().to_lowercase();
    answer != "n" && answer != "no"
}

async fn read_bootstrap_version(venv: &str) -> Option<BootstrapVersion> {
    let raw = tokio::fs::read_to_string(Path::new(venv).join(BOOTSTRAP_VERSION_FILE))
        .await
        .ok()?;
    let parsed: Value = serde_json::from_str(&raw).ok()?;
    let object = parsed.as_object()?;
    let schema = object.get("schema")?.as_f64()?;
    let extra_uv_args = match object.get("extraUvArgs") {
        Some(Value::Array(items)) if items.iter().all(|item| item.is_string()) => Some(
            items
                .iter()
                .filter_map(|item| item.as_str().map(|text| text.to_string()))
                .collect(),
        ),
        _ => None,
    };
    let mut python_skills: Option<Vec<BootstrapPythonSkill>> = None;
    if let Some(Value::Array(items)) = object.get("pythonSkills") {
        let mut skills = Vec::new();
        for item in items {
            let Some(skill) = item.as_object() else { return None };
            let import_name = skill.get("importName").and_then(|value| value.as_str())?;
            let package_path = skill.get("packagePath").and_then(|value| value.as_str())?;
            let pyproject_path = skill.get("pyprojectPath").and_then(|value| value.as_str())?;
            let pyproject_hash = skill.get("pyprojectHash").and_then(|value| value.as_str())?;
            skills.push(BootstrapPythonSkill {
                import_name: import_name.to_string(),
                package_path: package_path.to_string(),
                pyproject_path: pyproject_path.to_string(),
                pyproject_hash: pyproject_hash.to_string(),
            });
        }
        python_skills = Some(skills);
    }
    Some(BootstrapVersion {
        schema,
        runtime: object.get("runtime").and_then(|value| value.as_str()).map(|v| v.to_string()),
        snapshot: object.get("snapshot").and_then(|value| value.as_str()).map(|v| v.to_string()),
        extra_uv_args,
        python_skills,
    })
}

fn extra_uv_args_match(a: Option<&Vec<String>>, b: Option<&Vec<String>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn python_skills_match(a: Option<&Vec<BootstrapPythonSkill>>, b: &[BootstrapPythonSkill]) -> bool {
    let left: &[BootstrapPythonSkill] = a.map(|value| value.as_slice()).unwrap_or(&[]);
    if left.len() != b.len() {
        return false;
    }
    left.iter().zip(b.iter()).all(|(skill, expected)| {
        skill.import_name == expected.import_name
            && skill.package_path == expected.package_path
            && skill.pyproject_path == expected.pyproject_path
            && skill.pyproject_hash == expected.pyproject_hash
    })
}

fn bootstrap_base_version_current(version: Option<&BootstrapVersion>, runtime_identity: &str) -> bool {
    let Some(version) = version else { return false };
    version.schema == BOOTSTRAP_SCHEMA
        && version.runtime.as_deref() == Some(runtime_identity)
        && version.snapshot.as_deref() == Some(STATE_SNAPSHOT_REQUIREMENT)
        && extra_uv_args_match(version.extra_uv_args.as_ref(), Some(&default_rlm_extra_uv_args()))
}

fn bootstrap_version_current(
    version: Option<&BootstrapVersion>,
    runtime_identity: &str,
    python_skills: &[BootstrapPythonSkill],
) -> bool {
    version.is_some()
        && bootstrap_base_version_current(version, runtime_identity)
        && python_skills_match(version.and_then(|value| value.python_skills.as_ref()), python_skills)
}

async fn write_bootstrap_version(
    venv: &str,
    runtime_identity: &str,
    python_skills: &[BootstrapPythonSkill],
) -> Result<(), KernelError> {
    let version = BootstrapVersion {
        schema: BOOTSTRAP_SCHEMA,
        runtime: Some(runtime_identity.to_string()),
        snapshot: Some(STATE_SNAPSHOT_REQUIREMENT.to_string()),
        extra_uv_args: Some(default_rlm_extra_uv_args()),
        python_skills: Some(python_skills.to_vec()),
    };
    let body = format!("{}\n", serde_json::to_string(&version).unwrap_or_default());
    tokio::fs::write(Path::new(venv).join(BOOTSTRAP_VERSION_FILE), body)
        .await
        .map_err(|error| KernelError::new(error.to_string()))
}

/// `getPackageDir()` (TODO(slice): needs config::get_package_dir).
fn get_package_dir() -> String {
    if let Ok(env_dir) = std::env::var("PI_PACKAGE_DIR") {
        if !env_dir.is_empty() {
            return expand_home(&env_dir);
        }
    }
    let mut dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    loop {
        if dir.join("package.json").exists() {
            return dir.to_string_lossy().into_owned();
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent.to_path_buf(),
            _ => break,
        }
    }
    dir.to_string_lossy().into_owned()
}

fn module_dir() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .to_string_lossy()
        .into_owned()
}

fn runtime_candidate_dirs() -> Vec<String> {
    let module_dir = module_dir();
    // dist/prime-agent-runtime is listed first deliberately: it is the only path stable
    // across every shipped layout (dist/, dist/bundle/, bun), where import.meta.url-relative
    // resolution breaks. `npm run build` rebuilds it from live source (copy-assets does
    // rm -rf + cp), so the staleness hash still refreshes on every build. The relative
    // paths below cover running from source (tsx) where dist/ hasn't been built.
    vec![
        Path::new(&get_package_dir())
            .join("dist")
            .join("prime-agent-runtime")
            .to_string_lossy()
            .into_owned(),
        // Native bundles keep packages/coding-agent and prime-agent-runtime
        // under one release root; the executable lives in that root's bin/.
        resolve_relative(&get_package_dir(), &["..", "..", "prime-agent-runtime"]),
        resolve_relative(&module_dir, &["..", "prime-agent-runtime"]),
        resolve_relative(&module_dir, &["..", "..", "prime-agent-runtime"]),
        resolve_relative(
            &module_dir,
            &["..", "..", "..", "..", "..", "prime-agent-runtime"],
        ),
    ]
}

fn resolve_relative(base: &str, segments: &[&str]) -> String {
    let mut path = PathBuf::from(base);
    for segment in segments {
        path.push(segment);
    }
    normalize_path(&path).to_string_lossy().into_owned()
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

async fn resolve_runtime_source_dir() -> Option<String> {
    for candidate in runtime_candidate_dirs() {
        if tokio::fs::metadata(Path::new(&candidate).join("pyproject.toml"))
            .await
            .is_ok()
        {
            return Some(candidate);
        }
    }
    None
}

/// Identity of the runtime to be installed. For a local source checkout this is a
/// content hash of every rlm/*.py file plus pyproject.toml, so any runtime code or
/// dependency change invalidates an existing venv automatically. Falls back to the
/// bare package name when the runtime resolves to a registry install (no local source).
pub async fn resolve_runtime_identity() -> Result<String, KernelError> {
    let Some(source_dir) = resolve_runtime_source_dir().await else {
        return Ok(RUNTIME_REQUIREMENT.to_string());
    };
    // TS: a hash failure here must surface rather than fall back to
    // RUNTIME_REQUIREMENT, which would permanently mask later source changes.
    hash_runtime_source(&source_dir).await
}

/// Throws if the local source can't be read. A failure here must surface rather than
/// fall back to RUNTIME_REQUIREMENT: that constant is the registry-install identity, and
/// recording it for a local checkout would permanently mask later source changes.
async fn hash_runtime_source(source_dir: &str) -> Result<String, KernelError> {
    let rlm_dir = Path::new(source_dir).join("src").join("rlm");
    let mut files: Vec<PathBuf> = vec![Path::new(source_dir).join("pyproject.toml")];
    let mut stack = vec![rlm_dir.clone()];
    while let Some(dir) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&dir)
            .await
            .map_err(|error| KernelError::new(error.to_string()))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| KernelError::new(error.to_string()))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| KernelError::new(error.to_string()))?;
            let full = entry.path();
            if file_type.is_dir() {
                stack.push(full);
            } else if file_type.is_file()
                && entry.file_name().to_string_lossy().ends_with(".py")
            {
                files.push(full);
            }
        }
    }
    files.sort_by_key(|path| path.to_string_lossy().into_owned());

    let mut hash = Sha256::new();
    for file in files {
        let relative = file
            .strip_prefix(source_dir)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|_| file.to_string_lossy().into_owned());
        hash.update(relative.as_bytes());
        hash.update(b"\0");
        let bytes = tokio::fs::read(&file)
            .await
            .map_err(|error| KernelError::new(error.to_string()))?;
        hash.update(&bytes);
        hash.update(b"\0");
    }
    let digest = hash.finish();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    Ok(format!("sha256:{out}"))
}

pub fn kernel_venv_python(venv: &str, platform: Option<&str>) -> String {
    let platform = platform.unwrap_or(if is_windows() { "win32" } else { "linux" });
    if platform == "win32" {
        Path::new(venv)
            .join("Scripts")
            .join("python.exe")
            .to_string_lossy()
            .into_owned()
    } else {
        Path::new(venv).join("bin").join("python").to_string_lossy().into_owned()
    }
}

async fn bootstrap_venv(
    venv: &str,
    python_skills: &[BootstrapPythonSkill],
    options: &EnsureKernelPythonOptions,
) -> Result<(), KernelError> {
    if let Some(parent) = Path::new(venv).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| KernelError::new(error.to_string()))?;
    }
    let uv = ensure_uv(options).await?;
    let python = kernel_venv_python(venv, None);
    let source_dir = resolve_runtime_source_dir().await;
    let runtime_requirement = source_dir.unwrap_or_else(|| RUNTIME_REQUIREMENT.to_string());
    let runtime_identity = resolve_runtime_identity().await?;

    run(
        &uv,
        vec!["python".to_string(), "install".to_string(), PYTHON_VERSION.to_string()],
        RunStdio::Ignore,
    )
    .await?;
    run(
        &uv,
        vec![
            "venv".to_string(),
            venv.to_string(),
            "--python".to_string(),
            PYTHON_VERSION.to_string(),
            "--seed".to_string(),
        ],
        RunStdio::Ignore,
    )
    .await?;
    let mut install_args = vec![
        "pip".to_string(),
        "install".to_string(),
        "--python".to_string(),
        python.clone(),
        runtime_requirement,
        STATE_SNAPSHOT_REQUIREMENT.to_string(),
    ];
    install_args.extend(default_rlm_extra_uv_args());
    run(&uv, install_args, RunStdio::Ignore).await?;
    sync_python_skills(&uv, venv, &python, &runtime_identity, python_skills, options).await
}

async fn sync_python_skills(
    uv: &str,
    venv: &str,
    python: &str,
    runtime_identity: &str,
    python_skills: &[BootstrapPythonSkill],
    options: &EnsureKernelPythonOptions,
) -> Result<(), KernelError> {
    let version = read_bootstrap_version(venv).await;
    let mut installed_python_skills: Vec<BootstrapPythonSkill> = Vec::new();
    let current_python_skills: Vec<(String, BootstrapPythonSkill)> = version
        .as_ref()
        .and_then(|value| value.python_skills.clone())
        .unwrap_or_default()
        .into_iter()
        .map(|skill| {
            (
                format!("{}\u{0}{}", skill.import_name, skill.package_path),
                skill,
            )
        })
        .collect();
    let python_skills_by_project_name: Vec<(String, BootstrapPythonSkill)> = python_skills
        .iter()
        .fold(Vec::new(), |mut by_name, skill| {
            let key = read_python_skill_project_name(skill).replace('_', "-").to_lowercase();
            // TypeScript Map construction keeps the LAST duplicate key.
            if let Some(existing) = by_name.iter_mut().find(|(existing, _)| *existing == key) {
                existing.1 = skill.clone();
            } else {
                by_name.push((key, skill.clone()));
            }
            by_name
        });
    let dependencies_by_skill: Vec<(BootstrapPythonSkill, Vec<BootstrapPythonSkill>)> = python_skills
        .iter()
        .map(|skill| {
            let dependencies: Vec<BootstrapPythonSkill> = read_python_skill_dependency_names(skill)
                .into_iter()
                .filter_map(|dependency_name| {
                    python_skills_by_project_name
                        .iter()
                        .find(|(name, _)| *name == dependency_name)
                        .map(|(_, skill)| skill.clone())
                        .or_else(|| resolve_sibling_python_skill_dependency(skill, &dependency_name))
                })
                .collect();
            (skill.clone(), dependencies)
        })
        .collect();

    for skill in sort_python_skills_for_install(python_skills) {
        let key = format!("{}\u{0}{}", skill.import_name, skill.package_path);
        let existing_skill = current_python_skills
            .iter()
            .find(|(existing_key, _)| *existing_key == key)
            .map(|(_, skill)| skill.clone());
        if let Some(existing_skill) = existing_skill.as_ref() {
            if existing_skill.pyproject_path == skill.pyproject_path
                && existing_skill.pyproject_hash == skill.pyproject_hash
            {
                installed_python_skills.push(skill);
                continue;
            }
        }

        let local_dependencies: Vec<BootstrapPythonSkill> = dependencies_by_skill
            .iter()
            .find(|(owner, _)| *owner == skill)
            .map(|(_, dependencies)| dependencies.clone())
            .unwrap_or_default();
        let mut local_dependency_args: Vec<String> = Vec::new();
        for dependency in &local_dependencies {
            let installed_dependency = current_python_skills
                .iter()
                .find(|(existing_key, _)| {
                    *existing_key == format!("{}\u{0}{}", dependency.import_name, dependency.package_path)
                })
                .map(|(_, skill)| skill.clone());
            let installed_this_sync = installed_python_skills.iter().any(|installed| {
                installed.import_name == dependency.import_name
                    && installed.package_path == dependency.package_path
                    && installed.pyproject_path == dependency.pyproject_path
                    && installed.pyproject_hash == dependency.pyproject_hash
            });
            let already_current = installed_dependency
                .as_ref()
                .map(|installed| {
                    installed.pyproject_path == dependency.pyproject_path
                        && installed.pyproject_hash == dependency.pyproject_hash
                })
                .unwrap_or(false);
            if !(installed_this_sync || already_current) {
                local_dependency_args.extend(format_python_skill_install_args(dependency));
            }
        }

        let mut install_args = vec![
            "pip".to_string(),
            "install".to_string(),
            "--python".to_string(),
            python.to_string(),
        ];
        install_args.extend(format_python_skill_install_args(&skill));
        install_args.extend(local_dependency_args);
        match run(uv, install_args, RunStdio::Ignore).await {
            Ok(()) => {
                installed_python_skills.push(skill.clone());
                for dependency in local_dependencies {
                    if !installed_python_skills.contains(&dependency) {
                        installed_python_skills.push(dependency);
                    }
                }
            }
            Err(error) => {
                report_progress(
                    options,
                    &format!(
                        "Warning: Python skill {} failed to install and will be unavailable: {error}",
                        skill.import_name
                    ),
                );
            }
        }
    }
    write_bootstrap_version(venv, runtime_identity, &installed_python_skills).await
}

async fn kernel_base_ready(python: &str, venv: &str, runtime_identity: &str) -> bool {
    has_prime_agent_runtime(python).await
        && bootstrap_base_version_current(read_bootstrap_version(venv).await.as_ref(), runtime_identity)
}

async fn kernel_ready(
    python: &str,
    venv: &str,
    runtime_identity: &str,
    python_skills: &[BootstrapPythonSkill],
) -> bool {
    has_prime_agent_runtime(python).await
        && bootstrap_version_current(
            read_bootstrap_version(venv).await.as_ref(),
            runtime_identity,
            python_skills,
        )
}

fn format_bootstrap_failure(error: &KernelError) -> KernelError {
    KernelError::new(format!(
        "Failed to set up the Python kernel runtime. {error}\nFirst-time setup needs internet to install uv, Python, prime-agent-runtime, and default Python packages; once set up, Optimus runs offline. Set PRIME_AGENT_KERNEL_PYTHON to a Python with a current prime-agent-runtime and default Python packages installed to skip auto-bootstrap."
    ))
}

async fn ensure_kernel_python_uncached(
    options: &EnsureKernelPythonOptions,
    python_skills: &[BootstrapPythonSkill],
) -> Result<String, KernelError> {
    if let Ok(override_value) = std::env::var("PRIME_AGENT_KERNEL_PYTHON") {
        if !override_value.is_empty() {
            let python = resolve_path(&expand_home(&override_value)).to_string_lossy().into_owned();
            if is_batch_shim(&python) {
                return Err(KernelError::new(format!(
                    "PRIME_AGENT_KERNEL_PYTHON must point directly to a Python executable, not a Windows batch shim: {python}"
                )));
            }
            let mut missing: Vec<String> = Vec::new();
            if !has_prime_agent_runtime(&python).await {
                missing.push("a current prime-agent-runtime with callable rlm.run, rlm.create_session, rlm.host_request, and explicit harness CRUD methods".to_string());
            }
            if missing.is_empty() {
                let missing_extra_imports = missing_rlm_extra_import_labels(&python).await;
                if !missing_extra_imports.is_empty() {
                    missing.push(format!(
                        "default Python packages ({})",
                        missing_extra_imports.join(", ")
                    ));
                }
            }
            if missing.is_empty() && !python_skills.is_empty() {
                let missing_python_skills = missing_python_skill_import_labels(
                    &python,
                    options.python_skills.as_deref().unwrap_or(&[]),
                )
                .await;
                if !missing_python_skills.is_empty() {
                    report_progress(
                        options,
                        &format!(
                            "Warning: Python skills unavailable in PRIME_AGENT_KERNEL_PYTHON and will be disabled: {}",
                            missing_python_skills.join(", ")
                        ),
                    );
                }
            }
            if missing.is_empty() {
                return Ok(python);
            }
            return Err(KernelError::new(format!(
                "PRIME_AGENT_KERNEL_PYTHON points to a Python missing {}: {python}",
                missing.join(" and ")
            )));
        }
    }

    let venv = resolve_writable_kernel_venv_dir().await?;
    let python = kernel_venv_python(&venv, None);
    let runtime_identity = resolve_runtime_identity().await?;
    if kernel_ready(&python, &venv, &runtime_identity, python_skills).await {
        return Ok(python);
    }

    let release_lock = acquire_bootstrap_lock(&venv).await?;
    let result = async {
        if kernel_ready(&python, &venv, &runtime_identity, python_skills).await {
            return Ok(());
        }
        if kernel_base_ready(&python, &venv, &runtime_identity).await {
            let uv = ensure_uv(options).await?;
            sync_python_skills(&uv, &venv, &python, &runtime_identity, python_skills, options).await?;
            return Ok(());
        }

        let had_venv = exists(&venv);
        report_progress(options, "\u{203a} setting up python kernel (one-time, ~30s)\u{2026}");
        if had_venv {
            report_progress(options, "rebuilding kernel venv");
            // TS `rm(venv, { recursive: true, force: true })` tolerates only a
            // missing path; any other teardown failure must surface.
            if let Err(remove_error) = tokio::fs::remove_dir_all(&venv).await {
                if remove_error.kind() != std::io::ErrorKind::NotFound {
                    return Err(KernelError::new(remove_error.to_string()));
                }
            }
        }

        bootstrap_venv(&venv, python_skills, options).await
    }
    .await;
    release_lock().await;
    result.map_err(|error| format_bootstrap_failure(&error))?;

    report_progress(options, "\u{2713} ready");
    Ok(python)
}

static IN_FLIGHT_ENSURE_KERNEL_PYTHON: OnceLock<Mutex<Option<(String, Arc<SharedPromise<String>>)>>> =
    OnceLock::new();

fn in_flight() -> &'static Mutex<Option<(String, Arc<SharedPromise<String>>)>> {
    IN_FLIGHT_ENSURE_KERNEL_PYTHON.get_or_init(|| Mutex::new(None))
}

pub async fn ensure_kernel_python(options: EnsureKernelPythonOptions) -> Result<String, KernelError> {
    let python_skills = normalize_python_skills(options.python_skills.as_deref());
    let key = ensure_kernel_python_key(&python_skills);
    // The lock is released BEFORE awaiting: `if let ... = guard.as_ref()` keeps a borrow of the
    // guard alive to the end of its scope, so a `return ....await` inside that scope still counts as
    // using a `MutexGuard` across an await and the enclosing future stops being `Send`.
    let existing_promise = {
        let guard = in_flight().lock().unwrap();
        guard
            .as_ref()
            .and_then(|(existing_key, promise)| (*existing_key == key).then(|| promise.clone()))
    };
    if let Some(promise) = existing_promise {
        return promise.wait().await;
    }

    let promise = Arc::new(SharedPromise::<String>::new());
    *in_flight().lock().unwrap() = Some((key, promise.clone()));

    let outcome = ensure_kernel_python_uncached(&options, &python_skills).await;
    promise.settle(outcome.clone());

    let mut guard = in_flight().lock().unwrap();
    if guard
        .as_ref()
        .map(|(_, existing)| Arc::ptr_eq(existing, &promise))
        .unwrap_or(false)
    {
        *guard = None;
    }
    drop(guard);
    outcome
}

/// Local helper so tests can inspect the ready-check string.
pub fn runtime_ready_check_for_test() -> String {
    runtime_ready_check()
}

pub fn sha256_hex_for_test(data: &[u8]) -> String {
    sha256_hex(data)
}

#[cfg(test)]
fn parity_test_state_root(case: &str) -> PathBuf {
    match std::env::var("PARITY_STATE_ROOT") {
        Ok(root) if !root.trim().is_empty() => {
            PathBuf::from(root).join("unit").join(case)
        }
        _ => std::env::temp_dir()
            .join("pi-coding-agent-bootstrap-tests")
            .join(case),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Multi-block input exercises the streaming buffer boundary.
        let long = "a".repeat(1_000_000);
        assert_eq!(
            sha256_hex(long.as_bytes()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn batch_shim_invocation_wraps_values_in_env_variables() {
        let base_env = vec![("PATH".to_string(), "C:\\Windows".to_string())];
        let invocation =
            build_batch_shim_invocation("C:\\tools\\x.cmd", &["--version".to_string()], &base_env, Some("abc123".into()))
                .expect("valid shim");
        assert_eq!(
            invocation.args,
            vec![
                "/d".to_string(),
                "/v:off".to_string(),
                "/s".to_string(),
                "/c".to_string(),
                "\"\"%PRIME_AGENT_BATCH_abc123_0%\" \"%PRIME_AGENT_BATCH_abc123_1%\"\"".to_string()
            ]
        );
        assert!(invocation
            .env
            .contains(&("PRIME_AGENT_BATCH_abc123_0".to_string(), "C:\\tools\\x.cmd".to_string())));
        assert!(invocation
            .env
            .contains(&("PRIME_AGENT_BATCH_abc123_1".to_string(), "--version".to_string())));
    }

    #[test]
    fn batch_shim_invocation_rejects_unsafe_values() {
        let bad_token = build_batch_shim_invocation("cmd", &[], &[], Some("bad token".into()));
        assert!(bad_token.is_err());
        let bad_value = build_batch_shim_invocation("cmd", &["a\"b".to_string()], &[], Some("t".into()));
        assert_eq!(
            bad_value.unwrap_err().to_string(),
            "Windows batch shim paths and arguments cannot contain quotes, NUL, or line breaks"
        );
    }

    #[test]
    fn windows_executable_candidates_follow_pathext_order() {
        assert_eq!(
            windows_executable_candidates("python", Some(".COM;.EXE;.BAT;.CMD")),
            vec!["python", "python.com", "python.exe", "python.bat", "python.cmd"]
        );
        // An explicit extension is used as-is.
        assert_eq!(windows_executable_candidates("run.cmd", Some(".EXE")), vec!["run.cmd"]);
        // Unsupported extensions are filtered; an empty result falls back to the default.
        assert_eq!(
            windows_executable_candidates("x", Some(".PS1")),
            vec!["x", "x.com", "x.exe", "x.bat", "x.cmd"]
        );
    }

    #[test]
    fn dependency_package_name_is_normalized() {
        assert_eq!(parse_dependency_package_name("Requests>=2").as_deref(), Some("requests"));
        assert_eq!(parse_dependency_package_name("my_pkg ; extra").as_deref(), Some("my-pkg"));
        assert_eq!(parse_dependency_package_name("   "), None);
    }

    #[test]
    fn dependency_names_are_read_from_the_project_section() {
        let dir = std::env::temp_dir().join(format!("pi-skill-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pyproject = dir.join("pyproject.toml");
        std::fs::write(
            &pyproject,
            "[project]\nname = \"my_skill\"\ndependencies = [\"rlm-bash\", \"other_pkg>=1; python_version>'3'\"],\n\n[tool.uv]\nx = 1\n",
        )
        .unwrap();
        let skill = BootstrapPythonSkill {
            import_name: "my_skill".to_string(),
            package_path: dir.to_string_lossy().into_owned(),
            pyproject_path: pyproject.to_string_lossy().into_owned(),
            pyproject_hash: "hash".to_string(),
        };
        assert_eq!(read_python_skill_project_name(&skill), "my_skill");
        let names: Vec<String> = read_python_skill_dependency_names(&skill).into_iter().collect();
        assert_eq!(names, vec!["other-pkg".to_string(), "rlm-bash".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn project_name_falls_back_to_the_import_name() {
        let dir = std::env::temp_dir().join(format!("pi-skill-name-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pyproject = dir.join("pyproject.toml");
        std::fs::write(&pyproject, "[build-system]\nrequires = []\n").unwrap();
        let skill = BootstrapPythonSkill {
            import_name: "my_skill".to_string(),
            package_path: dir.to_string_lossy().into_owned(),
            pyproject_path: pyproject.to_string_lossy().into_owned(),
            pyproject_hash: "hash".to_string(),
        };
        assert_eq!(read_python_skill_project_name(&skill), "my-skill");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn install_order_puts_dependencies_first() {
        let dir = std::env::temp_dir().join(format!("pi-skill-order-{}", std::process::id()));
        let leaf = dir.join("leaf");
        let root = dir.join("root");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            leaf.join("pyproject.toml"),
            "[project]\nname = \"leaf\"\ndependencies = []\n",
        )
        .unwrap();
        std::fs::write(
            root.join("pyproject.toml"),
            "[project]\nname = \"root\"\ndependencies = [\"leaf\"]\n",
        )
        .unwrap();
        let skills = vec![
            BootstrapPythonSkill {
                import_name: "root".to_string(),
                package_path: root.to_string_lossy().into_owned(),
                pyproject_path: root.join("pyproject.toml").to_string_lossy().into_owned(),
                pyproject_hash: "h".to_string(),
            },
            BootstrapPythonSkill {
                import_name: "leaf".to_string(),
                package_path: leaf.to_string_lossy().into_owned(),
                pyproject_path: leaf.join("pyproject.toml").to_string_lossy().into_owned(),
                pyproject_hash: "h".to_string(),
            },
        ];
        let sorted = sort_python_skills_for_install(&skills);
        assert_eq!(sorted[0].import_name, "leaf");
        assert_eq!(sorted[1].import_name, "root");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn venv_python_uses_the_platform_layout() {
        let win = kernel_venv_python("C:\\v", Some("win32"));
        assert!(win.ends_with("Scripts\\python.exe") || win.ends_with("Scripts/python.exe"));
        let posix = kernel_venv_python("/v", Some("linux"));
        assert!(posix.ends_with("bin/python") || posix.ends_with("bin\\python"));
    }

    #[test]
    fn bootstrap_version_comparisons_are_exact() {
        let skills = vec![BootstrapPythonSkill {
            import_name: "a".to_string(),
            package_path: "p".to_string(),
            pyproject_path: "pp".to_string(),
            pyproject_hash: "h".to_string(),
        }];
        let version = BootstrapVersion {
            schema: BOOTSTRAP_SCHEMA,
            runtime: Some("rt".to_string()),
            snapshot: Some(STATE_SNAPSHOT_REQUIREMENT.to_string()),
            extra_uv_args: Some(default_rlm_extra_uv_args()),
            python_skills: Some(skills.clone()),
        };
        assert!(bootstrap_version_current(Some(&version), "rt", &skills));
        assert!(!bootstrap_version_current(Some(&version), "other", &skills));
        assert!(!bootstrap_version_current(None, "rt", &skills));

        let mut changed = version.clone();
        changed.python_skills = Some(vec![BootstrapPythonSkill {
            pyproject_hash: "different".to_string(),
            ..skills[0].clone()
        }]);
        assert!(!bootstrap_version_current(Some(&changed), "rt", &skills));

        let mut wrong_args = version.clone();
        wrong_args.extra_uv_args = Some(vec!["numpy".to_string()]);
        assert!(!bootstrap_base_version_current(Some(&wrong_args), "rt"));
        assert!(extra_uv_args_match(None, None));
        assert!(!extra_uv_args_match(None, Some(&vec!["a".to_string()])));
    }

    #[test]
    fn ready_check_embeds_the_harness_method_list() {
        let check = runtime_ready_check();
        let methods = check.split("_harness_methods = ").nth(1).unwrap().split(';').next().unwrap();
        assert_eq!(serde_json::from_str::<Vec<String>>(methods).unwrap(), REQUIRED_HARNESS_METHODS);
        assert!(check.contains("assert _repl.PROTOCOL_VERSION == 4"));
    }

    /// G-29: write the marker through the real writer and read it back
    /// through the real reader; every field must survive, including the
    /// nested skill records and nonempty extra uv args.
    #[tokio::test]
    async fn bootstrap_marker_roundtrip_with_skills() {
        let case = super::parity_test_state_root("t01-marker");
        let _ = std::fs::remove_dir_all(&case);
        std::fs::create_dir_all(&case).unwrap();
        let venv = case.join("venv");
        let unicode_dir = case.join("路径 with spaces");
        std::fs::create_dir_all(&unicode_dir).unwrap();
        let pyproject = unicode_dir.join("pyproject.toml");
        std::fs::write(&pyproject, "[project]\nname = \"parity-skill\"\n").unwrap();
        let skills = vec![
            BootstrapPythonSkill {
                import_name: "parity_skill_a".to_string(),
                package_path: unicode_dir.to_string_lossy().into_owned(),
                pyproject_path: pyproject.to_string_lossy().into_owned(),
                pyproject_hash: "sha256:hash-a".to_string(),
            },
            BootstrapPythonSkill {
                import_name: "parity_skill_b".to_string(),
                package_path: case.join("plain").to_string_lossy().into_owned(),
                pyproject_path: case.join("plain").join("py.toml").to_string_lossy().into_owned(),
                pyproject_hash: "sha256:hash-b".to_string(),
            },
        ];
        std::fs::create_dir_all(&venv).unwrap();
        let runtime = "sha256:identity".to_string();
        write_bootstrap_version(venv.to_string_lossy().as_ref(), &runtime, &skills)
            .await
            .unwrap();

        let raw = std::fs::read_to_string(venv.join(BOOTSTRAP_VERSION_FILE)).unwrap();
        assert!(
            raw.contains("\"extraUvArgs\"") && raw.contains("\"pythonSkills\""),
            "marker must be written with camelCase keys, raw={raw}"
        );
        assert!(
            raw.contains("\"importName\"") && raw.contains("\"pyprojectHash\""),
            "nested skills must use the TypeScript camelCase keys, raw={raw}"
        );

        let version = read_bootstrap_version(venv.to_string_lossy().as_ref()).await;
        let version = version.expect("marker must parse");
        assert_eq!(version.schema, BOOTSTRAP_SCHEMA);
        assert_eq!(version.runtime.as_deref(), Some(runtime.as_str()));
        assert_eq!(version.snapshot.as_deref(), Some(STATE_SNAPSHOT_REQUIREMENT));
        assert_eq!(version.extra_uv_args.as_deref(), Some(default_rlm_extra_uv_args().as_slice()));
        let stored = version.python_skills.as_ref().expect("skills survive").clone();
        assert_eq!(stored, skills, "nested skill records must round-trip field by field");

        assert!(bootstrap_version_current(Some(&version), &runtime, &skills));
        assert!(bootstrap_base_version_current(Some(&version), &runtime));

        // Empty skills stay current.
        write_bootstrap_version(venv.to_string_lossy().as_ref(), &runtime, &[])
            .await
            .unwrap();
        let version = read_bootstrap_version(venv.to_string_lossy().as_ref())
            .await
            .expect("empty marker");
        assert!(bootstrap_version_current(Some(&version), &runtime, &[]));

        // TypeScript camelCase fixture (positive control). The marker is only
        // "current" when extraUvArgs equals the default extra args, so the
        // fixture carries the real default list.
        let default_args_json = serde_json::to_string(&default_rlm_extra_uv_args()).unwrap();
        let ts_body = format!(
            "{{\"schema\":9,\"runtime\":\"rt-ts\",\"snapshot\":\"dill\",\"extraUvArgs\":{},\"pythonSkills\":[{{\"importName\":\"import_ts\",\"packagePath\":\"C:/ts pkg\",\"pyprojectPath\":\"C:/ts pkg/pyproject.toml\",\"pyprojectHash\":\"h\"}}]}}\n",
            default_args_json
        );
        std::fs::write(venv.join(BOOTSTRAP_VERSION_FILE), ts_body).unwrap();
        let version = read_bootstrap_version(venv.to_string_lossy().as_ref())
            .await
            .expect("ts fixture parses");
        assert_eq!(
            version.extra_uv_args.as_deref(),
            Some(default_rlm_extra_uv_args().as_slice())
        );
        let ts_skills = version.python_skills.as_ref().expect("ts skills");
        assert_eq!(ts_skills[0].import_name, "import_ts");
        assert_eq!(ts_skills[0].package_path, "C:/ts pkg");
        assert!(bootstrap_version_current(Some(&version), "rt-ts", ts_skills));

        // Legacy snake_case markers (written by the defective writer) read
        // safely as NOT current: one documented rebuild, then they normalize.
        let legacy = "{\"schema\":9,\"runtime\":\"rt-old\",\"snapshot\":\"dill\",\"extra_uv_args\":[\"numpy\"],\"python_skills\":[]}\n";
        std::fs::write(venv.join(BOOTSTRAP_VERSION_FILE), legacy).unwrap();
        let version = read_bootstrap_version(venv.to_string_lossy().as_ref())
            .await
            .expect("legacy marker parses without panicking");
        assert_eq!(
            version.extra_uv_args, None,
            "legacy snake_case args are unreadable by design of the fix"
        );
        // The reader looks for the camelCase keys only, so the legacy
        // python_skills array reads as absent (None), not an empty list.
        assert_eq!(version.python_skills, None);
        assert!(!bootstrap_base_version_current(Some(&version), "rt-old"));
    }

    /// G-07: duplicate project names must resolve like the TypeScript
    /// reference, where the LAST duplicate wins (Map.set overwrite). The
    /// dependent skill's dependency edge must therefore point at the last
    /// duplicate, which delays it in the topological order.
    #[test]
    fn duplicate_project_name_resolves_last_like_typescript() {
        let case = super::parity_test_state_root("t01-dup");
        let _ = std::fs::remove_dir_all(&case);
        let first = case.join("first");
        let second = case.join("second");
        let dependent = case.join("dependent");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::create_dir_all(&dependent).unwrap();
        std::fs::write(
            first.join("pyproject.toml"),
            "[project]\nname = \"parity-dup-name\"\n",
        )
        .unwrap();
        std::fs::write(
            second.join("pyproject.toml"),
            "[project]\nname = \"parity-dup-name\"\n",
        )
        .unwrap();
        std::fs::write(
            dependent.join("pyproject.toml"),
            "[project]\nname = \"parity-dependent\"\ndependencies = [\"parity-dup-name\"]\n",
        )
        .unwrap();
        let skill = |name: &str, dir: &Path| BootstrapPythonSkill {
            import_name: name.to_string(),
            package_path: dir.to_string_lossy().into_owned(),
            pyproject_path: dir.join("pyproject.toml").to_string_lossy().into_owned(),
            pyproject_hash: file_content_hash(dir.join("pyproject.toml").to_string_lossy().as_ref()),
        };
        let first_skill = skill("parity_first", &first);
        let second_skill = skill("parity_second", &second);
        let dependent_skill = skill("parity_dependent", &dependent);
        let sorted = sort_python_skills_for_install(&[first_skill, second_skill, dependent_skill]);
        let position_of = |needle: &str| {
            sorted
                .iter()
                .position(|candidate| candidate.package_path.replace('\\', "/").contains(needle))
                .unwrap_or(usize::MAX)
        };
        assert!(
            position_of("first") < position_of("second"),
            "the untouched duplicates must keep their original order: {sorted:?}"
        );
        assert!(
            position_of("second") < position_of("dependent"),
            "TS contract violated: the dependent sorted before the LAST duplicate \
             (first-wins defect): {sorted:?}"
        );
    }
}
