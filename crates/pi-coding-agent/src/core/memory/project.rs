//! Port of packages/coding-agent/src/core/memory/project.ts
use std::path::{Path, PathBuf};
use std::process::Command;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::utils::atomic_file::write_file_atomic_sync;

use super::evidence::hash;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectIdentity {
    pub id: String,
    pub root: String,
    pub aliases: Vec<String>,
}

pub fn normalize_remote(remote: &str) -> String {
    let value = {
        let trimmed = remote.trim();
        let value = regex::Regex::new(r"\.git/?$")
            .unwrap()
            .replace(trimmed, "")
            .to_string();
        regex::Regex::new(r"/$")
            .unwrap()
            .replace(&value, "")
            .to_string()
    };
    let scp_re = regex::Regex::new(r"^(?:[^/@]+@)?([^/:]+):([^/].*)$").unwrap();
    let drive_re = regex::Regex::new(r"^[A-Za-z]:").unwrap();
    if let Some(captures) = scp_re.captures(&value) {
        if !value.contains("://") && !drive_re.is_match(&value) {
            let host = captures.get(1).map(|m| m.as_str()).unwrap_or("");
            let path = captures.get(2).map(|m| m.as_str()).unwrap_or("");
            return format!("{}/{}", host.to_lowercase(), path);
        }
    }
    match url::Url::parse(&value) {
        Ok(url) => {
            let hostname = url.host_str().unwrap_or("").to_lowercase();
            let port = url
                .port()
                .map(|port| format!(":{port}"))
                .unwrap_or_default();
            format!("{hostname}{port}{}", url.path())
        }
        Err(_) => value,
    }
}

fn git(cwd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let trimmed = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn realpath_sync(path: &Path) -> String {
    std::fs::canonicalize(path)
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

/// Windows paths are case-insensitive; lock identity must match the OS view.
fn lock_key(path: &Path) -> String {
    let key = realpath_sync(path);
    #[cfg(windows)]
    let key = key.to_lowercase();
    key
}

fn projects_registry_path(agent_dir: &str) -> PathBuf {
    Path::new(agent_dir).join("memory").join("projects.json")
}

fn read_registry(file: &Path) -> IndexMap<String, Value> {
    match std::fs::read_to_string(file) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(map)) => map.into_iter().collect(),
            _ => IndexMap::new(),
        },
        Err(_) => IndexMap::new(),
    }
}

pub fn project_identity(
    cwd: &str,
    agent_dir: &str,
    bind_id: Option<&str>,
) -> Result<ProjectIdentity, String> {
    let root = git(cwd, &["rev-parse", "--show-toplevel"])
        .unwrap_or_else(|| realpath_sync(Path::new(cwd)));
    let common = git(
        cwd,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )
    .unwrap_or_else(|| root.clone());
    let remote = git(cwd, &["config", "--get", "remote.origin.url"]);
    let mut aliases = vec![format!("path:{}", realpath_sync(Path::new(&common)))];
    if let Some(remote) = &remote {
        aliases.push(format!("remote:{}", normalize_remote(remote)));
    }
    let dir = Path::new(agent_dir).join("memory");
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let file = dir.join("projects.json");
    let _lock = super::acquire_lock_sync(
        &lock_key(&dir),
        super::MemoryLockRetries {
            stale_ms: 10_000,
            retries: 0,
            min_timeout_ms: 0,
            max_timeout_ms: 0,
        },
    )?;
    let registry = read_registry(&file);
    let existing: Vec<String> = aliases
        .iter()
        .filter_map(|alias| registry.get(alias))
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    let id = match bind_id {
        Some(value) => value.to_string(),
        None => match existing.first() {
            Some(value) => value.clone(),
            None => {
                let last = aliases.last().cloned().unwrap_or_default();
                let digest = hash(&last);
                format!("project_{}", &digest[..24.min(digest.len())])
            }
        },
    };
    let id_re = regex::Regex::new(r"^project_[a-zA-Z0-9_-]{1,80}$").unwrap();
    if !id_re.is_match(&id) {
        return Err(
            "Project ID must start with project_ and contain only letters, digits, _ or -"
                .to_string(),
        );
    }
    if bind_id.is_none() && existing.iter().any(|value| value != &id) {
        return Err(
            "Project aliases conflict; explicitly bind the intended project ID".to_string(),
        );
    }
    if aliases
        .iter()
        .any(|alias| registry.get(alias).and_then(Value::as_str) != Some(id.as_str()))
    {
        let mut next: Map<String, Value> = Map::new();
        for (key, value) in &registry {
            next.insert(key.clone(), value.clone());
        }
        for alias in &aliases {
            next.insert(alias.clone(), Value::String(id.clone()));
        }
        let body = format!(
            "{}\n",
            serde_json::to_string_pretty(&Value::Object(next)).unwrap_or_default()
        );
        write_file_atomic_sync(
            &file.to_string_lossy(),
            &body,
            crate::utils::atomic_file::WriteFileAtomicOptions {
                mode: Some(0o600),
                ..Default::default()
            },
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(ProjectIdentity { id, root, aliases })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_authenticated_urls_without_leaking_credentials() {
        assert_eq!(
            normalize_remote("https://secret@github.com/telemusai/optimus-agent.git"),
            normalize_remote("git@github.com:telemusai/optimus-agent.git")
        );
        assert_eq!(
            normalize_remote("https://github.com/telemusai/prime.git"),
            "github.com/telemusai/prime"
        );
        assert_eq!(
            normalize_remote("git@github.com:telemusai/optimus-agent.git"),
            "github.com/telemusai/optimus-agent"
        );
    }

    #[test]
    fn keeps_ports_and_local_paths_intact() {
        assert_eq!(
            normalize_remote("https://example.com:8443/x/y/"),
            "example.com:8443/x/y"
        );
        // The pinned JS URL parser treats the drive letter as a URL scheme.
        assert_eq!(normalize_remote("C:/Users/x/repo"), "/Users/x/repo");
        assert_eq!(normalize_remote("/Users/x/repo"), "/Users/x/repo");
    }

    #[test]
    fn retains_identity_through_remote_rename_and_honours_explicit_bind() {
        let base = std::env::temp_dir().join(format!("prime-project-{}", uuid::Uuid::new_v4()));
        let cwd = base.join("repo");
        let agent_dir = base.join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let cwd = cwd.to_string_lossy().to_string();
        let agent_dir = agent_dir.to_string_lossy().to_string();
        let status = Command::new("git").arg("init").arg(&cwd).output().unwrap();
        assert!(status.status.success());
        let _ = Command::new("git")
            .args([
                "-C",
                &cwd,
                "remote",
                "add",
                "origin",
                "https://github.com/telemusai/prime.git",
            ])
            .output()
            .unwrap();
        let first = project_identity(&cwd, &agent_dir, None).expect("identity");
        let _ = Command::new("git")
            .args([
                "-C",
                &cwd,
                "remote",
                "set-url",
                "origin",
                "git@github.com:telemusai/optimus-agent.git",
            ])
            .output()
            .unwrap();
        assert_eq!(
            project_identity(&cwd, &agent_dir, None)
                .expect("identity")
                .id,
            first.id
        );
        assert_eq!(
            project_identity(&cwd, &agent_dir, Some("project_explicit"))
                .expect("identity")
                .id,
            "project_explicit"
        );
        std::fs::remove_dir_all(&base).ok();
    }
}
