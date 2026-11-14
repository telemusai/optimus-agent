//! Port of packages/coding-agent/src/modes/agent-connection/snapshot.ts

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::modes::agent_connection::types::{
    AgentConnectionArtifactReference, AgentConnectionResourceDiagnostics, AgentConnectionResourceSnapshot,
    AgentConnectionSlashCommand, AgentConnectionSnapshot, AgentConnectionSourceInfo, AgentConnectionState,
};

/// `AgentSession` fields this module reads. The session slice owns the real type.
pub struct AgentSessionSnapshotSource {
    pub session_id: String,
    pub cwd: String,
    pub session_dir: Option<String>,
    pub leaf_id: Option<String>,
    pub session_file: Option<String>,
    pub session_name: Option<String>,
    pub model: Option<pi_ai::types::Model>,
    pub thinking_level: pi_agent_core::types::ThinkingLevel,
    pub service_tier: pi_ai::types::ServiceTier,
    pub available_thinking_levels: Vec<pi_agent_core::types::ThinkingLevel>,
    pub is_streaming: bool,
    pub is_compacting: bool,
    pub is_bash_running: bool,
    pub retry_attempt: f64,
    pub steering_mode: String,
    pub follow_up_mode: String,
    pub auto_compaction_enabled: bool,
    pub message_count: f64,
    pub session_actions: serde_json::Value,
    pub compaction_count: f64,
    pub goal: serde_json::Value,
    pub scoped_models: Vec<crate::modes::agent_connection::types::AgentConnectionScopedModel>,
    pub active_tool_names: Vec<String>,
    pub occ3_native_dead: bool,
    pub occ3_global_opt_in: bool,
    /// True while this session owns an active TextMAS preparation run.
    pub textmas_active: bool,
    pub textmas_activity: Option<crate::core::occ::textmas_activity::TextMasActivitySnapshot>,
    pub context_usage: serde_json::Value,
    pub persisted_recap: Option<String>,
    pub messages: Vec<pi_agent_core::types::AgentMessage>,
    pub streaming_message: Option<pi_agent_core::types::AgentMessage>,
    pub session_context: Option<crate::modes::agent_connection::types::AgentConnectionSessionContext>,
    pub session_tree: Option<crate::modes::agent_connection::types::AgentConnectionSessionTree>,
    pub children: Vec<crate::modes::agent_connection::types::AgentConnectionRlmChildAgentSnapshot>,
}

/// `AgentSessionRuntime` fields this module reads.
pub struct AgentSessionRuntimeSnapshotSource {
    pub session: AgentSessionSnapshotSource,
}

fn persisted_recap(session: &AgentSessionSnapshotSource) -> Option<String> {
    session.persisted_recap.clone()
}

pub fn create_agent_connection_state(
    runtime: &AgentSessionRuntimeSnapshotSource,
    active_session_id: Option<String>,
) -> AgentConnectionState {
    let session = &runtime.session;
    AgentConnectionState {
        active_session_id,
        cwd: session.cwd.clone(),
        model: session.model.clone(),
        thinking_level: session.thinking_level,
        service_tier: session.service_tier.clone(),
        available_thinking_levels: session.available_thinking_levels.clone(),
        is_streaming: session.is_streaming,
        is_compacting: session.is_compacting,
        is_bash_running: session.is_bash_running,
        retry_attempt: session.retry_attempt,
        steering_mode: session.steering_mode.clone(),
        follow_up_mode: session.follow_up_mode.clone(),
        session_file: session.session_file.clone(),
        session_id: session.session_id.clone(),
        session_name: session.session_name.clone(),
        session_dir: session.session_dir.clone(),
        leaf_id: session.leaf_id.clone(),
        auto_compaction_enabled: session.auto_compaction_enabled,
        message_count: session.message_count,
        session_actions: session.session_actions.clone(),
        compaction_count: session.compaction_count,
        goal: session.goal.clone(),
        heartbeat: None,
        scoped_models: session.scoped_models.clone(),
        active_tool_names: session.active_tool_names.clone(),
        occ3_native_dead: session.occ3_native_dead,
        occ3_modes: occ3_modes(session.session_file.as_deref(), session.model.as_ref(),
            session.occ3_global_opt_in, &session.active_tool_names, session.occ3_native_dead,
            session.textmas_active, session.textmas_activity.clone()),
        context_usage: session.context_usage.clone(),
        // Baseline recap; the daemon overlays the live summary when attaching.
        recap: persisted_recap(session),
    }
}

/// Display only: preference and local enrollment are never completed-use evidence.
fn occ3_modes(
    session_file: Option<&str>,
    model: Option<&pi_ai::types::Model>,
    global_opt_in: bool,
    active_tool_names: &[String],
    native_dead: bool,
    textmas_active: bool,
    textmas_activity: Option<crate::core::occ::textmas_activity::TextMasActivitySnapshot>,
) -> Option<crate::modes::agent_connection::types::Occ3Modes> {
    use crate::core::occ::{session_settings, textmas_settings};
    use crate::core::occ3_connectors::activation::{configured_connector, Occ3ActivationRequest};
    let session_file = session_file?;
    // An unreadable preference is unavailable, not a misleading OFF/default.
    let textmas = textmas_settings::resolve(session_file).ok()?;
    let latent = session_settings::resolve(session_file).ok()?;
    let latent_enabled = if latent.occ3_source_session_id.is_some() {
        latent.occ3_enabled
    } else {
        global_opt_in
    };
    let latent_available = model.is_some_and(|model|
        configured_connector(&Occ3ActivationRequest::new(model, true, true)).is_ok());
    let latent_active = latent_enabled && !native_dead &&
        active_tool_names.iter().any(|name| name == "occ3_handoff");
    Some(crate::modes::agent_connection::types::Occ3Modes {
        textmas_mode: textmas.mode,
        textmas_active,
        textmas_activity,
        latent_enabled,
        latent_active,
        latent_available,
    })
}

pub fn create_agent_connection_snapshot(
    runtime: &AgentSessionRuntimeSnapshotSource,
    active_session_id: Option<String>,
) -> AgentConnectionSnapshot {
    let session = &runtime.session;
    AgentConnectionSnapshot {
        state: create_agent_connection_state(runtime, active_session_id),
        messages: session.messages.clone(),
        history: None,
        streaming_message: session.streaming_message.clone(),
        session_context: session.session_context.clone(),
        session_tree: session.session_tree.clone(),
        parent: None,
        children: Some(session.children.clone()),
        last_event_sequence: None,
        last_event_cursor: None,
        replay: None,
    }
}

/// `extensionRunner.getRegisteredCommands()` entry.
#[derive(Debug, Clone, Default)]
pub struct RegisteredCommandEntry {
    pub invocation_name: String,
    pub name: String,
    pub description: Option<String>,
    pub source_info: AgentConnectionSourceInfo,
}

/// `promptTemplates` entry.
#[derive(Debug, Clone, Default)]
pub struct PromptTemplateEntry {
    pub name: String,
    pub description: Option<String>,
    pub argument_hint: Option<String>,
    pub source_info: AgentConnectionSourceInfo,
}

/// `resourceLoader.getSkills().skills` entry.
#[derive(Debug, Clone, Default)]
pub struct SkillEntry {
    pub name: String,
    pub description: Option<String>,
    pub source_info: AgentConnectionSourceInfo,
}

pub fn create_agent_connection_commands(
    registered_commands: &[RegisteredCommandEntry],
    prompt_templates: &[PromptTemplateEntry],
    skills: &[SkillEntry],
) -> Vec<AgentConnectionSlashCommand> {
    let mut commands = Vec::new();
    for entry in registered_commands {
        commands.push(AgentConnectionSlashCommand {
            name: entry.invocation_name.clone(),
            registered_name: Some(entry.name.clone()),
            description: entry.description.clone(),
            argument_hint: None,
            source: "extension".to_string(),
            source_info: entry.source_info.clone(),
        });
    }
    for entry in prompt_templates {
        commands.push(AgentConnectionSlashCommand {
            name: entry.name.clone(),
            registered_name: None,
            description: entry.description.clone(),
            argument_hint: entry.argument_hint.clone(),
            source: "prompt".to_string(),
            source_info: entry.source_info.clone(),
        });
    }
    for entry in skills {
        commands.push(AgentConnectionSlashCommand {
            name: format!("skill:{}", entry.name),
            registered_name: None,
            description: entry.description.clone(),
            argument_hint: None,
            source: "skill".to_string(),
            source_info: entry.source_info.clone(),
        });
    }
    commands
}

/// Loaded resource shapes `createAgentConnectionResourceSnapshot` reads.
#[derive(Debug, Clone, Default)]
pub struct AgentsFileEntry {
    pub path: String,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceSkillEntry {
    pub name: String,
    pub description: Option<String>,
    pub file_path: String,
    pub source_info: Option<AgentConnectionSourceInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct ResourcePromptEntry {
    pub name: String,
    pub description: Option<String>,
    pub argument_hint: Option<String>,
    pub file_path: String,
    pub source_info: Option<AgentConnectionSourceInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceExtensionEntry {
    pub path: String,
    pub source_info: Option<AgentConnectionSourceInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceThemeEntry {
    pub name: Option<String>,
    pub source_path: Option<String>,
    pub source_info: Option<AgentConnectionSourceInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct ExtensionLoadError {
    pub error: String,
    pub path: Option<String>,
}

pub fn create_agent_connection_resource_snapshot(
    session_id: &str,
    cwd: &str,
    agents_files: &[AgentsFileEntry],
    skills: &[ResourceSkillEntry],
    skill_diagnostics: Vec<crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic>,
    prompts: &[ResourcePromptEntry],
    prompt_diagnostics: Vec<crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic>,
    extensions: &[ResourceExtensionEntry],
    extension_errors: &[ExtensionLoadError],
    themes: &[ResourceThemeEntry],
    theme_diagnostics: Vec<crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic>,
) -> AgentConnectionResourceSnapshot {
    AgentConnectionResourceSnapshot {
        context_files: agents_files
            .iter()
            .map(|entry| crate::modes::agent_connection::types::AgentConnectionResourceContextFile {
                path: entry.path.clone(),
                artifact: create_artifact_reference(session_id, cwd, "context_file", Some(&entry.path)),
            })
            .collect(),
        skills: skills
            .iter()
            .map(|skill| crate::modes::agent_connection::types::AgentConnectionResourceSkill {
                name: skill.name.clone(),
                description: skill.description.clone(),
                file_path: skill.file_path.clone(),
                source_info: skill.source_info.clone(),
                artifact: create_artifact_reference(session_id, cwd, "skill", Some(&skill.file_path)),
            })
            .collect(),
        prompts: prompts
            .iter()
            .map(|prompt| crate::modes::agent_connection::types::AgentConnectionResourcePrompt {
                name: prompt.name.clone(),
                description: prompt.description.clone(),
                argument_hint: prompt.argument_hint.clone(),
                file_path: prompt.file_path.clone(),
                source_info: prompt.source_info.clone(),
                artifact: create_artifact_reference(session_id, cwd, "prompt", Some(&prompt.file_path)),
            })
            .collect(),
        extensions: extensions
            .iter()
            .map(|extension| crate::modes::agent_connection::types::AgentConnectionResourceExtension {
                path: extension.path.clone(),
                source_info: extension.source_info.clone(),
                artifact: create_artifact_reference(session_id, cwd, "extension", Some(&extension.path)),
            })
            .collect(),
        themes: themes
            .iter()
            .map(|theme| crate::modes::agent_connection::types::AgentConnectionResourceTheme {
                name: theme.name.clone(),
                source_path: theme.source_path.clone(),
                source_info: theme.source_info.clone(),
                artifact: create_artifact_reference(session_id, cwd, "theme", theme.source_path.as_deref()),
            })
            .collect(),
        diagnostics: AgentConnectionResourceDiagnostics {
            skills: skill_diagnostics,
            prompts: prompt_diagnostics,
            extensions: extension_errors
                .iter()
                .map(|error| crate::modes::agent_connection::types::AgentConnectionResourceDiagnostic {
                    type_: "error".to_string(),
                    message: error.error.clone(),
                    path: error.path.clone(),
                    collision: None,
                })
                .collect(),
            themes: theme_diagnostics,
        },
    }
}

/// `toConnectionModel(model)` - the TypeScript identity overload.
pub fn to_connection_model(model: Option<&pi_ai::types::Model>) -> Option<pi_ai::types::Model> {
    model.cloned()
}

pub fn create_artifact_reference(
    session_id: &str,
    cwd: &str,
    type_: &str,
    file_path: Option<&str>,
) -> Option<AgentConnectionArtifactReference> {
    let file_path = file_path?;
    let path_info = create_artifact_path_info(file_path, cwd);
    let mut hasher = Sha256::new();
    hasher.update(format!("{session_id}\u{0}{type_}\u{0}{file_path}").as_bytes());
    let digest = hasher.finalize();
    let id_hash = hex(&digest)[..16].to_string();
    Some(AgentConnectionArtifactReference {
        id: format!("artifact_{id_hash}"),
        session_id: session_id.to_string(),
        type_: type_.to_string(),
        logical_path: path_info.0,
        relative_path: path_info.1,
        mime_type: None,
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactPathInfo {
    pub logical_path: String,
    pub relative_path: Option<String>,
}

pub fn create_artifact_path_info(file_path: &str, cwd: &str) -> (String, Option<String>) {
    if file_path.starts_with('<') && file_path.ends_with('>') {
        return (file_path.to_string(), None);
    }

    let resolved_cwd = resolve_path(cwd);
    let resolved_path = resolve_path(file_path);
    let cwd_relative_path = to_posix_path(&relative_path(&resolved_cwd, &resolved_path));
    if !cwd_relative_path.is_empty()
        && !cwd_relative_path.starts_with("..")
        && !Path::new(&cwd_relative_path).is_absolute()
    {
        return (cwd_relative_path.clone(), Some(cwd_relative_path));
    }

    let base = Path::new(file_path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    if base.is_empty() {
        ("artifact".to_string(), None)
    } else {
        (base, None)
    }
}

/// Node's `path.resolve` (lexical when the path does not exist).
fn resolve_path(path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        normalize_lexically(candidate)
    } else {
        let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        normalize_lexically(&base.join(candidate))
    }
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            Component::Normal(part) => out.push(part),
        }
    }
    out
}

/// Node's `path.relative`.
fn relative_path(from: &Path, to: &Path) -> String {
    let from_components: Vec<_> = from.components().collect();
    let to_components: Vec<_> = to.components().collect();
    let mut common = 0;
    while common < from_components.len()
        && common < to_components.len()
        && path_components_equal(from_components[common], to_components[common])
    {
        common += 1;
    }
    if common == 0 {
        return to.to_string_lossy().to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for _ in common..from_components.len() {
        parts.push("..".to_string());
    }
    for component in &to_components[common..] {
        parts.push(component.as_os_str().to_string_lossy().to_string());
    }
    parts.join(std::path::MAIN_SEPARATOR_STR)
}

// Node path.relative is case-insensitive on win32; compare components
// case-insensitively there so drive-letter case differences do not break
// the cwd-relative artifact path.
fn path_components_equal(a: std::path::Component, b: std::path::Component) -> bool {
    if a == b {
        return true;
    }
    if cfg!(windows) {
        return a.as_os_str().to_string_lossy().to_lowercase()
            == b.as_os_str().to_string_lossy().to_lowercase();
    }
    false
}

fn to_posix_path(path: &str) -> String {
    if std::path::MAIN_SEPARATOR == '/' {
        path.to_string()
    } else {
        path.split(std::path::MAIN_SEPARATOR).collect::<Vec<_>>().join("/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occ31_modes_are_provider_neutral_and_never_delivery_evidence() {
        use crate::core::occ::{session_settings, textmas_settings, textmas::TextMasMode};
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        std::fs::create_dir(&sessions).unwrap();
        let file = sessions.join("ordinary.jsonl");
        std::fs::write(&file, "{\"rlmDepth\":0}\n").unwrap();
        let file = file.to_str().unwrap();
        let model = pi_ai::types::Model { id:"any-model".into(), provider:"custom-local-or-hosted".into(), ..Default::default() };
        let initial = occ3_modes(Some(file), Some(&model), false, &[], false, false, None).unwrap();
        assert_eq!(initial.textmas_mode, TextMasMode::Off);
        assert!(!initial.latent_enabled && !initial.latent_active && !initial.latent_available);
        textmas_settings::set_mode(file, TextMasMode::Hierarchical).unwrap();
        let enabled = occ3_modes(Some(file), Some(&model), false, &[], false, false, None).unwrap();
        assert_eq!(enabled.textmas_mode, TextMasMode::Hierarchical);
        assert!(enabled.textmas_activity.is_none(), "there is no invented activity before an actual run");
        assert!(!enabled.latent_enabled && !enabled.latent_active && !enabled.latent_available);
        session_settings::save(file, false).unwrap();
        assert_eq!(occ3_modes(Some(file), Some(&model), false, &[], false, false, None).unwrap(), enabled);
        std::fs::write(textmas_settings::preference_path(file), "bad").unwrap();
        assert!(occ3_modes(Some(file), Some(&model), false, &[], false, false, None).is_none());
    }

    #[test]
    fn occ32_display_honors_explicit_off_over_global_or_stale_enrollment() {
        use crate::core::occ::session_settings;
        use pi_ai::types::{Model, Occ3Capability, Occ3Validation, OCC3_API_VERSION_V1, OCC3_PROTOCOL_V1};
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        std::fs::create_dir(&sessions).unwrap();
        let file = sessions.join("native.jsonl");
        std::fs::write(&file, "{\"rlmDepth\":0}\n").unwrap();
        let file = file.to_str().unwrap();
        let model = Model { occ3: Some(Occ3Capability {
            protocol: OCC3_PROTOCOL_V1.into(), api_version: OCC3_API_VERSION_V1.into(),
            connector:"sglang".into(), endpoint:"http://127.0.0.1:30001".into(),
            same_model_only:true, enabled:true, validation:Occ3Validation::LiveVerified,
            fallback:Vec::new(), method:None, artifact_digest:None,
        }), ..Default::default() };
        let tools = vec!["occ3_handoff".to_string()];
        let global = occ3_modes(Some(file), Some(&model), true, &tools, false, false, None).unwrap();
        assert!(global.latent_enabled && global.latent_active && global.latent_available);
        session_settings::save(file, false).unwrap();
        assert!(occ3_modes(Some(file), Some(&model), true, &tools, false, false, None).unwrap().latent_enabled);
        session_settings::save_latent(file, false).unwrap();
        let vetoed = occ3_modes(Some(file), Some(&model), true, &tools, false, false, None).unwrap();
        assert!(!vetoed.latent_enabled && !vetoed.latent_active && vetoed.latent_available);
        session_settings::save_latent(file, true).unwrap();
        let dead = occ3_modes(Some(file), Some(&model), true, &tools, true, false, None).unwrap();
        assert!(dead.latent_enabled && !dead.latent_active);
        let not_enrolled = occ3_modes(Some(file), Some(&model), true, &[], false, false, None).unwrap();
        assert!(not_enrolled.latent_enabled && !not_enrolled.latent_active);
    }

    #[test]
    fn artifact_ids_hash_session_type_and_path() {
        let first = create_artifact_reference("s1", "C:/w", "skill", Some("C:/w/a.md")).unwrap();
        let second = create_artifact_reference("s1", "C:/w", "skill", Some("C:/w/a.md")).unwrap();
        let third = create_artifact_reference("s2", "C:/w", "skill", Some("C:/w/a.md")).unwrap();
        assert_eq!(first, second);
        assert_ne!(first.id, third.id);
        assert_eq!(first.id.len(), "artifact_".len() + 16);
        assert_eq!(first.type_, "skill");
    }

    #[test]
    fn missing_path_yields_no_reference() {
        assert!(create_artifact_reference("s1", "C:/w", "theme", None).is_none());
    }

    #[test]
    fn logical_paths_prefer_cwd_relative_then_basename() {
        let cwd = std::env::current_dir().unwrap();
        let inside = cwd.join("sub").join("file.md");
        let (logical, relative) = create_artifact_path_info(&inside.to_string_lossy(), &cwd.to_string_lossy());
        assert_eq!(relative.as_deref(), Some("sub/file.md"));
        assert_eq!(logical, "sub/file.md");

        let outside = if cfg!(windows) { "D:/elsewhere/x.md" } else { "/elsewhere/x.md" };
        let (logical, relative) = create_artifact_path_info(outside, &cwd.to_string_lossy());
        assert_eq!(logical, "x.md");
        assert_eq!(relative, None);
    }

    #[test]
    fn bracketed_paths_stay_logical() {
        assert_eq!(
            create_artifact_path_info("<builtin>", "C:/w"),
            ("<builtin>".to_string(), None)
        );
    }

    #[test]
    fn commands_keep_extension_then_prompt_then_skill_order() {
        let commands = create_agent_connection_commands(
            &[RegisteredCommandEntry {
                invocation_name: "inv".to_string(),
                name: "reg".to_string(),
                description: Some("d".to_string()),
                source_info: AgentConnectionSourceInfo::default(),
            }],
            &[PromptTemplateEntry {
                name: "p".to_string(),
                description: None,
                argument_hint: Some("[x]".to_string()),
                source_info: AgentConnectionSourceInfo::default(),
            }],
            &[SkillEntry {
                name: "s".to_string(),
                description: None,
                source_info: AgentConnectionSourceInfo::default(),
            }],
        );
        let sources: Vec<&str> = commands.iter().map(|command| command.source.as_str()).collect();
        assert_eq!(sources, vec!["extension", "prompt", "skill"]);
        assert_eq!(commands[2].name, "skill:s");
        assert_eq!(commands[1].argument_hint.as_deref(), Some("[x]"));
    }
}
