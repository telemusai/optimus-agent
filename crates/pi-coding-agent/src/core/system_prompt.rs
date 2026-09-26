//! Port of packages/coding-agent/src/core/system-prompt.ts

use std::collections::BTreeSet;

/// ROOT-CONTRACT v7 (Agent-guidance lane): the advisory hint block's OWNED
/// markers. `format_skill_hint` renders exactly these delimiters, so the
/// request-local refresh can find, replace or remove its OWN block without
/// ever regex-stripping user-supplied text.
pub const JEV_SKILL_HINT_BLOCK_START: &str = "<jev_skill_hint>";
pub const JEV_SKILL_HINT_BLOCK_END: &str = "</jev_skill_hint>";

const IMAGE_DISPLAY_GUIDANCE: &str = "# Showing Images\n\nWhen the user asks to see an image, screenshot, or chart in this chat, use the attach-image skill and emit it with `print(await attach_image(path))` in the Python workspace. Wait for capture or generation to finish before attaching the file. The attachment provides the inline preview when terminal image display is enabled. Local Markdown image links such as `![Screenshot](/tmp/shot.png)` only render as text links; they do not display images. If the user asks to show it again, emit a fresh attachment in that turn. If attachment fails, report the error instead of claiming the image is shown.";

const MATPLOTLIB_DISPLAY_GUIDANCE: &str = "Matplotlib figures in the Python workspace use an inline backend by default: new or changed open figures are attached when the cell finishes. Use `plt.show()` or `fig.show()` to display them explicitly or show them again. Keep the default backend for chat previews; an explicit `matplotlib.use('Agg')` or MPLBACKEND override disables automatic previews. Saved image files can be displayed with the attach-image skill when available.";

use serde::{Deserialize, Serialize};

use crate::core::prompts::rlm::{build_child_agent_doctrine, build_rlm_prompt, build_subagent_guidance, ChildAgentDoctrineOptions, RlmPromptOptions, SubagentGuidanceOptions};
use crate::core::refinement::refinement::{
    format_harness_state_for_prompt, FormatHarnessStateOptions, HarnessState, REFINE_SKILL_NAME,
};
use crate::core::skills::{format_skills_for_prompt, get_python_skill_runtime_info, Skill};

/// `BuildSystemPromptOptions`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildSystemPromptOptions {
    /// Custom system prompt (replaces default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_prompt: Option<String>,
    /// Active tools. Tool schemas carry tool descriptions outside the prompt body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_tools: Option<Vec<String>>,
    /// Optional one-line tool snippets keyed by tool name. Used only for custom prompts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_snippets: Option<serde_json::Map<String, serde_json::Value>>,
    /// Additional guideline bullets appended to the system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_guidelines: Option<Vec<String>>,
    /// Text to append to system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub append_system_prompt: Option<String>,
    /// Working directory.
    pub cwd: String,
    /// Conversation log path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages_path: Option<String>,
    /// Pre-loaded context files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_files: Option<Vec<ContextFile>>,
    /// Pre-loaded skills.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<Skill>>,
    /// ROOT-CONTRACT v7 (Agent-guidance lane): advisory skill hint block.
    /// Appended AFTER the roster block, never inside it; assessment only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_hint: Option<String>,
    /// Whether to include the model-facing rlm recursion guidance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_recursion: Option<bool>,
    /// Fixed recursive-agent depth for this session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_depth: Option<f64>,
    /// Human-readable parent name or id for child communication doctrine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_parent_agent: Option<String>,
    /// @deprecated Explicit legacy SDK injection. AgentSession delivers harness
    /// state in context to preserve prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harness_state: Option<HarnessState>,
    /// Enabled user-configured servers available through the generic kernel MCP API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generic_mcp_servers: Option<Vec<String>>,
}

/// `{ path; content }` context file entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFile {
    pub path: String,
    pub content: String,
}

/// Build the system prompt with tools, guidelines, and context.
pub fn build_system_prompt(options: &BuildSystemPromptOptions) -> String {
    let custom_prompt = options.custom_prompt.clone();
    let selected_tools = options.selected_tools.clone();
    let prompt_guidelines = options.prompt_guidelines.clone();
    let append_system_prompt = options.append_system_prompt.clone();
    let cwd = options.cwd.clone();
    let messages_path = options.messages_path.clone();
    let provided_context_files = options.context_files.clone();
    let provided_skills = options.skills.clone();
    let allow_recursion = options.allow_recursion;
    let harness_state = options.harness_state.clone();

    let prompt_cwd = cwd.replace('\\', "/");
    let prompt_messages_path = messages_path
        .unwrap_or_else(|| "not persisted".to_string())
        .replace('\\', "/");

    let date = current_date();

    let append_section = match &append_system_prompt {
        Some(value) => format!("\n\n{value}"),
        None => String::new(),
    };

    let context_files = provided_context_files.unwrap_or_default();
    let skills = provided_skills.unwrap_or_default();
    let tools = selected_tools.clone().unwrap_or_else(|| vec!["ipython".to_string()]);
    let has_ipython = tools.iter().any(|tool| tool == "ipython");
    let has_bash = tools.iter().any(|tool| tool == "bash");
    // Python-backed skills depend on the persistent workspace. Do not advertise
    // them as shell executables when the session exposes only direct tools.
    let skills: Vec<Skill> = skills.into_iter()
        .filter(|skill| has_ipython || !matches!(skill, Skill::Python(_)))
        .collect();
    let visible_skills: Vec<Skill> = skills
        .iter()
        .filter(|skill| !skill_is_disabled(skill))
        .cloned()
        .collect();
    let visible_python_skill_import_names: Vec<String> = get_python_skill_runtime_info(&visible_skills)
        .into_iter()
        .map(|skill| skill.python.import_name)
        .collect();
    let has_refine_skill = visible_skills
        .iter()
        .any(|skill| skill_name(skill) == REFINE_SKILL_NAME);
    let has_attach_image = has_ipython && visible_skills
        .iter()
        .any(|skill| skill_name(skill) == "attach-image");
    let generic_mcp_section = if has_ipython {
        format_generic_mcp_guidance(options.generic_mcp_servers.as_deref())
    } else {
        String::new()
    };

    if let Some(custom_prompt) = custom_prompt {
        let mut prompt = custom_prompt;
        if !has_ipython && has_bash {
            prompt.push_str(&format!("\n\n{}", crate::core::prompts::rlm::DIRECT_TOOL_PROMPT));
        }

        // Append project context files.
        if !context_files.is_empty() {
            prompt.push_str("\n\n# Project Context\n\n");
            prompt.push_str("Project-specific instructions and guidelines:\n\n");
            for file in &context_files {
                prompt.push_str(&format!("## {}\n\n{}\n\n", file.path, file.content));
            }
        }

        // Append skills section only when the model has a way to inspect skill files.
        let custom_prompt_has_file_access = match &selected_tools {
            None => true,
            Some(tools) => tools.iter().any(|tool| tool == "ipython" || tool == "bash"),
        };
        if custom_prompt_has_file_access && !skills.is_empty() {
            prompt.push_str(&format_skills_for_prompt(&skills));
            if let Some(hint) = &options.skill_hint { prompt.push_str(hint); }
        }

        // Add date and working directory last.
        prompt.push_str(&format!("\nCurrent date: {date}"));
        prompt.push_str(&format!("\nCurrent working directory: {prompt_cwd}"));

        let child_doctrine = build_child_agent_doctrine(ChildAgentDoctrineOptions {
            depth: options.rlm_depth.map(|depth| depth as i64),
            parent_agent: options.rlm_parent_agent.clone(),
            installed_skills: Some(visible_python_skill_import_names.clone()),
            active_tools: Some(tools.clone()),
        });
        if let Some(child_doctrine) = child_doctrine {
            prompt.push_str(&format!("\n\n{child_doctrine}"));
        }

        if let Some(harness_state) = &harness_state {
            prompt.push_str(&format!(
                "\n\n{}",
                format_harness_state_for_prompt(
                    harness_state,
                    FormatHarnessStateOptions {
                        max_entries_per_kind: None,
                        max_refinements: None,
                        max_content_length: None,
                        include_ipython_examples: Some(has_ipython),
                        include_shell_examples: Some(has_bash),
                        include_refine_examples: Some(has_ipython && has_refine_skill),
                    }
                )
            ));
        }

        if !generic_mcp_section.is_empty() {
            prompt.push_str(&format!("\n\n{generic_mcp_section}"));
        }

        if has_attach_image {
            prompt.push_str(&format!("\n\n{IMAGE_DISPLAY_GUIDANCE}"));
        }
        if has_ipython {
            prompt.push_str(&format!("\n\n{MATPLOTLIB_DISPLAY_GUIDANCE}"));
        }

        if !append_section.is_empty() {
            prompt.push_str(&append_section);
        }

        return prompt;
    }

    let mut prompt = build_rlm_prompt(&RlmPromptOptions {
        cwd: prompt_cwd.clone(),
        skills_dir: None,
        installed_skills: Some(visible_python_skill_import_names.clone()),
        messages_path: prompt_messages_path,
        allow_recursion,
        depth: options.rlm_depth.map(|depth| depth as i64),
        parent_agent: options.rlm_parent_agent.clone(),
        active_tools: Some(
            tools
                .iter()
                .filter(|name| name.as_str() == "ipython" || name.as_str() == "bash" || name.as_str() == "edit")
                .cloned()
                .collect(),
        ),
    });

    // Appended AFTER the trained buildRlmPrompt prefix, and before the harness-state
    // menu, so the model reads when/why to delegate and then sees the concrete subagent
    // specs it can match against - the same ordering as Claude Code's Agent tool.
    if allow_recursion.unwrap_or(true) && has_ipython {
        let visible_python_skill_names: BTreeSet<String> = get_python_skill_runtime_info(&visible_skills)
            .into_iter()
            .map(|skill| skill.python.import_name)
            .collect();
        prompt.push_str(&format!(
            "\n\n{}",
            build_subagent_guidance(SubagentGuidanceOptions {
                include_refine_examples: Some(has_refine_skill),
                has_agent_message: Some(visible_python_skill_names.contains("agent_message")),
                has_agent_observe: Some(visible_python_skill_names.contains("agent_observe")),
            })
        ));
    }

    if let Some(harness_state) = &harness_state {
        prompt.push_str(&format!(
            "\n\n{}",
            format_harness_state_for_prompt(
                harness_state,
                FormatHarnessStateOptions {
                    max_entries_per_kind: None,
                    max_refinements: None,
                    max_content_length: None,
                    include_ipython_examples: Some(has_ipython),
                    include_shell_examples: Some(has_bash),
                    include_refine_examples: Some(has_ipython && has_refine_skill),
                }
            )
        ));
    }

    if !generic_mcp_section.is_empty() {
        prompt.push_str(&format!("\n\n{generic_mcp_section}"));
    }

    if has_attach_image {
        prompt.push_str(&format!("\n\n{IMAGE_DISPLAY_GUIDANCE}"));
    }
    if has_ipython {
        prompt.push_str(&format!("\n\n{MATPLOTLIB_DISPLAY_GUIDANCE}"));
    }

    let guidelines = format_prompt_guidelines(prompt_guidelines.as_deref());
    if !guidelines.is_empty() {
        prompt.push_str(&format!("\n\n# Additional Guidance\n\n{guidelines}"));
    }

    // Append project context files.
    if !context_files.is_empty() {
        prompt.push_str("\n\n# Project Context\n\n");
        prompt.push_str("Project-specific instructions and guidelines:\n\n");
        for file in &context_files {
            prompt.push_str(&format!("## {}\n\n{}\n\n", file.path, file.content));
        }
    }

    // Append skills section only when the model has a way to inspect skill files.
    let has_file_access = tools.iter().any(|tool| tool == "ipython" || tool == "bash");
    if has_file_access && !skills.is_empty() {
        prompt.push_str(&format_skills_for_prompt(&skills));
        if let Some(hint) = &options.skill_hint { prompt.push_str(hint); }
    }

    if !append_section.is_empty() {
        prompt.push_str(&append_section);
    }

    prompt
}

fn skill_is_disabled(skill: &Skill) -> bool {
    match skill {
        Skill::Markdown(skill) => skill.base.disable_model_invocation,
        Skill::Python(skill) => skill.base.disable_model_invocation,
    }
}

fn skill_name(skill: &Skill) -> &str {
    match skill {
        Skill::Markdown(skill) => &skill.base.name,
        Skill::Python(skill) => &skill.base.name,
    }
}

/// `new Date()` formatted as `YYYY-MM-DD` in local time.
fn current_date() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

fn format_generic_mcp_guidance(servers: Option<&[String]>) -> String {
    let mut enabled_servers: Vec<String> = servers.unwrap_or(&[]).to_vec();
    enabled_servers.sort();
    enabled_servers.dedup();
    if enabled_servers.is_empty() {
        return String::new();
    }

    let mut lines: Vec<String> = vec![
        "# Generic MCP Connections".to_string(),
        String::new(),
        "Generic MCP connections are accessed through the pre-imported Python `mcp` object in the Python REPL, not as top-level native tool namespaces or installed Python skills.".to_string(),
        format!(
            "Enabled generic MCP servers: {}.",
            enabled_servers
                .iter()
                .map(|server| format!("`{server}`"))
                .collect::<Vec<String>>()
                .join(", ")
        ),
    ];
    for server in &enabled_servers {
        lines.push(format!(
            "For `{server}`, first discover its tools with `await mcp.list_tools(\"{server}\")`, then call one with `await mcp.call_tool(\"{server}\", \"<tool>\", arguments)`."
        ));
    }
    lines.join("\n")
}

fn format_prompt_guidelines(prompt_guidelines: Option<&[String]>) -> String {
    let mut guidelines_list: Vec<String> = Vec::new();
    let mut guidelines_set: BTreeSet<String> = BTreeSet::new();

    for guideline in prompt_guidelines.unwrap_or(&[]) {
        let normalized = guideline.trim();
        if !normalized.is_empty() && !guidelines_set.contains(normalized) {
            guidelines_set.insert(normalized.to_string());
            guidelines_list.push(normalized.to_string());
        }
    }

    guidelines_list
        .iter()
        .map(|guideline| format!("- {guideline}"))
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::skills::{BaseSkill, MarkdownSkill, SkillKind};

    fn markdown_skill(name: &str, disable_model_invocation: bool) -> Skill {
        Skill::Markdown(MarkdownSkill {
            base: BaseSkill {
                name: name.to_string(),
                description: "A skill".to_string(),
                file_path: format!("/skills/{name}/SKILL.md"),
                base_dir: format!("/skills/{name}"),
                source_info: crate::core::source_info::create_synthetic_source_info(
                    &format!("/skills/{name}/SKILL.md"),
                    &crate::core::source_info::SyntheticSourceInfoOptions {
                        source: "local".to_string(),
                        ..Default::default()
                    },
                ),
                disable_model_invocation,
            },
            kind: SkillKind::Markdown,
        })
    }

    #[test]
    fn guidelines_are_trimmed_deduplicated_and_bulleted() {
        assert_eq!(format_prompt_guidelines(None), "");
        assert_eq!(format_prompt_guidelines(Some(&[])), "");
        assert_eq!(
            format_prompt_guidelines(Some(&[
                "  keep me  ".to_string(),
                "keep me".to_string(),
                "   ".to_string(),
                "second".to_string(),
            ])),
            "- keep me\n- second"
        );
    }

    #[test]
    fn generic_mcp_guidance_sorts_dedupes_and_quotes_servers() {
        assert_eq!(format_generic_mcp_guidance(None), "");
        assert_eq!(format_generic_mcp_guidance(Some(&[])), "");
        let text = format_generic_mcp_guidance(Some(&[
            "zeta".to_string(),
            "alpha".to_string(),
            "zeta".to_string(),
        ]));
        assert!(text.starts_with("# Generic MCP Connections"));
        assert!(text.contains("Enabled generic MCP servers: `alpha`, `zeta`."));
        assert!(text.contains("await mcp.list_tools(\"alpha\")"));
        assert!(text.contains("await mcp.call_tool(\"zeta\", \"<tool>\", arguments)"));
        assert!(text.find("`alpha`").unwrap() < text.find("`zeta`").unwrap());
    }

    #[test]
    fn custom_prompt_appends_context_skills_date_and_cwd() {
        let options = BuildSystemPromptOptions {
            custom_prompt: Some("Base".to_string()),
            cwd: "C:\\work\\project".to_string(),
            context_files: Some(vec![ContextFile {
                path: "/ctx/AGENTS.md".to_string(),
                content: "Rules".to_string(),
            }]),
            skills: Some(vec![markdown_skill("alpha", false)]),
            prompt_guidelines: Some(vec!["ignored for custom prompts".to_string()]),
            append_system_prompt: Some("Tail".to_string()),
            ..Default::default()
        };
        let prompt = build_system_prompt(&options);
        assert!(prompt.starts_with("Base"));
        assert!(prompt.contains("# Project Context"));
        assert!(prompt.contains("## /ctx/AGENTS.md\n\nRules\n\n"));
        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("\nCurrent working directory: C:/work/project"));
        assert!(prompt.contains(&format!("\nCurrent date: {}", current_date())));
        assert!(prompt.ends_with("Tail"));
        assert!(!prompt.contains("# Additional Guidance"));
    }

    #[test]
    fn custom_prompt_skips_skills_without_file_access() {
        let options = BuildSystemPromptOptions {
            custom_prompt: Some("Base".to_string()),
            cwd: "/w".to_string(),
            selected_tools: Some(vec!["ipython".to_string()]),
            skills: Some(vec![markdown_skill("alpha", false)]),
            ..Default::default()
        };
        assert!(build_system_prompt(&options).contains("<available_skills>"));

        let no_access = BuildSystemPromptOptions {
            custom_prompt: Some("Base".to_string()),
            cwd: "/w".to_string(),
            selected_tools: Some(vec![]),
            skills: Some(vec![markdown_skill("alpha", false)]),
            ..Default::default()
        };
        assert!(!build_system_prompt(&no_access).contains("<available_skills>"));
    }

    #[test]
    fn default_prompt_includes_guidance_context_and_skills_sections() {
        let options = BuildSystemPromptOptions {
            cwd: "/work".to_string(),
            messages_path: Some("C:\\logs\\messages.jsonl".to_string()),
            prompt_guidelines: Some(vec!["Be direct".to_string()]),
            context_files: Some(vec![ContextFile {
                path: "/ctx/AGENTS.md".to_string(),
                content: "Rules".to_string(),
            }]),
            skills: Some(vec![
                markdown_skill("alpha", false),
                markdown_skill("hidden", true),
            ]),
            ..Default::default()
        };
        let prompt = build_system_prompt(&options);
        assert!(prompt.contains("# Additional Guidance\n\n- Be direct"));
        assert!(prompt.contains("# Project Context"));
        assert!(prompt.contains("<name>alpha</name>"));
        assert!(!prompt.contains("<name>hidden</name>"));
        assert!(prompt.contains("C:/logs/messages.jsonl"));
        let guidance_index = prompt.find("# Additional Guidance").unwrap();
        let context_index = prompt.find("# Project Context").unwrap();
        assert!(guidance_index < context_index);
    }

    #[test]
    fn disabled_skills_do_not_count_as_refine_or_python_skills() {
        let options = BuildSystemPromptOptions {
            cwd: "/work".to_string(),
            skills: Some(vec![markdown_skill("refine", true)]),
            ..Default::default()
        };
        let prompt = build_system_prompt(&options);
        assert!(!prompt.contains("<available_skills>"));
    }

    #[test]
    fn image_display_guidance_requires_an_enabled_skill_and_python_tool() {
        for custom_prompt in [None, Some("Custom prompt".to_string())] {
            let mut options = BuildSystemPromptOptions {
                custom_prompt,
                cwd: "/work".to_string(),
                skills: Some(vec![markdown_skill("attach-image", false)]),
                ..Default::default()
            };
            let prompt = build_system_prompt(&options);
            assert!(prompt.contains("print(await attach_image(path))"));
            assert!(prompt.contains("emit a fresh attachment in that turn"));
            assert!(prompt.contains("Use `plt.show()` or `fig.show()`"));
            options.selected_tools = Some(vec!["bash".to_string()]);
            assert!(!build_system_prompt(&options).contains("# Showing Images"));
            assert!(!build_system_prompt(&options).contains("inline backend by default"));
            options.selected_tools = None;
            options.skills = Some(vec![markdown_skill("attach-image", true)]);
            assert!(!build_system_prompt(&options).contains("# Showing Images"));
        }
    }
}
