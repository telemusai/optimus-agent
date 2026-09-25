//! Port of packages/coding-agent/src/core/prompts/rlm.ts
use super::super::kernel::bootstrap::default_rlm_extra_import_labels;

#[derive(Debug, Clone, Default)]
pub struct RlmPromptOptions {
    pub cwd: String,
    pub skills_dir: Option<String>,
    pub installed_skills: Option<Vec<String>>,
    pub messages_path: String,
    pub allow_recursion: Option<bool>,
    pub depth: Option<i64>,
    pub parent_agent: Option<String>,
    pub active_tools: Option<Vec<String>>,
}

/// The `'''` literal used by the `edit` skill guidance line.
const TRIPLE_QUOTE: &str = "'''";

const LONG_RUNNING_WORK_PROMPT: &str = "For slow or independently completing work, use a nonblocking control loop: start the work, record its handle or output location, then end your turn. A `bash()` handle left running beyond its creating cell sends a completion follow-up; when it arrives, inspect the saved handle and continue.\nWhen delegation is available and useful, assign independent substantive tasks to separate workers. Start independent workers without waiting for each one sequentially, and let them run in parallel.\nDo not keep the turn open by polling with `time.sleep()` or shell `sleep`, and do not replace polling with a long blocking `await`. Await only the short operation needed to start work or inspect a result that is already available; otherwise end the turn.";

const USER_PROGRESS_PROMPT: &str = "As the user-facing root agent, when work follows a plan, uses many subagents, or spans multiple turns, proactively give regular concise progress updates so the user does not have to ask. State the current plan, what has completed, any blockers, the proposed fixes, and the next actions. Lead with user-visible outcomes rather than internal process or gate names. Mention internal details only when they explain a blocker or decision. Send an update at meaningful milestones and before ending a turn while work is still running. Do not repeat unchanged status or interrupt short work with unnecessary updates.";

const SIMPLIFIED_TECHNICAL_ENGLISH_PROMPT: &str = "Use simplified technical English by default for user-facing prose.\nPrefer short sentences, common words, and concrete verbs. State one main action or fact per sentence when practical. Use lists for steps or conditions.\nKeep necessary technical terms, names, commands, code, paths, and exact quoted text unchanged. State uncertainty directly.\nTreat this as clarity guidance, not a claim of formal ASD-STE100 compliance. Preserve a user-requested format, tone, terminology, and necessary precision.";

const REPL_CONTROL_PROMPT: &str = "The `ipython` tool is a persistent Python REPL — the agent's long-lived control environment for reasoning, context management, state, tool orchestration, and recursive subcalls. Top-level `await` works directly. Use it to keep intermediate variables, inspect and transform outputs, and write small helper functions. Compaction removes individual variables whose serialized form exceeds 16 MiB; keep large source data on disk and reload it when needed.\n\nPython is the orchestration language: use Python for loops, conditionals, parsing, and state. Use `bash()` to invoke programs, not to write shell programs — no shell loops or heredocs; do those in Python.\n\nDo not assume the REPL is the native runtime of the external thing being investigated. A repository, package, service, dataset, paper, website, benchmark, or API may have its own environment and normal interface. Evaluate external systems through their own interface, then use the REPL to coordinate the process and analyze what comes back.\n\n`bash(command)` starts a shell command in the background and returns a handle immediately: `h = bash('npm test')`. Use `h.pid` / `h.running` for liveness, `h.tail(n)` / `h.output()` for combined stdout+stderr so far, `h.poll()` for a non-blocking result, `h.kill()` to terminate (SIGTERM, escalating to SIGKILL; on Windows kill() uses taskkill /T and detached or reparented descendants may survive), and `await h` (or `await bash('cmd')`) for the completed result with exit_code, output, and duration. Prefer bash() for long-running commands so the turn keeps working. Run shell commands with `bash()`, not `subprocess`/`os.system`: subprocess calls block the kernel, show the user nothing while they run, and spawn processes the harness cannot see or stop.\n\nImportant: do not install dependencies into the kernel just to make an external project import or run there. If a project import, test, script, CLI, or dependency check is needed, run it through that project's own environment and normal command interface. For example, in a Python repo use its documented commands, `uv run ...`, `.venv/bin/python ...`, or the active project interpreter from the repo root. Treat failures from that native environment as the relevant result.\n\nUse Python for reading, searching, and editing files — it gives you reusable variables you can slice, filter, and act on without re-reading. Always assign read/search results to named variables so you can revisit them later.\n\nEach `bash()` call is its own process, so shell state does not persist between calls; use `os.chdir(...)` for the working directory and `os.environ[...]` for environment variables — both persist in the REPL and apply to later `bash()` calls.\n\nPython state in the kernel persists across cells: named variables, helper functions, classes, imports, notes, parsed outputs, and helper data structures all remain available in every later turn. Tool calls are themselves Python `await` expressions, so their return values can be bound to variables and composed into program logic just like any other call.\n\nContinual harness state is available as `rlm.harness` and `rlm.get_harness_state()`. CRUD calls are local to this Prime Agent session by default: `rlm.harness.create_memory(...)`, `rlm.harness.update_memory(...)`, `rlm.harness.delete_memory(...)`, `rlm.harness.create_skill(...)`, `rlm.harness.update_skill(...)`, `rlm.harness.delete_skill(...)`, `rlm.harness.create_subagent(...)`, `rlm.harness.update_subagent(...)`, `rlm.harness.delete_subagent(...)`, `rlm.harness.create_prompt_note(...)`, `rlm.harness.update_prompt_note(...)`, `rlm.harness.delete_prompt_note(...)`, plus `rlm.harness.record_refinement(...)` and `rlm.harness.overview()`. Use `global_=True` only for stable cross-session lessons; Python reserves `global`, so literal `global=True` is invalid syntax.\n\nTerminology: continual harness names the persisted prompt, memory, skill, and subagent layer; RLM names the runtime, Python REPL kernel, and native call interface exposed to the model.\n\nRLM-native call contract: installed Python skills are pre-imported modules. Read the matching SKILL.md and call its documented function, such as `await <skill_import>.<function>(...)`; when a CLI exists, use `<skill_import> ...` from shell. Continual harness skill entries are Python REPL skills with an explicit Python `reference` and `arguments` contract. Spawn a reusable delegation spec with `await rlm('sub-task')`; admission returns a child handle immediately. Results arrive only through an available messaging capability or files, never as an `rlm()` return value. Do not invent non-native wrappers such as `call_skill(...)` or `run_subagent(...)`.";

#[derive(Debug, Clone, Default)]
pub struct ChildAgentDoctrineOptions {
    pub depth: Option<i64>,
    pub parent_agent: Option<String>,
    pub installed_skills: Option<Vec<String>>,
    pub active_tools: Option<Vec<String>>,
}

pub fn build_child_agent_doctrine(options: ChildAgentDoctrineOptions) -> Option<String> {
    let depth = options.depth.unwrap_or(0);
    let has_ipython = match &options.active_tools {
        None => true,
        Some(tools) => tools.iter().any(|tool| tool == "ipython"),
    };
    let has_agent_message = options
        .installed_skills
        .as_ref()
        .map(|skills| skills.iter().any(|skill| skill == "agent_message"))
        .unwrap_or(false);
    if depth <= 0 {
        return None;
    }
    let mut lines = vec![
        format!(
            "You are a child agent spawned by {}. Task prompts are labeled `[task from parent]`.",
            options.parent_agent.unwrap_or_else(|| "your parent agent".to_string())
        ),
        "For a child task, finish with exactly one final line: `RLM_CHILD_STATUS: complete`, `RLM_CHILD_STATUS: blocked`, or `RLM_CHILD_STATUS: failed`. A progress update or stopReason:length is not completion; if an output limit ends a response, send a concise partial-result report and terminal status instead of starting more work.".to_string(),
    ];
    if has_agent_message && has_ipython {
        lines.push(
            "When a task calls for an answer, reply explicitly with `await agent_message.send(message, receiver_role=\"parent\")`. Not every message or task needs a reply; continue cleanup after sending and go idle normally.".to_string(),
        );
    }
    Some(lines.join("\n"))
}

pub fn build_rlm_prompt(options: &RlmPromptOptions) -> String {
    let installed_skills = options.installed_skills.clone().unwrap_or_default();
    let has_agent_message = installed_skills
        .iter()
        .any(|skill| skill == "agent_message");
    let has_agent_observe = installed_skills
        .iter()
        .any(|skill| skill == "agent_observe");
    let allow_recursion = options.allow_recursion.unwrap_or(true);
    let depth = options.depth.unwrap_or(0);
    let active_tools = options.active_tools.clone().unwrap_or_default();
    let has_ipython = match &options.active_tools {
        None => true,
        Some(tools) => tools.iter().any(|tool| tool == "ipython"),
    };
    let can_run_shell_skills = has_ipython || active_tools.iter().any(|tool| tool == "bash");
    let mut parts: Vec<String> = vec![
        "You are a general purpose agent that uses code to solve tasks.".to_string(),
        "You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and iterating one step at a time.".to_string(),
        "When you are done, stop calling tools and state your final answer.".to_string(),
        String::new(),
        LONG_RUNNING_WORK_PROMPT.to_string(),
        String::new(),
    ];
    if depth == 0 {
        parts.push(USER_PROGRESS_PROMPT.to_string());
        parts.push(String::new());
    }
    parts.push(SIMPLIFIED_TECHNICAL_ENGLISH_PROMPT.to_string());
    parts.push(String::new());
    parts.push(format!("Working directory: {}", options.cwd));
    parts.push(format!("Conversation log: {}", options.messages_path));
    parts.push(format!("Recursive agent depth: {depth}"));
    parts.push(format!(
        "Pre-installed Python packages: {}.",
        default_rlm_extra_import_labels().join(", ")
    ));
    parts.push(
        "Install additional packages with `uv pip install <pkg>` (this is a uv-managed venv with no pip module)."
            .to_string(),
    );

    let child_doctrine = build_child_agent_doctrine(ChildAgentDoctrineOptions {
        depth: options.depth,
        parent_agent: options.parent_agent.clone(),
        installed_skills: options.installed_skills.clone(),
        active_tools: options.active_tools.clone(),
    });
    if let Some(doctrine) = child_doctrine {
        parts.push(String::new());
        parts.push(doctrine);
    }

    let mut skill_lines: Vec<String> = Vec::new();
    if let Some(skills_dir) = &options.skills_dir {
        skill_lines.push(format!(
            "Local skills live under {skills_dir}. Read their SKILL.md files when helpful."
        ));
    }
    if !installed_skills.is_empty() {
        let installed = installed_skills
            .iter()
            .map(|skill| format!("`{skill}`"))
            .collect::<Vec<_>>()
            .join(", ");
        if has_ipython {
            skill_lines.push(format!(
                "Installed Python skill modules (pre-imported): {installed}."
            ));
            skill_lines.push(
                "Read each skill's SKILL.md for its API. Inspect a module with `help(<skill>)` or `dir(<skill>)`, then inspect a documented callable with `inspect.signature(<skill>.<function>)`.".to_string(),
            );
        } else if can_run_shell_skills {
            skill_lines.push(format!(
                "Installed skills available as shell commands: {installed}."
            ));
        }
        if can_run_shell_skills {
            skill_lines.push(
                "Each skill is also available as a shell command by the same name: `<skill> ...`. Discover its CLI usage with `<skill> --help`.".to_string(),
            );
        }
        if has_ipython && installed_skills.iter().any(|skill| skill == "edit") {
            skill_lines.push(format!(
                "For targeted existing-file edits, prefer the pre-imported async `edit` skill from the REPL: `old = {}...{}; new = {}...{}; await edit(path=\"pkg/file.py\", old_str=old, new_str=new)`. Use exact old/new strings; if the text contains triple double quotes, use triple single-quoted variables or build `old`/`new` from inspected file slices.",
                TRIPLE_QUOTE, TRIPLE_QUOTE, TRIPLE_QUOTE, TRIPLE_QUOTE
            ));
        }
    }
    if !skill_lines.is_empty() {
        parts.push(String::new());
        parts.extend(skill_lines);
    }
    if has_agent_message {
        parts.push(
            "Agent messaging is restricted to your parent, siblings, and direct children; roots are siblings, and deeper communication relays through the intermediate child.".to_string(),
        );
    }
    if has_agent_observe {
        parts.push(
            "Agent observation is restricted to your parent, siblings, and direct children; roots are siblings, and deeper inspection relays through the intermediate child.".to_string(),
        );
    }

    if depth == 0 && has_ipython {
        parts.push(String::new());
        parts.push(
            "From a daemon-backed depth-0 session, use `await rlm.create_session('task', name='researcher')` to start a separate top-level session. The call returns after the daemon creates the session and accepts its first prompt. Inline and nested sessions cannot use it. `rlm(...)` still creates a child.".to_string(),
        );
    }

    if allow_recursion && has_ipython {
        parts.push(String::new());
        parts.push(
            "A callable `rlm` is already in your global namespace. `await rlm('sub-task')` spawns a child and returns immediately after task admission with `rlm_child_id`, `name`, `session_dir`, and `model`; it never waits for or returns the child's answer.".to_string(),
        );
        parts.push(
            "Choose a stable child name with `await rlm('sub-task', name='api-reviewer')`; names must be unique among siblings. If omitted, the host generates a readable unique name.".to_string(),
        );
        parts.push(
            "A child inherits your model. If a different model is explicitly requested, use `await rlm.find_models(...)` and an exact returned selector. An unavailable requested model fails spawn; decide whether to retry or omit `model`. Children also inherit your thinking level; the `thinking` option overrides it with any level the resolved child model supports, and an unsupported level fails spawn.".to_string(),
        );
        if has_agent_message {
            parts.push(
                "Children reply explicitly with `await agent_message.send(message, receiver_role='parent')` when an answer is needed. Replies and follow-ups arrive as ordinary agent messages; not every task requires a reply.".to_string(),
            );
            parts.push(
                "Use `await agent_message.list_agents()` to discover family and `await rlm.list_subagents()` to recover direct child handles. Use `agent_message.send(..., receiver_role='child', receiver_name=child.name)` for follow-ups.".to_string(),
            );
        } else {
            parts.push(
                "Use `await rlm.list_subagents()` to recover direct child handles after admission."
                    .to_string(),
            );
        }
        if has_agent_observe {
            parts.push(
                "Use `agent_observe` to inspect a child's rollout. Observation is restricted to your parent, siblings, and direct children; relay through the intermediate child for deeper descendants.".to_string(),
            );
        } else {
            parts.push(
                "Inspect files a child wrote when you need to collect its work without an observation capability.".to_string(),
            );
        }
        parts.push(
            "Spawn independent children in separate calls and end your turn instead of awaiting completion. Multiple replies may arrive over multiple turns. Delete a direct child explicitly with `await rlm.delete_subagent(child)` when it is no longer needed.".to_string(),
        );
    }

    if has_ipython {
        parts.push(String::new());
        parts.push(REPL_CONTROL_PROMPT.to_string());
        parts.push("For code search, retrieve candidates with ripgrep (`rg --json` through bash()), `fd` for file discovery, AST or symbol tools first. Keep results in a named variable. Use `from rlm.code_search import from_ripgrep, present`; `candidates = from_ripgrep(result.output)` parses ripgrep matches. `present(candidates)` in a cell with no other output exposes optional file/symbol/grep/reference/test candidates to host-side Jev relevance scoring when enabled. Each candidate has `kind`, `path`, optional `line`, `snippet`, and `mandatory=True` for required evidence. Jev never performs the search or edits code; inspect retained candidates with normal tools. Unscored and uncertain candidates remain available; the original variable stays complete.".to_string());
        if installed_skills.iter().any(|skill| skill == "refine") {
            parts.push(String::new());
            parts.push(
                "Treat continual harness refinement as a small, evidence-backed update after observing a repeated failure or reusable tactic: diagnose the issue, update the smallest relevant continual harness component, validate on the next action, then record the outcome. Use `await refine.run()` to turn repeated delegation patterns into reusable subagent specs, repeated procedures into skills, durable facts/preferences into memories, and narrow behavioral policies into prompt addendums. It returns immediately and is only queued; no harness change is saved until a later Refinement complete outcome appears. Continue working normally after calling it, but never tell the user a refinement is saved or locked in based only on the queued response. Do not rewrite the whole continual harness when a focused memory, skill, prompt note, or subagent spec is enough.".to_string(),
            );
        }
    }

    parts.join("\n")
}

#[derive(Debug, Clone, Default)]
pub struct SubagentGuidanceOptions {
    pub include_refine_examples: Option<bool>,
    pub has_agent_message: Option<bool>,
    pub has_agent_observe: Option<bool>,
}

/// Supplemental sub-agent delegation guidance, appended after the base RLM
/// prompt (see system-prompt.ts). The recursion block covers the mechanics
/// (`rlm(...)` admission and handle management); this block adds the
/// when and why in the same When -> Why -> menu order Claude Code's Agent tool
/// uses. The subagent-spec menu itself renders just after this, inside the
/// harness-state block.
pub fn build_subagent_guidance(options: SubagentGuidanceOptions) -> String {
    let mut lines: Vec<String> = vec![
        "# Delegating to sub-agents".to_string(),
        String::new(),
        "Spawn independent, self-contained work with `handle = await rlm('task', name='worker')`. This returns at admission, not completion; keep the handle to stop or inspect the child later.".to_string(),
        "Optimus delegation uses these native child sessions so the host owns their status, messages, cancellation, and recovery. Do not substitute external agent CLI processes such as `codex exec`, Claude Code, or nested Optimus launches unless the user explicitly requests that external agent. Ordinary build, test, and utility commands still use `bash()`.".to_string(),
        "On resume, use `await rlm.list_subagents()` to recover native children. An earlier transcript error saying native subagents were unimplemented may describe an older build; use the current native interface for new delegated work. If it fails now, report the current error instead of silently falling back to an external agent. Preserve any existing external worker results and account for still-running work before creating overlapping tasks.".to_string(),
    ];
    if options.has_agent_message.unwrap_or(false) {
        lines.push(
            "Ask for an explicit reply when needed. A child replies with `await agent_message.send(message, receiver_role='parent')`; parent follow-ups use `receiver_role='child'` plus the child's name or id. Not every message needs a reply.".to_string(),
        );
    }
    lines.push("Use `await rlm.list_subagents()` after kernel restart or compaction.".to_string());
    if options.has_agent_observe.unwrap_or(false) {
        lines.push("Use `agent_observe` for bounded transcript inspection.".to_string());
    }
    lines.push("Collect bounded direct-child result previews with `await rlm.collect(targets, timeout_ms=0)` without steering the parent. Targets may be handles, names or ids; omit targets for all direct children. A positive timeout waits only for settlement or that deadline, then returns current snapshots. Large outputs belong in files read selectively.".to_string());
    lines.push(
        "Delegate parallel context-heavy research or independent implementation; do a single known lookup, edit, or command inline.".to_string(),
    );
    if options.include_refine_examples.unwrap_or(true) {
        lines.push(
            "Persist genuinely reusable delegation patterns with `await refine.run()`.".to_string(),
        );
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> RlmPromptOptions {
        RlmPromptOptions {
            cwd: "/repo".to_string(),
            messages_path: "/sessions/s.jsonl".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn builds_a_depth_zero_prompt_with_the_progress_note() {
        let prompt = build_rlm_prompt(&base());
        assert!(
            prompt.starts_with("You are a general purpose agent that uses code to solve tasks.")
        );
        assert!(prompt.contains("As the user-facing root agent,"));
        assert!(prompt.contains("Working directory: /repo"));
        assert!(prompt.contains("Conversation log: /sessions/s.jsonl"));
        assert!(prompt.contains("Recursive agent depth: 0"));
        assert!(prompt.contains("Pre-installed Python packages: requests, httpx, yaml (PyYAML), tomli, dotenv (python-dotenv), pandas, numpy, scipy, bs4 (Beautiful Soup), lxml, pydantic, tyro."));
        assert!(prompt.contains("A callable `rlm` is already in your global namespace."));
        assert!(prompt.contains(REPL_CONTROL_PROMPT));
        assert!(prompt.contains("`fd` for file discovery"));
        assert!(prompt.contains("from rlm.code_search import from_ripgrep, present"));
        assert!(!prompt.contains("You are a child agent spawned by"));
    }

    #[test]
    fn child_prompts_include_the_doctrine_and_skip_root_only_blocks() {
        let options = RlmPromptOptions {
            depth: Some(1),
            parent_agent: Some("lead".to_string()),
            installed_skills: Some(vec!["agent_message".to_string(), "refine".to_string()]),
            ..base()
        };
        let prompt = build_rlm_prompt(&options);
        assert!(prompt.contains("You are a child agent spawned by lead."));
        assert!(prompt.contains("RLM_CHILD_STATUS: complete"));
        assert!(prompt.contains(
            "reply explicitly with `await agent_message.send(message, receiver_role=\"parent\")`"
        ));
        assert!(!prompt.contains("As the user-facing root agent,"));
        assert!(!prompt.contains("rlm.create_session"));
        assert!(prompt.contains("Children reply explicitly with `await agent_message.send(message, receiver_role='parent')`"));
        assert!(!prompt.contains("Use `agent_observe` to inspect a child's rollout."));
    }

    #[test]
    fn shell_only_sessions_use_shell_skill_lines() {
        let options = RlmPromptOptions {
            installed_skills: Some(vec!["edit".to_string()]),
            active_tools: Some(vec!["bash".to_string()]),
            ..base()
        };
        let prompt = build_rlm_prompt(&options);
        assert!(prompt.contains("Installed skills available as shell commands: `edit`."));
        assert!(prompt.contains("Discover its CLI usage with `<skill> --help`."));
        assert!(!prompt.contains("Installed Python skill modules (pre-imported)"));
        assert!(!prompt.contains("await rlm('sub-task')"));
    }

    #[test]
    fn build_child_agent_doctrine_requires_a_positive_depth() {
        assert_eq!(
            build_child_agent_doctrine(ChildAgentDoctrineOptions::default()),
            None
        );
        let doctrine = build_child_agent_doctrine(ChildAgentDoctrineOptions {
            depth: Some(2),
            installed_skills: Some(vec!["agent_message".to_string()]),
            active_tools: Some(vec!["ipython".to_string()]),
            ..Default::default()
        })
        .expect("doctrine");
        assert!(doctrine.starts_with("You are a child agent spawned by your parent agent."));
        assert!(doctrine.contains("continue cleanup after sending and go idle normally."));
    }

    #[test]
    fn builds_the_subagent_guidance_block() {
        let guidance = build_subagent_guidance(SubagentGuidanceOptions::default());
        assert!(guidance.starts_with("# Delegating to sub-agents"));
        assert!(guidance.contains("await rlm.collect(targets, timeout_ms=0)"));
        assert!(guidance.contains("Large outputs belong in files read selectively"));
        assert!(guidance.ends_with(
            "Persist genuinely reusable delegation patterns with `await refine.run()`."
        ));
        let without = build_subagent_guidance(SubagentGuidanceOptions {
            include_refine_examples: Some(false),
            has_agent_message: Some(true),
            has_agent_observe: Some(true),
        });
        assert!(without.contains("Ask for an explicit reply when needed."));
        assert!(without.contains("Use `agent_observe` for bounded transcript inspection."));
        assert!(!without.contains("refine.run()"));
    }
}
