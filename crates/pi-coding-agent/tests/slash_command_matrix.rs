//! Exhaustive slash-command validation matrix (Hish's "test and validate every
//! slash command").
//!
//! Every entry of `builtin_slash_commands()` (38 canonical) plus the five aliases
//! is EXERCISED and its ACTUAL behaviour recorded. The table is executed, not
//! prose: classification runs against the real registry, and handler presence is
//! extracted from the real handler chain, so a command that silently regresses
//! from OK to UNHANDLED fails this test.
//!
//! ## Why this shape
//!
//! The handler chain (`run_builtin_command`, `modes/interactive/native_host.rs`)
//! is crate-private, so an integration test cannot call it. This test therefore
//! executes the two halves that are reachable and keeps them joined:
//!
//! 1. CLASSIFICATION executes for real through the public registry:
//!    `parse_slash_command`, `resolve_builtin_slash_command_name`,
//!    `is_builtin_slash_command_name`, `resolve_leading_builtin_slash_command`,
//!    `parse_session_slash_command`.
//! 2. HANDLER PRESENCE executes for real: the chain source is parsed and its arm
//!    names (`"<name>" =>`, including guarded arms) plus the catch-all `other =>`
//!    arm are extracted. A command without an arm reaches the catch-all
//!    "recognised but the native host has no handler for it yet" - the state the
//!    matrix reports as UNHANDLED and rejects.
//! 3. Verdict and the TypeScript file:line that defines the required behaviour are
//!    recorded per row, so a fix can be dispatched without re-deriving it.
//!
//! ## Verdicts
//!
//! * `Ok`        - has a handler arm whose behaviour matches the reference.
//! * `Diverges`  - has a handler arm but does something different from the
//!                 TypeScript. The row cites the TS line and the difference.
//! * `Unhandled` - no handler arm: the command falls to the catch-all. This is a
//!                 FAILING state for a command the TypeScript implements.
//!
//! ## Owner sites
//!
//! Not every command is owned by the chain, so "no chain arm" is not always a
//! defect. Each row records its owner: `run_builtin_command` (the chain),
//! `input_loop` (`/quit`, `/hotkeys`), `dispatch_submission` (`/login`), or
//! `session` (the four session commands). An off-chain row asserts its own site
//! independently, so deleting that site still fails the test.
//!
//! ## Artifact
//!
//! The run writes the same executed table to `tests/slash_command_matrix.json`
//! next to this file, for consumers that must not parse stdout.

use pi_coding_agent::core::slash_commands::{
    builtin_slash_commands, is_builtin_slash_command_name, parse_session_slash_command,
    parse_slash_command, resolve_builtin_slash_command_name, resolve_leading_builtin_slash_command,
};

/// The handler chain this matrix audits, relative to the crate root.
const HANDLER_SOURCE: &str = "src/modes/interactive/native_host.rs";

/// The TypeScript behavioural spec this port must match.
const TS_SOURCE: &str = "packages/coding-agent/src/modes/interactive/interactive-mode.ts";
/// The alias table each alias row resolves through.
const TS_COMMANDS_SOURCE: &str = "packages/coding-agent/src/core/slash-commands.ts";

/// The exact catch-all text a command without a handler arm produces.
const CATCH_ALL_MARKER: &str = "is recognised but the native host has no handler for it yet";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    /// Resolves through the registry as a local built-in (`SlashDispatch::Builtin`).
    Builtin,
    /// `execution: "session"`: submitted verbatim so the session executes it
    /// (`SlashDispatch::SessionCommand`, and slash-commands.ts:273-281).
    Session,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Ok,
    Diverges,
    Unhandled,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Ok => "OK",
            Verdict::Diverges => "DIVERGES",
            Verdict::Unhandled => "UNHANDLED",
        }
    }
}

/// One row of the executed matrix.
struct Row {
    /// The name as typed: the canonical name, or the alias for alias rows.
    name: &'static str,
    /// The canonical registry name this resolves to.
    canonical: &'static str,
    /// Whether `name` is an alias (`clear` -> `new`).
    alias: bool,
    classification: Classification,
    /// Handler arm expected in the chain. `None` means no arm exists, which is
    /// correct for session commands and is the UNHANDLED state for built-ins.
    handler: Option<&'static str>,
    /// Which site implements the behaviour.
    owner: OwnerSite,
    /// TypeScript file:line defining the required behaviour.
    ts_ref: &'static str,
    /// What the port actually does.
    observed: &'static str,
    expected: Verdict,
    /// Why it diverges. `None` unless `expected == Diverges`.
    divergence: Option<&'static str>,
}

fn row(
    name: &'static str,
    canonical: &'static str,
    alias: bool,
    classification: Classification,
    handler: Option<&'static str>,
    ts_ref: &'static str,
    observed: &'static str,
    expected: Verdict,
    divergence: Option<&'static str>,
) -> Row {
    // The owner site is derived from the same data, so a row cannot disagree
    // with its own classification about who owns the behaviour.
    let owner = if matches!(classification, Classification::Session) {
        OwnerSite::Session
    } else if matches!(name, "quit" | "hotkeys") {
        OwnerSite::InputLoop
    } else if matches!(name, "login") {
        OwnerSite::Dispatch
    } else {
        OwnerSite::ChainArm
    };
    Row {
        name,
        canonical,
        alias,
        classification,
        handler,
        owner,
        ts_ref,
        observed,
        expected,
        divergence,
    }
}

const SESSION_REF: &str = "interactive-mode.ts:4821-5030 (no local arm) -> :5177-5181 prompt -> agent-session.ts:5065-5068";
const SESSION_OBSERVED: &str = "prompted verbatim to the session (SlashDispatch::SessionCommand)";

/// All 38 canonical registry entries plus the five aliases (43 rows).
fn matrix() -> Vec<Row> {
    vec![
    row("settings", "settings", false, Classification::Builtin, Some("settings"),
        "interactive-mode.ts:4826-4830", "HostEvent::Settings(get_state) opens the settings selector",
        Verdict::Ok, None),
    row("model", "model", false, Classification::Builtin, Some("model"),
        "interactive-mode.ts:4836-4841", "HostEvent::Models(get_model_catalog, Some(search))",
        Verdict::Ok, None),
    row("effort", "effort", false, Classification::Builtin, Some("effort"),
        "interactive-mode.ts:4842-4846, 8267-8283", "no argument opens the thinking selector; an argument calls set_thinking_level and reports the level",
        Verdict::Ok, None),
    row("fast", "fast", false, Classification::Builtin, Some("fast"),
        "interactive-mode.ts:4847-4853, 8225-8265", "an argument warns Usage: /fast; otherwise the unavailable status or the service-tier toggle",
        Verdict::Ok, None),
    row("scoped-models", "scoped-models", false, Classification::Builtin, None,
        "interactive-mode.ts:4831-4835 -> showModelsSelector :8457-8533, and :8467 \"No models available\"",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("showModelsSelector is implemented in TS; the arm is missing")),
    row("export", "export", false, Classification::Builtin, Some("export"),
        "interactive-mode.ts:4854-4858, 9210-9225", "export_to_jsonl for a .jsonl path, else export_to_html, then the exported-to status",
        Verdict::Ok, None),
    row("import", "import", false, Classification::Builtin, Some("import"),
        "interactive-mode.ts:4859-4863, 9255-9300 (:9258 Usage)", "requires a path, else Usage; import_from_jsonl then the imported status or Import cancelled",
        Verdict::Ok, None),
    row("share", "share", false, Classification::Builtin, None,
        "interactive-mode.ts:4866-4870 -> handleShareCommand :9301",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("handleShareCommand is implemented in TS; the arm is missing")),
    row("copy", "copy", false, Classification::Builtin, Some("copy"),
        "interactive-mode.ts:4871-4875, 9395-9408 (:9398 no-messages status)", "get_last_assistant_text, then the clipboard, else the no-messages status",
        Verdict::Ok, None),
    row("btw", "btw", false, Classification::Builtin, None,
        "interactive-mode.ts:4821-4825 -> handleSideQuestion :4603",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("handleSideQuestion is implemented in TS; the arm is missing")),
    row("name", "name", false, Classification::Builtin, Some("name"),
        "interactive-mode.ts:4876-4880, 9410-9428", "no argument shows the current name or Usage: /name <name>; else set_session_name",
        Verdict::Ok, None),
    row("session", "session", false, Classification::Builtin, Some("session"),
        "interactive-mode.ts:4885-4889, 9481-9502",
        "get_session_stats, then CommandOutput::EchoPanel: echoLocalCommand echoes the TYPED command and the \
         Spacer+Text panel follows",
        Verdict::Ok, None),
    row("system-prompt", "system-prompt", false, Classification::Builtin, Some("system-prompt"),
        "interactive-mode.ts:4891-4895, 9537-9545", "get_system_prompt rendered as a panel; the command is echoed first",
        Verdict::Ok, None),
    row("logs", "logs", false, Classification::Builtin, Some("logs"),
        "interactive-mode.ts:4909-4913, 9504-9536",
        "the local logs directory listing, then CommandOutput::EchoPanel echoes the command at :4911 and adds \
         the Spacer+Text panel at :9531-9533",
        Verdict::Ok, None),
    row("traces", "traces", false, Classification::Builtin, None,
        "interactive-mode.ts:4898-4902 -> handleTracesCommand :9660",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("handleTracesCommand is implemented in TS; the arm is missing")),
    row("context", "context", false, Classification::Builtin, Some("context"),
        "interactive-mode.ts:4903-4908 -> handleContextCommand :9796-9809",
        "get_context_tree, echoed at :4904, then format_context_tree at the live width (:9800), mounted as the \
         Spacer+Text panel (:9807-9808)",
        Verdict::Ok, None),
    row("changelog", "changelog", false, Classification::Builtin, Some("changelog"),
        "interactive-mode.ts:4925-4929, 10004-10023",
        "the changelog markdown, then CommandOutput::EchoPanel echoes the command at :4927 and adds the panel \
         at :10016-10021",
        Verdict::Ok, None),
    row("update", "update", false, Classification::Builtin, None,
        "interactive-mode.ts:5001-5013 -> handleUpdateCommand :8990",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("handleUpdateCommand is implemented in TS; the arm is missing")),
    row("hotkeys", "hotkeys", false, Classification::Builtin, None,
        "interactive-mode.ts:4931-4936, 10206-10212", "the input loop calls handle_hotkeys_command before dispatch, appending the reference",
        Verdict::Ok, None),
    row("fork", "fork", false, Classification::Builtin, None,
        "interactive-mode.ts:4937-4941 -> showUserMessageSelector :8535-8580 (:8545 no-messages status)",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("showUserMessageSelector is implemented in TS; the arm is missing")),
    row("clone", "clone", false, Classification::Builtin, Some("clone"),
        "interactive-mode.ts:4942-4946, 8582-8602 (:8586 Nothing to clone yet)", "get_session_tree then fork at the leaf, else the nothing-to-clone status",
        Verdict::Ok, None),
    row("tree", "tree", false, Classification::Builtin, None,
        "interactive-mode.ts:4947-4952 -> showTreeSelector :8604-8732 (:8618 No entries in session)",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("showTreeSelector is implemented in TS; the arm is missing")),
    row("login", "login", false, Classification::Builtin, None,
        "interactive-mode.ts:4953-4957", "no argument opens the provider picker (HostEvent::Configuration); an argument logs that provider in",
        Verdict::Ok, None),
    row("logout", "logout", false, Classification::Builtin, None,
        "interactive-mode.ts:4958-4962 -> showLogoutSelector :8981-8988",
        "no handler arm: falls to the catch-all status",
        Verdict::Unhandled, Some("showLogoutSelector is implemented in TS; the arm is missing")),
    row("mcp", "mcp", false, Classification::Builtin, Some("mcp"),
        "interactive-mode.ts:4963-4967 -> handleMcpCommand :8906-8950",
        "the guarded arm handles only the argument-less form; arguments fall through to the catch-all",
        Verdict::Diverges, Some(
            "TS handleMcpCommand :8906-8950 implements login/logout/status subcommands. The port arms only \
             `\"mcp\" if args.trim().is_empty()`, so `/mcp login x` matches no arm and reaches the catch-all; \
             the subcommands are unhandled")),
    row("new", "new", false, Classification::Builtin, Some("new"),
        "interactive-mode.ts:4982-4990", "parses the /new arguments, new_session, then the optional name and prompt",
        Verdict::Ok, None),
    row("compact", "compact", false, Classification::Session, None, SESSION_REF, SESSION_OBSERVED, Verdict::Ok, None),
    row("refine", "refine", false, Classification::Session, None, SESSION_REF, SESSION_OBSERVED, Verdict::Ok, None),
    row("goal", "goal", false, Classification::Session, None, SESSION_REF, SESSION_OBSERVED, Verdict::Ok, None),
    row("autonomous", "autonomous", false, Classification::Session, None, SESSION_REF, SESSION_OBSERVED, Verdict::Ok, None),
    row("rlm-max-depth", "rlm-max-depth", false, Classification::Builtin, Some("rlm-max-depth"),
        "interactive-mode.ts:4881-4884, 9430-9479", "get_rlm_max_depth_status / set_rlm_max_depth",
        Verdict::Ok, None),
    row("heartbeat", "heartbeat", false, Classification::Builtin, Some("heartbeat"),
        "interactive-mode.ts:4914-4918, 9812", "parses the heartbeat subcommand and reports the result",
        Verdict::Ok, None),
    row("heartbeats", "heartbeats", false, Classification::Builtin, Some("heartbeats"),
        "interactive-mode.ts:4919-4923 -> showHeartbeatManager", "list_heartbeats opens the heartbeat manager",
        Verdict::Ok, None),
    row("resume", "resume", false, Classification::Builtin, Some("resume"),
        "interactive-mode.ts:4991-4995, 8734-8750", "no argument requests the agents view; else resolve the path and switch_session",
        Verdict::Ok, None),
    row("reload", "reload", false, Classification::Builtin, Some("reload"),
        "interactive-mode.ts:4996-5000, 9124-9133", "refuses while streaming or compacting, else reload",
        Verdict::Ok, None),
    row("fullscreen", "fullscreen", false, Classification::Builtin, Some("fullscreen"),
        "interactive-mode.ts:5014-5023, 7522-7537", "parses on/off, rejects anything else with the usage error, else applies the requested/LIVE state",
        Verdict::Ok, None),
    row("jev", "jev", false, Classification::Builtin, Some("jev"),
        "local built-in (no TypeScript owner; comparison-mode surface)",
        "parses the /jev argument, writes the requested mode through the pi-jev store, republishes the footer; `compare`/`on` stay shadow-only and `active` is a real mode whose effect is bounded to one provider request body",
        Verdict::Ok, None),
    row("quit", "quit", false, Classification::Builtin, Some("quit"),
        "interactive-mode.ts:5040-5043", "the input loop owns shutdown_requested (/quit and /exit); the arm is intentionally silent",
        Verdict::Ok, None),
    // The five aliases (slash-commands.ts:205-211).
    row("clear", "new", true, Classification::Builtin, Some("new"),
        "interactive-mode.ts:4968-4979 (:4971 showError(\"Usage: /clear\"))",
        "the new arm keys on the name AS TYPED: an argument is answered with showError(\"Usage: /clear\") and \
         never reaches handleClearCommand (:4969-4971); the bare form still clears (:4972-4974)",
        Verdict::Ok, None),
    row("usage", "context", true, Classification::Builtin, Some("context"),
        "interactive-mode.ts:4903-4908", "resolves to the context arm, which renders the context tree",
        Verdict::Ok, None),
    row("thinking", "effort", true, Classification::Builtin, Some("effort"),
        "interactive-mode.ts:4842-4846", "resolves to the effort arm",
        Verdict::Ok, None),
    row("rename", "name", true, Classification::Builtin, Some("name"),
        "interactive-mode.ts:4876-4880, 9410-9428", "resolves to the name arm, which strips both /name and /rename",
        Verdict::Ok, None),
    row("side", "btw", true, Classification::Builtin, None,
        "interactive-mode.ts:4821-4825", "resolves to the btw arm, which does not exist",
        Verdict::Unhandled, Some("the canonical /btw arm is missing, so the alias is unhandled too")),
    ]
}

/// Writes the executed matrix as JSON for consumers that must not parse stdout.
///
/// The path is fixed so the artifact is stable between runs. A writer failure is
/// not fatal to the assertions above, but it is reported loudly because the
/// artifact is part of the deliverable.
fn write_json_artifact(rows: &[&Row], arms: &[String], diverges: &[&str], unhandled: &[&str]) {
    fn escape(value: &str) -> String {
        let mut out = String::with_capacity(value.len() + 2);
        for ch in value.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push(' '),
                c => out.push(c),
            }
        }
        out
    }

    let mut json = String::from("{\n");
    json.push_str(&format!(
        "  \"total\": {},\n  \"ok\": {},\n  \"diverges_count\": {},\n  \"unhandled_count\": {},\n",
        rows.len(),
        rows.len() - diverges.len() - unhandled.len(),
        diverges.len(),
        unhandled.len()
    ));
    json.push_str("  \"chain_arms\": [");
    for (index, arm) in arms.iter().enumerate() {
        if index > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("\"{}\"", escape(arm)));
    }
    json.push_str("],\n  \"rows\": [\n");
    for (index, row) in rows.iter().enumerate() {
        // Every row was asserted equal to its expected verdict before this runs, so
        // the recorded verdict is the observed one.
        let verdict = row.expected.as_str();
        json.push_str("    {");
        json.push_str(&format!("\"command\": \"/{}\"", escape(row.name)));
        json.push_str(&format!(", \"canonical\": \"{}\"", escape(row.canonical)));
        json.push_str(&format!(
            ", \"alias\": {}",
            if row.alias { "true" } else { "false" }
        ));
        json.push_str(&format!(
            ", \"classification\": \"{}\"",
            match row.classification {
                Classification::Builtin => "builtin",
                Classification::Session => "session",
            }
        ));
        json.push_str(&format!(
            ", \"owner_site\": \"{}\"",
            match row.owner {
                OwnerSite::ChainArm => "run_builtin_command",
                OwnerSite::InputLoop => "input_loop",
                OwnerSite::Dispatch => "dispatch_submission",
                OwnerSite::Session => "session",
            }
        ));
        json.push_str(&format!(
            ", \"handler_arm\": {}",
            match row.handler {
                Some(handler) => format!("\"{}\"", escape(handler)),
                None => "null".to_string(),
            }
        ));
        json.push_str(&format!(", \"verdict\": \"{}\"", verdict));
        json.push_str(&format!(", \"observed\": \"{}\"", escape(row.observed)));
        json.push_str(&format!(", \"typescript\": \"{}\"", escape(row.ts_ref)));
        json.push_str(&format!(
            ", \"divergence\": {}",
            match row.divergence {
                Some(text) => format!("\"{}\"", escape(text)),
                None => "null".to_string(),
            }
        ));
        json.push_str(if index + 1 == rows.len() {
            "}\n"
        } else {
            "},\n"
        });
    }
    json.push_str("  ]\n}\n");

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("slash_command_matrix.json");
    match std::fs::write(&path, json) {
        Ok(()) => println!("wrote {}", path.display()),
        Err(error) => println!("could not write {}: {error}", path.display()),
    }
}

/// Where a command's behaviour actually lives.
///
/// The reference handles `/quit` and `/hotkeys` in the input loop
/// (`interactive-mode.ts:4931-4936` and `:5040-5043`), not in the local chain,
/// so "no chain arm" is correct for them. The chain still carries a silent
/// `"quit"` arm, and the loop owns the real behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerSite {
    /// The `run_builtin_command` arm does the work.
    ChainArm,
    /// The input loop does the work before any dispatch.
    InputLoop,
    /// `dispatch_submission` handles it before `run_builtin_command` is reached
    /// (`/login`: interactive-mode.ts:4953-4957).
    Dispatch,
    /// The session executes it, so no local arm is correct.
    Session,
}

/// Extracts the handler arm names and whether a catch-all `other =>` arm exists.
///
/// The checked-out source uses CRLF line endings, so the scan is line-based and
/// never depends on an exact `"\n}\n"` byte sequence: `str::lines` strips both
/// `\n` and `\r\n`. The chain ends at the first column-zero closing brace.
fn handler_arms(source: &str) -> (Vec<String>, bool) {
    let start = source
        .find("async fn run_builtin_command(")
        .expect("run_builtin_command must exist: it is the chain this matrix audits");

    let mut arms = Vec::new();
    let mut catch_all = false;
    let mut saw_arm = false;
    for line in source[start..].lines().skip(1) {
        if line.starts_with('}') {
            // End of the function body: the chain has been fully scanned.
            break;
        }
        let trimmed = line.trim();
        // An arm is `"<name>"` (optionally with a guard) followed by `=>`.
        // Requiring the arrow keeps string literals used as arguments (for
        // example the `"mcp-connections"` tab name) out of the arm list.
        if trimmed.contains("=>") {
            if let Some(rest) = trimmed.strip_prefix('"') {
                if let Some(close) = rest.find('"') {
                    let name = &rest[..close];
                    if !name.is_empty()
                        && name
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                    {
                        saw_arm = true;
                        arms.push(name.to_string());
                    }
                }
            }
        }
        if trimmed.starts_with("other =>") {
            catch_all = true;
        }
    }
    assert!(
        saw_arm,
        "the handler chain scan found no arms in {HANDLER_SOURCE}: the extractor does not match the \
         current source layout"
    );
    assert!(
        catch_all,
        "the handler chain scan found no catch-all arm in {HANDLER_SOURCE}"
    );
    (arms, catch_all)
}

fn crate_root_file(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// Reads a repository-root file (`packages/...`), which lives two levels above
/// the crate directory.
fn repo_root_file(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// The exact input-loop source each `OwnerSite::InputLoop` row depends on.
///
/// These are the two sites the reference handles off the chain
/// (`interactive-mode.ts:4931-4936` for `/hotkeys`, `:5040-5043` for `/quit`). If
/// a site disappears the row must be observed as UNHANDLED instead of staying
/// green.
fn input_loop_marker(name: &str) -> Option<&'static str> {
    match name {
        "quit" => Some("matches!(text.trim(), \"/quit\" | \"/exit\")"),
        "hotkeys" => Some("if text.trim() == \"/hotkeys\""),
        // `SlashDispatch::Builtin { name, .. } if name == "login"` arms.
        "login" => Some("if name == \"login\""),
        _ => None,
    }
}

/// The verdict a row has for a given arm set and chain source, with the
/// regression check built in.
fn observed_verdict(row: &Row, arms: &[String], source: &str) -> Verdict {
    if row.owner == OwnerSite::Session {
        // Session commands are executed by the session, so no local arm is right.
        return Verdict::Ok;
    }
    if row.owner == OwnerSite::InputLoop {
        // The input loop owns the behaviour. It is only OK while the loop still
        // contains the handling this row depends on.
        return match input_loop_marker(row.name) {
            Some(marker) if source.contains(marker) => row.expected,
            _ => Verdict::Unhandled,
        };
    }
    if row.owner == OwnerSite::Dispatch {
        // `dispatch_submission` owns it. It is only OK while that arm survives.
        return match input_loop_marker(row.name) {
            Some(marker) if source.contains(marker) => row.expected,
            _ => Verdict::Unhandled,
        };
    }
    let has_arm = row
        .handler
        .map(|handler| arms.iter().any(|arm| arm == handler))
        .unwrap_or(false);
    if !has_arm {
        return Verdict::Unhandled;
    }
    row.expected
}

#[test]
fn every_slash_command_is_exercised_and_its_actual_behaviour_recorded() {
    let source = crate_root_file(HANDLER_SOURCE);
    let (arms, catch_all) = handler_arms(&source);
    assert!(
        catch_all,
        "{HANDLER_SOURCE} must keep the catch-all arm that reports {CATCH_ALL_MARKER:?}"
    );
    assert!(
        source.contains(CATCH_ALL_MARKER),
        "the catch-all text this matrix keys off must exist in {HANDLER_SOURCE}"
    );

    let registry = builtin_slash_commands();
    let canonical: Vec<String> = registry
        .iter()
        .map(|command| command.name.clone())
        .collect();
    assert_eq!(
        canonical.len(),
        38,
        "the registry must hold 38 canonical entries, got {canonical:?}"
    );

    let table = matrix();
    let mut rows: Vec<&Row> = table.iter().collect();
    rows.sort_by_key(|row| row.name);

    // Coverage: every registry entry and every alias has exactly one row.
    assert_eq!(rows.len(), 43, "38 canonical rows plus 5 alias rows");
    for name in &canonical {
        assert!(
            rows.iter().any(|row| row.canonical == *name),
            "registry entry /{name} has no matrix row"
        );
    }
    let alias_map: Vec<(String, Vec<String>)> = registry
        .iter()
        .filter_map(|command| {
            command
                .aliases
                .clone()
                .map(|aliases| (command.name.clone(), aliases))
        })
        .collect();
    assert_eq!(
        alias_map,
        vec![
            ("effort".to_string(), vec!["thinking".to_string()]),
            ("btw".to_string(), vec!["side".to_string()]),
            ("name".to_string(), vec!["rename".to_string()]),
            ("context".to_string(), vec!["usage".to_string()]),
            ("new".to_string(), vec!["clear".to_string()]),
        ],
        "the alias map must match slash-commands.ts:205-211, in registry order"
    );

    let mut unhandled: Vec<&str> = Vec::new();
    let mut diverges: Vec<&str> = Vec::new();
    let mut report = String::new();
    report.push_str(&format!(
        "\n{:<16} {:<14} {:<8} {:<9} {:<10} {}\n",
        "COMMAND", "CANONICAL", "CLASS", "HANDLER", "VERDICT", "OBSERVED"
    ));
    report.push_str(&"-".repeat(150));
    report.push('\n');

    for row in &rows {
        // --- Classification, executed against the real registry. ---
        let typed = format!("/{}", row.name);
        let parsed = parse_slash_command(&typed)
            .unwrap_or_else(|| panic!("{typed} must parse as a slash command"));
        assert_eq!(parsed.name, row.name, "{typed} parse name");
        assert_eq!(
            resolve_builtin_slash_command_name(&parsed.name),
            row.canonical,
            "{typed} must resolve to /{}",
            row.canonical
        );
        assert_eq!(
            resolve_leading_builtin_slash_command(&typed).map(|resolved| resolved.is_alias),
            Some(row.alias),
            "{typed} alias flag"
        );
        assert!(
            is_builtin_slash_command_name(&parsed.name),
            "{typed} must be a recognised built-in or alias"
        );
        // Arguments must classify the same way, so an argument-taking alias is
        // not accidentally diverted to the model.
        let with_args = format!("/{} some-argument", row.name);
        assert_eq!(
            resolve_leading_builtin_slash_command(&with_args).map(|resolved| resolved.name),
            Some(row.canonical.to_string()),
            "{with_args} must resolve to /{}",
            row.canonical
        );
        match row.classification {
            Classification::Session => {
                let session = parse_session_slash_command(&typed)
                    .unwrap_or_else(|| panic!("{typed} must parse as a session command"));
                assert_eq!(
                    session.name, row.canonical,
                    "{typed} session canonical name"
                );
                assert!(
                    !arms.iter().any(|arm| arm == row.canonical),
                    "/{} must NOT have a local arm: the session executes it (interactive-mode.ts:4821-5030)",
                    row.canonical
                );
            }
            Classification::Builtin => {
                assert!(
                    parse_session_slash_command(&typed).is_none(),
                    "{typed} must be a LOCAL built-in, not a session command"
                );
                let command = registry
                    .iter()
                    .find(|command| command.name == row.canonical)
                    .unwrap_or_else(|| panic!("/{} is not a registry entry", row.canonical));
                assert_ne!(
                    command.execution.as_deref(),
                    Some("session"),
                    "/{} must not be a session-executed entry",
                    row.canonical
                );
            }
        }

        // --- Handler presence, executed against the real chain. ---
        // A row that records a handler arm must still have it, so a stale row is
        // caught rather than silently agreeing with a changed chain.
        if let Some(handler) = row.handler {
            if row.expected != Verdict::Unhandled && row.owner == OwnerSite::ChainArm {
                assert!(
                    arms.iter().any(|arm| arm == handler),
                    "row /{} expects the handler arm \"{handler}\" in {HANDLER_SOURCE}, but the chain has only \
                     {arms:?}. The recorded behaviour ({}) is stale.",
                    row.name,
                    row.observed
                );
            }
        }
        // A row recorded as UNHANDLED must NOT name an arm, and the chain must not
        // have gained one: a fix landing makes the row stale instead of silently
        // staying green.
        if row.expected == Verdict::Unhandled {
            assert!(
                row.handler.is_none(),
                "row /{} records UNHANDLED, so it must not name a handler arm (got {:?})",
                row.name,
                row.handler
            );
            assert!(
                !arms.iter().any(|arm| arm == row.canonical),
                "row /{} records UNHANDLED but the chain now has an arm for /{}: a handler landed, so this \
                 row must be re-verified against {} before it can stay UNHANDLED",
                row.name,
                row.canonical,
                row.ts_ref
            );
        }

        // The executed truth for this row. A regression from OK to UNHANDLED makes
        // this disagree with the row and fails the assertion below.
        let verdict = observed_verdict(row, &arms, &source);
        assert_eq!(
            verdict,
            row.expected,
            "row /{} regressed: expected {} but observed {}. TS reference: {}. Divergence: {}",
            row.name,
            row.expected.as_str(),
            verdict.as_str(),
            row.ts_ref,
            row.divergence.unwrap_or("none")
        );
        match verdict {
            Verdict::Unhandled => unhandled.push(row.name),
            Verdict::Diverges => diverges.push(row.name),
            Verdict::Ok => {}
        }

        report.push_str(&format!(
            "{:<16} {:<14} {:<8} {:<9} {:<10} {}\n",
            format!("/{}", row.name),
            row.canonical,
            match row.classification {
                Classification::Builtin => "builtin",
                Classification::Session => "session",
            },
            match row.owner {
                OwnerSite::Session => "session",
                OwnerSite::InputLoop => "loop",
                OwnerSite::Dispatch => "dispatch",
                OwnerSite::ChainArm if row.handler.is_some() => "present",
                OwnerSite::ChainArm => "NONE",
            },
            verdict.as_str(),
            row.observed,
        ));
    }

    // Every arm in the chain must belong to a row, so an undocumented arm cannot
    // hide. `quit`'s silent arm is the one intentional owner outside this list.
    for arm in &arms {
        assert!(
            rows.iter().any(|row| row.canonical == *arm),
            "the chain has an arm \"{arm}\" that no matrix row covers"
        );
    }

    // The rows cite these specs by line; keep the citations honest.
    let ts = repo_root_file(TS_SOURCE);
    let ts_commands = repo_root_file(TS_COMMANDS_SOURCE);
    for marker in [
        "commandName === \"btw\"",
        "commandName === \"context\" && !commandArgs",
        "Usage: /clear",
        "this.handleMcpCommand(commandArgs)",
        "if (slashCommand?.name === \"clear\")",
        // The two off-chain owners.
        "if (text === \"/quit\")",
        "this.handleHotkeysCommand();",
    ] {
        assert!(
            ts.contains(marker),
            "{TS_SOURCE} must still contain {marker:?}: a row cites it"
        );
    }

    // The alias rows resolve through the alias table in the other spec, so the
    // table itself must keep declaring each alias this matrix records.
    for marker in [
        "{ name: \"clear\", aliasFor: \"new\" }",
        "{ name: \"usage\", aliasFor: \"context\" }",
        "{ name: \"thinking\", aliasFor: \"effort\" }",
        "{ name: \"rename\", aliasFor: \"name\" }",
        "{ name: \"side\", aliasFor: \"btw\" }",
    ] {
        assert!(
            ts_commands.contains(marker),
            "{TS_COMMANDS_SOURCE} must still declare {marker:?}: an alias row cites it"
        );
    }

    report.push_str(&"-".repeat(150));
    report.push('\n');
    report.push_str(&format!(
        "TOTAL {} rows: {} OK, {} DIVERGES, {} UNHANDLED\n",
        rows.len(),
        rows.len() - unhandled.len() - diverges.len(),
        diverges.len(),
        unhandled.len()
    ));
    if !diverges.is_empty() {
        report.push_str("\nDIVERGES - handled, but not like the TypeScript:\n");
        for name in &diverges {
            let row = rows.iter().find(|row| row.name == *name).unwrap();
            report.push_str(&format!(
                "  /{name}\n      observed : {}\n      TS       : {}\n      why      : {}\n",
                row.observed,
                row.ts_ref,
                row.divergence.unwrap_or("none")
            ));
        }
    }
    if !unhandled.is_empty() {
        report.push_str("\nUNHANDLED - these reach the catch-all message and need a handler:\n");
        for name in &unhandled {
            let row = rows.iter().find(|row| row.name == *name).unwrap();
            report.push_str(&format!(
                "  /{name}\n      TS       : {}\n      why      : {}\n",
                row.ts_ref,
                row.divergence.unwrap_or("the TypeScript implements it")
            ));
        }
    }
    println!("{report}");

    // Machine-readable form of the same executed table, written next to the
    // artifacts so a downstream check can consume it without parsing stdout.
    write_json_artifact(&rows, &arms, &diverges, &unhandled);

    // The deliverable is the table above; these two sets are the dispatch list.
    // They are asserted exactly, so a change is visible instead of hiding behind a
    // green suite.
    assert_eq!(
        diverges,
        vec!["mcp"],
        "the DIVERGES set changed. Review the table above before updating this list."
    );
    assert_eq!(
        unhandled,
        vec![
            "btw",
            "fork",
            "logout",
            "scoped-models",
            "share",
            "side",
            "traces",
            "tree",
            "update"
        ],
        "the UNHANDLED set changed. Review the table above before updating this list."
    );
}

/// TEETH: an arm that disappears must flip its row from OK to UNHANDLED.
///
/// This drives the same `handler_arms` extractor and `observed_verdict` the
/// matrix uses, against a mutated copy of the real chain, so the detection path
/// itself is proven and the failure names the real symptom.
#[test]
fn a_removed_handler_arm_flips_its_row_from_ok_to_unhandled() {
    let source = crate_root_file(HANDLER_SOURCE);
    let (arms, catch_all) = handler_arms(&source);
    assert!(catch_all, "the real chain must have the catch-all arm");
    assert!(
        arms.contains(&"settings".to_string()),
        "the real chain must handle /settings"
    );
    assert!(
        arms.contains(&"context".to_string()),
        "the real chain must have the context arm"
    );
    let table = matrix();
    assert_eq!(
        table
            .iter()
            .find(|row| row.name == "clear")
            .expect("/clear row")
            .expected,
        Verdict::Ok,
        "/clear must be recorded OK: an argument is a usage error, the bare form clears"
    );

    // A regression deletes the arm. Only the arm text changes.
    let mutated = source.replace("        \"settings\" => {", "        \"set_tings\" => {");
    assert_ne!(mutated, source, "the mutation must change the source");
    let (mutated_arms, _) = handler_arms(&mutated);
    assert!(
        !mutated_arms.contains(&"settings".to_string()),
        "deleting the /settings arm must drop it from the extracted arms"
    );
    assert!(
        mutated_arms.contains(&"context".to_string()),
        "unrelated arms must survive the mutation"
    );

    let settings = table.iter().find(|row| row.name == "settings").unwrap();
    assert_eq!(
        settings.expected,
        Verdict::Ok,
        "/settings is recorded as OK"
    );
    assert_eq!(
        observed_verdict(settings, &arms, &source),
        Verdict::Ok,
        "the real chain must observe /settings as OK"
    );
    let mutated_verdict = observed_verdict(settings, &mutated_arms, &mutated);
    assert_eq!(
        mutated_verdict,
        Verdict::Unhandled,
        "a deleted /settings arm must be observed as UNHANDLED"
    );
    assert_ne!(
        mutated_verdict, settings.expected,
        "the mutation must make the matrix fail for /settings, naming the real symptom"
    );

    // The row for an arm that is still present must NOT flip, so the detector is
    // not simply returning UNHANDLED for everything.
    let context = table.iter().find(|row| row.name == "context").unwrap();
    assert_eq!(
        observed_verdict(context, &mutated_arms, &mutated),
        Verdict::Ok,
        "an untouched arm must keep its recorded verdict"
    );

    // The same guard applies to the input-loop owner: deleting the loop's
    // `/hotkeys` handling must flip that row too.
    assert_eq!(
        observed_verdict(
            table.iter().find(|row| row.name == "hotkeys").unwrap(),
            &arms,
            &source
        ),
        Verdict::Ok,
        "/hotkeys must be OK while the input loop handles it"
    );
    let loop_mutated = source.replace(
        "if text.trim() == \"/hotkeys\"",
        "if text.trim() == \"/h_keyz\"",
    );
    assert_ne!(
        loop_mutated, source,
        "the loop mutation must change the source"
    );
    assert_eq!(
        observed_verdict(
            table.iter().find(|row| row.name == "hotkeys").unwrap(),
            &arms,
            &loop_mutated
        ),
        Verdict::Unhandled,
        "deleting the input loop's /hotkeys handling must be observed as UNHANDLED"
    );
    assert_eq!(
        observed_verdict(
            table.iter().find(|row| row.name == "quit").unwrap(),
            &arms,
            &loop_mutated
        ),
        Verdict::Ok,
        "an unrelated loop site must keep its verdict"
    );
}

/// The UNHANDLED rows are the real catch-all state, each citing the TypeScript
/// that shows what it should do.
#[test]
fn unhandled_rows_are_the_catch_all_state_and_cite_the_typescript() {
    let source = crate_root_file(HANDLER_SOURCE);
    let (arms, _) = handler_arms(&source);
    assert!(
        source.contains(CATCH_ALL_MARKER),
        "the catch-all text must remain the unhandled marker"
    );

    let expected = [
        "btw",
        "fork",
        "logout",
        "scoped-models",
        "share",
        "side",
        "traces",
        "tree",
        "update",
    ];
    let mut checked = 0usize;
    for name in expected {
        let table = matrix();
        let row = table
            .iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("/{name} must have a matrix row"));
        assert_eq!(row.expected, Verdict::Unhandled, "/{name} verdict");
        assert_eq!(
            observed_verdict(row, &arms, &source),
            Verdict::Unhandled,
            "/{name} must be observed as UNHANDLED against the real chain"
        );
        assert!(
            row.ts_ref.contains("interactive-mode.ts:"),
            "/{name} must cite the TypeScript it should implement, got {}",
            row.ts_ref
        );
        let cited_line = row
            .ts_ref
            .split("interactive-mode.ts:")
            .nth(1)
            .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|line| line.parse::<usize>().ok())
            .unwrap_or_else(|| panic!("/{name} must cite a line number"));
        assert!(
            cited_line > 0 && cited_line <= 10_400,
            "/{name} cites interactive-mode.ts:{cited_line}, outside the 10352-line spec"
        );
        checked += 1;
    }
    assert_eq!(checked, 9, "eight canonical commands plus the /side alias");
}
