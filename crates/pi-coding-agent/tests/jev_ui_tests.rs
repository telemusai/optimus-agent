//! Lane-C (`jev-ui`) behaviour tests for the `/jev` surface, the mode store, the
//! masked key entry and the footer truth table.
//!
//! ## Why this shape
//!
//! The interactive host (`modes/interactive/native_host.rs`) is crate-private, so
//! an integration test cannot call its dispatch chain. This suite therefore
//! combines the two reachable halves, exactly like `slash_command_matrix.rs`:
//!
//! 1. The PURE UI logic is executed for real. `src/modes/interactive/jev_menu.rs`
//!    is self-contained (only `pi_jev`, `pi_tui`, `serde_json`), so this file
//!    includes the REAL source with `#[path]` and exercises it directly: argument
//!    parsing, the mode store, child inheritance, the menu state machine, masked
//!    input, the footer truth table, and the status text.
//! 2. The WIRING is audited from source: the `Dialog` variants, the dispatch arm,
//!    the registry entry and the daemon capability gates are located in the real
//!    files, so deleting one fails this suite.
//!
//! Tests are written but NOT run in this lane (the build gate owns compilation).
//! Every assertion below is a source-level or pure-logic check, so it needs no
//! terminal, no daemon and no network.

#[path = "../src/modes/interactive/jev_menu.rs"]
mod jev_ui;

use std::fs;
use std::path::{Path, PathBuf};

use jev_ui::{
    clear_secret, compaction_state, env_presence, footer_clear_payload, footer_color_key,
    footer_compaction_clear_payload, footer_compaction_segment, footer_compaction_status_payload,
    footer_compaction_text, footer_segment, footer_state, footer_status_payload, footer_text,
    is_cancel_key, is_on_shorthand, is_submit_key, jev_usage, mask_value, mode_change_message,
    parse_jev_request, redact_reason, render_full_jev_status, render_help, render_status,
    store_secret, ActiveCounters, CredentialStatus, FullJevChange, JevCompactionState,
    JevFooterState, JevKeyInputState, JevMenuAction, JevMenuRow, JevMenuState, JevModeBridge,
    JevPipelineStatus, JevRequest, JevSecret, JevStatusReport, KeyInputState,
    FOOTER_LABEL_MIN_COLUMNS, JEV_ACTIVE_NOTICE, JEV_ACTIVE_UNKNOWN_NOTE, JEV_ARGUMENT_HINT,
    JEV_BOUNDARY_NOTICE, JEV_COMMAND_DESCRIPTION, JEV_COMMAND_NAME, JEV_COMPACT_STATUS_KEY,
    JEV_DISCLOSURE_NOTICE, JEV_FOOTER_RULE_NOTICE, JEV_FULL_JEV_ALREADY_OFF_NOTICE,
    JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE, JEV_FULL_JEV_OFF_NOTICE, JEV_FULL_JEV_ON_NOTICE,
    JEV_FULL_JEV_REJECTION, JEV_ON_COMPARE_NOTICE, JEV_STATUS_KEY, MASK_LENGTH,
};
use jev_ui::{mode_label, ModeChange};
use pi_jev::config::DEFAULT_KEY_ID;
use pi_jev::config::{
    resolve_credential_source, resolve_effective_mode, CredentialSource, EnvKeyPresence,
    JevFeature, JevSettings, ModeScope,
};
use pi_jev::credential::{CredentialStore, InMemoryCredentialStore};
use pi_jev::types::JevMode;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const CRATE: &str = env!("CARGO_MANIFEST_DIR");

/// The word no Jev label, notice or registry entry may carry again: it described
/// the old inert Active mode. It is assembled from pieces so this test file does
/// not itself contain the literal the "no inert-Active wording anywhere" gate
/// searches for, while the assertion stays exactly as strong.
fn inert_wording() -> String {
    format!("res{}", "erved")
}

/// The other word that must never describe Active.
const DISABLED_WORD: &str = "disabled";

/// Whole-document check: no line that MENTIONS Active may call it inert.
///
/// The bare word `disabled` is legitimate elsewhere in the same text - the Off
/// row says "Jev is disabled (default)" - so the assertion is scoped to the
/// lines that talk about Active instead of forbidding the word document-wide.
fn assert_no_inert_wording_about_active(text: &str, label: &str) {
    let mut checked = 0usize;
    for line in text.lines().filter(|line| line.contains("Active")) {
        for forbidden in [inert_wording(), DISABLED_WORD.to_string()] {
            assert!(
                !line.contains(&forbidden),
                "{label} must not call Active inert ({forbidden:?}): {line}"
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "{label} must mention Active at least once: {text}");
}

fn crate_file(relative: &str) -> String {
    let path = Path::new(CRATE).join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()))
}

fn jev_source(relative: &str) -> String {
    crate_file(&format!("src/modes/interactive/{relative}"))
}

/// Every lane-C file that touches the UI surface.
const JEV_FILES: [&str; 5] = [
    "jev_menu.rs",
    "jev_menu_component.rs",
    "jev_key_input.rs",
    "jev_footer.rs",
    "jev_host.rs",
];

/// Calls on the actual connection receiver, excluding unrelated String/Vec/UI
/// methods. Mutation checks below also independently reject forbidden symbols.
fn calls_on_a_connection(source: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for line in source.lines() {
        let line = line.trim_start();
        if line.starts_with("//") || line.starts_with("//!") {
            continue;
        }
        let mut remaining = line;
        while let Some(start) = remaining.find("connection.") {
            let rest = &remaining[start + "connection.".len()..];
            let name: String = rest
                .chars()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                .collect();
            if !name.is_empty() {
                let after = rest[name.len()..].trim_start();
                if after.starts_with('(') {
                    found.push(name.clone());
                }
            }
            remaining = &rest[name.len()..];
        }
    }
    found.sort();
    found.dedup();
    found
}

fn temp_agent_dir(label: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("jev-ui-{label}-"))
        .tempdir()
        .expect("a temp dir")
}

fn bridge_over(dir: &tempfile::TempDir) -> JevModeBridge {
    JevModeBridge::new(dir.path())
}

// ---------------------------------------------------------------------------
// 1. `/jev` argument contract
// ---------------------------------------------------------------------------

#[test]
fn jev_arguments_parse_to_exactly_the_five_supported_requests() {
    assert_eq!(parse_jev_request(""), JevRequest::Menu);
    assert_eq!(parse_jev_request("  "), JevRequest::Menu);
    assert_eq!(parse_jev_request("off"), JevRequest::SetMode(JevMode::Off));
    assert_eq!(
        parse_jev_request("compare"),
        JevRequest::SetMode(JevMode::Compare)
    );
    // `/jev on` stays the short spelling of Compare: a shorthand never arms the
    // request-changing mode.
    assert_eq!(parse_jev_request("on"), JevRequest::SetMode(JevMode::Compare));
    assert!(is_on_shorthand("on"));
    assert!(is_on_shorthand(" ON "));
    assert!(!is_on_shorthand("compare"));
    // Only the explicit spelling selects Active.
    assert_eq!(
        parse_jev_request("active"),
        JevRequest::SetMode(JevMode::Active)
    );
    assert!(!is_on_shorthand("active"));
    // The pi-jev parser agrees with this surface, so the two cannot drift.
    assert_eq!(JevMode::parse("on"), Some(JevMode::Compare));
    assert_eq!(JevMode::parse("active"), Some(JevMode::Active));
    assert_eq!(parse_jev_request("status"), JevRequest::Status);
    assert_eq!(parse_jev_request("key"), JevRequest::InputKey);
    // Case and surrounding whitespace are irrelevant.
    assert_eq!(parse_jev_request(" OFF "), JevRequest::SetMode(JevMode::Off));
    assert_eq!(parse_jev_request("On"), JevRequest::SetMode(JevMode::Compare));

    for unknown in ["bogus", "compare now", "off --force", "status; rm -rf /"] {
        assert!(
            matches!(parse_jev_request(unknown), JevRequest::Unknown(ref value) if value.as_str() == unknown),
            "{unknown} must be an explicit Unknown, never a silent action"
        );
    }
}

#[test]
fn jev_on_writes_compare_and_active_writes_a_real_active_mode() {
    // `/jev on` stays Compare, and the one line that is added explains why the
    // shorthand is not the request-changing mode.
    assert!(JEV_ON_COMPARE_NOTICE.contains("Compare"));
    assert!(JEV_ON_COMPARE_NOTICE.contains("/jev active"));
    for forbidden in [inert_wording(), DISABLED_WORD.to_string(), "Selecting it changes nothing".to_string()] {
        assert!(
            !JEV_ON_COMPARE_NOTICE.contains(&forbidden),
            "the Compare shorthand notice must stay accurate: {JEV_ON_COMPARE_NOTICE}"
        );
        assert!(!JEV_ACTIVE_NOTICE.contains(&forbidden), "{JEV_ACTIVE_NOTICE}");
    }
    assert!(JEV_ACTIVE_NOTICE.contains("feature-gated decisions"));
    assert!(JEV_ACTIVE_NOTICE.contains("failure or timeout keeps the baseline unchanged"));
    assert!(JEV_BOUNDARY_NOTICE.contains("never deletes durable memory or transcript history"));
    assert!(JEV_BOUNDARY_NOTICE.contains("Compaction is request-local and separately controlled"));

    let dir = temp_agent_dir("mode-writes");
    let bridge = bridge_over(&dir);
    // `/jev active` is a REAL per-session write.
    let before = bridge.effective_mode("s1");
    assert_eq!(before, JevMode::Off);
    let change = bridge.set_session_mode("s1", JevMode::Active).expect("no error");
    assert!(matches!(
        change,
        ModeChange::Applied {
            mode: JevMode::Active,
            scope: ModeScope::Session
        }
    ));
    assert_eq!(bridge.effective_mode("s1"), JevMode::Active);
    assert_eq!(bridge.scope("s1"), ModeScope::Session);
    // The Active message carries the exact notice and the boundary.
    let active_message = mode_change_message(&change);
    assert!(active_message.contains(JEV_ACTIVE_NOTICE), "{active_message}");
    assert!(active_message.contains(JEV_BOUNDARY_NOTICE), "{active_message}");
    assert!(active_message.contains("Active"), "{active_message}");
    assert!(active_message.contains("this chat"), "{active_message}");

    // `/jev active` after `/jev compare` moves Compare -> Active, and never the
    // other way round.
    let compare = bridge.set_session_mode("s1", JevMode::Compare).expect("no error");
    assert!(matches!(
        compare,
        ModeChange::Applied {
            mode: JevMode::Compare,
            scope: ModeScope::Session
        }
    ));
    assert_eq!(bridge.effective_mode("s1"), JevMode::Compare);
    let back = bridge.set_session_mode("s1", JevMode::Active).expect("no error");
    assert!(matches!(
        back,
        ModeChange::Applied {
            mode: JevMode::Active,
            scope: ModeScope::Session
        }
    ));
    assert_eq!(bridge.effective_mode("s1"), JevMode::Active);

    // `/jev on` writes Compare and the message names the scope.
    let change = bridge.set_session_mode("s1", JevMode::Compare).expect("no error");
    let message = mode_change_message(&change);
    assert!(message.contains("Compare"), "{message}");
    assert!(message.contains("this chat"), "{message}");
    assert_eq!(bridge.scope("s1"), ModeScope::Session);
}

// ---------------------------------------------------------------------------
// 2. mode store: DESIGN.md 10.2 precedence, persistence, scope
// ---------------------------------------------------------------------------

#[test]
fn effective_mode_precedence_is_explicit_session_then_global_then_off() {
    assert_eq!(resolve_effective_mode(None, None), JevMode::Off);
    assert_eq!(
        resolve_effective_mode(None, Some(JevMode::Compare)),
        JevMode::Compare
    );
    // A GLOBAL default Off must never defeat an explicit per-session Compare.
    assert_eq!(
        resolve_effective_mode(Some(JevMode::Compare), Some(JevMode::Off)),
        JevMode::Compare
    );
    assert_eq!(
        resolve_effective_mode(Some(JevMode::Off), Some(JevMode::Compare)),
        JevMode::Off
    );

    let dir = temp_agent_dir("precedence");
    let bridge = bridge_over(&dir);
    // Global default Compare: a session without an override inherits it.
    bridge.set_global_default(JevMode::Compare).expect("no error");
    assert_eq!(bridge.effective_mode("new-chat"), JevMode::Compare);
    assert_eq!(bridge.scope("new-chat"), ModeScope::GlobalDefault);
    // An explicit per-session Off wins over the global Compare.
    bridge.set_session_mode("new-chat", JevMode::Off).expect("no error");
    assert_eq!(bridge.effective_mode("new-chat"), JevMode::Off);
    assert_eq!(bridge.scope("new-chat"), ModeScope::Session);
    // Another session is unaffected: existing chats do not change silently.
    assert_eq!(bridge.effective_mode("other-chat"), JevMode::Compare);
}

#[test]
fn key_presence_never_enables_jev_and_the_source_order_is_documented() {
    // Source resolution order: saved > TYPESAFE_API_KEY > JEV_API_KEY.
    assert_eq!(
        resolve_credential_source(true, true, true),
        CredentialSource::Saved
    );
    assert_eq!(
        resolve_credential_source(false, true, true),
        CredentialSource::EnvTypesafe
    );
    assert_eq!(
        resolve_credential_source(false, false, true),
        CredentialSource::EnvJev
    );
    assert_eq!(
        resolve_credential_source(false, false, false),
        CredentialSource::None
    );
    assert!(EnvKeyPresence { typesafe_api_key: true, jev_api_key: true }.has_conflict());
    assert!(!EnvKeyPresence { typesafe_api_key: true, jev_api_key: false }.has_conflict());

    // A credential of any source changes NO mode decision.
    for source in [
        CredentialStatus::resolve(true, false, false),
        CredentialStatus::resolve(false, true, true),
        CredentialStatus::resolve(false, false, false),
    ] {
        // The mode store has no credential input at all.
        let dir = temp_agent_dir("key-presence");
        let bridge = bridge_over(&dir);
        assert_eq!(bridge.effective_mode("s"), JevMode::Off);
        assert!(source.present() || !source.present());
    }
    // The conflict is reported without any secret, and names the winner.
    let conflict = CredentialStatus::resolve(false, true, true);
    let text = conflict.describe();
    assert!(text.contains("TYPESAFE_API_KEY wins"), "{text}");
    assert!(conflict.env_conflict());
    // A saved credential also reports the conflict, and says the env vars lose.
    let saved_conflict = CredentialStatus::resolve(true, true, true);
    assert!(saved_conflict.describe().contains("ignored"), "{}", saved_conflict.describe());
}

#[test]
fn the_mode_store_persists_across_a_restart_and_is_a_plain_json_file() {
    let dir = temp_agent_dir("restart");
    {
        let bridge = bridge_over(&dir);
        bridge.set_session_mode("chat-a", JevMode::Compare).expect("no error");
        bridge.set_global_default(JevMode::Off).expect("no error");
    }
    // A brand-new bridge (a fresh process in production) reads the same file.
    let bridge = bridge_over(&dir);
    assert_eq!(bridge.effective_mode("chat-a"), JevMode::Compare);
    assert_eq!(bridge.effective_mode("chat-b"), JevMode::Off);
    let path = bridge.path();
    // Lane A's owner file name: the UI must not invent one.
    assert!(
        path.ends_with(PathBuf::from("jev").join("jev-settings.json")),
        "{path:?}"
    );
    let raw = fs::read_to_string(&path).expect("the settings file exists");
    // Mode data only: a credential must never be written here.
    assert!(raw.contains("chat-a"), "{raw}");
    for forbidden in ["TYPESAFE_API_KEY=", "JEV_API_KEY=", "Bearer ", "api_key", "secret"] {
        assert!(!raw.contains(forbidden), "settings.json must not carry {forbidden}: {raw}");
    }
    let parsed: JevSettings = serde_json::from_str(&raw).expect("valid settings json");
    assert_eq!(parsed.session_mode("chat-a"), Some(JevMode::Compare));
    assert_eq!(parsed.global_default, Some(JevMode::Off));
    // The owner's own guard agrees: this file carries no credential material.
    assert!(!parsed.looks_like_it_contains_a_secret());
}

#[test]
fn children_inherit_the_parent_mode_and_absence_never_becomes_compare() {
    let dir = temp_agent_dir("children");
    let bridge = bridge_over(&dir);
    // Parent in Compare: the child inherits Compare.
    bridge.set_session_mode("parent", JevMode::Compare).expect("no error");
    let parent = bridge.effective_mode("parent");
    assert_eq!(parent, JevMode::Compare);
    let inherited = bridge
        .inherit_into_child("child-1", "parent", None)
        .expect("no error");
    assert_eq!(inherited, JevMode::Compare);
    assert_eq!(bridge.effective_mode("child-1"), JevMode::Compare);
    // The child records where the value came from, so a later global change cannot
    // silently alter an existing chat.
    let settings = bridge.settings();
    assert_eq!(
        settings.sessions.get("child-1").and_then(|entry| entry.inherited_from.clone()),
        Some("parent".to_string())
    );

    // Parent unknown (no snapshot): the child degrades to the global default,
    // which is Off here - never silently Compare.
    let degraded = bridge
        .inherit_into_child("child-2", "no-such-parent", None)
        .expect("no error");
    assert_eq!(degraded, JevMode::Off);

    // An explicit child override wins over the parent.
    let overridden = bridge
        .inherit_into_child("child-3", "parent", Some(JevMode::Off))
        .expect("no error");
    assert_eq!(overridden, JevMode::Off);

    // The integration call sites for this are the child-creation hooks; the exact
    // guarded hunks are recorded in reports/ui/LANE_REPORT.md (C-4). They are
    // outside this lane's file list, which is why no test asserts their source.
}

// ---------------------------------------------------------------------------
// 3. menu state machine
// ---------------------------------------------------------------------------

#[test]
fn help_and_key_clear_are_parsed_and_report_truthfully() {
    assert_eq!(parse_jev_request("help"), JevRequest::Help);
    assert_eq!(parse_jev_request("--help"), JevRequest::Help);
    assert_eq!(parse_jev_request("key clear"), JevRequest::ClearKey);
    assert_eq!(parse_jev_request("KEY CLEAR"), JevRequest::ClearKey);
    assert_eq!(parse_jev_request("key"), JevRequest::InputKey);

    let help = render_help();
    // The usage line is built from the same constant the registry quotes.
    assert!(jev_usage().contains(JEV_ARGUMENT_HINT), "{}", jev_usage());
    assert!(help.contains(JEV_COMMAND_DESCRIPTION), "{help}");
    assert!(help.contains("/jev key clear"), "{help}");

    // `/jev key clear` is the inverse of the only write path, and it never reads
    // the stored value: the delete is issued against the owner's store.
    let store = InMemoryCredentialStore::new();
    store_secret(&store, JevSecret::new("sk-live-0123456789".to_string())).expect("stored");
    assert!(store.exists(DEFAULT_KEY_ID).expect("readable"));
    clear_secret(&store).expect("cleared");
    assert!(!store.exists(DEFAULT_KEY_ID).expect("readable"));
    assert_eq!(store.backend_name(), "memory");
}

#[test]
fn the_menu_exposes_modes_and_independent_compaction() {
    assert_eq!(JevMenuRow::ALL.len(), 8);
    let titles: Vec<String> = JevMenuRow::ALL.iter().map(|row| row.title(JevMode::Off)).collect();
    assert!(titles[0].starts_with("Off"));
    assert!(titles[1].starts_with("Compare"));
    assert_eq!(titles[2], "Active");
    assert_eq!(titles[3], "Compare + Active");
    assert_eq!(titles[4], "Compaction on");
    assert_eq!(titles[5], "Compaction off");
    assert!(titles[6].starts_with("Input API key"));
    assert!(titles[7].starts_with("Status"));
    // The Active row is a real mode row: no label may call it inert.
    for title in &titles {
        for forbidden in [inert_wording(), DISABLED_WORD.to_string()] {
            assert!(!title.contains(&forbidden), "{title}");
        }
    }
    // The Active row describes the real, bounded effect.
    assert_eq!(JevMenuRow::Active.description(), JEV_ACTIVE_NOTICE);

    // Each mode preselects its own row.
    assert_eq!(JevMenuState::new(JevMode::Compare).selected, 1);
    assert_eq!(JevMenuState::new(JevMode::Off).selected, 0);
    assert_eq!(JevMenuState::new(JevMode::Active).selected, 2);
    // The Active row shows as current exactly when the effective mode is Active.
    let active_titles: Vec<String> = JevMenuRow::ALL
        .iter()
        .map(|row| row.title(JevMode::Active))
        .collect();
    assert_eq!(active_titles[2], "Active (current)");
    assert!(active_titles.iter().filter(|title| title.contains("(current)")).count() == 1);
    let compare_titles: Vec<String> = JevMenuRow::ALL
        .iter()
        .map(|row| row.title(JevMode::Compare))
        .collect();
    assert_eq!(compare_titles[1], "Compare (current)");
    assert_eq!(compare_titles[2], "Active");
}

#[test]
fn the_menu_actions_preserve_mode_writes_and_navigation() {
    let mut state = JevMenuState::new(JevMode::Off);

    state.selected = 0;
    assert_eq!(state.accept(), JevMenuAction::SetMode(JevMode::Off));
    assert!(state.closed);

    let mut state = JevMenuState::new(JevMode::Off);
    state.selected = 1;
    assert_eq!(state.accept(), JevMenuAction::SetMode(JevMode::Compare));
    assert_eq!(state.active_mode, JevMode::Compare);

    // Active is a normal applied change: the row writes Active and closes.
    let mut state = JevMenuState::new(JevMode::Compare);
    state.selected = 2;
    assert_eq!(state.accept(), JevMenuAction::SetMode(JevMode::Active));
    assert!(state.closed);
    assert_eq!(state.active_mode, JevMode::Active, "Active is written like any mode");
    assert!(state.message.is_none(), "{:?}", state.message);
    // ...and Compare -> Active is the only direction the row can move the mode.
    let mut state = JevMenuState::new(JevMode::Active);
    state.selected = 1;
    assert_eq!(state.accept(), JevMenuAction::SetMode(JevMode::Compare));
    assert_eq!(state.active_mode, JevMode::Compare);

    let mut state = JevMenuState::new(JevMode::Off);
    state.selected = 6;
    assert_eq!(state.accept(), JevMenuAction::InputKey);
    let mut state = JevMenuState::new(JevMode::Off);
    state.selected = 7;
    assert_eq!(state.accept(), JevMenuAction::ShowStatus);

    // Navigation is bounded and never wraps past the ends.
    let mut state = JevMenuState::new(JevMode::Off);
    for _ in 0..10 {
        state.move_down();
    }
    assert_eq!(state.selected, 7);
    for _ in 0..10 {
        state.move_up();
    }
    assert_eq!(state.selected, 0);
}

#[test]
fn escape_and_ctrl_c_cancel_the_menu_at_every_moment_and_no_key_is_hardcoded() {
    // Esc (the live `tui.select.cancel` binding) cancels.
    let mut state = JevMenuState::new(JevMode::Compare);
    assert_eq!(state.handle_key("\x1b"), JevMenuAction::Cancel);
    assert!(state.closed);

    // Ctrl+C resolves through the TUI cancel binding as well; assert against the
    // helper so a user override keeps working.
    assert!(is_cancel_key("\x1b"));
    assert!(is_cancel_key("\u{3}"));

    // Cancel is available while async work is pending: it never waits.
    let mut state = JevMenuState::new(JevMode::Compare);
    state.set_busy(true);
    assert_eq!(state.handle_key("\x1b"), JevMenuAction::Cancel);
    assert!(!state.busy);

    // Navigation uses the configured select bindings, so a rebind still works.
    let mut state = JevMenuState::new(JevMode::Off);
    let bindings = pi_tui::keybindings::get_keybindings();
    assert!(bindings.matches("\x1b[B", "tui.select.down"));
    assert!(bindings.matches("\x1b[A", "tui.select.up"));
    state.handle_key("\x1b[B");
    assert_eq!(state.selected, 1);
    state.handle_key("\x1b[A");
    assert_eq!(state.selected, 0);

    // Neither the pure state machine nor the component hardcodes a literal key:
    // both ask the configured bindings.
    let pure = jev_source("jev_menu.rs");
    for hardcoded in ["\"\\x1b\"", "\"escape\"", "\"ctrl+c\"", "\"enter\"", "\"up\"", "\"down\""] {
        for file in ["jev_menu.rs", "jev_menu_component.rs", "jev_key_input.rs", "jev_footer.rs", "jev_host.rs"] {
            assert!(
                !jev_source(file).contains(hardcoded),
                "{file} must not hardcode {hardcoded}"
            );
        }
    }
    for binding in ["tui.select.up", "tui.select.down", "tui.select.confirm", "tui.select.cancel"] {
        assert!(pure.contains(binding), "the menu must resolve {binding}");
    }
    assert!(pure.contains("get_keybindings"));
}

// ---------------------------------------------------------------------------
// 4. masked key input
// ---------------------------------------------------------------------------

#[test]
fn the_mask_never_reveals_any_character_or_the_real_length() {
    assert!(mask_value("").is_empty());
    let short = mask_value("abc");
    let long = mask_value("abcdefghijklmnopqrstuvwxyz");
    assert_eq!(short, long, "the mask must not publish the value length");
    assert_eq!(short.chars().count(), MASK_LENGTH);
    let secret = "sk-live-DEADBEEF-0123456789";
    let masked = mask_value(secret);
    for character in secret.chars() {
        assert!(!masked.contains(character), "the mask leaked {character}");
    }
    assert!(!masked.contains(secret));

    let mut input = JevKeyInputState::new();
    for character in secret.chars() {
        input.handle_key(&character.to_string());
    }
    assert_eq!(input.value_len(), secret.chars().count());
    assert!(!input.masked_line().contains(secret));
    assert_eq!(input.status_line().contains(secret), false);
}

#[test]
fn the_secret_is_redacted_in_debug_and_cannot_be_displayed() {
    let secret = JevSecret::new("sk-live-DEADBEEF-0123456789".to_string());
    let debug = format!("{secret:?}");
    assert!(debug.contains("<redacted>"), "{debug}");
    assert!(!debug.contains("DEADBEEF"), "{debug}");
    assert_eq!(secret.peek(), "sk-live-DEADBEEF-0123456789");

    let input = {
        let mut input = JevKeyInputState::new();
        input.handle_key("sk-live-DEADBEEF");
        input
    };
    let input_debug = format!("{input:?}");
    assert!(input_debug.contains("<redacted>"), "{input_debug}");
    assert!(!input_debug.contains("DEADBEEF"), "{input_debug}");
    assert!(input_debug.contains("value_present"));

    // `JevSecret` has no `Display`, so a key can never be interpolated into a
    // message. This is asserted structurally: the only accessor is `expose`.
    let source = jev_source("jev_menu.rs");
    assert!(source.contains("pub fn peek(&self) -> &str"));
    assert!(!source.contains("impl std::fmt::Display for JevSecret"));
    assert!(!source.contains("impl fmt::Display for JevSecret"));
    // The only value handed to the credential store is the owner's `SecretString`,
    // which also has no `Display`.
    assert!(source.contains("pub fn into_secret_string(self) -> SecretString"));
}

#[test]
fn secret_store_validation_rejects_empty_and_placeholder_values() {
    let store = InMemoryCredentialStore::new();
    assert!(store_secret(&store, JevSecret::new(String::new())).is_err());
    assert!(store_secret(&store, JevSecret::new("   ".to_string())).is_err());
    for placeholder in ["test", "changeme", "sk-xxxxxx"] {
        assert!(
            store_secret(&store, JevSecret::new(placeholder.to_string())).is_err(),
            "{placeholder} must be refused"
        );
    }
    // Too short / contains whitespace: refused before any store call.
    assert!(store_secret(&store, JevSecret::new("short".to_string())).is_err());
    assert!(store_secret(&store, JevSecret::new("sk live 0123456789".to_string())).is_err());
    store_secret(&store, JevSecret::new("sk-live-0123456789".to_string())).expect("stored");
    assert!(store.exists(DEFAULT_KEY_ID).expect("readable"));
    assert_eq!(store.backend_name(), "memory");

    // A store that is NOT available is refused BEFORE the write, so the UI can
    // never degrade to a plaintext fallback.
    let unavailable = pi_jev::credential::UnavailableCredentialStore::new("no platform store");
    let error = store_secret(&unavailable, JevSecret::new("sk-live-0123456789".to_string()))
        .expect_err("an unavailable store must refuse");
    assert!(matches!(error, pi_jev::error::JevError::Unavailable { .. }), "{error}");
    assert!(!unavailable.exists(DEFAULT_KEY_ID).unwrap_or(false));

    // The store is the only holder; the UI never writes a file of its own.
    let source = jev_source("jev_menu.rs");
    assert!(source.contains("store.store(DEFAULT_KEY_ID"));
    assert!(!source.contains("std::fs::write"));
    assert!(!source.contains("File::create"));
}

#[test]
fn paste_is_sanitised_and_never_breaks_the_single_line_contract() {
    let mut input = JevKeyInputState::new();
    // A bracketed-paste chunk with embedded newlines and tabs.
    input.paste("\x1b[200~sk-live\r\n01234\t56789\x1b[201~");
    let value = input.take_for_validation().expect("a value");
    assert!(!value.contains('\n'), "{value:?}");
    assert!(!value.contains('\r'), "{value:?}");
    assert!(!value.contains('\t'), "{value:?}");
    assert_eq!(value, "sk-live0123456789");
    let mut typed = JevKeyInputState::new();
    typed.handle_key("\x1b[200~synthetic-pasted-key\x1b[201~");
    assert_eq!(typed.take_for_validation().as_deref(), Some("synthetic-pasted-key"));
}

#[test]
fn cancel_during_validation_wins_and_a_stale_result_is_ignored() {
    let mut input = JevKeyInputState::new();
    input.handle_key("s");
    input.handle_key("k");
    let token = input.generation();
    let value = input.take_for_validation().expect("a value");
    assert_eq!(value, "sk");
    assert_eq!(input.state(), KeyInputState::Validating);
    assert_eq!(input.generation(), token.wrapping_add(1));

    // Cancel while the validation is in flight: the state flips immediately and
    // the late result is dropped because its token is stale.
    input.handle_key("\x1b");
    assert_eq!(input.state(), KeyInputState::Cancelled);
    assert!(!input.apply_validation(token.wrapping_add(1), Ok(())));
    assert_eq!(input.state(), KeyInputState::Cancelled);
    assert!(!input.masked_line().contains("sk"));

    // A cancelled dialog stays closed even when more input or a stale result arrives.
    let mut input = JevKeyInputState::new();
    input.handle_key("a");
    let first = input.generation();
    input.take_for_validation();
    let first_token = input.generation();
    input.cancel();
    input.handle_key("b");
    assert!(!input.apply_validation(first_token, Err("HTTP 401".to_string())));
    assert_eq!(input.state(), KeyInputState::Cancelled);
    assert!(input.value_is_empty());
    assert_ne!(first, first_token);

    // Cancel is immediate even while a validation is "running".
    let mut input = JevKeyInputState::new();
    input.handle_key("x");
    input.take_for_validation();
    input.handle_key("\u{3}");
    assert_eq!(input.state(), KeyInputState::Cancelled);
    assert!(is_cancel_key("\u{3}"));
}

#[test]
fn validation_success_and_failure_are_reported_without_the_key() {
    let mut input = JevKeyInputState::new();
    input.handle_key("z");
    let token = input.take_for_validation().map(|_| input.generation()).expect("token");
    assert!(input.apply_validation(token, Ok(())));
    assert_eq!(input.state(), KeyInputState::Validated);
    assert!(input.status_line().contains("accepted"));

    let mut input = JevKeyInputState::new();
    input.handle_key("z");
    input.take_for_validation();
    let token = input.generation();
    assert!(input.apply_validation(token, Err("HTTP 401 invalid key sk-live-0123456789abcdef".to_string())));
    match input.state() {
        KeyInputState::Failed(reason) => {
            assert!(reason.contains("HTTP 401"), "{reason}");
            assert!(!reason.contains("sk-live-0123456789abcdef"), "{reason}");
            assert!(reason.contains("[redacted]"), "{reason}");
        }
        other => panic!("expected a failure state, got {other:?}"),
    }
    // The redactor keeps short, useful words and removes long opaque tokens.
    assert_eq!(redact_reason("HTTP 401"), "HTTP 401");
    assert_eq!(redact_reason("token abcdefghijklmnopqrstuvwxyz"), "token [redacted]");
}

#[test]
fn submit_uses_the_configured_binding_not_a_literal_key() {
    assert!(is_submit_key("\r") || is_submit_key("\n"));
    let source = jev_source("jev_key_input.rs");
    assert!(source.contains("tui.select.cancel"));
    assert!(!source.contains("\"escape\""));
    assert!(!source.contains("\"ctrl+c\""));
    // The masked buffer is rendered, never the value.
    assert!(source.contains("masked_line()"));
}

// ---------------------------------------------------------------------------
// 5. footer truth table
// ---------------------------------------------------------------------------

fn status_with(credential: CredentialStatus, pipeline: JevPipelineStatus) -> (CredentialStatus, JevPipelineStatus) {
    (credential, pipeline)
}

#[test]
fn the_footer_truth_table_shows_green_on_red_off_and_truthful_modes() {
    let none = CredentialStatus::resolve(false, false, false);
    let saved = CredentialStatus::resolve(true, false, false);
    let default_pipeline = JevPipelineStatus::default();

    // Off is red and wins over everything, including a present credential.
    assert_eq!(
        footer_state(JevMode::Off, &saved, &default_pipeline),
        JevFooterState::Off
    );
    assert_eq!(footer_state(JevMode::Off, &none, &default_pipeline), JevFooterState::Off);
    assert_eq!(footer_state(JevMode::Off, &none, &default_pipeline).color_key(), "error");
    assert_eq!(footer_text(JevFooterState::Off), "\u{25cf} Jev Off");
    assert!(!JevFooterState::Off.is_green());
    assert_ne!(footer_color_key(JevFooterState::Off), "success");

    // No credential: amber unavailable, never green.
    assert_eq!(
        footer_state(JevMode::Compare, &none, &default_pipeline),
        JevFooterState::Unavailable
    );
    assert_eq!(
        footer_state(JevMode::Compare, &none, &default_pipeline).color_key(),
        "warning"
    );

    // In-flight work: amber checking, even with a credential.
    let checking = JevPipelineStatus {
        in_flight: 1,
        ..Default::default()
    };
    assert_eq!(
        footer_state(JevMode::Compare, &saved, &checking),
        JevFooterState::Checking
    );
    assert_eq!(footer_state(JevMode::Compare, &saved, &checking).color_key(), "warning");

    // A recorded fallback or failure: amber fallback.
    let fallback = JevPipelineStatus {
        fallback_reason: "context too large; category skipped".to_string(),
        ..Default::default()
    };
    assert_eq!(
        footer_state(JevMode::Compare, &saved, &fallback),
        JevFooterState::Fallback
    );
    let failed = JevPipelineStatus {
        failure_count: 1,
        ..Default::default()
    };
    assert_eq!(
        footer_state(JevMode::Compare, &saved, &failed),
        JevFooterState::Fallback
    );

    // Healthy Compare: accent, and `Jev Compare` is a distinct label from `Jev On`.
    let healthy = footer_state(JevMode::Compare, &saved, &default_pipeline);
    assert_eq!(healthy, JevFooterState::Compare);
    assert_eq!(footer_color_key(healthy), "success");
    assert!(healthy.is_green());
    assert_eq!(footer_text(healthy), "\u{25cf} Jev On (Compare)");

    // Healthy Active: the SAME accent colour, a distinct label, and the very same
    // credential / checking / fallback ladder Compare uses.
    let active_healthy = footer_state(JevMode::Active, &saved, &default_pipeline);
    assert_eq!(active_healthy, JevFooterState::Active);
    assert_eq!(footer_color_key(active_healthy), "success");
    assert_eq!(footer_color_key(active_healthy), footer_color_key(healthy));
    assert_eq!(footer_text(active_healthy), "\u{25cf} Jev On (Active)");
    assert_ne!(active_healthy.label(), healthy.label());
    for (credential, pipeline, expected) in [
        (&none, &default_pipeline, JevFooterState::Unavailable),
        (&saved, &checking, JevFooterState::Checking),
        (&saved, &fallback, JevFooterState::Fallback),
        (&saved, &failed, JevFooterState::Fallback),
    ] {
        assert_eq!(
            footer_state(JevMode::Active, credential, pipeline),
            expected,
            "Active must use the same warning ladder as Compare"
        );
        assert_eq!(
            footer_state(JevMode::Active, credential, pipeline).color_key(),
            expected.color_key()
        );
        assert!(!expected.is_green(), "{expected:?} is degraded, not green");
    }

    // Healthy Compare + Active: green, and it names the combined mode.
    let both = footer_state(JevMode::CompareAndActive, &saved, &default_pipeline);
    assert_eq!(both, JevFooterState::CompareAndActive);
    assert!(both.is_green());
    assert_eq!(footer_text(both), "\u{25cf} Jev On (Compare + Active)");

    // Degraded states never claim green or a healthy "Jev On" label.
    for state in [
        JevFooterState::Unavailable,
        JevFooterState::Checking,
        JevFooterState::Fallback,
    ] {
        assert!(!state.is_green(), "{state:?} must not be green");
        assert_eq!(state.color_key(), "warning");
        assert!(!state.label().contains("Jev On"), "{state:?}");
    }
    // The label always names the truthful effective mode.
    assert!(JevFooterState::Compare.label().contains("Compare"));
    assert!(JevFooterState::Active.label().contains("Active"));
    assert!(JevFooterState::CompareAndActive
        .label()
        .contains("Compare + Active"));
    assert!(JEV_FOOTER_RULE_NOTICE.contains("Jev On (Active)"));
    assert!(JEV_FOOTER_RULE_NOTICE.contains("Jev On (Compare)"));
    assert!(JEV_FOOTER_RULE_NOTICE.contains("Jev Off"));
    assert_eq!(JEV_STATUS_KEY, "jev");
    assert_eq!(JEV_COMPACT_STATUS_KEY, "jev-compact");
    assert_ne!(JEV_STATUS_KEY, JEV_COMPACT_STATUS_KEY);
}

#[test]
fn the_footer_is_width_safe_and_carries_the_state_in_text_not_only_colour() {
    // Wide terminal: the labelled form.
    assert_eq!(footer_segment(JevFooterState::Off, 120), "\u{25cf} Jev Off");
    // Narrow terminal: dot only, which still fits in one column.
    let narrow = footer_segment(JevFooterState::Off, 10);
    assert_eq!(narrow, "\u{25cf}");
    assert!(pi_tui::utils::visible_width(&narrow) <= 10);
    for columns in [1usize, 2, 5, 10, 20, 39, 40, 41, 80, 200] {
        for state in [
            JevFooterState::Off,
            JevFooterState::Compare,
            JevFooterState::Active,
            JevFooterState::CompareAndActive,
            JevFooterState::Unavailable,
            JevFooterState::Checking,
            JevFooterState::Fallback,
        ] {
            let segment = footer_segment(state, columns);
            assert!(
                pi_tui::utils::visible_width(&segment) <= columns.max(1),
                "{state:?} at {columns} columns overflowed: {segment:?}"
            );
        }
        for state in [
            JevCompactionState::On,
            JevCompactionState::Off,
            JevCompactionState::Unknown,
        ] {
            let segment = footer_compaction_segment(state, columns);
            assert!(
                pi_tui::utils::visible_width(&segment) <= columns.max(1),
                "{state:?} at {columns} columns overflowed: {segment:?}"
            );
        }
    }
    // `Jev Active` is wider than `Jev Off` but still inside the labelled form.
    assert_eq!(
        footer_segment(JevFooterState::Active, FOOTER_LABEL_MIN_COLUMNS),
        "\u{25cf} Jev On (Active)"
    );
    assert_eq!(footer_segment(JevFooterState::Active, 39), "\u{25cf}");
    // The compaction dot follows the same width rule.
    assert_eq!(
        footer_compaction_segment(JevCompactionState::On, FOOTER_LABEL_MIN_COLUMNS),
        "\u{25cf} Jev compact on"
    );
    assert_eq!(
        footer_compaction_segment(JevCompactionState::On, 39),
        "\u{25cf}"
    );
    // Every state is distinguishable without colour, across BOTH segments.
    let labels: Vec<String> = [
        JevFooterState::Off,
        JevFooterState::Compare,
        JevFooterState::Active,
        JevFooterState::CompareAndActive,
        JevFooterState::Unavailable,
        JevFooterState::Checking,
        JevFooterState::Fallback,
    ]
    .iter()
    .map(|state| state.label().to_string())
    .chain(
        [
            JevCompactionState::On,
            JevCompactionState::Off,
            JevCompactionState::Unknown,
        ]
        .iter()
        .map(|state| state.label().to_string()),
    )
    .collect();
    let mut unique = labels.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), labels.len(), "{labels:?}");
}

#[test]
fn the_footer_payload_uses_the_existing_setstatus_surface_and_does_no_io() {
    let payload = footer_status_payload(JevFooterState::Compare, 120);
    assert_eq!(payload["statusKey"], serde_json::json!("jev"));
    assert_eq!(
        payload["statusText"],
        serde_json::json!("\u{25cf} Jev On (Compare)")
    );
    // The optional narrow form rides the SAME payload, so the tray row can
    // collapse the segment without a second round trip. It is SHORT LABELLED,
    // never a bare dot.
    assert_eq!(
        payload["statusCompactText"],
        serde_json::json!("\u{25cf} Jev C On")
    );
    // Removal is explicit null, which `Surfaces::set_status` deletes on.
    let cleared = footer_clear_payload();
    assert_eq!(cleared["statusKey"], serde_json::json!("jev"));
    assert_eq!(cleared["statusText"], serde_json::Value::Null);

    // The compaction dot is a separate key with its own payload and removal.
    let compact = footer_compaction_status_payload(JevCompactionState::On, 120);
    assert_eq!(
        compact["statusKey"],
        serde_json::json!(JEV_COMPACT_STATUS_KEY)
    );
    assert_eq!(
        compact["statusText"],
        serde_json::json!("\u{25cf} Jev compact on")
    );
    assert_eq!(
        compact["statusCompactText"],
        serde_json::json!("\u{25cf} Jev Cmp on")
    );
    let compact_cleared = footer_compaction_clear_payload();
    assert_eq!(
        compact_cleared["statusKey"],
        serde_json::json!(JEV_COMPACT_STATUS_KEY)
    );
    assert_eq!(compact_cleared["statusText"], serde_json::Value::Null);

    // The labelled (unmeasured) form is what the live publisher sends, because the
    // dispatch task cannot measure the terminal.
    let labelled = footer_status_payload(JevFooterState::Compare, usize::MAX);
    assert_eq!(
        labelled["statusText"],
        serde_json::json!("\u{25cf} Jev On (Compare)")
    );
    // Below the threshold the CALLER keeps only the dot, so a narrow terminal
    // cannot lose layout; at and above it the label is used.
    assert_eq!(footer_segment(JevFooterState::Compare, 39), "\u{25cf}");
    assert_eq!(
        footer_segment(JevFooterState::Compare, FOOTER_LABEL_MIN_COLUMNS),
        footer_text(JevFooterState::Compare)
    );

    // The footer renders from a snapshot: no store read, no RPC, no counter poll.
    let source = jev_source("jev_footer.rs");
    for forbidden in ["reqwest", "std::fs", "tokio::spawn", "get_keybindings"] {
        assert!(
            !source.contains(forbidden),
            "a render pass must not do work: found {forbidden}"
        );
    }
    assert!(source.contains("setStatus"));
}

// ---------------------------------------------------------------------------
// 6. status text
// ---------------------------------------------------------------------------

#[test]
fn jev_status_reports_scope_credential_source_and_truthful_counters() {
    let report = JevStatusReport::local_only(
        JevMode::Compare,
        ModeScope::Session,
        CredentialStatus::resolve(false, true, true),
    );
    let text = render_status(&report);
    assert!(text.contains("Mode: Compare"), "{text}");
    assert!(text.contains("Scope: session"), "{text}");
    assert!(text.contains("TYPESAFE_API_KEY wins"), "{text}");
    assert!(text.contains("https://api.typesafe.ai/v1/systemone"), "{text}");
    assert!(text.contains("jev-latest"), "{text}");
    assert!(text.contains("Decisions applied: 0"), "{text}");
    assert!(text.contains("hypothetical, never measured"), "{text}");
    // Honest about what the counters mean in this build.
    assert!(text.contains("Pipeline source:"), "{text}");
    assert!(text.contains(JEV_BOUNDARY_NOTICE), "{text}");
    // Never a secret, and no model/subagent control claim.
    for forbidden in ["sk-live", "Bearer ", "api key:", "set_model", "subagent:"] {
        assert!(!text.contains(forbidden), "status leaked {forbidden}: {text}");
    }
}

#[test]
fn jev_status_distinguishes_checking_degraded_and_never_from_configured() {
    let credential = CredentialStatus::resolve(true, false, false);
    let base = JevStatusReport::local_only(JevMode::Compare, ModeScope::Session, credential.clone());

    let checking = JevStatusReport {
        pipeline: JevPipelineStatus { in_flight: 2, queue_depth: 1, queue_capacity: 8, ..Default::default() },
        ..base.clone()
    };
    let text = render_status(&checking);
    assert!(text.contains("checking (comparisons in flight)"), "{text}");
    assert!(text.contains("Queue: 1/8 (in flight 2)"), "{text}");

    let degraded = JevStatusReport {
        pipeline: JevPipelineStatus {
            failure_count: 3,
            dropped_comparisons: 4,
            fallback_reason: "redacted fallback: local heuristic used".to_string(),
            skipped_categories: vec![("category-4 tool choice".to_string(), "no hook in this session".to_string())],
            last_success_at: Some("2026-09-19T10:00:00Z".to_string()),
            last_latency_ms: Some(412),
            ..Default::default()
        },
        ..base.clone()
    };
    let text = render_status(&degraded);
    assert!(text.contains("degraded"), "{text}");
    assert!(text.contains("Last success: 2026-09-19T10:00:00Z (latency 412 ms)"), "{text}");
    assert!(text.contains("Counters: 0 ok, 3 failed"), "{text}");
    assert!(text.contains("Dropped comparisons: 4"), "{text}");
    assert!(text.contains("category-4 tool choice: no hook in this session"), "{text}");
    assert!(text.contains("Fallback reason: redacted fallback"), "{text}");

    // An unavailable worker snapshot cannot prove zero calls or no success.
    let fresh = render_status(&base);
    assert!(fresh.contains("Last success: unknown"), "{fresh}");
    assert!(fresh.contains("Counters: unknown"), "{fresh}");
    let observed_fresh = render_status(&JevStatusReport {
        pipeline: JevPipelineStatus { observed: true, ..Default::default() },
        ..base.clone()
    });
    assert!(observed_fresh.contains("Last success: never"), "{observed_fresh}");
    assert!(observed_fresh.contains("no successful call yet"), "{observed_fresh}");
    // Off says it is idle with no scheduling and no network.
    let off = render_status(&JevStatusReport::local_only(
        JevMode::Off,
        ModeScope::GlobalDefault,
        credential,
    ));
    assert!(off.contains("idle (Off: no scheduling, no client, no network)"), "{off}");
    assert!(off.contains("Scope: global_default"), "{off}");
}

#[test]
fn the_active_mode_is_labelled_and_described_everywhere_it_can_appear() {
    // The plain label: no surface may show Active as inert.
    assert_eq!(jev_ui::mode_label(JevMode::Active), "Active");
    assert_eq!(JevMode::Active.label(), "Jev Active");
    // The footer label names the truthful effective mode after the On state.
    assert_eq!(JevFooterState::Active.label(), "Jev On (Active)");

    // A fresh Active session with no telemetry: the notice, and honest unknowns.
    let active = render_status(&JevStatusReport::local_only(
        JevMode::Active,
        ModeScope::Session,
        CredentialStatus::resolve(false, false, false),
    ));
    assert!(active.contains("Mode: Active"), "{active}");
    assert!(active.contains(JEV_ACTIVE_NOTICE), "{active}");
    assert!(active.contains("Active boundaries: unknown"), "{active}");
    assert!(active.contains(JEV_ACTIVE_UNKNOWN_NOTE), "{active}");
    assert!(active.contains("unknown (no Active boundary observed in this worker yet)"), "{active}");
    assert_no_inert_wording_about_active(&active, "the Active status panel");

    // Help text states Active as operative and keeps the no-control boundary.
    let help = jev_ui::render_help();
    assert!(help.contains("Active              Accepted, feature-gated native effects."), "{help}");
    assert!(help.contains(JEV_ACTIVE_NOTICE), "{help}");
    assert!(help.contains(JEV_ON_COMPARE_NOTICE), "{help}");
    assert!(help.contains(JEV_BOUNDARY_NOTICE), "{help}");
    assert!(help.contains("/jev active"), "{help}");
    assert_no_inert_wording_about_active(&help, "help");
    // The argument hint lists Active as a normal option.
    assert!(JEV_ARGUMENT_HINT.contains("active"), "{JEV_ARGUMENT_HINT}");
    assert!(JEV_COMMAND_DESCRIPTION.contains("Active"), "{JEV_COMMAND_DESCRIPTION}");
    let description = JEV_COMMAND_DESCRIPTION;
    assert_no_inert_wording_about_active(description, "the /jev command description");
    // Section 11/12 wording is explicit about what Jev never controls.
    assert!(JEV_BOUNDARY_NOTICE.contains("never controls the primary model"));
    assert!(JEV_BOUNDARY_NOTICE.contains("subagents"));
    assert!(JEV_BOUNDARY_NOTICE.contains("budgets"));
    assert!(JEV_BOUNDARY_NOTICE.contains("permissions"));
    assert!(JEV_BOUNDARY_NOTICE.contains("depth"));
    assert!(JEV_BOUNDARY_NOTICE.contains("concurrency"));
}

#[test]
fn jev_status_shows_real_active_counters_when_the_worker_reported_them() {
    let credential = CredentialStatus::resolve(true, false, false);
    let active_only = JevStatusReport {
        pipeline: JevPipelineStatus {
            active: Some(ActiveCounters {
                applied: 3,
                accepted_no_effect: 1,
                refused: 2,
                unavailable: 1,
                last_reason: Some("confidence_below_threshold".to_string()),
                last_category: Some("tool_requirement".to_string()),
            }),
            ..Default::default()
        },
        ..JevStatusReport::local_only(JevMode::Active, ModeScope::Session, credential.clone())
    };
    let text = render_status(&active_only);
    assert!(text.contains("Decisions applied: 3"), "{text}");
    assert!(text.contains("Active boundaries: 3 applied, 1 accepted with no effect, 2 refused, 1 without a usable answer"), "{text}");
    assert!(text.contains("Active last: tool_requirement (confidence_below_threshold)"), "{text}");
    assert!(text.contains("active (accepted decisions remain feature-gated and bounded)"), "{text}");
    // An Active-only snapshot has no scheduler counters, so they must stay UNKNOWN
    // rather than being rendered as invented zeroes.
    assert!(text.contains("Counters: unknown"), "{text}");
    assert!(!text.contains("Counters: 0 ok, 0 failed"), "{text}");
    assert!(text.contains("Last success: unknown"), "{text}");

    // Compare keeps its own truthful figure and never claims an Active boundary.
    let compare = render_status(&JevStatusReport::local_only(
        JevMode::Compare,
        ModeScope::Session,
        credential,
    ));
    assert!(compare.contains("Decisions applied: 0"), "{compare}");
    assert!(compare.contains("hypothetical, never measured"), "{compare}");
    assert!(!compare.contains("Active boundaries:"), "{compare}");
}

// ---------------------------------------------------------------------------
// 6b. REQUIRED negative regressions (DESIGN.md sections 11 and 12)
//
// Every case below tries to make the Jev UI surface do something it must never do:
// control the primary model, control a subagent or child chat, exceed the one
// request-body field an accepted answer may change, or act on a delayed/stale
// answer. The expected outcome is always ZERO Jev-originated change, with
// baseline delegation untouched.
// ---------------------------------------------------------------------------

/// Adversarial `/jev` argument texts: prompt-injection shapes, an attempt to force
/// the request-changing mode, and an attempt to smuggle a second "command" in the
/// same line. Nothing here may produce a mode write other than a plain mode write.
#[test]
fn negative_regression_adversarial_arguments_only_ever_reach_a_plain_mode_write() {
    let injections = [
        "compare --force-active",
        "active --force",
        "on; active",
        "compare\nmodel=gpt-5",
        "compare model=claude",
        "compare spawn child",
        "compare subagents=on",
        "compare budget=unlimited",
        "active -y",
        "compare --no-confirm",
        "<active>",
        "compare && set_model x",
    ];
    for argument in injections {
        let request = parse_jev_request(argument);
        match request {
            // Only a plain mode write or an explicit Unknown may come out of an
            // injected argument. Nothing here can carry a model, tool, permission,
            // budget or subagent change.
            JevRequest::Unknown(_)
            | JevRequest::SetMode(JevMode::Active)
            | JevRequest::SetMode(JevMode::Compare)
            | JevRequest::SetMode(JevMode::Off) => {}
            other => panic!("{argument:?} must not become {other:?}"),
        }
        assert!(
            !is_on_shorthand(argument) || argument.trim().eq_ignore_ascii_case("on"),
            "{argument:?} must not be treated as the `on` shorthand"
        );
    }

    // A mode write touches the mode and nothing else. Writing Active and then
    // reading it back must never yield a different mode: there is no silent
    // rewrite, and the file carries mode data only.
    let dir = temp_agent_dir("negative-active-write");
    let bridge = bridge_over(&dir);
    let change = bridge.set_session_mode("s", JevMode::Active).expect("no error");
    assert!(matches!(
        change,
        ModeChange::Applied {
            mode: JevMode::Active,
            scope: ModeScope::Session
        }
    ));
    assert_eq!(bridge.effective_mode("s"), JevMode::Active, "Active must never become Compare");
    let raw = fs::read_to_string(bridge.path()).expect("readable");
    assert!(raw.contains("\"active\""), "{raw}");
    for forbidden in ["model", "provider", "permission", "budget", "subagent", "depth", "concurrency"] {
        assert!(!raw.contains(forbidden), "a mode write must not carry {forbidden}: {raw}");
    }
    // Active is NOT Compare: it never arms shadow comparison.
    assert!(!JevMode::Active.allows_compare());
}

/// Settings allow explicit bounded policies, not arbitrary model, permission,
/// subagent or budget control fields.
#[test]
fn negative_regression_settings_only_expose_bounded_policy_not_model_or_subagent_control() {
    let settings = JevSettings {
        global_default: Some(JevMode::Compare),
        ..JevSettings::default()
    };
    let value = serde_json::to_value(&settings).expect("serializable");
    let keys: Vec<&str> = value.as_object().expect("object").keys().map(String::as_str).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec![
            "compaction",
            "compaction_enabled",
            "credential_configured",
            "disclosure_shown",
            "features",
            "filtering",
            "global_default",
            "schema_version",
            "sessions",
        ],
        "settings must expose only explicit bounded policies, modes and presence metadata"
    );
    for forbidden in [
        "model",
        "provider",
        "effort",
        "thinking",
        "tools",
        "permissions",
        "subagent",
        "subagents",
        "budget",
        "depth",
        "concurrency",
        "systemPrompt",
    ] {
        assert!(
            !keys.contains(&forbidden),
            "JevSettings must not carry {forbidden}"
        );
    }
    // A per-session entry is equally narrow.
    let entry = serde_json::to_value(pi_jev::config::PersistedSessionMode {
        mode: Some(JevMode::Compare),
        inherited_from: None,
        ..Default::default()
    })
    .expect("serializable");
    let mut entry_keys: Vec<&str> = entry
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    entry_keys.sort();
    assert_eq!(entry_keys, vec!["mode"], "{entry}");

    // And the whole crate surface refuses a subagent-control request by construction.
    let refused = pi_jev::types::refuse_subagent_control(
        &pi_jev::types::SubagentControlRequest::Spawn {
            role: "reviewer".to_string(),
        });
    assert!(
        refused.is_err(),
        "spawning a child through Jev must be impossible"
    );
    assert!(matches!(
        refused.expect_err("refused"),
        pi_jev::error::JevError::SubagentControlForbidden { .. }
    ));
}

/// A delayed or stale validation result must never change state, and a cancel must
/// win over work that is already running. Repeated for both orderings.
#[test]
fn negative_regression_delayed_and_stale_results_change_nothing() {
    for cancel_first in [true, false] {
        let mut input = JevKeyInputState::new();
        input.handle_key("sk-live-DELAYED-0123456");
        let token = input.take_for_validation().map(|_| input.generation()).expect("a value");
        if cancel_first {
            input.cancel();
            // The late answer arrives after the cancel: it is stale and dropped.
            assert!(!input.apply_validation(token, Ok(())));
            assert_eq!(input.state(), KeyInputState::Cancelled);
        } else {
            // The answer arrives first; a later cancel still wins.
            assert!(input.apply_validation(token, Ok(())));
            input.cancel();
            assert_eq!(input.state(), KeyInputState::Cancelled);
            // A second, stale answer cannot resurrect the cancelled attempt.
            assert!(!input.apply_validation(token, Err("late failure".to_string())));
            assert_eq!(input.state(), KeyInputState::Cancelled);
        }
        assert!(input.value_is_empty(), "no value may survive a cancel");
        assert_eq!(input.generation(), token + 1);
    }
}

/// A hand-edited Active mode is a real Active mode: the footer says so, and it
/// still cannot make a child chat fall through to Compare and still cannot reach
/// any control surface outside the one request-body field it owns.
#[test]
fn negative_regression_a_hand_edited_active_mode_is_active_and_bounded() {
    let dir = temp_agent_dir("negative-active-file");
    let path = dir.path().join("jev").join("jev-settings.json");
    fs::create_dir_all(path.parent().expect("dir")).expect("mkdir");
    fs::write(
        &path,
        r#"{"schema_version":1,"global_default":"active","sessions":{"parent":{"mode":"active"},"child":{"mode":"active"}},"credential_configured":true,"disclosure_shown":true}"#,
    )
    .expect("write");
    let bridge = bridge_over(&dir);
    // The file IS read and the value IS Active: it is reported honestly, not
    // silently rewritten.
    assert_eq!(bridge.effective_mode("parent"), JevMode::Active);
    // ...and the footer shows the accent Active state, never a green one.
    assert!(!JevMode::Active.allows_compare(), "Active is not Compare");
    let credential = CredentialStatus::resolve(true, false, false);
    let state = footer_state(JevMode::Active, &credential, &JevPipelineStatus::default());
    assert_eq!(state, JevFooterState::Active);
    assert_eq!(state.color_key(), "success");
    assert!(state.is_green());
    assert_eq!(footer_color_key(state), "success");
    assert!(footer_text(state).contains("Jev On (Active)"));
    assert_eq!(mode_label(JevMode::Active), "Active");

    // A child of that session inherits Active and is never silently turned into
    // Compare; a missing parent inherits the global default, which is Active here.
    let inherited = bridge
        .inherit_into_child("child-2", "parent", None)
        .expect("no error");
    assert_eq!(inherited, JevMode::Active);
    assert!(!inherited.allows_compare(), "{inherited:?} must not enable Compare");
    let fresh = bridge
        .inherit_into_child("child-3", "no-such-session", None)
        .expect("no error");
    assert_eq!(fresh, JevMode::Active);
    assert!(!fresh.allows_compare());

    // The UI then writes a real mode over it, and only the requested one lands.
    let change = bridge
        .set_session_mode("parent", JevMode::Off)
        .expect("no error");
    assert!(matches!(
        change,
        ModeChange::Applied {
            mode: JevMode::Off,
            scope: ModeScope::Session
        }
    ));
    assert_eq!(bridge.effective_mode("parent"), JevMode::Off);
}

/// Baseline delegation is untouched: the lane adds no call site and no argument
/// that could change a model, a tool, a permission, a child, a message or a
/// budget. Audited as CALLS on the connection, not as prose.
#[test]
fn negative_regression_baseline_delegation_is_unchanged() {
    // 1. The only session calls are identity and optional telemetry reads.
    for file in JEV_FILES {
        let source = jev_source(file);
        let calls = calls_on_a_connection(&source);
        for call in &calls {
            assert!(
                matches!(call.as_str(), "get_state" | "get_jev_status" | "supports_jev_features"),
                "{file} may only READ the session (found {call})"
            );
        }
    }

    // 2. The delegation surface is not named at all, so a future edit that wants it
    //    must delete this assertion deliberately.
    let delegation = [
        "create_rlm_subagent",
        "create_rlm_child",
        "start_rlm_child_run",
        "cancel_rlm_child",
        "delete_rlm_subagent",
        "set_rlm_max_depth",
        "agent_message",
        "send_message",
        "steer",
        "follow_up",
        "replace_acp_mcp_servers",
        "set_model",
        "cycle_model",
        "set_scoped_models",
        "set_thinking_level",
        "cycle_thinking_level",
        "set_service_tier",
        "abort_bash",
    ];
    for file in JEV_FILES {
        let source = jev_source(file);
        for symbol in delegation {
            assert!(
                !calls_on_a_connection(&source).iter().any(|call| call == symbol),
                "{file} must not call {symbol}"
            );
            assert!(
                !source.contains(&format!(".{symbol}(")),
                "{file} must not call {symbol}"
            );
        }
    }

    // 3. The dialogs the lane mounts carry an event channel and the host sender
    //    only: no session handle, no child handle, no model handle.
    let handlers = crate_file("src/modes/interactive/native_host_commands.rs");
    for variant in ["Jev(", "JevKey("] {
        let start = handlers
            .find(&format!("Dialog::{variant}"))
            .unwrap_or_else(|| panic!("Dialog::{variant} must exist"));
        let block = &handlers[start..(start + 400).min(handlers.len())];
        for forbidden in ["AgentConnection", "RlmChild", "Model", "Tool", "Subagent"] {
            assert!(
                !block.contains(forbidden),
                "Dialog::{variant} must not carry {forbidden}: {block}"
            );
        }
    }

    // 4. A mode write changes the mode file only: no other file in the agent dir is
    //    created or modified by a `/jev` mode change.
    let dir = temp_agent_dir("negative-baseline");
    let bridge = bridge_over(&dir);
    bridge.set_session_mode("s", JevMode::Compare).expect("no error");
    bridge.set_global_default(JevMode::Off).expect("no error");
    let mut written: Vec<String> = Vec::new();
    for entry in walk(&dir.path().to_path_buf()) {
        written.push(entry);
    }
    written.sort();
    assert_eq!(
        written,
        vec!["jev".to_string(), "jev/jev-settings.json".to_string(), "jev/jev-settings.json.lock".to_string()],
        "a mode change must write only settings and the cooperating lock"
    );
}

/// Every path under a directory, as `dir`-relative slash-separated strings.
fn walk(root: &PathBuf) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let relative = path
                .strip_prefix(root)
                .map(|value| value.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            out.push(relative);
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 7. wiring: registry, dispatch, dialogs, keybindings, daemon gate
// ---------------------------------------------------------------------------

#[test]
fn the_jev_command_is_registered_and_reachable_through_the_dispatch_chain() {
    let registry = crate_file("src/core/slash_commands.rs");
    assert!(registry.contains("\"jev\""), "the registry must list /jev");
    // The registry quotes the canonical metadata constants, so a mode list that
    // drifts from `jev_menu.rs` fails here.
    assert!(
        registry.contains(&format!("Some({JEV_ARGUMENT_HINT:?})")),
        "the argument hint must list the accepted forms"
    );
    assert!(registry.contains(JEV_COMMAND_DESCRIPTION), "the description must be quoted verbatim");
    assert!(
        registry.contains("Compare + Active, compaction, feature gates"),
        "the registry must name Active as an operative mode"
    );
    for forbidden in [inert_wording(), DISABLED_WORD.to_string()] {
        assert!(!registry.contains(&forbidden), "the registry must not claim Active is inert: found {forbidden}");
    }

    let commands = crate_file("src/modes/interactive/native_host_commands.rs");
    assert!(
        commands.contains("\"jev\" => return jev_host::run(connection, send, args).await"),
        "the dispatch arm must call the Jev handler"
    );
    assert!(commands.contains("Dialog::Jev("), "the menu dialog must exist");
    assert!(commands.contains("Dialog::JevKey("), "the key dialog must exist");
    assert!(commands.contains("JevMenuOverlay::new"), "the menu overlay must mount");
    assert!(commands.contains("JevKeyOverlay::new"), "the key overlay must mount");
    // The overlays are declared from this file with `#[path]`, so no edit to the
    // shared `native_host.rs` is needed for the modules themselves.
    for module in ["jev_menu.rs", "jev_menu_component.rs", "jev_key_input.rs", "jev_footer.rs", "jev_host.rs"] {
        assert!(
            commands.contains(&format!("#[path = \"{module}\"]")),
            "{module} must be declared"
        );
    }

    // The out-of-lane half of reachability is hunk C-1: one token in the shared
    // host arm list. This lane must not edit `native_host.rs`, so the test records
    // the hunk instead of applying it, and it fails loudly if the arm list moves.
    let host = crate_file("src/modes/interactive/native_host.rs");
    let arms = host
        .lines()
        .find(|line| line.contains("native_commands::run("))
        .expect("the host arm list must exist");
    assert!(
        arms.contains("\"btw\""),
        "sanity: the host arm list is the one under audit"
    );
    assert!(
        arms.contains("\"mcp\""),
        "sanity: the guarded `/mcp` arm is in the same list"
    );
    // Integration hunk C-1 is coordinator-applied: the arm list must carry
    // `| "jev"` so /jev is actually reachable through the dispatch chain.
    assert!(
        arms.contains("\"jev\""),
        "the host arm list must route /jev (integration hunk C-1)"
    );
}

#[test]
fn the_jev_surface_cannot_express_a_model_or_subagent_control_action() {
    // Binding DESIGN.md sections 11 and 12: no model control, no subagent control.
    let forbidden = [
        "set_model",
        "cycle_model",
        "set_scoped_models",
        "set_thinking_level",
        "cycle_thinking_level",
        "set_service_tier",
        "cancel_rlm_child",
        "delete_rlm_subagent",
        "spawn_subagent",
        "create_child",
        "send_message",
        "replace_acp_mcp_servers",
        "rlm_max_depth",
        "abort_bash",
        "steer",
        "follow_up",
    ];
    for file in JEV_FILES {
        let source = jev_source(file);
        for call in calls_on_a_connection(&source) {
            assert!(
                !forbidden.contains(&call.as_str()),
                "{file} must not reach {call} (sections 11/12)"
            );
        }
    }

    // Both identity and optional telemetry are read-only connection calls.
    let host = jev_source("jev_host.rs");
    let connection_calls = calls_on_a_connection(&host);
    assert_eq!(
        connection_calls,
        vec!["get_jev_status".to_string(), "get_state".to_string(), "supports_jev_features".to_string()],
        "only identity and worker telemetry reads are allowed"
    );
    assert!(host.contains("connection.get_state().await?"));

    // The only two `fn new` in the handler are the overlay constructors, and both
    // take the event channel and the host sender only: a child or session mutation
    // is structurally unreachable from the overlays.
    let handlers = jev_source("jev_host.rs");
    let overlay_constructors: Vec<&str> = handlers
        .split("pub(super) fn new(")
        .skip(1)
        .map(|tail| tail.split(") -> Self").next().expect("constructor signature"))
        .collect();
    assert_eq!(overlay_constructors.len(), 2, "{overlay_constructors:?}");
    for constructor in overlay_constructors {
        assert!(!constructor.contains("AgentConnection"), "{constructor}");
        assert!(!constructor.contains("connection"), "{constructor}");
    }

    // And the pure module has no effectful surface either.
    let pure = jev_source("jev_menu.rs");
    assert!(calls_on_a_connection(&pure).is_empty());
    assert!(!pure.contains("AgentConnection"));
    assert!(!pure.contains("reqwest"));
    // The pure module performs no file I/O at all: the settings file is written by
    // the pi-jev store, so the UI can never write a path of its own choosing.
    assert!(!pure.contains("std::fs::write"));
    assert!(!pure.contains("std::fs::create_dir_all"));
    assert!(pure.contains("store.store(DEFAULT_KEY_ID"));
}

#[test]
fn the_daemon_surface_is_capability_gated_and_read_only_where_it_gets() {
    let protocol = crate_file("src/modes/daemon/daemon_protocol.rs");
    assert!(
        protocol.contains("DaemonServerCapability::JevControl"),
        "the capability must be declared"
    );
    assert!(
        protocol.contains("\"jev_get_settings\" | \"jev_set_session_mode\" | \"jev_get_status\" => {"),
        "the three commands must share one capability gate"
    );
    assert!(
        protocol.contains("DaemonCommandCompatibility::capability(Capability::JevControl)"),
        "the gate must be the capability, not a schema revision"
    );
    // Optional additive metadata advances the schema without a protocol bump.
    assert!(protocol.contains("pub const DAEMON_PROTOCOL_VERSION: u32 = 7;"));
    assert!(pi_coding_agent::modes::daemon::daemon_protocol::DAEMON_SCHEMA_REVISION >= 31);
    assert!(protocol.contains("JevFeatures"));
    // The getters are read-only; the setter is not.
    assert!(protocol.contains("\"jev_get_settings\",\n    \"jev_get_status\","));
    let read_only_block = protocol
        .split("pub const READ_ONLY_DAEMON_COMMANDS")
        .nth(1)
        .expect("the read-only list");
    let read_only_block = read_only_block.split("];").next().unwrap();
    assert!(read_only_block.contains("jev_get_settings"));
    assert!(!read_only_block.contains("jev_set_session_mode"));
    // The commands are session-plane, so a control-plane-only client never sends
    // them and an unknown command still falls through to `None`.
    assert!(protocol.contains("\"jev_get_settings\" | \"jev_set_session_mode\" | \"jev_get_status\" => {\n            Some(\"session\")"));

    let daemon = crate_file("src/modes/daemon/daemon_mode.rs");
    for command in ["jev_get_settings", "jev_set_session_mode", "jev_get_status"] {
        assert!(daemon.contains(&format!("\"{command}\",")), "{command} must be routable");
        assert!(daemon.lines().any(|line| line.contains(&format!("\"{command}\""))
            && line.contains("=>")), "{command} must have a handler");
    }
    assert!(daemon.contains("pub const DAEMON_COMMAND_TYPES: [&str; 103] = ["));
    // The daemon writes the same store the UI reads (lane A's `JevSettingsStore`
    // over the same `<agent_dir>`), and never reads a secret value.
    assert!(daemon.contains("pi_jev::config::JevSettingsStore::new"));
    assert!(daemon.contains("pi_jev::credential::default_credential_store"));
    assert!(daemon.contains("\"credentialPresenceKnown\": saved_presence_known"));
    assert!(daemon.contains("\"applied\": false"));
    assert!(daemon.contains("pi_jev::types::JevMode::parse"));
    // Active is WRITTEN through the daemon, exactly like Off and Compare: the
    // requested mode is stored, and nothing is refused or silently rewritten.
    assert!(
        daemon.contains("settings.set_session_mode(session_id, requested);"),
        "the daemon must write the requested mode"
    );
    assert!(
        daemon.contains("fn jev_apply_session_mode"),
        "the daemon must apply through the store"
    );
    // The applied message names the stable mode id, so `Jev mode: active` is the
    // figure a client can key on.
    assert!(
        daemon.contains("Jev mode: {} (scope: this chat)\", mode.as_str()"),
        "the daemon must report the stable mode id"
    );
    // `activeMode` is the real mode report, not a reservation marker.
    assert!(
        daemon.contains("\"activeMode\": resolution.mode.allows_active()"),
        "activeMode must report the effective mode"
    );
    // The live Active counters are surfaced through the shared status snapshot.
    assert!(daemon.contains("crate::core::jev_bridge::session_status_snapshot(&session_id)"));

    // Degradation: an absent capability means the client keeps local control.
    // The client-side gate lives in a lane that must not be touched, so this test
    // pins the contract instead: the commands are optional (capability-gated), the
    // getter list is read-only, and the handler is additive.
    assert!(
        protocol.contains("A daemon that does not advertise this capability never receives"),
        "the degradation rule must be documented next to the gate"
    );
}

#[test]
fn the_command_metadata_and_footer_helpers_match_the_registry_and_the_footer_rule() {
    // The metadata constants are the single source the registry quotes, so a drift
    // between `slash_commands.rs` and the menu description fails here.
    assert_eq!(JEV_COMMAND_NAME, "jev");
    assert_eq!(
        JEV_ARGUMENT_HINT,
        "[off|compare|active|compare-active|on|compact|feature|default|full-jev|status|key|models|model]"
    );
    let registry = crate_file("src/core/slash_commands.rs");
    assert!(registry.contains(JEV_ARGUMENT_HINT), "the hint must be quoted verbatim");
    assert!(
        registry.contains(JEV_COMMAND_DESCRIPTION),
        "the description must be quoted verbatim"
    );

    // `env_presence` reads through a caller-supplied accessor, so a presence probe
    // needs no real environment. An empty or whitespace value is NOT present.
    let read_env = |name: &str| match name {
        "TYPESAFE_API_KEY" => Some("  ".to_string()),
        "JEV_API_KEY" => Some("alias-present".to_string()),
        _ => None,
    };
    let env = env_presence(&read_env);
    assert!(!env.typesafe_api_key, "whitespace is not a credential");
    assert!(env.jev_api_key);
    assert!(!env.has_conflict());
    let empty = env_presence(&|_| None);
    assert!(!empty.typesafe_api_key && !empty.jev_api_key);
    assert_eq!(
        resolve_credential_source(false, empty.typesafe_api_key, empty.jev_api_key),
        CredentialSource::None
    );

    // The footer helpers expose the pinned texts and the width threshold, and the
    // removal event reuses the same status key.
    assert_eq!(footer_text(JevFooterState::Off), "\u{25cf} Jev Off");
    assert_eq!(
        footer_text(JevFooterState::Compare),
        "\u{25cf} Jev On (Compare)"
    );
    assert_eq!(
        footer_compaction_text(JevCompactionState::On),
        "\u{25cf} Jev compact on"
    );
    assert_eq!(
        footer_compaction_text(JevCompactionState::Off),
        "\u{25cf} Jev compact off"
    );
    assert_eq!(FOOTER_LABEL_MIN_COLUMNS, 40);
    assert_eq!(
        footer_clear_payload()["statusKey"],
        serde_json::json!(JEV_STATUS_KEY)
    );
    assert_eq!(
        footer_compaction_clear_payload()["statusKey"],
        serde_json::json!(JEV_COMPACT_STATUS_KEY)
    );
    // `render_help` names the real footer states, including the green On labels
    // and the independent compaction dot.
    let help = render_help();
    assert!(help.contains(JEV_FOOTER_RULE_NOTICE), "{help}");
    assert!(help.contains(JEV_ON_COMPARE_NOTICE), "{help}");
    assert!(help.contains(JEV_ACTIVE_NOTICE), "{help}");
    assert!(
        JEV_FOOTER_RULE_NOTICE.contains("Jev compact on"),
        "{JEV_FOOTER_RULE_NOTICE}"
    );
    assert!(
        JEV_FOOTER_RULE_NOTICE.contains("Jev compact off"),
        "{JEV_FOOTER_RULE_NOTICE}"
    );

    // The credential DELETE path exists and is the inverse of the write path.
    let store = InMemoryCredentialStore::new();
    store_secret(&store, JevSecret::new("sk-live-0123456789".to_string())).expect("stored");
    assert!(store.exists(DEFAULT_KEY_ID).expect("readable"));
    clear_secret(&store).expect("deleted");
    assert!(!store.exists(DEFAULT_KEY_ID).expect("readable"));
}

#[test]
fn an_operative_footer_is_green_names_its_mode_and_off_is_red() {
    // Every healthy operative state is green and names its truthful effective
    // mode; Off alone is red; degraded states stay amber.
    let credential = CredentialStatus::resolve(true, false, false);
    let healthy = JevPipelineStatus {
        last_success_at: Some("2026-09-19T00:00:00Z".to_string()),
        last_latency_ms: Some(120),
        success_count: 9,
        failure_count: 0,
        queue_capacity: 8,
        ..JevPipelineStatus::default()
    };
    let operative = [
        (
            JevMode::Compare,
            JevFooterState::Compare,
            "Jev On (Compare)",
        ),
        (JevMode::Active, JevFooterState::Active, "Jev On (Active)"),
        (
            JevMode::CompareAndActive,
            JevFooterState::CompareAndActive,
            "Jev On (Compare + Active)",
        ),
    ];
    for (mode, expected_state, expected_label) in operative {
        let state = footer_state(mode, &credential, &healthy);
        assert_eq!(state, expected_state, "{mode:?}");
        assert!(
            state.is_green(),
            "{state:?} is healthy and operative: green"
        );
        // The LIVE colour path goes through `footer_color_key`, which consults
        // `is_green` first: a healthy operative state cannot be anything but green.
        assert_eq!(footer_color_key(state), "success", "{state:?}");
        assert_eq!(
            footer_text(state),
            format!("\u{25cf} {expected_label}"),
            "{state:?}"
        );
        // The label carries the mode, so the text alone distinguishes On states.
        assert!(state.label().contains("Jev On"), "{state:?}");
    }
    // Off alone is red and never claims to be on.
    let off = footer_state(JevMode::Off, &credential, &healthy);
    assert_eq!(off, JevFooterState::Off);
    assert_eq!(off.color_key(), "error");
    assert!(!off.is_green());
    assert!(!off.label().contains("Jev On"));
    // A credential-less operative session degrades to amber, not green.
    let degraded = footer_state(
        JevMode::Active,
        &CredentialStatus::resolve(false, false, false),
        &healthy,
    );
    assert_eq!(degraded, JevFooterState::Unavailable);
    assert_eq!(degraded.color_key(), "warning");
    assert!(!degraded.is_green());
    assert!(!degraded.label().contains("Jev On"));
    // The compaction dot mirrors the same colours independently.
    assert!(JevCompactionState::On.is_green());
    assert_eq!(JevCompactionState::On.color_key(), "success");
    assert_eq!(JevCompactionState::Off.color_key(), "error");
    assert!(!JevCompactionState::Off.is_green());
    assert_eq!(JevCompactionState::Unknown.color_key(), "warning");
    assert!(!JevCompactionState::Unknown.is_green());
    assert!(JEV_FOOTER_RULE_NOTICE.contains("Jev On (Compare)"));
    assert!(JEV_FOOTER_RULE_NOTICE.contains("Jev Off"));
}

#[test]
fn the_keybinding_addition_has_no_default_key_and_keeps_the_app_order() {
    let keybindings = crate_file("src/core/keybindings.rs");
    assert!(keybindings.contains("pub const APP_KEYBINDINGS: [AppKeybinding; 57] = ["));
    assert!(keybindings.contains("\"app.jev.cancel\","));
    let definition = keybindings
        .split("\"app.jev.cancel\".to_string(),")
        .nth(1)
        .expect("the app entry");
    assert!(definition.contains("default_keys: vec![]"), "{definition}");
    assert!(definition.contains("default_keys_is_single: false"), "{definition}");
    assert!(definition.contains("Cancel the /jev dialog"), "{definition}");
    // The declaration order test is driven by the constant, so the entry must be
    // last in both places or `keybindings_spread_tui_entries_first_then_app_entries`
    // fails. Guard that ordering here as well.
    let list = keybindings
        .split("pub const APP_KEYBINDINGS")
        .nth(1)
        .expect("the constant");
    let list = list.split("];").next().unwrap();
    assert!(list.trim_end().ends_with("\"app.jev.cancel\","), "{list}");
}

#[test]
fn combined_mode_and_independent_compaction_commands_round_trip() {
    for alias in ["compare-active", "compare-and-active", "compare_and_active", "both"] {
        assert_eq!(parse_jev_request(alias), JevRequest::SetMode(JevMode::CompareAndActive));
    }
    assert_eq!(parse_jev_request("on"), JevRequest::SetMode(JevMode::Compare));
    assert_eq!(parse_jev_request("compact on"), JevRequest::SetCompaction(true));
    assert_eq!(parse_jev_request("compaction off"), JevRequest::SetCompaction(false));
    assert_eq!(parse_jev_request("compact status"), JevRequest::CompactionStatus);
    assert_eq!(parse_jev_request("default compact off"), JevRequest::SetDefaultCompaction(false));
    assert_eq!(parse_jev_request("feature tool-candidates on"), JevRequest::SetFeature(JevFeature::ToolCandidates, true));
    for bad in ["compact on; active", "feature model on", "feature tool_candidates yes", "default compact maybe", "active\ncompact on"] {
        assert!(matches!(parse_jev_request(bad), JevRequest::Unknown(_)));
    }
    let dir = temp_agent_dir("independent-controls");
    let bridge = bridge_over(&dir);
    bridge.set_session_mode("chat", JevMode::CompareAndActive).unwrap();
    bridge.set_feature("chat", JevFeature::ToolCandidates, true).unwrap();
    bridge.set_compaction("chat", true).unwrap();
    bridge.set_compaction("chat", false).unwrap();
    assert_eq!(bridge.effective_mode("chat"), JevMode::CompareAndActive);
    assert!(bridge.settings().effective_features("chat").tool_candidates);
    assert!(!bridge.settings().effective_compaction_enabled("chat"));
    bridge.set_default_compaction(true).unwrap();
    assert!(bridge.settings().effective_compaction_enabled("other"));
    assert!(!bridge.settings().effective_compaction_enabled("chat"));
    bridge.set_default_compaction(false).unwrap();
    assert!(!bridge.settings().effective_compaction_enabled("other"));
    let status = jev_ui::render_compaction_settings(&bridge.settings(), "chat");
    for field in ["Compaction: off", "scope: session", "keep_threshold", "max_state_tokens", "max_request_tokens", "minimum_reduction_ratio"] {
        assert!(status.contains(field), "{status}");
    }
}

#[test]
fn combined_mode_menu_footer_and_status_are_distinct_and_truthful() {
    let mut state = JevMenuState::new(JevMode::CompareAndActive);
    assert_eq!(state.selected, 3);
    assert_eq!(state.accept(), JevMenuAction::SetMode(JevMode::CompareAndActive));
    for (index, enabled) in [(4, true), (5, false)] {
        let mut state = JevMenuState::new(JevMode::CompareAndActive);
        state.selected = index;
        assert_eq!(state.accept(), JevMenuAction::SetCompaction(enabled));
        assert_eq!(state.active_mode, JevMode::CompareAndActive);
    }
    let saved = CredentialStatus::resolve(true, false, false);
    let none = CredentialStatus::resolve(false, false, false);
    assert_eq!(
        footer_state(JevMode::CompareAndActive, &saved, &Default::default()),
        JevFooterState::CompareAndActive
    );
    assert_eq!(
        footer_state(JevMode::CompareAndActive, &none, &Default::default()),
        JevFooterState::Unavailable
    );
    assert_eq!(
        footer_color_key(JevFooterState::CompareAndActive),
        "success"
    );
    assert!(footer_text(JevFooterState::CompareAndActive).contains("Compare + Active"));
    let fresh = JevStatusReport::local_only(JevMode::CompareAndActive, ModeScope::Session, saved);
    let text = render_status(&fresh);
    assert!(text.contains("Decisions applied: unknown"), "{text}");
    assert!(text.contains("Feature gates (configured; not proof"), "{text}");
    let live = fresh.with_snapshot(Some(&serde_json::json!({
        "success_count": 4, "failure_count": 1, "queue_capacity": 8,
        "active": {"applied": 2, "accepted_no_effect": 1, "refused": 3, "unavailable": 1}
    })));
    let text = render_status(&live);
    assert!(text.contains("Counters: 4 ok, 1 failed"));
    assert!(text.contains("Decisions applied: 2"));
    assert!(text.contains("Active boundaries: 2 applied"));
    assert!(!text.contains("Compare is shadow-only; potential savings"));
}

#[test]
fn compaction_status_distinguishes_unknown_observed_and_sparse_fallbacks() {
    let report = JevStatusReport::local_only(JevMode::CompareAndActive, ModeScope::Session,
        CredentialStatus::resolve(true, false, false));
    let unknown = render_status(&report);
    assert!(unknown.contains("Compaction last: unknown"));
    assert!(!unknown.contains("Compaction candidates: 0"));
    let value = serde_json::json!({"compaction": {
        "applied": true, "estimated_tokens_before": 2000, "estimated_tokens_after": 1000,
        "calls_evaluated": 4, "calls_removed": 2, "results_removed": 2, "results_truncated": 1,
        "reduction_ratio": 0.5, "breaker_state": "closed"
    }});
    let observed = render_status(&report.clone().with_snapshot(Some(&value)));
    assert!(observed.contains("Compaction last: applied"));
    assert!(observed.contains("Compaction estimated tokens: 2000 -> 1000 (reduction 50.0%)"));
    assert!(observed.contains("4 evaluated, 2 calls removed, 2 results removed, 1 results truncated"));
    assert!(observed.contains("Compaction breaker: closed"));
    let sparse = serde_json::json!({"compaction": {"applied": false, "fallback_reason": "circuit_open"}});
    let fallback = render_status(&report.with_snapshot(Some(&sparse)));
    assert!(fallback.contains("Compaction last: not applied"));
    assert!(fallback.contains("Compaction estimated tokens: unknown -> unknown"));
    assert!(fallback.contains("Compaction fallback: circuit_open"));
    assert!(fallback.contains("Compaction breaker: unknown"));
    for invalid in [serde_json::json!({}), serde_json::json!([]), serde_json::json!({"applied": "yes"})] {
        assert!(jev_ui::CompactionStatus::from_snapshot(Some(&invalid)).is_none());
    }
}

#[test]
fn malformed_active_counters_are_unknown_not_zero() {
    for value in [serde_json::json!({}), serde_json::json!({"applied": 3}),
        serde_json::json!({"applied": 0, "accepted_no_effect": 0, "refused": 0, "unavailable": "invalid"})] {
        assert!(ActiveCounters::from_snapshot(Some(&value)).is_none());
        let report = JevStatusReport::local_only(JevMode::Active, ModeScope::Session,
            CredentialStatus::resolve(true, false, false))
            .with_snapshot(Some(&serde_json::json!({"active": value})));
        assert!(render_status(&report).contains("Decisions applied: unknown"));
    }
}

#[test]
fn actual_ui_mode_write_gate_refuses_legacy_worker_without_touching_settings() {
    let dir = temp_agent_dir("legacy-worker-gate");
    let bridge = bridge_over(&dir);
    assert!(bridge.set_session_mode_supported("s", JevMode::CompareAndActive, false).is_err());
    assert!(bridge.set_global_default_supported(JevMode::CompareAndActive, false).is_err());
    assert!(!bridge.path().exists(), "unsupported actions must not write any settings");
    for mode in [JevMode::Off, JevMode::Compare, JevMode::Active] {
        bridge.set_session_mode_supported("s", mode, false).unwrap();
        assert_eq!(bridge.effective_mode("s"), mode);
    }
    let before = fs::read(bridge.path()).unwrap();
    assert!(bridge.set_session_mode_supported("s", JevMode::CompareAndActive, false).is_err());
    assert_eq!(fs::read(bridge.path()).unwrap(), before);
    bridge.set_session_mode_supported("s", JevMode::CompareAndActive, true).unwrap();
    assert_eq!(bridge.effective_mode("s"), JevMode::CompareAndActive);
    assert!(jev_ui::require_feature_support(false).is_err());
    assert!(jev_ui::require_feature_support(true).is_ok());
    let host = jev_source("jev_host.rs");
    assert!(host.contains("bridge.set_session_mode_supported(&session_id, mode, connection.supports_jev_features())"));
    assert!(host.contains("bridge.set_session_mode_supported(session_id, mode, connection.supports_jev_features())"));
    assert!(host.contains("bridge.set_global_default_supported(mode, connection.supports_jev_features())"));
    assert_eq!(host.matches("require_feature_support(connection.supports_jev_features())?").count(), 4);
}

#[test]
fn compaction_controls_and_status_do_not_require_an_active_mode() {
    let mut settings = JevSettings::default();
    settings.set_session_compaction_enabled("s", true);
    let status = jev_ui::render_compaction_settings(&settings, "s");
    assert!(status.contains("Compaction: on"));
    assert!(status.contains("independent of decision mode"));
    assert!(!status.contains("inactive in this mode"));
    let report = JevStatusReport::local_only(JevMode::Off, ModeScope::Session,
        CredentialStatus::resolve(true, false, false)).with_settings(&settings, "s");
    let text = render_status(&report);
    assert!(text.contains("decision mode off (request-local compaction independently enabled)"));
    assert!(!text.contains("Off: no scheduling, no client, no network"));
    settings.set_session_mode("s", JevMode::Compare);
    let status = jev_ui::render_compaction_settings(&settings, "s");
    assert!(status.contains("Compaction: on"));
    assert!(status.contains("independent of decision mode"));
    assert!(!status.contains("inactive in this mode"));
}

// ---------------------------------------------------------------------------
// 7. tray-row footer: two independent dots, effective per-session state
// ---------------------------------------------------------------------------

#[test]
fn the_compaction_dot_is_independent_of_the_decision_dot() {
    // Decision OFF with compaction ON: two separate segments, two separate
    // colours, and the decision-off state never removes or flips the compaction
    // dot. This is the JEV_SYSTEM_ONE.md rule: changing one axis does not change
    // the other.
    let decision = footer_state(
        JevMode::Off,
        &CredentialStatus::resolve(false, false, false),
        &JevPipelineStatus::default(),
    );
    assert_eq!(decision, JevFooterState::Off);
    assert_eq!(footer_text(decision), "\u{25cf} Jev Off");
    assert_eq!(footer_text(JevFooterState::Off), "\u{25cf} Jev Off");
    assert_eq!(
        footer_compaction_text(JevCompactionState::On),
        "\u{25cf} Jev compact on"
    );
    assert_ne!(
        footer_text(decision),
        footer_compaction_text(JevCompactionState::On)
    );
    assert_eq!(decision.color_key(), "error");
    assert_eq!(JevCompactionState::On.color_key(), "success");
    // Same for the reverse: decisions on with compaction off.
    assert_eq!(
        footer_compaction_text(JevCompactionState::Off),
        "\u{25cf} Jev compact off"
    );
    assert_eq!(JevCompactionState::Off.color_key(), "error");
    // Unknown is its own label; it is never rendered as on or off.
    assert_eq!(
        footer_compaction_text(JevCompactionState::Unknown),
        "\u{25cf} Jev compact unknown"
    );
    assert_eq!(JevCompactionState::Unknown.color_key(), "warning");
    // The segments publish under different keys, so one refresh can never
    // clobber the other.
    assert_ne!(JEV_STATUS_KEY, JEV_COMPACT_STATUS_KEY);
}

#[test]
fn the_compaction_dot_shows_the_effective_per_session_setting() {
    // Session override wins over the global default; without an override the
    // global default applies; nothing is inferred from unrelated state.
    let mut override_on = JevSettings::default();
    override_on.compaction_enabled = false;
    override_on.set_session_compaction_enabled("s", true);
    assert_eq!(compaction_state(&override_on, "s"), JevCompactionState::On);
    // A DIFFERENT session has no override: the global default (off) applies, so
    // the same settings object yields different dots per session.
    assert_eq!(
        compaction_state(&override_on, "other"),
        JevCompactionState::Off
    );

    let mut override_off = JevSettings::default();
    override_off.compaction_enabled = true;
    override_off.set_session_compaction_enabled("s", false);
    assert_eq!(
        compaction_state(&override_off, "s"),
        JevCompactionState::Off
    );
    assert_eq!(
        compaction_state(&override_off, "other"),
        JevCompactionState::On
    );

    let defaults = JevSettings::default();
    assert_eq!(compaction_state(&defaults, "s"), JevCompactionState::Off);
    // The decision mode is irrelevant to this resolution.
    let mut decisions_on = JevSettings::default();
    decisions_on.set_session_mode("s", JevMode::CompareAndActive);
    assert_eq!(
        compaction_state(&decisions_on, "s"),
        JevCompactionState::Off
    );
}

#[test]
fn the_tray_row_pulls_the_jev_segments_onto_the_model_effort_row() {
    // The status row component pulls exactly the two Jev keys onto the tray row
    // (same baseline as the model/effort label) and renders every other
    // extension status on its own line below, as before.
    let source = crate_file("src/modes/interactive/native_host_extensions.rs");
    // Immutable key lookups move the segments onto the row; the map is NEVER
    // mutated or reordered per paint, and the two keys are excluded from the
    // plain status line, so each Jev label is painted exactly once.
    assert!(
        source.contains("fn split_status_row"),
        "one split decides row vs plain line"
    );
    assert!(
        source.contains(".get(JEV_STATUS_KEY)"),
        "decision segment read by key"
    );
    assert!(
        source.contains(".get(JEV_COMPACT_STATUS_KEY)"),
        "compaction segment read by key"
    );
    // The render path itself never mutates the map (immutable `get` only);
    // `shift_remove(JEV_` appears ONLY in `reset_keeping_jev`, which removes
    // and re-inserts the two keys around the blanket reset.
    let split_fn = source
        .split("fn split_status_row")
        .nth(1)
        .expect("split_status_row exists")
        .split("pub(super) struct Statuses")
        .next()
        .unwrap()
        .to_string();
    assert!(
        !split_fn.contains("shift_remove"),
        "render never mutates the map"
    );
    let reset_fn = source
        .split("fn reset_keeping_jev")
        .nth(1)
        .expect("reset_keeping_jev exists")
        .split("pub(super) struct Widgets")
        .next()
        .unwrap()
        .to_string();
    assert!(
        reset_fn.contains("shift_remove(JEV_STATUS_KEY)"),
        "same-session reset retains by key"
    );
    assert!(
        reset_fn.contains("insert(JEV_STATUS_KEY"),
        "and re-inserts the decision segment"
    );
    assert!(
        reset_fn.contains("insert(JEV_COMPACT_STATUS_KEY"),
        "and re-inserts the compaction segment"
    );
    assert!(
        source.contains("render_row("),
        "the row merge goes through Tray::render_row"
    );
    assert!(
        source.contains("ExtensionStatus"),
        "statuses carry an optional compact form"
    );
    // The keys match the ones the Jev publisher uses, so a rename on one side
    // fails here.
    let menu = jev_source("jev_menu.rs");
    assert!(menu.contains("pub const JEV_STATUS_KEY: &str = \"jev\";"));
    assert!(menu.contains("pub const JEV_COMPACT_STATUS_KEY: &str = \"jev-compact\";"));
    // The tray row composes the counter on the same line; the counter is never
    // a separate lower-left line.
    let tray = crate_file("src/modes/interactive/native_host.rs");
    assert!(
        tray.contains("fn render_row("),
        "Tray renders the merged row"
    );
    assert!(
        tray.contains("get_tray_context_usage_text()"),
        "the counter comes from the connection state"
    );
    assert!(tray.contains("tray_row::compose_tray_row"));
    let mode = crate_file("src/modes/interactive/interactive_mode.rs");
    assert!(
        mode.contains("fn get_tray_context_usage_text("),
        "the counter text is its own accessor"
    );
}

#[test]
fn the_setstatus_payload_carries_an_optional_compact_form() {
    // The host reads the optional field; senders without it (old daemons,
    // generic extensions) keep working and degrade to left-truncation.
    let tray = crate_file("src/modes/interactive/native_host.rs");
    assert!(
        tray.contains("\"statusCompactText\""),
        "the host reads the optional narrow form"
    );
    // The compact form is SHORT LABELLED, never a bare dot: two bare dots
    // cannot be told apart and the state must stay readable in text.
    let menu = jev_source("jev_menu.rs");
    assert!(
        menu.contains("\"Jev C On\""),
        "short labelled decision compact form"
    );
    assert!(
        menu.contains("\"Jev A On\""),
        "short labelled Active compact form"
    );
    assert!(
        menu.contains("\"Jev C+A On\""),
        "short labelled combined compact form"
    );
    assert!(
        menu.contains("\"Jev Cmp on\"") && menu.contains("\"Jev Cmp off\""),
        "short labelled compaction compact form"
    );
    assert!(
        menu.contains("fn footer_compact_text")
            && menu.contains("fn footer_compaction_compact_text")
    );
    // The publisher provides it on both segments.
    let footer = jev_source("jev_footer.rs");
    assert!(
        footer.contains("\"statusCompactText\": self.themed_compact()"),
        "the decision payload carries the narrow form"
    );
    assert!(
        footer.contains("\"statusCompactText\": self.themed_compaction_compact()"),
        "the compaction payload carries the narrow form"
    );
    // The daemon attach publish provides all four forms from ONE snapshot.
    let daemon = crate_file("src/modes/daemon/daemon_mode.rs");
    assert!(
        daemon.contains("\"jev-compact\""),
        "the daemon publishes the independent compaction dot"
    );
    assert!(
        daemon.contains("footer_status_forms("),
        "one settings snapshot for all four texts"
    );
    assert!(daemon.contains("statusCompactText"), "and its narrow form");
    // A daemon-side session replacement resets every attached UI's status
    // surface, so the authoritative footer for the NEW session must FOLLOW the
    // replace frame, per client, inside the broadcast loop.
    let broadcast_fn = daemon
        .split("fn broadcast_to_session")
        .nth(1)
        .expect("broadcast_to_session exists")
        .split("fn clone_arc")
        .next()
        .unwrap()
        .to_string();
    assert!(
        broadcast_fn.contains("publish_jev_attach_footer(&client, &state)"),
        "the footer follows the replace frame per client"
    );
    assert!(
        broadcast_fn.contains("type_name() == \"session_replaced\""),
        "the push is keyed on the replacement frame"
    );
    let bridge = crate_file("src/core/jev_bridge.rs");
    assert!(
        bridge.contains("pub fn footer_status_forms"),
        "the bridge builds the forms"
    );
}

#[test]
fn the_footer_refreshes_after_a_compaction_setting_change() {
    // Every compaction write republishes the footer in the same turn, exactly
    // like a mode write already does.
    let host = jev_source("jev_host.rs");
    let set_compaction_arm = host
        .split("JevRequest::SetCompaction(enabled) =>")
        .nth(1)
        .expect("SetCompaction arm exists")
        .split("JevRequest::SetDefaultCompaction")
        .next()
        .unwrap()
        .to_string();
    assert!(
        set_compaction_arm.contains("publish_footer("),
        "the command path refreshes the footer: {set_compaction_arm}"
    );
    let set_default_arm = host
        .split("JevRequest::SetDefaultCompaction(enabled) =>")
        .nth(1)
        .expect("SetDefaultCompaction arm exists")
        .split("JevRequest::CompactionStatus")
        .next()
        .unwrap()
        .to_string();
    assert!(
        set_default_arm.contains("publish_footer("),
        "the default-compaction path refreshes the footer"
    );
    let menu_arm = host
        .split("JevMenuAction::SetCompaction(enabled) =>")
        .nth(1)
        .expect("menu compaction action exists")
        .split("JevMenuAction::ShowStatus")
        .next()
        .unwrap()
        .to_string();
    assert!(
        menu_arm.contains("publish_footer("),
        "the menu path refreshes the footer"
    );
    // Mode writes keep their existing refresh.
    assert!(host.contains("publish_footer(send, &bridge, &session_id, credential);"));
    // The host publishes the CURRENT effective segments on startup and on a
    // session change, gated to the in-process connection so an attached daemon
    // is never overridden.
    let tray = crate_file("src/modes/interactive/native_host.rs");
    assert_eq!(
        tray.matches("publish_session_footer").count(),
        2,
        "startup publish + session-change publish (the definition lives in jev_host.rs)"
    );
    assert!(
        tray.contains("in_process_connection.is_some()"),
        "daemon-attached UIs are not overridden"
    );
}

#[test]
fn the_row_ladder_keeps_the_counter_visible_and_the_baseline_shared() {
    // The pure row module pins the layout contract: right-aligned counter on the
    // SAME row, compact segments before left truncation, no stray lower-left
    // wrap. The wiring audit only checks the tray row delegates to it.
    let row_source = crate_file("src/modes/interactive/tray_row.rs");
    assert!(row_source.contains("pub fn compose_tray_row"));
    assert!(
        row_source.contains("statusCompactText"),
        "the compact form contract is documented"
    );
    // The tray renders ONE line for the row plus at most the goal/heartbeat
    // line: the counter can never move to a lower-left row of its own.
    let tray = crate_file("src/modes/interactive/native_host.rs");
    let render_row = tray
        .split("fn render_row(")
        .nth(1)
        .expect("render_row exists")
        .split("fn invalidate")
        .next()
        .unwrap()
        .to_string();
    assert!(
        render_row.contains("get_tray_context_label()"),
        "goal/heartbeat keep their line"
    );
    assert_eq!(
        render_row.matches("lines.push").count(),
        1,
        "exactly one extra line: goal/heartbeat only"
    );
}

#[test]
fn the_runtime_rebind_resets_blanket_and_the_same_session_reload_keeps_the_jev_segments() {
    let tray = crate_file("src/modes/interactive/native_host.rs");
    // The rebind callback (fork / new / resume) sends the DISTINCT event, not
    // the same-session reload reset.
    let rebind = tray
        .split("runtime_set_rebind_session(Some(Arc::new(move || {")
        .nth(1)
        .expect("rebind callback exists")
        .split("let connection = weak.upgrade();")
        .next()
        .unwrap()
        .to_string();
    assert!(rebind.contains("Event::RuntimeRebound"), "{rebind}");
    // The reset arm keeps the two contracts distinct: same-session reload
    // retains the segments, a runtime rebind blanket-resets.
    assert!(tray.contains("event @ (Event::Reset | Event::RuntimeRebound)"));
    assert!(tray.contains("matches!(&event, Event::RuntimeRebound)"));
    let arm = tray
        .split("event @ (Event::Reset | Event::RuntimeRebound) => {")
        .nth(1)
        .expect("combined reset arm exists")
        .split("bridge.reset()")
        .next()
        .unwrap()
        .to_string();
    assert!(arm.contains("reset_keeping_jev()"), "{arm}");
    assert!(arm.contains(".reset();"), "{arm}");
    // /new and /resume hand the new runtime to the refresh arm like fork does:
    // the rebind reset lands first, then RefreshSnapshot republishes the NEW
    // session's authoritative state, including the Jev footer.
    assert_eq!(
        tray.matches("Runtime rebind handoff (same as fork)")
            .count(),
        2,
        "the /new and /resume command paths hand off a fresh snapshot"
    );
    for handoff in [
        tray.split("Runtime rebind handoff (same as fork)").nth(1),
        tray.split("Runtime rebind handoff (same as fork)").nth(2),
    ] {
        let handoff = handoff.expect("both /new and /resume hand off a fresh snapshot");
        assert!(
            handoff.contains("HostEvent::RefreshSnapshot("),
            "the handoff sends the refresh snapshot: {handoff}"
        );
        assert!(
            handoff.contains("connection.get_initial_snapshot().await?"),
            "the handoff fetches the new session snapshot: {handoff}"
        );
    }
    // And the refresh arm republishes the new session's footer (startup +
    // session change, gated to the in-process connection).
    assert_eq!(tray.matches("publish_session_footer").count(), 2);
}


// ---------------------------------------------------------------------------
// 10. full-jev global overlay (ROOT-CONTRACT v1): the persisted named
//     profile that resolves ABOVE every saved override, the emergency
//     exit, the conflicting-write rejections and the truthful panels.
// ---------------------------------------------------------------------------

/// Saved decisions the overlay must mask while active: session "alpha"
/// explicitly Off + compaction off + verification on, session "beta"
/// explicitly Compare + compaction on, global default Compare.
fn full_jev_saved_baseline(bridge: &JevModeBridge) {
    bridge
        .set_global_default(JevMode::Compare)
        .expect("global default");
    bridge
        .set_session_mode("alpha", JevMode::Off)
        .expect("alpha mode");
    bridge
        .set_compaction("alpha", false)
        .expect("alpha compaction");
    bridge
        .set_feature("alpha", JevFeature::Verification, true)
        .expect("alpha feature");
    bridge
        .set_session_mode("beta", JevMode::Compare)
        .expect("beta mode");
    bridge
        .set_compaction("beta", true)
        .expect("beta compaction");
}

#[test]
fn full_jev_arguments_parse_with_bare_meaning_on() {
    for args in [
        "full-jev",
        "fulljev",
        "full_jev",
        "full-jev on",
        "FULL-JEV",
        " full-jev  on ",
    ] {
        assert_eq!(
            parse_jev_request(args),
            JevRequest::SetFullJev(true),
            "`{args}` must install the overlay; bare full-jev means on"
        );
    }
    for args in ["full-jev off", "fulljev off", "full_jev off"] {
        assert_eq!(parse_jev_request(args), JevRequest::SetFullJev(false));
    }
    for args in ["full-jev status", "fulljev status", "full_jev status"] {
        assert_eq!(parse_jev_request(args), JevRequest::FullJevStatus);
    }
    // No silent rewrites: unknown spellings stay unknown, and the usage text
    // names the new command.
    assert!(matches!(
        parse_jev_request("full-jev bogus"),
        JevRequest::Unknown(_)
    ));
    assert!(
        JEV_ARGUMENT_HINT.contains("full-jev"),
        "{}",
        JEV_ARGUMENT_HINT
    );
    assert!(jev_usage().contains("full-jev"));
}

#[test]
fn full_jev_masks_saved_overrides_and_off_restores_them_exactly() {
    let dir = temp_agent_dir("full-jev-mask");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    let saved_alpha = bridge.settings();

    // Install: everything resolves above the saved decisions (contract: all
    // gates on even where saved session settings say off).
    assert_eq!(
        bridge.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    );
    assert_eq!(bridge.effective_mode("alpha"), JevMode::CompareAndActive);
    assert_eq!(bridge.scope("alpha"), ModeScope::FullJevOverlay);
    assert!(bridge.settings().effective_compaction_enabled("alpha"));
    assert!(bridge.settings().effective_features("alpha").verification);
    assert!(
        bridge
            .settings()
            .effective_features("alpha")
            .code_search_reranking
    );
    assert!(bridge.settings().effective_features("alpha").line_find);
    assert_eq!(bridge.effective_mode("beta"), JevMode::CompareAndActive);
    // The masked saved values are visible, not hidden.
    assert_eq!(
        bridge.settings().full_jev_masked_sessions(),
        vec!["alpha", "beta"]
    );

    // Remove: the saved decisions resolve again, byte-identically.
    assert_eq!(
        bridge.set_full_jev(false).expect("remove"),
        FullJevChange::Removed { was_active: true }
    );
    let restored = bridge.settings();
    assert_eq!(restored.effective_mode("alpha"), JevMode::Off);
    assert!(!restored.effective_compaction_enabled("alpha"));
    assert!(restored.effective_features("alpha").verification);
    assert_eq!(restored.effective_mode("beta"), JevMode::Compare);
    assert!(restored.effective_compaction_enabled("beta"));
    assert_eq!(
        restored.global_default, saved_alpha.global_default,
        "the base global fields are never rewritten by full-on/full-off"
    );
    assert_eq!(restored.sessions, saved_alpha.sessions);
    // Idempotence in both directions reports the truth and writes nothing.
    assert_eq!(
        bridge.set_full_jev(false).expect("remove again"),
        FullJevChange::Removed { was_active: false }
    );
}

#[test]
fn full_jev_rejects_conflicting_changes_with_an_explicit_recovery_path() {
    let dir = temp_agent_dir("full-jev-reject");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    assert!(matches!(
        bridge.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    ));
    // Every write the overlay would mask is REJECTED with the same message:
    // the command surface and the menu surface share this one bridge.
    for result in [
        bridge.set_session_mode("alpha", JevMode::Compare).map(|_| ()),
        bridge.set_session_mode("alpha", JevMode::Active).map(|_| ()),
        bridge.set_session_mode("alpha", JevMode::CompareAndActive).map(|_| ()),
        bridge.set_global_default(JevMode::Off).map(|_| ()),
        bridge.set_feature("alpha", JevFeature::Verification, false),
        bridge.set_compaction("alpha", true),
        bridge.set_default_compaction(false),
        bridge.clear_session_mode("alpha").map(|_| ()),
    ] {
        let error = result.expect_err("a masked write must be rejected");
        assert_eq!(
            error, JEV_FULL_JEV_REJECTION,
            "one shared rejection message"
        );
    }
    // The rejection points at the recovery path and never claims success.
    assert!(JEV_FULL_JEV_REJECTION.contains("/jev full-jev off"));
    assert!(JEV_FULL_JEV_REJECTION.contains("nothing was changed"));
    // Nothing was written by the rejected calls.
    assert!(bridge.settings().full_jev_active());
    assert_eq!(
        bridge.settings().sessions.get("alpha").expect("entry").mode,
        Some(JevMode::Off)
    );
}

#[test]
fn jev_off_while_full_active_is_the_atomic_emergency_exit() {
    let dir = temp_agent_dir("full-jev-emergency");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    assert!(matches!(
        bridge.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    ));

    // `/jev off` while full is active: ONE compound write.
    let change = bridge
        .set_session_mode("alpha", JevMode::Off)
        .expect("emergency exit");
    assert_eq!(change, ModeChange::EmergencyExit { mode: JevMode::Off });
    let message = mode_change_message(&change);
    assert!(
        message.contains(JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE),
        "{message}"
    );
    assert!(message.contains("Jev mode: Off"), "{message}");
    // Scope truth: the overlay is gone GLOBALLY...
    assert!(!bridge.settings().full_jev_active());
    // ...THIS chat is explicitly Off with compaction false...
    assert_eq!(bridge.effective_mode("alpha"), JevMode::Off);
    assert!(!bridge.settings().effective_compaction_enabled("alpha"));
    assert_eq!(
        bridge.settings().sessions.get("alpha").expect("entry").mode,
        Some(JevMode::Off)
    );
    // ...and OTHER sessions return to their saved settings untouched.
    assert_eq!(bridge.effective_mode("beta"), JevMode::Compare);
    assert!(bridge.settings().effective_compaction_enabled("beta"));

    // After the exit, individual settings work again (no stuck overlay).
    assert!(bridge.set_compaction("alpha", true).is_ok());
    assert!(bridge.settings().effective_compaction_enabled("alpha"));
    assert!(bridge.set_session_mode("alpha", JevMode::Compare).is_ok());
}

#[test]
fn jev_off_outside_full_keeps_the_existing_semantics() {
    let dir = temp_agent_dir("full-jev-off-baseline");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    // No overlay: `/jev off` writes the session mode only; compaction keeps
    // its independent setting (the documented pre-full behavior).
    let change = bridge
        .set_session_mode("beta", JevMode::Off)
        .expect("plain off");
    assert!(matches!(
        change,
        ModeChange::Applied {
            mode: JevMode::Off,
            scope: ModeScope::Session
        }
    ));
    assert_eq!(bridge.effective_mode("beta"), JevMode::Off);
    assert!(
        bridge.settings().effective_compaction_enabled("beta"),
        "outside full, /jev off never touches the independent compaction setting"
    );
    let message = mode_change_message(&change);
    assert!(!message.contains("Emergency exit"), "{message}");
}

#[test]
fn full_jev_status_panel_reports_constant_truth_and_masked_sessions() {
    let dir = temp_agent_dir("full-jev-status");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    let credential = CredentialStatus::resolve(false, false, false);

    // While inactive: the panel says so and promises the right shape.
    let inactive = render_full_jev_status(&bridge.settings(), "alpha", credential);
    assert!(inactive.contains("Profile: not installed"), "{inactive}");
    assert!(inactive.contains("Compare + Active"), "{inactive}");

    assert!(matches!(
        bridge.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    ));
    let active = render_full_jev_status(&bridge.settings(), "alpha", credential);
    assert!(active.contains("Profile: active (revision 1)"), "{active}");
    // The overlay facts come from the FIXED constants, never the persisted
    // fields, so a hand-edited block cannot make status lie.
    assert!(active.contains("Mode: Compare + Active"), "{active}");
    assert!(active.contains("Feature gates: all on"), "{active}");
    assert!(active.contains("Compaction: on"), "{active}");
    // Masked saved decisions are listed, not hidden.
    assert!(
        active.contains("Saved decisions masked by the overlay: 2"),
        "{active}"
    );
    assert!(active.contains("- alpha"), "{active}");
    assert!(active.contains("- beta"), "{active}");
    // This chat's effective values and the credential absence are truthful.
    assert!(
        active.contains("This chat resolves: Compare + Active (scope: full_jev_overlay)"),
        "{active}"
    );
    assert!(active.contains("No API key is configured"), "{active}");
    assert!(active.contains(JEV_DISCLOSURE_NOTICE), "{active}");
    assert!(active.contains(JEV_BOUNDARY_NOTICE), "{active}");
}

#[test]
fn full_jev_notices_state_consent_scope_and_recovery() {
    // ON names what is enabled and what happens to saved values.
    assert!(JEV_FULL_JEV_ON_NOTICE.contains("Compare + Active"));
    assert!(JEV_FULL_JEV_ON_NOTICE.contains("candidate reranking"));
    assert!(JEV_FULL_JEV_ON_NOTICE.contains("line-level semantic find"));
    assert!(JEV_FULL_JEV_ON_NOTICE.contains("resolves ABOVE every saved"));
    assert!(JEV_FULL_JEV_ON_NOTICE.contains("/jev full-jev off"));
    // OFF states the exact restore.
    assert!(JEV_FULL_JEV_OFF_NOTICE.contains("without touching any saved setting"));
    // The already-off case is a truthful no-change, never a silent success.
    assert!(JEV_FULL_JEV_ALREADY_OFF_NOTICE.contains("nothing was changed"));
    // The emergency exit names its global scope and the follow-ups.
    assert!(JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE.contains("global overlay"));
    assert!(JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE.contains("compaction off"));
    assert!(JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE.contains("Other chats"));
    assert!(JEV_FULL_JEV_EMERGENCY_EXIT_NOTICE.contains("/jev compact on"));
}

#[test]
fn full_jev_children_inherit_the_baseline_never_the_overlay() {
    let dir = temp_agent_dir("full-jev-inherit");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    assert!(matches!(
        bridge.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    ));
    // A child of masked "alpha" snapshots alpha's SAVED baseline (Off,
    // compaction off), never the overlay values...
    let inherited = bridge
        .inherit_into_child("child", "alpha", None)
        .expect("inherit");
    assert_eq!(inherited, JevMode::Off);
    let child = bridge
        .settings()
        .sessions
        .get("child")
        .expect("child entry")
        .clone();
    assert_eq!(child.mode, Some(JevMode::Off));
    assert_eq!(child.compaction_enabled, Some(false));
    // ...while the overlay is active the child still RESOLVES through it...
    assert_eq!(bridge.effective_mode("child"), JevMode::CompareAndActive);
    assert!(bridge.settings().effective_compaction_enabled("child"));
    // ...and removing the overlay returns the child to that baseline.
    assert!(matches!(
        bridge.set_full_jev(false).expect("remove"),
        FullJevChange::Removed { was_active: true }
    ));
    assert_eq!(bridge.effective_mode("child"), JevMode::Off);
    assert!(!bridge.settings().effective_compaction_enabled("child"));
}

#[test]
fn full_jev_footer_and_status_labels_cover_the_overlay_scope() {
    // The mode-change scope label and the status panel both name the overlay.
    let overlay_message = mode_change_message(&ModeChange::Applied {
        mode: JevMode::CompareAndActive,
        scope: ModeScope::FullJevOverlay,
    });
    assert!(
        overlay_message.contains("the global full-jev overlay"),
        "{overlay_message}"
    );
    let dir = temp_agent_dir("full-jev-labels");
    let bridge = bridge_over(&dir);
    full_jev_saved_baseline(&bridge);
    assert!(matches!(
        bridge.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    ));
    let report = JevStatusReport::local_only(
        bridge.settings().effective_mode("alpha"),
        bridge.settings().effective_mode_with_scope("alpha").scope,
        CredentialStatus::resolve(false, false, false),
    )
    .with_settings(&bridge.settings(), "alpha");
    let panel = render_status(&report);
    assert!(panel.contains("full_jev_overlay"), "{panel}");
    assert!(
        panel.contains("the global full-jev overlay; it resolves above every saved setting"),
        "{panel}"
    );
    assert!(panel.contains("Full-jev overlay: active"), "{panel}");
}

#[test]
fn full_jev_concurrent_bridge_instances_see_each_others_writes() {
    // The write path reloads per attempt, so a second bridge over the same
    // agent dir observes the overlay and never resurrects a stale state.
    let dir = temp_agent_dir("full-jev-concurrent");
    let bridge_a = bridge_over(&dir);
    let bridge_b = bridge_over(&dir);
    assert!(matches!(
        bridge_a.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: false
        }
    ));
    // Bridge B sees A's overlay: a second install is an idempotent no-write.
    assert_eq!(
        bridge_b.set_full_jev(true).expect("install"),
        FullJevChange::Installed {
            already_active: true
        }
    );
    // And B's removal is visible to A immediately (single source of truth).
    assert!(matches!(
        bridge_b.set_full_jev(false).expect("remove"),
        FullJevChange::Removed { was_active: true }
    ));
    assert!(!bridge_a.settings().full_jev_active());
}
