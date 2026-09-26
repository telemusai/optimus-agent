//! Port of packages/tui/src/keys.ts.

use once_cell::sync::Lazy;
use regex::Regex;
use std::sync::atomic::{AtomicBool, Ordering};

static KITTY_PROTOCOL_ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn set_kitty_protocol_active(active: bool) {
    KITTY_PROTOCOL_ACTIVE.store(active, Ordering::SeqCst);
}

pub fn is_kitty_protocol_active() -> bool {
    KITTY_PROTOCOL_ACTIVE.load(Ordering::SeqCst)
}

fn _kitty_protocol_active() -> bool {
    is_kitty_protocol_active()
}

/// Event types from Kitty keyboard protocol (flag 2)
/// 1 = key press, 2 = key repeat, 3 = key release
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEventType {
    Press,
    Repeat,
    Release,
}

/// Key identifier strings accepted by [`matches_key`] (e.g. "ctrl+c", "escape").
pub type KeyId = str;

/// Port of the `Key` constant: key-name identifiers accepted by [`matches_key`].
#[allow(non_upper_case_globals)]
pub mod key {
    // Key-name identifiers (e.g. `Key::escape`, `Key::page_up`).
    pub const escape: &str = "escape";
    pub const esc: &str = "esc";
    pub const enter: &str = "enter";
    pub const r#return: &str = "return";
    pub const tab: &str = "tab";
    pub const space: &str = "space";
    pub const backspace: &str = "backspace";
    pub const delete: &str = "delete";
    pub const insert: &str = "insert";
    pub const clear: &str = "clear";
    pub const home: &str = "home";
    pub const end: &str = "end";
    pub const page_up: &str = "pageUp";
    pub const page_down: &str = "pageDown";
    pub const up: &str = "up";
    pub const down: &str = "down";
    pub const left: &str = "left";
    pub const right: &str = "right";
    pub const f1: &str = "f1";
    pub const f2: &str = "f2";
    pub const f3: &str = "f3";
    pub const f4: &str = "f4";
    pub const f5: &str = "f5";
    pub const f6: &str = "f6";
    pub const f7: &str = "f7";
    pub const f8: &str = "f8";
    pub const f9: &str = "f9";
    pub const f10: &str = "f10";
    pub const f11: &str = "f11";
    pub const f12: &str = "f12";
    pub const backtick: &str = "`";
    pub const hyphen: &str = "-";
    pub const equals: &str = "=";
    pub const leftbracket: &str = "[";
    pub const rightbracket: &str = "]";
    pub const backslash: &str = "\\\\";
    pub const semicolon: &str = ";";
    pub const quote: &str = "'";
    pub const comma: &str = ",";
    pub const period: &str = ".";
    pub const slash: &str = "/";
    pub const exclamation: &str = "!";
    pub const at: &str = "@";
    pub const hash: &str = "#";
    pub const dollar: &str = "$";
    pub const percent: &str = "%";
    pub const caret: &str = "^";
    pub const ampersand: &str = "&";
    pub const asterisk: &str = "*";
    pub const leftparen: &str = "(";
    pub const rightparen: &str = ")";
    pub const underscore: &str = "_";
    pub const plus: &str = "+";
    pub const pipe: &str = "|";
    pub const tilde: &str = "~";
    pub const leftbrace: &str = "{";
    pub const rightbrace: &str = "}";
    pub const colon: &str = ":";
    pub const lessthan: &str = "<";
    pub const greaterthan: &str = ">";
    pub const question: &str = "?";
}

const SYMBOL_KEYS: &[&str] = &[
    "`", "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_",
    "+", "|", "~", "{", "}", ":", "<", ">", "?",
];

fn is_symbol_key(key: &str) -> bool {
    SYMBOL_KEYS.contains(&key)
}

const MODIFIER_SHIFT: i64 = 1;
const MODIFIER_ALT: i64 = 2;
const MODIFIER_CTRL: i64 = 4;
const MODIFIER_SUPER: i64 = 8;

const LOCK_MASK: i64 = 64 + 128; // Caps Lock + Num Lock

const CODEPOINT_ESCAPE: i64 = 27;
const CODEPOINT_TAB: i64 = 9;
const CODEPOINT_ENTER: i64 = 13;
const CODEPOINT_SPACE: i64 = 32;
const CODEPOINT_BACKSPACE: i64 = 127;
const CODEPOINT_KP_ENTER: i64 = 57414; // Numpad Enter (Kitty protocol)

const ARROW_UP: i64 = -1;
const ARROW_DOWN: i64 = -2;
const ARROW_RIGHT: i64 = -3;
const ARROW_LEFT: i64 = -4;

const FUNCTIONAL_DELETE: i64 = -10;
const FUNCTIONAL_INSERT: i64 = -11;
const FUNCTIONAL_PAGE_UP: i64 = -12;
const FUNCTIONAL_PAGE_DOWN: i64 = -13;
const FUNCTIONAL_HOME: i64 = -14;
const FUNCTIONAL_END: i64 = -15;

/// Kitty functional key equivalents (numpad and navigation cluster).
fn kitty_functional_key_equivalent(codepoint: i64) -> Option<i64> {
    Some(match codepoint {
        57399 => 48,                  // KP_0 -> 0
        57400 => 49,                  // KP_1 -> 1
        57401 => 50,                  // KP_2 -> 2
        57402 => 51,                  // KP_3 -> 3
        57403 => 52,                  // KP_4 -> 4
        57404 => 53,                  // KP_5 -> 5
        57405 => 54,                  // KP_6 -> 6
        57406 => 55,                  // KP_7 -> 7
        57407 => 56,                  // KP_8 -> 8
        57408 => 57,                  // KP_9 -> 9
        57409 => 46,                  // KP_DECIMAL -> .
        57410 => 47,                  // KP_DIVIDE -> /
        57411 => 42,                  // KP_MULTIPLY -> *
        57412 => 45,                  // KP_SUBTRACT -> -
        57413 => 43,                  // KP_ADD -> +
        57415 => 61,                  // KP_EQUAL -> =
        57416 => 44,                  // KP_SEPARATOR -> ,
        57417 => ARROW_LEFT,
        57418 => ARROW_RIGHT,
        57419 => ARROW_UP,
        57420 => ARROW_DOWN,
        57421 => FUNCTIONAL_PAGE_UP,
        57422 => FUNCTIONAL_PAGE_DOWN,
        57423 => FUNCTIONAL_HOME,
        57424 => FUNCTIONAL_END,
        57425 => FUNCTIONAL_INSERT,
        57426 => FUNCTIONAL_DELETE,
        _ => return None,
    })
}

fn normalize_kitty_functional_codepoint(codepoint: i64) -> i64 {
    kitty_functional_key_equivalent(codepoint).unwrap_or(codepoint)
}

fn normalize_shifted_letter_identity_codepoint(codepoint: i64, modifier: i64) -> i64 {
    let effective_modifier = modifier & !LOCK_MASK;
    if (effective_modifier & MODIFIER_SHIFT) != 0 && (65..=90).contains(&codepoint) {
        return codepoint + 32;
    }
    codepoint
}

const LEGACY_KEY_SEQUENCES_UP: &[&str] = &["\x1b[A", "\x1bOA"];
const LEGACY_KEY_SEQUENCES_DOWN: &[&str] = &["\x1b[B", "\x1bOB"];
const LEGACY_KEY_SEQUENCES_RIGHT: &[&str] = &["\x1b[C", "\x1bOC"];
const LEGACY_KEY_SEQUENCES_LEFT: &[&str] = &["\x1b[D", "\x1bOD"];
const LEGACY_KEY_SEQUENCES_HOME: &[&str] = &["\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"];
const LEGACY_KEY_SEQUENCES_END: &[&str] = &["\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"];
const LEGACY_KEY_SEQUENCES_INSERT: &[&str] = &["\x1b[2~"];
const LEGACY_KEY_SEQUENCES_DELETE: &[&str] = &["\x1b[3~"];
const LEGACY_KEY_SEQUENCES_PAGE_UP: &[&str] = &["\x1b[5~", "\x1b[[5~"];
const LEGACY_KEY_SEQUENCES_PAGE_DOWN: &[&str] = &["\x1b[6~", "\x1b[[6~"];
const LEGACY_KEY_SEQUENCES_CLEAR: &[&str] = &["\x1b[E", "\x1bOE"];
const LEGACY_KEY_SEQUENCES_F1: &[&str] = &["\x1bOP", "\x1b[11~", "\x1b[[A"];
const LEGACY_KEY_SEQUENCES_F2: &[&str] = &["\x1bOQ", "\x1b[12~", "\x1b[[B"];
const LEGACY_KEY_SEQUENCES_F3: &[&str] = &["\x1bOR", "\x1b[13~", "\x1b[[C"];
const LEGACY_KEY_SEQUENCES_F4: &[&str] = &["\x1bOS", "\x1b[14~", "\x1b[[D"];
const LEGACY_KEY_SEQUENCES_F5: &[&str] = &["\x1b[15~", "\x1b[[E"];
const LEGACY_KEY_SEQUENCES_F6: &[&str] = &["\x1b[17~"];
const LEGACY_KEY_SEQUENCES_F7: &[&str] = &["\x1b[18~"];
const LEGACY_KEY_SEQUENCES_F8: &[&str] = &["\x1b[19~"];
const LEGACY_KEY_SEQUENCES_F9: &[&str] = &["\x1b[20~"];
const LEGACY_KEY_SEQUENCES_F10: &[&str] = &["\x1b[21~"];
const LEGACY_KEY_SEQUENCES_F11: &[&str] = &["\x1b[23~"];
const LEGACY_KEY_SEQUENCES_F12: &[&str] = &["\x1b[24~"];

/// Keys that have a legacy sequence list in the TypeScript `LEGACY_KEY_SEQUENCES` object.
const LEGACY_MODIFIER_KEYS: &[&str] = &[
    "up", "down", "right", "left", "clear", "insert", "delete", "pageUp", "pageDown", "home", "end",
];

fn legacy_key_sequences(key: &str) -> Option<&'static [&'static str]> {
    Some(match key {
        "up" => LEGACY_KEY_SEQUENCES_UP,
        "down" => LEGACY_KEY_SEQUENCES_DOWN,
        "right" => LEGACY_KEY_SEQUENCES_RIGHT,
        "left" => LEGACY_KEY_SEQUENCES_LEFT,
        "home" => LEGACY_KEY_SEQUENCES_HOME,
        "end" => LEGACY_KEY_SEQUENCES_END,
        "insert" => LEGACY_KEY_SEQUENCES_INSERT,
        "delete" => LEGACY_KEY_SEQUENCES_DELETE,
        "pageUp" => LEGACY_KEY_SEQUENCES_PAGE_UP,
        "pageDown" => LEGACY_KEY_SEQUENCES_PAGE_DOWN,
        "clear" => LEGACY_KEY_SEQUENCES_CLEAR,
        "f1" => LEGACY_KEY_SEQUENCES_F1,
        "f2" => LEGACY_KEY_SEQUENCES_F2,
        "f3" => LEGACY_KEY_SEQUENCES_F3,
        "f4" => LEGACY_KEY_SEQUENCES_F4,
        "f5" => LEGACY_KEY_SEQUENCES_F5,
        "f6" => LEGACY_KEY_SEQUENCES_F6,
        "f7" => LEGACY_KEY_SEQUENCES_F7,
        "f8" => LEGACY_KEY_SEQUENCES_F8,
        "f9" => LEGACY_KEY_SEQUENCES_F9,
        "f10" => LEGACY_KEY_SEQUENCES_F10,
        "f11" => LEGACY_KEY_SEQUENCES_F11,
        "f12" => LEGACY_KEY_SEQUENCES_F12,
        _ => return None,
    })
}

fn legacy_shift_sequences(key: &str) -> Option<&'static [&'static str]> {
    Some(match key {
        "up" => &["\x1b[a"],
        "down" => &["\x1b[b"],
        "right" => &["\x1b[c"],
        "left" => &["\x1b[d"],
        "clear" => &["\x1b[e"],
        "insert" => &["\x1b[2$"],
        "delete" => &["\x1b[3$"],
        "pageUp" => &["\x1b[5$"],
        "pageDown" => &["\x1b[6$"],
        "home" => &["\x1b[7$"],
        "end" => &["\x1b[8$"],
        _ => return None,
    })
}

fn legacy_ctrl_sequences(key: &str) -> Option<&'static [&'static str]> {
    Some(match key {
        "up" => &["\x1bOa"],
        "down" => &["\x1bOb"],
        "right" => &["\x1bOc"],
        "left" => &["\x1bOd"],
        "clear" => &["\x1bOe"],
        "insert" => &["\x1b[2^"],
        "delete" => &["\x1b[3^"],
        "pageUp" => &["\x1b[5^"],
        "pageDown" => &["\x1b[6^"],
        "home" => &["\x1b[7^"],
        "end" => &["\x1b[8^"],
        _ => return None,
    })
}

/// `LEGACY_SEQUENCE_KEY_IDS` - exact sequences that map straight to a key id.
fn legacy_sequence_key_id(data: &str) -> Option<&'static str> {
    Some(match data {
        "\x1bOA" => "up",
        "\x1bOB" => "down",
        "\x1bOC" => "right",
        "\x1bOD" => "left",
        "\x1bOH" => "home",
        "\x1bOF" => "end",
        "\x1b[E" => "clear",
        "\x1bOE" => "clear",
        "\x1bOe" => "ctrl+clear",
        "\x1b[e" => "shift+clear",
        "\x1b[2~" => "insert",
        "\x1b[2$" => "shift+insert",
        "\x1b[2^" => "ctrl+insert",
        "\x1b[3$" => "shift+delete",
        "\x1b[3^" => "ctrl+delete",
        "\x1b[[5~" => "pageUp",
        "\x1b[[6~" => "pageDown",
        "\x1b[a" => "shift+up",
        "\x1b[b" => "shift+down",
        "\x1b[c" => "shift+right",
        "\x1b[d" => "shift+left",
        "\x1bOa" => "ctrl+up",
        "\x1bOb" => "ctrl+down",
        "\x1bOc" => "ctrl+right",
        "\x1bOd" => "ctrl+left",
        "\x1b[5$" => "shift+pageUp",
        "\x1b[6$" => "shift+pageDown",
        "\x1b[7$" => "shift+home",
        "\x1b[8$" => "shift+end",
        "\x1b[5^" => "ctrl+pageUp",
        "\x1b[6^" => "ctrl+pageDown",
        "\x1b[7^" => "ctrl+home",
        "\x1b[8^" => "ctrl+end",
        "\x1bOP" => "f1",
        "\x1bOQ" => "f2",
        "\x1bOR" => "f3",
        "\x1bOS" => "f4",
        "\x1b[11~" => "f1",
        "\x1b[12~" => "f2",
        "\x1b[13~" => "f3",
        "\x1b[14~" => "f4",
        "\x1b[[A" => "f1",
        "\x1b[[B" => "f2",
        "\x1b[[C" => "f3",
        "\x1b[[D" => "f4",
        "\x1b[[E" => "f5",
        "\x1b[15~" => "f5",
        "\x1b[17~" => "f6",
        "\x1b[18~" => "f7",
        "\x1b[19~" => "f8",
        "\x1b[20~" => "f9",
        "\x1b[21~" => "f10",
        "\x1b[23~" => "f11",
        "\x1b[24~" => "f12",
        "\x1bb" => "alt+left",
        "\x1bf" => "alt+right",
        "\x1bp" => "alt+up",
        "\x1bn" => "alt+down",
        _ => return None,
    })
}

fn matches_legacy_sequence(data: &str, sequences: &[&str]) -> bool {
    sequences.contains(&data)
}

fn matches_legacy_modifier_sequence(data: &str, key: &str, modifier: i64) -> bool {
    if modifier == MODIFIER_SHIFT {
        return match legacy_shift_sequences(key) {
            Some(seqs) => matches_legacy_sequence(data, seqs),
            None => false,
        };
    }
    if modifier == MODIFIER_CTRL {
        return match legacy_ctrl_sequences(key) {
            Some(seqs) => matches_legacy_sequence(data, seqs),
            None => false,
        };
    }
    false
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedKittySequence {
    codepoint: i64,
    shifted_key: Option<i64>,
    base_layout_key: Option<i64>,
    modifier: i64,
    event_type: KeyEventType,
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedModifyOtherKeysSequence {
    codepoint: i64,
    modifier: i64,
}

thread_local! {
    static LAST_EVENT_TYPE: std::cell::Cell<KeyEventType> = const { std::cell::Cell::new(KeyEventType::Press) };
}

fn set_last_event_type(event_type: KeyEventType) {
    LAST_EVENT_TYPE.with(|c| c.set(event_type));
}

fn _last_event_type() -> KeyEventType {
    LAST_EVENT_TYPE.with(|c| c.get())
}

/// Check if the last parsed key event was a key release.
pub fn is_key_release(data: &str) -> bool {
    // Don't treat bracketed paste content as key release, even if it contains
    // patterns like ":3F" (e.g., bluetooth MAC addresses like "90:62:3F:A5").
    if data.contains("\x1b[200~") {
        return false;
    }

    data.contains(":3u")
        || data.contains(":3~")
        || data.contains(":3A")
        || data.contains(":3B")
        || data.contains(":3C")
        || data.contains(":3D")
        || data.contains(":3H")
        || data.contains(":3F")
}

/// Check if the last parsed key event was a key repeat.
pub fn is_key_repeat(data: &str) -> bool {
    if data.contains("\x1b[200~") {
        return false;
    }

    data.contains(":2u")
        || data.contains(":2~")
        || data.contains(":2A")
        || data.contains(":2B")
        || data.contains(":2C")
        || data.contains(":2D")
        || data.contains(":2H")
        || data.contains(":2F")
}

fn parse_event_type(event_type_str: Option<&str>) -> KeyEventType {
    let s = match event_type_str {
        Some(s) if !s.is_empty() => s,
        _ => return KeyEventType::Press,
    };
    match s.parse::<i64>() {
        Ok(2) => KeyEventType::Repeat,
        Ok(3) => KeyEventType::Release,
        _ => KeyEventType::Press,
    }
}

static CSI_U_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$").unwrap());
static KITTY_ARROW_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$").unwrap());
static KITTY_FUNC_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$").unwrap());
static KITTY_HOME_END_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\x1b\[1;(\d+)(?::(\d+))?([HF])$").unwrap());
static MODIFY_OTHER_KEYS_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\x1b\[27;(\d+);(\d+)~$").unwrap());

fn parse_kitty_sequence(data: &str) -> Option<ParsedKittySequence> {
    if let Some(caps) = CSI_U_PATTERN.captures(data) {
        let codepoint: i64 = caps.get(1)?.as_str().parse().ok()?;
        let shifted_raw = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let shifted_key = if !shifted_raw.is_empty() {
            shifted_raw.parse::<i64>().ok()
        } else {
            None
        };
        let base_layout_key = caps.get(3).and_then(|m| m.as_str().parse::<i64>().ok());
        let mod_value = caps
            .get(4)
            .and_then(|m| m.as_str().parse::<i64>().ok())
            .unwrap_or(1);
        let event_type = parse_event_type(caps.get(5).map(|m| m.as_str()));
        set_last_event_type(event_type);
        return Some(ParsedKittySequence {
            codepoint,
            shifted_key,
            base_layout_key,
            modifier: mod_value - 1,
            event_type,
        });
    }

    if let Some(caps) = KITTY_ARROW_PATTERN.captures(data) {
        let mod_value: i64 = caps.get(1)?.as_str().parse().ok()?;
        let event_type = parse_event_type(caps.get(2).map(|m| m.as_str()));
        let codepoint = match caps.get(3)?.as_str() {
            "A" => ARROW_UP,
            "B" => ARROW_DOWN,
            "C" => ARROW_RIGHT,
            "D" => ARROW_LEFT,
            _ => return None,
        };
        set_last_event_type(event_type);
        return Some(ParsedKittySequence {
            codepoint,
            shifted_key: None,
            base_layout_key: None,
            modifier: mod_value - 1,
            event_type,
        });
    }

    if let Some(caps) = KITTY_FUNC_PATTERN.captures(data) {
        let key_num: i64 = caps.get(1)?.as_str().parse().ok()?;
        let mod_value = caps
            .get(2)
            .and_then(|m| m.as_str().parse::<i64>().ok())
            .unwrap_or(1);
        let event_type = parse_event_type(caps.get(3).map(|m| m.as_str()));
        let codepoint = match key_num {
            2 => Some(FUNCTIONAL_INSERT),
            3 => Some(FUNCTIONAL_DELETE),
            5 => Some(FUNCTIONAL_PAGE_UP),
            6 => Some(FUNCTIONAL_PAGE_DOWN),
            7 => Some(FUNCTIONAL_HOME),
            8 => Some(FUNCTIONAL_END),
            _ => None,
        };
        if let Some(codepoint) = codepoint {
            set_last_event_type(event_type);
            return Some(ParsedKittySequence {
                codepoint,
                shifted_key: None,
                base_layout_key: None,
                modifier: mod_value - 1,
                event_type,
            });
        }
    }

    if let Some(caps) = KITTY_HOME_END_PATTERN.captures(data) {
        let mod_value: i64 = caps.get(1)?.as_str().parse().ok()?;
        let event_type = parse_event_type(caps.get(2).map(|m| m.as_str()));
        let codepoint = if caps.get(3)?.as_str() == "H" {
            FUNCTIONAL_HOME
        } else {
            FUNCTIONAL_END
        };
        set_last_event_type(event_type);
        return Some(ParsedKittySequence {
            codepoint,
            shifted_key: None,
            base_layout_key: None,
            modifier: mod_value - 1,
            event_type,
        });
    }

    None
}

fn matches_kitty_sequence(data: &str, expected_codepoint: i64, expected_modifier: i64) -> bool {
    let parsed = match parse_kitty_sequence(data) {
        Some(p) => p,
        None => return false,
    };
    let actual_mod = parsed.modifier & !LOCK_MASK;
    let expected_mod = expected_modifier & !LOCK_MASK;

    if actual_mod != expected_mod {
        return false;
    }

    let normalized_codepoint = normalize_shifted_letter_identity_codepoint(
        normalize_kitty_functional_codepoint(parsed.codepoint),
        parsed.modifier,
    );
    let normalized_expected_codepoint = normalize_shifted_letter_identity_codepoint(
        normalize_kitty_functional_codepoint(expected_codepoint),
        expected_modifier,
    );

    if normalized_codepoint == normalized_expected_codepoint {
        return true;
    }

    // Alternate match: use base layout key for non-Latin keyboard layouts.
    if let Some(base_layout_key) = parsed.base_layout_key {
        if base_layout_key == expected_codepoint {
            let cp = normalized_codepoint;
            let is_latin_letter = (97..=122).contains(&cp); // a-z
            let is_known_symbol = cp >= 0 && cp <= 0x10ffff && is_symbol_key(&char_from_code(cp));
            if !is_latin_letter && !is_known_symbol {
                return true;
            }
        }
    }

    false
}

fn char_from_code(cp: i64) -> String {
    match u32::try_from(cp).ok().and_then(char::from_u32) {
        Some(c) => c.to_string(),
        None => String::new(),
    }
}

fn parse_modify_other_keys_sequence(data: &str) -> Option<ParsedModifyOtherKeysSequence> {
    let caps = MODIFY_OTHER_KEYS_PATTERN.captures(data)?;
    let mod_value: i64 = caps.get(1)?.as_str().parse().ok()?;
    let codepoint: i64 = caps.get(2)?.as_str().parse().ok()?;
    Some(ParsedModifyOtherKeysSequence {
        codepoint,
        modifier: mod_value - 1,
    })
}

fn matches_modify_other_keys(data: &str, expected_keycode: i64, expected_modifier: i64) -> bool {
    match parse_modify_other_keys_sequence(data) {
        Some(parsed) => parsed.codepoint == expected_keycode && parsed.modifier == expected_modifier,
        None => false,
    }
}

fn is_windows_terminal_session() -> bool {
    let get = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    get("WT_SESSION").is_some()
        && get("SSH_CONNECTION").is_none()
        && get("SSH_CLIENT").is_none()
        && get("SSH_TTY").is_none()
}

/// Raw 0x08 (BS) is ambiguous in legacy terminals.
fn matches_raw_backspace(data: &str, expected_modifier: i64) -> bool {
    if data == "\x7f" {
        return expected_modifier == 0;
    }
    if data != "\x08" {
        return false;
    }
    if is_windows_terminal_session() {
        expected_modifier == MODIFIER_CTRL
    } else {
        expected_modifier == 0
    }
}

/// Get the control character for a key. Uses the universal formula: code & 0x1f.
fn raw_ctrl_char(key: &str) -> Option<char> {
    let ch = key.to_lowercase().chars().next()?;
    let code = ch as u32;
    if (97..=122).contains(&code) || ch == '[' || ch == '\\' || ch == ']' || ch == '_' {
        return char::from_u32(code & 0x1f);
    }
    if ch == '-' {
        return char::from_u32(31); // Same as Ctrl+_
    }
    None
}

fn is_digit_key(key: &str) -> bool {
    key.len() == 1 && key.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
}

fn matches_printable_modify_other_keys(data: &str, expected_keycode: i64, expected_modifier: i64) -> bool {
    if expected_modifier == 0 {
        return false;
    }
    let parsed = match parse_modify_other_keys_sequence(data) {
        Some(p) => p,
        None => return false,
    };
    if parsed.modifier != expected_modifier {
        return false;
    }
    normalize_shifted_letter_identity_codepoint(parsed.codepoint, parsed.modifier)
        == normalize_shifted_letter_identity_codepoint(expected_keycode, expected_modifier)
}

fn format_key_name_with_modifiers(key_name: &str, modifier: i64) -> Option<String> {
    let mut mods: Vec<&str> = Vec::new();
    let effective_mod = modifier & !LOCK_MASK;
    let supported_modifier_mask = MODIFIER_SHIFT | MODIFIER_CTRL | MODIFIER_ALT | MODIFIER_SUPER;
    if (effective_mod & !supported_modifier_mask) != 0 {
        return None;
    }
    if (effective_mod & MODIFIER_SHIFT) != 0 {
        mods.push("shift");
    }
    if (effective_mod & MODIFIER_CTRL) != 0 {
        mods.push("ctrl");
    }
    if (effective_mod & MODIFIER_ALT) != 0 {
        mods.push("alt");
    }
    if (effective_mod & MODIFIER_SUPER) != 0 {
        mods.push("super");
    }
    if mods.is_empty() {
        Some(key_name.to_string())
    } else {
        Some(format!("{}+{}", mods.join("+"), key_name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedKeyId {
    key: String,
    ctrl: bool,
    shift: bool,
    alt: bool,
    super_modifier: bool,
}

fn parse_key_id(key_id: &str) -> Option<ParsedKeyId> {
    let lowered = key_id.to_lowercase();
    let parts: Vec<&str> = lowered.split('+').collect();
    let key = *parts.last()?;
    if key.is_empty() {
        return None;
    }
    Some(ParsedKeyId {
        key: key.to_string(),
        ctrl: parts.contains(&"ctrl"),
        shift: parts.contains(&"shift"),
        alt: parts.contains(&"alt"),
        super_modifier: parts.contains(&"super"),
    })
}

/// Opt-in decoding for macOS terminals that type a character instead of Option-as-Meta.
/// Keep this separate from normal matching so text editors can still accept `ß`.
pub fn matches_option_composed_key(data: &str, key_id: &str) -> bool {
    let Some(parsed) = parse_key_id(key_id) else {
        return false;
    };
    parsed.alt && !parsed.ctrl && !parsed.shift && !parsed.super_modifier
        && parsed.key == "s" && data == "ß"
}

/// Match input data against a key identifier string.
pub fn matches_key(data: &str, key_id: &str) -> bool {
    // Legacy macOS terminals encode Option as an extra ESC prefix around the
    // sequence for the remaining modifiers (for example ESC + Ctrl+Up).
    if data.starts_with("\x1b\x1b[") || data.starts_with("\x1b\x1bO") {
        if let Some(parsed_legacy_meta) = parse_key_id(key_id) {
            if parsed_legacy_meta.alt {
                let without_alt = key_id
                    .split('+')
                    .filter(|part| *part != "alt")
                    .collect::<Vec<_>>()
                    .join("+");
                let slice: String = data.chars().skip(1).collect();
                if matches_key(&slice, &without_alt) {
                    return true;
                }
            }
        }
    }
    let parsed = match parse_key_id(key_id) {
        Some(p) => p,
        None => return false,
    };

    let key = parsed.key.as_str();
    let mut modifier = 0i64;
    if parsed.shift {
        modifier |= MODIFIER_SHIFT;
    }
    if parsed.alt {
        modifier |= MODIFIER_ALT;
    }
    if parsed.ctrl {
        modifier |= MODIFIER_CTRL;
    }
    if parsed.super_modifier {
        modifier |= MODIFIER_SUPER;
    }

    match key {
        "escape" | "esc" => {
            if modifier != 0 {
                return false;
            }
            data == "\x1b"
                || matches_kitty_sequence(data, CODEPOINT_ESCAPE, 0)
                || matches_modify_other_keys(data, CODEPOINT_ESCAPE, 0)
        }

        "space" => {
            if !_kitty_protocol_active() {
                if modifier == MODIFIER_CTRL && data == "\x00" {
                    return true;
                }
                if modifier == MODIFIER_ALT && data == "\x1b " {
                    return true;
                }
            }
            if modifier == 0 {
                return data == " "
                    || matches_kitty_sequence(data, CODEPOINT_SPACE, 0)
                    || matches_modify_other_keys(data, CODEPOINT_SPACE, 0);
            }
            matches_kitty_sequence(data, CODEPOINT_SPACE, modifier)
                || matches_modify_other_keys(data, CODEPOINT_SPACE, modifier)
        }

        "tab" => {
            if modifier == MODIFIER_SHIFT {
                return data == "\x1b[Z"
                    || matches_kitty_sequence(data, CODEPOINT_TAB, MODIFIER_SHIFT)
                    || matches_modify_other_keys(data, CODEPOINT_TAB, MODIFIER_SHIFT);
            }
            if modifier == 0 {
                return data == "\t" || matches_kitty_sequence(data, CODEPOINT_TAB, 0);
            }
            matches_kitty_sequence(data, CODEPOINT_TAB, modifier)
                || matches_modify_other_keys(data, CODEPOINT_TAB, modifier)
        }

        "enter" | "return" => {
            if modifier == MODIFIER_SHIFT {
                if matches_kitty_sequence(data, CODEPOINT_ENTER, MODIFIER_SHIFT)
                    || matches_kitty_sequence(data, CODEPOINT_KP_ENTER, MODIFIER_SHIFT)
                {
                    return true;
                }
                if matches_modify_other_keys(data, CODEPOINT_ENTER, MODIFIER_SHIFT) {
                    return true;
                }
                if _kitty_protocol_active() {
                    return data == "\x1b\r" || data == "\n";
                }
                return false;
            }
            if modifier == MODIFIER_ALT {
                if matches_kitty_sequence(data, CODEPOINT_ENTER, MODIFIER_ALT)
                    || matches_kitty_sequence(data, CODEPOINT_KP_ENTER, MODIFIER_ALT)
                {
                    return true;
                }
                if matches_modify_other_keys(data, CODEPOINT_ENTER, MODIFIER_ALT) {
                    return true;
                }
                if !_kitty_protocol_active() {
                    return data == "\x1b\r";
                }
                return false;
            }
            if modifier == 0 {
                return data == "\r"
                    || (!_kitty_protocol_active() && data == "\n")
                    || data == "\x1bOM" // SS3 M (numpad enter in some terminals)
                    || matches_kitty_sequence(data, CODEPOINT_ENTER, 0)
                    || matches_kitty_sequence(data, CODEPOINT_KP_ENTER, 0);
            }
            matches_kitty_sequence(data, CODEPOINT_ENTER, modifier)
                || matches_kitty_sequence(data, CODEPOINT_KP_ENTER, modifier)
                || matches_modify_other_keys(data, CODEPOINT_ENTER, modifier)
        }

        "backspace" => {
            if modifier == MODIFIER_ALT {
                if data == "\x1b\x7f" || data == "\x1b\x08" {
                    return true;
                }
                return matches_kitty_sequence(data, CODEPOINT_BACKSPACE, MODIFIER_ALT)
                    || matches_modify_other_keys(data, CODEPOINT_BACKSPACE, MODIFIER_ALT);
            }
            if modifier == MODIFIER_CTRL {
                if matches_raw_backspace(data, MODIFIER_CTRL) {
                    return true;
                }
                return matches_kitty_sequence(data, CODEPOINT_BACKSPACE, MODIFIER_CTRL)
                    || matches_modify_other_keys(data, CODEPOINT_BACKSPACE, MODIFIER_CTRL);
            }
            if modifier == 0 {
                return matches_raw_backspace(data, 0)
                    || matches_kitty_sequence(data, CODEPOINT_BACKSPACE, 0)
                    || matches_modify_other_keys(data, CODEPOINT_BACKSPACE, 0);
            }
            matches_kitty_sequence(data, CODEPOINT_BACKSPACE, modifier)
                || matches_modify_other_keys(data, CODEPOINT_BACKSPACE, modifier)
        }

        "insert" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_INSERT)
                    || matches_kitty_sequence(data, FUNCTIONAL_INSERT, 0);
            }
            if matches_legacy_modifier_sequence(data, "insert", modifier) {
                return true;
            }
            matches_kitty_sequence(data, FUNCTIONAL_INSERT, modifier)
        }

        "delete" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_DELETE)
                    || matches_kitty_sequence(data, FUNCTIONAL_DELETE, 0);
            }
            if matches_legacy_modifier_sequence(data, "delete", modifier) {
                return true;
            }
            matches_kitty_sequence(data, FUNCTIONAL_DELETE, modifier)
        }

        "clear" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_CLEAR);
            }
            matches_legacy_modifier_sequence(data, "clear", modifier)
        }

        "home" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_HOME)
                    || matches_kitty_sequence(data, FUNCTIONAL_HOME, 0);
            }
            if matches_legacy_modifier_sequence(data, "home", modifier) {
                return true;
            }
            matches_kitty_sequence(data, FUNCTIONAL_HOME, modifier)
        }

        "end" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_END)
                    || matches_kitty_sequence(data, FUNCTIONAL_END, 0);
            }
            if matches_legacy_modifier_sequence(data, "end", modifier) {
                return true;
            }
            matches_kitty_sequence(data, FUNCTIONAL_END, modifier)
        }

        "pageup" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_PAGE_UP)
                    || matches_kitty_sequence(data, FUNCTIONAL_PAGE_UP, 0);
            }
            if matches_legacy_modifier_sequence(data, "pageUp", modifier) {
                return true;
            }
            matches_kitty_sequence(data, FUNCTIONAL_PAGE_UP, modifier)
        }

        "pagedown" => {
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_PAGE_DOWN)
                    || matches_kitty_sequence(data, FUNCTIONAL_PAGE_DOWN, 0);
            }
            if matches_legacy_modifier_sequence(data, "pageDown", modifier) {
                return true;
            }
            matches_kitty_sequence(data, FUNCTIONAL_PAGE_DOWN, modifier)
        }

        "up" => {
            if modifier == MODIFIER_ALT {
                return data == "\x1b[1;3A"
                    || data == "\x1bp"
                    || matches_kitty_sequence(data, ARROW_UP, MODIFIER_ALT);
            }
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_UP)
                    || matches_kitty_sequence(data, ARROW_UP, 0);
            }
            if matches_legacy_modifier_sequence(data, "up", modifier) {
                return true;
            }
            matches_kitty_sequence(data, ARROW_UP, modifier)
        }

        "down" => {
            if modifier == MODIFIER_ALT {
                return data == "\x1b[1;3B"
                    || data == "\x1bn"
                    || matches_kitty_sequence(data, ARROW_DOWN, MODIFIER_ALT);
            }
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_DOWN)
                    || matches_kitty_sequence(data, ARROW_DOWN, 0);
            }
            if matches_legacy_modifier_sequence(data, "down", modifier) {
                return true;
            }
            matches_kitty_sequence(data, ARROW_DOWN, modifier)
        }

        "left" => {
            if modifier == MODIFIER_ALT {
                return data == "\x1b[1;3D"
                    || (!_kitty_protocol_active() && data == "\x1bB")
                    || data == "\x1bb"
                    || matches_kitty_sequence(data, ARROW_LEFT, MODIFIER_ALT);
            }
            if modifier == MODIFIER_CTRL {
                return data == "\x1b[1;5D"
                    || matches_legacy_modifier_sequence(data, "left", MODIFIER_CTRL)
                    || matches_kitty_sequence(data, ARROW_LEFT, MODIFIER_CTRL);
            }
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_LEFT)
                    || matches_kitty_sequence(data, ARROW_LEFT, 0);
            }
            if matches_legacy_modifier_sequence(data, "left", modifier) {
                return true;
            }
            matches_kitty_sequence(data, ARROW_LEFT, modifier)
        }

        "right" => {
            if modifier == MODIFIER_ALT {
                return data == "\x1b[1;3C"
                    || (!_kitty_protocol_active() && data == "\x1bF")
                    || data == "\x1bf"
                    || matches_kitty_sequence(data, ARROW_RIGHT, MODIFIER_ALT);
            }
            if modifier == MODIFIER_CTRL {
                return data == "\x1b[1;5C"
                    || matches_legacy_modifier_sequence(data, "right", MODIFIER_CTRL)
                    || matches_kitty_sequence(data, ARROW_RIGHT, MODIFIER_CTRL);
            }
            if modifier == 0 {
                return matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES_RIGHT)
                    || matches_kitty_sequence(data, ARROW_RIGHT, 0);
            }
            if matches_legacy_modifier_sequence(data, "right", modifier) {
                return true;
            }
            matches_kitty_sequence(data, ARROW_RIGHT, modifier)
        }

        "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8" | "f9" | "f10" | "f11" | "f12" => {
            if modifier != 0 {
                return false;
            }
            match legacy_key_sequences(key) {
                Some(sequences) => matches_legacy_sequence(data, sequences),
                None => false,
            }
        }

        _ => {
            if key.chars().count() == 1 {
                let ch = key.chars().next().unwrap();
                let is_letter = ch.is_ascii_lowercase();
                let is_digit = ch.is_ascii_digit();
                let is_symbol = is_symbol_key(key);
                if !(is_letter || is_digit || is_symbol) {
                    return false;
                }

                let codepoint = ch as i64;
                let raw_ctrl = raw_ctrl_char(key);
                let is_letter = ch >= 'a' && ch <= 'z';
                let is_digit = is_digit_key(key);

                if modifier == MODIFIER_CTRL + MODIFIER_ALT && !_kitty_protocol_active() {
                    if let Some(raw_ctrl) = raw_ctrl {
                        // Legacy: ctrl+alt+key is ESC followed by the control character.
                        if data == format!("\x1b{raw_ctrl}") {
                            return true;
                        }
                    }
                }

                if modifier == MODIFIER_ALT && !_kitty_protocol_active() && (is_letter || is_digit) {
                    // Legacy: alt+letter/digit is ESC followed by the key
                    if data == format!("\x1b{key}") {
                        return true;
                    }
                }

                if modifier == MODIFIER_CTRL {
                    if let Some(raw_ctrl) = raw_ctrl {
                        if data == raw_ctrl.to_string() {
                            return true;
                        }
                    }
                    return matches_kitty_sequence(data, codepoint, MODIFIER_CTRL)
                        || matches_printable_modify_other_keys(data, codepoint, MODIFIER_CTRL);
                }

                if modifier == MODIFIER_SHIFT + MODIFIER_CTRL {
                    return matches_kitty_sequence(data, codepoint, MODIFIER_SHIFT + MODIFIER_CTRL)
                        || matches_printable_modify_other_keys(data, codepoint, MODIFIER_SHIFT + MODIFIER_CTRL);
                }

                if modifier == MODIFIER_SHIFT {
                    // Legacy: shift+letter produces uppercase
                    if is_letter && data == key.to_uppercase() {
                        return true;
                    }
                    return matches_kitty_sequence(data, codepoint, MODIFIER_SHIFT)
                        || matches_printable_modify_other_keys(data, codepoint, MODIFIER_SHIFT);
                }

                if modifier != 0 {
                    return matches_kitty_sequence(data, codepoint, modifier)
                        || matches_printable_modify_other_keys(data, codepoint, modifier);
                }

                return data == key || matches_kitty_sequence(data, codepoint, 0);
            }
            false
        }
    }
}

fn format_parsed_key(codepoint: i64, modifier: i64, base_layout_key: Option<i64>) -> Option<String> {
    let normalized_codepoint = normalize_kitty_functional_codepoint(codepoint);
    let identity_codepoint = normalize_shifted_letter_identity_codepoint(normalized_codepoint, modifier);

    let is_latin_letter = (97..=122).contains(&identity_codepoint); // a-z
    let is_digit = (48..=57).contains(&identity_codepoint); // 0-9
    let is_known_symbol = is_symbol_key(&char_from_code(identity_codepoint));
    let effective_codepoint = if is_latin_letter || is_digit || is_known_symbol {
        identity_codepoint
    } else {
        base_layout_key.unwrap_or(identity_codepoint)
    };

    let key_name: Option<String> = if effective_codepoint == CODEPOINT_ESCAPE {
        Some("escape".to_string())
    } else if effective_codepoint == CODEPOINT_TAB {
        Some("tab".to_string())
    } else if effective_codepoint == CODEPOINT_ENTER || effective_codepoint == CODEPOINT_KP_ENTER {
        Some("enter".to_string())
    } else if effective_codepoint == CODEPOINT_SPACE {
        Some("space".to_string())
    } else if effective_codepoint == CODEPOINT_BACKSPACE {
        Some("backspace".to_string())
    } else if effective_codepoint == FUNCTIONAL_DELETE {
        Some("delete".to_string())
    } else if effective_codepoint == FUNCTIONAL_INSERT {
        Some("insert".to_string())
    } else if effective_codepoint == FUNCTIONAL_HOME {
        Some("home".to_string())
    } else if effective_codepoint == FUNCTIONAL_END {
        Some("end".to_string())
    } else if effective_codepoint == FUNCTIONAL_PAGE_UP {
        Some("pageUp".to_string())
    } else if effective_codepoint == FUNCTIONAL_PAGE_DOWN {
        Some("pageDown".to_string())
    } else if effective_codepoint == ARROW_UP {
        Some("up".to_string())
    } else if effective_codepoint == ARROW_DOWN {
        Some("down".to_string())
    } else if effective_codepoint == ARROW_LEFT {
        Some("left".to_string())
    } else if effective_codepoint == ARROW_RIGHT {
        Some("right".to_string())
    } else if (48..=57).contains(&effective_codepoint) {
        Some(char_from_code(effective_codepoint))
    } else if (97..=122).contains(&effective_codepoint) {
        Some(char_from_code(effective_codepoint))
    } else if is_symbol_key(&char_from_code(effective_codepoint)) {
        Some(char_from_code(effective_codepoint))
    } else {
        None
    };

    let key_name = key_name?;
    format_key_name_with_modifiers(&key_name, modifier)
}

/// Parse input data and return the key identifier if recognized.
pub fn parse_key(data: &str) -> Option<String> {
    if let Some(kitty) = parse_kitty_sequence(data) {
        return format_parsed_key(kitty.codepoint, kitty.modifier, kitty.base_layout_key);
    }

    if let Some(modify_other_keys) = parse_modify_other_keys_sequence(data) {
        return format_parsed_key(modify_other_keys.codepoint, modify_other_keys.modifier, None);
    }

    if _kitty_protocol_active() && (data == "\x1b\r" || data == "\n") {
        return Some("shift+enter".to_string());
    }

    if let Some(legacy_sequence_key_id) = legacy_sequence_key_id(data) {
        return Some(legacy_sequence_key_id.to_string());
    }

    // Legacy sequences (used when Kitty protocol is not active, or for unambiguous sequences)
    match data {
        "\x1b" => return Some("escape".to_string()),
        "\x1c" => return Some("ctrl+\\".to_string()),
        "\x1d" => return Some("ctrl+]".to_string()),
        "\x1f" => return Some("ctrl+-".to_string()),
        "\x1b\x1b" => return Some("ctrl+alt+[".to_string()),
        "\x1b\x1c" => return Some("ctrl+alt+\\".to_string()),
        "\x1b\x1d" => return Some("ctrl+alt+]".to_string()),
        "\x1b\x1f" => return Some("ctrl+alt+-".to_string()),
        "\t" => return Some("tab".to_string()),
        "\x00" => return Some("ctrl+space".to_string()),
        " " => return Some("space".to_string()),
        "\x7f" => return Some("backspace".to_string()),
        "\x1b[Z" => return Some("shift+tab".to_string()),
        "\x1b\x7f" | "\x1b\x08" => return Some("alt+backspace".to_string()),
        "\x1b[A" => return Some("up".to_string()),
        "\x1b[B" => return Some("down".to_string()),
        "\x1b[C" => return Some("right".to_string()),
        "\x1b[D" => return Some("left".to_string()),
        "\x1b[H" | "\x1bOH" => return Some("home".to_string()),
        "\x1b[F" | "\x1bOF" => return Some("end".to_string()),
        "\x1b[3~" => return Some("delete".to_string()),
        "\x1b[5~" => return Some("pageUp".to_string()),
        "\x1b[6~" => return Some("pageDown".to_string()),
        _ => {}
    }
    if data == "\r" || (!_kitty_protocol_active() && data == "\n") || data == "\x1bOM" {
        return Some("enter".to_string());
    }
    if data == "\x08" {
        return Some(if is_windows_terminal_session() {
            "ctrl+backspace".to_string()
        } else {
            "backspace".to_string()
        });
    }
    if !_kitty_protocol_active() && data == "\x1b\r" {
        return Some("alt+enter".to_string());
    }
    if !_kitty_protocol_active() && data == "\x1b " {
        return Some("alt+space".to_string());
    }
    if !_kitty_protocol_active() && data == "\x1bB" {
        return Some("alt+left".to_string());
    }
    if !_kitty_protocol_active() && data == "\x1bF" {
        return Some("alt+right".to_string());
    }
    if !_kitty_protocol_active() && data.len() == 2 && data.starts_with('\x1b') {
        let code = data.as_bytes()[1] as i64;
        if (1..=26).contains(&code) {
            return Some(format!("ctrl+alt+{}", char_from_code(code + 96)));
        }
        // Legacy alt+letter/digit (ESC followed by the key)
        if (97..=122).contains(&code) || (48..=57).contains(&code) {
            return Some(format!("alt+{}", char_from_code(code)));
        }
    }

    if data.chars().count() == 1 {
        let code = data.chars().next().unwrap() as i64;
        if (1..=26).contains(&code) {
            return Some(format!("ctrl+{}", char_from_code(code + 96)));
        }
        if (32..=126).contains(&code) {
            return Some(data.to_string());
        }
    }

    None
}

static KITTY_CSI_U_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$").unwrap());

const KITTY_PRINTABLE_ALLOWED_MODIFIERS: i64 = MODIFIER_SHIFT | LOCK_MASK;

/// Decode a Kitty CSI-u sequence into a printable character, if applicable.
pub fn decode_kitty_printable(data: &str) -> Option<String> {
    let caps = KITTY_CSI_U_REGEX.captures(data)?;

    // CSI-u groups: <codepoint>[:<shifted>[:<base>]];<mod>[:<event>]u
    let codepoint: i64 = caps.get(1).and_then(|m| m.as_str().parse().ok())?;

    let shifted_raw = caps.get(2).map(|m| m.as_str()).unwrap_or("");
    let shifted_key = if !shifted_raw.is_empty() {
        shifted_raw.parse::<i64>().ok()
    } else {
        None
    };
    let mod_value = caps
        .get(4)
        .and_then(|m| m.as_str().parse::<i64>().ok())
        .unwrap_or(1);
    // Modifiers are 1-indexed in CSI-u; normalize to our bitmask.
    let modifier = mod_value - 1;

    // Only accept printable CSI-u input for plain or Shift-modified text keys.
    if (modifier & !KITTY_PRINTABLE_ALLOWED_MODIFIERS) != 0 {
        return None;
    }
    if (modifier & (MODIFIER_ALT | MODIFIER_CTRL)) != 0 {
        return None;
    }

    // Prefer the shifted keycode when Shift is held.
    let mut effective_codepoint = codepoint;
    if (modifier & MODIFIER_SHIFT) != 0 {
        if let Some(shifted_key) = shifted_key {
            effective_codepoint = shifted_key;
        }
    }
    effective_codepoint = normalize_kitty_functional_codepoint(effective_codepoint);
    // Drop control characters or invalid codepoints.
    if effective_codepoint < 32 {
        return None;
    }

    let c = u32::try_from(effective_codepoint)
        .ok()
        .and_then(char::from_u32)?;
    Some(c.to_string())
}

fn decode_modify_other_keys_printable(data: &str) -> Option<String> {
    let parsed = parse_modify_other_keys_sequence(data)?;
    let modifier = parsed.modifier & !LOCK_MASK;
    if (modifier & !MODIFIER_SHIFT) != 0 {
        return None;
    }
    if parsed.codepoint < 32 {
        return None;
    }
    let c = u32::try_from(parsed.codepoint).ok().and_then(char::from_u32)?;
    Some(c.to_string())
}

pub fn decode_printable_key(data: &str) -> Option<String> {
    decode_kitty_printable(data).or_else(|| decode_modify_other_keys_printable(data))
}

/// Port of the `Key` helper object: modifier-prefixed key identifiers.
pub fn key_ctrl(key: &str) -> String {
    format!("ctrl+{key}")
}
pub fn key_shift(key: &str) -> String {
    format!("shift+{key}")
}
pub fn key_alt(key: &str) -> String {
    format!("alt+{key}")
}
pub fn key_super(key: &str) -> String {
    format!("super+{key}")
}
pub fn key_ctrl_shift(key: &str) -> String {
    format!("ctrl+shift+{key}")
}
pub fn key_shift_ctrl(key: &str) -> String {
    format!("shift+ctrl+{key}")
}
pub fn key_ctrl_alt(key: &str) -> String {
    format!("ctrl+alt+{key}")
}
pub fn key_alt_ctrl(key: &str) -> String {
    format!("alt+ctrl+{key}")
}
pub fn key_shift_alt(key: &str) -> String {
    format!("shift+alt+{key}")
}
pub fn key_alt_shift(key: &str) -> String {
    format!("alt+shift+{key}")
}
pub fn key_ctrl_super(key: &str) -> String {
    format!("ctrl+super+{key}")
}
pub fn key_super_ctrl(key: &str) -> String {
    format!("super+ctrl+{key}")
}
pub fn key_shift_super(key: &str) -> String {
    format!("shift+super+{key}")
}
pub fn key_super_shift(key: &str) -> String {
    format!("super+shift+{key}")
}
pub fn key_alt_super(key: &str) -> String {
    format!("alt+super+{key}")
}
pub fn key_super_alt(key: &str) -> String {
    format!("super+alt+{key}")
}
pub fn key_ctrl_shift_alt(key: &str) -> String {
    format!("ctrl+shift+alt+{key}")
}
pub fn key_ctrl_shift_super(key: &str) -> String {
    format!("ctrl+shift+super+{key}")
}

/// Port of the `LEGACY_MODIFIER_KEYS` type union helper (kept for parity checks).
pub fn is_legacy_modifier_key(key: &str) -> bool {
    LEGACY_MODIFIER_KEYS.contains(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct KittyGuard;

    impl KittyGuard {
        fn new(active: bool) -> Self {
            set_kitty_protocol_active(active);
            KittyGuard
        }
    }

    impl Drop for KittyGuard {
        fn drop(&mut self) {
            set_kitty_protocol_active(false);
        }
    }

    #[test]
    fn matches_plain_keys() {
        let _guard = KittyGuard::new(false);
        assert!(matches_key("\x1b", "escape"));
        assert!(matches_key("\r", "enter"));
        assert!(matches_key("\t", "tab"));
        assert!(matches_key("\x1b[Z", "shift+tab"));
        assert!(!matches_key("\x1b", "enter"));
    }

    #[test]
    fn matches_control_and_alt_letters() {
        let _guard = KittyGuard::new(false);
        assert!(matches_key("\x03", "ctrl+c"));
        assert!(matches_key("\x1b", "escape"));
        assert!(matches_key("\x1bx", "alt+x"));
        assert!(matches_key("\x1b\x03", "ctrl+alt+c"));
        assert!(!matches_key("\x03", "ctrl+d"));
    }

    #[test]
    fn option_composed_matching_is_explicit_and_modifier_specific() {
        for active in [false, true] {
            let _guard = KittyGuard::new(active);
            assert!(matches_option_composed_key("ß", "alt+s"));
            assert!(!matches_key("ß", "alt+s"));
            for key in ["s", "alt+a", "ctrl+alt+s", "shift+alt+s", "super+alt+s", "alt+"] {
                assert!(!matches_option_composed_key("ß", key), "{key}");
            }
            for data in ["s", "ẞ", "ßs", "\x1b[200~ß\x1b[201~"] {
                assert!(!matches_option_composed_key(data, "alt+s"), "{data:?}");
            }
        }
    }

    #[test]
    fn matches_arrow_sequences_and_modifiers() {
        let _guard = KittyGuard::new(false);
        assert!(matches_key("\x1b[A", "up"));
        assert!(matches_key("\x1b[1;5D", "ctrl+left"));
        assert!(matches_key("\x1b[1;3C", "alt+right"));
        assert!(matches_key("\x1bb", "alt+left"));
        assert!(!matches_key("\x1b[A", "down"));
    }

    #[test]
    fn matches_kitty_csi_u_sequences() {
        let _guard = KittyGuard::new(true);
        assert!(matches_key("\x1b[99;5u", "ctrl+c"));
        assert!(matches_key("\x1b[13;2u", "shift+enter"));
        assert!(matches_key("\x1b[27u", "escape"));
        assert!(matches_key("\x1b[9;2u", "shift+tab"));
        assert!(!matches_key("\x1b[99;5u", "ctrl+d"));
    }

    #[test]
    fn parse_key_reports_legacy_and_kitty() {
        let _guard = KittyGuard::new(false);
        assert_eq!(parse_key("\x1b[A").as_deref(), Some("up"));
        assert_eq!(parse_key("\x03").as_deref(), Some("ctrl+c"));
        assert_eq!(parse_key("\x1b[Z").as_deref(), Some("shift+tab"));
        assert_eq!(parse_key("\x1b[99;5u").as_deref(), Some("ctrl+c"));
        assert_eq!(parse_key("x").as_deref(), Some("x"));
    }

    #[test]
    fn decode_printable_key_accepts_shifted_csi_u() {
        assert_eq!(decode_kitty_printable("\x1b[97u").as_deref(), Some("a"));
        assert_eq!(decode_kitty_printable("\x1b[97;2u").as_deref(), Some("a"));
        assert_eq!(decode_kitty_printable("\x1b[97:65;2u").as_deref(), Some("A"));
        assert_eq!(decode_kitty_printable("\x1b[97;5u"), None);
        assert_eq!(decode_printable_key("\x1b[27;2;97~").as_deref(), Some("a"));
    }

    #[test]
    fn key_release_and_repeat_detection() {
        assert!(is_key_release("\x1b[99;5:3u"));
        assert!(!is_key_release("\x1b[200~:3u"));
        assert!(is_key_repeat("\x1b[99;5:2u"));
        assert!(!is_key_repeat("\x1b[99;5:3u"));
    }

    #[test]
    fn helper_constructors_build_key_ids() {
        assert_eq!(key_ctrl("c"), "ctrl+c");
        assert_eq!(key_ctrl_shift("p"), "ctrl+shift+p");
        assert_eq!(key_super("k"), "super+k");
    }
}
