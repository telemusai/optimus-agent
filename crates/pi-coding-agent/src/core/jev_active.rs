//! Pure request-body transformation for Jev Active mode.
//!
//! This module owns exactly one thing: turning an accepted Jev decision
//! (a category plus a value) into bounded, provider-key-aware edits of an
//! outgoing provider request body. It does not decide *whether* a decision is
//! accepted, how it is scored, or what the confidence policy is; those live in
//! crate `pi-jev`. It also does not know about provider SDK types, transports,
//! or retries: callers hand it a `serde_json::Value` body and read back the
//! list of applied changes.
//!
//! Guarantees: never panics, never creates a key that was absent, and never
//! touches a key other than the ones it is asked to change.

use serde_json::Value;

/// One applied change to the outgoing provider request body. `key` is the
/// request-body key that changed; `from`/`to` are bounded JSON renderings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedChange {
    pub key: String,
    pub from: Option<String>,
    pub to: Option<String>,
    pub category: String,
}

/// Request-body key holding the advertised tool schema list.
pub const TOOLS_KEY: &str = "tools";
/// Request-body key holding the tool-selection hint.
pub const TOOL_CHOICE_KEY: &str = "tool_choice";
/// Request-body key holding the provider reasoning-effort hint.
pub const REASONING_EFFORT_KEY: &str = "reasoning_effort";

/// Supported reasoning-effort values in increasing order.
pub const REASONING_LADDER: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/// Maximum byte length of a rendered `from`/`to` value before truncation.
const MAX_RENDER_BYTES: usize = 120;

const TRUNCATION_MARKER: &str = "...";

fn render(value: &Value) -> String {
    let text = serde_json::to_string(value).unwrap_or_else(|_| String::from("<unrenderable>"));
    truncate_bytes(text, MAX_RENDER_BYTES)
}

/// Truncate to at most `max` bytes on a UTF-8 boundary, appending the marker
/// when anything was dropped.
fn truncate_bytes(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + TRUNCATION_MARKER.len());
    out.push_str(&text[..end]);
    out.push_str(TRUNCATION_MARKER);
    out
}

/// Index of a reasoning-effort value, if it is on the supported ladder.
fn ladder_index(effort: &str) -> Option<usize> {
    REASONING_LADDER.iter().position(|step| *step == effort)
}

/// Apply one accepted Jev decision to an outgoing provider request body.
/// Pure: mutates `params` in place, returns the applied changes (empty when
/// nothing was applied). Never panics, never adds a key that does not exist.
pub fn apply_decision(params: &mut Value, category: &str, value: &str) -> Vec<AppliedChange> {
    let Some(object) = params.as_object_mut() else {
        return Vec::new();
    };

    if category == "tool_requirement" {
        return apply_tool_requirement(object, category, value);
    }
    if category == "complexity" {
        return apply_complexity(object, category, value);
    }

    Vec::new()
}

/// Drop the tool surface when the decision says no tool is required.
///
/// A missing or empty `tools` array means there is nothing to disable, so the
/// body is left byte-identical.
fn apply_tool_requirement(
    object: &mut serde_json::Map<String, Value>,
    category: &str,
    value: &str,
) -> Vec<AppliedChange> {
    if !value.trim().eq_ignore_ascii_case("none") {
        return Vec::new();
    }
    let has_tools = matches!(object.get(TOOLS_KEY), Some(Value::Array(tools)) if !tools.is_empty());
    if !has_tools {
        return Vec::new();
    }

    let mut changes = Vec::new();
    if let Some(removed) = object.remove(TOOLS_KEY) {
        changes.push(AppliedChange {
            key: TOOLS_KEY.to_string(),
            from: Some(render(&removed)),
            to: None,
            category: category.to_string(),
        });
    }
    if let Some(removed) = object.remove(TOOL_CHOICE_KEY) {
        changes.push(AppliedChange {
            key: TOOL_CHOICE_KEY.to_string(),
            from: Some(render(&removed)),
            to: None,
            category: category.to_string(),
        });
    }
    changes
}

/// Nudge `reasoning_effort` by exactly one ladder step.
///
/// Only a present, ladder-known string effort is touched; anything else keeps
/// the body byte-identical. Clamping that would not move the value records no
/// change.
fn apply_complexity(
    object: &mut serde_json::Map<String, Value>,
    category: &str,
    value: &str,
) -> Vec<AppliedChange> {
    let direction = match value.trim().to_ascii_lowercase().as_str() {
        "low" => -1i32,
        "high" => 1i32,
        _ => return Vec::new(),
    };

    let Some(current) = object
        .get(REASONING_EFFORT_KEY)
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return Vec::new();
    };
    let Some(index) = ladder_index(current.as_str()) else {
        return Vec::new();
    };

    let next_index = if direction < 0 {
        if index <= ladder_index("low").unwrap_or(0) {
            return Vec::new();
        }
        index - 1
    } else {
        if index + 1 >= REASONING_LADDER.len() {
            return Vec::new();
        }
        index + 1
    };

    let next = REASONING_LADDER[next_index];
    object.insert(
        REASONING_EFFORT_KEY.to_string(),
        Value::String(next.to_string()),
    );

    vec![AppliedChange {
        key: REASONING_EFFORT_KEY.to_string(),
        from: Some(current),
        to: Some(next.to_string()),
        category: category.to_string(),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat_body() -> Value {
        json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "shell_exec"}}],
            "tool_choice": "auto",
            "stream": true
        })
    }

    #[test]
    fn tool_requirement_none_removes_tools_and_tool_choice() {
        let mut body = chat_body();
        let changes = apply_decision(&mut body, "tool_requirement", "none");

        assert!(body.get(TOOLS_KEY).is_none());
        assert!(body.get(TOOL_CHOICE_KEY).is_none());
        assert_eq!(body["model"], json!("gpt-5"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["messages"].as_array().map(Vec::len), Some(1));

        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].key, TOOLS_KEY);
        assert_eq!(changes[0].category, "tool_requirement");
        assert_eq!(changes[0].to, None);
        assert!(changes[0]
            .from
            .as_deref()
            .unwrap_or_default()
            .contains("shell_exec"));
        assert_eq!(changes[1].key, TOOL_CHOICE_KEY);
        assert_eq!(changes[1].from.as_deref(), Some("\"auto\""));
        assert_eq!(changes[1].to, None);
    }

    #[test]
    fn tool_requirement_none_is_case_insensitive_and_trimmed() {
        let mut body = chat_body();
        let changes = apply_decision(&mut body, "tool_requirement", "  NoNe ");
        assert_eq!(changes.len(), 2);
        assert!(body.get(TOOLS_KEY).is_none());
        assert!(body.get(TOOL_CHOICE_KEY).is_none());
    }

    #[test]
    fn tool_requirement_none_with_empty_tools_is_byte_identical() {
        let mut body = json!({
            "model": "gpt-5",
            "messages": [],
            "tools": [],
            "tool_choice": "auto",
            "stream": true
        });
        let before = serde_json::to_string(&body).expect("serialize");
        let changes = apply_decision(&mut body, "tool_requirement", "none");
        assert!(changes.is_empty());
        assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
    }

    #[test]
    fn tool_requirement_none_with_absent_tools_is_byte_identical() {
        let mut body = json!({"model": "gpt-5", "tool_choice": "auto", "stream": true});
        let before = serde_json::to_string(&body).expect("serialize");
        let changes = apply_decision(&mut body, "tool_requirement", "none");
        assert!(changes.is_empty());
        assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
    }

    #[test]
    fn tool_requirement_other_values_change_nothing() {
        let mut body = chat_body();
        let before = serde_json::to_string(&body).expect("serialize");
        assert!(apply_decision(&mut body, "tool_requirement", "optional").is_empty());
        assert!(apply_decision(&mut body, "tool_requirement", "required").is_empty());
        assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
    }

    #[test]
    fn non_object_params_return_empty_and_do_not_panic() {
        for mut body in [json!("plain"), json!(7), json!([1, 2, 3]), Value::Null] {
            let before = serde_json::to_string(&body).expect("serialize");
            assert!(apply_decision(&mut body, "tool_requirement", "none").is_empty());
            assert!(apply_decision(&mut body, "complexity", "low").is_empty());
            assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
        }
    }

    #[test]
    fn unknown_category_leaves_body_byte_identical() {
        let mut body = chat_body();
        let before = serde_json::to_string(&body).expect("serialize");
        assert!(apply_decision(&mut body, "verbosity", "concise").is_empty());
        assert!(apply_decision(&mut body, "", "none").is_empty());
        assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
    }

    #[test]
    fn complexity_low_lowers_one_step() {
        let mut body = json!({"model": "gpt-5", "reasoning_effort": "high"});
        let changes = apply_decision(&mut body, "complexity", "low");
        assert_eq!(body[REASONING_EFFORT_KEY], json!("medium"));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].key, REASONING_EFFORT_KEY);
        assert_eq!(changes[0].from.as_deref(), Some("high"));
        assert_eq!(changes[0].to.as_deref(), Some("medium"));
        assert_eq!(changes[0].category, "complexity");

        let mut body = json!({"reasoning_effort": "max"});
        let changes = apply_decision(&mut body, "complexity", "low");
        assert_eq!(body[REASONING_EFFORT_KEY], json!("xhigh"));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].from.as_deref(), Some("max"));
        assert_eq!(changes[0].to.as_deref(), Some("xhigh"));
    }

    #[test]
    fn complexity_low_is_case_insensitive() {
        let mut body = json!({"reasoning_effort": "high"});
        let changes = apply_decision(&mut body, "complexity", "LOW");
        assert_eq!(body[REASONING_EFFORT_KEY], json!("medium"));
        assert_eq!(changes.len(), 1);
    }

    #[test]
    fn complexity_low_below_or_at_low_step_changes_nothing() {
        for effort in ["minimal", "low"] {
            let mut body = json!({"reasoning_effort": effort});
            let before = serde_json::to_string(&body).expect("serialize");
            assert!(apply_decision(&mut body, "complexity", "low").is_empty());
            assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
        }
    }

    #[test]
    fn complexity_high_raises_one_step_and_clamps() {
        let mut body = json!({"model": "gpt-5", "reasoning_effort": "low", "stream": true});
        let changes = apply_decision(&mut body, "complexity", "high");
        assert_eq!(body[REASONING_EFFORT_KEY], json!("medium"));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].from.as_deref(), Some("low"));
        assert_eq!(changes[0].to.as_deref(), Some("medium"));

        let mut body = json!({"model": "gpt-5", "reasoning_effort": "max", "stream": true});
        let before = serde_json::to_string(&body).expect("serialize");
        assert!(apply_decision(&mut body, "complexity", "high").is_empty());
        assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
    }

    #[test]
    fn complexity_with_unsupported_value_or_effort_changes_nothing() {
        for value in ["minimal", "medium", "xhigh", "max", "extreme", ""] {
            let mut body = json!({"reasoning_effort": "high"});
            let before = serde_json::to_string(&body).expect("serialize");
            assert!(apply_decision(&mut body, "complexity", value).is_empty());
            assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
        }

        for effort in [json!("extreme"), json!(3), json!(null), json!(["high"])] {
            let mut body = json!({"reasoning_effort": effort.clone()});
            let before = serde_json::to_string(&body).expect("serialize");
            assert!(apply_decision(&mut body, "complexity", "high").is_empty());
            assert!(apply_decision(&mut body, "complexity", "low").is_empty());
            assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
        }
    }

    #[test]
    fn complexity_with_absent_reasoning_effort_leaves_body_unchanged() {
        let mut body = json!({"model": "gpt-5", "messages": [], "stream": true});
        let before = serde_json::to_string(&body).expect("serialize");
        assert!(apply_decision(&mut body, "complexity", "low").is_empty());
        assert!(apply_decision(&mut body, "complexity", "high").is_empty());
        assert_eq!(serde_json::to_string(&body).expect("serialize"), before);
    }

    #[test]
    fn from_render_is_truncated_to_120_bytes() {
        let long_name = "a".repeat(400);
        let mut body = json!({
            "tools": [{"type": "function", "function": {"name": long_name}}],
            "tool_choice": "auto"
        });
        let changes = apply_decision(&mut body, "tool_requirement", "none");
        assert_eq!(changes.len(), 2);

        let rendered = changes[0].from.as_deref().expect("from rendered");
        assert_eq!(rendered.len(), MAX_RENDER_BYTES + TRUNCATION_MARKER.len());
        assert!(rendered.ends_with(TRUNCATION_MARKER));
        assert!(rendered.starts_with('['));
        assert!(body.get(TOOLS_KEY).is_none());
    }

    #[test]
    fn truncation_respects_utf8_boundaries() {
        // 3-byte characters straddle the 120-byte cut point.
        let text: String = std::iter::repeat('\u{20ac}').take(80).collect();
        let truncated = truncate_bytes(text.clone(), MAX_RENDER_BYTES);
        assert!(truncated.ends_with(TRUNCATION_MARKER));
        assert!(truncated.len() <= MAX_RENDER_BYTES + TRUNCATION_MARKER.len());
        assert!(truncated.is_char_boundary(0));
        assert_eq!(truncate_bytes(String::from("short"), MAX_RENDER_BYTES), "short");
    }

    #[test]
    fn short_from_render_is_not_truncated() {
        let mut body = json!({"tools": [{"type": "function"}], "tool_choice": "auto"});
        let changes = apply_decision(&mut body, "tool_requirement", "none");
        assert_eq!(changes[0].from.as_deref(), Some("[{\"type\":\"function\"}]"));
    }
}
