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


/// Optional catalog pruning is independent of the legacy whole-catalog switch.
/// An empty optional allowlist intentionally preserves every tool.
pub struct PreparedToolPruning {
    pub state: Value,
    names: Vec<String>,
    catalog_fingerprint: String,
    options: pi_jev::filtering::FilteringOptions,
}

impl PreparedToolPruning {
    pub fn questions(&self) -> Vec<pi_jev::evaluators::PreparedQuestion> {
        pi_jev::filtering::candidate_questions(&self.state, pi_jev::types::DecisionCategory::ToolCandidates, "optional_tools")
            .unwrap_or_default()
    }

    pub fn apply(
        &self,
        params: &mut Value,
        decisions: &[pi_jev::active::ActiveDecision],
        request_id: &str,
        turn: u64,
    ) -> Vec<AppliedChange> {
        if !pruning_choice_is_auto(params) { return Vec::new(); }
        let Some(tools) = params.get(TOOLS_KEY) else { return Vec::new(); };
        if pi_jev::snapshot::fingerprint_of(tools) != self.catalog_fingerprint { return Vec::new(); }
        let dropped = pi_jev::filtering::dropped_candidate_indices_with_options(
            decisions, pi_jev::types::DecisionCategory::ToolCandidates, self.names.len(), request_id,
            turn, std::time::SystemTime::now(), &self.options,
        );
        let removed: Vec<&str> = dropped.iter().map(|index| self.names[*index].as_str()).collect();
        let Some(tools) = params.get_mut(TOOLS_KEY).and_then(Value::as_array_mut) else { return Vec::new(); };
        if removed.is_empty() || removed.len() >= tools.len() { return Vec::new(); }
        tools.retain(|tool| !tool_name_for_pruning(tool).is_some_and(|name| removed.contains(&name)));
        removed.into_iter().map(|name| AppliedChange {
            key: TOOLS_KEY.to_string(), from: Some(name.to_string()), to: None,
            category: "tool_candidates".to_string(),
        }).collect()
    }
}

fn pruning_choice_is_auto(params: &Value) -> bool {
    match params.get(TOOL_CHOICE_KEY) {
        None => true,
        Some(Value::String(choice)) => choice == "auto",
        Some(Value::Object(choice)) => choice.len() == 1 && choice.get("type").and_then(Value::as_str) == Some("auto"),
        _ => false,
    }
}

fn tool_name_for_pruning(tool: &Value) -> Option<&str> {
    let name = if tool.get("type").and_then(Value::as_str) == Some("function") {
        if tool.get("function").is_some() {
            tool.get("function")?.get("name")?.as_str()?
        } else {
            tool.get("name")?.as_str()?
        }
    } else if tool.get("type").is_none() && tool.get("input_schema").is_some_and(Value::is_object) {
        tool.get("name")?.as_str()?
    } else {
        return None;
    };
    if name.is_empty() || name.len() > 64
        || !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"_-.".contains(&byte)) {
        return None;
    }
    Some(name)
}

pub fn prepare_tool_pruning(
    params: &Value,
    query: &str,
    options: &pi_jev::filtering::FilteringOptions,
) -> PreparedToolPruning {
    let mut plan = PreparedToolPruning {
        state: serde_json::json!({
            "user_text_excerpt": pi_jev::redact::bounded_excerpt(query, pi_jev::filtering::MAX_FILTER_EXCERPT_CHARS),
            "optional_tools": [],
        }),
        names: Vec::new(), catalog_fingerprint: String::new(), options: options.clone(),
    };
    if options.validate().is_err() || !pruning_choice_is_auto(params) { return plan; }
    let Some(catalog) = params.get(TOOLS_KEY) else { return plan; };
    let Some(tools) = catalog.as_array().filter(|tools| tools.len() <= 128) else { return plan; };
    let mut seen = std::collections::BTreeSet::new();
    let mut candidates = Vec::new();
    for tool in tools {
        let Some(name) = tool_name_for_pruning(tool) else {
            // An unknown schema may carry provider-specific tool references.
            return plan;
        };
        if !seen.insert(name) { return plan; }
        let mandatory = name == "ipython" || name.starts_with("__") || name.starts_with("rlm")
            || name.starts_with("agent_") || options.mandatory_tool_names.iter().any(|item| item == name);
        if mandatory || !options.optional_tool_names.iter().any(|item| item == name)
            || candidates.len() >= options.max_candidates { continue; }
        let description = tool.get("function").unwrap_or(tool).get("description").and_then(Value::as_str).unwrap_or("");
        let excerpt = format!("{name}: {}", pi_jev::redact::bounded_excerpt(description, 160));
        candidates.push(serde_json::json!({"id":plan.names.len().to_string(), "excerpt":excerpt}));
        plan.names.push(name.to_string());
    }
    plan.state["optional_tools"] = Value::Array(candidates);
    plan.catalog_fingerprint = pi_jev::snapshot::fingerprint_of(catalog);
    plan
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


    fn pruning_options(names: &[&str]) -> pi_jev::filtering::FilteringOptions {
        pi_jev::filtering::FilteringOptions {
            optional_tool_names: names.iter().map(|name| name.to_string()).collect(),
            ..Default::default()
        }
    }

    fn pruning_decision(index: usize, value: &str) -> pi_jev::active::ActiveDecision {
        pi_jev::active::ActiveDecision {
            category: pi_jev::types::DecisionCategory::ToolCandidates,
            question_id: format!("tool_candidates.{index}"), value: value.to_string(),
            confidence: 0.99, response_model: None, request_id: "r1".to_string(),
            turn: 2, decided_at: std::time::SystemTime::now(),
        }
    }

    #[test]
    fn optional_pruning_supports_native_provider_schemas_and_preserves_authority() {
        for tools in [
            json!([{"type":"function","function":{"name":"ipython"}}, {"type":"function","function":{"name":"lookup"}}]),
            json!([{"type":"function","name":"ipython"}, {"type":"function","name":"lookup"}]),
            json!([{"name":"ipython","input_schema":{}}, {"name":"lookup","input_schema":{}}]),
        ] {
            let mut body = json!({"model":"unchanged","messages":[{"role":"user","content":"task"}],"tools":tools,"tool_choice":"auto","reasoning_effort":"high"});
            let original = body.clone();
            let plan = prepare_tool_pruning(&body,"task",&pruning_options(&["ipython","lookup"]));
            assert_eq!(plan.questions().len(), 1);
            let changes = plan.apply(&mut body,&[pruning_decision(0,"drop")],"r1",2);
            assert_eq!(changes.len(),1);
            assert_eq!(changes[0].from.as_deref(),Some("lookup"));
            assert_eq!(body["tools"].as_array().unwrap().len(),1);
            assert_eq!(tool_name_for_pruning(&body["tools"][0]),Some("ipython"));
            for key in ["model","messages","tool_choice","reasoning_effort"] { assert_eq!(body[key],original[key]); }
            assert_eq!(original["tools"].as_array().unwrap().len(),2);
        }
    }

    #[test]
    fn optional_pruning_pins_explicit_mandatory_internal_forced_and_unlisted_tools() {
        let options = pi_jev::filtering::FilteringOptions {
            mandatory_tool_names: vec!["guard".to_string()],
            ..pruning_options(&["ipython","guard","rlm_spawn","agent_message","__internal","lookup"])
        };
        let tools: Vec<Value> = ["ipython","guard","rlm_spawn","agent_message","__internal","unlisted","lookup"]
            .iter().map(|name|json!({"type":"function","name":name})).collect();
        let mut body = json!({"tools":tools});
        let plan = prepare_tool_pruning(&body,"task",&options);
        assert_eq!(plan.questions().len(),1);
        plan.apply(&mut body,&[pruning_decision(0,"drop")],"r1",2);
        assert_eq!(body["tools"].as_array().unwrap().len(),6);
        for choice in [json!("required"),json!({"type":"function","name":"lookup"}),json!({"type":"tool","name":"lookup"}),json!(null)] {
            let body = json!({"tools":tools,"tool_choice":choice});
            assert!(prepare_tool_pruning(&body,"task",&options).questions().is_empty());
        }
    }

    #[test]
    fn optional_pruning_fails_open_for_invalid_unknown_changed_or_empty_catalogs() {
        let options = pruning_options(&["lookup"]);
        for tools in [json!([{"type":"web_search"}]),json!([{"type":"function"}]),
            json!([{"type":"function","name":"lookup"},{"type":"function","name":"lookup"}])] {
            let mut body = json!({"tools":tools}); let original=body.clone();
            let plan=prepare_tool_pruning(&body,"task",&options);
            assert!(plan.questions().is_empty());
            assert!(plan.apply(&mut body,&[pruning_decision(0,"drop")],"r1",2).is_empty());
            assert_eq!(body,original);
        }
        let mut only = json!({"tools":[{"type":"function","name":"lookup"}]});
        let original=only.clone();
        let plan=prepare_tool_pruning(&only,"task",&options);
        assert!(plan.apply(&mut only,&[pruning_decision(0,"drop")],"r1",2).is_empty());
        assert_eq!(only,original);
        let mut body = json!({"tools":[{"type":"function","name":"ipython"},{"type":"function","name":"lookup"}]});
        let plan=prepare_tool_pruning(&body,"task",&options);
        body["tools"][1]["description"] = json!("changed"); let original=body.clone();
        assert!(plan.apply(&mut body,&[pruning_decision(0,"drop")],"r1",2).is_empty());
        assert_eq!(body,original);
    }

    #[test]
    fn short_from_render_is_not_truncated() {
        let mut body = json!({"tools": [{"type": "function"}], "tool_choice": "auto"});
        let changes = apply_decision(&mut body, "tool_requirement", "none");
        assert_eq!(changes[0].from.as_deref(), Some("[{\"type\":\"function\"}]"));
    }
}
