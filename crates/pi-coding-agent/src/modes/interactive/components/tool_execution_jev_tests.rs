use super::*;
use pi_tui::utils::{strip_ansi, visible_width};
use serde_json::json;

fn setup() {
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
}

fn arguments() -> Value {
    json!({
        "state": "REQUEST_CONTEXT",
        "questions": {"coin": {"type":"choice", "instructions":"Flip a coin",
            "criteria":{"heads":"Heads", "tails":"Tails"}}},
        "sample":["coin"]
    })
}

fn result() -> ToolExecutionResult {
    let value = json!({"category":"dynamic", "model":"jev-fixture", "answers":{
        "coin":{"choice":"heads", "probabilities":{"heads":0.6,"tails":0.4}, "confidence":0.2}
    }, "sampled":{"coin":{"choice":"tails","method":"jev_distribution"}},
        "usage":{"input_tokens":312,"output_tokens":18}, "latency_ms":12, "attempts":1});
    ToolExecutionResult {
        content: vec![ResultContentBlock::from_text(
            &serde_json::to_string_pretty(&value).unwrap(),
        )],
        details: Some(value),
        ..Default::default()
    }
}

fn component(definition: Option<ToolExecutionDefinition>) -> ToolExecutionComponent {
    ToolExecutionComponent::new(
        "jev_decide",
        "fixture",
        arguments(),
        ToolExecutionOptions::default(),
        definition,
        "/synthetic",
    )
}

#[test]
fn jev_json_is_folded_while_pending_streaming_and_complete_and_retained_when_expanded() {
    setup();
    for definition in [None, Some(ToolExecutionDefinition::default())] {
        let mut tool = component(definition);
        assert!(strip_ansi(&tool.render_lines(120.0).join("\n")).contains("queued"));
        tool.mark_execution_started();
        tool.update_args(arguments());
        assert!(strip_ansi(&tool.render_lines(120.0).join("\n")).contains("running"));
        for partial in [true, false] {
            tool.update_result(result(), partial);
            for width in [24, 80, 160] {
                let lines = tool.render_lines(width as f64);
                assert_eq!(lines.len(), 1);
                assert!(lines.iter().all(|line| visible_width(line) <= width));
                assert!(!strip_ansi(&lines.join("\n")).contains("REQUEST_CONTEXT"));
            }
        }
        let collapsed = strip_ansi(&tool.render_lines(120.0).join("\n"));
        assert!(collapsed.contains("Jev decision · done · 1 question"));
        assert!(collapsed.contains("Ctrl+O to expand"));
        tool.set_expanded(true);
        let expanded = strip_ansi(&tool.render_lines(120.0).join("\n"));
        for text in [
            "REQUEST_CONTEXT",
            "probabilities",
            "jev_distribution",
            "input_tokens",
            "attempts",
        ] {
            assert!(expanded.contains(text), "missing {text}");
        }
        assert!(expanded.contains("Ctrl+O to collapse"));
        assert_eq!(tool.args, arguments());
        assert_eq!(
            tool.result.as_ref().unwrap().content[0].text,
            result().content[0].text
        );
        assert_eq!(tool.result.as_ref().unwrap().details, result().details);
        tool.set_expanded(false);
        assert_eq!(tool.render_lines(120.0).len(), 1);
    }
}

#[test]
fn jev_error_summary_stays_visible_and_expansion_keeps_all_details() {
    setup();
    let mut tool = component(None);
    tool.update_result(
        ToolExecutionResult {
            content: vec![ResultContentBlock::from_text(
                "Jev Dynamic cancelled.\nERROR_DIAGNOSTIC",
            )],
            is_error: true,
            ..Default::default()
        },
        false,
    );
    let collapsed = strip_ansi(&tool.render_lines(120.0).join("\n"));
    assert!(collapsed.contains("error"));
    assert!(collapsed.contains("Jev Dynamic cancelled."));
    assert!(!collapsed.contains("ERROR_DIAGNOSTIC"));
    assert!(!collapsed.contains("REQUEST_CONTEXT"));
    tool.set_expanded(true);
    let expanded = strip_ansi(&tool.render_lines(120.0).join("\n"));
    assert!(expanded.contains("ERROR_DIAGNOSTIC"));
    assert!(expanded.contains("REQUEST_CONTEXT"));
}

#[test]
fn jev_hint_respects_custom_disabled_and_hidden_bindings() {
    setup();
    let mut bindings = pi_tui::keybindings::get_keybindings();
    bindings.set_user_bindings(indexmap::IndexMap::from([(
        "app.tools.expand".into(),
        vec!["ctrl+y".into()],
    )]));
    pi_tui::keybindings::set_keybindings(bindings.clone());
    let mut tool = component(None);
    assert!(strip_ansi(&tool.render_lines(120.0).join("\n")).contains("Ctrl+Y to expand"));
    tool.set_show_expand_hint(false);
    assert!(!strip_ansi(&tool.render_lines(120.0).join("\n")).contains("to expand"));
    tool.set_show_expand_hint(true);
    bindings.set_user_bindings(indexmap::IndexMap::from([(
        "app.tools.expand".into(),
        vec![],
    )]));
    pi_tui::keybindings::set_keybindings(bindings);
    tool.update_args(arguments());
    assert!(!strip_ansi(&tool.render_lines(120.0).join("\n")).contains("to expand"));
    setup();
}

#[test]
fn jev_folding_does_not_change_other_generic_tools() {
    setup();
    let mut tool = ToolExecutionComponent::new(
        "another_tool",
        "fixture",
        arguments(),
        ToolExecutionOptions::default(),
        None,
        "/synthetic",
    );
    tool.update_result(result(), false);
    let text = strip_ansi(&tool.render_lines(120.0).join("\n"));
    assert!(text.contains("REQUEST_CONTEXT"));
    assert!(text.contains("probabilities"));
}
