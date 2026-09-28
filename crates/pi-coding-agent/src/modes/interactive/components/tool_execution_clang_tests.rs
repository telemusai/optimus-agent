use super::*;
use pi_tui::utils::{strip_ansi, visible_width};
use serde_json::json;

#[test]
fn clang_cells_fold_source_output_and_diagnostics_without_changing_results() {
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let code = "#include <cstdio>\nint hidden_cpp_source = 42;\n";
    for details in [None, Some(Value::Null), Some(json!({"language":"cpp","cwd":"/project"}))] {
        let mut cell = ToolExecutionComponent::new(
            "clang", "cpp-cell", json!({"code":code}),
            ToolExecutionOptions::default(), None, "/project",
        );
        let queued = strip_ansi(&cell.render_lines(160.0).join("\n"));
        assert!(queued.contains("cpp"), "{queued}");
        assert!(queued.contains("#include <cstdio>"), "{queued}");
        assert!(queued.contains("to expand"), "{queued}");
        assert!(!queued.contains("hidden_cpp_source"));
        assert!(!queued.contains("\"code\""));
        cell.mark_execution_started();
        cell.set_args_complete();
        for (partial, error) in [(true, false), (false, false), (false, true)] {
            let output = if error { "fixture.cpp:2:1: error: CPP_DIAGNOSTIC\nfull compiler context" }
                else { "CPP_OUTPUT\nsecond output line" };
            cell.update_result(ToolExecutionResult {
                content: vec![ResultContentBlock::from_text(output)],
                details: details.clone(), is_error: error,
            }, partial);
            let collapsed = strip_ansi(&cell.render_lines(160.0).join("\n"));
            assert!(collapsed.contains("cpp"), "{collapsed}");
            assert!(!collapsed.contains("hidden_cpp_source"));
            assert!(!collapsed.contains("CPP_OUTPUT"));
            assert!(!collapsed.contains("CPP_DIAGNOSTIC"));
            if error { assert!(collapsed.contains('✗'), "{collapsed}"); }
            cell.set_expanded(true);
            let expanded = strip_ansi(&cell.render_lines(160.0).join("\n"));
            assert!(expanded.contains("hidden_cpp_source"), "{expanded}");
            for line in output.lines() { assert!(expanded.contains(line), "{expanded}"); }
            assert!(expanded.contains("to collapse"), "{expanded}");
            assert!(!expanded.contains("\"code\""));
            for width in [1, 20, 80] {
                assert!(cell.render_lines(width as f64).iter().all(|line| visible_width(line) <= width));
            }
            cell.set_expanded(false);
            assert_eq!(cell.result.as_ref().unwrap().details, details);
            assert_eq!(cell.result.as_ref().unwrap().content[0].text.as_deref(), Some(output));
        }
    }
}

#[test]
fn clang_custom_renderers_keep_their_existing_shell() {
    let cell = ToolExecutionComponent::new(
        "clang", "custom-cpp", json!({"code":"int n = 1;"}),
        ToolExecutionOptions::default(), Some(ToolExecutionDefinition {
            has_render_call: true, ..Default::default()
        }), "/project",
    );
    assert!(!cell.should_use_ipython_renderer());
    assert!(cell.ipython_cell_component.is_none());
}
