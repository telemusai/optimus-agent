//! Port of packages/coding-agent/src/modes/interactive/components/context-tree-format.ts

use pi_ai::types::Usage;
use pi_tui::utils::{truncate_to_width, visible_width, wrap_text_with_ansi};

use crate::core::context_tree::{ContextTreeNode, ContextUsage};
use crate::core::occ::textmas::TextMasUsageSummary;
use crate::core::usage::{add_assistant_usage, empty_usage};
use crate::modes::interactive::agent_activity::format_token_count;
use crate::modes::interactive::theme::theme::theme;

const CONTEXT_BAR_WIDTH: usize = 10;
const MIN_LABEL_WIDTH: usize = 16;

/// `ContextTreeRow`
struct ContextTreeRow {
    node_index: usize,
    /// Tree-drawing prefix, e.g. "│  └─ ".
    prefix: String,
}

/// Port of `statusIcon`.
fn status_icon(status: &str) -> String {
    match status {
        "active" => theme().fg("accent", "\u{25cf}"),
        "queued" => theme().fg("dim", "\u{25c7}"),
        "running" => theme().fg("accent", "\u{25c6}"),
        "done" => theme().fg("success", "\u{2713}"),
        "error" | "cancelled" => theme().fg("error", "\u{2717}"),
        // The TypeScript's `default` branch is an exhaustive-never check; an
        // unknown status cannot come from the ported union, so it renders empty.
        _ => String::new(),
    }
}

/// Port of `flattenContextTree`. The port records each row's node index instead
/// of the node itself, so the borrow of the tree stays immutable.
fn flatten_context_tree(root: &ContextTreeNode) -> Vec<ContextTreeRow> {
    let mut rows: Vec<ContextTreeRow> = vec![ContextTreeRow {
        node_index: 0,
        prefix: String::new(),
    }];
    let mut index = 1usize;
    fn walk(
        children: &[ContextTreeNode],
        ancestors: &str,
        rows: &mut Vec<ContextTreeRow>,
        index: &mut usize,
    ) {
        for (child_index, child) in children.iter().enumerate() {
            let is_last = child_index == children.len() - 1;
            rows.push(ContextTreeRow {
                node_index: *index,
                prefix: format!(
                    "{ancestors}{}",
                    if is_last {
                        "\u{2514}\u{2500} "
                    } else {
                        "\u{251c}\u{2500} "
                    }
                ),
            });
            *index += 1;
            walk(
                &child.children,
                &format!("{ancestors}{}", if is_last { "   " } else { "\u{2502}  " }),
                rows,
                index,
            );
        }
    }
    walk(&root.children, "", &mut rows, &mut index);
    rows
}

/// Node lookup by the pre-order index produced by [`flatten_context_tree`].
fn node_at(root: &ContextTreeNode, index: usize) -> &ContextTreeNode {
    let mut current_index = 0usize;
    fn walk<'a>(
        node: &'a ContextTreeNode,
        index: usize,
        current: &mut usize,
    ) -> Option<&'a ContextTreeNode> {
        if *current == index {
            return Some(node);
        }
        *current += 1;
        for child in &node.children {
            if let Some(found) = walk(child, index, current) {
                return Some(found);
            }
        }
        None
    }
    walk(root, index, &mut current_index).unwrap_or(root)
}

/// Spend-relevant token count, matching the "Total" line of /usage.
fn spent_tokens(usage: &Usage) -> f64 {
    usage.input + usage.output + usage.cache_read + usage.cache_write
}

/// Port of `formatCost`.
fn format_cost(cost: f64) -> String {
    format!("${:.2}", cost)
}

/// This is each node's own overhead category, already included in its usage totals.
fn format_textmas_usage(summary: &TextMasUsageSummary, prefix: &str, width: usize) -> Vec<String> {
    if width == 0 { return Vec::new(); }
    let indent = " ".repeat((visible_width(prefix) + 2).min(width.saturating_sub(1)));
    let available = width - indent.len();
    let mut details = vec![theme().fg("dim", &format!(
        "OCC3.1 TextMAS overhead: {} calls · known subtotal {} tokens / ${:.4} (included)",
        summary.calls, format_token_count(summary.known_tokens),
        summary.known_subtotal.cost.total,
    ))];
    if summary.unknown_usage_calls > 0 || summary.unknown_cost_calls > 0 {
        details.push(theme().fg("warning", &format!(
            "Unknown usage: {} calls · unknown cost: {} calls",
            summary.unknown_usage_calls, summary.unknown_cost_calls,
        )));
    }
    details.into_iter().flat_map(|detail| wrap_text_with_ansi(&detail, available))
        .map(|line| format!("{indent}{line}")).collect()
}

/// Port of `formatContextColumn`.
fn format_context_column(context_usage: Option<&ContextUsage>, with_bar: bool) -> String {
    let Some(context_usage) = context_usage else {
        return theme().fg("dim", "-");
    };
    if context_usage.tokens.is_none() || context_usage.percent.is_none() {
        return theme().fg("dim", "unknown after compaction");
    }
    let percent = js_round(context_usage.percent.unwrap_or(0.0));
    let detail = format!(
        "{}/{}",
        format_token_count(context_usage.tokens.unwrap_or(0.0)),
        format_token_count(context_usage.context_window)
    );
    let text = format!("{percent}% {}", theme().fg("dim", &format!("({detail})")));
    if !with_bar {
        return text;
    }
    let percent_value = context_usage.percent.unwrap_or(0.0);
    let filled = std::cmp::max(
        0,
        std::cmp::min(
            CONTEXT_BAR_WIDTH as i64,
            js_round((percent_value / 100.0) * CONTEXT_BAR_WIDTH as f64),
        ),
    ) as usize;
    let bar_color = if percent_value >= 80.0 {
        "warning"
    } else {
        "accent"
    };
    let bar = format!(
        "{}{}",
        theme().fg(bar_color, &"\u{2593}".repeat(filled)),
        theme().fg("dim", &"\u{2591}".repeat(CONTEXT_BAR_WIDTH - filled))
    );
    format!("{bar} {text}")
}

/// `Math.round` for a finite JavaScript number.
fn js_round(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    (value + 0.5).floor() as i64
}

/// Port of `padEndAnsi`.
fn pad_end_ansi(text: &str, width: usize) -> String {
    format!(
        "{text}{}",
        " ".repeat(width.saturating_sub(visible_width(text)))
    )
}

/// Port of `padStartAnsi`.
fn pad_start_ansi(text: &str, width: usize) -> String {
    format!(
        "{}{text}",
        " ".repeat(width.saturating_sub(visible_width(text)))
    )
}

/// Port of `countNodes`.
fn count_nodes(root: &ContextTreeNode) -> usize {
    1 + root.children.iter().map(count_nodes).sum::<usize>()
}

/// Own usage summed over the whole tree: exact even while children are mid-run.
fn sum_own_usage(root: &ContextTreeNode) -> Usage {
    let mut total = empty_usage();
    fn walk(node: &ContextTreeNode, total: &mut Usage) {
        add_assistant_usage(total, &node.own_usage);
        for child in &node.children {
            walk(child, total);
        }
    }
    walk(root, &mut total);
    total
}

/// `Number.prototype.toLocaleString()` with the default locale, matching the
/// grouping the TypeScript prints in the Tokens/Cost sections.
fn to_locale_string(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    let rounded = value.round();
    let negative = rounded < 0.0;
    let digits = format!("{}", rounded.abs() as i64);
    let mut grouped = String::new();
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(character);
    }
    if negative {
        format!("-{grouped}")
    } else {
        grouped
    }
}

/// Render the /context overview: a tree with one row per agent showing its own
/// tokens and cost (descendants excluded, so columns add up) plus per-agent
/// context-window utilization, followed by grand totals.
pub fn format_context_tree(root: &ContextTreeNode, width: f64) -> String {
    let rows = flatten_context_tree(root);
    let nodes: Vec<&ContextTreeNode> = rows
        .iter()
        .map(|row| node_at(root, row.node_index))
        .collect();

    let token_cells: Vec<String> = nodes
        .iter()
        .map(|node| format_token_count(spent_tokens(&node.own_usage)))
        .collect();
    let cost_cells: Vec<String> = nodes
        .iter()
        .map(|node| format_cost(node.own_usage.cost.total))
        .collect();
    let token_header = "tokens";
    let cost_header = "cost";
    let context_header = "context";
    let token_width = std::cmp::max(
        token_header.len(),
        token_cells.iter().map(|cell| cell.len()).max().unwrap_or(0),
    );
    let cost_width = std::cmp::max(
        cost_header.len(),
        cost_cells.iter().map(|cell| cell.len()).max().unwrap_or(0),
    );
    // `width - tokenWidth - costWidth - 28` can go negative in JavaScript; the
    // outer `Math.max(MIN_LABEL_WIDTH, ...)` still floors the result at 16.
    let widest_label = rows
        .iter()
        .zip(nodes.iter())
        .map(|(row, node)| row.prefix.encode_utf16().count() + 2 + visible_width(&node.label))
        .max()
        .unwrap_or(0);
    let label_width = std::cmp::max(
        MIN_LABEL_WIDTH,
        std::cmp::min(
            widest_label,
            (width.max(0.0) as i64 - token_width as i64 - cost_width as i64 - 28).max(0) as usize,
        ),
    );

    let mut lines: Vec<String> = Vec::new();
    lines.push(theme().bold("Context"));
    lines.push(String::new());
    if let Some(model) = &root.model {
        lines.push(format!(
            "{} {}/{}",
            theme().fg("dim", "Model:"),
            model.provider,
            model.id
        ));
        lines.push(String::new());
    }

    lines.push(theme().fg(
        "dim",
        &format!(
            "{}{}  {}  {}  {}",
            " ".repeat(2),
            pad_end_ansi("agent", label_width),
            pad_start_ansi(token_header, token_width),
            pad_start_ansi(cost_header, cost_width),
            context_header
        ),
    ));

    for (index, row) in rows.iter().enumerate() {
        let node = nodes[index];
        let label_space = std::cmp::max(
            1,
            label_width
                .saturating_sub(row.prefix.encode_utf16().count())
                .saturating_sub(2),
        );
        let label = truncate_to_width(&node.label, label_space as f64, "...", false);
        let label_cell = pad_end_ansi(
            &format!(
                "{}{} {label}",
                theme().fg("dim", &row.prefix),
                status_icon(&node.status)
            ),
            label_width + 2,
        );
        let token_cell = pad_start_ansi(&token_cells[index], token_width);
        let cost_cell = pad_start_ansi(&theme().fg("dim", &cost_cells[index]), cost_width);
        let context_cell = format_context_column(node.context_usage.as_ref(), node.id == "root");
        lines.push(format!(
            "{label_cell}  {token_cell}  {cost_cell}  {context_cell}"
        ));
        if let Some(summary) = &node.textmas_usage {
            lines.extend(format_textmas_usage(summary, &row.prefix, width.max(0.0) as usize));
        }
    }

    let totals = sum_own_usage(root);
    let agent_count = count_nodes(root);
    lines.push(String::new());
    lines.push(format!(
        "{} {} tokens {} {}{}",
        theme().fg("dim", "Total:"),
        format_token_count(spent_tokens(&totals)),
        theme().fg("dim", "\u{b7}"),
        format_cost(totals.cost.total),
        if agent_count > 1 {
            theme().fg("dim", &format!(" across {agent_count} agents"))
        } else {
            String::new()
        }
    ));

    lines.push(String::new());
    lines.push(theme().bold("Tokens"));
    lines.push(format!(
        "{} {}",
        theme().fg("dim", "Input:"),
        to_locale_string(totals.input)
    ));
    lines.push(format!(
        "{} {}",
        theme().fg("dim", "Output:"),
        to_locale_string(totals.output)
    ));
    if totals.cache_read > 0.0 {
        lines.push(format!(
            "{} {}",
            theme().fg("dim", "Cache Read:"),
            to_locale_string(totals.cache_read)
        ));
    }
    if totals.cache_write > 0.0 {
        lines.push(format!(
            "{} {}",
            theme().fg("dim", "Cache Write:"),
            to_locale_string(totals.cache_write)
        ));
    }
    lines.push(format!(
        "{} {}",
        theme().fg("dim", "Total:"),
        to_locale_string(spent_tokens(&totals))
    ));

    if totals.cost.total > 0.0 {
        lines.push(String::new());
        lines.push(theme().bold("Cost"));
        lines.push(format!(
            "{} ${:.4}",
            theme().fg("dim", "Total:"),
            totals.cost.total
        ));
    }

    if let Some(root_context) = root.context_usage.as_ref() {
        lines.push(String::new());
        lines.push(theme().bold("Context"));
        if root_context.tokens.is_none() || root_context.percent.is_none() {
            lines.push(format!(
                "{} unknown after compaction",
                theme().fg("dim", "Current:")
            ));
        } else {
            // `Math.round(percent * 10) / 10` keeps one decimal place.
            let percent_value = root_context.percent.unwrap_or(0.0);
            let percent = format!("{}", js_round(percent_value * 10.0) as f64 / 10.0);
            lines.push(format!(
                "{} {} / {} ({percent}%)",
                theme().fg("dim", "Current:"),
                to_locale_string(root_context.tokens.unwrap_or(0.0)),
                to_locale_string(root_context.context_window)
            ));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context_tree::ContextTreeModel;
    use crate::modes::interactive::theme::theme::init_theme;
    use pi_ai::types::UsageCost;

    fn init() {
        init_theme(Some("prime"), false);
    }

    fn usage(input: f64, output: f64, cost: f64) -> Usage {
        Usage {
            input,
            output,
            cache_read: 0.0,
            cache_write: 0.0,
            total_tokens: input + output,
            cost: UsageCost {
                total: cost,
                ..Default::default()
            },
        }
    }

    fn node(id: &str, label: &str, status: &str) -> ContextTreeNode {
        ContextTreeNode {
            id: id.to_string(),
            label: label.to_string(),
            status: status.to_string(),
            model: None,
            own_usage: usage(1000.0, 500.0, 0.5),
            total_usage: usage(1000.0, 500.0, 0.5),
            context_usage: None,
            textmas_usage: None,
            children: Vec::new(),
        }
    }

    fn strip_ansi(text: &str) -> String {
        let mut result = String::new();
        let mut chars = text.chars().peekable();
        while let Some(character) = chars.next() {
            if character == '\u{1b}' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                result.push(character);
            }
        }
        result
    }

    #[test]
    fn renders_header_rows_and_totals() {
        init();
        let mut root = node("root", "Session", "running");
        root.model = Some(ContextTreeModel {
            provider: "anthropic".to_string(),
            id: "claude".to_string(),
        });
        root.children = vec![node("sub-1", "Child", "done")];
        let text = strip_ansi(&format_context_tree(&root, 100.0));
        assert!(text.starts_with("Context\n"));
        assert!(text.contains("Model: anthropic/claude"));
        assert!(text.contains("agent"));
        assert!(text.contains("tokens"));
        assert!(text.contains("cost"));
        assert!(text.contains("context"));
        assert!(text.contains("Session"));
        assert!(text.contains("Child"));
        assert!(text.contains("\u{2514}\u{2500} \u{2713} Child"));
        assert!(text.contains("across 2 agents"));
        assert!(text.contains("Tokens"));
        assert!(text.contains("Input: 2,000"));
        assert!(text.contains("Output: 1,000"));
        assert!(text.contains("Total: 3,000"));
        assert!(text.contains("Cost"));
        assert!(text.contains("$1.0000"));
    }

    #[test]
    fn context_column_shows_unknown_after_compaction() {
        init();
        let mut root = node("root", "Session", "done");
        root.context_usage = Some(ContextUsage {
            tokens: None,
            context_window: 100000.0,
            percent: None,
        });
        let text = strip_ansi(&format_context_tree(&root, 100.0));
        assert!(text.contains("unknown after compaction"));
    }

    #[test]
    fn context_column_bar_only_renders_for_the_root() {
        init();
        let mut root = node("root", "Session", "done");
        root.context_usage = Some(ContextUsage {
            tokens: Some(50000.0),
            context_window: 100000.0,
            percent: Some(50.0),
        });
        root.children = vec![{
            let mut child = node("sub-1", "Child", "running");
            child.context_usage = Some(ContextUsage {
                tokens: Some(25000.0),
                context_window: 100000.0,
                percent: Some(25.0),
            });
            child
        }];
        let text = strip_ansi(&format_context_tree(&root, 100.0));
        // The root row carries the bar; the child row does not.
        assert_eq!(text.matches('\u{2593}').count(), 5);
        assert!(text.contains("50% (50k/100k)"));
        assert!(text.contains("25% (25k/100k)"));
    }

    #[test]
    fn single_agent_omits_the_across_suffix() {
        init();
        let root = node("root", "Session", "done");
        let text = strip_ansi(&format_context_tree(&root, 100.0));
        assert!(!text.contains("across"));
    }

    fn textmas(calls: usize, unknown_usage_calls: usize, unknown_cost_calls: usize) -> TextMasUsageSummary {
        TextMasUsageSummary {
            calls, unknown_usage_calls, unknown_cost_calls,
            known_tokens: 150.0,
            known_subtotal: usage(100.0, 50.0, 0.1),
        }
    }

    #[test]
    fn textmas_known_overhead_is_visible_and_not_added_to_totals_twice() {
        init();
        let mut root = node("root", "Session", "done");
        let legacy = strip_ansi(&format_context_tree(&root, 120.0));
        root.textmas_usage = Some(textmas(3, 0, 0));
        let text = strip_ansi(&format_context_tree(&root, 120.0));
        assert!(text.contains("OCC3.1 TextMAS overhead: 3 calls · known subtotal 150 tokens / $0.1000 (included)"));
        assert!(!text.contains("Unknown usage"));
        assert_eq!(text.lines().filter(|line| !line.contains("TextMAS")).collect::<Vec<_>>().join("\n"), legacy);
        assert!(text.contains("Total: 1,500") && text.contains("Total: $0.5000"));
    }

    #[test]
    fn textmas_unknown_usage_and_cost_are_explicit_even_with_zero_subtotal() {
        init();
        for (unknown_usage, unknown_cost) in [(1, 2), (0, 3), (3, 3)] {
            let mut root = node("root", "Session", "done");
            let mut summary = textmas(3, unknown_usage, unknown_cost);
            summary.known_subtotal = Usage::default();
            summary.known_tokens = 0.0;
            root.textmas_usage = Some(summary);
            let text = strip_ansi(&format_context_tree(&root, 120.0));
            assert!(text.contains("known subtotal 0 tokens / $0.0000 (included)"));
            assert!(text.contains(&format!("Unknown usage: {unknown_usage} calls · unknown cost: {unknown_cost} calls")));
        }
    }

    #[test]
    fn textmas_category_preserves_total_only_and_mixed_call_token_counts() {
        use crate::core::occ::textmas::{TextMasCallStatus, TextMasRole, TextMasUsageLedger, TextMasUsageRecord};
        init();
        let mut ledger = TextMasUsageLedger::default();
        let total_only = Usage { total_tokens: 80.0, ..Usage::default() };
        ledger.observe(TextMasUsageRecord::new("total-only", TextMasRole::Planner,
            TextMasCallStatus::Completed, Some(total_only)));
        let summary = ledger.summary().unwrap();
        let rows = format_textmas_usage(&summary, "", 120);
        assert!(strip_ansi(&rows.join("\n")).contains("known subtotal 80 tokens"));
        let component_only = Usage { input: 100.0, output: 40.0, ..Usage::default() };
        ledger.observe(TextMasUsageRecord::new("component-only", TextMasRole::Critic,
            TextMasCallStatus::Completed, Some(component_only)));
        let summary = ledger.summary().unwrap();
        let rows = format_textmas_usage(&summary, "", 120);
        let text = strip_ansi(&rows.join("\n"));
        assert!(text.contains("2 calls · known subtotal 220 tokens"), "{text}");
        assert!(text.contains("unknown cost: 2 calls"), "{text}");
    }

    #[test]
    fn textmas_nested_categories_stay_own_and_legacy_absence_changes_nothing() {
        init();
        let mut root = node("root", "Session", "done");
        let mut child = node("child", "Child", "done");
        child.children.push(node("nested", "Nested", "running"));
        root.children.push(child);
        let legacy = format_context_tree(&root, 120.0);
        let mut old_wire = serde_json::to_value(&root).unwrap();
        old_wire.as_object_mut().unwrap().remove("textmasUsage");
        let restored: ContextTreeNode = serde_json::from_value(old_wire).unwrap();
        assert_eq!(format_context_tree(&restored, 120.0), legacy);
        root.textmas_usage = Some(textmas(3, 0, 0));
        root.children[0].textmas_usage = Some(textmas(6, 1, 2));
        let text = strip_ansi(&format_context_tree(&root, 120.0));
        assert_eq!(text.matches("TextMAS overhead:").count(), 2);
        assert!(text.contains("TextMAS overhead: 3 calls") && text.contains("TextMAS overhead: 6 calls"));
        assert!(!text.contains("TextMAS overhead: 9 calls"));
        let overheads = text.lines().filter(|line| line.contains("TextMAS")).collect::<Vec<_>>();
        assert!(overheads[0].starts_with("  OCC3.1"));
        assert!(overheads[1].starts_with("     OCC3.1"));
        assert!(text.contains("across 3 agents") && text.contains("Total: 4,500"));
        assert!(text.contains("Total: $1.5000"));
        let without_categories = text.lines()
            .filter(|line| !line.contains("TextMAS") && !line.contains("Unknown usage"))
            .collect::<Vec<_>>().join("\n");
        assert_eq!(without_categories, strip_ansi(&legacy));
    }

    #[test]
    fn textmas_rows_wrap_within_width_without_hiding_unknown_counts() {
        init();
        let summary = textmas(123, 12, 34);
        for width in [0, 1, 2, 8, 20, 40, 80, 110, 160] {
            for prefix in ["", "└─ ", "│  └─ "] {
                let rows = format_textmas_usage(&summary, prefix, width);
                assert!(rows.iter().all(|line| visible_width(line) <= width), "{width}: {rows:?}");
                if width >= 20 {
                    let text = strip_ansi(&rows.join(" "));
                    let words = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    assert!(words.contains("OCC3.1 TextMAS overhead: 123 calls"), "{width}: {words}");
                    assert!(words.contains("Unknown usage: 12 calls") && words.contains("unknown cost: 34 calls"), "{width}: {words}");
                }
            }
        }
    }

    #[test]
    fn locale_string_groups_thousands() {
        assert_eq!(to_locale_string(999.0), "999");
        assert_eq!(to_locale_string(1000.0), "1,000");
        assert_eq!(to_locale_string(1234567.0), "1,234,567");
    }

    #[test]
    fn label_width_floors_at_sixteen_columns() {
        init();
        let root = node("root", "S", "done");
        let text = strip_ansi(&format_context_tree(&root, 200.0));
        let header = text
            .lines()
            .find(|line| line.contains("agent"))
            .expect("header")
            .to_string();
        assert!(header.starts_with("  agent"));
    }
}
