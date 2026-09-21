//! Pure bottom-row composition for the interactive tray.
//!
//! The tray's bottom row carries, on ONE baseline:
//!
//! ```text
//! agents/resume depth0 GPT-6 Astra xhigh  ● Jev On (Compare) ● Jev compact on      146k (14%)
//! |---- left (navigation/model/effort) ----|------ Jev segments ------| right counter
//! ```
//!
//! * The context usage counter is right-aligned on the SAME row as the left
//!   label, never on a separate lower-left line, and it is never hidden behind a
//!   full-width left string.
//! * The Jev decision segment and the independent compaction segment ride after
//!   the left label (the effort label ends the left group in the common case).
//! * Widths are measured with [`pi_tui::utils::visible_width`], which strips ANSI
//!   escapes and counts Unicode graphemes, so ANSI-coloured segments and wide
//!   glyphs cannot desynchronise the layout.
//!
//! Narrow-terminal fallback ladder (priority: counter, then left label, then
//! segment labels):
//!
//! 1. Everything at full length.
//! 2. Segments collapse to their compact form when the publisher provided one
//!    (`statusCompactText`); a segment without a compact form keeps its full
//!    text (old-daemon fallback).
//! 3. The left label is ellipsis-truncated to the remaining space. If even the
//!    compact segments cannot share the row with the counter, the segments drop
//!    first: the counter stays visible on the row, nothing wraps to a stray
//!    lower-left line.
//!
//! The function is pure: it only measures and concatenates its inputs, so a
//! render pass can never do I/O or an RPC.

use pi_tui::utils::{truncate_to_width, visible_width};

/// One pre-coloured segment with an optional narrow form.
///
/// `full` is the labelled form the publisher computed. `compact` is the optional
/// narrow form (`statusCompactText`); publishers that cannot provide one leave
/// `None`, and the row then keeps the full form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraySegment<'a> {
    pub full: &'a str,
    pub compact: Option<&'a str>,
}

impl<'a> TraySegment<'a> {
    pub fn new(full: &'a str) -> Self {
        Self {
            full,
            compact: None,
        }
    }

    pub fn with_compact(full: &'a str, compact: &'a str) -> Self {
        Self {
            full,
            compact: Some(compact),
        }
    }

    /// The form this segment contributes when the row must narrow.
    pub fn narrow(&self) -> &'a str {
        self.compact.unwrap_or(self.full)
    }
}

/// The inputs of one tray row. Strings are pre-coloured; this module never calls
/// the theme, so the composition stays pure and testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrayRow<'a> {
    /// Override or location label (navigation/model/effort), already dim-wrapped.
    pub left: &'a str,
    /// Jev decision segment (e.g. `● Jev On (Compare)`), if published.
    pub jev_decision: Option<TraySegment<'a>>,
    /// Independent Jev compaction segment (e.g. `● Jev compact on`), if published.
    pub jev_compaction: Option<TraySegment<'a>>,
    /// Context usage counter (e.g. `146k (14%)`), if known, already dim-wrapped.
    pub right: Option<&'a str>,
}

/// Separator between the left label and the Jev segments. Two spaces, matching
/// the location label's own part separator.
pub const TRAY_ROW_SEPARATOR: &str = "  ";

/// Compose one tray row for `width` columns.
///
/// The result is at most `width` visible columns wide. The right counter, when
/// present, is always rendered right-aligned on the same single line.
pub fn compose_tray_row(row: TrayRow, width: usize) -> String {
    let width = width.max(1);
    let right_width = row.right.map(|right| visible_width(right)).unwrap_or(0);
    let (full_segments, narrow_segments) = segment_forms(&row);

    // Ladder 1: full length everywhere. The fit check measures the UNTRUNCATED
    // row; `assemble`'s final width clamp is a safety net for inputs that cannot
    // fit at all, never a ladder step.
    if fits(row.left, &full_segments, row.right, right_width, width) {
        return assemble(row.left, &full_segments, row.right, width, None);
    }

    // Ladder 2: collapse the segments to their compact forms where available.
    if fits(row.left, &narrow_segments, row.right, right_width, width) {
        return assemble(row.left, &narrow_segments, row.right, width, None);
    }

    // Ladder 3: ellipsis-truncate the left label so the segments and the counter
    // keep their place on the same row. If even the compact segments cannot
    // share the row with the counter, the segments drop first: the counter is
    // never hidden and never moved to another line.
    let segments_width = visible_width(&join_parts(&narrow_segments));
    let separator_width = if segments_width > 0 && !row.left.is_empty() {
        visible_width(TRAY_ROW_SEPARATOR)
    } else {
        0
    };
    let counter_room = if row.right.is_some() {
        right_width + 1
    } else {
        0
    };
    if segments_width + separator_width + counter_room <= width {
        let left_budget = width - segments_width - separator_width - counter_room;
        return assemble(
            row.left,
            &narrow_segments,
            row.right,
            width,
            Some(left_budget),
        );
    }
    let left_budget = width.saturating_sub(counter_room);
    assemble(row.left, &[], row.right, width, Some(left_budget))
}

/// True when the left label, the segments and (optionally) the counter all fit
/// on one `width`-column row with at least one separator column before the
/// counter.
fn fits(
    left: &str,
    segments: &[&str],
    right: Option<&str>,
    right_width: usize,
    width: usize,
) -> bool {
    let mut parts: Vec<&str> = vec![left];
    parts.extend(segments.iter().copied());
    let mut needed = visible_width(&join_parts(&parts));
    if right.is_some() {
        needed += 1 + right_width;
    }
    needed <= width
}

fn segment_forms<'a>(row: &TrayRow<'a>) -> (Vec<&'a str>, Vec<&'a str>) {
    let mut full = Vec::new();
    let mut narrow = Vec::new();
    for segment in [row.jev_decision, row.jev_compaction].into_iter().flatten() {
        full.push(segment.full);
        narrow.push(segment.narrow());
    }
    (full, narrow)
}

/// Join non-empty parts with the two-space separator.
fn join_parts(parts: &[&str]) -> String {
    parts
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(TRAY_ROW_SEPARATOR)
}

/// Build the row: optional truncated left, segments, right-aligned counter.
///
/// `left_budget` of `None` keeps the left label at full length. The result is
/// clamped to `width` columns as a final guard, so no input combination can
/// overflow the line.
fn assemble(
    left: &str,
    segments: &[&str],
    right: Option<&str>,
    width: usize,
    left_budget: Option<usize>,
) -> String {
    let left_owned = match left_budget {
        Some(budget) => truncate_to_width(left, budget as f64, "…", false),
        None => left.to_string(),
    };
    let mut parts = vec![left_owned.as_str()];
    parts.extend(segments.iter().copied());
    let joined = join_parts(&parts);
    let row = match right {
        Some(right) => {
            let pad = width.saturating_sub(visible_width(&joined) + visible_width(right));
            format!("{joined}{}{right}", " ".repeat(pad))
        }
        None => joined,
    };
    if visible_width(&row) > width {
        truncate_to_width(&row, width as f64, "…", false)
    } else {
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GREEN: &str = "\u{1b}[38;2;74;222;128m";
    const RED: &str = "\u{1b}[38;2;248;113;113m";
    const RESET: &str = "\u{1b}[39m";
    const DIM: &str = "\u{1b}[38;2;113;113;122m";

    fn colored(color: &str, text: &str) -> String {
        format!("{color}{text}{RESET}")
    }

    fn segment<'a>(full: &'a str, compact: Option<&'a str>) -> Option<TraySegment<'a>> {
        Some(TraySegment { full, compact })
    }

    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' {
                for escape in chars.by_ref() {
                    if escape.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(ch);
            }
        }
        out
    }

    const LEFT: &str = "agents/resume depth0 GPT-6 Astra xhigh";
    const DECISION_ON: &str = "\u{25cf} Jev On (Compare)";
    const COMPACTION_ON: &str = "\u{25cf} Jev compact on";
    const USAGE: &str = "146k (14%)";

    #[test]
    fn counter_is_right_aligned_on_the_same_row_when_everything_fits() {
        let left = colored(DIM, LEFT);
        let decision_full = colored(GREEN, DECISION_ON);
        let decision_narrow = colored(GREEN, "\u{25cf}");
        let compaction_full = colored(GREEN, COMPACTION_ON);
        let compaction_narrow = colored(GREEN, "\u{25cf}");
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, Some(&decision_narrow)),
            jev_compaction: segment(&compaction_full, Some(&compaction_narrow)),
            right: Some(&usage),
        };
        let composed = compose_tray_row(row, 120);
        // One physical line, exact width, counter on the right edge.
        assert!(!composed.contains('\n'));
        let plain = strip_ansi(&composed);
        let pad = 120 - 38 - 2 - 18 - 2 - 16 - 10;
        let expected = format!(
            "{LEFT}  {DECISION_ON}  {COMPACTION_ON}{}{USAGE}",
            " ".repeat(pad)
        );
        assert_eq!(plain, expected);
        assert_eq!(visible_width(&plain), 120);
    }

    #[test]
    fn segments_follow_the_left_label_on_the_same_baseline() {
        let left = colored(DIM, "GPT-6 Astra xhigh");
        let decision_full = colored(GREEN, DECISION_ON);
        let usage = colored(DIM, "5k (1%)");
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, None),
            jev_compaction: None,
            right: Some(&usage),
        };
        let plain = strip_ansi(&compose_tray_row(row, 80));
        assert!(
            plain.starts_with("GPT-6 Astra xhigh  \u{25cf} Jev On (Compare)"),
            "{plain:?}"
        );
        assert!(plain.ends_with("5k (1%)"));
    }

    #[test]
    fn narrow_row_compacts_segments_before_touching_the_left_label() {
        let left = colored(DIM, LEFT);
        let decision_full = colored(GREEN, "\u{25cf} Jev On (Compare + Active)");
        let decision_narrow = colored(GREEN, "\u{25cf} Jev C+A On");
        let compaction_full = colored(GREEN, COMPACTION_ON);
        let compaction_narrow = colored(GREEN, "\u{25cf} Jev Cmp on");
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, Some(&decision_narrow)),
            jev_compaction: segment(&compaction_full, Some(&compaction_narrow)),
            right: Some(&usage),
        };
        // 44 columns: the full forms cannot fit, the SHORT LABELLED compact
        // forms plus a truncated left label can. The compact forms are never
        // bare dots: both identities and states stay readable in text.
        let plain = strip_ansi(&compose_tray_row(row, 44));
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert!(!plain.contains("Jev On"), "{plain:?}");
        assert!(!plain.contains("Jev compact"), "{plain:?}");
        assert!(plain.contains("Jev C+A On"), "{plain:?}");
        assert!(plain.contains("Jev Cmp on"), "{plain:?}");
        // Both dots survive and the left label is ellipsised, all on one line.
        assert_eq!(plain.matches('\u{25cf}').count(), 2, "{plain:?}");
        assert!(plain.contains('\u{2026}'), "{plain:?}");
        assert_eq!(visible_width(&plain), 44);
    }

    #[test]
    fn screenshot_widths_keep_the_model_effort_label_both_jev_states_and_the_counter() {
        // Regression for the real terminal widths reviewers screenshot at. The
        // left label is the ACTUAL model/effort tray label; both Jev segments
        // are enabled. At every width the row must show an identifiable Jev
        // decision mode, an identifiable compaction state and the counter on
        // the right edge, and the model/effort label must stay readable.
        let left = colored(DIM, LEFT);
        let decision_full = colored(GREEN, DECISION_ON);
        let decision_narrow = colored(GREEN, "\u{25cf} Jev C On");
        let compaction_full = colored(GREEN, COMPACTION_ON);
        let compaction_narrow = colored(GREEN, "\u{25cf} Jev Cmp on");
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, Some(&decision_narrow)),
            jev_compaction: segment(&compaction_full, Some(&compaction_narrow)),
            right: Some(&usage),
        };
        // 80 columns: full forms do not fit, the short labelled forms do, and
        // the left label is NOT truncated.
        let plain = strip_ansi(&compose_tray_row(row, 80));
        assert!(plain.contains("Jev C On"), "{plain:?}");
        assert!(plain.contains("Jev Cmp on"), "{plain:?}");
        assert!(
            !plain.contains('\u{2026}'),
            "model/effort stays visible: {plain:?}"
        );
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert_eq!(visible_width(&plain), 80);
        // 100 and 120 columns: the full labelled forms fit.
        for width in [100usize, 120] {
            let plain = strip_ansi(&compose_tray_row(row, width));
            assert!(plain.contains(DECISION_ON), "{width}: {plain:?}");
            assert!(plain.contains(COMPACTION_ON), "{width}: {plain:?}");
            assert!(!plain.contains('\u{2026}'), "{width}: {plain:?}");
            assert!(plain.ends_with(USAGE), "{width}: {plain:?}");
            assert_eq!(visible_width(&plain), width);
        }
    }

    #[test]
    fn screenshot_widths_keep_off_and_compaction_on_distinguishable() {
        // Decision OFF with compaction ON must stay readable as two DIFFERENT
        // states at ordinary widths, not two identical dots.
        let left = colored(DIM, LEFT);
        let decision_full = colored(RED, "\u{25cf} Jev Off");
        let decision_narrow = colored(RED, "\u{25cf} Jev Off");
        let compaction_full = colored(GREEN, COMPACTION_ON);
        let compaction_narrow = colored(GREEN, "\u{25cf} Jev Cmp on");
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, Some(&decision_narrow)),
            jev_compaction: segment(&compaction_full, Some(&compaction_narrow)),
            right: Some(&usage),
        };
        let plain = strip_ansi(&compose_tray_row(row, 80));
        assert!(plain.contains("Jev Off"), "{plain:?}");
        assert!(plain.contains(COMPACTION_ON), "{plain:?}");
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert!(!plain.contains('\u{2026}'), "{plain:?}");
        assert_eq!(visible_width(&plain), 80);
        let plain = strip_ansi(&compose_tray_row(row, 120));
        assert!(plain.contains("Jev Off"), "{plain:?}");
        assert!(plain.contains(COMPACTION_ON), "{plain:?}");
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert_eq!(visible_width(&plain), 120);
    }

    #[test]
    fn very_narrow_row_drops_segments_but_never_hides_the_counter() {
        let left = colored(DIM, "GPT-6 Astra xhigh");
        let decision_full = colored(GREEN, DECISION_ON);
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, None),
            jev_compaction: None,
            right: Some(&usage),
        };
        let composed = compose_tray_row(row, 20);
        let plain = strip_ansi(&composed);
        // The counter survives, right-aligned, on the same single line.
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert!(!plain.contains('\n'));
        assert_eq!(visible_width(&plain), 20);
        assert!(
            plain.contains('…'),
            "the left label is ellipsised: {plain:?}"
        );
    }

    #[test]
    fn full_width_left_string_still_leaves_room_for_the_counter() {
        // A left string that alone overflows the terminal must not push the
        // counter off the row or onto a second line.
        let left = colored(DIM, &"x".repeat(200));
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: None,
            jev_compaction: None,
            right: Some(&usage),
        };
        let plain = strip_ansi(&compose_tray_row(row, 80));
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert_eq!(visible_width(&plain), 80);
        assert!(!plain.contains('\n'));
    }

    #[test]
    fn segments_without_a_compact_form_fall_back_to_left_truncation() {
        // Old daemon: no statusCompactText. The full segment stays, the left
        // label truncates, the counter survives on the same row.
        let left = colored(DIM, LEFT);
        let decision_full = colored(GREEN, DECISION_ON);
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, None),
            jev_compaction: None,
            right: Some(&usage),
        };
        let plain = strip_ansi(&compose_tray_row(row, 50));
        assert!(plain.ends_with(USAGE), "{plain:?}");
        assert!(plain.contains(DECISION_ON), "{plain:?}");
        assert!(visible_width(&plain) <= 50);
        assert!(!plain.contains('\n'));
    }

    #[test]
    fn no_counter_renders_left_and_segments_only() {
        let left = colored(DIM, "GPT-6 Astra xhigh");
        let decision_full = colored(RED, "\u{25cf} Jev Off");
        let compaction_full = colored(RED, "\u{25cf} Jev compact off");
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, None),
            jev_compaction: Some(TraySegment::new(&compaction_full)),
            right: None,
        };
        let plain = strip_ansi(&compose_tray_row(row, 80));
        assert_eq!(
            plain,
            "GPT-6 Astra xhigh  \u{25cf} Jev Off  \u{25cf} Jev compact off"
        );
    }

    #[test]
    fn unicode_widths_are_counted_as_graphemes_not_bytes() {
        // Wide CJK glyphs are 2 columns each; the counter must still land on the
        // right edge without overlap.
        let left_text = "\u{6a21}\u{578b} GPT-6 Astra xhigh";
        let left = colored(DIM, left_text);
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: None,
            jev_compaction: None,
            right: Some(&usage),
        };
        let plain = strip_ansi(&compose_tray_row(row, 40));
        let expected = format!("{left_text}{}{USAGE}", " ".repeat(40 - 22 - 10));
        assert_eq!(plain, expected, "{plain:?}");
        assert_eq!(visible_width(&plain), 40);
    }

    #[test]
    fn one_column_terminal_never_panics_or_overflows() {
        let left = colored(DIM, "GPT-6 Astra xhigh");
        let decision_full = colored(GREEN, DECISION_ON);
        let decision_narrow = colored(GREEN, "\u{25cf}");
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, Some(&decision_narrow)),
            jev_compaction: None,
            right: Some(&usage),
        };
        for width in [1usize, 2, 3, 5, 9, 15, 20] {
            let composed = compose_tray_row(row, width);
            assert!(
                visible_width(&strip_ansi(&composed)) <= width,
                "{width}: {composed:?}"
            );
        }
    }

    #[test]
    fn compact_form_is_used_only_when_the_full_row_does_not_fit() {
        // At a comfortable width the labelled forms are shown, not the dots.
        let left = colored(DIM, "GPT-6 Astra xhigh");
        let decision_full = colored(GREEN, DECISION_ON);
        let decision_narrow = colored(GREEN, "\u{25cf}");
        let compaction_full = colored(RED, "\u{25cf} Jev compact off");
        let usage = colored(DIM, USAGE);
        let row = TrayRow {
            left: &left,
            jev_decision: segment(&decision_full, Some(&decision_narrow)),
            jev_compaction: Some(TraySegment::new(&compaction_full)),
            right: Some(&usage),
        };
        let plain = strip_ansi(&compose_tray_row(row, 80));
        assert!(plain.contains(DECISION_ON), "{plain:?}");
        assert!(plain.contains("\u{25cf} Jev compact off"), "{plain:?}");
    }
}
