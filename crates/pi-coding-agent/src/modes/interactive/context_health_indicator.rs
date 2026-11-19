//! Context risk beside streaming TPS, using only the current UI snapshot.
//!
//! The 70/85 pressure bands are ported from Claude HUD's `getContextColor` at
//! 5f725e737a882211bb62b8fc56ab593d4d97f18a (MIT, Copyright 2026 Jarrod Watts).
//! See resources/agent/docs/third-party/claude-hud-MIT.txt for the notice.

use crate::core::context_health::{context_health, ContextHealthBand, ContextHealthInput};
use pi_tui::utils::visible_width;

/// The established CTX-health green, also used by OCC3.1 role projections.
/// Keep this as the single source of the ANSI-256 token for both renderers.
pub(super) const CONTEXT_HEALTH_GREEN: u8 = 82;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Risk {
    Green,
    Amber,
    Red,
}

// Port of src/render/colors.ts:getContextColor's default threshold policy.
fn pressure_risk(percent: f64) -> Risk {
    if percent >= 85.0 {
        Risk::Red
    } else if percent >= 70.0 {
        Risk::Amber
    } else {
        Risk::Green
    }
}

struct Indicator {
    risk: Option<Risk>,
    score: Option<f64>,
    compacting: bool,
}

impl Indicator {
    fn from_input(input: &ContextHealthInput) -> Self {
        let valid = input.tokens.is_some_and(|n| n.is_finite() && n >= 0.0)
            && input.context_window.is_finite()
            && input.context_window > 0.0
            && input.compaction_cycles.is_finite()
            && input.compaction_cycles >= 0.0;
        if !valid {
            return Self {
                risk: input.is_compacting.then_some(Risk::Amber),
                score: None,
                compacting: input.is_compacting,
            };
        }
        let health = context_health(input);
        let score_risk = match health.band {
            ContextHealthBand::Healthy => Risk::Green,
            ContextHealthBand::Straining | ContextHealthBand::Degraded => Risk::Amber,
            ContextHealthBand::Critical => Risk::Red,
        };
        let pressure = pressure_risk(health.utilisation.unwrap_or(1.5) * 100.0);
        let risk = score_risk.max(pressure).max(if input.is_compacting {
            Risk::Amber
        } else {
            Risk::Green
        });
        Self {
            risk: Some(risk),
            score: Some(health.score),
            compacting: input.is_compacting,
        }
    }

    fn label(&self, compact: bool) -> String {
        let (name, short, colour) = match self.risk {
            Some(Risk::Green) => ("GREEN", "G", CONTEXT_HEALTH_GREEN),
            Some(Risk::Amber) => ("AMBER", "A", 214),
            Some(Risk::Red) => ("RED", "R", 196),
            None => ("UNKNOWN", "?", 244),
        };
        let value = if self.compacting {
            if compact {
                "…".into()
            } else {
                " compacting".into()
            }
        } else {
            self.score
                .map(|score| format!("{}{:.0}", if compact { "" } else { " " }, score.round()))
                .unwrap_or_default()
        };
        let text = if compact {
            format!("CTX {short}{value}")
        } else {
            format!("CTX HEALTH {name}{value}")
        };
        if self.risk == Some(Risk::Amber) {
            return crate::modes::interactive::theme::theme::theme().fg("warning", &text);
        }
        // Match the TPS segment's partial reset so selection backgrounds survive.
        format!("\x1b[1;38;5;{colour}m{text}\x1b[22;39m")
    }
}

/// Narrow in stages, preserving TPS and health together before the optional
/// communication labels. The caller owns border padding and final truncation.
pub(super) fn status_line(
    communication: &str,
    queue_badge: &str,
    throughput: &str,
    input: &ContextHealthInput,
    available: usize,
) -> String {
    let health = Indicator::from_input(input);
    let full = health.label(false);
    let compact = health.label(true);
    let with_badge = if queue_badge.is_empty() {
        communication.to_string()
    } else {
        format!("{communication}  {queue_badge}")
    };
    for text in [
        format!("{with_badge}  {throughput}  {full}"),
        format!("{communication}  {throughput}  {full}"),
        format!("{with_badge}  {throughput}  {compact}"),
        format!("{communication}  {throughput}  {compact}"),
        format!("{throughput}  {full}"),
        format!("{throughput}  {compact}"),
    ] {
        if visible_width(&text) <= available {
            return text;
        }
    }
    pi_tui::utils::truncate_to_width(&compact, available as f64, "…", false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_tui::utils::strip_ansi;

    fn input(tokens: f64) -> ContextHealthInput {
        ContextHealthInput {
            tokens: Some(tokens),
            context_window: 1000.0,
            ..Default::default()
        }
    }

    #[test]
    fn claude_hud_pressure_boundaries_are_inclusive_and_unrounded() {
        for (percent, expected) in [
            (0.0, Risk::Green),
            (69.999, Risk::Green),
            (70.0, Risk::Amber),
            (84.999, Risk::Amber),
            (85.0, Risk::Red),
            (150.0, Risk::Red),
        ] {
            assert_eq!(pressure_risk(percent), expected, "{percent}");
        }
        // A nearly full fresh session must be red even when its legacy score is >40.
        assert!(context_health(&input(850.0)).score > 40.0);
        assert_eq!(Indicator::from_input(&input(850.0)).risk, Some(Risk::Red));
        assert!(strip_ansi(&Indicator::from_input(&input(850.0)).label(false)).ends_with("RED 63"));
    }

    #[test]
    fn retained_score_and_compaction_pressure_can_raise_the_band() {
        assert_eq!(Indicator::from_input(&input(146.0)).risk, Some(Risk::Green));
        let mut compacted = input(146.0);
        compacted.compaction_cycles = 5.0;
        let indicator = Indicator::from_input(&compacted);
        assert_eq!(indicator.risk, Some(Risk::Amber));
        assert_eq!(indicator.score, Some(70.0));
        compacted.tokens = Some(950.0);
        assert_eq!(Indicator::from_input(&compacted).risk, Some(Risk::Red));
    }

    #[test]
    fn unknown_invalid_and_post_compaction_usage_never_claim_green_or_a_score() {
        let mut cases = vec![ContextHealthInput::default()];
        for tokens in [None, Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
            cases.push(ContextHealthInput {
                tokens,
                compaction_cycles: 4.0,
                ..input(0.0)
            });
        }
        for window in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            cases.push(ContextHealthInput {
                context_window: window,
                ..input(0.0)
            });
        }
        for cycles in [-1.0, f64::NAN, f64::INFINITY] {
            cases.push(ContextHealthInput {
                compaction_cycles: cycles,
                ..input(0.0)
            });
        }
        for input in cases {
            let indicator = Indicator::from_input(&input);
            assert_eq!(indicator.risk, None, "{input:?}");
            assert_eq!(indicator.score, None);
            assert_eq!(strip_ansi(&indicator.label(false)), "CTX HEALTH UNKNOWN");
        }
    }

    #[test]
    fn compaction_is_at_least_amber_and_clears_when_the_snapshot_recovers() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        for tokens in [None, Some(100.0), Some(900.0)] {
            let mut input = ContextHealthInput {
                tokens,
                is_compacting: true,
                ..input(0.0)
            };
            let indicator = Indicator::from_input(&input);
            assert!(indicator.risk >= Some(Risk::Amber));
            assert!(strip_ansi(&indicator.label(false)).ends_with("compacting"));
            input.is_compacting = false;
            input.tokens = Some(100.0);
            assert_eq!(Indicator::from_input(&input).risk, Some(Risk::Green));
        }
    }

    #[test]
    fn all_bands_have_explicit_text_and_green_amber_red_ansi_colours() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        for (tokens, name, code) in [
            (100.0, "GREEN", 82),
            (750.0, "AMBER", 214),
            (900.0, "RED", 196),
        ] {
            let text = Indicator::from_input(&input(tokens)).label(false);
            assert!(text.contains(name));
            if name == "AMBER" {
                assert_eq!(text, crate::modes::interactive::theme::theme::theme().fg("warning", &strip_ansi(&text)));
            } else {
                assert!(text.starts_with(&format!("\x1b[1;38;5;{code}m")));
                assert!(text.ends_with("\x1b[22;39m"));
            }
        }
    }

    #[test]
    fn queue_badge_drops_before_delivery_slots_at_width_boundary() {
        let communication = "○ TXT  ○ OCC1  ○ OCC2  ○ OCC3.1  ● OCC3.2";
        let throughput = "STREAM TPS ~50.0";
        let input = input(100.0);
        let expected = format!("{communication}  {throughput}  {}", Indicator::from_input(&input).label(true));
        let width = visible_width(&expected);
        let rendered = status_line(communication, "● QUEUED 2", throughput, &input, width);
        assert_eq!(rendered, expected);
        assert!(!rendered.contains("QUEUED"));
        assert!(rendered.contains("● OCC3.2"));
        assert_eq!(visible_width(&rendered), width);
        let full = format!("{communication}  {throughput}  {}", Indicator::from_input(&input).label(false));
        let compact_with_badge = format!("{communication}  ● QUEUED 2  {throughput}  {}", Indicator::from_input(&input).label(true));
        let width = visible_width(&full).max(visible_width(&compact_with_badge));
        let rendered = status_line(communication, "● QUEUED 2", throughput, &input, width);
        assert_eq!(rendered, full, "drop badge before compacting health");
    }

    #[test]
    fn tps_and_health_stay_adjacent_before_communication_labels_are_dropped() {
        let throughput = "\x1b[1;38;5;196mSTREAM TPS ~50.0\x1b[22;39m";
        for width in [0, 8, 20, 30, 40, 60, 80, 120] {
            let text = status_line("○ TXT  ○ OCC1  ○ OCC2", "", throughput, &input(100.0), width);
            let plain = strip_ansi(&text);
            if width >= 25 {
                assert!(plain.contains("STREAM TPS ~50.0  CTX"), "{width}: {plain}");
            }
            if width >= 80 {
                assert!(plain.contains("○ OCC2  STREAM TPS ~50.0  CTX HEALTH GREEN 100"));
            }
            if width >= 8 {
                assert!(visible_width(&text) <= width);
            }
        }
    }
}
