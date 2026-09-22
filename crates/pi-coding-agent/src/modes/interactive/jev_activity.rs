//! Cached Jev activity for the right side of the bordered agents bar.

use pi_jev::config::JevMode;
use pi_jev::telemetry::SessionUsage;
use serde_json::Value;

use crate::modes::interactive::components::subagent_summary_line::RightStatus;
use crate::modes::interactive::theme::theme::theme;

fn number(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

pub(super) fn status(
    mode: JevMode,
    full: bool,
    compaction: bool,
    response: Option<&Value>,
) -> Option<RightStatus> {
    if !mode.is_enabled() && !compaction {
        return None;
    }
    let label = if full {
        "Jev Full"
    } else {
        match mode {
            JevMode::Compare => "Jev Compare",
            JevMode::Active => "Jev Active",
            JevMode::CompareAndActive => "Jev C+A",
            JevMode::Off => "Jev Cmp",
        }
    };
    let pipeline = response.and_then(|value| value.get("pipeline"));
    let usage: Option<SessionUsage> = pipeline
        .and_then(|value| value.get("usage"))
        .and_then(|value| serde_json::from_value(value.clone()).ok());
    let fallback = pipeline.is_some_and(|value| {
        [
            value.get("fallback_reason"),
            value
                .get("active")
                .and_then(|active| active.get("last_reason")),
        ]
        .into_iter()
        .flatten()
        .any(|reason| reason.as_str().is_some_and(|reason| !reason.is_empty()))
    });
    let phase = if let Some(usage) = &usage {
        if usage.in_flight > 0 {
            match usage.activity.as_str() {
                "compacting" => "compacting",
                "checking tools" => "checking tools",
                "searching" => "searching",
                "selecting skills" => "selecting skills",
                "checking context" => "checking context",
                "checking safety" => "checking safety",
                "checking effort" => "checking effort",
                "verifying" => "verifying",
                _ => "evaluating",
            }
        } else if fallback {
            "fallback"
        } else {
            "idle"
        }
    } else if response.is_none() {
        "stats unavailable"
    } else if fallback {
        "fallback"
    } else {
        "idle"
    };
    let color = if usage.as_ref().is_some_and(|usage| usage.in_flight > 0)
        || fallback
        || response.is_none()
    {
        "warning"
    } else {
        "success"
    };
    let (detail, short) = if let Some(usage) = usage {
        let count = |value: Option<u64>| value.map(number).unwrap_or_else(|| "—".into());
        let partial = if usage.incomplete_usage { "+" } else { "" };
        let total = match (usage.input_tokens, usage.output_tokens) {
            (None, None) => "—".into(),
            (input, output) => number(input.unwrap_or(0).saturating_add(output.unwrap_or(0))),
        };
        let latency = usage
            .last_latency_ms
            .map(|ms| format!(" · {}ms", number(ms)))
            .unwrap_or_default();
        let errors = if usage.failed > 0 {
            format!(" · {} err", number(usage.failed))
        } else {
            String::new()
        };
        (
            format!(
                "{} req · {} in/{} out{partial}{latency}{errors}",
                number(usage.requests),
                count(usage.input_tokens),
                count(usage.output_tokens)
            ),
            format!("{}r {total}{partial}t", number(usage.requests)),
        )
    } else {
        ("req — · tokens —".into(), "tokens —".into())
    };
    let compact_phase = match phase {
        "checking tools" => "tools",
        "selecting skills" => "skills",
        "checking context" => "context",
        "checking safety" => "safety",
        "checking effort" => "effort",
        "stats unavailable" => "unavailable",
        "evaluating" => "busy",
        "compacting" => "compact",
        "searching" => "search",
        "verifying" => "verify",
        other => other,
    };
    let compact_label = match mode {
        JevMode::Compare if !full => "Jev C",
        JevMode::Active if !full => "Jev A",
        _ => label,
    };
    Some(RightStatus {
        full: theme().fg(color, &format!("{label} · {phase} · {detail}")),
        compact: theme().fg(color, &format!("{compact_label} {compact_phase} {short}")),
        minimal: theme().fg(color, label),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_hidden_and_old_workers_never_invent_usage() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        assert!(status(JevMode::Off, false, false, None).is_none());
        let old = status(
            JevMode::Compare,
            false,
            false,
            Some(&serde_json::json!({"pipeline": {"success_count": 8}})),
        )
        .unwrap();
        assert!(old.full.contains("tokens —"));
        assert!(!old.full.contains("0 req"));
        assert!(status(JevMode::Off, false, true, None)
            .unwrap()
            .full
            .contains("Jev Cmp"));
    }

    #[test]
    fn full_jev_shows_real_usage_and_bounded_activity() {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let usage = SessionUsage {
            requests: 12,
            in_flight: 1,
            input_tokens: Some(1200),
            output_tokens: Some(45),
            activity: "checking tools".into(),
            ..Default::default()
        };
        let response = serde_json::json!({"pipeline": {"usage": usage}});
        let rendered = status(JevMode::CompareAndActive, true, true, Some(&response)).unwrap();
        assert!(rendered
            .full
            .contains("Jev Full · checking tools · 12 req · 1.2k in/45 out"));
        assert!(rendered.compact.contains("12r 1.2kt"));
        let partial = serde_json::json!({"pipeline": {"usage": SessionUsage { input_tokens: Some(0), incomplete_usage: true, activity: "\u{1b}[2Jsecret".into(), ..usage }}});
        let rendered = status(JevMode::Compare, false, false, Some(&partial)).unwrap();
        assert!(rendered.full.contains("evaluating"));
        assert!(rendered.full.contains("0 in/45 out+"));
        assert!(!rendered.full.contains("secret"));
    }
}
