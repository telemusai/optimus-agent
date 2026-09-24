use super::*;
use crate::core::goals::{GoalState, GoalStatus};
use pi_tui::utils::strip_ansi;

fn goal() -> GoalState {
    GoalState {
        active: true,
        status: GoalStatus::Active,
        goal_id: Some("footer-goal".into()),
        objective: Some("Finish the update".into()),
        time_used_seconds: 65.0,
        created_at: Some(1_000.0),
        updated_at: Some(66_000.0),
        ..Default::default()
    }
}

#[test]
fn goal_clock_ticks_and_accepts_authoritative_counters_without_double_counting() {
    let mut clock = GoalClock::default();
    let now = Instant::now();
    let mut goal = goal();
    assert_eq!(clock.elapsed("s1", &goal, now), Some(65.0));
    assert_eq!(clock.elapsed("s1", &goal, now + Duration::from_secs(5)), Some(70.0));
    goal.time_used_seconds = 70.0;
    assert_eq!(clock.elapsed("s1", &goal, now + Duration::from_secs(5)), Some(70.0));
    assert_eq!(clock.elapsed("s1", &goal, now + Duration::from_secs(8)), Some(73.0));
}

#[test]
fn goal_clock_excludes_pauses_and_resets_for_other_goals_and_sessions() {
    let mut clock = GoalClock::default();
    let now = Instant::now();
    let mut goal = goal();
    clock.elapsed("s1", &goal, now);
    goal.status = GoalStatus::Paused;
    goal.active = false;
    goal.time_used_seconds = 70.0;
    assert_eq!(clock.elapsed("s1", &goal, now + Duration::from_secs(5)), None);
    goal.status = GoalStatus::Active;
    goal.active = true;
    assert_eq!(clock.elapsed("s1", &goal, now + Duration::from_secs(100)), Some(70.0));
    goal.goal_id = Some("next-goal".into());
    goal.time_used_seconds = 0.0;
    assert_eq!(clock.elapsed("s1", &goal, now + Duration::from_secs(110)), Some(0.0));
    assert_eq!(clock.elapsed("s2", &goal, now + Duration::from_secs(120)), Some(0.0));
    for status in [GoalStatus::Complete, GoalStatus::Idle, GoalStatus::Error, GoalStatus::BudgetLimited] {
        goal.status = status;
        assert_eq!(clock.elapsed("s2", &goal, now + Duration::from_secs(130)), None);
    }
}

#[test]
fn goal_status_and_heartbeat_share_the_painted_context_row() {
    for style in ["neon", "prime"] {
        let h = super::super::ui_tests::FrameHarness::new("goal-footer");
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        h.mode.borrow_mut().fullscreen_enabled = true;
        h.mode.borrow_mut().patch_connection_state(|state| {
            state.goal = goal();
            state.context_usage = local::ContextUsage {
                tokens: Some(146_000.0), context_window: 1_000_000.0, percent: Some(14.6),
            };
            state.heartbeat = Some(local::AgentCronJob {
                id: "goal-footer-heartbeat".into(), source: Some("heartbeat".into()),
                status: "active".into(), session_id: state.session_id.clone(), ..Default::default()
            });
        });
        GOAL_CLOCK.with(|clock| *clock.borrow_mut() = GoalClock::default());
        for width in [80, 120, 200] {
            h.width.set(width);
            let frame = h.paint();
            let footer = frame.iter().find(|line| line.contains("146k (15%)")).unwrap();
            assert!(footer.contains("HB user on"), "{style}/{width}: {footer}");
            assert!(footer.contains("Pursuing Goal 1m 05s"), "{style}/{width}: {footer}");
            assert!(footer.find("HB").unwrap() < footer.find("Pursuing Goal").unwrap());
            assert!(footer.find("Pursuing Goal").unwrap() < footer.find("146k").unwrap());
            assert!(footer.trim_end().ends_with("146k (15%)"));
            if style == "neon" && width >= 100 {
                assert!(footer.contains("[━"), "context bar must stay visible: {footer}");
            }
            assert!(frame.iter().all(|line| visible_width(line) <= width));
        }
        for status in [GoalStatus::Paused, GoalStatus::Complete, GoalStatus::BudgetLimited, GoalStatus::Error, GoalStatus::Idle] {
            h.mode.borrow_mut().patch_connection_state(|state| state.goal.status = status);
            let frame = h.paint();
            let footer = frame.iter().find(|line| line.contains("146k (15%)")).unwrap();
            assert!(!footer.contains("Pursuing Goal"));
            assert!(footer.contains("HB user on"));
        }
    }
}

#[test]
fn goal_footer_handles_missing_usage_no_heartbeat_and_narrow_widths() {
    let mut mode = super::super::tests::stash_mode("goal-footer-width");
    mode.apply_connection_state_snapshot(local::AgentConnectionState {
        session_id: "goal-footer-width".into(), goal: goal(), ..Default::default()
    });
    GOAL_CLOCK.with(|clock| *clock.borrow_mut() = GoalClock::default());
    assert_eq!(strip_ansi(&right_status(&mode, None, 80).unwrap()), "Pursuing Goal 1m 05s");
    mode.patch_connection_state(|state| state.heartbeat = Some(local::AgentCronJob {
        source: Some("heartbeat".into()), status: "active".into(), ..Default::default()
    }));
    for width in [1, 5, 10, 13, 20, 32, 40, 60, 80, 120] {
        let right = right_status(&mode, Some("146k (15%)"), width);
        let row = tray_row::compose_tray_row(tray_row::TrayRow {
            left: "界界 model xhigh", jev_decision: None, jev_compaction: None, right: right.as_deref(),
        }, width);
        assert!(visible_width(&row) <= width);
        if width >= 10 { assert!(strip_ansi(&row).ends_with("146k (15%)")); }
        if width >= 32 { assert!(row.contains("HB") && row.contains("Goal"), "{width}: {row}"); }
    }
}
