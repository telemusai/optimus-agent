use super::*;
use crate::modes::agent_connection::daemon_agent_connection::DaemonTransportClient;
use pi_ai::types::BoxFuture;
use pi_tui::utils::{strip_ansi, visible_width};

struct Roster(Mutex<Vec<serde_json::Value>>);
impl AgentConnectionRosterStore for Roster {
    fn attach(&self, _: Arc<dyn DaemonTransportClient>) -> BoxFuture<Result<bool, String>> {
        Box::pin(async { Ok(true) })
    }
    fn on_update(&self, _: Arc<dyn Fn() + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }
    fn summaries(&self) -> Vec<serde_json::Value> {
        self.0.lock().unwrap().clone()
    }
    fn dispose(&self) -> BoxFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn graph_header_reads_roster_pushes_and_stops_its_timer_when_hidden() {
    let h = super::super::ui_tests::FrameHarness::new("graph-host");
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    h.mode.borrow_mut().patch_connection_state(|state| {
        state.session_id = "root".into();
        state.active_session_id = Some("root-active".into());
    });
    let row = |id: &str, parent: Option<&str>| {
        serde_json::to_value(crate::modes::daemon::daemon_session_list::SessionSummary {
            id: id.into(),
            session_id: id.into(),
            active_session_id: Some(id.into()),
            parent_session_id: parent.map(str::to_string),
            lifecycle: "live".into(),
            activity: "working".into(),
            is_session_active: true,
            runtime_kind: Some(
                if parent.is_some() {
                    "subagent"
                } else {
                    "top-level"
                }
                .into(),
            ),
            extra: [(
                "communicationActivity".into(),
                serde_json::json!({"generation":"g", "deliveries":[]}),
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        })
        .unwrap()
    };
    let roster = Arc::new(Roster(Mutex::new(vec![
        row("root", None),
        row("child", Some("root")),
    ])));
    let mut bar = Bar::new(h.mode.clone());
    *bar.roster.lock().unwrap() = Some(roster.clone());
    let (send, receive) = mpsc::channel();
    *bar.render_send.borrow_mut() = Some(send);
    let before = bar.graph_rows(24, 4);
    assert_eq!(before.len(), 4);
    assert!(before.iter().all(|line| visible_width(line) == 24));
    assert_eq!(strip_ansi(&before.join("")).matches('●').count(), 1);
    assert!(bar.graph_task.is_some());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(matches!(receive.try_recv(), Ok(HostEvent::Render)));
    roster.0.lock().unwrap()[1]["communicationActivity"]["deliveries"] = serde_json::json!([{
        "id":"fresh", "sourceSessionId":"root", "kind":"occ1", "timestamp":chrono::Utc::now().timestamp_millis(),
    }]);
    let after = bar.graph_rows(24, 4);
    assert_ne!(before, after);
    let bar = Rc::new(RefCell::new(bar));
    h.transcript.borrow_mut().subagents = Some(bar.clone());
    let mut header =
        super::super::native_neon::Header(h.mode.clone(), Rc::downgrade(&h.transcript));
    for width in [110, 120, 160] {
        for budget in [7, 9, 12, 16] {
            let frame = header.render_with_height(width as f64, budget);
            let expected = if width >= 160 { 5 } else { 6 };
            assert_eq!(frame.len(), expected);
            assert!(frame.iter().all(|line| visible_width(line) == width));
            assert!(strip_ansi(&frame.join("\n")).contains("MAIN CPU"));
            assert!(strip_ansi(&frame[expected - 2]).starts_with('┌'));
            assert!(strip_ansi(&frame[expected - 1]).starts_with('├'));
        }
    }
    header.render_with_height(80.0, 6);
    assert!(bar.borrow().graph_task.is_none());
    assert!(bar.borrow().graph_deadline.is_none());
}

#[tokio::test]
async fn graph_uses_only_authoritative_session_activity_and_hiding_cancels_its_existing_timer() {
    let h = super::super::ui_tests::FrameHarness::new("textmas-graph-host");
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let activity = super::agent_graph::activity_tests::sequential_fixture();
    h.mode.borrow_mut().patch_connection_state(|state| {
        state.session_id = "root".into();
        state.active_session_id = Some("root-active".into());
        state.occ3_modes = Some(super::super::wire::Occ3Modes {
            textmas_mode: crate::core::occ::textmas::TextMasMode::Sequential,
            textmas_active: true,
            textmas_activity: Some(activity.clone()),
            ..Default::default()
        });
    });
    let roster = Arc::new(Roster(Mutex::new(vec![serde_json::to_value(
        crate::modes::daemon::daemon_session_list::SessionSummary {
            id: "root".into(),
            session_id: "root".into(),
            active_session_id: Some("root-active".into()),
            lifecycle: "live".into(),
            runtime_kind: Some("top-level".into()),
            ..Default::default()
        },
    )
    .unwrap()])));
    let mut bar = Bar::new(h.mode.clone());
    *bar.roster.lock().unwrap() = Some(roster.clone());
    let (send, _receive) = mpsc::channel();
    *bar.render_send.borrow_mut() = Some(send);
    let frame = bar.graph_rows(24, 6);
    let plain = strip_ansi(&frame.join("\n"));
    assert!(
        plain.contains("P…") && plain.contains("C…") && plain.contains("R…"),
        "{plain}"
    );
    assert!(frame.join("\n").contains("38;5;82m"));
    assert!(
        bar.graph_task.is_some(),
        "the existing graph timer, not a new activity loop, updates progress"
    );
    assert_eq!(
        roster.0.lock().unwrap().len(),
        1,
        "activity cannot add a roster row"
    );
    assert_eq!(roster.0.lock().unwrap()[0]["sessionId"], "root");

    let replace =
        |activity: &crate::core::occ::textmas_activity::TextMasActivitySnapshot,
         action: &str,
         status: crate::core::occ::textmas_activity::TextMasActivityStatus| {
            let mut replacement = activity.clone();
            replacement.action_id = action.into();
            replacement.active = matches!(
                status,
                crate::core::occ::textmas_activity::TextMasActivityStatus::Running
            );
            replacement.stale = false;
            for call in &mut replacement.calls {
                call.call_id = call.call_id.replace("action-1", action);
                call.depends_on = call
                    .depends_on
                    .iter()
                    .map(|id| id.replace("action-1", action))
                    .collect();
                call.status = status;
            }
            assert!(replacement.is_valid());
            h.mode.borrow_mut().patch_connection_state(|state| {
                state.occ3_modes.as_mut().unwrap().textmas_activity = Some(replacement.clone());
            });
            replacement
        };
    replace(
        &activity,
        "second",
        crate::core::occ::textmas_activity::TextMasActivityStatus::Running,
    );
    let before_late = strip_ansi(&bar.graph_rows(24, 6).join("\n"));
    replace(
        &activity,
        "action-1",
        crate::core::occ::textmas_activity::TextMasActivityStatus::Cancelled,
    );
    let after_late = strip_ansi(&bar.graph_rows(24, 6).join("\n"));
    assert!(
        after_late.contains("C…"),
        "late old action is not a new authoritative plan: {after_late}"
    );
    assert!(before_late.contains("C…"));
    let cached_before_stop = h
        .mode
        .borrow()
        .connection_state
        .as_ref()
        .unwrap()
        .occ3_modes
        .as_ref()
        .unwrap()
        .textmas_activity
        .clone();
    super::super::native_state::stop_activity(&mut h.mode.borrow_mut());
    let disconnected = strip_ansi(&bar.graph_rows(24, 6).join("\n"));
    assert!(disconnected.contains("disconnected"), "{disconnected}");
    {
        let mode = h.mode.borrow();
        let modes = mode
            .connection_state
            .as_ref()
            .unwrap()
            .occ3_modes
            .as_ref()
            .unwrap();
        assert!(!modes.textmas_active, "disconnect clears the owner flag");
        assert_eq!(
            modes.textmas_activity.as_ref(),
            cached_before_stop.as_ref(),
            "disconnect changes only the owner flag, never the cached projection"
        );
    }
    assert!(
        disconnected.contains("C?") && !disconnected.contains("C×"),
        "{disconnected}"
    );
    h.mode.borrow_mut().patch_connection_state(|state| {
        state.session_id = "other".into();
        state.active_session_id = Some("other-active".into());
    });
    assert!(
        !bar.graph_rows(24, 6).join("\n").contains("38;5;82m"),
        "old session activity cannot cross attachment"
    );
    assert!(bar.graph_rows(0, 0).is_empty());
    assert!(bar.graph_task.is_none());
    assert!(bar.graph_deadline.is_none());
}
