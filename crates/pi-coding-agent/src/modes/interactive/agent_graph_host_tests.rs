use super::*;
use super::native_host::host_event_channel;
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
    let (send, receive) = host_event_channel(64 * 1024);
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
