//! Offline snapshot-to-render proof for steady modes versus receipt-backed pulses.
use super::*;
use crate::core::agent_messages::{
    create_agent_session_message, stamp_occ3_delivery_record, AgentSessionMessageEndpoint,
    AgentSessionMessagePayload, AgentSessionMessageSender, Occ3DeliveryDetails,
};
use crate::core::occ::{activity::from_admitted_messages, textmas::TextMasMode};
use crate::modes::agent_connection::daemon_agent_connection::DaemonTransportClient;
use crate::modes::agent_connection::types::Occ3Modes;
use crate::modes::interactive::components::subagent_summary_line::RightStatus;
use pi_ai::types::BoxFuture;
use pi_tui::utils::{strip_ansi, visible_width};
use serde_json::json;

struct Roster(Mutex<Vec<serde_json::Value>>);
impl AgentConnectionRosterStore for Roster {
    fn attach(&self, _: Arc<dyn DaemonTransportClient>) -> BoxFuture<Result<bool, String>> {
        Box::pin(async { Ok(true) })
    }
    fn on_update(&self, _: Arc<dyn Fn() + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }
    fn summaries(&self) -> Vec<serde_json::Value> { self.0.lock().unwrap().clone() }
    fn dispose(&self) -> BoxFuture<()> { Box::pin(async {}) }
}

fn snapshot(session: &str, modes: Option<Occ3Modes>) -> local::AgentConnectionState {
    super::super::project_state(wire::AgentConnectionState {
        session_id: session.into(),
        active_session_id: Some(format!("active-{session}")),
        context_usage: json!({"tokens":100.0,"contextWindow":1000.0,"percent":10.0}),
        occ3_modes: modes,
        ..Default::default()
    })
}

fn modes(textmas_mode: TextMasMode, enabled: bool, active: bool, available: bool) -> Occ3Modes {
    Occ3Modes { textmas_mode, latent_enabled: enabled, latent_active: active, latent_available: available }
}

fn roster() -> Arc<Roster> {
    let rows = ["parent", "child"].into_iter().map(|id| {
        serde_json::to_value(crate::modes::daemon::daemon_session_list::SessionSummary {
            id: id.into(), session_id: id.into(), lifecycle: "live".into(),
            active_session_id: Some(format!("active-{id}")),
            runtime_kind: Some(if id == "child" { "subagent" } else { "top-level" }.into()),
            parent_session_id: (id == "child").then(|| "parent".into()),
            extra: [("communicationActivity".into(), json!({"generation":"g","deliveries":[]}))]
                .into_iter().collect(),
            ..Default::default()
        }).unwrap()
    }).collect();
    Arc::new(Roster(Mutex::new(rows)))
}

fn deliver(roster: &Roster, id: &str, marker: Option<&Occ3DeliveryDetails>) {
    let mut record = create_agent_session_message(&AgentSessionMessagePayload {
        id: id.into(), message: "OCC3.1 complete; OCC3.2 latent_native completed".into(),
        from: Some(AgentSessionMessageSender { session_id: Some("parent".into()), ..Default::default() }),
        target: AgentSessionMessageEndpoint { session_id: "child".into(), ..Default::default() },
        ..Default::default()
    }, chrono::Utc::now().timestamp_millis());
    if let Some(marker) = marker { stamp_occ3_delivery_record(&mut record, marker).unwrap(); }
    roster.0.lock().unwrap()[1]["communicationActivity"] = serde_json::to_value(
        from_admitted_messages(&[AgentMessage::Custom(record)], "child", "g"),
    ).unwrap();
}

fn capture(name: &str, rows: &[String]) {
    if let Ok(dir) = std::env::var("OPTIMUS_OCC3_MODES_CAPTURE_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/{name}.ansi"), rows.join("\n")).unwrap();
        std::fs::write(format!("{dir}/{name}.txt"), strip_ansi(&rows.join("\n"))).unwrap();
    }
}

#[test]
fn occ3_mode_bar_projects_all_modes_without_providers_or_activity() {
    for style in ["prime", "neon"] {
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        let h = super::super::ui_tests::FrameHarness::new("mode-fixture");
        let mut bar = Bar::new(h.mode.clone());
        {
            let mut jev = bar.jev.lock().unwrap();
            jev.select("mode-fixture".into());
            jev.status = Some(RightStatus {
                full: "Jev C+A · 12 req · 1.2k in/45 out".into(),
                compact: "Jev 12r".into(), minimal: "Jev".into(),
            });
        }
        for (name, mode, textmas, latent) in [
            ("absent", None, "unavailable", "unavailable"),
            ("off", Some(modes(TextMasMode::Off, false, false, false)), "off", "off"),
            ("disabled", Some(modes(TextMasMode::Off, false, true, true)), "off", "off"),
            ("seq", Some(modes(TextMasMode::Sequential, false, false, false)), "seq", "off"),
            ("hier-blocked", Some(modes(TextMasMode::Hierarchical, true, false, false)), "hier", "blocked"),
            ("not-active", Some(modes(TextMasMode::Sequential, true, false, true)), "seq", "blocked"),
            ("unavailable", Some(modes(TextMasMode::Sequential, true, true, false)), "seq", "blocked"),
            ("on", Some(modes(TextMasMode::Sequential, true, true, true)), "seq", "on"),
        ] {
            h.mode.borrow_mut().apply_connection_state_snapshot(snapshot("mode-fixture", mode));
            for focused in [false, true] {
                bar.line.set_focused(focused);
                for width in [0, 1, 2, 8, 20, 30, 40, 80, 110, 160] {
                    let rows = bar.render(width as f64);
                    assert!(rows.iter().all(|row| visible_width(row) <= width), "{style}/{name}/{width}: {rows:?}");
                    let text = strip_ansi(&rows.join("\n"));
                    if width >= 30 {
                        assert!(text.contains("OCC3.1") && text.contains("OCC3.2"), "{style}/{name}/{width}: {text}");
                    }
                    if width == 20 {
                        assert!(text.contains("CFG /occ"), "{text}");
                    }
                    if width >= 80 {
                        assert!(text.contains("CONFIG"));
                        assert!(text.contains(&format!("OCC3.1 {textmas}")), "{text}");
                        assert!(text.contains(&format!("OCC3.2 {latent}")), "{text}");
                        assert!(text.contains("STREAM TPS —  CTX HEALTH GREEN 100"), "{text}");
                        assert!(text.contains("Jev"), "{text}");
                        assert_eq!(rows.len(), if width >= 160 { 3 } else { 4 },
                            "configuration reuses the title, not another normal-width row");
                        if textmas == "seq" || textmas == "hier" {
                            assert!(rows[0].contains(&theme().fg("success", &format!("● OCC3.1 {textmas}"))));
                        }
                        if latent == "blocked" {
                            assert!(rows[0].contains(&theme().fg("warning", "! OCC3.2 blocked")));
                        }
                        if !focused { capture(&format!("{style}-{name}-{width}"), &rows); }
                    }
                    assert!(bar.pulse_deadline.is_none(), "config is not an animation trigger");
                    assert!(!bar.communication.indicators(Instant::now()).contains('●'));
                }
            }
        }
    }
}

#[test]
fn occ3_mode_bar_refresh_and_session_change_never_keep_old_modes() {
    let h = super::super::ui_tests::FrameHarness::new("mode-refresh");
    let mut bar = Bar::new(h.mode.clone());
    for (session, mode, expected) in [
        ("old", Some(modes(TextMasMode::Sequential, true, true, true)), ["OCC3.1 seq", "OCC3.2 on"]),
        ("old", Some(modes(TextMasMode::Hierarchical, true, false, false)), ["OCC3.1 hier", "OCC3.2 blocked"]),
        ("old", None, ["OCC3.1 unavailable", "OCC3.2 unavailable"]),
        ("new", Some(modes(TextMasMode::Off, false, false, false)), ["OCC3.1 off", "OCC3.2 off"]),
    ] {
        h.mode.borrow_mut().apply_connection_state_snapshot(snapshot(session, mode));
        let text = strip_ansi(&bar.render(110.0).join("\n"));
        assert!(expected.iter().all(|label| text.contains(label)), "{text}");
        assert!(bar.pulse_deadline.is_none());
    }
}

#[test]
fn occ3_mode_bar_completed_receipt_is_independent_of_config_and_queue() {
    for (name, mode) in [
        ("old-daemon", None),
        ("off", Some(modes(TextMasMode::Off, false, false, false))),
        ("blocked", Some(modes(TextMasMode::Hierarchical, true, false, false))),
    ] {
        let h = super::super::ui_tests::FrameHarness::new("receipt-fixture");
        h.mode.borrow_mut().apply_connection_state_snapshot(snapshot("parent", mode));
        let mut bar = Bar::new(h.mode.clone());
        let roster = roster();
        *bar.roster.lock().unwrap() = Some(roster.clone());
        bar.render(160.0);
        let identity = "a".repeat(64);
        let pending = Occ3DeliveryDetails::pending("fixture", 1, &identity).unwrap();
        deliver(&roster, "receipt", Some(&pending));
        roster.0.lock().unwrap()[1]["communicationQueue"] = json!({"pendingTotal":1});
        let queued = bar.render(160.0);
        assert!(strip_ansi(&queued.join("\n")).contains("QUEUED 1"));
        assert!(!bar.communication.indicators(Instant::now()).contains('●'));
        assert!(bar.pulse_deadline.is_some(), "only the independent queue flash is active");
        let mut completed = Occ3DeliveryDetails::latent_native("fixture", 1, &identity, &"b".repeat(64)).unwrap();
        completed.receipt_sha256 = None;
        deliver(&roster, "receipt", Some(&completed));
        bar.render(160.0);
        assert!(!bar.communication.indicators(Instant::now()).contains('●'), "receipt-less native claims stay suppressed");
        completed.receipt_sha256 = Some("b".repeat(64));
        deliver(&roster, "receipt", Some(&completed));
        roster.0.lock().unwrap()[1].as_object_mut().unwrap().remove("communicationQueue");
        let received = bar.render(160.0);
        let rendered = received.join("\n");
        assert!(rendered.contains("\x1b[1;38;5;39m● OCC3.2\x1b[22;39m"), "{rendered}");
        assert!(!strip_ansi(&rendered).contains("QUEUED"));
        assert!(!bar.communication.indicators(Instant::now()).contains("OCC3.1"));
        let deadline = bar.pulse_deadline.unwrap();
        bar.render(160.0);
        assert_eq!(bar.pulse_deadline, Some(deadline), "same receipt must not extend its pulse");
        assert!(!bar.communication.indicators(deadline).contains('●'));
        capture(&format!("receipt-{name}-160"), &received);
    }
}

#[test]
fn occ3_mode_bar_body_claims_only_pulse_txt() {
    let h = super::super::ui_tests::FrameHarness::new("body-claim-fixture");
    h.mode.borrow_mut().apply_connection_state_snapshot(snapshot("parent", Some(
        modes(TextMasMode::Sequential, true, true, true),
    )));
    let mut bar = Bar::new(h.mode.clone());
    let roster = roster();
    *bar.roster.lock().unwrap() = Some(roster.clone());
    bar.render(160.0);
    deliver(&roster, "body-claim", None);
    let rows = bar.render(160.0);
    let delivery = strip_ansi(&bar.communication.indicators(Instant::now()));
    assert_eq!(delivery, "● TXT  ○ OCC1  ○ OCC2  ○ OCC3.2");
    assert!(strip_ansi(&rows.join("\n")).contains("OCC3.1 seq"));
    assert!(strip_ansi(&rows.join("\n")).contains("OCC3.2 on"));
}

#[test]
fn occ3_mode_bar_fullscreen_paint_preserves_modes_tps_context_and_draft() {
    for style in ["prime", "neon"] {
        let h = super::super::ui_tests::FrameHarness::new("mode-fullscreen");
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        h.mode.borrow_mut().apply_connection_state_snapshot(snapshot("mode-fullscreen", Some(
            modes(TextMasMode::Sequential, true, false, false),
        )));
        h.editor.borrow_mut().editor_mut().set_text("preserve this draft");
        let bar = Rc::new(RefCell::new(Bar::new(h.mode.clone())));
        h.transcript.borrow_mut().subagents = Some(bar);
        h.ui.borrow_mut().exit_fullscreen(pi_tui::tui::ExitFullscreenOptions {
            flush: false, leave_alt_screen: false,
        });
        super::super::native_settings::fullscreen(true, &h.mode, &h.editor, &h.ui, &h.transcript);
        for width in [80, 110, 160] {
            h.width.set(width);
            h.height.set(32);
            h.ui.borrow_mut().request_render_forced();
            let frame = h.paint();
            assert!(frame.iter().all(|row| visible_width(row) <= width), "{style}/{width}: {frame:?}");
            let text = frame.join("\n");
            for label in ["CONFIG", "OCC3.1 seq", "OCC3.2 blocked", "STREAM TPS", "CTX HEALTH", "preserve this draft"] {
                assert!(text.contains(label), "{style}/{width} missing {label}: {text}");
            }
            capture(&format!("fullscreen-{style}-{width}"), &frame);
        }
    }
}
