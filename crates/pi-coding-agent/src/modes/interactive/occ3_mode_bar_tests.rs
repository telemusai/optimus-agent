//! Offline snapshot-to-render proof: the tray shows delivery lights only.
//! Steady mode configuration (OCC3.1/OCC3.2 enablement) never renders in the
//! bar; the OCC3.2 light blinks only on a completed native-use receipt.
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
            extra: [("communicationActivity".into(), json!({"generation":"g","deliveries":[]})) ]
                .into_iter().collect(),
            ..Default::default()
        }).unwrap()
    }).collect();
    Arc::new(Roster(Mutex::new(rows)))
}

fn deliver(roster: &Roster, id: &str, marker: Option<&Occ3DeliveryDetails>) {
    let mut record = create_agent_session_message(&AgentSessionMessagePayload {
        id: id.into(), message: "body claim only".into(),
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

/// Whatever the OCC3.1/OCC3.2 configuration is, the tray never shows it:
/// no CONFIG row, no mode labels in the title, no extra rows. Configuration
/// belongs to /occ panels; the tray owns delivery lights only.
#[test]
fn occ3_mode_bar_never_renders_configuration() {
    for style in ["prime", "neon"] {
        let h = super::super::ui_tests::FrameHarness::new("mode-fixture");
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        let mut bar = Bar::new(h.mode.clone());
        *bar.roster.lock().unwrap() = Some(roster());
        {
            let mut jev = bar.jev.lock().unwrap();
            jev.select("parent".into());
            jev.status = Some(RightStatus {
                full: "Jev C+A · 12 req · 1.2k in/45 out".into(),
                compact: "Jev 12r".into(), minimal: "Jev".into(),
            });
        }
        for (name, mode) in [
            ("absent", None),
            ("off", Some(modes(TextMasMode::Off, false, false, false))),
            ("seq", Some(modes(TextMasMode::Sequential, false, false, false))),
            ("hier-blocked", Some(modes(TextMasMode::Hierarchical, true, false, false))),
            ("on", Some(modes(TextMasMode::Sequential, true, true, true))),
        ] {
            h.mode.borrow_mut().apply_connection_state_snapshot(snapshot("parent", mode));
            for focused in [false, true] {
                bar.line.set_focused(focused);
                for width in [0, 1, 2, 8, 20, 30, 40, 80, 110, 160] {
                    let rows = bar.render(width as f64);
                    assert!(rows.iter().all(|row| visible_width(row) <= width), "{style}/{name}/{width}: {rows:?}");
                    let text = strip_ansi(&rows.join("\n"));
                    assert!(!text.contains("CONFIG") && !text.contains("CFG")
                        && !text.contains("OCC3.1") && !text.contains("OCC3.2 on")
                        && !text.contains("OCC3.2 off") && !text.contains("OCC3.2 blocked"),
                        "{style}/{name}/{width}: configuration leaked into the tray: {text}");
                    if width >= 80 {
                        assert!(text.contains("OCC3.2"), "{style}/{name}/{width}: delivery light missing: {text}");
                        assert!(text.contains("○ TXT  ○ OCC1  ○ OCC2  ○ OCC3.2"), "{style}/{name}/{width}: {text}");
                        assert!(text.contains("STREAM TPS —  CTX HEALTH GREEN 100"), "{text}");
                        if width >= 110 { assert!(text.contains("Jev"), "{text}"); }
                        let title = strip_ansi(&rows[0]);
                        assert!(!title.contains("OCC3.1") && !title.contains("OCC3.2"),
                            "{style}/{name}/{width}: title carries modes: {title}");
                    }
                    if !focused { capture(&format!("{style}-{name}-{width}"), &rows); }
                }
            }
        }
    }
}

#[test]
fn occ3_mode_bar_refresh_and_session_change_stay_configuration_free() {
    let h = super::super::ui_tests::FrameHarness::new("mode-refresh");
    let mut bar = Bar::new(h.mode.clone());
    for (session, mode) in [
        ("old", Some(modes(TextMasMode::Sequential, true, true, true))),
        ("old", Some(modes(TextMasMode::Hierarchical, true, false, false))),
        ("old", None),
        ("new", Some(modes(TextMasMode::Off, false, false, false))),
    ] {
        h.mode.borrow_mut().apply_connection_state_snapshot(snapshot(session, mode));
        let rows = bar.render(110.0);
        let text = strip_ansi(&rows.join("\n"));
        assert!(!text.contains("CONFIG") && !text.contains("OCC3.1"), "bottom row: {text}");
        assert!(text.contains("OCC3.2"), "delivery light lost: {text}");
        assert!(bar.pulse_deadline.is_none());
    }
}

/// The OCC3.2 light blinks on a completed native-use receipt only: never on
/// enablement, queueing, or a receipt-less native claim. Config states and
/// queue badges never light the delivery slots.
#[test]
fn occ3_mode_bar_uses_delivery_pulses_only() {
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
    let text = strip_ansi(&rows.join("\n"));
    assert!(!text.contains("OCC3.1"), "configuration leaked: {text}");
}

#[test]
fn occ3_mode_bar_fullscreen_paint_preserves_delivery_tps_context_and_draft() {
    for style in ["prime", "neon"] {
        let h = super::super::ui_tests::FrameHarness::new("mode-fullscreen");
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        h.mode.borrow_mut().apply_connection_state_snapshot(snapshot("parent", Some(
            modes(TextMasMode::Sequential, true, false, false),
        )));
        h.editor.borrow_mut().editor_mut().set_text("preserve this draft");
        let mut plain_bar = Bar::new(h.mode.clone());
        *plain_bar.roster.lock().unwrap() = Some(roster());
        let bar = Rc::new(RefCell::new(plain_bar));
        h.transcript.borrow_mut().subagents = Some(bar);
        h.ui.borrow_mut().exit_fullscreen(pi_tui::tui::ExitFullscreenOptions {
            flush: false, leave_alt_screen: false,
        });
        super::super::native_settings::fullscreen(true, &h.mode, &h.editor, &h.ui, &h.transcript);
        for width in [40, 80, 110, 160] {
            for height in [18, 24, 32] {
                h.width.set(width);
                h.height.set(height);
                h.ui.borrow_mut().request_render_forced();
                let frame = h.paint();
                assert!(frame.iter().all(|row| visible_width(row) <= width), "{style}/{width}/{height}: {frame:?}");
                let text = frame.join("\n");
                let plain = strip_ansi(&text);
                assert!(!plain.contains("CONFIG") && !plain.contains("OCC3.1"),
                    "{style}/{width}/{height}: configuration leaked: {plain}");
                for label in ["STREAM TPS", "CTX", "preserve this draft"] {
                    assert!(text.contains(label), "{style}/{width}/{height} missing {label}: {text}");
                }
                if width >= 80 {
                    assert!(text.contains("○ TXT  ○ OCC1  ○ OCC2  ○ OCC3.2"),
                        "{style}/{width}/{height} missing delivery lights: {text}");
                }
                capture(&format!("fullscreen-{style}-{width}-{height}"), &frame);
            }
        }
    }
}
