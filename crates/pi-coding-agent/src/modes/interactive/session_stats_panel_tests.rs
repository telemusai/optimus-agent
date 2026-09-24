use super::super::stats_data::account;
use super::*;
use serde_json::json;

fn panel() -> (StatsPanel, mpsc::Receiver<HostEvent>) {
    crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let (send, receive) = mpsc::channel();
    let mut usage = crate::core::usage::empty_usage();
    usage.input = 2000.0;
    usage.output = 100.0;
    usage.cache_read = 800.0;
    usage.total_tokens = 2900.0;
    let accounting = account(
        &[json!({"id":"a", "type":"message", "message":{
        "role":"assistant", "provider":"fixture", "model":"astra", "usage":usage}})],
        Some("a"),
    );
    let snapshot = Snapshot {
        name: "Synthetic stats fixture".into(),
        session_id: "chat".into(),
        accounting,
        context: json!({"tokens":64000,"contextWindow":200000}),
        jev_mode: "active".into(),
        ..Default::default()
    };
    (
        StatsPanel {
            data: Arc::new(Mutex::new(LiveData {
                snapshot: Some(snapshot),
                error: None,
            })),
            include_children: Arc::new(AtomicBool::new(false)),
            refresh: Arc::new(Notify::new()),
            task: None,
            send,
            page: 0,
            scroll: 0,
            rows: 40,
        },
        receive,
    )
}

#[test]
fn panel_uses_ascii_and_fits_narrow_and_short_terminals() {
    let (mut panel, _) = panel();
    for width in [1, 10, 28, 40, 80, 160] {
        for height in [4, 8, 24, 50] {
            for page in 0..4 {
                panel.page = page;
                let lines = panel.render_at(width, height);
                assert!(
                    lines.len() <= height,
                    "{width}x{height}: {} lines",
                    lines.len()
                );
                for line in lines {
                    assert!(visible_width(&line) <= width, "{width}: {line:?}");
                    assert!(pi_tui::utils::strip_ansi(&line).is_ascii());
                }
            }
        }
    }
}

#[test]
fn section_scope_and_escape_are_local_and_remappable() {
    let (mut panel, receive) = panel();
    panel.handle_input("\t");
    assert_eq!(panel.page, 1);
    panel.handle_input("s");
    assert!(panel.include_children.load(Ordering::Relaxed));
    assert!(panel
        .render_at(80, 24)
        .join("\n")
        .contains("Updating scope"));
    let mut config = crate::core::keybindings::KeybindingsConfig::new();
    config.insert(
        "tui.select.cancel".into(),
        crate::core::keybindings::KeybindingSetting::Single("ctrl+x".into()),
    );
    config.insert(
        "app.stats.toggleSubagents".into(),
        crate::core::keybindings::KeybindingSetting::Single("z".into()),
    );
    crate::core::keybindings::KeybindingsManager::new(config, None).install();
    panel.handle_input("s");
    assert!(panel.include_children.load(Ordering::Relaxed));
    panel.handle_input("z");
    assert!(!panel.include_children.load(Ordering::Relaxed));
    panel.handle_input("\x18");
    assert!(receive
        .try_iter()
        .any(|event| matches!(event, HostEvent::CloseCommandDialog)));
    crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
}

#[test]
fn missing_metrics_are_not_rendered_as_proven_savings() {
    let (mut panel, _) = panel();
    let text = panel.render_at(100, 50).join("\n");
    assert!(text.contains("Search     unavailable"));
    assert!(text.contains("Live JEV activity: unavailable"));
    assert!(!text.contains("$0.0000"));
    panel.page = 3;
    assert!(panel.render_at(100, 50).join("\n").contains("#"));
}
