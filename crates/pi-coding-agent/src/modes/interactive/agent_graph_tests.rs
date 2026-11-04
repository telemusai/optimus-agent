use super::*;
use pi_tui::utils::{strip_ansi, visible_width};
use serde_json::json;

fn parent(id: &str) -> RosterParent {
    RosterParent {
        session_id: Some(id.into()),
        active_session_id: Some(format!("active-{id}")),
        ..Default::default()
    }
}
fn row(id: &str, owner: Option<&str>, working: bool) -> SessionSummary {
    SessionSummary {
        id: id.into(),
        session_id: id.into(),
        active_session_id: Some(format!("active-{id}")),
        parent_session_id: owner.map(str::to_string),
        lifecycle: "live".into(),
        runtime_kind: Some(
            if owner.is_some() {
                "subagent"
            } else {
                "top-level"
            }
            .into(),
        ),
        activity: if working { "working" } else { "idle" }.into(),
        extra: [
            (
                "communicationActivity".into(),
                json!({"generation":"g", "deliveries":[]}),
            ),
            ("communicationObservationEpoch".into(), json!(1)),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    }
}
fn deliver(row: &mut SessionSummary, id: &str, from: &str) {
    row.extra.get_mut("communicationActivity").unwrap()["deliveries"] = json!([{
        "id":id, "sourceSessionId":from, "kind":"text", "timestamp":chrono::Utc::now().timestamp_millis(),
    }]);
}
/// A completed native OCC-3 handoff reports a `latent` delivery: it must
/// pulse the OCC3 indicator and flash the sender->receiver edge in the
/// top-right agents graph exactly like an ordinary message link.
fn deliver_latent(row: &mut SessionSummary, id: &str, from: &str) {
    row.extra.get_mut("communicationActivity").unwrap()["deliveries"] = json!([{
        "id":id, "sourceSessionId":from, "kind":"latent", "timestamp":chrono::Utc::now().timestamp_millis(),
    }]);
}

#[test]
fn graph_uses_nested_roster_links_and_live_work_without_cross_session_nodes() {
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![
        row("nested", Some("child"), true),
        row("root", None, true),
        row("child", Some("root"), false),
        row("foreign", None, true),
        row("foreign-child", Some("foreign"), true),
    ];
    graph.observe(&rows, &parent("root"), true, now);
    assert_eq!(
        graph
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        ["root", "child", "nested"]
    );
    assert_eq!(graph.nodes[2].parent.as_deref(), Some("child"));
    assert!(graph.nodes[0].working && graph.nodes[2].working);
    assert!(!graph.nodes[1].working);
    rows[0].lifecycle = "saved".into();
    graph.observe(&rows, &parent("root"), false, now);
    assert!(graph.nodes.iter().all(|node| !node.working));
    rows.retain(|row| row.session_id != "nested");
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.nodes.len(), 2);
    graph.observe(&rows, &parent("foreign"), false, now);
    assert_eq!(
        graph
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        ["foreign", "foreign-child"]
    );
}

#[test]
fn graph_flashes_the_edge_and_occ3_indicator_for_a_completed_native_handoff() {
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![
        row("root", None, false),
        row("a", Some("root"), false),
        row("b", Some("root"), false),
    ];
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 0);
    // A completed native OCC-3 handoff from sibling `a` to sibling `b`
    // reports one latent delivery on the receiver's roster row.
    deliver_latent(&mut rows[2], "occ3-a-to-b", "a");
    graph.observe(&rows, &parent("root"), false, now);
    let links = graph
        .activity
        .live_links(now)
        .map(|(from, to, _)| (from, to))
        .collect::<Vec<_>>();
    assert_eq!(links, [("a", "b")]);
    assert_eq!(
        strip_ansi(&graph.activity.indicators(now)),
        "○ TXT  ○ OCC1  ○ OCC2  ● OCC3"
    );
    // The link expires with the pulse window like every other message edge.
    graph.observe(&rows, &parent("root"), false, now + PULSE);
    assert_eq!(graph.activity.live_links(now + PULSE).count(), 0);
}

#[test]
fn graph_messages_light_exact_edges_expire_and_never_replay_on_attach_or_reconnect() {
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![
        row("root", None, false),
        row("a", Some("root"), false),
        row("b", Some("root"), false),
        row("nested", Some("a"), false),
    ];
    deliver(&mut rows[1], "historical", "root");
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 0);
    deliver(&mut rows[2], "sibling", "a");
    deliver(&mut rows[3], "nested-task", "a");
    graph.observe(&rows, &parent("root"), false, now);
    let links = graph
        .activity
        .live_links(now)
        .map(|(a, b, _)| (a, b))
        .collect::<Vec<_>>();
    assert_eq!(links, [("a", "b"), ("a", "nested")]);
    graph.observe(
        &rows,
        &parent("root"),
        false,
        now + Duration::from_millis(900),
    );
    assert_eq!(graph.activity.live_links(now + PULSE).count(), 0);
    rows[0]
        .extra
        .insert("communicationObservationEpoch".into(), json!(2));
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 0);
    deliver(&mut rows[1], "unrelated", "foreign");
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 0);
    deliver(&mut rows[1], "fresh", "root");
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 1);
    rows[1].extra.get_mut("communicationActivity").unwrap()["generation"] = json!("restarted");
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 0);
    for row in &mut rows {
        row.extra.remove("communicationActivity");
    }
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.activity.live_links(now).count(), 0);
}

#[test]
fn graph_preserves_positions_on_roster_reorder_and_repaint_and_sleeps_when_settled() {
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![
        row("root", None, false),
        row("a", Some("root"), false),
        row("b", Some("a"), false),
    ];
    graph.observe(&rows, &parent("root"), false, now);
    graph.nodes[1].x = 0.75;
    graph.nodes[1].y = 0.35;
    let initial = graph.nodes.iter().map(|n| (n.x, n.y)).collect::<Vec<_>>();
    graph.render("root", 24, 3, now);
    let first = graph
        .nodes
        .iter()
        .map(|n| (n.x, n.y, n.vx, n.vy))
        .collect::<Vec<_>>();
    for _ in 0..20 {
        graph.render("root", 24, 3, now);
    }
    assert_eq!(
        first,
        graph
            .nodes
            .iter()
            .map(|n| (n.x, n.y, n.vx, n.vy))
            .collect::<Vec<_>>(),
        "input repaints without elapsed time must not advance or damp the layout"
    );
    for frame in 0..90 {
        let time = now + FRAME * frame;
        graph.render("root", 24, 3, time);
    }
    let settled = graph.nodes.iter().map(|n| (n.x, n.y)).collect::<Vec<_>>();
    assert_ne!(
        initial, settled,
        "spring/repulsion forces must actually move nodes"
    );
    assert!(graph.next_frame(now + SETTLE).is_none());
    rows.reverse();
    graph.observe(&rows, &parent("root"), false, now + SETTLE);
    graph.render("root", 24, 3, now + SETTLE);
    assert_eq!(
        settled,
        graph.nodes.iter().map(|n| (n.x, n.y)).collect::<Vec<_>>()
    );
    assert!(graph.nodes.iter().all(|n| n.x.is_finite()
        && n.y.is_finite()
        && (0.0..=1.0).contains(&n.x)
        && (0.0..=1.0).contains(&n.y)));
    rows.push(row("new", Some("root"), true));
    graph.observe(&rows, &parent("root"), false, now + SETTLE);
    assert!(graph.next_frame(now + SETTLE).is_some());
    assert_eq!(graph.nodes[0].x, 0.5);
}

#[test]
fn graph_resizes_bounds_large_families_and_handles_missing_metadata_and_cycles() {
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![row("root", None, false)];
    rows.extend((0..200).map(|i| row(&format!("child-{i:03}"), Some("root"), i % 2 == 0)));
    rows[0].parent_session_id = Some("child-000".into());
    rows[0].runtime_kind = Some("subagent".into());
    for row in &mut rows {
        row.extra.clear();
    }
    graph.observe(&rows, &parent("root"), false, now);
    assert_eq!(graph.nodes.len(), MAX_NODES);
    assert_eq!(graph.omitted, 137);
    assert_eq!(graph.activity.live_links(now).count(), 0);
    for (width, height) in [(0, 0), (5, 3), (8, 1), (24, 3), (24, 10), (40, 12), (40, 100)] {
        let rows = graph.render("root", width, height, now);
        assert!(rows.len() <= height);
        assert!(rows.iter().all(|row| visible_width(row) <= width));
        if width >= 8 {
            assert!(strip_ansi(&rows.join("\n")).contains("+137"));
        }
    }
    assert!(graph.render("different-session", 24, 3, now).is_empty());
}

#[test]
fn graph_frame_fixture_shows_neon_blue_work_and_communication_beside_resources() {
    use super::super::super::native_neon::{graph_height, render_header, HeaderData};
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![
        row("root", None, false),
        row("reader", Some("root"), true),
        row("tests", Some("root"), false),
        row("review", Some("reader"), true),
        row("docs", Some("root"), false),
    ];
    graph.observe(&rows, &parent("root"), false, now);
    for frame in 0..70 {
        graph.render("root", 24, 4, now + FRAME * frame);
    }
    let time = now + Duration::from_secs(6);
    deliver(&mut rows[2], "handoff", "reader");
    graph.observe(&rows, &parent("root"), false, time);
    let capture = std::env::var("OPTIMUS_GRAPH_CAPTURE_DIR").ok();
    for (name, time) in [
        ("active", time + Duration::from_millis(160)),
        ("expired", time + PULSE),
    ] {
        let resources = vec![
            "\x1b[1;38;5;39mMAIN CPU 12% RAM 512M TPS 50.0\x1b[22;39m".into(),
            "\x1b[1;38;5;39mreader CPU 8% RAM 512M TPS 42.0\x1b[22;39m".into(),
        ];
        for width in [80, 110, 120, 160, 200] {
            let mut data = HeaderData {
                cwd: "/workspace", session: "Live agents", model: "offline-fixture",
                execution: None, phase: "WORKING", jev: Some("Jev standard mode"),
                clock: "12:34:56", resources: &resources, graph: &[],
            };
            let original = render_header(width, 12, &data);
            let graph_rows = graph.render("root", 24, graph_height(width, 12, &original).max(3), time);
            assert!(graph_rows.iter().all(|row| visible_width(row) == 24));
            assert!(graph_rows.join("").contains("38;5;39m"));
            let plain = strip_ansi(&graph_rows.join(""));
            assert_eq!(plain.matches('◆').count(), 1);
            assert_eq!(plain.matches('●').count(), 2);
            assert_eq!(plain.matches('○').count(), 2);
            data.graph = &graph_rows;
            let header = render_header(width, 12, &data);
            assert!(header.iter().all(|row| visible_width(row) == width));
            if width >= 110 {
                let first = strip_ansi(&header[0]);
                let metric = strip_ansi(&header[1]);
                assert_eq!(
                    visible_width(&first[..first.find("telemus.ai").unwrap()]),
                    visible_width(&metric[..metric.find("MAIN CPU").unwrap()])
                );
                assert!(header[..graph_rows.len()].iter().any(|row| strip_ansi(row)
                    .chars()
                    .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))));
            }
            if let Some(dir) = &capture {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(format!("{dir}/{name}-{width}.ansi"), header.join("\n")).unwrap();
                std::fs::write(
                    format!("{dir}/{name}-{width}.txt"),
                    strip_ansi(&header.join("\n")),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn graph_centers_sparse_and_symmetric_families_in_the_existing_rows() {
    let now = Instant::now();
    for height in [3, 4, 5] {
        let mut graph = Graph::default();
        let mut rows = vec![row("root", None, false)];
        graph.observe(&rows, &parent("root"), false, now);
        let frame = graph.render("root", 24, height, now);
        assert_eq!(frame.len(), height);
        assert_eq!(frame.iter().position(|line| line.contains('◆')), Some(height / 2));
        rows.extend((0..12).map(|i| row(&format!("agent-{i:02}"), Some("root"), false)));
        graph.observe(&rows, &parent("root"), false, now);
        for i in 0..80 { graph.render("root", 24, height, now + FRAME * i); }
        let frame = graph.render("root", 24, height, now + SETTLE);
        let counts: Vec<_> = frame.iter().map(|line| strip_ansi(line).matches('○').count()).collect();
        assert_eq!(counts.iter().sum::<usize>(), 12, "centering must retain every node");
        for i in 0..height / 2 {
            assert_eq!(counts[i], counts[height - i - 1], "symmetric agents must balance above and below the center: {counts:?}");
        }
        assert!(frame.iter().all(|line| visible_width(line) == 24));
        assert_eq!(frame.iter().position(|line| line.contains('◆')), Some(height / 2));
    }
}

#[test]
fn graph_is_centered_between_the_header_top_and_horizontal_transcript_border() {
    use super::super::super::native_neon::{graph_height, render_header, HeaderData};
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let resources = vec!["\x1b[1;38;5;39mMAIN CPU 12% RAM 512M TPS 50.0\x1b[22;39m".into(),
        "\x1b[1;38;5;39mworker CPU 8% RAM 512M TPS 42.0\x1b[22;39m".into()];
    for width in [110, 160, 221] {
        for samples in [&resources[..], &[][..]] {
            let mut data = HeaderData { cwd: "/workspace", session: "Graph",
                model: if width < 160 { "fixture" } else { "claude-opus-5-5" },
                execution: (width >= 160).then_some("IPython [F6]"), phase: "READY", jev: Some("Jev Full"),
                clock: "12:34:56", resources: samples, graph: &[] };
            let original = render_header(width, 20, &data);
            let height = graph_height(width, 20, &original);
            assert_eq!(height, original.len() - 1, "use every row above the horizontal border");
            let now = Instant::now();
            let mut graph = Graph::default();
            let mut rows = vec![row("root", None, false)];
            rows.extend((0..12).map(|i| row(&format!("agent-{i:02}"), Some("root"), false)));
            graph.observe(&rows, &parent("root"), false, now);
            for i in 0..80 { graph.render("root", 24, height, now + FRAME * i); }
            let frame = graph.render("root", 24, height, now + SETTLE);
            data.graph = &frame;
            let header = render_header(width, 20, &data);
            assert_eq!(header.len(), original.len());
            assert_eq!(header.last(), original.last());
            let occupied: Vec<_> = header.iter().enumerate()
                .filter_map(|(i, line)| (line.contains('○') || line.contains('◆')).then_some(i)).collect();
            assert_eq!(occupied.first(), Some(&0));
            assert_eq!(occupied.last(), Some(&(height - 1)), "graph must reach the model row above the border");
            assert_eq!(strip_ansi(&header.join("\n")).matches('○').count(), 12);
            assert!(header[height - 1].contains(data.model));
            if let Some(execution) = data.execution { assert!(header[height - 1].contains(execution)); }
            assert!(header[height - 1].contains("Jev Full"));
            let metadata = strip_ansi(&original[height - 1]);
            let model_column = visible_width(&metadata[..metadata.find('●').unwrap()]);
            let status = |line: &str| pi_tui::utils::extract_segments(line, 0, model_column,
                width - model_column, false).after;
            assert_eq!(status(&header[height - 1]), status(&original[height - 1]),
                "graph overlay must preserve model and execution colours at its boundary");
            if !samples.is_empty() {
                if let Ok(dir) = std::env::var("OPTIMUS_GRAPH_CAPTURE_DIR") {
                    std::fs::create_dir_all(&dir).unwrap();
                    std::fs::write(format!("{dir}/balanced-{width}.ansi"), header.join("\n")).unwrap();
                }
            }
        }
    }
}

#[test]
fn compact_graph_balances_siblings_and_keeps_nested_branches_separate() {
    let now = Instant::now();
    let mut graph = Graph::default();
    let mut rows = vec![row("root", None, false)];
    for branch in ["a", "b", "c", "d"] {
        rows.push(row(branch, Some("root"), false));
        for leaf in ["1", "2"] {
            rows.push(row(&format!("{branch}-{leaf}"), Some(branch), false));
        }
    }
    graph.observe(&rows, &parent("root"), false, now);
    let positions = |g: &Graph| g.nodes.iter().map(|n| (n.id.clone(), (n.x, n.y))).collect::<BTreeMap<_, _>>();
    let seeded = positions(&graph);
    let mut reversed = Graph::default();
    rows.reverse();
    reversed.observe(&rows, &parent("root"), false, now);
    assert_eq!(seeded, positions(&reversed), "roster order must not change branch sectors");
    for frame in 0..80 {
        graph.render("root", 24, 4, now + FRAME * frame);
    }
    let frame = graph.render("root", 24, 4, now + SETTLE);
    assert_eq!(frame.len(), 4);
    let markers: Vec<_> = frame.iter().enumerate().flat_map(|(y, line)| {
        strip_ansi(line).chars().enumerate().filter_map(move |(x, c)| matches!(c, '◆' | '○' | '●').then_some((x as i32, y as i32))).collect::<Vec<_>>()
    }).collect();
    assert_eq!(markers.len(), rows.len());
    assert!(markers.iter().map(|p| p.1).max().unwrap() - markers.iter().map(|p| p.1).min().unwrap() >= 3);
    for (i, &(x, y)) in markers.iter().enumerate() {
        for &(other_x, other_y) in &markers[i + 1..] {
            assert!((x - other_x).abs() + (y - other_y).abs() * 2 >= 2, "markers need a readable gap");
        }
    }
    for node in graph.nodes.iter().skip(1) {
        let branch = node.id.split('-').next().unwrap();
        let owner = &seeded[branch];
        assert_eq!((node.x - 0.5).is_sign_positive(), (owner.0 - 0.5).is_sign_positive());
        assert_eq!((node.y - 0.5).is_sign_positive(), (owner.1 - 0.5).is_sign_positive());
    }
    let settled = positions(&graph);
    rows.push(row("a-new", Some("a"), true));
    graph.observe(&rows, &parent("root"), false, now + SETTLE);
    for (id, position) in settled { assert_eq!(positions(&graph)[&id], position, "roster updates must animate, not jump"); }
    assert!(graph.next_frame(now + SETTLE).is_some());
}
