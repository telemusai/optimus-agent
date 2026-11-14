use super::tests::{parent, row};
use super::*;
use crate::core::occ::textmas::{TextMasCallStatus, TextMasMode, TextMasRole, TextMasUsageRecord};
use crate::core::occ::textmas_activity::{
    project_textmas_activity, textmas_call_id, TextMasActivityCall, TextMasActivitySnapshot,
    TextMasActivityStatus, TEXTMAS_ACTIVITY_SCHEMA,
};
use pi_tui::utils::{strip_ansi, visible_width};

pub(crate) fn sequential_fixture() -> TextMasActivitySnapshot {
    fixture(
        Some(TextMasMode::Sequential),
        12,
        1,
        TextMasActivityStatus::Running,
    )
}

fn fixture(
    mode: Option<TextMasMode>,
    mappers: usize,
    reducers: usize,
    status: TextMasActivityStatus,
) -> TextMasActivitySnapshot {
    let active = matches!(
        status,
        TextMasActivityStatus::Queued | TextMasActivityStatus::Running
    );
    let mut calls = Vec::new();
    for (index, role) in TextMasRole::ALL.into_iter().enumerate() {
        let depends_on = if mode == Some(TextMasMode::Sequential) && index > 0 {
            vec![textmas_call_id(
                "action-1",
                TextMasRole::ALL[index - 1],
                None,
            )]
        } else {
            Vec::new()
        };
        let mut call = TextMasActivityCall::queued("action-1", role, None, None, depends_on);
        call.status = status;
        calls.push(call);
    }
    for index in 0..mappers {
        let mut call = TextMasActivityCall::queued(
            "action-1",
            TextMasRole::Mapper,
            Some(index as u32),
            Some(0),
            vec![textmas_call_id("action-1", TextMasRole::Refiner, None)],
        );
        call.status = status;
        calls.push(call);
    }
    for index in 0..reducers {
        let depends_on = if index == 0 {
            (0..mappers)
                .map(|index| textmas_call_id("action-1", TextMasRole::Mapper, Some(index as u32)))
                .collect()
        } else {
            vec![textmas_call_id(
                "action-1",
                TextMasRole::Reducer,
                Some((index - 1) as u32),
            )]
        };
        let mut call = TextMasActivityCall::queued(
            "action-1",
            TextMasRole::Reducer,
            Some(index as u32),
            Some(1 + index as u32 / 2),
            depends_on,
        );
        call.status = status;
        calls.push(call);
    }
    let snapshot = TextMasActivitySnapshot {
        schema: TEXTMAS_ACTIVITY_SCHEMA.into(),
        session_id: "root".into(),
        action_id: "action-1".into(),
        mode,
        active,
        stale: status == TextMasActivityStatus::Stale,
        calls,
    };
    assert!(snapshot.is_valid());
    snapshot
}

fn graph_with_activity(
    snapshot: Option<&TextMasActivitySnapshot>,
) -> (Graph, Vec<SessionSummary>, Instant) {
    let now = Instant::now();
    let mut graph = Graph::default();
    let rows = vec![row("root", None, false), row("child", Some("root"), false)];
    graph.observe(&rows, &parent("root"), false, now);
    graph.observe_activity(snapshot, now);
    (graph, rows, now)
}

// Keep genuine cohort strokes but never connect the independent Critic marker
// to Refiner through an unbroken braille line (the reported false edge).
fn assert_hierarchical_roles_have_no_rendered_critic_refiner_link(row: &str) {
    let critic = row
        .find('C')
        .unwrap_or_else(|| panic!("missing Critic marker: {row}"));
    let refiner = row
        .find('R')
        .unwrap_or_else(|| panic!("missing Refiner marker: {row}"));
    assert!(critic < refiner, "unexpected primary-role layout: {row}");
    let between = &row[critic + 1..refiner];
    let connected = between
        .chars()
        .next()
        .is_some_and(|glyph| ('\u{2800}'..='\u{28ff}').contains(&glyph));
    assert!(
        !connected,
        "hierarchical C-R continuity was rendered without that dependency: {between:?} in {row}"
    );
}

#[test]
fn hierarchical_render_does_not_imply_a_critic_refiner_dependency() {
    let snapshot = fixture(
        Some(TextMasMode::Hierarchical),
        12,
        1,
        TextMasActivityStatus::Completed,
    );
    let (mut graph, _, now) = graph_with_activity(Some(&snapshot));
    let nodes = graph.textmas.nodes();
    let critic = nodes.iter().find(|node| node.glyph == 'C').unwrap();
    let refiner = nodes.iter().find(|node| node.glyph == 'R').unwrap();
    assert!(!refiner.dependencies.contains(&critic.id));
    assert!(
        nodes
            .iter()
            .any(|node| node.glyph == 'M' && node.dependencies.contains(&refiner.id)),
        "the genuine Refiner-to-Mapper dependency must remain recorded"
    );
    for height in 2..=6 {
        let rendered = strip_ansi(
            &graph
                .render("root", 24, height, now + Duration::from_millis(1200))
                .join("\n"),
        );
        let first = rendered.lines().next().unwrap();
        assert_hierarchical_roles_have_no_rendered_critic_refiner_link(first);
    }
}

fn root_only_graph(snapshot: &TextMasActivitySnapshot) -> (Graph, Instant) {
    let now = Instant::now();
    let mut graph = Graph::default();
    let rows = vec![row("root", None, false)];
    graph.observe(&rows, &parent("root"), false, now);
    graph.observe_activity(Some(snapshot), now);
    (graph, now)
}

fn constrained_reducer_fixture() -> TextMasActivitySnapshot {
    let mut snapshot = fixture(
        Some(TextMasMode::Hierarchical),
        2,
        1,
        TextMasActivityStatus::Completed,
    );
    let reducer = textmas_call_id("action-1", TextMasRole::Reducer, Some(0));
    for call in &mut snapshot.calls {
        if matches!(
            call.role,
            TextMasRole::Planner | TextMasRole::Critic | TextMasRole::Refiner
        ) {
            call.depends_on = vec![reducer.clone()];
        } else if call.role == TextMasRole::Mapper {
            call.depends_on.clear();
        }
        call.status = match call.role {
            TextMasRole::Mapper if call.call_index == Some(0) => TextMasActivityStatus::Running,
            TextMasRole::Reducer => TextMasActivityStatus::Queued,
            _ => TextMasActivityStatus::Completed,
        };
    }
    snapshot.active = true;
    assert!(snapshot.is_valid());
    snapshot
}

#[test]
fn constrained_sigma_critic_route_does_not_read_as_critic_refiner() {
    let snapshot = constrained_reducer_fixture();
    let (mut graph, now) = root_only_graph(&snapshot);
    let nodes = graph.textmas.nodes();
    let critic = nodes.iter().find(|node| node.glyph == 'C').unwrap();
    let refiner = nodes.iter().find(|node| node.glyph == 'R').unwrap();
    let reducer = nodes.iter().find(|node| node.glyph == 'Σ').unwrap();
    let mappers = nodes
        .iter()
        .filter(|node| node.glyph == 'M')
        .collect::<Vec<_>>();
    assert_eq!(mappers.len(), 2);
    assert_eq!(mappers[0].index_range, Some((0, 0)));
    assert_eq!(mappers[1].index_range, Some((1, 1)));
    assert!(critic.dependencies.contains(&reducer.id));
    assert!(!refiner.dependencies.contains(&critic.id));
    assert_eq!(reducer.dependencies.len(), 2);
    assert!(mappers
        .iter()
        .all(|mapper| reducer.dependencies.contains(&mapper.id)));
    assert!(!reducer.dependencies.contains(&critic.id));

    let graph_rows = graph.render("root", 24, 2, now + Duration::from_millis(1200));
    let plain = strip_ansi(&graph_rows.join("\n"));
    let rendered = plain.lines().collect::<Vec<_>>();
    assert_eq!(
        rendered,
        vec![
            "    P       C       R   ",
            "    M      ◆M⠤⠤⠤⠤⠤⠤⠤Σ   ",
        ],
        "constrained routing must keep the real M1-to-Sigma stroke without an ambiguous C-to-R presentation"
    );
    assert_hierarchical_roles_have_no_rendered_critic_refiner_link(rendered[0]);

    let detailed = graph.render("root", 40, 6, now + Duration::from_millis(1200));
    assert_eq!(detailed.len(), 6);
    assert!(detailed.iter().all(|line| visible_width(line) == 40));
    let details = strip_ansi(&detailed.join("\n"));
    assert!(details.contains("4/6"), "{details}");
    assert!(details.contains("M0-0 q0r1 ✓0"), "{details}");
    assert!(details.contains("M1-1 q0r0 ✓1"), "{details}");
    assert!(details.contains("Σ[L1]:1 q1r0 ✓0"), "{details}");
    assert!(graph.textmas.is_active());

    let recovered_snapshot = fixture(None, 2, 1, TextMasActivityStatus::Completed);
    let (mut recovered, recovered_now) = root_only_graph(&recovered_snapshot);
    assert_eq!(recovered.textmas.nodes().len(), 1);
    assert!(recovered.textmas.nodes()[0].dependencies.is_empty());
    let recovered_frame = strip_ansi(
        &recovered
            .render("root", 40, 3, recovered_now + Duration::from_millis(1200))
            .join("\n"),
    );
    assert!(
        recovered_frame.contains("T? unknown 6/6"),
        "{recovered_frame}"
    );
}

#[test]
fn genuine_adjacent_primary_dependencies_remain_visible() {
    let (planner, critic, refiner) = ((8, 2), (24, 2), (40, 2));
    let mut canvas = Canvas::new(24, 1);
    canvas.activity_dependency(planner, critic, &BTreeSet::from([(20, 0)]));
    canvas.activity_dependency(critic, refiner, &BTreeSet::from([(4, 0)]));
    for (point, glyph) in [(planner, 'P'), (critic, 'C'), (refiner, 'R')] {
        canvas.marker_green(point.0, point.1, glyph, 2);
    }
    let rendered = strip_ansi(&canvas.render().join("\n"));
    let row = rendered.lines().next().unwrap();
    for (from, to) in [('P', 'C'), ('C', 'R')] {
        let start = row.find(from).unwrap();
        let end = row.find(to).unwrap();
        assert!(start < end, "{row}");
        assert!(
            row[start + 1..end]
                .chars()
                .all(|glyph| ('\u{2800}'..='\u{28ff}').contains(&glyph)),
            "genuine {from}-to-{to} adjacency was suppressed: {row}"
        );
    }
}

#[test]
fn roles_and_recorded_edges_distinguish_sequential_from_hierarchical_runs() {
    for (mode, sequential) in [
        (TextMasMode::Sequential, true),
        (TextMasMode::Hierarchical, false),
    ] {
        let snapshot = fixture(Some(mode), 0, 0, TextMasActivityStatus::Completed);
        let (mut graph, _, now) = graph_with_activity(Some(&snapshot));
        let nodes = graph.textmas.nodes();
        let roles = nodes.iter().map(|node| node.glyph).collect::<String>();
        assert_eq!(roles, "PCR");
        for (index, expected) in [0, 1, 1].into_iter().enumerate() {
            assert_eq!(
                nodes[index].dependencies.len(),
                if sequential { expected } else { 0 }
            );
        }
        assert_eq!(
            strip_ansi(&graph.render("root", 24, 5, now).join("\n")).contains("P✓ C✓ R✓"),
            sequential || !sequential
        );
    }
}

#[test]
fn mapper_cohorts_preserve_real_ranges_levels_dependencies_and_roster_capacity() {
    let mut rows = vec![row("root", None, false)];
    rows.extend((0..200).map(|index| row(&format!("real-agent-{index:03}"), Some("root"), false)));
    let mut graph = Graph::default();
    let now = Instant::now();
    graph.observe(&rows, &parent("root"), false, now);
    let snapshot = fixture(
        Some(TextMasMode::Sequential),
        64,
        4,
        TextMasActivityStatus::Queued,
    );
    graph.observe_activity(Some(&snapshot), now);
    assert_eq!(graph.nodes.len(), 64);
    assert_eq!(
        graph.omitted, 137,
        "TextMAS calls are never roster capacity or omissions"
    );
    let mapper_nodes = graph
        .textmas
        .nodes()
        .iter()
        .filter(|node| node.glyph == 'M')
        .collect::<Vec<_>>();
    let reducer_nodes = graph
        .textmas
        .nodes()
        .iter()
        .filter(|node| node.glyph == 'Σ')
        .collect::<Vec<_>>();
    assert!(mapper_nodes.len() <= 3 && reducer_nodes.len() <= 3);
    assert_eq!(
        mapper_nodes
            .iter()
            .map(|node| node.counts.total)
            .sum::<usize>(),
        64
    );
    assert_eq!(
        reducer_nodes
            .iter()
            .map(|node| node.counts.total)
            .sum::<usize>(),
        4
    );
    assert!(mapper_nodes
        .iter()
        .any(|node| node.index_range == Some((0, 21))));
    assert!(reducer_nodes
        .iter()
        .flat_map(|node| node.levels.iter().copied())
        .collect::<BTreeSet<_>>()
        .is_superset(&BTreeSet::from([1, 2])));
    let mappers = mapper_nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(reducer_nodes
        .iter()
        .flat_map(|node| node.dependencies.iter())
        .any(|id| mappers.contains(id.as_str())));
    let short = reducer_nodes
        .iter()
        .map(|node| node.short_label())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        short.contains("L1") && short.contains("L2"),
        "recorded reduction levels stay visible when space permits: {short}"
    );
    for height in 3..=6 {
        let frame = graph.render("root", 24, height, now);
        assert_eq!(frame.len(), height);
        assert!(frame.iter().all(|line| visible_width(line) == 24));
    }
}

#[test]
fn every_role_and_cohort_stays_green_and_exposes_all_activity_states() {
    for (status, glyph, active) in [
        (TextMasActivityStatus::Queued, 'q', true),
        (TextMasActivityStatus::Running, '…', true),
        (TextMasActivityStatus::Completed, '✓', false),
        (TextMasActivityStatus::Failed, '!', false),
        (TextMasActivityStatus::Cancelled, '×', false),
        (TextMasActivityStatus::TimedOut, 't', false),
        (TextMasActivityStatus::Skipped, 's', false),
        (TextMasActivityStatus::Stale, 'o', false),
    ] {
        let snapshot = fixture(Some(TextMasMode::Hierarchical), 12, 1, status);
        let (mut graph, _, now) = graph_with_activity(Some(&snapshot));
        let lines = graph.textmas.lines(40, now, 4);
        let text = strip_ansi(&lines.join("\n"));
        assert!(
            text.contains(&format!("P{glyph} C{glyph} R{glyph}")),
            "{status:?}: {text}"
        );
        assert!(
            text.contains("q"),
            "total/queued/running/completed/failed/cancelled counters: {text}"
        );
        assert!(
            text.contains('✓') && text.contains('!') && text.contains('×'),
            "{text}"
        );
        assert!(text.contains("12") && text.contains('Σ'), "{text}");
        assert_eq!(graph.textmas.is_active(), active);
        let frame = graph.render("root", 24, 6, now).join("\n");
        let green = format!(
            "38;5;{}m",
            super::context_health_indicator::CONTEXT_HEALTH_GREEN
        );
        assert!(
            frame.contains(&green),
            "same CTX health token for {status:?}"
        );
        assert!(
            !frame.contains("38;5;196m")
                && !frame.contains("38;5;214m")
                && !frame.contains("38;5;244m")
        );
        assert!(frame.contains("38;5;39m"), "existing blue agents stay blue");
        let expected_intensity = if status == TextMasActivityStatus::Running {
            2
        } else if matches!(
            status,
            TextMasActivityStatus::Queued
                | TextMasActivityStatus::Skipped
                | TextMasActivityStatus::Stale
        ) {
            1
        } else {
            2
        };
        assert_eq!(
            graph.textmas.nodes()[0]
                .intensity(matches!(status, TextMasActivityStatus::Running), false),
            expected_intensity
        );
        assert_eq!(
            graph.next_frame(now + SETTLE).is_none(),
            !active,
            "settled terminal activity must release the existing repaint timer"
        );
    }
}

#[test]
fn retries_replacements_late_old_actions_and_unknown_recovery_are_honest() {
    let original = fixture(
        Some(TextMasMode::Sequential),
        0,
        0,
        TextMasActivityStatus::Completed,
    );
    let (mut graph, _, now) = graph_with_activity(Some(&original));
    let t0 = graph.textmas.elapsed(now);
    let mut replacement = fixture(
        Some(TextMasMode::Hierarchical),
        4,
        0,
        TextMasActivityStatus::Running,
    );
    replacement.action_id = "action-2".into();
    for call in &mut replacement.calls {
        call.call_id = call.call_id.replace("action-1", "action-2");
        call.depends_on = call
            .depends_on
            .iter()
            .map(|id| id.replace("action-1", "action-2"))
            .collect();
    }
    assert!(replacement.is_valid());
    graph.observe_activity(Some(&replacement), now + Duration::from_secs(8));
    assert_eq!(
        graph.textmas.elapsed(now + Duration::from_secs(8)),
        Some(Duration::ZERO)
    );
    let before = graph.render("root", 24, 5, now + Duration::from_secs(8));
    graph.observe_activity_state(Some(&original), true, now + Duration::from_secs(9));
    assert_eq!(
        graph.render("root", 24, 5, now + Duration::from_secs(8)),
        before,
        "late action A cannot replace current action B"
    );
    graph.observe_activity_state(Some(&original), false, now + Duration::from_secs(10));
    let paused = strip_ansi(
        &graph
            .textmas
            .lines(24, now + Duration::from_secs(10), 3)
            .join("\\n"),
    );
    assert!(
        paused.contains("disconnected") && !paused.contains("C×"),
        "{paused}"
    );
    assert!(graph.textmas.nodes().iter().any(|node| node.glyph == 'M'));
    assert!(graph
        .textmas
        .nodes()
        .iter()
        .all(|node| node.counts.running >= 1 && node.counts.cancelled == 0));
    assert_eq!(
        graph.textmas.elapsed(now + Duration::from_secs(30)),
        Some(Duration::from_secs(2)),
        "a late old payload cannot prevent the current run's observation age from pausing"
    );
    graph.observe_activity_state(Some(&original), true, now + Duration::from_secs(30));
    assert!(!graph.textmas.is_disconnected());

    let mut retry = original.clone();
    retry.active = true;
    retry.calls[0].status = TextMasActivityStatus::Running;
    let (mut retried, _, retry_now) = graph_with_activity(Some(&original));
    retried.observe_activity(Some(&retry), retry_now + Duration::from_secs(20));
    assert_eq!(
        retried.textmas.elapsed(retry_now + Duration::from_secs(20)),
        Some(Duration::from_secs(20))
    );
    assert!(retried.textmas.is_active());
    retried.observe_activity(Some(&retry), retry_now + Duration::from_secs(21));
    assert_eq!(
        retried.textmas.last_changed(),
        Some(retry_now + Duration::from_secs(20))
    );
    assert_eq!(t0, Some(Duration::ZERO));

    let recovery = fixture(None, 8, 1, TextMasActivityStatus::Completed);
    let (mut recovered, _, recovered_now) = graph_with_activity(Some(&recovery));
    assert_eq!(recovered.textmas.nodes().len(), 1);
    assert!(recovered.textmas.nodes()[0].dependencies.is_empty());
    assert!(recovered.textmas.nodes()[0].levels.is_empty());
    assert!(recovered.textmas.nodes()[0].index_range.is_none());
    assert!(
        strip_ansi(&recovered.textmas.lines(24, recovered_now, 3).join("\n")).contains("unknown")
    );
}

#[test]
fn switching_sessions_malformed_activity_and_stale_orphans_do_not_create_agents() {
    let original = fixture(
        Some(TextMasMode::Sequential),
        12,
        1,
        TextMasActivityStatus::Running,
    );
    let (mut graph, rows, now) = graph_with_activity(Some(&original));
    graph.observe(&rows, &parent("other"), false, now + Duration::from_secs(1));
    graph.observe_activity(Some(&original), now + Duration::from_secs(1));
    assert!(graph.textmas.is_empty());
    assert!(!graph
        .render("other", 24, 5, now + Duration::from_secs(1))
        .join("\n")
        .contains("38;5;82m"));
    let mut invalid = fixture(
        Some(TextMasMode::Sequential),
        0,
        0,
        TextMasActivityStatus::Running,
    );
    invalid.calls[0].call_id = "fabricated".into();
    graph.observe_activity(Some(&invalid), now + Duration::from_secs(1));
    assert!(graph.textmas.is_empty());
    graph.observe(&rows, &parent("root"), false, now + Duration::from_secs(2));
    graph.observe_activity(Some(&original), now + Duration::from_secs(2));
    assert_eq!(graph.textmas.nodes().len(), 7);
    let previous_changed = graph.textmas.last_changed();
    let mut invalid_revision = original.clone();
    invalid_revision.calls[0]
        .depends_on
        .push("missing-dependency".into());
    assert!(!invalid_revision.is_valid());
    graph.observe_activity(Some(&invalid_revision), now + Duration::from_secs(3));
    assert_eq!(graph.textmas.nodes().len(), 7);
    assert_eq!(graph.textmas.last_changed(), previous_changed);

    // A real durable Started reservation recovers as stale on interruption. No
    // renderer or reducer relabels it Cancelled, or creates a usage record.
    let orphan = project_textmas_activity(
        "root",
        "interrupted",
        Some(TextMasMode::Sequential),
        false,
        TextMasRole::ALL
            .into_iter()
            .map(|role| TextMasActivityCall::queued("interrupted", role, None, None, Vec::new()))
            .collect(),
        vec![TextMasUsageRecord::new(
            "interrupted",
            TextMasRole::Planner,
            TextMasCallStatus::Started,
            None,
        )],
    );
    graph.observe_activity(Some(&orphan), now + Duration::from_secs(6));
    assert!(graph
        .textmas
        .nodes()
        .iter()
        .any(|node| node.counts.stale == 1));
    assert!(!graph
        .textmas
        .nodes()
        .iter()
        .any(|node| node.counts.cancelled > 0));
    assert!(!graph.textmas.is_active());
    assert!(graph.next_frame(now + Duration::from_secs(13)).is_none());
    graph.observe_activity(Some(&orphan), now + Duration::from_secs(14));
    assert!(
        graph.next_frame(now + Duration::from_secs(14)).is_none(),
        "identical reconnect/resync is a terminal no-op"
    );
}

#[test]
fn absence_removes_all_activity_space_and_bytes_match_the_base_rendering() {
    let (mut graph, rows, now) = graph_with_activity(None);
    for frame in 0..90 {
        graph.render("root", 24, 5, now + FRAME * frame);
    }
    let off = graph.render("root", 24, 5, now + SETTLE);
    let expected_off: Vec<String> = serde_json::from_str(
        r#"[
  "                        \u001b[22;39m",
  "                        \u001b[22;39m",
  "           \u001b[22;2;38;5;39m◆⠤⠤⠤⠤⠤⠤⠤⠤⠤○\u001b[22;2;38;5;39m  \u001b[22;39m",
  "                        \u001b[22;39m",
  "                        \u001b[22;39m"
]"#,
    )
    .expect("base-rendered OFF golden");
    assert_eq!(off, expected_off);
    let snapshot = fixture(
        Some(TextMasMode::Sequential),
        0,
        0,
        TextMasActivityStatus::Completed,
    );
    graph.observe_activity(Some(&snapshot), now + SETTLE);
    let active = graph.render("root", 24, 5, now + SETTLE);
    assert_eq!(active.len(), off.len());
    assert_ne!(active, off);
    graph.observe(&rows, &parent("root"), false, now + SETTLE * 2);
    graph.observe_activity(None, now + SETTLE * 2);
    let off_again = graph.render("root", 24, 5, now + SETTLE * 3);
    assert_eq!(
        off_again, off,
        "dismissing activity restores the exact original bytes"
    );
    graph.observe_activity(None, now + SETTLE * 3);
    assert!(graph.textmas.is_empty());
    assert!(graph.textmas.lines(24, now, 4).is_empty());
}

#[test]
fn real_native_renderer_captures_activity_at_existing_dimensions() {
    use super::super::super::native_neon::{graph_height, graph_space, render_header, HeaderData};
    use serde_json::Value;
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    let capture = std::env::var("OPTIMUS_GRAPH_CAPTURE_DIR").ok();
    let mut manifest: Vec<Value> = Vec::new();
    let resources = vec!["\x1b[1;38;5;39mMAIN CPU 12% RAM 512M TPS 50.0\x1b[22;39m".into()];
    let mut capture_frames =
        |phase: &str, graph: &mut Graph, at: Instant, snapshot_active: bool| {
            let animated = graph.textmas.is_active();
            let disconnected = graph.textmas.is_disconnected();
            for width in [80, 110, 120, 160, 200] {
                let mut data = HeaderData {
                    cwd: "/workspace",
                    session: "TextMAS",
                    model: "offline-fixture",
                    execution: None,
                    phase: if disconnected {
                        "DISCONNECTED"
                    } else if animated {
                        "WORKING"
                    } else {
                        "RECOVERED"
                    },
                    jev: Some("Jev standard mode"),
                    clock: "12:34:56",
                    resources: &resources,
                    graph: &[],
                };
                let original = render_header(width, 12, &data);
                let graph_width = graph_space(width, 12);
                let height = graph_height(width, 12, &original);
                let graph_rows = graph.render("root", graph_width, height, at);
                assert!(graph_rows
                    .iter()
                    .all(|line| visible_width(line) == graph_width.min(40)));
                assert!(
                    graph_rows.len() <= 6,
                    "the existing six-row canvas is unchanged"
                );
                data.graph = &graph_rows;
                let header = render_header(width, 12, &data);
                assert_eq!(header.len(), original.len());
                assert!(
                    header.iter().all(|line| visible_width(line) == width),
                    "{phase} {width}"
                );
                if width >= 110 {
                    let text = strip_ansi(&graph_rows.join("\n"));
                    if phase != "recovered-unknown" {
                        assert!(
                            text.contains('M') && text.contains('Σ'),
                            "{phase} {width}: {text}"
                        );
                    }
                    if phase.starts_with("sequential")
                        || phase.starts_with("hierarchical")
                        || phase.starts_with("reconnect")
                    {
                        assert!(
                            text.contains("P") && text.contains("C") && text.contains("R"),
                            "{text}"
                        );
                    }
                    if phase.starts_with("hierarchical") {
                        assert_hierarchical_roles_have_no_rendered_critic_refiner_link(
                            &strip_ansi(&graph_rows[0]),
                        );
                    }
                    if phase == "sequential-active" {
                        assert!(
                            text.contains("7/16") && !text.contains("16/16"),
                            "completion is a count, not overall success: {text}"
                        );
                    }
                    if phase == "failure" {
                        assert!(text.contains("C!") && text.contains("Rs"), "{text}");
                    }
                    if phase == "cancelled" {
                        assert!(text.contains("C×"), "{text}");
                    }
                    if phase == "recovered-unknown" {
                        assert!(text.contains("unknown"), "{text}");
                    }
                    if disconnected {
                        assert!(text.contains("disconnected"), "{text}");
                        assert!(
                            !text.contains("cancelled"),
                            "disconnect cannot relabel calls: {text}"
                        );
                    }
                    if animated {
                        assert!(
                            graph.next_frame(at).is_some(),
                            "reuse the existing active frame cadence"
                        );
                    } else {
                        assert!(
                            graph.next_frame(at + SETTLE).is_none(),
                            "terminal projections stop after the existing settle window"
                        );
                    }
                } else {
                    assert!(graph_rows.is_empty());
                }
                if let Some(dir) = &capture {
                    std::fs::create_dir_all(dir).unwrap();
                    let base = format!("{dir}/textmas-{phase}-{width}");
                    std::fs::write(format!("{base}.ansi"), header.join("\n")).unwrap();
                    std::fs::write(format!("{base}.txt"), strip_ansi(&header.join("\n"))).unwrap();
                    manifest.push(serde_json::json!({"phase":phase,"width":width,
                    "snapshotActive":snapshot_active,"ownerActive":graph.textmas.owner_active(),
                    "paused":disconnected,"animated":animated,
                    "ansi":format!("{base}.ansi"),"text":format!("{base}.txt")}));
                }
            }
        };
    let phases = [
        "sequential-active",
        "sequential-completed",
        "hierarchical-active",
        "hierarchical-completed",
        "mapper-cohorts",
        "failure",
        "cancelled",
        "recovered-unknown",
    ];
    for phase in phases {
        let hierarchical = phase.starts_with("hierarchical");
        let mode = if phase == "recovered-unknown" {
            None
        } else if hierarchical {
            Some(TextMasMode::Hierarchical)
        } else {
            Some(TextMasMode::Sequential)
        };
        let (map_count, reducer_count) = if phase == "mapper-cohorts" {
            (64, 4)
        } else {
            (12, 1)
        };
        let mut snapshot = fixture(
            mode,
            map_count,
            reducer_count,
            TextMasActivityStatus::Completed,
        );
        let active = phase.ends_with("active") || phase == "mapper-cohorts";
        snapshot.active = active;
        if active {
            for call in &mut snapshot.calls {
                call.status = match call.role {
                    TextMasRole::Planner => TextMasActivityStatus::Completed,
                    TextMasRole::Critic => TextMasActivityStatus::Running,
                    TextMasRole::Refiner => TextMasActivityStatus::Queued,
                    TextMasRole::Mapper if phase == "mapper-cohorts" => [
                        TextMasActivityStatus::Completed,
                        TextMasActivityStatus::Queued,
                        TextMasActivityStatus::Running,
                        TextMasActivityStatus::Failed,
                        TextMasActivityStatus::TimedOut,
                        TextMasActivityStatus::Cancelled,
                    ][call.call_index.unwrap() as usize % 6],
                    TextMasRole::Mapper => {
                        if call.call_index.unwrap() < 6 {
                            TextMasActivityStatus::Completed
                        } else {
                            TextMasActivityStatus::Queued
                        }
                    }
                    TextMasRole::Reducer => TextMasActivityStatus::Queued,
                }
            }
        } else if phase == "failure" {
            for call in &mut snapshot.calls {
                call.status = match call.role {
                    TextMasRole::Planner => TextMasActivityStatus::Completed,
                    TextMasRole::Critic => TextMasActivityStatus::Failed,
                    _ => TextMasActivityStatus::Skipped,
                };
            }
        } else if phase == "cancelled" {
            for call in &mut snapshot.calls {
                call.status = match call.role {
                    TextMasRole::Planner => TextMasActivityStatus::Completed,
                    TextMasRole::Critic => TextMasActivityStatus::Cancelled,
                    _ => TextMasActivityStatus::Skipped,
                };
            }
        } else if phase == "recovered-unknown" {
            // No plan or mode is recovered. Indexed call identity survives,
            // but no dependencies or reduction levels are invented.
            snapshot = project_textmas_activity(
                "root",
                "action-1",
                None,
                false,
                Vec::new(),
                vec![
                    TextMasUsageRecord::new_indexed(
                        "action-1",
                        TextMasRole::Planner,
                        None,
                        TextMasCallStatus::Started,
                        None,
                    ),
                    TextMasUsageRecord::new_indexed(
                        "action-1",
                        TextMasRole::Mapper,
                        Some(0),
                        TextMasCallStatus::Completed,
                        None,
                    ),
                    TextMasUsageRecord::new_indexed(
                        "action-1",
                        TextMasRole::Reducer,
                        Some(0),
                        TextMasCallStatus::Completed,
                        None,
                    ),
                ],
            );
        }
        snapshot.stale = snapshot
            .calls
            .iter()
            .any(|call| call.status == TextMasActivityStatus::Stale);
        assert!(snapshot.is_valid(), "{phase}");
        let now = Instant::now();
        let (mut graph, _, _) = graph_with_activity(Some(&snapshot));
        capture_frames(
            phase,
            &mut graph,
            now + Duration::from_millis(1200),
            snapshot.active,
        );
    }
    // Capture a real connection-state transition. The recorded calls are not changed.
    let mut snapshot = fixture(
        Some(TextMasMode::Sequential),
        12,
        1,
        TextMasActivityStatus::Running,
    );
    snapshot.action_id = "reconnect".into();
    for call in &mut snapshot.calls {
        call.call_id = call.call_id.replace("action-1", "reconnect");
        call.depends_on = call
            .depends_on
            .iter()
            .map(|id| id.replace("action-1", "reconnect"))
            .collect();
    }
    assert!(snapshot.is_valid());
    let (mut graph, _, now) = graph_with_activity(Some(&snapshot));
    let running_counts = graph
        .textmas
        .nodes()
        .iter()
        .map(|node| node.counts.running)
        .collect::<Vec<_>>();
    capture_frames(
        "reconnect-before",
        &mut graph,
        now + Duration::from_millis(1200),
        snapshot.active,
    );
    graph.observe_activity_state(Some(&snapshot), false, now + Duration::from_millis(1500));
    assert_eq!(
        graph
            .textmas
            .nodes()
            .iter()
            .map(|node| node.counts.running)
            .collect::<Vec<_>>(),
        running_counts
    );
    assert!(graph
        .textmas
        .nodes()
        .iter()
        .all(|node| node.counts.cancelled == 0));
    assert_eq!(
        graph.textmas.elapsed(now + Duration::from_secs(90)),
        Some(Duration::from_millis(1500))
    );
    capture_frames(
        "disconnected",
        &mut graph,
        now + Duration::from_secs(90),
        snapshot.active,
    );
    graph.observe_activity_state(Some(&snapshot), true, now + Duration::from_secs(91));
    assert!(graph.textmas.is_active() && !graph.textmas.is_disconnected());
    assert_eq!(
        graph.textmas.elapsed(now + Duration::from_secs(91)),
        Some(Duration::from_secs(91))
    );
    assert_eq!(
        graph
            .textmas
            .nodes()
            .iter()
            .map(|node| node.counts.running)
            .collect::<Vec<_>>(),
        running_counts
    );
    capture_frames(
        "reconnected",
        &mut graph,
        now + Duration::from_secs(91),
        snapshot.active,
    );
    if let Some(dir) = capture {
        std::fs::write(format!("{dir}/agent_graph_activity_golden.json"), serde_json::to_string_pretty(&serde_json::json!({
            "renderer":"existing native Graph plus native_neon::render_header",
            "observationAge":"The age value is time since this fixture observed the authoritative projection, not provider runtime.",
            "tokens":{"agentsBlue":39,"contextHealthAndTextMasGreen":super::context_health_indicator::CONTEXT_HEALTH_GREEN},
            "states":{"q":"queued","…":"running","✓":"completed","!":"failed","×":"cancelled","t":"timed out","s":"skipped","o":"stale/orphaned Started"},
            "captures":manifest,
        })).unwrap()).unwrap();
    }
}

#[test]
fn disconnected_running_is_paused_not_cancelled_and_reconnect_accepts_the_same_action() {
    let mut snapshot = fixture(
        Some(TextMasMode::Sequential),
        4,
        1,
        TextMasActivityStatus::Running,
    );
    let (mut graph, _, now) = graph_with_activity(Some(&snapshot));
    graph.observe_activity_state(Some(&snapshot), false, now + Duration::from_secs(3));
    assert!(!graph.textmas.is_active());
    assert!(graph.textmas.is_disconnected());
    assert_eq!(
        graph.textmas.elapsed(now + Duration::from_secs(90)),
        Some(Duration::from_secs(3)),
        "observation age freezes at the confirmed disconnect"
    );
    let paused = strip_ansi(
        &graph
            .textmas
            .lines(24, now + Duration::from_secs(4), 4)
            .join("\n"),
    );
    assert!(paused.contains("disconnected"), "{paused}");
    assert!(
        paused.contains("age3.0s"),
        "frozen observation age stays visible: {paused}"
    );
    assert!(paused.contains("P?"), "{paused}");
    assert!(graph
        .textmas
        .nodes()
        .iter()
        .all(|node| node.counts.running >= 1 && node.counts.cancelled == 0));
    graph.observe_activity_state(Some(&snapshot), true, now + Duration::from_secs(4));
    assert!(graph.textmas.is_active() && !graph.textmas.is_disconnected());
    assert_eq!(
        graph.textmas.elapsed(now + Duration::from_secs(4)),
        Some(Duration::from_secs(4))
    );
    assert_eq!(graph.textmas.last_changed(), Some(now));
    snapshot.active = false;
    snapshot.stale = true;
    for call in &mut snapshot.calls {
        call.status = TextMasActivityStatus::Stale;
    }
    assert!(snapshot.is_valid());
    graph.observe_activity_state(Some(&snapshot), false, now + Duration::from_secs(5));
    assert!(!graph.textmas.is_disconnected());
    assert!(graph
        .textmas
        .nodes()
        .iter()
        .all(|node| node.counts.stale == node.counts.total));
    assert!(graph.next_frame(now + Duration::from_secs(12)).is_none());
}

#[test]
fn reducer_levels_remain_visible_in_bounded_native_text() {
    let snapshot = fixture(
        Some(TextMasMode::Sequential),
        4,
        4,
        TextMasActivityStatus::Completed,
    );
    let (graph, _, now) = graph_with_activity(Some(&snapshot));
    let text = strip_ansi(&graph.textmas.lines(24, now, 4).join("\n"));
    assert!(text.contains("Σ[L1-2]:4"), "{text}");
}
