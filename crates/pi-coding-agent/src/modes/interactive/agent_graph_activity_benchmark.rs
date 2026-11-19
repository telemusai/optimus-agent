use super::*;

impl Graph {
    #[inline(always)]
    fn textmas_is_empty_for_bench(&self) -> bool {
        self.textmas.is_empty()
    }
}
#[inline(always)]
fn graph_no_activity_for_bench(graph: &mut Graph, now: Instant) {
    graph.observe_activity(None, now);
}
#[test]
#[ignore]
fn offline_render_off_benchmark() {
    let rows = [
        super::tests::row("root", None, false),
        super::tests::row("child", Some("root"), false),
    ];
    let owner = super::tests::parent("root");
    let now = Instant::now();
    let mut graph = Graph::default();
    graph.observe(&rows, &owner, false, now);
    graph.render("root", 24, 5, now);
    let at = now + SETTLE;
    let off = graph.render("root", 24, 5, at);
    assert!(graph.textmas_is_empty_for_bench());
    assert!(graph.next_frame(at).is_none());
    if let Ok(dir) = std::env::var("OPTIMUS_GRAPH_CAPTURE_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            format!("{dir}/feature-off-render.json"),
            serde_json::to_string_pretty(&off).unwrap(),
        )
        .unwrap();
    }
    let base_off = std::env::var("OPTIMUS_GRAPH_BASE_OFF_FRAME")
        .ok()
        .map(|path| {
            let text =
                std::fs::read_to_string(path).expect("same-base OFF capture must be readable");
            serde_json::from_str::<Vec<String>>(&text)
                .expect("same-base OFF capture must contain the rendered frame")
        });
    let base_byte_equal = base_off.as_ref().map(|base| {
        assert_eq!(
            &off, base,
            "OFF/legacy renderer bytes must match the exact same-base frame"
        );
        true
    });
    const OBSERVATIONS: usize = 5_000;
    const RENDERS: usize = 4_000;
    for _ in 0..50 {
        graph_no_activity_for_bench(&mut graph, at);
        graph.observe(&rows, &owner, false, at);
        std::hint::black_box(graph.render("root", 24, 5, at));
    }
    let start = Instant::now();
    for _ in 0..OBSERVATIONS {
        graph.observe(&rows, &owner, false, at);
        graph_no_activity_for_bench(&mut graph, at);
        std::hint::black_box(&graph);
    }
    let observation = start.elapsed();
    let start = Instant::now();
    for _ in 0..RENDERS {
        std::hint::black_box(graph.render("root", 24, 5, at));
    }
    let render = start.elapsed();
    let result = serde_json::json!({"mode":"off","observations":OBSERVATIONS,"renders":RENDERS,
        "observation_total_ns":observation.as_nanos().to_string(),"render_total_ns":render.as_nanos().to_string(),
        "observation_ns_each":observation.as_nanos() as f64/OBSERVATIONS as f64,
        "render_ns_each":render.as_nanos() as f64/RENDERS as f64,
        "base_byte_equal":base_byte_equal,
        "fixture":"one real roster parent and one real roster child; no TextMAS activity"});
    println!("OFF_BENCHMARK {result}");
    if let Ok(dir) = std::env::var("OPTIMUS_GRAPH_CAPTURE_DIR") {
        std::fs::write(
            format!("{dir}/off-performance.json"),
            serde_json::to_string_pretty(&result).unwrap(),
        )
        .unwrap();
    }
}

#[test]
#[ignore]
fn offline_render_active_benchmark() {
    use crate::core::occ::textmas::{TextMasMode, TextMasRole};
    use crate::core::occ::textmas_activity::{
        textmas_call_id, TextMasActivityCall, TextMasActivitySnapshot, TextMasActivityStatus,
        TEXTMAS_ACTIVITY_SCHEMA,
    };
    let mut calls = Vec::new();
    for (index, role) in TextMasRole::ALL.into_iter().enumerate() {
        let mut call = TextMasActivityCall::queued(
            "perf",
            role,
            None,
            None,
            if index == 0 {
                Vec::new()
            } else {
                vec![textmas_call_id("perf", TextMasRole::ALL[index - 1], None)]
            },
        );
        call.status = TextMasActivityStatus::Running;
        calls.push(call);
    }
    for index in 0..64u32 {
        let mut call = TextMasActivityCall::queued(
            "perf",
            TextMasRole::Mapper,
            Some(index),
            Some(0),
            vec![textmas_call_id("perf", TextMasRole::Refiner, None)],
        );
        call.status = match index % 3 {
            0 => TextMasActivityStatus::Completed,
            1 => TextMasActivityStatus::Queued,
            _ => TextMasActivityStatus::Running,
        };
        calls.push(call);
    }
    for index in 0..63u32 {
        let depends = if index == 0 {
            (0..64)
                .map(|index| textmas_call_id("perf", TextMasRole::Mapper, Some(index)))
                .collect()
        } else {
            vec![textmas_call_id("perf", TextMasRole::Reducer, Some(0))]
        };
        let mut call = TextMasActivityCall::queued(
            "perf",
            TextMasRole::Reducer,
            Some(index),
            Some(if index == 0 { 1 } else { 2 }),
            depends,
        );
        call.status = if index % 4 == 0 {
            TextMasActivityStatus::Completed
        } else {
            TextMasActivityStatus::Queued
        };
        calls.push(call);
    }
    let snapshot = TextMasActivitySnapshot {
        schema: TEXTMAS_ACTIVITY_SCHEMA.into(),
        session_id: "root".into(),
        action_id: "perf".into(),
        mode: Some(TextMasMode::Sequential),
        active: true,
        stale: false,
        calls,
    };
    assert!(snapshot.is_valid());
    assert_eq!(snapshot.calls.len(), 130);
    let rows = [
        super::tests::row("root", None, false),
        super::tests::row("child", Some("root"), false),
    ];
    let owner = super::tests::parent("root");
    let now = Instant::now();
    let mut graph = Graph::default();
    graph.observe(&rows, &owner, false, now);
    let projection_start = Instant::now();
    graph.observe_activity(Some(&snapshot), now);
    let projection = projection_start.elapsed();
    assert_eq!(graph.textmas.nodes().len(), 9);
    assert_eq!(
        graph
            .textmas
            .nodes()
            .iter()
            .filter(|node| node.glyph == 'M')
            .count(),
        3
    );
    assert_eq!(
        graph
            .textmas
            .nodes()
            .iter()
            .filter(|node| node.glyph == 'Σ')
            .count(),
        3
    );
    const EVENTS: usize = 5_000;
    const RENDERS: usize = 4_000;
    for _ in 0..50 {
        graph.observe_activity(Some(&snapshot), now);
        std::hint::black_box(graph.render("root", 24, 5, now + SETTLE));
    }
    let start = Instant::now();
    for _ in 0..EVENTS {
        graph.observe_activity(Some(&snapshot), now);
        std::hint::black_box(&graph);
    }
    let duplicate_events = start.elapsed();
    let start = Instant::now();
    for _ in 0..RENDERS {
        std::hint::black_box(graph.render("root", 24, 5, now + SETTLE));
    }
    let render = start.elapsed();
    let result = serde_json::json!({"mode":"active","calls":130,"visual_activity_nodes":9,"events":EVENTS,"renders":RENDERS,
        "initial_projection_ns":projection.as_nanos().to_string(),
        "duplicate_event_total_ns":duplicate_events.as_nanos().to_string(),"render_total_ns":render.as_nanos().to_string(),
        "duplicate_event_ns_each":duplicate_events.as_nanos() as f64/EVENTS as f64,
        "render_ns_each":render.as_nanos() as f64/RENDERS as f64,
        "scheme":"deterministic fixed-time, offline, 130-call activity projection; no provider or daemon calls"});
    println!("ACTIVE_BENCHMARK {result}");
    if let Ok(dir) = std::env::var("OPTIMUS_GRAPH_CAPTURE_DIR") {
        std::fs::write(
            format!("{dir}/active-performance.json"),
            serde_json::to_string_pretty(&result).unwrap(),
        )
        .unwrap();
    }
}
