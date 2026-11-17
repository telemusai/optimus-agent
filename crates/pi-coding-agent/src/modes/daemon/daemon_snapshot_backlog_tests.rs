use super::*;
use super::daemon_parity_tests::{ReportingSession, ScriptedDaemonFixture};
use serde_json::json;

fn fixture(case: &str) -> ScriptedDaemonFixture {
    let root = daemon_parity_tests::state_root(case);
    let file = root.join("sessions/scripted.jsonl").to_string_lossy().into_owned();
    ScriptedDaemonFixture::new(case, ReportingSession::arc("active", "saved", &file, None, None))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backlog_snapshot_capture_and_stream_keep_the_published_cursor() {
    let mut fixture = fixture("backlog-snapshot-capture");
    let entry = fixture.daemon.session_entry_for_state(&fixture.state);
    let saved: AgentMessage = serde_json::from_value(serde_json::json!({"role":"user", "content":"published", "timestamp":1})).unwrap();
    *entry.snapshot_boundary.lock().unwrap() = Some(PublishedTranscript {
        messages: Arc::new(vec![saved.clone()]), streaming_message: None, state_flags: (false, false, false), sequence: 5, generation: "captured".into(),
    });
    fixture.state.lock().unwrap().last_event_sequence = 99;
    let snapshot = fixture.daemon.create_session_snapshot(&fixture.state, false).await.unwrap();
    assert_eq!(snapshot["lastEventSequence"], 5);
    assert_eq!(snapshot["messages"][0]["content"], "published");
    let transcript = create_snapshot_transcript_chunks(CreateSnapshotTranscriptChunksOptions {
        active_session_id: fixture.active_session_id.clone(), snapshot_id: "captured-transfer".into(), messages: vec![saved], target_chunk_bytes: Some(SNAPSHOT_TARGET_CHUNK_BYTES), aborted: false,
    });
    let signal = mark_client_snapshot_streaming(&fixture.client, &fixture.active_session_id);
    fixture.daemon.stream_worker_snapshot(&fixture.client, &fixture.state, "captured-transfer", snapshot, 1, transcript, "attach", signal, true).await.unwrap();
    let mut end = None;
    while let Ok(bytes) = fixture.outbound.try_recv() {
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        if value["type"] == "session_snapshot_end" { end = Some(value); }
    }
    let end = end.expect("successful terminal frame");
    assert_eq!(end["lastEventSequence"], 5, "streamer must not substitute the newer live counter");
    assert_eq!(end["lastEventCursor"]["generation"], "captured");
}

#[tokio::test]
async fn backlog_snapshot_replays_bounded_worker_frames_without_a_full_catchup() {
    let mut fixture = fixture("backlog-snapshot-frames");
    let signal = mark_client_snapshot_streaming(&fixture.client, &fixture.active_session_id);
    for sequence in [5, 6, 7] {
        let frame = DaemonOutbound::Raw(serde_json::json!({
            "type":"session_event", "activeSessionId":fixture.active_session_id,
            "event":{"type":"message_end", "message":{"role":"user", "content":sequence.to_string(), "timestamp":sequence}},
            "meta":{"sequence":sequence, "cursor":{"generation":"generation", "sequence":sequence}}
        }));
        assert!(fixture.daemon.defer_snapshot_frame(&fixture.client, &fixture.active_session_id, &frame));
    }
    assert!(!signal.is_cancelled());
    fixture.daemon.finish_snapshot_and_replay(&fixture.client, &fixture.active_session_id, 5, "generation", false);
    let mut sequences = Vec::new();
    while let Ok(bytes) = fixture.outbound.try_recv() {
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        sequences.push(value["meta"]["sequence"].as_u64().unwrap());
    }
    assert_eq!(sequences, vec![6, 7]);
    assert!(!fixture.client.snapshot_streaming());
    assert!(fixture.client.state.lock().unwrap().catchup_active_session_ids.as_ref().is_none_or(|ids| ids.is_empty()));
    assert!(fixture.client.deferred_snapshot_frames.lock().unwrap().is_empty());
}


fn fixture_with_children(case: &str, children: Vec<Value>) -> ScriptedDaemonFixture {
    let root = daemon_parity_tests::state_root(case);
    let file = root.join("sessions/scripted.jsonl").to_string_lossy().into_owned();
    let mut session = ReportingSession::arc("active", "saved", &file, None, None);
    Arc::get_mut(&mut session).unwrap().child_snapshots = children;
    ScriptedDaemonFixture::new(case, session)
}

fn write_child_session(root: &Path, id: &str) -> (String, String) {
    let directory = root.join("session-artifacts").join(id);
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("session.jsonl");
    std::fs::write(&file, format!("{}\n", json!({"type":"session","id":format!("session-{id}"),"version":3,"timestamp":"2026-01-01T00:00:00Z","cwd":root.join("workspace")}))).unwrap();
    (directory.to_string_lossy().into_owned(), file.to_string_lossy().into_owned())
}

fn register_resident_child(fixture: &ScriptedDaemonFixture, root_file: &str, index: usize) -> (String, String, String) {
    let id = format!("resident-{index}");
    let (directory, file) = write_child_session(&fixture.root, &id);
    let active = format!("active-{id}");
    let session = ReportingSession::arc(&active, &format!("session-{id}"), &file, None, None);
    let metadata = AgentSessionRuntimeMetadata {
        kind: Some("subagent".into()),
        parent_active_session_id: Some(fixture.active_session_id.clone()),
        parent_session_file: Some(root_file.into()),
        rlm_child_id: Some(id.clone()),
        ..Default::default()
    };
    let state = Arc::new(StdMutex::new(ActiveSessionState::new(
        active.clone(),
        crate::modes::daemon::active_session_state::AgentSessionRuntime {
            session: ActiveSessionRuntimeSession {
                session_id: format!("session-{id}"),
                session_name: Some(id.clone()),
                session_file: Some(file.clone()),
                ..Default::default()
            },
            metadata: Some(metadata.clone()),
            model_fallback_message: None,
        },
    )));
    fixture.daemon.sessions.lock().unwrap().insert(active, Arc::new(DaemonSessionState {
        state,
        session,
        runtime_metadata: metadata,
        snapshot_boundary: StdMutex::new(None),
    }));
    (directory, file, id)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backlog_resident_children_skip_passive_display_and_transcript_stats() {
    let child_snapshots: Vec<Value> = (0..21).map(|index| json!({"id":format!("resident-{index}"),"sessionName":format!(" Resident {index} "),"label":format!("resident-{index}"),"status":"running"})).collect();
    let fixture = fixture_with_children("backlog-resident-fast-path", child_snapshots);
    let root_file = fixture.daemon.session_of(&fixture.state).session_file().unwrap();
    let mut resident_display_paths = Vec::new();
    for index in 0..21 {
        let (directory, file, id) = register_resident_child(&fixture, &root_file, index);
        resident_display_paths.push(rlm_subagent_display_path(&directory));
        let display = RlmSubagentDisplayEntry::new(&id, &id, &directory, &file, "running");
        assert!(write_rlm_subagent_display_entry(&display).unwrap());
        fixture.daemon.rlm_spawn_ledger().append_spawn(RlmSpawnInput {
            child_id: id,
            parent: root_file.clone(),
            child: file,
            depth: 1,
            name: format!("resident-{index}"),
        }).await.unwrap();
    }

    let projected = fixture.daemon.build_rlm_child_snapshots_plumbing(&fixture.active_session_id);
    let mut projected_ids: Vec<&str> = projected.iter().filter_map(|child| child["id"].as_str()).collect();
    projected_ids.sort_unstable();
    let expected_ids: Vec<String> = (0..21).map(|index| format!("resident-{index}")).collect();
    assert_eq!(projected_ids.len(), 21);
    assert!(projected.iter().all(|child| child["activeSessionId"].as_str().is_some_and(|id| id.starts_with("active-resident-"))));
    assert!(expected_ids.iter().all(|id| projected_ids.contains(&id.as_str())));

    let resident_only = fixture.daemon.walk_passive_rlm_subagents(&[], false).await;
    assert!(resident_only.result.is_empty());
    let root_registry = fixture.daemon.legacy_rlm_subagent_registry_path(&root_file, &fixture.daemon.session_of(&fixture.state).session_id());
    assert!(!resident_only.input_stats.contains_key(&resolve_path(&root_registry)), "resident-only walk read the root legacy registry");
    assert!(resident_only.input_stats.is_empty(), "resident-only walk performed passive input reads: {:?}", resident_only.input_stats);

    let (passive_dir, passive_file) = write_child_session(&fixture.root, "passive");
    let passive_display = RlmSubagentDisplayEntry::new("passive", "passive", &passive_dir, &passive_file, "completed");
    assert!(write_rlm_subagent_display_entry(&passive_display).unwrap());
    fixture.daemon.rlm_spawn_ledger().append_spawn(RlmSpawnInput {
        child_id: "passive".into(),
        parent: root_file,
        child: passive_file,
        depth: 1,
        name: "passive".into(),
    }).await.unwrap();

    let walked = fixture.daemon.walk_passive_rlm_subagents(&[], false).await;
    assert_eq!(walked.result.len(), 1, "only the completed nonresident child is passive");
    assert_eq!(walked.result[0].entry.child_id, "passive");
    for display_path in resident_display_paths {
        assert!(!walked.input_stats.contains_key(&resolve_path(&display_path)), "resident display was read: {display_path}");
    }
    assert!(walked.input_stats.contains_key(&resolve_path(&rlm_subagent_display_path(&passive_dir))), "nonresident display stat is required for memo invalidation");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backlog_attach_response_is_enqueued_before_snapshot_stream() {
    let mut fixture = fixture("backlog-attach-response-order");
    let saved: Vec<AgentMessage> = (0..128).map(|index| serde_json::from_value(json!({"role":"user","content":"x".repeat(32 * 1024),"timestamp":index})).unwrap()).collect();
    *fixture.daemon.session_entry_for_state(&fixture.state).snapshot_boundary.lock().unwrap() = Some(PublishedTranscript {
        messages: Arc::new(saved), streaming_message: None, state_flags: (false, false, false), sequence: 5, generation: "captured".into(),
    });
    fixture.client.state.lock().unwrap().transport = Some("private-framed".into());
    fixture.send(json!({"type":"attach","id":"attach-order","activeSessionId":fixture.active_session_id,"capabilities":["chunked_snapshot","slim_attach"]})).await;

    let mut decoder = PrivateFrameDecoder::new();
    let mut response_seen = false;
    for _ in 0..64 {
        let bytes = tokio::time::timeout(Duration::from_secs(10), fixture.outbound.recv()).await.unwrap().unwrap();
        for frame in decoder.push(&bytes).unwrap() {
            let value: Value = serde_json::from_slice(&frame.payload).unwrap();
            if value["type"] == "response" && value["id"] == "attach-order" {
                response_seen = true;
                break;
            }
            assert!(!matches!(value["type"].as_str(), Some("session_snapshot_begin" | "session_snapshot_chunk" | "session_snapshot_end")), "snapshot frame preceded attach response: {value}");
        }
        if response_seen { break; }
    }
    assert!(response_seen, "attach response was not emitted");
}


#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backlog_attach_response_write_failure_cleans_pending_snapshot() {
    let mut fixture = fixture("backlog-attach-response-write-failed");
    let saved: Vec<AgentMessage> = (0..16).map(|index| serde_json::from_value(json!({"role":"user","content":"x".repeat(8 * 1024),"timestamp":index})).unwrap()).collect();
    let entry = fixture.daemon.session_entry_for_state(&fixture.state);
    *entry.snapshot_boundary.lock().unwrap() = Some(PublishedTranscript {
        messages: Arc::new(saved), streaming_message: None, state_flags: (false, false, false), sequence: 5, generation: "captured".into(),
    });
    // Stop capture before the response write. This makes deferred-state admission
    // deterministic without making the spawned gate task race the test.
    let boundary_hold = entry.snapshot_boundary.lock().unwrap();
    fixture.client.state.lock().unwrap().transport = Some("private-framed".into());
    fixture.client.set_backpressured(true);
    // Private framing bypasses the backpressured short-circuit in this checkout.
    // End the writer to force the real response-enqueue failure this test targets.
    fixture.client.writer.end();
    let daemon = Arc::clone(&fixture.daemon);
    let client = Arc::clone(&fixture.client);
    let command = serialize_json_line(&json!({"type":"attach","id":"attach-write-failed","activeSessionId":fixture.active_session_id,"capabilities":["chunked_snapshot","slim_attach"]}));
    let attach = tokio::spawn(async move { daemon.handle_line(client, command).await; });
    tokio::time::timeout(Duration::from_secs(10), async {
        while !fixture.client.snapshot_streaming() { tokio::task::yield_now().await; }
    }).await.expect("attach did not mark its snapshot while capture was held");

    let deferred_event = |sequence| DaemonOutbound::Raw(json!({
        "type":"session_event", "activeSessionId":fixture.active_session_id,
        "event":{"type":"message_end", "message":{"role":"user", "content":"deferred", "timestamp":sequence}},
        "meta":{"sequence":sequence, "cursor":{"generation":"captured", "sequence":sequence}}
    }));
    assert!(fixture.daemon.defer_snapshot_frame(&fixture.client, &fixture.active_session_id, &deferred_event(6)));
    assert!(fixture.client.deferred_snapshot_frames.lock().unwrap().contains_key(&fixture.active_session_id));
    drop(boundary_hold);
    tokio::time::timeout(Duration::from_secs(10), attach).await.expect("attach did not finish").unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while fixture.client.snapshot_streaming() { tokio::task::yield_now().await; }
    }).await.expect("the gated snapshot task did not finish");

    // The gate task is the sole failed-stream cleanup owner on this path.
    assert!(fixture.client.attached_active_session_ids().is_empty(), "the failed response left a phantom attachment");
    assert!(!fixture.state.lock().unwrap().clients.iter().any(|candidate| Arc::ptr_eq(candidate, &fixture.client.state)));
    assert!(fixture.client.deferred_snapshot_frames.lock().unwrap().is_empty(), "the failed response retained unpublished deferred frames");
    assert!(!fixture.daemon.defer_snapshot_frame(&fixture.client, &fixture.active_session_id, &deferred_event(7)));
    assert!(matches!(fixture.outbound.try_recv(), Err(tokio::sync::mpsc::error::TryRecvError::Empty)), "a response or snapshot record escaped despite the rejected response");
}
