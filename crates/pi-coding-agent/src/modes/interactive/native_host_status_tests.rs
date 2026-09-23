use super::*;
use pi_tui::utils::{strip_ansi, visible_width};

fn job(id: &str, source: &str, status: &str, session_id: &str) -> local::AgentCronJob {
    local::AgentCronJob {
        id: id.into(),
        source: Some(source.into()),
        status: status.into(),
        session_id: session_id.into(),
        active_session_id: format!("active-{session_id}"),
        ..Default::default()
    }
}

fn mode() -> InteractiveMode {
    let mut mode = super::super::tests::stash_mode("heartbeat-fixture");
    mode.apply_connection_state_snapshot(local::AgentConnectionState {
        session_id: "heartbeat-fixture".into(),
        active_session_id: Some("active-heartbeat-fixture".into()),
        context_usage: local::ContextUsage {
            tokens: Some(146_000.0),
            context_window: 1_000_000.0,
            percent: Some(14.6),
        },
        ..Default::default()
    });
    mode
}

#[test]
fn heartbeat_snapshot_fallback_and_catalog_ownership_are_truthful() {
    let mut mode = mode();
    let user = job("u", "heartbeat", "active", "heartbeat-fixture");
    mode.connection_state.as_mut().unwrap().heartbeat = Some(user.clone());
    assert_eq!(
        heartbeat_counts(&mode).label(false).as_deref(),
        Some("HB user on")
    );
    mode.heartbeat_catalog = vec![
        local::AgentConnectionHeartbeat {
            job: local::AgentCronJob {
                status: "paused".into(),
                ..user
            },
            ..Default::default()
        },
        local::AgentConnectionHeartbeat {
            job: job("a", "rlm_heartbeat", "active", "heartbeat-fixture"),
            ..Default::default()
        },
        local::AgentConnectionHeartbeat {
            job: job("foreign", "heartbeat", "active", "other"),
            ..Default::default()
        },
        local::AgentConnectionHeartbeat {
            job: job("cron", "cron", "active", "heartbeat-fixture"),
            ..Default::default()
        },
        local::AgentConnectionHeartbeat {
            job: job("done", "rlm_heartbeat", "completed", "heartbeat-fixture"),
            ..Default::default()
        },
    ];
    assert_eq!(
        heartbeat_counts(&mode),
        HeartbeatCounts {
            user_paused: 1,
            agent_on: 1,
            ..Default::default()
        }
    );
    assert_eq!(
        heartbeat_counts(&mode).label(false).as_deref(),
        Some("HB user paused · agent on")
    );
    mode.subagent_snapshots.insert(
        "child".into(),
        local::AgentConnectionRlmChildAgentSnapshot {
            active_session_id: Some("active-child".into()),
            ..Default::default()
        },
    );
    mode.heartbeat_catalog
        .push(local::AgentConnectionHeartbeat {
            job: job("child", "rlm_heartbeat", "active", "child"),
            ..Default::default()
        });
    assert_eq!(heartbeat_counts(&mode).agent_on, 2);
    mode.heartbeat_catalog.clear();
    mode.heartbeat_catalog_authoritative = true;
    assert_eq!(
        heartbeat_counts(&mode).label(false),
        None,
        "removed catalog entries must not resurrect a stale snapshot"
    );
    mode.heartbeat_catalog_authoritative = false;
    for running in [false, true] {
        mode.connection_state.as_mut().unwrap().is_streaming = running;
        assert!(!right_status(&mode, Some("146k (15%)"), 120)
            .unwrap()
            .contains("running"));
    }
}

#[test]
fn heartbeat_off_paused_and_narrow_context_render_without_overflow() {
    let mut mode = mode();
    assert_eq!(
        right_status(&mode, Some("146k (15%)"), 80).as_deref(),
        Some("146k (15%)")
    );
    mode.connection_state.as_mut().unwrap().heartbeat =
        Some(job("u", "heartbeat", "paused", "heartbeat-fixture"));
    assert!(right_status(&mode, Some("146k (15%)"), 80)
        .unwrap()
        .contains("paused"));
    mode.connection_state
        .as_mut()
        .unwrap()
        .heartbeat
        .as_mut()
        .unwrap()
        .status = "active".into();
    for width in [1, 5, 10, 13, 20, 36, 80, 100, 120] {
        let right = right_status(&mode, Some("146k (15%)"), width);
        let row = tray_row::compose_tray_row(
            tray_row::TrayRow {
                left: "界界 fixture xhigh",
                jev_decision: None,
                jev_compaction: None,
                right: right.as_deref(),
            },
            width,
        );
        assert!(visible_width(&row) <= width, "{width}: {row}");
        assert!(!row.contains('\n'));
        if width >= 13 {
            assert!(row.contains("HB"), "{width}: {row}");
        }
        if width >= 10 {
            assert!(strip_ansi(&row).ends_with("146k (15%)"));
        }
    }
    assert!(right_status(&mode, None, 80)
        .unwrap()
        .contains("HB user on"));
    mode.connection_state
        .as_mut()
        .unwrap()
        .heartbeat
        .as_mut()
        .unwrap()
        .status = "cancelled".into();
    assert_eq!(right_status(&mode, None, 80), None);
}

#[test]
fn heartbeat_neon_and_classic_painted_footer_keeps_status_beside_context() {
    for style in ["neon", "prime"] {
        for running in [false, true] {
            let h = super::super::ui_tests::FrameHarness::new("heartbeat-fixture");
            *h.mode.borrow_mut() = mode();
            crate::modes::interactive::theme::theme::init_theme(Some(style), false);
            h.mode.borrow_mut().fullscreen_enabled = true;
            h.mode
                .borrow_mut()
                .connection_state
                .as_mut()
                .unwrap()
                .is_streaming = running;
            h.mode
                .borrow_mut()
                .connection_state
                .as_mut()
                .unwrap()
                .heartbeat = Some(job("u", "heartbeat", "active", "heartbeat-fixture"));
            for width in [20, 36, 80, 120] {
                h.width.set(width);
                let frame = h.paint();
                let footer = frame
                    .iter()
                    .find(|row| row.contains("146k (15%)"))
                    .unwrap_or_else(|| panic!("missing usage {style}/{width}: {frame:?}"));
                assert!(footer.contains("HB"), "{style}/{width}: {footer}");
                assert!(footer.trim_end().ends_with("146k (15%)"), "{footer}");
                assert!(
                    frame.iter().all(|row| visible_width(row) <= width),
                    "{style}/{width}: {frame:?}"
                );
            }
            h.mode
                .borrow_mut()
                .connection_state
                .as_mut()
                .unwrap()
                .heartbeat = None;
            assert!(!h.paint().join("\n").contains("HB"));
        }
    }
}

#[test]
fn shell_completion_live_history_and_neon_classic_are_one_content_line() {
    use crate::core::messages::{create_async_bash_completion_message, AsyncBashCompletionDetails};
    let message = AgentMessage::Custom(create_async_bash_completion_message(
        AsyncBashCompletionDetails {
            pid: 33504,
            command: "fixture/full/path\nInspect BashHandle private instructions".into(),
            exit_code: 0,
        },
        1,
    ));
    let original = serde_json::to_value(&message).unwrap();
    for style in ["neon", "prime"] {
        let mode = Rc::new(RefCell::new(mode()));
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        mode.borrow_mut().fullscreen_enabled = true;
        let transcript = Rc::new(RefCell::new(Transcript::new(mode.clone())));
        apply_event(
            &mode,
            &transcript,
            wire::AgentConnectionSessionEvent::MessageStart {
                message: message.clone(),
            },
        );
        apply_event(
            &mode,
            &transcript,
            wire::AgentConnectionSessionEvent::MessageEnd {
                message: message.clone(),
            },
        );
        for history in [false, true] {
            if history {
                transcript.borrow_mut().replace(Vec::new());
                transcript
                    .borrow_mut()
                    .replace_history(vec![message.clone()], 1.0);
            }
            for width in [8, 20, 36, 80, 120] {
                let rows = transcript.borrow_mut().render(width as f64);
                let plain = strip_ansi(&rows.join("\n"));
                assert!(
                    !plain.contains("fixture/full/path") && !plain.contains("BashHandle"),
                    "{plain}"
                );
                assert_eq!(rows.len(), 1, "{style}/{width}: {rows:?}");
                assert!(visible_width(&rows[0]) <= width);
                if width >= 80 {
                    assert!(
                        plain.contains("Shell finished — exit 0 (PID 33504)."),
                        "{plain}"
                    );
                }
            }
        }
    }
    assert_eq!(serde_json::to_value(&message).unwrap(), original);
}

#[test]
fn refinement_status_is_visible_while_idle_and_clears_on_all_terminal_events() {
    for style in ["neon", "prime"] {
        let h = super::super::ui_tests::FrameHarness::new("refinement-fixture");
        crate::modes::interactive::theme::theme::init_theme(Some(style), false);
        h.mode
            .borrow_mut()
            .connection_state
            .as_mut()
            .unwrap()
            .is_streaming = false;
        for terminal in [
            wire::AgentConnectionSessionEvent::RefinementUpdate {
                active: false,
                reason: Some("Refinement saved".into()),
            },
            wire::AgentConnectionSessionEvent::RefineFailed {
                error: "fixture refinement failed".into(),
            },
            wire::AgentConnectionSessionEvent::RefineComplete {
                result: Default::default(),
            },
        ] {
            for reason in [
                "Preparing refinement",
                "Reviewing refinement",
                "Saving refinement",
            ] {
                apply_event(
                    &h.mode,
                    &h.transcript,
                    wire::AgentConnectionSessionEvent::RefinementUpdate {
                        active: true,
                        reason: Some(reason.into()),
                    },
                );
                assert!(h.paint().join("\n").contains(reason));
                assert!(!h.mode.borrow().is_agent_streaming());
            }
            apply_event(&h.mode, &h.transcript, terminal);
            assert!(h.transcript.borrow().refinement_progress.is_none());
            assert!(!h.paint().join("\n").contains("Saving refinement"));
        }
        apply_event(
            &h.mode,
            &h.transcript,
            wire::AgentConnectionSessionEvent::RefinementUpdate {
                active: true,
                reason: None,
            },
        );
        h.transcript.borrow_mut().replace(Vec::new());
        assert!(h.transcript.borrow().refinement_progress.is_none());
    }
}

#[tokio::test]
async fn heartbeat_refresh_coalesces_changes_without_a_render_or_idle_poll_cost() {
    let connection = Arc::new(super::super::tests::RecordingConnection::default());
    let mut refresh = HeartbeatRefresh::new();
    for _ in 0..100 {
        refresh.request(connection.clone(), "s1", true);
    }
    // Let the single local fixture complete; the next poll schedules exactly one
    // trailing request and discards the pre-change catalog.
    tokio::task::yield_now().await;
    assert!(refresh.task.as_ref().unwrap().is_finished());
    assert!(refresh.poll(connection.clone(), "s1").await.is_none());
    tokio::task::yield_now().await;
    assert_eq!(
        refresh.poll(connection.clone(), "s1").await,
        Some((Vec::new(), true))
    );
    assert_eq!(connection.calls().len(), 2);
    for _ in 0..100 {
        assert!(refresh.poll(connection.clone(), "s1").await.is_none());
        let _ = right_status(&mode(), Some("146k (15%)"), 80);
    }
    assert_eq!(
        connection.calls().len(),
        2,
        "rendering/idle frames must not make requests"
    );
}

#[tokio::test]
async fn heartbeat_refresh_discards_old_session_and_cancelled_results() {
    let connection = Arc::new(super::super::tests::RecordingConnection::default());
    let mut refresh = HeartbeatRefresh::new();
    refresh.request(connection.clone(), "old", false);
    tokio::task::yield_now().await;
    assert!(refresh.poll(connection.clone(), "new").await.is_none());
    assert!(refresh.task.is_none());
    refresh.request(connection.clone(), "new", false);
    tokio::task::yield_now().await;
    refresh.cancel();
    assert!(refresh.poll(connection.clone(), "new").await.is_none());
}

// test.sh and the UI lane's focused commands use --test-threads=1 because themes
// are process-global. Keep these native frame tests in that serialized scope.
#[tokio::test]
async fn heartbeat_initial_empty_refresh_respects_negotiated_catalog_support() {
    for supported in [Some(true), Some(false), None] {
        let connection = Arc::new(super::super::tests::RecordingConnection::default());
        *connection.heartbeat_catalog_support.lock().unwrap() = supported;
        let mut mode = mode();
        mode.connection_state.as_mut().unwrap().heartbeat = Some(job(
            "stale-snapshot",
            "heartbeat",
            "active",
            "heartbeat-fixture",
        ));
        let mut refresh = HeartbeatRefresh::new();
        // Exactly the initial attach path: no change event makes this authoritative.
        refresh.request(connection.clone(), "heartbeat-fixture", false);
        tokio::task::yield_now().await;
        let (catalog, authoritative) = refresh
            .poll(connection.clone(), "heartbeat-fixture")
            .await
            .expect("completed initial empty catalog");
        assert!(catalog.is_empty());
        apply_catalog_result(&mut mode, &catalog, authoritative);
        assert_eq!(
            mode.heartbeat_catalog_authoritative,
            supported == Some(true)
        );
        assert_eq!(
            heartbeat_counts(&mode).user_on,
            usize::from(supported != Some(true))
        );
        assert_eq!(
            connection.calls().len(),
            1,
            "one optional lookup, never an idle poll"
        );
        assert!(refresh
            .poll(connection.clone(), "heartbeat-fixture")
            .await
            .is_none());
        assert_eq!(connection.calls().len(), 1);
    }
}
