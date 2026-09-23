// Included inside t11_refinement_lifecycle_tests to reuse its isolated faux-provider fixture.

async fn rf_bounded<F: std::future::Future>(future: F) -> F::Output {
    tokio::time::timeout(std::time::Duration::from_secs(3), future)
        .await
        .expect("RF fixture must settle within the test budget")
}

struct RfPlannerGate {
    started: Arc<tokio::sync::Notify>,
    release: CancellationToken,
}

impl Drop for RfPlannerGate {
    fn drop(&mut self) {
        self.release.cancel();
    }
}

fn rf_block_planner(t: &T11Session, proposal: Value) -> RfPlannerGate {
    let gate = RfPlannerGate {
        started: Arc::new(tokio::sync::Notify::new()),
        release: CancellationToken::new(),
    };
    let started = gate.started.clone();
    let release = gate.release.clone();
    t.provider
        .append_responses(vec![FauxResponseStep::Factory(Arc::new(
            move |_, options, _, _| {
                let started = started.clone();
                let release = release.clone();
                let signal = options.and_then(|options| options.signal.clone()).unwrap();
                let proposal = proposal.clone();
                Box::pin(async move {
                    started.notify_one();
                    tokio::select! {
                        biased;
                        _ = signal.cancelled() => {},
                        _ = release.cancelled() => {},
                    }
                    faux_assistant_message(FauxAssistantContent::Text(proposal.to_string()), None)
                })
            },
        ))]);
    gate
}

fn rf_progress(session: &Arc<AgentSession>) -> Arc<Mutex<Vec<(bool, String)>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::RefinementUpdate { active, reason } = event {
            sink.lock()
                .unwrap()
                .push((active, reason.unwrap_or_default()));
        }
    }));
    seen
}

fn rf_queued_prompt_options() -> PromptOptions {
    PromptOptions {
        queue_if_busy: Some(true),
        streaming_behavior: Some(SESSION_INPUT_SCHEDULE_FOLLOW_UP.to_string()),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rf_prompt_admitted_while_planner_blocked_and_execution_stays_ordered() {
    let t = T11Session::new(
        "rf-admission-order",
        false,
        serde_json::json!({"enabled": false}),
        vec![],
    )
    .await;
    append_user_turn(&t);
    let progress = rf_progress(&t.session);
    let gate = rf_block_planner(&t, proposal_json("rf-preference"));
    let session = t.session.clone();
    let command = tokio::spawn(async move {
        session
            .prompt_and_wait("/refine remember the fixture preference", None)
            .await
    });
    rf_bounded(gate.started.notified()).await;
    assert!(
        t.local_harness().refinements.is_empty(),
        "admission is not a saved refinement"
    );
    assert!(
        !command.is_finished(),
        "the command's completion waits for a durable result"
    );
    assert!(
        t.session
            .session_action_commit_owner
            .lock()
            .unwrap()
            .is_none(),
        "planning does not own admission"
    );
    for text in ["RF first input", "RF second input"] {
        rf_bounded(
            t.session
                .prompt_until_accepted(text, Some(rf_queued_prompt_options())),
        )
        .await
        .unwrap();
    }
    assert!(
        t.agent.prompt_batches.lock().unwrap().is_empty(),
        "input is accepted, not executed beside the planner"
    );
    let pending: Vec<_> = t
        .session
        .action_store
        .lock()
        .unwrap()
        .queued_actions(None)
        .into_iter()
        .filter_map(|action| match action.payload {
            QueuedActionPayload::Turn(turn) => Some(turn.base.text),
            _ => None,
        })
        .collect();
    assert_eq!(pending, ["RF first input", "RF second input"]);
    gate.release.cancel();
    rf_bounded(command).await.unwrap().unwrap();
    rf_bounded(t.session.wait_for_session_input_idle())
        .await
        .unwrap();
    let delivered: Vec<String> = t
        .agent
        .prompt_batches
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .filter_map(|message| {
            if let AgentMessage::Message(Message::User(user)) = message {
                return Some(user.content.text());
            }
            None
        })
        .filter(|text| text.starts_with("RF "))
        .collect();
    assert_eq!(
        delivered,
        ["RF first input", "RF second input"],
        "no loss, reordering or duplicate execution"
    );
    assert_eq!(t.local_harness().refinements.len(), 1);
    let stages = progress.lock().unwrap().clone();
    assert_eq!(
        stages,
        vec![
            (true, "Refinement queued".into()),
            (true, "Preparing refinement".into()),
            (true, "Reviewing refinement".into()),
            (true, "Saving refinement".into()),
            (false, "Refinement saved".into()),
        ]
    );
    t.session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rf_stop_cancel_epoch_and_branch_change_reject_at_reacquired_commit() {
    for mode in ["stop", "cancel", "epoch", "branch"] {
        let t = T11Session::new(
            &format!("rf-commit-{mode}"),
            false,
            serde_json::json!({"enabled": false}),
            vec![],
        )
        .await;
        append_user_turn(&t);
        let progress = rf_progress(&t.session);
        let session = Arc::downgrade(&t.session);
        let saw_fence = Arc::new(AtomicBool::new(false));
        let observed = saw_fence.clone();
        t.session.subscribe(Arc::new(move |event| {
            if let AgentSessionEvent::RefinementUpdate {
                active: true,
                reason,
            } = event
            {
                if reason.as_deref() == Some("Saving refinement") {
                    let session = session.upgrade().unwrap();
                    observed.store(
                        session
                            .session_action_commit_owner
                            .lock()
                            .unwrap()
                            .is_some(),
                        Ordering::SeqCst,
                    );
                    match mode {
                        "stop" => session.request_abort(),
                        "cancel" => session.abort_refinement(),
                        "epoch" => {
                            session
                                .session_input_pump_epoch
                                .fetch_add(1, Ordering::SeqCst);
                        }
                        "branch" => {
                            session
                                .auto_refine_branch_version
                                .fetch_add(1, Ordering::SeqCst);
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }));
        t.queue_json(proposal_json("rf-must-not-apply"));
        let error = rf_bounded(t.session.prompt_and_wait("/refine fixture", None))
            .await
            .unwrap_err();
        assert!(error.contains("aborted"), "{mode}: {error}");
        // Stop rejects the caller immediately; that receipt is not owned-work settlement.
        // Join the actual queued pump before checking the final save/flight/progress state.
        let pump = t.session.session_input_pump.lock().unwrap().clone();
        rf_bounded(pump).await.unwrap();
        assert!(
            saw_fence.load(Ordering::SeqCst),
            "manual apply reacquires the admission fence"
        );
        assert!(
            t.local_harness().refinements.is_empty(),
            "{mode}: stale plan never saved"
        );
        assert!(t.session.refine_in_flight.lock().unwrap().is_none());
        assert!(t
            .session
            .session_action_commit_owner
            .lock()
            .unwrap()
            .is_none());
        assert_eq!(
            progress.lock().unwrap().last(),
            Some(&(false, "Refinement cancelled".into()))
        );
        if mode == "stop" {
            assert!(t.session.explicitly_stopped());
            assert!(t
                .session
                .session_input_pump_suspended
                .load(Ordering::SeqCst));
            let calls = t.provider.call_count();
            assert!(t
                .session
                .refine_with_options(&RefineOptions::default(), false, None)
                .await
                .is_err());
            assert_eq!(
                t.provider.call_count(),
                calls,
                "fresh cancellation controller cannot bypass stop"
            );
        }
        t.session.dispose_async(Some(false)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rf_concurrent_refinements_serialize_planning_and_apply() {
    let t = T11Session::new(
        "rf-serialize",
        false,
        serde_json::json!({"enabled": false}),
        vec![],
    )
    .await;
    append_user_turn(&t);
    let gate = rf_block_planner(&t, proposal_json("rf-first"));
    let first_session = t.session.clone();
    let first = tokio::spawn(async move {
        first_session
            .refine_with_options(&RefineOptions::default(), false, None)
            .await
    });
    rf_bounded(gate.started.notified()).await;
    t.queue_json(proposal_json("rf-second"));
    let second_session = t.session.clone();
    let options = RefineOptions::default();
    let mut second = Box::pin(second_session.refine_with_options(&options, false, None));
    assert!(futures::poll!(&mut second).is_pending());
    assert_eq!(
        t.provider.call_count(),
        1,
        "the second planner has not started"
    );
    let admission = rf_bounded(t.session.acquire_direct_turn_admission_fence())
        .await
        .unwrap();
    admission.release();
    gate.release.cancel();
    rf_bounded(first).await.unwrap().unwrap();
    rf_bounded(second).await.unwrap();
    assert_eq!(t.provider.call_count(), 2);
    assert_eq!(t.local_harness().refinements.len(), 2);
    assert!(t.session.refine_in_flight.lock().unwrap().is_none());
    t.session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rf_automatic_save_no_edit_failure_and_drop_settle_progress() {
    for mode in ["save", "no-edit", "failure", "cancel", "drop"] {
        let t = T11Session::new(
            &format!("rf-progress-{mode}"),
            false,
            serde_json::json!({"enabled": false}),
            vec![],
        )
        .await;
        append_user_turn(&t);
        let progress = rf_progress(&t.session);
        let gate = rf_block_planner(
            &t,
            match mode {
                "no-edit" => {
                    serde_json::json!({"summary": "No change", "rationale": "No useful edit", "edits": []})
                }
                "failure" => serde_json::json!({"invalid": true}),
                _ => proposal_json("rf-automatic"),
            },
        );
        if mode == "failure" {
            t.queue_json(serde_json::json!({"alsoInvalid": true}));
        }
        let session = t.session.clone();
        let work = tokio::spawn(async move {
            session
                .refine_with_options(
                    &RefineOptions::default(),
                    false,
                    Some(REFINEMENT_SOURCE_AUTO),
                )
                .await
        });
        rf_bounded(gate.started.notified()).await;
        let result = match mode {
            "drop" => {
                work.abort();
                assert!(rf_bounded(work).await.unwrap_err().is_cancelled());
                None
            }
            "cancel" => {
                t.session.abort_refinement();
                Some(rf_bounded(work).await.unwrap())
            }
            _ => {
                gate.release.cancel();
                Some(rf_bounded(work).await.unwrap())
            }
        };
        let expected = match mode {
            "save" => "Refinement saved",
            "no-edit" => "No refinement changes needed",
            "failure" => "Refinement failed",
            _ => "Refinement cancelled",
        };
        assert_eq!(
            progress.lock().unwrap().last(),
            Some(&(false, expected.to_string()))
        );
        assert!(t.session.refine_in_flight.lock().unwrap().is_none());
        assert!(
            t.session.refinement_execution.try_lock().is_ok(),
            "dropped task releases operation ownership"
        );
        if matches!(mode, "save" | "no-edit") {
            let result = result.unwrap().unwrap();
            assert_eq!(
                result
                    .applied_edits
                    .iter()
                    .filter(|edit| edit.applied)
                    .count(),
                usize::from(mode == "save")
            );
            assert_eq!(
                t.local_harness().refinements.len(),
                1,
                "no-edit outcomes are still recorded"
            );
        } else {
            if let Some(result) = result {
                assert!(result.is_err());
            }
            assert!(t.local_harness().refinements.is_empty());
        }
        gate.release.cancel();
        t.session.dispose_async(Some(false)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rf_newer_harness_entry_survives_a_blocked_stale_plan() {
    let t = T11Session::new(
        "rf-stale-entry",
        false,
        serde_json::json!({"enabled": false}),
        vec![],
    )
    .await;
    append_user_turn(&t);
    t.queue_json(proposal_json("rf-shared-entry"));
    t.session
        .refine_with_options(&RefineOptions::default(), false, None)
        .await
        .unwrap();
    let mut proposal = proposal_json("rf-shared-entry");
    proposal["edits"][0]["action"] = serde_json::json!("update");
    proposal["edits"][0]["content"] = serde_json::json!("obsolete planner content");
    let gate = rf_block_planner(&t, proposal);
    let session = t.session.clone();
    let pending = tokio::spawn(async move {
        session
            .refine_with_options(&RefineOptions::default(), false, None)
            .await
    });
    rf_bounded(gate.started.notified()).await;
    let mut state = t.local_harness();
    state
        .entries
        .get_mut("memory")
        .unwrap()
        .get_mut("rf-shared-entry")
        .unwrap()
        .content = "newer external content".into();
    save_harness_state(&t.local_harness_dir(), &state).unwrap();
    gate.release.cancel();
    let result = rf_bounded(pending).await.unwrap().unwrap();
    assert!(!result.applied_edits[0].applied);
    assert_eq!(
        result.applied_edits[0].error.as_deref(),
        Some("entry changed during refinement planning")
    );
    assert_eq!(
        t.local_harness().entries["memory"]["rf-shared-entry"].content,
        "newer external content"
    );
    t.session.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rf_post_save_extension_does_not_hold_prompt_admission() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = CancellationToken::new();
    let gate = RfPlannerGate {
        started: started.clone(),
        release: release.clone(),
    };
    let extension: crate::core::extensions::types::ExtensionFactory = Arc::new(move |pi| {
        let started = started.clone();
        let release = release.clone();
        pi.on(
            "refine_complete",
            Arc::new(move |_, _| {
                let started = started.clone();
                let release = release.clone();
                Box::pin(async move {
                    started.notify_one();
                    release.cancelled().await;
                    None
                })
            }),
        );
        Box::pin(async { Ok(()) })
    });
    let t = T11Session::new(
        "rf-completion-hook",
        false,
        serde_json::json!({"enabled": false}),
        vec![extension],
    )
    .await;
    append_user_turn(&t);
    let progress = rf_progress(&t.session);
    t.queue_json(proposal_json("rf-hook-saved"));
    let session = t.session.clone();
    let pending =
        tokio::spawn(async move { session.prompt_and_wait("/refine fixture", None).await });
    rf_bounded(gate.started.notified()).await;
    assert_eq!(
        t.local_harness().refinements.len(),
        1,
        "save completes before the extension hook"
    );
    assert_eq!(
        progress.lock().unwrap().last(),
        Some(&(false, "Refinement saved".into()))
    );
    rf_bounded(
        t.session
            .prompt_until_accepted("RF queued after save", Some(rf_queued_prompt_options())),
    )
    .await
    .unwrap();
    assert!(
        t.agent.prompt_batches.lock().unwrap().is_empty(),
        "execution still waits for the selected command"
    );
    gate.release.cancel();
    rf_bounded(pending).await.unwrap().unwrap();
    rf_bounded(t.session.wait_for_session_input_idle())
        .await
        .unwrap();
    t.session.dispose_async(Some(false)).await;
}
