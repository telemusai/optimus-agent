use super::*;
use pi_ai::types::{AssistantMessageEvent, ContentBlock};
use pi_ai::utils::event_stream::AssistantMessageEventStream;

async fn pair() -> (tempfile::TempDir, Arc<AgentSession>, Arc<AgentSession>) {
    crate::modes::interactive::theme::theme::init_theme(Some("prime"), false);
    let parent = post_compaction_continuation_tests::test_session_with_credentials().await;
    let child = post_compaction_continuation_tests::test_session_with_credentials_and_tools(
        1, Some(vec!["ipython".into()]),
    ).await;
    let dir = tempfile::tempdir().unwrap();
    // The initial fixture provisioner is unused. Settle it before changing identity.
    let previous = child.ipython_kernel_provisioner.lock().unwrap().take();
    if let Some(previous) = previous { previous.dispose(Some(false)).await; }
    *child.session_manager.lock().unwrap() = SessionManager::create(&child.cwd, Some(&dir.path().to_string_lossy())).unwrap();
    let mut state = child.agent.state();
    state.model.api = "openai-completions".into();
    child.agent.set_state(state);
    child.build_runtime(Some(vec!["ipython".into()]), false);
    child.set_active_tools_by_name(&["ipython".into()]);
    assert!(child.retention_profile());
    parent.rlm_child_sessions.lock().unwrap().insert("child".into(), RetainedRlmChild { session: child.clone(), run: None });
    (dir, parent, child)
}

fn local_provider(child: &Arc<AgentSession>, entered: CancellationToken, hold: bool) {
    child.agent.set_stream_fn(Arc::new(move |model, _, _| {
        let entered = entered.clone();
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new_owned();
            entered.cancel();
            if hold {
                stream.spawn(std::future::pending::<()>());
            } else {
                let message = AssistantMessage { model:model.id, provider:model.provider, api:model.api,
                    content:vec![ContentBlock::Text(TextContent::new("audit complete\nRLM_CHILD_STATUS: complete"))],
                    stop_reason:"stop".into(), timestamp:now_ms_i64(), ..Default::default() };
                stream.push(AssistantMessageEvent::Done { reason:"stop".into(), message });
            }
            stream
        })
    }));
}

fn assert_native_settlement(receipt: &Value) {
    // Kernel Job Object ownership is currently proved only on Windows.
    assert_eq!(receipt["settled"], cfg!(windows), "{receipt}");
    if !cfg!(windows) {
        assert_eq!(receipt["status"], "failed_settlement");
        assert_eq!(receipt["error_code"], "execution_ownership_or_settlement_unproved");
        assert_eq!(receipt["acknowledged"]["kernel"], false);
        assert_eq!(receipt["acknowledged"]["owned_processes"], false);
    }
}

#[cfg(windows)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retained_stop_native_admission_settlement_history_restore_and_audit() {
    let (_dir, parent, child) = pair().await;
    let entered = CancellationToken::new();
    local_provider(&child, entered.clone(), true);
    let actor = child.clone();
    let running = tokio::spawn(async move { actor.prompt("original task", None).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.cancelled()).await.unwrap();
    let active = parent.active_child_execution("child").unwrap();
    let generation = active["execution_generation"].as_str().unwrap();
    let payload = serde_json::json!({"target":"child","execution_generation":generation,"message_id":"one","message":"diagnostic"});
    assert_eq!(parent.send_active_child_message(payload.clone()).unwrap()["deliveryStatus"], "accepted");
    assert_eq!(parent.send_active_child_message(payload.clone()).unwrap()["deliveryStatus"], "duplicate");
    let receipt = parent.stop_retained_child("child", 5000).await.unwrap();
    assert_eq!(receipt["settled"], true, "{receipt}");
    assert_eq!(receipt["retained"], true);
    assert_eq!(parent.send_active_child_message(payload).unwrap()["deliveryStatus"], "declined_stopped");
    tokio::time::timeout(std::time::Duration::from_secs(3), running).await.unwrap().unwrap().unwrap();
    assert!(child.prompt_until_accepted("incidental old work", None).await.is_err());
    assert!(!child.resume_stopped_queue());
    let file = child.session_file().unwrap();
    assert!(Path::new(&file).is_file());
    let original_id = child.session_id();
    *child.session_manager.lock().unwrap() = SessionManager::open(&file, None, None).unwrap();
    *child.explicit_stop.lock().unwrap() = ExplicitStopState::default();
    *child.retained_stop.lock().unwrap() = None;
    child.restore_explicit_stop();
    assert!(child.explicitly_stopped());
    assert_eq!(child.session_id(), original_id);
    assert_eq!(child.retained_stop.lock().unwrap().as_ref().unwrap()["settled"], true);
    assert!(parent.resume_retained_audit("child", "wrong", "Review only").is_err());
    let audited = CancellationToken::new();
    local_provider(&child, audited.clone(), false);
    let resumed = parent.resume_retained_audit("child", receipt["stop_generation"].as_str().unwrap(), "Review the saved transcript only").unwrap();
    assert_eq!(resumed["old_work_replayed"], false);
    assert_ne!(resumed["audit_generation"], receipt["stop_generation"]);
    tokio::time::timeout(std::time::Duration::from_secs(5), audited.cancelled()).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), child.wait_for_idle()).await.unwrap().unwrap();
    assert!(child.retained_kernel_epoch.lock().unwrap().is_some());
    assert!(parent.active_child_execution("child").unwrap()["active"] == false);
    parent.rlm_child_sessions.lock().unwrap().clear();
    child.dispose_async(Some(false)).await;
    parent.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn retained_stop_rejects_nonowned_and_keeps_unsupported_unsettled() {
    let (_dir, parent, child) = pair().await;
    assert!(parent.stop_retained_child("not-a-child", 0).await.is_err());
    assert!(parent.stop_retained_child("child", 10001).await.is_err());
    child.set_active_tools_by_name(&[]);
    let report = parent.stop_retained_child("child", 1000).await.unwrap();
    assert_eq!(report["accepted"], true);
    assert_eq!(report["settled"], false);
    assert_eq!(report["retained"], true);
    assert!(parent.resume_retained_audit("child", report["stop_generation"].as_str().unwrap(), "audit").is_err());
    parent.rlm_child_sessions.lock().unwrap().clear();
    child.dispose_async(Some(false)).await;
    parent.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retained_stop_commit_fence_cancellation_cannot_unlink_predecessor() {
    let session = post_compaction_continuation_tests::test_session_with_credentials().await;
    let first = session.acquire_commit_fence(false).await.unwrap();
    let cancelled = session.clone();
    let waiter = tokio::spawn(async move { cancelled.acquire_commit_fence(false).await });
    tokio::task::yield_now().await;
    waiter.abort();
    let _ = waiter.await;
    let next = session.clone();
    let mut third = tokio::spawn(async move { next.acquire_commit_fence(false).await });
    assert!(tokio::time::timeout(std::time::Duration::from_millis(25), &mut third).await.is_err());
    first.release();
    let permit = tokio::time::timeout(std::time::Duration::from_secs(2), third).await.unwrap().unwrap().unwrap();
    drop(permit);
    session.dispose_async(Some(false)).await;
}

#[cfg(windows)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retained_stop_real_python_tool_and_contained_descendant_settle_together() {
    use crate::core::tools::ipython::{create_ipython_tool, default_kernel_client_factory, IpythonKernelProvisioner, IpythonToolOptions};
    use pi_ai::types::ToolCall;
    use std::sync::atomic::AtomicU32;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_QUERY_LIMITED_INFORMATION};
    struct Process(HANDLE);
    impl Drop for Process { fn drop(&mut self) { unsafe { CloseHandle(self.0); } } }

    let (dir, parent, child) = pair().await;
    let old = child.ipython_kernel_provisioner.lock().unwrap().take().unwrap();
    old.dispose(Some(false)).await;
    let runtime = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../prime-agent-runtime/src").canonicalize().unwrap();
    let python = std::env::var("KERNEL_CONTAINMENT_TEST_PYTHON").expect("explicit fixture interpreter");
    let factory = default_kernel_client_factory();
    let native = IpythonKernelProvisioner::new(&child.cwd, Some(IpythonToolOptions {
        session_id: Some(child.session_id()),
        env: Some(vec![
            ("PYTHONPATH".into(), runtime.to_string_lossy().into()),
            ("PI_CODING_AGENT_DIR".into(), dir.path().to_string_lossy().into()),
            ("PRIME_AGENT_CODING_AGENT_DIR".into(), dir.path().to_string_lossy().into()),
            ("PRIME_AGENT_INTERNAL_ORPHAN_PROCESS_JOURNAL".into(), dir.path().join("orphans.jsonl").to_string_lossy().into()),
        ]),
        ..Default::default()
    }), Arc::new(move |mut options| { options.python = Some(python.clone()); factory(options) }));
    *child.ipython_kernel_provisioner.lock().unwrap() = Some(native.clone());
    let tool = create_ipython_tool(&child.cwd, Some(IpythonToolOptions { provisioner: Some(native.clone()), ..Default::default() }));
    let mut state = child.agent.state();
    state.tools = Some(vec![tool]);
    child.agent.set_state(state);
    let ready = CancellationToken::new();
    let finished = CancellationToken::new();
    let pid = Arc::new(AtomicU32::new(0));
    let errors = Arc::new(Mutex::new(Vec::new()));
    let updates = Arc::new(Mutex::new(String::new()));
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let unsubscribe = child.agent.subscribe(Arc::new({
        let ready = ready.clone(); let pid = pid.clone(); let errors = errors.clone();
        let updates = updates.clone(); let outcomes = outcomes.clone(); let finished = finished.clone();
        move |event, _| {
            if let AgentEvent::ToolExecutionUpdate { partial_result, .. } = &event {
                // Real ipython updates stream text in content, not details.stdout.
                // Accumulate chunks and require a complete line so a split PID is not truncated.
                let mut stdout = updates.lock().unwrap();
                for block in &partial_result.content {
                    if let pi_agent_core::types::ContentBlock::Text(text) = block { stdout.push_str(&text.text); }
                }
                if let Some(value) = stdout.split_inclusive('\n').filter(|line| line.ends_with('\n'))
                    .find_map(|line| line.split_once("OWNED_CHILD_READY ").and_then(|(_, pid)| pid.trim().parse::<u32>().ok())) {
                    pid.store(value, Ordering::SeqCst); ready.cancel();
                }
            }
            if let AgentEvent::ToolExecutionEnd { is_error, result, .. } = event {
                errors.lock().unwrap().push(is_error);
                let mut outcomes = outcomes.lock().unwrap();
                if outcomes.len() < 4 { outcomes.push(serde_json::to_value(result).unwrap()); }
                finished.cancel();
            }
            Box::pin(async {})
        }
    }));
    let script = "import subprocess, sys, threading\nowned_child = subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait()'], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)\nprint('OWNED_CHILD_READY', owned_child.pid, flush=True)\nthreading.Event().wait()";
    let issued = Arc::new(AtomicBool::new(false));
    child.agent.set_stream_fn(Arc::new(move |model, _, _| {
        let first = !issued.swap(true, Ordering::SeqCst);
        Box::pin(async move {
            let stream = AssistantMessageEventStream::new_owned();
            let reason = if first { "toolUse" } else { "stop" };
            let content = if first {
                vec![ContentBlock::ToolCall(ToolCall::new("native-stop-cell", "ipython", serde_json::json!({"code":script}).as_object().unwrap().clone()))]
            } else { vec![ContentBlock::Text(TextContent::new("local fixture returned"))] };
            let message = AssistantMessage { api:model.api, provider:model.provider, model:model.id,
                content, stop_reason:reason.into(), ..Default::default() };
            stream.push(AssistantMessageEvent::Done { reason:reason.into(), message });
            stream
        })
    }));
    let session = child.clone();
    let running = tokio::spawn(async move { session.prompt("Run the local retained-stop fixture", None).await });
    let started = tokio::time::timeout(std::time::Duration::from_secs(15), async {
        tokio::select! { biased; _ = ready.cancelled() => true, _ = finished.cancelled() => false }
    }).await;
    if !matches!(started, Ok(true)) {
        let cleanup = parent.stop_retained_child("child", 10_000).await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), running).await;
        let kernel = native.shutdown_and_settle(&child.session_id(), 1_000).await;
        panic!("real Python tool did not reach the local child; cleanup={cleanup:?}; kernel={kernel:?}; updates={:?}; outcomes={:?}", updates.lock().unwrap(), outcomes.lock().unwrap());
    }
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | 0x0010_0000, 0, pid.load(Ordering::SeqCst)) };
    if handle.is_null() {
        let cleanup = parent.stop_retained_child("child", 10_000).await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), running).await;
        panic!("could not retain the fixture descendant handle; cleanup={cleanup:?}");
    }
    let process = Process(handle);
    assert_ne!(unsafe { WaitForSingleObject(process.0, 0) }, WAIT_OBJECT_0);
    let receipt = parent.stop_retained_child("child", 10_000).await.unwrap();
    if receipt["settled"] != true {
        let kernel = native.shutdown_and_settle(&child.session_id(), 1_000).await;
        panic!("retained native fixture did not settle; receipt={receipt}; kernel={kernel:?}; outcomes={:?}", outcomes.lock().unwrap());
    }
    assert_eq!(unsafe { WaitForSingleObject(process.0, 0) }, WAIT_OBJECT_0, "real detached descendant must have exited");
    tokio::time::timeout(std::time::Duration::from_secs(2), running).await.unwrap().unwrap().unwrap();
    assert_eq!(*errors.lock().unwrap(), vec![true], "real aborted ipython reaches the public error event");
    assert!(Path::new(&child.session_file().unwrap()).is_file());
    unsubscribe();
    parent.rlm_child_sessions.lock().unwrap().clear();
    child.dispose_async(Some(false)).await;
    parent.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn retained_stop_release_claim_rejects_a_stop_before_any_cleanup_await() {
    let (_dir, parent, child) = pair().await;
    assert!(parent.try_claim_rlm_runtime_release("child", &child));
    // Emulates the daemon paused on its first deletion/checkpoint await.
    assert!(parent.stop_retained_child("child", 0).await.unwrap_err().contains("cleanup"));
    assert!(!parent.retained_stop_ids.lock().unwrap().contains_key("child"));
    assert!(!child.explicitly_stopped());
    assert!(parent.try_claim_rlm_runtime_release("child", &child), "core/daemon claim is idempotent");
    parent.rlm_child_sessions.lock().unwrap().clear();
    child.dispose_async(Some(false)).await;
    parent.dispose_async(Some(false)).await;
}

#[tokio::test]
async fn retained_stop_wins_late_release_and_keeps_the_child_addressable() {
    let (_dir, parent, child) = pair().await;
    parent.rlm_child_sessions.lock().unwrap().clear();
    let mut current = empty_rlm_child_run("child");
    current.session = Some(child.clone());
    current.session_name = "late-child".into();
    current.status = "cancelled".into();
    let run = Arc::new(Mutex::new(current.clone()));
    parent.active_rlm_child_runs.lock().unwrap().insert("child".into(), run.clone());
    parent.unsettled_rlm_child_runs.lock().unwrap().push(run.clone());
    // The finalizer has a stale pre-stop snapshot, like the original race.
    let receipt = parent.stop_retained_child("child", 5_000).await.unwrap();
    assert_native_settlement(&receipt);
    assert!(!parent.try_claim_rlm_runtime_release("child", &child));
    assert!(parent.finish_retained_rlm_run(&run, &current));
    assert!(!parent.active_rlm_child_runs.lock().unwrap().contains_key("child"));
    assert!(parent.rlm_child_sessions.lock().unwrap().contains_key("child"));
    assert!(run.lock().unwrap().settled);
    assert_eq!(parent.active_child_execution("child").unwrap()["session_id"], child.session_id());
    assert_eq!(parent.stop_retained_child("child", 0).await.unwrap()["stop_generation"], receipt["stop_generation"]);
    assert!(Path::new(&child.session_file().unwrap()).is_file());
    parent.rlm_child_sessions.lock().unwrap().clear();
    child.dispose_async(Some(false)).await;
    parent.dispose_async(Some(false)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retained_stop_rejects_an_atomically_reserved_explicit_deletion() {
    let (_dir, parent, child) = pair().await;
    let entered = CancellationToken::new();
    let release = CancellationToken::new();
    let owner = parent.clone();
    let started = entered.clone();
    let finish = release.clone();
    let deleting = tokio::spawn(async move {
        let entry = RlmSubagentRegistryEntry { rlm_child_id:"child".into(), ..Default::default() };
        owner.track_rlm_subagent_deletion(&entry, Box::new(move |_| Box::pin(async move {
            started.cancel();
            finish.cancelled().await;
            Err("synthetic selector preflight veto".into())
        }))).await
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), entered.cancelled()).await.unwrap();
    assert!(parent.stop_retained_child("child", 0).await.unwrap_err().contains("cleanup"));
    assert!(!child.explicitly_stopped());
    release.cancel();
    assert!(tokio::time::timeout(std::time::Duration::from_secs(2), deleting).await.unwrap().unwrap().is_err());
    // A vetoed preflight did not admit destructive cleanup or poison later stop.
    let stopped = parent.stop_retained_child("child", 5_000).await.unwrap();
    assert_native_settlement(&stopped);
    assert!(Path::new(&child.session_file().unwrap()).is_file());
    parent.rlm_child_sessions.lock().unwrap().clear();
    child.dispose_async(Some(false)).await;
    parent.dispose_async(Some(false)).await;
}
