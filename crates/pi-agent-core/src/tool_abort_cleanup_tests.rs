use super::*;
use crate::agent::Agent;
use serde_json::json;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use tokio::time::timeout;

const BUDGET: Duration = Duration::from_secs(3);

async fn cancellation_releases_drain(queue_update: bool) {
    let signal = CancellationToken::new();
    let update_started = CancellationToken::new();
    let release_update = CancellationToken::new();
    let release_cleanup = CancellationToken::new();
    let cleanup_finished = CancellationToken::new();
    let delivered = Arc::new(AtomicUsize::new(0));
    let agent = Agent::new(Default::default());
    let weak_agent = Arc::downgrade(&agent);
    let emit: AgentEventSink = {
        let agent = agent.clone();
        let update_started = update_started.clone();
        let release_update = release_update.clone();
        let delivered = delivered.clone();
        Arc::new(move |_| {
            let agent = agent.clone();
            let update_started = update_started.clone();
            let release_update = release_update.clone();
            let delivered = delivered.clone();
            Box::pin(async move {
                let _agent = agent;
                update_started.cancel();
                release_update.cancelled().await;
                delivered.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        })
    };
    let tool = AgentTool {
        name: "abort-cleanup".into(),
        description: "isolated cancellation fixture".into(),
        label: "Abort cleanup".into(),
        parameters: json!({"type":"object","properties":{}}),
        prepare_arguments: None,
        execution_mode: Some(ToolExecutionMode::Sequential),
        execute: Arc::new({
            let release_cleanup = release_cleanup.clone();
            let cleanup_finished = cleanup_finished.clone();
            move |_, _, signal, update| {
                let release_cleanup = release_cleanup.clone();
                let cleanup_finished = cleanup_finished.clone();
                let update_started = update_started.clone();
                Box::pin(async move {
                    if queue_update {
                        update.unwrap()(AgentToolResult::new(vec![AgentContentBlock::text("old-run")], json!({})));
                        update_started.cancelled().await;
                    }
                    signal.unwrap().cancel();
                    // The tool's own cleanup must survive the abort race.
                    release_cleanup.cancelled().await;
                    cleanup_finished.cancel();
                    Ok(AgentToolResult::new(Vec::new(), json!({})))
                })
            }
        }),
    };
    let prepared = PreparedToolCall {
        tool_call: AgentToolCall::new("abort-call", "abort-cleanup", Map::new()),
        args: json!({}),
        tool,
    };
    let outcome = timeout(BUDGET, execute_prepared_tool_call(&prepared, Some(&signal), &emit, None))
        .await.expect("tool cancellation did not return");
    assert!(outcome.is_error);
    release_cleanup.cancel();
    timeout(BUDGET, cleanup_finished.cancelled()).await.expect("detached tool cleanup was killed");
    drop(prepared);
    drop(emit);
    drop(agent);
    release_update.cancel();
    timeout(BUDGET, async {
        while weak_agent.strong_count() != 0 {
            tokio::task::yield_now().await;
        }
    }).await.expect("cancelled update drain retained the Agent");
    assert_eq!(delivered.load(Ordering::SeqCst), 0, "old updates escaped after cancellation returned");
}

#[tokio::test]
async fn core006_cancelled_empty_drain_releases_agent_and_preserves_tool_cleanup() {
    cancellation_releases_drain(false).await;
}

#[tokio::test]
async fn core006_cancelled_blocked_update_cannot_escape_into_next_run() {
    cancellation_releases_drain(true).await;
}
