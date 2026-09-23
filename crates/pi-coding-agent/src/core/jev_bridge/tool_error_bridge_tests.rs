//! Offline bridge regression: only the kernel and model transport are fake.
//! The ipython factory, tool wrapper, agent loop, public envelope and Jev trace
//! are production paths. Structured receipts are not test attestations.

use super::JevBridgeCore;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::extensions::types::ExtensionEvent;
use crate::core::kernel::shared::{
    parse_execution_reports, AbortSignal, BoxFuture, ExecError, ExecuteResult, ExecuteStatus,
    KernelError, KernelStartOptions, StreamName,
};
use crate::core::kernel::state_snapshot::RestoreResult;
use crate::core::tools::ipython::{
    create_ipython_tool, IpythonKernelProvisioner, IpythonToolOptions, KernelClient, PruneResult,
};
use pi_agent_core::agent_loop::run_agent_loop;
use pi_agent_core::types::{AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, StreamFn};
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Message, Model, TextContent, ToolCall,
    UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_jev::config::JevSettings;
use pi_jev::observation::{RetryFailureKind, VerificationEvidence};
use serde_json::{json, Value};

const CELL: &str = "# native tool-error bridge fixture\npass";

fn clean_result() -> ExecuteResult {
    let mut result = ExecuteResult::aborted(1.0);
    result.status = ExecuteStatus::Ok;
    result
}

struct FixtureKernel {
    result: ExecuteResult,
    bootstrap_calls: AtomicUsize,
    cell_calls: AtomicUsize,
}

impl KernelClient for FixtureKernel {
    fn is_running(&self) -> bool { true }
    fn is_defunct(&self) -> bool { false }
    fn start(&self, _options: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
        Box::pin(async { Ok(()) })
    }
    fn execute(
        &self,
        code: &str,
        _signal: Option<AbortSignal>,
        _on_stream: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
    ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
        let result = if code == CELL {
            self.cell_calls.fetch_add(1, Ordering::SeqCst);
            self.result.clone()
        } else {
            self.bootstrap_calls.fetch_add(1, Ordering::SeqCst);
            clean_result()
        };
        Box::pin(async move { Ok(result) })
    }
    fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
        Box::pin(async { Ok(None) })
    }
    fn shutdown(&self, _snapshot: bool, _drain: bool) -> BoxFuture<'static, Result<(), KernelError>> {
        Box::pin(async { Ok(()) })
    }
    fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> { Box::pin(async { Ok(()) }) }
    fn prune_oversized_variables(&self) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
        Box::pin(async { Ok(None) })
    }
    fn list_namespace_names(&self, _signal: Option<AbortSignal>) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
        Box::pin(async { Ok(None) })
    }
}

fn reported_result(code: i64, expected: Vec<i64>, receipt: Value, error: bool) -> ExecuteResult {
    // This is the real parser for the kernel's explicit executionReports field.
    // It rejects contradictory flags; the test never scans printed JSON/text.
    let wire = json!([{
        "schema":"optimus.script-result.v1", "stage":"fixture-check", "scriptId":"fixture-1",
        "exitCode":code, "durationSeconds":0.01, "expectedExitCodes":expected,
        "isError":error, "receipt":receipt
    }]);
    let mut result = clean_result();
    result.execution_reports = parse_execution_reports(Some(&wire)).expect("valid typed report");
    result
}

async fn run_case(name: &str, kernel_result: ExecuteResult, expected_error: bool) {
    let workspace = tempfile::tempdir().unwrap();
    let cwd = workspace.path().to_string_lossy().into_owned();
    let kernel = Arc::new(FixtureKernel {
        result: kernel_result.clone(), bootstrap_calls: AtomicUsize::new(0), cell_calls: AtomicUsize::new(0),
    });
    let provisioner = IpythonKernelProvisioner::new(&cwd, None, Arc::new({
        let kernel = kernel.clone();
        move |_| kernel.clone()
    }));
    let tool = create_ipython_tool(&cwd, Some(IpythonToolOptions {
        provisioner: Some(provisioner.clone()), ..Default::default()
    }));
    let model = Model::new("fixture", "Fixture", "offline-tool-error", "offline-tool-error", "http://example.invalid");
    let call = AssistantMessage {
        api: model.api.clone(), provider: model.provider.clone(), model: model.id.clone(),
        content: vec![ContentBlock::ToolCall(ToolCall::new("fixture-call", "ipython", json!({"code":CELL}).as_object().unwrap().clone()))],
        stop_reason: "toolUse".to_string(), ..Default::default()
    };
    let terminal = AssistantMessage {
        content: vec![ContentBlock::Text(TextContent::new("Fixture done; verification remains unknown."))],
        stop_reason: "stop".to_string(), ..Default::default()
    };
    let replies = Arc::new(Mutex::new(VecDeque::from([call, terminal])));
    let provider_tools = Arc::new(Mutex::new(Vec::new()));
    let stream_fn: StreamFn = Arc::new({
        let replies = replies.clone();
        let provider_tools = provider_tools.clone();
        move |_, context, _| {
            provider_tools.lock().unwrap().push(context.tools.clone());
            let message = replies.lock().unwrap().pop_front().expect("no unbounded extra model turn");
            Box::pin(async move {
                let stream = AssistantMessageEventStream::new();
                stream.push(AssistantMessageEvent::Done { reason: message.stop_reason.clone(), message });
                stream
            })
        }
    });
    let trace = Arc::new(JevBridgeCore::new(JevSettings::default()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let emit = Arc::new({
        let trace = trace.clone();
        let events = events.clone();
        move |event: AgentEvent| {
            if matches!(event, AgentEvent::ToolExecutionEnd { .. }) {
                // Consume the ACTUAL emitted event, not a constructed error flag.
                let wire = serde_json::to_value(&event).unwrap();
                let extension_event: ExtensionEvent = serde_json::from_value(wire).unwrap();
                trace.note_observation("fixture-session", &extension_event);
            }
            events.lock().unwrap().push(event);
            Box::pin(async { Ok(()) }) as futures::future::BoxFuture<'static, anyhow::Result<()>>
        }
    });
    let messages = tokio::time::timeout(std::time::Duration::from_secs(5), run_agent_loop(
        vec![UserMessage::new(UserContent::Text("Run the fixture once".to_string()), 0).into()],
        AgentContext { system_prompt: String::new(), messages: Vec::new(), tools: Some(vec![tool]) },
        AgentLoopConfig::new(model), emit, None, Some(stream_fn),
    )).await.expect("local fixture bound").expect("real agent loop");
    assert_eq!(kernel.cell_calls.load(Ordering::SeqCst), 1, "{name}");
    assert_eq!(kernel.bootstrap_calls.load(Ordering::SeqCst), 1, "{name}");
    assert!(replies.lock().unwrap().is_empty());
    let requests = provider_tools.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|tools| tools.as_ref().is_some_and(|tools| tools.iter().any(|tool| tool.name == "ipython"))));
    drop(requests);

    let public: Vec<_> = messages.iter().filter_map(|message| match message {
        AgentMessage::Message(Message::ToolResult(result)) => Some(result), _ => None,
    }).collect();
    assert_eq!(public.len(), 1, "{name}");
    assert_eq!(public[0].is_error, expected_error, "public ToolResultMessage {name}");
    assert_eq!(serde_json::to_value(public[0]).unwrap()["isError"], expected_error);
    let events = events.lock().unwrap();
    let ended: Vec<_> = events.iter().filter_map(|event| match event {
        AgentEvent::ToolExecutionEnd { result, is_error, .. } => Some((result, is_error)), _ => None,
    }).collect();
    assert_eq!(ended.len(), 1, "{name}");
    assert_eq!(ended[0].0.is_error, Some(expected_error), "factory result {name}");
    assert_eq!(*ended[0].1, expected_error, "event result {name}");
    assert_eq!(public[0].details.as_ref(), Some(&ended[0].0.details));
    assert_eq!(ended[0].0.details["executionReports"].is_array(), kernel_result.execution_reports.is_some());
    drop(events);

    let summary = trace.observation("fixture-session");
    assert_eq!(summary.tool_results, 1, "{name}");
    assert_eq!(summary.tool_errors, u64::from(expected_error), "{name}");
    assert_eq!(summary.failure_kind, expected_error.then_some(RetryFailureKind::ToolFailure));
    assert_eq!(summary.verification, VerificationEvidence::Unknown, "transport/tool success is never verification");
    provisioner.dispose(None).await;
}

#[tokio::test]
async fn stop001_real_ipython_factory_public_envelope_and_jev_trace() {
    let mut error = clean_result();
    error.status = ExecuteStatus::Error;
    error.error = Some(ExecError { ename: "ValueError".to_string(), evalue: "fixture".to_string(), traceback: vec!["fixture traceback".to_string()] });
    run_case("kernel Error", error, true).await;
    run_case("kernel Aborted", ExecuteResult::aborted(1.0), true).await;
    run_case("script nonzero", reported_result(7, vec![0], Value::Null, true), true).await;
    run_case("failed receipt exit zero", reported_result(0, vec![0], json!({"status":"error"}), true), true).await;
    run_case("fatal receipt cannot be expected away", reported_result(0, vec![0,7], json!({"fatal":true,"status":"fatal"}), true), true).await;
    run_case("handled expected nonzero", reported_result(7, vec![0,7], Value::Null, false), false).await;
    let mut legacy = clean_result();
    legacy.stdout = "BashResult(exit_code=7): unreported legacy process result".to_string();
    run_case("legacy nonzero text is not metadata", legacy, false).await;
    let mut printed = clean_result();
    printed.stdout = "Traceback (most recent call last):\nValueError: example source text".to_string();
    printed.stderr = "fatal error is an example string, not a receipt".to_string();
    run_case("arbitrary Traceback text", printed, false).await;
}
