use pi_coding_agent::core::tools::node::NodeRuntime;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn text(result: &pi_agent_core::types::AgentToolResult) -> String {
    serde_json::to_string(&result.content).unwrap()
}

#[tokio::test]
async fn persistent_javascript_await_files_modules_errors_and_reset() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_str().unwrap();
    let runtime = NodeRuntime::default();
    let first = runtime
        .execute(
            cwd,
            "const counter = 41; console.log('ready'); counter",
            10.,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(text(&first).contains("ready") && text(&first).contains("41"));
    let second = runtime
        .execute(cwd, "await Promise.resolve(counter + 1)", 10., None, None)
        .await
        .unwrap();
    assert!(text(&second).contains("42"));
    let io = runtime.execute(cwd, "var fs = require('node:fs/promises'); await fs.writeFile('node-test.txt', '8080'); await fs.readFile('node-test.txt', 'utf8')", 10., None, None).await.unwrap();
    assert!(text(&io).contains("8080"));
    let import = runtime
        .execute(
            cwd,
            "var imported = await nodeImport('node:path'); imported.basename('/tmp/test')",
            10.,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(text(&import).contains("test"));
    let package = dir.path().join("node_modules/optimus-esm-fixture");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(
        package.join("package.json"),
        r#"{"type":"module","exports":{"import":"./index.mjs"}}"#,
    )
    .unwrap();
    std::fs::write(package.join("index.mjs"), "export default 42;").unwrap();
    let esm = runtime
        .execute(
            cwd,
            "(await nodeImport('optimus-esm-fixture')).default",
            10.,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(esm.is_error, Some(false));
    assert!(text(&esm).contains("42"));
    let error = runtime
        .execute(cwd, "throw new Error('deliberate')", 10., None, None)
        .await
        .unwrap();
    assert_eq!(error.is_error, Some(true));
    assert!(text(
        &runtime
            .execute(cwd, "counter + 2", 10., None, None)
            .await
            .unwrap()
    )
    .contains("43"));
    let timeout = runtime
        .execute(cwd, "while (true) {}", 0.1, None, None)
        .await
        .unwrap_err();
    assert!(timeout.contains("timed out") && timeout.contains("reset"));
    let reset = runtime
        .execute(cwd, "typeof counter", 10., None, None)
        .await
        .unwrap();
    assert!(text(&reset).contains("undefined"));
    runtime.dispose().await;
}

#[tokio::test]
async fn cancellation_and_large_output_do_not_jam_the_next_cell() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_str().unwrap();
    let runtime = NodeRuntime::default();
    let large = runtime
        .execute(cwd, "console.log('🌄'.repeat(100000)); 42", 10., None, None)
        .await
        .unwrap();
    assert!(text(&large).len() < 70000);
    let token = CancellationToken::new();
    let cancel = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();
    });
    let cancelled = tokio::time::timeout(
        Duration::from_secs(5),
        runtime.execute(cwd, "await new Promise(() => {})", 60., Some(token), None),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(cancelled.contains("cancelled") && cancelled.contains("reset"));
    assert!(text(
        &runtime
            .execute(cwd, "21 * 2", 10., None, None)
            .await
            .unwrap()
    )
    .contains("42"));
    runtime.dispose().await;
}

#[tokio::test]
async fn dropped_execution_future_discards_busy_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_str().unwrap();
    let runtime = NodeRuntime::default();
    assert!(tokio::time::timeout(
        Duration::from_millis(200),
        runtime.execute(
            cwd,
            "const abandoned = true; while(true) {}",
            60.,
            None,
            None
        )
    )
    .await
    .is_err());
    let next = tokio::time::timeout(
        Duration::from_secs(5),
        runtime.execute(cwd, "typeof abandoned", 10., None, None),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(text(&next).contains("undefined"));
    runtime.dispose().await;
}
