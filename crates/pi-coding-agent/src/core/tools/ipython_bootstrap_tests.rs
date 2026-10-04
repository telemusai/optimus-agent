use super::*;
use crate::core::kernel::repl_manager::new_repl_kernel_manager;
use std::time::{Duration, Instant};

fn fixture() -> (tempfile::TempDir, Arc<IpythonKernelProvisioner>) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("rlm")).unwrap();
    std::fs::write(dir.path().join("rlm/__init__.py"), "").unwrap();
    // Real stdio kernel protocol: ready succeeds, but the first bootstrap stalls.
    // Shutdown remains responsive so the test also verifies orderly teardown.
    std::fs::write(dir.path().join("rlm/repl.py"), r#"
import json, os, pathlib, sys, time
root = pathlib.Path(__file__).parent.parent
with (root / 'starts').open('a') as f:
    f.write(str(os.getpid()) + '\n')
print(json.dumps({'event':'ready', 'protocol':3}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    with (root / 'requests').open('a') as f:
        f.write(request['type'] + '\n')
    if request['type'] == 'execute':
        if not (root / 'enabled').exists():
            continue
        if request['code'] == 'slow user work':
            time.sleep(0.3)
        print(json.dumps({'event':'done', 'id':request['id'], 'status':'ok'}), flush=True)
    elif request['type'] == 'restore':
        print(json.dumps({'event':'done', 'id':request['id'], 'status':'ok', 'restored':[], 'failed':[]}), flush=True)
    elif request['type'] == 'shutdown':
        print(json.dumps({'event':'done', 'id':request['id'], 'status':'ok'}), flush=True)
        break
"#).unwrap();
    let python = std::env::var("KERNEL_CONTAINMENT_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let path = dir.path().to_string_lossy().into_owned();
    let mut provisioner = IpythonKernelProvisioner::new(
        &path,
        Some(IpythonToolOptions {
            python: Some(python),
            env: Some(vec![
                ("PYTHONPATH".into(), path.clone()),
                ("PRIME_AGENT_CODING_AGENT_DIR".into(), dir.path().join("profile").to_string_lossy().into_owned()),
            ]),
            // A failed bootstrap must not save an incomplete namespace here.
            snapshot_dir: Some(dir.path().join("snapshot").to_string_lossy().into_owned()),
            ..Default::default()
        }),
        Arc::new(|options| Arc::new(ReplKernelClient {
            manager: new_repl_kernel_manager(options),
        })),
    );
    Arc::get_mut(&mut provisioner).unwrap().bootstrap_timeout = Duration::from_millis(100);
    (dir, provisioner)
}

#[tokio::test]
async fn ready_kernel_with_stuck_bootstrap_is_stopped_once_and_can_retry_fresh() {
    let (dir, provisioner) = fixture();
    let started = Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(10), provisioner.ensure(None, None))
        .await.expect("bootstrap hung after ready");
    let error = result.err().expect("unresponsive bootstrap was published");
    assert!(error.to_string().contains("Python runtime bootstrap did not finish"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(provisioner.manager().is_none());
    assert!(provisioner.last_restore_for_tests().is_none());
    let starts = std::fs::read_to_string(dir.path().join("starts")).unwrap();
    assert_eq!(starts.lines().count(), 1, "bootstrap failure must not auto-retry");
    let requests = std::fs::read_to_string(dir.path().join("requests")).unwrap();
    assert!(requests.ends_with("shutdown\n"), "{requests}");
    assert!(!requests.contains("snapshot"), "failed bootstrap overwrote saved state: {requests}");

    std::fs::write(dir.path().join("enabled"), "").unwrap();
    let manager = provisioner.ensure(None, None).await.unwrap();
    let started = Instant::now();
    let result = manager.execute("slow user work", None, None).await.unwrap();
    provisioner.dispose(Some(false)).await;
    assert_eq!(result.status, ExecuteStatus::Ok);
    assert!(started.elapsed() >= Duration::from_millis(300), "user cells acquired the bootstrap deadline");
    let starts = std::fs::read_to_string(dir.path().join("starts")).unwrap();
    assert_eq!(starts.lines().count(), 2);
    let pids: Vec<_> = starts.lines().collect();
    assert_ne!(pids[0], pids[1], "retry reused the failed kernel");
}

#[tokio::test]
async fn cancelling_bootstrap_is_not_reported_as_a_timeout() {
    let (dir, mut provisioner) = fixture();
    Arc::get_mut(&mut provisioner).unwrap().bootstrap_timeout = Duration::from_secs(30);
    let signal = AbortSignal::new();
    let p = provisioner.clone();
    let child_signal = signal.clone();
    let pending = tokio::spawn(async move { p.ensure(None, Some(child_signal)).await });
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if std::fs::read_to_string(dir.path().join("requests")).unwrap_or_default().contains("execute") { break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await.unwrap();
    signal.abort(Some(KernelError::new("user cancelled startup")));
    let result = tokio::time::timeout(Duration::from_secs(5), pending).await.unwrap().unwrap();
    let error = result.err().expect("cancelled bootstrap succeeded");
    assert!(!error.to_string().contains("did not finish within"), "{error}");
    // Dispose owns and joins the abandoned startup cleanup.
    provisioner.dispose(Some(false)).await;
    assert!(provisioner.manager().is_none());
}
