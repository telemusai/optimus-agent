use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct ControlledKernel {
    entered: CancellationToken,
    start_release: CancellationToken,
    stopping: CancellationToken,
    stop_release: CancellationToken,
    stops: Arc<AtomicUsize>,
}
impl KernelClient for ControlledKernel {
    fn is_running(&self) -> bool {
        self.entered.is_cancelled() && self.stops.load(Ordering::SeqCst) == 0
    }
    fn is_defunct(&self) -> bool {
        self.stops.load(Ordering::SeqCst) > 0
    }
    fn start(&self, _: KernelStartOptions) -> BoxFuture<'static, Result<(), KernelError>> {
        let entered = self.entered.clone();
        let release = self.start_release.clone();
        async move {
            entered.cancel();
            release.cancelled().await;
            Ok(())
        }
        .boxed()
    }
    fn execute(
        &self,
        _: &str,
        _: Option<AbortSignal>,
        _: Option<Arc<dyn Fn(&str, StreamName) + Send + Sync>>,
    ) -> BoxFuture<'static, Result<ExecuteResult, KernelError>> {
        async {
            let mut result = ExecuteResult::aborted(0.0);
            result.status = ExecuteStatus::Ok;
            Ok(result)
        }
        .boxed()
    }
    fn restore_state(&self) -> BoxFuture<'static, Result<Option<RestoreResult>, KernelError>> {
        async { Ok(None) }.boxed()
    }
    fn shutdown(&self, _: bool, _: bool) -> BoxFuture<'static, Result<(), KernelError>> {
        self.kill()
    }
    fn kill(&self) -> BoxFuture<'static, Result<(), KernelError>> {
        let entered = self.stopping.clone();
        let release = self.stop_release.clone();
        let stops = self.stops.clone();
        async move {
            entered.cancel();
            release.cancelled().await;
            stops.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        .boxed()
    }
    fn prune_oversized_variables(
        &self,
    ) -> BoxFuture<'static, Result<Option<PruneResult>, KernelError>> {
        async { Ok(None) }.boxed()
    }
    fn list_namespace_names(
        &self,
        _: Option<AbortSignal>,
    ) -> BoxFuture<'static, Result<Option<Vec<String>>, KernelError>> {
        async { Ok(None) }.boxed()
    }
}

async fn until(token: &CancellationToken) {
    tokio::time::timeout(Duration::from_secs(3), token.cancelled())
        .await
        .unwrap();
}

#[tokio::test]
async fn kill_during_startup_never_publishes_old_manager_or_overlaps_replacement() {
    let old = Arc::new(ControlledKernel::default());
    let new = Arc::new(ControlledKernel::default());
    new.start_release.cancel();
    new.stop_release.cancel();
    let creates = Arc::new(AtomicUsize::new(0));
    let provisioner = IpythonKernelProvisioner::new(
        "/tmp",
        None,
        Arc::new({
            let old = old.clone();
            let new = new.clone();
            let creates = creates.clone();
            move |_| {
                if creates.fetch_add(1, Ordering::SeqCst) == 0 {
                    old.clone()
                } else {
                    new.clone()
                }
            }
        }),
    );
    let first = tokio::spawn({
        let p = provisioner.clone();
        async move { p.ensure(None, None).await }
    });
    until(&old.entered).await;
    let stop = provisioner.kill();
    tokio::pin!(stop);
    assert!(futures::poll!(&mut stop).is_pending());
    let second = provisioner.ensure(None, None);
    tokio::pin!(second);
    assert!(futures::poll!(&mut second).is_pending());
    assert_eq!(creates.load(Ordering::SeqCst), 1);
    old.start_release.cancel();
    until(&old.stopping).await;
    assert!(
        first.await.unwrap().is_err(),
        "late startup must not hand out the retired manager"
    );
    assert!(provisioner.manager().is_none());
    assert!(
        futures::poll!(&mut second).is_pending(),
        "must wait for teardown, not just startup"
    );
    old.stop_release.cancel();
    stop.await;
    let replacement = tokio::time::timeout(Duration::from_secs(3), second)
        .await
        .unwrap()
        .unwrap();
    let expected: Arc<dyn KernelClient> = new;
    assert!(Arc::ptr_eq(&replacement, &expected));
    assert_eq!(creates.load(Ordering::SeqCst), 2);
    assert_eq!(old.stops.load(Ordering::SeqCst), 1);
    provisioner.dispose(Some(false)).await;
}

#[tokio::test]
async fn concurrent_disposal_joins_flush_even_if_first_caller_is_cancelled() {
    let client = Arc::new(ControlledKernel::default());
    client.start_release.cancel();
    let provisioner = IpythonKernelProvisioner::new(
        "/tmp",
        None,
        Arc::new({
            let client = client.clone();
            move |_| client.clone()
        }),
    );
    provisioner.ensure(None, None).await.unwrap();
    let first = tokio::spawn({
        let p = provisioner.clone();
        async move { p.dispose(None).await }
    });
    until(&client.stopping).await;
    first.abort();
    let _ = first.await;
    let second = provisioner.dispose(None);
    tokio::pin!(second);
    assert!(
        futures::poll!(&mut second).is_pending(),
        "all disposal callers must join the flush"
    );
    assert!(provisioner.ensure(None, None).await.is_err());
    assert!(provisioner.manager().is_none());
    client.stop_release.cancel();
    tokio::time::timeout(Duration::from_secs(3), second)
        .await
        .unwrap();
    assert_eq!(client.stops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn abort_while_waiting_for_stop_does_not_cancel_teardown_or_start_replacement() {
    let client = Arc::new(ControlledKernel::default());
    client.start_release.cancel();
    let provisioner = IpythonKernelProvisioner::new(
        "/tmp",
        None,
        Arc::new({
            let client = client.clone();
            move |_| client.clone()
        }),
    );
    provisioner.ensure(None, None).await.unwrap();
    let stopping = provisioner.begin_stop(None);
    until(&client.stopping).await;
    let signal = AbortSignal::new();
    let waiting = provisioner.ensure(None, Some(signal.clone()));
    tokio::pin!(waiting);
    assert!(futures::poll!(&mut waiting).is_pending());
    signal.abort(None);
    assert!(waiting.await.is_err());
    assert_eq!(client.stops.load(Ordering::SeqCst), 0);
    client.stop_release.cancel();
    stopping.await.unwrap();
    assert_eq!(client.stops.load(Ordering::SeqCst), 1);
}
