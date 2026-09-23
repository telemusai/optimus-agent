//! Per-agent local work ownership. A cancelled loop is not task settlement.
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use pi_ai::utils::event_stream::StreamTaskReceipt;
use tokio::task::{AbortHandle, JoinHandle};
tokio::task_local! { static CURRENT: ExecutionScope; }

/// Known ipython cancellation adapters use this rather than an unowned spawn.
pub fn spawn_auxiliary(future: impl Future<Output = ()> + Send + 'static) {
    match CURRENT.try_with(Clone::clone) {
        Ok(scope) => { let _ = scope.spawn(future, true, true); }
        Err(_) => { tokio::spawn(future); }
    }
}

#[derive(Default)]
struct State {
    fenced: bool,
    pending: usize,
    failed: bool,
    uncovered: bool,
    tasks: Vec<OwnedTask>,
    streams: Vec<StreamTaskReceipt>,
}
#[derive(Clone, Default)]
pub struct ExecutionScope(Arc<Mutex<State>>, Arc<tokio::sync::Notify>);
#[derive(Clone, Copy, Debug)]
pub struct ExecutionStatus {
    pub fenced: bool,
    pub tools_settled: bool,
    pub model_settled: bool,
    pub supported: bool,
    pub failed: bool,
}
struct OwnedTask {
    terminal: AbortHandle,
    abort_on_stop: bool,
}

// Rust drops struct fields in declaration order, including before first poll.
// Never capture the guard separately in an async block: its capture can be
// dropped before the future it is supposed to acknowledge.
struct OwnedFuture<F> {
    future: Pin<Box<F>>,
    _completion: Completion,
}

impl<F: Future> Future for OwnedFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().future.as_mut().poll(cx)
    }
}

struct Completion(ExecutionScope);
impl Drop for Completion {
    fn drop(&mut self) {
        let mut state = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
        state.pending -= 1;
        state.failed |= std::thread::panicking();
        self.0.1.notify_waiters();
    }
}
impl ExecutionScope {
    /// Own the future and its captured payload through actual task termination.
    /// The caller still owns JoinHandle/output consumption; arbitrary resource-
    /// bearing return values are outside this receipt's future-lifetime proof.
    pub fn spawn<F>(&self, future: F, abort_on_stop: bool, covered: bool) -> Result<JoinHandle<F::Output>, &'static str>
    where F: Future + Send + 'static, F::Output: Send + 'static {
        let runtime = tokio::runtime::Handle::try_current().map_err(|_| "execution scope requires Tokio runtime")?;
        let mut state = self.0.lock().unwrap();
        if state.fenced { return Err("execution scope stopped"); }
        state.uncovered |= !covered;
        state.pending += 1;
        let owned = OwnedFuture {
            future: Box::pin(future),
            _completion: Completion(self.clone()),
        };
        let task = runtime.spawn(CURRENT.scope(self.clone(), owned));
        // JoinHandle remains owned by the caller. Its cloneable AbortHandle
        // shares Tokio's terminal bit, set only after the task future is dropped.
        // Keep it even for tasks that must finish cooperative cleanup on stop.
        state.tasks.push(OwnedTask { terminal: task.abort_handle(), abort_on_stop });
        Ok(task)
    }
    pub fn register_stream(&self, receipt: StreamTaskReceipt) -> Result<(), &'static str> {
        let mut state = self.0.lock().unwrap();
        if state.fenced { receipt.request_cancel(); return Err("execution scope stopped"); }
        state.streams.push(receipt);
        Ok(())
    }
    /// Cancellation fence is synchronous. Tools retain their cleanup futures.
    pub fn request_cancel(&self) {
        let mut state = self.0.lock().unwrap();
        state.fenced = true;
        for stream in &state.streams { stream.request_cancel(); }
        for task in &state.tasks {
            if task.abort_on_stop { task.terminal.abort(); }
        }
    }
    pub fn status(&self) -> ExecutionStatus {
        let state = self.0.lock().unwrap();
        let statuses: Vec<_> = state.streams.iter().map(|stream| stream.status()).collect();
        let terminal = state.pending == 0 && state.tasks.iter().all(|task| task.terminal.is_finished());
        ExecutionStatus {
            fenced: state.fenced,
            tools_settled: state.fenced && terminal && !state.failed,
            model_settled: state.fenced && terminal && !state.failed && statuses.iter().all(|s| s.settled),
            supported: !state.uncovered && statuses.iter().all(|s| s.supported),
            failed: state.failed || statuses.iter().any(|s| s.failed_tasks > 0),
        }
    }
    pub async fn settle(&self, timeout: std::time::Duration) -> ExecutionStatus {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let wake = self.1.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if self.0.lock().unwrap().pending == 0 { break; }
            if tokio::time::timeout_at(deadline, wake).await.is_err() { return self.status(); }
        }
        // Completion::drop notifies before Tokio publishes its terminal bit.
        // Yield through that final runtime transition rather than treating the
        // guard counter as a JoinHandle acknowledgement. The original bound
        // still covers this wait, and cancellation keeps every receipt owned.
        let terminal = async {
            loop {
                if self.0.lock().unwrap().tasks.iter().all(|task| task.terminal.is_finished()) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        };
        if tokio::time::timeout_at(deadline, terminal).await.is_err() { return self.status(); }
        let streams = self.0.lock().unwrap().streams.clone();
        for stream in streams {
            stream.settle(deadline.saturating_duration_since(tokio::time::Instant::now())).await;
        }
        self.status()
    }
    /// Only a separately authorized audit task can reset a fully settled scope.
    pub fn reset_after_settlement(&self) -> Result<(), &'static str> {
        let status = self.status();
        if !status.supported || !status.tools_settled || !status.model_settled {
            return Err("execution scope is not settled");
        }
        *self.0.lock().unwrap() = State::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn detached_tool_cannot_be_mistaken_for_settlement() {
        let scope = ExecutionScope::default();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let task = scope.spawn(async move { let _ = rx.await; }, false, true).unwrap();
        drop(task);
        scope.request_cancel();
        assert!(!scope.status().tools_settled);
        assert!(scope.spawn(async {}, false, true).is_err());
        tx.send(()).unwrap();
        assert!(scope.settle(std::time::Duration::from_secs(1)).await.tools_settled);
        scope.reset_after_settlement().unwrap();
        assert!(!scope.status().fenced);
    }
    struct HeldDropFuture {
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
        polled: Arc<std::sync::atomic::AtomicBool>,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Future for HeldDropFuture {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            self.polled.store(true, std::sync::atomic::Ordering::SeqCst);
            Poll::Pending
        }
    }
    impl Drop for HeldDropFuture {
        fn drop(&mut self) {
            let _ = self.entered.send(());
            self.release.recv_timeout(std::time::Duration::from_secs(5)).expect("release held destructor");
            self.dropped.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn abort_before_first_poll_keeps_destructor_owned_until_runtime_terminal() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let scope = ExecutionScope::default();
        let (entered, ready) = std::sync::mpsc::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let polled = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let inspecting_scope = scope.clone();
        let inspecting_polled = polled.clone();
        let inspecting_dropped = dropped.clone();
        let inspector = std::thread::spawn(move || {
            ready.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            let status = inspecting_scope.status();
            assert!(!status.tools_settled);
            assert!(!status.model_settled);
            assert!(!status.failed);
            assert!(!inspecting_polled.load(Ordering::SeqCst));
            assert!(!inspecting_dropped.load(Ordering::SeqCst));
            // An expired waiter must neither acknowledge nor forget the task.
            let waiter_runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let timed_out = waiter_runtime.block_on(inspecting_scope.settle(std::time::Duration::ZERO));
            assert!(!timed_out.tools_settled);
            assert!(!timed_out.model_settled);
            release.send(()).unwrap();
        });
        runtime.block_on(async {
            let task = scope.spawn(HeldDropFuture { entered, release: blocked, polled: polled.clone(), dropped: dropped.clone() }, true, true).unwrap();
            // Current-thread runtime cannot poll the spawned task until this await.
            scope.request_cancel();
            assert!(!task.is_finished());
            assert!(task.await.unwrap_err().is_cancelled());
            let done = scope.settle(std::time::Duration::from_secs(1)).await;
            assert!(done.tools_settled && done.model_settled && done.supported);
            assert!(!done.failed);
        });
        inspector.join().unwrap();
        assert!(dropped.load(Ordering::SeqCst));
        assert!(!polled.load(Ordering::SeqCst));
    }

    struct PanicOnDrop;
    impl Future for PanicOnDrop {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }
    impl Drop for PanicOnDrop {
        fn drop(&mut self) { panic!("synthetic destructor panic"); }
    }

    #[tokio::test]
    async fn abort_before_first_poll_destructor_panic_is_sticky_failure() {
        let scope = ExecutionScope::default();
        let task = scope.spawn(PanicOnDrop, true, true).unwrap();
        scope.request_cancel();
        assert!(task.await.unwrap_err().is_panic());
        let done = scope.settle(std::time::Duration::from_secs(1)).await;
        assert!(done.failed);
        assert!(!done.tools_settled);
        assert!(!done.model_settled);
        assert!(scope.reset_after_settlement().is_err());
    }

    #[tokio::test]
    async fn running_cancellation_preserves_joinhandle_contract_and_drops_payload() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Mark(Arc<AtomicBool>);
        impl Drop for Mark {
            fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
        }
        let scope = ExecutionScope::default();
        let dropped = Arc::new(AtomicBool::new(false));
        let mark = Mark(dropped.clone());
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = scope.spawn(async move {
            let _mark = mark;
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        }, true, true).unwrap();
        ready.await.unwrap();
        scope.request_cancel();
        let done = scope.settle(std::time::Duration::from_secs(1)).await;
        assert!(done.tools_settled && done.model_settled);
        assert!(dropped.load(Ordering::SeqCst));
        assert!(task.is_finished());
        assert!(task.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn nonaborted_task_terminal_and_return_value_remain_owned_by_caller() {
        let scope = ExecutionScope::default();
        let (release, ready) = tokio::sync::oneshot::channel();
        let task = scope.spawn(async move { ready.await.unwrap(); 37_u32 }, false, true).unwrap();
        scope.request_cancel();
        assert!(!scope.settle(std::time::Duration::ZERO).await.tools_settled);
        release.send(()).unwrap();
        let done = scope.settle(std::time::Duration::from_secs(1)).await;
        assert!(done.tools_settled && done.model_settled);
        assert!(task.is_finished());
        assert_eq!(task.await.unwrap(), 37);
    }

    #[tokio::test]
    async fn abort_acknowledgement_and_unsupported_tools_are_truthful() {
        let scope = ExecutionScope::default();
        let task = scope.spawn(std::future::pending::<()>(), true, false).unwrap();
        scope.request_cancel();
        assert!(!scope.status().tools_settled);
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(scope.status().tools_settled);
        assert!(!scope.status().supported);
        assert!(scope.reset_after_settlement().is_err());
    }
}
