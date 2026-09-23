//! Port of packages/ai/src/utils/event-stream.ts
//!
//! The TypeScript `EventStream` is an async-iterable queue with waiter callbacks.
//! The port uses `Arc` + `tokio::sync::Mutex` + `Notify` so the same object can be
//! cloned into background tasks (the TS `new AssistantMessageEventStream(); (async () => {})()`
//! pattern) and consumed with `next()` / `into_stream()`.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::{join_all, BoxFuture, Shared};
use futures::FutureExt;
use tokio::sync::Notify;
use tokio::task::AbortHandle;

use crate::types::{AssistantMessage, AssistantMessageEvent};

/// Local task acknowledgement, not proof that a remote vendor stopped computing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamTaskStatus {
    pub supported: bool,
    pub cancel_requested: bool,
    pub pending_tasks: usize,
    pub completed_tasks: usize,
    pub cancelled_tasks: usize,
    pub failed_tasks: usize,
    pub settled: bool,
}

#[derive(Clone)]
enum TaskOutcome {
    Completed,
    Cancelled,
    Failed,
}

type TaskJoin = Shared<BoxFuture<'static, TaskOutcome>>;

struct OwnedTask {
    abort: AbortHandle,
    join: TaskJoin,
}

#[derive(Default)]
struct StreamTaskState {
    cancel_requested: bool,
    covered_streams: usize,
    uncovered_streams: usize,
    pending: Vec<OwnedTask>,
    completed: usize,
    cancelled: usize,
    failed: usize,
    close: Vec<Box<dyn Fn() + Send + Sync>>,
}

/// Retain this before invoking a deferred stream function. Channel closure and
/// cancellation admission are never join acknowledgements. Timeout/cancelled
/// waiters only drop clones of shared joins; this receipt keeps the originals.
#[derive(Clone)]
pub struct StreamTaskReceipt(Arc<Mutex<StreamTaskState>>);

tokio::task_local! {
    static STREAM_TASK_RECEIPT: StreamTaskReceipt;
}

impl StreamTaskReceipt {
    pub fn new_unsupported() -> Self {
        Self(Arc::new(Mutex::new(StreamTaskState::default())))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StreamTaskState> {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The scope must include calling the stream function, not just awaiting
    /// the returned future. Owned tasks inherit it; arbitrary tokio tasks do not.
    pub async fn scope<F: Future>(&self, future: F) -> F::Output {
        STREAM_TASK_RECEIPT.scope(self.clone(), future).await
    }

    fn register_stream(&self, covered: bool, close: Box<dyn Fn() + Send + Sync>) {
        let mut state = self.lock();
        if covered {
            state.covered_streams += 1;
        } else {
            state.uncovered_streams += 1;
        }
        if state.cancel_requested {
            close();
        }
        state.close.push(close);
    }

    fn spawn<F>(&self, future: F) -> bool
    where
        F: Future<Output = ()> + Send + 'static,
    {
        // Admission and insertion share the cancellation fence. No task may be
        // spawned and then registered after a concurrent stop snapshot.
        let mut state = self.lock();
        if state.cancel_requested {
            return false;
        }
        let receipt = self.clone();
        let task = tokio::spawn(async move { receipt.scope(future).await });
        let abort = task.abort_handle();
        let join = async move {
            match task.await {
                Ok(()) => TaskOutcome::Completed,
                Err(error) if error.is_cancelled() => TaskOutcome::Cancelled,
                Err(_) => TaskOutcome::Failed,
            }
        }.boxed().shared();
        state.pending.push(OwnedTask { abort, join });
        true
    }

    /// Immediate, idempotent stop request. Call status/settle separately for proof.
    pub fn request_cancel(&self) -> StreamTaskStatus {
        {
            let mut state = self.lock();
            state.cancel_requested = true;
            for task in &state.pending {
                task.abort.abort();
            }
            for close in &state.close {
                close();
            }
        }
        self.status()
    }

    /// Nonblocking join collection. No error/panic payload or model content is exposed.
    pub fn status(&self) -> StreamTaskStatus {
        let mut state = self.lock();
        let mut remaining = Vec::new();
        for task in std::mem::take(&mut state.pending) {
            match task.join.clone().now_or_never() {
                Some(TaskOutcome::Completed) => state.completed += 1,
                Some(TaskOutcome::Cancelled) => state.cancelled += 1,
                Some(TaskOutcome::Failed) => state.failed += 1,
                None => remaining.push(task),
            }
        }
        state.pending = remaining;
        let supported = state.covered_streams > 0 && state.uncovered_streams == 0;
        StreamTaskStatus {
            supported,
            cancel_requested: state.cancel_requested,
            pending_tasks: state.pending.len(),
            completed_tasks: state.completed,
            cancelled_tasks: state.cancelled,
            failed_tasks: state.failed,
            settled: supported && state.cancel_requested && state.pending.is_empty() && state.failed == 0,
        }
    }

    /// Wait at most `timeout`, without changing active-provider deadlines/retries.
    /// A timed-out or dropped waiter never relinquishes ownership of pending joins.
    pub async fn settle(&self, timeout: Duration) -> StreamTaskStatus {
        let wait = async {
            loop {
                let status = self.status();
                if status.pending_tasks == 0 {
                    return;
                }
                let joins: Vec<_> = self.lock().pending.iter().map(|task| task.join.clone()).collect();
                join_all(joins).await;
            }
        };
        let _ = tokio::time::timeout(timeout, wait).await;
        self.status()
    }
}

struct StreamConsumerLease {
    tasks: StreamTaskReceipt,
    stream: EventStream<AssistantMessageEvent, AssistantMessage>,
}

impl Drop for StreamConsumerLease {
    fn drop(&mut self) {
        // Normal terminal delivery must not discard asynchronous usage/ledger
        // observers. A retained receipt still owns and can stop those tasks.
        if self.stream.result_if_ready().is_none() {
            self.tasks.request_cancel();
        }
    }
}

type IsComplete<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;
type ExtractResult<T, R> = Box<dyn Fn(&T) -> R + Send + Sync>;

struct EventStreamState<T, R> {
    queue: VecDeque<T>,
    done: bool,
    result: Option<R>,
    is_complete: IsComplete<T>,
    extract_result: ExtractResult<T, R>,
}

// The state mutex is a std mutex on purpose: `push`/`end` are synchronous (the
// TypeScript calls them from plain callbacks) and the lock is never held across
// an await point.

/// `class EventStream<T, R = T>`.
pub struct EventStream<T, R = T> {
    state: Arc<Mutex<EventStreamState<T, R>>>,
    /// Wakes blocked readers when an event arrives or the stream ends.
    notify: Arc<Notify>,
    /// Wakes `result()` waiters when the final result is available.
    result_notify: Arc<Notify>,
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            notify: self.notify.clone(),
            result_notify: self.result_notify.clone(),
        }
    }
}

impl<T, R> EventStream<T, R> {
    pub fn new(is_complete: IsComplete<T>, extract_result: ExtractResult<T, R>) -> Self {
        Self {
            state: Arc::new(Mutex::new(EventStreamState {
                queue: VecDeque::new(),
                done: false,
                result: None,
                is_complete,
                extract_result,
            })),
            notify: Arc::new(Notify::new()),
            result_notify: Arc::new(Notify::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, EventStreamState<T, R>> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `push(event: T): void`
    pub fn push(&self, event: T) {
        let complete;
        {
            let mut state = self.lock();
            if state.done {
                return;
            }

            complete = (state.is_complete)(&event);
            if complete {
                state.done = true;
                let result = (state.extract_result)(&event);
                state.result = Some(result);
            }
            state.queue.push_back(event);
        }

        self.notify.notify_waiters();
        if complete {
            self.result_notify.notify_waiters();
        }
    }

    /// `end(result?: R): void`
    ///
    /// The TypeScript resolves `finalResultPromise` (event-stream.ts:35-44), and a
    /// promise resolves once: the first resolution wins, so a `push` of a
    /// complete event (event-stream.ts:28-31) already fixes `result()` and a later
    /// `end(result)` cannot replace it. Only set the result when it is still unset.
    pub fn end(&self, result: Option<R>) {
        {
            let mut state = self.lock();
            state.done = true;
            if state.result.is_none() {
                state.result = result;
            }
        }
        self.notify.notify_waiters();
        self.result_notify.notify_waiters();
    }

    /// `for await (const event of stream)` - `None` means the stream ended.
    pub async fn next(&self) -> Option<T> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            // Register interest before reading the state so a concurrent push
            // cannot be missed between the check and the await.
            notified.as_mut().enable();
            {
                let mut state = self.lock();
                if let Some(event) = state.queue.pop_front() {
                    return Some(event);
                }
                if state.done {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// `result(): Promise<R>`
    pub async fn result(&self) -> R
    where
        R: Clone,
    {
        loop {
            let notified = self.result_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.lock();
                if let Some(result) = state.result.as_ref() {
                    return result.clone();
                }
            }
            notified.await;
        }
    }

    pub fn result_if_ready(&self) -> Option<R>
    where
        R: Clone,
    {
        self.lock().result.clone()
    }

    pub async fn result_or_end(&self) -> Option<R>
    where
        R: Clone,
    {
        loop {
            let notified = self.result_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.lock();
                if state.done {
                    return state.result.clone();
                }
            }
            notified.await;
        }
    }

    pub fn is_done(&self) -> bool {
        self.lock().done
    }

    /// The async-iterable form: `AsyncIterator<T>`.
    pub fn into_stream(self) -> impl futures::Stream<Item = T> {
        futures::stream::unfold(self, |stream| async move {
            let next = stream.next().await;
            next.map(|event| (event, stream))
        })
    }
}

/// `class AssistantMessageEventStream extends EventStream<AssistantMessageEvent, AssistantMessage>`.
#[derive(Clone)]
pub struct AssistantMessageEventStream {
    stream: EventStream<AssistantMessageEvent, AssistantMessage>,
    tasks: StreamTaskReceipt,
    _consumer: Option<Arc<StreamConsumerLease>>,
}

impl Default for AssistantMessageEventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl AssistantMessageEventStream {
    pub fn new() -> Self {
        Self::with_ownership(false)
    }

    /// Only audited providers owning every output-producing task may opt in.
    /// Uncovered streams in the same invocation keep its receipt unsupported.
    pub fn new_owned() -> Self {
        Self::with_ownership(true)
    }

    fn with_ownership(covered: bool) -> Self {
        let stream = EventStream::new(
            Box::new(|event| {
                matches!(event, AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. })
            }),
            Box::new(|event| match event {
                AssistantMessageEvent::Done { message, .. } => message.clone(),
                AssistantMessageEvent::Error { error, .. } => error.clone(),
                _ => panic!("Unexpected event type for final result"),
            }),
        );
        let tasks = STREAM_TASK_RECEIPT.try_with(Clone::clone)
            .unwrap_or_else(|_| StreamTaskReceipt::new_unsupported());
        let close_stream = stream.clone();
        tasks.register_stream(covered, Box::new(move || close_stream.end(None)));
        Self {
            _consumer: Some(Arc::new(StreamConsumerLease { tasks: tasks.clone(), stream: stream.clone() })),
            stream,
            tasks,
        }
    }

    /// A producer must not hold a consumer lease, or consumer-drop cancellation
    /// could be kept alive by the very task it needs to cancel.
    pub fn producer_handle(&self) -> Self {
        Self { stream: self.stream.clone(), tasks: self.tasks.clone(), _consumer: None }
    }

    pub fn task_receipt(&self) -> StreamTaskReceipt {
        self.tasks.clone()
    }

    pub fn request_cancel(&self) -> StreamTaskStatus {
        self.tasks.request_cancel()
    }

    pub fn push(&self, event: AssistantMessageEvent) {
        self.stream.push(event);
    }

    pub fn end(&self, result: Option<AssistantMessage>) {
        self.stream.end(result);
    }

    pub async fn next(&self) -> Option<AssistantMessageEvent> {
        self.stream.next().await
    }

    pub async fn result(&self) -> AssistantMessage {
        self.stream.result().await
    }

    pub fn result_if_ready(&self) -> Option<AssistantMessage> {
        self.stream.result_if_ready()
    }

    pub async fn result_or_end(&self) -> Option<AssistantMessage> {
        self.stream.result_or_end().await
    }

    pub fn is_done(&self) -> bool {
        self.stream.is_done()
    }

    pub fn into_stream(self) -> impl futures::Stream<Item = AssistantMessageEvent> {
        // Keep the consumer lease for the iterator's entire lifetime.
        futures::stream::unfold(self, |stream| async move {
            let next = stream.next().await;
            next.map(|event| (event, stream))
        })
    }

    /// Runs a registered producer/observer body in the invocation's scope.
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.try_spawn(future);
    }

    /// False means the stop fence rejected it; the future is never polled.
    pub fn try_spawn<F>(&self, future: F) -> bool
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.tasks.spawn(future)
    }
}


#[cfg(test)]
mod settlement_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::oneshot;

    struct MarkDropped(Arc<AtomicBool>);
    impl Drop for MarkDropped {
        fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
    }

    #[tokio::test]
    async fn ended_channel_is_not_joined_and_stop_is_idempotent() {
        let stream = AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = MarkDropped(dropped.clone());
        let (started, ready) = oneshot::channel();
        stream.spawn(async move {
            let _guard = guard;
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        ready.await.unwrap();
        stream.end(None);
        assert!(stream.next().await.is_none());
        assert_eq!(receipt.status().pending_tasks, 1);
        assert!(!receipt.status().settled);
        let admitted = receipt.request_cancel();
        assert!(admitted.cancel_requested);
        assert!(!admitted.settled);
        let joined = receipt.settle(Duration::from_secs(1)).await;
        assert!(joined.settled, "{joined:?}");
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(joined.cancelled_tasks, 1);
        assert_eq!(receipt.request_cancel(), joined);
        assert_eq!(receipt.settle(Duration::ZERO).await, joined);
    }

    #[tokio::test]
    async fn already_completed_task_is_joined_not_merely_result_ready() {
        let stream = AssistantMessageEventStream::new_owned();
        let producer = stream.producer_handle();
        stream.spawn(async move { producer.end(Some(AssistantMessage::default())); });
        stream.result().await;
        let receipt = stream.task_receipt();
        let active = receipt.settle(Duration::from_secs(1)).await;
        assert_eq!(active.completed_tasks, 1);
        assert!(!active.cancel_requested);
        assert!(!active.settled);
        let done = receipt.request_cancel();
        assert!(done.settled);
        assert_eq!(done, receipt.request_cancel());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_and_dropped_waiter_keep_the_uninterruptible_join() {
        let stream = AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        let (release, blocked) = std::sync::mpsc::channel();
        let (started, ready) = oneshot::channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = MarkDropped(dropped.clone());
        stream.spawn(async move {
            let _guard = guard;
            started.send(()).unwrap();
            let _ = blocked.recv(); // Simulates an active synchronous callback.
        });
        ready.await.unwrap();
        receipt.request_cancel();
        let timed_out = receipt.settle(Duration::ZERO).await;
        assert!(!timed_out.settled);
        assert_eq!(timed_out.pending_tasks, 1);
        assert!(!dropped.load(Ordering::SeqCst));
        let mut waiter = Box::pin(receipt.settle(Duration::from_secs(10)));
        assert!(futures::poll!(waiter.as_mut()).is_pending());
        drop(waiter);
        assert_eq!(receipt.status().pending_tasks, 1);
        release.send(()).unwrap();
        let done = receipt.settle(Duration::from_secs(1)).await;
        assert!(done.settled, "{done:?}");
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(done.completed_tasks + done.cancelled_tasks, 1);
    }

    #[tokio::test]
    async fn panic_is_acknowledged_but_never_safe_retention() {
        let stream = AssistantMessageEventStream::new_owned();
        stream.spawn(async { panic!("synthetic task panic"); });
        let receipt = stream.task_receipt();
        let done = receipt.settle(Duration::from_secs(1)).await;
        assert_eq!(done.pending_tasks, 0);
        assert_eq!(done.failed_tasks, 1);
        assert!(!receipt.request_cancel().settled);
        assert_eq!(receipt.status().failed_tasks, 1);
    }

    #[tokio::test]
    async fn last_consumer_drop_cancels_without_a_producer_lease_cycle() {
        let stream = AssistantMessageEventStream::new_owned();
        let other_consumer = stream.clone();
        let producer = stream.producer_handle();
        let receipt = stream.task_receipt();
        stream.spawn(async move {
            std::future::pending::<()>().await;
            producer.end(None);
        });
        drop(stream);
        assert!(!receipt.status().cancel_requested);
        drop(other_consumer);
        assert!(receipt.status().cancel_requested);
        assert!(receipt.settle(Duration::from_secs(1)).await.settled);
    }

    #[tokio::test]
    async fn completed_consumer_drop_preserves_normal_observer_completion() {
        let stream = AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        let (release, gate) = oneshot::channel();
        stream.spawn(async move { gate.await.unwrap(); });
        stream.end(Some(AssistantMessage::default()));
        drop(stream);
        assert!(!receipt.status().cancel_requested);
        assert_eq!(receipt.status().pending_tasks, 1);
        release.send(()).unwrap();
        let done = receipt.settle(Duration::from_secs(1)).await;
        assert_eq!(done.completed_tasks, 1);
        assert_eq!(done.cancelled_tasks, 0);
        assert!(receipt.request_cancel().settled);
    }

    #[tokio::test]
    async fn dropping_async_iterator_cancels_its_producer() {
        let stream = AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        stream.spawn(std::future::pending());
        let iterator = stream.into_stream();
        assert!(!receipt.status().cancel_requested);
        drop(iterator);
        assert!(receipt.settle(Duration::from_secs(1)).await.settled);
    }

    #[tokio::test]
    async fn late_spawn_is_rejected_and_dropped_without_polling() {
        let stream = AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        receipt.request_cancel();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = MarkDropped(dropped.clone());
        assert!(!stream.try_spawn(async move {
            let _guard = guard;
            panic!("rejected future must never execute");
        }));
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(receipt.status().pending_tasks, 0);
        assert_eq!(receipt.status().failed_tasks, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn active_task_cannot_register_a_child_after_the_stop_fence() {
        let stream = AssistantMessageEventStream::new_owned();
        let receipt = stream.task_receipt();
        let producer = stream.producer_handle();
        let (release, blocked) = std::sync::mpsc::channel();
        let (started, ready) = oneshot::channel();
        let rejected = Arc::new(AtomicBool::new(false));
        let captured = rejected.clone();
        stream.spawn(async move {
            started.send(()).unwrap();
            let _ = blocked.recv();
            captured.store(!producer.try_spawn(async { panic!("late child ran"); }), Ordering::SeqCst);
        });
        ready.await.unwrap();
        receipt.request_cancel();
        release.send(()).unwrap();
        let done = receipt.settle(Duration::from_secs(1)).await;
        assert!(done.settled, "{done:?}");
        assert!(rejected.load(Ordering::SeqCst));
        assert_eq!(done.completed_tasks + done.cancelled_tasks, 1);
    }

    #[tokio::test]
    async fn deferred_producer_is_owned_before_stream_future_returns() {
        let receipt = StreamTaskReceipt::new_unsupported();
        let inner_receipt = receipt.clone();
        let (created, ready) = oneshot::channel();
        let (release, gate) = oneshot::channel();
        let outer = tokio::spawn(async move {
            inner_receipt.scope(async move {
                let stream = AssistantMessageEventStream::new_owned();
                stream.spawn(std::future::pending());
                created.send(()).unwrap();
                gate.await.unwrap();
                assert!(!stream.try_spawn(async { panic!("late deferred task ran"); }));
                stream
            }).await
        });
        ready.await.unwrap();
        assert_eq!(receipt.status().pending_tasks, 1);
        receipt.request_cancel();
        release.send(()).unwrap();
        let stream = outer.await.unwrap();
        assert!(stream.is_done());
        assert!(receipt.settle(Duration::from_secs(1)).await.settled);
        assert_eq!(stream.task_receipt().status(), receipt.status());
    }

    #[tokio::test]
    async fn cancelled_scope_rejects_a_stream_created_after_resolution_delay() {
        let receipt = StreamTaskReceipt::new_unsupported();
        receipt.request_cancel();
        let stream = receipt.scope(async {
            let stream = AssistantMessageEventStream::new_owned();
            assert!(!stream.try_spawn(async { panic!("late producer ran"); }));
            stream
        }).await;
        assert!(stream.is_done());
        assert!(receipt.status().settled);
    }

    #[tokio::test]
    async fn stopping_one_invocation_never_cancels_another() {
        let first = StreamTaskReceipt::new_unsupported();
        let second = StreamTaskReceipt::new_unsupported();
        let a = first.scope(async {
            let stream = AssistantMessageEventStream::new_owned();
            stream.spawn(std::future::pending());
            stream
        }).await;
        let (release, ready) = oneshot::channel();
        let b = second.scope(async {
            let stream = AssistantMessageEventStream::new_owned();
            let producer = stream.producer_handle();
            stream.spawn(async move {
                ready.await.unwrap();
                producer.end(Some(AssistantMessage::default()));
            });
            stream
        }).await;
        first.request_cancel();
        assert!(first.settle(Duration::from_secs(1)).await.settled);
        assert!(!second.status().cancel_requested);
        assert_eq!(second.status().pending_tasks, 1);
        release.send(()).unwrap();
        b.result().await;
        assert_eq!(second.settle(Duration::from_secs(1)).await.completed_tasks, 1);
        assert!(!second.status().cancel_requested);
        assert!(a.next().await.is_none());
    }

    #[tokio::test]
    async fn unknown_wrapper_cannot_borrow_downstream_provider_coverage() {
        let receipt = StreamTaskReceipt::new_unsupported();
        let (covered, unknown) = receipt.scope(async {
            (AssistantMessageEventStream::new_owned(), AssistantMessageEventStream::new())
        }).await;
        assert!(!receipt.status().supported);
        assert!(!receipt.request_cancel().settled);
        assert!(covered.is_done());
        assert!(unknown.is_done());
        let alone = AssistantMessageEventStream::new();
        assert!(!alone.request_cancel().supported);
    }
}

/// Factory function for AssistantMessageEventStream (for use in extensions)
pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
    AssistantMessageEventStream::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::STOP_REASON_STOP;

    fn message(text: &str) -> AssistantMessage {
        let mut message = AssistantMessage::default();
        message.content = vec![crate::types::ContentBlock::Text(crate::types::TextContent::new(text))];
        message
    }

    #[tokio::test]
    async fn queued_events_are_delivered_in_order() {
        let stream = AssistantMessageEventStream::new();
        let partial = message("a");
        stream.push(AssistantMessageEvent::Start { partial: partial.clone() });
        stream.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "a".to_string(),
            partial: partial.clone(),
        });
        stream.push(AssistantMessageEvent::Done {
            reason: STOP_REASON_STOP.to_string(),
            message: partial,
        });

        assert_eq!(stream.next().await.unwrap().event_type(), "start");
        assert_eq!(stream.next().await.unwrap().event_type(), "text_delta");
        assert_eq!(stream.next().await.unwrap().event_type(), "done");
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn done_event_resolves_the_final_result() {
        let stream = AssistantMessageEventStream::new();
        let final_message = message("done");
        stream.push(AssistantMessageEvent::Done {
            reason: STOP_REASON_STOP.to_string(),
            message: final_message.clone(),
        });
        assert_eq!(stream.result().await, final_message);
        assert!(stream.is_done());
    }

    #[tokio::test]
    async fn events_after_completion_are_dropped() {
        let stream = AssistantMessageEventStream::new();
        stream.push(AssistantMessageEvent::Error {
            reason: "error".to_string(),
            error: message("boom"),
        });
        stream.push(AssistantMessageEvent::Start {
            partial: message("late"),
        });
        assert_eq!(stream.next().await.unwrap().event_type(), "error");
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn waiting_reader_is_woken_by_a_later_push() {
        let stream = AssistantMessageEventStream::new();
        let producer = stream.clone();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            producer.push(AssistantMessageEvent::Start {
                partial: message("later"),
            });
            producer.push(AssistantMessageEvent::Done {
                reason: STOP_REASON_STOP.to_string(),
                message: message("later"),
            });
        });

        let first = stream.next().await.unwrap();
        assert_eq!(first.event_type(), "start");
        let second = stream.next().await.unwrap();
        assert_eq!(second.event_type(), "done");
    }

    #[tokio::test]
    async fn end_does_not_replace_an_already_resolved_result() {
        // event-stream.ts:35-44: `end(result?)` resolves the final-result promise, so a
        // complete event pushed first (event-stream.ts:28-31) keeps its message.
        let stream = AssistantMessageEventStream::new();
        let resolved = message("first");
        stream.push(AssistantMessageEvent::Done {
            reason: STOP_REASON_STOP.to_string(),
            message: resolved.clone(),
        });
        stream.end(Some(message("second")));
        assert_eq!(stream.result().await, resolved);

        // `end(Some(...))` still resolves the result when nothing resolved it first.
        let late = AssistantMessageEventStream::new();
        let fallback = message("fallback");
        late.end(Some(fallback.clone()));
        assert_eq!(late.result().await, fallback);
    }

    #[tokio::test]
    async fn end_without_result_terminates_the_iterator() {
        let stream = AssistantMessageEventStream::new();
        stream.end(None);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn error_event_reason_is_preserved() {
        let stream = AssistantMessageEventStream::new();
        let mut error_message = message("");
        error_message.stop_reason = "aborted".to_string();
        stream.push(AssistantMessageEvent::Error {
            reason: "aborted".to_string(),
            error: error_message.clone(),
        });
        let event = stream.next().await.unwrap();
        match event {
            AssistantMessageEvent::Error { reason, error } => {
                assert_eq!(reason, "aborted");
                assert_eq!(error.stop_reason, "aborted");
            }
            other => panic!("unexpected event {}", other.event_type()),
        }
        assert_eq!(stream.result().await, error_message);
    }

    #[test]
    fn factory_matches_constructor() {
        let stream = create_assistant_message_event_stream();
        assert!(!stream.is_done());
    }
}
