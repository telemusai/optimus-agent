use super::*;
use futures::channel::oneshot;
use futures::future::{self, BoxFuture, FutureExt};
use futures::stream::{self, StreamExt};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::Duration;

type TestResult = Result<String, String>;
type TestCall = BoxFuture<'static, TestResult>;

struct DropFlag(Arc<AtomicBool>);
impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn tracked_call(
    result: Option<TestResult>,
    delay: Duration,
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicBool>,
) -> TestCall {
    async move {
        let _drop = DropFlag(drops);
        polls.fetch_add(1, Ordering::SeqCst);
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        match result {
            Some(result) => result,
            None => future::pending::<TestResult>().await,
        }
    }
    .boxed()
}

fn calls_with_flags(
    results: Vec<Option<TestResult>>,
    polls: &[Arc<AtomicUsize>],
    drops: &[Arc<AtomicBool>],
) -> Vec<(usize, TestCall)> {
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            (
                index,
                tracked_call(
                    result,
                    Duration::ZERO,
                    polls[index].clone(),
                    drops[index].clone(),
                ),
            )
        })
        .collect()
}

#[tokio::test]
async fn later_index_failure_cancels_an_earlier_unfinished_call() {
    let polls: Vec<_> = (0..6).map(|_| Arc::new(AtomicUsize::new(0))).collect();
    let drops: Vec<_> = (0..6).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let calls = calls_with_flags(
        vec![
            None,
            Some(Err("PRIMARY_PROVIDER_ERROR".into())),
            Some(Ok("two".into())),
            None,
            None,
            None,
        ],
        &polls,
        &drops,
    );
    let collected = tokio::time::timeout(
        Duration::from_secs(2),
        collect_indexed_textmas_calls(
            stream::iter(calls)
                .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
        ),
    )
    .await;
    let error = collected
        .expect("a later-index failure must not wait for an earlier pending call")
        .expect_err("the provider error remains primary");
    assert_eq!(error, "PRIMARY_PROVIDER_ERROR");
    assert_eq!(polls[0].load(Ordering::SeqCst), 1);
    assert!(
        drops[0].load(Ordering::SeqCst),
        "admitted mapper 0 was dropped"
    );
    // A same-poll success may admit one replacement, never the whole remainder.
    for index in 4..6 {
        assert_eq!(
            polls[index].load(Ordering::SeqCst),
            0,
            "unadmitted call {index}"
        );
        assert!(!drops[index].load(Ordering::SeqCst));
    }
}

#[tokio::test]
async fn first_observed_error_is_primary_among_multiple_failures() {
    let polls: Vec<_> = (0..4).map(|_| Arc::new(AtomicUsize::new(0))).collect();
    let drops: Vec<_> = (0..4).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let calls = vec![
        (
            0,
            tracked_call(None, Duration::ZERO, polls[0].clone(), drops[0].clone()),
        ),
        (
            1,
            tracked_call(
                Some(Err("LATER_FAILURE".into())),
                Duration::from_millis(50),
                polls[1].clone(),
                drops[1].clone(),
            ),
        ),
        (
            2,
            tracked_call(
                Some(Err("FIRST_OBSERVED_FAILURE".into())),
                Duration::ZERO,
                polls[2].clone(),
                drops[2].clone(),
            ),
        ),
        (
            3,
            tracked_call(None, Duration::ZERO, polls[3].clone(), drops[3].clone()),
        ),
    ];
    let error = tokio::time::timeout(
        Duration::from_secs(2),
        collect_indexed_textmas_calls(
            stream::iter(calls)
                .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
        ),
    )
    .await
    .expect("first observed failure must be promptly observable")
    .expect_err("one provider failure remains primary");
    assert_eq!(error, "FIRST_OBSERVED_FAILURE");
    assert!(drops[0].load(Ordering::SeqCst));
}

#[tokio::test]
async fn collector_drop_after_timeout_drops_only_admitted_calls() {
    let polls: Vec<_> = (0..6).map(|_| Arc::new(AtomicUsize::new(0))).collect();
    let drops: Vec<_> = (0..6).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let calls = calls_with_flags(vec![None; 6], &polls, &drops);
    let collected = tokio::time::timeout(
        Duration::from_millis(30),
        collect_indexed_textmas_calls(
            stream::iter(calls)
                .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
        ),
    )
    .await;
    assert!(collected.is_err(), "the watchdog bounds pending roles");
    for index in 0..3 {
        assert_eq!(polls[index].load(Ordering::SeqCst), 1);
        assert!(drops[index].load(Ordering::SeqCst));
    }
    for index in 3..6 {
        assert_eq!(polls[index].load(Ordering::SeqCst), 0);
        assert!(!drops[index].load(Ordering::SeqCst));
    }
}

struct ActiveGuard(Arc<AtomicUsize>);
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn active_call(
    index: usize,
    delay: Duration,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
) -> TestCall {
    async move {
        let _active = ActiveGuard(active.clone());
        peak.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Ok(format!("SOURCE_{index}"))
    }
    .boxed()
}

#[tokio::test]
async fn out_of_order_success_restores_source_order_at_concurrency_three() {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let delays = [40, 20, 10, 5, 5, 5, 5];
    let calls = delays.into_iter().enumerate().map(|(index, delay)| {
        (
            index,
            active_call(
                index,
                Duration::from_millis(delay),
                active.clone(),
                peak.clone(),
            ),
        )
    });
    let values = tokio::time::timeout(
        Duration::from_secs(2),
        collect_indexed_textmas_calls(
            stream::iter(calls)
                .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
        ),
    )
    .await
    .expect("success is bounded by the test watchdog")
    .expect("all calls succeed");
    assert_eq!(
        values,
        (0..7)
            .map(|index| format!("SOURCE_{index}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(peak.load(Ordering::SeqCst), 3);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn reducer_level_success_uses_the_same_indexed_collection() {
    let (release_first, wait_first) = oneshot::channel::<()>();
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let first_active = active.clone();
    let first_peak = peak.clone();
    let first = async move {
        let _active = ActiveGuard(first_active.clone());
        first_peak.fetch_max(
            first_active.fetch_add(1, Ordering::SeqCst) + 1,
            Ordering::SeqCst,
        );
        wait_first.await.expect("test gate remains open");
        Ok("REDUCED_SOURCE_0".to_string())
    }
    .boxed();
    let calls = vec![
        (0, first),
        (
            4,
            active_call(4, Duration::from_millis(10), active.clone(), peak.clone()),
        ),
        (
            8,
            active_call(8, Duration::from_millis(10), active.clone(), peak.clone()),
        ),
    ];
    let collector = tokio::spawn(collect_indexed_textmas_calls(
        stream::iter(calls)
            .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
    ));
    tokio::task::yield_now().await;
    assert!(
        !collector.is_finished(),
        "one level cannot finish without every brief"
    );
    release_first.send(()).expect("collector is alive");
    let values = tokio::time::timeout(Duration::from_secs(2), collector)
        .await
        .expect("reduction completion is bounded")
        .expect("collector task joined")
        .expect("all reductions succeed");
    assert_eq!(values, ["REDUCED_SOURCE_0", "SOURCE_4", "SOURCE_8"]);
    // Construction of the next reduction level remains in the caller after collection.
    assert_eq!(peak.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn simultaneous_ready_failures_use_the_first_observed_error() {
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pending_drops = Arc::new(AtomicBool::new(false));
    let pending_polls = Arc::new(AtomicUsize::new(0));
    let unadmitted_polls = Arc::new(AtomicUsize::new(0));
    let unadmitted_drops = Arc::new(AtomicBool::new(false));
    let error_call = |index: usize, observed: Arc<std::sync::Mutex<Vec<usize>>>| {
        async move {
            observed.lock().unwrap().push(index);
            Err::<String, String>(format!("ERROR_{index}"))
        }
        .boxed()
    };
    let calls = vec![
        (
            0,
            tracked_call(None, Duration::ZERO, pending_polls, pending_drops.clone()),
        ),
        (1, error_call(1, observed.clone())),
        (2, error_call(2, observed.clone())),
        (
            3,
            tracked_call(
                None,
                Duration::ZERO,
                unadmitted_polls.clone(),
                unadmitted_drops,
            ),
        ),
    ];
    let error = tokio::time::timeout(
        Duration::from_secs(2),
        collect_indexed_textmas_calls(
            stream::iter(calls)
                .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
        ),
    )
    .await
    .expect("ready failures cannot wait for mapper 0")
    .expect_err("one error is primary");
    let observed = observed.lock().unwrap().clone();
    assert_eq!(
        observed.len(),
        1,
        "collection stops at its first observed error"
    );
    assert_eq!(error, format!("ERROR_{}", observed[0]));
    assert!(pending_drops.load(Ordering::SeqCst));
    assert_eq!(unadmitted_polls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn outer_cancellation_drops_only_admitted_calls() {
    let polls: Vec<_> = (0..6).map(|_| Arc::new(AtomicUsize::new(0))).collect();
    let drops: Vec<_> = (0..6).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let calls = calls_with_flags(vec![None; 6], &polls, &drops);
    let cancellation = CancellationToken::new();
    let trigger = cancellation.clone();
    let trigger_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(5)).await;
        trigger.cancel();
    });
    {
        let collector = collect_indexed_textmas_calls(
            stream::iter(calls)
                .map(|(index, call)| async move { call.await.map(|text| (index, text)) }),
        );
        tokio::pin!(collector);
        let watchdog = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(watchdog);
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => {}
            result = &mut collector => panic!("pending roles unexpectedly settled: {result:?}"),
            _ = &mut watchdog => panic!("cancellation failed to observe the pending collector"),
        }
    }
    trigger_task.await.expect("cancellation trigger joined");
    for index in 0..3 {
        assert_eq!(polls[index].load(Ordering::SeqCst), 1);
        assert!(drops[index].load(Ordering::SeqCst));
    }
    for index in 3..6 {
        assert_eq!(polls[index].load(Ordering::SeqCst), 0);
        assert!(!drops[index].load(Ordering::SeqCst));
    }
}
