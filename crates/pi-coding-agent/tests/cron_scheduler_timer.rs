//! Drive the real timer callback, including a dispatch that suspends at admission.
use pi_coding_agent::core::cron_jobs::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

#[tokio::test]
async fn fired_timer_keeps_dispatch_alive_across_rearm_wake_and_stop() {
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(
        AgentCronJobStore::new(
            Some(
                directory
                    .path()
                    .join("jobs.json")
                    .to_string_lossy()
                    .into_owned(),
            ),
            false,
        )
        .unwrap(),
    );
    let job = store
        .create_rlm_heartbeat(&CreateAgentCronJobInput {
            active_session_id: "fixture".into(),
            session_id: "fixture".into(),
            session_file: directory
                .path()
                .join("session.jsonl")
                .to_string_lossy()
                .into_owned(),
            cwd: directory.path().to_string_lossy().into_owned(),
            prompt: "Fixture heartbeat".into(),
            schedule_text: "every 10s".into(),
            now: Some(0.0),
            ..Default::default()
        })
        .unwrap();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let ended = Arc::new(Notify::new());
    let scheduler = AgentCronScheduler::new(
        store.clone(),
        Arc::new(AgentCronSchedulerHooks {
            run_job: Arc::new({
                let started = started.clone();
                let release = release.clone();
                move |_| {
                    let started = started.clone();
                    let release = release.clone();
                    Box::pin(async move {
                        started.notify_one();
                        release.notified().await;
                        Some(RUN_RESULT_RAN.into())
                    })
                }
            }),
            begin_dispatch: Some(Arc::new({
                let ended = ended.clone();
                move |_| {
                    let ended = ended.clone();
                    Some(Arc::new(move || ended.notify_one()))
                }
            })),
            now: Some(Arc::new(|| 10_000.0)),
            on_error: None,
        }),
    );
    scheduler.start();
    tokio::time::timeout(Duration::from_secs(3), started.notified())
        .await
        .unwrap();
    assert!(store.get_claimed_job(&job.id).is_some());
    scheduler.wake();
    // Let the replacement timer run as well before stopping future ticks.
    tokio::time::sleep(Duration::from_millis(20)).await;
    scheduler.stop();
    release.notify_one();
    let completed = tokio::time::timeout(Duration::from_secs(3), ended.notified()).await;
    assert!(
        completed.is_ok(),
        "rearming the timer cancelled the suspended dispatch"
    );
    assert!(store.get_claimed_job(&job.id).is_none());
    let recorded = store.list().pop().unwrap();
    assert_eq!(recorded.run_count, 1.0);
    assert!(recorded.last_error.is_none());
}
