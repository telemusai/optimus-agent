//! Explicit typed questions share client validation and session cancellation,
//! but do not pass through automatic control or comparison policies.

use std::future::Future;
use std::time::Duration;

use crate::types::{DecisionBundle, DecisionOutcome, SystemOne};

pub const DEADLINE: Duration = Duration::from_secs(30);

struct CancelOnDrop(crate::scheduler::CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.0.cancel();
        }
    }
}

pub async fn decide_scoped(
    client: &dyn SystemOne,
    bundle: DecisionBundle,
    still_allowed: impl Fn() -> bool,
    cancelled: impl Future<Output = ()>,
) -> Result<DecisionOutcome, &'static str> {
    if !client.mode().allows_active() || !still_allowed() {
        return Err("dynamic_disabled");
    }
    let cancellation = crate::scheduler::CancellationToken::new();
    let work =
        crate::scheduler::register_session_work(&bundle.session_id, Some(cancellation.clone()))
            .ok_or("session_stopped")?;
    // Also settle cancellation when the host drops the entire tool future.
    // This guard drops before `work`, but after the nested transport future.
    let _cancel_on_drop = CancelOnDrop(cancellation.clone());
    let retained = work.token();
    let result = {
        let request = client.decide(bundle);
        tokio::pin!(request, cancelled);
        let deadline = tokio::time::sleep(DEADLINE);
        tokio::pin!(deadline);
        let mut changes = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                biased;
                _ = &mut cancelled => break Err("cancelled"),
                _ = retained.cancelled() => break Err("session_stopped"),
                _ = &mut deadline => break Err("timeout"),
                _ = changes.tick() => {
                    if !still_allowed() { break Err("settings_changed"); }
                }
                outcome = &mut request => {
                    break if still_allowed() { Ok(outcome) } else { Err("settings_changed") };
                }
            }
        }
    };
    // The transport future has dropped before session settlement is acknowledged.
    if result.is_err() {
        cancellation.cancel();
    }
    work.finish();
    result
}
