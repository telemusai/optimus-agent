//! Cached goal and heartbeat status. Rendering never asks the daemon for state.
use super::*;
use pi_tui::utils::visible_width;

#[derive(Default, Debug, PartialEq, Eq)]
struct HeartbeatCounts {
    user_on: usize,
    user_paused: usize,
    agent_on: usize,
    agent_paused: usize,
}

impl HeartbeatCounts {
    fn add(&mut self, job: &local::AgentCronJob) {
        let count = match (job.source.as_deref(), job.status.as_str()) {
            (Some("heartbeat"), "active") => &mut self.user_on,
            (Some("heartbeat"), "paused") => &mut self.user_paused,
            (Some("rlm_heartbeat"), "active") => &mut self.agent_on,
            (Some("rlm_heartbeat"), "paused") => &mut self.agent_paused,
            _ => return,
        };
        *count += 1;
    }

    fn label(&self, compact: bool) -> Option<String> {
        let on = self.user_on + self.agent_on;
        let paused = self.user_paused + self.agent_paused;
        if on + paused == 0 {
            return None;
        }
        if compact {
            return Some(if on == 0 {
                "HB paused".into()
            } else if paused == 0 {
                "HB on".into()
            } else {
                "HB on/paused".into()
            });
        }
        let mut parts = Vec::new();
        for (owner, active, paused) in [
            ("user", self.user_on, self.user_paused),
            ("agent", self.agent_on, self.agent_paused),
        ] {
            if active + paused == 0 {
                continue;
            }
            let state = match (active, paused) {
                (0, _) => "paused".to_string(),
                (_, 0) => "on".to_string(),
                _ => format!("{active} on/{paused} paused"),
            };
            parts.push(format!("{owner} {state}"));
        }
        Some(format!("HB {}", parts.join(" · ")))
    }
}

fn heartbeat_counts(mode: &InteractiveMode) -> HeartbeatCounts {
    let jobs = mode.get_scoped_heartbeats();
    let mut counts = HeartbeatCounts::default();
    for heartbeat in &jobs {
        counts.add(&heartbeat.job);
    }
    // The initial state includes the user's heartbeat even on an older daemon
    // without a catalog. Catalog entries win when the same job is present.
    if let Some(job) = mode
        .connection_state
        .as_ref()
        .and_then(|state| state.heartbeat.as_ref())
    {
        if !mode.heartbeat_catalog_authoritative
            && !jobs.iter().any(|heartbeat| heartbeat.job.id == job.id)
        {
            counts.add(job);
        }
    }
    counts
}

#[derive(Default)]
struct GoalClock {
    sample: Option<(String, crate::core::goals::GoalState, Instant)>,
}

impl GoalClock {
    fn elapsed(&mut self, session: &str, goal: &crate::core::goals::GoalState, now: Instant) -> Option<f64> {
        use crate::core::goals::GoalStatus;
        if goal.status != GoalStatus::Active {
            self.sample = None;
            return None;
        }
        let unchanged = self.sample.as_ref().is_some_and(|(previous_session, previous, _)| {
            previous_session == session && previous.goal_id == goal.goal_id
                && previous.created_at == goal.created_at && previous.updated_at == goal.updated_at
                && previous.time_used_seconds == goal.time_used_seconds
        });
        if !unchanged {
            self.sample = Some((session.into(), goal.clone(), now));
        }
        let (_, _, received) = self.sample.as_ref().unwrap();
        let recorded = if goal.time_used_seconds.is_finite() { goal.time_used_seconds.max(0.0) } else { 0.0 };
        Some(recorded + now.saturating_duration_since(*received).as_secs_f64())
    }
}

thread_local! {
    // Only the visible session owns a clock. Switching sessions replaces it;
    // authoritative goal counters always supersede the local interpolation.
    static GOAL_CLOCK: RefCell<GoalClock> = RefCell::new(GoalClock::default());
}

pub(super) fn right_status(
    mode: &InteractiveMode,
    usage: Option<&str>,
    width: usize,
) -> Option<String> {
    let goal = mode.get_goal_state();
    let session = mode.connection_state.as_ref().map(|state| state.session_id.as_str()).unwrap_or("");
    let elapsed = GOAL_CLOCK.with(|clock| clock.borrow_mut().elapsed(session, &goal, Instant::now()));
    let goal_full = elapsed.map(|seconds| theme().fg("accent", &format!("Pursuing Goal {}", mode.format_goal_elapsed(seconds))));
    let goal_short = elapsed.map(|seconds| theme().fg("accent", &format!("Goal {}", mode.format_goal_elapsed(seconds))));
    let counts = heartbeat_counts(mode);
    let heartbeat_color =
        if counts.user_on + counts.agent_on > 0 {
            "accent"
        } else {
            "dim"
        };
    let heartbeat_full = counts.label(false).map(|label| theme().fg(heartbeat_color, &label));
    let heartbeat_short = counts.label(true).map(|label| theme().fg(heartbeat_color, &label));
    let heartbeat_minimal = heartbeat_full.as_ref().map(|_| theme().fg(heartbeat_color, "HB"));
    for (heartbeat, goal) in [
        (&heartbeat_full, &goal_full),
        (&heartbeat_short, &goal_full),
        (&heartbeat_minimal, &goal_full),
        (&heartbeat_short, &goal_short),
        (&heartbeat_minimal, &goal_short),
        (&None, &goal_short),
        (&heartbeat_minimal, &None),
    ] {
        let parts: Vec<_> = [heartbeat.as_deref(), goal.as_deref(), usage].into_iter().flatten().collect();
        if !parts.is_empty() {
            let label = parts.join(" ");
            if visible_width(&label) <= width { return Some(label); }
        }
    }
    // The context counter retains priority when the terminal is too narrow.
    usage.map(str::to_string)
}

pub(super) fn apply_catalog_result(
    mode: &mut InteractiveMode,
    catalog: &[wire::AgentConnectionHeartbeat],
    authoritative: bool,
) {
    mode.heartbeat_catalog_authoritative |= authoritative || !catalog.is_empty();
    native_heartbeats::apply_catalog(mode, catalog);
}

/// One optional lookup after attach/resync or a change event, not a timer.
/// Multiple change events share one in-flight request and one trailing refresh.
pub(super) struct HeartbeatRefresh {
    task: Option<
        tokio::task::JoinHandle<Result<(Vec<wire::AgentConnectionHeartbeat>, bool), String>>,
    >,
    session_id: String,
    dirty: bool,
    authoritative: bool,
}

impl HeartbeatRefresh {
    pub(super) fn new() -> Self {
        Self {
            task: None,
            session_id: String::new(),
            dirty: false,
            authoritative: false,
        }
    }

    pub(super) fn request(
        &mut self,
        connection: Arc<dyn wire::AgentConnection>,
        session_id: &str,
        authoritative: bool,
    ) {
        if self.session_id != session_id {
            self.cancel();
            self.session_id = session_id.into();
            self.authoritative = false;
        }
        self.authoritative |= authoritative;
        if self.task.is_some() {
            self.dirty = true;
            return;
        }
        self.dirty = false;
        self.task = Some(tokio::spawn(async move {
            // list_heartbeats negotiates heartbeat_catalog and degrades to [] on
            // old daemons. Failure never blocks attachment or agent execution.
            let catalog =
                tokio::time::timeout(Duration::from_secs(10), connection.list_heartbeats())
                    .await
                    .unwrap_or_else(|_| Err("Heartbeat status unavailable".into()))?;
            Ok((
                catalog,
                connection.heartbeat_catalog_supported() == Some(true),
            ))
        }));
    }

    pub(super) fn cancel(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        self.dirty = false;
    }

    pub(super) async fn poll(
        &mut self,
        connection: Arc<dyn wire::AgentConnection>,
        session_id: &str,
    ) -> Option<(Vec<wire::AgentConnectionHeartbeat>, bool)> {
        if self.session_id != session_id {
            self.cancel();
            return None;
        }
        if !self.task.as_ref().is_some_and(|task| task.is_finished()) {
            return None;
        }
        let result = self.task.take().unwrap().await.ok().and_then(Result::ok);
        if self.dirty {
            self.request(connection, session_id, self.authoritative);
            return None;
        }
        result.map(|(catalog, supported)| (catalog, self.authoritative || supported))
    }
}

impl Drop for HeartbeatRefresh {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
#[path = "native_host_status_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "native_host_goal_status_tests.rs"]
mod goal_tests;
