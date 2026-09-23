//! Cached heartbeat status. Rendering never asks the daemon for state.
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

pub(super) fn right_status(
    mode: &InteractiveMode,
    usage: Option<&str>,
    width: usize,
) -> Option<String> {
    let counts = heartbeat_counts(mode);
    let Some(full) = counts.label(false) else {
        return usage.map(str::to_string);
    };
    let compact = counts.label(true).unwrap();
    let usage_width = usage.map(visible_width).unwrap_or(0);
    let room = width.saturating_sub(usage_width + usize::from(usage.is_some()));
    let label = if visible_width(&full) <= room.min(width / 2) {
        &full
    } else if visible_width(&compact) <= room {
        &compact
    } else {
        "HB"
    };
    let heartbeat = theme().fg(
        if counts.user_on + counts.agent_on > 0 {
            "accent"
        } else {
            "dim"
        },
        label,
    );
    if let Some(usage) = usage {
        if visible_width(&heartbeat) + 1 + usage_width <= width {
            return Some(format!("{heartbeat} {usage}"));
        }
        // At unusably small widths keep the established context priority.
        return Some(usage.to_string());
    }
    Some(heartbeat)
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
