//! Bounded presentation projection for provider-neutral OCC3.1 activity.
//!
//! This module owns no sessions, roster rows, usage or provider calls. Call
//! dependencies are rendered only when present in the activity data plane.

use crate::core::occ::textmas::{TextMasMode, TextMasRole};
use crate::core::occ::textmas_activity::{
    TextMasActivityCall, TextMasActivitySnapshot, TextMasActivityStatus,
};
use pi_tui::utils::{truncate_to_width, visible_width};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

pub(super) const MAX_MAPPER_COHORTS: usize = 3;
pub(super) const MAX_REDUCER_COHORTS: usize = 3;
const MAX_RETIRED_ACTIONS: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub total: usize,
    pub queued: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub timed_out: usize,
    pub skipped: usize,
    pub stale: usize,
}

impl Counts {
    fn observe(&mut self, status: TextMasActivityStatus) {
        self.total += 1;
        *match status {
            TextMasActivityStatus::Queued => &mut self.queued,
            TextMasActivityStatus::Running => &mut self.running,
            TextMasActivityStatus::Completed => &mut self.completed,
            TextMasActivityStatus::Failed => &mut self.failed,
            TextMasActivityStatus::Cancelled => &mut self.cancelled,
            TextMasActivityStatus::TimedOut => &mut self.timed_out,
            TextMasActivityStatus::Skipped => &mut self.skipped,
            TextMasActivityStatus::Stale => &mut self.stale,
        } += 1;
    }

    pub(super) fn glyph(self, role: TextMasRole) -> char {
        match role {
            TextMasRole::Planner => 'P',
            TextMasRole::Critic => 'C',
            TextMasRole::Refiner => 'R',
            TextMasRole::Mapper => 'M',
            TextMasRole::Reducer => 'Σ',
        }
    }

    fn state_glyph(self) -> char {
        if self.failed > 0 {
            '!'
        } else if self.running > 0 {
            '…'
        } else if self.queued > 0 {
            'q'
        } else if self.timed_out > 0 {
            't'
        } else if self.cancelled > 0 {
            '×'
        } else if self.stale > 0 {
            'o'
        } else if self.skipped > 0 {
            's'
        } else if self.total > 0 && self.completed == self.total {
            '✓'
        } else {
            '?'
        }
    }

    fn detail(self, label: &str) -> String {
        let mut detail = format!(
            "{label}{} q{}r{} ✓{} !{} ×{}",
            self.total, self.queued, self.running, self.completed, self.failed, self.cancelled
        );
        for (value, glyph) in [
            (self.timed_out, 't'),
            (self.skipped, 's'),
            (self.stale, 'o'),
        ] {
            if value > 0 {
                detail.push_str(&format!(" {glyph}{value}"));
            }
        }
        detail
    }
}

fn recorded_levels(levels: &[u32], incomplete: bool) -> String {
    let mut text = match levels {
        [] => String::new(),
        [level] => format!("L{level}"),
        _ if levels.windows(2).all(|pair| pair[1] == pair[0] + 1) => {
            format!("L{}-{}", levels[0], levels[levels.len() - 1])
        }
        _ => format!(
            "L{}",
            levels
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    };
    if incomplete {
        text = if text.is_empty() {
            "L?".to_string()
        } else {
            format!("{text}+?")
        };
    }
    text
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ActivityNode {
    pub id: String,
    pub glyph: char,
    pub counts: Counts,
    /// Projected node IDs corresponding to observed dependency IDs only.
    pub dependencies: Vec<String>,
    /// An absent range/level is unknown, never inferred from a position.
    pub index_range: Option<(u32, u32)>,
    pub levels: Vec<u32>,
    level_incomplete: bool,
    role: Option<TextMasRole>,
}

impl ActivityNode {
    pub(super) fn intensity(self: &Self, working: bool, pulse: bool) -> u8 {
        if working {
            if pulse {
                3
            } else {
                2
            }
        } else if self.counts.queued > 0 || self.counts.skipped > 0 || self.counts.stale > 0 {
            1
        } else {
            2
        }
    }

    pub(super) fn short_label(&self) -> String {
        match (self.role, self.index_range) {
            (Some(TextMasRole::Mapper), Some((first, last))) => format!("M{first}-{last}"),
            (Some(TextMasRole::Reducer), _) if !self.levels.is_empty() || self.level_incomplete => {
                format!("Σ{}", recorded_levels(&self.levels, self.level_incomplete))
            }
            (Some(role), _) => format!("{}{}", self.counts.glyph(role), self.counts.state_glyph()),
            (None, _) => "T? unknown".into(),
        }
    }

    fn detail(&self) -> String {
        if self.role == Some(TextMasRole::Reducer)
            && (!self.levels.is_empty() || self.level_incomplete)
        {
            let label = format!(
                "Σ[{}]:",
                recorded_levels(&self.levels, self.level_incomplete)
            );
            return self.counts.detail(&label);
        }
        let label = match self.role {
            Some(TextMasRole::Planner) => "P",
            Some(TextMasRole::Critic) => "C",
            Some(TextMasRole::Refiner) => "R",
            Some(TextMasRole::Mapper) => "M",
            Some(TextMasRole::Reducer) => "Σ",
            None => "T?",
        };
        self.counts.detail(label)
    }
}

#[derive(Default)]
pub(super) struct Projection {
    snapshot: Option<TextMasActivitySnapshot>,
    nodes: Vec<ActivityNode>,
    first_observed: Option<Instant>,
    last_changed: Option<Instant>,
    owner_active: bool,
    paused_since: Option<Instant>,
    retired_actions: VecDeque<String>,
}

impl Projection {
    pub(super) fn is_active(&self) -> bool {
        self.owner_active
            && self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.active)
    }

    pub(super) fn owner_active(&self) -> bool {
        self.owner_active
    }

    pub(super) fn is_disconnected(&self) -> bool {
        !self.owner_active
            && self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.active)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.snapshot.is_none()
    }

    pub(super) fn nodes(&self) -> &[ActivityNode] {
        &self.nodes
    }

    /// Consume authoritative snapshots only. Refresh-only event names never
    /// enter this reducer, so a late old-action event cannot paint old calls.
    pub(super) fn observe(
        &mut self,
        session: &str,
        snapshot: Option<&TextMasActivitySnapshot>,
        now: Instant,
    ) -> bool {
        self.observe_with_owner(
            session,
            snapshot,
            snapshot.is_some_and(|snapshot| snapshot.active),
            now,
        )
    }

    /// The owner flag is host-local connection knowledge, not a call status.
    /// Clearing it pauses rather than relabelling orphaned Started calls.
    pub(super) fn observe_with_owner(
        &mut self,
        session: &str,
        snapshot: Option<&TextMasActivitySnapshot>,
        owner_active: bool,
        now: Instant,
    ) -> bool {
        if snapshot.is_some_and(|snapshot| snapshot.session_id != session) {
            let changed = !self.is_empty() || self.owner_active || self.paused_since.is_some();
            *self = Self::default();
            return changed;
        }
        // The connection owner belongs to the current validated cache, not to
        // a late, malformed, or retired activity payload. Apply its transition
        // before payload acceptance so a late old action cannot hide a disconnect.
        let owner_changed = self.owner_active != owner_active;
        if owner_changed {
            self.owner_active = owner_active;
            self.paused_since = if !owner_active
                && self
                    .snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.active)
            {
                self.paused_since.or(Some(now))
            } else {
                None
            };
        }
        if snapshot.is_some_and(|snapshot| snapshot.calls.is_empty() || !snapshot.is_valid()) {
            // Malformed optional data cannot erase a valid cached run. An
            // authoritative None below is a separate, intentional clear.
            return owner_changed;
        }
        if self.snapshot.as_ref() == snapshot
            && snapshot.is_none_or(|snapshot| snapshot.session_id == session)
        {
            return owner_changed;
        }
        let snapshot = snapshot.filter(|snapshot| {
            !snapshot.calls.is_empty()
                && snapshot.is_valid()
                && snapshot.session_id == session
                && snapshot.mode != Some(TextMasMode::Off)
        });
        if let Some(snapshot) = snapshot {
            if snapshot.action_id
                != self
                    .snapshot
                    .as_ref()
                    .map(|current| current.action_id.as_str())
                    .unwrap_or("")
            {
                // Retire replaced identities within a bounded session-local window.
                // Never compare opaque action IDs lexically or invent a generation.
                if self
                    .retired_actions
                    .iter()
                    .any(|id| id == &snapshot.action_id)
                {
                    return owner_changed;
                }
            }
        }
        let replaced = self
            .snapshot
            .as_ref()
            .is_none_or(|old| Some(&old.action_id) != snapshot.map(|next| &next.action_id));
        if let Some(previous) = self
            .snapshot
            .take()
            .filter(|previous| snapshot.is_some_and(|next| next.action_id != previous.action_id))
        {
            self.retired_actions.retain(|id| id != &previous.action_id);
            self.retired_actions.push_back(previous.action_id);
            if self.retired_actions.len() > MAX_RETIRED_ACTIONS {
                self.retired_actions.pop_front();
            }
        }
        if replaced || snapshot.is_none() {
            self.first_observed = snapshot.map(|_| now);
        }
        self.last_changed = snapshot.map(|_| now);
        self.owner_active = owner_active;
        self.paused_since =
            (!owner_active && snapshot.is_some_and(|snapshot| snapshot.active)).then_some(now);
        self.nodes = snapshot.map(project).unwrap_or_default();
        self.snapshot = snapshot.cloned();
        true
    }

    pub(super) fn elapsed(&self, now: Instant) -> Option<Duration> {
        self.first_observed.map(|at| {
            self.paused_since
                .unwrap_or(now)
                .saturating_duration_since(at)
        })
    }

    #[cfg(test)]
    pub(super) fn last_changed(&self) -> Option<Instant> {
        self.last_changed
    }

    /// Text stays inside existing graph rows. Detailed statuses wrap before
    /// optional topology captions, with at least two rows for the graph itself.
    pub(super) fn lines(&self, width: usize, now: Instant, maximum: usize) -> Vec<String> {
        let Some(snapshot) = &self.snapshot else {
            return Vec::new();
        };
        if width == 0 || maximum == 0 {
            return Vec::new();
        }
        let mode = match snapshot.mode {
            Some(TextMasMode::Sequential) => "seq",
            Some(TextMasMode::Hierarchical) => "hier",
            Some(TextMasMode::Off) | None => "unknown",
        };
        let counts = self
            .nodes
            .iter()
            .fold(Counts::default(), |mut total, node| {
                total.total += node.counts.total;
                total.completed += node.counts.completed;
                total
            });
        let elapsed = self.elapsed(now).unwrap_or_default();
        let elapsed = if elapsed < Duration::from_secs(60) {
            format!("{:.1}s", elapsed.as_secs_f32())
        } else {
            format!("{}:{:02}", elapsed.as_secs() / 60, elapsed.as_secs() % 60)
        };
        let mut first = if snapshot.mode.is_none() {
            "T? unknown".to_string()
        } else {
            self.nodes
                .iter()
                .filter(|node| {
                    !matches!(node.role, Some(TextMasRole::Mapper | TextMasRole::Reducer))
                })
                .map(ActivityNode::short_label)
                .collect::<Vec<_>>()
                .join(" ")
        };
        let disconnected = self.is_disconnected();
        if disconnected {
            // The current owner is gone. Cached running is not proof of an
            // active provider call, but is not relabelled as cancelled either.
            first = first.replace('…', "?");
        }
        // Completion is a count, never an aggregate success claim. Failure and
        // active states remain explicit on their individual roles/cohorts.
        let tokens = if self.is_disconnected() {
            vec!["disconnected".into()]
        } else {
            vec![
                format!("{}/{}", counts.completed, counts.total),
                format!("age{elapsed}"),
                mode.into(),
            ]
        };
        for token in tokens {
            if visible_width(&format!("{first} {token}")) <= width {
                first.push(' ');
                first.push_str(&token);
            }
        }
        for node in &self.nodes {
            if matches!(node.role, Some(TextMasRole::Mapper | TextMasRole::Reducer)) {
                let label = node.short_label();
                if visible_width(&format!("{first} {label}")) <= width {
                    first.push(' ');
                    first.push_str(&label);
                }
            }
        }
        let mut lines = vec![truncate_to_width(&first, width as f64, "…", false)];
        if disconnected && maximum > 1 {
            let mut progress = format!("{}/{}", counts.completed, counts.total);
            for token in [format!("age{elapsed}"), mode.into()] {
                if visible_width(&format!("{progress} {token}")) <= width {
                    progress.push(' ');
                    progress.push_str(&token);
                }
            }
            lines.push(truncate_to_width(&progress, width as f64, "…", false));
        }
        let auxiliary = self
            .nodes
            .iter()
            .filter(|node| matches!(node.role, Some(TextMasRole::Mapper | TextMasRole::Reducer)))
            .count();
        if auxiliary > maximum.saturating_sub(lines.len()) {
            // Keep real agents and both role totals ahead of optional per-cohort
            // detail. Recorded levels survive the bounded role aggregation.
            for role in [TextMasRole::Mapper, TextMasRole::Reducer] {
                if lines.len() >= maximum {
                    break;
                }
                let calls = snapshot.calls.iter().filter(|call| call.role == role);
                let counts = calls.fold(Counts::default(), |mut counts, call| {
                    counts.observe(call.status);
                    counts
                });
                if counts.total > 0 {
                    let mut levels = self
                        .nodes
                        .iter()
                        .filter(|node| node.role == Some(role))
                        .flat_map(|node| node.levels.iter().copied())
                        .collect::<Vec<_>>();
                    levels.sort_unstable();
                    levels.dedup();
                    let incomplete = self
                        .nodes
                        .iter()
                        .any(|node| node.role == Some(role) && node.level_incomplete);
                    let label = if role == TextMasRole::Mapper {
                        "M".to_string()
                    } else if !levels.is_empty() || incomplete {
                        format!("Σ[{}]:", recorded_levels(&levels, incomplete))
                    } else {
                        "Σ".to_string()
                    };
                    lines.push(truncate_to_width(
                        &counts.detail(&label),
                        width as f64,
                        "…",
                        false,
                    ));
                }
            }
        } else {
            for node in &self.nodes {
                if !matches!(node.role, Some(TextMasRole::Mapper | TextMasRole::Reducer))
                    && snapshot.mode.is_some()
                {
                    continue;
                }
                if lines.len() >= maximum {
                    break;
                }
                // Show an exact recorded range only when it fits. Unknown levels
                // and ranges stay absent rather than filling gaps by speculation.
                let full = match (node.role, node.index_range) {
                    (Some(TextMasRole::Mapper), Some((from, _))) => format!(
                        "M{from}-{} {}",
                        node.index_range.unwrap().1,
                        node.detail().split_once(' ').map(|x| x.1).unwrap_or("")
                    ),
                    _ => node.detail(),
                };
                let text = if visible_width(&full) <= width {
                    full
                } else {
                    node.detail()
                };
                lines.push(truncate_to_width(&text, width as f64, "…", false));
            }
        }
        lines
            .into_iter()
            .map(|line| {
                format!(
                    "\x1b[22;38;5;{}m{}\x1b[22;39m",
                    super::context_health_indicator::CONTEXT_HEALTH_GREEN,
                    line
                )
            })
            .collect()
    }
}

fn project<'a>(snapshot: &'a TextMasActivitySnapshot) -> Vec<ActivityNode> {
    if snapshot.mode.is_none() {
        let counts = snapshot
            .calls
            .iter()
            .fold(Counts::default(), |mut counts, call| {
                counts.observe(call.status);
                counts
            });
        return vec![ActivityNode {
            id: format!("textmas:{}:aggregate", snapshot.action_id),
            glyph: '?',
            counts,
            dependencies: Vec::new(),
            index_range: None,
            levels: Vec::new(),
            level_incomplete: false,
            role: None,
        }];
    }
    let mut nodes = Vec::new();
    let mut locations: BTreeMap<&'a str, String> = BTreeMap::new();
    let mut add =
        |role: Option<TextMasRole>, calls: Vec<&'a TextMasActivityCall>, cohort: String| {
            let id = format!("textmas:{cohort}");
            for call in &calls {
                locations.insert(call.call_id.as_str(), id.clone());
            }
            let counts = calls.iter().fold(Counts::default(), |mut counts, call| {
                counts.observe(call.status);
                counts
            });
            let indices = calls
                .iter()
                .filter_map(|call| call.call_index)
                .collect::<Vec<_>>();
            let index_range = (indices.len() == calls.len() && !indices.is_empty()).then(|| {
                (
                    *indices.iter().min().unwrap(),
                    *indices.iter().max().unwrap(),
                )
            });
            let mut levels = calls
                .iter()
                .filter_map(|call| call.level)
                .collect::<Vec<_>>();
            levels.sort_unstable();
            levels.dedup();
            nodes.push(ActivityNode {
                id,
                glyph: role.map(|role| counts.glyph(role)).unwrap_or('?'),
                counts,
                dependencies: Vec::new(),
                index_range,
                levels,
                level_incomplete: role == Some(TextMasRole::Reducer)
                    && calls.iter().any(|call| call.level.is_none()),
                role,
            });
        };
    for role in TextMasRole::ALL {
        if let Some(call) = snapshot.calls.iter().find(|call| call.role == role) {
            add(Some(role), vec![call], call.call_id.clone());
        }
    }
    for (role, limit) in [
        (TextMasRole::Mapper, MAX_MAPPER_COHORTS),
        (TextMasRole::Reducer, MAX_REDUCER_COHORTS),
    ] {
        let mut calls = snapshot
            .calls
            .iter()
            .filter(|call| call.role == role)
            .collect::<Vec<_>>();
        calls.sort_by_key(|call| (call.level, call.call_index, call.call_id.as_str()));
        let size = calls.len().div_ceil(limit).max(1);
        for (index, cohort) in calls.chunks(size).enumerate() {
            let first = cohort
                .first()
                .map(|call| call.call_index.unwrap_or(index as u32))
                .unwrap_or(index as u32);
            add(
                Some(role),
                cohort.to_vec(),
                format!(
                    "{}:{first}:{index}:{cohort_size}",
                    role.as_str(),
                    cohort_size = cohort.len()
                ),
            );
        }
    }
    for node in &mut nodes {
        let Some(role) = node.role else {
            continue;
        };
        let calls = snapshot.calls.iter().filter(|call| {
            call.role == role && locations.get(call.call_id.as_str()) == Some(&node.id)
        });
        for call in calls {
            for dependency in &call.depends_on {
                if let Some(parent) = locations.get(dependency.as_str()) {
                    if parent != &node.id && !node.dependencies.contains(parent) {
                        node.dependencies.push(parent.clone());
                    }
                }
            }
        }
        node.dependencies.sort();
    }
    nodes
}
