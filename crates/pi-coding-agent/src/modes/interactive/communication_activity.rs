//! Monotonic, session-scoped pulses over optional content-free roster metadata.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use crate::core::occ::activity::{CommunicationActivity, CommunicationKind, MAX_ACTIVITY_MESSAGES};
use crate::modes::daemon::daemon_session_list::SessionSummary;
use crate::modes::agent_connection::types::Occ3Modes;
use crate::modes::interactive::components::subagent_summary_line::{linkage_row_for_roster_summary, RightStatus, RosterParent};
use crate::modes::interactive::theme::theme::theme;

/// Configuration is a steady snapshot, not a message or completed-use receipt.
pub(super) fn mode_status(modes: Option<&Occ3Modes>) -> RightStatus {
    use crate::core::occ::textmas::TextMasMode;
    let (textmas, textmas_marker, textmas_color) = match modes.map(|modes| modes.textmas_mode) {
        Some(TextMasMode::Off) => ("off", "○", "muted"),
        Some(TextMasMode::Sequential) => ("seq", "●", "success"),
        Some(TextMasMode::Hierarchical) => ("hier", "●", "success"),
        None => ("unavailable", "-", "muted"),
    };
    let (latent, latent_marker, latent_color) = match modes {
        Some(modes) if !modes.latent_enabled => ("off", "○", "muted"),
        Some(modes) if modes.latent_active && modes.latent_available => ("on", "●", "success"),
        Some(_) => ("blocked", "!", "warning"),
        None => ("unavailable", "-", "muted"),
    };
    let pair = |compact: bool| {
        let label = |name: &str, value: &str, marker: &str, color: &str| {
            let text = if compact {
                format!("{name} {}", if value == "unavailable" { "n/a" } else { value })
            } else {
                format!("{marker} {name} {value}")
            };
            theme().fg(color, &text)
        };
        format!("{}  {}", label("OCC3.1", textmas, textmas_marker, textmas_color),
            label("OCC3.2", latent, latent_marker, latent_color))
    };
    RightStatus {
        full: format!("{} {}", theme().fg("muted", "CONFIG"), pair(false)),
        compact: format!("{} {}", theme().fg("muted", "CFG"), pair(true)),
        minimal: pair(true),
    }
}

pub(super) const PULSE: Duration = Duration::from_millis(1400);
const QUEUE_FLASH: Duration = Duration::from_millis(500);
const MAX_RECEIVERS: usize = 256;
const MAX_SEEN: usize = 256;

#[derive(Default)]
struct Receiver {
    generation: String,
    seen: VecDeque<String>,
}

#[derive(Default)]
pub(super) struct ActivityState {
    identity: (String, Option<String>, u64),
    initialized: bool,
    receivers: BTreeMap<String, Receiver>,
    until: [Option<Instant>; 4],
    supported: bool,
    queue_started: Option<Instant>,
    observed_since_ms: Option<i64>,
    links: BTreeMap<(String, String), Instant>,
}

impl ActivityState {
    pub(super) fn observe(&mut self, rows: &[SessionSummary], parent: &RosterParent, now: Instant) -> bool {
        let children: BTreeSet<String> = rows.iter().filter(|row| {
            row.runtime_kind.as_deref() == Some("subagent")
                && crate::modes::agents_view::agents_view_state::is_direct_agent_child(
                    &linkage_row_for_roster_summary(row), parent.active_session_id.as_deref(),
                    parent.session_id.as_deref(), parent.session_file.as_deref(),
                )
        }).map(|row| row.session_id.clone()).collect();
        self.observe_scope(rows, parent, &children, now)
    }

    pub(super) fn observe_scope(&mut self, rows: &[SessionSummary], parent: &RosterParent,
        children: &BTreeSet<String>, now: Instant) -> bool {
        let epoch = rows.first().and_then(|row| row.extra.get("communicationObservationEpoch"))
            .and_then(|value| value.as_u64()).unwrap_or(0);
        let identity = (parent.session_id.clone().unwrap_or_default(), parent.active_session_id.clone(), epoch);
        if self.identity != identity {
            *self = Self { identity, ..Self::default() };
        }
        let wall_now_ms = chrono::Utc::now().timestamp_millis();
        let observed_since_ms = *self.observed_since_ms.get_or_insert(wall_now_ms);
        let parent_id = parent.session_id.as_deref().unwrap_or("");
        let family = |id: &str| id == parent_id || children.contains(id);
        self.links.retain(|(from, to), until| family(from) && family(to) && *until > now);
        let mut received = BTreeSet::new();
        let mut pulse = false;
        self.supported = false;
        for row in rows.iter().filter(|row| family(&row.session_id)).take(MAX_RECEIVERS) {
            let Some(activity) = row.extra.get("communicationActivity")
                .and_then(|value| serde_json::from_value::<CommunicationActivity>(value.clone()).ok()) else { continue; };
            self.supported = true;
            received.insert(row.session_id.clone());
            let previous = self.receivers.get(&row.session_id);
            // Reconnects baseline history; newly spawned children may already have a fresh task.
            let new_receiver = previous.is_none();
            let baseline = !self.initialized || previous.is_some_and(|state| state.generation != activity.generation);
            let receiver = self.receivers.entry(row.session_id.clone()).or_default();
            if receiver.generation != activity.generation {
                receiver.generation = activity.generation;
                receiver.seen.clear();
                self.links.retain(|(from, to), _| from != &row.session_id && to != &row.session_id);
            }
            for delivery in activity.deliveries.iter().rev().take(MAX_ACTIVITY_MESSAGES).rev() {
                if receiver.seen.contains(&delivery.id) { continue; }
                receiver.seen.push_back(delivery.id.clone());
                while receiver.seen.len() > MAX_SEEN { receiver.seen.pop_front(); }
                if baseline || (new_receiver && delivery.timestamp < observed_since_ms)
                    || !family(&delivery.source_session_id)
                    || (!children.contains(row.session_id.as_str()) && !children.contains(delivery.source_session_id.as_str())) {
                    continue;
                }
                let index = match delivery.kind {
                    CommunicationKind::Text => 0,
                    CommunicationKind::Occ1 => 1,
                    // OCC2 is live: the roster projection reports Symbolic only
                    // for delivered bodies that decoded through the strict
                    // OCC-2 symbolic path, so the pulse reflects the wire
                    // encoding, not the sender's feature flag. Latent is
                    // reported only for a completed native OCC-3 delivery, so
                    // OCC3.2 pulses on delivery, never on intent.
                    CommunicationKind::Symbolic => 2,
                    CommunicationKind::Latent => 3,
                };
                self.until[index] = Some(now + PULSE);
                let edge = (delivery.source_session_id.clone(), row.session_id.clone());
                if self.links.len() < MAX_SEEN || self.links.contains_key(&edge) {
                    self.links.insert(edge, now + PULSE);
                }
                pulse = true;
            }
        }
        self.receivers.retain(|id, _| received.contains(id));
        self.links.retain(|(_, to), _| received.contains(to));
        if !self.supported { self.until = [None; 4]; self.links.clear(); }
        self.initialized = self.supported;
        pulse
    }

    pub(super) fn live_links(&self, now: Instant) -> impl Iterator<Item = (&str, &str, Instant)> {
        self.links.iter().filter(move |(_, until)| **until > now)
            .map(|((from, to), until)| (from.as_str(), to.as_str(), *until))
    }

    pub(super) fn next_deadline(&self, now: Instant) -> Option<Instant> {
        let queue_deadline = self.queue_started.map(|start| {
            let elapsed = now.saturating_duration_since(start);
            let remainder = Duration::from_nanos((elapsed.as_nanos() % QUEUE_FLASH.as_nanos()) as u64);
            now + (QUEUE_FLASH - remainder)
        });
        self.until.iter().flatten().copied().filter(|deadline| *deadline > now)
            .chain(queue_deadline).min()
    }

    /// The advisory queue badge from the same roster rows: pending OCC-3
    /// queue entries of the roster. Its independent flash is not a delivery pulse: completed-use
    /// deliveries remain the only pulse trigger, and a queued notice lights
    /// no TXT/OCC slot. The roster subscription already strips
    /// `communicationQueue` when the daemon does not advertise the (separate)
    /// queue capability, so field presence is the gate — the badge stays
    /// independent of the activity capability.
    pub(super) fn queue_badge(&mut self, rows: &[SessionSummary], now: Instant) -> String {
        let mut pending = 0usize;
        for row in rows.iter().filter(|row| !row.session_id.is_empty()) {
            let Some(queue) = row.extra.get("communicationQueue") else { continue };
            // The projection's full pending count survives the display
            // bound; counting the bounded entries is only the fallback for
            // hand-shaped rows that carry no pendingTotal.
            if let Some(total) = queue.get("pendingTotal").and_then(serde_json::Value::as_u64) {
                pending = pending.saturating_add(total.min(usize::MAX as u64) as usize);
                continue;
            }
            let Some(entries) = queue.get("entries").and_then(|value| value.as_array()) else { continue };
            for entry in entries {
                let Some(state) = entry.get("state").and_then(serde_json::Value::as_str) else { continue };
                if crate::core::occ::queue::is_waiting_state(state) {
                    pending = pending.saturating_add(1);
                }
            }
        }
        if pending == 0 {
            self.queue_started = None;
            String::new()
        } else {
            let start = *self.queue_started.get_or_insert(now);
            let bright = (now.saturating_duration_since(start).as_nanos() / QUEUE_FLASH.as_nanos()) % 2 == 0;
            let intensity = if bright { "1" } else { "2" };
            let marker = if bright { "●" } else { "○" };
            format!("\x1b[{intensity};38;5;39m{marker} QUEUED {pending}\x1b[22;39m")
        }
    }

    pub(super) fn indicators(&self, now: Instant) -> String {
        ["TXT", "OCC1", "OCC2", "OCC3.2"].iter().enumerate().map(|(index, label)| {
            let active = self.until[index].is_some_and(|until| now < until);
            let marker = if !self.supported { "-" } else if active { "●" } else { "○" };
            let intensity = if active { "1" } else { "2" };
            // ANSI 256-color 39 is neon blue and works without truecolor support.
            format!("\x1b[{intensity};38;5;39m{marker} {label}\x1b[22;39m")
        }).collect::<Vec<_>>().join("  ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use pi_tui::utils::strip_ansi;

    const IDLE: &str = "○ TXT  ○ OCC1  ○ OCC2  ○ OCC3.2";

    fn row(id: &str, child: bool, messages: serde_json::Value) -> SessionSummary {
        SessionSummary {
            id: id.into(), session_id: id.into(), lifecycle: "live".into(),
            active_session_id: Some(id.into()), runtime_kind: Some(if child { "subagent" } else { "top-level" }.into()),
            parent_session_id: child.then(|| "parent".into()),
            extra: std::collections::HashMap::from([
                ("communicationActivity".into(), json!({"generation":"g", "deliveries":messages})),
                ("communicationObservationEpoch".into(), json!(1)),
            ]), ..Default::default()
        }
    }
    fn message(id: &str, from: &str, kind: &str) -> serde_json::Value {
        json!({"id":id, "sourceSessionId":from, "kind":kind, "timestamp":0})
    }
    fn parent() -> RosterParent { RosterParent { session_id: Some("parent".into()), ..Default::default() } }

    #[test]
    fn communication_pulses_on_new_deliveries_expires_and_coalesces_bursts() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let baseline = row("child", true, json!([message("old", "parent", "text")]));
        assert!(!state.observe(&[baseline], &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), IDLE);
        let fresh = row("child", true, json!([message("new", "parent", "text")]));
        assert!(state.observe(&[fresh.clone()], &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "● TXT  ○ OCC1  ○ OCC2  ○ OCC3.2");
        assert!(!state.observe(&[fresh], &parent(), now + Duration::from_millis(500)));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE)), IDLE);
        assert!(state.observe(&[row("child", true, json!([message("burst", "parent", "occ1")]))], &parent(), now + PULSE));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE)), "○ TXT  ● OCC1  ○ OCC2  ○ OCC3.2");
        assert!(state.observe(&[row("child", true, json!([message("burst2", "parent", "occ1")]))], &parent(), now + PULSE + Duration::from_millis(500)));
        assert!(strip_ansi(&state.indicators(now + PULSE * 2)).contains("● OCC1"));
        // A symbolic delivery pulses OCC2 with the same lifetime and
        // coalescing rules; it never lights the OCC1 slot. burst2's OCC1
        // pulse is still live here, so OCC2 lights beside it.
        assert!(state.observe(&[row("child", true, json!([message("sym", "parent", "symbolic")]))], &parent(), now + PULSE * 2));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE * 2)), "○ TXT  ● OCC1  ● OCC2  ○ OCC3.2");
        assert!(!state.observe(&[row("child", true, json!([message("sym", "parent", "symbolic")]))], &parent(), now + PULSE * 2 + Duration::from_millis(500)));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE * 3)), IDLE);
    }

    #[test]
    fn communication_family_scope_includes_siblings_and_child_to_parent_but_not_other_sessions() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut rows = vec![row("parent", false, json!([])), row("a", true, json!([])), row("b", true, json!([]))];
        state.observe(&rows, &parent(), now);
        rows[2] = row("b", true, json!([message("1", "a", "occ1")]));
        assert!(state.observe(&rows, &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "○ TXT  ● OCC1  ○ OCC2  ○ OCC3.2");
        rows[0] = row("parent", false, json!([message("2", "b", "text")]));
        assert!(state.observe(&rows, &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "● TXT  ● OCC1  ○ OCC2  ○ OCC3.2");
        // A text delivery from outside the family stays filtered (the
        // scope rule), while in-family symbolic and latent deliveries pulse
        // OCC2 and OCC3.
        rows[1] = row("a", true, json!([message("3", "unrelated", "text"), message("4", "parent", "symbolic"), message("5", "parent", "latent")]));
        assert!(state.observe(&rows, &parent(), now + PULSE));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE)), "○ TXT  ○ OCC1  ● OCC2  ● OCC3.2");
        assert!(!state.indicators(now + PULSE).contains("LAT"));
    }

    #[test]
    fn communication_history_reconnect_and_session_changes_never_flash() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut rows = vec![row("child", true, json!([]))];
        state.observe(&rows, &parent(), now);
        rows[0] = row("child", true, json!([message("1", "parent", "text")]));
        rows[0].extra.insert("communicationObservationEpoch".into(), json!(2));
        assert!(!state.observe(&rows, &parent(), now));
        rows[0].extra.get_mut("communicationActivity").unwrap()["generation"] = json!("new-runtime");
        assert!(!state.observe(&rows, &parent(), now));
        let other = RosterParent { session_id: Some("other".into()), ..Default::default() };
        assert!(!state.observe(&rows, &other, now));
        assert_eq!(strip_ansi(&state.indicators(now)), "- TXT  - OCC1  - OCC2  - OCC3.2");
        assert!(!state.observe(&rows, &parent(), now));
        rows[0].extra.remove("communicationActivity");
        assert!(!state.observe(&rows, &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "- TXT  - OCC1  - OCC2  - OCC3.2");
    }

    #[test]
    fn symbolic_deliveries_pulse_the_occ2_slot_exactly_like_occ1() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        // Baseline first, so both bursts are non-baseline deliveries.
        state.observe(&[row("child", true, json!([message("base", "parent", "text")]))], &parent(), now);
        // An occ1 delivery pulses the OCC1 slot only.
        let occ = row("child", true, json!([message("occ", "parent", "occ1")]));
        assert!(state.observe(&[occ], &parent(), now + Duration::from_millis(100)));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(100))), "○ TXT  ● OCC1  ○ OCC2  ○ OCC3.2");
        // A symbolic delivery drives the same pulse path (dedupe, family
        // scope, PULSE lifetime) with no special-casing: it lights OCC2
        // only, never OCC1.
        let sym = row("child", true, json!([message("sym", "parent", "symbolic")]));
        assert!(state.observe(&[sym.clone()], &parent(), now + Duration::from_millis(200)));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(200))), "○ TXT  ● OCC1  ● OCC2  ○ OCC3.2");
        // Identical replay is a no-op, exactly like an occ1 replay.
        assert!(!state.observe(&[sym], &parent(), now + Duration::from_millis(300)));
        // The OCC2 pulse expires on the same PULSE clock as OCC1: both lit
        // at +200ms are dark again at +200ms + PULSE.
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(200) + PULSE)), IDLE);
        // A fresh symbolic delivery after expiry pulses again.
        assert!(state.observe(&[row("child", true, json!([message("sym2", "parent", "symbolic")]))], &parent(), now + Duration::from_millis(200) + PULSE));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(200) + PULSE)), "○ TXT  ○ OCC1  ● OCC2  ○ OCC3.2");
    }

    #[test]
    fn idle_supported_row_renders_four_inactive_segments_and_pulses_nothing() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let rows = [row("parent", false, json!([])), row("child", true, json!([]))];
        assert!(!state.observe(&rows, &parent(), now));
        // Exact current spacing: marker, one space, label, two spaces between segments.
        assert_eq!(strip_ansi(&state.indicators(now)), "○ TXT  ○ OCC1  ○ OCC2  ○ OCC3.2");
        assert_eq!(strip_ansi(&state.indicators(now)).chars().count(), 31);
        assert!(!state.indicators(now).contains('●'));
        assert_eq!(state.next_deadline(now), None);
        // An unchanged idle row keeps every segment inactive.
        assert!(!state.observe(&rows, &parent(), now + Duration::from_millis(1)));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(1))), IDLE);
        assert_eq!(state.next_deadline(now + Duration::from_millis(1)), None);
    }

    fn pulsing_kind(kind: &str) -> String {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("child", true, json!([]))], &parent(), now);
        assert!(state.observe(&[row("child", true, json!([message("one", "parent", kind)]))], &parent(), now));
        strip_ansi(&state.indicators(now))
    }

    #[test]
    fn queue_badge_counts_pending_entries_without_pulsing() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut rows = vec![
            row("parent", false, json!([])),
            row("child", true, json!([])),
        ];
        state.observe(&rows, &parent(), now);
        // No queue field: no badge.
        assert_eq!(state.queue_badge(&rows, now), "");
        rows[1].extra.insert("communicationQueue".into(), json!({
            "generation":"g",
            "entries":[
                {"queueId":"q1","state":"pending"},
                {"queueId":"q2","state":"armed"},
                {"queueId":"q3","state":"fallback"},
                {"queueId":"q4","state":"failed"},
                {"queueId":"q5","state":"admitted"},
                {"queueId":"q6","state":"future-unknown"}
            ]
        }));
        // Only pending + armed = 2; all other states
        // (fallback/failed, and the corrected completed/expired) excluded.
        rows[1].extra.get_mut("communicationQueue").unwrap()["entries"].as_array_mut().unwrap()
            .push(json!({"queueId":"q7","state":"completed"}));
        rows[1].extra.get_mut("communicationQueue").unwrap()["entries"].as_array_mut().unwrap()
            .push(json!({"queueId":"q8","state":"expired"}));
        assert_eq!(state.queue_badge(&rows, now), "\x1b[1;38;5;39m● QUEUED 2\x1b[22;39m");
        // pendingTotal is authoritative when present: it survives the
        // display bound, so larger configured queues are not undercounted.
        rows[1].extra.get_mut("communicationQueue").unwrap()["pendingTotal"] = json!(11);
        assert_eq!(state.queue_badge(&rows, now), "\x1b[1;38;5;39m● QUEUED 11\x1b[22;39m");
        // The independent flash never lights a delivery slot.
        assert_eq!(strip_ansi(&state.indicators(now)), IDLE);
        assert_eq!(state.next_deadline(now), Some(now + QUEUE_FLASH));
        assert_eq!(state.queue_badge(&rows, now + QUEUE_FLASH), "\x1b[2;38;5;39m○ QUEUED 11\x1b[22;39m");
        assert_eq!(state.next_deadline(now + QUEUE_FLASH), Some(now + QUEUE_FLASH * 2));
        assert_eq!(state.queue_badge(&rows, now + QUEUE_FLASH * 2), "\x1b[1;38;5;39m● QUEUED 11\x1b[22;39m");
        // Intermediate renders retain the original boundary rather than postponing it.
        assert_eq!(state.next_deadline(now + Duration::from_millis(750)), Some(now + QUEUE_FLASH * 2));
        // A queue field with no entries array and no pendingTotal is ignored.
        rows[1].extra.insert("communicationQueue".into(), json!({"generation":"g"}));
        assert_eq!(state.queue_badge(&rows, now), "");
        assert_eq!(state.next_deadline(now), None);
    }

    #[test]
    fn queue_flash_is_independent_of_activity_and_stops_on_capability_loss() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut rows = vec![row("parent", false, json!([]))];
        rows[0].extra.remove("communicationActivity");
        rows[0].extra.insert("communicationQueue".into(), json!({"pendingTotal": 1}));
        state.observe(&rows, &parent(), now);
        assert!(state.queue_badge(&rows, now).contains("● QUEUED 1"));
        assert_eq!(state.next_deadline(now), Some(now + QUEUE_FLASH));
        assert_eq!(strip_ansi(&state.indicators(now)), "- TXT  - OCC1  - OCC2  - OCC3.2");
        // Delivery expiry takes precedence when earlier than the flash boundary.
        state.until[3] = Some(now + Duration::from_millis(100));
        assert_eq!(state.next_deadline(now), state.until[3]);
        rows[0].extra.remove("communicationQueue");
        assert_eq!(state.queue_badge(&rows, now), "");
        assert_eq!(state.next_deadline(now + Duration::from_millis(100)), None);
        rows[0].extra.insert("communicationQueue".into(), json!({"pendingTotal": 1}));
        assert!(state.queue_badge(&rows, now + QUEUE_FLASH).starts_with("\x1b[1;38;5;39m"));
        // Session changes discard the old animation clock.
        state.observe(&[], &RosterParent::default(), now + QUEUE_FLASH);
        assert_eq!(state.next_deadline(now + QUEUE_FLASH), None);
    }

    #[test]
    fn queue_projection_admission_drains_waiting_badge_and_deadline() {
        use crate::core::occ::queue::{from_queue_notices, QUEUE_NOTICE_CUSTOM_TYPE};
        use pi_agent_core::types::{AgentMessage, CustomAgentMessage, CustomMessageContent};
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut rows = vec![row("parent", false, json!([]))];
        let mut messages = Vec::new();
        for (id, phase, count) in [("q1", "pending", 1), ("q2", "armed", 2),
            ("q1", "admitted", 1), ("q2", "admitted", 0)] {
            messages.push(AgentMessage::Custom(CustomAgentMessage::Custom {
                custom_type: QUEUE_NOTICE_CUSTOM_TYPE.into(),
                content: CustomMessageContent::Text(String::new()), display: false,
                details: Some(json!({"queueId": id, "state": phase, "position": 1,
                    "receiver": "child:a"})), timestamp: 0,
            }));
            let snapshot = from_queue_notices(&messages, "parent", "g");
            assert_eq!(snapshot.pending_total, count);
            rows[0].extra.insert("communicationQueue".into(), serde_json::to_value(snapshot).unwrap());
            let badge = state.queue_badge(&rows, now);
            if count == 0 {
                assert!(badge.is_empty());
                assert_eq!(state.next_deadline(now), None);
            } else {
                assert_eq!(strip_ansi(&badge), format!("● QUEUED {count}"));
                assert_eq!(state.next_deadline(now), Some(now + QUEUE_FLASH));
            }
        }
        for phase in ["published", "admitted", "delivered", "failed", "cancelled", "uncertain"] {
            rows[0].extra.insert("communicationQueue".into(), json!({"entries": [{"state": phase}]}));
            assert!(state.queue_badge(&rows, now).is_empty(), "{phase}");
        }
    }

    #[test]
    fn each_delivery_kind_pulses_only_its_own_segment() {
        for (kind, expected) in [
            ("text", "● TXT  ○ OCC1  ○ OCC2  ○ OCC3.2"),
            ("occ1", "○ TXT  ● OCC1  ○ OCC2  ○ OCC3.2"),
            ("symbolic", "○ TXT  ○ OCC1  ● OCC2  ○ OCC3.2"),
            ("latent", "○ TXT  ○ OCC1  ○ OCC2  ● OCC3.2"),
        ] {
            assert_eq!(pulsing_kind(kind), expected, "{kind}");
        }
    }

    #[test]
    fn latent_pulse_uses_the_existing_neon_blue_style_and_leaves_other_segments_dim() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("child", true, json!([]))], &parent(), now);
        assert!(state.observe(&[row("child", true, json!([message("lat", "parent", "latent")]))], &parent(), now));
        let rendered = state.indicators(now);
        assert!(rendered.contains("\x1b[1;38;5;39m● OCC3.2\x1b[22;39m"), "{rendered}");
        assert!(rendered.contains("\x1b[2;38;5;39m○ TXT\x1b[22;39m"), "{rendered}");
        assert!(rendered.contains("\x1b[2;38;5;39m○ OCC1\x1b[22;39m"), "{rendered}");
        assert!(rendered.contains("\x1b[2;38;5;39m○ OCC2\x1b[22;39m"), "{rendered}");
        assert_eq!(rendered.matches('●').count(), 1);
        assert_eq!(state.next_deadline(now), Some(now + PULSE));
    }

    #[test]
    fn occ3_delivery_pulses_once_per_id_and_never_replays_after_expiry() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("child", true, json!([]))], &parent(), now);
        let delivery = row("child", true, json!([message("lat", "parent", "latent")]));
        assert!(state.observe(&[delivery.clone()], &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "○ TXT  ○ OCC1  ○ OCC2  ● OCC3.2");
        assert_eq!(state.next_deadline(now), Some(now + PULSE));
        // The same id inside the pulse window is suppressed and keeps the deadline.
        assert!(!state.observe(&[delivery.clone()], &parent(), now + Duration::from_millis(100)));
        assert_eq!(state.next_deadline(now + Duration::from_millis(100)), Some(now + PULSE));
        // The identical roster row replayed after expiry must not re-pulse.
        assert!(!state.observe(&[delivery], &parent(), now + PULSE));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE)), IDLE);
        assert_eq!(state.next_deadline(now + PULSE), None);
        // A new delivery id after expiry pulses OCC3 again.
        assert!(state.observe(&[row("child", true, json!([message("lat2", "parent", "latent")]))], &parent(), now + PULSE));
        assert_eq!(strip_ansi(&state.indicators(now + PULSE)), "○ TXT  ○ OCC1  ○ OCC2  ● OCC3.2");
    }

    #[test]
    fn occ3_pulse_expires_within_pulse_and_clears_its_deadline() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("child", true, json!([]))], &parent(), now);
        assert!(state.observe(&[row("child", true, json!([message("lat", "parent", "latent")]))], &parent(), now));
        assert!(state.indicators(now).contains("● OCC3.2"));
        assert!(state.indicators(now + PULSE - Duration::from_millis(1)).contains("● OCC3.2"));
        assert!(state.indicators(now + PULSE).contains("○ OCC3.2"));
        assert_eq!(state.next_deadline(now), Some(now + PULSE));
        assert_eq!(state.next_deadline(now + PULSE), None);
    }

    #[test]
    fn latent_delivery_from_outside_the_family_never_pulses() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("parent", false, json!([])), row("child", true, json!([]))], &parent(), now);
        let rows = [
            row("parent", false, json!([])),
            row("child", true, json!([message("lat", "unrelated", "latent")])),
        ];
        assert!(!state.observe(&rows, &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), IDLE);
        assert_eq!(state.next_deadline(now), None);
    }

    #[test]
    fn unsupported_rows_render_four_dashes_and_never_pulse_occ3() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut plain = row("child", true, json!([]));
        plain.extra.remove("communicationActivity");
        assert!(!state.observe(&[plain.clone()], &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "- TXT  - OCC1  - OCC2  - OCC3.2");
        assert!(!state.indicators(now).contains('●'));
        assert_eq!(state.next_deadline(now), None);
        // Losing the projection clears a live OCC3 pulse.
        state.observe(&[row("child", true, json!([]))], &parent(), now);
        assert!(state.observe(&[row("child", true, json!([message("lat", "parent", "latent")]))], &parent(), now));
        assert!(state.indicators(now).contains("● OCC3.2"));
        assert!(!state.observe(&[plain], &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "- TXT  - OCC1  - OCC2  - OCC3.2");
        assert_eq!(state.next_deadline(now), None);
    }

    #[test]
    fn generation_change_resets_the_pulse_array_to_all_inactive() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("child", true, json!([]))], &parent(), now);
        assert!(state.observe(&[row("child", true, json!([message("lat", "parent", "latent")]))], &parent(), now));
        assert_eq!(strip_ansi(&state.indicators(now)), "○ TXT  ○ OCC1  ○ OCC2  ● OCC3.2");
        assert_eq!(state.next_deadline(now), Some(now + PULSE));
        // A new observation epoch is a new roster generation: the whole pulse
        // array is replaced, so the live OCC3 deadline is gone at once.
        let epoch = |messages: serde_json::Value| {
            let mut row = row("child", true, messages);
            row.extra.insert("communicationObservationEpoch".into(), json!(2));
            row
        };
        assert!(!state.observe(&[epoch(json!([message("lat", "parent", "latent")]))], &parent(), now + Duration::from_millis(1)));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(1))), IDLE);
        assert_eq!(state.next_deadline(now + Duration::from_millis(1)), None);
        // The same delivery id in the new generation is baselined, not replayed.
        assert!(!state.observe(&[epoch(json!([message("lat", "parent", "latent")]))], &parent(), now + Duration::from_millis(2)));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(2))), IDLE);
        assert_eq!(state.next_deadline(now + Duration::from_millis(2)), None);
        // A per-receiver runtime generation change baselines as well: no pulse.
        let mut reconnected = epoch(json!([message("lat", "parent", "latent")]));
        reconnected.extra.get_mut("communicationActivity").unwrap()["generation"] = json!("restarted");
        assert!(!state.observe(&[reconnected], &parent(), now + Duration::from_millis(3)));
        assert_eq!(strip_ansi(&state.indicators(now + Duration::from_millis(3))), IDLE);
        assert_eq!(state.next_deadline(now + Duration::from_millis(3)), None);
    }

    #[test]
    fn four_segments_degrade_without_panic_or_overflow_at_narrow_widths() {
        use crate::core::context_health::ContextHealthInput;
        use crate::modes::interactive::components::subagent_summary_line::SubagentSummaryLine;
        use pi_tui::{tui::Component, utils::visible_width};
        use super::super::context_health_indicator::status_line;
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("parent", false, json!([])), row("child", true, json!([]))], &parent(), now);
        assert!(state.observe(&[row("parent", false, json!([])), row("child", true, json!([message("lat", "parent", "latent")]))], &parent(), now));
        let indicators = state.indicators(now);
        assert!(indicators.contains("● OCC3.2"), "{indicators}");
        assert_eq!(visible_width(&strip_ansi(&indicators)), 31);
        for width in [2usize, 5, 8, 20, 30, 40, 60, 80, 120] {
            let status = status_line(&indicators, "", "", &ContextHealthInput::default(), width.saturating_sub(2));
            if width >= 8 {
                assert!(visible_width(&status) <= width.saturating_sub(2), "{width}: {status:?}");
            }
            let mut line = SubagentSummaryLine::default();
            line.set_always_visible(true);
            line.set_communication_status(Some(status));
            for rendered in line.render(width as f64) {
                assert!(visible_width(&rendered) <= width, "{width}: {rendered:?}");
            }
        }
    }

    #[test]
    fn admitted_message_projection_reaches_neon_bar_for_all_three_formats() {
        use crate::core::agent_messages::{create_agent_session_message, AgentSessionMessagePayload, AgentSessionMessageSender, AgentSessionMessageEndpoint};
        use crate::core::occ::{activity::from_admitted_messages, contract::AgentCommMessage, dialect::DialectCard, occ2};
        use pi_agent_core::types::AgentMessage;
        use crate::modes::interactive::components::subagent_summary_line::SubagentSummaryLine;
        use pi_tui::tui::Component;
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let contract = AgentCommMessage::new("TASK", "test-task", "pending");
        let cases = [
            ("TXT", "Please inspect this task".to_string()),
            ("OCC1", contract.to_occ1().unwrap().encode().unwrap()),
            ("OCC2", occ2::encode_dual(&contract, &DialectCard::sym_v1()).unwrap()),
        ];
        for (label, body) in cases {
            let now = Instant::now();
            let mut state = ActivityState::default();
            let parent_row = row("parent", false, json!([]));
            state.observe(&[parent_row.clone()], &parent(), now);
            let message = AgentMessage::Custom(create_agent_session_message(&AgentSessionMessagePayload {
                id: "fresh-delivery".into(), message: body,
                from: Some(AgentSessionMessageSender { session_id: Some("parent".into()), ..Default::default() }),
                target: AgentSessionMessageEndpoint { session_id: "child".into(), ..Default::default() },
                ..Default::default()
            }, chrono::Utc::now().timestamp_millis()));
            let mut child = row("child", true, json!([]));
            child.extra.insert("communicationActivity".into(), serde_json::to_value(from_admitted_messages(&[message], "child", "g")).unwrap());
            let rows = [parent_row, child];
            assert!(state.observe(&rows, &parent(), now));
            let mut line = SubagentSummaryLine::default();
            line.set_always_visible(true);
            line.set_communication_status(Some(state.indicators(now)));
            let rendered = line.render(80.0).join("\n");
            assert!(rendered.contains(&format!("\x1b[1;38;5;39m● {label}")), "{rendered}");
            assert!(!state.observe(&rows, &parent(), now));
            assert_eq!(state.next_deadline(now), Some(now + PULSE));
            assert!(state.next_deadline(now + PULSE).is_none());
            line.set_communication_status(Some(state.indicators(now + PULSE)));
            assert!(strip_ansi(&line.render(80.0).join("\n")).contains(&format!("○ {label}")));
        }
    }

    #[test]
    fn communication_new_child_first_delivery_flashes_but_old_history_does_not() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        state.observe(&[row("parent", false, json!([]))], &parent(), now);
        let mut fresh = message("spawn:new", "parent", "symbolic");
        fresh["timestamp"] = json!(chrono::Utc::now().timestamp_millis());
        let rows = vec![row("parent", false, json!([])), row("new-child", true, json!([fresh]))];
        assert!(state.observe(&rows, &parent(), now));
        assert!(state.indicators(now).contains("\x1b[1;38;5;39m● OCC2"));
        assert!(!state.observe(&rows, &parent(), now));
        let old = vec![row("parent", false, json!([])), row("old-child", true, json!([message("historical", "parent", "text")]))];
        assert!(!state.observe(&old, &parent(), now + PULSE));
        assert!(!state.indicators(now + PULSE).contains('●'));
    }

    #[test]
    fn communication_reply_from_just_completed_child_still_flashes() {
        let now = Instant::now();
        let mut state = ActivityState::default();
        let mut rows = vec![row("parent", false, json!([])), row("child", true, json!([]))];
        state.observe(&rows, &parent(), now);
        rows[1].lifecycle = "saved".into();
        rows[0] = row("parent", false, json!([message("reply", "child", "occ1")]));
        assert!(state.observe(&rows, &parent(), now));
        assert!(state.indicators(now).contains("\x1b[1;38;5;39m● OCC1"));
    }

}
