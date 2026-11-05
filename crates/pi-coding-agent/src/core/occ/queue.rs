//! Content-free queue visibility projected from the sender's own notice
//! records, mirroring the activity projection's discipline.
//!
//! The OCC-3 sender queue (a separate owner) appends bounded
//! `occ3_queue_notice` custom records to the owning session's transcript on
//! every queue state change. This module is the READ side: it rebuilds the
//! current queue view from those records so roster metadata can surface
//! pending native handoffs without any new plumbing between the queue and
//! the session-summary projection. The records are advisory UI metadata:
//! they never feed the TXT/OCC1/OCC2/OCC3 pulse projection, which only ever
//! scans `agent_message` records (`crate::core::occ::activity`).
//!
//! The sender also emits internal custom records this projection must
//! tolerate but never read: the fallback re-emission prompt
//! (`occ3_fallback_prompt`, details carry only a queueId). The scan filters
//! by [`QUEUE_NOTICE_CUSTOM_TYPE`], so every other custom type — including
//! that prompt — is skipped without interpretation.
//!
//! Two record forms share one custom type:
//! - an INDIVIDUAL notice: one queue id's new state;
//! - a SNAPSHOT notice (`"snapshot": true` with an `entries` array): the
//!   queue's full current view in one record. The newest snapshot is the
//!   authoritative base; individual notices newer than it override or add.
//!   Because terminal entries are popped only after their terminal notice,
//!   a snapshot plus the newer individual notices reconstructs the queue
//!   exactly, and the scan never depends on stale history depth.

use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The custom type of one queue-notice record, written by the queue owner
/// into its own transcript. Deliberately distinct from
/// [`crate::core::agent_messages::AGENT_MESSAGE_CUSTOM_TYPE`] so notices
/// can never be mistaken for
/// delivered agent messages (and so they never pulse).
pub const QUEUE_NOTICE_CUSTOM_TYPE: &str = "occ3_queue_notice";
/// How many queue entries the snapshot's `entries` list may carry. The
/// full pending count survives in `pending_total` and the remainder is
/// reported in `omitted`, so a larger configured queue is never truncated
/// silently.
pub const MAX_QUEUE_ENTRIES: usize = 8;
/// How many transcript records the projection scans, newest first. With
/// snapshot notices the scan stops at the newest snapshot, so this bound
/// only guards the individual-notice form; 512 covers a full cycle of the
/// queue (one notice per transition) with headroom for popped history.
const QUEUE_SCAN_LIMIT: usize = 512;
const MAX_QUEUE_ID_CHARS: usize = 128;

/// Queue states the sender queue treats as terminal for visibility:
/// `fallback` (only after the ordinary fallback delivery was acknowledged),
/// `failed`, `completed` and `expired` leave no pending work. Every other
/// state — `pending`, `armed`, `published`, `admitted`, the intermediate
/// `fallback_pending` (ordinary fallback in flight, not yet acknowledged),
/// and unknown future states — still represents queued work and stays
/// visible. Conservative for visibility: an unrecognized state keeps the
/// badge honest rather than hiding work.
pub fn is_terminal_state(state: &str) -> bool {
    matches!(state, "fallback" | "failed" | "completed" | "expired")
}

/// The sender queue's requested UI label for one raw state. Terminal and
/// in-flight native states map to their user-facing names;
/// `fallback_pending` (ordinary fallback in flight, still counted as
/// pending) renders as "fallback". Unknown states pass through unchanged
/// so an unrecognized state stays visible and honest rather than
/// relabeled.
pub fn display_state(state: &str) -> &str {
    match state {
        "pending" => "queued",
        "armed" => "armed",
        "published" => "in-flight",
        // A transport acknowledgment is NOT consumption: the receiver's
        // inbox admitted the body, but completed use is proven only by the
        // receiver's receipt. The label states exactly what happened.
        "admitted" => "acknowledged",
        // Unfinished ordinary re-emission: pending work, rendered so users
        // distinguish it from the terminal, acknowledged fallback.
        "fallback_pending" => "fallback (pending)",
        "fallback" => "fallback",
        "completed" => "completed",
        "expired" => "expired",
        "failed" => "failed",
        // A possibly-consumed publication awaiting explicit receipt
        // reconciliation: visible, counted pending, never auto-resolved or
        // TTL-expired. "uncertain" is the sender's frozen final name;
        // earlier provisional spellings stay mapped for old records.
        "uncertain" | "reconciling" | "blocked" => "uncertain (pending receipt)",
        other => other,
    }
}

fn bounded_text(value: &serde_json::Value) -> Option<&str> {
    value
        .as_str()
        .filter(|text| !text.is_empty() && text.chars().count() <= MAX_QUEUE_ID_CHARS && !text.chars().any(char::is_control))
}

fn bounded_count(value: &serde_json::Value) -> Option<u32> {
    let number = value.as_u64()?;
    (number <= u32::MAX as u64).then_some(number as u32)
}

/// One queue entry, exactly as the wire carries it: content-free ids,
/// selector and counters. No handles, no receipts, no digests, no bodies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunicationQueueEntry {
    pub queue_id: String,
    /// Absent until the daemon mints the real handoff message id at publish
    /// time; rendered as an explicit placeholder, never fabricated.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message_id: String,
    pub position: u32,
    pub receiver: String,
    pub state: String,
    #[serde(default)]
    pub attempts: u32,
    pub timestamp: i64,
    /// The ordinary-delivery transport verdict for fallback states,
    /// `ordinary_delivery:delivered|queued` in the notice, bounded and
    /// passed through verbatim. Absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinary_delivery: Option<String>,
    /// The manual resolution that drained an uncertain entry:
    /// `manual_reemit` (uncertain -> fallback_pending) or `manual_cancel`
    /// (uncertain -> failed). Whitelist-filtered like the transport
    /// verdict, so free-text reasons never surface. Absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// One queue-level enqueue refusal event, as the sender's refusal notice
/// carries it: a whitelisted reason and the event timestamp. Content-free;
/// never an entry and never counted as queued work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunicationQueueRefusal {
    pub reason: String,
    pub timestamp: i64,
}

/// The queue view of one session: the owning session's runtime generation,
/// the bounded newest entries, and the counts that survive the display
/// bound. `pending_total` counts every non-terminal entry before
/// truncation; `omitted` is how many PENDING entries the bounded list
/// dropped. Terminal history beyond the bound is not queued work and is
/// not counted, so a deep completion history can never inflate the
/// "+N more queued" rendering. `blocked` carries the sender's whole-queue
/// load failure (a whitelisted `unsupported_schema` / `unreadable` /
/// `unreadable_io` value from the newest snapshot notice): the queue
/// refuses new handoffs until the model starts a fresh queue. `refused`
/// carries the newest enqueue-refusal event, cleared when a newer
/// entry-bearing notice proves admission resumed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunicationQueueSnapshot {
    pub generation: String,
    pub entries: Vec<CommunicationQueueEntry>,
    #[serde(default)]
    pub omitted: usize,
    #[serde(default)]
    pub pending_total: usize,
    /// Whole-queue blocked state from the newest snapshot notice, when the
    /// sender's persisted queue file could not be loaded safely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked: Option<String>,
    /// The newest queue-level enqueue refusal event (queue full / blocked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused: Option<CommunicationQueueRefusal>,
    /// The sender's durable notice outbox hit its bound at the newest
    /// snapshot emission: some notices are undeliverable until the outbox
    /// drains. A visible fail state, never a silent drop, and it clears only
    /// when a newer snapshot omits the flag.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub notice_overflow: bool,
}

impl CommunicationQueueSnapshot {
    /// Entries that still represent queued work. Terminal entries
    /// (fallback/failed/completed/expired) stay in [`Self::entries`] for
    /// reporting, but the badge and the pending count use
    /// [`Self::pending_total`], which also counts entries beyond the
    /// display bound.
    pub fn pending_entries(&self) -> impl Iterator<Item = &CommunicationQueueEntry> {
        self.entries
            .iter()
            .filter(|entry| !is_terminal_state(&entry.state))
    }
}

/// Parses one notice payload (or one snapshot element) into an entry.
/// `record_timestamp` backs elements that carry no own timestamp. Position
/// is required: without it ordering cannot be reported truthfully.
fn entry_from_details(details: &serde_json::Value, record_timestamp: i64) -> Option<CommunicationQueueEntry> {
    let queue_id = bounded_text(&details["queueId"])?;
    let receiver = bounded_text(&details["receiver"])?;
    let state = bounded_text(&details["state"])?;
    let position = bounded_count(&details["position"])?;
    Some(CommunicationQueueEntry {
        queue_id: queue_id.to_owned(),
        message_id: bounded_text(&details["messageId"]).unwrap_or_default().to_owned(),
        position,
        receiver: receiver.to_owned(),
        state: state.to_owned(),
        attempts: bounded_count(&details["attempts"]).unwrap_or(0),
        timestamp: details
            .get("timestamp")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(record_timestamp),
        ordinary_delivery: bounded_text(&details["ordinary_delivery"])
            .map(str::to_owned)
            .filter(|reason| reason == "delivered" || reason == "queued"),
        reason: bounded_text(&details["reason"])
            .map(str::to_owned)
            .filter(|reason| reason == "manual_reemit" || reason == "manual_cancel"),
    })
}

/// Rebuild the current queue view from the newest `occ3_queue_notice`
/// records in one session's transcript. The newest snapshot record (when
/// the queue emits any) is the authoritative base; individual notices
/// newer than it override or add entries, and everything older than the
/// snapshot is superseded. Without snapshots the newest individual notice
/// per queue id wins, bounded by [`QUEUE_SCAN_LIMIT`]. The result is
/// bounded, ordered by queue position, and empty when the session has no
/// queue (or a notice does not name this session).
pub fn from_queue_notices(messages: &[AgentMessage], session_id: &str, generation: &str) -> CommunicationQueueSnapshot {
    let mut newest: BTreeMap<String, CommunicationQueueEntry> = BTreeMap::new();
    let mut base: Option<BTreeMap<String, CommunicationQueueEntry>> = None;
    let mut blocked: Option<String> = None;
    let mut refusal: Option<CommunicationQueueRefusal> = None;
    let mut notice_overflow = false;
    let mut newest_admission_timestamp: Option<i64> = None;
    for message in messages.iter().rev().take(QUEUE_SCAN_LIMIT) {
        let AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, details: Some(details), timestamp, .. }) = message else { continue };
        if custom_type != QUEUE_NOTICE_CUSTOM_TYPE {
            continue;
        }
        // The notice contract carries no sessionId; when one is present it
        // must name this session, so a foreign record never projects onto
        // this row.
        if details.get("sessionId").is_some_and(|value| value.as_str() != Some(session_id)) {
            continue;
        }
        // A queue-level refusal notice (the sender's queue_refusal_notice):
        // entry-less by construction, so it never reaches the entry paths
        // below. Only the whitelisted values surface; the newest refusal
        // in the scan window wins, and it clears when a newer
        // entry-bearing notice proves the queue admitted work again.
        if let Some(reason) = details.get("refused").and_then(serde_json::Value::as_str) {
            if matches!(reason, "queue_full" | "queue_blocked") {
                // Newest refusal wins: the scan is newest-first, so only
                // the first refusal record found sets the event.
                if refusal.is_none() {
                    refusal = Some(CommunicationQueueRefusal {
                        reason: reason.to_owned(),
                        timestamp: *timestamp,
                    });
                }
                continue;
            }
        }
        // A snapshot was already adopted from newer records: everything
        // older is superseded by it, so the scan stops here.
        if base.is_some() {
            break;
        }
        if details.get("snapshot").and_then(serde_json::Value::as_bool) == Some(true) {
            let mut entries = BTreeMap::new();
            if let Some(list) = details.get("entries").and_then(serde_json::Value::as_array) {
                for item in list {
                    if let Some(entry) = entry_from_details(item, *timestamp) {
                        entries.insert(entry.queue_id.clone(), entry);
                    }
                }
            }
            // The whole-queue blocked state rides the snapshot notice; only
            // the whitelisted load-failure values surface.
            if let Some(reason) = details.get("blocked").and_then(serde_json::Value::as_str) {
                if matches!(reason, "unsupported_schema" | "unreadable" | "unreadable_io") {
                    blocked = Some(reason.to_owned());
                }
            }
            // The outbox overflow flag rides the snapshot too: the newest
            // snapshot is authoritative, so a later snapshot without the
            // flag clears it.
            notice_overflow = details.get("noticeOverflow").and_then(serde_json::Value::as_bool) == Some(true);
            // A snapshot alone never proves admission (the drain emits one
            // after a refusal too): only an entry actually enqueued after
            // the refusal does. An EMPTY snapshot proves nothing either —
            // the queue may have emptied through expiry, not admission —
            // so only entry timestamps carried by the snapshot count.
            if let Some(snapshot_admission) = entries
                .values()
                .map(|entry| entry.timestamp)
                .max()
            {
                newest_admission_timestamp = Some(
                    newest_admission_timestamp
                        .map_or(snapshot_admission, |previous: i64| previous.max(snapshot_admission)),
                );
            }
            base = Some(entries);
            continue;
        }
        // Newest first: only the first notice per queue id is kept. An
        // entry-bearing individual notice proves the queue admitted work
        // at its own timestamp (its entries carry the enqueue time).
        if let Some(entry) = entry_from_details(details, *timestamp) {
            newest_admission_timestamp = Some(newest_admission_timestamp.unwrap_or(*timestamp).max(*timestamp));
            newest.entry(entry.queue_id.clone()).or_insert(entry);
        }
    }
    let mut all: BTreeMap<String, CommunicationQueueEntry> = base.unwrap_or_default();
    for (queue_id, entry) in newest {
        // Individual notices newer than the snapshot override its view.
        all.insert(queue_id, entry);
    }
    // A refusal only stays visible while it is the newest event: an
    // entry-bearing notice (or an adopted snapshot) recorded after it proves
    // the queue admitted work again, so the banner clears instead of
    // claiming a refusal that no longer holds.
    let refused = match (&refusal, newest_admission_timestamp) {
        (Some(refusal), Some(admitted_at)) => (refusal.timestamp > admitted_at).then(|| refusal.clone()),
        (Some(refusal), None) => Some(refusal.clone()),
        (None, _) => None,
    };
    let pending_total = all
        .values()
        .filter(|entry| !is_terminal_state(&entry.state))
        .count();
    // The bounded display list: live (non-terminal) entries are the
    // queue's work, so they claim the display slots first, by position.
    // Terminal entries are already popped on the sender and kept only
    // for reporting; they fill the remaining slots, newest first. A
    // deep terminal history can therefore never crowd live work out of
    // the bounded list.
    let mut live: Vec<CommunicationQueueEntry> = all
        .values()
        .filter(|entry| !is_terminal_state(&entry.state))
        .cloned()
        .collect();
    let mut terminal: Vec<CommunicationQueueEntry> = all
        .values()
        .filter(|entry| is_terminal_state(&entry.state))
        .cloned()
        .collect();
    live.sort_by(|a, b| (a.position, a.queue_id.as_str()).cmp(&(b.position, b.queue_id.as_str())));
    terminal.sort_by(|a, b| {
        b.timestamp
            .cmp(&a.timestamp)
            .then_with(|| a.queue_id.as_str().cmp(b.queue_id.as_str()))
    });
    // `omitted` counts only the live entries that did not fit, so the
    // "+N more queued" rendering stays honest: dropped terminal history
    // is not queued work.
    let omitted = live.len().saturating_sub(MAX_QUEUE_ENTRIES);
    live.truncate(MAX_QUEUE_ENTRIES);
    let room = MAX_QUEUE_ENTRIES.saturating_sub(live.len());
    live.extend(terminal.into_iter().take(room));
    // Display order stays position-then-id across the kept set.
    live.sort_by(|a, b| (a.position, a.queue_id.as_str()).cmp(&(b.position, b.queue_id.as_str())));
    CommunicationQueueSnapshot {
        generation: generation.to_owned(),
        entries: live,
        omitted,
        pending_total,
        blocked,
        refused,
        notice_overflow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent_messages::AGENT_MESSAGE_CUSTOM_TYPE;
    use pi_agent_core::types::CustomMessageContent;
    use serde_json::json;

    fn notice(queue_id: &str, state: &str, position: u64, timestamp: i64) -> AgentMessage {
        AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: QUEUE_NOTICE_CUSTOM_TYPE.to_string(),
            content: CustomMessageContent::Text("queue notice".to_string()),
            display: false,
            details: Some(json!({
                "queueId": queue_id,
                "messageId": format!("msg-{queue_id}"),
                "position": position,
                "receiver": "child:receiver-a",
                "state": state,
                "attempts": 0,
                "reason": "internal-only-field",
                "body": "private handoff body",
            })),
            timestamp,
        })
    }

    fn with_details(details: serde_json::Value, timestamp: i64) -> AgentMessage {
        AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: QUEUE_NOTICE_CUSTOM_TYPE.to_string(),
            content: CustomMessageContent::Text("queue notice".to_string()),
            display: false,
            details: Some(details),
            timestamp,
        })
    }

    #[test]
    fn queue_snapshot_takes_the_newest_notice_per_entry_bounded_and_ordered() {
        let mut messages = vec![
            notice("q1", "pending", 1, 1000),
            notice("q1", "armed", 1, 2000),
            notice("q2", "pending", 2, 3000),
        ];
        for index in 0..24i64 {
            messages.push(notice(&format!("bulk{index}"), "pending", 100 + index as u64, 4000 + index));
        }
        let snapshot = from_queue_notices(&messages, "sender", "g1");
        // The display list is bounded, but nothing is hidden: the full
        // count survives in pending_total and the remainder in omitted.
        assert_eq!(snapshot.entries.len(), MAX_QUEUE_ENTRIES);
        assert_eq!(snapshot.pending_total, 26);
        assert_eq!(snapshot.omitted, 18);
        // q1's newest notice (armed) wins over its older pending notice.
        let first = &snapshot.entries[0];
        assert_eq!((first.queue_id.as_str(), first.state.as_str(), first.position), ("q1", "armed", 1));
        // Ordered by position, then id; bulk entries are the lowest of the rest.
        assert!(snapshot.entries.windows(2).all(|pair| {
            (pair[0].position, pair[0].queue_id.as_str()) < (pair[1].position, pair[1].queue_id.as_str())
        }));
        assert_eq!(snapshot.generation, "g1");
    }

    #[test]
    fn queue_snapshot_is_content_free_and_skips_foreign_or_malformed_records() {
        // A record that names a different session is skipped even though it
        // sits in this transcript; an absent sessionId (the contract) is fine.
        let foreign = with_details(json!({
            "sessionId": "other-session", "queueId": "q9", "position": 1,
            "receiver": "child:receiver-a", "state": "pending", "attempts": 0,
        }), 1000);
        let mut not_a_notice = notice("q8", "pending", 2, 1000);
        if let AgentMessage::Custom(CustomAgentMessage::Custom { custom_type, .. }) = &mut not_a_notice {
            *custom_type = AGENT_MESSAGE_CUSTOM_TYPE.to_string();
        }
        let missing_fields = with_details(json!({
            "queueId": "q7", "receiver": "child:receiver-a", "state": "pending", "attempts": 0,
        }), 1000);
        let long_id = "x".repeat(129);
        let oversized = notice(&long_id, "pending", 4, 1000);
        // No message id yet (armed before publish) and no sessionId: still
        // projects, with an empty message id.
        let unminted = with_details(json!({
            "queueId": "q6", "position": 5, "receiver": "child:receiver-a",
            "state": "armed", "attempts": 1,
        }), 1000);
        // The fallback transport verdict rides the notice and must surface;
        // the opaque dispatchToken is tolerated but never displayed.
        let fallback_in_flight = with_details(json!({
            "queueId": "q5", "position": 7, "receiver": "sibling:worker-d",
            "state": "fallback_pending", "attempts": 2,
            "ordinary_delivery": "queued", "dispatchToken": "opaque-token",
            "reason": "manual_reemit",
        }), 1500);
        // Free-text and unknown reasons never surface: whitelist only.
        let free_text_reason = with_details(json!({
            "queueId": "q10", "position": 8, "receiver": "child:receiver-a",
            "state": "failed", "attempts": 3, "reason": "operator note about disk full",
        }), 1600);
        // The internal fallback prompt custom record is tolerated, unread.
        let fallback_prompt = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "occ3_fallback_prompt".to_string(),
            content: CustomMessageContent::Text("[requires explicit manual resolution (re-emit or cancel); no automatic retry or duplicate will occur]".to_string()),
            display: false,
            details: Some(json!({"queueId": "q5"})),
            timestamp: 1700,
        });
        let messages = vec![foreign, not_a_notice, missing_fields, oversized, unminted, fallback_in_flight, free_text_reason, fallback_prompt, notice("q3", "admitted", 6, 1000)];
        let snapshot = from_queue_notices(&messages, "sender", "g");
        assert_eq!(snapshot.entries.len(), 4);
        // Ascending by position: q6 (5), q3 (6), q5 (7), q10 (8).
        assert_eq!(snapshot.entries[0].queue_id, "q6");
        assert_eq!(snapshot.entries[0].message_id, "");
        assert_eq!(snapshot.entries[1].queue_id, "q3");
        assert_eq!(snapshot.entries[1].message_id, "msg-q3");
        let fallback = &snapshot.entries[2];
        assert_eq!(fallback.queue_id, "q5");
        assert_eq!(fallback.state, "fallback_pending");
        // The transport verdict surfaces; invalid verdicts are dropped,
        // the whitelisted manual-resolution reason surfaces, and the
        // dispatchToken never serializes.
        assert_eq!(fallback.ordinary_delivery.as_deref(), Some("queued"));
        assert_eq!(fallback.reason.as_deref(), Some("manual_reemit"));
        // Free-text reasons are dropped entirely (content-free whitelist).
        let cancelled = &snapshot.entries[3];
        assert_eq!(cancelled.queue_id, "q10");
        assert_eq!(cancelled.state, "failed");
        assert_eq!(cancelled.reason, None);
        let serialized = serde_json::to_string(&snapshot).unwrap();
        assert!(!serialized.contains("private handoff body"));
        assert!(!serialized.contains("operator note"));
        assert!(!serialized.contains("disk full"));
        assert!(!serialized.contains("dispatchToken"));
        assert!(!serialized.contains("opaque-token"));
        assert!(!serialized.contains("manual resolution"));
        assert!(from_queue_notices(&[], "sender", "g").entries.is_empty());
    }

    #[test]
    fn display_state_maps_the_sender_labels_and_passes_unknown_through() {
        // The sender queue's requested UI labels, corrected so users can
        // distinguish unfinished re-emission from accepted delivery and
        // never read a transport acknowledgment as consumption.
        assert_eq!(display_state("pending"), "queued");
        assert_eq!(display_state("armed"), "armed");
        assert_eq!(display_state("published"), "in-flight");
        // Transport acknowledgment only: honest, not consumed.
        assert_eq!(display_state("admitted"), "acknowledged");
        // Unfinished ordinary re-emission stays visibly pending.
        assert_eq!(display_state("fallback_pending"), "fallback (pending)");
        // Terminal, acknowledged ordinary delivery.
        assert_eq!(display_state("fallback"), "fallback");
        assert_eq!(display_state("completed"), "completed");
        assert_eq!(display_state("expired"), "expired");
        assert_eq!(display_state("failed"), "failed");
        // The sender's frozen final name for possibly-consumed publications
        // awaiting receipt reconciliation; earlier provisional spellings
        // stay mapped for old records.
        assert_eq!(display_state("uncertain"), "uncertain (pending receipt)");
        for state in ["reconciling", "blocked"] {
            assert_eq!(display_state(state), "uncertain (pending receipt)", "{state}");
            assert!(!is_terminal_state(state), "{state}");
        }
        assert!(!is_terminal_state("uncertain"));
        // Unknown states pass through unchanged, never relabeled or hidden.
        assert_eq!(display_state("some-future-state"), "some-future-state");
    }

    #[test]
    fn terminal_states_are_excluded_from_pending_counts_but_kept_for_reporting() {
        let snapshot = from_queue_notices(
            &[
                notice("q1", "pending", 1, 1000),
                notice("q2", "fallback", 2, 2000),
                notice("q3", "failed", 3, 3000),
                notice("q4", "completed", 4, 4000),
                notice("q5", "expired", 5, 5000),
                // The intermediate fallback state is ordinary delivery in
                // flight: still pending work.
                notice("q6", "fallback_pending", 6, 6000),
                notice("q7", "future-state", 7, 7000),
            ],
            "sender",
            "g",
        );
        assert_eq!(snapshot.entries.len(), 7);
        assert_eq!(snapshot.pending_total, 3);
        let pending: Vec<_> = snapshot.pending_entries().map(|entry| entry.queue_id.as_str()).collect();
        assert_eq!(pending, vec!["q1", "q6", "q7"]);
        for state in ["fallback", "failed", "completed", "expired"] {
            assert!(is_terminal_state(state), "{state}");
        }
        for state in ["pending", "armed", "published", "admitted", "fallback_pending", "unknown"] {
            assert!(!is_terminal_state(state), "{state}");
        }
    }

    #[test]
    fn snapshot_notice_is_the_authoritative_base_and_newer_notices_override() {
        // Oldest: a full snapshot with three entries at their then-positions.
        let snapshot_notice = with_details(json!({
            "snapshot": true,
            "entries": [
                { "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
                  "state": "pending", "attempts": 0, "timestamp": 1000 },
                { "queueId": "q2", "position": 2, "receiver": "child:receiver-b",
                  "state": "armed", "attempts": 1, "messageId": "m-2", "timestamp": 1000 },
                { "queueId": "q3", "position": 3, "receiver": "sibling:worker-c",
                  "state": "published", "attempts": 0, "timestamp": 1000 }
            ]
        }), 1000);
        // Newer: q1 advances and shifts position; q3 terminates; q4 is new.
        let q1_advanced = with_details(json!({
            "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
            "state": "admitted", "attempts": 0, "messageId": "m-1",
        }), 2000);
        let q3_terminal = with_details(json!({
            "queueId": "q3", "position": 2, "receiver": "sibling:worker-c",
            "state": "fallback", "attempts": 0,
        }), 3000);
        let q4_new = with_details(json!({
            "queueId": "q4", "position": 3, "receiver": "child:receiver-a",
            "state": "pending", "attempts": 0,
        }), 4000);
        // Older than the snapshot and contradicting it: must be ignored,
        // which also proves the scan stops at the snapshot.
        let stale_q1 = with_details(json!({
            "queueId": "q1", "position": 9, "receiver": "child:receiver-a",
            "state": "failed", "attempts": 9,
        }), 500);
        let messages = vec![stale_q1, snapshot_notice, q1_advanced, q3_terminal, q4_new];
        let snapshot = from_queue_notices(&messages, "sender", "g");
        assert_eq!(snapshot.entries.len(), 4);
        assert_eq!(snapshot.pending_total, 3);
        // Ascending by the CURRENT positions: q1 (1), q3 (2), q4 (3), q2 (still 2? no: q2 kept its snapshot position 2).
        // q3 overrode to position 2 and q2 stayed at 2; the tie breaks by id: q2 < q3.
        assert_eq!(snapshot.entries[0].queue_id, "q1");
        assert_eq!(snapshot.entries[0].state, "admitted");
        assert_eq!(snapshot.entries[0].message_id, "m-1");
        assert_eq!(snapshot.entries[1].queue_id, "q2");
        assert_eq!(snapshot.entries[2].queue_id, "q3");
        assert_eq!(snapshot.entries[2].state, "fallback");
        assert_eq!(snapshot.entries[3].queue_id, "q4");
        // The stale pre-snapshot record never resurfaced.
        assert!(snapshot.entries.iter().all(|entry| entry.position != 9));
    }

    #[test]
    fn individual_notice_history_stays_reconstructible_within_the_scan_bound() {
        // A snapshot at the far end of a long individual-notice history:
        // the scan bound must still reach it, and the snapshot plus the
        // newer individuals must rebuild the active queue even when the
        // history between them is far deeper than the old 64-record bound.
        let mut messages = Vec::new();
        for index in 0..200i64 {
            messages.push(notice(&format!("popped{index}"), "completed", index as u64 + 1, 1000 + index));
        }
        messages.push(with_details(json!({
            "snapshot": true,
            "entries": [
                { "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
                  "state": "pending", "attempts": 0 },
                { "queueId": "q2", "position": 2, "receiver": "child:receiver-b",
                  "state": "admitted", "attempts": 2 }
            ]
        }), 2000));
        let q1_new = with_details(json!({
            "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
            "state": "armed", "attempts": 1,
        }), 3000);
        messages.push(q1_new);
        let snapshot = from_queue_notices(&messages, "sender", "g");
        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(snapshot.pending_total, 2);
        assert_eq!(snapshot.entries[0].queue_id, "q1");
        assert_eq!(snapshot.entries[0].state, "armed");
        assert_eq!(snapshot.entries[1].queue_id, "q2");
        assert_eq!(snapshot.entries[1].state, "admitted");

        // Without any snapshot record, the same history depth must still
        // reconstruct from individual notices alone: an active entry whose
        // newest notice sits 100 terminal records deep stays visible within
        // the scan bound (the pre-correction 64-record bound lost it).
        // The terminal history uses one-shot queue ids, exactly the shape
        // a long-running session accumulates: every finished handoff
        // leaves one terminal notice behind.
        let mut individual_only = Vec::new();
        individual_only.push(notice("q-live", "pending", 1, 100));
        for index in 0..100i64 {
            individual_only.push(notice(&format!("done{index}"), "completed", index as u64 + 2, 200 + index));
        }
        individual_only.push(notice("q-live2", "armed", 2, 5000));
        let snapshot = from_queue_notices(&individual_only, "sender", "g");
        // Both live entries survive 100 terminal records of history, and
        // terminal history fills only the leftover display slots: it can
        // never crowd live work out of the bounded list, and dropped
        // terminal history never counts as omitted queued work.
        assert_eq!(snapshot.pending_total, 2);
        assert_eq!(snapshot.omitted, 0);
        let live = snapshot.entries.iter().find(|entry| entry.queue_id == "q-live").expect("deep active entry survives");
        assert_eq!(live.state, "pending");
        let live2 = snapshot.entries.iter().find(|entry| entry.queue_id == "q-live2").expect("newest active entry present");
        assert_eq!(live2.state, "armed");
        // The list is full, and the terminal slots report the NEWEST
        // outcomes rather than the oldest positions.
        assert_eq!(snapshot.entries.len(), MAX_QUEUE_ENTRIES);
        let reported: Vec<&str> = snapshot
            .entries
            .iter()
            .filter(|entry| is_terminal_state(&entry.state))
            .map(|entry| entry.queue_id.as_str())
            .collect();
        assert_eq!(reported, vec!["done94", "done95", "done96", "done97", "done98", "done99"]);
    }

    #[test]
    fn refusal_notices_never_project_as_entries_or_pending_work() {
        // The sender's queue_refusal_notice shape, exactly as its details()
        // emits it: empty queue id, state "refused", whitelisted reason in
        // both `reason` and `refused`. It must surface as an event, never
        // as an entry or pending work.
        let refusal = with_details(json!({
            "queueId": "", "receiver": "unknown", "state": "refused",
            "attempts": 0, "position": 0, "timestamp": 5000,
            "reason": "queue_full", "refused": "queue_full",
        }), 5000);
        let snapshot = from_queue_notices(&[refusal], "sender", "g");
        assert!(snapshot.entries.is_empty());
        assert_eq!(snapshot.pending_total, 0);
        let refused = snapshot.refused.expect("refusal surfaces as an event");
        assert_eq!(refused.reason, "queue_full");
        assert_eq!(refused.timestamp, 5000);
        // A non-whitelisted refused value never surfaces, and the badge
        // stays honest: a refusal is never queued work.
        let bogus = with_details(json!({
            "queueId": "", "receiver": "unknown", "state": "refused",
            "position": 0, "refused": "disk on fire",
        }), 5000);
        let bogus_projection = from_queue_notices(&[bogus], "sender", "g");
        assert!(bogus_projection.refused.is_none());
        assert_eq!(bogus_projection.pending_total, 0);
    }

    // A fresh queue_full refusal record, exactly as the sender's
    // queue_refusal_notice emits it. Each call mints a new record so no
    // test moves a value it later reuses. Not a test: a fixture builder.
    fn queue_full_refusal() -> AgentMessage {
        with_details(json!({
            "queueId": "", "receiver": "unknown", "state": "refused",
            "position": 0, "refused": "queue_full",
        }), 1000)
    }

    #[test]
    fn refusal_clears_when_newer_activity_proves_admission_resumed() {
        // Refusal, then a successful enqueue recorded after it: the newer
        // entry-bearing notice proves the queue admitted work again, so
        // the banner clears instead of claiming a stale refusal.
        let accepted = notice("q-after", "pending", 1, 2000);
        let snapshot = from_queue_notices(&[queue_full_refusal(), accepted], "sender", "g");
        assert!(snapshot.refused.is_none(), "newer admission clears the refusal");
        assert_eq!(snapshot.pending_total, 1);

        // Refusal newer than every entry-bearing record stays visible.
        let older = notice("q-before", "completed", 1, 500);
        let snapshot = from_queue_notices(&[older, queue_full_refusal()], "sender", "g");
        let refused = snapshot.refused.expect("newest event is the refusal");
        assert_eq!(refused.reason, "queue_full");

        // Newest refusal wins when several are in the window: the OLDER
        // queue_blocked refusal must not overwrite the newer queue_full
        // one (the scan is newest-first).
        let older_refusal = with_details(json!({
            "queueId": "", "receiver": "unknown", "state": "refused",
            "position": 0, "refused": "queue_blocked",
        }), 900);
        let snapshot = from_queue_notices(&[older_refusal, queue_full_refusal()], "sender", "g");
        let refused = snapshot.refused.expect("a refusal is still the newest event");
        assert_eq!(refused.reason, "queue_full", "the newest refusal wins");
        assert_eq!(refused.timestamp, 1000);

        // A snapshot alone (the drain emits one after a refusal too)
        // carries only pre-refusal entries: it does NOT clear the banner.
        let snapshot_notice = with_details(json!({
            "snapshot": true,
            "entries": [
                { "queueId": "q-old", "position": 1, "receiver": "child:receiver-a",
                  "state": "pending", "attempts": 0, "timestamp": 400 }
            ]
        }), 1500);
        let snapshot = from_queue_notices(&[snapshot_notice, queue_full_refusal()], "sender", "g");
        let refused = snapshot.refused.expect("a snapshot with only pre-refusal entries does not clear the banner");
        assert_eq!(refused.reason, "queue_full");

        // A snapshot whose entries were enqueued AFTER the refusal does
        // clear it: those admissions prove the queue resumed.
        let recovered = with_details(json!({
            "snapshot": true,
            "entries": [
                { "queueId": "q-new", "position": 1, "receiver": "child:receiver-a",
                  "state": "pending", "attempts": 0, "timestamp": 3000 }
            ]
        }), 3000);
        let snapshot = from_queue_notices(&[recovered, queue_full_refusal()], "sender", "g");
        assert!(snapshot.refused.is_none(), "post-refusal enrollment clears the banner");
    }

    // Helper mirroring the sender's snapshot_details shape for reuse.
    fn overflow_snapshot_from(timestamp: i64) -> AgentMessage {
        with_details(json!({
            "snapshot": true, "noticeOverflow": true,
            "entries": [
                { "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
                  "state": "pending", "attempts": 0 }
            ]
        }), timestamp)
    }

    #[test]
    fn notice_overflow_flag_rides_the_snapshot_and_clears_on_newer_snapshots() {
        // The sender's snapshot_details sets noticeOverflow only when the
        // durable outbox is over its bound; the newest snapshot wins.
        let snapshot = from_queue_notices(&[overflow_snapshot_from(1000)], "sender", "g");
        assert!(snapshot.notice_overflow);
        assert_eq!(snapshot.entries.len(), 1);

        // A newer snapshot without the flag clears it.
        let drained_snapshot = with_details(json!({
            "snapshot": true,
            "entries": [
                { "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
                  "state": "completed", "attempts": 0 }
            ]
        }), 2000);
        let snapshot = from_queue_notices(&[overflow_snapshot_from(1000), drained_snapshot], "sender", "g");
        assert!(!snapshot.notice_overflow);

        // No snapshot at all: no overflow to report.
        let individual = notice("q1", "pending", 1, 1000);
        assert!(!from_queue_notices(&[individual], "sender", "g").notice_overflow);
    }

    #[test]
    fn whole_queue_blocked_state_surfaces_from_the_snapshot_notice() {
        // The sender's snapshot_details sets `blocked` when the persisted
        // queue file could not be loaded safely; only whitelisted values
        // surface, and the entries still project.
        let blocked_snapshot = with_details(json!({
            "snapshot": true, "blocked": "unreadable_io",
            "entries": [
                { "queueId": "q1", "position": 1, "receiver": "child:receiver-a",
                  "state": "pending", "attempts": 0 }
            ]
        }), 1000);
        let snapshot = from_queue_notices(&[blocked_snapshot], "sender", "g");
        assert_eq!(snapshot.blocked.as_deref(), Some("unreadable_io"));
        assert_eq!(snapshot.entries.len(), 1);
        // A non-whitelisted blocked value never surfaces.
        let bogus = with_details(json!({
            "snapshot": true, "blocked": "cosmic rays",
            "entries": []
        }), 1000);
        assert!(from_queue_notices(&[bogus], "sender", "g").blocked.is_none());
        // Without any snapshot record there is no blocked state to report.
        let individual = notice("q1", "pending", 1, 1000);
        assert!(from_queue_notices(&[individual], "sender", "g").blocked.is_none());
    }
}
