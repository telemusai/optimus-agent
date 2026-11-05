//! Per-session sender FIFO queue for experimental native OCC-3 handoffs.
//!
//! One live sender session may queue multiple native handoffs. The source
//! capture slot stays one-shot and single-armed (see [`crate::core::agent_session::occ3_state`]);
//! this queue decides WHICH enqueued handoff the next arm belongs to, keeps
//! stable ids, persists a bounded projection for restart recovery, and drives
//! the explicit fresh-source advance prompt the model must answer between
//! handoffs. It never captures, publishes or admits on its own: every wire
//! action still goes through the handoff tool and the authenticated family
//! controller.
//!
//! Safety invariants:
//! * only the queue head may arm; an armed capture is never retargeted;
//! * no raw handles, receipts or digests are persisted - ids, selectors,
//!   counters and bounded reason strings only;
//! * an unreadable/unknown-schema file NEVER becomes "empty": the file is
//!   preserved byte-for-byte, a terminal notice is emitted, and the queue
//!   opens blocked until the model starts a fresh queue;
//! * queued entries never consume the source-capture retry budget: only
//!   real capture failures may increment `failed_attempts`;
//! * a fallback entry is only marked delivered after the ordinary OCC-1
//!   send of that entry is acknowledged.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const OCC3_SENDER_QUEUE_SCHEMA: &str = "optimus.occ3-sender-queue.v1";
/// The custom type of the PARENT-side record for a child's sender-queue
/// notice: distinct from [`OCC3_QUEUE_NOTICE_CUSTOM_TYPE`] so a child
/// notice written into the parent's transcript never projects onto the
/// parent's own queue view. Carries the content-free notice details plus
/// `childSessionId` for attribution, and the `noticeId` for replay dedup.
pub const OCC3_CHILD_QUEUE_NOTICE_CUSTOM_TYPE: &str = "occ3_child_queue_notice";
/// Default maximum queued entries per sender session. Configurable via
/// `OCC3_SENDER_QUEUE_MAX_ENTRIES` (validated, clamped
/// [`OCC3_SENDER_QUEUE_MIN_ENTRIES`]..=[`OCC3_SENDER_QUEUE_MAX_ENTRIES_LIMIT`]).
pub const OCC3_SENDER_QUEUE_MAX_ENTRIES: usize = 8;
pub const OCC3_SENDER_QUEUE_MIN_ENTRIES: usize = 1;
pub const OCC3_SENDER_QUEUE_MAX_ENTRIES_LIMIT: usize = 64;
/// Default entry time-to-live (seconds). Configurable via
/// `OCC3_SENDER_QUEUE_TTL_SECONDS` (clamped to [`OCC3_SENDER_QUEUE_MIN_TTL_SECS`]..=[`OCC3_SENDER_QUEUE_MAX_TTL_SECS`]).
pub const OCC3_SENDER_QUEUE_TTL_SECS: u64 = 1800;
pub const OCC3_SENDER_QUEUE_MIN_TTL_SECS: u64 = 60;
pub const OCC3_SENDER_QUEUE_MAX_TTL_SECS: u64 = 24 * 3600;
/// The sender-transcript notice custom type. Distinct from `agent_message`
/// so the communication-activity projection never pulses on it.
pub const OCC3_QUEUE_NOTICE_CUSTOM_TYPE: &str = "occ3_queue_notice";

/// The sender-context fallback re-emission prompt custom type. The prompt
/// that drives the model's ordinary re-emission turn; its details carry the
/// queueId (already public in notices) but never the dispatch token - the
/// token stays private in host memory.
pub const OCC3_FALLBACK_PROMPT_CUSTOM_TYPE: &str = "occ3_fallback_prompt";

/// Blocking reason when a persisted queue could not be safely loaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "blocked_reason")]
pub enum Occ3SenderQueueBlocked {
    /// The file's schema is not one this build understands. The file was
    /// preserved untouched; a human or newer build must resolve it.
    UnsupportedSchema { found: String },
    /// The file exists but could not be parsed as its declared schema.
    Unreadable { error: String },
    /// The file could not be read at all (permissions, IO).
    UnreadableIo { error: String },
}

impl Occ3SenderQueueBlocked {
    pub fn as_str(&self) -> &'static str {
        match self {
            Occ3SenderQueueBlocked::UnsupportedSchema { .. } => "unsupported_schema",
            Occ3SenderQueueBlocked::Unreadable { .. } => "unreadable",
            Occ3SenderQueueBlocked::UnreadableIo { .. } => "unreadable_io",
        }
    }
}

/// Entry states. `published` means the connector publication succeeded and
/// the authenticated send returned; `admitted` means the receiver's inbox
/// accepted it (the sender tool's success condition).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Occ3SenderQueueState {
    /// Queued, not yet armed. Only a `pending` head may arm.
    Pending,
    /// The queue head's capture is armed (budget set, no lease yet).
    Armed,
    /// The captured turn's publish succeeded.
    Published,
    /// The receiver admitted the handoff.
    Admitted,
    /// Native path terminally failed for this entry; ordinary OCC-1
    /// re-emission is in flight but NOT yet acknowledged. The entry
    /// stays visible (pending-like) until the ordinary delivery is
    /// acknowledged or it expires.
    FallbackPending,
    /// Outcome cannot be proven (restart with an in-flight publication,
    /// or a consumed report without a receipt digest). Visible and
    /// explicit: never auto-recaptured, auto-fallen-back or TTL-expired;
    /// resolves only through the explicit host resolution path.
    Uncertain,
    /// Ordinary OCC-1 re-emission acknowledged (receipt
    /// delivered|queued). Terminal.
    Fallback,
    /// Receiver proved completed use of the native state. Terminal.
    Completed,
    /// Entry TTL expired without completing. Terminal.
    Expired,
    /// Terminal failure with no fallback (queue full is reported at
    /// enqueue instead).
    Failed,
}

impl Occ3SenderQueueState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Occ3SenderQueueState::Pending => "pending",
            Occ3SenderQueueState::Armed => "armed",
            Occ3SenderQueueState::Published => "published",
            Occ3SenderQueueState::Admitted => "admitted",
            Occ3SenderQueueState::FallbackPending => "fallback_pending",
            Occ3SenderQueueState::Uncertain => "uncertain",
            Occ3SenderQueueState::Fallback => "fallback",
            Occ3SenderQueueState::Completed => "completed",
            Occ3SenderQueueState::Expired => "expired",
            Occ3SenderQueueState::Failed => "failed",
        }
    }
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Occ3SenderQueueState::Fallback
                | Occ3SenderQueueState::Completed
                | Occ3SenderQueueState::Expired
                | Occ3SenderQueueState::Failed
        )
    }
    pub fn is_in_flight(&self) -> bool {
        matches!(
            self,
            Occ3SenderQueueState::Armed | Occ3SenderQueueState::Published | Occ3SenderQueueState::Admitted
        )
    }
    /// Uncertain outcomes: visible, never auto-resolved, never TTL-expired.
    pub fn is_uncertain(&self) -> bool {
        matches!(self, Occ3SenderQueueState::Uncertain)
    }
}

/// One queued handoff. Stable from enqueue to terminal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Occ3SenderQueueEntry {
    /// `occ3q_<uuid>`; minted at enqueue, never changes.
    pub queue_id: String,
    /// The real agent-message id, present only once the daemon minted it at
    /// the authorized handoff (`agentmsg_<uuid>`). Absent for entries that
    /// never reached the authorize/publish step. The CURRENT attempt's id;
    /// prior attempts stay in [`Self::message_ids`] (never overwritten).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Attempt history: `(attempt_label, message_id)` pairs, appended per
    /// attempt, never overwritten. Bounded at 8 by the loader; `restart`
    /// attempts record the id the restart made ambiguous.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub message_ids: Vec<(String, String)>,
    pub receiver_role: String,
    pub receiver_name: String,
    pub enqueued_at_epoch: u64,
    pub expires_at_epoch: u64,
    pub state: Occ3SenderQueueState,
    /// Capture failures and restart demotions observed for this entry.
    pub attempts: u32,
    /// Bounded, content-free reason for the current terminal or fallback
    /// state (max 128 chars, no handles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Trusted host-side dispatch token issued when this entry entered
    /// `fallback_pending` and returned with the ordinary send receipt.
    /// Correlation is by TOKEN, never by receiver identity alone, so an
    /// unrelated ordinary message to the same receiver cannot falsely
    /// acknowledge this entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_dispatch_token: Option<String>,
    /// The AUTHENTICATED receiver session id captured at arm time from the
    /// authorized native handoff (`authorized.target_session_id()`), never
    /// model-supplied. When present, the fallback dispatch token is granted
    /// only for a send target that resolves to this id OR the entry's own
    /// persisted receiver selector - so a different agent that merely shares
    /// a display name cannot close the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receiver_session_id: Option<String>,
    /// Receiver-busy re-captures already scheduled for this entry. Each one
    /// follows a pre-event busy refusal (proven non-use of the published
    /// capture). Bounded by [`OCC3_RECEIVER_BUSY_MAX_RETRIES`]; separate from
    /// `attempts` and from the source-capture retry budget, because a busy
    /// receiver is not a capture failure. Absent on older artifacts -> 0.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub busy_retries: u32,
    /// Earliest epoch second at which the next busy re-capture may arm (the
    /// backoff). Set only on a `pending` entry returned by a busy refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_not_before_epoch: Option<u64>,
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

/// Maximum receiver-busy re-captures per queue entry before the entry falls
/// back to ordinary OCC-1 re-emission.
pub const OCC3_RECEIVER_BUSY_MAX_RETRIES: u32 = 3;

/// Queue reason of a pending head whose armed SOURCE capture the server
/// refused as busy before admission (nothing captured, nothing published).
pub const OCC3_SOURCE_BUSY_RETRY_REASON: &str = "source_busy_retry";
/// Upper bound on any busy re-capture wait the handoff tool honours, in
/// seconds. Matches the bounded server `Retry-After` the client accepts.
pub const OCC3_BUSY_BACKOFF_MAX_SECS: u64 = 30;

/// Backoff before receiver-busy re-capture `retry` (1-based): 2s, 4s, 8s.
/// Matches the session provider retry policy's base delays.
pub fn receiver_busy_backoff_secs(retry: u32) -> u64 {
    2u64.saturating_pow(retry.clamp(1, OCC3_RECEIVER_BUSY_MAX_RETRIES))
}

impl Occ3SenderQueueEntry {
    pub fn receiver_selector(&self) -> String {
        format!("{}:{}", self.receiver_role, self.receiver_name)
    }
    fn expired(&self, now_epoch: u64) -> bool {
        // Uncertain work never TTL-expires: an uncertain entry may hold a
        // possibly-consumed publication whose outcome must be resolved
        // explicitly, never aged into an automatic fallback.
        !self.state.is_in_flight() && !self.state.is_uncertain() && self.expires_at_epoch <= now_epoch
    }
}

/// Load outcome. `Blocked` never discards the file.
#[derive(Debug)]
pub enum Occ3SenderQueueLoad {
    /// No file: a fresh empty queue.
    Fresh,
    /// The surviving entries plus one terminal `expired` notice per entry
    /// that expired on disk: restart expiry is never a silent drop. The
    /// caller journals each notice so the sender and the parent-side
    /// roster learn of the loss on resume, exactly like a live TTL sweep
    /// would have reported it.
    Loaded(VecDeque<Occ3SenderQueueEntry>, Vec<Occ3QueueNotice>),
    /// File exists but cannot be used. It has NOT been modified.
    Blocked(Occ3SenderQueueBlocked),
}

#[derive(Debug, Clone, Copy)]
enum Ack {
    Transcript,
    Parent,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersistedQueue {
    schema: String,
    entries: Vec<Occ3SenderQueueEntry>,
    /// Durable notice outbox (schema v1 addition, backward compatible:
    /// absent on old artifacts -> empty). Persisted atomically with the
    /// entries in this same file, so a queue transition and its notices
    /// can never be split across two files' crash windows.
    #[serde(default)]
    outbox: Vec<Occ3DurableQueueNotice>,
    /// Set (and persisted) when a notice had to be refused a outbox slot
    /// because the bound was hit: a visible fail state, never a silent
    /// drop. Cleared when the outbox drains below the bound.
    #[serde(default)]
    outbox_overflow: bool,
}

/// Upper bound on durable outbox records. Bounded so a stuck delivery
/// path cannot grow the artifact without limit; hitting the bound flips
/// the persisted `outbox_overflow` flag (surfaced in
/// [`Occ3SenderQueue::snapshot_details`]) instead of silently dropping
/// the notice.
pub const OCC3_SENDER_QUEUE_OUTBOX_CAP: usize = 64;

/// One durable outbox record: the content-free notice plus its stable id
/// and per-target delivery flags. The id is minted once when the notice
/// is recorded and never changes, so consumers dedup replays; the record
/// is removed only after BOTH deliveries are acknowledged (durable
/// transcript write, parent notification).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Occ3DurableQueueNotice {
    /// Stable `occ3n_<uuid>`, minted at record time.
    pub notice_id: String,
    pub queue_id: String,
    /// The real daemon-minted message id when one exists for the entry.
    pub message_id: Option<String>,
    pub receiver: String,
    pub state: String,
    pub attempts: u32,
    /// 1-based position the entry held at emission time.
    pub position: u32,
    pub reason: Option<String>,
    /// Whitelisted queue-level refusal value, present only on refusal
    /// records (see [`queue_refusal_notice`]).
    pub refusal: Option<String>,
    pub timestamp: u64,
    /// The sender-transcript record was durably written and flushed.
    pub transcript_done: bool,
    /// The parent notification was accepted (or no parent surface
    /// exists for this session).
    pub parent_done: bool,
}

impl Occ3DurableQueueNotice {
    /// The `details` payload of the `occ3_queue_notice` custom message:
    /// the same content-free shape an in-flight notice serializes, plus
    /// the stable `noticeId` for consumer-side dedup. Never a dispatch
    /// token, handle or receipt.
    pub fn details(&self) -> Value {
        let mut details = json!({
            "noticeId": self.notice_id,
            "queueId": self.queue_id,
            "receiver": self.receiver,
            "state": self.state,
            "attempts": self.attempts,
            "position": self.position,
            "timestamp": self.timestamp,
        });
        if let Some(message_id) = self.message_id.as_deref() {
            details["messageId"] = json!(message_id);
        }
        if let Some(reason) = self.reason.as_deref() {
            details["reason"] = json!(reason);
        }
        if let Some(refused) = self.refusal.as_deref() {
            details["refused"] = json!(refused);
        }
        details
    }

    /// The human transcript line, identical in shape to the in-flight
    /// notice rendering.
    pub fn transcript_line(&self) -> String {
        let uncertain_note = if self.state == "uncertain" {
            " [requires explicit manual resolution (re-emit or cancel); no automatic retry or duplicate will occur]"
        } else {
            ""
        };
        format!(
            "OCC-3 sender queue: {} handoff {} for receiver {}{}.{}",
            self.state,
            self.queue_id,
            self.receiver,
            self.reason
                .as_deref()
                .map(|reason| format!(" ({reason})"))
                .unwrap_or_default(),
            uncertain_note
        )
    }
}

/// The in-memory queue. All mutating methods return the notice(s) to emit;
/// the host persists the file (`persist`) and appends the notices.
#[derive(Debug)]
pub struct Occ3SenderQueue {
    entries: VecDeque<Occ3SenderQueueEntry>,
    blocked: Option<Occ3SenderQueueBlocked>,
    ttl_secs: u64,
    max_entries: usize,
    /// Durable notice outbox: notices not yet acknowledged by BOTH
    /// delivery targets. Persisted atomically with the entries.
    outbox: Vec<Occ3DurableQueueNotice>,
    /// Visible fail state when the outbox bound was hit (never a silent
    /// drop); surfaced in `snapshot_details`.
    outbox_overflow: bool,
}

/// One sender-transcript notice, content-free.
#[derive(Debug, Clone)]
pub struct Occ3QueueNotice {
    pub queue_id: String,
    pub message_id: Option<String>,
    pub receiver: String,
    pub state: &'static str,
    pub attempts: u32,
    /// 1-based position from the queue head at emission time. Terminal
    /// entries emit the position they held.
    pub position: u32,
    pub timestamp: u64,
    pub reason: Option<String>,
    /// Trusted host-side dispatch token for ordinary-send acknowledgment
    /// correlation (present only on `fallback_pending` notices): the host
    /// returns it with the send receipt. Random, never derived from
    /// receiver identity or message content.
    pub dispatch_token: Option<String>,
    /// Queue-level refusal marker, present only on refusal notices built
    /// by [`queue_refusal_notice`]: a whitelisted `queue_full` /
    /// `queue_blocked` value. Emitted in `details()` as `refused` for the
    /// parent-side visibility projection; never set on entry-level
    /// notices.
    pub refusal: Option<String>,
}

impl Occ3QueueNotice {
    /// The `details` object of the `occ3_queue_notice` custom message.
    /// Bounded strings/ints only; unknown-field tolerant on read.
    pub fn details(&self) -> Value {
        let mut details = json!({
            "queueId": self.queue_id,
            "receiver": self.receiver,
            "state": self.state,
            "attempts": self.attempts,
            "position": self.position,
            "timestamp": self.timestamp,
        });
        if let Some(message_id) = self.message_id.as_deref() {
            details["messageId"] = json!(message_id);
        }
        if let Some(reason) = self.reason.as_deref() {
            details["reason"] = json!(reason);
        }
        if let Some(refused) = self.refusal.as_deref() {
            details["refused"] = json!(refused);
        }
        // The dispatch token is PRIVATE host bookkeeping: it lives only in
        // the in-memory notice struct and the persisted queue artifact, and
        // is never serialized into a notice, the transcript, or the UI.
        details
    }
}

impl Occ3SenderQueue {
    pub fn new(ttl_secs: u64) -> Self {
        Self::with_limits(ttl_secs, env_max_entries())
    }

    /// Fully explicit constructor (tests): TTL and the entry ceiling.
    pub fn with_limits(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            blocked: None,
            ttl_secs: ttl_secs.clamp(OCC3_SENDER_QUEUE_MIN_TTL_SECS, OCC3_SENDER_QUEUE_MAX_TTL_SECS),
            max_entries: max_entries.clamp(
                OCC3_SENDER_QUEUE_MIN_ENTRIES,
                OCC3_SENDER_QUEUE_MAX_ENTRIES_LIMIT,
            ),
            outbox: Vec::new(),
            outbox_overflow: false,
        }
    }

    /// Records one notice into the durable outbox. Bounded: when the cap
    /// is hit the record is refused a slot and the persisted overflow
    /// flag flips instead of a silent drop. Returns the stored record
    /// (with its freshly minted stable id) when accepted.
    pub fn outbox_record(
        &mut self,
        notice: &Occ3QueueNotice,
    ) -> Option<Occ3DurableQueueNotice> {
        if self.outbox.len() >= OCC3_SENDER_QUEUE_OUTBOX_CAP {
            self.outbox_overflow = true;
            return None;
        }
        let record = Occ3DurableQueueNotice {
            notice_id: format!("occ3n_{}", uuid::Uuid::new_v4()),
            queue_id: notice.queue_id.clone(),
            message_id: notice.message_id.clone(),
            receiver: notice.receiver.clone(),
            state: notice.state.to_string(),
            attempts: notice.attempts,
            position: notice.position,
            reason: notice.reason.clone(),
            refusal: notice.refusal.clone(),
            timestamp: notice.timestamp,
            transcript_done: false,
            parent_done: false,
        };
        self.outbox.push(record.clone());
        Some(record)
    }

    /// Marks the sender-transcript delivery of one outbox record
    /// acknowledged (durable write happened). Removes the record when
    /// both deliveries are done. Returns the remaining record when it is
    /// still pending, `None` when unknown or fully delivered.
    pub fn outbox_ack_transcript(&mut self, notice_id: &str) -> Option<Occ3DurableQueueNotice> {
        self.outbox_ack(notice_id, Ack::Transcript)
    }

    /// Marks the parent notification of one outbox record acknowledged.
    pub fn outbox_ack_parent(&mut self, notice_id: &str) -> Option<Occ3DurableQueueNotice> {
        self.outbox_ack(notice_id, Ack::Parent)
    }

    fn outbox_ack(&mut self, notice_id: &str, ack: Ack) -> Option<Occ3DurableQueueNotice> {
        let index = self.outbox.iter().position(|record| record.notice_id == notice_id)?;
        match ack {
            Ack::Transcript => self.outbox[index].transcript_done = true,
            Ack::Parent => self.outbox[index].parent_done = true,
        }
        if self.outbox[index].transcript_done && self.outbox[index].parent_done {
            let _removed = self.outbox.remove(index);
            if self.outbox.len() < OCC3_SENDER_QUEUE_OUTBOX_CAP {
                self.outbox_overflow = false;
            }
            return None;
        }
        Some(self.outbox[index].clone())
    }

    /// Content-free outbox projection (pending durable notices with
    /// their delivery flags) for read-only visibility surfaces.
    pub fn outbox_snapshot(&self) -> Vec<Value> {
        self.outbox
            .iter()
            .map(|record| {
                let mut value = record.details();
                value["transcriptDone"] = json!(record.transcript_done);
                value["parentDone"] = json!(record.parent_done);
                value
            })
            .collect()
    }

    /// Outbox records not yet fully delivered (for restart re-drain).
    pub fn outbox_pending(&self) -> Vec<Occ3DurableQueueNotice> {
        self.outbox
            .iter()
            .filter(|record| !record.transcript_done || !record.parent_done)
            .cloned()
            .collect()
    }

    /// The persisted overflow flag (visible fail state).
    pub fn outbox_overflow(&self) -> bool {
        self.outbox_overflow
    }

    /// The effective entry ceiling (default 8; env-configurable).
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn from_configured() -> Self {
        Self::with_limits(env_ttl_secs(), env_max_entries())
    }

    pub fn blocked(&self) -> Option<&Occ3SenderQueueBlocked> {
        self.blocked.as_ref()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Immutable snapshot in arrival order.
    pub fn snapshot(&self) -> &[Occ3SenderQueueEntry] {
        self.entries.as_slices().0
    }

    /// Owned snapshot in arrival order (the deque may be split across its
    /// two slices; `as_slices().0` alone would truncate it).
    pub fn snapshot_vec(&self) -> Vec<Occ3SenderQueueEntry> {
        self.entries.iter().cloned().collect()
    }

    /// Loads a queue artifact. `Fresh` (file absent) opens an empty queue.
    /// `Blocked` preserves the file untouched for manual or future-build
    /// resolution; the caller must emit the blocked notice.
    pub fn load(path: &Path, ttl_secs: u64) -> (Self, Occ3SenderQueueLoad) {
        let mut queue = Self::new(ttl_secs);
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return (queue, Occ3SenderQueueLoad::Fresh);
            }
            Err(error) => {
                queue.blocked = Some(Occ3SenderQueueBlocked::UnreadableIo {
                    error: error.to_string(),
                });
                let blocked = queue.blocked.clone().unwrap();
                return (queue, Occ3SenderQueueLoad::Blocked(blocked));
            }
        };
        let parsed: Result<PersistedQueue, _> = serde_json::from_slice(&bytes);
        match parsed {
            Ok(persisted) if persisted.schema == OCC3_SENDER_QUEUE_SCHEMA => {
                // The outbox (and its overflow flag) rides the same
                // artifact as the entries: one atomic file, one crash
                // window. Absent on pre-outbox artifacts -> empty.
                let (persisted_outbox, persisted_overflow) =
                    (persisted.outbox, persisted.outbox_overflow);
                let mut entries: VecDeque<Occ3SenderQueueEntry> = persisted.entries.into();
                // Dropped ceiling guard: a file written by a bigger queue is
                // still honored (never silently truncated), but a corrupt
                // giant file cannot pass the entry bound below cheaply either.
                if entries.iter().any(|entry| {
                    entry.queue_id.len() > 128
                        || entry.receiver_role.is_empty()
                        || entry.receiver_name.is_empty()
                        || entry.receiver_name.len() > 128
                        || entry.attempts > 64
                        || entry.busy_retries > OCC3_RECEIVER_BUSY_MAX_RETRIES
                        || entry.message_ids.len() > 8
                }) {
                    queue.blocked = Some(Occ3SenderQueueBlocked::Unreadable {
                        error: "entry bound violation".into(),
                    });
                    let blocked = queue.blocked.clone().unwrap();
                    return (queue, Occ3SenderQueueLoad::Blocked(blocked));
                }
                // Restart never replays an in-flight capture: an armed,
                // published or admitted entry is demoted to fallback with
                // the model free to re-arm it inside the normal retry budget.
                let now = now_epoch();
                for entry in entries.iter_mut() {
                    if entry.state.is_in_flight() {
                        // Restart never replays and never blindly retries:
                        // the outcome is split by how far the handoff had
                        // progressed before the restart.
                        // * `armed` (no publication exists yet): a proven
                        //   pre-publication failure - a fresh explicit arm
                        //   within the remaining retry budget is safe.
                        // * `published`/`admitted` (a publication may have
                        //   been consumed): `uncertain`, visible and
                        //   explicit; never auto-recaptured, never
                        //   auto-fallen-back, never TTL-expired. Resolution
                        //   happens through the host receipt reconciliation
                        //   path only.
                        entry.attempts = entry.attempts.saturating_add(1);
                        entry.message_ids.push((
                            "restart_ambiguous".to_string(),
                            entry.message_id.clone().unwrap_or_default(),
                        ));
                        if entry.state == Occ3SenderQueueState::Armed {
                            entry.reason = Some("restart_pre_publication".into());
                            entry.state = if entry.attempts
                                > crate::core::agent_session::occ3_state::MAX_NATIVE_CAPTURE_RETRIES
                            {
                                Occ3SenderQueueState::FallbackPending
                            } else {
                                Occ3SenderQueueState::Pending
                            };
                        } else {
                            entry.reason = Some("restart_uncertain_publication".into());
                            entry.state = Occ3SenderQueueState::Uncertain;
                        }
                    }
                }
                // Expired entries at load are terminal expired: each one
                // is journaled as a terminal `expired` notice (mirroring
                // `sweep_expired`) BEFORE it is dropped, so restart expiry
                // is never a silent loss.
                let expired_notices: Vec<Occ3QueueNotice> = entries
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| {
                        entry.expired(now)
                            && matches!(
                                entry.state,
                                Occ3SenderQueueState::Pending
                                    | Occ3SenderQueueState::FallbackPending
                            )
                    })
                    .map(|(index, entry)| Occ3QueueNotice {
                        queue_id: entry.queue_id.clone(),
                        message_id: entry.message_id.clone(),
                        receiver: entry.receiver_selector(),
                        state: "expired",
                        attempts: entry.attempts,
                        position: (index + 1) as u32,
                        timestamp: now,
                        reason: Some("ttl_expired".into()),
                        dispatch_token: None,
                        refusal: None,
                    })
                    .collect();
                entries.retain(|entry| {
                    !(entry.expired(now)
                        && matches!(
                            entry.state,
                            Occ3SenderQueueState::Pending | Occ3SenderQueueState::FallbackPending
                        ))
                });
                queue.entries = entries;
                queue.outbox = persisted_outbox;
                queue.outbox_overflow = persisted_overflow;
                let loaded = VecDeque::clone(&queue.entries);
                (queue, Occ3SenderQueueLoad::Loaded(loaded, expired_notices))
            }
            Ok(persisted) => {
                queue.blocked = Some(Occ3SenderQueueBlocked::UnsupportedSchema {
                    found: persisted.schema,
                });
                let blocked = queue.blocked.clone().unwrap();
                (queue, Occ3SenderQueueLoad::Blocked(blocked))
            }
            Err(error) => {
                queue.blocked = Some(Occ3SenderQueueBlocked::Unreadable {
                    error: error.to_string(),
                });
                let blocked = queue.blocked.clone().unwrap();
                (queue, Occ3SenderQueueLoad::Blocked(blocked))
            }
        }
    }

    /// Atomically replaces the artifact (temp file + rename), mirroring the
    /// OCC session preference writer. Never called with a blocked queue.
    pub fn persist(&self, path: &Path) -> Result<(), String> {
        if self.blocked.is_some() {
            return Err("occ3 sender queue is blocked; refusing to overwrite its artifact".into());
        }
        let persisted = PersistedQueue {
            schema: OCC3_SENDER_QUEUE_SCHEMA.into(),
            entries: self.entries.iter().cloned().collect(),
            outbox: self.outbox.clone(),
            outbox_overflow: self.outbox_overflow,
        };
        let parent = path.parent().ok_or("Invalid occ3 sender queue path")?;
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let mut pending = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        serde_json::to_writer(pending.as_file_mut(), &persisted).map_err(|e| e.to_string())?;
        use std::io::Write;
        pending.write_all(b"\n").map_err(|e| e.to_string())?;
        pending.as_file().sync_all().map_err(|e| e.to_string())?;
        pending.persist(path).map_err(|e| e.to_string()).map(|_| ())
    }

    /// Enqueues a new handoff. Returns `None` (no state change) when the
    /// queue is blocked, full, or would exceed the TTL ceiling.
    pub fn enqueue(
        &mut self,
        receiver_role: &str,
        receiver_name: &str,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        if self.blocked.is_some() {
            return None;
        }
        if self.entries.len() >= self.max_entries {
            return None;
        }
        if receiver_role.is_empty() || receiver_name.is_empty() || receiver_name.len() > 128 {
            return None;
        }
        self.entries.push_back(Occ3SenderQueueEntry {
            queue_id: format!("occ3q_{}", uuid::Uuid::new_v4()),
            message_id: None,
            message_ids: Vec::new(),
            receiver_role: receiver_role.to_string(),
            receiver_name: receiver_name.to_string(),
            enqueued_at_epoch: now_epoch,
            expires_at_epoch: now_epoch.saturating_add(self.ttl_secs),
            state: Occ3SenderQueueState::Pending,
            attempts: 0,
            reason: None,
            fallback_dispatch_token: None,
            receiver_session_id: None,
            busy_retries: 0,
            retry_not_before_epoch: None,
        });
        let position = self.entries.len() as u32;
        self.notice_at((position - 1) as usize, now_epoch, None)
    }

    /// The head entry, if any. Only a `pending` head may arm.
    pub fn head(&self) -> Option<&Occ3SenderQueueEntry> {
        self.entries.front()
    }

    /// Test-only: drop the head's busy backoff so a test can skip the wait.
    #[cfg(test)]
    pub(crate) fn clear_head_backoff_for_test(&mut self) {
        if let Some(head) = self.entries.front_mut() {
            head.retry_not_before_epoch = None;
        }
    }

    /// Marks the head as armed and records the daemon-minted message id plus
    /// the AUTHENTICATED receiver session id from the authorization.
    pub fn mark_armed(
        &mut self,
        message_id: Option<&str>,
        receiver_session_id: Option<&str>,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        let head = self.entries.front_mut()?;
        if head.state != Occ3SenderQueueState::Pending {
            return None;
        }
        head.state = Occ3SenderQueueState::Armed;
        head.retry_not_before_epoch = None;
        // A busy-retry reason describes the refusal that returned the entry
        // to pending; it must not outlive the fresh arm.
        if matches!(
            head.reason.as_deref(),
            Some(OCC3_SOURCE_BUSY_RETRY_REASON) | Some("receiver_busy_retry")
        ) {
            head.reason = None;
        }
        head.receiver_session_id = receiver_session_id.map(str::to_string);
        if let Some(id) = message_id {
            let id = id.to_string();
            if head.message_id.as_deref() != Some(id.as_str()) {
                head.message_ids.push(("armed".into(), id.clone()));
                if head.message_ids.len() > 8 {
                    let overflow = head.message_ids.len() - 8;
                    head.message_ids.drain(0..overflow);
                }
            }
            head.message_id = Some(id);
        }
        self.notice_at(0, now_epoch, None)
    }

    /// The head advanced to `published`/`admitted` (records the message id).
    pub fn mark_progressed(
        &mut self,
        state: Occ3SenderQueueState,
        message_id: &str,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        let head = self.entries.front_mut()?;
        if head.state.is_terminal() {
            return None;
        }
        if !matches!(
            state,
            Occ3SenderQueueState::Published | Occ3SenderQueueState::Admitted
        ) {
            return None;
        }
        head.state = state;
        head.message_id = Some(message_id.to_string());
        self.notice_at(0, now_epoch, None)
    }

    /// A consumption/failure event for one of this sender's in-flight
    /// messages. Entries are matched by message id, not position: the
    /// receiver's arrival order need not match queue order.
    pub fn record_delivery_event(
        &mut self,
        message_id: &str,
        consumed: bool,
        receipt_sha256: Option<&str>,
        now_epoch: u64,
    ) -> Vec<Occ3QueueNotice> {
        let Some(index) = self.entries.iter().position(|entry| {
            entry.message_id.as_deref() == Some(message_id) && entry.state.is_in_flight()
        }) else {
            return Vec::new();
        };
        let position = (index + 1) as u32;
        if consumed && receipt_sha256.is_some() {
            // Proven completion (consumed + receipt digest): the entry pops
            // with its terminal `completed` notice; the completion proof
            // itself lives with the receiver's marker.
            let entry = &mut self.entries[index];
            entry.state = Occ3SenderQueueState::Completed;
            let entry = self.entries.remove(index).expect("index just found");
            vec![Occ3QueueNotice {
                queue_id: entry.queue_id.clone(),
                message_id: entry.message_id.clone(),
                receiver: entry.receiver_selector(),
                state: "completed",
                attempts: entry.attempts,
                position,
                timestamp: now_epoch,
                reason: None,
                dispatch_token: None,
                refusal: None,
            }]
        } else if consumed {
            // A consumed report WITHOUT a receipt digest cannot prove
            // completion: the entry resolves to `uncertain` - visible and
            // explicit, never auto-advanced, never TTL-expired. Resolution
            // happens through the host reconciliation path only.
            let entry = &mut self.entries[index];
            entry.state = Occ3SenderQueueState::Uncertain;
            entry.reason = Some("consumed_without_receipt_digest".into());
            vec![Occ3QueueNotice {
                queue_id: entry.queue_id.clone(),
                message_id: entry.message_id.clone(),
                receiver: entry.receiver_selector(),
                state: "uncertain",
                attempts: entry.attempts,
                position,
                timestamp: now_epoch,
                reason: entry.reason.clone(),
                dispatch_token: None,
                refusal: None,
            }]
        } else {
            // Proven terminal failure (the receiver's exactly-once terminal
            // poll: Failed|Expired): demote to fallback_pending with a fresh
            // dispatch token for the ordinary re-emission.
            let entry = &mut self.entries[index];
            entry.state = Occ3SenderQueueState::FallbackPending;
            entry.reason = Some("receiver_outcome_unconsumed".into());
            entry.attempts = entry.attempts.saturating_add(1);
            entry.fallback_dispatch_token = Some(new_dispatch_token());
            let token = entry.fallback_dispatch_token.clone();
            vec![Occ3QueueNotice {
                queue_id: entry.queue_id.clone(),
                message_id: entry.message_id.clone(),
                receiver: entry.receiver_selector(),
                state: "fallback_pending",
                attempts: entry.attempts,
                position,
                timestamp: now_epoch,
                reason: entry.reason.clone(),
                dispatch_token: token,
                refusal: None,
            }]
        }
    }

    /// The receiver refused this sender's attached request as busy before
    /// any stream event: the published capture was proven unused, but it is
    /// spent (one publication per capture), so it is never resent. Within
    /// [`OCC3_RECEIVER_BUSY_MAX_RETRIES`] the in-flight entry returns to
    /// `pending` with a backoff, and the model is prompted for a FRESH
    /// capture. After that the entry falls back to ordinary OCC-1
    /// re-emission exactly like any other proven non-use outcome.
    pub fn record_receiver_busy(
        &mut self,
        message_id: &str,
        now_epoch: u64,
    ) -> Vec<Occ3QueueNotice> {
        let Some(index) = self.entries.iter().position(|entry| {
            entry.message_id.as_deref() == Some(message_id) && entry.state.is_in_flight()
        }) else {
            return Vec::new();
        };
        if self.entries[index].busy_retries >= OCC3_RECEIVER_BUSY_MAX_RETRIES {
            let queue_id = self.entries[index].queue_id.clone();
            return self
                .mark_fallback(Some(&queue_id), "receiver_busy_exhausted", now_epoch)
                .into_iter()
                .collect();
        }
        let entry = &mut self.entries[index];
        entry.busy_retries = entry.busy_retries.saturating_add(1);
        entry.state = Occ3SenderQueueState::Pending;
        entry.reason = Some("receiver_busy_retry".into());
        entry.retry_not_before_epoch =
            Some(now_epoch.saturating_add(receiver_busy_backoff_secs(entry.busy_retries)));
        self.notice_at(index, now_epoch, None).into_iter().collect()
    }

    /// The server refused the armed head's SOURCE capture as busy before
    /// admission: nothing was captured or published, so this is not a
    /// capture failure and never touches `attempts` or the session's
    /// capture-failure budget. It shares the per-entry busy budget
    /// ([`OCC3_RECEIVER_BUSY_MAX_RETRIES`]) with receiver-busy refusals.
    /// Within it the head returns to `pending` with a backoff anchored at
    /// the refusal time (`retry_after_secs` when the server sent a valid one,
    /// else 2s/4s/8s); after it the entry falls back to OCC-1 re-emission.
    /// Returns `None` when there is no armed head (the caller then counts an
    /// ordinary capture failure so the attempt stays bounded).
    pub fn record_source_busy(
        &mut self,
        retry_after_secs: Option<u64>,
        refused_at_epoch: u64,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        let head = self.entries.front()?;
        if head.state != Occ3SenderQueueState::Armed {
            return None;
        }
        if head.busy_retries >= OCC3_RECEIVER_BUSY_MAX_RETRIES {
            let queue_id = head.queue_id.clone();
            return self.mark_fallback(Some(&queue_id), "source_busy_exhausted", now_epoch);
        }
        let head = self.entries.front_mut()?;
        head.busy_retries = head.busy_retries.saturating_add(1);
        head.state = Occ3SenderQueueState::Pending;
        head.reason = Some(OCC3_SOURCE_BUSY_RETRY_REASON.into());
        let wait = retry_after_secs
            .filter(|seconds| (1..=OCC3_BUSY_BACKOFF_MAX_SECS).contains(seconds))
            .unwrap_or_else(|| receiver_busy_backoff_secs(head.busy_retries));
        let anchor = if refused_at_epoch == 0 { now_epoch } else { refused_at_epoch.min(now_epoch) };
        head.retry_not_before_epoch = Some(anchor.saturating_add(wait));
        self.notice_at(0, now_epoch, None)
    }

    /// Demotes the head or a matched entry to fallback_pending (ordinary
    /// OCC-1 re-emission in flight, not yet acknowledged). The entry STAYS:
    /// terminal `fallback` is emitted only by [`Self::acknowledge_fallback`]
    /// after the ordinary delivery was acknowledged.
    pub fn mark_fallback(
        &mut self,
        queue_id: Option<&str>,
        reason: &str,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        let index = match queue_id {
            Some(id) => self
                .entries
                .iter()
                .position(|entry| entry.queue_id == id && !entry.state.is_terminal())?,
            None => {
                let head = self.entries.front_mut()?;
                if head.state.is_in_flight() {
                    // In-flight entries fall back only through
                    // `record_delivery_event` or restart; a manual fallback
                    // applies to pending entries.
                    return None;
                }
                0
            }
        };
        let entry = &mut self.entries[index];
        entry.state = Occ3SenderQueueState::FallbackPending;
        entry.reason = Some(bounded_reason(reason));
        entry.attempts = entry.attempts.saturating_add(1);
        entry.fallback_dispatch_token = Some(new_dispatch_token());
        let token = entry.fallback_dispatch_token.clone();
        let position = (index + 1) as u32;
        Some(Occ3QueueNotice {
            queue_id: entry.queue_id.clone(),
            message_id: entry.message_id.clone(),
            receiver: entry.receiver_selector(),
            state: "fallback_pending",
            attempts: entry.attempts,
            position,
            timestamp: now_epoch,
            reason: entry.reason.clone(),
            dispatch_token: token,
            refusal: None,
        })
    }

    /// F3: EXPLICIT manual resolution of an `uncertain` entry. Never
    /// automatic: uncertain entries (a possibly-consumed publication) stay
    /// visible indefinitely until a human-driven host action resolves
    /// them, because re-sending could duplicate a consumed handoff.
    /// `reemit=true` returns the entry to `fallback_pending` with a FRESH
    /// dispatch token (the human explicitly accepts the duplication risk
    /// and the ordinary re-emission proceeds through the token-gated
    /// path); `reemit=false` abandons it as terminal `failed`
    /// (`manual_cancel`) with no re-send. Any other state refuses.
    pub fn resolve_uncertain(
        &mut self,
        queue_id: &str,
        reemit: bool,
        now_epoch: u64,
    ) -> Result<Option<Occ3QueueNotice>, String> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.queue_id == queue_id)
            .ok_or_else(|| "unknown queue id".to_string())?;
        if self.entries[index].state != Occ3SenderQueueState::Uncertain {
            return Err("entry is not uncertain".to_string());
        }
        if reemit {
            let token = new_dispatch_token();
            let entry = &mut self.entries[index];
            entry.state = Occ3SenderQueueState::FallbackPending;
            entry.fallback_dispatch_token = Some(token.clone());
            entry.reason = Some("manual_reemit".to_string());
            let notice = Occ3QueueNotice {
                queue_id: entry.queue_id.clone(),
                message_id: entry.message_id.clone(),
                receiver: entry.receiver_selector(),
                state: "fallback_pending",
                attempts: entry.attempts,
                position: (index + 1) as u32,
                timestamp: now_epoch,
                reason: entry.reason.clone(),
                dispatch_token: Some(token),
                refusal: None,
            };
            Ok(Some(notice))
        } else {
            let entry = self
                .entries
                .remove(index)
                .expect("index was just located above");
            let receiver = entry.receiver_selector();
            Ok(Some(Occ3QueueNotice {
                queue_id: entry.queue_id,
                message_id: entry.message_id,
                receiver,
                state: "failed",
                attempts: entry.attempts,
                position: 0,
                timestamp: now_epoch,
                reason: Some("manual_cancel".to_string()),
                dispatch_token: None,
                refusal: None,
            }))
        }
    }

    /// Acknowledges the ordinary OCC-1 re-emission of a fallback_pending
    /// entry (receipt delivered|queued). ONLY this call emits the terminal
    /// `fallback` notice and pops the entry.
    /// Token-gated acknowledgment: `dispatch_token` must equal the entry's
    /// live token, so only the trusted host that issued the ordinary send
    /// (with the token in its host-side context) can complete the entry. A
    /// wrong or absent token acknowledges nothing.
    pub fn acknowledge_fallback(
        &mut self,
        queue_id: &str,
        dispatch_token: Option<&str>,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        let index = self.entries.iter().position(|entry| {
            entry.queue_id == queue_id
                && entry.state == Occ3SenderQueueState::FallbackPending
                && entry.fallback_dispatch_token.is_some()
                && dispatch_token.is_some_and(|token| {
                    Some(token) == entry.fallback_dispatch_token.as_deref()
                })
        })?;
        let entry = &mut self.entries[index];
        entry.state = Occ3SenderQueueState::Fallback;
        let position = (index + 1) as u32;
        let notice = Occ3QueueNotice {
            queue_id: entry.queue_id.clone(),
            message_id: entry.message_id.clone(),
            receiver: entry.receiver_selector(),
            state: "fallback",
            attempts: entry.attempts,
            position,
            timestamp: now_epoch,
            reason: entry.reason.clone(),
            dispatch_token: None,
            refusal: None,
        };
        self.entries.remove(index).expect("index just found");
        Some(notice)
    }

    /// Silent retirement for a deleted owner session (an RLM child that was
    /// removed). Drops every entry and every undelivered outbox record and
    /// clears the overflow flag. It emits NO notice: the owner is gone, so
    /// nothing may report on its behalf. Returns the number of entries and
    /// outbox records removed.
    pub fn retire_all_silently(&mut self) -> (usize, usize) {
        let entries = self.entries.len();
        let records = self.outbox.len();
        self.entries.clear();
        self.outbox.clear();
        self.outbox_overflow = false;
        (entries, records)
    }

    /// Pops the head after its terminal notice has been emitted.
    pub fn pop_terminal_head(&mut self) -> Option<Occ3SenderQueueEntry> {
        let terminal = self.entries.front()?.state.is_terminal();
        if terminal {
            self.entries.pop_front()
        } else {
            None
        }
    }

    /// A capture failure for the armed head. This is NOT a queue retry: the
    /// PR235 slot budget owns capture retries. The queue demotes the entry
    /// to fallback only when the native path is terminally unavailable.
    pub fn capture_failed(
        &mut self,
        terminal: bool,
        reason: &str,
        now_epoch: u64,
    ) -> Option<Occ3QueueNotice> {
        let head = self.entries.front_mut()?;
        if head.state != Occ3SenderQueueState::Armed {
            return None;
        }
        head.attempts = head.attempts.saturating_add(1);
        if terminal {
            let queue_id = head.queue_id.clone();
            return self.mark_fallback(Some(&queue_id), reason, now_epoch);
        }
        // Non-terminal: the entry returns to pending for the model's
        // explicit re-arm inside the PR235 budget.
        head.state = Occ3SenderQueueState::Pending;
        head.reason = None;
        self.notice_at(0, now_epoch, Some(bounded_reason(reason)))
    }

    /// Drops expired pending and fallback_pending entries to terminal
    /// `expired` notices. In-flight entries never TTL-drop (the lease
    /// deadline owns them). Expired entries pop: expiry is terminal.
    pub fn sweep_expired(&mut self, now_epoch: u64) -> Vec<Occ3QueueNotice> {
        let mut notices = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            let expired = {
                let entry = &self.entries[index];
                entry.expired(now_epoch)
                    && matches!(
                        entry.state,
                        Occ3SenderQueueState::Pending | Occ3SenderQueueState::FallbackPending
                    )
            };
            if expired {
                let entry = self
                    .entries
                    .remove(index)
                    .expect("index bounded by len above");
                notices.push(Occ3QueueNotice {
                    queue_id: entry.queue_id.clone(),
                    message_id: entry.message_id.clone(),
                    receiver: entry.receiver_selector(),
                    state: "expired",
                    attempts: entry.attempts,
                    position: (index + 1) as u32,
                    timestamp: now_epoch,
                    reason: Some("ttl_expired".into()),
                    dispatch_token: None,
                    refusal: None,
                });
            } else {
                index += 1;
            }
        }
        notices
    }

    fn notice_at(
        &self,
        index: usize,
        now_epoch: u64,
        reason: Option<String>,
    ) -> Option<Occ3QueueNotice> {
        let entry = self.entries.get(index)?;
        Some(Occ3QueueNotice {
            queue_id: entry.queue_id.clone(),
            message_id: entry.message_id.clone(),
            receiver: entry.receiver_selector(),
            state: entry.state.as_str_static(),
            attempts: entry.attempts,
            position: (index + 1) as u32,
            timestamp: now_epoch,
            reason: reason.or_else(|| entry.reason.clone()),
            dispatch_token: (entry.state == Occ3SenderQueueState::FallbackPending)
                .then(|| entry.fallback_dispatch_token.clone())
                .flatten(),
            refusal: None,
        })
    }
}

impl Occ3SenderQueue {
    /// The snapshot-form notice details: ONE record carrying the full
    /// current queue (bounded by the max-entries ceiling). Emitted by the
    /// host after every state change so a bounded transcript scan always
    /// finds an authoritative base.
    pub fn snapshot_details(&self) -> Value {
        let entries: Vec<Value> = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let mut value = json!({
                    "queueId": entry.queue_id,
                    "position": (index + 1) as u32,
                    "receiver": entry.receiver_selector(),
                    "state": entry.state.as_str(),
                    "attempts": entry.attempts,
                    "timestamp": entry.enqueued_at_epoch,
                });
                if let Some(message_id) = entry.message_id.as_deref() {
                    value["messageId"] = json!(message_id);
                }
                value
            })
            .collect();
        let mut details = json!({"snapshot": true, "entries": entries});
        if let Some(blocked) = self.blocked.as_ref() {
            details["blocked"] = json!(blocked.as_str());
        }
        if self.outbox_overflow {
            // Visible fail state: the durable notice outbox hit its bound;
            // some notices are undeliverable until it drains. Never a
            // silent drop, never a silent clear.
            details["noticeOverflow"] = json!(true);
        }
        details
    }
}

impl Occ3SenderQueueState {
    fn as_str_static(&self) -> &'static str {
        self.as_str()
    }
}

pub fn bounded_reason(reason: &str) -> String {
    reason.chars().take(128).collect()
}

/// Mints a fresh dispatch token (random, content-free).
fn new_dispatch_token() -> String {
    format!("occ3dt_{}", uuid::Uuid::new_v4())
}

/// One queue-level refusal notice for a refused enqueue (`queue_full` /
/// `queue_blocked`). Content-free and entry-less: the empty queue id means
/// it can never project as a queue entry (the read side rejects empty
/// ids), so a refusal never counts as queued work; the parent-side
/// visibility reads the whitelisted `refused` value from the details and
/// surfaces a refusal banner instead.
pub fn queue_refusal_notice(refused: &'static str, timestamp: u64) -> Occ3QueueNotice {
    debug_assert!(matches!(refused, "queue_full" | "queue_blocked"));
    Occ3QueueNotice {
        queue_id: String::new(),
        message_id: None,
        receiver: "unknown".into(),
        state: "refused",
        attempts: 0,
        position: 0,
        timestamp,
        reason: Some(refused.into()),
        dispatch_token: None,
        refusal: Some(refused.into()),
    }
}

/// Notice states that may raise a LIVE follow-up prompt in the parent.
/// Intermediate states (pending, armed, published, fallback_pending,
/// uncertain, refused) are still recorded durably in the parent
/// transcript, but they do not wake the parent: only an outcome the
/// parent can act on does.
pub fn parent_notice_prompts_live(state: &str) -> bool {
    matches!(
        state,
        "admitted" | "delivered" | "completed" | "fallback" | "failed" | "expired" | "cancelled"
    )
}

/// Coalescing key for the parent's live follow-up prompt of one child
/// queue entry. A newer notice for the same entry replaces a still-queued
/// older prompt; the child prefix lets deletion drop all of them.
pub fn parent_notice_queue_key(child_session_id: &str, queue_id: &str) -> String {
    format!("{}{queue_id}", parent_notice_queue_key_prefix(child_session_id))
}

/// Prefix shared by every parent notice prompt of one child session.
pub fn parent_notice_queue_key_prefix(child_session_id: &str) -> String {
    format!("occ3-child-notice:{child_session_id}:")
}

pub fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn env_ttl_secs() -> u64 {
    std::env::var("OCC3_SENDER_QUEUE_TTL_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(OCC3_SENDER_QUEUE_TTL_SECS)
}

fn env_max_entries() -> usize {
    std::env::var("OCC3_SENDER_QUEUE_MAX_ENTRIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .map(|value: usize| {
            value.clamp(OCC3_SENDER_QUEUE_MIN_ENTRIES, OCC3_SENDER_QUEUE_MAX_ENTRIES_LIMIT)
        })
        .unwrap_or(OCC3_SENDER_QUEUE_MAX_ENTRIES)
}

/// The artifact path for one sender session file, following the OCC
/// session-preference layout (`session-artifacts/<id>/...`).
pub fn sender_queue_path(session_file: &str) -> PathBuf {
    PathBuf::from(
        crate::core::session_manager::get_session_artifact_path_for_file(session_file, None),
    )
    .join("occ3-sender-queue.json")
}

/// The OCC-1 re-emission prompt for a fallback_pending head. The prior
/// native capture is spent; the model re-emits the source facts through the
/// ordinary agent messaging path.
pub fn fallback_prompt(receiver: &str) -> String {
    format!(
        "OCC-3 native handoff for {receiver} fell back to ordinary messaging. Re-emit the essential source facts as a normal agent_message to that receiver now. Do not reuse or replay any earlier capture; send fresh content."
    )
}

/// The re-capture prompt for a pending head the receiver refused as busy.
/// The earlier capture was proven unused but is spent; the model emits the
/// facts again for a fresh capture.
pub fn receiver_busy_prompt(receiver: &str, retry: u32) -> String {
    format!(
        "OCC-3 receiver {receiver} was busy and did not use the native handoff (busy retry {retry} of {OCC3_RECEIVER_BUSY_MAX_RETRIES}). \
Call occ3_handoff with receiver_role and receiver_name for that receiver to arm a fresh capture, then emit the concise \
source facts again as assistant content and call occ3_handoff once more. Do not reuse or replay any earlier capture."
    )
}

/// The re-capture prompt for a pending head whose source capture the
/// server refused as busy. Nothing was captured, so the facts must be
/// emitted again into a fresh capture. Busy retries are counted only at a
/// refusal, never per handoff call.
pub fn source_busy_prompt(receiver: &str, retry: u32) -> String {
    format!(
        "OCC-3 native handoff to {receiver} was not delivered: the server was busy and captured nothing (busy retry {retry} of {OCC3_RECEIVER_BUSY_MAX_RETRIES}). \
If occ3_handoff has not already returned capture_armed since then, call it with receiver_role and receiver_name for that receiver; it waits out the \
short backoff and arms a fresh capture. Then emit the concise source facts again as assistant content and call occ3_handoff once more. \
Do not reuse or replay any earlier capture."
    )
}

/// Dedup key for the host's one-shot head prompt: a receiver-busy retry or a
/// later fallback is a new prompting occasion for the same queue entry.
pub fn head_prompt_key(entry: &Occ3SenderQueueEntry) -> String {
    format!("{}#{}#{}", entry.queue_id, entry.state.as_str(), entry.busy_retries)
}

/// The advance prompt the model receives when the queue advances to a new
/// head, when a capture failed and the native path is still available, or
/// when a blocked/empty restart needs a fresh start. Explicitly instructs a
/// FRESH source emission; never carries prior capture content.
pub fn advance_prompt(queue_len: usize, receiver: &str) -> String {
    format!(
        "OCC-3 sender queue advanced. {queue_len} handoff(s) remain. The next queued receiver is {receiver}. \
Emit fresh concise source facts as assistant content, then call occ3_handoff once with \
receiver_role and receiver_name for that receiver. Do not reuse or replay any earlier \
capture; each handoff uses a fresh capture of this turn."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue() -> Occ3SenderQueue {
        Occ3SenderQueue::new(OCC3_SENDER_QUEUE_TTL_SECS)
    }

    #[test]
    fn fifo_order_and_stable_ids() {
        let mut queue = queue();
        let a = queue.enqueue("child", "alpha", 100).unwrap();
        let b = queue.enqueue("child", "beta", 101).unwrap();
        assert_ne!(a.queue_id, b.queue_id);
        assert_eq!(a.position, 1);
        assert_eq!(b.position, 2);
        assert_eq!(queue.head().unwrap().receiver_name, "alpha");
        queue.mark_armed(Some("agentmsg_1"), None, 102).unwrap();
        let entry = queue.snapshot()[0].clone();
        assert_eq!(entry.state, Occ3SenderQueueState::Armed);
        assert_eq!(entry.message_id.as_deref(), Some("agentmsg_1"));
        queue.mark_progressed(Occ3SenderQueueState::Admitted, "agentmsg_1", 103).unwrap();
        // Consumption pops the entry as terminal completed; beta becomes the
        // head with its stable id.
        let notices = queue.record_delivery_event("agentmsg_1", true, Some("sha"), 104);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].state, "completed");
        assert_eq!(queue.head().unwrap().queue_id, b.queue_id);
        // Unconsumed terminal outcome demotes to fallback_pending and
        // STAYS until the ordinary delivery is acknowledged.
        queue.mark_armed(None, None, 105).unwrap();
        queue.mark_progressed(Occ3SenderQueueState::Published, "agentmsg_2", 106).unwrap();
        let notices = queue.record_delivery_event("agentmsg_2", false, None, 107);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].state, "fallback_pending");
        assert_eq!(queue.len(), 1);
        // Acknowledgment pops it as terminal fallback (token-gated).
        let notice = queue
            .acknowledge_fallback(&notices[0].queue_id, notices[0].dispatch_token.as_deref(), 108)
            .unwrap();
        assert_eq!(notice.state, "fallback");
        assert!(queue.is_empty());
    }

    /// A receiver-busy refusal returns the in-flight entry to `pending`
    /// with a 2s/4s/8s backoff for a fresh capture, keeps it at the head,
    /// prompts once per retry, and falls back to OCC-1 after the budget.
    #[test]
    fn receiver_busy_recaptures_with_backoff_then_falls_back() {
        let mut queue = queue();
        let head = queue.enqueue("child", "busy", 100).unwrap();
        queue.enqueue("child", "behind", 100).unwrap();
        let mut now = 100;
        let mut keys = std::collections::HashSet::new();
        for retry in 1..=OCC3_RECEIVER_BUSY_MAX_RETRIES {
            let message_id = format!("agentmsg_{retry}");
            queue.mark_armed(Some(&message_id), None, now).unwrap();
            queue
                .mark_progressed(Occ3SenderQueueState::Admitted, &message_id, now)
                .unwrap();
            let notices = queue.record_receiver_busy(&message_id, now);
            assert_eq!(notices.len(), 1);
            assert_eq!(notices[0].state, "pending", "retry {retry}");
            assert_eq!(notices[0].reason.as_deref(), Some("receiver_busy_retry"));
            assert!(notices[0].dispatch_token.is_none(), "no OCC-1 fallback yet");
            let entry = queue.head().unwrap();
            assert_eq!(entry.queue_id, head.queue_id, "busy entry keeps the head");
            assert_eq!(entry.busy_retries, retry);
            assert_eq!(entry.attempts, 0, "busy is not a capture failure");
            assert_eq!(
                entry.retry_not_before_epoch,
                Some(now + receiver_busy_backoff_secs(retry))
            );
            assert!(keys.insert(head_prompt_key(entry)), "each retry prompts once");
            now += 10;
        }
        assert_eq!(
            (1..=3).map(receiver_busy_backoff_secs).collect::<Vec<_>>(),
            vec![2, 4, 8]
        );
        queue.mark_armed(Some("agentmsg_last"), None, now).unwrap();
        assert_eq!(queue.head().unwrap().retry_not_before_epoch, None, "arm clears the backoff");
        queue
            .mark_progressed(Occ3SenderQueueState::Admitted, "agentmsg_last", now)
            .unwrap();
        let notices = queue.record_receiver_busy("agentmsg_last", now);
        assert_eq!(notices[0].state, "fallback_pending");
        assert_eq!(notices[0].reason.as_deref(), Some("receiver_busy_exhausted"));
        assert!(notices[0].dispatch_token.is_some());
        assert!(keys.insert(head_prompt_key(queue.head().unwrap())), "fallback prompts too");
        // Unknown or non-in-flight ids change nothing.
        assert!(queue.record_receiver_busy("agentmsg_unknown", now).is_empty());
        assert!(queue.record_receiver_busy("agentmsg_last", now).is_empty());
    }

    /// Older artifacts without the busy fields still load; the fields
    /// round-trip, and an out-of-bound busy count blocks the load.
    #[test]
    fn source_busy_backs_off_then_falls_back_without_touching_attempts() {
        let mut queue = queue();
        let now = 1_000;
        queue.enqueue("child", "receiver", now).unwrap();
        for retry in 1..=OCC3_RECEIVER_BUSY_MAX_RETRIES {
            queue.mark_armed(None, Some("receiver"), now).unwrap();
            let notice = queue.record_source_busy(None, now, now).expect("armed head");
            assert_eq!(notice.state, "pending");
            assert_eq!(notice.reason.as_deref(), Some(OCC3_SOURCE_BUSY_RETRY_REASON));
            let head = queue.head().unwrap();
            assert_eq!(head.busy_retries, retry);
            assert_eq!(head.attempts, 0, "busy is not a capture failure");
            assert_eq!(head.retry_not_before_epoch, Some(now + receiver_busy_backoff_secs(retry)));
        }
        queue.mark_armed(None, Some("receiver"), now).unwrap();
        let notice = queue.record_source_busy(None, now, now).unwrap();
        assert_eq!(notice.state, "fallback_pending");
        assert_eq!(notice.reason.as_deref(), Some("source_busy_exhausted"));
        // Not armed: nothing to absorb.
        assert!(queue.record_source_busy(None, now, now).is_none());
    }

    #[test]
    fn source_busy_honours_bounded_retry_after_anchored_at_the_refusal() {
        let mut queue = queue();
        queue.enqueue("child", "receiver", 100).unwrap();
        queue.mark_armed(None, None, 100).unwrap();
        queue.record_source_busy(Some(5), 100, 103).unwrap();
        assert_eq!(queue.head().unwrap().retry_not_before_epoch, Some(105));
        queue.mark_armed(None, None, 110).unwrap();
        assert_eq!(queue.head().unwrap().reason, None);
        // Out-of-bound Retry-After falls back to the 2/4/8 s schedule.
        queue.record_source_busy(Some(31), 110, 110).unwrap();
        assert_eq!(queue.head().unwrap().retry_not_before_epoch, Some(114));
        // A refusal clock ahead of now is clamped to now.
        queue.mark_armed(None, None, 120).unwrap();
        queue.record_source_busy(None, 500, 120).unwrap();
        assert_eq!(queue.head().unwrap().retry_not_before_epoch, Some(128));
    }

    #[test]
    fn source_busy_entry_persists_and_reloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queue.json");
        let mut queue = queue();
        queue.enqueue("child", "receiver", now_epoch()).unwrap();
        queue.mark_armed(None, None, now_epoch()).unwrap();
        queue.record_source_busy(None, now_epoch(), now_epoch()).unwrap();
        queue.persist(&path).unwrap();
        let (loaded, outcome) = Occ3SenderQueue::load(&path, 3600);
        assert!(matches!(outcome, Occ3SenderQueueLoad::Loaded(..)));
        let head = loaded.head().unwrap();
        assert_eq!(head.state, Occ3SenderQueueState::Pending);
        assert_eq!(head.busy_retries, 1);
        assert_eq!(head.reason.as_deref(), Some(OCC3_SOURCE_BUSY_RETRY_REASON));
    }

    #[test]
    fn source_busy_prompt_is_truthful() {
        let prompt = source_busy_prompt("sibling:r", 2);
        assert!(prompt.contains("busy retry 2 of 3"));
        assert!(prompt.contains("captured nothing"));
        assert!(prompt.contains("emit the concise source facts again"));
        assert!(prompt.contains("Do not reuse or replay"));
    }

    #[test]
    fn busy_fields_are_backward_compatible_and_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        let legacy = json!({"schema": OCC3_SENDER_QUEUE_SCHEMA, "entries": [{
            "queueId": "occ3q_legacy", "receiverRole": "child", "receiverName": "r",
            "enqueuedAtEpoch": 1, "expiresAtEpoch": u64::MAX / 2,
            "state": "pending", "attempts": 0
        }]});
        std::fs::write(&path, legacy.to_string()).unwrap();
        let (queue, load) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        assert!(matches!(load, Occ3SenderQueueLoad::Loaded(..)));
        assert_eq!(queue.head().unwrap().busy_retries, 0);
        assert_eq!(queue.head().unwrap().retry_not_before_epoch, None);
        let mut over = legacy.clone();
        over["entries"][0]["busyRetries"] = json!(OCC3_RECEIVER_BUSY_MAX_RETRIES + 1);
        std::fs::write(&path, over.to_string()).unwrap();
        let (_, load) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        assert!(matches!(load, Occ3SenderQueueLoad::Blocked(..)));
    }

    #[test]
    fn enqueue_bounds_and_blocked() {
        let mut queue = queue();
        for i in 0..OCC3_SENDER_QUEUE_MAX_ENTRIES {
            assert!(queue.enqueue("child", &format!("r{i}"), 100).is_some());
        }
        assert!(queue.enqueue("child", "overflow", 100).is_none());
        queue.blocked = Some(Occ3SenderQueueBlocked::UnsupportedSchema {
            found: "future".into(),
        });
        assert!(queue.enqueue("child", "x", 100).is_none());
        assert!(queue.persist(Path::new("/nonexistent/z/occ3-sender-queue.json")).is_err());
    }

    #[test]
    fn ttl_sweep_only_pending() {
        let mut queue = Occ3SenderQueue::new(60);
        queue.enqueue("child", "slow", 0).unwrap();
        queue.mark_armed(None, None, 1).unwrap();
        // Armed entries never TTL-drop.
        assert!(queue.sweep_expired(10_000).is_empty());
        // Force a pending entry and expire it.
        let mut queue2 = Occ3SenderQueue::new(60);
        queue2.enqueue("child", "p", 0).unwrap();
        let notices = queue2.sweep_expired(120);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].state, "expired");
        assert_eq!(notices[0].reason.as_deref(), Some("ttl_expired"));
        assert!(queue2.is_empty());
    }

    #[test]
    fn persistence_round_trip_and_restart_demotion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        let mut queue = queue();
        // `load` stamps its own real clock onto the TTL check, so the
        // fixture must enqueue at real epoch times (tiny 1970 timestamps
        // read as millennia-expired and get dropped at restart).
        let now = now_epoch();
        queue.enqueue("child", "kept", now).unwrap();
        queue.mark_armed(Some("agentmsg_9"), None, now + 1).unwrap();
        queue.enqueue("sibling", "later", now + 2).unwrap();
        queue.persist(&path).unwrap();
        let (loaded, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        match outcome {
            Occ3SenderQueueLoad::Loaded(entries, expired_notices) => {
                assert_eq!(entries.len(), 2);
                // Nothing expired on disk: no expiry notices, no loss.
                assert!(expired_notices.is_empty());
                // F1 ARMED split: pre-publication, budget remaining =>
                // PENDING for a fresh explicit arm; attempt history
                // recorded without overwrite.
                assert_eq!(entries[0].state, Occ3SenderQueueState::Pending);
                assert_eq!(entries[0].reason.as_deref(), Some("restart_pre_publication"));
                assert_eq!(entries[0].attempts, 1);
                assert_eq!(entries[0].message_id.as_deref(), Some("agentmsg_9"));
                assert!(entries[0]
                    .message_ids
                    .iter()
                    .any(|(label, id)| label == "restart_ambiguous" && id == "agentmsg_9"));
                assert_eq!(entries[1].state, Occ3SenderQueueState::Pending);
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        assert_eq!(loaded.snapshot_vec()[0].queue_id, queue.snapshot_vec()[0].queue_id);
    }

    #[test]
    fn unknown_schema_blocks_and_preserves_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        std::fs::write(&path, "{\"schema\":\"optimus.occ3-sender-queue.v999\",\"entries\":[]}")
            .unwrap();
        let bytes_before = std::fs::read(&path).unwrap();
        let (queue, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        match outcome {
            Occ3SenderQueueLoad::Blocked(blocked) => {
                assert_eq!(blocked.as_str(), "unsupported_schema");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
        assert!(queue.blocked().is_some());
        // The file is preserved byte-for-byte.
        assert_eq!(std::fs::read(&path).unwrap(), bytes_before);
        // No legacy-file data loss: a missing file is Fresh, not Blocked.
        let (fresh, outcome) =
            Occ3SenderQueue::load(&dir.path().join("absent.json"), OCC3_SENDER_QUEUE_TTL_SECS);
        assert!(matches!(outcome, Occ3SenderQueueLoad::Fresh));
        assert!(fresh.is_empty());
    }

    #[test]
    fn corrupt_file_blocks_without_loss() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        std::fs::write(&path, "{not json").unwrap();
        let (queue, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        assert!(matches!(
            outcome,
            Occ3SenderQueueLoad::Blocked(Occ3SenderQueueBlocked::Unreadable { .. })
        ));
        assert!(queue.blocked().is_some());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{not json");
    }

    #[test]
    fn delivery_event_matches_by_message_id_anywhere() {
        let mut queue = queue();
        queue.enqueue("child", "one", 100).unwrap();
        queue.enqueue("child", "two", 101).unwrap();
        // The second entry completes first (receiver FIFO != queue order).
        let notices = queue.record_delivery_event("agentmsg_two", true, None, 102);
        assert!(notices.is_empty(), "unknown id: no event");
        queue.mark_armed(None, None, 102).unwrap();
        queue
            .mark_progressed(Occ3SenderQueueState::Published, "agentmsg_two", 103)
            .unwrap();
        let notices = queue.record_delivery_event("agentmsg_two", true, None, 104);
        assert_eq!(notices.len(), 1);
    }

    #[test]
    fn capture_failure_semantics() {
        let mut queue = queue();
        queue.enqueue("child", "r", 100).unwrap();
        queue.mark_armed(None, None, 101).unwrap();
        // Non-terminal: returns to pending, attempts bumped, no budget eaten.
        let notice = queue.capture_failed(false, "watchdog_abort", 102).unwrap();
        assert_eq!(notice.state, "pending");
        assert_eq!(notice.attempts, 1);
        assert!(queue.head().unwrap().state == Occ3SenderQueueState::Pending);
        // Terminal native death: fallback_pending, entry stays for the
        // ordinary re-emission and its acknowledgment.
        queue.mark_armed(None, None, 103).unwrap();
        let notice = queue.capture_failed(true, "native_path_deactivated", 104).unwrap();
        assert_eq!(notice.state, "fallback_pending");
        assert_eq!(queue.len(), 1);
        let popped = queue
            .acknowledge_fallback(&notice.queue_id, notice.dispatch_token.as_deref(), 105)
            .unwrap();
        assert_eq!(popped.state, "fallback");
        assert!(queue.is_empty());
    }

    #[test]
    fn blocked_notice_shape_and_notice_details() {
        let notice = Occ3QueueNotice {
            queue_id: "occ3q_x".into(),
            message_id: Some("agentmsg_y".into()),
            receiver: "child:beta".into(),
            state: "fallback",
            attempts: 2,
            position: 1,
            timestamp: 1234,
            reason: Some("ttl_expired".into()),
            dispatch_token: None,
            refusal: None,
        };
        let details = notice.details();
        assert_eq!(details["queueId"], "occ3q_x");
        assert_eq!(details["messageId"], "agentmsg_y");
        assert_eq!(details["receiver"], "child:beta");
        assert_eq!(details["state"], "fallback");
        assert_eq!(details["attempts"], 2);
        assert_eq!(details["position"], 1);
        assert_eq!(details["timestamp"], 1234);
        assert_eq!(details["reason"], "ttl_expired");
        // Content-free: no handle-like keys anywhere.
        let text = details.to_string();
        assert!(!text.contains("handle"));
        assert!(!text.contains("receipt"));
    }


    #[test]
    fn restart_with_exhausted_budget_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        let mut queue = queue();
        // Real epoch times: load's TTL check runs against the real clock.
        let now = now_epoch();
        queue.enqueue("child", "spent", now).unwrap();
        queue.mark_armed(Some("agentmsg_spent"), None, now + 1).unwrap();
        {
            let head = queue.entries.front_mut().unwrap();
            head.attempts = crate::core::agent_session::occ3_state::MAX_NATIVE_CAPTURE_RETRIES;
        }
        queue.persist(&path).unwrap();
        let (loaded, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        match outcome {
            Occ3SenderQueueLoad::Loaded(entries, expired_notices) => {
                assert!(expired_notices.is_empty());
                // Budget exhausted at restart (armed, pre-publication):
                // ordinary re-emission, not a fresh arm.
                assert_eq!(entries[0].state, Occ3SenderQueueState::FallbackPending);
                assert_eq!(entries[0].reason.as_deref(), Some("restart_pre_publication"));
                assert_eq!(entries[0].attempts, crate::core::agent_session::occ3_state::MAX_NATIVE_CAPTURE_RETRIES + 1);
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        assert!(loaded.snapshot_vec()[0].state.is_in_flight() == false);
    }


    #[test]
    fn expired_entries_journal_notices_at_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        let mut queue = queue();
        // Real epoch times: the queue's own TTL check at load runs against
        // the real clock.
        let now = now_epoch();
        let stale_pending = queue.enqueue("child", "stale-pending", now).unwrap();
        let stale_fallback = queue.enqueue("child", "stale-fallback", now).unwrap();
        let live = queue.enqueue("child", "live", now).unwrap();
        // Two entries expire on disk: a pending one and a fallback_pending
        // one (the same predicate the live sweep enforces).
        {
            let mut it = queue.entries.iter_mut();
            let first = it.next().unwrap();
            first.expires_at_epoch = now - 1;
            let second = it.next().unwrap();
            second.state = Occ3SenderQueueState::FallbackPending;
            second.expires_at_epoch = now - 1;
        }
        queue.persist(&path).unwrap();
        let (loaded, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        match outcome {
            Occ3SenderQueueLoad::Loaded(entries, expired_notices) => {
                // Only the live entry survives.
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].queue_id, live.queue_id);
                // Restart expiry is never silent: one terminal expired
                // notice per dropped entry, mirroring the live sweep's
                // shape (state, reason, position held at drop).
                assert_eq!(expired_notices.len(), 2);
                assert_eq!(expired_notices[0].queue_id, stale_pending.queue_id);
                assert_eq!(expired_notices[0].state, "expired");
                assert_eq!(expired_notices[0].reason.as_deref(), Some("ttl_expired"));
                assert_eq!(expired_notices[0].position, 1);
                assert_eq!(expired_notices[1].queue_id, stale_fallback.queue_id);
                assert_eq!(expired_notices[1].position, 2);
                assert!(expired_notices
                    .iter()
                    .all(|notice| notice.dispatch_token.is_none()));
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        assert_eq!(loaded.snapshot_vec().len(), 1);
    }

    #[test]
    fn outbox_record_ack_and_overflow_bound() {
        // The outbox cap is independent of the queue entry ceiling: the
        // ceiling is raised to its documented limit so that the OUTBOX is
        // the surface that refuses, not the queue. (CAP == LIMIT == 64.)
        let mut queue =
            Occ3SenderQueue::with_limits(OCC3_SENDER_QUEUE_TTL_SECS, OCC3_SENDER_QUEUE_MAX_ENTRIES_LIMIT);
        let notice = queue.enqueue("child", "r", 1000).unwrap();
        // Record + ack transcript: still pending (parent leg open).
        let record = queue.outbox_record(&notice).expect("recorded");
        assert!(record.notice_id.starts_with("occ3n_"));
        let still = queue.outbox_ack_transcript(&record.notice_id).expect("still pending");
        assert!(still.transcript_done && !still.parent_done);
        // Ack parent: fully delivered, removed.
        assert!(queue.outbox_ack_parent(&record.notice_id).is_none());
        assert!(queue.outbox_pending().is_empty());
        assert!(!queue.outbox_overflow());
        // Unknown id: no-op.
        assert!(queue.outbox_ack_parent("occ3n_missing").is_none());
        // Overflow: the cap refuses further records and flips the flag;
        // draining below the cap clears it. The ack lifecycle above
        // REMOVED the initial record (that is its proof), so the outbox
        // counts only the loop's records: CAP - 1 bulk enqueues stay
        // inside the entry ceiling (LIMIT 64 = 1 initial + 63 bulk) and
        // fill the outbox to CAP - 1.
        for _ in 0..OCC3_SENDER_QUEUE_OUTBOX_CAP - 1 {
            let notice = queue.enqueue("child", "bulk", 2000).unwrap();
            assert!(queue.outbox_record(&notice).is_some());
        }
        // Hand-built notices do not consume entry slots, so they fill
        // the outbox to EXACTLY the cap without touching the queue
        // ceiling: the filler is the CAP-th record (accepted), and the
        // probe is the CAP+1-th (refused by the outbox cap itself - the
        // entry ceiling cannot mask the refusing surface).
        let overflow_notice = |queue_id: &str| Occ3QueueNotice {
            queue_id: queue_id.into(),
            message_id: None,
            receiver: "child:bulk".into(),
            state: "pending",
            attempts: 0,
            position: 0,
            timestamp: 3000,
            reason: None,
            refusal: None,
            // No fallback dispatch token: these synthetic probes only
            // exercise the outbox cap - a token exists solely for
            // fallback-pending entries acknowledged through the
            // ordinary re-emission flow.
            dispatch_token: None,
        };
        assert!(
            queue
                .outbox_record(&overflow_notice("occ3q_overflow_filler"))
                .is_some(),
            "the CAP-th record is accepted: exactly at the cap"
        );
        assert_eq!(
            queue.outbox_pending().len(),
            OCC3_SENDER_QUEUE_OUTBOX_CAP,
            "outbox count is exactly the cap before the probe"
        );
        assert!(
            queue
                .outbox_record(&overflow_notice("occ3q_overflow_probe"))
                .is_none(),
            "the CAP+1-th record is refused by the outbox cap"
        );
        assert!(queue.outbox_overflow());
        let snapshot = queue.snapshot_details();
        assert_eq!(snapshot["noticeOverflow"], true);
        // Acking everything drains below the cap and clears the flag.
        for record in queue.outbox_pending() {
            let _ = queue.outbox_ack_transcript(&record.notice_id);
            let _ = queue.outbox_ack_parent(&record.notice_id);
        }
        assert!(!queue.outbox_overflow());
    }

    #[test]
    fn outbox_persists_and_reloads_with_the_queue() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("occ3-sender-queue.json");
        let mut queue = queue();
        let notice = queue.enqueue("child", "kept", now_epoch()).unwrap();
        let record = queue.outbox_record(&notice).expect("recorded");
        queue.persist(&path).unwrap();
        // Reload: the outbox record rides the same artifact.
        let (mut reloaded, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        match outcome {
            Occ3SenderQueueLoad::Loaded(entries, _) => assert_eq!(entries.len(), 1),
            other => panic!("expected Loaded, got {other:?}"),
        }
        let pending = reloaded.outbox_pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].notice_id, record.notice_id);
        // Old artifact without outbox fields migrates safely: empty.
        std::fs::write(
            &path,
            serde_json::json!({
                "schema": "optimus.occ3-sender-queue.v1",
                "entries": [],
            })
            .to_string(),
        )
        .unwrap();
        let (legacy, outcome) = Occ3SenderQueue::load(&path, OCC3_SENDER_QUEUE_TTL_SECS);
        match outcome {
            Occ3SenderQueueLoad::Loaded(entries, expired) => {
                assert!(entries.is_empty() && expired.is_empty());
            }
            other => panic!("expected Loaded, got {other:?}"),
        }
        assert!(legacy.outbox_pending().is_empty());
        assert!(!legacy.outbox_overflow());
    }

    #[test]
    fn configurable_entry_ceiling_is_enforced() {
        let mut queue = Occ3SenderQueue::with_limits(OCC3_SENDER_QUEUE_TTL_SECS, 2);
        assert_eq!(queue.max_entries(), 2);
        assert!(queue.enqueue("child", "a", 100).is_some());
        assert!(queue.enqueue("child", "b", 100).is_some());
        assert!(queue.enqueue("child", "c", 100).is_none(), "ceiling 2 enforced");
    }

    #[test]
    fn retire_all_silently_drops_entries_and_undelivered_notices() {
        let mut queue = queue();
        let first = queue.enqueue("child", "a", 100).unwrap();
        queue.outbox_record(&first);
        let second = queue.enqueue("child", "b", 100).unwrap();
        queue.outbox_record(&second);
        assert_eq!(queue.outbox_pending().len(), 2);
        assert_eq!(queue.retire_all_silently(), (2, 2));
        assert!(queue.is_empty());
        assert!(queue.outbox_pending().is_empty(), "no notice survives deletion");
        assert!(!queue.outbox_overflow());
    }

    #[test]
    fn only_terminal_notice_states_prompt_the_parent_live() {
        for state in ["admitted", "delivered", "completed", "fallback", "failed", "expired", "cancelled"] {
            assert!(parent_notice_prompts_live(state), "{state} prompts");
        }
        for state in ["pending", "armed", "published", "fallback_pending", "uncertain", "refused", "unknown"] {
            assert!(!parent_notice_prompts_live(state), "{state} stays durable-only");
        }
        assert_eq!(
            parent_notice_queue_key("child-1", "occ3q_x"),
            format!("{}occ3q_x", parent_notice_queue_key_prefix("child-1"))
        );
    }

    #[test]
    fn advance_prompt_names_receiver_and_forbids_replay() {
        let prompt = advance_prompt(2, "child:beta");
        assert!(prompt.contains("2 handoff(s) remain"));
        assert!(prompt.contains("child:beta"));
        assert!(prompt.contains("fresh"));
        assert!(prompt.contains("Do not reuse or replay"));
    }
}
