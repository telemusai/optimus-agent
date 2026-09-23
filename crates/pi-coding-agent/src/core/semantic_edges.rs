//! Port of packages/coding-agent/src/core/semantic-edges.ts
//!
//! ACP semantic-edges-v1 producer: a durable per-agent ledger of model-request
//! events plus a pure fold that derives sparse semantic edges between logical
//! request IDs (published downstream under ai.prime.acp/semantic-edges-v1).
//!
//! Ledger events are written before the effects they describe. Edges are
//! commit-gated: they materialize only when their target request finishes.
//! prime-agent has no prompt rollback (unlike nano-rlm's checkpoint/restore),
//! so a failed request simply returns its inbound edges to the session's
//! pending set and they attach to the next committed request in that session.
//! A request whose stream never completes (hard crash, torn tail) stays
//! in-flight and produces no edges.
//!
//! Divergence from nano-rlm: compaction summary requests claim no pending
//! edges (and no spawn). nano-rlm's single summary request can claim safely;
//! prime-agent's split turns run several racing slices, so pending edges would
//! land on whichever slice started first — a dead end when a different slice
//! commits last. Pending therefore defers past summary slices; a COMPLETED
//! compaction flushes it to its last-committed slice — the same request that
//! sources the compaction edge — so pending always lands on a committed
//! request even when the session never runs another turn. Failed or cancelled
//! compactions leave pending for the next turn.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::core::event_log::{EventLog, EventLogOptions, ReplayOptions};

pub const MODEL_REQUEST_ID_HEADER: &str = "X-ACP-Model-Request-ID";
pub const IDEMPOTENCY_KEY_HEADER: &str = "Idempotency-Key";
pub const SEMANTIC_EDGES_LEDGER_FILENAME: &str = "semantic-edges.jsonl";

pub type SemanticEdgeType = String;
pub type CompactionStatus = String;

pub const SEMANTIC_EDGE_CONTINUATION: &str = "continuation";
pub const SEMANTIC_EDGE_SUBAGENT_CALL: &str = "subagent_call";
pub const SEMANTIC_EDGE_SUBAGENT_RETURN: &str = "subagent_return";
pub const SEMANTIC_EDGE_COMPACTION: &str = "compaction";

pub const COMPACTION_STATUS_COMPLETED: &str = "completed";
pub const COMPACTION_STATUS_FAILED: &str = "failed";
pub const COMPACTION_STATUS_CANCELLED: &str = "cancelled";

/// One ledger event. Field names match the JSONL wire format exactly; absent
/// optional keys stay absent (`skip_serializing_if`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum SemanticEdgeLedgerEvent {
    #[serde(rename = "session_registered")]
    SessionRegistered {
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_session_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        spawned_by_request_id: Option<String>,
    },
    #[serde(rename = "request_started")]
    RequestStarted {
        request_id: String,
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        compaction_id: Option<String>,
    },
    #[serde(rename = "request_finished")]
    RequestFinished { request_id: String },
    #[serde(rename = "request_failed")]
    RequestFailed { request_id: String },
    #[serde(rename = "compaction_begun")]
    CompactionBegun {
        compaction_id: String,
        session_id: String,
    },
    #[serde(rename = "compaction_finished")]
    CompactionFinished {
        compaction_id: String,
        status: CompactionStatus,
    },
    #[serde(rename = "child_returned")]
    ChildReturned {
        /// The session whose ledger this is: the parent claiming the return.
        session_id: String,
        child_session_id: String,
        /// The child's last committed request, captured at the success point.
        request_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SemanticEdge {
    pub source_request_id: String,
    pub target_request_id: String,
    #[serde(rename = "type")]
    pub type_: SemanticEdgeType,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SemanticEdgesResult {
    pub edges: Vec<SemanticEdge>,
}

/// The one derivation of where a session's ledger lives; recorder and outbox must agree.
pub fn semantic_edge_ledger_path(
    rlm_session_dir: Option<&str>,
    session_artifact_dir: Option<&str>,
) -> Option<String> {
    let dir = rlm_session_dir.or(session_artifact_dir)?;
    Some(
        std::path::Path::new(dir)
            .join(SEMANTIC_EDGES_LEDGER_FILENAME)
            .to_string_lossy()
            .to_string(),
    )
}

pub fn model_request_headers(request_id: &str) -> serde_json::Map<String, Value> {
    let mut headers = serde_json::Map::new();
    headers.insert(
        MODEL_REQUEST_ID_HEADER.to_string(),
        Value::String(request_id.to_string()),
    );
    headers.insert(
        IDEMPOTENCY_KEY_HEADER.to_string(),
        Value::String(request_id.to_string()),
    );
    headers
}

fn mint_id() -> String {
    uuid::Uuid::new_v4().to_string().replace('-', "")
}

/// Content hash of one turn call body; Idempotency-Key reuse is only safe for a byte-identical retry.
pub fn hash_turn_body(
    model: &pi_ai::types::Model,
    context: &pi_ai::types::Context,
    options: Option<&pi_ai::types::SimpleStreamOptions>,
) -> String {
    let payload = serde_json::json!({
        "provider": model.provider,
        "model": model.id,
        "systemPrompt": context.system_prompt,
        "messages": context.messages,
        "tools": context.tools,
        "reasoning": options.and_then(|options| options.reasoning.clone()),
        "thinkingBudgets": options.and_then(|options| options.thinking_budgets.clone()),
        "temperature": options.and_then(|options| options.stream.temperature),
        "maxTokens": options.and_then(|options| options.stream.max_tokens),
        "serviceTier": options.and_then(|options| options.stream.service_tier.clone()),
    });
    let mut hasher = Sha256::new();
    hasher.update(
        serde_json::to_string(&payload)
            .unwrap_or_default()
            .as_bytes(),
    );
    format!("{:x}", hasher.finalize())
}

/// Append-only semantic-edge recorder for one agent session. It only WRITES
/// events; edge semantics live in the pure [`derive_semantic_edges`] fold.
///
/// Constructing a recorder over an existing ledger replays it; registration is
/// idempotent across resumes.
pub struct SemanticEdgeRecorder {
    pub session_id: String,
    ledger_path: Option<String>,
    event_log: Option<EventLog>,
    disabled: bool,
    epoch: u64,
    last_turn: Option<LastTurn>,
    parked_retry: Option<LastTurn>,
    last_committed_request_id: Option<String>,
    open_compactions: BTreeSet<String>,
    /// Captured once so `_disable` logs the same path the TypeScript closure sees.
    log_target: String,
}

#[derive(Debug, Clone, PartialEq)]
struct LastTurn {
    request_id: String,
    epoch: u64,
    body_hash: Option<String>,
}

impl SemanticEdgeRecorder {
    pub fn new(
        ledger_path: Option<String>,
        session_id: String,
        parent_session_id: Option<String>,
        spawned_by_request_id: Option<String>,
    ) -> Self {
        let log_target = ledger_path.clone().unwrap_or_default();
        let event_log = ledger_path
            .as_ref()
            .map(|path| EventLog::new(path.clone(), EventLogOptions::default()));
        let mut recorder = SemanticEdgeRecorder {
            session_id,
            ledger_path,
            event_log,
            disabled: false,
            epoch: 0,
            last_turn: None,
            parked_retry: None,
            last_committed_request_id: None,
            open_compactions: BTreeSet::new(),
            log_target,
        };

        let existing = match recorder.event_log.as_ref() {
            Some(event_log) => {
                match event_log.replay_sync(parse_semantic_edge_line, ReplayOptions::default()) {
                    Ok(events) => events,
                    Err(error) => {
                        recorder.disable(&error);
                        return recorder;
                    }
                }
            }
            None => Vec::new(),
        };
        let registered = existing.iter().any(|event| {
            matches!(
                event,
                SemanticEdgeLedgerEvent::SessionRegistered { session_id, .. }
                    if session_id == &recorder.session_id
            )
        });
        if registered {
            for event in existing {
                recorder.replay(&event);
            }
            return recorder;
        }

        recorder.append(SemanticEdgeLedgerEvent::SessionRegistered {
            session_id: recorder.session_id.clone(),
            parent_session_id,
            spawned_by_request_id,
        });
        recorder
    }

    // Provenance is best-effort: the first ledger failure permanently disables all
    // writes (one warning), and callers stop emitting request IDs on the wire so
    // the ledger-before-wire invariant is preserved rather than weakened.
    fn disable(&mut self, error: &str) {
        if self.disabled {
            return;
        }
        self.disabled = true;
        eprintln!(
            "semantic-edge ledger disabled at {}: {error}",
            self.log_target
        );
    }

    pub fn last_turn_request_id(&self) -> Option<String> {
        self.last_turn.as_ref().map(|turn| turn.request_id.clone())
    }

    pub fn last_committed_request_id(&self) -> Option<String> {
        self.last_committed_request_id.clone()
    }

    /// Mint (or, for a body-identical retry, reuse) the request ID for one turn
    /// call. The body hash is captured eagerly, before the wire call, so a later
    /// mutation of the live message objects can never alias two different bodies
    /// under one parked Idempotency-Key. A reused retry re-logs request_started
    /// so the fold re-claims the failed attempt's returned pending edges.
    pub fn start_turn_request(&mut self, body_hash: Option<String>) -> Option<String> {
        let parked = self.parked_retry.clone();
        let request_id = match &parked {
            Some(parked)
                if parked.epoch == self.epoch
                    && parked.body_hash.is_some()
                    && parked.body_hash == body_hash =>
            {
                parked.request_id.clone()
            }
            _ => mint_id(),
        };
        if Some(request_id.as_str()) == parked.as_ref().map(|parked| parked.request_id.as_str()) {
            self.parked_retry = None;
        }
        let recorded = self.append(SemanticEdgeLedgerEvent::RequestStarted {
            request_id: request_id.clone(),
            session_id: self.session_id.clone(),
            compaction_id: None,
        });
        if !recorded {
            return None;
        }
        self.last_turn = Some(LastTurn {
            request_id: request_id.clone(),
            epoch: self.epoch,
            body_hash,
        });
        Some(request_id)
    }

    /// Park the last turn request so the upcoming auto-retry reuses its ID.
    pub fn prepare_turn_retry(&mut self) {
        self.parked_retry = self.last_turn.clone();
    }

    pub fn clear_turn_retry(&mut self) {
        self.parked_retry = None;
    }

    pub fn finish_request(&mut self, request_id: Option<&str>) {
        let Some(request_id) = request_id else {
            return;
        };
        self.append(SemanticEdgeLedgerEvent::RequestFinished {
            request_id: request_id.to_string(),
        });
    }

    pub fn fail_request(&mut self, request_id: Option<&str>) {
        let Some(request_id) = request_id else {
            return;
        };
        self.append(SemanticEdgeLedgerEvent::RequestFailed {
            request_id: request_id.to_string(),
        });
    }

    pub fn begin_compaction(&mut self) -> String {
        let compaction_id = mint_id();
        self.append(SemanticEdgeLedgerEvent::CompactionBegun {
            compaction_id: compaction_id.clone(),
            session_id: self.session_id.clone(),
        });
        compaction_id
    }

    /// Mint the summary request for a wire compaction; extension-supplied summaries make no request.
    pub fn start_compaction_request(&mut self, compaction_id: &str) -> Option<String> {
        let request_id = mint_id();
        let recorded = self.append(SemanticEdgeLedgerEvent::RequestStarted {
            request_id: request_id.clone(),
            session_id: self.session_id.clone(),
            compaction_id: Some(compaction_id.to_string()),
        });
        if recorded {
            Some(request_id)
        } else {
            None
        }
    }

    pub fn finish_compaction(
        &mut self,
        compaction_id: &str,
        status: CompactionStatus,
    ) -> Result<(), String> {
        if self.disabled {
            return Ok(());
        }
        if !self.open_compactions.contains(compaction_id) {
            return Err(format!("unknown semantic-edge compaction: {compaction_id}"));
        }
        self.append(SemanticEdgeLedgerEvent::CompactionFinished {
            compaction_id: compaction_id.to_string(),
            status,
        });
        Ok(())
    }

    /// The parent claims a successfully returned child; failed or cancelled children never return.
    pub fn record_child_returned(
        &mut self,
        child_session_id: &str,
        child_last_committed_request_id: Option<&str>,
    ) {
        let Some(child_last_committed_request_id) = child_last_committed_request_id else {
            return;
        };
        self.append(SemanticEdgeLedgerEvent::ChildReturned {
            session_id: self.session_id.clone(),
            child_session_id: child_session_id.to_string(),
            request_id: child_last_committed_request_id.to_string(),
        });
    }

    fn replay(&mut self, event: &SemanticEdgeLedgerEvent) {
        match event {
            SemanticEdgeLedgerEvent::RequestStarted {
                request_id,
                session_id,
                compaction_id,
            } => {
                // Restores spawn attribution after resume; the body hash is unknowable, so
                // a parked retry from a replayed request can never be reused.
                if session_id == &self.session_id && compaction_id.is_none() {
                    self.last_turn = Some(LastTurn {
                        request_id: request_id.clone(),
                        epoch: self.epoch,
                        body_hash: None,
                    });
                }
            }
            SemanticEdgeLedgerEvent::RequestFinished { request_id } => {
                self.last_committed_request_id = Some(request_id.clone());
            }
            SemanticEdgeLedgerEvent::CompactionBegun {
                compaction_id,
                session_id,
            } => {
                if session_id == &self.session_id {
                    self.open_compactions.insert(compaction_id.clone());
                }
            }
            SemanticEdgeLedgerEvent::CompactionFinished {
                compaction_id,
                status,
            } => {
                if self.open_compactions.remove(compaction_id)
                    && status == COMPACTION_STATUS_COMPLETED
                {
                    self.epoch += 1;
                }
            }
            _ => {}
        }
    }

    // Durable append first, in-memory state second: a failed write must not leave
    // commit state pointing at events that never reached the ledger.
    fn append(&mut self, event: SemanticEdgeLedgerEvent) -> bool {
        if self.disabled {
            return false;
        }
        if let Some(event_log) = &self.event_log {
            let value = serde_json::to_value(&event).unwrap_or(Value::Null);
            if let Err(error) = event_log.append_sync(&[value], false, None) {
                self.disable(&error);
                return false;
            }
        }
        self.replay(&event);
        true
    }
}

fn parse_semantic_edge_line(
    line: &str,
    index: usize,
) -> Result<Option<SemanticEdgeLedgerEvent>, String> {
    match serde_json::from_str::<SemanticEdgeLedgerEvent>(line) {
        Ok(event) => Ok(Some(event)),
        Err(error) => Err(format!(
            "corrupt semantic-edge ledger line {}: {}",
            index + 1,
            error
        )),
    }
}

pub fn read_semantic_edge_ledger(path: &str) -> Result<Vec<SemanticEdgeLedgerEvent>, String> {
    // A missing ledger stays loud for explicit readers; the recorder treats absence as empty.
    EventLog::new(path.to_string(), EventLogOptions::default()).replay_sync(
        parse_semantic_edge_line,
        ReplayOptions {
            missing_file_throws: true,
        },
    )
}

#[derive(Debug, Clone, Default)]
struct FoldSession {
    spawned_by_request_id: Option<String>,
    spawn_claimed: bool,
    last_request_id: Option<String>,
    pending: Vec<FoldPending>,
}

#[derive(Debug, Clone, PartialEq)]
struct FoldPending {
    source: String,
    type_: SemanticEdgeType,
}

#[derive(Debug, Clone)]
struct InFlightRequest {
    session_id: String,
    summary: bool,
    inbound: Vec<FoldPending>,
}

#[derive(Debug, Clone)]
struct FoldCompaction {
    session_id: String,
    summary_request_ids: BTreeSet<String>,
}

/// Pure fold from one-or-more ledgers to the semantic edge set. Each event
/// carries the session it acts on, and child returns are recorded in the
/// parent's ledger with the child's last committed request, so ledgers can be
/// folded in any order without cross-ledger sequencing.
pub fn derive_semantic_edges(ledgers: &[Vec<SemanticEdgeLedgerEvent>]) -> SemanticEdgesResult {
    let mut sessions: BTreeMap<String, FoldSession> = BTreeMap::new();
    let mut in_flight: BTreeMap<String, InFlightRequest> = BTreeMap::new();
    let mut compactions: BTreeMap<String, FoldCompaction> = BTreeMap::new();
    let mut returned_children: BTreeSet<String> = BTreeSet::new();
    let mut edges: Vec<SemanticEdge> = Vec::new();

    for event in ledgers.iter().flatten() {
        match event {
            SemanticEdgeLedgerEvent::SessionRegistered {
                session_id,
                spawned_by_request_id,
                ..
            } => {
                sessions
                    .entry(session_id.clone())
                    .or_insert_with(|| FoldSession {
                        spawned_by_request_id: spawned_by_request_id.clone(),
                        spawn_claimed: false,
                        last_request_id: None,
                        pending: Vec::new(),
                    });
            }
            SemanticEdgeLedgerEvent::RequestStarted {
                request_id,
                session_id,
                compaction_id,
            } => {
                let is_summary = compaction_id.is_some();
                // Summary slices claim no pending edges and no spawn: pending defers to
                // the post-compaction turn, the request that actually consumes it.
                let mut inbound: Vec<FoldPending> = Vec::new();
                {
                    let state = sessions.entry(session_id.clone()).or_default();
                    if !is_summary {
                        inbound = std::mem::take(&mut state.pending);
                        if !state.spawn_claimed && state.spawned_by_request_id.is_some() {
                            inbound.push(FoldPending {
                                source: state.spawned_by_request_id.clone().unwrap_or_default(),
                                type_: SEMANTIC_EDGE_SUBAGENT_CALL.to_string(),
                            });
                            state.spawn_claimed = true;
                        }
                    }
                    if let Some(last_request_id) = &state.last_request_id {
                        if !inbound.iter().any(|edge| &edge.source == last_request_id) {
                            inbound.push(FoldPending {
                                source: last_request_id.clone(),
                                type_: SEMANTIC_EDGE_CONTINUATION.to_string(),
                            });
                        }
                    }
                }
                in_flight.insert(
                    request_id.clone(),
                    InFlightRequest {
                        session_id: session_id.clone(),
                        summary: is_summary,
                        inbound,
                    },
                );
                if is_summary {
                    if let Some(compaction_id) = compaction_id {
                        if let Some(compaction) = compactions.get_mut(compaction_id) {
                            // A split-turn compaction sends several summary slices; all belong to it.
                            if &compaction.session_id == session_id {
                                compaction.summary_request_ids.insert(request_id.clone());
                            }
                        }
                    }
                }
            }
            SemanticEdgeLedgerEvent::RequestFinished { request_id } => {
                let Some(request) = in_flight.remove(request_id) else {
                    continue;
                };
                for inbound in &request.inbound {
                    edges.push(SemanticEdge {
                        source_request_id: inbound.source.clone(),
                        target_request_id: request_id.clone(),
                        type_: inbound.type_.clone(),
                    });
                }
                sessions
                    .entry(request.session_id)
                    .or_default()
                    .last_request_id = Some(request_id.clone());
            }
            SemanticEdgeLedgerEvent::RequestFailed { request_id } => {
                let Some(request) = in_flight.remove(request_id) else {
                    continue;
                };
                // A failed summary slice returns nothing: it claimed nothing, and its
                // continuation regenerates from the unchanged last commit (several
                // failed slices would otherwise requeue duplicate continuations).
                if !request.summary {
                    let state = sessions.entry(request.session_id).or_default();
                    state.pending = request
                        .inbound
                        .iter()
                        .cloned()
                        .chain(state.pending.iter().cloned())
                        .collect();
                }
            }
            SemanticEdgeLedgerEvent::CompactionBegun {
                compaction_id,
                session_id,
            } => {
                compactions.insert(
                    compaction_id.clone(),
                    FoldCompaction {
                        session_id: session_id.clone(),
                        summary_request_ids: BTreeSet::new(),
                    },
                );
            }
            SemanticEdgeLedgerEvent::CompactionFinished {
                compaction_id,
                status,
            } => {
                let compaction = compactions.remove(compaction_id);
                let Some(compaction) = compaction else {
                    continue;
                };
                if status != COMPACTION_STATUS_COMPLETED {
                    continue;
                }
                let state = sessions.entry(compaction.session_id.clone()).or_default();
                // The session's last commit must be one of the compaction's summary slices;
                // an interrupted or extension-supplied compaction produces no edge.
                let last_request_id = state.last_request_id.clone();
                if let Some(slice) = last_request_id {
                    if compaction.summary_request_ids.contains(&slice) {
                        // nano's source-only suppression, applied at flush time: a pending edge
                        // from X replaces the slice's generated continuation from X, whatever
                        // the pending edge's type, so the flush can never emit a duplicate.
                        let sources: BTreeSet<String> = state
                            .pending
                            .iter()
                            .map(|edge| edge.source.clone())
                            .collect();
                        for source in sources {
                            if let Some(position) = edges.iter().position(|edge| {
                                edge.source_request_id == source
                                    && edge.target_request_id == slice
                                    && edge.type_ == SEMANTIC_EDGE_CONTINUATION
                            }) {
                                edges.remove(position);
                            }
                        }
                        // Terminal flush: deliver deferred pending edges to the committed slice
                        // now, so they survive even when the session never runs another turn.
                        for pending in &state.pending {
                            edges.push(SemanticEdge {
                                source_request_id: pending.source.clone(),
                                target_request_id: slice.clone(),
                                type_: pending.type_.clone(),
                            });
                        }
                        state.pending = vec![FoldPending {
                            source: slice,
                            type_: SEMANTIC_EDGE_COMPACTION.to_string(),
                        }];
                    }
                }
            }
            SemanticEdgeLedgerEvent::ChildReturned {
                session_id,
                child_session_id,
                request_id,
            } => {
                if returned_children.contains(child_session_id) {
                    continue;
                }
                returned_children.insert(child_session_id.clone());
                sessions
                    .entry(session_id.clone())
                    .or_default()
                    .pending
                    .push(FoldPending {
                        source: request_id.clone(),
                        type_: SEMANTIC_EDGE_SUBAGENT_RETURN.to_string(),
                    });
            }
        }
    }

    SemanticEdgesResult { edges }
}

/// `Symbol.for("prime-agent.semantic-edges.inner-stream-fn")` has no Rust
/// equivalent on a trait object, so wrapped functions are keyed in a
/// process-local registry by the address of the wrapped allocation. The entry is
/// removed when that allocation is dropped, so a key can never outlive its value.
type InnerStreamFnRegistry = Mutex<HashMap<usize, pi_agent_core::types::StreamFn>>;

fn inner_stream_fn_registry() -> &'static InnerStreamFnRegistry {
    static REGISTRY: std::sync::OnceLock<InnerStreamFnRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Holds the registry key of one wrapped stream function and removes it on drop.
struct InnerStreamFnRegistration {
    key: Arc<std::sync::OnceLock<usize>>,
}

impl Drop for InnerStreamFnRegistration {
    fn drop(&mut self) {
        if let Some(key) = self.key.get() {
            inner_stream_fn_registry().lock().unwrap().remove(key);
        }
    }
}

/// Unwrap a semantic-edge-bound stream function; aux calls outside session history use this.
pub fn unwrap_semantic_edge_stream_fn(
    stream_fn: &pi_agent_core::types::StreamFn,
) -> pi_agent_core::types::StreamFn {
    let key = Arc::as_ptr(stream_fn) as *const () as usize;
    inner_stream_fn_registry()
        .lock()
        .unwrap()
        .get(&key)
        .cloned()
        .unwrap_or_else(|| stream_fn.clone())
}

// Captured before awaiting a deferred stream, so cancellation/panic before
// resolution also records failure. A completed result still commits if the
// observer is cancelled after the terminal event but before it was scheduled.
struct SemanticRequestCompletion {
    recorder: Arc<Mutex<SemanticEdgeRecorder>>,
    request_id: Option<String>,
    stream: Option<pi_ai::utils::event_stream::AssistantMessageEventStream>,
}

impl SemanticRequestCompletion {
    fn record(&mut self) {
        let Some(request_id) = self.request_id.take() else { return; };
        let result = self.stream.as_ref().and_then(|stream| stream.result_if_ready());
        let mut recorder = self.recorder.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        match result {
            Some(message) if message.stop_reason != pi_ai::types::STOP_REASON_ERROR
                && message.stop_reason != pi_ai::types::STOP_REASON_ABORTED => {
                recorder.finish_request(Some(&request_id));
            }
            _ => recorder.fail_request(Some(&request_id)),
        }
    }
}

impl Drop for SemanticRequestCompletion {
    fn drop(&mut self) {
        self.record();
    }
}

/// Bind a stream function to one session's recorder. Re-wrapping an already
/// wrapped function rebinds the original, so a child session that inherits its
/// parent's streamFn attributes calls to its own ledger. request_started is
/// appended before the wire call; the request commits or fails when its stream
/// resolves (an error/aborted final message is a failure). When the recorder is
/// disabled (its ledger failed), calls carry no request ID at all.
pub fn wrap_stream_fn_with_semantic_edges(
    stream_fn: pi_agent_core::types::StreamFn,
    recorder: Arc<Mutex<SemanticEdgeRecorder>>,
) -> pi_agent_core::types::StreamFn {
    let inner = unwrap_semantic_edge_stream_fn(&stream_fn);
    // The registry keeps its own handle: the closure below moves `inner` too.
    let inner_for_registry = inner.clone();
    let key = Arc::new(std::sync::OnceLock::new());
    let registration = InnerStreamFnRegistration { key: key.clone() };
    let wrapped: pi_agent_core::types::StreamFn = Arc::new(move |model, context, options| {
        let _keep_registration = &registration;
        let request_id = recorder
            .lock()
            .unwrap()
            .start_turn_request(Some(hash_turn_body(&model, &context, Some(&options))));
        let Some(request_id) = request_id else {
            return inner(model, context, options);
        };
        let mut call_options = options.clone();
        let headers = call_options
            .stream
            .headers
            .get_or_insert_with(indexmap::IndexMap::new);
        for (header, value) in model_request_headers(&request_id) {
            if let Value::String(value) = value {
                headers.insert(header, value);
            }
        }
        let stream_future = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            inner(model, context, call_options)
        })) {
            Ok(stream) => stream,
            Err(payload) => {
                recorder.lock().unwrap().fail_request(Some(&request_id));
                std::panic::resume_unwind(payload);
            }
        };
        let mut completion = SemanticRequestCompletion {
            recorder: recorder.clone(),
            request_id: Some(request_id),
            stream: None,
        };
        Box::pin(async move {
            let stream = stream_future.await;
            completion.stream = Some(stream.producer_handle());
            let observed = stream.producer_handle();
            let observe = async move {
                // end(None) must not strand a result observer indefinitely.
                observed.result_or_end().await;
                completion.record();
            };
            if tokio::runtime::Handle::try_current().is_ok() {
                stream.spawn(observe);
            }
            stream
        })
    });
    let address = Arc::as_ptr(&wrapped) as *const () as usize;
    let _ = key.set(address);
    inner_stream_fn_registry()
        .lock()
        .unwrap()
        .insert(address, inner_for_registry);
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn started(
        request_id: &str,
        session_id: &str,
        compaction_id: Option<&str>,
    ) -> SemanticEdgeLedgerEvent {
        SemanticEdgeLedgerEvent::RequestStarted {
            request_id: request_id.to_string(),
            session_id: session_id.to_string(),
            compaction_id: compaction_id.map(str::to_string),
        }
    }

    fn finished(request_id: &str) -> SemanticEdgeLedgerEvent {
        SemanticEdgeLedgerEvent::RequestFinished {
            request_id: request_id.to_string(),
        }
    }

    fn failed(request_id: &str) -> SemanticEdgeLedgerEvent {
        SemanticEdgeLedgerEvent::RequestFailed {
            request_id: request_id.to_string(),
        }
    }

    fn registered(session_id: &str, spawned_by: Option<&str>) -> SemanticEdgeLedgerEvent {
        SemanticEdgeLedgerEvent::SessionRegistered {
            session_id: session_id.to_string(),
            parent_session_id: None,
            spawned_by_request_id: spawned_by.map(str::to_string),
        }
    }

    #[test]
    fn consecutive_commits_produce_continuation_edges() {
        let ledger = vec![
            registered("s", None),
            started("r1", "s", None),
            finished("r1"),
            started("r2", "s", None),
            finished("r2"),
        ];
        let edges = derive_semantic_edges(&[ledger]).edges;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_request_id, "r1");
        assert_eq!(edges[0].target_request_id, "r2");
        assert_eq!(edges[0].type_, "continuation");
    }

    #[test]
    fn spawn_attribution_becomes_a_subagent_call_edge() {
        let ledger = vec![
            registered("child", Some("parent-request")),
            started("r1", "child", None),
            finished("r1"),
        ];
        let edges = derive_semantic_edges(&[ledger]).edges;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_request_id, "parent-request");
        assert_eq!(edges[0].type_, "subagent_call");
    }

    #[test]
    fn failed_requests_return_their_inbound_edges_to_pending() {
        let ledger = vec![
            registered("s", None),
            started("r1", "s", None),
            finished("r1"),
            started("r2", "s", None),
            failed("r2"),
            started("r3", "s", None),
            finished("r3"),
        ];
        let edges = derive_semantic_edges(&[ledger]).edges;
        let pairs: Vec<(&str, &str)> = edges
            .iter()
            .map(|edge| {
                (
                    edge.source_request_id.as_str(),
                    edge.target_request_id.as_str(),
                )
            })
            .collect();
        assert_eq!(pairs, vec![("r1", "r3")]);
    }

    #[test]
    fn failed_summary_slices_claim_nothing() {
        let ledger = vec![
            registered("s", None),
            SemanticEdgeLedgerEvent::CompactionBegun {
                compaction_id: "c1".to_string(),
                session_id: "s".to_string(),
            },
            started("r1", "s", None),
            finished("r1"),
            started("summary", "s", Some("c1")),
            failed("summary"),
            started("r2", "s", None),
            finished("r2"),
        ];
        let edges = derive_semantic_edges(&[ledger]).edges;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_request_id, "r1");
        assert_eq!(edges[0].target_request_id, "r2");
    }

    #[test]
    fn completed_compaction_flushes_pending_to_the_last_slice() {
        let mut ledger = vec![
            registered("s", None),
            started("r1", "s", None),
            finished("r1"),
            started("failed-turn", "s", None),
            failed("failed-turn"),
            SemanticEdgeLedgerEvent::CompactionBegun {
                compaction_id: "c1".to_string(),
                session_id: "s".to_string(),
            },
            started("summary-a", "s", Some("c1")),
            finished("summary-a"),
            started("summary-b", "s", Some("c1")),
            finished("summary-b"),
            SemanticEdgeLedgerEvent::CompactionFinished {
                compaction_id: "c1".to_string(),
                status: "completed".to_string(),
            },
        ];
        let edges = derive_semantic_edges(&[ledger.clone()]).edges;
        // r1 -> summary-a (continuation), summary-a -> summary-b (continuation),
        // then the completed compaction flushes the deferred pending r1 edge onto
        // summary-b and replaces it with the compaction edge.
        let pairs: Vec<(&str, &str, &str)> = edges
            .iter()
            .map(|edge| {
                (
                    edge.source_request_id.as_str(),
                    edge.target_request_id.as_str(),
                    edge.type_.as_str(),
                )
            })
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("r1", "summary-a", "continuation"),
                ("summary-a", "summary-b", "continuation"),
                ("r1", "summary-b", "continuation"),
            ]
        );
        // The outbound compaction edge is pending until its target commits.
        assert!(!edges.iter().any(|edge| edge.type_ == "compaction"));
        ledger.extend([started("next-turn", "s", None), finished("next-turn")]);
        let edges = derive_semantic_edges(&[ledger]).edges;
        assert_eq!(edges.last(), Some(&SemanticEdge {
            source_request_id: "summary-b".to_string(),
            target_request_id: "next-turn".to_string(),
            type_: "compaction".to_string(),
        }));
    }

    #[test]
    fn child_returns_are_claimed_once() {
        let ledger = vec![
            registered("parent", None),
            SemanticEdgeLedgerEvent::ChildReturned {
                session_id: "parent".to_string(),
                child_session_id: "child".to_string(),
                request_id: "child-r1".to_string(),
            },
            SemanticEdgeLedgerEvent::ChildReturned {
                session_id: "parent".to_string(),
                child_session_id: "child".to_string(),
                request_id: "child-r1".to_string(),
            },
            started("r1", "parent", None),
            finished("r1"),
        ];
        let edges = derive_semantic_edges(&[ledger]).edges;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_request_id, "child-r1");
        assert_eq!(edges[0].type_, "subagent_return");
    }

    #[test]
    fn unknown_request_ids_are_ignored() {
        let ledger = vec![finished("never-started"), failed("never-started")];
        assert!(derive_semantic_edges(&[ledger]).edges.is_empty());
    }

    #[test]
    fn headers_and_ledger_paths_match_the_wire_contract() {
        let headers = model_request_headers("req-1");
        assert_eq!(
            headers.get("X-ACP-Model-Request-ID").unwrap(),
            &json!("req-1")
        );
        assert_eq!(headers.get("Idempotency-Key").unwrap(), &json!("req-1"));
        assert_eq!(
            semantic_edge_ledger_path(Some("C:/sessions/sub-1"), None).unwrap(),
            std::path::Path::new("C:/sessions/sub-1")
                .join("semantic-edges.jsonl")
                .to_string_lossy()
                .to_string()
        );
        assert_eq!(
            semantic_edge_ledger_path(None, Some("C:/artifacts")).unwrap(),
            std::path::Path::new("C:/artifacts")
                .join("semantic-edges.jsonl")
                .to_string_lossy()
                .to_string()
        );
        assert!(semantic_edge_ledger_path(None, None).is_none());
    }

    #[test]
    fn hash_turn_body_is_stable_and_body_sensitive() {
        let model = pi_ai::types::Model::new(
            "gpt-5",
            "GPT-5",
            "openai-responses",
            "openai",
            "https://api.openai.com/v1",
        );
        let context = pi_ai::types::Context {
            system_prompt: Some("sys".to_string()),
            messages: vec![pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
                pi_ai::types::UserContent::Text("hi".to_string()),
                1,
            ))],
            tools: None,
        };
        let first = hash_turn_body(&model, &context, None);
        let second = hash_turn_body(&model, &context, None);
        assert_eq!(first, second);
        let mut other = context.clone();
        other.system_prompt = Some("other".to_string());
        assert_ne!(first, hash_turn_body(&model, &other, None));
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn recorder_registers_once_and_replays_on_resume() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join("semantic-edges.jsonl")
            .to_string_lossy()
            .to_string();
        {
            let mut recorder = SemanticEdgeRecorder::new(
                Some(path.clone()),
                "s".to_string(),
                Some("p".to_string()),
                None,
            );
            let request_id = recorder
                .start_turn_request(Some("hash".to_string()))
                .unwrap();
            recorder.finish_request(Some(&request_id));
            assert_eq!(
                recorder.last_committed_request_id().as_deref(),
                Some(request_id.as_str())
            );
            assert_eq!(
                recorder.last_turn_request_id().as_deref(),
                Some(request_id.as_str())
            );
        }
        let events = read_semantic_edge_ledger(&path).unwrap();
        assert_eq!(events.len(), 3);
        match &events[0] {
            SemanticEdgeLedgerEvent::SessionRegistered {
                session_id,
                parent_session_id,
                spawned_by_request_id,
            } => {
                assert_eq!(session_id, "s");
                assert_eq!(parent_session_id.as_deref(), Some("p"));
                assert!(spawned_by_request_id.is_none());
            }
            other => panic!("unexpected event {other:?}"),
        }

        let mut resumed =
            SemanticEdgeRecorder::new(Some(path.clone()), "s".to_string(), None, None);
        assert_eq!(
            resumed.last_committed_request_id(),
            Some(events[2].clone().into_request_id().unwrap())
        );
        assert!(resumed.last_turn_request_id().is_some());
        // A replayed request has no body hash, so a parked retry can never reuse it.
        resumed.prepare_turn_retry();
        let reused = resumed
            .start_turn_request(Some("hash".to_string()))
            .unwrap();
        assert_ne!(reused, events[2].clone().into_request_id().unwrap());
    }

    #[test]
    fn parked_retry_reuses_the_same_id_for_the_same_body() {
        let mut recorder = SemanticEdgeRecorder::new(None, "s".to_string(), None, None);
        let first = recorder
            .start_turn_request(Some("hash".to_string()))
            .unwrap();
        recorder.prepare_turn_retry();
        let second = recorder
            .start_turn_request(Some("hash".to_string()))
            .unwrap();
        assert_eq!(first, second);
        recorder.prepare_turn_retry();
        let third = recorder
            .start_turn_request(Some("other".to_string()))
            .unwrap();
        assert_ne!(first, third);
        recorder.prepare_turn_retry();
        recorder.clear_turn_retry();
        let fourth = recorder
            .start_turn_request(Some("other".to_string()))
            .unwrap();
        assert_ne!(third, fourth);
    }

    #[test]
    fn compaction_requests_and_finish_validate_ids() {
        let mut recorder = SemanticEdgeRecorder::new(None, "s".to_string(), None, None);
        let compaction_id = recorder.begin_compaction();
        let request_id = recorder.start_compaction_request(&compaction_id).unwrap();
        recorder.finish_request(Some(&request_id));
        assert!(recorder
            .finish_compaction(&compaction_id, "completed".to_string())
            .is_ok());
        assert_eq!(
            recorder
                .finish_compaction(&compaction_id, "completed".to_string())
                .unwrap_err(),
            format!("unknown semantic-edge compaction: {compaction_id}")
        );
        assert!(recorder
            .finish_compaction("missing", "failed".to_string())
            .is_err());
    }

    #[test]
    fn disabled_recorder_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        // A directory in place of the ledger file makes the first append fail.
        let path = dir.path().to_string_lossy().to_string();
        let mut recorder = SemanticEdgeRecorder::new(Some(path), "s".to_string(), None, None);
        assert!(recorder.start_turn_request(None).is_none());
        assert!(recorder
            .finish_compaction("x", "failed".to_string())
            .is_ok());
    }

    #[test]
    fn child_returns_require_a_committed_child_request() {
        let mut recorder = SemanticEdgeRecorder::new(None, "s".to_string(), None, None);
        recorder.record_child_returned("child", None);
        assert!(recorder.last_committed_request_id().is_none());
        recorder.record_child_returned("child", Some("child-r1"));
        // No ledger means nothing is written, but the call must not panic.
        assert!(recorder.last_turn_request_id().is_none());
    }
}


#[cfg(test)]
mod settlement_tests {
    use super::*;
    use std::time::Duration;
    use futures::FutureExt;
    use pi_ai::types::{AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions};
    use pi_ai::utils::event_stream::{AssistantMessageEventStream, StreamTaskReceipt};

    fn recorder() -> (tempfile::TempDir, Arc<Mutex<SemanticEdgeRecorder>>, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("semantic-edges.jsonl").to_string_lossy().to_string();
        let recorder = Arc::new(Mutex::new(SemanticEdgeRecorder::new(Some(path.clone()), "fixture-session".into(), None, None)));
        (dir, recorder, path)
    }

    fn assert_terminal(path: &str, success: bool) {
        let events = read_semantic_edge_ledger(path).unwrap();
        let finished = events.iter().filter(|e| matches!(e, SemanticEdgeLedgerEvent::RequestFinished { .. })).count();
        let failed = events.iter().filter(|e| matches!(e, SemanticEdgeLedgerEvent::RequestFailed { .. })).count();
        assert_eq!((finished, failed), if success { (1, 0) } else { (0, 1) });
    }

    #[tokio::test]
    async fn semantic_observer_is_joined_with_provider_and_preserves_commit() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            let stream = AssistantMessageEventStream::new_owned();
            let producer = stream.producer_handle();
            stream.spawn(async move {
                producer.push(AssistantMessageEvent::Done { reason: "stop".into(), message: AssistantMessage::default() });
            });
            Box::pin(async move { stream })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder.clone());
        let receipt = StreamTaskReceipt::new_unsupported();
        let stream = receipt.scope(async { wrapped(Model::default(), Context::default(), SimpleStreamOptions::default()).await }).await;
        stream.result().await;
        let done = receipt.settle(Duration::from_secs(1)).await;
        assert!(done.supported);
        assert_eq!(done.completed_tasks, 2);
        assert_eq!(done.pending_tasks, 0);
        assert_terminal(&path, true);
        assert!(recorder.lock().unwrap().last_committed_request_id().is_some());
    }

    #[tokio::test]
    async fn semantic_stop_records_failure_before_join_acknowledgement() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            let stream = AssistantMessageEventStream::new_owned();
            stream.spawn(std::future::pending());
            Box::pin(async move { stream })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder.clone());
        let stream = wrapped(Model::default(), Context::default(), SimpleStreamOptions::default()).await;
        let receipt = stream.task_receipt();
        stream.request_cancel();
        let done = receipt.settle(Duration::from_secs(1)).await;
        assert!(done.settled, "{done:?}");
        assert_eq!(done.cancelled_tasks, 2);
        assert_terminal(&path, false);
        assert!(recorder.lock().unwrap().last_committed_request_id().is_none());
    }

    #[tokio::test]
    async fn end_without_result_does_not_strand_the_semantic_observer() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            let stream = AssistantMessageEventStream::new_owned();
            stream.end(None);
            Box::pin(async move { stream })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder);
        let stream = wrapped(Model::default(), Context::default(), SimpleStreamOptions::default()).await;
        let done = stream.task_receipt().settle(Duration::from_secs(1)).await;
        assert_eq!(done.pending_tasks, 0);
        assert_eq!(done.completed_tasks, 1);
        assert_terminal(&path, false);
    }

    #[tokio::test]
    async fn stopped_unscheduled_observer_preserves_already_completed_result() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            let stream = AssistantMessageEventStream::new_owned();
            stream.end(Some(AssistantMessage::default()));
            Box::pin(async move { stream })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder);
        let stream = wrapped(Model::default(), Context::default(), SimpleStreamOptions::default()).await;
        stream.request_cancel();
        assert!(stream.task_receipt().settle(Duration::from_secs(1)).await.settled);
        assert_terminal(&path, true);
    }

    #[tokio::test]
    async fn cancelled_deferred_wrapper_keeps_created_producer_in_scope() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            let stream = AssistantMessageEventStream::new_owned();
            stream.spawn(std::future::pending());
            Box::pin(async move {
                std::future::pending::<()>().await;
                stream
            })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder);
        let receipt = StreamTaskReceipt::new_unsupported();
        let mut invocation = Box::pin(receipt.scope(async {
            wrapped(Model::default(), Context::default(), SimpleStreamOptions::default()).await
        }));
        assert!(futures::poll!(invocation.as_mut()).is_pending());
        assert_eq!(receipt.status().pending_tasks, 1);
        receipt.request_cancel();
        drop(invocation);
        assert!(receipt.settle(Duration::from_secs(1)).await.settled);
        assert_terminal(&path, false);
    }

    #[tokio::test]
    async fn deferred_wrapper_panic_fails_the_ledger_instead_of_leaving_inflight() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            Box::pin(async { panic!("synthetic deferred wrapper failure") })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder);
        let result = std::panic::AssertUnwindSafe(wrapped(Model::default(), Context::default(), SimpleStreamOptions::default())).catch_unwind().await;
        assert!(result.is_err());
        assert_terminal(&path, false);
    }

    #[tokio::test]
    async fn semantic_wrapper_cannot_certify_unknown_backend() {
        let (_dir, recorder, path) = recorder();
        let fake: pi_agent_core::types::StreamFn = Arc::new(|_, _, _| {
            let stream = AssistantMessageEventStream::new();
            stream.end(Some(AssistantMessage::default()));
            Box::pin(async move { stream })
        });
        let wrapped = wrap_stream_fn_with_semantic_edges(fake, recorder);
        let stream = wrapped(Model::default(), Context::default(), SimpleStreamOptions::default()).await;
        let receipt = stream.task_receipt();
        receipt.settle(Duration::from_secs(1)).await;
        assert!(!receipt.request_cancel().supported);
        assert!(!receipt.status().settled);
        assert_terminal(&path, true);
    }
}

impl SemanticEdgeLedgerEvent {
    #[cfg(test)]
    fn into_request_id(self) -> Option<String> {
        match self {
            SemanticEdgeLedgerEvent::RequestStarted { request_id, .. }
            | SemanticEdgeLedgerEvent::RequestFinished { request_id }
            | SemanticEdgeLedgerEvent::RequestFailed { request_id } => Some(request_id),
            _ => None,
        }
    }
}
