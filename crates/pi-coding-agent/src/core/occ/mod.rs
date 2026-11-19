//! Optimus Compact Communication (OCC).
//!
//! OCC is an OPTIONAL communication framework layered on the existing
//! daemon transport. The transport stays authoritative: sender identity is
//! established by the daemon, family scoping and reach restrictions are
//! unchanged, and plain English remains the default. OCC levels are
//! interchangeable encodings of one typed payload, not separate agent
//! lifecycles:
//!
//! - OCC-0: ordinary English (always available; the default for
//!   conversational prose).
//! - OCC-1: compact structured English ([`occ1`]) encoding the typed contract
//!   ([`contract`]); the default for structured task and result bodies.
//! - OCC-2: symbolic dialects ([`occ2`] + [`dialect`]): a table-driven
//!   symbolic codec for the same typed contract, with strict decode and a
//!   conservative OCC-1 fallback. The codec, registry and the
//!   hand-designed `sym-v1` dialect are implemented; the offline
//!   discovery harness that evolves dialects is future work.
//! - OCC-3: true latent communication between compatible local models (not
//!   implemented; hosted black-box APIs cannot support it).
//!
//! Modules:
//! - [`occ1`]: the OCC-1 line grammar codec (implemented here; frozen API).
//! - [`occ2`]: the OCC-2 symbolic codec for the same typed contract
//!   (implemented here; foundation for learned dialects).
//! - [`dialect`]: the OCC-2 dialect registry: dialects as data, codebook
//!   hashing and capability negotiation (implemented here; foundation).
//! - [`dialect_load`]: the DIALECT/1 research-card promotion path: load,
//!   checksum-verify and translate frozen research cards into registry
//!   cards behind an explicit, non-default constructor.
//! - [`selection`]: deterministic OCC-1 vs OCC-2 selection: a measured
//!   policy (codebook amortization, message size, strike ledger,
//!   evaluation records) plus a validated-registry constructor.
//! - [`contract`]: the `optimus.agent-comm.v1` typed task/result contract
//!   (implemented here; frozen API).
//! - `delta`: change-only updates on top of the contract (WORKER B owns this
//!   file; the public API below is FROZEN).
//! - `consolidation`: MOC-inspired evidence consolidation (WORKER C; FROZEN).
//! - `prune`: AgentPrune-inspired redundant-message suppression (WORKER C;
//!   FROZEN).
//!
//! Campaign rules for all code in this tree:
//! - Pure, deterministic, offline-testable logic. No network, no model APIs,
//!   no filesystem access inside these modules.
//! - Never silently drop contradictory evidence or uncertainty.
//! - Never invent missing required values; reject instead.
//! - All iteration is over ordered collections (BTreeMap/BTreeSet/Vec), so
//!   results are reproducible regardless of hash seeds.
//!
//! ---------------------------------------------------------------
//! FROZEN PUBLIC API SPEC for `delta.rs` (worker B implements exactly):
//!
//! ```text
//! pub struct Delta {
//!     pub task_id: String,
//!     pub base_sequence: u64,
//!     pub sequence: u64,
//!     pub updates: Vec<FieldUpdate>,
//!     pub new_findings: Vec<Finding>,
//!     pub new_evidence: Vec<EvidenceRef>,
//!     pub next_actions: Vec<String>,
//!     pub reason: Option<String>,
//! }
//!
//! pub enum FieldUpdate {
//!     SetStatus(String),
//!     UpdateFinding { id: String, patch: FindingPatch },
//!     AddEvidence(EvidenceRef),
//!     SetVerification(Verification),
//! }
//!
//! pub struct FindingPatch {
//!     pub statement: Option<String>,
//!     pub confidence: Option<Option<f32>>,
//!     pub add_evidence: Vec<String>,
//!     pub remove_evidence: Vec<String>,
//! }
//!
//! pub enum DeltaError {
//!     WrongTask { expected: String, got: String },
//!     Duplicate { sequence: u64 },
//!     Gap { expected: u64, got: u64 },
//!     StaleBase { base: u64, got_base: u64 },
//!     UnknownFinding { id: String },
//!     UnknownEvidence { id: String },
//!     BaseUnavailable,
//! }
//!
//! pub struct DeltaTracker { /* private state */ }
//!
//! impl DeltaTracker {
//!     pub fn new(task_id: impl Into<String>, base_sequence: u64) -> Self;
//!     pub fn task_id(&self) -> &str;
//!     pub fn next_sequence(&self) -> u64;
//!     /// Records the delta: rejects wrong task, duplicate replay, gaps and
//!     /// stale bases; returns what was applied so callers can update their
//!     /// own snapshot state.
//!     pub fn observe(&mut self, delta: &Delta)
//!         -> Result<DeltaReceipt, DeltaError>;
//!     pub fn missing_sequences(&self) -> Vec<u64>;
//!     pub fn needs_full_snapshot(&self) -> bool;
//! }
//!
//! pub struct DeltaReceipt {
//!     pub applied: bool,   // false when the delta was buffered for a gap
//!     pub buffered: bool,
//!     pub sequence: u64,
//! }
//!
//! impl Delta {
//!     /// OCC-1 wire encoding of a delta (see rules below).
//!     pub fn to_occ1(&self) -> Result<Occ1Message, ContractError>;
//!     /// Strict OCC-1 decode; syntax only, semantics are checked by
//!     /// `DeltaTracker::observe`.
//!     pub fn from_occ1(message: &Occ1Message) -> Result<Self, ContractError>;
//! }
//! ```
//!
//! Delta OCC-1 wire encoding (FROZEN):
//! - `TASK`, `SEQ`, `BASE` are required (`BASE` = base_sequence).
//! - `STATUS=<status>` encodes `SetStatus`.
//! - `UPDATE=<finding-id>:<field>=<value>` encodes one `FindingPatch` item;
//!   `<field>` is `stmt`, `conf`, `ev+<id>` or `ev-<id>`.
//! - `EADD=<evidence-id>` (repeatable, after any `E<n>` entries) encodes
//!   `FieldUpdate::AddEvidence` references to evidence introduced by this
//!   same delta; the referenced id must exist among the delta's new
//!   evidence. Non-positional new-evidence ids therefore cannot
//!   round-trip, matching the finding-id rule.
//! - `NEW<n>=<statement>` encodes new findings (1-based contiguous), with
//!   optional `NEWC<n>=<confidence>` and `NEWE<n>=E1,E2` evidence links.
//! - `E<n>=<kind>:<locator>` with optional `EL<n>`, `ED<n>`, `ER<n>`, `EV<n>`
//!   encode new evidence exactly as in the contract encoding.
//! - `V`/`VS`/`VM`/optional `VR` encode `SetVerification` (all-or-nothing).
//! - `NEXT=<action>` (repeatable) encodes next actions; `WHY=<reason>` is the
//!   optional reason.
//!
//! Semantics (FROZEN):
//! - A delta applies to one task only; a delta for another task is
//!   `WrongTask` and must never be applied.
//! - `base_sequence` identifies the snapshot the delta stream is rooted at;
//!   a tracker built on a different base reports `StaleBase` for the first
//!   delta and `BaseUnavailable` until a full snapshot replaces it.
//! - Sequences must be contiguous from `base_sequence + 1`. Out-of-order
//!   deltas are buffered (not applied) until the gap fills; duplicates are
//!   idempotent no-ops reported as `Duplicate`.
//! - `needs_full_snapshot()` is true while any gap remains unfilled.
//! - Restart recovery: re-observing the same delta stream after a restart
//!   (from a persisted list of applied sequences) yields identical state.
//! - Overflowing buffers is an error, not a crash: cap buffered deltas at
//!   1024 and report `BaseUnavailable` beyond that.
//!
//! ---------------------------------------------------------------
//! FROZEN PUBLIC API SPEC for `consolidation.rs` (worker C implements):
//!
//! ```text
//! pub struct ConsolidationOptions {
//!     pub merge_same_statement: bool,          // default true
//!     pub evidence_key: EvidenceKeyMode,      // default LocatorAndDigest
//! }
//!
//! pub enum EvidenceKeyMode { Locator, LocatorAndDigest, LocatorDigestRevision }
//!
//! pub struct ConsolidationStats {
//!     pub messages_in: usize,
//!     pub findings_in: usize,
//!     pub findings_out: usize,
//!     pub evidence_in: usize,
//!     pub evidence_out: usize,
//!     pub conflicts: usize,
//!     pub uncertainties: usize,
//! }
//!
//! pub struct ConsolidatedFinding {
//!     pub finding: Finding,
//!     pub sources: Vec<String>,   // contributing message ids, ordered
//!     pub agreement: usize,       // number of messages asserting it
//! }
//!
//! pub struct ConflictPosition {
//!     pub statement: String,
//!     pub source_message_id: String,
//!     pub evidence_ids: Vec<String>,
//! }
//!
//! pub struct Conflict {
//!     pub topic: String,
//!     pub positions: Vec<ConflictPosition>,
//! }
//!
//! pub struct ConsolidationReport {
//!     pub findings: Vec<ConsolidatedFinding>,
//!     pub evidence: Vec<EvidenceRef>,
//!     pub conflicts: Vec<Conflict>,
//!     pub uncertainties: Vec<String>,
//!     pub dependencies: Vec<String>,
//!     pub stats: ConsolidationStats,
//! }
//!
//! pub fn consolidate(
//!     messages: &[AgentCommMessage],
//!     options: &ConsolidationOptions,
//! ) -> ConsolidationReport;
//! ```
//!
//! Semantics (FROZEN):
//! - Deterministic matching only: findings merge when their normalized
//!   statements (whitespace-collapsed, lowercase) are equal and
//!   `merge_same_statement` is on; evidence merges by the chosen key mode.
//! - Same statement with DIFFERENT verification statuses or materially
//!   different evidence is a conflict, not a merge: both positions are
//!   preserved verbatim in `conflicts`, never dropped.
//! - Uncertainties (confidence < 0.5, or verification status `unknown`) are
//!   carried into `uncertainties` as human-readable strings; they are
//!   additionally preserved in the findings themselves.
//! - Task dependencies are the ordered union across messages.
//! - Output ordering is deterministic: findings sorted by first-appearance
//!   order, evidence by first appearance, conflicts by first detection.
//! - Inputs are never mutated.
//!
//! ---------------------------------------------------------------
//! FROZEN PUBLIC API SPEC for `prune.rs` (worker C implements):
//!
//! ```text
//! pub struct PrunePolicy {
//!     pub suppress_duplicate_findings: bool,   // default true
//!     pub suppress_unchanged_progress: bool,  // default true
//!     pub suppress_repeated_evidence: bool,   // default true
//!     pub max_history: usize,                 // default 64 messages
//! }
//!
//! pub enum PruneReason {
//!     DuplicateFindings,
//!     UnchangedProgress,
//!     RepeatedEvidence,
//! }
//!
//! pub enum PruneDecision {
//!     Deliver,
//!     Suppress { reason: PruneReason },
//! }
//!
//! pub struct DeliveryRecord {
//!     pub message_id: Option<String>,
//!     pub task_id: String,
//!     pub kind: String,
//!     pub status: String,
//!     pub sequence: Option<u64>,
//!     pub finding_digests: Vec<String>,
//!     pub evidence_digests: Vec<String>,
//! }
//!
//! pub struct Pruner { /* private history */ }
//!
//! impl Pruner {
//!     pub fn new(policy: PrunePolicy) -> Self;
//!     /// Classifies one outgoing message. NEVER suppress: BLOCKED or
//!     /// CANCELLED kinds, FAILED status, CORRECTION or ACK kinds,
//!     /// verification failures, contradictory findings (same normalized
//!     /// statement, different verification status), messages carrying
//!     /// dependencies the recipient has not seen, or the first message
//!     /// for a task. When in doubt: Deliver.
//!     pub fn classify(&mut self, message: &AgentCommMessage) -> PruneDecision;
//! }
//! ```
//!
//! Semantics (FROZEN): safety rules run BEFORE any suppression heuristic.
//! Digests for duplicate detection are stable sha256 hex strings over the
//! canonical content (statement for findings, kind+locator+digest for
//! evidence).

pub mod activity;
pub mod consolidation;
pub mod queue;
pub mod contract;
pub mod delta;
pub mod dialect;
pub mod dialect_load;
pub mod occ1;
pub mod occ2;
pub mod occ3;
pub mod prune;
pub mod selection;
pub mod session_settings;
pub mod textmas;
pub mod textmas_activity;
pub mod textmas_context;
pub mod textmas_settings;

#[cfg(test)]
mod codec_tests;
#[cfg(test)]
mod consolidation_tests;
#[cfg(test)]
mod delta_tests;
#[cfg(test)]
mod prune_tests;
