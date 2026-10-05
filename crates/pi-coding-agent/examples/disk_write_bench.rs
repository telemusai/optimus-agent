//! Standalone disk-write benchmark harness for the REAL durable-write code
//! paths behind coding-file writes and session persistence.
//!
//! Measured through public entry points only (no production file is modified
//! and no new dependency is introduced):
//!   - `pi_coding_agent::utils::atomic_file::write_file_atomic_sync` and
//!     `write_file_atomic` — temp file beside the destination, short-write-safe
//!     write loop, optional fsync, rename with the Windows retry loop, optional
//!     directory fsync. Coding-file sizes 4 KiB / 32 KiB / 256 KiB / 2 MiB,
//!     file counts 1 / 20 / 200, and the option variants real callers pass:
//!     (fsync, fsync_dir) in {(false,false), (true,false), (false,true), (true,true)}
//!     for the sync twin; default and full-durability for the async twin.
//!   - A rewrite-storm point: 200 sequential rewrites of ONE destination
//!     (rename-over-existing each time) at 32 KiB for every option variant.
//!   - `pi_coding_agent::core::session_manager::SessionManager` — the real
//!     JSONL session persistence: the first persisting append rewrites the
//!     whole file through the session manager's atomic stand-in, later turns
//!     append line-by-line through `OpenOptions::append`. Turns 10/100/1000.
//!   - `pi_coding_agent::core::tools::edit::execute_edit` with the default
//!     `LocalEditOperations` write side (`std::fs::write`), observed through a
//!     counting wrapper that only delegates to the real operations.
//!   - `pi_coding_agent::core::event_log::EventLog::append_sync` — the
//!     semantic-edge ledger append used per model request (RequestStarted /
//!     RequestFinished events, one event per append, `durable=false` exactly as
//!     `core::semantic_edges.rs` calls it; one durable=true variant). Ledger
//!     sizes ~4 KiB / 60 KiB / 500 KiB, 20 appends per point. Every measured
//!     append includes the full `repair_tail_sync` probe (metadata + read/write
//!     open + tail byte check; the double full-read path only runs on a torn
//!     tail, which real traffic does not have after a clean append).
//!
//! Determinism: every written byte comes from a fixed-seed generator, so every
//! run writes the SAME bytes. The parity gate re-digests every written file
//! (content + path list) after every pass and exits 1 on any mismatch with the
//! expected digest computed from the generator itself (never from a run).
//! Session files additionally contain run-volatile fields (timestamps, uuid
//! ids); those are normalized before digesting and the raw values are checked
//! structurally (entry count, type/role sequence, parent chain, message
//! content digests, persist-notification count).
//!
//! Instrumentation: counting global allocator (allocs/bytes per op), Windows
//! process I/O counters via `GetProcessIoCounters` (read/write/other operation
//! counts; metadata operations such as create/rename/fsync are NOT included),
//! the `before_rename` seam (one hit per completed temp write), the
//! `on_persist` seam (one hit per persisted session entry), and the
//! `EditOperations` seam (exact read/write call counts for the edit tool).
//!
//! Everything here is measurement tooling: no production file depends on it and
//! it must keep working unchanged BEFORE and AFTER optimization work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use pi_agent_core::types::AgentMessage;
use pi_ai::types::{AssistantMessage, ContentBlock, TextContent, Usage, UserContent, UserMessage};
use pi_coding_agent::core::event_log::{EventLog, EventLogOptions};
use pi_coding_agent::core::semantic_edges::SemanticEdgeLedgerEvent;
use pi_coding_agent::core::session_manager::{NewSessionOptions, SessionManager};
use pi_coding_agent::core::tools::edit::{
	execute_edit, EditOperations, EditToolInput, LocalEditOperations,
};
use pi_coding_agent::core::tools::edit_diff::Edit;
use pi_coding_agent::utils::atomic_file::{
	write_file_atomic, write_file_atomic_sync, WriteFileAtomicAsyncOptions, WriteFileAtomicOptions,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// Edit-tool calls per measured pass (each starts from a reset base file).
const EDIT_REPEATS: usize = 20;
/// Fixed session id so the session file name is deterministic.
const SESSION_ID: &str = "benchsession";
/// Measured appends per event-log pass (one event per append, like the real
/// per-model-request caller: RequestStarted then RequestFinished).
const EVENTLOG_APPENDS: usize = 20;
/// Seed batches use the same real `append_sync`; batching only reduces fixture
/// setup time (the measured appends stay one event per call).
const EVENTLOG_SEED_BATCH: usize = 64;
/// Untimed `create_dir_all` control iterations for the session points.
const SESSION_CREATE_DIR_ALL_CONTROL: usize = 100;

// ---------------------------------------------------------------------------
// Counting global allocator (std-only; gated so timing runs stay clean)
// ---------------------------------------------------------------------------

static ALLOC_GATE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if ALLOC_GATE.load(Ordering::Relaxed) {
			ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
			ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
		}
		System.alloc(layout)
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		System.dealloc(ptr, layout)
	}
	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		if ALLOC_GATE.load(Ordering::Relaxed) {
			ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
			ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
		}
		System.realloc(ptr, layout, new_size)
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if ALLOC_GATE.load(Ordering::Relaxed) {
			ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
			ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
		}
		System.alloc_zeroed(layout)
	}
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_snapshot() -> (u64, u64) {
	(ALLOC_COUNT.load(Ordering::Relaxed), ALLOC_BYTES.load(Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Windows process I/O counters (syscall-count proxy; no new dependency:
// windows-sys is already a target dependency of this crate)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct IoSnapshot {
	read_ops: u64,
	write_ops: u64,
	other_ops: u64,
	read_bytes: u64,
	write_bytes: u64,
	other_bytes: u64,
}

#[cfg(windows)]
fn io_snapshot() -> Option<IoSnapshot> {
	use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessIoCounters, IO_COUNTERS};
	unsafe {
		let mut counters: IO_COUNTERS = std::mem::zeroed();
		if GetProcessIoCounters(GetCurrentProcess(), &mut counters) == 0 {
			return None;
		}
		Some(IoSnapshot {
			read_ops: counters.ReadOperationCount,
			write_ops: counters.WriteOperationCount,
			other_ops: counters.OtherOperationCount,
			read_bytes: counters.ReadTransferCount,
			write_bytes: counters.WriteTransferCount,
			other_bytes: counters.OtherTransferCount,
		})
	}
}

#[cfg(not(windows))]
fn io_snapshot() -> Option<IoSnapshot> {
	None
}

fn io_delta(before: Option<IoSnapshot>, after: Option<IoSnapshot>) -> Option<IoSnapshot> {
	match (before, after) {
		(Some(before), Some(after)) => Some(IoSnapshot {
			read_ops: after.read_ops.saturating_sub(before.read_ops),
			write_ops: after.write_ops.saturating_sub(before.write_ops),
			other_ops: after.other_ops.saturating_sub(before.other_ops),
			read_bytes: after.read_bytes.saturating_sub(before.read_bytes),
			write_bytes: after.write_bytes.saturating_sub(before.write_bytes),
			other_bytes: after.other_bytes.saturating_sub(before.other_bytes),
		}),
		_ => None,
	}
}

fn io_json(delta: Option<IoSnapshot>) -> Value {
	match delta {
		None => Value::Null,
		Some(delta) => json!({
			"read_ops": delta.read_ops,
			"write_ops": delta.write_ops,
			"other_ops": delta.other_ops,
			"read_bytes": delta.read_bytes,
			"write_bytes": delta.write_bytes,
			"other_bytes": delta.other_bytes,
		}),
	}
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
	let digest = Sha256::digest(bytes);
	digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
	let mut hash: u64 = 0xcbf29ce484222325;
	for byte in bytes {
		hash ^= *byte as u64;
		hash = hash.wrapping_mul(0x100000001b3);
	}
	hash
}

fn stats_f64_ms(samples: &[f64]) -> Value {
	if samples.is_empty() {
		return json!({"n": 0});
	}
	let mut sorted = samples.to_vec();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
	let count = sorted.len();
	let mean = sorted.iter().sum::<f64>() / count as f64;
	let variance = sorted.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / count as f64;
	let percentile = |fraction: f64| sorted[((fraction * (count - 1) as f64).round() as usize).min(count - 1)];
	json!({
		"n": count,
		"mean_ms": mean,
		"p50_ms": percentile(0.50),
		"p95_ms": percentile(0.95),
		"p99_ms": percentile(0.99),
		"min_ms": sorted[0],
		"max_ms": sorted[count - 1],
		"stddev_ms": variance.sqrt(),
	})
}

fn stats_ms(samples_ns: &[u64]) -> Value {
	let samples: Vec<f64> = samples_ns.iter().map(|ns| *ns as f64 / 1e6).collect();
	stats_f64_ms(&samples)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).map_err(|error| format!("mkdir: {error}"))?;
	}
	let body = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
	std::fs::write(path, body).map_err(|error| format!("write {}: {error}", path.display()))?;
	Ok(())
}

fn exe_sha256() -> Result<String, String> {
	let exe = std::env::current_exe().map_err(|error| error.to_string())?;
	let bytes = std::fs::read(&exe).map_err(|error| format!("read exe: {error}"))?;
	Ok(sha256_hex(&bytes))
}

fn find_repo_root() -> Option<PathBuf> {
	let mut dir = std::env::current_exe().ok()?;
	dir.pop();
	for _ in 0..8 {
		if dir.join(".git").exists() {
			return Some(dir);
		}
		if !dir.pop() {
			return None;
		}
	}
	None
}

fn git_info() -> Value {
	let repo_root = find_repo_root();
	let run = |args: &[&str]| -> Option<String> {
		let mut command = std::process::Command::new("git");
		if let Some(root) = repo_root.as_deref() {
			command.current_dir(root);
		}
		let output = command.args(args).output().ok()?;
		if !output.status.success() {
			return None;
		}
		Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
	};
	json!({
		"commit": run(&["rev-parse", "HEAD"]),
		"describe": run(&["describe", "--tags", "--always", "--dirty"]),
		"branch": run(&["rev-parse", "--abbrev-ref", "HEAD"]),
	})
}

// ---------------------------------------------------------------------------
// Deterministic content generator (fixed seed; every run writes the SAME bytes)
// ---------------------------------------------------------------------------

const SEED: u64 = 0xD15C_0DE5_BA5E_2026;

struct Lcg(u64);

impl Lcg {
	fn new(seed: u64) -> Self {
		Lcg(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
	}
	fn next_u32(&mut self) -> u32 {
		self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
		(self.0 >> 33) as u32
	}
	fn next_below(&mut self, bound: u32) -> u32 {
		self.next_u32() % bound.max(1)
	}
}

const WORDS: [&str; 16] = [
	"buffer", "payload", "stream", "cursor", "lease", "guard", "window", "ticket", "frame", "token",
	"digest", "queue", "shard", "cache", "vertex", "batch",
];

/// Code-like ASCII text of exactly `target_bytes` bytes. Depends only on
/// (seed, target_bytes): identical across runs, invocations, and before/after
/// optimization work.
fn code_like_content(seed: u64, target_bytes: usize) -> String {
	let mut rng = Lcg::new(seed);
	let mut out = String::with_capacity(target_bytes + 160);
	let mut line_no: u32 = 0;
	while out.len() < target_bytes {
		line_no = line_no.wrapping_add(1);
		let line = match rng.next_below(10) {
			0 => format!("fn handler_{line_no:04}(ctx: &Context, req: Request) -> Response {{\n"),
			1 => format!(
				"    let value_{line_no:04} = compute_{:04}(data_{:04}, 0x{:08x});\n",
				line_no % 977,
				line_no % 331,
				rng.next_u32()
			),
			2 => format!(
				"    // invariant: the {} stays {} until the loop drains\n",
				WORDS[rng.next_below(16) as usize],
				WORDS[rng.next_below(16) as usize]
			),
			3 => format!("    match ctx.{}.get({}) {{\n", WORDS[rng.next_below(16) as usize], rng.next_below(9)),
			4 => format!("        Some(entry) => entry.merge_{}(),\n", WORDS[rng.next_below(16) as usize]),
			5 => "    }\n".to_string(),
			6 => format!("    for index in 0..{} {{\n", rng.next_below(64) + 1),
			7 => format!("        totals[{}] += weights[index] * 0x{:04x};\n", rng.next_below(8), rng.next_below(65536)),
			8 => format!(
				"    assert!(checksum_{:04} == 0x{:08x}, \"digest mismatch\");\n",
				line_no % 613,
				rng.next_u32()
			),
			_ => format!("    return Response::from_{}(payload_{:04});\n", WORDS[rng.next_below(16) as usize], line_no % 997),
		};
		out.push_str(&line);
	}
	out.truncate(target_bytes);
	out
}

// ---------------------------------------------------------------------------
// Point definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
	AtomicSync,
	AtomicAsync,
	Rewrite,
	Session,
	Edit,
	EventLog,
}

impl Scenario {
	fn as_str(&self) -> &'static str {
		match self {
			Scenario::AtomicSync => "atomic_sync",
			Scenario::AtomicAsync => "atomic_async",
			Scenario::Rewrite => "rewrite",
			Scenario::Session => "session",
			Scenario::Edit => "edit",
			Scenario::EventLog => "event_log",
		}
	}
}

#[derive(Debug, Clone)]
struct PointDef {
	name: String,
	scenario: Scenario,
	/// Bytes per file (atomic/rewrite/edit scenarios).
	size: usize,
	/// Files per pass (atomic), rewrites per pass (rewrite), turns per pass (session).
	count: usize,
	fsync: bool,
	fsync_dir: bool,
	/// Edits per tool call (edit scenario).
	edits: usize,
}

impl PointDef {
	/// Content of file index `i` (deterministic per point + index).
	fn content(&self, index: usize) -> String {
		let point_seed = fnv1a(self.name.as_bytes());
		code_like_content(SEED ^ point_seed ^ (index as u64).wrapping_mul(0x517C_C1B7_2722_0A95), self.size)
	}
	/// Operations per measured pass (writes / rewrites / turns / edit calls /
	/// ledger appends).
	fn ops_per_pass(&self) -> usize {
		match self.scenario {
			Scenario::Session => self.count,
			Scenario::Edit => EDIT_REPEATS,
			Scenario::EventLog => EVENTLOG_APPENDS,
			_ => self.count,
		}
	}
}

const SIZES: [usize; 4] = [4 * 1024, 32 * 1024, 256 * 1024, 2 * 1024 * 1024];
const COUNTS: [usize; 3] = [1, 20, 200];
const SYNC_VARIANTS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
const ASYNC_VARIANTS: [(bool, bool); 2] = [(false, false), (true, true)];
const REWRITE_SIZE: usize = 32 * 1024;
const REWRITE_COUNT: usize = 200;
const SESSION_TURNS: [usize; 3] = [10, 100, 1000];
const EVENTLOG_SIZES: [usize; 3] = [4 * 1024, 60 * 1024, 500 * 1024];
const EDIT_SIZES: [usize; 3] = [4 * 1024, 32 * 1024, 256 * 1024];
const EDIT_COUNTS: [usize; 2] = [1, 5];

fn kb(size: usize) -> String {
	if size % (1024 * 1024) == 0 {
		format!("{}m", size / (1024 * 1024))
	} else {
		format!("{}k", size / 1024)
	}
}

fn full_matrix() -> Vec<PointDef> {
	let mut points = Vec::new();
	for size in SIZES {
		for count in COUNTS {
			for (fsync, fsync_dir) in SYNC_VARIANTS {
				points.push(PointDef {
					name: format!("a_sync-s{}-n{}-f{}-d{}", kb(size), count, fsync as u8, fsync_dir as u8),
					scenario: Scenario::AtomicSync,
					size,
					count,
					fsync,
					fsync_dir,
					edits: 0,
				});
			}
			for (fsync, fsync_dir) in ASYNC_VARIANTS {
				points.push(PointDef {
					name: format!("a_async-s{}-n{}-f{}-d{}", kb(size), count, fsync as u8, fsync_dir as u8),
					scenario: Scenario::AtomicAsync,
					size,
					count,
					fsync,
					fsync_dir,
					edits: 0,
				});
			}
		}
	}
	for (fsync, fsync_dir) in SYNC_VARIANTS {
		points.push(PointDef {
			name: format!("a_rw-s{}-r{}-f{}-d{}", kb(REWRITE_SIZE), REWRITE_COUNT, fsync as u8, fsync_dir as u8),
			scenario: Scenario::Rewrite,
			size: REWRITE_SIZE,
			count: REWRITE_COUNT,
			fsync,
			fsync_dir,
			edits: 0,
		});
	}
	for turns in SESSION_TURNS {
		points.push(PointDef {
			name: format!("sess-t{turns}"),
			scenario: Scenario::Session,
			size: 0,
			count: turns,
			fsync: false,
			fsync_dir: false,
			edits: 0,
		});
	}
	for size in EDIT_SIZES {
		for edits in EDIT_COUNTS {
			points.push(PointDef {
				name: format!("edit-s{}-e{}", kb(size), edits),
				scenario: Scenario::Edit,
				size,
				count: 0,
				fsync: false,
				fsync_dir: false,
				edits,
			});
		}
	}
	// Semantic-edge ledger appends: seeded ledger sizes ~4 KiB / 60 KiB / 500
	// KiB, 20 measured appends per pass, durable=false like the per-model-request
	// caller; one durable=true variant at 60 KiB for the fsync share.
	for size in EVENTLOG_SIZES {
		points.push(PointDef {
			name: format!("evlog-s{}-d0", kb(size)),
			scenario: Scenario::EventLog,
			size,
			count: EVENTLOG_APPENDS,
			fsync: false,
			fsync_dir: false,
			edits: 0,
		});
	}
	points.push(PointDef {
		name: format!("evlog-s{}-d1", kb(60 * 1024)),
		scenario: Scenario::EventLog,
		size: 60 * 1024,
		count: EVENTLOG_APPENDS,
		fsync: true,
		fsync_dir: false,
		edits: 0,
	});
	points
}

fn quick_matrix() -> Vec<PointDef> {
	let mut points = Vec::new();
	let sizes = [4 * 1024, 256 * 1024];
	let counts = [1usize, 20];
	let variants = [(false, false), (true, true)];
	for size in sizes {
		for count in counts {
			for (fsync, fsync_dir) in variants {
				points.push(PointDef {
					name: format!("a_sync-s{}-n{}-f{}-d{}", kb(size), count, fsync as u8, fsync_dir as u8),
					scenario: Scenario::AtomicSync,
					size,
					count,
					fsync,
					fsync_dir,
					edits: 0,
				});
				points.push(PointDef {
					name: format!("a_async-s{}-n{}-f{}-d{}", kb(size), count, fsync as u8, fsync_dir as u8),
					scenario: Scenario::AtomicAsync,
					size,
					count,
					fsync,
					fsync_dir,
					edits: 0,
				});
			}
		}
	}
	for (fsync, fsync_dir) in [(false, false), (true, true)] {
		points.push(PointDef {
			name: format!("a_rw-s{}-r50-f{}-d{}", kb(REWRITE_SIZE), fsync as u8, fsync_dir as u8),
			scenario: Scenario::Rewrite,
			size: REWRITE_SIZE,
			count: 50,
			fsync,
			fsync_dir,
			edits: 0,
		});
	}
	for turns in [10usize, 100] {
		points.push(PointDef {
			name: format!("sess-t{turns}"),
			scenario: Scenario::Session,
			size: 0,
			count: turns,
			fsync: false,
			fsync_dir: false,
			edits: 0,
		});
	}
	points.push(PointDef {
		name: format!("edit-s{}-e1", kb(32 * 1024)),
		scenario: Scenario::Edit,
		size: 32 * 1024,
		count: 0,
		fsync: false,
		fsync_dir: false,
		edits: 1,
	});
	points.push(PointDef {
		name: format!("evlog-s{}-d0", kb(60 * 1024)),
		scenario: Scenario::EventLog,
		size: 60 * 1024,
		count: EVENTLOG_APPENDS,
		fsync: false,
		fsync_dir: false,
		edits: 0,
	});
	points
}

// ---------------------------------------------------------------------------
// Parity: digest every written file (content + path list)
// ---------------------------------------------------------------------------

struct TreeDigest {
	digest_hex: String,
	file_count: usize,
	total_bytes: u64,
	temp_leftovers: Vec<String>,
}

fn digest_dir_tree(dir: &Path) -> Result<TreeDigest, String> {
	let mut rel_paths: Vec<PathBuf> = Vec::new();
	collect_files(dir, Path::new(""), &mut rel_paths)?;
	rel_paths.sort();
	let mut hasher = Sha256::new();
	let mut total_bytes = 0u64;
	let mut temp_leftovers = Vec::new();
	for rel in &rel_paths {
		let rel_string = rel.to_string_lossy().replace('\\', "/");
		hasher.update(rel_string.as_bytes());
		hasher.update(&[0]);
		let bytes = std::fs::read(dir.join(rel)).map_err(|error| format!("read {}: {error}", rel.display()))?;
		hasher.update(Sha256::digest(&bytes));
		total_bytes += bytes.len() as u64;
		if rel_string.ends_with(".tmp") {
			temp_leftovers.push(rel_string);
		}
	}
	Ok(TreeDigest {
		digest_hex: sha256_hex(&hasher.finalize()),
		file_count: rel_paths.len(),
		total_bytes,
		temp_leftovers,
	})
}

fn collect_files(root: &Path, rel: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
	let entries = std::fs::read_dir(root.join(rel)).map_err(|error| format!("read_dir {}: {error}", rel.display()))?;
	for entry in entries {
		let entry = entry.map_err(|error| format!("dir entry: {error}"))?;
		let file_type = entry.file_type().map_err(|error| error.to_string())?;
		let child = rel.join(entry.file_name());
		if file_type.is_dir() {
			collect_files(root, &child, out)?;
		} else {
			out.push(child);
		}
	}
	Ok(())
}

/// Expected tree digest computed from the generator (no run involved).
/// Session files are covered by the structural + normalized-digest gate; the
/// returned digest still locks the file set shape (one empty placeholder).
fn expected_tree_digest(point: &PointDef) -> (String, u64, usize) {
	let mut hasher = Sha256::new();
	let mut total_bytes = 0u64;
	let mut file_count = 0usize;
	let mut feed = |rel: &str, content: &str| {
		hasher.update(rel.as_bytes());
		hasher.update(&[0]);
		hasher.update(Sha256::digest(content.as_bytes()));
		total_bytes += content.len() as u64;
		file_count += 1;
	};
	match point.scenario {
		Scenario::AtomicSync | Scenario::AtomicAsync => {
			for index in 0..point.count {
				feed(&format!("file-{index:04}.txt"), &point.content(index));
			}
		}
		Scenario::Rewrite => feed("target.txt", &point.content(point.count - 1)),
		Scenario::Edit => feed("target.rs", &expected_edit_result(point).0),
		Scenario::EventLog => feed("artifacts/semantic-edges.jsonl", &ledger_expected_bytes(point)),
		Scenario::Session => feed(&format!("sessions/{SESSION_ID}.jsonl"), ""),
	}
	(sha256_hex(&hasher.finalize()), total_bytes, file_count)
}

// ---------------------------------------------------------------------------
// Semantic-edge ledger case construction (deterministic)
// ---------------------------------------------------------------------------

fn ledger_seed_events(target_bytes: usize) -> Vec<SemanticEdgeLedgerEvent> {
	let mut events = Vec::new();
	let line_len = |event: &SemanticEdgeLedgerEvent| -> usize {
		serde_json::to_string(event).map(|line| line.len() + 1).unwrap_or(0)
	};
	let mut total = 0usize;
	let registered = SemanticEdgeLedgerEvent::SessionRegistered {
		session_id: SESSION_ID.to_string(),
		parent_session_id: None,
		spawned_by_request_id: None,
	};
	total += line_len(&registered);
	events.push(registered);
	let mut request_no = 0usize;
	while total < target_bytes {
		let request_id = format!("seed-req-{request_no:06}");
		let started = SemanticEdgeLedgerEvent::RequestStarted {
			request_id: request_id.clone(),
			session_id: SESSION_ID.to_string(),
			compaction_id: None,
		};
		let finished = SemanticEdgeLedgerEvent::RequestFinished { request_id };
		total += line_len(&started) + line_len(&finished);
		events.push(started);
		events.push(finished);
		request_no += 1;
	}
	events
}

/// The measured appends: 20 single-event appends, alternating RequestStarted /
/// RequestFinished like the real per-model-request traffic (10 requests).
fn ledger_measured_events() -> Vec<SemanticEdgeLedgerEvent> {
	(0..EVENTLOG_APPENDS)
		.map(|index| {
			if index % 2 == 0 {
				SemanticEdgeLedgerEvent::RequestStarted {
					request_id: format!("bench-req-{index:04}"),
					session_id: SESSION_ID.to_string(),
					compaction_id: None,
				}
			} else {
				SemanticEdgeLedgerEvent::RequestFinished {
					request_id: format!("bench-req-{:04}", index - 1),
				}
			}
		})
		.collect()
}

fn ledger_expected_bytes(point: &PointDef) -> String {
	let mut content = String::new();
	for event in ledger_seed_events(point.size) {
		content.push_str(&serde_json::to_string(&event).unwrap_or_default());
		content.push('\n');
	}
	for event in ledger_measured_events() {
		content.push_str(&serde_json::to_string(&event).unwrap_or_default());
		content.push('\n');
	}
	content
}

/// One measured pass of the event-log scenario: seed a ledger of ~`point.size`
/// bytes through the real `append_sync` (untimed, batched for setup speed),
/// then `EVENTLOG_APPENDS` single-event appends exactly as
/// `core::semantic_edges.rs` issues them per model request.
async fn event_log_pass(point: &PointDef, pass_index: usize, pass_dir: &Path, count_allocs: bool) -> PassRecord {
	let artifacts_dir = pass_dir.join("artifacts");
	let ledger_path = artifacts_dir.join("semantic-edges.jsonl");
	let ledger = EventLog::new(ledger_path.to_string_lossy().to_string(), EventLogOptions::default());
	// Untimed fixture seeding through the real append path.
	let seed_values: Vec<Value> = ledger_seed_events(point.size)
		.iter()
		.map(|event| serde_json::to_value(event).unwrap_or(Value::Null))
		.collect();
	for batch in seed_values.chunks(EVENTLOG_SEED_BATCH) {
		if let Err(error) = ledger.append_sync(batch, false, None) {
			return PassRecord::failed(pass_index, format!("seed append failed: {error}"));
		}
	}
	let measured_events = ledger_measured_events();
	let mut op_ns = Vec::with_capacity(EVENTLOG_APPENDS);
	let io_before = io_snapshot();
	let alloc_before = if count_allocs {
		ALLOC_GATE.store(true, Ordering::Relaxed);
		Some(alloc_snapshot())
	} else {
		None
	};
	let started = Instant::now();
	for event in &measured_events {
		let value = serde_json::to_value(event).unwrap_or(Value::Null);
		let op_started = Instant::now();
		if let Err(error) = ledger.append_sync(&[value], point.fsync, None) {
			ALLOC_GATE.store(false, Ordering::Relaxed);
			return PassRecord::failed(pass_index, format!("measured append failed: {error}"));
		}
		op_ns.push(op_started.elapsed().as_nanos() as u64);
	}
	let wall_ns = started.elapsed().as_nanos() as u64;
	let alloc = match (count_allocs, alloc_before) {
		(true, Some((count, bytes))) => {
			let (after_count, after_bytes) = alloc_snapshot();
			ALLOC_GATE.store(false, Ordering::Relaxed);
			Some((after_count.saturating_sub(count), after_bytes.saturating_sub(bytes)))
		}
		_ => None,
	};
	let io = io_delta(io_before, io_snapshot());
	// Parity: the ledger bytes must equal seed + measured lines exactly.
	let actual = match std::fs::read_to_string(&ledger_path) {
		Ok(actual) => actual,
		Err(error) => return PassRecord::failed(pass_index, format!("read ledger: {error}")),
	};
	let expected = ledger_expected_bytes(point);
	if actual != expected {
		return PassRecord::failed(pass_index, format!(
			"ledger bytes mismatch: sha256 {} != {} ({} vs {} bytes)",
			sha256_hex(actual.as_bytes()),
			sha256_hex(expected.as_bytes()),
			actual.len(),
			expected.len()
		));
	}
	let tree = match digest_dir_tree(pass_dir) {
		Ok(tree) => tree,
		Err(error) => return PassRecord::failed(pass_index, error),
	};
	PassRecord {
		pass_index,
		ok: true,
		error: None,
		op_ns,
		wall_ns,
		alloc,
		io,
		before_rename_count: None,
		persist_count: None,
		edit_calls: None,
		edit_reads: None,
		edit_writes: None,
		digest_hex: tree.digest_hex,
		file_count: tree.file_count,
		total_bytes: tree.total_bytes,
		temp_leftovers: tree.temp_leftovers,
		session_digest_hex: None,
		session_lines: Some(measured_events.len() + ledger_seed_events(point.size).len()),
		create_dir_all_control_ns: None,
	}
}

// ---------------------------------------------------------------------------
// Edit-tool case construction (deterministic)
// ---------------------------------------------------------------------------

/// Base file plus (old, new) marker pairs. Expected final content is the base
/// with every unique marker line replaced.
fn edit_case(point: &PointDef) -> (String, Vec<(String, String)>) {
	let point_seed = fnv1a(point.name.as_bytes());
	let mut rng = Lcg::new(SEED ^ point_seed);
	let mut lines: Vec<String> = Vec::new();
	let mut out_len = 0usize;
	let budget = point.size.saturating_sub(72 * point.edits.max(1));
	while out_len < budget {
		let line_no = lines.len() as u32 + 1;
		let line = match rng.next_below(8) {
			0 => format!("fn step_{line_no:04}(input: &Input) -> Output {{\n"),
			1 => format!("    let staged_{line_no:04} = transform_{:04}(input, 0x{:04x});\n", line_no % 977, rng.next_below(65536)),
			2 => format!(
				"    // keep the {} aligned with the {}\n",
				WORDS[rng.next_below(16) as usize],
				WORDS[rng.next_below(16) as usize]
			),
			3 => format!("    if staged_{line_no:04}.is_empty() {{ return Output::default(); }}\n"),
			4 => format!("    totals_{line_no:04} += input.weights[{}];\n", rng.next_below(32)),
			5 => "}\n".to_string(),
			6 => format!("    queue_{:04}.push(staged_{line_no:04});\n", line_no % 331),
			_ => format!("    trace!(\"step {line_no:04} sampled payload\");\n"),
		};
		out_len += line.len();
		lines.push(line);
	}
	let mut pairs = Vec::with_capacity(point.edits);
	for edit_index in 0..point.edits {
		let old = format!("    let marker_{edit_index:03} = stale_{edit_index:03}(payload); // MARKER-{edit_index:03}\n");
		let new = format!("    let marker_{edit_index:03} = fresh_{edit_index:03}(payload); // REPLACED-{edit_index:03}\n");
		pairs.push((old, new));
	}
	for (edit_index, pair) in pairs.iter().enumerate() {
		let position = (lines.len() + 1) * (edit_index + 1) / (point.edits + 1);
		lines.insert(position.min(lines.len()), pair.0.clone());
	}
	(lines.join(""), pairs)
}

fn expected_edit_result(point: &PointDef) -> (String, Vec<(String, String)>) {
	let (base, pairs) = edit_case(point);
	let mut result = base.clone();
	for (old, new) in &pairs {
		result = result.replacen(old, new, 1);
	}
	(result, pairs)
}

/// Navigate `Map`/`Value` fields by path segments; numeric segments index arrays.
fn field<'a>(value: &'a Map<String, Value>, path: &[&str]) -> Option<&'a Value> {
	let mut current: &Value = value.get(path[0])?;
	for segment in &path[1..] {
		current = match current {
			Value::Object(map) => map.get(*segment)?,
			Value::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
			_ => return None,
		};
	}
	Some(current)
}

// ---------------------------------------------------------------------------
// Pass record and scenario runners
// ---------------------------------------------------------------------------

struct PassRecord {
	pass_index: usize,
	ok: bool,
	error: Option<String>,
	op_ns: Vec<u64>,
	wall_ns: u64,
	alloc: Option<(u64, u64)>,
	io: Option<IoSnapshot>,
	before_rename_count: Option<u64>,
	persist_count: Option<u64>,
	edit_calls: Option<u64>,
	edit_reads: Option<u64>,
	edit_writes: Option<u64>,
	digest_hex: String,
	file_count: usize,
	total_bytes: u64,
	temp_leftovers: Vec<String>,
	session_digest_hex: Option<String>,
	session_lines: Option<usize>,
	/// Untimed control: `create_dir_all` on the existing session dir (the same
	/// call `SessionManager::persist` issues on every append), per-call ns.
	create_dir_all_control_ns: Option<Vec<u64>>,
}

impl PassRecord {
	fn failed(pass_index: usize, error: String) -> PassRecord {
		PassRecord {
			pass_index,
			ok: false,
			error: Some(error),
			op_ns: Vec::new(),
			wall_ns: 0,
			alloc: None,
			io: None,
			before_rename_count: None,
			persist_count: None,
			edit_calls: None,
			edit_reads: None,
			edit_writes: None,
			digest_hex: String::new(),
			file_count: 0,
			total_bytes: 0,
			temp_leftovers: Vec::new(),
			session_digest_hex: None,
			session_lines: None,
			create_dir_all_control_ns: None,
		}
	}
}

/// The before_rename seam: a closure that only counts (the real write
/// continues inside the production function).
fn before_rename_counter(counter: Arc<AtomicU64>) -> Option<Box<dyn FnOnce(&str)>> {
	Some(Box::new(move |_temp_path: &str| {
		counter.fetch_add(1, Ordering::Relaxed);
	}))
}

/// One measured pass of the atomic scenarios: `count` writes through the real
/// `write_file_atomic_sync` / `write_file_atomic`.
async fn atomic_pass(point: &PointDef, pass_index: usize, pass_dir: &Path, count_allocs: bool) -> PassRecord {
	let before_rename_count = Arc::new(AtomicU64::new(0));
	let mut op_ns = Vec::with_capacity(point.count);
	let io_before = io_snapshot();
	let alloc_before = if count_allocs {
		ALLOC_GATE.store(true, Ordering::Relaxed);
		Some(alloc_snapshot())
	} else {
		None
	};
	let started = Instant::now();
	let mut failure: Option<String> = None;
	for index in 0..point.count {
		let path = pass_dir.join(format!("file-{index:04}.txt"));
		let content = point.content(index);
		let counter = before_rename_counter(before_rename_count.clone());
		let result = match point.scenario {
			Scenario::AtomicSync => {
				let options = WriteFileAtomicOptions {
					mode: None,
					fsync: point.fsync,
					fsync_dir: point.fsync_dir,
					before_rename: counter,
				};
				let op_started = Instant::now();
				let result = write_file_atomic_sync(&path.to_string_lossy(), &content, options);
				op_ns.push(op_started.elapsed().as_nanos() as u64);
				result
			}
			Scenario::AtomicAsync => {
				let options = WriteFileAtomicAsyncOptions {
					mode: None,
					fsync: point.fsync,
					fsync_dir: point.fsync_dir,
					before_rename: counter,
					rename_retry: None,
				};
				let op_started = Instant::now();
				let result = write_file_atomic(&path.to_string_lossy(), &content, options).await;
				op_ns.push(op_started.elapsed().as_nanos() as u64);
				result
			}
			_ => unreachable!("atomic_pass on non-atomic scenario"),
		};
		if let Err(error) = result {
			failure = Some(format!("write {index} failed: {error}"));
			break;
		}
	}
	let wall_ns = started.elapsed().as_nanos() as u64;
	let alloc = match (count_allocs, alloc_before) {
		(true, Some((count, bytes))) => {
			let (after_count, after_bytes) = alloc_snapshot();
			ALLOC_GATE.store(false, Ordering::Relaxed);
			Some((after_count.saturating_sub(count), after_bytes.saturating_sub(bytes)))
		}
		_ => None,
	};
	let io = io_delta(io_before, io_snapshot());
	if let Some(failure) = failure {
		return PassRecord::failed(pass_index, failure);
	}
	let tree = match digest_dir_tree(pass_dir) {
		Ok(tree) => tree,
		Err(error) => return PassRecord::failed(pass_index, error),
	};
	PassRecord {
		pass_index: pass_index,
		ok: true,
		error: None,
		op_ns,
		wall_ns,
		alloc,
		io,
		before_rename_count: Some(before_rename_count.load(Ordering::Relaxed)),
		persist_count: None,
		edit_calls: None,
		edit_reads: None,
		edit_writes: None,
		digest_hex: tree.digest_hex,
		file_count: tree.file_count,
		total_bytes: tree.total_bytes,
		temp_leftovers: tree.temp_leftovers,
		session_digest_hex: None,
		session_lines: None,
		create_dir_all_control_ns: None,
	}
}

/// One measured pass of the rewrite storm: `count` sequential rewrites of ONE
/// destination file.
async fn rewrite_pass(point: &PointDef, pass_index: usize, pass_dir: &Path, count_allocs: bool) -> PassRecord {
	let target = pass_dir.join("target.txt");
	let before_rename_count = Arc::new(AtomicU64::new(0));
	let mut op_ns = Vec::with_capacity(point.count);
	let io_before = io_snapshot();
	let alloc_before = if count_allocs {
		ALLOC_GATE.store(true, Ordering::Relaxed);
		Some(alloc_snapshot())
	} else {
		None
	};
	let started = Instant::now();
	let mut failure: Option<String> = None;
	for index in 0..point.count {
		let content = point.content(index);
		let counter = before_rename_counter(before_rename_count.clone());
		let options = WriteFileAtomicOptions {
			mode: None,
			fsync: point.fsync,
			fsync_dir: point.fsync_dir,
			before_rename: counter,
		};
		let op_started = Instant::now();
		let result = write_file_atomic_sync(&target.to_string_lossy(), &content, options);
		op_ns.push(op_started.elapsed().as_nanos() as u64);
		if let Err(error) = result {
			failure = Some(format!("rewrite {index} failed: {error}"));
			break;
		}
	}
	let wall_ns = started.elapsed().as_nanos() as u64;
	let alloc = match (count_allocs, alloc_before) {
		(true, Some((count, bytes))) => {
			let (after_count, after_bytes) = alloc_snapshot();
			ALLOC_GATE.store(false, Ordering::Relaxed);
			Some((after_count.saturating_sub(count), after_bytes.saturating_sub(bytes)))
		}
		_ => None,
	};
	let io = io_delta(io_before, io_snapshot());
	if let Some(failure) = failure {
		ALLOC_GATE.store(false, Ordering::Relaxed);
		return PassRecord::failed(pass_index, failure);
	}
	let tree = match digest_dir_tree(pass_dir) {
		Ok(tree) => tree,
		Err(error) => return PassRecord::failed(pass_index, error),
	};
	PassRecord {
		pass_index: pass_index,
		ok: true,
		error: None,
		op_ns,
		wall_ns,
		alloc,
		io,
		before_rename_count: Some(before_rename_count.load(Ordering::Relaxed)),
		persist_count: None,
		edit_calls: None,
		edit_reads: None,
		edit_writes: None,
		digest_hex: tree.digest_hex,
		file_count: tree.file_count,
		total_bytes: tree.total_bytes,
		temp_leftovers: tree.temp_leftovers,
		session_digest_hex: None,
		session_lines: None,
		create_dir_all_control_ns: None,
	}
}

/// Deterministic per-turn session content.
fn session_user_text(turn: usize) -> String {
	format!(
		"Turn {turn}: update the parser so nested quotes survive round-trips. Reproduce with parse_case_{turn:04}() and keep the fixture list stable. Files in scope: parser.rs, parser_test.rs."
	)
}

fn session_assistant_text(turn: usize) -> String {
	code_like_content(SEED ^ 0xA55E_D0E5 ^ (turn as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15), 3072)
}

/// One measured pass of the session scenario: real `SessionManager` JSONL
/// persistence, `count` turns of (user append + assistant append). The first
/// persisting append rewrites the whole file; later appends append lines.
async fn session_pass(point: &PointDef, pass_index: usize, pass_dir: &Path, count_allocs: bool) -> PassRecord {
	let sessions_dir = pass_dir.join("sessions");
	if let Err(error) = std::fs::create_dir_all(&sessions_dir) {
		return PassRecord::failed(pass_index, format!("mkdir sessions: {error}"));
	}
	let cwd = pass_dir.to_string_lossy().to_string();
	let mut manager = match SessionManager::create(&cwd, Some(&sessions_dir.to_string_lossy())) {
		Ok(manager) => manager,
		Err(error) => return PassRecord::failed(pass_index, format!("SessionManager::create: {error}")),
	};
	if let Err(error) = manager.new_session(Some(&NewSessionOptions {
		id: Some(SESSION_ID.to_string()),
		..Default::default()
	})) {
		return PassRecord::failed(pass_index, format!("new_session: {error}"));
	}
	let persist_count = Arc::new(AtomicU64::new(0));
	let listener_count = persist_count.clone();
	let _unsubscribe = manager.on_persist(Box::new(move |_session_file: &str| {
		listener_count.fetch_add(1, Ordering::Relaxed);
	}));

	let turns = point.count;
	let mut op_ns = Vec::with_capacity(turns);
	let io_before = io_snapshot();
	let alloc_before = if count_allocs {
		ALLOC_GATE.store(true, Ordering::Relaxed);
		Some(alloc_snapshot())
	} else {
		None
	};
	let started = Instant::now();
	for turn in 0..turns {
		let user = AgentMessage::from(UserMessage::new(
			UserContent::Text(session_user_text(turn)),
			1_772_000_000_000 + turn as i64,
		));
		let assistant = AgentMessage::from(AssistantMessage {
			content: vec![ContentBlock::Text(TextContent::new(session_assistant_text(turn)))],
			api: "bench-fixture".to_string(),
			provider: "bench-fixture".to_string(),
			model: "bench-model".to_string(),
			usage: Usage::zero(),
			timestamp: 1_772_000_000_000 + turn as i64,
			..Default::default()
		});
		let op_started = Instant::now();
		if let Err(error) = manager.append_message(user) {
			ALLOC_GATE.store(false, Ordering::Relaxed);
			return PassRecord::failed(pass_index, format!("append user turn {turn}: {error}"));
		}
		if let Err(error) = manager.append_message(assistant) {
			ALLOC_GATE.store(false, Ordering::Relaxed);
			return PassRecord::failed(pass_index, format!("append assistant turn {turn}: {error}"));
		}
		op_ns.push(op_started.elapsed().as_nanos() as u64);
	}
	let wall_ns = started.elapsed().as_nanos() as u64;
	let alloc = match (count_allocs, alloc_before) {
		(true, Some((count, bytes))) => {
			let (after_count, after_bytes) = alloc_snapshot();
			ALLOC_GATE.store(false, Ordering::Relaxed);
			Some((after_count.saturating_sub(count), after_bytes.saturating_sub(bytes)))
		}
		_ => None,
	};
	let io = io_delta(io_before, io_snapshot());
	drop(_unsubscribe);
	drop(manager);

	// Untimed control isolating the per-append `create_dir_all` cost component:
	// `SessionManager::persist` issues this exact call on the existing sessions
	// dir before every line append; the measured turn times above include it.
	let mut create_dir_all_control_ns = Vec::with_capacity(SESSION_CREATE_DIR_ALL_CONTROL);
	for _ in 0..SESSION_CREATE_DIR_ALL_CONTROL {
		let control_started = Instant::now();
		let _ = std::fs::create_dir_all(&sessions_dir);
		create_dir_all_control_ns.push(control_started.elapsed().as_nanos() as u64);
	}

	// Parity: exactly one JSONL file with the deterministic name, structural
	// checks on every entry, normalized digest over volatile fields.
	let jsonl_files: Vec<PathBuf> = match std::fs::read_dir(&sessions_dir) {
		Ok(entries) => entries
			.filter_map(|entry| entry.ok())
			.map(|entry| entry.path())
			.filter(|path| path.extension().map(|ext| ext == "jsonl").unwrap_or(false))
			.collect(),
		Err(error) => return PassRecord::failed(pass_index, format!("read sessions dir: {error}")),
	};
	if jsonl_files.len() != 1 {
		return PassRecord::failed(pass_index, format!(
			"expected exactly one session file, found {}",
			jsonl_files.len()
		));
	}
	let session_file = jsonl_files[0].clone();
	if session_file.file_name().and_then(|name| name.to_str()) != Some(&format!("{SESSION_ID}.jsonl")) {
		return PassRecord::failed(pass_index, format!(
			"unexpected session file name: {}",
			session_file.display()
		));
	}
	let raw = match std::fs::read_to_string(&session_file) {
		Ok(raw) => raw,
		Err(error) => return PassRecord::failed(pass_index, format!("read session file: {error}")),
	};
	let lines: Vec<&str> = raw.lines().filter(|line| !line.trim().is_empty()).collect();
	let expected_lines = 2 * turns + 1;
	if lines.len() != expected_lines {
		return PassRecord::failed(pass_index, format!(
			"expected {expected_lines} lines, found {}",
			lines.len()
		));
	}
	let mut normalized_hasher = Sha256::new();
	let mut previous_id: Option<String> = None;
	let mut checked_entries = 0usize;
	for (line_index, line) in lines.iter().enumerate() {
		let mut value: Map<String, Value> = match serde_json::from_str(line) {
			Ok(Value::Object(map)) => map,
			_ => {
				return PassRecord::failed(pass_index, format!("line {line_index} is not a JSON object"))
			}
		};
		let turn = line_index.saturating_sub(1) / 2;
		if line_index == 0 {
			if value.get("type").and_then(Value::as_str) != Some("session") {
				return PassRecord::failed(pass_index, "line 0 is not a session header".to_string());
			}
			if value.get("id").and_then(Value::as_str) != Some(SESSION_ID) {
				return PassRecord::failed(pass_index, "session header id mismatch".to_string());
			}
			// The header is not part of the message parent chain: `new_session`
			// leaves `leaf_id` None, so the first message carries `parentId: null`.
		} else {
			if value.get("type").and_then(Value::as_str) != Some("message") {
				return PassRecord::failed(pass_index, format!("line {line_index} is not a message entry"));
			}
			let parent = value.get("parentId").and_then(Value::as_str).map(str::to_string);
			if parent != previous_id {
				return PassRecord::failed(pass_index, format!(
					"line {line_index} breaks the parent chain: {parent:?} != {previous_id:?}"
				));
			}
			let role = field(&value, &["message", "role"]).and_then(Value::as_str).unwrap_or_default();
			let expected_role = if line_index % 2 == 1 { "user" } else { "assistant" };
			if role != expected_role {
				return PassRecord::failed(pass_index, format!(
					"line {line_index} role {role:?} != {expected_role:?}"
				));
			}
			let text = if expected_role == "user" {
				field(&value, &["message", "content"]).and_then(Value::as_str)
			} else {
				field(&value, &["message", "content", "0", "text"]).and_then(Value::as_str)
			};
			let expected_text = if expected_role == "user" {
				session_user_text(turn)
			} else {
				session_assistant_text(turn)
			};
			let text = match text {
				Some(text) => text,
				None => {
					return PassRecord::failed(pass_index, format!("line {line_index} has no text content"))
				}
			};
			if text != expected_text {
				return PassRecord::failed(pass_index, format!("line {line_index} content mismatch"));
			}
			previous_id = value.get("id").and_then(Value::as_str).map(str::to_string);
			checked_entries += 1;
		}
		// Normalize run-volatile fields (entry timestamp, generated ids, harness cwd).
		if value.contains_key("timestamp") {
			value.insert("timestamp".to_string(), Value::String("<TS>".to_string()));
		}
		if value.contains_key("id") {
			value.insert("id".to_string(), Value::String("<ID>".to_string()));
		}
		if value.contains_key("parentId") {
			value.insert("parentId".to_string(), Value::String("<PID>".to_string()));
		}
		if value.contains_key("cwd") {
			value.insert("cwd".to_string(), Value::String("<CWD>".to_string()));
		}
		normalized_hasher.update(serde_json::to_string(&Value::Object(value)).unwrap_or_default().as_bytes());
		normalized_hasher.update(&[b'\n']);
	}
	if checked_entries != 2 * turns {
		return PassRecord::failed(pass_index, format!(
			"checked {checked_entries} entries, expected {}",
			2 * turns
		));
	}
	let session_digest_hex = sha256_hex(&normalized_hasher.finalize());
	let tree = match digest_dir_tree(pass_dir) {
		Ok(tree) => tree,
		Err(error) => return PassRecord::failed(pass_index, error),
	};
	PassRecord {
		pass_index: pass_index,
		ok: true,
		error: None,
		op_ns,
		wall_ns,
		alloc,
		io,
		before_rename_count: None,
		persist_count: Some(persist_count.load(Ordering::Relaxed)),
		edit_calls: None,
		edit_reads: None,
		edit_writes: None,
		digest_hex: tree.digest_hex,
		file_count: tree.file_count,
		total_bytes: tree.total_bytes,
		temp_leftovers: tree.temp_leftovers,
		session_digest_hex: Some(session_digest_hex),
		session_lines: Some(lines.len()),
		create_dir_all_control_ns: Some(create_dir_all_control_ns),
	}
}

/// Counting wrapper that only delegates to the real `LocalEditOperations`.
struct CountingEditOperations {
	inner: LocalEditOperations,
	calls: AtomicU64,
	reads: AtomicU64,
	writes: AtomicU64,
}

impl CountingEditOperations {
	fn new() -> Arc<Self> {
		Arc::new(CountingEditOperations {
			inner: LocalEditOperations,
			calls: AtomicU64::new(0),
			reads: AtomicU64::new(0),
			writes: AtomicU64::new(0),
		})
	}
}

impl EditOperations for CountingEditOperations {
	fn read_file(&self, absolute_path: &str) -> std::io::Result<Vec<u8>> {
		self.reads.fetch_add(1, Ordering::Relaxed);
		self.inner.read_file(absolute_path)
	}
	fn write_file(&self, absolute_path: &str, content: &str) -> std::io::Result<()> {
		self.writes.fetch_add(1, Ordering::Relaxed);
		self.inner.write_file(absolute_path, content)
	}
	fn access(&self, absolute_path: &str) -> std::io::Result<()> {
		self.calls.fetch_add(1, Ordering::Relaxed);
		self.inner.access(absolute_path)
	}
}

/// One measured pass of the edit scenario: real `execute_edit` calls with a
/// counting wrapper around the default local operations. Each repeat starts
/// from a reset base file (unmeasured) so every call sees identical input.
async fn edit_pass(point: &PointDef, pass_index: usize, pass_dir: &Path, count_allocs: bool) -> PassRecord {
	let (base, pairs) = edit_case(point);
	let (expected_final, _) = expected_edit_result(point);
	let target = pass_dir.join("target.rs");
	let input = EditToolInput {
		path: "target.rs".to_string(),
		edits: pairs
			.iter()
			.map(|(old, new)| Edit {
				old_text: old.clone(),
				new_text: new.clone(),
			})
			.collect(),
	};
	let cwd = pass_dir.to_string_lossy().to_string();
	let ops = CountingEditOperations::new();
	let mut op_ns = Vec::with_capacity(EDIT_REPEATS);
	// The EditOperations seam counts exactly; process-wide I/O counters would
	// also include the harness verification reads, so io stays null here.
	let io_before = None;
	let alloc_before = if count_allocs {
		ALLOC_GATE.store(true, Ordering::Relaxed);
		Some(alloc_snapshot())
	} else {
		None
	};
	let verify_each = !count_allocs;
	let started = Instant::now();
	for repeat in 0..EDIT_REPEATS {
		if let Err(error) = std::fs::write(&target, &base) {
			ALLOC_GATE.store(false, Ordering::Relaxed);
			return PassRecord::failed(pass_index, format!("reset base: {error}"));
		}
		let op_started = Instant::now();
		match execute_edit(&cwd, Some(ops.clone()), &input, None).await {
			Ok(_message) => {
				op_ns.push(op_started.elapsed().as_nanos() as u64);
			}
			Err(error) => {
				ALLOC_GATE.store(false, Ordering::Relaxed);
				return PassRecord::failed(pass_index, format!("edit call {repeat} failed: {error}"));
			}
		}
		if verify_each {
			let actual = match std::fs::read_to_string(&target) {
				Ok(actual) => actual,
				Err(error) => {
					return PassRecord::failed(pass_index, format!("read target: {error}"))
				}
			};
			if actual != expected_final {
				return PassRecord::failed(pass_index, format!(
					"edit result mismatch after call {repeat}: sha256 {} != {}",
					sha256_hex(actual.as_bytes()),
					sha256_hex(expected_final.as_bytes())
				));
			}
		}
	}
	let wall_ns = started.elapsed().as_nanos() as u64;
	let alloc = match (count_allocs, alloc_before) {
		(true, Some((count, bytes))) => {
			let (after_count, after_bytes) = alloc_snapshot();
			ALLOC_GATE.store(false, Ordering::Relaxed);
			Some((after_count.saturating_sub(count), after_bytes.saturating_sub(bytes)))
		}
		_ => None,
	};
	let io = io_delta(io_before, io_snapshot());
	if !verify_each {
		let actual = match std::fs::read_to_string(&target) {
			Ok(actual) => actual,
			Err(error) => return PassRecord::failed(pass_index, format!("read target: {error}")),
		};
		if actual != expected_final {
			return PassRecord::failed(pass_index, format!(
				"edit result mismatch after final call: sha256 {} != {}",
				sha256_hex(actual.as_bytes()),
				sha256_hex(expected_final.as_bytes())
			));
		}
	}
	let tree = match digest_dir_tree(pass_dir) {
		Ok(tree) => tree,
		Err(error) => return PassRecord::failed(pass_index, error),
	};
	PassRecord {
		pass_index: pass_index,
		ok: true,
		error: None,
		op_ns,
		wall_ns,
		alloc,
		io,
		before_rename_count: None,
		persist_count: None,
		edit_calls: Some(ops.calls.load(Ordering::Relaxed)),
		edit_reads: Some(ops.reads.load(Ordering::Relaxed)),
		edit_writes: Some(ops.writes.load(Ordering::Relaxed)),
		digest_hex: tree.digest_hex,
		file_count: tree.file_count,
		total_bytes: tree.total_bytes,
		temp_leftovers: tree.temp_leftovers,
		session_digest_hex: None,
		session_lines: None,
		create_dir_all_control_ns: None,
	}
}

// ---------------------------------------------------------------------------
// Point execution
// ---------------------------------------------------------------------------

struct RunOptions {
	runs: usize,
	warmup: usize,
	alloc_runs: usize,
	root: PathBuf,
}

fn pass_label(pass_index: usize, warmup: usize, runs: usize) -> (bool, bool) {
	let is_warmup = pass_index < warmup;
	let is_alloc = pass_index >= warmup + runs;
	(is_warmup, is_alloc)
}

async fn run_point(point: &PointDef, options: &RunOptions) -> Result<Value, String> {
	let scenario_dir = options.root.join(point.scenario.as_str()).join(&point.name);
	let (expected_digest, expected_bytes, expected_files) = expected_tree_digest(point);
	let mut passes: Vec<Value> = Vec::new();
	let mut pass_records: Vec<PassRecord> = Vec::new();
	let total_passes = options.warmup + options.runs + options.alloc_runs;
	for pass_index in 0..total_passes {
		let (is_warmup, is_alloc) = pass_label(pass_index, options.warmup, options.runs);
		let pass_dir = scenario_dir.join(format!("pass-{pass_index:02}"));
		if pass_dir.exists() {
			std::fs::remove_dir_all(&pass_dir).map_err(|error| format!("clean pass dir: {error}"))?;
		}
		std::fs::create_dir_all(&pass_dir).map_err(|error| format!("mkdir pass dir: {error}"))?;
		let record = match point.scenario {
			Scenario::AtomicSync | Scenario::AtomicAsync => atomic_pass(point, pass_index, &pass_dir, is_alloc).await,
			Scenario::Rewrite => rewrite_pass(point, pass_index, &pass_dir, is_alloc).await,
			Scenario::Session => session_pass(point, pass_index, &pass_dir, is_alloc).await,
			Scenario::Edit => edit_pass(point, pass_index, &pass_dir, is_alloc).await,
			Scenario::EventLog => event_log_pass(point, pass_index, &pass_dir, is_alloc).await,
		};
		passes.push(pass_json(&record, is_warmup, is_alloc));
		pass_records.push(record);
		// Remove the pass tree immediately (disk hygiene); it was already digested.
		let _ = std::fs::remove_dir_all(&pass_dir);
	}

	// Parity across the point: every pass must match the expected digest, the
	// expected file count, and leave no temp files behind.
	let mut parity_ok = true;
	let mut parity_errors: Vec<String> = Vec::new();
	for (index, record) in pass_records.iter().enumerate() {
		if !record.ok {
			parity_ok = false;
			parity_errors.push(format!("pass {index} failed: {}", record.error.clone().unwrap_or_default()));
			continue;
		}
		if !record.temp_leftovers.is_empty() {
			parity_ok = false;
			parity_errors.push(format!("pass {index} left temp files: {:?}", record.temp_leftovers));
		}
		if point.scenario != Scenario::Session {
			if record.digest_hex != expected_digest {
				parity_ok = false;
				parity_errors.push(format!(
					"pass {index} digest {} != expected {expected_digest}",
					record.digest_hex
				));
			}
			if record.file_count != expected_files {
				parity_ok = false;
				parity_errors.push(format!(
					"pass {index} file count {} != expected {expected_files}",
					record.file_count
				));
			}
		}
	}
	// Session: normalized digests must agree across passes; persist count must
	// equal entries-1 (the first user append does not persist: no assistant yet).
	if point.scenario == Scenario::Session {
		let reference = pass_records
			.iter()
			.find_map(|record| record.session_digest_hex.clone())
			.unwrap_or_default();
		if reference.is_empty() {
			parity_ok = false;
			parity_errors.push("no successful session pass produced a normalized digest".to_string());
		}
		for (index, record) in pass_records.iter().enumerate() {
			if record.ok && record.session_digest_hex.as_deref() != Some(reference.as_str()) {
				parity_ok = false;
				parity_errors.push(format!("pass {index} normalized session digest differs from reference"));
			}
		}
		let expected_persists = (2 * point.count - 1) as u64;
		for (index, record) in pass_records.iter().enumerate() {
			if record.ok && record.persist_count != Some(expected_persists) {
				parity_ok = false;
				parity_errors.push(format!(
					"pass {index} persist count {:?} != expected {expected_persists}",
					record.persist_count
				));
			}
		}
	}
	// Edit: exact operation counts through the counting seam.
	if point.scenario == Scenario::Edit {
		let repeats = EDIT_REPEATS as u64;
		for (index, record) in pass_records.iter().enumerate() {
			if record.ok
				&& (record.edit_calls != Some(repeats)
					|| record.edit_reads != Some(repeats)
					|| record.edit_writes != Some(repeats))
			{
				parity_ok = false;
				parity_errors.push(format!(
					"pass {index} edit op counts {:?}/{:?}/{:?} != {repeats}/{repeats}/{repeats}",
					record.edit_calls, record.edit_reads, record.edit_writes
				));
			}
		}
	}
	// Atomic: before_rename must fire exactly once per write.
	if matches!(point.scenario, Scenario::AtomicSync | Scenario::AtomicAsync | Scenario::Rewrite) {
		for (index, record) in pass_records.iter().enumerate() {
			if record.ok && record.before_rename_count != Some(point.count as u64) {
				parity_ok = false;
				parity_errors.push(format!(
					"pass {index} before_rename count {:?} != expected {}",
					record.before_rename_count, point.count
				));
			}
		}
	}

	let summary = summarize_point(point, &pass_records, options);
	Ok(json!({
		"point_name": point.name,
		"scenario": point.scenario.as_str(),
		"config": point_json(point),
		"expected": {
			"digest": if point.scenario == Scenario::Session { Value::Null } else { json!(expected_digest) },
			"bytes_total": expected_bytes,
			"files": expected_files,
		},
		"passes": passes,
		"summary": summary,
		"parity": {
			"ok": parity_ok,
			"errors": parity_errors,
		},
	}))
}

fn pass_json(record: &PassRecord, is_warmup: bool, is_alloc: bool) -> Value {
	json!({
		"pass_index": record.pass_index,
		"warmup": is_warmup,
		"alloc_run": is_alloc,
		"ok": record.ok,
		"error": record.error,
		"op_ns": record.op_ns,
		"first_op_ns": record.op_ns.first().copied(),
		"wall_ns": record.wall_ns,
		"alloc": record.alloc.map(|(count, bytes)| json!({"count": count, "bytes": bytes})),
		"io": io_json(record.io),
		"before_rename_count": record.before_rename_count,
		"persist_count": record.persist_count,
		"edit_op_counts": {
			"access": record.edit_calls,
			"reads": record.edit_reads,
			"writes": record.edit_writes,
		},
		"digest": record.digest_hex,
		"file_count": record.file_count,
		"bytes_total": record.total_bytes,
		"temp_leftovers": record.temp_leftovers,
		"session_normalized_digest": record.session_digest_hex,
		"session_lines": record.session_lines,
		"create_dir_all_control": record.create_dir_all_control_ns.as_ref().map(|samples| stats_ms(samples)),
	})
}

fn point_json(point: &PointDef) -> Value {
	json!({
		"scenario": point.scenario.as_str(),
		"size_bytes": point.size,
		"count": point.count,
		"fsync": point.fsync,
		"fsync_dir": point.fsync_dir,
		"edits": point.edits,
	})
}

fn summarize_point(point: &PointDef, records: &[PassRecord], options: &RunOptions) -> Value {
	let measured: Vec<&PassRecord> = records
		.iter()
		.filter(|record| record.ok)
		.filter(|record| {
			let (is_warmup, is_alloc) = pass_label(record.pass_index, options.warmup, options.runs);
			!is_warmup && !is_alloc
		})
		.collect();
	let op_divisor = point.ops_per_pass().max(1) as u64;
	let mut all_ops: Vec<u64> = Vec::new();
	let mut first_ops: Vec<u64> = Vec::new();
	let mut warm_ops: Vec<u64> = Vec::new();
	let mut walls: Vec<u64> = Vec::new();
	for record in &measured {
		all_ops.extend(record.op_ns.iter().copied());
		if let Some(first) = record.op_ns.first() {
			first_ops.push(*first);
		}
		warm_ops.extend(record.op_ns.iter().skip(1).copied());
		walls.push(record.wall_ns);
	}
	let alloc_counts: Vec<u64> = records
		.iter()
		.filter_map(|record| record.alloc.map(|(count, _)| count))
		.collect();
	let alloc_bytes: Vec<u64> = records
		.iter()
		.filter_map(|record| record.alloc.map(|(_, bytes)| bytes))
		.collect();
	let io_deltas: Vec<IoSnapshot> = records.iter().filter_map(|record| record.io).collect();
	let persists: Vec<u64> = records.iter().filter_map(|record| record.persist_count).collect();
	let before_renames: Vec<u64> = records
		.iter()
		.filter_map(|record| record.before_rename_count)
		.collect();
	let create_dir_all: Vec<u64> = records
		.iter()
		.filter_map(|record| record.create_dir_all_control_ns.clone())
		.flatten()
		.collect();
	let mean = |values: &[u64]| -> Option<u64> {
		if values.is_empty() {
			None
		} else {
			Some(values.iter().sum::<u64>() / values.len() as u64)
		}
	};
	let per_op = |values: &[u64]| mean(values).map(|value| value as f64 / op_divisor as f64);
	let io_summary = if io_deltas.is_empty() {
		Value::Null
	} else {
		json!({
			"read_ops_per_op": per_op(&io_deltas.iter().map(|delta| delta.read_ops).collect::<Vec<_>>()),
			"write_ops_per_op": per_op(&io_deltas.iter().map(|delta| delta.write_ops).collect::<Vec<_>>()),
			"other_ops_per_op": per_op(&io_deltas.iter().map(|delta| delta.other_ops).collect::<Vec<_>>()),
			"read_bytes_per_op": per_op(&io_deltas.iter().map(|delta| delta.read_bytes).collect::<Vec<_>>()),
			"write_bytes_per_op": per_op(&io_deltas.iter().map(|delta| delta.write_bytes).collect::<Vec<_>>()),
			"other_bytes_per_op": per_op(&io_deltas.iter().map(|delta| delta.other_bytes).collect::<Vec<_>>()),
		})
	};
	json!({
		"measured_passes": measured.len(),
		"op_label": match point.scenario {
			Scenario::Session => "turn",
			Scenario::Edit => "edit_call",
			Scenario::EventLog => "append",
			_ => "write",
		},
		"op_ms": stats_ms(&all_ops),
		"first_op_ms": stats_ms(&first_ops),
		"warm_ops_ms": stats_ms(&warm_ops),
		"wall_ms": stats_ms(&walls),
		"alloc_per_op": per_op(&alloc_counts),
		"alloc_bytes_per_op": per_op(&alloc_bytes),
		"persist_per_op": per_op(&persists),
		"before_rename_per_op": per_op(&before_renames),
		"create_dir_all_control_ms": if create_dir_all.is_empty() { Value::Null } else { stats_ms(&create_dir_all) },
		"io": io_summary,
	})
}

// ---------------------------------------------------------------------------
// Report and table
// ---------------------------------------------------------------------------

fn build_report(subcommand: &str, args: &Value, points: Vec<Value>, tmp_root: &Path) -> Value {
	json!({
		"kind": "disk_write_bench",
		"subcommand": subcommand,
		"args": args,
		"generated_utc": Utc::now().to_rfc3339(),
		"binary": {"sha256": exe_sha256().ok(), "git": git_info()},
		"host": {
			"os": std::env::consts::OS,
			"arch": std::env::consts::ARCH,
			"available_parallelism": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
			"tmp_root": tmp_root.to_string_lossy(),
		},
		"notes": [
			"measured paths: utils::atomic_file::write_file_atomic_sync + write_file_atomic; core::session_manager::SessionManager JSONL persistence; core::tools::edit::execute_edit with LocalEditOperations (std::fs::write); core::event_log::EventLog::append_sync (semantic-edge ledger pattern, one event per append, EventLogOptions::default, ledger file artifacts/semantic-edges.jsonl)",
			"session_manager rewrites the whole file on the first persisting append (private atomic stand-in), then appends line-by-line; the bench drives both through the public SessionManager API",
			"parity gate: every pass re-digests every written file (content + path list) and compares against the generator-derived expected digest; session files are additionally structurally checked and normalized (timestamp/id/parentId/cwd) before digesting",
			"cold vs warm: the first op of each pass writes to a destination that did not exist in the fresh pass directory; OS-level caches are NOT controlled (documented limitation)",
			"process I/O counters (Windows GetProcessIoCounters) count read/write operations issued by the process; metadata operations (create/rename/fsync) are NOT included; alloc runs include tokio runtime allocations in the async scenario",
			"sequential writes only; AtomicFileWriteCoordinator concurrency is not covered",
			"event-log points seed the ledger through the real append_sync (batched 64 events per call, untimed) to ~4 KiB / 60 KiB / 500 KiB; measured appends are single-event calls exactly like core::semantic_edges.rs (durable=false); the repair_tail_sync probe runs on every append and early-exits on the normal terminated tail (the torn-tail double-read path is not exercised)",
			"session points include an untimed create_dir_all control on the existing sessions dir (SESSION_CREATE_DIR_ALL_CONTROL=100 calls) isolating the per-append cost component that SessionManager::persist issues before every line append",
		],
		"points": points,
	})
}

fn report_parity_ok(report: &Value) -> bool {
	report["points"]
		.as_array()
		.map(|points| points.iter().all(|point| point["parity"]["ok"].as_bool().unwrap_or(false)))
		.unwrap_or(false)
}

fn print_table_header() {
	println!(
		"{:<28} {:>6} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>13} {:>4}",
		"point", "ops", "n", "op_p50", "op_p95", "op_p99", "first_p50", "wall_p50", "alloc/op", "wrop/op", "pers/op", "rn/op", "digest12", "ok"
	);
	println!("{}", "-".repeat(178));
}

fn print_point_row(point: &Value) {
	let name = point["point_name"].as_str().unwrap_or("?");
	let summary = &point["summary"];
	let stat = |key: &str, field: &str| summary[key][field].as_f64();
	let fmt = |value: Option<f64>| match value {
		Some(value) => format!("{value:.3}"),
		None => "-".to_string(),
	};
	let scenario = point["scenario"].as_str().unwrap_or("");
	let op_count = if scenario == "edit" {
		EDIT_REPEATS
	} else {
		point["config"]["count"].as_u64().unwrap_or(0) as usize
	};
	let _ = scenario;
	let measured = summary["measured_passes"].as_u64().unwrap_or(0);
	let total_ops = op_count as u64 * measured;
	let digest = point["expected"]["digest"]
		.as_str()
		.map(|digest| digest.get(..12).unwrap_or(digest).to_string())
		.or_else(|| {
			point["passes"].as_array().and_then(|passes| {
				passes
					.iter()
					.find_map(|pass| pass["session_normalized_digest"].as_str().map(str::to_string))
			})
		})
		.map(|digest| digest.get(..12).unwrap_or(&digest).to_string())
		.unwrap_or_else(|| "(session)".to_string());
	println!(
		"{:<28} {:>6} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>13} {:>4}",
		name,
		total_ops,
		measured,
		fmt(stat("op_ms", "p50_ms")),
		fmt(stat("op_ms", "p95_ms")),
		fmt(stat("op_ms", "p99_ms")),
		fmt(stat("first_op_ms", "p50_ms")),
		fmt(stat("wall_ms", "p50_ms")),
		fmt(summary["alloc_per_op"].as_f64()),
		fmt(summary["io"]["write_ops_per_op"].as_f64()),
		fmt(summary["persist_per_op"].as_f64()),
		fmt(summary["before_rename_per_op"].as_f64()),
		digest,
		if point["parity"]["ok"].as_bool().unwrap_or(false) { "yes" } else { "NO" },
	);
}

// ---------------------------------------------------------------------------
// Hand-rolled CLI
// ---------------------------------------------------------------------------

struct Cli {
	subcommand: String,
	options: HashMap<String, String>,
}

impl Cli {
	fn parse(args: &[String]) -> Result<Cli, String> {
		let mut subcommand = String::new();
		let mut options = HashMap::new();
		let mut index = 0usize;
		while index < args.len() {
			let arg = &args[index];
			index += 1;
			if subcommand.is_empty() {
				subcommand = arg.clone();
				continue;
			}
			if let Some(rest) = arg.strip_prefix("--") {
				if let Some((key, value)) = rest.split_once('=') {
					options.insert(key.to_string(), value.to_string());
				} else if index < args.len() && !args[index].starts_with("--") {
					options.insert(rest.to_string(), args[index].clone());
					index += 1;
				} else {
					options.insert(rest.to_string(), String::new());
				}
			} else {
				return Err(format!("unexpected positional argument: {arg}"));
			}
		}
		if subcommand.is_empty() {
			return Err(usage());
		}
		Ok(Cli { subcommand, options })
	}

	fn required(&self, key: &str) -> Result<String, String> {
		self.options
			.get(key)
			.cloned()
			.filter(|value| !value.is_empty())
			.ok_or_else(|| format!("missing required option --{key}"))
	}

	fn optional(&self, key: &str) -> Option<String> {
		self.options.get(key).cloned().filter(|value| !value.is_empty())
	}

	fn number_usize(&self, key: &str, default: usize) -> Result<usize, String> {
		match self.optional(key) {
			None => Ok(default),
			Some(raw) => raw.parse::<usize>().map_err(|error| format!("--{key}: {error}")),
		}
	}

	fn flag(&self, key: &str) -> bool {
		self.options.contains_key(key)
	}
}

fn usage() -> String {
	"usage:\n\
	disk_write_bench matrix --out <json> [--runs R] [--warmup W] [--alloc-runs A]\n\
	[--quick] [--keep-tmp]\n\
	disk_write_bench point --out <json> --name <point> [--runs R] [--warmup W]\n\
	[--alloc-runs A] [--keep-tmp]\n\
	disk_write_bench list\n\
	disk_write_bench help\n\
	defaults: runs=3 warmup=1 alloc-runs=1\n\
	--quick: reduced matrix (smoke)\n\
	--keep-tmp: keep the harness tmp tree after the run\n\
	tmp root: <system temp>/optimus-diskio-bench (wiped at start and exit)"
		.to_string()
}

fn prepare_tmp_root() -> Result<PathBuf, String> {
	let root = std::env::temp_dir().join("optimus-diskio-bench");
	if root.exists() {
		std::fs::remove_dir_all(&root).map_err(|error| format!("clean tmp root {}: {error}", root.display()))?;
	}
	std::fs::create_dir_all(&root).map_err(|error| format!("create tmp root: {error}"))?;
	Ok(root)
}

fn cleanup_tmp_root(root: &Path, keep: bool) {
	if !keep {
		let _ = std::fs::remove_dir_all(root);
	}
}

async fn run_matrix(cli: &Cli, quick: bool) -> Result<(), String> {
	let out = cli.required("out")?;
	let runs = cli.number_usize("runs", 3)?;
	let warmup = cli.number_usize("warmup", 1)?;
	let alloc_runs = cli.number_usize("alloc-runs", 1)?;
	let keep = cli.flag("keep-tmp");
	let root = prepare_tmp_root()?;
	let points = if quick { quick_matrix() } else { full_matrix() };
	println!(
		"# {} matrix: {} points, runs={} warmup={} alloc-runs={}",
		if quick { "quick" } else { "full" },
		points.len(),
		runs,
		warmup,
		alloc_runs
	);
	let options = RunOptions { runs, warmup, alloc_runs, root: root.clone() };
	let started = Instant::now();
	let mut point_reports = Vec::with_capacity(points.len());
	print_table_header();
	for point in &points {
		println!("# running {}", point.name);
		let report = run_point(point, &options).await?;
		print_point_row(&report);
		point_reports.push(report);
	}
	let args = json!({"quick": quick, "runs": runs, "warmup": warmup, "alloc_runs": alloc_runs});
	let report = build_report("matrix", &args, point_reports, &root);
	write_json(Path::new(&out), &report)?;
	cleanup_tmp_root(&root, keep);
	println!("matrix complete in {:.1}s, wrote {out}", started.elapsed().as_secs_f64());
	if !report_parity_ok(&report) {
		return Err("parity failure: see report".to_string());
	}
	Ok(())
}

async fn run_single_point(cli: &Cli) -> Result<(), String> {
	let out = cli.required("out")?;
	let name = cli.required("name")?;
	let runs = cli.number_usize("runs", 3)?;
	let warmup = cli.number_usize("warmup", 1)?;
	let alloc_runs = cli.number_usize("alloc-runs", 1)?;
	let keep = cli.flag("keep-tmp");
	let points = full_matrix();
	let point = points
		.iter()
		.find(|point| point.name == name)
		.ok_or_else(|| format!("unknown point: {name} (see `list`)"))?
		.clone();
	let root = prepare_tmp_root()?;
	let options = RunOptions { runs, warmup, alloc_runs, root: root.clone() };
	println!("# running {}", point.name);
	let report = run_point(&point, &options).await?;
	print_table_header();
	print_point_row(&report);
	let args = json!({"name": name, "runs": runs, "warmup": warmup, "alloc_runs": alloc_runs});
	let full = build_report("point", &args, vec![report], &root);
	write_json(Path::new(&out), &full)?;
	cleanup_tmp_root(&root, keep);
	if !report_parity_ok(&full) {
		return Err("parity failure: see report".to_string());
	}
	Ok(())
}

#[tokio::main]
async fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let cli = match Cli::parse(&args) {
		Ok(cli) => cli,
		Err(error) => {
			eprintln!("{error}");
			std::process::exit(2);
		}
	};
	let result = match cli.subcommand.as_str() {
		"matrix" => {
			let quick = cli.flag("quick");
			run_matrix(&cli, quick).await
		}
		"point" => run_single_point(&cli).await,
		"list" => {
			for point in full_matrix() {
				println!("{}", point.name);
			}
			Ok(())
		}
		"help" => {
			println!("{}", usage());
			Ok(())
		}
		other => Err(format!("unknown subcommand: {other}\n{}", usage())),
	};
	if let Err(error) = result {
		eprintln!("error: {error}");
		std::process::exit(1);
	}
}
