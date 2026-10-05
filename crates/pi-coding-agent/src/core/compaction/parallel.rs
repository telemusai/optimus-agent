//! Data-parallel execution for the local compaction pipeline.
//!
//! Scope: the local CPU stages of auto-compaction (token estimation, cut-point
//! walk, preparation extraction, summarizer serialization, session-context
//! restore) process long sessions with per-entry/per-message work that has no
//! order dependencies. This module runs that element-wise work on scoped std
//! threads.
//!
//! Contract (in order):
//! 1. **Identical results.** Element transforms are pure functions of their
//!    input; outputs are assembled in slice order, and reductions that produce
//!    floats are folded sequentially in the same order the sequential code
//!    used, so results are bit-identical.
//! 2. **Sequential fallback on any failure.** Below the size threshold, when
//!    parallelism is disabled, or when any worker errors or panics, the map
//!    returns `None` and the caller runs its unchanged sequential
//!    implementation for the whole operation. A parallel-only failure is
//!    therefore unobservable except as (identical) output plus a one-line
//!    diagnostic.
//! 3. **No new dependencies.** `std::thread::scope` only; workers hold
//!    disjoint output slots, so there is no shared mutable state.
//!
//! Runtime switches (read once per process):
//! - `PRIME_AGENT_COMPACTION_PARALLEL=0|off|false|sequential` disables every
//!   parallel site (forces the sequential path).
//! - `PRIME_AGENT_COMPACTION_PARALLEL=fault` makes every parallel site fail
//!   its admission check and fall back, which is how the fallback path is
//!   exercised end to end (tests use the seam below; the env value covers
//!   whole-process runs such as the replay harness).
//!
//! Size thresholds live here as named constants. Their values are the measured
//! crossover points from `reports/C1/crossover.md` (this repository's replay
//! harness, `bench-local` kernels, distinct before/after binaries); each is the
//! smallest input size at which the parallel path reproducibly beat the
//! sequential one with margin on the development host.

use pi_agent_core::types::AgentMessage;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::LazyLock;

/// Worker cap. The development host is a shared 6c/12t machine; the cap keeps
/// compaction parallelism at or below the physical core count.
const MAX_WORKER_THREADS: usize = 6;

/// Mode selected once per process from `PRIME_AGENT_COMPACTION_PARALLEL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Parallel sites active, gated by their size thresholds.
    Auto,
    /// Every site takes the sequential path.
    Sequential,
    /// Every site fails admission and falls back (fallback verification).
    Fault,
}

static MODE: LazyLock<Mode> = LazyLock::new(|| {
    match std::env::var("PRIME_AGENT_COMPACTION_PARALLEL")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "0" | "off" | "false" | "seq" | "sequential" => Mode::Sequential,
        "fault" | "fail" => Mode::Fault,
        _ => Mode::Auto,
    }
});

/// Test seam: force every parallel site to fail admission and fall back.
static FAULT_INJECTION: AtomicBool = AtomicBool::new(false);

/// Test seam: panic inside a parallel worker after admission, to prove the
/// scoped join survives and the sequential fallback still produces identical
/// output. Never set by production code.
static PANIC_INJECTION: AtomicBool = AtomicBool::new(false);

/// Number of times a parallel site fell back to the sequential path.
static FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Test seam for the fallback contract. Never called by production code.
#[doc(hidden)]
pub fn set_parallel_fault_injection(enabled: bool) {
    FAULT_INJECTION.store(enabled, Ordering::SeqCst);
}

/// Test seam for the worker-panic fallback path. Never called by production
/// code.
#[doc(hidden)]
pub fn set_parallel_panic_injection(enabled: bool) {
    PANIC_INJECTION.store(enabled, Ordering::SeqCst);
}

/// Number of sequential fallbacks taken by parallel sites so far.
/// Test/benchmark observability only.
#[doc(hidden)]
pub fn parallel_fallback_count() -> u64 {
    FALLBACK_COUNT.load(Ordering::SeqCst)
}

fn worker_threads() -> usize {
    static CACHED: LazyLock<usize> = LazyLock::new(|| {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .min(MAX_WORKER_THREADS)
    });
    *CACHED
}

fn record_fallback(reason: &'static str) {
    FALLBACK_COUNT.fetch_add(1, Ordering::SeqCst);
    eprintln!(
        "compaction parallel path fell back to sequential execution: {reason}"
    );
}

/// Element-wise map over `items` on scoped threads, preserving slice order.
///
/// Returns `None` (and the caller then runs its sequential implementation)
/// when `items.len() < threshold`, when parallelism is disabled or faulted,
/// or when any worker panics. Worker panics are caught so the scoped join
/// itself never unwinds.
pub(crate) fn try_parallel_map<T: Sync, R: Send>(
    items: &[T],
    threshold: usize,
    transform: impl Fn(&T, usize) -> R + Sync,
) -> Option<Vec<R>> {
    if items.len() < threshold || items.len() < 2 {
        return None;
    }
    if FAULT_INJECTION.load(Ordering::SeqCst) || *MODE == Mode::Fault {
        record_fallback("fault injection");
        return None;
    }
    if *MODE == Mode::Sequential {
        return None;
    }
    let workers = worker_threads().min(items.len());
    if workers < 2 {
        return None;
    }
    // Contiguous ranges keep each worker's memory accesses local and make the
    // ordered assembly below a concatenation.
    let chunk = items.len().div_ceil(workers);
    let ranges: Vec<std::ops::Range<usize>> = (0..workers)
        .map(|worker| {
            let start = worker * chunk;
            (start..(start + chunk).min(items.len()))
        })
        .take_while(|range| range.start < range.end)
        .collect();
    if ranges.len() < 2 {
        return None;
    }
    let mut slots: Vec<Option<Vec<R>>> = (0..ranges.len()).map(|_| None).collect();
    let failed = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for (slot, range) in slots.iter_mut().zip(ranges.iter()) {
            let transform = &transform;
            let failed = &failed;
            scope.spawn(move || {
                let rendered = catch_unwind(AssertUnwindSafe(|| {
                    if PANIC_INJECTION.load(Ordering::Relaxed) {
                        // Test seam only; see `set_parallel_panic_injection`.
                        panic!("compaction parallel worker panic injection");
                    }
                    items[range.start..range.end]
                        .iter()
                        .enumerate()
                        .map(|(offset, item)| transform(item, range.start + offset))
                        .collect::<Vec<R>>()
                }));
                match rendered {
                    Ok(rendered) => *slot = Some(rendered),
                    Err(_) => {
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    });
    if failed.load(Ordering::SeqCst) > 0 {
        record_fallback("worker panic");
        return None;
    }
    let mut output = Vec::with_capacity(items.len());
    for slot in slots {
        output.extend(slot?);
    }
    Some(output)
}

// ---------------------------------------------------------------------------
// Size gates (measured crossovers; evidence in reports/C1/crossover.md)
// ---------------------------------------------------------------------------
//
// Measurement basis: `compaction_replay bench-local` on the development host
// (shared 6c/12t EPYC 74F3, busy), min-of-15 iterations, sequential reference
// from the pre-change binary and from `PRIME_AGENT_COMPACTION_PARALLEL=0` on
// the post-change binary (both agree within noise). Parallel dispatch costs
// ~0.3-0.55 ms per site call on this host, so each gate is set at the smallest
// measured point where the parallel path beat sequential by >= ~1.25x, with
// the element count and byte terms that separated winners from losers.
//
// Byte terms use `approx_*_bytes` lower bounds (O(fields), no decoding), so a
// session just below the true byte crossover stays sequential.

/// Byte-gate scale shared by the byte-aware sites (24 MiB).
pub(crate) const LARGE_SESSION_BYTES: usize = 24 * 1024 * 1024;

/// Per-message token estimation, full-scan fallback: wins 2.9x at 37 MB,
/// loses below ~13 MB (0.3 ms work cannot amortize dispatch). Byte term
/// interpolated between the 13 MB loss and the 37 MB win.
pub(crate) const TOKEN_ESTIMATE_MESSAGES_THRESHOLD: usize = 512;
pub(crate) const TOKEN_ESTIMATE_BYTES_THRESHOLD: usize = 20 * 1024 * 1024;

/// Message extraction in `prepare_compaction`: wins 1.58x at 4096 entries /
/// 37 MB; even at 13 MB; loses below 10 MB.
pub(crate) const EXTRACT_MESSAGES_THRESHOLD: usize = 1024;
pub(crate) const EXTRACT_BYTES_THRESHOLD: usize = LARGE_SESSION_BYTES;

/// `convert_to_llm` per-message conversion: wins 1.44x at 4096 messages /
/// 37 MB; even at 9-13 MB.
pub(crate) const CONVERT_MESSAGES_THRESHOLD: usize = 1024;
pub(crate) const CONVERT_BYTES_THRESHOLD: usize = LARGE_SESSION_BYTES;

/// `serialize_conversation` part rendering: wins 1.36-1.62x at >= 8 MB with
/// >= 4 KB average message size; even-to-loss at 13 MB of 1.6 KB messages
/// (many tiny parts pay dispatch plus allocator contention per part).
pub(crate) const SERIALIZE_MESSAGES_THRESHOLD: usize = 256;
pub(crate) const SERIALIZE_BYTES_THRESHOLD: usize = 8 * 1024 * 1024;
pub(crate) const SERIALIZE_AVG_MESSAGE_BYTES: usize = 4096;

/// `get_branch` per-entry `Value` clone: allocation-count-bound. Wins from
/// 4096 entries regardless of size (1.38x at 0.6 MB, 1.27x at 13 MB), and
/// from 1024 entries when the entries are text-heavy (1.24x at 9.3 MB,
/// 1.43x at 37 MB); loses at 1024 x 1.7 MB and is even at 2048 x 1.2 MB.
pub(crate) const ENTRY_CLONE_THRESHOLD: usize = 4096;
pub(crate) const ENTRY_CLONE_HEAVY_COUNT: usize = 1024;
pub(crate) const ENTRY_CLONE_HEAVY_ENTRY_BYTES: usize = 4096;

/// `compaction_session_entry_from` per-entry parse: same per-entry `Value`
/// shape as the branch clone; contributes to the prepare-phase win at xlarge
/// (1276 entries / 32 MB, replay phase rows 99 -> 74 ms).
pub(crate) const ENTRY_PARSE_THRESHOLD: usize = ENTRY_CLONE_THRESHOLD;
pub(crate) const ENTRY_PARSE_HEAVY_COUNT: usize = ENTRY_CLONE_HEAVY_COUNT;
pub(crate) const ENTRY_PARSE_HEAVY_ENTRY_BYTES: usize = ENTRY_CLONE_HEAVY_ENTRY_BYTES;

/// Session-context restore, per-entry parse: wins 1.17-1.61x from 1536
/// entries (per-entry serde parse dominates, independent of total bytes);
/// 384-1024 entries sit within run-to-run noise.
pub(crate) const RESTORE_ENTRIES_THRESHOLD: usize = 1536;

/// Session-context restore, final per-message clone of the context messages.
pub(crate) const CONTEXT_CLONE_THRESHOLD: usize = 1536;

/// Cheap lower bound of a message's byte weight: text lengths only, plus a
/// fixed per-block allowance for tool-call arguments (serializing them would
/// cost a large fraction of the work being gated). O(fields), no decoding.
pub(crate) fn approx_agent_message_bytes(message: &AgentMessage) -> usize {
    use pi_agent_core::types::{CustomAgentMessage, CustomMessageContent};
    use pi_ai::types::{ContentBlock, ImageOrTextContent, Message, UserContent};
    match message {
        AgentMessage::Message(Message::User(user)) => match &user.content {
            UserContent::Text(text) => text.len(),
            UserContent::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    ImageOrTextContent::Text(text) => text.text.len(),
                    ImageOrTextContent::Image(_) => 4096,
                })
                .sum(),
        },
        AgentMessage::Message(Message::Assistant(assistant)) => assistant
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text(text) => text.text.len(),
                ContentBlock::Thinking(thinking) => thinking.thinking.len(),
                ContentBlock::ToolCall(call) => call.name.len() + 64,
            })
            .sum(),
        AgentMessage::Message(Message::ToolResult(result)) => result
            .content
            .iter()
            .map(|block| match block {
                ImageOrTextContent::Text(text) => text.text.len(),
                ImageOrTextContent::Image(_) => 4096,
            })
            .sum(),
        AgentMessage::Custom(CustomAgentMessage::BashExecution {
            command, output, ..
        }) => command.len() + output.len(),
        AgentMessage::Custom(CustomAgentMessage::Custom { content, .. }) => match content {
            CustomMessageContent::Text(text) => text.len(),
            CustomMessageContent::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    pi_agent_core::types::ContentBlock::Text(text) => text.text.len(),
                    pi_agent_core::types::ContentBlock::Image(_) => 4096,
                })
                .sum(),
        },
        AgentMessage::Custom(CustomAgentMessage::BranchSummary { summary, .. }) => summary.len(),
        AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
            summary,
            harness_digest,
            ..
        }) => summary.len() + harness_digest.as_deref().map_or(0, str::len),
    }
}

/// Byte lower bound for a slice of agent messages.
pub(crate) fn approx_agent_messages_bytes(messages: &[AgentMessage]) -> usize {
    messages.iter().map(approx_agent_message_bytes).sum()
}

/// Byte lower bound for converted LLM messages (the `serialize_conversation`
/// input shape).
pub(crate) fn approx_llm_messages_bytes(messages: &[pi_ai::types::Message]) -> usize {
    use pi_ai::types::{ContentBlock, ImageOrTextContent, Message, UserContent};
    messages
        .iter()
        .map(|message| match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => text.len(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|block| match block {
                        ImageOrTextContent::Text(text) => text.text.len(),
                        ImageOrTextContent::Image(_) => 4096,
                    })
                    .sum(),
            },
            Message::Assistant(assistant) => assistant
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::Text(text) => text.text.len(),
                    ContentBlock::Thinking(thinking) => thinking.thinking.len(),
                    ContentBlock::ToolCall(call) => call.name.len() + 64,
                })
                .sum(),
            Message::ToolResult(result) => result
                .content
                .iter()
                .map(|block| match block {
                    ImageOrTextContent::Text(text) => text.text.len(),
                    ImageOrTextContent::Image(_) => 4096,
                })
                .sum(),
        })
        .sum()
}

/// Recursive string-length sum of a JSON value: a cheap byte-weight estimate
/// for one entry (bounded cost; used only on a single probe entry, never over
/// a whole branch).
pub(crate) fn approx_value_string_bytes(value: &serde_json::Value) -> usize {
    use serde_json::Value;
    match value {
        Value::String(text) => text.len(),
        Value::Array(items) => items.iter().map(approx_value_string_bytes).sum(),
        Value::Object(fields) => fields
            .values()
            .map(approx_value_string_bytes)
            .sum::<usize>()
            + fields.len() * 16,
        Value::Number(_) => 24,
        Value::Bool(_) => 4,
        Value::Null => 0,
    }
}
