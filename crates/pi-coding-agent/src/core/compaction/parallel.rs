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
// Size thresholds (measured crossovers; see reports/C1/crossover.md)
// ---------------------------------------------------------------------------

/// Per-message token estimation (`estimate_tokens` maps over whole sessions).
pub(crate) const TOKEN_ESTIMATE_MESSAGES_THRESHOLD: usize = 512;

/// Message extraction in `prepare_compaction` (per-entry clone into the
/// summarizer input).
pub(crate) const EXTRACT_MESSAGES_THRESHOLD: usize = 256;

/// `convert_to_llm` per-message conversion (and the identity policy clone).
pub(crate) const CONVERT_MESSAGES_THRESHOLD: usize = 256;

/// `serialize_conversation` per-message part rendering.
pub(crate) const SERIALIZE_MESSAGES_THRESHOLD: usize = 256;

/// `get_branch` per-entry `Value` clone.
pub(crate) const ENTRY_CLONE_THRESHOLD: usize = 512;

/// `compaction_session_entry_from` per-entry parse.
pub(crate) const ENTRY_PARSE_THRESHOLD: usize = 256;

/// Session-context restore: per-entry `append_message` parse/clone.
pub(crate) const RESTORE_ENTRIES_THRESHOLD: usize = 256;

/// Session-context restore: final per-message clone of the context messages.
pub(crate) const CONTEXT_CLONE_THRESHOLD: usize = 512;
