//! Timed write-buffer flushing.
//!
//! Shared infrastructure for the write buffers that coalesce small appends
//! (session JSONL lines, event-log lines) into fewer, larger write syscalls.
//! A single background thread polls registered buffers and flushes the ones
//! whose deadline passed. The thread is runtime-independent (std::thread) so
//! buffers work in tests without a tokio runtime; it is detached because a
//! process exit loses at most one flush period of buffered bytes, which is
//! the documented crash window of the buffering feature itself.

use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

/// How often the flusher thread wakes to check deadlines. Well below the
/// smallest supported flush period so deadline accuracy is not the limiter.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// A buffer that can flush itself when its deadline has passed.
pub(crate) trait TimedFlush: Send + Sync {
    /// Flush buffered bytes if the deadline passed; no-op otherwise.
    fn poll_flush(&self, now: Instant);
}

fn registry() -> &'static Mutex<Vec<Weak<dyn TimedFlush>>> {
    static REGISTRY: OnceLock<Mutex<Vec<Weak<dyn TimedFlush>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a buffer with the flusher thread. Spawns the thread on first use.
pub(crate) fn register_timed_flush(buffer: &Arc<dyn TimedFlush>) {
    {
        let mut registry = registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.retain(|weak| weak.strong_count() > 0);
        registry.push(Arc::downgrade(buffer));
    }
    spawn_flusher_once();
}

fn spawn_flusher_once() {
    static SPAWNED: OnceLock<()> = OnceLock::new();
    SPAWNED.get_or_init(|| {
        std::thread::Builder::new()
            .name("prime-write-flusher".to_string())
            .spawn(|| loop {
                std::thread::sleep(POLL_INTERVAL);
                let now = Instant::now();
                let buffers: Vec<Arc<dyn TimedFlush>> = {
                    let mut registry = registry()
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    registry.retain(|weak| weak.strong_count() > 0);
                    registry.iter().filter_map(|weak| weak.upgrade()).collect()
                };
                for buffer in buffers {
                    buffer.poll_flush(now);
                }
            })
            .expect("spawn prime-write-flusher thread");
    });
}

/// Default session-JSONL flush period, in milliseconds.
///
/// Probe-measured (R5 item 1; 40 events/s, release build, 800 appends per
/// phase): append latency plateaus at T >= 50 — session op_p50 508us at
/// write-through vs 156-167us for every T in {50,100,250,500,1000} — so past
/// 50 the only tradeoff is write-op count vs the crash window. Session lines
/// carry conversation entries, so the default stays conservative: 100ms is a
/// 5x write-op reduction (800 -> 161) with at most 100ms of entries at risk.
pub(crate) const SESSION_WRITE_FLUSH_MS: u64 = 100;

/// Default event-log flush period, in milliseconds.
///
/// The semantic-edge ledger is a crash-tolerant cache: `replay_sync` skips a
/// torn final line and the append path truncates it, so losing up to 250ms of
/// records to a crash costs only re-derivable edges. The same probe table
/// shows the deeper period buys an 11x write-op reduction (800 -> 73) at the
/// same plateau latency — the disk-churn reduction is the point on this
/// AV-on-write host.
pub(crate) const EVENT_LOG_WRITE_FLUSH_MS: u64 = 250;

/// Effective flush period from `PRIME_AGENT_WRITE_FLUSH_MS`, falling back to
/// `default_ms`. `0` disables coalescing (every append writes through). One
/// env knob deliberately governs both streams so benches and probes can pin
/// a uniform T across them.
fn write_flush_period_or(default_ms: u64) -> Duration {
    let ms = std::env::var("PRIME_AGENT_WRITE_FLUSH_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default_ms);
    Duration::from_millis(ms)
}

/// Flush period for the session JSONL write buffer.
pub(crate) fn session_write_flush_period() -> Duration {
    write_flush_period_or(SESSION_WRITE_FLUSH_MS)
}

/// Flush period for the event-log write buffers.
pub(crate) fn event_log_write_flush_period() -> Duration {
    write_flush_period_or(EVENT_LOG_WRITE_FLUSH_MS)
}

/// A byte buffer with a flush deadline. Shared shape of the session and
/// event-log buffers; the owner decides what a flush means.
pub(crate) struct TimedBytes {
    pub(crate) bytes: Vec<u8>,
    pub(crate) deadline: Option<Instant>,
}

impl TimedBytes {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            deadline: None,
        }
    }

    /// Push `line` (already newline-terminated) and arm the deadline for the
    /// first pending byte. Leading-edge policy: the deadline covers the oldest
    /// unflushed byte, so every byte lands within one period of its append.
    pub(crate) fn push(&mut self, line: &[u8], period: Duration, now: Instant) {
        if self.bytes.is_empty() {
            self.deadline = Some(now + period);
        }
        self.bytes.extend_from_slice(line);
    }
}
