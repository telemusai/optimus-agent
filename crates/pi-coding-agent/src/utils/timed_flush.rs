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
        let mut registry = registry().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
                    let mut registry =
                        registry().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    registry.retain(|weak| weak.strong_count() > 0);
                    registry
                        .iter()
                        .filter_map(|weak| weak.upgrade())
                        .collect()
                };
                for buffer in buffers {
                    buffer.poll_flush(now);
                }
            })
            .expect("spawn prime-write-flusher thread");
    });
}

/// The default flush period for buffered appends.
// PROBE-PENDING: placeholder until the T probe (R5 item 1) measures the knee;
// the shipped default must be the measured value, not this hypothesis.
pub(crate) const DEFAULT_WRITE_FLUSH_MS: u64 = 250;

/// Effective flush period from `PRIME_AGENT_WRITE_FLUSH_MS`, falling back to
/// the default. `0` disables buffering-time coalescing (every append flushes).
pub(crate) fn write_flush_period() -> Duration {
    let ms = std::env::var("PRIME_AGENT_WRITE_FLUSH_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_WRITE_FLUSH_MS);
    Duration::from_millis(ms)
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
