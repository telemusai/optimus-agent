//! Bounded, aggregated diagnostics for repeated warnings.
//!
//! A per-entry warning inside a loop can dominate a log (an observed 25-minute
//! burst of 10,530 projection warnings filled 97.8% of one daemon log). Callers
//! aggregate their own counts and route the single summary line through
//! [`limited_warning`], which keeps at most one line per key per window and
//! reports how many identical repeats it suppressed.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default suppression window for one key.
pub const DEFAULT_WARNING_WINDOW_MS: u64 = 60_000;
/// Upper bound on tracked keys; stale entries are pruned before the map grows.
const MAX_TRACKED_KEYS: usize = 256;

#[derive(Default)]
struct LimiterEntry {
    last_ms: u64,
    suppressed: u64,
}

fn limiter_state() -> &'static Mutex<HashMap<String, LimiterEntry>> {
    static STATE: OnceLock<Mutex<HashMap<String, LimiterEntry>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `Date.now()` in milliseconds.
pub fn wall_clock_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

/// Suppress repeats of the same `scope`/`key` inside `window_ms`.
///
/// Returns `Some(line)` when the warning should be emitted. `line` is `message`,
/// extended with the number of identical warnings suppressed since the last
/// emission, so a bounded log still reports the true repeat count. Returns
/// `None` while the window is still open.
pub fn limited_warning(
    scope: &str,
    key: &str,
    message: &str,
    now_ms: u64,
    window_ms: u64,
) -> Option<String> {
    let window_ms = window_ms.max(1);
    let composed_key = format!("{scope}:{key}");
    let mut state = match limiter_state().lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    if !state.contains_key(&composed_key) && state.len() >= MAX_TRACKED_KEYS {
        // Prune expired keys first. If the map is still full, report without
        // tracking: an untracked warning is never hidden by the limiter.
        let cutoff = now_ms.saturating_sub(window_ms);
        state.retain(|_, entry| entry.last_ms >= cutoff);
        if state.len() >= MAX_TRACKED_KEYS {
            return Some(message.to_string());
        }
    }
    let entry = state.entry(composed_key).or_default();
    if entry.last_ms != 0 && now_ms.saturating_sub(entry.last_ms) < window_ms {
        entry.suppressed = entry.suppressed.saturating_add(1);
        return None;
    }
    let suppressed = entry.suppressed;
    entry.last_ms = now_ms;
    entry.suppressed = 0;
    if suppressed == 0 {
        return Some(message.to_string());
    }
    Some(format!(
        "{message} (+{suppressed} identical warning(s) suppressed in the last {}s)",
        window_ms / 1000
    ))
}

/// Serialises tests that share the process-global limiter.
///
/// `cargo test` runs test functions on parallel threads inside one process, so a
/// test that clears or fills the map would otherwise change another test's
/// result. Cross-module tests that exercise the limiter take this guard first.
#[cfg(test)]
pub fn limiter_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Forget every tracked key. Tests use this to start from a clean limiter.
pub fn reset_limited_warnings() {
    let mut state = match limiter_state().lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    state.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    // The limiter is process-global, so every test takes `limiter_test_lock()`
    // (cross-module, same process) and then starts from a clean map.

    #[test]
    fn first_warning_is_emitted_and_repeats_are_suppressed() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        assert_eq!(
            limited_warning("first", "sig", "Warning: 3 entries dropped", 1_000, 60_000),
            Some("Warning: 3 entries dropped".to_string())
        );
        for offset in 1..50u64 {
            assert_eq!(
                limited_warning(
                    "first",
                    "sig",
                    "Warning: 3 entries dropped",
                    1_000 + offset,
                    60_000
                ),
                None
            );
        }
    }

    #[test]
    fn next_emission_reports_the_suppressed_repeat_count() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        limited_warning("second", "sig", "Warning: dropped", 1_000, 60_000);
        for offset in 1..=4u64 {
            limited_warning("second", "sig", "Warning: dropped", 1_000 + offset, 60_000);
        }
        assert_eq!(
            limited_warning("second", "sig", "Warning: dropped", 61_000, 60_000),
            Some(
                "Warning: dropped (+4 identical warning(s) suppressed in the last 60s)".to_string()
            )
        );
        // The counter resets after an emission.
        assert_eq!(
            limited_warning("second", "sig", "Warning: dropped", 122_000, 60_000),
            Some("Warning: dropped".to_string())
        );
    }

    #[test]
    fn distinct_scopes_and_keys_are_independent() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        assert!(limited_warning("third-a", "k1", "one", 1_000, 60_000).is_some());
        assert!(limited_warning("third-a", "k2", "two", 1_000, 60_000).is_some());
        assert!(limited_warning("third-b", "k1", "three", 1_000, 60_000).is_some());
        assert!(limited_warning("third-a", "k1", "one", 1_001, 60_000).is_none());
    }

    #[test]
    fn tracked_keys_stay_bounded() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        // A distinct scope keeps this test's keys separate from the other tests'
        // scopes; the bound is what is asserted, not the exact map contents.
        for index in 0..(MAX_TRACKED_KEYS * 2) {
            let key = format!("k{index}");
            limited_warning("bounded", &key, "message", 10_000, 60_000);
        }
        let state = limiter_state().lock().unwrap();
        assert!(
            state.len() <= MAX_TRACKED_KEYS,
            "tracked keys grew to {}",
            state.len()
        );
    }

    #[test]
    fn an_untrackable_key_is_reported_rather_than_hidden() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        // Fill the map, then confirm a brand-new key at a later time is still
        // emitted (an untracked warning must never be swallowed by the limiter).
        for index in 0..(MAX_TRACKED_KEYS * 2) {
            limited_warning("overflow", &format!("k{index}"), "message", 1_000, 60_000);
        }
        let line = limited_warning("overflow", "fresh-key", "message", 1_000, 60_000);
        assert!(
            line.is_some(),
            "a key the limiter cannot track must still be reported"
        );
    }
}
