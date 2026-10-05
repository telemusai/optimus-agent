//! Byte-preserving SSE line framing. JSON and provider error policy belong to callers.

#[derive(Default)]
pub(crate) struct SseFrames {
    line: Vec<u8>,
    data: Vec<String>,
    skip_lf: bool,
}

impl SseFrames {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut frames = Vec::new();
        for &byte in bytes {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            if byte != b'\r' && byte != b'\n' {
                self.line.push(byte);
                continue;
            }
            self.skip_lf = byte == b'\r';
            if self.line.is_empty() {
                if let Some(data) = self.take_data() {
                    frames.push(data);
                }
            } else {
                self.finish_line();
            }
        }
        frames
    }

    // Bedrock historically decodes a final undelimited payload; Codex does not.
    pub(crate) fn finish(&mut self) -> Vec<String> {
        if !self.line.is_empty() {
            self.finish_line();
        }
        self.take_data().into_iter().collect()
    }

    fn finish_line(&mut self) {
        let line = String::from_utf8_lossy(&self.line);
        if let Some(data) = line.strip_prefix("data:") {
            self.data.push(data.strip_prefix(' ').unwrap_or(data).to_string());
        }
        self.line.clear();
    }

    fn take_data(&mut self) -> Option<String> {
        let data = self.data.join("\n");
        self.data.clear();
        if data.trim().is_empty() || data.trim() == "[DONE]" {
            None
        } else {
            Some(data)
        }
    }
}

// ---------------------------------------------------------------------------
// Inter-chunk inactivity deadline for SSE response bodies
// ---------------------------------------------------------------------------

/// Environment override for the SSE inter-chunk inactivity deadline, in milliseconds.
pub(crate) const ENV_SSE_IDLE_TIMEOUT_MS: &str = "PRIME_AGENT_SSE_IDLE_TIMEOUT_MS";

/// Default inter-chunk inactivity deadline. The request-timeout deadline of the
/// OpenAI-style SDKs covers CONNECT + response headers only; the body that follows
/// has no deadline, so a silently stalled body otherwise hangs a request until the
/// transport or server gives up (measured: attempts failing after ~1,483,861ms and
/// ~1,449,350ms on 2026-09-24, sessions 01a0d54e/01a0d550).
pub(crate) const DEFAULT_SSE_IDLE_TIMEOUT_MS: u64 = 120_000;
const SSE_IDLE_TIMEOUT_MIN_MS: u64 = 5_000;
const SSE_IDLE_TIMEOUT_MAX_MS: u64 = 600_000;

/// Test-only injection point: a positive value replaces both the default and the
/// environment variable, bypassing the clamp so tests can use sub-second budgets.
#[cfg(test)]
pub(crate) static SSE_IDLE_TIMEOUT_TEST_OVERRIDE_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Resolve the inter-chunk inactivity deadline for one SSE response body.
///
/// The deadline is reset by every received chunk, so a healthy stream that keeps
/// producing data (even long thinking gaps that still emit keep-alive comments or
/// empty deltas) never trips it. Only a fully silent body aborts, which is already
/// a failed attempt by every observed measurement.
pub(crate) fn resolve_sse_idle_timeout() -> std::time::Duration {
    #[cfg(test)]
    {
        let override_ms = SSE_IDLE_TIMEOUT_TEST_OVERRIDE_MS.load(std::sync::atomic::Ordering::Relaxed);
        if override_ms > 0 {
            return std::time::Duration::from_millis(override_ms);
        }
    }
    let requested = std::env::var(ENV_SSE_IDLE_TIMEOUT_MS)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_SSE_IDLE_TIMEOUT_MS);
    // Hard clamp: an accidental 0/negative/garbage value must not disable the guard,
    // and no value may exceed the 10-minute ceiling the header deadline already uses.
    std::time::Duration::from_millis(requested.clamp(SSE_IDLE_TIMEOUT_MIN_MS, SSE_IDLE_TIMEOUT_MAX_MS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiline_comments_mixed_terminators_and_done_are_preserved() {
        let wire = b":comment\r\ndata: {\rdata: \"delta\":\"ok\"}\n\rdata: [DONE]\r\r";
        let mut frames = SseFrames::default();
        let result: Vec<_> = wire.iter().flat_map(|byte| frames.push(&[*byte])).collect();
        assert_eq!(result, vec!["{\n\"delta\":\"ok\"}"]);
    }

    #[test]
    fn finish_decodes_only_the_remaining_tail_once() {
        let mut frames = SseFrames::default();
        assert!(frames.push("data: \"日本\"".as_bytes()).is_empty());
        assert_eq!(frames.finish(), vec!["\"日本\""]);
        assert!(frames.finish().is_empty());
    }

    #[test]
    fn idle_timeout_test_override_default_env_and_clamp_are_sequential() {
        // One sequential test: the resolver state (env + test override) is process
        // global, so parallel #[test]s mutating it would race each other.
        use std::sync::atomic::Ordering;

        let saved_override = SSE_IDLE_TIMEOUT_TEST_OVERRIDE_MS.load(Ordering::Relaxed);
        SSE_IDLE_TIMEOUT_TEST_OVERRIDE_MS.store(250, Ordering::Relaxed);
        assert_eq!(resolve_sse_idle_timeout(), std::time::Duration::from_millis(250));
        SSE_IDLE_TIMEOUT_TEST_OVERRIDE_MS.store(0, Ordering::Relaxed);

        // The default matches the review evidence: a 120s inactivity budget, far
        // below the observed 10-25 minute silent stalls.
        std::env::remove_var(ENV_SSE_IDLE_TIMEOUT_MS);
        assert_eq!(resolve_sse_idle_timeout(), std::time::Duration::from_millis(120_000));

        // 0 clamps to the 5s floor; it never disables the guard.
        std::env::set_var(ENV_SSE_IDLE_TIMEOUT_MS, "0");
        assert_eq!(resolve_sse_idle_timeout(), std::time::Duration::from_millis(5_000));
        // Garbage and negative values fall back to the safe default rather
        // than disabling the guard.
        for value in ["garbage", "-1"] {
            std::env::set_var(ENV_SSE_IDLE_TIMEOUT_MS, value);
            assert_eq!(
                resolve_sse_idle_timeout(),
                std::time::Duration::from_millis(120_000),
                "{value}"
            );
        }

        std::env::set_var(ENV_SSE_IDLE_TIMEOUT_MS, "30000");
        assert_eq!(resolve_sse_idle_timeout(), std::time::Duration::from_millis(30_000));

        // No value may exceed the 10-minute ceiling the header deadline already uses.
        std::env::set_var(ENV_SSE_IDLE_TIMEOUT_MS, "3600000");
        assert_eq!(
            resolve_sse_idle_timeout(),
            std::time::Duration::from_millis(600_000)
        );

        std::env::remove_var(ENV_SSE_IDLE_TIMEOUT_MS);
        SSE_IDLE_TIMEOUT_TEST_OVERRIDE_MS.store(saved_override, Ordering::Relaxed);
    }
}
