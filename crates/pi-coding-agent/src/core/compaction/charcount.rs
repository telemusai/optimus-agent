//! Dispatched Unicode-scalar counting and scanning for the compaction hot paths.
//!
//! Every character walk in auto-compaction goes through this module: token
//! estimates and cut-point costs (`count_chars`, `count_json_chars` for
//! serialized tool-call arguments and tool schemas), summary truncation and
//! chunking (`scan_chars_forward`, `scan_chars_backward` and the str helpers
//! `prefix_end`, `suffix_start`, `take_chars`, `last_chars`), typed and
//! task-state limits, and the Jev compaction fingerprint
//! (`count_noncontinuation` over raw streamed bytes). All of them reduce to
//! one byte classifier: a UTF-8 non-continuation byte (`(b & 0xC0) != 0x80`)
//! starts exactly one Unicode scalar of any valid UTF-8 string.
//!
//! Layout for kernel work:
//! - `charcount/x86_64.s` + `charcount/asm.rs`: the hand-written x86-64
//!   assembly kernels (SSE2 count, AVX2 count, AVX2 forward and backward
//!   scans) and their safe wrappers. Built when the `compaction-asm` feature
//!   is on (the default) and not under Miri.
//! - `charcount/x86.rs`: the Rust-intrinsic x86-64 SIMD kernels (SSE2/AVX2
//!   counts, AVX2 forward and backward scans) plus the kernel contract table.
//!   They are the fallback when the assembly is off or rejected.
//! - `charcount/neon.rs`: the only aarch64 SIMD code (NEON count).
//! - this file: safe wrappers, the scalar reference walks, dispatch, the
//!   startup self-check, and the differential tests. A new backend replaces
//!   kernel functions only; everything here stays unchanged.
//!
//! Backends, selected once per process, first candidate that passes the
//! self-check wins:
//! - x86-64, in order:
//!   1. `asm-avx2`: the assembly kernels, when AVX2, BMI1, BMI2, POPCNT and
//!      LZCNT are all detected (counts and scans at every length).
//!   2. `avx2`: Rust AVX2 intrinsics when AVX2 is detected (counts for
//!      `len >= 64`, scans always).
//!   3. `asm-sse2`: the assembly SSE2 count at every length (SSE2 is part of
//!      the x86-64 baseline, so it needs no runtime detection); scalar scans.
//!   4. `sse2`: Rust SSE2 intrinsics (counts for `len >= 16`); scalar scans.
//!   Below the Rust kernels' lengths a scalar loop runs.
//!   `PRIME_AGENT_COMPACT_ASM=0` (or `false`) removes the two assembly
//!   candidates for the process.
//! - aarch64: NEON counts. NEON is baseline hardware on aarch64 (enabled by
//!   default for the target), so the kernel needs no runtime feature check.
//!   Scans use the scalar walk.
//! - other architectures: the portable `chars().count()` reference, which the
//!   compiler autovectorizes on most targets, and the scalar scans.
//!
//! Selection happens behind a `OnceLock` that first runs a differential
//! self-check of each candidate backend against `chars().count()` on fixed
//! edge-case fixtures (empty input, every byte length 1..=64, load alignments
//! 0..=31, NUL, every UTF-8 scalar width, and a > 128 KiB mixed buffer so the
//! aarch64 fold path is exercised), and of its forward and backward scans
//! against the scalar walks. A backend that disagrees on any fixture is
//! disabled and the next candidate is tried; the compact diagnostic (no
//! transcript content) is retrievable via [`diagnostic`].
//! `PRIME_AGENT_COMPACT_RUST_COUNT=1` (or `true`) pins the reference
//! implementation regardless of hardware (it overrides
//! `PRIME_AGENT_COMPACT_ASM`).
//!
//! Every counting result is bit-identical to `chars().count()` for valid
//! UTF-8, and every scan result to the `char_indices()` walk it replaced;
//! nothing else about the compaction path changes.

use std::sync::OnceLock;

/// Active counting backend after the one-time startup self-check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable `chars().count()` (compiler-autovectorized on most targets).
    Reference,
    /// Explicit SSE2 kernel (x86-64 baseline feature).
    #[cfg(target_arch = "x86_64")]
    Sse2,
    /// Explicit AVX2 kernel (runtime-detected).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Assembly SSE2 count (x86-64 baseline); scalar scans.
    #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
    AsmSse2,
    /// Assembly AVX2 count and scans (AVX2+BMI1+BMI2+POPCNT+LZCNT detected).
    #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
    AsmAvx2,
    /// NEON kernel (baseline on aarch64).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Reference => "reference",
            #[cfg(target_arch = "x86_64")]
            Backend::Sse2 => "sse2",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => "avx2",
            #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
            Backend::AsmSse2 => "asm-sse2",
            #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
            Backend::AsmAvx2 => "asm-avx2",
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => "neon",
        }
    }
}

/// `PRIME_AGENT_COMPACT_RUST_COUNT=1|true` pins the reference implementation.
/// Read once at self-check time, never per call.
fn forced_reference() -> bool {
    std::env::var("PRIME_AGENT_COMPACT_RUST_COUNT")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// `PRIME_AGENT_COMPACT_ASM=0|false` (any case) disables the assembly
/// backends. Any other value, or no value, leaves them enabled.
#[cfg_attr(
    not(all(target_arch = "x86_64", feature = "compaction-asm", not(miri))),
    allow(dead_code)
)]
fn asm_switch_disables(value: &str) -> bool {
    value == "0" || value.eq_ignore_ascii_case("false")
}

/// Read once per candidate list (the self-check), never per call.
#[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
fn asm_disabled() -> bool {
    std::env::var("PRIME_AGENT_COMPACT_ASM")
        .map(|value| asm_switch_disables(&value))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Reference and scalar helpers
// ---------------------------------------------------------------------------

#[inline]
fn count_chars_reference(s: &str) -> usize {
    s.chars().count()
}

#[inline]
fn count_noncontinuation_scalar(bytes: &[u8]) -> usize {
    bytes.iter().filter(|byte| **byte & 0xc0 != 0x80).count()
}

/// Scalar forward scan shared by every non-AVX2 backend: byte offset just
/// past the `target`-th non-continuation byte at/after `start` (the trailing
/// char's continuation bytes are included), plus how many were found.
fn scan_chars_forward_scalar(bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    let mut pos = start;
    let mut seen = 0usize;
    while pos < bytes.len() && seen < target {
        if bytes[pos] & 0xc0 != 0x80 {
            seen += 1;
        }
        pos += 1;
    }
    while pos < bytes.len() && bytes[pos] & 0xc0 == 0x80 {
        pos += 1;
    }
    (pos, seen)
}

/// Scalar backward scan shared by every non-AVX2 backend: byte offset of the
/// `target`-th non-continuation byte counted backwards from `end`
/// (exclusive), plus how many were found. Returns `(end, 0)` for
/// `target == 0` and `(0, found)` when `bytes[..end]` holds fewer than
/// `target`. Requires `end <= bytes.len()` (the public wrapper clamps it).
fn scan_chars_backward_scalar(bytes: &[u8], end: usize, target: usize) -> (usize, usize) {
    if target == 0 {
        return (end, 0);
    }
    let mut pos = end;
    let mut seen = 0usize;
    while pos > 0 {
        pos -= 1;
        if bytes[pos] & 0xc0 != 0x80 {
            seen += 1;
            if seen == target {
                return (pos, seen);
            }
        }
    }
    (0, seen)
}

// ---------------------------------------------------------------------------
// Architecture kernels
//
// Every SIMD instruction of the compaction character work lives in one file
// per architecture: `charcount/x86.rs` (SSE2/AVX2) and `charcount/neon.rs`.
// The kernel contract is documented at the top of `charcount/x86.rs`; an
// assembly backend replaces those functions and keeps the safe wrappers,
// dispatch, self-check and tests below unchanged.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
mod asm;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Dispatch thresholds from the kernel-lab measurements: a scalar loop beats
/// SSE2 setup below 16 bytes and AVX2 setup below 64 bytes. The SSE2 path
/// stays a valid choice for 16..63-byte slices on hosts without AVX2.
#[cfg(target_arch = "x86_64")]
fn count_noncontinuation_sse2(bytes: &[u8]) -> usize {
    if bytes.len() < 16 {
        return count_noncontinuation_scalar(bytes);
    }
    // SAFETY: SSE2 is baseline on x86-64; loads are chunks_exact(16)-bounded.
    unsafe { x86::count_noncontinuation_sse2(bytes) }
}

#[cfg(target_arch = "x86_64")]
fn count_noncontinuation_avx2(bytes: &[u8]) -> usize {
    if bytes.len() < 64 {
        return count_noncontinuation_scalar(bytes);
    }
    // SAFETY: the dispatch state only selects the AVX2 backend after
    // `is_x86_feature_detected!("avx2")` and a passed self-check.
    unsafe { x86::count_noncontinuation_avx2(bytes) }
}

#[cfg(target_arch = "x86_64")]
fn scan_noncontinuation_avx2(bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    // SAFETY: same dispatch-state guarantee as `count_noncontinuation_avx2`.
    unsafe { x86::scan_noncontinuation_avx2(bytes, start, target) }
}

#[cfg(target_arch = "x86_64")]
fn scan_noncontinuation_backward_avx2(bytes: &[u8], end: usize, target: usize) -> (usize, usize) {
    // SAFETY: same dispatch-state guarantee as `count_noncontinuation_avx2`;
    // the public wrapper has already clamped `end <= bytes.len()`.
    unsafe { x86::scan_noncontinuation_backward_avx2(bytes, end, target) }
}

#[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
fn count_noncontinuation_asm_avx2(bytes: &[u8]) -> usize {
    // SAFETY: the dispatch state only selects `AsmAvx2` after
    // `asm::asm_avx2_supported()` and a passed self-check. No length
    // threshold: the assembly handles every length itself.
    unsafe { asm::count_noncontinuation_avx2(bytes) }
}

#[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
fn scan_noncontinuation_asm_avx2(bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    // SAFETY: same dispatch-state guarantee as `count_noncontinuation_asm_avx2`.
    unsafe { asm::scan_noncontinuation_avx2(bytes, start, target) }
}

#[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
fn scan_noncontinuation_backward_asm_avx2(bytes: &[u8], end: usize, target: usize) -> (usize, usize) {
    // SAFETY: same dispatch-state guarantee as `count_noncontinuation_asm_avx2`.
    unsafe { asm::scan_noncontinuation_backward_avx2(bytes, end, target) }
}

#[cfg(target_arch = "aarch64")]
fn count_noncontinuation_neon(bytes: &[u8]) -> usize {
    // SAFETY: NEON is baseline on aarch64; loads are chunks_exact(16)-bounded.
    unsafe { neon::count_noncontinuation_neon(bytes) }
}

// ---------------------------------------------------------------------------
// Dispatch: one-time backend selection + startup self-check
// ---------------------------------------------------------------------------

struct DispatchState {
    backend: Backend,
    diagnostic: Option<&'static str>,
}

fn candidate_backends() -> Vec<Backend> {
    let mut candidates = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        {
            if !asm_disabled() && asm::asm_avx2_supported() {
                candidates.push(Backend::AsmAvx2);
            }
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            candidates.push(Backend::Avx2);
        }
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        {
            if !asm_disabled() {
                candidates.push(Backend::AsmSse2);
            }
        }
        candidates.push(Backend::Sse2);
    }
    #[cfg(target_arch = "aarch64")]
    {
        candidates.push(Backend::Neon);
    }
    candidates
}

fn run_backend(backend: Backend, bytes: &[u8]) -> Option<usize> {
    match backend {
        Backend::Reference => Some(count_noncontinuation_scalar(bytes)),
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => Some(count_noncontinuation_sse2(bytes)),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => Some(count_noncontinuation_avx2(bytes)),
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        Backend::AsmSse2 => Some(asm::count_noncontinuation_sse2(bytes)),
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        Backend::AsmAvx2 => Some(count_noncontinuation_asm_avx2(bytes)),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => Some(count_noncontinuation_neon(bytes)),
    }
}

/// Fixed edge-case fixtures for the self-check (and the unit tests): empty,
/// NUL, every byte length 1..=64 over a mixed alphabet (all tails and block
/// boundaries), load alignments 0..=31 in front of a multibyte payload, every
/// UTF-8 scalar width and its boundary scalars, a 4 KiB deterministic mixed
/// buffer, and a >= 128 KiB repeat of it (beyond the NEON fold interval) so
/// the self-check covers the fold path on aarch64.
fn self_check_fixtures() -> Vec<String> {
    let mut fixtures: Vec<String> = vec![
        String::new(),
        "a".to_string(),
        "\u{0}".to_string(),
        "ab".to_string(),
        "abc".to_string(),
        "\n\r\t".to_string(),
        "hello world".to_string(),
        "\u{e9}\u{e8}\u{ea}".to_string(),
        "\u{4e2d}\u{6587}\u{6d4b}\u{8bd5}".to_string(),
        "\u{1f600}\u{1f9d1}\u{200d}\u{1f4bb}".to_string(),
        "a\u{301}e\u{308}i\u{302}".to_string(),
        "\u{10ffff}".to_string(),
        "\u{80}".to_string(),
        "\u{7ff}".to_string(),
        "\u{800}".to_string(),
        "\u{ffff}".to_string(),
        "\u{10000}".to_string(),
    ];
    let alphabet: Vec<char> = "aZ09 \n\"\u{e9}\u{4e2d}\u{1f600}\u{0301}\u{0}\u{7f}"
        .chars()
        .collect();
    for len in 1..=64usize {
        let mut s = String::new();
        for i in 0..len {
            s.push(alphabet[i % alphabet.len()]);
        }
        fixtures.push(s);
    }
    let payload = "\u{4e2d}\u{1f600}abc\u{e9}\u{0301}def\u{10ffff}";
    for pad in 0..=31usize {
        fixtures.push(format!("{}{}", "x".repeat(pad), payload));
    }
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut big = String::new();
    let pool: Vec<char> =
        "abcdefgh \n\"\u{4e2d}\u{6587}\u{1f600}\u{1f44d}\u{0301}\u{0308}\u{0}\u{7f}\u{2603}"
            .chars()
            .collect();
    while big.len() < 4096 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        big.push(pool[(state % pool.len() as u64) as usize]);
    }
    fixtures.push(big.clone());
    // A buffer beyond the NEON fold interval (4096 blocks x 16 B = 65 536 B)
    // so the per-process self-check executes the aarch64 kernel's u16-lane
    // fold path (and the unrolled loops at >64 KiB sizes on x86-64). Built by
    // repeating the 4 KiB buffer: deterministic, and generation is a memcpy.
    fixtures.push(big.repeat(32));
    fixtures
}

/// Raw-byte fixtures for the byte-slice entry point (`count_noncontinuation`):
/// the empty slice, every single byte value, the full 0..=255 sequence at
/// several rotations (invalid UTF-8 included — the kernels classify each
/// byte independently, so the reference is the scalar byte filter), every
/// length 1..=64 of a mixed alphabet containing NUL, continuation bytes and
/// lead bytes, and a > 128 KiB deterministic buffer beyond the aarch64 fold
/// interval. These harden the per-process self-check for the raw-byte path
/// the fingerprint writer depends on.
fn self_check_byte_fixtures() -> Vec<Vec<u8>> {
    let mut fixtures: Vec<Vec<u8>> = vec![Vec::new()];
    for byte in 0..=255u8 {
        fixtures.push(vec![byte]);
    }
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    for rotation in [0usize, 1, 3, 16, 17, 31] {
        let mut rotated = all_bytes.clone();
        rotated.rotate_left(rotation);
        fixtures.push(rotated);
    }
    let alphabet: [u8; 12] = [
        b'a', b'\n', 0x00, 0x7f, 0x80, 0x8f, 0xbf, 0xc2, 0xe0, 0xf0, 0xfd, 0xff,
    ];
    for len in 1..=64usize {
        let mut bytes = Vec::with_capacity(len);
        for i in 0..len {
            bytes.push(alphabet[i % alphabet.len()]);
        }
        fixtures.push(bytes);
    }
    // > 128 KiB of deterministic mixed bytes (xorshift), beyond the NEON
    // fold interval (4096 blocks x 16 B = 65 536 B) so the fold path runs.
    let mut state = 0x0fed_cba9_8765_4321u64;
    let mut big = Vec::with_capacity(128 * 1024 + 64);
    while big.len() < 128 * 1024 + 64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        big.extend_from_slice(&state.to_le_bytes());
    }
    fixtures.push(big);
    fixtures
}

/// Forward scan through one backend's kernel (scalar for backends without a
/// scan kernel). Shared by the self-check and the public dispatch so both
/// exercise exactly the same code.
///
/// The SIMD kernels require `start <= len` and `start` at `len` or at a
/// non-continuation byte; every production caller passes a char boundary.
/// Any other `start` takes the scalar walk here, so the result equals the
/// scalar walk for every input on every backend.
fn backend_scan_forward(backend: Backend, bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    match bytes.get(start) {
        Some(byte) if byte & 0xc0 != 0x80 => {}
        None if start == bytes.len() => {}
        _ => return scan_chars_forward_scalar(bytes, start, target),
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => scan_noncontinuation_avx2(bytes, start, target),
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        Backend::AsmAvx2 => scan_noncontinuation_asm_avx2(bytes, start, target),
        _ => scan_chars_forward_scalar(bytes, start, target),
    }
}

/// Backward scan through one backend's kernel (scalar for backends without
/// a scan kernel). `end` must already be clamped to `bytes.len()`.
fn backend_scan_backward(backend: Backend, bytes: &[u8], end: usize, target: usize) -> (usize, usize) {
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => scan_noncontinuation_backward_avx2(bytes, end, target),
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        Backend::AsmAvx2 => scan_noncontinuation_backward_asm_avx2(bytes, end, target),
        _ => scan_chars_backward_scalar(bytes, end, target),
    }
}

/// Scan fixtures up to this size get the full start/target matrix in the
/// self-check. Larger fixtures get only four long walks: the scan kernels
/// carry no long-range state (only the running `seen` count), so the 4 KiB
/// fixtures already cover many full blocks, and the startup cost stays
/// small.
const SCAN_CHECK_MAX_BYTES: usize = 4096;

/// Targets around the AVX2 block size (32 bytes), the fixture midpoint, the
/// exact total, and one past it (the "fewer than target" path).
fn scan_check_targets(total: usize) -> [usize; 10] {
    [0, 1, 2, 31, 32, 33, 64, total / 2, total, total + 1]
}

/// Differential scan check for one backend against the scalar walks (which
/// the unit tests pin to `char_indices()` semantics).
///
/// - Valid-UTF-8 fixtures up to [`SCAN_CHECK_MAX_BYTES`]: both directions
///   from char boundaries at the start, the middle and the end.
/// - Raw-byte fixtures up to that size: the backward scan (defined for any
///   bytes) from the start, the middle and the end, and the forward scan
///   from offset 0, the middle, the first non-continuation byte at or after
///   the middle, and the end. Arbitrary bytes catch kernels that assume
///   canonical multi-byte sequences; the public forward scan accepts any
///   `start`.
/// - Every larger fixture (the > 128 KiB buffers): one forward walk from 0
///   and one backward walk from the end, to half the total and to one past
///   it, so the long unrolled loops run at every process start.
fn scans_match_reference(backend: Backend, fixtures: &[String], byte_fixtures: &[Vec<u8>]) -> bool {
    for fixture in fixtures.iter().filter(|fixture| fixture.len() <= SCAN_CHECK_MAX_BYTES) {
        let bytes = fixture.as_bytes();
        let total = count_noncontinuation_scalar(bytes);
        let mut middle = bytes.len() / 2;
        while !fixture.is_char_boundary(middle) {
            middle -= 1;
        }
        for position in [0, middle, bytes.len()] {
            for target in scan_check_targets(total) {
                if backend_scan_forward(backend, bytes, position, target)
                    != scan_chars_forward_scalar(bytes, position, target)
                    || backend_scan_backward(backend, bytes, position, target)
                        != scan_chars_backward_scalar(bytes, position, target)
                {
                    return false;
                }
            }
        }
    }
    for bytes in byte_fixtures.iter().filter(|bytes| bytes.len() <= SCAN_CHECK_MAX_BYTES) {
        let total = count_noncontinuation_scalar(bytes);
        let middle = bytes.len() / 2;
        let lead = bytes[middle..]
            .iter()
            .position(|byte| byte & 0xc0 != 0x80)
            .map_or(bytes.len(), |offset| middle + offset);
        for end in [0, middle, bytes.len()] {
            for target in scan_check_targets(total) {
                if backend_scan_backward(backend, bytes, end, target)
                    != scan_chars_backward_scalar(bytes, end, target)
                {
                    return false;
                }
            }
        }
        for start in [0, middle, lead, bytes.len()] {
            for target in scan_check_targets(total) {
                if backend_scan_forward(backend, bytes, start, target)
                    != scan_chars_forward_scalar(bytes, start, target)
                {
                    return false;
                }
            }
        }
    }
    let large = fixtures
        .iter()
        .map(|fixture| fixture.as_bytes())
        .chain(byte_fixtures.iter().map(|bytes| bytes.as_slice()))
        .filter(|bytes| bytes.len() > SCAN_CHECK_MAX_BYTES);
    for bytes in large {
        let total = count_noncontinuation_scalar(bytes);
        for target in [total / 2, total + 1] {
            if backend_scan_forward(backend, bytes, 0, target) != scan_chars_forward_scalar(bytes, 0, target)
                || backend_scan_backward(backend, bytes, bytes.len(), target)
                    != scan_chars_backward_scalar(bytes, bytes.len(), target)
            {
                return false;
            }
        }
    }
    true
}

fn init_dispatch() -> DispatchState {
    if forced_reference() {
        return DispatchState {
            backend: Backend::Reference,
            diagnostic: Some("charcount: forced reference by PRIME_AGENT_COMPACT_RUST_COUNT"),
        };
    }
    let fixtures = self_check_fixtures();
    let reference: Vec<usize> = fixtures.iter().map(|s| count_chars_reference(s)).collect();
    let byte_fixtures = self_check_byte_fixtures();
    let byte_reference: Vec<usize> = byte_fixtures
        .iter()
        .map(|bytes| count_noncontinuation_scalar(bytes))
        .collect();
    for backend in candidate_backends() {
        let mut ok = true;
        for (i, fixture) in fixtures.iter().enumerate() {
            // The expected value always comes from the reference, never from
            // the backend under test, so a mismatch cannot poison the retry.
            match run_backend(backend, fixture.as_bytes()) {
                Some(got) if got == reference[i] => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            for (i, fixture) in byte_fixtures.iter().enumerate() {
                // Byte fixtures may be invalid UTF-8; the reference is the
                // scalar byte filter (the exact pre-SIMD semantics of the
                // fingerprint writer), never the backend under test.
                match run_backend(backend, fixture) {
                    Some(got) if got == byte_reference[i] => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if ok {
            // The scan kernels are part of the backend: a backend whose scans
            // disagree with the scalar walks is rejected as a whole.
            ok = scans_match_reference(backend, &fixtures, &byte_fixtures);
        }
        if ok {
            return DispatchState {
                backend,
                diagnostic: None,
            };
        }
    }
    DispatchState {
        backend: Backend::Reference,
        diagnostic: Some("charcount: all accelerated backends failed self-check; reference active"),
    }
}

fn dispatch_state() -> &'static DispatchState {
    static STATE: OnceLock<DispatchState> = OnceLock::new();
    STATE.get_or_init(init_dispatch)
}

/// The selected backend after the one-time startup self-check.
pub fn backend() -> Backend {
    dispatch_state().backend
}

/// Compact one-line diagnostic (empty when an accelerated backend is active).
/// Content-free: fixture names only, never transcript text. The library does
/// not print it (pi-coding-agent has no logging dependency and stderr writes
/// would corrupt the TUI); tests and tooling surface it via this getter.
pub fn diagnostic() -> &'static str {
    dispatch_state().diagnostic.unwrap_or("")
}

/// Explicitly run the self-check (idempotent; also runs lazily on the first
/// counting call). Returns the active backend.
pub fn self_check() -> Backend {
    dispatch_state().backend
}

/// Dispatched non-continuation-byte count over any byte slice: bit-identical
/// to `bytes.iter().filter(|byte| **byte & 0xc0 != 0x80).count()` for every
/// input, valid UTF-8 or not (each byte is classified independently). For
/// valid UTF-8 this is the Unicode scalar count. This is the streaming
/// primitive for `Write`-based counters — e.g. the Jev compaction
/// fingerprint — whose chunks are raw bytes, not `&str`.
#[inline]
pub fn count_noncontinuation(bytes: &[u8]) -> usize {
    match dispatch_state().backend {
        Backend::Reference => count_noncontinuation_scalar(bytes),
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => count_noncontinuation_sse2(bytes),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => count_noncontinuation_avx2(bytes),
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        Backend::AsmSse2 => asm::count_noncontinuation_sse2(bytes),
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        Backend::AsmAvx2 => count_noncontinuation_asm_avx2(bytes),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => count_noncontinuation_neon(bytes),
    }
}

/// Dispatched Unicode-scalar count of a valid UTF-8 string. Bit-identical to
/// `s.chars().count()` for every valid `&str`: each scalar contributes
/// exactly one non-continuation byte, so the byte kernel is exact.
#[inline]
pub fn count_chars(s: &str) -> usize {
    count_noncontinuation(s.as_bytes())
}

/// Dispatched forward scan over raw bytes: returns the byte offset just past
/// the `target`-th non-continuation byte at/after `start` (the trailing
/// char's continuation bytes are completed) and how many scalars were found
/// (clamped by the end of `bytes`).
///
/// For valid UTF-8 and a `start` on a char boundary this is exactly the end
/// byte offset of `s.chars().skip(start_chars).take(target)`'s last char, so
/// `&s[scan.0..next]` reproduces `chars().skip().take()` slicing. The AVX2
/// backend scans 32-byte blocks with movemask+popcount; other backends use
/// the scalar byte walk.
#[inline]
pub fn scan_chars_forward(bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    backend_scan_forward(dispatch_state().backend, bytes, start, target)
}

/// Dispatched backward scan over raw bytes: returns the byte offset of the
/// `target`-th non-continuation byte counted backwards from `end`
/// (exclusive; clamped to `bytes.len()`) and how many were found. Returns
/// `(end, 0)` for `target == 0` and `(0, found)` when fewer exist.
///
/// For valid UTF-8 and an `end` on a char boundary this is exactly the byte
/// offset of `s[..end].char_indices().rev().nth(target - 1)`, so
/// `&s[scan.0..end]` holds the last `target` chars. Each byte is classified
/// independently, so the result is also defined for invalid UTF-8.
#[inline]
pub fn scan_chars_backward(bytes: &[u8], end: usize, target: usize) -> (usize, usize) {
    let end = end.min(bytes.len());
    backend_scan_backward(dispatch_state().backend, bytes, end, target)
}

/// Byte offset where the first `n` chars of `s` end (`s.len()` when `s` has
/// fewer, 0 for `n == 0`). Bit-identical to
/// `s.char_indices().nth(n - 1).map_or(s.len(), |(i, c)| i + c.len_utf8())`.
#[inline]
pub fn prefix_end(s: &str, n: usize) -> usize {
    scan_chars_forward(s.as_bytes(), 0, n).0
}

/// Byte offset where the last `n` chars of `s` start (0 when `s` has fewer,
/// `s.len()` for `n == 0`). Bit-identical to
/// `s.char_indices().rev().nth(n - 1).map_or(0, |(i, _)| i)`.
#[inline]
pub fn suffix_start(s: &str, n: usize) -> usize {
    scan_chars_backward(s.as_bytes(), s.len(), n).0
}

/// The first `n` chars of `s` as a borrowed slice: the same text as
/// `s.chars().take(n).collect::<String>()`, without per-char pushes.
#[inline]
pub fn take_chars(s: &str, n: usize) -> &str {
    &s[..prefix_end(s, n)]
}

/// The last `n` chars of `s` as a borrowed slice: the same text as
/// `s.chars().skip(s.chars().count().saturating_sub(n)).collect::<String>()`.
#[inline]
pub fn last_chars(s: &str, n: usize) -> &str {
    &s[suffix_start(s, n)..]
}

/// `io::Write` sink that counts the non-continuation bytes streamed into it
/// with the dispatched kernel. Chunk boundaries do not matter: each byte is
/// classified independently.
#[derive(Default)]
struct CharCountWriter {
    chars: usize,
}

impl std::io::Write for CharCountWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.chars += count_noncontinuation(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Unicode-scalar count of `serde_json::to_string(value)` without building
/// the string: `serde_json` writes the same bytes through `to_writer`, and the
/// streamed count is chunk-additive. `None` exactly when `to_string` fails.
pub fn count_json_chars<T: serde::Serialize + ?Sized>(value: &T) -> Option<usize> {
    let mut writer = CharCountWriter::default();
    serde_json::to_writer(&mut writer, value).ok()?;
    Some(writer.chars)
}

// ---------------------------------------------------------------------------
// Tests: differential edge-case corpus ported from the kernel lab
// (lab/kernels, reports/07-kernel-lab). Every accelerated backend and the
// dispatch are checked against `chars().count()` / `char_indices()` on the
// full fixture matrix.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Deterministic corpus generator (xorshift64*, fixed seeds, std-only).
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed | 1 }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FixtureClass {
        Ascii,
        Cjk,
        EmojiCombining,
        MixedCrlf,
        HugeLine,
        SmallConcat,
    }

    const ASCII_POOL: &[char] = &[
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3',
        '4', '5', '6', '7', '8', '9', ' ', '.', ',', ':', ';', '-', '_', '/', '(', ')', '\n', '\t',
        '"', '\\', '\u{0}', '\u{1f}', '\u{7f}',
    ];

    const CJK_POOL: &[char] = &[
        '\u{4e00}', '\u{4e8c}', '\u{4e09}', '\u{56db}', '\u{4e94}', '\u{516d}', '\u{4e03}',
        '\u{516b}', '\u{4e5d}', '\u{5341}', '\u{767e}', '\u{5343}', '\u{4e07}', '。', '，', '：',
        '；', '\n', ' ', '-',
    ];

    const EMOJI_POOL: &[char] = &[
        '\u{1f600}',
        '\u{1f601}',
        '\u{1f602}',
        '\u{1f44d}',
        '\u{1f44e}',
        '\u{1f4bb}',
        '\u{1f4bc}',
        '\u{1f525}',
        '\u{2764}',
        '\u{0301}',
        '\u{0308}',
        '\u{0302}',
        '\u{200d}',
        'a',
        'b',
        'c',
        ' ',
        '\n',
        '"',
        '\\',
    ];

    const MIXED_POOL: &[char] = &[
        'a',
        'B',
        'z',
        '0',
        '9',
        ' ',
        '.',
        '\n',
        '"',
        '\\',
        '\u{e9}',
        '\u{e8}',
        '\u{fc}',
        '\u{4e2d}',
        '\u{6587}',
        '\u{6d4b}',
        '\u{8bd5}',
        '\u{1f600}',
        '\u{1f44d}',
        '\u{0301}',
        '\u{0}',
        '\u{7f}',
        '\u{2603}',
    ];

    fn pool_char(rng: &mut Lcg, pool: &[char]) -> char {
        pool[rng.below(pool.len() as u64) as usize]
    }

    fn generate_text(class: FixtureClass, target_bytes: usize, seed: u64) -> String {
        let mut rng = Lcg::new(seed);
        let mut out = String::with_capacity(target_bytes + 16);
        match class {
            FixtureClass::Ascii => {
                while out.len() < target_bytes {
                    out.push(pool_char(&mut rng, ASCII_POOL));
                }
            }
            FixtureClass::Cjk => {
                while out.len() < target_bytes {
                    out.push(pool_char(&mut rng, CJK_POOL));
                }
            }
            FixtureClass::EmojiCombining => {
                while out.len() < target_bytes {
                    out.push(pool_char(&mut rng, EMOJI_POOL));
                }
            }
            FixtureClass::MixedCrlf => {
                let mut since_nl = 0u64;
                let nl_every = 30 + rng.below(30);
                while out.len() < target_bytes {
                    out.push(pool_char(&mut rng, MIXED_POOL));
                    since_nl += 1;
                    if since_nl >= nl_every {
                        out.push('\r');
                        out.push('\n');
                        since_nl = 0;
                    }
                }
            }
            FixtureClass::HugeLine => {
                while out.len() < target_bytes {
                    let c = pool_char(&mut rng, MIXED_POOL);
                    if c != '\n' {
                        out.push(c);
                    }
                }
            }
            FixtureClass::SmallConcat => {
                while out.len() < target_bytes {
                    let piece_len = 40 + rng.below(260) as usize;
                    for _ in 0..piece_len {
                        out.push(pool_char(&mut rng, MIXED_POOL));
                    }
                    out.push_str("\n\n");
                }
            }
        }
        out
    }

    fn all_classes() -> [FixtureClass; 6] {
        [
            FixtureClass::Ascii,
            FixtureClass::Cjk,
            FixtureClass::EmojiCombining,
            FixtureClass::MixedCrlf,
            FixtureClass::HugeLine,
            FixtureClass::SmallConcat,
        ]
    }

    /// The full differential case matrix: self-check fixtures plus the corpus
    /// classes at two sizes.
    fn count_cases() -> Vec<(String, String)> {
        let mut cases: Vec<(String, String)> = self_check_fixtures()
            .into_iter()
            .enumerate()
            .map(|(i, s)| (format!("self-check-{i}"), s))
            .collect();
        for class in all_classes() {
            for size in [10 * 1024usize, 128 * 1024] {
                cases.push((
                    format!("{class:?}-{size}"),
                    generate_text(class, size, 0xBEEF_0000 + size as u64),
                ));
            }
        }
        cases
    }

    // -----------------------------------------------------------------------
    // Count kernels vs `chars().count()`
    // -----------------------------------------------------------------------

    #[test]
    fn count_kernels_match_chars_count_on_the_full_matrix() {
        for (name, text) in count_cases() {
            let expected = count_chars_reference(&text);
            assert_eq!(
                count_chars(&text),
                expected,
                "dispatch failed on {name} ({} bytes)",
                text.len()
            );
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: SSE2 is baseline on x86-64.
                assert_eq!(
                    unsafe { x86::count_noncontinuation_sse2(text.as_bytes()) },
                    expected,
                    "sse2 failed on {name}"
                );
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 detected at runtime.
                    assert_eq!(
                        unsafe { x86::count_noncontinuation_avx2(text.as_bytes()) },
                        expected,
                        "avx2 failed on {name}"
                    );
                }
            }
            for backend in candidate_backends() {
                assert_eq!(
                    run_backend(backend, text.as_bytes()),
                    Some(expected),
                    "{} backend failed on {name}",
                    backend.name()
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: NEON is baseline on aarch64.
                assert_eq!(
                    unsafe { neon::count_noncontinuation_neon(text.as_bytes()) },
                    expected,
                    "neon failed on {name}"
                );
            }
        }
    }

    #[test]
    fn count_handles_all_scalar_widths_and_extremes() {
        // Every scalar width, boundary scalars, NUL, DEL and max scalar.
        let cases = [
            "",
            "\u{0}",
            "\u{7f}",
            "\u{80}",
            "\u{7ff}",
            "\u{800}",
            "\u{ffff}",
            "\u{10000}",
            "\u{10ffff}",
            "a\u{0}\u{7f}\u{80}\u{7ff}\u{800}\u{ffff}\u{10000}\u{10ffff}",
        ];
        let max_scalar_run = "\u{10ffff}".repeat(1000);
        let cases = cases
            .into_iter()
            .chain(std::iter::once(max_scalar_run.as_str()));
        for case in cases {
            assert_eq!(count_chars(case), case.chars().count(), "{case:?}");
        }
    }

    #[test]
    fn threshold_routing_stays_exact_for_small_inputs() {
        // The dispatch thresholds (16/64) route small slices to the scalar
        // loop; verify the routed result on both sides of each threshold.
        let alphabet: Vec<char> = "a\u{e9}\u{4e2d}\u{1f600}\u{0}".chars().collect();
        for len in 0..=130usize {
            let mut s = String::new();
            for i in 0..len {
                s.push(alphabet[i % alphabet.len()]);
            }
            assert_eq!(count_chars(&s), s.chars().count(), "len {len}");
        }
    }

    // -----------------------------------------------------------------------
    // Forward scan vs `chars().skip().take()` byte semantics
    // -----------------------------------------------------------------------

    /// Reference: byte end of the `target`-th char at/after `start`, computed
    /// with the std char iterator (the pre-optimization implementation shape).
    fn scan_reference(text: &str, start: usize, target: usize) -> (usize, usize) {
        if target == 0 {
            return (start, 0);
        }
        let total = text[start..].chars().count();
        let seen = total.min(target);
        let end = match text[start..].char_indices().nth(seen.saturating_sub(1)) {
            Some((offset, ch)) if seen == target => start + offset + ch.len_utf8(),
            _ => {
                // Ran out of chars: position after the last one.
                text.len()
            }
        };
        (end, seen)
    }

    #[test]
    fn scan_chars_forward_matches_char_indices_on_a_boundary_matrix() {
        let text = "abc\u{4e2d}\u{1f600}def\u{e9}\u{0301}gh\u{0}ijk\u{10ffff}lmn";
        let mut starts = vec![0usize];
        starts.extend(text.char_indices().skip(1).map(|(i, _)| i));
        for start in starts {
            for target in 0..=(text[start..].chars().count() + 2) {
                let expected = scan_reference(text, start, target);
                let got = scan_chars_forward(text.as_bytes(), start, target);
                assert_eq!(
                    got, expected,
                    "start {start} target {target}: got {got:?} want {expected:?}"
                );
            }
        }
    }

    #[test]
    fn scan_chars_forward_scan_starts_on_every_alignment() {
        // The AVX2 scan loads from unaligned `start` positions. `start` is a
        // char boundary (the production invariant), so sweep boundaries at
        // every byte offset 0..=40 in front of a multibyte payload: the loads
        // themselves are unaligned by construction.
        let payload = "\u{4e2d}\u{1f600}abc\u{e9}\u{0301}def\u{10ffff}\u{0}";
        for pad in 0..=40usize {
            let text = format!("{}{}", "x".repeat(pad), payload.repeat(8));
            let starts = std::iter::once(0)
                .chain(text.char_indices().skip(1).map(|(i, _)| i))
                .collect::<Vec<_>>();
            for start in starts {
                for target in [0usize, 1, 7, 31, 32, 33, 64, 65, 10_000] {
                    let expected = scan_reference(&text, start, target);
                    let got = scan_chars_forward(text.as_bytes(), start, target);
                    assert_eq!(got, expected, "pad {pad} start {start} target {target}");
                }
            }
        }
    }

    #[test]
    fn scan_chars_forward_reproduces_the_chunk_loop_shape() {
        // The production use: consecutive chunks sliced by scalar budget must
        // tile the conversation exactly like `chars().skip().take()`.
        for class in all_classes() {
            let text = generate_text(class, 64 * 1024, 0x3333);
            for budget in [1usize, 7, 100, 1000, 65_536, usize::MAX / 4] {
                let mut cursor = 0usize;
                let mut offset = 0usize;
                let total = text.chars().count();
                loop {
                    let (end, seen) = scan_chars_forward(text.as_bytes(), cursor, budget);
                    assert_eq!(
                        seen,
                        budget.min(total - offset),
                        "{} budget {budget}",
                        class as u32
                    );
                    // The slice between the two boundaries must be valid UTF-8
                    // and contain exactly `seen` chars.
                    let chunk = &text[cursor..end];
                    assert_eq!(
                        chunk.chars().count(),
                        seen,
                        "{} budget {budget}",
                        class as u32
                    );
                    cursor = end;
                    offset += seen;
                    if offset >= total {
                        assert_eq!(cursor, text.len(), "{} budget {budget}", class as u32);
                        break;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Backward scan vs `char_indices().rev()`, plus the str helpers built on
    // the two scans and the streamed JSON counter
    // -----------------------------------------------------------------------

    /// Reference: byte start of the `target`-th char counted backwards from
    /// `end`, computed with the std char iterator (the shape of the
    /// `char_boundary_before` walk it replaces).
    fn scan_backward_reference(text: &str, end: usize, target: usize) -> (usize, usize) {
        if target == 0 {
            return (end, 0);
        }
        let head = &text[..end];
        let seen = head.chars().count().min(target);
        let start = match head.char_indices().rev().nth(target - 1) {
            Some((offset, _)) => offset,
            None => 0,
        };
        (start, seen)
    }

    /// Each backend's backward kernel, called directly (not via dispatch).
    fn backward_kernels() -> Vec<(&'static str, fn(&[u8], usize, usize) -> (usize, usize))> {
        let mut kernels: Vec<(&'static str, fn(&[u8], usize, usize) -> (usize, usize))> =
            vec![("scalar", scan_chars_backward_scalar), ("dispatch", scan_chars_backward)];
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", scan_noncontinuation_backward_avx2));
        }
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        {
            if asm::asm_avx2_supported() {
                kernels.push(("asm-avx2", scan_noncontinuation_backward_asm_avx2));
            }
        }
        kernels
    }

    #[test]
    fn scan_chars_backward_matches_char_indices_on_a_boundary_matrix() {
        let text = "abc\u{4e2d}\u{1f600}def\u{e9}\u{0301}gh\u{0}ijk\u{10ffff}lmn";
        let mut ends: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        ends.push(text.len());
        for (name, kernel) in backward_kernels() {
            for &end in &ends {
                for target in 0..=(text[..end].chars().count() + 2) {
                    let expected = scan_backward_reference(text, end, target);
                    let got = kernel(text.as_bytes(), end, target);
                    assert_eq!(got, expected, "{name} end {end} target {target}");
                }
            }
        }
    }

    #[test]
    fn scan_chars_backward_scan_ends_on_every_alignment() {
        // The AVX2 kernel loads `bytes[pos - 32..pos]` from unaligned ends;
        // sweep char-boundary ends behind 0..=40 bytes of padding.
        let payload = "\u{4e2d}\u{1f600}abc\u{e9}\u{0301}def\u{10ffff}\u{0}";
        for (name, kernel) in backward_kernels() {
            for pad in 0..=40usize {
                let text = format!("{}{}", "x".repeat(pad), payload.repeat(8));
                let mut ends: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
                ends.push(text.len());
                for &end in &ends {
                    for target in [0usize, 1, 7, 31, 32, 33, 64, 65, 10_000] {
                        let expected = scan_backward_reference(&text, end, target);
                        let got = kernel(text.as_bytes(), end, target);
                        assert_eq!(got, expected, "{name} pad {pad} end {end} target {target}");
                    }
                }
            }
        }
    }

    #[test]
    fn scan_chars_backward_matches_the_reference_on_the_full_matrix() {
        for (case, text) in count_cases() {
            let total = text.chars().count();
            for (name, kernel) in backward_kernels() {
                for target in [1usize, 2, 31, 32, 33, 500, 2000, total / 2, total, total + 1] {
                    let expected = scan_backward_reference(&text, text.len(), target);
                    assert_eq!(
                        kernel(text.as_bytes(), text.len(), target),
                        expected,
                        "{name} {case} target {target}"
                    );
                }
            }
        }
    }

    #[test]
    fn scan_chars_backward_is_exact_on_arbitrary_bytes() {
        // The backward contract classifies each byte independently, so the
        // accelerated kernels must equal the scalar walk on any bytes and any
        // `end`, not only on char boundaries.
        let mut rng = Lcg::new(0x0BAC_CAFE);
        for size in [0usize, 1, 31, 32, 33, 63, 64, 65, 97, 200] {
            let mut bytes = Vec::with_capacity(size);
            for _ in 0..size {
                bytes.push((rng.next_u64() >> 32) as u8);
            }
            let total = count_noncontinuation_scalar(&bytes);
            for (name, kernel) in backward_kernels() {
                for end in 0..=size {
                    for target in [0usize, 1, 2, 15, 31, 32, 33, 64, total / 2, total, total + 1] {
                        assert_eq!(
                            kernel(&bytes, end, target),
                            scan_chars_backward_scalar(&bytes, end, target),
                            "{name} size {size} end {end} target {target}"
                        );
                    }
                }
            }
        }
        // The public wrapper clamps an out-of-range `end`.
        let bytes = b"abc";
        assert_eq!(scan_chars_backward(bytes, 99, 2), (1, 2));
        assert_eq!(scan_chars_backward(bytes, 99, 0), (3, 0));
    }

    /// Every backend whose scans the host can run, called through
    /// `backend_scan_forward` / `backend_scan_backward` (not the dispatch
    /// state). The assembly backend is included whenever the CPU supports
    /// it, even when `PRIME_AGENT_COMPACT_ASM=0` hides it from dispatch.
    fn scan_backends() -> Vec<Backend> {
        let mut backends = vec![Backend::Reference];
        backends.extend(candidate_backends());
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        {
            if asm::asm_avx2_supported() && !backends.contains(&Backend::AsmAvx2) {
                backends.push(Backend::AsmAvx2);
            }
        }
        backends
    }

    /// Random bytes in one of four shapes: uniform, ASCII-heavy,
    /// continuation-heavy (long invalid runs), and lead-heavy.
    fn random_bytes(rng: &mut Lcg, size: usize, mode: u64) -> Vec<u8> {
        (0..size)
            .map(|_| {
                let value = rng.next_u64();
                let byte = (value >> 32) as u8;
                match mode {
                    0 => byte,
                    1 if value % 8 != 0 => byte & 0x7f,
                    2 if value % 4 != 0 => 0x80 | (byte & 0x3f),
                    3 if value % 3 != 0 => 0xc0 | byte,
                    _ => byte,
                }
            })
            .collect()
    }

    #[test]
    fn scan_chars_forward_is_exact_on_arbitrary_bytes() {
        // The forward contract classifies each byte independently. Every
        // backend must equal the scalar walk on any bytes and on every
        // `start`, including continuation-byte starts (the assembly wrapper
        // routes those to the scalar walk) and starts past the end.
        let mut rng = Lcg::new(0xF0D_5CA9);
        let sizes = [
            0usize, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 97, 127, 128, 129, 160,
            161, 200, 255, 256, 257, 300,
        ];
        for mode in 0..4u64 {
            for size in sizes {
                let bytes = random_bytes(&mut rng, size, mode);
                let total = count_noncontinuation_scalar(&bytes);
                for backend in scan_backends() {
                    for start in 0..=size + 1 {
                        for target in [
                            0usize,
                            1,
                            2,
                            3,
                            7,
                            8,
                            15,
                            16,
                            31,
                            32,
                            33,
                            63,
                            64,
                            65,
                            79,
                            160,
                            total / 2,
                            total,
                            total + 1,
                            usize::MAX,
                        ] {
                            assert_eq!(
                                backend_scan_forward(backend, &bytes, start, target),
                                scan_chars_forward_scalar(&bytes, start, target),
                                "{} mode {mode} size {size} start {start} target {target}",
                                backend.name()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scans_are_exact_above_128_kib() {
        // The long unrolled loops (128..512-byte blocks) and every block
        // boundary of a > 128 KiB buffer, for valid UTF-8 of each corpus
        // class and for arbitrary bytes.
        let mut buffers: Vec<(String, Vec<u8>)> = all_classes()
            .into_iter()
            .map(|class| {
                (
                    format!("{class:?}"),
                    generate_text(class, 300 * 1024 + 17, 0x5CA1_AB1E).into_bytes(),
                )
            })
            .collect();
        let mut rng = Lcg::new(0xB16_B0FF);
        for mode in 0..4u64 {
            buffers.push((format!("random-{mode}"), random_bytes(&mut rng, 300 * 1024 + 29, mode)));
        }
        for (name, bytes) in buffers {
            let total = count_noncontinuation_scalar(&bytes);
            for backend in scan_backends() {
                assert_eq!(run_backend(backend, &bytes), Some(total), "{} count {name}", backend.name());
                let middle = bytes.len() / 2;
                let lead = bytes[middle..]
                    .iter()
                    .position(|byte| byte & 0xc0 != 0x80)
                    .map_or(bytes.len(), |offset| middle + offset);
                for target in [1usize, 1000, 65_537, total / 3, total / 2, total - 1, total, total + 1, usize::MAX] {
                    for start in [0, lead] {
                        assert_eq!(
                            backend_scan_forward(backend, &bytes, start, target),
                            scan_chars_forward_scalar(&bytes, start, target),
                            "{} forward {name} start {start} target {target}",
                            backend.name()
                        );
                    }
                    for end in [bytes.len(), middle, bytes.len() - 1] {
                        assert_eq!(
                            backend_scan_backward(backend, &bytes, end, target),
                            scan_chars_backward_scalar(&bytes, end, target),
                            "{} backward {name} end {end} target {target}",
                            backend.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn asm_switch_accepts_only_0_and_false() {
        assert!(asm_switch_disables("0"));
        assert!(asm_switch_disables("false"));
        assert!(asm_switch_disables("FALSE"));
        assert!(asm_switch_disables("False"));
        assert!(!asm_switch_disables("1"));
        assert!(!asm_switch_disables("true"));
        assert!(!asm_switch_disables("no"));
        assert!(!asm_switch_disables(""));
        assert!(!asm_switch_disables(" 0"));
    }

    #[test]
    fn self_check_prefers_the_assembly_backend_when_supported() {
        #[cfg(all(target_arch = "x86_64", feature = "compaction-asm", not(miri)))]
        {
            if forced_reference() || asm_disabled() {
                return;
            }
            let candidates = candidate_backends();
            assert!(candidates.contains(&Backend::AsmSse2));
            if asm::asm_avx2_supported() {
                assert_eq!(candidates[0], Backend::AsmAvx2);
                assert_eq!(self_check(), Backend::AsmAvx2);
                assert_eq!(diagnostic(), "");
            }
        }
    }

    #[test]
    fn str_helpers_match_the_char_iterator_forms_they_replace() {
        let mut cases = count_cases();
        cases.push(("empty".to_string(), String::new()));
        for (case, text) in cases {
            let total = text.chars().count();
            for n in [0usize, 1, 2, 7, 8, 16, 31, 32, 33, 160, 2000, total / 2, total, total + 1] {
                let take: String = text.chars().take(n).collect();
                assert_eq!(take_chars(&text, n), take, "{case} take {n}");
                let last: String = text.chars().skip(total.saturating_sub(n)).collect();
                assert_eq!(last_chars(&text, n), last, "{case} last {n}");
                let after = if n == 0 {
                    0
                } else {
                    text.char_indices()
                        .nth(n - 1)
                        .map_or(text.len(), |(byte, ch)| byte + ch.len_utf8())
                };
                assert_eq!(prefix_end(&text, n), after, "{case} prefix_end {n}");
                let before = if n == 0 {
                    text.len()
                } else {
                    text.char_indices().rev().nth(n - 1).map_or(0, |(byte, _)| byte)
                };
                assert_eq!(suffix_start(&text, n), before, "{case} suffix_start {n}");
            }
        }
    }

    #[test]
    fn count_json_chars_matches_to_string_then_count() {
        let values = vec![
            serde_json::json!(null),
            serde_json::json!({}),
            serde_json::json!({"path": "src/\u{4e2d}\u{6587}.rs", "n": 3, "ok": true}),
            serde_json::json!({"text": "quote \" backslash \\ nl \n tab \t nul \u{0} del \u{7f}"}),
            serde_json::json!({"emoji": "\u{1f600}\u{1f44d}\u{200d}", "list": [1.5, -2, "\u{e9}\u{0301}"]}),
            serde_json::json!({"big": generate_text(FixtureClass::MixedCrlf, 64 * 1024, 0x15_0A)}),
        ];
        for value in &values {
            let expected = serde_json::to_string(value).ok().map(|json| json.chars().count());
            assert_eq!(count_json_chars(value), expected, "{value}");
        }
        // A map with non-string keys fails `to_string`; the counter reports
        // `None` exactly then, never a partial count.
        let mut bad = std::collections::BTreeMap::new();
        bad.insert((1u8, 2u8), "x");
        assert!(serde_json::to_string(&bad).is_err());
        assert_eq!(count_json_chars(&bad), None);
    }

    #[test]
    fn every_candidate_backend_passes_the_scan_self_check() {
        let fixtures = self_check_fixtures();
        let byte_fixtures = self_check_byte_fixtures();
        assert!(scans_match_reference(Backend::Reference, &fixtures, &byte_fixtures));
        for backend in candidate_backends() {
            assert!(
                scans_match_reference(backend, &fixtures, &byte_fixtures),
                "{} scan kernels disagree with the scalar walks",
                backend.name()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Dispatch, self-check, force switch
    // -----------------------------------------------------------------------

    #[test]
    fn self_check_activates_an_accelerated_backend_with_no_diagnostic() {
        let backend = self_check();
        // With PRIME_AGENT_COMPACT_RUST_COUNT=1|true the forced reference is
        // the expected state, and its diagnostic says so.
        if forced_reference() {
            assert_eq!(backend, Backend::Reference);
            assert!(diagnostic().contains("forced reference"));
            return;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert_ne!(
            backend,
            Backend::Reference,
            "no accelerated backend passed the self-check"
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(backend, Backend::Reference);
        // A healthy backend selection carries no diagnostic.
        assert_eq!(diagnostic(), "");
    }

    #[test]
    fn force_switch_accepts_only_1_and_true() {
        let parse = |value: &str| value == "1" || value.eq_ignore_ascii_case("true");
        assert!(parse("1"));
        assert!(parse("true"));
        assert!(parse("TRUE"));
        assert!(parse("True"));
        assert!(!parse("0"));
        assert!(!parse("yes"));
        assert!(!parse(""));
        assert!(!parse("2"));
        assert!(!parse(" 1"));
    }

    #[test]
    fn self_check_corpus_exceeds_the_neon_fold_interval() {
        // The aarch64 kernel folds its u16 lanes every 4096 blocks x 16 B =
        // 65 536 B. The per-process self-check must see a fixture at least
        // that large so the fold path is differentially validated on every
        // process start (the corpus carries a >= 128 KiB buffer).
        let max_len = self_check_fixtures()
            .iter()
            .map(|fixture| fixture.len())
            .max()
            .unwrap();
        assert!(
            max_len >= 65_536,
            "largest self-check fixture is {max_len} bytes; the NEON fold path would never run"
        );
    }

    #[test]
    fn count_is_deterministic_across_repeated_calls() {
        let text = generate_text(FixtureClass::MixedCrlf, 32 * 1024, 0xDEAD);
        let first = count_chars(&text);
        for _ in 0..3 {
            assert_eq!(count_chars(&text), first);
        }
        assert_eq!(first, count_chars_reference(&text));
    }

    // -----------------------------------------------------------------------
    // Raw-byte entry point (`count_noncontinuation`), used by the Jev
    // fingerprint writer: byte-exact vs the scalar filter on valid UTF-8,
    // arbitrary bytes and streamed chunkings.
    // -----------------------------------------------------------------------

    #[test]
    fn count_noncontinuation_matches_chars_count_on_valid_utf8() {
        for (name, text) in count_cases() {
            let expected = count_chars_reference(&text);
            assert_eq!(
                count_noncontinuation(text.as_bytes()),
                expected,
                "dispatch failed on {name} ({} bytes)",
                text.len()
            );
            assert_eq!(
                count_chars(&text),
                expected,
                "count_chars changed on {name}"
            );
        }
    }

    #[test]
    fn count_noncontinuation_is_exact_on_arbitrary_bytes() {
        let mut cases: Vec<(String, Vec<u8>)> = self_check_byte_fixtures()
            .into_iter()
            .enumerate()
            .map(|(i, bytes)| (format!("self-check-bytes-{i}"), bytes))
            .collect();
        let mut rng = Lcg::new(0xC0FF_EE01);
        for size in [
            1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 4096, 65_535, 65_536, 65_537, 131_072,
        ] {
            let mut bytes = Vec::with_capacity(size);
            for _ in 0..size {
                bytes.push((rng.next_u64() >> 32) as u8);
            }
            cases.push((format!("random-{size}"), bytes));
        }
        for (name, bytes) in cases {
            let expected = count_noncontinuation_scalar(&bytes);
            assert_eq!(
                count_noncontinuation(&bytes),
                expected,
                "dispatch failed on {name}"
            );
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: SSE2 is baseline on x86-64.
                assert_eq!(
                    unsafe { x86::count_noncontinuation_sse2(&bytes) },
                    expected,
                    "sse2 failed on {name}"
                );
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 detected at runtime.
                    assert_eq!(
                        unsafe { x86::count_noncontinuation_avx2(&bytes) },
                        expected,
                        "avx2 failed on {name}"
                    );
                }
            }
            for backend in candidate_backends() {
                assert_eq!(
                    run_backend(backend, &bytes),
                    Some(expected),
                    "{} backend failed on {name}",
                    backend.name()
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: NEON is baseline on aarch64.
                assert_eq!(
                    unsafe { neon::count_noncontinuation_neon(&bytes) },
                    expected,
                    "neon failed on {name}"
                );
            }
        }
    }

    #[test]
    fn count_noncontinuation_is_chunk_additive_like_a_write_stream() {
        // `Write`-based counters (the Jev fingerprint) sum one count per
        // chunk; the byte classifier is position-independent, so any
        // chunking must reproduce the whole-buffer count.
        let text = generate_text(FixtureClass::MixedCrlf, 96 * 1024, 0x5CA1);
        let bytes = text.as_bytes();
        let expected = count_noncontinuation_scalar(bytes);
        for chunk in [1usize, 7, 16, 31, 32, 33, 64, 100, 4096] {
            let mut got = 0usize;
            for piece in bytes.chunks(chunk) {
                got += count_noncontinuation(piece);
            }
            assert_eq!(got, expected, "chunk {chunk}");
        }
        // Irregular chunk sizes.
        let mut rng = Lcg::new(0xABCD_1234);
        let mut got = 0usize;
        let mut rest = bytes;
        while !rest.is_empty() {
            let take = (1 + rng.below(97) as usize).min(rest.len());
            let (head, tail) = rest.split_at(take);
            got += count_noncontinuation(head);
            rest = tail;
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn self_check_byte_corpus_exceeds_the_neon_fold_interval() {
        let max_len = self_check_byte_fixtures()
            .iter()
            .map(|fixture| fixture.len())
            .max()
            .unwrap();
        assert!(
            max_len >= 65_536,
            "largest byte self-check fixture is {max_len} bytes; the NEON fold path would never run"
        );
    }

    #[test]
    fn self_check_byte_corpus_covers_every_byte_value() {
        let mut seen = [false; 256];
        for fixture in self_check_byte_fixtures() {
            for byte in fixture {
                seen[byte as usize] = true;
            }
        }
        assert!(
            seen.iter().all(|&value| value),
            "the byte self-check corpus must classify every byte value"
        );
    }
}
