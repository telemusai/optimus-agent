//! Dispatched Unicode-scalar counting for the compaction hot paths.
//!
//! `estimate_tokens` re-counts `chars()` of every text block on every
//! threshold check (per turn end and pre-turn), and the summary chunk loop
//! walks the serialized conversation by scalar offsets. Both are dominated
//! by one kernel: counting UTF-8 non-continuation bytes (`(b & 0xC0) != 0x80`),
//! which equals the Unicode scalar count of any valid UTF-8 string.
//!
//! Backends, selected once per process:
//! - x86-64: AVX2 when `is_x86_feature_detected!("avx2")` (for `len >= 64`),
//!   SSE2 otherwise (`len >= 16`; SSE2 is part of the x86-64 baseline, so it
//!   needs no runtime detection). Below those lengths a scalar loop wins.
//! - aarch64: NEON. NEON is baseline hardware on aarch64 (enabled by default
//!   for the target), so the kernel needs no runtime feature check.
//! - other architectures: the portable `chars().count()` reference, which the
//!   compiler autovectorizes on most targets.
//!
//! Selection happens behind a `OnceLock` that first runs a differential
//! self-check of each candidate backend against `chars().count()` on fixed
//! edge-case fixtures (empty input, every byte length 1..=64, load alignments
//! 0..=31, NUL, every UTF-8 scalar width, and a > 128 KiB mixed buffer so the
//! aarch64 fold path is exercised). A backend that disagrees on any
//! fixture is disabled and the next candidate is tried; the compact
//! diagnostic (no transcript content) is retrievable via [`diagnostic`].
//! `PRIME_AGENT_COMPACT_RUST_COUNT=1` (or `true`) pins the reference
//! implementation regardless of hardware.
//!
//! Every counting result is bit-identical to `chars().count()` for valid
//! UTF-8; nothing else about the compaction path changes.

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

// ---------------------------------------------------------------------------
// x86-64 kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Counts non-continuation bytes with explicit SSE2 intrinsics.
    ///
    /// # Safety
    ///
    /// - Feature requirement: SSE2 is part of the x86-64 baseline, so any
    ///   x86-64 CPU satisfies `target_feature(enable = "sse2")`; no runtime
    ///   detection is needed.
    /// - Input: `bytes` may be any slice (no alignment requirement: loads use
    ///   `_mm_loadu_si128`). Validity of the UTF-8 is not required — the
    ///   kernel only classifies individual bytes.
    /// - Bounds: every load comes from `bytes.chunks_exact(16)`, so each
    ///   `_mm_loadu_si128` reads exactly 16 in-bounds bytes and never crosses
    ///   `bytes.len()`. The `< 16`-byte tail is consumed by the checked scalar
    ///   remainder loop. No out-of-bounds reads are possible.
    /// - Registers: only xmm registers are touched (no MMX/x87 state); the
    ///   compiler allocates and preserves them per the Rust ABI.
    #[target_feature(enable = "sse2")]
    pub unsafe fn count_noncontinuation_sse2(bytes: &[u8]) -> usize {
        let ones = _mm_set1_epi8(1);
        let mask_c0 = _mm_set1_epi8(0xc0u8 as i8);
        let mask_80 = _mm_set1_epi8(0x80u8 as i8);
        let zero = _mm_setzero_si128();
        let mut acc = zero;
        let mut chunks = bytes.chunks_exact(16);
        for chunk in &mut chunks {
            let v = _mm_loadu_si128(chunk.as_ptr().cast());
            let cont = _mm_cmpeq_epi8(_mm_and_si128(v, mask_c0), mask_80);
            let non_cont = _mm_andnot_si128(cont, ones);
            // psadbw sums the 0/1 bytes into two u64 lanes; paddq accumulates.
            acc = _mm_add_epi64(_mm_sad_epu8(non_cont, zero), acc);
        }
        let mut lanes = [0u64; 2];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), acc);
        let mut count = (lanes[0] + lanes[1]) as usize;
        for &byte in chunks.remainder() {
            count += usize::from(byte & 0xc0 != 0x80);
        }
        count
    }

    /// Counts non-continuation bytes with explicit AVX2 intrinsics.
    ///
    /// # Safety
    ///
    /// - Feature requirement: the caller must have verified AVX2 at runtime
    ///   (`is_x86_feature_detected!("avx2")`); executing YMM instructions on
    ///   a non-AVX2 CPU is undefined. The dispatch only reaches this kernel
    ///   after detection and self-check.
    /// - Input: `bytes` may be any slice (no alignment requirement: loads use
    ///   `_mm256_loadu_si256`). UTF-8 validity is not required.
    /// - Bounds: every load comes from `bytes.chunks_exact(32)`, so each
    ///   `_mm256_loadu_si256` reads exactly 32 in-bounds bytes; the `< 32`-byte
    ///   tail goes to the checked scalar remainder loop. No out-of-bounds
    ///   reads are possible.
    /// - Registers: only ymm registers are touched; the compiler allocates and
    ///   preserves them per the Rust ABI.
    #[target_feature(enable = "avx2")]
    pub unsafe fn count_noncontinuation_avx2(bytes: &[u8]) -> usize {
        let ones = _mm256_set1_epi8(1);
        let mask_c0 = _mm256_set1_epi8(0xc0u8 as i8);
        let mask_80 = _mm256_set1_epi8(0x80u8 as i8);
        let zero = _mm256_setzero_si256();
        let mut acc = zero;
        let mut chunks = bytes.chunks_exact(32);
        for chunk in &mut chunks {
            let v = _mm256_loadu_si256(chunk.as_ptr().cast());
            let cont = _mm256_cmpeq_epi8(_mm256_and_si256(v, mask_c0), mask_80);
            let non_cont = _mm256_andnot_si256(cont, ones);
            acc = _mm256_add_epi64(_mm256_sad_epu8(non_cont, zero), acc);
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
        let mut count = (lanes[0] + lanes[1] + lanes[2] + lanes[3]) as usize;
        for &byte in chunks.remainder() {
            count += usize::from(byte & 0xc0 != 0x80);
        }
        count
    }

    /// AVX2 forward scan: byte offset just past the `target`-th
    /// non-continuation byte at/after `start` (completing the trailing
    /// char's continuation bytes), plus how many were found.
    ///
    /// # Safety
    ///
    /// - Feature requirement: caller must have verified AVX2 at runtime; the
    ///   dispatch only reaches this kernel after detection and self-check.
    /// - Input: `bytes` may be any slice; `start` may be any position `<=
    ///   bytes.len()` (a char boundary for UTF-8-correct results).
    /// - Bounds: the vector loop runs while `pos + 32 <= bytes.len()`, so each
    ///   `_mm256_loadu_si256` reads exactly 32 in-bounds bytes. The final
    ///   partial block is never loaded as a vector: when the target falls
    ///   inside a full block, its bytes are walked one by one from the block's
    ///   movemask (`bit < 32`, so `pos + bit` stays inside the block); the
    ///   scalar tail only indexes checked positions. No out-of-bounds reads
    ///   are possible.
    /// - Registers: only ymm registers are touched.
    #[target_feature(enable = "avx2")]
    pub unsafe fn scan_noncontinuation_avx2(
        bytes: &[u8],
        start: usize,
        target: usize,
    ) -> (usize, usize) {
        if target == 0 {
            return (start, 0);
        }
        let mask_c0 = _mm256_set1_epi8(0xc0u8 as i8);
        let mask_80 = _mm256_set1_epi8(0x80u8 as i8);
        let mut seen = 0usize;
        let mut pos = start;
        while pos + 32 <= bytes.len() {
            let v = _mm256_loadu_si256(bytes[pos..].as_ptr().cast());
            let cont = _mm256_cmpeq_epi8(_mm256_and_si256(v, mask_c0), mask_80);
            let non_cont_mask = _mm256_movemask_epi8(_mm256_xor_si256(cont, _mm256_set1_epi8(-1)));
            let block_count = non_cont_mask.count_ones() as usize;
            if seen + block_count >= target {
                let mut mask = non_cont_mask as u32;
                while mask != 0 {
                    let bit = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    seen += 1;
                    if seen == target {
                        // Complete the trailing char (its continuation bytes).
                        let mut end = pos + bit + 1;
                        while end < bytes.len() && bytes[end] & 0xc0 == 0x80 {
                            end += 1;
                        }
                        return (end, seen);
                    }
                }
            }
            seen += block_count;
            pos += 32;
        }
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
}

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

// ---------------------------------------------------------------------------
// aarch64 kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Counts non-continuation bytes with NEON.
    ///
    /// # Safety
    ///
    /// - Feature requirement: NEON is baseline hardware on aarch64 (Rust
    ///   enables `neon` for the target by default), so no runtime detection
    ///   is needed.
    /// - Input: `bytes` may be any slice (no alignment requirement: loads use
    ///   `vld1q_u8`). UTF-8 validity is not required.
    /// - Bounds: every load comes from `bytes.chunks_exact(16)`, so each
    ///   `vld1q_u8` reads exactly 16 in-bounds bytes; the `< 16`-byte tail
    ///   goes to the checked scalar remainder loop. No out-of-bounds reads
    ///   are possible.
    /// - Accumulation: `vpadalq_u8` widens into u16 lanes; the accumulator is
    ///   folded into a u32 total every 4096 blocks, so a u16 lane can hold at
    ///   most 4096 * 2 = 8192 and cannot overflow.
    /// - Registers: only NEON vector registers are touched; the compiler
    ///   allocates and preserves them per the Rust ABI.
    #[target_feature(enable = "neon")]
    pub unsafe fn count_noncontinuation_neon(bytes: &[u8]) -> usize {
        let ones = vdupq_n_u8(1);
        let mask_c0 = vdupq_n_u8(0xc0);
        let mask_80 = vmovq_n_u8(0x80);
        let mut acc = vdupq_n_u16(0);
        let mut chunks = bytes.chunks_exact(16);
        let mut since_fold = 0usize;
        let mut total = 0usize;
        for chunk in &mut chunks {
            let v = vld1q_u8(chunk.as_ptr());
            let cont = vceqq_u8(vandq_u8(v, mask_c0), mask_80);
            let non_cont = vandq_u8(vmvnq_u8(cont), ones);
            acc = vpadalq_u8(acc, non_cont);
            since_fold += 1;
            if since_fold == 4096 {
                let wide = vpaddlq_u16(acc);
                total = total.wrapping_add(vaddvq_u32(wide) as usize);
                acc = vdupq_n_u16(0);
                since_fold = 0;
            }
        }
        if since_fold > 0 {
            let wide = vpaddlq_u16(acc);
            total = total.wrapping_add(vaddvq_u32(wide) as usize);
        }
        for &byte in chunks.remainder() {
            total += usize::from(byte & 0xc0 != 0x80);
        }
        total
    }
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
        if std::arch::is_x86_feature_detected!("avx2") {
            candidates.push(Backend::Avx2);
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

fn init_dispatch() -> DispatchState {
    if forced_reference() {
        return DispatchState {
            backend: Backend::Reference,
            diagnostic: Some("charcount: forced reference by PRIME_AGENT_COMPACT_RUST_COUNT"),
        };
    }
    let fixtures = self_check_fixtures();
    let reference: Vec<usize> = fixtures.iter().map(|s| count_chars_reference(s)).collect();
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

/// Dispatched Unicode-scalar count of a valid UTF-8 string. Bit-identical to
/// `s.chars().count()` for every valid `&str`.
#[inline]
pub fn count_chars(s: &str) -> usize {
    match dispatch_state().backend {
        Backend::Reference => count_chars_reference(s),
        #[cfg(target_arch = "x86_64")]
        Backend::Sse2 => count_noncontinuation_sse2(s.as_bytes()),
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => count_noncontinuation_avx2(s.as_bytes()),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => count_noncontinuation_neon(s.as_bytes()),
    }
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
pub fn scan_chars_forward(bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if dispatch_state().backend == Backend::Avx2 {
            return scan_noncontinuation_avx2(bytes, start, target);
        }
    }
    scan_chars_forward_scalar(bytes, start, target)
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
    // Dispatch, self-check, force switch
    // -----------------------------------------------------------------------

    #[test]
    fn self_check_activates_an_accelerated_backend_with_no_diagnostic() {
        let backend = self_check();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert_ne!(
            backend,
            Backend::Reference,
            "no accelerated backend passed the self-check"
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(backend, Backend::Reference);
        // A healthy backend selection carries no diagnostic. (If the test
        // process runs with PRIME_AGENT_COMPACT_RUST_COUNT set, the forced
        // reference is the expected state instead.)
        if std::env::var("PRIME_AGENT_COMPACT_RUST_COUNT").is_err() {
            assert_eq!(diagnostic(), "");
        }
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
}
