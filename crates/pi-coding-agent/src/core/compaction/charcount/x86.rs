//! Rust-intrinsic x86-64 SIMD kernels for `core::compaction::charcount`.
//!
//! Every hot character walk in compaction (token estimates, cut-point costs,
//! summary truncation and chunking, typed/task state limits, and the Jev
//! fingerprint) reaches the CPU through four kernels selected by the
//! dispatch in `charcount.rs`. The preferred x86-64 implementation of those
//! kernels is the hand-written assembly in `charcount/x86_64.s` (wrapped by
//! `charcount/asm.rs`). This file holds the Rust-intrinsic versions, used
//! when the `compaction-asm` feature is off, under Miri, when
//! `PRIME_AGENT_COMPACT_ASM=0`, on AVX2 hosts without BMI1/BMI2/POPCNT/LZCNT,
//! or when an assembly kernel fails the startup self-check. Both
//! implementations follow the same contract.
//!
//! Kernel contract (shared by every backend that implements these entry
//! points, and checked by the startup self-check and the unit tests):
//!
//! | Kernel | Inputs | Result |
//! |---|---|---|
//! | `count_noncontinuation_sse2` / `_avx2` | any byte slice | number of bytes with `(b & 0xC0) != 0x80` |
//! | `scan_noncontinuation_avx2` | bytes, `start`, `target` | `(end, seen)`: `end` is just past the `target`-th non-continuation byte at/after `start`, extended over its trailing continuation bytes; `seen = min(target, available)`; `(start, 0)` for `target == 0` |
//! | `scan_noncontinuation_backward_avx2` | bytes, `end <= len`, `target` | `(start, seen)`: `start` is the offset of the `target`-th non-continuation byte counted backwards from `end` (exclusive); `(0, available)` when fewer exist; `(end, 0)` for `target == 0` |
//!
//! Each byte is classified independently, so results are defined for any
//! bytes (valid UTF-8 or not). For valid UTF-8 the count is the Unicode
//! scalar count and the scan offsets are char boundaries.

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

/// AVX2 backward scan: byte offset of the `target`-th non-continuation byte
/// counted backwards from `end` (exclusive), plus how many were found. When
/// `bytes[..end]` holds fewer than `target`, returns `(0, found)`.
///
/// # Safety
///
/// - Feature requirement: caller must have verified AVX2 at runtime; the
///   dispatch only reaches this kernel after detection and self-check.
/// - Input: `end <= bytes.len()` (the dispatch wrapper clamps it). `end`
///   should be a char boundary for UTF-8-correct results.
/// - Bounds: the vector loop runs while `pos >= 32` and loads
///   `bytes[pos - 32..pos]`; because `pos <= end <= bytes.len()`, each
///   `_mm256_loadu_si256` reads exactly 32 in-bounds bytes. When the target
///   falls inside a block, its bytes are walked from the block's movemask
///   (`bit < 32`, so `block + bit < pos`); the scalar head only indexes
///   checked positions below `pos`. No out-of-bounds reads are possible.
/// - Registers: only ymm registers are touched.
#[target_feature(enable = "avx2")]
pub unsafe fn scan_noncontinuation_backward_avx2(
    bytes: &[u8],
    end: usize,
    target: usize,
) -> (usize, usize) {
    if target == 0 {
        return (end, 0);
    }
    let mask_c0 = _mm256_set1_epi8(0xc0u8 as i8);
    let mask_80 = _mm256_set1_epi8(0x80u8 as i8);
    let mut seen = 0usize;
    let mut pos = end;
    while pos >= 32 {
        let block = pos - 32;
        let v = _mm256_loadu_si256(bytes[block..pos].as_ptr().cast());
        let cont = _mm256_cmpeq_epi8(_mm256_and_si256(v, mask_c0), mask_80);
        let non_cont_mask =
            _mm256_movemask_epi8(_mm256_xor_si256(cont, _mm256_set1_epi8(-1))) as u32;
        let block_count = non_cont_mask.count_ones() as usize;
        if seen + block_count >= target {
            // `seen < target <= seen + block_count`, so the mask still holds
            // the wanted bit; walk set bits from the highest address down.
            let mut mask = non_cont_mask;
            while mask != 0 {
                let bit = 31 - mask.leading_zeros() as usize;
                mask &= !(1u32 << bit);
                seen += 1;
                if seen == target {
                    return (block + bit, seen);
                }
            }
        }
        seen += block_count;
        pos = block;
    }
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
