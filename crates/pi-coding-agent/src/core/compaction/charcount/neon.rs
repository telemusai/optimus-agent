//! aarch64 NEON kernel for `core::compaction::charcount`.
//!
//! NEON is baseline on aarch64, so this kernel needs no runtime detection.
//! It implements the same count contract as `charcount/x86.rs`; the scans use
//! the portable scalar walk on this architecture.

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
