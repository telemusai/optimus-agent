//! Hand-written x86-64 assembly backend for `core::compaction::charcount`.
//!
//! `x86_64.s` holds the four compaction character kernels (SSE2 count, AVX2
//! count, AVX2 forward scan, AVX2 backward scan) as one portable
//! Intel-syntax file. It is assembled by LLVM's integrated assembler through
//! `global_asm!` for ELF (Linux), Mach-O (macOS) and COFF (Windows); the
//! file uses no object-format directives, and each function carries both a
//! `name` and a `_name` global so the Mach-O C-symbol prefix resolves.
//!
//! ABI: every kernel is `extern "sysv64"` on every OS, so one body serves all
//! x86-64 targets. The kernels are leaf functions: no stack use, no calls,
//! callee-saved registers untouched, `vzeroupper` before returning after any
//! ymm use. On Windows this is what makes them legal without unwind data.
//! Pairs return as a `#[repr(C)]` two-word struct in RAX:RDX.
//!
//! Memory: each kernel reads only `[ptr, ptr + len)` (`[ptr, ptr + end)` for
//! the backward scan), never past it, not even within the same page.
//!
//! Instruction sets: the `_sse2` kernel uses SSE2 and base integer
//! instructions only (the x86-64 baseline). The `_avx2` kernels also use
//! BMI1, BMI2, POPCNT and LZCNT, so the dispatch selects them only after
//! detecting all five features (see `asm_avx2_supported`), and only after
//! the startup self-check passes. No AVX-512 and no PDEP/PEXT.
//!
//! This module is the only owner of the `global_asm!` (the symbols are
//! `pi_cc_v1_*`); no other crate may assemble the same file.

use std::arch::global_asm;

// `options(raw)`: the file is plain assembly; `{` and `}` are not format braces.
global_asm!(include_str!("x86_64.s"), options(raw));

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScanResult {
    pos: usize,
    seen: usize,
}

extern "sysv64" {
    fn pi_cc_v1_count_noncont_sse2(ptr: *const u8, len: usize) -> usize;
    fn pi_cc_v1_count_noncont_avx2(ptr: *const u8, len: usize) -> usize;
    fn pi_cc_v1_scan_fwd_avx2(ptr: *const u8, len: usize, start: usize, target: usize) -> ScanResult;
    fn pi_cc_v1_scan_bwd_avx2(ptr: *const u8, end: usize, target: usize) -> ScanResult;
}

/// True when the CPU has every instruction set the `_avx2` kernels use.
pub fn asm_avx2_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("popcnt")
        && std::arch::is_x86_feature_detected!("lzcnt")
}

/// Assembly SSE2 count: the number of bytes with `(b & 0xC0) != 0x80`.
///
/// Safe for any slice: SSE2 is part of the x86-64 baseline and the kernel
/// reads only `bytes`. The empty slice returns without a call.
#[inline]
pub fn count_noncontinuation_sse2(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    // SAFETY: SSE2 is baseline on x86-64; the kernel reads only
    // `[ptr, ptr + len)` of this live slice and follows the sysv64 ABI.
    unsafe { pi_cc_v1_count_noncont_sse2(bytes.as_ptr(), bytes.len()) }
}

/// Assembly AVX2 count: the number of bytes with `(b & 0xC0) != 0x80`.
///
/// # Safety
///
/// The caller must have confirmed [`asm_avx2_supported`].
#[inline]
pub unsafe fn count_noncontinuation_avx2(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    // SAFETY: the caller guarantees AVX2/BMI1/BMI2/POPCNT/LZCNT; the kernel
    // reads only `[ptr, ptr + len)` of this live slice.
    unsafe { pi_cc_v1_count_noncont_avx2(bytes.as_ptr(), bytes.len()) }
}

/// Assembly AVX2 forward scan (same contract as the scalar forward walk).
///
/// # Safety
///
/// The caller must have confirmed [`asm_avx2_supported`], and `start` must
/// be `bytes.len()` or the offset of a non-continuation byte (the dispatch
/// in `backend_scan_forward` routes every other `start` to the scalar walk).
#[inline]
pub unsafe fn scan_noncontinuation_avx2(bytes: &[u8], start: usize, target: usize) -> (usize, usize) {
    debug_assert!(start == bytes.len() || bytes.get(start).is_some_and(|byte| byte & 0xc0 != 0x80));
    // SAFETY: the caller guarantees the ISA and the `start` precondition
    // (so `start <= len`); the kernel reads only `[ptr, ptr + len)`.
    let result = unsafe { pi_cc_v1_scan_fwd_avx2(bytes.as_ptr(), bytes.len(), start, target) };
    (result.pos, result.seen)
}

/// Assembly AVX2 backward scan (same contract as the scalar backward walk;
/// defined for any bytes). `end` must already be clamped to `bytes.len()`.
///
/// # Safety
///
/// The caller must have confirmed [`asm_avx2_supported`].
#[inline]
pub unsafe fn scan_noncontinuation_backward_avx2(bytes: &[u8], end: usize, target: usize) -> (usize, usize) {
    let end = end.min(bytes.len());
    // SAFETY: the caller guarantees the ISA; `end <= len`, and the kernel
    // reads only `[ptr, ptr + end)`.
    let result = unsafe { pi_cc_v1_scan_bwd_avx2(bytes.as_ptr(), end, target) };
    (result.pos, result.seen)
}
