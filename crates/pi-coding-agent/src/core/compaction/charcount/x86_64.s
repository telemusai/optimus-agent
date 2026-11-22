# Compaction character kernels, x86-64 (Intel syntax, LLVM integrated assembler).
#
# CONTRACT (shared by every function in this file)
#   Calling convention: System V AMD64 on EVERY OS. Rust declares these as
#   `extern "sysv64"`, also on Windows, so one body serves Linux, macOS and
#   Windows. Args: rdi, rsi, rdx, rcx. Scalar return: rax. Pair return
#   (#[repr(C)] { pos, seen }): rax = pos, rdx = seen.
#   Registers: may clobber rax, rcx, rdx, rsi, rdi, r8-r11 and all xmm/ymm.
#   Must preserve rbx, rbp, rsp, r12-r15. No stack use at all (no push/pop,
#   no red zone): the Windows build has no unwind info for these leaves.
#   If any ymm register is touched, execute `vzeroupper` before `ret`.
#   DF is clear on entry and must stay clear.
#   Memory: read ONLY bytes inside [ptr, ptr + len) (scan_bwd: [ptr, ptr + end)).
#   No over-reads, not even inside the same page. Overlapping loads that stay
#   inside the buffer are fine. When len == 0 the pointer may be dangling:
#   do not dereference it.
#   Instruction sets: *_sse2 may use SSE2 only (x86-64 baseline; plus nothing
#   else). *_avx2 may use AVX, AVX2, BMI1, BMI2, POPCNT, LZCNT (all present
#   on every AVX2 CPU the dispatch accepts). No AVX-512.
#
# PORTABILITY (the same text is assembled for ELF, Mach-O and COFF)
#   Use only: .text, .p2align, .globl, labels, instructions.
#   No .type/.size/.section/.def/.cfi directives.
#   Each function has two global names at the same address: `name` (ELF, COFF)
#   and `_name` (Mach-O adds a leading underscore to C symbols).
#   Labels inside functions: numeric local labels only (`2:` + `2b`/`2f`), and
#   every numeric label must contain a digit 2-9 (Intel syntax would read
#   `1b`/`10b` as binary literals). Named labels would split Mach-O atoms.
#   Constants: build them in registers (e.g. mov eax, imm / vmovd / vpbroadcastb);
#   no .rodata/.section literal pools.
#
# FUNCTIONS
#   usize pi_cc_v1_count_noncont_sse2(const u8 *ptr, usize len)
#   usize pi_cc_v1_count_noncont_avx2(const u8 *ptr, usize len)
#       Number of bytes b in [ptr, ptr+len) with (b & 0xC0) != 0x80.
#       Defined for ANY bytes (each byte classified independently).
#   {pos, seen} pi_cc_v1_scan_fwd_avx2(const u8 *ptr, usize len, usize start, usize target)
#       Precondition: start <= len, and (start == len or ptr[start] is not a
#       continuation byte). Exactly the scalar walk:
#         pos = start; seen = 0
#         while pos < len && seen < target: if !cont(ptr[pos]) seen++; pos++
#         while pos < len && cont(ptr[pos]): pos++
#         return (pos, seen)
#   {pos, seen} pi_cc_v1_scan_bwd_avx2(const u8 *ptr, usize end, usize target)
#       Defined for ANY bytes. Exactly the scalar walk:
#         if target == 0: return (end, 0)
#         pos = end; seen = 0
#         while pos > 0: pos--; if !cont(ptr[pos]) { seen++; if seen == target return (pos, seen) }
#         return (0, seen)
#   where cont(b) := (b & 0xC0) == 0x80.
#
# KERNEL DESIGNS (each is the merge of competing round-1 candidates; every
# version passed a differential suite, guard-page over-read tests, an ABI
# check and a mutant-catching gate suite in the compaction kernel lab)
#   count_noncont_sse2: 0..15 bytes signed scalar compare + SWAR; exact
#       16/32/48/64-byte vector paths; 16..512 one byte accumulator and one
#       psadbw reduction; > 512 four-accumulator psadbw loop.
#   count_noncont_avx2: <= 64 bytes SWAR and overlapping-load paths;
#       65..256 128-byte unrolled movemask + 64-bit POPCNT; 257..512 one
#       byte accumulator + vpsadbw; > 512 512-byte unrolled loop.
#   scan_fwd_avx2: < 32 remaining bytes or target <= 32: scalar prologue,
#       overlapping 4/8/16-byte tails and bit selection; otherwise a
#       64/32-byte movemask + POPCNT walk with an exact overlapping tail.
#   scan_bwd_avx2: full walks <= 512 bytes: small/medium overlapping path;
#       > 512: 128-byte loop; targeted walks over > 160 bytes: block skip
#       with in-block bit selection (BLSR / LZCNT, never PDEP).
# Measured on Zen 2 against the Rust-intrinsic kernels they replace
# (median of 21 shuffled-order rounds, rust_ns / asm_ns, geometric mean over
# sizes 1 B..16 MiB): sse2 count 1.31x, avx2 count 1.90x (production-shaped
# mix 1.42x), forward scan 2.13x, backward scan 2.76x.

.text

.p2align 5
.globl pi_cc_v1_count_noncont_sse2
.globl _pi_cc_v1_count_noncont_sse2
pi_cc_v1_count_noncont_sse2:
_pi_cc_v1_count_noncont_sse2:
    cmp rsi, 16
    jb 6f
    cmp rsi, 64
    ja 8f
    test rsi, 15
    jnz 8f
    cmp rsi, 16
    jne 2f
    mov eax, 0xbfbfbfbf
    movd xmm0, eax
    pshufd xmm0, xmm0, 0
    pxor xmm1, xmm1
    pxor xmm5, xmm5
    movdqu xmm2, [rdi]
    pcmpgtb xmm2, xmm0
    psubb xmm5, xmm2
    psadbw xmm5, xmm1
    movq rax, xmm5
    psrldq xmm5, 8
    movq rcx, xmm5
    add rax, rcx
    ret
2:
    cmp rsi, 32
    jne 2f
    mov eax, 0xbfbfbfbf
    movd xmm0, eax
    pshufd xmm0, xmm0, 0
    pxor xmm1, xmm1
    pxor xmm5, xmm5
    movdqu xmm2, [rdi]
    movdqu xmm3, [rdi + 16]
    pcmpgtb xmm2, xmm0
    pcmpgtb xmm3, xmm0
    psubb xmm5, xmm2
    psubb xmm5, xmm3
    psadbw xmm5, xmm1
    movq rax, xmm5
    psrldq xmm5, 8
    movq rcx, xmm5
    add rax, rcx
    ret
2:
    cmp rsi, 48
    jne 2f
    mov eax, 0xbfbfbfbf
    movd xmm0, eax
    pshufd xmm0, xmm0, 0
    pxor xmm1, xmm1
    pxor xmm5, xmm5
    movdqu xmm2, [rdi]
    movdqu xmm3, [rdi + 16]
    movdqu xmm4, [rdi + 32]
    pcmpgtb xmm2, xmm0
    pcmpgtb xmm3, xmm0
    pcmpgtb xmm4, xmm0
    psubb xmm5, xmm2
    psubb xmm5, xmm3
    psubb xmm5, xmm4
    psadbw xmm5, xmm1
    movq rax, xmm5
    psrldq xmm5, 8
    movq rcx, xmm5
    add rax, rcx
    ret
2:
    cmp rsi, 64
    jne 8f
    mov eax, 0xbfbfbfbf
    movd xmm0, eax
    pshufd xmm0, xmm0, 0
    pxor xmm1, xmm1
    pxor xmm5, xmm5
    movdqu xmm2, [rdi]
    movdqu xmm3, [rdi + 16]
    pcmpgtb xmm2, xmm0
    pcmpgtb xmm3, xmm0
    psubb xmm5, xmm2
    psubb xmm5, xmm3
    movdqu xmm2, [rdi + 32]
    movdqu xmm3, [rdi + 48]
    pcmpgtb xmm2, xmm0
    pcmpgtb xmm3, xmm0
    psubb xmm5, xmm2
    psubb xmm5, xmm3
    psadbw xmm5, xmm1
    movq rax, xmm5
    psrldq xmm5, 8
    movq rcx, xmm5
    add rax, rcx
    ret

.p2align 5
8:
    cmp rsi, 512
    ja 7f

    # 16..512: one byte accumulator and one vector reduction.
    mov eax, 0xbfbfbfbf
    movd xmm0, eax
    pshufd xmm0, xmm0, 0
    pxor xmm1, xmm1
    pxor xmm2, xmm2
2:
    cmp rsi, 64
    jb 3f
    movdqu xmm7, [rdi]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    movdqu xmm7, [rdi + 16]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    movdqu xmm7, [rdi + 32]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    movdqu xmm7, [rdi + 48]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    add rdi, 64
    sub rsi, 64
    jmp 2b
3:
    cmp rsi, 16
    jb 4f
    movdqu xmm7, [rdi]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    add rdi, 16
    sub rsi, 16
    jmp 3b
4:
    psadbw xmm2, xmm1
    movq r8, xmm2
    psrldq xmm2, 8
    movq rax, xmm2
    add r8, rax
    test rsi, rsi
    jz 5f
    cmp rsi, 3
    ja 8f
2:
    movzx ecx, byte ptr [rdi]
    and ecx, 0xc0
    cmp ecx, 0x80
    setne cl
    movzx ecx, cl
    add r8, rcx
    inc rdi
    dec rsi
    jnz 2b
5:
    mov rax, r8
    ret
8:
    mov r9, rsi
    mov r10, 0x8080808080808080
    mov r11, r10
    shr r11, 7
    cmp rsi, 8
    jb 4f
    mov rax, [rdi]
    add rdi, 8
    sub rsi, 8
3:
    mov rcx, rax
    add rax, rax
    not rax
    and rcx, rax
    and rcx, r10
    shr rcx, 7
    imul rcx, r11
    shr rcx, 56
    sub r9, rcx
    test rsi, rsi
    jz 5f
4:
    cmp rsi, 4
    jb 2f
    mov eax, dword ptr [rdi]
    mov edx, dword ptr [rdi + rsi - 4]
    lea rcx, [rsi * 8 - 32]
    shl rdx, cl
    or rax, rdx
    jmp 9f
2:
    cmp rsi, 2
    jb 2f
    movzx eax, word ptr [rdi]
    movzx edx, word ptr [rdi + rsi - 2]
    lea rcx, [rsi * 8 - 16]
    shl rdx, cl
    or rax, rdx
    jmp 9f
2:
    movzx eax, byte ptr [rdi]
9:
    xor esi, esi
    jmp 3b
5:
    add r8, r9
    mov rax, r8
    ret

    # 0..15: scalar through eight bytes, packed SWAR above eight.
6:
    cmp rsi, 4
    jae 9f
    cmp rsi, 1
    jbe 2f
    xor eax, eax
    cmp byte ptr [rdi], 0xc0
    setge al
    xor ecx, ecx
    cmp byte ptr [rdi + 1], 0xc0
    setge cl
    add eax, ecx
    cmp rsi, 2
    je 5f
    xor ecx, ecx
    cmp byte ptr [rdi + 2], 0xc0
    setge cl
    add eax, ecx
5:
    ret
2:
    test rsi, rsi
    jz 3f
    xor eax, eax
    cmp byte ptr [rdi], 0xc0
    setge al
    ret
3:
    xor eax, eax
    ret
9:
    cmp rsi, 4
    jne 2f
    mov eax, dword ptr [rdi]
    mov ecx, eax
    add eax, eax
    not eax
    and ecx, eax
    and ecx, 0x80808080
    shr ecx, 7
    imul ecx, ecx, 0x01010101
    shr ecx, 24
    mov eax, 4
    sub eax, ecx
    ret
2:
    cmp rsi, 8
    jae 8f
    # Count four bytes with SWAR, then the remaining 1..3 signed-byte tests.
    mov r8d, 4
    mov eax, dword ptr [rdi]
    mov ecx, eax
    add eax, eax
    not eax
    and ecx, eax
    and ecx, 0x80808080
    shr ecx, 7
    imul ecx, ecx, 0x01010101
    shr ecx, 24
    sub r8d, ecx
    xor eax, eax
    cmp byte ptr [rdi + 4], 0xc0
    setge al
    add r8d, eax
    cmp rsi, 5
    je 5f
    xor eax, eax
    cmp byte ptr [rdi + 5], 0xc0
    setge al
    add r8d, eax
    cmp rsi, 6
    je 5f
    xor eax, eax
    cmp byte ptr [rdi + 6], 0xc0
    setge al
    add r8d, eax
5:
    mov eax, r8d
    ret
8:
    mov r9, rsi
    mov r10, 0x8080808080808080
    mov r11, r10
    shr r11, 7
    cmp rsi, 8
    jb 4f
    mov rax, [rdi]
    add rdi, 8
    sub rsi, 8
3:
    mov rcx, rax
    add rax, rax
    not rax
    and rcx, rax
    and rcx, r10
    shr rcx, 7
    imul rcx, r11
    shr rcx, 56
    sub r9, rcx
    test rsi, rsi
    jz 5f
4:
    cmp rsi, 4
    jb 2f
    mov eax, dword ptr [rdi]
    mov edx, dword ptr [rdi + rsi - 4]
    lea rcx, [rsi * 8 - 32]
    shl rdx, cl
    or rax, rdx
    jmp 9f
2:
    cmp rsi, 2
    jb 2f
    movzx eax, word ptr [rdi]
    movzx edx, word ptr [rdi + rsi - 2]
    lea rcx, [rsi * 8 - 16]
    shl rdx, cl
    or rax, rdx
    jmp 9f
2:
    movzx eax, byte ptr [rdi]
9:
    xor esi, esi
    jmp 3b
5:
    mov rax, r9
    ret

    # >512: retain the round-1 four-accumulator implementation.
.p2align 5
7:
    # pcmpgtb(v, 0xbf) is true exactly for non-continuation bytes:
    # signed continuations are -128..-65, all others are greater than -65.
    mov eax, 0xbfbfbfbf
    movd xmm0, eax
    pshufd xmm0, xmm0, 0
    pxor xmm1, xmm1
    pxor xmm2, xmm2
    pxor xmm3, xmm3
    pxor xmm4, xmm4
    pxor xmm5, xmm5
    pxor xmm6, xmm6

    # Keep at least 16 bytes after every main loop so the final vector load
    # never has to reach into bytes already accumulated by an earlier block.
    # Each accumulator receives at most 252 additions before this flush.
    # The residual 16-byte loop can add at most two more before final psadbw.
2:
    cmp rsi, 16144
    jb 3f
    mov ecx, 252
9:
    movdqu xmm7, [rdi]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    movdqu xmm7, [rdi + 16]
    pcmpgtb xmm7, xmm0
    psubb xmm3, xmm7
    movdqu xmm7, [rdi + 32]
    pcmpgtb xmm7, xmm0
    psubb xmm4, xmm7
    movdqu xmm7, [rdi + 48]
    pcmpgtb xmm7, xmm0
    psubb xmm5, xmm7
    add rdi, 64
    sub rsi, 64
    dec ecx
    jnz 9b

    psadbw xmm2, xmm1
    paddq xmm6, xmm2
    pxor xmm2, xmm2
    psadbw xmm3, xmm1
    paddq xmm6, xmm3
    pxor xmm3, xmm3
    psadbw xmm4, xmm1
    paddq xmm6, xmm4
    pxor xmm4, xmm4
    psadbw xmm5, xmm1
    paddq xmm6, xmm5
    pxor xmm5, xmm5
    jmp 2b

3:
    cmp rsi, 80
    jb 4f
    movdqu xmm7, [rdi]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    movdqu xmm7, [rdi + 16]
    pcmpgtb xmm7, xmm0
    psubb xmm3, xmm7
    movdqu xmm7, [rdi + 32]
    pcmpgtb xmm7, xmm0
    psubb xmm4, xmm7
    movdqu xmm7, [rdi + 48]
    pcmpgtb xmm7, xmm0
    psubb xmm5, xmm7
    add rdi, 64
    sub rsi, 64
    jmp 3b

    # Leave 16..31 bytes. Their first 0..15 bytes go through exact-width
    # SWAR below; the final 16 bytes use one in-bounds vector load.
4:
    cmp rsi, 32
    jb 5f
    movdqu xmm7, [rdi]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    add rdi, 16
    sub rsi, 16
    jmp 4b
5:
    sub rsi, 16
    xor r8d, r8d
    test rsi, rsi
    jz 6f
    mov r8, rsi
    mov r10, 0x8080808080808080
    mov r11, r10
    shr r11, 7
    cmp rsi, 8
    jb 9f
    mov rax, [rdi]
    add rdi, 8
    sub rsi, 8
    jmp 3f
9:
    cmp rsi, 4
    jb 4f
    mov eax, dword ptr [rdi]
    mov edx, dword ptr [rdi + rsi - 4]
    lea rcx, [rsi * 8 - 32]
    shl rdx, cl
    or rax, rdx
    jmp 7f
4:
    cmp rsi, 2
    jb 5f
    movzx eax, word ptr [rdi]
    movzx edx, word ptr [rdi + rsi - 2]
    lea rcx, [rsi * 8 - 16]
    shl rdx, cl
    or rax, rdx
    jmp 7f
5:
    movzx eax, byte ptr [rdi]
7:
    add rdi, rsi
    xor esi, esi
3:
    mov r9, rax
    add rax, rax
    not rax
    and r9, rax
    and r9, r10
    shr r9, 7
    imul r9, r11
    shr r9, 56
    sub r8, r9
    test rsi, rsi
    jz 6f
    cmp rsi, 4
    jb 4f
    mov eax, dword ptr [rdi]
    mov edx, dword ptr [rdi + rsi - 4]
    lea rcx, [rsi * 8 - 32]
    shl rdx, cl
    or rax, rdx
    jmp 7f
4:
    cmp rsi, 2
    jb 5f
    movzx eax, word ptr [rdi]
    movzx edx, word ptr [rdi + rsi - 2]
    lea rcx, [rsi * 8 - 16]
    shl rdx, cl
    or rax, rdx
    jmp 7f
5:
    movzx eax, byte ptr [rdi]
7:
    add rdi, rsi
    xor esi, esi
3:
    mov r9, rax
    add rax, rax
    not rax
    and r9, rax
    and r9, r10
    shr r9, 7
    imul r9, r11
    shr r9, 56
    sub r8, r9
6:
    movdqu xmm7, [rdi + rsi]
    pcmpgtb xmm7, xmm0
    psubb xmm2, xmm7
    psadbw xmm2, xmm1
    paddq xmm6, xmm2
    psadbw xmm3, xmm1
    paddq xmm6, xmm3
    psadbw xmm4, xmm1
    paddq xmm6, xmm4
    psadbw xmm5, xmm1
    paddq xmm6, xmm5
    movq rax, xmm6
    psrldq xmm6, 8
    movq rcx, xmm6
    add rax, rcx
    add rax, r8
    ret

    # Exact-width/SWAR path for 0..15 bytes. For each byte, bit 7 of
    # (x & ~(x << 1) & 0x80..) is set iff the byte is a continuation.
    # Multiplying the 0/1 lanes by 0x0101.. sums them in the top byte.
8:
    test rsi, rsi
    jnz 2f
    xor eax, eax
    ret
2:
    cmp rsi, 1
    jne 9f
    movzx eax, byte ptr [rdi]
    and eax, 0xc0
    cmp eax, 0x80
    setne al
    movzx eax, al
    ret
9:
    cmp rsi, 2
    jne 9f
    movzx eax, word ptr [rdi]
    mov ecx, eax
    and eax, 0xc0
    and ecx, 0xc000
    cmp eax, 0x80
    setne al
    cmp ecx, 0x8000
    setne cl
    movzx eax, al
    movzx ecx, cl
    add eax, ecx
    ret
9:
    cmp rsi, 3
    jne 9f
    movzx eax, word ptr [rdi]
    mov ecx, eax
    and eax, 0xc0
    and ecx, 0xc000
    movzx edx, byte ptr [rdi + 2]
    and edx, 0xc0
    xor r8d, r8d
    cmp edx, 0x80
    setne r8b
    cmp eax, 0x80
    setne al
    cmp ecx, 0x8000
    setne cl
    movzx eax, al
    movzx ecx, cl
    add eax, ecx
    add eax, r8d
    ret
9:
    cmp rsi, 4
    jne 9f
    mov eax, dword ptr [rdi]
    mov ecx, eax
    add eax, eax
    not eax
    and ecx, eax
    and ecx, 0x80808080
    shr ecx, 7
    imul ecx, ecx, 0x01010101
    shr ecx, 24
    mov eax, 4
    sub eax, ecx
    ret
9:
    mov r8, rsi
    mov r10, 0x8080808080808080
    mov r11, r10
    shr r11, 7
    cmp rsi, 8
    jb 9f
    mov rax, [rdi]
    add rdi, 8
    sub rsi, 8
    jmp 3f
9:
    cmp rsi, 4
    jb 4f
    mov eax, dword ptr [rdi]
    mov edx, dword ptr [rdi + rsi - 4]
    lea rcx, [rsi * 8 - 32]
    shl rdx, cl
    or rax, rdx
    jmp 7f
4:
    cmp rsi, 2
    jb 5f
    movzx eax, word ptr [rdi]
    movzx edx, word ptr [rdi + rsi - 2]
    lea rcx, [rsi * 8 - 16]
    shl rdx, cl
    or rax, rdx
    jmp 7f
5:
    movzx eax, byte ptr [rdi]
7:
    xor esi, esi
3:
    mov r9, rax
    add rax, rax
    not rax
    and r9, rax
    and r9, r10
    shr r9, 7
    imul r9, r11
    shr r9, 56
    sub r8, r9
    test rsi, rsi
    jz 6f
    # The remaining 1..7 bytes use the same exact-width packing scheme.
    cmp rsi, 4
    jb 4f
    mov eax, dword ptr [rdi]
    mov edx, dword ptr [rdi + rsi - 4]
    lea rcx, [rsi * 8 - 32]
    shl rdx, cl
    or rax, rdx
    jmp 7f
4:
    cmp rsi, 2
    jb 5f
    movzx eax, word ptr [rdi]
    movzx edx, word ptr [rdi + rsi - 2]
    lea rcx, [rsi * 8 - 16]
    shl rdx, cl
    or rax, rdx
    jmp 7f
5:
    movzx eax, byte ptr [rdi]
7:
    xor esi, esi
3:
    mov r9, rax
    add rax, rax
    not rax
    and r9, rax
    and r9, r10
    shr r9, 7
    imul r9, r11
    shr r9, 56
    sub r8, r9
6:
    mov rax, r8
    ret
7:
    xor eax, eax
    ret

.p2align 5
.globl pi_cc_v1_count_noncont_avx2
.globl _pi_cc_v1_count_noncont_avx2
pi_cc_v1_count_noncont_avx2:
_pi_cc_v1_count_noncont_avx2:
    cmp rsi, 64
    jbe 2f
    cmp rsi, 256
    jbe 4f
    cmp rsi, 512
    ja 3f
    # 257..512: the large loop's SAD strategy without its 512-byte
    # batching setup.  At most sixteen comparisons enter the byte lanes.
    mov ecx, -65
    vmovd xmm4, ecx
    vpbroadcastb ymm4, xmm4
    vpxor ymm5, ymm5, ymm5
    vpxor ymm6, ymm6, ymm6
    cmp rsi, 128
    jb 8f
7:
    vmovdqu ymm10, ymmword ptr [rdi]
    vmovdqu ymm11, ymmword ptr [rdi + 32]
    vmovdqu ymm12, ymmword ptr [rdi + 64]
    vmovdqu ymm13, ymmword ptr [rdi + 96]
    vpcmpgtb ymm10, ymm10, ymm4
    vpcmpgtb ymm11, ymm11, ymm4
    vpcmpgtb ymm12, ymm12, ymm4
    vpcmpgtb ymm13, ymm13, ymm4
    vpsubb ymm6, ymm6, ymm10
    vpsubb ymm6, ymm6, ymm11
    vpsubb ymm6, ymm6, ymm12
    vpsubb ymm6, ymm6, ymm13
    add rdi, 128
    sub rsi, 128
    cmp rsi, 128
    jae 7b
8:
    cmp rsi, 32
    jb 8f
    vmovdqu ymm10, ymmword ptr [rdi]
    vpcmpgtb ymm10, ymm10, ymm4
    vpsubb ymm6, ymm6, ymm10
    add rdi, 32
    sub rsi, 32
    jmp 8b
8:
    vpsadbw ymm6, ymm6, ymm5
    vextracti128 xmm1, ymm6, 1
    vpaddq xmm6, xmm6, xmm1
    vmovq rax, xmm6
    vpextrq rdx, xmm6, 1
    add rax, rdx
    test rsi, rsi
    jz 9f
    vmovdqu ymm10, ymmword ptr [rdi + rsi - 32]
    vpcmpgtb ymm10, ymm10, ymm4
    vpmovmskb edx, ymm10
    mov ecx, 32
    sub ecx, esi
    shr edx, cl
    popcnt edx, edx
    add rax, rdx
9:
    vzeroupper
    ret

4:
    # Mid-size inputs avoid the large loop's nine accumulator initialisations
    # and final SAD tree.  Two masks form one 64-bit popcount per 64 bytes.
    xor eax, eax
    mov ecx, -65
    vmovd xmm4, ecx
    vpbroadcastb ymm4, xmm4
    cmp rsi, 128
    jb 5f
    .p2align 4
4:
    vmovdqu ymm0, ymmword ptr [rdi]
    vpcmpgtb ymm0, ymm0, ymm4
    vmovdqu ymm1, ymmword ptr [rdi + 32]
    vpcmpgtb ymm1, ymm1, ymm4
    vmovdqu ymm2, ymmword ptr [rdi + 64]
    vpcmpgtb ymm2, ymm2, ymm4
    vmovdqu ymm3, ymmword ptr [rdi + 96]
    vpcmpgtb ymm3, ymm3, ymm4
    vpmovmskb r8d, ymm0
    vpmovmskb edx, ymm1
    shl rdx, 32
    or rdx, r8
    popcnt rdx, rdx
    add rax, rdx
    vpmovmskb r8d, ymm2
    vpmovmskb edx, ymm3
    shl rdx, 32
    or rdx, r8
    popcnt rdx, rdx
    add rax, rdx
    add rdi, 128
    sub rsi, 128
    cmp rsi, 128
    jae 4b
5:
    cmp rsi, 64
    jb 5f
    vmovdqu ymm0, ymmword ptr [rdi]
    vpcmpgtb ymm0, ymm0, ymm4
    vmovdqu ymm1, ymmword ptr [rdi + 32]
    vpcmpgtb ymm1, ymm1, ymm4
    vpmovmskb r8d, ymm0
    vpmovmskb edx, ymm1
    shl rdx, 32
    or rdx, r8
    popcnt rdx, rdx
    add rax, rdx
    add rdi, 64
    sub rsi, 64
5:
    cmp rsi, 32
    jb 5f
    vmovdqu ymm0, ymmword ptr [rdi]
    vpcmpgtb ymm0, ymm0, ymm4
    vpmovmskb edx, ymm0
    popcnt edx, edx
    add rax, rdx
    add rdi, 32
    sub rsi, 32
5:
    test rsi, rsi
    jz 6f
    vmovdqu ymm0, ymmword ptr [rdi + rsi - 32]
    vpcmpgtb ymm0, ymm0, ymm4
    vpmovmskb edx, ymm0
    mov ecx, 32
    sub ecx, esi
    shr edx, cl
    popcnt edx, edx
    add rax, rdx
6:
    vzeroupper
    ret
3:
    # A continuation is exactly signed byte -128..-65.  Each compare
    # produces -1 for a non-continuation, which subtraction turns into +1.
    mov ecx, -65
    vmovd xmm4, ecx
    vpbroadcastb ymm4, xmm4
    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    vpxor ymm2, ymm2, ymm2
    vpxor ymm3, ymm3, ymm3
    vpxor ymm5, ymm5, ymm5
    vpxor ymm6, ymm6, ymm6
    vpxor ymm7, ymm7, ymm7
    vpxor ymm8, ymm8, ymm8
    vpxor ymm9, ymm9, ymm9

9:
    cmp rsi, 512
    jb 9f
    mov ecx, 63
    .p2align 5
4:
    vmovdqu ymm10, ymmword ptr [rdi]
    vmovdqu ymm11, ymmword ptr [rdi + 32]
    vmovdqu ymm12, ymmword ptr [rdi + 64]
    vmovdqu ymm13, ymmword ptr [rdi + 96]
    vpcmpgtb ymm10, ymm10, ymm4
    vpcmpgtb ymm11, ymm11, ymm4
    vpcmpgtb ymm12, ymm12, ymm4
    vpcmpgtb ymm13, ymm13, ymm4
    vpsubb ymm6, ymm6, ymm10
    vpsubb ymm7, ymm7, ymm11
    vpsubb ymm8, ymm8, ymm12
    vpsubb ymm9, ymm9, ymm13
    vmovdqu ymm10, ymmword ptr [rdi + 128]
    vmovdqu ymm11, ymmword ptr [rdi + 160]
    vmovdqu ymm12, ymmword ptr [rdi + 192]
    vmovdqu ymm13, ymmword ptr [rdi + 224]
    vpcmpgtb ymm10, ymm10, ymm4
    vpcmpgtb ymm11, ymm11, ymm4
    vpcmpgtb ymm12, ymm12, ymm4
    vpcmpgtb ymm13, ymm13, ymm4
    vpsubb ymm6, ymm6, ymm10
    vpsubb ymm7, ymm7, ymm11
    vpsubb ymm8, ymm8, ymm12
    vpsubb ymm9, ymm9, ymm13
    vmovdqu ymm10, ymmword ptr [rdi + 256]
    vmovdqu ymm11, ymmword ptr [rdi + 288]
    vmovdqu ymm12, ymmword ptr [rdi + 320]
    vmovdqu ymm13, ymmword ptr [rdi + 352]
    vpcmpgtb ymm10, ymm10, ymm4
    vpcmpgtb ymm11, ymm11, ymm4
    vpcmpgtb ymm12, ymm12, ymm4
    vpcmpgtb ymm13, ymm13, ymm4
    vpsubb ymm6, ymm6, ymm10
    vpsubb ymm7, ymm7, ymm11
    vpsubb ymm8, ymm8, ymm12
    vpsubb ymm9, ymm9, ymm13
    vmovdqu ymm10, ymmword ptr [rdi + 384]
    vmovdqu ymm11, ymmword ptr [rdi + 416]
    vmovdqu ymm12, ymmword ptr [rdi + 448]
    vmovdqu ymm13, ymmword ptr [rdi + 480]
    vpcmpgtb ymm10, ymm10, ymm4
    vpcmpgtb ymm11, ymm11, ymm4
    vpcmpgtb ymm12, ymm12, ymm4
    vpcmpgtb ymm13, ymm13, ymm4
    vpsubb ymm6, ymm6, ymm10
    vpsubb ymm7, ymm7, ymm11
    vpsubb ymm8, ymm8, ymm12
    vpsubb ymm9, ymm9, ymm13
    add rdi, 512
    sub rsi, 512
    dec ecx
    jz 5f
    cmp rsi, 512
    jae 4b
5:
    # At most 252 additions per byte lane since the previous flush.
    vpsadbw ymm6, ymm6, ymm5
    vpsadbw ymm7, ymm7, ymm5
    vpsadbw ymm8, ymm8, ymm5
    vpsadbw ymm9, ymm9, ymm5
    vpaddq ymm0, ymm0, ymm6
    vpaddq ymm1, ymm1, ymm7
    vpaddq ymm2, ymm2, ymm8
    vpaddq ymm3, ymm3, ymm9
    vpxor ymm6, ymm6, ymm6
    vpxor ymm7, ymm7, ymm7
    vpxor ymm8, ymm8, ymm8
    vpxor ymm9, ymm9, ymm9
    jmp 9b

9:
    cmp rsi, 128
    jb 3f
    vmovdqu ymm10, ymmword ptr [rdi]
    vmovdqu ymm11, ymmword ptr [rdi + 32]
    vmovdqu ymm12, ymmword ptr [rdi + 64]
    vmovdqu ymm13, ymmword ptr [rdi + 96]
    vpcmpgtb ymm10, ymm10, ymm4
    vpcmpgtb ymm11, ymm11, ymm4
    vpcmpgtb ymm12, ymm12, ymm4
    vpcmpgtb ymm13, ymm13, ymm4
    vpsubb ymm6, ymm6, ymm10
    vpsubb ymm7, ymm7, ymm11
    vpsubb ymm8, ymm8, ymm12
    vpsubb ymm9, ymm9, ymm13
    add rdi, 128
    sub rsi, 128
    jmp 9b

3:
    cmp rsi, 32
    jb 7f
    vmovdqu ymm10, ymmword ptr [rdi]
    vpcmpgtb ymm10, ymm10, ymm4
    vpsubb ymm6, ymm6, ymm10
    add rdi, 32
    sub rsi, 32
    jmp 3b

7:
    vpsadbw ymm6, ymm6, ymm5
    vpsadbw ymm7, ymm7, ymm5
    vpsadbw ymm8, ymm8, ymm5
    vpsadbw ymm9, ymm9, ymm5
    vpaddq ymm0, ymm0, ymm6
    vpaddq ymm1, ymm1, ymm7
    vpaddq ymm2, ymm2, ymm8
    vpaddq ymm3, ymm3, ymm9
    vpaddq ymm0, ymm0, ymm1
    vpaddq ymm2, ymm2, ymm3
    vpaddq ymm0, ymm0, ymm2
    vextracti128 xmm1, ymm0, 1
    vpaddq xmm0, xmm0, xmm1
    vmovq rax, xmm0
    vpextrq rdx, xmm0, 1
    add rax, rdx
    test rsi, rsi
    jz 8f
    # Reload only the final 32 in-bounds bytes.  Discard the low 32-rem
    # mask bits, which belong to bytes counted by the preceding blocks.
    lea r8, [rdi + rsi - 32]
    vmovdqu ymm10, ymmword ptr [r8]
    vpcmpgtb ymm10, ymm10, ymm4
    vpmovmskb edx, ymm10
    mov ecx, 32
    sub ecx, esi
    shr edx, cl
    popcnt edx, edx
    add rax, rdx
8:
    vzeroupper
    ret
2:
    # Tiny inputs stay in GPRs.  SWAR marks continuation bytes at bit 7:
    # bit7 & !bit6 = v & ~(v << 1) & 0x80 per byte.
    cmp rsi, 4
    jae 3f
    test rsi, rsi
    jz 8f
    cmp rsi, 1
    jne 2f
    movzx eax, byte ptr [rdi]
    and eax, 0xc0
    cmp eax, 0x80
    setne al
    movzx eax, al
    ret
2:
    movzx eax, word ptr [rdi]
    movzx ecx, word ptr [rdi + rsi - 2]
    mov edx, eax
    mov r8d, ecx
    shl edx, 1
    shl r8d, 1
    not edx
    not r8d
    and eax, edx
    and ecx, r8d
    and eax, 0x8080
    and ecx, 0x8080
    # The second overlapping word contributes only its high 16-len bytes.
    mov edx, 4
    sub edx, esi
    shl edx, 3
    mov r8d, ecx
    mov ecx, edx
    shr r8d, cl
    shl r8d, cl
    mov ecx, r8d
    popcnt rax, rax
    popcnt rcx, rcx
    add rax, rcx
    sub rsi, rax
    mov rax, rsi
    ret
3:
    cmp rsi, 8
    jae 4f
    mov eax, dword ptr [rdi]
    mov ecx, dword ptr [rdi + rsi - 4]
    mov edx, eax
    mov r8d, ecx
    shl edx, 1
    shl r8d, 1
    not edx
    not r8d
    and eax, edx
    and ecx, r8d
    and eax, 0x80808080
    and ecx, 0x80808080
    mov edx, 8
    sub edx, esi
    shl edx, 3
    mov r8, rcx
    mov ecx, edx
    shr r8, cl
    shl r8, cl
    mov rcx, r8
    cmp rsi, 8
    jne 9f
    xor ecx, ecx
9:
    popcnt rax, rax
    popcnt rcx, rcx
    add rax, rcx
    sub rsi, rax
    mov rax, rsi
    ret
4:
    cmp rsi, 16
    jae 5f
    mov rax, qword ptr [rdi]
    mov rcx, qword ptr [rdi + rsi - 8]
    mov rdx, rax
    mov r8, rcx
    shl rdx, 1
    shl r8, 1
    not rdx
    not r8
    and rax, rdx
    and rcx, r8
    mov rdx, 0x8080808080808080
    and rax, rdx
    and rcx, rdx
    mov edx, 16
    sub edx, esi
    shl edx, 3
    mov r8, rcx
    mov ecx, edx
    shr r8, cl
    shl r8, cl
    mov rcx, r8
    cmp rsi, 8
    jne 9f
    xor ecx, ecx
9:
    popcnt rax, rax
    popcnt rcx, rcx
    add rax, rcx
    sub rsi, rax
    mov rax, rsi
    ret
5:
    # Signed > -64 is exactly NOT (10xxxxxx).
    mov eax, 0xbf
    vmovd xmm0, eax
    vpbroadcastb ymm6, xmm0
    cmp rsi, 32
    jae 6f
    # Two 16-byte windows cover 16..31 bytes.  Discard duplicated bytes.
    vmovdqu xmm0, xmmword ptr [rdi]
    vmovdqu xmm1, xmmword ptr [rdi + rsi - 16]
    vpcmpgtb xmm0, xmm0, xmm6
    vpcmpgtb xmm1, xmm1, xmm6
    vpmovmskb r8d, xmm0
    vpmovmskb eax, xmm1
    mov ecx, 32
    sub ecx, esi
    mov rdx, rax
    shr rdx, cl
    shl rdx, cl
    mov eax, edx
    cmp rsi, 16
    jne 9f
    xor eax, eax
9:
    popcnt rax, rax
    popcnt r8, r8
    add rax, r8
    vzeroupper
    ret
6:
    cmp rsi, 64
    jae 7f
    # Two 32-byte windows cover 32..63 bytes.
    vmovdqu ymm0, ymmword ptr [rdi]
    vmovdqu ymm1, ymmword ptr [rdi + rsi - 32]
    vpcmpgtb ymm0, ymm0, ymm6
    vpcmpgtb ymm1, ymm1, ymm6
    vpmovmskb r8d, ymm0
    vpmovmskb eax, ymm1
    mov ecx, 64
    sub ecx, esi
    mov rdx, rax
    shr rdx, cl
    shl rdx, cl
    mov eax, edx
    cmp rsi, 32
    jne 9f
    xor eax, eax
9:
    popcnt rax, rax
    popcnt r8, r8
    add rax, r8
    vzeroupper
    ret
7:
    cmp rsi, 64
    jne 8f
    vmovdqu ymm0, ymmword ptr [rdi]
    vmovdqu ymm1, ymmword ptr [rdi + 32]
    vpcmpgtb ymm0, ymm0, ymm6
    vpcmpgtb ymm1, ymm1, ymm6
    vpmovmskb r8d, ymm0
    vpmovmskb eax, ymm1
    popcnt rax, rax
    popcnt r8, r8
    add rax, r8
    vzeroupper
    ret
8:
    xor eax, eax
    ret

.p2align 5
.globl pi_cc_v1_scan_fwd_avx2
.globl _pi_cc_v1_scan_fwd_avx2
pi_cc_v1_scan_fwd_avx2:
_pi_cc_v1_scan_fwd_avx2:
    cmp rdx, rsi
    ja 24f
    mov r8, rsi
    sub r8, rdx
    cmp r8, 32
    jb 23f
    cmp rcx, 32
    ja 24f
23:
    .p2align 5
    # A zero target never examines memory (the pointer may be dangling).
    test rcx, rcx
    jnz 3f
    mov rax, rdx
    xor edx, edx
    ret
3:
    # Degenerate inputs of 0..3 bytes avoid vector setup and vzeroupper.
    mov r8, rsi
    sub r8, rdx
    cmp r8, 7
    jae 2f
    xor r9d, r9d
8:
    cmp rdx, rsi
    jae 7f
    movzx eax, byte ptr [rdi + rdx]
    and eax, 0xC0
    cmp eax, 0x80
    je 6f
    inc r9
6:
    inc rdx
    cmp r9, rcx
    jb 8b
4:
    cmp rdx, rsi
    jae 7f
    movzx eax, byte ptr [rdi + rdx]
    and eax, 0xC0
    cmp eax, 0x80
    jne 7f
    inc rdx
    jmp 4b
7:
    mov rax, rdx
    mov rdx, r9
    ret
2:

    # Continuation-byte classifier for full and partial vector loads.
    mov eax, 0xC0
    vmovd xmm1, eax
    vpbroadcastb ymm1, xmm1
    mov eax, 0x80
    vmovd xmm2, eax
    vpbroadcastb ymm2, xmm2
    xor r9d, r9d                    # seen

    # Full blocks: all loads are exactly 32 in-bounds bytes.
2:
    mov r8, rsi
    sub r8, rdx
    cmp r8, 32
    jb 3f
    vmovdqu ymm0, ymmword ptr [rdi + rdx]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb r11d, ymm0
    not r11d                        # non-continuation bits, exactly 32
    lea r10, [rdx + 32]             # block end
    popcnt rax, r11
    mov r8, rcx
    sub r8, r9                      # chars still needed
    cmp rax, r8
    jae 6f
    add r9, rax
    mov rdx, r10
    jmp 2b

    # Fewer than 32 bytes remain. Pair overlapping loads at both ends so
    # every byte is read exactly inside the buffer; duplicate bytes classify
    # identically and collapse in the OR of the shifted masks.
3:
    cmp r8, 16
    jb 4f
    vmovdqu xmm0, xmmword ptr [rdi + rdx]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb r11d, xmm0
    not r11d
    and r11d, 0xFFFF
    vmovdqu xmm0, xmmword ptr [rdi + rsi - 16]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb r10d, xmm0
    not r10d
    and r10d, 0xFFFF
    sub r8, 16
    shlx r10, r10, r8
    or r11, r10
    mov r10, rsi                    # block end = input end
    jmp 5f
4:
    cmp r8, 8
    jb 7f
    vmovq xmm0, qword ptr [rdi + rdx]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb r11d, xmm0
    not r11d
    and r11d, 0xFF
    vmovq xmm0, qword ptr [rdi + rsi - 8]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb r10d, xmm0
    not r10d
    and r10d, 0xFF
    sub r8, 8
    shlx r10, r10, r8
    or r11, r10
    mov r10, rsi
    jmp 5f
7:
    cmp r8, 4
    jb 9f                         # only 0..3 bytes need scalar handling
    vmovd xmm0, dword ptr [rdi + rdx]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb r11d, xmm0
    not r11d
    and r11d, 0xF
    vmovd xmm0, dword ptr [rdi + rsi - 4]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb r10d, xmm0
    not r10d
    and r10d, 0xF
    sub r8, 4
    shlx r10, r10, r8
    or r11, r10
    mov r10, rsi
5:
    # The final partial block either completes the scan or contains target.
    popcnt rax, r11
    mov r8, rcx                    # restore target after mask shifts
    sub r8, r9
    cmp rax, r8
    jae 6f
    add r9, rax
    mov rdx, rsi
    jmp 2f

6:
    # If every character in this block is needed, the highest set bit is the
    # answer. Otherwise use blsr, never PDEP (very slow on Zen 2).
    mov r9, rcx                    # target is now the returned seen count
    cmp r8, rax
    jne 7f
    bsr rax, r11
    jmp 8f
7:
    dec r8
    jz 7f
6:
    blsr r11, r11
    dec r8
    jnz 6b
7:
    tzcnt rax, r11                 # selected byte's block-relative index
8:
    lea r8, [rdx + rax + 1]        # one past the selected byte

    # Complete its continuation run from bits already in this block. A 64-bit
    # shift also handles index 31 (shift count 32) correctly.
    inc rax
    shrx r11, r11, rax
    tzcnt rax, r11
    test r11, r11
    jz 8f
    add r8, rax
    mov rdx, r8
    jmp 2f
8:
    mov rdx, r10                   # continuation run reaches the block end
    cmp rdx, rsi
    jae 6f

    # A run may continue into later blocks. Consume full 32-byte runs and
    # finish with at most 31 checked scalar bytes (no over-read).
3:
    mov r8, rsi
    sub r8, rdx
    cmp r8, 32
    jb 4f
    vmovdqu ymm0, ymmword ptr [rdi + rdx]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb r11d, ymm0
    not r11d
    test r11d, r11d
    jz 5f
    tzcnt rax, r11
    add rdx, rax
    jmp 2f
5:
    add rdx, 32
    jmp 3b
4:
    cmp rdx, rsi
    jae 6f
    movzx eax, byte ptr [rdi + rdx]
    and eax, 0xC0
    cmp eax, 0x80
    jne 6f
    inc rdx
    jmp 4b
6:
    mov rax, rdx
    jmp 2f

    # Degenerate tails of 0..3 bytes are cheaper and simpler as a scalar walk.
9:
    cmp rdx, rsi
    jae 2f
    movzx eax, byte ptr [rdi + rdx]
    and eax, 0xC0
    cmp eax, 0x80
    je 7f
    inc r9
7:
    inc rdx
    cmp r9, rcx
    jb 9b
    # The selected non-continuation byte was consumed; finish its run.
    jmp 4b

2:
    mov rax, rdx
    mov rdx, r9
    vzeroupper
    ret

24:
    .p2align 5
    cmp rdx, rsi
    ja 2f
    mov rax, rdx
    lea rdi, [rdi + rdx]
    mov r8, rsi
    sub r8, rdx
    xor edx, edx
    test rcx, rcx
    jz 2f
    test r8, r8
    jz 2f
    cmp r8, 32
    jb 4f
    mov r9, rcx
    mov r10d, 0xc0
    vmovd xmm0, r10d
    vpbroadcastb ymm0, xmm0
    mov r10d, 0x80
    vmovd xmm3, r10d
    vpbroadcastb ymm3, xmm3
3:
    cmp r8, 64
    jb 5f
    cmp r9, 64
    jb 5f
    vmovdqu ymm1, [rdi]
    vmovdqu ymm2, [rdi + 32]
    vpand ymm1, ymm1, ymm0
    vpand ymm2, ymm2, ymm0
    vpcmpeqb ymm1, ymm1, ymm3
    vpcmpeqb ymm2, ymm2, ymm3
    vpmovmskb r11d, ymm1
    vpmovmskb r10d, ymm2
    not r11d
    not r10d
    popcnt eax, r11d
    popcnt ecx, r10d
    add eax, ecx
    cmp rax, r9
    ja 6f
    add rdi, 64
    sub r8, 64
    sub r9, rax
    add rdx, rax
    jmp 3b
6:
    cmp r9, 32
    jb 5f
    add rdi, 32
    sub r8, 32
    popcnt eax, r11d
    sub r9, rax
    add rdx, rax
    test r9, r9
    jz 6f
    mov r11d, r10d
    jmp 7f
5:
    test r9, r9
    jz 6f
    cmp r8, 32
    jb 8f
    vmovdqu ymm1, [rdi]
    vpand ymm1, ymm1, ymm0
    vpcmpeqb ymm1, ymm1, ymm3
    vpmovmskb r11d, ymm1
    not r11d
    popcnt eax, r11d
    cmp rax, r9
    ja 7f
    add rdi, 32
    sub r8, 32
    sub r9, rax
    add rdx, rax
    jmp 5b
7:
    add rdx, r9
7:
    dec r9
    jz 7f
    blsr r11, r11
    jmp 7b
7:
    tzcnt r11, r11
    inc r11
    add rdi, r11
    sub r8, r11
    xor r9d, r9d
    jmp 6f
8:
    test r9, r9
    jz 6f
    test r8, r8
    jz 9f
    vmovdqu ymm1, [rdi + r8 - 32]
    vpand ymm1, ymm1, ymm0
    vpcmpeqb ymm1, ymm1, ymm3
    vpmovmskb r11d, ymm1
    not r11d
    mov r10d, 32
    sub r10, r8
    shrx r11d, r11d, r10d
    popcnt eax, r11d
    cmp rax, r9
    ja 7f
    add rdx, rax
    add rdi, r8
    xor r8d, r8d
    jmp 9f
7:
    add rdx, r9
7:
    dec r9
    jz 7f
    blsr r11, r11
    jmp 7b
7:
    tzcnt r11, r11
    inc r11
    add rdi, r11
    sub r8, r11
    xor r9d, r9d
    jmp 6f
6:
    test r8, r8
    jz 9f
    movzx r10d, byte ptr [rdi]
    and r10d, 0xc0
    cmp r10d, 0x80
    jne 9f
    inc rdi
    dec r8
    jmp 6b
9:
    mov rax, rsi
    sub rax, r8
    vzeroupper
    ret
4:
5:
    movzx r10d, byte ptr [rdi]
    and r10d, 0xc0
    inc rdi
    dec r8
    cmp r10d, 0x80
    je 6f
    dec rcx
    inc rdx
    test rcx, rcx
    jz 7f
6:
    test r8, r8
    jnz 5b
    jmp 9f
7:
    test r8, r8
    jz 9f
    movzx r10d, byte ptr [rdi]
    and r10d, 0xc0
    cmp r10d, 0x80
    jne 9f
    inc rdi
    dec r8
    jmp 7b
9:
    mov rax, rsi
    sub rax, r8
    ret
2:
    ret

.p2align 5
.globl pi_cc_v1_scan_bwd_avx2
.globl _pi_cc_v1_scan_bwd_avx2
pi_cc_v1_scan_bwd_avx2:
_pi_cc_v1_scan_bwd_avx2:
    cmp rsi, 512
    ja 2002f
    # rax = unexamined prefix length, r8 = non-continuations already seen.
    mov rax, rsi
    xor r8d, r8d
    test rdx, rdx
    jz 1091f
    test rax, rax
    jz 1091f
    cmp rax, 1
    je 1047f
    cmp rax, 4
    jb 10402f
    cmp rax, 8
    jb 1048f
    cmp rax, 64
    jbe 1004f
    cmp rax, 160
    jbe 1004f
    cmp rdx, 160
    jne 1004f
    jmp 2002f
1004:
    mov ecx, 0xC0
    vmovd xmm0, ecx
    vpbroadcastb ymm0, xmm0
    mov ecx, 0x80
    vmovd xmm1, ecx
    vpbroadcastb ymm1, xmm1

1002:  # Full 64-byte blocks. The high half is nearest the original end.
    cmp rax, 64
    jb 1003f
    vmovdqu ymm2, [rdi + rax - 32]
    vpand ymm2, ymm2, ymm0
    vpcmpeqb ymm2, ymm2, ymm1
    vpmovmskb r10d, ymm2
    not r10d
    vmovdqu ymm2, [rdi + rax - 64]
    vpand ymm2, ymm2, ymm0
    vpcmpeqb ymm2, ymm2, ymm1
    vpmovmskb r9d, ymm2
    not r9d
    shl r10, 32
    or r9, r10
    popcnt r11, r9
    mov rcx, rdx
    sub rcx, r8
    cmp rcx, r11
    jbe 1006f
    add r8, r11
    sub rax, 64
    jmp 1002b

1003:  # 32..63 bytes: take the last 32, then finish the masked head below.
    cmp rsi, 64
    jae 1004f
    cmp rax, 32
    jb 1004f
    vmovdqu ymm2, [rdi + rax - 32]
    vpand ymm2, ymm2, ymm0
    vpcmpeqb ymm2, ymm2, ymm1
    vpmovmskb r9d, ymm2
    not r9d
    popcnt r11, r9
    mov rcx, rdx
    sub rcx, r8
    cmp rcx, r11
    jbe 1005f
    add r8, r11
    sub rax, 32

1004:  # Exact-width in-bounds head; bzhi discards overlapped bytes.
    cmp rax, 1
    je 10401f
    test rax, rax
    jz 1008f
    cmp rax, 32
    jb 1042f
    vmovdqu ymm2, [rdi + 32]
    vpand ymm2, ymm2, ymm0
    vpcmpeqb ymm2, ymm2, ymm1
    vpmovmskb r10d, ymm2
    not r10d
    vmovdqu ymm2, [rdi]
    vpand ymm2, ymm2, ymm0
    vpcmpeqb ymm2, ymm2, ymm1
    vpmovmskb r9d, ymm2
    not r9d
    shl r10, 32
    or r9, r10
    bzhi r9, r9, rax
    popcnt r11, r9
    mov rcx, rdx
    sub rcx, r8
    cmp rcx, r11
    jbe 1072f
    add r8, r11
    jmp 1008f
10401: # A single residual byte always finishes at offset zero.
    movzx ecx, byte ptr [rdi]
    and ecx, 0xC0
    cmp ecx, 0x80
    setne cl
    movzx ecx, cl
    add r8, rcx
    jmp 1008f
1042:
    cmp rsi, 16
    jb 1043f
    xor r11d, r11d
    cmp rax, 16
    jb 1421f
    mov r11, rax
    sub r11, 16
1421:
    vmovdqu xmm2, [rdi + r11]
    vpand xmm2, xmm2, xmm0
    vpcmpeqb xmm2, xmm2, xmm1
    vpmovmskb r9d, xmm2
    not r9d
    movzx r9d, r9w
    bzhi r9, r9, rax
    mov r10, r11
    popcnt r11, r9
    mov rcx, rdx
    sub rcx, r8
    cmp rcx, r11
    ja 1422f
    jne 1423f
    mov r11, r10
    tzcnt r10, r9
    lea rax, [r11 + r10]
    vzeroupper
    ret
1423:
    mov r11, r10
    jmp 1071f
1422:
    add r8, r11
    cmp rax, 16
    jbe 1008f
    sub rax, 16
    jmp 1042b
1043:
    cmp rsi, 8
    jb 1044f
    xor r11d, r11d
    cmp rax, 8
    jb 1431f
    mov r11, rax
    sub r11, 8
1431:
    vmovq xmm2, [rdi + r11]
    vpand xmm2, xmm2, xmm0
    vpcmpeqb xmm2, xmm2, xmm1
    vpmovmskb r9d, xmm2
    not r9d
    and r9d, 0xFF
    bzhi r9, r9, rax
    mov r10, r11
    popcnt r11, r9
    mov rcx, rdx
    sub rcx, r8
    cmp rcx, r11
    ja 1432f
    jne 1433f
    mov r11, r10
    tzcnt r10, r9
    lea rax, [r11 + r10]
    vzeroupper
    ret
1433:
    mov r11, r10
    jmp 1071f
1432:
    add r8, r11
    cmp rax, 8
    jbe 1008f
    sub rax, 8
    jmp 1043b
1047: # One byte: the answer is always offset zero; only seen varies.
    movzx edx, byte ptr [rdi]
    and edx, 0xC0
    cmp edx, 0x80
    setne dl
    movzx edx, dl
    xor eax, eax
    ret

1048: # Four bytes: classify high bits without SIMD, then walk the scalar head.
    mov r9d, dword ptr [rdi + rax - 4]
    mov r10d, r9d
    and r9d, 0x80808080
    and r10d, 0x40404040
    shl r10d, 1
    mov r11d, 0x80808080
    andn r10d, r10d, r9d
    andn r9d, r10d, r11d
    popcnt r11d, r9d
    mov rcx, rdx
    sub rcx, r8
    cmp rcx, r11
    ja 1049f
    sub rax, 4
    cmp rcx, r11
    jne 1481f
    tzcnt r9d, r9d
    shr r9d, 3
    add rax, r9
    ret
1481:
    bsr r10d, r9d
    btr r9d, r10d
    dec rcx
    jnz 1481b
    shr r10d, 3
    add rax, r10
    ret
1049:
    add r8, r11
    sub rdx, r11
    sub rax, 4
    jz 1046f
    jmp 1044f

10402: # Small target-one suffix: stop at the first backward non-continuation.
    cmp rdx, 1
    jne 1044f
    cmp rax, 3
    jne 104021f
    movzx ecx, byte ptr [rdi + 2]
    and ecx, 0xC0
    cmp ecx, 0x80
    jne 104023f
    movzx ecx, byte ptr [rdi + 1]
    and ecx, 0xC0
    cmp ecx, 0x80
    jne 104024f
    movzx edx, byte ptr [rdi]
    and edx, 0xC0
    cmp edx, 0x80
    setne dl
    movzx edx, dl
    xor eax, eax
    ret
104023:
    mov eax, 2
    ret
104024:
    mov eax, 1
    ret
104021:
    dec rax
    movzx ecx, byte ptr [rdi + rax]
    and ecx, 0xC0
    cmp ecx, 0x80
    jne 104022f
    test rax, rax
    jnz 104021b
    xor edx, edx
104022:
    ret

1044: # One-to-three-byte scalar head; count down to avoid seen/cmp each byte.
    mov r9, rdx
10441:
    test rax, rax
    jz 10442f
    dec rax
    movzx r10d, byte ptr [rdi + rax]
    and r10d, 0xC0
    cmp r10d, 0x80
    je 10441b
    dec rdx
    jnz 10441b
10442:
    sub r9, rdx
    add r8, r9
1046:
    mov rdx, r8
    ret

1005:  # Wanted byte is in a final 32-byte block.
    sub rax, 32
    cmp rcx, r11
    jne 1051f
    tzcnt r10, r9
    add rax, r10
    vzeroupper
    ret
1051:
    mov r11, rax
    jmp 1071f
1006:  # Split the 64-bit mask first, bounding the bit walk to 32 steps.
    cmp rcx, r11
    jne 1063f
    sub rax, 64
    tzcnt r10, r9
    add rax, r10
    vzeroupper
    ret
1063:
    lea r11, [rax - 64]
    mov r10, r9
    shr r10, 32
    popcnt r10, r10
    cmp rcx, r10
    ja 1061f
    shr r9, 32
    add r11, 32
    jmp 1071f
1061:
    sub rcx, r10
    mov r9d, r9d
    jmp 1071f
1072: # Wanted byte is in the masked head at offset zero.
    cmp rcx, r11
    jne 1721f
    tzcnt rax, r9
    vzeroupper
    ret
1721:
    xor r11d, r11d
1071: # kth-highest set bit, k = target - seen; k is at least one.
    bsr r10, r9
    btr r9, r10
    dec rcx
    jnz 1071b
    lea rax, [r11 + r10]
    vzeroupper
    ret
1008:  # The whole prefix was examined without reaching target.
    xor eax, eax
1009:
    mov rdx, r8
    vzeroupper
    ret
1091: # Zero-length or target-zero entry: no AVX state was touched.
    mov rdx, r8
    ret

2002:

    test rdx, rdx
    jz 2009f
    mov r8, rdx
    cmp rsi, 5
    jb 2007f
    cmp rsi, 8
    jae 2070f
    cmp rdx, rsi
    jb 2007f
2070:
    mov eax, 0xC0
    vmovd xmm1, eax
    vpbroadcastb ymm1, xmm1
    mov eax, 0x80
    vmovd xmm2, eax
    vpbroadcastb ymm2, xmm2

    cmp rsi, 32
    jb 2030f
    cmp rsi, 128
    jb 2004f
.p2align 5
2002:
    xor ecx, ecx
    vmovdqu ymm0, ymmword ptr [rdi + rsi - 32]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb eax, ymm0
    not eax
    popcnt r9d, eax
    add ecx, r9d

    vmovdqu ymm0, ymmword ptr [rdi + rsi - 64]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb eax, ymm0
    not eax
    popcnt r9d, eax
    add ecx, r9d

    vmovdqu ymm0, ymmword ptr [rdi + rsi - 96]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb eax, ymm0
    not eax
    popcnt r9d, eax
    add ecx, r9d

    vmovdqu ymm0, ymmword ptr [rdi + rsi - 128]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb eax, ymm0
    not eax
    popcnt r9d, eax
    add ecx, r9d

    cmp rdx, rcx
    jbe 2004f
    sub rdx, rcx
    sub rsi, 128
    cmp rsi, 128
    jae 2002b
    jmp 2004f

2030:
    cmp rsi, 16
    jb 2031f
    # Two exact 16-byte loads cover 16..31 bytes without over-reading.
    mov rcx, rsi
    sub rcx, 16
    vmovdqu xmm0, xmmword ptr [rdi + rsi - 16]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb eax, xmm0
    not eax
    movzx eax, ax
    shl eax, cl
    mov r9d, eax
    vmovdqu xmm0, xmmword ptr [rdi]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb eax, xmm0
    not eax
    bzhi eax, eax, ecx
    or eax, r9d
    jmp 2033f
2031:
    cmp rsi, 8
    jb 2032f
    # Two exact 8-byte loads cover 8..15 bytes without over-reading.
    mov rcx, rsi
    sub rcx, 8
    vmovq xmm0, qword ptr [rdi + rsi - 8]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb eax, xmm0
    not eax
    movzx eax, al
    shl eax, cl
    mov r9d, eax
    vmovq xmm0, qword ptr [rdi]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb eax, xmm0
    not eax
    bzhi eax, eax, ecx
    or eax, r9d
    jmp 2033f
2032:
    # Two exact 4-byte loads cover 4..7 bytes without over-reading.
    mov rcx, rsi
    sub rcx, 4
    vmovd xmm0, dword ptr [rdi + rsi - 4]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb eax, xmm0
    not eax
    and eax, 15
    shl eax, cl
    mov r9d, eax
    vmovd xmm0, dword ptr [rdi]
    vpand xmm0, xmm0, xmm1
    vpcmpeqb xmm0, xmm0, xmm2
    vpmovmskb eax, xmm0
    not eax
    bzhi eax, eax, ecx
    or eax, r9d
2033:
    popcnt ecx, eax
    cmp rdx, rcx
    jbe 2005f
    sub rdx, rcx
    jmp 2008f

2004:
    cmp rsi, 32
    jb 2003f
    vmovdqu ymm0, ymmword ptr [rdi + rsi - 32]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb eax, ymm0
    not eax
    popcnt ecx, eax
    cmp rdx, rcx
    jbe 2005f
    sub rdx, rcx
    sub rsi, 32
    jmp 2004b
2003:
    test rsi, rsi
    jz 2008f
    vmovdqu ymm0, ymmword ptr [rdi]
    vpand ymm0, ymm0, ymm1
    vpcmpeqb ymm0, ymm0, ymm2
    vpmovmskb eax, ymm0
    not eax
    bzhi eax, eax, esi
    popcnt ecx, eax
    cmp rdx, rcx
    jbe 2005f
    sub rdx, rcx
    jmp 2008f
2005:
    xor ecx, ecx
    cmp rsi, 32
    jb 2051f
    lea rcx, [rsi - 32]
2051:
    mov r11, rcx
    xor r10d, r10d
    mov ecx, eax
    shr ecx, 16
    popcnt r9d, ecx
    cmp rdx, r9
    jbe 2053f
    sub rdx, r9
    movzx eax, ax
    jmp 2054f
2053:
    shr eax, 16
    add r10d, 16
2054:
    mov ecx, eax
    shr ecx, 8
    popcnt r9d, ecx
    cmp rdx, r9
    jbe 2055f
    sub rdx, r9
    movzx eax, al
    jmp 2056f
2055:
    shr eax, 8
    add r10d, 8
2056:
    mov ecx, eax
    shr ecx, 4
    popcnt r9d, ecx
    cmp rdx, r9
    jbe 2057f
    sub rdx, r9
    and eax, 15
    jmp 2058f
2057:
    shr eax, 4
    add r10d, 4
2058:
    mov ecx, eax
    shr ecx, 2
    popcnt r9d, ecx
    cmp rdx, r9
    jbe 2059f
    sub rdx, r9
    and eax, 3
    jmp 2060f
2059:
    shr eax, 2
    add r10d, 2
2060:
    mov ecx, eax
    shr ecx, 1
    popcnt r9d, ecx
    cmp rdx, r9
    jbe 2063f
    and eax, 1
    jmp 2064f
2063:
    shr eax, 1
    add r10d, 1
2064:
    tzcnt eax, eax
    add rax, r10
    add rax, r11
    mov rdx, r8
    vzeroupper
    ret
2007:
    test rsi, rsi
    jz 2072f
    dec rsi
    movzx eax, byte ptr [rdi + rsi]
    and eax, 0xC0
    cmp eax, 0x80
    je 2007b
    dec rdx
    jz 2071f
    jmp 2007b
2071:
    mov rax, rsi
    mov rdx, r8
    ret
2072:
    xor eax, eax
    sub r8, rdx
    mov rdx, r8
    ret
2008:
    xor eax, eax
    sub r8, rdx
    mov rdx, r8
    vzeroupper
    ret
2009:
    mov rax, rsi
    xor edx, edx
    ret
