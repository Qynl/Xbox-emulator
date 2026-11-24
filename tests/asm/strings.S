/* String instructions: movs/stos/lods/cmps/scas with and without REP, and the
 * DF direction flag. Compared against Unicorn by tools/diff_unicorn.py.
 *
 * The data sits at image offset 0x1000, which the harness loads at 0x1000, so
 * src is at 0x2000, dst at 0x2010 and wide at 0x2038 in the memory window.
 */
    .text
    .globl _start
_start:
    /* forward rep movsb: dst <- src */
    cld
    lea     src(%rip), %rsi
    lea     dst(%rip), %rdi
    mov     $16, %rcx
    rep movsb

    /* rep stosb: fill dst+16 with 0xaa */
    lea     dst+16(%rip), %rdi
    mov     $8, %rcx
    mov     $0xaa, %al
    rep stosb

    /* rep lodsb: AL ends up holding src[3] */
    lea     src(%rip), %rsi
    mov     $4, %rcx
    rep lodsb

    /* repne scasb: AL = 4, dst[0] = 1, so it stops after one iteration */
    lea     dst(%rip), %rdi
    mov     $16, %rcx
    repne scasb

    /* repe cmpsb over identical blocks: runs to completion, ZF = 1 */
    lea     src(%rip), %rsi
    lea     dst(%rip), %rdi
    mov     $16, %rcx
    repe cmpsb

    /* backwards rep movsb: both pointers walk down, so the block keeps its
     * order but lands at dst+16..dst+32. r8/r9 record where the pointers ended
     * up, which is what actually proves DF was honoured. */
    std
    lea     src+15(%rip), %rsi
    lea     dst+31(%rip), %rdi
    mov     $16, %rcx
    rep movsb
    mov     %rsi, %r8
    mov     %rdi, %r9
    cld

    /* single (non-REP) 8-byte move to prove the wide form works */
    lea     src(%rip), %rsi
    lea     wide(%rip), %rdi
    movsq

    int3

    .org 0x1000
        .align 16
src:    .byte 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
dst:    .zero 40
wide:   .zero 8
