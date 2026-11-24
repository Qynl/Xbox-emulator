/* SSE2 fixture: instructions gcc emits for ordinary double and integer code.
 *
 * Everything the fixture relies on lives in .text (the fixture builder extracts
 * only that section), and the final state is compared against Unicorn by
 * tools/diff_unicorn.py.
 */
    .text
    .globl _start
_start:
    /* ---- packed integer: paddd / psubd / pxor / pshufd / movdqu */
    movdqa  va(%rip), %xmm0
    movdqa  vb(%rip), %xmm1
    paddd   %xmm1, %xmm0
    psubd   %xmm1, %xmm0
    pxor    %xmm2, %xmm2
    pxor    %xmm0, %xmm2
    pshufd  $0x1b, %xmm2, %xmm3
    movdqu  %xmm3, dst(%rip)
    movdqu  dst(%rip), %xmm4

    /* ---- packed 64-bit lanes: paddq, then byte shifts across the register */
    movdqa  %xmm4, %xmm5
    paddq   %xmm5, %xmm5
    psrldq  $4, %xmm5
    pslldq  $4, %xmm5

    /* ---- compares and logic on packed integers */
    pcmpeqd %xmm1, %xmm6
    movdqa  %xmm6, %xmm7
    pand    %xmm1, %xmm7
    por     %xmm7, %xmm6
    punpckldq %xmm4, %xmm6

    /* ---- scalar double arithmetic */
    mov     $7, %eax
    cvtsi2sd %eax, %xmm8        /* 7.0 */
    movsd   fa(%rip), %xmm9     /* 1.5 */
    addsd   fa(%rip), %xmm9     /* 3.0 */
    mulsd   fb(%rip), %xmm9     /* 7.5 */
    cvttsd2si %xmm9, %edx       /* 7 */
    comisd  fa(%rip), %xmm9     /* 1.5 < 7.5 -> CF=1 */
    sqrtsd  fc(%rip), %xmm10    /* sqrt(2.25) = 1.5 */

    /* ---- movd / movq transfer between GPRs and XMM */
    movd    %eax, %xmm11        /* 7 */
    movq    %xmm11, %rax
    movd    %xmm11, %ecx
    movsd   %xmm9, dbl(%rip)    /* store a double through memory */

    /* ---- keep the vector results in memory so the hash covers them */
    movups  %xmm0, vout(%rip)
    movups  %xmm5, vout+16(%rip)
    movups  %xmm6, vout+32(%rip)
    movups  %xmm10, vout+48(%rip)

    int3

    .align 16
va:     .long 1, 2, 3, 4
vb:     .long 10, 20, 30, 40
fa:     .double 1.5
fb:     .double 2.5
fc:     .double 2.25
        .align 16
dst:    .zero 16
dbl:    .zero 8
vout:   .zero 64
