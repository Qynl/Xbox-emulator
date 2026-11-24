/* Entry stub for the milestone-2 compiler gate.
 *
 * The image is loaded at 0x1000 with RSP pointing at the top of the test
 * window. The System V ABI wants RSP 16-byte aligned at a call site, so align
 * it here and then call the gcc-compiled body, which is what a real crt0 does.
 * `int3` is the sentinel the harnesses look for.
 *
 * `.text.entry` comes first in tests/c/gate.ld, so `_start` really is at 0x1000
 * regardless of how gcc ordered the C functions.
 */
    .section .text.entry, "ax", @progbits
    .globl _start
    .type _start, @function
_start:
    andq $-16, %rsp
    call gate_main
    int3
    .size _start, .-_start
