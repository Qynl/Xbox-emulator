/* Effective addresses: RIP-relative LEA, scaled SIB, disp8, sized stores. */
.intel_syntax noprefix
.text
.globl start
start:
    jmp 1f
    .align 8
table:
    .quad 10, 20, 30, 40
1:
    lea rbx, [rip + table]        /* rbx = &table           */
    mov rax, 3
    mov rdx, [rbx + rax*8]        /* table[3] = 40          */
    mov [rbx + 8], rdx            /* table[1] = 40          */
    mov r8, [rbx + 8]             /* 40                     */
    mov dword ptr [rbx + 16], 0x1234
    mov r9d, dword ptr [rbx + 16] /* 0x1234                 */
    mov rcx, 2
    lea rsi, [rbx + rcx*8 + 8]    /* &table[3]              */
    mov r10, [rsi]                /* 40                     */
    int3
