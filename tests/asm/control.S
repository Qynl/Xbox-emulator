/* Control flow: a counted loop, call/ret and conditional branching. */
.intel_syntax noprefix
.text
.globl start
start:
    xor eax, eax
    mov ecx, 100
1:
    add rax, rcx            /* sum 1..100 -> 5050     */
    dec rcx
    jne 1b
    mov rdi, 1000
    call double_it          /* 2000                   */
    cmp rdi, 2000
    je 2f
    mov r12, 1              /* failure marker         */
2:
    xor r13d, r13d
    int3
double_it:
    add rdi, rdi
    ret
