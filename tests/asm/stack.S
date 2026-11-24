/* Stack discipline: push/pop of registers and immediates, nested call/ret. */
.intel_syntax noprefix
.text
.globl start
start:
    mov rax, 0x1111
    push rax
    xor eax, eax
    pop rax                 /* 0x1111                  */
    push 0x2222
    pop rbx                 /* 0x2222                  */
    mov rcx, 5
    push rcx
    pop rdx                 /* 5                       */
    mov rdi, 7
    call triple             /* 21                      */
    int3
triple:
    lea rdi, [rdi + rdi*2]
    ret
