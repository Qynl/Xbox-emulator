/* Shift and rotate group, immediate and CL counts, arithmetic vs logical. */
.intel_syntax noprefix
.text
.globl start
start:
    mov rax, 1
    shl rax, 5          /* 32                         */
    mov rcx, 3
    shl rax, cl         /* 256                        */
    sar rax, 2          /* 64                         */
    mov rbx, -8
    sar rbx, 1          /* -4                         */
    shr rbx, 2          /* 0x3FFFFFFFFFFFFFFF         */
    mov rdx, 0xF0
    mov rcx, 4
    shr rdx, cl         /* 0x0F                       */
    mov r9, 0x8000000000000000
    shr r9, 63          /* 1                          */
    mov rdi, 5
    rol rdi, 1          /* 10                         */
    int3
