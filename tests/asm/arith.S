/* Arithmetic, width handling and flag-setting on the path to int3. */
.intel_syntax noprefix
.text
.globl start
start:
    mov rax, -1
    add rax, 1              /* 0, CF=1, ZF=1          */
    mov rbx, 0x7F
    add bl, 1               /* 0x80, OF=1, SF=1       */
    mov rcx, 0x1000
    sub rcx, 1              /* 0xFFF                  */
    imul rdx, rcx, 3        /* 0x2FFD                 */
    xor rsi, rsi
    inc rsi                 /* 1                      */
    dec rsi                 /* 0                      */
    or rdi, 0x42            /* 0x42                   */
    mov r8d, 0xDEADBEEF
    movzx r9, r8w           /* 0xBEEF                 */
    movsx r10, r8b          /* sign-extend 0xEF       */
    neg r11                 /* 0                      */
    not r11                 /* all ones               */
    test r11, r11           /* SF=1, ZF=0             */
    int3
