/* Native hardware oracle for the SSE floating-point rules.
 *
 * Emits the exact instruction bytes under test into executable memory and runs
 * them on the real CPU, so the emulator can be checked against the hardware
 * instead of against another emulator. Reads "form op a_hex b_hex" lines on
 * stdin and prints the 128-bit result of `a <op> b` for each one.
 *
 *   form  ss | sd | ps | pd
 *   op    two hex digits, the second opcode byte (58 add, 59 mul, 5C sub,
 *         5E div, 5D min, 5F max)
 *
 * The parsed operands are copied into zero-padded 16-byte buffers, so a full
 * 128-bit comparison is meaningful for the scalar forms too: only the low lane
 * is written by them, exactly as on the emulator side.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef void (*blob_fn)(const void *a, const void *b, void *out);

static blob_fn build(const char *form, unsigned op)
{
    unsigned char *code = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (code == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    int n = 0;
    /* movups (%rdi), %xmm0 */
    code[n++] = 0x0F; code[n++] = 0x10; code[n++] = 0x07;
    /* movups (%rsi), %xmm1 */
    code[n++] = 0x0F; code[n++] = 0x10; code[n++] = 0x0E;
    unsigned pfx = 0;
    if (!strcmp(form, "ss")) {
        pfx = 0xF3;
    } else if (!strcmp(form, "sd")) {
        pfx = 0xF2;
    } else if (!strcmp(form, "pd")) {
        pfx = 0x66;
    } else if (strcmp(form, "ps")) {
        fprintf(stderr, "nan_rule: unknown form %s\n", form);
        exit(2);
    }
    if (pfx) {
        code[n++] = (unsigned char)pfx;
    }
    /* xmm0 = xmm0 <op> xmm1 */
    code[n++] = 0x0F; code[n++] = (unsigned char)op; code[n++] = 0xC1;
    /* movups %xmm0, (%rdx) */
    code[n++] = 0x0F; code[n++] = 0x11; code[n++] = 0x02;
    code[n++] = 0xC3; /* ret */
    return (blob_fn)code;
}

int main(void)
{
    char form[8];
    unsigned op;
    unsigned long long a_lo, b_lo;
    while (scanf("%7s %x %llx %llx", form, &op, &a_lo, &b_lo) == 4) {
        blob_fn fn = build(form, op);
        uint64_t a[2] __attribute__((aligned(16))) = { a_lo, 0 };
        uint64_t b[2] __attribute__((aligned(16))) = { b_lo, 0 };
        uint64_t out[2] __attribute__((aligned(16))) = { 0, 0 };
        fn(a, b, out);
        printf("%016llx%016llx\n", (unsigned long long)out[1],
               (unsigned long long)out[0]);
    }
    return 0;
}
