/* Native hardware oracle for the shift/rotate flag rules.
 *
 * Intel defines OF only for a count of one, and RCL/RCR for a count of one as
 * well, but real silicon keeps computing a formula for every count -- and so
 * does the reference used for differential testing, except for SHLD/SHRD.
 * Rather than guess, this program emits the exact instruction bytes under test
 * into executable memory, seeds the incoming flags itself, runs them on the
 * real CPU, and prints both the bytes and the resulting flags/register. The
 * emulator then runs the same bytes, so nothing about the encoding is
 * duplicated on the comparison side.
 *
 * Reads "op width value fill count flags" lines on stdin, where
 *   op     rol ror rcl rcr shl shr sar shld shrd
 *   width  16 32 64
 *   count  the count placed in CL (the instruction's own mask applies)
 *   flags  the RFLAGS value loaded before the instruction
 * and prints "<program hex> <flags hex> <result hex>" per line. The result is
 * read from RBX, whose upper bits are whatever the width left untouched.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef void (*blob_fn)(void);

static unsigned char *emit(unsigned char *p, unsigned long long value)
{
    memcpy(p, &value, 8);
    return p + 8;
}

/* Group-2 shift digits and the two shld/shrd opcodes. */
static int digit_of(const char *op)
{
    if (!strcmp(op, "rol")) return 0;
    if (!strcmp(op, "ror")) return 1;
    if (!strcmp(op, "rcl")) return 2;
    if (!strcmp(op, "rcr")) return 3;
    if (!strcmp(op, "shl")) return 4;
    if (!strcmp(op, "shr")) return 5;
    if (!strcmp(op, "sar")) return 7;
    return -1; /* shld / shrd */
}

static unsigned char *build(unsigned char *code, const char *op, unsigned width,
                            unsigned long long value, unsigned long long fill,
                            unsigned count, unsigned flags_in)
{
    unsigned char *p = code;
    *p++ = 0x68;                       /* push imm32 */
    uint32_t flags32 = (uint32_t)flags_in;
    memcpy(p, &flags32, 4); p += 4;
    *p++ = 0x9d;                       /* popfq */
    *p++ = 0x48; *p++ = 0xbb; p = emit(p, value);   /* mov rbx, imm64 */
    *p++ = 0x48; *p++ = 0xbe; p = emit(p, fill);    /* mov rsi, imm64 */
    *p++ = 0xb1; *p++ = (unsigned char)count;       /* mov cl, imm8 */

    if (width == 16) *p++ = 0x66;
    if (width == 64) *p++ = 0x48;

    int digit = digit_of(op);
    if (digit >= 0) {
        *p++ = 0xd3;                                 /* op r/m, cl */
        *p++ = (unsigned char)(0xc0 | (digit << 3) | 3);   /* rbx */
    } else {
        *p++ = 0x0f;
        *p++ = (unsigned char)(!strcmp(op, "shld") ? 0xa5 : 0xad);
        *p++ = (unsigned char)(0xc0 | (6 << 3) | 3); /* rsi, rbx */
    }

    *p++ = 0x9c;                       /* pushfq */
    *p++ = 0x58;                       /* pop rax */
    *p++ = 0x48; *p++ = 0x89; *p++ = 0xda;  /* mov rdx, rbx */
    *p++ = 0xc3;                       /* ret */
    return p;
}

int main(void)
{
    unsigned char *code = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (code == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    char op[8];
    unsigned width, count, flags_in;
    unsigned long long value, fill;
    while (scanf("%7s %u %llx %llx %u %x", op, &width, &value, &fill, &count,
                 &flags_in) == 6) {
        unsigned char *end = build(code, op, width, value, fill, count, flags_in);
        uint64_t flags, result;
        __asm__ volatile("call *%2\n\tmovq %%rax, %0\n\tmovq %%rdx, %1"
                         : "=r"(flags), "=r"(result)
                         : "r"((blob_fn)code)
                         : "rax", "rdx", "rbx", "rsi", "rcx", "memory", "cc");
        for (unsigned char *p = code; p < end; p++) {
            printf("%02x", *p);
        }
        printf(" %03x %016llx\n", (unsigned)(flags & 0x8c5),
               (unsigned long long)result);
    }
    return 0;
}
