/* Milestone 2 gate: real gcc -O2 code, run on our core and on the host.
 *
 * The same source is compiled twice:
 *
 *   gcc -O2 -DHOST_TEST ...        -> host binary, prints the answers
 *   gcc -O2 -ffreestanding ...     -> x86-64 image our interpreter executes
 *
 * Both must produce identical results. The emulated build writes its answer
 * block to the fixed address 0x9000, inside the harness's memory window, and
 * ends on `int3`; the host build prints the same fields as hex so the Rust and
 * Unicorn harnesses can compare them exactly.
 *
 * Nothing here calls libc: `memcpy`/`memset` are ours, written with the string
 * instructions the core implements, so the gate also covers `rep movsb`,
 * `rep stosb`, RIP-relative .rodata access and the SSE2 path gcc emits for
 * every double computation on x86-64.
 */

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long u64;
typedef long i64;
typedef double f64;

#define GATE_OUT_ADDR 0x9000UL

/* All fields are u64 so the struct layout is exactly an array of 64-bit words;
 * floats are compared through their bit patterns, which is exact. */
struct gate_out {
    u64 magic0;
    u64 sum;
    u64 fib;
    u64 mix;
    u64 copied0;
    u64 copied1;
    u64 dots0;
    u64 dots1;
    u64 fsum_bits;
    u64 fpoly_bits;
    u64 divmod;
    u64 lea_value;
    u64 cmov_value;
    u64 byte_sum;
    u64 magic1;
};

void *memcpy(void *dst, const void *src, u64 n) {
    void *d = dst;
    void *s = (void *)src;
    u64 count = n;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(count) : : "memory");
    return d;
}

void *memset(void *dst, int c, u64 n) {
    void *d = dst;
    u64 value = (u64)(u8)c;
    u64 count = n;
    __asm__ volatile("rep stosb" : "+D"(d), "+c"(count) : "a"(value) : "memory");
    return d;
}

void gate_compute(struct gate_out *o) {
    __asm__ volatile("cld" ::: "cc");

    /* 64-bit integer loop: imul/lea/add/compare/branch. */
    u64 sum = 0;
    for (u64 i = 1; i <= 1000; ++i) {
        sum += i * 3u;
    }

    /* Fibonacci: register pressure, add/adc-free 64-bit wrap. */
    u64 a = 0;
    u64 b = 1;
    for (u32 i = 0; i < 90; ++i) {
        u64 t = a + b;
        a = b;
        b = t;
    }

    /* Shift/multiply mixing, the shape of every hash. */
    u64 mix = 0x0123456789abcdefUL;
    mix ^= mix >> 33;
    mix *= 0xff51afd7ed558ccdUL;
    mix ^= mix >> 29;
    mix = (mix << 7) | (mix >> 57);

    /* rep movsb / rep stosb through the helpers above. */
    static const u8 table[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    u8 buf[32];
    memcpy(buf, table, 16);
    u64 byte_sum = 0;
    for (u32 i = 0; i < 16; ++i) {
        byte_sum += buf[i];
    }
    u64 copied0 = 0;
    u64 copied1 = 0;
    memcpy(&copied0, buf, 8);
    memcpy(&copied1, buf + 8, 8);

    memset(buf, 0xa5, 32);
    u64 dots0 = 0;
    u64 dots1 = 0;
    memcpy(&dots0, buf, 8);
    memcpy(&dots1, buf + 8, 8);

    /* Scalar double math: addsd/mulsd/cvtsi2sd plus stack spills. */
    f64 acc = 0.0;
    for (u32 i = 0; i < 100; ++i) {
        acc += (f64)i * 0.5;
    }

    f64 poly = 1.0;
    for (u32 i = 0; i < 8; ++i) {
        poly = poly * 1.5 + 0.25;
    }

    u64 fsum_bits = 0;
    u64 fpoly_bits = 0;
    memcpy(&fsum_bits, &acc, 8);
    memcpy(&fpoly_bits, &poly, 8);

    /* div/idiv. */
    u64 numerator = 1234567890123UL;
    u64 divmod = (numerator / 7919) ^ (numerator % 7919);

    /* Stack array + scaled indexing (SIB). */
    u64 arr[4] = {10, 20, 30, 40};
    u64 lea_value = arr[2] + arr[3] * 2;

    /* cmovcc. */
    i64 left = -17;
    i64 right = 42;
    u64 cmov_value = (u64)(left > right ? left : right);

    o->magic0 = 0x584f4e45324f4bUL; /* "XONE2OK" */
    o->sum = sum;
    o->fib = a;
    o->mix = mix;
    o->copied0 = copied0;
    o->copied1 = copied1;
    o->dots0 = dots0;
    o->dots1 = dots1;
    o->fsum_bits = fsum_bits;
    o->fpoly_bits = fpoly_bits;
    o->divmod = divmod;
    o->lea_value = lea_value;
    o->cmov_value = cmov_value;
    o->byte_sum = byte_sum;
    o->magic1 = 0xfeedfacecafebeefUL;
}

void gate_main(void) {
    gate_compute((struct gate_out *)GATE_OUT_ADDR);
    __asm__ volatile("int3");
}

#ifdef HOST_TEST
#include <stdio.h>

int main(void) {
    struct gate_out o;
    gate_compute(&o);
    printf("%016llx\n", (unsigned long long)o.magic0);
    printf("%016llx\n", (unsigned long long)o.sum);
    printf("%016llx\n", (unsigned long long)o.fib);
    printf("%016llx\n", (unsigned long long)o.mix);
    printf("%016llx\n", (unsigned long long)o.copied0);
    printf("%016llx\n", (unsigned long long)o.copied1);
    printf("%016llx\n", (unsigned long long)o.dots0);
    printf("%016llx\n", (unsigned long long)o.dots1);
    printf("%016llx\n", (unsigned long long)o.fsum_bits);
    printf("%016llx\n", (unsigned long long)o.fpoly_bits);
    printf("%016llx\n", (unsigned long long)o.divmod);
    printf("%016llx\n", (unsigned long long)o.lea_value);
    printf("%016llx\n", (unsigned long long)o.cmov_value);
    printf("%016llx\n", (unsigned long long)o.byte_sum);
    printf("%016llx\n", (unsigned long long)o.magic1);
    return 0;
}
#endif
