/* Milestone 2 gate, part two: gcc -O3 auto-vectorised code.
 *
 * The same source is compiled twice, exactly like m2_gate.c:
 *
 *   gcc -O3 -DHOST_TEST ...        -> host binary, prints the answers
 *   gcc -O3 -ffreestanding ...     -> x86-64 image our interpreter executes
 *
 * Plain C, no intrinsics: every packed-integer instruction here is whatever
 * gcc's vectoriser chose to emit at -O3 (byte/word adds and multiplies, SAD,
 * min/max reductions, shifts, interleaves, equality masks, and float adds and
 * multiplies on both widths). That is the difference between "our packed ops
 * pass a fuzzer we wrote" and "the packed ops a real compiler emits actually
 * run". No -march flag is used, so the baseline is SSE2 — the widest thing the
 * core claims in `cpuid`.
 *
 * The emulated build writes its answer block to 0x9000 and ends on `int3`; the
 * host build prints the same fields as hex, so the Rust and Unicorn harnesses
 * compare them exactly.
 */

typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef short i16;
typedef unsigned int u32;
typedef signed int i32;
typedef unsigned long u64;
typedef long i64;
typedef float f32;
typedef double f64;

#define SIMD_OUT_ADDR 0x9000UL
#define LEN 64

struct simd_out {
    u64 magic0;
    u64 byte_sum;
    u64 sad;
    u64 min_u8;
    u64 max_u8;
    u64 min_i16;
    u64 max_i16;
    u64 dot_i16;
    u64 mul_i16;
    u64 shl_i16;
    u64 sar_i16;
    u64 interleave0;
    u64 interleave1;
    u64 eq_count;
    u64 add_u16;
    u64 sub_u16;
    u64 f32_sum_bits;
    u64 f32_mul_bits;
    u64 f64_sum_bits;
    u64 f64_mul_bits;
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

/* Static so the image is self-contained and nothing lands on the stack. */
static u8 bytes_a[LEN];
static u8 bytes_b[LEN];
static i16 words_a[LEN];
static i16 words_b[LEN];
static i16 words_out[LEN];
static i16 interleaved[2 * LEN];
static f32 floats[LEN];
static f64 doubles[LEN];

static u64 bits_of_f32(f32 v) {
    u32 raw = 0;
    memcpy(&raw, &v, 4);
    return raw;
}

static u64 bits_of_f64(f64 v) {
    u64 raw = 0;
    memcpy(&raw, &v, 8);
    return raw;
}

void gate_compute(struct simd_out *o) {
    __asm__ volatile("cld" ::: "cc");

    for (u32 i = 0; i < LEN; ++i) {
        bytes_a[i] = (u8)(i * 37 + 11);
        bytes_b[i] = (u8)(i * 13 + 7);
        words_a[i] = (i16)(i * 517 - 9000);
        words_b[i] = (i16)(3000 - i * 191);
        floats[i] = (f32)i * 0.25f - 3.0f;
        doubles[i] = (f64)i * 0.5 - 7.0;
    }

    /* Byte sum: paddb into a wide accumulator / psadbw. */
    u64 byte_sum = 0;
    for (u32 i = 0; i < LEN; ++i) {
        byte_sum += bytes_a[i];
    }

    /* Sum of absolute differences, the psadbw shape. */
    u64 sad = 0;
    for (u32 i = 0; i < LEN; ++i) {
        i32 d = (i32)bytes_a[i] - (i32)bytes_b[i];
        sad += (u64)(d < 0 ? -d : d);
    }

    /* Unsigned byte reductions: pminub / pmaxub. */
    u8 min_u8 = 0xff;
    u8 max_u8 = 0;
    for (u32 i = 0; i < LEN; ++i) {
        if (bytes_a[i] < min_u8) {
            min_u8 = bytes_a[i];
        }
        if (bytes_a[i] > max_u8) {
            max_u8 = bytes_a[i];
        }
    }

    /* Signed word reductions: pminsw / pmaxsw. */
    i16 min_i16 = 0x7fff;
    i16 max_i16 = -0x8000;
    for (u32 i = 0; i < LEN; ++i) {
        if (words_a[i] < min_i16) {
            min_i16 = words_a[i];
        }
        if (words_a[i] > max_i16) {
            max_i16 = words_a[i];
        }
    }

    /* 16-bit dot product accumulated in 32 bits: pmullw + pmaddwd + paddd. */
    i32 dot = 0;
    for (u32 i = 0; i < LEN; ++i) {
        dot += (i32)words_a[i] * (i32)words_b[i];
    }

    /* Wrapping 16-bit multiply, shifts in both directions. */
    for (u32 i = 0; i < LEN; ++i) {
        words_out[i] = (i16)(words_a[i] * 3);
    }
    u64 mul_i16 = 0;
    for (u32 i = 0; i < LEN; ++i) {
        mul_i16 = mul_i16 * 31 + (u16)words_out[i];
    }

    for (u32 i = 0; i < LEN; ++i) {
        words_out[i] = (i16)(words_a[i] << 3);
    }
    u64 shl_i16 = 0;
    for (u32 i = 0; i < LEN; ++i) {
        shl_i16 = shl_i16 * 33 + (u16)words_out[i];
    }

    for (u32 i = 0; i < LEN; ++i) {
        words_out[i] = (i16)(words_a[i] >> 2);
    }
    u64 sar_i16 = 0;
    for (u32 i = 0; i < LEN; ++i) {
        sar_i16 = sar_i16 * 35 + (u16)words_out[i];
    }

    /* Interleave: punpcklwd / punpckhwd. */
    for (u32 i = 0; i < LEN; ++i) {
        interleaved[2 * i] = words_a[i];
        interleaved[2 * i + 1] = words_b[i];
    }
    u64 interleave0 = 0;
    u64 interleave1 = 0;
    memcpy(&interleave0, interleaved, 8);
    memcpy(&interleave1, interleaved + LEN, 8);

    /* Byte equality count: pcmpeqb + pmovmskb. */
    u32 eq_count = 0;
    for (u32 i = 0; i < LEN; ++i) {
        eq_count += bytes_a[i] == bytes_b[i];
    }

    /* Wrapping 16-bit add and subtract: paddw / psubw. */
    u64 add_u16 = 0;
    for (u32 i = 0; i < LEN; ++i) {
        words_out[i] = (i16)((u16)words_a[i] + (u16)words_b[i]);
    }
    for (u32 i = 0; i < LEN; ++i) {
        add_u16 = add_u16 * 37 + (u16)words_out[i];
    }
    u64 sub_u16 = 0;
    for (u32 i = 0; i < LEN; ++i) {
        words_out[i] = (i16)((u16)words_a[i] - (u16)words_b[i]);
    }
    for (u32 i = 0; i < LEN; ++i) {
        sub_u16 = sub_u16 * 39 + (u16)words_out[i];
    }

    /* Float and double reductions: addps/mulps and addpd/mulpd. */
    f32 fsum = 0.0f;
    for (u32 i = 0; i < LEN; ++i) {
        fsum += floats[i];
    }
    f32 fmul = 1.0f;
    for (u32 i = 0; i < LEN; ++i) {
        fmul *= floats[i] * 0.5f + 1.0f;
    }
    f64 dsum = 0.0;
    for (u32 i = 0; i < LEN; ++i) {
        dsum += doubles[i];
    }
    f64 dmul = 1.0;
    for (u32 i = 0; i < LEN; ++i) {
        dmul *= doubles[i] * 0.25 + 1.5;
    }

    o->magic0 = 0x584f4e45334f4bUL; /* "XONE3OK" */
    o->byte_sum = byte_sum;
    o->sad = sad;
    o->min_u8 = min_u8;
    o->max_u8 = max_u8;
    o->min_i16 = (u16)min_i16;
    o->max_i16 = (u16)max_i16;
    o->dot_i16 = (u32)dot;
    o->mul_i16 = mul_i16;
    o->shl_i16 = shl_i16;
    o->sar_i16 = sar_i16;
    o->interleave0 = interleave0;
    o->interleave1 = interleave1;
    o->eq_count = eq_count;
    o->add_u16 = add_u16;
    o->sub_u16 = sub_u16;
    o->f32_sum_bits = bits_of_f32(fsum);
    o->f32_mul_bits = bits_of_f32(fmul);
    o->f64_sum_bits = bits_of_f64(dsum);
    o->f64_mul_bits = bits_of_f64(dmul);
    o->magic1 = 0x0ddc0ffee5badf00UL;
}

void gate_main(void) {
    gate_compute((struct simd_out *)SIMD_OUT_ADDR);
    __asm__ volatile("int3");
}

#ifdef HOST_TEST
#include <stdio.h>

/* One hex value per line, the format the Rust and Unicorn harnesses parse. */
static void show(u64 value) {
    printf("%016llx\n", (unsigned long long)value);
}

int main(void) {
    struct simd_out o;
    gate_compute(&o);
    show(o.magic0);
    show(o.byte_sum);
    show(o.sad);
    show(o.min_u8);
    show(o.max_u8);
    show(o.min_i16);
    show(o.max_i16);
    show(o.dot_i16);
    show(o.mul_i16);
    show(o.shl_i16);
    show(o.sar_i16);
    show(o.interleave0);
    show(o.interleave1);
    show(o.eq_count);
    show(o.add_u16);
    show(o.sub_u16);
    show(o.f32_sum_bits);
    show(o.f32_mul_bits);
    show(o.f64_sum_bits);
    show(o.f64_mul_bits);
    show(o.magic1);
    return 0;
}
#endif
