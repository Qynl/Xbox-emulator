#!/usr/bin/env python3
"""Randomised differential fuzzer: our x86-64 core vs Unicorn.

Each case builds a short straight-line program mixing three families:

  * SSE/SSE2: moves, packed and scalar arithmetic, sqrt/min/max, compares,
    shuffles and unpacks, packed shifts including the byte shifts
    pslldq/psrldq, conversions, the lane extract pextrw, and movd/movq
    transfers that touch the general-purpose registers (and therefore the
    flags via comis*);
  * integer ALU/shift/inc-dec/setcc, so flags and GPRs interact with the
    vector state;
  * string instructions with REP/REPE/REPNE under both setting of DF.

State is seeded on both sides: all sixteen GPRs and all sixteen XMM registers
get values from float and integer edge-case pools (signed zeros, infinities,
quiet and signalling NaNs, denormals, large integers, random bits), and a
512-byte arena of mixed data is loaded at 0x4000 so loads and stores hit real
content at varied alignments. Memory operands use only registers pinned to that
arena, so every generated address is in range and Unicorn never faults on
address translation -- a case that does fault is reported, not silently
ignored.

GPRs, XMM registers, the arithmetic flags and the exact bytes of the low memory
window must all match Unicorn.

One documented exception exists: a 0x66 prefix on `xchg rAX, r` means a 16-bit
exchange on hardware, while QEMU (and therefore Unicorn) ignores the prefix.
Cases that contain exactly that form are skipped only when they diverge.

Usage:
    .venv/bin/python tools/fuzz_unicorn.py [--iterations N] [--seeds 1 2 3]
"""

import argparse
import os
import random
import re
import subprocess
import sys

try:
    import unicorn.x86_const as ux
    from unicorn import Uc, UcError, UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64
except ImportError:
    sys.exit(
        "unicorn is not installed.\n"
        "  python3 -m venv .venv && .venv/bin/pip install unicorn capstone"
    )

GPR_NAMES = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
]
REG_IDS = {name: getattr(ux, "UC_X86_REG_" + name.upper()) for name in GPR_NAMES}
XMM_IDS = [getattr(ux, f"UC_X86_REG_XMM{i}") for i in range(16)]

# CF|PF|ZF|SF|DF|OF. AF is architecturally undefined after logical ops.
FLAG_MASK = 0x8C5
MEM_BASE = 0
MEM_LEN = 0x10000
MAP_LEN = 0x6000  # what we compare: code, arena and the stack window
CODE = 0x1000
STACK = 0x8000
ARENA = 0x4000
ARENA_LEN = 0x200
STEP_BUDGET = 256
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GPRs the generator may write: nothing that carries an address, the stack
# pointer, or the REP/shift count.
GPRS = [0, 2, 5, 8, 9, 10, 11, 12]  # rax rdx rbp r8 r9 r10 r11 r12

EDGE_VALUES = [
    0,
    1,
    2,
    7,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0xFFFF,
    0x8000,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    0x8000_0000_0000_0000,
    0xFFFF_FFFF_FFFF_FFFF,
    0x0123_4567_89AB_CDEF,
]

# 32-bit float bit patterns: zeros, one, fractions, infinities, NaNs
# (quiet and signalling), denormals, and integers that stress conversion.
F32_POOL = [
    0x0000_0000,  # +0.0
    0x8000_0000,  # -0.0
    0x3F80_0000,  # 1.0
    0xBF80_0000,  # -1.0
    0x3F00_0000,  # 0.5
    0xBF00_0000,  # -0.5
    0x4000_0000,  # 2.0
    0x4040_0000,  # 3.0
    0x40A0_0000,  # 5.0
    0x7F80_0000,  # +inf
    0xFF80_0000,  # -inf
    0x7FC0_0000,  # qNaN
    0xFFC0_0000,  # -qNaN
    0x7F80_0001,  # sNaN
    0xFF80_0001,  # -sNaN
    0x0000_0001,  # smallest denormal
    0x007F_FFFF,  # largest denormal
    0x0080_0000,  # smallest normal
    0x7F7F_FFFF,  # largest finite
    0x4B00_0000,  # 2^23
    0x4F00_0000,  # 2^31
    0xCF00_0000,  # -2^31
    0x5F00_0000,  # 2^63
]

# 64-bit double bit patterns, same idea.
F64_POOL = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3FF0_0000_0000_0000,  # 1.0
    0xBFF0_0000_0000_0000,  # -1.0
    0x3FE0_0000_0000_0000,  # 0.5
    0x4000_0000_0000_0000,  # 2.0
    0x4008_0000_0000_0000,  # 3.0
    0x7FF0_0000_0000_0000,  # +inf
    0xFFF0_0000_0000_0000,  # -inf
    0x7FF8_0000_0000_0000,  # qNaN
    0xFFF8_0000_0000_0000,  # -qNaN
    0x7FF0_0000_0000_0001,  # sNaN
    0x0000_0000_0000_0001,  # smallest denormal
    0x000F_FFFF_FFFF_FFFF,  # largest denormal
    0x7FEF_FFFF_FFFF_FFFF,  # largest finite
    0x4330_0000_0000_0000,  # 2^53
    0x43E0_0000_0000_0000,  # 2^63
    0xC3E0_0000_0000_0000,  # -2^63
]


# --------------------------------------------------------------- encoding

def _rex(w=0, r=0, x=0, b=0):
    if not (w or r or x or b):
        return b""
    return bytes([0x40 | (w << 3) | (r << 2) | (x << 1) | b])


def _modrm(mod, reg, rm):
    return bytes([(mod << 6) | ((reg & 7) << 3) | (rm & 7)])


# Base registers pinned to the data arena, and their (r/m field, REX.B) pairs.
BASE_ENC = {"rbx": (3, 0), "r14": (6, 1), "r15": (7, 1)}


def _mem(rng, reg_field, aligned=False):
    """Encode a memory operand whose address always lands in the arena."""
    if aligned:
        # movaps/movdqa require 16-byte alignment: fixed aligned base, no SIB.
        rm, b = BASE_ENC[rng.choice(["r14", "r15"])]
        disp = rng.choice([0, 16, 32, 48])
        return _modrm(2, reg_field, rm) + disp.to_bytes(4, "little"), b, 0

    rm, b = BASE_ENC[rng.choice(["rbx", "rbx", "r14", "r15"])]
    mod = rng.choice([0, 1, 2])
    if mod == 0:
        disp_bytes = b""
    elif mod == 1:
        disp_bytes = bytes([rng.randrange(0, 0x50)])
    else:
        disp_bytes = rng.randrange(0, 0x1C0).to_bytes(4, "little")

    if rng.random() < 0.3:
        # [base + r13*scale + disp]; r13 is seeded small.
        sib = bytes([(rng.choice([0, 1, 2, 3]) << 6) | (5 << 3) | rm])
        return _modrm(mod, reg_field, 4) + sib + disp_bytes, b, 1
    return _modrm(mod, reg_field, rm) + disp_bytes, b, 0


def _sse_op(rng, prefix, opcode, aligned=False, imm=None):
    """`op xmm, xmm/m128`: dest in ModRM.reg, source in r/m."""
    dst = rng.randrange(16)
    if rng.random() < 0.45:
        src = rng.randrange(16)
        out = prefix + _rex(0, dst >> 3, 0, src >> 3) + opcode + _modrm(3, dst, src)
    else:
        body, b, x = _mem(rng, dst, aligned)
        out = prefix + _rex(0, dst >> 3, x, b) + opcode + body
    if imm is not None:
        out += bytes([imm])
    return out


# ------------------------------------------------------------- generators

MOVES = [
    (b"", 0x10, 0x11, False),      # movups
    (b"\x66", 0x10, 0x11, False),  # movupd
    (b"\xf3", 0x10, 0x11, False),  # movss
    (b"\xf2", 0x10, 0x11, False),  # movsd
    (b"", 0x28, 0x29, True),       # movaps
    (b"\x66", 0x28, 0x29, True),   # movapd
    (b"\x66", 0x6F, 0x7F, True),   # movdqa
    (b"\xf3", 0x6F, 0x7F, False),  # movdqu
]

ARITH_OPS = [0x58, 0x59, 0x5C, 0x5E, 0x51, 0x5D, 0x5F]
# Everything reached through `66 0F xx /r`: bitwise, compares, the pack family,
# multiplies, min/max, saturating add/sub and the vector-count shifts.
PACKED_INT_OPS = [
    0xDB, 0xDF, 0xEB, 0xEF, 0xFE, 0xFA, 0xD4, 0xFB, 0x74, 0x75, 0x76,
    0x64, 0x65, 0x66,                    # pcmpgt{b,w,d}
    0x63, 0x67, 0x6B,                    # packsswb / packuswb / packssdw
    0xD5, 0xE4, 0xE5, 0xF5, 0xF6,        # pmullw / pmulhuw / pmulhw / pmaddwd / psadbw
    0xDA, 0xDE, 0xEA, 0xEE,              # pminub / pmaxub / pminsw / pmaxsw
    0xD8, 0xD9, 0xDC, 0xDD,              # psubusb / psubusw / paddusb / paddusw
    0xE8, 0xE9, 0xEC, 0xED,              # psubsb / psubsw / paddsb / paddsw
    0xF8, 0xF9, 0xFC, 0xFD,              # psubb / psubw / paddb / paddw
    0xD1, 0xD2, 0xD3, 0xE1, 0xE2, 0xF1, 0xF2, 0xF3,  # shifts by a vector count
]
UNPACK_OPS = [0x60, 0x61, 0x62, 0x68, 0x69, 0x6A, 0x6C, 0x6D]


def _width_prefix(rng):
    """Pick an operand size: (leading prefix byte, REX.W bit)."""
    choice = rng.choice([b"", b"", b"\x66", b"\x48"])
    if choice == b"\x66":
        return b"\x66", 0
    if choice == b"\x48":
        return b"", 1
    return b"", 0


def gen_move(rng):
    if rng.random() < 0.16:
        # movlps/movhps/movlpd/movhpd: memory operand only.
        prefix = rng.choice([b"", b"\x66"])
        opcode = rng.choice([0x12, 0x13, 0x16, 0x17])
        dst = rng.randrange(16)
        body, b, x = _mem(rng, dst)
        return prefix + _rex(0, dst >> 3, x, b) + b"\x0f" + bytes([opcode]) + body
    prefix, load, store, aligned = rng.choice(MOVES)
    opcode = load if rng.random() < 0.5 else store
    return _sse_op(rng, prefix, b"\x0f" + bytes([opcode]), aligned=aligned)


def gen_arith(rng):
    prefix = rng.choice([b"", b"", b"\x66", b"\xf3", b"\xf2"])
    return _sse_op(rng, prefix, b"\x0f" + bytes([rng.choice(ARITH_OPS)]))


def gen_compare(rng):
    if rng.random() < 0.5:
        prefix = rng.choice([b"", b"\x66"])
        opcode = rng.choice([0x2E, 0x2F])
        return _sse_op(rng, prefix, b"\x0f" + bytes([opcode]))
    prefix = rng.choice([b"", b"\x66", b"\xf3", b"\xf2"])
    return _sse_op(rng, prefix, b"\x0f\xc2", imm=rng.randrange(8))


def gen_shuffle(rng):
    roll = rng.random()
    if roll < 0.35:
        prefix = rng.choice([b"\x66", b"\xf3", b"\xf2"])
        return _sse_op(rng, prefix, b"\x0f\x70", imm=rng.randrange(256))
    if roll < 0.55:
        prefix = rng.choice([b"", b"\x66"])
        return _sse_op(rng, prefix, b"\x0f\xc6", imm=rng.randrange(256))
    if rng.random() < 0.5:
        prefix = b"\x66"
        opcode = rng.choice(UNPACK_OPS)
    else:
        prefix = rng.choice([b"", b"\x66"])
        opcode = rng.choice([0x14, 0x15])
    return _sse_op(rng, prefix, b"\x0f" + bytes([opcode]))


def gen_shift(rng):
    opcode = rng.choice([0x71, 0x72, 0x73])
    groups = {0x71: [2, 4, 6], 0x72: [2, 4, 6], 0x73: [2, 3, 6, 7]}[opcode]
    group = rng.choice(groups)
    if opcode == 0x73 and group in (3, 7):
        # pslldq / psrldq shift the whole register by bytes, so 0..16 is the
        # meaningful range and the ends are where an off-by-one hides. Counts
        # past the register width must clear it, so a few of those are included.
        imm = rng.choice(list(range(17)) + [17, 31, 255])
    else:
        imm = rng.choice([0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127])
    # Register destination only: the memory form of these encodings is not a
    # defined instruction (GAS: "operand size mismatch") and the core refuses
    # it, so it cannot be compared.
    src = rng.randrange(16)
    body, b, x = _modrm(3, group, src), src >> 3, 0
    return b"\x66" + _rex(0, 0, x, b) + b"\x0f" + bytes([opcode]) + body + bytes([imm])


def gen_convert(rng):
    kind = rng.randrange(5)
    if kind == 0:
        # cvtsi2ss / cvtsi2sd
        prefix = rng.choice([b"\xf3", b"\xf2"])
        w = 1 if rng.random() < 0.4 else 0
        dst = rng.randrange(16)
        if rng.random() < 0.5:
            gpr = rng.choice(GPRS)
            return prefix + _rex(w, dst >> 3, 0, gpr >> 3) + b"\x0f\x2a" + _modrm(3, dst, gpr)
        body, b, x = _mem(rng, dst)
        return prefix + _rex(w, dst >> 3, x, b) + b"\x0f\x2a" + body
    if kind == 1:
        # cvttss2si / cvttsd2si, and the rounding forms cvtss2si / cvtsd2si
        prefix = rng.choice([b"\xf3", b"\xf2"])
        opcode = 0x2C if rng.random() < 0.5 else 0x2D
        w = 1 if rng.random() < 0.4 else 0
        gpr = rng.choice(GPRS)
        if rng.random() < 0.5:
            src = rng.randrange(16)
            return (
                prefix
                + _rex(w, gpr >> 3, 0, src >> 3)
                + b"\x0f"
                + bytes([opcode])
                + _modrm(3, gpr, src)
            )
        body, b, x = _mem(rng, gpr)
        return prefix + _rex(w, gpr >> 3, x, b) + b"\x0f" + bytes([opcode]) + body
    if kind == 3:
        # cvtdq2ps
        return _sse_op(rng, b"", b"\x0f\x5b")
    if kind == 4:
        # cvttpd2dq / cvtpd2dq / cvtdq2pd
        prefix = rng.choice([b"\x66", b"\xf2", b"\xf3"])
        return _sse_op(rng, prefix, b"\x0f\xe6")
    # cvtps2pd / cvtpd2ps / cvtss2sd / cvtsd2ss
    prefix = rng.choice([b"", b"\x66", b"\xf3", b"\xf2"])
    return _sse_op(rng, prefix, b"\x0f\x5a")


def gen_bitwise(rng):
    if rng.random() < 0.35:
        prefix = rng.choice([b"", b"\x66"])
        opcode = rng.choice([0x54, 0x55, 0x56, 0x57])
    else:
        prefix = b"\x66"
        opcode = rng.choice(PACKED_INT_OPS)
    return _sse_op(rng, prefix, b"\x0f" + bytes([opcode]))


def gen_gpr_transfer(rng):
    if rng.random() < 0.22:
        # pextrw: one 16-bit vector lane into a GPR, zero-extended. gcc -O3 emits
        # this to drain a packed reduction, and it is the only lane extract in
        # the subset, so it gets its own differential coverage. The vector
        # operand is register-only (GAS rejects a memory operand).
        gpr = rng.choice(GPRS)
        src = rng.randrange(16)
        return (
            b"\x66"
            + _rex(0, gpr >> 3, 0, src >> 3)
            + b"\x0f\xc5"
            + _modrm(3, gpr, src)
            + bytes([rng.randrange(8)])
        )
    if rng.random() < 0.3:
        # movmskps / movmskpd / pmovmskb: GPR destination, vector source
        which = rng.randrange(3)
        src_prefix, opcode = (b"\x66", 0xD7) if which == 2 else (b"\x66" if which == 1 else b"", 0x50)
        gpr = rng.choice(GPRS)
        # The vector operand is register-only: GAS rejects a memory operand
        # ("operand size mismatch"), so the encoding is not one to generate.
        src = rng.randrange(16)
        return (
            src_prefix
            + _rex(0, gpr >> 3, 0, src >> 3)
            + b"\x0f"
            + bytes([opcode])
            + _modrm(3, gpr, src)
        )
    if rng.random() < 0.3:
        # movq xmm, xmm/m64 (F3 0F 7E)
        dst = rng.randrange(16)
        if rng.random() < 0.5:
            src = rng.randrange(16)
            return b"\xf3" + _rex(0, dst >> 3, 0, src >> 3) + b"\x0f\x7e" + _modrm(3, dst, src)
        body, b, x = _mem(rng, dst)
        return b"\xf3" + _rex(0, dst >> 3, x, b) + b"\x0f\x7e" + body

    to_xmm = rng.random() < 0.5
    w = 1 if rng.random() < 0.4 else 0
    x = rng.randrange(16)
    opcode = b"\x6e" if to_xmm else b"\x7e"
    if rng.random() < 0.5:
        gpr = rng.choice(GPRS)
        return b"\x66" + _rex(w, x >> 3, 0, gpr >> 3) + b"\x0f" + opcode + _modrm(3, x, gpr)
    body, b, xx = _mem(rng, x)
    return b"\x66" + _rex(w, x >> 3, xx, b) + b"\x0f" + opcode + body


def gen_sse(rng):
    kind = rng.choices(
        ["move", "arith", "compare", "shuffle", "shift", "convert", "bitwise", "gpr"],
        weights=[18, 17, 7, 13, 8, 12, 18, 7],
    )[0]
    return {
        "move": gen_move,
        "arith": gen_arith,
        "compare": gen_compare,
        "shuffle": gen_shuffle,
        "shift": gen_shift,
        "convert": gen_convert,
        "bitwise": gen_bitwise,
        "gpr": gen_gpr_transfer,
    }[kind](rng)


def gen_int(rng):
    kind = rng.randrange(6)
    if kind == 0:
        # ALU reg, reg (8/16/32/64-bit, high-byte registers included)
        opcode = rng.choice([o for o in range(0x00, 0x3E) if (o & 7) < 4])
        prefix, w = _width_prefix(rng)
        dst, src = rng.choice(GPRS), rng.choice(GPRS)
        return (
            prefix
            + _rex(w, dst >> 3, 0, src >> 3)
            + bytes([opcode])
            + _modrm(3, dst, src)
        )
    if kind == 1:
        # group 1 immediate
        opcode = rng.choice([0x80, 0x81, 0x83])
        group = rng.randrange(8)
        prefix, w = (b"", 0) if opcode == 0x80 else _width_prefix(rng)
        dst = rng.choice(GPRS)
        if opcode == 0x81:
            imm_size = 2 if prefix == b"\x66" else 4
            imm = rng.getrandbits(imm_size * 8).to_bytes(imm_size, "little")
        else:
            imm = bytes([rng.randrange(256)])
        return (
            prefix
            + _rex(w, 0, 0, dst >> 3)
            + bytes([opcode])
            + _modrm(3, group, dst)
            + imm
        )
    if kind == 2:
        # rol/ror/rcl/rcr/shl/shr/sar by 1, by imm8, or by CL. The rotate-with-
        # carry pair only appears with a count of one: for every other count the
        # architecture leaves OF undefined, hardware and the reference compute
        # different formulas, and a following setcc would turn that difference
        # into a register difference. tools/check_shift_flags.py covers those
        # counts against the real CPU instead.
        group = rng.choice([0, 1, 2, 3, 4, 5, 7])
        form = 0xD1 if group in (2, 3) else rng.choice([0xD1, 0xC1, 0xD3])
        prefix, w = _width_prefix(rng)
        dst = rng.choice(GPRS)
        out = prefix + _rex(w, 0, 0, dst >> 3) + bytes([form]) + _modrm(3, group, dst)
        if form == 0xC1:
            out += bytes([rng.randrange(0, 64)])
        return out
    if kind == 3:
        # inc / dec
        opcode = 0xFE if rng.random() < 0.3 else 0xFF
        prefix, w = (b"", 0) if opcode == 0xFE else _width_prefix(rng)
        group = rng.randrange(2)
        dst = rng.choice(GPRS)
        return (
            prefix
            + _rex(w, 0, 0, dst >> 3)
            + bytes([opcode])
            + _modrm(3, group, dst)
        )
    if kind == 5:
        # shld/shrd r/m, r, 1 -- imm8 by one only, for the same reason: the
        # count-one form is the only one whose flags the architecture defines.
        opcode = rng.choice([0xA4, 0xAC])
        prefix, w = _width_prefix(rng)
        dst, src = rng.choice(GPRS), rng.choice(GPRS)
        out = (
            prefix
            + _rex(w, src >> 3, 0, dst >> 3)
            + b"\x0f"
            + bytes([opcode])
            + _modrm(3, src, dst)
        )
        out += b"\x01"
        return out
    # test / setcc: make the flags observable in a register
    if rng.random() < 0.5:
        dst, src = rng.choice(GPRS), rng.choice(GPRS)
        return _rex(0, dst >> 3, 0, src >> 3) + b"\x85" + _modrm(3, dst, src)
    dst = rng.choice(GPRS)
    cc = rng.randrange(16)
    return _rex(0, 0, 0, dst >> 3) + b"\x0f" + bytes([0x90 + cc]) + _modrm(3, 0, dst)


def gen_string(rng):
    out = bytearray()
    if rng.random() < 0.6:
        out += b"\xfd" if rng.random() < 0.5 else b"\xfc"
    for _ in range(rng.randrange(1, 3)):
        opcode = rng.choice([0xA4, 0xA5, 0xAA, 0xAB, 0xAC, 0xAD, 0xA6, 0xA7, 0xAE, 0xAF])
        if opcode in (0xA6, 0xA7, 0xAE, 0xAF):
            rep = rng.choice([b"", b"\xf3", b"\xf2"])
        else:
            rep = rng.choice([b"", b"", b"\xf3"])
        prefix, w = _width_prefix(rng)
        out += rep + prefix + _rex(w) + bytes([opcode])
    if rng.random() < 0.5:
        out += b"\xfc"
    return bytes(out)


# ------------------------------------------------------------------ case

class Case:
    def __init__(self, program, seeds, xmm_seeds, arena, documented):
        self.program = program
        self.seeds = seeds
        self.xmm_seeds = xmm_seeds
        self.arena = arena
        self.documented = documented


def make_xmm_seed(rng):
    mode = rng.randrange(4)
    if mode == 0:
        return rng.getrandbits(128)
    if mode == 1:
        value = 0
        for lane in range(2):
            word = rng.choice(F64_POOL) if rng.random() < 0.85 else rng.getrandbits(64)
            value |= word << (64 * lane)
        return value
    value = 0
    for lane in range(4):
        word = rng.choice(F32_POOL) if rng.random() < 0.85 else rng.getrandbits(32)
        value |= word << (32 * lane)
    return value


def make_arena(rng):
    out = bytearray()
    while len(out) < ARENA_LEN:
        choice = rng.random()
        if choice < 0.55:
            out += rng.choice(F32_POOL).to_bytes(4, "little")
        elif choice < 0.75:
            out += rng.choice(F64_POOL).to_bytes(8, "little")
        elif choice < 0.9:
            out += bytes([rng.choice([0x00, 0xFF, 0x7F, 0x80])]) * 8
        else:
            out += rng.getrandbits(64).to_bytes(8, "little")
    return bytes(out[:ARENA_LEN])


def build_case(rng):
    seeds = {
        # Address bases and string pointers pinned to the arena.
        "rbx": ARENA + rng.choice([0, 1, 2, 3, 5, 7, 8, 15]),
        "r14": ARENA + 0x100,
        "r15": ARENA + 0x180,
        "r13": rng.randrange(4),
        "rsi": ARENA + rng.randrange(0x20),
        "rdi": ARENA + 0x40 + rng.randrange(0x20),
        "rcx": rng.randrange(0, 9),
    }
    for name in ("rax", "rdx", "rbp", "r8", "r9", "r10", "r11", "r12"):
        seeds[name] = rng.choice(EDGE_VALUES)
    xmm_seeds = [make_xmm_seed(rng) for _ in range(16)]
    arena = make_arena(rng)

    if rng.random() < 0.02:
        # The one documented divergence, isolated in its own case: a 0x66
        # prefix on `xchg rAX, r` is a 16-bit exchange on hardware, while
        # QEMU ignores the prefix.
        index = rng.randrange(16)
        program = (
            b"\x66"
            + _rex(0, 0, 0, index >> 3)
            + bytes([0x90 | (index & 7)])
            + b"\xcc"
        )
        return Case(program, seeds, xmm_seeds, arena, documented=True)

    program = bytearray()
    for _ in range(rng.randrange(6, 11)):
        family = rng.choices(["sse", "int", "string"], weights=[62, 23, 15])[0]
        if family == "sse":
            program += gen_sse(rng)
        elif family == "int":
            program += gen_int(rng)
        else:
            program += gen_string(rng)
    program += b"\xcc"
    return Case(bytes(program), seeds, xmm_seeds, arena, documented=False)


# ----------------------------------------------------------------- running

def find_binary():
    for rel in ("target/release/xone", "target/debug/xone"):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            return path
    sys.exit("build the emulator first: cargo build --release")


def run_ours(binary, case, workdir):
    prog_path = os.path.join(workdir, "case.bin")
    arena_path = os.path.join(workdir, "arena.bin")
    with open(prog_path, "wb") as handle:
        handle.write(case.program)
    with open(arena_path, "wb") as handle:
        handle.write(case.arena)

    args = [
        binary,
        "run",
        prog_path,
        "--state",
        "--steps",
        str(STEP_BUDGET),
        "--dump",
        f"0:{MAP_LEN}",
        "--load",
        f"0x{ARENA:x}:{arena_path}",
    ]
    for name, value in case.seeds.items():
        args += ["--reg", f"{name}=0x{value:x}"]
    for index, value in enumerate(case.xmm_seeds):
        args += ["--xmm", f"{index}=0x{value:032x}"]

    proc = subprocess.run(args, capture_output=True, text=True)
    state = re.search(r"^STATE (.+)$", proc.stdout, re.MULTILINE)
    dump = re.search(r"^DUMP [0-9a-f]+ ([0-9a-f]+)$", proc.stdout, re.MULTILINE)
    if not state or not dump:
        raise RuntimeError(f"unexpected output; stderr: {proc.stderr.strip()}")
    fields = dict(token.split("=", 1) for token in state.group(1).split())
    return {
        "regs": {name: int(fields[name], 16) for name in GPR_NAMES},
        "xmm": [int(fields[f"xmm{i}"], 16) for i in range(16)],
        "flags": int(fields["rflags"], 16) & FLAG_MASK,
        "mem": bytes.fromhex(dump.group(1)),
        "stop": fields["stop"],
        "detail": proc.stderr.strip(),
    }


def run_unicorn(case, sentinel):
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(MEM_BASE, MEM_LEN)
    uc.mem_write(CODE, case.program)
    uc.mem_write(ARENA, case.arena)
    for name, reg_id in REG_IDS.items():
        uc.reg_write(reg_id, case.seeds.get(name, 0))
    uc.reg_write(ux.UC_X86_REG_RSP, STACK)
    for index, value in enumerate(case.xmm_seeds):
        uc.reg_write(XMM_IDS[index], value)

    executed = []
    uc.hook_add(UC_HOOK_CODE, lambda u, addr, size, data: executed.append(addr))
    try:
        uc.emu_start(CODE, 0xFFFFF000, 0, STEP_BUDGET)
    except UcError:
        pass

    return {
        "regs": {name: uc.reg_read(reg_id) for name, reg_id in REG_IDS.items()},
        "xmm": [uc.reg_read(XMM_IDS[i]) for i in range(16)],
        "flags": uc.reg_read(ux.UC_X86_REG_EFLAGS) & FLAG_MASK,
        "mem": bytes(uc.mem_read(MEM_BASE, MAP_LEN)),
        "reached": (executed[-1] if executed else None) == sentinel,
    }


def diff_state(ours, ref):
    """Structured description of every field that differs."""
    diff = {"stop": False, "regs": {}, "xmm": {}, "flags": False, "mem": None}
    diff["stop"] = ours["stop"] != "int3"
    for name, value in ours["regs"].items():
        if ref["regs"][name] != value:
            diff["regs"][name] = (value, ref["regs"][name])
    for index, value in enumerate(ours["xmm"]):
        if ref["xmm"][index] != value:
            diff["xmm"][index] = (value, ref["xmm"][index])
    diff["flags"] = ref["flags"] != ours["flags"]
    if ref["mem"] != ours["mem"]:
        diff["mem"] = next(
            i for i in range(min(len(ours["mem"]), len(ref["mem"])))
            if ours["mem"][i] != ref["mem"][i]
        )
    return diff


def describe(ours, ref, diff):
    problems = []
    if diff["stop"]:
        problems.append(f"our core stopped on {ours['stop']!r}, not the int3 sentinel")
    for name, (mine, theirs) in diff["regs"].items():
        problems.append(f"{name}: ours=0x{mine:016x} unicorn=0x{theirs:016x}")
    for index, (mine, theirs) in diff["xmm"].items():
        problems.append(f"xmm{index}: ours=0x{mine:032x} unicorn=0x{theirs:032x}")
    if diff["flags"]:
        problems.append(f"rflags: ours=0x{ours['flags']:04x} unicorn=0x{ref['flags']:04x}")
    if diff["mem"] is not None:
        offset = diff["mem"]
        problems.append(
            f"memory at 0x{offset:04x}: ours={ours['mem'][offset:offset + 16].hex()} "
            f"unicorn={ref['mem'][offset:offset + 16].hex()}"
        )
    return problems


ARITH = (0x58, 0x59, 0x5C, 0x5E)
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def is_f32_nan(bits):
    return bits & 0x7F80_0000 == 0x7F80_0000 and bits & 0x007F_FFFF != 0


def is_f64_nan(bits):
    return bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 and bits & 0x000F_FFFF_FFFF_FFFF != 0


def has_arith(program):
    """True if the stream contains an add/sub/mul/div on packed or scalar floats."""
    return any(
        program[i] == 0x0F and program[i + 1] in ARITH for i in range(len(program) - 1)
    )


def reference_nan_defect(case, diff):
    """Is the whole difference the documented reference defect?

    Unicorn (QEMU) hands double-precision arithmetic to a soft-float path
    inherited from x87, which returns the NaN with the *larger significand*
    when both operands are NaNs. Hardware returns the destination operand's
    NaN; this core follows the hardware, and tests/c/nan_rule.c re-establishes
    that on the real CPU for every row of the matrix, so the claim is checked
    rather than asserted.

    The defect affects the packed forms of both widths, so adjudication is
    still narrow: nothing but XMM lanes may differ, the case must contain a
    floating-point arithmetic instruction, and every differing lane -- read as
    single or as double, whichever explains all of them -- must be a NaN on
    both sides. Any other difference, and any non-NaN payload difference, is a
    real failure. A genuine NaN-selection regression is caught independently by
    tests/c/nan_rule.c against the native CPU.
    """
    if diff["stop"] or diff["flags"] or diff["mem"] is not None or diff["regs"]:
        return False
    if not diff["xmm"] or not has_arith(case.program):
        return False

    def all_nan(width, mask, is_nan):
        for ours_value, ref_value in diff["xmm"].values():
            for lane in range(128 // width):
                mine = (ours_value >> (width * lane)) & mask
                theirs = (ref_value >> (width * lane)) & mask
                if mine != theirs and not (is_nan(mine) and is_nan(theirs)):
                    return False
        return True

    return all_nan(32, MASK32, is_f32_nan) or all_nan(64, MASK64, is_f64_nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=400, help="cases per seed")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3], help="PRNG seeds to run"
    )
    parser.add_argument("--max-report", type=int, default=10)
    parser.add_argument(
        "--replay",
        help="re-run one case by seed:index and print its full state",
    )
    options = parser.parse_args()

    binary = find_binary()
    workdir = os.path.join(ROOT, "build")
    os.makedirs(workdir, exist_ok=True)

    if options.replay:
        seed_text, _, index_text = options.replay.partition(":")
        seed, index = int(seed_text), int(index_text)
        rng = random.Random(seed)
        case = None
        for _ in range(index + 1):
            case = build_case(rng)
        sentinel = CODE + len(case.program) - 1
        ref = run_unicorn(case, sentinel)
        ours = run_ours(binary, case, workdir)
        print(f"program={case.program.hex()}")
        print(f"reached={ref['reached']} documented={case.documented}")
        print(f"seeds={case.seeds}")
        print("xmm=" + " ".join(f"{v:032x}" for v in case.xmm_seeds))
        print(f"arena={case.arena.hex()}")
        for problem in describe(ours, ref, diff_state(ours, ref)):
            print(f"     {problem}")
        return 0

    total_compared = 0
    total_documented = 0
    total_faulted = 0
    total_failed = 0
    total_ref_defects = 0

    for seed in options.seeds:
        rng = random.Random(seed)
        compared = documented = faulted = failed = ref_defects = 0
        for index in range(options.iterations):
            case = build_case(rng)
            sentinel = CODE + len(case.program) - 1
            ref = run_unicorn(case, sentinel)

            if not ref["reached"]:
                # Unicorn faulted or rejected the encoding: not a comparison.
                faulted += 1
                continue

            ours = run_ours(binary, case, workdir)
            diff = diff_state(ours, ref)
            problems = describe(ours, ref, diff)
            if problems and case.documented:
                documented += 1
                continue
            if problems and reference_nan_defect(case, diff):
                ref_defects += 1
                if ref_defects <= options.max_report:
                    print(f"REFDEF seed={seed} index={index} {case.program.hex()}")
                    for problem in problems:
                        print(f"     {problem}")
                continue
            if problems:
                failed += 1
                if failed <= options.max_report:
                    print(f"FAIL seed={seed} index={index} {case.program.hex()}")
                    for problem in problems:
                        print(f"     {problem}")
                    if ours["detail"]:
                        print(f"     core: {ours['detail']}")
                    print(f"     seeds={case.seeds}")
                    print("     xmm=" + " ".join(f"{v:032x}" for v in case.xmm_seeds))
                    print(f"     arena={case.arena.hex()}")
                continue
            compared += 1

        total_compared += compared
        total_documented += documented
        total_faulted += faulted
        total_failed += failed
        total_ref_defects += ref_defects
        print(
            f"seed {seed}: {options.iterations} cases -> "
            f"{compared} matched, {documented} documented 0x66-xchg skips, "
                            f"{ref_defects} reference-NaN differences, "
            f"{faulted} unicorn faults, {failed} failed"
        )

    print()
    print(
        f"total: {total_compared} cases matched Unicorn field-for-field, "
        f"{total_documented} documented 0x66-xchg skips, "
        f"{total_ref_defects} reference double-NaN differences, "
        f"{total_faulted} unicorn faults, {total_failed} diverged"
    )
    if total_compared == 0:
        print("warning: nothing was actually compared")
        return 1
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
