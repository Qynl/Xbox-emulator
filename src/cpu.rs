//! x86-64 CPU interpreter (milestone 1 of the Xbox emulator).
//!
//! Durango is an 8-core AMD Jaguar x86-64 console, so the first thing a real
//! emulator needs is a working x86-64 execution core. This module decodes and
//! executes actual machine code; every instruction that reports success really
//! produced its architectural effect.
//!
//! Scope of the implemented subset (all validated by `tools/diff_unicorn.py`):
//!   * REX / 0x66 / 0x67 / segment / LOCK / REP prefixes
//!   * mov (reg, imm, mem), movzx, movsx, movsxd, lea, xchg
//!   * add, or, adc, sbb, and, sub, xor, cmp, test, inc, dec, neg, not
//!   * imul (2- and 3-operand, plus one-operand), mul, div, idiv
//!   * shl/sal, shr, sar, rol, ror with imm and CL counts
//!   * push, pop, pushf/popf, call, ret, leave, jmp, all Jcc, cmovcc, setcc
//!   * bsf/bsr/tzcnt/lzcnt/popcnt, cpuid, rdtsc, fences (as no-ops)
//!   * cdq/cqo/cwde/cdqe, bswap, int3 (breakpoint), hlt, ud2, syscall
//!   * movs/stos/lods/cmps/scas with REP/REPE/REPNE and the DF flag
//!   * SSE2: the 16 XMM registers, moves (`movaps`/`movups`/`movdqa`/`movdqu`/
//!     `movss`/`movsd`/`movd`/`movq`), packed integer ops (`paddd`, `psubq`,
//!     `pxor`, `pand`, `por`, `pcmpeq*`, `pshufd`, `punpck*`, packed shifts),
//!     packed and scalar float arithmetic in single and double precision,
//!     `sqrt*`, `min*`/`max*`, `comis*`/`ucomis*`, `cmpps`/`cmpsd` predicates,
//!     and the `cvtsi2sd`/`cvttsd2si`/`cvtss2sd`/`cvtsd2ss` conversions.
//!
//! Two-byte opcodes have one source of truth for which vector encodings
//! exist: everything the integer half of `two_byte` does not claim is handed
//! to `sse`, so adding an instruction needs an arm there and nowhere else.
//!
//! Known gaps, tracked in ROADMAP.md: AVX/AVX2, MMX, x87, MMU paging,
//! interrupts and I/O ports are not implemented yet. Encountering an opcode outside the subset
//! stops with `Stop::BadOpcode` rather than silently doing nothing, and
//! `Cpu::last_insn` / `last_opcode` / `last_ext` record exactly where.

use crate::flags::Flags;
use crate::memory::{width_mask, Memory};

/// Why the interpreter stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    None,
    Halted,
    Int3,
    Ud2,
    Syscall,
    BadOpcode,
    StepLimit,
    DivideError,
}

impl Stop {
    pub fn as_str(self) -> &'static str {
        match self {
            Stop::None => "running",
            Stop::Halted => "hlt",
            Stop::Int3 => "int3",
            Stop::Ud2 => "ud2",
            Stop::Syscall => "syscall",
            Stop::BadOpcode => "bad-opcode",
            Stop::StepLimit => "step-limit",
            Stop::DivideError => "divide-error",
        }
    }
}

// Register indices, matching the x86 encoding order used by ModRM/SIB.
pub const RAX: usize = 0;
pub const RCX: usize = 1;
pub const RDX: usize = 2;
pub const RBX: usize = 3;
pub const RSP: usize = 4;
pub const RBP: usize = 5;
pub const RSI: usize = 6;
pub const RDI: usize = 7;
pub const R8: usize = 8;
pub const R9: usize = 9;
pub const R10: usize = 10;
pub const R11: usize = 11;
pub const R12: usize = 12;
pub const R13: usize = 13;
pub const R14: usize = 14;
pub const R15: usize = 15;

pub const PROGRAM_BASE: u64 = 0x1000;
/// Stack top used by the bring-up harness. Kept low so a whole run's memory
/// (code + stack) fits in a small, hashable window.
pub const STACK_TOP: u64 = 0x8000;

/// A decoded operand: either a register or an effective address.
#[derive(Clone, Copy, Debug)]
struct Op {
    bits: u32,
    is_reg: bool,
    idx: usize,
    addr: u64,
    /// AH/CH/DH/BH: 8-bit access to bits 8..16 of a low register.
    high8: bool,
}

/// Re-target a decoded register operand at the instruction's destination
/// width. movzx/movsx/movsxd decode the r/m operand with the *source* width,
/// but always write a 16/32/64-bit destination.
fn widen_destination(op: &mut Op, bits: u32) {
    op.bits = bits;
    op.high8 = false;
}

/// Bit-pattern view of one 32-bit SSE lane.
#[inline]
fn f32_lane(v: u128, lane: u32) -> f32 {
    f32::from_bits((v >> (32 * lane)) as u32)
}

/// Bit-pattern view of one 64-bit SSE lane.
#[inline]
fn f64_lane(v: u128, lane: u32) -> f64 {
    f64::from_bits((v >> (64 * lane)) as u64)
}

#[inline]
fn with_f32_lane(v: u128, lane: u32, x: f32) -> u128 {
    let mask = 0xFFFF_FFFFu128 << (32 * lane);
    (v & !mask) | ((x.to_bits() as u128) << (32 * lane))
}

#[inline]
fn with_f64_lane(v: u128, lane: u32, x: f64) -> u128 {
    let mask = 0xFFFF_FFFF_FFFF_FFFFu128 << (64 * lane);
    (v & !mask) | ((x.to_bits() as u128) << (64 * lane))
}

/// Bit-pattern view of one 16-bit SSE lane, used by the word shuffles.
#[inline]
fn u16_lane(v: u128, lane: u32) -> u16 {
    (v >> (16 * lane)) as u16
}

#[inline]
fn with_u16_lane(v: u128, lane: u32, x: u16) -> u128 {
    let mask = 0xFFFFu128 << (16 * lane);
    (v & !mask) | ((x as u128) << (16 * lane))
}

/// NaN selection for the SSE arithmetic family, done on raw bit patterns.
///
/// Measured hardware behaviour (see `tests/c/nan_rule.c`, which the test
/// script runs against this emulator): the result is the *first* (destination)
/// operand's NaN, quietened; the source's NaN only surfaces when the
/// destination is not itself a NaN. This cannot be left to `a + b` in Rust:
/// LLVM is free to commute a floating-point add, which silently changes which
/// payload survives, and it does for the packed loops below.
#[inline]
fn fp32_nan_sel(a: u32, b: u32) -> Option<u32> {
    const EXP: u32 = 0x7F80_0000;
    const MANT: u32 = 0x007F_FFFF;
    const QUIET: u32 = 0x0040_0000;
    if a & EXP == EXP && a & MANT != 0 {
        return Some(a | QUIET);
    }
    if b & EXP == EXP && b & MANT != 0 {
        return Some(b | QUIET);
    }
    None
}

#[inline]
fn fp64_nan_sel(a: u64, b: u64) -> Option<u64> {
    const EXP: u64 = 0x7FF0_0000_0000_0000;
    const MANT: u64 = 0x000F_FFFF_FFFF_FFFF;
    const QUIET: u64 = 0x0008_0000_0000_0000;
    if a & EXP == EXP && a & MANT != 0 {
        return Some(a | QUIET);
    }
    if b & EXP == EXP && b & MANT != 0 {
        return Some(b | QUIET);
    }
    None
}

/// `op` is the 0F opcode: 0x58 add, 0x59 mul, 0x5C sub, 0x5E div.
#[inline]
fn fp32_arith(op: u8, a: u32, b: u32) -> u32 {
    if let Some(nan) = fp32_nan_sel(a, b) {
        return nan;
    }
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    let r = match op {
        0x58 => x + y,
        0x59 => x * y,
        0x5C => x - y,
        _ => x / y,
    };
    r.to_bits()
}

#[inline]
fn fp64_arith(op: u8, a: u64, b: u64) -> u64 {
    if let Some(nan) = fp64_nan_sel(a, b) {
        return nan;
    }
    let (x, y) = (f64::from_bits(a), f64::from_bits(b));
    let r = match op {
        0x58 => x + y,
        0x59 => x * y,
        0x5C => x - y,
        _ => x / y,
    };
    r.to_bits()
}

/// Round-to-nearest-even float→integer conversion, as `cvtss2si`/`cvtsd2si`
/// perform under the default MXCSR rounding mode.
fn cvt_round(value: f64, bits: u32) -> u64 {
    cvt_trunc(value.round_ties_even(), bits)
}

/// Saturating narrowing conversions used by the `pack*` family.
#[inline]
fn sat_i8(v: i16) -> u8 {
    v.clamp(-128, 127) as i8 as u8
}

#[inline]
fn sat_u8(v: i16) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn sat_i16(v: i32) -> u16 {
    v.clamp(-32768, 32767) as i16 as u16
}

#[inline]
fn sat_u16(v: i32) -> u16 {
    v.clamp(0, 65535) as u16
}

#[inline]
fn sat_add_i8(a: i8, b: i8) -> i8 {
    (a as i16 + b as i16).clamp(-128, 127) as i8
}

#[inline]
fn sat_add_i16(a: i16, b: i16) -> i16 {
    (a as i32 + b as i32).clamp(-32768, 32767) as i16
}

#[inline]
fn sat_sub_i8(a: i8, b: i8) -> i8 {
    (a as i16 - b as i16).clamp(-128, 127) as i8
}

#[inline]
fn sat_sub_i16(a: i16, b: i16) -> i16 {
    (a as i32 - b as i32).clamp(-32768, 32767) as i16
}

#[inline]
fn sat_add_u8(a: u8, b: u8) -> u8 {
    (a as u16 + b as u16).min(255) as u8
}

#[inline]
fn sat_add_u16(a: u16, b: u16) -> u16 {
    (a as u32 + b as u32).min(65535) as u16
}

#[inline]
fn sat_sub_u8(a: u8, b: u8) -> u8 {
    a.saturating_sub(b)
}

#[inline]
fn sat_sub_u16(a: u16, b: u16) -> u16 {
    a.saturating_sub(b)
}

/// x86 `minss`/`minsd`: if either operand is NaN the result is the second one.
fn x86_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        b
    } else if a < b {
        a
    } else {
        b
    }
}

fn x86_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        b
    } else if a > b {
        a
    } else {
        b
    }
}

fn x86_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        b
    } else if a < b {
        a
    } else {
        b
    }
}

fn x86_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        b
    } else if a > b {
        a
    } else {
        b
    }
}

/// Truncating float→integer conversion. Out-of-range and NaN inputs produce
/// the architectural "integer indefinite" value instead of Rust's saturation.
fn cvt_trunc(value: f64, bits: u32) -> u64 {
    let indefinite = if bits == 64 {
        0x8000_0000_0000_0000
    } else {
        0x8000_0000
    };
    if value.is_nan() {
        return indefinite;
    }
    let truncated = value.trunc();
    let limit = if bits == 64 {
        9_223_372_036_854_775_808.0
    } else {
        2_147_483_648.0
    };
    if truncated < -limit || truncated >= limit {
        indefinite
    } else {
        truncated as i64 as u64
    }
}

/// SSE comparison predicate (imm8 of `cmpps`/`cmpsd`/...), predicates 0..=7.
/// Values 8..=15 differ only in whether QNaN raises #IA, not in the result.
fn cmp_predicate(pred: u8, a: f64, b: f64) -> bool {
    match pred & 7 {
        0 => a == b,
        1 => a < b,
        2 => a <= b,
        3 => a.is_nan() || b.is_nan(),
        4 => !(a == b),
        5 => !(a < b),
        6 => !(a <= b),
        _ => !(a.is_nan() || b.is_nan()),
    }
}

pub struct Cpu {
    pub mem: Memory,
    pub r: [u64; 16],
    /// 16 SSE/SSE2 vector registers; lane 0 lives in the low bits.
    pub xmm: [u128; 16],
    pub rip: u64,
    pub flags: Flags,
    pub steps: u64,
    pub stop: Stop,
    pub trace: bool,
    /// Address of the instruction being executed, and the opcode bytes that
    /// decoded it. When a run dies on `BadOpcode` this is the first thing you
    /// need, so the front-end prints it.
    pub last_insn: u64,
    pub last_opcode: u8,
    pub last_ext: Option<u8>,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            mem: Memory::new(),
            r: [0; 16],
            xmm: [0; 16],
            rip: PROGRAM_BASE,
            flags: Flags::default(),
            steps: 0,
            stop: Stop::None,
            trace: false,
            last_insn: 0,
            last_opcode: 0,
            last_ext: None,
        }
    }

    /// Zero the architectural state and point the CPU at `entry`.
    pub fn reset(&mut self, entry: u64) {
        self.r = [0; 16];
        self.r[RSP] = STACK_TOP;
        self.xmm = [0; 16];
        self.rip = entry;
        self.flags = Flags::default();
        self.steps = 0;
        self.stop = Stop::None;
        self.last_insn = entry;
        self.last_opcode = 0;
        self.last_ext = None;
    }

    pub fn load_image(&mut self, addr: u64, bytes: &[u8]) {
        self.mem.write_bytes(addr, bytes);
    }

    /// Run until the program stops or `max_steps` instructions have executed.
    pub fn run(&mut self, max_steps: u64) -> Stop {
        while self.steps < max_steps {
            let stop = self.step();
            self.steps += 1;
            if stop != Stop::None {
                self.stop = stop;
                return stop;
            }
        }
        self.stop = Stop::StepLimit;
        Stop::StepLimit
    }

    // ---------------------------------------------------------------- registers

    #[inline]
    pub fn get_reg(&self, idx: usize, bits: u32) -> u64 {
        match bits {
            64 => self.r[idx],
            32 => self.r[idx] & 0xFFFF_FFFF,
            16 => self.r[idx] & 0xFFFF,
            8 => self.r[idx] & 0xFF,
            _ => unreachable!("bad operand width {bits}"),
        }
    }

    #[inline]
    pub fn set_reg(&mut self, idx: usize, value: u64, bits: u32) {
        match bits {
            64 => self.r[idx] = value,
            32 => self.r[idx] = value & 0xFFFF_FFFF,
            16 => self.r[idx] = (self.r[idx] & !0xFFFF) | (value & 0xFFFF),
            8 => self.r[idx] = (self.r[idx] & !0xFF) | (value & 0xFF),
            _ => unreachable!("bad operand width {bits}"),
        }
    }

    fn sign_extend(value: u64, from_bits: u32) -> u64 {
        match from_bits {
            8 => value as u8 as i8 as i64 as u64,
            16 => value as u16 as i16 as i64 as u64,
            32 => value as u32 as i32 as i64 as u64,
            _ => value,
        }
    }

    // ---------------------------------------------------------------- fetching

    #[inline]
    fn fetch8(&mut self) -> u8 {
        let v = self.mem.read8(self.rip);
        self.rip = self.rip.wrapping_add(1);
        v
    }

    // Wider fetches are one unaligned load rather than a byte-at-a-time loop:
    // every instruction pays for at least one of them.
    #[inline]
    fn fetch16(&mut self) -> u16 {
        let addr = self.rip;
        self.rip = addr.wrapping_add(2);
        self.mem.read(addr, 16) as u16
    }

    #[inline]
    pub fn fetch32(&mut self) -> u32 {
        let addr = self.rip;
        self.rip = addr.wrapping_add(4);
        self.mem.read(addr, 32) as u32
    }

    #[inline]
    fn fetch64(&mut self) -> u64 {
        let addr = self.rip;
        self.rip = addr.wrapping_add(8);
        self.mem.read(addr, 64)
    }

    #[inline]
    fn fetch_imm(&mut self, bits: u32) -> u64 {
        match bits {
            8 => self.fetch8() as u64,
            16 => self.fetch16() as u64,
            32 => self.fetch32() as u64,
            64 => self.fetch64(),
            _ => 0,
        }
    }

    // ---------------------------------------------------------------- operands

    /// Decode a ModRM byte (plus SIB and displacement) into the r/m operand and
    /// the register-field operand. `has_rex` matters because a REX prefix
    /// re-purposes the 8-bit high registers.
    fn decode_rm(&mut self, bits: u32, rex: u8, has_rex: bool) -> (Op, Op) {
        let m = self.fetch8();
        let md = m >> 6;
        let rex_r = ((rex >> 2) & 1) as usize;
        let rex_x = ((rex >> 1) & 1) as usize;
        let rex_b = (rex & 1) as usize;
        let reg_field = ((m >> 3) & 7) as usize;
        let rm_field = (m & 7) as usize;

        let reg = Op {
            bits,
            is_reg: true,
            idx: reg_field | (rex_r << 3),
            addr: 0,
            high8: bits == 8 && !has_rex && reg_field >= 4,
        };

        if md == 3 {
            return (
                Op {
                    bits,
                    is_reg: true,
                    idx: rm_field | (rex_b << 3),
                    addr: 0,
                    high8: bits == 8 && !has_rex && rm_field >= 4,
                },
                reg,
            );
        }

        let mut addr: u64;
        if rm_field == 4 {
            // SIB byte
            let sib = self.fetch8();
            let scale = 1u64 << (sib >> 6);
            let index = (((sib >> 3) & 7) as usize) | (rex_x << 3);
            let base_field = (sib & 7) as usize;
            let index_value = if index == 4 {
                0
            } else {
                self.r[index].wrapping_mul(scale)
            };
            addr = if base_field == 5 && md == 0 {
                // no base register, disp32 only
                self.fetch32() as i32 as i64 as u64
            } else {
                self.r[base_field | (rex_b << 3)]
            };
            addr = addr.wrapping_add(index_value);
        } else if rm_field == 5 && md == 0 {
            // RIP-relative
            let disp = self.fetch32() as i32 as i64;
            addr = self.rip.wrapping_add(disp as u64);
        } else {
            addr = self.r[rm_field | (rex_b << 3)];
        }

        if md == 1 {
            addr = addr.wrapping_add(self.fetch8() as i8 as i64 as u64);
        } else if md == 2 {
            addr = addr.wrapping_add(self.fetch32() as i32 as i64 as u64);
        }

        (
            Op {
                bits,
                is_reg: false,
                idx: 0,
                addr,
                high8: false,
            },
            reg,
        )
    }

    #[inline]
    fn read_op(&self, op: &Op) -> u64 {
        if op.is_reg {
            if op.high8 {
                (self.r[op.idx - 4] >> 8) & 0xFF
            } else {
                self.get_reg(op.idx, op.bits)
            }
        } else {
            self.mem.read(op.addr, op.bits)
        }
    }

    #[inline]
    fn write_op(&mut self, op: &Op, value: u64) {
        if op.is_reg {
            if op.high8 {
                let idx = op.idx - 4;
                self.r[idx] = (self.r[idx] & !0xFF00) | ((value & 0xFF) << 8);
            } else {
                self.set_reg(op.idx, value, op.bits);
            }
        } else {
            self.mem.write(op.addr, value, op.bits);
        }
    }

    fn push(&mut self, value: u64, bits: u32) {
        let bytes = (bits / 8) as u64;
        self.r[RSP] = self.r[RSP].wrapping_sub(bytes);
        self.mem.write(self.r[RSP], value, bits);
    }

    fn pop(&mut self, bits: u32) -> u64 {
        let v = self.mem.read(self.r[RSP], bits);
        self.r[RSP] = self.r[RSP].wrapping_add((bits / 8) as u64);
        v
    }

    // ------------------------------------------------------------------- flags

    fn szp(&mut self, res: u64, bits: u32) {
        let res = res & width_mask(bits);
        self.flags.zf = res == 0;
        self.flags.sf = (res >> (bits - 1)) & 1 == 1;
        self.flags.pf = (res as u8).count_ones() % 2 == 0;
    }

    /// 0 add, 1 or, 2 adc, 3 sbb, 4 and, 5 sub, 6 xor, 7 cmp.
    fn alu(&mut self, op: u8, a: u64, b: u64, bits: u32) -> u64 {
        let mask = width_mask(bits);
        let a = a & mask;
        let b = b & mask;
        let carry_in = if op == 2 || op == 3 {
            self.flags.cf as u64
        } else {
            0
        };

        match op {
            0 | 2 => {
                let sum = a as u128 + b as u128 + carry_in as u128;
                let res = (sum as u64) & mask;
                self.flags.cf = (sum >> bits) != 0;
                self.flags.of = ((a ^ res) & (b ^ res)) >> (bits - 1) & 1 == 1;
                self.flags.af = ((a ^ b ^ res) >> 4) & 1 == 1;
                self.szp(res, bits);
                res
            }
            3 | 5 | 7 => {
                let diff = a as i128 - b as i128 - carry_in as i128;
                let res = (diff as u64) & mask;
                self.flags.cf = diff < 0;
                self.flags.of = ((a ^ b) & (a ^ res)) >> (bits - 1) & 1 == 1;
                self.flags.af = ((a ^ b ^ res) >> 4) & 1 == 1;
                self.szp(res, bits);
                if op == 7 {
                    a
                } else {
                    res
                }
            }
            _ => {
                let res = match op {
                    1 => a | b,
                    4 => a & b,
                    _ => a ^ b,
                };
                self.flags.cf = false;
                self.flags.of = false;
                self.flags.af = false;
                self.szp(res, bits);
                res
            }
        }
    }

    /// Shift/rotate group. `kind`: 0 shl, 1 shr, 2 sar, 3 rol, 4 ror.
    /// `rcl`/`rcr`: the carry behaves as an extra bit, so the rotation happens
    /// in a `bits + 1` wide field (the "ring") and the result is the count
    /// reduced modulo `bits + 1`. `kind` is 5 for rcl and 6 for rcr.
    fn rotate_carry(&mut self, kind: u8, value: u64, count: u8, bits: u32) -> u64 {
        let mask = width_mask(bits);
        let v = value & mask;
        let masked = count as u32 & if bits == 64 { 0x3F } else { 0x1F };
        if masked == 0 {
            // Count masked to zero: the instruction does nothing at all,
            // flags included.
            return v;
        }
        let wide = bits + 1;
        // A count of `bits + 1` (reachable for 16-bit operands, whose count is
        // masked to five bits) rotates the ring a whole turn: the result is
        // unchanged but CF and OF are still written, as hardware does. The
        // overflow pair below therefore indexes the ring by the *masked* count,
        // not by the count reduced modulo the ring width.
        let c = masked % wide;
        let carry = if self.flags.cf { 1u128 } else { 0 };
        let full = ((v as u128) << 1) | carry;
        let span = 1u128 << wide;
        let rot = if c == 0 {
            full
        } else if kind == 5 {
            ((full << c) | (full >> (wide - c))) & (span - 1)
        } else {
            ((full >> c) | (full << (wide - c))) & (span - 1)
        };
        self.flags.cf = rot & 1 == 1;
        let res = ((rot >> 1) as u64) & mask;
        // OF is architecturally defined only for a count of one, where it is
        // MSB(result) ^ CF for rcl and CF ^ MSB(input) for rcr. Measured on
        // hardware (tools/check_shift_flags.py) the same expression holds for
        // every count: the XOR of the two ring bits the last step moved across
        // the carry boundary. Nothing else in the flags depends on the count.
        let lo = if kind == 5 {
            (bits - c) % wide
        } else {
            (bits + c) % wide
        };
        let hi = if kind == 5 {
            (lo + 1) % wide
        } else {
            (lo + wide - 1) % wide
        };
        self.flags.of = (full >> lo & 1) != (full >> hi & 1);
        res
    }

    /// `shld`: shift `value` left, feeding in the high bits of `fill`.
    fn shld(&mut self, value: u64, fill: u64, count: u8, bits: u32) -> u64 {
        let mask = width_mask(bits);
        let v = value & mask;
        let f = fill & mask;
        let c = count as u32 & if bits == 64 { 0x3F } else { 0x1F };
        if c == 0 {
            return v;
        }
        let res;
        if c <= bits {
            res = ((((v as u128) << c) | ((f as u128) >> (bits - c))) as u64) & mask;
            self.flags.cf = (v >> (bits - c)) & 1 == 1;
        } else {
            // Reachable only for a 16-bit operand, whose count is masked to five
            // bits and so can exceed the width. The architecture leaves that
            // undefined; the behaviour both the reference and the silicon
            // exhibit is the natural one from a double-width shifter: treat
            // `fill:value` as a single field, rotate it left, keep the low half.
            let span = 2 * bits;
            let wide = ((f as u128) << bits) | v as u128;
            res = (((wide << c) | (wide >> (span - c))) as u64) & mask;
            self.flags.cf = (f >> (span - c)) & 1 == 1;
        }
        // shld's overflow is the shift-left formula: result sign XOR carry.
        // A count of exactly the operand size is the one exception measured on
        // hardware (tools/check_shift_flags.py): the whole register has been
        // shifted out, so the overflow is the carry itself.
        self.flags.of = if c == bits {
            self.flags.cf
        } else {
            ((res >> (bits - 1)) & 1 == 1) != self.flags.cf
        };
        self.szp(res, bits);
        res
    }

    /// `shrd`: shift `value` right, feeding in the low bits of `fill`.
    fn shrd(&mut self, value: u64, fill: u64, count: u8, bits: u32) -> u64 {
        let mask = width_mask(bits);
        let v = value & mask;
        let f = fill & mask;
        let c = count as u32 & if bits == 64 { 0x3F } else { 0x1F };
        if c == 0 {
            return v;
        }
        let res;
        if c <= bits {
            res = ((((f as u128) << (bits - c)) | ((v as u128) >> c)) as u64) & mask;
            self.flags.cf = (v >> (c - 1)) & 1 == 1;
        } else {
            // Undefined on the architecture, as in `shld`: the double-width
            // field rotated right, keeping the low half, with the carry taken
            // from the last bit the rotation dropped.
            let span = 2 * bits;
            let wide = ((f as u128) << bits) | v as u128;
            res = (((wide >> c) | (wide << (span - c))) as u64) & mask;
            self.flags.cf = (wide >> (c - 1)) & 1 == 1;
        }
        // shrd's overflow compares the top two bits of the result, mirroring the
        // shift-right formula rather than the shift-left one.
        self.flags.of =
            ((res >> (bits - 1)) & 1 == 1) != ((res >> (bits - 2)) & 1 == 1);
        self.szp(res, bits);
        res
    }

    fn shift(&mut self, kind: u8, value: u64, count: u8, bits: u32) -> u64 {
        let mask = width_mask(bits);
        let v = value & mask;
        let mut c = count as u32;
        c &= if bits == 64 { 0x3F } else { 0x1F };
        if c == 0 {
            return v;
        }
        let sign = (v >> (bits - 1)) & 1 == 1;
        let res;

        match kind {
            0 => {
                if c < bits {
                    let wide = (v as u128) << c;
                    res = (wide as u64) & mask;
                    self.flags.cf = (wide >> bits) & 1 == 1;
                } else {
                    res = 0;
                    self.flags.cf = c == bits && v & 1 == 1;
                }
                // OF is architecturally undefined for a count other than one,
                // but hardware keeps computing the count-1 formula anyway
                // (MSB of the result XOR the carry), measured on the real CPU
                // and matched by the differential reference. Computing it
                // always beats leaking a stale bit from an earlier
                // instruction.
                self.flags.of = ((res >> (bits - 1)) & 1 == 1) != self.flags.cf;
                self.szp(res, bits);
            }
            1 => {
                if c < bits {
                    res = v >> c;
                    self.flags.cf = (v >> (c - 1)) & 1 == 1;
                } else {
                    res = 0;
                    self.flags.cf = c == bits && sign;
                }
                self.flags.of = if c == 1 { sign } else { false };
                self.szp(res, bits);
            }
            2 => {
                let sv = Self::sign_extend(v, bits) as i64;
                if c < bits {
                    res = ((sv >> c) as u64) & mask;
                    self.flags.cf = ((sv >> (c - 1)) & 1) == 1;
                } else {
                    res = if sign { mask } else { 0 };
                    self.flags.cf = sign;
                }
                self.flags.of = false;
                self.szp(res, bits);
            }
            3 => {
                // A masked count of zero left earlier, so for a 16-bit operand
                // `cc` is already zero -- the rotation is a no-op but CF and OF
                // still get evaluated, as hardware does for count 16.
                let cc = c % bits;
                res = if cc == 0 {
                    v
                } else {
                    ((v << cc) | (v >> (bits - cc))) & mask
                };
                self.flags.cf = res & 1 == 1;
                // Rotates touch only CF and OF; SF, ZF, PF, AF are unaffected.
                // As with shl, hardware evaluates the count-1 formula for every
                // count: the result sign XOR the carry (measured natively).
                self.flags.of = ((res >> (bits - 1)) & 1 == 1) != self.flags.cf;
            }
            _ => {
                let cc = c % bits;
                res = if cc == 0 {
                    v
                } else {
                    ((v >> cc) | (v << (bits - cc))) & mask
                };
                self.flags.cf = (res >> (bits - 1)) & 1 == 1;
                // ror compares the top two bits of the result, again for every
                // count; SF/ZF/PF are left alone, as on hardware.
                self.flags.of =
                    ((res >> (bits - 1)) & 1 == 1) != ((res >> (bits - 2)) & 1 == 1);
            }
        }
        res
    }

    fn cond(&self, code: usize) -> bool {
        match code {
            0x0 => self.flags.of,
            0x1 => !self.flags.of,
            0x2 => self.flags.cf,
            0x3 => !self.flags.cf,
            0x4 => self.flags.zf,
            0x5 => !self.flags.zf,
            0x6 => self.flags.cf || self.flags.zf,
            0x7 => !self.flags.cf && !self.flags.zf,
            0x8 => self.flags.sf,
            0x9 => !self.flags.sf,
            0xA => self.flags.pf,
            0xB => !self.flags.pf,
            0xC => self.flags.sf != self.flags.of,
            0xD => self.flags.sf == self.flags.of,
            0xE => self.flags.zf || (self.flags.sf != self.flags.of),
            0xF => !self.flags.zf && (self.flags.sf == self.flags.of),
            _ => false,
        }
    }

    // -------------------------------------------------------------- instruction

    pub fn step(&mut self) -> Stop {
        let start = self.rip;
        if start.wrapping_add(15) >= self.mem.size() {
            return Stop::BadOpcode;
        }

        // Prefixes
        let mut p66 = false;
        let mut p_f2 = false; // REPNE, or the mandatory F2 prefix of an SSE opcode
        let mut p_f3 = false; // REP/REPE, or the mandatory F3 prefix of an SSE opcode
        let mut has_rex = false;
        let mut rex: u8 = 0;
        let opcode: u8 = loop {
            let b = self.fetch8();
            match b {
                0x66 => p66 = true,
                0x67 => {}                                   // address-size override: ignored
                0xF0 => {}                                   // LOCK: single-core core, already atomic
                0xF2 => p_f2 = true,
                0xF3 => p_f3 = true,
                0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 => {} // segment overrides: flat memory
                0x40..=0x4F => {
                    rex = b;
                    has_rex = true;
                }
                _ => break b,
            }
        };

        let rexw = rex & 0x08 != 0;
        let opsz: u32 = if rexw {
            64
        } else if p66 {
            16
        } else {
            32
        };

        self.last_insn = start;
        self.last_opcode = opcode;
        self.last_ext = None;

        if self.trace {
            eprintln!("{start:08x}  op={opcode:02x} rex={rex:02x}");
        }

        match opcode {
            // ---------------------------------------------------- ALU group
            0x00..=0x3F if (opcode & 0x07) <= 0x05 => {
                let aluop = opcode >> 3;
                let form = opcode & 0x07;
                match form {
                    0 | 1 => {
                        let bits = if form == 0 { 8 } else { opsz };
                        let (dst, reg) = self.decode_rm(bits, rex, has_rex);
                        let a = self.read_op(&dst);
                        let b = self.read_op(&reg);
                        let res = self.alu(aluop, a, b, bits);
                        if aluop != 7 {
                            self.write_op(&dst, res);
                        }
                    }
                    2 | 3 => {
                        let bits = if form == 2 { 8 } else { opsz };
                        let (src, reg) = self.decode_rm(bits, rex, has_rex);
                        let a = self.read_op(&reg);
                        let b = self.read_op(&src);
                        let res = self.alu(aluop, a, b, bits);
                        if aluop != 7 {
                            self.write_op(&reg, res);
                        }
                    }
                    4 => {
                        let imm = self.fetch8();
                        let a = self.get_reg(RAX, 8);
                        let res = self.alu(aluop, a, imm as u64, 8);
                        if aluop != 7 {
                            self.set_reg(RAX, res, 8);
                        }
                    }
                    _ => {
                        let imm = if opsz == 64 {
                            self.fetch32() as i32 as i64 as u64
                        } else {
                            self.fetch_imm(opsz)
                        };
                        let a = self.get_reg(RAX, opsz);
                        let res = self.alu(aluop, a, imm, opsz);
                        if aluop != 7 {
                            self.set_reg(RAX, res, opsz);
                        }
                    }
                }
                Stop::None
            }

            // --------------------------------------------------- push/pop reg
            0x50..=0x57 => {
                let idx = (opcode & 7) as usize | ((rex & 1) as usize) << 3;
                let bits = if p66 { 16 } else { 64 };
                let v = self.get_reg(idx, bits);
                self.push(v, bits);
                Stop::None
            }
            0x58..=0x5F => {
                let idx = (opcode & 7) as usize | ((rex & 1) as usize) << 3;
                let bits = if p66 { 16 } else { 64 };
                let v = self.pop(bits);
                self.set_reg(idx, v, bits);
                Stop::None
            }

            0x63 => {
                // movsxd r64, r/m32
                let (src, mut reg) = self.decode_rm(32, rex, has_rex);
                let v = self.read_op(&src) as u32 as i32 as i64 as u64;
                widen_destination(&mut reg, opsz);
                self.write_op(&reg, v);
                Stop::None
            }
            0x68 => {
                if p66 {
                    let imm = self.fetch16() as u64;
                    self.push(imm, 16);
                } else {
                    let imm = self.fetch32() as i32 as i64 as u64;
                    self.push(imm, 64);
                }
                Stop::None
            }
            0x6A => {
                let imm = self.fetch8() as i8 as i64 as u64;
                self.push(imm, if p66 { 16 } else { 64 });
                Stop::None
            }
            0x69 | 0x6B => {
                let bits = opsz;
                let (src, reg) = self.decode_rm(bits, rex, has_rex);
                let imm = if opcode == 0x6B {
                    self.fetch8() as i8 as i64 as u64
                } else if bits == 64 {
                    self.fetch32() as i32 as i64 as u64
                } else {
                    self.fetch_imm(bits)
                };
                let a = self.read_op(&src);
                let product = (Self::sign_extend(a, bits) as i128).wrapping_mul(imm as i128);
                let res = (product as u64) & width_mask(bits);
                // Two/three-operand imul only defines CF and OF: they report
                // whether the full product fits back into the destination.
                let fits = product == Self::sign_extend(res, bits) as i128;
                self.flags.cf = !fits;
                self.flags.of = !fits;
                self.write_op(&reg, res);
                Stop::None
            }

            // ---------------------------------------------------------- jcc
            0x70..=0x7F => {
                let rel = self.fetch8() as i8 as i64 as u64;
                if self.cond((opcode & 0x0F) as usize) {
                    self.rip = self.rip.wrapping_add(rel);
                }
                Stop::None
            }

            // -------------------------------------------------- group 1 ALU
            0x80 | 0x81 | 0x82 | 0x83 => {
                let bits = if opcode == 0x80 || opcode == 0x82 {
                    8
                } else {
                    opsz
                };
                let (dst, reg) = self.decode_rm(bits, rex, has_rex);
                let group = reg.idx & 7;
                let imm = match opcode {
                    0x80 | 0x82 => self.fetch8() as u64,
                    0x81 => {
                        if opsz == 64 {
                            self.fetch32() as i32 as i64 as u64
                        } else {
                            self.fetch_imm(opsz)
                        }
                    }
                    _ => self.fetch8() as i8 as i64 as u64,
                };
                let a = self.read_op(&dst);
                let res = self.alu(group as u8, a, imm, bits);
                if group != 7 {
                    self.write_op(&dst, res);
                }
                Stop::None
            }

            0x84 | 0x85 => {
                let bits = if opcode == 0x84 { 8 } else { opsz };
                let (src, reg) = self.decode_rm(bits, rex, has_rex);
                let res = self.read_op(&src) & self.read_op(&reg);
                self.flags.cf = false;
                self.flags.of = false;
                self.flags.af = false;
                self.szp(res, bits);
                Stop::None
            }
            0x86 | 0x87 => {
                let bits = if opcode == 0x86 { 8 } else { opsz };
                let (dst, reg) = self.decode_rm(bits, rex, has_rex);
                let a = self.read_op(&dst);
                let b = self.read_op(&reg);
                self.write_op(&dst, b);
                self.write_op(&reg, a);
                Stop::None
            }
            0x88 | 0x89 => {
                let bits = if opcode == 0x88 { 8 } else { opsz };
                let (dst, reg) = self.decode_rm(bits, rex, has_rex);
                let v = self.read_op(&reg);
                self.write_op(&dst, v);
                Stop::None
            }
            0x8A | 0x8B => {
                let bits = if opcode == 0x8A { 8 } else { opsz };
                let (src, reg) = self.decode_rm(bits, rex, has_rex);
                let v = self.read_op(&src);
                self.write_op(&reg, v);
                Stop::None
            }
            0x8D => {
                let (src, reg) = self.decode_rm(opsz, rex, has_rex);
                if src.is_reg {
                    let v = self.read_op(&src);
                    self.write_op(&reg, v);
                } else {
                    self.set_reg(reg.idx, src.addr, 64);
                }
                Stop::None
            }
            0x8F => {
                let (dst, _) = self.decode_rm(64, rex, has_rex);
                let v = self.pop(64);
                self.write_op(&dst, v);
                Stop::None
            }

            0x90..=0x97 => {
                // Note: a 0x66 override makes this a 16-bit exchange, per Intel's
                // operand-size rules. QEMU (and therefore Unicorn) ignores the
                // prefix on this encoding, so the fuzzer does not generate the
                // 16-bit form; hardware is the reference here.
                let idx = (opcode & 7) as usize | ((rex & 1) as usize) << 3;
                if opcode == 0x90 && idx == RAX {
                    // nop (`pause` with an F3 prefix is the same instruction here)
                } else {
                    // The exchange happens at the operand width, so the 32-bit
                    // form still zero-extends both registers.
                    let bits = if p66 {
                        16
                    } else if rexw {
                        64
                    } else {
                        32
                    };
                    let a = self.get_reg(RAX, bits);
                    let b = self.get_reg(idx, bits);
                    self.set_reg(RAX, b, bits);
                    self.set_reg(idx, a, bits);
                }
                Stop::None
            }
            0x98 => {
                match opsz {
                    16 => {
                        let v = self.get_reg(RAX, 8) as i8 as i16 as u64;
                        self.set_reg(RAX, v, 16);
                    }
                    64 => {
                        let v = self.get_reg(RAX, 32) as i32 as i64 as u64;
                        self.set_reg(RAX, v, 64);
                    }
                    _ => {}
                }
                Stop::None
            }
            0x99 => {
                match opsz {
                    16 => {
                        let sign = (self.get_reg(RAX, 16) as i16) < 0;
                        self.set_reg(RDX, if sign { 0xFFFF } else { 0 }, 16);
                    }
                    32 => {
                        let sign = (self.get_reg(RAX, 32) as i32) < 0;
                        self.set_reg(RDX, if sign { 0xFFFF_FFFF } else { 0 }, 32);
                    }
                    _ => {
                        let sign = (self.get_reg(RAX, 64) as i64) < 0;
                        self.set_reg(RDX, if sign { u64::MAX } else { 0 }, 64);
                    }
                }
                Stop::None
            }
            0x9C => {
                let v = self.flags.pack();
                self.push(v, if p66 { 16 } else { 64 });
                Stop::None
            }
            0x9D => {
                let bits = if p66 { 16 } else { 64 };
                let v = self.pop(bits);
                self.flags = Flags::unpack(v);
                Stop::None
            }

            // ------------------------------------------- string instructions
            0xA4..=0xA7 | 0xAA..=0xAF => {
                let bits = match opcode {
                    0xA4 | 0xA6 | 0xAA | 0xAC | 0xAE => 8,
                    _ => opsz,
                };
                let kind = match opcode {
                    0xA4 | 0xA5 => 0, // movs
                    0xAA | 0xAB => 1, // stos
                    0xAC | 0xAD => 2, // lods
                    0xA6 | 0xA7 => 3, // cmps
                    _ => 4,           // scas
                };
                let rep = if p_f3 {
                    Some(true)
                } else if p_f2 {
                    Some(false)
                } else {
                    None
                };
                self.string_op(kind, bits, rep)
            }

            0xA8 | 0xA9 => {
                let bits = if opcode == 0xA8 { 8 } else { opsz };
                let imm = if bits == 64 {
                    self.fetch32() as i32 as i64 as u64
                } else {
                    self.fetch_imm(bits)
                };
                let res = self.get_reg(RAX, bits) & imm;
                self.flags.cf = false;
                self.flags.of = false;
                self.flags.af = false;
                self.szp(res, bits);
                Stop::None
            }

            0xB0..=0xB7 => {
                let idx = (opcode & 7) as usize | ((rex & 1) as usize) << 3;
                let imm = self.fetch8() as u64;
                if !has_rex && idx >= 4 {
                    let reg = Op {
                        bits: 8,
                        is_reg: true,
                        idx,
                        addr: 0,
                        high8: true,
                    };
                    self.write_op(&reg, imm);
                } else {
                    self.set_reg(idx, imm, 8);
                }
                Stop::None
            }
            0xB8..=0xBF => {
                let idx = (opcode & 7) as usize | ((rex & 1) as usize) << 3;
                let imm = self.fetch_imm(opsz);
                self.set_reg(idx, imm, opsz);
                Stop::None
            }

            // -------------------------------------------------- shift group
            0xC0 | 0xC1 | 0xD0 | 0xD1 | 0xD2 | 0xD3 => {
                let bits = if opcode == 0xC0 || opcode == 0xD0 || opcode == 0xD2 {
                    8
                } else {
                    opsz
                };
                let (dst, reg) = self.decode_rm(bits, rex, has_rex);
                let kind = match reg.idx & 7 {
                    0 => 3u8, // rol
                    1 => 4,   // ror
                    2 => 5,   // rcl
                    3 => 6,   // rcr
                    4 | 6 => 0, // shl/sal
                    5 => 1,   // shr
                    _ => 2,   // sar
                };
                let count = match opcode {
                    0xD0 | 0xD1 => 1u8,
                    0xD2 | 0xD3 => self.get_reg(RCX, 8) as u8,
                    _ => self.fetch8(),
                };
                let v = self.read_op(&dst);
                let res = if kind >= 5 {
                    self.rotate_carry(kind, v, count, bits)
                } else {
                    self.shift(kind, v, count, bits)
                };
                self.write_op(&dst, res);
                Stop::None
            }

            0xC2 => {
                let imm = self.fetch16() as u64;
                let target = self.pop(64);
                self.rip = target;
                self.r[RSP] = self.r[RSP].wrapping_add(imm);
                Stop::None
            }
            0xC3 => {
                let target = self.pop(64);
                self.rip = target;
                Stop::None
            }
            0xC6 | 0xC7 => {
                let bits = if opcode == 0xC6 { 8 } else { opsz };
                let (dst, _) = self.decode_rm(bits, rex, has_rex);
                let imm = if bits == 64 {
                    self.fetch32() as i32 as i64 as u64
                } else {
                    self.fetch_imm(bits)
                };
                self.write_op(&dst, imm);
                Stop::None
            }
            0xC9 => {
                self.r[RSP] = self.r[RBP];
                let v = self.pop(64);
                self.r[RBP] = v;
                Stop::None
            }
            0xCC => Stop::Int3,

            0xE8 => {
                let rel = self.fetch32() as i32 as i64 as u64;
                let ret = self.rip;
                self.push(ret, 64);
                self.rip = self.rip.wrapping_add(rel);
                Stop::None
            }
            0xE9 => {
                let rel = self.fetch32() as i32 as i64 as u64;
                self.rip = self.rip.wrapping_add(rel);
                Stop::None
            }
            0xEB => {
                let rel = self.fetch8() as i8 as i64 as u64;
                self.rip = self.rip.wrapping_add(rel);
                Stop::None
            }

            0xF4 => Stop::Halted,

            0xF6 | 0xF7 => {
                let bits = if opcode == 0xF6 { 8 } else { opsz };
                self.group3(bits, rex, has_rex)
            }

            0xF8..=0xFD => {
                match opcode {
                    0xF8 => self.flags.cf = false,
                    0xF9 => self.flags.cf = true,
                    0xFA => self.flags.cf = !self.flags.cf,
                    0xFC => self.flags.df = false,
                    0xFD => self.flags.df = true,
                    _ => {}
                }
                Stop::None
            }

            0xFE | 0xFF => self.group45(opcode, opsz, rex, has_rex),

            0x0F => self.two_byte(opsz, rex, has_rex, p66, p_f2, p_f3),

            _ => Stop::BadOpcode,
        }
    }

    // ----------------------------------------------------- string instructions

    /// `kind`: 0 movs, 1 stos, 2 lods, 3 cmps, 4 scas.
    ///
    /// `rep`: `Some(true)` for REP/REPE (F3), `Some(false)` for REPNE (F2),
    /// `None` for a single iteration. Pointer direction comes from DF, and RCX
    /// is left holding the number of iterations that did not run, exactly as
    /// the architecture specifies.
    fn string_op(&mut self, kind: u8, bits: u32, rep: Option<bool>) -> Stop {
        let size = (bits / 8) as u64;
        let mut count = if rep.is_some() { self.r[RCX] } else { 1 };
        if count == 0 {
            return Stop::None;
        }
        // A `rep movs` with a huge RCX would otherwise spin the host inside a
        // single instruction; the flat window is a natural bound.
        let cap = (crate::memory::MEM_SIZE as u64 / size).max(1);
        let mut done: u64 = 0;
        loop {
            match kind {
                0 => {
                    let v = self.mem.read(self.r[RSI], bits);
                    self.mem.write(self.r[RDI], v, bits);
                }
                1 => {
                    let v = self.get_reg(RAX, bits);
                    self.mem.write(self.r[RDI], v, bits);
                }
                2 => {
                    let v = self.mem.read(self.r[RSI], bits);
                    self.set_reg(RAX, v, bits);
                }
                3 => {
                    let a = self.mem.read(self.r[RSI], bits);
                    let b = self.mem.read(self.r[RDI], bits);
                    self.alu(7, a, b, bits);
                }
                _ => {
                    let a = self.get_reg(RAX, bits);
                    let b = self.mem.read(self.r[RDI], bits);
                    self.alu(7, a, b, bits);
                }
            }
            let delta = if self.flags.df {
                size.wrapping_neg()
            } else {
                size
            };
            if kind == 0 || kind == 2 || kind == 3 {
                self.r[RSI] = self.r[RSI].wrapping_add(delta);
            }
            if kind != 2 {
                self.r[RDI] = self.r[RDI].wrapping_add(delta);
            }
            count -= 1;
            done += 1;
            if count == 0 {
                break;
            }
            if let Some(want_equal) = rep {
                // REPE/REPNE stop as soon as the comparison flags disagree.
                if kind >= 3 && self.flags.zf != want_equal {
                    break;
                }
            }
            if done >= cap {
                self.set_reg(RCX, count, 64);
                return Stop::StepLimit;
            }
        }
        if rep.is_some() {
            self.set_reg(RCX, count, 64);
        }
        Stop::None
    }

    // ----------------------------------------------------------- vector (SSE)

    fn read_mem128(&self, addr: u64) -> u128 {
        (self.mem.read(addr, 64) as u128) | ((self.mem.read(addr.wrapping_add(8), 64) as u128) << 64)
    }

    fn write_mem128(&mut self, addr: u64, value: u128) {
        self.mem.write(addr, value as u64, 64);
        self.mem.write(addr.wrapping_add(8), (value >> 64) as u64, 64);
    }

    fn xmm_rm_read(&self, rm: &Op) -> u128 {
        if rm.is_reg {
            self.xmm[rm.idx]
        } else {
            self.read_mem128(rm.addr)
        }
    }

    fn xmm_rm_write(&mut self, rm: &Op, value: u128) {
        if rm.is_reg {
            self.xmm[rm.idx] = value;
        } else {
            self.write_mem128(rm.addr, value);
        }
    }

    fn gpr_rm_read(&self, rm: &Op, bits: u32) -> u64 {
        if rm.is_reg {
            self.get_reg(rm.idx, bits)
        } else {
            self.mem.read(rm.addr, bits)
        }
    }

    /// COMISS/UCOMISS/COMISD/UCOMISD flag results (ZF/PF/CF only).
    fn comis(&mut self, a: f64, b: f64) {
        self.flags.of = false;
        self.flags.sf = false;
        self.flags.af = false;
        if a.is_nan() || b.is_nan() {
            self.flags.zf = true;
            self.flags.pf = true;
            self.flags.cf = true;
        } else if a > b {
            self.flags.zf = false;
            self.flags.pf = false;
            self.flags.cf = false;
        } else if a < b {
            self.flags.zf = false;
            self.flags.pf = false;
            self.flags.cf = true;
        } else {
            self.flags.zf = true;
            self.flags.pf = false;
            self.flags.cf = false;
        }
    }

    fn sse(&mut self, b2: u8, rex: u8, has_rex: bool, p66: bool, p_f2: bool, p_f3: bool) -> Stop {
        let rexw = rex & 0x08 != 0;
        match b2 {
            // -------------------------------------------------------- moves
            0x10 | 0x11 => {
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let load = b2 == 0x10;
                if p_f3 {
                    // movss
                    if load {
                        let src = self.xmm_rm_read(&rm);
                        if rm.is_reg {
                            let v = with_f32_lane(self.xmm[reg.idx], 0, f32_lane(src, 0));
                            self.xmm[reg.idx] = v;
                        } else {
                            self.xmm[reg.idx] = src & 0xFFFF_FFFF;
                        }
                    } else if rm.is_reg {
                        let v = with_f32_lane(self.xmm[rm.idx], 0, f32_lane(self.xmm[reg.idx], 0));
                        self.xmm[rm.idx] = v;
                    } else {
                        let v = self.xmm[reg.idx] as u64;
                        self.mem.write(rm.addr, v, 32);
                    }
                } else if p_f2 {
                    // movsd
                    if load {
                        let src = self.xmm_rm_read(&rm);
                        if rm.is_reg {
                            let v = with_f64_lane(self.xmm[reg.idx], 0, f64_lane(src, 0));
                            self.xmm[reg.idx] = v;
                        } else {
                            self.xmm[reg.idx] = src & 0xFFFF_FFFF_FFFF_FFFF;
                        }
                    } else if rm.is_reg {
                        let v = with_f64_lane(self.xmm[rm.idx], 0, f64_lane(self.xmm[reg.idx], 0));
                        self.xmm[rm.idx] = v;
                    } else {
                        let v = self.xmm[reg.idx] as u64;
                        self.mem.write(rm.addr, v, 64);
                    }
                } else if load {
                    // movups / movupd: a full 128-bit load
                    let v = self.xmm_rm_read(&rm);
                    self.xmm[reg.idx] = v;
                } else {
                    // movups / movupd store
                    let v = self.xmm[reg.idx];
                    self.xmm_rm_write(&rm, v);
                }
                Stop::None
            }
            0x28 | 0x29 => {
                // movaps/movapd: identical to movups/movupd except for the
                // alignment requirement, which we do not fault on yet.
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                if b2 == 0x28 {
                    let v = self.xmm_rm_read(&rm);
                    self.xmm[reg.idx] = v;
                } else {
                    let v = self.xmm[reg.idx];
                    self.xmm_rm_write(&rm, v);
                }
                Stop::None
            }
            0x6F | 0x7F => {
                // movdqa / movdqu (66 or F3). The plain form is MMX, which we
                // do not implement, so it must not be silently accepted.
                if !(p66 || p_f3) {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                if b2 == 0x6F {
                    let v = self.xmm_rm_read(&rm);
                    self.xmm[reg.idx] = v;
                } else {
                    let v = self.xmm[reg.idx];
                    self.xmm_rm_write(&rm, v);
                }
                Stop::None
            }
            0x12 | 0x13 | 0x16 | 0x17 => {
                // movlps / movhps (and movlpd / movhpd under 66): one 64-bit
                // half of the register, memory operand only. The register forms
                // of these opcodes are movhlps/movlhps, which we do not model,
                // so refusing them keeps MOVHLPS from silently doing the wrong
                // thing.
                // F2/F3 turn these opcodes into movddup/movsldup/movshdup, which
                // are different instructions entirely.
                if p_f2 || p_f3 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(64, rex, has_rex);
                if rm.is_reg {
                    return Stop::BadOpcode;
                }
                let high_half = b2 == 0x16 || b2 == 0x17;
                let load = b2 == 0x12 || b2 == 0x16;
                if load {
                    let src = self.gpr_rm_read(&rm, 64);
                    let dst = self.xmm[reg.idx];
                    let v = if high_half {
                        (dst & 0xFFFF_FFFF_FFFF_FFFF) | ((src as u128) << 64)
                    } else {
                        (dst & !0xFFFF_FFFF_FFFF_FFFFu128) | src as u128
                    };
                    self.xmm[reg.idx] = v;
                } else {
                    let v = if high_half {
                        (self.xmm[reg.idx] >> 64) as u64
                    } else {
                        self.xmm[reg.idx] as u64
                    };
                    self.mem.write(rm.addr, v, 64);
                }
                Stop::None
            }
            0x6E => {
                // movd / movq into an XMM register (66 prefix, REX.W for 64-bit).
                // The bare form is MMX MOVD, which is not implemented.
                if !p66 {
                    return Stop::BadOpcode;
                }
                let bits = if rexw { 64 } else { 32 };
                let (rm, reg) = self.decode_rm(bits, rex, has_rex);
                let v = self.gpr_rm_read(&rm, bits);
                self.xmm[reg.idx] = v as u128;
                Stop::None
            }
            0x7E | 0xD6 => {
                if b2 == 0x7E && p_f3 {
                    // movq xmm, xmm/m64: load the low 64 bits into the
                    // destination and zero the upper half.
                    let (rm, reg) = self.decode_rm(64, rex, has_rex);
                    let v = if rm.is_reg {
                        self.xmm[rm.idx] as u64
                    } else {
                        self.mem.read(rm.addr, 64)
                    };
                    self.xmm[reg.idx] = v as u128;
                } else if p66 {
                    let bits = if b2 == 0xD6 || rexw { 64 } else { 32 };
                    let (rm, reg) = self.decode_rm(bits, rex, has_rex);
                    let v = self.xmm[reg.idx] as u64;
                    if rm.is_reg {
                        self.set_reg(rm.idx, v, bits);
                    } else {
                        self.mem.write(rm.addr, v, bits);
                    }
                } else {
                    // MMX MOVD/MOVQ: not implemented.
                    return Stop::BadOpcode;
                }
                Stop::None
            }
            0x14 | 0x15 => {
                // unpcklps / unpckhps, and the pd forms under 66
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = if p66 {
                    let lane = if b2 == 0x14 { 0 } else { 1 };
                    let lo = f64_lane(dst, lane);
                    let hi = f64_lane(src, lane);
                    with_f64_lane(with_f64_lane(0, 0, lo), 1, hi)
                } else {
                    let base = if b2 == 0x14 { 0 } else { 2 };
                    let mut o = 0u128;
                    o = with_f32_lane(o, 0, f32_lane(dst, base));
                    o = with_f32_lane(o, 1, f32_lane(src, base));
                    o = with_f32_lane(o, 2, f32_lane(dst, base + 1));
                    o = with_f32_lane(o, 3, f32_lane(src, base + 1));
                    o
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x60 | 0x61 | 0x62 | 0x68 | 0x69 | 0x6A | 0x6C | 0x6D => {
                // The punpck family (66). The plain forms are MMX, which is not
                // implemented, so they must not be silently accepted.
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let width: u32 = match b2 {
                    0x60 | 0x68 => 8,
                    0x61 | 0x69 => 16,
                    0x62 | 0x6A => 32,
                    _ => 64,
                };
                let high = matches!(b2, 0x68 | 0x69 | 0x6A | 0x6D);
                let per = 128 / width;
                let start = if high { per / 2 } else { 0 };
                let mask = (1u128 << width) - 1;
                let mut out = 0u128;
                let mut i = 0;
                while i < per / 2 {
                    let lane = start + i;
                    let a = (dst >> (width * lane)) & mask;
                    let b = (src >> (width * lane)) & mask;
                    out |= a << (width * (2 * i));
                    out |= b << (width * (2 * i + 1));
                    i += 1;
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // ------------------------------- packed integer bitwise and adds
            0x54 | 0x55 | 0x56 | 0x57 => {
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = match b2 {
                    0x54 => dst & src,          // andps/andpd
                    0x55 => (!dst) & src,       // andnps/andnpd
                    0x56 => dst | src,          // orps/orpd
                    _ => dst ^ src,             // xorps/xorpd
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0xDB | 0xDF | 0xEB | 0xEF => {
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = match b2 {
                    0xDB => dst & src,          // pand
                    0xDF => (!dst) & src,       // pandn
                    0xEB => dst | src,          // por
                    _ => dst ^ src,             // pxor
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x74 | 0x75 | 0x76 => {
                // pcmpeqb / pcmpeqw / pcmpeqd
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let width = match b2 {
                    0x74 => 8,
                    0x75 => 16,
                    _ => 32,
                };
                let mut out = 0u128;
                let mut bit = 0;
                while bit < 128 {
                    let mask = ((1u128 << width) - 1) << bit;
                    if dst & mask == src & mask {
                        out |= mask;
                    }
                    bit += width;
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0xD4 | 0xFA | 0xFB | 0xFE => {
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = match b2 {
                    0xD4 => {
                        // paddq: two independent 64-bit adds
                        let lo = (dst as u64).wrapping_add(src as u64) as u128;
                        let hi = ((dst >> 64) as u64).wrapping_add((src >> 64) as u64) as u128;
                        lo | (hi << 64)
                    }
                    0xFB => {
                        let lo = (dst as u64).wrapping_sub(src as u64) as u128;
                        let hi = ((dst >> 64) as u64).wrapping_sub((src >> 64) as u64) as u128;
                        lo | (hi << 64)
                    }
                    0xFE | 0xFA => {
                        // paddd / psubd: four independent 32-bit lanes
                        let mut o = 0u128;
                        for lane in 0..4u32 {
                            let a = (dst >> (32 * lane)) as u32;
                            let b = (src >> (32 * lane)) as u32;
                            let r = if b2 == 0xFE {
                                a.wrapping_add(b)
                            } else {
                                a.wrapping_sub(b)
                            };
                            o |= (r as u128) << (32 * lane);
                        }
                        o
                    }
                    _ => unreachable!(),
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x70 => {
                // pshufd (66), pshufhw (F3), pshuflw (F2). The bare form is MMX
                // pshufw and is not implemented.
                if !(p66 || p_f3 || p_f2) {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let imm = self.fetch8();
                let out = if p66 {
                    // Four 32-bit lanes, any permutation.
                    let mut out = 0u128;
                    for lane in 0..4u32 {
                        let from = ((imm >> (2 * lane)) & 3) as u32;
                        out = with_f32_lane(out, lane, f32_lane(src, from));
                    }
                    out
                } else if p_f3 {
                    // pshufhw: low 64 bits copied, high four words shuffled.
                    let mut out = src & 0xFFFF_FFFF_FFFF_FFFF;
                    for lane in 0..4u32 {
                        let from = ((imm >> (2 * lane)) & 3) as u32;
                        out = with_u16_lane(out, 4 + lane, u16_lane(src, 4 + from));
                    }
                    out
                } else {
                    // pshuflw: high 64 bits copied, low four words shuffled.
                    let mut out = src & !0xFFFF_FFFF_FFFF_FFFFu128;
                    for lane in 0..4u32 {
                        let from = ((imm >> (2 * lane)) & 3) as u32;
                        out = with_u16_lane(out, lane, u16_lane(src, from));
                    }
                    out
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x71 | 0x72 | 0x73 => {
                // Packed shifts by an immediate; the ModRM.reg field selects
                // the operation. The bare forms are MMX and are not implemented.
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                // These encodings only exist with a register destination (GAS
                // rejects `psrlw $3, (%rax)`), even though the ModRM byte can
                // spell a memory operand. Refuse the undefined form instead of
                // inventing a read-modify-write; QEMU instead ignores the mod
                // field and shifts the register named by r/m.
                if !rm.is_reg {
                    return Stop::BadOpcode;
                }
                let group = reg.idx & 7;
                let imm = self.fetch8() as u32;
                let v = self.xmm_rm_read(&rm);
                let out = match (b2, group) {
                    (0x71, 2) => {
                        let mut o = 0u128;
                        for i in 0..8u32 {
                            let x = (v >> (16 * i)) as u16;
                            let y = if imm >= 16 { 0 } else { x >> imm };
                            o |= (y as u128) << (16 * i);
                        }
                        o
                    }
                    (0x71, 4) => {
                        let mut o = 0u128;
                        for i in 0..8u32 {
                            let x = (v >> (16 * i)) as u16 as i16;
                            let y = if imm >= 16 {
                                if x < 0 { 0xFFFFu16 } else { 0 }
                            } else {
                                (x >> imm) as u16
                            };
                            o |= (y as u128) << (16 * i);
                        }
                        o
                    }
                    (0x71, 6) => {
                        let mut o = 0u128;
                        for i in 0..8u32 {
                            let x = (v >> (16 * i)) as u16;
                            let y = if imm >= 16 { 0 } else { x << imm };
                            o |= (y as u128) << (16 * i);
                        }
                        o
                    }
                    (0x72, 2) | (0x72, 4) | (0x72, 6) => {
                        let mut o = 0u128;
                        for i in 0..4u32 {
                            let raw = (v >> (32 * i)) as u32;
                            let x = raw as i32;
                            let y = match group {
                                2 => {
                                    if imm >= 32 {
                                        0
                                    } else {
                                        (raw >> imm) as i32
                                    }
                                }
                                4 => {
                                    if imm >= 32 {
                                        if x < 0 { -1 } else { 0 }
                                    } else {
                                        x >> imm
                                    }
                                }
                                _ => {
                                    if imm >= 32 {
                                        0
                                    } else {
                                        (raw << imm) as i32
                                    }
                                }
                            };
                            o |= ((y as u32) as u128) << (32 * i);
                        }
                        o
                    }
                    (0x73, 2) | (0x73, 6) => {
                        let mut o = 0u128;
                        for i in 0..2u32 {
                            let x = (v >> (64 * i)) as u64;
                            let y = if imm >= 64 {
                                0
                            } else if group == 2 {
                                x >> imm
                            } else {
                                x << imm
                            };
                            o |= (y as u128) << (64 * i);
                        }
                        o
                    }
                    (0x73, 3) => {
                        // psrldq: shift the whole register right by bytes
                        if imm >= 16 {
                            0
                        } else {
                            v >> (8 * imm)
                        }
                    }
                    (0x73, 7) => {
                        // pslldq
                        if imm >= 16 {
                            0
                        } else {
                            v << (8 * imm)
                        }
                    }
                    _ => return Stop::BadOpcode,
                };
                // The ModRM.reg field selects the operation here; the register
                // being shifted is the r/m operand.
                self.xmm_rm_write(&rm, out);
                Stop::None
            }

            // --------------------------------------------- float arithmetic
            0x58 | 0x59 | 0x5C | 0x5E => {
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = if p66 {
                    let mut o = dst;
                    for lane in 0..2u32 {
                        let a = (dst >> (64 * lane)) as u64;
                        let b = (src >> (64 * lane)) as u64;
                        let r = fp64_arith(b2, a, b);
                        o = with_f64_lane(o, lane, f64::from_bits(r));
                    }
                    o
                } else if p_f3 {
                    let r = fp32_arith(b2, dst as u32, src as u32);
                    with_f32_lane(dst, 0, f32::from_bits(r))
                } else if p_f2 {
                    let r = fp64_arith(b2, dst as u64, src as u64);
                    with_f64_lane(dst, 0, f64::from_bits(r))
                } else {
                    let mut o = dst;
                    for lane in 0..4u32 {
                        let a = (dst >> (32 * lane)) as u32;
                        let b = (src >> (32 * lane)) as u32;
                        let r = fp32_arith(b2, a, b);
                        o = with_f32_lane(o, lane, f32::from_bits(r));
                    }
                    o
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x51 => {
                // sqrtps / sqrtpd / sqrtss / sqrtsd
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = if p66 {
                    let mut o = dst;
                    for lane in 0..2u32 {
                        o = with_f64_lane(o, lane, f64_lane(src, lane).sqrt());
                    }
                    o
                } else if p_f3 {
                    with_f32_lane(dst, 0, f32_lane(src, 0).sqrt())
                } else if p_f2 {
                    with_f64_lane(dst, 0, f64_lane(src, 0).sqrt())
                } else {
                    let mut o = dst;
                    for lane in 0..4u32 {
                        o = with_f32_lane(o, lane, f32_lane(src, lane).sqrt());
                    }
                    o
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x5D | 0x5F => {
                // minps/maxps/minpd/maxpd/minss/maxss/minsd/maxsd
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let is_min = b2 == 0x5D;
                let out = if p66 {
                    let mut o = dst;
                    for lane in 0..2u32 {
                        let a = f64_lane(dst, lane);
                        let b = f64_lane(src, lane);
                        let r = if is_min { x86_min_f64(a, b) } else { x86_max_f64(a, b) };
                        o = with_f64_lane(o, lane, r);
                    }
                    o
                } else if p_f3 {
                    let a = f32_lane(dst, 0);
                    let b = f32_lane(src, 0);
                    let r = if is_min { x86_min_f32(a, b) } else { x86_max_f32(a, b) };
                    with_f32_lane(dst, 0, r)
                } else if p_f2 {
                    let a = f64_lane(dst, 0);
                    let b = f64_lane(src, 0);
                    let r = if is_min { x86_min_f64(a, b) } else { x86_max_f64(a, b) };
                    with_f64_lane(dst, 0, r)
                } else {
                    let mut o = dst;
                    for lane in 0..4u32 {
                        let a = f32_lane(dst, lane);
                        let b = f32_lane(src, lane);
                        let r = if is_min { x86_min_f32(a, b) } else { x86_max_f32(a, b) };
                        o = with_f32_lane(o, lane, r);
                    }
                    o
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x5A => {
                // cvtps2pd / cvtpd2ps / cvtss2sd / cvtsd2ss
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = if p66 {
                    // cvtpd2ps: two doubles -> two singles, upper half zeroed
                    let lo = f64_lane(src, 0) as f32;
                    let hi = f64_lane(src, 1) as f32;
                    with_f32_lane(with_f32_lane(0, 0, lo), 1, hi)
                } else if p_f3 {
                    // cvtss2sd: low single widened, upper 64 bits preserved
                    let v = with_f64_lane(dst, 0, f32_lane(src, 0) as f64);
                    v
                } else if p_f2 {
                    // cvtsd2ss: low double narrowed, upper 96 bits preserved
                    let v = with_f32_lane(dst, 0, f64_lane(src, 0) as f32);
                    v
                } else {
                    // cvtps2pd: two low singles -> two doubles
                    let lo = f32_lane(src, 0) as f64;
                    let hi = f32_lane(src, 1) as f64;
                    with_f64_lane(with_f64_lane(0, 0, lo), 1, hi)
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0x2A => {
                // cvtsi2ss / cvtsi2sd
                let bits = if rexw { 64 } else { 32 };
                let (rm, reg) = self.decode_rm(bits, rex, has_rex);
                let raw = self.gpr_rm_read(&rm, bits);
                if p_f2 {
                    let v = if bits == 64 {
                        raw as i64 as f64
                    } else {
                        raw as u32 as i32 as f64
                    };
                    let out = with_f64_lane(self.xmm[reg.idx], 0, v);
                    self.xmm[reg.idx] = out;
                } else if p_f3 {
                    let v = if bits == 64 {
                        raw as i64 as f32
                    } else {
                        raw as u32 as i32 as f32
                    };
                    let out = with_f32_lane(self.xmm[reg.idx], 0, v);
                    self.xmm[reg.idx] = out;
                } else {
                    return Stop::BadOpcode;
                }
                Stop::None
            }
            0x2C => {
                // cvttss2si / cvttsd2si: the destination is the ModRM.reg GPR
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let bits = if rexw { 64 } else { 32 };
                let value = if p_f2 {
                    cvt_trunc(f64_lane(src, 0), bits)
                } else if p_f3 {
                    cvt_trunc(f32_lane(src, 0) as f64, bits)
                } else {
                    return Stop::BadOpcode;
                };
                self.set_reg(reg.idx, value, bits);
                Stop::None
            }
            0x2E | 0x2F => {
                // ucomiss / comiss / ucomisd / comisd
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let (a, b) = if p66 {
                    (f64_lane(self.xmm[reg.idx], 0), f64_lane(src, 0))
                } else {
                    (
                        f32_lane(self.xmm[reg.idx], 0) as f64,
                        f32_lane(src, 0) as f64,
                    )
                };
                self.comis(a, b);
                Stop::None
            }
            0xC2 => {
                // cmpps / cmppd / cmpss / cmpsd with an imm8 predicate
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let imm = self.fetch8();
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let out = if p66 {
                    let mut o = 0u128;
                    for lane in 0..2u32 {
                        if cmp_predicate(imm, f64_lane(dst, lane), f64_lane(src, lane)) {
                            o |= 0xFFFF_FFFF_FFFF_FFFFu128 << (64 * lane);
                        }
                    }
                    o
                } else if p_f2 {
                    // cmpsd writes only the low 64 bits; the upper half of the
                    // destination is architecturally unchanged.
                    let mask: u128 = if cmp_predicate(imm, f64_lane(dst, 0), f64_lane(src, 0)) {
                        0xFFFF_FFFF_FFFF_FFFF
                    } else {
                        0
                    };
                    (dst & !0xFFFF_FFFF_FFFF_FFFFu128) | mask
                } else if p_f3 {
                    // cmpss, likewise for the low 32 bits.
                    let mask: u128 = if cmp_predicate(
                        imm,
                        f32_lane(dst, 0) as f64,
                        f32_lane(src, 0) as f64,
                    ) {
                        0xFFFF_FFFF
                    } else {
                        0
                    };
                    (dst & !0xFFFF_FFFFu128) | mask
                } else {
                    let mut o = 0u128;
                    for lane in 0..4u32 {
                        if cmp_predicate(imm, f32_lane(dst, lane) as f64, f32_lane(src, lane) as f64)
                        {
                            o |= 0xFFFF_FFFFu128 << (32 * lane);
                        }
                    }
                    o
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // --------------------------------------- packed integer compare
            0x64 | 0x65 | 0x66 => {
                // pcmpgtb / pcmpgtw / pcmpgtd (66): signed greater-than, all
                // ones where true. The bare forms are MMX.
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let width: u32 = match b2 {
                    0x64 => 8,
                    0x65 => 16,
                    _ => 32,
                };
                let mask = (1u128 << width) - 1;
                let sign = 1u128 << (width - 1);
                let mut out = 0u128;
                for i in 0..(128 / width) {
                    let a = (dst >> (width * i)) & mask;
                    let b = (src >> (width * i)) & mask;
                    // Flipping the sign bit turns the signed compare into an
                    // unsigned one.
                    if (a ^ sign) > (b ^ sign) {
                        out |= mask << (width * i);
                    }
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // --------------------------------------------------- pack family
            0x63 | 0x67 | 0x6B => {
                // packsswb / packuswb / packssdw (66): narrow with saturation,
                // destination lanes first, then the source lanes.
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let mut out = 0u128;
                if b2 == 0x6B {
                    for i in 0..4u32 {
                        let a = ((dst >> (32 * i)) as u32) as i32;
                        let b = ((src >> (32 * i)) as u32) as i32;
                        out |= (sat_i16(a) as u128) << (16 * i);
                        out |= (sat_i16(b) as u128) << (16 * (i + 4));
                    }
                } else {
                    for i in 0..8u32 {
                        let a = ((dst >> (16 * i)) as u16) as i16;
                        let b = ((src >> (16 * i)) as u16) as i16;
                        let (pa, pb) = if b2 == 0x63 {
                            (sat_i8(a), sat_i8(b))
                        } else {
                            (sat_u8(a), sat_u8(b))
                        };
                        out |= (pa as u128) << (8 * i);
                        out |= (pb as u128) << (8 * (i + 8));
                    }
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // -------------------------------------------- packed multiply
            0xD5 | 0xE4 | 0xE5 => {
                // pmullw / pmulhw / pmulhuw (66)
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let mut out = 0u128;
                for i in 0..8u32 {
                    let a = (dst >> (16 * i)) as u16;
                    let b = (src >> (16 * i)) as u16;
                    let r: u16 = match b2 {
                        0xD5 => (a as u32 * b as u32) as u16,
                        0xE5 => (((a as i16 as i32) * (b as i16 as i32)) >> 16) as u16,
                        _ => ((a as u32 * b as u32) >> 16) as u16,
                    };
                    out |= (r as u128) << (16 * i);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0xF5 => {
                // pmaddwd: multiply signed word pairs and add into dwords
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let mut out = 0u128;
                for i in 0..4u32 {
                    let a0 = (dst >> (32 * i)) as u32 as i16 as i32;
                    let a1 = (dst >> (32 * i + 16)) as u32 as i16 as i32;
                    let b0 = (src >> (32 * i)) as u32 as i16 as i32;
                    let b1 = (src >> (32 * i + 16)) as u32 as i16 as i32;
                    let r = a0 * b0 + a1 * b1;
                    out |= ((r as u32) as u128) << (32 * i);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0xF6 => {
                // psadbw: sum of absolute byte differences, one word per half
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let mut out = 0u128;
                for half in 0..2u32 {
                    let mut sum = 0u32;
                    for i in 0..8u32 {
                        let a = ((dst >> (64 * half + 8 * i)) as u8) as i16;
                        let b = ((src >> (64 * half + 8 * i)) as u8) as i16;
                        sum += (a - b).unsigned_abs() as u32;
                    }
                    out |= ((sum as u16) as u128) << (64 * half);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // ------------------------------------------------ packed min/max
            0xDA | 0xDE | 0xEA | 0xEE => {
                // pminub/pmaxub (unsigned bytes), pminsw/pmaxsw (signed words)
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let byte = b2 == 0xDA || b2 == 0xDE;
                let is_min = b2 == 0xDA || b2 == 0xEA;
                let width: u32 = if byte { 8 } else { 16 };
                let mask = (1u128 << width) - 1;
                let mut out = 0u128;
                for i in 0..(128 / width) {
                    let a = (dst >> (width * i)) & mask;
                    let b = (src >> (width * i)) & mask;
                    let keep_a = if byte {
                        if is_min { a < b } else { a > b }
                    } else {
                        let sa = (((a as u64) << (64 - width)) as i64) >> (64 - width);
                        let sb = (((b as u64) << (64 - width)) as i64) >> (64 - width);
                        if is_min { sa < sb } else { sa > sb }
                    };
                    let r = if keep_a { a } else { b };
                    out |= r << (width * i);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // ------------------------------ packed add/sub, saturating or not
            0xD8 | 0xD9 | 0xDC | 0xDD | 0xE8 | 0xE9 | 0xEC | 0xED | 0xF8 | 0xF9 | 0xFC | 0xFD => {
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                // Even opcodes are the byte forms, odd ones the word forms.
                let byte = b2 & 1 == 0;
                let subtract = b2 & 4 == 0;
                // 0xD8..0xDD unsigned saturating, 0xE8..0xED signed, 0xF8..0xFD
                // plain wrapping arithmetic.
                let kind = if b2 < 0xE0 {
                    0u8
                } else if b2 < 0xF0 {
                    1
                } else {
                    2
                };
                let width: u32 = if byte { 8 } else { 16 };
                let mask = (1u128 << width) - 1;
                let mut out = 0u128;
                for i in 0..(128 / width) {
                    let a = ((dst >> (width * i)) & mask) as u16;
                    let b = ((src >> (width * i)) & mask) as u16;
                    let r: u16 = if byte {
                        let (a, b) = (a as u8, b as u8);
                        match (kind, subtract) {
                            (0, true) => sat_sub_u8(a, b) as u16,
                            (0, false) => sat_add_u8(a, b) as u16,
                            (1, true) => sat_sub_i8(a as i8, b as i8) as u8 as u16,
                            (1, false) => sat_add_i8(a as i8, b as i8) as u8 as u16,
                            (_, true) => a.wrapping_sub(b) as u16,
                            (_, false) => a.wrapping_add(b) as u16,
                        }
                    } else {
                        match (kind, subtract) {
                            (0, true) => sat_sub_u16(a, b),
                            (0, false) => sat_add_u16(a, b),
                            (1, true) => sat_sub_i16(a as i16, b as i16) as u16,
                            (1, false) => sat_add_i16(a as i16, b as i16) as u16,
                            (_, true) => a.wrapping_sub(b),
                            (_, false) => a.wrapping_add(b),
                        }
                    };
                    out |= (r as u128) << (width * i);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // ------------------------------------- packed shifts by a vector
            0xD1 | 0xD2 | 0xD3 | 0xE1 | 0xE2 | 0xF1 | 0xF2 | 0xF3 => {
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                // Only the low 64 bits of the source hold the count.
                let count = src as u64;
                let width: u32 = match b2 {
                    0xD1 | 0xE1 | 0xF1 => 16,
                    0xD2 | 0xE2 | 0xF2 => 32,
                    _ => 64,
                };
                let arithmetic = b2 == 0xE1 || b2 == 0xE2;
                let left = b2 == 0xF1 || b2 == 0xF2 || b2 == 0xF3;
                let mask = (1u128 << width) - 1;
                let mut out = 0u128;
                for i in 0..(128 / width) {
                    let a = (dst >> (width * i)) & mask;
                    let r: u128 = if arithmetic {
                        let sv = (((a as u64) << (64 - width)) as i64) >> (64 - width);
                        let shifted = if count >= width as u64 {
                            if sv < 0 {
                                -1
                            } else {
                                0
                            }
                        } else {
                            sv >> (count as u32)
                        };
                        ((shifted as u64) as u128) & mask
                    } else if left {
                        if count >= width as u64 {
                            0
                        } else {
                            (a << (count as u32)) & mask
                        }
                    } else if count >= width as u64 {
                        0
                    } else {
                        a >> (count as u32)
                    };
                    out |= r << (width * i);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // --------------------------------------------------- mask extracts
            0xD7 => {
                // pmovmskb: the top bit of each of the sixteen bytes. The
                // source must be a register -- GAS rejects a memory operand
                // ("operand size mismatch") -- so refuse that form instead of
                // inventing a load the architecture does not define.
                if !p66 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                if !rm.is_reg {
                    return Stop::BadOpcode;
                }
                let src = self.xmm_rm_read(&rm);
                let mut out = 0u64;
                for i in 0..16u32 {
                    if (src >> (8 * i + 7)) & 1 == 1 {
                        out |= 1 << i;
                    }
                }
                self.set_reg(reg.idx, out, 32);
                Stop::None
            }
            0xC5 => {
                // pextrw: one 16-bit lane into a GPR, zero-extended. gcc -O3
                // emits this to drain a packed min/max reduction.
                if !p66 || p_f2 || p_f3 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                if !rm.is_reg {
                    return Stop::BadOpcode;
                }
                let imm = self.fetch8();
                let src = self.xmm_rm_read(&rm);
                let lane = (imm & 7) as u32;
                let word = ((src >> (16 * lane)) & 0xFFFF) as u64;
                self.set_reg(reg.idx, word, 32);
                Stop::None
            }
            0x50 => {
                // movmskps (bare) / movmskpd (66): sign bits to a GPR. As with
                // pmovmskb the vector operand must be a register.
                if p_f2 || p_f3 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                if !rm.is_reg {
                    return Stop::BadOpcode;
                }
                let src = self.xmm_rm_read(&rm);
                let mut out = 0u64;
                let lanes = if p66 { 2 } else { 4 };
                let stride: u32 = if p66 { 64 } else { 32 };
                for i in 0..lanes {
                    if (src >> (stride * i + stride - 1)) & 1 == 1 {
                        out |= 1 << i;
                    }
                }
                self.set_reg(reg.idx, out, 32);
                Stop::None
            }

            // -------------------------------------------------------- shuffles
            0xC6 => {
                // shufps (bare) / shufpd (66)
                if p_f2 || p_f3 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let dst = self.xmm[reg.idx];
                let imm = self.fetch8();
                let out = if p66 {
                    let lo = if imm & 1 == 0 { dst } else { dst >> 64 };
                    let hi = if imm & 2 == 0 { src } else { src >> 64 };
                    (lo & 0xFFFF_FFFF_FFFF_FFFF) | ((hi & 0xFFFF_FFFF_FFFF_FFFF) << 64)
                } else {
                    let mut o = 0u128;
                    o = with_f32_lane(o, 0, f32_lane(dst, (imm & 3) as u32));
                    o = with_f32_lane(o, 1, f32_lane(dst, ((imm >> 2) & 3) as u32));
                    o = with_f32_lane(o, 2, f32_lane(src, ((imm >> 4) & 3) as u32));
                    o = with_f32_lane(o, 3, f32_lane(src, ((imm >> 6) & 3) as u32));
                    o
                };
                self.xmm[reg.idx] = out;
                Stop::None
            }

            // --------------------------------------------- integer conversions
            0x2D => {
                // cvtss2si / cvtsd2si, rounding under the current MXCSR mode
                // (which this core keeps at the default, nearest-even).
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let bits = if rexw { 64 } else { 32 };
                let value = if p_f2 {
                    cvt_round(f64_lane(src, 0), bits)
                } else if p_f3 {
                    cvt_round(f32_lane(src, 0) as f64, bits)
                } else {
                    return Stop::BadOpcode;
                };
                self.set_reg(reg.idx, value, bits);
                Stop::None
            }
            0x5B => {
                // cvtdq2ps: four signed dwords to four singles
                if p66 || p_f2 {
                    return Stop::BadOpcode;
                }
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                let mut out = 0u128;
                for i in 0..4u32 {
                    let v = ((src >> (32 * i)) as u32) as i32 as f32;
                    out = with_f32_lane(out, i, v);
                }
                self.xmm[reg.idx] = out;
                Stop::None
            }
            0xE6 => {
                // cvttpd2dq (66) / cvtpd2dq (F2) / cvtdq2pd (F3)
                let (rm, reg) = self.decode_rm(128, rex, has_rex);
                let src = self.xmm_rm_read(&rm);
                if p66 || p_f2 {
                    let conv = if p66 { cvt_trunc } else { cvt_round };
                    let lo = conv(f64_lane(src, 0), 32) as u32;
                    let hi = conv(f64_lane(src, 1), 32) as u32;
                    self.xmm[reg.idx] = (lo as u128) | ((hi as u128) << 32);
                } else if p_f3 {
                    let lo = ((src as u32) as i32) as f64;
                    let hi = (((src >> 32) as u32) as i32) as f64;
                    self.xmm[reg.idx] = with_f64_lane(with_f64_lane(0, 0, lo), 1, hi);
                } else {
                    return Stop::BadOpcode;
                }
                Stop::None
            }
            _ => Stop::BadOpcode,
        }
    }

    // ------------------------------------------------------------- system ops

    /// CPUID reports what *this core implements*, not what a real Jaguar does:
    /// advertising AVX/AVX2/SSE4 would let a compiler emit instructions we stop
    /// on, which is exactly the silent-failure mode this project refuses.
    fn cpuid(&mut self) {
        let leaf = self.get_reg(RAX, 32) as u32;
        let (a, b, c, d): (u32, u32, u32, u32) = match leaf {
            0 => (1, 0x6874_7541, 0x444D_4163, 0x6974_6E65), // "AuthenticAMD"
            1 => (
                0x0070_0F01, // family 0x16 (Jaguar), model 0, stepping 1
                0x0008_0000, // 8 logical processors, brand index 0
                0x0000_0201, // SSE3, SSSE3
                0x0780_0191, // FPU, TSC, CX8, CMOV, MMX, FXSR, SSE, SSE2
            ),
            0x8000_0000 => (0x8000_0008, 0, 0, 0),
            0x8000_0001 => (0, 0, 0, 0x2010_0000), // NX, long mode
            0x8000_0008 => (0x3030, 0, 0, 0),      // 48-bit virtual and physical
            _ => (0, 0, 0, 0),
        };
        self.set_reg(RAX, a as u64, 32);
        self.set_reg(RBX, b as u64, 32);
        self.set_reg(RCX, c as u64, 32);
        self.set_reg(RDX, d as u64, 32);
    }

    fn group3(&mut self, bits: u32, rex: u8, has_rex: bool) -> Stop {
        let (src, reg) = self.decode_rm(bits, rex, has_rex);
        let group = reg.idx & 7;
        let mask = width_mask(bits);
        match group {
            0 | 1 => {
                let imm = if bits == 64 {
                    self.fetch32() as i32 as i64 as u64
                } else {
                    self.fetch_imm(bits)
                };
                let res = self.read_op(&src) & imm;
                self.flags.cf = false;
                self.flags.of = false;
                self.flags.af = false;
                self.szp(res, bits);
            }
            2 => {
                let v = !self.read_op(&src) & mask;
                self.write_op(&src, v);
            }
            3 => {
                let v = self.read_op(&src);
                let res = self.alu(5, 0, v, bits);
                self.write_op(&src, res);
            }
            4 => {
                // unsigned mul -> rdx:rax
                let a = self.get_reg(RAX, bits) as u128;
                let b = self.read_op(&src) as u128;
                let p = a * b;
                let lo = (p as u64) & mask;
                let hi = ((p >> bits) as u64) & mask;
                self.set_reg(RAX, lo, bits);
                self.set_reg(RDX, hi, bits);
                self.flags.cf = hi != 0;
                self.flags.of = hi != 0;
            }
            5 => {
                // signed mul -> rdx:rax
                let a = Self::sign_extend(self.get_reg(RAX, bits), bits) as i64 as i128;
                let b = Self::sign_extend(self.read_op(&src), bits) as i64 as i128;
                let p = a * b;
                let lo = (p as u64) & mask;
                let hi = ((p >> bits) as u64) & mask;
                self.set_reg(RAX, lo, bits);
                self.set_reg(RDX, hi, bits);
                let sign = if (lo >> (bits - 1)) & 1 == 1 { mask } else { 0 };
                self.flags.cf = hi != sign;
                self.flags.of = hi != sign;
            }
            6 => {
                let d = self.read_op(&src);
                if d == 0 {
                    return Stop::DivideError;
                }
                let n = ((self.get_reg(RDX, bits) as u128) << bits) | self.get_reg(RAX, bits) as u128;
                let q = n / d as u128;
                let rem = n % d as u128;
                self.set_reg(RAX, q as u64, bits);
                self.set_reg(RDX, rem as u64, bits);
            }
            _ => {
                let d = Self::sign_extend(self.read_op(&src), bits) as i64 as i128;
                if d == 0 {
                    return Stop::DivideError;
                }
                let n_hi = Self::sign_extend(self.get_reg(RDX, bits), bits) as i64 as i128;
                let n_lo = Self::sign_extend(self.get_reg(RAX, bits), bits) as i64 as i128;
                let n = (n_hi << bits) | (n_lo & mask as i128);
                let q = n / d;
                let rem = n % d;
                self.set_reg(RAX, q as u64, bits);
                self.set_reg(RDX, rem as u64, bits);
            }
        }
        Stop::None
    }

    fn group45(&mut self, opcode: u8, opsz: u32, rex: u8, has_rex: bool) -> Stop {
        let bits = if opcode == 0xFE { 8 } else { opsz };
        let (dst, reg) = self.decode_rm(bits, rex, has_rex);
        let group = reg.idx & 7;
        let cf_before = self.flags.cf;
        match group {
            0 => {
                let a = self.read_op(&dst);
                let res = self.alu(0, a, 1, bits);
                self.flags.cf = cf_before; // inc does not touch CF
                self.write_op(&dst, res);
            }
            1 => {
                let a = self.read_op(&dst);
                let res = self.alu(5, a, 1, bits);
                self.flags.cf = cf_before; // dec does not touch CF
                self.write_op(&dst, res);
            }
            2 => {
                // call r/m64
                let target = if dst.is_reg {
                    self.get_reg(dst.idx, 64)
                } else {
                    self.mem.read(dst.addr, 64)
                };
                let ret = self.rip;
                self.push(ret, 64);
                self.rip = target;
            }
            4 => {
                // jmp r/m64
                self.rip = if dst.is_reg {
                    self.get_reg(dst.idx, 64)
                } else {
                    self.mem.read(dst.addr, 64)
                };
            }
            6 => {
                let v = if dst.is_reg {
                    self.get_reg(dst.idx, 64)
                } else {
                    self.mem.read(dst.addr, 64)
                };
                self.push(v, 64);
            }
            _ => return Stop::BadOpcode,
        }
        Stop::None
    }

    fn two_byte(
        &mut self,
        opsz: u32,
        rex: u8,
        has_rex: bool,
        p66: bool,
        p_f2: bool,
        p_f3: bool,
    ) -> Stop {
        let b2 = self.fetch8();
        self.last_ext = Some(b2);
        if self.trace {
            eprintln!("           ext={b2:02x}");
        }
        match b2 {
            0x05 => Stop::Syscall,
            0x0B => Stop::Ud2,
            // prefetch / multi-byte / hint nops: decode the ModRM so the
            // instruction length stays right, then do nothing.
            0x0D | 0x18 | 0x1E | 0x1F => {
                let _ = self.decode_rm(opsz, rex, has_rex);
                Stop::None
            }
            0x40..=0x4F => {
                // cmovcc
                let (src, dst) = self.decode_rm(opsz, rex, has_rex);
                if self.cond((b2 & 0x0F) as usize) {
                    let v = self.read_op(&src);
                    self.write_op(&dst, v);
                }
                Stop::None
            }
            0x80..=0x8F => {
                // jcc rel32
                let rel = self.fetch32() as i32 as i64 as u64;
                if self.cond((b2 & 0x0F) as usize) {
                    self.rip = self.rip.wrapping_add(rel);
                }
                Stop::None
            }
            0xA4 | 0xA5 | 0xAC | 0xAD => {
                // shld/shrd r/m, r, imm8|CL: a shift that feeds in bits from
                // the second register instead of zeros or the sign. The
                // destination is the r/m operand, the fill register is
                // ModRM.reg.
                let (dst, src) = self.decode_rm(opsz, rex, has_rex);
                let count = match b2 {
                    0xA4 | 0xAC => self.fetch8(),
                    _ => self.get_reg(RCX, 8) as u8,
                };
                let value = self.read_op(&dst);
                let fill = self.read_op(&src);
                let res = if b2 == 0xA4 || b2 == 0xA5 {
                    self.shld(value, fill, count, opsz)
                } else {
                    self.shrd(value, fill, count, opsz)
                };
                self.write_op(&dst, res);
                Stop::None
            }
            0xAF => {
                // imul r, r/m: only CF/OF are defined (product fits?), and the
                // result is truncated to the destination width.
                let (src, dst) = self.decode_rm(opsz, rex, has_rex);
                let a = Self::sign_extend(self.read_op(&dst), opsz) as i128;
                let b = Self::sign_extend(self.read_op(&src), opsz) as i128;
                let product = a.wrapping_mul(b);
                let res = (product as u64) & width_mask(opsz);
                let fits = product == Self::sign_extend(res, opsz) as i128;
                self.flags.cf = !fits;
                self.flags.of = !fits;
                self.write_op(&dst, res);
                Stop::None
            }
            0x31 => {
                // rdtsc: a deterministic, monotonic counter derived from the
                // instruction count (4 ticks/instruction). Determinism matters
                // more than a plausible frequency while bring-up is the goal.
                let tsc = self.steps.wrapping_mul(4).wrapping_add(0x1000);
                self.set_reg(RAX, tsc, 32);
                self.set_reg(RDX, tsc >> 32, 32);
                Stop::None
            }
            0xA2 => {
                self.cpuid();
                Stop::None
            }
            0x90..=0x9F => {
                // setcc r/m8
                let (dst, _) = self.decode_rm(8, rex, has_rex);
                let v = if self.cond((b2 & 0x0F) as usize) { 1 } else { 0 };
                self.write_op(&dst, v);
                Stop::None
            }
            0xAE => {
                // lfence / mfence / sfence are no-ops on a single in-order core.
                let (dst, reg) = self.decode_rm(opsz, rex, has_rex);
                let group = reg.idx & 7;
                if dst.is_reg && (5..=7).contains(&group) {
                    Stop::None
                } else {
                    // fxsave/fxrstor/clflush/... are genuinely unimplemented.
                    Stop::BadOpcode
                }
            }
            0xB8 if p_f3 => {
                // popcnt r, r/m
                let (src, dst) = self.decode_rm(opsz, rex, has_rex);
                let v = self.read_op(&src);
                self.flags.zf = v == 0;
                self.flags.cf = false;
                self.flags.of = false;
                self.flags.sf = false;
                self.flags.af = false;
                self.flags.pf = false;
                self.write_op(&dst, (v & width_mask(opsz)).count_ones() as u64);
                Stop::None
            }
            0xBC | 0xBD => {
                // bsf/bsr, or tzcnt/lzcnt when the F3 prefix turns them into
                // the BMI encodings.
                let (src, dst) = self.decode_rm(opsz, rex, has_rex);
                let v = self.read_op(&src) & width_mask(opsz);
                let width = opsz;
                let value = if p_f3 {
                    self.flags.cf = v == 0;
                    self.flags.zf = v == 0;
                    if v == 0 {
                        0
                    } else if b2 == 0xBC {
                        v.trailing_zeros() as u64
                    } else {
                        (match width {
                            64 => v.leading_zeros(),
                            32 => (v as u32).leading_zeros(),
                            16 => (v as u16).leading_zeros() as u32,
                            _ => (v as u8).leading_zeros() as u32,
                        }) as u64
                    }
                } else if v == 0 {
                    self.flags.zf = true;
                    return Stop::None;
                } else {
                    self.flags.zf = false;
                    if b2 == 0xBC {
                        v.trailing_zeros() as u64
                    } else {
                        (63 - v.leading_zeros()) as u64
                    }
                };
                self.write_op(&dst, value);
                Stop::None
            }
            0xB6 | 0xB7 | 0xBE | 0xBF => {
                let src_bits = if b2 == 0xB6 || b2 == 0xBE { 8 } else { 16 };
                let (src, mut dst) = self.decode_rm(src_bits, rex, has_rex);
                let raw = self.read_op(&src);
                let value = match b2 {
                    0xB6 | 0xB7 => raw,
                    _ => Self::sign_extend(raw, src_bits),
                };
                // movzx/movsx always widen into a 16/32/64-bit destination,
                // regardless of the r/m operand's width.
                widen_destination(&mut dst, opsz);
                self.write_op(&dst, value);
                Stop::None
            }
            0xC8..=0xCF => {
                // bswap r32/r64. A 16-bit operand size is architecturally
                // undefined, so refuse it rather than invent behaviour.
                if p66 {
                    return Stop::BadOpcode;
                }
                let idx = ((b2 & 7) as usize) | ((rex & 1) as usize) << 3;
                let bits = if rex & 8 != 0 { 64 } else { 32 };
                let v = self.get_reg(idx, bits);
                let out = match bits {
                    64 => v.swap_bytes(),
                    _ => (v as u32).swap_bytes() as u64,
                };
                self.set_reg(idx, out, bits);
                Stop::None
            }
            // Everything else two-byte is an SSE/SSE2 encoding, and `sse`
            // is the single place that decides which of those exist: a new
            // vector instruction needs an arm there and nothing here.
            _ => self.sse(b2, rex, has_rex, p66, p_f2, p_f3),
        }
    }
}
