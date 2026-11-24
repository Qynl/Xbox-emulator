//! x86-64 RFLAGS subset that the interpreter tracks.
//!
//! Only the seven flags a normal program can observe are modelled. The reserved
//! bit 1 is reported as set, matching real hardware and Unicorn, which keeps
//! differential comparisons meaningful.

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Flags {
    pub cf: bool,
    pub pf: bool,
    pub af: bool,
    pub zf: bool,
    pub sf: bool,
    pub of: bool,
    pub df: bool,
}

pub const FLAG_CF: u64 = 1 << 0;
pub const FLAG_PF: u64 = 1 << 2;
pub const FLAG_AF: u64 = 1 << 4;
pub const FLAG_ZF: u64 = 1 << 6;
pub const FLAG_SF: u64 = 1 << 7;
pub const FLAG_DF: u64 = 1 << 10;
pub const FLAG_OF: u64 = 1 << 11;

impl Flags {
    /// Pack the modelled flags into an RFLAGS-shaped value.
    pub fn pack(&self) -> u64 {
        let mut v = 1 << 1; // reserved, always 1
        if self.cf {
            v |= FLAG_CF;
        }
        if self.pf {
            v |= FLAG_PF;
        }
        if self.af {
            v |= FLAG_AF;
        }
        if self.zf {
            v |= FLAG_ZF;
        }
        if self.sf {
            v |= FLAG_SF;
        }
        if self.df {
            v |= FLAG_DF;
        }
        if self.of {
            v |= FLAG_OF;
        }
        v
    }

    /// Rebuild the tracked flags from an RFLAGS value.
    pub fn unpack(v: u64) -> Self {
        Flags {
            cf: v & FLAG_CF != 0,
            pf: v & FLAG_PF != 0,
            af: v & FLAG_AF != 0,
            zf: v & FLAG_ZF != 0,
            sf: v & FLAG_SF != 0,
            of: v & FLAG_OF != 0,
            df: v & FLAG_DF != 0,
        }
    }
}
