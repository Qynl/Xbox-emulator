//! Flat physical memory.
//!
//! Durango has a unified 8 GiB memory space. Until the MMU/memory controller
//! milestone lands, the interpreter runs against a flat, zero-initialised
//! window with little-endian accessors. All accesses wrap modulo the window
//! size so a bad pointer traps later (when we add page faults) instead of
//! panicking the emulator.

pub const MEM_SIZE: usize = 16 << 20; // 16 MiB bring-up window
const MEM_MASK: u64 = (MEM_SIZE as u64) - 1;

/// Mask of the low `bits` bits, shared with the interpreter's flag and ALU
/// code so operand widths are masked in exactly one place.
#[inline]
pub(crate) fn width_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

pub struct Memory {
    data: Vec<u8>,
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            data: vec![0u8; MEM_SIZE],
        }
    }

    pub fn size(&self) -> u64 {
        MEM_SIZE as u64
    }

    #[inline]
    pub fn read8(&self, addr: u64) -> u8 {
        self.data[(addr & MEM_MASK) as usize]
    }

    #[inline]
    pub fn write8(&mut self, addr: u64, value: u8) {
        self.data[(addr & MEM_MASK) as usize] = value;
    }

    /// Little-endian load of `bits` (8/16/32/64).
    ///
    /// This sits under the interpreter's hottest paths (instruction fetch and
    /// every memory operand), so it is one unaligned 8-byte load followed by a
    /// mask, rather than a lookup on the operand width. Widening a narrow load
    /// to eight bytes is only sound while memory is flat RAM: see [`write`] for
    /// the assumption and what has to change when MMIO and page faults land.
    ///
    /// [`write`]: Memory::write
    #[inline]
    pub fn read(&self, addr: u64, bits: u32) -> u64 {
        debug_assert!(matches!(bits, 8 | 16 | 32 | 64));
        let offset = (addr & MEM_MASK) as usize;
        let raw = if offset + 8 <= MEM_SIZE {
            u64::from_le_bytes(self.data[offset..offset + 8].try_into().unwrap())
        } else {
            let mut value = 0u64;
            for i in 0..(bits / 8) as u64 {
                value |= (self.read8(addr.wrapping_add(i)) as u64) << (8 * i);
            }
            return value;
        };
        raw & width_mask(bits)
    }

    /// Little-endian store of `bits` (8/16/32/64).
    ///
    /// **A narrow store writes a whole eight-byte word.** The other bytes are
    /// read back and written with the values they already held, so the
    /// guest-visible effect is a store of exactly `bits`; it costs one load and
    /// one store instead of a branch per width.
    ///
    /// That is sound only because this window is plain, flat, single-threaded
    /// RAM where reading a byte has no effect. It must be revisited when MMIO
    /// and page faults land:
    ///
    /// * a device register write has to keep its encoded width. Widening it can
    ///   hit a neighbouring register, and the read half of the read-modify-write
    ///   is itself a side effect on read-clear registers;
    /// * a store that would cross a page boundary has to fault at the boundary
    ///   rather than silently perform an access the guest never made. The
    ///   `offset + 8 <= MEM_SIZE` guard below is a *wrap* guard for the flat
    ///   window, not a page or MMIO guard.
    ///
    /// `read` widens narrow loads in the same way, with the same caveat.
    #[inline]
    pub fn write(&mut self, addr: u64, value: u64, bits: u32) {
        debug_assert!(matches!(bits, 8 | 16 | 32 | 64));
        let offset = (addr & MEM_MASK) as usize;
        if offset + 8 <= MEM_SIZE {
            let mask = width_mask(bits);
            let raw = u64::from_le_bytes(self.data[offset..offset + 8].try_into().unwrap());
            let merged = (raw & !mask) | (value & mask);
            self.data[offset..offset + 8].copy_from_slice(&merged.to_le_bytes());
        } else {
            for i in 0..(bits / 8) as u64 {
                self.write8(addr.wrapping_add(i), (value >> (8 * i)) as u8);
            }
        }
    }

    pub fn write_bytes(&mut self, addr: u64, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            self.write8(addr.wrapping_add(i as u64), *b);
        }
    }

    pub fn bytes(&self, addr: u64, len: usize) -> Vec<u8> {
        (0..len as u64)
            .map(|i| self.read8(addr.wrapping_add(i)))
            .collect()
    }
}
