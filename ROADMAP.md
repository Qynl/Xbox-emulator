# Roadmap

Ordered so that every milestone is independently testable. No milestone gets
declared "done" on the strength of a boot screen; each one ships with tests.

## Console generations

The x86-64 core and machine shell target the two x86-64 generations:

* **Xbox One / One S / One X** (Durango, AMD Jaguar x86-64) — what the
  milestones below describe.
* **Xbox Series S / X** (Scarlett, AMD Zen 2 x86-64) — the same ISA family, so
  the same core plus the newer extensions (AVX/AVX2, and the RDNA 2 GPU
  instead of GCN). This is a second *machine*, not a second CPU core.

The 7th and 6th generations are different CPU architectures and each needs a
core of its own behind the same shell:

* **Xbox 360** (Xenon: triple-core IBM PowerPC, big-endian) — a PowerPC core,
  plus its own GPU (Xenos/EDRAM).
* **Original Xbox** (Pentium III: 32-bit x86, SSE) — a 32-bit x86 core, plus the
  NV2A GPU.

Those two are listed under "Other generations" at the end, not folded into the
x86-64 milestones, because pretending one decoder covers them would be exactly
the kind of dishonesty this project avoids.

## M1 — x86-64 interpreter (done)

Executes the integer subset of the Jaguar's ISA faithfully, verified
differentially against Unicorn. Includes ModRM/SIB/RIP-relative addressing,
REX and operand-size handling, flags, stack, branching, and the string
instructions with REP/REPE/REPNE and DF.

## M2 — SSE2 + compiler gate (done)

XMM register file, the moves, packed integer ops, packed and scalar float
arithmetic, comparisons and conversions that `gcc -O2` emits, plus `cpuid` and
`rdtsc`.

**Gate:** `tests/c/m2_gate.c` (integer and scalar SSE, `gcc -O2`) and
`tests/c/simd_gate.c` (auto-vectorised packed integer and float code,
`gcc -O3`) are each compiled twice — as a freestanding x86-64 image the core
executes, and as a host binary — and the results must be identical. Both also
run in Unicorn, so the gate is checked against native hardware semantics *and*
an independent emulator. Currently: 1306 instructions of `-O2` and 5138 of
`-O3` vectorised output, all registers, flags and memory identical on every
side.

**Verification:** beyond the fixtures, the SSE2 surface is fuzzed differentially
with seeded XMM state (1200 randomised SSE/string/ALU cases across three seeds
in `tools/test.sh`, each compared against Unicorn field for field), and the
floating-point rules are checked against the real CPU by
`tests/c/nan_rule.c` (3656 operand cases across every scalar/packed,
single/double form). That oracle is what pins NaN selection down: hardware
returns the destination operand's NaN, QEMU returns the larger significand, and
this core follows the hardware.

Also covered since: `rcl`/`rcr`/`shld`/`shrd`, the packed-integer arithmetic,
saturating and min/max forms, multiplies, `pmaddwd`/`psadbw`, vector-count
packed shifts, `shufps`/`shufpd`, `movmskps`/`movmskpd`/`pmovmskb`, and the
`cvt*2si`/`cvtdq2ps` family. Shift and rotate flags are pinned by a native
oracle (`tools/check_shift_flags.py`, 8184 cases) rather than by the reference.

**Throughput.** `tools/bench.py` measures the interpreter on a hand-assembled
loop and A/Bs two builds with interleaved runs, because a single run on the
shared VM proves nothing. Replacing the byte-at-a-time memory accessors with
one unaligned load and a masked read-modify-write store is 1.20x on a mixed
body and 1.13x on SSE-only code (medians of 15 interleaved runs, 15/15 paired
wins, sign test p = 6e-05, 95% CI 1.189-1.221 and 1.126-1.146); absolute MIPS
move with host load, the ratio does not. The two-byte opcode dispatch also has
a single source of truth now: the integer half of `two_byte` claims what it
handles and hands everything else to `sse`, so a new vector instruction cannot
be half-added.

**Open in M2:** MXCSR (rounding modes and exception masks are fixed at the
default), AVX/AVX2 (deliberately not advertised in `cpuid` until they execute —
this is also what separates Durango from Scarlett), x87, MMX.

## M3 — Memory management and the Durango map

4-level page tables, NX/WP bits, page-fault exceptions (`#PF`), and the
documented Durango physical map: 8 GiB RAM, the APU register blocks, the
southbridge, the flash controller window. Gate: a test that walks a page table
the emulator itself built, and faults exactly where the real hardware would.

This is also where the width-widening in `Memory::read`/`write` has to be undone
(see the doc comments in `src/memory.rs`): an MMIO register access and a store
that crosses a page boundary both need the guest's exact access width, not a
masked read-modify-write of eight bytes.

## M4 — Protected-mode machinery: IDT, exceptions, interrupts

Descriptor tables, `#GP`/`#UD`/`#PF`/`#DF` delivery, `iretq`, APIC timer as the
interrupt source, and the beginnings of the hypervisor partition model
(hypercalls in the documented `0xEA`-family range, VP/VMCS-shaped state). This
is also where `int3`/`syscall` stop being special-cased stop reasons and become
real exception delivery.

## M5 — PE/COFF loader and title runtime surface

Load x64 PE images with relocations, the Durango title memory layout, and the
first kernel-mode imports (`ExAllocatePool`, `KeQueryPerformanceCounter`,
threads and synchronisation). Gate: a *homebrew* x64 PE executes its entry point
and returns through the emulator's syscall boundary. Retail titles are explicitly
out of scope here — see "Keys" below.

## M6 — GPU: command processor and GCN

Ring-buffer command processor, GCN wavefront execution for the shader subset,
resource/texture binding, and a software rasteriser that can present a frame.
This is the longest milestone by far. Gate: a hand-written draw command produces
the expected pixels, compared against a reference render.

## M7 — Storage, audio, input

VHD-style virtual disk, the console's audio block, and controller/HID mapping
(the emulator's window maps an Xbox pad or keyboard/mouse onto Durango input
reports).

## M8 — System integration

Boot the console's early boot chain as far as the absence of signing keys
allows, wire the virtual GPU to a window, and implement save states (serialise
CPU + memory + device state; save states are also the best regression test we
will ever have).

## Other generations

* **N1 — Xbox 360 (Xenon).** A big-endian PowerPC core, VMX/AltiVec, and the
  Xenos GPU with its EDRAM. Nothing here is shared with the x86-64 core except
  the machine shell (memory, devices, loader, debugger); it is a separate
  multi-month project and is deliberately not started until the x86-64 path
  boots something.
* **N2 — Original Xbox (2001).** A 32-bit x86 core (Pentium III with SSE) and
  the NV2A GPU — much closer to the existing core in shape, since the integer
  and SSE pieces overlap, but the 32-bit and segment/real-mode behaviour is a
  different execution mode, not a flag.

## Honest constraints

* **Signing keys.** Retail Durango boot is fused and signed. Without keys, the
  achievable target is unsigned/homebrew execution and platform bring-up, not
  retail discs. Anyone claiming otherwise is guessing.
* **Documentation.** Large parts of the GPU and hypervisor are undocumented;
  progress there is reverse-engineering work, not typing.
* **Timeline.** M1–M2 were weeks. M3–M5 are months. M6 is a project of its own
  (compare xemu and Xenia, both multi-year efforts with teams behind them).

## Near-term backlog (next few turns)

1. Model MXCSR: rounding modes, the exception-mask bits, `ldmxcsr`/`stmxcsr`,
   and the sticky exception flags, so the fuzzer can vary rounding instead of
   pinning the default.
2. `int 0x20` / `int imm8` and a minimal IDT, so `int3` stops being a special
   case before M3 needs real faults.
3. 0x67 address-size overrides (32-bit addressing in 64-bit mode), currently
   parsed and ignored.
4. Multi-core: the Jaguar has eight cores; `LOCK` semantics only matter once a
   second core exists.
5. AVX/AVX2, so the same auto-vectorised gate can be run at the instruction
   level Scarlett (Series S|X) actually uses.
