# xone — an Xbox emulator, built from the ground up

This is a real emulator project, not a mock-up. It targets the x86-64 Xbox
consoles: the Xbox One family (Microsoft codename **Durango** — an 8-core AMD
Jaguar x86-64 APU with a custom GCN GPU, 8 GiB of unified GDDR3, and a
hypervisor-mediated OS) and, at the instruction-set level, the Xbox Series S and
Series X (**Scarlett** — AMD Zen 2 x86-64), which execute the same
architecture.

Which generation is covered by what:

| console | CPU | status |
| --- | --- | --- |
| Xbox One, One S, One X (Durango) | AMD Jaguar, x86-64 | this core |
| Xbox Series S, Series X (Scarlett) | AMD Zen 2, x86-64 | same core; AVX/AVX2 not implemented yet |
| Xbox 360 (Xenon) | IBM PowerPC, triple-core | needs its own core — see the roadmap |
| Original Xbox (2001) | Intel Pentium III, 32-bit x86 | needs its own core — see the roadmap |

So this is not one decoder for every Xbox: the two older consoles are different
CPU architectures and need separate cores behind the same machine shell. What
is shared — memory model, devices, loader, debugger — is what makes a single
project able to grow that way, and the x86-64 core here already spans the two
newest generations.

Nothing here pretends to be further along than it is. The repository currently
contains **milestones 1 and 2: a working x86-64 CPU interpreter with SSE2**,
verified differentially against QEMU's core, plus a flat memory model and the
bring-up shell that will grow into the console.

## What works right now

**Integer core.** mov/movzx/movsx/movsxd, lea, the full integer ALU and flag
model, mul/imul/div/idiv, every shift and rotate (`shl`, `shr`, `sar`, `rol`,
`ror`, `rcl`, `rcr`, `shld`, `shrd`), push/pop, call/ret/leave, every Jcc,
cmovcc, setcc, bsf/bsr/tzcnt/lzcnt/popcnt, bswap, cdq/cqo/cwde/cdqe, 8/16/32/64-bit
operand sizes, AH/CH/DH/BH, REX, ModRM/SIB, RIP-relative addressing.

**String instructions.** movs/stos/lods/cmps/scas in all widths, with REP /
REPE / REPNE, the DF direction flag, and RCX accounting that matches hardware.

**SSE2.** All 16 XMM registers. Moves (`movaps`, `movups`, `movdqa`, `movdqu`,
`movss`, `movsd`, `movd`, `movq`, `movlps`, `movhps`), packed and scalar float
arithmetic in single and double precision (`add`, `sub`, `mul`, `div`, `sqrt`,
`min`, `max`), `comiss`/`ucomiss`/`comisd`/`ucomisd` flag semantics,
`cmpps`/`cmpsd` predicates, `shufps`/`shufpd`, and the conversions gcc emits
(`cvtsi2ss`, `cvtsi2sd`, `cvttss2si`, `cvttsd2si`, `cvtss2si`, `cvtsd2si`,
`cvtss2sd`, `cvtsd2ss`, `cvtps2pd`, `cvtpd2ps`, `cvtdq2ps`, `cvtpd2dq`,
`cvttpd2dq`, `cvtdq2pd`).

**SSE2 packed integers.** The whole arithmetic set an auto-vectorising compiler
emits: `padd{b,w,d,q}`, `psub{b,w,d,q}`, saturating `padds`/`paddus`/`psubs`/
`psubus` in byte and word forms, `pxor`, `pand`, `pandn`, `por`, `pcmpeq{b,w,d}`,
`pcmpgt{b,w,d}`, `packsswb`/`packuswb`/`packssdw`, `pmullw`/`pmulhw`/`pmulhuw`,
`pmaddwd`/`psadbw`, `pminub`/`pmaxub`/`pminsw`/`pmaxsw`, `pshufd`, the whole
`punpck` family, packed shifts by immediate, by vector count and by bytes
(`pslldq`/`psrldq`), the mask extracts `pmovmskb`/`movmskps`/`movmskpd` and the
lane extract `pextrw` (register forms only — GAS rejects a memory operand, so
that encoding is refused rather than invented).

**System-ish instructions.** `cpuid`, `rdtsc`, fences (as no-ops on a
single in-order core), `syscall`, `int3`, `hlt`, `ud2`.

**Machine shell.** Deterministic execution: instruction budget, stop reasons,
register/flag/SSE/memory state dumps, `--trace`, `--reg` seeding, `--dump`, and
a hard stop on anything outside the implemented subset.

### Deliberate honesty

* Unsupported opcodes stop with `bad-opcode` and report the address and the
  opcode bytes. The core never silently skips an instruction it does not
  understand; silent wrongness is worse than a hard stop when you are chasing a
  boot hang.
* `cpuid` reports **what this core implements**, not what a real Jaguar does.
  Advertising AVX or SSE4 would let a compiler emit instructions we refuse, so
  only SSE/SSE2/SSE3/SSSE3 are claimed.
* `rdtsc` is deterministic — four ticks per retired instruction — because
  reproducible runs matter more than a plausible frequency during bring-up.
* Architecturally undefined encodings are refused rather than invented, e.g. a
  16-bit `bswap`, or `pmovmskb`/`movmskps`/`movmskpd` with a memory operand.
* **Flags the manual leaves undefined are taken from a measurement, not from a
  guess.** Intel defines OF for shifts and rotates only at a count of one, yet
  silicon keeps computing a formula for every count; `tools/check_shift_flags.py`
  runs 8184 shift/rotate cases on the real CPU and the core reproduces them
  exactly, including the quirks (a 16-bit rotate count of `16` still writes CF
  and OF, a full-turn `rcl`/`rcr` still writes them, `shld` at a count equal to
  the operand size sets OF from the carry).
* One intentional divergence from QEMU is documented in `src/cpu.rs`: on the
  `xchg rAX, r` encoding, a `0x66` prefix is honoured (16-bit exchange) as
  hardware does, while QEMU ignores it. The fuzzer skips that one form.
* **NaN selection follows the hardware, not QEMU.** When both operands of an
  SSE add/sub/mul/div are NaNs, the result is the destination operand's NaN,
  quietened -- measured on a real CPU by `tests/c/nan_rule.c`, which emits the
  instruction bytes and runs them natively. QEMU's soft-float path returns the
  operand with the larger significand for the packed forms and for double
  precision, so the fuzzer reports those cases in their own bucket instead of
  counting them as matches or as failures.
* **Nothing is left to the host FPU by accident.** LLVM is free to commute a
  floating-point add, which silently changes which NaN payload survives; the
  arithmetic was doing exactly that for packed lanes. The rule above is
  implemented on raw bit patterns so the result is deterministic.

## Verification

Six layers, all run by `tools/test.sh`:

1. **`cargo test`** — 14 integration tests over real inputs: GNU-as assembly
   fixtures and two gcc programs, assembled/compiled by `build.rs`.
2. **Compiler gates (milestone 2).** Two C programs are each compiled twice: as
   a freestanding x86-64 image our core executes, and as a host binary, and the
   results must be identical. `tests/c/m2_gate.c` is `gcc -O2` integer and
   scalar-SSE code; `tests/c/simd_gate.c` is `gcc -O3` plain C whose loops the
   vectoriser turns into packed-integer code — byte/word adds, `pmaddwd` dot
   products, `pminub`/`pmaxsw` reductions, shifts, interleaves, `pcmpeqb` masks
   and `addps`/`mulpd` — so the packed ops are proven by compiler output, not
   only by the fuzzer. The vectoriser was what demanded `pextrw`, which the
   core did not have until this gate existed.
3. **Fixture differential** (`tools/diff_unicorn.py`) — every fixture runs in
   our core *and* in Unicorn (QEMU-derived x86-64). All 16 GPRs, the arithmetic
   flags and a hash of the whole low memory window must match exactly, and both
   sides must reach the `int3` sentinel. Today all 9 fixtures match, including
   1306 instructions of `-O2` and 5138 of `-O3` vectorised output.
4. **Native floating-point oracle** (`tests/c/nan_rule.c` +
   `tools/check_nan_rule.py`) — the SSE add/sub/mul/div/min/max instruction
   bytes are emitted into executable memory and run on the real CPU, and the
   emulator must agree with that hardware for 3656 operand cases: quiet and
   signalling NaNs with distinct payloads and signs, infinities, signed zeros,
   denormals and random bit patterns, in all four scalar/packed, single/double
   forms. This is what makes "we follow the hardware" a checked claim rather
   than a comment.
5. **Native shift/rotate flag oracle** (`tests/c/shift_flags.c` +
   `tools/check_shift_flags.py`) — 8184 cases over `rol`, `ror`, `rcl`, `rcr`,
   `shl`, `shr`, `sar`, `shld`, `shrd`, in 16/32/64-bit forms, with counts from
   0 to 63 and four incoming flag states. The oracle emits the instruction bytes
   under test, seeds the flags itself and runs them on the real CPU; the
   emulator then runs *those same bytes*, so no encoding is duplicated between
   the two sides. This is what pins down the flags Intel leaves undefined for a
   count other than one: hardware keeps computing a formula, and the reference
   (QEMU) computes a different one for `rcl`/`rcr` and `shld`/`shrd`.
6. **Randomised differential fuzzing** (`tools/fuzz_unicorn.py`) — 1200 cases
   by default (400 x 3 seeds; `FUZZ_ITERATIONS`/`FUZZ_SEEDS` widen it). Each
   case seeds all 16 GPRs and all 16 XMM registers with float and integer edge
   cases, loads a 512-byte mixed arena at `0x4000`, and runs 6-10 randomly
   encoded instructions mixing SSE/SSE2 (moves, packed and scalar arithmetic,
   comparisons, shuffles/unpacks, packed shifts including the byte shifts
   `pslldq`/`psrldq`, the lane extract `pextrw`, conversions, sqrt/min/max,
   `movlps`/`movhps`, memory operands at varied alignments), integer ALU/shift
   code, and string instructions with REP/REPE/REPNE under both settings of DF.
   GPRs, all XMM registers, the arithmetic flags and every byte of the low
   memory window must match Unicorn exactly. Measured: 1186 matched field for
   field, 10 reference NaN-selection differences, 4 cases Unicorn faulted on
   (not comparisons), 0 divergences. The default corpus contains 95 `pextrw`
   and 98 `pslldq`/`psrldq` instances, i.e. the instructions the `-O3` gate
   newly covers are differentially compared too, not just compiled.
   Cases this emulator refuses are failures, not skips.
   `rcl`/`rcr` and `shld`/`shrd` appear only with a count of one in this corpus:
   for other counts OF is undefined, the reference computes its own formula, and
   a following `setcc` would turn that difference into a register difference.
   Those counts are verified against hardware by layer 5 instead.

## What is still unverified

* **MXCSR is not modelled.** Rounding mode and exception masks are fixed at the
  reset default (round-to-nearest, all masked). The fuzzer therefore keeps the
  default MXCSR on both sides; behaviour under other rounding modes, and
  denormal handling controlled by MXCSR, is untested.
* **Instruction families absent entirely:** AVX/AVX2, x87, MMX, SSE3/SSSE3/SSE4
  (no `haddps`, `lddqu`, `movddup`, `pshufb`, ...). `cpuid` does not advertise
  them.
* **No MMU, interrupts, privilege levels or I/O ports.** Memory is flat and
  always readable; page faults, `#GP`/`#PF` delivery and `int imm8` do not
  exist yet. String instructions are therefore verified for their REP/DF/RCX
  semantics over flat memory, not for fault behaviour across a page boundary.
* **`0F 71/72/73` with a memory operand is refused**, because it is not a
  defined instruction: GAS rejects `psrlw $3, (%rax)` with "operand size
  mismatch" and QEMU silently shifts the register named by the r/m field
  instead. The fuzzer generates the register form only.
* **16-bit `shld`/`shrd` with a count above 16** (counts 17-31 are reachable
  because a 16-bit count is masked to five bits) shift wider than the operand.
  The architecture leaves that undefined; hardware answers with what look like
  microcode artefacts that no formula reproduces, so this core keeps the natural
  double-width reading (the same one the reference uses) and the native oracle
  skips those rows rather than pretending to match.

The fixture harnesses need Unicorn, kept out of the Rust dependency graph:

```bash
python3 -m venv .venv && .venv/bin/pip install unicorn capstone
```

## Build and run

Requires Rust (https://rustup.rs), plus `gcc`/`binutils` (fixtures are built by
`build.rs` with the real toolchain).

```bash
cargo build --release
./target/release/xone run build/fixtures/m2_gate.bin    # any flat x86-64 image
./target/release/xone run <image> --trace               # per-instruction trace
./target/release/xone run <image> --state               # machine-readable state
./target/release/xone run <image> --reg rax=0x10        # seed registers
./target/release/xone run <image> --dump 0x9000:64      # hex dump after the run
```

Full check (build + tests + Unicorn fixtures + native oracles + fuzzing):

```bash
bash tools/test.sh
```

### Throughput

This is an interpreter, so speed is instructions per second, not frames:

```bash
.venv/bin/python tools/bench.py --body mixed     # MIPS, median of 3 runs
.venv/bin/python tools/bench.py --repeat 7 \
    --binary build/xone-before --binary target/release/xone
```

`tools/bench.py` hand-assembles a loop (integer ALU, loads/stores, packed
SSE2) and runs it to a fixed instruction budget, failing loudly unless the whole
budget retired on the step limit. Repeat `--binary` to A/B two builds with
interleaved runs, which is the only comparison worth trusting here: on this
2-vCPU cloud VM, run-to-run noise is about +-3%. With that protocol, replacing
the byte-at-a-time memory accessors with one unaligned load and a masked
read-modify-write store took the mixed body from 42.0 to 51.0 MIPS and the
SSE-only body from 39.2 to 45.6 (medians of 7 interleaved runs). Absolute
numbers belong to the host that produced them; the ratio is the result.

## Reality check on "Xbox One on PC"

Writing an Xbox One emulator is a multi-year, multi-person effort. This project
is honest about the hard parts:

* **Firmware and keys.** Retail consoles boot through signed bootloaders fused
  into the APU. Without those keys, a clean-room emulator cannot run retail
  game executables; realistic goals are homebrew/unsigned code, devkit-style
  images, and the documented-surface parts of the platform.
* **The GPU.** Durango's command processor and GCN shader ISA need a real
  rasteriser, not a shim. This is the largest single milestone.
* **The hypervisor and OS.** Games expect a partitioned environment with
  hypercalls, virtual GPU channels and a title runtime.

See [ROADMAP.md](ROADMAP.md) for the milestone plan and what each one unlocks.

## Layout

```
src/cpu.rs        x86-64 interpreter (decode + execute, integer + SSE2)
src/memory.rs     flat physical memory
src/flags.rs      RFLAGS model
src/main.rs       CLI bring-up shell
tests/asm/*.S     real assembly fixtures
tests/c/          compiler gate: freestanding C, entry stub, linker script
tests/cpu_tests.rs integration tests
build.rs          assembles/compiles fixtures at build time
tools/            differential fixture harness, fuzzer, native oracles,
                  throughput benchmark, test runner
```

## Legal

No Microsoft firmware, keys, game data or copyrighted ROMs are included or
downloaded by this project. Bring your own legally obtained dumps.
