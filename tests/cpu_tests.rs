//! Integration tests for the x86-64 interpreter.
//!
//! The fixtures are real GNU-as assembly or real gcc -O2 output, built by
//! `build.rs`, so these assertions exercise encoded instructions rather than a
//! Rust-side model. `tools/diff_unicorn.py` cross-checks the same binaries
//! against Unicorn.

use xone::cpu::{
    Cpu, Stop, PROGRAM_BASE, RAX, RBX, RCX, RDX, RDI, R8, R9, R10, R11, R12, R13, RSI, RSP,
};

fn run_fixture(name: &str) -> Cpu {
    let path = format!("{}/{}.bin", env!("FIXTURE_DIR"), name);
    let image = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut cpu = Cpu::new();
    cpu.load_image(PROGRAM_BASE, &image);
    cpu.reset(PROGRAM_BASE);
    let stop = cpu.run(200_000);
    assert_eq!(stop, Stop::Int3, "{name} stopped early: {stop:?}");
    cpu
}

fn f64_bits(value: f64) -> u64 {
    value.to_bits()
}

#[test]
fn arithmetic_results_and_widths() {
    let c = run_fixture("arith");
    assert_eq!(c.r[RAX], 0, "add overflow");
    assert_eq!(c.r[RBX], 0x80, "8-bit add keeps upper bits zeroed");
    assert_eq!(c.r[RCX], 0x1000 - 1);
    assert_eq!(c.r[RDX], 0x2FFD, "3-operand imul");
    assert_eq!(c.r[RSI], 0, "inc then dec");
    assert_eq!(c.r[RDI], 0x42, "or with immediate");
    assert_eq!(c.r[R8], 0xDEAD_BEEF, "mov r8d zero-extends");
    assert_eq!(c.r[R9], 0xBEEF, "movzx");
    assert_eq!(c.r[R10], 0xFFFF_FFFF_FFFF_FFEF, "movsx sign-extends");
    assert_eq!(c.r[R11], u64::MAX, "neg/not");
}

#[test]
fn arithmetic_flags() {
    let c = run_fixture("arith");
    assert!(c.flags.sf, "test r11,r11 sets SF");
    assert!(!c.flags.zf, "test r11,r11 clears ZF");
}

#[test]
fn loops_calls_and_branches() {
    let c = run_fixture("control");
    assert_eq!(c.r[RAX], 5050, "sum 1..=100");
    assert_eq!(c.r[RDI], 2000, "call/ret through double_it");
    assert_eq!(c.r[R12], 0, "je must be taken");
    assert_eq!(c.r[R13], 0);
}

#[test]
fn effective_addresses() {
    let c = run_fixture("memory");
    assert_eq!(c.r[RDX], 40, "scaled SIB load of table[3]");
    assert_eq!(c.r[R8], 40, "store then load through disp8");
    assert_eq!(c.r[R9], 0x1234, "dword store/load");
    assert_eq!(c.r[R10], 40, "SIB with base+index*8+disp8");
    assert_eq!(c.r[RAX], 3);
}

#[test]
fn shifts_and_rotates() {
    let c = run_fixture("shifts");
    assert_eq!(c.r[RAX], 64, "shl imm, shl cl, sar imm");
    assert_eq!(c.r[RBX], 0x3FFF_FFFF_FFFF_FFFF, "sar then logical shr");
    assert_eq!(c.r[RDX], 0x0F, "shr by cl");
    assert_eq!(c.r[R9], 1, "shr 63 of sign bit");
    assert_eq!(c.r[RDI], 10, "rol 1");
}

#[test]
fn stack_and_calls() {
    let c = run_fixture("stack");
    assert_eq!(c.r[RAX], 0x1111, "push/pop register");
    assert_eq!(c.r[RBX], 0x2222, "push imm32/pop register");
    assert_eq!(c.r[RDX], 5);
    assert_eq!(c.r[RDI], 21, "call into triple (lea rdi,[rdi+rdi*2])");
    assert_eq!(c.r[RSP], xone::cpu::STACK_TOP, "stack is balanced");
}

#[test]
fn sse2_packed_and_scalar_paths() {
    let c = run_fixture("sse");
    // paddd then psubd must give the original vector back.
    let want = 1u128 | (2u128 << 32) | (3u128 << 64) | (4u128 << 96);
    assert_eq!(c.xmm[0], want, "paddd/psubd lane-wise 32-bit add");
    // pshufd $0x1b reverses the four lanes.
    let reversed = 4u128 | (3u128 << 32) | (2u128 << 64) | (1u128 << 96);
    assert_eq!(c.xmm[3], reversed, "pshufd immediate shuffle");
    // cvtsi2sd / addsd / mulsd / sqrtsd
    assert_eq!(c.xmm[8] as u64, f64_bits(7.0));
    assert_eq!(c.xmm[9] as u64, f64_bits(7.5), "addsd then mulsd");
    assert_eq!(c.xmm[10] as u64, f64_bits(1.5), "sqrtsd(2.25)");
    // cvttsd2si truncates; movd/movq move the low lane both ways.
    assert_eq!(c.r[RDX], 7, "cvttsd2si");
    assert_eq!(c.r[RCX], 7, "movd xmm -> gpr");
    assert_eq!(c.r[RAX], 7, "movq xmm -> gpr");
    // comisd 7.5, 1.5 -> greater than: CF = ZF = 0.
    assert!(!c.flags.cf && !c.flags.zf, "comisd flags for a > b");
}

#[test]
fn string_instructions_with_rep_and_direction_flag() {
    let c = run_fixture("strings");
    let src = c.mem.bytes(0x2000, 16);
    assert_eq!(src, (1..=16).collect::<Vec<u8>>(), "fixture data layout");
    assert_eq!(
        c.mem.bytes(0x2010, 16),
        (1..=16).collect::<Vec<u8>>(),
        "rep movsb copied the block"
    );
    // With DF set both pointers walk down, so the block keeps its order but is
    // written over the 0xaa fill from rep stosb.
    assert_eq!(
        c.mem.bytes(0x2020, 16),
        (1..=16).collect::<Vec<u8>>(),
        "std rep movsb copied the block downwards"
    );
    assert_eq!(c.r[R8], 0x1FFF, "DF made rsi walk backwards from src+15");
    assert_eq!(c.r[R9], 0x201F, "DF made rdi walk backwards from dst+31");
    assert_eq!(
        c.mem.bytes(0x2038, 8),
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        "movsq copied one qword"
    );
    assert_eq!(c.r[RAX], 4, "rep lodsb left src[3] in AL");
    assert_eq!(c.r[RCX], 0, "rep exhausted the counter");
    assert_eq!(c.r[RSI], 0x2008, "rsi advanced by the final movsq");
    assert_eq!(c.r[RDI], 0x2040, "rdi advanced by the final movsq");
    assert!(c.flags.zf, "repe cmpsb over equal blocks sets ZF");
}

#[test]
fn cpuid_reports_amd_identity_and_only_implemented_features() {
    let mut cpu = Cpu::new();
    // mov eax, 0 ; cpuid ; int3
    cpu.load_image(PROGRAM_BASE, &[0xB8, 0, 0, 0, 0, 0x0F, 0xA2, 0xCC]);
    cpu.reset(PROGRAM_BASE);
    assert_eq!(cpu.run(8), Stop::Int3);

    let mut vendor = Vec::new();
    for value in [cpu.r[RBX], cpu.r[RDX], cpu.r[RCX]] {
        vendor.extend_from_slice(&(value as u32).to_le_bytes());
    }
    assert_eq!(
        &vendor,
        b"AuthenticAMD",
        "Durango is an AMD Jaguar, so CPUID vendor must say so"
    );

    // Leaf 1: SSE2 advertised, AVX deliberately not (we do not execute it).
    let mut cpu = Cpu::new();
    cpu.load_image(PROGRAM_BASE, &[0xB8, 1, 0, 0, 0, 0x0F, 0xA2, 0xCC]);
    cpu.reset(PROGRAM_BASE);
    assert_eq!(cpu.run(8), Stop::Int3);
    let edx = cpu.r[RDX] as u32;
    let ecx = cpu.r[RCX] as u32;
    assert_ne!(edx & (1 << 26), 0, "SSE2 must be advertised");
    assert_ne!(edx & (1 << 25), 0, "SSE must be advertised");
    assert_eq!(ecx & (1 << 28), 0, "AVX must not be advertised");
    assert_eq!(
        ecx & (1 << 19),
        0,
        "SSE4.1 must not be advertised (unimplemented)"
    );
}

#[test]
fn rdtsc_is_deterministic_and_monotonic() {
    let mut cpu = Cpu::new();
    // rdtsc ; rdtsc ; int3
    cpu.load_image(PROGRAM_BASE, &[0x0F, 0x31, 0x0F, 0x31, 0xCC]);
    cpu.reset(PROGRAM_BASE);
    assert_eq!(cpu.run(8), Stop::Int3);
    // The first rdtsc ran on step 0, the second on step 1.
    assert_eq!(cpu.r[RAX] as u32, 0x1000 + 4, "second read is four ticks later");
    assert_eq!(cpu.r[RDX], 0, "counter stays below 2^32 in a short run");
}

/// Execute one compiler-gate image and compare its answer block against the
/// output of the host build of the same source.
fn check_compiler_gate(name: &str, fields: usize) {
    let dir = env!("FIXTURE_DIR");
    let image = std::fs::read(format!("{dir}/{name}.bin"))
        .unwrap_or_else(|e| panic!("build.rs should have produced {name}.bin: {e}"));
    let mut cpu = Cpu::new();
    cpu.load_image(PROGRAM_BASE, &image);
    cpu.reset(PROGRAM_BASE);
    let stop = cpu.run(5_000_000);
    assert_eq!(
        stop,
        Stop::Int3,
        "{name} stopped on {stop:?} at rip=0x{:x} opcode={:02x}",
        cpu.rip,
        cpu.last_opcode
    );

    let expected = std::fs::read_to_string(format!("{dir}/{name}.expected"))
        .unwrap_or_else(|e| panic!("build.rs should have run the {name} reference: {e}"));
    let want: Vec<u64> = expected
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| u64::from_str_radix(line.trim(), 16).expect("hex answer"))
        .collect();
    assert_eq!(want.len(), fields, "{name} answer block size");

    for (index, expected) in want.iter().enumerate() {
        let got = cpu.mem.read(0x9000 + 8 * index as u64, 64);
        assert_eq!(
            got, *expected,
            "{name} field {index} differs from native execution"
        );
    }
}

#[test]
fn compiler_generated_code_matches_native_execution() {
    check_compiler_gate("m2_gate", 15);
}

#[test]
fn auto_vectorised_code_matches_native_execution() {
    check_compiler_gate("simd_gate", 21);
}

#[test]
fn unsupported_opcode_stops_instead_of_guessing() {
    let mut cpu = Cpu::new();
    // D9 E8 = fld1: x87, which this core does not implement.
    cpu.load_image(PROGRAM_BASE, &[0xD9, 0xE8]);
    cpu.reset(PROGRAM_BASE);
    assert_eq!(cpu.run(16), Stop::BadOpcode);
    assert_eq!(cpu.last_insn, PROGRAM_BASE, "fault address is reported");
    assert_eq!(cpu.last_opcode, 0xD9);
}

#[test]
fn step_budget_is_respected() {
    let mut cpu = Cpu::new();
    // jmp $ : an infinite loop, so only the budget can stop us.
    cpu.load_image(PROGRAM_BASE, &[0xEB, 0xFE]);
    cpu.reset(PROGRAM_BASE);
    assert_eq!(cpu.run(1_000), Stop::StepLimit);
    assert_eq!(cpu.steps, 1_000);
}
