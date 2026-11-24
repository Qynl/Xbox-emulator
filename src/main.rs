//! `xone` command line front-end: load a flat x86-64 image and run it.
//!
//! This is the bring-up shell of the emulator. Today it boots flat binaries
//! (test fixtures, hand-written demos, compiler-generated freestanding C). The
//! PE loader and Durango boot path arrive in later milestones.

use std::process::ExitCode;

use xone::cpu::{Cpu, PROGRAM_BASE};
use xone::Stop;

const REG_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

const HASH_BYTES: usize = 0x10000;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn parse_num(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}

fn usage() {
    eprintln!(
        "xone {} - Xbox emulator bring-up shell (x86-64 core: Xbox One, Series S|X)\n\n\
         usage: xone run <image.bin> [options]\n\n\
         options:\n  \
         --entry <addr>     load/entry address (default 0x{:x})\n  \
         --steps <n>        instruction budget (default 10000000)\n  \
         --reg <name=val>   seed a general-purpose register (repeatable)\n  \
         --xmm <idx=val>     seed an XMM register with a 128-bit hex value\n  \
         --load <addr:file> load a raw data blob at an address (repeatable)\n  \
         --dump <addr:len>  print a hex dump of memory after the run\n  \
         --trace            log each instruction's address and opcode\n  \
         --state            machine-readable final state (for test harnesses)",
        env!("CARGO_PKG_VERSION"),
        PROGRAM_BASE
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return ExitCode::SUCCESS;
    }
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    if args[0] != "run" {
        usage();
        return ExitCode::from(2);
    }

    let mut path: Option<String> = None;
    let mut entry = PROGRAM_BASE;
    let mut steps: u64 = 10_000_000;
    let mut state = false;
    let mut trace = false;
    let mut regs: [Option<u64>; 16] = [None; 16];
    let mut xmm: [Option<u128>; 16] = [None; 16];
    let mut loads: Vec<(u64, String)> = Vec::new();
    let mut dump: Option<(u64, usize)> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--entry" | "--steps" | "--reg" | "--xmm" | "--load" | "--dump" => {
                let flag = args[i].clone();
                i += 1;
                if i >= args.len() {
                    eprintln!("xone: {flag} needs a value");
                    return ExitCode::from(2);
                }
                let value = &args[i];
                match flag.as_str() {
                    "--entry" => entry = parse_num(value),
                    "--steps" => steps = parse_num(value),
                    "--reg" => {
                        let Some((name, raw)) = value.split_once('=') else {
                            eprintln!("xone: --reg expects <name>=<value>, got {value}");
                            return ExitCode::from(2);
                        };
                        let name = name.trim().to_ascii_lowercase();
                        let Some(idx) = REG_NAMES.iter().position(|n| *n == name) else {
                            eprintln!("xone: unknown register {name}");
                            return ExitCode::from(2);
                        };
                        regs[idx] = Some(parse_num(raw));
                    }
                    "--xmm" => {
                        let Some((index, raw)) = value.split_once('=') else {
                            eprintln!("xone: --xmm expects <index>=<value>, got {value}");
                            return ExitCode::from(2);
                        };
                        let index: usize = match index.trim().parse() {
                            Ok(v) if v < 16 => v,
                            _ => {
                                eprintln!("xone: XMM index must be 0..15, got {index}");
                                return ExitCode::from(2);
                            }
                        };
                        let raw = raw.trim();
                        let digits = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")).unwrap_or(raw);
                        match u128::from_str_radix(digits, 16) {
                            Ok(v) => xmm[index] = Some(v),
                            Err(_) => {
                                eprintln!("xone: bad 128-bit hex value for --xmm: {raw}");
                                return ExitCode::from(2);
                            }
                        }
                    }
                    "--load" => {
                        let Some((addr, file)) = value.split_once(':') else {
                            eprintln!("xone: --load expects <addr>:<file>, got {value}");
                            return ExitCode::from(2);
                        };
                        loads.push((parse_num(addr), file.to_string()));
                    }
                    _ => {
                        let Some((addr, len)) = value.split_once(':') else {
                            eprintln!("xone: --dump expects <addr>:<len>, got {value}");
                            return ExitCode::from(2);
                        };
                        dump = Some((parse_num(addr), parse_num(len) as usize));
                    }
                }
            }
            "--state" => state = true,
            "--trace" => trace = true,
            other => path = Some(other.to_string()),
        }
        i += 1;
    }

    let Some(path) = path else {
        usage();
        return ExitCode::from(2);
    };

    let image = match std::fs::read(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("xone: cannot read {path}: {e}");
            return ExitCode::from(1);
        }
    };

    let mut cpu = Cpu::new();
    cpu.load_image(entry, &image);
    for (addr, file) in &loads {
        match std::fs::read(file) {
            Ok(bytes) => cpu.load_image(*addr, &bytes),
            Err(e) => {
                eprintln!("xone: cannot read {file}: {e}");
                return ExitCode::from(1);
            }
        }
    }
    cpu.reset(entry);
    cpu.trace = trace;
    for (idx, value) in regs.iter().enumerate() {
        if let Some(value) = value {
            cpu.r[idx] = *value;
        }
    }
    for (idx, value) in xmm.iter().enumerate() {
        if let Some(value) = value {
            cpu.xmm[idx] = *value;
        }
    }

    let stop = cpu.run(steps);
    let memhash = fnv1a(&cpu.mem.bytes(0, HASH_BYTES));
    let rflags = cpu.flags.pack();

    if stop == Stop::BadOpcode {
        let ext = cpu
            .last_ext
            .map(|b| format!(" {b:02x}"))
            .unwrap_or_default();
        eprintln!(
            "xone: unsupported instruction at 0x{:016x}: opcode {:02x}{}",
            cpu.last_insn, cpu.last_opcode, ext
        );
    }

    if state {
        let mut line = format!(
            "STATE stop={} rip=0x{:016x} steps={} rflags=0x{:016x} memhash=0x{:016x}",
            stop.as_str(),
            cpu.rip,
            cpu.steps,
            rflags,
            memhash
        );
        for (idx, name) in REG_NAMES.iter().enumerate() {
            line.push_str(&format!(" {name}=0x{:016x}", cpu.r[idx]));
        }
        for (idx, value) in cpu.xmm.iter().enumerate() {
            line.push_str(&format!(" xmm{idx}=0x{value:032x}"));
        }
        println!("{line}");
    } else {
        println!("xone {}", env!("CARGO_PKG_VERSION"));
        println!("  image   {path} ({} bytes)", image.len());
        println!("  entry   0x{entry:016x}");
        println!("  stopped {}", stop.as_str());
        println!("  rip     0x{:016x}", cpu.rip);
        println!("  steps   {}", cpu.steps);
        println!("  rflags  0x{rflags:016x}");
        println!("  memhash 0x{memhash:016x}");
        println!();
        for (idx, name) in REG_NAMES.iter().enumerate() {
            println!("  {name:>3}  0x{:016x}", cpu.r[idx]);
        }
        println!();
        for (idx, value) in cpu.xmm.iter().enumerate() {
            println!("  xmm{idx:<2} 0x{value:032x}");
        }
    }

    if let Some((addr, len)) = dump {
        let bytes = cpu.mem.bytes(addr, len.min(1 << 20));
        let mut hex = String::with_capacity(bytes.len() * 2);
        for b in &bytes {
            hex.push_str(&format!("{b:02x}"));
        }
        println!("DUMP {addr:016x} {hex}");
    }

    ExitCode::SUCCESS
}
