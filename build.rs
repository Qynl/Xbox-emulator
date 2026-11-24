//! Prepare the test fixtures at build time.
//!
//! Two kinds, both built from real sources rather than hand-encoded bytes:
//!
//!  * `tests/asm/*.S` — GNU-as assembly, assembled with `gcc -c` and flattened
//!    with `objcopy`. A wrong opcode is an assembler error, not a silently
//!    wrong test.
//!  * `tests/c/m2_gate.c` — ordinary C, compiled twice with `gcc -O2`: once as
//!    a freestanding x86-64 image for the interpreter, and once as a host
//!    binary whose output is the expected answer. That is the milestone-2
//!    gate: compiler-generated code must produce the same results as the host.
//!
//! Everything lands in `build/fixtures` (git-ignored) so both `cargo test` and
//! the Python differential harnesses see the same files.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn os(s: &str) -> OsString {
    OsString::from(s)
}

fn path(p: &Path) -> OsString {
    p.as_os_str().to_os_string()
}

/// Run a tool, panicking with its stderr if it fails (build scripts should die
/// loudly rather than produce a fixture nobody validated).
fn run(program: &str, args: &[OsString], what: &str) -> Vec<u8> {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {program} ({what}): {e}"));
    if !output.status.success() {
        panic!(
            "{what} failed\n  program: {program}\n  args: {args:?}\n  status: {}\n  stderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.stdout
}

/// Build one compiler gate: the host binary whose output is the reference, and
/// the freestanding image the interpreter executes, both from the same source
/// at the same optimisation level.
fn build_gate(c_dir: &Path, fixtures: &Path, name: &str, source: &str, opt: &str) {
    let gate_c = c_dir.join(source);
    let gate_entry = c_dir.join("entry.S");
    let gate_ld = c_dir.join("gate.ld");
    println!("cargo:rerun-if-changed={}", gate_c.display());
    let what = format!("{name} ({opt})");

    let native = fixtures.join(format!("{name}_native"));
    run(
        "gcc",
        &[
            os(opt),
            os("-fno-builtin"),
            os("-DHOST_TEST"),
            os("-o"),
            path(&native),
            path(&gate_c),
        ],
        &format!("build native reference for {what}"),
    );
    let expected = run(
        native.to_str().expect("utf-8 fixture path"),
        &[],
        &format!("run native reference for {what}"),
    );
    fs::write(fixtures.join(format!("{name}.expected")), expected)
        .expect("write gate expectations");

    let gate_obj = fixtures.join(format!("{name}.o"));
    let entry_obj = fixtures.join(format!("{name}_entry.o"));
    let gate_elf = fixtures.join(format!("{name}.elf"));
    run(
        "gcc",
        &[
            os(opt),
            os("-ffreestanding"),
            os("-fno-builtin"),
            os("-fno-pie"),
            os("-fno-stack-protector"),
            os("-fno-asynchronous-unwind-tables"),
            os("-fno-unwind-tables"),
            os("-fno-ident"),
            os("-c"),
            os("-o"),
            path(&gate_obj),
            path(&gate_c),
        ],
        &format!("compile freestanding image for {what}"),
    );
    run(
        "gcc",
        &[
            os("-c"),
            os("-nostdlib"),
            os("-fno-pie"),
            os("-o"),
            path(&entry_obj),
            path(&gate_entry),
        ],
        &format!("assemble entry stub for {what}"),
    );
    run(
        "ld",
        &[
            os("-static"),
            os("-T"),
            path(&gate_ld),
            os("-o"),
            path(&gate_elf),
            path(&entry_obj),
            path(&gate_obj),
        ],
        &format!("link image for {what}"),
    );
    run(
        "objcopy",
        &[
            os("-O"),
            os("binary"),
            path(&gate_elf),
            path(&fixtures.join(format!("{name}.bin"))),
        ],
        &format!("flatten image for {what}"),
    );
    let _ = fs::remove_file(&gate_obj);
    let _ = fs::remove_file(&entry_obj);
    let _ = fs::remove_file(&gate_elf);
}

fn main() {
    println!("cargo:rerun-if-changed=tests/asm");
    println!("cargo:rerun-if-changed=tests/c");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let fixtures = manifest.join("build").join("fixtures");
    fs::create_dir_all(&fixtures).expect("create build/fixtures");

    // ------------------------------------------------------------ assembly
    let asm_dir = manifest.join("tests").join("asm");
    let mut sources: Vec<PathBuf> = fs::read_dir(&asm_dir)
        .expect("tests/asm must exist")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|p| p.extension().map(|e| e == "S").unwrap_or(false))
        .collect();
    sources.sort();

    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let obj = fixtures.join(format!("{stem}.o"));
        let bin = fixtures.join(format!("{stem}.bin"));
        run(
            "gcc",
            &[
                os("-c"),
                os("-nostdlib"),
                os("-fno-pie"),
                os("-o"),
                path(&obj),
                path(src),
            ],
            &format!("assemble {stem}.S"),
        );
        run(
            "objcopy",
            &[
                os("-O"),
                os("binary"),
                os("-j"),
                os(".text"),
                path(&obj),
                path(&bin),
            ],
            &format!("flatten {stem}.o"),
        );
        let _ = fs::remove_file(&obj);
    }

    // ------------------------------------------------- compiler gates (M2)
    let c_dir = manifest.join("tests").join("c");
    println!("cargo:rerun-if-changed={}", c_dir.join("entry.S").display());
    println!("cargo:rerun-if-changed={}", c_dir.join("gate.ld").display());
    build_gate(&c_dir, &fixtures, "m2_gate", "m2_gate.c", "-O2");
    build_gate(&c_dir, &fixtures, "simd_gate", "simd_gate.c", "-O3");

    println!(
        "cargo:rustc-env=FIXTURE_DIR={}",
        fixtures.to_string_lossy()
    );
}
