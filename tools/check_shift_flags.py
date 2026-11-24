#!/usr/bin/env python3
"""Check the emulator's shift/rotate flag rules against the real CPU.

`tests/c/shift_flags.c` emits the exact instruction bytes for each case, seeds
the incoming flags itself, runs them natively, and prints the bytes it ran.
This script feeds those same bytes to the emulator, so the encoding is never
duplicated: whatever the hardware executed is what the emulator is asked to
execute.

It exists because the reference (QEMU via Unicorn) disagrees with hardware on
SHLD/SHRD overflow for counts other than one: hardware keeps computing a
formula, QEMU does not. The emulator follows the hardware, which is why the
differential fuzzer reports those cases in their own documented bucket and why
this oracle exists to prove the rule rather than assert it.

Usage:
    .venv/bin/python tools/check_shift_flags.py [--verbose]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XONE = os.path.join(ROOT, "target", "release", "xone")
BUILD = os.path.join(ROOT, "build")
SOURCE = os.path.join(ROOT, "tests", "c", "shift_flags.c")

OPS = ["rol", "ror", "rcl", "rcr", "shl", "shr", "sar", "shld", "shrd"]
WIDTHS = [16, 32, 64]
COUNTS = [0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63]
# (value, fill) pairs: zero, a single low bit, the sign bit, all ones, a mixed
# pattern, and a pair chosen so the shift-out bit and the sign disagree.
PAIRS = [
    (0x0, 0x0),
    (0x1, 0x0),
    (0x8000_0000_0000_0000, 0x0),
    (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
    (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
    (0xAAAA_5555_AAAA_5555, 0x5555_AAAA_5555_AAAA),
]
# Incoming flags: nothing set, carry set (matters for rcl/rcr), overflow set,
# and both.
FLAGS = [0x000, 0x001, 0x800, 0x881]


def compile_oracle(out_dir):
    binary = os.path.join(out_dir, "shift_flags")
    res = subprocess.run(["gcc", "-O1", "-o", binary, SOURCE], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("gcc failed:\n" + res.stderr)
    return binary


def hardware_cases(oracle, cases):
    stdin = "".join(
        f"{op} {width} {value:016x} {fill:016x} {count} {flags:03x}\n"
        for op, width, value, fill, count, flags in cases
    )
    res = subprocess.run([oracle], input=stdin, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("shift_flags failed:\n" + res.stderr)
    out = []
    for line in res.stdout.split("\n"):
        if not line.strip():
            continue
        program, flags, result = line.split()
        out.append((bytes.fromhex(program), int(flags, 16), int(result, 16)))
    if len(out) != len(cases):
        sys.exit(f"shift_flags returned {len(out)} results for {len(cases)} cases")
    return out


def emulator_state(program, workdir):
    path = os.path.join(workdir, "prog.bin")
    with open(path, "wb") as fh:
        fh.write(program)
    res = subprocess.run([XONE, "run", path, "--state"], capture_output=True, text=True)
    match = re.search(r"^STATE .*rflags=0x([0-9a-f]+).* rbx=0x([0-9a-f]+)", res.stdout, re.M)
    if not match:
        detail = res.stdout.strip() or res.stderr.strip()
        sys.exit(f"no state from xone for {program.hex()}:\n{detail}")
    return int(match.group(1), 16) & 0x8C5, int(match.group(2), 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(XONE):
        sys.exit(f"{XONE} is missing; run `cargo build --release` first")

    cases = []
    for op in OPS:
        for width in WIDTHS:
            for count in COUNTS:
                # A 16-bit SHLD/SHRD count is masked to five bits, so 17..31 ask
                # for a shift wider than the operand. The architecture leaves
                # that undefined and hardware answers with microcode artefacts
                # that no formula reproduces; this core keeps the natural
                # double-width reading instead (and the differential reference
                # agrees with it), so those rows are out of scope here rather
                # than failing.
                if op in ("shld", "shrd") and width == 16 and count > 16:
                    continue
                for value, fill in PAIRS:
                    for flags in FLAGS:
                        cases.append((op, width, value, fill, count, flags))

    failures = 0
    with tempfile.TemporaryDirectory(dir=BUILD) as workdir:
        oracle = compile_oracle(workdir)
        reference = hardware_cases(oracle, cases)
        for (op, width, value, fill, count, flags), (program, want_flags, want_result) in zip(
            cases, reference
        ):
            # The oracle ends in `ret`; the emulator stops on the int3 sentinel.
            assert program[-1] == 0xC3, program.hex()
            body = program[:-1] + b"\xcc"
            got_flags, got_rbx = emulator_state(body, workdir)
            keep = (1 << width) - 1
            if got_flags != want_flags or (got_rbx & keep) != (want_result & keep):
                failures += 1
                if failures <= 20 or args.verbose:
                    print(
                        f"DIFF {op} {width}-bit value={value:016x} fill={fill:016x} "
                        f"count={count} flags_in={flags:03x}: "
                        f"ours flags={got_flags:03x} result={got_rbx & keep:016x} "
                        f"hardware flags={want_flags:03x} result={want_result & keep:016x}"
                    )

    print(
        f"shift flags: {len(cases)} cases against native hardware, "
        f"{failures} mismatched"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
