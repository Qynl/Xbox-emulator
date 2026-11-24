#!/usr/bin/env python3
"""Check the emulator's SSE floating-point rules against the real CPU.

`tests/c/nan_rule.c` emits the exact instruction bytes under test into
executable memory and runs them natively, so this comparison is against
hardware, not against another emulator. It exists because the Unicorn/QEMU
reference disagrees with hardware on one point -- which NaN surfaces when both
operands of a double-precision arithmetic instruction are NaNs (QEMU returns
the larger significand; hardware returns the destination's) -- and the
emulator deliberately follows the hardware.

Every other row of the matrix must agree with the native CPU, or this exits 1.

Usage:
    .venv/bin/python tools/check_nan_rule.py [--verbose]
"""

import argparse
import os
import random
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XONE = os.path.join(ROOT, "target", "release", "xone")
BUILD = os.path.join(ROOT, "build")
SOURCE = os.path.join(ROOT, "tests", "c", "nan_rule.c")

FORMS = ["ss", "sd", "ps", "pd"]
OPS = [0x58, 0x59, 0x5C, 0x5E, 0x5D, 0x5F]

# Quiet and signalling NaNs with distinct payloads, signed variants, and plain
# numbers, so the selection rule is exercised in every combination.
VALUES = [
    0x7FF8_0000_0000_0001,  # qNaN, payload 1
    0x7FF8_0000_0000_0002,  # qNaN, payload 2
    0xFFF8_0000_0000_0002,  # -qNaN, payload 2
    0x7FF0_0000_0000_0001,  # sNaN, payload 1
    0x7FF0_0000_0000_0002,  # sNaN, payload 2
    0xFFF0_0000_0000_0002,  # -sNaN, payload 2
    0x3FF0_0000_0000_0000,  # 1.0
    0xBFF0_0000_0000_0000,  # -1.0
    0x7FF0_0000_0000_0000,  # +inf
    0xFFF0_0000_0000_0000,  # -inf
    0x0000_0000_0000_0001,  # smallest denormal
    0x8000_0000_0000_0000,  # -0.0
]


def wide(value):
    """Zero-pad a 64-bit pattern into the 16-byte operand layout."""
    return value.to_bytes(8, "little") + bytes(8)


def compile_oracle(out_dir):
    binary = os.path.join(out_dir, "nan_rule")
    cmd = ["gcc", "-O1", "-o", binary, SOURCE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("gcc failed:\n" + res.stderr)
    return binary


def hardware_results(oracle, cases):
    stdin = "".join(f"{form} {op:02x} {a:016x} {b:016x}\n" for form, op, a, b in cases)
    res = subprocess.run([oracle], input=stdin, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("nan_rule failed:\n" + res.stderr)
    lines = res.stdout.split()
    if len(lines) != len(cases):
        sys.exit(f"nan_rule returned {len(lines)} results for {len(cases)} cases")
    return [int(line, 16) for line in lines]


def emulator_result(form, op, a, b, workdir):
    prefix = {"ss": b"\xf3", "sd": b"\xf2", "ps": b"", "pd": b"\x66"}[form]
    program = (
        b"\x0f\x10\x04\x25\x00\x40\x00\x00"      # movups xmm0, [0x4000]
        b"\x0f\x10\x0c\x25\x10\x40\x00\x00"      # movups xmm1, [0x4010]
        + prefix + bytes([0x0f, op, 0xc1])       # xmm0 = xmm0 <op> xmm1
        + b"\xcc"
    )
    prog_path = os.path.join(workdir, "prog.bin")
    data_path = os.path.join(workdir, "data.bin")
    with open(prog_path, "wb") as fh:
        fh.write(program)
    with open(data_path, "wb") as fh:
        fh.write(wide(a) + wide(b))
    res = subprocess.run(
        [XONE, "run", prog_path, "--state", "--load", "0x4000:" + data_path],
        capture_output=True,
        text=True,
    )
    match = re.search(r"xmm0=0x([0-9a-f]+)", res.stdout)
    if not match:
        sys.exit(f"no state from xone for {form} {op:02x}:\n{res.stdout}{res.stderr}")
    return int(match.group(1), 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(XONE):
        sys.exit(f"{XONE} is missing; run `cargo build --release` first")

    rng = random.Random(0)
    cases = []
    for form in FORMS:
        for op in OPS:
            for a in VALUES:
                for b in VALUES:
                    cases.append((form, op, a, b))
    for _ in range(200):
        form = rng.choice(FORMS)
        op = rng.choice(OPS)
        cases.append((form, op, rng.getrandbits(64), rng.getrandbits(64)))

    with tempfile.TemporaryDirectory(dir=BUILD) as workdir:
        oracle = compile_oracle(workdir)
        reference = hardware_results(oracle, cases)
        failures = 0
        for (form, op, a, b), want in zip(cases, reference):
            got = emulator_result(form, op, a, b, workdir)
            if got != want:
                failures += 1
                print(
                    f"DIFF {form} op={op:02x} a={a:016x} b={b:016x} "
                    f"ours={got:032x} hardware={want:032x}"
                )

    print(
        f"nan rule: {len(cases)} cases against native hardware, "
        f"{failures} mismatched"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
