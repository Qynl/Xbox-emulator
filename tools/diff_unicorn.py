#!/usr/bin/env python3
"""Differential validation of the `xone` x86-64 core against Unicorn.

Each fixture is a real assembled program that ends in `int3`. We execute the
same bytes in our interpreter and in Unicorn (a QEMU-derived x86 emulator),
then compare all 16 general-purpose registers, the arithmetic flags and a hash
of the whole low memory window (code + stack + written data).

A fixture only passes if:
  * our interpreter stops on int3 (not on a bad opcode or the step budget),
  * Unicorn reaches the same int3 sentinel (so it did not fault earlier on an
    instruction we decode differently), and
  * registers, flags and memory hash match exactly.
"""

import glob
import os
import re
import subprocess
import sys

try:
    from unicorn import Uc, UcError, UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64
    from unicorn.x86_const import (
        UC_X86_REG_EFLAGS,
        UC_X86_REG_R8,
        UC_X86_REG_R9,
        UC_X86_REG_R10,
        UC_X86_REG_R11,
        UC_X86_REG_R12,
        UC_X86_REG_R13,
        UC_X86_REG_R14,
        UC_X86_REG_R15,
        UC_X86_REG_RAX,
        UC_X86_REG_RBP,
        UC_X86_REG_RBX,
        UC_X86_REG_RCX,
        UC_X86_REG_RDI,
        UC_X86_REG_RDX,
        UC_X86_REG_RIP,
        UC_X86_REG_RSI,
        UC_X86_REG_RSP,
    )
except ImportError:
    sys.exit(
        "unicorn is not installed.\n"
        "  python3 -m venv .venv && .venv/bin/pip install unicorn capstone"
    )

REG_IDS = {
    "rax": UC_X86_REG_RAX,
    "rcx": UC_X86_REG_RCX,
    "rdx": UC_X86_REG_RDX,
    "rbx": UC_X86_REG_RBX,
    "rsp": UC_X86_REG_RSP,
    "rbp": UC_X86_REG_RBP,
    "rsi": UC_X86_REG_RSI,
    "rdi": UC_X86_REG_RDI,
    "r8": UC_X86_REG_R8,
    "r9": UC_X86_REG_R9,
    "r10": UC_X86_REG_R10,
    "r11": UC_X86_REG_R11,
    "r12": UC_X86_REG_R12,
    "r13": UC_X86_REG_R13,
    "r14": UC_X86_REG_R14,
    "r15": UC_X86_REG_R15,
}

# CF|PF|ZF|SF|DF|OF. AF is architecturally undefined after logical ops, so
# comparing it would report false mismatches against QEMU's arbitrary choice.
FLAG_MASK = 0x8C5
MEM_BASE = 0
MEM_LEN = 0x10000
CODE = 0x1000
STACK = 0x8000
STEP_BUDGET = 200_000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fnv1a(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def find_binary():
    for rel in ("target/release/xone", "target/debug/xone"):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            return path
    sys.exit("build the emulator first: cargo build --release")


def run_ours(binary, image_path):
    proc = subprocess.run(
        [binary, "run", image_path, "--state", "--steps", str(STEP_BUDGET)],
        capture_output=True,
        text=True,
    )
    match = re.search(r"^STATE (.+)$", proc.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"no STATE line; stderr: {proc.stderr.strip()}")
    fields = dict(token.split("=", 1) for token in match.group(1).split())
    return {
        "regs": {name: int(fields[name], 16) for name in REG_IDS},
        "rip": int(fields["rip"], 16),
        "flags": int(fields["rflags"], 16) & FLAG_MASK,
        "memhash": int(fields["memhash"], 16),
        "stop": fields["stop"],
        "steps": int(fields["steps"]),
    }


def run_unicorn(image_path):
    image = open(image_path, "rb").read()
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(MEM_BASE, MEM_LEN)
    uc.mem_write(CODE, image)
    for rid in REG_IDS.values():
        uc.reg_write(rid, 0)
    uc.reg_write(UC_X86_REG_RSP, STACK)

    executed = []
    uc.hook_add(UC_HOOK_CODE, lambda u, addr, size, data: executed.append(addr))

    error = None
    try:
        uc.emu_start(CODE, 0xFFFFF000, 0, STEP_BUDGET)
    except UcError as exc:  # int3 surfaces as an exception with no handler
        error = exc

    return {
        "regs": {name: uc.reg_read(rid) for name, rid in REG_IDS.items()},
        "rip": uc.reg_read(UC_X86_REG_RIP),
        "flags": uc.reg_read(UC_X86_REG_EFLAGS) & FLAG_MASK,
        "memhash": fnv1a(bytes(uc.mem_read(MEM_BASE, MEM_LEN))),
        "error": error,
        "executed": executed,
        "image": image,
    }


def compare(name, ours, ref):
    problems = []

    if ours["stop"] != "int3":
        problems.append(f"our core stopped on {ours['stop']!r}, expected int3")

    last = ref["executed"][-1] if ref["executed"] else None
    offset = None if last is None else last - CODE
    reached_sentinel = (
        offset is not None and 0 <= offset < len(ref["image"]) and ref["image"][offset] == 0xCC
    )
    if not reached_sentinel:
        problems.append(
            f"Unicorn stopped away from the int3 sentinel (last pc={last!r}, error={ref['error']!r})"
        )

    for reg, value in ours["regs"].items():
        if ref["regs"][reg] != value:
            problems.append(
                f"{reg}: ours=0x{value:016x} unicorn=0x{ref['regs'][reg]:016x}"
            )

    if ref["flags"] != ours["flags"]:
        problems.append(f"rflags: ours=0x{ours['flags']:04x} unicorn=0x{ref['flags']:04x}")

    if ref["memhash"] != ours["memhash"]:
        problems.append(
            f"memory: ours=0x{ours['memhash']:016x} unicorn=0x{ref['memhash']:016x}"
        )

    return problems


def main():
    binary = find_binary()
    pattern = os.path.join(ROOT, "build", "fixtures", "*.bin")
    fixtures = sorted(glob.glob(pattern))
    if not fixtures:
        sys.exit(f"no fixtures found in {pattern}; run tools/build_fixtures.sh")

    failures = 0
    for path in fixtures:
        name = os.path.basename(path)[:-4]
        try:
            ours = run_ours(binary, path)
            ref = run_unicorn(path)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"FAIL {name}: {exc}")
            failures += 1
            continue

        problems = compare(name, ours, ref)
        if problems:
            failures += 1
            print(f"FAIL {name}  ({ours['steps']} instructions in our core)")
            for problem in problems:
                print(f"       {problem}")
        else:
            print(
                f"ok   {name:<10} {ours['steps']:>4} instructions, "
                f"registers/flags/memory identical to Unicorn"
            )

    print()
    if failures:
        print(f"{failures}/{len(fixtures)} fixtures diverged from Unicorn")
        return 1
    print(f"all {len(fixtures)} fixtures match Unicorn exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
