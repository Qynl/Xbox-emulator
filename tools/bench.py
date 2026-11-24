#!/usr/bin/env python3
"""Throughput benchmark for the interpreter.

Builds a small hand-assembled loop, runs it to a fixed instruction budget with
the release binary, and reports millions of instructions per second (MIPS).
The loop body is a mixture of the work a real program does -- integer ALU with
flag updates, loads and stores, and packed SSE2 arithmetic -- so the number
tracks the interpreter's hot path rather than one favourable instruction.

Every run is checked: the machine must consume the whole instruction budget and
stop on the step limit, not on a bad opcode or a crash, so a typo in the hand
assembled bytes shows up as a failure instead of a suspiciously fast result.

Usage:
    .venv/bin/python tools/bench.py [--steps N] [--repeat N] [--body mixed]
"""

import argparse
import os
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BODIES = {
    # Integer ALU, a store, a load, packed adds, a multiply and a shuffle.
    "mixed": bytes.fromhex(
        "4801d8"                    # add    rax, rbx
        "4831c2"                    # xor    rdx, rax
        "48898300010000"            # mov    [rbx+0x100], rax
        "488bb300010000"            # mov    rsi, [rbx+0x100]
        "660fd4c1"                  # paddq  xmm0, xmm1
        "4d0fafc1"                  # imul   r8, r9
        "660f70d21b"                # pshufd xmm2, xmm2, 0x1b
        "4983c201"                  # add    r10, 1
        "660f7f8300020000"          # movdqa [rbx+0x200], xmm0
        "660f6f9b00020000"          # movdqa xmm3, [rbx+0x200]
    ),
    # General-purpose only: no vector work at all.
    "int": bytes.fromhex(
        "4801d8"                    # add    rax, rbx
        "4829ca"                    # sub    rdx, rcx
        "4831c2"                    # xor    rdx, rax
        "4d0fafc1"                  # imul   r8, r9
        "48d3e0"                    # shl    rax, cl
        "48898300010000"            # mov    [rbx+0x100], rax
        "488bb300010000"            # mov    rsi, [rbx+0x100]
        "4983c201"                  # add    r10, 1
    ),
    # SSE2 only: the packed integer and move path.
    "sse": bytes.fromhex(
        "660fd4c1"                  # paddq  xmm0, xmm1
        "660ffac1"                  # psubd  xmm0, xmm1
        "660fefd3"                  # pxor   xmm2, xmm3
        "660f70d21b"                # pshufd xmm2, xmm2, 0x1b
        "660fe9c1"                  # psubw  xmm0, xmm1
        "660f7f8300020000"          # movdqa [rbx+0x200], xmm0
        "660f6f9b00020000"          # movdqa xmm3, [rbx+0x200]
    ),
}


def image(body):
    """`mov ecx, -1` ; body ; `dec ecx` ; `jnz body` ; `int3`."""
    loop_start = 5  # after `mov ecx, imm32`
    code = b"\xb9\xff\xff\xff\xff" + body + b"\xff\xc9"
    rel = loop_start - (len(code) + 6)  # relative to the end of the jnz
    code += b"\x0f\x85" + struct.pack("<i", rel)
    return code + b"\xcc"


def find_binary():
    for rel in ("target/release/xone", "target/debug/xone"):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            return path
    sys.exit("build the emulator first: cargo build --release")


def run(binary, code, steps, workdir):
    path = os.path.join(workdir, "bench.bin")
    with open(path, "wb") as handle:
        handle.write(code)
    args = [
        binary,
        "run",
        path,
        "--state",
        "--steps",
        str(steps),
        "--reg",
        "rbx=0x4000",
        "--reg",
        "r9=0x12345678",
    ]
    started = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        sys.exit(f"emulator exited {proc.returncode}: {proc.stderr.strip()}")
    fields = dict(
        token.split("=", 1)
        for line in proc.stdout.splitlines()
        if line.startswith("STATE ")
        for token in line.split()[1:]
    )
    if fields.get("stop") != "step-limit" or int(fields.get("steps", 0)) != steps:
        sys.exit(
            f"benchmark program did not run: stop={fields.get('stop')} "
            f"steps={fields.get('steps')} {proc.stderr.strip()}"
        )
    return steps, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--body", choices=sorted(BODIES), default="mixed")
    parser.add_argument(
        "--binary",
        action="append",
        help="binary to benchmark; repeat it to compare several (runs are "
        "interleaved so drift hits every binary equally)",
    )
    options = parser.parse_args()

    binaries = options.binary or [find_binary()]
    workdir = os.path.join(ROOT, "build")
    os.makedirs(workdir, exist_ok=True)
    code = image(BODIES[options.body])

    samples = {binary: [] for binary in binaries}
    for _ in range(options.repeat):
        for binary in binaries:
            steps, elapsed = run(binary, code, options.steps, workdir)
            samples[binary].append(elapsed)

    for binary in binaries:
        times = sorted(samples[binary])
        median = times[len(times) // 2]
        label = binary if len(binaries) > 1 else options.body
        print(
            f"{label:>28}: {steps} instructions, median {median:.3f}s "
            f"= {steps / median / 1e6:.1f} MIPS (best {steps / times[0] / 1e6:.1f})"
        )


if __name__ == "__main__":
    main()
