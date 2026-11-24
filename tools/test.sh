#!/usr/bin/env bash
# Full verification: build, Rust tests, differential validation against Unicorn
# over the fixtures, then randomised instruction-level differential fuzzing.
set -euo pipefail
cd "$(dirname "$0")/.."

CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
command -v "$CARGO" >/dev/null 2>&1 || CARGO=cargo

"$CARGO" build --release
"$CARGO" test

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x .venv/bin/python ]; then
        PY=.venv/bin/python
    else
        PY=python3
    fi
fi

echo
echo "== differential fixtures =="
"$PY" tools/diff_unicorn.py

echo
echo "== SSE floating-point rules against native hardware =="
"$PY" tools/check_nan_rule.py

echo
echo "== shift/rotate flag rules against native hardware =="
"$PY" tools/check_shift_flags.py

echo
echo "== randomised differential fuzzing =="
# shellcheck disable=SC2086
"$PY" tools/fuzz_unicorn.py --iterations "${FUZZ_ITERATIONS:-400}" --seeds ${FUZZ_SEEDS:-1 2 3}
