#!/usr/bin/env bash
# Supply preserved before/after binaries explicitly; never delete old results.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $# -ge 2 ]] || { echo "usage: BIN_DIR=... $0 before:AVX2 after:AVX2 [...]" >&2; exit 1; }
if [[ ${WAIT_QUIET:-0} == 1 ]]; then bash "$HERE/wait_quiet.sh"; fi
exec bash "$HERE/ab_compare.sh" "$@"
