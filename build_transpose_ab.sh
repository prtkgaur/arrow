#!/usr/bin/env bash
# Build the current transpose implementation in isolated AVX2 O2/O3 trees.
# For a before/after comparison, run in each source checkout with distinct roots.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MATRIX_POINTS='tr_O2_256:AVX2:-O2:256 tr_O3_256:AVX2:-O3:256'
export BUILD_ROOT=${BUILD_ROOT:-$HERE/build-transpose-matrix}
exec bash "$HERE/build_width_matrix.sh"
