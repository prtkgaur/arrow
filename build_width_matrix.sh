#!/usr/bin/env bash
# Separate build directories preserve the shared libraries for every variant.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BUILD_ROOT=${BUILD_ROOT:-$PWD/build-width-matrix}
OUT=${OUT:-$BUILD_ROOT/artifacts}
JOBS=${JOBS:-6}
mkdir -p "$OUT/bin"
MATRIX_POINTS=${MATRIX_POINTS:-'O2_128:SSE4_2:-O2:128 O2_256:AVX2:-O2:256 O3_128:SSE4_2:-O3:128 O3_256:AVX2:-O3:256'}
for spec in $MATRIX_POINTS; do
  IFS=: read -r name level opt width <<< "$spec"
  build="$BUILD_ROOT/$name"
  cmake -S cpp -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DARROW_SIMD_LEVEL="$level" \
    -DCMAKE_CXX_FLAGS_RELEASE="$opt -DNDEBUG -mprefer-vector-width=$width" \
    -DCMAKE_C_FLAGS_RELEASE="$opt -DNDEBUG -mprefer-vector-width=$width" \
    -DARROW_PARQUET=ON -DARROW_BUILD_BENCHMARKS=ON \
    -DARROW_WITH_ZSTD=ON -DARROW_WITH_LZ4=ON -DARROW_BUILD_TESTS=OFF \
    > "$OUT/configure_$name.log" 2>&1
  cmake --build "$build" --target parquet-pfor-comparison-benchmark -j "$JOBS" \
    > "$OUT/build_$name.log" 2>&1
  git rev-parse HEAD > "$OUT/commit_$name.txt"
  git diff HEAD > "$OUT/source_$name.diff"
  binary=$(realpath "$build/release/parquet-pfor-comparison-benchmark")
  ln -sfn "$binary" "$OUT/bin/bench_$name"
  cp "$build/CMakeCache.txt" "$OUT/CMakeCache_$name.txt"
  objdump -dC --no-show-raw-insn "$binary" | awk '
    /^[0-9a-f]+ </ { inside = index($0, "fastlanes::InterleavedPforDecode<") > 0 }
    inside' > "$OUT/kernel_$name.asm"
  test -s "$OUT/kernel_$name.asm" || { echo "missing kernel disassembly: $name" >&2; exit 1; }
  if [[ $width == 256 ]]; then
    grep -q '%ymm' "$OUT/kernel_$name.asm" || { echo "missing YMM: $name" >&2; exit 1; }
  fi
  sha256sum "$binary" "$build"/release/libarrow.so* "$build"/release/libparquet.so* > "$OUT/hashes_$name.txt"
done
echo "Binaries: $OUT/bin (each retains its own build and shared libraries)"
