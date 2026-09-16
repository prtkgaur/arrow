#!/bin/bash
# Builds fl5_corpus (the 5-arm corpus harness) and seq_granularity (the per-block
# call-overhead diagnostic). Works on x86-64 and
# aarch64; the only difference is -march.
#
#   ARROW=/path/to/arrow           source checkout (has cpp/src)
#   ARROW_BUILD=/path/to/build     configured+built Arrow (has src/arrow/util/config.h
#                                  and release/libarrow.so)
#   XSIMD=/path/to/xsimd/include   xsimd headers (Arrow vendors them under
#                                  <build>/_deps/xsimd-src/include)
set -euo pipefail
# This script lives at <arrow>/fl5_corpus/, so the checkout is one level up and
# needs no configuring. ARROW_BUILD and XSIMD do: point them at your own build.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARROW=${ARROW:-$(dirname "$HERE")}
ARROW_BUILD=${ARROW_BUILD:?set ARROW_BUILD to a configured+built Arrow dir (has src/arrow/util/config.h)}
# Default dependency location; override XSIMD when reusing another build.
XSIMD=${XSIMD:-$ARROW_BUILD/_deps/xsimd-src/include}
LIBDIR=${LIBDIR:-$ARROW_BUILD/release}

for f in "$ARROW/cpp/src/arrow/util/fastlanes/interleaved_pfor.h" \
         "$ARROW_BUILD/src/arrow/util/config.h" \
         "$XSIMD/xsimd/xsimd.hpp"; do
  [ -f "$f" ] || { echo "build.sh: missing $f" >&2; exit 1; }
done

case "$(uname -m)" in
  x86_64)  ARCH_FLAGS=${ARCH_FLAGS:-"-march=haswell -mprefer-vector-width=256"} ;;
  aarch64) ARCH_FLAGS=${ARCH_FLAGS:-"-march=armv8-a+simd"} ;;
  *)       ARCH_FLAGS=${ARCH_FLAGS:-""} ;;
esac

set -x
mkdir -p "${OUT_DIR:-$HERE}"
for SRC in fl5_corpus seq_granularity; do
  ${CXX:-g++} -std=c++20 -O3 $ARCH_FLAGS -DNDEBUG \
    -I"$ARROW/cpp/src" -I"$ARROW_BUILD/src" -I"$ARROW/cpp/build-support" -I"$XSIMD" \
    "$HERE/$SRC.cpp" -o "${OUT_DIR:-$HERE}/$SRC" \
    -L"$LIBDIR" -larrow -Wl,-rpath,"$LIBDIR"
done

python3 - "$ARROW" "$ARROW_BUILD" "${OUT_DIR:-$HERE}" "$ARCH_FLAGS" "${CXX:-g++}" "$HERE" "$XSIMD" "$LIBDIR" <<'PY'
import hashlib, json, pathlib, subprocess, sys, shlex
source, build, output = map(pathlib.Path, sys.argv[1:4])
def sha(p):
    with p.open('rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
        return digest.hexdigest()
info = dict(source=str(source), arrow_build=str(build), arch_flags=sys.argv[4],
            compiler=subprocess.check_output(shlex.split(sys.argv[5]) + ['--version'], text=True),
            harness=sys.argv[6], xsimd=sys.argv[7], libdir=sys.argv[8],
            commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            flags='-std=c++20 -O3 -DNDEBUG',
            source_hashes={str(p): sha(p) for p in pathlib.Path(sys.argv[6]).glob('*') if p.suffix in ('.cpp', '.h')},
            binary_hashes={name: sha(output/name) for name in ('fl5_corpus', 'seq_granularity')})
cache = build/'CMakeCache.txt'
if cache.exists():
    info['arrow_build_cache'] = cache.read_text()
(output/'build-info.json').write_text(json.dumps(info, indent=2)+'\n')
PY
