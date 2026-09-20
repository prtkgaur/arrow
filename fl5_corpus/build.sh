#!/bin/bash
# Builds ./layouts, the one binary that holds every bit-unpacking measurement in
# this directory. Works on x86-64 and aarch64; the only difference is -march.
#
#   ./build.sh                     build ./layouts, then run it with no arguments
#                                  for the index of measurements
#   OPT='-O2 -ftree-vectorize' ./build.sh
#                                  Arrow's own Release level. Quote figures from
#                                  this one and bound them with the -O3 default:
#                                  the grid decoders are inlined from headers and
#                                  so answer to the level given here, while the
#                                  shipped vectorized unpacker lives in
#                                  libarrow.so and does not.
#
# Each measurement is its own translation unit and they are compiled separately,
# so a file's kernels come out the same as they did when that file was a binary of
# its own. Nothing is compiled with link-time optimization, deliberately: it would
# let one measurement's code generation depend on another's.
#
#   ARROW=/path/to/arrow           source checkout (has cpp/src)
#   ARROW_BUILD=/path/to/build     configured+built Arrow (has src/arrow/util/config.h
#                                  and release/libarrow.so)
#   XSIMD=/path/to/xsimd/include   xsimd headers (Arrow vendors them under
#                                  <build>/_deps/xsimd-src/include)
#   OPT='-O2 -ftree-vectorize'     optimization level, default -O3
#   OUT=name                       output binary, default layouts
#   JOBS=n                         compile n files at once, default one per core
#   FRESH=1                        discard kept object files and compile all of them
#   OBJECT=study_bit_width         compile that one file to an object file and
#                                  stop, for reading its code generation with
#                                  objdump
set -euo pipefail
# This script lives at <arrow>/fl5_corpus/, so the checkout is one level up and
# needs no configuring. ARROW_BUILD and XSIMD do: point them at your own build.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARROW=${ARROW:-$(dirname "$HERE")}
ARROW_BUILD=${ARROW_BUILD:?set ARROW_BUILD to a configured+built Arrow dir (has src/arrow/util/config.h)}
# Arrow vendors xsimd under the build tree; fall back to a sibling build if the
# caller did not say.
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

OPT=${OPT:--O3}
# Every .cpp beside this script is either the dispatcher or one measurement, so
# the file list is the directory. A new measurement needs no edit here.
SOURCES=("$HERE"/layouts.cpp "$HERE"/study_*.cpp "$HERE"/control_*.cpp)
OUT=${OUT:-layouts}
CXX=${CXX:-g++}

# One file to an object file and stop. The binary always holds every measurement,
# because the dispatcher names them all; this exists for reading disassembly, not
# for building a subset.
if [ -n "${OBJECT:-}" ]; then
  [ -f "$HERE/$OBJECT.cpp" ] || { echo "build.sh: missing $HERE/$OBJECT.cpp" >&2; exit 1; }
  ( set -x
    $CXX -std=c++20 $OPT $ARCH_FLAGS -DNDEBUG -c \
      -I"$ARROW/cpp/src" -I"$ARROW_BUILD/src" -I"$ARROW/cpp/build-support" -I"$XSIMD" \
      "$HERE/$OBJECT.cpp" -o "$HERE/$OBJECT.o" )
  echo "build.sh: built $HERE/$OBJECT.o"
  exit 0
fi

# Object files are kept, in a directory named after the flags that produced them,
# so editing one measurement or the dispatcher does not recompile the other ten.
# An object is reused only if it is newer than its own source AND newer than every
# header it could have read -- the ones in this directory and the ones in Arrow's
# util tree. Set FRESH=1 to discard them.
OBJDIR="$HERE/.objects/$(echo "$OPT $ARCH_FLAGS" | tr -c 'A-Za-z0-9._' '_')"
[ -n "${FRESH:-}" ] && rm -rf "$OBJDIR"
mkdir -p "$OBJDIR"

compile_one() {
  local src=$1 obj=$2
  echo "  compiling $(basename "$src")"
  if ! $CXX -std=c++20 $OPT $ARCH_FLAGS -DNDEBUG -c \
         -I"$ARROW/cpp/src" -I"$ARROW_BUILD/src" -I"$ARROW/cpp/build-support" \
         -I"$XSIMD" "$src" -o "$obj" 2> "$obj.log"; then
    cat "$obj.log" >&2
    echo "$src" >> "$OBJDIR/.failed"
  fi
}

is_stale() {
  local src=$1 obj=$2
  [ -f "$obj" ] || return 0
  [ "$src" -nt "$obj" ] && return 0
  # Any header newer than the object, anywhere it includes from.
  local newer
  newer=$(find "$HERE" "$ARROW/cpp/src/arrow/util" -name '*.h' -newer "$obj" -print -quit)
  [ -n "$newer" ]
}

JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}
rm -f "$OBJDIR/.failed"
OBJS=()
for src in "${SOURCES[@]}"; do
  obj="$OBJDIR/$(basename "${src%.cpp}").o"
  OBJS+=("$obj")
  if is_stale "$src" "$obj"; then
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
    compile_one "$src" "$obj" &
  fi
done
wait
if [ -f "$OBJDIR/.failed" ]; then
  echo "build.sh: failed to compile:" >&2
  cat "$OBJDIR/.failed" >&2
  exit 1
fi

( set -x
  $CXX -std=c++20 $OPT "${OBJS[@]}" -o "$HERE/$OUT" \
    -L"$LIBDIR" -larrow -Wl,-rpath,"$LIBDIR" )
echo "build.sh: built $HERE/$OUT -- run it with no arguments for the index"
