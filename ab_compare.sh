#!/usr/bin/env bash
# Alternate variants; retain all repetitions and logs. Only compare within arm groups.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./bench_arms.sh
BIN_DIR=${BIN_DIR:-$PWD/build-width-matrix/artifacts/bin}
OUT=${OUT:-$PWD/ab-results-$(date -u +%Y%m%dT%H%M%SZ)}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-7}
CORE=${CORE:-2}
MIN_TIME=${MIN_TIME:-0.2s}
ARM_SETS=${ARM_SETS:-'ARMS_LAYOUT ARMS_ORDER'}
SETS=()
for name in $ARM_SETS; do
  case $name in ARMS_LAYOUT|ARMS_ORDER|ARMS_DEST|ARMS_SHIPPED) ;; *) echo "unknown arm set: $name" >&2; exit 1;; esac
  SETS+=("${!name}")
done
FILTER=$(bench_filter "${SETS[@]}")
[[ $# -ge 1 ]] || { echo "usage: $0 binary:SIMD [binary:SIMD ...]" >&2; exit 1; }
mkdir "$OUT"
specs=("$@")
for ((round=0; round<ROUNDS; round++)); do
  for ((j=0; j<${#specs[@]}; j++)); do
    # Rotate process order as well as randomizing the benchmarks within each.
    spec=${specs[$(((j+round)%${#specs[@]}))]}
    IFS=: read -r b cap <<< "$spec"
    case $cap in AVX2|SSE4_2) ;; *) echo "explicit AVX2 or SSE4_2 cap required" >&2; exit 1;; esac
    binary=$(realpath "$BIN_DIR/$b")
    prefix="$OUT/${b//\//_}__${cap}__$round"
    sha256sum "$binary" > "$prefix.binary.sha256"
    ldd "$binary" > "$prefix.ldd.txt"
    cache="$(dirname "$binary")/../CMakeCache.txt"
    [[ -f $cache ]] || { echo "binary must retain its original build tree" >&2; exit 1; }
    grep -qx "ARROW_SIMD_LEVEL:STRING=$cap" "$cache" || { echo "build/runtime width mismatch" >&2; exit 1; }
    cp "$cache" "$prefix.CMakeCache.txt"
    python3 - "$prefix.ldd.txt" "$prefix.libraries.sha256" <<'PYLIB'
import hashlib, pathlib, sys
paths = {p for p in pathlib.Path(sys.argv[1]).read_text().split() if p.startswith('/') and pathlib.Path(p).is_file()}
with open(sys.argv[2], 'w') as out:
    for path in sorted(paths):
        with open(path, 'rb') as f:
            digest = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 20), b''):
                digest.update(chunk)
            out.write(digest.hexdigest() + '  ' + path + '\n')
PYLIB
    ARROW_USER_SIMD_LEVEL=$cap "$binary" --benchmark_filter="$FILTER" --benchmark_list_tests=true > "$prefix.cases.txt"
    for name in $ARM_SETS; do
      case $name in
        ARMS_LAYOUT) required='PforPlainSeqDecode PforPlainInterleavedDecode' ;;
        ARMS_ORDER) required='InterleavedPforDecode InterleavedPforFlOrderRawDecode InterleavedPforFlOrderDecode' ;;
        ARMS_DEST) required='PforWholeSeqDecode PforWholeInterleavedDecode PforReuseSeqDecode PforReuseInterleavedDecode' ;;
        ARMS_SHIPPED) required='PforDecode' ;;
      esac
      for arm in $required; do
        grep -q "^BM_$arm/" "$prefix.cases.txt" || { echo "missing requested arm: $arm" >&2; exit 1; }
      done
    done
    ARROW_USER_SIMD_LEVEL=$cap taskset -c "$CORE" "$binary" \
      --benchmark_filter="$FILTER" --benchmark_repetitions="$REPS" \
      --benchmark_min_time="$MIN_TIME" --benchmark_enable_random_interleaving=true \
      --benchmark_report_aggregates_only=false --benchmark_out="$prefix.json" \
      --benchmark_out_format=json > "$prefix.log" 2>&1
    python3 - "$prefix.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])).get('benchmarks', [])
if not rows or any(r.get('error_occurred') for r in rows):
    sys.exit('empty or failed benchmark run')
PY
  done
done
echo "Results and logs: $OUT"
