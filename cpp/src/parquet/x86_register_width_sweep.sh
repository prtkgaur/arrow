#!/usr/bin/env bash
# Run one separately compiled build per width. AVX2 is the primary comparison.
# See PFOR_LAYOUT_RERUN.md. DRY_RUN=1 validates registrations without timing.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
[[ $# -gt 0 ]] || { echo "usage: $0 AVX2=/path/to/binary [SSE4_2=/path/to/binary]" >&2; exit 1; }
OUTDIR=${OUTDIR:-$PWD/pfor-sweep-$(date -u +%Y%m%dT%H%M%SZ)}
REPS=${REPS:-7}
MIN_TIME=${MIN_TIME:-0.2s}
CORE=${CORE:-2}
# Keep production layout and standalone ordering groups; no delta comparisons.
FILTER=${FILTER:-'BM_(PforPlainSeqDecode|PforPlainInterleavedDecode|InterleavedPforDecode|InterleavedPforFlOrderRawDecode|InterleavedPforFlOrderDecode)/'}
mkdir "$OUTDIR"
lscpu > "$OUTDIR/machine.txt"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >> "$OUTDIR/machine.txt"
git -C "$REPO" rev-parse HEAD > "$OUTDIR/commit.txt"
git -C "$REPO" diff HEAD > "$OUTDIR/source.diff"
result_files=()
declare -A seen=()
for pair in "$@"; do
  [[ $pair == *=* ]] || { echo "expected LEVEL=PATH" >&2; exit 1; }
  level=${pair%%=*}
  binary=$(realpath "${pair#*=}")
  case $level in AVX2) flag=avx2 ;; SSE4_2) flag=sse4_2 ;; *) echo "use AVX2 or SSE4_2; 512 is asymmetric" >&2; exit 1;; esac
  [[ -z ${seen[$level]:-} ]] || { echo "duplicate build level" >&2; exit 1; }
  seen[$level]=1
  grep -qw "$flag" /proc/cpuinfo || { echo "CPU lacks $flag" >&2; exit 1; }
  cache="$(dirname "$binary")/../CMakeCache.txt"
  [[ -f $cache ]] || { echo "pass the binary in its build/release directory" >&2; exit 1; }
  grep -qx "ARROW_SIMD_LEVEL:STRING=$level" "$cache" || { echo "build baseline does not match $level" >&2; exit 1; }
  cp "$cache" "$OUTDIR/CMakeCache_$level.txt"
  python3 - "$binary" "$OUTDIR/provenance_$level.json" "$level" "$CORE" <<'PY'
import hashlib, json, pathlib, subprocess, sys
binary, output, level, core = sys.argv[1:]
linked = subprocess.check_output(['ldd', binary], text=True)
paths = {binary}
for token in linked.split():
    if token.startswith('/') and pathlib.Path(token).is_file():
        paths.add(token)
hashes = {}
for path in paths:
    with open(path, 'rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
        hashes[path] = digest.hexdigest()
pathlib.Path(output).write_text(json.dumps(dict(binary=binary, ldd=linked, hashes=hashes, simd=level, core=int(core)), indent=2)+'\n')
PY
  objdump -dC --no-show-raw-insn "$binary" | awk '
    /^[0-9a-f]+ </ { inside = index($0, "UnpackBlockFlToFileOrder<") > 0 || index($0, "fastlanes::InterleavedPforDecode<") > 0 }
    inside' > "$OUTDIR/kernels_$level.asm"
  if [[ $level == AVX2 ]]; then
    grep -q 'UnpackBlockFlToFileOrder<' "$OUTDIR/kernels_$level.asm" || { echo "fused kernel missing" >&2; exit 1; }
    grep -q '%ymm' "$OUTDIR/kernels_$level.asm" || { echo "YMM instructions missing" >&2; exit 1; }
  fi
  ARROW_USER_SIMD_LEVEL=$level "$binary" --benchmark_filter="$FILTER" \
    --benchmark_list_tests=true > "$OUTDIR/cases_$level.txt"
  python3 - "$HERE" "$OUTDIR/cases_$level.txt" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from pfor_layout_tables import ARMS, SIZES, NAME
names = pathlib.Path(sys.argv[2]).read_text().splitlines()
actual = set()
for name in names:
    m = NAME.match(name)
    if not m or m[1] not in ARMS:
        sys.exit(f'unexpected benchmark: {name}')
    actual.add((m[1], m[2], int(m[3])))
expected = {(a, d, n) for _, d, _ in actual for a in ARMS for n in SIZES}
if not actual or actual != expected:
    sys.exit('filter/build does not contain all five arms at all five sizes for every selected column')
print(f'validated {len(actual)} registrations')
PY
  [[ ${DRY_RUN:-0} != 1 ]] || continue
  ARROW_USER_SIMD_LEVEL=$level taskset -c "$CORE" "$binary" \
    --benchmark_filter="$FILTER" --benchmark_repetitions="$REPS" \
    --benchmark_min_time="$MIN_TIME" --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=false --benchmark_out_format=json \
    --benchmark_out="$OUTDIR/results_$level.json" > "$OUTDIR/run_$level.log" 2>&1
  result_files+=("$level:$OUTDIR/results_$level.json")
done
[[ ${DRY_RUN:-0} != 1 ]] || { echo "dry run passed: $OUTDIR"; exit 0; }
python3 - "$OUTDIR/combined_results.json" "${result_files[@]}" <<'PY'
import json, pathlib, sys
rows = []
contexts = {}
for pair in sys.argv[2:]:
    level, path = pair.split(':', 1)
    data = json.loads(pathlib.Path(path).read_text())
    if not data.get('benchmarks') or any(b.get('error_occurred') for b in data['benchmarks']):
        sys.exit(f'empty or failed run: {path}')
    contexts[level] = data.get('context', {})
    rows.extend(dict(b, build_simd_level=level) for b in data['benchmarks'])
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(contexts=contexts, benchmarks=rows), indent=2)+'\n')
PY
status=0
python3 "$HERE/pfor_layout_tables.py" "$OUTDIR/combined_results.json" > "$OUTDIR/tables.txt" || status=$?
tar -czf "$OUTDIR.tar.gz" -C "$OUTDIR" .
echo "Results: $OUTDIR.tar.gz; report status=$status (2 means inconclusive measurements)"
exit "$status"
