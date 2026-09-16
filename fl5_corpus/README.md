# FOR layout and output-footprint microbenchmarks

This is a synthetic, single-threaded kernel experiment. The runner targets Linux. It does not load records
from ClickBench/TPC/NYC datasets, measure production Parquet pages, exceptions,
delta reconstruction, decompression, or a downstream consumer. Dataset names
identify generators inspired by those distributions. Compression ratios describe
those generated values only.

## Build

From any directory:

```bash
ARROW_BUILD=/path/to/configured/arrow-build \
XSIMD=/path/to/xsimd/include \
OUT_DIR=/path/to/new/binaries \
/path/to/arrow/fl5_corpus/build.sh
```

The Arrow build must supply `src/arrow/util/config.h` and `release/libarrow.so`
from compatible sources. `XSIMD` defaults to `$ARROW_BUILD/_deps/xsimd-src/include`;
set it explicitly when dependencies were reused from another build.
`OUT_DIR` defaults to this directory. GCC/Clang and C++20 are required.
On x86 the default flags are `-march=haswell -mprefer-vector-width=256`; on ARM,
`-march=armv8-a+simd`. `ARCH_FLAGS` can override them. Build flags, source hashes,
binary hashes and the Arrow CMake cache are recorded in `build-info.json`.

The two outputs are `fl5_corpus` and `seq_granularity`. The latter measures
per-block dispatch versus a whole-buffer call at fixed widths. Its read bounds
now match the corpus harness. Do not divide corpus results by its ratios: it is
a diagnostic with different data and framing, not a universal correction factor.

## Verify, then time

```bash
ARROW_USER_SIMD_LEVEL=AVX2 taskset -c 2 /path/to/binaries/fl5_corpus all --verify-only
ARROW_USER_SIMD_LEVEL=AVX2 taskset -c 2 /path/to/binaries/seq_granularity all --verify-only

python3 fl5_corpus/run.py /path/to/binaries/fl5_corpus results-clientip \
  --dataset ClientIP --core 2 --simd AVX2
python3 fl5_corpus/gen_tables.py results-clientip/results.csv results-clientip/report.html
```

Choose a CPU in your allowed affinity mask. Keep it and its SMT sibling quiet;
use consistent power settings and record them. The runner records the command,
source revision/diff/hashes, binary and linked-library hashes, build information,
CPU/governor, elapsed time and exit code. It refuses to overwrite an existing
output directory. `--simd NONE` disables optional SVE on ARM while retaining its mandatory NEON baseline; the fused FL transpose is
currently AVX2-only, and the ARM fallback must be labelled as such.

The binary also accepts `[dataset-filter|all] [output.csv]`, followed by
`--repetitions=N` (default 7, minimum 3), `--bytes-per-run=N` (default 1 GiB),
`--seed=N` (default 7), `--min-time-ms=N` (default 100), `--exact`, and
`--verify-only`. Binary filters are substrings unless `--exact` is supplied;
the Python runner selects exact names to avoid also matching `NearSortedUnixTime`. Unmatched
filters and output-file errors fail. Verify-only validates all selected cases
without timings; its CSV is header-only and cannot be reported as a benchmark.

Calibration first selects a fixed iteration count per arm that reaches the
minimum duration and byte budget. Each repetition shuffles arm order with the recorded seed. Every arm gets the
same source/destination addresses, a refill outside timing, and one warm pass.
The raw/grid controls use the same compiled callable with different payloads.
A compiler memory barrier protects every timed iteration. The table uses median
throughput and retains individual measurements in `output.csv.raw.csv`, including
order, duration and iteration count. Runs traverse at least the full source and
destination rings. Shorter byte budgets are useful for smoke tests, not claims
about small differences. Repeat processes with a different seed to check stability.

Exit 1 indicates an input, I/O or correctness failure. Exit 2 indicates a noisy
or mismatched-control run: any arm CV exceeds 5%, or `fl_unpk/intlv` lies outside
`[1/1.05, 1.05]` for any column/point. Results are retained for diagnosis. This
does not identify the cause: noise, code generation, and memory placement can
all matter. Do not remove failing columns just to obtain a passing geomean.

## What is measured

| Point | Decoded bytes per call | Rotating source payload | Rotating decoded output |
|---|---:|---:|---:|
| page16k | 16 KiB | one payload | one slice |
| page256k | 256 KiB | one payload | one slice |
| page1m | 1 MiB | one payload | one slice |
| scan4m | 1 MiB | at least 4 MiB per decoder | one slice |
| scan48m | 1 MiB | at least 48 MiB per decoder | one slice |
| batch48m | 1 MiB | one payload | 48 MiB |

These names are decoded sizes, not encoded page-size guarantees. A rotating
48 MiB ring can fit in a large LLC; consult the local cache hierarchy. This does
not flush caches or measure multi-core bandwidth saturation. Source footprints
count allocated payload bytes (including the sequential arm's 64 readable pad
bytes), not stride gaps or measured hardware traffic. The CSV separately records
sequential/grid payload footprints and the address spans including padding.
Source rings contain distinct copies of the same generated page, not different
page contents. Per-block sequential metadata is small and reused, unlike inline grid headers;
that framing difference is part of this implementation comparison.

`seq_scal` uses generated scalar kernels, `seq_simd` calls Arrow's runtime
unpacker per 1024 values, and `intlv` uses the standalone grid in file order.
`fl_unpk` returns permuted values; `fl_tpos` restores file order. All five are
verified, as is the write-only `pure_st` pattern, at every source/destination
slot with an output overrun canary. `pure_st` omits the packed-input stream and
still reads minima and computes values; treat it as a reference, not a proven
hardware ceiling. Ratios include dispatch, framing and layout implementation.

The production PFOR comparison is separate: see
`cpp/src/parquet/PFOR_LAYOUT_RERUN.md`. Never divide its production arms by these
standalone arms and call the result a layout-only effect.

The checked-in `fl5_corpus_x86.*` and `seq_granularity.txt` are historical results
from an older harness/schema. Their cache labels and fastest-of-five timings do
not describe this version. The report generator requires an explicit current CSV and its raw sidecar, verifies the timing
arithmetic/medians/CVs, and rejects incomplete/duplicate/invalid rows; it does not read those files or
insert architectural conclusions into the report.
