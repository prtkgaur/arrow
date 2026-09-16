# PFOR comparison rerun

There are two independent comparison groups over synthetic generated int32
columns. Both sweep 16 KiB, 400 KiB, 1.5 MiB, 4 MiB and 32 MiB of decoded output.
Those are output sizes, not total working sets or cache-residency guarantees.

| Group | Arms | Valid comparison |
|---|---|---|
| Production layout | `BM_PforPlainSeqDecode`, `BM_PforPlainInterleavedDecode` | Same production PFOR planner/exception policy, delta disabled, different packing layout |
| Standalone ordering | `BM_InterleavedPforDecode`, `BM_InterleavedPforFlOrderRawDecode`, `BM_InterleavedPforFlOrderDecode` | Same standalone min/max FOR policy without exceptions; file-order grid, raw FL order, restored FL order |

Do not divide across these groups. The standalone paths have different framing
and exception policy from production PFOR. `BM_PforDecode` permits delta and is
not the production layout baseline. The raw FL arm also returns a different
ordering; it is a control, not a positional decoder.

## Build once per compile-time baseline

```bash
cmake -S cpp -B build-x86-AVX2 -DCMAKE_BUILD_TYPE=Release \
  -DARROW_SIMD_LEVEL=AVX2 -DARROW_PARQUET=ON -DARROW_BUILD_BENCHMARKS=ON \
  -DARROW_WITH_ZSTD=ON -DARROW_WITH_LZ4=ON -DARROW_BUILD_TESTS=OFF
cmake --build build-x86-AVX2 --target parquet-pfor-comparison-benchmark -j 6
```

AVX2 is the primary x86 experiment. Interleaved kernels are auto-vectorized at
compile time, while the sequential unpacker has runtime dispatch. Setting only
`ARROW_USER_SIMD_LEVEL` cannot widen the interleaved kernels. For an SSE4_2
comparison use a separate build with that baseline and cap runtime dispatch to
SSE4_2. Its restored-FL arm uses the unfused scratch fallback. AVX512 is omitted:
sequential dispatch is capped at AVX2 and the fused transpose is fixed at 256 bits.

`build_width_matrix.sh` optionally builds O2/O3 × SSE4_2/AVX2 in four independent
build directories and preserves their caches, hashes and disassembly. It must
not reuse one build directory: copied executables would otherwise load the last
variant's rebuilt shared libraries. `ab_compare.sh` rotates process order and
randomizes cases, preserving individual repetitions and failure logs.

## Validate registrations before timing

```bash
DRY_RUN=1 ./cpp/src/parquet/x86_register_width_sweep.sh \
  AVX2=build-x86-AVX2/release/parquet-pfor-comparison-benchmark
```

The driver checks the build cache, CPU support, AVX2 fused-kernel presence, YMM
evidence, and all five arms/sizes for every selected column. Pass a binary in
its original build directory so its cache and libraries can be identified.
The disassembly is evidence, not proof of every runtime-dispatch decision.

## Time

```bash
CORE=2 REPS=7 MIN_TIME=0.2s ./cpp/src/parquet/x86_register_width_sweep.sh \
  AVX2=build-x86-AVX2/release/parquet-pfor-comparison-benchmark
```

`FILTER` can select a smaller set of columns, but it must retain the five arms
and five sizes for each column. `OUTDIR` chooses a new output directory. Keep the
machine and the pinned core's SMT sibling quiet; preserve its power settings.
The archive records host/governor, source revision/diff, CMake caches, executable
and linked-library hashes, disassembly, benchmark contexts, raw repetitions,
logs, and reports. It retains artifacts on failed runs.

The report prints production `interleaved/seq` separately from standalone
`raw/grid` and `raw/restored`. It requires a complete matched matrix and CVs,
rejects duplicate/error/non-finite rows, and labels the number of selected
columns rather than assuming all 43 ran. It exits 2 if any control gap or CV is
greater than 5%; these are inconclusive measurements, not an automatic diagnosis
of cache conflicts. Missing or invalid data exit 1. Small CV within a process
does not exclude bias between processes; repeat with a fresh process when needed.

```bash
python3 cpp/src/parquet/pfor_layout_tables.py path/to/combined_results.json
```

This page-level harness materializes and reuses one output array. A decline at
large sizes does not establish a store-bandwidth bottleneck. The separate
`fl5_corpus` harness rotates source and destination independently and includes
a write-only reference; see its README for the boundaries of that experiment.
Neither benchmark times a downstream consumer or establishes end-to-end query
speedups, production compression ratios, or multi-core scan throughput.
