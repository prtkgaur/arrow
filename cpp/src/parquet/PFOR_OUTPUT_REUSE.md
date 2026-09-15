# WIP: PFOR output-buffer reuse

The layout crossover can depend on whether decoding materializes a complete
output array or reuses a small destination. This experiment holds the compressed
page and block-decoding loop fixed while varying the destination footprint.

| Variant | Packing | Destination for each 1024-value block |
|---|---|---|
| `BM_PforWholeSeqDecode` | Sequential | Next 4 KiB of the output array |
| `BM_PforWholeInterleavedDecode` | Interleaved | Next 4 KiB of the output array |
| `BM_PforReuseSeqDecode` | Sequential | Same 4 KiB buffer |
| `BM_PforReuseInterleavedDecode` | Interleaved | Same 4 KiB buffer |

All four use the production PFOR encoder with delta disabled and call the same
production `DecodeVector` entry point. The planner, exceptions and page format
are retained. Every decoded block is checked against its input before timing.
Both destination policies have a compiler memory barrier after every block.

Page setup and validation are outside timing for all four variants. There is
no timed downstream consumer. These are controlled block-loop measurements,
not timings of the unchanged page-level decoder API or a complete query.

## Reproduce

Build with an AVX2 baseline so the interleaved kernels and sequential runtime
target both use AVX2. An existing AVX2 build can be rebuilt after applying the
change.

```bash
cmake -S cpp -B build-pfor-output-reuse \
  -DCMAKE_BUILD_TYPE=Release -DARROW_SIMD_LEVEL=AVX2 \
  -DARROW_RUNTIME_SIMD_LEVEL=MAX -DARROW_PARQUET=ON \
  -DARROW_BUILD_BENCHMARKS=ON -DARROW_BUILD_TESTS=OFF \
  -DARROW_WITH_ZSTD=ON -DARROW_WITH_LZ4=ON
cmake --build build-pfor-output-reuse \
  --target parquet-pfor-comparison-benchmark -j 6
python3 cpp/src/parquet/pfor_output_reuse.py \
  --benchmark build-pfor-output-reuse/release/parquet-pfor-comparison-benchmark \
  --output pfor-output-reuse.json
```

The default run covers four synthetic columns, five input counts and four arms:
80 cases, five repetitions, 0.2-second minimum per repetition, randomized order.
It pins the runtime SIMD level to AVX2 and, on Linux, selects CPU 2 if allowed.
Use `--cpu` to select another allowed CPU. `--datasets`, `--repetitions` and
`--min-time` adjust the run. The measurements below took about 132 seconds,
excluding compilation; runtime will vary by machine.

Keep the generated JSON, log, `.run.json` and `.summary.json` files. The JSON
retains individual repetitions, compression sizes and active output-buffer
sizes. The log records the CPU/cache information reported by Google Benchmark.
To summarize an existing run:

```bash
python3 cpp/src/parquet/pfor_output_reuse.py --input pfor-output-reuse.json
```

## Initial observation

AMD Ryzen AI 9 HX PRO 470, CPU 2, AVX2 Release build with GCC 16. The observed
cache domain has 48 KiB L1 data, 1 MiB L2 and 16 MiB L3. Frequency scaling was
enabled. Data: `OS`, `TpcdsItemSk`, `SensorDropouts`, `SortedUnixTime`.

Geometric means of per-column median CPU-time ratios; greater than one favors
interleaving or output reuse, respectively:

| Logical decoded volume | Interleaved / seq, whole | Interleaved / seq, reuse | Seq gain from reuse | Interleaved gain from reuse |
|---|---:|---:|---:|---:|
| 16 KiB | 2.628x | 2.626x | 0.995x | 0.994x |
| 400 KiB | 2.319x | 2.458x | 1.010x | 1.070x |
| 1.5 MiB | 1.592x | 2.462x | 1.039x | 1.608x |
| 4 MiB | 1.524x | 2.253x | 1.039x | 1.535x |
| 32 MiB | 0.762x | 1.855x | 1.963x | 4.779x |

All 80 cases passed correctness checks. Compressed sizes matched across the
four variants for every input. Median CPU-time CV was 1.63%; three cases exceeded
5% and none exceeded 10%. This is a four-column pilot on one machine.

The observed large-output reversal disappears with a reusable destination.
This supports testing streaming output separately from full materialization.
It does not establish a universal layout advantage or a specific hardware
bandwidth bottleneck.

## Interpretation

- Logical decoded volume is not the live output footprint: the reused
  destination is always 4 KiB, even when processing 32 MiB of decoded values.
- The compressed input still grows, reaching approximately 5-17 MiB at the
  largest point here. Output reuse does not make the entire workload L1-resident.
- Reported decoded GiB/s includes every block processed. It is not a measurement
  of DRAM traffic or unique output bytes retained.
- A streaming consumer must use a batch before the next block overwrites it.
  Summing, filtering or another downstream operation needs a separate timed
  experiment. Copying batches into a full array restores materialization traffic.
- Full materialization remains relevant when callers need retained values,
  random access, repeated passes or an independently owned output array.
