# PFOR delta payload layout: PatchedDelta or DBPDelta?

## What the three variants are

All three variants first turn the original values into differences between adjacent
values. They differ in how those differences are stored and decoded:

- **PFOR-PatchedDelta** is the current PFOR delta layout. Each independently decodable
  1,024-value vector selects one frame and one bit width. Most differences are stored
  as fixed-width offsets from that frame; differences that do not fit are stored as
  exceptions and patched before the prefix sum reconstructs the original values.
- **PFOR-DBPDelta (baseline DBP)** replaces that per-vector PFOR payload with a
  canonical Parquet DELTA_BINARY_PACKED stream. For INT32 it divides the vector into
  eight 128-value blocks and 32 miniblocks; for INT64 it uses four 256-value blocks and
  16 miniblocks. Every block stores a minimum difference and every miniblock can use a
  different bit width. The baseline Arrow decoder unpacks and reconstructs one
  miniblock at a time.
- **PFOR-DBPDelta (optimized DBP)** has exactly the same bytes and compression ratio as
  baseline DBP. Only its decoder changes: it combines adjacent equal-width miniblocks
  into larger unpack calls and uses an SIMD register prefix scan to reconstruct values.

Consequently, this comparison separates two questions: whether PFOR's patched,
single-width layout is preferable to DBP's block/miniblock layout, and how much of the
original speed difference came from DBP decoder implementation rather than the format.

## Result

When PFOR selects delta mode for a 1,024-value vector, it can store differences with
PFOR's frame, one bit width, and exceptions (`PFOR-PatchedDelta`), or put an independent
Parquet DELTA_BINARY_PACKED stream in that vector (`PFOR-DBPDelta`). PatchedDelta is
smaller and remains faster to decompress, but optimizing the existing DBP decoder
closes much of the decode gap: from 2.62x to **1.65x** on delta-beneficial datasets and
from 2.95x to **1.61x** on non-beneficial datasets. Thus the original gap was partly an
implementation gap and partly a layout gap.

## Method

Both layouts restart at the same 1,024-value PFOR vector boundaries and include the
same outer page header and four-byte vector offsets. Delta mode is forced. A forced-raw
PFOR run classifies a case as *delta-beneficial* when PatchedDelta is smaller than raw
PFOR. Baseline DBP is Arrow's decoder before the two optimization commits. Optimized
DBP coalesces adjacent equal-width miniblocks into one unpack call (`c3b6f870d6`) and
reconstructs deltas with a SIMD register prefix scan (`9dc8d52b17`). These are
decode-only changes, so baseline
and optimized DBP have identical bytes and encode throughput.

The corpus has 54 generated cases (39 INT32, 15 INT64), including ClickBench, TPC-DS,
TPC-H, taxi, timestamp, counter, random-walk, sawtooth, sentinel, and bimodal shapes;
each has 102,400 values. Results are geometric means of five repetitions from a
Release build compiled with `-O3 -DNDEBUG`. Ratios are uncompressed/encoded bytes;
throughput is uncompressed GB/s.

## Delta-beneficial datasets

| Compression | Cases | PatchedDelta ratio | Baseline DBP ratio | Optimized DBP ratio | Patched size vs DBP |
|---|---:|---:|---:|---:|---:|
| INT32 | 9 | 8.150 | 5.303 | 5.303 | 65.1% |
| INT64 | 9 | 12.447 | 8.055 | 8.055 | 64.7% |
| Combined | 18 | 10.072 | 6.536 | 6.536 | **64.9%** |

| Decompression | Cases | PatchedDelta GB/s | Baseline DBP GB/s | Optimized DBP GB/s | Patched vs baseline | Patched vs optimized |
|---|---:|---:|---:|---:|---:|---:|
| INT32 | 9 | 7.481 | 2.338 | 4.338 | 3.20x | **1.72x** |
| INT64 | 9 | 12.976 | 6.055 | 8.198 | 2.14x | **1.58x** |
| Combined | 18 | 9.852 | 3.763 | 5.964 | 2.62x | **1.65x** |

## Datasets that do not benefit from delta mode

| Compression | Cases | PatchedDelta ratio | Baseline DBP ratio | Optimized DBP ratio | Patched size vs DBP |
|---|---:|---:|---:|---:|---:|
| INT32 | 30 | 2.999 | 2.927 | 2.927 | 97.6% |
| INT64 | 6 | 2.113 | 2.107 | 2.107 | 99.7% |
| Combined | 36 | 2.829 | 2.771 | 2.771 | **97.9%** |

| Decompression | Cases | PatchedDelta GB/s | Baseline DBP GB/s | Optimized DBP GB/s | Patched vs baseline | Patched vs optimized |
|---|---:|---:|---:|---:|---:|---:|
| INT32 | 30 | 7.297 | 2.283 | 4.399 | 3.20x | **1.66x** |
| INT64 | 6 | 11.280 | 5.668 | 8.009 | 1.99x | **1.41x** |
| Combined | 36 | 7.847 | 2.657 | 4.861 | 2.95x | **1.61x** |

## Layout: where the remaining work comes from

Every bracket below is inside one independently decodable 1,024-value PFOR vector.

```text
PFOR-PatchedDelta
+----------------+-------+----------------------+---------------------------+
| frame,width,   | first | 1,024 packed offsets | sparse exception positions|
| delta flag,... | value | at ONE bit width     | and exception differences |
+----------------+-------+----------------------+---------------------------+
                    one unpack call                 patch before prefix sum

PFOR-DBPDelta (INT32; INT64 has 4 blocks and 16 miniblocks)
+------------+-------+----------------------------------------------------+
| DBP header | first | 8 blocks x [min_delta | 4 widths | 4 miniblocks]   |
+------------+-------+----------------------------------------------------+
                        32 miniblocks; widths may change at each boundary
```

DBP's block metadata can represent four local widths without exceptions, but decoding
must observe those boundaries. Optimized DBP removes avoidable overhead within that
layout; it cannot turn unequal-width miniblocks, or miniblocks in different blocks,
into PatchedDelta's single-width run.

## Decode pseudocode

```text
PatchedDelta(vector):
  read one frame, width, start value, and exception list
  unpack_bias(all 1,024 offsets, width, frame)       // one unpack invocation
  patch sparse exception differences
  values = prefix_sum(differences, start)            // one scalar pass today

BaselineDBP(vector):
  for each block:
    read min_delta and four widths
    for each miniblock:
      GetBatch(one miniblock, its width)              // 32/16 invocations total
      for each delta: value = previous + min_delta + offset

OptimizedDBP(vector):
  for each block:
    read min_delta and four widths
    coalesce adjacent miniblocks having the same width
    for each equal-width run: GetBatch(run, width)    // fewer, data-dependent calls
    SIMD inclusive_scan(offsets + min_delta, carry)  // register prefix scan
```

Both layouts ultimately use Arrow's SIMD unpack kernels. The two DBP commits prove
that implementation mattered: optimized DBP is 1.59x faster than baseline on the
beneficial class and 1.83x faster on the non-beneficial class. PatchedDelta still wins
because its metadata selects one width for the whole vector, its frame bias is fused
into one unpack, and only sparse exceptions intervene before one prefix sum. DBP still
parses per-block minima and widths and executes a data-dependent number of unpack
runs. A separate control found that restarting DBP every 1,024 values retains 95.6%
of whole-page DBP throughput, so repeated vector headers account for only about 4.4%.

## Recommendation

Keep `PFOR-PatchedDelta`. On delta-beneficial data it is 35.1% smaller and 1.65x faster
to decompress than optimized DBPDelta; on non-beneficial data it is 2.1% smaller and
1.61x faster. DBPDelta encodes 1.87x and 1.59x faster respectively, but the optimized
decode results show that neither the baseline 2.6-3.0x result nor a claim that both
delta layouts should perform alike is accurate.
