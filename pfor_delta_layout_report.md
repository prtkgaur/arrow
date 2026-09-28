# PFOR delta payload layout: PatchedDelta or DBPDelta?

## Question and result

When PFOR selects delta mode for a 1,024-value vector, should that vector store its
deltas using PFOR's frame, bit width, and exceptions (`PFOR-PatchedDelta`), or as an
independent Parquet DELTA_BINARY_PACKED stream (`PFOR-DBPDelta`)? The measurements
favor **PFOR-PatchedDelta**: it is smaller and substantially faster to decompress in
both dataset classes. DBPDelta encodes faster, but does not improve the read path or
the aggregate compression ratio.

## Method

Both arms restart at exactly the same 1,024-value PFOR vector boundaries. DBPDelta
uses the canonical DBP layout inside every vector (128-value blocks, four 32-value
miniblocks for INT32; 256-value blocks and four 64-value miniblocks for INT64),
including a new DBP header and first value per vector. Both include the
same outer PFOR page header and four-byte vector offsets. Delta mode is forced so the
PFOR planner cannot silently substitute raw PFOR. A separate forced-raw PFOR run
classifies a dataset as *delta-beneficial* when PatchedDelta is smaller than raw PFOR;
all others are *non-beneficial*.

The corpus has 54 generated column cases (39 INT32, 15 INT64), including ClickBench,
TPC-DS, TPC-H, taxi, timestamp, counter, random-walk, sawtooth, sentinel, and bimodal
shapes. Each column has 102,400 values. Results are geometric means of five repetitions
from a Release build compiled with `-O3 -DNDEBUG`. Compression ratio is uncompressed
bytes divided by encoded bytes; higher is better. Throughput is uncompressed GB/s.

## Delta-beneficial datasets

| Compression | Cases | PatchedDelta ratio | DBPDelta ratio | PatchedDelta size vs DBP | Size wins |
|---|---:|---:|---:|---:|---:|
| INT32 | 9 | 8.150 | 5.303 | 65.1% | 8/9 |
| INT64 | 9 | 12.447 | 8.055 | 64.7% | 6/9 |
| Combined | 18 | 10.072 | 6.536 | **64.9%** | 14/18 |

| Decompression | Cases | PatchedDelta GB/s | DBPDelta GB/s | PatchedDelta speedup |
|---|---:|---:|---:|---:|
| INT32 | 9 | 7.450 | 2.338 | **3.19x** |
| INT64 | 9 | 12.912 | 6.055 | **2.13x** |
| Combined | 18 | 9.808 | 3.763 | **2.61x** |

## Datasets that do not benefit from delta mode

| Compression | Cases | PatchedDelta ratio | DBPDelta ratio | PatchedDelta size vs DBP | Size wins |
|---|---:|---:|---:|---:|---:|
| INT32 | 30 | 2.999 | 2.927 | 97.6% | 29/30 |
| INT64 | 6 | 2.113 | 2.107 | 99.7% | 5/6 |
| Combined | 36 | 2.829 | 2.771 | **97.9%** | 34/36 |

| Decompression | Cases | PatchedDelta GB/s | DBPDelta GB/s | PatchedDelta speedup |
|---|---:|---:|---:|---:|
| INT32 | 30 | 7.256 | 2.283 | **3.18x** |
| INT64 | 6 | 11.314 | 5.668 | **2.00x** |
| Combined | 36 | 7.814 | 2.657 | **2.94x** |

## Why PatchedDelta decompresses faster

Both implementations call Arrow's same SIMD `internal::unpack`; the difference is
the work around it. PatchedDelta has one bit width for a 1,024-value vector, invokes
the unpacker once, folds its frame bias into that unpack, patches sparse exceptions,
and makes one prefix-sum pass. DBPDelta divides the same vector into 32 INT32 or 16
INT64 miniblocks. For each block/miniblock the decoder parses a min delta and width,
re-enters `BitReader::GetBatch`, and reconstructs each value from the unpacked offset,
min delta, and previous value. Thus it pays more dispatch/state transitions and more
reconstruction work while using the same underlying unpack kernel. A control run of
whole-page DBP versus DBP restarted every 1,024 values retained 95.6% of decode
throughput, so per-vector header parsing explains only about 4.4%; the miniblock and
reconstruction path explains the bulk of the measured gap.

## Recommendation

Keep the current per-vector planner and `PFOR-PatchedDelta` payload. On columns where
delta mode helps, PatchedDelta is 35.1% smaller and 2.61x faster to decompress than
DBPDelta. Even when delta mode should not be selected, PatchedDelta remains 2.1%
smaller and 2.94x faster. DBPDelta's advantage is encode throughput: 1.87x faster on
the beneficial class and 1.60x faster on the non-beneficial class. That trade is not
enough to replace the current layout for a storage encoding optimized for compression
and repeated reads.
