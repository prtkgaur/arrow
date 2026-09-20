# layouts — Parquet's continuous bit packing against the FastLanes grid

One binary. One measurement per subcommand. `./layouts` with no arguments prints
an index of them, built from the table at the top of `layouts.cpp`. This file says
the same things at more length.

Parquet packs a block of values as one continuous LSB-first bit stream. FastLanes
packs the same block as a grid across 32 lanes. The two decode at different
speeds, and the ratio between them is not a single number: it moves with the
packed bit width, with the width of the element the reader writes out, with where
that output lands in the memory hierarchy, and with the column's own values. There
is one measurement per axis so that a figure can be quoted with the three things
that pin it down — packed width, output element width, and where the output went.

Portable to x86-64 and aarch64; `build.sh` picks `-march` from `uname -m`.

## Build

Needs a configured and built Arrow in the same checkout: the harness links
`libarrow` and includes headers from it, because the baseline it measures against
is Arrow's shipped vectorized unpacker rather than a copy of it.

```bash
git clone -b pgaur_interleavedPlusFastLanesDelta https://github.com/prtkgaur/arrow.git
cd arrow/cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DARROW_BUILD_TESTS=OFF \
         -DARROW_BUILD_BENCHMARKS=OFF -DARROW_PARQUET=OFF \
         -DARROW_DEPENDENCY_SOURCE=BUNDLED
cmake --build . -j

cd ../../fl5_corpus
ARROW_BUILD=$(pwd)/../cpp/build ./build.sh
```

`OPT` selects the optimization level and defaults to `-O3`. Arrow's own Release
level is `-O2 -ftree-vectorize`, and the two differ here: the grid decoders are
inlined from headers and answer to whatever level is given, while the shipped
vectorized unpacker lives inside `libarrow.so` and does not. Figures quoted from
one level should be bounded by the other.

Each measurement is a separate translation unit and is compiled separately, with
no link-time optimization, so a file's kernels come out the same as they did when
that file was a binary of its own. Object files are kept under `.objects/`, in a
directory named after the flags that produced them, so editing one measurement
recompiles one file; `JOBS=n` sets how many compile at once and `FRESH=1` throws
them away. An object is reused only when it is newer than its own source and newer
than every header it could have read, including Arrow's.

## Run

```bash
taskset -c 2 ./layouts width > width.txt 2> width.err
echo "exit=$?"
```

Pin it to one core, keep the machine otherwise idle, and give the whole-column
measurements a few minutes each. Every measurement checks its decoders bit-exact
against the generated values before timing anything and exits non-zero if a check
fails, so hand back the exit status along with the output.

## What each measurement varies

### One variable at a time

| name | varies | holds fixed |
|---|---|---|
| `width` | the packed bit width, 1 to 32 | 32-bit output, working set inside L2, frame applied |
| `output` | the output element width, against the packed width | one block per unit of work, L1-resident, no frame |
| `memory` | where the decoded output goes, 16 KiB to past last-level cache | three fixed widths, no frame |

`width` exists because a ratio quoted without its width is underspecified. A
single width also carries real noise — the timing control, which runs the
identical kernel over identical bytes and owes 1.00×, reads as far off as 1.09× —
so quote the median or the band, never one row.

`output` exists because a 128-bit register holds 4 ints or 16 bytes, so the same
kernel retires four times the values per store when the output element is a byte.
It applies no frame of reference, which is why its ratios are wider than the
whole-column ones: the frame add is a near-common term that pulls every ratio
toward 1.00.

`memory` runs two sweeps that disagree, on purpose. One grows a single
destination until it leaves cache, which is the shape a reader that materialises a
whole column has. The other keeps one block-sized destination and reuses it,
which is the shape a reader that consumes a vector at a time has. No figure from
either is quotable without naming which one it is.

### Whole columns, at reader scale

| name | varies | holds fixed |
|---|---|---|
| `corpus` | the column, over 43 generated integer columns, and the working-set size | 32-bit output, page-scale decode call, frame applied |
| `corpus-narrow` | the output element, chosen per column from that column's range | the same 43 columns, the same call, frame applied |
| `delta` | the layout with differences taken instead of residuals | the same 43 columns and working-set sizes |

`corpus` is the reader-shaped measurement: real column shapes, the output element
the C++ reader materialises today, and six working-set sizes that hold the decode
call at page scale while growing the stream around it. A Parquet data page
defaults to about 1 MiB of encoded bytes, so a 32-MiB decode call occurs in no
reader; what genuinely leaves cache is the stream, and that is what the six points
grow. It writes a CSV beside its table, and it prints a validity table at the end
that decides per working-set size whether the timing control tied. Read that table
before quoting a point.

`corpus-narrow` joins the two questions: real columns, output sized to what an
`INT(8)` or `INT(16)` annotation would carry. The choice is made from the column's
values and not from its residuals, because the frame comes back before the value
is stored. Most of the corpus does not qualify, which is a result rather than a
gap.

`delta` is the one place where the decoders do not move the same number of bytes:
lane-distance differences are wider than adjacent ones. Throughput is reported per
value produced, and bits per value is reported beside it. Reading the speed
columns without the size columns credits a layout for a byte count.

### Controls

These decide whether a figure above is measuring what it claims.

| name | question |
|---|---|
| `call-shape` | Does calling the shipped unpacker once per block, as a page decoder must, handicap it? |
| `counters` | Hardware counters behind the store claim in `memory`. Raw ARMv8 event numbers, so aarch64 only. |
| `output-address` | How much does the output buffer's address modulo 4096 move a figure? |
| `bytes-moved` | What does each layout read and write per value, so throughput can be checked against it? |
| `lane-order` | What does returning values in file order rather than lane order cost? |
| `single-width` | The smallest thing that reproduces one cell, for checking a suspicious row by hand. |

`output-address` is the reason every measurement writes into one process-wide
aligned arena: left to a vector each, the output address moves per decoder and per
width, and that is a larger effect than the layout difference being measured.

## Two things about the output to know before comparing runs

- The whole-column measurements print which transposing decoder was compiled in.
  A kernel that unpacks and permutes in registers exists for AVX2 and for NEON;
  on a target with neither, the header says `UNFUSED fallback` and that decoder
  unpacks into a scratch grid and transposes out of it, which is a lower bound
  rather than the layout's cost. Recorded runs from before the NEON kernel existed
  carry the fallback header, so that column is not comparable across the two. The
  grid decoders that return file order directly are portable and carry no such
  caveat.
- `corpus` also reports a store-only reference that does no unpacking. It is not a
  ceiling: it is regularly slower than the decoders writing the same bytes, and it
  moves 2.07× between gcc and clang on fixed hardware, so it bounds its own code
  generation and nothing else.

## Adding a measurement

Write `study_<axis>.cpp` for a new axis or `control_<name>.cpp` for a new check,
with everything in an unnamed namespace and a single entry point outside it.
Declare that entry point in `layouts.cpp` and add one row to the table there.
`build.sh` compiles every `study_*.cpp` and `control_*.cpp` beside it, so it needs
no edit.

## Checked-in results

`fl5_corpus_x86.{txt,csv}` was recorded on x86 by an earlier version of this
harness. Its three points are labelled L1/L2/DRAM by output size alone and it
carries no store-only reference, so its rows do not correspond to the six points
`corpus` prints now. Read it for the x86 transposing kernel and for the ratios
within a row, not as a working-set comparison against the current points. The
other `.txt` files beside it are named for the run that produced them and record
the level and machine in their own headers.
