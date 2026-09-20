// layouts -- every bit-unpacking layout measurement in this directory, behind one
// binary, one measurement per subcommand.
//
// The question all of them answer is the same: Parquet packs a block of values as
// one continuous LSB-first bit stream, FastLanes packs the same block as a grid
// across 32 lanes, and the two decode at different speeds. The reason there is
// more than one measurement is that the ratio between them is not a single
// number. It moves with the packed bit width, with the width of the element the
// reader writes out, with where that output lands in the memory hierarchy, and
// with the column's own value distribution. Each subcommand below pins all of
// those but one.
//
//   ./layouts                 the index: every measurement, what it varies, what
//                             it holds fixed
//   ./layouts <name>          run one
//   ./layouts <name> [args]   arguments after the name go to that measurement
//
// Pin the process to one core and keep the machine otherwise idle:
//
//   taskset -c 2 ./layouts width > width.txt
//
// Every measurement checks its decoders bit-exact against the generated values
// before it times anything, and returns non-zero if a check fails, so a run that
// exits zero has verified what it reported.
//
// Adding one: write study_<axis>.cpp or control_<name>.cpp with a single entry
// point, declare it below, and add one row to kStudies. Nothing else in the
// directory needs to know about it. Each file keeps its own kernels, its own
// generators and its own tables in an unnamed namespace, so it is compiled
// exactly as it would be on its own and two files may hold the same name for
// different things.

#include <cstdio>
#include <cstring>

// One per file, in the order the index prints them.
int RunBitWidthStudy(int argc, char** argv);
int RunOutputWidthStudy(int argc, char** argv);
int RunMemoryStudy(int argc, char** argv);
int RunCorpusStudy(int argc, char** argv);
int RunCorpusNarrowStudy(int argc, char** argv);
int RunCorpusDeltaStudy(int argc, char** argv);
int RunCallShapeControl(int argc, char** argv);
int RunStoreCounters(int argc, char** argv);
int RunOutputAddressControl(int argc, char** argv);
int RunBytesMovedControl(int argc, char** argv);
int RunLaneOrderControl(int argc, char** argv);
int RunSingleWidthControl(int argc, char** argv);

namespace {

struct Study {
  const char* group;
  const char* name;
  const char* varies;
  const char* fixed;
  const char* note;  // Two lines: what it is for, and what it cannot be read as.
  const char* usage;  // Arguments, or "" for none.
  int (*run)(int argc, char** argv);
};

// Three groups, three kinds of question, and a figure should be quoted with the
// group it came from. A one-variable measurement says how a ratio behaves. A
// whole-column measurement says what a reader would see. A control says whether a
// figure from the other two was measuring what it claimed.
constexpr char kOneVariable[] = "One variable at a time";
constexpr char kWholeColumns[] = "Whole columns, at reader scale";
constexpr char kControls[] = "Controls -- these decide whether a figure above is real";

const Study kStudies[] = {
    {kOneVariable, "width",
     "the packed bit width, every width from 1 to 32",
     "32-bit output, 128 blocks, working set inside L2, frame of reference applied",
     "A layout ratio is a function of the bit width: a narrow width leaves most of\n"
     "    the unpack in shifts, a wide one leaves most of it in loads and stores. A\n"
     "    ratio quoted without its width is underspecified, and this supplies it.\n"
     "    One width carries real noise -- the timing control that owes 1.00x reads as\n"
     "    far off as 1.09x -- so quote the median or the band, never one row.",
     "", RunBitWidthStudy},

    {kOneVariable, "output",
     "the width of the output element, against the packed width",
     "one block per unit of work, working set in L1, no frame, no exceptions",
     "A 128-bit register holds 4 ints or 16 bytes, so the same kernel retires four\n"
     "    times the values per store when the output element is a byte. Writing\n"
     "    narrower than 32 bits is what the INT logical type annotation permits, not\n"
     "    a trick. There is no frame add here, which is why these ratios are wider\n"
     "    than the corpus ones: a frame add is a near-common term that pulls every\n"
     "    ratio toward 1.00.",
     "", RunOutputWidthStudy},

    {kOneVariable, "memory",
     "where the decoded output goes, from 16 KiB up past last-level cache",
     "three fixed widths, the same frameless kernels the output study uses",
     "Two sweeps. The first grows one destination until it leaves cache, which is\n"
     "    the shape a reader that materialises a whole column has. The second keeps\n"
     "    one block-sized destination and reuses it, which is the shape a reader that\n"
     "    consumes a vector at a time has. They do not agree, and the difference is\n"
     "    the decoder's shape rather than the layout, so no figure from either is\n"
     "    quotable without naming which one it is.",
     "", RunMemoryStudy},

    {kWholeColumns, "corpus",
     "the column, over 43 generated integer columns, and the working-set size",
     "32-bit output, one page-scale decode call, frame of reference applied",
     "The reader-shaped measurement: real column shapes, the output element the C++\n"
     "    reader materialises today, and six working-set sizes that hold the decode\n"
     "    call at page scale while growing the stream around it. Writes a CSV beside\n"
     "    the table. Read the validity table it prints at the end before quoting any\n"
     "    point: it decides per working-set size whether the timing control tied, and\n"
     "    names the points that failed.",
     "", RunCorpusStudy},

    {kWholeColumns, "corpus-narrow",
     "the output element, chosen per column from that column's own value range",
     "the same 43 columns, the same page-scale call, frame of reference applied",
     "The two measurements above joined: real columns, output sized to what an\n"
     "    INT(8) or INT(16) annotation would carry. Most of this corpus does not\n"
     "    qualify, which is a result and not a gap -- the choice is made from the\n"
     "    values, not from the residuals, because the frame comes back before the\n"
     "    value is stored.",
     "", RunCorpusNarrowStudy},

    {kWholeColumns, "delta",
     "the layout again, but with differences taken instead of residuals",
     "the same 43 columns and the same six working-set sizes",
     "PFOR-DELTA, where the layouts differ in what they can parallelise: adjacent\n"
     "    differences are one 1024-long dependent chain per block, lane-distance\n"
     "    differences are 32 independent ones. The arms here do NOT move the same\n"
     "    bytes, so throughput is reported per value produced and bits per value is\n"
     "    reported beside it. Reading the speed columns without the size columns will\n"
     "    credit a layout for a byte count.",
     "[dataset] [csv]", RunCorpusDeltaStudy},

    {kControls, "call-shape",
     "how often the shipped vectorized unpacker is called",
     "one width, one buffer, the same bytes through all three",
     "Every measurement here calls the baseline once per 1024-value block, because\n"
     "    with real data each block picks its own width and its own frame. If that\n"
     "    call granularity is itself a cost, the baseline is handicapped and every\n"
     "    ratio above is inflated. This prices it: per block, per whole buffer, and\n"
     "    per block with the width known at compile time.",
     "", RunCallShapeControl},

    {kControls, "counters",
     "nothing -- it counts instead of timing",
     "one width, one working set larger than cache",
     "Hardware counters behind the memory study's store claim: written bytes that\n"
     "    turn into cache line fills, store instructions retired, backend stalls.\n"
     "    The event numbers are raw ARMv8 PMU numbers, so this reports usefully on\n"
     "    aarch64 only; an event the core does not implement prints as a dash rather\n"
     "    than as a zero.",
     "", RunStoreCounters},

    {kControls, "output-address",
     "the output buffer's address modulo 4096",
     "one width, one layout, one working set",
     "Why every measurement here writes into one process-wide aligned arena. Left to\n"
     "    a vector each, the output address moves per arm and per width, and it is a\n"
     "    larger effect than the layout difference being measured.",
     "[values]", RunOutputAddressControl},

    {kControls, "bytes-moved",
     "nothing -- it counts bytes, not time",
     "one block",
     "What each layout reads and writes per value, so a throughput figure can be\n"
     "    checked against the bytes it had to move. A layout that looks faster while\n"
     "    moving more bytes is a different claim from one that moves the same bytes.",
     "", RunBytesMovedControl},

    {kControls, "lane-order",
     "whether the decoder returns values in file order or lane order",
     "one width, one layout, one working set",
     "The permutation priced on its own. A scan or a filter does not care what order\n"
     "    the values arrive in, a positional reader does, and the difference between\n"
     "    those two is not free.",
     "[values]", RunLaneOrderControl},

    {kControls, "single-width",
     "one width at a time, given on the command line",
     "whatever that width implies",
     "The smallest thing that reproduces a single cell, for checking a suspicious row\n"
     "    of the width study by hand without running the whole sweep.",
     "[values]", RunSingleWidthControl},
};

void PrintIndex() {
  printf(
      "layouts -- Parquet's continuous bit packing against the FastLanes grid, one\n"
      "measurement per subcommand.\n\n"
      "Run one with:  taskset -c 2 ./layouts <name> > <name>.txt\n"
      "Pin it to one core and keep the machine idle. Every measurement verifies its\n"
      "decoders bit-exact before timing and exits non-zero if a check fails.\n");
  const char* group = nullptr;
  for (const Study& s : kStudies) {
    if (group == nullptr || strcmp(group, s.group) != 0) {
      group = s.group;
      printf("\n%s\n", group);
    }
    printf("\n  %s%s%s\n", s.name, s.usage[0] ? "  " : "", s.usage);
    printf("    varies   %s\n", s.varies);
    printf("    holds    %s\n", s.fixed);
    printf("    %s\n", s.note);
  }
  printf(
      "\nA figure is quotable with three things attached: the packed width, the output\n"
      "element width, and where the output went. Two of the three are held fixed in\n"
      "every measurement above, and which two is the first line of each entry.\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
    PrintIndex();
    return 0;
  }
  for (const Study& s : kStudies) {
    if (strcmp(argv[1], s.name) == 0) {
      // argv[0] becomes the subcommand name, so each measurement keeps the
      // argument indexing it had when it was a binary of its own.
      return s.run(argc - 1, argv + 1);
    }
  }
  fprintf(stderr, "layouts: no measurement named '%s'. Run ./layouts for the index.\n",
          argv[1]);
  return 2;
}
